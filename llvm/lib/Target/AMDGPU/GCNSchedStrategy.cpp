//===-- GCNSchedStrategy.cpp - GCN Scheduler Strategy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This contains a MachineSchedStrategy implementation for maximizing wave
/// occupancy on GCN hardware.
///
/// This pass will apply multiple scheduling stages to the same function.
/// Regions are first recorded in GCNScheduleDAGMILive::schedule. The actual
/// entry point for the scheduling of those regions is
/// GCNScheduleDAGMILive::runSchedStages.

/// Generally, the reason for having multiple scheduling stages is to account
/// for the kernel-wide effect of register usage on occupancy.  Usually, only a
/// few scheduling regions will have register pressure high enough to limit
/// occupancy for the kernel, so constraints can be relaxed to improve ILP in
/// other regions.
///
//===----------------------------------------------------------------------===//

#include "GCNSchedStrategy.h"
#include "AMDGPUIGroupLP.h"
#include "GCNRegPressure.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <limits>
#include <sstream>
#include <string>

#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

static cl::opt<bool> DisableUnclusterHighRP(
    "amdgpu-disable-unclustered-high-rp-reschedule", cl::Hidden,
    cl::desc("Disable unclustered high register pressure "
             "reduction scheduling stage."),
    cl::init(false));

static cl::opt<bool> DisableClusteredLowOccupancy(
    "amdgpu-disable-clustered-low-occupancy-reschedule", cl::Hidden,
    cl::desc("Disable clustered low occupancy "
             "rescheduling for ILP scheduling stage."),
    cl::init(false));

static cl::opt<bool> DisablePreRARemat(
    "amdgpu-disable-pre-ra-remat", cl::Hidden,
    cl::desc("Disable pre-RA rematerialization for ILP scheduling stage."),
    cl::init(false));

static cl::opt<bool> DisablePreRARematRollback(
    "amdgpu-disable-pre-ra-rollback", cl::Hidden,
    cl::desc("Disable rollback in pre-RA rematerialization for ILP scheduling "
             "stage."),
    cl::init(false));

static cl::opt<unsigned> ScheduleMetricBias(
    "amdgpu-schedule-metric-bias", cl::Hidden,
    cl::desc(
        "Sets the bias which adds weight to occupancy vs latency. Set it to "
        "100 to chase the occupancy only."),
    cl::init(10));

static cl::opt<bool>
    RelaxedOcc("amdgpu-schedule-relaxed-occupancy", cl::Hidden,
               cl::desc("Relax occupancy targets for kernels which are memory "
                        "bound (amdgpu-membound-threshold), or "
                        "Wave Limited (amdgpu-limit-wave-threshold)."),
               cl::init(false));

static cl::opt<bool> GCNTrackers(
    "amdgpu-use-amdgpu-trackers", cl::Hidden,
    cl::desc("Use the AMDGPU specific RPTrackers during scheduling"),
    cl::init(false));

const unsigned ScheduleMetrics::ScaleFactor = 100;

GCNSchedStrategy::GCNSchedStrategy(const MachineSchedContext *C)
    : GenericScheduler(C), TargetOccupancy(0), MF(nullptr),
      DownwardTracker(*C->LIS), UpwardTracker(*C->LIS), HasHighPressure(false) {
}

void GCNSchedStrategy::initialize(ScheduleDAGMI *DAG) {
  GenericScheduler::initialize(DAG);

  MF = &DAG->MF;

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();

  SGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::SGPR_32RegClass);
  VGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::VGPR_32RegClass);

  SIMachineFunctionInfo &MFI = *MF->getInfo<SIMachineFunctionInfo>();
  // Set the initial TargetOccupnacy to the maximum occupancy that we can
  // achieve for this function. This effectively sets a lower bound on the
  // 'Critical' register limits in the scheduler.
  // Allow for lower occupancy targets if kernel is wave limited or memory
  // bound, and using the relaxed occupancy feature.
  TargetOccupancy =
      RelaxedOcc ? MFI.getMinAllowedOccupancy() : MFI.getOccupancy();
  SGPRCriticalLimit =
      std::min(ST.getMaxNumSGPRs(TargetOccupancy, true), SGPRExcessLimit);

  if (!KnownExcessRP) {
    VGPRCriticalLimit = std::min(
        ST.getMaxNumVGPRs(TargetOccupancy, MFI.getDynamicVGPRBlockSize()),
        VGPRExcessLimit);
  } else {
    // This is similar to ST.getMaxNumVGPRs(TargetOccupancy) result except
    // returns a reasonably small number for targets with lots of VGPRs, such
    // as GFX10 and GFX11.
    LLVM_DEBUG(dbgs() << "Region is known to spill, use alternative "
                         "VGPRCriticalLimit calculation method.\n");
    unsigned DynamicVGPRBlockSize = MFI.getDynamicVGPRBlockSize();
    unsigned Granule =
        AMDGPU::IsaInfo::getVGPRAllocGranule(&ST, DynamicVGPRBlockSize);
    unsigned Addressable =
        AMDGPU::IsaInfo::getAddressableNumVGPRs(&ST, DynamicVGPRBlockSize);
    unsigned VGPRBudget = alignDown(Addressable / TargetOccupancy, Granule);
    VGPRBudget = std::max(VGPRBudget, Granule);
    VGPRCriticalLimit = std::min(VGPRBudget, VGPRExcessLimit);
  }

  // Subtract error margin and bias from register limits and avoid overflow.
  SGPRCriticalLimit -= std::min(SGPRLimitBias + ErrorMargin, SGPRCriticalLimit);
  VGPRCriticalLimit -= std::min(VGPRLimitBias + ErrorMargin, VGPRCriticalLimit);
  SGPRExcessLimit -= std::min(SGPRLimitBias + ErrorMargin, SGPRExcessLimit);
  VGPRExcessLimit -= std::min(VGPRLimitBias + ErrorMargin, VGPRExcessLimit);

  LLVM_DEBUG(dbgs() << "VGPRCriticalLimit = " << VGPRCriticalLimit
                    << ", VGPRExcessLimit = " << VGPRExcessLimit
                    << ", SGPRCriticalLimit = " << SGPRCriticalLimit
                    << ", SGPRExcessLimit = " << SGPRExcessLimit << "\n\n");
}

/// Checks whether \p SU can use the cached DAG pressure diffs to compute the
/// current register pressure.
///
/// This works for the common case, but it has a few exceptions that have been
/// observed through trial and error:
///   - Explicit physical register operands
///   - Subregister definitions
///
/// In both of those cases, PressureDiff doesn't represent the actual pressure,
/// and querying LiveIntervals through the RegPressureTracker is needed to get
/// an accurate value.
///
/// We should eventually only use PressureDiff for maximum performance, but this
/// already allows 80% of SUs to take the fast path without changing scheduling
/// at all. Further changes would either change scheduling, or require a lot
/// more logic to recover an accurate pressure estimate from the PressureDiffs.
static bool canUsePressureDiffs(const SUnit &SU) {
  if (!SU.isInstr())
    return false;

  // Cannot use pressure diffs for subregister defs or with physregs, it's
  // imprecise in both cases.
  for (const auto &Op : SU.getInstr()->operands()) {
    if (!Op.isReg() || Op.isImplicit())
      continue;
    if (Op.getReg().isPhysical() ||
        (Op.isDef() && Op.getSubReg() != AMDGPU::NoSubRegister))
      return false;
  }
  return true;
}

static void getRegisterPressures(
    bool AtTop, const RegPressureTracker &RPTracker, SUnit *SU,
    std::vector<unsigned> &Pressure, std::vector<unsigned> &MaxPressure,
    GCNDownwardRPTracker &DownwardTracker, GCNUpwardRPTracker &UpwardTracker,
    ScheduleDAGMI *DAG, const SIRegisterInfo *SRI) {
  // getDownwardPressure() and getUpwardPressure() make temporary changes to
  // the tracker, so we need to pass those function a non-const copy.
  RegPressureTracker &TempTracker = const_cast<RegPressureTracker &>(RPTracker);
  if (!GCNTrackers) {
    AtTop
        ? TempTracker.getDownwardPressure(SU->getInstr(), Pressure, MaxPressure)
        : TempTracker.getUpwardPressure(SU->getInstr(), Pressure, MaxPressure);

    return;
  }

  // GCNTrackers
  Pressure.resize(4, 0);
  MachineInstr *MI = SU->getInstr();
  GCNRegPressure NewPressure;
  if (AtTop) {
    GCNDownwardRPTracker TempDownwardTracker(DownwardTracker);
    NewPressure = TempDownwardTracker.bumpDownwardPressure(MI, SRI);
  } else {
    GCNUpwardRPTracker TempUpwardTracker(UpwardTracker);
    TempUpwardTracker.recede(*MI);
    NewPressure = TempUpwardTracker.getPressure();
  }
  Pressure[AMDGPU::RegisterPressureSets::SReg_32] = NewPressure.getSGPRNum();
  Pressure[AMDGPU::RegisterPressureSets::VGPR_32] =
      NewPressure.getArchVGPRNum();
  Pressure[AMDGPU::RegisterPressureSets::AGPR_32] = NewPressure.getAGPRNum();
}

void GCNSchedStrategy::initCandidate(SchedCandidate &Cand, SUnit *SU,
                                     bool AtTop,
                                     const RegPressureTracker &RPTracker,
                                     const SIRegisterInfo *SRI,
                                     unsigned SGPRPressure,
                                     unsigned VGPRPressure, bool IsBottomUp) {
  Cand.SU = SU;
  Cand.AtTop = AtTop;

  if (!DAG->isTrackingPressure())
    return;

  Pressure.clear();
  MaxPressure.clear();

  // We try to use the cached PressureDiffs in the ScheduleDAG whenever
  // possible over querying the RegPressureTracker.
  //
  // RegPressureTracker will make a lot of LIS queries which are very
  // expensive, it is considered a slow function in this context.
  //
  // PressureDiffs are precomputed and cached, and getPressureDiff is just a
  // trivial lookup into an array. It is pretty much free.
  //
  // In EXPENSIVE_CHECKS, we always query RPTracker to verify the results of
  // PressureDiffs.
  if (AtTop || !canUsePressureDiffs(*SU) || GCNTrackers) {
    getRegisterPressures(AtTop, RPTracker, SU, Pressure, MaxPressure,
                         DownwardTracker, UpwardTracker, DAG, SRI);
  } else {
    // Reserve 4 slots.
    Pressure.resize(4, 0);
    Pressure[AMDGPU::RegisterPressureSets::SReg_32] = SGPRPressure;
    Pressure[AMDGPU::RegisterPressureSets::VGPR_32] = VGPRPressure;

    for (const auto &Diff : DAG->getPressureDiff(SU)) {
      if (!Diff.isValid())
        continue;
      // PressureDiffs is always bottom-up so if we're working top-down we need
      // to invert its sign.
      Pressure[Diff.getPSet()] +=
          (IsBottomUp ? Diff.getUnitInc() : -Diff.getUnitInc());
    }

#ifdef EXPENSIVE_CHECKS
    std::vector<unsigned> CheckPressure, CheckMaxPressure;
    getRegisterPressures(AtTop, RPTracker, SU, CheckPressure, CheckMaxPressure,
                         DownwardTracker, UpwardTracker, DAG, SRI);
    if (Pressure[AMDGPU::RegisterPressureSets::SReg_32] !=
            CheckPressure[AMDGPU::RegisterPressureSets::SReg_32] ||
        Pressure[AMDGPU::RegisterPressureSets::VGPR_32] !=
            CheckPressure[AMDGPU::RegisterPressureSets::VGPR_32]) {
      errs() << "Register Pressure is inaccurate when calculated through "
                "PressureDiff\n"
             << "SGPR got " << Pressure[AMDGPU::RegisterPressureSets::SReg_32]
             << ", expected "
             << CheckPressure[AMDGPU::RegisterPressureSets::SReg_32] << "\n"
             << "VGPR got " << Pressure[AMDGPU::RegisterPressureSets::VGPR_32]
             << ", expected "
             << CheckPressure[AMDGPU::RegisterPressureSets::VGPR_32] << "\n";
      report_fatal_error("inaccurate register pressure calculation");
    }
#endif
  }

  unsigned NewSGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
  unsigned NewVGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];

  // If two instructions increase the pressure of different register sets
  // by the same amount, the generic scheduler will prefer to schedule the
  // instruction that increases the set with the least amount of registers,
  // which in our case would be SGPRs.  This is rarely what we want, so
  // when we report excess/critical register pressure, we do it either
  // only for VGPRs or only for SGPRs.

  // FIXME: Better heuristics to determine whether to prefer SGPRs or VGPRs.
  const unsigned MaxVGPRPressureInc = 16;
  bool ShouldTrackVGPRs = VGPRPressure + MaxVGPRPressureInc >= VGPRExcessLimit;
  bool ShouldTrackSGPRs = !ShouldTrackVGPRs && SGPRPressure >= SGPRExcessLimit;

  // FIXME: We have to enter REG-EXCESS before we reach the actual threshold
  // to increase the likelihood we don't go over the limits.  We should improve
  // the analysis to look through dependencies to find the path with the least
  // register pressure.

  // We only need to update the RPDelta for instructions that increase register
  // pressure. Instructions that decrease or keep reg pressure the same will be
  // marked as RegExcess in tryCandidate() when they are compared with
  // instructions that increase the register pressure.
  if (ShouldTrackVGPRs && NewVGPRPressure >= VGPRExcessLimit) {
    HasHighPressure = true;
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
    Cand.RPDelta.Excess.setUnitInc(NewVGPRPressure - VGPRExcessLimit);
  }

  if (ShouldTrackSGPRs && NewSGPRPressure >= SGPRExcessLimit) {
    HasHighPressure = true;
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::SReg_32);
    Cand.RPDelta.Excess.setUnitInc(NewSGPRPressure - SGPRExcessLimit);
  }

  // Register pressure is considered 'CRITICAL' if it is approaching a value
  // that would reduce the wave occupancy for the execution unit.  When
  // register pressure is 'CRITICAL', increasing SGPR and VGPR pressure both
  // has the same cost, so we don't need to prefer one over the other.

  int SGPRDelta = NewSGPRPressure - SGPRCriticalLimit;
  int VGPRDelta = NewVGPRPressure - VGPRCriticalLimit;

  if (SGPRDelta >= 0 || VGPRDelta >= 0) {
    HasHighPressure = true;
    if (SGPRDelta > VGPRDelta) {
      Cand.RPDelta.CriticalMax =
          PressureChange(AMDGPU::RegisterPressureSets::SReg_32);
      Cand.RPDelta.CriticalMax.setUnitInc(SGPRDelta);
    } else {
      Cand.RPDelta.CriticalMax =
          PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
      Cand.RPDelta.CriticalMax.setUnitInc(VGPRDelta);
    }
  }
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeFromQueue()
void GCNSchedStrategy::pickNodeFromQueue(SchedBoundary &Zone,
                                         const CandPolicy &ZonePolicy,
                                         const RegPressureTracker &RPTracker,
                                         SchedCandidate &Cand,
                                         bool IsBottomUp) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = 0;
  unsigned VGPRPressure = 0;
  if (DAG->isTrackingPressure()) {
    if (!GCNTrackers) {
      SGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
      VGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];
    } else {
      GCNRPTracker *T = IsBottomUp
                            ? static_cast<GCNRPTracker *>(&UpwardTracker)
                            : static_cast<GCNRPTracker *>(&DownwardTracker);
      SGPRPressure = T->getPressure().getSGPRNum();
      VGPRPressure = T->getPressure().getArchVGPRNum();
    }
  }
  ReadyQueue &Q = Zone.Available;
  for (SUnit *SU : Q) {

    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, IsBottomUp);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    tryCandidate(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      Cand.setBest(TryCand);
      LLVM_DEBUG(traceCandidate(Cand));
    }
  }
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeBidirectional()
SUnit *GCNSchedStrategy::pickNodeBidirectional(bool &IsTopNode) {
  // Schedule as far as possible in the direction of no choice. This is most
  // efficient, but also provides the best heuristics for CriticalPSets.
  if (SUnit *SU = Bot.pickOnlyChoice()) {
    IsTopNode = false;
    return SU;
  }
  if (SUnit *SU = Top.pickOnlyChoice()) {
    IsTopNode = true;
    return SU;
  }
  // Set the bottom-up policy based on the state of the current bottom zone and
  // the instructions outside the zone, including the top zone.
  CandPolicy BotPolicy;
  setPolicy(BotPolicy, /*IsPostRA=*/false, Bot, &Top);
  // Set the top-down policy based on the state of the current top zone and
  // the instructions outside the zone, including the bottom zone.
  CandPolicy TopPolicy;
  setPolicy(TopPolicy, /*IsPostRA=*/false, Top, &Bot);

  // See if BotCand is still valid (because we previously scheduled from Top).
  LLVM_DEBUG(dbgs() << "Picking from Bot:\n");
  if (!BotCand.isValid() || BotCand.SU->isScheduled ||
      BotCand.Policy != BotPolicy) {
    BotCand.reset(CandPolicy());
    pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), BotCand,
                      /*IsBottomUp=*/true);
    assert(BotCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(BotCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), TCand,
                        /*IsBottomUp=*/true);
      assert(TCand.SU == BotCand.SU &&
             "Last pick result should correspond to re-picking right now");
    }
#endif
  }

  // Check if the top Q has a better candidate.
  LLVM_DEBUG(dbgs() << "Picking from Top:\n");
  if (!TopCand.isValid() || TopCand.SU->isScheduled ||
      TopCand.Policy != TopPolicy) {
    TopCand.reset(CandPolicy());
    pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TopCand,
                      /*IsBottomUp=*/false);
    assert(TopCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(TopCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TCand,
                        /*IsBottomUp=*/false);
      assert(TCand.SU == TopCand.SU &&
             "Last pick result should correspond to re-picking right now");
    }
#endif
  }

  // Pick best from BotCand and TopCand.
  LLVM_DEBUG(dbgs() << "Top Cand: "; traceCandidate(TopCand);
             dbgs() << "Bot Cand: "; traceCandidate(BotCand););
  SchedCandidate Cand = BotCand;
  TopCand.Reason = NoCand;
  tryCandidate(Cand, TopCand, nullptr);
  if (TopCand.Reason != NoCand) {
    Cand.setBest(TopCand);
  }
  LLVM_DEBUG(dbgs() << "Picking: "; traceCandidate(Cand););

  IsTopNode = Cand.AtTop;
  return Cand.SU;
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNode()
SUnit *GCNSchedStrategy::pickNode(bool &IsTopNode) {
  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }
  SUnit *SU;
  do {
    if (RegionPolicy.OnlyTopDown) {
      SU = Top.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        TopCand.reset(NoPolicy);
        pickNodeFromQueue(Top, NoPolicy, DAG->getTopRPTracker(), TopCand,
                          /*IsBottomUp=*/false);
        assert(TopCand.Reason != NoCand && "failed to find a candidate");
        SU = TopCand.SU;
      }
      IsTopNode = true;
    } else if (RegionPolicy.OnlyBottomUp) {
      SU = Bot.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        BotCand.reset(NoPolicy);
        pickNodeFromQueue(Bot, NoPolicy, DAG->getBotRPTracker(), BotCand,
                          /*IsBottomUp=*/true);
        assert(BotCand.Reason != NoCand && "failed to find a candidate");
        SU = BotCand.SU;
      }
      IsTopNode = false;
    } else {
      SU = pickNodeBidirectional(IsTopNode);
    }
  } while (SU->isScheduled);

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  LLVM_DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                    << *SU->getInstr());
  return SU;
}

void GCNSchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  if (GCNTrackers) {
    MachineInstr *MI = SU->getInstr();
    IsTopNode ? (void)DownwardTracker.advance(MI, false)
              : UpwardTracker.recede(*MI);
  }

  return GenericScheduler::schedNode(SU, IsTopNode);
}

GCNSchedStageID GCNSchedStrategy::getCurrentStage() {
  assert(CurrentStage && CurrentStage != SchedStages.end());
  return *CurrentStage;
}

bool GCNSchedStrategy::advanceStage() {
  assert(CurrentStage != SchedStages.end());
  if (!CurrentStage)
    CurrentStage = SchedStages.begin();
  else
    CurrentStage++;

  return CurrentStage != SchedStages.end();
}

bool GCNSchedStrategy::hasNextStage() const {
  assert(CurrentStage);
  return std::next(CurrentStage) != SchedStages.end();
}

GCNSchedStageID GCNSchedStrategy::getNextStage() const {
  assert(CurrentStage && std::next(CurrentStage) != SchedStages.end());
  return *std::next(CurrentStage);
}

GCNMaxOccupancySchedStrategy::GCNMaxOccupancySchedStrategy(
    const MachineSchedContext *C, bool IsLegacyScheduler)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::OccInitialSchedule);
  SchedStages.push_back(GCNSchedStageID::UnclusteredHighRPReschedule);
  SchedStages.push_back(GCNSchedStageID::ClusteredLowOccupancyReschedule);
  SchedStages.push_back(GCNSchedStageID::PreRARematerialize);
  GCNTrackers = GCNTrackers & !IsLegacyScheduler;
}

GCNMaxILPSchedStrategy::GCNMaxILPSchedStrategy(const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::ILPInitialSchedule);
}

bool GCNMaxILPSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                          SchedCandidate &TryCand,
                                          SchedBoundary *Zone) const {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  // Avoid spilling by exceeding the register limit.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;

    // Avoid critical resource consumption and balance the schedule.
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;

    // Unconditionally try to reduce latency.
    if (tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);
  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max pressure of the entire region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
                  Cand, RegMax, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Fall through to original instruction order.
    if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
        (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
      TryCand.Reason = NodeOrder;
      return true;
    }
  }
  return false;
}

GCNMaxMemoryClauseSchedStrategy::GCNMaxMemoryClauseSchedStrategy(
    const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::MemoryClauseInitialSchedule);
}

/// GCNMaxMemoryClauseSchedStrategy tries best to clause memory instructions as
/// much as possible. This is achieved by:
//  1. Prioritize clustered operations before stall latency heuristic.
//  2. Prioritize long-latency-load before stall latency heuristic.
///
/// \param Cand provides the policy and current best candidate.
/// \param TryCand refers to the next SUnit candidate, otherwise uninitialized.
/// \param Zone describes the scheduled zone that we are extending, or nullptr
///             if Cand is from a different zone than TryCand.
/// \return \c true if TryCand is better than Cand (Reason is NOT NoCand)
bool GCNMaxMemoryClauseSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                                   SchedCandidate &TryCand,
                                                   SchedBoundary *Zone) const {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  if (DAG->isTrackingPressure()) {
    // Avoid exceeding the target's limit.
    if (tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                    RegExcess, TRI, DAG->MF))
      return TryCand.Reason != NoCand;

    // Avoid increasing the max critical pressure in the scheduled region.
    if (tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                    TryCand, Cand, RegCritical, TRI, DAG->MF))
      return TryCand.Reason != NoCand;
  }

  // MaxMemoryClause-specific: We prioritize clustered instructions as we would
  // get more benefit from clausing these memory instructions.
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);
  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;

  // We only compare a subset of features when comparing nodes between
  // Top and Bottom boundary. Some properties are simply incomparable, in many
  // other instances we should only override the other boundary if something
  // is a clear good pick on one boundary. Skip heuristics that are more
  // "tie-breaking" in nature.
  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // For loops that are acyclic path limited, aggressively schedule for
    // latency. Within an single cycle, whenever CurrMOps > 0, allow normal
    // heuristics to take precedence.
    if (Rem.IsAcyclicLatencyLimited && !Zone->getCurrMOps() &&
        tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // MaxMemoryClause-specific: Prioritize long latency memory load
    // instructions in top-bottom order to hide more latency. The mayLoad check
    // is used to exclude store-like instructions, which we do not want to
    // scheduler them too early.
    bool TryMayLoad =
        TryCand.SU->isInstr() && TryCand.SU->getInstr()->mayLoad();
    bool CandMayLoad = Cand.SU->isInstr() && Cand.SU->getInstr()->mayLoad();

    if (TryMayLoad || CandMayLoad) {
      bool TryLongLatency =
          TryCand.SU->Latency > 10 * Cand.SU->Latency && TryMayLoad;
      bool CandLongLatency =
          10 * TryCand.SU->Latency < Cand.SU->Latency && CandMayLoad;

      if (tryGreater(Zone->isTop() ? TryLongLatency : CandLongLatency,
                     Zone->isTop() ? CandLongLatency : TryLongLatency, TryCand,
                     Cand, Stall))
        return TryCand.Reason != NoCand;
    }
    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;
  }

  if (SameBoundary) {
    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  // Avoid increasing the max pressure of the entire region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
                  Cand, RegMax, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Avoid critical resource consumption and balance the schedule.
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;

    // Avoid serializing long latency dependence chains.
    // For acyclic path limited loops, latency was already checked above.
    if (!RegionPolicy.DisableLatencyHeuristic && TryCand.Policy.ReduceLatency &&
        !Rem.IsAcyclicLatencyLimited && tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Fall through to original instruction order.
    if (Zone->isTop() == (TryCand.SU->NodeNum < Cand.SU->NodeNum)) {
      assert(TryCand.SU->NodeNum != Cand.SU->NodeNum);
      TryCand.Reason = NodeOrder;
      return true;
    }
  }

  return false;
}

GCNScheduleDAGMILive::GCNScheduleDAGMILive(
    MachineSchedContext *C, std::unique_ptr<MachineSchedStrategy> S)
    : ScheduleDAGMILive(C, std::move(S)), ST(MF.getSubtarget<GCNSubtarget>()),
      MFI(*MF.getInfo<SIMachineFunctionInfo>()),
      StartingOccupancy(MFI.getOccupancy()), MinOccupancy(StartingOccupancy),
      RegionLiveOuts(this, /*IsLiveOut=*/true) {

  // We want regions with a single MI to be scheduled so that we can reason
  // about them correctly during scheduling stages that move MIs between regions
  // (e.g., rematerialization).
  ScheduleSingleMIRegions = true;
  LLVM_DEBUG(dbgs() << "Starting occupancy is " << StartingOccupancy << ".\n");
  if (RelaxedOcc) {
    MinOccupancy = std::min(MFI.getMinAllowedOccupancy(), StartingOccupancy);
    if (MinOccupancy != StartingOccupancy)
      LLVM_DEBUG(dbgs() << "Allowing Occupancy drops to " << MinOccupancy
                        << ".\n");
  }
}

std::unique_ptr<GCNSchedStage>
GCNScheduleDAGMILive::createSchedStage(GCNSchedStageID SchedStageID) {
  switch (SchedStageID) {
  case GCNSchedStageID::OccInitialSchedule:
    return std::make_unique<OccInitialScheduleStage>(SchedStageID, *this);
  case GCNSchedStageID::UnclusteredHighRPReschedule:
    return std::make_unique<UnclusteredHighRPStage>(SchedStageID, *this);
  case GCNSchedStageID::ClusteredLowOccupancyReschedule:
    return std::make_unique<ClusteredLowOccStage>(SchedStageID, *this);
  case GCNSchedStageID::PreRARematerialize:
    return std::make_unique<PreRARematStage>(SchedStageID, *this);
  case GCNSchedStageID::ILPInitialSchedule:
    return std::make_unique<ILPInitialScheduleStage>(SchedStageID, *this);
  case GCNSchedStageID::MemoryClauseInitialSchedule:
    return std::make_unique<MemoryClauseInitialScheduleStage>(SchedStageID,
                                                              *this);
  }

  llvm_unreachable("Unknown SchedStageID.");
}

void GCNScheduleDAGMILive::schedule() {
  // Collect all scheduling regions. The actual scheduling is performed in
  // GCNScheduleDAGMILive::finalizeSchedule.
  Regions.push_back(std::pair(RegionBegin, RegionEnd));
}

GCNRegPressure
GCNScheduleDAGMILive::getRealRegPressure(unsigned RegionIdx) const {
  if (Regions[RegionIdx].first == Regions[RegionIdx].second)
    return llvm::getRegPressure(MRI, LiveIns[RegionIdx]);
  GCNDownwardRPTracker RPTracker(*LIS);
  RPTracker.advance(Regions[RegionIdx].first, Regions[RegionIdx].second,
                    &LiveIns[RegionIdx]);
  return RPTracker.moveMaxPressure();
}

static MachineInstr *getLastMIForRegion(MachineBasicBlock::iterator RegionBegin,
                                        MachineBasicBlock::iterator RegionEnd) {
  auto REnd = RegionEnd == RegionBegin->getParent()->end()
                  ? std::prev(RegionEnd)
                  : RegionEnd;
  return &*skipDebugInstructionsBackward(REnd, RegionBegin);
}

void GCNScheduleDAGMILive::computeBlockPressure(unsigned RegionIdx,
                                                const MachineBasicBlock *MBB) {
  GCNDownwardRPTracker RPTracker(*LIS);

  // If the block has the only successor then live-ins of that successor are
  // live-outs of the current block. We can reuse calculated live set if the
  // successor will be sent to scheduling past current block.

  // However, due to the bug in LiveInterval analysis it may happen that two
  // predecessors of the same successor block have different lane bitmasks for
  // a live-out register. Workaround that by sticking to one-to-one relationship
  // i.e. one predecessor with one successor block.
  const MachineBasicBlock *OnlySucc = nullptr;
  if (MBB->succ_size() == 1) {
    auto *Candidate = *MBB->succ_begin();
    if (!Candidate->empty() && Candidate->pred_size() == 1) {
      SlotIndexes *Ind = LIS->getSlotIndexes();
      if (Ind->getMBBStartIdx(MBB) < Ind->getMBBStartIdx(Candidate))
        OnlySucc = Candidate;
    }
  }

  // Scheduler sends regions from the end of the block upwards.
  size_t CurRegion = RegionIdx;
  for (size_t E = Regions.size(); CurRegion != E; ++CurRegion)
    if (Regions[CurRegion].first->getParent() != MBB)
      break;
  --CurRegion;

  auto I = MBB->begin();
  auto LiveInIt = MBBLiveIns.find(MBB);
  auto &Rgn = Regions[CurRegion];
  auto *NonDbgMI = &*skipDebugInstructionsForward(Rgn.first, Rgn.second);
  if (LiveInIt != MBBLiveIns.end()) {
    auto LiveIn = std::move(LiveInIt->second);
    RPTracker.reset(*MBB->begin(), &LiveIn);
    MBBLiveIns.erase(LiveInIt);
  } else {
    I = Rgn.first;
    auto LRS = BBLiveInMap.lookup(NonDbgMI);
#ifdef EXPENSIVE_CHECKS
    assert(isEqual(getLiveRegsBefore(*NonDbgMI, *LIS), LRS));
#endif
    RPTracker.reset(*I, &LRS);
  }

  for (;;) {
    I = RPTracker.getNext();

    if (Regions[CurRegion].first == I || NonDbgMI == I) {
      LiveIns[CurRegion] = RPTracker.getLiveRegs();
      RPTracker.clearMaxPressure();
    }

    if (Regions[CurRegion].second == I) {
      Pressure[CurRegion] = RPTracker.moveMaxPressure();
      if (CurRegion-- == RegionIdx)
        break;
      auto &Rgn = Regions[CurRegion];
      NonDbgMI = &*skipDebugInstructionsForward(Rgn.first, Rgn.second);
    }
    RPTracker.advanceToNext();
    RPTracker.advanceBeforeNext();
  }

  if (OnlySucc) {
    if (I != MBB->end()) {
      RPTracker.advanceToNext();
      RPTracker.advance(MBB->end());
    }
    RPTracker.advanceBeforeNext();
    MBBLiveIns[OnlySucc] = RPTracker.moveLiveRegs();
  }
}

DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
GCNScheduleDAGMILive::getRegionLiveInMap() const {
  assert(!Regions.empty());
  std::vector<MachineInstr *> RegionFirstMIs;
  RegionFirstMIs.reserve(Regions.size());
  for (auto &[RegionBegin, RegionEnd] : reverse(Regions))
    RegionFirstMIs.push_back(
        &*skipDebugInstructionsForward(RegionBegin, RegionEnd));

  return getLiveRegMap(RegionFirstMIs, /*After=*/false, *LIS);
}

DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
GCNScheduleDAGMILive::getRegionLiveOutMap() const {
  assert(!Regions.empty());
  std::vector<MachineInstr *> RegionLastMIs;
  RegionLastMIs.reserve(Regions.size());
  for (auto &[RegionBegin, RegionEnd] : reverse(Regions))
    RegionLastMIs.push_back(getLastMIForRegion(RegionBegin, RegionEnd));

  return getLiveRegMap(RegionLastMIs, /*After=*/true, *LIS);
}

void RegionPressureMap::buildLiveRegMap() {
  IdxToInstruction.clear();

  RegionLiveRegMap =
      IsLiveOut ? DAG->getRegionLiveOutMap() : DAG->getRegionLiveInMap();
  for (unsigned I = 0; I < DAG->Regions.size(); I++) {
    MachineInstr *RegionKey =
        IsLiveOut
            ? getLastMIForRegion(DAG->Regions[I].first, DAG->Regions[I].second)
            : &*DAG->Regions[I].first;
    IdxToInstruction[I] = RegionKey;
  }
}

void GCNScheduleDAGMILive::finalizeSchedule() {
  // Start actual scheduling here. This function is called by the base
  // MachineScheduler after all regions have been recorded by
  // GCNScheduleDAGMILive::schedule().
  LiveIns.resize(Regions.size());
  Pressure.resize(Regions.size());
  RegionsWithHighRP.resize(Regions.size());
  RegionsWithExcessRP.resize(Regions.size());
  RegionsWithIGLPInstrs.resize(Regions.size());
  RegionsWithHighRP.reset();
  RegionsWithExcessRP.reset();
  RegionsWithIGLPInstrs.reset();

  runSchedStages();
}

void GCNScheduleDAGMILive::runSchedStages() {
  LLVM_DEBUG(dbgs() << "All regions recorded, starting actual scheduling.\n");

  if (!Regions.empty()) {
    BBLiveInMap = getRegionLiveInMap();
    if (GCNTrackers)
      RegionLiveOuts.buildLiveRegMap();
  }

  GCNSchedStrategy &S = static_cast<GCNSchedStrategy &>(*SchedImpl);
  while (S.advanceStage()) {
    auto Stage = createSchedStage(S.getCurrentStage());
    if (!Stage->initGCNSchedStage())
      continue;

    for (auto Region : Regions) {
      RegionBegin = Region.first;
      RegionEnd = Region.second;
      // Setup for scheduling the region and check whether it should be skipped.
      if (!Stage->initGCNRegion()) {
        Stage->advanceRegion();
        exitRegion();
        continue;
      }

      if (GCNTrackers) {
        GCNDownwardRPTracker *DownwardTracker = S.getDownwardTracker();
        GCNUpwardRPTracker *UpwardTracker = S.getUpwardTracker();
        GCNRPTracker::LiveRegSet *RegionLiveIns =
            &LiveIns[Stage->getRegionIdx()];

        reinterpret_cast<GCNRPTracker *>(DownwardTracker)
            ->reset(MRI, *RegionLiveIns);
        reinterpret_cast<GCNRPTracker *>(UpwardTracker)
            ->reset(MRI, RegionLiveOuts.getLiveRegsForRegionIdx(
                             Stage->getRegionIdx()));
      }

      ScheduleDAGMILive::schedule();
      Stage->finalizeGCNRegion();
    }

    Stage->finalizeGCNSchedStage();
  }
}

#ifndef NDEBUG
raw_ostream &llvm::operator<<(raw_ostream &OS, const GCNSchedStageID &StageID) {
  switch (StageID) {
  case GCNSchedStageID::OccInitialSchedule:
    OS << "Max Occupancy Initial Schedule";
    break;
  case GCNSchedStageID::UnclusteredHighRPReschedule:
    OS << "Unclustered High Register Pressure Reschedule";
    break;
  case GCNSchedStageID::ClusteredLowOccupancyReschedule:
    OS << "Clustered Low Occupancy Reschedule";
    break;
  case GCNSchedStageID::PreRARematerialize:
    OS << "Pre-RA Rematerialize";
    break;
  case GCNSchedStageID::ILPInitialSchedule:
    OS << "Max ILP Initial Schedule";
    break;
  case GCNSchedStageID::MemoryClauseInitialSchedule:
    OS << "Max memory clause Initial Schedule";
    break;
  }

  return OS;
}
#endif

GCNSchedStage::GCNSchedStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
    : DAG(DAG), S(static_cast<GCNSchedStrategy &>(*DAG.SchedImpl)), MF(DAG.MF),
      MFI(DAG.MFI), ST(DAG.ST), StageID(StageID) {}

bool GCNSchedStage::initGCNSchedStage() {
  if (!DAG.LIS)
    return false;

  LLVM_DEBUG(dbgs() << "Starting scheduling stage: " << StageID << "\n");
  return true;
}

bool UnclusteredHighRPStage::initGCNSchedStage() {
  if (DisableUnclusterHighRP)
    return false;

  if (!GCNSchedStage::initGCNSchedStage())
    return false;

  if (DAG.RegionsWithHighRP.none() && DAG.RegionsWithExcessRP.none())
    return false;

  SavedMutations.swap(DAG.Mutations);
  DAG.addMutation(
      createIGroupLPDAGMutation(AMDGPU::SchedulingPhase::PreRAReentry));

  InitialOccupancy = DAG.MinOccupancy;
  // Aggressivly try to reduce register pressure in the unclustered high RP
  // stage. Temporarily increase occupancy target in the region.
  S.SGPRLimitBias = S.HighRPSGPRBias;
  S.VGPRLimitBias = S.HighRPVGPRBias;
  if (MFI.getMaxWavesPerEU() > DAG.MinOccupancy)
    MFI.increaseOccupancy(MF, ++DAG.MinOccupancy);

  LLVM_DEBUG(
      dbgs()
      << "Retrying function scheduling without clustering. "
         "Aggressivly try to reduce register pressure to achieve occupancy "
      << DAG.MinOccupancy << ".\n");

  return true;
}

bool ClusteredLowOccStage::initGCNSchedStage() {
  if (DisableClusteredLowOccupancy)
    return false;

  if (!GCNSchedStage::initGCNSchedStage())
    return false;

  // Don't bother trying to improve ILP in lower RP regions if occupancy has not
  // been dropped. All regions will have already been scheduled with the ideal
  // occupancy targets.
  if (DAG.StartingOccupancy <= DAG.MinOccupancy)
    return false;

  LLVM_DEBUG(
      dbgs() << "Retrying function scheduling with lowest recorded occupancy "
             << DAG.MinOccupancy << ".\n");
  return true;
}

/// Allows to easily filter for this stage's debug output.
#define REMAT_PREFIX "[PreRARemat] "
#define REMAT_DEBUG(X) LLVM_DEBUG(dbgs() << REMAT_PREFIX; X;)

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
Printable PreRARematStage::printChain(unsigned RootIdx) const {
  return Printable([&, RootIdx](raw_ostream &OS) {
    SmallDenseSet<unsigned, 4> Visited;
    SmallVector<unsigned> RegIndices;
    std::function<void(unsigned)> Visit = [&](unsigned RegIdx) {
      RegIndices.push_back(RegIdx);
      for (const RematReg::Dependency &Dep : RDAG.getReg(RegIdx).Dependencies) {
        int Status = RematRegs[Dep.RegIdx].Status;
        if (Status != RegLiveness::NON_ROOT)
          continue;
        if (Visited.insert(Dep.RegIdx).second)
          Visit(Dep.RegIdx);
      }
    };
    int Status = RematRegs[RootIdx].Status;
    if (Status == RegLiveness::INVALID)
      RegIndices.push_back(RootIdx);
    else
      Visit(RootIdx);

    auto *RegIdxIt = RegIndices.begin();
    OS << '(' << *RegIdxIt;
    while (++RegIdxIt != RegIndices.end())
      OS << ',' << *RegIdxIt;
    OS << ')';
  });
}

Printable PreRARematStage::printReg(unsigned RegIdx) const {
  return Printable([&, RegIdx](raw_ostream &OS) {
    OS << printChain(RegIdx) << ' ' << RDAG.getReg(RegIdx).print() << '\n';
    const auto [_, LiveReg] = getReg(RegIdx);
    for (unsigned I : LiveReg.Live.set_bits()) {
      OS << REMAT_PREFIX << "  [" << I << "] RP diff"
         << (LiveReg.ApproximateDiff[I] ? " (approximate)" : "") << ": "
         << GCNRPTarget::RPDiff(LiveReg.Save, /*NegPressure=*/true) << '\n';
    }
  });
}

Printable PreRARematStage::RegionsInfo::print() const {
  return Printable([&](raw_ostream &OS) {
    if (TargetRegions.none()) {
      dbgs() << REMAT_PREFIX << "No target regions\n";
      return;
    }
    dbgs() << REMAT_PREFIX << "Target regions:\n";
    for (unsigned I : TargetRegions.set_bits()) {
      dbgs() << REMAT_PREFIX << "  [" << I << "] " << RPTargets[I] << '\n';
      if (TargetLiveIn[I])
        dbgs() << REMAT_PREFIX << "   Live-in pressure above target\n";
      if (TargetLiveOut[I])
        dbgs() << REMAT_PREFIX << "   Live-out pressure above target\n";
    }
  });
}

Printable PreRARematStage::RematCandidate::print() const {
  return Printable([&](raw_ostream &OS) {
    OS << '(' << MaxFreq << ", " << FreqDiff << ", " << RegionImpact << ')';
  });
}
#endif

bool PreRARematStage::initGCNSchedStage() {
  // FIXME: This pass will invalidate cached BBLiveInMap and MBBLiveIns for
  // regions inbetween the defs and region we sinked the def to. Will need to be
  // fixed if there is another pass after this pass.
  assert(!S.hasNextStage());
  if (DisablePreRARemat)
    return false;

  if (!GCNSchedStage::initGCNSchedStage() || DAG.Regions.size() <= 1)
    return false;

  DAG.RegionLiveOuts.buildLiveRegMap();

  RegionsInfo RI(DAG.Regions.size());
  TargetOcc = RI.determineObjective(DAG);

  // Set an objective for the stage based on current RP in each region.
  REMAT_DEBUG({
    dbgs() << "Analyzing ";
    MF.getFunction().printAsOperand(dbgs(), false);
    dbgs() << ": ";
    if (!RI.TargetRegions.any()) {
      dbgs() << "no objective to achieve, occupancy is maximal at "
             << MFI.getMaxWavesPerEU() << '\n';
    }
    if (TargetOcc) {
      dbgs() << "increase occupancy from " << *TargetOcc - 1 << '\n'
             << RI.print();
    } else {
      dbgs() << "reduce spilling (minimum target occupancy is "
             << MFI.getMinWavesPerEU() << ")\n"
             << RI.print();
    }
  });

  if (!RDAG.build()) {
    REMAT_DEBUG(dbgs() << "Nothing to rematerialize\n");
    return false;
  }

  // for (unsigned I : RI.TargetRegions.set_bits()) {
  //   dbgs() << "For target region [" << I << "]\n";
  //   for (auto MI = DAG.Regions[I].first; MI != DAG.Regions[I].second; ++MI)
  //     dbgs() << "| " << *MI;

  //   auto &LiveIns = DAG.LiveIns[I];
  //   auto &LiveOuts = DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I);

  //   auto PrintRegMap = [&](const auto &Live) {
  //     for (const auto &[Reg, Mask] : Live) {
  //       auto Defs = DAG.MRI.def_operands(Reg);
  //       dbgs() << "    " << llvm::printReg(Reg, DAG.TRI, 0, &DAG.MRI)
  //              << " with " << std::distance(Defs.begin(), Defs.end())
  //              << " defs:\n";
  //       for (MachineOperand &DefMO : Defs) {
  //         dbgs() << "      " << *DefMO.getParent();
  //       }
  //     }
  //   };
  //   dbgs() << "  Live-ins: (" << LiveIns.size() << ")\n";
  //   PrintRegMap(LiveIns);
  //   dbgs() << "  Live-outs: (" << LiveOuts.size() << ")\n";
  //   PrintRegMap(LiveOuts);
  // }

  if (!performCrossRegionRemats(RI) && !TargetOcc) {
    // When cross-region rematerializations have not succeeded in achieving the
    // target pressure everywhere and we are trying to reduce spilling, perform
    // more aggressive rematerializations within regions.
    // performSameRegionRemats(RI);
  }

  if (RescheduleRegions.none())
    return false;

  // Commit all pressure changes to the DAG and compute minimum achieved
  // occupancy in impacted regions.
  REMAT_DEBUG(dbgs() << "==== REMAT RESULTS ====\n");
  unsigned DynamicVGPRBlockSize = MFI.getDynamicVGPRBlockSize();
  for (unsigned I : RescheduleRegions.set_bits()) {
    DAG.Pressure[I] = RI.RPTargets[I].getCurrentRP();
    REMAT_DEBUG(dbgs() << '[' << I << "] Achieved occupancy "
                       << DAG.Pressure[I].getOccupancy(ST, DynamicVGPRBlockSize)
                       << " (" << RI.RPTargets[I] << ")\n");
  }
  AchievedOcc = MFI.getMaxWavesPerEU();
  for (const GCNRegPressure &RP : DAG.Pressure) {
    AchievedOcc =
        std::min(AchievedOcc, RP.getOccupancy(ST, DynamicVGPRBlockSize));
  }

  REMAT_DEBUG({
    dbgs() << "Retrying function scheduling with new min. occupancy of "
           << AchievedOcc << " from rematerializing (original was "
           << DAG.MinOccupancy;
    if (TargetOcc)
      dbgs() << ", target was " << *TargetOcc;
    dbgs() << ")\n";
  });
  if (AchievedOcc > DAG.MinOccupancy) {
    DAG.MinOccupancy = AchievedOcc;
    SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
    MFI.increaseOccupancy(MF, DAG.MinOccupancy);
  }
  return true;
}

void GCNSchedStage::finalizeGCNSchedStage() {
  DAG.finishBlock();
  LLVM_DEBUG(dbgs() << "Ending scheduling stage: " << StageID << "\n");
}

void UnclusteredHighRPStage::finalizeGCNSchedStage() {
  SavedMutations.swap(DAG.Mutations);
  S.SGPRLimitBias = S.VGPRLimitBias = 0;
  if (DAG.MinOccupancy > InitialOccupancy) {
    LLVM_DEBUG(dbgs() << StageID
                      << " stage successfully increased occupancy to "
                      << DAG.MinOccupancy << '\n');
  }

  GCNSchedStage::finalizeGCNSchedStage();
}

bool GCNSchedStage::initGCNRegion() {
  // Skip empty scheduling region.
  if (DAG.begin() == DAG.end())
    return false;

  // Check whether this new region is also a new block.
  if (DAG.RegionBegin->getParent() != CurrentMBB)
    setupNewBlock();

  unsigned NumRegionInstrs = std::distance(DAG.begin(), DAG.end());
  DAG.enterRegion(CurrentMBB, DAG.begin(), DAG.end(), NumRegionInstrs);

  // Skip regions with 1 schedulable instruction.
  if (DAG.begin() == std::prev(DAG.end()))
    return false;

  LLVM_DEBUG(dbgs() << "********** MI Scheduling **********\n");
  LLVM_DEBUG(dbgs() << MF.getName() << ":" << printMBBReference(*CurrentMBB)
                    << " " << CurrentMBB->getName()
                    << "\n  From: " << *DAG.begin() << "    To: ";
             if (DAG.RegionEnd != CurrentMBB->end()) dbgs() << *DAG.RegionEnd;
             else dbgs() << "End";
             dbgs() << " RegionInstrs: " << NumRegionInstrs << '\n');

  // Save original instruction order before scheduling for possible revert.
  Unsched.clear();
  Unsched.reserve(DAG.NumRegionInstrs);
  if (StageID == GCNSchedStageID::OccInitialSchedule ||
      StageID == GCNSchedStageID::ILPInitialSchedule) {
    const SIInstrInfo *SII = static_cast<const SIInstrInfo *>(DAG.TII);
    for (auto &I : DAG) {
      Unsched.push_back(&I);
      if (SII->isIGLPMutationOnly(I.getOpcode()))
        DAG.RegionsWithIGLPInstrs[RegionIdx] = true;
    }
  } else {
    for (auto &I : DAG)
      Unsched.push_back(&I);
  }

  PressureBefore = DAG.Pressure[RegionIdx];

  LLVM_DEBUG(
      dbgs() << "Pressure before scheduling:\nRegion live-ins:"
             << print(DAG.LiveIns[RegionIdx], DAG.MRI)
             << "Region live-in pressure:  "
             << print(llvm::getRegPressure(DAG.MRI, DAG.LiveIns[RegionIdx]))
             << "Region register pressure: " << print(PressureBefore));

  S.HasHighPressure = false;
  S.KnownExcessRP = isRegionWithExcessRP();

  if (DAG.RegionsWithIGLPInstrs[RegionIdx] &&
      StageID != GCNSchedStageID::UnclusteredHighRPReschedule) {
    SavedMutations.clear();
    SavedMutations.swap(DAG.Mutations);
    bool IsInitialStage = StageID == GCNSchedStageID::OccInitialSchedule ||
                          StageID == GCNSchedStageID::ILPInitialSchedule;
    DAG.addMutation(createIGroupLPDAGMutation(
        IsInitialStage ? AMDGPU::SchedulingPhase::Initial
                       : AMDGPU::SchedulingPhase::PreRAReentry));
  }

  return true;
}

bool UnclusteredHighRPStage::initGCNRegion() {
  // Only reschedule regions that have excess register pressure (i.e.
  // spilling) or had minimum occupancy at the beginning of the stage (as long
  // as rescheduling of previous regions did not make occupancy drop back down
  // to the initial minimum).
  unsigned DynamicVGPRBlockSize = DAG.MFI.getDynamicVGPRBlockSize();
  if (!DAG.RegionsWithExcessRP[RegionIdx] &&
      (DAG.MinOccupancy <= InitialOccupancy ||
       DAG.Pressure[RegionIdx].getOccupancy(ST, DynamicVGPRBlockSize) !=
           InitialOccupancy))
    return false;

  return GCNSchedStage::initGCNRegion();
}

bool ClusteredLowOccStage::initGCNRegion() {
  // We may need to reschedule this region if it wasn't rescheduled in the
  // last stage, or if we found it was testing critical register pressure
  // limits in the unclustered reschedule stage. The later is because we may
  // not have been able to raise the min occupancy in the previous stage so
  // the region may be overly constrained even if it was already rescheduled.
  if (!DAG.RegionsWithHighRP[RegionIdx])
    return false;

  return GCNSchedStage::initGCNRegion();
}

bool PreRARematStage::initGCNRegion() {
  return RescheduleRegions[RegionIdx] && GCNSchedStage::initGCNRegion();
}

void GCNSchedStage::setupNewBlock() {
  if (CurrentMBB)
    DAG.finishBlock();

  CurrentMBB = DAG.RegionBegin->getParent();
  DAG.startBlock(CurrentMBB);
  // Get real RP for the region if it hasn't be calculated before. After the
  // initial schedule stage real RP will be collected after scheduling.
  if (StageID == GCNSchedStageID::OccInitialSchedule ||
      StageID == GCNSchedStageID::ILPInitialSchedule ||
      StageID == GCNSchedStageID::MemoryClauseInitialSchedule)
    DAG.computeBlockPressure(RegionIdx, CurrentMBB);
}

void GCNSchedStage::finalizeGCNRegion() {
  DAG.Regions[RegionIdx] = std::pair(DAG.RegionBegin, DAG.RegionEnd);
  if (S.HasHighPressure)
    DAG.RegionsWithHighRP[RegionIdx] = true;

  // Revert scheduling if we have dropped occupancy or there is some other
  // reason that the original schedule is better.
  checkScheduling();

  if (DAG.RegionsWithIGLPInstrs[RegionIdx] &&
      StageID != GCNSchedStageID::UnclusteredHighRPReschedule)
    SavedMutations.swap(DAG.Mutations);

  DAG.exitRegion();
  advanceRegion();
}

void GCNSchedStage::checkScheduling() {
  // Check the results of scheduling.
  PressureAfter = DAG.getRealRegPressure(RegionIdx);

  LLVM_DEBUG(dbgs() << "Pressure after scheduling: " << print(PressureAfter));
  LLVM_DEBUG(dbgs() << "Region: " << RegionIdx << ".\n");

  unsigned DynamicVGPRBlockSize = DAG.MFI.getDynamicVGPRBlockSize();

  if (PressureAfter.getSGPRNum() <= S.SGPRCriticalLimit &&
      PressureAfter.getVGPRNum(ST.hasGFX90AInsts()) <= S.VGPRCriticalLimit) {
    DAG.Pressure[RegionIdx] = PressureAfter;

    // Early out if we have achieved the occupancy target.
    LLVM_DEBUG(dbgs() << "Pressure in desired limits, done.\n");
    return;
  }

  unsigned TargetOccupancy = std::min(
      S.getTargetOccupancy(), ST.getOccupancyWithWorkGroupSizes(MF).second);
  unsigned WavesAfter = std::min(
      TargetOccupancy, PressureAfter.getOccupancy(ST, DynamicVGPRBlockSize));
  unsigned WavesBefore = std::min(
      TargetOccupancy, PressureBefore.getOccupancy(ST, DynamicVGPRBlockSize));
  LLVM_DEBUG(dbgs() << "Occupancy before scheduling: " << WavesBefore
                    << ", after " << WavesAfter << ".\n");

  // We may not be able to keep the current target occupancy because of the
  // just scheduled region. We might still be able to revert scheduling if the
  // occupancy before was higher, or if the current schedule has register
  // pressure higher than the excess limits which could lead to more spilling.
  unsigned NewOccupancy = std::max(WavesAfter, WavesBefore);

  // Allow memory bound functions to drop to 4 waves if not limited by an
  // attribute.
  if (WavesAfter < WavesBefore && WavesAfter < DAG.MinOccupancy &&
      WavesAfter >= MFI.getMinAllowedOccupancy()) {
    LLVM_DEBUG(dbgs() << "Function is memory bound, allow occupancy drop up to "
                      << MFI.getMinAllowedOccupancy() << " waves\n");
    NewOccupancy = WavesAfter;
  }

  if (NewOccupancy < DAG.MinOccupancy) {
    DAG.MinOccupancy = NewOccupancy;
    MFI.limitOccupancy(DAG.MinOccupancy);
    LLVM_DEBUG(dbgs() << "Occupancy lowered for the function to "
                      << DAG.MinOccupancy << ".\n");
  }
  // The maximum number of arch VGPR on non-unified register file, or the
  // maximum VGPR + AGPR in the unified register file case.
  unsigned MaxVGPRs = ST.getMaxNumVGPRs(MF);
  // The maximum number of arch VGPR for both unified and non-unified register
  // file.
  unsigned MaxArchVGPRs = std::min(MaxVGPRs, ST.getAddressableNumArchVGPRs());
  unsigned MaxSGPRs = ST.getMaxNumSGPRs(MF);

  if (PressureAfter.getVGPRNum(ST.hasGFX90AInsts()) > MaxVGPRs ||
      PressureAfter.getArchVGPRNum() > MaxArchVGPRs ||
      PressureAfter.getAGPRNum() > MaxArchVGPRs ||
      PressureAfter.getSGPRNum() > MaxSGPRs) {
    DAG.RegionsWithHighRP[RegionIdx] = true;
    DAG.RegionsWithExcessRP[RegionIdx] = true;
  }

  // Revert if this region's schedule would cause a drop in occupancy or
  // spilling.
  if (shouldRevertScheduling(WavesAfter))
    revertScheduling();
  else
    DAG.Pressure[RegionIdx] = PressureAfter;
}

unsigned
GCNSchedStage::computeSUnitReadyCycle(const SUnit &SU, unsigned CurrCycle,
                                      DenseMap<unsigned, unsigned> &ReadyCycles,
                                      const TargetSchedModel &SM) {
  unsigned ReadyCycle = CurrCycle;
  for (auto &D : SU.Preds) {
    if (D.isAssignedRegDep()) {
      MachineInstr *DefMI = D.getSUnit()->getInstr();
      unsigned Latency = SM.computeInstrLatency(DefMI);
      unsigned DefReady = ReadyCycles[DAG.getSUnit(DefMI)->NodeNum];
      ReadyCycle = std::max(ReadyCycle, DefReady + Latency);
    }
  }
  ReadyCycles[SU.NodeNum] = ReadyCycle;
  return ReadyCycle;
}

#ifndef NDEBUG
struct EarlierIssuingCycle {
  bool operator()(std::pair<MachineInstr *, unsigned> A,
                  std::pair<MachineInstr *, unsigned> B) const {
    return A.second < B.second;
  }
};

static void printScheduleModel(std::set<std::pair<MachineInstr *, unsigned>,
                                        EarlierIssuingCycle> &ReadyCycles) {
  if (ReadyCycles.empty())
    return;
  unsigned BBNum = ReadyCycles.begin()->first->getParent()->getNumber();
  dbgs() << "\n################## Schedule time ReadyCycles for MBB : " << BBNum
         << " ##################\n# Cycle #\t\t\tInstruction          "
            "             "
            "                            \n";
  unsigned IPrev = 1;
  for (auto &I : ReadyCycles) {
    if (I.second > IPrev + 1)
      dbgs() << "****************************** BUBBLE OF " << I.second - IPrev
             << " CYCLES DETECTED ******************************\n\n";
    dbgs() << "[ " << I.second << " ]  :  " << *I.first << "\n";
    IPrev = I.second;
  }
}
#endif

ScheduleMetrics
GCNSchedStage::getScheduleMetrics(const std::vector<SUnit> &InputSchedule) {
#ifndef NDEBUG
  std::set<std::pair<MachineInstr *, unsigned>, EarlierIssuingCycle>
      ReadyCyclesSorted;
#endif
  const TargetSchedModel &SM = ST.getInstrInfo()->getSchedModel();
  unsigned SumBubbles = 0;
  DenseMap<unsigned, unsigned> ReadyCycles;
  unsigned CurrCycle = 0;
  for (auto &SU : InputSchedule) {
    unsigned ReadyCycle =
        computeSUnitReadyCycle(SU, CurrCycle, ReadyCycles, SM);
    SumBubbles += ReadyCycle - CurrCycle;
#ifndef NDEBUG
    ReadyCyclesSorted.insert(std::make_pair(SU.getInstr(), ReadyCycle));
#endif
    CurrCycle = ++ReadyCycle;
  }
#ifndef NDEBUG
  LLVM_DEBUG(
      printScheduleModel(ReadyCyclesSorted);
      dbgs() << "\n\t"
             << "Metric: "
             << (SumBubbles
                     ? (SumBubbles * ScheduleMetrics::ScaleFactor) / CurrCycle
                     : 1)
             << "\n\n");
#endif

  return ScheduleMetrics(CurrCycle, SumBubbles);
}

ScheduleMetrics
GCNSchedStage::getScheduleMetrics(const GCNScheduleDAGMILive &DAG) {
#ifndef NDEBUG
  std::set<std::pair<MachineInstr *, unsigned>, EarlierIssuingCycle>
      ReadyCyclesSorted;
#endif
  const TargetSchedModel &SM = ST.getInstrInfo()->getSchedModel();
  unsigned SumBubbles = 0;
  DenseMap<unsigned, unsigned> ReadyCycles;
  unsigned CurrCycle = 0;
  for (auto &MI : DAG) {
    SUnit *SU = DAG.getSUnit(&MI);
    if (!SU)
      continue;
    unsigned ReadyCycle =
        computeSUnitReadyCycle(*SU, CurrCycle, ReadyCycles, SM);
    SumBubbles += ReadyCycle - CurrCycle;
#ifndef NDEBUG
    ReadyCyclesSorted.insert(std::make_pair(SU->getInstr(), ReadyCycle));
#endif
    CurrCycle = ++ReadyCycle;
  }
#ifndef NDEBUG
  LLVM_DEBUG(
      printScheduleModel(ReadyCyclesSorted);
      dbgs() << "\n\t"
             << "Metric: "
             << (SumBubbles
                     ? (SumBubbles * ScheduleMetrics::ScaleFactor) / CurrCycle
                     : 1)
             << "\n\n");
#endif

  return ScheduleMetrics(CurrCycle, SumBubbles);
}

bool GCNSchedStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (WavesAfter < DAG.MinOccupancy)
    return true;

  // For dynamic VGPR mode, we don't want to waste any VGPR blocks.
  if (DAG.MFI.isDynamicVGPREnabled()) {
    unsigned BlocksBefore = AMDGPU::IsaInfo::getAllocatedNumVGPRBlocks(
        &ST, DAG.MFI.getDynamicVGPRBlockSize(),
        PressureBefore.getVGPRNum(false));
    unsigned BlocksAfter = AMDGPU::IsaInfo::getAllocatedNumVGPRBlocks(
        &ST, DAG.MFI.getDynamicVGPRBlockSize(),
        PressureAfter.getVGPRNum(false));
    if (BlocksAfter > BlocksBefore)
      return true;
  }

  return false;
}

bool OccInitialScheduleStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (PressureAfter == PressureBefore)
    return false;

  if (GCNSchedStage::shouldRevertScheduling(WavesAfter))
    return true;

  if (mayCauseSpilling(WavesAfter))
    return true;

  return false;
}

bool UnclusteredHighRPStage::shouldRevertScheduling(unsigned WavesAfter) {
  // If RP is not reduced in the unclustered reschedule stage, revert to the
  // old schedule.
  if ((WavesAfter <=
           PressureBefore.getOccupancy(ST, DAG.MFI.getDynamicVGPRBlockSize()) &&
       mayCauseSpilling(WavesAfter)) ||
      GCNSchedStage::shouldRevertScheduling(WavesAfter)) {
    LLVM_DEBUG(dbgs() << "Unclustered reschedule did not help.\n");
    return true;
  }

  // Do not attempt to relax schedule even more if we are already spilling.
  if (isRegionWithExcessRP())
    return false;

  LLVM_DEBUG(
      dbgs()
      << "\n\t      *** In shouldRevertScheduling ***\n"
      << "      *********** BEFORE UnclusteredHighRPStage ***********\n");
  ScheduleMetrics MBefore = getScheduleMetrics(DAG.SUnits);
  LLVM_DEBUG(
      dbgs()
      << "\n      *********** AFTER UnclusteredHighRPStage ***********\n");
  ScheduleMetrics MAfter = getScheduleMetrics(DAG);
  unsigned OldMetric = MBefore.getMetric();
  unsigned NewMetric = MAfter.getMetric();
  unsigned WavesBefore = std::min(
      S.getTargetOccupancy(),
      PressureBefore.getOccupancy(ST, DAG.MFI.getDynamicVGPRBlockSize()));
  unsigned Profit =
      ((WavesAfter * ScheduleMetrics::ScaleFactor) / WavesBefore *
       ((OldMetric + ScheduleMetricBias) * ScheduleMetrics::ScaleFactor) /
       NewMetric) /
      ScheduleMetrics::ScaleFactor;
  LLVM_DEBUG(dbgs() << "\tMetric before " << MBefore << "\tMetric after "
                    << MAfter << "Profit: " << Profit << "\n");
  return Profit < ScheduleMetrics::ScaleFactor;
}

bool ClusteredLowOccStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (PressureAfter == PressureBefore)
    return false;

  if (GCNSchedStage::shouldRevertScheduling(WavesAfter))
    return true;

  if (mayCauseSpilling(WavesAfter))
    return true;

  return false;
}

bool PreRARematStage::shouldRevertScheduling(unsigned WavesAfter) {
  return GCNSchedStage::shouldRevertScheduling(WavesAfter) ||
         mayCauseSpilling(WavesAfter) || (TargetOcc && WavesAfter < TargetOcc);
}

bool ILPInitialScheduleStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (mayCauseSpilling(WavesAfter))
    return true;

  return false;
}

bool MemoryClauseInitialScheduleStage::shouldRevertScheduling(
    unsigned WavesAfter) {
  return mayCauseSpilling(WavesAfter);
}

bool GCNSchedStage::mayCauseSpilling(unsigned WavesAfter) {
  if (WavesAfter <= MFI.getMinWavesPerEU() && isRegionWithExcessRP() &&
      !PressureAfter.less(MF, PressureBefore)) {
    LLVM_DEBUG(dbgs() << "New pressure will result in more spilling.\n");
    return true;
  }

  return false;
}

void GCNSchedStage::revertScheduling() {
  LLVM_DEBUG(dbgs() << "Attempting to revert scheduling.\n");
  DAG.RegionEnd = DAG.RegionBegin;
  int SkippedDebugInstr = 0;
  for (MachineInstr *MI : Unsched) {
    if (MI->isDebugInstr()) {
      ++SkippedDebugInstr;
      continue;
    }

    if (MI->getIterator() != DAG.RegionEnd) {
      DAG.BB->splice(DAG.RegionEnd, DAG.BB, MI);
      if (!MI->isDebugInstr())
        DAG.LIS->handleMove(*MI, true);
    }

    // Reset read-undef flags and update them later.
    for (auto &Op : MI->all_defs())
      Op.setIsUndef(false);
    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *DAG.TRI, DAG.MRI, DAG.ShouldTrackLaneMasks, false);
    if (!MI->isDebugInstr()) {
      if (DAG.ShouldTrackLaneMasks) {
        // Adjust liveness and add missing dead+read-undef flags.
        SlotIndex SlotIdx = DAG.LIS->getInstructionIndex(*MI).getRegSlot();
        RegOpers.adjustLaneLiveness(*DAG.LIS, DAG.MRI, SlotIdx, MI);
      } else {
        // Adjust for missing dead-def flags.
        RegOpers.detectDeadDefs(*MI, *DAG.LIS);
      }
    }
    DAG.RegionEnd = MI->getIterator();
    ++DAG.RegionEnd;
    LLVM_DEBUG(dbgs() << "Scheduling " << *MI);
  }

  // After reverting schedule, debug instrs will now be at the end of the
  // block and RegionEnd will point to the first debug instr. Increment
  // RegionEnd pass debug instrs to the actual end of the scheduling region.
  while (SkippedDebugInstr-- > 0)
    ++DAG.RegionEnd;

  // If Unsched.front() instruction is a debug instruction, this will actually
  // shrink the region since we moved all debug instructions to the end of the
  // block. Find the first instruction that is not a debug instruction.
  DAG.RegionBegin = Unsched.front()->getIterator();
  if (DAG.RegionBegin->isDebugInstr()) {
    for (MachineInstr *MI : Unsched) {
      if (MI->isDebugInstr())
        continue;
      DAG.RegionBegin = MI->getIterator();
      break;
    }
  }

  // Then move the debug instructions back into their correct place and set
  // RegionBegin and RegionEnd if needed.
  DAG.placeDebugValues();

  DAG.Regions[RegionIdx] = std::pair(DAG.RegionBegin, DAG.RegionEnd);
}

std::optional<unsigned> PreRARematStage::RegionsInfo::determineObjective(
    const GCNScheduleDAGMILive &DAG) {
  const Function &F = DAG.MF.getFunction();

  // Set up "spilling targets" for all regions.
  unsigned MaxSGPRs = DAG.ST.getMaxNumSGPRs(F);
  unsigned MaxVGPRs = DAG.ST.getMaxNumVGPRs(F);
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    const GCNRegPressure &RP = DAG.Pressure[I];
    GCNRPTarget &Target =
        RPTargets.emplace_back(MaxSGPRs, MaxVGPRs, DAG.MF, RP);
    if (!Target.satisfied())
      calculatePressureInfo(I, DAG);
  }

  // In addition to register usage being above addressable limits, occupancy
  // below the minimum is considered like "spilling" as well.
  if (TargetRegions.any() || DAG.MinOccupancy >= DAG.MFI.getMaxWavesPerEU())
    return std::nullopt;

  // There is no spilling and room to improve occupancy; set up "increased
  // occupancy targets" for all regions.
  unsigned TargetOcc = DAG.MinOccupancy + 1;
  const unsigned VGPRBlockSize = DAG.MFI.getDynamicVGPRBlockSize();
  MaxSGPRs = DAG.ST.getMaxNumSGPRs(TargetOcc, false);
  MaxVGPRs = DAG.ST.getMaxNumVGPRs(TargetOcc, VGPRBlockSize);
  for (auto [I, Target] : enumerate(RPTargets)) {
    Target.setTarget(MaxSGPRs, MaxVGPRs);
    if (!Target.satisfied())
      calculatePressureInfo(I, DAG);
  }
  return TargetOcc;
}

void PreRARematStage::RegionsInfo::clearIfSatisfied(const BitVector &Regions) {
  for (unsigned I : Regions.set_bits()) {
    if (TargetRegions[I] && RPTargets[I].satisfied()) {
      REMAT_DEBUG(dbgs() << "  [" << I << "] Target reached!\n");
      TargetRegions.reset(I);
    }
  }
}

bool PreRARematStage::RegionsInfo::updateAndVerify(
    const BitVector &Regions, const GCNScheduleDAGMILive &DAG) {
  bool TooOptimistic = false;
  for (unsigned I : Regions.set_bits()) {
    bool WasNotTarget = !TargetRegions[I];
    calculatePressureInfo(I, DAG);
    if (WasNotTarget && TargetRegions[I]) {
      // Since we were optimistic in assessing RP decreases in these regions, we
      // may need to re-tag the target region if RP didn't decrease as expected.
      REMAT_DEBUG(dbgs() << "  [" << I << "] Reverting target status\n");
      TooOptimistic = true;
      TargetRegions.set(I);
    }
  }
  return TooOptimistic;
}

void PreRARematStage::RegionsInfo::calculatePressureInfo(
    unsigned I, const GCNScheduleDAGMILive &DAG) {
  GCNRPTarget &Target = RPTargets[I];
  const auto &[RegionBegin, RegionEnd] = DAG.Regions[I];
  TargetSlots.erase(I);

  GCNDownwardRPTracker RPTracker(*DAG.LIS);
  if (RegionBegin == RegionEnd ||
      !RPTracker.reset(*RegionBegin, &DAG.LiveIns[I])) {
    bool AboveTarget =
        !Target.satisfied(getRegPressure(DAG.MRI, DAG.LiveIns[I]));
    TargetRegions[I] = TargetLiveIn[I] = TargetLiveOut[I] = AboveTarget;
    return;
  }

  SmallVector<SlotIndex, 4> Slots;

  bool AboveTarget = !Target.satisfied(RPTracker.getPressure());
  TargetLiveIn[I] = AboveTarget;

  while (RPTracker.getNext() != RegionEnd) {
    RPTracker.advance();
    if (Target.satisfied(RPTracker.getPressure())) {
      if (AboveTarget) {
        const MachineInstr &MI = *RPTracker.getLastTrackedMI();
        Slots.push_back(DAG.LIS->getInstructionIndex(MI));
        AboveTarget = false;
      }
    } else if (!AboveTarget) {
      const MachineInstr &MI = *RPTracker.getLastTrackedMI();
      Slots.push_back(DAG.LIS->getInstructionIndex(MI));
      AboveTarget = true;
    }
  }

  if (!Slots.empty())
    TargetSlots.try_emplace(I, Slots);

  TargetLiveOut[I] = AboveTarget;
  TargetRegions[I] = TargetLiveIn[I] || TargetLiveOut[I] || !Slots.empty();
}

PreRARematStage::RegLiveness::RegLiveness(unsigned RegIdx,
                                          const GCNScheduleDAGMILive &DAG,
                                          const RematDAG &RDAG,
                                          bool SkipLiveCheck)
    : LiveIn(DAG.Regions.size()), LiveOut(DAG.Regions.size()),
      Live(DAG.Regions.size()), ApproximateDiff(DAG.Regions.size()) {
  const RematReg &Reg = RDAG.getReg(RegIdx);

  // Mark regions in which the rematerializable register is live before being
  // potentially rematerialized.
  Register DefReg = Reg.getDefReg();
  if (!SkipLiveCheck) {
    for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
      auto LiveInIt = DAG.LiveIns[I].find(DefReg);
      if (LiveInIt != DAG.LiveIns[I].end() && LiveInIt->second.any())
        LiveIn.set(I);

      if (auto *LiveOuts = DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I)) {
        auto LiveOutIt = LiveOuts->find(DefReg);
        if (LiveOutIt != LiveOuts->end() && LiveOutIt->second.any())
          LiveOut.set(I);
      }
    }
    Live |= LiveIn;
    Live |= LiveOut;
  }
  Live.set(Reg.DefRegion);

  // RP will decrease or stay the same in regions where the register is live
  // (even if the register will only be partially rematerialized it will stop
  // being live till the end of its own region so RP may decrease). In regions
  // where it is used or is not a live-through RP may not decrease.
  Save.inc(DefReg, LaneBitmask::getNone(), Reg.Mask, DAG.MRI);
  for (unsigned I : Live.set_bits()) {
    if (!LiveIn[I] || !LiveOut[I] || Reg.Uses.contains(I))
      ApproximateDiff.set(I);
  }
}

PreRARematStage::FreqInfo::FreqInfo(MachineFunction &MF,
                                    const GCNScheduleDAGMILive &DAG) {
  assert(DAG.MLI && "MLI not defined in DAG");
  MachineBranchProbabilityInfo MBPI;
  MachineBlockFrequencyInfo MBFI(MF, MBPI, *DAG.MLI);

  const unsigned NumRegions = DAG.Regions.size();
  uint64_t MinFreq = MBFI.getEntryFreq().getFrequency();
  Regions.reserve(NumRegions);
  for (unsigned I = 0; I < NumRegions; ++I) {
    MachineBasicBlock *MBB = DAG.Regions[I].first->getParent();
    uint64_t BlockFreq = MBFI.getBlockFreq(MBB).getFrequency();
    Regions.push_back(BlockFreq);
    if (BlockFreq && BlockFreq < MinFreq)
      MinFreq = BlockFreq;
    else if (BlockFreq > MaxFreq)
      MaxFreq = BlockFreq;
  }
  if (!MinFreq)
    return;

  // Normalize to minimum observed frequency to avoid underflows/overflows when
  // combining frequencies.
  for (uint64_t &Freq : Regions)
    Freq /= MinFreq;
  MaxFreq /= MinFreq;
}

PreRARematStage::RematCandidate::RematCandidate(unsigned RootIdx,
                                                const RematDAG &RDAG,
                                                const MachineRegisterInfo &MRI)
    : RootIdx(RootIdx) {
  const RematReg &RootReg = RDAG.getReg(RootIdx);

  for (const auto &[UseRegion, _] : RootReg.Uses) {
    if (UseRegion == RootReg.DefRegion)
      continue;

    // We will rematerialize the register to this using region.
    RematInfo &RI = Remats.try_emplace(UseRegion).first->getSecond();

    // Identify which registers will need to be rematerialized with it.
    SmallVector<const RematReg *, 4> Chain{&RootReg};
    do {
      const RematReg &Reg = *Chain.pop_back_val();
      for (const RematReg::Dependency &Dep : Reg.Dependencies) {
        const RematReg &DepReg = RDAG.getReg(Dep.RegIdx);

        // FIXME: This mimicks the rematerializer's policy to decide what
        // dependencies will be rematerialized as well. The idea is to not
        // account for dependencies that it will not rematerialize in this using
        // region. We should proabably do this the other way around i.e. we
        // should tell the rematerializer what dependencies to bring in the
        // using region and it should just follow our instructions, even if it
        // ends up extending a live range (which we may want to do strategically
        // in the future).
        if (DepReg.Remats.contains(Dep.RegIdx))
          continue;

        if (RI.Dependencies.insert(Dep.RegIdx).second)
          Chain.push_back(&DepReg);
      }
    } while (!Chain.empty());
  }
}

void PreRARematStage::RematCandidate::update(const RegionsInfo &RI,
                                             const FreqInfo &FI,
                                             const PreRARematStage &Stage) {
  const auto [Reg, LiveReg] = Stage.getReg(RootIdx);
  MaxFreq = 0;
  RegionImpact = 0;

  // Accumulate frequencies of defining and using regions. A rematerialization
  // from the least frequent region to the most frequent region will yield the
  // greatest latency penalty and therefore should get minimum score.
  // Reciprocally, a rematerialization in the other direction should get
  // maximum score. Default to values that will yield the worst possible
  // score given known frequencies in order to penalize rematerializations
  // from or into regions whose frequency is unknown.
  FreqDiff = Reg.isFullyRematerializable()
                 ? std::max(FI.Regions[Reg.DefRegion], (uint64_t)1)
                 : 0;

  for (auto &[UseRegion, RI] : make_early_inc_range(Remats)) {
    // It is possible another rematerialization decision has already
    // rematerialized the register in one of the region in which it had used
    // when the candidate was created. We don't need to rematerialize it to this
    // region anymore.
    if (!Reg.Uses.contains(UseRegion)) {
      Remats.erase(UseRegion);
      continue;
    }

    // Account for the register itself being rematerialized to its using region.
    uint64_t RegionFreq = FI.getOrMax(UseRegion);
    FreqDiff -= RegionFreq;

    for (unsigned DepIdx : make_early_inc_range(RI.Dependencies)) {
      const RematReg &DepReg = Stage.RDAG.getReg(DepIdx);

      // The same thing can happen for dependencies; another rematerialization
      // may have made it available in the using region since the candidate was
      // created.
      if (DepReg.Remats.contains(UseRegion)) {
        RI.Dependencies.erase(DepIdx);
        continue;
      }

      // Account for the dependency being rematerialized to the using region.
      FreqDiff -= RegionFreq;
    }

    // Determine maximal local increase in RP this rematerialization can yield.
    RI.RPInc = GCNRegPressure();
    for (const RematReg::Dependency &Dep : Reg.Dependencies) {
      if (RI.Dependencies.contains(Dep.RegIdx))
        RI.RPInc += Stage.RematRegs[Dep.RegIdx].Save;
    }
    for (unsigned DepIdx : RI.Dependencies) {
      GCNRegPressure DepRP;
      for (const RematReg::Dependency &Dep :
           Stage.RDAG.getReg(DepIdx).Dependencies) {
        if (RI.Dependencies.contains(Dep.RegIdx))
          DepRP += Stage.RematRegs[Dep.RegIdx].Save;
      }
      RI.RPInc = max(RI.RPInc, DepRP);
    }
  }

  // Rematerialization can increase RP in regions where dependencies of the root
  // need to be rematerialized even though they have no users of their own.
  // Ensure we don't push RP above target in those regions.
  if (maybeDetrimental(RI, Stage.DAG, Stage.RDAG))
    return;

  for (unsigned RegionIdx : RI.TargetRegions.set_bits()) {
    if (!LiveReg.Live[RegionIdx])
      continue;

    // The rematerialization must contribute positively in at least one register
    // class with usage above the RP target for this region to contribute to the
    // score.
    const GCNRPTarget &RegionTarget = RI.RPTargets[RegionIdx];
    unsigned NetSave = RegionTarget.getTotalNetBeneficialSave(
        GCNRPTarget::RPDiff(LiveReg.Save, /*NegPressure=*/true));
    if (!NetSave)
      continue;

    // Regions in which RP is guaranteed to decrease have more weight.
    RegionImpact += (LiveReg.ApproximateDiff[RegionIdx] ? 2 : 1) * NetSave;

    if (!Stage.TargetOcc) {
      uint64_t RegionFreq = FI.Regions[RegionIdx];
      if (!LiveReg.ApproximateDiff[RegionIdx]) {
        // Apply a frequency penalty in regions in which we are not sure
        // that RP will decrease.
        RegionFreq /= 2;
      }
      MaxFreq = std::max(MaxFreq, RegionFreq);
    }
  }
}

bool PreRARematStage::RematCandidate::maybeDetrimental(
    const RegionsInfo &RI, const GCNScheduleDAGMILive &DAG,
    const RematDAG &RDAG) const {
  const RematReg &Reg = RDAG.getReg(RootIdx);
  for (const auto &[I, Remat] : Remats) {
    if (Remat.Dependencies.empty())
      continue;
    if (auto Uses = Reg.Uses.find(I); Uses != Reg.Uses.end()) {
      // We must not locally increase RP above target in the using region. New
      // instructions will be inserted at the first use, make sure that is a
      // point in the region that can tolerate the RP increase.
      GCNDownwardRPTracker RPTracker(*DAG.LIS);
      RPTracker.advance(DAG.Regions[I].first, Uses->getSecond().FirstMI,
                        &DAG.LiveIns[I]);
      GCNRegPressure ResultRP = RPTracker.getPressure() + Remat.RPInc;
      if (!RI.RPTargets[I].satisfied(ResultRP))
        return false;
    }
  }
  return false;
}

bool PreRARematStage::maybeBeneficial(unsigned RegIdx,
                                      const RegionsInfo &RI) const {
  const auto [Reg, LiveReg] = getReg(RegIdx);
  if (!Reg.isUsefulToRematerialize())
    return false;

  for (unsigned I : RI.TargetRegions.set_bits()) {
    if (!LiveReg.Live[I] || Reg.Remats.contains(I) ||
        !RI.RPTargets[I].isSaveBeneficial(LiveReg.Save))
      continue;

    // Determine the range inside the region where the register is live now but
    // would no longer be live after rematerialization. It must intersect with
    // at least one range where RP is above target for us to consider this
    // rematerialization beneficial.

    if (Reg.DefRegion == I) {
      // In the defining region, rematerializating the register will reduce RP
      // from the point of definition to the end of the region.
      if (RI.TargetLiveOut[I])
        return true;

      // Since RP is above target somewhere in the region but not at the end we
      // expect at least one transition to happen somewhere in the region. The
      // last slot in the region's transitions indicates a transition from above
      // target to below target.
      ArrayRef<SlotIndex> RegionSlots = RI.TargetSlots.at(I);
      SlotIndex DefSlot = DAG.LIS->getInstructionIndex(*Reg.DefMI);
      if (DefSlot <= RegionSlots.back())
        return true;
    } else {
      assert(LiveReg.LiveIn[I] && "expected register to be live-in");

      // If the register is live-through and not used inside the region its
      // rematerialization will reduce RP in the whole region, necessarily
      // crossing a range above target. Since rematerialization makes the
      // register local to a region, it also is guaranteed to reduce pressure at
      // the region live-ins or live-outs.
      if (RI.TargetLiveIn[I])
        return true;
      const auto &Uses = Reg.Uses.find(I);
      if (LiveReg.LiveOut[I] && (RI.TargetLiveOut[I] || Uses == Reg.Uses.end()))
        return true;

      // The register will no longer be live before its first use in the region.
      // We know that RP is below target at the region's live-ins so the first
      // slot in the region's transitions indicates a transition from below
      // target to above target.
      assert(Uses != Reg.Uses.end() && "expected register uses in region");
      const RematReg::RegionUses &RUses = Uses->getSecond();
      ArrayRef<SlotIndex> RegionSlots = RI.TargetSlots.at(I);
      SlotIndex FirstUse = DAG.LIS->getInstructionIndex(*RUses.FirstMI);
      if (RegionSlots.front() < FirstUse)
        return true;
      if (LiveReg.LiveOut[I]) {
        assert(!RI.TargetLiveOut[I] && "should be below target at live-out");

        // When the register was a live-out, rematerializing it will reduce RP
        // between its last use in the region and the end of the region. We know
        // that RP is below target at the region's live-outs so the first
        // last slot in the region's transitions indicates a transition from
        // above target to below target.
        SlotIndex LastUse = DAG.LIS->getInstructionIndex(*RUses.LastMI);
        if (LastUse <= RegionSlots.back())
          return true;
      }
    }
  }
  return false;
}

void PreRARematStage::removeFromLiveMaps(unsigned RegIdx) {
  const RegLiveness &LiveReg = RematRegs[RegIdx];
  Register DefReg = RDAG.getDefReg(RegIdx);
  for (unsigned I : LiveReg.LiveIn.set_bits())
    DAG.LiveIns[I].erase(DefReg);
  for (unsigned I : LiveReg.LiveOut.set_bits())
    DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).erase(DefReg);
}

void PreRARematStage::addToLiveMaps(unsigned RegIdx) {
  const auto [Reg, LiveReg] = getReg(RegIdx);
  std::pair<Register, LaneBitmask> RegAndMask(Reg.getDefReg(), Reg.Mask);
  for (unsigned I : LiveReg.LiveIn.set_bits())
    DAG.LiveIns[I].insert(RegAndMask);
  for (unsigned I : LiveReg.LiveOut.set_bits())
    DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).insert(RegAndMask);
}

bool PreRARematStage::buildChainAtReg(
    unsigned RegIdx, const RegionsInfo &RI,
    SmallVectorImpl<RematCandidate> &Candidates) {
  auto [Reg, LiveReg] = getReg(RegIdx);
  assert(Reg.DefMI && "dead register");
  if (LiveReg.Status != RegLiveness::INVALID)
    return false;

  // All dependencies must be either rematerialized or needing to be
  // rematerialized with a parent to be useful.
  for (const RematReg::Dependency &Dep : Reg.Dependencies) {
    const auto [_, DepLiveReg] = getReg(Dep.RegIdx);
    if (DepLiveReg.Status == RegLiveness::INVALID ||
        DepLiveReg.Status == RegLiveness::ROOT)
      return false;
  }

  if (maybeBeneficial(RegIdx, RI)) {
    LiveReg.Status = RegLiveness::ROOT;
    Candidates.emplace_back(RegIdx, RDAG, DAG.MRI);
    REMAT_DEBUG(dbgs() << "** ADD CANDIDATE " << printReg(RegIdx));
    return false;
  }
  REMAT_DEBUG(dbgs() << "** UPGRADE TO NON-ROOT " << printReg(RegIdx));
  LiveReg.Status = RegLiveness::NON_ROOT;
  return true;
}

template <typename T>
static std::function<bool(unsigned, unsigned)>
sortFromArray(ArrayRef<T> Candidates) {
  return [Candidates](unsigned LHS, unsigned RHS) {
    return Candidates[LHS] < Candidates[RHS];
  };
}

bool PreRARematStage::performCrossRegionRemats(RegionsInfo &RI) {
  SmallVector<RematCandidate, 16> Candidates;

  // Create liveness information for rematerializable registers, and identify an
  // initial set of candidates for rematerialization. Registers are initially in
  // def-use order so we can determine the first round of valid candidates in
  // one pass.
  RematRegs.reserve(RDAG.getNumRegs());
  for (const auto &[RegIdx, Reg] : enumerate(RDAG.getRegs())) {
    RematRegs.emplace_back(RegIdx, DAG, RDAG);
    REMAT_DEBUG(dbgs() << RDAG.printID(RegIdx) << " " << Reg.print() << '\n');
    buildChainAtReg(RegIdx, RI, Candidates);
  }

#ifndef NDEBUG
  unsigned RoundNum = 0;
#endif

  FreqInfo Freq(MF, DAG);

  // Rematerialize registers in successive rounds until all RP targets are
  // satisifed or until we run out of rematerialization candidates.
  while (!Candidates.empty()) {
    bool TargetsUnsat;
    BitVector RecomputeRP(DAG.Regions.size());
    unsigned NumNewRegs = 0;

    // This stores indices to candidates in the vector of current candidates.
    // This makes the element copies induced by sorting cheaper.
    SmallVector<unsigned, 16> CandidateOrder;
    CandidateOrder.reserve(Candidates.size());
    for (unsigned I = 0, E = Candidates.size(); I < E; ++I)
      CandidateOrder.push_back(I);

    do {
      // (Re-)Score and (re-)sort all remaining candidates.
      for (unsigned CandIdx : CandidateOrder)
        Candidates[CandIdx].update(RI, Freq, *this);
      sort(CandidateOrder, sortFromArray(ArrayRef<RematCandidate>(Candidates)));

      REMAT_DEBUG({
        dbgs() << "==== ROUND " << RoundNum << " ====\n"
               << REMAT_PREFIX
               << "Current candidates, in rematerialization order:\n";
        for (unsigned CandIdx : reverse(CandidateOrder)) {
          const RematCandidate &Cand = Candidates[CandIdx];
          dbgs() << REMAT_PREFIX << "  " << Cand.print() << " | "
                 << printChain(Cand.RootIdx) << ' '
                 << RDAG.getReg(Cand.RootIdx).print() << '\n';
        }
        dbgs() << RI.print();
      });

      // Rematerialize candidates in decreasing score order until we estimate
      // that all RP targets are satisfied or until rematerialization candidates
      // are no longer useful to decrease RP.
      bool NoRemat = true;
      do {
        const unsigned CandIdx = CandidateOrder.pop_back_val();
        const RematCandidate &Cand = Candidates[CandIdx];
        // It is possible that candidates become useslss as we re-score the same
        // candidates in successive rounds. In that case all remaining
        // candidates are useless and we can try identifying more.
        if (Cand.hasNullScore())
          break;
        NoRemat = false;

        const unsigned RegIdx = Cand.RootIdx;
        // Rematerialization can increase RP in regions where dependencies of
        // the root need to be rematerialized even though they have no users of
        // their own. Ensure we don't push RP above target in those regions.
        // if (Cand.maybeDetrimental(RI, DAG, RDAG))
        //   continue;

        // When previous rematerializations in this round have already satisfied
        // RP targets in all regions this rematerialization can impact, we have
        // a good indication that our scores have diverged significantly from
        // reality, in which case we interrupt this round and re-score. This
        // also ensures that every rematerialization we perform is possibly
        // impactful in at least one target region.
        if (!maybeBeneficial(RegIdx, RI))
          continue;

        // Rematerialize the register and all its unrematerialized dependencies.
        NumNewRegs += rematerialize(RegIdx);

        const RegLiveness &LiveReg = RematRegs[RegIdx];
        RecomputeRP |= LiveReg.ApproximateDiff;
        RescheduleRegions |= LiveReg.Live;
        for (unsigned I : LiveReg.Live.set_bits())
          RI.RPTargets[I].saveRP(LiveReg.Save);
        RI.clearIfSatisfied(LiveReg.Live);
      } while (!CandidateOrder.empty() && RI.TargetRegions.any());
#ifndef NDEBUG
      ++RoundNum;
#endif
      REMAT_DEBUG({
        if (!RI.TargetRegions.any()) {
          dbgs() << "** Interrupt round on all targets achieved\n";
        } else if (CandidateOrder.empty()) {
          dbgs() << "** Stop on exhausted rematerialization candidates\n";
        } else {
          dbgs() << "** Interrupt round on null score for "
                 << printChain(Candidates[CandidateOrder.back()].RootIdx)
                 << '\n';
        }
      });
      if (NoRemat)
        break;

      // Commit changes to the live intervals before verifying RP in regions
      // affected unpredictably.
      RDAG.updateLiveIntervals();
      TargetsUnsat =
          RI.updateAndVerify(RecomputeRP, DAG) || RI.TargetRegions.any();
    } while (TargetsUnsat && !CandidateOrder.empty());

    if (!TargetsUnsat)
      break;

    SmallVector<RematCandidate, 16> NewCandidates;
    std::function<void(unsigned)> BuildAtRegUsers = [&](unsigned RegIdx) {
      const auto &[Reg, _] = getReg(RegIdx);
      for (const auto &[_, RegionUses] : Reg.Uses) {
        for (const MachineInstr *UseMI : RegionUses.Users) {
          unsigned UserIdx = RDAG.getRematRegIdx(*UseMI);
          if (UserIdx != RDAG.getNumRegs())
            if (buildChainAtReg(UserIdx, RI, NewCandidates))
              BuildAtRegUsers(UserIdx);
        }
      }
    };

    // New registers may allow other rematerializations to proceed in their
    // defining regions.
    const unsigned NumRegs = RematRegs.size();
    for (unsigned RegIdx = NumRegs - NumNewRegs; RegIdx < NumRegs; ++RegIdx) {
      if (buildChainAtReg(RegIdx, RI, NewCandidates))
        BuildAtRegUsers(RegIdx);
    }

    // All the current unrematerialized candidates are useless as is. They might
    // become useful if rematerialized with their parent(s).
    for (const RematCandidate &Cand : Candidates) {
      const auto &[Reg, LiveReg] = getReg(Cand.RootIdx);
      if (LiveReg.Status == RegLiveness::ROOT) {
        // This register is useless on its own and can only be rematerialized
        // with its parent(s). This may allow other rematerializations in its
        // defining regions to proceed.
        REMAT_DEBUG(dbgs() << "** DOWNGRADE TO NON-ROOT "
                           << printChain(Cand.RootIdx) << '\n');
        LiveReg.Status = RegLiveness::NON_ROOT;
        BuildAtRegUsers(Cand.RootIdx);
      } else {
        // This register was useful to rematerialize on its own, allowing any
        // remaining users in its original defining region to be rematerialized
        // now. If the register was fully rematerialized it has no remaining
        // user in its defining region by construction.
        if (Reg.DefMI)
          BuildAtRegUsers(Cand.RootIdx);
      }
    }
    Candidates = std::move(NewCandidates);
  }

  return RI.TargetRegions.none();
}

PreRARematStage::SameRegionCand::SameRegionCand(unsigned RegIdx,
                                                const PreRARematStage &Stage)
    : RegIdx(RegIdx) {
  const RematReg &Reg = Stage.RDAG.getReg(RegIdx);
  RP.inc(Reg.getDefReg(), LaneBitmask::getNone(), Reg.Mask, Stage.DAG.MRI);
  NumRegs = RP.getSGPRNum() + RP.getVGPRNum(Stage.ST.hasGFX90AInsts());
}

bool PreRARematStage::performSameRegionRemats(RegionsInfo &RI) {

  REMAT_DEBUG(dbgs() << "Trying same-region rematerialization\n");

  // For each remaining target region, we identify slot ranges in which RP is
  // higher than the target. This will inform which rematerializations are
  // likely to be useful to reduce RP. For each slot vector, the first element
  // indicates the beginning of a range above the target, the second the end of
  // that range, the third the beginning of the next range above the target,
  // etc. The parity of the number of elements indicates whether RP is above
  // target at the region live-outs. Slots at even positions map to the first MI
  // of each range above RP. Slots at odd positions map to the last MI of each
  // range above RP. The last slot is the region upper boundary.
  DenseMap<unsigned, SmallVector<SlotIndex, 4>> RegionInfo;
  for (unsigned RegionIdx : RI.TargetRegions.set_bits()) {
    const auto &[RegionBegin, RegionEnd] = DAG.Regions[RegionIdx];

    // We cannot do anything if the region is empty or has no non-debug MIs.
    if (RegionBegin == RegionEnd)
      continue;
    GCNDownwardRPTracker RPTracker(*DAG.LIS);
    if (!RPTracker.reset(*RegionBegin, &DAG.LiveIns[RegionIdx]))
      continue;

    SmallVectorImpl<SlotIndex> &Slots = RegionInfo[RegionIdx];
    const GCNRPTarget &Target = RI.RPTargets[RegionIdx];
    bool WasAboveTarget = Target.satisfied(RPTracker.getPressure());
    if (WasAboveTarget)
      Slots.push_back(DAG.LIS->getInstructionIndex(*RegionBegin));
    while (RPTracker.advance()) {
      if (Target.satisfied(RPTracker.getPressure())) {
        if (WasAboveTarget) {
          const MachineInstr &MI = *RPTracker.getLastTrackedMI();
          Slots.push_back(DAG.LIS->getInstructionIndex(MI));
          WasAboveTarget = false;
        }
      } else if (!WasAboveTarget) {
        const MachineInstr &MI = *RPTracker.getLastTrackedMI();
        Slots.push_back(DAG.LIS->getInstructionIndex(MI));
        WasAboveTarget = true;
      }
    }
    assert(!Slots.empty() && "no range above target");

    // To simplify later processing add the region boundary slot at the end.
    if (RegionEnd == RegionBegin->getParent()->end())
      Slots.push_back(DAG.LIS->getMBBEndIdx(RegionBegin->getParent()));
    else
      Slots.push_back(DAG.LIS->getInstructionIndex(*RegionEnd));

    REMAT_DEBUG({
      dbgs() << '[' << RegionIdx << "] ("
             << DAG.LIS->getInstructionIndex(*RegionBegin) << ") -> ";
      for (SlotIndex Delim : drop_end(Slots))
        dbgs() << Delim << " -> ";
      dbgs() << '(' << Slots.back() << ")\n";
    });
  }

  // Collect candidates in target regions.
  DenseMap<unsigned, SmallVector<SameRegionCand>> Candidates;
  for (unsigned RegIdx = 0, E = RematRegs.size(); RegIdx < E; ++RegIdx) {
    const RematReg &Reg = RDAG.getReg(RegIdx);
    if (!Reg.DefMI || Reg.DefMI->mayLoad() ||
        !RI.TargetRegions[Reg.DefRegion] || Reg.Uses.size() != 1)
      continue;
    const auto &[Region, DefUses] = *Reg.Uses.begin();
    if (Region != Reg.DefRegion || DefUses.Users.size() < 2)
      continue;

    auto RegionInfoIt = RegionInfo.find(Reg.DefRegion);
    if (RegionInfoIt == RegionInfo.end())
      continue;
    ArrayRef<SlotIndex> DelimSlots = RegionInfoIt->getSecond();

    SmallVectorImpl<SameRegionCand> &RegionCands = Candidates[Reg.DefRegion];
    SameRegionCand &Cand = RegionCands.emplace_back(RegIdx, *this);

    // We don't want to rematerialize or extend the live range of the register's
    // dependencies so we cannot rematerialize the register further than the end
    // of the live range of any of its dependencies.
    SlotIndex DefSlot =
        DAG.LIS->getInstructionIndex(*Reg.DefMI).getRegSlot(true);
    SlotIndex RematBoundary = DelimSlots.back();
    for (const RematReg::Dependency &Dep : Reg.Dependencies) {
      Register DepReg = Reg.DefMI->getOperand(Dep.MOIdx).getReg();
      const LiveInterval &LI = DAG.LIS->getInterval(DepReg);
      const LiveRange::Segment *RegSegment = LI.getSegmentContaining(DefSlot);
      if (RegSegment->end < RematBoundary) {
        RematBoundary = RegSegment->end;
        Cand.InsertPos = DAG.LIS->getInstructionFromIndex(RegSegment->end);
        assert(Cand.InsertPos != nullptr && "not an MI");
      }
    }
    if (RematBoundary <= DAG.LIS->getInstructionIndex(*DefUses.FirstMI)) {
      RegionCands.pop_back();
      continue;
    }

    REMAT_DEBUG(dbgs() << RDAG.printID(RegIdx)
                       << " may qualify for same-region rematerialization\n");

    // Separate users that can be rematerialized to directly from those that are
    // located after the register's dependencies have stopped being available.
    SmallVector<std::pair<SlotIndex, MachineInstr *>, 4> Users;
    for (MachineInstr *UseMI : DefUses.Users) {
      SlotIndex UseSlot = DAG.LIS->getInstructionIndex(*UseMI);
      if (UseSlot < RematBoundary)
        Users.emplace_back(UseSlot, UseMI);
      else
        Cand.RemainingUsers.push_back(UseMI);
    }
    if (Users.empty()) {
      RegionCands.pop_back();
      continue;
    }

    // Sort users for which we can rematerialize the register in instruction
    // order.
    sort(Users, [](std::pair<SlotIndex, MachineInstr *> &LHS,
                   std::pair<SlotIndex, MachineInstr *> &RHS) {
      return LHS.first < RHS.first;
    });

    // Find in which range the first register's use is located.
    SlotIndex FirstUseSlot = Users.front().first;
    auto *It = lower_bound(DelimSlots, FirstUseSlot);
    assert(It != DelimSlots.end() && "user out of region");
    // RangeIdx smallest integer s.t. FirstUseSlot >= DelimSlots[RangeIdx].
    unsigned RangeIdx = std::distance(DelimSlots.begin(), It);
    bool RangeAboveTarget =
        !(RangeIdx & 1) || DelimSlots[RangeIdx - 1] == FirstUseSlot;

    // Rematerializing to this user is only useful if it kills its live-range
    // over at least one part of a range above RP. Skip the first user which
    // will continue to use the original register.
    for (const auto &[UseSlot, UseMI] : llvm::drop_begin(Users)) {
      auto *It = lower_bound(DelimSlots, UseSlot);
      unsigned UseRangeIdx = std::distance(DelimSlots.begin(), It);
      if (UseRangeIdx == RangeIdx && !RangeAboveTarget) {
        // User will use the previous rematerialization. If there are no
        // rematerialization groups the user will continue using the existing
        // register.
        if (!Cand.RematGroups.empty()) {
          Cand.RematGroups.back().push_back(UseMI);
        }
      } else {
        // User will use a new rematerialization of the register.
        Cand.RematGroups.emplace_back().push_back(UseMI);
      }
      if (UseRangeIdx != RangeIdx) {
        RangeIdx = UseRangeIdx;
        RangeAboveTarget =
            !(RangeIdx & 1) || DelimSlots[RangeIdx - 1] == UseSlot;
      }
    }

    // No point in rematerializing if we couldn't form useful rematerialization
    // groups.
    if (Cand.RematGroups.empty())
      RegionCands.pop_back();
  }

  // In each regions where we found valid candidates, rematerialize registers
  // until reaching the RP target or running out of candidates.
  for (const auto &[RegionIdx, RegionCands] : Candidates) {
    if (RegionCands.empty())
      continue;

    REMAT_DEBUG(dbgs() << "[" << RegionIdx
                       << "] Rematerializing intra-region\n");

    // Order candidate by their impact
    SmallVector<unsigned, 16> CandidateOrder;
    CandidateOrder.reserve(RegionCands.size());
    for (unsigned I = 0, E = RegionCands.size(); I < E; ++I)
      CandidateOrder.push_back(I);
    sort(CandidateOrder, sortFromArray(ArrayRef<SameRegionCand>(RegionCands)));

    GCNRPTarget &TargetRegion = RI.RPTargets[RegionIdx];
    while (!CandidateOrder.empty()) {
      const SameRegionCand &Cand = RegionCands[CandidateOrder.pop_back_val()];
      for (ArrayRef<MachineInstr *> Users : Cand.RematGroups) {
        REMAT_DEBUG(dbgs() << "  Rematerializing " << RDAG.printID(Cand.RegIdx)
                           << " to " << RDAG.printUser(Users.front()) << '\n');
        RDAG.rematerializeToUses(Cand.RegIdx, Users.front(), Users, {});
      }
      if (!Cand.RemainingUsers.empty()) {
        REMAT_DEBUG(dbgs() << "  Rematerializing " << RDAG.printID(Cand.RegIdx)
                           << " to " << RDAG.printUser(&*Cand.InsertPos)
                           << '\n');
        RDAG.rematerializeToUses(Cand.RegIdx, Cand.InsertPos,
                                 Cand.RemainingUsers, {});
      }
      TargetRegion.saveRP(Cand.RP);
      if (TargetRegion.satisfied()) {
        RDAG.updateLiveIntervals();
        TargetRegion.setRP(DAG.getRealRegPressure(RegionIdx));
        if (TargetRegion.satisfied()) {
          REMAT_DEBUG(dbgs() << "  [" << RegionIdx << "] Target reached!\n");
          break;
        }
      }
    }
    RDAG.updateLiveIntervals();
    RescheduleRegions.set(RegionIdx);
  }

  return RI.TargetRegions.none();
}

unsigned PreRARematStage::rematerialize(unsigned RootIdx) {
  unsigned NumNewRegs = RDAG.rematerialize(RootIdx);
  REMAT_DEBUG(dbgs() << "  ** REMAT (" << RootIdx << ") added "
                     << RDAG.getNumRegs() - RematRegs.size()
                     << " new register(s)\n");

  // Every rematerialization we do is likely to add more instructions overall
  // and move instructions to higher frequency regions, increasing the total sum
  // latency of computing the register. This is acceptable if we are eliminating
  // spills in the process, but when the goal is increasing occupancy we get
  // nothing out of rematerialization if occupancy is not increased in the end;
  // in such cases we want to roll back the rematerialization decision.
  if (TargetOcc && !DisablePreRARematRollback)
    Rollbacks.push_back(RootIdx);

  // Create corresponding live information for new register. By construction
  // these are only live within their defining region so there is no need to add
  // them to live-in/live-out maps.
  unsigned I = RematRegs.size(), E = RDAG.getNumRegs();
  RematRegs.reserve(E);
  for (; I < E; ++I) {
    const RematReg &Reg = RDAG.getReg(I);
    const RematReg &ParentReg = RDAG.getReg(*Reg.Parent);

    // If the parent is no longer considered useful to rematerialize then it
    // means it is no longer needed as a live-in/live-out anywhere. If it is
    // dead it is considered rematerialized.
    if (!ParentReg.isUsefulToRematerialize())
      removeFromLiveMaps(*Reg.Parent);

    RematRegs.emplace_back(I, DAG, RDAG, /*SkipLiveCheck=*/true);
    REMAT_DEBUG(dbgs() << "    " << RDAG.printID(I) << " "
                       << RDAG.getReg(I).print() << '\n');
  }
  return NumNewRegs;
}

void PreRARematStage::finalizeGCNSchedStage() {
  // We consider that reducing spilling is always beneficial so we never
  // rollback rematerializations in such cases. It's also possible that
  // rescheduling lowers occupancy over the one achieved just through remats,
  // in which case we do not want to rollback either (the rescheduling was
  // already reverted in PreRARematStage::shouldRevertScheduling in such
  // cases).
  unsigned MaxOcc = std::max(AchievedOcc, DAG.MinOccupancy);
  if (!TargetOcc || MaxOcc >= *TargetOcc)
    return;

  // Rollback, then recompute pressure in all affected regions.
  REMAT_DEBUG(dbgs() << "==== ROLLBACK ====\n");
  BitVector RecomputeRP(DAG.Regions.size());
  for (unsigned RegIdx : reverse(Rollbacks)) {
    REMAT_DEBUG(dbgs() << "** ROLLBACK " << RDAG.printID(RegIdx) << '\n');
    RDAG.rollback(RegIdx);
    addToLiveMaps(RegIdx);
    RecomputeRP |= RematRegs[RegIdx].Live;
  }
  RDAG.updateLiveIntervals();
  for (unsigned I : RecomputeRP.set_bits())
    DAG.Pressure[I] = DAG.getRealRegPressure(I);

  GCNSchedStage::finalizeGCNSchedStage();
}

static bool hasIGLPInstrs(ScheduleDAGInstrs *DAG) {
  const SIInstrInfo *SII = static_cast<const SIInstrInfo *>(DAG->TII);
  return any_of(*DAG, [SII](MachineBasicBlock::iterator MI) {
    return SII->isIGLPMutationOnly(MI->getOpcode());
  });
}

GCNPostScheduleDAGMILive::GCNPostScheduleDAGMILive(
    MachineSchedContext *C, std::unique_ptr<MachineSchedStrategy> S,
    bool RemoveKillFlags)
    : ScheduleDAGMI(C, std::move(S), RemoveKillFlags) {}

void GCNPostScheduleDAGMILive::schedule() {
  HasIGLPInstrs = hasIGLPInstrs(this);
  if (HasIGLPInstrs) {
    SavedMutations.clear();
    SavedMutations.swap(Mutations);
    addMutation(createIGroupLPDAGMutation(AMDGPU::SchedulingPhase::PostRA));
  }

  ScheduleDAGMI::schedule();
}

void GCNPostScheduleDAGMILive::finalizeSchedule() {
  if (HasIGLPInstrs)
    SavedMutations.swap(Mutations);

  ScheduleDAGMI::finalizeSchedule();
}
