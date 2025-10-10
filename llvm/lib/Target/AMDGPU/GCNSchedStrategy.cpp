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
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>
#include <limits>
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
void PreRARematStage::printTargetRegions(bool PrintAll) const {
  if (PrintAll) {
    for (auto [I, Target] : enumerate(RPTargets))
      dbgs() << REMAT_PREFIX << "  [" << I << "] " << Target << '\n';
    return;
  }
  if (TargetRegions.none()) {
    dbgs() << REMAT_PREFIX << "No target regions\n";
    return;
  }
  dbgs() << REMAT_PREFIX << "Target regions:\n";
  for (unsigned I : TargetRegions.set_bits())
    dbgs() << REMAT_PREFIX << "  [" << I << "] " << RPTargets[I] << '\n';
}

void PreRARematStage::RematChain::print(ArrayRef<RematReg> RematRegs) const {
  SmallDenseSet<const RematReg *, 4> Visited;
  std::function<void(const RematReg &, unsigned)> PrintRemat =
      [&](const RematReg &Reg, unsigned Depth) -> void {
    if (!Visited.insert(&Reg).second)
      return;
    for (const RematReg::Dependency &Dep : Reg.Dependencies) {
      if (!Dep.isRematInChain())
        continue;
      PrintRemat(RematRegs[*Dep.RematIdx], Depth + 1);
    }
    std::string Shift(2 * Depth, ' ');
    std::string Sep = Depth ? " | " : "-> ";
    dbgs() << REMAT_PREFIX << "    " << Shift << Sep << *Reg.DefMI;
  };
  dbgs() << REMAT_PREFIX << "  Chain in [" << root().DefRegion << "] with "
         << size() << " registers\n";
  PrintRemat(root(), 0);
}

void PreRARematStage::RematChain::printScore() const {
  ScoreTy ShiftScore = Score;
  ScoreTy RegionImpact = ShiftScore & ((1 << RegionImpactWidth) - 1);
  ShiftScore >>= RegionImpactWidth;
  ScoreTy FreqDiff = ShiftScore & ((1 << FreqDiffWidth) - 1);
  ShiftScore >>= FreqDiffWidth;
  ScoreTy MaxFreq = ShiftScore;
  dbgs() << REMAT_PREFIX << '(' << MaxFreq << ", " << FreqDiff << ", "
         << RegionImpact << ") -> " << *root().DefMI;
}

void PreRARematStage::RollbackInfo::print() const {
  /// TODO: this
  // for (const auto &[UseRegion, RematMI] : RegionRemats)
  //   dbgs() << REMAT_PREFIX << "  [... -> " << UseRegion << "] " << *RematMI;
}
#endif

bool PreRARematStage::initGCNSchedStage() {
  // FIXME: This pass will invalidate cached BBLiveInMap and MBBLiveIns for
  // regions inbetween the defs and region we sinked the def to. Will need to be
  // fixed if there is another pass after this pass.
  assert(!S.hasNextStage());

  if (!GCNSchedStage::initGCNSchedStage() || DAG.Regions.size() <= 1)
    return false;

  // Maps all MIs (except lone terminators, which are not part of any region) to
  // their parent region. Non-lone terminators are considered part of the region
  // they delimitate.
  DenseMap<MachineInstr *, unsigned> MIRegion(MF.getInstructionCount());

  // Before performing any IR modification record the parent region of each MI
  // and the parent MBB of each region.
  const unsigned NumRegions = DAG.Regions.size();
  for (unsigned I = 0; I < NumRegions; ++I) {
    RegionBoundaries Region = DAG.Regions[I];
    for (auto MI = Region.first; MI != Region.second; ++MI)
      MIRegion.insert({&*MI, I});
    MachineBasicBlock *ParentMBB = Region.first->getParent();
    if (Region.second != ParentMBB->end())
      MIRegion.insert({&*Region.second, I});
    RegionBB.push_back(ParentMBB);
  }

  // Set an objective for the stage based on current RP in each region.
  REMAT_DEBUG({
    dbgs() << "Analyzing ";
    MF.getFunction().printAsOperand(dbgs(), false);
    dbgs() << ": ";
  });
  if (!setObjective()) {
    LLVM_DEBUG(dbgs() << "no objective to achieve, occupancy is maximal at "
                      << MFI.getMaxWavesPerEU() << '\n');
    return false;
  }
  LLVM_DEBUG({
    if (TargetOcc) {
      dbgs() << "increase occupancy from " << *TargetOcc - 1 << '\n';
    } else {
      dbgs() << "reduce spilling (minimum target occupancy is "
             << MFI.getMinWavesPerEU() << ")\n";
    }
    printTargetRegions(/*PrintAll=*/TargetRegions.none());
  });

  if (!buildRematGraph(MIRegion)) {
    REMAT_DEBUG(dbgs() << "Nothing to rematerialize\n");
    return false;
  }

  // Compute frequency information.
  const RematChain::FreqInfo FreqInfo(MF, DAG);
  REMAT_DEBUG({
    dbgs() << "Region frequencies\n";
    for (auto [I, Freq] : enumerate(FreqInfo.Regions)) {
      dbgs() << REMAT_PREFIX << "  [" << I << "] ";
      if (Freq)
        dbgs() << Freq;
      else
        dbgs() << "unknown ";
      dbgs() << " -> start @ " << *DAG.Regions[I].first;
    }
  });

  // Create chains.
  SmallVector<RematChain> RematChains;
  RematChains.reserve(ChainRoots.count());
  for (unsigned I : ChainRoots.set_bits())
    RematChains.emplace_back(I, RematRegs, FreqInfo, DAG);
  REMAT_DEBUG({
    dbgs() << "Rematerializable chains:\n";
    for (const RematChain &Chain : RematChains)
      Chain.print(RematRegs);
  });

// Rematerialize registers in successive rounds until all RP targets are
// satisifed or until we run out of rematerialization candidates.
#ifndef NDEBUG
  unsigned RoundNum = 0;
#endif
  BitVector RecomputeRP(NumRegions);
  do {
    // (Re-)Score and (re-)sort all chains in increasing score order.
    for (RematChain &Chain : RematChains)
      Chain.updateScore(TargetRegions, RPTargets, FreqInfo, !TargetOcc);
    sort(RematChains);

    REMAT_DEBUG({
      dbgs() << "==== ROUND " << RoundNum << " ====\n";
      for (const RematChain &Chain : RematChains)
        Chain.printScore();
      printTargetRegions();
    });

    RecomputeRP.reset();
    int RematIdx = RematChains.size() - 1;

    // Rematerialize chains in decreasing score order until we estimate that all
    // RP targets are satisfied or until rematerialization candidates are no
    // longer useful to decrease RP.
    for (; RematIdx >= 0 && TargetRegions.any(); --RematIdx) {
      const RematChain &Chain = RematChains[RematIdx];
      // FIXME: I think we now need to continue rounds in such cases because of
      // new register increment case.
      if (RematChains[RematIdx].hasNullScore()) {
        REMAT_DEBUG(dbgs() << "*** Stop on null score | "
                           << *Chain.root().DefMI);
        break;
      }

      // When previous rematerializations in this round have already satisfied
      // RP targets in all regions this rematerialization can impact, we have a
      // good indication that our scores have diverged significantly from
      // reality, in which case we interrupt this round and re-score. This also
      // ensures that every rematerialization we perform is possibly impactful
      // in at least one target region.
      if (!Chain.maybeBeneficial(TargetRegions, RPTargets)) {
        REMAT_DEBUG(dbgs() << "*** Stop round on stale score | "
                           << *Chain.root().DefMI);
        break;
      }

      REMAT_DEBUG(dbgs() << "*** REMAT [ ... -> ...] | "
                         << *Chain.root().DefMI);
      // Every rematerialization we do here is likely to move the instruction
      // into a higher frequency region, increasing the total sum latency of the
      // instruction itself. This is acceptable if we are eliminating a spill in
      // the process, but when the goal is increasing occupancy we get nothing
      // out of rematerialization if occupancy is not increased in the end; in
      // such cases we want to roll back the rematerialization.
      if (TargetOcc) {
        RollbackInfo &Rollback = Rollbacks.emplace_back();
        rematerialize(Chain, RecomputeRP, Rollback);
        LLVM_DEBUG(Rollback.print());
      } else {
        RollbackInfo Rollback;
        rematerialize(Chain, RecomputeRP, Rollback);
        LLVM_DEBUG(Rollback.print());
      }
      unsetSatisifedRPTargets(Chain.Live);
    }
    if (RematIdx == static_cast<int>(RematChains.size()) - 1)
      RematIdx = -1;

#ifndef NDEBUG
    ++RoundNum;
#endif
    REMAT_DEBUG({
      if (!TargetRegions.any())
        dbgs() << "*** Stop round on all targets achieved\n";
      else if (RematIdx == -1)
        dbgs() << "*** Stop on exhausted remat opportunities\n";
    });

    // Peel off registers we already rematerialized from the vector's tail.
    RematChains.truncate(RematIdx + 1);
  } while ((updateAndVerifyRPTargets(RecomputeRP) || TargetRegions.any()) &&
           !RematChains.empty());
  if (RescheduleRegions.none())
    return false;

  // Commit all pressure changes to the DAG and compute minimum achieved
  // occupancy in impacted regions.
  REMAT_DEBUG(dbgs() << "==== REMAT RESULTS ====\n");
  unsigned DynamicVGPRBlockSize = MFI.getDynamicVGPRBlockSize();
  AchievedOcc = MFI.getMaxWavesPerEU();
  for (unsigned I : RescheduleRegions.set_bits()) {
    const GCNRegPressure &RP = RPTargets[I].getCurrentRP();
    DAG.Pressure[I] = RP;
    unsigned NewRegionOcc = RP.getOccupancy(ST, DynamicVGPRBlockSize);
    AchievedOcc = std::min(AchievedOcc, NewRegionOcc);
    REMAT_DEBUG(dbgs() << '[' << I << "] Achieved occupancy " << NewRegionOcc
                       << " (" << RPTargets[I] << ")\n");
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
  // Only reschedule regions that have excess register pressure (i.e. spilling)
  // or had minimum occupancy at the beginning of the stage (as long as
  // rescheduling of previous regions did not make occupancy drop back down to
  // the initial minimum).
  unsigned DynamicVGPRBlockSize = DAG.MFI.getDynamicVGPRBlockSize();
  if (!DAG.RegionsWithExcessRP[RegionIdx] &&
      (DAG.MinOccupancy <= InitialOccupancy ||
       DAG.Pressure[RegionIdx].getOccupancy(ST, DynamicVGPRBlockSize) !=
           InitialOccupancy))
    return false;

  return GCNSchedStage::initGCNRegion();
}

bool ClusteredLowOccStage::initGCNRegion() {
  // We may need to reschedule this region if it wasn't rescheduled in the last
  // stage, or if we found it was testing critical register pressure limits in
  // the unclustered reschedule stage. The later is because we may not have been
  // able to raise the min occupancy in the previous stage so the region may be
  // overly constrained even if it was already rescheduled.
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

  // We may not be able to keep the current target occupancy because of the just
  // scheduled region. We might still be able to revert scheduling if the
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

  // After reverting schedule, debug instrs will now be at the end of the block
  // and RegionEnd will point to the first debug instr. Increment RegionEnd
  // pass debug instrs to the actual end of the scheduling region.
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

bool PreRARematStage::setObjective() {
  const Function &F = MF.getFunction();

  // Set up "spilling targets" for all regions.
  unsigned MaxSGPRs = ST.getMaxNumSGPRs(F);
  unsigned MaxVGPRs = ST.getMaxNumVGPRs(F);
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    const GCNRegPressure &RP = DAG.Pressure[I];
    GCNRPTarget &Target = RPTargets.emplace_back(MaxSGPRs, MaxVGPRs, MF, RP);
    if (!Target.satisfied())
      TargetRegions.set(I);
  }

  if (TargetRegions.any() || DAG.MinOccupancy >= MFI.getMaxWavesPerEU()) {
    // In addition to register usage being above addressable limits, occupancy
    // below the minimum is considered like "spilling" as well.
    TargetOcc = std::nullopt;
  } else {
    // There is no spilling and room to improve occupancy; set up "increased
    // occupancy targets" for all regions.
    TargetOcc = DAG.MinOccupancy + 1;
    const unsigned VGPRBlockSize = MFI.getDynamicVGPRBlockSize();
    MaxSGPRs = ST.getMaxNumSGPRs(*TargetOcc, false);
    MaxVGPRs = ST.getMaxNumVGPRs(*TargetOcc, VGPRBlockSize);
    for (auto [I, Target] : enumerate(RPTargets)) {
      Target.setTarget(MaxSGPRs, MaxVGPRs);
      if (!Target.satisfied())
        TargetRegions.set(I);
    }
  }

  return TargetRegions.any();
}

static bool isAvailableAtUse(const VNInfo *OVNI, LaneBitmask Mask,
                             SlotIndex UseIdx, const LiveInterval &LI) {
  assert(OVNI);
  if (OVNI != LI.getVNInfoAt(UseIdx))
    return false;

  // Check that subrange is live at user.
  if (LI.hasSubRanges()) {
    for (const LiveInterval::SubRange &SR : LI.subranges()) {
      if ((SR.LaneMask & Mask).none())
        continue;
      if (!SR.liveAt(UseIdx))
        return false;

      // Early exit if all used lanes are checked. No need to continue.
      Mask &= ~SR.LaneMask;
      if (Mask.none())
        break;
    }
  }
  return true;
}

bool PreRARematStage::buildRematGraph(
    const DenseMap<MachineInstr *, unsigned> &MIRegion) {
  // We need up-to-date live-out info. to query live-out register masks in
  // regions containing rematerializable registers.
  DAG.RegionLiveOuts.buildLiveRegMap();

  auto InRegion = [&](MachineInstr &UseMI) {
    return !MIRegion.contains(&UseMI);
  };

  SmallVector<MachineInstr *> Candidates;
  DenseMap<unsigned, unsigned> RegToCandIdx;

  // Identify candidate registers for rematerialization in the function.
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    RegionBoundaries Bounds = DAG.Regions[I];
    for (auto MI = Bounds.first; MI != Bounds.second; ++MI) {
      // The instruction must be rematerializable.
      MachineInstr &DefMI = *MI;
      if (!isReMaterializable(DefMI))
        continue;

      // We only support rematerializing virtual registers with one definition.
      Register Reg = DefMI.getOperand(0).getReg();
      if (!Reg.isVirtual() || !DAG.MRI.hasOneDef(Reg))
        continue;

      // Register must have at least one user and all users must be trackable.
      auto Users = DAG.MRI.use_nodbg_instructions(Reg);
      if (Users.empty() || any_of(Users, InRegion))
        continue;

      // This is a valid candidate.
      RegToCandIdx.insert({Reg, Candidates.size()});
      Candidates.push_back(&DefMI);
    }
  }
  const unsigned NumCands = Candidates.size();

  dbgs() << "There are " << NumCands << " candidates\n";
  for (const auto &[I, CandMI] : enumerate(Candidates))
    dbgs() << "  (" << I << ") " << *CandMI;
  dbgs() << "First filtering pass:\n";

  // The candidates we think are rematerializable. Start with all candidates and
  // filter them down progressively.
  BitVector Rematable(NumCands, true);
  // For each candidate (i.e., at each corresponding vector index), mark other
  // candidates which become unrematerializable if the former turns out to be
  // unrematerializable.
  SmallVector<BitVector> InvDeps(NumCands, BitVector(NumCands));

  // Perform a first filtering pass on potential candidates.
  //
  // Rematerializing a register makes it dead in its original defining region.
  // If a candidate has uses in its own region that are not rematerializable as
  // well, then it cannot be rematerialized.
  unsigned Idx = Candidates.size();
  for (const MachineInstr *CandMI : reverse(Candidates)) {
    --Idx;
    unsigned DefRegion = MIRegion.at(CandMI);

    // Returns whether UseMI is a non-rematerializable instruction in the same
    // region as the candidate register.
    auto IsNonRematRegionUse = [&](const MachineInstr &UseMI) -> bool {
      if (MIRegion.at(&UseMI) != DefRegion)
        return false;

      // Only instructions whose first operand is a register are
      // rematerializable.
      if (!UseMI.getOperand(0).isReg())
        return true;

      // Since we iterate in the reverse candidate detection order, which was
      // just program order, for two candidates defined in the same region
      // we are guaranteed to hit all uses before the definition i.e., if the
      // use is potentially rematerializable it has already been marked.
      auto UseIt = RegToCandIdx.find(UseMI.getOperand(0).getReg());
      if (UseIt == RegToCandIdx.end() || !Rematable[UseIt->second])
        return true;

      // The current candidate can only be rematerialized together with its
      // users from the same region. If the user is not rematerializable, then
      // the candidate is not rematerializable as well.
      dbgs() << "| (" << Idx << ") depends on (" << UseIt->second << ")\n";
      InvDeps[UseIt->second].set(Idx);
      return false;
    };

    Register DefReg = CandMI->getOperand(0).getReg();
    if (any_of(DAG.MRI.use_nodbg_instructions(DefReg), IsNonRematRegionUse))
      Rematable.reset(Idx);
  }

  struct RegInfo {
    RematReg::RematUses Uses;
    SmallVector<RematReg::Dependency> Dependencies;
    bool AvailableAtUses = true;
  };
  DenseMap<Register, RegInfo> Visited;

  // Initially mark all potentially rematerializable registers as root. We will
  // prune the bitvector during the second filtering pass. The set bits are
  // always a subset of those set in Rematable.
  BitVector CandRoots(Rematable);

  // Invalidates candidate I and all candidates that transitively depend on it
  // being rematerializable.
  std::function<bool(unsigned)> Invalidate = [&](unsigned I) -> bool {
    Rematable.reset(I);
    CandRoots.reset(I);
    for (unsigned J : InvDeps[I].set_bits()) {
      Rematable.reset(J);
      CandRoots.reset(J);
      Invalidate(J);
    }
    // Clear the dependency vector so that further recursive calls do not go
    // though the same invalidation chain unnecessarily.
    InvDeps[I].reset();
    return false;
  };

  dbgs() << "After first filtering pass there are " << Rematable.count()
         << " candidates left (" << NumCands - Rematable.count()
         << " eliminated)\n";
  for (const unsigned I : Rematable.set_bits())
    dbgs() << "  (" << I << ") " << *Candidates[I];
  dbgs() << "Second filtering pass:\n";

  // Determines whether the unique chain containing the ParentIdx -> I edge is
  // rematerializable. ParentIdx is null for chain roots, in which case the
  // chain being validated is the unique one rooted at candidate I.
  std::function<bool(unsigned, std::optional<unsigned>)> ValidateCandChain =
      [&](unsigned I, std::optional<unsigned> ParentIdx) -> bool {
    dbgs() << "  Validating node (" << I << ") (from ("
           << (ParentIdx ? std::to_string(*ParentIdx) : "null") << "))\n";

    assert(Rematable[I] && "non-rematable register is not valid candidate");
    const LiveIntervals &LIS = *DAG.LIS;
    const MachineInstr &CandMI = *Candidates[I];
    Register DefReg = CandMI.getOperand(0).getReg();
    unsigned DefRegion = MIRegion.at(&CandMI);

    auto [RIIt, FirstVisit] = Visited.try_emplace(DefReg);
    RegInfo &RI = RIIt->getSecond();
    SmallVector<SlotIndex, 4> ExtendedSlots;
    if (FirstVisit) {
      // Collect candidate register users in different regions the first time
      // we visit the register. Those in the same region are rematerializable
      // and will need to be rematerialized if the candidate is rematerialized.
      for (MachineInstr &UseMI : DAG.MRI.use_nodbg_instructions(DefReg)) {
        if (unsigned UseRegion = MIRegion.at(&UseMI); UseRegion != DefRegion) {
          if (auto It = RI.Uses.find(UseRegion); It != RI.Uses.end())
            It->getSecond().addUser(&UseMI, LIS);
          else
            RI.Uses.emplace_or_assign(UseRegion, &UseMI, LIS);
        }
      }
      for (const auto &[_, RegionUsers] : RI.Uses)
        ExtendedSlots.push_back(RegionUsers.LastAliveIdx);

      dbgs() << "    Collecting users on first visit:\n";
      for (const auto &[UseRegion, RegionUses] : RI.Uses) {
        dbgs() << "    | " << RegionUses.Users.size() << " users in region ["
               << UseRegion << "]; first is " << *RegionUses.InsertPos;
      }
    }
    assert(!RI.Uses.contains(DefRegion) && "cannot have users in region");

    if (ParentIdx) {
      dbgs() << "    Integrating parent's uses into child's uses:\n";
      const LiveInterval &LI = LIS.getInterval(DefReg);
      const MachineInstr &ParentCandMI = *Candidates[*ParentIdx];
      SlotIndex ParentDefSlot = LIS.getInstructionIndex(ParentCandMI);
      const VNInfo *DefVN = LI.getVNInfoAt(ParentDefSlot.getRegSlot(true));
      unsigned SubIdx = CandMI.getOperand(0).getSubReg();
      LaneBitmask Mask = SubIdx ? DAG.TRI->getSubRegIndexLaneMask(SubIdx)
                                : DAG.MRI.getMaxLaneMaskForVReg(DefReg);

      auto ExtendForRematUser = [&](SlotIndex RematIdx) -> void {
        ExtendedSlots.push_back(RematIdx);
        if (RI.AvailableAtUses)
          RI.AvailableAtUses = isAvailableAtUse(DefVN, Mask, RematIdx, LI);
      };

      // In addition to its own direct users, the candidate would need to be
      // rematerialized at points where its parents in the chain need it. We
      // already went through the parent, so its users are already available.
      const RegInfo &ParentRI = Visited.at(ParentCandMI.getOperand(0).getReg());
      for (const auto &[J, ParentRegionUses] : ParentRI.Uses) {
        assert(J != DefRegion && "cannot have users in region");
        auto It = RI.Uses.find(J);
        SlotIndex RematIdx =
            LIS.getInstructionIndex(*ParentRegionUses.InsertPos);
        if (It == RI.Uses.end()) {
          // The current candidate has no user in the region but one of its
          // parent needs to be rematerialized in it, so this one may need to be
          // rematerialized to (unless it is available there already).
          dbgs() << "    | Indirect requirement in [" << J << "]\n";
          RI.Uses.try_emplace(J, ParentRegionUses, LIS);
          ExtendForRematUser(RematIdx);
        } else {
          // The current candidate has users or is already needed by other
          // parent registers in the region. The rematerialization point of the
          // current candidate may need to be adjusted to be usable by the
          // parent register.

          RematReg::RegionUses &Uses = It->getSecond();
          if (RematIdx > Uses.LastAliveIdx) {
            dbgs() << "    | Extending avail. requirement in [" << J << "]\n";
            ExtendForRematUser(RematIdx);
            Uses.LastAliveIdx = RematIdx;
          } else if (RematIdx < LIS.getInstructionIndex(*Uses.InsertPos)) {
            dbgs() << "    | Moving insert pos. earlier in [" << J << "]\n";
            Uses.InsertPos = ParentRegionUses.InsertPos;
          }
        }
      }

      // A register which is not available at all its parents' uses cannot be
      // the root of a different chain; it must be rematerialized with its
      // parents as to not lengthen any live range.
      if (!RI.AvailableAtUses) {
        dbgs() << "    | Not a root\n";
        CandRoots.reset(I);
      }
    }

    auto RecordDepNonAvailable = [&](RematReg::Dependency &Dep) {
      assert(Dep.RematIdx && "dependency must be rematerializable");
      // Record an additional dependency when the rematerializability of the
      // current candidate requires that its dependency is rematerializable.
      // This ensures invalidation if we find out later that the dependency is
      // unrematerializable.
      InvDeps[*Dep.RematIdx].set(I);

      if (MIRegion.at(Candidates[*Dep.RematIdx]) == DefRegion) {
        // A register used to compute a rematerializable register in the
        // same region and unavailable at its uses must be rematerialized
        // with the latter; hence it cannot be a chain root.
        dbgs() << "      | Not a root\n";
        CandRoots.reset(*Dep.RematIdx);
      }
    };

    if (!FirstVisit) {
      dbgs() << "    Updating dependencies:\n";

      // We have already established the list of dependencies on the first visit
      // and it cannot change or further visits. However, rematerializability
      // and availability of some dependencies may have changed. The former
      // happens when another rematerialization chain the dependency is a part
      // of was invalidated, whereas the latter happens when the live ranges at
      // which the dependency is expected to be available have lengthened.
      for (RematReg::Dependency &Dep : RI.Dependencies) {
        assert(Dep.RematIdx || Dep.AvailableAtUses && "already unrematable");
        if (Dep.RematIdx) {
          unsigned DepIdx = *Dep.RematIdx;
          if (!Rematable[DepIdx]) {
            Dep.RematIdx = std::nullopt;
          } else {
            if (!ExtendedSlots.empty())
              continue;

            // Recurse on dependency to check whether it is available at
            // possible new slot indices.
            if (!ValidateCandChain(DepIdx, I))
              return Invalidate(I);
            if (Dep.AvailableAtUses && !Visited[DepIdx].AvailableAtUses) {
              Dep.AvailableAtUses = false;
              RecordDepNonAvailable(Dep);
            }
            continue;
          }
        }

        // The dependency is no longer rematerializable, it must be available at
        // all uses for the chain to remain valid.
        if (!Dep.AvailableAtUses)
          return Invalidate(I);

        // Only check slot extensions; previous visits of this candidate have
        // already validated that it is available at other ranges.
        const MachineOperand &MO = CandMI.getOperand(Dep.MOIdx);
        if (!isMOAvailableAtUses(MO, ExtendedSlots))
          return Invalidate(I);
      }

      // Dependencies are still valid; the candidate appears to still be
      // rematerializable.
      dbgs() << "    ** Candidate still OK\n";
      return true;
    }
    dbgs() << "    Creating dependencies on first visit:\n";

    // Derive and store the candidate's dependencies on the first visit. Recurse
    // on each of them to determine rematerializability/availability w.r.t. the
    // current chain and previously analyzed chains. If the same register
    // appears multiple times it only needs to be added once as a dependency.
    SmallDenseSet<Register, 4> AllDepRegs;
    for (const auto &[MOIdx, MO] : enumerate(CandMI.operands())) {
      Register DepReg = RematReg::isDependency(MO);
      if (!DepReg || AllDepRegs.contains(DepReg))
        continue;
      AllDepRegs.insert(DepReg);
      dbgs() << "    | Checking dependency for operand " << MOIdx << "\n";

      // The dependency must either be rematerializable or be available at all
      // of the candidate's uses, otherwise rematerializing the candidate would
      // result in a live range extension.
      RematReg::Dependency &Dep = RI.Dependencies.emplace_back(MOIdx);
      auto DepIt = RegToCandIdx.find(DepReg);
      if (DepIt != RegToCandIdx.end() && Rematable[DepIt->second]) {
        // The dependency is itself potentially rematerializable.
        unsigned DepIdx = DepIt->second;
        bool DepRematable = ValidateCandChain(DepIdx, I);
        Dep.SameRegion = DefRegion == MIRegion.at(Candidates[DepIdx]);
        Dep.AvailableAtUses = !Visited[DepIdx].AvailableAtUses;
        dbgs() << "      | Is rematable candidate (" << DepIdx << ")\n";
        dbgs() << "      | In " << (Dep.SameRegion ? "same" : "different")
               << " region\n";
        dbgs() << "      | "
               << (Dep.AvailableAtUses ? "Available" : "Not available")
               << " at uses\n";
        if (DepRematable) {
          Dep.RematIdx = DepIdx;
          if (!Dep.AvailableAtUses)
            RecordDepNonAvailable(Dep);
        } else if (!Dep.AvailableAtUses) {
          dbgs() << "      | ** Unrematable and unavailable candidate\n";
          return Invalidate(I);
        }
      } else if (!isMOAvailableAtUses(MO, ExtendedSlots)) {
        dbgs() << "      | ** Not a candidate and unavailable at uses\n";
        return Invalidate(I);
      } else {
        dbgs() << "      | ** Not a candidate but available at uses\n";
      }
    }

    // As far as we can see it is possible to rematerialize this candidate. The
    // candidate will be revisited as many times as there are valid paths to it
    // in the rematerialization tree, at which point we may realize that it is
    // in fact not rematerializable.
    dbgs() << "    ** Candidate OK\n";
    return true;
  };

  // Perform a second (recursive) filtering pass on potential candidates.
  //
  // Rematerialization should never extend a live range. If a rematerialization
  // chain depends on a non-rematerializable register that is not available at
  // all points where it would need to be used by registers in the chain were
  // the latter rematerialized, then the whole chain is unrematerializable.
  // Explore the tree recursively from candidate roots, invalidating candidates
  // as we go.
  for (unsigned I : CandRoots.set_bits()) {
    dbgs() << "Validating root [" << I << "]\n";
    if (!ValidateCandChain(I, std::nullopt))
      Invalidate(I);
  }

  // Perform a third and last filtering pass on potential candidates.
  //
  // All the current candidates could potentially be rematerialized but for the
  // sake of simplifying the rematerialization logic for now we eliminate chains
  // that share registers with other chains (this removes the need for chain
  // clustering), or chains whose registers use registers from other chains
  // (this removes the need to track dependencies between chains and update
  // users as rematerializations are being performed).
  BitVector PartOfChain(Rematable.size());
  for (unsigned RootIdx : CandRoots.set_bits()) {
    SmallVector<unsigned> Explore({RootIdx});
    SmallDenseSet<unsigned, 4> VisitedInChain;
    while (!Explore.empty()) {
      unsigned I = Explore.pop_back_val();
      // Filter out registers part of multiple chains.
      if (PartOfChain[I] || !Rematable[I]) {
        Invalidate(RootIdx);
        break;
      }
      PartOfChain.set(I);
      Register Reg = Candidates[I]->getOperand(0).getReg();
      for (const RematReg::Dependency &Dep : Visited.at(Reg).Dependencies) {
        if (!Dep.RematIdx)
          continue;
        if (!Dep.SameRegion) {
          assert(CandRoots[*Dep.RematIdx] && "expected root");
          // The root of a chain from another region has a user in the current
          // chain, creating a dependency between the two. Disable the current
          // chain to avoid having to track that.
          Invalidate(RootIdx);
          break;
        }
        if (Dep.AvailableAtUses) {
          assert(CandRoots[*Dep.RematIdx] && "expected root");
          // The root of another chain is available at the current chain's uses,
          // creating a dependency between the two. Disable the other chain to
          // avoid having to track that.
          Invalidate(*Dep.RematIdx);
        } else if (VisitedInChain.insert(*Dep.RematIdx).second) {
          Explore.push_back(*Dep.RematIdx);
        }
      }
    }
  }

  const unsigned NumRemats = Rematable.count();
  RematRegs.reserve(NumRemats);
  ChainRoots.resize(NumRemats);

  // All remaining candidates are rematerializable and form valid chains of
  // possible rematerializations. Re-index registers now that we have eliminated
  // all non-rematerializable candidates.
  DenseMap<unsigned, unsigned> CandToRematIdxRemap;
  CandToRematIdxRemap.reserve(NumRemats);
  for (auto [RematIdx, CandIdx] : enumerate(Rematable.set_bits()))
    CandToRematIdxRemap.insert({CandIdx, RematIdx});

  // Create all rematerializable registers from pre-computed info, fixing
  // rematerializable dependency indices and marking chain roots along the way.
  for (unsigned I : Rematable.set_bits()) {
    RegInfo &RI = Visited[Candidates[I]->getOperand(0).getReg()];
    // Fix-up dependencies with new indices.
    for (RematReg::Dependency &Dep : RI.Dependencies) {
      if (Dep.RematIdx)
        Dep.RematIdx = CandToRematIdxRemap.at(*Dep.RematIdx);
    }
    RematRegs.emplace_back(Candidates[I], MIRegion.at(Candidates[I]),
                           std::move(RI.Uses), std::move(RI.Dependencies), DAG);
    if (CandRoots[I])
      ChainRoots.set(CandToRematIdxRemap.at(I));
  }

  return !RematRegs.empty();
}

bool PreRARematStage::isMOAvailableAtUses(const MachineOperand &MO,
                                          ArrayRef<SlotIndex> Uses) const {
  Register DepReg = MO.getReg();
  unsigned SubIdx = MO.getSubReg();
  LaneBitmask Mask = SubIdx ? DAG.TRI->getSubRegIndexLaneMask(SubIdx)
                            : DAG.MRI.getMaxLaneMaskForVReg(DepReg);
  const LiveInterval &DepLI = DAG.LIS->getInterval(DepReg);
  const VNInfo *DefVN = DepLI.getVNInfoAt(
      DAG.LIS->getInstructionIndex(*MO.getParent()).getRegSlot(true));
  for (SlotIndex UseIdx : Uses) {
    if (!isAvailableAtUse(DefVN, Mask, UseIdx, DepLI))
      return false;
  }
  return true;
}

PreRARematStage::RematReg::RematReg(MachineInstr *DefMI, unsigned DefRegion,
                                    RematUses &&Uses,
                                    SmallVectorImpl<Dependency> &&Dependencies,
                                    GCNScheduleDAGMILive &DAG)
    : DefMI(DefMI), LiveIn(DAG.Regions.size()), LiveOut(DAG.Regions.size()),
      Live(DAG.Regions.size()), DefRegion(DefRegion), Uses(std::move(Uses)),
      Dependencies(std::move(Dependencies)) {

  // Mark regions in which the rematerializable register is live before being
  // potentially rematerialized.
  Register Reg = getReg();
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    auto LiveInIt = DAG.LiveIns[I].find(Reg);
    if (LiveInIt != DAG.LiveIns[I].end() && LiveInIt->second.any())
      LiveIn.set(I);
    auto LiveOutIt = DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).find(Reg);
    auto LiveOutEnd = DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).end();
    if (LiveOutIt != LiveOutEnd && LiveOutIt->second.any())
      LiveOut.set(I);
  }
  Live |= LiveIn;
  Live |= LiveOut;

  // Store the register's lane bitmask.
  unsigned SubIdx = DefMI->getOperand(0).getSubReg();
  Mask = SubIdx ? DAG.TRI->getSubRegIndexLaneMask(SubIdx)
                : DAG.MRI.getMaxLaneMaskForVReg(Reg);
}

bool PreRARematStage::RematReg::maybeBeneficial(
    const BitVector &TargetRegions, ArrayRef<GCNRPTarget> RPTargets) const {
  Register Reg = getReg();
  for (unsigned I : TargetRegions.set_bits()) {
    if (Live[I] && RPTargets[I].isSaveBeneficial(Reg))
      return true;
  }
  return false;
}

void PreRARematStage::RematReg::insertMI(unsigned RegionIdx,
                                         MachineInstr *RematMI,
                                         GCNScheduleDAGMILive &DAG) const {
  RegionBoundaries &Bounds = DAG.Regions[RegionIdx];
  if (Bounds.first == std::next(MachineBasicBlock::iterator(RematMI)))
    Bounds.first = RematMI;
  DAG.LIS->InsertMachineInstrInMaps(*RematMI);
}

void PreRARematStage::RematReg::RegionUses::addUser(MachineInstr *NewUser,
                                                    const LiveIntervals &LIS) {
  Users.insert(NewUser);
  if (LIS.getInstructionIndex(*NewUser) < LIS.getInstructionIndex(*InsertPos))
    InsertPos = *NewUser;
}

Register PreRARematStage::RematReg::isDependency(const MachineOperand &MO) {
  if (!MO.isReg() || !MO.readsReg())
    return Register();
  Register Reg = MO.getReg();
  if (Reg.isPhysical()) {
    // By the requirements on trivially rematerializable instructions, a
    // physical register use is either constant or ignorable.
    return Register();
  }
  return Reg;
}

PreRARematStage::RematChain::FreqInfo::FreqInfo(
    MachineFunction &MF, const GCNScheduleDAGMILive &DAG) {
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

  // Normalize to minimum observed frequency to avoid overflows when adding up
  // frequencies.
  MaxFreq /= MinFreq;
  for (uint64_t &Freq : Regions) {
    if (Freq) {
      Freq /= MinFreq;
      TotalFreq += Freq;
    } else {
      TotalFreq += MaxFreq;
    }
  }

  // Compute the scaling factor for scoring frequency differences.
  const uint64_t MaxDiff = TotalFreq - 1;
  const uint64_t MaxReprFreqValue = (1 << FreqDiffWidth) - 1;
  RescaleIsDenom = (2 * MaxDiff) & ~MaxReprFreqValue;
  if (RescaleIsDenom)
    RescaleFactor = (2 * MaxDiff) >> FreqDiffWidth;
  else
    RescaleFactor = MaxDiff ? MaxReprFreqValue / (2 * MaxDiff) : 1;
}

PreRARematStage::RematChain::RematChain(unsigned RootIdx,
                                        ArrayRef<RematReg> RematRegs,
                                        const FreqInfo &Freq,
                                        const GCNScheduleDAGMILive &DAG)
    : Live(RematRegs.front().Live.size()) {

  SmallVector<unsigned, 4> ToExplore({RootIdx});
  SmallDenseSet<unsigned, 4> Visited;
  while (!ToExplore.empty()) {
    unsigned I = ToExplore.pop_back_val();
    const RematReg &Remat = RematRegs[I];
    Regs.push_back(&Remat);
    Live |= Remat.Live;

    // Enqueue dependencies part of the same chain.
    for (const RematReg::Dependency &Dep : Remat.Dependencies) {
      if (!Dep.isRematInChain())
        continue;
      if (Visited.insert(*Dep.RematIdx).second)
        ToExplore.push_back(*Dep.RematIdx);
    }
  }

  // Initialize constant scoring data.
  for (const RematReg *Reg : Regs) {
    const TargetRegisterClass &RC = *DAG.MRI.getRegClass(Reg->getReg());
    unsigned RegSize = DAG.TRI->getRegSizeInBits(RC);
    if (unsigned SubIdx = Reg->DefMI->getOperand(0).getSubReg()) {
      // The following may return -1 (i.e., a large unsigned number) on indices
      // that may be used to access subregisters of multiple sizes; in such cases
      // fallback on the size derived from the register class.
      unsigned SubRegSize = DAG.TRI->getSubRegIdxSize(SubIdx);
      if (SubRegSize < RegSize)
        RegSize = SubRegSize;
    }
    NumRegsPerReg.push_back(divideCeil(RegSize, 32));

    // Get frequencies of defining and using regions. A rematerialization from
    // the least frequent region to the most frequent region will yield the
    // greatest latency penalty and therefore should get minimum score.
    // Reciprocally, a rematerialization in the other direction should get
    // maximum score. Default to values that will yield the worst possible score
    // given known frequencies in order to penalize rematerializations from or
    // into regions whose frequency is unknown.
    uint64_t DefOrOne = Freq.Regions[Reg->DefRegion];
    if (!DefOrOne)
      DefOrOne = 1;
    uint64_t UseOrMax = 0;
    for (const auto &[UseRegion, _] : Reg->Uses) {
      uint64_t RegionFreq = Freq.Regions[UseRegion];
      UseOrMax += RegionFreq ? RegionFreq : Freq.MaxFreq;
    }

    // The difference between defining and using frequency is in the range
    // [-MaxDiff, MaxDiff], shift it to [0,2 x MaxDiff] to stay in the positive
    // range.
    const uint64_t MaxDiff = Freq.TotalFreq - 1;
    FreqDiff += (MaxDiff + (DefOrOne - UseOrMax));
  }
  // Rescale the frequency difference to be in the [0, 2^FreqDiffWidth - 1]
  // range, averaged over the number of registers part of the chain.
  if (Freq.RescaleIsDenom)
    FreqDiff /= Freq.RescaleFactor;
  else
    FreqDiff *= Freq.RescaleFactor;
  FreqDiff /= size();
}

bool PreRARematStage::RematChain::maybeBeneficial(
    const BitVector &TargetRegions, ArrayRef<GCNRPTarget> RPTargets) const {
  for (const RematReg *Reg : Regs) {
    if (Reg->maybeBeneficial(TargetRegions, RPTargets))
      return true;
  }
  return false;
}

void PreRARematStage::RematChain::updateScore(const BitVector &TargetRegions,
                                         ArrayRef<GCNRPTarget> RPTargets,
                                         const FreqInfo &FreqInfo,
                                         bool ReduceSpill) {
  ScoreTy MaxFreq = 0, RegionImpact = 0;
  for (const auto [Reg, NumRegs] : zip_equal(Regs, NumRegsPerReg)) {
    // We shouldn't increase pressure above the target in any region. In such
    // cases set the score to 0.
    for (const auto &[UseRegion, _] : Reg->Uses) {
      if (Reg->Live[UseRegion])
        continue;
      // Even if the regions is not above the RP target this rematerialization
      // could push it above.
      if (TargetRegions[UseRegion] ||
          RPTargets[UseRegion].isAddDetrimental(Reg->getReg(), Reg->Mask))
        return setNullScore();
    }

    uint64_t RegMaxFreq = 0;
    unsigned NumBenefitingRegions = 0;
    for (unsigned I : TargetRegions.set_bits()) {
      if (!Reg->Live[I] || !RPTargets[I].isSaveBeneficial(Reg->getReg()))
        continue;
      bool UnusedLT = Reg->isUnusedLiveThrough(I);

      // Regions in which RP is guaranteed to decrease have more weight.
      NumBenefitingRegions += UnusedLT ? 2 : 1;

      if (ReduceSpill) {
        uint64_t Freq = FreqInfo.Regions[I];
        if (!UnusedLT) {
          // Apply a frequency penalty in regions in which we are not sure that
          // RP will decrease.
          Freq /= 2;
        }
        RegMaxFreq = std::max(RegMaxFreq, Freq);
      }
    }
    MaxFreq = std::max(MaxFreq, RegMaxFreq);
    RegionImpact += NumBenefitingRegions * NumRegs;
  }

  // Average this component over all registers.
  RegionImpact /= size();

  setNullScore();
  setMaxFreqScore(MaxFreq);
  setFreqDiffScore(FreqDiff);
  setRegionImpactScore(RegionImpact);
}

void PreRARematStage::rematerialize(const RematReg &Remat,
                                    BitVector &RecomputeRP,
                                    RollbackInfo &Info) {
  SmallVector<Register, 2> DepRegsToSubst;
  // All dependencies must be rematerialized before the current register.
  for (const RematReg::Dependency &Dep : Remat.Dependencies) {
    if (!Dep.isRematInChain())
      continue;
    Register DepReg = Remat.getDepReg(Dep);
    DepRegsToSubst.push_back(DepReg);
    if (!Info.RegMap.contains(DepReg))
      rematerialize(RematRegs[*Dep.RematIdx], RecomputeRP, Info);
    assert(Info.RegMap.contains(DepReg) && "missing remat info");
  }
  Info.RematOrder.push_back({&Remat, Remat.getReg()});

  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineInstr &DefMI = *Remat.DefMI;
  Register Reg = DefMI.getOperand(0).getReg();
  DenseMap<unsigned, MachineInstr *> &MyMap = Info.RegMap[Reg];

  // Rematerialize the register in every region where it is used.
  for (const auto &[UseRegion, RegionUsers] : Remat.Uses) {
    Register NewReg = DAG.MRI.cloneVirtualRegister(Reg);
    MachineBasicBlock::iterator IP = RegionUsers.InsertPos;
    TII->reMaterialize(*IP->getParent(), IP, NewReg, 0, DefMI, *DAG.TRI);
    MachineInstr &RematMI = *std::prev(IP);

    // The new instruction needs to use new registers previously rematerialized
    // as part of the chain.
    for (Register DepReg : DepRegsToSubst) {
      MachineInstr *DepRematMI = Info.RegMap.at(DepReg).at(UseRegion);
      Register NewDepReg = DepRematMI->getOperand(0).getReg();
      RematMI.substituteRegister(DepReg, NewDepReg, 0, *DAG.TRI);
    }
    Remat.insertMI(UseRegion, &RematMI, DAG);
    MyMap.insert({UseRegion, &RematMI});

    // Users of the rematerialized register in the region need to use the new
    // register.
    for (MachineInstr *UserMI : RegionUsers.Users)
      UserMI->substituteRegister(Reg, NewReg, 0, *DAG.TRI);

    // If the register is rematerialized in a region where it is not initially
    // used it might increase RP in that region.
    if (!Remat.Live[UseRegion]) {
      RPTargets[UseRegion].addReg(Reg, Remat.Mask, DAG.MRI);
      RecomputeRP.set(UseRegion);
      RescheduleRegions.set(UseRegion);
    }
  }

  // Remove the register from all regions where it is a live-in or live-out
  // and adjust RP targets.
  for (unsigned I : Remat.Live.set_bits()) {
#ifdef EXPENSIVE_CHECKS
    if (!Remat.LiveIn[I] && Remat.LiveOut[I]) {
      // All uses are known to be available / live at the remat point. Thus,
      // the uses should already be live in to the region.
      for (MachineOperand &MO : DefMI.operands()) {
        if (!MO.isReg() || !MO.getReg() || !MO.readsReg())
          continue;

        Register UseReg = MO.getReg();
        if (!UseReg.isVirtual())
          continue;

        LiveInterval &LI = DAG.LIS->getInterval(UseReg);
        LaneBitmask LM = DAG.MRI.getMaxLaneMaskForVReg(MO.getReg());
        if (LI.hasSubRanges() && MO.getSubReg())
          LM = DAG.TRI->getSubRegIndexLaneMask(MO.getSubReg());

        LaneBitmask LiveInMask = DAG.LiveIns[I].at(UseReg);
        LaneBitmask UncoveredLanes = LM & ~(LiveInMask & LM);
        // If this register has lanes not covered by the LiveIns, be sure they
        // do not map to any subrange. ref:
        // machine-scheduler-sink-trivial-remats.mir::omitted_subrange
        if (UncoveredLanes.any()) {
          assert(LI.hasSubRanges());
          for (LiveInterval::SubRange &SR : LI.subranges())
            assert((SR.LaneMask & UncoveredLanes).none());
        }
      }
    }
#endif

    // This save is exact in beneficial regions but optimistic in all other
    // regions where the register is live.
    RPTargets[I].saveReg(Reg, Remat.Mask, DAG.MRI);
    DAG.LiveIns[I].erase(Reg);
    DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).erase(Reg);
    if (!Remat.isUnusedLiveThrough(I))
      RecomputeRP.set(I);
  }

  RescheduleRegions |= Remat.Live;
}

void PreRARematStage::rematerialize(const RematChain &Chain,
                                    BitVector &RecomputeRP,
                                    RollbackInfo &Info) {
  rematerialize(Chain.root(), RecomputeRP, Info);

  // Create virtual register intervals for all rematerialized instructions and
  // delete original instructions.
  for (const auto &[_, RematInstrs] : Info.RegMap) {
    for (const auto &[_, NewMI] : RematInstrs)
      DAG.LIS->createAndComputeVirtRegInterval(NewMI->getOperand(0).getReg());
  }
  for (const auto &[Remat, _] : Info.RematOrder)
    DAG.deleteMI(Remat->DefRegion, Remat->DefMI);
}

void PreRARematStage::rollback(const RollbackInfo &Info,
                               BitVector &ImpactedRegions) const {
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  DenseMap<const RematReg *, Register> RematToNewReg;

  // Rollback in the same order as we rematerialized.
  for (const auto &[Remat, OrigReg] : Info.RematOrder) {
    const DenseMap<unsigned, MachineInstr *> &MyRegMap =
        Info.RegMap.at(OrigReg);

    // Recreate the original MI from one of the rematerializations.
    MachineInstr *ModelMI = MyRegMap.begin()->second;
    unsigned DefRegion = Remat->DefRegion;
    MachineBasicBlock *MBB = RegionBB[DefRegion];
    Register Reg = ModelMI->getOperand(0).getReg();
    Register NewReg = DAG.MRI.cloneVirtualRegister(Reg);

    // Re-rematerialize MI in its original region. Note that it may not be
    // rematerialized exactly in the same position as originally within the
    // region, but it should not matter much.
    MachineBasicBlock::iterator InsertPos(DAG.Regions[DefRegion].second);
    TII->reMaterialize(*MBB, InsertPos, NewReg, 0, *ModelMI, *DAG.TRI);
    MachineInstr &ReRematMI = *std::prev(InsertPos);
    REMAT_DEBUG(dbgs() << "[" << DefRegion << "] Re-rematerialized as "
                       << ReRematMI;);

    // Use re-rematerialized register in all regions.
    for (const auto &[UseRegion, RematMI] : MyRegMap) {
      REMAT_DEBUG({
        dbgs() << "  [" << UseRegion << "] Deleted (";
        RematMI->getOperand(0).print(dbgs());
        dbgs() << " -> ";
        std::prev(InsertPos)->getOperand(0).print(dbgs());
        dbgs() << ")\n";
      });
      Register OldReg = RematMI->getOperand(0).getReg();
      for (MachineInstr *UseMI : Remat->Uses.at(UseRegion).Users)
        UseMI->substituteRegister(OldReg, NewReg, 0, *DAG.TRI);
      if (!Remat->Live[UseRegion])
        ImpactedRegions.set(UseRegion);
    }

    // Replace dependencies with re-rematerialized registers from the chain. By
    // construction all dependencies have already been rolled back.
    for (const RematReg::Dependency &Dep : Remat->Dependencies) {
      if (!Dep.isRematInChain())
        continue;
      Register ReRematDep = RematToNewReg.at(&RematRegs[*Dep.RematIdx]);
      ReRematMI.getOperand(Dep.MOIdx).substVirtReg(ReRematDep, 0, *DAG.TRI);
    }

    Remat->insertMI(DefRegion, &ReRematMI, DAG);
    RematToNewReg.insert({Remat, NewReg});

    // Re-add the register as a live-in/live-out in all regions it used to be
    // one in, and mark regions whose RP is impacted.
    std::pair<Register, LaneBitmask> LiveReg(NewReg, Remat->Mask);
    for (unsigned I : Remat->LiveIn.set_bits())
      DAG.LiveIns[I].insert(LiveReg);
    for (unsigned I : Remat->LiveOut.set_bits())
      DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).insert(LiveReg);
    ImpactedRegions |= Remat->Live;
  }

  // Delete all rematerializations.
  for (const auto &[_, RematInstrs] : Info.RegMap) {
    for (const auto &[UseRegion, RematMI] : RematInstrs)
      DAG.deleteMI(UseRegion, RematMI);
  }
  // Create register intervals for all re-rematerialized registers.
  for (const auto &[_, ReRematReg] : RematToNewReg)
    DAG.LIS->createAndComputeVirtRegInterval(ReRematReg);
}

void PreRARematStage::unsetSatisifedRPTargets(const BitVector &Regions) {
  for (unsigned I : Regions.set_bits()) {
    if (TargetRegions[I] && RPTargets[I].satisfied()) {
      REMAT_DEBUG(dbgs() << "  [" << I << "] Target reached!\n");
      TargetRegions.reset(I);
    }
  }
}

bool PreRARematStage::updateAndVerifyRPTargets(const BitVector &Regions) {
  bool TooOptimistic = false;
  for (unsigned I : Regions.set_bits()) {
    GCNRPTarget &Target = RPTargets[I];
    Target.setRP(DAG.getRealRegPressure(I));

    // Since we were optimistic in assessing RP decreases in these regions, we
    // may need to remark the target as a target region if RP didn't decrease
    // as expected.
    if (!TargetRegions[I] && !Target.satisfied()) {
      REMAT_DEBUG(dbgs() << "  [" << I << "] Incorrect RP estimation\n");
      TooOptimistic = true;
      TargetRegions.set(I);
    }
  }
  return TooOptimistic;
}

// Copied from MachineLICM
bool PreRARematStage::isReMaterializable(const MachineInstr &MI) {
  if (!DAG.TII->isReMaterializable(MI))
    return false;

  for (const MachineOperand &MO : MI.all_uses()) {
    // We can't remat physreg uses, unless it is a constant or an ignorable
    // use (e.g. implicit exec use on VALU instructions)
    if (MO.getReg().isPhysical()) {
      if (DAG.MRI.isConstantPhysReg(MO.getReg()) || DAG.TII->isIgnorableUse(MO))
        continue;
      return false;
    }
  }

  return true;
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
  for (const RollbackInfo &Rollback : Rollbacks)
    rollback(Rollback, RecomputeRP);
  for (unsigned I : RecomputeRP.set_bits())
    DAG.Pressure[I] = DAG.getRealRegPressure(I);

  GCNSchedStage::finalizeGCNSchedStage();
}

void GCNScheduleDAGMILive::deleteMI(unsigned RegionIdx, MachineInstr *MI) {
  // It's not possible for the deleted instruction to be upper region boundary
  // since we don't delete region terminators.
  if (Regions[RegionIdx].first == MI)
    Regions[RegionIdx].first = std::next(MachineBasicBlock::iterator(MI));
  LIS->removeInterval(MI->getOperand(0).getReg());
  LIS->RemoveMachineInstrFromMaps(*MI);
  MI->eraseFromParent();
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
