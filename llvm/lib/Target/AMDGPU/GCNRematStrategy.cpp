//===-- GCNRematStrategy.cpp - Rematerialization strategy -*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#include "GCNRematStrategy.h"
#include "GCNRegPressure.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/Debug.h"
#include <utility>

#define DEBUG_TYPE "remat-strategy"

using namespace llvm;
using RegSubRegPair = TargetInstrInfo::RegSubRegPair;

static void
checkLivenessInRegions(Register Reg, BitVector &LiveIn, BitVector &LiveOut,
                       ArrayRef<GCNRPTracker::LiveRegSet> LiveIns,
                       ArrayRef<GCNRPTracker::LiveRegSet> LiveOuts) {
  assert(LiveIns.size() == LiveOuts.size() && "region number mismatch");
  const unsigned NumRegions = LiveIns.size();
  LiveIn.resize(NumRegions);
  LiveOut.resize(NumRegions);
  for (unsigned I = 0; I < NumRegions; ++I) {
    auto LiveInIt = LiveIns[I].find(Reg);
    if (LiveInIt != LiveIns[I].end() && LiveInIt->second.any())
      LiveIn.set(I);

    auto LiveOutIt = LiveOuts[I].find(Reg);
    if (LiveOutIt != LiveOuts[I].end() && LiveOutIt->second.any())
      LiveOut.set(I);
  }
}

namespace {

struct Candidate {
  /// The root register this candidates represent.
  unsigned RootIdx;
  /// Cache the defined register (to keep track of it in case the candidate is
  /// fully rematerialized).
  Register DefReg;
  /// Regions in which the register is live-in or live-out.
  BitVector LiveIn, LiveOut;
  /// RP save induced by rematerializing this register.
  GCNRegPressure Save;

  struct RegionRematInfo {
    SmallVector<unsigned, 4> DepRemats;
    Rematerializer::DependencyReuseInfo DRI;
  };
  /// Rematerialization data for each region in which we intend to rematerialize
  /// the candidate.
  DenseMap<unsigned, RegionRematInfo> Remats;
  /// Registers whose live range needs to be extended to perform the
  /// rematerialization.
  SmallDenseMap<Register, LaneBitmask, 2> LRExtensions;

  Candidate(unsigned RootIdx, const Rematerializer::Reg &Reg,
            const MachineRegisterInfo &MRI,
            ArrayRef<GCNRPTracker::LiveRegSet> LiveIns,
            ArrayRef<GCNRPTracker::LiveRegSet> LiveOuts, bool SkipLiveCheck);

  /// Rematerializes the candidate to all its using regions. After the call,
  /// the dependency data maps all of the candidate's transitive dependencies to
  /// the dependencies the rematerialed versions of the register are using.
  void rematerialize(Rematerializer &Remater);

  /// Updates the candidate's score w.r.t. the current target regions.
  void update(bool RematForSpillAvoidance, const Rematerializer &Remater,
              const RegionTargets &Regions, const RegionFrequency &FI,
              const LiveIntervals &LIS);

  /// Returns whether the current score is null, indicating the
  /// rematerialization is useless.
  bool hasNullScore() const { return !RegionImpact; }

  /// Whether the current score is considered stale enough w.r.t. the current
  /// target regions set in \p TargetRegions to make the rematerialization
  /// decision potentially not helpful at all.
  bool isScoreStale(const BitVector &LiveInTargets,
                    const BitVector &LiveOutTargets) const;

  /// Whether rematerializing the candidate is considered to be actively
  /// detrimental to performance.
  bool isDetrimental(const MachineFunction &MF) const;

  /// Determines whether it is possible/desirable to rematerialize this
  /// register *now*. Other rematerializations generally invalidate this check.
  bool isRematPlanValid(const Rematerializer &Remater,
                        const DenseSet<unsigned> &ChangedUsers) const;

  /// For each pair of candidates the most important scoring component with
  /// non-equal values determine the result of the comparison (higher is
  /// better).
  bool operator<(const Candidate &O) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  Printable print(const Rematerializer &Remater) const;
#endif

private:
  // The members below are the scoring components, top to bottom from most
  // important to least important when comparing candidates.

  /// Frequency of impacted target region with highest known frequency. This
  /// only matters when trying to reduce spilling, so it is always 0 when it is
  /// not. Higher is better.
  ///
  /// When at least one region is spilling, the register allocator will have
  /// to decide which registers to spill and where. In the worst case, it will
  /// spill in the highest frequency regions, hence we want to increase the
  /// change that no high-frequency region will require spilling i.e. we would
  /// like to give priority in clearing out high RP in high-frequency regions.
  /// This does not matter when trying to reduce occupancy since in those cases
  /// even if RA is not able to achieve the RP we think it will be able to, no
  /// spill code will be inserted.
  uint64_t MaxFreqNoUse, MaxFreqUse;
  /// Frequency difference between defining and using regions. Negative values
  /// indicate we are rematerializing to higher frequency regions; positive
  /// values indicate the contrary. Higher is better.
  int64_t FreqDiff;
  /// Expected number of target regions impacted by the rematerialization,
  /// scaled by the size of the register being rematerialized. Higher is
  /// better.
  unsigned RegionImpact;
  /// Subset of current target regions where the candidate may help reduce RP.
  BitVector BeneficialRegions;

  bool makeRematPlan(const Rematerializer &Remater, const LiveIntervals &LIS);

  void computeRegionDependencies(unsigned UseRegion, SlotIndex Use,
                                 const Rematerializer &Remater);
};

} // namespace

Candidate::Candidate(unsigned RootIdx, const Rematerializer::Reg &Reg,
                     const MachineRegisterInfo &MRI,
                     ArrayRef<GCNRPTracker::LiveRegSet> LiveIns,
                     ArrayRef<GCNRPTracker::LiveRegSet> LiveOuts,
                     bool SkipLiveCheck)
    : RootIdx(RootIdx), LiveIn(LiveIns.size()), LiveOut(LiveIns.size()),
      BeneficialRegions(LiveIns.size()) {
  // Mark regions in which the rematerializable register is live.
  DefReg = Reg.getDefReg();
  if (!SkipLiveCheck)
    checkLivenessInRegions(DefReg, LiveIn, LiveOut, LiveIns, LiveOuts);
  Save.inc(DefReg, LaneBitmask::getNone(), Reg.Mask, MRI);
}

void Candidate::rematerialize(Rematerializer &Remater) {
  LLVM_DEBUG(dbgs() << "** REMAT " << Remater.printID(RootIdx) << " to "
                    << Remats.size() << " region(s)\n");
  for (auto &[UseRegion, RDI] : Remats) {
    Remater.rematerializeToRegion(RootIdx, UseRegion, RDI.DRI);
    LLVM_DEBUG({
      unsigned E = Remater.getNumRegs() - 1;
      dbgs() << "  -> " << Remater.printID(E) << " with "
             << RDI.DepRemats.size() << " dependencies\n";
      for (unsigned I = E - RDI.DepRemats.size(); I < E; ++I) {
        dbgs() << "  " << Remater.printID(Remater.getOriginOf(I)) << " -> "
               << Remater.printID(I) << '\n';
      }
    });
  }
}

void Candidate::update(bool RematForSpillAvoidance,
                       const Rematerializer &Remater,
                       const RegionTargets &Regions, const RegionFrequency &FI,
                       const LiveIntervals &LIS) {
  BeneficialRegions.reset();
  MaxFreqNoUse = MaxFreqUse = RegionImpact = 0;
  if (!makeRematPlan(Remater, LIS))
    return;
  const Rematerializer::Reg &RootReg = Remater.getReg(RootIdx);

  // Accumulate frequencies of defining and using regions. A rematerialization
  // from the least frequent region to the most frequent region will yield the
  // greatest latency penalty and therefore should get minimum score.
  // Reciprocally, a rematerialization in the other direction should get
  // maximum score. Default to values that will yield the worst possible
  // score given known frequencies in order to penalize rematerializations
  // from or into regions whose frequency is unknown.
  FreqDiff = RootReg.hasUsersInDefRegion()
                 ? 0
                 : FI.getFrequencyOrDefault(RootReg.DefRegion, 1);
  for (const auto &[UseRegion, RRI] : Remats) {
    assert(RootReg.Uses.contains(UseRegion) && "root register lost uses");
    if (RootReg.DefRegion == UseRegion)
      continue;

    // Account for the register and its dependencies being rematerialized to
    // the using region.
    uint64_t RegionFreq = FI.getFrequencyOrMax(UseRegion);
    FreqDiff -= (RRI.DepRemats.size() + 1) * RegionFreq;
  }

  for (unsigned I = 0, E = Remater.getNumRegions(); I < E; ++I) {
    const GCNRPTarget &LiveInTarget = Regions.RPLiveIn[I];
    const GCNRPTarget &LiveOutTarget = Regions.RPLiveOut[I];

    if (LiveIn[I] && !LiveInTarget.satisfied()) {
      RegionImpact += LiveInTarget.getNumRegsBenefit(Save);
      BeneficialRegions.set(I);
    }
    if (LiveOut[I] && !LiveOutTarget.satisfied()) {
      RegionImpact += LiveOutTarget.getNumRegsBenefit(Save);
      BeneficialRegions.set(I);
    }

    // if (RematForSpillAvoidance) {
    //   uint64_t RegionFreq = FI.getFrequencyOrDefault(RegionIdx);
    //   uint64_t &MaxFreq = UsedInRegion ? MaxFreqUse : MaxFreqNoUse;
    //   MaxFreq = std::max(MaxFreq, RegionFreq);
    // }
  }
}

bool Candidate::isDetrimental(const MachineFunction &MF) const {
  GCNRegPressure ExtendRPInc;
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (const auto &[Reg, Mask] : LRExtensions)
    ExtendRPInc.inc(Reg, LaneBitmask::getNone(), Mask, MRI);
  if (Save.less(MF, ExtendRPInc))
    return true;

  for (const auto &[UseRegion, RRI] : Remats) {
    // FIXME: magic number.
    if (RRI.DepRemats.size() > 3)
      return true;
    // if (RRI.DepRemats.empty() || !Strat.isTargetRegion(UseRegion))
    //   continue;
    // When the rematerialization to the region will bring in additional
    // dependencies, it will (at least) locally increase RP, so we do not want
    // to be too aggressive in ranges where RP is already high.
    // return true;
  }
  return false;
}

bool Candidate::isScoreStale(const BitVector &LiveInTargets,
                             const BitVector &LiveOutTargets) const {
  // When the current target regions have nothing in common with the regions
  // which we computed would benefit from the rematerialization we have a good
  // indication that scores have diverged significantly from reality.
  return !LiveInTargets.anyCommon(BeneficialRegions) &&
         !LiveOutTargets.anyCommon(BeneficialRegions);
}

bool Candidate::isRematPlanValid(const Rematerializer &Remater,
                                 const DenseSet<unsigned> &ChangedUsers) const {
  if (ChangedUsers.contains(RootIdx))
    return false;

  // It is possible one of the dependencies we were planning to use was
  // deleted (if it lost all its users or was itself fully rematerialized) or
  // had had its live range change. In such cases we no longer know which
  // dependencies we need to rematerialize with the root and cannot safely
  // rematerialize this candidate.
  for (const auto &[_, RRI] : Remats) {
    for (unsigned DepIdx : RRI.DepRemats) {
      if (ChangedUsers.contains(DepIdx))
        return false;
    }
    for (const auto &[_, ToRegIdx] : RRI.DRI.DependencyMap) {
      if (ChangedUsers.contains(ToRegIdx))
        return false;
    }
  }
  return true;
}

bool Candidate::operator<(const Candidate &O) const {
  assert(!hasNullScore() && "this has null score");
  assert(!O.hasNullScore() && "this has null score");
  if (MaxFreqNoUse != O.MaxFreqNoUse)
    return MaxFreqNoUse < O.MaxFreqNoUse;
  if (MaxFreqUse != O.MaxFreqUse)
    return MaxFreqUse < O.MaxFreqUse;
  if (FreqDiff != O.FreqDiff)
    return FreqDiff < O.FreqDiff;
  if (RegionImpact != O.RegionImpact)
    return RegionImpact < O.RegionImpact;
  // Break ties using register index.
  return RootIdx > O.RootIdx;
}

bool Candidate::makeRematPlan(const Rematerializer &Remater,
                              const LiveIntervals &LIS) {
  Remats.clear();
  LRExtensions.clear();
  const Rematerializer::Reg &RootReg = Remater.getReg(RootIdx);
  if (!RootReg.hasUsersOutsideDefRegion())
    return false;

  // Collect all slots where the register would be rematerialized to.
  for (const auto &[UseRegion, RegionUses] : RootReg.Uses) {
    if (UseRegion == RootReg.DefRegion)
      continue;
    MachineInstr *FirstUser = RootReg.getRegionUseBounds(UseRegion, LIS).first;
    computeRegionDependencies(
        UseRegion, LIS.getInstructionIndex(*FirstUser).getRegSlot(true),
        Remater);
  }
  return true;
}

void Candidate::computeRegionDependencies(unsigned UseRegion, SlotIndex Use,
                                          const Rematerializer &Remater) {
  SmallVector<unsigned, 4> RematDAG{RootIdx};
  SmallDenseSet<unsigned, 4> SeenDeps;
  RegionRematInfo &RRI = Remats[UseRegion];
  do {
    unsigned DAGRegIdx = RematDAG.pop_back_val();
    const Rematerializer::Reg &DAGReg = Remater.getReg(DAGRegIdx);
    SlotIndex RefSlot = Remater.getLIS()
                            .getInstructionIndex(*DAGReg.getFirstDef())
                            .getRegSlot();

    // Unrematerializable operands are either available and should be re-used
    // or must have their live-range extended for the rematerialization to be
    // possible.
    for (const auto &[Reg, Mask] : Remater.getUnrematableOprds(DAGRegIdx)) {
      if (!Remater.isRegIdenticalAtUses(Reg, Mask, RefSlot, Use))
        LRExtensions[Reg] |= Mask;
    }

    // All dependencies must either be available at uses or rematerializable
    // and recursively meeting this criteria.
    for (Rematerializer::RegisterIdx DepRegIdx : DAGReg.Dependencies) {
      if (!SeenDeps.insert(DepRegIdx).second)
        continue;
      const Rematerializer::Reg &DepReg = Remater.getReg(DepRegIdx);
      if (Remater.isRegIdenticalAtUses(DepReg.getDefReg(), DepReg.Mask, RefSlot,
                                       Use)) {
        // Tells the rematerializer to just re-use the existing register.
        RRI.DRI.reuse(DepRegIdx);
      } else {
        unsigned RematIdx =
            Remater.findRematInRegion(DepRegIdx, UseRegion, Use);
        if (RematIdx != Rematerializer::NoReg) {
          // Tells the rematerializer to use a rematerialized version of the
          // register.
          RRI.DRI.useRemat(DepRegIdx, RematIdx);
        } else {
          // The rematerializer will also rematerialize this register to the
          // region.
          RRI.DepRemats.push_back(DepRegIdx);
          RematDAG.push_back(DepRegIdx);
        }
      }
    }
  } while (!RematDAG.empty());
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
Printable Candidate::print(const Rematerializer &Remater) const {
  return Printable([&, Remater](raw_ostream &OS) {
    OS << '(' << MaxFreqNoUse << ", " << MaxFreqUse << ", " << FreqDiff << ", "
       << RegionImpact << ") | " << Remater.printRematReg(RootIdx) << '\n';
    for (const auto &[UseRegion, RRI] : Remats) {
      const unsigned NumUsers =
          Remater.getReg(RootIdx).Uses.at(UseRegion).size();
      OS << "  [" << UseRegion << "] (" << NumUsers << " uses) { ";
      bool First = true;
      for (unsigned DepIdx : RRI.DepRemats) {
        OS << (First ? "" : ", ") << "remat " << Remater.printID(DepIdx);
        First = false;
      }
      for (const auto &[FromDepIdx, ToDepIdx] : RRI.DRI.DependencyMap) {
        OS << (First ? "" : ", ");
        if (FromDepIdx == ToDepIdx) {
          OS << "reuse " << Remater.printID(ToDepIdx);
        } else {
          OS << "use remat " << Remater.printID(ToDepIdx) << " -> "
             << Remater.printID(ToDepIdx);
        }
        First = false;
      }
      for (const auto &[Reg, _] : LRExtensions) {
        OS << (First ? "" : ", ") << "extend "
           << printReg(Reg, nullptr, 0, nullptr);
        First = false;
      }
      OS << " }\n";
    }
  });
}
#endif

RegionFrequency::RegionFrequency(const MachineFunction &MF,
                                 const MachineLoopInfo &MLI,
                                 const Rematerializer &Remater) {
  // Compute region frequency data.
  MachineBranchProbabilityInfo MBPI;
  MachineBlockFrequencyInfo MBFI(MF, MBPI, MLI);
  uint64_t MinFreq = MBFI.getEntryFreq().getFrequency();
  Frequencies.reserve(Remater.getNumRegions());
  for (unsigned I = 0, E = Remater.getNumRegions(); I < E; ++I) {
    MachineBasicBlock *MBB = Remater.getRegion(I).first->getParent();
    uint64_t BlockFreq = MBFI.getBlockFreq(MBB).getFrequency();
    Frequencies.push_back(BlockFreq);
    if (BlockFreq && BlockFreq < MinFreq)
      MinFreq = BlockFreq;
    else if (BlockFreq > MaxFreq)
      MaxFreq = BlockFreq;
  }
  if (!MinFreq)
    return;

  // Normalize to minimum observed frequency to avoid underflows/overflows when
  // combining frequencies.
  for (uint64_t &Freq : Frequencies)
    Freq /= MinFreq;
  MaxFreq /= MinFreq;
}

RematForRPTarget::RematForRPTarget(
    Rematerializer &Remater, unsigned MaxSGPRs, unsigned MaxVGPRs,
    const MachineLoopInfo &MLI,
    SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveIns,
    SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveOuts)
    : MF(Remater.getMF()), LIS(Remater.getLIS()), Remater(Remater),
      LiveIns(LiveIns), LiveOuts(LiveOuts), Targets(Remater.getNumRegions()),
      FI(MF, MLI, Remater) {
  const unsigned NumRegions = Remater.getNumRegions();
  assert(LiveIns.size() == NumRegions && "mismatch in number of regions");
  assert(LiveOuts.size() == NumRegions && "mismatch in number of regions");

  StartLiveIns.reserve(LiveIns.size());
  StartLiveIns.insert(StartLiveIns.begin(), LiveIns.begin(), LiveIns.end());
  StartLiveOuts.reserve(LiveIns.size());
  StartLiveOuts.insert(StartLiveOuts.begin(), LiveOuts.begin(), LiveOuts.end());

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (unsigned I = 0, E = NumRegions; I != E; ++I) {
    GCNRegPressure LiveInRP = getRegPressure(MRI, LiveIns[I]);
    GCNRPTarget &LiveInTarget =
        Targets.RPLiveIn.emplace_back(MaxSGPRs, MaxVGPRs, MF, LiveInRP);
    if (!LiveInTarget.satisfied())
      Targets.ExcessLiveIn.set(I);

    GCNRegPressure LiveOutRP = getRegPressure(MRI, LiveOuts[I]);
    GCNRPTarget &LiveOutTarget =
        Targets.RPLiveOut.emplace_back(MaxSGPRs, MaxVGPRs, MF, LiveOutRP);
    if (!LiveOutTarget.satisfied())
      Targets.ExcessLiveOut.set(I);
  }

  LLVM_DEBUG(dbgs() << "Initialized rematerialization strategy: "
                    << printTargetRegions());
}

bool RematForRPTarget::rematerialize(BitVector &AffectedRegions,
                                     bool RematForSpillAvoidance) {
  LLVM_DEBUG({
    dbgs() << "Rematerializing across regions of function ";
    MF.getFunction().printAsOperand(dbgs(), /*PrintType=*/false);
    dbgs() << " to reduce RP below target.\n";
    dbgs() << Remater.getNumRegs()
           << " rematerializable registers were identified in the function\n"
           << printTargetRegions();
  });

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  AffectedRegions.resize(getNumRegions());
  if (!hasTargetRegions())
    return true;

  SmallVector<Candidate, 4> Candidates;
  auto CreateCandidates = [&](bool SkipLiveChecks) -> void {
    const unsigned NumRegs = Remater.getNumRegs();
    Candidates.reserve(NumRegs);
    LLVM_DEBUG(dbgs() << "Creating " << NumRegs - Candidates.size()
                      << " new candidates\n");
    for (unsigned RegIdx = Candidates.size(); RegIdx < NumRegs; ++RegIdx) {
      Candidates.emplace_back(RegIdx, Remater.getReg(RegIdx), MRI, LiveIns,
                              LiveOuts, SkipLiveChecks);
      LLVM_DEBUG(dbgs() << "  " << Remater.printRematReg(RegIdx) << '\n');
    }
  };

  // Create corresponding candidates for all rematerializable registers.
  CreateCandidates(/*SkipLiveChecks=*/false);

  BitVector RecomputeRP(getNumRegions());
  DenseSet<unsigned> ChangedUsers;
  SmallVector<unsigned> CandidateOrder;

  // Rematerialize registers in successive rounds until all RP targets are
  // satisfied or until we run out of rematerialization candidates.
  for (;;) {
    RecomputeRP.reset();
    ChangedUsers.clear();
    CandidateOrder.clear();

    // Order possibly useful candidates by score.
    for (auto [I, Cand] : enumerate(Candidates)) {
      Cand.update(RematForSpillAvoidance, Remater, Targets, FI, LIS);
      if (!Cand.hasNullScore())
        CandidateOrder.push_back(I);
    }
    if (CandidateOrder.empty())
      return false;
    sort(CandidateOrder);

    LLVM_DEBUG({
      dbgs() << "==== NEW REMAT ROUND ====\n"
             << "Current candidates, in rematerialization order:\n";
      for (unsigned CandIdx : reverse(CandidateOrder))
        dbgs() << Candidates[CandIdx].print(Remater);
      dbgs() << printTargetRegions();
    });

    // Rematerialize candidates in decreasing score order.
    while (!CandidateOrder.empty()) {
      Candidate &Cand = Candidates[CandidateOrder.back()];

      // Previous rematerializations this round may have made our intended
      // rematerialization plan invalid or our score stale. In such cases we
      // want to interrupt the round to regenerate a plan and rescore the
      // candidate.
      if (!Cand.isRematPlanValid(Remater, ChangedUsers) ||
          Cand.isScoreStale(Targets.ExcessLiveIn, Targets.ExcessLiveOut))
        break;
      CandidateOrder.pop_back();

      // Just skip rematerializations we think will be detrimental. Do not
      // interrupt the round as valid rematerializations may exist further down
      // the candidate order. Such candidates may become non detrimental in the
      // future if other rematerializations improve RP enough in regions that
      // make this candidate a bad choice currently.
      if (Cand.isDetrimental(MF)) {
        LLVM_DEBUG(dbgs() << "** SKIP " << Remater.printRematReg(Cand.RootIdx)
                          << "\n");
        continue;
      }

      Cand.rematerialize(Remater);

      // Extended live ranges become live-ins/live-outs in regions in which the
      // rematerialized register was a live-in/live-out.
      for (const auto &[Reg, Mask] : Cand.LRExtensions)
        addToLiveMaps(Reg, Mask, Cand.LiveIn, Cand.LiveOut);

      // Update live maps for all rematerialized registers and their
      // dependencies. Also identify all registers whose users have changed as a
      // result of the rematerialization. Since the set of depedencies a
      // register must be rematerialized with depends on user positions,
      // changing the latter invalidates the pre-computed set and scoring
      // metrics.
      ChangedUsers.insert(Cand.RootIdx);
      const Rematerializer::Reg &Reg = Remater.getReg(Cand.RootIdx);
      removeFromLiveMaps(Cand.DefReg, Reg.Mask, Cand.LiveIn, Cand.LiveOut);
      for (auto &[_, RRI] : Cand.Remats) {
        for (const auto &[OldDepIdx, NewDepIdx] : RRI.DRI.DependencyMap) {
          // New dependencies gained a new user (the rematerialized
          // instruction). If this was a full rematerialization, old
          // dependencies lost a user (the now deleted original register).
          ChangedUsers.insert(NewDepIdx);
          if (!Reg.isAlive()) {
            ChangedUsers.insert(OldDepIdx);

            // Dependencies that lost a user as the result of the
            // rematerialization may no longer have out-of-defining-region users
            // or even exist (if the candidate was their last user). In both
            // cases they no longer need to be live-in/live-out anywhere.
            const Rematerializer::Reg &DepReg = Remater.getReg(OldDepIdx);
            if (!DepReg.hasUsersOutsideDefRegion()) {
              const Candidate &DepCand = Candidates[OldDepIdx];
              removeFromLiveMaps(DepCand.DefReg, DepReg.Mask, DepCand.LiveIn,
                                 DepCand.LiveOut);
            }
          }
        }
      }

      // Account for the expected change in RP induced by the rematerialization.
      AffectedRegions |= Cand.LiveIn;
      AffectedRegions |= Cand.LiveOut;
      if (!hasTargetRegions()) {
        LLVM_DEBUG(dbgs() << "** Objectives achieved!\n");
        Remater.updateLiveIntervals();
        return true;
      }
    }

    LLVM_DEBUG({
      if (CandidateOrder.empty())
        dbgs() << "** Stop round on exhausted rematerialization candidates\n";
      else if (Candidates.size() != Remater.getNumRegs())
        dbgs() << "** Interrupt round on stale data / modified state\n";
      else
        dbgs() << "** Stop inter-region rematerializations\n";
    });

    // Stop if nothing was rematerialized.
    if (Candidates.size() == Remater.getNumRegs())
      return false;

    // Create candidates for new registers. By construction these are only live
    // within their defining region so there is no need to add them to
    // live-in/live-out maps.
    Remater.updateLiveIntervals();
    CreateCandidates(/*SkipLiveChecks=*/true);
  }
}

void RematForRPTarget::rollback() {
  LiveIns.clear();
  LiveOuts.clear();
  for (GCNRPTracker::LiveRegSet &RegionLiveIns : StartLiveIns)
    LiveIns.push_back(std::move(RegionLiveIns));
  for (GCNRPTracker::LiveRegSet &RegionLiveOuts : StartLiveOuts)
    LiveOuts.push_back(std::move(RegionLiveOuts));
}

static void
removeFromLiveMapsImpl(Register Reg, LaneBitmask Mask,
                       SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveRegs,
                       BitVector &UnsatTargets,
                       SmallVectorImpl<GCNRPTarget> &Targets,
                       const BitVector &LiveRegions) {
  for (unsigned I : LiveRegions.set_bits()) {
    GCNRPTracker::LiveRegSet &RegionLiveOuts = LiveRegs[I];
    auto LiveOut = RegionLiveOuts.find(Reg);
    if (LiveOut == RegionLiveOuts.end())
      continue;
    LaneBitmask &CurrentMask = LiveOut->getSecond();
    CurrentMask &= ~Mask;
    if (CurrentMask.none())
      RegionLiveOuts.erase(Reg);
    Targets[I].saveReg(Reg, Mask);
    if (UnsatTargets[I] && Targets[I].satisfied()) {
      LLVM_DEBUG(dbgs() << "  [" << I << "] Target reached!\n");
      UnsatTargets.reset(I);
    }
  }
}

void RematForRPTarget::removeFromLiveMaps(Register Reg, LaneBitmask Mask,
                                          const BitVector &LiveIn,
                                          const BitVector &LiveOut) {
  removeFromLiveMapsImpl(Reg, Mask, LiveIns, Targets.ExcessLiveIn,
                         Targets.RPLiveIn, LiveIn);
  removeFromLiveMapsImpl(Reg, Mask, LiveOuts, Targets.ExcessLiveOut,
                         Targets.RPLiveOut, LiveOut);
}

static void addToLiveMapsImpl(
    Register Reg, LaneBitmask Mask,
    SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveRegs,
    BitVector &UnsatTargets, SmallVectorImpl<GCNRPTarget> &Targets,
    const BitVector &AddLiveRegions, const MachineRegisterInfo &MRI) {
  for (unsigned I : AddLiveRegions.set_bits()) {
    GCNRPTracker::LiveRegSet &RegionLiveIns = LiveRegs[I];
    auto LiveIn = RegionLiveIns.find(Reg);
    if (LiveIn != RegionLiveIns.end())
      LiveIn->getSecond() |= Mask;
    else
      RegionLiveIns.insert({Reg, Mask});
    Targets[I].inc(Reg, Mask, MRI);
    if (!UnsatTargets[I] && !Targets[I].satisfied()) {
      LLVM_DEBUG(dbgs() << "  [" << I << "] Target exceeded!\n");
      UnsatTargets.set(I);
    }
  }
}

void RematForRPTarget::addToLiveMaps(Register Reg, LaneBitmask Mask,
                                     const BitVector &LiveIn,
                                     const BitVector &LiveOut) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  addToLiveMapsImpl(Reg, Mask, LiveIns, Targets.ExcessLiveIn, Targets.RPLiveIn,
                    LiveIn, MRI);
  addToLiveMapsImpl(Reg, Mask, LiveOuts, Targets.ExcessLiveOut,
                    Targets.RPLiveOut, LiveOut, MRI);
}

Printable RematForRPTarget::printTargetRegions() const {
  return Printable([&](raw_ostream &OS) {
    if (Targets.ExcessLiveIn.none() && Targets.ExcessLiveOut.none()) {
      dbgs() << "No target regions\n";
      return;
    }
    dbgs() << "Target regions are:\n";
    for (unsigned I = 0, E = getNumRegions(); I < E; ++I) {
      const auto &[RegionBegin, RegionEnd] = Remater.getRegion(I);
      const bool IsEmptyRegion = RegionBegin == RegionEnd;
      if (Targets.ExcessLiveIn[I]) {
        dbgs() << "  [" << I << "] live-ins";
        if (!IsEmptyRegion) {
          MachineBasicBlock::iterator FirstNonDebug =
              skipDebugInstructionsForward(RegionBegin, RegionEnd);
          if (FirstNonDebug != RegionEnd) {
            dbgs() << " @ " << LIS.getInstructionIndex(*FirstNonDebug) << ": ";
            FirstNonDebug->print(dbgs(), /*IsStandalone=*/true,
                                 /*SkipOpers=*/true, /*SkipDebugLoc=*/false,
                                 /*AddNewLine=*/false);
          }
        }
        dbgs() << "\n    " << Targets.RPLiveIn[I] << "\n";
      }
      if (Targets.ExcessLiveOut[I]) {
        dbgs() << "  [" << I << "] live-outs";
        if (!IsEmptyRegion) {
          MachineBasicBlock::iterator LastNonDebug =
              skipDebugInstructionsBackward(RegionEnd, RegionBegin);
          if (LastNonDebug != RegionBegin) {
            dbgs() << " @ " << LIS.getInstructionIndex(*LastNonDebug) << ": ";
            LastNonDebug->print(dbgs(), /*IsStandalone=*/true,
                                /*SkipOpers=*/true, /*SkipDebugLoc=*/false,
                                /*AddNewLine=*/false);
          }
        }
        dbgs() << "\n    " << Targets.RPLiveOut[I] << "\n";
      }
    }
  });
}
