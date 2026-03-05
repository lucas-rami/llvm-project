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
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "remat-strategy"

using namespace llvm;

namespace {
template <typename T> struct CandPtr {
  CandPtr(T *Ptr) : Ptr(Ptr) {}
  bool operator<(const CandPtr &O) const { return *Ptr < *O.Ptr; }
  T &operator*() { return *Ptr; }
  const T &operator*() const { return *Ptr; }

private:
  T *const Ptr;
};

struct Candidate {
  using RegSubRegPair = TargetInstrInfo::RegSubRegPair;

  /// The root register this candidates represent.
  unsigned RootIdx;
  /// Cache the defined register (to keep track of it in case the candidate is
  /// fully rematerialized).
  Register DefReg;
  /// Regions in which the register is live-in/live-out/live anywhere.
  BitVector LiveIn, LiveOut, Live;
  /// Subset of \ref Live regions in which the computed RP difference is
  /// approximate.
  BitVector ApproximateDiff;
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
  SmallDenseSet<RegSubRegPair, 4> LRExtensions;

  Candidate(unsigned RootIdx, bool SkipLiveCheck,
            const RematAcrossForRPTarget &Strat);

  /// Rematerializes the candidate to all its using regions. After the call,
  /// the dependency data maps all of the candidate's transitive dependencies to
  /// the dependencies the rematerialed versions of the register are using.
  void rematerialize(Rematerializer &Remater);

  /// Updates the candidate's score w.r.t. the current target regions.
  bool update(bool FavorSaferRemats, const RematAcrossForRPTarget &Strat);

  /// Whether the current score is considered stale enough w.r.t. the current
  /// target regions set in \p TargetRegions to make the rematerialization
  /// decision potentially not helpful at all.
  bool isScoreStale(const BitVector &TargetRegions) const;

  /// Whether rematerializing the candidate is considered to be actively
  /// detrimental to performance.
  bool isDetrimental(const RematAcrossForRPTarget &Strat) const;

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

  /// Returns whether the current score is null, indicating the
  /// rematerialization is useless.
  bool hasNullScore() const { return !RegionImpact; }

  bool makeRematPlan(const Rematerializer &Remater, const LiveIntervals &LIS);

  void computeRegionDependencies(unsigned UseRegion, SlotIndex Use,
                                 const Rematerializer &Remater);
};

} // namespace

Candidate::Candidate(unsigned RootIdx, bool SkipLiveCheck,
                     const RematAcrossForRPTarget &Strat)
    : RootIdx(RootIdx), LiveIn(Strat.getNumRegions()),
      LiveOut(Strat.getNumRegions()), Live(Strat.getNumRegions()),
      ApproximateDiff(Strat.getNumRegions()),
      BeneficialRegions(Strat.getNumRegions()) {
  const Rematerializer &Remater = Strat.getRematerializer();
  const Rematerializer::Reg &RootReg = Remater.getReg(RootIdx);

  // Mark regions in which the rematerializable register is live before being
  // potentially rematerialized.
  DefReg = RootReg.getDefReg();
  if (!SkipLiveCheck) {
    Strat.checkLivenessInRegions(DefReg, LiveIn, LiveOut);
    Live |= LiveIn;
    Live |= LiveOut;
  }
  Live.set(RootReg.DefRegion);

  // Maximum RP can decrease or stay the same in regions where the register is
  // live. In regions where it is used or is not a live-through RP may not
  // decrease.
  Save.inc(DefReg, LaneBitmask::getNone(), RootReg.Mask, Strat.MRI);
  for (unsigned I : Live.set_bits()) {
    if (!LiveIn[I] || !LiveOut[I] || RootReg.Uses.contains(I))
      ApproximateDiff.set(I);
  }
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

bool Candidate::update(bool RematForSpillAvoidance,
                       const RematAcrossForRPTarget &Strat) {
  const Rematerializer &Remater = Strat.getRematerializer();
  if (!makeRematPlan(Remater, Strat.LIS))
    return false;
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
                 : Strat.getFrequencyOrDefault(RootReg.DefRegion, 1);
  for (const auto &[UseRegion, RRI] : Remats) {
    assert(RootReg.Uses.contains(UseRegion) && "root register lost uses");

    // Account for the register and its dependencies being rematerialized to
    // the using region.
    uint64_t RegionFreq = Strat.getFrequencyOrMax(UseRegion);
    FreqDiff -= (RRI.DepRemats.size() + 1) * RegionFreq;
  }

  BeneficialRegions.reset();
  MaxFreqNoUse = MaxFreqUse = RegionImpact = 0;
  for (unsigned RegionIdx : Strat.getTargetRegions().set_bits()) {
    if (!Live[RegionIdx])
      continue;

    // The rematerialization must contribute positively in at least one
    // register class with usage above the RP target for this region to
    // contribute to the score.
    const GCNRPTarget &RegionTarget = Strat.getTarget(RegionIdx);
    const unsigned NetSave = RegionTarget.getNumRegsBenefit(Save);
    if (!NetSave)
      continue;
    BeneficialRegions.set(RegionIdx);

    const bool UsedInRegion = RootReg.Uses.contains(RegionIdx);

    // Favor rematerializations in which RP is guaranteed to decrease at region
    // bounds and across a region. This favors rematerialization across regions
    // where the register is not used.
    unsigned ImpactMultiplier = 1;
    if (LiveIn[RegionIdx] && !Strat.satisfiedAtLiveIn(RegionIdx))
      ++ImpactMultiplier;
    if (LiveOut[RegionIdx] && !Strat.satisfiedAtLiveOut(RegionIdx))
      ++ImpactMultiplier;
    if (LiveIn[RegionIdx] && LiveOut[RegionIdx] && !UsedInRegion)
      ++ImpactMultiplier;
    RegionImpact += ImpactMultiplier * NetSave;

    if (RematForSpillAvoidance) {
      uint64_t RegionFreq = Strat.getFrequencyOrDefault(RegionIdx);
      uint64_t &MaxFreq = UsedInRegion ? MaxFreqUse : MaxFreqNoUse;
      MaxFreq = std::max(MaxFreq, RegionFreq);
    }
  }

  return !hasNullScore();
}

bool Candidate::isDetrimental(const RematAcrossForRPTarget &Strat) const {
  // FIXME: We don't know how to account for and handle live-range extensions
  // yet.
  if (!LRExtensions.empty())
    return true;

  for (const auto &[UseRegion, RRI] : Remats) {
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

bool Candidate::isScoreStale(const BitVector &TargetRegions) const {
  // When the current target regions have nothing in common with the regions
  // which we computed would benefit from the rematerialization we have a good
  // indication that scores have diverged significantly from reality.
  return !TargetRegions.anyCommon(BeneficialRegions);
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
  assert(!hasNullScore() && "cannot compare candidate with null score");
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

    // Unrematerializable operands are either available and should be re-used
    // or must have their live-range extended for the rematerialization to be
    // possible.
    const Rematerializer::Reg &DAGReg = Remater.getReg(DAGRegIdx);
    for (unsigned MOIdx : Remater.getUnrematableOprds(DAGRegIdx)) {
      MachineOperand &MO = DAGReg.DefMI->getOperand(MOIdx);
      if (!Remater.isMOIdenticalAtUses(MO, Use))
        LRExtensions.insert({MO.getReg(), MO.getSubReg()});
    }

    // All dependencies must either be available at uses or rematerializable
    // and recursively meeting this criteria.
    for (const Rematerializer::Reg::Dependency &Dep : DAGReg.Dependencies) {
      if (!SeenDeps.insert(Dep.RegIdx).second)
        continue;
      if (Remater.isMOIdenticalAtUses(DAGReg.DefMI->getOperand(Dep.MOIdx),
                                      Use)) {
        // Tells the rematerializer to just re-use the existing register.
        RRI.DRI.reuse(Dep.RegIdx);
      } else {
        unsigned RematIdx =
            Remater.findRematInRegion(Dep.RegIdx, UseRegion, Use);
        if (RematIdx != Rematerializer::NoReg) {
          // Tells the rematerializer to use a rematerialized version of the
          // register.
          RRI.DRI.useRemat(Dep.RegIdx, RematIdx);
        } else {
          // The rematerializer will also rematerialize this register to the
          // region.
          RRI.DepRemats.push_back(Dep.RegIdx);
          RematDAG.push_back(Dep.RegIdx);
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
      for (const RegSubRegPair &RegSubReg : LRExtensions) {
        OS << (First ? "" : ", ") << "extend "
           << printReg(RegSubReg.Reg, nullptr, RegSubReg.SubReg, nullptr);
        First = false;
      }
      OS << " }\n";
    }
  });
}
#endif

RematAcrossForRPTarget::RematAcrossForRPTarget(
    MachineFunction &MF, SmallVectorImpl<RegionBoundaries> &Regions,
    LiveIntervals &LIS, unsigned MaxSGPRs, unsigned MaxVGPRs,
    bool SupportRollback, const MachineLoopInfo &MLI,
    SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveIns,
    SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveOuts)
    : MF(MF), MRI(MF.getRegInfo()), LIS(LIS), SupportRollback(SupportRollback),
      Remater(MF, Regions, LIS), TargetRegions(Regions.size()),
      LiveIns(LiveIns), LiveOuts(LiveOuts) {

  const unsigned NumRegions = Remater.getNumRegions();
  assert(LiveIns.size() == NumRegions && "mismatch in number of regions");
  assert(LiveOuts.size() == NumRegions && "mismatch in number of regions");

  // Compute target region data.
  RPTargets.reserve(NumRegions);
  RPExcessAtBounds.reserve(NumRegions);
  for (unsigned RegionIdx = 0, E = NumRegions; RegionIdx != E; ++RegionIdx) {
    GCNRegPressure RP = getRealRegPressure(RegionIdx);
    GCNRPTarget &Target = RPTargets.emplace_back(MaxSGPRs, MaxVGPRs, MF, RP);
    RPExcessAtBounds.emplace_back();
    if (!Target.satisfied()) {
      TargetRegions.set(RegionIdx);
      updateRegionBounds(RegionIdx);
    }
  }

  // Compute region frequency data.
  MachineBranchProbabilityInfo MBPI;
  MachineBlockFrequencyInfo MBFI(MF, MBPI, MLI);
  uint64_t MinFreq = MBFI.getEntryFreq().getFrequency();
  Frequencies.reserve(NumRegions);
  for (unsigned I = 0; I < NumRegions; ++I) {
    MachineBasicBlock *MBB = Regions[I].first->getParent();
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

  Remater.analyze();
}

bool RematAcrossForRPTarget::performRematerializations(
    BitVector &AffectedRegions, bool RematForSpillAvoidance) {

  LLVM_DEBUG({
    dbgs() << "Rematerializing across regions of function ";
    MF.getFunction().printAsOperand(dbgs(), /*PrintType=*/false);
    dbgs() << " to reduce RP below target.\n";
    dbgs() << Remater.getNumRegs()
           << " rematerializable registers were identified in the function\n"
           << printTargetRegions();
  });

  AffectedRegions.resize(getNumRegions());

  SmallVector<Candidate, 4> Candidates;
  auto CreateCandidates = [&](bool SkipLiveChecks) -> void {
    const unsigned NumRegs = Remater.getNumRegs();
    Candidates.reserve(NumRegs);
    LLVM_DEBUG(dbgs() << "Creating " << NumRegs - Candidates.size()
                      << " new candidates\n");
    for (unsigned RegIdx = Candidates.size(); RegIdx < NumRegs; ++RegIdx) {
      Candidates.emplace_back(RegIdx, SkipLiveChecks, *this);
      LLVM_DEBUG(dbgs() << "  " << Remater.printRematReg(RegIdx) << '\n');
    }
  };

  // Create corresponding candidates for all rematerializable registers.
  CreateCandidates(/*SkipLiveChecks=*/false);

  BitVector RecomputeRP(getNumRegions());
  DenseSet<unsigned> ChangedUsers;
  SmallVector<CandPtr<Candidate>> CandidateOrder;

  if (SupportRollback) {
    Rollback = std::make_unique<Rollbacker>();
    Remater.addListener(Rollback.get());
  }

  // Rematerialize registers in successive rounds until all RP targets are
  // satisfied or until we run out of rematerialization candidates.
  for (;;) {
    RecomputeRP.reset();
    ChangedUsers.clear();
    CandidateOrder.clear();

    // Order possibly useful candidates by score.
    for (Candidate &Cand : Candidates) {
      if (Cand.update(RematForSpillAvoidance, *this))
        CandidateOrder.push_back(&Cand);
    }
    if (CandidateOrder.empty())
      return false;
    sort(CandidateOrder);

    LLVM_DEBUG({
      dbgs() << "==== NEW REMAT ROUND ====\n"
             << "Current candidates, in rematerialization order:\n";
      for (const CandPtr<Candidate> Cand : reverse(CandidateOrder))
        dbgs() << (*Cand).print(Remater);
      dbgs() << printTargetRegions();
    });

    // Rematerialize candidates in decreasing score order.
    while (!CandidateOrder.empty()) {
      Candidate &Cand = *CandidateOrder.back();

      // Previous rematerializations this round may have made our intended
      // rematerialization plan invalid or our score stale. In such cases we
      // want to interrupt the round to regenerate a plan and rescore the
      // candidate.
      if (!Cand.isRematPlanValid(Remater, ChangedUsers) ||
          Cand.isScoreStale(TargetRegions))
        break;
      CandidateOrder.pop_back();

      // Just skip rematerializations we think will be detrimental. Do not
      // interrupt the round as valid rematerializations may exist further down
      // the candidate order. Such candidates may become non detrimental in the
      // future if other rematerializations improve RP enough in regions that
      // make this candidate a bad choice currently.
      if (Cand.isDetrimental(*this)) {
        LLVM_DEBUG(dbgs() << "** SKIP " << Remater.printRematReg(Cand.RootIdx)
                          << "\n");
        continue;
      }

      Cand.rematerialize(Remater);

      // Update live maps for all rematerialized registers and their
      // dependencies. Also identify all registers whose users have changed as a
      // result of the rematerialization. Since the set of depedencies a
      // register must be rematerialized with depends on user positions,
      // changing the latter invalidates the pre-computed set and scoring
      // metrics.
      ChangedUsers.insert(Cand.RootIdx);
      removeFromLiveMaps(Cand.RootIdx, Cand.DefReg, Cand.LiveIn, Cand.LiveOut);
      const Rematerializer::Reg &Reg = Remater.getReg(Cand.RootIdx);
      const bool FullRemat = Reg.Uses.empty();
      for (auto &[_, RRI] : Cand.Remats) {
        for (const auto &[OldDepIdx, NewDepIdx] : RRI.DRI.DependencyMap) {
          // New dependencies gained a new user (the rematerialized
          // instruction). If this was a full rematerialization, old
          // dependencies lost a user (the now deleted original register).
          ChangedUsers.insert(NewDepIdx);
          if (FullRemat) {
            ChangedUsers.insert(OldDepIdx);

            // Dependencies that lost a user as the result of the
            // rematerialization may no longer have out-of-defining-region users
            // or even exist (if the candidate was their last user). In both
            // cases they no longer need to be live-in/live-out anywhere.
            const Rematerializer::Reg &FromReg = Remater.getReg(OldDepIdx);
            if (!FromReg.hasUsersOutsideDefRegion()) {
              const Candidate &DepCand = Candidates[OldDepIdx];
              removeFromLiveMaps(DepCand.RootIdx, DepCand.DefReg,
                                 DepCand.LiveIn, DepCand.LiveOut);
            }
          }
        }
      }

      // Account for the expected change in RP induced by the rematerialization.
      RecomputeRP |= Cand.ApproximateDiff;
      AffectedRegions |= Cand.Live;
      for (unsigned I : Cand.Live.set_bits())
        RPTargets[I].saveRP(Cand.Save);

      if (clearIfSatisfied(Cand.Live) && TargetRegions.none()) {
        LLVM_DEBUG(dbgs() << "** All targets cleared, verifing...\n");
        // Commit changes to the live intervals before verifying RP in regions
        // affected unpredictably.
        Remater.updateLiveIntervals();
        // FIXME: we are doing expensive verification in non-target regions, can
        // we skip them? At least if we know no negative impact can occur?
        if (!verify(RecomputeRP) && TargetRegions.none()) {
          LLVM_DEBUG(dbgs() << "** Objectives achieved!\n");
          return true;
        }
        LLVM_DEBUG(printTargetRegions());
        break;
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

  return TargetRegions.none();
}

void RematAcrossForRPTarget::rollbackRematerializations() {
  LLVM_DEBUG(dbgs() << "Rollback!\n");
  Rollback->rollback(Remater);
  for (const auto &[RegIdx, Liveness] : RemovedFromLiveMaps) {
    const Rematerializer::Reg &Reg = Remater.getReg(RegIdx);
    addToLiveMaps(Reg.getDefReg(), Reg.Mask, Liveness.first, Liveness.second);
  }
}

bool RematAcrossForRPTarget::clearIfSatisfied(const BitVector &Regions) {
  bool AnyChange = false;
  for (unsigned I : Regions.set_bits()) {
    if (TargetRegions[I] && RPTargets[I].satisfied()) {
      LLVM_DEBUG(dbgs() << "  [" << I << "] Target reached!\n");
      TargetRegions.reset(I);
      AnyChange = true;
    }
  }
  return AnyChange;
}

bool RematAcrossForRPTarget::verify(const BitVector &Regions) {
  bool TooOptimistic = false;
  for (unsigned RegionIdx : Regions.set_bits()) {
    GCNRPTarget &Target = RPTargets[RegionIdx];
    bool WasTarget = TargetRegions[RegionIdx];
    Target.setRP(getRealRegPressure(RegionIdx));
    if (!Target.satisfied()) {
      TargetRegions.set(RegionIdx);
      updateRegionBounds(RegionIdx);
      // Since we were optimistic in assessing RP decreases in these regions, we
      // may need to re-tag the target region if RP didn't decrease as expected.
      if (!WasTarget) {
        LLVM_DEBUG(dbgs() << "  [" << RegionIdx
                          << "] Reverting target status\n");
        TooOptimistic = true;
      }
    } else if (WasTarget) {
      RPExcessAtBounds[RegionIdx] = {false, false};
    }
  }
  return TooOptimistic;
}

void RematAcrossForRPTarget::updateRegionBounds(unsigned RegionIdx) {
  assert(TargetRegions[RegionIdx] && "should be target region");
  assert(!RPTargets[RegionIdx].satisfied() && "target should not be satisfied");

  const GCNRPTarget &Target = RPTargets[RegionIdx];
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  std::pair<bool, bool> &RPAtBounds = RPExcessAtBounds[RegionIdx];
  RPAtBounds.first = Target.satisfied(getRegPressure(MRI, LiveIns[RegionIdx]));
  RPAtBounds.second =
      Target.satisfied(getRegPressure(MRI, LiveOuts[RegionIdx]));

  // GCNDownwardRPTracker RPTracker(LIS);
  // if (RBegin == REnd || !RPTracker.reset(*RBegin, &LiveIns[RegionIdx])) {
  //   Ranges.push_back({getBoundSlot(), getBoundSlot()});
  //   return;
  // }

  // unsigned VGPRExcess = 0;
  // if (ST.hasGFX90AInsts()) {
  //   const unsigned MaxVGPRsInRegion = MaxRP.getVGPRNum(true);
  //   if (MaxVGPRsInRegion > Target.getMaxUnifiedVGPRs())
  //     VGPRExcess = MaxVGPRsInRegion - Target.getMaxUnifiedVGPRs();
  // }

  // auto IsAboveTarget = [&]() -> bool {
  //   GCNRegPressure RP = RPTracker.getPressure();
  //   if (VGPRExcess) {
  //     if (RP.getArchVGPRNum() + VGPRExcess > MaxRP.getArchVGPRNum())
  //       return true;
  //     if (RP.getAGPRNum() + VGPRExcess > MaxRP.getAGPRNum())
  //       return true;
  //   }
  //   return !Target.satisfied(RP);
  // };

  // SlotIndex BeginRange = getBoundSlot();
  // bool AboveTarget = IsAboveTarget();

  // while (RPTracker.getNext() != REnd) {
  //   RPTracker.advance();
  //   if (IsAboveTarget()) {
  //     if (!AboveTarget) {
  //       // Start a new range above target.
  //       const MachineInstr &MI = *RPTracker.getLastTrackedMI();
  //       BeginRange = LIS.getInstructionIndex(MI);
  //       AboveTarget = true;
  //     }
  //   } else if (AboveTarget) {
  //     // End a range above target.
  //     const MachineInstr &MI = *RPTracker.getLastTrackedMI();
  //     SlotIndex EndRange = LIS.getInstructionIndex(MI);
  //     Ranges.push_back({BeginRange, EndRange});
  //     AboveTarget = false;
  //   }
  // }

  // if (AboveTarget)
  //   Ranges.push_back({BeginRange, getBoundSlot()});
  // assert(!Ranges.empty() && "target region should have target range");
}

GCNRegPressure
RematAcrossForRPTarget::getRealRegPressure(unsigned RegionIdx) const {
  const auto [RegionStart, RegionEnd] = Remater.getRegion(RegionIdx);
  if (RegionStart == RegionEnd)
    return llvm::getRegPressure(MRI, LiveIns[RegionIdx]);
  GCNDownwardRPTracker RPTracker(LIS);
  RPTracker.advance(RegionStart, RegionEnd, &LiveIns[RegionIdx]);
  return RPTracker.moveMaxPressure();
}

void RematAcrossForRPTarget::checkLivenessInRegions(Register Reg,
                                                    BitVector &LiveIn,
                                                    BitVector &LiveOut) const {
  const unsigned NumRegions = Remater.getNumRegions();
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

void RematAcrossForRPTarget::removeFromLiveMaps(unsigned RegIdx,
                                                Register DefReg,
                                                const BitVector &LiveIn,
                                                const BitVector &LiveOut) {
  if (!RemovedFromLiveMaps.insert({RegIdx, {LiveIn, LiveOut}}).second)
    return;
  for (unsigned I : LiveIn.set_bits())
    LiveIns[I].erase(DefReg);
  for (unsigned I : LiveOut.set_bits())
    LiveOuts[I].erase(DefReg);
}

void RematAcrossForRPTarget::addToLiveMaps(Register Reg, LaneBitmask Mask,
                                           const BitVector &LiveIn,
                                           const BitVector &LiveOut) {
  std::pair<Register, LaneBitmask> RegAndMask(Reg, Mask);
  for (unsigned I : LiveIn.set_bits())
    LiveIns[I].insert(RegAndMask);
  for (unsigned I : LiveOut.set_bits())
    LiveOuts[I].insert(RegAndMask);
}

Printable RematAcrossForRPTarget::printTargetRegions() const {
  return Printable([&](raw_ostream &OS) {
    if (TargetRegions.none()) {
      dbgs() << "No target regions\n";
      return;
    }
    const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
    const unsigned VGPRBlockSize = MFI.getDynamicVGPRBlockSize();
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

    dbgs() << "Target regions are:\n";
    for (unsigned I : TargetRegions.set_bits()) {
      dbgs() << "  [" << I << "] " << RPTargets[I] << " |";
      // for (const auto &[Begin, End] : RegionRanges[I]) {
      //   dbgs() << " (";
      //   if (Begin == getBoundSlot())
      //     dbgs() << "live-ins";
      //   else
      //     dbgs() << Begin;
      //   dbgs() << " -> ";
      //   if (End == getBoundSlot())
      //     dbgs() << "live-outs";
      //   else
      //     dbgs() << End;
      //   dbgs() << ')';
      // }
      dbgs() << '\n';
      dbgs() << "    @live-ins: "
             << print(getRegPressure(MRI, LiveIns[I]), &ST, VGPRBlockSize);
      dbgs() << "    @live-outs: "
             << print(getRegPressure(MRI, LiveOuts[I]), &ST, VGPRBlockSize);
    }
  });
}