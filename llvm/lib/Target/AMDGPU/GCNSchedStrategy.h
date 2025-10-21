//===-- GCNSchedStrategy.h - GCN Scheduler Strategy -*- C++ -*-------------===//
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

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNSCHEDSTRATEGY_H
#define LLVM_LIB_TARGET_AMDGPU_GCNSCHEDSTRATEGY_H

#include "GCNRegPressure.h"
#include "GCNRematDAG.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include <cstdint>
#include <limits>

namespace llvm {

class SIMachineFunctionInfo;
class SIRegisterInfo;
class GCNSubtarget;
class GCNSchedStage;

enum class GCNSchedStageID : unsigned {
  OccInitialSchedule = 0,
  UnclusteredHighRPReschedule = 1,
  ClusteredLowOccupancyReschedule = 2,
  PreRARematerialize = 3,
  ILPInitialSchedule = 4,
  MemoryClauseInitialSchedule = 5
};

#ifndef NDEBUG
raw_ostream &operator<<(raw_ostream &OS, const GCNSchedStageID &StageID);
#endif

/// This is a minimal scheduler strategy.  The main difference between this
/// and the GenericScheduler is that GCNSchedStrategy uses different
/// heuristics to determine excess/critical pressure sets.
class GCNSchedStrategy : public GenericScheduler {
protected:
  SUnit *pickNodeBidirectional(bool &IsTopNode);

  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand, bool IsBottomUp);

  void initCandidate(SchedCandidate &Cand, SUnit *SU, bool AtTop,
                     const RegPressureTracker &RPTracker,
                     const SIRegisterInfo *SRI, unsigned SGPRPressure,
                     unsigned VGPRPressure, bool IsBottomUp);

  std::vector<unsigned> Pressure;

  std::vector<unsigned> MaxPressure;

  unsigned SGPRExcessLimit;

  unsigned VGPRExcessLimit;

  unsigned TargetOccupancy;

  MachineFunction *MF;

  // Scheduling stages for this strategy.
  SmallVector<GCNSchedStageID, 4> SchedStages;

  // Pointer to the current SchedStageID.
  SmallVectorImpl<GCNSchedStageID>::iterator CurrentStage = nullptr;

  // GCN RP Tracker for top-down scheduling
  mutable GCNDownwardRPTracker DownwardTracker;

  // GCN RP Tracker for botttom-up scheduling
  mutable GCNUpwardRPTracker UpwardTracker;

public:
  // schedule() have seen register pressure over the critical limits and had to
  // track register pressure for actual scheduling heuristics.
  bool HasHighPressure;

  // Schedule known to have excess register pressure. Be more conservative in
  // increasing ILP and preserving VGPRs.
  bool KnownExcessRP = false;

  // An error margin is necessary because of poor performance of the generic RP
  // tracker and can be adjusted up for tuning heuristics to try and more
  // aggressively reduce register pressure.
  unsigned ErrorMargin = 3;

  // Bias for SGPR limits under a high register pressure.
  const unsigned HighRPSGPRBias = 7;

  // Bias for VGPR limits under a high register pressure.
  const unsigned HighRPVGPRBias = 7;

  unsigned SGPRCriticalLimit;

  unsigned VGPRCriticalLimit;

  unsigned SGPRLimitBias = 0;

  unsigned VGPRLimitBias = 0;

  GCNSchedStrategy(const MachineSchedContext *C);

  SUnit *pickNode(bool &IsTopNode) override;

  void schedNode(SUnit *SU, bool IsTopNode) override;

  void initialize(ScheduleDAGMI *DAG) override;

  unsigned getTargetOccupancy() { return TargetOccupancy; }

  void setTargetOccupancy(unsigned Occ) { TargetOccupancy = Occ; }

  GCNSchedStageID getCurrentStage();

  // Advances stage. Returns true if there are remaining stages.
  bool advanceStage();

  bool hasNextStage() const;

  GCNSchedStageID getNextStage() const;

  GCNDownwardRPTracker *getDownwardTracker() { return &DownwardTracker; }

  GCNUpwardRPTracker *getUpwardTracker() { return &UpwardTracker; }
};

/// The goal of this scheduling strategy is to maximize kernel occupancy (i.e.
/// maximum number of waves per simd).
class GCNMaxOccupancySchedStrategy final : public GCNSchedStrategy {
public:
  GCNMaxOccupancySchedStrategy(const MachineSchedContext *C,
                               bool IsLegacyScheduler = false);
};

/// The goal of this scheduling strategy is to maximize ILP for a single wave
/// (i.e. latency hiding).
class GCNMaxILPSchedStrategy final : public GCNSchedStrategy {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

public:
  GCNMaxILPSchedStrategy(const MachineSchedContext *C);
};

/// The goal of this scheduling strategy is to maximize memory clause for a
/// single wave.
class GCNMaxMemoryClauseSchedStrategy final : public GCNSchedStrategy {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

public:
  GCNMaxMemoryClauseSchedStrategy(const MachineSchedContext *C);
};

class ScheduleMetrics {
  unsigned ScheduleLength;
  unsigned BubbleCycles;

public:
  ScheduleMetrics() {}
  ScheduleMetrics(unsigned L, unsigned BC)
      : ScheduleLength(L), BubbleCycles(BC) {}
  unsigned getLength() const { return ScheduleLength; }
  unsigned getBubbles() const { return BubbleCycles; }
  unsigned getMetric() const {
    unsigned Metric = (BubbleCycles * ScaleFactor) / ScheduleLength;
    // Metric is zero if the amount of bubbles is less than 1% which is too
    // small. So, return 1.
    return Metric ? Metric : 1;
  }
  static const unsigned ScaleFactor;
};

inline raw_ostream &operator<<(raw_ostream &OS, const ScheduleMetrics &Sm) {
  dbgs() << "\n Schedule Metric (scaled by " << ScheduleMetrics::ScaleFactor
         << " ) is: " << Sm.getMetric() << " [ " << Sm.getBubbles() << "/"
         << Sm.getLength() << " ]\n";
  return OS;
}

class GCNScheduleDAGMILive;
class RegionPressureMap {
  GCNScheduleDAGMILive *DAG;
  // The live in/out pressure as indexed by the first or last MI in the region
  // before scheduling.
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> RegionLiveRegMap;
  // The mapping of RegionIDx to key instruction
  DenseMap<unsigned, MachineInstr *> IdxToInstruction;
  // Whether we are calculating LiveOuts or LiveIns
  bool IsLiveOut;

public:
  RegionPressureMap() {}
  RegionPressureMap(GCNScheduleDAGMILive *GCNDAG, bool LiveOut)
      : DAG(GCNDAG), IsLiveOut(LiveOut) {}
  // Build the Instr->LiveReg and RegionIdx->Instr maps
  void buildLiveRegMap();

  // Retrieve the LiveReg for a given RegionIdx
  GCNRPTracker::LiveRegSet &getLiveRegsForRegionIdx(unsigned RegionIdx) {
    assert(IdxToInstruction.contains(RegionIdx));
    MachineInstr *Key = IdxToInstruction[RegionIdx];
    return RegionLiveRegMap[Key];
  }
};

class GCNScheduleDAGMILive final : public ScheduleDAGMILive {
  friend class GCNSchedStage;
  friend class OccInitialScheduleStage;
  friend class UnclusteredHighRPStage;
  friend class ClusteredLowOccStage;
  friend class PreRARematStage;
  friend class ILPInitialScheduleStage;
  friend class RegionPressureMap;

  const GCNSubtarget &ST;

  SIMachineFunctionInfo &MFI;

  // Occupancy target at the beginning of function scheduling cycle.
  unsigned StartingOccupancy;

  // Minimal real occupancy recorder for the function.
  unsigned MinOccupancy;

  // Vector of regions recorder for later rescheduling
  SmallVector<RegionBoundaries, 32> Regions;

  // Record regions with high register pressure.
  BitVector RegionsWithHighRP;

  // Record regions with excess register pressure over the physical register
  // limit. Register pressure in these regions usually will result in spilling.
  BitVector RegionsWithExcessRP;

  // Regions that have IGLP instructions (SCHED_GROUP_BARRIER or IGLP_OPT).
  BitVector RegionsWithIGLPInstrs;

  // Region live-in cache.
  SmallVector<GCNRPTracker::LiveRegSet, 32> LiveIns;

  // Region pressure cache.
  SmallVector<GCNRegPressure, 32> Pressure;

  // Temporary basic block live-in cache.
  DenseMap<const MachineBasicBlock *, GCNRPTracker::LiveRegSet> MBBLiveIns;

  // The map of the initial first region instruction to region live in registers
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> BBLiveInMap;

  // Calculate the map of the initial first region instruction to region live in
  // registers
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> getRegionLiveInMap() const;

  // Calculate the map of the initial last region instruction to region live out
  // registers
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
  getRegionLiveOutMap() const;

  // The live out registers per region. These are internally stored as a map of
  // the initial last region instruction to region live out registers, but can
  // be retreived with the regionIdx by calls to getLiveRegsForRegionIdx.
  RegionPressureMap RegionLiveOuts;

  // Return current region pressure.
  GCNRegPressure getRealRegPressure(unsigned RegionIdx) const;

  // Compute and cache live-ins and pressure for all regions in block.
  void computeBlockPressure(unsigned RegionIdx, const MachineBasicBlock *MBB);

  void runSchedStages();

  std::unique_ptr<GCNSchedStage> createSchedStage(GCNSchedStageID SchedStageID);

  void insertMI(unsigned RegionIdx, MachineInstr *RematMI);

  void deleteMI(unsigned RegionIdx, MachineInstr *MI);

public:
  GCNScheduleDAGMILive(MachineSchedContext *C,
                       std::unique_ptr<MachineSchedStrategy> S);

  void schedule() override;

  void finalizeSchedule() override;
};

// GCNSchedStrategy applies multiple scheduling stages to a function.
class GCNSchedStage {
protected:
  GCNScheduleDAGMILive &DAG;

  GCNSchedStrategy &S;

  MachineFunction &MF;

  SIMachineFunctionInfo &MFI;

  const GCNSubtarget &ST;

  const GCNSchedStageID StageID;

  // The current block being scheduled.
  MachineBasicBlock *CurrentMBB = nullptr;

  // Current region index.
  unsigned RegionIdx = 0;

  // Record the original order of instructions before scheduling.
  std::vector<MachineInstr *> Unsched;

  // RP before scheduling the current region.
  GCNRegPressure PressureBefore;

  // RP after scheduling the current region.
  GCNRegPressure PressureAfter;

  std::vector<std::unique_ptr<ScheduleDAGMutation>> SavedMutations;

  GCNSchedStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG);

public:
  // Initialize state for a scheduling stage. Returns false if the current stage
  // should be skipped.
  virtual bool initGCNSchedStage();

  // Finalize state after finishing a scheduling pass on the function.
  virtual void finalizeGCNSchedStage();

  // Setup for scheduling a region. Returns false if the current region should
  // be skipped.
  virtual bool initGCNRegion();

  // Track whether a new region is also a new MBB.
  void setupNewBlock();

  // Finalize state after scheudling a region.
  void finalizeGCNRegion();

  // Check result of scheduling.
  void checkScheduling();

  // computes the given schedule virtual execution time in clocks
  ScheduleMetrics getScheduleMetrics(const std::vector<SUnit> &InputSchedule);
  ScheduleMetrics getScheduleMetrics(const GCNScheduleDAGMILive &DAG);
  unsigned computeSUnitReadyCycle(const SUnit &SU, unsigned CurrCycle,
                                  DenseMap<unsigned, unsigned> &ReadyCycles,
                                  const TargetSchedModel &SM);

  // Returns true if scheduling should be reverted.
  virtual bool shouldRevertScheduling(unsigned WavesAfter);

  // Returns true if current region has known excess pressure.
  bool isRegionWithExcessRP() const {
    return DAG.RegionsWithExcessRP[RegionIdx];
  }

  // The region number this stage is currently working on
  unsigned getRegionIdx() { return RegionIdx; }

  // Returns true if the new schedule may result in more spilling.
  bool mayCauseSpilling(unsigned WavesAfter);

  // Attempt to revert scheduling for this region.
  void revertScheduling();

  void advanceRegion() { RegionIdx++; }

  virtual ~GCNSchedStage() = default;
};

class OccInitialScheduleStage : public GCNSchedStage {
public:
  bool shouldRevertScheduling(unsigned WavesAfter) override;

  OccInitialScheduleStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

class UnclusteredHighRPStage : public GCNSchedStage {
private:
  // Save the initial occupancy before starting this stage.
  unsigned InitialOccupancy;

public:
  bool initGCNSchedStage() override;

  void finalizeGCNSchedStage() override;

  bool initGCNRegion() override;

  bool shouldRevertScheduling(unsigned WavesAfter) override;

  UnclusteredHighRPStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

// Retry function scheduling if we found resulting occupancy and it is
// lower than used for other scheduling passes. This will give more freedom
// to schedule low register pressure blocks.
class ClusteredLowOccStage : public GCNSchedStage {
public:
  bool initGCNSchedStage() override;

  bool initGCNRegion() override;

  bool shouldRevertScheduling(unsigned WavesAfter) override;

  ClusteredLowOccStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

/// Attempts to reduce function spilling or, if there is no spilling, to
/// increase function occupancy by one with respect to register usage by sinking
/// rematerializable instructions to their use. When the stage estimates that
/// reducing spilling or increasing occupancy is possible, it tries to
/// rematerialize as few registers as possible to reduce potential negative
/// effects on function latency.
///
/// The stage only supports rematerializing registers that meet all of the
/// following constraints.
/// 1. The register is virtual and has a single defining instruction.
/// 2. The single defining instruction is either deemed rematerializable by the
///    target-independent logic, or if not, has no non-constant and
///    non-ignorable physical register use.
/// 3  The register has no virtual register use whose live range would be
///    extended by the rematerialization.
/// 4. The register has no non-debug user in its defining region.
///
/// TODO: explain rematerialization dependencies
/// TODO: clearly state current limitations
/// TODO: mention that chains always live in a single region
///
/// In a tree of possible rematerializations, a given rematerializable register
/// is a root if and only if it has no rematerializable user that is both
/// (1) in the same region as itself and
/// (2) non-available at all points where its user would be rematerialized.
///
/// Roots identify rematerialization chains i.e., roots map 1-to-1 to chains.
/// From the root, the chain is defined as the smallest possible set of
/// rematerializable registers in the same region as the root that have to be
/// rematerialized along with the root as to not lengthen any live range.
/// Furthermore, a chain of registers is said to be rematerializable if and only
/// if no register in the chain depends on a register from another chain that
/// hasn't been rematerialized yet, unless that register is available at all
/// points where it would be needed if the chain were to be rematerialized. This
/// ensures that rematerializing a chain does not lengthen any live range.
///
/// While roots always belong to a single chain, this is not generally true of
/// non-root rematerializable registers; these belong to at least one chain, but
/// potentially more. Chains which share at least one non-root rematerializable
/// register cannot be rematerialized separately. These *chain clusters* must be
/// rematerialized all at once and are rematerializable if and only if every
/// rematerialization chain it contains is rematerializable. Chain clusters can
/// be identified and kept up-to-date using a union-find data-structure.
class PreRARematStage : public GCNSchedStage {
private:
  struct LiveRematReg {
    /// Regions in which the register is live-in/live-out/live anywhere.
    BitVector LiveIn, LiveOut, Live;
    /// The rematerializable register's lane bitmask.
    LaneBitmask Mask;

    LiveRematReg(const RematDAG::RematReg &RematReg, GCNScheduleDAGMILive &DAG);
  };

  /// Determines if the register is both unused and live-through in region \p
  /// I. This guarantees that rematerializing it will reduce RP in the region.
  bool isUnusedLiveThrough(unsigned RegIdx, unsigned RegionIdx) const {
    assert(RegIdx < RematRegs.size() && "register index out of range");
    assert(RegionIdx < DAG.Regions.size() && "region index out of range");
    const LiveRematReg &LRR = RematRegs[RegIdx];
    return LRR.LiveIn[RegionIdx] && LRR.LiveOut[RegionIdx] &&
           !RDAG.getReg(RegIdx).Uses.contains(RegionIdx);
  }

  struct LiveRematChain {
    /// Regions in which at least one register in the chain is live.
    BitVector Live;
    /// Predicted RP difference per impacted region induced by rematerializing
    /// the chain. Impacted regions are exactly the same as the bits set in \ref
    /// Live.
    DenseMap<unsigned, GCNRPTarget::RPDiff> RPDiffs;
    /// Subset of \ref Live regions in which the computed RP difference is
    /// approximate.
    BitVector ApproximateDiff;
    /// Subset of \ref Live regions and \ref ApproximateDiff in which the
    /// overall effect of rematerializing the chain can have a negative effect
    /// on RP.
    BitVector Penalized;

    LiveRematChain(const RematDAG::RematChain &RematChain,
                   const PreRARematStage &Stage);

    /// Determines whether this rematerialization may be beneficial in at least
    /// one target region.
    bool maybeBeneficial(const BitVector &TargetRegions,
                         ArrayRef<GCNRPTarget> RPTargets,
                         const BitVector &PenalizedRegions) const;
  };

  /// A scored rematerialization candidate. Higher scores indicate more
  /// beneficial rematerializations. A null score indicate the rematerialization
  /// is not helpful to reduce RP in target regions.
  struct ScoredRemat {
    /// The chain under consideration.
    unsigned ChainIdx;

    /// Execution frequency information required by scoring heuristics.
    struct FreqInfo {
      /// Per-region execution frequencies, normalized to minimum observed
      /// frequency. 0 when unknown.
      SmallVector<uint64_t> Regions;
      /// Maximum observed frequency, normalized to minimum observed frequency.
      uint64_t MaxFreq = 0;

      FreqInfo(MachineFunction &MF, const GCNScheduleDAGMILive &DAG);
    };

    /// This only initializes state-independent characteristics of the score.
    ScoredRemat(unsigned ChainIdx, const FreqInfo &Freq,
                const PreRARematStage &Stage);

    /// Updates the rematerialization's score w.r.t. the current \p
    /// TargetRegions and \p RPTargets.
    void update(const BitVector &TargetRegions, ArrayRef<GCNRPTarget> RPTargets,
                const FreqInfo &Freq, const PreRARematStage &Stage);

    /// Returns whether the current score is null, indicating the
    /// rematerialization is useless.
    bool hasNullScore() const { return !MaxFreq && !RegionImpact; }

    /// For each pair of candidates the most important scoring component with
    /// non-equal values determine the result of the comparison (higher is
    /// better).
    bool operator<(const ScoredRemat &O) const {
      if (hasNullScore())
        return true;
      if (O.hasNullScore())
        return false;
      if (MaxFreq != O.MaxFreq)
        return MaxFreq < O.MaxFreq;
      if (FreqDiff != O.FreqDiff)
        return FreqDiff < O.FreqDiff;
      if (RegionImpact != O.RegionImpact)
        return RegionImpact < O.RegionImpact;
      // Break ties using chain index.
      return ChainIdx > O.ChainIdx;
    }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    Printable print() const;
#endif

  private:
    SmallVector<unsigned, 4> NumRegsPerReg;

    // The three members below are the scoring components, top to bottom from
    // most important to least important when comparing candidates.

    /// Frequency of impacted target region with highest known frequency. This
    /// only matters when the stage is trying to reduce spilling, so it is
    /// always 0 when it is not.
    uint64_t MaxFreq;
    /// Frequency difference between defining and using regions. Negative values
    /// indicate we are rematerializing to higher frequency regions; positive
    /// values indicate the contrary.
    int64_t FreqDiff;
    /// Expected number of target regions impacted by the rematerialization,
    /// scaled by the size of the register being rematerialized.
    unsigned RegionImpact;

    unsigned getNumRegs(const GCNScheduleDAGMILive &DAG) const;

    int64_t getFreqDiff(const FreqInfo &Freq, const RematDAG &RDAG) const;
  };

  /// A rematerialization decision, and enough information to roll it back.
  struct RematDecision {
    /// The rematerialization chain under consideration.
    const unsigned ChainIdx;
    /// Maps each register index part of a rematerialization decision to its
    /// rematerializations, themselves mapped from their containing region.
    DenseMap<unsigned, DenseMap<unsigned, Register>> NewRegs;

    /// Creates new virtual registers needed to rematerialize the chain.
    RematDecision(unsigned ChainIdx, MachineRegisterInfo &MRI,
                  const RematDAG &RDAG);
  };

  /// Parent MBB to each region, in region order.
  SmallVector<MachineBasicBlock *> RegionBB;

  /// The target occupancy the set is trying to achieve. Empty when the
  /// objective is spilling reduction.
  std::optional<unsigned> TargetOcc;
  /// Achieved occupancy *only* through rematerializations (pre-rescheduling).
  /// Smaller than or equal to the target occupancy, when it is defined.
  unsigned AchievedOcc;

  /// Rematerialization DAG.
  RematDAG RDAG;
  /// List of rematerializable registers.
  SmallVector<LiveRematReg, 4> RematRegs;
  /// List of rematerializable registers.
  SmallVector<LiveRematChain, 4> RematChains;
  /// List of rematerializations to rollback if rematerialization does not end
  /// up being beneficial.
  SmallVector<RematDecision> Rollbacks;
  /// After successful stage initialization, indicates which regions should be
  /// rescheduled.
  BitVector RescheduleRegions;

  /// Determines the stage's objective (increasing occupancy or reducing
  /// spilling, set in \ref TargetOcc). Defines \ref RPTargets in all regions to
  /// achieve that objective and mark those that don't achieve it in \ref
  /// TargetRegions. Returns whether there is any target region.
  bool setObjective(BitVector &TargetRegions,
                    SmallVectorImpl<GCNRPTarget> &RPTargets);

  /// Unsets target regions in \p Regions whose RP target has been reached.
  void unsetSatisifedRPTargets(BitVector &TargetRegions,
                               ArrayRef<GCNRPTarget> RPTargets,
                               const BitVector &RegionsToCheck);

  /// Fully recomputes RP from the DAG in \p Regions. Among those regions, sets
  /// again all \ref TargetRegions that were optimistically marked as satisfied
  /// but are actually not, and returns whether there were any such regions.
  bool updateAndVerifyRPTargets(BitVector &TargetRegions,
                                SmallVectorImpl<GCNRPTarget> &RPTargets,
                                const BitVector &RegionsToCheck);

  /// Rematerializes \p Chain and removes all rematerialized registers part of
  /// the chain from live-in/out lists in the DAG. Fills \p Rollback with
  /// required information to be able to rollback the rematerialization
  /// post-rescheduling.
  void rematerialize(const RematDecision &Remat) const;

  /// Rolls back the rematerialization decision represented by \p Rollback and
  /// live-in/out lists in the DAG.
  void rollback(const RematDecision &Remat) const;

  /// If remat alone did not increase occupancy to the target one, rolls back
  /// all rematerializations and resets live-ins/RP in all regions impacted by
  /// the stage to their pre-stage values.
  void finalizeGCNSchedStage() override;

public:
  bool initGCNSchedStage() override;

  bool initGCNRegion() override;

  bool shouldRevertScheduling(unsigned WavesAfter) override;

  PreRARematStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG),
        RDAG(DAG.Regions, DAG.MRI, *DAG.TRI, *DAG.TII, *DAG.LIS),
        RescheduleRegions(DAG.Regions.size()) {
    RegionBB.reserve(DAG.Regions.size());
  }
};

class ILPInitialScheduleStage : public GCNSchedStage {
public:
  bool shouldRevertScheduling(unsigned WavesAfter) override;

  ILPInitialScheduleStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

class MemoryClauseInitialScheduleStage : public GCNSchedStage {
public:
  bool shouldRevertScheduling(unsigned WavesAfter) override;

  MemoryClauseInitialScheduleStage(GCNSchedStageID StageID,
                                   GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

class GCNPostScheduleDAGMILive final : public ScheduleDAGMI {
private:
  std::vector<std::unique_ptr<ScheduleDAGMutation>> SavedMutations;

  bool HasIGLPInstrs = false;

public:
  void schedule() override;

  void finalizeSchedule() override;

  GCNPostScheduleDAGMILive(MachineSchedContext *C,
                           std::unique_ptr<MachineSchedStrategy> S,
                           bool RemoveKillFlags);
};

} // End namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNSCHEDSTRATEGY_H
