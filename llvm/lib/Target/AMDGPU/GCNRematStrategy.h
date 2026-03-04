//===-- GCNRematStrategy.h - Rematerialization strategy -*- C++ -*---------===//
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

#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/Rematerializer.h"

namespace llvm {

using RegionBoundaries = Rematerializer::RegionBoundaries;

class RematAcrossForRPTarget final {
public:
  RematAcrossForRPTarget(MachineFunction &MF,
                         SmallVectorImpl<RegionBoundaries> &Regions,
                         LiveIntervals &LIS, unsigned MaxSGPRs,
                         unsigned MaxVGPRs, bool SupportRollback,
                         const MachineLoopInfo &MLI,
                         SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveIns,
                         SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveOuts);

  bool performRematerializations(BitVector &AffectedRegions,
                                 bool RematForSpillAvoidance);

  void commitRematerializations();

  void rollbackRematerializations();

  const Rematerializer &getRematerializer() const { return Remater; }

  unsigned getNumRegions() const { return Remater.getNumRegions(); }

  bool isTargetRegion(unsigned RegionIdx) const {
    assert(RegionIdx < getNumRegions() && "region out of bounds");
    return TargetRegions[RegionIdx];
  }

  const BitVector &getTargetRegions() const { return TargetRegions; }

  const GCNRPTarget &getTarget(unsigned RegionIdx) const {
    assert(RegionIdx < getNumRegions() && "region out of bounds");
    return RPTargets[RegionIdx];
  }

  const GCNRegPressure &getRegionPressure(unsigned RegionIdx) const {
    assert(RegionIdx < getNumRegions() && "region out of bounds");
    return RPTargets[RegionIdx].getCurrentRP();
  }

  bool satisfiedAtLiveIn(unsigned I) const {
    return !RPExcessAtBounds[I].first;
  }

  bool satisfiedAtLiveOut(unsigned I) const {
    return !RPExcessAtBounds[I].second;
  }

  GCNRegPressure getRealRegPressure(unsigned RegionIdx) const;

  void checkLivenessInRegions(Register Reg, BitVector &LiveIn,
                              BitVector &LiveOut) const;

  inline uint64_t getFrequencyOrDefault(unsigned RegionIdx,
                                        uint64_t Default = 0) const {
    uint64_t RegionFreq = Frequencies[RegionIdx];
    return RegionFreq ? RegionFreq : Default;
  }

  inline uint64_t getFrequencyOrMax(unsigned RegionIdx) const {
    return getFrequencyOrDefault(MaxFreq);
  }

  Printable printTargetRegions() const;

private:
  using RegisterIdx = Rematerializer::RegisterIdx;

  const MachineFunction &MF;
  const bool SupportRollback;

  /// Rematerializer.
  Rematerializer Remater;
  /// Set bits indicate regions in which RP is currently above target.
  BitVector TargetRegions;
  /// RP targets for all regions.
  SmallVector<GCNRPTarget> RPTargets;
  /// Whether RP is above target at the live-ins (first element) and/or
  /// live-outs (second element) of each region.
  SmallVector<std::pair<bool, bool>> RPExcessAtBounds;
  /// Current live-ins and live-outs for each region.
  SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveIns, &LiveOuts;

  /// Per-region execution frequencies, normalized to minimum observed
  /// frequency. 0 when unknown.
  /// FIXME: normalize to e.g. 100 to increase resolution (already upstream).
  SmallVector<uint64_t> Frequencies;
  /// Maximum observed frequency, normalized to minimum observed frequency.
  uint64_t MaxFreq = 0;

  /// Rollbacker.
  std::unique_ptr<Rollbacker> Rollback;
  /// Indices of registers removed from live-maps during rematerialization,
  /// mapped to the original regions in which they were live-in/live-out.
  DenseMap<unsigned, std::pair<BitVector, BitVector>> RemovedFromLiveMaps;

  void updateRegionBounds(unsigned RegionIdx);
  
  /// Clears target regions in \p Regions whose RP target has been reached.
  /// Returns whether any target region was cleared.
  bool clearIfSatisfied(const BitVector &Regions);

  /// Fully recomputes RP from the DAG in \p Regions. Among those regions,
  /// sets again all target regions that were optimistically marked as
  /// satisfied but are actually not, and returns whether there were any such
  /// regions.
  bool verify(const BitVector &Regions);


  void removeFromLiveMaps(RegisterIdx RegIdx, Register DefReg,
                          const BitVector &LiveIn, const BitVector &LiveOut);

  void addToLiveMaps(Register Reg, LaneBitmask Mask, const BitVector &LiveIn,
                     const BitVector &LiveOut);
};

} // namespace llvm
