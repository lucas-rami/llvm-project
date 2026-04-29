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
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/Rematerializer.h"

namespace llvm {

using RegionBoundaries = Rematerializer::RegionBoundaries;

struct RegionFrequency {
  /// Per-region execution frequencies, normalized to minimum observed
  /// frequency. 0 when unknown.
  /// FIXME: normalize to e.g. 100 to increase resolution (already upstream).
  SmallVector<uint64_t> Frequencies;
  /// Maximum observed frequency, normalized to minimum observed frequency.
  uint64_t MaxFreq = 0;

  RegionFrequency(const MachineFunction &MF, const MachineLoopInfo &MLI,
                  const Rematerializer &Remater);

  inline uint64_t getFrequencyOrDefault(unsigned RegionIdx,
                                        uint64_t Default = 0) const {
    uint64_t RegionFreq = Frequencies[RegionIdx];
    return RegionFreq ? RegionFreq : Default;
  }

  inline uint64_t getFrequencyOrMax(unsigned RegionIdx) const {
    return getFrequencyOrDefault(RegionIdx, MaxFreq);
  }
};

struct RegionTargets {
  /// RP targets at live-ins and live-outs of each region.
  SmallVector<GCNRPTarget> RPLiveIn, RPLiveOut;
  /// Whether RP is above target at the live-ins and/or live-outs of each
  /// region.
  BitVector ExcessLiveIn, ExcessLiveOut;

  RegionTargets(unsigned NumRegions)
      : ExcessLiveIn(NumRegions), ExcessLiveOut(NumRegions) {
    RPLiveIn.reserve(NumRegions);
    RPLiveOut.reserve(NumRegions);
  }
};

class RematForRPTarget final {
public:
  const MachineFunction &MF;
  const LiveIntervals &LIS;

  RematForRPTarget(Rematerializer &Remater, unsigned MaxSGPRs,
                   unsigned MaxVGPRs, const MachineLoopInfo &MLI,
                   SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveIns,
                   SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveOuts);

  bool rematerialize(BitVector &AffectedRegions, bool RematForSpillAvoidance);

  void rollback();

  const Rematerializer &getRematerializer() const { return Remater; }

  unsigned getNumRegions() const { return Remater.getNumRegions(); }

  bool hasTargetRegions() const {
    return Targets.ExcessLiveIn.any() || Targets.ExcessLiveOut.any();
  }

  const GCNRegPressure &getRegionLiveInPressure(unsigned RegionIdx) const {
    assert(RegionIdx < getNumRegions() && "region out of bounds");
    return Targets.RPLiveIn[RegionIdx].getCurrentRP();
  }

private:
  using RegisterIdx = Rematerializer::RegisterIdx;

  /// Rematerializer.
  Rematerializer &Remater;
  /// Current live-ins and live-outs for each region.
  SmallVectorImpl<GCNRPTracker::LiveRegSet> &LiveIns, &LiveOuts;
  /// Maintains per-region RP targets.
  RegionTargets Targets;
  /// Maintains per-region frequency information.
  RegionFrequency FI;

  SmallVector<GCNRPTracker::LiveRegSet> StartLiveIns, StartLiveOuts;

  // void saveAtBoundaries(const GCNRegPressure &Save, const BitVector
  // &LiveInSave,
  //                       const BitVector &LiveOutSave);

  void removeFromLiveMaps(Register Reg, LaneBitmask Mask,
                          const BitVector &LiveIn, const BitVector &LiveOut);

  void addToLiveMaps(Register Reg, LaneBitmask Mask, const BitVector &LiveIn,
                     const BitVector &LiveOut);

  Printable printTargetRegions() const;
};

} // namespace llvm
