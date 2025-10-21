//=====-- GCNRematDAG.h - TODO ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// TODO
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

namespace llvm {

/// A region's boundaries i.e. a pair of instruction bundle iterators. The lower
/// boundary is inclusive, the upper boundary is exclusive.
using RegionBoundaries =
    std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>;

class RematDAG {
public:
  // A rematerializable register.
  struct RematReg {
    /// Single MI defining the rematerializable register.
    MachineInstr *DefMI;
    /// Region of defining instruction.
    unsigned DefRegion;

    /// Represents uses in a particular region.
    struct RegionUses {
      /// The latest position at which this register can be inserted into the
      /// region to be available for its uses.
      MachineBasicBlock::iterator InsertPos;
      /// Latest index at which the register should be alive in the region to be
      /// available for its uses.
      SlotIndex LastAliveIdx;
      /// List of existing users in the region.
      SmallDenseSet<MachineInstr *, 4> Users;

      RegionUses(MachineInstr *User, const LiveIntervals &LIS)
          : InsertPos(User), Users({User}) {
        LastAliveIdx = LIS.getInstructionIndex(*User);
      }

      RegionUses(const RegionUses &ParentUses, const LiveIntervals &LIS)
          : InsertPos(ParentUses.InsertPos) {
        LastAliveIdx = LIS.getInstructionIndex(*InsertPos);
      }

      void addUser(MachineInstr *NewUser, const LiveIntervals &LIS) {
        Users.insert(NewUser);
        if (LIS.getInstructionIndex(*NewUser) <
            LIS.getInstructionIndex(*InsertPos))
          InsertPos = *NewUser;
      }
    };
    using RematUses = SmallDenseMap<unsigned, RegionUses, 2>;
    /// All uses of the rematerializable register, grouped per region.
    RematUses Uses;

    /// A register read by \ref DefMI which the register depends on to determine its value.
    struct Dependency {
      unsigned MOIdx;
      std::optional<unsigned> RematIdx = std::nullopt;
      bool AvailableAtUses = true;
      // Doesn't matter if not rematerializable.
      // FIXME: not great as a member? Derivable from the dep remat itself.
      bool SameRegion;

      Dependency(unsigned MOIdx) : MOIdx(MOIdx) {}

      bool isRematInChain() const {
        return RematIdx && SameRegion && !AvailableAtUses;
      }
    };
    SmallVector<Dependency, 2> Dependencies;

    RematReg(MachineInstr *DefMI, unsigned DefRegion, RematUses &&Uses,
             SmallVectorImpl<Dependency> &&Dependencies)
        : DefMI(DefMI), DefRegion(DefRegion), Uses(std::move(Uses)),
          Dependencies(std::move(Dependencies)) {}

    /// Returns the rematerializable register from its defining instruction.
    inline Register getDefReg() const { return DefMI->getOperand(0).getReg(); }
  };

  /// A rematerializable chain of registers.
  struct RematChain {
    /// List of rematerializable registers making up the chain. The order from
    /// from to back indicates a possible rematerialization order for registers
    /// part of the chains that honor all dependencies between them, ending with
    /// the chain root.
    SmallVector<unsigned, 4> Regs;

    RematChain(unsigned RootIdx, ArrayRef<RematReg> RematRegs);

    inline unsigned root() const { return Regs.back(); }
    inline unsigned size() const { return Regs.size(); }
  };

  RematDAG(ArrayRef<RegionBoundaries> Regions, const MachineRegisterInfo &MRI,
           const TargetRegisterInfo &TRI, const TargetInstrInfo &TII,
           const LiveIntervals &LIS)
      : Regions(Regions), MRI(MRI), TRI(TRI), TII(TII), LIS(LIS) {}

  bool build();

  inline ArrayRef<RematReg> getRegs() const { return Regs; };
  inline const RematReg &getReg(unsigned Idx) const { return Regs[Idx]; };
  inline unsigned getNumRegs() const { return Regs.size(); };

  inline ArrayRef<RematChain> getChains() const { return Chains; };
  inline const RematChain &getChain(unsigned Idx) const { return Chains[Idx]; };
  inline unsigned getNumChains() const { return Chains.size(); };
  inline const RematReg &getChainRoot(unsigned ChainIdx) const {
    return Regs[getChain(ChainIdx).root()];
  }

  bool getRematRegs(SmallVectorImpl<RematReg> &RematRegs,
                    BitVector &ChainRoots);

  Printable printReg(unsigned RegIdx, bool SkipRegions=false) const;
  Printable printChain(unsigned ChainIdx, bool RootOnly=true) const;

private:
  ArrayRef<RegionBoundaries> Regions;
  const MachineRegisterInfo &MRI;
  const TargetRegisterInfo &TRI;
  const TargetInstrInfo &TII;
  const LiveIntervals &LIS;

  /// Maps all MIs (except lone terminators, which are not part of any region)
  /// to their parent region. Non-lone terminators are considered part of the
  /// region they delimitate.Maps all MIs to their containing region.
  DenseMap<MachineInstr *, unsigned> MIRegion;
  /// Candidate rematerializable registers, mapped to their index in the
  /// underlying vector.
  MapVector<Register, unsigned> Candidates;
  /// The candidates we think are rematerializable, and those which we think
  /// are chain roots.
  BitVector Rematable, CandRoots;
  /// For each candidate (i.e., at each corresponding vector index), mark
  /// other candidates which become unrematerializable if the former turns out
  /// to be unrematerializable.
  SmallVector<BitVector> InvDeps;
  /// During DAG construction, holds data for a candidate register. 
  struct CandInfo {
    RematReg::RematUses Uses;
    SmallVector<RematReg::Dependency> Dependencies;
    bool AvailableAtUses = true;
  };
  /// Candidate index to data.
  DenseMap<unsigned, CandInfo> CandData;

  // Rematerializable registers.
  SmallVector<RematReg, 8> Regs;
  // Rematerializable chains.
  SmallVector<RematChain, 8> Chains;

  /// Whether the MI is rematerializable
  bool isReMaterializable(const MachineInstr &MI) const;

  bool isMOAvailableAtUses(const MachineOperand &MO,
                           ArrayRef<SlotIndex> Uses) const;

  /// Invalidates candidate \p I and all those that transitively depend on it.
  void invalidate(unsigned I);

  std::pair<CandInfo &, bool> getOrInitCandInfo(unsigned CandIdx);

  void collectCandidates();

  /// Rematerializing a register makes it dead in its original defining
  /// region. If a candidate has uses in its own region that are not
  /// rematerializable as well, then it cannot be rematerialized.
  void sameRegionFiltering();

  /// Rematerialization should never extend a live range. If a
  /// rematerialization chain depends on a non-rematerializable register that
  /// is not available at all points where it would need to be used by
  /// registers in the chain were the latter rematerialized, then the whole
  /// chain is unrematerializable.
  void liveRangeFiltering();

  bool validateCandLiveRange(unsigned I, std::optional<unsigned> ParentIdx);

  /// All the current candidates could potentially be rematerialized but for
  /// the sake of simplifying the rematerialization logic for now we eliminate
  /// chains that share registers with other chains (this removes the need for
  /// chain clustering), or chains whose registers use registers from other
  /// chains (this removes the need to track dependencies between chains and
  /// update users as rematerializations are being performed).
  void dependentClusterFiltering();

  Register getCandidateReg(unsigned CandIdx) const {
    return Candidates.getArrayRef()[CandIdx].first;
  }

  MachineInstr *getDefMI(Register Reg) const {
    return MRI.getOneDef(Reg)->getParent();
  }

  void clear();

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  SmallVector<unsigned, 4> Path;
  void enterRec(unsigned DepIdx) { Path.push_back(DepIdx); };
  void exitRec(unsigned DepIdx) { Path.pop_back(); };

  raw_ostream &rdbgs() const {
    for (unsigned I : drop_end(Path))
      dbgs() << '(' << I << ") -> ";
    dbgs() << '(' << Path.back() << ") ";
    return dbgs();
  };
#endif
};
} // namespace llvm