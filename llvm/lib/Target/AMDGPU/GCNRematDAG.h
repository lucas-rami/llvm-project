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

#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

namespace llvm {

/// A region's boundaries i.e. a pair of instruction bundle iterators. The lower
/// boundary is inclusive, the upper boundary is exclusive.
using RegionBoundaries =
    std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>;

class RematDAG {
public:
  /// A rematerializable register.
  ///
  /// A rematerializable register has a list of dependencies, which correspond
  /// to the unique read register operands of its defining instruction.
  /// Dependencies can themselves be rematerializable, in which case they are
  /// stored in the \ref Dependencies vector. Unrematerializable dependencies
  /// are not represented explicitly, and corresponding read registers are
  /// considered to be both live and have the same value at all of the defined
  /// register's uses (otherwise the register cannot be rematerialized, see
  /// below).
  ///
  /// A register is partially rematerializable (i.e., rematerializable to all
  /// its non-defining using regions *but not* deletable from its defining
  /// region) if and only if all of its dependencies are either (1) available at
  /// all its direct uses or (2) partially rematerializable along with it as
  /// well. A register is fully rematerializable (i.e., rematerializable to all
  /// its non-defining using regions *and* deletable from its defining region)
  /// if and only if it is (1) partially rematerializable and (2) has no
  /// non-rematerializable use in its defining region. Importantly, only partial
  /// rematerializability can be determined from the register alone. Full
  /// rematerializability can in the general case only be determined when
  /// looking at the register through the lens of a chain.
  struct RematReg {
    /// Single MI defining the rematerializable register.
    MachineInstr *DefMI;
    /// Region of defining instruction.
    unsigned DefRegion;
    /// The rematerializable register's lane bitmask.
    LaneBitmask Mask;

    /// Represents uses in a particular region.
    struct RegionUses {
      /// The latest position at which this register can be inserted into the
      /// region to be available for its uses.
      MachineBasicBlock::iterator InsertPos;
      /// List of existing users in the region.
      SmallVector<MachineInstr *, 4> Users;

      RegionUses(MachineInstr *User, const LiveIntervals &LIS)
          : InsertPos(User), Users({User}) {
      }

      void addUser(MachineInstr *NewUser, const LiveIntervals &LIS) {
        Users.push_back(NewUser);
        if (LIS.getInstructionIndex(*NewUser) <
            LIS.getInstructionIndex(*InsertPos))
          InsertPos = *NewUser;
      }
    };
    /// Uses of the register outside its region, mapped by region.
    SmallDenseMap<unsigned, RegionUses, 2> Uses;
    /// Users of the register inside its defining region.
    SmallVector<MachineInstr *, 4> DefRegionUsers;

    /// Returns the rematerializable register from its defining instruction.
    /// Illegal to call after rematerializing the register, but legal again
    /// after rolling back the rematerialization.
    inline Register getDefReg() const { return DefMI->getOperand(0).getReg(); }

    // ==== Chain-related information ====

    /// A register read by \ref DefMI which the register depends on to determine
    /// its value.
    struct Dependency {
      unsigned MOIdx;
      unsigned RegIdx;

      Dependency(unsigned MOIdx, unsigned RegIdx)
          : MOIdx(MOIdx), RegIdx(RegIdx) {}
    };
    SmallVector<Dependency, 2> Dependencies;

    /// Bits correspond to register indices of registers part of the chain
    /// rooted at this register i.e., the minimal set of registers that are
    /// required to be (partially) rematerialized along with this register. This
    /// includes the index of the register itself.
    BitVector Chain;

    /// Bits correspond to register indices that correspond to the subset of
    /// registers part of the chain that will only be partially rematerialized
    /// as a result of rematerialing this register.
    BitVector PartialRemats;

    const BitVector& getRegRematRegions(unsigned RegIdx) const {
      return RematRegions.at(RegIdx);
    }

    inline iterator_range<BitVector::const_set_bits_iterator> chain() const {
      return Chain.set_bits();
    }
    inline unsigned chainSize() const { return Chain.count(); }

    Printable print(bool SkipRegions = false) const;

  private:
    /// Whether it is impossible to fully rematerialize the register.
    bool AlwaysPartialRemat = false;

    /// Maps register indices part of the chain to the list of regions they will
    /// be rematerialized in through this chain.
    DenseMap<unsigned, BitVector> RematRegions;

    /// Bits correspond to register indices of registers which are roots of
    /// chains which this register is a part of. This does not include this
    /// register's own index, since a register is always trivially part of the
    /// chain rooted at itself.
    BitVector OtherChains;

    friend RematDAG;
  };

  RematDAG(SmallVectorImpl<RegionBoundaries> &Regions, bool RegionsTopDown,
           MachineRegisterInfo &MRI, LiveIntervals &LIS, MachineFunction &MF,
           const TargetRegisterInfo &TRI, const TargetInstrInfo &TII)
      : Regions(Regions), MRI(MRI), LIS(LIS), MF(MF), TRI(TRI), TII(TII),
        RegionsTopDown(RegionsTopDown) {}

  bool build();

  inline ArrayRef<RematReg> getRegs() const { return Regs; };
  inline const RematReg &getReg(unsigned Idx) const { return Regs[Idx]; };
  inline unsigned getNumRegs() const { return Regs.size(); };

  /// For each register, determine which of the registers part of its chain
  /// will only be partially be rematerialized as part of the chain.
  BitVector getPartialRematsInChain(unsigned RegIdx) const;

  void rematerialize(unsigned RegIdx);

  void rollback(unsigned RegIdx);

  Printable print(unsigned RegIdx, bool RootOnly = true,
                  bool SkipRegions = false) const;

private:
  SmallVectorImpl<RegionBoundaries> &Regions;
  MachineRegisterInfo &MRI;
  LiveIntervals &LIS;
  MachineFunction &MF;
  const TargetRegisterInfo &TRI;
  const TargetInstrInfo &TII;
  bool RegionsTopDown;

  // DAG construction information, cleared at the end of the build process.

  /// Maps all MIs (except lone terminators, which are not part of any region)
  /// to their parent region. Non-lone terminators are considered part of the
  /// region they delimitate.Maps all MIs to their containing region.
  DenseMap<MachineInstr *, unsigned> MIRegion;
  /// Parent MBB to each region, in region order.
  SmallVector<MachineBasicBlock *> RegionBB;
  /// Rematerializable registers.
  SmallVector<RematReg, 8> Regs;
  /// Candidate rematerializable registers, mapped to their index in the
  /// underlying \ref Regs vector.
  DenseMap<Register, unsigned> RegToIdx;

  /// Maps each rematerialized register index to its rematerializations,
  /// themselves mapped from their containing region.
  DenseMap<unsigned, DenseMap<unsigned, Register>> RematerializedRegs;

  /// Whether the MI is rematerializable
  bool isReMaterializable(const MachineInstr &MI) const;

  bool isMOAvailableAtUses(const MachineOperand &MO,
                           SmallDenseMap<unsigned, SlotIndex, 4> Uses) const;

  void collectRematRegs(unsigned I);

  void setRegDependency(unsigned RegIdx, unsigned DepRegIdx);

  void insertMI(unsigned RegionIdx, MachineInstr *RematMI) {
    RegionBoundaries &Bounds = Regions[RegionIdx];
    if (Bounds.first == std::next(MachineBasicBlock::iterator(RematMI)))
      Bounds.first = RematMI;
    LIS.InsertMachineInstrInMaps(*RematMI);
  }

  void deleteMI(unsigned RegionIdx, MachineInstr *MI) {
    // It is not possible for the deleted instruction to be the upper region
    // boundary since we don't rematerialize region terminators.
    if (Regions[RegionIdx].first == MI)
      Regions[RegionIdx].first = std::next(MachineBasicBlock::iterator(MI));
    LIS.RemoveMachineInstrFromMaps(*MI);
    MI->eraseFromParent();
  }

  void clear();

  static void resizeBitVectorAndSet(BitVector &BV, unsigned I) {
    if (BV.size() < I + 1)
      BV.resize(I + 1);
    BV.set(I);
  }
};
} // namespace llvm