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

#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <iterator>

namespace llvm {

/// A region's boundaries i.e. a pair of instruction bundle iterators. The lower
/// boundary is inclusive, the upper boundary is exclusive.
using RegionBoundaries =
    std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>;

class RematDAG;

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
  /// Single MI defining the rematerializable register. Set to nullptr after full rematerialization.
  MachineInstr *DefMI;
  /// Region of \p DefRegion.
  unsigned DefRegion;
  /// The rematerializable register's lane bitmask.
  LaneBitmask Mask;

  /// Represents uses in a particular region.
  struct RegionUses {
    /// The latest position at which this register can be inserted into the
    /// region to be available for its uses.
    MachineBasicBlock::iterator InsertPos;
    /// List of existing users in the region.
    SmallDenseSet<MachineInstr *, 4> Users;

    RegionUses(MachineInstr *User, const LiveIntervals &LIS)
        : InsertPos(User), Users({User}) {}

    void addUser(MachineInstr *NewUser, const LiveIntervals &LIS) {
      Users.insert(NewUser);
      if (Users.empty() || LIS.getInstructionIndex(*NewUser) <
                               LIS.getInstructionIndex(*InsertPos))
        InsertPos = *NewUser;
    }

    void eraseUser(MachineInstr *DeletedUser, const LiveIntervals &LIS) {
      assert(Users.contains(DeletedUser));
      Users.erase(DeletedUser);
      SlotIndex IPSlot = LIS.getInstructionIndex(*InsertPos);
      if (LIS.getInstructionIndex(*DeletedUser) == IPSlot) {
        // Recompute earliest insert position when the deleted user was the
        // earliest one.
        if (!Users.empty()) {
          auto UsersIt = Users.begin();
          InsertPos = *UsersIt;
          for (auto E = Users.end(); UsersIt != E; ++UsersIt)
            if (LIS.getInstructionIndex(**UsersIt) <
                LIS.getInstructionIndex(*InsertPos))
              InsertPos = *UsersIt;
        }
      }
    }
  };
  /// Uses of the register outside its region, mapped by region.
  SmallDenseMap<unsigned, RegionUses, 2> Uses;
  /// Users of the register inside its defining region.
  SmallDenseSet<MachineInstr *, 4> DefRegionUsers;

  /// Returns the rematerializable register from its defining instruction.
  /// Illegal to call after rematerializing the register, but legal again
  /// after rolling back the rematerialization.
  inline Register getDefReg() const { return DefMI->getOperand(0).getReg(); }

  /// A register read by \ref DefMI which the register depends on to determine
  /// its value.
  struct Dependency {
    unsigned MOIdx;
    bool Available;
    unsigned RegIdx;

    Dependency(unsigned MOIdx, bool Available, unsigned RegIdx)
        : MOIdx(MOIdx), Available(Available), RegIdx(RegIdx) {}
  };
  /// This register's dependencies, one per unique rematerializable register
  /// operand.
  SmallVector<Dependency, 2> Dependencies;

  /// Maps regions in which the register was rematerialized to the register
  /// index that corresponds to this rematerialization.
  DenseMap<unsigned, unsigned> Remats;

  inline bool isUsefulToRematerialize() const { return DefMI && !Uses.empty(); }

  inline bool isFullyRematerializable() const {
    return DefMI && DefRegionUsers.empty();
  }

  Printable print(bool SkipRegions = false) const;

private:
  void rematTo(RematReg &Remat, unsigned UseRegion,
               MachineBasicBlock::iterator &InsertPos,
               const LiveIntervals &LIS);

  void addUser(MachineInstr *MI, unsigned Region, const LiveIntervals &LIS);

  void eraseUser(MachineInstr *MI, unsigned Region, const LiveIntervals &LIS);

  void addUsers(const SmallDenseSet<MachineInstr *, 4> &NewUsers,
                unsigned Region, const LiveIntervals &LIS);

  friend RematDAG;
};

class ChainIterator {
  const ArrayRef<RematReg> Regs;

  SmallDenseSet<unsigned, 4> Visited;
  SmallVector<std::pair<unsigned, int>> Path;
  unsigned RegIdx;

  void advance() {
    assert((Path.size() > 1 || Path.front().second != -1) &&
           "trying to advance past end");
    if (Path.back().second == -1)
      Path.pop_back();
    nextInPreOrder();
  }

  void nextInPreOrder() {
    // Move "vertically" in the DAG to the deepest dependency that hasn't
    // been visited yet.
    while (true) {
      RegIdx = Path.back().first;
      int &DepIdx = Path.back().second;
      const RematReg &Reg = Regs[RegIdx];
      int NumDeps = Reg.Dependencies.size();
      // Move "horizontally" in the DAG through the dependencies until we find
      // one that is part of the chain and hasn't been visited yet.
      for (; DepIdx < NumDeps; ++DepIdx) {
        if (!Reg.Dependencies[DepIdx].Available &&
            Visited.insert(DepIdx).second)
          break;
      }
      if (DepIdx == NumDeps) {
        DepIdx = -1;
        return;
      }
      Path.push_back({DepIdx, 0});
    }
  }

public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = unsigned;
  using pointer = const value_type *;
  using reference = value_type;

  ChainIterator(unsigned RootIdx, ArrayRef<RematReg> Regs)
      : Regs(Regs), RegIdx(RootIdx) {
    Path.push_back({RootIdx, 0});
    nextInPreOrder();
  }

  ChainIterator(unsigned RootIdx) : Regs({}), RegIdx(RootIdx) {
    Path.push_back({RootIdx, -1});
  }

  ChainIterator(const ChainIterator &) = default;

  ChainIterator operator++(int) {
    auto Prev = *this;
    advance();
    return Prev;
  }

  ChainIterator &operator++() {
    advance();
    return *this;
  }

  unsigned operator*() const { return RegIdx; }

  bool operator==(const ChainIterator &Other) const {
    if (RegIdx != Other.RegIdx || Path.size() != Other.Path.size())
      return false;
    for (const auto &[Node, OtherNode] : zip_equal(Path, Other.Path)) {
      if (Node != OtherNode)
        return false;
    }
    return true;
  }

  bool operator!=(const ChainIterator &Other) const {
    return !(*this == Other);
  }
};

class RematDAG {
public:
  RematDAG(SmallVectorImpl<RegionBoundaries> &Regions, bool RegionsTopDown,
           MachineRegisterInfo &MRI, LiveIntervals &LIS, MachineFunction &MF,
           const TargetRegisterInfo &TRI, const TargetInstrInfo &TII)
      : Regions(Regions), MRI(MRI), LIS(LIS), MF(MF), TRI(TRI), TII(TII),
        RegionsTopDown(RegionsTopDown) {}

  bool build();

  inline ArrayRef<RematReg> getRegs() const { return Regs; };
  inline const RematReg &getReg(unsigned Idx) const { return Regs[Idx]; };
  inline unsigned getNumRegs() const { return Regs.size(); };

  inline unsigned getRegion(const MachineInstr *MI) const {
    return MIRegion.at(MI);
  }

  ChainIterator chainBegin(unsigned RootIdx) const {
    return ChainIterator(RootIdx, Regs);
  }
  ChainIterator chainEnd(unsigned RootIdx) const {
    return ChainIterator(RootIdx);
  }
  iterator_range<ChainIterator> getChain(unsigned RootIdx) const {
    return make_range(chainBegin(RootIdx), chainEnd(RootIdx));
  }

  unsigned rematerialize(unsigned RootIdx);

  void rollback(unsigned RootIdx);

  unsigned getRematRegIdx(const MachineInstr &MI) const;

  Printable print(unsigned RootIdx, bool SkipRoot=false) const;

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

  DenseSet<unsigned> NewRegs;
  DenseSet<unsigned> RecomputeLiveIntervals;

  /// Doesn't create any new register, just moves MIs within RegIdx's defining
  /// region.
  bool moveEarlierInDefRegion(unsigned RegIdx,
                              MachineBasicBlock::iterator InsertPos);

  unsigned rematRegInRegion(unsigned RegIdx, unsigned UseRegion,
                            MachineBasicBlock::iterator InsertPos);

  void partiallyRollbackReg(unsigned RegIdx);

  void rollbackReg(unsigned RegIdx);

  void deleteRegIfUnused(unsigned RegIdx);

  void collectRematRegs(unsigned DefRegion);

  void substituteUserReg(MachineInstr &UserMI, unsigned FromIdx,
                         unsigned ToIdx);

  void substituteRegDependencies(unsigned FromIdx, unsigned ToIdx);

  void updateLiveIntervals();

  /// Whether the MI is rematerializable
  bool isReMaterializable(const MachineInstr &MI) const;

  void insertMI(unsigned RegionIdx, MachineInstr *MI) {
    RegionBoundaries &Bounds = Regions[RegionIdx];
    if (Bounds.first == std::next(MachineBasicBlock::iterator(MI)))
      Bounds.first = MI;
    LIS.InsertMachineInstrInMaps(*MI);
    MIRegion.emplace_or_assign(MI, RegionIdx);
  }

  void deleteMI(unsigned RegionIdx, MachineInstr *MI) {
    // It is not possible for the deleted instruction to be the upper region
    // boundary since we don't rematerialize region terminators.
    if (Regions[RegionIdx].first == MI)
      Regions[RegionIdx].first = std::next(MachineBasicBlock::iterator(MI));
    LIS.RemoveMachineInstrFromMaps(*MI);
    MI->eraseFromParent();
    MIRegion.erase(MI);
  }
};

} // namespace llvm