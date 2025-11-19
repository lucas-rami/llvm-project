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
  /// Single MI defining the rematerializable register. Set to nullptr after
  /// full rematerialization.
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

    void addUser(MachineInstr *NewUser, const LiveIntervals &LIS);

    void eraseUser(MachineInstr *DeletedUser, const LiveIntervals &LIS);
  };
  /// Uses of the register, mapped by region.
  SmallDenseMap<unsigned, RegionUses, 2> Uses;

  /// Returns the rematerializable register from its defining instruction.
  inline Register getDefReg() const {
    assert(DefMI && "register was fully rematerialized");
    return DefMI->getOperand(0).getReg();
  }

  /// A register read by \ref DefMI which the register depends on to determine
  /// its value.
  struct Dependency {
    unsigned MOIdx;
    unsigned RegIdx;

    Dependency(unsigned MOIdx, unsigned RegIdx)
        : MOIdx(MOIdx), RegIdx(RegIdx) {}
  };
  /// This register's dependencies, one per unique rematerializable register
  /// operand.
  SmallVector<Dependency, 2> Dependencies;
  SmallVector<unsigned, 2> UnrematableOprds;

  /// Keep track of rematerialized versions of this register, per-region.
  DenseMap<unsigned, SmallVector<unsigned, 2>> Remats;
  /// If this register was rematerialized from another one, remember it.
  std::optional<unsigned> Parent;

  inline bool isFullyRematerializable() const {
    return DefMI && !Uses.contains(DefRegion);
  }

  inline bool isUsefulToRematerialize() const {
    if (!DefMI || Uses.empty())
      return false;
    return Uses.size() > 1 || !Uses.contains(DefRegion);
  }

  Printable print(bool SkipRegions = false) const;

private:
  void addUser(MachineInstr *MI, unsigned Region, const LiveIntervals &LIS);

  void eraseUser(MachineInstr *MI, unsigned Region, const LiveIntervals &LIS);

  void addUsers(const SmallDenseSet<MachineInstr *, 4> &NewUsers,
                unsigned Region, const LiveIntervals &LIS);

  MachineBasicBlock::iterator
  getInsertPos(MachineBasicBlock::iterator InsertPos, unsigned UseRegion,
               const LiveIntervals &LIS) const;

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
        if (Visited.insert(DepIdx).second)
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

/// This only supports rematerializing registers that meet all of the
/// following constraints.
/// 1. The register is virtual and has a single defining instruction.
/// 2. The single defining instruction is either deemed rematerializable by the
///    target-independent logic, or if not, has no non-constant and
///    non-ignorable physical register use.
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

  void rematerializeToUse(unsigned RegIdx, MachineInstr *UseMI,
                          ArrayRef<unsigned> Dependencies);

  unsigned rematerializeInDefRegion(unsigned RegIdx);

  void rollback(unsigned RootIdx);

  unsigned getRematRegIdx(const MachineInstr &MI) const;

  void updateLiveIntervals();

  bool isMOAvailableAtUses(const MachineOperand &MO,
                           ArrayRef<SlotIndex> Uses) const;

  Printable print(unsigned RootIdx) const;
  Printable printID(unsigned RegIdx) const;
  Printable printRegUsers(unsigned RegIdx) const;
  Printable printUser(const MachineInstr *MI) const;

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

  struct DeadRegInfo {
    Register Reg;
    SlotIndex Slot;

    DeadRegInfo(Register Reg, SlotIndex Slot) : Reg(Reg), Slot(Slot) {}
  };
  DenseMap<unsigned, DeadRegInfo> DeadRegs;

  DenseSet<unsigned> LISUpdates;
  DenseSet<Register> UnrematLISUpdates;

  unsigned findAndMoveBestRemat(ArrayRef<unsigned> Remats,
                                MachineBasicBlock::iterator InsertPos);

  unsigned createReg(unsigned RegIdx, unsigned UseRegion,
                     SmallVectorImpl<RematReg::Dependency> &&Dependencies,
                     MachineBasicBlock::iterator InsertPos);

  unsigned rematRegInRegion(unsigned RegIdx, unsigned UseRegion,
                            MachineBasicBlock::iterator InsertPos);

  void partiallyRollbackReg(unsigned RegIdx);

  bool deleteRegIfUnused(unsigned RegIdx);

  void collectRematRegs(unsigned DefRegion);

  void substituteDependencies(unsigned FromRegIdx, unsigned ToRegIdx);

  void substituteUserReg(unsigned FromRegIdx, unsigned ToRegIdx,
                         MachineInstr &UserMI);

  /// Whether the MI is rematerializable
  bool isReMaterializable(const MachineInstr &MI) const;

  DeadRegInfo reviveDeadReg(unsigned RegIdx) {
    DeadRegInfo DeadInfo = DeadRegs.at(RegIdx);
    DeadRegs.erase(RegIdx);
    return DeadInfo;
  }

  void reMaterializeTo(unsigned NewRegIdx, MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator InsertPos, Register NewReg,
                       MachineInstr &OrigMI) {
    TII.reMaterialize(MBB, InsertPos, NewReg, 0, OrigMI, TRI);
    RematReg &Reg = Regs[NewRegIdx];
    Reg.DefMI = &*std::prev(InsertPos);
    insertMI(Reg);
    LISUpdates.insert(NewRegIdx);
  }

  void insertMI(const RematReg &Reg) {
    RegionBoundaries &Bounds = Regions[Reg.DefRegion];
    if (Bounds.first == std::next(MachineBasicBlock::iterator(Reg.DefMI)))
      Bounds.first = Reg.DefMI;
    LIS.InsertMachineInstrInMaps(*Reg.DefMI);
    MIRegion.emplace_or_assign(Reg.DefMI, Reg.DefRegion);
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

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  unsigned CallDepth = 0;
  raw_ostream &rdbgs() const {
    for (unsigned I = 0; I < CallDepth; ++I)
      dbgs() << "  ";
    return dbgs();
  }
#endif
};

} // namespace llvm