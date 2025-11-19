//=====-- GCNRematDAG.cpp - TODO --------------------------------*- C++ -*-===//
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

#include "GCNRematDAG.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rdag"

using namespace llvm;

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

static Register isRegDependency(const MachineOperand &MO) {
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

static SlotIndex getLastMISlot(const SmallDenseSet<MachineInstr *, 4> Instrs,
                               const LiveIntervals &LIS) {
  assert(!Instrs.empty());
  auto InstrsIt = Instrs.begin(), InstrsEnd = Instrs.end();
  SlotIndex LastSlot = LIS.getInstructionIndex(**InstrsIt);
  while (++InstrsIt != InstrsEnd)
    LastSlot = std::max(LastSlot, LIS.getInstructionIndex(**InstrsIt));
  return LastSlot;
}

unsigned RematDAG::findAndMoveBestRemat(ArrayRef<unsigned> Remats,
                                        MachineBasicBlock::iterator InsertPos) {
  assert(!Remats.empty() && "no rematerializations");

  // Find the latest rematerialization that is earlier than the desired insert
  // position.
  unsigned BestRegIdx = Remats.front();
  SlotIndex BestSlot = LIS.getInstructionIndex(*Regs[BestRegIdx].DefMI);
  SlotIndex MaxSlot = LIS.getInstructionIndex(*InsertPos);
  bool NeedMove = BestSlot > MaxSlot;
  for (unsigned RegIdx : Remats.drop_front()) {
    RematReg &Reg = Regs[RegIdx];
    assert(Regions[Reg.DefRegion].first->getParent() ==
               InsertPos->getParent() &&
           "insert pos not in same region");

    SlotIndex RegSlot = LIS.getInstructionIndex(*Reg.DefMI);
    if (MaxSlot >= RegSlot && RegSlot > BestSlot) {
      BestSlot = RegSlot;
      BestRegIdx = RegIdx;
      NeedMove = false;
    }
  }
  if (!NeedMove)
    return BestRegIdx;

  LLVM_DEBUG({
    rdbgs() << "Moving " << printID(BestRegIdx)
            << " to earlier position in defining region\n";
    ++CallDepth;
  });

  RematReg &Reg = Regs[BestRegIdx];

  // First make sure all its dependencies will be available at that point.
  for (const RematReg::Dependency &Dep : Reg.Dependencies) {
    LISUpdates.insert(Dep.RegIdx);
    if (Regs[Dep.RegIdx].DefRegion == Reg.DefRegion)
      findAndMoveBestRemat(Dep.RegIdx, InsertPos);
  }

  // Rematerialize the new instruction earlier in the region.
  MachineInstr *OrigMI = Reg.DefMI;
  reMaterializeTo(BestRegIdx, *InsertPos->getParent(), InsertPos,
                  Reg.getDefReg(), *OrigMI);

  LLVM_DEBUG(rdbgs() << "** Moved " << printID(BestRegIdx) << " to ";
             LIS.getInstructionIndex(*Reg.DefMI).print(dbgs());
             dbgs() << '\n';);

  // All rematerializable registers that this register depends on should be
  // aware that the MI has changed.
  for (const RematReg::Dependency &Dep : Reg.Dependencies) {
    RematReg &DepReg = Regs[Dep.RegIdx];
    DepReg.addUser(Reg.DefMI, Reg.DefRegion, LIS);
    DepReg.eraseUser(OrigMI, Reg.DefRegion, LIS);
    LLVM_DEBUG({
      rdbgs() << "Swapping " << printID(Dep.RegIdx) << " user:\n";
      rdbgs() << "  From: " << printUser(OrigMI) << '\n';
      rdbgs() << "  To: " << printUser(Reg.DefMI) << '\n';
    });
  }

  // Delete the original instruction.
  deleteMI(Reg.DefRegion, OrigMI);
  LISUpdates.insert(BestRegIdx);
  LLVM_DEBUG(--CallDepth);
  return BestRegIdx;
}

unsigned RematDAG::rematRegInRegion(unsigned RegIdx, unsigned UseRegion,
                                    MachineBasicBlock::iterator InsertPos) {
  assert(Regs[RegIdx].DefRegion != UseRegion && "cannot remat in def region");

  // Simply reuse a rematerialized version of the register in the region if one
  // already exists.
  const auto &RegRemats = Regs[RegIdx].Remats;
  if (auto Remat = RegRemats.find(UseRegion); Remat != RegRemats.end())
    return findAndMoveBestRemat(Remat->second, InsertPos);

  LLVM_DEBUG({
    rdbgs() << "Rematerializing " << printID(RegIdx) << " to [" << UseRegion
            << ']';
    if (LIS.hasInterval(Regs[RegIdx].getDefReg())) {
      dbgs() << ": ";
      LIS.getInterval(Regs[RegIdx].getDefReg()).print(dbgs());
    }
    dbgs() << '\n';
    ++CallDepth;
  });

  // If the register has its own uses in the using region, it must be
  // rematerialized before them.
  InsertPos = getReg(RegIdx).getInsertPos(InsertPos, UseRegion, LIS);

  // Make a copy of the dependencies because references to registers in Regs may
  // become invalid after adding new registers to the vector.
  SmallVector<RematReg::Dependency, 2> CurrentDeps(Regs[RegIdx].Dependencies);
  SmallVector<RematReg::Dependency, 2> NewDeps;

  // Create dependencies for the new register. This will recursively
  // rematerialize any missing dependency in the using region.
  for (const RematReg::Dependency &Dep : CurrentDeps) {
    const RematReg &DepReg = getReg(Dep.RegIdx);
    assert(DepReg.DefRegion != UseRegion && "dependency in use region");
    auto DepRematInUseRegion = DepReg.Remats.find(UseRegion);
    // Either the dependency was already rematerialized in the region and we can
    // reuse it, or it must first be rematerialized in the region itself.
    unsigned NewDepIdx =
        DepRematInUseRegion == DepReg.Remats.end()
            ? rematRegInRegion(Dep.RegIdx, UseRegion, InsertPos)
            : findAndMoveBestRemat(DepRematInUseRegion->getSecond(), InsertPos);
    NewDeps.emplace_back(Dep.MOIdx, NewDepIdx);
  }

  // Create the new register.
  const unsigned NewRegIdx =
      createReg(RegIdx, UseRegion, std::move(NewDeps), InsertPos);
  RematReg &NewReg = Regs[NewRegIdx];

  // The users of the original register in the using region become the users of
  // the new register in its defining region.
  SmallDenseMap<unsigned, RematReg::RegionUses, 2> &RegUses = Regs[RegIdx].Uses;
  if (auto RegionUses = RegUses.find(UseRegion); RegionUses != RegUses.end()) {
    for (MachineInstr *UserMI : RegionUses->getSecond().Users)
      substituteUserReg(RegIdx, NewRegIdx, *UserMI);
    NewReg.Uses.try_emplace(UseRegion, std::move(RegionUses->getSecond()));
    RegUses.erase(UseRegion);
  }

  deleteRegIfUnused(RegIdx);
  LLVM_DEBUG(--CallDepth);
  return NewRegIdx;
}

unsigned RematDAG::rematerialize(unsigned RootIdx) {
  unsigned NumRegsBeforeRemat = Regs.size();
  // Copy some register information because rematerializing can invalidate
  // references to registers.
  const RematReg &Reg = getReg(RootIdx);
  unsigned DefRegion = Reg.DefRegion;
  SmallDenseMap<unsigned, RematReg::RegionUses, 2> Uses(Reg.Uses);
  for (const auto &[UseRegion, RegionUses] : Uses) {
    if (UseRegion != DefRegion)
      rematRegInRegion(RootIdx, UseRegion, RegionUses.InsertPos);
  }
  return Regs.size() - NumRegsBeforeRemat;
}

void RematDAG::rematerializeToUse(unsigned RegIdx, MachineInstr *UseMI,
                                  ArrayRef<unsigned> Dependencies) {
  // Pre-allocate space for the new registers in the vector to avoid invalid
  // references during rematerialization.
  Regs.reserve(Regs.size() + Dependencies.size() + 1);
  unsigned UseRegion = MIRegion.at(UseMI);
  assert(Regs[RegIdx].Uses.at(UseRegion).Users.contains(UseMI) && "not a user");

  LLVM_DEBUG({
    rdbgs() << "Rematerializing " << printID(RegIdx) << " to "
            << printUser(UseMI) << '\n';
    ++CallDepth;
  });

  SmallDenseMap<unsigned, unsigned> DepRemap;
  auto RematToUse = [&](unsigned RematIdx) {
    // Create dependencies for the new register.
    SmallVector<RematReg::Dependency, 2> NewDeps;
    for (const RematReg::Dependency &Dep : Regs[RematIdx].Dependencies) {
      unsigned NewDepIdx = DepRemap.lookup_or(Dep.RegIdx, Dep.RegIdx);
      NewDeps.emplace_back(Dep.MOIdx, NewDepIdx);
    }
    return createReg(RegIdx, UseRegion, std::move(NewDeps), UseMI);
  };

  // First rematerialize dependencies (assumed to be in post-order), then the
  // register itself.
  for (unsigned DepIdx : Dependencies) {
    DepRemap.insert({DepIdx, RematToUse(DepIdx)});
    LLVM_DEBUG(rdbgs() << "Rematerialized dependency: " << printID(DepIdx)
                       << " to " << printID(DepRemap.at(DepIdx)) << '\n');
  }
  substituteUserReg(RegIdx, RematToUse(RegIdx), *UseMI);
  // FIXME: delete any of the original registers if they have no use?
  LLVM_DEBUG(--CallDepth);
}

unsigned RematDAG::rematerializeInDefRegion(unsigned RegIdx) {
  return 0;
  // RematReg &Reg = Regs[RegIdx];
  // auto DefRegionUses = Reg.Uses.find(Reg.DefRegion);
  // if (DefRegionUses == Reg.Uses.end() ||
  //     DefRegionUses->getSecond().Users.size() < 2)
  //   return 0;
  // unsigned NumRegsBeforeRemat = Regs.size();

  // // Copy the defining region uses because rematerializing can invalidate
  // // references to registers.
  // RematReg::RegionUses DefUses(DefRegionUses->getSecond());
  // for (MachineInstr *UseMI : DefUses.Users) {
  //   // The use at the insert position for the region is the earliest one by
  //   // construction. This one will continue to use the original register.
  //   if (DefUses.InsertPos != UseMI)
  //     substituteUserReg(RegIdx, rematToUser(RegIdx, UseMI), *UseMI);
  // }
  // return Regs.size() - NumRegsBeforeRemat;
}

void RematDAG::partiallyRollbackReg(unsigned RegIdx) {
  RematReg &Reg = Regs[RegIdx];
  if (Reg.DefMI) {
    // The register still exists in its defining region, nothing to do.
    LLVM_DEBUG(rdbgs() << printID(RegIdx) << " still exists\n");
    return;
  }

  LLVM_DEBUG({
    rdbgs() << "Partially rolling back " << printID(RegIdx) << '\n';
    ++CallDepth;
  });

  // We shouldn't ever call this on a register that was deleted and has no
  // active rematerialization. Such registers can arise as a result of rollback,
  // but we shouldn't try to roll them back.
  assert(!Reg.Remats.empty() && "no register and no remats");

  // The register no longer exist in its defining region, we first need to
  // re-rematerialize it in its original defining region. Any of its
  // dependencies that were fully rematerialized need to be partially rollbacked
  // first as well.
  for (const RematReg::Dependency &Dep : Reg.Dependencies) {
    LISUpdates.insert(Dep.RegIdx);
    partiallyRollbackReg(Dep.RegIdx);
    assert(getReg(Dep.RegIdx).DefMI && "dependency was not rolled back");
  }

  // Re-rematerialize MI in its original region from one of the
  // rematerializations. Note that it may not be rematerialized exactly in the
  // same position as originally within the region, but it should not matter
  // much. We cannot get the parent basic block from the region boundaries
  // because the latter may have been completely emptied out in which case both
  // bounds can point to the MBB's end guard.
  unsigned ModelRegIdx = Reg.Remats.begin()->second.front();
  RematReg &ModelReg = Regs[ModelRegIdx];
  auto [OldDefReg, OldSlot] = reviveDeadReg(RegIdx);

  // Try to re-create the original instruction as close as possible to the
  // initial position.
  SlotIndex NextSlot = LIS.getSlotIndexes()->getNextNonNullIndex(OldSlot);
  MachineInstr *NextMI = LIS.getInstructionFromIndex(NextSlot);
  MachineBasicBlock::iterator InsertPos;
  if (!NextMI || MIRegion.at(NextMI) != Reg.DefRegion)
    InsertPos = Regions[Reg.DefRegion].second;
  else
    InsertPos = NextMI;

  MachineBasicBlock &MBB = *RegionBB[Reg.DefRegion];
  reMaterializeTo(RegIdx, MBB, InsertPos, OldDefReg, *ModelReg.DefMI);
  substituteDependencies(ModelRegIdx, RegIdx);

  LISUpdates.insert(ModelRegIdx);
  for (unsigned MOIdx : Reg.UnrematableOprds)
    UnrematLISUpdates.insert(Reg.DefMI->getOperand(MOIdx).getReg());

  LLVM_DEBUG({
    rdbgs() << "** Partially rolled back " << printID(RegIdx) << " @ ";
    LIS.getInstructionIndex(*Reg.DefMI).print(dbgs());
    dbgs() << '\n';
    --CallDepth;
  });
}

void RematDAG::rollback(unsigned RootIdx) {
  LLVM_DEBUG({
    rdbgs() << "Fully rolling back " << printID(RootIdx) << '\n';
    ++CallDepth;
  });

  RematReg &Reg = Regs[RootIdx];
  // It is possible that the rematerialized registers were themselves fully
  // rematerialized. If so, we need to recursively roll them back too.
  for (const auto &[_, RematRegs] : Reg.Remats) {
    for (unsigned RematRegIdx : RematRegs)
      rollback(RematRegIdx);
  }
  partiallyRollbackReg(RootIdx);

  // Now go through all of the rematerialization's users and make them use the
  // rolled back register.
  for (const auto &[UseRegion, RematRegs] : Reg.Remats) {
    for (unsigned RematRegIdx : RematRegs) {
      RematReg &RematReg = Regs[RematRegIdx];

      // Substitute rematerialized register for rolled back one in all users,
      // and transfer users to rolled back register.
      for (const auto &[UseRegion, RegionUsers] : RematReg.Uses) {
        for (MachineInstr *UseMI : RegionUsers.Users)
          substituteUserReg(RematRegIdx, RootIdx, *UseMI);
        Reg.addUsers(RegionUsers.Users, UseRegion, LIS);
      }

      // The rematerialization will be deleted.
      RematReg.Uses.clear();
      deleteRegIfUnused(RematRegIdx);
    }
  }

  // FIXME: In theory we could leave the map as is and reuse the existing index
  // if we are to rematerialize the register again but it makes tracking things
  // more difficult and I don't expect this to be a common use case.
  Reg.Remats.clear();
  LLVM_DEBUG({
    rdbgs() << "** Fully rolled back " << printID(RootIdx) << '\n';
    --CallDepth;
  });
}

bool RematDAG::deleteRegIfUnused(unsigned RegIdx) {
  RematReg &Reg = Regs[RegIdx];
  if (!Reg.DefMI || !Reg.Uses.empty())
    return false;
  LLVM_DEBUG({
    rdbgs() << "Deleting " << printID(RegIdx) << " with no users\n";
    ++CallDepth;
  });

  Register DefReg = Reg.getDefReg();
  for (const RematReg::Dependency &Dep : Reg.Dependencies) {
    LLVM_DEBUG(rdbgs() << "Deleting user from " << printID(Dep.RegIdx) << "\n");
    Regs[Dep.RegIdx].eraseUser(Reg.DefMI, Reg.DefRegion, LIS);
    deleteRegIfUnused(Dep.RegIdx);
  }

  DeadRegs.try_emplace(RegIdx, DefReg, LIS.getInstructionIndex(*Reg.DefMI));
  LIS.removeInterval(DefReg);
  LISUpdates.erase(RegIdx);
  for (unsigned MOIdx : Reg.UnrematableOprds)
    UnrematLISUpdates.insert(Reg.DefMI->getOperand(MOIdx).getReg());

  deleteMI(Reg.DefRegion, Reg.DefMI);
  Reg.DefMI = nullptr;
  if (Reg.Parent) {
    assert(Regs[*Reg.Parent].Remats.contains(Reg.DefRegion) && "broken link");
    Regs[*Reg.Parent].Remats.erase(Reg.DefRegion);
  }
  LLVM_DEBUG(--CallDepth);
  return true;
}

void RematDAG::substituteDependencies(unsigned FromRegIdx, unsigned ToRegIdx) {
  RematReg &FromReg = Regs[FromRegIdx], &ToReg = Regs[ToRegIdx];
  LLVM_DEBUG(rdbgs() << "Substituting dependencies in "
                     << printUser(ToReg.DefMI) << ":\n");

  auto ZipedDeps = zip_equal(FromReg.Dependencies, ToReg.Dependencies);
  for (const auto &[OldDep, NewDep] : ZipedDeps) {
    assert(OldDep.MOIdx == NewDep.MOIdx && "operand mismatch");
    LLVM_DEBUG(rdbgs() << "  Operand #" << OldDep.MOIdx << ": "
                       << printID(OldDep.RegIdx) << " -> "
                       << printID(NewDep.RegIdx) << '\n');

    RematReg &NewDepReg = Regs[NewDep.RegIdx];
    if (OldDep.RegIdx != NewDep.RegIdx) {
      Register OldDefReg = FromReg.DefMI->getOperand(OldDep.MOIdx).getReg();
      ToReg.DefMI->substituteRegister(OldDefReg, NewDepReg.getDefReg(), 0, TRI);
      LISUpdates.insert(OldDep.RegIdx);
    }
    NewDepReg.addUser(ToReg.DefMI, ToReg.DefRegion, LIS);
    LISUpdates.insert(NewDep.RegIdx);
  }
}

void RematDAG::substituteUserReg(unsigned FromRegIdx, unsigned ToRegIdx,
                                 MachineInstr &UserMI) {
  LLVM_DEBUG(rdbgs() << "User transfer from " << printID(FromRegIdx) << " to "
                     << printID(ToRegIdx) << ": " << printUser(&UserMI)
                     << '\n');

  UserMI.substituteRegister(getReg(FromRegIdx).getDefReg(),
                            getReg(ToRegIdx).getDefReg(), 0, TRI);

  LISUpdates.insert(FromRegIdx);
  LISUpdates.insert(ToRegIdx);

  // If the user is rematerializable, we must change its dependency to the
  // new register.
  unsigned UserRegIdx = getRematRegIdx(UserMI);
  if (UserRegIdx == getNumRegs())
    return;
  RematReg &UserReg = Regs[UserRegIdx];

  // Look for the user dependency that matches the register.
  auto *UserDep = find_if(UserReg.Dependencies,
                          [FromRegIdx](const RematReg::Dependency &Dep) {
                            return Dep.RegIdx == FromRegIdx;
                          });
  assert(UserDep && "broken dependency");
  UserDep->RegIdx = ToRegIdx;
}

void RematDAG::updateLiveIntervals() {
  for (unsigned RegIdx : LISUpdates) {
    const RematReg &Reg = getReg(RegIdx);
    assert(Reg.DefMI && "trying to update dead register");

    Register DefReg = Reg.getDefReg();
    if (LIS.hasInterval(DefReg))
      LIS.removeInterval(DefReg);
    LIS.createAndComputeVirtRegInterval(DefReg);

    LLVM_DEBUG({
      rdbgs() << "Re-computed interval for " << printID(RegIdx) << ": ";
      LIS.getInterval(DefReg).print(dbgs());
      rdbgs() << '\n' << printRegUsers(RegIdx);
    });
  }
  for (Register Reg : UnrematLISUpdates) {
    LIS.removeInterval(Reg);
    LIS.createAndComputeVirtRegInterval(Reg);
    LLVM_DEBUG(dbgs() << "Updated interval for unrematable register "
                      << printReg(Reg) << '\n');
  }
  LISUpdates.clear();
  UnrematLISUpdates.clear();
}

bool RematDAG::isMOAvailableAtUses(const MachineOperand &MO,
                                   ArrayRef<SlotIndex> Uses) const {
  if (Uses.empty())
    return true;
  Register Reg = MO.getReg();
  unsigned SubIdx = MO.getSubReg();
  LaneBitmask Mask = SubIdx ? TRI.getSubRegIndexLaneMask(SubIdx)
                            : MRI.getMaxLaneMaskForVReg(Reg);
  const LiveInterval &LI = LIS.getInterval(Reg);
  const VNInfo *DefVN =
      LI.getVNInfoAt(LIS.getInstructionIndex(*MO.getParent()).getRegSlot(true));
  for (SlotIndex Use : Uses) {
    if (!isAvailableAtUse(DefVN, Mask, Use, LI))
      return false;
  }
  return true;
}

bool RematDAG::isReMaterializable(const MachineInstr &MI) const {
  if (!TII.isReMaterializable(MI))
    return false;

  for (const MachineOperand &MO : MI.all_uses()) {
    // We can't remat physreg uses, unless it is a constant or an ignorable
    // use (e.g. implicit exec use on VALU instructions)
    if (MO.getReg().isPhysical()) {
      if (MRI.isConstantPhysReg(MO.getReg()) || TII.isIgnorableUse(MO))
        continue;
      return false;
    }
  }

  return true;
}

bool RematDAG::build() {
  MIRegion.clear();
  RegionBB.clear();
  Regs.clear();
  RegToIdx.clear();
  if (Regions.empty())
    return false;

  // Maps each basic block number to regions that are part of the BB.
  DenseMap<unsigned, SmallVector<unsigned, 4>> RegionsPerBlock;

  const unsigned NumRegions = Regions.size();
  for (unsigned I = 0; I < NumRegions; ++I) {
    RegionBoundaries Region = Regions[I];
    for (auto MI = Region.first; MI != Region.second; ++MI)
      MIRegion.insert({&*MI, I});
    MachineBasicBlock *MBB = Region.first->getParent();
    if (Region.second != MBB->end())
      MIRegion.insert({&*Region.second, I});
    RegionBB.push_back(MBB);
    RegionsPerBlock[MBB->getNumber()].push_back(I);
  }

  // Visit regions in dominator tree pre-order to ensure that regions defining
  // registers come before regions using them.
  MachineDominatorTree MDT(MF);
  for (MachineDomTreeNode *MBB : depth_first(&MDT)) {
    auto MBBRegions = RegionsPerBlock.find(MBB->getBlock()->getNumber());
    if (MBBRegions == RegionsPerBlock.end())
      continue;
    auto MBBRegionsIt = RegionsTopDown ? MBBRegions->getSecond()
                                       : reverse(MBBRegions->getSecond());
    for (unsigned I : MBBRegionsIt)
      collectRematRegs(I);
  }

  LLVM_DEBUG({
    for (unsigned I = 0, E = getNumRegs(); I < E; ++I)
      dbgs() << print(I) << '\n';
  });
  return !Regs.empty();
}

void RematDAG::collectRematRegs(unsigned DefRegion) {
  // Collect partially rematerializable registers in instruction order within
  // each region. This guarantees that, within a single region, partially
  // rematerializable registers used in instructions defining other partially
  // rematerializable registers are visited first. This is important to
  // guarantee that all of a register's dependencies are visited before the
  // register itself.
  RegionBoundaries Bounds = Regions[DefRegion];
  for (auto MI = Bounds.first; MI != Bounds.second; ++MI) {
    // The instruction must be rematerializable.
    MachineInstr &DefMI = *MI;
    if (!isReMaterializable(DefMI))
      continue;

    // We only support rematerializing virtual registers with one definition.
    Register DefReg = DefMI.getOperand(0).getReg();
    if (!DefReg.isVirtual() || !MRI.hasOneDef(DefReg))
      continue;

    const unsigned RegIdx = Regs.size();

    // Create the new register and do some basic initialization.
    RematReg &Reg = Regs.emplace_back();
    Reg.DefMI = &DefMI;
    Reg.DefRegion = DefRegion;
    unsigned SubIdx = DefMI.getOperand(0).getSubReg();
    Reg.Mask = SubIdx ? TRI.getSubRegIndexLaneMask(SubIdx)
                      : MRI.getMaxLaneMaskForVReg(DefReg);
    LLVM_DEBUG(dbgs() << "Evaluating " << printUser(&DefMI) << '\n');

    // Collect the candidate's direct users, both rematerializable and
    // unrematerializable.
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(DefReg)) {
      auto UseRegion = MIRegion.find(&UseMI);
      if (UseRegion == MIRegion.end()) {
        // Only lone MI terminators can trigger this condition. They are not
        // part of any region so we cannot rematerialize next to them. Just
        // consider this register unrematerializable.
        Reg.Uses.clear();
        break;
      }
      Reg.addUser(&UseMI, UseRegion->second, LIS);
    }
    if (Reg.Uses.empty()) {
      LLVM_DEBUG(
          dbgs()
          << "  -> Eliminated: no non-debug users or lone terminator user\n");
      Regs.pop_back();
      continue;
    }

    // Derive slots at which dependencies must be available for this register to
    // be at least partially rematerializable. This is the latest register user
    // in each region with at least one register user.
    SmallVector<SlotIndex, 4> UseSlots;
    for (const auto &[_, RegionUses] : Reg.Uses)
      UseSlots.push_back(getLastMISlot(RegionUses.Users, LIS));

    // Collect the candidate's dependencies. If the same register is used
    // multiple times we just need to store it once.
    SmallDenseSet<Register, 4> AllDepRegs;
    for (const auto &[MOIdx, MO] : enumerate(Reg.DefMI->operands())) {
      Register DepReg = isRegDependency(MO);
      if (!DepReg || AllDepRegs.contains(DepReg))
        continue;
      AllDepRegs.insert(DepReg);

      bool Available = isMOAvailableAtUses(MO, UseSlots);
      auto DepIt = RegToIdx.find(DepReg);
      if (DepIt != RegToIdx.end()) {
        const unsigned DepRegIdx = DepIt->second;
        assert(DepRegIdx < RegIdx && "dependency should come first");
        Reg.Dependencies.emplace_back(MOIdx, DepRegIdx);
      } else if (Available) {
        Reg.UnrematableOprds.push_back(MOIdx);
      } else {
        // The dependency is both unavailable at uses and unrematerializable, so
        // the register can never be even partially rematerializable.
        LLVM_DEBUG(dbgs() << "  -> Eliminated: operand #" << MOIdx << " ("
                          << printReg(DepReg, &TRI, MO.getSubReg(), &MRI)
                          << ") unrematable and unavaialble\n");
        Regs.pop_back();
        break;
      }
    }
    if (RegIdx == Regs.size())
      continue;

    // All transitive unrematerializable dependencies in the chain must be
    // available at all of the register's uses.
    DenseSet<unsigned> VisitedDependencies;
    DenseSet<Register> UnrematRegCache;
    std::function<bool(unsigned)> CheckUnrematableDeps =
        [&](unsigned DepIdx) -> bool {
      if (!VisitedDependencies.insert(RegIdx).second)
        return true;

      // Retrieve direct unrematerializable dependencies for this dependency.
      // All of them must be available at the current candidate's uses.
      const RematReg &DepReg = Regs[DepIdx];
      for (unsigned MOIdx : DepReg.UnrematableOprds) {
        MachineOperand &MO = DepReg.DefMI->getOperand(MOIdx);
        if (UnrematRegCache.insert(MO.getReg()).second &&
            !isMOAvailableAtUses(MO, UseSlots)) {
          LLVM_DEBUG(dbgs() << "  -> Eliminated: dependency " << printID(DepIdx)
                            << " has operand register #" << MOIdx << " ("
                            << printReg(MO.getReg(), &TRI, MO.getSubReg(), &MRI)
                            << ") which is unrematable and unavailable at "
                               "candidate's uses\n");
          return false;
        }
      }

      // Recurse on the dependencies' own dependencies.
      for (const RematReg::Dependency &DepDep : DepReg.Dependencies) {
        if (!CheckUnrematableDeps(DepDep.RegIdx))
          return false;
      }
      return true;
    };

    for (const RematReg::Dependency &Dep : Reg.Dependencies) {
      if (!CheckUnrematableDeps(Dep.RegIdx)) {
        Regs.pop_back();
        break;
      }
    }
    if (RegIdx == Regs.size())
      continue;

    // The register is at least partially rematerializable.
    RegToIdx.insert({DefReg, RegIdx});
    LLVM_DEBUG(dbgs() << "  -> Valid!\n");
  }
}

unsigned RematDAG::getRematRegIdx(const MachineInstr &MI) const {
  if (!MI.getNumOperands() || !MI.getOperand(0).isReg() ||
      MI.getOperand(0).readsReg())
    return getNumRegs();
  Register Reg = MI.getOperand(0).getReg();
  auto UserRegIt = RegToIdx.find(Reg);
  if (UserRegIt == RegToIdx.end())
    return getNumRegs();
  return UserRegIt->second;
}

unsigned
RematDAG::createReg(unsigned RegIdx, unsigned UseRegion,
                    SmallVectorImpl<RematReg::Dependency> &&Dependencies,
                    MachineBasicBlock::iterator InsertPos) {
  unsigned NewRegIdx = Regs.size();
  RematReg &NewReg = Regs.emplace_back();
  RematReg &Reg = Regs[RegIdx];
  NewReg.Mask = Reg.Mask;
  NewReg.DefRegion = UseRegion;
  NewReg.UnrematableOprds = Reg.UnrematableOprds;
  NewReg.Dependencies = std::move(Dependencies);

  // Link the rematerializations together.
  Reg.Remats[UseRegion].push_back(NewRegIdx);
  NewReg.Parent = RegIdx;

  Register NewDefReg = MRI.cloneVirtualRegister(Reg.getDefReg());
  RegToIdx.insert({NewDefReg, NewRegIdx});
  reMaterializeTo(NewRegIdx, *InsertPos->getParent(), InsertPos, NewDefReg,
                  *Reg.DefMI);
  substituteDependencies(RegIdx, NewRegIdx);

  for (unsigned MOIdx : NewReg.UnrematableOprds)
    UnrematLISUpdates.insert(NewReg.DefMI->getOperand(MOIdx).getReg());

  LLVM_DEBUG({
    rdbgs() << "** Rematerialized " << printID(RegIdx) << " as "
            << printID(NewRegIdx) << " @ ";
    LIS.getInstructionIndex(*NewReg.DefMI).print(dbgs());
    dbgs() << '\n';
  });
  return NewRegIdx;
}

void RematReg::addUser(MachineInstr *MI, unsigned Region,
                       const LiveIntervals &LIS) {
  if (auto UsesIt = Uses.try_emplace(Region, MI, LIS); !UsesIt.second)
    UsesIt.first->getSecond().addUser(MI, LIS);
}

void RematReg::eraseUser(MachineInstr *MI, unsigned Region,
                         const LiveIntervals &LIS) {
  assert(Uses.contains(Region) && "broken dependency");
  RegionUses &RUses = Uses.find(Region)->getSecond();
  RUses.eraseUser(MI, LIS);
  if (RUses.Users.empty())
    Uses.erase(Region);
}

void RematReg::addUsers(const SmallDenseSet<MachineInstr *, 4> &NewUsers,
                        unsigned Region, const LiveIntervals &LIS) {
  if (NewUsers.empty())
    return;

  auto NewUsersIt = NewUsers.begin(), NewUsersEnd = NewUsers.end();
  auto RUsesIt = Uses.try_emplace(Region, *NewUsersIt, LIS);
  RegionUses RUses = RUsesIt.first->getSecond();
  if (!RUsesIt.second)
    RUses.addUser(*NewUsersIt, LIS);
  while (++NewUsersIt != NewUsersEnd)
    RUses.addUser(*NewUsersIt, LIS);
}

MachineBasicBlock::iterator
RematReg::getInsertPos(MachineBasicBlock::iterator InsertPos,
                       unsigned UseRegion, const LiveIntervals &LIS) const {
  // TODO: add defining region case
  if (auto RegionUses = Uses.find(UseRegion); RegionUses != Uses.end()) {
    MachineBasicBlock::iterator EarliestUser =
        RegionUses->getSecond().InsertPos;
    if (LIS.getInstructionIndex(*EarliestUser) <
        LIS.getInstructionIndex(*InsertPos))
      return EarliestUser;
  }
  return InsertPos;
}

void RematReg::RegionUses::addUser(MachineInstr *NewUser,
                                   const LiveIntervals &LIS) {
  assert(!Users.empty() && "no region users");
  Users.insert(NewUser);
  if (LIS.getInstructionIndex(*NewUser) < LIS.getInstructionIndex(*InsertPos))
    InsertPos = NewUser;
}

void RematReg::RegionUses::eraseUser(MachineInstr *DeletedUser,
                                     const LiveIntervals &LIS) {
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

Printable RematReg::print(bool SkipRegions) const {
  return Printable([&, SkipRegions](raw_ostream &OS) {
    if (!SkipRegions) {
      OS << "[" << DefRegion;
      if (!Uses.empty()) {
        auto It = Uses.begin();
        OS << " -> " << It->first;
        while (++It != Uses.end())
          OS << "," << It->first;
      }
      OS << "] ";
    }
    DefMI->print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
                 /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
  });
}

Printable RematDAG::print(unsigned RootIdx) const {
  return Printable([&, RootIdx](raw_ostream &OS) {
    DenseMap<unsigned, unsigned> RegDepths;
    std::function<void(unsigned, unsigned)> WalkChain =
        [&](unsigned RegIdx, unsigned Depth) -> void {
      unsigned MaxDepth = std::max(RegDepths.lookup_or(RegIdx, Depth), Depth);
      RegDepths.emplace_or_assign(RegIdx, MaxDepth);
      for (const RematReg::Dependency &Dep : getReg(RegIdx).Dependencies)
        WalkChain(Dep.RegIdx, Depth + 1);
    };
    WalkChain(RootIdx, 0);

    // Sort in decreasing depth order to print root at the bottom.
    SmallVector<std::pair<unsigned, unsigned>> Regs(RegDepths.begin(),
                                                    RegDepths.end());
    sort(Regs, [](const auto &LHS, const auto &RHS) {
      return LHS.second > RHS.second;
    });

    OS << printID(RootIdx) << " has " << Regs.size() - 1 << " dependencies\n";
    for (const auto &[RegIdx, Depth] : Regs) {
      std::string Shift(2 * Depth, ' ');
      std::string Sep = Depth ? "| " : "-> ";
      OS << Shift << Sep << '(' << RegIdx << ") "
         << getReg(RegIdx).print(/*SkipRegions=*/Depth) << '\n';
    }
    OS << printRegUsers(RootIdx);
  });
}

Printable RematDAG::printID(unsigned RegIdx) const {
  return Printable([&, RegIdx](raw_ostream &OS) {
    const RematReg &Reg = getReg(RegIdx);
    OS << '(' << RegIdx << '/';
    Register DefReg = Reg.DefMI ? Reg.getDefReg() : DeadRegs.at(RegIdx).Reg;
    unsigned SubIdx = Reg.DefMI ? Reg.DefMI->getOperand(0).getSubReg() : 0;
    OS << printReg(DefReg, &TRI, SubIdx, &MRI) << ")[" << Reg.DefRegion << "]";
  });
}

Printable RematDAG::printRegUsers(unsigned RegIdx) const {
  return Printable([&, RegIdx](raw_ostream &OS) {
    for (const auto &[UseRegion, Uses] : getReg(RegIdx).Uses) {
      for (MachineInstr *MI : Uses.Users)
        rdbgs() << "  User " << printUser(MI) << '\n';
    }
  });
}

Printable RematDAG::printUser(const MachineInstr *MI) const {
  return Printable([&, MI](raw_ostream &OS) {
    unsigned RegIdx = getRematRegIdx(*MI);
    if (RegIdx != getNumRegs())
      OS << printID(RegIdx) << " ";
    MI->print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
              /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
    OS << " @ ";
    LIS.getInstructionIndex(*MI).print(dbgs());
  });
}
