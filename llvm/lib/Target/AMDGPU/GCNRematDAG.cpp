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
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/Register.h"

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

static Register getPotentialRematReg(const MachineInstr &MI) {
  if (!MI.getNumOperands() || !MI.getOperand(0).isReg() ||
      MI.getOperand(0).readsReg())
    return AMDGPU::NoRegister;
  return MI.getOperand(0).getReg();
}

void RematDAG::rematerialize(unsigned RegIdx) {
  // Start by rematerializing all registers in the chain. The register iteration
  // order is an order that honors all dependencies between registers inside the
  // chain.
  // TODO: the above will no longer be true once we start adding new registers
  // to the list as we rematerialize.
  const RematReg &Reg = getReg(RegIdx);
  for (unsigned ChainRegIdx : Reg.chain()) {
    const RematReg &ChainReg = getReg(ChainRegIdx);
    assert(ChainReg.DefMI && "register already fully rematerialized");
    Register DefReg = ChainReg.getDefReg();
    auto NewRegsIt = RematerializedRegs.try_emplace(ChainRegIdx);
    DenseMap<unsigned, Register> &MyNewRegs = NewRegsIt.first->getSecond();

    // Rematerialize the register in every region where it is needed by the
    // chain.
    const BitVector &RematRegions = ChainReg.getRegRematRegions(ChainRegIdx);
    for (unsigned UseRegion : RematRegions.set_bits()) {
      Register NewReg = MRI.cloneVirtualRegister(DefReg);
      MyNewRegs.insert({UseRegion, NewReg});

      // The register must be available at or before the latest possible insert
      // position for each register in the chain which has the register in its
      // own chain and users in the region (including this register itself).
      MachineBasicBlock::iterator InsertPos =
          std::prev(Regions[UseRegion].second);
      for (unsigned PosRegIdx : ChainReg.chain()) {
        const RematReg &PosReg = getReg(PosRegIdx);
        if (ChainRegIdx >= PosReg.Chain.size() || !PosReg.Chain[ChainRegIdx])
          continue;
        auto RegRegUses = PosReg.Uses.find(UseRegion);
        if (RegRegUses != ChainReg.Uses.end()) {
          // Check the register's latest possible insert position in the region.
          MachineBasicBlock::iterator RegPos =
              *RegRegUses->getSecond().InsertPos;
          if (LIS.getInstructionIndex(*RegPos) <
              LIS.getInstructionIndex(*InsertPos))
            InsertPos = RegPos;
        }
      }

      TII.reMaterialize(*InsertPos->getParent(), InsertPos, NewReg, 0,
                        *ChainReg.DefMI, TRI);
      MachineInstr &RematMI = *std::prev(InsertPos);

      // The new instruction needs to use new registers previously
      // rematerialized as part of the chain. This expects these have already
      // been rematerialized and will assert if not.
      for (const RematDAG::RematReg::Dependency &Dep : ChainReg.Dependencies) {
        Register OldDepReg = ChainReg.DefMI->getOperand(Dep.MOIdx).getReg();
        Register NewDepReg = RematerializedRegs.at(Dep.RegIdx).at(UseRegion);
        RematMI.substituteRegister(OldDepReg, NewDepReg, 0, TRI);
      }
      insertMI(UseRegion, &RematMI);

      // Users of the rematerialized register in the region need to use the
      // new register.
      auto ChainRegUses = ChainReg.Uses.find(UseRegion);
      if (ChainRegUses != ChainReg.Uses.end()) {
        for (MachineInstr *UserMI : ChainRegUses->getSecond().Users)
          UserMI->substituteRegister(DefReg, NewReg, 0, TRI);
      }
    }
  }

  // Create virtual register intervals for all rematerialized instructions,
  // delete original instructions, and update RP in impacted regions.
  for (unsigned RegIdx : Reg.chain()) {
    for (const auto &[_, NewReg] : RematerializedRegs.at(RegIdx))
      LIS.createAndComputeVirtRegInterval(NewReg);
    RematReg &Reg = Regs[RegIdx];
    Register DefReg = Reg.getDefReg();
    LIS.removeInterval(DefReg);
    if (!Reg.PartialRemats[RegIdx]) {
      deleteMI(Reg.DefRegion, Reg.DefMI);
      // Setting the DefMI to nullptr signifies that the register was deleted.
      Reg.DefMI = nullptr;
    } else {
      // The register has shrunk to only include uses in the defining region.
      // FIXME: Manually update the interval instead of fully recomputing.
      LIS.createAndComputeVirtRegInterval(DefReg);
    }
  }
}

void RematDAG::rollback(unsigned RegIdx) {
  const RematReg &Reg = getReg(RegIdx);
  for (unsigned RegIdx : Reg.chain()) {
    RematDAG::RematReg &Reg = Regs[RegIdx];
    const DenseMap<unsigned, Register> &MyRematRegs =
        RematerializedRegs.at(RegIdx);

    // If the defining instruction was not deleted then the original register is
    // still alive and can be reused i.e. we don't need to rematerialize
    // anything, just substitute the rematerialized register uses for the old
    // register.
    Register DefReg;
    if (!Reg.DefMI) {
      assert(!Reg.PartialRemats[RegIdx] && "reg should be dead");
      // Recreate the original MI from one of the rematerializations.
      Register ModelReg = MyRematRegs.begin()->second;
      MachineInstr *ModelMI = MRI.getOneDef(ModelReg)->getParent();
      DefReg = MRI.cloneVirtualRegister(ModelReg);

      // Re-rematerialize MI in its original region. Note that it may not be
      // rematerialized exactly in the same position as originally within the
      // region, but it should not matter much.
      MachineBasicBlock::iterator IP(Regions[Reg.DefRegion].second);
      MachineBasicBlock *MBB = RegionBB[Reg.DefRegion];
      TII.reMaterialize(*MBB, IP, DefReg, 0, *ModelMI, TRI);
      Reg.DefMI = &*std::prev(IP);

      // Replace dependencies with re-rematerialized registers from the chain.
      // By construction all dependencies have already been rolled back.
      for (const RematDAG::RematReg::Dependency &Dep : Reg.Dependencies) {
        Register OldDepReg = Reg.DefMI->getOperand(Dep.MOIdx).getReg();
        Register NewDepReg = getReg(Dep.RegIdx).getDefReg();
        Reg.DefMI->substituteRegister(OldDepReg, NewDepReg, 0, TRI);
      }
      insertMI(Reg.DefRegion, Reg.DefMI);
    } else {
      DefReg = Reg.getDefReg();
    }

    // Use defined register in all regions.
    for (const auto &[UseRegion, RematReg] : MyRematRegs) {
      for (MachineInstr *UseMI : Reg.Uses.at(UseRegion).Users)
        UseMI->substituteRegister(RematReg, DefReg, 0, TRI);
    }
  }

  // Delete all rematerializations and re-create live intervals for all
  // re-rematerialized registers.
  for (unsigned RegIdx : Reg.chain()) {
    for (const auto &[UseRegion, RematReg] : RematerializedRegs.at(RegIdx)) {
      LIS.removeInterval(RematReg);
      deleteMI(UseRegion, MRI.getOneDef(RematReg)->getParent());
    }
    Register DefReg = getReg(RegIdx).getDefReg();
    // If the register already existed; remove its existing interval before
    // recomputing it.
    if (Reg.PartialRemats[RegIdx])
      LIS.removeInterval(DefReg);
    LIS.createAndComputeVirtRegInterval(DefReg);
    RematerializedRegs.erase(RegIdx);
  }
}

Printable RematDAG::RematReg::print(bool SkipRegions) const {
  return Printable([&, SkipRegions](raw_ostream &OS) {
    if (!SkipRegions) {
      // Concatenate all region numbers in which the register is used.
      std::string UsingRegions;
      for (const auto &[UseRegion, _] : Uses) {
        if (!UsingRegions.empty())
          UsingRegions += ",";
        UsingRegions += std::to_string(UseRegion);
      }
      OS << "[" << DefRegion << " -> " << UsingRegions << "] ";
    }
    DefMI->print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
                 /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
  });
}

Printable RematDAG::print(unsigned RegIdx, bool RootOnly,
                          bool SkipRegions) const {
  if (RootOnly) {
    return Printable([&, RegIdx](raw_ostream &OS) {
      OS << "-> " << getReg(RegIdx).print(SkipRegions);
    });
  }
  return Printable([&, RegIdx, SkipRegions](raw_ostream &OS) {
    SmallMapVector<unsigned, unsigned, 4> RegDepths;
    std::function<void(unsigned, unsigned)> WalkChain =
        [&](unsigned RegIdx, unsigned Depth) -> void {
      auto *RegIt = RegDepths.find(RegIdx);
      if (RegIt != RegDepths.end() && RegIt->second < Depth) {
        std::pair<unsigned, unsigned> NewDepth(RegIdx, Depth);
        RegIt->swap(NewDepth);
      } else {
        RegDepths.insert({RegIdx, Depth});
      }
      for (const RematReg::Dependency &Dep : getReg(RegIdx).Dependencies)
        WalkChain(Dep.RegIdx, Depth + 1);
    };
    WalkChain(RegIdx, 0);

    // Sort in decreasing depth order to print root at the bottom.
    SmallVector<std::pair<unsigned, unsigned>> Regs(RegDepths.takeVector());
    sort(Regs, [](const auto &LHS, const auto &RHS) {
      return LHS.second > RHS.second;
    });

    for (const auto &[RegIdx, Depth] : Regs) {
      std::string Shift(2 * Depth, ' ');
      std::string Sep = Depth ? "| " : "-> ";
      OS << Shift << Sep << getReg(RegIdx).print(SkipRegions) << '\n';
    }
  });
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

bool RematDAG::isMOAvailableAtUses(
    const MachineOperand &MO,
    SmallDenseMap<unsigned, SlotIndex, 4> Uses) const {
  if (Uses.empty())
    return true;
  Register DepReg = MO.getReg();
  unsigned SubIdx = MO.getSubReg();
  LaneBitmask Mask = SubIdx ? TRI.getSubRegIndexLaneMask(SubIdx)
                            : MRI.getMaxLaneMaskForVReg(DepReg);
  const LiveInterval &DepLI = LIS.getInterval(DepReg);
  const VNInfo *DefVN = DepLI.getVNInfoAt(
      LIS.getInstructionIndex(*MO.getParent()).getRegSlot(true));
  for (const auto &[_, UseIdx] : Uses) {
    if (!isAvailableAtUse(DefVN, Mask, UseIdx, DepLI))
      return false;
  }
  return true;
}

bool RematDAG::build() {
  clear();
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
    ArrayRef<unsigned> MBBRegions =
        RegionsPerBlock.at(MBB->getBlock()->getNumber());
    auto MBBRegionsIt = RegionsTopDown ? MBBRegions : reverse(MBBRegions);
    for (unsigned I : MBBRegionsIt)
      collectRematRegs(I);
  }

  return !Regs.empty();
}

void RematDAG::setRegDependency(unsigned RegIdx, unsigned DepRegIdx) {
  // The dependency's dependencies are transitively part of the chain.
  RematReg &Reg = Regs[RegIdx];
  RematReg &DepReg = Regs[DepRegIdx];
  Reg.Chain |= DepReg.Chain;

  // Notify all registers added to the chain that they are now a part of it, and
  // add regions in which each of these registers will need to be
  // rematerialized.
  const BitVector& RootRegionUses = Reg.RematRegions[RegIdx];
  for (unsigned ChainRegIdx : DepReg.chain()) {
    resizeBitVectorAndSet(Regs[ChainRegIdx].OtherChains, RegIdx);

    BitVector &DepRematRegions = Reg.RematRegions[ChainRegIdx];
    DepRematRegions |= DepReg.RematRegions[ChainRegIdx];
    DepRematRegions |= RootRegionUses;
    LLVM_DEBUG({
      dbgs() << "Added remat regions for (" << RegIdx << "/" << ChainRegIdx
             << "): is now ";
      for (unsigned I : DepRematRegions.set_bits())
        dbgs() << I << ' ';
      dbgs() << '\n';
    });
  }
}

void RematDAG::collectRematRegs(unsigned DefRegion) {
  // Collect partially rematerializable registers in instruction order within
  // each region. This guarantees that, within a single region, partially
  // rematerializable registers used in instructions defining other partially
  // rematerializable registers are visited first. This is important to
  // guarantee that all of a register's dependencies are visited before the
  // register itself.
  LLVM_DEBUG(dbgs() << "Collecting registers in " << DefRegion << '\n');
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
    resizeBitVectorAndSet(Reg.Chain, RegIdx);
    unsigned SubIdx = DefMI.getOperand(0).getSubReg();
    Reg.Mask = SubIdx ? TRI.getSubRegIndexLaneMask(SubIdx)
                      : MRI.getMaxLaneMaskForVReg(DefReg);
    BitVector &RematRegions = Reg.RematRegions[RegIdx];
    RematRegions.resize(Regions.size());

    LLVM_DEBUG(dbgs() << "Candidate register (" << RegIdx << "): " << DefMI);

    // Collect the candidate's direct users, both rematerializable and
    // unrematerializable.
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(DefReg)) {
      auto UseRegion = MIRegion.find(&UseMI);
      if (UseRegion == MIRegion.end()) {
        // Only lone MI terminators can trigger this condition. They are not
        // part of any region so we cannot rematerialize next to them, so we
        // just error out in these cases.
        Reg.Uses.clear();
        break;
      }
      if (UseRegion->second == DefRegion) {
        Reg.DefRegionUsers.push_back(&UseMI);
      } else {
        if (auto It = Reg.Uses.find(UseRegion->second); It != Reg.Uses.end()) {
          It->getSecond().addUser(&UseMI, LIS);
        } else {
          RematRegions.set(UseRegion->second);
          Reg.Uses.try_emplace(UseRegion->second, &UseMI, LIS);
        }
      }
    }
    if (Reg.Uses.empty() && Reg.DefRegionUsers.empty()) {
      LLVM_DEBUG(dbgs() << "  -> Eliminated (no non-debug users)\n");
      Regs.pop_back();
      continue;
    }

    // Derive slots at which dependencies must be available. We map them from
    // their containing region so that we can perform more efficient slot
    // merging when creating chains.
    SmallDenseMap<unsigned, SlotIndex, 4> UseSlots;
    for (const auto &[UseRegion, RegionUses] : Reg.Uses) {
      for (const MachineInstr *UseMI : RegionUses.Users) {
        SlotIndex UseSlot = LIS.getInstructionIndex(*UseMI);
        UseSlots.insert({UseRegion, UseSlot.getRegSlot(true)});
      }
    }

    // Collect the candidate's dependencies. If the same register is used
    // multiple times we just need to store it once.
    SmallDenseSet<Register, 4> AllDepRegs;
    SmallVector<unsigned> UnavailableDeps;
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
        // The dependency is itself partially rematerializable.
        Reg.Dependencies.emplace_back(MOIdx, DepRegIdx);
        if (!Available)
          UnavailableDeps.push_back(Reg.Dependencies.size() - 1);
      } else if (!Available) {
        // The dependency is both unavailable at uses and unrematerializable, so
        // the register can never be even partially rematerializable.
        LLVM_DEBUG(dbgs() << "  -> Eliminated (operand #" << MOIdx
                          << " unrematable and unavaialble)\n");
        Regs.pop_back();
        break;
      }
    }
    if (RegIdx == Regs.size())
      continue;

    // Now we know the register is at least partially rematerializable.
    RegToIdx.insert({DefReg, RegIdx});

    // Dependencies available at all of their parent's uses are not part of the
    // chain. Unavailable dependencies need to be rematerialized along with the
    // register.
    for (unsigned I : UnavailableDeps)
      setRegDependency(RegIdx, Reg.Dependencies[I].RegIdx);
  }

  // FIXME: may be able to merge the two loops below i.e. iterate only once over
  // all registers again.

  // Mark register which cannot ever be fully rematerialized. These are
  // registers which have at least one unreamaterializable user in their region.
  // Iterate in reverse register order to visit the users of a register before
  // the register itself.
  for (RematReg &Reg : reverse(Regs)) {
    if (Reg.AlwaysPartialRemat)
      continue;

    for (const MachineInstr *DefMI : Reg.DefRegionUsers) {
      Register DefReg = getPotentialRematReg(*DefMI);
      if (DefReg == AMDGPU::NoRegister) {
        Reg.AlwaysPartialRemat = true;
        break;
      }
      auto UserRegIdx = RegToIdx.find(DefReg);
      if (UserRegIdx == RegToIdx.end() ||
          Regs[UserRegIdx->second].AlwaysPartialRemat) {
        Reg.AlwaysPartialRemat = true;
        break;
      }
    }

    if (Reg.AlwaysPartialRemat) {
      // All its dependencies in the same region will also always be only
      // partially rematerializable.
      for (unsigned DepRegIdx : Reg.chain())
        Regs[DepRegIdx].AlwaysPartialRemat = true;
      continue;
    }
  }

  // Compute partial rematerializability for each register in every chain.
  // FIXME: Can probably make this more efficient by caching intermediate
  // results along the way.
  for (unsigned RootIdx = 0, E = getNumRegs(); RootIdx < E; ++RootIdx) {
    RematReg &RootReg = Regs[RootIdx];
    RootReg.PartialRemats.resize(RootReg.Chain.size());

    BitVector Visited(RootReg.Chain.size());
    std::function<void(unsigned)> CheckPartialRemat = [&](unsigned RegIdx) {
      assert(RootReg.Chain[RegIdx] && "register must be in chain");
      if (Visited[RegIdx])
        return;
      Visited.set(RegIdx);

      const RematReg &Reg = getReg(RegIdx);
      if (Reg.AlwaysPartialRemat)
        return;

      // All users in the same region must be fully rematerializable.
      for (const MachineInstr *UserMI : Reg.DefRegionUsers) {
        // We know the user is rematerializable, otherwise the register would be
        // always partially rematerializable.
        unsigned UserIdx = RegToIdx.at(UserMI->getOperand(0).getReg());
        const RematReg &UserReg = getReg(UserIdx);
        if (Reg.DefRegion != UserReg.DefRegion)
          continue;
        if (UserIdx >= RootReg.Chain.size() || !RootReg.Chain[UserIdx]) {
          // The user is not in the chain so the original register will need to
          // remain in its region.
          return;
        }
        if (!Visited[UserIdx])
          CheckPartialRemat(UserIdx);
        if (RootReg.PartialRemats[UserIdx]) {
          // The user is in the chain but is partially rematerializable; the
          // original user will remain and so should the register.
          return;
        }
      }

      // The register is fully rematerializable as part of this chain.
      RootReg.PartialRemats.reset(RegIdx);
    };

    for (unsigned ChainRegIdx : RootReg.Chain.set_bits())
      CheckPartialRemat(ChainRegIdx);
  }
}

void RematDAG::clear() {
  MIRegion.clear();
  RegionBB.clear();
  Regs.clear();
  RegToIdx.clear();
  RematerializedRegs.clear();
}
