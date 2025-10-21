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

RematDAG::RematChain::RematChain(unsigned RootIdx,
                                 ArrayRef<RematReg> RematRegs) {
  SmallDenseSet<unsigned, 4> Visited;
  std::function<void(unsigned)> VisitReg = [&](unsigned RegIdx) {
    // Don't visit a register more than once.
    if (!Visited.insert(RegIdx).second)
      return;

    // Visit all dependencies before the register itself.
    const RematReg &Reg = RematRegs[RegIdx];
    for (const RematReg::Dependency &Dep : Reg.Dependencies) {
      if (Dep.isRematInChain())
        VisitReg(*Dep.RematIdx);
    }
    Regs.push_back(RegIdx);
  };
  VisitReg(RootIdx);
}
bool RematDAG::build() {
#ifndef NDEBUG
  auto PrintCandidates = [&](StringRef StepName) {
    dbgs() << StepName << " there are " << Rematable.count()
           << " rematerializable candidates and " << CandRoots.count()
           << " potential chain roots:\n";
    for (unsigned I : Rematable.set_bits()) {
      dbgs() << "| (" << I << ") ";
      if (CandRoots[I])
        dbgs() << "--root--> ";
      MRI.getOneDef(getCandidateReg(I))
          ->getParent()
          ->print(dbgs(), true, false, false, false);
      if (InvDeps[I].count()) {
        dbgs() << " / depended on by ";
        for (unsigned J : InvDeps[I].set_bits())
          dbgs() << "(" << J << ") ";
      }
      dbgs() << '\n';
    }
  };
#endif

  const unsigned NumRegions = Regions.size();
  for (unsigned I = 0; I < NumRegions; ++I) {
    RegionBoundaries Region = Regions[I];
    for (auto MI = Region.first; MI != Region.second; ++MI)
      MIRegion.insert({&*MI, I});
    if (Region.second != Region.first->getParent()->end())
      MIRegion.insert({&*Region.second, I});
  }

  // Collect candidate registers.
  collectCandidates();
  const unsigned NumCands = Candidates.size();

  // Start with all candidates and filter them down progressively.
  Rematable.resize(NumCands, true);
  CandRoots.resize(NumCands, true);
  InvDeps.resize(NumCands, BitVector(NumCands));

  LLVM_DEBUG(PrintCandidates("At start"));
  sameRegionFiltering();
  LLVM_DEBUG(PrintCandidates("After same-region filtering"));
  liveRangeFiltering();
  LLVM_DEBUG(PrintCandidates("After range extension filtering"));
  dependentClusterFiltering();
  LLVM_DEBUG(PrintCandidates("After anti-cluster/anti-dependency filtering"));

  const unsigned NumRemats = Rematable.count();
  Regs.clear();
  Regs.reserve(NumRemats);

  // All remaining candidates are rematerializable and form valid chains of
  // possible rematerializations. Re-index registers now that we have eliminated
  // all non-rematerializable candidates.
  DenseMap<unsigned, unsigned> CandToRematIdxRemap;
  CandToRematIdxRemap.reserve(NumRemats);
  for (auto [NewIdx, OldIdx] : enumerate(Rematable.set_bits()))
    CandToRematIdxRemap.insert({OldIdx, NewIdx});

  // Create rematerializable registers from pre-computed info, fixing
  // rematerializable dependency indices and marking chain roots along the way.
  BitVector ChainRoots(NumRemats);
  for (unsigned I : Rematable.set_bits()) {
    CandInfo &Cand = CandData[I];
    for (RematReg::Dependency &Dep : Cand.Dependencies) {
      if (Dep.RematIdx)
        Dep.RematIdx = CandToRematIdxRemap.at(*Dep.RematIdx);
    }
    MachineInstr *DefMI = getDefMI(getCandidateReg(I));
    Regs.emplace_back(DefMI, MIRegion.at(DefMI), std::move(Cand.Uses),
                      std::move(Cand.Dependencies));
    if (CandRoots[I])
      ChainRoots.set(CandToRematIdxRemap.at(I));
  }

  // Create rematerializable chains.
  Chains.clear();
  Chains.reserve(ChainRoots.count());
  for (unsigned I : ChainRoots.set_bits())
    Chains.emplace_back(I, Regs);

  clear();
  return !Regs.empty();
}

void RematDAG::clear() {
  MIRegion.clear();
  Candidates.clear();
  Rematable.clear();
  CandRoots.clear();
  InvDeps.clear();
  CandData.clear();
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

bool RematDAG::isMOAvailableAtUses(const MachineOperand &MO,
                                   ArrayRef<SlotIndex> Uses) const {
  Register DepReg = MO.getReg();
  unsigned SubIdx = MO.getSubReg();
  LaneBitmask Mask = SubIdx ? TRI.getSubRegIndexLaneMask(SubIdx)
                            : MRI.getMaxLaneMaskForVReg(DepReg);
  const LiveInterval &DepLI = LIS.getInterval(DepReg);
  const VNInfo *DefVN = DepLI.getVNInfoAt(
      LIS.getInstructionIndex(*MO.getParent()).getRegSlot(true));
  for (SlotIndex UseIdx : Uses) {
    if (!isAvailableAtUse(DefVN, Mask, UseIdx, DepLI))
      return false;
  }
  return true;
}

void RematDAG::invalidate(unsigned I) {
  Rematable.reset(I);
  CandRoots.reset(I);
  // It is possible that two candidates mutually depend on each other. To avoid
  // infinite invalidation loops we have to clear the vector before recursively
  // invalidating from it.
  BitVector InvCopy(InvDeps[I]);
  InvDeps[I].reset();
  for (unsigned J : InvCopy.set_bits()) {
    Rematable.reset(J);
    CandRoots.reset(J);
    invalidate(J);
  }
}

void RematDAG::collectCandidates() {
  auto InRegion = [&](MachineInstr &UseMI) {
    return !MIRegion.contains(&UseMI);
  };

  // Identify candidate registers for rematerialization in the function.
  for (unsigned I = 0, E = Regions.size(); I != E; ++I) {
    RegionBoundaries Bounds = Regions[I];
    for (auto MI = Bounds.first; MI != Bounds.second; ++MI) {
      // The instruction must be rematerializable.
      MachineInstr &DefMI = *MI;
      if (!isReMaterializable(DefMI))
        continue;

      // We only support rematerializing virtual registers with one definition.
      Register Reg = DefMI.getOperand(0).getReg();
      if (!Reg.isVirtual() || !MRI.hasOneDef(Reg))
        continue;

      // Register must have at least one user and all users must be trackable.
      auto Users = MRI.use_nodbg_instructions(Reg);
      if (Users.empty() || any_of(Users, InRegion))
        continue;

      // This is a valid candidate.
      Candidates.insert({Reg, Candidates.size()});
    }
  }
}

std::pair<RematDAG::CandInfo &, bool>
RematDAG::getOrInitCandInfo(unsigned CandIdx) {
  auto [It, NewReg] = CandData.try_emplace(CandIdx);
  CandInfo &Cand = It->getSecond();
  if (!NewReg)
    return {Cand, false};

  Register Reg = getCandidateReg(CandIdx);
  MachineInstr *DefMI = getDefMI(Reg);
  unsigned DefRegion = MIRegion.at(DefMI);
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
    unsigned UseRegion = MIRegion.at(&UseMI);
    if (auto It = Cand.Uses.find(UseRegion); It != Cand.Uses.end())
      It->getSecond().addUser(&UseMI, LIS);
    else
      Cand.Uses.emplace_or_assign(UseRegion, &UseMI, LIS);
  }

  // Store the candidate's dependencies on the first visit. If the same register
  // is used multiple times we just need to store it once.
  SmallDenseSet<Register, 4> AllDepRegs;
  for (const auto &[MOIdx, MO] : enumerate(DefMI->operands())) {
    Register DepReg = isRegDependency(MO);
    if (!DepReg || AllDepRegs.contains(DepReg))
      continue;
    AllDepRegs.insert(DepReg);
    RematReg::Dependency &Dep = Cand.Dependencies.emplace_back(MOIdx);
    auto *DepIt = Candidates.find(DepReg);
    if (DepIt != Candidates.end() && Rematable[DepIt->second]) {
      // The dependency is itself potentially rematerializable.
      Dep.RematIdx = DepIt->second;
      Dep.SameRegion = DefRegion == MIRegion.at(getDefMI(DepIt->first));
    }
  }

  LLVM_DEBUG({
    rdbgs() << "Collecting users on first visit:\n";
    for (const auto &[UseRegion, RegionUses] : Cand.Uses) {
      rdbgs() << "| " << RegionUses.Users.size() << " users in region ["
              << UseRegion << "]"
              << (UseRegion == DefRegion ? " (defining region)" : "")
              << "; first is " << *RegionUses.InsertPos;
    }
    rdbgs() << "Collecting dependencies on first visit:\n";
    for (const RematReg::Dependency &Dep : Cand.Dependencies) {
      rdbgs() << "| Dependency " << DefMI->getOperand(Dep.MOIdx);
      if (Dep.RematIdx)
        dbgs() << " (" << *Dep.RematIdx << ')';
      dbgs() << '\n';
    }
  });
  return {Cand, true};
}

void RematDAG::sameRegionFiltering() {
  for (const std::pair<Register, unsigned> &Cand : reverse(Candidates)) {
    Register Reg = Cand.first;
    unsigned Idx = Cand.second;
    unsigned DefRegion = MIRegion.at(getDefMI(Reg));

    // If the use is in the same region as the defining instruction then it must
    // be rematerializable.
    auto IsValidUse = [&](const MachineInstr &UseMI) -> bool {
      if (MIRegion.at(&UseMI) != DefRegion)
        return true;

      // Only instructions whose first operand is a defined register are
      // rematerializable.
      if (!UseMI.getOperand(0).isReg() || UseMI.getOperand(0).readsReg())
        return false;

      // Since we iterate in the reverse candidate detection order, which was
      // just program order, for two candidates defined in the same region
      // we are guaranteed to hit all uses before the definition i.e., if the
      // use is not rematerializable it has already been unmarked.
      auto *UseIt = Candidates.find(UseMI.getOperand(0).getReg());
      if (UseIt == Candidates.end() || !Rematable[UseIt->second])
        return false;

      // The current candidate can only be rematerialized together with its
      // users from the same region. If the user is not rematerializable, then
      // the candidate is not rematerializable as well.
      InvDeps[UseIt->second].set(Idx);
      return true;
    };

    if (!all_of(MRI.use_nodbg_instructions(Reg), IsValidUse)) {
      Rematable.reset(Idx);
      CandRoots.reset(Idx);
    }
  }
}

void RematDAG::liveRangeFiltering() {
  for (unsigned I : CandRoots.set_bits()) {
    LLVM_DEBUG({
      enterRec(I);
      rdbgs() << "Validating potential root: " << *getDefMI(getCandidateReg(I));
    });
    if (!validateCandLiveRange(I, std::nullopt))
      invalidate(I);
    LLVM_DEBUG(exitRec(I));
  }
}

bool RematDAG::validateCandLiveRange(unsigned I,
                                     std::optional<unsigned> ParentIdx) {
  assert(Rematable[I] && "non-rematable register is not valid candidate");
  Register Reg = getCandidateReg(I);
  const MachineInstr &CandMI = *getDefMI(Reg);

  auto RegCandInfo = getOrInitCandInfo(I);
  CandInfo &Cand = RegCandInfo.first;
  SmallVector<SlotIndex, 4> ExtendedSlots;
  if (RegCandInfo.second) {
    for (const auto &[_, RegionUsers] : Cand.Uses)
      ExtendedSlots.push_back(RegionUsers.LastAliveIdx);
  }

  if (ParentIdx) {
    const LiveInterval &LI = LIS.getInterval(Reg);
    const MachineInstr &ParentCandMI = *getDefMI(getCandidateReg(*ParentIdx));
    SlotIndex ParentDefSlot = LIS.getInstructionIndex(ParentCandMI);
    const VNInfo *DefVN = LI.getVNInfoAt(ParentDefSlot.getRegSlot(true));
    unsigned SubIdx = CandMI.getOperand(0).getSubReg();
    LaneBitmask Mask = SubIdx ? TRI.getSubRegIndexLaneMask(SubIdx)
                              : MRI.getMaxLaneMaskForVReg(Reg);

    auto ExtendForRematUser = [&](SlotIndex RematIdx) -> void {
      ExtendedSlots.push_back(RematIdx);
      if (Cand.AvailableAtUses) {
        Cand.AvailableAtUses = isAvailableAtUse(DefVN, Mask, RematIdx, LI);
        // A register which is not available at all its parents' uses cannot
        // be the root of a different chain; it must be rematerialized with
        // its parents as to not lengthen any live range.
        if (!Cand.AvailableAtUses) {
          LLVM_DEBUG(rdbgs() << "| -> Candidate cannot be a root\n");
          CandRoots.reset(I);
        }
      }
    };

    LLVM_DEBUG(rdbgs() << "Integrating new parent(s) uses:\n");

    // In addition to its own direct users, the candidate would need to be
    // rematerialized at points where its parents in the chain need it. We
    // already went through the parent, so its users are already available.
    const CandInfo &ParentCand = CandData.at(*ParentIdx);
    for (const auto &[UseRegion, ParentRegionUses] : ParentCand.Uses) {
      LLVM_DEBUG(rdbgs() << "| [" << UseRegion << "] ");
      SlotIndex RematIdx = LIS.getInstructionIndex(*ParentRegionUses.InsertPos);
      if (auto It = Cand.Uses.find(UseRegion); It == Cand.Uses.end()) {
        // The current candidate has no user in the region but one of its
        // parent needs to be rematerialized in it, so this one may need to be
        // rematerialized to (unless it is available there already).
        LLVM_DEBUG(dbgs() << "Indirect requirement\n");
        Cand.Uses.try_emplace(UseRegion, ParentRegionUses, LIS);
        ExtendForRematUser(RematIdx);
      } else {
        // The current candidate has users or is already needed by other
        // parent registers in the region. The rematerialization point of the
        // current candidate may need to be adjusted to be usable by the
        // parent register.
        RematReg::RegionUses &Uses = It->getSecond();
        if (RematIdx > Uses.LastAliveIdx) {
          LLVM_DEBUG(dbgs() << "Extending avail. requirement\n");
          ExtendForRematUser(RematIdx);
          Uses.LastAliveIdx = RematIdx;
        } else if (RematIdx < LIS.getInstructionIndex(*Uses.InsertPos)) {
          LLVM_DEBUG(dbgs() << "Moving insert pos. earlier\n");
          Uses.InsertPos = ParentRegionUses.InsertPos;
        } else {
          LLVM_DEBUG(dbgs() << "Nothing to do\n");
        }
      }
    }
  }

  auto IsValidDependency = [&](RematReg::Dependency &Dep) -> bool {
    assert(Dep.RematIdx || Dep.AvailableAtUses && "already unrematable");
    LLVM_DEBUG(rdbgs() << "| Verifying " << CandMI.getOperand(Dep.MOIdx)
                       << '\n');
    if (!Dep.RematIdx) {
      // Only check live range extensions; previous visits of this candidate
      // have already validated that it is available at other inserion points.
      const MachineOperand &MO = CandMI.getOperand(Dep.MOIdx);
      if (!isMOAvailableAtUses(MO, ExtendedSlots)) {
        LLVM_DEBUG(rdbgs() << "** Dependency unavailable at new uses\n");
        return false;
      }
      return true;
    }

    // The dependency corresponds to a candidate register.
    unsigned DepIdx = *Dep.RematIdx;
    if (Rematable[DepIdx]) {
      LLVM_DEBUG(enterRec(DepIdx));
      // We currently believe the dependency is rematerializable, make sure
      // that it is still the case by recursviely validating the chain
      // starting at the dependency.
      bool RematableDep = validateCandLiveRange(DepIdx, I);
      if (!CandData.at(*Dep.RematIdx).AvailableAtUses) {
        // A dependency unavailable at its parent's uses needs to be
        // rematerializable for the parent to be rematerializable.
        Dep.AvailableAtUses = false;
        InvDeps[*Dep.RematIdx].set(I);
      }
      if (!RematableDep) {
        invalidate(DepIdx);
        if (!Dep.AvailableAtUses) {
          LLVM_DEBUG({
            rdbgs() << "** Unrematable and unavailable dependency\n";
            exitRec(DepIdx);
          });
          return false;
        }
      }
      LLVM_DEBUG(exitRec(DepIdx));
      return true;
    }

    // The dependency has been marked unrematerializable since last time we
    // visited the candidate.
    LLVM_DEBUG(rdbgs() << "| Marked unrematerializable\n");
    Dep.RematIdx = std::nullopt;
    const MachineOperand &MO = CandMI.getOperand(Dep.MOIdx);
    if (!Dep.AvailableAtUses || !isMOAvailableAtUses(MO, ExtendedSlots)) {
      LLVM_DEBUG(rdbgs() << "** Unrematable and unavailable dependency\n");
      return false;
    }
    return true;
  };

  // Make sure all dependencies are either rematerializable or available at all
  // uses. Rematerializability and availability of some dependencies may have
  // changed. The former happens when another rematerialization chain the
  // dependency is a part of was invalidated, whereas the latter happens when
  // the live ranges at which the dependency is expected to be available have
  // lengthened.
  LLVM_DEBUG(rdbgs() << "Verifying dependencies:\n");
  if (all_of(Cand.Dependencies, IsValidDependency)) {
    // Dependencies are still valid; the candidate appears to still be
    // rematerializable.
    LLVM_DEBUG(rdbgs() << "** Candidate still OK\n");
    return true;
  }
  return false;
}

void RematDAG::dependentClusterFiltering() {
  BitVector PartOfChain(Rematable.size());
  for (unsigned RootIdx : CandRoots.set_bits()) {
    SmallVector<unsigned> Explore({RootIdx});
    SmallDenseSet<unsigned, 4> VisitedInChain;
    while (!Explore.empty()) {
      unsigned I = Explore.pop_back_val();
      // Filter out registers part of multiple chains.
      if (PartOfChain[I] || !Rematable[I]) {
        invalidate(RootIdx);
        break;
      }
      PartOfChain.set(I);
      for (RematReg::Dependency &Dep : CandData[I].Dependencies) {
        if (!Dep.RematIdx)
          continue;
        if (!Dep.SameRegion) {
          // The root of a chain from another region has a user in the current
          // chain, creating a dependency between the two. Disable the current
          // chain to avoid having to track that.
          invalidate(RootIdx);
          break;
        }
        if (Dep.AvailableAtUses) {
          // The root of another chain is available at the current chain's uses,
          // creating a dependency between the two. Disable the other chain to
          // avoid having to track that.
          invalidate(*Dep.RematIdx);
          Dep.RematIdx = std::nullopt;
        } else if (VisitedInChain.insert(*Dep.RematIdx).second) {
          Explore.push_back(*Dep.RematIdx);
        }
      }
    }
  }
}

Printable RematDAG::printReg(unsigned RegIdx, bool SkipRegions) const {
  return Printable([&, RegIdx, SkipRegions](raw_ostream &OS) {
    const RematReg &Reg = getReg(RegIdx);
    if (!SkipRegions) {
      // Concatenate all region numbers in which the register is used.
      std::string UsingRegions;
      for (const auto &[UseRegion, _] : Reg.Uses) {
        if (UseRegion == Reg.DefRegion)
          continue;
        if (!UsingRegions.empty())
          UsingRegions += ",";
        UsingRegions += std::to_string(UseRegion);
      }
      OS << "[" << Reg.DefRegion << " -> " << UsingRegions << "] ";
    }
    Reg.DefMI->print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
                     /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
  });
}

Printable RematDAG::printChain(unsigned ChainIdx, bool RootOnly) const {
  if (RootOnly) {
    return Printable([&, ChainIdx](raw_ostream &OS) {
      OS << "-> " << printReg(getChain(ChainIdx).root());
    });
  }

  return Printable([&, ChainIdx](raw_ostream &OS) {
    // Do a very simple recursive chain walk to determine the maximum distance
    // between each node and the root. This ensures that a register's dependencies
    // are always printed at a deeper nesting level than the register itself.
    // Quite inneficient, but this is only for debug.
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
      for (const RematReg::Dependency &Dep : getReg(RegIdx).Dependencies) {
        if (Dep.isRematInChain())
          WalkChain(*Dep.RematIdx, Depth + 1);
      }
    };
    WalkChain(getChain(ChainIdx).root(), 0);

    // Sort in decreasing depth order to print root at the bottom.
    SmallVector<std::pair<unsigned, unsigned>> Regs(RegDepths.takeVector());
    sort(Regs, [](const auto &LHS, const auto &RHS) {
      return LHS.second > RHS.second;
    });
   
    for (const auto &[RegIdx, Depth] : Regs) {
      std::string Shift(2 * Depth, ' ');
      std::string Sep = Depth ? "| " : "-> ";
      OS << Shift << Sep << printReg(RegIdx, /*SkipRegions=*/false) << '\n';
    }
  });
}
