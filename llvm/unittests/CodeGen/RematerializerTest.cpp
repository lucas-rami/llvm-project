//===- RematerializerTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Rematerializer.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;
using RegisterIdx = Rematerializer::RegisterIdx;

namespace {

void initLLVM() {
  InitializeAllTargets();
  InitializeAllTargetMCs();
}

std::unique_ptr<TargetMachine> createTargetMachine() {
  Triple TargetTriple("amdgcn--");
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
  if (!T)
    return nullptr;
  TargetOptions Options;
  return std::unique_ptr<TargetMachine>(T->createTargetMachine(
      TargetTriple, "gfx950", "", Options, std::nullopt));
}

/// Wraps a rematerializer (with pointer-like access semantics through ->) next
/// to other members generally used by unit tests.
struct TestRematerializer {
  MachineFunction &MF;
  LiveIntervals &LIS;
  Rematerializer Remater;

  /// Region sizes for regions passed to the rematerializer. Initialized at
  /// construction to the correct value, can then be modified to track expected
  /// changes.
  SmallVector<unsigned> RegionSizes;
  /// Number of rematerializable registers identified by the rematerializer.
  /// Initialized at construction to the correct value, can then be modified to
  /// track expected changes.
  unsigned NumRematRegs;

  using RegionBoundaries = Rematerializer::RegionBoundaries;
  TestRematerializer(MachineFunction &MF,
                     SmallVectorImpl<RegionBoundaries> &Regions,
                     LiveIntervals &LIS)
      : MF(MF), LIS(LIS), Remater(MF, Regions, LIS) {
    for (const RegionBoundaries &Region : Regions)
      RegionSizes.push_back(std::distance(Region.first, Region.second));
    Remater.analyze();
    NumRematRegs = Remater.getNumRegs();
  }

  Rematerializer *operator->() { return &Remater; }
  const Rematerializer *operator->() const { return &Remater; }
  Rematerializer &operator*() { return Remater; }
  const Rematerializer &operator*() const { return Remater; }

  /// Returns the number of users of rematerializable register \p RegIdx.
  unsigned getNumUsers(RegisterIdx RegIdx) const {
    unsigned NumUsers = 0;
    for (const auto &[_, RegionUses] : Remater.getReg(RegIdx).Uses)
      NumUsers += RegionUses.size();
    return NumUsers;
  }

  /// Returns the number of MIs in region \p RegionIdx.
  unsigned getRegionSize(unsigned RegionIdx) const {
    const RegionBoundaries &Region = Remater.getRegion(RegionIdx);
    return std::distance(Region.first, Region.second);
  }

  /// Expects that \p NumMIs were added to region \p RegionIdx.
  TestRematerializer &addMIs(unsigned RegionIdx, unsigned NumMIs) {
    RegionSizes[RegionIdx] += NumMIs;
    return *this;
  }

  /// Expects that \p NumMIs were removed from region \p RegionIdx.
  TestRematerializer &removeMIs(unsigned RegionIdx, unsigned NumMIs) {
    RegionSizes[RegionIdx] -= NumMIs;
    return *this;
  }

  /// Expects that \p NumMIs were move from region \p FromRegionIdx to region \p
  /// ToRegionIdx.
  TestRematerializer &moveMIs(unsigned FromRegionIdx, unsigned ToRegionIdx,
                              unsigned NumMIs) {
    return removeMIs(FromRegionIdx, NumMIs).addMIs(ToRegionIdx, NumMIs);
  }

  /// Expects that \p NumRegs rematerializable registers were added to the
  /// rematerializer.
  TestRematerializer &addRematRegs(unsigned NumRegs) {
    NumRematRegs += NumRegs;
    return *this;
  }
};

/// All custon asserts/expects assume that a TestRematerializer is in scope and
/// named TR.

/// Asserts that the number of expected rematerializable registers indeed tracks
/// the actual number correctly.
#define ASSERT_NUM_REMAT_REGS() ASSERT_EQ(TR->getNumRegs(), TR.NumRematRegs)

/// Asserts that all regions match expected sizes from the test rematerializer.
#define ASSERT_REGION_SIZES()                                                  \
  {                                                                            \
    for (const auto [RegionIdx, ExpectedSize] : enumerate(TR.RegionSizes))     \
      ASSERT_EQ(TR.getRegionSize(RegionIdx), ExpectedSize);                    \
  }

/// Expects that register RegIdx in the rematerializer has a total of N users.
#define EXPECT_NUM_USERS(RegIdx, N)                                            \
  EXPECT_EQ(TR.getNumUsers(RegIdx), static_cast<unsigned>(N))

/// Expects that register RegIdx in the rematerializer has a total of N
/// dependencies.
#define EXPECT_NUM_DEPENDENCIES(RegIdx, N)                                     \
  EXPECT_EQ(TR->getReg(RegIdx).Dependencies.size(), static_cast<unsigned>(N))

/// Expects that register RegIdx in the rematerializer has no users.
#define EXPECT_NO_USERS(RegIdx) EXPECT_NUM_USERS(RegIdx, 0)

/// Expects that rematerialized register RegIdx has origin OriginIdx, is defined
/// in region DefRegionIdx, and has a total of NumUsers users.
#define EXPECT_REMAT(RegIdx, OriginIdx, DefRegionIdx, NumUsers)                \
  {                                                                            \
    const Rematerializer::Reg &RematReg = TR->getReg(RegIdx);                  \
    EXPECT_EQ(TR->getOriginOf(RegIdx), OriginIdx);                             \
    EXPECT_EQ(RematReg.DefRegion, DefRegionIdx);                               \
    EXPECT_NUM_USERS(RegIdx, NumUsers);                                        \
  }

using RematerializerTestFn = std::function<void(TestRematerializer &TR)>;

static void rematerializerTest(StringRef MIRBody, RematerializerTestFn Test) {
  initLLVM();
  std::unique_ptr<TargetMachine> TM = createTargetMachine();
  if (!TM)
    GTEST_SKIP();

  LLVMContext Context;
  SMDiagnostic Diagnostic;

  SmallString<512> S;
  StringRef MIRString = (Twine(R"MIR(
---
...
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
)MIR") + Twine(MIRBody) + Twine("...\n"))
                            .toNullTerminatedStringRef(S);

  std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRString);
  std::unique_ptr<MIRParser> MIR = createMIRParser(std::move(MBuffer), Context);
  ASSERT_TRUE(MIR);

  std::unique_ptr<Module> M = MIR->parseIRModule();
  ASSERT_TRUE(M);
  M->setDataLayout(TM->createDataLayout());

  std::unique_ptr<MachineModuleInfo> MMI =
      std::make_unique<MachineModuleInfo>(TM.get());

  LoopAnalysisManager LAM;
  MachineFunctionAnalysisManager MFAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB(TM.get());
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerMachineFunctionAnalyses(MFAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM, &MFAM);
  MAM.registerPass([&] { return MachineModuleAnalysis(*MMI); });

  ASSERT_FALSE(MIR->parseMachineFunctions(*M, MAM));

  MachineFunction &MF =
      FAM.getResult<MachineFunctionAnalysis>(*M->getFunction("func")).getMF();
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

  SmallVector<Rematerializer::RegionBoundaries> Regions;
  MachineInstr *FirstMI = nullptr;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!FirstMI)
        FirstMI = &MI;
      if (MI.isTerminator()) {
        if (FirstMI != &MI)
          Regions.push_back({FirstMI, MI});
        FirstMI = nullptr;
      }
    }
    if (FirstMI) {
      Regions.push_back({FirstMI, MBB.end()});
      FirstMI = nullptr;
    }
  }

  TestRematerializer TR(MF, Regions, LIS);
  Test(TR);

  TR->updateLiveIntervals();
  EXPECT_TRUE(MF.verify());
}

} // end anonymous namespace

/// Rematerializes a tree of registers to a single user in different ways using
/// the dependency reuse mechanics and the coarse-grained or more fine-grained
/// API. Rollback rematerializations in-between each different wave of
/// rematerializations.
TEST(RematerializerTest, TreeRematRollback) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode
    %2:vgpr_32 = V_ADD_U32_e32 %0, %1, implicit $exec
    %3:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 3, implicit $exec, implicit $mode
    %4:vgpr_32 = V_ADD_U32_e32 %2, %3, implicit $exec

  bb.1:
    S_NOP 0, implicit %4
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    Rematerializer::DependencyReuseInfo DRI;
    Rollbacker Rollbacker;
    TR->addListener(&Rollbacker);

    const unsigned MBB0 = 0, MBB1 = 1;
    const RegisterIdx Cst0 = 0, Cst1 = 1, Add01 = 2, Cst3 = 3, Add23 = 4;

    // Rematerialize Add23 with all transitive dependencies.
    TR->rematerializeToRegion(Add23, MBB1, DRI);
    TR->updateLiveIntervals();

    EXPECT_NO_USERS(Cst0);
    EXPECT_NO_USERS(Cst1);
    EXPECT_NO_USERS(Add01);
    EXPECT_NO_USERS(Cst3);
    EXPECT_NO_USERS(Add23);

    TR.moveMIs(MBB0, MBB1, 5).addRematRegs(5);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();

    // After rollback all rematerializations are removed from the MIR.
    Rollbacker.rollback(*TR);
    TR.moveMIs(MBB1, MBB0, 5);
    ASSERT_REGION_SIZES();

    // Rematerialize Add23 only with its direct dependencies, reuse the rest.
    DRI.clear().reuse(Cst0).reuse(Cst1);
    TR->rematerializeToRegion(Add23, MBB1, DRI);
    TR->updateLiveIntervals();

    EXPECT_NUM_USERS(Cst0, 1);
    EXPECT_NUM_USERS(Cst1, 1);
    EXPECT_NO_USERS(Add01);
    EXPECT_NO_USERS(Cst3);
    EXPECT_NO_USERS(Add23);

    TR.moveMIs(MBB0, MBB1, 3).addRematRegs(3);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();

    // After rollback all rematerializations are removed from the MIR.
    Rollbacker.rollback(*TR);
    TR.moveMIs(MBB1, MBB0, 3);
    ASSERT_REGION_SIZES();

    // Rematerialize Add23 only with its direct dependencies as before, but
    // with as fine-grained operations as possible.
    MachineInstr *NopMI = &*TR->getRegion(MBB1).first;

    DRI.clear().reuse(Cst0).reuse(Cst1);
    const RegisterIdx RematAdd01 =
        TR->rematerializeToPos(Add01, MBB1, NopMI, DRI);
    EXPECT_NO_USERS(RematAdd01);
    EXPECT_NUM_USERS(Add01, 1);
    EXPECT_NUM_USERS(Cst0, 2);
    EXPECT_NUM_USERS(Cst1, 2);

    DRI.clear();
    const RegisterIdx RematCst3 =
        TR->rematerializeToPos(Cst3, MBB1, NopMI, DRI);
    EXPECT_NO_USERS(RematCst3);
    EXPECT_NUM_USERS(Cst3, 1);

    DRI.clear().useRemat(Add01, RematAdd01).useRemat(Cst3, RematCst3);
    const RegisterIdx RematAdd23 =
        TR->rematerializeToPos(Add23, MBB1, NopMI, DRI);
    EXPECT_NO_USERS(RematAdd23);
    EXPECT_NUM_USERS(Add23, 1);
    EXPECT_NUM_USERS(RematAdd01, 1);
    EXPECT_NUM_USERS(RematCst3, 1);

    TR->transferUser(Add23, RematAdd23, MBB1, *NopMI);
    EXPECT_NO_USERS(Add23);
    EXPECT_NUM_USERS(RematAdd23, 1);

    TR.moveMIs(MBB0, MBB1, 3).addRematRegs(3);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();
  });
}

/// To rematerialize %3 along with all its dependencies before its only use in
/// bb.1, we must first rematerialize %0 and %1 (in any order), then %2, and
/// finally %3. The rematerializer had a rematerialization order bug wherein,
/// because %0 is also used directly in the MI defining %3, it was
/// rematerialized after %2, breaking the invariant that dependencies of a
/// register must always be rematerialized before the register itself.
TEST(RematerializerTest, MultiplePathsRematOrder) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode
    %2:vgpr_32 = V_ADD_U32_e32 %0, %1, implicit $exec
    %3:vgpr_32 = V_ADD_U32_e32 %0, %2, implicit $exec

  bb.1:
    S_NOP 0, implicit %3
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    Rematerializer::DependencyReuseInfo DRI;
    const unsigned MBB1 = 1;
    const RegisterIdx Add02 = 3;
    TR->rematerializeToRegion(Add02, MBB1, DRI);
  });
}

/// Rematerializes a single register to multiple regions, tracking that
/// rematerializations are linked correctly and making sure that the original
/// register is deleted automatically when it no longer has any uses.
TEST(RematerializerTest, MultiRegionsRemat) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode

  bb.1:
    S_NOP 0, implicit %0, implicit %0

  bb.2:
    S_NOP 0, implicit %0
    S_NOP 0, implicit %0

  bb.3:
    S_NOP 0, implicit %0
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    Rematerializer::DependencyReuseInfo DRI;

    const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2, MBB3 = 3;
    const RegisterIdx Cst0 = 0;

    // Rematerialization to MBB1.
    const RegisterIdx RematBB1 = TR->rematerializeToRegion(Cst0, MBB1, DRI);
    TR.addMIs(MBB1, 1);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematBB1, Cst0, MBB1, 1);

    // Rematerialization to MBB2.
    DRI.clear();
    const RegisterIdx RematBB2 = TR->rematerializeToRegion(Cst0, MBB2, DRI);
    TR.addMIs(MBB2, 1);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematBB2, Cst0, MBB2, 2);

    // Rematerialization to MBB3. Rematerializing to the last original user
    // deletes the original register.
    DRI.clear();
    const RegisterIdx RematBB3 = TR->rematerializeToRegion(Cst0, MBB3, DRI);
    TR.moveMIs(MBB0, MBB3, 1);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematBB3, Cst0, MBB3, 1);
  });
}

/// Rematerializes a tree of register with some unrematerializable operands to a
/// final destination in two steps, creating rematerializations of
/// rematerializations in the process. Make sure that origins of
/// rematerializations are always original registers.
TEST(RematerializerTest, MultiStep) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode, implicit-def $m0
    %2:vgpr_32 = V_ADD_U32_e32 %0, %1, implicit $exec
    S_NOP 0, implicit %0

  bb.1:
    %3:vgpr_32 = V_ADD_U32_e32 %2, %2, implicit $exec

  bb.2:
    S_NOP 0, implicit %3
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    Rematerializer::DependencyReuseInfo DRI;

    const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2;
    const RegisterIdx Cst0 = 0, Add01 = 1, Add22 = 2, RematCst0 = 3,
                      RematAdd01 = 4, RematRematAdd01 = 5, RematAdd22 = 6;

    // Rematerialize Add01 from the first to the second block along with its
    // single rematerializable dependency (constant 0). The constant 1 has an
    // implicit def that is non-ignorable so it cannot be rematerialized. The
    // constant 0 remains in the first block because it has a user there, but
    // the add is deleted.
    TR->rematerializeToRegion(Add01, MBB1, DRI);
    TR.removeMIs(MBB0, 1).addMIs(MBB1, 2);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematCst0, Cst0, MBB1, 1);
    EXPECT_REMAT(RematAdd01, Add01, MBB1, 1);

    // We are going to re-rematerialize a register so the LIS need to be
    // up-to-date.
    TR->updateLiveIntervals();

    // Rematerialize Add22 from the second to the third block, which will also
    // indirectly rematerialize RematAdd01; make sure the latter's
    // rematerialization's origin is the original register, not RematAdd01.
    DRI.clear().reuse(RematCst0);
    TR->rematerializeToRegion(Add22, MBB2, DRI);
    TR.moveMIs(MBB1, MBB2, 2);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematRematAdd01, Add01, MBB2, 1);
    EXPECT_REMAT(RematAdd22, Add22, MBB2, 1);
  });
}

/// Checks that it is possible to rematerialize inside a region that was
/// rendered empty by previous rematerializations (as long as the region ends
/// with a terminator).
TEST(RematerializerTest, EmptyRegion) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode

  bb.1:
    %2:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 2, implicit $exec, implicit $mode

  bb.2:
    %3:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 3, implicit $exec, implicit $mode
    S_BRANCH %bb.3

  bb.3:
    S_NOP 0, implicit %0, implicit %1
    S_NOP 0, implicit %2, implicit %3
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    Rematerializer::DependencyReuseInfo DRI;

    const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2, MBB3 = 3;
    const RegisterIdx Cst0 = 0, Cst1 = 1, Cst2 = 2, Cst3 = 3;

    // After rematerializing %2 and %3 to bb.3, their respective original
    // defining regions are empty. %2's region ends at the end of its parent
    // block, whereas %3's region ends at a terminator MI (S_BRANCH).
    TR->rematerializeToRegion(Cst2, MBB3, DRI);
    TR->rematerializeToRegion(Cst3, MBB3, DRI.clear());
    TR.removeMIs(MBB1, 1).removeMIs(MBB2, 1).addMIs(MBB3, 2);
    ASSERT_REGION_SIZES();

    // Move %0 to the empty MBB1 block/region.
    const RegisterIdx RematCst0 =
        TR->rematerializeToRegion(Cst0, MBB1, DRI.clear());
    TR->transferRegionUsers(Cst0, RematCst0, MBB3);

    // Move %1 to the empty MBB2 region, right before the S_BRANCH terminator.
    const RegisterIdx RematCst1 = TR->rematerializeToPos(
        Cst1, MBB2, TR->getRegion(MBB2).first, DRI.clear());
    TR->transferRegionUsers(Cst1, RematCst1, MBB3);

    TR.removeMIs(MBB0, 2).addMIs(MBB1, 1).addMIs(MBB2, 1);
    ASSERT_REGION_SIZES();
  });
}

/// Checks rematerializability of sub-registers in various situations. Performs
/// simple rematerializations on them as well.
TEST(RematerializerTest, SubRegRematSupport) {
  StringRef MIRBody = R"MIR(
  bb.0:
    undef %01.sub0:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %01.sub1:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode

    undef %2.sub0:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 2, implicit $exec, implicit $mode

    undef %34.sub0:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 3, implicit $exec, implicit $mode
    
    undef %56.sub0:sreg_64 = S_MOV_B32 5
    %56.sub1:sreg_64 = S_MOV_B32 6, implicit-def $m0    
    
    undef %78.sub0:sreg_64 = S_MOV_B32 7
    S_NOP 0, implicit %78.sub0
    %78.sub1:sreg_64 = S_MOV_B32 8
    
    undef %99.sub0:sreg_64 = S_MOV_B32 9
    %99.sub1:sreg_64 = S_MOV_B32 %99.sub0
    
  bb.1:
    %34.sub1:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 4, implicit $exec, implicit $mode

    S_NOP 0, implicit %01, implicit %2, implicit %34, implicit %56, implicit %78, implicit %99
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    const unsigned MBB0 = 0, MBB1 = 1;
    const RegisterIdx Cst01 = 0, Cst2 = 1, Cst99 = 2;

    // - %34 is not rematerializable because it is defined over multiple
    // regions.
    // - %56 is not rematerializable because the second defining MI is
    // unrematerializable due to the implicit def.
    // - %78 is not rematerializable because it is read by an MI not defining it
    // before its last definition.
    EXPECT_EQ(TR->getNumRegs(), 3U);

    auto CheckBasicRemat = [&](RegisterIdx RegIdx,
                               unsigned NumExpectDefs) -> void {
      Rematerializer::DependencyReuseInfo DRI;
      EXPECT_EQ(TR->getReg(RegIdx).Defs.size(), NumExpectDefs);
      const RegisterIdx Remat = TR->rematerializeToRegion(RegIdx, MBB1, DRI);
      TR.moveMIs(MBB0, MBB1, NumExpectDefs);
      ASSERT_REGION_SIZES();
      EXPECT_REMAT(Remat, RegIdx, MBB1, 1);
    };

    CheckBasicRemat(Cst01, 2);
    CheckBasicRemat(Cst2, 1);
    CheckBasicRemat(Cst99, 2);
  });
}

/// Checks that the user transfer logic works correctly when different defining
/// MIs of the same rematerializable register start dependening on different
/// versions (original and rematerialized) of the same register.
TEST(RematerializerTest, SubRegUserTransfer) {
  StringRef MIRBody = R"MIR(
  bb.0:
    undef %01.sub0:sreg_64 = S_MOV_B32 0
    %01.sub1:sreg_64 = S_MOV_B32 1    
    
  bb.1:
    undef %23.sub0:sreg_64 = S_MOV_B32 %01.sub0
    %23.sub1:sreg_64 = S_MOV_B32 %01.sub1
    S_NOP 0, implicit %23
  
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    Rematerializer::DependencyReuseInfo DRI;
    Rollbacker Rollback;
    TR->addListener(&Rollback);

    const unsigned MBB1 = 1;
    const RegisterIdx Cst01 = 0, Cst23 = 1;
    EXPECT_EQ(TR->getReg(Cst01).Defs.size(), 2U);
    EXPECT_EQ(TR->getReg(Cst23).Defs.size(), 2U);
    MachineInstr *Cst23FirstDef = TR->getReg(Cst23).Defs[0];
    MachineInstr *Cst23SecondDef = TR->getReg(Cst23).Defs[1];

    // Create a rematerialization of %01 just before %23.
    const RegisterIdx RematCst01 =
        TR->rematerializeToPos(Cst01, MBB1, Cst23FirstDef, DRI);
    EXPECT_NUM_USERS(Cst01, 2);
    EXPECT_NUM_USERS(RematCst01, 0);
    EXPECT_NUM_USERS(Cst23, 1);
    EXPECT_NUM_DEPENDENCIES(Cst23, 1);

    // Have the first def of %23 use the rematerialization of %01 (the second
    // def still uses %01). This transfers a user to the rematerialization of
    // %01 and adds the rematerialization of %01 as a rematerializable
    // dependency to %23.
    TR->transferUser(Cst01, RematCst01, MBB1, *Cst23FirstDef);
    EXPECT_NUM_USERS(Cst01, 1);
    EXPECT_NUM_USERS(RematCst01, 1);
    EXPECT_NUM_USERS(Cst23, 1);
    EXPECT_NUM_DEPENDENCIES(Cst23, 2);

    // Have the second def of %23 use the rematerialization of %01 as well. This
    // transfers a user to the rematerialization of %01 and removes %01 as a
    // rematerializable dependency of %23.
    TR->transferUser(Cst01, RematCst01, MBB1, *Cst23SecondDef);
    EXPECT_NUM_USERS(Cst01, 0);
    EXPECT_NUM_USERS(RematCst01, 2);
    EXPECT_NUM_DEPENDENCIES(Cst23, 1);

    // Rollback should restore everything to its original state.
    Rollback.rollback(*TR);
    EXPECT_NUM_USERS(Cst01, 2);
    EXPECT_NUM_USERS(RematCst01, 0);
    EXPECT_NUM_USERS(Cst23, 1);
    EXPECT_NUM_DEPENDENCIES(Cst23, 1);
  });
}

TEST(RematerializerTest, SubRegRollback) {
  StringRef MIRBody = R"MIR(
  bb.0:
    undef %01.sub0:sreg_64 = S_MOV_B32 0
    %unremat0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode, implicit-def $m0
    %01.sub1:sreg_64 = S_MOV_B32 1    
    %unremat1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode, implicit-def $m0
  
  bb.1:
    undef %23.sub0:sreg_64 = S_MOV_B32 2
    %23.sub1:sreg_64 = S_MOV_B32 3

  bb.2:
    undef %45.sub0:sreg_64 = S_MOV_B32 4
    undef %67.sub0:sreg_64 = S_MOV_B32 6
    %45.sub1:sreg_64 = S_MOV_B32 5
    %67.sub1:sreg_64 = S_MOV_B32 7

  bb.3:
    S_NOP 0, implicit %01, implicit %23, implicit %45, implicit %67
    S_NOP 0, implicit %unremat0, implicit %unremat1 
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    Rematerializer::DependencyReuseInfo DRI;
    Rollbacker Rollback;
    TR->addListener(&Rollback);

    const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2, MBB3 = 3;
    const RegisterIdx Cst01 = 0, Cst23 = 1, Cst45 = 2, Cst67 = 3;

    EXPECT_EQ(TR->getReg(Cst01).Defs.size(), 2U);
    EXPECT_EQ(TR->getReg(Cst23).Defs.size(), 2U);
    EXPECT_EQ(TR->getReg(Cst45).Defs.size(), 2U);
    EXPECT_EQ(TR->getReg(Cst67).Defs.size(), 2U);

    auto GetNextMI = [&](MachineInstr *MI) -> MachineInstr * {
      return &*std::next(MI->getIterator());
    };

    auto GetDefMI = [&](RegisterIdx RegIdx, unsigned DefIdx) -> MachineInstr * {
      return TR->getReg(RegIdx).Defs[DefIdx];
    };

    // Rematerialize and rollback %01.
    MachineInstr *Unremat0 = GetNextMI(GetDefMI(Cst01, 0));
    MachineInstr *Unremat1 = GetNextMI(GetDefMI(Cst01, 1));
    const RegisterIdx RematCst01 =
        TR->rematerializeToRegion(Cst01, MBB3, DRI.clear());
    TR.moveMIs(MBB0, MBB3, 2);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematCst01, Cst01, MBB3, 1);

    // Rollback must re-create MIs in the same order.
    Rollback.rollback(*TR);
    TR.moveMIs(MBB3, MBB0, 2);
    ASSERT_REGION_SIZES();
    EXPECT_EQ(Unremat0, GetNextMI(GetDefMI(Cst01, 0)));
    EXPECT_EQ(Unremat1, GetNextMI(GetDefMI(Cst01, 1)));

    // Rematerialize and rollback %23.
    MachineBasicBlock::iterator EndOfMBB1 =
        std::next(GetDefMI(Cst23, 1)->getIterator());
    const RegisterIdx RematCst23 =
        TR->rematerializeToRegion(Cst23, MBB3, DRI.clear());
    TR.moveMIs(MBB1, MBB3, 2);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematCst23, Cst23, MBB3, 1);

    // Rollback must re-create MIs in the same order.
    Rollback.rollback(*TR);
    TR.moveMIs(MBB3, MBB1, 2);
    ASSERT_REGION_SIZES();
    MachineInstr *Cst23Def0 = GetDefMI(Cst23, 0);
    MachineInstr *Cst23Def1 = GetDefMI(Cst23, 1);
    EXPECT_EQ(Cst23Def1, GetNextMI(Cst23Def0));
    EXPECT_EQ(EndOfMBB1, std::next(Cst23Def1->getIterator()));

    // Rematerialize and rollback %45 and %67.
    MachineBasicBlock::iterator EndOfMBB2 =
        std::next(GetDefMI(Cst67, 1)->getIterator());
    const RegisterIdx RematCst45 =
        TR->rematerializeToRegion(Cst45, MBB3, DRI.clear());
    const RegisterIdx RematCst67 =
        TR->rematerializeToRegion(Cst67, MBB3, DRI.clear());
    TR.moveMIs(MBB2, MBB3, 4);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematCst45, Cst45, MBB3, 1);
    EXPECT_REMAT(RematCst67, Cst67, MBB3, 1);

    // Rollback must re-create MIs in the same order.
    Rollback.rollback(*TR);
    TR.moveMIs(MBB3, MBB2, 4);
    ASSERT_REGION_SIZES();
    MachineInstr *Cst45Def0 = GetDefMI(Cst45, 0);
    MachineInstr *Cst67Def0 = GetDefMI(Cst67, 0);
    MachineInstr *Cst45Def1 = GetDefMI(Cst45, 1);
    MachineInstr *Cst67Def1 = GetDefMI(Cst67, 1);
    EXPECT_EQ(Cst67Def0, GetNextMI(Cst45Def0));
    EXPECT_EQ(Cst45Def1, GetNextMI(Cst67Def0));
    EXPECT_EQ(Cst67Def1, GetNextMI(Cst45Def1));
    EXPECT_EQ(EndOfMBB2, std::next(Cst67Def1->getIterator()));
  });
}

/// Checks that instructions which use a rematerializable register as their
/// first operand (here the KILL pseudo) are not treated as defining
/// instructions for that register.
TEST(RematerializerTest, FirstOperandNotDef) {
  StringRef MIRBody = R"MIR(
  bb.0:
    undef %0.sub0:sgpr_64 = S_MOV_B32 0
    KILL %0
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    Rematerializer::DependencyReuseInfo DRI;

    const RegisterIdx Cst0 = 0;
    EXPECT_EQ(TR->getNumRegs(), 1U);
    EXPECT_EQ(TR->getReg(Cst0).Defs.size(), 1U);
    EXPECT_NUM_USERS(Cst0, 1);
  });
}

/// The rematerializer had a bug where re-creating the interval of a
/// non-rematerializable super-register defined over multiple MIs, some of which
/// defining entirely dead subregisters, could cause a crash when changing the
/// order of sub-definitions (for example during scheduling) because the
/// re-created interval could end up with multiple connected components, which
/// is illegal. The solution is to split separate components of the interval in
/// such cases.
TEST(RematerializerTest, SplitSubRegDeadDef) {
  StringRef MIRBody = R"MIR(
  bb.0:
    undef %0.sub0:vreg_64 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode, implicit-def $m0
    %0.sub1:vreg_64 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode, implicit-def $m0
    %1:vgpr_32 = V_ADD_U32_e32 %0.sub0, %0.sub0, implicit $exec

  bb.1:
    S_NOP 0, implicit %1
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    MachineFunction &MF = TR.MF;
    LiveIntervals &LIS = TR.LIS;

    // Replicates the scheduler's effect on LIS on an intra-block move of MI.
    auto MoveMIAndAdjustLiveness = [&](MachineInstr &MI) {
      LIS.handleMove(MI);
      const MachineRegisterInfo &MRI = MF.getRegInfo();
      const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
      RegisterOperands RegOpers;
      RegOpers.collect(MI, TRI, MRI, true, /*IgnoreDead=*/false);
      SlotIndex Sub1Slot = LIS.getInstructionIndex(MI).getRegSlot();
      RegOpers.adjustLaneLiveness(LIS, MRI, Sub1Slot, &MI);
    };

    MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
    MachineInstr &Sub0Def = *MBB0.begin();
    MachineInstr &Sub1Def = *MBB0.begin()->getNextNode();

    // Flip %0's subdefinition order.
    MBB0.splice(Sub0Def.getIterator(), &MBB0, Sub1Def.getIterator());
    MoveMIAndAdjustLiveness(Sub1Def);

    // Rematerialize %1 to bb.1. This triggers a live-interval update of %0 when
    // calling TR->updateLiveIntervals(), during which its interval is split.
    Rematerializer::DependencyReuseInfo DRI;
    const unsigned MBB1 = 1;
    const RegisterIdx Add = 0;
    TR->rematerializeToRegion(Add, MBB1, DRI);
    TR->updateLiveIntervals();

    // If we didn't split %0 before, its definitions would now look like:
    // dead undef %0.sub1:vreg_64 = IMPLICIT_DEF
    // undef %0.sub0:vreg_64 = IMPLICIT_DEF
    //
    // Trying to flip back %0's definition order then triggers an error in
    // LIS.handleMove because its live interval is made up of multiple connected
    // components.
    ASSERT_NE(Sub0Def.getOperand(0).getReg(), Sub1Def.getOperand(0).getReg());
    MBB0.splice(MBB0.end(), &MBB0, Sub1Def.getIterator());
    MoveMIAndAdjustLiveness(Sub1Def);
  });
}

/// Checks that rollback works as expected when the rollback listener is added
/// mid-rematerializations.
TEST(RematerializerTest, Rollback) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode

  bb.1:
    S_NOP 0, implicit %0, implicit %1

  bb.2:
    S_NOP 0, implicit %0, implicit %1
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    Rematerializer::DependencyReuseInfo DRI;

    const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2;
    const RegisterIdx Cst0 = 0, Cst1 = 1;

    // Rematerialize %0 to MBB1, taking one user from the original register.
    RegisterIdx RematCst0MBB1 = TR->rematerializeToRegion(Cst0, MBB1, DRI);
    TR.addMIs(MBB1, 1).addRematRegs(1);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();

    Rollbacker Rollback;
    TR->addListener(&Rollback);

    // Rematerialize %0 to MBB2 and %1 to MBB1/MBB2; each rematerialization ends
    // up with a single user and both original registers are deleted.
    RegisterIdx RematCst0MBB2 =
        TR->rematerializeToRegion(Cst0, MBB2, DRI.clear());
    RegisterIdx RematCst1MBB1 =
        TR->rematerializeToRegion(Cst1, MBB1, DRI.clear());
    RegisterIdx RematCst1MBB2 =
        TR->rematerializeToRegion(Cst1, MBB2, DRI.clear());

    TR.removeMIs(MBB0, 2).addMIs(MBB1, 1).addMIs(MBB2, 2).addRematRegs(3);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();

    EXPECT_NO_USERS(Cst0);
    EXPECT_NO_USERS(Cst1);
    EXPECT_NUM_USERS(RematCst0MBB1, 1);
    EXPECT_NUM_USERS(RematCst0MBB2, 1);
    EXPECT_NUM_USERS(RematCst1MBB1, 1);
    EXPECT_NUM_USERS(RematCst1MBB2, 1);

    // Rollback all changes since the rollbacker was added. The first
    // rematerialization of %0 to MBB1 happened before so it is not rolled back.
    // However %0 is re-created because it was deleted after.
    Rollback.rollback(*TR);

    TR.addMIs(MBB0, 2).removeMIs(MBB1, 1).removeMIs(MBB2, 2);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();

    EXPECT_NUM_USERS(Cst0, 1);
    EXPECT_NUM_USERS(Cst1, 2);
    EXPECT_NUM_USERS(RematCst0MBB1, 1);
    EXPECT_NO_USERS(RematCst0MBB2);
    EXPECT_NO_USERS(RematCst1MBB1);
    EXPECT_NO_USERS(RematCst1MBB2);
  });
}

/// Checks that rollback re-creates MIs at correct positions when the order of
/// register deletions makes cached insert positions stale.
TEST(RematerializerTest, RollbackOrder) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode
    %2:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 2, implicit $exec, implicit $mode
    %3:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 3, implicit $exec, implicit $mode

  bb.1:
    S_NOP 0, implicit %0, implicit %1, implicit %2, implicit %3
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](TestRematerializer &TR) {
    Rematerializer::DependencyReuseInfo DRI;
    Rollbacker Rollback;
    TR->addListener(&Rollback);

    const unsigned MBB0 = 0, MBB1 = 1;
    const RegisterIdx Cst0 = 0, Cst1 = 1, Cst2 = 2, Cst3 = 3;

    auto RematToMBB1 = [&](RegisterIdx RegIdx) -> void {
      // Rematerialize %RegIdx to MBB1, deleting the original register.
      TR->rematerializeToRegion(RegIdx, MBB1, DRI.clear());
      TR.moveMIs(MBB0, MBB1, 1);
      ASSERT_REGION_SIZES();
    };

    auto GetNextMI = [&](MachineInstr *MI) -> MachineInstr * {
      return &*std::next(MI->getIterator());
    };

    auto RollbackAndCheckOriginalOrder = [&]() -> void {
      // Rollback and check for correct instruction order in the original
      // defining region. The asserts on region sizes ensure that all original
      // registers were indeed deleted and will be re-created in the original
      // region.
      Rollback.rollback(*TR);
      TR.moveMIs(MBB1, MBB0, 3);
      ASSERT_REGION_SIZES();

      MachineInstr *DefCst0 = TR->getReg(Cst0).getFirstDef();
      MachineInstr *DefCst1 = TR->getReg(Cst1).getFirstDef();
      MachineInstr *DefCst2 = TR->getReg(Cst2).getFirstDef();
      MachineInstr *DefCst3 = TR->getReg(Cst3).getFirstDef();
      EXPECT_EQ(GetNextMI(DefCst0), DefCst1);
      EXPECT_EQ(GetNextMI(DefCst1), DefCst2);
      EXPECT_EQ(GetNextMI(DefCst2), DefCst3);
    };

    // Test every possible rematerialization order.

    RematToMBB1(Cst0);
    RematToMBB1(Cst1);
    RematToMBB1(Cst2);
    RollbackAndCheckOriginalOrder();

    RematToMBB1(Cst0);
    RematToMBB1(Cst2);
    RematToMBB1(Cst1);
    RollbackAndCheckOriginalOrder();

    RematToMBB1(Cst1);
    RematToMBB1(Cst0);
    RematToMBB1(Cst2);
    RollbackAndCheckOriginalOrder();

    RematToMBB1(Cst1);
    RematToMBB1(Cst2);
    RematToMBB1(Cst0);
    RollbackAndCheckOriginalOrder();

    RematToMBB1(Cst2);
    RematToMBB1(Cst0);
    RematToMBB1(Cst1);
    RollbackAndCheckOriginalOrder();

    RematToMBB1(Cst2);
    RematToMBB1(Cst1);
    RematToMBB1(Cst0);
    RollbackAndCheckOriginalOrder();
  });
}
