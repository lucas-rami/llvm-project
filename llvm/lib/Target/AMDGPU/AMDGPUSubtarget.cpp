//===-- AMDGPUSubtarget.cpp - AMDGPU Subtarget Information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implements the AMDGPU specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSubtarget.h"
#include "AMDGPUCallLowering.h"
#include "AMDGPUInstructionSelector.h"
#include "AMDGPULegalizerInfo.h"
#include "AMDGPURegisterBankInfo.h"
#include "R600Subtarget.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/GlobalISel/InlineAsmLowering.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/IR/MDBuilder.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-subtarget"

AMDGPUSubtarget::AMDGPUSubtarget(Triple TT) : TargetTriple(std::move(TT)) {}

bool AMDGPUSubtarget::useRealTrue16Insts() const {
  return hasTrue16BitInsts() && EnableRealTrue16Insts;
}

// Returns the maximum per-workgroup LDS allocation size (in bytes) that still
// allows the given function to achieve an occupancy of NWaves waves per
// SIMD / EU, taking into account only the function's *maximum* workgroup size.
unsigned
AMDGPUSubtarget::getMaxLocalMemSizeWithWaveCount(unsigned NWaves,
                                                 const Function &F) const {
  const unsigned WaveSize = getWavefrontSize();
  const unsigned WorkGroupSize = getFlatWorkGroupSizes(F).second;
  const unsigned WavesPerWorkgroup =
      std::max(1u, (WorkGroupSize + WaveSize - 1) / WaveSize);

  const unsigned WorkGroupsPerCU =
      std::max(1u, (NWaves * getEUsPerCU()) / WavesPerWorkgroup);

  return getLocalMemorySize() / WorkGroupsPerCU;
}

unsigned
AMDGPUSubtarget::getOccupancyWithLocalMemSize(unsigned LDSNumGroups,
                                              unsigned WorkGroupSize) const {
  // Compute the maximum number of workgroups of the specified size that can fit
  // entirely on a CU concurrently.
  const unsigned MaxWorkGroupsPerCu = getMaxWorkGroupsPerCU(WorkGroupSize);
  if (!MaxWorkGroupsPerCu)
    return 0;

  // Compute the maximum number of waves that can be on the CU concurrently.
  // This is based on the maximum amount of concurrent workgroups on a CU, a
  // function of work group size and LDS usage, as well as the number of waves
  // in each workgroup.
  unsigned MaxWaves = std::min(MaxWorkGroupsPerCu, LDSNumGroups) *
                      divideCeil(WorkGroupSize, getWavefrontSize());

  // FIXME: Does the final number of waves need to be a multiple of the group
  // size?

  // Return the maximum number of waves on any EU, assuming that all wavefronts
  // are spread across all EUs as evenly as possible. Clamp to the maximum
  // number of concurrent wavefronts that a EU can support.
  return std::min(divideCeil(MaxWaves, getEUsPerCU()), getMaxWavesPerEU());
}

std::pair<unsigned, unsigned>
AMDGPUSubtarget::getOccupancyWithLocalMemSize(uint32_t Bytes,
                                              const Function &F) const {
  // FIXME: Is there an allocation granularity for the LDS? If so we need to
  // make sure the amount of bytes is aligned on that granularity.

  // Compute occupancy restriction based on LDS usage.
  const unsigned LDSNumGroups = getLocalMemorySize() / (Bytes ? Bytes : 1u);
  
  // Queried LDS size may be larger than available on a CU.
  if (!LDSNumGroups)
    return {1, 1};

  // The maximum group size (second element of the group size pair) will yield
  // the minimum occupancy so it is the first element of the returned pair.
  std::pair<unsigned, unsigned> WorkGroupSizeRange = getFlatWorkGroupSizes(F);
  return {getOccupancyWithLocalMemSize(LDSNumGroups, WorkGroupSizeRange.second),
          getOccupancyWithLocalMemSize(LDSNumGroups, WorkGroupSizeRange.first)};
}

std::pair<unsigned, unsigned>
AMDGPUSubtarget::getOccupancyWithLocalMemSize(const MachineFunction &MF) const {
  const auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
  return getOccupancyWithLocalMemSize(MFI->getLDSSize(), MF.getFunction());
}

std::pair<unsigned, unsigned>
AMDGPUSubtarget::getDefaultFlatWorkGroupSize(CallingConv::ID CC) const {
  switch (CC) {
  case CallingConv::AMDGPU_VS:
  case CallingConv::AMDGPU_LS:
  case CallingConv::AMDGPU_HS:
  case CallingConv::AMDGPU_ES:
  case CallingConv::AMDGPU_GS:
  case CallingConv::AMDGPU_PS:
    return std::pair(1, getWavefrontSize());
  default:
    return std::pair(1u, getMaxFlatWorkGroupSize());
  }
}

std::pair<unsigned, unsigned>
AMDGPUSubtarget::getFlatWorkGroupSizes(const Function &F) const {
  // Default minimum/maximum flat work group sizes.
  std::pair<unsigned, unsigned> Default =
      getDefaultFlatWorkGroupSize(F.getCallingConv());

  // Requested minimum/maximum flat work group sizes.
  std::pair<unsigned, unsigned> Requested = AMDGPU::getIntegerPairAttribute(
      F, "amdgpu-flat-work-group-size", Default);

  // Make sure requested minimum is less than requested maximum.
  if (Requested.first > Requested.second)
    return Default;

  // Make sure requested values do not violate subtarget's specifications.
  if (Requested.first < getMinFlatWorkGroupSize())
    return Default;
  if (Requested.second > getMaxFlatWorkGroupSize())
    return Default;

  return Requested;
}

std::pair<unsigned, unsigned> AMDGPUSubtarget::getEffectiveWavesPerEU(
    std::pair<unsigned, unsigned> Requested,
    std::pair<unsigned, unsigned> FlatWorkGroupSizes) const {
  // Default minimum/maximum number of waves per execution unit.
  std::pair<unsigned, unsigned> Default(1, getMaxWavesPerEU());

  // If minimum/maximum flat work group sizes were explicitly requested using
  // "amdgpu-flat-workgroup-size" attribute, then set default minimum/maximum
  // number of waves per execution unit to values implied by requested
  // minimum/maximum flat work group sizes.
  unsigned MinImpliedByFlatWorkGroupSize =
      getWavesPerEUForWorkGroup(FlatWorkGroupSizes.second);
  Default.first = MinImpliedByFlatWorkGroupSize;

  // Make sure requested minimum is less than requested maximum.
  if (Requested.second && Requested.first > Requested.second)
    return Default;

  // Make sure requested values do not violate subtarget's specifications.
  if (Requested.first < getMinWavesPerEU() ||
      Requested.second > getMaxWavesPerEU())
    return Default;

  // Make sure requested values are compatible with values implied by
  // requested minimum/maximum flat work group sizes.
  if (Requested.first < MinImpliedByFlatWorkGroupSize)
    return Default;

  return Requested;
}

std::pair<unsigned, unsigned> AMDGPUSubtarget::getWavesPerEU(
    const Function &F, std::pair<unsigned, unsigned> FlatWorkGroupSizes) const {
  // Default minimum/maximum number of waves per execution unit.
  std::pair<unsigned, unsigned> Default(1, getMaxWavesPerEU());

  // Requested minimum/maximum number of waves per execution unit.
  std::pair<unsigned, unsigned> Requested =
      AMDGPU::getIntegerPairAttribute(F, "amdgpu-waves-per-eu", Default, true);
  return getEffectiveWavesPerEU(Requested, FlatWorkGroupSizes);
}

static unsigned getReqdWorkGroupSize(const Function &Kernel, unsigned Dim) {
  auto *Node = Kernel.getMetadata("reqd_work_group_size");
  if (Node && Node->getNumOperands() == 3)
    return mdconst::extract<ConstantInt>(Node->getOperand(Dim))->getZExtValue();
  return std::numeric_limits<unsigned>::max();
}

bool AMDGPUSubtarget::isMesaKernel(const Function &F) const {
  return isMesa3DOS() && !AMDGPU::isShader(F.getCallingConv());
}

unsigned AMDGPUSubtarget::getMaxWorkitemID(const Function &Kernel,
                                           unsigned Dimension) const {
  unsigned ReqdSize = getReqdWorkGroupSize(Kernel, Dimension);
  if (ReqdSize != std::numeric_limits<unsigned>::max())
    return ReqdSize - 1;
  return getFlatWorkGroupSizes(Kernel).second - 1;
}

bool AMDGPUSubtarget::isSingleLaneExecution(const Function &Func) const {
  for (int I = 0; I < 3; ++I) {
    if (getMaxWorkitemID(Func, I) > 0)
      return false;
  }

  return true;
}

bool AMDGPUSubtarget::makeLIDRangeMetadata(Instruction *I) const {
  Function *Kernel = I->getParent()->getParent();
  unsigned MinSize = 0;
  unsigned MaxSize = getFlatWorkGroupSizes(*Kernel).second;
  bool IdQuery = false;

  // If reqd_work_group_size is present it narrows value down.
  if (auto *CI = dyn_cast<CallInst>(I)) {
    const Function *F = CI->getCalledFunction();
    if (F) {
      unsigned Dim = UINT_MAX;
      switch (F->getIntrinsicID()) {
      case Intrinsic::amdgcn_workitem_id_x:
      case Intrinsic::r600_read_tidig_x:
        IdQuery = true;
        [[fallthrough]];
      case Intrinsic::r600_read_local_size_x:
        Dim = 0;
        break;
      case Intrinsic::amdgcn_workitem_id_y:
      case Intrinsic::r600_read_tidig_y:
        IdQuery = true;
        [[fallthrough]];
      case Intrinsic::r600_read_local_size_y:
        Dim = 1;
        break;
      case Intrinsic::amdgcn_workitem_id_z:
      case Intrinsic::r600_read_tidig_z:
        IdQuery = true;
        [[fallthrough]];
      case Intrinsic::r600_read_local_size_z:
        Dim = 2;
        break;
      default:
        break;
      }

      if (Dim <= 3) {
        unsigned ReqdSize = getReqdWorkGroupSize(*Kernel, Dim);
        if (ReqdSize != std::numeric_limits<unsigned>::max())
          MinSize = MaxSize = ReqdSize;
      }
    }
  }

  if (!MaxSize)
    return false;

  // Range metadata is [Lo, Hi). For ID query we need to pass max size
  // as Hi. For size query we need to pass Hi + 1.
  if (IdQuery)
    MinSize = 0;
  else
    ++MaxSize;

  APInt Lower{32, MinSize};
  APInt Upper{32, MaxSize};
  if (auto *CI = dyn_cast<CallBase>(I)) {
    ConstantRange Range(Lower, Upper);
    CI->addRangeRetAttr(Range);
  } else {
    MDBuilder MDB(I->getContext());
    MDNode *MaxWorkGroupSizeRange = MDB.createRange(Lower, Upper);
    I->setMetadata(LLVMContext::MD_range, MaxWorkGroupSizeRange);
  }
  return true;
}

unsigned AMDGPUSubtarget::getImplicitArgNumBytes(const Function &F) const {
  assert(AMDGPU::isKernel(F.getCallingConv()));

  // We don't allocate the segment if we know the implicit arguments weren't
  // used, even if the ABI implies we need them.
  if (F.hasFnAttribute("amdgpu-no-implicitarg-ptr"))
    return 0;

  if (isMesaKernel(F))
    return 16;

  // Assume all implicit inputs are used by default
  const Module *M = F.getParent();
  unsigned NBytes =
      AMDGPU::getAMDHSACodeObjectVersion(*M) >= AMDGPU::AMDHSA_COV5 ? 256 : 56;
  return F.getFnAttributeAsParsedInteger("amdgpu-implicitarg-num-bytes",
                                         NBytes);
}

uint64_t AMDGPUSubtarget::getExplicitKernArgSize(const Function &F,
                                                 Align &MaxAlign) const {
  assert(F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
         F.getCallingConv() == CallingConv::SPIR_KERNEL);

  const DataLayout &DL = F.getDataLayout();
  uint64_t ExplicitArgBytes = 0;
  MaxAlign = Align(1);

  for (const Argument &Arg : F.args()) {
    if (Arg.hasAttribute("amdgpu-hidden-argument"))
      continue;

    const bool IsByRef = Arg.hasByRefAttr();
    Type *ArgTy = IsByRef ? Arg.getParamByRefType() : Arg.getType();
    Align Alignment = DL.getValueOrABITypeAlignment(
        IsByRef ? Arg.getParamAlign() : std::nullopt, ArgTy);
    uint64_t AllocSize = DL.getTypeAllocSize(ArgTy);
    ExplicitArgBytes = alignTo(ExplicitArgBytes, Alignment) + AllocSize;
    MaxAlign = std::max(MaxAlign, Alignment);
  }

  return ExplicitArgBytes;
}

unsigned AMDGPUSubtarget::getKernArgSegmentSize(const Function &F,
                                                Align &MaxAlign) const {
  if (F.getCallingConv() != CallingConv::AMDGPU_KERNEL &&
      F.getCallingConv() != CallingConv::SPIR_KERNEL)
    return 0;

  uint64_t ExplicitArgBytes = getExplicitKernArgSize(F, MaxAlign);

  unsigned ExplicitOffset = getExplicitKernelArgOffset();

  uint64_t TotalSize = ExplicitOffset + ExplicitArgBytes;
  unsigned ImplicitBytes = getImplicitArgNumBytes(F);
  if (ImplicitBytes != 0) {
    const Align Alignment = getAlignmentForImplicitArgPtr();
    TotalSize = alignTo(ExplicitArgBytes, Alignment) + ImplicitBytes;
    MaxAlign = std::max(MaxAlign, Alignment);
  }

  // Being able to dereference past the end is useful for emitting scalar
  // loads.
  return alignTo(TotalSize, 4);
}

AMDGPUDwarfFlavour AMDGPUSubtarget::getAMDGPUDwarfFlavour() const {
  return getWavefrontSize() == 32 ? AMDGPUDwarfFlavour::Wave32
                                  : AMDGPUDwarfFlavour::Wave64;
}

const AMDGPUSubtarget &AMDGPUSubtarget::get(const MachineFunction &MF) {
  if (MF.getTarget().getTargetTriple().getArch() == Triple::amdgcn)
    return static_cast<const AMDGPUSubtarget &>(
        MF.getSubtarget<GCNSubtarget>());
  return static_cast<const AMDGPUSubtarget &>(MF.getSubtarget<R600Subtarget>());
}

const AMDGPUSubtarget &AMDGPUSubtarget::get(const TargetMachine &TM,
                                            const Function &F) {
  if (TM.getTargetTriple().getArch() == Triple::amdgcn)
    return static_cast<const AMDGPUSubtarget &>(
        TM.getSubtarget<GCNSubtarget>(F));
  return static_cast<const AMDGPUSubtarget &>(
      TM.getSubtarget<R600Subtarget>(F));
}

SmallVector<unsigned>
AMDGPUSubtarget::getMaxNumWorkGroups(const Function &F) const {
  return AMDGPU::getIntegerVecAttribute(F, "amdgpu-max-num-workgroups", 3,
                                        std::numeric_limits<uint32_t>::max());
}
