//===---------- llvm/unittests/Target/AMDGPU/AMDGPUUnitTests.h ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TARGET_AMDGPU_AMDGPUUNITTESTS_H
#define LLVM_UNITTESTS_TARGET_AMDGPU_AMDGPUUNITTESTS_H

#include <memory>
#include <string>

namespace llvm {

class GCNTargetMachine;
class StringRef;

void initializeAMDGPUTarget();

void initializeAMDGPUTargetOnce(std::once_flag &Flag);

std::unique_ptr<GCNTargetMachine>
createAMDGPUTargetMachine(std::string TStr, StringRef CPU, StringRef FS);

std::unique_ptr<Module> parseMIR(LLVMContext &Context, const TargetMachine &TM,
                                 StringRef MIRCode, MachineModuleInfo &MMI);
} // end namespace llvm

#endif // LLVM_UNITTESTS_TARGET_AMDGPU_AMDGPUUNITTESTS_H
