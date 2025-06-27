; REQUIRES: amdgpu-registered-target

; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: define void @valid_amdgpu_waves_per_eu_range() #0
define void @valid_amdgpu_waves_per_eu_range() "amdgpu-waves-per-eu"="2,4" {
  ret void
}

; CHECK: define void @valid_amdgpu_waves_per_eu_min_only() #1
define void @valid_amdgpu_waves_per_eu_min_only() "amdgpu-waves-per-eu"="2" {
  ret void
}

; CHECK: define void @valid_amdgpu_waves_per_eu_max_only() #2
define void @valid_amdgpu_waves_per_eu_max_only() "amdgpu-waves-per-eu"="0,4" {
  ret void
}

; CHECK: attributes #0 = { "amdgpu-waves-per-eu"="2,4" }
; CHECK: attributes #1 = { "amdgpu-waves-per-eu"="2" }
; CHECK: attributes #2 = { "amdgpu-waves-per-eu"="0,4" }
