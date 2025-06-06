; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=vector-combine -S -mtriple=x86_64-- -mattr=SSE2 | FileCheck %s --check-prefixes=CHECK,SSE
; RUN: opt < %s -passes=vector-combine -S -mtriple=x86_64-- -mattr=AVX2 | FileCheck %s --check-prefixes=CHECK,AVX

define <2 x i64> @add_constant(i64 %x) {
; CHECK-LABEL: @add_constant(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = add i64 [[X:%.*]], 42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = add <2 x i64> %ins, <i64 42, i64 undef>
  ret <2 x i64> %bo
}

define <2 x i64> @add_constant_not_undef_lane(i64 %x) {
; CHECK-LABEL: @add_constant_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = add i64 [[X:%.*]], 42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = add <2 x i64> %ins, <i64 42, i64 -42>
  ret <2 x i64> %bo
}

define <2 x i64> @add_constant_load(ptr %p) {
; CHECK-LABEL: @add_constant_load(
; CHECK-NEXT:    [[LD:%.*]] = load i64, ptr [[P:%.*]], align 8
; CHECK-NEXT:    [[INS:%.*]] = insertelement <2 x i64> poison, i64 [[LD]], i32 0
; CHECK-NEXT:    [[BO:%.*]] = add <2 x i64> [[INS]], <i64 42, i64 -42>
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ld = load i64, ptr %p
  %ins = insertelement <2 x i64> poison, i64 %ld, i32 0
  %bo = add <2 x i64> %ins, <i64 42, i64 -42>
  ret <2 x i64> %bo
}

; IR flags are not required, but they should propagate.

define <4 x i32> @sub_constant_op0(i32 %x) {
; CHECK-LABEL: @sub_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = sub nuw nsw i32 -42, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <4 x i32> poison, i32 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <4 x i32> [[BO]]
;
  %ins = insertelement <4 x i32> poison, i32 %x, i32 1
  %bo = sub nsw nuw <4 x i32> <i32 undef, i32 -42, i32 undef, i32 undef>, %ins
  ret <4 x i32> %bo
}

define <4 x i32> @sub_constant_op0_not_undef_lane(i32 %x) {
; CHECK-LABEL: @sub_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = sub nuw i32 42, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <4 x i32> poison, i32 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <4 x i32> [[BO]]
;
  %ins = insertelement <4 x i32> poison, i32 %x, i32 1
  %bo = sub nuw <4 x i32> <i32 1, i32 42, i32 42, i32 -42>, %ins
  ret <4 x i32> %bo
}

define <8 x i16> @sub_constant_op1(i16 %x) {
; CHECK-LABEL: @sub_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = sub nuw i16 [[X:%.*]], 42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <8 x i16> poison, i16 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <8 x i16> [[BO]]
;
  %ins = insertelement <8 x i16> poison, i16 %x, i32 0
  %bo = sub nuw <8 x i16> %ins, <i16 42, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef>
  ret <8 x i16> %bo
}

define <8 x i16> @sub_constant_op1_not_undef_lane(i16 %x) {
; CHECK-LABEL: @sub_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = sub nuw i16 [[X:%.*]], 42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <8 x i16> poison, i16 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <8 x i16> [[BO]]
;
  %ins = insertelement <8 x i16> poison, i16 %x, i32 0
  %bo = sub nuw <8 x i16> %ins, <i16 42, i16 -42, i16 0, i16 1, i16 -2, i16 3, i16 -4, i16 5>
  ret <8 x i16> %bo
}

define <16 x i8> @mul_constant(i8 %x) {
; CHECK-LABEL: @mul_constant(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = mul i8 [[X:%.*]], -42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <16 x i8> poison, i8 [[BO_SCALAR]], i64 2
; CHECK-NEXT:    ret <16 x i8> [[BO]]
;
  %ins = insertelement <16 x i8> poison, i8 %x, i32 2
  %bo = mul <16 x i8> %ins, <i8 undef, i8 undef, i8 -42, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef>
  ret <16 x i8> %bo
}

define <3 x i64> @mul_constant_not_undef_lane(i64 %x) {
; CHECK-LABEL: @mul_constant_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = mul i64 [[X:%.*]], -42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <3 x i64> poison, i64 [[BO_SCALAR]], i64 2
; CHECK-NEXT:    ret <3 x i64> [[BO]]
;
  %ins = insertelement <3 x i64> poison, i64 %x, i32 2
  %bo = mul <3 x i64> %ins, <i64 42, i64 undef, i64 -42>
  ret <3 x i64> %bo
}

define <16 x i8> @mul_constant_multiuse(i8 %a0, <16 x i8> %a1) {
; SSE-LABEL: @mul_constant_multiuse(
; SSE-NEXT:    [[INS:%.*]] = insertelement <16 x i8> <i8 undef, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, i8 [[A0:%.*]], i32 0
; SSE-NEXT:    [[MUL:%.*]] = mul <16 x i8> [[INS]], <i8 3, i8 7, i8 9, i8 11, i8 13, i8 15, i8 17, i8 19, i8 21, i8 23, i8 25, i8 27, i8 29, i8 31, i8 33, i8 35>
; SSE-NEXT:    [[AND:%.*]] = and <16 x i8> [[INS]], [[A1:%.*]]
; SSE-NEXT:    [[XOR:%.*]] = xor <16 x i8> [[AND]], [[MUL]]
; SSE-NEXT:    ret <16 x i8> [[XOR]]
;
; AVX-LABEL: @mul_constant_multiuse(
; AVX-NEXT:    [[INS:%.*]] = insertelement <16 x i8> <i8 undef, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, i8 [[A0:%.*]], i32 0
; AVX-NEXT:    [[MUL_SCALAR:%.*]] = mul i8 [[A0]], 3
; AVX-NEXT:    [[MUL:%.*]] = insertelement <16 x i8> <i8 undef, i8 7, i8 18, i8 33, i8 52, i8 75, i8 102, i8 -123, i8 -88, i8 -49, i8 -6, i8 41, i8 92, i8 -109, i8 -50, i8 13>, i8 [[MUL_SCALAR]], i64 0
; AVX-NEXT:    [[AND:%.*]] = and <16 x i8> [[INS]], [[A1:%.*]]
; AVX-NEXT:    [[XOR:%.*]] = xor <16 x i8> [[AND]], [[MUL]]
; AVX-NEXT:    ret <16 x i8> [[XOR]]
;
  %ins = insertelement <16 x i8> <i8 undef, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, i8 %a0, i32 0
  %mul = mul <16 x i8> %ins, <i8 3, i8 7, i8 9, i8 11, i8 13, i8 15, i8 17, i8 19, i8 21, i8 23, i8 25, i8 27, i8 29, i8 31, i8 33, i8 35>
  %and = and <16 x i8> %ins, %a1
  %xor = xor <16 x i8> %and, %mul
  ret <16 x i8> %xor
}

define <2 x i64> @shl_constant_op0(i64 %x) {
; CHECK-LABEL: @shl_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = shl i64 2, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = shl <2 x i64> <i64 undef, i64 2>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @shl_constant_op0_not_undef_lane(i64 %x) {
; CHECK-LABEL: @shl_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = shl i64 2, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = shl <2 x i64> <i64 5, i64 2>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @shl_constant_op0_load(ptr %p) {
; CHECK-LABEL: @shl_constant_op0_load(
; CHECK-NEXT:    [[LD:%.*]] = load i64, ptr [[P:%.*]], align 8
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = shl i64 2, [[LD]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ld = load i64, ptr %p
  %ins = insertelement <2 x i64> poison, i64 %ld, i32 1
  %bo = shl <2 x i64> <i64 undef, i64 2>, %ins
  ret <2 x i64> %bo
}

define <4 x i32> @shl_constant_op0_multiuse(i32 %a0, <4 x i32> %a1) {
; CHECK-LABEL: @shl_constant_op0_multiuse(
; CHECK-NEXT:    [[INS:%.*]] = insertelement <4 x i32> <i32 undef, i32 1, i32 2, i32 3>, i32 [[A0:%.*]], i32 0
; CHECK-NEXT:    [[MUL_SCALAR:%.*]] = shl i32 [[A0]], 3
; CHECK-NEXT:    [[MUL:%.*]] = insertelement <4 x i32> <i32 0, i32 16, i32 64, i32 192>, i32 [[MUL_SCALAR]], i64 0
; CHECK-NEXT:    [[AND:%.*]] = and <4 x i32> [[INS]], [[A1:%.*]]
; CHECK-NEXT:    [[XOR:%.*]] = xor <4 x i32> [[AND]], [[MUL]]
; CHECK-NEXT:    ret <4 x i32> [[XOR]]
;
  %ins = insertelement <4 x i32> <i32 undef, i32 1, i32 2, i32 3>, i32 %a0, i32 0
  %mul = shl <4 x i32> %ins, <i32 3, i32 4, i32 5, i32 6>
  %and = and <4 x i32> %ins, %a1
  %xor = xor <4 x i32> %and, %mul
  ret <4 x i32> %xor
}

define <2 x i64> @shl_constant_op1(i64 %x) {
; CHECK-LABEL: @shl_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = shl nuw i64 [[X:%.*]], 5
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = shl nuw <2 x i64> %ins, <i64 5, i64 undef>
  ret <2 x i64> %bo
}

define <2 x i64> @shl_constant_op1_not_undef_lane(i64 %x) {
; CHECK-LABEL: @shl_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = shl nuw i64 [[X:%.*]], 5
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = shl nuw <2 x i64> %ins, <i64 5, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @shl_constant_op1_load(ptr %p) {
; CHECK-LABEL: @shl_constant_op1_load(
; CHECK-NEXT:    [[LD:%.*]] = load i64, ptr [[P:%.*]], align 8
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = shl nuw i64 [[LD]], 5
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ld = load i64, ptr %p
  %ins = insertelement <2 x i64> poison, i64 %ld, i32 0
  %bo = shl nuw <2 x i64> %ins, <i64 5, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @ashr_constant_op0(i64 %x) {
; CHECK-LABEL: @ashr_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = ashr exact i64 2, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = ashr exact <2 x i64> <i64 undef, i64 2>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @ashr_constant_op0_not_undef_lane(i64 %x) {
; CHECK-LABEL: @ashr_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = ashr exact i64 2, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = ashr exact <2 x i64> <i64 5, i64 2>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @ashr_constant_op1(i64 %x) {
; CHECK-LABEL: @ashr_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = ashr i64 [[X:%.*]], 5
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = ashr <2 x i64> %ins, <i64 5, i64 undef>
  ret <2 x i64> %bo
}

define <2 x i64> @ashr_constant_op1_not_undef_lane(i64 %x) {
; CHECK-LABEL: @ashr_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = ashr i64 [[X:%.*]], 5
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = ashr <2 x i64> %ins, <i64 5, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @lshr_constant_op0(i64 %x) {
; CHECK-LABEL: @lshr_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = lshr i64 5, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = lshr <2 x i64> <i64 5, i64 undef>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @lshr_constant_op0_not_undef_lane(i64 %x) {
; CHECK-LABEL: @lshr_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = lshr i64 5, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = lshr <2 x i64> <i64 5, i64 2>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @lshr_constant_op1(i64 %x) {
; CHECK-LABEL: @lshr_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = lshr exact i64 [[X:%.*]], 2
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = lshr exact <2 x i64> %ins, <i64 undef, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @lshr_constant_op1_not_undef_lane(i64 %x) {
; CHECK-LABEL: @lshr_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = lshr exact i64 [[X:%.*]], 2
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = lshr exact <2 x i64> %ins, <i64 5, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @urem_constant_op0(i64 %x) {
; CHECK-LABEL: @urem_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = urem i64 5, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> zeroinitializer, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> splat (i64 1), i64 %x, i32 0
  %bo = urem <2 x i64> <i64 5, i64 undef>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @urem_constant_op0_not_undef_lane(i64 %x) {
; CHECK-LABEL: @urem_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = urem i64 5, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> zeroinitializer, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> splat (i64 1), i64 %x, i32 0
  %bo = urem <2 x i64> <i64 5, i64 2>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @urem_constant_op1(i64 %x) {
; CHECK-LABEL: @urem_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = urem i64 [[X:%.*]], 2
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = urem <2 x i64> %ins, <i64 2, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @urem_constant_op1_not_undef_lane(i64 %x) {
; CHECK-LABEL: @urem_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = urem i64 [[X:%.*]], 2
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = urem <2 x i64> %ins, <i64 5, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @srem_constant_op0(i64 %x) {
; CHECK-LABEL: @srem_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = srem i64 5, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = srem <2 x i64> <i64 5, i64 undef>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @srem_constant_op0_not_undef_lane(i64 %x) {
; CHECK-LABEL: @srem_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = srem i64 5, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> zeroinitializer, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> splat (i64 1), i64 %x, i32 0
  %bo = srem <2 x i64> <i64 5, i64 2>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @srem_constant_op1(i64 %x) {
; CHECK-LABEL: @srem_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = srem i64 [[X:%.*]], 2
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = srem <2 x i64> %ins, <i64 2, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @srem_constant_op1_not_undef_lane(i64 %x) {
; CHECK-LABEL: @srem_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = srem i64 [[X:%.*]], 2
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = srem <2 x i64> %ins, <i64 5, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @udiv_constant_op0(i64 %x) {
; CHECK-LABEL: @udiv_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = udiv exact i64 5, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> <i64 5, i64 undef>, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> splat (i64 1), i64 %x, i32 0
  %bo = udiv exact <2 x i64> <i64 5, i64 undef>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @udiv_constant_op0_not_undef_lane(i64 %x) {
; CHECK-LABEL: @udiv_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = udiv exact i64 5, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> <i64 5, i64 2>, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> splat (i64 1), i64 %x, i32 0
  %bo = udiv exact <2 x i64> <i64 5, i64 2>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @udiv_constant_op1(i64 %x) {
; CHECK-LABEL: @udiv_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = udiv i64 [[X:%.*]], 2
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = udiv <2 x i64> %ins, <i64 2, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @udiv_constant_op1_not_undef_lane(i64 %x) {
; CHECK-LABEL: @udiv_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = udiv i64 [[X:%.*]], 2
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = udiv <2 x i64> %ins, <i64 5, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @sdiv_constant_op0(i64 %x) {
; CHECK-LABEL: @sdiv_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = sdiv i64 5, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> <i64 5, i64 undef>, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> splat (i64 1), i64 %x, i32 0
  %bo = sdiv <2 x i64> <i64 5, i64 undef>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @sdiv_constant_op0_not_undef_lane(i64 %x) {
; CHECK-LABEL: @sdiv_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = sdiv i64 5, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> <i64 5, i64 2>, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> splat (i64 1), i64 %x, i32 0
  %bo = sdiv <2 x i64> <i64 5, i64 2>, %ins
  ret <2 x i64> %bo
}

define <2 x i64> @sdiv_constant_op1(i64 %x) {
; CHECK-LABEL: @sdiv_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = sdiv exact i64 [[X:%.*]], 2
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = sdiv exact <2 x i64> %ins, <i64 2, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @sdiv_constant_op1_not_undef_lane(i64 %x) {
; CHECK-LABEL: @sdiv_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = sdiv exact i64 [[X:%.*]], 2
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = sdiv exact <2 x i64> %ins, <i64 5, i64 2>
  ret <2 x i64> %bo
}

define <2 x i64> @and_constant(i64 %x) {
; CHECK-LABEL: @and_constant(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = and i64 [[X:%.*]], 42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = and <2 x i64> %ins, <i64 42, i64 undef>
  ret <2 x i64> %bo
}

define <2 x i64> @and_constant_not_undef_lane(i64 %x) {
; CHECK-LABEL: @and_constant_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = and i64 [[X:%.*]], 42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = and <2 x i64> %ins, <i64 42, i64 -42>
  ret <2 x i64> %bo
}

define <2 x i64> @or_constant(i64 %x) {
; CHECK-LABEL: @or_constant(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = or i64 [[X:%.*]], -42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = or <2 x i64> %ins, <i64 undef, i64 -42>
  ret <2 x i64> %bo
}

define <2 x i64> @or_constant_not_undef_lane(i64 %x) {
; CHECK-LABEL: @or_constant_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = or i64 [[X:%.*]], -42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 1
  %bo = or <2 x i64> %ins, <i64 42, i64 -42>
  ret <2 x i64> %bo
}

define <2 x i64> @xor_constant(i64 %x) {
; CHECK-LABEL: @xor_constant(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = xor i64 [[X:%.*]], 42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = xor <2 x i64> %ins, <i64 42, i64 undef>
  ret <2 x i64> %bo
}

define <2 x i64> @xor_constant_not_undef_lane(i64 %x) {
; CHECK-LABEL: @xor_constant_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = xor i64 [[X:%.*]], 42
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x i64> poison, i64 [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x i64> [[BO]]
;
  %ins = insertelement <2 x i64> poison, i64 %x, i32 0
  %bo = xor <2 x i64> %ins, <i64 42, i64 -42>
  ret <2 x i64> %bo
}

define <2 x double> @fadd_constant(double %x) {
; CHECK-LABEL: @fadd_constant(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fadd double [[X:%.*]], 4.200000e+01
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 0
  %bo = fadd <2 x double> %ins, <double 42.0, double undef>
  ret <2 x double> %bo
}

define <2 x double> @fadd_constant_not_undef_lane(double %x) {
; CHECK-LABEL: @fadd_constant_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fadd double [[X:%.*]], -4.200000e+01
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 1
  %bo = fadd <2 x double> %ins, <double 42.0, double -42.0>
  ret <2 x double> %bo
}

define <2 x double> @fsub_constant_op0(double %x) {
; CHECK-LABEL: @fsub_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fsub fast double 4.200000e+01, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 0
  %bo = fsub fast <2 x double> <double 42.0, double undef>, %ins
  ret <2 x double> %bo
}

define <2 x double> @fsub_constant_op0_not_undef_lane(double %x) {
; CHECK-LABEL: @fsub_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fsub nsz double -4.200000e+01, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 1
  %bo = fsub nsz <2 x double> <double 42.0, double -42.0>, %ins
  ret <2 x double> %bo
}

define <2 x double> @fsub_constant_op1(double %x) {
; CHECK-LABEL: @fsub_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fsub double [[X:%.*]], 4.200000e+01
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 1
  %bo = fsub <2 x double> %ins, <double undef, double 42.0>
  ret <2 x double> %bo
}

define <2 x double> @fsub_constant_op1_not_undef_lane(double %x) {
; CHECK-LABEL: @fsub_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fsub double [[X:%.*]], 4.200000e+01
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 0
  %bo = fsub <2 x double> %ins, <double 42.0, double -42.0>
  ret <2 x double> %bo
}

define <2 x double> @fmul_constant(double %x) {
; CHECK-LABEL: @fmul_constant(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fmul reassoc double [[X:%.*]], 4.200000e+01
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 0
  %bo = fmul reassoc <2 x double> %ins, <double 42.0, double undef>
  ret <2 x double> %bo
}

define <2 x double> @fmul_constant_not_undef_lane(double %x) {
; CHECK-LABEL: @fmul_constant_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fmul double [[X:%.*]], -4.200000e+01
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 1
  %bo = fmul <2 x double> %ins, <double 42.0, double -42.0>
  ret <2 x double> %bo
}

define <2 x double> @fdiv_constant_op0(double %x) {
; CHECK-LABEL: @fdiv_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fdiv nnan double 4.200000e+01, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 1
  %bo = fdiv nnan <2 x double> <double undef, double 42.0>, %ins
  ret <2 x double> %bo
}

define <2 x double> @fdiv_constant_op0_not_undef_lane(double %x) {
; CHECK-LABEL: @fdiv_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fdiv ninf double 4.200000e+01, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 0
  %bo = fdiv ninf <2 x double> <double 42.0, double -42.0>, %ins
  ret <2 x double> %bo
}

define <2 x double> @fdiv_constant_op1(double %x) {
; CHECK-LABEL: @fdiv_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fdiv double [[X:%.*]], 4.200000e+01
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 0
  %bo = fdiv <2 x double> %ins, <double 42.0, double undef>
  ret <2 x double> %bo
}

define <2 x double> @fdiv_constant_op1_not_undef_lane(double %x) {
; CHECK-LABEL: @fdiv_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = fdiv double [[X:%.*]], 4.200000e+01
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 0
  %bo = fdiv <2 x double> %ins, <double 42.0, double -42.0>
  ret <2 x double> %bo
}

define <2 x double> @frem_constant_op0(double %x) {
; CHECK-LABEL: @frem_constant_op0(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = frem fast double 4.200000e+01, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 0
  %bo = frem fast <2 x double> <double 42.0, double undef>, %ins
  ret <2 x double> %bo
}

define <2 x double> @frem_constant_op0_not_undef_lane(double %x) {
; CHECK-LABEL: @frem_constant_op0_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = frem double -4.200000e+01, [[X:%.*]]
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 1
  %bo = frem <2 x double> <double 42.0, double -42.0>, %ins
  ret <2 x double> %bo
}

define <2 x double> @frem_constant_op1(double %x) {
; CHECK-LABEL: @frem_constant_op1(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = frem ninf double [[X:%.*]], 4.200000e+01
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 1
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 1
  %bo = frem ninf <2 x double> %ins, <double undef, double 42.0>
  ret <2 x double> %bo
}

define <2 x double> @frem_constant_op1_not_undef_lane(double %x) {
; CHECK-LABEL: @frem_constant_op1_not_undef_lane(
; CHECK-NEXT:    [[BO_SCALAR:%.*]] = frem nnan double [[X:%.*]], 4.200000e+01
; CHECK-NEXT:    [[BO:%.*]] = insertelement <2 x double> poison, double [[BO_SCALAR]], i64 0
; CHECK-NEXT:    ret <2 x double> [[BO]]
;
  %ins = insertelement <2 x double> poison, double %x, i32 0
  %bo = frem nnan <2 x double> %ins, <double 42.0, double -42.0>
  ret <2 x double> %bo
}
