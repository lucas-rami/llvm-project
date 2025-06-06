; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -p indvars -S %s | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"

declare void @foo(i32)

define void @add_nsw_zext_fold_results_in_sext(i64 %len) {
; CHECK-LABEL: define void @add_nsw_zext_fold_results_in_sext(
; CHECK-SAME: i64 [[LEN:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[LEN_TRUNC:%.*]] = trunc i64 [[LEN]] to i32
; CHECK-NEXT:    [[LZ:%.*]] = tail call range(i32 0, 33) i32 @llvm.ctlz.i32(i32 [[LEN_TRUNC]], i1 false)
; CHECK-NEXT:    [[SUB_I:%.*]] = lshr i32 [[LZ]], 3
; CHECK-NEXT:    [[ADD_I:%.*]] = sub i32 5, [[SUB_I]]
; CHECK-NEXT:    [[PRECOND:%.*]] = icmp eq i32 [[SUB_I]], 5
; CHECK-NEXT:    br i1 [[PRECOND]], label %[[EXIT:.*]], label %[[LOOP_PREHEADER:.*]]
; CHECK:       [[LOOP_PREHEADER]]:
; CHECK-NEXT:    [[TMP1:%.*]] = zext nneg i32 [[ADD_I]] to i64
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[TMP1]], %[[LOOP_PREHEADER]] ], [ [[INDVARS_IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[IV:%.*]] = trunc nuw i64 [[INDVARS_IV]] to i32
; CHECK-NEXT:    [[IV_NEXT:%.*]] = add i32 [[IV]], 1
; CHECK-NEXT:    [[SH_PROM:%.*]] = zext nneg i32 [[IV_NEXT]] to i64
; CHECK-NEXT:    [[SHR:%.*]] = lshr i64 1, [[SH_PROM]]
; CHECK-NEXT:    [[TMP0:%.*]] = trunc nuw nsw i64 [[SHR]] to i32
; CHECK-NEXT:    call void @foo(i32 [[TMP0]])
; CHECK-NEXT:    [[EC:%.*]] = icmp eq i32 [[IV_NEXT]], 0
; CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    br i1 [[EC]], label %[[EXIT_LOOPEXIT:.*]], label %[[LOOP]]
; CHECK:       [[EXIT_LOOPEXIT]]:
; CHECK-NEXT:    br label %[[EXIT]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  %len.trunc  = trunc i64 %len to i32
  %lz = tail call range(i32 0, 33) i32 @llvm.ctlz.i32(i32 %len.trunc, i1 false)
  %sub.i = lshr i32 %lz, 3
  %add.i = sub nuw nsw i32 5, %sub.i
  %precond = icmp eq i32 %sub.i, 5
  br i1 %precond, label %exit, label %loop

loop:
  %iv = phi i32 [ %add.i, %entry ], [ %iv.next, %loop ]
  %iv.next = add i32 %iv, 1
  %sh_prom = zext nneg i32 %iv.next to i64
  %shr = lshr i64 1, %sh_prom
  %2 = trunc nuw nsw i64 %shr to i32
  call void @foo(i32 %2)
  %ec = icmp eq i32 %iv.next, 0
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @add_nsw_zext_fold_results_in_sext_known_positive(i32 %mask, ptr %src, i1 %c) {
; CHECK-LABEL: define void @add_nsw_zext_fold_results_in_sext_known_positive(
; CHECK-SAME: i32 [[MASK:%.*]], ptr [[SRC:%.*]], i1 [[C:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[SPEC_SELECT:%.*]] = select i1 [[C]], i32 0, i32 6
; CHECK-NEXT:    [[ADD:%.*]] = add i32 [[SPEC_SELECT]], [[MASK]]
; CHECK-NEXT:    [[PRECOND:%.*]] = icmp slt i32 [[ADD]], 0
; CHECK-NEXT:    br i1 [[PRECOND]], label %[[EXIT:.*]], label %[[PH:.*]]
; CHECK:       [[PH]]:
; CHECK-NEXT:    [[TMP0:%.*]] = sub i32 78, [[SPEC_SELECT]]
; CHECK-NEXT:    [[TMP1:%.*]] = zext nneg i32 [[TMP0]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr i32, ptr [[SRC]], i64 [[TMP2]]
; CHECK-NEXT:    [[L:%.*]] = load i32, ptr [[GEP]], align 1
; CHECK-NEXT:    call void @foo(i32 [[L]])
; CHECK-NEXT:    br label %[[LOOP]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  %spec.select = select i1 %c, i32 0, i32 6
  %add = add i32 %spec.select, %mask
  %precond = icmp slt i32 %add, 0
  br i1 %precond, label %exit, label %ph

ph:
  %start = sub i32 79, %spec.select
  br label %loop

loop:                                     ; preds = %loop, %ph
  %iv = phi i32 [ %start, %ph ], [ %dec, %loop ]
  %iv.ext = zext i32 %iv to i64
  %gep = getelementptr i32, ptr %src, i64 %iv.ext
  %l = load i32, ptr %gep, align 1
  call void @foo(i32 %l)
  %dec = add i32 %iv, 0
  br label %loop

exit:
  ret void
}
