; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc --mtriple=loongarch32 -mattr=+32s,+d \
; RUN:   --min-jump-table-entries=4 < %s \
; RUN:   --loongarch-annotate-tablejump \
; RUN:   | FileCheck %s --check-prefix=LA32-JT
; RUN: llc --mtriple=loongarch64 -mattr=+d \
; RUN:   --min-jump-table-entries=4 < %s \
; RUN:   --loongarch-annotate-tablejump \
; RUN:   | FileCheck %s --check-prefix=LA64-JT

define void @switch_4_arms(i32 %in, ptr %out) nounwind {
; LA32-JT-LABEL: switch_4_arms:
; LA32-JT:       # %bb.0: # %entry
; LA32-JT-NEXT:    addi.w $a3, $a0, -1
; LA32-JT-NEXT:    ori $a2, $zero, 3
; LA32-JT-NEXT:    bltu $a2, $a3, .LBB0_7
; LA32-JT-NEXT:  # %bb.1: # %entry
; LA32-JT-NEXT:    pcalau12i $a4, %pc_hi20(.LJTI0_0)
; LA32-JT-NEXT:    addi.w $a4, $a4, %pc_lo12(.LJTI0_0)
; LA32-JT-NEXT:    alsl.w $a3, $a3, $a4, 2
; LA32-JT-NEXT:    ld.w $a3, $a3, 0
; LA32-JT-NEXT:  .Ljrtb_0:
; LA32-JT-NEXT:    jr $a3
; LA32-JT-NEXT:  .LBB0_2: # %bb1
; LA32-JT-NEXT:    ori $a3, $zero, 4
; LA32-JT-NEXT:    b .LBB0_6
; LA32-JT-NEXT:  .LBB0_3: # %bb2
; LA32-JT-NEXT:    ori $a3, $zero, 3
; LA32-JT-NEXT:    b .LBB0_6
; LA32-JT-NEXT:  .LBB0_4: # %bb3
; LA32-JT-NEXT:    ori $a3, $zero, 2
; LA32-JT-NEXT:    b .LBB0_6
; LA32-JT-NEXT:  .LBB0_5: # %bb4
; LA32-JT-NEXT:    ori $a3, $zero, 1
; LA32-JT-NEXT:  .LBB0_6: # %exit
; LA32-JT-NEXT:    st.w $a3, $a1, 0
; LA32-JT-NEXT:  .LBB0_7: # %exit
; LA32-JT-NEXT:    addi.w $a3, $a0, -5
; LA32-JT-NEXT:    bltu $a2, $a3, .LBB0_9
; LA32-JT-NEXT:  # %bb.8: # %exit
; LA32-JT-NEXT:    pcalau12i $a4, %pc_hi20(.LJTI0_1)
; LA32-JT-NEXT:    addi.w $a4, $a4, %pc_lo12(.LJTI0_1)
; LA32-JT-NEXT:    alsl.w $a3, $a3, $a4, 2
; LA32-JT-NEXT:    ld.w $a3, $a3, 0
; LA32-JT-NEXT:  .Ljrtb_1:
; LA32-JT-NEXT:    jr $a3
; LA32-JT-NEXT:  .LBB0_9: # %exit2
; LA32-JT-NEXT:    ret
;
; LA64-JT-LABEL: switch_4_arms:
; LA64-JT:       # %bb.0: # %entry
; LA64-JT-NEXT:    addi.w $a0, $a0, 0
; LA64-JT-NEXT:    addi.d $a3, $a0, -1
; LA64-JT-NEXT:    ori $a2, $zero, 3
; LA64-JT-NEXT:    bltu $a2, $a3, .LBB0_7
; LA64-JT-NEXT:  # %bb.1: # %entry
; LA64-JT-NEXT:    slli.d $a3, $a3, 3
; LA64-JT-NEXT:    pcalau12i $a4, %pc_hi20(.LJTI0_0)
; LA64-JT-NEXT:    addi.d $a4, $a4, %pc_lo12(.LJTI0_0)
; LA64-JT-NEXT:    ldx.d $a3, $a4, $a3
; LA64-JT-NEXT:  .Ljrtb_0:
; LA64-JT-NEXT:    jr $a3
; LA64-JT-NEXT:  .LBB0_2: # %bb1
; LA64-JT-NEXT:    ori $a3, $zero, 4
; LA64-JT-NEXT:    b .LBB0_6
; LA64-JT-NEXT:  .LBB0_3: # %bb2
; LA64-JT-NEXT:    ori $a3, $zero, 3
; LA64-JT-NEXT:    b .LBB0_6
; LA64-JT-NEXT:  .LBB0_4: # %bb3
; LA64-JT-NEXT:    ori $a3, $zero, 2
; LA64-JT-NEXT:    b .LBB0_6
; LA64-JT-NEXT:  .LBB0_5: # %bb4
; LA64-JT-NEXT:    ori $a3, $zero, 1
; LA64-JT-NEXT:  .LBB0_6: # %exit
; LA64-JT-NEXT:    st.w $a3, $a1, 0
; LA64-JT-NEXT:  .LBB0_7: # %exit
; LA64-JT-NEXT:    addi.d $a3, $a0, -5
; LA64-JT-NEXT:    bltu $a2, $a3, .LBB0_9
; LA64-JT-NEXT:  # %bb.8: # %exit
; LA64-JT-NEXT:    slli.d $a3, $a3, 3
; LA64-JT-NEXT:    pcalau12i $a4, %pc_hi20(.LJTI0_1)
; LA64-JT-NEXT:    addi.d $a4, $a4, %pc_lo12(.LJTI0_1)
; LA64-JT-NEXT:    ldx.d $a3, $a4, $a3
; LA64-JT-NEXT:  .Ljrtb_1:
; LA64-JT-NEXT:    jr $a3
; LA64-JT-NEXT:  .LBB0_9: # %exit2
; LA64-JT-NEXT:    ret
entry:
  switch i32 %in, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
  ]
bb1:
  store i32 4, ptr %out
  br label %exit
bb2:
  store i32 3, ptr %out
  br label %exit
bb3:
  store i32 2, ptr %out
  br label %exit
bb4:
  store i32 1, ptr %out
  br label %exit
exit:
  switch i32 %in, label %exit2 [
    i32 5, label %bb1
    i32 6, label %bb2
    i32 7, label %bb3
    i32 8, label %bb4
  ]
exit2:
  ret void
}

; UTC_ARGS: --disable

; LA32-JT-LABEL: .LJTI0_0:
; LA32-JT:       .section .discard.tablejump_annotate,"",@progbits
; LA32-JT-NEXT:  .word .Ljrtb_0
; LA32-JT-NEXT:  .word .LJTI0_0
; LA32-JT-NEXT:  .word .Ljrtb_1
; LA32-JT-NEXT:  .word .LJTI0_1

; UTC_ARGS: --disable
; LA64-JT-LABEL: .LJTI0_0:
; LA64-JT:       .section .discard.tablejump_annotate,"",@progbits
; LA64-JT-NEXT:  .dword .Ljrtb_0
; LA64-JT-NEXT:  .dword .LJTI0_0
; LA64-JT-NEXT:  .dword .Ljrtb_1
; LA64-JT-NEXT:  .dword .LJTI0_1
