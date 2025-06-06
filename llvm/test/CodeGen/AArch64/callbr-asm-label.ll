; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s

@X = common local_unnamed_addr global i32 0, align 4

define i32 @test1() {
; CHECK-LABEL: test1:
; CHECK:         .word b
; CHECK-NEXT:    .word .LBB0_2
; CHECK: // %bb.1:
; CHECK: .LBB0_2: // Inline asm indirect target
entry:
  callbr void asm sideeffect "1:\0A\09.word b, ${0:l}\0A\09", "!i"()
          to label %cleanup [label %indirect]

indirect:
  br label %cleanup

cleanup:
  %retval.0 = phi i32 [ 1, %indirect ], [ 0, %entry ]
  ret i32 %retval.0
}

define void @test2() {
; CHECK-LABEL: test2:
entry:
  %0 = load i32, ptr @X, align 4
  %and = and i32 %0, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.end10, label %if.then

if.then:
; CHECK:       .word b
; CHECK-NEXT:  .word .LBB1_3
; CHECK:       .LBB1_3: // Inline asm indirect target
  callbr void asm sideeffect "1:\0A\09.word b, ${0:l}\0A\09", "!i"()
          to label %if.then4 [label %if.end6]

if.then4:
  %call5 = tail call i32 @g()
  br label %if.end6

if.end6:
  %.pre = load i32, ptr @X, align 4
  %.pre13 = and i32 %.pre, 1
  %phitmp = icmp eq i32 %.pre13, 0
  br i1 %phitmp, label %if.end10, label %if.then9

if.then9:
; CHECK: .LBB1_5: // Inline asm indirect target
  callbr void asm sideeffect "", "!i"()
          to label %if.end10 [label %l_yes]

if.end10:
  br label %l_yes

l_yes:
  ret void
}

declare i32 @g(...)
