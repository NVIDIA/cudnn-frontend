-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
import Std

/-!
One reusable slot, one aggregate producer and one aggregate reader. Counters
are logical generations, NOT hardware mbarrier parity. A reader represents
ALL accesses that must finish before reuse (including asynchronous accesses).
This is a protocol specification, not a semantics of Python, PTX or CUDA.
The adjacent source checks do not constitute a refinement proof to this model.
-/
namespace FrostLifecycle

structure State where
  issued : Nat := 0
  completed : Nat := 0
  acquired : Nat := 0
  retired : Nat := 0
  released : Nat := 0
  deriving DecidableEq, Repr

/-- Outstanding work has capacity one per physical slot. -/
def Safe (s : State) : Prop :=
  s.released ≤ s.retired ∧ s.retired ≤ s.acquired ∧
  s.acquired ≤ s.completed ∧ s.completed ≤ s.issued ∧
  s.issued ≤ s.released + 1

inductive Step : State → State → Prop where
  | issue (s) (credit : s.issued = s.released) :
      Step s { s with issued := s.issued + 1 }
  | complete (s) (pending : s.completed < s.issued) :
      Step s { s with completed := s.completed + 1 }
  | acquire (s) (published : s.acquired < s.completed) :
      Step s { s with acquired := s.acquired + 1 }
  | retire (s) (pending : s.retired < s.acquired) :
      Step s { s with retired := s.retired + 1 }
  | release (s) (done : s.released < s.retired) :
      Step s { s with released := s.released + 1 }

/-- Any interleaving of legal transitions preserves the resource invariant. -/
theorem step_preserves {s t : State} (h : Safe s) (step : Step s t) : Safe t := by
  cases step <;> simp_all [Safe] <;> omega

inductive Reachable : State → Prop where
  | initial : Reachable {}
  | next {s t} : Reachable s → Step s t → Reachable t

/-- Unbounded induction: this statement is not limited to tested loop counts. -/
theorem reachable_safe {s : State} (h : Reachable s) : Safe s := by
  induction h with
  | initial => simp [Safe]
  | next _ step ih => exact step_preserves ih step

/-- Starting another write requires every previous read to have retired. -/
theorem reuse_only_after_retirement {s : State} (h : Reachable s)
    (credit : s.issued = s.released) :
    s.issued = s.retired ∧ s.retired = s.acquired := by
  have := reachable_safe h
  simp [Safe] at this
  omega

/-- A reader cannot acquire an unfinished write. -/
theorem acquire_only_completed {s : State} (h : Reachable s)
    (published : s.acquired < s.completed) :
    s.acquired + 1 ≤ s.completed ∧ s.completed ≤ s.issued := by
  have := reachable_safe h
  simp [Safe] at this
  omega

/-- A release is justified by completed reads, not just issued reads. -/
theorem release_only_retired {s : State}
    (done : s.released < s.retired) : s.released + 1 ≤ s.retired := by
  omega

/-- Equal parity is meaningful ONLY after proving the one-credit bound. -/
theorem parity_recovers_generation (produced consumed : Nat)
    (lo : consumed ≤ produced) (hi : produced ≤ consumed + 1)
    (same : produced % 2 = consumed % 2) : produced = consumed := by
  omega

/-- Without that bound, two generations are invisible to a parity comparison. -/
theorem parity_alone_is_insufficient :
    (2 : Nat) % 2 = 0 % 2 ∧ (2 : Nat) ≠ 0 := by decide

/-- Non-vacuity: a full transaction is reachable at EVERY generation. -/
theorem completed_cycle (n : Nat) : Reachable ⟨n, n, n, n, n⟩ := by
  induction n with
  | zero => exact Reachable.initial
  | succ n ih =>
    have a := Reachable.next ih (Step.issue _ rfl)
    have b := Reachable.next a (Step.complete _ (by simp))
    have c := Reachable.next b (Step.acquire _ (by simp))
    have d := Reachable.next c (Step.retire _ (by simp))
    exact Reachable.next d (Step.release _ (by simp))

#print axioms step_preserves
#print axioms reachable_safe
#print axioms reuse_only_after_retirement
#print axioms acquire_only_completed
#print axioms release_only_retired
#print axioms parity_recovers_generation
#print axioms parity_alone_is_insufficient
#print axioms completed_cycle
end FrostLifecycle
