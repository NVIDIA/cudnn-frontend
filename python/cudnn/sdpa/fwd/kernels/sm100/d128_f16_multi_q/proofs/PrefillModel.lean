-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
import Std

/-!
An executable numerical model of the existing SM100 D128 prefill softmax step.
Operators are ordered, opaque functions, not real-number arithmetic. In
particular, there are NO associativity, commutativity, or exp/log axioms.

The source tracer checks the existing helper against `softmaxModel` below.
This does not verify compiler contraction/FTZ, memory/barrier correctness, or
the instruction semantics. Source-level mul/sub is not claimed to be two
separately rounded instructions after lowering.
-/
namespace FrostPrefill

inductive Expr where
  | var (index : Nat)
  | lit (bits : String)
  | unary (op : String) (x : Expr)
  | binary (op : String) (x y : Expr)
  | ternary (op : String) (x y z : Expr)
  deriving DecidableEq, Repr

structure Algebra (α : Type) where
  input : Nat → α
  literal : String → α
  unary : String → α → α
  binary : String → α → α → α
  ternary : String → α → α → α → α

def eval (p : Algebra α) : Expr → α
  | .var i => p.input i
  | .lit b => p.literal b
  | .unary op x => p.unary op (eval p x)
  | .binary op x y => p.binary op (eval p x) (eval p y)
  | .ternary op x y z => p.ternary op (eval p x) (eval p y) (eval p z)

theorem equal_expr_equal_bits (p : Algebra α) (a b : Expr) (h : a = b) :
    eval p a = eval p b := congrArg (eval p) h

def zero : Expr := .lit "00000000"
def negInf : Expr := .lit "ff800000"
def add := Expr.binary "add"
def mul := Expr.binary "mul"
def sub := Expr.binary "sub"
def maxFTZ := Expr.binary "max_ftz"
def exp2 := Expr.unary "exp2_fast"
def choose := Expr.ternary "select"

-- The real helper folds each group of up to three, then repeats.
def maxRound : List Expr → List Expr
  | a :: b :: c :: rest => maxFTZ (maxFTZ a b) c :: maxRound rest
  | [a, b] => [maxFTZ a b]
  | xs => xs

def maxTree : Nat → List Expr → Expr
  | 0, xs => xs.headD negInf
  | fuel + 1, xs =>
    if xs.length ≤ 1 then xs.headD negInf else maxTree fuel (maxRound xs)

def leftSum : List Expr → Expr
  | [] => zero
  | x :: xs => xs.foldl add x

instance : Inhabited Expr := ⟨zero⟩

def laneSum (prob : List Expr) (chunk parity : Nat) : Expr :=
  leftSum ((List.range 32).map fun k => prob[chunk * 64 + parity + k * 2]!)

structure SoftmaxStep where
  maximum : Expr
  safeMaximum : Expr
  evenSum : Expr
  oddSum : Expr
  alpha : Expr
  probability : List Expr
  deriving DecidableEq, Repr

-- Input variables: max=0, safe_max=1, even_sum=2, odd_sum=3, scale=4;
-- incoming QK accumulator columns start at 16. Each is an arbitrary term.
def scoreInputs (visible : Nat → Bool) : List Expr :=
  (List.range 128).map fun k => if visible k then .var (16 + k) else negInf

def softmaxModel (scores : List Expr) : SoftmaxStep :=
  let current := mul (Expr.binary "max" (maxTree 4 (scores.take 64))
    (maxTree 4 (scores.drop 64))) (.var 4)
  let update := Expr.binary "gt" (sub current (.var 0)) (.lit "41000000")
  let maximum := choose update current (.var 0)
  let safe := choose (Expr.binary "eq" maximum negInf) zero maximum
  let alpha := exp2 (Expr.binary "min" (sub (.var 1) safe) zero)
  let prob := scores.map fun s => exp2 (sub (mul s (.var 4)) safe)
  { maximum := maximum
    safeMaximum := safe
    evenSum := add (mul (.var 2) alpha) (add (laneSum prob 0 0) (laneSum prob 1 0))
    oddSum := add (mul (.var 3) alpha) (add (laneSum prob 0 1) (laneSum prob 1 1))
    alpha := alpha
    probability := prob.map (Expr.unary "cvt_bf16") }

def evaluateStep (p : Algebra α) (s : SoftmaxStep) :
    α × α × α × α × α × List α :=
  (eval p s.maximum, eval p s.safeMaximum, eval p s.evenSum,
   eval p s.oddSum, eval p s.alpha, s.probability.map (eval p))

theorem equal_step_equal_bits (p : Algebra α) (a b : SoftmaxStep) (h : a = b) :
    evaluateStep p a = evaluateStep p b := congrArg (evaluateStep p) h

/- The ordering theorem is independent of what `step` computes. Instantiating
   it with an attention state includes every rounding performed by that step.
   It allows arbitrary interleaving of DIFFERENT rows, not permuting KV tiles
   within one row. Output ownership of the real kernel remains an obligation. -/
def serial (step : σ → τ → σ) : List τ → σ → σ
  | [], s => s
  | t :: ts, s => serial step ts (step s t)

structure Event (τ : Type) where
  row : Nat
  tile : τ

def applyEvent (step : σ → τ → σ) (states : Nat → σ) (e : Event τ) : Nat → σ :=
  fun r => if r = e.row then step (states r) e.tile else states r

def runEvents (step : σ → τ → σ) : List (Event τ) → (Nat → σ) → (Nat → σ)
  | [], states => states
  | e :: es, states => runEvents step es (applyEvent step states e)

def rowTiles (row : Nat) : List (Event τ) → List τ
  | [] => []
  | e :: es => if row = e.row then e.tile :: rowTiles row es else rowTiles row es

theorem project_interleaved_rows (step : σ → τ → σ) (events : List (Event τ))
    (states : Nat → σ) (row : Nat) :
    runEvents step events states row = serial step (rowTiles row events) (states row) := by
  induction events generalizing states with
  | nil => rfl
  | cons e es ih =>
    rw [runEvents, ih]
    by_cases h : row = e.row
    · simp [rowTiles, applyEvent, h, serial]
    · simp [rowTiles, applyEvent, h]

theorem schedule_independent_for_row (step : σ → τ → σ)
    (a b : List (Event τ)) (states : Nat → σ) (row : Nat)
    (sameOrder : rowTiles row a = rowTiles row b) :
    runEvents step a states row = runEvents step b states row := by
  rw [project_interleaved_rows, project_interleaved_rows, sameOrder]

-- Match the logical request, even when its physical batch slot changes.
theorem request_position_independent (step : σ → τ → σ)
    (a b : List (Event τ)) (left right : Nat → σ) (i j : Nat)
    (sameInitial : left i = right j)
    (sameTiles : rowTiles i a = rowTiles j b) :
    runEvents step a left i = runEvents step b right j := by
  rw [project_interleaved_rows, project_interleaved_rows, sameInitial, sameTiles]

-- A local source/model equality can be lifted to any finite KV traversal.
-- The caller still has to connect each real QK/PV/mask step to that equality.
theorem serial_step_congruence (source model : σ → τ → σ)
    (sameStep : ∀ s t, source s t = model s t) (tiles : List τ) (s : σ) :
    serial source tiles s = serial model tiles s := by
  induction tiles generalizing s with
  | nil => rfl
  | cons t ts ih =>
    simp only [serial, sameStep]
    exact ih (model s t)

theorem segmentation_without_reset (step : σ → τ → σ) (a b : List τ) (s : σ) :
    serial step (a ++ b) s = serial step b (serial step a s) := by
  induction a generalizing s with
  | nil => rfl
  | cons t ts ih => simpa [serial] using ih (step s t)

-- A deliberately non-associative recurrence: inner-tile reorder is observable.
example : serial (fun s t : Nat => s * 10 + t) [1, 2] 0 ≠
    serial (fun s t : Nat => s * 10 + t) [2, 1] 0 := by decide

/- Existing correction's warp-wide ballot can skip multiplication. The proof
   names the exact primitive property needed to erase that lane dependency.
   `mulIdentity` must be justified for the admitted values/instruction mode;
   Lean is not deriving IEEE or FTZ semantics here. -/
def ballotCorrection (mul : α → α → α) (o alpha : α) (allOne : Bool) : α :=
  if allOne then o else mul o alpha

theorem ballot_does_not_change_row (mul : α → α → α) (isOne : α → Bool)
    (o alpha : α) (allOne : Bool)
    (voteSound : allOne = true → isOne alpha = true)
    (mulIdentity : isOne alpha = true → mul o alpha = o) :
    ballotCorrection mul o alpha allOne = mul o alpha := by
  cases h : allOne with
  | false => simp [ballotCorrection]
  | true =>
    change o = mul o alpha
    exact (mulIdentity (voteSound h)).symm

-- Without mulIdentity, the proposed theorem really can be false.
example : ballotCorrection Nat.add 7 1 true ≠
    ballotCorrection Nat.add 7 1 false := by decide

#print axioms equal_step_equal_bits
#print axioms project_interleaved_rows
#print axioms schedule_independent_for_row
#print axioms request_position_independent
#print axioms serial_step_congruence
#print axioms segmentation_without_reset
#print axioms ballot_does_not_change_row

end FrostPrefill
