# F06 prefill verification prototypes

These development tools belong to `sdpa_fwd_sm100_d128_multi_q` (F06, engine
20535). They model the existing `../kernel.py`; they do not modify the kernel,
the engine's public notes, admission, dispatch, or launch behavior. Run them
from a source checkout. No Lean, CUDA or tensor-framework dependency is added
to the runtime package.

The initial proof lived in `benchmark/batch_invariance/` and read the former
`sm100/prefill_d128_f16.py`. This version follows the isolated multi-Q owner,
with no dependency on that benchmark directory or its reference-batch planning
prototype. `PrefillModel.lean` is unchanged from that initial model.

## Resource lifecycle and related work

The [lifecycle pilot](LIFECYCLE.md) adds selected K/V, final-Stats and O
reuse checks plus a separate unbounded Lean protocol proof. It includes a
step-by-step proof walkthrough and explicit remaining obligations. The
[research survey](RELATED_WORK.md) compares recent numerical, safety,
PTX-level and instrumentation approaches.

The numerical checker below and the lifecycle checker are independent;
run both. Neither adds a public safety or numerical guarantee to the engine.

## Numerical checked scope

The source/model checks concern **one invocation of `_softmax_kv_body`** with
D128 BF16, 128 KV columns, two CTAs, threshold 8, and the dense top-left-causal
plain path. Bias, KV data masks, sink, score side outputs, packing, THD, paged
KV, split-KV, mixed exp2, fused hardware max, and padded-tail premasking are
outside this profile. QK accumulators and prior softmax state are symbolic
inputs; this is not a random floating-point test.

Five cases cover an unmasked interior tile, causal visibility of 1, 64 or all
128 columns, and a fully masked tile. The unmasked case is an **interior of
the causal profile**, not a claim about the noncausal engine path. Both query
sub-tile owners are compared by the source tracer. These five geometries do
not establish every mask, shape or configuration supported by the engine.

The source's specialization assignments are evaluated using the declared
profile, then checked. Unsupported flags fail rather than being silently
forced off. Reports retain the resolved flags and source/bridge/model hashes.
Host factory/dispatch eligibility is not verified by this tool.

The refactored source includes new numerical variants: mixed exp2, joint row
max, and a D64 balanced tail reduction. They are not automatically equivalent
to this profile. `scope.json` records the evidence boundary; it is **development
metadata, not a public numerical note or an admission policy**.

## Run

Requires Python's standard library and Lean **4.15.0**, with `Std` only:

```bash
python python/cudnn/sdpa/fwd/kernels/sm100/d128_f16_multi_q/proofs/check.py \
  --lean-bin /path/to/lean-4.15.0/bin/lean \
  --output-dir /tmp/f06-lean-results
```

Expected: one `baseline: PASS` and four `REJECTED AS EXPECTED` messages.
The checker generates `PrefillSource.lean`, exact Lean logs and `report.json`.
It copies sources into temporary directories for each negative control:

1. Reverse the existing even/odd left-fold order.
2. Move rescaling after the addition of the current tile's sum.
3. Change chunk 0's `exp2` primitive mode.
4. Widen the causal mask's exclusive upper edge by one column.

All mutations must fail at the intended source/model equality, not at a parser
or import error. The baseline accepts only the standard `propext` axiom and
does not use `sorry`, `admit` or `native_decide`. Expected failure logs may
contain Lean's error-recovery `sorryAx`; those files are not accepted proofs.

## How the proof works

`trace_softmax.py` executes the real Python helper bodies over a small symbolic
vector algebra. It records an ordered expression tree, including the grouped
max tree, even/odd left folds, threshold skip, alpha clamp and BF16 conversion.
The independent `PrefillModel.lean` describes that sequence. Lean checks that
the emitted expression and model are equal, so evaluation under any shared
deterministic primitive interpretation agrees. Floating-point associativity
and real-number softmax identities are not assumed.

The new bit-word masking helpers are also extracted from source. Their two
specific saturating-shift PTX snippets have an explicit host interpretation;
unknown snippets are rejected. This interpreter, the Python extraction layer,
and integer geometry are **not formally verified**. Publish fences and barrier
events are recorded, not proved. `track_true_max=False` is explicit, and its
unused state is checked to pass through unchanged.

The Lean file separately proves that arbitrary interleaving of independent
rows, or changing a request's batch slot, preserves results if its initial
state and ordered tile sequence are unchanged. The loop-segmentation theorem
requires carrying the exact state; it does not cover resetting split-KV
partials and merging them. These abstract results are not yet connected to
the real scheduler and memory ownership.

## Relationship to numerical notes

This evidence currently supports only a scoped **source softmax expression
match**. It does not justify marking the whole engine deterministic, batch
invariant, or bitwise equal to the single-Q/quadrant/swap-AB engines.

An eventual selected-plan note needs the admitted numerical profile and
configuration domain. Cross-engine equality additionally names the other
profile/implementation and comparison scope. Sharing an engine directory, or
passing this local proof, is not sufficient to establish such a relation.

Remaining obligations include QK/PV operand mapping and MMA issue order,
correction and O/LSE epilogues, compiler contraction/FTZ, primitive semantics,
address ownership, synchronization, and actual plan/partition selection.
The ballot lemma explicitly assumes multiplication by one preserves the
admitted value's bits; it does not prove that hardware property.

No GPU validation is performed by this checker. Numerical kernel changes or
public notes require the repository's determinism qualification separately.
