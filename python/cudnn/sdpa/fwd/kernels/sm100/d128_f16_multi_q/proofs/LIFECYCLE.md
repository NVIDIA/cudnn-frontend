# Resource-lifecycle pilot: how to read the evidence

The useful claim is **an abstract one-slot protocol is safe for arbitrarily
many generations, and selected F06 source segments retain the operations
needed to implement it**. The connection between these two statements is
still a review obligation. This is not a proof that F06 is race-free.

The kernel is unchanged. This pilot follows the existing F06 implementation
instead of replacing it with a proof-specific kernel.

## Reproduce

From the repository root, using Python's standard library and Lean 4.15.0:

```bash
python python/cudnn/sdpa/fwd/kernels/sm100/d128_f16_multi_q/proofs/check_lifecycle.py \
  --lean-bin /path/to/lean-4.15.0/bin/lean \
  --output-dir /tmp/f06-lifecycle
```

Expected: 39 source obligations, five rejected source mutations, eight Lean
theorems and one rejected Lean model mutation. `lifecycle.json` records exact source sites, enclosing control flow,
source hashes, mutation diagnostics and the Lean axiom audit. `Lifecycle.log`
retains the actual Lean output. An absent tool or a failed baseline fails the
command; it never becomes a successful skip. No GPU is used.

## The three resource families

The source checks inspect selected segments for the dense, unsplit D128
BF16 two-Q, two-CTA, TILE_N=128 profile of the numerical prototype. They do
not evaluate the host factory or certify every specialization of this engine.

| Resource | Existing release/reacquire mechanism | What this pilot checks | Still missing |
|---|---|---|---|
| K/V SMEM ring | `mb_k_empty` / `mb_v_empty` use `Producer.MMA_COMMIT`; the loader waits before filling again | K's two QK readers and V's four PV issue sites precede their completion-tracking release; dense fills name the full barrier and follow a credit wait | Slot arithmetic, barrier counts, multicast recipient sets, actual completion semantics, full/empty phase refinement and teardown |
| S/P/Stats TMEM family | Correction reads final Stats from the S head, then signals `mb_stats_read`; next work item's QK waits | Final Stats LD → LD wait → credit, both next-QK gates, and the empty-work credit waits | This covers only the **final-Stats-to-next-QK backedge**. P/S overlap, intra-item alpha reuse, all TMEM fences and the entire memory map remain unverified |
| O SMEM staging | Correction waits for empty before its first store; TMA store warp drains source reads before releasing empty | First-chunk gate; proxy fence and publication; TMA issue → commit → wait(0) → empty arrive; wrapper uses `read=True` | Address/layout ownership, participation and counts, all output/alias specializations, compiler scheduling and final GMEM visibility |

The source checker uses AST sites and reviewed enclosing-control-flow
snapshots in `lifecycle_contexts.json`. Thus putting a wait under `if False`
cannot satisfy an obligation merely because its source line precedes a store.
The snapshot is a **regression baseline**, not a specification of correct
control flow. Updating it requires reviewing the changed protocol; do not
regenerate it just to silence a failure. Unknown or additional unselected
operations are not comprehensively analyzed by this pilot.

A sixth negative control weakens the Lean release rule to require only an
issued read. The invariant proof must then fail at `step_preserves`; a parser
error does not count.

The five source mutations change the K completion credit to a plain thread
arrival, remove the final Stats load wait, weaken the O drain to wait(1), hide
the O wait in a false branch, and move the O empty gate from chunk 0 to chunk 1.
Each must fail the intended obligation. These are source diagnostics, not GPU
race counterexamples. Production files are copied before mutation.

## Walk through the Lean proof

Open [Lifecycle.lean](Lifecycle.lean). It has no arithmetic model of attention;
the resource protocol does not need one. A state contains five **natural-number
counters** for one physical slot:

```text
issued     number of writes issued
completed  number of writes completed / published
acquired   number of aggregate reads started
retired    number of aggregate reads completed
released   number of credits returned for reuse
```

An aggregate read represents **every** access by all consumers of that slot,
including async engines. Showing that the real barrier covers those consumers
is a separate obligation; it is not obtained by naming them "aggregate".

The invariant `Safe` is:

```text
released <= retired <= acquired <= completed <= issued <= released + 1
```

`Step` describes five permitted transitions. For example, issuing a write
requires `issued = released`; releasing a credit requires `released < retired`.
The latter is a completed-read condition, not an issued-read condition.

1. `step_preserves` checks every transition. `cases step` splits the five
   constructors; `simp_all [Safe]` unfolds the field updates; `omega` discharges
   the resulting integer inequalities and produces a proof checked by Lean.
2. `reachable_safe` inducts over an arbitrary-length execution. It uses the
   previous lemma at each step, so the result is not restricted to a bounded
   number of iterations.
3. `reuse_only_after_retirement` combines the invariant with the write credit.
   If `issued = released`, every counter between them must also be equal.
   Therefore no old read is outstanding when the next write begins.
4. `parity_recovers_generation` proves that equal parity identifies equal
   generations **given a one-credit bound**. `parity_alone_is_insufficient`
   exhibits the missing-bound problem: generations 0 and 2 have equal parity.
5. `completed_cycle` constructs a reachable completed cycle for every `n`.
   The model allows useful executions, rather than proving safety by forbidding
   all work. This is not a fairness or deadlock-freedom theorem.

`#print axioms` audits all eight theorems. The checker permits only Lean's
standard `propext`, `Classical.choice` and `Quot.sound`, and rejects unfinished
proofs and native-evaluation proof shortcuts. Review the specification and
preconditions as carefully as the proof: a true theorem about an overly
restrictive protocol does not validate a real kernel.

## What increases confidence, and what would complete the argument

This PR provides two reusable checks: numerical expression conformance and
resource-protocol regression obligations, alongside an unbounded proof of a
small protocol. Neither is a whole-kernel certificate. In particular, the
Python AST extractor, the mapping of source events to protocol transitions,
and the CUDA/compiler implementation of those events are not verified.

The next useful increment is a compiler-level event extraction for **one
backedge**, preferably O staging: extract allocation ranges, accesses, async
issue/completion, proxy fences, barrier recipients/counts and generations from
the actual lowered program. Compare against the same source profile. Add a
bounded schedule explorer with concrete violating traces and a source/IR
refinement lemma, then connect its transitions to `Step`. Explicitly include
empty work, first use, wraparound and exit. Only then extend to the TMEM alias
and clustered K/V protocols.

For `tcgen05`, completion and cross-thread ordering are separate obligations;
source order alone establishes neither. Inspect the actual lowering against
the [PTX specialized synchronization rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-specialized-synchronization)
and the repository's Python Rule 11 and SDPA Rule S3. This pilot makes no
finding that a missing spelling in Python is a hardware bug.

A selected-plan evidence record should identify kernel/configuration, source
and compiler versions, theorem/preconditions, trusted primitives, device
checks and exclusions. Keep numerical invariance, functional correctness,
race freedom, bounds safety and progress as separate claims. No new public
numerical note or capability is justified by this pilot alone.
