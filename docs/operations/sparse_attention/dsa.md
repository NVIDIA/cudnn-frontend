# DeepSeek Sparse Attention (DSA)

**This is an experimental API and subject to change.**

Stream parameter types are operation-specific. Sparse-attention forward uses
the current PyTorch stream when its stream argument is `None`; an explicit
stream must be a concrete `cuda.bindings.driver.CUstream`. Convert a
`torch.cuda.Stream` with `cuda.bindings.driver.CUstream(stream.cuda_stream)`.
The special `CUstream(2)` per-thread default sentinel is rejected by Torch
interop: it cannot safely represent an allocator stream identity across host
threads. Use a concrete stream instead.

## Overview

The DeepSeek Sparse Attention (DSA) module integrates a set of CuTe-DSL
kernels that support the sparse-attention path used by DeepSeek-style models.
Most kernels target Hopper (SM90) and Blackwell (SM100+) GPUs. Sparse-attention
forward supports SM90 and the mapped SM100-family capabilities 10.0, 10.3,
and 10.7; the combined compressed-logits + Top-K path remains SM100-only.
Dense Indexer Forward and standalone Indexer Top-K support both architectures.
The kernels are delivered as Python classes /
wrappers that follow the same `APIBase` pattern as other cuDNN Frontend
operations.

**Scope:** this module ships CuTe-DSL kernels for sparse-attention forward and
backward, indexer scores/top-K, sparse/dense score recompute, and sparse/dense
indexer backward. On SM90, sparse forward covers H16/H32/H64/H96/H128 Prefill;
on SM100 it covers regular H64 and small-top-k H128 Prefill. It does not
include a decode or split-KV forward mode.

The module packages the following operations:

1. **Sparse Attention Forward** – sparse Prefill MQA on SM90 and SM100.
2. **Sparse Attention Backward** – DSA backward for flat MQA tensors on
   SM90/SM100.
3. **Indexer Forward** – CuTe-DSL score kernel (Q @ K^T, ReLU, head reduce,
   ratio causal mask) that materializes dense scores.
4. **Combined Indexer Forward + Top-K** – SM100 compact score generation,
   Top-K selection, and optional Top-K softmax in one public API call.
5. **Indexer Top-K** – SM90+ CuTe-DSL radix top-K kernel with per-row
   ``seq_lens``.
6. **Sparse Indexer / Attention Score Recompute** – sparse (top-K) recompute
   of indexer and attention scores for training loss.
7. **Dense Indexer / Attention Score Recompute** – dense (full-KV) analogues
   of the above.
8. **Indexer Backward** – three-stage pipeline (score-grad, three
   GEMMs, dtype cast) for sparse top-K score tensors.
9. **Dense Indexer Backward** – full-KV counterpart of Indexer Backward.

### Architecture

```text
Q, K, W ──┬─► IndexerForward ──► scores ──► IndexerTopK ──► topk_idxs
          └─► IndexerForwardTopK ──► topk_idxs, logits, predict
                                                             │
                                                             v
                              SparseAttentionForward ──► out, lse
                                                             │
                                                      dout ──┤
                                                             v
                                               SparseAttentionBackward
                                                             │
                                                             v
                                                    dq, dkv, d_sink

Training-score loss path:
   attn_score, index_score ──► IndexerBackward ──► d_index_q, d_weights, d_index_k
   (SparseIndexer/AttnScoreRecompute and DenseIndexer/AttnScoreRecompute
   produce these score tensors; DenseIndexerBackward consumes dense raw scores.)
```

---

## Installation

```bash
pip install nvidia-cudnn-frontend
```

---

## API Usage

### DSA Namespace

```python
from cudnn import DSA

DSA.SparseAttentionBackward
DSA.sparse_attention_backward_wrapper

DSA.SparseAttentionForward
DSA.sparse_attention_forward_wrapper

DSA.IndexerForward
DSA.indexer_forward_wrapper
DSA.indexer_forward_top_k_wrapper
DSA.compress_topk_cand_buffer_size
DSA.compress_topk_cand_buffer_size_thd

DSA.IndexerTopK
DSA.indexer_top_k_wrapper
DSA.local_to_global_wrapper
DSA.compactify_wrapper

DSA.SparseIndexerScoreRecompute
DSA.sparse_indexer_score_recompute_wrapper

DSA.SparseAttnScoreRecompute
DSA.sparse_attn_score_recompute_wrapper

DSA.DenseIndexerScoreRecompute
DSA.dense_indexer_score_recompute_wrapper

DSA.DenseAttnScoreRecompute
DSA.dense_attn_score_recompute_wrapper

DSA.IndexerBackward
DSA.indexer_backward_wrapper

DSA.DenseIndexerBackward
DSA.dense_indexer_backward_wrapper
```

---

## Components

### 1. Sparse Attention Forward

Sparse Prefill forward for flat MQA tensors. For query row `t`, head `h`, and
valid top-k slot `j`, the score is
`softmax_scale * dot(q[t,h,:], kv[topk_idxs[t,j],:])`. Invalid and OOB indices
have score `-inf`; duplicate indices remain distinct softmax slots. The
attention sink participates only in the output denominator, not in any stats.

- **Inputs**
  - `q`: `(total_S_q, H, D_qk)` FP16 or BF16
  - `kv`: `(total_S_kv, D_qk)`, with the same dtype as `q` (K = V latent; MQA)
  - `topk_idxs`: `(total_S_q, K)` INT32 global token ids; K may be any
    nonnegative value. SM100 pads a tail internally; SM90 masks it in place.
  - `attn_sink` (optional): `(H,)` FP32
  - `topk_length` (optional): `(total_S_q,)` INT32, clamped to `[0, K]`
  - `softmax_scale` (optional runtime float): defaults to `1 / sqrt(D_qk)`
  - `indexer_topk`: compile-time prefix length
- **Outputs** — stable `TupleDict` keys:
  - `out`: `(total_S_q, H, 512)`, with the same dtype as `q`
  - `max_logits`: `(total_S_q, H)` FP32
  - `lse`: `(total_S_q, H)` FP32, KV-only and excluding the sink
  - `lse_indexer`: `(total_S_q, H)` FP32, or `None` for `indexer_topk=0`
- **Variants**
  - SM90: `H in {16, 32, 64, 96, 128}`, `D_qk in {512, 576}`, and
    `indexer_topk in {0, 512, 1024, 2048}`
  - SM100 H64: `D_qk in {512, 576}`, `indexer_topk in {0, 512, 1024, 2048}`
  - SM100 H128 small-top-k Prefill: `D_qk=512`,
    `indexer_topk in {0, 512, 1024}`
- **Indexer prefix constraint** — `indexer_topk <= K`. When `topk_length` is
  present, a meaningful `lse_indexer` additionally requires
  `topk_length >= indexer_topk` for that row. Shorter rows retain the SM100
  source kernel's `+inf` sentinel; their `out`, `max_logits`, and full `lse`
  outputs remain valid.
- **Empty rows** — `out=0`, `max_logits=-inf`, `lse=+inf`; an enabled
  `lse_indexer` is also `+inf`.
- **Compilation lifecycle** — on SM90, `SparseAttentionForward.compile()`
  compiles a storage-free symbolic plan and `execute()` only launches it with
  caller-owned outputs. The convenience wrapper allocates outputs and stages
  layouts before calling the same plan. The existing SM100 direct path retains
  its concrete-tensor cache and first-execute compilation.
- **CUDA Graph and workspace** — the prepared SM90 `execute()` path is
  allocation-free, performs no host read or JIT, and requires no workspace.
  Compile before capture and provide preallocated outputs. Runtime `total_S_q`,
  `total_S_kv`, and logical K may change per `execute()`. H, Dqk, dtypes, and
  optional-operand presence must match the declaration; a plan for another
  supported H reuses the same compiled kernel without a new specialization.
- **Aliasing** — prepared SM90 outputs must be byte-range-disjoint from every
  input and from one another. Empty tensors occupy no byte range.
- **Layout and stream** — the convenience wrapper materializes non-contiguous
  or under-aligned inputs on the selected stream. Prepared SM90 `execute()`
  instead requires contiguous, adequately aligned tensors and raises before
  launch if the contract is not met. An explicit `current_stream` for prepared
  execution (or `stream` for the wrapper) must be a
  `cuda.bindings.driver.CUstream` belonging to `q.device`.

For `D_qk=576`, QK uses all 576 dimensions while V/output uses only the first
512 KV dimensions. `attn_sink=None` is equivalent to an all-`-inf` sink.

#### Credits and cuDNN Frontend extensions

The SM90 kernel schedule and pipeline are adapted primarily from
[FlashMLA's MIT-licensed Hopper sparse-prefill implementation](https://github.com/deepseek-ai/FlashMLA/tree/4f38f29ef6793c228363e4af5be66d44e81167ba/csrc/kernels/sm90/prefill/sparse).
The cuDNN Frontend port extends that foundation with H16/H32/H64/H96/H128
coverage, arbitrary logical K tails and boundary handling, a bounded symbolic
compile cache, and an allocation-free prepared-execution contract for CUDA
Graph capture and replay.

```python
import math

result = DSA.sparse_attention_forward_wrapper(
    q,
    kv,
    topk_idxs,
    attn_sink=attn_sink,
    topk_length=topk_length,
    softmax_scale=1.0 / math.sqrt(q.shape[-1]),
    indexer_topk=512,
)
out = result["out"]
max_logits = result["max_logits"]
lse = result["lse"]
lse_indexer = result["lse_indexer"]
```

### 2. Sparse Attention Backward

Backward pass for DeepSeek Sparse Attention. Expects the forward wrapper's
`out` and KV-only `lse` (or equivalent tensors).

- **Inputs**
  - `q`: `(total_S_q, H, D)` BF16/FP16
  - `kv`: `(total_S_kv, D)` (K = V; MQA)
  - `out`, `dout`: `(total_S_q, H, D_v)`
  - `lse`: `(total_S_q, H)` FP32
  - `attn_sink`: `(H,)` FP32
  - `topk_idxs`: `(total_S_q, topk_max)` INT32 (global)
  - `topk_length` (optional): `(total_S_q,)` INT32 — per-query valid count.
    The H128 two-CTA backends (D512 and D576) clamp it to `[0, topk_max]`
    and ignore every slot after that prefix, including valid indices pointing
    to nonfinite KV rows; the other backends expect `topk_length <= topk_max`.
    Omitting this tensor uses all `topk_max` slots.

On Blackwell SM100/SM103, the public backward entry point automatically selects
the tuned kernel from the device, dtype, and tensor shape. On SM100 (10, 0) and
SM103 (10, 3) devices, BF16 H128 with `head_dim = head_dim_v = 512` and
`topk_max ∈ {128, 512, 1024, 1152, 2048}` uses the two-CTA specialization.
Contiguous BF16 H128 with `head_dim = 576`, `head_dim_v = 512`, and the same
`topk_max` set uses the H128/D576 two-CTA specialization. H16 with
`head_dim=576` uses the dedicated M128 sparse-row pipeline. FP16, other head
counts and dimensions, and every other `topk_max` retain the existing
generic/H16/H32 selection. Other compute capabilities, including SM107, do not
select the two-CTA paths. No backend or tile-size argument is required. SM90
continues to use its Hopper-specific implementation. Every input must be
contiguous on every route: `check_support()` declines a strided tensor with
`NotImplementedError` naming it and its strides, and `execute()` rejects one
with `ValueError`; the plan never repacks an input.

The SM100 H16/H32/H96 D576 specializations compile the physical row stride of
`topk_idxs` from `topk_max`, while the query count and per-query `topk_length`
remain dynamic. H64 and the generic/H128 D512 paths retain their existing
layout; H128 D576 already uses a static `topk_max` stride.

For contiguous BF16 H16/H32 D576 inputs, `q_cluster_mode="adaptive_pair"`
enables adjacent-query KV sharing. A device-side transform partitions each
pair into one tile-aligned intersection and two unique segments, then evaluates
the intersection with twice the head count. Its bounded hash storage scales
with `topk_max` rather than `total_S_kv`; invalid, duplicate, low-overlap, and
an odd final query retain the ordinary per-query work. The mode is opt-in,
non-deterministic, CUDA Graph compatible, and supports `topk_max <= 2048`.
It is intended for high-overlap workloads; the device-side partition still
runs when a pair falls back to ordinary work. H16 `topk_max <= 256` and H32
`topk_max <= 128` accept the mode but compile the ordinary backend directly.

With the default `deterministic=False`, the H128 specialization keeps the five tensor-core products in one
two-CTA main kernel. It publishes FP32 O-dot-dO and folded-LSE statistics to the
caller-provided scratch workspace, converts the FP32 dKV workspace to the public BF16
output, and completes dSink with a separate FP32 reduction kernel. The helper
launches do not change the two-CTA topology of the core computation.

The H128/D576 specialization applies the same two-CTA topology to the MLA
layout (`head_dim = 576`, `head_dim_v = 512`). Its compiled sequence
with `deterministic=False` initializes the FP32 dKV workspace and the dSink output on the launch stream,
accumulates dKV in that workspace, and converts it to the public BF16 output
with a helper kernel. With the default `deterministic=False`, short query
batches split each query's top-k range across otherwise idle two-SM clusters;
the split count is fixed when the plan is compiled.

On SM90, `deterministic=True` uses a row-aware relay with logical cluster size
`C in {4, 8, 16, 32, 64}` for H16/H32/H64/H96/H128. The public
`deterministic_relay_cluster_size` argument selects `C` and defaults to 16.
Each 128-query wave gives one FP32 dKV plane to every logical relay group. C4,
C8, and C16 launch one physical cluster per group; C32 and C64 run each group
as two or four same-stream physical C16 phases. Kernel completion orders phase
reuse without making a CTA wait for a nonresident cluster. A setup kernel builds
packed `C`-bit rank masks for every referenced
`(query wave, M64 head tile, KV row)`. A relay CTA waits only for lower logical
ranks that actually update the same row; a 128-thread WG1
barrier carries the acquired visibility to every reducer lane, and the matching
barrier publishes all relaxed FP32 reductions before one lane release-clears
that rank's ownership bit at GPU scope when a successor exists. Head tiles and
later query waves are stream-ordered, while disjoint sparse rows within a
cluster can progress independently. The postprocess kernel combines the
`128 / C` planes in fixed order, and one warp per head computes `d_sink` with a
fixed strided-lane/tree reduction order. Duplicate valid top-k indices within
one query row remain outside this deterministic contract.

The SM90 deterministic dKV allocation is
`(128 / C) * (K64 + mask_rows) * D32 * sizeof(float)`, where
`K64 = round_up(total_S_kv, 64)`, `D32 = round_up(D, 32)`, and
`mask_rows = ceil(ceil(total_S_q / 128) * ceil(H / 64) *
(K64 / (mask_bits / C)) * (mask_bits / 32) / D32)`, where `mask_bits` is
64 for C64 and 32 otherwise.
The caller-owned workspace is cleared and the masks are rebuilt on every
execution, so changing `topk_idxs` or `topk_length` between launches is safe.
After one eager execution populates the SM90 concrete-tensor compile caches,
the caller-owned output/workspace path is CUDA Graph capturable. Replay keeps
the declared tensor shapes and addresses fixed while allowing tensor contents,
including `topk_idxs` and `topk_length`, to change.
On the target H200 sweep (BF16 H64 D576/Dv512, KV=65536, top-k 2048), C16
measured 1.47--1.54x the non-deterministic latency with 1.13--1.26 GiB of
workspace and remains the conservative default. C32 measured 1.34--1.37x
with 0.57--0.70 GiB, the best time/memory point in that workload. C64 reduced
workspace further to 0.29--0.41 GiB, but four sequential physical phases
raised latency to 2.30--2.37x non-deterministic. C32 and C64 are therefore
explicit workload-specific choices rather than the automatic default; the
full table and environment are recorded in `benchmark/dsa/README.md`.

On SM100/SM103, `deterministic=True` retains the two-CTA Tensor Core kernels
for the supported BF16 H128/D512 and H128/D576 envelopes above. Queries run in
same-stream waves with one two-CTA pair per query: D512 spreads the queries
evenly over waves of at most 74, and D576 waves hold one query per two-SM
cluster (at most 76) per wave round, at most 74 per round in three-round plans
(devices with 152 or more SMs leave the remaining SMs idle there). Each wave
accumulates dKV either in private FP32 slots or in conflict-rank storage
(below); ordered duplicate-index writers and stream ordering between waves give
each update a fixed accumulation order, and fixed-order reductions produce
`dKV` and `d_sink`. With either H128 two-CTA layout, duplicate indices within a
query row are inside the deterministic contract, as with the generic M64
conflict-rank layouts (below) and unlike its shard and relay layouts and the
SM90 deterministic path. The plan option `deterministic_dkv_mode` (below)
picks the layout of H128 two-CTA plans: under `'records'` (the default) the
plan chooses between the two layouts from the shapes, `'shards'` keeps the
private-plane layout for every shape, and `'chain'` is rejected. The
private-plane layouts hold 128 FP32 dKV planes for D512 and 96 for D576; H96
D576 two-CTA plans always use their 96-plane layout. `'shards'` restores the
private-plane layout but, for D512, not the bits of releases without
conflict-rank storage: D512 private-plane plans stream the dK/dV round operands
in four 32-head stages, which sums dKV in a different fixed order (dQ and
`d_sink` are unchanged).

**Conflict-rank dKV storage (deterministic H128 D512/D576).** Within a wave,
the queries that select a KV row are ranked by their position in the wave. The
query of rank `k < P` accumulates into dense FP32 plane `k` (the first write of
a row above plane 0 stores, later ones add); higher ranks write compact
overflow records, so every plane row and every record has a single writer per
wave and the main kernel needs no writer ordering. A row that one query selects
more than once takes per-slot records in every slot of that wave that selects
it.
A per-wave tail adds the overflow records to plane 0 in a fixed order (rank
order, with long chains summed in fixed segments) and the per-slot records in
(query, slot) order, with sorted slots for D576 top-k 1024 and wider; the final
fold adds each row's used planes in plane order. Small setup kernels chained
to the main kernels with programmatic dependent launch compute each wave's
ranks and destination records before it runs. D512 plans with top-k 512 or
less that keep at least three dense planes next to a doubled overflow area run
these steps in a setup warp in each CTA of the preceding main kernel instead,
and each wave's tail in the next main kernel's reducer warps, which shift a
row's ranks off plane 0 while that tail is pending.

- **When it is selected** — in `'records'` plans (the default; `'shards'`
  plans keep the private-plane layout), automatically, from the KV extent and
  the top-k width, whenever the conflict-rank workspace fits the private-plane
  layout (D576 at top-k 128 also requires the layout sized for its expected
  conflict load to need at most three quarters of it). Short KV extents
  relative to the top-k width, where the private planes are already small,
  keep the private-plane layout. The choice depends only on the batch's
  shapes and the mode, so one problem always takes the same path.
- **Workspace** — `scratch_workspace_bytes()` reports the selected layout; it
  never exceeds the private-plane size of the same problem. The dense planes
  fill a budget of about 1 GiB, but at least one plane (and the per-row state)
  is kept, so past a few hundred thousand KV rows the workspace grows with the
  KV extent: at `total_S_q=4096` about 2.1-2.4 GiB per million KV rows for D512
  and 2.5-2.7 GiB for D576, about 1% of the private-plane layout (256 and
  216 GiB per million rows). The D576 size also depends on the plan device's SM
  count, through the cluster pairs each wave round launches: query
  `scratch_workspace_bytes()` for the device that executes the plan, since a
  size from a device with fewer SMs can be too small (`execute()` then rejects
  the workspace). Every execution re-initializes the conflict-rank state it
  reads, so one workspace can be reused across executions with different
  `topk_idxs` and `topk_length`, and a captured plan replays new input contents
  like the private-plane plans.
- **Determinism** — outputs are bitwise reproducible from run to run on one
  GPU and software runtime. Duplicate indices within a query row, `-1` and
  out-of-range indices, and any `topk_length` are inside this contract. The
  FP32 summation order is a fixed function of the plan and the indices: D512
  waves follow the query count, D576 waves also the device's SM count, and D576
  one-round plans wider than top-k 128 that would stride their waves first run
  a probe launch that counts one wave's writers and picks strided or
  consecutive waves from the index contents. Results are not claimed to be
  bitwise equal across GPUs or software versions. Conflict-rank and
  private-plane plans sum dKV and `d_sink` in different orders; their results
  agree within FP32 rounding, not bitwise. dQ is bitwise equal in both layouts
  except for D576 conflict-rank plans with top-k 1024 and wider, which sort
  each query's rows by ranges of the KV extent. A sequence's dKV is not batch
  invariant: packing it with other sequences changes the waves, the ranks and
  the plane count that order its sums, and the sorted D576 plans also change
  its dQ. `d_sink` follows the query positions in both layouts (see
  [Determinism Support](/operations/determinism)).

Other deterministic envelopes with H16/H32/H64/H96/H128 run the generic M64
kernel. Its dKV storage is a plan-time choice, `deterministic_dkv_mode`
(below): by default on SM100/SM103, conflict-rank records wherever their
workspace does not exceed the shard or relay layout of the same plan, and that
layout elsewhere and on other devices. H16/H32 use a masked M64 head tile.
Deterministic H96 D576 outside the two-CTA envelope (FP16, other top-k widths,
or other devices) runs the generic M64 kernel only with conflict-rank storage,
so only on SM100/SM103; every other such plan composes the H64 kernel with the
tuned H32 tail, keeps 128 FP32 dKV planes, and rejects an explicit
`deterministic_relay_cluster_size`.

**Deterministic dKV storage (`deterministic_dkv_mode`).** The option of
`SparseAttentionBackward` and `sparse_attention_backward_wrapper` selects the
dKV storage of deterministic SM100/SM103 plans of the generic M64 kernel as
follows, and of the H128 two-CTA plans as described above:

- `'records'` (default) — conflict-rank storage. Each query wave runs 148
  query CTAs per head block (74 for 64 < H <= 128 when a full wave's records
  would exceed 0.4 GiB, and for split pairs, below). Within a wave, the
  queries that select a KV row are ranked by their wave position: rank `k`
  below the plan's plane count `P` accumulates into dense FP32 plane `k` of
  the row, higher ranks into slot-private overflow records, so every plane
  row and every record has one writer per wave. Setup kernels before each
  main kernel assign the wave's destinations and add the previous wave's
  overflow records to plane 0 in rank order (long runs in fixed segments); a
  row that one query selects more than once takes per-slot records, added in
  (query, slot) order. The conversion folds each row's used planes in plane
  order. `P` follows the expected number of writers per row and wave
  (`W * top-k / total_S_kv` for `W` query CTAs per wave) plus a margin,
  within about 1 GiB of planes.
  H96/H128 run both 64-head blocks of a query as one two-CTA cluster: head
  block 1 adds a tile after head block 0's reducer threads published their
  adds of that tile through per-thread GPU-scope release/acquire completion
  words. H <= 64 plans whose full-wave workspace exceeds 1.65 times the
  larger of the half-wave workspace and 1 GiB run each query as a two-CTA
  pair that splits its top-k tiles; the second CTA hands its FP32 dQ partial
  to the first through the workspace. Plans with at least eight waves whose
  KV rows fit in 48 MiB interleave the waves' queries (wave `w` of `n` holds
  queries `w, w + n, ...`).
- `'chain'` — the chain relay. Every head block of a wave runs in one launch
  of 296 query positions. CTAs claim (position, head block) tickets in start
  order and add into a shared plane row only after the row's previous chain
  member signalled it, so each plane row accumulates in a fixed order without
  overflow records. It applies only when the top-k width is at most 2048, the
  plan's dense planes span at most 2^19 rows (about 174K KV rows at its floor
  of three planes), and its workspace does not exceed the `'shards'` layout
  (for H96 D576, the 128-plane layout of the H64 + H32 composition); every
  other plan follows the `'records'` rule. In the table below, chain plans
  need less workspace than records plans; which mode is faster depends on how
  much the queries' top-k rows overlap, so profile both on the target
  workload.
- `'shards'` — the shard layout or the cluster relay described below, chosen
  from the query and KV extents alone, with the kernels, workspace and
  bitwise results of releases without this option.

Further properties of the option:

- **Validation** — values other than the default require `deterministic=True`
  and unknown values are rejected. `'chain'` cannot be combined with
  `deterministic_relay_cluster_size` and is rejected on SM90 and on the
  two-CTA backends. On the H128 two-CTA backends `'records'` chooses between
  conflict-rank and private-plane storage from the shapes and `'shards'`
  selects the private-plane layout; H96 D576 two-CTA plans keep their 96-plane
  layout under both. Under `'records'` and `'shards'`, an explicit relay size
  selects that cluster relay. The option is part of the compiled plan and of
  the wrapper's plan cache key.
- **Devices** — conflict-rank storage (the `'records'` and `'chain'` layouts
  and the H128 two-CTA conflict-rank layout) is provided on SM100 and SM103
  (compute capability 10.0 and 10.3). On other SM100-family devices (compute
  capability 10.7), a `'records'` plan runs as a `'shards'` plan, with the
  same kernels, workspace and results, and `'chain'` is rejected with a
  `ValueError` that names the compute capability.
  `flash_attn_bwd_sm100_workspace_size()` takes the keyword-only `max_topk`,
  `launch_pairs`, `dkv_mode` and `device_capability`: without `max_topk` it
  returns the shard, relay or private-plane size, an upper bound for every
  conflict-rank plan of the shape, and the default `device_capability=None`
  sizes for an SM100/SM103 device.
- **Workspace** — `scratch_workspace_bytes()` reports the selected layout and
  is authoritative. It never exceeds the `'shards'` layout of the same plan
  (for generic H96 D576 plans, the 128-plane layout of the H64 + H32
  composition that their `'shards'` plans run). A records plan sizes a full
  wave even when `total_S_q` is smaller, so its workspace does not shrink
  further with the query count. Every execution re-initializes the state it
  reads, so one workspace serves executions and CUDA Graph replays with
  different `topk_idxs` and `topk_length`.
- **Determinism** — each mode is bitwise reproducible from run to run on one
  GPU and software runtime. `-1` and out-of-range indices and `topk_length`
  values up to `topk_max` are inside the contract of every layout; records and
  chain plans also clamp larger lengths to `topk_max`. Duplicate indices
  within a query row are inside the contract of the records and chain layouts
  but not of the shard and relay layouts, which the other modes also select
  for some shapes; callers that need bitwise repeatability for every shape
  should pass unique valid indices within each query's active top-k prefix.
  The layouts sum dKV in different orders, so their dKV agrees within FP32
  rounding, not bitwise. `d_sink` has the same bits under every layout of the
  generic M64 kernel: one warp per head sums the queries in the same order (the
  H64 + H32 composition of H96 D576 is a separate kernel). dQ does not depend on the
  dKV layout either, except for chain plans, which gather a query's rows in
  sorted order, and records plans with split pairs, which add two dQ partials.
  dKV is not batch invariant once a batch spans several waves, with any layout:
  the layout, the waves and the plane or shard count follow the batch's query
  and KV extents. `d_sink` is: a uniform shift of the queries only rotates the
  lanes of that warp (see [Determinism Support](/operations/determinism)).
- **Compilation** — the wave size and the split and interleave choices are
  compile-time constants of a records plan. For given heads, head dimension,
  dtype, top-k width and KV extent, the wave order gives records plans two
  compiled variants over the query count (interleaved or not), and 64-bit
  addressing, which large query extents need, adds another; the storage a plan
  selects depends on its query and KV extents. Each plan still compiles only
  in `compile()`.
- **Launches** — conflict-rank plans chain their kernels with programmatic
  dependent launch. A records wave launches two setup kernels and its main
  kernel; a chain wave does the same, with one more scan for the first wave,
  and computes O.dO in its setup. Each execution adds an initialization, the
  last wave's setup, the fold and the `d_sink` reduction (records plans also
  launch a first-wave marking kernel and the O.dO kernel). The last grid of an
  execution completes only after the fold, so the next launch on the stream
  observes the final `dKV`. The 148-CTA wave matches the SM count of B200. The
  wave and setup-grid sizes are plan constants that do not depend on the
  device's SM count, and correctness relies only on the two CTAs of a cluster
  being co-resident.

Scratch (`scratch_workspace_bytes()`) of representative BF16 plans:

| H | D_qk | Q | KV | top-k | `'shards'` | `'records'` | `'chain'` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 576 | 128 | 32768 | 2048 | 2.250 GiB | 1.382 GiB | 1.003 GiB |
| 64 | 576 | 16384 | 32768 | 2048 | 9.008 GiB | 1.389 GiB | 1.011 GiB |
| 64 | 512 | 16384 | 32768 | 2048 | 8.008 GiB | 1.239 GiB | 0.963 GiB |
| 64 | 576 | 4096 | 32768 | 512 | 9.002 GiB | 0.526 GiB | 0.505 GiB |
| 64 | 576 | 4096 | 65536 | 2048 | 2.283 GiB | 1.387 GiB | 1.008 GiB |
| 64 | 576 | 16384 | 131072 | 2048 | 9.258 GiB | 1.540 GiB | 0.879 GiB |
| 64 | 576 | 16384 | 524288 | 2048 | 10.008 GiB | 1.859 GiB | 1.859 GiB (records) |
| 64 | 576 | 16384 | 1048576 | 2048 | 20.008 GiB | 3.035 GiB | 3.035 GiB (records) |

**Shard and relay layouts.** `deterministic_dkv_mode='shards'`, an explicit
relay size, and shapes whose conflict-rank workspace would be larger use the
bounded-wave shard layout. Ordinary extents select 32 dKV shards for
`total_S_q <= 128`, 64 for `129..256`, and 128 for larger query batches.
Queries run in same-stream waves with one CTA per shard; CTA lane `i` is the
sole writer of FP32 dKV shard `i` in each wave. H96/H128 split each query wave
into ordered M64 head-block launches, preserving single-writer ownership
without multiplying the number of shards. Kernel launch ordering serializes
both head blocks and shard reuse across waves. The 32-shard specialization
converts with one fixed-order reduction; the 64- and 128-shard
specializations first fold groups of eight and then combine those groups in
fixed order.

If the selected shard layout would exceed 16 GiB, the generic M64 plan
keeps each query wave at 128 CTAs and assigns one FP32 dKV plane to each
cluster. Queries below 8192 rows prefer sixteen eight-CTA clusters; larger
queries prefer thirty-two four-CTA clusters for additional reducer parallelism.
If that preferred layout's complete FP32-plane and packed-mask allocation would
exceed 32 GiB, the plan selects an eight-plane, sixteen-CTA memory-first relay.
The long-query policy deliberately skips cluster-8: GB200 sweeps at 393K, 512K,
and 768K KV found it slower than cluster-16 while retaining twice as many FP32
planes. A setup kernel builds packed 4-, 8-, 16-, 32-, or 64-bit rank masks for
every `(query wave, head block, KV row)`. Reducer threads wait only for lower cluster
ranks that actually update the same KV row, then publish that row with
GPU-scope release/acquire ordering. The participating reducer-thread CTA
barrier carries the release/acquire causality, so no additional GPU-wide fence
is needed. Rank zero skips the predecessor handoff, the last rank skips the
successor publication, and the final release needs no second rendezvous.
Disjoint sparse rows continue in parallel; the fixed rank order handles both
low- and high-overlap top-k rows without a global semaphore dependency on
non-resident CTAs. The selected 2, 4, 8, 16, or 32 planes are combined directly in
the fixed-order conversion kernel, without an intermediate FP32 prefold.
On SM100/SM103 under the default `'records'` mode, the automatic relay runs
only where records do not apply: top-k widths above 14169 (a records wave
counts its `148 * top-k` slots in a 21-bit field) and shapes whose records
would need more workspace than the relay.
Duplicate top-k indices within one query row remain outside the deterministic
contract of the shard and relay layouts. This condition is not checked at
runtime; callers that require bitwise repeatability must provide unique valid
indices within each query's active top-k prefix.

Callers that have profiled their exact SM100 workload may set
`deterministic_relay_cluster_size` to `4`, `8`, `16`, `32`, or `64` to select
that many CTAs per logical relay group instead of using the automatic policy.
The corresponding number of FP32 relay planes is `32`, `16`, `8`, `4`, or
`2`. Because SM100 hardware clusters contain at most 16 CTAs, the logical
32- and 64-CTA groups run as two and four same-stream 16-CTA phases,
respectively. Each kernel boundary completes the preceding physical phase
before the next ranks reuse the same plane, avoiding cross-cluster residency
dependencies. Cluster-64 uses a 64-bit packed row mask and is intended as an
explicit emergency memory-first choice; the automatic policy never selects
it because its lower per-phase parallelism regresses profiled workloads. This
is a plan-time specialization: it is part of the compile/cache key, fixes the workspace size,
and remains unchanged during `execute()` and CUDA Graph replay. The default
`None` preserves automatic selection. Requests on non-deterministic plans,
non-generic backends (including deterministic H96 D576 plans that keep the
H64 + H32 composition), shapes that do not require the large-context relay,
and `deterministic_dkv_mode='chain'` plans are rejected rather than silently
ignored. An explicit choice may exceed the
automatic policy's 32 GiB memory-first budget, so callers should always allocate
the exact value returned by `scratch_workspace_bytes()`.

For the shard layout, additional dKV workspace is
`selected_shards * K8 * D8 * sizeof(float)`, where
`K8 = round_up(total_S_kv, 8)` and `D8 = round_up(D, 8)`. The cluster relay
uses `relay_planes * (K8 + mask_rows) * D8 * sizeof(float)`, where
`relay_planes = 128 / relay_width`. The packed mask uses a 32-bit word for
relay widths up to 32 and a 64-bit word for relay width 64. With
`mask_word_bits` equal to 32 or 64, its FP32 storage rows are
`mask_rows = ceil(ceil(total_S_q / 128) * ceil(H / 64) *
ceil(K8 / (mask_word_bits / relay_width)) * (mask_word_bits / 32) / D8)`.
The conflict-rank layouts depend on the plane count, the wave and the top-k
width. One CTA per head reduces `d_sink`. Use `scratch_workspace_bytes()` as the
authoritative full scratch size; a larger reusable workspace remains valid.
Treat this as a reproducibility requirement rather than a performance-tuning
knob: keep the default `False` when bitwise run-to-run stability is not needed.

The measurements below compare the shard and relay layouts
(`deterministic_dkv_mode='shards'`) with the fixed 128-plane layout. The
H128/D576 two-CTA row measures its private-plane layout, which `'shards'`
plans keep; a default plan of that shape stores dKV in conflict-rank storage
(0.916 GiB of scratch on a 148-SM B200; the size follows the SM count).

On GB200 with BF16, `total_S_q=16384`, `top_k=2048`, `H=64`, `D_qk=576`, and
`D_v=512`, the memory-first cluster-16 route reduces full scratch from
288.008 GiB to 20.008 GiB at `total_S_kv=1048576`. The upstream allocation
does not fit on the 185.0 GiB test GPU. At `total_S_kv=524288`, where both
implementations fit, scratch falls from 144.008 GiB to 10.008 GiB and warmed
`execute()` latency changes from 99.973 ms to 98.999 ms.

At `total_S_kv=1048576` with CP64 (`total_S_q=16384` per launch), BF16,
top-k 2048, D576/Dv512, and contiguous inputs, the official upstream baseline
and this change require:

| Heads | Upstream scratch | This change scratch | Scratch reduction |
|---:|---:|---:|---:|
| 16 | 288.002 GiB | 20.002 GiB | 268.000 GiB (93.1%) |
| 32 | 288.004 GiB | 20.004 GiB | 268.000 GiB (93.1%) |
| 64 | 288.008 GiB | 20.008 GiB | 268.000 GiB (93.1%) |

The upstream 128-plane baseline does not fit in the test system's 185.0 GiB
usable memory, so its 1M-KV latency cannot be measured. The following direct
A/B covers every deterministic dispatch selected by the adaptive policy, plus
the H128/D576 two-CTA private-plane control. All rows use BF16, D576/Dv512,
top-k 2048, and contiguous inputs; H is 64 except for the H128 control.

| Dispatch | Q | KV | Upstream scratch | This change scratch | Scratch change | Upstream latency | This change latency | Latency change |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary 32-plane | 128 | 32768 | 2.250 GiB | 2.250 GiB | 0.0% | 1.737 ms | 1.737 ms | +0.01% |
| ordinary 64-plane | 256 | 32768 | 4.500 GiB | 4.500 GiB | 0.0% | 2.845 ms | 2.844 ms | -0.03% |
| ordinary 128-plane | 16384 | 32768 | 9.008 GiB | 9.008 GiB | 0.0% | 38.566 ms | 38.551 ms | -0.04% |
| cluster-8 / 16-plane | 4096 | 65536 | 18.002 GiB | 2.283 GiB | -87.3% | 16.967 ms | 23.618 ms | +39.2% |
| cluster-4 / 32-plane | 16384 | 131072 | 36.008 GiB | 9.258 GiB | -74.3% | 49.858 ms | 57.753 ms | +15.8% |
| cluster-16 / 8-plane | 16384 | 524288 | 144.008 GiB | 10.008 GiB | -93.1% | 99.973 ms | 98.999 ms | -1.0% |
| H128/D576 two-CTA | 16384 | 32768 | 6.766 GiB | 6.766 GiB | 0.0% | 58.482 ms | 58.651 ms | +0.3% |

Latency values are medians of three alternating-order A/B rounds, each with
three warm-ups and 20 timed, allocation-free `execute()` calls. The unchanged
ordinary and two-CTA routes stay within 0.3%. Cluster-8 and cluster-4 trade
39.2% and 15.8% latency, respectively, for substantial workspace reduction;
at 512K KV, cluster-16 is 1.0% faster than upstream while using 93.1% less
scratch.

The follow-up relay scheduling pass was measured against merged commit
`1dea4471a6e6c05e29d3bf281281fd156d345a84` on the same SM100 environment with
PyTorch 2.14.1+cu130, CuTe DSL 4.8.1, cuDNN 9.24.0, BF16 H64, top-k 2048,
five warm-ups, and 40 allocation-free calls. The automatic policy preserves
the conservative eight-CTA layout below Q=8K because an earlier cluster-4
crossover did not generalize across SM100 devices.

| Dqk | Q | KV | Merged latency | Optimized latency | Speedup | Merged scratch | Optimized scratch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 576 | 1024 | 65536 | 6.957 ms | 4.466 ms | 1.56x | 2.258 GiB | 2.258 GiB |
| 576 | 8192 | 131072 | 31.677 ms | 24.142 ms | 1.31x | 9.129 GiB | 9.129 GiB |
| 576 | 16384 | 262144 | 64.390 ms | 48.320 ms | 1.33x | 18.508 GiB | 18.508 GiB |
| 576 | 16384 | 524288 | 101.409 ms | 64.426 ms | 1.57x | 10.008 GiB | 10.008 GiB |
| 512 | 8192 | 131072 | 29.207 ms | 21.055 ms | 1.39x | 8.129 GiB | 8.129 GiB |
| 512 | 16384 | 262144 | 60.060 ms | 42.305 ms | 1.42x | 16.508 GiB | 16.508 GiB |
| 512 | 16384 | 524288 | 96.221 ms | 60.170 ms | 1.60x | 9.008 GiB | 9.008 GiB |

For H64, a three-round, 20-call, reversed-order A/B measured prepared-plan
eager execution at 104.579 ms and CUDA Graph replay at 104.225 ms (0.997x).
Capture completed with PyTorch synchronization-debug error mode enabled. The
cluster-16 regression also passed 100 repeated bitwise executions, graph
capture/replay with poisoned outputs and workspace, and changed device-side
`topk_length` values. Compilation and allocation remain plan-time operations;
capture uses caller-owned outputs and workspace, and replay requires the same
declared tensor shapes and storage addresses while their contents may change.

`SparseAttentionBackward.scratch_workspace_bytes()` reports the full scratch
requirement on every backend: the SM100 FP32 statistics and dKV-accumulator
planes, and on SM90 the FP32 dPsum and log2-scaled LSE vectors plus the dKV
accumulator. The ordinary SM90 dKV term is
`align128(round64(total_S_kv) * round32(D) * 4)` bytes; deterministic SM90 uses
the selected relay term above. Both add
`2 * align128(round64(total_S_q) * H * 4)` bytes of statistics. Pass a contiguous CUDA
`uint8` tensor of at least this size to `execute(..., workspace=workspace)` and
reuse it across calls; the launch sequence re-initializes the accumulators on
every execution (in-kernel on SM100, and with one stream-ordered memset for the
SM90 dKV accumulator and for `d_sink` on the routes that accumulate it
atomically). `execute(q, kv, out, dout, lse, attn_sink, topk_idxs, dq, dkv,
..., d_sink=...)` requires caller-provided `dq`, `dkv`, and the keyword-only
`d_sink` on every backend and never allocates or copies during execution.
Every SM100 plan compiles its selected ordinary, deterministic, two-CTA, or
adaptive-pair artifact in `compile()` from storage-free tensor metadata, so
its first `execute()` performs no CuTe JIT and may be captured directly. The
SM90 implementation retains its concrete-tensor compile caches and compiles on
its first execution. The high-level wrapper allocates the outputs and the
scratch on the launch stream when they are omitted. Inputs, outputs, and the
live workspace must not alias; concurrent executions need distinct output and
workspace storage.

- **Outputs** — tuple `(dq, dkv, d_sink)`
- **Constraints** — SM90 or Blackwell SM100/SM103; SM90 supports flat MQA tensors with `head_dim ∈ {512, 576}`

```python
result = DSA.sparse_attention_backward_wrapper(
    q, kv, out, dout, lse, attn_sink, topk_idxs,
    softmax_scale=1.0 / math.sqrt(D),
    topk_length=topk_length,
    deterministic=True,  # optional; SM90/SM100 H16/H32/H64/H96/H128
    deterministic_relay_cluster_size=None,  # optional: force 4, 8, 16, 32, or 64 CTAs/group
    deterministic_dkv_mode="records",  # optional: "chain" or "shards" (deterministic SM100/SM103 dKV storage)
    q_cluster_mode="off",  # optional: "adaptive_pair" for SM100 H16/H32 D576
)
dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
```

### 3. Indexer Forward (score-only)

Computes dense indexer scores:
``S[b, q, k] = sum_h ReLU(Q_h · K_h^T) · W_h`` with a ratio-causal mask.
For local query row `q_local`, valid KV columns satisfy
`k_local < clamp((q_causal_offsets[b] + q_local + 1) // ratio, 0, seqlen_k_b)`.
When `q_causal_offsets` is omitted, all offsets are zero.

`q_causal_offsets[b]` is the global uncompressed token index corresponding to
local `q[0]` for batch or THD segment `b`. It is not the packed storage offset
from `cu_seqlens_q`: `cu_seqlens_q` locates where a local Q segment is stored,
while `q_causal_offsets` locates that segment in the global causal timeline.
The K columns are assumed to be a compressed-KV prefix starting at global
compressed column 0.

- **Inputs**
  - `q`: `(B, S_q, H_q, D)` BF16, or the architecture-specific FP8 or FP4
    format described below (packed FP4 stores `D // 2` per row).
  - `k`: `(B, S_k, H_kv, D)` BF16, or the architecture-specific FP8 or FP4
    format described below.
  - `w`: `(B, S_q, H_q)` BF16 or FP32 with BF16 Q/K, on SM90 and SM100.
    FP32 head weights retain their precision through score generation, without
    changing the Q/K dtype; FP32 W requires unit stride in its last dimension.
    Unsupported strides are rejected instead of copied. The SM90 FP8 path also
    accepts FP32 when weights have already been pre-scaled by
    `q_scale * sm_scale`; SM100 MXFP8 and MXFP4 require BF16 weights.
  - `q_causal_offsets` (optional): CUDA INT32 tensor with one entry per
    batch/THD segment, on the same device as `q`.
  - `block_table` + `kv_lens` (optional, SM100 BF16): read `k` straight from
    a paged key cache instead of a dense tensor, with the block-table
    contract of the QSA compressed-key cache. `k` is the page pool
    `(num_pages, page_size, H_kv, D)` or `(num_pages, page_size, D)`;
    `page_size` is a multiple of 8 that divides 128 or is a multiple of it
    (16, 32, 64, 128, 256 all work). `block_table[b, i]` (`(B, max_pages)` INT32, unit column
    stride) is the pool row of sequence `b`'s `i`-th page and `kv_lens[b]`
    (`(B,)` INT32) its compressed-key count; the ratio-causal limit still
    applies on top. `cu_seqlens_k` must be `None` (THD queries keep
    `cu_seqlens_q`), and the score width is `max_seqlen_k`, which defaults to
    `max_pages * page_size`; keys at or past it are never scored, nor are pool
    rows past `kv_lens[b]` inside the last live page.
- **Output** — `scores`: `(B, S_q, S_k)` FP32. On SM100 this is the
  `[..., :S_k]` view over a `(B, S_q, ceil4(S_k))` (THD:
  `(total_q, ceil4(max_seqlen_k))`) allocation: the kernel binds `out` through
  a 16-byte-aligned layout, so the fp32 row stride must be a multiple of 4
  elements and at least `ceil4(S_k)`. When `S_k % 4 != 0` the returned tensor
  is therefore a strided view, not a fresh contiguous copy. A caller-provided
  `out` must be such a view (`ValueError` otherwise); nothing is staged or
  copied back. `IndexerForward` takes the contiguous padded
  `(B, S_q, S_k_padded)` buffer both at construction and in `execute()`, binds
  `out[..., :S_k]` directly, and never writes columns `[S_k, S_k_padded)`.
- **Strided inputs** — `IndexerForward.check_support()` declines `q`/`k`/`w`
  whose innermost stride is not 1 with `NotImplementedError` (the kernel
  addresses them natively; nothing is repacked), and `execute()` raises
  `ValueError` for such live tensors. `indexer_forward_wrapper` and
  `indexer_forward_top_k_wrapper` make strided inputs contiguous on the launch
  stream as a torch-op convenience, at the cost of one copy kernel per strided
  tensor.
- **Scope** — FP32 weights are supported by Indexer Forward and Combined
  Indexer Forward + Top-K. The separate indexer score-recompute and backward
  APIs still require BF16 weights.
- **Precision paths** — selected by `precision` on `indexer_forward_wrapper`
  and `indexer_forward_top_k_wrapper`; the `IndexerForward` class is
  BF16-only.
  - SM90 `precision="fp8"`: Q/K use E4M3 and `q_scale`/`k_scale` are FP32
    descales with one value per token/head. Set `return_lse=True` (or provide
    `lse_out`) to compute LSE in the same kernel invocation.
  - SM100 `precision="mxfp8"`: Q/K use E4M3 with block-scaled, packed E8M0
    scale tensors; `sf_vec_size` is currently fixed at 32 and
    `qhead_per_kv_head ∈ {4, 8, 16, 32, 64}` (the tile is always 128 packed
    rows -- the scale atom -- so a 4-head group packs 32 query tokens per
    tile; all-ones scale blobs make it a plain E4M3 x E4M3 scorer).
    THD inputs additionally require
    `cu_seqlens_q_scale_padded`/`cu_seqlens_k_scale_padded`: contiguous CUDA
    INT32 prefix tensors of shape `(B + 1,)` on the Q/K device. The interface
    deliberately performs no device-to-host copy to inspect their values.
    The caller guarantees that each device-side prefix starts at zero, is
    monotonic, covers the corresponding logical sequence, satisfies the
    packed-row alignment (each Q span times `qhead_per_kv_head` is a multiple
    of 128 rows; each K span is a multiple of 128 tokens), and stays within
    the packed scale storage.
  - SM100 `precision="mxfp4"`: Q/K are `torch.float4_e2m1fn_x2`, two E2M1
    values per byte with element `2i` in the low nibble, so their last dim is
    `D // 2` (64); a packed `uint8` buffer passes zero-copy through
    `.view(torch.float4_e2m1fn_x2)`. `q_scale`/`k_scale` are the MXFP8 E8M0
    scales unchanged: one per 32 head_dim values, so `D // 32` (4) per row
    although the last dim is `D // 2`, in the same packed layout, built
    from logical `(..., D // 32)` scales by the packers in
    `cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_utils`:
    `pack_q_scale_bshd`/`pack_k_scale_bshd` for BSHD, and
    `pack_q_scale_thd`/`pack_k_scale_thd` into a caller-provided `out` with
    prefixes from `make_scale_cu_seqlens_padded` for THD.
    `sf_vec_size` is fixed at 32, and THD inputs take the same
    `cu_seqlens_q_scale_padded`/`cu_seqlens_k_scale_padded` scale-prefix
    contract. Its kernel differs from the MXFP8 one only in the MMA:
    tcgen05 `kind::mxf4`, K = 64 per MMA. Requires a PyTorch build with
    `torch.float4_e2m1fn_x2`; otherwise `precision="mxfp4"` raises
    `NotImplementedError`.
- **Constraints** — `head_dim == 128`. The SM90 direct path supports
  `qhead_per_kv_head ∈ {16, 32, 64}` and currently requires `H_kv == 1`;
  SM100 BF16, MXFP8 and MXFP4 dense and combined Top-K paths support
  `qhead_per_kv_head ∈ {4, 8, 16, 32, 64}` at a fixed 128-row MMA tile
  (4 heads x 32 tokens, ..., 64 heads x 2 tokens), so `m_block_size` must be
  128. All currently
  require `H_kv == 1` (MQA). A query may be up to `ratio - 1` tokens longer
  than `ratio * seqlen_k` (its trailing tokens have not completed a compressed
  block and score the complete blocks only); a longer one is refused.

```python
result = DSA.indexer_forward_wrapper(
    q, k, w, ratio=4, q_causal_offsets=q_causal_offsets,
)
scores = result["scores"]
```

### 4. Combined Indexer Forward + Top-K

`DSA.indexer_forward_top_k_wrapper` provides a one-call alternative to the
two-call dense `indexer_forward_wrapper` + `indexer_top_k_wrapper` sequence
when compact score generation is desired. It returns aligned `indices` INT32
and `logits` FP32, plus `softmax` FP32 by default, without materializing dense
scores. Pass
`return_softmax=False` to omit `softmax`. BSHD output shape is
`(B, S_q, top_k)`; THD output shape is `(total_q, top_k)`. Padded slots are
`-1`/`-inf`. The selected set is Top-K, but its slot order is not guaranteed to
be descending. Set `deterministic=True` to break exact-value ties at the K-th
boundary toward the smallest local KV indices, making the selected set
reproducible across launches. This does not sort the output slots; the default
`False` path retains the faster scheduling-dependent tie-break.

The combined compressed path is SM100-only. Both BSHD and THD support
BF16 Q/K with BF16 or FP32 weights, and MXFP8 or MXFP4 Q/K with BF16 weights.
`topk_indices_global=True` is the default. Optional caller-owned
candidate/output/softmax/LSE buffers avoid per-call allocations; size the
candidate buffer with `compress_topk_cand_buffer_size` for BSHD or
`compress_topk_cand_buffer_size_thd` for THD. LSE is supported for BSHD and THD
with BF16, MXFP8, and MXFP4. Explicit microbatching is BF16 BSHD-only and cannot
be combined with LSE or explicit `q_causal_offsets`. BSHD with explicit
`q_causal_offsets` currently computes its per-batch candidate offsets eagerly
and is not CUDA-graph-capturable; THD capture requires the caller-provided
offsets and buffers returned by `compress_topk_cand_buffer_size_thd`. Compact
addressing requires `0 <= q_causal_offsets[b]`; rows extending beyond the KV
prefix are clamped to `seqlen_k_b`. THD MXFP8 and MXFP4 use the same
caller-guaranteed device-side scale-prefix contract described in §3; the
interface does not validate prefix values with a device-to-host copy.

Leave `microbatch_rows=-1` to use the BSHD long-sequence memory policy; it
enables windowing only for compatible BF16 shapes. Pass `0` to force one
launch, or a positive value only when deliberately controlling the scratch
footprint. Size the buffer with the launch's `microbatch_rows` and
`precision`: `compress_topk_cand_buffer_size` sizes MXFP8 and MXFP4 launches,
which never microbatch, for one full launch.

Paged K (`block_table` + `kv_lens`, the §3 contract) serves BSHD and THD
queries with BF16 Q/K in one launch and returns local key ids
(`topk_indices_global=False`), which is what serving engines consume before
mapping ids to physical cache rows. Caller-owned candidate buffers are sized
with `compress_topk_cand_buffer_size(..., paged=True)` for BSHD and
`compress_topk_cand_buffer_size_thd(cu_seqlens_q, None, ratio, q_causal_offsets)`
for THD: both return the single-launch size, and the THD call takes no key
prefix because a paged pool has none. No explicit microbatching and no MXFP8
yet.

Global versus local ids is primarily an interoperability and allocation
choice, not an Indexer Backward tuning choice. Keep the default global ids
when a downstream consumer requires flat KV ids. If every consumer accepts
local ids, prefer `topk_indices_global=False`. Avoiding the local-to-global
conversion can slightly improve wrapper performance, particularly for short
sequences. Local ids also avoid conversion temporaries and enable a
strictly zero-extra-allocation captured path when all documented buffers are
supplied. The global-id conversion is CUDA-graph-safe; during capture, its
temporaries are recorded in the graph's private pool. Include any downstream
conversion when comparing end-to-end performance.

```python
cand_floats = DSA.compress_topk_cand_buffer_size(
    B, S_q, S_k, ratio=4, return_lse=True,
    precision="bf16",  # must match the launch
)
cand = torch.empty(cand_floats, dtype=torch.float32, device=q.device)

result = DSA.indexer_forward_top_k_wrapper(
    q, k, w, top_k=512,
    ratio=4,
    cand_buffer=cand,
    return_lse=True,
)
topk_indices, topk_logits = result["indices"], result["logits"]
predict = result["softmax"]
lse = result["lse"]
```

### 5. Indexer Top-K

Radix top-K kernel for selecting candidate KV indices from indexer scores,
with variable per-row effective length.

- **Inputs**
  - `input_values`: `(n_rows, num_cols)` FP32/FP16/BF16
  - `seq_lens`: `(batch_size,)` INT32 (per-batch effective column count)
- **Outputs** — tuple `(indices, values)` (values is `None` when
  `return_val=False`). Use `return_val=False` when only the indices are
  consumed, so no values output buffer is allocated or written.
- **Tie policy** — `tie_break=0` (default) permits arbitrary cutoff ties;
  `1` prefers smaller source-column indices and `2` prefers larger indices.
  The selected set follows this policy. When `deterministic=False`, output
  order remains unspecified.
  Ties use equal radix keys: `+0` ranks before `-0`, as in the existing kernel;
  NaN ordering is unspecified. The policy applies only to this standalone API.
- **Deterministic mode** — `deterministic=True` makes the selected set and
  output slots repeatable. With the default `tie_break=0`, smaller source
  indices win a cutoff tie. The returned local indices are canonicalized in
  ascending order, carrying values with them. Selection and canonicalization
  are fused into this CuTe DSL kernel rather than launching a separate finalize
  kernel.
- **Cost** — nonzero tie policies scan source columns inside the selection
  kernel only when equal cutoff keys compete for the remaining slots. The
  scan stops once enough ties are found. Deterministic mode additionally sorts
  the selected indices in CTA shared memory. The compiled specialization
  chooses between an index-bitmap scan and an in-place bitonic sort according
  to the target architecture and shape; on SM100, `top_k < 256` uses bitonic
  ordering directly, and on SM103 bitonic ordering is used from 128K bucketed
  columns at `top_k <= 128` (`<= 512` with more than 148 rows). Neither path
  adds a kernel, device allocation, or host synchronization; the default
  specialization retains the existing selection path.
- **Constraints** — SM90+, `0 < top_k ≤ 2048`, nonzero row/column/batch
  extents, and `next_n > 0`. `seq_lens` is contiguous INT32 on the input CUDA
  device, and every entry must be in `[0, num_cols]`; the device-resident
  values are not copied to the host for validation. On SM120,
  `deterministic=True` supports at most 16,384 columns for FP32 and 32,768
  columns for FP16/BF16 because of the 99 KiB per-block shared-memory limit;
  `check_support()` rejects larger deterministic shapes before compilation.

```python
result = DSA.indexer_top_k_wrapper(
    scores.reshape(-1, scores.shape[-1]),
    seq_lens,
    top_k=512,
    deterministic=True,
)
indices, values = result["indices"], result["values"]
```

`IndexerTopK` owns no device memory. `execute(input_values, seq_lens,
out_indices, out_values=None, current_stream=None, workspace=None)` writes into
caller-owned outputs: `out_indices` is `(n_rows, top_k)` INT32 and `out_values`
is `(n_rows, top_k)` in the input dtype, both contiguous and on `input_values`'
device. `out_values` is required when the object was built with
`return_val=True` and must be `None` otherwise. The radix scratch is carved from
`workspace`, a CUDA `uint8` tensor of at least `scratch_workspace_bytes()` bytes
(`n_rows * (2 for FP32, else 1) * num_cols * 4`, rounded up to 128) that is
32-byte aligned; an absent or undersized workspace raises before launch.
The writable `out_indices`, optional `out_values`, and carved scratch byte
ranges must not overlap one another or the read-only inputs. Disjoint views of
the same backing allocation are valid. The class API also requires a 32-byte
aligned input base; the eager wrapper copies a contiguous storage-offset view
when needed to satisfy this alignment.
`compile()` builds the kernel from fake tensors and allocates nothing.
`indexer_top_k_wrapper` allocates the outputs and the workspace per call on the
launch stream. Rows the `next_n` stagger leaves empty are canonicalized to
`-1` indices and `-inf` values.

#### Compact vs. non-compact sparse indices

Providing `topk_length` declares a compact input layout for Sparse Attention
Backward and the sparse score-recompute kernels: in each row, the first
`topk_length[row]` slots must contain valid ids and every later slot must be
filled with `-1`. The padding value is required even when `topk_length` is
provided, because the SM100 attention score-recompute wrapper may select a
non-compact kernel that identifies invalid slots by their index values. Use
`compactify_wrapper` first when invalid entries are interspersed. A 3-D input
is returned flattened as `(B * S_q, topk)` with a `(B * S_q,)` length tensor;
reshape both when a BSHD score-recompute wrapper expects batched shapes. The
operation preserves the input id convention; it does not convert local ids to
global ids.

There is no universal faster representation. Compact execution can skip
sparse tiles when valid prefixes are substantially shorter than the physical
top-k width, while non-compact execution can benefit from static loop bounds
when rows are nearly full. The SM100 sparse attention score-recompute wrapper
uses separately tuned tile policies and can internally select its non-compact
path when that has better code generation. If conversion is required solely
for performance, benchmark the entire `compactify_wrapper` + consumer sequence
for the target length distribution. Indexer Backward has no `topk_length`
argument and always processes its fixed slot width; its score tensors and ids
must remain aligned slot-for-slot.

For BF16 workloads with causal sparse indices, the following are useful
starting points; the best choice still depends on the shape and valid-length
distribution:

- On SM100, prefer trying the non-compact path for Sparse Attention Score
  Recompute when optimizing this consumer in isolation; it can be slightly
  faster. For Sparse Indexer Score Recompute, preserve the producer's
  representation, since the performance difference is generally small.
- On SM90, prefer compact inputs for both sparse score-recompute kernels when
  the valid-prefix length is already available; compact execution can be
  slightly faster.
- For Sparse Attention Backward on either architecture, prefer compact inputs
  when the valid-prefix length is already available. The benefit may be small
  when most rows are full.

Retain local ids when the producer and every downstream consumer accept them.
If `compactify_wrapper` or an id conversion must be added, time that operation
together with the consumer before choosing a representation.

### 6. Sparse Indexer Score Recompute

Computes softmax over top-K entries of the indexer score:
``predict[b, q, i] = softmax_i(sum_h ReLU(Q_h · K_{topk[i]}^T) · W_h)``.

- **Inputs**: `q_indexer`, `k_indexer`, `weights`, `topk_indices`
  (optional `topk_length`). `topk_indices` are per-batch local KV ids by
  default; pass `topk_indices_global=True` when using ids encoded as
  `batch_idx * S_k + local_idx`.
- **Output** — `predict`: `(B, S_q, topk)` FP32.
- **Layout contract (class vs wrapper)** — `SparseIndexerScoreRecompute.execute()`
  takes `out` as a required, contiguous FP32 tensor matching `sample_out`;
  `q_indexer` / `k_indexer` / `weights` need a unit innermost stride and
  `topk_indices` / `topk_length` must be contiguous int32. `check_support()`
  declines anything else with `NotImplementedError` naming the tensor and its
  strides, and `execute()` re-validates the live tensors with `ValueError`;
  nothing is copied, cast, or allocated on the execute path. A prepared SM90
  plan also rejects an `out` whose byte range overlaps any input. The wrapper
  keeps the convenience: it copies strided inputs contiguous, casts the index
  tensors to int32, allocates `out` when omitted, and stages an overlapping
  SM90 output before copying it back, all on `stream`.
- **Prepared SM90 plans** — `compile()` builds the CuTeDSL artifact from symbolic
  batch and sequence extents. Exact `B`, `S_q`, `S_k`, runtime scale, pointers,
  and stream do not specialize the artifact. Physical top-K widths in
  `[1, 2176]` use one of 17 bounded 128-slot capacity buckets. `execute()` reads
  the public `(B, S_q, H)` weights layout directly, performs no layout copy or
  allocation, and is CUDA Graph capturable with caller-owned `out`. The kernel
  has a fixed reduction tree and one writer per output element, so identical
  inputs and one compiled plan produce bitwise-identical output bytes.

### 7. Sparse Attn Score Recompute

L1-normalised head-summed softmax over top-K entries:
``target[b, q, i] = sum_h exp(Q_h · K_{topk[i]}^T · scale - LSE_h) / Z``.

- **Inputs**: `q_attn`, `k_attn`, `lse`, `topk_indices`, `softmax_scale`
  (optional `topk_length`). `topk_indices` are per-batch local KV ids by
  default; pass `topk_indices_global=True` when using ids encoded as
  `batch_idx * S_k + local_idx`.
- **Output** — `target`: `(B, S_q, topk)` FP32.
- Note: the wrapper handles the `-log2(e) * lse` preprocessing internally.
- **Layout contract** — as in §6: the class requires a contiguous `out` at
  `execute()`, declines a non-unit innermost stride on `q_attn` / `k_attn` /
  `lse` and non-contiguous or non-int32 `topk_indices` / `topk_length` in
  `check_support()`, and never copies; the wrapper keeps the convenience copies
  and allocates `out` when omitted. On SM100 a plan built without `topk_length` compiles the
  `mTopkLength` operand out entirely (no placeholder tensor).
- **Prepared SM90 plans** — the same symbolic compile and bounded Top-K buckets
  as §6 apply. `softmax_scale` is a runtime scalar rather than a compile key,
  and the kernel reads `(B, S_q, H)` LSE directly. A warmed class plan with its
  caller-owned, disjoint `out` is allocation-free, CUDA Graph capturable on an
  explicit non-default stream, and bitwise deterministic for identical inputs.

### 8. Dense Indexer / Dense Attn Score Recompute

Full-KV (no top-K) analogues of §6 and §7. Each returns `{'out', 'denom'}`.
They apply the same ratio-causal mask as Indexer Forward; masked positions are
written as `-inf` and excluded from `denom`. Pass the same `q_causal_offsets` to
all dense score tensors that feed the same loss path.

On SM100, Indexer Forward and Dense Indexer Score Recompute use the same
unified kernel implementation: forward runs it with `compute_lse=False` and
passes `denom=None`, so the LSE slot is compiled out (no placeholder buffer),
while dense indexer score recompute runs it with `compute_lse=True` and a live
`denom`. The shared implementation lives in `score_recompute`;
`indexer_forward` only imports it.
Dense Attention Score Recompute has separate BF16 and MXFP8 kernels because its
score and normalization semantics differ from the indexer path; the MXFP8 one
shares the indexer kernels' warp-specialized scaffolding and block-scale
plumbing.

- **Precision** — on SM100, `dense_indexer_score_recompute_wrapper` takes
  `precision="bf16"`, `"mxfp8"`, or `"mxfp4"` with the Q/K and scale
  contract of Indexer Forward (§3); its block-scaled precisions require
  `qhead_per_kv_head ∈ {32, 64}`. `dense_attn_score_recompute_wrapper` takes
  `"bf16"` or `"mxfp8"` and rejects `"mxfp4"`. The
  `DenseIndexerScoreRecompute` and `DenseAttnScoreRecompute` classes, and
  SM90, are BF16-only.
- **Layouts** — BSHD on SM90 and SM100+. THD packed (`cu_seqlens_q`/`cu_seqlens_k`
  with `q`: `(total_q, H_q, D)`, `out`: `(total_q, max_seqlen_k)`) on SM100+ only:
  SM90 declines THD in `check_support()` with `NotImplementedError` because its
  kernel is BSHD-native and serving THD would need a host copy of `cu_seqlens`
  per call.
- **THD launch envelope** — `DenseIndexerScoreRecompute` / `DenseAttnScoreRecompute`
  take `max_seqlen_q` and `max_seqlen_k` as plan-time ints in `__init__`:
  `max_seqlen_q` is required and `max_seqlen_k` defaults to (and must equal)
  `sample_out.shape[1]`. `execute()` never derives them from `cu_seqlens`, and
  caller-passed values must match the plan's. The wrappers derive them from
  `cu_seqlens` when omitted, at the cost of one blocking device-to-host read per
  call (not CUDA-graph capturable); pass both.
- **Layout contract (class vs wrapper)** — `execute()` takes `out` and
  `denom_out` as required, contiguous FP32 tensors matching the samples;
  `q` / `k` / `weights` (or `lse`) need a unit innermost stride, and
  `cu_seqlens_q` / `cu_seqlens_k` / `q_causal_offsets` must be contiguous 1-D
  int32. `check_support()` declines strided samples with `NotImplementedError`
  naming the tensor and its strides; `execute()` re-validates the live tensors
  with `ValueError`. Nothing is copied, cast, or allocated on the execute path
  (the SM90 kernel's per-head `(B,S,H)->(B,H,S)` operand transpose is the one
  remaining copy, tracked as a deferred kernel change). The wrappers keep the
  convenience: strided inputs are copied contiguous, `cu_seqlens_*` are cast to
  int32, and `out` / `denom_out` are allocated when omitted, all on `stream`.

### 9. Indexer Backward

Three-stage sparse top-K pipeline that produces the training gradients for the
indexer tower:

1. `ScoreGradSm90` / `ScoreGradSm100` (kernel 1) — in-place score-grad precompute
   that overwrites `attn_score` (target) and reads `index_score` (predict)
   without modifying it.
2. `IndexerBackwardSm90` / `IndexerBackwardSm100` (kernel 2) — three
   warp-specialised GEMMs produce `d_index_q`, `d_weights`, and a
   `dIndexK_f32` accumulator.
3. In ordinary execution with a BF16 `d_index_k`, a pure-torch copy (kernel 3)
   casts `dIndexK_f32` to BF16; an fp32 `d_index_k` is the accumulator itself
   and needs no kernel 3. On SM100-family GPUs that private BF16-path accumulator
   is column-permuted, so kernel 2 can issue 16-byte reductions straight from
   registers without SMEM staging; the same copy restores the natural order.
   Deterministic execution instead uses the fixed-order reduction below to
   write the requested output dtype directly.

`IndexerBackward` / `indexer_backward_wrapper` run on SM90 and on SM100/SM103/SM107
(SM107 needs a CuTe DSL with the `sm_107a` target).
`check_support` admits `D == 128`, `block_I == 128`, a positive top-k that is a
multiple of `block_I` (callers with other logical extents pad the top-k tensors
with `-1` indices and zero scores), and a finite `sm_scale > 0`; SM100/SM103/SM107
additionally require `H in {32, 64}` and SM90 requires `H == 64`. `d_index_q`
and `d_weights` are BF16; `d_index_k` is BF16 or fp32.

With `deterministic=True` on SM90, SM100, SM103 or SM107, kernel 2 uses fixed
producer-plane ownership and accumulates dK into a CTA-private FP32 workspace. The compile-time
`deterministic_dk_planes` specialization selects 96, 128, or 148 producer
planes (default 96). One CuTe DSL reduction kernel folds the planes in fixed
CTA order, converting only once to the requested `d_index_k` dtype. The dQ
and dW stores remain single-writer.

The plan owns no device memory. For non-deterministic execution, an fp32
`d_index_k` is the accumulator itself (zeroed on the launch stream inside
`execute`: fused into the score-grad kernel on SM100, a memset on SM90; the
BF16 scratch is cleared the same way, so callers never pre-zero); a BF16 `d_index_k` accumulates into a `B * S_k * D` fp32 view carved
from the caller's workspace. `IndexerBackward.scratch_workspace_bytes()` reports
that ordinary scratch size (0 for fp32 `d_index_k`).
`execute(..., workspace=)` takes the uint8 buffer on `index_q.device`, and a
missing or undersized buffer raises `ValueError` before `attn_score` is touched.
`indexer_backward_wrapper` allocates it per call on the launch stream and accepts
the same optional `workspace=` keyword for reuse; executions that share one
buffer must not overlap on the device.

`deterministic=True` is request-or-fail and supports SM90 with `H == 64` and
SM100/SM103/SM107 with `H in {32, 64}`. All supported devices require `D == 128`,
`block_I == 128`, and `topk % 128 == 0` for
`128 <= topk <= 2048`. Each producer CTA owns one plane. SM100-family kernels
process fixed grid-stride row stripes; SM90 submits bounded waves of at most
`deterministic_dk_planes` one-row CTAs on the same stream, so a later wave cannot
reuse a plane until its predecessor completes. The main GEMM therefore retains
its existing Tensor Core math while dK has a fixed association order across
launches. Scratch is exactly
`ws_align(deterministic_dk_planes * B * S_k * D * sizeof(float))`. It is cleared on the launch stream at every execute, then
reduced once in fixed plane order. The live scratch byte range must not overlap
any input or output; `execute` validates this before clearing the planes, while
allowing disjoint views of the same allocation. A warmed plan with caller-owned
outputs and workspace is CUDA Graph compatible. To capture the convenience
wrapper, warm it first and pass every output plus `workspace=` explicitly;
omitted buffers are allocated per call. A fixed plane count is bitwise
reproducible across runs; changing the plane count changes the FP32
accumulation grouping, so results from two plane counts are not required to be
bitwise identical. As with the existing deterministic sparse-attention
backward, valid IDs in a semantic top-k row are expected to be distinct.

**The TileLang fallback present in the upstream repo is dropped here
(CuTe-DSL only).** If the CuTe-DSL path fails the wrapper raises
`RuntimeError` rather than silently falling back.

```python
grad_loss = torch.ones((), dtype=torch.float32, device=index_q.device)
result = DSA.indexer_backward_wrapper(
    index_q, weights, index_k,
    attn_score, index_score, topk_indices,
    grad_loss=grad_loss, sm_scale=1.0, loss_coeff=1.0, block_I=128,
    deterministic=True, deterministic_dk_planes=128,
)
d_index_q, d_weights, d_index_k = (
    result["d_index_q"], result["d_weights"], result["d_index_k"],
)
```

When compressed forward returns its fused `softmax`, backward can skip both
indexer Q@K score recompute and the separate logits softmax. Pass `softmax`
directly as `index_score`; backward treats this buffer as read-only.
`attn_score` must use the same valid-slot mask. Because compressed forward
returns global indices by default, also pass `topk_indices_global=True` unless
forward used `topk_indices_global=False`. The public sparse
`indexer_backward_wrapper` has a
BSHD-shaped interface; BF16 THD tensors can use zero-copy `B=1` views (squeeze
the singleton K head and add a batch dimension) together with global Top-K
indices. FP8, MXFP8, and MXFP4 indexer backward are not currently supported
because the backward wrapper requires BF16 Q/K/W inputs.

On SM100, `topk_indices_global` describes the input
encoding; it is not a performance-tuning switch. Local ids are range-checked and converted to flat ids in registers before the
same optimized gather path. Do not add a separate `local_to_global_wrapper`
launch only for Indexer Backward; preserve the producer's id convention and
set `topk_indices_global` to match it.

On SM100, the kernel accepts compact H32 `index_q`, `weights`,
`d_index_q`, and `d_weights` tensors directly. TCGEN05 still executes its
minimum M64 tile internally: the upper Q rows and weights are zero-filled, and
the dQ/dW stores are clipped to the logical H32 extent. Callers must not pad
these tensors to H64 or slice H64 outputs. This compact path also applies to
`deterministic=True`: fixed producer ownership changes only the dK destination
and row schedule, while the same H32 input padding and output predicates remain
active. Other sub-M64 head counts remain unsupported; H64 and larger head
counts retain their existing behavior.

On SM90, also prefer preserving the producer's id convention. The performance
difference between local and global ids is generally small, so a separate
conversion solely for Indexer Backward is unlikely to help. Deterministic SM90
uses the same fixed producer-plane ownership and fixed-order reducer described
above. Its stream-ordered bounded waves also form the synchronization boundary:
every one-row CTA in wave N completes before wave N+1 can reuse the same producer
planes, and the fixed reducer runs only after the last wave.

### 10. Dense Indexer Backward

Full-KV counterpart to Indexer Backward. It consumes raw dense score tensors
and denominators produced by Dense Indexer / Dense Attn Score Recompute.

- **Inputs**
  - `index_q`: `(B, S_q, H, D)` BF16
  - `weights`: `(B, S_q, H)` BF16
  - `index_k`: `(B, S_k, D)` BF16
  - `attn_score`, `index_score`: `(B, S_q, S_k)` FP32 raw dense scores
  - `attn_l1norm`, `index_lse`: `(B, S_q)` FP32 denominators
  - `q_causal_offsets` (optional): same offsets used for the corresponding
    Dense Indexer / Dense Attn Score Recompute outputs.
- **Outputs** — `d_index_q`, `d_weights`, `d_index_k`
- **Constraints** — SM90, SM100, SM103 or SM107, `H == 64`, `ratio >= 1`;
  SM100-family GPUs accept `D in {64, 128}` and `block_I in {64, 128}`, SM90
  `D == 128` and `block_I == 128`
- **Scratch** — the dK reduction targets a pre-zeroed fp32 buffer: an fp32
  `d_index_k` in place, a BF16 `d_index_k` through an fp32 accumulator carved
  from the caller's workspace (`DenseIndexerBackward.scratch_workspace_bytes()`
  bytes on `index_q.device`, 0 for fp32 `d_index_k`; `execute(...,
  workspace=)`). `dense_indexer_backward_wrapper` allocates it per call on the
  launch stream unless `workspace=` is passed; calls sharing one buffer must
  not overlap.

```python
grad_loss = torch.ones((), dtype=torch.float32, device=index_q.device)
dense_index = DSA.dense_indexer_score_recompute_wrapper(
    index_q, index_k.unsqueeze(2), weights,
    q_causal_offsets=q_causal_offsets,
)
dense_attn = DSA.dense_attn_score_recompute_wrapper(
    attn_q, attn_k, lse, softmax_scale,
    q_causal_offsets=q_causal_offsets,
)

result = DSA.dense_indexer_backward_wrapper(
    index_q, weights, index_k,
    dense_attn["out"], dense_attn["denom"],
    dense_index["out"], dense_index["denom"],
    grad_loss=grad_loss, sm_scale=1.0, loss_coeff=1.0, block_I=128, ratio=1,
    q_causal_offsets=q_causal_offsets,
)
```

---

### 11. Pooled DSA Training Backward

The pooled variant of the indexer scores pools of ``pool_size`` consecutive keys instead of single keys. Each pool key is the
softmax-weighted (``gates + ape``) mean of its tokens; the top-``K`` pools are selected per query, and the attention itself runs
on the tokens of the selected pools plus the incomplete tail of tokens that precede the query position. The training APIs live
in ``cudnn.deepseek_sparse_attention`` (lazily exported) and compose with the token-level kernels above.

| API | Computes | Returns |
|---|---|---|
| ``pooled_keys_forward(keys, gates, ape, pool_size)`` | pooled keys ``(B, S_k // P, D)`` from bf16 ``keys (B, S_k, D)``, bf16 ``gates (B, S_k)``, fp32 ``ape (P, D)``; ``D`` is 128 | ``{'pooled'}`` |
| ``unpool_backward(keys, gates, ape, d_pooled, pool_size)`` | gradients of ``pooled_keys_forward`` | ``{'d_keys', 'd_gates', 'd_ape'}`` |
| ``pooled_indexer_select(index_q, pooled, weights, k_pools, pool_size, sm_scale)`` | top-``k_pools`` pools per query under the ratio-causal rule and their softmax | ``{'pool_ids' (B, S, K) INT32, 'predict' (B, S, K) FP32, 'logits'}`` |
| ``pooled_expand_compact(pool_ids, positions, pool_size, seqlen_k)`` | compact token expansion: valid pools' tokens, then the tail, then ``-1`` | ``{'token_ids' (B, S, W), 'topk_length' (B, S)}`` with ``W = K*P + P - 1`` rounded up to a multiple of 64 |
| ``pooled_sparse_attention_backward(q, kv, out, dout, lse, pool_ids, positions, pool_size)`` | sparse attention backward over the expanded tokens | ``{'dq', 'dkv', 'd_sink', 'token_ids', 'topk_length'}`` |
| ``pooled_sparse_attention_backward_with_teacher(..., k_pools, ...)`` | the same backward, plus the pool-level teacher summed inside the kernel | ``{'dq', 'dkv', 'd_sink', 'target' (B, S, K) FP32, 'token_ids', 'topk_length'}`` |
| ``pooled_teacher_recompute(q_attn, k_attn, lse, token_ids, topk_length, k_pools, pool_size)`` | pool-level teacher from the sparse attention score recompute | ``{'target' (B, S, K)}`` |
| ``pooled_indexer_target(token_target, pool_ids, pool_size)`` | pool-level teacher from a token-level teacher (tail dropped, renormalized) | ``{'target'}`` |
| ``pooled_indexer_backward(index_q, weights, keys, gates, ape, pool_ids, target, grad_loss, pool_size, sm_scale, loss_coeff, pooled, predict)`` | indexer-loss backward on pooled keys through the unpooling | ``{'d_index_q', 'd_weights', 'd_keys', 'd_gates', 'd_ape', 'd_pooled'}`` |

Layouts: ``q``/``out``/``dout`` are ``(B, S, H, D)``, ``kv`` is ``(B, S_k, D)``, ``lse`` is ``(B, S, H)`` FP32, ``pool_ids`` are local pool
indices with ``-1`` for unselected slots (expected last), ``positions`` ``(B, S)`` are the query positions. The teacher ``target`` is the
attention probability mass of each selected pool summed over heads and renormalized over the valid pools of the query; tail tokens
are excluded. ``pooled_indexer_backward`` consumes ``target`` in place and recomputes ``predict`` when it is not supplied.

The underlying kernels take two options used by this composition: ``sparse_attn_score_recompute_wrapper(..., pool_size=P)`` emits the
per-pool sums directly, and ``SparseAttentionBackward`` on SM100 accepts ``pool_target``/``pool_size`` to emit them from the backward's
own probabilities, accumulating fp32 partials across the attention head blocks (bit-reproducible up to two blocks, i.e. 128 heads);
the option is not available with ``deterministic=True``. Supported on SM100-class GPUs with the generic single-CTA backward backend; ``pool_size`` 4 and 16 are supported.

## Limitations

- **CuTe DSL requirement** — install `nvidia-cutlass-dsl[cu13]>=4.8.0`.
  SM90 forward requires the loaded CUDA 13 DSL compiler flavor and rejects a
  CUDA 12 compiler before JIT; the package version alone is not sufficient.
  Sparse Attention Forward emits the few SM100 instructions that are not yet
  exposed by the public CuTe facade through the MLIR LLVM inline-assembly
  interface available in the supported CuTe DSL builds.
- **Architecture support** — Sparse Attention Forward supports SM90 and the
  mapped SM100-family capabilities 10.0, 10.3, and 10.7.
  Sparse Attention Backward, Score Recompute, Indexer Forward, Indexer Top-K,
  and Indexer Backward support SM90 and SM100. Dense score recompute in the THD
  (`cu_seqlens`) layout is SM100+ only; SM90 serves BSHD. The combined
  compressed-logits + Top-K forward is SM100-only; the standalone Indexer Top-K
  remains SM90+.
- **Forward scope** — Prefill only; no FP8 cache, decode, or split-KV forward
  path. SM90 uses one CTA per query and 64-head tile, padding the last head tile
  internally when H is not a multiple of 64.
- **Sparse gather path** — SM90 uses predicated 16-byte `cp.async` gathers,
  matching the Hopper C++ kernel. On SM100, the public CuTe facade does not yet
  expose the TMA `gather4` operation, so the forward uses a feature-local LLVM
  inline-assembly bridge with the compiler-owned descriptor; the issued data
  movement remains hardware TMA `gather4`.
- **Indexer Forward only supports `head_dim = 128`**. SM90 supports
  `qhead_per_kv_head ∈ {16, 32, 64}` with `H_kv = 1`; SM100 BF16, MXFP8, and
  MXFP4 support `qhead_per_kv_head ∈ {4, 8, 16, 32, 64}`. Both the dense and
  combined Top-K paths require `H_kv = 1`. MXFP4 runs on the SM100-family
  capabilities 10.0, 10.3, and 10.7 (no SM90 path).
- **Standalone Top-K only up to 2048**; `top_k > 2048` is not supported by
  its radix Top-K kernel. The combined compressed path uses a separate stage-2
  implementation.
- **Compressed-path limits** — the stage-1 compact score kernel is MQA-only
  (`H_kv = 1`); every precision keeps its 128-row tile for every head
  group; explicit microbatching cannot be combined with MXFP8, MXFP4, LSE, or
  explicit per-batch causal offsets.


## Native PyTorch training composition

`DSA.sparse_attention` composes the existing native forward and backward
wrappers, without FlashMLA. It accepts contiguous BF16 `q[S_q,H,D]`,
`kv[S_kv,D]`, INT32 `indices[S_q,K]`, optional FP32 `attn_sink[H]`, and
optional INT32 `topk_length[S_q]`, on one SM100 device. Supported pairs are
H16/H32/H64 with D512/D576 and H128 with D512; output V dimension is 512.
H16/H32 forward explicitly pads heads to 64 and slices the result. This cost
belongs to the semantic adapter and is included when benchmarking the call.
Physical K and sequence extents must be positive. K is padded to 128 for launch.

```python
from cudnn import DSA
result = DSA.sparse_attention(q, kv, indices, attn_sink, topk_length=lengths)
result["out"].float().square().mean().backward()
score = DSA.sparse_attention_score_recompute(
    q.detach(), kv.detach(), result["lse"], indices, topk_length=lengths,
)
```

Only `out` is differentiable (Q, KV and sink, first order). `lse` is KV-only
natural-log LSE, excluding sink; `max_logits` and LSE are non-differentiable.
No sink means an effective all-`-inf` sink vector. Launches and adapter tensor
operations use the current PyTorch stream. First use compiles the kernels;
warm before CUDA graph capture. This allocating convenience composition is
not a new allocation-free APIBase `execute` implementation.

Safe mode maps all out-of-range indices to -1, bounds lengths to [0,K], masks
inactive suffixes, and compacts valid entries when lengths are supplied. Both
passes use the same normalized metadata. Duplicate indices remain distinct
slots. `trusted_compact_metadata=True` is an explicit producer contract:
bounded valid active prefixes, negative inactive suffixes, bounded lengths;
without lengths every nonnegative index is in range. It skips normalization,
not launch padding. Violating that contract is unsafe.

Score recompute preserves original slots, including holes, and returns both
`target` and effective `indices`. It sums per-head probabilities, masks invalid
slots, then normalizes across the retained slots; all-masked rows return zeros.
Targets are detached. Indexer training needs its own loss; discrete Top-K
selection is not differentiated. This contract uses the existing native `out`
return key, not the old unmerged bridge's `output` key.
