# DeepSeek Sparse Attention (DSA)

**This is an experimental API and subject to change.**

Torch-facing calls accept the ordinary default stream or a concrete
`torch.cuda.Stream` / raw stream handle. The special `CUstream(2)` per-thread
default sentinel is rejected by Torch interop: it cannot safely represent an
allocator stream identity across host threads. Use a concrete stream instead.

## Overview

The DeepSeek Sparse Attention (DSA) module integrates a set of CuTe-DSL
kernels that support the sparse-attention path used by DeepSeek-style models.
Most kernels target Hopper (SM90) and Blackwell (SM100+) GPUs. Sparse-attention
forward and the combined compressed-logits + Top-K path are SM100-only;
sparse-attention forward is limited to the mapped SM100-family capabilities
10.0, 10.3, and 10.7. Dense Indexer Forward and standalone Indexer Top-K
support both architectures. The kernels are delivered as Python classes /
wrappers that follow the same `APIBase` pattern as other cuDNN Frontend
operations.

**Scope:** this module ships CuTe-DSL kernels for sparse-attention forward and
backward, indexer scores/top-K, sparse/dense score recompute, and sparse/dense
indexer backward. Sparse forward covers regular H64 and small-top-k Prefill
H128; it does not include the H128 decode/split-KV mode.

The module packages the following operations:

1. **Sparse Attention Forward** – sparse Prefill MQA for the supported SM100
   H64/H128 shapes.
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
    nonnegative value and is padded internally to a multiple of 64
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
  - H64: `D_qk in {512, 576}`, `indexer_topk in {0, 512, 1024, 2048}`
  - H128 small-top-k Prefill: `D_qk=512`,
    `indexer_topk in {0, 512, 1024}`
- **Indexer prefix constraint** — `indexer_topk <= K`; when `topk_length` is
  supplied and indexer LSE is enabled, every row must have
  `topk_length >= indexer_topk`, matching the tile-boundary snapshot
  contract.
- **Empty rows** — `out=0`, `max_logits=-inf`, `lse=+inf`; an enabled
  `lse_indexer` is also `+inf`.
- **Compilation lifecycle** — `SparseAttentionForward.compile()` validates
  and primes the API object, while the concrete CuTe kernel is JIT-compiled
  on its first `execute()`/wrapper call because padding and live DLPack
  layouts are runtime properties.
- **Layout and stream** — non-contiguous or under-aligned Q/KV inputs are
  materialized on the selected stream. An explicit `stream` must be a
  `cuda.bindings.driver.CUstream` belonging to `q.device`.

For `D_qk=576`, QK uses all 576 dimensions while V/output uses only the first
512 KV dimensions. `attn_sink=None` is equivalent to an all-`-inf` sink.

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

On SM100/SM103, `deterministic=True` retains the two-CTA Tensor Core kernels
for the supported BF16 H128/D512 and H128/D576 envelopes above. Queries run in
same-stream waves of at most 74 query tokens, with one two-CTA pair per query.
Private FP32 accumulator slots and ordered duplicate-index writers give each
update a fixed accumulation order. Stream ordering serializes slot reuse
between waves, and fixed-order reductions produce `dKV` and `d_sink`. Both
specializations keep the public workspace capacity of 128 FP32 dKV planes.

Other deterministic envelopes with H16/H32/H64/H96/H128 use the bounded-wave
M64 implementation. Ordinary extents select 32 dKV shards for
`total_S_q <= 128`, 64 for `129..256`, and 128 for larger query batches.
Queries run in same-stream waves with one CTA per shard; CTA lane `i` is the
sole writer of FP32 dKV shard `i` in each wave. H16/H32 use a masked M64 head
tile. H96/H128 split each query wave into ordered M64 head-block launches,
preserving single-writer ownership without multiplying the number of shards.
Kernel launch ordering serializes both head blocks and shard reuse across
waves. The 32-shard specialization converts with one fixed-order reduction;
the 64- and 128-shard specializations first fold groups of eight and then
combine those groups in fixed order.

If the selected bounded-wave layout would exceed 16 GiB, the generic M64 plan
keeps each query wave at 128 CTAs and assigns one FP32 dKV plane to each
cluster. Queries below 8192 rows prefer sixteen eight-CTA clusters; larger
queries prefer thirty-two four-CTA clusters for additional reducer parallelism.
If that preferred layout's complete FP32-plane and packed-mask allocation would
exceed 32 GiB, the plan selects an eight-plane, sixteen-CTA memory-first relay.
The long-query policy deliberately skips cluster-8: GB200 sweeps at 393K, 512K,
and 768K KV found it slower than cluster-16 while retaining twice as many FP32
planes. A setup kernel builds packed 16-bit, 8-bit, or 4-bit rank masks for every
`(query wave, head block, KV row)`. Reducer threads wait only for lower cluster
ranks that actually update the same KV row, then publish that row with
GPU-scope release/acquire ordering. The participating reducer-thread CTA
barrier carries the release/acquire causality, so no additional GPU-wide fence
is needed. Rank zero skips the predecessor handoff, the last rank skips the
successor publication, and the final release needs no second rendezvous.
Disjoint sparse rows continue in parallel; the fixed rank order handles both
low- and high-overlap top-k rows without a global semaphore dependency on
non-resident CTAs. The selected 8, 16, or 32 planes are combined directly in
the fixed-order conversion kernel, without an intermediate FP32 prefold.
Duplicate top-k indices within one query row remain outside the deterministic
contract, as in the bounded-wave path. This condition is not checked at
runtime; callers that require bitwise repeatability must provide unique valid
indices within each query's active top-k prefix.

For the ordinary policy, additional dKV workspace is
`selected_shards * K8 * D8 * sizeof(float)`, where
`K8 = round_up(total_S_kv, 8)` and `D8 = round_up(D, 8)`. The cluster relay
uses `relay_planes * (K8 + mask_rows) * D8 * sizeof(float)`, where
`relay_planes` is 8, 16, or 32, `relay_width` is 16, 8, or 4, and
`mask_rows = ceil(ceil(total_S_q / 128) * ceil(H / 64) *
(K8 / (32 / relay_width)) / D8)`.
One CTA per head reduces `d_sink`. Use `scratch_workspace_bytes()` as the
authoritative full scratch size; a larger reusable workspace remains valid.
Treat this as a reproducibility requirement rather than a performance-tuning
knob: keep the default `False` when bitwise run-to-run stability is not needed.

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
the unchanged H128/D576 two-CTA control. All rows use BF16, D576/Dv512,
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
accumulator (`2 * align128(round64(total_S_q) * H * 4) +
align128(round64(total_S_kv) * round32(D) * 4)` bytes). Pass a contiguous CUDA
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
    deterministic=True,  # optional; SM100 H16/H32/H64/H96/H128
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
  SM100 BF16 dense and combined Top-K paths support
  `qhead_per_kv_head ∈ {4, 8, 16, 32, 64}` (a group packs up to 8 query
  tokens per MMA tile: 4 heads x 8 tokens = 32 columns, ..., 64 heads x 2
  tokens = 128), their MXFP8 and MXFP4 paths the same set at a fixed
  128-row tile (4 heads x 32 tokens, ..., 64 heads x 2 tokens). All currently
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
mapping ids to physical cache rows. THD capture preallocates through
`compress_topk_cand_buffer_size_thd(cu_seqlens_q, None, ratio, q_causal_offsets)`,
since a paged pool has no key prefix. No explicit microbatching and no MXFP8
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
  ordering directly. Neither path adds a kernel, device allocation, or host
  synchronization; the default specialization retains the existing selection
  path.
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
  nothing is copied, cast, or allocated on the execute path. The wrapper keeps
  the convenience: it copies strided inputs contiguous, casts the index tensors
  to int32, and allocates `out` when omitted, all on `stream`.

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
  `execute()`, declines strided `q_attn` / `k_attn` / `lse` and non-contiguous
  or non-int32 `topk_indices` / `topk_length` in `check_support()`, and never
  copies; the wrapper keeps the convenience copies and allocates `out` when
  omitted. On SM100 a plan built without `topk_length` compiles the
  `mTopkLength` operand out entirely (no placeholder tensor).

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
Dense Attention Score Recompute has a separate MXFP8 kernel because its score
and normalization semantics differ from the indexer path.

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
3. In ordinary execution, a pure-torch dtype cast (kernel 3) converts
   `dIndexK_f32` to a BF16 output. Deterministic execution instead uses the
   fixed-order reduction below to write the requested output dtype directly.

With `deterministic=True` on SM100 or SM103, kernel 2 launches a fixed producer
grid and accumulates dK into a CTA-private FP32 workspace. The compile-time
`deterministic_dk_planes` specialization selects 96, 128, or 148 producer
planes (default 96). One CuTe DSL reduction kernel folds the planes in fixed
CTA order, converting only once to the requested `d_index_k` dtype. The dQ
and dW stores remain single-writer.

The plan owns no device memory. For non-deterministic execution, an fp32
`d_index_k` is the accumulator itself (zeroed on the launch stream inside
`execute`); a BF16 `d_index_k` accumulates into a `B * S_k * D` fp32 view carved
from the caller's workspace. `IndexerBackward.scratch_workspace_bytes()` reports
that ordinary scratch size (0 for fp32 `d_index_k` on the default backend).
`execute(..., workspace=)` takes the uint8 buffer on `index_q.device`, and a
missing or undersized buffer raises `ValueError` before `attn_score` is touched.
`indexer_backward_wrapper` allocates it per call on the launch stream and accepts
the same optional `workspace=` keyword for reuse; executions that share one
buffer must not overlap on the device.

`deterministic=True` is request-or-fail and supports SM100/SM103, `D == 128`,
`block_I == 128`, and `topk % 128 == 0` for `128 <= topk <= 2048`. The default
backend accepts `H in {32, 64}` while `sm100_v2` accepts `H == 64`. Each
producer CTA owns one plane and processes the fixed row stripe `cta`,
`cta + num_ctas`, ...; the v2 backend disables its ordinary dynamic ticket
assignment for this path. The main GEMM therefore retains its existing Tensor
Core math while dK has a fixed association order across launches. Scratch is
exactly `ws_align(deterministic_dk_planes * B * S_k * D * sizeof(float))` for
either backend. It is cleared on the launch stream at every execute, then
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

For the default SM100 backend, `topk_indices_global` describes the input
encoding; it is not a performance-tuning switch. When Gather4 is available,
local ids are range-checked and converted to flat ids in registers before the
same optimized gather path. Do not add a separate `local_to_global_wrapper`
launch only for Indexer Backward; preserve the producer's id convention and
set `topk_indices_global` to match it.

The default SM100 backend accepts compact H32 `index_q`, `weights`,
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
conversion solely for Indexer Backward is unlikely to help.

#### SM100-family sparse backward v2 (opt-in)

`backend="sm100_v2"` on `IndexerBackward` / `indexer_backward_wrapper`
selects an alternative kernel-2 (GEMM stage) implementation. `backend` is a
string enum, one of `{"default", "sm100_v2"}` (default `"default"`; an
unknown value raises `ValueError`). The semantics are
**request-or-fail**: outside the envelope below the wrapper raises
`ValueError` (or `RuntimeError` outside SM100/SM103) and never silently falls
back to the default backend.

Start with the default backend when throughput is the goal. Select `sm100_v2` for
its two-term gradient representation, deterministic `d_weights`, or FP32
output contract, not as an automatic performance-tuned alternative; benchmark
the complete wrapper on the target shape when those properties are required.

- **What it computes differently** — weights are upcast to fp32 in-register
  (exact) and the fp32 per-slot gradient matrix is split into a two-term BF16
  expansion (`hi + lo`) before the MMAs, so each individual product in the
  dQ/dK contractions is exact in the fp32 accumulator when it is finite
  there (~675x lower
  gradient-matrix representation error than the default single-BF16 rounding,
  measured as an aggregate over 1e6 random values; individual values gain
  less when their `lo` term is small). `lo` is ~2^-9 of `hi`, so it starts to
  underflow for gradient-matrix magnitudes below ~1e-35 (measured 518x at
  1e-35, 63x at 1e-36, and no advantage left at 1e-38), and the FP32->BF16
  conversion is non-saturating, so a magnitude at or above BF16's
  round-to-nearest overflow threshold (`2^128 - 2^119` ~= 3.3962e38, itself
  above BF16's 3.3895e38 maximum) becomes an infinity in `hi` and the opposite
  infinity in `lo`, which sum to NaN rather than being clamped. `d_weights` is
  reduced deterministically — `d_weights` and `d_index_q` are bitwise
  run-to-run stable. Ordinarily, `d_index_k` stays in the same fp32-atomic
  summation-order class as the default backend; `deterministic=True` routes
  those atomics into fixed-owner planes and folds them in a fixed order,
  making all three gradients bitwise reproducible.
- **Output dtype selects output precision** — `d_weights` and `d_index_k`
  accept FP32 buffers, which receive the fp32 accumulators directly; **the
  headline accuracy gains require caller-supplied FP32 buffers.** The default
  wrapper-allocated outputs keep the input dtypes (BF16), which rounds the
  extra accuracy back to the BF16 representation floor (~1.7e-3 relative).
  FP32 `d_index_k` is zeroed internally (no caller pre-zero contract).
  `d_index_q` is always BF16.
- **Envelope** (validated by `check_support` when a plan is created, and
  re-validated against the real tensors on every execute -- cache hits
  included -- before any buffer is mutated) —
  SM100 (10, 0) or SM103 (10, 3); `H == 64`, `D == 128`, `block_I == 128`;
  `topk % 128 == 0` with `128 <= topk <= 2048`; `sm_scale > 0`; contiguous
  same-device tensors; `attn_score`/`index_score`/`topk_indices` shaped
  `(B, S_q, topk)`. `sm_scale` folds into the gradient in kernel 2 while the
  relu gate reads the *unscaled* score, which is equivalent for any positive
  scale except at the underflow-to-zero boundary: for a slot whose scaled
  score `sm_scale * S` rounds to zero the default backend drops the slot and
  this backend keeps it, and because the gate is a step function that slot's
  full dQ/dK contribution (`g' * w`, not scaled by `S`) is what differs. The
  default backend's multiply preserves subnormals, so reaching it takes an
  exact `sm_scale * S` below `2^-150` (~7.0e-46).
- **Scratch / workspace behavior** — `attn_score` is consumed in place and left
  holding kernel 1's `grad_signal`, while `index_score` is read-only and
  preserved. `sm_scale` folds inside kernel 2 without touching either buffer.
  The plan owns no workspace: `scratch_workspace_bytes()` is
  `ws_align(8)` for the dynamic-ticket counter plus, for a BF16 `d_index_k`,
  the `B * S_k * D` fp32 accumulator (2 MiB = 2,097,152 bytes at B=1,
  S_k=4096, D=128, growing with `S_k`); both are carved from the buffer passed
  to `execute(..., workspace=)` — counter first — and zeroed on the launch
  stream at every execute, so a buffer another engine used in between is
  fine. The wrapper allocates the workspace per call on the launch stream
  unless you pass `workspace=`, and still allocates any output buffer you do
  not pass in.
  With `deterministic=True`, the dense accumulator and counter region are
  replaced by `deterministic_dk_planes` FP32 planes regardless of the output
  dtype; the accepted compile-time values are 96, 128, and 148, and
  deterministic row ownership does not draw tickets.
- **Concurrency** — executions that share one workspace buffer must not
  overlap on the device (ordinary launches would interleave ticket draws;
  deterministic launches would race on the same dK planes); a plan is bound
  to no device and no stream, and
  the workspace must live on `index_q.device` (rejected before `attn_score`
  is overwritten otherwise). The wrapper's per-call workspace makes its calls
  independent; it additionally keys its plan cache on the CUDA device and on
  the **resolved** stream (`stream` when given, otherwise
  `torch.cuda.current_stream()` at call time), so one cached plan's executes
  are stream-ordered. `cudaStreamPerThread` is rejected; use a concrete
  stream for each host thread. Users driving `IndexerBackward`
  objects directly may share one object across streams and devices as long
  as each in-flight execute has its own workspace.
- **Local top-k ids** (`topk_indices_global=False`) are masked against the
  per-batch `S_k` before the batch offset is applied, in-kernel: ids `< 0` or
  `>= S_k` contribute nothing and can never alias a neighbouring batch.

```python
# fp32 output buffers unlock the full accuracy gain
d_weights = torch.empty(B, S_q, H, dtype=torch.float32, device="cuda")
d_index_k = torch.empty(B, S_k, D, dtype=torch.float32, device="cuda")
result = DSA.indexer_backward_wrapper(
    index_q, weights, index_k,
    attn_score, index_score, topk_indices,
    grad_loss=grad_loss, sm_scale=1.0, loss_coeff=1.0, block_I=128,
    backend="sm100_v2",
    d_weights=d_weights, d_index_k=d_index_k,
)
```

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
- **Constraints** — SM90 or SM100+, `H >= 64`, `ratio >= 1`
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

## Limitations

- **CuTe DSL requirement** — install `nvidia-cutlass-dsl[cu13]>=4.5.0`.
  Sparse Attention Forward emits the few SM100 instructions that are not yet
  exposed by the public CuTe facade through the MLIR LLVM inline-assembly
  interface shared by CuTe DSL 4.5 and newer releases.
- **Architecture support** — Sparse Attention Forward supports the mapped
  SM100-family capabilities 10.0, 10.3, and 10.7 only.
  Sparse Attention Backward, Score Recompute, Indexer Forward, Indexer Top-K,
  and Indexer Backward support SM90 and SM100. Dense score recompute in the THD
  (`cu_seqlens`) layout is SM100+ only; SM90 serves BSHD. The combined
  compressed-logits + Top-K forward is SM100-only; the standalone Indexer Top-K
  remains SM90+.
- **Forward scope** — only the 11 supported Prefill instances described above;
  no SM90, regular H128, FP8 cache, decode, or split-KV forward path.
- **Sparse gather path** — the public CuTe facade does not yet expose the TMA
  `gather4` operation. Sparse Attention Forward therefore
  uses a feature-local LLVM inline-assembly bridge with the compiler-owned TMA
  descriptor; the issued data movement remains hardware TMA `gather4`.
- **Indexer Forward only supports `head_dim = 128`**. SM90 supports
  `qhead_per_kv_head ∈ {16, 32, 64}` with `H_kv = 1`; SM100 BF16, MXFP8, and
  MXFP4 support `qhead_per_kv_head ∈ {4, 8, 16, 32, 64}`. Both the dense and
  combined Top-K paths require `H_kv = 1`. MXFP4 runs on the SM100-family
  capabilities 10.0, 10.3, and 10.7 (no SM90 path).
- **Standalone Top-K only up to 2048**; `top_k > 2048` is not supported by
  its radix Top-K kernel. The combined compressed path uses a separate stage-2
  implementation.
- **Compressed-path limits** — the stage-1 compact score kernel is MQA-only
  (`H_kv = 1`); MXFP8 and MXFP4 keep their 128-row tile for every head
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
