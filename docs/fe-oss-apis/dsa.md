# DeepSeek Sparse Attention (DSA)

**This is an experimental API and subject to change.**

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
pip install nvidia-cudnn-frontend[cutedsl]
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
  - `topk_length` (optional): `(total_S_q,)` INT32 — per-query valid count
- **Outputs** — tuple `(dq, dkv, d_sink)`
- **Constraints** — SM90 or SM100; SM90 supports flat MQA tensors with `head_dim ∈ {512, 576}`

```python
result = DSA.sparse_attention_backward_wrapper(
    q, kv, out, dout, lse, attn_sink, topk_idxs,
    softmax_scale=1.0 / math.sqrt(D),
    topk_length=topk_length,
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
  - `q`: `(B, S_q, H_q, D)` BF16, or the architecture-specific FP8 format
    described below.
  - `k`: `(B, S_k, H_kv, D)` BF16, or the architecture-specific FP8 format
    described below.
  - `w`: `(B, S_q, H_q)` BF16. The SM90 FP8 path also accepts FP32 when
    weights have already been pre-scaled by `q_scale * sm_scale`.
  - `q_causal_offsets` (optional): CUDA INT32 tensor with one entry per
    batch/THD segment, on the same device as `q`.
- **Output** — `scores`: `(B, S_q, S_k)` FP32.
- **Precision paths**
  - SM90 `precision="fp8"`: Q/K use E4M3 and `q_scale`/`k_scale` are FP32
    descales with one value per token/head. Set `return_lse=True` (or provide
    `lse_out`) to compute LSE in the same kernel invocation.
  - SM100 `precision="mxfp8"`: Q/K use E4M3 with block-scaled, packed E8M0
    scale tensors; `sf_vec_size` is currently fixed at 32 and
    `qhead_per_kv_head ∈ {32, 64}`.
    THD inputs additionally require
    `cu_seqlens_q_scale_padded`/`cu_seqlens_k_scale_padded`: contiguous CUDA
    INT32 prefix tensors of shape `(B + 1,)` on the Q/K device. The interface
    deliberately performs no device-to-host copy to inspect their values.
    The caller guarantees that each device-side prefix starts at zero, is
    monotonic, covers the corresponding logical sequence, satisfies the
    packed-row alignment (each Q span times `qhead_per_kv_head` is a multiple
    of 128 rows; each K span is a multiple of 128 tokens), and stays within
    the packed scale storage.
- **Constraints** — `head_dim == 128`. The SM90 direct path supports
  `qhead_per_kv_head ∈ {16, 32, 64}` and currently requires `H_kv == 1`;
  SM100 BF16 dense and combined Top-K paths support
  `qhead_per_kv_head ∈ {32, 64}`, as do their MXFP8 paths. All currently
  require `H_kv == 1` (MQA).

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
BF16 and MXFP8. `topk_indices_global=True` is the default. Optional caller-owned
candidate/output/softmax/LSE buffers avoid per-call allocations; size the
candidate buffer with `compress_topk_cand_buffer_size` for BSHD or
`compress_topk_cand_buffer_size_thd` for THD. LSE is supported for BSHD and THD
with both BF16 and MXFP8. Explicit microbatching is BF16 BSHD-only and cannot
be combined with LSE or explicit `q_causal_offsets`. BSHD with explicit
`q_causal_offsets` currently computes its per-batch candidate offsets eagerly
and is not CUDA-graph-capturable; THD capture requires the caller-provided
offsets and buffers returned by `compress_topk_cand_buffer_size_thd`. Compact
addressing requires `0 <= q_causal_offsets[b]`; rows extending beyond the KV
prefix are clamped to `seqlen_k_b`. THD MXFP8 uses the same caller-guaranteed
device-side scale-prefix contract described in §3; the interface does not
validate prefix values with a device-to-host copy.

```python
cand_floats = DSA.compress_topk_cand_buffer_size(
    B, S_q, S_k, ratio=4, return_lse=True,
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
  `return_val=False`)
- **Constraints** — SM90+, `top_k ≤ 2048`

```python
result = DSA.indexer_top_k_wrapper(
    scores.reshape(-1, scores.shape[-1]),
    seq_lens, top_k=512,
)
indices, values = result["indices"], result["values"]
```

### 6. Sparse Indexer Score Recompute

Computes softmax over top-K entries of the indexer score:
``predict[b, q, i] = softmax_i(sum_h ReLU(Q_h · K_{topk[i]}^T) · W_h)``.

- **Inputs**: `q_indexer`, `k_indexer`, `weights`, `topk_indices`
  (optional `topk_length`). `topk_indices` are per-batch local KV ids by
  default; pass `topk_indices_global=True` when using ids encoded as
  `batch_idx * S_k + local_idx`.
- **Output** — `predict`: `(B, S_q, topk)` FP32.

### 7. Sparse Attn Score Recompute

L1-normalised head-summed softmax over top-K entries:
``target[b, q, i] = sum_h exp(Q_h · K_{topk[i]}^T · scale - LSE_h) / Z``.

- **Inputs**: `q_attn`, `k_attn`, `lse`, `topk_indices`, `softmax_scale`
  (optional `topk_length`). `topk_indices` are per-batch local KV ids by
  default; pass `topk_indices_global=True` when using ids encoded as
  `batch_idx * S_k + local_idx`.
- **Output** — `target`: `(B, S_q, topk)` FP32.
- Note: the wrapper handles the `-log2(e) * lse` preprocessing internally.

### 8. Dense Indexer / Dense Attn Score Recompute

Full-KV (no top-K) analogues of §6 and §7. Each returns `{'out', 'denom'}`.
They apply the same ratio-causal mask as Indexer Forward; masked positions are
written as `-inf` and excluded from `denom`. Pass the same `q_causal_offsets` to
all dense score tensors that feed the same loss path.

On SM100, Indexer Forward and Dense Indexer Score Recompute use the same
unified kernel implementation: forward runs it with `compute_lse=False`, while
dense indexer score recompute runs it with `compute_lse=True`. The shared
implementation lives in `score_recompute`; `indexer_forward` only imports it.
Dense Attention Score Recompute has a separate MXFP8 kernel because its score
and normalization semantics differ from the indexer path.

### 9. Indexer Backward

Three-stage sparse top-K pipeline that produces the training gradients for the
indexer tower:

1. `ScoreGradSm90` / `ScoreGradSm100` (kernel 1) — in-place score-grad precompute from
   `attn_score` (target) and `index_score` (predict).
2. `IndexerBackwardSm90` / `IndexerBackwardSm100` (kernel 2) — three
   warp-specialised GEMMs produce `d_index_q`, `d_weights`, and a
   `dIndexK_f32` accumulator.
3. Pure-torch dtype cast (kernel 3) converts `dIndexK_f32` to the output
   dtype.

**The TileLang fallback present in the upstream repo is dropped here
(CuTe-DSL only).** If the CuTe-DSL path fails the wrapper raises
`RuntimeError` rather than silently falling back.

```python
grad_loss = torch.ones((), dtype=torch.float32, device=index_q.device)
result = DSA.indexer_backward_wrapper(
    index_q, weights, index_k,
    attn_score, index_score, topk_indices,
    grad_loss=grad_loss, sm_scale=1.0, loss_coeff=1.0, block_I=128,
)
d_index_q, d_weights, d_index_k = (
    result["d_index_q"], result["d_weights"], result["d_index_k"],
)
```

When compressed forward returns its fused `softmax`, backward can skip both
indexer Q@K score recompute and the separate logits softmax. Pass `softmax`
directly as `index_score`; backward consumes and overwrites this buffer, so
pass `softmax.clone()` if it must be preserved. `attn_score` must use the same
valid-slot mask. Because compressed forward returns global indices by default,
also pass `topk_indices_global=True` unless forward used
`topk_indices_global=False`. The public sparse `indexer_backward_wrapper` has a
BSHD-shaped interface; BF16 THD tensors can use zero-copy `B=1` views (squeeze
the singleton K head and add a batch dimension) together with global Top-K
indices. FP8 and MXFP8 indexer backward are not currently supported because
the backward wrapper requires BF16 Q/K/W inputs.

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
  and Indexer Backward support SM90 and SM100. The combined compressed-logits
  + Top-K forward is SM100-only; the standalone Indexer Top-K remains SM90+.
- **Forward scope** — only the 11 supported Prefill instances described above;
  no SM90, regular H128, FP8 cache, decode, or split-KV forward path.
- **Sparse gather path** — the public CuTe facade does not yet expose the TMA
  `gather4` operation. Sparse Attention Forward therefore
  uses a feature-local LLVM inline-assembly bridge with the compiler-owned TMA
  descriptor; the issued data movement remains hardware TMA `gather4`.
- **Indexer Forward only supports `head_dim = 128`**. SM90 supports
  `qhead_per_kv_head ∈ {16, 32, 64}` with `H_kv = 1`; SM100 BF16 and MXFP8
  support `qhead_per_kv_head ∈ {32, 64}`. Both the dense and combined Top-K
  paths require `H_kv = 1`.
- **Standalone Top-K only up to 2048**; `top_k > 2048` is not supported by
  its radix Top-K kernel. The combined compressed path uses a separate stage-2
  implementation.
- **Compressed-path limits** — the stage-1 compact score kernel is MQA-only
  (`H_kv = 1`); MXFP8 requires `qhead_per_kv_head ∈ {32, 64}`; explicit
  microbatching cannot be combined with MXFP8, LSE, or explicit per-batch
  causal offsets.
