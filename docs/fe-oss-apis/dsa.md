# DeepSeek Sparse Attention (DSA)

**This is an experimental API and subject to change.**

## Overview

The DeepSeek Sparse Attention (DSA) module integrates a set of CuTe-DSL
kernels that support the sparse-attention path used by DeepSeek-style models.
The kernels target Hopper (SM90) and Blackwell (SM100+) GPUs and are exposed
through PyTorch and JAX classes and convenience wrappers. See the JAX support
matrix below for operation-specific layout restrictions.

**Scope:** this module ships CuTe-DSL kernels for DSA backward, indexer
scores/top-K, sparse/dense score recompute, and sparse/dense indexer
backward. The production
sparse-attention forward kernel (FlashMLA) is C++ and is **not** integrated
here; when evaluating the backward, use the pure-PyTorch reference in
`test/python/fe_api/dsa/dsa_reference.py::ref_sparse_attention_forward`.

The module packages the following operations:

1. **Sparse Attention Backward** – DSA backward (FlashMLA-shape, SM90/SM100).
2. **Indexer Forward** – CuTe-DSL score kernel (Q @ K^T, ReLU, head reduce,
   ratio causal mask). Non-fused; pair with **Indexer Top-K** for the
   top-K step.
3. **Indexer Top-K** – CuTe-DSL radix top-K kernel with per-row
   ``seq_lens``.
4. **Sparse Indexer / Attention Score Recompute** – sparse (top-K) recompute
   of indexer and attention scores for training loss.
5. **Dense Indexer / Attention Score Recompute** – dense (full-KV) analogues
   of the above.
6. **Indexer Backward** – three-stage pipeline (score-grad, three
   GEMMs, dtype cast) for sparse top-K score tensors.
7. **Dense Indexer Backward** – full-KV counterpart of Indexer Backward.

### Architecture

```
Q, K, W ──► IndexerForward ──► scores ──► IndexerTopK ──► topk_idxs
                                                             │
                                                             v
                      [FlashMLA fwd — external, C++] ──► out, lse
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

PyTorch:

```bash
pip install 'nvidia-cudnn-frontend[cutedsl]'
```

JAX:

```bash
pip install 'nvidia-cudnn-frontend[jax]'
```

---

## API Usage

### PyTorch namespace

```python
from cudnn import DSA

DSA.SparseAttentionBackward
DSA.sparse_attention_backward_wrapper

DSA.IndexerForward
DSA.indexer_forward_wrapper

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

### JAX namespace and wrappers

Importing `cudnn.jax` validates the optional JAX and CUTLASS JAX
dependencies. DSA operation classes and wrappers are then loaded lazily. Each
public symbol is available both directly and through `cudnn.jax.DSA`:

```python
import cudnn.jax as cudnn_jax

cudnn_jax.IndexerForward
cudnn_jax.indexer_forward_wrapper

cudnn_jax.DSA.IndexerForward
cudnn_jax.DSA.indexer_forward_wrapper
```

The function wrappers are already decorated with `jax.jit`; their tuning and
target arguments, including maximum sequence lengths, are static compilation
options. They infer output metadata and return a `TupleDict` of JAX arrays:

```python
result = cudnn_jax.indexer_forward_wrapper(q, k, w, ratio=4)
scores = result["scores"]
```

Layout strings describe the public JAX axis order and are static compilation
arguments. `IndexerForward` accepts batch-major or sequence-major fixed
inputs while keeping the head dimension contiguous:

```python
result = cudnn_jax.indexer_forward_wrapper(
    q_sbhd,
    k_sbhd,
    w_sbh,
    q_layout="SBHD",
    k_layout="SBHD",
    w_layout="SBH",
    output_layout="SBK",
    ratio=4,
)
scores_sbk = result["scores"]
```

The defaults are `q_layout="BSHD"`, `k_layout="BSHD"`,
`w_layout="BSH"`, and `output_layout="BSK"`. Backward and score-recompute
adapters use the same convention and additionally expose layout selectors for
their score, denominator, and gradient tensors. Packed inputs use `THD` and
`TH`, with a `TK` output. The adapters map these public axes to the kernel's
canonical axes; callers do not need to transpose arrays into canonical order.

Use the class API when the caller needs to choose the enclosing JAX
transformation. Classes specialize from array-like exemplars that expose
`shape` and `dtype`; `jax.ShapeDtypeStruct` makes that intent explicit:

```python
import jax

q_spec = jax.ShapeDtypeStruct(q.shape, q.dtype)
k_spec = jax.ShapeDtypeStruct(k.shape, k.dtype)
w_spec = jax.ShapeDtypeStruct(w.shape, w.dtype)

op = cudnn_jax.IndexerForward(q_spec, k_spec, w_spec, ratio=4)
scores = jax.jit(op)(q, k, w)["scores"]
```

The JAX adapters allocate outputs and private workspaces through XLA. Where a
class accepts optional `sample_*` output arguments, those exemplars validate
an explicit output signature; otherwise the adapter infers it. Backward APIs
are functional: they do not update caller-owned gradient buffers in place.
For `IndexerForward`, an explicit `sample_out` describes the physical FP32
buffer whose last extent is rounded up to four; the returned `scores` view is
sliced back to the logical K extent.
`SparseAttentionBackward` returns `dq`, `dkv`, and `d_sink`;
`IndexerBackward` and `DenseIndexerBackward` return `d_index_q`, `d_weights`,
and `d_index_k`.

### JAX target selection

`target_compute_capability` is the exact compilation target, such as `90`,
`100`, `103`, or `107`. When it is omitted, the adapter infers it from a
homogeneous set of local JAX GPUs. An explicit target must match every local
GPU. Device-free AOT and remote compilation are not currently supported.

### JAX support matrix

The SM100+ column covers the implemented SM100-family targets: SM100, SM103,
and SM107.

| Operation | JAX layout | SM90 | SM100+ |
| --- | --- | --- | --- |
| Sparse Attention Backward | Flat MQA tensors | Yes | Yes; inputs must be BF16 |
| Indexer Forward | Fixed BSHD/SBHD or packed THD | Yes; `H_kv=1` and default tuning | Yes |
| Indexer Top-K and index utilities | Flattened rows; fixed or packed index conversion | Yes | Yes |
| Sparse score recompute | Fixed BSHD/SBHD outer layouts, MQA | Yes | Yes |
| Dense score recompute | Fixed BSHD/SBHD outer layouts, MQA | Yes | Yes |
| Dense score recompute | Packed THD/MQA | No | Yes |
| Indexer Backward | Fixed BSHD/SBHD; JAX-only packed THD with global indices | Yes | Yes |
| Dense Indexer Backward | Fixed BSHD/SBHD or packed THD | Yes | Yes |

SM90 packed THD dense score recompute is rejected because that backend needs
host-side cumulative-length reads, which cannot be represented during JAX
tracing. Sparse score recompute has no packed THD signature and its SM90
kernel requires a top-K extent divisible by 128. Both indexer
backward variants require `H >= 64`, `D == 128`, and `block_I == 128`; sparse
Indexer Backward also requires its top-K extent to be divisible by 128. Its
packed THD signature requires `topk_indices_global=True` on JAX.

---

## Components

### 1. Sparse Attention Backward

Backward pass for DeepSeek Sparse Attention. Expects the forward outputs
(`out`, `lse`) from FlashMLA (or the PyTorch reference).

- **Inputs**
  - `q`: `(total_S_q, H, D)` BF16/FP16
  - `kv`: `(total_S_kv, D)` (K = V; MQA)
  - `out`, `dout`: `(total_S_q, H, D_v)`
  - `lse`: `(total_S_q, H)` FP32
  - `attn_sink`: `(H,)` FP32
  - `topk_idxs`: `(total_S_q, topk_max)` INT32 (global)
  - `topk_length` (optional): `(total_S_q,)` INT32 — per-query valid count
- **Outputs** — tuple `(dq, dkv, d_sink)`
- **Constraints** — SM90 or SM100+, `head_dim ∈ {512, 576}`. SM100+
  currently requires BF16 inputs; SM90 also supports FP16. Every
  `topk_length` value is interpreted on device and clamped to
  `[0, topk_max]`; zero-length rows are supported. When `topk_length` is
  supplied, every index before it must be in `[0, total_S_kv)`. When it is
  omitted, negative entries are padding sentinels.

```python
result = DSA.sparse_attention_backward_wrapper(
    q, kv, out, dout, lse, attn_sink, topk_idxs,
    softmax_scale=1.0 / math.sqrt(D),
    topk_length=topk_length,
)
dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
```

### 2. Indexer Forward (score-only)

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
  - Fixed: `q` `(B, S_q, H_q, D)`, `k` `(B, S_k, H_kv, D)`, and
    `w` `(B, S_q, H_q)`, all BF16.
  - Packed THD: `q` `(T_q, H_q, D)`, `k` `(T_k, H_kv, D)`, and
    `w` `(T_q, H_q)`, with `cu_seqlens_q`, `cu_seqlens_k`,
    `max_seqlen_q`, and `max_seqlen_k`.
  - `q_causal_offsets` (optional): INT32 device array with one entry per
    batch/THD segment, on the same device as `q`.
- **Output** — `scores`: `(B, S_q, S_k)` or `(T_q, max_seqlen_k)` FP32.
- **Constraints** — SM90 or SM100+, `head_dim == 128`,
  `qhead_per_kv_head ∈ {32, 64}`. SM90 requires `H_kv == 1` and the
  default tuning configuration.

```python
result = DSA.indexer_forward_wrapper(
    q, k, w, ratio=4, q_causal_offsets=q_causal_offsets,
)
scores = result["scores"]
```

### 3. Indexer Top-K

Radix top-K kernel for selecting candidate KV indices from indexer scores,
with variable per-row effective length.

- **Inputs**
  - `input_values`: `(n_rows, num_cols)` FP32/FP16/BF16
  - `seq_lens`: `(batch_size,)` INT32 (per-batch effective column count)
- **Outputs** — tuple `(indices, values)` (values is `None` when
  `return_val=False`)
- **Constraints** — SM90 or SM100+, `top_k ≤ min(2048, num_cols)` and
  `top_k + next_n - 1 ≤ seq_lens[b] ≤ num_cols` for every batch item.

```python
result = DSA.indexer_top_k_wrapper(
    scores.reshape(-1, scores.shape[-1]),
    seq_lens, top_k=512,
)
indices, values = result["indices"], result["values"]
```

JAX also exposes `local_to_global_wrapper` for fixed or packed local index
conversion and `compactify_wrapper` for packing nonnegative indices and
returning each row's valid `topk_length`.

### 4. Sparse Indexer Score Recompute

Computes softmax over top-K entries of the indexer score:
``predict[b, q, i] = softmax_i(sum_h ReLU(Q_h · K_{topk[i]}^T) · W_h)``.

- **Inputs**: `q_indexer`, `k_indexer`, `weights`, `topk_indices`
  (optional `topk_length`). `topk_indices` are per-batch local KV ids by
  default; pass `topk_indices_global=True` when using ids encoded as
  `batch_idx * S_k + local_idx`.
- **Output** — `predict`: `(B, S_q, topk)` FP32.

### 5. Sparse Attn Score Recompute

L1-normalised head-summed softmax over top-K entries:
``target[b, q, i] = sum_h exp(Q_h · K_{topk[i]}^T · scale - LSE_h) / Z``.

- **Inputs**: `q_attn`, `k_attn`, `lse`, `topk_indices`, `softmax_scale`
  (optional `topk_length`). `topk_indices` are per-batch local KV ids by
  default; pass `topk_indices_global=True` when using ids encoded as
  `batch_idx * S_k + local_idx`.
- **Output** — `target`: `(B, S_q, topk)` FP32.
- Note: the wrapper handles the `-log2(e) * lse` preprocessing internally.

### 6. Dense Indexer / Dense Attn Score Recompute

Full-KV (no top-K) analogues of §4 and §5. Each returns `{'out', 'denom'}`.
They apply the same ratio-causal mask as Indexer Forward; masked positions are
excluded from `denom`. JAX initializes skipped output positions to `-inf`.
Pass the same `q_causal_offsets` to all dense score tensors that feed the same
loss path.

### 7. Indexer Backward

Three-stage sparse top-K pipeline that produces the training gradients for the
indexer tower:

1. `ScoreGradSm90` / `ScoreGradSm100` (kernel 1) — score-grad precompute from
   `attn_score` (target) and `index_score` (predict) into a private temporary.
2. `IndexerBackwardSm90` / `IndexerBackwardSm100` (kernel 2) — three
   warp-specialised GEMMs produce `d_index_q`, `d_weights`, and a
   `dIndexK_f32` accumulator.
3. A dtype cast converts `dIndexK_f32` to the output dtype.

The JAX wrapper treats `grad_loss` as a runtime array operand and returns new
gradient arrays; no input or caller-owned output buffer is mutated.

The common fixed signature uses `index_q` `(B, S_q, H, 128)`, `weights`
`(B, S_q, H)`, `index_k` `(B, S_k, 128)`, and score/index tensors
`(B, S_q, topk)`. JAX additionally accepts a packed signature with
`index_q` `(T_q, H, 128)`, `weights` `(T_q, H)`, `index_k` `(T_k, 128)`, and
score/index tensors `(T_q, topk)`. The packed adapter presents these tensors
to the kernel as a synthetic batch of one, so it requires
`topk_indices_global=True`; each top-K entry must address the flattened packed
K range directly. It does not infer original batch-local IDs from cumulative
sequence lengths. The PyTorch sparse wrapper remains fixed BSHD.

**The TileLang fallback present in the upstream repo is dropped here
(CuTe-DSL only).** If the CuTe-DSL path fails the wrapper raises
`RuntimeError` rather than silently falling back.

```python
result = DSA.indexer_backward_wrapper(
    index_q, weights, index_k,
    attn_score, index_score, topk_indices,
    sm_scale=1.0, loss_coeff=1.0, grad_loss=1.0, block_I=128,
)
d_index_q, d_weights, d_index_k = (
    result["d_index_q"], result["d_weights"], result["d_index_k"],
)
```

### 8. Dense Indexer Backward

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
    sm_scale=1.0, loss_coeff=1.0, grad_loss=1.0, block_I=128, ratio=1,
    q_causal_offsets=q_causal_offsets,
)
```

---

## Limitations

- **Architecture support** — Sparse Attention Backward, Score Recompute, and
  Indexer operations support SM90 and SM100+ subject to the layout restrictions
  in the JAX support matrix.
- **No fused forward** — the production forward is FlashMLA (C++); this
  module ships only the CuTe-DSL kernels.
- **Indexer Forward only supports `head_dim = 128`** and
  `qhead_per_kv_head ∈ {32, 64}`.
- **Top-K only up to 2048**; `top_k > 2048` is not supported by the
  underlying radix top-K kernel.
- **JAX Top-K does not use the PyTorch row-chunking fallback** when its private
  workspace would exceed the INT32 indexing range.
