# DeepSeek Sparse Attention (DSA)

**This is an experimental API and subject to change.**

## Overview

The DeepSeek Sparse Attention (DSA) module integrates a set of CuTe-DSL
kernels that support the sparse-attention path used by DeepSeek-style models.
Most kernels target Hopper (SM90) and Blackwell (SM100+) GPUs; Indexer
Forward and Indexer Top-K remain SM100+ only. The kernels are
delivered as Python classes / wrappers that follow the same `APIBase`
pattern as other cuDNN Frontend operations.

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
3. **Indexer Top-K** – SM100 CuTe-DSL radix top-K kernel with per-row
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
- **Constraints** — SM90 or SM100; SM90 supports the FlashMLA DSA shape with `head_dim ∈ {512, 576}`

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
``S[b, q, k] = sum_h ReLU(Q_h · K_h^T) · W_h`` with a ratio causal mask
(positions with `k >= (S_k * ratio - S_q + q + 1)//ratio` masked; this
reduces to `(q+1)//ratio` when `S_q == S_k * ratio`).

- **Inputs**
  - `q`: `(B, S_q, H_q, D)` BF16
  - `k`: `(B, S_k, H_kv, D)` BF16
  - `w`: `(B, S_q, H_q)` BF16
- **Output** — `scores`: `(B, S_q, S_k)` FP32
- **Constraints** — SM100+, `head_dim == 128`, `qhead_per_kv_head ∈ {32, 64}`

```python
result = DSA.indexer_forward_wrapper(q, k, w, ratio=4)
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
- **Constraints** — SM100+, `top_k ≤ 2048`

```python
result = DSA.indexer_top_k_wrapper(
    scores.reshape(-1, scores.shape[-1]),
    seq_lens, top_k=512,
)
indices, values = result["indices"], result["values"]
```

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
They apply the same bottom-right ratio causal mask as Indexer Forward; masked
positions are written as zero and excluded from `denom`.

### 7. Indexer Backward

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
- **Outputs** — `d_index_q`, `d_weights`, `d_index_k`
- **Constraints** — SM90 or SM100+, `H >= 64`, `ratio >= 1`

```python
dense_index = DSA.dense_indexer_score_recompute_wrapper(index_q, index_k.unsqueeze(2), weights)
dense_attn = DSA.dense_attn_score_recompute_wrapper(attn_q, attn_k, lse, softmax_scale)

result = DSA.dense_indexer_backward_wrapper(
    index_q, weights, index_k,
    dense_attn["out"], dense_attn["denom"],
    dense_index["out"], dense_index["denom"],
    sm_scale=1.0, loss_coeff=1.0, grad_loss=1.0, block_I=128, ratio=1,
)
```

---

## Limitations

- **Architecture support** — Sparse Attention Backward, Score Recompute, and
  Indexer Backward support SM90 and SM100; Indexer Forward and Indexer Top-K
  remain SM100+ only.
- **No fused forward** — the production forward is FlashMLA (C++); this
  module ships only the CuTe-DSL kernels.
- **Indexer Forward only supports `head_dim = 128`** and
  `qhead_per_kv_head ∈ {32, 64}`.
- **Top-K only up to 2048**; `top_k > 2048` is not supported by the
  underlying radix top-K kernel.
