# DeepSeek Sparse Attention (DSA)

**This is an experimental API and subject to change.**

## Overview

The DeepSeek Sparse Attention (DSA) module integrates a set of CuTe-DSL
kernels that support the sparse-attention path used by DeepSeek-style models.
Most kernels target Hopper (SM90) and Blackwell (SM100+) GPUs; Indexer
Forward and Indexer Top-K remain SM100+ only. The kernels are
delivered as Python classes / wrappers that follow the same `ApiBaseTorch`
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

```text
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

``````{tab-set}
:sync-group: frontend-framework

`````{tab-item} PyTorch
:sync: torch
:selected:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

`````

`````{tab-item} JAX
:sync: jax

```bash
pip install nvidia-cudnn-frontend[jax]
```

The JAX optional dependency set requires Python 3.11 or newer.

`````

``````

---

## API Usage

### DSA Namespace

PyTorch remains the default and exposes every DSA class and wrapper below.
The optional JAX namespace exposes functional wrappers for the supported
fixed-shape subsets; the PyTorch class APIs remain unchanged.

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

The explicit JAX facade exposes its implemented functional subset:

```python
from cudnn.jax import DSA

DSA.indexer_forward_wrapper
DSA.indexer_top_k_wrapper
DSA.local_to_global_wrapper
DSA.compactify_wrapper
DSA.sparse_indexer_score_recompute_wrapper
DSA.sparse_attn_score_recompute_wrapper
DSA.dense_indexer_score_recompute_wrapper
DSA.dense_attn_score_recompute_wrapper
DSA.indexer_backward_wrapper
DSA.dense_indexer_backward_wrapper
DSA.sparse_attention_backward_wrapper
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

``````{tab-set}
:sync-group: frontend-framework

`````{tab-item} PyTorch
:sync: torch
:selected:

```python
import math

from cudnn import DSA

result = DSA.sparse_attention_backward_wrapper(
    q, kv, out, dout, lse, attn_sink, topk_idxs,
    softmax_scale=1.0 / math.sqrt(D),
    topk_length=topk_length,
)
dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
```

`````

`````{tab-item} JAX
:sync: jax

```python
import math

import jax
from cudnn.jax import DSA

@jax.jit
def sparse_attention_bwd(q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length):
    return DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=1.0 / math.sqrt(512),
        topk_length=topk_length,
    )

result = sparse_attention_bwd(
    q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length
)
dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
```

The JAX wrapper returns
`TupleDict(dq=..., dkv=..., d_sink=...)`. It supports the
fixed SM100 flat-MQA domain only: BF16 `q`, `out`, and `dout` have shape
`(S_q, H, 512)`, BF16 `kv` has shape `(S_kv, 512)`, `lse` and `attn_sink` are
FP32, and `H` is divisible by 64. Indices are global INT32 values and the
optional runtime `topk_length` has shape `(S_q,)`.

`softmax_scale`, `block_tile=64`, and whether `topk_length` is present are
static compilation choices. Tensor data, indices, and lengths remain runtime
operands. XLA owns the zero-initialized reduction workspaces and output
accumulators. This is an explicit backward operation, not a registered JAX
autodiff rule; SM90 and broader packed/variable-length layouts remain
PyTorch-only.

`````

``````

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
  - `q`: `(B, S_q, H_q, D)` BF16
  - `k`: `(B, S_k, H_kv, D)` BF16
  - `w`: `(B, S_q, H_q)` BF16
  - `q_causal_offsets` (optional): CUDA INT32 tensor with one entry per
    batch/THD segment, on the same device as `q`.
- **Output** — `scores`: `(B, S_q, S_k)` FP32
- **Constraints** — SM100+, `head_dim == 128`, `qhead_per_kv_head ∈ {32, 64}`

``````{tab-set}
:sync-group: frontend-framework

`````{tab-item} PyTorch
:sync: torch
:selected:

```python
from cudnn import DSA

result = DSA.indexer_forward_wrapper(
    q, k, w, ratio=4, q_causal_offsets=q_causal_offsets,
)
scores = result["scores"]
```

The existing wrapper returns a `TupleDict` and retains its optional stream,
THD inputs, and explicit sequence-bound controls.

`````

`````{tab-item} JAX
:sync: jax

```python
import jax
from cudnn.jax import DSA

@jax.jit
def indexer_scores(q, k, w):
    return DSA.indexer_forward_wrapper(q, k, w, ratio=4)

scores = indexer_scores(q, k, w)["scores"]
```

The JAX wrapper returns `TupleDict(scores=...)`. Its public result
has shape `(B, S_q, S_k)`. Internally, the custom call uses an FP32 result whose
last dimension is padded to a multiple of four for TMA. That physical result is
initialized to `-inf`, aliased into the custom call, and sliced back to `S_k`;
the initialization preserves positions that the causal kernel deliberately
does not write.

The JAX wrapper supports concrete, compact BSHD inputs only. THD inputs,
`cu_seqlens_*`, `q_causal_offsets`, and shape-polymorphic export are
unsupported. Tuning options and `sm_scale` are Python-static compilation
state, and XLA supplies the runtime stream.

`````

``````

### 3. Indexer Top-K

Radix top-K kernel for selecting candidate KV indices from indexer scores,
with variable per-row effective length.

- **Inputs**
  - `input_values`: `(n_rows, num_cols)` FP32/FP16/BF16
  - `seq_lens`: `(batch_size,)` INT32 (per-batch effective column count)
- **Outputs** — tuple `(indices, values)` (values is `None` when
  `return_val=False`)
- **Constraints** — SM100+, `top_k ≤ 2048`

``````{tab-set}
:sync-group: frontend-framework

`````{tab-item} PyTorch
:sync: torch
:selected:

```python
from cudnn import DSA

result = DSA.indexer_top_k_wrapper(
    scores.reshape(-1, scores.shape[-1]),
    seq_lens, top_k=512,
)
indices, values = result["indices"], result["values"]
```

`````

`````{tab-item} JAX
:sync: jax

```python
import jax
from cudnn.jax import DSA

@jax.jit
def select_scores(scores, seq_lens):
    return DSA.indexer_top_k_wrapper(
        scores.reshape(-1, scores.shape[-1]),
        seq_lens,
        top_k=512,
        next_n=1,
        return_val=True,
    )

result = select_scores(scores, seq_lens)
indices, values = result["indices"], result["values"]
```

The JAX wrapper returns `TupleDict(indices=..., values=...)`, with both
arrays shaped `(n_rows, top_k)`. The JAX API requires `return_val=True`; the
PyTorch wrapper retains its `return_val=False` mode. The
radix kernel needs an INT32 temporary of shape
`(n_rows, buffer_count, num_cols)`, where `buffer_count` is two for FP32 input
and one for FP16/BF16. The adapter declares that temporary as an uninitialized
custom-call result and drops it from the public result, allowing XLA to own its
lifetime instead of caching mutable workspace in Python. The kernel consumes
the global scratch path when its bucketed column count exceeds the in-CTA
candidate capacity; the launcher retains the same buffer ABI for smaller
problems.

`top_k`, `next_n`, `return_val`, and `num_copy_bits` are Python-static. Shapes
must be concrete, `n_rows` must equal `seq_lens.shape[0] * next_n`, and
`0 < top_k <= min(num_cols, 2048)`. The kernel has no scalar tail when it
selects a vectorized output path, so the wrapper rejects configurations where
`top_k` is not divisible by the selected output vector width. Runtime lengths
are trusted kernel inputs:
every `seq_lens[b]` must satisfy
`top_k + next_n - 1 <= seq_lens[b] <= num_cols`, which keeps every staggered
row at least `top_k` elements long and prevents out-of-range reads. JAX tracing
does not copy those values to the host for validation. The JAX wrapper rejects
workspace sizes above the INT32 indexing limit instead of implementing the
PyTorch path's row-chunked fallback, and it does not expose the persistent
global-counter path.

`````

``````

### 4. Sparse Indexer Score Recompute

Computes softmax over top-K entries of the indexer score:
``predict[b, q, i] = softmax_i(sum_h ReLU(Q_h · K_{topk[i]}^T) · W_h)``.

- **Inputs**: `q_indexer`, `k_indexer`, `weights`, `topk_indices`
  (optional `topk_length`). `topk_indices` are per-batch local KV ids by
  default; pass `topk_indices_global=True` when using ids encoded as
  `batch_idx * S_k + local_idx`.
- **Output** — `predict`: `(B, S_q, topk)` FP32.

``````{tab-set}
:sync-group: frontend-framework

`````{tab-item} PyTorch
:sync: torch
:selected:

```python
from cudnn import DSA

result = DSA.sparse_indexer_score_recompute_wrapper(
    q_indexer,
    k_indexer,
    weights,
    topk_indices,
    topk_length=topk_length,
)
predict = result["predict"]
```

The PyTorch wrapper returns a `TupleDict`, supports SM90 and SM100+, and retains
optional output-buffer and stream controls.

`````

`````{tab-item} JAX
:sync: jax

```python
import jax
from cudnn.jax import DSA

@jax.jit
def recompute_predict(q_indexer, k_indexer, weights, topk_indices, topk_length):
    return DSA.sparse_indexer_score_recompute_wrapper(
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        topk_length=topk_length,
    )

predict = recompute_predict(
    q_indexer, k_indexer, weights, topk_indices, topk_length
)["predict"]
```

The JAX wrapper returns
`TupleDict(predict=...)`. It supports the SM100 fixed
batched MQA layout: BF16 `q_indexer` has shape `(B, S_q, H_q, D)`, BF16
`k_indexer` has shape `(B, S_k, D)`, BF16 `weights` has shape
`(B, S_q, H_q)`, and INT32 `topk_indices` has shape `(B, S_q, topk)`. The
optional INT32 `topk_length` has shape `(B, S_q)`. The FP32 result has the same
shape as `topk_indices` and is fully overwritten by the kernel, so it needs no
initialized alias or algorithmic scratch workspace. When `topk_length` is
omitted, the adapter supplies one hidden `(1, 1)` INT32 placeholder required by
the native launch ABI; the non-compact kernel does not read it.

`````

``````

### 5. Sparse Attn Score Recompute

L1-normalised head-summed softmax over top-K entries:
``target[b, q, i] = sum_h exp(Q_h · K_{topk[i]}^T · scale - LSE_h) / Z``.

- **Inputs**: `q_attn`, `k_attn`, `lse`, `topk_indices`, `softmax_scale`
  (optional `topk_length`). `topk_indices` are per-batch local KV ids by
  default; pass `topk_indices_global=True` when using ids encoded as
  `batch_idx * S_k + local_idx`.
- **Output** — `target`: `(B, S_q, topk)` FP32.
- Note: the wrapper handles the `-log2(e) * lse` preprocessing internally.

``````{tab-set}
:sync-group: frontend-framework

`````{tab-item} PyTorch
:sync: torch
:selected:

```python
from cudnn import DSA

result = DSA.sparse_attn_score_recompute_wrapper(
    q_attn,
    k_attn,
    lse,
    topk_indices,
    softmax_scale,
    topk_length=topk_length,
)
target = result["target"]
```

The PyTorch wrapper returns a `TupleDict`, supports SM90 and SM100+, and retains
optional output-buffer and stream controls.

`````

`````{tab-item} JAX
:sync: jax

```python
import math

import jax
from cudnn.jax import DSA

softmax_scale = 1.0 / math.sqrt(D)

@jax.jit
def recompute_target(q_attn, k_attn, lse, topk_indices, topk_length):
    return DSA.sparse_attn_score_recompute_wrapper(
        q_attn,
        k_attn,
        lse,
        topk_indices,
        softmax_scale,
        topk_length=topk_length,
    )

target = recompute_target(
    q_attn, k_attn, lse, topk_indices, topk_length
)["target"]
```

The JAX wrapper returns `TupleDict(target=...)`. It uses
the same SM100 fixed batched MQA shapes as the indexer variant, with BF16
`q_attn` and `k_attn`, FP32 `lse` shaped `(B, S_q, H_q)`, INT32 indices and
optional lengths, and an FP32 `(B, S_q, topk)` result. The kernel fully
overwrites the result and needs no algorithmic scratch workspace. As in the
indexer variant, omitting `topk_length` creates only the unused hidden ABI
placeholder.

`````

``````

For both JAX sparse score wrappers, shapes must be concrete and compact. THD
inputs and the SM90 kernels remain PyTorch-only. The sparse MQA kernel requires
`qhead_per_kv_head == H_q`, with `H_q` in `{32, 64, 128}`, and a positive head
dimension divisible by 64. The indexer variant requires `topk` to be divisible
by 128; the attention variant requires divisibility by its selected 64- or
128-column tile. `qhead_per_kv_head`, `topk_indices_global`, and
`softmax_scale` (for the attention variant) are Python-static compilation
state; whether `topk_length` is present selects a static kernel variant. Runtime
index and length values are trusted: valid entries must identify KV rows in the
selected local or global index space, and each `topk_length` must be between
zero and `topk`. Positions at or beyond that length are returned as zero.

### 6. Dense Indexer / Dense Attn Score Recompute

Full-KV (no top-K) analogues of §4 and §5. Each returns `{'out', 'denom'}`.
They apply the same ratio-causal mask as Indexer Forward; masked positions are
written as zero and excluded from `denom`. Pass the same `q_causal_offsets`
to all dense score tensors that feed the same loss path.

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

``````{tab-set}
:sync-group: frontend-framework

`````{tab-item} PyTorch
:sync: torch
:selected:

```python
from cudnn import DSA

result = DSA.indexer_backward_wrapper(
    index_q, weights, index_k,
    attn_score, index_score, topk_indices,
    sm_scale=1.0, loss_coeff=1.0, grad_loss=1.0, block_I=128,
)
d_index_q, d_weights, d_index_k = (
    result["d_index_q"], result["d_weights"], result["d_index_k"],
)
```

The PyTorch score-gradient stage overwrites `attn_score` and `index_score`
in place, preserving the existing eager pipeline and class API.

`````

`````{tab-item} JAX
:sync: jax

```python
import jax
from cudnn.jax import DSA

@jax.jit
def indexer_bwd(
    index_q,
    weights,
    index_k,
    attn_score,
    index_score,
    topk_indices,
    grad_loss,
):
    return DSA.indexer_backward_wrapper(
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
        sm_scale=1.0,
        loss_coeff=1.0,
        grad_loss=grad_loss,
        block_I=128,
        topk_indices_global=False,
    )

result = indexer_bwd(
    index_q,
    weights,
    index_k,
    attn_score,
    index_score,
    topk_indices,
    grad_loss,
)
d_index_q = result["d_index_q"]
d_weights = result["d_weights"]
d_index_k = result["d_index_k"]
```

The JAX wrapper returns
`TupleDict(d_index_q=..., d_weights=..., d_index_k=...)` for the
fixed SM100 BSHD subset. It requires BF16 `index_q=(B,S_q,64,128)`, BF16
`weights=(B,S_q,64)`, BF16 `index_k=(B,S_k,128)`, FP32 score tensors, and
INT32 indices with common shape `(B,S_q,topk)`. `topk` must be divisible by
`block_I`, and the JAX path currently requires `block_I=128`.

`grad_loss` is a runtime FP32 scalar or one-element array. `sm_scale`,
`loss_coeff`, `block_I`, and `topk_indices_global` are static compilation
state. Unlike the PyTorch API, JAX leaves both score inputs unchanged: an
XLA-owned hidden buffer carries `grad_signal` between the score-gradient and
GEMM stages, and an XLA-owned zeroed FP32 accumulator is cast to the returned
BF16 `d_index_k`. This is a standalone backward wrapper and does not register
a custom VJP.

`````

``````

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

``````{tab-set}
:sync-group: frontend-framework

`````{tab-item} PyTorch
:sync: torch
:selected:

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

`````

`````{tab-item} JAX
:sync: jax

```python
import jax
from cudnn.jax import DSA

@jax.jit
def dense_indexer_bwd(
    index_q,
    weights,
    index_k,
    attn_score,
    attn_l1norm,
    index_score,
    index_lse,
    grad_loss,
):
    return DSA.dense_indexer_backward_wrapper(
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        sm_scale=1.0,
        loss_coeff=1.0,
        grad_loss=grad_loss,
        block_I=128,
        ratio=1,
    )
```

The JAX binding covers fixed-shape SM100 BSHD inputs with `H=64`, `D=128`,
and `block_I=128`. Inputs use BF16 model tensors, FP32 scores and
denominators, and a runtime FP32 scalar or one-element `grad_loss` array.
`sm_scale`, `loss_coeff`, `block_I`, and `ratio` are static compilation state.
Packed THD inputs and `q_causal_offsets` remain PyTorch-only.

Both score inputs remain immutable. The score-gradient and GEMM kernels run in
one custom call on XLA's stream, using an XLA-owned FP32 `grad_signal`
workspace. The GEMM launch clears its XLA-owned FP32 `d_index_k` accumulator
before the bulk reductions. The returned
`d_index_q`, `d_weights`, and `d_index_k` arrays are BF16.

`````

``````

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
- **JAX coverage includes Indexer Forward, Indexer Top-K, sparse and dense
  score recompute, sparse and dense Indexer Backward, and Sparse Attention
  Backward.**
  The wrappers require concrete compact shapes and do not define autodiff,
  `vmap`, or automatic sharding rules. Their broader SM90/THD PyTorch paths
  remain unchanged.
