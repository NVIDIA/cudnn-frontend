# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX ops for the DSv4 indexer, top-k and sparse attention (``cudnn.ops.dsa``)."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute

from cudnn.jax import call, neg_inf_init, zeros_init
from cudnn.tensor_adapter import get_compute_capability
from .score_recompute.indexer_score_unified_sm100 import IndexerScoreUnifiedSm100
from .sparse_attention_forward.dsa_fwd_sm100_head64 import SparseAttentionForwardSm100Head64
from .sparse_attention_backward.dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
from .indexer_backward.dense_indexer_backward_sm100 import DenseIndexerBackward2QGemmSm100, ScoreGradDense
from .indexer_backward.indexer_backward_sm100 import IndexerBackwardSm100, ScoreGradSm100
from .indexer_top_k.indexer_top_k_decode_varlen import IndexerTopKKernelVarlenDecode, _bucket_num_cols

_indexer_kernels: dict[tuple, Any] = {}
_sparse_fwd_kernels: dict[tuple, Any] = {}
_sparse_bwd_kernels: dict[tuple, Any] = {}
_dense_indexer_bwd_kernels: dict[tuple, Any] = {}
_indexer_top_k_kernels: dict[tuple, Any] = {}
_sparse_indexer_bwd_kernels: dict[tuple, Any] = {}


def check_sm100():
    major, minor = get_compute_capability()
    if major != 10:
        raise RuntimeError(f"DSA JAX ops require an SM100-family GPU, but found SM{major}{minor}")


@cute.jit
def _indexer_adapter(stream, q, k, weights, scores, denom, *, kernel, sm_scale, seqlen_q, seqlen_k):
    kernel(
        q,
        k,
        weights,
        scores,
        denom,
        cutlass.Float32(sm_scale),
        cutlass.Int32(seqlen_q),
        cutlass.Int32(seqlen_k),
        None,
        None,
        None,
        stream,
    )


def indexer_forward_jax(q: Any, k: Any, weights: Any, *, ratio: int = 4, sm_scale: float = 1.0, return_lse: bool = False) -> Any:
    """Compute dense ratio-causal DSv4 indexer scores using CuTeDSL.

    With ``return_lse=True`` returns ``(scores, lse)`` where ``lse`` is the ``(B, S_q)`` FP32 log-sum-exp
    of each row's causally valid scores (``-inf`` for rows without a valid block), as consumed by
    :func:`dense_indexer_backward_jax`.
    """
    check_sm100()
    if q.ndim != 4 or k.ndim != 4 or weights.ndim != 3:
        raise ValueError("q/k/weights must use BSHD/BSHD/BSH layouts")
    batch, seqlen_q, heads, head_dim = q.shape
    if k.shape[0] != batch or k.shape[2:] != (1, head_dim):
        raise ValueError(f"k must have shape ({batch}, S_k, 1, {head_dim}), got {k.shape}")
    seqlen_k = k.shape[1]
    if weights.shape != (batch, seqlen_q, heads):
        raise ValueError(f"weights must have shape {(batch, seqlen_q, heads)}, got {weights.shape}")
    if q.dtype != jnp.bfloat16 or k.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16:
        raise TypeError("q, k, and weights must be bfloat16")
    if head_dim != 128 or heads != 64 or ratio != 4:
        raise ValueError("DSv4 JAX indexer requires H=64, D=128, ratio=4")
    if seqlen_k % 4:
        raise ValueError("indexer K length must be a multiple of four for FP32 TMA stores")

    key = (head_dim, heads, ratio, bool(return_lse))
    kernel = _indexer_kernels.get(key)
    if kernel is None:
        kernel = IndexerScoreUnifiedSm100(
            head_dim=head_dim,
            qhead_per_kvhead=heads,
            m_block_size=128,
            n_block_size=128,
            k_block_size=64,
            kv_stage=4,
            ratio=ratio,
            is_varlen=False,
            compute_lse=bool(return_lse),
            is_compressed_logits=False,
        )
        _indexer_kernels[key] = kernel

    scores, lse = call(
        _indexer_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((batch, seqlen_q, seqlen_k), jnp.float32),
            jax.ShapeDtypeStruct((batch, seqlen_q), jnp.float32),
        ),
        initialized_outputs={0: neg_inf_init, 1: zeros_init},
        kernel=kernel,
        sm_scale=float(sm_scale),
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
    )(q, k, weights)
    return (scores, lse) if return_lse else scores


@cute.jit
def _sparse_fwd_adapter(stream, q, kv, indices, topk_length, attn_sink, out, max_logits, lse, lse_indexer, *, kernel, softmax_scale):
    kernel(
        q,
        kv,
        indices,
        out,
        max_logits,
        lse,
        lse_indexer,
        attn_sink,
        topk_length,
        cutlass.Float32(softmax_scale),
        stream,
    )


def sparse_attention_forward_jax(
    q: Any,
    kv: Any,
    topk_indices: Any,
    topk_length: Any,
    attn_sink: Any,
    *,
    softmax_scale: float | None = None,
    indexer_topk: int = 512,
) -> tuple[Any, Any, Any, Any]:
    """Run DSv4 sparse attention forward on flat global index lists."""
    check_sm100()
    if q.ndim != 3 or q.shape[1:] != (64, 512) or q.dtype != jnp.bfloat16:
        raise ValueError("q must have shape (Tq, 64, 512) and dtype bfloat16")
    if kv.ndim != 2 or kv.shape[1] != 512 or kv.dtype != q.dtype:
        raise ValueError("kv must have shape (Tkv, 512) with q's dtype")
    if topk_indices.ndim != 2 or topk_indices.shape[0] != q.shape[0] or topk_indices.dtype != jnp.int32:
        raise ValueError("topk_indices must be (Tq, K) int32")
    if topk_indices.shape[1] % 64:
        raise ValueError("physical sparse-attention K must be a multiple of 64")
    if topk_length.shape != (q.shape[0],) or topk_length.dtype != jnp.int32:
        raise ValueError("topk_length must be (Tq,) int32")
    if attn_sink.shape != (64,) or attn_sink.dtype != jnp.float32:
        raise ValueError("attn_sink must be (64,) float32")
    # indexer_topk=0 disables the indexer-prefix LSE (e.g. HCA, whose index list is
    # a fixed local window plus causal compressed blocks); lse_indexer is then unwritten.
    if indexer_topk not in (0, 512, 1024, 2048) or indexer_topk > topk_indices.shape[1]:
        raise ValueError("indexer_topk must be 0 or a supported prefix no wider than K")

    kernel_key = (512, int(indexer_topk))
    kernel = _sparse_fwd_kernels.get(kernel_key)
    if kernel is None:
        kernel = SparseAttentionForwardSm100Head64(head_dim=512, indexer_topk=indexer_topk)
        _sparse_fwd_kernels[kernel_key] = kernel
    scale = 1.0 / math.sqrt(512) if softmax_scale is None else float(softmax_scale)
    tq = q.shape[0]
    return call(
        _sparse_fwd_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((tq, 64, 512), q.dtype),
            jax.ShapeDtypeStruct((tq, 64), jnp.float32),
            jax.ShapeDtypeStruct((tq, 64), jnp.float32),
            jax.ShapeDtypeStruct((tq, 64), jnp.float32),
        ),
        kernel=kernel,
        softmax_scale=scale,
    )(q, kv, topk_indices, topk_length, attn_sink)


@cute.jit
def _sparse_bwd_adapter(
    stream,
    q,
    kv,
    out,
    dout,
    lse,
    attn_sink,
    indices,
    topk_length,
    dq,
    dkv,
    dsink,
    workspace_lse_odo,
    workspace_dkv,
    *,
    kernel,
    softmax_scale,
    total_q,
    total_kv,
):
    problem_shape = (
        cutlass.Int32(total_q),
        cutlass.Int32(total_kv),
        cutlass.Int32(512),
        (cutlass.Int32(64), cutlass.Int32(1)),
    )
    kernel(
        problem_shape,
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        indices,
        topk_length,
        dq,
        dkv,
        dsink,
        workspace_lse_odo,
        workspace_dkv,
        cutlass.Float32(softmax_scale),
        stream,
    )


def sparse_attention_backward_jax(
    q: Any,
    kv: Any,
    out: Any,
    dout: Any,
    lse: Any,
    attn_sink: Any,
    topk_indices: Any,
    topk_length: Any,
    *,
    softmax_scale: float | None = None,
) -> tuple[Any, Any, Any]:
    """Explicit JAX custom call for the generic H64/D512 sparse backward."""
    check_sm100()
    if q.shape[1:] != (64, 512) or q.dtype != jnp.bfloat16:
        raise ValueError("q must be (Tq, 64, 512) bfloat16")
    if kv.ndim != 2 or kv.shape[1] != 512 or kv.dtype != q.dtype:
        raise ValueError("kv must be (Tkv, 512) bfloat16")
    if out.shape != q.shape or dout.shape != q.shape or out.dtype != q.dtype or dout.dtype != q.dtype:
        raise ValueError("out and dout must match q")
    if lse.shape != q.shape[:2] or lse.dtype != jnp.float32:
        raise ValueError("lse must be (Tq, 64) float32")
    if attn_sink.shape != (64,) or attn_sink.dtype != jnp.float32:
        raise ValueError("attn_sink must be (64,) float32")
    if topk_indices.ndim != 2 or topk_indices.shape[0] != q.shape[0] or topk_indices.dtype != jnp.int32:
        raise ValueError("topk_indices must be (Tq, K) int32")
    if topk_length.shape != (q.shape[0],) or topk_length.dtype != jnp.int32:
        raise ValueError("topk_length must be (Tq,) int32")

    total_q, total_kv, max_topk = q.shape[0], kv.shape[0], topk_indices.shape[1]
    key = (max_topk,)
    kernel = _sparse_bwd_kernels.get(key)
    if kernel is None:
        kernel = FlashAttentionDSABackwardSm100(
            element_dtype=cutlass.BFloat16,
            head_dim=512,
            head_dim_v=512,
            block_tile=64,
            max_topk=max_topk,
        )
        _sparse_bwd_kernels[key] = kernel
    workspace_lse_shape = FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(total_q, 512, 64, 1, cutlass.Float32)
    workspace_dkv_shape = FlashAttentionDSABackwardSm100._get_workspace_size_dKV(total_kv, 512, 1, cutlass.Float32)
    scale = 1.0 / math.sqrt(512) if softmax_scale is None else float(softmax_scale)
    results = call(
        _sparse_bwd_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(kv.shape, kv.dtype),
            jax.ShapeDtypeStruct((64,), jnp.float32),
            jax.ShapeDtypeStruct(workspace_lse_shape, jnp.uint8),
            jax.ShapeDtypeStruct(workspace_dkv_shape, jnp.uint8),
        ),
        initialized_outputs={0: zeros_init, 1: zeros_init, 2: zeros_init, 3: zeros_init, 4: zeros_init},
        kernel=kernel,
        softmax_scale=scale,
        total_q=total_q,
        total_kv=total_kv,
    )(q, kv, out, dout, lse, attn_sink, topk_indices, topk_length)
    return results[:3]


@cute.jit
def _dense_indexer_bwd_adapter(
    stream,
    q,
    weights,
    k,
    attn_score,
    attn_l1norm,
    index_score,
    index_lse,
    grad_loss,
    d_q,
    d_weights,
    d_k,
    *,
    score_grad,
    gemm,
    grad_scale,
    sm_scale,
    seqlen_q,
    seqlen_k,
):
    # Kernel 1 turns (index_score, attn_score) into the score gradient *in place* in index_score.
    score_grad(
        index_score,
        attn_score,
        index_lse,
        attn_l1norm,
        None,
        None,
        None,
        grad_loss,
        cutlass.Float32(grad_scale),
        cutlass.Int32(seqlen_q),
        cutlass.Int32(seqlen_k),
        stream,
    )
    # Kernel 2 consumes that gradient and reduce-adds dK (FP32, zero-initialized) across Q tokens.
    gemm(
        q,
        weights,
        k,
        d_q,
        d_weights,
        d_k,
        index_score,
        None,
        None,
        None,
        cutlass.Float32(sm_scale),
        cutlass.Int32(seqlen_q),
        cutlass.Int32(seqlen_k),
        stream,
    )


def dense_indexer_backward_jax(
    q: Any,
    k: Any,
    weights: Any,
    attn_score: Any,
    attn_l1norm: Any,
    index_score: Any,
    index_lse: Any,
    grad_loss: Any,
    *,
    ratio: int = 4,
    sm_scale: float = 1.0,
) -> tuple[Any, Any, Any]:
    """Dense full-KV indexer backward for the DSv4 KL indexer loss using CuTeDSL.

    Computes ``d_q``, ``d_weights`` and ``d_k`` of ``KL(target || softmax(index_score))`` where
    ``target = attn_score / attn_l1norm`` and ``index_score``/``index_lse`` come from
    ``indexer_forward_jax(..., return_lse=True)``. Scores are masked by the ratio-causal rule
    ``block < (query + 1) // ratio``. The per-token KL gradient is scaled by
    ``grad_loss / (B * S_q)``, so a mean-over-tokens loss uses ``grad_loss`` equal to its cotangent.

    ``attn_score`` is read-only here. ``index_score`` is consumed as scratch by the kernel, so XLA
    copies it unless the caller donates it. ``d_k`` is returned in FP32 (the kernel reduce-adds
    into it); ``d_q`` and ``d_weights`` are BF16.
    """
    check_sm100()
    if q.ndim != 4 or k.ndim != 4 or weights.ndim != 3:
        raise ValueError("q/k/weights must use BSHD/BSHD/BSH layouts")
    batch, seqlen_q, heads, head_dim = q.shape
    if k.shape[0] != batch or k.shape[2:] != (1, head_dim):
        raise ValueError(f"k must have shape ({batch}, S_k, 1, {head_dim}), got {k.shape}")
    seqlen_k = k.shape[1]
    if weights.shape != (batch, seqlen_q, heads):
        raise ValueError(f"weights must have shape {(batch, seqlen_q, heads)}, got {weights.shape}")
    if q.dtype != jnp.bfloat16 or k.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16:
        raise TypeError("q, k, and weights must be bfloat16")
    if head_dim != 128 or heads != 64 or ratio != 4:
        raise ValueError("DSv4 JAX dense indexer backward requires H=64, D=128, ratio=4")
    score_shape = (batch, seqlen_q, seqlen_k)
    if attn_score.shape != score_shape or index_score.shape != score_shape:
        raise ValueError(f"attn_score/index_score must have shape {score_shape}")
    if attn_l1norm.shape != (batch, seqlen_q) or index_lse.shape != (batch, seqlen_q):
        raise ValueError(f"attn_l1norm/index_lse must have shape {(batch, seqlen_q)}")
    for name, tensor in (("attn_score", attn_score), ("attn_l1norm", attn_l1norm), ("index_score", index_score), ("index_lse", index_lse)):
        if tensor.dtype != jnp.float32:
            raise TypeError(f"{name} must be float32")
    if grad_loss.size != 1 or grad_loss.dtype != jnp.float32:
        raise ValueError("grad_loss must be a single float32 element")
    if seqlen_k % 4:
        raise ValueError("indexer K length must be a multiple of four for FP32 TMA stores")

    key = (head_dim, heads, ratio)
    kernels = _dense_indexer_bwd_kernels.get(key)
    if kernels is None:
        kernels = (
            ScoreGradDense(ratio=ratio, block_I=128),
            DenseIndexerBackward2QGemmSm100(head_dim=head_dim, heads=heads, block_I=128, ratio=ratio),
        )
        _dense_indexer_bwd_kernels[key] = kernels
    score_grad, gemm = kernels

    d_q, d_weights, d_k, _ = call(
        _dense_indexer_bwd_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(q.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(weights.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct((batch, seqlen_k, head_dim), jnp.float32),
            jax.ShapeDtypeStruct(score_shape, jnp.float32),
        ),
        initialized_outputs={2: zeros_init},
        # index_score is overwritten in place with the score gradient; alias it to a (discarded) output
        # so XLA copies the buffer when the caller still needs the original.
        input_output_aliases={5: 3},
        score_grad=score_grad,
        gemm=gemm,
        grad_scale=1.0 / (batch * seqlen_q),
        sm_scale=float(sm_scale),
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
    )(q, weights, k.reshape(batch, seqlen_k, head_dim), attn_score, attn_l1norm, index_score, index_lse, grad_loss.reshape(1))
    return d_q, d_weights, d_k


@cute.jit
def _indexer_top_k_adapter(stream, scores, seq_lens, indices, scratch, *, kernel):
    kernel(scores, None, scratch, None, seq_lens, indices, None, stream)


def indexer_top_k_forward_jax(scores: Any, seq_lens: Any, *, top_k: int, tie_break: int = 1) -> Any:
    """Per-row top-``top_k`` column indices of ``scores`` using the CuTeDSL radix kernel.

    ``scores`` is ``(n_rows, num_cols)`` float32/float16/bfloat16 and ``seq_lens`` is ``(n_rows,)`` int32: row ``i``
    only considers its first ``seq_lens[i]`` columns (a prefix; the rest is never read). Returns ``(n_rows, top_k)``
    int32 column ids. The ids are **unsorted**. Rows with ``seq_lens[i] <= top_k`` list ``0..seq_lens[i]-1`` in
    ascending order and are padded with ``-1``. ``tie_break`` is 0 (arbitrary), 1 (smaller column id wins among
    equal scores, like ``jax.lax.top_k``) or 2 (larger).
    """
    check_sm100()
    if scores.ndim != 2 or seq_lens.shape != (scores.shape[0],):
        raise ValueError(f"scores must be (n_rows, num_cols) and seq_lens (n_rows,), got {scores.shape} and {seq_lens.shape}")
    if seq_lens.dtype != jnp.int32:
        raise TypeError("seq_lens must be int32")
    cutlass_dtypes = {jnp.dtype(jnp.float32): cutlass.Float32, jnp.dtype(jnp.float16): cutlass.Float16, jnp.dtype(jnp.bfloat16): cutlass.BFloat16}
    if scores.dtype not in cutlass_dtypes:
        raise TypeError("scores must be float32, float16 or bfloat16")
    if not 0 < top_k <= 2048:
        raise ValueError(f"top_k must be in (0, 2048], got {top_k}")
    if tie_break not in (0, 1, 2):
        raise ValueError("tie_break must be 0 (arbitrary), 1 (smaller index) or 2 (larger index)")
    n_rows, num_cols = scores.shape
    num_copy_bits = 256
    # The kernel issues 256-bit vector loads, so the row pitch must be a multiple of 32 bytes. The padding is
    # never read (it lies beyond every seq_len) as long as seq_lens <= num_cols, which the caller guarantees.
    vec = num_copy_bits // (8 * scores.dtype.itemsize)
    padded_cols = -(-num_cols // vec) * vec
    if padded_cols != num_cols:
        scores = jnp.pad(scores, ((0, 0), (0, padded_cols - num_cols)), constant_values=-jnp.inf)
    buffer_numbers = 2 if scores.dtype == jnp.float32 else 1
    if n_rows * buffer_numbers * padded_cols > (1 << 31) - 1:
        raise ValueError("n_rows * num_cols is too large for the kernel's int32 scratch indexing; split the rows")

    large_occupancy = n_rows > 148
    key = (scores.dtype, _bucket_num_cols(padded_cols), top_k, large_occupancy, num_copy_bits, tie_break)
    kernel = _indexer_top_k_kernels.get(key)
    if kernel is None:
        kernel = IndexerTopKKernelVarlenDecode(
            cutlass_dtypes[scores.dtype],
            key[1],
            top_k,
            1,
            num_copy_bits=num_copy_bits,
            return_val=False,
            large_occupancy=large_occupancy,
            tie_break=tie_break,
        )
        _indexer_top_k_kernels[key] = kernel

    indices, _ = call(
        _indexer_top_k_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((n_rows, top_k), jnp.int32),
            jax.ShapeDtypeStruct((n_rows, buffer_numbers, padded_cols), jnp.int32),  # radix scratch
        ),
        kernel=kernel,
    )(scores, seq_lens)
    return indices


@cute.jit
def _sparse_indexer_bwd_adapter(
    stream,
    q,
    weights,
    k,
    attn_score,
    index_score,
    topk_indices,
    grad_loss,
    d_q,
    d_weights,
    d_k,
    *,
    score_grad,
    gemm,
    grad_scale,
    sm_scale,
):
    # Kernel 1 turns (index_score, attn_score) into the score gradient *in place* in attn_score.
    score_grad(attn_score, index_score, grad_loss, d_k, cutlass.Float32(grad_scale), stream)
    # Kernel 2 consumes it (launched without PDL, so it starts after kernel 1 has finished) and reduce-adds dK.
    gemm(q, weights, k, d_q, d_weights, d_k, attn_score, topk_indices, cutlass.Float32(sm_scale), stream)


def indexer_backward_jax(
    q: Any,
    k: Any,
    weights: Any,
    attn_score: Any,
    index_score: Any,
    topk_indices: Any,
    grad_loss: Any,
    *,
    sm_scale: float = 1.0,
) -> tuple[Any, Any, Any]:
    """Sparse (top-k) indexer backward for the DSv4 KL indexer loss using CuTeDSL.

    Computes ``d_q``, ``d_weights`` and ``d_k`` of ``KL(target || softmax(logits))`` where the softmax runs over
    each query's ``topk`` selected compressed blocks only. Inputs (all per selected slot, ``[B, S_q, topk]``):

    * ``attn_score``: the teacher distribution over the selected slots (read as ``target``; clipped at ``e^-100``).
      It is consumed as scratch by the kernel, so XLA copies it unless the caller donates it.
    * ``index_score``: the student softmax **probabilities** over the selected slots (``0`` for invalid slots).
    * ``topk_indices``: int32 *local* block ids into this batch element's ``S_k`` blocks; ``< 0`` marks an invalid slot.

    The per-token KL gradient is scaled by ``grad_loss / (B * S_q)``. ``d_k`` is returned in FP32 (the kernel
    reduce-adds into it, zero-initialized here); ``d_q`` and ``d_weights`` are BF16. Requires H=64, D=128 and
    ``topk == 512``.
    """
    check_sm100()
    if q.ndim != 4 or k.ndim != 4 or weights.ndim != 3:
        raise ValueError("q/k/weights must use BSHD/BSHD/BSH layouts")
    batch, seqlen_q, heads, head_dim = q.shape
    if k.shape[0] != batch or k.shape[2:] != (1, head_dim):
        raise ValueError(f"k must have shape ({batch}, S_k, 1, {head_dim}), got {k.shape}")
    seqlen_k = k.shape[1]
    if weights.shape != (batch, seqlen_q, heads):
        raise ValueError(f"weights must have shape {(batch, seqlen_q, heads)}, got {weights.shape}")
    if q.dtype != jnp.bfloat16 or k.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16:
        raise TypeError("q, k, and weights must be bfloat16")
    if head_dim != 128 or heads != 64:
        raise ValueError("DSv4 JAX sparse indexer backward requires H=64, D=128")
    topk = topk_indices.shape[-1]
    if topk != 512:
        # Only the production top-k is validated here. At topk=128 the persistent GEMM kernel was observed to
        # stall (deadlock) on a 152-SM GB300 for some row counts (e.g. B=4, S=1024) while completing with a
        # smaller persistent grid, so shorter rows are not exposed until that is understood.
        raise ValueError(f"the JAX sparse indexer backward supports topk=512 only, got {topk}")
    score_shape = (batch, seqlen_q, topk)
    if attn_score.shape != score_shape or index_score.shape != score_shape or topk_indices.shape != score_shape:
        raise ValueError(f"attn_score/index_score/topk_indices must have shape {score_shape}")
    if attn_score.dtype != jnp.float32 or index_score.dtype != jnp.float32:
        raise TypeError("attn_score and index_score must be float32")
    if topk_indices.dtype != jnp.int32:
        raise TypeError("topk_indices must be int32")
    if grad_loss.size != 1 or grad_loss.dtype != jnp.float32:
        raise ValueError("grad_loss must be a single float32 element")

    sm_count = jax.devices()[0].core_count
    key = (head_dim, heads, topk, batch * seqlen_q, batch * seqlen_k, sm_count)
    kernels = _sparse_indexer_bwd_kernels.get(key)
    if kernels is None:
        kernels = (
            ScoreGradSm100(topk=topk, zero_dk_f32=False),
            IndexerBackwardSm100(
                head_dim=head_dim,
                heads=heads,
                block_I=128,
                topk=topk,
                total_seqlen_k=batch * seqlen_k,
                total_rows=batch * seqlen_q,
                persistent_grid_size=sm_count,
                topk_indices_global=False,
                enable_score_pdl=False,
            ),
        )
        _sparse_indexer_bwd_kernels[key] = kernels
    score_grad, gemm = kernels

    d_q, d_weights, d_k, _ = call(
        _sparse_indexer_bwd_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(q.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(weights.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct((batch, seqlen_k, head_dim), jnp.float32),
            jax.ShapeDtypeStruct(score_shape, jnp.float32),
        ),
        initialized_outputs={2: zeros_init},
        # attn_score is overwritten in place with the score gradient; alias it to a (discarded) output so XLA
        # copies the buffer when the caller still needs the original.
        input_output_aliases={3: 3},
        score_grad=score_grad,
        gemm=gemm,
        grad_scale=1.0 / (batch * seqlen_q),
        sm_scale=float(sm_scale),
    )(q, weights, k.reshape(batch, seqlen_k, head_dim), attn_score, index_score, topk_indices, grad_loss.reshape(1))
    return d_q, d_weights, d_k


__all__ = [
    "indexer_forward_jax",
    "dense_indexer_backward_jax",
    "indexer_top_k_forward_jax",
    "indexer_backward_jax",
    "sparse_attention_forward_jax",
    "sparse_attention_backward_jax",
]
