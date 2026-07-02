# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for the DSA sparse score-recompute kernels."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp

from ..._jax.cutedsl import BufferSpec, call_cutedsl
from ..._jax.validation import require_dtype
from .config import (
    resolve_dense_score_kernel_config,
    resolve_sparse_score_kernel_config,
)


class SparseIndexerScoreRecomputeResult(NamedTuple):
    """Normalized sparse indexer scores."""

    predict: Any


class SparseAttnScoreRecomputeResult(NamedTuple):
    """Normalized sparse attention scores."""

    target: Any


class DenseScoreRecomputeResult(NamedTuple):
    """Raw dense scores and their per-query normalization denominator."""

    out: Any
    denom: Any


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    score_type: str,
    head_dim: int,
    qhead_per_kv_head: int,
    topk: int,
    m_block_size: int,
    n_block_size: int,
    k_block_size: int | None,
    kv_stage: int,
    have_topk_length: bool,
    topk_in_smem: bool,
    topk_indices_global: bool,
    softmax_scale: float,
):
    # Load the architecture-specific kernel only when tracing the operation.
    from cutlass import Float32

    from .sparse_score_recompute_sm100 import SparseScoreRecomputeSm100

    kernel = SparseScoreRecomputeSm100(
        head_dim=head_dim,
        qhead_per_kvhead=qhead_per_kv_head,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        topk=topk,
        kv_stage=kv_stage,
        score_type=score_type,
        have_topk_length=have_topk_length,
        topk_in_smem=topk_in_smem,
        k_block_size=k_block_size,
        topk_indices_global=topk_indices_global,
    )

    if have_topk_length:

        def launch(stream, q, k, per_head, topk_indices, topk_length, out):
            kernel(
                q,
                k,
                per_head,
                topk_indices,
                out,
                topk_length,
                Float32(softmax_scale),
                stream,
            )

    else:

        def launch(stream, q, k, per_head, topk_indices, out, topk_length_workspace):
            kernel(
                q,
                k,
                per_head,
                topk_indices,
                out,
                topk_length_workspace,
                Float32(softmax_scale),
                stream,
            )

    return launch


@lru_cache(maxsize=None)
def _make_dense_launcher(
    *,
    score_type: str,
    head_dim: int,
    qhead_per_kv_head: int,
    ratio: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    have_q_causal_offsets: bool,
):
    from cutlass import Float32, Int32

    from .dense_score_recompute_sm100 import DenseScoreRecomputeSm100

    config = resolve_dense_score_kernel_config(
        score_type=score_type,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
    )
    kernel = DenseScoreRecomputeSm100(
        head_dim=head_dim,
        qhead_per_kvhead=qhead_per_kv_head,
        m_block_size=config.m_block_size,
        n_block_size=config.n_block_size,
        k_block_size=config.k_block_size,
        kv_stage=config.kv_stage,
        score_type=score_type,
        ratio=ratio,
        is_varlen=False,
    )
    if have_q_causal_offsets:

        def launch(stream, q, k, per_head, q_causal_offsets, out, denom):
            kernel(
                q,
                k,
                per_head,
                out,
                denom,
                Float32(scale),
                Int32(max_seqlen_q),
                Int32(max_seqlen_k),
                None,
                None,
                q_causal_offsets,
                stream,
            )

    else:

        def launch(stream, q, k, per_head, out, denom):
            kernel(
                q,
                k,
                per_head,
                out,
                denom,
                Float32(scale),
                Int32(max_seqlen_q),
                Int32(max_seqlen_k),
                None,
                None,
                None,
                stream,
            )

    return launch


def _dense_score_recompute(
    q: Any,
    k: Any,
    per_head: Any,
    *,
    score_type: str,
    per_head_name: str,
    per_head_dtype: Any,
    scale: float,
    qhead_per_kv_head: Optional[int],
    ratio: int,
    q_causal_offsets: Any | None,
) -> DenseScoreRecomputeResult:
    arrays = (
        ("q", q, 4),
        ("k", k, 4),
        (per_head_name, per_head, 3),
    )
    for name, value, rank in arrays:
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{name} must have shape and dtype metadata")
        if len(value.shape) != rank:
            raise ValueError(f"{name} must have rank {rank}, got shape {value.shape}")

    require_dtype("q.dtype", q, (jnp.bfloat16,))
    require_dtype("k.dtype", k, (jnp.bfloat16,))
    require_dtype(f"{per_head_name}.dtype", per_head, (per_head_dtype,))

    batch, seqlen_q, num_query_heads, head_dim = q.shape
    k_batch, seqlen_k, num_kv_heads, k_head_dim = k.shape
    dimensions = {
        "batch": batch,
        "S_q": seqlen_q,
        "S_k": seqlen_k,
        "H_q": num_query_heads,
        "H_kv": num_kv_heads,
        "head dimension": head_dim,
    }
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("Dense score-recompute dimensions must be positive, got " + ", ".join(nonpositive))
    if k_batch != batch:
        raise ValueError(f"q and k batch dimensions must match, got {batch} and {k_batch}")
    if k_head_dim != head_dim:
        raise ValueError("q and k head dimensions must match, got " f"{head_dim} and {k_head_dim}")
    if tuple(per_head.shape) != (batch, seqlen_q, num_query_heads):
        raise ValueError(f"{per_head_name} shape must be " f"{(batch, seqlen_q, num_query_heads)}, got {per_head.shape}")
    if num_query_heads % num_kv_heads:
        raise ValueError(f"H_q ({num_query_heads}) must be divisible by H_kv ({num_kv_heads})")

    inferred_qhead_per_kv_head = num_query_heads // num_kv_heads
    if qhead_per_kv_head is None:
        qhead_per_kv_head = inferred_qhead_per_kv_head
    if qhead_per_kv_head != inferred_qhead_per_kv_head:
        raise ValueError("qhead_per_kv_head must equal H_q / H_kv, got " f"{qhead_per_kv_head} and {num_query_heads} / {num_kv_heads}")
    if ratio < 1:
        raise ValueError(f"ratio must be at least 1, got {ratio}")

    inputs = (q, k, per_head)
    if q_causal_offsets is not None:
        if not hasattr(q_causal_offsets, "shape") or not hasattr(q_causal_offsets, "dtype"):
            raise TypeError("q_causal_offsets must have shape and dtype metadata")
        if tuple(q_causal_offsets.shape) != (batch,):
            raise ValueError(f"q_causal_offsets must have shape {(batch,)}, got " f"{q_causal_offsets.shape}")
        require_dtype(
            "q_causal_offsets.dtype",
            q_causal_offsets,
            (jnp.int32,),
        )
        inputs += (q_causal_offsets,)

    resolved_scale = float(scale)
    out, denom = call_cutedsl(
        _make_dense_launcher(
            score_type=score_type,
            head_dim=head_dim,
            qhead_per_kv_head=qhead_per_kv_head,
            ratio=int(ratio),
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            scale=resolved_scale,
            have_q_causal_offsets=q_causal_offsets is not None,
        ),
        inputs,
        outputs=(
            BufferSpec(
                "out",
                (batch, seqlen_q, seqlen_k),
                jnp.float32,
                fill_value=float("-inf"),
            ),
            BufferSpec("denom", (batch, seqlen_q), jnp.float32),
        ),
        use_static_tensors=True,
    )
    return DenseScoreRecomputeResult(out=out, denom=denom)


def _sparse_score_recompute(
    q: Any,
    k: Any,
    per_head: Any,
    topk_indices: Any,
    *,
    score_type: str,
    output_name: str,
    per_head_name: str,
    per_head_dtype: Any,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int],
    topk_length: Optional[Any],
    topk_indices_global: bool,
) -> Any:
    arrays = (
        ("q", q, 4),
        ("k", k, 3),
        (per_head_name, per_head, 3),
        ("topk_indices", topk_indices, 3),
    )
    for name, value, rank in arrays:
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{name} must be a JAX array with shape and dtype metadata")
        if len(value.shape) != rank:
            raise ValueError(f"{name} must have rank {rank}, got shape {value.shape}")

    q_shape = tuple(q.shape)
    k_shape = tuple(k.shape)
    per_head_shape = tuple(per_head.shape)
    topk_shape = tuple(topk_indices.shape)

    require_dtype("q.dtype", q, (jnp.bfloat16,))
    require_dtype("k.dtype", k, (jnp.bfloat16,))
    require_dtype(f"{per_head_name}.dtype", per_head, (per_head_dtype,))
    require_dtype("topk_indices.dtype", topk_indices, (jnp.int32,))

    batch, seqlen_q, num_query_heads, head_dim = q_shape
    k_batch, seqlen_k, k_head_dim = k_shape
    topk_batch, topk_seqlen_q, topk = topk_shape
    dimensions = {
        "batch": batch,
        "S_q": seqlen_q,
        "S_k": seqlen_k,
        "H_q": num_query_heads,
        "head dimension": head_dim,
        "topk": topk,
    }
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("Sparse score-recompute dimensions must be positive, got " + ", ".join(nonpositive))
    if k_batch != batch:
        raise ValueError(f"q and k batch dimensions must match, got {batch} and {k_batch}")
    if k_head_dim != head_dim:
        raise ValueError(f"q and k head dimensions must match, got {head_dim} and {k_head_dim}")
    if per_head_shape != (batch, seqlen_q, num_query_heads):
        raise ValueError(f"{per_head_name} shape must be {(batch, seqlen_q, num_query_heads)}, " f"got {per_head_shape}")
    if (topk_batch, topk_seqlen_q) != (batch, seqlen_q):
        raise ValueError(
            "topk_indices leading dimensions must match q's batch and sequence " f"dimensions {(batch, seqlen_q)}, got {(topk_batch, topk_seqlen_q)}"
        )

    if qhead_per_kv_head is None:
        qhead_per_kv_head = num_query_heads
    if qhead_per_kv_head != num_query_heads:
        raise ValueError("qhead_per_kv_head must equal H_q for the MQA sparse score kernel, " f"got {qhead_per_kv_head} and H_q={num_query_heads}")

    if topk_length is not None:
        if not hasattr(topk_length, "shape") or not hasattr(topk_length, "dtype"):
            raise TypeError("topk_length must be a JAX array with shape and dtype metadata")
        if len(topk_length.shape) != 2:
            raise ValueError(f"topk_length must have rank 2, got shape {topk_length.shape}")
        topk_length_shape = tuple(topk_length.shape)
        require_dtype("topk_length.dtype", topk_length, (jnp.int32,))
        if topk_length_shape != (batch, seqlen_q):
            raise ValueError(f"topk_length shape must be {(batch, seqlen_q)}, " f"got {topk_length_shape}")

    # Keep an explicit length on the compact path. The Torch-only optimization
    # that sometimes drops it assumes every tail index was already set to -1;
    # the JAX contract treats topk_length itself as authoritative.
    config = resolve_sparse_score_kernel_config(
        score_type=score_type,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=topk,
        have_topk_length=topk_length is not None,
    )

    launcher = _make_launcher(
        score_type=score_type,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=topk,
        m_block_size=config.m_block_size,
        n_block_size=config.n_block_size,
        k_block_size=config.k_block_size,
        kv_stage=config.kv_stage,
        have_topk_length=config.have_topk_length,
        topk_in_smem=config.topk_in_smem,
        topk_indices_global=topk_indices_global,
        softmax_scale=softmax_scale,
    )

    inputs = (q, k, per_head, topk_indices)
    workspaces = ()
    if topk_length is not None:
        inputs += (topk_length,)
    else:
        # The SM100 kernel has a mandatory tensor argument even when its static
        # ``have_topk_length`` mode is false. It never reads this dummy buffer.
        workspaces = (BufferSpec("topk_length_workspace", (1, 1), jnp.int32),)

    (out,) = call_cutedsl(
        launcher,
        inputs,
        outputs=(BufferSpec(output_name, (batch, seqlen_q, topk), jnp.float32),),
        workspaces=workspaces,
        use_static_tensors=True,
    )
    return out


def sparse_indexer_score_recompute_wrapper(
    q_indexer: Any,
    k_indexer: Any,
    weights: Any,
    topk_indices: Any,
    qhead_per_kv_head: Optional[int] = None,
    topk_length: Optional[Any] = None,
    topk_indices_global: bool = False,
) -> SparseIndexerScoreRecomputeResult:
    """Recompute normalized sparse indexer scores with the SM100 kernel.

    Inputs have shapes ``(B, S_q, H_q, D)``, ``(B, S_k, D)``,
    ``(B, S_q, H_q)``, and ``(B, S_q, topk)``. The first three tensors use
    ``bfloat16`` and ``topk_indices`` uses ``int32``. ``topk_length``, when
    present, is an ``int32`` tensor of shape ``(B, S_q)``. The returned
    ``predict`` tensor has shape ``(B, S_q, topk)`` and dtype ``float32``.

    This API supports fixed-shape SM100 execution. Configuration and
    ``topk_indices_global`` are compile-time values.
    """

    predict = _sparse_score_recompute(
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        score_type="indexer",
        output_name="predict",
        per_head_name="weights",
        per_head_dtype=jnp.bfloat16,
        softmax_scale=1.0,
        qhead_per_kv_head=qhead_per_kv_head,
        topk_length=topk_length,
        topk_indices_global=topk_indices_global,
    )
    return SparseIndexerScoreRecomputeResult(predict=predict)


def sparse_attn_score_recompute_wrapper(
    q_attn: Any,
    k_attn: Any,
    lse: Any,
    topk_indices: Any,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    topk_length: Optional[Any] = None,
    topk_indices_global: bool = False,
) -> SparseAttnScoreRecomputeResult:
    """Recompute normalized sparse attention scores with the SM100 kernel.

    ``q_attn`` and ``k_attn`` use ``bfloat16``; ``lse`` uses ``float32``;
    indices and optional lengths use ``int32``. Shapes match
    :func:`sparse_indexer_score_recompute_wrapper`, and the returned ``target``
    is ``(B, S_q, topk)`` ``float32``. ``softmax_scale`` and all configuration
    values are compile-time values.
    """

    target = _sparse_score_recompute(
        q_attn,
        k_attn,
        lse,
        topk_indices,
        score_type="attention",
        output_name="target",
        per_head_name="lse",
        per_head_dtype=jnp.float32,
        softmax_scale=softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        topk_length=topk_length,
        topk_indices_global=topk_indices_global,
    )
    return SparseAttnScoreRecomputeResult(target=target)


def dense_indexer_score_recompute_wrapper(
    q: Any,
    k: Any,
    weights: Any,
    qhead_per_kv_head: Optional[int] = None,
    sm_scale: float = 1.0,
    ratio: int = 1,
    q_causal_offsets: Any | None = None,
) -> DenseScoreRecomputeResult:
    """Compute dense indexer scores and their log-sum-exp denominator.

    This binding currently supports fixed-shape SM100 BSHD tensors. ``q`` and
    ``k`` have shapes ``(B, S_q, H_q, D)`` and ``(B, S_k, H_kv, D)``;
    ``weights`` has shape ``(B, S_q, H_q)``. All inputs use ``bfloat16``.
    The returned ``out`` and ``denom`` arrays use ``float32`` and have shapes
    ``(B, S_q, S_k)`` and ``(B, S_q)``.
    """

    return _dense_score_recompute(
        q,
        k,
        weights,
        score_type="indexer",
        per_head_name="weights",
        per_head_dtype=jnp.bfloat16,
        scale=sm_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
        q_causal_offsets=q_causal_offsets,
    )


def dense_attn_score_recompute_wrapper(
    q: Any,
    k: Any,
    lse: Any,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    ratio: int = 1,
    q_causal_offsets: Any | None = None,
) -> DenseScoreRecomputeResult:
    """Compute dense attention scores and their L1 denominator.

    This binding currently supports fixed-shape SM100 BSHD tensors. ``q`` and
    ``k`` use ``bfloat16`` and ``lse`` uses ``float32``. The returned ``out``
    and ``denom`` arrays use ``float32`` and have shapes ``(B, S_q, S_k)`` and
    ``(B, S_q)``.
    """

    return _dense_score_recompute(
        q,
        k,
        lse,
        score_type="attention",
        per_head_name="lse",
        per_head_dtype=jnp.float32,
        scale=softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
        q_causal_offsets=q_causal_offsets,
    )


__all__ = [
    "DenseScoreRecomputeResult",
    "SparseAttnScoreRecomputeResult",
    "SparseIndexerScoreRecomputeResult",
    "dense_attn_score_recompute_wrapper",
    "dense_indexer_score_recompute_wrapper",
    "sparse_attn_score_recompute_wrapper",
    "sparse_indexer_score_recompute_wrapper",
]
