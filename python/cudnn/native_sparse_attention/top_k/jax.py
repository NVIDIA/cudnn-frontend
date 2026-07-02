# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed-shape SM100 NSA top-K reduction."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, NamedTuple

import jax.numpy as jnp
from cutlass.jax import TensorSpec, jax_to_cutlass_dtype

from ..._jax.cutedsl import BufferSpec, call_cutedsl
from ..._jax.validation import require_dtype
from ..jax_utils import bhsd_storage_spec, require_bhsd_qkv


class TopKReductionResult(NamedTuple):
    """Functional outputs from :func:`topk_reduction_wrapper`."""

    topk_scores_tensor: Any
    topk_indices_tensor: Any


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    element_dtype: Any,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    k_value: int,
    selection_block_size: int,
    compress_stride: int,
    is_causal: bool,
    scale_softmax: float,
):
    from cutlass import Float32, Int32

    from .nsa_top_k_reduction_fwd import FineGrainedReductionQK

    kernel = FineGrainedReductionQK(
        element_dtype=element_dtype,
        acc_dtype=Float32,
        k_value=k_value,
        selection_block_size=selection_block_size,
        compress_block_sliding_stride=compress_stride,
        mma_tiler=(128, 128, head_dim),
        is_causal=is_causal,
    )
    problem_size = tuple(
        Int32(value)
        for value in (
            batch,
            seqlen_q,
            seqlen_k,
            num_query_heads,
            num_kv_heads,
            head_dim,
        )
    )

    def launch(stream, q, k, lse, topk_scores, topk_indices):
        kernel(
            problem_size,
            q,
            k,
            lse,
            topk_scores,
            topk_indices,
            Float32(scale_softmax * math.log2(math.e)),
            None,
            None,
            stream,
        )

    return launch


def _require_topk_config(
    *,
    k_value: int,
    selection_block_size: int,
    compress_stride: int,
    mma_tiler_mn: tuple[int, int],
) -> None:
    if mma_tiler_mn != (128, 128):
        raise ValueError(f"mma_tiler_mn must be (128, 128), got {mma_tiler_mn}")
    if selection_block_size <= 0 or compress_stride <= 0:
        raise ValueError("selection_block_size and compress_stride must be positive, got " f"{selection_block_size} and {compress_stride}")
    if selection_block_size % compress_stride:
        raise ValueError("selection_block_size must be divisible by compress_stride, got " f"{selection_block_size} and {compress_stride}")
    reduction_width = selection_block_size // compress_stride
    if reduction_width > mma_tiler_mn[1] or mma_tiler_mn[1] % reduction_width:
        raise ValueError("selection_block_size / compress_stride must divide the MMA N tile, " f"got {reduction_width} and {mma_tiler_mn[1]}")
    candidate_count = mma_tiler_mn[1] // reduction_width
    if k_value <= 0 or k_value % 4 or k_value > candidate_count:
        raise ValueError("k_value must be a positive multiple of 4 no greater than the " f"per-tile candidate count ({candidate_count}), got {k_value}")


def topk_reduction_wrapper(
    q_tensor: Any,
    k_tensor: Any,
    lse_tensor: Any,
    acc_dtype: Any = None,
    k_value: int = 16,
    selection_block_size: int = 64,
    compress_stride: int = 32,
    is_causal: bool = True,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    scale_softmax: float | None = None,
) -> TopKReductionResult:
    """Select top compressed-KV blocks for fixed-shape BHSD inputs on SM100.

    ``q_tensor`` and ``k_tensor`` have logical shapes ``(B, H, S, D)`` and a
    shared ``float16`` or ``bfloat16`` dtype. ``lse_tensor`` has shape
    ``(B, H_q, S_q)`` and dtype ``float32``. Outputs have logical shape
    ``(B, H_kv, S_q, k_value)`` and dtypes ``float32`` and ``int32``.

    Variable-length THD inputs are not part of this API. Configuration values
    must be static while tracing with :func:`jax.jit`.
    """

    (
        batch,
        num_query_heads,
        num_kv_heads,
        seqlen_q,
        seqlen_k,
        head_dim,
        input_dtype,
    ) = require_bhsd_qkv(q_tensor, k_tensor)
    require_dtype("acc_dtype", acc_dtype, (jnp.float32,), default=jnp.float32)
    if not hasattr(lse_tensor, "shape") or not hasattr(lse_tensor, "dtype"):
        raise TypeError("lse_tensor must have shape and dtype metadata")
    if tuple(lse_tensor.shape) != (batch, num_query_heads, seqlen_q):
        raise ValueError("lse_tensor must have shape " f"{(batch, num_query_heads, seqlen_q)}, got {lse_tensor.shape}")
    require_dtype("lse_tensor.dtype", lse_tensor, (jnp.float32,))
    _require_topk_config(
        k_value=int(k_value),
        selection_block_size=int(selection_block_size),
        compress_stride=int(compress_stride),
        mma_tiler_mn=mma_tiler_mn,
    )

    resolved_scale = 1.0 / math.sqrt(head_dim) if scale_softmax is None else float(scale_softmax)
    bhsd_spec = bhsd_storage_spec(present_as_bshd=False)
    lse_spec = TensorSpec(layout=(2, 1, 0), mode=(0, 1, 2))
    output_shape = (batch, num_kv_heads, seqlen_q, int(k_value))
    topk_scores, topk_indices = call_cutedsl(
        _make_launcher(
            element_dtype=jax_to_cutlass_dtype(input_dtype),
            batch=batch,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            k_value=int(k_value),
            selection_block_size=int(selection_block_size),
            compress_stride=int(compress_stride),
            is_causal=bool(is_causal),
            scale_softmax=resolved_scale,
        ),
        (q_tensor, k_tensor, lse_tensor),
        outputs=(
            BufferSpec(
                "topk_scores_tensor",
                output_shape,
                jnp.float32,
                tensor_spec=bhsd_spec,
                fill_value=float("-inf"),
            ),
            BufferSpec(
                "topk_indices_tensor",
                output_shape,
                jnp.int32,
                tensor_spec=bhsd_spec,
                fill_value=-1,
            ),
        ),
        input_specs=(bhsd_spec, bhsd_spec, lse_spec),
        use_static_tensors=True,
    )
    return TopKReductionResult(
        topk_scores_tensor=topk_scores,
        topk_indices_tensor=topk_indices,
    )


__all__ = ["TopKReductionResult", "topk_reduction_wrapper"]
