# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared validation and layout helpers for the JAX SDPA APIs."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from .._jax.api_base import require_dtype


def bhsd_tensor_spec() -> TensorSpec:
    """Present a logical BHSD JAX array to a CuTe kernel as compact BSHD.

    The public shape remains ``(B, H, S, D)``. XLA lays out the custom-call
    buffer with H inside S, matching the existing Torch API's transposed view,
    and ``mode`` presents the kernel with ``(B, S, H, D)`` modes.
    """

    return TensorSpec(
        layout=(3, 1, 2, 0),
        mode=(0, 2, 1, 3),
    )


def require_bhsd_qkv(q: Any, k: Any, v: Any) -> tuple[int, int, int, int, int, int, Any]:
    """Validate fixed-shape BHSD Q/K/V tensors and return their dimensions."""

    for name, value in (("q_tensor", q), ("k_tensor", k), ("v_tensor", v)):
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{name} must be a JAX array with shape and dtype metadata")
        if len(value.shape) != 4:
            raise ValueError(f"{name} must have rank 4 (B, H, S, D), got shape {value.shape}")

    batch, num_query_heads, seqlen_q, head_dim = tuple(q.shape)
    k_batch, num_kv_heads, seqlen_k, k_head_dim = tuple(k.shape)
    v_batch, num_value_heads, v_seqlen, value_dim = tuple(v.shape)

    dimensions = {
        "batch": batch,
        "H_q": num_query_heads,
        "H_kv": num_kv_heads,
        "S_q": seqlen_q,
        "S_kv": seqlen_k,
        "D": head_dim,
    }
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("SDPA dimensions must be positive, got " + ", ".join(nonpositive))
    if (k_batch, v_batch) != (batch, batch):
        raise ValueError("q_tensor, k_tensor, and v_tensor batch dimensions must match, " f"got {batch}, {k_batch}, and {v_batch}")
    if (num_value_heads, v_seqlen) != (num_kv_heads, seqlen_k):
        raise ValueError("k_tensor and v_tensor head and sequence dimensions must match, " f"got {(num_kv_heads, seqlen_k)} and {(num_value_heads, v_seqlen)}")
    if (k_head_dim, value_dim) != (head_dim, head_dim):
        raise ValueError("q_tensor, k_tensor, and v_tensor head dimensions must match, " f"got {head_dim}, {k_head_dim}, and {value_dim}")
    if head_dim != 256:
        raise ValueError(f"head dimension must be 256, got {head_dim}")
    if num_query_heads % num_kv_heads:
        raise ValueError(f"H_q ({num_query_heads}) must be divisible by H_kv ({num_kv_heads})")

    dtype = require_dtype("q_tensor.dtype", q, (jnp.float16, jnp.bfloat16))
    require_dtype("k_tensor.dtype", k, (dtype,))
    require_dtype("v_tensor.dtype", v, (dtype,))
    return (
        batch,
        num_query_heads,
        num_kv_heads,
        seqlen_q,
        seqlen_k,
        head_dim,
        dtype,
    )


def require_array(name: str, value: Any, shape: tuple[int, ...], dtype: Any) -> None:
    """Require a JAX array to match an expected shape and dtype."""

    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be a JAX array with shape and dtype metadata")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}")
    require_dtype(f"{name}.dtype", value, (dtype,))


def resolve_sdpa_config(
    *,
    seqlen_q: int,
    seqlen_k: int,
    tile_extent: int,
    is_causal: bool,
    window_size: tuple[int, int],
    scale_softmax: float | None,
) -> tuple[float, int, int, str]:
    """Resolve shared mask and scaling configuration for fixed-shape SDPA."""

    if len(window_size) != 2:
        raise ValueError(f"window_size must contain two values, got {window_size}")
    window_size_left, window_size_right = window_size
    if is_causal:
        window_size_right = 0
    elif (window_size_left, window_size_right) != (-1, -1):
        raise NotImplementedError("window_size must be (-1, -1) for non-causal mode, got " f"{(window_size_left, window_size_right)}")

    if window_size_left >= seqlen_k - 1:
        raise ValueError(f"window_size_left must be less than S_kv - 1 ({seqlen_k - 1}), " f"got {window_size_left}")
    if window_size_right >= seqlen_q - 1:
        raise ValueError(f"window_size_right must be less than S_q - 1 ({seqlen_q - 1}), " f"got {window_size_right}")

    resolved_scale = 1.0 / math.sqrt(256) if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
    mask_kind = "window"
    if not is_causal and tile_extent % 128:
        mask_kind = "residual"
    return resolved_scale, window_size_left, window_size_right, mask_kind


__all__ = [
    "bhsd_tensor_spec",
    "require_array",
    "require_bhsd_qkv",
    "resolve_sdpa_config",
]
