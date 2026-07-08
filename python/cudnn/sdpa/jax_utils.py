# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared fixed-shape metadata and configuration for JAX SDPA."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from .. import data_type
from .._jax import JaxApiBase, JaxTensorDesc
from .._jax.datatypes import normalize_jax_dtype
from .._jax.layout import mode_from_layout, stride_order_to_public, to_public_axes

FIXED_LAYOUTS = ("BHSD", "BSHD")
KERNEL_AXES = "BSHD"
KERNEL_STRIDE_ORDER = (3, 2, 1, 0)


def normalize_sdpa_layout(layout: str | None, rank: int) -> str:
    """Normalize the public fixed/packed axis order for an SDPA call."""

    if layout is None:
        if rank == 4:
            return "BHSD"
        if rank == 3:
            return "THD"
        raise ValueError(f"SDPA layout cannot be inferred from rank {rank}")
    if not isinstance(layout, str):
        raise TypeError(f"layout must be a string or None, got {layout!r}")
    normalized = "".join(
        character for character in layout.upper() if character.isalpha()
    )
    if normalized not in (*FIXED_LAYOUTS, "THD"):
        raise ValueError(f"layout must be BHSD, BSHD, or THD, got {layout!r}")
    expected_rank = 3 if normalized == "THD" else 4
    if rank != expected_rank:
        raise ValueError(
            f"{normalized} layout requires rank-{expected_rank} tensors, got rank {rank}"
        )
    return normalized


def fixed_data_mode(layout: str) -> tuple[int, ...]:
    if layout not in FIXED_LAYOUTS:
        raise ValueError(f"fixed layout must be one of {FIXED_LAYOUTS}, got {layout!r}")
    return mode_from_layout(layout, kernel_axes=KERNEL_AXES)


def describe_fixed_data(
    sample: Any,
    name: str,
    *,
    layout: str,
    init_value: bool | int | float | None = None,
) -> JaxTensorDesc:
    """Describe fixed data with the BSHD layout hard-coded by the kernel."""

    mode = fixed_data_mode(layout)

    return JaxApiBase._to_tensor_desc(
        sample,
        name,
        mode=mode,
        public_stride_order=stride_order_to_public(KERNEL_STRIDE_ORDER, mode),
        init_value=init_value,
    )


def make_fixed_output(
    public_shape: tuple[int, ...],
    dtype: Any,
    name: str,
    *,
    layout: str,
    init_value: bool | int | float | None = None,
) -> JaxTensorDesc:
    return describe_fixed_data(
        jax.ShapeDtypeStruct(public_shape, dtype),
        name,
        layout=layout,
        init_value=init_value,
    )


def require_fixed_qkv(
    q_desc: JaxTensorDesc,
    k_desc: JaxTensorDesc,
    v_desc: JaxTensorDesc,
) -> tuple[int, int, int, int, int, int]:
    for desc in (q_desc, k_desc, v_desc):
        if desc.ndim != 4:
            raise ValueError(
                f"{desc.name} must have rank 4 (B, H, S, D), got {desc.shape}"
            )
    if q_desc.cudnn_dtype not in (data_type.HALF, data_type.BFLOAT16):
        raise ValueError(
            f"SDPA requires float16 or bfloat16 inputs, got {q_desc.dtype}"
        )
    if (
        k_desc.cudnn_dtype != q_desc.cudnn_dtype
        or v_desc.cudnn_dtype != q_desc.cudnn_dtype
    ):
        raise ValueError("Q, K, and V must have the same dtype")

    logical_bhsd_mode = mode_from_layout("BHSD", kernel_axes=KERNEL_AXES)
    q_shape = to_public_axes(q_desc.shape, logical_bhsd_mode)
    k_shape = to_public_axes(k_desc.shape, logical_bhsd_mode)
    v_shape = to_public_axes(v_desc.shape, logical_bhsd_mode)
    batch, num_query_heads, seqlen_q, head_dim = q_shape
    k_batch, num_kv_heads, seqlen_k, k_head_dim = k_shape
    v_batch, num_value_heads, v_seqlen, value_dim = v_shape
    dimensions = (batch, num_query_heads, num_kv_heads, seqlen_q, seqlen_k, head_dim)
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"SDPA dimensions must be positive, got {dimensions}")
    if (k_batch, v_batch) != (batch, batch):
        raise ValueError("Q, K, and V batch dimensions must match")
    if (num_value_heads, v_seqlen) != (num_kv_heads, seqlen_k):
        raise ValueError("K and V head and sequence dimensions must match")
    if (k_head_dim, value_dim) != (head_dim, head_dim):
        raise ValueError("Q, K, and V head dimensions must match")
    if head_dim != 256:
        raise ValueError(f"head dimension must be 256, got {head_dim}")
    if num_query_heads % num_kv_heads:
        raise ValueError(
            f"H_q ({num_query_heads}) must be divisible by H_kv ({num_kv_heads})"
        )
    return dimensions


def require_float32_dtype(value: Any | None, name: str) -> Any:
    dtype = normalize_jax_dtype(value, jnp.float32, name)
    if dtype != jnp.dtype(jnp.float32):
        raise ValueError(f"{name} must be float32, got {dtype}")
    return dtype


def resolve_sdpa_config(
    *,
    seqlen_q: int,
    seqlen_k: int,
    tile_extent: int,
    is_causal: bool,
    window_size: tuple[int, int],
    scale_softmax: float | None,
) -> tuple[float, int, int, str]:
    if len(window_size) != 2:
        raise ValueError(f"window_size must contain two values, got {window_size}")
    window_size_left, window_size_right = window_size
    if is_causal:
        window_size_right = 0
    elif (window_size_left, window_size_right) != (-1, -1):
        raise NotImplementedError(
            f"window_size must be (-1, -1) for non-causal mode, got {window_size}"
        )
    if window_size_left >= seqlen_k - 1:
        raise ValueError(
            f"window_size_left must be less than S_kv - 1 ({seqlen_k - 1}), got {window_size_left}"
        )
    if window_size_right >= seqlen_q - 1:
        raise ValueError(
            f"window_size_right must be less than S_q - 1 ({seqlen_q - 1}), got {window_size_right}"
        )

    scale = (
        1.0 / math.sqrt(256)
        if scale_softmax is None or scale_softmax == 0.0
        else float(scale_softmax)
    )
    mask_kind = "residual" if not is_causal and tile_extent % 128 else "window"
    return scale, window_size_left, window_size_right, mask_kind


__all__ = [
    "FIXED_LAYOUTS",
    "describe_fixed_data",
    "fixed_data_mode",
    "make_fixed_output",
    "normalize_sdpa_layout",
    "require_fixed_qkv",
    "require_float32_dtype",
    "resolve_sdpa_config",
]
