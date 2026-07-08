# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared tensor metadata and validation for JAX NSA adapters."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .. import data_type
from .._jax import JaxApiBase, JaxTensorDesc
from .._jax.datatypes import normalize_jax_dtype
from .._jax.layout import mode_from_layout, stride_order_to_public, to_public_axes

FIXED_LAYOUTS = ("BHSD", "BSHD")
BHS_TO_BSH_MODE = mode_from_layout("BHS", kernel_axes="BSH")


def normalize_attention_layout(
    layout: str | None,
    rank: int,
    *,
    allow_packed: bool = True,
) -> str:
    """Normalize a fixed BHSD/BSHD or packed THD public layout."""

    if layout is None:
        if rank == 4:
            return "BHSD"
        if allow_packed and rank == 3:
            return "THD"
        raise ValueError(f"attention layout cannot be inferred from rank {rank}")
    if not isinstance(layout, str):
        raise TypeError(f"layout must be a string or None, got {layout!r}")
    normalized = "".join(
        character for character in layout.upper() if character.isalpha()
    )
    allowed = (*FIXED_LAYOUTS, "THD") if allow_packed else FIXED_LAYOUTS
    if normalized not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"layout must be one of {choices}, got {layout!r}")
    expected_rank = 3 if normalized == "THD" else 4
    if rank != expected_rank:
        raise ValueError(
            f"{normalized} layout requires rank-{expected_rank} tensors, got rank {rank}"
        )
    return normalized


def fixed_data_mode(layout: str, *, kernel_axes: str) -> tuple[int, ...]:
    if layout not in FIXED_LAYOUTS:
        raise ValueError(f"fixed layout must be one of {FIXED_LAYOUTS}, got {layout!r}")
    return mode_from_layout(layout, kernel_axes=kernel_axes)


def describe_fixed_data(
    sample: Any,
    name: str,
    *,
    layout: str,
    kernel_axes: str,
    kernel_stride_order: tuple[int, ...] | None = None,
    init_value: bool | int | float | None = None,
) -> JaxTensorDesc:
    """Describe public fixed-shape metadata in canonical kernel axes."""

    mode = fixed_data_mode(layout, kernel_axes=kernel_axes)
    public_stride_order = (
        None
        if kernel_stride_order is None
        else stride_order_to_public(kernel_stride_order, mode)
    )
    return JaxApiBase._to_tensor_desc(
        sample,
        name,
        mode=mode,
        public_stride_order=public_stride_order,
        init_value=init_value,
    )


def describe_bhs_as_bsh(
    sample: Any,
    name: str,
    *,
    init_value: bool | int | float | None = None,
) -> JaxTensorDesc:
    """Describe public BHS metadata as the BSH modes consumed by a kernel."""

    return JaxApiBase._to_tensor_desc(
        sample,
        name,
        mode=BHS_TO_BSH_MODE,
        init_value=init_value,
    )


def make_fixed_output(
    public_shape: tuple[int, ...],
    dtype: Any,
    name: str,
    *,
    layout: str,
    kernel_axes: str,
    kernel_stride_order: tuple[int, ...] | None = None,
    init_value: bool | int | float | None = None,
) -> JaxTensorDesc:
    return describe_fixed_data(
        jax.ShapeDtypeStruct(public_shape, dtype),
        name,
        layout=layout,
        kernel_axes=kernel_axes,
        kernel_stride_order=kernel_stride_order,
        init_value=init_value,
    )


def require_fixed_qkv(
    q_desc: JaxTensorDesc,
    k_desc: JaxTensorDesc,
    v_desc: JaxTensorDesc | None = None,
    *,
    operation_name: str,
    head_dims: tuple[int, ...] = (32, 64, 128),
    kernel_axes: str = "BHSD",
    input_dtypes: tuple[data_type, ...] = (data_type.HALF, data_type.BFLOAT16),
) -> tuple[int, int, int, int, int, int]:
    """Validate fixed-shape public BHSD Q/K/V signatures."""

    for desc in (q_desc, k_desc, v_desc):
        if desc is not None and desc.ndim != 4:
            raise ValueError(
                f"{desc.name} must have rank 4 (B, H, S, D), got {desc.shape}"
            )
    if q_desc.cudnn_dtype not in input_dtypes:
        expected = ", ".join(str(dtype) for dtype in input_dtypes)
        raise ValueError(
            f"{operation_name} requires one of {{{expected}}}, got {q_desc.dtype}"
        )
    for desc in (k_desc, v_desc):
        if desc is not None and desc.cudnn_dtype != q_desc.cudnn_dtype:
            raise ValueError(f"{desc.name} must have the same dtype as {q_desc.name}")

    logical_bhsd_mode = mode_from_layout("BHSD", kernel_axes=kernel_axes)
    q_shape = to_public_axes(q_desc.shape, logical_bhsd_mode)
    k_shape = to_public_axes(k_desc.shape, logical_bhsd_mode)
    batch, num_query_heads, seqlen_q, head_dim = q_shape
    k_batch, num_kv_heads, seqlen_k, k_head_dim = k_shape
    dimensions = (batch, num_query_heads, num_kv_heads, seqlen_q, seqlen_k, head_dim)
    if any(value <= 0 for value in dimensions):
        raise ValueError(
            f"{operation_name} dimensions must be positive, got {dimensions}"
        )
    if k_batch != batch:
        raise ValueError(
            f"Q and K batch dimensions must match, got {batch} and {k_batch}"
        )
    if k_head_dim != head_dim:
        raise ValueError(
            f"Q and K head dimensions must match, got {head_dim} and {k_head_dim}"
        )
    if head_dim not in head_dims:
        expected = ", ".join(str(value) for value in head_dims)
        raise ValueError(f"head dimension must be one of {expected}, got {head_dim}")
    if num_query_heads % num_kv_heads:
        raise ValueError(
            f"H_q ({num_query_heads}) must be divisible by H_kv ({num_kv_heads})"
        )

    if v_desc is not None:
        v_batch, num_value_heads, v_seqlen, value_dim = to_public_axes(
            v_desc.shape, logical_bhsd_mode
        )
        if (v_batch, num_value_heads, v_seqlen) != (batch, num_kv_heads, seqlen_k):
            raise ValueError("K and V batch, head, and sequence dimensions must match")
        if value_dim != head_dim:
            raise ValueError(
                f"V head dimension must match Q/K ({head_dim}), got {value_dim}"
            )
    return dimensions


def normalize_supported_dtype(
    value: Any | None, default: Any, name: str, allowed: tuple[Any, ...]
) -> Any:
    dtype = normalize_jax_dtype(value, default, name)
    allowed_dtypes = tuple(jnp.dtype(item) for item in allowed)
    if dtype not in allowed_dtypes:
        expected = ", ".join(str(item) for item in allowed_dtypes)
        raise ValueError(f"{name} must be one of {expected}, got {dtype}")
    return dtype


__all__ = [
    "BHS_TO_BSH_MODE",
    "FIXED_LAYOUTS",
    "describe_bhs_as_bsh",
    "describe_fixed_data",
    "fixed_data_mode",
    "make_fixed_output",
    "normalize_attention_layout",
    "normalize_supported_dtype",
    "require_fixed_qkv",
]
