# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared JAX tensor metadata for FE-OSS GEMM operations."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from .validation import require_dtype


def ceil_div(value: int, divisor: int) -> int:
    """Return ``ceil(value / divisor)`` for positive integers."""

    return (value + divisor - 1) // divisor


def require_array(name: str, value: Any, rank: int) -> tuple[Any, ...]:
    """Return an array-like value's shape after checking its rank."""

    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must have shape and dtype metadata")
    shape = tuple(value.shape)
    if len(shape) != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {shape}")
    return shape


def require_gemm_inputs(
    a_tensor: Any,
    b_tensor: Any,
) -> tuple[int, int, int, int, Any]:
    """Validate common dense GEMM metadata and return ``M, N, K, L, dtype``."""

    a_shape = require_array("a_tensor", a_tensor, 3)
    b_shape = require_array("b_tensor", b_tensor, 3)
    m, k, batch = a_shape
    n, b_k, b_batch = b_shape
    if b_k != k or b_batch != batch:
        raise ValueError("a_tensor and b_tensor must have matching K and L dimensions, " f"got {a_shape} and {b_shape}")
    dimensions = {"M": m, "N": n, "K": k, "L": batch}
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("GEMM dimensions must be positive, got " + ", ".join(nonpositive))
    a_dtype = jnp.dtype(a_tensor.dtype)
    b_dtype = jnp.dtype(b_tensor.dtype)
    if b_dtype != a_dtype:
        raise ValueError(f"a_tensor and b_tensor dtypes must match, got {a_dtype} and {b_dtype}")
    return m, n, k, batch, a_dtype


def require_fp8_block_scales(
    sfa_tensor: Any,
    sfb_tensor: Any,
    *,
    m: int,
    n: int,
    k: int,
    batch: int,
    sf_vec_size: int,
) -> None:
    """Validate the native MXFP8 scale-factor ABI used by dense kernels."""

    if sf_vec_size != 32:
        raise NotImplementedError("The JAX MXFP8 path requires sf_vec_size=32, " f"got {sf_vec_size}")
    sfa_shape = require_array("sfa_tensor", sfa_tensor, 6)
    sfb_shape = require_array("sfb_tensor", sfb_tensor, 6)
    scale_k_tiles = ceil_div(ceil_div(k, sf_vec_size), 4)
    expected_sfa_shape = (32, 4, ceil_div(m, 128), 4, scale_k_tiles, batch)
    expected_sfb_shape = (32, 4, ceil_div(n, 128), 4, scale_k_tiles, batch)
    if sfa_shape != expected_sfa_shape:
        raise ValueError(f"sfa_tensor must have shape {expected_sfa_shape}, got {sfa_shape}")
    if sfb_shape != expected_sfb_shape:
        raise ValueError(f"sfb_tensor must have shape {expected_sfb_shape}, got {sfb_shape}")
    require_dtype("sfa_tensor.dtype", sfa_tensor, (jnp.float8_e8m0fnu,))
    require_dtype("sfb_tensor.dtype", sfb_tensor, (jnp.float8_e8m0fnu,))


def require_16_byte_extent(name: str, elements: int, dtype: Any) -> None:
    """Require the kernel's contiguous mode to span a multiple of 16 bytes."""

    dtype = jnp.dtype(dtype)
    if elements * dtype.itemsize % 16:
        raise ValueError(f"{name}'s contiguous extent must be 16-byte aligned, got " f"{elements} elements of dtype {dtype}")


def gemm_a_tensor_spec(major: str) -> TensorSpec:
    """Describe a logical ``(M, K, L)`` tensor with the requested major mode."""

    try:
        layout = {"m": (0, 1, 2), "k": (1, 0, 2)}[major]
    except KeyError:
        raise ValueError(f"a_major must be either 'm' or 'k', got {major!r}") from None
    return TensorSpec(layout=layout, mode=(0, 1, 2))


def gemm_b_tensor_spec(major: str) -> TensorSpec:
    """Describe a logical ``(N, K, L)`` tensor with the requested major mode."""

    try:
        layout = {"n": (0, 1, 2), "k": (1, 0, 2)}[major]
    except KeyError:
        raise ValueError(f"b_major must be either 'n' or 'k', got {major!r}") from None
    return TensorSpec(layout=layout, mode=(0, 1, 2))


def gemm_c_tensor_spec(major: str) -> TensorSpec:
    """Describe a logical ``(M, N, L)`` tensor with the requested major mode."""

    try:
        layout = {"m": (0, 1, 2), "n": (1, 0, 2)}[major]
    except KeyError:
        raise ValueError(f"c_major must be either 'm' or 'n', got {major!r}") from None
    return TensorSpec(layout=layout, mode=(0, 1, 2))


def block_scale_tensor_spec() -> TensorSpec:
    """Describe the compact six-dimensional block-scale atom layout."""

    return TensorSpec(
        layout=(2, 1, 4, 0, 3, 5),
        mode=(0, 1, 2, 3, 4, 5),
    )


def probability_tensor_spec() -> TensorSpec:
    """Describe a logical ``(M, 1, L)`` tensor with contiguous ``M`` mode."""

    return TensorSpec(layout=(0, 1, 2), mode=(0, 1, 2))


__all__ = [
    "block_scale_tensor_spec",
    "ceil_div",
    "gemm_a_tensor_spec",
    "gemm_b_tensor_spec",
    "gemm_c_tensor_spec",
    "probability_tensor_spec",
    "require_array",
    "require_fp8_block_scales",
    "require_gemm_inputs",
    "require_16_byte_extent",
]
