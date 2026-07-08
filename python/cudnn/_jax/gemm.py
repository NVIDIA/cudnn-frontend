# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared JAX tensor metadata for FE-OSS GEMM operations."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from ..gemm_validation import (
    require_block_scale_shapes,
    require_contiguous_alignment,
    require_gemm_shapes,
)
from .api_base import JaxTensorDesc, as_dtype, require_array

GEMM_A_LAYOUTS = ("LMK", "LKM")
GEMM_B_LAYOUTS = ("LNK", "LKN")
GEMM_C_LAYOUTS = ("LMN", "LNM")


def require_layout(name: str, value: str, supported: tuple[str, ...]) -> str:
    """Return a canonical public axis-order string from a supported set."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    value = value.upper()
    if value not in supported:
        choices = ", ".join(repr(item) for item in supported)
        raise ValueError(f"{name} must be one of {{{choices}}}, got {value!r}")
    return value


def _gemm_tensor_spec(
    name: str,
    layout: str,
    *,
    kernel_modes: str,
    supported: tuple[str, ...],
) -> TensorSpec:
    """Map a compact row-major public layout to canonical kernel modes."""

    layout = require_layout(name, layout, supported)
    mode = tuple(layout.index(dim) for dim in kernel_modes)
    return TensorSpec(
        layout=(2, 1, 0),
        mode=mode,
    )


def as_gemm_tensor_desc(
    name: str,
    value: Any,
    tensor_spec: TensorSpec,
) -> JaxTensorDesc:
    """Return kernel-visible GEMM metadata for an array or existing descriptor."""

    if isinstance(value, JaxTensorDesc):
        expected_layout = tuple(tensor_spec.layout)
        expected_mode = tuple(tensor_spec.mode)
        if value.layout != expected_layout or value.mode != expected_mode:
            raise ValueError(
                f"{name} descriptor layout does not match the requested GEMM layout: "
                f"expected layout={expected_layout}, mode={expected_mode}; "
                f"got layout={value.layout}, mode={value.mode}"
            )
        return value
    return JaxTensorDesc.from_value(
        value,
        tensor_spec=tensor_spec,
        name=name,
    )


def require_gemm_inputs(
    a_tensor: Any,
    b_tensor: Any,
) -> tuple[int, int, int, int, Any]:
    """Validate common dense GEMM metadata and return ``M, N, K, L, dtype``."""

    a_shape = require_array(a_tensor, name="a_tensor", rank=3)
    a_dtype = as_dtype(a_tensor)
    b_shape = require_array(b_tensor, name="b_tensor", rank=3, dtype=a_dtype)
    m, n, k, batch = require_gemm_shapes(a_shape, b_shape)
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
    sfa_shape = require_array(
        sfa_tensor,
        name="sfa_tensor",
        rank=6,
        dtype=jnp.float8_e8m0fnu,
    )
    sfb_shape = require_array(
        sfb_tensor,
        name="sfb_tensor",
        rank=6,
        dtype=jnp.float8_e8m0fnu,
    )
    require_block_scale_shapes(
        sfa_shape,
        sfb_shape,
        m=m,
        n=n,
        k=k,
        batch=batch,
        sf_vec_size=sf_vec_size,
    )


def require_16_byte_extent(name: str, elements: int, dtype: Any) -> None:
    """Require the kernel's contiguous mode to span a multiple of 16 bytes."""

    dtype = jnp.dtype(dtype)
    require_contiguous_alignment(name, elements, dtype.itemsize * 8)


def gemm_a_tensor_spec(layout: str) -> TensorSpec:
    """Describe public A layout ``LMK`` or ``LKM`` as kernel ``(M,K,L)``."""

    return _gemm_tensor_spec(
        "a_layout",
        layout,
        kernel_modes="MKL",
        supported=GEMM_A_LAYOUTS,
    )


def gemm_b_tensor_spec(layout: str) -> TensorSpec:
    """Describe public B layout ``LNK`` or ``LKN`` as kernel ``(N,K,L)``."""

    return _gemm_tensor_spec(
        "b_layout",
        layout,
        kernel_modes="NKL",
        supported=GEMM_B_LAYOUTS,
    )


def gemm_c_tensor_spec(layout: str, *, name: str = "c_layout") -> TensorSpec:
    """Describe public C/D layout ``LMN`` or ``LNM`` as kernel ``(M,N,L)``."""

    return _gemm_tensor_spec(
        name,
        layout,
        kernel_modes="MNL",
        supported=GEMM_C_LAYOUTS,
    )


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
    "GEMM_A_LAYOUTS",
    "GEMM_B_LAYOUTS",
    "GEMM_C_LAYOUTS",
    "as_gemm_tensor_desc",
    "block_scale_tensor_spec",
    "gemm_a_tensor_spec",
    "gemm_b_tensor_spec",
    "gemm_c_tensor_spec",
    "probability_tensor_spec",
    "require_fp8_block_scales",
    "require_gemm_inputs",
    "require_16_byte_extent",
    "require_layout",
]
