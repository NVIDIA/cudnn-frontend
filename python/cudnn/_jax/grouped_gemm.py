# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared JAX metadata validation for contiguous grouped GEMMs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from ..gemm_validation import block_scale_shape, require_shape
from .api_base import require_dtype
from .gemm import require_array


def require_grouped_gemm_inputs(
    a_tensor: Any,
    b_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    *,
    max_experts: int,
    valid_ab_dtypes: Iterable[Any] | None = None,
) -> tuple[int, int, int, int, Any]:
    """Validate the common contiguous grouped-GEMM inputs.

    A has logical shape ``(M, K, 1)`` while B has ``(N, K, L)``. The
    grouped dimension is flattened into A's M dimension and described at
    runtime by ``padded_offsets``.
    """

    a_shape = require_array("a_tensor", a_tensor, 3)
    b_shape = require_array("b_tensor", b_tensor, 3)
    m, k, a_batch = a_shape
    n, b_k, experts = b_shape
    if a_batch != 1:
        raise ValueError(f"a_tensor must have shape (M, K, 1), got {a_shape}")
    if b_k != k:
        raise ValueError(f"a_tensor and b_tensor must have matching K dimensions, got {a_shape} and {b_shape}")
    dimensions = {"M": m, "N": n, "K": k, "L": experts}
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("Grouped GEMM dimensions must be positive, got " + ", ".join(nonpositive))
    if experts > max_experts:
        raise ValueError(f"The number of experts must be at most {max_experts}, got {experts}")

    if valid_ab_dtypes is None:
        valid_ab_dtypes = (jnp.float8_e4m3fn, jnp.float8_e5m2)
    a_dtype = require_dtype("a_tensor.dtype", a_tensor, valid_ab_dtypes)
    require_dtype("b_tensor.dtype", b_tensor, (a_dtype,))

    offsets_shape = require_array("padded_offsets", padded_offsets, 1)
    require_shape("padded_offsets", offsets_shape, (experts,))
    require_dtype("padded_offsets.dtype", padded_offsets, (jnp.int32,))

    alpha_shape = require_array("alpha_tensor", alpha_tensor, 1)
    require_shape("alpha_tensor", alpha_shape, (experts,))
    require_dtype("alpha_tensor.dtype", alpha_tensor, (jnp.float32,))
    return m, n, k, experts, a_dtype


def require_grouped_fp8_scales(
    sfa_tensor: Any,
    sfb_tensor: Any,
    *,
    m: int,
    n: int,
    k: int,
    experts: int,
    sf_vec_size: int,
) -> Any:
    """Validate the E8M0 scale-factor ABI for contiguous grouped GEMMs."""

    return require_grouped_block_scales(
        sfa_tensor,
        sfb_tensor,
        m=m,
        n=n,
        k=k,
        experts=experts,
        sf_vec_size=sf_vec_size,
        valid_dtypes=(jnp.float8_e8m0fnu,),
    )


def require_grouped_block_scales(
    sfa_tensor: Any,
    sfb_tensor: Any,
    *,
    m: int,
    n: int,
    k: int,
    experts: int,
    sf_vec_size: int,
    valid_dtypes: Iterable[Any],
) -> Any:
    """Validate native block-scale shapes and a caller-supplied dtype set."""

    sfa_shape = require_array("sfa_tensor", sfa_tensor, 6)
    sfb_shape = require_array("sfb_tensor", sfb_tensor, 6)
    require_shape("sfa_tensor", sfa_shape, block_scale_shape(m, k, 1, sf_vec_size))
    require_shape("sfb_tensor", sfb_shape, block_scale_shape(n, k, experts, sf_vec_size))
    sf_dtype = require_dtype("sfa_tensor.dtype", sfa_tensor, valid_dtypes)
    require_dtype("sfb_tensor.dtype", sfb_tensor, (sf_dtype,))
    return sf_dtype


def require_grouped_vector(
    name: str,
    tensor: Any,
    *,
    length: int,
    dtype: Any = None,
) -> Any:
    """Validate a contiguous one-dimensional grouped-GEMM tensor."""

    if dtype is None:
        dtype = jnp.float32
    shape = require_array(name, tensor, 1)
    require_shape(name, shape, (length,))
    return require_dtype(f"{name}.dtype", tensor, (dtype,))


def require_grouped_probability(name: str, tensor: Any, *, m: int) -> None:
    """Validate a per-row FP32 probability tensor."""

    shape = require_array(name, tensor, 3)
    require_shape(name, shape, (m, 1, 1))
    require_dtype(f"{name}.dtype", tensor, (jnp.float32,))


def grouped_bias_tensor_spec() -> TensorSpec:
    """Describe a logical ``(N, L)`` bias with contiguous N mode."""

    return TensorSpec(layout=(0, 1), mode=(0, 1))


def grouped_workspace_tensor_spec() -> TensorSpec:
    """Describe the 128-byte-aligned grouped-kernel workspace."""

    return TensorSpec(ptr_assumed_align=128)


__all__ = [
    "grouped_bias_tensor_spec",
    "grouped_workspace_tensor_spec",
    "require_grouped_block_scales",
    "require_grouped_fp8_scales",
    "require_grouped_gemm_inputs",
    "require_grouped_probability",
    "require_grouped_vector",
]
