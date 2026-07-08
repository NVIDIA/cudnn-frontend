# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared JAX metadata validation for contiguous grouped GEMMs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from ..gemm_validation import block_scale_shape
from .api_base import as_dtype, require_array


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

    if valid_ab_dtypes is None:
        valid_ab_dtypes = (jnp.float8_e4m3fn, jnp.float8_e5m2)
    a_shape = require_array(
        a_tensor,
        name="a_tensor",
        rank=3,
        dtype=valid_ab_dtypes,
    )
    a_dtype = as_dtype(a_tensor)
    b_shape = require_array(b_tensor, name="b_tensor", rank=3, dtype=a_dtype)
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

    require_array(
        padded_offsets,
        name="padded_offsets",
        shape=(experts,),
        dtype=jnp.int32,
    )
    require_array(
        alpha_tensor,
        name="alpha_tensor",
        shape=(experts,),
        dtype=jnp.float32,
    )
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

    require_array(
        sfa_tensor,
        name="sfa_tensor",
        shape=block_scale_shape(m, k, 1, sf_vec_size),
        dtype=valid_dtypes,
    )
    sf_dtype = as_dtype(sfa_tensor)
    require_array(
        sfb_tensor,
        name="sfb_tensor",
        shape=block_scale_shape(n, k, experts, sf_vec_size),
        dtype=sf_dtype,
    )
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
    require_array(tensor, name=name, shape=(length,), dtype=dtype)
    return as_dtype(tensor)


def require_grouped_probability(name: str, tensor: Any, *, m: int) -> None:
    """Validate a per-row FP32 probability tensor."""

    require_array(
        tensor,
        name=name,
        shape=(m, 1, 1),
        dtype=jnp.float32,
    )


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
