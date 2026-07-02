# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared validation and layout helpers for JAX NSA operations."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from .._jax.api_base import require_dtype


def bhsd_storage_spec(*, present_as_bshd: bool) -> TensorSpec:
    """Describe logical BHSD data backed by compact BSHD storage.

    NSA Torch tensors use logical ``(B, H, S, D)`` shapes with H inside S in
    physical memory. Some kernels consume that BHSD mode order directly while
    compression attention consumes a ``(B, S, H, D)`` view.
    """

    return TensorSpec(
        layout=(3, 1, 2, 0),
        mode=(0, 2, 1, 3) if present_as_bshd else (0, 1, 2, 3),
    )


def bhs_lse_as_bsh_spec() -> TensorSpec:
    """Present a public ``(B, H, S)`` LSE array as ``(B, S, H)``."""

    return TensorSpec(layout=(2, 1, 0), mode=(0, 2, 1))


def require_bhsd_qkv(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any | None = None,
) -> tuple[int, int, int, int, int, int, Any]:
    """Validate fixed-shape NSA BHSD inputs and return their dimensions."""

    arrays = [("q_tensor", q_tensor), ("k_tensor", k_tensor)]
    if v_tensor is not None:
        arrays.append(("v_tensor", v_tensor))
    for name, value in arrays:
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{name} must have shape and dtype metadata")
        if len(value.shape) != 4:
            raise ValueError(f"{name} must have rank 4 (B, H, S, D), got shape {value.shape}")

    batch, num_query_heads, seqlen_q, head_dim = tuple(q_tensor.shape)
    k_batch, num_kv_heads, seqlen_k, k_head_dim = tuple(k_tensor.shape)
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
        raise ValueError("NSA dimensions must be positive, got " + ", ".join(nonpositive))
    if k_batch != batch:
        raise ValueError(f"q_tensor and k_tensor batch dimensions must match, got {batch} and {k_batch}")
    if k_head_dim != head_dim:
        raise ValueError("q_tensor and k_tensor head dimensions must match, got " f"{head_dim} and {k_head_dim}")
    if head_dim not in (32, 64, 128):
        raise ValueError(f"head dimension must be 32, 64, or 128, got {head_dim}")
    if num_query_heads % num_kv_heads:
        raise ValueError(f"H_q ({num_query_heads}) must be divisible by H_kv ({num_kv_heads})")

    dtype = require_dtype(
        "q_tensor.dtype",
        q_tensor,
        (jnp.float16, jnp.bfloat16),
    )
    require_dtype("k_tensor.dtype", k_tensor, (dtype,))

    if v_tensor is not None:
        v_batch, num_value_heads, v_seqlen, value_dim = tuple(v_tensor.shape)
        if (v_batch, num_value_heads, v_seqlen) != (
            batch,
            num_kv_heads,
            seqlen_k,
        ):
            raise ValueError(
                "k_tensor and v_tensor batch, head, and sequence dimensions "
                f"must match, got {(batch, num_kv_heads, seqlen_k)} and "
                f"{(v_batch, num_value_heads, v_seqlen)}"
            )
        if value_dim != head_dim:
            raise ValueError(f"V head dimension must match Q/K ({head_dim}), got {value_dim}")
        require_dtype("v_tensor.dtype", v_tensor, (dtype,))

    return (
        batch,
        num_query_heads,
        num_kv_heads,
        seqlen_q,
        seqlen_k,
        head_dim,
        dtype,
    )


__all__ = [
    "bhs_lse_as_bsh_spec",
    "bhsd_storage_spec",
    "require_bhsd_qkv",
]
