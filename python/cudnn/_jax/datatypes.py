# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Conversions between JAX and canonical cuDNN data types."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .. import data_type

_DTYPE_NAMES = (
    ("bool_", "BOOLEAN"),
    ("float16", "HALF"),
    ("bfloat16", "BFLOAT16"),
    ("float32", "FLOAT"),
    ("float64", "DOUBLE"),
    ("int8", "INT8"),
    ("int32", "INT32"),
    ("int64", "INT64"),
    ("uint8", "UINT8"),
    ("float8_e4m3fn", "FP8_E4M3"),
    ("float8_e5m2", "FP8_E5M2"),
    ("float8_e8m0fnu", "FP8_E8M0"),
    ("float4_e2m1fn", "FP4_E2M1"),
    ("int4", "INT4"),
)


def _make_mappings():
    jax_to_cudnn = {}
    cudnn_to_jax = {}
    for jax_name, cudnn_name in _DTYPE_NAMES:
        jax_scalar_type = getattr(jnp, jax_name, None)
        cudnn_dtype = getattr(data_type, cudnn_name, None)
        if jax_scalar_type is None or cudnn_dtype is None:
            continue
        jax_dtype = jnp.dtype(jax_scalar_type)
        jax_to_cudnn[jax_dtype] = cudnn_dtype
        cudnn_to_jax[cudnn_dtype] = jax_dtype
    return jax_to_cudnn, cudnn_to_jax


_JAX_TO_CUDNN_DATA_TYPE, _CUDNN_TO_JAX_DATA_TYPE = _make_mappings()


def normalize_jax_dtype(value: Any | None, default: Any, name: str) -> Any:
    """Normalize a JAX dtype-like value and identify invalid arguments by name."""

    try:
        return jnp.dtype(default if value is None else value)
    except TypeError as error:
        raise TypeError(f"{name} must be a JAX dtype, got {value!r}") from error


def jax_to_cudnn_dtype(dtype: Any) -> data_type:
    """Return the canonical cuDNN type for a JAX dtype-like value."""

    return _JAX_TO_CUDNN_DATA_TYPE.get(jnp.dtype(dtype), data_type.NOT_SET)


def cudnn_to_jax_dtype(dtype: data_type) -> Any:
    """Return the JAX dtype corresponding to a canonical cuDNN type."""

    try:
        return _CUDNN_TO_JAX_DATA_TYPE[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported JAX data type {dtype}") from error


__all__ = ["cudnn_to_jax_dtype", "jax_to_cudnn_dtype", "normalize_jax_dtype"]
