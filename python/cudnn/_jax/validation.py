# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Validation helpers shared by JAX operation wrappers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp

_NO_DEFAULT = object()


def as_dtype(value: Any) -> Any:
    """Return a JAX dtype without retaining a dtype-bearing value."""

    # Scalar dtype classes such as numpy.float32 expose an instance-level
    # dtype descriptor, so only unwrap dtype-bearing values, not classes.
    if not isinstance(value, type) and hasattr(value, "dtype"):
        value = value.dtype
    return jnp.dtype(value)


def as_optional_dtype(value: Any | None) -> Any | None:
    """Return ``None`` or a JAX dtype without retaining the source value."""

    return None if value is None else as_dtype(value)


def require_dtype(
    name: str,
    value: Any,
    valid_dtypes: Iterable[Any],
    *,
    default: Any = _NO_DEFAULT,
) -> Any:
    """Return a supported dtype from a dtype-like value or object with ``dtype``."""

    if value is None:
        if default is _NO_DEFAULT:
            raise ValueError(f"{name} must not be None")
        value = default

    dtype = as_dtype(value)
    valid_dtypes = tuple(as_dtype(item) for item in valid_dtypes)
    if dtype not in valid_dtypes:
        supported = ", ".join(item.name for item in valid_dtypes)
        raise ValueError(f"{name} must be one of {{{supported}}}, got {dtype}")
    return dtype


__all__ = ["as_dtype", "as_optional_dtype", "require_dtype"]
