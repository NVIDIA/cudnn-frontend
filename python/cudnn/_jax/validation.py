# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Validation helpers shared by JAX operation wrappers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp

_NO_DEFAULT = object()


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

    def as_dtype(item: Any) -> Any:
        # Scalar dtype classes such as numpy.float32 expose an instance-level
        # dtype descriptor, so only unwrap dtype-bearing values, not classes.
        if not isinstance(item, type) and hasattr(item, "dtype"):
            item = item.dtype
        return jnp.dtype(item)

    dtype = as_dtype(value)
    valid_dtypes = tuple(as_dtype(item) for item in valid_dtypes)
    if dtype not in valid_dtypes:
        supported = ", ".join(item.name for item in valid_dtypes)
        raise ValueError(f"{name} must be one of {{{supported}}}, got {dtype}")
    return dtype


__all__ = ["require_dtype"]
