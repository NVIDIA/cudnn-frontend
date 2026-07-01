# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared utilities for optional JAX wrappers.

Static-value helpers use Python numeric protocols so they reject JAX arrays and
tracers without importing JAX or attempting device-to-host scalar conversion.
"""

from __future__ import annotations

from numbers import Integral, Real
import operator
from typing import Any, Optional


def _type_name(value: Any) -> str:
    return type(value).__name__


def require_static_int(value: Any, *, name: str) -> int:
    """Normalize a host-static integer compile parameter."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{name} must be a Python-static integer scalar; close over it or " "mark it static with jax.jit(static_argnames=...) " f"(got {_type_name(value)})"
        )
    try:
        return operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a Python-static integer scalar (got {_type_name(value)})") from exc


def optional_static_int(value: Any, *, name: str) -> Optional[int]:
    """Normalize an optional host-static integer compile parameter."""

    return None if value is None else require_static_int(value, name=name)


def require_static_bool(value: Any, *, name: str) -> bool:
    """Normalize a host-static boolean compile parameter."""

    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be a Python-static bool; close over it or mark it "
            "static with jax.jit(static_argnames=...) "
            f"(got {_type_name(value)})"
        )
    return value


def require_static_float(value: Any, *, name: str) -> float:
    """Normalize a host-static real-valued compile parameter."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{name} must be a Python-static real scalar; close over it or " "mark it static with jax.jit(static_argnames=...) " f"(got {_type_name(value)})"
        )
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a Python-static real scalar (got {_type_name(value)})") from exc


__all__ = [
    "optional_static_int",
    "require_static_bool",
    "require_static_float",
    "require_static_int",
]
