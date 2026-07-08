# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free public-axis bindings for dense JAX GEMMs."""

from __future__ import annotations

from .layout import mode_from_layout

GEMM_A_LAYOUTS = ("LMK", "LKM")
GEMM_B_LAYOUTS = ("LNK", "LKN")
GEMM_OUTPUT_LAYOUTS = ("LMN", "LNM")

ROW_MAJOR_STRIDE_ORDER_3D = (2, 1, 0)
# Public row-major shape: (L, tiles_M_or_N, rest_K_or_N, 32, 4, 4).
BLOCK_SCALE_MODE = (3, 4, 1, 5, 2, 0)
BLOCK_SCALE_STRIDE_ORDER = (3, 1, 0, 4, 2, 5)
# Public row-major shape: (L, 1, M).
PROBABILITY_MODE = (2, 1, 0)
PROBABILITY_STRIDE_ORDER = (0, 1, 2)


def require_layout(name: str, layout: str, supported: tuple[str, ...]) -> str:
    """Validate an explicit, case-sensitive public axis-order string."""

    if not isinstance(layout, str):
        raise TypeError(f"{name} must be a string, got {type(layout).__name__}")
    if layout not in supported:
        choices = ", ".join(repr(value) for value in supported)
        raise ValueError(f"{name} must be one of ({choices}), got {layout!r}")
    return layout


def gemm_a_mode(layout: str) -> tuple[int, ...]:
    """Map public A layout ``LMK`` or ``LKM`` to canonical ``MKL`` axes."""

    layout = require_layout("a_layout", layout, GEMM_A_LAYOUTS)
    return mode_from_layout(layout, kernel_axes="MKL")


def gemm_b_mode(layout: str) -> tuple[int, ...]:
    """Map public B layout ``LNK`` or ``LKN`` to canonical ``NKL`` axes."""

    layout = require_layout("b_layout", layout, GEMM_B_LAYOUTS)
    return mode_from_layout(layout, kernel_axes="NKL")


def gemm_output_mode(layout: str, *, name: str = "c_layout") -> tuple[int, ...]:
    """Map public output layout ``LMN`` or ``LNM`` to canonical ``MNL`` axes."""

    layout = require_layout(name, layout, GEMM_OUTPUT_LAYOUTS)
    return mode_from_layout(layout, kernel_axes="MNL")


__all__ = [
    "BLOCK_SCALE_MODE",
    "BLOCK_SCALE_STRIDE_ORDER",
    "GEMM_A_LAYOUTS",
    "GEMM_B_LAYOUTS",
    "GEMM_OUTPUT_LAYOUTS",
    "PROBABILITY_MODE",
    "PROBABILITY_STRIDE_ORDER",
    "ROW_MAJOR_STRIDE_ORDER_3D",
    "gemm_a_mode",
    "gemm_b_mode",
    "gemm_output_mode",
    "require_layout",
]
