# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Small integer helpers shared by public contracts and private backends."""

from __future__ import annotations


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def round_up(value: int, multiple: int) -> int:
    return ceil_div(value, multiple) * multiple


__all__ = ["ceil_div", "round_up"]
