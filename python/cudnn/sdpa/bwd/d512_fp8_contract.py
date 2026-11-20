# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared compiler, architecture and launch admission for the D512 quantized pairs."""

import re


def environment_error(cc, version):
    if tuple(cc) not in ((10, 0), (10, 3), (10, 7)):
        return f"D512 FP8 backward requires SM100, SM103 or SM107; got {cc}"
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version[1]) if version and version[0] == "nvidia-cutlass-dsl" else None
    if match is None or tuple(map(int, match.groups())) < (4, 8, 0):
        return f"D512 FP8 backward requires nvidia-cutlass-dsl >= 4.8.0; found {version}"
    return None


def launch_error(batch, heads, kv_heads, sq, sk):
    """The dense D512 owners' additional limits, after ``validate_dense_bounds``.

    Both precisions launch four CTAs per gradient tile and use the same
    rounded Int32 delta rows. Two-byte gradients set the TMA stride bound
    even when the activation payloads are FP8. These checks also apply to
    causal schedules: the helper and owner grids still use batch/head axes.
    """
    if max(batch, heads, kv_heads) > 65535:
        return "batch/head grid dimensions must fit CUDA grid.y/z (65535)"
    if 4 * ((max(sq, sk) + 127) // 128) > 2**31 - 1:
        return "gradient tile grid exceeds CUDA grid.x"
    if batch * heads * ((sq + 127) // 128 * 128) > 2**31 - 1:
        return "rounded delta index exceeds the Int32 range"
    if max(sq, sk) * heads * 512 * 2 >= 2**40:
        return "BSHD byte strides exceed TMA limits"
    return None
