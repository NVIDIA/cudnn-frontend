# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared SM120 per-tensor FP8 backward declaration bounds."""

import re

from .d128_contract import validate_accumulation_scale


def environment_error(cc, version):
    if tuple(cc or ()) not in ((12, 0), (12, 1)):
        return f"SM120 FP8 backward requires SM120 or SM121; got {cc}"
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version[1]) if version and version[0] == "nvidia-cutlass-dsl" else None
    if match is None or tuple(map(int, match.groups())) < (4, 8, 0):
        return f"SM120 FP8 backward requires nvidia-cutlass-dsl >= 4.8.0; found {version}"
    return None


def validate_bounds(batch, heads, kv_heads, sq, skv, dq, dv, scale, window_left):
    if any(type(n) is not int or n <= 0 for n in (batch, heads, kv_heads, sq, skv, dq, dv)):
        raise ValueError("SM120 FP8 backward requires positive integer dimensions")
    if not (dv <= dq <= 512 and dq % 8 == 0 and dv % 8 == 0):
        raise ValueError("SM120 FP8 backward requires eight-aligned 0 < D_V <= D_QK <= 512")
    if heads % kv_heads or max(batch, heads, kv_heads) > 65535:
        raise ValueError("SM120 FP8 backward requires an integral head ratio and batch/heads <= 65535")
    if window_left is not None and (type(window_left) is not int or window_left < 0):
        raise ValueError("SM120 FP8 backward left window must be a nonnegative integer")
    window = 0 if window_left is None else window_left + 1
    if type(window) is not int or window < 0 or max(sq, skv) + window + 128 >= 2**31:
        raise ValueError("SM120 FP8 backward sequence/window tile coordinates exceed Int32")
    if batch * heads * ((sq + 127) // 128 * 128) >= 2**31:
        raise ValueError("SM120 FP8 backward rounded delta index exceeds Int32")
    if ((max(sq, skv) + 15) // 16) * ((dq + 127) // 128) >= 2**31:
        raise ValueError("SM120 FP8 backward gradient grid exceeds Int32")
    validate_accumulation_scale(dq**-0.5 if scale is None else scale)
