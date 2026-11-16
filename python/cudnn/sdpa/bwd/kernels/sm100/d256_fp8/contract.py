# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100/SM103 dense D256 per-tensor FP8 backward contract."""

import re

from cudnn.sdpa.bwd.d128_contract import validate_accumulation_scale

NAME = "sdpa_bwd_sm100_d256_fp8"
KERNEL_FILE = "sm100/d256_fp8/kernel.py"


def environment_error(cc, version):
    if tuple(cc) not in ((10, 0), (10, 3)):
        return f"D256 FP8 backward requires SM100 or SM103; got {cc}"
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version[1]) if version and version[0] == "nvidia-cutlass-dsl" else None
    if match is None or tuple(map(int, match.groups())) < (4, 8, 0):
        return f"D256 FP8 backward requires nvidia-cutlass-dsl >= 4.8.0; found {version}"
    return None


def declaration_error(batch, heads, heads_kv, sq, skv, scale, window_left=None):
    """One metadata-only bound for both graph admission and the direct API."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, heads_kv, sq, skv)):
        return "D256 FP8 dimensions must be positive integers"
    if heads < heads_kv or heads % heads_kv:
        return "D256 FP8 requires an integral Q/KV head ratio"
    if max(batch, heads, heads_kv) > 65535:
        return "D256 FP8 batch/head dimensions exceed CUDA grid.y/z"
    imax = 2**31 - 1
    if max(sq, skv) > imax - 127:
        return "D256 FP8 sequence tile rounding exceeds Int32"
    q_rounded, k_rounded = ((s + 127) // 128 * 128 for s in (sq, skv))
    if window_left is not None:
        if type(window_left) is not int or window_left < 0:
            return "D256 FP8 left window must be a nonnegative integer"
        # Kernel WINDOW is left + 1. A KV owner computes its exclusive Q
        # endpoint as (owner + 1)*128 + WINDOW - 1 + 127 before division.
        if k_rounded + window_left + 127 > imax:
            return "D256 FP8 sliding-window tile endpoint exceeds Int32"
    if batch * heads * q_rounded > imax:
        return "D256 FP8 rounded delta index exceeds Int32"
    if 2 * max(q_rounded, k_rounded) // 128 > imax:
        return "D256 FP8 gradient tile grid exceeds CUDA grid.x"
    # Compact FP16/BF16 outputs have the widest tensor-map byte strides.
    if max(sq * heads, skv * heads_kv) * 256 * 2 >= 2**40:
        return "D256 FP8 BSHD byte stride exceeds the tensor-map limit"
    try:
        validate_accumulation_scale(256**-0.5 if scale is None else scale)
    except ValueError as error:
        return str(error).replace("dense d128 accumulation", "D256 FP8 backward")
    return None


def graph_error(facts, version):
    return environment_error(facts.device_cc, version) or declaration_error(
        facts.b, facts.h_q, facts.h_kv, facts.s_q, facts.s_kv, facts.scale, facts.window_left
    )
