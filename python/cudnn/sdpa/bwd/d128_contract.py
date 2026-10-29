# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only selection and workspace contract for dense d128 backward."""

import math
import re
import struct

ACCUMULATION = "accumulation"
RECOMPUTE = "recompute"
WORKSPACE_ALIGNMENT = 256
WORKSPACE_SLACK = WORKSPACE_ALIGNMENT - 16
ABI_VERSION = 3


def validate_dense_bounds(batch, heads, sq, skv, window_left=None):
    """Bound dense MHA declarations before either implementation is selected."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, sq, skv)):
        raise ValueError("dense d128 dimensions must be positive integers")
    if window_left is not None and (type(window_left) is not int or window_left < 0 or max(sq, skv) + window_left + 129 > 2**31 - 1):
        raise ValueError("dense d128 sliding-window envelope exceeds safe Int32 tile rounding")
    # The recompute body uses Int32 sequence coordinates, including the
    # one-past-tile endpoint and ceil_div(endpoint, 128) in the causal bound.
    if max(sq, skv) > 2**31 - 128:
        raise ValueError("dense d128 sequence envelope exceeds safe Int32 tile rounding")
    # Its one-dimensional grid contains both Q and KV owners. Bounding the
    # full grid also bounds batch/head coordinates and decoder products.
    owners = (sq + 127) // 128 + (skv + 127) // 128
    if batch * heads * owners > 2**31 - 1:
        raise ValueError("dense d128 launch envelope exceeds the grid or Int32 work-id range")
    # The recompute Stats coordinates do not explicitly widen before applying
    # their compact strides; accumulation also uses an Int32 row index.
    if batch * heads * sq > 2**31 - 1:
        raise ValueError("dense d128 Stats index exceeds the Int32 range")
    # Prepared operands normalize singleton pitches to these compact BSHD
    # strides. Every 16-bit tensor-map byte stride must be strictly below 2**40.
    if max(sq, skv) * heads * 128 * 2 >= 2**40:
        raise ValueError("dense d128 BSHD byte stride exceeds the tensor-map stride range")


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def validate_accumulation_scale(scale):
    """Check both FP32 scales consumed by the accumulation implementation."""
    try:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError
        rounded = _float32(scale)
        exponent = _float32(rounded * _float32(1.4426950408889634))
        if not math.isfinite(rounded) or rounded <= 0 or not math.isfinite(exponent) or exponent <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError, struct.error):
        raise ValueError("dense d128 accumulation requires scale and scale*log2(e) finite and positive in FP32") from None


# Compute capabilities the accumulation chain (dense, THD and grouped) is qualified on.
ACCUMULATION_CCS = ((10, 0), (10, 3))


def select_implementation(cc, dsl_version, batch, heads, sq, skv, scale, d_qk=128, d_v=128, input_bytes=2):
    """Choose from plan facts; recompute stays the fallback outside the accumulation envelope.

    The accumulation kernels round both sequence lengths up to whole 128-row
    tiles: TMA zero-fills the tail loads and clips the tail stores and dQ
    reductions, and the tail Stats rows publish -inf, so unaligned S_q / S_kv
    take the same path as aligned ones.
    """
    if tuple(cc) not in ACCUMULATION_CCS or not dsl_version or dsl_version[0] != "nvidia-cutlass-dsl":
        return RECOMPUTE
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", dsl_version[1])
    if match is None or tuple(map(int, match.groups())) < (4, 8, 0):
        return RECOMPUTE
    if any(type(n) is not int or n <= 0 for n in (batch, heads, sq, skv)):
        return RECOMPUTE
    # The tile rounding and the one-past-tile endpoints must stay Int32.
    if max(sq, skv) > 2**31 - 128:
        return RECOMPUTE
    q_tiles, k_tiles = (sq + 127) // 128, (skv + 127) // 128
    # Stats / delta rows are indexed over the rounded S_q; the grid holds one
    # work item per (batch, head, KV tile).
    if batch > 65535 or heads > 65535 or batch * heads * q_tiles * 128 > 2**31 - 1 or batch * heads * k_tiles > 2**31 - 1:
        return RECOMPUTE
    # The causal scheduler divides Int32 tile ids by swizzle * num_k_tiles.
    # Match the compiled family's Q/dO footprint, including narrower FP8
    # payloads: a smaller row produces a larger scheduler divisor.
    heads_per_l2 = (50 * 1024 * 1024) // (sq * (d_qk + d_v) * input_bytes)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    if swizzle * k_tiles > 2**31 - 1:
        return RECOMPUTE
    if input_bytes == 1:
        # The separate FP8 dQ scheduler sizes its swizzle from K/V, then
        # divides work ids by that swizzle times the number of Q tiles.
        kv_heads_per_l2 = (50 * 1024 * 1024) // (skv * (d_qk + d_v) * input_bytes)
        kv_swizzle = 1 << (kv_heads_per_l2.bit_length() - 1) if kv_heads_per_l2 > 0 else 1
        if kv_swizzle * q_tiles > 2**31 - 1:
            return RECOMPUTE
    if max(sq, skv) * heads * 128 * 2 >= 2**40:
        return RECOMPUTE
    try:
        validate_accumulation_scale(scale)
    except ValueError:
        return RECOMPUTE
    return ACCUMULATION


def workspace_layout(batch, heads, sq, d=128):
    """Return aligned-payload-relative regions and advertised allocation bytes."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, sq)) or d not in (64, 128, 192):
        raise ValueError("dense d128 workspace dimensions must be positive integers")
    rounded = (sq + 127) // 128 * 128
    dq_bytes = 4 * batch * rounded * heads * d
    delta_bytes = 4 * batch * heads * rounded
    regions = (
        (0, (batch, rounded, heads, d), (rounded * heads * d, heads * d, d, 1)),
        (dq_bytes, (batch, heads, rounded), (heads * rounded, rounded, 1)),
    )
    total = dq_bytes + delta_bytes + WORKSPACE_SLACK
    if total > 2**63 - 1:
        raise ValueError("dense d128 workspace exceeds the signed pointer range")
    return regions, total


def validate_launch(pointers, spans, workspace, workspace_bytes, scale):
    """Validate current addresses and scale before any of the three launches."""
    validate_accumulation_scale(scale)
    if len(pointers) != 9 or len(spans) != 9:
        raise ValueError("dense d128 accumulation requires nine bound operands")
    if workspace <= 0 or workspace % 16 or workspace + workspace_bytes > 2**63 - 1:
        raise ValueError("dense d128 accumulation needs an aligned caller workspace within the signed pointer range")
    for i, (ptr, span) in enumerate(zip(pointers, spans)):
        if ptr <= 0 or span <= 0 or ptr + span > 2**63 - 1:
            raise ValueError("dense d128 operand exceeds the signed pointer range")
        if ptr < workspace + workspace_bytes and workspace < ptr + span:
            raise ValueError("dense d128 caller workspace overlaps an operand")
        if i >= 6:
            for j in range(i):
                if ptr < pointers[j] + spans[j] and pointers[j] < ptr + span:
                    raise ValueError("dense d128 gradient output overlaps another operand")
