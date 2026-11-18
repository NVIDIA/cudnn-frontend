# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only selection and workspace contract for dense d128 backward."""

import math
import re
import struct

ACCUMULATION = "accumulation"
RECOMPUTE = "recompute"
PAIR = "pair"
RELAY = "relay"
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


# The parts the accumulation chain (dense, THD and grouped) is admitted on, BY NAME: GB200 (cc 10.0, the measured part), GB300 (cc 10.3,
# admitted 2026-10-28: the same tcgen05 / cta_group::2 / TMEM / CLC feature set) and Rubin (cc 10.7, admitted 2026-11-03: the same bodies,
# unchanged).  The compile target follows the device in every prepared host (``--gpu-arch sm_{sm}a`` with ``sm = cc[0] * 10 + cc[1]``), so
# admitting a part is this tuple plus the engine row's sm range -- never a copy of a kernel into another arch directory.
ACCUMULATION_CCS = ((10, 0), (10, 3), (10, 7))
# The text every decline off these parts carries (the graph row's declaration check and the THD / grouped standalone adapters pin ONE token).
ACCUMULATION_PARTS = "SM100, SM103 or SM107 (cc 10.0 / 10.3 / 10.7)"
# The dense MHA envelope of the ``sdpa_bwd_sm100_d128`` row (its ``Capabilities.sm_lo`` / ``sm_hi``, inclusive): the one-launch recompute body
# serves every cc in it, the accumulation chain the ``ACCUMULATION_CCS`` parts only; cc 10.7 closes the range (SM108+ is declined by name
# until a part is admitted).  Read by the engine spec and the dense adapter's ``check_support`` -- one place.
DENSE_SM_RANGE = (100, 107)
# The parts the deterministic kernel pair is measured on; elsewhere a deterministic dense MHA request keeps the recompute body.
PAIR_CCS = ((10, 0), (10, 3))
D192_FP8_CCS = ACCUMULATION_CCS


def select_implementation(cc, dsl_version, batch, heads, sq, skv, scale, d_qk=128, d_v=128, input_bytes=2, ccs=ACCUMULATION_CCS):
    """Choose from plan facts; recompute stays the fallback outside the accumulation envelope.

    The accumulation kernels round both sequence lengths up to whole 128-row
    tiles: TMA zero-fills the tail loads and clips the tail stores and dQ
    reductions, and the tail Stats rows publish -inf, so unaligned S_q / S_kv
    take the same path as aligned ones.
    """
    if tuple(cc) not in ccs or not dsl_version or dsl_version[0] != "nvidia-cutlass-dsl":
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


PAIR_SPLITS = (1, 2, 4)
# A q-range split's fp32 partial tile costs about this many KV steps of the dK / dV kernel to write and fold back.
_PAIR_SPLIT_STEP_COST = 4.3


def pair_split(batch, heads_kv, skv, sq, ratio, sms, masked=False):
    """The q-range split of the pair's dK / dV kernel: the member of PAIR_SPLITS whose wave-quantization tail plus
    partial traffic is lowest; 1 under a mask (its tiles are not uniform) or without a device SM count."""
    if masked or not sms or sms <= 0:
        return 1
    ctas = batch * heads_kv * ((skv + 127) // 128)
    steps = max(1, ratio * ((sq + 127) // 128))

    def cost(split):
        waves = -(-ctas * split // sms)
        tail = (waves * sms - ctas * split) / (waves * sms)
        return tail + _PAIR_SPLIT_STEP_COST * split / steps

    return min(PAIR_SPLITS, key=lambda split: (cost(split), split))


def pair_workspace_layout(batch, heads, sq, split=1, heads_kv=0, skv=0, d=0):
    """The deterministic pair's scratch: the fp32 delta rows over the 128-row-rounded S_q and, under a q-range split, one
    fp32 dV and one fp32 dK partial plane per (batch, split) over the 128-row-rounded S_kv."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, sq)):
        raise ValueError("dense d128 workspace dimensions must be positive integers")
    if split not in PAIR_SPLITS or (split > 1 and any(type(n) is not int or n <= 0 for n in (heads_kv, skv, d))):
        raise ValueError("dense d128 q-range split requires positive KV dimensions")
    rounded = (sq + 127) // 128 * 128
    regions = [(0, (batch, heads, rounded), (heads * rounded, rounded, 1))]
    total = 4 * batch * heads * rounded
    if split > 1:
        rows = (skv + 127) // 128 * 128
        plane = (split * batch, rows, heads_kv, d)
        strides = (rows * heads_kv * d, heads_kv * d, d, 1)
        part_bytes = 4 * split * batch * rows * heads_kv * d
        regions.append((total, plane, strides))
        regions.append((total + part_bytes, plane, strides))
        total += 2 * part_bytes
    total += WORKSPACE_SLACK
    if total > 2**63 - 1:
        raise ValueError("dense d128 workspace exceeds the signed pointer range")
    return tuple(regions), total


def validate_relay_launch(batch, heads):
    """The relay chain's launches place the batch and the query heads on grid.y / grid.z."""
    if batch > 65535 or heads > 65535:
        raise ValueError("relay chain batch and query-head counts must be at most 65535")


def relay_workspace_layout(batch, heads, sq, d):
    """The ordered-relay chain's scratch: the fp32 dQ accumulator and delta rows over the 128-row-rounded S_q, then one
    int32 turn counter per (batch, query head, 64-row query tile) and one key-tile ticket per (batch, query head)."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, sq)) or d not in (64, 128):
        raise ValueError("relay workspace dimensions must be positive integers with a head dimension of 64 or 128")
    rounded = (sq + 127) // 128 * 128
    slots = rounded // 64 + 1  # one turn counter per 64-row query tile, then the head's key-tile ticket
    dq_bytes = 4 * batch * rounded * heads * d
    delta_bytes = 4 * batch * heads * rounded
    turn_bytes = 4 * batch * heads * slots
    regions = (
        (0, (batch, rounded, heads, d), (rounded * heads * d, heads * d, d, 1)),
        (dq_bytes, (batch, heads, rounded), (heads * rounded, rounded, 1)),
        (dq_bytes + delta_bytes, (batch, heads, slots), (heads * slots, slots, 1)),
    )
    total = dq_bytes + delta_bytes + turn_bytes + WORKSPACE_SLACK
    if total > 2**63 - 1:
        raise ValueError("relay workspace exceeds the signed pointer range")
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
