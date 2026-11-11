# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declaration bounds shared by graph and standalone SM120 packed backward."""

from .d128_contract import validate_accumulation_scale


def native_head_dims(d_qk, d_v):
    from cudnn.sdpa.fwd.config_sm80 import FLAVOR_CFGS, pick_flavor

    cfg = FLAVOR_CFGS[pick_flavor(d_qk, d_v)]
    return cfg.D_QK, cfg.D_V


def native_tiles(d_qk, d_v, tile_m=None, tile_n=None):
    """Resolve the packed mma.sync tile inside the 99 KiB SM120 budget."""
    padded_qk, padded_v = native_head_dims(d_qk, d_v)

    def smem(m, n):
        # One Q/dO stage, resident K/V, FP32 P and half transposed dS.
        return 2 * (m + n) * (padded_qk + padded_v) + 6 * m * n

    default = 64 if smem(64, 64) <= 99 * 1024 else 32
    m = default if tile_m is None else tile_m
    n = default if tile_n is None else tile_n
    if m not in (32, 64) or n not in (32, 64) or m % n:
        raise ValueError("SM120 THD backward requires 32/64-row tiles with TILE_M a multiple of TILE_N")
    if smem(m, n) > 99 * 1024:
        raise ValueError(f"SM120 THD backward tile {m}x{n} exceeds the 99 KiB shared-memory budget")
    return m, n


def knob_mismatch(facts, requested):
    if facts.d_qk > 256:
        if requested and (requested.tile_m not in (None, 16) or requested.tile_n not in (None, 16)):
            return "SM120 wide backward requires native 16x16 tiles"
        return None
    if requested and (requested.tile_m == 16 or requested.tile_n == 16):
        return "SM120 16x16 backward tiles require D_QK > 256"
    if not facts.thd:
        return None
    try:
        native_tiles(facts.d_qk, facts.d_v, requested.tile_m if requested else None, requested.tile_n if requested else None)
    except ValueError as error:
        return str(error)
    return None


def validate_bounds(batch, heads, kv_heads, sq, skv, d_qk, d_v, cap_q, cap_kv, scale):
    if not (0 < d_v <= d_qk <= 256) or d_qk % 8 or d_v % 8:
        raise ValueError("SM120 THD backward requires eight-aligned 0 < D_V <= D_QK <= 256")
    if kv_heads <= 0 or heads % kv_heads:
        raise ValueError("SM120 THD backward requires H_Q divisible by positive H_KV")
    if not (0 < batch <= 65535 and 0 < heads <= 65535):
        raise ValueError("SM120 THD backward batch/head counts exceed the native grid limits")
    if not (0 < sq <= 2**31 - 64 and 0 < skv <= 2**31 - 64):
        raise ValueError("SM120 THD backward sequence envelope exceeds safe Int32 tile rounding")
    # Independent-origin delta and dQ cast grids use the full B*max_SQ
    # envelope, not the packed capacity (which can stay tiny for empty rows).
    if batch * sq >= 2**31:
        raise ValueError("SM120 THD backward batch times Q envelope exceeds the Int32 launch range")
    # Independent dK/dV origins reduce GQA partials with one block for every
    # (sequence, KV row), even when packed token capacity is much smaller.
    if batch * skv >= 2**31:
        raise ValueError("SM120 THD backward batch times KV envelope exceeds the Int32 launch range")
    if cap_q is None or cap_kv is None or min(cap_q, cap_kv) <= 0:
        raise ValueError("SM120 THD backward requires positive declared packed capacities")
    # Native scratch/reduction launch counts use Int32 element counts;
    # physical addressing itself retains each port's Int64 strides/origins.
    if max(cap_q, cap_kv) * heads * max(native_head_dims(d_qk, d_v)) >= 2**31:
        raise ValueError("SM120 THD backward packed capacity exceeds the Int32 element launch range")
    validate_accumulation_scale(d_qk**-0.5 if scale is None else scale)


def graph_mismatch(facts, version):
    if facts.d_qk > 256:
        from .sm120_wide_contract import graph_mismatch as wide_mismatch

        return wide_mismatch(facts, version)
    if not facts.thd:
        return None
    if facts.device_cc not in ((12, 0), (12, 1)):
        return "SM120 THD backward requires cc 12.0 or 12.1"
    for enabled, label in (
        (facts.deterministic, "deterministic"),
        (facts.window_left is not None or facts.right_band_widening, "window"),
        (facts.has_sink or facts.has_dsink, "sink / dSink"),
    ):
        if enabled:
            return f"SM120 THD backward does not support {label}"
    try:
        validate_bounds(
            facts.b,
            facts.h_q,
            facts.h_kv,
            facts.s_q,
            facts.s_kv,
            facts.d_qk,
            facts.d_v,
            facts.max_total_seq_len_q,
            facts.max_total_seq_len_kv,
            facts.scale,
        )
    except ValueError as error:
        return str(error)
    if facts.stats_t is not None:
        from cudnn.sdpa.graph_analyzer import thd_stats_packing

        strides = tuple(facts.stats_t.get_stride())
        if len(strides) != 4:
            return "SM120 THD Stats strides must have rank 4"
        if thd_stats_packing(strides[1], strides[2], facts.h_q) == "head_major" and strides[1] < facts.max_total_seq_len_q:
            return "SM120 THD Stats head stride must cover the physical Q-token capacity"
    return None
