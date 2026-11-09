# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declaration bounds shared by graph and standalone SM120 packed backward."""

from .d128_contract import validate_accumulation_scale


def validate_bounds(batch, heads, kv_heads, sq, skv, d_qk, d_v, cap_q, cap_kv, scale):
    if (d_qk, d_v) != (128, 128) or heads != kv_heads:
        raise ValueError("SM120 THD backward requires MHA with exact D_QK = D_V = 128")
    if not (0 < batch <= 65535 and 0 < heads <= 65535):
        raise ValueError("SM120 THD backward batch/head counts exceed the native grid limits")
    if not (0 < sq <= 2**31 - 64 and 0 < skv <= 2**31 - 64):
        raise ValueError("SM120 THD backward sequence envelope exceeds safe Int32 tile rounding")
    # Independent-origin delta and dQ cast grids use the full B*max_SQ
    # envelope, not the packed capacity (which can stay tiny for empty rows).
    if batch * sq >= 2**31:
        raise ValueError("SM120 THD backward batch times Q envelope exceeds the Int32 launch range")
    if cap_q is None or cap_kv is None or min(cap_q, cap_kv) <= 0:
        raise ValueError("SM120 THD backward requires positive declared packed capacities")
    # Native scratch/reduction launch counts use Int32 element counts;
    # physical addressing itself retains each port's Int64 strides/origins.
    if max(cap_q, cap_kv) * heads * 128 >= 2**31:
        raise ValueError("SM120 THD backward packed capacity exceeds the Int32 element launch range")
    validate_accumulation_scale(128**-0.5 if scale is None else scale)


def graph_mismatch(facts, version):
    if not facts.thd:
        return None
    if facts.device_cc not in ((12, 0), (12, 1)):
        return "SM120 THD backward requires cc 12.0 or 12.1"
    for enabled, label in (
        (facts.deterministic, "deterministic"),
        (facts.bottom_right, "bottom-right causal"),
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
