# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared declaration bounds for native SM120 wide backward."""

from .d128_contract import validate_accumulation_scale


def validate_bounds(batch, heads, kv_heads, sq, skv, dq, dv, cap_q, cap_kv, scale, thd, deterministic=False):
    if not (256 < dq <= 512 and 0 < dv <= dq and dq % 8 == dv % 8 == 0):
        raise ValueError("SM120 wide backward requires eight-aligned 256 < D_QK <= 512 and 0 < D_V <= D_QK")
    if not (0 < batch <= 65535 and 0 < heads <= 65535 and kv_heads > 0 and heads % kv_heads == 0):
        raise ValueError("SM120 wide backward batch/head counts exceed grid limits or H_Q is not divisible by H_KV")
    if not (0 < sq <= 2**31 - 64 and 0 < skv <= 2**31 - 64 and batch * sq < 2**31 and batch * skv < 2**31):
        raise ValueError("SM120 wide backward sequence envelope exceeds the Int32 launch range")
    q_extent, kv_extent = (cap_q, cap_kv) if thd else (batch * sq, batch * skv)
    if q_extent is None or kv_extent is None or min(q_extent, kv_extent) <= 0:
        raise ValueError("SM120 wide backward requires positive token capacities")
    if max(q_extent, kv_extent) * heads * 512 >= 2**31:
        raise ValueError("SM120 wide backward scratch exceeds the Int32 element launch range")
    if deterministic and batch * heads * 2 * ((sq + 15) // 16) >= 2**31:
        raise ValueError("SM120 wide backward deterministic relay exceeds the Int32 index range")
    validate_accumulation_scale(dq**-0.5 if scale is None else scale)


def graph_mismatch(facts, version):
    if facts.device_cc not in ((12, 0), (12, 1)):
        return "SM120 wide backward requires cc 12.0 or 12.1"
    if facts.has_bias or facts.has_dbias or facts.has_sink or facts.has_dsink:
        return "SM120 wide backward bias and sink operands are not yet qualified"
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
            facts.thd,
            facts.deterministic,
        )
    except ValueError as error:
        return str(error)
    if facts.thd and facts.stats_t is not None:
        from cudnn.sdpa.graph_analyzer import thd_stats_packing

        strides = tuple(facts.stats_t.get_stride())
        if len(strides) != 4:
            return "SM120 THD Stats strides must have rank 4"
        if thd_stats_packing(strides[1], strides[2], facts.h_q) == "head_major" and strides[1] < facts.max_total_seq_len_q:
            return "SM120 THD Stats head stride must cover the physical Q-token capacity"
    return None
