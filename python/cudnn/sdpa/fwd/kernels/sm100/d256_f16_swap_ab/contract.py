# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The swap-AB body's public identity and local geometry contract."""

from cudnn.sdpa.fwd.config_common import pack_gqa_supported

NAME = "sdpa_fwd_sm100_d256_swap_ab"
TEMPLATE_NAME = "decode_d256_f16"
KERNEL_FILE = "sm100/d256_f16_swap_ab/kernel.py"
QUERY_TILES = (16, 32)
CGA = 1


def mismatch(facts, knobs):
    tile = QUERY_TILES[0] if knobs is None or knobs.tile_m is None else knobs.tile_m
    if facts.thd:
        return "the D256 swap-AB implementation requires dense queries; THD uses lookahead"
    if tile not in QUERY_TILES:
        return "D256 swap-AB TILE_M (N_Q) must be 16 or 32"
    if facts.h_kv <= 0 or facts.h_q <= 0 or facts.h_q % facts.h_kv:
        return "D256 swap-AB requires positive heads and an integral Q/KV head ratio"
    group = facts.h_q // facts.h_kv if knobs is not None and knobs.pack_gqa else 1
    if (facts.d_qk, facts.d_v) == (128, 128):
        if not facts.has_paged_kv or tile != 16 or not 0 < facts.s_q * group <= 8:
            return "D128 half swap-AB requires paged KV, TILE_M=16 and at most eight live Q columns"
        if facts.has_bias or facts.has_kv_mask or facts.has_score_max or facts.has_score_sum_exp:
            return "D128 half swap-AB has no bias, KV-mask or score-output stores"
    if group > tile:
        return "D256 swap-AB packs a whole KV-head group, which must fit TILE_M (N_Q)"
    if knobs is not None and knobs.pack_gqa and not pack_gqa_supported(facts.h_q, facts.h_kv, partial=True):
        return "D256 swap-AB retains the SM100 half PackGQA head-ratio domain"
    if knobs is not None and knobs.sched_policy not in (None, 0):
        return "D256 swap-AB has the direct grid only (SCHED_POLICY=0)"
    return None
