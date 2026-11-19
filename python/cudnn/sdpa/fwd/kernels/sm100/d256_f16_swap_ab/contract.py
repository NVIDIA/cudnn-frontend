# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The swap-AB body's public identity and local geometry contract."""

from cudnn.sdpa.fwd.config_common import pack_gqa_supported
from ..half_flavor_contract import padded_shape

NAME = "sdpa_fwd_sm100_d256_swap_ab"
TEMPLATE_NAME = "decode_d256_f16"
KERNEL_FILE = "sm100/d256_f16_swap_ab/kernel.py"
QUERY_TILES = (16, 32)
CGA = 1

# The in-launch reduction of the split partials (the engine's CLUSTER_K_REDUCE knob; kernel.py compile(cluster_combine=True)).
# One unit's chunks form one thread-block cluster and reduce with the separate pass's lean group, which holds 8 splits.
CLUSTER_REDUCE_SPLITS = (2, 8)
CLUSTER_REDUCE_QUERY_TILE = 16
# The compute capabilities the form has run on. Another part of the line stays declined until a device run exists for it.
CLUSTER_REDUCE_VERIFIED_CCS = ((10, 0),)


def cluster_reduce_decline(
    *, device_cc, d_shape, tile, split_kv, q_rows=None, thd=False, has_sink=False, stats_log2=False, gated=False, o_matches_qkv=True, use_pdl=False
):
    """Why ``CLUSTER_K_REDUCE=1`` is not served for this plan, or None.

    The one statement of the form's domain: the engine row (:func:`mismatch`), the heuristics' proposal and the standalone
    adapter's support check all read it, so a set the heuristics mark is a set the row admits and the adapter builds.
    ``d_shape`` is the kernel flavor the graph rides, ``tile`` the Q tile, ``q_rows`` the plan's tokens times the heads one
    tile packs (None = not stated), ``o_matches_qkv`` whether O is stored in the Q/K/V half type. A graph outside it keeps its
    two-launch and unsplit plans; only the request is declined."""
    lo, hi = CLUSTER_REDUCE_SPLITS
    if tuple(device_cc) not in CLUSTER_REDUCE_VERIFIED_CCS:
        return "CLUSTER_K_REDUCE=1 is verified on compute capability 10.0 only; it is unverified on this device"
    if tuple(d_shape) != (256, 256) or tile != CLUSTER_REDUCE_QUERY_TILE:
        return "CLUSTER_K_REDUCE=1 is implemented on the 16-column D256 swap-AB rendering"
    if not lo <= split_kv <= hi:
        return f"CLUSTER_K_REDUCE=1 needs {lo} to {hi} KV chunks (SPLIT_KV) in one thread-block cluster"
    if q_rows is not None and q_rows > tile:
        return "CLUSTER_K_REDUCE=1 is implemented for one token unit per head group (tokens x packed heads within TILE_M)"
    if thd or has_sink or stats_log2 or gated or not o_matches_qkv:
        return (
            "CLUSTER_K_REDUCE=1 does not carry a sink, base-2 Stats, ragged queries, an epilogue gate or a quantized output; "
            "the separate reduction pass folds those"
        )
    if use_pdl:
        return "CLUSTER_K_REDUCE=1 has no programmatic-dependent-launch form"
    return None


def mismatch(facts, knobs):
    tile = QUERY_TILES[0] if knobs is None or knobs.tile_m is None else knobs.tile_m
    if facts.thd:
        return "the D256 swap-AB implementation requires dense queries; THD uses lookahead"
    if tile not in QUERY_TILES:
        return "D256 swap-AB TILE_M (N_Q) must be 16 or 32"
    if facts.h_kv <= 0 or facts.h_q <= 0 or facts.h_q % facts.h_kv:
        return "D256 swap-AB requires positive heads and an integral Q/KV head ratio"
    group = facts.h_q // facts.h_kv if knobs is not None and knobs.pack_gqa else 1
    if padded_shape(facts) == (128, 128):
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
    if knobs is not None and knobs.cluster_reduce:
        return cluster_reduce_decline(
            device_cc=facts.device_cc,
            d_shape=padded_shape(facts),
            tile=tile,
            split_kv=knobs.split_kv or 1,
            q_rows=facts.s_q * group,
            thd=facts.thd,
            has_sink=facts.has_sink,
            stats_log2=facts.has_stats_log2,
            gated=facts.has_epilogue_gate,
            o_matches_qkv=facts.dtype_o in (None, facts.dtype),
        )
    return None
