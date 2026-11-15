# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Local identity and admission for the native SM107 d256_f16_swap_ab body."""

from cudnn.sdpa.fwd.config_sm107 import decode_d256_wide_tile_admits
from ..half_flavor_contract import padded_shape

NAME = "sdpa_fwd_sm107_d256_swap_ab"
TEMPLATE_NAME = "decode_d256_f16"
KERNEL_FILE = "sm107/d256_f16_swap_ab/kernel.py"


def mismatch(facts, knobs):
    if padded_shape(facts) != (256, 256) or not facts.uniform_dtype:
        return "native D256 swap-AB requires a same-dtype D256/V256 envelope"
    if facts.thd or facts.attn_scale_prefolded:
        return "native D256 swap-AB has dense queries and applies the scale in-kernel"
    if facts.has_epilogue_gate and knobs is not None and (knobs.split_kv or 1) <= 1:
        return "native D256 swap-AB applies a gate only in the split combine"
    if knobs is not None:
        if knobs.cga not in (None, 1) or knobs.sched_policy not in (None, 0):
            return "native D256 swap-AB requires one CTA and the direct grid"
        if knobs.tile_m not in (None, 16, 32, 64) or knobs.tile_n not in (None, 128):
            return "native D256 swap-AB requires TILE_M=16/32/64 and TILE_N=128"
        group = facts.h_q // facts.h_kv if knobs.pack_gqa and facts.h_kv > 0 else 1
        if group > (knobs.tile_m or 16):
            return "the packed head group must fit the selected query tile"
        if knobs.tile_m == 64 and not decode_d256_wide_tile_admits(
            group, paged=facts.has_paged_kv, masked=facts.causal or facts.right_band_widening or facts.window_left is not None
        ):
            return "the native 64-column tile requires packed heads and a causal/window mask when paged"
    return None
