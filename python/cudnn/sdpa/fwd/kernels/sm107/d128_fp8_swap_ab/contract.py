# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Rubin graph domain for the existing shared FP8 swap-AB body."""

from dataclasses import dataclass
from cudnn import data_type
from ...sm100.d128_fp8_swap_ab.contract import Geometry as SharedGeometry
from ..small_fp8_contract import in_domain
from ..fp8_flavor_contract import padded_shape

NAME = "sdpa_fwd_sm107_d128_fp8_swap_ab"
LEGACY_ROW = "sdpa_fwd_prefill_sm107_fp8"


@dataclass(frozen=True)
class Geometry(SharedGeometry):
    implementation: str = NAME
    lowering_row_name: str = LEGACY_ROW


def mismatch(facts, knobs):
    if not in_domain(facts) or padded_shape(facts) != (128, 128):
        return "swap-AB requires a D128 per-tensor FP8 envelope on SM107"
    if facts.thd or facts.has_paged_kv or not (0 < facts.b <= 65535):
        return "swap-AB requires dense nonpaged queries and batch in [1, 65535]"
    if facts.o_block_scale or facts.dtype_o not in (data_type.FP8_E4M3, data_type.FP8_E5M2, data_type.HALF, data_type.BFLOAT16):
        return "swap-AB requires scalar half or FP8 output"
    if facts.softmax_precision not in (None, data_type.FLOAT) or facts.attn_scale_prefolded:
        return "swap-AB requires FP32 softmax without pre-folded attention scale"
    if facts.has_epilogue_gate or facts.has_bias or facts.has_kv_mask or facts.has_score_max or facts.has_score_sum_exp:
        return "swap-AB does not carry extra score or output epilogues"
    if facts.h_kv <= 0 or facts.h_q <= 0 or facts.h_q % facts.h_kv:
        return "swap-AB requires an integral Q/KV head ratio"
    if knobs is not None and (knobs.cga not in (None, 1) or knobs.tile_m not in (None, 16) or knobs.tile_n not in (None, 128)):
        return "swap-AB requires TILE_M=16, TILE_N=128 and TILE_CGA_M=1"
    group = facts.h_q // facts.h_kv if knobs is not None and knobs.pack_gqa else 1
    if group > 8 or not (0 < facts.s_q * group <= 8):
        return "swap-AB requires every live query row in eight columns"
    return None
