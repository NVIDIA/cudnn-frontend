# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Identity of the small-head FP8 swap_ab implementation."""

from dataclasses import dataclass, replace

from cudnn import data_type
from cudnn.sdpa.fwd.config_fp8_decode_common import TemplateParamsDecode
from ..small_fp8_contract import in_domain

NAME = "sdpa_fwd_sm100_d128_fp8_swap_ab"
TEMPLATE_NAME = "decode_d128_fp8"
KERNEL_FILE = "sm100/d128_fp8_swap_ab/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm100_fp8"


@dataclass(frozen=True)
class Geometry:
    d_flavor: int = 128
    implementation: str = NAME
    template: str = TEMPLATE_NAME
    kernel_file: str = KERNEL_FILE
    lowering_row_name: str = LEGACY_ROW
    query_tile: int = 16
    cga: int = 1
    cluster_m: int = 1
    cta_group: int = 1
    decode_q_tile: int = 16

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=self.query_tile,
            tile_n=128 if knobs.tile_n is None else knobs.tile_n,
            cga=self.cga,
            sched_policy=0,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
            swap_ab=None,
        )

    def configure_params(self, params):
        values = vars(params) | dict(d_flavor=self.d_flavor, cta_mma=1, dual_kv_stream=False, decode_q_tile=16, decode_head_dim=128, decode_live_columns=8)
        return TemplateParamsDecode(**values)


def mismatch(facts, knobs):
    if not in_domain(facts):
        return "swap-AB requires exact D64/D128 per-tensor FP8 on SM100/SM103"
    if facts.thd or not (0 < facts.b <= 65535):
        return "swap-AB requires dense queries and batch in [1, 65535]"
    if facts.o_block_scale or facts.dtype_o not in (data_type.FP8_E4M3, data_type.FP8_E5M2, data_type.HALF, data_type.BFLOAT16):
        return "swap-AB requires scalar half or FP8 output"
    if facts.softmax_precision not in (None, data_type.FLOAT) or facts.attn_scale_prefolded:
        return "swap-AB requires FP32 softmax without pre-folded attention scale"
    if facts.has_epilogue_gate or facts.has_bias or facts.has_kv_mask or facts.has_score_max or facts.has_score_sum_exp:
        return "swap-AB does not carry extra score or output epilogues"
    if facts.h_kv <= 0 or facts.h_q <= 0 or facts.h_q % facts.h_kv:
        return "swap-AB requires an integral Q/KV head ratio"
    if knobs is not None:
        if knobs.cga not in (None, 1) or knobs.tile_m not in (None, 16) or knobs.tile_n not in (None, 128):
            return "swap-AB requires TILE_M=16, TILE_N=128 and TILE_CGA_M=1"
    group = facts.h_q // facts.h_kv if knobs is not None and knobs.pack_gqa else 1
    if group > 8:
        return "swap-AB packs a whole group into eight live softmax columns"
    if facts.d_qk == 64:
        if facts.device_cc != (10, 0) or not facts.has_paged_kv:
            return "D64 swap-AB requires SM100 paged KV"
        if not (0 < facts.s_q <= 4 * (8 // group)):
            return "D64 swap-AB supports at most four eight-column token units"
    else:
        if facts.has_paged_kv and facts.device_cc != (10, 0):
            return "paged D128 swap-AB requires SM100"
        if not (0 < facts.s_q * group <= 8):
            return "D128 swap-AB requires every live query row in eight columns"
    return None
