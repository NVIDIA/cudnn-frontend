# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Rubin contracts for two existing SM100 bodies, without copying device code."""

from dataclasses import dataclass, replace
from cudnn import data_type
from ..sm100.d128_f16_single_q import contract as single_q
from ..sm100.d256_f16_lookahead import contract as lookahead

LEGACY_ROW = "sdpa_fwd_prefill_sm107"
SINGLE_Q = "sdpa_fwd_sm107_shared_single_q"
LOOKAHEAD = "sdpa_fwd_sm107_shared_d256_lookahead"
IMPLEMENTATIONS = frozenset({SINGLE_Q, LOOKAHEAD})
SHAPES = frozenset({(32, 32), (64, 64), (128, 128), (192, 128), (256, 256)})


def in_domain(facts):
    return (
        facts.device_cc == (10, 7)
        and (facts.d_qk, facts.d_v) in SHAPES
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
        and not (facts.invalid or facts.is_backward or facts.is_fp8 or facts.is_mxfp8)
    )


@dataclass(frozen=True)
class Geometry:
    implementation: str
    d_qk: int
    lowering_row_name = LEGACY_ROW
    query_tile = 128
    decode_q_tile = 0

    @property
    def cga(self):
        return 1 if self.implementation == SINGLE_Q else 2

    @property
    def cluster_m(self):
        return self.cga

    @property
    def cta_group(self):
        return self.cga

    @property
    def template(self):
        return single_q.TEMPLATE_NAME if self.implementation == SINGLE_Q else lookahead.TEMPLATE_NAME

    @property
    def kernel_file(self):
        return single_q.KERNEL_FILE if self.implementation == SINGLE_Q else lookahead.KERNEL_FILE

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=128,
            tile_n=128,
            cga=self.cga,
            sched_policy=0 if knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
            warp_spec_cfg=0,
        )

    def configure_params(self, params):
        if self.implementation == LOOKAHEAD:
            return params
        return single_q.Geometry(d_flavor=64 if self.d_qk <= 64 else 128, head_dim=192 if self.d_qk == 192 else 128).configure_params(params)


def explicit_geometry(implementation, facts, knobs):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown shared SM107 half implementation: {implementation}")
    return Geometry(implementation, facts.d_qk)


def mismatch(implementation, facts, knobs):
    if not in_domain(facts):
        return "requires SM107 and exact small/mid-width half forward graphs"
    if facts.has_bias or facts.has_kv_mask or facts.has_score_max or facts.has_score_sum_exp:
        return "the shared half bodies have no additive-bias, KV-mask or score-output stores"
    if knobs is not None and (knobs.tile_m not in (None, 128) or knobs.tile_n not in (None, 128) or knobs.warp_spec_cfg not in (None, 0)):
        return "the shared half bodies require M128 N128 and their ordinary warp topology"
    if implementation == LOOKAHEAD:
        if (facts.d_qk, facts.d_v) != (256, 256):
            return "shared lookahead requires exact D256/V256"
        if not ((facts.has_paged_kv and facts.uniform_dtype) or (facts.thd and not facts.has_paged_kv)):
            return "shared lookahead serves same-dtype paged KV or nonpaged THD split"
        if not facts.has_paged_kv and knobs is not None and (knobs.split_kv or 1) <= 1:
            return "nonpaged THD shared lookahead requires split-KV"
    else:
        if facts.d_qk > 192:
            return "shared single-Q serves D32/D64/D128 and D192/V128"
        if facts.d_qk == 128 and not facts.thd and not facts.has_paged_kv and facts.attn_scale_prefolded:
            return "dense D128 shared single-Q does not support the pre-folded attention scale"
        if not facts.uniform_dtype and facts.d_qk != 128:
            return "shared single-Q mixed KV is supported at D128 only"
        if facts.d_qk == 192:
            if not facts.thd or facts.has_paged_kv or (knobs is not None and knobs.pack_gqa):
                return "D192 shared single-Q requires nonpaged unpacked THD"
            # The legacy single-CTA admission uses the packed-partial feature
            # domain even when this record is unsplit. Only the total-Q
            # workspace bound is relaxed for an explicit unsplit record.
            if facts.s_q <= 0 or facts.has_sink or facts.has_epilogue_gate:
                return "D192 shared single-Q requires positive query capacity without sink or epilogue gate"
            if facts.stats_t is not None and getattr(facts.stats_t, "ragged_offset", None) is None:
                return "D192 shared single-Q requires packed Stats when Stats is declared"
        elif facts.thd:
            if knobs is not None and (knobs.split_kv or 1) <= 1:
                return "small-head shared single-Q THD requires split-KV"
        elif facts.has_paged_kv:
            return "dense paged small-head queries use the native SM107 body"
    cga = explicit_geometry(implementation, facts, knobs).cga
    if knobs is not None and knobs.cga not in (None, cga):
        return f"the selected shared body requires TILE_CGA_M={cga}"
    return None
