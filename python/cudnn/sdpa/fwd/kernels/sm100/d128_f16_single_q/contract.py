# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed half single-Q body with D64/D128 and D192 geometry variants."""

from dataclasses import dataclass, replace
from ..d192_half_contract import in_domain as in_d192_domain
from ..small_half_contract import in_domain as in_small_domain

NAME = "sdpa_fwd_sm100_d128_single_q"
TEMPLATE_NAME = "decode_d128_f16"
KERNEL_FILE = "sm100/d128_f16_single_q/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm100"


@dataclass(frozen=True)
class Geometry:
    implementation: str = NAME
    template: str = TEMPLATE_NAME
    kernel_file: str = KERNEL_FILE
    lowering_row_name: str = LEGACY_ROW
    query_tile: int = 128
    cga: int = 1
    cluster_m: int = 1
    cta_group: int = 1
    decode_q_tile: int = 0
    d_flavor: int = 128
    head_dim: int = 192
    early_q: bool = False

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=self.query_tile,
            tile_n=128 if knobs.tile_n is None else knobs.tile_n,
            cga=self.cga,
            sched_policy=0 if knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
        )

    def configure_params(self, params):
        return replace(
            params,
            d_flavor=self.d_flavor,
            single_q_head_dim=self.head_dim,
            cta_mma=1,
            decode_tile=self.d_flavor == 64,
            decode_quad=False,
            decode_q_tile=0,
            early_q=self.early_q,
            dual_kv_stream=False,
        )


def small_geometry(facts, knobs, *, d_flavor=None):
    """Local geometry of an already-selected single-Q implementation."""
    d_flavor = (64 if max(facts.d_qk, facts.d_v) <= 64 else 128) if d_flavor is None else d_flavor
    early_q = facts.device_cc in ((10, 0), (10, 3)) and (not facts.thd or (facts.device_cc == (10, 0) and d_flavor == 128 and (knobs.split_kv or 1) > 1))
    return Geometry(d_flavor=d_flavor, head_dim=128, early_q=early_q)


def mismatch(facts, knobs):
    if not (in_d192_domain(facts) or in_small_domain(facts)):
        return "half single-Q requires supported dtypes and a D32/D64/D128 or D192/V128 covering geometry on SM100/SM103"
    if facts.has_bias or facts.has_kv_mask or facts.has_score_max or facts.has_score_sum_exp:
        return "half single-Q has no bias, KV-mask or score-output stores"
    if knobs is not None:
        if knobs.cga not in (None, 1) or knobs.tile_m not in (None, 128) or knobs.tile_n not in (None, 128):
            return "half single-Q requires a 128x128 physical tile and TILE_CGA_M=1"
    if in_d192_domain(facts):
        if not facts.thd or facts.has_paged_kv:
            return "D192 single-Q requires nonpaged THD queries"
        if knobs is not None and knobs.pack_gqa:
            return "D192 single-Q requires unpacked heads"
    elif facts.thd and knobs is not None and (knobs.split_kv or 1) == 1:
        if max(facts.d_qk, facts.d_v) > 64 or not facts.has_paged_kv or knobs.pack_gqa or facts.s_q > 128:
            return "small-head single-Q THD requires split-KV, or unpacked paged D64 with at most 128 query rows"
    return None
