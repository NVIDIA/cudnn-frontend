# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed shared single-Q body; the initial graph owner admits its D192 leg."""

from dataclasses import dataclass, replace
from ..d192_half_contract import in_domain

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
            params, d_flavor=128, single_q_head_dim=192, cta_mma=1, decode_tile=False, decode_quad=False, decode_q_tile=0, early_q=False, dual_kv_stream=False
        )


def mismatch(facts, knobs):
    if not in_domain(facts):
        return "this single-Q owner currently requires exact half D192/V128 on SM100/SM103"
    if not facts.thd or facts.has_paged_kv:
        return "D192 single-Q requires nonpaged THD queries"
    if knobs is not None:
        if knobs.cga not in (None, 1) or knobs.tile_m not in (None, 128):
            return "D192 single-Q requires TILE_M=128 and TILE_CGA_M=1"
        if knobs.pack_gqa:
            return "D192 single-Q requires unpacked heads"
    return None
