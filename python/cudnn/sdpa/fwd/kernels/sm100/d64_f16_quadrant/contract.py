# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed quadrant cooperation and its live-row safety bound."""

from dataclasses import dataclass, replace

from ..small_half_contract import in_domain

NAME = "sdpa_fwd_sm100_d64_quadrant"
TEMPLATE_NAME = "decode_d64_f16_quadrant"
KERNEL_FILE = "sm100/d64_f16_quadrant/kernel.py"
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
        return replace(params, d_flavor=64, cta_mma=1, decode_tile=True, decode_quad=True, decode_q_tile=0, early_q=True, dual_kv_stream=False)


def mismatch(facts, knobs):
    if not in_domain(facts) or facts.d_qk > 64:
        return "quadrant half implementation requires the D64 envelope on SM100/SM103"
    if facts.thd or facts.has_bias or facts.has_kv_mask:
        return "quadrant cooperation does not support THD, bias or KV-mask operands"
    if facts.h_q <= 0 or facts.h_kv <= 0 or facts.h_q % facts.h_kv:
        return "quadrant cooperation requires an integral positive head ratio"
    pack_g = facts.h_q // facts.h_kv if knobs is not None and knobs.pack_gqa else 1
    if 32 % pack_g or not 0 < facts.s_q * pack_g <= 32:
        return "quadrant cooperation requires a packed group dividing 32 and at most 32 live Q rows"
    if knobs is not None and (knobs.cga not in (None, 1) or knobs.tile_m not in (None, 128) or knobs.tile_n not in (None, 128)):
        return "quadrant cooperation requires a 128x128 physical tile and one CTA"
    return None
