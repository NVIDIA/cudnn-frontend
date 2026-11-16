# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Identity and geometry of the half D192 multi-Q pipeline."""

from dataclasses import dataclass, replace
from ..d192_half_contract import in_domain

NAME = "sdpa_fwd_sm100_d192_multi_q"
TEMPLATE_NAME = "prefill_d192_d128_f16"
KERNEL_FILE = "sm100/d192_f16_multi_q/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm100"


@dataclass(frozen=True)
class Geometry:
    implementation: str = NAME
    template: str = TEMPLATE_NAME
    kernel_file: str = KERNEL_FILE
    lowering_row_name: str = LEGACY_ROW
    query_tile: int = 128
    cga: int = 2
    decode_q_tile: int = 0

    @property
    def cluster_m(self):
        return self.cga

    @property
    def cta_group(self):
        return self.cga

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
        return replace(params, cta_mma=self.cga, decode_tile=False, decode_quad=False, decode_q_tile=0, early_q=False, dual_kv_stream=False)


def mismatch(facts, knobs):
    if not in_domain(facts):
        return "D192 multi-Q requires exact half D192/V128 on SM100/SM103"
    if knobs is not None:
        if knobs.cga not in (None, 1, 2) or knobs.tile_m not in (None, 128):
            return "D192 multi-Q requires TILE_M=128 and TILE_CGA_M=1 or 2"
        if facts.thd and not facts.has_paged_kv and knobs.cga == 1:
            return "nonpaged THD with one CTA belongs to the single-Q implementation"
    return None
