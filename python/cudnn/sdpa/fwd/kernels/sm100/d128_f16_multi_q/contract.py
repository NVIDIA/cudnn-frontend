# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed half multi-Q implementation with D32/D64/D128 geometry variants."""

from dataclasses import dataclass, replace

from ..small_half_contract import in_domain

NAME = "sdpa_fwd_sm100_d128_multi_q"
TEMPLATE_NAME = "prefill_d128_f16"
KERNEL_FILE = "sm100/d128_f16_multi_q/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm100"


@dataclass(frozen=True)
class Geometry:
    d_flavor: int = 128
    cga: int = 2
    tile_n: int = 128
    implementation: str = NAME
    template: str = TEMPLATE_NAME
    kernel_file: str = KERNEL_FILE
    lowering_row_name: str = LEGACY_ROW
    query_tile: int = 128
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
            tile_n=self.tile_n if knobs.tile_n is None else knobs.tile_n,
            cga=self.cga,
            sched_policy=0 if knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
        )

    def configure_params(self, params):
        return replace(
            params, d_flavor=self.d_flavor, cta_mma=self.cga, decode_tile=False, decode_quad=False, decode_q_tile=0, early_q=False, dual_kv_stream=False
        )


def mismatch(facts, knobs):
    if not in_domain(facts):
        return "small-head half multi-Q requires supported dtypes and exact D32/D64/D128 on SM100/SM103"
    if facts.thd and knobs is not None and (knobs.split_kv or 1) > 1:
        return "THD split-KV requires the single-Q implementation"
    if knobs is not None and (knobs.cga not in (None, 1, 2) or knobs.tile_m not in (None, 128)):
        return "half multi-Q requires TILE_M=128 and one or two CTAs"
    return None
