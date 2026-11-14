# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Identity of the small-head FP8 dual_kv implementation."""

from dataclasses import dataclass, replace

from ..small_fp8_contract import in_domain

NAME = "sdpa_fwd_sm100_d128_fp8_dual_kv"
TEMPLATE_NAME = "prefill_d128_fp8"
KERNEL_FILE = "sm100/d128_fp8_dual_kv/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm100_fp8"


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
            swap_ab=None,
        )

    def configure_params(self, params):
        return replace(params, d_flavor=128, cta_mma=1, dual_kv_stream=True, decode_q_tile=0)


def mismatch(facts, knobs):
    if not in_domain(facts) or facts.d_qk != 128:
        return "dual-KV requires exact D128 per-tensor FP8 on SM100/SM103"
    if facts.thd:
        return "dual-KV requires dense queries"
    if knobs is not None:
        if knobs.cga not in (None, 1):
            return "dual-KV requires TILE_CGA_M=1"
        if knobs.tile_m not in (None, 128):
            return "dual-KV requires TILE_M=128"
    return None
