# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed per-tensor FP8 D192/V128 identity and geometry; no runtime imports."""

from dataclasses import dataclass, replace

from cudnn import data_type

NAME = "sdpa_fwd_sm100_d192_fp8_multi_q"
TEMPLATE_NAME = "prefill_d192_d128_fp8"
KERNEL_FILE = "sm100/d192_fp8_multi_q/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm100_fp8"


def in_domain(facts):
    return (
        facts.device_cc in ((10, 0), (10, 3))
        and (facts.d_qk, facts.d_v) == (192, 128)
        and not (facts.invalid or facts.is_backward)
        and facts.is_fp8
        and not facts.is_mxfp8
        and facts.dtype in (data_type.FP8_E4M3, data_type.FP8_E5M2)
        and facts.uniform_dtype
    )


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
        return replace(params, mma_2x2=False, d_flavor=192)


def mismatch(facts, knobs):
    if facts.device_cc not in ((10, 0), (10, 3)):
        return f"requires SM100/SM103; current device is {facts.device_cc}"
    if not in_domain(facts):
        return "requires exact D192/V128 per-tensor FP8 on SM100/SM103"
    if knobs is not None:
        if knobs.cga not in (None, 1, 2):
            return "FP8 D192/V128 multi-Q requires TILE_CGA_M=1 or 2 (physical launch cluster)"
        if knobs.tile_m not in (None, 128):
            return "FP8 D192/V128 multi-Q requires TILE_M=128"
    return None
