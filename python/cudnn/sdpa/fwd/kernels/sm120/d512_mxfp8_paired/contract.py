# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed SM120 wide-head block-scaled MXFP8 identity and geometry; no runtime imports."""

from dataclasses import dataclass, replace

from cudnn import data_type

NAME = "sdpa_fwd_sm120_d512_mxfp8_paired"
TEMPLATE_NAME = "prefill_d512_mxfp8"
KERNEL_FILE = "sm120/d512_mxfp8_paired/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm120_mxfp8"
# The legacy row's arch range (120..129); sixteen-aligned heads above the floor pad onto D512/V512, as on the row.
DEVICE_CCS = tuple((12, minor) for minor in range(10))
SHAPE = (512, 512)
ENVELOPE_FLOOR = 256


def padded_shape(facts):
    """D512/V512 when the head pair is exact or sixteen-aligned with both heads in (256, 512], else None."""
    if (facts.d_qk, facts.d_v) == SHAPE:
        return SHAPE
    if facts.d_qk % 16 or facts.d_v % 16 or min(facts.d_qk, facts.d_v) <= ENVELOPE_FLOOR or max(facts.d_qk, facts.d_v) > 512:
        return None
    return SHAPE


def in_domain(facts):
    return (
        facts.device_cc in DEVICE_CCS
        and padded_shape(facts) is not None
        and not (facts.invalid or facts.is_backward)
        and facts.is_mxfp8
        and facts.dtype in (data_type.FP8_E4M3, data_type.FP8_E5M2)
    )


@dataclass(frozen=True)
class Geometry:
    implementation: str = NAME
    template: str = TEMPLATE_NAME
    kernel_file: str = KERNEL_FILE
    lowering_row_name: str = LEGACY_ROW
    query_tile: int = 64
    kv_tile: int = 64
    cga: int = 1

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=self.query_tile,
            tile_n=self.kv_tile,
            cga=self.cga,
            sched_policy=0 if knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
        )

    def compiled_mismatch(self, template, params, knobs):
        """``(config, reason)``: the compiled body's identity and why it differs from this selection.

        The body has no CFG record; its template parameters carry the launch geometry."""
        config = dict(
            template=template,
            query_tile=params.q_tile,
            kv_tile=params.kv_tile,
            scheduler=params.sched_policy,
            pack_gqa=bool(params.pack_gqa),
            split_kv=params.split_kv,
        )
        expected = dict(
            template=self.template,
            query_tile=self.query_tile,
            kv_tile=self.kv_tile,
            scheduler=knobs.sched_policy,
            pack_gqa=bool(knobs.pack_gqa),
            split_kv=knobs.split_kv,
        )
        return config, (None if config == expected else f"expected {expected}")


def mismatch(facts, knobs):
    if facts.device_cc not in DEVICE_CCS:
        return f"requires SM12x (the SM120 family); current device is {facts.device_cc}"
    if not in_domain(facts):
        return "requires D512/V512 MXFP8 (sixteen-aligned heads above 256) on the SM120 family"
    if knobs is not None:
        if knobs.tile_m not in (None, 64) or knobs.tile_n not in (None, 64):
            return "SM120 wide-head MXFP8 paired-warp requires TILE_M=64 and TILE_N=64"
        if knobs.cga not in (None, 1):
            return "SM120 wide-head MXFP8 paired-warp runs one-CTA clusters (TILE_CGA_M=1)"
    return None
