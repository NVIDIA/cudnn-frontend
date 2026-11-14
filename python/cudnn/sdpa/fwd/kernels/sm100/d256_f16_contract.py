# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight identities/geometry shared by the two exact-D256 half engines.

No performance policy, DSL imports or adapter state lives here. Heuristics
chooses a geometry; support and build consume that same immutable choice.
"""

from dataclasses import dataclass, replace

from cudnn import data_type
from cudnn.sdpa.fwd.config_common import pack_gqa_supported

LOOKAHEAD = "sdpa_fwd_sm100_d256_lookahead"
SWAP_AB = "sdpa_fwd_sm100_d256_swap_ab"
IMPLEMENTATIONS = frozenset({LOOKAHEAD, SWAP_AB})


def in_pilot_domain(facts):
    return (
        facts.device_cc in ((10, 0), (10, 3))
        and (facts.d_qk, facts.d_v) == (256, 256)
        and not (facts.invalid or facts.is_backward)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
        and not (facts.is_fp8 or facts.is_mxfp8)
        and facts.uniform_dtype
    )


@dataclass(frozen=True)
class D256Geometry:
    implementation: str
    query_tile: int

    @property
    def cga(self):
        return 1 if self.implementation == SWAP_AB else 2

    @property
    def decode_q_tile(self):
        return self.query_tile if self.implementation == SWAP_AB else 0

    @property
    def template(self):
        return "decode_d256_f16" if self.implementation == SWAP_AB else "prefill_d256_f16"

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=self.query_tile,
            tile_n=128 if knobs.tile_n is None else knobs.tile_n,
            cga=self.cga,
            sched_policy=0 if self.implementation == SWAP_AB or knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
        )


def explicit_geometry(implementation, knobs):
    """Fixed defaults within an already-selected implementation; no crossover."""
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown D256 implementation: {implementation}")
    default = 16 if implementation == SWAP_AB else 128
    return D256Geometry(implementation, default if knobs is None or knobs.tile_m is None else knobs.tile_m)


def mismatch(implementation, facts, knobs):
    """The implementation-specific part of support, after common graph checks."""
    if not in_pilot_domain(facts):
        return "this implementation serves exact D_qk=D_v=256 half graphs on SM100/SM103"
    if facts.has_epilogue_gate or facts.attn_scale_prefolded:
        return "the SM100 D256 half implementations do not carry a gate or pre-folded scale"
    geometry = explicit_geometry(implementation, knobs)
    if implementation == SWAP_AB:
        if facts.thd:
            return "the D256 swap-AB implementation requires dense queries; THD uses lookahead"
        if geometry.query_tile not in (16, 32):
            return "D256 swap-AB TILE_M (N_Q) must be 16 or 32"
        if facts.h_kv <= 0 or facts.h_q <= 0 or facts.h_q % facts.h_kv:
            return "D256 swap-AB requires positive heads and an integral Q/KV head ratio"
        group = facts.h_q // facts.h_kv if knobs is not None and knobs.pack_gqa else 1
        if group > geometry.query_tile:
            return "D256 swap-AB packs a whole KV-head group, which must fit TILE_M (N_Q)"
        if knobs is not None and knobs.pack_gqa and not pack_gqa_supported(facts.h_q, facts.h_kv, partial=True):
            return "D256 swap-AB retains the SM100 half PackGQA head-ratio domain"
        if knobs is not None and knobs.sched_policy not in (None, 0):
            return "D256 swap-AB has the direct grid only (SCHED_POLICY=0)"
    elif geometry.query_tile != 128:
        return "D256 lookahead TILE_M must be 128"
    if knobs is not None and knobs.cga not in (None, geometry.cga):
        return f"{implementation} requires TILE_CGA_M={geometry.cga}"
    return None
