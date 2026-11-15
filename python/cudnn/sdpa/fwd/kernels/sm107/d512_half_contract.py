# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Immutable SM107 wide-half geometry; performance selection lives in heuristics."""

from dataclasses import dataclass, replace
from cudnn import data_type
from .d512_f16_role_split import contract as role_split
from .d512_f16_2x2 import contract as two_by_two

LEGACY_ROW = "sdpa_fwd_prefill_sm107"
ROLE_SPLIT = role_split.NAME
TWO_BY_TWO = two_by_two.NAME
_CONTRACTS = {ROLE_SPLIT: role_split, TWO_BY_TWO: two_by_two}
IMPLEMENTATIONS = frozenset(_CONTRACTS)


def in_domain(facts):
    return (
        facts.device_cc == (10, 7)
        and (facts.d_qk, facts.d_v) in ((512, 512), (576, 512))
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
        and not (facts.invalid or facts.is_backward or facts.is_fp8 or facts.is_mxfp8)
    )


@dataclass(frozen=True)
class Geometry:
    implementation: str
    d_qk: int = 512
    mixed_kv: bool = False
    lowering_row_name = LEGACY_ROW
    decode_q_tile = 0
    cta_group = 2

    @property
    def query_tile(self):
        return _CONTRACTS[self.implementation].QUERY_TILE

    @property
    def cluster_m(self):
        return 2 if self.implementation == TWO_BY_TWO and (self.d_qk == 576 or self.mixed_kv) else 4

    @property
    def cga(self):
        return self.cluster_m

    @property
    def template(self):
        return _CONTRACTS[self.implementation].TEMPLATE_NAME

    @property
    def kernel_file(self):
        return _CONTRACTS[self.implementation].KERNEL_FILE

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=self.query_tile,
            tile_n=128 if knobs.tile_n is None else knobs.tile_n,
            cga=self.cluster_m,
            sched_policy=0 if knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
        )

    def configure_params(self, params):
        return replace(params, mma_2x2=self.implementation == TWO_BY_TWO, d_flavor=self.d_qk)


def explicit_geometry(implementation, facts):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown SM107 wide-half implementation: {implementation}")
    return Geometry(implementation, facts.d_qk, not facts.uniform_dtype)


def mismatch(implementation, facts, knobs):
    if not in_domain(facts):
        return "requires exact SM107 D512/V512 or D576/V512 half graphs"
    reason = _CONTRACTS[implementation].mismatch(facts, knobs)
    if reason is not None:
        return reason
    cluster = explicit_geometry(implementation, facts).cluster_m
    if knobs is not None and knobs.cga not in (None, cluster):
        return f"SM107 wide-half requires TILE_CGA_M={cluster} (physical launch cluster)"
    return None
