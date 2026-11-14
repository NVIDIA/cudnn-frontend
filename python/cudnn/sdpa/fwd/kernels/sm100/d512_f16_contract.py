# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable wide-half implementation geometry; no performance policy or runtime imports."""

from dataclasses import dataclass, replace

from cudnn import data_type
from .d512_f16_role_split import contract as role_split
from .d512_f16_2x2 import contract as two_by_two

ROLE_SPLIT = role_split.NAME
TWO_BY_TWO = two_by_two.NAME
_CONTRACTS = {ROLE_SPLIT: role_split, TWO_BY_TWO: two_by_two}
IMPLEMENTATIONS = frozenset(_CONTRACTS)


def in_domain(facts):
    return (
        facts.device_cc in ((10, 0), (10, 3))
        and (facts.d_qk, facts.d_v) in ((512, 512), (576, 512))
        and not (facts.invalid or facts.is_backward)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
        and not (facts.is_fp8 or facts.is_mxfp8)
        and facts.uniform_dtype
    )


@dataclass(frozen=True)
class D512Geometry:
    implementation: str
    d_qk: int = 512

    @property
    def query_tile(self):
        return _CONTRACTS[self.implementation].QUERY_TILE

    @property
    def cga(self):
        # Preserve the existing public knob's MMA-group interpretation. The
        # full launch cluster is separate and checked against the compiled CFG.
        return _CONTRACTS[self.implementation].CTA_MMA

    @property
    def cluster_m(self):
        return role_split.CLUSTER_M if self.implementation == ROLE_SPLIT else two_by_two.cluster_m(self.d_qk)

    @property
    def decode_q_tile(self):
        return 0

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
            cga=self.cga,
            sched_policy=0 if knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
        )

    def configure_params(self, params):
        return replace(params, mma_2x2=self.implementation == TWO_BY_TWO, d_flavor=self.d_qk)


def explicit_geometry(implementation, facts):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown wide half implementation: {implementation}")
    return D512Geometry(implementation, facts.d_qk)


def mismatch(implementation, facts, knobs):
    if facts.device_cc not in ((10, 0), (10, 3)):
        return f"requires SM100/SM103; current device is {facts.device_cc}"
    if not in_domain(facts):
        return "this implementation serves exact D512/V512 or D576/V512 half graphs on SM100/SM103"
    reason = _CONTRACTS[implementation].mismatch(facts, knobs)
    if reason is not None:
        return reason
    if knobs is not None and knobs.cga not in (None, 2):
        return "wide half implementations require a two-CTA MMA group (TILE_CGA_M=2)"
    return None
