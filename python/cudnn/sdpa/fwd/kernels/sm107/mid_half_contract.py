# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed native mid-width half descriptors; no performance selection or DSL imports."""

from dataclasses import dataclass, replace
from cudnn import data_type
from .d192_f16_multi_q import contract as d192
from .d256_f16_lookahead import contract as lookahead
from .d256_f16_swap_ab import contract as swap_ab
from .half_flavor_contract import padded_shape

LEGACY_ROW = "sdpa_fwd_prefill_sm107"
D192 = d192.NAME
LOOKAHEAD = lookahead.NAME
SWAP_AB = swap_ab.NAME
_CONTRACTS = {D192: d192, LOOKAHEAD: lookahead, SWAP_AB: swap_ab}
IMPLEMENTATIONS = frozenset(_CONTRACTS)


@dataclass(frozen=True)
class Geometry:
    implementation: str
    d_qk: int
    query_tile: int = 128
    lowering_row_name = LEGACY_ROW

    @property
    def cga(self):
        return 1 if self.implementation == SWAP_AB or self.d_qk == 128 else 2

    @property
    def cta_group(self):
        return self.cga

    @property
    def cluster_m(self):
        return self.cga

    @property
    def template(self):
        return _CONTRACTS[self.implementation].TEMPLATE_NAME

    @property
    def kernel_file(self):
        return _CONTRACTS[self.implementation].KERNEL_FILE

    @property
    def decode_q_tile(self):
        return self.query_tile if self.implementation == SWAP_AB else 0

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=self.query_tile,
            tile_n=128 if knobs.tile_n is None else knobs.tile_n,
            cga=self.cga,
            sched_policy=0 if self.implementation == SWAP_AB or knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
            warp_spec_cfg=1 if self.implementation == LOOKAHEAD and self.d_qk == 128 else (knobs.warp_spec_cfg or 0),
        )

    def configure_params(self, params):
        # The existing D192 and D256 factories retain generic d_flavor=128.
        # Their local geometry determines operand widths; do not rewrite it.
        if self.implementation == LOOKAHEAD and self.d_qk == 128:
            return replace(params, softmax_wgs=1, s_slots=2, cta_mma=1)
        if self.implementation == SWAP_AB:
            return replace(params, decode_q_tile=self.query_tile, cta_mma=1)
        return params


def explicit_geometry(implementation, facts, knobs):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown native SM107 mid-width half implementation: {implementation}")
    tile = (knobs.tile_m or 16) if implementation == SWAP_AB else 128
    return Geometry(implementation, padded_shape(facts)[0], tile)


def mismatch(implementation, facts, knobs):
    if facts.device_cc != (10, 7) or facts.dtype not in (data_type.HALF, data_type.BFLOAT16):
        return "requires SM107 and FP16/BF16 queries for native mid-width half implementations"
    if facts.invalid or facts.is_backward or facts.is_fp8 or facts.is_mxfp8:
        return "requires a valid half forward graph"
    return _CONTRACTS[implementation].mismatch(facts, knobs)
