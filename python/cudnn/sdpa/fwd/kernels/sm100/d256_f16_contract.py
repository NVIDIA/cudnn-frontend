# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight identities/geometry for lookahead and half swap-AB engines.

No performance policy, DSL imports or adapter state lives here. Heuristics
chooses a geometry; support and build consume that same immutable choice.
"""

from dataclasses import dataclass, replace

from cudnn import data_type
from .half_flavor_contract import padded_shape
from .d256_f16_lookahead import contract as lookahead
from .d256_f16_swap_ab import contract as swap_ab

LOOKAHEAD = lookahead.NAME
SWAP_AB = swap_ab.NAME
_CONTRACTS = {LOOKAHEAD: lookahead, SWAP_AB: swap_ab}
IMPLEMENTATIONS = frozenset({LOOKAHEAD, SWAP_AB})


def in_pilot_domain(facts):
    return (
        facts.device_cc in ((10, 0), (10, 3))
        and padded_shape(facts) == (256, 256)
        and not (facts.invalid or facts.is_backward)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
        and not (facts.is_fp8 or facts.is_mxfp8)
        and facts.uniform_dtype
    )


@dataclass(frozen=True)
class D256Geometry:
    implementation: str
    query_tile: int
    head_dim: int = 256
    live_columns: int = 0
    early_q: bool = False
    lowering_row_name: str = "sdpa_fwd_prefill_sm100"

    @property
    def cga(self):
        return _CONTRACTS[self.implementation].CGA

    @property
    def decode_q_tile(self):
        return self.query_tile if self.implementation == SWAP_AB else 0

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
            sched_policy=0 if self.implementation == SWAP_AB or knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
        )

    def configure_params(self, params):
        if self.head_dim == 256:
            return params
        from cudnn.sdpa.fwd.config_fp8_decode_common import TemplateParamsDecode

        values = vars(replace(params, decode_q_tile=self.query_tile, dual_kv_stream=False, early_q=self.early_q)).copy()
        values.update(decode_head_dim=self.head_dim, decode_live_columns=self.live_columns)
        return TemplateParamsDecode(**values)


def explicit_geometry(implementation, knobs, *, facts=None):
    """Fixed defaults within an already-selected implementation; no crossover."""
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown D256 implementation: {implementation}")
    default = 16 if implementation == SWAP_AB else 128
    if implementation == SWAP_AB and facts is not None and padded_shape(facts) == (128, 128):
        return D256Geometry(implementation, default if knobs is None or knobs.tile_m is None else knobs.tile_m, head_dim=128, live_columns=8, early_q=True)
    return D256Geometry(implementation, default if knobs is None or knobs.tile_m is None else knobs.tile_m)


def mismatch(implementation, facts, knobs):
    """The implementation-specific part of support, after common graph checks."""
    if facts.device_cc not in ((10, 0), (10, 3)):
        return f"requires SM100/SM103; current device is {facts.device_cc}"
    from .small_half_contract import in_domain as in_small_domain

    small_swap = implementation == SWAP_AB and padded_shape(facts) == (128, 128) and in_small_domain(facts) and facts.uniform_dtype
    if not (in_pilot_domain(facts) or small_swap):
        return "this implementation serves the D256 half envelope on SM100/SM103"
    if facts.has_epilogue_gate or facts.attn_scale_prefolded:
        return "the SM100 D256 half implementations do not carry a gate or pre-folded scale"
    geometry = explicit_geometry(implementation, knobs, facts=facts)
    reason = _CONTRACTS[implementation].mismatch(facts, knobs)
    if reason is not None:
        return reason
    if knobs is not None and knobs.cga not in (None, geometry.cga):
        return f"{implementation} requires TILE_CGA_M={geometry.cga}"
    return None
