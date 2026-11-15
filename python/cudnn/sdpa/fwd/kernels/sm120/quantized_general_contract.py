# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity and geometry shared by the SM120 per-tensor FP8 and MXFP8 general-tile owners.

Each owns its row's exact general head tiles (multiples of 32 up to 256 on both sides); the
sixteen-aligned envelopes keep the legacy rows. No performance policy or DSL imports live here.
"""

from dataclasses import dataclass, replace

from cudnn import data_type
from .fp8_general import contract as fp8
from .mxfp8_general import contract as mxfp8

FP8 = fp8.NAME
MXFP8 = mxfp8.NAME
_CONTRACTS = {FP8: fp8, MXFP8: mxfp8}
IMPLEMENTATIONS = frozenset(_CONTRACTS)
LEGACY_ROWS = frozenset(c.LEGACY_ROW for c in _CONTRACTS.values())
# The legacy rows' arch range (120..129).
DEVICE_CCS = tuple((12, minor) for minor in range(10))
# config_sm120.FP8_GENERAL_HEAD_TILES; a contract test keeps them equal.
HEAD_TILES = tuple(range(32, 257, 32))
SHAPES = frozenset((q, v) for q in HEAD_TILES for v in HEAD_TILES)


def owner(facts):
    """The explicit owner of these facts, or None."""
    if facts.device_cc not in DEVICE_CCS or (facts.d_qk, facts.d_v) not in SHAPES or facts.invalid or facts.is_backward:
        return None
    if facts.dtype not in (data_type.FP8_E4M3, data_type.FP8_E5M2):
        return None
    if facts.is_mxfp8:
        return MXFP8
    return FP8 if facts.is_fp8 else None


def in_domain(facts):
    return owner(facts) is not None


@dataclass(frozen=True)
class GeneralGeometry:
    implementation: str
    query_tile: int
    cga: int = 1

    @property
    def template(self):
        return _CONTRACTS[self.implementation].TEMPLATE_NAME

    @property
    def kernel_file(self):
        return _CONTRACTS[self.implementation].KERNEL_FILE

    @property
    def lowering_row_name(self):
        return _CONTRACTS[self.implementation].LEGACY_ROW

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=self.query_tile,
            cga=self.cga,
            sched_policy=0 if knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
        )

    def compiled_mismatch(self, template, params, knobs):
        """``(config, reason)``: the compiled body's identity and why it differs from this selection.

        These bodies have no CFG record; their template parameters carry the launch geometry. An
        unset TILE_N is the adapter's own fit and is reported, not compared."""
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
            kv_tile=params.kv_tile if knobs.tile_n is None else knobs.tile_n,
            scheduler=knobs.sched_policy,
            pack_gqa=bool(knobs.pack_gqa),
            split_kv=knobs.split_kv,
        )
        return config, (None if config == expected else f"expected {expected}")


def explicit_geometry(implementation, knobs):
    """Fixed defaults within an already-selected implementation; no crossover."""
    contract = _CONTRACTS[implementation]
    tile = contract.DEFAULT_QUERY_TILE if knobs is None or knobs.tile_m is None else knobs.tile_m
    return GeneralGeometry(implementation, tile)


def select_geometry(facts, knobs):
    return explicit_geometry(owner(facts), knobs)


def mismatch(implementation, facts, knobs):
    if facts.device_cc not in DEVICE_CCS:
        return f"requires SM12x (the SM120 family); current device is {facts.device_cc}"
    if owner(facts) != implementation:
        kind = "MXFP8" if implementation == MXFP8 else "per-tensor FP8"
        return f"requires {kind} exact general head tiles (multiples of 32 up to 256) on the SM120 family"
    if knobs is not None and knobs.tile_m not in (None, *_CONTRACTS[implementation].QUERY_TILES):
        return f"{implementation} runs TILE_M in {_CONTRACTS[implementation].QUERY_TILES}"
    if knobs is not None and knobs.cga not in (None, 1):
        return f"{implementation} runs one-CTA clusters (TILE_CGA_M=1)"
    return None
