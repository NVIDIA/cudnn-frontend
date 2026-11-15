# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity and geometry shared by the SM120 half engines on exact 16-wide head tiles.

TILE_M names the body: 16 / 32 run the key-group decode body on every tile;
64 / 128 run the all-compute D256 body at (256, 256) and the general body
elsewhere. No performance policy or DSL imports live here.
"""

from dataclasses import dataclass, replace

from cudnn import data_type
from .d256_f16_all_compute import contract as all_compute
from .decode_f16_key_group import contract as key_group
from .f16_general import contract as general

ALL_COMPUTE = all_compute.NAME
GENERAL = general.NAME
KEY_GROUP = key_group.NAME
_CONTRACTS = {ALL_COMPUTE: all_compute, GENERAL: general, KEY_GROUP: key_group}
IMPLEMENTATIONS = frozenset(_CONTRACTS)
LEGACY_ROW = general.LEGACY_ROW
# The legacy row's arch range (120..129). Eight-aligned heads pad to the next 16-wide tile, as on the row.
DEVICE_CCS = tuple((12, minor) for minor in range(10))
# config_sm120.GENERAL_HEAD_TILES; a test pins the equality.
HEAD_TILES = tuple(range(16, 257, 16))
D256 = (256, 256)
SHAPES = frozenset((q, v) for q in HEAD_TILES for v in HEAD_TILES)
_SHAPES = {ALL_COMPUTE: frozenset({D256}), GENERAL: SHAPES - {D256}, KEY_GROUP: SHAPES}


def padded_shape(facts):
    """The 16-wide tile an eight-aligned head pair runs on (the legacy row's envelope), or None."""
    if facts.d_qk % 8 or facts.d_v % 8:
        return None
    shape = (-(-facts.d_qk // 16) * 16, -(-facts.d_v // 16) * 16)
    return shape if shape in SHAPES else None


def in_domain(facts):
    return (
        facts.device_cc in DEVICE_CCS
        and padded_shape(facts) is not None
        and not (facts.invalid or facts.is_backward)
        and not (facts.is_fp8 or facts.is_mxfp8)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
    )


@dataclass(frozen=True)
class HalfGeometry:
    implementation: str
    query_tile: int
    lowering_row_name: str = LEGACY_ROW
    cga: int = 1

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
    return HalfGeometry(implementation, tile)


def select_geometry(facts, knobs):
    """The body a legacy-policy candidate names: its TILE_M, then the head tile."""
    if knobs.tile_m in key_group.QUERY_TILES:
        implementation = KEY_GROUP
    else:
        implementation = ALL_COMPUTE if padded_shape(facts) == D256 else GENERAL
    return explicit_geometry(implementation, knobs)


def mismatch(implementation, facts, knobs):
    if facts.device_cc not in DEVICE_CCS:
        return f"requires SM12x (the SM120 family); current device is {facts.device_cc}"
    if not in_domain(facts) or padded_shape(facts) not in _SHAPES[implementation]:
        shapes = "D256/V256 (eight-aligned heads above 240)" if implementation == ALL_COMPUTE else "eight-aligned heads on 16-wide tiles up to 256"
        return f"{implementation} requires {shapes} FP16/BF16 on the SM120 family"
    reason = _CONTRACTS[implementation].mismatch(facts, knobs)
    if reason is not None:
        return reason
    if knobs is not None and knobs.cga not in (None, 1):
        return f"{implementation} runs one-CTA clusters (TILE_CGA_M=1)"
    return None
