# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity and geometry shared by the two exact-D256 SM120 half engines.

TILE_M names the body: 64 / 128 run the all-compute D256 body, 16 / 32 the
key-group decode body. No performance policy or DSL imports live here.
"""

from dataclasses import dataclass, replace

from cudnn import data_type
from .d256_f16_all_compute import contract as all_compute
from .decode_f16_key_group import contract as key_group

ALL_COMPUTE = all_compute.NAME
KEY_GROUP = key_group.NAME
_CONTRACTS = {ALL_COMPUTE: all_compute, KEY_GROUP: key_group}
IMPLEMENTATIONS = frozenset(_CONTRACTS)
LEGACY_ROW = all_compute.LEGACY_ROW
# The legacy row's arch range (120..129); exact D256 only, the eight-aligned
# (240, 256] envelope keeps the legacy owner.
DEVICE_CCS = tuple((12, minor) for minor in range(10))
SHAPE = (256, 256)


def in_domain(facts):
    return (
        facts.device_cc in DEVICE_CCS
        and (facts.d_qk, facts.d_v) == SHAPE
        and not (facts.invalid or facts.is_backward)
        and not (facts.is_fp8 or facts.is_mxfp8)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
    )


@dataclass(frozen=True)
class D256Geometry:
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
    return D256Geometry(implementation, tile)


def select_geometry(facts, knobs):
    """The body a legacy-policy candidate names: its TILE_M."""
    implementation = KEY_GROUP if knobs.tile_m in key_group.QUERY_TILES else ALL_COMPUTE
    return explicit_geometry(implementation, knobs)


def mismatch(implementation, facts, knobs):
    if facts.device_cc not in DEVICE_CCS:
        return f"requires SM12x (the SM120 family); current device is {facts.device_cc}"
    if not in_domain(facts):
        return "requires exact D256/V256 FP16/BF16 on the SM120 family"
    reason = _CONTRACTS[implementation].mismatch(facts, knobs)
    if reason is not None:
        return reason
    if knobs is not None and knobs.cga not in (None, 1):
        return f"{implementation} runs one-CTA clusters (TILE_CGA_M=1)"
    return None
