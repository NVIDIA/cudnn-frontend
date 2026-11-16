# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain and geometry shared by the SM80 half engines.

SM80 admits eight-aligned head dims and runs the smallest flavor envelope that
covers them (config_sm80.pick_flavor). Each engine owns the envelopes its body
serves, so the head dims name the body; the adapter keeps choosing the tile
from the shape. No performance policy or DSL imports live here.
"""

from dataclasses import dataclass

from cudnn import data_type
from .d256_kv_prefetch import contract as d256_kv_prefetch
from .f16_k_prefetch import contract as k_prefetch
from .wide_cooperative import contract as wide_cooperative

LEGACY_ROW = k_prefetch.LEGACY_ROW
DEVICE_CCS = ((8, 0),)
# config_sm80.FLAVOR_CFGS (D_QK, D_V) in pick order; a test pins the equality.
FLAVOR_DIMS = (("gptoss", (64, 64)), ("llama", (128, 128)), ("dsv3", (192, 128)), ("qwen", (256, 256)), ("wide512", (512, 512)), ("mla576", (576, 512)))
_CONTRACTS = {c.NAME: c for c in (k_prefetch, d256_kv_prefetch, wide_cooperative)}
IMPLEMENTATIONS = frozenset(_CONTRACTS)
_OWNER_OF_FLAVOR = {flavor: c.NAME for c in _CONTRACTS.values() for flavor in c.FLAVORS}


def pick_flavor(d_qk, d_v):
    return next((name for name, (q, v) in FLAVOR_DIMS if d_qk <= q and d_v <= v), None)


def implementation_for(d_qk, d_v):
    """The engine whose envelope serves these head dims, or None outside SM80's eight-aligned domain."""
    if d_qk <= 0 or d_v <= 0 or d_qk % 8 or d_v % 8:
        return None
    flavor = pick_flavor(d_qk, d_v)
    return None if flavor is None else _OWNER_OF_FLAVOR[flavor]


_D_QK_MAX, _D_V_MAX = max(q for _, (q, _) in FLAVOR_DIMS), max(v for _, (_, v) in FLAVOR_DIMS)
SHAPES = {
    name: frozenset((q, v) for q in range(8, _D_QK_MAX + 1, 8) for v in range(8, _D_V_MAX + 1, 8) if implementation_for(q, v) == name)
    for name in IMPLEMENTATIONS
}


def owner(facts):
    """The SM80 half engine that owns this graph's head dims, or None."""
    if facts.device_cc not in DEVICE_CCS or facts.invalid or facts.is_backward or facts.is_fp8 or facts.is_mxfp8:
        return None
    if facts.dtype not in (data_type.HALF, data_type.BFLOAT16):
        return None
    return implementation_for(facts.d_qk, facts.d_v)


def in_domain(facts):
    return owner(facts) is not None


@dataclass(frozen=True)
class Geometry:
    implementation: str
    lowering_row_name: str = LEGACY_ROW

    @property
    def template(self):
        return _CONTRACTS[self.implementation].TEMPLATE_NAME

    @property
    def kernel_file(self):
        return _CONTRACTS[self.implementation].KERNEL_FILE

    def canonical_knobs(self, knobs):
        # Tiles stay unset unless requested: the adapter's shape rule chooses them, as on the legacy row.
        return knobs

    def compiled_mismatch(self, template, params, knobs):
        """``(config, reason)``: the compiled body's identity and why it differs from this selection."""
        config = dict(
            template=template,
            d_qk=params.d_qk,
            d_v=params.d_v,
            query_tile=params.tile_m,
            kv_tile=params.tile_n,
            scheduler=params.sched_policy,
            split_kv=params.split_kv,
        )
        expected = dict(config, template=self.template)
        for key, value in (("query_tile", knobs.tile_m), ("kv_tile", knobs.tile_n), ("scheduler", knobs.sched_policy), ("split_kv", knobs.split_kv)):
            if value is not None:
                expected[key] = value
        return config, (None if config == expected else f"expected {expected}")


def explicit_geometry(implementation, knobs=None):
    """The fixed body; no crossover."""
    return Geometry(implementation)


def select_geometry(facts, knobs):
    """The body a legacy-policy candidate runs: the owner of the head dims."""
    return Geometry(owner(facts))


def mismatch(implementation, facts, knobs):
    if facts.device_cc not in DEVICE_CCS:
        return f"requires SM80 (A100); current device is {facts.device_cc}"
    if owner(facts) != implementation:
        flavors = ", ".join(_CONTRACTS[implementation].FLAVORS)
        return f"{implementation} serves eight-aligned FP16/BF16 head dims in the {flavors} flavor envelopes"
    return None
