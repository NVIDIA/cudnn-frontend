# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact SM107 MXFP8 owners: one body per native head shape; no performance policy or runtime imports."""

from dataclasses import dataclass

from .d128_mxfp8_multi_q import contract as d128
from .d192_mxfp8_multi_q import contract as d192
from .d256_mxfp8_single_q import contract as d256
from .d512_mxfp8_role_split import contract as d512

LEGACY_ROW = "sdpa_fwd_prefill_sm107_mxfp8"
DEVICE_CCS = ((10, 7),)
QUERY_TILE = 128
_BY_SHAPE = {c.SHAPE: c for c in (d128, d192, d256, d512)}
_CONTRACTS = {c.NAME: c for c in _BY_SHAPE.values()}
IMPLEMENTATIONS = frozenset(_CONTRACTS)
SHAPES = frozenset(_BY_SHAPE)


def owner_of_shape(shape):
    return _BY_SHAPE[shape].NAME


def owner(facts):
    """The MXFP8 body that serves this graph's head shape on SM107, or None."""
    if facts.device_cc not in DEVICE_CCS or not facts.is_mxfp8 or facts.invalid or facts.is_backward:
        return None
    contract = _BY_SHAPE.get((facts.d_qk, facts.d_v))
    return None if contract is None else contract.NAME


def in_domain(facts):
    return owner(facts) is not None


@dataclass(frozen=True)
class Geometry:
    implementation: str
    cga: object = None
    lowering_row_name: str = LEGACY_ROW
    query_tile: int = QUERY_TILE
    decode_q_tile: int = 0

    @property
    def template(self):
        return _CONTRACTS[self.implementation].TEMPLATE_NAME

    @property
    def kernel_file(self):
        return _CONTRACTS[self.implementation].KERNEL_FILE

    @property
    def cta_group(self):
        return self.cga

    def canonical_knobs(self, knobs):
        # Unset knobs stay unset: the adapter derives them as it does for the legacy row.
        return knobs

    def compiled_mismatch(self, template, params, knobs):
        """``(config, reason)``: the compiled body's identity and why it differs from this selection."""
        config = dict(template=template, cta_group=params.cta_mma, scheduler=params.sched_policy, split_kv=params.split_kv)
        expected = dict(config, template=self.template)
        for key, value in (("cta_group", self.cga), ("scheduler", knobs.sched_policy), ("split_kv", knobs.split_kv)):
            if value is not None:
                expected[key] = value
        return config, (None if config == expected else f"expected {expected}")


def explicit_geometry(implementation, knobs=None):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown SM107 MXFP8 implementation: {implementation}")
    # A record's CTA width stays its own; unset, the adapter derives it as on the row.
    return Geometry(implementation, None if knobs is None else knobs.cga)


def select_geometry(facts, knobs):
    """The body a legacy-policy candidate runs: the owner of the head shape."""
    return explicit_geometry(owner(facts), knobs)


def mismatch(implementation, facts, knobs):
    if facts.device_cc not in DEVICE_CCS:
        return f"requires SM107; current device is {facts.device_cc}"
    if owner(facts) != implementation:
        (shape,) = (c.SHAPE for c in _CONTRACTS.values() if c.NAME == implementation)
        return f"{implementation} serves exact D_QK/D_V {shape} MXFP8 graphs on SM107"
    if knobs is not None and knobs.tile_m not in (None, QUERY_TILE):
        return f"SM107 MXFP8 requires TILE_M={QUERY_TILE}"
    return None
