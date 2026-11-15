# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact D512 MXFP8 owners on SM100/SM103 and the facts that name each body; no performance policy or runtime imports."""

from dataclasses import dataclass, replace

from .d512_mxfp8_retained_p import contract as retained_p
from .d512_mxfp8_slice_recompute import contract as slice_recompute

SLICE_RECOMPUTE = slice_recompute.NAME
RETAINED_P = retained_p.NAME
_CONTRACTS = {SLICE_RECOMPUTE: slice_recompute, RETAINED_P: retained_p}
IMPLEMENTATIONS = frozenset(_CONTRACTS)
LEGACY_ROW = "sdpa_fwd_prefill_sm100_mxfp8"
DEVICE_CCS = ((10, 0), (10, 3))
SHAPE = (512, 512)
KV_TILE = 128


def in_domain(facts):
    return facts.device_cc in DEVICE_CCS and (facts.d_qk, facts.d_v) == SHAPE and facts.is_mxfp8 and not (facts.invalid or facts.is_backward)


def retains_p(facts, split_kv):
    """config_sm100.d512_mxfp8_retains_p in graph terms: top-left causal with a left window of one KV tile, no widened
    right band, no dense per-batch KV lengths, unsplit."""
    return (
        facts.causal
        and not facts.right_band_widening
        and not facts.bottom_right
        and facts.window_left == KV_TILE - 1
        and (facts.thd or not facts.padded)
        and split_kv == 1
    )


def owner(facts, knobs):
    """The body these facts and knobs name; an unset SPLIT_KV runs unsplit."""
    split_kv = 1 if knobs is None or knobs.split_kv is None else knobs.split_kv
    return RETAINED_P if retains_p(facts, split_kv) else SLICE_RECOMPUTE


@dataclass(frozen=True)
class Geometry:
    implementation: str
    lowering_row_name: str = LEGACY_ROW
    query_tile: int = 128
    cga: int = 1
    cluster_m: int = 1
    cta_group: int = 1
    decode_q_tile: int = 0

    @property
    def template(self):
        return _CONTRACTS[self.implementation].TEMPLATE_NAME

    @property
    def kernel_file(self):
        return _CONTRACTS[self.implementation].KERNEL_FILE

    def canonical_knobs(self, knobs):
        # The scheduler stays unset unless requested: the adapter derives it for D512 MXFP8, as on the legacy row.
        return replace(
            knobs,
            tile_m=self.query_tile,
            tile_n=KV_TILE if knobs.tile_n is None else knobs.tile_n,
            cga=self.cga,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
        )

    def compiled_mismatch(self, template, params, knobs):
        """``(config, reason)``: the compiled body's identity and why it differs from this selection."""
        config = dict(template=template, cta_group=params.cta_mma, scheduler=params.sched_policy, split_kv=params.split_kv)
        expected = dict(config, template=self.template, cta_group=self.cta_group)
        for key, value in (("scheduler", knobs.sched_policy), ("split_kv", knobs.split_kv)):
            if value is not None:
                expected[key] = value
        return config, (None if config == expected else f"expected {expected}")


def explicit_geometry(implementation):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown D512 MXFP8 implementation: {implementation}")
    return Geometry(implementation)


def select_geometry(facts, knobs):
    """The body a legacy-policy candidate runs."""
    return Geometry(owner(facts, knobs))


def mismatch(implementation, facts, knobs):
    if facts.device_cc not in DEVICE_CCS:
        return f"requires SM100/SM103; current device is {facts.device_cc}"
    if not in_domain(facts):
        return "this implementation serves exact D512/V512 MXFP8 graphs on SM100/SM103"
    if knobs is not None and knobs.tile_m not in (None, 128):
        return "D512 MXFP8 requires TILE_M=128"
    if knobs is None:
        # Without knobs either body may still be named: a split runs output-slice recompute on every band.
        if implementation == RETAINED_P and not retains_p(facts, 1):
            return "retained-P serves exact top-left causal SWA of one KV tile (left bound 128), unsplit, without dense per-batch KV lengths"
        return None
    if owner(facts, knobs) != implementation:
        if implementation == RETAINED_P:
            return "retained-P serves exact top-left causal SWA of one KV tile (left bound 128), unsplit, without dense per-batch KV lengths"
        return "exact top-left causal SWA of one KV tile (left bound 128), unsplit, runs the retained-P implementation"
    return None
