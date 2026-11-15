# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed work-item geometry for the two native small-half implementations."""

from dataclasses import dataclass, replace
from cudnn import data_type
from cudnn.sdpa import band
from ... import config_sm107 as config
from .d128_f16_multi_q import contract as ordinary
from .small_f16_batch_pair import contract as pair

LEGACY_ROW = "sdpa_fwd_prefill_sm107"
MULTI_Q = ordinary.NAME
BATCH_PAIR = pair.NAME
_CONTRACTS = {MULTI_Q: ordinary, BATCH_PAIR: pair}
IMPLEMENTATIONS = frozenset(_CONTRACTS)


def in_domain(facts):
    mixed = facts.kv_dtype in (data_type.FP8_E4M3, data_type.FP8_E5M2) and facts.uniform_qo_dtype
    return (
        facts.device_cc == (10, 7)
        and not (facts.invalid or facts.is_backward or facts.is_fp8 or facts.is_mxfp8)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
        and (facts.d_qk, facts.d_v) in ((32, 32), (64, 64), (128, 128), (192, 128))
        and (facts.uniform_dtype or (mixed and facts.d_qk in (64, 128)))
    )


def kv_padded(facts):
    """The adapter's scalar tail mask also counts as padding for bias geometry."""
    return facts.padded or (facts.s_kv % 128 != 0 and not band.BandFacts.from_sdpa_facts(facts).covers_kv_tail(facts.s_q, facts.s_kv))


@dataclass(frozen=True)
class Geometry:
    implementation: str
    d_flavor: int
    cga: int
    tile_n: int
    n_chunk: int = 0
    softmax_wgs: int = 2
    lowering_row_name = LEGACY_ROW
    query_tile = 128
    decode_q_tile = 0

    def __post_init__(self):
        if self.implementation not in IMPLEMENTATIONS:
            raise ValueError(f"unknown native small-half implementation: {self.implementation}")
        if (self.n_chunk > 0) != (self.implementation == BATCH_PAIR):
            raise ValueError("n_chunk belongs to the other work-item implementation")

    @property
    def template(self):
        return _CONTRACTS[self.implementation].TEMPLATE_NAME

    @property
    def kernel_file(self):
        return _CONTRACTS[self.implementation].KERNEL_FILE

    @property
    def cta_group(self):
        return self.cga

    @property
    def cluster_m(self):
        return self.cga

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=128,
            tile_n=self.tile_n,
            cga=self.cga,
            sched_policy=0 if knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
            warp_spec_cfg=0,
        )

    def configure_params(self, params):
        return replace(
            params,
            d_flavor=self.d_flavor,
            cta_mma=self.cga,
            tile_n=self.tile_n,
            n_chunk=self.n_chunk,
            softmax_wgs=self.softmax_wgs,
            decode_tile=False,
            decode_quad=False,
            decode_q_tile=0,
            early_q=False,
            dual_kv_stream=False,
        )


def explicit_geometry(implementation, facts, knobs, *, d_flavor=None):
    """Derive internal variants of the named body, without selecting another body."""
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown native small-half implementation: {implementation}")
    is_pair = implementation == BATCH_PAIR
    d = facts.d_qk if d_flavor is None else d_flavor
    domain = (64, 128) if d <= 128 else (128,)
    tile_n = (
        knobs.tile_n
        if knobs.tile_n is not None
        else config.sm107_tile_n_pick(d, facts.s_kv, has_bias=facts.has_bias, has_kv_mask=facts.has_kv_mask, padded=kv_padded(facts), domain=domain)
    )
    cga = knobs.cga
    if cga is None:
        if is_pair:
            cga = 1
        elif facts.has_bias:
            cga = config.sm107_bias_auto_cga(d, facts.s_kv, config.SM107_SMALL_D_CGAS.get((d, d), (2,)), pair=False, tiles_q=2) or 1
        else:
            cga = 1 if d <= 64 else 2
    # The ordinary body always has two query groups. Only the pair body can
    # request the existing three-group variant and its multiple-of-three chunk.
    wgs = (
        config.sm107_softmax_wgs_pick(
            d_flavor=d,
            s_kv=facts.s_kv,
            tile_n=tile_n,
            pair=is_pair,
            has_bias=facts.has_bias,
            has_kv_mask=facts.has_kv_mask,
            padded=kv_padded(facts),
            cga=cga,
            b_flat=facts.b,
        )
        if is_pair
        else 2
    )
    chunk = 0
    if is_pair:
        chunk = config.sm107_n_chunk_rule(facts.b, facts.h_q, facts.s_q, facts.device_sm_count or 0, rows_per_step=wgs)
        chunk = config.sm107_n_chunk_refine(chunk, facts.b, facts.h_q, facts.s_q, facts.device_sm_count or 0, s_kv=facts.s_kv, rows_per_step=wgs)
    return Geometry(implementation, d, cga, tile_n, chunk, wgs)


def mismatch(implementation, facts, knobs):
    if not in_domain(facts):
        return "requires SM107 and supported half small-head graph declarations"
    return _CONTRACTS[implementation].mismatch(facts, knobs)
