# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.sdpa.fwd.config_sm100.make_cfg_d192`` -- the SM100 d192x128 forward config's per-tile knobs, host-only.

Pins the fields the half-input single-CTA tile's L2-prefetching loader (``TemplateParams.cga1_l2_prefetch``) sets apart
from the develop loader and the pair tile -- the K/V L2 prefetch, the TMA-LDG kv-loop unroll factor, the dense-causal
exp2 tail emulation -- the K sub-stage ring and V ring depths, the validator pins, and the adapter's per-arch claim of the
flag.  Device-independent: nothing here compiles or launches.
"""

from __future__ import annotations

import dataclasses

import pytest

from cudnn.sdpa.fwd import config_sm100 as c100
from cudnn.sdpa.fwd.config_sm100 import TemplateParams

pytestmark = pytest.mark.L0

_E4M3 = c100.DTYPE_E4M3


def _cfg(**kw):
    cfg, _ = c100.make_cfg_d192(TemplateParams(**kw))
    return cfg


def _loader(cfg):
    return (cfg.L2_PREFETCH_KV, cfg.L2_PREFETCH_NEXT_TILE, cfg.L2_PREFETCH_MIN_STEPS, cfg.LDG_UNROLL, cfg.EXP2_TAIL_MIX)


def test_half_input_cga1_with_the_claim_prefetches_and_unrolls_once():
    cfg = _cfg(dtype_qkv=2, cta_mma=1, cga1_l2_prefetch=True)
    assert _loader(cfg) == (1, 1, 4, 1, 0)
    assert (cfg.STAGES_K, cfg.K_SUB, cfg.STAGES_V) == (1, 3, 2)


def test_half_input_cga1_without_the_claim_keeps_the_develop_loader():
    cfg = _cfg(dtype_qkv=2, cta_mma=1)
    assert _loader(cfg) == (0, 0, 4, 6, 1)
    assert (cfg.STAGES_K, cfg.K_SUB, cfg.STAGES_V) == (1, 3, 2)


@pytest.mark.parametrize("claim", [False, True])
def test_pair_tile_ignores_the_claim(claim):
    cfg = _cfg(dtype_qkv=2, cta_mma=2, cga1_l2_prefetch=claim)
    assert _loader(cfg) == (0, 0, 4, 6, 1)
    assert (cfg.STAGES_K, cfg.K_SUB, cfg.STAGES_V) == (2, 1, 2)


def test_fp8_single_cta_tile_ignores_the_claim():
    cfg = _cfg(dtype_qkv=_E4M3, cta_mma=1, cga1_l2_prefetch=True)
    assert _loader(cfg) == (0, 0, 4, 6, 1)
    assert (cfg.K_SUB, cfg.STAGES_V) == (1, cfg.STAGES_KV)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("L2_PREFETCH_KV", 2, "L2-prefetching loader"),
        ("L2_PREFETCH_NEXT_TILE", 0, "next-tile L2 prefetch"),
        ("L2_PREFETCH_MIN_STEPS", 0, "L2_PREFETCH_MIN_STEPS"),
        ("LDG_UNROLL", 6, "unrolled"),
        ("EXP2_TAIL_MIX", 1, "tail emulation"),
    ],
)
def test_validator_pins_the_prefetching_loader_knobs(field, value, match):
    cfg = _cfg(dtype_qkv=2, cta_mma=1, cga1_l2_prefetch=True)
    with pytest.raises(ValueError, match=match):
        c100._validate_cfg_d192(dataclasses.replace(cfg, **{field: value}))


def test_validator_keeps_the_prefetching_loader_off_the_pair_tile():
    cfg = _cfg(dtype_qkv=2, cta_mma=2)
    with pytest.raises(ValueError, match="half-input cga1 tile"):
        c100._validate_cfg_d192(dataclasses.replace(cfg, L2_PREFETCH_KV=1, L2_PREFETCH_NEXT_TILE=1, LDG_UNROLL=1, EXP2_TAIL_MIX=0))


def test_adapter_claims_the_prefetching_loader_per_arch():
    from cudnn.sdpa.fwd.api_dsl import _cga1_l2_prefetch_for

    assert _cga1_l2_prefetch_for((10, 3), kind="f16", flavor=(192, 128)) is True
    assert _cga1_l2_prefetch_for((10, 0), kind="f16", flavor=(192, 128)) is False
    assert _cga1_l2_prefetch_for((10, 7), kind="f16", flavor=(192, 128)) is False
    assert _cga1_l2_prefetch_for((10, 3), kind="fp8", flavor=(192, 128)) is False
    assert _cga1_l2_prefetch_for((10, 3), kind="f16", flavor=(128, 128)) is False
