# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.sdpa.fwd.config_sm100.make_cfg_d192`` -- the SM100 d192x128 forward config's per-tile knobs, host-only.

Pins the fields the half-input single-CTA tile and the pair tile differ in: the K/V L2 prefetch, the TMA-LDG kv-loop
unroll factor, the K sub-stage ring and the V ring depth.  Device-independent: nothing here compiles or launches.
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


def test_half_input_cga1_prefetches_and_unrolls_once():
    cfg = _cfg(dtype_qkv=2, cta_mma=1)
    assert (cfg.L2_PREFETCH_KV, cfg.L2_PREFETCH_NEXT_TILE, cfg.L2_PREFETCH_MIN_STEPS, cfg.LDG_UNROLL) == (1, 1, 4, 1)
    assert (cfg.STAGES_K, cfg.K_SUB, cfg.STAGES_V) == (1, 3, 2)


def test_pair_tile_keeps_the_classic_loader():
    cfg = _cfg(dtype_qkv=2, cta_mma=2)
    assert (cfg.L2_PREFETCH_KV, cfg.L2_PREFETCH_NEXT_TILE, cfg.LDG_UNROLL) == (0, 0, 6)
    assert (cfg.STAGES_K, cfg.K_SUB, cfg.STAGES_V) == (2, 1, 2)


def test_fp8_single_cta_tile_does_not_prefetch():
    cfg = _cfg(dtype_qkv=_E4M3, cta_mma=1)
    assert (cfg.L2_PREFETCH_KV, cfg.L2_PREFETCH_NEXT_TILE, cfg.LDG_UNROLL) == (0, 0, 6)
    assert (cfg.K_SUB, cfg.STAGES_V) == (1, cfg.STAGES_KV)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("L2_PREFETCH_KV", 0, "L2 prefetch"),
        ("L2_PREFETCH_NEXT_TILE", 0, "L2 prefetch"),
        ("L2_PREFETCH_MIN_STEPS", 0, "L2_PREFETCH_MIN_STEPS"),
        ("LDG_UNROLL", 6, "unrolled"),
    ],
)
def test_validator_pins_the_single_cta_loader_knobs(field, value, match):
    cfg = _cfg(dtype_qkv=2, cta_mma=1)
    with pytest.raises(ValueError, match=match):
        c100._validate_cfg_d192(dataclasses.replace(cfg, **{field: value}))
