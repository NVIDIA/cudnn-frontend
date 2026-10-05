# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pins for ``sdpa.mxfp8.block_scaled_o_draw``: the fold that decides whether a
test_mhas_v2 MXFP8 forward draw runs the block-scaled O epilogue (``sf_o``) or a
plain MXFP8 forward.

The ``sf_o`` output has no backend lowering, so a draw the harness admits and the
FROST MXFP8 engine then declines FAILS the test (not a skip). The predicate must
therefore mirror the engine rows exactly -- including their ARCH RANGE: the SM120
CI lane (cc 12.0, so ``major >= 10``) failed every admitted draw on #1180 because
no SM120 MXFP8 engine exists. Pure python; runs on every lane."""

import cudnn
import pytest

from sdpa.mxfp8 import block_scaled_o_draw

pytestmark = pytest.mark.L0

TL, BR = cudnn.diagonal_alignment.TOP_LEFT, cudnn.diagonal_alignment.BOTTOM_RIGHT


def _draw(o_block_scale=32, **over):
    kw = dict(
        sm=100,
        is_infer=True,
        is_paged=False,
        with_unfuse_fma=False,
        d_qk=128,
        d_vo=128,
        s_qo=256,
        s_kv=256,
        right_bound=None,
        diag_align=TL,
        engines_enabled=True,
        has_fp4=True,
    )
    kw.update(over)
    return block_scaled_o_draw(o_block_scale, **kw)


@pytest.mark.parametrize("sm", [100, 103, 107, 110])
def test_admitted_on_the_sm100_line(sm):
    assert _draw(32, sm=sm) == 32
    assert _draw(16, sm=sm) == 16


@pytest.mark.parametrize("sm", [80, 90, 120, 121])
def test_folds_off_the_mxfp8_engine_rows(sm):
    """SM120 has no MXFP8 kernel; Hopper and Ampere have no FROST MXFP8 row at all."""
    assert _draw(32, sm=sm) == 0
    assert _draw(16, sm=sm) == 0


def test_folds_when_the_engines_are_off_or_the_graph_is_not_a_dense_inference_forward():
    assert _draw(engines_enabled=False) == 0
    assert _draw(is_infer=False) == 0
    assert _draw(is_paged=True) == 0
    assert _draw(with_unfuse_fma=True) == 0  # backend-only attribute; the FROST engines decline it
    assert _draw(d_qk=64, d_vo=64) == 0
    assert _draw(d_qk=192, d_vo=128) == 0
    assert _draw(0) == 0


def test_kv_tail_rule_mirrors_the_engine():
    """A KV tail that is not a whole 128-tile is served only under a causal band that
    covers it: top-left s_q + right_bound <= s_kv; bottom-right plain causal only."""
    assert _draw(s_qo=300, s_kv=300, right_bound=None) == 0
    assert _draw(s_qo=300, s_kv=300, right_bound=0, diag_align=TL) == 32
    assert _draw(s_qo=300, s_kv=300, right_bound=64, diag_align=TL) == 0
    assert _draw(s_qo=200, s_kv=300, right_bound=64, diag_align=TL) == 32
    assert _draw(s_qo=300, s_kv=300, right_bound=0, diag_align=BR) == 32
    assert _draw(s_qo=300, s_kv=300, right_bound=1, diag_align=BR) == 0
    assert _draw(s_qo=157, s_kv=545, right_bound=29, diag_align=TL) == 32  # a sweep draw that ran on FROST
    assert _draw(s_qo=1919, s_kv=1919, right_bound=0, diag_align=TL, with_unfuse_fma=True) == 0  # a sweep draw that folded


def test_fp4_needs_the_packed_dtype():
    assert _draw(16, has_fp4=False) == 0
    assert _draw(32, has_fp4=False) == 32
