# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM120 tile rule: shape and SM count in, (tile_m, tile_n) out.

Arithmetic over facts, so no device and no DSL — and no arch gate either. One
rule serves both SM120 cells (f16 and per-tensor FP8); the thresholds below are
the ones their sweeps agreed on, so a change here is a claim about the kernels,
not about the hardware.
"""

from dataclasses import dataclass
from typing import Optional

import pytest

from cudnn.sdpa.fwd.heuristics import _sm120_tiles

pytestmark = pytest.mark.L0

SMS = 188  # RTX PRO 6000 Blackwell, the part the rule was measured on


@dataclass
class _Facts:
    """Only the fields the rule reads."""

    s_q: int
    s_kv: int
    h_q: int
    b: int
    device_sm_count: Optional[int] = SMS
    causal: bool = False
    is_fp8: bool = False
    d_qk: int = 128
    d_v: int = 128


@dataclass
class _Caps:
    tile_ns: frozenset = frozenset({64, 128})


def tiles(s_q, s_kv=None, h_q=16, b=1, **kw):
    return _sm120_tiles(_Caps(), _Facts(s_q, s_kv if s_kv is not None else s_q, h_q, b, **kw))


@pytest.mark.parametrize("fp8", [False, True], ids=["f16", "fp8"])
@pytest.mark.parametrize("s_q,h_q,b", [(512, 16, 1), (4096, 16, 1), (1024, 16, 1), (16384, 16, 1), (2048, 32, 4)])
def test_tile_n_is_128_wherever_it_fits(s_q, h_q, b, fp8):
    """Fastest in every shape swept, on both cells. It was 64 while P was
    staged through SMEM, because that traffic scaled with the KV tile."""
    assert tiles(s_q, h_q=h_q, b=b, is_fp8=fp8)[1] == 128


def test_grid_too_small_to_fill_the_machine_takes_the_finer_q_tile():
    # grid = ceil(512/128) * 16 = 64 CTAs of 188: doubling them still fits.
    assert tiles(512)[0] == 64


def test_a_short_sequence_does_not_amortize_the_finer_q_tile():
    # Same underfilled machine, but 8 KV tiles cannot absorb the extra Q-tile loop.
    assert tiles(1024)[0] == 128


def test_underfilled_and_long_takes_the_finer_q_tile():
    assert tiles(2048)[0] == 64
    assert tiles(2048, h_q=8, b=2)[0] == 64


@pytest.mark.parametrize("s_q", [4096, 8192, 16384])
def test_a_full_machine_takes_the_coarser_q_tile(s_q):
    assert tiles(s_q)[0] == 128


def test_the_grid_bound_sits_between_240_and_320_ctas():
    """Where the two tiles stop trading evenly on a 188-SM part."""
    assert tiles(2560, h_q=12)[0] == 64  # grid 240
    assert tiles(2560, h_q=16)[0] == 128  # grid 320


def test_batch_and_heads_count_toward_the_grid():
    """The rule reads grid, not sequence length: the same s_q flips once the
    batch supplies enough CTAs on its own."""
    assert tiles(2048)[0] == 64
    assert tiles(2048, b=4)[0] == 128


def test_a_causal_mask_halves_the_effective_grid():
    """Causal halves the work per CTA, so the machine empties sooner and the
    finer Q tile keeps paying at a grid that would otherwise be full."""
    assert tiles(4096)[0] == 128
    assert tiles(4096, causal=True)[0] == 64


def test_choice_is_always_in_the_domain():
    for s_q in (128, 512, 1024, 4096, 32768):
        for b in (1, 8):
            tile_m, tile_n = tiles(s_q, b=b)
            assert tile_m in (64, 128) and tile_n in (64, 128)


def test_unknown_sm_count_falls_back_without_dividing_by_zero():
    assert tiles(4096, device_sm_count=0) == (128, 128)
    assert tiles(4096, device_sm_count=None) == (128, 128)


def test_a_wide_head_gives_up_tile_n_128_only_when_it_cannot_fit():
    """SMEM, not speed, is what takes 128 away — and FP8 keeps it further out
    because its KV tile is a byte per element while O still stages in half."""
    assert tiles(4096, d_qk=192, d_v=192)[1] == 128
    assert tiles(4096, d_qk=256, d_v=256)[1] == 64
    assert tiles(4096, d_qk=256, d_v=256, is_fp8=True)[1] == 128
