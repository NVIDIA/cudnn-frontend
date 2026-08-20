# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the KV-split chooser.

Pure arithmetic over (tile count, KV length, SM count) — no GPU, no kernel
build, so these run anywhere and pin the POLICY rather than any one device.
The B200 numbers below are the shapes the split was built for, taken from the
d128 geometry: a cga2 cluster covers TILES_Q * TILE_M * CTA_MMA = 512 Q rows on
2 CTAs, and TILE_N = 128 sets the KV tile.
"""

import pytest

from cudnn.sdpa.fwd.engines import Capabilities, SdpaFwdKnobs, mismatch
from cudnn.sdpa.fwd.heuristics import _SPLIT_KV_MAX, _SPLIT_KV_MIN_TILES, choose_split_kv

# Pure arithmetic — no device, no kernel build — so every case is L0.
pytestmark = pytest.mark.L0

B200_SMS = 148


def _d128_cga2(s_q, s_kv, heads, batch, sm_count=B200_SMS, **kw):
    """choose_split_kv for the d128 cga2 geometry."""
    return choose_split_kv(
        q_tiles=-(-s_q // 512),
        heads_q=heads,
        batch=batch,
        kv_tiles=-(-s_kv // 128),
        sm_count=sm_count,
        ctas_per_tile=2,
        **kw,
    )


# --- the case the feature exists for --------------------------------------


def _single_wave(base_ctas, split, sm_count=B200_SMS):
    """Does the split leave the launch inside one wave? True for an UNDER-full
    launch, where the idle SMs are free to fill; not a global rule (see
    test_may_exceed_the_sm_count_to_smooth_a_tail)."""
    return base_ctas * split <= sm_count


def test_decode_shape_splits():
    """S_q=128, S_kv=32K, H=16, B=1 — 32 CTAs on a 148-SM part."""
    split = _d128_cga2(128, 32768, 16, 1)
    assert split > 1, "a launch filling 32 of 148 SMs must split"
    assert _single_wave(32, split), "the chosen split must stay inside one wave"
    assert not _single_wave(32, split * 2), "and be the LARGEST power of two that does"


def test_more_heads_need_less_split():
    """Heads are already parallelism: the more there are, the less splitting."""
    few = _d128_cga2(128, 32768, 4, 1)
    many = _d128_cga2(128, 32768, 64, 1)
    assert few >= many


def test_bigger_batch_needs_less_split():
    assert _d128_cga2(128, 32768, 8, 1) >= _d128_cga2(128, 32768, 8, 8)


def test_smaller_chip_needs_less_split():
    """The same shape on a smaller part needs fewer splits to fill it."""
    big = _d128_cga2(128, 32768, 16, 1, sm_count=148)
    small = _d128_cga2(128, 32768, 16, 1, sm_count=48)
    assert small <= big


# --- cases that must NOT split --------------------------------------------


def test_chip_already_full_does_not_split():
    """Long prefill: 4096 Q rows x 16 heads is 8 * 16 * 2 = 256 CTAs > 148."""
    assert _d128_cga2(4096, 2048, 16, 1) == 1


def test_nearly_full_chip_does_not_split():
    """128 CTAs on 148 SMs — 86% — must not split: doubling tips it into a
    second partial wave, so the makespan is flat and only the reduction is
    added. Measured, not assumed."""
    base_ctas = 4 * 16 * 2  # s_q=2048 -> 4 q tiles, 16 heads, cga2
    assert base_ctas == 128
    assert _d128_cga2(2048, 65536, 16, 1) == 1


def test_exactly_full_does_not_split():
    """base_ctas == sm_count is 'filled' — no reduction for zero gain."""
    assert choose_split_kv(q_tiles=1, heads_q=B200_SMS, batch=1, kv_tiles=256, sm_count=B200_SMS) == 1


def test_single_kv_tile_cannot_split():
    assert _d128_cga2(128, 128, 1, 1) == 1


def test_short_kv_does_not_over_split():
    """Splits below _SPLIT_KV_MIN_TILES KV tiles are prologue-dominated."""
    kv_tiles = 4
    split = choose_split_kv(q_tiles=1, heads_q=1, batch=1, kv_tiles=kv_tiles, sm_count=B200_SMS)
    assert split <= kv_tiles // _SPLIT_KV_MIN_TILES


def test_degenerate_inputs_do_not_split():
    for kw in (
        {"q_tiles": 0},
        {"heads_q": 0},
        {"batch": 0},
        {"kv_tiles": 0},
        {"sm_count": 0},
        {"sm_count": -1},
    ):
        args = {"q_tiles": 1, "heads_q": 1, "batch": 1, "kv_tiles": 256, "sm_count": B200_SMS}
        args.update(kw)
        assert choose_split_kv(**args) == 1, f"{kw} must fall back to no split"


# --- invariants over a sweep ----------------------------------------------


@pytest.mark.parametrize("s_kv", [1024, 4096, 16384, 32768, 131072])
@pytest.mark.parametrize("heads", [1, 2, 8, 16, 64])
def test_invariants(s_kv, heads):
    kv_tiles = -(-s_kv // 128)
    split = _d128_cga2(128, s_kv, heads, 1)
    assert 1 <= split <= _SPLIT_KV_MAX
    assert split <= kv_tiles, "more splits than KV tiles would leave empty splits"
    if split > 1:
        assert -(-kv_tiles // split) >= _SPLIT_KV_MIN_TILES


def test_longer_kv_never_needs_fewer_splits():
    """Monotone in KV length: a longer loop is never served by less splitting."""
    prev = 0
    for s_kv in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        split = _d128_cga2(128, s_kv, 16, 1)
        assert split >= prev, f"S_kv={s_kv} chose {split} after {prev}"
        prev = split


@pytest.mark.parametrize("heads", [1, 2, 3, 5, 8, 11, 16, 32, 64])
@pytest.mark.parametrize("s_kv", [4096, 65536, 131072])
def test_choice_is_always_a_power_of_two(heads, s_kv):
    """split_kv is a compile-cache key, so the set is bounded to {1,2,4,8,16}."""
    split = _d128_cga2(512, s_kv, heads, 1)
    assert split & (split - 1) == 0, f"{split} is not a power of two"
    assert split <= _SPLIT_KV_MAX


def test_may_exceed_the_sm_count_to_smooth_a_tail():
    """Over-subscribing the SMs is allowed, and sometimes required: 160 CTAs on
    148 SMs already wastes most of a second wave, and splitting finer shrinks
    that tail rather than adding a wave."""
    split = choose_split_kv(q_tiles=1, heads_q=80, batch=1, kv_tiles=512, sm_count=148, ctas_per_tile=2)
    assert split > 1
    assert 160 * split > 148, "this shape is exactly the case that wants over-subscription"


def test_exactly_balanced_launch_never_splits():
    """base_ctas = k * sm_count has no tail to recover, so every split is pure
    overhead. Provable rather than fitted."""
    for sm_count in (16, 48, 108, 148, 256):
        for k in (1, 2, 3, 4):
            for kv_tiles in (16, 64, 256, 512, 1024):
                split = choose_split_kv(q_tiles=1, heads_q=k * sm_count, batch=1, kv_tiles=kv_tiles, sm_count=sm_count, ctas_per_tile=1)
                assert split == 1, f"base={k * sm_count} == {k}x{sm_count} SMs: nothing to smooth, got {split}"


# (base_ctas, chosen split) pinned against a per-split sweep on B300 (148 SMs,
# d128, 512 KV tiles). A change that moves any of these is a policy change and
# needs its own measurement.
_B300_FIT = [(8, 16), (16, 8), (32, 4), (64, 2), (88, 8), (100, 4), (120, 1), (128, 1), (150, 4), (160, 4), (200, 2), (296, 1)]


@pytest.mark.parametrize("base_ctas,expected", _B300_FIT, ids=[f"{b}ctas" for b, _ in _B300_FIT])
def test_reproduces_the_b300_fit(base_ctas, expected):
    got = choose_split_kv(q_tiles=1, heads_q=base_ctas // 2, batch=1, kv_tiles=512, sm_count=148, ctas_per_tile=2)
    assert got == expected


def test_response_is_not_monotone_in_occupancy():
    """At 88 and 100 CTAs split 2 loses while 4 and 8 win, because 2 lands just
    over a wave boundary and 4 does not. The chooser must search, not
    interpolate."""
    assert choose_split_kv(q_tiles=1, heads_q=44, batch=1, kv_tiles=512, sm_count=148, ctas_per_tile=2) != 2
    assert choose_split_kv(q_tiles=1, heads_q=50, batch=1, kv_tiles=512, sm_count=148, ctas_per_tile=2) != 2


def test_max_split_is_respected():
    assert _d128_cga2(128, 1 << 20, 1, 1, max_split=4) <= 4


# --- the knob plumbing ------------------------------------------------------


@pytest.mark.parametrize("requested", [1, 2, 4, 8, 16])
def test_any_split_kv_request_makes_the_engine_ineligible(requested):
    """The adapters choose the split themselves and nothing forwards
    knobs.split_kv into them, so NO request can be honored -- including
    split_kv=1, since "do not split" is exactly what the lowering would ignore.
    An empty domain is the honest encoding: honored or ineligible, never
    silently dropped."""
    caps = Capabilities(sm_lo=100, sm_hi=100, phase="prefill", d_qk=frozenset({128}), d_v=frozenset({128}))
    assert caps.split_kvs == frozenset()
    why = mismatch(caps, _facts(), SdpaFwdKnobs(split_kv=requested))
    assert why is not None and "split_kv" in why


def test_no_split_request_leaves_the_engine_eligible():
    """A caller with no preference must still reach the adapter's own choice."""
    caps = Capabilities(sm_lo=100, sm_hi=100, phase="prefill", d_qk=frozenset({128}), d_v=frozenset({128}))
    why = mismatch(caps, _facts(), SdpaFwdKnobs(split_kv=None)) or ""
    assert "split_kv" not in why


def _facts():
    from cudnn.sdpa import graph_analyzer as ga

    return ga.SdpaGraphFacts()


def test_no_engine_advertises_a_split_domain():
    """Guards the pairing: a row may only advertise split_kvs once its lowering
    forwards knobs.split_kv. Widening one without the plumbing reintroduces the
    silently-dropped knob."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    specs = ENGINE_SPECS.values() if hasattr(ENGINE_SPECS, "values") else ENGINE_SPECS
    advertising = [sp.name for sp in specs if getattr(getattr(sp, "capabilities", None), "split_kvs", frozenset())]
    assert advertising == [], f"these rows advertise a split domain without the plumbing: {advertising}"
