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
from cudnn.sdpa.fwd.heuristics import _SPLIT_KV_MIN_TILES, choose_split_kv, split_kv_candidates

# Pure arithmetic — no device, no kernel build — so every case is L0.
pytestmark = pytest.mark.L0

B200_SMS = 148


def _d128_cga2(s_q, s_kv, heads, batch, sm_count=B200_SMS, **kw):
    """choose_split_kv for the d128 cga2 geometry.

    ``combine_rows`` is the combine kernel's grid (S_q, H, B) — derived here so
    every case exercises the real two-kernel model rather than a configuration
    production never builds."""
    kw.setdefault("combine_rows", s_q * heads * batch)
    return choose_split_kv(
        q_tiles=-(-s_q // 512),
        heads_q=heads,
        batch=batch,
        kv_tiles=-(-s_kv // 128),
        sm_count=sm_count,
        ctas_per_tile=2,
        **kw,
    )


def _ladder(sm_count=B200_SMS, kv_tiles=1 << 20):
    return split_kv_candidates(sm_count=sm_count, kv_tiles=kv_tiles)


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
    assert choose_split_kv(q_tiles=1, heads_q=B200_SMS, batch=1, kv_tiles=256, sm_count=B200_SMS, combine_rows=128 * B200_SMS) == 1


def test_single_kv_tile_cannot_split():
    assert _d128_cga2(128, 128, 1, 1) == 1


def test_short_kv_does_not_over_split():
    """Splits below _SPLIT_KV_MIN_TILES KV tiles are prologue-dominated."""
    kv_tiles = 4
    split = choose_split_kv(q_tiles=1, heads_q=1, batch=1, kv_tiles=kv_tiles, sm_count=B200_SMS, combine_rows=128)
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
        args = {"q_tiles": 1, "heads_q": 1, "batch": 1, "kv_tiles": 256, "sm_count": B200_SMS, "combine_rows": 128}
        args.update(kw)
        assert choose_split_kv(**args) == 1, f"{kw} must fall back to no split"


# --- invariants over a sweep ----------------------------------------------


@pytest.mark.parametrize("s_kv", [1024, 4096, 16384, 32768, 131072])
@pytest.mark.parametrize("heads", [1, 2, 8, 16, 64])
def test_invariants(s_kv, heads):
    kv_tiles = -(-s_kv // 128)
    split = _d128_cga2(128, s_kv, heads, 1)
    assert 1 <= split <= max(_ladder(kv_tiles=kv_tiles))
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
    """split_kv is a compile-cache key, so the search is a power-of-two ladder."""
    split = _d128_cga2(512, s_kv, heads, 1)
    assert split & (split - 1) == 0, f"{split} is not a power of two"
    assert split in _ladder(kv_tiles=-(-s_kv // 128))


def test_may_exceed_the_sm_count_to_smooth_a_tail():
    """Over-subscribing the SMs is allowed, and sometimes required: 160 CTAs on
    148 SMs already wastes most of a second wave, and splitting finer shrinks
    that tail rather than adding a wave."""
    split = choose_split_kv(q_tiles=1, heads_q=80, batch=1, kv_tiles=512, sm_count=148, ctas_per_tile=2, combine_rows=512 * 80)
    assert split > 1
    assert 160 * split > 148, "this shape is exactly the case that wants over-subscription"


def test_exactly_balanced_launch_never_splits():
    """base_ctas = k * sm_count has no tail to recover, so every split is pure
    overhead. Provable rather than fitted."""
    for sm_count in (16, 48, 108, 148, 256):
        for k in (1, 2, 3, 4):
            for kv_tiles in (16, 64, 256, 512, 1024):
                split = choose_split_kv(
                    q_tiles=1, heads_q=k * sm_count, batch=1, kv_tiles=kv_tiles, sm_count=sm_count, ctas_per_tile=1, combine_rows=128 * k * sm_count
                )
                assert split == 1, f"base={k * sm_count} == {k}x{sm_count} SMs: nothing to smooth, got {split}"


# The Q extent the fit below is pinned at: one full cga2 cluster tile
# (TILES_Q*TILE_M*CTA_MMA = 512 rows), so q_tiles == 1 and the combine grid is
# 512 * heads rows.
_FIT_S_Q = 512

# (base_ctas, chosen split) pinned against a per-split sweep on B300 (148 SMs,
# d128, 512 KV tiles, S_q=512, bf16, mask-free), re-measured for the two-kernel
# cost model. A change that moves any of these is a policy change and needs its
# own measurement.
#
# The model matches the measured optimum on 8 of 12; mean regret 1.009, worst
# 1.035. The misses (88, 100, 150, 160) all sit within ~10% of a FULL machine,
# where the measured curve is nearly flat -- e.g. base=88 spans 2.264..2.512 ms
# across every split -- so the ranking there is worth little and the model
# prefers the cheap answer. Regret, not exact agreement, is the bar.
_B300_FIT = [(8, 16), (16, 8), (32, 4), (64, 2), (88, 1), (100, 1), (120, 1), (128, 1), (150, 2), (160, 2), (200, 2), (296, 1)]


@pytest.mark.parametrize("base_ctas,expected", _B300_FIT, ids=[f"{b}ctas" for b, _ in _B300_FIT])
def test_reproduces_the_b300_fit(base_ctas, expected):
    got = choose_split_kv(q_tiles=1, heads_q=base_ctas // 2, batch=1, kv_tiles=512, sm_count=148, ctas_per_tile=2, combine_rows=_FIT_S_Q * (base_ctas // 2))
    assert got == expected


def test_response_is_not_monotone_in_occupancy():
    """At 88 and 100 CTAs split 2 loses while 4 and 8 win, because 2 lands just
    over a wave boundary and 4 does not. The chooser must search, not
    interpolate."""
    assert choose_split_kv(q_tiles=1, heads_q=44, batch=1, kv_tiles=512, sm_count=148, ctas_per_tile=2, combine_rows=_FIT_S_Q * 44) != 2
    assert choose_split_kv(q_tiles=1, heads_q=50, batch=1, kv_tiles=512, sm_count=148, ctas_per_tile=2, combine_rows=_FIT_S_Q * 50) != 2


def test_candidates_bound_the_choice():
    """The chooser never returns a split outside the list it was given."""
    for cand in ([1], [1, 2], [1, 2, 4], [1, 2, 4, 8, 16]):
        got = _d128_cga2(128, 1 << 20, 1, 1, candidates=cand)
        assert got in cand


# --- the knob plumbing ------------------------------------------------------


@pytest.mark.parametrize("requested", [2, 4, 8, 16])
def test_split_request_outside_the_domain_makes_the_engine_ineligible(requested):
    """The default row serves only split_kv=1 ("off"): a split request on a row
    whose lowering has no split path is honored-or-ineligible, never silently
    dropped."""
    caps = Capabilities(sm_lo=100, sm_hi=100, phase="prefill", d_shapes=frozenset({(128, 128)}))
    assert caps.split_kv_supported is False
    why = mismatch(caps, _facts(), SdpaFwdKnobs(split_kv=requested))
    assert why is not None and "split_kv" in why


def test_no_split_and_explicit_one_leave_the_engine_eligible():
    """No preference passes; so does an EXPLICIT split_kv=1 — "do not split"
    is a real point on the axis, not an ignored request."""
    caps = Capabilities(sm_lo=100, sm_hi=100, phase="prefill", d_shapes=frozenset({(128, 128)}))
    for knobs in (SdpaFwdKnobs(split_kv=None), SdpaFwdKnobs(split_kv=1)):
        why = mismatch(caps, _facts(), knobs) or ""
        assert "split_kv" not in why


def _facts():
    from cudnn.sdpa import graph_analyzer as ga

    return ga.SdpaGraphFacts()


def test_split_declines_when_the_kv_tail_needs_synthesized_padding():
    """A ragged S_kv on a skv_tail_via_padding row is served through the
    padded kernel path (synthesized per-batch KV lengths) — the one path the
    split cannot ride. The gate must mirror lower_dsl_prefill's predicate so
    the plan is never listed, not declined at build."""
    from cudnn.sdpa import graph_analyzer as ga

    caps = Capabilities(
        sm_lo=100,
        sm_hi=100,
        phase="prefill",
        d_shapes=frozenset({(128, 128)}),
        skv_tail_via_padding=True,
        split_kv_supported=True,
    )
    ragged = ga.SdpaGraphFacts(s_q=128, s_kv=1000)  # 1000 % 128 != 0, mask-free
    why = mismatch(caps, ragged, SdpaFwdKnobs(split_kv=2))
    assert why is not None and "split_kv" in why
    # A causal band that provably masks the tail needs no synthesized padding,
    # so the same request passes the split gate (later gates may still apply).
    covered = ga.SdpaGraphFacts(s_q=128, s_kv=1000, causal=True)
    why = mismatch(caps, covered, SdpaFwdKnobs(split_kv=2)) or ""
    assert "split_kv" not in why


def test_split_domains_match_the_wired_lowerings():
    """Guards the pairing: a row sets split_kv_supported exactly when its
    adapter forwards the knob into TemplateParams and launches the combine.
    Setting one without the plumbing reintroduces the silently-dropped knob."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    advertising = {sp.name for sp in ENGINE_SPECS if sp.capabilities.split_kv_supported}
    assert advertising == {
        "sdpa_fwd_prefill_sm100",
        "sdpa_fwd_prefill_sm100_mxfp8",
        "sdpa_fwd_prefill_sm100_fp8",
        "sdpa_fwd_prefill_sm107_fp8",
        "sdpa_fwd_prefill_sm120",
        "sdpa_fwd_prefill_sm120_fp8",
    }, f"split domains drifted from the wired lowerings: {sorted(advertising)}"


def test_split_points_feeds_the_exact_cluster_extent():
    """_split_points must measure the launch in CLUSTERS, not CTA tiles.

    The SM100 d128 cluster covers TILES_Q * TILE_M * CTA_MMA = 512 Q rows on
    its 2 CTAs — the same extent every helper above assumes. Feeding the model
    ``tile_m * cga`` (256) instead doubles the apparent tile count, so a
    half-empty machine reads as full and the chooser under-splits: the ar_dit
    chunked-prefill shape below measured 4 on the true geometry and 2 on the
    approximation, worth 1.72x vs 1.33x on B300.
    """
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS
    from cudnn.sdpa.fwd.heuristics import _split_points

    caps = next(sp for sp in ENGINE_SPECS if sp.name == "sdpa_fwd_prefill_sm100").capabilities
    # ar_dit: B1 x H9 x D128 bf16, 985 new tokens against a 62208-token clip.
    facts = ga.SdpaGraphFacts(
        b=1,
        h_q=9,
        h_kv=9,
        s_q=985,
        s_kv=62208,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_sm_count=B200_SMS,
        device_cc=(10, 0),
    )
    points = _split_points(caps, facts, 128, 128, 2)
    assert points[0] == 4, f"expected the 512-row cluster extent to choose 4, got {points}"
    assert points[-1] == 1, "no-split must remain reachable behind the chosen split"


# --- the candidate ladder ---------------------------------------------------


@pytest.mark.parametrize("sm_count,top", [(148, 256), (132, 256), (108, 128), (84, 128), (16, 16), (1, 1)])
def test_ladder_is_derived_from_the_device(sm_count, top):
    """THE single split list, derived per device rather than declared per row:
    powers of two up to 2**ceil(log2(sm_count)) — you never need more CTA-tiles
    than the machine has SMs."""
    got = split_kv_candidates(sm_count=sm_count, kv_tiles=1 << 20)
    assert got[0] == 1 and got[-1] == top
    assert got == [1 << i for i in range(len(got))]


def test_ladder_is_bounded_by_the_thinnest_split():
    """kv_tiles // MIN_TILES is the largest split whose thinnest chunk still
    clears the floor — the loop guard restated as a bound."""
    for kv_tiles in (4, 17, 64, 486, 512):
        got = split_kv_candidates(sm_count=B200_SMS, kv_tiles=kv_tiles)
        assert max(got) <= max(1, kv_tiles // _SPLIT_KV_MIN_TILES)
        for s in got:
            if s > 1:
                assert kv_tiles // s >= _SPLIT_KV_MIN_TILES


def test_degenerate_device_yields_the_no_split_ladder():
    assert split_kv_candidates(sm_count=0, kv_tiles=512) == [1]
    assert split_kv_candidates(sm_count=148, kv_tiles=0) == [1]


# --- the combine term -------------------------------------------------------


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 16])
def test_longer_q_never_splits_more(heads):
    """The combine's grid is (S_q, H, B), so more Q rows means more combine
    waves to pay per split. At a FIXED base_ctas (every S_q here is one cga2
    cluster tile, so q_tiles == 1) a longer Q extent must never ask for a
    LARGER split. Only the combine term can express this — the wave term does
    not see S_q at all."""
    got = [
        choose_split_kv(q_tiles=1, heads_q=heads, batch=1, kv_tiles=512, sm_count=B200_SMS, ctas_per_tile=2, combine_rows=s_q * heads)
        for s_q in (64, 128, 256, 512)
    ]
    assert all(a >= b for a, b in zip(got, got[1:])), f"S_q 64/128/256/512 -> {got}"


def test_decode_rows_barely_pay_for_the_combine():
    """A decode-shaped launch reduces one block per (row, head, batch) — far
    fewer rows than SMs, so one combine wave. The reduction must not price it
    out of the splitting it exists for."""
    decode = choose_split_kv(q_tiles=1, heads_q=8, batch=1, kv_tiles=512, sm_count=B200_SMS, ctas_per_tile=2, combine_rows=8)
    prefill = choose_split_kv(q_tiles=1, heads_q=8, batch=1, kv_tiles=512, sm_count=B200_SMS, ctas_per_tile=2, combine_rows=8 * 4096)
    assert decode > 1
    assert decode >= prefill
