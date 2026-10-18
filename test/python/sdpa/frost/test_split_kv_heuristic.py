# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the KV-split chooser.

Pure arithmetic over (tile count, KV length, SM count) — no GPU, no kernel
build, so these run anywhere and pin the POLICY rather than any one device.
The B200 numbers below are the shapes the split was built for, taken from the
d128 geometry: a cga2 cluster covers TILES_Q * TILE_M * CTA_MMA = 512 Q rows on
2 CTAs, and TILE_N = 128 sets the KV tile.
"""

import sdpa_binding_reference as binding_reference

import pytest

from cudnn.sdpa.fwd.engines import Capabilities, SdpaFwdKnobs, mismatch
from frost_test_utils import requires_dsl
from cudnn.sdpa.fwd.heuristics import _SERIAL_COMBINE_MIN_SQ, _SPLIT_KV_MIN_TILES, choose_decode_tile_split_kv, choose_split_kv, split_kv_candidates

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
# d128, 512 KV tiles, S_q=512, bf16, mask-free). A change that moves any of
# these is a policy change and needs its own measurement.
#
# Re-swept end-to-end (split kernel + combine) once SM100 splits started writing
# fp32 partials, which changed BOTH kernels: the epilogue stopped staging O
# through SMEM and TMA, and the combine reads wider partials. The measured
# optimum moved at 88 (1 -> 4), 150 (2 -> 4) and 160 (2 -> 4), so these entries
# and _SPLIT_KV_COMBINE_COST were refitted together.
#
# The model now matches the measured optimum on 11 of 12; mean regret 1.0001,
# worst 1.0007. The one miss is base=100, where split 1 and split 4 measure
# 0.6701 vs 0.6706 ms -- a 0.07% tie the ranking cannot meaningfully resolve.
# Regret, not exact agreement, is the bar.
_B300_FIT = [(8, 16), (16, 8), (32, 4), (64, 2), (88, 4), (100, 4), (120, 1), (128, 1), (150, 4), (160, 4), (200, 2), (296, 1)]


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
    kernel's KV-tail mask (kv_tail_mask, #1425). Outside the qualified
    exact-D128 architectures this remains declined before plan build."""
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


@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("fp8", [False, True])
def test_explicit_scalar_tail_split_admission(device_cc, fp8):
    """Tail admission follows the native lowerings, without pinning a heuristic winner."""
    from dataclasses import replace
    from cudnn.sdpa import graph_analyzer as ga

    caps = Capabilities(
        sm_lo=100,
        sm_hi=107,
        phase="prefill",
        d_shapes=frozenset({(128, 128)}),
        skv_tail_via_padding=True,
        split_kv_supported=True,
    )
    facts = ga.SdpaGraphFacts(s_q=64, s_kv=257, d_qk=128, d_v=128, device_cc=device_cc, is_fp8=fp8)
    knobs = SdpaFwdKnobs(split_kv=4)
    assert "split_kv" not in (mismatch(caps, facts, knobs) or "")
    for other in (
        replace(facts, device_cc=(10, 8)),
        replace(facts, d_qk=112),
        replace(facts, d_qk=192),
        replace(facts, is_fp8=False, is_mxfp8=True),
    ):
        assert "split_kv" in (mismatch(caps, other, knobs) or "")


def test_split_kv_tail_rule_exempts_paged_but_not_dense_padded():
    """The synthesized-padding exclusion is about the DENSE mask-free path. A
    paged graph is padded by construction (per-batch KV lengths are mandatory
    and bound the walk on device), so a declared ``paged_attention_max_seq_len_kv``
    that is not a 128-multiple — FlashInfer passes its true max, e.g. 4000 —
    must not block the split. The neighbouring dense cases keep their verdicts:
    a dense padded graph still declines the split (its padded path yields no
    per-split partials), and scalar-tail splitting still needs a qualified
    native dimension and architecture."""
    from cudnn.sdpa import graph_analyzer as ga

    caps = Capabilities(
        sm_lo=100,
        sm_hi=100,
        phase="prefill",
        d_shapes=frozenset({(128, 128)}),
        skv_tail_via_padding=True,
        split_kv_supported=True,
        paged_kv=True,
        padded=True,
    )
    paged = ga.SdpaGraphFacts(s_q=1, s_kv=4000, padded=True, has_paged_kv=True, page_size=16)
    why = mismatch(caps, paged, SdpaFwdKnobs(split_kv=2)) or ""
    assert "split_kv" not in why, why
    # Unchanged: a dense padded graph never splits, whatever its S_kv.
    for s_kv in (4000, 4096):
        dense_padded = ga.SdpaGraphFacts(s_q=1, s_kv=s_kv, padded=True)
        why = mismatch(caps, dense_padded, SdpaFwdKnobs(split_kv=2))
        assert why is not None and "split_kv" in why, (s_kv, why)
    # Unchanged: a dense mask-free ragged S_kv rides the synthesized padding.
    dense_ragged = ga.SdpaGraphFacts(s_q=1, s_kv=4000)
    why = mismatch(caps, dense_ragged, SdpaFwdKnobs(split_kv=2))
    assert why is not None and "split_kv" in why, why


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
        "sdpa_fwd_prefill_sm107",
        "sdpa_fwd_prefill_sm107_fp8",
        "sdpa_fwd_prefill_sm120",
        "sdpa_fwd_prefill_sm120_fp8",
        "sdpa_fwd_prefill_sm120_mxfp8",
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


def test_unsplit_leg_pays_no_combine():
    """The unsplit leg runs ONE kernel -- no combine is launched -- so the model
    charges it none. Charging it one combine wave-set (``s = 1`` in the combine
    term) under-priced the reduction a split adds by exactly that much, which
    flipped small-KV launches with many output rows to a split that measures
    slower (B200, d128 bf16, S_kv=4096 = 32 KV tiles, kernel time):

        b=8 h=32/8 S_q=64 paged, cga1, 16384 rows:  unsplit 67.6 us, split 2 75.0 us
        b=4 h=32/8 S_q=128 dense causal, cga2:      unsplit 120.2 us, split 2 155.5 us

    while a cheap combine still lets the model split where that pays:

        b=8 h=32/8 S_q=16 paged, cga1, 4096 rows:   split 2 49.4 us, unsplit 65.3 us
        b=8 h=64/4 S_q=1 paged, cga1, 512 rows:     split 4 27.5 us, unsplit 63.6 us
    """
    common = dict(q_tiles=1, kv_tiles=32, sm_count=B200_SMS)
    assert choose_split_kv(heads_q=8, batch=8, ctas_per_tile=1, combine_rows=64 * 32 * 8, **common) == 1
    assert choose_split_kv(heads_q=8, batch=4, ctas_per_tile=2, combine_rows=128 * 32 * 4, **common) == 1
    assert choose_split_kv(heads_q=8, batch=8, ctas_per_tile=1, combine_rows=16 * 32 * 8, **common) == 2
    assert choose_split_kv(heads_q=4, batch=8, ctas_per_tile=1, combine_rows=1 * 64 * 8, **common) == 4


def test_unsplit_leg_accounting_reaches_the_wide_head_flavors():
    """The unsplit leg carrying no combine term is not a d128 rule: every
    flavor's split leg runs the same arithmetic. On the d192x128 and d256 f16
    flavors (cga2 clusters, 128-row KV tiles) the launches it moves are
    few-unit 2k-KV chunks with thousands of output rows, where the phantom
    combine wave-set used to outweigh halving a 16-tile loop. Measured on B200
    (bf16 dense, kernel time, heuristic lead against the split 2 the old
    accounting led with, pinned by knobs):

        d256 b=1 h=64/4 S_q=128 S_kv=2048 mask-free:  unsplit 51.7 us, split 2 75.3 us
        d192 b=1 h=32/8 S_q=256 S_kv=2048 causal:     unsplit 35.3 us, split 2 60.9 us
    """
    wide = dict(batch=1, kv_tiles=16, sm_count=B200_SMS, ctas_per_tile=2, combine_rows=8192)
    # d256: a 256-row cga2 cluster, packed G=16 -> 2048 rows per unit -> 8 Q tiles x 4 KV heads.
    assert choose_split_kv(q_tiles=8, heads_q=4, **wide) == 1
    # d192: a 512-row cga2 cluster, packed G=4 -> 1024 rows per unit -> 2 Q tiles x 8 KV heads.
    assert choose_split_kv(q_tiles=2, heads_q=8, **wide) == 1


def test_lone_combine_blocks_pay_the_latency_floor():
    """A combine block walks its ``s`` partials serially; with fewer output rows
    than the machine holds at once nothing hides that chain, so a partial costs
    the lone-block price (_SPLIT_KV_COMBINE_FLOOR) however few combine waves
    the rows make. Fed the per-wave price alone, a few-unit launch read the
    combine as nearly free and split until the wave was full -- one power of
    two past the measured optimum (B200, d128 bf16 paged, kernel time, the
    grid inside one wave throughout):

        b=1 h=16/2 S_q=1 S_kv=32k, cga1 (2 units, 16 rows):  split 64 50.8 us, split 32 39.7 us
        b=1 h=64/4 S_q=1 S_kv=4k,  cga1 (4 units, 64 rows):  split 16 22.3 us, split 8  20.3 us
        b=1 h=32/8 S_q=1 S_kv=4k,  cga1 (8 units, 32 rows):  split 16 23.6 us, split 8  20.3 us

    while a ladder the wave boundary already stops keeps its choice:

        b=1 h=64/4 S_q=1 S_kv=32k, cga1:  split 32 43.0 us (split 16 47.2 us)
        b=4 h=64/4 S_q=1 S_kv=4k,  cga1:  split 8  21.5 us (split 4  25.4 us)
        b=8 h=64/4 S_q=1 S_kv=4k,  cga1 (512 rows: four combine waves, above the floor):  split 4 26.6 us
    """
    lone = dict(q_tiles=1, batch=1, sm_count=B200_SMS, ctas_per_tile=1)
    assert choose_split_kv(heads_q=2, kv_tiles=256, combine_rows=16, **lone) == 32
    assert choose_split_kv(heads_q=4, kv_tiles=32, combine_rows=64, **lone) == 8
    assert choose_split_kv(heads_q=8, kv_tiles=32, combine_rows=32, **lone) == 8
    assert choose_split_kv(heads_q=4, kv_tiles=256, combine_rows=64, **lone) == 32
    assert choose_split_kv(q_tiles=1, heads_q=4, batch=4, kv_tiles=32, combine_rows=256, sm_count=B200_SMS, ctas_per_tile=1) == 8
    assert choose_split_kv(q_tiles=1, heads_q=4, batch=8, kv_tiles=32, combine_rows=512, sm_count=B200_SMS, ctas_per_tile=1) == 4
    # The floor is a lone block's price and nothing more: it must sit below the
    # per-wave price of the smallest launch _B300_FIT pins (2048 rows), or a
    # future retune of either constant would move the fitted choices. (Imported
    # here so the behaviour asserts above collect on a tree without the floor.)
    from cudnn.sdpa.fwd.heuristics import _SPLIT_KV_COMBINE_COST, _SPLIT_KV_COMBINE_FLOOR

    assert _SPLIT_KV_COMBINE_FLOOR <= -(-(_FIT_S_Q * 4) // B200_SMS) * _SPLIT_KV_COMBINE_COST


# d128 cga1, 8/2 heads packed 4:1 (one 128-row Q tile for 16 queries, 2 KV-head
# CTAs), the 16-query verify step of a speculative decode, 148 SMs.  Measured
# with pinned splits on B200 after the serial combine's 16-deep walk, main +
# combine in us:
#   2048 keys:  split 4 12.5, split 8 11.8            (combine 3.4 / 4.1)
#   8192 keys:  split 8 17.2, split 32 14.1           (combine 4.1 / 5.4)
# Before the deep walk (0.27 us a split) the serial combine made split 32 at
# 8192 keys 28 % slower than split 8 (19.0 against 14.8 us on B300).
_SERIAL_Q16 = dict(q_tiles=1, heads_q=2, batch=1, sm_count=148, ctas_per_tile=1, combine_rows=16 * 8)


def test_serial_combine_bounds_the_split_depth():
    """S_q = 16 runs the serial combine, whose per-split latency the per-row
    term cannot see.  With the term the long launch stays inside the measured
    16-32 band and the short one keeps its measured split 8; the term is small
    enough (the walk is 16 deep) not to push either below its optimum."""
    assert choose_split_kv(kv_tiles=64, serial_combine=True, **_SERIAL_Q16) in (16, 32)
    assert choose_split_kv(kv_tiles=16, serial_combine=True, **_SERIAL_Q16) == 8
    # The per-row model alone never prices the walk: the term can only lower the pick.
    assert choose_split_kv(kv_tiles=64, serial_combine=True, **_SERIAL_Q16) <= choose_split_kv(kv_tiles=64, **_SERIAL_Q16)


def test_serial_combine_term_grows_with_kv_but_never_runs_away():
    """Longer KV still earns more splits -- the main kernel's saving grows --
    but the serial walk caps the depth below the 128-split pick the per-row
    model made at 128k keys for a 3-query, 1-KV-head launch, where 64 and 128
    splits measured alike (d64, B200: 27.4 us) and 32 lost (37.3 us)."""
    tp8_q3 = dict(q_tiles=1, heads_q=1, batch=1, sm_count=148, ctas_per_tile=1, combine_rows=3 * 4)
    got = [choose_split_kv(kv_tiles=-(-kv // 128), serial_combine=True, **tp8_q3) for kv in (8192, 32768, 131072)]
    assert all(a <= b for a, b in zip(got, got[1:])), got
    assert 32 <= got[-1] <= 64, got
    # The per-row model's lone-block floor already curbs this launch; the serial
    # term can only lower the pick further, never raise it.
    assert choose_split_kv(kv_tiles=1024, **tp8_q3) >= got[-1]


def test_serial_combine_term_leaves_decode_and_saturated_launches_alone():
    """The decode form (S_q <= 2) is priced by the per-row term alone, so the
    flag must not move a one-row launch; and a launch already past the SM
    count (b=128 x 8 KV-head CTAs) never split anyway, so the serial term
    cannot change what it does."""
    one_row = dict(q_tiles=1, heads_q=2, batch=1, kv_tiles=64, sm_count=148, ctas_per_tile=1, combine_rows=8)
    assert choose_split_kv(**one_row) == choose_split_kv(serial_combine=False, **one_row)
    full = dict(q_tiles=1, heads_q=8, batch=128, kv_tiles=64, sm_count=148, ctas_per_tile=1, combine_rows=16 * 32 * 128)
    assert choose_split_kv(**full) == choose_split_kv(serial_combine=True, **full) == 1
    # No combine runs at split 1, so the serial term never makes unsplit dearer.
    assert choose_split_kv(kv_tiles=1, serial_combine=True, **_SERIAL_Q16) == 1


def test_serial_combine_quantized_tile_pays_less_per_split():
    """The per-tensor fp8 d128 tile is MMA-bound on a decode shape, ~3x the
    bf16 decode tile's time per KV tile, so one split of serial combine is a
    third as many tile units and the quantized pick is never below the bf16
    one.  Pinned on tp8 b=1 q=3 at 128k keys (1 KV head, 1024 tiles): fp8
    main 17 + combine 20 us at 64 splits against 46 us at 32 (B300)."""
    from cudnn.sdpa.fwd.heuristics import _SPLIT_KV_SERIAL_COMBINE_COST, _SPLIT_KV_SERIAL_COMBINE_COST_QUANTIZED

    assert 0 < _SPLIT_KV_SERIAL_COMBINE_COST_QUANTIZED < _SPLIT_KV_SERIAL_COMBINE_COST
    tp8_q3 = dict(q_tiles=1, heads_q=1, batch=1, kv_tiles=1024, sm_count=148, ctas_per_tile=1, combine_rows=3 * 8)
    bf16 = choose_split_kv(serial_combine=True, **tp8_q3)
    fp8 = choose_split_kv(serial_combine=True, serial_combine_cost=_SPLIT_KV_SERIAL_COMBINE_COST_QUANTIZED, **tp8_q3)
    assert 32 <= bf16 <= fp8 == 64, (bf16, fp8)
    assert fp8 <= choose_split_kv(**tp8_q3)


@requires_dsl
def test_serial_combine_threshold_matches_the_kernel_routing():
    """The model's S_q threshold must be the one sm100/split_combine routes
    on: ``SQ <= ROWS_PER_BLOCK // 2`` takes the decode form, everything above
    the serial one."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine

    assert _SERIAL_COMBINE_MIN_SQ == split_combine.ROWS_PER_BLOCK // 2 + 1


# --- the d256 decode tile's own model ----------------------------------------
#
# sm100/decode_d256_f16.py runs one cta_group::1 CTA per (KV-head group, batch,
# split) unit and is HBM-bound, so its split trades the device's idle SMs
# against the split path's second launch.  The model has TWO LINES, two fits of
# two sibling kernels: the Blackwell line (``device_cc=(10, 0)``: the B200 / B300
# fit, 148 SMs, the second launch at its device-side latency, the combine's
# decode / serial forms priced apart) and the cc 10.7 line (``device_cc=None``
# or (10, 7): the Rubin performance class's fit -- per-width stream rates, the
# wave term, the combine priced by the form its grid takes).  ``_decode_tile``
# reads the cc 10.7 line (the model's default), ``_blackwell_tile`` the Blackwell
# line; a cell measured on one line is pinned on that line, a statement that
# holds on both is pinned on both.  The shapes are Qwen3.5 32/2 (2 KV heads,
# packed 16:1) unless a cell says otherwise.
_LINES = {"blackwell": (10, 0), "rubin": None}


def _decode_tile(units, s_kv, sm_count=B200_SMS, **kw):
    return choose_decode_tile_split_kv(units=units, kv_tiles=-(-s_kv // 128), sm_count=sm_count, **kw)


def _blackwell_tile(units, s_kv, sm_count=B200_SMS, **kw):
    kw.setdefault("device_cc", (10, 0))
    return _decode_tile(units, s_kv, sm_count=sm_count, **kw)


@pytest.mark.parametrize("line", sorted(_LINES))
def test_decode_tile_serving_shape_stays_unsplit(line):
    """b=32 x 2 KV heads (64 units) over 4096 keys: split 2 measured 8 % SLOWER
    than unsplit on B200 (58.0 -> 62.9 us: the second kernel's own launch and
    the combine eat the saving), so the default does not split it; the
    captured caller's optimum -- the launch term at 0 -- is the split 2 runner-up.
    The cc 10.7 line reads the same on its own charge (the serving shape's 6 us
    of GPU saving does not cover its second launch)."""
    cc = _LINES[line]
    assert _decode_tile(64, 4096, device_cc=cc) == 1
    assert _decode_tile(64, 4096, device_cc=cc, launch_cost=0.0) == 2


def test_decode_tile_short_kv_at_batch_one_splits():
    """b=1 x 2 KV heads (2 units) over 2048 keys (16 tiles): unsplit, two CTAs
    stream 16 tiles each (25 us on B300); split 8 puts 16 streams on the device
    for two tiles each plus one combine.  A Blackwell-line cell (its second
    launch is priced at the device-side latency the native binder measured)."""
    assert _blackwell_tile(2, 2048) == 8
    assert _blackwell_tile(2, 8192) >= 16


@pytest.mark.parametrize("line", sorted(_LINES))
def test_decode_tile_small_batch_splits_to_fill_the_machine(line):
    """16 units (b=8) over 4096 keys: unsplit, 16 of 148 SMs stream 32 tiles
    each; split 8 puts 128 streams on the device and the GPU saving (~56 ->
    ~20 us) pays for the launch. Fewer units split at least as much.  Both
    lines read it so."""
    cc = _LINES[line]
    assert _decode_tile(16, 4096, device_cc=cc) == 8
    assert _decode_tile(8, 4096, device_cc=cc) >= 8
    assert _decode_tile(2, 4096, device_cc=cc) == 16  # 32 tiles: the thinnest split keeps two


def test_decode_tile_wide_q_tile_splits_the_serving_shape():
    """The same 64 units on the 32-column tile (S_q=2 x 16:1, the MTP step) ON THE
    BLACKWELL LINE (``device_cc=(10, 0)``): a lone CTA there streams a tile 1.6x
    slower (90.2 vs 57.1 us unsplit), so split 2 saves ~32 us of GPU time --
    enough to pay for the launch -- and leads in both regimes; a saturated launch
    still does not split.  The adapter does not route that tile on that line
    (config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS keeps (16, 32] rows on the prefill
    tile: even split, 96-103 us eager against the prefill tile's 66 us); its fit
    stays pinned for when it is routed.  The Rubin line's sibling kernel has its
    own, measured, per-width table (the test below)."""
    assert _decode_tile(64, 4096, q_tile=32, device_cc=(10, 0)) == 2
    assert _decode_tile(64, 4096, q_tile=32, device_cc=(10, 0), launch_cost=0.0) == 2
    assert _decode_tile(16, 4096, q_tile=32, device_cc=(10, 0)) == 8
    assert _decode_tile(256, 4096, q_tile=32, device_cc=(10, 0)) == 1


def test_decode_tile_rubin_widths_are_priced_by_their_measured_stream_rates():
    """The cc 10.7 decode tile's per-CTA time per 128-key KV tile, measured on the Rubin
    performance class (212 SMs, device time): 16-column 0.90-1.00 us, 32-column 1.03-1.07,
    64-column 1.55-1.60 -- so the model prices a lone CTA's tile at 1.0 / 1.1 / 1.65 by
    width there (the 32-column tile is NOT the Blackwell line's issue-bound 1.6x), and one
    64-column unit costs three quarters of two 32-column units on the same KV range.  What
    the 32-column price decides: 128 CTAs of the 32-column tile over 256 tiles, priced as
    128 single units, stay UNSPLIT -- the Blackwell constant split them -- and so do 512 of
    them over 32 tiles.  (The B 32 x kv 32K MTP step's two-unit 32-column route IS those 128
    CTAs, but its two units per (sequence, KV head) share one KV stream per split: priced
    that way, the test below, it splits to the measured optimum.)  ``device_cc=None`` reads
    the Rubin table (the only line that routes a width past 16; the 16-column tile costs 1.0
    on both)."""
    from cudnn.sdpa.fwd.heuristics import _decode_tile_cost_for

    assert [_decode_tile_cost_for(w, (10, 7)) for w in (16, 32, 64)] == [1.0, 1.1, 1.65]
    assert [_decode_tile_cost_for(w, None) for w in (16, 32, 64)] == [1.0, 1.1, 1.65]
    assert _decode_tile_cost_for(16, (10, 0)) == 1.0 and _decode_tile_cost_for(32, (10, 0)) == 1.6 and _decode_tile_cost_for(32, (10, 3)) == 1.6
    assert _decode_tile(128, 32768, sm_count=212, q_tile=32) == 1
    assert _decode_tile(128, 32768, sm_count=212, q_tile=32, launch_cost=0.0) == 1
    assert _decode_tile(512, 4096, sm_count=212, q_tile=32) == 1
    # the serving shape's 32-column one-unit form (S_q 2 x 16:1 at B 32) still splits once -- on the stream rate, not the launch
    assert _decode_tile(64, 4096, sm_count=212, q_tile=32, launch_cost=0.0) == 2


def test_decode_tile_two_unit_routes_share_one_kv_stream_per_split():
    """A route in TWO token units (the MTP step past a width's packed rows: two CTAs per (sequence,
    KV head, split), each a slice of the tokens) streams the SAME KV range from both units, so a
    wave of its CTAs carries half as many HBM streams as it has CTAs -- the model prices a wave by
    its UNIQUE streams (``token_units``, the units of one (sequence, KV head) counted once per
    split), never below a lone CTA's rate, and the CTA waves themselves stay what they are.

    Measured on the Rubin performance class (212 SMs), the B 32 x kv 32K MTP step on the two-unit
    32-column route (128 CTAs x 256 tiles, two units per KV head) at forced splits: 277.6 us unsplit,
    307.8 at 2, 219.4 at 3, 249.1 at 4 -- the order 3 < 4 < 1 < 2.  Priced as 128 single units the
    model had unsplit leading (283.8 against 293.8 at 3 and 383.2 at 2): the cost of a wave of
    shared streams was counted as two waves' worth of HBM.  With the sharing the costs order as the
    measurement does and the pick is 3 (one wave of UNIQUE streams: 384 CTAs, 192 streams) in both
    regimes; a deeper split, whose unique streams spill past one wave of SMs, is outside what was
    measured for a two-unit route and is not taken until it is.  ``token_units=1`` (the default) is
    the single-unit pricing exactly: every single-unit pick is unchanged by naming it."""
    import cudnn.sdpa.fwd.heuristics as heur

    two = dict(units=128, kv_tiles=256, sm_count=212, tile_cost=1.1, launch_cost=0.0, grid=(32, 24, 4), q_tile=32)
    c = {s: heur._decode_tile_cost(s, token_units=2, **two) for s in (1, 2, 3, 4)}
    assert c[3] < c[4] < c[1] < c[2], c  # measured 219.4 < 249.1 < 277.6 < 307.8 us
    one = {s: heur._decode_tile_cost(s, **two) for s in (1, 2, 3, 4)}  # the same CTAs priced as 128 single units: unsplit leads
    assert one[1] < one[3] < one[4] < one[2] and one[1] == heur._decode_tile_cost(1, token_units=1, **two), one
    cell = dict(units=128, s_kv=32768, sm_count=212, q_tile=32, combine_grid=(32, 24, 4))
    assert _decode_tile(token_units=2, **cell) == 3 and _decode_tile(token_units=2, launch_cost=0.0, **cell) == 3
    assert _decode_tile(**cell) == 1 and _decode_tile(launch_cost=0.0, **cell) == 1  # priced single-unit: the former reading
    assert _decode_tile(token_units=2, sm_count=204, units=128, s_kv=32768, q_tile=32, combine_grid=(32, 24, 4)) == 3
    # the measured regime of a two-unit route: its unique streams within one wave of SMs
    inside = heur.decode_tile_split_inside_measured_waves
    assert inside(3, units=128, sm_count=212, token_units=2) and not inside(4, units=128, sm_count=212, token_units=2)
    assert inside(1, units=128, sm_count=212, token_units=1) and not inside(2, units=128, sm_count=212, token_units=1)
    assert inside(2, units=256, sm_count=212, token_units=1) and not inside(
        4, units=256, sm_count=212, token_units=1
    )  # the measured floor of a launch past one wave
    assert inside(1, units=0, sm_count=212, token_units=1)
    # the single-unit picks are untouched by the parameter (its default): a sweep of the measured cells, both regimes
    for units, s_kv, q_tile, grid in (
        (64, 32768, 64, (32, 24, 4)),
        (2, 32768, 16, (1, 24, 1)),
        (8, 131072, 16, (4, 24, 1)),
        (256, 4096, 64, (128, 24, 4)),
        (64, 4096, 16, (32, 32, 1)),
    ):
        for lc in (17.0, 0.0):
            kw = dict(sm_count=212, q_tile=q_tile, combine_grid=grid, launch_cost=lc)
            assert _decode_tile(units, s_kv, token_units=1, **kw) == _decode_tile(units, s_kv, **kw) == _decode_tile(units, s_kv, token_units=0, **kw), (
                units,
                s_kv,
            )
    # the 24/2 MTP steps in two units of the 64-column tile (S_q 6 .. 8: 128 CTAs at B 32) price the same way at 212 SMs: the pick
    # is within one wave of unique streams and a member of the ladder
    for s_kv in (4096, 32768, 131072):
        pick = _decode_tile(128, s_kv, sm_count=212, q_tile=64, combine_grid=(32, 24, 8), token_units=2)
        assert inside(pick, units=128, sm_count=212, token_units=2) and pick in heur.decode_tile_split_candidates(sm_count=212, kv_tiles=s_kv // 128), (
            s_kv,
            pick,
        )


def test_decode_tile_keeps_the_previous_pick_past_the_measured_waves():
    """Past the regime the wave term was measured in -- a pick that puts a single-unit launch past one
    wave of SMs (many waves of short CTAs), except the measured floor of a launch that is past one wave
    unsplit -- the model is an extrapolation: the per-width CTA cost was fitted on ONE row (B 128 x kv 4K)
    and the only measured analog of the cells it moves, B 4 x kv 128K at S_q 1 on the 16-column tile,
    ordered the splits 16 < 64 < 128 < 32 while the wave arithmetic ranks the S_q 3 / 4 twins on the
    64-column tile 64 < 128 < 16 < 32.  Until those cells are measured the model returns the
    saturation-only pricing's pick there (the pricing before the wave term: a lone CTA's rate or the
    stream count against the saturation point, one tile of fixed cost per CTA; the candidate list is the
    same), so B 4 x kv 128K at S_q 3 / 4 stays at 32 (the wave arithmetic said 64 = 512 CTAs) and B 128
    at kv 32K / 128K stays unsplit on a 212-SM part (it said 4 = 1024 CTAs), both regimes -- while every
    pick inside the regime is the wave term's: the one-wave split 3, the B 1 / B 4 optima, the
    B 128 x kv 4K floor (eager 1, replay 2) and, on a 204-SM part where 256 units are a wave and a
    quarter, the floor's split 2 at kv 32K."""
    import cudnn.sdpa.fwd.heuristics as heur

    inside = heur.decode_tile_split_inside_measured_waves
    for sm in (212, 204):
        for s_q in (3, 4):
            for hq in (24, 32):
                for lc in (17.0, 0.0):
                    assert _decode_tile(8, 131072, sm_count=sm, q_tile=64, combine_grid=(4, hq, s_q), launch_cost=lc) == 32, (sm, s_q, hq, lc)
    for kv in (32768, 131072):
        for s_q, q_tile in ((1, 16), (2, 32), (3, 64), (4, 64)):
            for lc in (17.0, 0.0):
                assert _decode_tile(256, kv, sm_count=212, q_tile=q_tile, combine_grid=(128, 24, s_q), launch_cost=lc) == 1, (kv, s_q, lc)
    # what the wave arithmetic says there (outside the regime) and what the saturation-only pricing says: the record of the fallback
    kw = dict(units=8, kv_tiles=1024, sm_count=212, tile_cost=1.65, launch_cost=0.0, grid=(4, 24, 4), q_tile=64)
    c = {s: heur._decode_tile_cost(s, **kw) for s in (16, 32, 64, 128)}
    assert min(c, key=c.get) == 64 and not inside(64, units=8, sm_count=212) and inside(16, units=8, sm_count=212), c
    sat = {s: heur._decode_tile_saturation_cost(s, **kw) for s in (16, 32, 64, 128)}
    assert min(sat, key=sat.get) == 32, sat
    assert heur._decode_tile_saturation_cost(1, **kw) == pytest.approx((1024 + heur._DECODE_TILE_UNIT_CTA_COST) * 1.65)
    # inside the regime the wave term decides: the measured cells
    assert _decode_tile(64, 32768, sm_count=212, q_tile=64, combine_grid=(32, 24, 4)) == 3
    assert _decode_tile(8, 131072, sm_count=212, combine_grid=(4, 24, 1)) == 16  # the measured analog: the one-wave split
    big = dict(units=256, s_kv=4096, q_tile=64, combine_grid=(128, 24, 4))
    assert _decode_tile(sm_count=212, **big) == 1 and _decode_tile(sm_count=212, launch_cost=0.0, **big) == 2
    assert inside(2, units=256, sm_count=204) and not inside(4, units=256, sm_count=204)
    for lc in (17.0, 0.0):
        assert _decode_tile(256, 32768, sm_count=204, combine_grid=(128, 24, 1), launch_cost=lc) == 2, lc


@pytest.mark.parametrize("line", sorted(_LINES))
def test_decode_tile_saturated_launch_never_splits(line):
    """256 units (b=128) already saturate HBM twice over, 128 once: the bytes
    do not change with the split, only the fixed costs, so neither regime
    splits -- on both lines."""
    cc = _LINES[line]
    for units in (128, 256):
        assert _decode_tile(units, 4096, device_cc=cc) == 1
        assert _decode_tile(units, 4096, device_cc=cc, launch_cost=0.0) == 1


def test_decode_tile_long_kv_pays_for_the_launch():
    """At 64 units the GPU saving of a split grows with the KV length (8192
    keys ~22 us, 16384 ~45 us) and covers the second launch at every length;
    past saturation a finer split only adds fixed cost.  The Blackwell line
    (its second launch at the device-side latency)."""
    assert _blackwell_tile(64, 8192) == 2
    assert _blackwell_tile(64, 8192, launch_cost=0.0) == 2
    assert _blackwell_tile(64, 16384) == 2
    assert _blackwell_tile(64, 65536) == 2


def test_decode_tile_long_kv_pays_for_the_launch_on_the_cc_10_7_line():
    """The same 64 units on the cc 10.7 line, whose eager charge is the row's own
    (_DECODE_TILE_SPLIT_LAUNCH_COST_CC107): 8192 keys (~22 us of saving) does not
    cover it, 16384 (~45 us) does; a captured caller splits at 8192 too; past
    saturation a finer split only adds fixed cost."""
    assert _decode_tile(64, 8192) == 1
    assert _decode_tile(64, 8192, launch_cost=0.0) == 2
    assert _decode_tile(64, 16384) == 2
    assert _decode_tile(64, 65536) == 2


def test_decode_tile_short_kv_splits_once_it_can():
    """Two or three tiles cannot be cut (the thinnest split keeps two), four and
    eight save too little at b=1 to pay for the second launch (a captured
    caller would still split eight), sixteen do (B200/B300: 25 -> 14 us split 8).
    The Blackwell line."""
    assert _blackwell_tile(2, 256) == 1
    assert _blackwell_tile(2, 384) == 1
    assert _blackwell_tile(2, 512) == 1
    assert _blackwell_tile(2, 1024) == 1
    assert _blackwell_tile(2, 1024, launch_cost=0.0) == 4
    assert _blackwell_tile(2, 2048) == 8


def test_decode_tile_short_kv_does_not_split_on_the_cc_10_7_line():
    """Two or three tiles cannot be cut (the thinnest split keeps two); a
    handful can be, but saves too little GPU time to pay for the cc 10.7 row's
    second launch -- a captured caller still splits them."""
    assert _decode_tile(2, 256) == 1
    assert _decode_tile(2, 384) == 1
    assert _decode_tile(2, 1024) == 1
    assert _decode_tile(2, 1024, launch_cost=0.0) > 1


@pytest.mark.parametrize("q_tile", [16, 32, 64])
@pytest.mark.parametrize("units", [2, 16, 64, 128])
@pytest.mark.parametrize("grid", [None, (1, 24, 1), (4, 24, 4)], ids=["lean", "grid_b1", "grid_b4_sq4"])
def test_decode_tile_longer_kv_never_splits_less(units, q_tile, grid):
    """On tile-aligned KV lengths the cc 10.7 line's choice is monotone in the length and a member
    of the decode ladder (the powers of two plus the wave-filling 3 / 5 / 6 / 7: split_kv is a
    kernel-module cache key, so the ladder is a closed list), under the lean and the
    decode-grid combine forms alike."""
    from cudnn.sdpa.fwd.heuristics import decode_tile_split_candidates

    got = [_decode_tile(units, 128 << i, q_tile=q_tile, combine_grid=grid) for i in range(12)]
    assert all(a <= b for a, b in zip(got, got[1:])), got
    for i, s in enumerate(got):
        assert s in decode_tile_split_candidates(sm_count=B200_SMS, kv_tiles=1 << i), (i, s, got)


@pytest.mark.parametrize("q_tile", [16, 32])
@pytest.mark.parametrize("units", [2, 16, 64, 128])
@pytest.mark.parametrize("serial", [False, True], ids=["decode_form", "serial_form"])
def test_decode_tile_longer_kv_never_splits_less_on_the_blackwell_line(units, q_tile, serial):
    """On tile-aligned KV lengths the Blackwell line's choice is monotone in the length and a
    power of two (its ladder is :func:`split_kv_candidates`), under either combine form."""
    got = [_blackwell_tile(units, 128 << i, q_tile=q_tile, serial_combine=serial) for i in range(12)]
    assert all(a <= b for a, b in zip(got, got[1:])), got
    assert all(s & (s - 1) == 0 for s in got), got


def test_decode_tile_ladder_fills_the_machine_and_reaches_one_tile_per_cta():
    """The decode tile's own candidate list: the powers of two up to the device's (the shared
    ladder) PLUS the splits that fill a wave of SMs that no power of two fills -- 3 / 5 / 6 / 7
    -- and, unlike the prefill ladder, down to ONE KV tile per CTA (the tile handles any
    chunking, empty ranges included; a batch-1 step on a 4K cache measured its optimum at
    32 splits x 1 tile).  Bounded by the tile count, ascending, starting at 1."""
    from cudnn.sdpa.fwd.heuristics import _DECODE_TILE_ODD_SPLITS, decode_tile_split_candidates

    assert _DECODE_TILE_ODD_SPLITS == (3, 5, 6, 7)
    assert decode_tile_split_candidates(sm_count=212, kv_tiles=256) == [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256]
    assert decode_tile_split_candidates(sm_count=148, kv_tiles=32) == [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]
    assert decode_tile_split_candidates(sm_count=212, kv_tiles=4) == [1, 2, 3, 4]
    assert decode_tile_split_candidates(sm_count=212, kv_tiles=1) == [1]
    assert decode_tile_split_candidates(sm_count=0, kv_tiles=32) == [1] and decode_tile_split_candidates(sm_count=212, kv_tiles=0) == [1]
    for sm in (148, 152, 204, 212):
        lad = decode_tile_split_candidates(sm_count=sm, kv_tiles=1 << 12)
        assert lad == sorted(set(lad)) and lad[0] == 1 and lad[-1] == 256, (sm, lad)
        assert set(split_kv_candidates(sm_count=sm, kv_tiles=1 << 12)) <= set(lad)


def test_decode_tile_wave_term_takes_the_one_wave_split_on_the_mtp_step():
    """The wave term.  A CTA streams its KV range serially, so a launch of more CTAs than SMs
    pays its tail wave in full: 256 CTAs on 212 SMs is two waves' worth of time, not 1.2.  The
    model charges every full wave at the HBM-saturated rate and the tail wave at its own stream
    count, and offers the splits that fill ONE wave (3 / 5 / 6 / 7 beside the powers of two).

    Measured on the Rubin performance class (212 SMs), the S_q 4 MTP step of the 24/2 geometry
    at B 32 x kv 32K on the 64-column tile (64 units x 256 tiles) at forced splits: split 3
    (192 CTAs, one wave) streams in 166.3 us -- the attention term 170.4 -- where the former
    pick, split 4 (256 CTAs, a full wave plus a 44-CTA tail), takes 245.8 and split 2 219.2,
    6 181.2, 8 207.6.  So the model picks 3 at 212 and 204 SMs; on a 148 / 152-SM part 192
    CTAs is itself a tail wave and the one-wave pick is 2 (128 CTAs).  A non-power-of-two
    split is taken only when it beats the best power of two by more than
    _DECODE_TILE_ODD_SPLIT_MARGIN (the fits' own scatter; it mints another specialization):
    the Blackwell 16-unit serving shape keeps 8 over 5, which the model prices 1 % apart.
    Many waves of SHORT CTAs are worse than the stream term alone says (the fixed cost
    of a CTA scales with its rows: at B 128 x kv 4K the 64-column tile ran 111.5 / 115.6 /
    147.6 us at splits 2 / 4 / 8, 16 / 8 / 4 tiles per CTA), so the per-CTA cost is one tile
    per 16 rows: 1 / 2 / 4 on the 16- / 32- / 64-column tiles -- and that cell stays at the
    measured floor (unsplit for an eager caller, 2 for a replaying one)."""
    import cudnn.sdpa.fwd.heuristics as heur

    mtp = dict(units=64, s_kv=32768, q_tile=64, combine_grid=(32, 24, 4))  # 768 lean blocks: the lean combine
    assert _decode_tile(sm_count=212, **mtp) == 3 and _decode_tile(sm_count=212, launch_cost=0.0, **mtp) == 3
    assert _decode_tile(sm_count=204, **mtp) == 3 and _decode_tile(sm_count=204, launch_cost=0.0, **mtp) == 3
    assert _decode_tile(sm_count=152, **mtp) == 2 and _decode_tile(sm_count=148, **mtp) == 2
    # the costs the pick rests on, at 212 SMs: one wave of 3 beats the two-wave 4 by more than the margin
    kw = dict(units=64, kv_tiles=256, sm_count=212, tile_cost=1.65, launch_cost=0.0, grid=(32, 24, 4), q_tile=64)
    c = {s: heur._decode_tile_cost(s, **kw) for s in (2, 3, 4, 6, 8)}
    assert c[3] < c[6] < c[8] < c[2] < c[4], c
    assert c[3] < (1 - heur._DECODE_TILE_ODD_SPLIT_MARGIN) * min(c[2], c[4], c[8])
    # the per-width CTA cost and the deep-split cell it governs: B 128 x kv 4K x S_q 4 (256 units of the 64-column tile)
    assert [heur._decode_tile_cta_cost_for(w) for w in (16, 32, 64)] == [1.0, 2.0, 4.0]
    big = dict(units=256, s_kv=4096, q_tile=64, combine_grid=(128, 24, 4))
    assert _decode_tile(sm_count=212, **big) == 1 and _decode_tile(sm_count=212, launch_cost=0.0, **big) == 2
    # the margin: the Blackwell 16-unit shape stays on 8 (5 is cheaper by ~1 %, inside the margin); the margin is small
    assert 0.0 < heur._DECODE_TILE_ODD_SPLIT_MARGIN <= 0.1
    assert _decode_tile(16, 4096) == 8
    assert heur._decode_tile_cost(5, units=16, kv_tiles=32, sm_count=B200_SMS, tile_cost=1.0, launch_cost=17.0, q_tile=16) < heur._decode_tile_cost(
        8, units=16, kv_tiles=32, sm_count=B200_SMS, tile_cost=1.0, launch_cost=17.0, q_tile=16
    )
    # the B 1 and B 4 serving cells with the decode-grid combine (the picks the measured optima name), at 212 / 204 / 152 SMs
    for sm in (212, 204, 152):
        assert _decode_tile(2, 4096, sm_count=sm, combine_grid=(1, 24, 1)) == 32, sm  # 32 x 1 tile: measured 10.5 us against 11.5 at 16
        assert _decode_tile(2, 32768, sm_count=sm, combine_grid=(1, 24, 1)) == 64, sm
        assert _decode_tile(2, 131072, sm_count=sm, combine_grid=(1, 24, 1)) == 64, sm  # 128 = 256 CTAs: the two-wave trap (measured 36.5 vs 30.6 us)
        assert _decode_tile(8, 131072, sm_count=sm, combine_grid=(4, 24, 1)) == 16, sm  # 32 = 256 CTAs likewise (107.4 vs 94.1 us)
        for kv, want in ((4096, 32), (32768, 64), (131072, 64)):
            assert _decode_tile(2, kv, sm_count=sm, launch_cost=0.0, combine_grid=(1, 24, 1)) == want, (sm, kv)
    # the lean form keeps the lean picks (the ladder's new 32 at kv 4K does not pay for a lean combine of 32 partials)
    assert _decode_tile(2, 4096, sm_count=212) == 16


def test_decode_tile_batch_one_prices_every_partial_of_the_combine(monkeypatch):
    """Batch 1 x 2 KV heads (2 units) on a 212-SM part under the LEAN combine kernel
    (no combine grid given, or a grid the lean kernel serves): the combine is one wave
    of warps walking the split axis serially, so every partial costs it a step
    (COMBINE_PER_PARTIAL). Measured optima of kernel + combine (grouped-load
    combine, 2376 MHz lock): kv 4K 16 (14.6 us), kv 32K 32 (24.1), kv 128K 64
    (41.6); 8 units (batch 4) x 128K 16 (96.8). Without the term the model took
    64 at kv 32K (26.0 us measured) and 64 at kv 128K with the old combine
    (68.3 us against 61.1 at 32).  The cc 10.7 row prices every partial of its LEAN
    forms at that measured rate (decode_tile_combine_per_partial, the row's ``per_split``:
    the same number the line reads by default, so the picks below hold with and without
    it); the Blackwell line carries its own fit."""
    import cudnn.sdpa.fwd.heuristics as heur

    assert _decode_tile(2, 4096, sm_count=212) == 16
    assert _decode_tile(2, 32768, sm_count=212) == 32
    assert _decode_tile(2, 131072, sm_count=212) == 64
    assert _decode_tile(8, 131072, sm_count=212) == 16
    # a grid the lean kernel serves (a large-batch decode: B 32 x 24 heads = 768 lean blocks) prices the same way
    assert _decode_tile(2, 32768, sm_count=212, combine_grid=(32, 24, 1)) == 32
    # the captured caller (no launch term) keeps the same splits at these shapes
    assert _decode_tile(2, 32768, sm_count=212, launch_cost=0.0) == 32
    assert _decode_tile(2, 131072, sm_count=212, launch_cost=0.0) == 64

    def rubin(units, s_kv, **kw):
        return _decode_tile(units, s_kv, sm_count=212, per_split=heur._DECODE_TILE_COMBINE_PER_PARTIAL, **kw)

    assert rubin(2, 4096) == 16
    assert rubin(2, 32768) == 32
    assert rubin(2, 131072) == 64
    assert rubin(8, 131072) == 16
    # the captured caller (no launch term) keeps the same splits at these shapes
    assert rubin(2, 32768, launch_cost=0.0) == 32
    assert rubin(2, 131072, launch_cost=0.0) == 64
    # the per-partial term is what decides: with it at 0 the model runs to 64 at kv 32K and 128K (the flat per-wave combine price
    # makes every finer split free once the loop term stops paying for it -- the 40 us combine measured behind a 12 us kernel)
    monkeypatch.setattr(heur, "_DECODE_TILE_COMBINE_PER_PARTIAL", 0.0)
    assert rubin(2, 32768) == 64
    assert rubin(2, 131072) == 64


def test_decode_tile_serial_combine_bounds_the_split_depth():
    """One KV head at 128k keys (1024 tiles, 1 unit): with no per-split term the
    model took 128 splits at every S_q.  At S_q >= 3 the serial combine pays
    ~0.45 us a split, so the 128-split pass alone was ~60 of the measured 72 us
    (qwen3.5-tp8 b=1 q=4 on B300, 1.5x behind the best flashinfer backend).
    The serial term caps it; the decode form's smaller term still lets the
    one-row launch split deep, and the saturated serving shape is untouched."""
    serial = _blackwell_tile(1, 131072, serial_combine=True)
    decode = _blackwell_tile(1, 131072)
    assert serial <= 64, serial
    assert decode >= serial, (decode, serial)
    assert decode >= 32, decode
    assert _blackwell_tile(64, 4096, serial_combine=True) == 1
    assert _blackwell_tile(2, 2048, serial_combine=True) == _blackwell_tile(2, 2048) == 8


@pytest.mark.parametrize("units", [1, 2, 8])
def test_decode_tile_serial_combine_never_splits_more(units):
    """Pricing the serial walk can only lower the split, never raise it (the Blackwell line, which
    reads the flag; the cc 10.7 line prices its combine by the form the grid takes and does not)."""
    for s_kv in (2048, 8192, 32768, 131072):
        assert _blackwell_tile(units, s_kv, serial_combine=True) <= _blackwell_tile(units, s_kv)
        assert _decode_tile(units, s_kv, serial_combine=True) == _decode_tile(units, s_kv)


def test_decode_tile_prices_the_combine_by_the_form_its_grid_takes():
    """The combine's per-partial price depends on which kernel the launch takes, and that is
    decided by the combine GRID (``combine_grid`` = the graph's (B, H_q, S_q): the lean kernel's
    ceil(S_q / 4) x H_q x B blocks, :func:`decode_combine_lean_blocks`, bound which launches
    change at all; the rows S_q x H_q x B, :func:`decode_combine_rows`, are what the decode-grid
    kernel's own time grows with): the DECODE-GRID kernel spreads the
    split axis over eight lane chains, so its time is FLAT in the split count -- measured on
    the Rubin performance class at B 1 x 24 heads: 2.0-2.1 us (S_q 1) / 2.4-2.7 (S_q 4)
    standalone from 4 to 128 splits, 3.1-3.7 us in the step, a slope of 0.0004-0.009 us per
    partial against the lean kernel's 0.127-0.177 -- so a batch-1 launch on it splits as deep
    as the stream saturation allows.  Measured step optima with the decode-grid combine at
    B 1 x 24 heads: kv 32K 64 (14.4 us against 23.4 at the lean split 32), kv 128K 64 (30.6),
    B 4 x kv 128K 16 (94.1); the lean form's picks (the test above) are unchanged.  The
    decode-grid intercept grows with the ROWS (standalone 2.08 / 2.4 / 4.93 us at 24 / 96 / 384
    rows: B 1 x S_q 1, B 1 x S_q 4 = B 4 x S_q 1, B 4 x S_q 4 at 24 heads), which keeps a
    mid-sized grid from splitting past what its combine saves."""
    import cudnn.sdpa.fwd.heuristics as heur

    assert (
        heur.decode_combine_lean_blocks(1, 24, 1) == 24  # one row per block at S_q 1: the lean grid is per (head, batch)
        and heur.decode_combine_lean_blocks(4, 24, 4) == 96
        and heur.decode_combine_lean_blocks(2, 4, 1024) == 2048
        and heur.decode_combine_rows(1, 24, 1) == 24
        and heur.decode_combine_rows(1, 24, 4) == heur.decode_combine_rows(4, 24, 1) == 96
        and heur.decode_combine_rows(4, 24, 4) == 384
    )
    assert heur._DECODE_TILE_COMBINE_PER_PARTIAL_DECODE_GRID < heur._DECODE_TILE_COMBINE_PER_PARTIAL
    six = dict(sm_count=212, combine_grid=(1, 24, 1))
    assert _decode_tile(2, 32768, **six) == 64 and _decode_tile(2, 32768, launch_cost=0.0, **six) == 64
    assert _decode_tile(2, 131072, **six) == 64 and _decode_tile(2, 131072, launch_cost=0.0, **six) == 64
    assert _decode_tile(8, 131072, sm_count=212, combine_grid=(4, 24, 1)) == 16
    # the price is the form's, not the batch's: the lean form at the same shapes keeps the lean picks
    assert _decode_tile(2, 32768, sm_count=212, combine_grid=(128, 32, 1)) == 32
    # the combine price per partial is what the form changes: the decode-grid intercept at a small grid is the lean one
    assert heur._decode_tile_combine_cost(64, grid=(1, 24, 1)) < heur._decode_tile_combine_cost(64, grid=None)
    assert heur._decode_tile_combine_cost(64, grid=(1, 24, 1)) == pytest.approx(
        heur._DECODE_TILE_COMBINE_COST + 64 * heur._DECODE_TILE_COMBINE_PER_PARTIAL_DECODE_GRID
    )
    assert heur._decode_tile_combine_cost(64, grid=None) == pytest.approx(heur._DECODE_TILE_COMBINE_COST + 64 * heur._DECODE_TILE_COMBINE_PER_PARTIAL)
    # the row-count term of the decode-grid intercept: flat up to the small-grid bound (96 rows either way), growing past it
    assert heur._decode_tile_combine_cost(64, grid=(4, 24, 1)) == pytest.approx(heur._decode_tile_combine_cost(64, grid=(1, 24, 4)))
    assert heur._decode_tile_combine_cost(64, grid=(1, 24, 4)) == pytest.approx(heur._decode_tile_combine_cost(64, grid=(1, 24, 1)))
    assert heur._decode_tile_combine_cost(64, grid=(4, 24, 4)) > heur._decode_tile_combine_cost(64, grid=(4, 24, 1))


def test_decode_combine_form_rule_is_split_aware_at_the_crossover():
    """Which combine kernel a split launch takes (``decode_combine_grid_selected``), measured on the
    Rubin performance class with the combine alone at 24 heads: the decode-grid kernel is flat in
    the split count but costs more blocks per row, so on a SMALL grid (up to 128 rows: B 1 x S_q 1
    = 24 rows, B 1 x S_q 4 and B 4 x S_q 1 = 96) it wins from split 8 on (2.1-2.5 us against the
    lean kernel's 3.4-18 us) and loses only at split 4 (by 0.03-0.4 us), while at B 4 x S_q 4
    (384 rows) its 4.9-5.4 us loses to the lean walk up to split 16 (3.9 us) and wins from
    split 32 (6.0 us) -- the crossover is near 23.  The rule: the decode-grid kernel when the
    launch is inside the kernel module's own grid bound (``DECODE_COMBINE_MAX_LEAN_BLOCKS`` lean
    blocks), the split is at least ``DECODE_COMBINE_MIN_SPLIT`` (8), and the grid is small or the
    split at least ``DECODE_COMBINE_DEEP_SPLIT`` (32); the lean kernel otherwise.  The kernel
    module's own entry takes the decode-grid kernel on every launch inside its bound, so
    ``decode_combine_force_lean`` names exactly the launches the adapter must pin to the lean
    kernel; a launch outside the bound is the lean kernel's by construction either way."""
    from cudnn.sdpa.fwd import heuristics as heur

    assert (heur.DECODE_COMBINE_MIN_SPLIT, heur.DECODE_COMBINE_DEEP_SPLIT, heur.DECODE_COMBINE_SMALL_GRID_ROWS) == (8, 32, 128)
    sel, force = heur.decode_combine_grid_selected, heur.decode_combine_force_lean
    # the B 4 x S_q 4 crossover (384 rows, 96 lean blocks): lean up to 16, the decode-grid kernel from 32
    assert [sel(4, 24, 4, s) for s in (4, 8, 16, 32, 64, 128)] == [False, False, False, True, True, True]
    assert [force(4, 24, 4, s) for s in (4, 8, 16, 32, 64)] == [True, True, True, False, False]
    # the small grids: B 1 x S_q 1 (24 rows), B 1 x S_q 4 and B 4 x S_q 1 (96 rows): the decode-grid kernel from split 8
    for grid in ((1, 24, 1), (1, 24, 4), (4, 24, 1), (1, 32, 1), (4, 32, 1)):
        assert [sel(*grid, s) for s in (2, 4, 8, 16, 32, 64, 128)] == [False, False, True, True, True, True, True], grid
        assert force(*grid, 4) and not force(*grid, 8) and not force(*grid, 64), grid
    # the band between (more than 128 rows inside the kernel's bound): the deep splits only
    for grid in ((2, 32, 4), (4, 32, 4), (2, 24, 8), (4, 32, 2)):
        assert [sel(*grid, s) for s in (8, 16, 32, 64)] == [False, False, True, True], grid
        assert force(*grid, 16) and not force(*grid, 32), grid
    # past the kernel module's bound (B 8 x 32 heads x S_q 1 = 256 lean blocks; a prefill grid): the lean kernel by
    # construction, nothing to force
    for grid in ((8, 32, 1), (32, 24, 1), (128, 24, 4), (2, 4, 1024)):
        assert not any(sel(*grid, s) for s in (8, 32, 64, 128)) and not any(force(*grid, s) for s in (4, 16, 64)), grid
    # exactly at the bound (B 4 x 32 heads x S_q 1 = 128 lean blocks = 128 rows): a small grid
    assert sel(4, 32, 1, 8) and sel(4, 32, 1, 16) and not sel(4, 32, 1, 4)
    # the split model prices the crossover cell by the rule: B 4 x 24/2 x kv 4K x S_q 4 on the 64-column tile stays at 16
    # (its lean combine) rather than jumping to the decode-grid kernel's deep splits
    assert _decode_tile(8, 4096, sm_count=212, q_tile=64, combine_grid=(4, 24, 4)) == 16
    assert heur._decode_tile_combine_cost(16, grid=(4, 24, 4)) == pytest.approx(heur._decode_tile_combine_cost(16, grid=None))
    assert heur._decode_tile_combine_cost(32, grid=(4, 24, 4)) < heur._decode_tile_combine_cost(32, grid=None)


@requires_dsl
def test_decode_combine_lean_blocks_is_the_combine_modules_own_grid():
    """The heuristics' reading of the lean combine grid (``decode_combine_lean_blocks``) is the kernel module's
    ``lean_grid_blocks`` -- the host twin of what ``_launch_combine`` computes -- over a grid of shapes, so the split
    model and the launch agree on which form a launch takes."""
    from cudnn.sdpa.fwd.heuristics import DECODE_COMBINE_LEAN_ROWS_PER_BLOCK, decode_combine_lean_blocks
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    assert DECODE_COMBINE_LEAN_ROWS_PER_BLOCK == comb.ROWS_PER_BLOCK
    for b in (1, 2, 3, 4, 32, 128):
        for h in (1, 4, 24, 32, 34):
            for s_q in (1, 2, 3, 4, 5, 8, 17, 1024):
                assert decode_combine_lean_blocks(b, h, s_q) == comb.lean_grid_blocks(b, h, s_q), (b, h, s_q)


@pytest.mark.parametrize("line", sorted(_LINES))
def test_decode_tile_degenerate_inputs_do_not_split(line):
    cc = _LINES[line]
    assert choose_decode_tile_split_kv(units=0, kv_tiles=32, sm_count=B200_SMS, device_cc=cc) == 1
    assert choose_decode_tile_split_kv(units=64, kv_tiles=0, sm_count=B200_SMS, device_cc=cc) == 1
    assert choose_decode_tile_split_kv(units=64, kv_tiles=32, sm_count=0, device_cc=cc) == 1


def test_decode_tile_lines_are_named_by_the_device_cc_and_the_row():
    """The line selector: cc 10.0 / 10.3 read the Blackwell fit, ``None`` and cc 10.7 the Rubin one; a ROW
    names its line by its arch (the Rubin f16 row cc 10.7, every other row the Blackwell line), so the
    ranked plans of a graph do not depend on the host device; the line's default eager charge is its own
    (the Blackwell line's device-side launch latency, the cc 10.7 row's charge)."""
    from cudnn.sdpa.fwd import heuristics as heur
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    assert [heur._decode_tile_blackwell_line(cc) for cc in ((10, 0), (10, 3), (10, 7), None, (8, 0), (12, 0))] == [True, True, False, False, False, False]
    rows = {s.capabilities.sm_lo: heur.decode_tile_line_cc(s.capabilities) for s in ENGINE_SPECS}
    assert rows[107] == (10, 7) and rows[100] == (10, 0)
    assert (
        heur.decode_tile_split_launch_cost(next(s.capabilities for s in ENGINE_SPECS if s.capabilities.sm_lo == 107))
        == heur._DECODE_TILE_SPLIT_LAUNCH_COST_CC107
    )
    assert heur.decode_tile_split_launch_cost(next(s.capabilities for s in ENGINE_SPECS if s.capabilities.sm_lo == 100)) == heur._DECODE_TILE_SPLIT_LAUNCH_COST
    # the default charge is the line's: naming it changes nothing
    for units, s_kv in ((64, 4096), (64, 8192), (2, 2048), (2, 1024)):
        assert _blackwell_tile(units, s_kv) == _blackwell_tile(units, s_kv, launch_cost=heur._DECODE_TILE_SPLIT_LAUNCH_COST)
        assert _decode_tile(units, s_kv) == _decode_tile(units, s_kv, launch_cost=heur._DECODE_TILE_SPLIT_LAUNCH_COST_CC107)
        assert _decode_tile(units, s_kv, device_cc=(10, 7)) == _decode_tile(units, s_kv)


# --- a quantized O is a legal split target ---------------------------------
#
# The split kernels write HALF partials whatever the O dtype and the combine
# performs the only cast down to it, so nothing about an FP8 output makes the
# reduction narrower. Both the eligibility gate and the candidate generator
# used to decline it; these pin that they no longer do.


def _fp8_caps():
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    return next(sp for sp in ENGINE_SPECS if sp.name == "sdpa_fwd_prefill_sm100_fp8").capabilities


def _fp8_facts(dtype_o, *, mx=False):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga

    return ga.SdpaGraphFacts(
        b=1,
        h_q=8,
        h_kv=1,
        s_q=512,
        s_kv=16384,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=dtype_o,
        is_fp8=not mx,
        is_mxfp8=mx,
        device_sm_count=B200_SMS,
        device_cc=(10, 0),
    )


@pytest.mark.parametrize("out_name", ["FP8_E4M3", "FP8_E5M2", "HALF", "BFLOAT16"])
@pytest.mark.parametrize("mx", [False, True], ids=["fp8", "mxfp8"])
def test_quantized_output_does_not_block_a_split(out_name, mx):
    """mismatch() must not decline split_kv on the O dtype alone."""
    import cudnn

    why = mismatch(_fp8_caps(), _fp8_facts(getattr(cudnn.data_type, out_name), mx=mx), SdpaFwdKnobs(split_kv=4)) or ""
    assert "split_kv" not in why, f"quantized O declined the split: {why}"


@pytest.mark.parametrize("out_name", ["FP8_E4M3", "FP8_E5M2"])
def test_quantized_output_still_gets_split_candidates(out_name):
    """And the generator must actually PROPOSE one — passing the eligibility
    gate is not enough if the candidate list is still hard-coded to [1]."""
    import cudnn

    from cudnn.sdpa.fwd.heuristics import _split_points

    points = _split_points(_fp8_caps(), _fp8_facts(getattr(cudnn.data_type, out_name)), 128, 128, 2)
    assert points[0] > 1, f"a decode-shaped FP8-out graph got no split: {points}"
    assert points[-1] == 1, "no-split must remain reachable behind the chosen split"


def test_quantized_and_half_outputs_choose_the_same_split():
    """The O dtype is not a performance input: partials are half either way, so
    the chooser must land on the same split for an FP8 and a bf16 output."""
    import cudnn

    from cudnn.sdpa.fwd.heuristics import _split_points

    caps = _fp8_caps()
    fp8 = _split_points(caps, _fp8_facts(cudnn.data_type.FP8_E4M3), 128, 128, 2)
    half = _split_points(caps, _fp8_facts(cudnn.data_type.BFLOAT16), 128, 128, 2)
    assert fp8 == half, f"O dtype moved the split choice: {fp8} vs {half}"


@requires_dsl
@pytest.mark.parametrize("kind", ["dense", "ragged", "quantized"])
@pytest.mark.parametrize("stats", [False, True])
def test_every_combine_call_site_matches_the_compiled_arity(kind, stats):
    """Actual prepared frames must match each pointer host's runtime ABI.

    The fixture contains metadata and sentinel addresses only. Both compiled
    launch functions are intercepted; no CUDA operation receives these addresses.
    """
    import inspect
    from typing import get_origin

    import cutlass
    from cudnn.sdpa.fwd import prepared as prep
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine
    from test_sdpa_prepared_geometry import _split_fixture

    host = {"dense": split_combine._host_ptr, "ragged": split_combine._host_ptr_ragged, "quantized": split_combine._host_ptr_quantized}[kind]
    params = [
        param
        for param in inspect.signature(host).parameters.values()
        if param.annotation is not cutlass.Constexpr and get_origin(param.annotation) is not cutlass.Constexpr
    ]
    signature = inspect.Signature(params)
    seen = []

    def check_frame(*args):
        assert len(args) == len(params), (kind, len(args), list(signature.parameters))
        signature.bind(*args)
        assert args[-1] == 17
        seen.append(args)

    spec, facts = _split_fixture(stats)
    spec.fn = lambda *args: None
    spec.combine = spec.combine._replace(fn=check_frame)
    if kind == "ragged":
        spec.ragged, spec.ragged_i64 = True, False
        spec.ragged_lse_head_major = False  # The fixture below declares token-major Stats.
        spec.total_q = spec.b * spec.s_q_max
        spec.ragged_divs = (spec.qh * spec.d_qk, spec.qh * spec.d_v, spec.qh)
        for i, role in enumerate(("ragged_q", "ragged_o", "ragged_lse")):
            if role != "ragged_lse" or stats:
                facts[role] = prep.BufferFacts(0x50000 + i * 0x1000, "int32", (2, 0), spec.b + 1, (spec.b + 1,), (1,))
        if stats:
            facts["lse"] = facts["lse"]._replace(shape=(spec.total_q, spec.qh), strides=(spec.qh, 1))
    if kind == "quantized":
        spec.quant = prep.QuantizedLaunchSpec(True, 65536)
        for role in ("q", "k", "v"):
            spec.expect[role] = "float8_e4m3fn"
            facts[role] = facts[role]._replace(dtype="float8_e4m3fn")
        for i, role in enumerate(("descale_q", "descale_k", "descale_v", "scale_o", "amax_o")):
            facts[role] = prep.BufferFacts(0x60000 + i * 0x1000, "float32", (2, 0), 1, (1,), (1,))
            spec.index[role + "_ptr"] = len(spec.template)
            spec.template.append(None)
        # All scalars are present, so execute_quantized needs no CUDA fill.
        assert binding_reference.execute_quantized(spec, facts, 0x100000, 17, 17)
    else:
        _, args = binding_reference.bind_dense_split(spec, facts, 0x100000, 17, 17)
        spec.combine.fn(*args)
    assert len(seen) == 1, "no prepared combine frame checked"


def test_every_split_capable_sm100_kernel_has_the_slot():
    """A split-capable SM100 kernel either carries the o_partial_f32 slot, or the
    adapter must know it does not.

    _fp32_partial_split() is unconditional for SM100, so a kernel that wires
    make_split_helpers WITHOUT the slot gets handed an argument it never
    declares -- a launch-time arity error on whatever shape happens to split.
    That is exactly how sm100/prefill_d512_mxfp8 arrived: split-capable from the
    day it landed, written against the staged epilogue. Scanning the directory
    means the NEXT such kernel fails here instead of in production.

    A kernel without the slot is fine, provided _SLOTLESS_FLAVORS records it and
    the predicate excludes that flavor.
    """
    import pathlib

    from cudnn.sdpa.fwd import api_dsl

    # (mxfp8, d_shape) pairs the adapter deliberately keeps on half partials.
    _SLOTLESS_FLAVORS = {"prefill_d512_mxfp8.py"}

    kdir = pathlib.Path(api_dsl.__file__).parent / "kernels" / "sm100"
    assert kdir.is_dir(), f"kernel directory moved: {kdir}"

    offenders = []

    def has_slot(src: str) -> bool:
        if "\nEXPLICIT_ABI = True" in src:
            # explicit host entry: the slot is the o_partial_ptr parameter (the shared prologue
            # names o_partial_f32 in every host, so the source string is no evidence)
            sig = src[src.index("\ndef _host(") : src.index(") -> None:", src.index("\ndef _host("))]
            return "o_partial_ptr" in sig
        return "o_partial_f32" in src

    for f in sorted(kdir.glob("prefill_*.py")):
        src = f.read_text()
        if "make_split_helpers" not in src:
            continue  # not split-capable, nothing to carry
        if has_slot(src):
            continue  # wired
        if f.name in _SLOTLESS_FLAVORS:
            continue  # known, and excluded by _fp32_partial_split
        offenders.append(f.name)

    assert not offenders, (
        f"split-capable SM100 kernels with no o_partial_f32 slot: {offenders}. "
        "Either port the fp32 partial store into them, or exclude their flavor in "
        "api_dsl.SdpaFwdDsl._fp32_partial_split and list them in _SLOTLESS_FLAVORS."
    )

    # ...and the recorded exceptions must still be real, or this guard rots.
    for name in _SLOTLESS_FLAVORS:
        src = (kdir / name).read_text()
        assert "make_split_helpers" in src, f"{name}: no longer split-capable; drop it from _SLOTLESS_FLAVORS"
        assert not has_slot(src), f"{name}: now carries the slot; drop it from _SLOTLESS_FLAVORS and let the predicate return True"


@pytest.mark.parametrize("min_tiles", [1, 2, 4])
def test_candidates_and_choice_respect_a_supplied_min_tiles(min_tiles):
    for kv_tiles in (1, 3, 16, 17, 512):
        assert all(kv_tiles // s >= min_tiles for s in split_kv_candidates(sm_count=188, kv_tiles=kv_tiles, min_tiles=min_tiles) if s > 1)
        s = choose_split_kv(q_tiles=1, heads_q=1, batch=1, kv_tiles=kv_tiles, sm_count=188, combine_rows=8, min_tiles=min_tiles, combine_floor=0.1)
        assert s == 1 or kv_tiles // s >= min_tiles


@pytest.mark.parametrize(
    "device_cc", [None, (10, 7), (10, 0), (10, 3), (12, 0), (12, 1), (9, 0), (8, 0)], ids=lambda cc: "default" if cc is None else f"cc{cc[0]}_{cc[1]}"
)
def test_decode_combine_form_rule_holds_on_the_cc_10_7_line_only(device_cc):
    """``decode_combine_grid_selected`` / ``decode_combine_force_lean`` read the plan's DEVICE: the split-aware rule is the cc 10.7
    line's (measured on the Rubin performance class), so on cc 10.7 -- and on the model's default line, ``device_cc=None`` -- it
    answers the measured rule, while on EVERY other device (the Blackwell line, whose combine module's own forms are measured on
    B200 / B300 by their author, and any line the rule was never run on: SM120, Hopper, Ampere) it answers False and the adapter
    forces the module's own forms on every launch inside the kernel module's bound, and nothing outside it, where the module's
    entry is its own forms by construction.  The adapter's plan-time pin is pinned per line beside this
    (test_sdpa_prepared_geometry.py)."""
    from cudnn.sdpa.fwd import heuristics as heur

    def sel(*grid_and_split):
        return heur.decode_combine_grid_selected(*grid_and_split, device_cc=device_cc)

    def force(*grid_and_split):
        return heur.decode_combine_force_lean(*grid_and_split, device_cc=device_cc)

    rule_line = device_cc is None or tuple(device_cc) == (10, 7)
    # the small serving grid (B 1 x 24 heads x S_q 1 = 24 lean blocks, 24 rows): the rule takes the decode-grid kernel from split 8
    assert [sel(1, 24, 1, s) for s in (4, 8, 16, 32, 128)] == ([False, True, True, True, True] if rule_line else [False] * 5)
    assert [force(1, 24, 1, s) for s in (4, 8, 16, 32, 128)] == ([True, False, False, False, False] if rule_line else [True] * 5)
    # the B 4 x S_q 4 crossover (96 lean blocks, 384 rows): the rule takes the decode-grid kernel from split 32
    assert [sel(4, 24, 4, s) for s in (16, 32, 64)] == ([False, True, True] if rule_line else [False] * 3)
    assert [force(4, 24, 4, s) for s in (16, 32, 64)] == ([True, False, False] if rule_line else [True] * 3)
    # past the kernel module's bound (B 8 x 32 heads x S_q 1 = 256 lean blocks) nothing is selected or forced on any line
    assert not any(sel(8, 32, 1, s) or force(8, 32, 1, s) for s in (8, 32, 128))
    # ... and there the LEAN GROUP WALK (decode_combine_lean_walk_selected, the module's restored former lean kernel) serves
    # the launch on the cc 10.7 line -- the measured B 32 / B 128 x 24 heads cells at split 2-3 (768 / 3072 lean blocks), up to
    # the walk's up-front group (split 4: the measured regime; a deeper split keeps the module's own forms on every line) -- and
    # never inside the bound (the small grids keep the decode-grid kernel or the module's own forms) nor on any other line
    walk = lambda *grid_and_split: heur.decode_combine_lean_walk_selected(*grid_and_split, device_cc=device_cc)
    assert all(walk(8, 32, 1, s) is rule_line for s in (2, 3, 4))
    assert not any(walk(8, 32, 1, s) for s in (5, 8, 32, 128))
    assert heur.DECODE_COMBINE_LEAN_WALK_MAX_SPLIT == 4
    assert all(walk(*cell) is rule_line for cell in ((32, 24, 1, 2), (128, 24, 1, 2), (32, 24, 4, 3), (128, 24, 4, 2), (32, 32, 4, 3)))
    assert not any(walk(*grid, s) for grid in ((1, 24, 1), (4, 24, 4), (4, 32, 1)) for s in (2, 4, 8, 16, 32, 128))
    assert heur.DECODE_COMBINE_LEAN_WALK_MIN_LEAN_BLOCKS == heur.DECODE_COMBINE_MAX_LEAN_BLOCKS + 1
