# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""#601 phase C matrix: the canonical band model and the band-support decisions.

Host-side only: every case builds an :class:`~cudnn.sdpa.graph_analyzer.SdpaGraphFacts`
directly and asks a row's probe (or the model) for a decision, so this runs on
any host that can import the frontend.  Kernel-level mask numerics stay with the
per-kernel suites; what is pinned here is the DECISION surface, for every
discovered forward and backward row (enumerated dynamically -- no engine count
is hardcoded).

The oracle in each test is the row's own DECLARED claim (its legacy flags /
``Capabilities.padded`` / ``layouts`` / ...), never a copy of the model's
internal mapping, so a probe that stopped consulting the declaration would fail
here rather than agree with itself.
"""

from __future__ import annotations

import dataclasses

import cudnn
import pytest

from cudnn.sdpa import band
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.bwd import engines as bwd_engines
from cudnn.sdpa.fwd import engines as fwd_engines

# The default pytest.ini addopts is `-m L0`; mark the whole module so it runs.
pytestmark = pytest.mark.L0


# ---------------------------------------------------------------------------
# row discovery + legal facts sampling
# ---------------------------------------------------------------------------


def _rows():
    """``[(kind, EngineSpec)]`` -- every row the registry actually ships."""
    return [("fwd", spec) for spec in fwd_engines.ENGINE_SPECS] + [("bwd", spec) for spec in bwd_engines.ENGINE_SPECS]


def _probe(kind: str):
    return fwd_engines.mismatch if kind == "fwd" else bwd_engines.mismatch


_DTYPE_ORDER = ("HALF", "BFLOAT16", "FP8_E4M3", "FP8_E5M2")


def _pick_dtype(dtypes):
    by_name = {getattr(d, "name", str(d)): d for d in dtypes}
    for name in _DTYPE_ORDER:
        if name in by_name:
            return by_name[name]
    return sorted(dtypes, key=lambda d: getattr(d, "name", str(d)))[0]


def _dims(kind: str, caps):
    if kind == "fwd":
        shapes = sorted(tuple(s) for s in caps.d_shapes)
        return (128, 128) if (128, 128) in shapes else shapes[0]
    dims = sorted(caps.d)
    return (128, 128) if 128 in dims else (dims[-1], dims[-1])


def legal_facts(kind: str, caps, **overrides) -> ga.SdpaGraphFacts:
    """Facts that satisfy every NON-mask constraint of this row.

    The band axes are then varied one at a time on top of this, so an arch /
    dtype / envelope / layout rejection can never stand in for a mask decision.
    """
    d_qk, d_v = _dims(kind, caps)
    quantized = bool(getattr(caps, "is_mxfp8", False) or getattr(caps, "is_fp8", False))
    dtype = _pick_dtype(caps.dtypes)
    out_dtypes = getattr(caps, "out_dtypes", frozenset())
    values = dict(
        b=2,
        h_q=8,
        h_kv=8,
        s_q=128,
        s_kv=128,
        d_qk=d_qk,
        d_v=d_v,
        dtype=dtype,
        dtype_o=_pick_dtype(out_dtypes) if (quantized and out_dtypes) else dtype,
        uniform_dtype=True,
        uniform_out_dtype=True,
        is_mxfp8=bool(getattr(caps, "is_mxfp8", False)),
        is_fp8=bool(getattr(caps, "is_fp8", False)),
        is_backward=(kind == "bwd"),
        padded=False,
        thd=False,
        wants_stats=False,
        seq_q_trim=False,
        device_cc=(int(caps.sm_lo) // 10, int(caps.sm_lo) % 10),
        device_sm_count=1024,
        scale=0.1,
    )
    values.update(overrides)
    return ga.SdpaGraphFacts(**values)


# Band samples in the analyzer's own resolved vocabulary: causal == right_bound
# 0, right_band_widening == right_bound > 0, window_left == API length - 1.
def band_unmasked():
    return dict(causal=False, right_band_widening=False, right_bound=None, bottom_right=False, window_left=None)


def band_causal_top_left():
    return dict(causal=True, right_band_widening=False, right_bound=0, bottom_right=False, window_left=None)


def band_causal_bottom_right():
    return dict(causal=True, right_band_widening=False, right_bound=0, bottom_right=True, window_left=None)


def band_right(bound=4, bottom_right=False):
    return dict(causal=False, right_band_widening=True, right_bound=bound, bottom_right=bottom_right, window_left=None)


def band_window(window_left=3, causal=False, bottom_right=False):
    return dict(
        causal=causal,
        right_band_widening=False,
        right_bound=0 if causal else None,
        bottom_right=bottom_right,
        window_left=window_left,
    )


def declares(caps, *, right=None, left_window=False, bottom_right=False) -> bool:
    """The declaration oracle: what the row's own flags claim.

    ``right=None`` means "do not consult the right axis at all".
    """
    if right is None or right == band.RIGHT_UNBOUNDED:
        ok = True
    elif right == band.RIGHT_CAUSAL:
        ok = bool(caps.causal)
    elif right == band.RIGHT_FINITE:
        ok = bool(caps.right_band_widening)
    else:
        raise AssertionError(f"unknown right mode {right!r}")
    if left_window and not caps.swa:
        ok = False
    if bottom_right and not caps.bottom_right:
        ok = False
    return ok


# ---------------------------------------------------------------------------
# 1. the model exists on every row, and is a faithful normalization
# ---------------------------------------------------------------------------


def test_every_row_carries_a_normalized_band_support():
    seen = set()
    for kind, spec in _rows():
        caps = spec.capabilities
        assert spec.name not in seen, f"duplicate engine id {spec.name}"
        seen.add(spec.name)
        assert isinstance(caps.band, band.BandSupport), f"{spec.name}: band not normalized"
        # The normalization layer is the ONLY mapping of the legacy flags.
        assert caps.band == band.BandSupport.from_legacy_flags(
            causal=caps.causal,
            bottom_right=caps.bottom_right,
            swa=caps.swa,
            right_band_widening=caps.right_band_widening,
        ), spec.name
        # Each axis is exactly what the flag said -- no accidental widening.
        assert caps.band.serves_right_mode(band.RIGHT_UNBOUNDED) is True
        assert caps.band.serves_right_mode(band.RIGHT_CAUSAL) == bool(caps.causal)
        assert caps.band.serves_right_mode(band.RIGHT_FINITE) == bool(caps.right_band_widening)
        assert caps.band.serves_left_mode(band.LEFT_WINDOW) == bool(caps.swa)
        assert caps.band.serves_left_mode(band.LEFT_NONE) is True
        assert caps.band.serves_anchor(band.BandFacts(anchor=band.ANCHOR_BOTTOM_RIGHT)) == bool(caps.bottom_right)
        assert caps.band.serves_anchor(band.BandFacts(anchor=band.ANCHOR_TOP_LEFT)) is True
    assert {kind for kind, _ in _rows()} == {"fwd", "bwd"}


def test_unmasked_baseline_is_accepted_by_every_row():
    # If a row rejected its own non-mask-legal baseline, every mask comparison
    # for that row would be measuring the wrong thing.
    for kind, spec in _rows():
        facts = legal_facts(kind, spec.capabilities, **band_unmasked())
        assert _probe(kind)(spec.capabilities, facts) is None, spec.name


def test_band_support_rejects_unknown_or_empty_axis_claims():
    with pytest.raises(ValueError):
        band.BandSupport(right=frozenset({"sometimes"}))
    with pytest.raises(ValueError):
        band.BandSupport(left=frozenset({"leftish"}))
    with pytest.raises(ValueError):
        band.BandSupport(anchors=frozenset({"middle"}))
    with pytest.raises(ValueError):
        band.BandSupport(right=frozenset())


def test_shipped_rows_express_both_a_full_and_a_restricted_claim():
    # The tree contains RESTRICTED rows next to the full ones -- the MXFP8
    # backward (causal only, top-left only) and the SM89 forward (top-left
    # anchor only, no right-band widening) -- which is why "every built-in row
    # supports causal" cannot be read as "the causal axis can be deleted": a
    # claim narrower than the full set on at least one axis is declarable, and
    # it is what the probe decides with.
    full = band.BandSupport.full_masks()
    claims = {spec.name: spec.capabilities.band for _, spec in _rows()}
    assert full in claims.values()
    restricted = {name: claim for name, claim in claims.items() if claim != full}
    assert restricted, "no restricted row left in the tree; the model's restricted path is no longer exercised"
    for name, claim in restricted.items():
        assert claim.right < full.right or claim.left < full.left or claim.anchors < full.anchors, name
    # Both named restricted shapes the model ships are in the tree.
    assert band.BandSupport.causal_and_unmasked() in restricted.values()


# ---------------------------------------------------------------------------
# 2. a RESTRICTED row is expressible and is correctly rejected
# ---------------------------------------------------------------------------


def _restricted_row(kind: str = "fwd", *, sm_lo: int = 89, sm_hi: int = 89, phase: str = "prefill"):
    """A future narrow-capability row: an SM89 d64 kernel that serves ONLY
    unmasked graphs.  It must be declarable, and every mask must be declined by
    the probe -- the reason the canonical model exists (the legacy flags cannot
    say "no" about anything they do not mention)."""
    if kind == "fwd":
        return fwd_engines.EngineSpec(
            name="dummy_fwd_sm89_unmasked_only",
            capabilities=fwd_engines.Capabilities(
                sm_lo=sm_lo,
                sm_hi=sm_hi,
                phase=phase,
                d_shapes=frozenset({(64, 64)}),
                d_pad_multiple=8,
                dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
                band=band.BandSupport.unmasked_only(),
                decode=False,
                padded=False,
                skv_tile=0,
            ),
            lower=lambda *a, **k: None,
        )
    return bwd_engines.EngineSpec(
        name="dummy_bwd_sm89_unmasked_only",
        capabilities=bwd_engines.Capabilities(
            sm_lo=sm_lo,
            sm_hi=sm_hi,
            d=frozenset({64}),
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            band=band.BandSupport.unmasked_only(),
            decode=False,
        ),
        lower=lambda *a, **k: None,
    )


@pytest.mark.parametrize("kind", ["fwd", "bwd"])
def test_unmasked_only_row_is_expressible_and_declines_every_mask(kind):
    spec = _restricted_row(kind)
    caps = spec.capabilities
    # Stated positively: the claim exists without leaning on the legacy flags.
    assert caps.causal is False and caps.swa is False and caps.bottom_right is False and caps.right_band_widening is False
    assert caps.band == band.BandSupport.unmasked_only()
    assert caps.band.serves_anchor(band.BandFacts(anchor=band.ANCHOR_BOTTOM_RIGHT)) is False

    probe = _probe(kind)
    assert probe(caps, legal_facts(kind, caps, **band_unmasked())) is None
    assert probe(caps, legal_facts(kind, caps, s_kv=64, **band_causal_top_left())) == band.feature_reason(band.LABEL_RIGHT_CAUSAL)
    assert probe(caps, legal_facts(kind, caps, s_kv=64, **band_right(4))) == band.feature_reason(band.LABEL_RIGHT_FINITE)
    assert probe(caps, legal_facts(kind, caps, s_kv=64, **band_window(3))) == band.REASON_LEFT_WINDOW
    assert probe(caps, legal_facts(kind, caps, s_kv=64, **band_causal_bottom_right())) == band.feature_reason(band.LABEL_RIGHT_CAUSAL)
    # A causal-only row (the other restricted shape) is expressible as well.
    causal_only = dataclasses.replace(
        spec.capabilities,
        causal=True,
        band=band.BandSupport.causal_and_unmasked(),
    )
    assert causal_only.band.serves(band.BandFacts(right_mode=band.RIGHT_CAUSAL))
    assert probe(causal_only, legal_facts(kind, causal_only, s_kv=64, **band_causal_top_left())) is None
    assert probe(causal_only, legal_facts(kind, causal_only, s_kv=64, **band_right(4))) == band.feature_reason(band.LABEL_RIGHT_FINITE)
    assert probe(causal_only, legal_facts(kind, causal_only, s_kv=64, **band_window(3))) == band.REASON_LEFT_WINDOW


@pytest.mark.parametrize("kind", ["fwd", "bwd"])
def test_declaring_flags_and_band_together_is_explicit(kind):
    row_cls = fwd_engines.Capabilities if kind == "fwd" else bwd_engines.Capabilities
    kwargs = dict(sm_lo=89, sm_hi=89, phase="prefill", d_shapes=frozenset({(64, 64)})) if kind == "fwd" else dict(sm_lo=89, sm_hi=89, d=frozenset({64}))
    # Same claim in both spellings: accepted (this is what dataclasses.replace
    # round-trips a declared row through).
    same = row_cls(**kwargs, causal=True, band=band.BandSupport.causal_and_unmasked())
    assert same.band == band.BandSupport.causal_and_unmasked()
    # A flag OUTSIDE the declared band is a contradiction, and is refused by
    # naming both spellings.
    with pytest.raises(ValueError) as excinfo:
        row_cls(**kwargs, causal=True, band=band.BandSupport.unmasked_only())
    message = str(excinfo.value)
    assert "causal" in message and "never silently resolved" in message
    # A BAND-ONLY declaration needs no legacy flags at all -- including the
    # narrowest claims, which the flags cannot spell (from_legacy_flags always
    # serves the unbounded case as well).
    assert row_cls(**kwargs, band=band.BandSupport.causal_and_unmasked()).band == band.BandSupport.causal_and_unmasked()
    causal_only = band.BandSupport(right=frozenset({band.RIGHT_CAUSAL}))
    assert row_cls(**kwargs, band=causal_only).band == causal_only
    with pytest.raises(ValueError) as excinfo:
        row_cls(**kwargs, swa=True, band=causal_only)
    assert "swa" in str(excinfo.value) and "never silently resolved" in str(excinfo.value)


@pytest.mark.parametrize("kind", ["fwd", "bwd"])
def test_a_band_only_claim_is_enforced_on_every_axis(kind):
    """A restricted claim answers for the axes the graph does NOT flag too.

    Every legacy flag could only WIDEN a row and a missing flag meant "not
    requested", so nothing stopped a narrower claim from serving the case it
    does not mention -- a row that serves the causal diagonal alone would still
    accept an unmasked graph.  The band is a SET on both sides, so the matcher
    has to ask the mode the graph requests by leaving a flag unset: unbounded
    right band, no left bound, top-left anchor.
    """
    base = _restricted_row(kind).capabilities
    probe = _probe(kind)
    noun = "engine" if kind == "bwd" else "kernel"
    unmasked = band_unmasked()
    causal_top_left = band_causal_top_left()
    cases = (
        # claim, band it SERVES, band it must DECLINE, expected reason
        (
            band.BandSupport(right=frozenset({band.RIGHT_CAUSAL})),
            causal_top_left,
            unmasked,
            band.feature_reason(band.LABEL_RIGHT_UNBOUNDED),
        ),
        (
            band.BandSupport(left=frozenset({band.LEFT_WINDOW})),
            band_window(3),
            unmasked,
            band.feature_reason(band.LABEL_LEFT_NONE),
        ),
        (
            band.BandSupport(
                right=frozenset({band.RIGHT_UNBOUNDED, band.RIGHT_CAUSAL}),
                anchors=frozenset({band.ANCHOR_BOTTOM_RIGHT}),
            ),
            band_causal_bottom_right(),
            causal_top_left,
            band.anchor_reason(noun, anchor=band.ANCHOR_TOP_LEFT),
        ),
    )
    for claim, served, declined, expected in cases:
        row = dataclasses.replace(base, band=claim)
        assert row.band == claim
        assert probe(row, legal_facts(kind, row, s_kv=64, **served)) is None, (claim, served)
        sample = legal_facts(kind, row, s_kv=64, **declined)
        assert probe(row, sample) == expected, (claim, declined, probe(row, sample))
        # The one-call model decision is the same decision (its anchor wording
        # is the forward probe's "kernel"), not a second implementation.
        model_reason = row.band.decline(sample.band)
        assert model_reason in {expected, band.anchor_reason("kernel", anchor=band.ANCHOR_TOP_LEFT)}, (claim, model_reason)


# ---------------------------------------------------------------------------
# 3. geometry: S_q vs S_kv, and the band x tail rule
# ---------------------------------------------------------------------------


def test_geometry_relations_do_not_change_the_band_decision():
    # A causal top-left band is served whatever the Q/KV relation; the LENGTH
    # relation must not leak into the band decision.
    for kind, spec in _rows():
        caps = spec.capabilities
        expected = None if declares(caps, right=band.RIGHT_CAUSAL) else band.feature_reason(band.LABEL_RIGHT_CAUSAL)
        for s_q, s_kv in ((128, 128), (64, 128), (256, 128)):
            reason = _probe(kind)(caps, legal_facts(kind, caps, s_q=s_q, s_kv=s_kv, **band_causal_top_left()))
            assert reason == expected, (spec.name, s_q, s_kv, reason)
            # ... and the unmasked decision is geometry-independent too.
            assert _probe(kind)(caps, legal_facts(kind, caps, s_q=s_q, s_kv=s_kv, **band_unmasked())) is None, spec.name


def test_band_covers_kv_tail_is_the_band_x_geometry_rule():
    # top-left: the last unmasked column is (S_q - 1) + R, so the tail is
    # covered iff S_q + R <= S_kv.
    top_left_causal = band.BandFacts(right_mode=band.RIGHT_CAUSAL, right_bound=0)
    assert top_left_causal.covers_kv_tail(128, 128) is True
    assert top_left_causal.covers_kv_tail(64, 128) is True
    assert top_left_causal.covers_kv_tail(256, 128) is False
    widened = band.BandFacts(right_mode=band.RIGHT_FINITE, right_bound=4)
    assert widened.covers_kv_tail(124, 128) is True
    assert widened.covers_kv_tail(125, 128) is False
    # bottom-right: the diagonal is the KV end, so only a plain causal (R == 0)
    # covers the tail; a widened bottom-right band does not.
    bottom_right_causal = band.BandFacts(right_mode=band.RIGHT_CAUSAL, right_bound=0, anchor=band.ANCHOR_BOTTOM_RIGHT)
    assert bottom_right_causal.covers_kv_tail(999, 128) is True
    assert band.BandFacts(right_mode=band.RIGHT_FINITE, right_bound=1, anchor=band.ANCHOR_BOTTOM_RIGHT).covers_kv_tail(64, 128) is False
    # No diagonal at all -> never covered, however the lengths line up.
    assert band.BandFacts().covers_kv_tail(1, 4096) is False
    assert band.BandFacts(left_mode=band.LEFT_WINDOW, left_window=7).covers_kv_tail(1, 4096) is False
    # The probe's KV-tail rule agrees with the model on the rows where it is live.
    live = [
        (kind, spec) for kind, spec in _rows() if getattr(spec.capabilities, "skv_tile", 0) and not getattr(spec.capabilities, "skv_tail_via_padding", False)
    ]
    assert live, "no row exercises the KV-tail rule any more; this test would be vacuous"
    for kind, spec in live:
        caps = spec.capabilities
        tile = int(caps.skv_tile)
        s_kv = tile * 2 - 1  # not a tile multiple: the tail rule is live
        uncovered = legal_facts(kind, caps, s_q=tile * 2, s_kv=s_kv, **band_causal_top_left())
        covered = legal_facts(kind, caps, s_q=2, s_kv=s_kv, **band_causal_top_left())
        reason = _probe(kind)(caps, uncovered)
        assert reason is not None and "must be a multiple of" in reason, (spec.name, reason)
        assert _probe(kind)(caps, covered) is None, spec.name


# ---------------------------------------------------------------------------
# 4. right bound: default / causal / legal positive / illegal
# ---------------------------------------------------------------------------


def test_right_bound_modes_follow_the_declared_claim():
    for kind, spec in _rows():
        caps = spec.capabilities
        for facts, right, label in (
            (band_unmasked(), band.RIGHT_UNBOUNDED, None),
            (band_causal_top_left(), band.RIGHT_CAUSAL, band.feature_reason(band.LABEL_RIGHT_CAUSAL)),
            (band_right(1), band.RIGHT_FINITE, band.feature_reason(band.LABEL_RIGHT_FINITE)),
            (band_right(4), band.RIGHT_FINITE, band.feature_reason(band.LABEL_RIGHT_FINITE)),
            (band_right(100000), band.RIGHT_FINITE, band.feature_reason(band.LABEL_RIGHT_FINITE)),
        ):
            sample = legal_facts(kind, caps, s_kv=256, **facts)
            model_reason = caps.band.decline(sample.band)
            probe_reason = _probe(kind)(caps, sample)
            assert model_reason == probe_reason, (spec.name, facts, model_reason, probe_reason)
            expected = None if declares(caps, right=right) else label
            assert probe_reason == expected, (spec.name, facts, probe_reason)


def test_negative_right_bound_is_still_refused_by_the_forward_probe():
    # Illegal per the API contract (the analyzer flags it before any engine
    # runs, which is why this is only reachable through a hand-built fact set --
    # the branch is still owned by the probe and must keep its message).
    for kind, spec in _rows():
        caps = spec.capabilities
        reason = _probe(kind)(caps, legal_facts(kind, caps, s_kv=128, **band_right(-1)))
        if kind == "fwd":
            # The forward probe asks the row's right-mode claim BEFORE it can
            # look at the value, so only a row that serves widening reaches the
            # negative-bound guard; one that does not reports the band.
            assert reason == (
                "negative diagonal_band_right_bound (-1) is not supported" if caps.right_band_widening else band.feature_reason(band.LABEL_RIGHT_FINITE)
            ), spec.name
        else:
            # The backward probe has no such early value guard: the row's band
            # claim decides, so a row that serves widening still serves -1 (the
            # analyzer rejects the value before any engine runs).
            assert reason == (None if caps.right_band_widening else band.feature_reason(band.LABEL_RIGHT_FINITE)), spec.name


# ---------------------------------------------------------------------------
# 5. left bound: default / length 1 / small / larger than the sequence
# ---------------------------------------------------------------------------


def test_left_bound_presence_and_offsets():
    for kind, spec in _rows():
        caps = spec.capabilities
        for window_left in (None, 0, 1, 3, 100000):
            facts = band_unmasked() if window_left is None else band_window(window_left)
            sample = legal_facts(kind, caps, s_kv=256, **facts)
            if window_left is None:
                expected = None
            else:
                expected = None if declares(caps, right=band.RIGHT_UNBOUNDED, left_window=True) else band.REASON_LEFT_WINDOW
            assert _probe(kind)(caps, sample) == expected, (spec.name, window_left)
            if window_left is not None:
                # window_left is the OFFSET: the API's length is offset + 1, so
                # length 1 (the current token only) is offset 0 -- and 0 is NOT
                # the same request as "no left bound".
                assert sample.band.left_window_length == window_left + 1, (spec.name, window_left)
                assert sample.band.has_left_window is True
    assert band.BandFacts.from_sdpa_facts(ga.SdpaGraphFacts(window_left=0)).left_mode == band.LEFT_WINDOW
    assert band.BandFacts.from_sdpa_facts(ga.SdpaGraphFacts(window_left=None)).left_mode == band.LEFT_NONE


# ---------------------------------------------------------------------------
# 6. anchors: top-left and bottom-right (+ the coherence constraint)
# ---------------------------------------------------------------------------


def test_bottom_right_anchor_needs_a_diagonal_and_a_claim():
    for kind, spec in _rows():
        caps = spec.capabilities
        # Coherence: an anchor is a property OF a diagonal, so a bottom-right
        # request with no right bound (only a left window) is incoherent for
        # EVERY row, claiming or not -- the combination constraint of 4.3.  It
        # is reported only when the left bound itself is served: the probe (and
        # the model) ask the axes in order, so a row that declines SWA reports
        # that first.
        incoherent = legal_facts(kind, caps, s_kv=256, **band_window(3, bottom_right=True))
        expected = band.REASON_ANCHOR_WITHOUT_DIAGONAL if caps.swa else band.REASON_LEFT_WINDOW
        assert _probe(kind)(caps, incoherent) == expected, spec.name
        assert caps.band.decline(incoherent.band) == expected, spec.name
        # The constraint cannot be derived from axis membership: this band asks
        # for exactly the unbounded right mode and the bottom-right anchor, both
        # of which a full-mask row serves, and it is still declined.
        anchor_without_diagonal = band.BandFacts(right_mode=band.RIGHT_UNBOUNDED, anchor=band.ANCHOR_BOTTOM_RIGHT)
        assert caps.band.serves_right_mode(band.RIGHT_UNBOUNDED)
        assert caps.band.serves_anchor(anchor_without_diagonal) == bool(caps.bottom_right)
        sample = legal_facts(kind, caps, s_kv=256, causal=False, right_band_widening=False, right_bound=None, bottom_right=True, window_left=None)
        assert _probe(kind)(caps, sample) == band.REASON_ANCHOR_WITHOUT_DIAGONAL, spec.name
        assert caps.band.decline(sample.band) == band.REASON_ANCHOR_WITHOUT_DIAGONAL, spec.name
        # Bottom-right WITH a diagonal: decided by the row's claim, and the
        # RIGHT AXIS is still asked first -- a row that declines the band
        # reports the band, not the anchor.
        anchor_noun = "engine" if kind == "bwd" else "kernel"
        for facts, right, axis_label in (
            (band_causal_bottom_right(), band.RIGHT_CAUSAL, band.LABEL_RIGHT_CAUSAL),
            (band_right(4, bottom_right=True), band.RIGHT_FINITE, band.LABEL_RIGHT_FINITE),
        ):
            sample = legal_facts(kind, caps, s_kv=256, **facts)
            if not declares(caps, right=right):
                expected = band.feature_reason(axis_label)
            elif not declares(caps, bottom_right=True):
                expected = band.anchor_reason(anchor_noun)
            else:
                expected = None
            assert _probe(kind)(caps, sample) == expected, (spec.name, facts, _probe(kind)(caps, sample))
        # Left window + causal diagonal + bottom-right: the SWA axis and the
        # anchor are independent.
        combined = legal_facts(kind, caps, s_kv=256, **band_window(3, causal=True, bottom_right=True))
        if declares(caps, right=band.RIGHT_CAUSAL, left_window=True, bottom_right=True):
            assert _probe(kind)(caps, combined) is None, spec.name
        else:
            assert _probe(kind)(caps, combined) is not None, spec.name


# ---------------------------------------------------------------------------
# 7. length metadata: no padding / per-batch padding / Q-side trim
# ---------------------------------------------------------------------------


def test_length_metadata_decisions_follow_the_row_claims():
    for kind, spec in _rows():
        caps = spec.capabilities
        padding_reason = "graph uses padding mask, which this engine does not support"
        trim_reason = "graph uses seq_len_q without padding mask, which this engine does not support"
        claims_padding = bool(getattr(caps, "padded", False))
        cases = (
            (dict(padded=True), None if claims_padding else padding_reason),
            (dict(seq_q_trim=True), None if getattr(caps, "seq_q_trim", False) else trim_reason),
        )
        for overrides, expected in cases:
            sample = legal_facts(kind, caps, s_kv=128, **{**band_causal_top_left(), **overrides})
            assert _probe(kind)(caps, sample) == expected, (spec.name, overrides, _probe(kind)(caps, sample))
        if kind == "fwd":
            # fwd-only combination: a padding mask plus generate_stats needs the
            # row's padded_stats claim; the padding gate itself comes first.
            sample = legal_facts(kind, caps, s_kv=128, **{**band_causal_top_left(), "padded": True, "wants_stats": True})
            if not claims_padding:
                expected = padding_reason
            elif getattr(caps, "padded_stats", False):
                expected = None
            else:
                expected = "padding mask with generate_stats is not supported by this kernel"
            assert _probe(kind)(caps, sample) == expected, (spec.name, _probe(kind)(caps, sample))


# ---------------------------------------------------------------------------
# 8. layouts: legal dense forms and illegal negatives
# ---------------------------------------------------------------------------


def test_dense_layout_envelope_decisions():
    for kind, spec in _rows():
        caps = spec.capabilities
        layouts = getattr(caps, "layouts", frozenset({"bshd"}))
        cases = (
            (dict(bshd_layout=True, packed_layout=True, dense_layout=True), True),
            (dict(bshd_layout=False, packed_layout=True, dense_layout=True), "dense_flex" in layouts),
            (dict(bshd_layout=False, packed_layout=True, dense_layout=False), False),
        )
        for layout, ok in cases:
            sample = legal_facts(kind, caps, s_kv=128, **{**band_causal_top_left(), **layout})
            reason = _probe(kind)(caps, sample)
            assert (reason is None) == ok, (spec.name, layout, reason)
            if not ok:
                assert "BSHD-physical" in reason or "innermost-contiguous" in reason, (spec.name, reason)


# ---------------------------------------------------------------------------
# 9. construction: positional prefix, keywords, replace, conflicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engines_module", [fwd_engines, bwd_engines])
def test_positional_construction_in_the_legacy_order_still_binds(engines_module):
    fields = {f.name: f for f in dataclasses.fields(engines_module.Capabilities)}
    names = [n for n in fields if n != "band"]
    required = {"sm_lo": 89, "sm_hi": 89, "phase": "prefill", "d_shapes": frozenset({(64, 64)}), "d": frozenset({64})}

    def value(name):
        f = fields[name]
        if name in ("causal", "swa", "bottom_right", "right_band_widening"):
            return True
        if f.default is not dataclasses.MISSING:
            return f.default
        if f.default_factory is not dataclasses.MISSING:
            return f.default_factory()
        return required[name]

    caps = engines_module.Capabilities(*[value(n) for n in names])
    # The legacy positional order still binds every field ...
    assert caps.sm_lo == 89 and caps.causal is True and caps.swa is True
    # ... and the appended band set is DERIVED from those flags.
    assert caps.band == band.BandSupport.full_masks()


@pytest.mark.parametrize("engines_module", [fwd_engines, bwd_engines])
def test_keyword_construction_and_dataclasses_replace(engines_module):
    kwargs = (
        dict(sm_lo=89, sm_hi=89, d=frozenset({64}))
        if engines_module is bwd_engines
        else dict(sm_lo=89, sm_hi=89, phase="prefill", d_shapes=frozenset({(64, 64)}))
    )
    restricted = engines_module.Capabilities(**kwargs, band=band.BandSupport.unmasked_only())
    assert restricted.band == band.BandSupport.unmasked_only()
    replaced = dataclasses.replace(restricted, sm_hi=90)
    assert replaced.sm_hi == 90 and replaced.band == restricted.band
    # Replacing an UNRELATED field of a flag-declared row keeps its claim.
    flagged = engines_module.Capabilities(**kwargs, causal=True, swa=True)
    assert dataclasses.replace(flagged, sm_hi=90).band == flagged.band
    # CLEARING a flag of a declared row cannot contradict the copied claim:
    # an all-default legacy block is the spelling of "nothing declared".
    cleared = dataclasses.replace(flagged, causal=False)
    assert cleared.band == flagged.band and cleared.band.serves_right_mode(band.RIGHT_CAUSAL)
    # WIDENING a flag past the copied claim does contradict it, and must raise
    # rather than silently re-deriving.
    with pytest.raises(ValueError):
        dataclasses.replace(flagged, right_band_widening=True)


# ---------------------------------------------------------------------------
# 10. non-mask rejections are untouched by the band migration
# ---------------------------------------------------------------------------


def test_non_mask_rejections_keep_their_own_vocabulary():
    for kind, spec in _rows():
        caps = spec.capabilities
        for overrides, cap_attr, label in (
            (dict(has_bias=True), "bias", "bias"),
            (dict(has_dropout=True), "dropout", "dropout"),
            (dict(has_block_mask=True), "block_mask", "block_mask"),
        ):
            if getattr(caps, cap_attr, False):
                continue
            reason = _probe(kind)(caps, legal_facts(kind, caps, s_kv=128, **{**band_unmasked(), **overrides}))
            assert reason == f"graph uses {label}, which this engine does not support", (spec.name, label, reason)
        # The band model is not consulted for a wrong-pass probe.
        other = bwd_engines.mismatch if kind == "fwd" else fwd_engines.mismatch
        assert "serves sdpa" in other(caps, legal_facts(kind, caps, s_kv=128, **band_unmasked()))


def test_arch_gate_precedes_the_band_decision():
    for kind, spec in _rows():
        caps = spec.capabilities
        reason = _probe(kind)(caps, legal_facts(kind, caps, s_kv=128, device_cc=(5, 0), **band_causal_top_left()))
        assert reason is not None and reason.startswith("requires SM"), (spec.name, reason)


# ---------------------------------------------------------------------------
# 11. the model's one-call decision agrees with the spliced probe decision
# ---------------------------------------------------------------------------


def test_model_and_probe_agree_on_every_band_sample():
    samples = (
        band_unmasked(),
        band_causal_top_left(),
        band_causal_bottom_right(),
        band_right(4),
        band_right(4, bottom_right=True),
        band_window(0),
        band_window(3),
        band_window(100000),
        band_window(3, causal=True),
        band_window(3, causal=True, bottom_right=True),
        band_window(3, bottom_right=True),
        dict(causal=False, right_band_widening=False, right_bound=None, bottom_right=True, window_left=None),
    )
    for kind, spec in _rows():
        caps = spec.capabilities
        for sample in samples:
            facts = legal_facts(kind, caps, s_kv=256, **sample)
            model_reason = caps.band.decline(facts.band)
            probe_reason = _probe(kind)(caps, facts)
            assert (model_reason is None) == (probe_reason is None), (spec.name, sample, model_reason, probe_reason)
            if model_reason is not None and probe_reason is not None and model_reason != probe_reason:
                # The one wording that legitimately differs per pass: the
                # canonical model says "kernel" (the forward probe's), the
                # backward probe says "engine".
                assert {model_reason, probe_reason} == {band.anchor_reason("kernel"), band.anchor_reason("engine")}, (
                    spec.name,
                    sample,
                    model_reason,
                    probe_reason,
                )
