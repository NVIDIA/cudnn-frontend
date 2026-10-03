# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model inputs follow candidate geometry; no timing/rank golden assertions."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from frost_test_utils import requires_dsl

import cudnn
from cudnn.sdpa.fwd import heuristics as heur
from cudnn.sdpa.fwd.config_sm100 import CfgD512, cga_ctas
from cudnn.sdpa.fwd.engines import ENGINE_SPECS, mismatch
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPEC = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")


def _facts(**kw):
    base = dict(b=1, h_q=32, h_kv=4, s_q=33, s_kv=512, d_qk=128, d_v=128, dtype=cudnn.data_type.BFLOAT16, device_cc=(10, 0), device_sm_count=148)
    base.update(kw)
    return SdpaGraphFacts(**base)


def _visible_tile_oracle(facts, token_span, tile_n):
    # Enumerate actual key indices independently of the optimized floor/ceil
    # formula. Include the full last cluster's padded Q span: the kernel's
    # loop bounds do too, even though its output masks the padded Q rows.
    diagonal = facts.s_kv - facts.s_q if facts.bottom_right else 0
    causal = facts.causal or facts.right_band_widening
    right = facts.right_bound if facts.right_band_widening else 0
    largest = 0
    for start in range(0, facts.s_q, token_span):
        kept = {
            k // tile_n
            for k in range(facts.s_kv)
            if k >= start + diagonal - facts.window_left and (not causal or k <= start + token_span - 1 + diagonal + right)
        }
        largest = max(largest, len(kept))
    return largest


@pytest.mark.parametrize("bottom_right", [False, True])
@pytest.mark.parametrize("s_q,s_kv", [(1, 256), (17, 511), (129, 256), (385, 512), (513, 257)])
@pytest.mark.parametrize("token_span", [1, 16, 64, 128, 256, 512])
@pytest.mark.parametrize("window,right", [(0, 0), (95, 0), (127, 0), (160, 31), (1024, 0)])
def test_swa_work_matches_visible_tile_oracle(bottom_right, s_q, s_kv, token_span, window, right):
    facts = _facts(s_q=s_q, s_kv=s_kv, causal=True, bottom_right=bottom_right, window_left=window, right_band_widening=right > 0, right_bound=right)
    assert heur._swa_kv_tiles(facts, token_span=token_span, tile_n=128) == _visible_tile_oracle(facts, token_span, 128)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("token_span", [16, 64, 256])
def test_paged_bottom_right_work_bounds_unknown_device_lengths(causal, token_span):
    declared = _facts(s_q=129, s_kv=511, causal=causal, bottom_right=True, window_left=95, padded=True, has_paged_kv=True)
    bound = heur._swa_kv_tiles(declared, token_span=token_span, tile_n=128)
    for actual_kv in range(1, 512):
        actual = replace(declared, s_kv=actual_kv)
        assert _visible_tile_oracle(actual, token_span, 128) <= bound


def test_swa_large_q_uses_bounded_residue_work(monkeypatch):
    # Count the iteration domain rather than timing a unit test.
    real_range, sizes = range, []

    def recording_range(stop):
        sizes.append(stop)
        return real_range(stop)

    monkeypatch.setattr(heur, "range", recording_range, raising=False)
    facts = _facts(s_q=1 << 30, s_kv=1 << 30, causal=True, window_left=127)
    assert heur._swa_kv_tiles(facts, token_span=1, tile_n=128) <= 2
    assert sizes == [128]


@pytest.mark.parametrize("dim,cga,physical", [(128, 1, 1), (128, 2, 2), (192, 1, 1), (192, 2, 2), (256, 2, 2), (512, 2, 4)])
def test_public_mma_width_is_not_always_physical_cta_count(dim, cga, physical):
    assert cga_ctas(dim, cga) == physical


def test_d512_model_counts_all_roles_in_split_and_unsplit(monkeypatch):
    seen = []
    real = heur.choose_split_kv

    def recording(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(heur, "choose_split_kv", recording)
    plans = heur.recommend("A", _facts(h_kv=1, s_q=1, s_kv=131072, d_qk=512, d_v=512), {SPEC.name: 20500})
    assert plans and seen
    # D512 is scored twice per leg: with the physical 4-CTA cluster and with the
    # MMA width the constants were fitted with; the physical count may only
    # LOWER the split (the increases it asks for are unmeasured).
    counts = {kw["ctas_per_tile"] for kw in seen}
    assert counts == {CfgD512.CGA_M, CfgD512.CTA_MMA}, counts
    for kw in seen:
        assert kw["unsplit_launch"].ctas_per_tile == kw["ctas_per_tile"]
    physical = [kw for kw in seen if kw["ctas_per_tile"] == CfgD512.CGA_M]
    fitted = [kw for kw in seen if kw["ctas_per_tile"] == CfgD512.CTA_MMA]
    assert plans[0].knobs.split_kv == min(heur.choose_split_kv(**physical[0]), heur.choose_split_kv(**fitted[0]))
    # Every scoring leg still uses the graph's output, not packed head count.
    assert all(kw["combine_rows"] == 32 for kw in seen)


def test_each_geometry_runner_recomputes_split_and_obeys_cap(monkeypatch):
    seen = []
    real = heur._split_points

    def recording(caps, facts, tile_m, tile_n, cga, pack_g=1, *, unsplit_knobs=None):
        seen.append((tile_m, tile_n, cga, pack_g))
        return real(caps, facts, tile_m, tile_n, cga, pack_g, unsplit_knobs=unsplit_knobs)

    monkeypatch.setattr(heur, "_split_points", recording)
    facts = _facts(h_q=32, h_kv=4, s_q=17, s_kv=8192)
    plans = heur.recommend("A", facts, {SPEC.name: 20500})
    knobs = [p.knobs for p in plans]
    assert len(knobs) == len(set(knobs)) <= heur._MAX_SETS_PER_ENGINE
    assert {k.pack_gqa for k in knobs} == {False, True}
    # The width follows each leg's own rows (decode-tile fit): 17 * 8 packed
    # rows overflow one 128-row tile (cga 2); 17 unpacked rows fit (cga 1).
    assert {k.cga for k in knobs if k.pack_gqa} == {2} and {k.cga for k in knobs if not k.pack_gqa} == {1}
    for k in knobs:
        assert mismatch(SPEC.capabilities, facts, k) is None
        group = heur._pack_gqa_group(SPEC.capabilities, facts, k.tile_m, k.pack_gqa)
        assert (k.tile_m, k.tile_n, k.cga, group) in seen
        if k.split_kv > 1:
            assert k.sched_policy == 0


def test_swa_split_model_receives_live_cluster_work(monkeypatch):
    seen = []
    real = heur.choose_split_kv

    def recording(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(heur, "choose_split_kv", recording)
    facts = _facts(s_q=256, s_kv=8192, causal=True, bottom_right=True, window_left=127)
    # Pin a candidate's geometry, not the performance policy's preferred one.
    knobs = heur.SdpaFwdKnobs(tile_m=128, tile_n=128, cga=2, pack_gqa=True, split_kv=1, sched_policy=0)
    heur._split_points(SPEC.capabilities, facts, knobs.tile_m, knobs.tile_n, knobs.cga, pack_g=8, unsplit_knobs=knobs)
    assert len(seen) == 1
    assert seen[0]["kv_tiles"] == 2  # full 64-token packed cluster plus its aligned band
    assert seen[0]["unsplit_launch"].kv_tiles == 2
    assert seen[0]["q_tiles"] == 4 and seen[0]["heads_q"] == 4


def _paged_split_facts(**overrides):
    pool = SimpleNamespace(get_stride=lambda: [4096, 2048, 128, 1])
    base = dict(h_q=8, h_kv=2, s_q=128, s_kv=8192, thd=True, padded=True, has_paged_kv=True, page_size=16, causal=True, bottom_right=True, k_t=pool)
    base.update(overrides)
    return _facts(**base)


@requires_dsl
@pytest.mark.parametrize("splits", [3, 4, 6, 7, 12])
def test_paged_split_record_and_older_native_extension_fallback(monkeypatch, splits):
    facts = _paged_split_facts()
    knobs = heur.SdpaFwdKnobs(cga=1, split_kv=splits, pack_gqa=False)
    assert mismatch(SPEC.capabilities, facts, knobs) is None
    assert heur.SdpaFwdKnobs.from_public({int(k): v for k, v in knobs.to_public().items()}) == knobs
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", type("PreviousNativeBinder", (), {}))
    assert "matching native" in mismatch(SPEC.capabilities, facts, knobs)
    candidates = heur._knob_sets(SPEC, facts)
    assert candidates and all(k.split_kv in (None, 1) for k in candidates)
    fallback = heur._fallback_knobs(SPEC, facts)
    assert fallback.split_kv == 1 and mismatch(SPEC.capabilities, facts, fallback) is None


@requires_dsl
@pytest.mark.parametrize("packed", [False, True])
def test_paged_split_proposal_preserves_selected_packing(monkeypatch, packed):
    """Transport the measured choice without asserting a performance ranking."""
    from cudnn.sdpa.fwd import placement

    facts = _paged_split_facts()
    monkeypatch.setattr(heur, "paged_thd_split_choice", lambda caps, facts: (3, packed), raising=False)
    selected = heur._knob_sets(SPEC, facts)[0]
    assert (selected.cga, selected.split_kv, selected.pack_gqa) == (1, 3, packed)
    assert mismatch(SPEC.capabilities, facts, selected) is None
    assert placement._place_sm100_f16(SPEC.capabilities, facts) == placement.LEAD


def _mla_split_facts(**overrides):
    base = dict(h_q=4, h_kv=4, s_q=129, s_kv=4097, d_qk=192, d_v=128, thd=True, padded=True)
    base.update(overrides)
    return _facts(**base)


@requires_dsl
@pytest.mark.parametrize("wants_stats", [False, True])
def test_mla_split_choice_transport_and_native_fallback(monkeypatch, wants_stats):
    """A supplied choice drives both proposals and placement, not a timing golden."""
    from cudnn.sdpa.fwd import placement

    facts = _mla_split_facts(wants_stats=wants_stats)
    with monkeypatch.context() as m:
        m.setattr(heur, "mla_thd_split_choice", lambda caps, facts: 3)
        selected = heur._knob_sets(SPEC, facts)[0]
        assert (selected.cga, selected.split_kv, selected.pack_gqa) == (1, 3, False)
        assert mismatch(SPEC.capabilities, facts, selected) is None
        assert placement._place_sm100_f16(SPEC.capabilities, facts) == placement.LEAD
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", type("PreviousNativeBinder", (), {}))
    assert heur.mla_thd_split_choice(SPEC.capabilities, facts) == 1
    assert all(k.split_kv in (None, 1) for k in heur._knob_sets(SPEC, facts))
    assert placement._place_sm100_f16(SPEC.capabilities, facts) == placement.TRAIL


@requires_dsl
@pytest.mark.parametrize("batch,heads,q,kv", [(1, 4, 64, 32768), (3, 4, 129, 4097), (4, 8, 128, 8192), (1, 64, 512, 8192)])
@pytest.mark.parametrize("sm_count", [0, 64, 148])
def test_mla_split_choice_obeys_physical_launch_bounds(batch, heads, q, kv, sm_count):
    facts = _mla_split_facts(b=batch, h_q=heads, h_kv=heads, s_q=q, s_kv=kv, device_sm_count=sm_count)
    splits = heur.mla_thd_split_choice(SPEC.capabilities, facts)
    if splits > 1:
        # Count real 128-row CTAs, including a partial Q tile in every batch.
        assert batch * heads * len(range(0, q, 128)) * splits <= sm_count
        assert len(range(0, kv, 128)) // splits >= 4
        assert mismatch(SPEC.capabilities, facts, heur.SdpaFwdKnobs(cga=1, split_kv=splits, pack_gqa=False)) is None
    elif not sm_count:
        assert splits == 1


@requires_dsl
@pytest.mark.parametrize("capacity", [None, 0, 64, 128, 129])
@pytest.mark.parametrize("paged,d", [(True, 128), (False, 192)])
def test_packed_split_override_requires_bounded_workspace(capacity, paged, d):
    facts = _paged_split_facts(shape_overrides=True, max_total_seq_len_q=capacity, has_paged_kv=paged, d_qk=d)
    knobs = heur.SdpaFwdKnobs(cga=1, split_kv=4, pack_gqa=False)
    assert (mismatch(SPEC.capabilities, facts, knobs) is None) == (capacity in (64, 128))


@requires_dsl
@pytest.mark.parametrize("overrides", [{"device_cc": (10, 3)}, {"device_cc": (10, 7)}, {"d_qk": 64, "d_v": 64}, {"has_paged_kv": False}, {"has_sink": True}])
def test_paged_split_public_request_declines_unsupported_geometry(overrides):
    facts = _paged_split_facts(**overrides)
    assert mismatch(SPEC.capabilities, facts, heur.SdpaFwdKnobs(cga=1, split_kv=4, pack_gqa=False)) is not None


@requires_dsl
@pytest.mark.parametrize("batch,heads,q,kv", [(1, 4, 128, 4096), (4, 8, 128, 8192), (1, 16, 512, 32768)])
def test_mla_bounded_overrides_use_declared_geometry(batch, heads, q, kv):
    # An explicit upper bound permits the same plan-time policy as the fixed
    # declaration. Do not pin a split count or a winning engine.
    args = dict(b=batch, h_q=heads, h_kv=heads, s_q=q, s_kv=kv, device_sm_count=148)
    exact = _mla_split_facts(**args)
    bounded = _mla_split_facts(**args, shape_overrides=True, max_total_seq_len_q=batch * q)
    assert heur.mla_thd_split_choice(SPEC.capabilities, bounded) == heur.mla_thd_split_choice(SPEC.capabilities, exact)
    unbounded = _mla_split_facts(**args, shape_overrides=True, max_total_seq_len_q=None)
    assert heur.mla_thd_split_choice(SPEC.capabilities, unbounded) == 1
