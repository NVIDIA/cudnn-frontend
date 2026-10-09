# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model inputs follow candidate geometry; no timing/rank golden assertions."""

from dataclasses import replace
from itertools import product
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
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3)])
def test_paged_split_record_and_older_native_extension_fallback(monkeypatch, splits, device_cc):
    facts = _paged_split_facts(device_cc=device_cc)
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
@pytest.mark.parametrize("batch,h_q,h_kv", [(1, 8, 2), (2, 8, 8), (4, 32, 4)])
@pytest.mark.parametrize("wants_stats", [False, True])
@pytest.mark.parametrize("d,device_cc,cga,packed", [(128, (10, 0), 1, False), (128, (10, 0), 1, True), (256, (10, 7), 2, False)])
def test_paged_split_proposal_preserves_selected_packing(monkeypatch, packed, batch, h_q, h_kv, d, device_cc, cga, wants_stats):
    """Transport the measured choice without asserting a performance ranking."""
    from cudnn.sdpa.fwd import placement

    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107") if d == 256 else SPEC
    facts = _paged_split_facts(b=batch, h_q=h_q, h_kv=h_kv, d_qk=d, d_v=d, device_cc=device_cc, wants_stats=wants_stats)
    monkeypatch.setattr(heur, "paged_thd_split_choice", lambda caps, facts: (3, packed), raising=False)
    selected = heur._knob_sets(spec, facts)[0]
    assert (selected.cga, selected.split_kv, selected.pack_gqa) == (cga, 3, packed)
    assert mismatch(spec.capabilities, facts, selected) is None
    assert placement.place(spec, facts) == placement.LEAD


def _mla_split_facts(**overrides):
    base = dict(h_q=4, h_kv=4, s_q=129, s_kv=4097, d_qk=192, d_v=128, thd=True, padded=True)
    base.update(overrides)
    return _facts(**base)


@requires_dsl
@pytest.mark.parametrize("wants_stats", [False, True])
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("d,packed", [(128, False), (128, True), (192, False)])
def test_nonpaged_split_choice_transport_and_native_fallback(monkeypatch, wants_stats, device_cc, d, packed):
    """A supplied choice drives both proposals and placement, not a timing golden."""
    from cudnn.sdpa.fwd import placement

    facts = _mla_split_facts(wants_stats=wants_stats, device_cc=device_cc, d_qk=d, **({"h_q": 8, "h_kv": 2} if packed else {}))
    with monkeypatch.context() as m:
        m.setattr(heur, "nonpaged_thd_split_choice", lambda caps, facts: (3, packed))
        selected = heur._knob_sets(SPEC, facts)[0]
        assert (selected.cga, selected.split_kv, selected.pack_gqa) == (1, 3, packed)
        assert mismatch(SPEC.capabilities, facts, selected) is None
        assert placement._place_sm100_f16(SPEC.capabilities, facts) == placement.LEAD
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", type("PreviousNativeBinder", (), {}))
    assert heur.nonpaged_thd_split_choice(SPEC.capabilities, facts) == (1, False)
    assert all(k.split_kv in (None, 1) for k in heur._knob_sets(SPEC, facts))
    assert placement._place_sm100_f16(SPEC.capabilities, facts) == placement.TRAIL


@requires_dsl
@pytest.mark.parametrize(
    "batch,heads,q,kv", [(1, 4, 64, 32768), (3, 4, 129, 4097), (4, 8, 128, 8192), (1, 64, 512, 8192), (1, 64, 69, 8192), (2, 16, 8, 8192), (4, 32, 37, 16384)]
)
@pytest.mark.parametrize("sm_count", [0, 64, 148])
@pytest.mark.parametrize("d,group", [(128, 1), (128, 4), (128, 16), (192, 1)])
@pytest.mark.parametrize("total", [None, "sum"])
def test_nonpaged_split_choice_obeys_physical_launch_bounds(batch, heads, q, kv, sm_count, d, group, total):
    h_kv = max(1, heads // group)
    # A declared token total below b * s_q: the batch is ragged.
    max_total = batch * q // 2 + 1 if total else None
    facts = _mla_split_facts(b=batch, h_q=heads, h_kv=h_kv, s_q=q, s_kv=kv, device_sm_count=sm_count, d_qk=d, max_total_seq_len_q=max_total)
    splits, packed = heur.nonpaged_thd_split_choice(SPEC.capabilities, facts)
    if splits > 1:
        # Count physical CTAs: 128 // group packed tokens of one KV head's group, or 128 rows of one Q head,
        # with at most one partial tile per sequence.
        g = heads // h_kv if packed else 1
        tokens = max_total if max_total is not None and d == 128 else batch * q
        tiles = min(batch * len(range(0, q, 128 // g)), (tokens + batch * (128 // g - 1)) // (128 // g))
        assert tiles * (heads // g) * splits <= sm_count * (2 if d == 128 else 1)
        assert len(range(0, kv, 128)) // splits >= 4
        assert mismatch(SPEC.capabilities, facts, heur.SdpaFwdKnobs(cga=1, split_kv=splits, pack_gqa=packed)) is None
    elif not sm_count:
        assert (splits, packed) == (1, False)


@requires_dsl
@pytest.mark.parametrize("capacity", [None, 0, 64, 128, 129])
@pytest.mark.parametrize("paged,d", [(True, 128), (False, 192), (False, 128)])
def test_packed_split_override_requires_bounded_workspace(capacity, paged, d):
    facts = _paged_split_facts(shape_overrides=True, max_total_seq_len_q=capacity, has_paged_kv=paged, d_qk=d)
    knobs = heur.SdpaFwdKnobs(cga=1, split_kv=4, pack_gqa=False)
    assert (mismatch(SPEC.capabilities, facts, knobs) is None) == (capacity in (64, 128))


@requires_dsl
@pytest.mark.parametrize("overrides", [{"device_cc": (10, 7)}, {"d_qk": 64, "d_v": 128}, {"has_paged_kv": False, "d_qk": 256, "d_v": 256}, {"has_sink": True}])
def test_paged_split_public_request_declines_unsupported_geometry(overrides):
    facts = _paged_split_facts(**overrides)
    assert mismatch(SPEC.capabilities, facts, heur.SdpaFwdKnobs(cga=1, split_kv=4, pack_gqa=False)) is not None


@requires_dsl
@pytest.mark.parametrize("wants_stats", [False, True])
@pytest.mark.parametrize("paged,d,arch", [(False, 128, "sm100"), (False, 192, "sm100"), (True, 128, "sm100"), (True, 128, "sm107")])
@pytest.mark.parametrize("batch,heads,q,kv", [(1, 4, 128, 4096), (4, 8, 128, 8192), (1, 16, 512, 32768)])
def test_bounded_overrides_use_declared_geometry(batch, heads, q, kv, paged, d, arch, wants_stats, monkeypatch):
    # An explicit upper bound permits the same plan-time policy as the fixed
    # declaration. Do not pin a split count or a winning engine.
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_" + arch)
    args = dict(
        b=batch,
        h_q=heads,
        h_kv=heads // 4 if paged else heads,
        s_q=q,
        s_kv=kv,
        device_sm_count=148,
        device_cc=(10, 7) if arch == "sm107" else (10, 0),
        d_qk=d,
        wants_stats=wants_stats,
    )
    facts = _paged_split_facts if paged else _mla_split_facts
    choose = heur.paged_thd_split_choice if paged else heur.nonpaged_thd_split_choice
    exact = facts(**args)
    bounded = facts(**args, shape_overrides=True, max_total_seq_len_q=batch * q)
    assert choose(spec.capabilities, bounded) == choose(spec.capabilities, exact)
    unbounded = facts(**args, shape_overrides=True, max_total_seq_len_q=None)
    assert choose(spec.capabilities, unbounded) == (1, False)


@requires_dsl
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("pack_gqa", [False, True])
def test_nonpaged_d128_split_explicit_contract(device_cc, pack_gqa, monkeypatch):
    """New packed geometry is legal only with its matching native binder."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    spec = next(s for s in ENGINE_SPECS if s.name == ("sdpa_fwd_prefill_sm107" if device_cc == (10, 7) else "sdpa_fwd_prefill_sm100"))
    facts = _mla_split_facts(d_qk=128, h_q=8, h_kv=2, device_cc=device_cc)
    knobs = heur.SdpaFwdKnobs(cga=1, split_kv=3, pack_gqa=pack_gqa)
    assert mismatch(spec.capabilities, facts, knobs) is None
    assert mismatch(spec.capabilities, facts, replace(knobs, split_kv=1)) is not None
    for invalid in (replace(facts, has_sink=True), replace(facts, has_epilogue_gate=True), replace(facts, d_qk=256, d_v=256)):
        assert mismatch(spec.capabilities, invalid, knobs) is not None
    previous = type("PreviousNativeBinder", (), {"supports_nonpaged_packed_split": True, "supports_paged_packed_split": True})
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", previous)
    assert "matching native" in mismatch(spec.capabilities, facts, knobs)
    mla = replace(facts, d_qk=192)
    assert mismatch(spec.capabilities, mla, replace(knobs, pack_gqa=False)) is None


@requires_dsl
@pytest.mark.parametrize("group", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("batch,q,kv", [(1, 2048, 2048), (4, 128, 16384), (8, 128, 512), (16, 128, 1024), (2, 1024, 32768)])
@pytest.mark.parametrize("dtype", [cudnn.data_type.BFLOAT16, cudnn.data_type.HALF])
def test_nonpaged_d128_thd_lead_serves_a_valid_packed_first_plan(group, causal, batch, q, kv, dtype):
    """An unsplit THD lead is backed by the packed plan it puts first, and every first plan is legal."""
    from cudnn.sdpa.fwd import placement

    facts = _mla_split_facts(d_qk=128, b=batch, h_q=64, h_kv=64 // group, s_q=q, s_kv=kv, causal=causal, bottom_right=causal, dtype=dtype)
    first = heur._knob_sets(SPEC, facts)[0]
    assert mismatch(SPEC.capabilities, facts, first) is None
    if placement._place_sm100_f16(SPEC.capabilities, facts) == placement.LEAD and heur.nonpaged_thd_split_choice(SPEC.capabilities, facts)[0] == 1:
        assert first.pack_gqa and first.split_kv in (None, 1)


@requires_dsl
@pytest.mark.parametrize("batch,h_q,h_kv,q,kv", [(2, 8, 2, 128, 16384), (1, 32, 4, 64, 32768), (4, 16, 16, 257, 8192), (2, 16, 2, 8, 16384)])
@pytest.mark.parametrize("bounded_total", [False, True])
def test_nonpaged_d128_rubin_split_obeys_physical_wave_budget(monkeypatch, batch, h_q, h_kv, q, kv, bounded_total):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    total = q + batch - 1 if bounded_total else None
    facts = _mla_split_facts(d_qk=128, b=batch, h_q=h_q, h_kv=h_kv, s_q=q, s_kv=kv, device_cc=(10, 7), device_sm_count=204, max_total_seq_len_q=total)
    splits, packed = heur.nonpaged_thd_split_choice(spec.capabilities, facts)
    if splits > 1:
        # Count the selected candidate's physical CTAs, including a possible
        # partial tile per sequence, without pinning its split or pack choice.
        group = h_q // h_kv if packed else 1
        rows = 128 // group
        # Independently enumerate the shortest sequence that needs each
        # tile count, then all small-batch distributions within the capacity.
        lengths = [0, *range(1, q + 1, rows)]
        tiles = max(
            sum(len(range(0, length, rows)) for length in batch_lengths)
            for batch_lengths in product(lengths, repeat=batch)
            if total is None or sum(batch_lengths) <= total
        )
        # Rubin's budget: two waves.
        assert tiles * (h_q // group) * splits <= 2 * 204
        assert len(range(0, kv, 128)) // splits >= 4
        assert mismatch(spec.capabilities, facts, heur.SdpaFwdKnobs(cga=1, split_kv=splits, pack_gqa=packed)) is None


@pytest.mark.parametrize("split", [1, 2])
@requires_dsl
def test_sm107_paged_d128_cga1_geometry_matches_selected_template(split):
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    facts = _paged_split_facts(device_cc=(10, 7), device_sm_count=204, s_q=65)
    params = TemplateParams(dtype_qkv=DTYPE_BF16, cta_mma=1, thd_varlen=True, paged_kv=True, page_size=16, seq_kv_lens_present=True, split_kv=split)
    mod = _load_sm100_kernel_module((128, 128), params, rubin=True)
    rows = heur._pack_gqa_tile_q(spec.capabilities, facts, 128, 1, split_kv=split)
    assert rows == mod.CGA_TILE_M
    launch = heur._split_launch(spec.capabilities, facts, 128, 128, 1, 4, split_kv=split)
    assert launch.q_tiles == len(range(0, facts.s_q * 4, mod.CGA_TILE_M))


@requires_dsl
def test_sm107_paged_cga1_domain_is_distinct_from_dense_and_quantized(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    from cudnn.sdpa.fwd.engines import effective_cgas

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    facts = _paged_split_facts(device_cc=(10, 7), device_sm_count=204)
    for packing in (False, True):
        assert mismatch(spec.capabilities, facts, heur.SdpaFwdKnobs(cga=1, split_kv=1, pack_gqa=packing)) is None
    for other in (replace(facts, has_paged_kv=False), replace(facts, thd=False), replace(facts, d_qk=256, d_v=256), replace(facts, is_fp8=True)):
        assert 1 not in effective_cgas(spec.capabilities, other, 1)
    # dense d128: cga1 is the shared decode tile (issue #1472); dense PAGED queries above stay on cga2 (not wired on cc 10.7)
    assert 1 in effective_cgas(spec.capabilities, replace(facts, thd=False, has_paged_kv=False, page_size=0), 1)


@requires_dsl
@pytest.mark.parametrize("splits", [2, 3, 8])
@pytest.mark.parametrize("cga", [None, 2])
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("paged", [False, True])
def test_d256_thd_split_explicit_contract(monkeypatch, splits, cga, device_cc, paged):
    """The D256 packed ABI requires CGA2, half storage and its native binder."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    spec = next(s for s in ENGINE_SPECS if s.name == ("sdpa_fwd_prefill_sm107" if device_cc == (10, 7) else "sdpa_fwd_prefill_sm100"))
    facts = _paged_split_facts(device_cc=device_cc, d_qk=256, d_v=256, has_paged_kv=paged, page_size=16 if paged else 0)
    knobs = heur.SdpaFwdKnobs(cga=cga, split_kv=splits, pack_gqa=False)
    assert mismatch(spec.capabilities, facts, knobs) is None
    assert heur.SdpaFwdKnobs.from_public({int(k): v for k, v in knobs.to_public().items()}) == knobs
    for invalid in (replace(knobs, cga=1), replace(knobs, pack_gqa=True)):
        assert mismatch(spec.capabilities, facts, invalid) is not None
    for invalid in (
        replace(facts, has_sink=True),
        replace(facts, has_epilogue_gate=True),
        replace(facts, attn_scale_prefolded=True),
        replace(facts, wants_stats=True, stats_t=SimpleNamespace(ragged_offset=None)),
    ):
        assert mismatch(spec.capabilities, invalid, knobs) is not None
    previous = type("PreviousNativeBinder", (), {"supports_paged_packed_split": True})
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", previous)
    assert "matching native" in mismatch(spec.capabilities, facts, knobs)
    assert mismatch(spec.capabilities, facts, replace(knobs, split_kv=1)) is None


@requires_dsl
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("splits", [2, 3, 16, 64])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("cga", [None, 1])
def test_paged_d64_split_explicit_contract(monkeypatch, device_cc, splits, packed, cga):
    """D64 THD uses its native decode tile only for an explicit paged split."""
    facts = _paged_split_facts(device_cc=device_cc, d_qk=64, d_v=64)
    knobs = heur.SdpaFwdKnobs(cga=cga, split_kv=splits, pack_gqa=packed)
    assert mismatch(SPEC.capabilities, facts, knobs) is None
    assert heur.SdpaFwdKnobs.from_public({int(k): v for k, v in knobs.to_public().items()}) == knobs
    for invalid in (replace(knobs, cga=2), replace(knobs, split_kv=1, pack_gqa=True)):
        assert mismatch(SPEC.capabilities, facts, invalid) is not None
    for invalid in (replace(facts, has_paged_kv=False), replace(facts, has_sink=True), replace(facts, has_epilogue_gate=True)):
        assert mismatch(SPEC.capabilities, invalid, knobs) is not None
    previous = type("PreviousNativeBinder", (), {"supports_paged_packed_split": True})
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", previous)
    assert "matching native" in mismatch(SPEC.capabilities, facts, knobs)
    assert mismatch(SPEC.capabilities, facts, replace(knobs, split_kv=1, pack_gqa=False)) is None


@requires_dsl
@pytest.mark.parametrize("split", [1, 2, 3])
def test_d64_paged_split_geometry_matches_selected_template(split):
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16

    facts = _paged_split_facts(d_qk=64, d_v=64, s_q=257)
    params = TemplateParams(
        dtype_qkv=DTYPE_BF16,
        d_flavor=64,
        cta_mma=1,
        decode_tile=split > 1,
        thd_varlen=True,
        paged_kv=True,
        page_size=16,
        seq_kv_lens_present=True,
        split_kv=split,
    )
    mod = _load_sm100_kernel_module((64, 64), params)
    assert heur._pack_gqa_tile_q(SPEC.capabilities, facts, 128, 1, split_kv=split) == mod.CGA_TILE_M
    launch = heur._split_launch(SPEC.capabilities, facts, 128, 128, 1, 1, split_kv=split)
    assert launch.q_tiles == len(range(0, facts.s_q, mod.CGA_TILE_M))


@pytest.mark.parametrize("batch,max_q,group", [(1, 129, 1), (2, 129, 1), (3, 17, 8), (3, 33, 4), (4, 9, 16), (3, 0, 4)])
def test_d128_thd_grid_bound_matches_all_ragged_partitions(batch, max_q, group):
    """Declared packed capacity bounds every legal distribution, including empty slots."""
    # Enumerate real sequence tiles instead of mirroring the closed-form bound.
    by_total = [0] * (batch * max_q + 1)
    for lengths in product(range(max_q + 1), repeat=batch):
        units = sum(len(range(0, q * group, 128)) for q in lengths) * 2
        total = sum(lengths)
        by_total[total] = max(by_total[total], units)
    facts = _paged_split_facts(b=batch, h_q=2 * group, h_kv=2, s_q=max_q)
    worst = 0
    for total, units in enumerate(by_total):
        worst = max(worst, units)
        bounded = replace(facts, max_total_seq_len_q=total)
        assert heur._d128_thd_split_units(bounded, group) == worst
    assert heur._d128_thd_split_units(facts, group) == worst


@requires_dsl
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 7)])
@pytest.mark.parametrize("group", [1, 4, 8])
@pytest.mark.parametrize("shape_overrides", [False, True])
@pytest.mark.parametrize("wants_stats", [False, True])
def test_paged_split_redundant_sequence_bound_preserves_geometry(device_cc, group, shape_overrides, wants_stats):
    """A total-token bound already limits every sequence; extra rectangular slack adds no work."""
    spec = next(s for s in ENGINE_SPECS if s.name == ("sdpa_fwd_prefill_sm107" if device_cc == (10, 7) else "sdpa_fwd_prefill_sm100"))
    facts = _paged_split_facts(
        device_cc=device_cc,
        device_sm_count=204,
        b=3,
        h_q=4 * group,
        h_kv=4,
        s_q=127,
        max_total_seq_len_q=127,
        shape_overrides=shape_overrides,
        wants_stats=wants_stats,
    )
    # Both declarations admit exactly the same query-length distributions.
    # Compare them to each other, without pinning a winning split or packing.
    assert heur.paged_thd_split_choice(spec.capabilities, facts) == heur.paged_thd_split_choice(spec.capabilities, replace(facts, s_q=1024))


@requires_dsl
def test_paged_split_empty_declared_capacity_has_no_division():
    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    facts = _paged_split_facts(device_cc=(10, 7), max_total_seq_len_q=0)
    assert heur.paged_thd_split_choice(spec.capabilities, facts)[0] == 1


@requires_dsl
@pytest.mark.parametrize(
    "row,device_cc,sm_count,dtype",
    [
        ("sdpa_fwd_prefill_sm120", (12, 0), 188, cudnn.data_type.BFLOAT16),
        ("sdpa_fwd_prefill_sm120_fp8", (12, 0), 188, cudnn.data_type.FP8_E4M3),
        ("sdpa_fwd_prefill_sm100", (10, 0), 148, cudnn.data_type.BFLOAT16),
    ],
)
def test_split_chooser_receives_the_row_cost_profile(monkeypatch, row, device_cc, sm_count, dtype):
    """SM120 half hands its own split floor/coefficient to the shared chooser; FP8 and other rows keep the defaults."""
    spec = next(s for s in ENGINE_SPECS if s.name == row)
    seen = []
    real = heur.choose_split_kv
    monkeypatch.setattr(heur, "choose_split_kv", lambda **kw: seen.append(kw) or real(**kw))
    heur._knob_sets(
        spec,
        _facts(
            b=1,
            h_q=8,
            h_kv=1,
            s_q=4,
            s_kv=8192,
            causal=True,
            bottom_right=True,
            device_cc=device_cc,
            device_sm_count=sm_count,
            dtype=dtype,
            is_fp8=dtype == cudnn.data_type.FP8_E4M3,
        ),
    )
    assert seen
    sm120 = row == "sdpa_fwd_prefill_sm120"
    for kw in seen:
        assert kw.get("min_tiles") == (heur._SM120_SPLIT_KV_MIN_TILES if sm120 else None)
        assert kw.get("combine_floor") == (heur._SM120_SPLIT_KV_COMBINE_FLOOR if sm120 else None)
