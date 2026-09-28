# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model inputs follow candidate geometry; no timing/rank golden assertions."""

from dataclasses import replace
from itertools import product

import pytest

import cudnn
from frost_test_utils import requires_blackwell, requires_dsl
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


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"shape_overrides": False},
        {"thd": False},
        {"causal": True},
        {"window_left": 31},
        {"has_paged_kv": True},
        {"has_sink": True},
        {"has_epilogue_gate": True},
        {"right_band_widening": True},
        {"device_cc": (10, 3)},
        {"device_cc": (10, 7)},
        {"d_qk": 128},
    ],
)
def test_runtime_cga_policy_has_explicit_record_and_bounded_domain(overrides):
    facts = _facts(h_q=16, h_kv=16, s_q=65536, s_kv=65536, d_qk=192, d_v=128, thd=True, shape_overrides=True)
    facts = replace(facts, **overrides)
    knobs = heur._knob_sets(SPEC, facts)
    policies = [k for k in knobs if k.cga_policy is not None]
    if overrides:
        assert not policies
        return
    assert len(policies) == 1
    policy = policies[0]
    record = policy.to_public()
    assert record[cudnn.knob_type.CGA_POLICY] == 2
    assert cudnn.knob_type.TILE_CGA_M not in record
    assert heur.SdpaFwdKnobs.from_public({int(k): v for k, v in record.items()}) == policy
    assert any(k.cga in (1, 2) and k.cga_policy is None for k in knobs)
    assert len(knobs) == len(set(knobs)) <= heur._MAX_SETS_PER_ENGINE
    assert heur._fallback_knobs(SPEC, facts).cga_policy is None


@pytest.mark.parametrize("bad", [{"cga": 1}, {"cga": 2}, {"split_kv": 2}, {"pack_gqa": True}, {"cga_policy": 0}, {"cga_policy": 3}])
def test_runtime_cga_policy_rejects_conflicting_or_unknown_requests(bad):
    facts = _facts(h_q=16, h_kv=16, d_qk=192, d_v=128, thd=True, shape_overrides=True)
    knobs = heur.SdpaFwdKnobs(cga_policy=1)
    reason = mismatch(SPEC.capabilities, facts, replace(knobs, **bad))
    assert reason is not None and ("CGA" in reason or "cga" in reason)


@pytest.mark.parametrize("policy", [1, 2])
def test_runtime_cga_policy_records_remain_rebuildable(policy):
    facts = _facts(h_q=16, h_kv=16, d_qk=192, d_v=128, thd=True, shape_overrides=True)
    knobs = heur.SdpaFwdKnobs(cga_policy=policy)
    assert mismatch(SPEC.capabilities, facts, knobs) is None
    assert heur.SdpaFwdKnobs.from_public({int(k): v for k, v in knobs.to_public().items()}) == knobs


@pytest.mark.parametrize("overrides", [{}, {"cta_mma": 1}, {"split_kv": 2}, {"thd_varlen": False}, {"paged_kv": True, "page_size": 128}, {"dtype_qkv": 0}])
def test_thd_pair_acquire_lowering_is_bounded(overrides):
    from cudnn.sdpa.fwd import config_sm100, config_sm107

    params = config_sm100.TemplateParams(dtype_qkv=2, cta_mma=2, thd_varlen=True, seq_kv_lens_present=True, thd_pair_acquire=True)
    params = replace(params, **overrides)
    if overrides:
        with pytest.raises(ValueError, match="thd_pair_acquire"):
            config_sm100.make_cfg_d192(params)
    else:
        cfg, _ = config_sm100.make_cfg_d192(params)
        assert cfg.THD_PAIR_ACQUIRE
    with pytest.raises(ValueError, match="thd_pair_acquire"):
        config_sm100.make_cfg_d128(params)
    with pytest.raises(ValueError, match="thd_pair_acquire"):
        config_sm107.make_cfg_d192(params)


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


@pytest.mark.parametrize(
    "thd,paged,cga,rubin", [(True, False, 1, False), (True, False, 2, False), (False, False, 1, False), (True, True, 1, False), (True, False, 2, True)]
)
def test_d192_model_rows_match_selected_kernel(thd, paged, cga, rubin):
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16

    params = TemplateParams(dtype_qkv=DTYPE_BF16, cta_mma=cga, thd_varlen=thd, paged_kv=paged, seq_kv_lens_present=thd or paged, page_size=128 if paged else 0)
    module = _load_sm100_kernel_module((192, 128), params, rubin=rubin)
    caps = replace(SPEC.capabilities, sm_lo=107, sm_hi=119) if rubin else SPEC.capabilities
    facts = _facts(d_qk=192, d_v=128, thd=thd, has_paged_kv=paged, device_cc=(10, 7) if rubin else (10, 0))
    rows = heur._pack_gqa_tile_q(caps, facts, 128, cga)
    assert rows == module.CGA_TILE_M
    launch = heur._split_launch(caps, facts, 128, 128, cga, 1)
    assert launch.q_tiles == (facts.s_q + module.CGA_TILE_M - 1) // module.CGA_TILE_M


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
@pytest.mark.parametrize("s_q", [0, 1, 3, 6])
@pytest.mark.parametrize("tile_rows", [1, 2, 4])
def test_packed_query_tile_bound_matches_length_distributions(batch_size, s_q, tile_rows):
    distributions = list(product(range(s_q + 1), repeat=batch_size))
    for capacity in range(batch_size * s_q + 2):
        expected = max(sum((length + tile_rows - 1) // tile_rows for length in lengths) for lengths in distributions if sum(lengths) <= capacity)
        assert heur._thd_q_tile_bound(batch_size, s_q, tile_rows, capacity) == expected
    assert heur._thd_q_tile_bound(batch_size, s_q, tile_rows) == batch_size * ((s_q + tile_rows - 1) // tile_rows)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("total_q", [None, 1024])
@pytest.mark.parametrize("b,h,q,sm_count", [(1, 16, 1024, 148), (2, 16, 1024, 148), (3, 4, 129, 132)])
def test_d192_selector_receives_declared_launch_geometry(monkeypatch, paged, total_q, b, h, q, sm_count):
    # Spy on the chooser inputs, not the performance policy's winning rank.
    # The optional capacity is a graph fact; device lengths remain unavailable.
    seen = []
    choose = heur.select_d192_auto_knobs

    def recording(params, **kwargs):
        seen.append((params, kwargs))
        return choose(params, **kwargs)

    monkeypatch.setattr(heur, "select_d192_auto_knobs", recording)
    facts = _facts(b=b, h_q=h, h_kv=h, s_q=q, s_kv=8192, d_qk=192, d_v=128, thd=True, has_paged_kv=paged, device_sm_count=sm_count, max_total_seq_len_q=total_q)
    heur._auto_sched_cga(SPEC, facts, split_kv=1, sched_policy=0, pack_gqa=False)
    assert len(seen) == 1
    params, args = seen[0]
    assert params.thd_varlen and params.paged_kv == paged
    assert (args["batch_size"], args["h_q"], args["s_q"], args["s_kv"]) == (b, h, q, 8192)
    assert args["device_sm_count"] == sm_count and args["device_cc"] == facts.device_cc
    assert args["max_total_seq_len_q"] == total_q


@requires_blackwell
@requires_dsl
def test_d192_standalone_and_graph_share_launch_facts(monkeypatch):
    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, q, k = 3, 4, 129, 513
    q_tensor = torch.empty(b, q, h, 192, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k_tensor = torch.empty(b, k, h, 192, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    v_tensor = torch.empty(b, k, h, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    out = torch.empty(b, q, h, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    seen = []
    choose = heur.select_d192_auto_knobs

    def recording(params, **kwargs):
        seen.append(kwargs)
        return choose(params, **kwargs)

    monkeypatch.setattr(heur, "select_d192_auto_knobs", recording)
    api = SdpaFwdDslSm100(q_tensor, k_tensor, v_tensor, out, thd=True)
    api.check_support()
    params = api.template_params()
    cc = torch.cuda.get_device_capability(q_tensor.device)
    sm_count = torch.cuda.get_device_properties(q_tensor.device).multi_processor_count
    facts = _facts(b=b, h_q=h, h_kv=h, s_q=q, s_kv=k, d_qk=192, d_v=128, thd=True, device_cc=cc, device_sm_count=sm_count)
    _, cga = heur._auto_sched_cga(SPEC, facts, split_kv=1, sched_policy=0, pack_gqa=False)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert params.cta_mma == cga
