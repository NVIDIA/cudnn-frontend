# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit SM90 implementation ownership and public recipe replay."""

import pytest
import torch
import cudnn
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm90.d64_f16_one_wg import contract as native
from cudnn.sdpa.fwd.kernels.sm90.f16_three_wg import contract as paged
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from determinism_utils import assert_bitwise, snapshot
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {native.NAME: 20539, paged.NAME: 20540}


def facts(**over):
    return SdpaGraphFacts(**(dict(b=2, h_q=4, h_kv=1, s_q=70, s_kv=100, d_qk=64, d_v=64, dtype=cudnn.data_type.HALF, device_cc=(9, 0)) | over))


@pytest.mark.parametrize("is_paged", [False, True])
def test_exact_d64_legacy_recipe_declines_and_has_one_owner(is_paged):
    f = facts(has_paged_kv=is_paged, page_size=16 if is_paged else 0)
    assert native.LEGACY_ROW not in SPECS
    assert native.in_domain(f) != paged.in_domain(f)
    other = native if is_paged else paged
    assert engines.mismatch(SPECS[other.NAME].capabilities, f) is not None


@pytest.mark.parametrize("sq", [1, 16, 65, 129])
@pytest.mark.parametrize("causal", [False, True])
def test_native_candidates_keep_public_records_and_placement(sq, causal):
    f = facts(s_q=sq, causal=causal)
    plans = heuristics.recommend("A", f, OFFERED)
    old = engines.sm100_legacy_policy_spec(native.LEGACY_ROW)
    expected = [k for k in heuristics._knob_sets(old, f) if engines.mismatch(old.capabilities, f, k) is None]
    assert len(plans) == len(expected) > 0
    assert [p.knobs for p in plans] == expected
    for plan in plans:
        assert plan.engine_id == 20539
        assert engines.SdpaFwdKnobs.from_public(plan.knobs.to_public()) == plan.knobs
        assert placement.place(SPECS[native.NAME], f, plan.knobs) == placement.place(old, f, plan.knobs)


@pytest.mark.parametrize("change", [dict(tile_m=128), dict(tile_n=32), dict(cga=2), dict(pack_gqa=True), dict(split_kv=2)])
def test_explicit_native_rejects_sibling_configuration(change):
    assert engines.mismatch(SPECS[native.NAME].capabilities, facts(), engines.SdpaFwdKnobs(**change)) is not None


@pytest.mark.parametrize("over", [dict(thd=True), dict(has_sink=True), dict(padded=True), dict(window_left=16), dict(device_cc=(10, 0))])
def test_explicit_native_preserves_feature_bounds(over):
    assert engines.mismatch(SPECS[native.NAME].capabilities, facts(**over)) is not None


def _graph(q, k, v, *, causal=False, stats=True, record=None, scheduler=0):
    io = cudnn.data_type.HALF if q.dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    nodes = [graph.tensor_like(t) for t in (q, k, v)]
    o = torch.empty_like(q)
    out, lse = graph.sdpa(q=nodes[0], k=nodes[1], v=nodes[2], generate_stats=stats, attn_scale=0.125, use_causal_mask=causal)
    out.set_output(True).set_dim(o.shape).set_stride(o.stride())
    vp = dict(zip(nodes, (q, k, v)))
    vp[out] = o
    results = {"o": o}
    if stats:
        lse_gpu = torch.empty(q.shape[:3], device="cuda", dtype=torch.float32)
        lse.set_output(True).set_dim([*lse_gpu.shape, 1]).set_stride([*lse_gpu.stride(), 1]).set_data_type(cudnn.data_type.FLOAT)
        vp[lse] = lse_gpu
        results["lse"] = lse_gpu
    graph.validate()
    graph.build_operation_graph()
    record = record or (20539, engines.SdpaFwdKnobs(tile_m=64, tile_n=64, cga=1, sched_policy=scheduler, pack_gqa=False, split_kv=1).to_public())
    graph.create_execution_plan(*record)
    graph.select_plan(graph.get_execution_plan_count() - 1)
    graph.check_support()
    graph.build_plans()
    assert graph.get_engine_and_knobs_at_index(graph._plan_index) == record
    assert graph.selected_engine == native.NAME or graph.selected_engine.name == native.NAME
    assert graph._compiled_plans[graph._plan_index]._prepared is not None
    ws = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
    return graph, vp, ws, results, record


@pytest.mark.skipif(_SM != 90, reason="requires native Hopper")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("stats", [False, True])
def test_public_recipe_replay_keeps_fixed_body_and_capture(monkeypatch, dtype, causal, stats):
    from cudnn.sdpa.fwd import api_dsl

    torch.manual_seed(1981)
    q = torch.randn(2, 70, 4, 64, device="cuda", dtype=dtype).transpose(1, 2)
    k, v = (torch.randn(2, 100, 1, 64, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2))
    loaded = []
    load = api_dsl._load_kernel_template

    def checked(path, *args, **kwargs):
        loaded.append(path)
        return load(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit SM90 body re-entered auto-selection or tensor execution")

    monkeypatch.setattr(api_dsl, "_load_kernel_template", checked)
    monkeypatch.setattr(api_dsl, "_sm90_pick_flavor", forbidden)
    monkeypatch.setattr(api_dsl, "_load_sm90_kernel_module", forbidden)
    monkeypatch.setattr(api_dsl.SdpaFwdDslSm90, "execute", forbidden)
    graph, vp, ws, outputs, record = _graph(q, k, v, causal=causal, stats=stats)
    graph.execute(vp, ws)
    expected = snapshot(outputs)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    replay, vp2, ws2, outputs2, _ = _graph(q2, k2, v2, causal=causal, stats=stats, record=record)
    replay.execute(vp2, ws2)
    assert_bitwise(snapshot(outputs2), expected, "public record rebuild")
    assert loaded and set(loaded) == {native.KERNEL_FILE}
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            torch.cuda.set_sync_debug_mode("error")
            try:
                replay.execute(vp2, ws2)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        v2.mul_(0.5)
        capture.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs2["o"], expected["o"] * 0.5, atol=5e-2, rtol=3e-2)
        if stats:
            assert_bitwise({"lse": outputs2["lse"]}, {"lse": expected["lse"]}, "captured stats")
    finally:
        capture.reset()
    # The retired public recipe must fail rather than rebuild a different body.
    with pytest.raises((RuntimeError, ValueError), match="owned by|supported|20517"):
        _graph(q, k, v, causal=causal, stats=stats, record=(20517, record[1]))


@pytest.mark.parametrize("owner", [native, paged])
def test_compiled_identity_rejects_legacy_or_sibling_parameter_records(owner):
    from importlib import import_module
    from cudnn.sdpa.fwd.config_sm90 import TemplateParams as LegacyParams

    geometry = owner.Geometry()
    knobs = geometry.canonical_knobs(engines.SdpaFwdKnobs())
    params_type = import_module(owner.__package__ + ".config").TemplateParams
    _, why = geometry.compiled_mismatch(owner.TEMPLATE_NAME, params_type(), knobs)
    assert why is None
    _, why = geometry.compiled_mismatch(owner.TEMPLATE_NAME, LegacyParams(), knobs)
    assert why is not None and "parameters belong" in why


@pytest.mark.skipif(_SM != 90, reason="requires native Hopper")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_paged_public_recipe_keeps_three_wg_body(monkeypatch, dtype):
    import test_sdpa_fwd_paged_sm120 as paging
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(paging, "sm120_half_owner", lambda *_: paged.NAME)
    loaded = []
    load = api_dsl._load_kernel_template

    def checked(path, *args, **kwargs):
        loaded.append(path)
        return load(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed three-WG owner re-entered legacy selection")

    monkeypatch.setattr(api_dsl, "_load_kernel_template", checked)
    monkeypatch.setattr(api_dsl, "_sm90_pick_flavor", forbidden)
    monkeypatch.setattr(api_dsl, "_load_sm90_kernel_module", forbidden)
    monkeypatch.setattr(api_dsl.SdpaFwdDslSm90, "execute", forbidden)
    graph, nodes, bufs, ws, _, _ = paging._case(
        dtype,
        2,
        8,
        True,
        False,
        dims=(64, 64),
        mask="br_swa",
        sink=True,
        sq=1,
        stats_log2=True,
        explicit_plan=True,
        independent_tables=True,
    )
    vp = {nodes[n]: t for n, t in bufs.items()}
    graph.execute(vp, ws)
    paging._check(bufs, 8, mask="br_swa", stats_log2=True)
    expected = snapshot({n: bufs[n] for n in ("o", "stats")})
    record = graph.get_engine_and_knobs_at_index(graph._plan_index)
    assert record[0] == 20540
    graph.create_execution_plan(*record)
    graph.select_plan(graph.get_execution_plan_count() - 1)
    graph.check_support()
    graph.build_plans()
    assert graph.get_engine_and_knobs_at_index(graph._plan_index) == record
    for name in ("o", "stats"):
        bufs[name].fill_(float("nan"))
    graph.execute(vp, ws)
    assert_bitwise({n: bufs[n] for n in ("o", "stats")}, expected, "paged public record rebuild")
    graph.create_execution_plan(20540, {})
    graph.select_plan(graph.get_execution_plan_count() - 1)
    graph.check_support()
    graph.build_plans()
    assert graph._compiled_plans[graph._plan_index]._compiled.implementation_config["scheduler"] == 2
    graph.execute(vp, ws)
    assert_bitwise({n: bufs[n] for n in ("o", "stats")}, expected, "paged omitted scheduler")
    assert loaded and set(loaded) == {paged.KERNEL_FILE}


@pytest.mark.parametrize("owner", [native, paged])
@pytest.mark.parametrize("causal,skv,expected", [(False, 100, 0), (True, 100, 2), (True, 300000, 1)])
def test_omitted_scheduler_keeps_adapter_default(owner, causal, skv, expected):
    from importlib import import_module

    lower = import_module(owner.__package__ + ".host").lower
    f = facts(causal=causal, s_kv=skv, has_paged_kv=owner is paged, padded=owner is paged, page_size=16 if owner is paged else 0)
    result = lower(SPECS[owner.NAME], f, None, lower_impl=lambda spec, facts, knobs, **kw: knobs, knob_type=engines.SdpaFwdKnobs)
    assert result.sched_policy == expected


@pytest.mark.skipif(_SM != 90, reason="requires native Hopper")
def test_native_public_omitted_scheduler():
    torch.manual_seed(1759)
    q = torch.randn(2, 70, 4, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k, v = (torch.randn(2, 100, 1, 64, device="cuda", dtype=q.dtype).transpose(1, 2) for _ in range(2))
    graph, vp, ws, output, _ = _graph(q, k, v, causal=True, record=(20539, {}))
    assert graph._compiled_plans[graph._plan_index]._compiled.implementation_config["scheduler"] == 2
    graph.execute(vp, ws)
    explicit, vp2, ws2, expected, _ = _graph(q, k, v, causal=True, scheduler=2)
    explicit.execute(vp2, ws2)
    assert_bitwise(output, expected, "native omitted scheduler")
