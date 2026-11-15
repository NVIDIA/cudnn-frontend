# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full Hopper ownership: retired records, fixed bodies and native replay."""

from dataclasses import asdict
import pytest
import torch
import cudnn
from cudnn.engines import manifest
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm90.d128_f16_multi_q import contract as d128
from cudnn.sdpa.fwd.kernels.sm90.f16_three_wg import contract as wide
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from frost_test_utils import _SM, requires_dsl
from determinism_utils import assert_bitwise, snapshot

pytestmark = [pytest.mark.L0, requires_dsl]
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
FAMILY = next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_fwd")
OFFERED = FAMILY.offered_ids()


def _facts(**over):
    return SdpaGraphFacts(
        **(dict(b=2, h_q=8, h_kv=2, s_q=1024, s_kv=4096, d_qk=128, d_v=128, dtype=cudnn.data_type.BFLOAT16, device_cc=(9, 0), device_sm_count=132) | over)
    )


def test_retired_sm90_id_is_not_offered_or_reused():
    assert d128.LEGACY_ROW not in SPECS and d128.LEGACY_ROW not in OFFERED
    assert 20517 not in OFFERED.values()
    assert OFFERED[wide.NAME] == 20540 and OFFERED[d128.NAME] == 20541
    assert manifest.engine_for_id(20517) is None


@pytest.mark.parametrize("dims", [(128, 128), (120, 72), (256, 256), (264, 512), (512, 512)])
@pytest.mark.parametrize(
    "features", [{}, dict(causal=True), dict(causal=True, thd=True), dict(padded=True), dict(has_paged_kv=True, padded=True, page_size=16), dict(has_sink=True)]
)
@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_full_ownership_preserves_policy_candidates_and_placement(dims, features, kind):
    f = _facts(d_qk=dims[0], d_v=dims[1], **features)
    old = engines.sm100_legacy_policy_spec(d128.LEGACY_ROW)
    candidates = heuristics._knob_sets(old, f) if kind == "A" else [heuristics._fallback_knobs(old, f)]
    expected = [k for k in candidates if engines.mismatch(old.capabilities, f, k) is None]
    plans = heuristics.recommend(kind, f, OFFERED)
    assert [p.knobs for p in plans] == expected
    for plan in plans:
        owner = d128 if plan.knobs.tile_m == 128 else wide
        assert plan.engine_id == OFFERED[owner.NAME]
        assert engines.mismatch(SPECS[owner.NAME].capabilities, f, plan.knobs) is None
        assert placement.place(SPECS[owner.NAME], f, plan.knobs) == placement.place(old, f, plan.knobs)


@pytest.mark.parametrize(
    "over",
    [
        dict(thd=True),
        dict(has_sink=True),
        dict(window_left=16),
        dict(padded=True),
        dict(has_paged_kv=True, padded=True, page_size=16),
        dict(s_q=1),
        dict(causal=True, bottom_right=True, s_kv=128),
    ],
)
def test_d128_declaration_declines_before_lowering(over):
    assert engines.mismatch(SPECS[d128.NAME].capabilities, _facts(**over)) is not None


@pytest.mark.parametrize("owner,tile", [(wide, 64), (d128, 128)])
@pytest.mark.parametrize("partial", [{}, dict(tile_m=None), dict(tile_n=None), dict(tile_m=None, tile_n=None)])
def test_partial_geometry_is_admitted_with_the_fixed_body_default(owner, tile, partial):
    from importlib import import_module

    values = dict(tile_m=tile, tile_n=tile)
    values.update(partial)
    k = engines.SdpaFwdKnobs(**values)
    assert engines.mismatch(SPECS[owner.NAME].capabilities, _facts(), k) is None
    fixed = import_module(owner.__package__ + ".host").lower(SPECS[owner.NAME], _facts(), k, lower_impl=lambda s, f, k, **kw: k, knob_type=engines.SdpaFwdKnobs)
    assert (fixed.tile_m, fixed.tile_n) == (tile, tile)


def test_three_wg_thd_omitted_scheduler_stays_natural():
    from cudnn.sdpa.fwd.kernels.sm90.f16_three_wg.host import lower

    k = lower(SPECS[wide.NAME], _facts(thd=True, causal=True), None, lower_impl=lambda s, f, k, **kw: k, knob_type=engines.SdpaFwdKnobs)
    assert k.sched_policy == 0


_NATIVE_CASES = [
    pytest.param(d128, (128, 128), (2, 4, 1, 129, 257), False, False, id="d128-dense"),
    pytest.param(d128, (128, 128), (4, 16, 4, 512, 512), True, False, id="d128-persistent"),
    pytest.param(d128, (128, 128), (4, 16, 4, 512, 512), True, True, id="d128-claim"),
    pytest.param(wide, (128, 128), (2, 4, 1, 129, 257), True, False, id="three-wg-d128"),
    pytest.param(wide, (120, 72), (2, 4, 1, 129, 257), False, False, id="three-wg-envelope"),
    pytest.param(wide, (256, 256), (2, 4, 1, 129, 257), True, False, id="three-wg-d256"),
    pytest.param(wide, (512, 512), (2, 4, 1, 129, 257), True, False, id="three-wg-d512"),
]


@pytest.mark.skipif(_SM != 90, reason="requires native Hopper")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("owner,dims,shape,causal,claim", _NATIVE_CASES)
def test_public_body_replay_matches_parent_parameters_and_bytes(monkeypatch, dtype, owner, dims, shape, causal, claim):
    from cudnn.sdpa.fwd import api_dsl
    from test_sdpa_fwd_dsl_sm100 import _bhsd, _ref_sdpa_full

    if claim:
        monkeypatch.setattr(api_dsl, "D128_CLAIM_MIN_ROUNDS", 0)
    b, hq, hk, sq, sk = shape
    dq, dv = dims
    tile = 128 if owner is d128 else 64
    torch.manual_seed(2081)
    q = _bhsd(b, hq, sq, dq, dtype)
    k = _bhsd(b, hk, sk, dq, dtype)
    v = _bhsd(b, hk, sk, dv, dtype)
    parent_o = _bhsd(b, hq, sq, dv, dtype)
    parent_lse = torch.empty(b, hq, sq, device="cuda", dtype=torch.float32)
    parent = api_dsl.SdpaFwdDslSm90(q, k, v, parent_o, sample_lse=parent_lse, tile_m=tile, tile_n=tile, is_causal=causal)
    parent.compile()
    parent_ws = torch.empty(max(1, parent.scratch_workspace_bytes()), device="cuda", dtype=torch.uint8)
    parent.execute(q, k, v, parent_o, lse_tensor=parent_lse, workspace=parent_ws)
    expected = snapshot({"o": parent_o, "lse": parent_lse})
    ref, stats = _ref_sdpa_full(q, k, v, scale=dq**-0.5, is_causal=causal, return_stats=True)
    torch.testing.assert_close(parent_o, ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(parent_lse, stats.squeeze(-1), atol=2e-2, rtol=2e-2)
    if claim:
        assert parent.params.causal_claim == 1

    def forbidden(*a, **kw):
        raise AssertionError("explicit body replay re-entered automatic selection or tensor execution")

    monkeypatch.setattr(api_dsl, "_sm90_pick_flavor", forbidden)
    monkeypatch.setattr(api_dsl, "_load_sm90_kernel_module", forbidden)
    monkeypatch.setattr(api_dsl.SdpaFwdDslSm90, "execute", forbidden)
    monkeypatch.setattr(heuristics, "select_sm90_geometry", forbidden)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    nodes = [graph.tensor_like(t) for t in (q, k, v)]
    o = torch.full_like(parent_o, float("nan"))
    lse = torch.full_like(parent_lse, float("nan"))
    out, st = graph.sdpa(q=nodes[0], k=nodes[1], v=nodes[2], generate_stats=True, attn_scale=dq**-0.5, use_causal_mask=causal)
    out.set_output(True).set_dim(o.shape).set_stride(o.stride())
    st.set_output(True).set_dim([*lse.shape, 1]).set_stride([*lse.stride(), 1]).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    graph.build_operation_graph()
    with pytest.raises(ValueError, match="no python engine on this graph owns engine_id 20517"):
        graph.create_execution_plan(20517, {})
    record = (OFFERED[owner.NAME], {})
    graph.create_execution_plan(*record)
    graph.select_plan(graph.get_execution_plan_count() - 1)
    graph.check_support()
    graph.build_plans()
    assert graph.get_engine_and_knobs_at_index(graph._plan_index) == record
    compiled = graph._compiled_plans[graph._plan_index]._compiled
    assert compiled.kernel_template == owner.TEMPLATE_NAME
    assert asdict(compiled.kernel_module.PARAMS) == asdict(parent.params)
    assert compiled.implementation_config["scheduler"] == parent.params.sched_policy
    ws = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
    vp = dict(zip(nodes, (q, k, v))) | {out: o, st: lse}
    graph.execute(vp, ws)
    assert_bitwise({"o": o, "lse": lse}, expected, "old body / explicit recipe")
    fresh = [t.clone() for t in (q, k, v)]
    vp.update(zip(nodes, fresh))
    graph.execute(vp, ws)
    assert_bitwise({"o": o, "lse": lse}, expected, "fresh binding")
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            torch.cuda.set_sync_debug_mode("error")
            try:
                graph.execute(vp, ws)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        fresh[2].mul_(0.5)
        capture.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(o, expected["o"] * 0.5, atol=5e-2, rtol=3e-2)
        assert_bitwise({"lse": lse}, {"lse": expected["lse"]}, "retained Stats")
    finally:
        capture.reset()


def test_d128_compiled_identity_rejects_shared_and_sibling_records():
    from cudnn.sdpa.fwd.config_sm90 import TemplateParams as Shared
    from cudnn.sdpa.fwd.kernels.sm90.f16_three_wg.config import TemplateParams as Sibling
    from cudnn.sdpa.fwd.kernels.sm90.d128_f16_multi_q.config import TemplateParams as Owned

    g = d128.Geometry()
    knobs = g.canonical_knobs(engines.SdpaFwdKnobs(sched_policy=0))
    for cls in (Shared, Sibling):
        _, reason = g.compiled_mismatch(d128.TEMPLATE_NAME, cls(head_tile=128), knobs)
        assert reason is not None and "parameters belong to" in reason
    assert g.compiled_mismatch(d128.TEMPLATE_NAME, Owned(head_tile=128), knobs)[1] is None


@pytest.mark.skipif(_SM != 90, reason="requires native Hopper")
@pytest.mark.parametrize(
    "owner,thd,d",
    [
        ("d64_f16_one_wg", False, 64),
        ("f16_three_wg", False, 512),
        ("f16_three_wg", True, 512),
        ("d128_f16_multi_q", False, 128),
    ],
    ids=["native-d64", "three-wg-dense", "three-wg-thd", "dedicated-d128"],
)
def test_owned_artifact_reloads_without_jit_in_fresh_process(tmp_path, owner, thd, d):
    from test_sdpa_prepared_sm90 import test_prepared_artifact_reloads_in_fresh_process

    test_prepared_artifact_reloads_in_fresh_process(thd, tmp_path, d=d, owner=owner)
