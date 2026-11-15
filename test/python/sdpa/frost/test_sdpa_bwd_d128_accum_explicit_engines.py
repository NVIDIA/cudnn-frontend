# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed accumulation records retain numerical, binding and cluster contracts."""

from dataclasses import replace

import pytest

import cudnn
from cudnn.sdpa.bwd import engines, heuristics
from cudnn.sdpa.bwd.d128_selection import ACCUMULATION_OWNER, LEGACY_ROW, TWO_CTA_OWNER
from frost_test_utils import requires_bwd_engine_row, requires_dsl

pytestmark = pytest.mark.L0
OWNERS = {ACCUMULATION_OWNER: 20630, TWO_CTA_OWNER: 20631}


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("sq,sk", [(128, 256), (4096, 8192), (8192, 8192), (16384, 8192)])
def test_grouped_default_order_and_explicit_alternative(monkeypatch, cc, thd, sq, sk):
    from cudnn.frost import buffers
    from test_sdpa_bwd_d128_grouped_eligibility import grouped_facts

    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    facts = grouped_facts(thd=thd, device_cc=cc, s_q=sq, s_kv=sk, max_total_seq_len_q=2 * sq, max_total_seq_len_kv=2 * sk)
    specs = {s.name: s for s in engines.ENGINE_SPECS}
    original = replace(specs[LEGACY_ROW].capabilities, select_implementation=None, delegated=frozenset())
    assert engines.mismatch(original, facts) is None
    assert engines.mismatch(specs[LEGACY_ROW].capabilities, facts) is not None
    for name in OWNERS:
        assert engines.mismatch(specs[name].capabilities, facts) is None
    expected = [20631, 20630] if min(sq, sk) >= 8192 else [20630, 20631]
    assert [p.engine_id for p in heuristics.recommend("A", facts, OWNERS)] == expected
    for name, eid in OWNERS.items():
        assert [p.engine_id for p in heuristics.recommend("A", facts, {name: eid})] == [eid]


def test_cluster_decline_does_not_change_explicit_body(monkeypatch):
    from cudnn.sdpa.bwd import prepared_sm100_d128_grouped_2cta as pair
    from cudnn.sdpa.bwd.kernels.sm100.d128_f16_accum.host import SdpaBwdDslSm100D128Accumulation
    from cudnn.sdpa.bwd.kernels.sm100.d128_f16_accum_2cta.host import SdpaBwdDslSm100D128AccumulationTwoCTA
    from test_sdpa_bwd_d128_cta_pair_heuristic import _declaration_api, _dense_geometry

    one = _declaration_api(monkeypatch, _dense_geometry(s=8192), api_type=SdpaBwdDslSm100D128Accumulation)
    two = _declaration_api(monkeypatch, _dense_geometry(s=8192), api_type=SdpaBwdDslSm100D128AccumulationTwoCTA)
    monkeypatch.setattr(pair, "max_active_pairs", lambda device: 0)
    monkeypatch.setattr(pair, "probe_failure", lambda device: "test device has no admitted cluster")

    def forbidden(*args, **kwargs):
        raise AssertionError("a fixed owner consulted the legacy body selector")

    monkeypatch.setattr(pair, "check_support_selected", forbidden)
    with pytest.raises(NotImplementedError, match="cluster.*test device has no admitted cluster"):
        two.check_support()
    assert one.check_support() is True
    assert one._implementation == "accumulation_grouped_dense"


@requires_dsl
@requires_bwd_engine_row(ACCUMULATION_OWNER)
@pytest.mark.parametrize("d,left", [(64, None), (64, 17), (128, None)])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_public_dense_accumulation_record_rebuild(monkeypatch, d, left, dtype_name):
    import torch

    import test_sdpa_bwd_d128_sm100 as helper
    from cudnn.sdpa.bwd import api_dsl_sm100_d128 as legacy
    from test_sdpa_bwd_d128_accum_sm100 import _require_accumulation

    _require_accumulation()

    def forbidden(*args, **kwargs):
        raise AssertionError("a fixed accumulation record consulted the legacy selector")

    monkeypatch.setattr(legacy, "select_implementation", forbidden)
    dtype = getattr(torch, dtype_name)
    case = helper._case(dt=dtype, d=d, causal=True, left=left, sq=65, sk=129, accumulation=True)
    record = case.graph.get_engine_and_knobs_at_index(case.graph._plan_index)
    assert record == (20630, {})
    graph, refs, outputs = helper._build_graph(
        b=1,
        hq=2,
        hkv=2,
        sq=65,
        skv=129,
        d=d,
        dt=dtype,
        use_causal_mask=True,
        **({"diagonal_band_left_bound": left} if left is not None else {}),
    )
    graph.create_execution_plan(*record)
    graph.select_plan(0)
    graph.check_support()
    graph.build_plans()
    assert graph.get_workspace_size() == case.workspace.numel()
    assert graph._compiled_plans[0]._prepared.spec.artifact.implementation == "accumulation"
    refs.update(zip(("dq", "dk", "dv"), outputs))
    for name in ("dq", "dk", "dv"):
        case.tensors[name].fill_(float("nan"))
    graph.execute({refs[name]: value for name, value in case.tensors.items()}, case.workspace)
    helper._check_case(case)


@requires_dsl
@requires_bwd_engine_row(ACCUMULATION_OWNER)
@pytest.mark.parametrize("owner", [ACCUMULATION_OWNER, TWO_CTA_OWNER])
@pytest.mark.parametrize("thd,packing", [(False, "TH1"), (True, "TH1"), (True, "HN")])
def test_public_grouped_record_rebuild_and_capture(monkeypatch, owner, thd, packing):
    import torch

    import d128_grouped_test_utils as helper
    from cudnn.sdpa.bwd import prepared_sm100_d128_grouped_2cta as pair
    from test_sdpa_bwd_d128_accum_sm100 import _require_accumulation

    _require_accumulation()
    if owner == TWO_CTA_OWNER and pair.max_active_pairs(torch.device("cuda", torch.cuda.current_device())) < 1:
        pytest.skip("device does not admit a two-CTA cluster")

    def forbidden(*args, **kwargs):
        raise AssertionError("a fixed record consulted the legacy grouped selector")

    monkeypatch.setattr(pair, "check_support_selected", forbidden)
    monkeypatch.setattr(pair, "compile_plan_selected", forbidden)
    lens_q, lens_kv = ((65, 0, 129), (129, 0, 257)) if thd else ((128, 128), (256, 256))
    case = helper.make_case(lens_q, lens_kv, thd=thd, causal=True, stats_layout=packing, gaps=thd)
    graph, refs = helper.make_graph(case, owner, OWNERS[owner])
    record = graph.get_engine_and_knobs_at_index(graph._plan_index)
    assert record == (OWNERS[owner], {})
    workspace = helper.make_workspace(graph.get_workspace_size())
    graph.execute(helper.variant_pack(case, refs), workspace.tensor)
    helper.check_case(case, workspace)
    graph2, refs2 = helper.make_graph(case, owner, OWNERS[owner], record=record)
    assert graph2.get_workspace_size() == graph.get_workspace_size()
    fresh = helper.make_case(lens_q, lens_kv, thd=thd, causal=True, stats_layout=packing, gaps=thd, seed=311)
    fresh_pack = helper.variant_pack(fresh, refs2)
    graph2.execute(fresh_pack, workspace.tensor)
    helper.check_case(fresh, workspace)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        graph2.execute(fresh_pack, workspace.tensor)
    helper.poison(fresh, workspace)
    capture.replay()
    helper.check_case(fresh, workspace)
    with pytest.raises((ValueError, RuntimeError), match="20609|owned by|not supported|No execution plans"):
        helper.make_graph(case, LEGACY_ROW, 20609, record=(20609, {}))
