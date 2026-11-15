# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The small-half recompute record fixes its body independently of legacy selection."""

from dataclasses import replace

import pytest

import cudnn
from cudnn.sdpa.bwd import engines, heuristics
from cudnn.sdpa.bwd.d128_selection import ACCUMULATION_OWNER, LEGACY_ROW, PAIR_OWNER, RECOMPUTE_OWNER, TWO_CTA_OWNER
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from frost_test_utils import requires_bwd_engine_row, requires_dsl

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7), (12, 0)])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("profile", ["dense", "negative_scale", "window", "deterministic", "grouped", "thd"])
def test_recompute_preserves_domain_and_default_order(monkeypatch, cc, d, profile):
    from cudnn.frost import buffers

    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    specs = {s.name: s for s in engines.ENGINE_SPECS}
    row, owner = specs[LEGACY_ROW], specs[RECOMPUTE_OWNER]
    old = replace(row.capabilities, select_implementation=None, delegated=frozenset())
    changes = {
        "dense": {},
        "negative_scale": {"scale": -0.125},
        "window": {"causal": True, "window_left": 64},
        "deterministic": {"deterministic": True},
        "grouped": {"h_kv": 1},
        "thd": {"thd": True, "padded": True, "max_total_seq_len_q": 256, "max_total_seq_len_kv": 512},
    }[profile]
    facts = SdpaGraphFacts(
        **(dict(is_backward=True, b=2, h_q=2, h_kv=2, s_q=128, s_kv=256, d_qk=d, d_v=d, dtype=cudnn.data_type.BFLOAT16, device_cc=cc) | changes)
    )
    baseline = engines.mismatch(old, facts)
    reasons = [engines.mismatch(s.capabilities, facts) for s in (row, owner, specs[PAIR_OWNER], specs[ACCUMULATION_OWNER], specs[TWO_CTA_OWNER])]
    assert (None in reasons) == (baseline is None)
    if baseline is None and profile not in ("grouped", "thd"):
        assert reasons[1] is None
    offered = {LEGACY_ROW: 20609, RECOMPUTE_OWNER: 20628, PAIR_OWNER: 20629, ACCUMULATION_OWNER: 20630, TWO_CTA_OWNER: 20631}
    plans = heuristics.recommend("A", facts, offered)
    if baseline is None:
        expected = [offered[name] for name in engines.legal_implementations(row.capabilities, facts)]
        assert [plan.engine_id for plan in plans] == expected
    else:
        assert not plans


@requires_dsl
@requires_bwd_engine_row(RECOMPUTE_OWNER)
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("left", [None, 17])
def test_public_recompute_record_rebuild_ignores_legacy_selection(monkeypatch, d, dtype_name, left):
    import torch

    from cudnn.sdpa.bwd import api_dsl_sm100_d128 as legacy
    import test_sdpa_bwd_d128_sm100 as helper

    def forbidden(*args, **kwargs):
        raise AssertionError("a fixed recompute record consulted the legacy selector")

    monkeypatch.setattr(legacy, "select_implementation", forbidden)
    dtype = getattr(torch, dtype_name)
    case = helper._case(dt=dtype, d=d, deterministic=True, causal=True, left=left)
    record = case.graph.get_engine_and_knobs_at_index(case.graph._plan_index)
    assert record == (20628, {})
    expected = [case.tensors[name].clone() for name in ("dq", "dk", "dv")]
    graph, refs, outputs = helper._half_bwd_graph(
        b=1,
        hq=2,
        sq=65,
        skv=129,
        d=d,
        dt=dtype,
        use_causal_mask=True,
        use_deterministic_algorithm=True,
        **({"diagonal_band_left_bound": left} if left is not None else {}),
    )
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plan(*record)
    graph.select_plan(0)
    graph.check_support()
    graph.build_plans()
    assert graph.get_workspace_size() == 0
    assert graph._compiled_plans[0]._prepared.spec.artifact.implementation == "recompute"
    refs.update(zip(("dq", "dk", "dv"), outputs))
    for name in ("dq", "dk", "dv"):
        case.tensors[name].fill_(float("nan"))
    graph.execute({refs[name]: value for name, value in case.tensors.items()}, None)
    helper._check_case(case)
    for name, want in zip(("dq", "dk", "dv"), expected):
        assert torch.equal(case.tensors[name].view(torch.int16), want.view(torch.int16)), name
