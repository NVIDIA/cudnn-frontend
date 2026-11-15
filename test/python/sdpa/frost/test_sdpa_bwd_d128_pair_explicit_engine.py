# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The explicit small-half pair fixes both main bodies before compilation."""

from dataclasses import replace

import cudnn
import pytest

from cudnn.sdpa.bwd import engines, heuristics
from cudnn.sdpa.bwd.d128_selection import ACCUMULATION_OWNER, LEGACY_ROW, PAIR_OWNER, RECOMPUTE_OWNER, TWO_CTA_OWNER
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from frost_test_utils import requires_bwd_engine_row, requires_dsl

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7), (12, 0)])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
@pytest.mark.parametrize("profile", ["plain", "deterministic", "negative_scale", "grouped", "grouped_deterministic", "thd"])
def test_pair_preserves_admission_and_default_order(monkeypatch, cc, d, dtype, profile):
    from cudnn.frost import buffers

    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    specs = {s.name: s for s in engines.ENGINE_SPECS}
    row = specs[LEGACY_ROW]
    original = replace(row.capabilities, select_implementation=None, delegated=frozenset())
    changes = {
        "plain": {},
        "deterministic": {"deterministic": True},
        "negative_scale": {"scale": -0.125},
        "grouped": {"h_kv": 1},
        "grouped_deterministic": {"h_kv": 1, "deterministic": True},
        "thd": {"thd": True, "padded": True, "max_total_seq_len_q": 256, "max_total_seq_len_kv": 512},
    }[profile]
    facts = SdpaGraphFacts(**(dict(is_backward=True, b=2, h_q=2, h_kv=2, s_q=128, s_kv=256, d_qk=d, d_v=d, dtype=dtype, device_cc=cc) | changes))
    owners = (LEGACY_ROW, RECOMPUTE_OWNER, PAIR_OWNER, ACCUMULATION_OWNER, TWO_CTA_OWNER)
    baseline = engines.mismatch(original, facts)
    reasons = {name: engines.mismatch(specs[name].capabilities, facts) for name in owners}
    assert (None in reasons.values()) == (baseline is None)
    offered = dict(zip(owners, (20609, 20628, 20629, 20630, 20631)))
    plans = heuristics.recommend("A", facts, offered)
    if baseline is not None:
        assert not plans
    elif profile in ("deterministic", "grouped_deterministic") and cc in ((10, 0), (10, 3)):
        assert plans[0].engine_id == 20629
        assert reasons[LEGACY_ROW] is not None
    elif profile == "plain" and cc in ((10, 0), (10, 3)):
        assert [p.engine_id for p in plans] == [20630, 20628, 20629]
    elif profile == "negative_scale":
        assert [p.engine_id for p in plans] == [20628]
    else:
        assert all(p.engine_id != 20629 for p in plans)


@requires_dsl
@requires_bwd_engine_row(PAIR_OWNER)
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("hkv,deterministic", [(1, True), (2, True), (2, False)])
@pytest.mark.parametrize("left", [None, 17])
def test_public_pair_record_rebuild_ignores_legacy_selector(monkeypatch, d, dtype_name, hkv, deterministic, left):
    import torch

    from cudnn.sdpa.bwd import api_dsl_sm100_d128 as legacy
    import test_sdpa_bwd_d128_pair_sm100 as helper
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph

    def forbidden(*args, **kwargs):
        raise AssertionError("a fixed pair record consulted the legacy selector")

    monkeypatch.setattr(legacy, "select_implementation", forbidden)
    dtype = getattr(torch, dtype_name)
    case = helper._grouped_case(dtype, d, True, left, 1, 2, hkv, 65, 129)
    record = case.graph.get_engine_and_knobs_at_index(case.graph._plan_index)
    assert record == (20629, {})
    expected = [case.tensors[name].clone() for name in ("dq", "dk", "dv")]
    graph, refs, outputs = _half_bwd_graph(
        b=1,
        hq=2,
        hkv=hkv,
        sq=65,
        skv=129,
        d=d,
        dt=dtype,
        use_causal_mask=True,
        use_deterministic_algorithm=deterministic,
        **({"diagonal_band_left_bound": left} if left is not None else {}),
    )
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plan(*record)
    graph.select_plan(0)
    graph.check_support()
    graph.build_plans()
    assert graph.get_workspace_size() == case.workspace.numel()
    assert graph._compiled_plans[0]._prepared.spec.artifact.implementation == "pair"
    refs.update(zip(("dq", "dk", "dv"), outputs))
    for name in ("dq", "dk", "dv"):
        case.tensors[name].fill_(float("nan"))
    case.workspace.fill_(0x5A)
    graph.execute({refs[name]: value for name, value in case.tensors.items()}, case.workspace)
    helper._check_case(case, allow_zero=True)
    for name, want in zip(("dq", "dk", "dv"), expected):
        assert torch.equal(case.tensors[name].view(torch.int16), want.view(torch.int16)), name
