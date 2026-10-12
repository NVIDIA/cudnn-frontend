# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-all must prepare both providers without losing the selected plan."""

import pytest
import torch

from cudnn.engines import is_backend_engine
from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import _SM, requires_dsl, select_engine

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM not in (90, 100, 103), reason="native/FROST SDPA parity on Hopper and Blackwell")]


@pytest.mark.parametrize("selected_provider", ["backend", "frost"])
def test_build_all_preserves_selection_and_executes_both_providers(selected_provider, monkeypatch):
    import cudnn

    # Hopper exposes the d512 FROST template; Blackwell also serves d128.
    head_dim = 512 if _SM == 90 else 128
    q = torch.zeros(1, 128, 2, head_dim, device="cuda", dtype=torch.float16).transpose(1, 2)
    k = torch.zeros_like(q)
    # With zero logits, the exact answer is the mean of each V head.
    v = ((torch.arange(q.numel(), device="cuda") % 7).reshape(1, 128, 2, head_dim) / 8).half().transpose(1, 2)
    o = torch.empty_like(q)
    expected = v.float().mean(dim=2, keepdim=True).expand_as(o)

    def make_graph():
        g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        qt, kt, vt = [g.tensor_like(t) for t in (q, k, v)]
        ot, _ = g.sdpa(q=qt, k=kt, v=vt, generate_stats=False, attn_scale=1.0)
        ot.set_output(True).set_dim(list(o.shape)).set_stride(list(o.stride()))
        g.validate()
        g.build_operation_graph()
        return g, {qt: q, kt: k, vt: v, ot: o}

    source, _ = make_graph()
    source.create_execution_plans([cudnn.heur_mode.A])
    native = None
    for i in range(source.get_execution_plan_count()):
        record = source.get_engine_and_knobs_at_index(i)
        if not is_backend_engine(record[0]):
            continue
        try:
            source.build_plan_at_index(i)
        except cudnn.cudnnGraphNotSupportedError:
            continue
        native = record
        break
    assert native is not None, "the representative graph must offer a buildable backend plan"
    select_engine(source, engine_name(arch="sm90" if _SM == 90 else "sm100"))
    frost = source.get_engine_and_knobs_at_index(source._plan_index)
    source.build_plans()

    # Put the selected provider second so ALL must go back to the first entry.
    records = [native, frost] if selected_provider == "frost" else [frost, native]
    graph, bindings = make_graph()
    for engine, knobs in records:
        graph.create_execution_plan(engine, knobs)
    graph.select_plan(1)
    graph.build_plans(cudnn.build_plan_policy.ALL)
    assert graph._plan_index == 1
    assert graph.get_engine_and_knobs_at_index(1) == records[1]
    monkeypatch.setattr(graph, "_build_plan_at", lambda *a, **kw: pytest.fail("ALL left a candidate unbuilt"))
    workspace = torch.empty(max(1, *(graph.get_workspace_size_plan_at_index(i) for i in range(2))), device="cuda", dtype=torch.uint8)
    for index in (0, 1):
        o.fill_(float("nan"))
        graph.execute_plan_at_index(bindings, workspace, index=index)
        torch.testing.assert_close(o.float(), expected, atol=5e-4, rtol=0)
    # At-index execution must not change the caller's selection either.
    assert graph._plan_index == 1
    o.fill_(float("nan"))
    graph.execute(bindings, workspace)
    torch.testing.assert_close(o.float(), expected, atol=5e-4, rtol=0)
