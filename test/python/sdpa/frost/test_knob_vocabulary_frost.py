# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A FROST SDPA plan is described, named and replayed in the same
``(engine_id, {cudnn.knob_type: int})`` vocabulary as a backend plan.

The frost slots are opt-in (conftest sets the flag per test)."""

from __future__ import annotations

import pytest
import torch

import cudnn
from cudnn.engines import is_backend_engine, is_python_engine
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = pytest.mark.L0

ENGINE = "sdpa_fwd_prefill_sm100"


def _dense_sdpa_graph(handle, q, k, v, o, stats):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )
    Q, K, V = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
    out_t, stats_t = g.sdpa(name="sdpa", q=Q, k=K, v=V, is_inference=False, attn_scale=q.shape[-1] ** -0.5, use_causal_mask=True)
    out_t.set_output(True).set_dim(o.shape).set_stride(o.stride())
    stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(stats.shape).set_stride(stats.stride())
    g.validate()
    g.build_operation_graph()
    return g, (Q, K, V, out_t, stats_t)


@requires_pre_rubin_blackwell
@requires_dsl
def test_frost_plan_reports_names_and_replays_public_knobs():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    B, H, S, D = 2, 8, 1024, 128
    q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    o = torch.empty_like(q)
    stats = torch.empty(B, H, S, 1, device=dev, dtype=torch.float32)
    handle = cudnn.create_handle()
    kt = cudnn.knob_type

    g, _ = _dense_sdpa_graph(handle, q, k, v, o, stats)
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

    frost_idx = [i for i in range(g.get_execution_plan_count()) if is_python_engine(g.get_engine_and_knobs_at_index(i)[0])]
    backend_idx = [i for i in range(g.get_execution_plan_count()) if is_backend_engine(g.get_engine_and_knobs_at_index(i)[0])]
    assert frost_idx, "the frost sm100 engine should be offered for a dense bf16 d128 causal graph"
    assert backend_idx, "the backend entries stay in the list"

    # 1. One record shape for every plan: dict keyed by cudnn.knob_type, ints, never None.
    for i in range(g.get_execution_plan_count()):
        engine_id, knobs = g.get_engine_and_knobs_at_index(i)
        assert isinstance(knobs, dict), (i, knobs)
        assert all(isinstance(key, cudnn.knob_type) and isinstance(val, int) for key, val in knobs.items()), knobs

    # 2. A frost plan speaks the shared vocabulary: its tiles are the backend's TILE_M/TILE_N types.
    engine_id, knobs = g.get_engine_and_knobs_at_index(frost_idx[0])
    assert kt.TILE_M in knobs and kt.TILE_N in knobs, knobs
    assert not any(isinstance(key, str) for key in knobs)
    name = g.get_plan_name_at_index(frost_idx[0])
    assert name.startswith(ENGINE + "[") and "TILE_M=" in name and "TILE_N=" in name, name
    # frontend-only axes, when set, are frontend-only types
    for key in knobs:
        if key in (kt.SCHED_POLICY, kt.PACK_GQA, kt.SPLIT_KV):
            assert cudnn.is_frontend_knob_type(key)

    # 3. The record replays on a fresh graph and round-trips exactly.
    g2, (Q2, K2, V2, O2, Stats2) = _dense_sdpa_graph(handle, q, k, v, o, stats)
    g2.create_execution_plan(engine_id, knobs)
    last = g2.get_execution_plan_count() - 1
    assert g2.get_engine_and_knobs_at_index(last) == (engine_id, knobs)
    assert g2.get_plan_name_at_index(last) == name
    # integer keys, as a JSON cache would hand them back, are accepted too
    g3, _ = _dense_sdpa_graph(handle, q, k, v, o, stats)
    g3.create_execution_plan(engine_id, {int(key): val for key, val in knobs.items()})
    assert g3.get_engine_and_knobs_at_index(g3.get_execution_plan_count() - 1) == (engine_id, knobs)

    # 4. And it runs: the replayed frost plan matches the reference.
    g2.check_support()
    g2.build_plans()
    ws = torch.empty(max(g2.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    g2.execute_plan_at_index({Q2: q, K2: k, V2: v, O2: o, Stats2: stats}, ws, index=last, handle=handle)
    torch.cuda.synchronize()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(o, ref, atol=2e-2, rtol=2e-2)


@requires_pre_rubin_blackwell
@requires_dsl
def test_frost_engine_refuses_a_knob_it_has_no_axis_for():
    dev = torch.device("cuda")
    q = torch.randn(1, 4, 256, 128, device=dev, dtype=torch.bfloat16)
    o = torch.empty_like(q)
    stats = torch.empty(1, 4, 256, 1, device=dev, dtype=torch.float32)
    handle = cudnn.create_handle()
    g, _ = _dense_sdpa_graph(handle, q, q, q, o, stats)
    g.create_execution_plans([cudnn.heur_mode.A])
    frost = [g.get_engine_and_knobs_at_index(i)[0] for i in range(g.get_execution_plan_count()) if is_python_engine(g.get_engine_and_knobs_at_index(i)[0])]
    assert frost
    g2, _ = _dense_sdpa_graph(handle, q, q, q, o, stats)
    with pytest.raises(ValueError, match="not a tuning axis"):
        g2.create_execution_plan(frost[0], {cudnn.knob_type.SPLIT_K_SLC: 2})
