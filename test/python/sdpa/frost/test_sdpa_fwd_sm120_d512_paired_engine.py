# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public SM120 wide-head half records fix the body across rebuild, rebind and retained capture."""

import math

import cudnn
import pytest
import torch

from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]

_NAME, _ID, _LEGACY_ID = "sdpa_fwd_sm120_d512_paired", 20526, 20505


def _graph(dq, dv, dtype, *, record=None, split=None, sq=97, skv=301):
    torch.manual_seed(5120 + dq)
    io = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bufs, tensors, vp = {}, {}, {}
    for name, heads, seq, dim in (("q", 8, sq, dq), ("k", 2, skv, dq), ("v", 2, skv, dv), ("o", 8, sq, dv)):
        bufs[name] = (torch.randn(2, seq, heads, dim, dtype=dtype, device="cuda") * 0.4).transpose(1, 2)
        if name != "o":
            tensors[name] = g.tensor_like(bufs[name], name=name)
            vp[tensors[name]] = bufs[name]
    o, lse = g.sdpa(q=tensors["q"], k=tensors["k"], v=tensors["v"], attn_scale=dq**-0.5, generate_stats=True, use_causal_mask=True)
    o.set_output(True).set_dim(bufs["o"].shape).set_stride(bufs["o"].stride())
    bufs["lse"] = torch.full((2, 8, sq, 1), float("nan"), device="cuda")
    lse.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(bufs["lse"].shape).set_stride(bufs["lse"].stride())
    tensors.update(o=o, lse=lse)
    vp[o], vp[lse] = bufs["o"], bufs["lse"]
    g.validate()
    g.build_operation_graph()
    if record is None:
        g.create_execution_plans([cudnn.heur_mode.A])
        index = next(i for i in range(len(g.plans)) if g.get_plan_name_at_index(i).startswith(_NAME))
        if split is not None:
            engine_id, knobs = g.get_engine_and_knobs_at_index(index)
            g.create_execution_plan(engine_id, {**knobs, cudnn.knob_type.SPLIT_KV: split})
            index = len(g.plans) - 1
        g.select_plan(index)
    else:
        g.create_execution_plan(*record)
        g.select_plan(0)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return g, vp, ws, bufs, tensors


def _check(bufs):
    q, k, v = (bufs[n].double() for n in ("q", "k", "v"))
    group = q.shape[1] // k.shape[1]
    scores = q @ k.repeat_interleave(group, 1).transpose(-1, -2) / math.sqrt(q.shape[-1])
    s_q, s_kv = scores.shape[-2:]
    scores.masked_fill_(torch.ones(s_q, s_kv, dtype=torch.bool, device=scores.device).triu(1), -math.inf)
    torch.testing.assert_close(bufs["o"].double(), scores.softmax(-1) @ v.repeat_interleave(group, 1), atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(bufs["lse"][..., 0].double(), scores.logsumexp(-1), atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("dims", [(512, 512), (576, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("split", [1, 3])
def test_public_record_rebuild_rebind_and_retained_capture(monkeypatch, dims, dtype, split):
    from cudnn.sdpa.fwd import heuristics

    g, vp, ws, bufs, tensors = _graph(*dims, dtype, split=split)
    record = g.get_engine_and_knobs_at_index(g._plan_index)
    assert record[0] == _ID and g.selected_engine.name == _NAME
    compiled = g._compiled_plans[g._plan_index]._compiled
    assert compiled.kernel_template == "prefill_d512_f16"
    metadata = compiled.implementation_config
    assert (metadata["query_tile"], metadata["kv_tile"], metadata["split_kv"]) == (64, 32, split)
    g.execute(vp, ws)
    _check(bufs)

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    rebuilt, rvp, rws, rbufs, rtensors = _graph(*dims, dtype, record=record)
    assert rebuilt.get_engine_and_knobs_at_index(rebuilt._plan_index) == record
    assert rebuilt._compiled_plans[rebuilt._plan_index]._compiled.implementation_config == metadata

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture, stream=stream):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                rebuilt.execute(rvp, rws)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.current_stream().wait_stream(stream)
        for scale in (0.5, 1.7):
            rbufs["v"].mul_(scale)
            rbufs["o"].fill_(float("nan"))
            rbufs["lse"].fill_(float("nan"))
            capture.replay()
            _check(rbufs)
    finally:
        capture.reset()

    # Eager execution rebinds caller storage independently of the retained capture's pointers.
    for name in ("q", "k", "v", "o", "lse"):
        rbufs[name] = rbufs[name].clone()
        rvp[rtensors[name]] = rbufs[name]
    rbufs["o"].fill_(float("nan"))
    rebuilt.execute(rvp, torch.empty_like(rws))
    _check(rbufs)


@pytest.mark.parametrize("dims", [(512, 512), (576, 512)])
def test_retired_legacy_record_is_declined_without_substitution(dims):
    g, *_ = _graph(*dims, torch.bfloat16)
    _, knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError), match="owned by"):
        _graph(*dims, torch.bfloat16, record=(_LEGACY_ID, knobs))


def test_padded_envelope_record_keeps_the_legacy_row():
    g, vp, ws, bufs, _ = _graph(384, 320, torch.bfloat16, record=(_LEGACY_ID, {}))
    assert g.selected_engine.name == "sdpa_fwd_prefill_sm120"
    assert g._compiled_plans[g._plan_index]._compiled.implementation_config is None
    g.execute(vp, ws)
    _check(bufs)
