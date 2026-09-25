# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM120 dense pointer launches preserve layouts and bind fresh storage."""

import math

import cudnn
import pytest
import torch

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_blackwell_geforce, requires_dsl, select_engine

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


def _case(dq=128, dv=128, dtype=torch.bfloat16, *, stats=True, padded=False, override=False, sq=129, skv=257):
    torch.manual_seed(873)
    io = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    g = cudnn.pygraph(
        io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, is_override_shape_enabled=override
    )
    bufs, tensors, storage, vp = {}, {}, {}, {}
    for name, heads, seq, dim in (("q", 4, sq, dq), ("k", 2, skv, dq), ("v", 2, skv, dv), ("o", 4, sq, dv)):
        raw = torch.randn(2, seq, heads, dim + (8 if padded else 0), dtype=dtype, device="cuda") * 0.4
        if name == "o":
            raw.fill_(123)
        buf = raw[..., :dim].transpose(1, 2)
        storage[name], bufs[name] = raw, buf
        if name != "o":
            t = g.tensor_like(buf, name=name)
            tensors[name], vp[t] = t, buf
    o, lse = g.sdpa(q=tensors["q"], k=tensors["k"], v=tensors["v"], attn_scale=dq**-0.5, generate_stats=stats)
    o.set_output(True).set_dim(bufs["o"].shape).set_stride(bufs["o"].stride())
    tensors["o"], vp[o] = o, bufs["o"]
    if stats:
        raw = torch.full((2, 4, sq * 2), float("nan"), device="cuda")
        buf = raw[..., ::2]
        storage["lse"], bufs["lse"] = raw, buf
        lse.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([2, 4, sq, 1]).set_stride([4 * sq * 2, sq * 2, 2, 1])
        tensors["lse"], vp[lse] = lse, buf
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    knobs = select_engine(g, engine_name(arch="sm120"), pack_gqa=False, split_kv=1).knobs
    assert (knobs.split_kv or 1) == 1
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == 0
    ws = torch.empty(1, device="cuda", dtype=torch.uint8)
    return g, vp, ws, bufs, tensors, storage


def _check(bufs):
    q, k, v = (bufs[n].float() for n in ("q", "k", "v"))
    scores = q @ k.repeat_interleave(2, 1).transpose(-1, -2) / math.sqrt(q.shape[-1])
    ref = scores.softmax(-1) @ v.repeat_interleave(2, 1)
    torch.testing.assert_close(bufs["o"].float(), ref, atol=2e-3, rtol=2e-2)
    if "lse" in bufs:
        torch.testing.assert_close(bufs["lse"], scores.logsumexp(-1), atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("dq,dv", [(96, 80), (128, 128), (256, 256), (384, 320)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sm120_prepared_rebind_strides_and_storage(dq, dv, dtype):
    g, vp, ws, bufs, tensors, storage = _case(dq, dv, dtype, padded=True)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "o"):
                dim = dq if name in ("q", "k") else dv
                storage[name] = storage[name].clone()
                bufs[name] = storage[name][..., :dim].transpose(1, 2)
                vp[tensors[name]] = bufs[name]
            bufs["q"].mul_(0.7)
            bufs["k"].mul_(1.5)
            bufs["v"].mul_(0.2)
            storage["lse"] = storage["lse"].clone()
            bufs["lse"] = storage["lse"][..., ::2]
            vp[tensors["lse"]] = bufs["lse"]
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        g.execute(vp, ws)
        _check(bufs)
        assert torch.all(storage["o"][..., dv:] == 123)
        assert torch.isnan(storage["lse"][..., 1::2]).all()


@pytest.mark.parametrize("d", [128, 512])
@pytest.mark.parametrize("stats", [False, True])
def test_sm120_prepared_capture_first_execute(d, stats, monkeypatch):
    import cutlass.cute as cute

    g, vp, ws, bufs, _, _ = _case(d, d, stats=stats)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute must not compile"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        g.execute(vp, ws)
    torch.cuda.current_stream().wait_stream(stream)
    for scale in (0.5, 1.7):
        bufs["v"].mul_(scale)
        bufs["o"].fill_(float("nan"))
        graph.replay()
        _check(bufs)
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        g.execute(vp, ws)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("d", [128, 512])
def test_sm120_prepared_bounded_geometry_override(d, monkeypatch):
    import cutlass.cute as cute

    g, vp, ws, bufs, tensors, _ = _case(d, d, override=True, padded=True)
    plan = g._compiled_plans[g._plan_index]
    assert plan._prepared is not None
    owner = plan._prepared.spec.owner
    uids, shapes, strides = [], [], []
    for name, t in tensors.items():
        length = 113 if name in ("k", "v") else 64
        bufs[name] = bufs[name][:1, :, :length]
        vp[t] = bufs[name]
        shape, stride = list(bufs[name].shape), list(bufs[name].stride())
        if name == "lse":
            shape.append(1)
            stride.append(1)
        uids.append(t.get_uid())
        shapes.append(shape)
        strides.append(stride)
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("override must reuse the artifact"))
    g.execute(vp, ws, override_uids=uids, override_shapes=shapes, override_strides=strides)
    assert plan._prepared.spec.owner is owner
    _check(bufs)


def test_sm120_prepared_graph_and_adapter_bind_the_same_frame(monkeypatch):
    g, vp, ws, bufs, _, _ = _case()
    plan = g._compiled_plans[g._plan_index]
    prepared = plan._prepared
    assert prepared is not None
    frames = []
    original = prepared.spec.fn

    def record(*args):
        frames.append(args)
        return original(*args)

    monkeypatch.setattr(prepared.spec, "fn", record)
    g.execute(vp, ws)
    _check(bufs)
    plan._prepared, plan.takes_variant_pack = None, False
    try:
        g.execute(vp, ws)
        _check(bufs)
    finally:
        plan._prepared, plan.takes_variant_pack = prepared, True
    assert len(frames) == 2
    for i, name in enumerate(prepared.spec.order):
        if name != "stream":
            assert str(frames[0][i]) == str(frames[1][i]), name
