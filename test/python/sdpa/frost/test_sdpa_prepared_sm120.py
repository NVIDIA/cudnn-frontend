# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM120 dense, split and THD pointer launches bind fresh storage and runtime geometry."""

import math

import cudnn
import pytest
import torch

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_blackwell_geforce, requires_dsl, select_engine

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


def _case(dq=128, dv=128, dtype=torch.bfloat16, *, stats=True, padded=False, override=False, sq=129, skv=257, split=1, stats_log2=False):
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
    o, lse = g.sdpa(q=tensors["q"], k=tensors["k"], v=tensors["v"], attn_scale=dq**-0.5, generate_stats=stats, stats_use_log2=stats_log2)
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
    chosen = select_engine(g, engine_name(arch="sm120"), pack_gqa=False)
    if (chosen.knobs.split_kv or 1) != split:
        from dataclasses import replace

        g.create_execution_plan(chosen.engine_id, replace(chosen.knobs, split_kv=split))
        g.select_plan(len(g.plans) - 1)
    assert (g.plans[g._plan_index].knobs.split_kv or 1) == split
    g.check_support()
    g.build_plans()
    assert (g.get_workspace_size() > 0) == (split > 1)
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return g, vp, ws, bufs, tensors, storage


def _check(bufs, stats_log2=False):
    q, k, v = (bufs[n].double() for n in ("q", "k", "v"))
    scores = q @ k.repeat_interleave(2, 1).transpose(-1, -2) / math.sqrt(q.shape[-1])
    ref = scores.softmax(-1) @ v.repeat_interleave(2, 1)
    torch.testing.assert_close(bufs["o"].double(), ref, atol=2e-3, rtol=2e-2)
    if "lse" in bufs:
        torch.testing.assert_close(bufs["lse"].double(), scores.logsumexp(-1) * (math.log2(math.e) if stats_log2 else 1), atol=3e-4, rtol=3e-4)


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


@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("split", [1, 2])
def test_sm120_prepared_bounded_geometry_override(d, split, monkeypatch):
    import cutlass.cute as cute

    g, vp, ws, bufs, tensors, _ = _case(d, d, override=True, padded=True, split=split)
    plan = g._compiled_plans[g._plan_index]
    assert plan._prepared is not None
    owner = plan._prepared.spec.owner
    uids, shapes, strides = [], [], []
    for name, t in tensors.items():
        length = 113 if name in ("k", "v") else (129 if split > 1 else 64)
        bufs[name] = bufs[name][: (2 if split > 1 else 1), :, :length]
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


@pytest.mark.parametrize("split", [1, 4])
def test_sm120_prepared_graph_and_adapter_bind_the_same_frame(monkeypatch, split):
    g, vp, ws, bufs, _, _ = _case(split=split)
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


@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("stats,stats_log2", [(False, False), (True, False), (True, True)])
def test_sm120_prepared_split_capture_rebind(d, stats, stats_log2, monkeypatch):
    """First execute captures both launches; new storage and inputs remain visible."""
    import cutlass.cute as cute

    g, vp, ws, bufs, tensors, storage = _case(d, d, stats=stats, padded=True, sq=17, skv=1024, split=4, stats_log2=stats_log2)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared is not None and prepared.spec.combine is not None
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute must not compile"))
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "o"):
                storage[name] = storage[name].clone()
                bufs[name] = storage[name][..., :d].transpose(1, 2)
                vp[tensors[name]] = bufs[name]
            ws = torch.empty_like(ws)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            g.execute(vp, ws)
        torch.cuda.current_stream().wait_stream(stream)
        bufs["v"].mul_(0.7)
        bufs["o"].fill_(float("nan"))
        captured.replay()
        _check(bufs, stats_log2)
        assert torch.all(storage["o"][..., d:] == 123)
        if stats:
            assert torch.isnan(storage["lse"][..., 1::2]).all()
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        g.execute(vp, ws)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("binder", ["native", "python"])
def test_sm120_prepared_thd_capture_rebind(d, binder, monkeypatch):
    """Both binders rebind lengths, metadata workspace and packed storage per call."""
    import cutlass.cute as cute
    from cudnn.sdpa.fwd.prepared import PreparedThdLaunch
    from test_sdpa_prepared_thd import _buffers, _pack, _reference, _thd_graph

    b, ql, kl, hq, hk = 3, 17, 65, 4, 2
    g, tensors = _thd_graph(b, ql, kl, hq, hk, d, arch="sm120")
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert isinstance(prepared, PreparedThdLaunch)
    assert prepared.spec.native is not None
    if binder == "python":
        prepared.spec.native = None
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute must not compile"))
    for iteration in range(2):
        bufs = _buffers(b, ql, kl, hq, hk, d, seed=iteration)
        ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
        pack = _pack(tensors, bufs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            g.execute(pack, ws)
        torch.cuda.current_stream().wait_stream(stream)
        bufs["v"].mul_(0.5)
        bufs["o"].fill_(float("nan"))
        captured.replay()
        ref, lse = _reference(bufs, b, ql, kl, hq, hk, d)
        torch.testing.assert_close(bufs["o"].float(), ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(bufs["lse"], lse, atol=1e-3, rtol=1e-3)
        # Force the standalone adapter behind the same graph and caller workspace.
        plan = g._compiled_plans[g._plan_index]
        plan._prepared, plan.takes_variant_pack = None, False
        try:
            bufs["o"].fill_(float("nan"))
            g.execute(pack, ws)
            torch.testing.assert_close(bufs["o"].float(), ref, atol=2e-2, rtol=2e-2)
        finally:
            plan._prepared, plan.takes_variant_pack = prepared, True
        torch.cuda.set_sync_debug_mode("error")
        try:
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            g.execute(pack, ws)
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        finally:
            torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("d", [128, 256, 512])
def test_sm120_prepared_thd_bounded_batch_override(d, monkeypatch):
    """Metadata producer and persistent consumer agree on a shrinking live batch."""
    import cutlass.cute as cute
    from test_sdpa_prepared_thd import _buffers, _pack, _reference, _thd_graph

    b, ql, kl, hq, hk = 4, 9, 33, 4, 2
    g, tensors = _thd_graph(b, ql, kl, hq, hk, d, arch="sm120", override_enabled=True)
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    owner = g._compiled_plans[g._plan_index]._prepared.spec.owner
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("override must reuse the artifact"))
    for live_b in (2, b):
        bufs = _buffers(live_b, ql, kl, hq, hk, d, seed=live_b)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        names = ("q", "k", "v", "o", "stats", "cu_q", "cu_kv", "off_q", "off_kv", "off_lse")
        shapes = [[live_b, hq, ql, d], [live_b, hk, kl, d], [live_b, hk, kl, d], [live_b, hq, ql, d], [live_b, hq, ql, 1]]
        shapes += [[live_b + 1, 1, 1, 1]] * 5
        g.execute(
            _pack(tensors, bufs),
            ws,
            override_uids=[tensors[n].get_uid() for n in names],
            override_shapes=shapes,
            override_strides=[tensors[n].get_stride() for n in names],
        )
        ref, lse = _reference(bufs, live_b, ql, kl, hq, hk, d)
        torch.testing.assert_close(bufs["o"].float(), ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(bufs["lse"], lse, atol=1e-3, rtol=1e-3)
    assert g._compiled_plans[g._plan_index]._prepared.spec.owner is owner


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d_qk,d_v", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("binder", ["native", "python"])
def test_sm120_thd_output_stride_int64(d_qk, d_v, binder, monkeypatch):
    """Every half THD arch/flavor preserves wide strides through device descriptor setup."""
    import math

    from cudnn.sdpa.fwd.engines import engine_name

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    b, hq, hk, ql, kl = 2, 4, 2, 1, 64
    row_stride = 2**32 + hq * d_v
    if torch.cuda.mem_get_info()[0] < 2 * row_stride + 2**30:
        pytest.skip("wide physical row-stride regression needs 9 GiB free")
    bf16, i32 = cudnn.data_type.BFLOAT16, cudnn.data_type.INT32
    graph = cudnn.pygraph(
        io_data_type=bf16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, is_override_shape_enabled=True
    )

    def lengths(name):
        return graph.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name=name)

    cu_q, cu_kv = lengths("cu_q"), lengths("cu_kv")
    off_q, off_k, off_v, off_o, off_lse = [lengths(n) for n in ("off_q", "off_k", "off_v", "off_o", "off_lse")]

    def operand(name, h, s, d, offset):
        return graph.tensor(dim=[b, h, s, d], stride=[s * h * d, d, h * d, 1], data_type=bf16, name=name).set_ragged_offset(offset)

    q, k, v = operand("q", hq, ql, d_qk, off_q), operand("k", hk, kl, d_qk, off_k), operand("v", hk, kl, d_v, off_v)
    o, stats = graph.sdpa(
        q=q,
        k=k,
        v=v,
        generate_stats=True,
        attn_scale=1 / math.sqrt(d_qk),
        use_padding_mask=True,
        cu_seq_len_q=cu_q,
        cu_seq_len_kv=cu_kv,
        max_total_seq_len_q=b * ql,
        max_total_seq_len_kv=b * kl,
    )
    o.set_output(True).set_dim([b, hq, ql, d_v]).set_stride([ql * hq * d_v, d_v, hq * d_v, 1]).set_ragged_offset(off_o)
    stats.set_output(True).set_dim([b, hq, ql, 1]).set_stride([ql * hq, 1, hq, 1]).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(off_lse)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    want = engine_name(arch="sm120")
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    graph.select_plan(next(i for i, name in enumerate(names) if name == want or name.startswith(want + "[")))
    graph.check_support()
    graph.build_plans()
    spec = graph._compiled_plans[graph._plan_index]._prepared.spec
    if binder == "native":
        assert spec.native is not None
    elif hasattr(spec, "native"):
        spec.native = None  # Compare the independent Python binder with the same compiled host.
    q_buf = torch.zeros((b * ql, hq, d_qk), device="cuda", dtype=torch.bfloat16)
    k_buf = torch.zeros((b * kl, hk, d_qk), device="cuda", dtype=torch.bfloat16)
    v_buf = torch.ones((b * kl, hk, d_v), device="cuda", dtype=torch.bfloat16)
    v_buf[kl:] *= 2
    o_buf = torch.empty_strided((b * ql, hq, d_v), (row_stride, d_v, 1), device="cuda", dtype=torch.bfloat16)
    lse_buf = torch.empty((b * ql, hq), device="cuda", dtype=torch.float32)
    cq = torch.arange(b + 1, device="cuda", dtype=torch.int32) * ql
    ck = torch.arange(b + 1, device="cuda", dtype=torch.int32) * kl
    pack = {
        q: q_buf,
        k: k_buf,
        v: v_buf,
        o: o_buf,
        stats: lse_buf,
        cu_q: cq,
        cu_kv: ck,
        off_q: cq * hq * d_qk,
        off_k: ck * hk * d_qk,
        off_v: ck * hk * d_v,
        off_o: cq * hq * d_v,
        off_lse: cq * hq,
    }
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    overrides = dict(override_uids=[o.get_uid()], override_shapes=[[b, hq, ql, d_v]], override_strides=[[hq * d_v, d_v, row_stride, 1]])
    expected = torch.ones((b * ql, hq, d_v), device="cuda", dtype=torch.bfloat16)
    expected[ql:] *= 2
    for replay in (False, True):
        if replay:
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                graph.execute(pack, workspace, **overrides)
            v_buf.mul_(0.5)
            expected.mul_(0.5)
        o_buf.fill_(float("nan"))
        lse_buf.fill_(float("nan"))
        if replay:
            captured.replay()
        else:
            graph.execute(pack, workspace, **overrides)
        torch.testing.assert_close(o_buf, expected, atol=0, rtol=0)
        torch.testing.assert_close(lse_buf, torch.full_like(lse_buf, math.log(kl)), atol=2e-6, rtol=0)
