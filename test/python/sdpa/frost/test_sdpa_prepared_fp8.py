# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Prepared FP8 launches rebind device scales and storage through the public graph API."""

import math

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, select_engine

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


def _case(
    *,
    thd=False,
    stats=True,
    amax=True,
    override=False,
    dtype=torch.float8_e4m3fn,
    b=2,
    sq=128,
    skv=128,
    d=128,
    dv=None,
    padded=False,
    output_dtype=torch.bfloat16,
    output_padding=0,
):
    dv = d if dv is None else dv
    torch.manual_seed(827)
    hq, hk = 4, 2
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=override,
    )
    buffers, tensors, vp = {}, {}, {}
    input_type = cudnn.data_type.FP8_E4M3 if dtype == torch.float8_e4m3fn else cudnn.data_type.FP8_E5M2
    for name, h, seq, dim in (("q", hq, sq, d), ("k", hk, skv, d), ("v", hk, skv, dv)):
        raw = (torch.randn(b, seq, h, dim + (16 if padded else 0), device="cuda") * 0.4).to(dtype)
        buf = raw.reshape(b * seq, h, dim) if thd else raw[..., :dim].transpose(1, 2)
        strides = [seq * h * dim, dim, h * dim, 1] if thd else list(buf.stride())
        t = g.tensor(dim=[b, h, seq, dim], stride=strides, data_type=input_type, name=name)
        buffers[name], tensors[name], vp[t] = buf, t, buf
    kwargs = {}
    if thd:
        for name, seq, h in (("q", sq, hq), ("kv", skv, hk)):
            cu = torch.arange(b + 1, device="cuda", dtype=torch.int32) * seq
            lens = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="cu_" + name)
            off = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="off_" + name)
            vp[lens], vp[off] = cu, cu * h * d
            tensors["cu_" + name], tensors["off_" + name] = lens, off
            for role in (("q",) if name == "q" else ("k", "v")):
                tensors[role].set_ragged_offset(off)
            if name == "kv" and dv != d:
                off_v = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="off_v")
                tensors["v"].set_ragged_offset(off_v)
                tensors["off_v"], vp[off_v] = off_v, cu * hk * dv
            kwargs["cu_seq_len_" + name] = lens
        kwargs["use_padding_mask"] = True
    for name, val in (("descale_q", 0.8), ("descale_k", 0.9), ("descale_v", 0.7), ("descale_s", 1.0), ("scale_s", 1.0), ("scale_o", 1.3)):
        t = g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT, name=name)
        buf = torch.full((1,), val, device="cuda", dtype=torch.float32)
        buffers[name], tensors[name], vp[t] = buf, t, buf
        kwargs[name] = t
    o, lse, _, am = g.sdpa_fp8(q=tensors["q"], k=tensors["k"], v=tensors["v"], attn_scale=1 / math.sqrt(d), generate_stats=stats, **kwargs)
    out_type = {
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float16: cudnn.data_type.HALF,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
        torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
    }[output_dtype]
    o.set_output(True).set_data_type(out_type).set_dim([b, hq, sq, dv]).set_stride([sq * hq * dv, dv, hq * dv, 1])
    out = torch.empty((b * sq, hq, dv) if thd else (b, sq, hq, dv), device="cuda", dtype=output_dtype)
    if not thd:
        out = out.transpose(1, 2)
    if output_padding:
        assert not thd
        storage = torch.full((b, sq, hq, dv + output_padding), 12, device="cuda", dtype=output_dtype)
        out = storage[..., :dv].transpose(1, 2)
        buffers["o_storage"] = storage
        o.set_stride(out.stride())
    if thd:
        if dv == d:
            o.set_ragged_offset(tensors["off_q"])
        else:
            off_o = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="off_o")
            o.set_ragged_offset(off_o)
            tensors["off_o"], vp[off_o] = off_o, torch.arange(b + 1, device="cuda", dtype=torch.int32) * sq * hq * dv
    buffers["o"], tensors["o"], vp[o] = out, o, out
    if stats:
        lse.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, hq, sq, 1])
        if thd:
            lse.set_stride([sq * hq, 1, hq, 1])
            off = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
            lse.set_ragged_offset(off)
            vp[off] = torch.arange(b + 1, device="cuda", dtype=torch.int32) * sq * hq
            tensors["off_lse"] = off
            lse_buf = torch.empty((b * sq, hq), device="cuda")
        else:
            lse.set_stride([hq * sq, sq, 1, 1])
            lse_buf = torch.empty((b, hq, sq), device="cuda")
        buffers["lse"], tensors["lse"], vp[lse] = lse_buf, lse, lse_buf
    if amax:
        am.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1])
        buffers["amax_o"] = torch.empty((1,), device="cuda")
        tensors["amax_o"], vp[am] = am, buffers["amax_o"]
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, engine_name(fp8=True), split_kv=1)
    g.check_support()
    g.build_plans()
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return g, vp, workspace, buffers, tensors


def _reference(bufs, *, thd, b=2, sq=128, skv=128):
    def dense(x, h, seq):
        return x.reshape(b, seq, h, x.shape[-1]).transpose(1, 2).float() if thd else x.float()

    q = dense(bufs["q"], 4, sq) * bufs["descale_q"]
    k = dense(bufs["k"], 2, skv).repeat_interleave(2, 1) * bufs["descale_k"]
    v = dense(bufs["v"], 2, skv).repeat_interleave(2, 1) * bufs["descale_v"]
    scores = q @ k.transpose(-1, -2) / math.sqrt(bufs["q"].shape[-1])
    out = scores.softmax(-1) @ v
    lse = scores.logsumexp(-1)
    am = out.abs().amax()
    out *= bufs["scale_o"]
    if thd:
        out = out.transpose(1, 2).reshape(b * sq, 4, bufs["v"].shape[-1])
        lse = lse.transpose(1, 2).reshape(b * sq, 4)
    return out, lse, am


def _check(bufs, *, thd, b=2, sq=128, skv=128):
    ref, stats, am = _reference(bufs, thd=thd, b=b, sq=sq, skv=skv)
    atol = 0.03
    if bufs["o"].dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # Same quantization-floor rule as the existing FP8 output suite.
        floor = (ref - ref.to(bufs["o"].dtype).float()).abs().max().item()
        atol = max(atol, 3.0 * floor)
    torch.testing.assert_close(bufs["o"].float(), ref, atol=atol, rtol=0.03)
    if "lse" in bufs:
        torch.testing.assert_close(bufs["lse"], stats, atol=0.01, rtol=0.01)
    if "amax_o" in bufs:
        torch.testing.assert_close(bufs["amax_o"][0], am, atol=0.03, rtol=0.03)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("stats,amax", [(True, True), (False, False)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_prepared_fp8_rebind_scales_and_buffers(thd, stats, amax, dtype, d, dv, output_dtype, monkeypatch):
    """All scalar output dtypes use the pointer entry, including every THD route."""
    from cudnn.sdpa.fwd.kernels.sm100 import fp8_host

    def legacy_fake(*args, **kwargs):
        pytest.fail("prepared launch reentered the legacy tensor compiler")

    monkeypatch.setattr(fp8_host, "make_fake_aux", legacy_fake)
    g, vp, ws, bufs, tensors = _case(d=d, dv=dv, thd=thd, stats=stats, amax=amax, dtype=dtype, output_dtype=output_dtype)
    plan = g._compiled_plans[g._plan_index]
    assert plan._prepared is not None, "FP8 must use the normalized prepared executor"
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "o", "descale_q", "descale_k", "descale_v", "scale_o"):
                bufs[name] = bufs[name].clone()
                vp[tensors[name]] = bufs[name]
            bufs["descale_q"].fill_(1.9)
            bufs["descale_k"].fill_(1.7)
            bufs["descale_v"].fill_(0.3)
            bufs["scale_o"].fill_(1.8)
            for name in ("lse", "amax_o"):
                if name in bufs:
                    bufs[name] = bufs[name].clone()
                    vp[tensors[name]] = bufs[name]
            ws = torch.empty_like(ws)
        bufs["o"].fill_(float("nan"))
        if amax:
            bufs["amax_o"].fill_(999)
        g.execute(vp, ws)
        _check(bufs, thd=thd)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_prepared_fp8_capture_replay_reads_current_scales(thd, monkeypatch, d, dv, output_dtype):
    g, vp, ws, bufs, _ = _case(d=d, dv=dv, thd=thd, output_dtype=output_dtype)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    torch.cuda.synchronize()
    import cutlass.cute as cute

    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute must not compile"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda.set_sync_debug_mode("error")
        try:
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            g.execute(vp, ws)
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        finally:
            torch.cuda.set_sync_debug_mode("default")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            g.execute(vp, ws)
    torch.cuda.current_stream().wait_stream(stream)
    for value in (0.4, 1.1):
        bufs["descale_v"].fill_(value)
        bufs["o"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        graph.replay()
        _check(bufs, thd=thd)


@pytest.mark.parametrize("thd", [False, True])
def test_prepared_fp8_declared_override_is_admitted(thd):
    g, vp, ws, bufs, _ = _case(thd=thd, override=True)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    _check(bufs, thd=thd)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_prepared_fp8_bounded_batch_override(thd, monkeypatch, d, dv):
    g, vp, ws, bufs, tensors = _case(d=d, dv=dv, thd=thd, override=True)
    plan = g._compiled_plans[g._plan_index]
    assert plan._prepared is not None
    owner = plan._prepared.spec.owner
    uids, shapes, strides = [], [], []
    for name in ("q", "k", "v", "o", "lse"):
        t = tensors[name]
        bufs[name] = bufs[name][: 128 if thd else 1].clone()
        vp[t] = bufs[name]
        shape = list(t.get_dim())
        shape[0] = 1
        uids.append(t.get_uid())
        shapes.append(shape)
        strides.append(list(t.get_stride()))
    if thd:
        for name in ("cu_q", "cu_kv", "off_q", "off_kv", "off_lse") + (("off_v", "off_o") if d != dv else ()):
            t = tensors[name]
            vp[t] = vp[t][:2].clone()
            uids.append(t.get_uid())
            shapes.append([2, 1, 1, 1])
            strides.append([1, 1, 1, 1])
    import cutlass.cute as cute

    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("an override must reuse the artifact"))
    if not thd and plan._prepared.spec.lpt_grid_fixed:
        monkeypatch.setattr(plan._prepared.spec, "fn", lambda *a: pytest.fail("invalid override must not launch"))
        with pytest.raises(ValueError, match="grouped LPT schedule"):
            g.execute(vp, ws, override_uids=uids, override_shapes=shapes, override_strides=strides)
        return
    g.execute(vp, ws, override_uids=uids, override_shapes=shapes, override_strides=strides)
    assert plan._prepared.spec.owner is owner
    _check(bufs, thd=thd, b=1)


@pytest.mark.parametrize("role,kind", [("descale_q", "span"), ("scale_o", "alignment"), ("amax_o", "alias"), ("amax_o", "bare_alias")])
def test_prepared_fp8_invalid_scalars_do_not_launch(role, kind, monkeypatch):
    from cudnn.sdpa.fwd import prepared as prep

    g, vp, ws, bufs, _ = _case()
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    facts = {name: prep.facts_of_tensor(t) for name, t in bufs.items()}
    if kind == "span":
        facts[role] = facts[role]._replace(span=0)
    elif kind == "alignment":
        facts[role] = facts[role]._replace(ptr=facts[role].ptr + 1)
    else:
        facts[role] = facts[role]._replace(ptr=facts["scale_o"].ptr)
        if kind == "bare_alias":
            facts[role] = facts[role]._replace(span=-1)
            facts["scale_o"] = facts["scale_o"]._replace(span=-1)
    monkeypatch.setattr(spec, "fn", lambda *a: pytest.fail("invalid scalar reached the kernel"))
    monkeypatch.setattr(prep._buffers, "memset_zero_async", lambda *a: pytest.fail("invalid scalar mutated output"))
    with pytest.raises(ValueError, match=role):
        prep.execute_quantized(spec, facts, ws.data_ptr(), None, 0)


@pytest.mark.parametrize("bare", [False, True])
def test_prepared_fp8_workspace_overlap_is_rejected(bare, monkeypatch):
    from cudnn.sdpa.fwd import prepared as prep

    g, vp, ws, bufs, _ = _case()
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    facts = {name: prep.facts_of_tensor(t) for name, t in bufs.items()}
    if bare:
        facts["q"] = facts["q"]._replace(span=-1)
    monkeypatch.setattr(spec, "fn", lambda *a: pytest.fail("aliased workspace reached the kernel"))
    with pytest.raises(ValueError, match="workspace overlaps"):
        prep.execute_quantized(spec, facts, facts["q"].ptr, None, 0)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_prepared_fp8_graph_and_adapter_bind_the_same_frame(thd, monkeypatch, d, dv):
    g, vp, ws, bufs, _ = _case(d=d, dv=dv, thd=thd)
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
    _check(bufs, thd=thd)
    plan._prepared, plan.takes_variant_pack = None, False
    try:
        g.execute(vp, ws)
        _check(bufs, thd=thd)
    finally:
        plan._prepared, plan.takes_variant_pack = prepared, True
    assert len(frames) == 2
    for i, name in enumerate(prepared.spec.order):
        if name != "stream":
            assert str(frames[0][i]) == str(frames[1][i]), name


@pytest.mark.parametrize("thd", [False, True])
def test_prepared_fp8_accepts_declared_bare_scalar_pointers(thd):
    g, vp, ws, bufs, tensors = _case(thd=thd)
    for name in ("descale_q", "descale_k", "descale_v", "scale_o", "amax_o"):
        vp[tensors[name]] = bufs[name].data_ptr()
    g.execute(vp, ws)
    _check(bufs, thd=thd)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_prepared_fp8_thd_output_row_stride_above_int32(dtype, d, dv):
    """The Int64 host ABI must not narrow again in device-side descriptor setup."""
    row_stride = 2**32 + 4 * dv
    if torch.cuda.mem_get_info()[0] < 2 * row_stride + 2**30:
        pytest.skip("wide physical row-stride regression needs 9 GiB free")
    g, vp, ws, bufs, tensors = _case(d=d, dv=dv, thd=True, override=True, dtype=dtype, sq=1, skv=64)
    plan = g._compiled_plans[g._plan_index]
    assert plan._prepared is not None
    owner = plan._prepared.spec.owner
    try:
        bufs["o"] = torch.empty_strided((2, 4, dv), (row_stride, dv, 1), device="cuda", dtype=torch.bfloat16)
    except torch.OutOfMemoryError:
        # A concurrent pytest worker may consume memory after the precheck.
        # Numerical failures and launch errors remain outside this guard.
        pytest.skip("wide physical row-stride storage unavailable under current GPU memory pressure")
    vp[tensors["o"]] = bufs["o"]
    overrides = dict(override_uids=[tensors["o"].get_uid()], override_shapes=[[2, 4, 1, dv]], override_strides=[[4 * dv, dv, row_stride, 1]])
    bufs["o"].fill_(float("nan"))
    g.execute(vp, ws, **overrides)
    _check(bufs, thd=True, sq=1, skv=64)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g.execute(vp, ws, **overrides)
    bufs["descale_v"].fill_(0.3)
    bufs["o"].fill_(float("nan"))
    bufs["lse"].fill_(float("nan"))
    bufs["amax_o"].fill_(999)
    graph.replay()
    _check(bufs, thd=True, sq=1, skv=64)
    assert plan._prepared.spec.owner is owner


@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_prepared_fp8_dense_kv_override(d, dv, monkeypatch):
    import cutlass.cute as cute

    g, vp, ws, bufs, tensors = _case(d=d, dv=dv, skv=256, override=True)
    owner = g._compiled_plans[g._plan_index]._prepared.spec.owner
    for name in ("k", "v"):
        bufs[name] = bufs[name][:, :, :128]
        vp[tensors[name]] = bufs[name]
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("override must reuse the artifact"))
    g.execute(
        vp,
        ws,
        override_uids=[tensors[n].get_uid() for n in ("k", "v")],
        override_shapes=[list(bufs[n].shape) for n in ("k", "v")],
        override_strides=[list(bufs[n].stride()) for n in ("k", "v")],
    )
    _check(bufs, thd=False, skv=128)
    assert g._compiled_plans[g._plan_index]._prepared.spec.owner is owner


@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_prepared_fp8_dense_input_strides(d, dv):
    """In particular, D192's rank-five K descriptor must use the caller's pitch."""
    g, vp, ws, bufs, tensors = _case(d=d, dv=dv, padded=True)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    _check(bufs, thd=False)
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        g.execute(vp, ws)
    bufs["descale_k"].fill_(1.5)
    bufs["o"].fill_(float("nan"))
    captured.replay()
    _check(bufs, thd=False)


@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_prepared_fp8_empty_thd_resets_amax_without_attention(d, dv, monkeypatch):
    """An empty packed Q has no host launch, but its reduction output is zero."""
    g, vp, ws, bufs, tensors = _case(d=d, dv=dv, thd=True)
    for name in ("q", "o", "lse"):
        buf = bufs[name]
        bufs[name] = torch.empty((0, *buf.shape[1:]), device=buf.device, dtype=buf.dtype)
        vp[tensors[name]] = bufs[name]
    for name in ("cu_q", "off_q", "off_o", "off_lse"):
        if name in tensors:
            vp[tensors[name]] = torch.zeros_like(vp[tensors[name]])
    spec = g._compiled_plans[g._plan_index]._prepared.spec
    monkeypatch.setattr(spec, "fn", lambda *a: pytest.fail("empty Q must not launch attention"))
    bufs["amax_o"].fill_(999)
    g.execute(vp, ws)
    torch.testing.assert_close(bufs["amax_o"], torch.zeros_like(bufs["amax_o"]))
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        g.execute(vp, ws)
    bufs["amax_o"].fill_(999)
    captured.replay()
    torch.testing.assert_close(bufs["amax_o"], torch.zeros_like(bufs["amax_o"]))


@pytest.mark.parametrize("output_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("padding", [8, 16])
def test_fp8_output_pitch_preserves_prepared_and_conversion_routes(output_dtype, padding):
    """One-byte O needs 16-element TMA alignment; other pitches retain conversion."""
    g, vp, ws, bufs, _ = _case(output_dtype=output_dtype, output_padding=padding)
    assert (g._compiled_plans[g._plan_index]._prepared is not None) == (padding == 16)
    g.execute(vp, ws)
    _check(bufs, thd=False)
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        g.execute(vp, ws)
    bufs["descale_v"].mul_(0.5)
    bufs["o"].fill_(float("nan"))
    captured.replay()
    _check(bufs, thd=False)
    assert torch.all(bufs["o_storage"][..., 128:] == 12)
