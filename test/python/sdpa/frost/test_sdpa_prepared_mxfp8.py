# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""MXFP8 prepared launches preserve opaque SF storage and current runtime bindings."""

import math

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, select_engine
from sdpa.frost.test_sdpa_prepared_fp8 import _cuda_graph

pytestmark = [requires_dsl, requires_pre_rubin_blackwell]


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
    arch="sm100",
    split_kv=1,
    hq=4,
    hk=2,
    explicit_plan=False,
    gate=None,
):
    dv = d if dv is None else dv
    torch.manual_seed(827)
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=override,
    )
    buffers, tensors, vp = {}, {}, {}
    input_type = cudnn.data_type.FP8_E4M3 if dtype == torch.float8_e4m3fn else cudnn.data_type.FP8_E5M2
    from sdpa.frost.test_sdpa_fwd_mxfp8_sm100 import _quantize, _quantize_seq

    sf_args = {}
    for name, h, seq, dim in (("q", hq, sq, d), ("k", hk, skv, d), ("v", hk, skv, dv)):
        x = torch.randn(b, h, seq, dim, device="cuda") * 0.4
        if thd:
            parts = [_quantize_seq(x[i : i + 1], h, seq, dim, dtype, columnwise=name == "v") for i in range(b)]
            x8 = torch.cat([part[0] for part in parts], dim=0)
            dq = torch.cat([part[1] for part in parts], dim=0)
            sf = torch.cat([part[2] for part in parts], dim=1).unsqueeze(0).contiguous()
        else:
            x8, sf, dq, _ = _quantize(x, b, h, seq, dim, dtype, columnwise=name == "v")
            sf = sf.view(torch.uint8).reshape(b, h, (seq + 127) // 128, -1)
        raw = x8.transpose(1, 2).contiguous()
        if padded:
            storage = torch.empty(b, seq, h, dim + 16, device="cuda", dtype=dtype)
            storage[..., :dim].copy_(raw)
            raw = storage[..., :dim]
        buf = raw.reshape(b * seq, h, dim) if thd else raw.transpose(1, 2)
        strides = [seq * h * dim, dim, h * dim, 1] if thd else list(buf.stride())
        t = g.tensor(dim=[b, h, seq, dim], stride=strides, data_type=input_type, name=name)
        buffers[name], tensors[name], vp[t] = buf, t, buf
        buffers["dq_" + name] = dq
        buffers["factor_" + name] = 1.0
        rows, cols = ((seq + 127) // 128 * 4, (dim + 127) // 128 * 128) if name == "v" else ((seq + 127) // 128 * 128, (dim + 127) // 128 * 4)
        sf_t = g.tensor(
            dim=[b, h, rows, cols],
            stride=[h * rows * cols, rows * cols, cols, 1],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
            name="sf_" + name,
        )
        buffers["sf_" + name], tensors["sf_" + name], vp[sf_t] = sf, sf_t, sf
        sf_args["descale_" + name] = sf_t
    kwargs = {}
    if thd:
        for name, seq, h in (("q", sq, hq), ("kv", skv, hk)):
            cu = torch.arange(b + 1, device="cuda", dtype=torch.int32) * seq
            lens = g.tensor(dim=[b if arch == "sm120" else b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="cu_" + name)
            off = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="off_" + name)
            vp[lens], vp[off] = (torch.full((b,), seq, device="cuda", dtype=torch.int32) if arch == "sm120" else cu), cu * h * d
            tensors["cu_" + name], tensors["off_" + name] = lens, off
            for role in (("q",) if name == "q" else ("k", "v")):
                tensors[role].set_ragged_offset(off)
            if name == "kv" and dv != d:
                off_v = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="off_v")
                tensors["v"].set_ragged_offset(off_v)
                tensors["off_v"], vp[off_v] = off_v, cu * hk * dv
            kwargs[("seq_len_" if arch == "sm120" else "cu_seq_len_") + name] = lens
        kwargs["use_padding_mask"] = True
    kwargs.update(sf_args)
    o, lse, am = g.sdpa_mxfp8(q=tensors["q"], k=tensors["k"], v=tensors["v"], attn_scale=1 / math.sqrt(d), generate_stats=stats, **kwargs)
    out_type = {
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float16: cudnn.data_type.HALF,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
        torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
    }[output_dtype]
    if gate is not None:
        assert not thd
        o.set_data_type(out_type).set_dim([b, hq, sq, dv]).set_stride([sq * hq * dv, dv, hq * dv, 1])
        gate_t = g.tensor_like(gate)
        buffers["gate"], tensors["gate"], vp[gate_t] = gate, gate_t, gate
        o = g.mul(a=o, b=g.sigmoid(input=gate_t, name="sigmoid_gate"), name="gated_o")
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
    if override or explicit_plan:
        # Pin the executor under test. Unrelated backend heuristics can fail
        # before the physical-stride fixture reaches a FROST launch.
        from cudnn.engines import MANIFEST
        from cudnn.sdpa.fwd.engines import ENGINE_SPECS, SdpaFwdKnobs

        name = engine_name(arch=arch, mxfp8=True)
        family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
        caps = next(s.capabilities for s in ENGINE_SPECS if s.name == name)
        cgas = dict(caps.cgas_by_d_shape).get((d, dv), caps.cgas)
        knobs = SdpaFwdKnobs(tile_m=max(caps.tile_ms), tile_n=max(caps.tile_ns), cga=max(cgas), split_kv=split_kv, sched_policy=0, pack_gqa=False)
        g.create_execution_plan(family.offered_ids()[name], knobs)
    else:
        g.create_execution_plans([cudnn.heur_mode.A])
    chosen = select_engine(g, engine_name(arch=arch, mxfp8=True), **({"pack_gqa": False} if arch == "sm120" else {}))
    if (chosen.knobs.split_kv or 1) != split_kv:
        from dataclasses import replace

        knobs = replace(chosen.knobs, split_kv=split_kv, sched_policy=0)
        if arch == "sm100" and (d, dv) == (192, 128) and split_kv > 1:
            knobs = replace(knobs, cga=2)
        g.create_execution_plan(chosen.engine_id, knobs)
        g.select_plan(len(g.plans) - 1)
    assert (g.plans[g._plan_index].knobs.split_kv or 1) == split_kv
    g.check_support()
    g.build_plans()
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return g, vp, workspace, buffers, tensors


def _check(bufs, *, thd, b=2, sq=128, skv=128):
    values = {}
    for name, seq in (("q", sq), ("k", skv), ("v", skv)):
        heads = bufs[name].shape[1]
        x = bufs[name].reshape(b, seq, heads, -1).transpose(1, 2) if thd else bufs[name]
        values[name] = x.double() * bufs["dq_" + name].double() * bufs["factor_" + name]
    q, k, v = (
        values["q"],
        values["k"].repeat_interleave(values["q"].shape[1] // values["k"].shape[1], 1),
        values["v"].repeat_interleave(values["q"].shape[1] // values["v"].shape[1], 1),
    )
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    ref, stats = scores.softmax(-1) @ v, scores.logsumexp(-1)
    am = ref.abs().amax()
    if "gate" in bufs:
        ref *= torch.sigmoid(bufs["gate"].float())
    if thd:
        ref = ref.transpose(1, 2).reshape(b * sq, q.shape[1], -1)
        stats = stats.transpose(1, 2).reshape(b * sq, q.shape[1])
    atol = 0.03
    if bufs["o"].dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        floor = (ref - ref.to(bufs["o"].dtype).double()).abs().max().item()
        atol = max(atol, 3 * floor)
    torch.testing.assert_close(bufs["o"].double(), ref, atol=atol, rtol=0.03)
    if "lse" in bufs:
        torch.testing.assert_close(bufs["lse"].double(), stats, atol=0.01, rtol=0.01)
    if "amax_o" in bufs:
        torch.testing.assert_close(bufs["amax_o"][0].double(), am, atol=0.03, rtol=0.03)


def _change_scales(bufs):
    for name, step in (("q", -1), ("k", -1), ("v", 1)):
        bufs["sf_" + name].add_(step)
        bufs["factor_" + name] *= 2.0**step


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("stats,amax", [(True, True), (False, False)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.L0
def test_prepared_mxfp8_rebind_scales_and_buffers(thd, stats, amax, dtype, d, dv, output_dtype):
    g, vp, ws, bufs, tensors = _case(thd=thd, stats=stats, amax=amax, dtype=dtype, d=d, dv=dv, output_dtype=output_dtype)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "o", "sf_q", "sf_k", "sf_v", "lse", "amax_o"):
                if name in tensors:
                    bufs[name] = bufs[name].clone()
                    vp[tensors[name]] = bufs[name]
            _change_scales(bufs)
        bufs["o"].fill_(float("nan"))
        if amax:
            bufs["amax_o"].fill_(999)
        g.execute(vp, ws)
        torch.cuda.synchronize()
        _check(bufs, thd=thd)


@pytest.mark.parametrize("thd,split", [(False, 1), (True, 1), (False, 4)])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.L0
def test_prepared_mxfp8_capture_reads_current_scales(thd, split, d, dv, output_dtype):
    g, vp, ws, bufs, _ = _case(thd=thd, split_kv=split, d=d, dv=dv, skv=512, output_dtype=output_dtype)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    torch.cuda.synchronize()
    with _cuda_graph() as graph:
        with torch.cuda.graph(graph):
            g.execute(vp, ws)
        _change_scales(bufs)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        graph.replay()
        torch.cuda.synchronize()
        _check(bufs, thd=thd, skv=512)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.L0
def test_prepared_mxfp8_thd_output_stride_above_int32(dtype, d, dv):
    """Both live sequences must reach their physical Int64 output row."""
    row_stride = 2**32 + 4 * dv
    if torch.cuda.mem_get_info()[0] < 2 * row_stride + 2**30:
        pytest.skip("physical row-stride regression needs 9 GiB free")
    g, vp, ws, bufs, ts = _case(thd=True, override=True, dtype=dtype, d=d, dv=dv, sq=1, skv=64)
    try:
        bufs["o"] = torch.empty_strided((2, 4, dv), (row_stride, dv, 1), dtype=torch.bfloat16, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("physical row-stride storage unavailable")
    vp[ts["o"]] = bufs["o"]
    overrides = dict(override_uids=[ts["o"].get_uid()], override_shapes=[[2, 4, 1, dv]], override_strides=[[4 * dv, dv, row_stride, 1]])
    owner = g._compiled_plans[g._plan_index]._prepared.spec.owner
    bufs["o"].fill_(float("nan"))
    g.execute(vp, ws, **overrides)
    _check(bufs, thd=True, sq=1, skv=64)
    with _cuda_graph() as captured:
        with torch.cuda.graph(captured):
            g.execute(vp, ws, **overrides)
        _change_scales(bufs)
        bufs["o"].fill_(float("nan"))
        captured.replay()
        _check(bufs, thd=True, sq=1, skv=64)
    assert g._compiled_plans[g._plan_index]._prepared.spec.owner is owner


@pytest.mark.gpu_exclusive
@pytest.mark.L1
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_prepared_mxfp8_sf_head_stride_above_int32_units(d, dv):
    """SF descriptor strides are in 16-byte units: multiply in Int64 first."""
    # Release the preceding parametrization's cached 128-GiB allocation before
    # checking physical free memory; only live allocations should cause a skip.
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 130 * 2**30:
        pytest.skip("physical SF head-stride regression needs 130 GiB free")
    g, vp, ws, bufs, ts = _case(thd=True, d=d, dv=dv, hq=2, hk=1)
    original = bufs["sf_q"]
    size = original.shape[-1]
    head_stride = 2**36 + 2 * size  # narrowing the 16-byte stride aliases two tiles past head 0
    tiles = head_stride // size
    try:
        wide = torch.empty((1, 2, tiles, size), device="cuda", dtype=torch.uint8)
    except torch.OutOfMemoryError:
        pytest.skip("physical SF head-stride storage unavailable")
    # Touch only the live descriptor rows and the deliberate narrow-stride alias.
    wide[:, :, : original.shape[2]].copy_(original)
    wide[:, 0, 2 : 2 + original.shape[2]].fill_(127)
    bufs["sf_q"] = wide
    vp[ts["sf_q"]] = wide
    bufs["o"].fill_(float("nan"))
    g.execute(vp, ws)
    torch.cuda.synchronize()
    _check(bufs, thd=True)
    with _cuda_graph() as captured:
        with torch.cuda.graph(captured):
            g.execute(vp, ws)
        bufs["o"].fill_(float("nan"))
        captured.replay()
        torch.cuda.synchronize()
        _check(bufs, thd=True)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.L0
@pytest.mark.parametrize("carrier", [torch.uint8, torch.int32])
def test_prepared_mxfp8_sf_physical_permutation(thd, d, dv, carrier):
    g, vp, ws, bufs, ts = _case(thd=thd, d=d, dv=dv)
    for name in ("sf_q", "sf_k", "sf_v"):
        # Same byte stream with a different logical axis order, deliberately noncontiguous.
        vp[ts[name]] = bufs[name].view(carrier).permute(3, 1, 0, 2)
    g.execute(vp, ws)
    _check(bufs, thd=thd)


@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.L0
def test_prepared_mxfp8_strided_dense_inputs(d, dv):
    g, vp, ws, bufs, _ = _case(d=d, dv=dv, padded=True)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    _check(bufs, thd=False)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.L0
def test_prepared_mxfp8_execute_has_no_allocation_or_sync(thd, split, monkeypatch):
    if thd and split > 1:
        pytest.skip("THD split is not served")
    g, vp, ws, bufs, _ = _case(thd=thd, split_kv=split, skv=512)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    with monkeypatch.context() as m:
        for name in ("empty", "zeros", "ones", "empty_like", "zeros_like", "ones_like"):
            m.setattr(torch, name, lambda *a, **kw: pytest.fail("execute allocated a tensor"))
        torch.cuda.set_sync_debug_mode("error")
        try:
            g.execute(vp, ws)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    torch.cuda.synchronize()
    _check(bufs, thd=thd, skv=512)


@pytest.mark.L0
@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_prepared_mxfp8_graph_and_adapter_bind_same_frame(thd, d, dv, monkeypatch):
    g, vp, ws, bufs, _ = _case(thd=thd, d=d, dv=dv)
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


@pytest.mark.L0
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("layout", ["padded", "batch_inner"])
def test_prepared_mxfp8_runtime_stats_layout_keeps_generic_branch(d, dv, layout):
    g, vp, ws, bufs, tensors = _case(d=d, dv=dv)
    original = bufs["lse"]
    b, h, sq = original.shape
    if layout == "padded":
        rebound = torch.empty_strided((b, h, sq), (h * (sq + 16), sq + 16, 1), device="cuda", dtype=torch.float32)
    else:
        rebound = torch.empty((h, sq, b), device="cuda").permute(2, 0, 1)
    assert rebound.stride() != original.stride()
    # Match the graph's explicit BH S1 axes so native rank normalization
    # does not reinterpret a physically contiguous axis permutation.
    vp[tensors["lse"]] = rebound.unsqueeze(-1)
    bufs["lse"] = rebound
    rebound.fill_(float("nan"))
    g.execute(vp, ws)
    _check(bufs, thd=False)
    with _cuda_graph() as captured:
        with torch.cuda.graph(captured):
            g.execute(vp, ws)
        _change_scales(bufs)
        rebound.fill_(float("nan"))
        captured.replay()
        _check(bufs, thd=False)
    # Reusing the originally declared layout remains valid.
    vp[tensors["lse"]] = bufs["lse"] = original
    original.fill_(float("nan"))
    g.execute(vp, ws)
    _check(bufs, thd=False)
