# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native SM120 D128 bias: declared storage, routing, rebinding and capture."""

from dataclasses import replace

import cudnn
import pytest
import torch

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_blackwell_geforce, requires_dsl, select_engine

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


def _case(
    dtype=torch.bfloat16,
    bias_fp32=True,
    broadcast=(1, 1),
    mask="none",
    d=128,
    split=1,
    tile_m=64,
    wide_bias=0,
    scale=None,
    stats=True,
    stats_log2=False,
    sink=False,
):
    torch.manual_seed(381)
    b, h, hk, sq, sk = 2, 4, 2, 67, 139
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bufs, tensors, vp, storage = {}, {}, {}, {}
    for name, heads, seq in (("q", h, sq), ("k", hk, sk), ("v", hk, sk), ("o", h, sq)):
        raw = torch.randn(b, seq, heads, d + 8, dtype=dtype, device="cuda") * 0.4
        if name == "o":
            raw.fill_(123)
        buf = raw[..., :d].transpose(1, 2)
        storage[name], bufs[name] = raw, buf
        if name != "o":
            tensors[name] = g.tensor_like(buf, name=name)
            vp[tensors[name]] = buf
    bdtype = torch.float32 if bias_fp32 else dtype
    raw = torch.randn(*broadcast, sq, 144, dtype=bdtype, device="cuda") * 1.5
    bufs["bias"], storage["bias"] = raw[..., :sk], raw
    if wide_bias:
        stride = list(bufs["bias"].stride())
        axis = 0 if wide_bias == 1 else 1
        stride[axis] = (2**32 + 128) if wide_bias == 1 else (2**31 + 128)
        need = 1 + sum((n - 1) * st for n, st in zip(bufs["bias"].shape, stride))
        if torch.cuda.mem_get_info()[0] < need * bufs["bias"].element_size() + 2**30:
            pytest.skip("physical Int64 bias stride needs additional device storage")
        backing = torch.empty(need, dtype=bdtype, device="cuda")
        backing[: min(need, 65536)].fill_(47)
        view = backing.as_strided(bufs["bias"].shape, stride)
        view.copy_(bufs["bias"])
        bufs["bias"], storage["bias"] = view, backing
    tensors["bias"] = g.tensor_like(bufs["bias"], name="bias")
    vp[tensors["bias"]] = bufs["bias"]
    kwargs = {}
    if mask in ("causal", "window", "bottom_right", "band"):
        kwargs["use_causal_mask"] = True
    if mask == "bottom_right":
        kwargs = {"use_causal_mask_bottom_right": True}
    if mask == "band":
        kwargs = {"diagonal_band_right_bound": 7}
    if mask == "padded":
        for name, values in (("seq_q", [51, sq]), ("seq_kv", [0, 107])):
            buf = torch.tensor(values, dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
            tensors[name] = g.tensor_like(buf, name=name)
            bufs[name], vp[tensors[name]] = buf, buf
        kwargs.update(use_padding_mask=True, seq_len_q=tensors["seq_q"], seq_len_kv=tensors["seq_kv"])
    if mask == "window":
        kwargs["diagonal_band_left_bound"] = 25
    if sink:
        bufs["sinks"] = torch.randn(1, h, 1, 1, device="cuda", dtype=torch.float32)
        tensors["sinks"] = g.tensor_like(bufs["sinks"], name="sink")
        vp[tensors["sinks"]] = bufs["sinks"]
        kwargs["sink_token"] = tensors["sinks"]
    o, stats_tensor = g.sdpa(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        bias=tensors["bias"],
        attn_scale=d**-0.5 if scale is None else scale,
        generate_stats=stats,
        stats_use_log2=stats_log2,
        **kwargs,
    )
    o.set_output(True).set_dim(bufs["o"].shape).set_stride(bufs["o"].stride())
    tensors["o"], vp[o] = o, bufs["o"]
    if stats:
        raw = torch.full((b, h, sq * 2), float("nan"), device="cuda")
        storage["lse"], bufs["lse"] = raw, raw[..., ::2]
        stats_tensor.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, h, sq, 1]).set_stride([h * sq * 2, sq * 2, 2, 1])
        tensors["lse"], vp[stats_tensor] = stats_tensor, bufs["lse"]
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    selected = select_engine(g, engine_name(arch="sm120"), pack_gqa=False)
    g.create_execution_plan(selected.engine_id, replace(selected.knobs, tile_m=tile_m, split_kv=split, pack_gqa=False))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    return g, vp, ws, bufs, tensors, storage


def _check(bufs, storage, mask, scale=None, stats_log2=False):
    q, k, v = (bufs[n].double() for n in ("q", "k", "v"))
    scores = (q @ k.repeat_interleave(2, 1).transpose(-1, -2)) * (q.shape[-1] ** -0.5 if scale is None else scale) + bufs["bias"].double()
    qr = torch.arange(q.shape[-2], device="cuda")[:, None]
    kr = torch.arange(k.shape[-2], device="cuda")[None, :]
    if mask in ("causal", "window", "bottom_right", "band"):
        offset = k.shape[-2] - q.shape[-2] if mask == "bottom_right" else 0
        scores.masked_fill_(kr > qr + offset + (7 if mask == "band" else 0), -float("inf"))
    if mask == "padded":
        for batch in range(q.shape[0]):
            scores[batch, :, int(bufs["seq_q"].flatten()[batch]) :] = -float("inf")
            scores[batch, :, :, int(bufs["seq_kv"].flatten()[batch]) :] = -float("inf")
    if mask == "window":
        scores.masked_fill_(kr < qr - 24, -float("inf"))
    logits = scores
    if "sinks" in bufs:
        logits = torch.cat((scores, bufs["sinks"].double().expand(q.shape[0], q.shape[1], q.shape[2], 1)), -1)
    ref = logits.softmax(-1).nan_to_num()[..., : k.shape[-2]] @ v.repeat_interleave(2, 1)
    torch.testing.assert_close(bufs["o"].double(), ref, atol=2e-3, rtol=2e-2)
    if "lse" in bufs:
        expected = logits.logsumexp(-1) * (1.4426950408889634 if stats_log2 else 1)
        torch.testing.assert_close(bufs["lse"].double(), expected, atol=3e-4, rtol=3e-4)
        assert torch.isnan(storage["lse"][..., 1::2]).all()
    assert torch.all(storage["o"][..., q.shape[-1] :] == 123)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias_fp32", [False, True])
@pytest.mark.parametrize("broadcast,mask", [((1, 1), "none"), ((2, 1), "causal"), ((1, 4), "window"), ((2, 4), "none")])
def test_frost_sm120_bias_rebind_capture(dtype, bias_fp32, broadcast, mask, monkeypatch, cudnn_handle):
    import cutlass.cute as cute

    g, vp, ws, bufs, tensors, storage = _case(dtype, bias_fp32, broadcast, mask)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared is not None and prepared.spec.has_bias
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute must not compile"))
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "o", "bias", "lse"):
                storage[name] = storage[name].clone()
                if name == "bias":
                    bufs[name] = storage[name][..., :139]
                elif name == "lse":
                    bufs[name] = storage[name][..., ::2]
                else:
                    bufs[name] = storage[name][..., :128].transpose(1, 2)
                vp[tensors[name]] = bufs[name]
        g.execute(vp, ws)
        _check(bufs, storage, mask)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        previous_stream = cudnn.get_stream(cudnn_handle)
        try:
            cudnn.set_stream(cudnn_handle, stream.cuda_stream)
            with torch.cuda.graph(captured, stream=stream):
                g.execute(vp, ws, handle=cudnn_handle)
            torch.cuda.current_stream().wait_stream(stream)
            bufs["bias"].mul_(-0.7)
            bufs["v"].mul_(0.6)
            bufs["o"].fill_(float("nan"))
            captured.replay()
            _check(bufs, storage, mask)
        finally:
            cudnn.set_stream(cudnn_handle, previous_stream)
            captured.reset()
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        g.execute(vp, ws)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _standalone(bufs, scale=128**-0.5, **kw):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    return SdpaFwdDslSm120(
        sample_q=bufs["q"],
        sample_k=bufs["k"],
        sample_v=bufs["v"],
        sample_o=bufs["o"],
        sample_lse=bufs["lse"],
        sample_bias=bufs["bias"],
        scale_softmax=scale,
        tile_m=64,
        pack_gqa=False,
        **kw,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias_fp32", [False, True])
def test_frost_sm120_bias_standalone_and_validation(dtype, bias_fp32, monkeypatch):
    g, vp, ws, bufs, tensors, storage = _case(dtype, bias_fp32, (2, 4))
    api = _standalone(bufs)
    api.check_support()
    api.compile()
    assert api.scratch_workspace_bytes() == 0
    original = api._dense_spec.fn
    launches = []

    def record(*args):
        launches.append(args)
        return original(*args)

    monkeypatch.setattr(api._dense_spec, "fn", record)
    for iteration in range(2):
        if iteration:
            storage["bias"] = storage["bias"].clone()
            bufs["bias"] = storage["bias"][..., :139]
            vp[tensors["bias"]] = bufs["bias"]
            bufs["bias"].mul_(-0.6)
        api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"])
        _check(bufs, storage, "none")
        g.execute(vp, ws)
        _check(bufs, storage, "none")
        from cuda.bindings import driver as cuda_driver

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(captured, stream=stream):
                api.execute(
                    bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"], current_stream=cuda_driver.CUstream(stream.cuda_stream)
                )
            torch.cuda.current_stream().wait_stream(stream)
            bufs["bias"].mul_(1.2)
            captured.replay()
            _check(bufs, storage, "none")
        finally:
            captured.reset()
    assert len(launches) == 4
    for bad in (None, bufs["bias"][..., :-1], bufs["bias"].to(torch.float64), bufs["bias"][..., ::2]):
        bufs["o"].fill_(123)
        bufs["lse"].fill_(456)
        with pytest.raises(ValueError, match="bias"):
            api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bad)
        assert len(launches) == 4
        assert torch.all(bufs["o"] == 123)
        assert torch.all(bufs["lse"] == 456)
    api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"])
    _check(bufs, storage, "none")


@pytest.mark.parametrize("change", [{"d_qk": 96}, {"d_v": 96}, {"thd": True}, {"split_kv": 2}, {"pack_gqa": True}, {"tile_m": 16}, {"bottom_right": True}])
def test_frost_sm120_bias_bounded_route(change):
    from cudnn.sdpa.fwd.config_sm120 import bias_decline

    kwargs = dict(d_qk=128, d_v=128, tile_m=64)
    assert bias_decline(**kwargs) is None
    kwargs.update(change)
    assert "dense exact D128" in bias_decline(**kwargs)


def test_frost_sm120_bias_row_claim():
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    caps = next(s.capabilities for s in ENGINE_SPECS if s.name == engine_name(arch="sm120"))
    assert caps.bias
    assert not caps.bias_io_dtype_only


@pytest.mark.parametrize("change", [{}, {"d_qk": 96}, {"d_v": 96}, {"thd": True}, {"shape_overrides": True}])
def test_frost_sm120_bias_graph_capability_gate(change):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, SdpaFwdKnobs, mismatch

    class Bias:
        def get_data_type(self):
            return cudnn.data_type.FLOAT

    base = dict(
        b=2,
        h_q=4,
        h_kv=2,
        s_q=67,
        s_kv=139,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(12, 0),
        has_bias=True,
        bias_t=Bias(),
        bias_dim=(1, 1, 67, 139),
        bias_stride=(67 * 144, 67 * 144, 144, 1),
    )
    base.update(change)
    facts = ga.SdpaGraphFacts(**base)
    caps = next(s.capabilities for s in ENGINE_SPECS if s.name == engine_name(arch="sm120"))
    reason = mismatch(caps, facts, SdpaFwdKnobs(tile_m=64, tile_n=64, cga=1, split_kv=1, pack_gqa=False))
    assert (reason is None) == (not change), reason


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask", ["band", "padded"])
def test_frost_sm120_bias_mask_composition(dtype, mask):
    g, vp, ws, bufs, _, storage = _case(dtype, True, (2, 4), mask, tile_m=128)
    # A dead row spans multiple KV tiles, including the unmasked loop.
    bufs["bias"][:, :, 0, :] = -float("inf")
    g.execute(vp, ws)
    _check(bufs, storage, mask)


@pytest.mark.parametrize("wide_bias,broadcast", [(1, (2, 1)), (2, (1, 4))])
def test_frost_sm120_bias_physical_int64_stride(wide_bias, broadcast, cudnn_handle):
    g, vp, ws, bufs, tensors, storage = _case(torch.bfloat16, False, broadcast, wide_bias=wide_bias)
    api = _standalone(bufs)
    api.check_support()
    api.compile()
    for run in (lambda: g.execute(vp, ws), lambda: api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"])):
        run()
        _check(bufs, storage, "none")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    previous_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        with torch.cuda.graph(captured, stream=stream):
            g.execute(vp, ws, handle=cudnn_handle)
        torch.cuda.current_stream().wait_stream(stream)
        bufs["bias"].mul_(-0.7)
        bufs["v"].mul_(0.6)
        captured.replay()
        _check(bufs, storage, "none")
    finally:
        cudnn.set_stream(cudnn_handle, previous_stream)
        captured.reset()


@pytest.mark.parametrize("scale", [-0.25, 1e-38, 3e38])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_frost_sm120_bias_scale_composition(scale, dtype):
    g, vp, ws, bufs, _, storage = _case(dtype, True, (2, 4), scale=scale)
    bufs["bias"].mul_(20)
    if scale > 1e30:
        bufs["q"].mul_(0.001)
    g.execute(vp, ws)
    _check(bufs, storage, "none", scale)
    api = _standalone(bufs, scale=scale)
    api.check_support()
    api.compile()
    for runtime_scale in (scale, scale * 0.5):
        api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"], scale_softmax=runtime_scale)
        _check(bufs, storage, "none", runtime_scale)


@pytest.mark.parametrize("stats,stats_log2,sink", [(False, False, False), (True, True, False), (True, False, True)])
def test_frost_sm120_bias_optional_stats_and_sink(stats, stats_log2, sink):
    g, vp, ws, bufs, _, storage = _case(stats=stats, stats_log2=stats_log2, sink=sink)
    bufs["bias"][:, :, 0, :] = -float("inf")
    g.execute(vp, ws)
    _check(bufs, storage, "none", stats_log2=stats_log2)


@pytest.mark.parametrize("logit", [3e38, -3e38])
def test_frost_sm120_bias_large_finite_logits(logit):
    g, vp, ws, bufs, _, storage = _case()
    bufs["bias"].fill_(logit)
    g.execute(vp, ws)
    _check(bufs, storage, "none")
    api = _standalone(bufs)
    api.check_support()
    api.compile()
    api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"])
    _check(bufs, storage, "none")


def test_frost_sm120_bias_declined_on_plain_specialization():
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    _, _, _, bufs, _, storage = _case()
    api = SdpaFwdDslSm120(bufs["q"], bufs["k"], bufs["v"], bufs["o"], sample_lse=bufs["lse"], tile_m=64, pack_gqa=False)
    api.check_support()
    api.compile()
    bufs["o"].fill_(123)
    bufs["lse"].fill_(456)
    with pytest.raises(ValueError, match="without a bias"):
        api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], bias_tensor=bufs["bias"])
    assert torch.all(bufs["o"] == 123)
    assert torch.all(bufs["lse"] == 456)
    api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"])
    _check({**bufs, "bias": torch.zeros_like(bufs["bias"])}, storage, "none")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sink_logit", [-float("inf"), float("inf")])
def test_frost_sm120_bias_sink_endpoints(dtype, sink_logit, cudnn_handle):
    g, vp, ws, bufs, _, storage = _case(dtype, sink=True)
    bufs["bias"][:, :, 0, :] = -float("inf")
    bufs["sinks"].fill_(sink_logit)
    g.execute(vp, ws)
    _check(bufs, storage, "none")
    api = _standalone(bufs, has_sink=True)
    api.check_support()
    api.compile()
    api.execute(bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["lse"], sinks=bufs["sinks"], bias_tensor=bufs["bias"])
    _check(bufs, storage, "none")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    previous_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        with torch.cuda.graph(captured, stream=stream):
            g.execute(vp, ws, handle=cudnn_handle)
        torch.cuda.current_stream().wait_stream(stream)
        for value in (0.7, sink_logit):
            bufs["sinks"].fill_(value)
            captured.replay()
            _check(bufs, storage, "none")
    finally:
        cudnn.set_stream(cudnn_handle, previous_stream)
        captured.reset()
