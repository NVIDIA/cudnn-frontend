# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental prepared norm/pointwise DAG, including graph-edge rounding."""

import pytest
import torch
import cudnn

from cudnn.norm.epilogue import analyze, prepare
from cudnn.norm.affine import lower
from cudnn.norm.graph_analyzer import _extract_facts

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def _supported():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("requires SM90/SM100/SM103")
    from cudnn.norm._lazy import dsl_unavailable

    reason = dsl_unavailable("norm epilogue")
    if reason:
        pytest.skip(reason)


def _typed(t, buf, output=False):
    t.set_dim(buf.shape).set_stride(buf.stride()).set_data_type(buf.dtype)
    if output:
        t.set_output(True)
    return t


def _graph(handle, kind, x, gamma, beta, gate, residual, *, fused=True, intermediate_dtype=None, branch=False, output_dtype=None):
    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    vp = {}

    def inp(buf, name):
        t = g.tensor_like(buf, name=name)
        vp[t] = buf
        return t

    eps = g.tensor_scalar(1e-3, cudnn.scalar_type.RUNTIME_PARAM).set_dim((1,) * x.ndim).set_stride((1,) * x.ndim).set_data_type(cudnn.data_type.FLOAT)
    xt = inp(x, "X")
    kwargs = dict(
        input=xt,
        scale=inp(gamma, "gamma"),
        bias=inp(beta, "beta") if beta is not None else None,
        epsilon=eps,
        norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
    )
    if kind == "rms":
        h, rt = g.rmsnorm(**kwargs)
        mt = None
    else:
        h, mt, rt = g.layernorm(**kwargs)
    _typed(h, x)
    norm_node = g.nodes[-1]
    result = {}
    for name, t in (("mean", mt), ("rstd", rt)):
        if t is not None:
            buf = torch.full((*x.shape[:-1], 1), float("nan"), dtype=torch.float32, device=x.device)
            vp[_typed(t, buf, True)] = buf
            result[name] = buf
    out = torch.full_like(x, float("nan"), dtype=output_dtype or x.dtype)
    if fused:
        gate_t, residual_t = inp(gate, "gate"), xt if residual is x else inp(residual, "residual")
        z = g.mul(gate_t, h)
        _typed(z, x)
        z.set_data_type(intermediate_dtype or x.dtype)
        if branch:
            # Shared norm value and a second operation, not a fixed linear chain.
            b = _typed(g.add(h, residual_t), x)
            y = g.add(z, b)
        else:
            y = g.add(residual_t, z)
        vp[_typed(y, out, True)] = out
    else:
        vp[_typed(h, out, True)] = out
    result["y"] = out
    return g, vp, result, norm_node


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "kind,prefix,d,gate_shape,branch,wide_intermediate",
    [
        ("rms", (2, 3), 128, (2, 1, 1), False, False),
        ("ln", (2, 3), 133, (133,), True, False),
        ("rms", (5,), 64, (5, 1), False, True),
        ("ln", (2, 3, 2), 129, (1, 3, 1, 1), False, True),
    ],
)
def test_frost_norm_pointwise_correctness_and_replay(cudnn_handle, dtype, kind, prefix, d, gate_shape, branch, wide_intermediate):
    torch.manual_seed(13)
    x = torch.randn((*prefix, d), device="cuda", dtype=dtype)
    gamma = torch.randn((*((1,) * len(prefix)), d), device="cuda", dtype=dtype)
    beta = torch.randn_like(gamma) if kind == "ln" else None
    gate = torch.randn(gate_shape, device="cuda", dtype=torch.float32)
    residual = torch.randn_like(x)
    intermediate_dtype = torch.float32 if wide_intermediate else dtype
    g, vp, got, _ = _graph(
        cudnn_handle,
        kind,
        x,
        gamma,
        beta,
        gate,
        residual,
        branch=branch,
        intermediate_dtype=intermediate_dtype,
        output_dtype=torch.float32 if wide_intermediate else dtype,
    )
    _base_g, base_vp, base, node = _graph(cudnn_handle, kind, x, gamma, beta, gate, residual, fused=False)
    plan, base_plan = prepare(g), lower(_extract_facts(node))

    def check():
        base_plan(base_vp)
        h = base["y"]
        z = (h.float() * gate).to(intermediate_dtype)
        ref = (z.float() + ((h.float() + residual.float()).to(dtype).float() if branch else residual.float())).to(got["y"].dtype)
        # Exact same norm kernel then separately rounded graph operations.
        torch.testing.assert_close(got["y"], ref, rtol=0, atol=0)
        for name in ("mean", "rstd"):
            if name in got:
                torch.testing.assert_close(got[name], base[name], rtol=0, atol=0)
        xf = x.float()
        mean = xf.mean(-1, keepdim=True) if kind == "ln" else torch.zeros_like(xf[..., :1])
        rstd = torch.rsqrt(((xf - mean) ** 2).mean(-1, keepdim=True) + 1e-3)
        norm_ref = (xf - mean) * rstd * gamma.float()
        if beta is not None:
            norm_ref = norm_ref + beta.float()
        torch.testing.assert_close(h.float(), norm_ref.to(dtype).float(), rtol=0.02, atol=0.02)
        torch.testing.assert_close(got["rstd"], rstd, rtol=2e-5, atol=2e-5)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan(vp)
        check()
    torch.cuda.current_stream().wait_stream(stream)
    captured = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(captured, stream=stream):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                plan(vp)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        from cuda.bindings import driver

        status, nodes, count = driver.cuGraphGetNodes(driver.CUgraph(captured.raw_cuda_graph()), 8)
        assert status == driver.CUresult.CUDA_SUCCESS
        assert count == 1, "prepared epilogue must capture exactly one kernel"
        status, node_type = driver.cuGraphNodeGetType(nodes[0])
        assert status == driver.CUresult.CUDA_SUCCESS
        assert node_type == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
        for delta in (0.25, -0.75):
            x.add_(delta)
            gate.add_(delta)
            residual.mul_(0.5)
            got["y"].fill_(float("nan"))
            captured.replay()
            check()
            saved = got["y"].clone()
            captured.replay()
            torch.testing.assert_close(got["y"], saved, rtol=0, atol=0)
    finally:
        captured.reset()


def test_frost_norm_pointwise_binding_rejection(cudnn_handle, monkeypatch):
    x = torch.randn((2, 64), dtype=torch.bfloat16, device="cuda")
    gamma = torch.ones((1, 64), dtype=x.dtype, device=x.device)
    gate, residual = torch.randn((2, 1), device=x.device), torch.randn_like(x)
    g, vp, result, _ = _graph(cudnn_handle, "rms", x, gamma, None, gate, residual)
    plan = prepare(g)
    plan(vp)
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("execute must not allocate, convert or compile")

    with monkeypatch.context() as m:
        for name in ("empty", "zeros", "empty_like", "zeros_like"):
            m.setattr(torch, name, forbidden)
        m.setattr(torch.Tensor, "to", forbidden)
        m.setattr(torch.Tensor, "contiguous", forbidden)
        m.setattr(cute, "compile", forbidden)
        plan(vp)
    bad = dict(vp)
    key = next(t for t, b in vp.items() if b is gate)
    bad[key] = torch.randn((2, 2), device=x.device)
    before = result["y"].clone()
    with pytest.raises(ValueError, match="binding"):
        plan(bad)
    torch.testing.assert_close(result["y"], before, rtol=0, atol=0)
    key = next(t for t, b in vp.items() if b is residual)
    bad = dict(vp)
    bad[key] = result["y"]
    with pytest.raises(ValueError, match="overlapping"):
        plan(bad)


def test_frost_norm_pointwise_declines_observable_intermediate(cudnn_handle):
    x = torch.ones((2, 64), device="cuda", dtype=torch.float16)
    gamma = torch.ones((1, 64), device=x.device, dtype=x.dtype)
    g, _, _, node = _graph(cudnn_handle, "rms", x, gamma, None, gamma, x)
    node.outputs["Y"].set_output(True)
    with pytest.raises(NotImplementedError, match="virtual"):
        analyze(g)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("broadcast", "broadcastable"),
        ("layout", "dense"),
        ("op", "add/mul"),
        ("intermediate", "observable"),
        ("statistics", "statistics"),
    ],
)
def test_frost_norm_pointwise_declaration_rejection(cudnn_handle, case, reason):
    x = torch.ones((2, 64), device="cuda", dtype=torch.float16)
    gamma = torch.ones((1, 64), device=x.device, dtype=x.dtype)
    gate = torch.ones((2, 1), device=x.device)
    g, _, _, node = _graph(cudnn_handle, "rms", x, gamma, None, gate, x)
    mul = g.nodes[1]
    if case == "broadcast":
        next(t for t in mul.inputs.values() if t is not node.outputs["Y"]).set_dim((3, 1))
    elif case == "layout":
        next(t for t in mul.inputs.values() if t is not node.outputs["Y"]).set_stride((2, 1))
    elif case == "op":
        # Build an unsupported but otherwise valid public graph, not a fake node.
        g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=cudnn_handle)
        eps = g.tensor_scalar(1e-3, cudnn.scalar_type.RUNTIME_PARAM).set_data_type(cudnn.data_type.FLOAT)
        h, _ = g.rmsnorm(input=g.tensor_like(x), scale=g.tensor_like(gamma), epsilon=eps, norm_forward_phase=cudnn.norm_forward_phase.TRAINING)
        _typed(h, x)
        _typed(g.relu(h), x, True)
    elif case == "intermediate":
        next(iter(mul.outputs.values())).set_output(True)
    else:
        g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=cudnn_handle)
        eps = g.tensor_scalar(1e-3, cudnn.scalar_type.RUNTIME_PARAM).set_data_type(cudnn.data_type.FLOAT)
        h, rt = g.rmsnorm(input=g.tensor_like(x), scale=g.tensor_like(gamma), epsilon=eps, norm_forward_phase=cudnn.norm_forward_phase.TRAINING)
        _typed(h, x)
        _typed(rt, gate, True).set_data_type(torch.float32)
        _typed(g.add(h, rt), x, True)
    with pytest.raises(NotImplementedError, match=reason):
        analyze(g)


def test_frost_norm_pointwise_shared_input_name_binding(cudnn_handle):
    x = torch.randn((2, 64), device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones((1, 64), device=x.device, dtype=x.dtype)
    gate = torch.randn((2, 1), device=x.device)
    g, vp, got, _ = _graph(cudnn_handle, "rms", x, gamma, None, gate, x)
    for i, t in enumerate(vp):
        t.set_name(f"binding{i}")
    plan = prepare(g)
    names = {t.get_name(): buf for t, buf in vp.items()}
    plan(names)
    object_result = got["y"].clone()
    got["y"].fill_(float("nan"))
    plan(vp)
    torch.testing.assert_close(got["y"], object_result, rtol=0, atol=0)
