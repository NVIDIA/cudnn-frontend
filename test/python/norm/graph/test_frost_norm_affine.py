# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive affine parameters and the gamma + 1 graph prologue."""

import json

import pytest
import torch
import cudnn

from cudnn.norm._lazy import dsl_unavailable

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def _supported(monkeypatch):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("FROST norms require SM100")
    reason = dsl_unavailable("FROST norm")
    if reason:
        pytest.skip(reason)
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def _output(t, buf):
    t.set_output(True).set_data_type(buf.dtype).set_dim(buf.shape).set_stride(buf.stride())
    return t


def _build(g, phase):
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    name = f"norm_{phase}_sm100"
    # Select by the public engine identity, never by heuristic order.
    candidates = [i for i, p in enumerate(g.plans) if p.engine_id >= 20_000]
    assert candidates, f"no FROST {phase} plan offered"
    g.select_plan(candidates[0])
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == name
    assert g.get_workspace_size() == 0


def _graph(
    handle,
    kind,
    x,
    gamma,
    beta,
    *,
    phase="fprop",
    adaptive=False,
    zero_centered_gamma=False,
    add_dtype=torch.float32,
    inference=False,
    dy=None,
    mean=None,
    rstd=None,
    swap=False,
    host_one=False,
    legacy_ada=False,
):
    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    vp = {}

    def inp(buf, name):
        if buf is None:
            return None
        t = g.tensor_like(buf, name=name)
        vp[t] = buf
        return t

    scalar_shape = (1,) * x.ndim
    xt, gt = inp(x, "X"), inp(gamma, "gamma")
    if zero_centered_gamma:
        one = (
            inp(torch.ones(scalar_shape, dtype=torch.float32), "one")
            if host_one
            else g.tensor_scalar(1.0, cudnn.scalar_type.RUNTIME_PARAM).set_dim(scalar_shape).set_stride(scalar_shape).set_data_type(cudnn.data_type.FLOAT)
        )
        gt = g.add(one, gt) if swap else g.add(gt, one)
        gt.set_data_type(add_dtype).set_dim(gamma.shape).set_stride(gamma.stride())
    if phase == "fprop":
        epsilon = g.tensor_scalar(1e-3, cudnn.scalar_type.RUNTIME_PARAM).set_dim(scalar_shape).set_stride(scalar_shape).set_data_type(cudnn.data_type.FLOAT)
        kw = dict(
            input=xt,
            scale=gt,
            bias=inp(beta, "beta"),
            epsilon=epsilon,
            norm_forward_phase=cudnn.norm_forward_phase.INFERENCE if inference else cudnn.norm_forward_phase.TRAINING,
        )
        if kind == "rms":
            yt, rt = g.rmsnorm(**kw, adaptive=adaptive)
            mt = None
        elif kind == "in":
            yt, mt, rt = g.instancenorm(**kw, adaptive=adaptive)
        elif legacy_ada:
            # Legacy AdaLN infers reduction axes from scale; callers can provide
            # per-token statistics explicitly for sequence-first input.
            yt, mt, rt = g.adalayernorm(**kw)
            if adaptive:
                stats = torch.empty((*x.shape[:-1], 1))
                for t in (mt, rt):
                    if t is not None:
                        t.set_dim(stats.shape).set_stride(stats.stride())
        else:
            yt, mt, rt = g.layernorm(**kw, adaptive=adaptive)
        result = {"y": torch.full_like(x, float("nan"))}
        vp[_output(yt, result["y"])] = result["y"]
        for name, t in (("mean", mt), ("rstd", rt)):
            if t is not None:
                stats_shape = (*x.shape[:2], *((1,) * (x.ndim - 2))) if kind == "in" else (*x.shape[:-1], 1)
                buf = torch.full(stats_shape, float("nan"), dtype=torch.float32, device=x.device)
                result[name] = buf
                vp[t.set_output(True).set_data_type(buf.dtype)] = buf
    else:
        kw = dict(input=xt, scale=gt, grad=inp(dy, "dy"), inv_variance=inp(rstd, "rstd"))
        if kind == "rms":
            outs = g.rmsnorm_backward(**kw, has_dbias=beta is not None, adaptive=adaptive)
        elif kind == "in":
            outs = g.instancenorm_backward(**kw, mean=inp(mean, "mean"), adaptive=adaptive)
        elif legacy_ada:
            outs = g.adalayernorm_backward(**kw, mean=inp(mean, "mean"))
        else:
            outs = g.layernorm_backward(**kw, mean=inp(mean, "mean"), adaptive=adaptive)
        result = {}
        for name, t, buf in zip(
            ("dx", "dgamma", "dbeta"),
            outs,
            (
                torch.full_like(x, float("nan")),
                torch.full_like(gamma, float("nan"), dtype=torch.float32),
                torch.full_like(gamma, float("nan"), dtype=torch.float32),
            ),
        ):
            if t is not None:
                result[name] = buf
                vp[_output(t, buf)] = buf
    return g, vp, result


def _reference(kind, x, gamma, beta, zero_centered_gamma, add_dtype):
    xf = x.float().detach().requires_grad_()
    gf = gamma.float().detach().requires_grad_()
    bf = beta.float().detach().requires_grad_() if beta is not None else None
    axes = tuple(range(2, xf.ndim)) if kind == "in" else (-1,)
    mean = xf.mean(axes, keepdim=True) if kind != "rms" else torch.zeros_like(xf[..., :1])
    var = (xf - mean).square().mean(axes, keepdim=True)
    rstd = (var + 1e-3).rsqrt()
    # Quantize the forward value while retaining the identity derivative of add.
    effective = gf
    if zero_centered_gamma:
        shifted = gf + 1
        effective = shifted + (shifted.to(add_dtype).float() - shifted).detach()
    if kind == "in":
        normalized = torch.nn.functional.instance_norm(xf, eps=1e-3)
    elif kind == "ln":
        normalized = torch.nn.functional.layer_norm(xf, (xf.shape[-1],), eps=1e-3)
    else:
        normalized = torch.nn.functional.rms_norm(xf, (xf.shape[-1],), eps=1e-3)
    y = normalized * effective
    if bf is not None:
        y = y + bf
    return y, mean, rstd, (xf, gf, bf)


def _loss_grad(yref, x):
    """Use the norm notebooks' mean-MSE loss and the kernel's DY dtype."""
    loss = torch.nn.functional.mse_loss(yref, torch.randn_like(x).float(), reduction="mean")
    return torch.autograd.grad(loss, yref, retain_graph=True)[0].to(x.dtype)


@pytest.mark.parametrize("kind", ["ln", "rms"])
@pytest.mark.parametrize(
    "layout,zero_centered_gamma", [("batch_first", False), ("sequence_first", False), ("batch_first", True), ("sequence_first", True), ("ordinary", True)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_frost_norm_affine_forward_backward(cudnn_handle, kind, layout, dtype, zero_centered_gamma):
    torch.manual_seed(21)
    shape = (3, 5, 128)
    scale_shape = {"batch_first": (3, 1, 128), "sequence_first": (1, 5, 128), "ordinary": (1, 1, 128)}[layout]
    x = torch.randn(shape, device="cuda", dtype=dtype)
    gamma = torch.randn(scale_shape, device="cuda", dtype=dtype)
    beta = torch.randn_like(gamma) if kind == "ln" else None
    add_dtype = dtype if layout == "sequence_first" else torch.float32
    yref, mean, rstd, leaves = _reference(kind, x, gamma, beta, zero_centered_gamma, add_dtype)
    dy = _loss_grad(yref, x)
    expected_grads = torch.autograd.grad(yref, [t for t in leaves if t is not None], dy.float())
    adaptive = layout != "ordinary"
    g, vp, result = _graph(
        cudnn_handle,
        kind,
        x,
        gamma,
        beta,
        adaptive=adaptive,
        zero_centered_gamma=zero_centered_gamma,
        add_dtype=add_dtype,
        swap=layout == "sequence_first",
        host_one=layout == "batch_first",
    )
    _build(g, "fprop")
    bg, bvp, grads = _graph(
        cudnn_handle,
        kind,
        x,
        gamma,
        beta,
        phase="bprop",
        adaptive=adaptive,
        zero_centered_gamma=zero_centered_gamma,
        add_dtype=add_dtype,
        dy=dy,
        mean=mean.detach(),
        rstd=rstd.detach(),
    )
    _build(bg, "bprop")
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.current_stream()
    old = cudnn.get_stream(cudnn_handle)
    cudnn.set_stream(cudnn_handle, stream.cuda_stream)
    try:
        g.execute(vp, workspace, handle=cudnn_handle)
        bg.execute(bvp, workspace, handle=cudnn_handle)
    finally:
        cudnn.set_stream(cudnn_handle, old)
    # Keep the output dtype: PyTorch uses its defaults for Y/DX and FP32
    # defaults for saved statistics and parameter gradients.
    torch.testing.assert_close(result["y"], yref.detach(), check_dtype=False)
    torch.testing.assert_close(result["rstd"], rstd.detach(), check_dtype=False)
    if kind == "ln":
        torch.testing.assert_close(result["mean"], mean.detach(), check_dtype=False)
    for (name, got), expected in zip(grads.items(), expected_grads):
        torch.testing.assert_close(got, expected, check_dtype=False)


@pytest.mark.parametrize("kind", ["ln", "rms", "in"])
def test_frost_norm_affine_inference_capture(cudnn_handle, monkeypatch, kind):
    x = torch.randn(2, 7, 96, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn((2, 7, 1) if kind == "in" else (2, 1, 96), device="cuda", dtype=torch.float32)
    g, vp, result = _graph(cudnn_handle, kind, x, gamma, None, adaptive=True, zero_centered_gamma=True, inference=True)
    _build(g, "fprop")
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
    old = cudnn.get_stream(cudnn_handle)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph = None
    try:
        cudnn.set_stream(cudnn_handle, capture_stream.cuda_stream)
        with torch.cuda.stream(capture_stream):
            # There must be no deferred compilation or allocation, including on
            # the first execute of an inference plan without statistics outputs.
            import cutlass.cute as cute

            def forbidden(*args, **kwargs):
                raise AssertionError("execute compiled or allocated")

            with monkeypatch.context() as mp:
                mp.setattr(cute, "compile", forbidden)
                for op in ("empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like"):
                    mp.setattr(torch, op, forbidden)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    g.execute(vp, workspace, handle=cudnn_handle)
            x.mul_(0.5)
            gamma.add_(0.25)
            graph.replay()
        torch.cuda.current_stream().wait_stream(capture_stream)
        ref, _, _, _ = _reference(kind, x, gamma, None, True, torch.float32)
        torch.testing.assert_close(result["y"], ref.detach(), check_dtype=False)
    finally:
        if graph is not None:
            graph.reset()
        cudnn.set_stream(cudnn_handle, old)


@pytest.mark.parametrize("kind", ["ln", "rms"])
@pytest.mark.parametrize("shape,scale_shape", [((1, 3, 1), (1, 1, 1)), ((1, 3, 33), (1, 3, 33)), ((3, 1, 33), (3, 1, 33)), ((2, 3, 32768), (1, 3, 32768))])
def test_frost_norm_affine_row_boundaries(cudnn_handle, kind, shape, scale_shape):
    x = torch.randn(shape, device="cuda")
    gamma = torch.randn(scale_shape, device="cuda")
    beta = torch.randn_like(gamma)
    g, vp, result = _graph(cudnn_handle, kind, x, gamma, beta, adaptive=True, zero_centered_gamma=True)
    stats_shape = (*shape[:-1], 1)
    node = g.nodes[-1]
    _build(g, "fprop")
    for key in ("mean", "inv_var"):
        if node.outputs.get(key) is not None:
            assert tuple(node.outputs[key].get_dim()) == stats_shape
    old = cudnn.get_stream(cudnn_handle)
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    try:
        g.execute(vp, torch.empty(0, device="cuda", dtype=torch.uint8), handle=cudnn_handle)
    finally:
        cudnn.set_stream(cudnn_handle, old)
    ref, _, _, leaves = _reference(kind, x, gamma, beta, True, torch.float32)
    torch.testing.assert_close(result["y"], ref.detach(), check_dtype=False)
    dy = _loss_grad(ref, x)
    expected = torch.autograd.grad(ref, leaves, dy)
    bg, bvp, grads = _graph(
        cudnn_handle,
        kind,
        x,
        gamma,
        beta,
        phase="bprop",
        adaptive=True,
        zero_centered_gamma=True,
        dy=dy,
        mean=result.get("mean"),
        rstd=result["rstd"],
    )
    _build(bg, "bprop")
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    try:
        bg.execute(bvp, torch.empty(0, device="cuda", dtype=torch.uint8), handle=cudnn_handle)
    finally:
        cudnn.set_stream(cudnn_handle, old)
    for got, want in zip(grads.values(), expected):
        torch.testing.assert_close(got, want, check_dtype=False)


@pytest.mark.parametrize("bad", ["dtype", "stride", "missing", "overlap", "offset"])
def test_frost_norm_affine_rejects_bindings_before_writes(cudnn_handle, bad):
    x = torch.randn(2, 3, 64, device="cuda")
    gamma = torch.randn(2, 1, 64, device="cuda")
    mean = x.mean(-1, keepdim=True)
    rstd = (x.var(-1, correction=0, keepdim=True) + 1e-3).rsqrt()
    g, vp, result = _graph(
        cudnn_handle, "ln", x, gamma, gamma, phase="bprop", adaptive=True, zero_centered_gamma=True, dy=torch.randn_like(x), mean=mean, rstd=rstd, host_one=True
    )
    _build(g, "bprop")
    bad_pack = dict(vp)
    gamma_t = next(t for t in vp if t.get_name() == "gamma")
    if bad == "dtype":
        bad_pack[gamma_t] = gamma.half()
    elif bad == "stride":
        bad_pack[gamma_t] = torch.randn(2, 1, 128, device="cuda")[..., ::2]
    elif bad == "missing":
        del bad_pack[gamma_t]
    elif bad == "overlap":
        db = next(t for t, buf in vp.items() if buf is result["dbeta"])
        bad_pack[db] = result["dgamma"]
    else:
        one = next(t for t in vp if t.get_name() == "one")
        bad_pack[one] = torch.full((1, 1, 1), 2.0)
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
    with pytest.raises((ValueError, RuntimeError), match="binding|storage|offset|missing|required"):
        g.execute(bad_pack, workspace, handle=cudnn_handle)
    for buf in result.values():
        assert torch.isnan(buf).all()


@pytest.mark.parametrize("mutation", ["observable_scale", "extra_node", "non_scalar", "wrong_constant", "strided_x", "bad_batch", "compute_half"])
def test_frost_norm_affine_declines_unsupported_graphs(cudnn_handle, mutation):
    from cudnn.norm.fprop.engine import FrostNormFwdEngine
    from cudnn.norm.fprop.engines import ENGINE_SPECS

    x = torch.randn(2, 3, 64, device="cuda")
    gamma = torch.randn(2, 1, 64, device="cuda")
    g, vp, _ = _graph(cudnn_handle, "ln", x, gamma, gamma, adaptive=True, zero_centered_gamma=True)
    add, norm = g.nodes
    if mutation == "observable_scale":
        norm.inputs["scale"].set_output(True)
    elif mutation == "extra_node":
        g.relu(norm.outputs["Y"])
    elif mutation == "non_scalar":
        one = list(add.inputs.values())[1]
        one.set_is_pass_by_value(False).set_dim(gamma.shape).set_stride(gamma.stride())
    elif mutation == "wrong_constant":
        list(add.inputs.values())[1].pass_by_value = 2.0
    elif mutation == "strided_x":
        norm.inputs["input"].set_stride([384, 128, 2])
    elif mutation == "bad_batch":
        norm.inputs["scale"].set_dim([2, 2, 64])
    else:
        norm.compute_data_type = cudnn.data_type.HALF
    engine = FrostNormFwdEngine(ENGINE_SPECS[0], 20_000)
    with pytest.raises(NotImplementedError):
        engine.check_support(g)


def test_frost_norm_affine_mutable_graph_and_rms_default(cudnn_handle):
    from cudnn.norm.graph_analyzer import analyze

    x = torch.randn(2, 3, 64, device="cuda")
    gamma = torch.randn(2, 1, 64, device="cuda")
    g, _, _ = _graph(cudnn_handle, "rms", x, gamma, None, adaptive=True)
    assert analyze(g).adaptive
    g.nodes[0].params["adaptive"] = False
    assert not analyze(g).adaptive
    assert g._backend_lowerable()


def test_frost_norm_affine_replan_reloads_artifact(cudnn_handle, monkeypatch, tmp_path):
    import cutlass.cute as cute

    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    x = torch.randn(2, 3, 80, device="cuda")
    gamma = torch.randn(2, 1, 80, device="cuda")
    g, _, _ = _graph(cudnn_handle, "rms", x, gamma, None, adaptive=True, zero_centered_gamma=True, inference=True)
    _build(g, "fprop")
    assert list(tmp_path.rglob("kernel.o"))
    g2, vp, out = _graph(cudnn_handle, "rms", x, gamma, None, adaptive=True, zero_centered_gamma=True, inference=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("second plan recompiled")

    monkeypatch.setattr(cute, "compile", forbidden)
    _build(g2, "fprop")
    old = cudnn.get_stream(cudnn_handle)
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    try:
        g2.execute(vp, torch.empty(0, device="cuda", dtype=torch.uint8), handle=cudnn_handle)
    finally:
        cudnn.set_stream(cudnn_handle, old)
    ref, _, _, _ = _reference("rms", x, gamma, None, True, torch.float32)
    torch.testing.assert_close(out["y"], ref.detach(), check_dtype=False)


@pytest.mark.parametrize("kind", ["ln", "rms", "in"])
def test_frost_norm_affine_backward_capture_and_rebind(cudnn_handle, monkeypatch, kind):
    x = torch.randn(2, 3, 192, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn((2, 3, 1) if kind == "in" else (1, 3, 192), device="cuda", dtype=torch.float32)
    beta = torch.randn_like(gamma)
    ref, mean, rstd, leaves = _reference(kind, x, gamma, beta, True, torch.float32)
    dy = _loss_grad(ref, x)
    expected = torch.autograd.grad(ref, leaves, dy.float())
    g, vp, outputs = _graph(
        cudnn_handle, kind, x, gamma, beta, phase="bprop", adaptive=True, zero_centered_gamma=True, dy=dy, mean=mean.detach(), rstd=rstd.detach()
    )
    _build(g, "bprop")
    rebound = {t: buf.clone() for t, buf in vp.items()}
    rebound_outputs = [rebound[t] for t, buf in vp.items() if any(buf is out for out in outputs.values())]
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    old = cudnn.get_stream(cudnn_handle)
    captured = None
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        with torch.cuda.stream(stream):
            import cutlass.cute as cute

            def forbidden(*args, **kwargs):
                raise AssertionError("backward execute compiled or allocated")

            with monkeypatch.context() as mp:
                mp.setattr(cute, "compile", forbidden)
                for op in ("empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like"):
                    mp.setattr(torch, op, forbidden)
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured, stream=stream):
                    g.execute(vp, workspace, handle=cudnn_handle)
                # Each replay must clear the atomic parameter reductions.
                captured.replay()
                captured.replay()
                g.execute(rebound, workspace, handle=cudnn_handle)
        torch.cuda.current_stream().wait_stream(stream)
        for buffers in (outputs.values(), rebound_outputs):
            for got, want in zip(buffers, expected):
                torch.testing.assert_close(got, want, check_dtype=False)
    finally:
        if captured is not None:
            captured.reset()
        cudnn.set_stream(cudnn_handle, old)


@pytest.mark.parametrize(
    "kind,adaptive,zero_centered_gamma,sequence_first",
    [("ln", True, False, False), ("ln", True, False, True), ("ln", False, True, False), ("rms", False, True, False)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inference", [False, True])
def test_frost_norm_affine_backend_parity(cudnn_handle, kind, adaptive, zero_centered_gamma, sequence_first, dtype, inference):
    shape = (2, 64, 256) if adaptive else (128, 1, 256)
    if sequence_first:
        shape = (64, 2, 256)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    scale_shape = (1, 2, 256) if sequence_first else (2 if adaptive else 1, 1, 256)
    gamma = torch.randn(scale_shape, device="cuda", dtype=dtype)
    beta = torch.randn_like(gamma)
    kwargs = dict(adaptive=adaptive, zero_centered_gamma=zero_centered_gamma, host_one=True, inference=inference)
    g, vp, outputs = _graph(cudnn_handle, kind, x, gamma, beta, **kwargs)
    _build(g, "fprop")
    backend, bvp, reference = _graph(cudnn_handle, kind, x, gamma, beta, **kwargs)
    backend.validate()
    backend.build_operation_graph()
    backend.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    candidates = [i for i, p in enumerate(backend.plans) if p.engine_id < 20_000]
    for idx in candidates:
        backend.select_plan(idx)
        try:
            backend.check_support()
            backend.build_plans()
        except (RuntimeError, cudnn.cudnnGraphNotSupportedError):
            continue
        break
    else:
        pytest.skip("installed cuDNN backend has no plan for this affine graph")
    assert backend.plans[idx].engine_id < 20_000
    old = cudnn.get_stream(cudnn_handle)
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    try:
        g.execute(vp, torch.empty(0, device="cuda", dtype=torch.uint8), handle=cudnn_handle)
        backend.execute(bvp, torch.empty(backend.get_workspace_size(), device="cuda", dtype=torch.uint8), handle=cudnn_handle)
    finally:
        cudnn.set_stream(cudnn_handle, old)
    for name in outputs:
        torch.testing.assert_close(outputs[name], reference[name], check_dtype=False)


@pytest.mark.parametrize("shape", [(2, 3, 33), (3, 5, 7, 9), (2, 4, 2, 3, 4), (1, 2, 128, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias,zero_centered_gamma", [(True, False), (False, True)])
def test_frost_adain_forward_backward(cudnn_handle, shape, dtype, has_bias, zero_centered_gamma):
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    affine_shape = (*shape[:2], *((1,) * (len(shape) - 2)))
    # Mixed affine/input types, including a zero effective scale in one row.
    gamma = torch.randn(affine_shape, device="cuda", dtype=torch.float32)
    gamma.flatten()[0] = -1.0 if zero_centered_gamma else 0.0
    beta = torch.randn_like(gamma) if has_bias else None
    ref, mean, rstd, leaves = _reference("in", x, gamma, beta, zero_centered_gamma, dtype)
    if not has_bias:
        # InstanceNorm backward always exposes DBias, even without a forward bias.
        bf = torch.zeros_like(gamma, requires_grad=True)
        leaves = (*leaves[:2], bf)
        ref = ref + bf
    dy = _loss_grad(ref, x)
    expected = torch.autograd.grad(ref, leaves, dy.float())
    g, vp, outputs = _graph(cudnn_handle, "in", x, gamma, beta, adaptive=True, zero_centered_gamma=zero_centered_gamma, add_dtype=dtype)
    assert not g._backend_lowerable()
    assert tuple(g.nodes[-1].outputs["mean"].get_dim()) == affine_shape
    _build(g, "fprop")
    bg, bvp, grads = _graph(
        cudnn_handle,
        "in",
        x,
        gamma,
        beta,
        phase="bprop",
        adaptive=True,
        zero_centered_gamma=zero_centered_gamma,
        add_dtype=dtype,
        dy=dy,
        mean=mean.detach(),
        rstd=rstd.detach(),
    )
    assert not bg._backend_lowerable()
    _build(bg, "bprop")
    old = cudnn.get_stream(cudnn_handle)
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    try:
        workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
        g.execute(vp, workspace, handle=cudnn_handle)
        bg.execute(bvp, workspace, handle=cudnn_handle)
    finally:
        cudnn.set_stream(cudnn_handle, old)
    torch.testing.assert_close(outputs["y"], ref.detach(), check_dtype=False)
    torch.testing.assert_close(outputs["mean"], mean.detach(), check_dtype=False)
    torch.testing.assert_close(outputs["rstd"], rstd.detach(), check_dtype=False)
    for got, want in zip(grads.values(), expected):
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, check_dtype=False)


def _backend_declaration(g):
    declaration = json.loads(repr(g._lower_to_cpp()))
    declaration.pop("gid")  # Backend graph IDs are unique even for identical declarations.
    return declaration


@pytest.mark.parametrize("phase", ["fprop", "bprop"])
@pytest.mark.parametrize("scale_shape", [(2, 1, 64), (1, 3, 64)])
def test_frost_layernorm_adaptive_alias(cudnn_handle, phase, scale_shape):
    x = torch.randn(2, 3, 64, device="cuda")
    gamma = torch.randn(scale_shape, device="cuda")
    _, mean, rstd, _ = _reference("ln", x, gamma, gamma, False, torch.float32)
    kw = dict(phase=phase, adaptive=True, dy=torch.randn_like(x), mean=mean.detach(), rstd=rstd.detach())
    modern, _, _ = _graph(cudnn_handle, "ln", x, gamma, gamma, **kw)
    legacy, _, _ = _graph(cudnn_handle, "ln", x, gamma, gamma, legacy_ada=True, **kw)
    assert modern.inspect() == legacy.inspect()
    assert modern._backend_lowerable() and legacy._backend_lowerable()
    # Both public spellings must lower to the same existing backend AdaLN op.
    assert _backend_declaration(modern) == _backend_declaration(legacy)


@pytest.mark.parametrize("kind", ["ln", "rms", "in"])
@pytest.mark.parametrize("phase", ["fprop", "bprop"])
def test_frost_norm_nonadaptive_backend_routing(cudnn_handle, kind, phase):
    from cudnn.norm.graph_analyzer import analyze

    x = torch.randn(2, 3, 64, device="cuda")
    gamma = torch.randn((1, 3, 1) if kind == "in" else (1, 1, 64), device="cuda")
    _, mean, rstd, _ = _reference(kind, x, gamma, gamma, False, torch.float32)
    g, _, _ = _graph(cudnn_handle, kind, x, gamma, gamma, phase=phase, adaptive=False, dy=torch.randn_like(x), mean=mean.detach(), rstd=rstd.detach())
    assert not analyze(g).adaptive
    assert g._backend_lowerable()
    # Lowering freezes the graph, so compare a separate default declaration.
    default, _, _ = _graph(cudnn_handle, kind, x, gamma, gamma, phase=phase, dy=torch.randn_like(x), mean=mean.detach(), rstd=rstd.detach())
    default.nodes[0].params.pop("adaptive", None)
    assert _backend_declaration(g) == _backend_declaration(default)


@pytest.mark.parametrize("bad", ["shared_scale", "spatial_scale", "channels_last", "missing_mean"])
def test_frost_adain_declines_unsupported(cudnn_handle, bad):
    from cudnn.norm.bprop.engine import FrostNormBwdEngine
    from cudnn.norm.bprop.engines import ENGINE_SPECS

    x = torch.randn(2, 3, 7, 9, device="cuda")
    gamma = torch.randn(2, 3, 1, 1, device="cuda")
    _, mean, rstd, _ = _reference("in", x, gamma, gamma, False, torch.float32)
    g, _, _ = _graph(cudnn_handle, "in", x, gamma, gamma, phase="bprop", adaptive=True, dy=torch.randn_like(x), mean=mean.detach(), rstd=rstd.detach())
    node = g.nodes[-1]
    if bad == "shared_scale":
        node.inputs["scale"].set_dim([1, 3, 1, 1])
    elif bad == "spatial_scale":
        node.inputs["scale"].set_dim([2, 3, 7, 9])
    elif bad == "channels_last":
        node.inputs["input"].set_stride([189, 1, 27, 3])
    else:
        del node.inputs["mean"]
    engine = FrostNormBwdEngine(ENGINE_SPECS[0], 20_000)
    with pytest.raises(NotImplementedError):
        engine.check_support(g)


@pytest.mark.parametrize("batch", [1, 2])
def test_frost_adalayernorm_preserves_multiaxis_stats(cudnn_handle, batch):
    x = torch.randn(batch, 3, 64, device="cuda")
    gamma = torch.ones_like(x)
    g, _, _ = _graph(cudnn_handle, "ln", x, gamma, gamma, legacy_ada=True)
    # Preserve the legacy scale-based inference, including singleton batches.
    for role in ("mean", "inv_var"):
        assert g.nodes[-1].outputs[role].get_dim() == [batch, 1, 1]
    assert g._backend_lowerable()
