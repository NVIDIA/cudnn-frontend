# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SSD numerics and graph/autograd integration, independent FP64 oracle."""

import importlib

import pytest
import torch
import torch.nn.functional as F

import cudnn
from cudnn.linear_attention import mamba2

ops = importlib.import_module("cudnn.linear_attention.ops.mamba2")
pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Mamba2FrostEngine currently supports SM100")
    from cudnn.frost import buffers

    installed, version = buffers.cutedsl_state()
    if not installed or buffers.cutedsl_too_old(version):
        pytest.skip("Mamba2FrostEngine requires CuTeDSL >= 4.7.0")


def inputs(length=33, heads=4, groups=2, mode="full", batch=1, state_size=128):
    torch.manual_seed(1729)

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", dtype=dtype)

    values = dict(
        x=rand((batch, length, heads, 64)),
        dt=rand((batch, length, heads)) - 2,
        A=-torch.rand(heads, device="cuda") - 0.1,
        B=rand((batch, length, groups, state_size)),
        C=rand((batch, length, groups, state_size)),
    )
    if mode != "bare":
        values.update(D=rand((heads,), torch.float32), dt_bias=rand((heads,), torch.float32) - 1)
    if mode in ("gated", "full"):
        values["z"] = rand(values["x"].shape)
    if mode in ("full", "state"):
        values["initial_state"] = rand((batch, heads, 64, state_size), torch.float32)
    return {name: t.requires_grad_() for name, t in values.items()}


def reference(values):
    v = {name: t.double() for name, t in values.items()}
    x, dt, A, B, C = (v[k] for k in ("x", "dt", "A", "B", "C"))
    b, l, h, p = x.shape
    B = B.repeat_interleave(h // B.shape[2], dim=2)
    C = C.repeat_interleave(h // C.shape[2], dim=2)
    delta = F.softplus(dt + v.get("dt_bias", 0))
    state = v.get("initial_state", torch.zeros((b, h, p, B.shape[-1]), device=x.device, dtype=torch.float64))
    outputs = []
    for t in range(l):
        state = state * torch.exp(delta[:, t] * A)[..., None, None] + delta[:, t, ..., None, None] * x[:, t, ..., None] * B[:, t, :, None, :]
        y = (state * C[:, t, :, None, :]).sum(-1)
        if "D" in v:
            y = y + x[:, t] * v["D"][None, :, None]
        if "z" in v:
            y = y * F.silu(v["z"][:, t])
        outputs.append(y)
    return torch.stack(outputs, dim=1), state


def assert_error(actual, expected):
    a, e = actual.double(), expected.double()
    assert torch.isfinite(a).all()
    diff = a - e
    rms = diff.square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-10)
    peak = diff.abs().max() / e.abs().max().clamp_min(1e-10)
    assert rms < 0.01, f"relative RMS error {rms.item()}"
    assert peak < 0.015, f"relative peak error {peak.item()}"


@pytest.mark.parametrize(
    "length,mode,precision,reuse",
    [
        (1, "bare", "float32", False),
        (31, "gated", "float32", False),
        (32, "full", "float32", True),
        (33, "full", "float32", False),
        (33, "state", "bfloat16", True),
        (65, "skip", "bfloat16", False),
        (128, "bare", "bfloat16", False),
        (128, "gated", "float32", True),
        (128, "gated", "float32", False),
        (129, "state", "bfloat16", False),
    ],
)
def test_numerics(length, mode, precision, reuse):
    values = inputs(length, mode=mode)
    refs = {name: t.detach().double().requires_grad_() for name, t in values.items()}
    out, final = mamba2(**values, return_final_state=True, intermediate_dtype=precision, reuse_forward_states=reuse)
    expected, expected_final = reference(refs)
    assert_error(out, expected)
    assert_error(final, expected_final)
    dy = torch.randn_like(out)
    ds = torch.randn_like(final)
    grads = torch.autograd.grad((out, final), tuple(values.values()), (dy, ds))
    ref_grads = torch.autograd.grad((expected, expected_final), tuple(refs.values()), (dy.double(), ds.double()))
    for name, actual, ref in zip(values, grads, ref_grads):
        try:
            assert_error(actual, ref)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc


@pytest.mark.parametrize("state_only", [False, True])
def test_one_output_loss(state_only):
    values = inputs(33)
    refs = {name: t.detach().double().requires_grad_() for name, t in values.items()}
    outputs = mamba2(**values, return_final_state=True)
    expected = reference(refs)
    idx = int(state_only)
    outputs[idx].float().sum().backward()
    expected[idx].sum().backward()
    for name, t in values.items():
        if refs[name].grad is None:
            assert t.grad is None or torch.count_nonzero(t.grad) == 0
        else:
            assert_error(t.grad, refs[name].grad)


def prepared(values):
    with torch.cuda.device(values["x"].device):
        graph, ports, handle = ops._get_graph(values, False, 32, "float32", True)
        out = ops._outputs(values, False, True)
        all_values = {**values, **out}
        pack = {p: all_values[name] for name, p in ports.items()}
        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    return graph, ports, handle, out, pack, workspace


def test_graph_rebind_capture_and_allocation_free_execute(monkeypatch):
    values = inputs(33)
    graph, ports, handle, out, pack, workspace = prepared(values)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(handle=handle, stream=stream.cuda_stream)

    def no_allocation(*args, **kwargs):
        raise AssertionError("allocation or compilation during graph.execute")

    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "zeros_like"):
            patch.setattr(torch, name, no_allocation)
        engine = importlib.import_module("cudnn.linear_attention.frost.mamba2_engine")
        patch.setattr(engine, "_compile_step", no_allocation)
        with torch.cuda.stream(stream):
            graph.execute(pack, workspace=workspace, handle=handle)
    stream.synchronize()
    assert_error(out["O"], reference(values)[0])
    changed = {**values, "x": (-values["x"]).detach()}
    pack[ports["x"]] = changed["x"]
    with torch.cuda.stream(stream):
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            graph.execute(pack, workspace=workspace, handle=handle)
        captured.replay()
    stream.synchronize()
    assert_error(out["O"], reference(changed)[0])
    with pytest.raises(ValueError, match="workspace"):
        graph.execute(pack, workspace=workspace[:1], handle=handle)


def test_concurrent_streams():
    values = inputs(33)
    mamba2(**values)  # compile before stream concurrency
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    changed = {**values, "x": (-values["x"]).detach()}
    results = []
    for stream, v in zip(streams, (values, changed)):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            results.append(mamba2(**v))
    for stream in streams:
        stream.synchronize()
    for actual, v in zip(results, (values, changed)):
        assert_error(actual, reference(v)[0])


def test_declines_and_dependency_gate(monkeypatch):
    values = inputs(33)
    graph, _, _, _, _, _ = prepared(values)
    engine = graph.selected_engine
    from cudnn.frost import buffers

    with monkeypatch.context() as patch:
        patch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.6.2")))
        with pytest.raises(NotImplementedError, match="4.7.0"):
            engine.check_support(graph)
    with monkeypatch.context() as patch:
        patch.setattr(buffers, "current_sm", lambda: 90)
        with pytest.raises(NotImplementedError, match="SM100"):
            engine.check_support(graph)
    for kwargs in (dict(chunk_size=64), dict(intermediate_dtype="float16")):
        with pytest.raises(ValueError):
            mamba2(**values, **kwargs)
    invalid = {**values, "x": values["x"].transpose(-1, -2).contiguous().transpose(-1, -2)}
    with pytest.raises(ValueError, match="contiguous"):
        mamba2(**invalid)
    invalid = {**values, "dt": values["dt"].float()}
    with pytest.raises(ValueError, match="dt"):
        mamba2(**invalid)


def test_torch_compile_training():
    values = inputs(33)

    def model(x, dt, A, B, C, D, dt_bias, z, initial_state):
        o, s = mamba2(x, dt, A, B, C, D, dt_bias, z, initial_state, return_final_state=True, intermediate_dtype="float32")
        return o.float().square().mean() + s.square().mean()

    eager = model(**values)
    eager_grads = torch.autograd.grad(eager, tuple(values.values()))
    compiled = torch.compile(model, fullgraph=True)
    actual = compiled(**values)
    grads = torch.autograd.grad(actual, tuple(values.values()))
    torch.testing.assert_close(actual, eager)
    for a, e in zip(grads, eager_grads):
        torch.testing.assert_close(a, e)


def test_plan_identity_and_optional_contract():
    values = inputs(33)
    out = mamba2(**values, plan_name="mamba2_frost")
    assert_error(out, reference(values)[0])
    graph, _, _, _, _, _ = prepared(values)
    assert graph.selected_engine.name == "mamba2_frost"
    assert graph.selected_engine.engine_id == 20900
    backward_graph = cudnn.pygraph()
    ports = {name: backward_graph.tensor(list(t.shape), data_type=ops.torch_dtype_to_cudnn(t.dtype), name=name) for name, t in values.items()}
    dO = backward_graph.tensor(list(values["x"].shape), data_type=cudnn.data_type.BFLOAT16, name="dO")
    backward_graph.mamba2_bwd(**ports, dO=dO)  # z without saved ungated output
    with pytest.raises(NotImplementedError, match="provided together"):
        graph.selected_engine.check_support(backward_graph)


def test_public_training_capture():
    values = inputs(33)

    def step():
        out, final = mamba2(**values, return_final_state=True)
        return torch.autograd.grad((out.float().square().mean() + final.square().mean()), tuple(values.values()))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = step()
        step()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            actual = step()
        for _ in range(3):
            captured.replay()
    stream.synchronize()
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("state_size", [64, 96, 256])
def test_declines_non_nemotron_state_size(state_size):
    from cudnn.linear_attention.frost.mamba2_engine import Mamba2FrostEngine

    values = inputs(33, mode="bare", state_size=state_size)
    graph = cudnn.pygraph()
    ports = {name: graph.tensor(list(t.shape), data_type=ops.torch_dtype_to_cudnn(t.dtype), name=name) for name, t in values.items()}
    graph.mamba2(**ports)
    with pytest.raises(NotImplementedError, match="state_dim=128"):
        Mamba2FrostEngine().check_support(graph)


@pytest.mark.parametrize("heads", [64, 128], ids=["nemotron_nano", "nemotron_super"])
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_nemotron_ssd_geometry(heads, precision):
    # Nemotron uses a separate GatedRMSNorm after SSD, so z stays absent here.
    # Match its P=64/N=128/G=8 geometry and timestep initialization range.
    values = inputs(65, heads=heads, groups=8, mode="skip")
    values["A"] = (-torch.empty(heads, device="cuda").uniform_(1, 16)).requires_grad_()
    delta = torch.exp(torch.empty(heads, device="cuda").uniform_(-6.9, -2.3))
    values["dt_bias"] = (delta + torch.log(-torch.expm1(-delta))).requires_grad_()
    refs = {name: t.detach().double().requires_grad_() for name, t in values.items()}
    actual = mamba2(**values, return_final_state=True, intermediate_dtype=precision, reuse_forward_states=True)
    expected = reference(refs)
    for a, e in zip(actual, expected):
        assert_error(a, e)
    cotangents = tuple(torch.randn_like(t) for t in actual)
    actual_grads = torch.autograd.grad(actual, tuple(values.values()), cotangents)
    expected_grads = torch.autograd.grad(expected, tuple(refs.values()), tuple(t.double() for t in cotangents))
    for name, a, e in zip(values, actual_grads, expected_grads):
        try:
            assert_error(a, e)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc


@pytest.mark.parametrize("surface", ["graph", "torch"])
def test_declines_bf16_intermediates_with_ssd_gate(surface):
    values = inputs(128, mode="gated")
    if surface == "torch":
        with pytest.raises(ValueError, match="SiLU gate requires intermediate_dtype=float32"):
            mamba2(**values, intermediate_dtype="bfloat16")
    else:
        graph = cudnn.pygraph()
        ports = {name: graph.tensor(list(t.shape), data_type=ops.torch_dtype_to_cudnn(t.dtype), name=name) for name, t in values.items()}
        graph.mamba2(**ports, intermediate_dtype="bfloat16")
        from cudnn.linear_attention.frost.mamba2_engine import Mamba2FrostEngine

        with pytest.raises(NotImplementedError, match="SiLU gate requires intermediate_dtype=float32"):
            Mamba2FrostEngine().check_support(graph)
