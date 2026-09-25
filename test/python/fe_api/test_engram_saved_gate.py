# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Saved-state gate correctness, native packed gradients and lifecycle contracts."""

import pytest
import torch
import cudnn
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

pytestmark = pytest.mark.L1


@pytest.fixture(autouse=True)
def supported():
    if cutedsl_too_old(cutedsl_state()[1]) or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100 and CuTe DSL >=4.7.0")
    pytest.importorskip("triton", minversion="3.7.0")


def tensors(n=128):
    torch.manual_seed(419253 + n)
    x = torch.randn(n, 4, 5120, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(n, 25600, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(4, 5120, device="cuda", dtype=torch.float32) + 0.5
    mask = torch.rand(n, device="cuda") > 0.13
    grad = torch.randn_like(x)
    return x, kv, w, mask, grad


def reference(x, kv, w, mask):
    h = x.float()
    key = kv[:, :20480].view(x.shape).float()
    value = kv[:, 20480:].float()
    scale = torch.rsqrt(h.square().mean(-1) + 1e-20) * torch.rsqrt(key.square().mean(-1) + 1e-20)
    dot = (h * w * key).sum(-1) * scale * 5120**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)).masked_fill(~mask[:, None], 0)
    return (h + gate[..., None] * value[:, None]).to(torch.bfloat16)


def close(actual, expected, limit):
    a, b = actual.double(), expected.double()
    assert torch.isfinite(a).all()
    diff = a - b
    relative = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b).clamp_min(1e-30)
    scaled = diff.abs().max() / b.abs().max().clamp_min(1e-30)
    assert relative <= limit, float(relative)
    assert scaled <= 2 * limit, float(scaled)


def plans_and_outputs(data):
    x, kv, w, mask, grad = data
    out = torch.empty_like(x)
    saved = torch.empty((x.shape[0], 4, 4), device=x.device, dtype=torch.float32)
    forward = cudnn.EngramGateSavedForward(x, kv, w, mask, backend="frost")
    backward = cudnn.EngramGateSavedBackward(x, kv, w, saved, grad, backend="frost")
    forward.compile()
    backward.compile()
    grads = (torch.empty_like(x), torch.empty_like(kv), torch.empty_like(w))
    workspace = backward.allocate_workspace()
    return forward, backward, out, saved, grads, workspace


@pytest.mark.parametrize("n", [64, 192, 2048, 4096, 8192])
@pytest.mark.parametrize("case", ["random", "masked", "low_norm"])
def test_source_forward_and_all_gate_gradients(n, case):
    data = tensors(n)
    x, kv, w, mask, grad = data
    if case == "masked":
        mask.zero_()
    if case == "low_norm":
        x.mul_(1e-4)
        kv.mul_(1e-4)
    forward, backward, out, saved, grads, workspace = plans_and_outputs(data)
    for t in (out, saved, *grads):
        t.fill_(float("nan"))
    workspace.fill_(255)
    leaves = [t.detach().clone().requires_grad_() for t in (x, kv, w)]
    expected = reference(*leaves, mask)
    expected_grads = torch.autograd.grad(expected, leaves, grad)
    forward.execute(x, kv, w, mask, out, saved)
    backward.execute(x, kv, w, saved, grad, *grads, workspace)
    close(out, expected, 0.01)
    for actual, ref in zip(grads, expected_grads, strict=True):
        close(actual, ref, 0.03)
    assert torch.equal(out[~mask], x[~mask])
    if case == "masked":
        assert torch.equal(grads[0], grad)
        assert torch.count_nonzero(grads[1]) == torch.count_nonzero(grads[2]) == 0


def test_wrappers_saved_lifetime_and_rebinding():
    x, kv, w, mask, grad = tensors()
    f = cudnn.engram_gate_saved_forward(x, kv, w, mask)
    b = cudnn.engram_gate_saved_backward(x, kv, w, f["saved"], grad)
    assert tuple(f.keys()) == ("out", "saved")
    assert tuple(b.keys()) == ("grad_x", "grad_kv", "grad_weight")
    first = f["out"].clone()
    x.neg_()
    second = cudnn.engram_gate_saved_forward(x, kv, w, mask)
    assert not torch.equal(first, second["out"])
    assert f["saved"].data_ptr() != second["saved"].data_ptr()
    leaves = [t.detach().clone().requires_grad_() for t in (x, kv, w)]
    ref = reference(*leaves, mask)
    expected = torch.autograd.grad(ref, leaves, grad)
    changed = cudnn.engram_gate_saved_backward(x, kv, w, second["saved"], grad)
    for actual, target in zip(changed.values(), expected, strict=True):
        close(actual, target, 0.03)


def test_graph_changed_inputs_no_jit_allocation_or_sync(monkeypatch):
    from triton.runtime.jit import JITFunction
    import cutlass.cute as cute

    data = tensors()
    x, kv, w, mask, grad = data
    f, b, out, saved, grads, workspace = plans_and_outputs(data)

    # Runtime detectors are verified to reject deliberate prohibited operations.
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected runtime allocation or compilation")

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    f.execute(x, kv, w, mask, out, saved)
    b.execute(x, kv, w, saved, grad, *grads, workspace)
    torch.cuda.synchronize()
    with monkeypatch.context() as m:
        m.setattr(JITFunction, "run", forbidden)
        m.setattr(cute, "compile", forbidden)
        m.setattr(torch, "empty", forbidden)
        m.setattr(torch, "empty_like", forbidden)
        with pytest.raises(AssertionError):
            torch.empty(1, device="cuda")
        torch.cuda.set_sync_debug_mode("error")
        try:
            with pytest.raises(RuntimeError):
                x[0, 0, 0].item()
            # Explicit raw launch stream differs from torch's current stream.
            f.execute(x, kv, w, mask, out, saved, current_stream=stream.cuda_stream)
            b.execute(x, kv, w, saved, grad, *grads, workspace, current_stream=stream.cuda_stream)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        with torch.cuda.graph(graph, stream=stream):
            f.execute(x, kv, w, mask, out, saved)
            b.execute(x, kv, w, saved, grad, *grads, workspace)
    stream.synchronize()
    first = out.clone()
    x.neg_()
    grad.neg_()
    out.fill_(float("nan"))
    saved.fill_(float("nan"))
    workspace.fill_(255)
    for tensor in grads:
        tensor.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(first, out)
    leaves = [t.detach().clone().requires_grad_() for t in (x, kv, w)]
    expected = reference(*leaves, mask)
    targets = torch.autograd.grad(expected, leaves, grad)
    close(out, expected, 0.01)
    for actual, target in zip(grads, targets, strict=True):
        close(actual, target, 0.03)


def test_metadata_only_support_and_errors():
    from cudnn.api_base import TensorDesc
    from cudnn.engram.api import EngramGateSavedForward

    x, kv, w, mask, grad = tensors()

    def desc(t):
        return TensorDesc(t.dtype, tuple(t.shape), tuple(t.stride()), tuple(range(t.ndim - 1, -1, -1)), t.device)

    p = EngramGateSavedForward(*(desc(t) for t in (x, kv, w, mask)))
    assert p.check_support()
    with pytest.raises(ValueError, match="backend"):
        EngramGateSavedForward(x, kv, w, mask, backend="cutedsl").check_support()
    with pytest.raises(ValueError, match="kv"):
        EngramGateSavedForward(x, kv[:, ::2], w, mask).check_support()
    with pytest.raises(ValueError, match="eps"):
        EngramGateSavedForward(x, kv, w, mask, eps=float("nan")).check_support()
    saved = torch.empty((1, 4, 4), dtype=torch.float32, device=x.device)
    with pytest.raises(ValueError, match="divisible"):
        cudnn.EngramGateSavedBackward(x[:1], kv[:1], w, saved, grad[:1]).check_support()


@pytest.mark.parametrize("mixed", [False, True])
def test_metadata_implicit_device_forward_and_backward(mixed):
    from cudnn.api_base import TensorDesc

    x, kv, w, mask, grad = tensors(64)

    def desc(tensor, index):
        device = tensor.device if mixed and index % 2 else torch.device("cuda")
        return TensorDesc(tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), tuple(range(tensor.ndim - 1, -1, -1)), device)

    out = torch.empty_like(x)
    saved = torch.empty((64, 4, 4), device=x.device, dtype=torch.float32)
    f = cudnn.EngramGateSavedForward(*(desc(tensor, i) for i, tensor in enumerate((x, kv, w, mask))))
    b = cudnn.EngramGateSavedBackward(*(desc(tensor, i) for i, tensor in enumerate((x, kv, w, saved, grad))))
    f.compile()
    b.compile()
    grads = (torch.empty_like(x), torch.empty_like(kv), torch.empty_like(w))
    workspace = b.allocate_workspace()
    f.execute(x, kv, w, mask, out, saved)
    b.execute(x, kv, w, saved, grad, *grads, workspace)
    leaves = [tensor.detach().clone().requires_grad_() for tensor in (x, kv, w)]
    expected = reference(*leaves, mask)
    close(out, expected, 0.01)
    for actual, target in zip(grads, torch.autograd.grad(expected, leaves, grad), strict=True):
        close(actual, target, 0.03)


def test_output_alias_alignment_workspace_and_compile_contract():
    data = tensors()
    x, kv, w, mask, grad = data
    f, b, out, saved, grads, workspace = plans_and_outputs(data)
    with pytest.raises(ValueError, match="overlap"):
        f.execute(x, kv, w, mask, x, saved)
    with pytest.raises(ValueError, match="overlap"):
        b.execute(x, kv, w, saved, grad, grads[0], kv, grads[2], workspace)
    with pytest.raises(ValueError, match="workspace"):
        b.execute(x, kv, w, saved, grad, *grads, workspace[:-1])
    storage = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)
    misaligned = storage[1:].view(x.shape)
    with pytest.raises(ValueError, match="alignment"):
        f.execute(x, kv, w, mask, misaligned, saved)
    with pytest.raises(RuntimeError, match="compile"):
        cudnn.EngramGateSavedForward(x, kv, w, mask).execute(x, kv, w, mask, out, saved)


def test_old_dsl_declined_at_family_and_plan_boundaries(monkeypatch):
    import cudnn.engram as family
    from cudnn.engram import api
    from cudnn.frost import buffers

    x, kv, w, mask, _ = tensors()
    plan = cudnn.EngramGateSavedForward(x, kv, w, mask)
    state = list(buffers.cutedsl_state())
    state[1] = ("nvidia-cutlass-dsl", "4.6.2")
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: tuple(state))
    monkeypatch.setattr(api, "cutedsl_state", lambda: tuple(state))
    with pytest.raises(RuntimeError, match="4.6.2"):
        family.__getattr__("EngramGateSavedForward")
    with pytest.raises(RuntimeError, match="4.6.2"):
        plan.check_support()
