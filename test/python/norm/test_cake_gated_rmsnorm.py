# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Cake gated RMSNorm contracts and independent gradient checks."""

import hashlib
import json
from pathlib import Path

import pytest

from cudnn.norm.cake.gated_rmsnorm import GatedRMSNormPlan, _geometry

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("shape", [(), (128,), (0, 128), (1, 64), (1, 1, 1, 128), (2**31, 128), (True, 128)])
def test_reject_geometry(shape):
    with pytest.raises(ValueError):
        _geometry(shape)


def test_vendor_files_match_pin():
    root = Path(__file__).resolve().parents[3] / "python/cudnn/_cake/_vendor"
    record = json.loads((root / "VENDORED_GATED_RMSNORM.json").read_text())
    package = root / "cake_rmsnorm_silu_gate_det_train"
    for name, digest in record["files"].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest, name


def test_invalid_bindings_fail_before_launch(monkeypatch):
    torch = pytest.importorskip("torch")
    plan = object.__new__(GatedRMSNormPlan)
    plan.shape, plan.tokens, plan.heads = (2, 128), 2, 1
    plan.rows, plan.eps, plan.chunks, plan.blocks_per_cta = 2, 1e-6, 1, 1
    plan.device = torch.device("cpu")
    monkeypatch.setattr(plan, "_run", lambda *args: pytest.fail("invalid bindings reached a launch"))
    x, z, dy, dx, dz = (torch.empty(plan.shape, dtype=torch.bfloat16) for _ in range(5))
    w = torch.empty(128, dtype=torch.bfloat16)
    dw = torch.empty(128, dtype=torch.float32)
    scratch = torch.empty(128, dtype=torch.float32)
    with pytest.raises(ValueError, match="overlap|contiguous"):
        plan.forward(x, z, w, x)
    with pytest.raises(ValueError, match="out must have shape"):
        plan.forward(x, z, w, dx[:1])
    with pytest.raises(ValueError, match="workspace needs"):
        plan.backward(dy, x, z, w, dx, dz, dw, scratch[:0])
    with pytest.raises(ValueError, match="workspace must have shape"):
        plan.backward(dy, x, z, w, dx, dz, dw, scratch.double())
    with pytest.raises(ValueError, match="overlap"):
        plan.backward(dy, x, z, w, dx, dx, dw, scratch)


@pytest.fixture
def torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake requires SM100a/SM103a/SM107a")
    from cudnn._cake import loader

    try:
        loader.check_toolkit(loader.device_arch(), "gated RMSNorm test")
    except RuntimeError as error:
        pytest.skip(str(error))
    return torch


@pytest.mark.parametrize("shape", [(3, 128), (129, 2, 128)])
def test_forward_backward_and_replay(torch_cuda, shape):
    torch = torch_cuda
    torch.manual_seed(7)
    device = torch.device("cuda", torch.cuda.current_device())
    x, z, dy = (torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(3))
    if len(shape) == 3:
        x, z, dy = (torch.randn((*shape[:-1], 256), device=device, dtype=torch.bfloat16)[..., :128] for _ in range(3))
    w = torch.randn(128, device=device, dtype=torch.bfloat16)
    plan = GatedRMSNormPlan(shape, device=device)
    y, dx, dz = (torch.empty_like(x) for _ in range(3))
    dw = torch.empty(128, dtype=torch.float32, device=device)
    scratch = torch.empty(plan.get_workspace_size() // 4, dtype=torch.float32, device=device)
    xr, zr, wr = (tensor.double().requires_grad_() for tensor in (x, z, w))
    ref = xr * torch.rsqrt(xr.square().mean(-1, keepdim=True) + 1e-6) * wr * torch.nn.functional.silu(zr)
    refs = torch.autograd.grad(ref, (xr, zr, wr), dy.double())

    def run():
        plan.forward(x, z, w, y)
        plan.backward(dy, x, z, w, dx, dz, dw, scratch)

    run()
    for actual, expected in zip((y, dx, dz, dw), (ref, *refs)):
        torch.testing.assert_close(actual.double(), expected, atol=0.025, rtol=0.015)
    expected = tuple(tensor.clone() for tensor in (y, dx, dz, dw))
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        run()
        run()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for actual, saved in zip((y, dx, dz, dw), expected):
        assert torch.equal(actual, saved)

    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        run()
    x.neg_()
    graph.replay()
    torch.cuda.current_stream(device).wait_stream(side)
    captured = y.clone()
    run()
    assert torch.equal(y, captured)

    old = (dx.clone(), dz.clone(), dw.clone())
    with pytest.raises(ValueError, match="workspace needs"):
        plan.backward(dy, x, z, w, dx, dz, dw, scratch[:0])
    for actual, saved in zip((dx, dz, dw), old):
        assert torch.equal(actual, saved)
    with pytest.raises(ValueError, match="overlap|contiguous"):
        plan.forward(x, z, w, x)


def test_public_wrappers_on_side_stream(torch_cuda):
    torch = torch_cuda
    from cudnn.ops import gated_rmsnorm_forward, gated_rmsnorm_backward

    x, z, dy = (torch.randn((4, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3))
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    result = gated_rmsnorm_forward(x, z, w, stream=side)
    grads = gated_rmsnorm_backward(dy, x, z, w, stream=side)
    torch.cuda.current_stream().wait_stream(side)
    xr, zr, wr = (tensor.double().requires_grad_() for tensor in (x, z, w))
    ref = xr * torch.rsqrt(xr.square().mean(-1, keepdim=True) + 1e-6) * wr * torch.nn.functional.silu(zr)
    refs = torch.autograd.grad(ref, (xr, zr, wr), dy.double())
    for actual, expected in zip((result["output"], grads["grad_x"], grads["grad_z"], grads["grad_weight"]), (ref, *refs)):
        torch.testing.assert_close(actual.double(), expected, atol=0.025, rtol=0.015)
