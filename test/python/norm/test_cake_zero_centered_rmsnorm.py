# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent zero-centered RMSNorm math, buffer contracts, and capture."""

import hashlib
import json
from pathlib import Path

import pytest

from cudnn.norm.cake.zero_centered_rmsnorm import ZeroCenteredRMSNormPlan, _geometry

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("shape", [(), (256,), (0, 256), (True, 256), (2**30, 256)])
def test_invalid_geometry(shape):
    with pytest.raises(ValueError):
        _geometry(shape)


def test_vendor_hashes():
    root = Path(__file__).resolve().parents[3] / "python/cudnn/_cake/_vendor"
    record = json.loads((root / "VENDORED_ZERO_RMSNORM.json").read_text())
    for name, digest in record["files"].items():
        assert hashlib.sha256((root / "cake_rmsnorm_det_train" / name).read_bytes()).hexdigest() == digest, name


@pytest.fixture
def torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake requires SM100a/SM103a/SM107a")
    from cudnn._cake import loader

    try:
        loader.check_toolkit(loader.device_arch(), "zero-centered RMSNorm test")
    except RuntimeError as error:
        pytest.skip(str(error))
    return torch


@pytest.mark.parametrize("shape", [(3, 256), (129, 1, 4096)])
def test_training_capture_and_contracts(torch_cuda, shape):
    torch = torch_cuda
    if torch.cuda.get_device_capability() == (10, 7) and shape[-1] == 4096:
        pytest.skip("SM107a H4096 needs oversized shared memory")
    torch.manual_seed(11)
    x, dy = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(2))
    w = (torch.randn(shape[-1], dtype=torch.float32, device="cuda") * 0.1).requires_grad_()
    plan = ZeroCenteredRMSNormPlan(shape)
    y, dx = (torch.empty_like(x) for _ in range(2))
    rstd = torch.empty(plan.rows, dtype=torch.float32, device="cuda")
    dw = torch.empty_like(w)
    workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")

    def run():
        plan.forward(x, w, y, rstd, workspace)
        plan.backward(dy, x, w, rstd, dx, dw, workspace)

    xr, wr = (tensor.detach().double().requires_grad_() for tensor in (x, w))
    ref = xr * torch.rsqrt(xr.square().mean(-1, keepdim=True) + 1e-6) * (1 + wr)
    refs = torch.autograd.grad(ref, (xr, wr), dy.double())
    run()
    for actual, expected in zip((y, dx, dw), (ref, *refs)):
        torch.testing.assert_close(actual.double(), expected, atol=0.02, rtol=0.012)
    expected = tuple(t.clone() for t in (y, dx, dw))
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        run()
        run()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for actual, saved in zip((y, dx, dw), expected):
        assert torch.equal(actual, saved)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        run()
    x.neg_()
    graph.replay()
    torch.cuda.current_stream().wait_stream(side)
    replay = tuple(t.clone() for t in (y, dx, dw))
    run()
    for actual, captured in zip((y, dx, dw), replay):
        assert torch.equal(actual, captured)
    with pytest.raises(ValueError, match="overlap"):
        plan.forward(x, w, x, rstd, workspace)
    old = dx.clone()
    with pytest.raises(ValueError, match="workspace needs"):
        plan.backward(dy, x, w, rstd, dx, dw, workspace[:0])
    assert torch.equal(old, dx)
    with pytest.raises(ValueError, match="weight must"):
        plan.forward(x, w.bfloat16(), y, rstd, workspace)


def test_public_wrappers_side_stream(torch_cuda):
    torch = torch_cuda
    from cudnn.ops import zero_centered_rmsnorm_forward, zero_centered_rmsnorm_backward

    x, dy = (torch.randn((2, 256), dtype=torch.bfloat16, device="cuda") for _ in range(2))
    w = torch.zeros(256, dtype=torch.float32, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    result = zero_centered_rmsnorm_forward(x, w, stream=side)
    grads = zero_centered_rmsnorm_backward(dy, x, w, result["rstd"], stream=side)
    torch.cuda.current_stream().wait_stream(side)
    xr, wr = (tensor.double().requires_grad_() for tensor in (x, w))
    ref = xr * torch.rsqrt(xr.square().mean(-1, keepdim=True) + 1e-6) * (1 + wr)
    refs = torch.autograd.grad(ref, (xr, wr), dy.double())
    for actual, expected in zip((result["output"], grads["grad_x"], grads["grad_weight"]), (ref, *refs)):
        torch.testing.assert_close(actual.double(), expected, atol=0.02, rtol=0.012)
