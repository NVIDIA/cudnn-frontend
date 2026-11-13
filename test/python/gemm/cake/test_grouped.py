# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ragged expert GEMM numerics, scratch ownership, and metadata replay."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from cudnn._cake import loader
from cudnn.gemm.cake.grouped import RaggedGroupedGemmPlan

pytestmark = pytest.mark.L0


@pytest.fixture
def device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Cake grouped GEMM adapter requires SM100")
    try:
        loader.check_toolkit("sm_100a", "test")
    except RuntimeError as error:
        pytest.skip(str(error))
    return torch.device("cuda", torch.cuda.current_device())


def _case(device, operation, m=129, n=256, k=256, dtype=torch.bfloat16):
    torch.manual_seed(7)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.125
    w = torch.randn(3, n, k, device=device, dtype=torch.bfloat16) * 0.125
    g = torch.randn(m, n, device=device, dtype=torch.bfloat16) * 0.125
    boundaries = [m // 3, m // 3, m]
    offsets = torch.tensor(boundaries, dtype=torch.int32, device=device)
    a, b = {"fwd": (x, w), "dgrad": (g, w), "wgrad": (g, x)}[operation]
    plan = RaggedGroupedGemmPlan(operation, a, b, offsets, output_dtype=dtype)
    out = torch.empty(plan.output_shape, device=device, dtype=dtype)
    scratch = torch.empty(plan.get_workspace_size(), device=device, dtype=torch.uint8)
    return plan, a, b, offsets, out, scratch, boundaries


def _reference(operation, a, b, boundaries):
    parts = []
    start = 0
    for expert, end in enumerate(boundaries):
        if operation == "fwd":
            parts.append(a[start:end].double() @ b[expert].double().T)
        elif operation == "dgrad":
            parts.append(a[start:end].double() @ b[expert].double())
        else:
            parts.append(a[start:end].double().T @ b[start:end].double())
        start = end
    return torch.stack(parts) if operation == "wgrad" else torch.cat(parts)


@pytest.mark.parametrize("operation,dtype", [("fwd", torch.bfloat16), ("dgrad", torch.bfloat16), ("wgrad", torch.bfloat16), ("wgrad", torch.float32)])
@pytest.mark.parametrize("m,n,k", [(129, 256, 256), (1025, 512, 512)])
def test_reference_and_repeatability(device, operation, dtype, m, n, k):
    plan, a, b, offsets, out, scratch, boundaries = _case(device, operation, m, n, k, dtype)
    plan.execute(a, b, offsets, out=out, workspace=scratch)
    reference = _reference(operation, a, b, boundaries)
    tol = 0.008 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(out.double(), reference, atol=tol, rtol=tol)
    previous = out.clone()
    scratch.fill_(137)
    plan.execute(a, b, offsets, out=out, workspace=scratch)
    assert torch.equal(previous.view(torch.uint8), out.view(torch.uint8))


@pytest.mark.parametrize("operation", ["fwd", "dgrad", "wgrad"])
def test_capture_rebind_and_no_allocation(device, operation, monkeypatch):
    plan, a, b, offsets, out, scratch, _ = _case(device, operation)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    plan.execute(a, b, offsets, out=out, workspace=scratch, stream=stream.cuda_stream)
    stream.synchronize()

    def forbidden(*args, **kwargs):
        raise AssertionError("execute allocated or compiled")

    with monkeypatch.context() as patch:
        for name in ("empty", "zeros", "empty_like", "zeros_like"):
            patch.setattr(torch, name, forbidden)
        patch.setattr(loader.JitSpec, "build", forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            plan.execute(a, b, offsets, out=out, workspace=scratch)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        plan.execute(a, b, offsets, out=out, workspace=scratch)
    boundaries = [7, 90, a.shape[0]]
    offsets.copy_(torch.tensor(boundaries, dtype=torch.int32, device=device))
    a.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(out.double(), _reference(operation, a, b, boundaries), atol=0.008, rtol=0.008)
    a2, b2, offsets2 = a.clone(), b.clone(), offsets.clone()
    out2, scratch2 = torch.empty_like(out), torch.empty_like(scratch)
    plan.execute(a2, b2, offsets2, out=out2, workspace=scratch2)
    assert torch.equal(out.view(torch.uint8), out2.view(torch.uint8))


def test_reject_before_writing(device):
    plan, a, b, offsets, out, scratch, _ = _case(device, "fwd")
    out.fill_(7)
    scratch.fill_(93)
    with pytest.raises(ValueError, match="workspace"):
        plan.execute(a, b, offsets, out=out, workspace=scratch[:0])
    assert bool(torch.all(out == 7)) and bool(torch.all(scratch == 93))
    with pytest.raises(ValueError, match="alias"):
        plan.execute(a, b, offsets, out=a, workspace=scratch)
    with pytest.raises(ValueError, match="shape"):
        plan.execute(a[:-1], b, offsets, out=out, workspace=scratch)


def test_sources_match_pinned_record():
    root = Path(__file__).resolve().parents[4]
    vendor = root / "python/cudnn/_cake/_vendor"
    record = json.loads((vendor / "VENDORED_MOE_GROUPED_GEMM.json").read_text())
    for name, expected in record["files"].items():
        assert hashlib.sha256((vendor / "cake_moe_grouped_gemm_det_train" / name).read_bytes()).hexdigest() == expected


def test_public_paired_api_and_preflight(device, monkeypatch):
    from cudnn.ops import ragged_grouped_gemm_forward, ragged_grouped_gemm_backward

    _, x, weight, offsets, _, _, boundaries = _case(device, "fwd")
    out = ragged_grouped_gemm_forward(x, weight, offsets)["out"]
    grad = torch.randn_like(out) * 0.125
    result = ragged_grouped_gemm_backward(grad, x, weight, offsets, weight_gradient_dtype=torch.float32)
    torch.testing.assert_close(out.double(), _reference("fwd", x, weight, boundaries), atol=0.008, rtol=0.008)
    torch.testing.assert_close(result["dx"].double(), _reference("dgrad", grad, weight, boundaries), atol=0.008, rtol=0.008)
    torch.testing.assert_close(result["dw"].double(), _reference("wgrad", grad, x, boundaries), atol=2e-5, rtol=2e-5)

    def forbidden(*args, **kwargs):
        raise AssertionError("backward launched before validating both gradients")

    monkeypatch.setattr(RaggedGroupedGemmPlan, "execute", forbidden)
    with pytest.raises(ValueError, match="forward dimensions"):
        ragged_grouped_gemm_backward(grad, x[:, :128], weight, offsets)
    with pytest.raises(ValueError, match="only wgrad"):
        ragged_grouped_gemm_backward(grad, x, weight, offsets, weight_gradient_dtype=torch.float64)
