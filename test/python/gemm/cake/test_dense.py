# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense projection/router numerics and allocation-free capture contracts."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from cudnn._cake import loader
from cudnn.gemm.cake.dense import DenseGemmPlan

pytestmark = pytest.mark.L0


@pytest.fixture
def device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Cake dense adapter requires SM100")
    try:
        loader.check_toolkit("sm_100a", "test")
    except RuntimeError as error:
        pytest.skip(str(error))
    return torch.device("cuda", torch.cuda.current_device())


def _case(device, operation, dtype, batch=False, fp32_out=False):
    torch.manual_seed(13)
    prefix = (2,) if batch else ()
    x = torch.randn(*prefix, 136, 256, device=device, dtype=dtype) * 0.125
    weight = torch.randn(*prefix, 64, 256, device=device, dtype=dtype) * 0.125
    grad = torch.randn(*prefix, 136, 64, device=device, dtype=dtype) * 0.125
    a, b = {"fwd": (x, weight.transpose(-1, -2)), "dgrad": (grad, weight), "wgrad": (grad.transpose(-1, -2), x)}[operation]
    out = torch.empty((*prefix, a.shape[-2], b.shape[-1]), dtype=torch.float32 if fp32_out else dtype, device=device)
    plan = DenseGemmPlan(a, b, out)
    scratch = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=device)
    return plan, a, b, out, scratch


@pytest.mark.parametrize("operation", ["fwd", "dgrad", "wgrad"])
@pytest.mark.parametrize(
    "dtype,batch,fp32_out", [(torch.bfloat16, False, False), (torch.bfloat16, True, False), (torch.bfloat16, False, True), (torch.float32, False, False)]
)
def test_numerics_and_repeats(device, operation, dtype, batch, fp32_out):
    if operation == "fwd" and fp32_out:
        with pytest.raises(NotImplementedError, match="no generated"):
            _case(device, operation, dtype, batch, fp32_out)
        return
    plan, a, b, out, scratch = _case(device, operation, dtype, batch, fp32_out)
    plan.execute(a, b, out=out, workspace=scratch)
    tol = 0.008 if out.dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(out.double(), a.double() @ b.double(), atol=tol, rtol=tol)
    previous = out.clone()
    scratch.fill_(231)
    plan.execute(a, b, out=out, workspace=scratch)
    assert torch.equal(previous.view(torch.uint8), out.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_capture_rebind_and_no_allocation(device, dtype, monkeypatch):
    plan, a, b, out, scratch = _case(device, "fwd", dtype)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    plan.execute(a, b, out=out, workspace=scratch, stream=stream.cuda_stream)
    stream.synchronize()

    def forbidden(*args, **kwargs):
        raise AssertionError("execute allocated or compiled")

    with monkeypatch.context() as patch:
        for name in ("empty", "zeros", "empty_like", "zeros_like"):
            patch.setattr(torch, name, forbidden)
        patch.setattr(loader.JitSpec, "build", forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            plan.execute(a, b, out=out, workspace=scratch)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        plan.execute(a, b, out=out, workspace=scratch)
    a.mul_(0.5)
    graph.replay()
    tol = 0.008 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(out.double(), a.double() @ b.double(), atol=tol, rtol=tol)
    a2, b2, out2, scratch2 = a.clone(), b.clone(), torch.empty_like(out), torch.empty_like(scratch)
    plan.execute(a2, b2, out=out2, workspace=scratch2)
    assert torch.equal(out.view(torch.uint8), out2.view(torch.uint8))


def test_reject_before_writing(device):
    plan, a, b, out, scratch = _case(device, "fwd", torch.bfloat16)
    out.fill_(7)
    scratch.fill_(93)
    with pytest.raises(ValueError, match="workspace"):
        plan.execute(a, b, out=out, workspace=scratch[:0])
    assert bool(torch.all(out == 7)) and bool(torch.all(scratch == 93))
    with pytest.raises(ValueError, match="shape"):
        plan.execute(a[:-1], b, out=out, workspace=scratch)
    with pytest.raises(ValueError, match="contiguous"):
        DenseGemmPlan(a[:, ::2], b[::2], out)


def test_source_hashes():
    vendor = Path(__file__).resolve().parents[4] / "python/cudnn/_cake/_vendor"
    record = json.loads((vendor / "VENDORED_DENSE_GEMM.json").read_text())
    for name, expected in record["files"].items():
        assert hashlib.sha256((vendor / "cake_dense_gemm_det_train" / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_public_paired_api(device, dtype):
    from cudnn.ops import dense_gemm_forward, dense_gemm_backward

    _, x, wt, _, _ = _case(device, "fwd", dtype)
    weight = wt.transpose(-1, -2)
    out = dense_gemm_forward(x, weight)["out"]
    grad = torch.randn_like(out) * 0.125
    result = dense_gemm_backward(grad, x, weight)
    tol = 0.008 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(out.double(), x.double() @ wt.double(), atol=tol, rtol=tol)
    torch.testing.assert_close(result["dx"].double(), grad.double() @ weight.double(), atol=tol, rtol=tol)
    torch.testing.assert_close(result["dw"].double(), grad.double().T @ x.double(), atol=tol, rtol=tol)
