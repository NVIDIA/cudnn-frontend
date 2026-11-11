# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cake causal convolution: independent math, gradients, and prepared execution."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cudnn.causal_conv1d_cake.api import CausalConv1dPlan, _geometry

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("dims", [(0, 8, 1), (1, 7, 1), (1, 8, 2), (True, 8, 1), (2**31, 8, 1)])
def test_invalid_geometry(dims):
    with pytest.raises(ValueError):
        _geometry(*dims)


def test_vendor_hashes():
    root = Path(__file__).resolve().parents[3] / "python/cudnn/_cake/_vendor"
    record = json.loads((root / "VENDORED_CAUSAL_CONV1D.json").read_text())
    for name, digest in record["files"].items():
        assert hashlib.sha256((root / "cake_causal_conv1d_det_train" / name).read_bytes()).hexdigest() == digest, name


@pytest.fixture
def torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake requires SM100a/SM103a/SM107a")
    from cudnn._cake import loader

    try:
        loader.check_toolkit(loader.device_arch(), "causal conv test")
    except RuntimeError as error:
        pytest.skip(str(error))
    return torch


def reference(torch, x, weight, bias, lengths):
    rows = []
    offset = 0
    for length in lengths:
        for token in range(length):
            terms = [x[offset + token - 3 + tap] * weight[:, tap] for tap in range(4) if token - 3 + tap >= 0]
            pre = sum(terms)
            if bias is not None:
                pre = pre + bias
            rows.append(torch.nn.functional.silu(pre))
        offset += length
    return torch.stack(rows)


@pytest.mark.parametrize("dtype,bias,channels", [("float32", False, 8), ("float32", False, 256), ("float32", True, 256), ("bfloat16", True, 256)])
def test_training_and_capture(torch_cuda, dtype, bias, channels):
    torch = torch_cuda
    torch.manual_seed(14)
    lengths = [1, 3, 29]
    x, dy = (torch.randn((sum(lengths), channels), dtype=torch.bfloat16, device="cuda") for _ in range(2))
    w = torch.randn((channels, 4), dtype=getattr(torch, dtype), device="cuda") * 0.2
    b = torch.randn(channels, dtype=getattr(torch, dtype), device="cuda") * 0.1 if bias else None
    cu = torch.tensor([0, 1, 4, sum(lengths)], dtype=torch.int32, device="cuda")
    plan = CausalConv1dPlan(*x.shape, len(lengths), weight_dtype=w.dtype, bias_dtype=None if b is None else b.dtype)
    y, dx = torch.empty_like(x), torch.empty_like(x)
    dw, db = torch.empty_like(w), None if b is None else torch.empty_like(b)
    scratch = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    xr, wr = (t.double().requires_grad_() for t in (x, w))
    br = None if b is None else b.double().requires_grad_()
    ref = reference(torch, xr, wr, br, lengths)
    refs = torch.autograd.grad(ref, (xr, wr) if br is None else (xr, wr, br), dy.double())

    def run():
        plan.forward(x, w, cu, y, scratch, bias=b)
        plan.backward(dy, x, w, cu, dx, dw, scratch, bias=b, grad_bias=db)

    run()
    actual = (y, dx, dw) if db is None else (y, dx, dw, db)
    for result, expected in zip(actual, (ref, *refs)):
        torch.testing.assert_close(
            result.double(), expected, atol=3e-5 if result.dtype == torch.float32 else 0.025, rtol=1e-4 if result.dtype == torch.float32 else 0.015
        )
    saved = tuple(t.clone() for t in actual)
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        run()
        run()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for result, expected in zip(actual, saved):
        assert torch.equal(result, expected)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        run()
    x.neg_()
    graph.replay()
    torch.cuda.current_stream().wait_stream(side)
    saved = tuple(t.clone() for t in actual)
    run()
    for result, expected in zip(actual, saved):
        assert torch.equal(result, expected)
    with pytest.raises(ValueError, match="overlap"):
        plan.forward(x, w, cu, x, scratch, bias=b)
    old = dx.clone()
    with pytest.raises(ValueError, match="workspace needs"):
        plan.backward(dy, x, w, cu, dx, dw, scratch[:0], bias=b, grad_bias=db)
    assert torch.equal(dx, old)


@pytest.mark.parametrize("packed", [False, True])
def test_existing_public_api_autograd(torch_cuda, packed):
    torch = torch_cuda
    from cudnn import ops
    from cudnn.ops.causal_conv1d import causal_conv1d

    assert ops.causal_conv1d is causal_conv1d

    x = torch.randn((1 if packed else 2, 7, 256), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = (torch.randn((256, 4), dtype=torch.float32, device="cuda") * 0.1).requires_grad_()
    b = torch.zeros(256, dtype=torch.float32, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    cu = torch.tensor([0, 2, 7], dtype=torch.int32, device="cuda") if packed else None
    y = causal_conv1d(x.transpose(1, 2), w, b, "silu", cu_seqlens=cu, backend="cake")
    grads = torch.autograd.grad(y, (x, w, b), dy.transpose(1, 2))
    xr, wr, br = (t.detach().double().requires_grad_() for t in (x, w, b))
    ref = reference(torch, xr.view(-1, 256), wr, br, [2, 5] if packed else [7, 7]).view(x.shape)
    refs = torch.autograd.grad(ref, (xr, wr, br), dy.double())
    for actual, expected in zip((y.transpose(1, 2), *grads), (ref, *refs)):
        torch.testing.assert_close(actual.double(), expected, atol=0.025, rtol=0.015)


@pytest.mark.parametrize("prefix", [[1, 4], [0, 5], [0, 2, 1, 4], [0, 0, 4]])
def test_device_prefix_guard(torch_cuda, prefix):
    code = """
import os
import torch
import tvm_ffi
from cudnn._cake import loader
from cudnn.causal_conv1d_cake.api import _validator
arch = loader.device_arch()
validate = _validator(arch, loader.find_nvcc(), loader.check_toolkit(arch, "test"))
cu = torch.tensor(PREFIX, dtype=torch.int32, device="cuda")
try:
    with tvm_ffi.use_torch_stream():
        validate(cu, 4)
    torch.cuda.synchronize()
except RuntimeError as error:
    if "illegal instruction" in str(error) or "device-side assert" in str(error):
        os._exit(0)
    raise
os._exit(3)
""".replace("PREFIX", repr(prefix))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr


def test_public_rejections(torch_cuda):
    torch = torch_cuda
    from cudnn import ops
    from cudnn.ops.causal_conv1d import causal_conv1d

    assert ops.causal_conv1d is causal_conv1d

    x = torch.randn((1, 7, 256), dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    w = torch.randn((256, 4), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="backend must"):
        causal_conv1d(x, w, activation="silu", backend="invalid")
    with pytest.raises(NotImplementedError, match="supports SiLU"):
        causal_conv1d(x, w, backend="cake")
    with pytest.raises(ValueError, match="channel-last"):
        causal_conv1d(x.contiguous(), w, activation="silu", backend="cake")
