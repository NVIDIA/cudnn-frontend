# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cake GQA source, host contracts, and native forward/backward checks."""

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cudnn.sdpa.cake import api

pytestmark = pytest.mark.L0


def test_workspace_layout():
    from cudnn._cake._vendor.cake_gqa_nondet_train import cake_jit

    for total in (1, 127, 128, 129, 1025):
        for heads in (2, 6, 16):
            for segments in (1, 3, 19):
                assert api.gqa_attention_workspace_size(total, heads, segments) == 4
                size = api.gqa_attention_workspace_size(total, heads, segments, backward=True)
                assert size == cake_jit.scratch_bytes(total, heads, segments)
                assert size % 256 == 0


@pytest.mark.parametrize("args", [(0, 2, 1), (1, 0, 1), (1, 3, 1), (1, 2, 0), (True, 2, 1), (2**31, 2, 1)])
def test_workspace_rejects_invalid_geometry(args):
    with pytest.raises(ValueError):
        api.gqa_attention_workspace_size(*args, backward=True)


def test_host_rejection_precedes_compilation(monkeypatch):
    monkeypatch.setattr(api, "_plan", lambda *_: pytest.fail("invalid input reached the compiler"))
    q = torch.empty((3, 2, 256), dtype=torch.bfloat16)
    cu = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(ValueError, match="CUDA"):
        api.gqa_attention_forward(q, q, q, cu, cu)


def test_vendor_files_match_record():
    source_root = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "_cake"
    wheel_root = Path(api.__file__).resolve().parents[2] / "_cake"
    record = json.loads((source_root / "VENDORED_GQA.json").read_text())
    for package, info in record["packages"].items():
        for relative, expected in info["files"].items():
            expected = info.get("local_modifications", {}).get(relative, {}).get("blob", expected)
            source_data = (source_root / "_vendor" / package / relative).read_bytes()
            assert hashlib.sha1(f"blob {len(source_data)}\0".encode() + source_data).hexdigest() == expected
            if Path(relative).name in {".gitignore", ".clang-format"}:
                continue
            wheel_data = (wheel_root / "_vendor" / package / relative).read_bytes()
            assert hashlib.sha1(f"blob {len(wheel_data)}\0".encode() + wheel_data).hexdigest() == expected


@pytest.fixture
def gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from cudnn._cake import loader

    try:
        arch = loader.device_arch(torch.cuda.current_device())
        loader.check_toolkit(arch, "GQA tests")
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(str(error))
    return torch.device("cuda", torch.cuda.current_device())


def _inputs(device):
    generator = torch.Generator().manual_seed(19)
    q = torch.randn((21, 4, 256), generator=generator, dtype=torch.bfloat16).to(device)
    k = torch.randn((17, 1, 256), generator=generator, dtype=torch.bfloat16).to(device)
    v = torch.randn((17, 1, 256), generator=generator, dtype=torch.bfloat16).to(device)
    grad = torch.randn(q.shape, generator=generator, dtype=torch.bfloat16).to(device)
    cu_q = torch.tensor([0, 7, 21], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 11, 17], dtype=torch.int32, device=device)
    return q, k, v, grad, cu_q, cu_k


def _reference(q, k, v, grad):
    q, k, v = [value.cpu().double().requires_grad_() for value in (q, k, v)]
    outs, stats = [], []
    for qb, qe, kb, ke in ((0, 7, 0, 11), (7, 21, 11, 17)):
        scores = torch.einsum("thd,shd->hts", q[qb:qe], k[kb:ke].expand(-1, 4, -1)) / 16
        mask = torch.arange(ke - kb)[None, :] <= torch.arange(qe - qb)[:, None] + (ke - kb) - (qe - qb)
        scores = scores.masked_fill(~mask, -torch.inf)
        lse = torch.logsumexp(scores, dim=-1)
        valid = mask.any(-1)[None, :, None]
        probs = scores.masked_fill(~valid, 0).softmax(-1) * valid
        outs.append(torch.einsum("hts,shd->thd", probs, v[kb:ke].expand(-1, 4, -1)))
        stats.append(lse)
    out = torch.cat(outs)
    gradients = torch.autograd.grad(out, (q, k, v), grad.cpu().double())
    return out, torch.cat(stats, dim=1), gradients


def _assert_outputs(result, expected):
    out, lse, grads = expected
    torch.testing.assert_close(result[0].cpu().double(), out.detach(), atol=0.02, rtol=0.02)
    torch.testing.assert_close(result[1].cpu().double(), lse.detach(), atol=2e-5, rtol=2e-5)
    for got, ref in zip(result[2:], grads):
        torch.testing.assert_close(got.cpu().double(), ref, atol=0.03, rtol=0.03)


def _buffers(q, k):
    out = torch.empty_like(q)
    lse = torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(k)
    fws = torch.empty(4, dtype=torch.uint8, device=q.device)
    bws = torch.empty(api.gqa_attention_workspace_size(q.shape[0], q.shape[1], 2, backward=True), dtype=torch.uint8, device=q.device)
    return out, lse, dq, dk, dv, fws, bws


def _run(inputs, buffers, stream=None):
    q, k, v, grad, cu_q, cu_k = inputs
    out, lse, dq, dk, dv, fws, bws = buffers
    forward = api.gqa_attention_forward(q, k, v, cu_q, cu_k, out=out, lse=lse, workspace=fws, stream=stream)
    backward = api.gqa_attention_backward(q, k, v, out, lse, grad, cu_q, cu_k, dq=dq, dk=dk, dv=dv, workspace=bws, stream=stream)
    assert forward["out"] is out and backward["dq"] is dq
    return out, lse, dq, dk, dv


def test_native_reference_and_side_stream(gpu):
    inputs = _inputs(gpu)
    buffers = _buffers(inputs[0], inputs[1])
    expected = _reference(*inputs[:4])
    _assert_outputs(_run(inputs, buffers), expected)
    side = torch.cuda.Stream(device=gpu)
    side.wait_stream(torch.cuda.current_stream(gpu))
    _run(inputs, buffers, side)
    side.synchronize()
    _assert_outputs(buffers[:5], expected)


def test_no_allocation_sync_and_changed_input_capture(gpu):
    inputs = _inputs(gpu)
    buffers = _buffers(inputs[0], inputs[1])
    _run(inputs, buffers)
    torch.cuda.synchronize(gpu)
    allocations = torch.cuda.memory_stats(gpu)["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            _run(inputs, buffers)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_stats(gpu)["allocation.all.allocated"] == allocations
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(inputs, buffers)
    inputs[0].mul_(0.5)
    graph.replay()
    torch.cuda.synchronize(gpu)
    _assert_outputs(buffers[:5], _reference(*inputs[:4]))


def test_rejects_alias_short_workspace_and_deterministic_before_launch(gpu, monkeypatch):
    inputs = _inputs(gpu)
    buffers = _buffers(inputs[0], inputs[1])
    _run(inputs, buffers)
    monkeypatch.setattr(api._Plan, "_launch", lambda *_: pytest.fail("invalid call launched a kernel"))
    q, k, v, grad, cu_q, cu_k = inputs
    out, lse, dq, dk, dv, fws, bws = buffers
    with pytest.raises(ValueError, match="alias"):
        api.gqa_attention_forward(q, k, v, cu_q, cu_k, out=q, lse=lse, workspace=fws)
    with pytest.raises(ValueError, match="workspace"):
        api.gqa_attention_backward(q, k, v, out, lse, grad, cu_q, cu_k, dq=dq, dk=dk, dv=dv, workspace=bws[:1])
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="no deterministic"):
            api.gqa_attention_backward(q, k, v, out, lse, grad, cu_q, cu_k)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("mode", ["bf16", "fp32", "rowmap"])
def test_native_empty_backward_outputs(monkeypatch, mode):
    from cudnn._cake._vendor.cake_gqa_nondet_train import cake_backend as backend

    monkeypatch.setattr(backend, "_check_operands", lambda *args: dict(total_q=0, total_k=1, num_heads_q=2, num_heads_kv=1))
    monkeypatch.setattr(backend, "_launch", lambda *args: pytest.fail("empty backward launched a kernel"))
    q = torch.empty((0, 2, 256), dtype=torch.bfloat16)
    k = torch.ones((1, 1, 256), dtype=torch.bfloat16)
    cu = torch.tensor([0, 0], dtype=torch.int32)
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    dk, dv = (torch.ones_like(k, dtype=dtype) for _ in range(2))
    kwargs = dict(dk=dk, dv=dv, dkv_dtype=dtype)
    if mode == "rowmap":
        kwargs = dict(dk_acc=dk, dv_acc=dv, row_map=torch.tensor([0], dtype=torch.int32))
    dq, actual_dk, actual_dv = backend.cake_gqa_nondet_train_backward(q, k, k, q, torch.empty((2, 0)), q, cu, cu, **kwargs)
    assert dq.shape == q.shape and dq.dtype == q.dtype
    assert actual_dk is dk and actual_dv is dv
    expected = torch.ones_like(dk) if mode == "rowmap" else torch.zeros_like(dk)
    assert torch.equal(dk, expected) and torch.equal(dv, expected)
