# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from cudnn.deepseek_sparse_attention.cake import sparse as api

pytestmark = pytest.mark.L0


def test_vendor_files():
    root = Path(api.__file__).resolve().parents[2] / "_cake"
    for package, record in json.loads((root / "VENDORED_DSA_SPARSE.json").read_text())["packages"].items():
        for name, expected in record["files"].items():
            data = (root / "_vendor" / package / name).read_bytes()
            assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


@pytest.mark.parametrize("rope", [False, True])
def test_workspace_matches_native(rope):
    b = api._backend()
    layout = b.workspace_layout if rope else b.workspace_layout_nope
    for backward in (False, True):
        assert (
            api.dsa_sparse_workspace_size(3, 17, 8, rope=rope, backward=backward)
            == layout(3, 17, 8, backward=backward, key_passes=1, dkv_direct=False)["total"] + 256
        )
    with pytest.raises(ValueError):
        api.dsa_sparse_workspace_size(0, 17, 8)


def test_cpu_rejected_before_native_load(monkeypatch):
    monkeypatch.setattr(api, "_backend", lambda: pytest.fail("CPU input reached native loader"))
    with pytest.raises(ValueError, match="CUDA"):
        api.dsa_sparse_forward(torch.empty((3, 64, 512)), None, None)


@pytest.fixture
def gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from cudnn._cake import loader

    try:
        loader.check_toolkit(loader.device_arch(), "DSA tests")
    except (RuntimeError, NotImplementedError) as e:
        pytest.skip(str(e))
    return torch.device("cuda", torch.cuda.current_device())


def _data(device, rope):
    gen = torch.Generator().manual_seed(142)

    def rand(shape):
        return (torch.randn(shape, generator=gen) * 0.5).to(device=device, dtype=torch.bfloat16)

    q = rand((3, 64, 512))
    kv = rand((17, 512))
    idx = torch.tensor([[0, 2, 5, 7, 11, -1, -1, -1], [1, 1, 4, 9, 16, 19, -1, -1], [-1] * 8], dtype=torch.int32, device=device)
    qr, kr = (rand((3, 64, 64)), rand((17, 64))) if rope else (None, None)
    return q, kv, idx, qr, kr, rand(q.shape)


def _reference(data, scale):
    q, kv, indices, qr, kr, do = data
    xs = [x.cpu().double().requires_grad_() for x in (q, kv) + ((qr, kr) if qr is not None else ())]
    q, kv = xs[:2]
    ids = indices.cpu().long()
    valid = (ids >= 0) & (ids < kv.shape[0])
    selected = kv[ids.clamp(0, kv.shape[0] - 1)]
    scores = torch.einsum("thd,tkd->thk", q, selected)
    if qr is not None:
        qr, kr = xs[2:]
        scores = scores + torch.einsum("thd,tkd->thk", qr, kr[ids.clamp(0, kv.shape[0] - 1)])
    scores = (scores * scale).masked_fill(~valid[:, None, :], -torch.inf)
    live = valid.any(-1)[:, None, None]
    p = scores.masked_fill(~live, 0).softmax(-1) * live
    out = torch.einsum("thk,tkd->thd", p, selected)
    grads = torch.autograd.grad(out, xs, do.cpu().double())
    return out.detach(), scores.logsumexp(-1).detach(), grads


@pytest.mark.parametrize("rope", [False, True])
def test_native_gradients_stream_capture_and_no_allocation(gpu, rope):
    data = _data(gpu, rope)
    q, kv, indices, qr, kr, dout = data
    scale = 576**-0.5 if rope else 1 / 16
    options = dict(q_rope=qr, k_rope=kr, softmax_scale=scale)
    out, lse, residual = api.dsa_sparse_forward(q, kv, indices, **options)
    grads = api.dsa_sparse_backward(q, kv, indices, out, lse, residual, dout, **options)
    workspace = torch.empty(api.dsa_sparse_workspace_size(3, 17, 8, rope=rope, backward=True), dtype=torch.uint8, device=gpu)

    def run(stream=None):
        api.dsa_sparse_forward(q, kv, indices, out=out, lse=lse, residual=residual, workspace=workspace, stream=stream, **options)
        api.dsa_sparse_backward(q, kv, indices, out, lse, residual, dout, workspace=workspace, stream=stream, **options, **grads)

    def check():
        expected_out, expected_lse, expected_grads = _reference(data, scale)
        torch.testing.assert_close(out.cpu().double(), expected_out, atol=8e-3, rtol=8e-3)
        torch.testing.assert_close(lse.cpu().double(), expected_lse, atol=2e-5, rtol=2e-5)
        for got, expected in zip(grads.values(), expected_grads):
            torch.testing.assert_close(got.cpu().double(), expected, atol=2e-2, rtol=2e-2)

    check()
    side = torch.cuda.Stream(device=gpu)
    side.wait_stream(torch.cuda.current_stream(gpu))
    run(side)
    torch.cuda.current_stream(gpu).wait_stream(side)
    check()
    torch.cuda.synchronize(gpu)
    allocations = torch.cuda.memory_stats(gpu)["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_stats(gpu)["allocation.all.allocated"] == allocations
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.neg_()
    graph.replay()
    torch.cuda.synchronize(gpu)
    check()
    with pytest.raises(ValueError, match="alias"):
        api.dsa_sparse_forward(q, kv, indices, out=q, **options)
    with pytest.raises(ValueError, match="workspace"):
        api.dsa_sparse_forward(q, kv, indices, workspace=workspace[:1], **options)
    old = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="deterministic"):
            api.dsa_sparse_backward(q, kv, indices, out, lse, residual, dout, **options)
    finally:
        torch.use_deterministic_algorithms(old, warn_only=old_warn)
