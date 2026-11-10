# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from cudnn.deepseek_sparse_attention.cake import deterministic as api

pytestmark = pytest.mark.L0


def test_vendor_files():
    root = Path(api.__file__).resolve().parents[2] / "_cake"
    for package, record in json.loads((root / "VENDORED_DSA_DET.json").read_text())["packages"].items():
        for name, expected in record["files"].items():
            data = (root / "_vendor" / package / name).read_bytes()
            assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


def test_segments_stable_and_exclude_holes():
    ids = torch.tensor([[2, -1, 1, 2], [3, 9, 1, -3]], dtype=torch.int32)
    order, starts, counts = api.dsa_prepare_segments(ids, 4)
    assert starts.tolist() == [3, 3, 5, 7] and counts.tolist() == [0, 2, 2, 1]
    assert order.tolist() == [1, 5, 7, 2, 6, 0, 3, 4]


def test_cpu_rejected(monkeypatch):
    monkeypatch.setattr(api, "_entry", lambda *_: pytest.fail("CPU reached compiler"))
    with pytest.raises(ValueError, match="CUDA"):
        api.dsa_det_forward(torch.empty((2, 8, 512)), None, None)


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


def _rand(shape, device):
    return (torch.randn(shape, generator=torch.Generator().manual_seed(45)) * 0.5).to(device=device, dtype=torch.bfloat16)


def _indices(device):
    idx = torch.full((3, 64), -1, dtype=torch.int32, device=device)
    idx[0, :5] = torch.tensor([0, 2, 2, 5, 7], device=device)
    idx[1, :4] = torch.tensor([1, 3, 8, 9], device=device)
    return idx


def _attention_ref(q, kv, idx, do):
    q = q.cpu().double().requires_grad_()
    kv = kv.cpu().double().requires_grad_()
    idx = idx.cpu().long()
    valid = (idx >= 0) & (idx < kv.shape[0])
    selected = kv[idx.clamp(0, kv.shape[0] - 1)]
    scores = (torch.einsum("thd,tkd->thk", q, selected) / q.shape[-1] ** 0.5).masked_fill(~valid[:, None, :], -torch.inf)
    live = valid.any(-1)[:, None, None]
    p = scores.masked_fill(~live, 0).softmax(-1) * live
    out = torch.einsum("thk,tkd->thd", p, selected[:, :, :512])
    grads = torch.autograd.grad(out, (q, kv), do.cpu().double())
    return out.detach(), scores.logsumexp(-1).detach() / torch.log(torch.tensor(2.0)).item(), grads


def _capture(run, change, check, device):
    run()
    check()
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    change()
    graph.replay()
    torch.cuda.synchronize(device)
    check()


@pytest.mark.parametrize("heads,dim", [(8, 512), (32, 576), (64, 512)])
def test_attention_gradients_and_capture(gpu, heads, dim):
    q = _rand((3, heads, dim), gpu)
    kv = _rand((11, dim), gpu)
    idx = _indices(gpu)
    do = _rand((3, heads, 512), gpu)
    groups = api.dsa_prepare_segments(idx, 11)
    out, lse = api.dsa_det_forward(q, kv, idx)
    grads = api.dsa_det_backward(q, kv, idx, out, lse, do, **groups)
    scratch = torch.empty(max(256, api.dsa_det_workspace_size(3, heads, dim, 11, 64)), dtype=torch.uint8, device=gpu)

    def run():
        api.dsa_det_forward(q, kv, idx, out=out, lse=lse, workspace=scratch)
        api.dsa_det_backward(q, kv, idx, out, lse, do, **groups, **grads, workspace=scratch)

    def check():
        ref, rlse, rg = _attention_ref(q, kv, idx, do)
        torch.testing.assert_close(out.cpu().double(), ref, atol=8e-3, rtol=8e-3)
        torch.testing.assert_close(lse[:2].cpu().double(), rlse[:2], atol=2e-5, rtol=2e-5)
        assert bool((lse[2] < -1e6).all())
        for got, want in zip(grads.values(), rg):
            torch.testing.assert_close(got.cpu().double(), want, atol=2e-2, rtol=2e-2)

    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        _capture(run, lambda: q.neg_(), check, gpu)
        first = [v.clone() for v in grads.values()]
        run()
        assert all(torch.equal(a, b) for a, b in zip(first, grads.values()))
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)


@pytest.mark.parametrize("heads", [8, 32, 64])
def test_indexer_logits_gradients_capture(gpu, heads):
    q = _rand((3, heads, 128), gpu)
    k = _rand((11, 128), gpu)
    weights = _rand((3, heads), gpu).float()
    idx = _indices(gpu)
    starts = torch.tensor([0, 2, 0], dtype=torch.int32, device=gpu)
    ends = torch.tensor([11, 8, 0], dtype=torch.int32, device=gpu)
    ds = _rand(idx.shape, gpu).float()
    groups = api.dsa_prepare_segments(idx, 11)
    logits = api.dsa_indexer_logits(q, k, weights, starts, ends)["logits"]
    grads = api.dsa_indexer_backward(q, k, weights, idx, ds, **groups)
    scratch = torch.empty(api.dsa_det_workspace_size(3, heads, 128, 11, 64, indexer=True), dtype=torch.uint8, device=gpu)

    def run():
        api.dsa_indexer_logits(q, k, weights, starts, ends, logits=logits)
        api.dsa_indexer_backward(q, k, weights, idx, ds, **groups, **grads, workspace=scratch)

    def check():
        qc = q.cpu().double().requires_grad_()
        kc = k.cpu().double().requires_grad_()
        wc = weights.cpu().double().requires_grad_()
        ids = idx.cpu().long()
        raw = torch.einsum("thd,kd->thk", qc, kc)
        scores = (raw.relu() * wc[:, :, None]).sum(1)
        mask = (torch.arange(11)[None, :] >= starts.cpu()[:, None]) & (torch.arange(11)[None, :] < ends.cpu()[:, None])
        torch.testing.assert_close(logits.cpu().double(), scores.detach().masked_fill(~mask, -torch.inf), atol=2e-4, rtol=2e-4)
        selected = scores.gather(1, ids.clamp(0, 10)) * (ids >= 0)
        expected = torch.autograd.grad(selected, (qc, kc, wc), ds.cpu().double())
        for got, want in zip(grads.values(), expected):
            torch.testing.assert_close(got.cpu().double(), want, atol=2e-2, rtol=2e-2)

    _capture(run, lambda: q.neg_(), check, gpu)
    first = [v.clone() for v in grads.values()]
    run()
    assert all(torch.equal(a, b) for a, b in zip(first, grads.values()))
    with pytest.raises(ValueError, match="alias"):
        api.dsa_indexer_backward(q, k, weights, idx, ds, **groups, dq=q)
    with pytest.raises(ValueError, match="workspace"):
        api.dsa_indexer_backward(q, k, weights, idx, ds, **groups, workspace=scratch[:1])
