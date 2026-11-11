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
            expected = record.get("local_modifications", {}).get(name, {}).get("blob", expected)
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
    idx = torch.full((3, 128), -1, dtype=torch.int32, device=device)
    idx[0, :5] = torch.tensor([0, 2, 2, 5, 7], device=device)
    idx[1, :4] = torch.tensor([1, 3, 8, 9], device=device)
    idx[:, 64] = 11
    idx[:, 127] = 2**31 - 1
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
    scratch = torch.empty(max(256, api.dsa_det_workspace_size(3, heads, dim, 11, idx.shape[1])), dtype=torch.uint8, device=gpu)

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
    scratch = torch.empty(api.dsa_det_workspace_size(3, heads, 128, 11, idx.shape[1], indexer=True), dtype=torch.uint8, device=gpu)

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
        selected = scores.gather(1, ids.clamp(0, 10)) * ((ids >= 0) & (ids < 11))
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


@pytest.mark.parametrize("heads,dim", [(8, 512), (8, 576), (16, 512), (16, 576), (32, 512), (32, 576), (64, 512), (64, 576)])
def test_attention_key_bounds_and_replay(gpu, heads, dim):
    from determinism_utils import assert_bitwise, assert_golden, snapshot

    q = torch.zeros((2, heads, dim), dtype=torch.bfloat16, device=gpu)
    backing = torch.full((4, dim), 7, dtype=torch.bfloat16, device=gpu)
    backing[0].fill_(1)
    kv = backing[:1]
    indices = torch.full((2, 128), -1, dtype=torch.int32, device=gpu)
    indices[:, [0, 63, 64, 127]] = torch.tensor([1, 3, 2**31 - 1, -(2**31)], dtype=torch.int32, device=gpu)
    indices[1, 65] = 0
    groups = api.dsa_prepare_segments(indices, 1)
    out = torch.empty((2, heads, 512), dtype=torch.bfloat16, device=gpu)
    lse = torch.empty((2, heads), dtype=torch.float32, device=gpu)
    grad = torch.ones_like(out)
    dq = torch.empty_like(q)
    dkv_backing = torch.full_like(backing, 13)
    dkv = dkv_backing[:1]
    scratch = torch.empty(api.dsa_det_workspace_size(2, heads, dim, 1, 128), dtype=torch.uint8, device=gpu)
    saved_indices = indices.clone()

    def run():
        api.dsa_det_forward(q, kv, indices, out=out, lse=lse, workspace=scratch)
        api.dsa_det_backward(q, kv, indices, out, lse, grad, **groups, dq=dq, dkv=dkv, workspace=scratch)
        return dict(out=out, lse=lse, dq=dq, dkv=dkv)

    def check():
        ref, ref_lse, grads = _attention_ref(q, kv, indices, grad)
        torch.testing.assert_close(out.cpu().double(), ref, atol=0, rtol=0)
        live = ((indices >= 0) & (indices < 1)).any(-1)
        torch.testing.assert_close(lse[live].cpu().double(), ref_lse[live.cpu()], atol=0, rtol=0)
        assert bool((lse[~live] < -1e6).all())
        for actual, expected in zip((dq, dkv), grads):
            torch.testing.assert_close(actual.cpu().double(), expected, atol=0, rtol=0)
        assert torch.equal(backing[1:], torch.full_like(backing[1:], 7))
        assert torch.equal(dkv_backing[1:], torch.full_like(dkv_backing[1:], 13))
        assert torch.equal(indices, saved_indices)

    def change():
        indices.copy_(indices.flip(0))
        saved_indices.copy_(indices)
        for name, value in api.dsa_prepare_segments(indices, 1).items():
            groups[name].copy_(value)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _capture(run, change, check, gpu)
        first = snapshot(run())
        assert_bitwise(run(), first, "bounded attention replay")
        one_q, one_indices = q[:1], indices[:1]
        one_groups = api.dsa_prepare_segments(one_indices, 1)
        one_out = api.dsa_det_forward(one_q, kv, one_indices)
        one_grads = api.dsa_det_backward(one_q, kv, one_indices, *one_out.values(), grad[:1], **one_groups)
        assert_bitwise(
            dict(**one_out, **one_grads),
            dict(out=first["out"][:1], lse=first["lse"][:1], dq=first["dq"][:1], dkv=first["dkv"]),
            "isolated live row with fresh buffers",
        )
        assert_golden(f"cake_dsa_key_bounds_h{heads}_d{dim}", dict(q=q, kv=kv, indices=indices, grad=grad), first)
    torch.cuda.current_stream().wait_stream(side)
