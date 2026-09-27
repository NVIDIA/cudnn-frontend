# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed SM80 forward pointer binding, replay and artifact reuse."""

import itertools
import math

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

pytestmark = [requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires SM80")]


def _prefix(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda")


def _inputs(d, dv, dtype, capq=384, capkv=448, seed=57):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(torch.randn(1, t, h, dim, generator=g, device="cuda", dtype=dtype) * 0.4 for t, h, dim in ((capq, 4, d), (capkv, 2, d), (capkv, 2, dv)))


def _run(tensors, cq, ck, *, causal=False, bottom=False, window=-1, sink=None, maxq=192, stream=None):
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80

    return sdpa_fwd_wrapper_sm80(
        *tensors,
        is_causal=causal,
        causal_bottom_right=bottom,
        window_size=(window, -1),
        sinks=sink,
        cum_seqlen_q_tensor=cq,
        cum_seqlen_k_tensor=ck,
        max_s_q=maxq,
        current_stream=stream,
    )


def _check(tensors, out, lq, lk, *, causal=False, bottom=False, window=-1, sink=None):
    q, k, v = tensors
    qs = ks = 0
    for nq, nk in zip(lq, lk):
        if nq:
            qq = q[0, qs : qs + nq].transpose(0, 1).double()
            kk = k[0, ks : ks + nk].transpose(0, 1).double().repeat_interleave(2, 0)
            vv = v[0, ks : ks + nk].transpose(0, 1).double().repeat_interleave(2, 0)
            logits = qq @ kk.transpose(-1, -2) / math.sqrt(q.shape[-1])
            row = torch.arange(nq, device=q.device)[:, None] + (nk - nq if bottom else 0)
            col = torch.arange(nk, device=q.device)[None, :]
            mask = torch.ones((nq, nk), dtype=torch.bool, device=q.device)
            if causal:
                mask &= col <= row
            if window >= 0:
                mask &= col >= row - window
            logits.masked_fill_(~mask, -torch.inf)
            stats = torch.logsumexp(logits, -1)
            if sink is not None:
                stats = torch.logaddexp(stats, sink.double()[:, None])
            probs = torch.exp(logits - stats[..., None]).nan_to_num()
            expected = (probs @ vv).transpose(0, 1)
            got = out["o_tensor"][0, qs : qs + nq].double()
            torch.testing.assert_close(got, expected, rtol=0.02, atol=0.003 * max(1.0, expected.abs().max().item()))
            torch.testing.assert_close(out["lse_tensor"][0, :, qs : qs + nq].double(), stats, rtol=0.002, atol=0.002)
        qs += nq
        ks += nk
    assert torch.count_nonzero(out["o_tensor"][:, qs:]) == 0
    assert torch.count_nonzero(out["lse_tensor"][..., qs:]) == 0


@pytest.mark.parametrize("d,dv", [(64, 64), (96, 96), (128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("features", [False, True])
@pytest.mark.L0
def test_thd_wrapper_rebind_and_capture(d, dv, dtype, features, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl

    tensors = _inputs(d, dv, dtype)
    lq, lk = (96, 129), (65, 193)
    cq, ck = _prefix(lq), _prefix(lk)
    sink = torch.tensor([-0.7, 0.4, 1.1, -0.2], device="cuda") if features else None
    kw = dict(causal=features, bottom=features, window=64 if features else -1, sink=sink)
    _check(tensors, _run(tensors, cq, ck, **kw), lq, lk, **kw)
    fresh = _inputs(d, dv, dtype, seed=81)
    # A return to tensor launch plumbing must fail even if numerics agree.
    monkeypatch.setattr(api_dsl, "_sm80_call", lambda *a, **k: pytest.fail("legacy tensor launch"))
    _check(fresh, _run(fresh, cq, ck, **kw), lq, lk, **kw)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            out = _run(tensors, cq, ck, **kw)
        lq, lk = (80, 145), (96, 162)
        cq.copy_(_prefix(lq))
        ck.copy_(_prefix(lk))
        for t, new in zip(tensors, fresh):
            t.copy_(new)
        if sink is not None:
            sink.add_(0.5)
        out["o_tensor"].fill_(float("nan"))
        out["lse_tensor"].fill_(float("nan"))
        graph.replay()
        _check(tensors, out, lq, lk, **kw)
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("cache_mode", ["disk", "disabled", "unknown_manifest"])
def test_thd_wrapper_artifact_survives_capacity_change(cache_mode, tmp_path, monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.fwd.kernels.sm80.prepared_host import compile_thd_host
    from cudnn.frost import compiled_cache

    compile_thd_host.cache_clear()
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    if cache_mode == "disabled":
        monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "1")
    elif cache_mode == "unknown_manifest":
        monkeypatch.setattr(compiled_cache, "environment_manifest", lambda: {"cuda_driver": "unknown"})
    a = _inputs(128, 128, torch.float16)
    lq, lk = (96, 129), (65, 193)
    cq, ck = _prefix(lq), _prefix(lk)
    _check(a, _run(a, cq, ck), lq, lk)
    before = compiled_cache.stats()
    if cache_mode == "disk":
        compile_thd_host.cache_clear()
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("artifact was not reloadable"))
    b = _inputs(128, 128, torch.float16, capq=512, capkv=640, seed=93)
    lq, lk = (129, 201), (80, 289)
    _check(b, _run(b, _prefix(lq), _prefix(lk), maxq=256), lq, lk)
    if cache_mode == "disk":
        assert compiled_cache.stats()["hits"] > before["hits"]


@pytest.mark.L0
def test_thd_wrapper_explicit_stream_and_no_sync():
    tensors = _inputs(64, 64, torch.bfloat16)
    lq, lk = (96, 129), (65, 193)
    cq, ck = _prefix(lq), _prefix(lk)
    _run(tensors, cq, ck)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = _run(tensors, cq, ck, stream=stream.cuda_stream)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    _check(tensors, out, lq, lk)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("role", ["q", "k", "v"])
@pytest.mark.parametrize("product", [False, True])
@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (192, 128), (256, 256)])
def test_thd_wrapper_physical_stride_and_product(role, product, d, dv):
    from sdpa.frost.test_sdpa_bwd_prepared_sm80 import _wide_buffer

    lens = (5,) if product else (2, 2)
    src = _inputs(d, dv, torch.bfloat16, capq=sum(lens), capkv=sum(lens))
    tensors = list(src)
    index = ("q", "k", "v").index(role)
    tensors[index], backing = _wide_buffer(src[index], product=product, axis=1)
    cu = _prefix(lens)
    out = _run(tensors, cu, cu, maxq=max(lens))
    # The reference reads the compact original; wrapped addresses contain NaNs
    # inside the allocation, so narrowing controls fail numerically and safely.
    _check(src, out, lens, lens)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = _run(tensors, cu, cu, maxq=max(lens))
        tensors[index].mul_(0.5)
        src[index].mul_(0.5)
        captured["o_tensor"].fill_(float("nan"))
        captured["lse_tensor"].fill_(float("nan"))
        graph.replay()
        _check(src, captured, lens, lens)
    finally:
        graph.reset()
    assert backing.numel() >= tensors[index].numel()


@pytest.mark.L0
@pytest.mark.parametrize("d,dv,rope", [(96, 96, False), (192, 96, False), (128, 128, True), (256, 256, True)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_tensor_fallback_survives_thd_fake_cleanup(d, dv, rope, dtype, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl, sdpa_fwd_wrapper_sm80

    q, k, v = (t.transpose(1, 2) for t in _inputs(d, dv, dtype, capq=128, capkv=256))
    freqs = torch.arange(256, device="cuda", dtype=torch.float32)[:, None] * torch.linspace(0.001, 0.01, d // 2, device="cuda")[None, :] if rope else None
    calls = []
    original = api_dsl._sm80_call

    def tensor_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(api_dsl, "_sm80_call", tensor_call)

    def run():
        return sdpa_fwd_wrapper_sm80(q, k, v, is_causal=True, rope_freqs=freqs)

    def check(out):
        qr, kr = q.double(), k.double()
        if rope:

            def rotate(t):
                angles = freqs[: t.shape[2]].double()[None, None]
                a, b = t.chunk(2, -1)
                return torch.cat((a * angles.cos() - b * angles.sin(), b * angles.cos() + a * angles.sin()), -1).to(dtype).double()

            qr, kr = rotate(qr), rotate(kr)
        logits = qr @ kr.repeat_interleave(2, 1).transpose(-1, -2) / math.sqrt(d)
        logits.masked_fill_(torch.arange(256, device="cuda")[None, :] > torch.arange(128, device="cuda")[:, None], -torch.inf)
        stats = logits.logsumexp(-1)
        ref = logits.softmax(-1) @ v.double().repeat_interleave(2, 1)
        torch.testing.assert_close(out["o_tensor"].double(), ref, rtol=0.02, atol=0.003)
        torch.testing.assert_close(out["lse_tensor"].double(), stats, rtol=0.002, atol=0.002)

    check(run())
    assert calls, "the dense off-flavor/RoPE case must exercise the retained tensor entry"
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run()
        q.mul_(0.8)
        if freqs is not None:
            freqs.add_(0.1)
        for value in captured.values():
            value.fill_(float("nan"))
        graph.replay()
        check(captured)
    finally:
        graph.reset()
