# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM80 packed/staged forward pointer binding, replay and artifact reuse."""

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
    from cudnn.sdpa.fwd import api_dsl, prepared_sm80_thd

    bound = []
    original = prepared_sm80_thd.execute

    def native_execute(launch, *args):
        assert type(launch).__name__ == "_SdpaSm80ThdBinder"
        bound.append(launch)
        return original(launch, *args)

    monkeypatch.setattr(prepared_sm80_thd, "execute", native_execute)

    tensors = _inputs(d, dv, dtype)
    lq, lk = (96, 129), (65, 193)
    cq, ck = _prefix(lq), _prefix(lk)
    sink = torch.tensor([-0.7, 0.4, 1.1, -0.2], device="cuda") if features else None
    kw = dict(causal=features, bottom=features, window=64 if features else -1, sink=sink)
    _check(tensors, _run(tensors, cq, ck, **kw), lq, lk, **kw)
    assert len(bound) == 1, "packed wrapper must use its native binder"
    fresh = _inputs(d, dv, dtype, seed=81)
    # A return to tensor launch plumbing must fail even if numerics agree.
    import cutlass.cute.runtime as runtime

    monkeypatch.setattr(runtime, "from_dlpack", lambda *a, **k: pytest.fail("legacy tensor launch"))
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
@pytest.mark.parametrize("d,dv", [(128, 128), (96, 80)])
def test_thd_wrapper_artifact_survives_capacity_change(cache_mode, d, dv, tmp_path, monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.fwd.prepared_sm80_thd import build_launch
    from cudnn.sdpa.fwd.kernels.sm80.prepared_host import compile_thd_host
    from cudnn.frost import compiled_cache

    build_launch.cache_clear()
    compile_thd_host.cache_clear()
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    if cache_mode == "disabled":
        monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "1")
    elif cache_mode == "unknown_manifest":
        monkeypatch.setattr(compiled_cache, "environment_manifest", lambda: {"cuda_driver": "unknown"})
    a = _inputs(d, dv, torch.float16)
    lq, lk = (96, 129), (65, 193)
    cq, ck = _prefix(lq), _prefix(lk)
    _check(a, _run(a, cq, ck), lq, lk)
    before = compiled_cache.stats()
    if cache_mode == "disk":
        # Discard both the immutable launch plan and its compiled-host cache;
        # this leg must exercise a real disk reload, not a still-warm binder.
        build_launch.cache_clear()
        compile_thd_host.cache_clear()
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("artifact was not reloadable"))
    b = _inputs(d, dv, torch.float16, capq=512, capkv=640, seed=93)
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
def test_dense_staged_pointer_launch_and_rope_replay(d, dv, rope, dtype, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl, sdpa_fwd_wrapper_sm80

    q, k, v = (t.transpose(1, 2) for t in _inputs(d, dv, dtype, capq=128, capkv=256))
    freqs = torch.arange(256, device="cuda", dtype=torch.float32)[:, None] * torch.linspace(0.001, 0.01, d // 2, device="cuda")[None, :] if rope else None
    calls = []
    from cudnn.sdpa.fwd import prepared_staged_sm80

    module, entry = prepared_staged_sm80, "execute" if rope else "_copy"
    original = getattr(module, entry)

    def staged_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, entry, staged_call)

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
    assert calls, "the dense off-flavor/RoPE case must exercise the staged entry"
    import cutlass.cute.runtime as runtime

    def forbid_dlpack(*args, **kwargs):
        pytest.fail("SM80 staged launch rebuilt DLPack operands")

    monkeypatch.setattr(runtime, "from_dlpack", forbid_dlpack)
    check(run())
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


@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("features", [False, True])
@pytest.mark.L0
def test_thd_wrapper_preserves_packed_row_origins(d, dv, dtype, features):
    tensors = _inputs(d, dv, dtype)
    lq, lk = (96, 0, 129), (65, 0, 193)
    base_q, base_k = 5, 13
    cq, ck = _prefix(lq) + base_q, _prefix(lk) + base_k
    sink = torch.tensor([-0.7, 0.4, 1.1, -0.2], device="cuda") if features else None
    kw = dict(causal=features, bottom=features, window=64 if features else -1, sink=sink)

    def check(out):
        # This standalone API accepts packed row offsets into the supplied
        # storage. Graph API cumulative-length normalization is a different path.
        q, k, v = tensors
        sliced = dict(o_tensor=out["o_tensor"][:, base_q:], lse_tensor=out["lse_tensor"][..., base_q:])
        _check((q[:, base_q:], k[:, base_k:], v[:, base_k:]), sliced, lq, lk, **kw)
        assert torch.count_nonzero(out["o_tensor"][:, :base_q]) == 0
        assert torch.count_nonzero(out["lse_tensor"][..., :base_q]) == 0

    check(_run(tensors, cq, ck, **kw))
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            out = _run(tensors, cq, ck, **kw)
        lq, lk = (80, 0, 145), (96, 0, 162)
        base_q, base_k = 9, 27
        cq.copy_(_prefix(lq) + base_q)
        ck.copy_(_prefix(lk) + base_k)
        for t, new in zip(tensors, _inputs(d, dv, dtype, seed=83)):
            t.copy_(new)
        for value in out.values():
            value.fill_(float("nan"))
        graph.replay()
        check(out)
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("d,dv,rope", [(96, 96, False), (192, 96, False), (128, 128, True), (256, 256, True)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_dense_staged_artifact_reload_in_new_process(d, dv, rope, dtype, tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    import cudnn

    child = r"""
import json, sys
from pathlib import Path
import torch, pytest, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache
from cudnn.sdpa.fwd.kernels.sm80 import prepared_host
tests, package, d, dv, rope, dtype, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve(), cudnn.__file__
assert torch.cuda.get_device_capability() == (8, 0)
sys.path.insert(0, tests)
from test_sdpa_sm80_thd_forward_prepared import test_dense_staged_pointer_launch_and_rope_replay
from cudnn.sdpa.fwd import prepared_staged_sm80
module, entry = prepared_staged_sm80, "compile_plan"
original = getattr(module, entry)
artifacts = []
def record(*args):
    result = original(*args)
    artifacts.append(result.core.artifact)
    artifacts.extend(c[0] for c in result.copies if c is not None)
    return result
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(module, entry, record)
    if reload == "1":
        def forbidden(*args, **kwargs):
            raise AssertionError("staged forward invoked JIT in the second process")
        patch.setattr(cute, "compile", forbidden)
    test_dense_staged_pointer_launch_and_rope_replay(int(d), int(dv), rope == "1", getattr(torch, dtype), patch)
assert artifacts
if reload == "1":
    assert all(hasattr(a, "_compiled_cache_raw") for a in artifacts)
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, str(Path(__file__).parent), cudnn.__file__, str(d), str(dv), str(int(rope)), dtype, str(reload)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    first, second = results
    assert first["misses"] > 0 and first["hits"] == 0, first
    assert second["misses"] == 0 and second["hits"] > 0, second


@pytest.mark.L0
@pytest.mark.parametrize("d,dv,rope", [(96, 96, False), (192, 96, False), (128, 128, True), (256, 256, True)])
@pytest.mark.parametrize("cache_mode", ["disabled", "unknown_manifest"])
def test_dense_staged_process_reuse_without_persistence(d, dv, rope, cache_mode, tmp_path, monkeypatch):
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache
    from cudnn.sdpa.fwd.api_dsl import _sm80_wrapper_cache

    from cudnn.sdpa.fwd.prepared_staged_sm80 import _compile_core
    from cudnn.sdpa.fwd.kernels.sm80.staged_copy import compile_gather
    from cudnn.sdpa.fwd.kernels.staged_copy import compile_copy

    for cached in (_compile_core, compile_gather, compile_copy):
        cached.cache_clear()
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    if cache_mode == "disabled":
        monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "1")
    else:
        monkeypatch.setattr(compiled_cache, "environment_manifest", lambda: {"cuda_driver": "unknown"})
    for reload in (0, 1):
        _sm80_wrapper_cache.clear()
        with pytest.MonkeyPatch.context() as patch:
            if reload:
                patch.setattr(cute, "compile", lambda *a, **k: pytest.fail("staged forward lost process-local artifact reuse"))
            test_dense_staged_pointer_launch_and_rope_replay(d, dv, rope, torch.float16, patch)


@pytest.mark.L0
@pytest.mark.parametrize(
    "role,invalid",
    [(role, invalid) for role in ("q", "k", "v", "o") for invalid in ("shape", "dtype", "device")]
    + [(role, invalid) for role in ("lse", "sinks", "seq_q_lens", "seq_kv_lens", "bias_tensor") for invalid in ("dtype", "device")]
    + [("bias_tensor", "empty_batch")],
)
def test_dense_staged_rejects_invalid_operands_before_staging(role, invalid, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl, prepared_staged_sm80

    q = torch.empty((2, 17, 4, 96), device="cuda", dtype=torch.float16).transpose(1, 2)
    kv = torch.empty((2, 33, 2, 96), device="cuda", dtype=torch.float16).transpose(1, 2)
    values = dict(q=q, k=kv, v=kv, o=torch.empty_like(q), lse=torch.empty((2, 4, 17), device="cuda"))
    values.update(
        sinks=torch.ones(4, device="cuda"),
        seq_q_lens=torch.ones(2, device="cuda", dtype=torch.int32),
        seq_kv_lens=torch.ones(2, device="cuda", dtype=torch.int32),
        bias_tensor=torch.zeros((1, 4, 17, 33), device="cuda"),
    )
    api = api_dsl.SdpaFwdDslSm80(
        **{"sample_" + n: values[n] for n in ("q", "k", "v", "o", "lse")},
        has_sink=True,
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
        bias_present=True,
        bias_fp32=True,
    )
    api.check_support()
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    monkeypatch.setattr(prepared_staged_sm80, "_copy", lambda *a, **k: pytest.fail("invalid operand reached workspace staging"))
    tensor = values[role]
    if invalid == "shape":
        values[role] = tensor[:, :, :-1]
    elif invalid == "dtype":
        values[role] = torch.empty_like(tensor, dtype=torch.float64)
    elif invalid == "empty_batch":
        values[role] = tensor[:0]
    else:
        values[role] = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
    with pytest.raises(ValueError):
        api.execute(
            **{n + "_tensor": values[n] for n in ("q", "k", "v", "o", "lse")},
            **{n: v for n, v in values.items() if n not in ("q", "k", "v", "o", "lse")},
            workspace=workspace,
        )
