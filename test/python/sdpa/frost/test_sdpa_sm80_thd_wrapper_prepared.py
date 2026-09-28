# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone packed backward shares the prepared graph chain without D2H."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from sdpa.frost.test_sdpa_bwd_thd_sm80 import _thd_case
from sdpa.frost.test_sdpa_bwd_prepared_thd_sm80 import _check

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires SM80")]


def _prefix(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def _wrapper(case, cu_q, cu_k, **kwargs):
    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_sm80

    return sdpa_bwd_wrapper_sm80(
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        is_causal=case.causal,
        causal_bottom_right=case.bottom_right,
        scale_softmax=case.scale,
        cum_seqlen_q_tensor=cu_q,
        cum_seqlen_k_tensor=cu_k,
        **kwargs,
    )


@pytest.mark.parametrize("d,d_v", [(64, 64), (96, 96), (128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_wrapper_rebind_capture_and_capacity_tails(d, d_v, dtype, monkeypatch):
    """Rebind fresh buffers, then change data/prefixes under a captured launch."""
    from cudnn.sdpa.bwd import api_dsl

    cases = [_thd_case((96, 160), (128, 96), 4, d, dtype, hkv=2, d_v=d_v, cap_q=512, cap_kv=512, poison=True, causal=True, seed=seed) for seed in (29, 43)]
    cu_q, cu_k = _prefix(cases[0].cu_q), _prefix(cases[0].cu_k)
    # Capacity exceeds B * the sequence hints. Neither physical Stats pitch
    # nor workspace binding may silently shrink to that logical envelope.
    kwargs = dict(deterministic=True, max_s_q=160, max_s_kv=128)
    first = _wrapper(cases[0], cu_q, cu_k, **kwargs)
    _check(cases[0], *(first[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))
    import cutlass.cute.runtime as runtime

    monkeypatch.setattr(runtime, "from_dlpack", lambda *a, **kw: pytest.fail("THD wrapper reached legacy tensor plumbing"))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        second = _wrapper(cases[1], cu_q, cu_k, **kwargs)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    _check(cases[1], *(second[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _wrapper(cases[1], cu_q, cu_k, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    changed = _thd_case((128, 64), (64, 128), 4, d, dtype, hkv=2, d_v=d_v, cap_q=512, cap_kv=512, poison=True, causal=True, seed=61)
    for name in ("q", "k", "v", "o", "do", "lse"):
        getattr(cases[1], name).copy_(getattr(changed, name))
    cu_q.copy_(_prefix(changed.cu_q))
    cu_k.copy_(_prefix(changed.cu_k))
    for value in captured.values():
        value.fill_(float("nan"))
    try:
        graph.replay()
        _check(changed, *(captured[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))
        for key, live in (("dq_tensor", changed.t_q), ("dk_tensor", changed.t_kv), ("dv_tensor", changed.t_kv)):
            assert torch.count_nonzero(captured[key][:, live:]) == 0
    finally:
        graph.reset()


def test_wrapper_without_length_hints_does_not_sync(monkeypatch):
    """The former default copied cu_k to CPU even after a complete warmup."""
    case = _thd_case((96, 160), (128, 96), 4, 128, torch.float16, hkv=2)
    cq, ck = _prefix(case.cu_q), _prefix(case.cu_k)
    _wrapper(case, cq, ck)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            cq[0].item()
        out = _wrapper(case, cq, ck)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    _check(case, *(out[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))


def test_wrapper_capacity_and_hint_changes_reload_artifact(tmp_path, monkeypatch):
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache
    from cudnn.sdpa.bwd.api_dsl import _sm80_thd_plan

    from cudnn.sdpa.bwd.kernels.sm80.prepared_host import _compile_thd_artifact

    _sm80_thd_plan.cache_clear()
    _compile_thd_artifact.cache_clear()
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    first = _thd_case((96, 160), (128, 96), 4, 128, torch.float16, hkv=2)
    _wrapper(first, _prefix(first.cu_q), _prefix(first.cu_k), deterministic=True, max_s_q=160, max_s_kv=128)
    before = compiled_cache.stats()
    _compile_thd_artifact.cache_clear()
    monkeypatch.setattr(cute, "compile", lambda *a, **kw: pytest.fail("packed capacity/hint change triggered a fresh JIT"))
    second = _thd_case((128, 64), (64, 128), 4, 128, torch.float16, hkv=2, cap_q=512, cap_kv=768)
    out = _wrapper(second, _prefix(second.cu_q), _prefix(second.cu_k), deterministic=True)
    after = compiled_cache.stats()
    assert after["misses"] == before["misses"]
    assert after["hits"] > before["hits"]
    _check(second, *(out[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))


@pytest.mark.parametrize("q_prefix,kv_prefix", [(False, False), (False, True), (True, False), (True, True)])
def test_direct_prepared_length_forms(q_prefix, kv_prefix):
    from cudnn.sdpa.bwd.api_dsl import _sm80_thd_plan

    case = _thd_case((96, 160), (128, 96), 4, 128, torch.float16, hkv=2)
    api = _sm80_thd_plan(2, 4, 2, 128, 128, case.cap_q, case.cap_kv, 160, 128, case.dtype, case.q.device, False, (None, None), False, False, True)
    ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    outputs = [torch.zeros_like(t) for t in (case.q, case.k, case.v)]
    api.execute(
        *(t.transpose(1, 2) for t in (case.q, case.k, case.v, case.o, case.do)),
        case.lse,
        *(t.transpose(1, 2) for t in outputs),
        workspace=ws,
        seq_q_lens=_prefix(case.cu_q if q_prefix else case.lens_q),
        seq_kv_lens=_prefix(case.cu_k if kv_prefix else case.lens_kv),
    )
    _check(case, *outputs)


@pytest.mark.parametrize("empty", ["q", "kv", "both"])
@pytest.mark.parametrize("hkv", [2, 4])
def test_wrapper_zero_capacity(empty, hkv):
    case = _thd_case((64, 64), (64, 64), 4, 128, torch.float16, hkv=hkv)
    if empty in ("q", "both"):
        case.q, case.o, case.do = (t[:, :0] for t in (case.q, case.o, case.do))
        case.lse = case.lse[..., :0]
        case.cu_q = [0, 0, 0]
    if empty in ("kv", "both"):
        case.k, case.v = case.k[:, :0], case.v[:, :0]
        case.cu_k = [0, 0, 0]
        case.o.zero_()
        case.lse.zero_()
    result = _wrapper(case, _prefix(case.cu_q), _prefix(case.cu_k))
    for name, inp in (("dq_tensor", case.q), ("dk_tensor", case.k), ("dv_tensor", case.v)):
        assert result[name].shape == inp.shape
        assert torch.count_nonzero(result[name]) == 0


@pytest.mark.parametrize("cache_mode", ["disabled", "unknown_manifest"])
@pytest.mark.parametrize("d,dv", [(64, 64), (96, 80), (128, 128), (192, 128), (256, 256)])
def test_wrapper_capacity_reuses_artifact_without_disk_cache(d, dv, cache_mode, tmp_path, monkeypatch):
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache
    from cudnn.sdpa.bwd.api_dsl import _sm80_thd_plan
    from cudnn.sdpa.bwd.kernels.sm80 import prepared_host

    _sm80_thd_plan.cache_clear()
    memo = getattr(prepared_host, "_compile_thd_artifact", None)
    if memo is not None:
        memo.cache_clear()
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    if cache_mode == "disabled":
        monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "1")
    else:
        monkeypatch.setattr(compiled_cache, "environment_manifest", lambda: {"cuda_driver": "unknown"})
    first = _thd_case((96, 160), (128, 96), 4, d, torch.float16, hkv=2, d_v=dv)
    out = _wrapper(first, _prefix(first.cu_q), _prefix(first.cu_k), deterministic=True, max_s_q=160, max_s_kv=128)
    _check(first, *(out[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))
    monkeypatch.setattr(cute, "compile", lambda *a, **kw: pytest.fail("new capacity plan recompiled without disk cache"))
    second = _thd_case((128, 64), (64, 128), 4, d, torch.float16, hkv=2, d_v=dv, cap_q=512, cap_kv=768)
    # Force another plan, including its different workspace and launch bounds.
    cq, ck = _prefix(second.cu_q), _prefix(second.cu_k)
    out = _wrapper(second, cq, ck, deterministic=True)
    _check(second, *(out[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = _wrapper(second, cq, ck, deterministic=True)
        for value in captured.values():
            value.fill_(float("nan"))
        graph.replay()
        _check(second, *(captured[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))
    finally:
        graph.reset()
