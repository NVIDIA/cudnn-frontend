# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense d128 backward: graph eligibility, one launch, zero workspace, replay."""

from types import SimpleNamespace

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, select_engine
from test_sdpa_bwd_dsl_sm107 import _adapter, _bshd_empty, _build_graph, _causal_keep, _check, _half_bwd_graph, _reference64

pytestmark = [pytest.mark.L0, requires_dsl]
_ENGINE = "sdpa_bwd_sm100_d128"


def _spec():
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    return next(s for s in ENGINE_SPECS if s.name == _ENGINE)


def _reason(monkeypatch, cc=(10, 0), **kw):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    graph, _, _ = _half_bwd_graph(d=kw.pop("d", 128), **kw)
    facts = ga.analyze(graph)
    assert facts is not None
    return mismatch(_spec().capabilities, facts)


@pytest.mark.parametrize(
    "kw",
    [
        dict(dt=torch.float16),
        dict(sq=1, skv=65),
        dict(sq=257, skv=129),
    ],
)
def test_dense_graph_accepts(monkeypatch, kw):
    assert _reason(monkeypatch, **kw) is None


@pytest.mark.parametrize(
    "kw,token",
    [
        (dict(d=64), "D"),
        (dict(d_v=64), "D_QK"),
        (dict(use_causal_mask_bottom_right=True), "bottom-right"),
        (dict(use_causal_mask=True, diagonal_band_left_bound=64), "sliding window"),
        (dict(diagonal_band_right_bound=8), "right-band"),
        (dict(padded=True), "padding mask"),
        (dict(bias=True), "bias"),
        (dict(sink=True), "Sink"),
        (dict(use_deterministic_algorithm=True), "deterministic"),
    ],
)
def test_dense_graph_declines(monkeypatch, kw, token):
    reason = _reason(monkeypatch, **kw)
    assert reason is not None and token.lower() in reason.lower(), reason


@pytest.mark.parametrize("port", ["q", "k", "dq", "dv"])
def test_padded_bshd_port_is_declined(monkeypatch, port):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    graph, tensors, outputs = _half_bwd_graph(d=128)
    tensors.update(zip(("dq", "dk", "dv"), outputs))
    tensor = tensors[port]
    strides = list(tensor.get_stride())
    strides[0] += 128
    tensor.set_stride(strides)
    facts = ga.analyze(graph)
    assert facts.bshd_layout, "This tests compactness, beyond the shared stride-order check"
    reason = mismatch(_spec().capabilities, facts)
    assert reason is not None and "compact BSHD" in reason, reason


def test_strided_stats_is_declined(monkeypatch):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    graph, tensors, _ = _half_bwd_graph(d=128)
    tensors["stats"].set_stride([1024, 512, 1, 1])
    assert "stats" in mismatch(_spec().capabilities, ga.analyze(graph)).lower()


def test_older_extension_declines_before_compilation(monkeypatch):
    from cudnn import _pybind_module
    from cudnn.frost import device
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128

    monkeypatch.setattr(device, "compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(_pybind_module, "_SdpaBwdBinder", SimpleNamespace())
    api = _adapter(SdpaBwdDslSm100D128, d=128)
    with pytest.raises(NotImplementedError, match="zero-workspace backward binding"):
        api.check_support()


@pytest.mark.parametrize(
    "kw,token",
    [
        (dict(thd=True), "THD"),
        (dict(causal_bottom_right=True), "bottom-right"),
        (dict(d=256), "128"),
        (dict(tile_m=128), "fixed tile"),
        (dict(tile_n=128), "fixed tile"),
    ],
)
def test_adapter_backstops_decline_unsupported_declarations(monkeypatch, kw, token):
    from cudnn import _pybind_module
    from cudnn.frost import device
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128

    monkeypatch.setattr(device, "compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(_pybind_module, "_SdpaBwdBinder", SimpleNamespace(supports_zero_workspace=True))
    api = _adapter(SdpaBwdDslSm100D128, **{"d": 128, **kw})
    with pytest.raises(NotImplementedError, match=token):
        api.check_support()


def _case(dt=torch.bfloat16, causal=False, b=1, h=2, sq=65, sk=129, seed=729, check_build=False, accumulation=False, scale=None):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(s):
        return torch.randn(b, s, h, 128, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, k, v, do = draw(sq), draw(sk), draw(sk), draw(sq)
    keep = _causal_keep(sq, sk) if causal else None
    o64, lse, _, *expected = _reference64(q, k, v, do, keep, scale=scale)
    tensors = dict(q=q, k=k, v=v, do=do, o=_bshd_empty(b, sq, h, 128, dt).copy_(o64), stats=lse.float().unsqueeze(-1).contiguous())
    tensors.update(
        dq=_bshd_empty(b, sq, h, 128, dt, fill=float("nan")),
        dk=_bshd_empty(b, sk, h, 128, dt, fill=float("nan")),
        dv=_bshd_empty(b, sk, h, 128, dt, fill=float("nan")),
    )
    graph, refs, outputs = _build_graph(b=b, hq=h, sq=sq, skv=sk, d=128, dt=dt, use_causal_mask=causal, scale="default" if scale is None else scale)
    select_engine(graph, _ENGINE)
    before = torch.cuda.memory_allocated()
    if check_build:
        torch.cuda.set_sync_debug_mode("error")
    try:
        graph.check_support()
        graph.build_plans()
    finally:
        if check_build:
            torch.cuda.set_sync_debug_mode("default")
    if check_build:
        assert torch.cuda.memory_allocated() == before
    required = 4 * b * h * sq * 129 + 240 if accumulation else 0
    assert graph.get_workspace_size() == required
    refs.update(zip(("dq", "dk", "dv"), outputs))
    pack = {refs[n]: value for n, value in tensors.items()}
    prepared = graph._compiled_plans[graph._plan_index]._prepared
    assert prepared is not None and prepared.spec.workspace_bytes == required
    assert prepared.spec.artifact.implementation == ("accumulation" if accumulation else "recompute")
    assert len(prepared.spec.operands) == 9 and not prepared.spec.scale_log2
    workspace = torch.full((required,), 0xA5, dtype=torch.uint8, device="cuda") if required else None
    graph.execute(pack, workspace)
    case = SimpleNamespace(graph=graph, tensors=tensors, refs=refs, pack=pack, expected=expected, keep=keep, dt=dt, workspace=workspace)
    _check_case(case)
    return case


def _check_case(case, tensors=None, expected=None):
    tensors = case.tensors if tensors is None else tensors
    expected = case.expected if expected is None else expected
    for name, want in zip(("dq", "dk", "dv"), expected):
        _check(name, tensors[name], want, case.dt)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize(
    "dt,causal,sq,sk",
    [
        (torch.bfloat16, False, 1, 65),
        (torch.bfloat16, True, 1, 65),
        (torch.float16, True, 65, 1),
        (torch.bfloat16, False, 65, 1),
        (torch.bfloat16, True, 65, 129),
        (torch.float16, False, 257, 193),
        (torch.bfloat16, False, 513, 521),
        (torch.float16, True, 257, 193),
    ],
)
def test_graph_matches_fp64(dt, causal, sq, sk):
    b, h = (2, 3) if (sq, sk) == (257, 193) else (1, 2)
    _case(dt=dt, causal=causal, sq=sq, sk=sk, b=b, h=h)


@requires_pre_rubin_blackwell
def test_one_launch_without_workspace_allocation_or_sync():
    case = _case(check_build=True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.set_sync_debug_mode("error")
    try:
        case.graph.execute(case.pack, None)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_allocated() == before
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        case.graph.execute(case.pack, None)
    events = [event for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    assert len(events) == 1, [event.name for event in events]
    assert "bprop_d128" in events[0].name
    _check_case(case)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt,causal", [(torch.bfloat16, False), (torch.float16, True)], ids=["bf16-unmasked", "fp16-causal"])
def test_rebind_and_replay(dt, causal):
    case = _case(dt=dt, causal=causal, b=2, h=3)
    tensors = {name: value.clone() for name, value in case.tensors.items()}
    pack = {case.refs[name]: value for name, value in tensors.items()}
    stream, other = torch.cuda.Stream(), torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    capture = torch.cuda.CUDAGraph()

    def refresh():
        tensors["q"].mul_(0.75)
        tensors["do"].mul_(1.25)
        o, lse, _, *expected = _reference64(tensors["q"], tensors["k"], tensors["v"], tensors["do"], case.keep)
        tensors["o"].copy_(o)
        tensors["stats"].copy_(lse.unsqueeze(-1))
        for name in ("dq", "dk", "dv"):
            tensors[name].fill_(float("nan"))
        return expected

    try:
        expected = refresh()
        stream.wait_stream(torch.cuda.current_stream())
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            case.graph.execute(pack, None, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        _check_case(case, tensors, expected)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            with torch.cuda.stream(other):
                case.graph.execute(pack, None, handle=handle)
        expected = refresh()
        capture.replay()
        torch.cuda.synchronize()
        _check_case(case, tensors, expected)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


@requires_pre_rubin_blackwell
def test_replan_reloads_artifact_without_compilation(monkeypatch, tmp_path):
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache

    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    monkeypatch.delenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", raising=False)
    compiled_cache.reset_stats()
    _case(sq=97, sk=131)
    first = compiled_cache.stats()
    assert first["misses"] > 0 and first["hits"] == 0, first

    def forbidden(*args, **kwargs):
        raise AssertionError("rebuilding the same declaration compiled instead of reloading its artifact")

    monkeypatch.setattr(cute, "compile", forbidden)
    case = _case(sq=97, sk=131, seed=730)
    assert compiled_cache.stats()["hits"] > first["hits"]
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            case.graph.execute(case.pack, None)
        for name in ("dq", "dk", "dv"):
            case.tensors[name].fill_(float("nan"))
        capture.replay()
        _check_case(case)
    finally:
        capture.reset()
