# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SM107 D128 graph binding, gradients, launch count and replay."""

from dataclasses import replace
from unittest.mock import patch

import cudnn
import pytest
import torch

from frost_test_utils import select_engine
import test_sdpa_bwd_d128_sm100 as dense_cases
import test_sdpa_bwd_d128_sm107_contract as contract_cases
from test_sdpa_bwd_dsl_sm107 import _build_graph, _check, _half_bwd_graph, _reference64

pytestmark = [pytest.mark.L0]
_ENGINE = "sdpa_bwd_sm107_d128"


def _require_native():
    from cudnn.frost import cutedsl_runtime_available
    from cudnn.frost.buffers import cutedsl_state
    from cudnn.sdpa.bwd.d128_sm107_contract import validate_dsl_version

    cc = torch.cuda.get_device_capability()
    if cc != (10, 7):
        pytest.skip(f"native D128 backward requires SM107; got {cc}")
    installed, version = cutedsl_state()
    try:
        validate_dsl_version(version if installed else None)
    except ValueError as error:
        pytest.skip(str(error))
    if not cutedsl_runtime_available(device_cc=cc):
        pytest.skip("native SM107 D128 backward requires support for this engine's target")


def _case(**kwargs):
    _require_native()
    # Share the existing FP64 oracle and tolerances, including its exact
    # accumulation workspace and natural-log Stats assertions.
    with patch.object(dense_cases, "_ENGINE", _ENGINE):
        return dense_cases._case(accumulation=True, **kwargs)


def test_native_binding_revalidates_current_storage():
    """Host only: the real extension binds the native host's twelve slots."""
    from cudnn.sdpa.bwd import prepared
    from cudnn.sdpa.fwd.prepared import BufferFacts, _native_pack_from_facts

    spec, _, frames = contract_cases.TestPreparedPlan().prepared()
    spec = replace(spec, native_binding=True)
    facts = {
        role: BufferFacts(0x1000000 + i * 0x1000000, op.dtype, (2, 0), op.span, op.shape, op.strides)
        for i, (role, op) in enumerate(zip(spec.roles, spec.operands))
    }
    indices, workspace = tuple(range(9)), 0x40000000

    def bind(current, scale=None):
        pack = _native_pack_from_facts(current, spec.roles)
        return tuple(spec.native.bind(pack, indices, workspace, 17, (), scale, True, None))

    expected = tuple(prepared._bind_python(spec, facts, workspace, 17, raw_storage=True))
    assert bind(facts) == expected
    rebound = {role: fact._replace(ptr=fact.ptr + 4096) for role, fact in facts.items()}
    assert bind(rebound)[:9] == tuple(rebound[role].ptr for role in spec.roles)
    assert bind(rebound, 0.25)[-2:] == (0.25, 17)
    for change in (dict(dtype="float32"), dict(ptr=facts["q"].ptr + 2), dict(span=1), dict(device=(2, 1))):
        with pytest.raises(ValueError):
            bind(facts | {"q": facts["q"]._replace(**change)})
    spec.fn(*expected)
    assert frames == [expected]


@pytest.mark.parametrize("key", ["score_mod", "score_mod_bprop"])
def test_real_graph_score_modifier_declines_before_lowering(monkeypatch, key):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd import engines

    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 7))
    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.1")))
    monkeypatch.setattr(engines, "cutedsl_arch_requirement_error", lambda cc: None)
    row = next(spec for spec in engines.ENGINE_SPECS if spec.name == _ENGINE)
    control, _, _ = _half_bwd_graph(d=128)
    assert engines.mismatch(row.capabilities, ga.analyze(control)) is None
    graph, _, _ = _half_bwd_graph(d=128, **{key: lambda graph, score: score})
    facts = ga.analyze(graph)
    assert facts.has_score_mod
    assert "score_mod" in engines.mismatch(row.capabilities, facts)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("sq,sk", [(128, 256), (257, 193)], ids=["aligned", "tails"])
def test_graph_matches_fp64(dtype, causal, sq, sk):
    _case(dt=dtype, causal=causal, b=2, h=3, sq=sq, sk=sk)


def test_three_native_launches_build_and_execute_without_allocation_or_sync():
    case = _case(dt=torch.bfloat16, causal=True, b=1, h=2, sq=128, sk=256, check_build=True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    old_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            case.graph.execute(case.pack, case.workspace)
    finally:
        torch.cuda.set_sync_debug_mode(old_mode)
    assert torch.cuda.memory_allocated() == before
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        case.graph.execute(case.pack, case.workspace)
    names = [event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    assert len(names) == 3, names
    assert any("bprop_d128_accum_sm107" in name for name in names), names
    dense_cases._check_case(case)


@pytest.mark.parametrize("dtype,causal", [(torch.bfloat16, False), (torch.float16, True)], ids=["bf16-dense", "fp16-causal"])
def test_fresh_binding_first_execute_capture_and_changed_input_replay(dtype, causal, cudnn_handle):
    case = _case(dt=dtype, causal=causal, b=2, h=3, sq=128, sk=256)
    tensors = {name: value.clone() for name, value in case.tensors.items()}
    graph, refs, outputs = _build_graph(b=2, hq=3, sq=128, skv=256, d=128, dt=dtype, use_causal_mask=causal, scale="default")
    select_engine(graph, _ENGINE)
    graph.check_support()
    graph.build_plans()
    refs.update(zip(("dq", "dk", "dv"), outputs))
    pack = {refs[name]: value for name, value in tensors.items()}
    capture, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    saved_stream = cudnn.get_stream(cudnn_handle)
    old_mode = torch.cuda.get_sync_debug_mode()
    try:
        for name in ("dq", "dk", "dv"):
            tensors[name].fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        # This graph's first execute happens inside capture. Guard only the
        # launch; entering/exiting Torch's capture scope can synchronize.
        with torch.cuda.graph(capture, stream=stream):
            torch.cuda.set_sync_debug_mode("error")
            try:
                graph.execute(pack, case.workspace, handle=cudnn_handle)
            finally:
                torch.cuda.set_sync_debug_mode(old_mode)
        capture.replay()
        torch.cuda.synchronize()
        dense_cases._check_case(case, tensors)
        tensors["q"].mul_(0.75)
        tensors["do"].mul_(1.25)
        o, lse, _, *expected = _reference64(tensors["q"], tensors["k"], tensors["v"], tensors["do"], case.keep)
        tensors["o"].copy_(o)
        tensors["stats"].copy_(lse.unsqueeze(-1))
        for name in ("dq", "dk", "dv"):
            tensors[name].fill_(float("nan"))
        capture.replay()
        torch.cuda.synchronize()
        dense_cases._check_case(case, tensors, expected)
    finally:
        torch.cuda.set_sync_debug_mode(old_mode)
        capture.reset()
        cudnn.set_stream(cudnn_handle, saved_stream)


def test_standalone_uses_caller_workspace_and_runtime_scale():
    from cudnn.sdpa.bwd.api_dsl_sm107_d128 import SdpaBwdDslSm107D128

    case = _case(dt=torch.bfloat16, causal=False, b=2, h=3, sq=128, sk=256)
    tensors = case.tensors
    roles = ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv")
    operands = tuple(tensors[role] for role in roles)
    api = SdpaBwdDslSm107D128(*operands)
    api.check_support()
    api.compile()
    for scale in (0.125, 0.25):
        o, lse, _, *expected = _reference64(tensors["q"], tensors["k"], tensors["v"], tensors["do"], scale=scale)
        tensors["o"].copy_(o)
        tensors["stats"].copy_(lse.unsqueeze(-1))
        for role in ("dq", "dk", "dv"):
            tensors[role].fill_(float("nan"))
        with pytest.raises(ValueError, match="workspace"):
            api.execute(*operands, workspace=case.workspace[:16], scale_softmax=scale)
        for role in ("dq", "dk", "dv"):
            assert torch.isnan(tensors[role]).all()
        api.execute(*operands, workspace=case.workspace, scale_softmax=scale)
        assert api._prepared.scale == 128**-0.5
        for name, want in zip(("dq", "dk", "dv"), expected):
            _check(name, tensors[name], want, case.dt)


def test_replan_reloads_prepared_artifact(monkeypatch, tmp_path):
    _require_native()
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache

    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    monkeypatch.delenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", raising=False)
    compiled_cache.reset_stats()
    _case(dt=torch.bfloat16, b=1, h=2, sq=97, sk=131)
    assert compiled_cache.stats()["misses"] > 0

    def forbidden(*args, **kwargs):
        raise AssertionError("replanning the native SM107 declaration recompiled")

    monkeypatch.setattr(cute, "compile", forbidden)
    _case(dt=torch.bfloat16, b=1, h=2, sq=97, sk=131, seed=813)
    assert compiled_cache.stats()["hits"] > 0
