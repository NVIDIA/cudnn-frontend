# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense d128 accumulation selection, prepared workspace and replay contract."""

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl
from test_sdpa_bwd_d128_sm100 import _case, _check_case
from test_sdpa_bwd_dsl_sm107 import _reference64

pytestmark = [pytest.mark.L0, requires_dsl]


def _require_accumulation():
    from cudnn.frost.buffers import cutedsl_state
    from cudnn.sdpa.bwd.d128_contract import ACCUMULATION, select_implementation

    if select_implementation(torch.cuda.get_device_capability(), cutedsl_state()[1], 1, 1, 128, 128, 0.125) != ACCUMULATION:
        pytest.skip("dense d128 accumulation requires SM100 and public CuTeDSL >= 4.8")


def test_implementation_selection_is_plan_time():
    from cudnn.sdpa.bwd.d128_contract import ACCUMULATION, RECOMPUTE, select_implementation

    args = dict(cc=(10, 0), dsl_version=("nvidia-cutlass-dsl", "4.8.0"), batch=2, heads=3, sq=128, skv=256, scale=0.125)
    assert select_implementation(**args) == ACCUMULATION
    for changed in (
        {"cc": (10, 3)},
        {"cc": (10, 7)},
        {"dsl_version": ("nvidia-cutlass-dsl", "4.7.1")},
        {"dsl_version": ("nvidia-cutlass-dsl-internal", "0.3.0")},
        {"dsl_version": None},
        {"sq": 129},
        {"skv": 129},
        {"sq": 1},
        {"batch": 65536},
        {"heads": 65536},
        {"batch": 65535, "heads": 257},
        {"skv": 2**39},
        {"scale": -0.125},
        {"scale": 0.0},
        {"scale": float("nan")},
        {"scale": float("inf")},
        {"scale": 1e-50},
        {"scale": 3e38},
    ):
        assert select_implementation(**(args | changed)) == RECOMPUTE, changed
    assert select_implementation(**(args | {"sq": 256, "skv": 128})) == ACCUMULATION


def test_accumulation_sequence_scalars_fit_int32():
    from cudnn.sdpa.bwd.d128_contract import ACCUMULATION, RECOMPUTE, select_implementation

    args = dict(cc=(10, 0), dsl_version=("nvidia-cutlass-dsl", "4.8.0"), batch=1, heads=1, sq=102400, skv=102400, scale=0.125)
    for field in ("sq", "skv"):
        assert select_implementation(**(args | {field: 2**31 - 128})) == ACCUMULATION
        assert select_implementation(**(args | {field: 2**31})) == RECOMPUTE
        assert select_implementation(**(args | {field: 2**31 + 128})) == RECOMPUTE


def test_accumulation_scheduler_divisor_fits_int32():
    from cudnn.sdpa.bwd.d128_contract import ACCUMULATION, RECOMPUTE, select_implementation

    args = dict(cc=(10, 0), dsl_version=("nvidia-cutlass-dsl", "4.8.0"), batch=1, heads=1, sq=128, scale=0.125)
    assert select_implementation(**args, skv=(2**22 - 1) * 128) == ACCUMULATION
    for skv in (2**29, 2**30):
        assert select_implementation(**args, skv=skv) == RECOMPUTE
    assert select_implementation(**(args | {"sq": 102400}), skv=2**31 - 128) == ACCUMULATION


@pytest.mark.parametrize(
    "dt,causal,sq,sk",
    [
        (torch.bfloat16, False, 128, 128),
        (torch.bfloat16, True, 128, 256),
        (torch.float16, False, 256, 128),
        (torch.float16, True, 128, 128),
    ],
)
def test_accumulation_matches_fp64(dt, causal, sq, sk):
    _require_accumulation()
    _case(dt=dt, causal=causal, b=2, h=3, sq=sq, sk=sk, accumulation=True)


@pytest.mark.parametrize("dt,causal", [(torch.bfloat16, False), (torch.float16, True)], ids=["bf16-unmasked", "fp16-causal"])
def test_negative_scale_preserves_recompute_route(dt, causal):
    _require_accumulation()
    _case(dt=dt, causal=causal, sq=128, sk=256, scale=-0.125)


@pytest.mark.parametrize("dt,causal", [(torch.bfloat16, True)])
def test_three_launches_without_allocation_or_sync(dt, causal):
    _require_accumulation()
    case = _case(dt=dt, causal=causal, sq=128, sk=256, accumulation=True, check_build=True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.set_sync_debug_mode("error")
    try:
        case.graph.execute(case.pack, case.workspace)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_allocated() == before
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        case.graph.execute(case.pack, case.workspace)
    names = [event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    assert len(names) == 3, names
    for name, expected in zip(names, ("dot_do_o_kernel", "bprop_d128_accum_kernel", "cast_dq_d128_kernel")):
        assert expected in name, names
    _check_case(case)


@pytest.mark.parametrize("dt,causal", [(torch.bfloat16, True)])
def test_workspace_alignment_poison_and_bounds(dt, causal):
    _require_accumulation()
    case = _case(dt=dt, causal=causal, sq=128, sk=256, accumulation=True)
    required = case.graph.get_workspace_size()
    raw = torch.empty(required + 320, dtype=torch.uint8, device="cuda")
    for residue in range(0, 256, 16):
        raw.fill_(0xA5)
        offset = (residue - raw.data_ptr()) % 256
        workspace = raw.narrow(0, offset, required)
        for name in ("dq", "dk", "dv"):
            case.tensors[name].fill_(float("nan"))
        case.graph.execute(case.pack, workspace)
        _check_case(case)
        payload_offset = ((workspace.data_ptr() + 255) & -256) - raw.data_ptr()
        payload_bytes = required - 240
        assert bool(torch.all(raw[:payload_offset] == 0xA5))
        assert bool(torch.all(raw[payload_offset + payload_bytes :] == 0xA5))
    for bad in (None, case.workspace[:-1]):
        with pytest.raises(ValueError, match="workspace"):
            case.graph.execute(case.pack, bad)


@pytest.mark.parametrize("dt,causal", [(torch.bfloat16, False), (torch.float16, True)], ids=["bf16-unmasked", "fp16-causal"])
def test_accumulation_rebind_and_capture(dt, causal):
    _require_accumulation()
    case = _case(dt=dt, causal=causal, b=2, h=3, sq=128, sk=256, accumulation=True)
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
        case.workspace.fill_(0xA5)
        for name in ("dq", "dk", "dv"):
            tensors[name].fill_(float("nan"))
        return expected

    try:
        expected = refresh()
        stream.wait_stream(torch.cuda.current_stream())
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            case.graph.execute(pack, case.workspace, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        _check_case(case, tensors, expected)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            with torch.cuda.stream(other):
                case.graph.execute(pack, case.workspace, handle=handle)
        expected = refresh()
        capture.replay()
        torch.cuda.synchronize()
        _check_case(case, tensors, expected)
        case.graph.execute(case.pack, case.workspace)
        _check_case(case)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("dt,causal", [(torch.bfloat16, False)])
def test_accumulation_reloads_artifact_without_compilation(monkeypatch, tmp_path, dt, causal):
    _require_accumulation()
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache

    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    monkeypatch.delenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", raising=False)
    compiled_cache.reset_stats()
    _case(dt=dt, causal=causal, sq=128, sk=256, accumulation=True)
    first = compiled_cache.stats()
    assert first["misses"] > 0 and first["hits"] == 0, first

    def forbidden(*args, **kwargs):
        raise AssertionError("rebuilding the declaration compiled instead of reloading its artifact")

    monkeypatch.setattr(cute, "compile", forbidden)
    case = _case(dt=dt, causal=causal, sq=128, sk=256, accumulation=True, seed=730)
    assert compiled_cache.stats()["hits"] > first["hits"]
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            case.graph.execute(case.pack, case.workspace)
        tensors = case.tensors
        tensors["q"].mul_(0.75)
        tensors["do"].mul_(1.25)
        o, lse, _, *expected = _reference64(tensors["q"], tensors["k"], tensors["v"], tensors["do"], case.keep)
        tensors["o"].copy_(o)
        tensors["stats"].copy_(lse.unsqueeze(-1))
        case.workspace.fill_(0xA5)
        for name in ("dq", "dk", "dv"):
            tensors[name].fill_(float("nan"))
        capture.replay()
        torch.cuda.synchronize()
        _check_case(case, expected=expected)
    finally:
        capture.reset()


@pytest.mark.parametrize("dt,causal", [(torch.float16, True)])
def test_accumulation_standalone_scale_and_current_stream(dt, causal, monkeypatch):
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128

    _require_accumulation()
    case = _case(dt=dt, causal=causal, b=2, h=3, sq=128, sk=256, accumulation=True)
    tensors = {name: value.clone() for name, value in case.tensors.items()}
    api = SdpaBwdDslSm100D128(**{"sample_" + name: value for name, value in tensors.items()}, is_causal=causal)
    api.check_support()
    api.compile()
    assert api._prepared.artifact.implementation == "accumulation"
    assert api.scratch_workspace_bytes() == case.workspace.numel()
    args = {name + "_tensor": value for name, value in tensors.items()}
    for name in ("dq", "dk", "dv"):
        tensors[name].fill_(float("nan"))
    api.execute(**args, workspace=case.workspace)
    _check_case(case, tensors)

    # The same prepared host must bind a changed positive scale and current data.
    scale = 0.125
    tensors["q"].mul_(0.75)
    tensors["do"].mul_(1.25)
    o, lse, _, *expected = _reference64(tensors["q"], tensors["k"], tensors["v"], tensors["do"], case.keep, scale=scale)
    tensors["o"].copy_(o)
    tensors["stats"].copy_(lse.unsqueeze(-1))
    case.workspace.fill_(0xA5)
    for name in ("dq", "dk", "dv"):
        tensors[name].fill_(float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    api.execute(**args, workspace=case.workspace, scale_softmax=scale, current_stream=stream.cuda_stream)
    torch.cuda.current_stream().wait_stream(stream)
    _check_case(case, tensors, expected)

    # Keep real tensor binding and the production guard, but trip before an
    # invalid scale can enter the compiled host even if validation regresses.
    entered = []

    def forbidden(*frame):
        entered.append(frame)
        raise AssertionError("invalid standalone scale reached the compiled host")

    with monkeypatch.context() as patch:
        patch.setattr(api._prepared.fn, "entry", forbidden)
        for invalid in (0.0, -0.125, float("nan"), float("inf"), 1e-50, 3e38):
            with pytest.raises(ValueError, match="scale"):
                api.execute(**args, workspace=case.workspace, scale_softmax=invalid)
        assert not entered
    api.execute(**args, workspace=case.workspace, scale_softmax=scale)
    _check_case(case, tensors, expected)
