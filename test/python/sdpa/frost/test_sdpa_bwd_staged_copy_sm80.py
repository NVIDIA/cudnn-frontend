# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared copies for the existing SM80 backward conversion domain."""

import pytest
import torch

from frost_test_utils import requires_dsl
from sdpa.frost.test_sdpa_bwd_prepared_sm80 import _case, _check

pytestmark = [
    requires_dsl,
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="requires SM80"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("features", [False, True])
def test_staged_execute_uses_prepared_copies(dtype, features, monkeypatch):
    import cutlass.cute as cute

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    case = _case(96, 80, dtype=dtype, features=features)
    _check(case)

    def forbidden(*args, **kwargs):
        raise AssertionError("staged backward rebuilt a tensor or launched a torch data conversion")

    with monkeypatch.context() as guards:
        for name in ("as_strided", "copy_", "zero_", "contiguous", "to"):
            guards.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "empty_like", "zeros", "zeros_like"):
            guards.setattr(torch, name, forbidden)
        guards.setattr(cute, "compile", forbidden)
        guards.setattr(cute.runtime, "make_fake_tensor", forbidden)
        guards.setattr(cute.runtime, "make_ptr", forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            case.graph.execute(case.pack, case.workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    _check(case)


def _adapter(case, *, declare_aux=True):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    bufs = case.bufs
    api = SdpaBwdDslSm80(
        *(bufs[n] for n in ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv")),
        sample_bias=bufs.get("bias"),
        sample_sink=bufs.get("sink"),
        sample_dbias=bufs.get("dbias") if declare_aux else None,
        sample_dsink=bufs.get("dsink") if declare_aux else None,
        has_bias=case.features,
        bias_is_fp32=True,
        is_causal=case.causal,
        seq_q_lens_present=case.features,
        seq_kv_lens_present=case.features,
    )
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    return api, workspace


def _execute(api, bufs, workspace, stream=None):
    api.execute(
        *(bufs[n] for n in ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv")),
        workspace=workspace,
        current_stream=stream,
        seq_q_lens=bufs.get("seq_q"),
        seq_kv_lens=bufs.get("seq_kv"),
        bias_tensor=bufs.get("bias"),
        dbias_tensor=bufs.get("dbias"),
        sink_tensor=bufs.get("sink"),
        dsink_tensor=bufs.get("dsink"),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("declare_aux", [False, True])
def test_auxiliary_casts_strided_outputs_and_replay(dtype, declare_aux):
    case = _case(96, 80, dtype=dtype, features=True)
    bias_owner = torch.full((*case.bufs["dbias"].shape, 2), 11.0, device="cuda", dtype=dtype)
    sink_owner = torch.full((8,), 17.0, device="cuda")
    case.bufs["dbias"] = bias_owner[..., 0]
    case.bufs["dsink"] = sink_owner[::2].view(1, 4, 1, 1)
    api, workspace = _adapter(case, declare_aux=declare_aux)
    _execute(api, case.bufs, workspace)
    _check(case)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _execute(api, case.bufs, workspace)
        for role in case.expected:
            case.bufs[role].fill_(float("nan"))
        workspace.fill_(0xBD)
        graph.replay()
        _check(case)
        assert (bias_owner[..., 1] == 11).all()
        assert (sink_owner[1::2] == 17).all()
    finally:
        graph.reset()
    if not declare_aux:
        case.bufs["dbias"] = torch.empty_like(case.bufs["dbias"], dtype=torch.float32)
        _execute(api, case.bufs, workspace)
        _check(case)


@pytest.mark.parametrize("role", ["bias", "sink", "dbias", "dsink"])
def test_invalid_auxiliary_binding_precedes_gather(role, monkeypatch):
    from cudnn.sdpa.bwd import staged_sm80

    case = _case(96, 80, features=True)
    api, workspace = _adapter(case)
    bufs = dict(case.bufs)
    if role in ("dbias", "dsink"):
        bufs[role] = bufs[role].flatten()[:1].expand(bufs[role].shape)
    else:
        bufs[role] = bufs[role].to(torch.int32)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid auxiliary binding reached a prepared gather")

    monkeypatch.setattr(staged_sm80, "_copy", forbidden)
    with pytest.raises(ValueError, match=role):
        _execute(api, bufs, workspace)


def test_packed_copy_capacity_is_not_a_compile_key(tmp_path, monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.bwd.kernels.sm80 import prepared_host
    from cudnn.sdpa.fwd.kernels.sm80.staged_copy import compile_gather
    from sdpa.frost.test_sdpa_bwd_thd_sm80 import _run_graph

    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    compile_gather.cache_clear()
    prepared_host._compile_thd_artifact.cache_clear()
    _run_graph((33, 17), (41, 25), h=4, hkv=2, d=96, d_v=80, stats_layout="token_major")

    def forbidden(*args, **kwargs):
        raise AssertionError("packed capacities leaked into the staging compile key")

    monkeypatch.setattr(cute, "compile", forbidden)
    compile_gather.cache_clear()
    prepared_host._compile_thd_artifact.cache_clear()
    _run_graph((33, 9), (41, 11), h=4, hkv=2, d=96, d_v=80, stats_layout="token_major")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_auxiliary_cast_matches_torch_rounding(dtype):
    from cudnn.frost.compiled_cache import positional_entry
    from cudnn.sdpa.bwd.kernels.sm80.staged_copy import compile_cast

    # Halfway ties on either side of one, subnormals, signed zero and infinities.
    values = [0.0, -0.0, 1.0, 1.0 + 2**-11, 1.0 + 3 * 2**-11, 1.0 + 2**-8, 1.0 + 3 * 2**-8, 2**-25, 3 * 2**-25, 2**-134, 3 * 2**-134, float("inf")]
    src = torch.tensor(values + [-x for x in values], device="cuda", dtype=torch.float32)
    owner = torch.full((src.numel(), 2), 13.0, device="cuda", dtype=dtype)
    out = owner[:, 0]
    fn = positional_entry(compile_cast((1, 1, 1, src.numel()), str(dtype).split(".")[-1]))
    fn(src.data_ptr(), out.data_ptr(), (0, 0, 0, 2), torch.cuda.current_stream().cuda_stream)
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    torch.testing.assert_close(out.contiguous().view(bits), src.to(dtype).view(bits), atol=0, rtol=0)
    assert (owner[:, 1] == 13).all()


@pytest.mark.parametrize("role", ["q", "dq", "dv"])
@pytest.mark.parametrize("product", [False, True])
def test_dense_copies_keep_physical_wide_strides(role, product):
    case = _case(96, 80, wide=role, product=product, axis=0, sq=17, skv=33)
    assert (case.bufs[role].shape[0] - 1) * case.bufs[role].stride(0) > 2**32
    _check(case)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            case.graph.execute(case.pack, case.workspace)
        for name in case.expected:
            case.bufs[name].fill_(float("nan"))
        case.workspace.fill_(0xBD)
        graph.replay()
        _check(case)
    finally:
        graph.reset()


def test_staged_copies_follow_handle_stream():
    import cudnn

    case = _case(96, 80, features=True)
    stream, other = torch.cuda.Stream(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    other.wait_stream(torch.cuda.current_stream())
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(other):
            case.graph.execute(case.pack, case.workspace, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        _check(case)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            with torch.cuda.stream(other):
                case.graph.execute(case.pack, case.workspace, handle=handle)
        for role in case.expected:
            case.bufs[role].fill_(float("nan"))
        case.workspace.fill_(0xBD)
        capture.replay()
        _check(case)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


def test_staged_stats_preserves_declared_storage_view():
    case = _case(96, 80)
    api, workspace = _adapter(case)
    bufs = dict(case.bufs)
    # The old standalone contract interprets Stats through the declared strides,
    # even if the runtime container is a flat view into a larger allocation.
    bufs["stats"] = bufs["stats"].as_strided((bufs["stats"].numel(),), (1,))
    _execute(api, bufs, workspace)
    _check(case)


@pytest.mark.parametrize("overlap", [False, True])
def test_ordered_gradient_copy_back(overlap, monkeypatch):
    from cudnn.sdpa.bwd import staged_sm80

    case = _case(96, 96, hk=4)
    if overlap:
        case.bufs["dk"] = case.bufs["dq"]
        case.bufs["dv"] = case.bufs["dq"]
    else:
        b, h, seq, d = case.bufs["dq"].shape
        owner = torch.empty((b, seq, h, 3, d), device="cuda", dtype=case.dtype)
        for i, role in enumerate(("dq", "dk", "dv")):
            case.bufs[role] = owner[..., i, :].transpose(1, 2)
    api, workspace = _adapter(case)
    calls = []
    original = staged_sm80._copy

    def record(entry, frame, stream):
        calls.append(len(frame[0]))
        return original(entry, frame, stream)

    monkeypatch.setattr(staged_sm80, "_copy", record)
    _execute(api, case.bufs, workspace)
    assert calls[-3:] == [1, 1, 1], "potentially overlapping gradient scatters must preserve copy-back ordering"
    if overlap:
        torch.testing.assert_close(case.bufs["dv"].float(), case.expected["dv"].float(), atol=0.03, rtol=0.03)
    else:
        _check(case)


@pytest.mark.parametrize("explicit_stream", [False, True])
def test_staged_execution_preserves_other_current_device(explicit_stream):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with torch.cuda.device(0):
        case = _case(96, 80)
        api, workspace = _adapter(case)
        launch = torch.cuda.Stream()
        launch.wait_stream(torch.cuda.current_stream())
        with torch.cuda.device(1):
            _execute(api, case.bufs, workspace, launch.cuda_stream if explicit_stream else None)
            assert torch.cuda.current_device() == 1, "staged execution leaked Q's device into the caller"
        launch.synchronize()
        _check(case)
