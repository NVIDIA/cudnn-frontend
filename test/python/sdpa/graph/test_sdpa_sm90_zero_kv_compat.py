# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Old-backend empty-KV correctness, including changing lengths during replay."""

import math

import cudnn
import pytest
import torch

pytestmark = pytest.mark.L0


def _graph(dtype, dqk, dv, stats, handle, pin_engine=None):
    b, hq, hk, sq, skv = 3, 4, 2, 33, 129
    generator = torch.Generator(device="cuda").manual_seed(1741)
    q = torch.randn(b, hq, sq, dqk, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(b, hk, skv, dqk, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(b, hk, skv, dv, device="cuda", dtype=dtype, generator=generator)
    o = torch.empty(b, hq, sq, dv, device="cuda", dtype=dtype)
    lse = torch.empty(b, hq, sq, 1, device="cuda", dtype=torch.float32) if stats else None
    q_lens = torch.tensor([17, 29, 33], device="cuda", dtype=torch.int32).view(b, 1, 1, 1)
    kv_lens = torch.tensor([0, 97, 129], device="cuda", dtype=torch.int32).view(b, 1, 1, 1)
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=dt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    Q, K, V, QL, KL = [graph.tensor_like(t) for t in (q, k, v, q_lens, kv_lens)]
    O, S = graph.sdpa(q=Q, k=K, v=V, attn_scale=1 / math.sqrt(dqk), use_padding_mask=True, seq_len_q=QL, seq_len_kv=KL, generate_stats=stats)
    O.set_output(True).set_dim(o.shape).set_stride(o.stride())
    pack = {Q: q, K: k, V: v, QL: q_lens, KL: kv_lens, O: o}
    if stats:
        S.set_output(True).set_dim(lse.shape).set_stride(lse.stride()).set_data_type(cudnn.data_type.FLOAT)
        pack[S] = lse
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    if pin_engine is not None:
        records = [graph.get_engine_and_knobs_at_index(i) for i in range(graph.get_execution_plan_count())]
        knobs = next((knobs for engine, knobs in records if engine == pin_engine), None)
        if knobs is None:
            pytest.skip(f"backend did not offer engine {pin_engine} for the boundary detector")
        graph.create_execution_plan(pin_engine, knobs)
        graph.select_plan(graph.get_execution_plan_count() - 1)
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
    return graph, pack, workspace, (q, k, v, o, lse, q_lens, kv_lens)


def _check(values, lengths):
    q, k, v, o, lse, _, _ = values
    tol = 0.015 if q.dtype == torch.bfloat16 else 0.002
    for batch, (nq, nk) in enumerate(zip((17, 29, 33), lengths)):
        actual = o[batch, :, :nq]
        if nk == 0:
            torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
            if lse is not None:
                assert torch.isneginf(lse[batch, :, :nq]).all()
            continue
        scores = q[batch, :, :nq].double() @ k[batch, :, :nk].double().repeat_interleave(2, 0).transpose(-1, -2)
        scores *= 1 / math.sqrt(q.shape[-1])
        reference = scores.softmax(-1) @ v[batch, :, :nk].double().repeat_interleave(2, 0)
        torch.testing.assert_close(actual.double(), reference, atol=tol, rtol=tol)
        if lse is not None:
            torch.testing.assert_close(lse[batch, :, :nq, 0].double(), scores.logsumexp(-1), atol=0.005, rtol=0.005)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(64, 64), (128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("stats", [False, True])
def test_default_zero_kv_and_changed_length_capture(dtype, dims, stats, cudnn_handle, monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper backend compatibility")
    if cudnn.backend_version() < 92400:
        pytest.skip("qualification starts with cuDNN 9.24")
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    original_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(original_stream)
    captured = None
    try:
        with torch.cuda.stream(stream):
            cudnn.set_stream(handle=cudnn_handle, stream=stream.cuda_stream)
            graph, pack, workspace, values = _graph(dtype, *dims, stats, cudnn_handle)
            graph.execute(pack, workspace)
            stream.synchronize()
            _check(values, (0, 97, 129))
            if cudnn.backend_version() < 92500:
                assert all(graph.get_engine_and_knobs_at_index(i)[0] != 9 for i in range(graph.get_execution_plan_count()))
                with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="zero.*KV"):
                    graph.create_execution_plan(9, {})
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                graph.execute(pack, workspace)
            for lengths in ((129, 0, 73), (0, 0, 0), (65, 97, 129)):
                values[0].mul_(0.75)
                values[2].add_(0.125)
                values[6].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32).view(3, 1, 1, 1))
                for execute in (lambda: graph.execute(pack, workspace), captured.replay):
                    values[3].fill_(float("nan"))
                    if values[4] is not None:
                        values[4].fill_(float("nan"))
                    torch.cuda.set_sync_debug_mode("error")
                    try:
                        execute()
                    finally:
                        torch.cuda.set_sync_debug_mode("default")
                    stream.synchronize()
                    _check(values, lengths)
    finally:
        if captured is not None:
            captured.reset()
        original_stream.wait_stream(stream)
        cudnn.set_stream(handle=cudnn_handle, stream=original_stream.cuda_stream)


@pytest.mark.parametrize("layout", ["ragged", "paged"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stats", [False, True])
def test_zero_kv_layouts_use_safe_default(layout, dtype, stats, cudnn_handle, request, monkeypatch):
    from sdpa import fp16
    from sdpa.random_config import ExecConfig

    if torch.cuda.get_device_capability() != (9, 0) or cudnn.backend_version() < 92400:
        pytest.skip("Hopper compatibility from cuDNN 9.24")
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    cfg = ExecConfig(
        data_type=dtype,
        rng_geom_seed=1741,
        rng_data_seed=1741,
        is_padding=True,
        is_ragged=layout == "ragged",
        is_paged=layout == "paged",
        batches=3,
        h_q=4,
        h_k=2,
        h_v=2,
        s_q=129,
        s_kv=257,
        d_qk=128,
        d_v=128,
        seq_len_q=[97, 0, 129],
        seq_len_kv=[0, 193, 257],
        block_size=64 if layout == "paged" else None,
        with_ragged_token_gap=False,
        with_ragged_head_gap=False,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        fwd_stats=stats,
    )
    cfg.fill_derived_fields()
    # Reuse the harness's allocation and independent reference, but build this
    # explicit compatibility case directly: the random suite's legacy validator
    # waives every empty sequence below 9.25, before any engine can be selected.
    allocs, tensors, _, _ = fp16.allocate_tensors(cfg, torch.Generator(device="cuda").manual_seed(1741), perf=False)
    try:

        def require_safe_plan(graph):
            if cudnn.backend_version() < 92500:
                assert all(graph.get_engine_and_knobs_at_index(i)[0] != 9 for i in range(graph.get_execution_plan_count()))

        graph, pack = fp16.create_forward_graph(cfg, tensors, cudnn_handle, plan_hook=require_safe_plan)
        fp16.execute_graph(graph, pack, allocs, tensors, cudnn_handle, request, label="Forward")
        fp16.compute_and_compare_reference(cfg, allocs, tensors, request.config.getoption("--diffs"))
    finally:
        fp16.cleanup_tensors(allocs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stats", [False, True])
def test_zero_kv_backend_fix_boundary_is_current(dtype, stats, cudnn_handle, monkeypatch):
    from cudnn.sdpa.fwd import backend_guard

    if torch.cuda.get_device_capability() != (9, 0) or cudnn.backend_version() < 92400:
        pytest.skip("Hopper compatibility from cuDNN 9.24")
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    # Execute the actual installed backend with just this guard lifted. A
    # future corrected 9.24 build must fail this detector, prompting a more
    # precise boundary instead of silently keeping the compatibility fallback.
    monkeypatch.setattr(backend_guard, "SM90_ZERO_KV_WRONG_O_FIXED_IN", 0)
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)
    graph, pack, workspace, values = _graph(dtype, 128, 128, stats, cudnn_handle, pin_engine=9)
    values[3].fill_(123)
    if stats:
        values[4].fill_(123)
    graph.execute(pack, workspace)
    torch.cuda.synchronize()
    if cudnn.backend_version() < 92500:
        assert (values[3][0, :, :17] == 123).all(), "engine 9 no longer leaves zero-KV O unwritten; update the guard boundary"
        if stats:
            assert (values[4][0, :, :17] == 123).all()
    else:
        _check(values, (0, 97, 129))
