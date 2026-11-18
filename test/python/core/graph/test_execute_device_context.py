# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A graph executes on its handle's device without changing the caller's context."""

import threading

import cudnn
import pytest
import torch

from cudnn.engines.engine_ids import FAMILY_BLOCK, FROST_SDPA_FWD_ID_BASE, is_backend_engine

pytestmark = pytest.mark.L0


def _current():
    from cuda.bindings import driver

    err, context = driver.cuCtxGetCurrent()
    assert int(err) == 0
    return int(context)


@pytest.fixture(params=["backend", "frost"])
def attention(request, monkeypatch):
    cc = torch.cuda.get_device_capability()
    if cudnn.backend_version() < 92400 or cc[0] < 8:
        pytest.skip("the representative decode graph requires Ampere+ and cuDNN 9.24+")
    if request.param == "frost":
        if cc[0] == 8 and cc != (8, 0):
            pytest.skip("the SM80 FROST implementation serves cc 8.0 only")
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

        installed, version = cutedsl_state()
        if not installed or cutedsl_too_old(version):
            pytest.skip("requires a supported CuTe DSL")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    handle = cudnn.create_handle()
    try:
        graph = cudnn.pygraph(
            handle=handle, io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
        )
        q = torch.randn(2, 4, 1, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, 4, 64, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        tq, tk, tv = [graph.tensor_like(x) for x in (q, k, v)]
        to, ts = graph.sdpa(q=tq, k=tk, v=tv, generate_stats=True, attn_scale=128**-0.5)
        o = torch.empty_like(q)
        stats = torch.empty(2, 4, 1, 1, device="cuda", dtype=torch.float32)
        to.set_output(True).set_dim(o.shape).set_stride(o.stride())
        ts.set_output(True).set_dim(stats.shape).set_stride(stats.stride()).set_data_type(cudnn.data_type.FLOAT)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        candidates = [
            i
            for i, cfg in enumerate(graph.plans)
            if (
                is_backend_engine(cfg.engine_id)
                if request.param == "backend"
                else FROST_SDPA_FWD_ID_BASE <= cfg.engine_id < FROST_SDPA_FWD_ID_BASE + FAMILY_BLOCK
            )
        ]
        for index in candidates:
            graph.select_plan(index)
            try:
                graph.build_plans()
            except (cudnn.cudnnGraphNotSupportedError, NotImplementedError):
                continue
            break
        else:
            pytest.fail(f"No {request.param} plan built for the supported decode graph")
        assert (graph.selected_engine is None) == (request.param == "backend")
        workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
        pack = dict(zip((tq, tk, tv, to, ts), (q, k, v, o, stats)))
        graph.execute(pack, workspace)
        torch.cuda.synchronize()
        yield graph, handle, pack, workspace, (q, k, v, o, stats)
    finally:
        cudnn.destroy_handle(handle)


def _check(values):
    q, k, v, o, stats = values
    scores = q.double() @ k.double().transpose(-2, -1) * 128**-0.5
    torch.testing.assert_close(o.double(), scores.softmax(-1) @ v.double(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(stats.double(), scores.logsumexp(-1, keepdim=True), atol=2e-3, rtol=2e-3)


def _execute(graph, handle, pack, workspace, entry):
    if entry == "mapping":
        # Exercise the graph's stored handle, too.
        return graph.execute(pack, workspace)
    buffers, uids = list(pack.values()), [tensor.get_uid() for tensor in pack]
    if entry == "indexed":
        return graph.execute_plan_at_index(buffers, workspace, index=graph._plan_index, handle=handle, tensor_uids=uids)
    return graph.execute(buffers, workspace, handle=handle, tensor_uids=uids)


@pytest.mark.parametrize("entry", ["mapping", "ordered", "indexed"])
@pytest.mark.parametrize("default_stream", [False, True])
def test_execute_preserves_foreign_device_and_replay(attention, entry, default_stream):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible GPUs")
    graph, handle, pack, workspace, values = attention
    target = handle.device.ordinal
    foreign = next(i for i in range(torch.cuda.device_count()) if i != target)
    values[0].mul_(0.75)
    values[3].fill_(float("nan"))
    values[4].fill_(float("nan"))
    stream = torch.cuda.default_stream(target) if default_stream else torch.cuda.Stream(device=target)
    stream.wait_stream(torch.cuda.current_stream(target))
    cudnn.set_stream(handle, stream.cuda_stream)

    def run():
        with torch.cuda.device(foreign):
            previous = _current()
            _execute(graph, handle, pack, workspace, entry)
            assert _current() == previous
            assert torch.cuda.current_device() == foreign

    run()
    stream.synchronize()
    _check(values)
    # Invalid bindings must also restore the foreign context before raising.
    with torch.cuda.device(foreign):
        previous = _current()
        with pytest.raises(ValueError, match="missing"):
            graph.execute(list(pack.values())[:-1], workspace, handle=handle, tensor_uids=[tensor.get_uid() for tensor in pack][:-1])
        assert _current() == previous

    if not default_stream:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            run()
        values[0].mul_(0.5)
        values[3].fill_(float("nan"))
        values[4].fill_(float("nan"))
        captured.replay()
        torch.cuda.synchronize(target)
        _check(values)


@pytest.mark.parametrize("python_fallback", [False, True])
def test_execute_restores_cold_worker_context(attention, monkeypatch, python_fallback):
    if python_fallback:
        from cudnn import _device

        monkeypatch.setattr(_device, "_native_context_matches", None)
        monkeypatch.setattr(_device, "_native_context", None)
    graph, handle, pack, workspace, values = attention
    result = {}

    def worker():
        try:
            assert _current() == 0
            _execute(graph, handle, pack, workspace, "ordered")
            assert _current() == 0
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive(), "graph.execute did not return"
    if "error" in result:
        raise result["error"]
    torch.cuda.synchronize(handle.device.ordinal)
    _check(values)
