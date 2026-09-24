# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared backend transport: immutable geometry and invocation-local pointers."""

from array import array
from concurrent.futures import ThreadPoolExecutor
import gc

import cudnn
import pytest
import torch

from cudnn.engines.base import PlanConfig
from cudnn.engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID, PYTHON_ENGINE_ID_BASE, is_backend_engine

pytestmark = pytest.mark.L0

_UIDS = (31, 7, 19)  # Deliberately different from creation order and sorted order.
_MAX_M, _MAX_N, _MAX_K = 256, 256, 128


def test_prepare_resolves_ranked_index_and_declines_python_plans(monkeypatch):
    graph = cudnn.pygraph()
    graph._is_built = graph._planning_done = True
    graph._lowered_graph = object()
    graph._plans = [PlanConfig(PYTHON_ENGINE_ID_BASE, {}), PlanConfig(1, {}, cpp_index=7)]
    calls = []
    monkeypatch.setattr(cudnn._pybind_module, "_PreparedBackendExecution", lambda *args: calls.append(args) or "prepared")
    # This tests index translation, not any particular performance ranking.
    assert graph._prepare_backend_execution(index=1) == "prepared"
    assert calls[0][2] == 7
    with pytest.raises(NotImplementedError, match="concrete native backend"):
        graph._prepare_backend_execution(index=0)
    graph._plans = [PlanConfig(BACKEND_HEURISTIC_ENGINE_ID, {})]
    with pytest.raises(NotImplementedError, match="concrete native backend"):
        graph._prepare_backend_execution()
    graph._is_built = False
    with pytest.raises(RuntimeError, match="Build the graph"):
        graph._prepare_backend_execution()


@pytest.fixture
def backend_graph(cudnn_handle):
    if cudnn.backend_version() < 92300:
        pytest.skip("Override workspace queries require cuDNN 9.23 or later")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("The representative BF16 backend graph requires Hopper or later")
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_dynamic_shape_enabled=True,
        is_override_shape_enabled=True,
    )
    a = graph.tensor(name="A", uid=_UIDS[0], dim=[1, _MAX_M, _MAX_K], stride=[_MAX_M * _MAX_K, _MAX_K, 1])
    b = graph.tensor(name="B", uid=_UIDS[1], dim=[1, _MAX_K, _MAX_N], stride=[_MAX_K * _MAX_N, 1, _MAX_K])
    out = graph.matmul(A=a, B=b)
    out.set_output(True).set_uid(_UIDS[2]).set_dim([1, _MAX_M, _MAX_N]).set_stride([_MAX_M * _MAX_N, _MAX_N, 1])
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    for index, plan in enumerate(graph.plans):
        if not is_backend_engine(plan.engine_id):
            continue
        graph.select_plan(index)
        try:
            graph.build_plans()
        except cudnn.cudnnGraphNotSupportedError:
            continue
        assert graph.selected_engine is None
        return graph
    pytest.skip("No native backend plan supports this BF16 matmul")


def _geometry(m=128, n=192, k=64):
    return (
        list(_UIDS),
        [[1, m, k], [1, k, n], [1, m, n]],
        [[_MAX_M * _MAX_K, _MAX_K, 1], [_MAX_N * _MAX_K, 1, _MAX_K], [_MAX_M * _MAX_N, _MAX_N, 1]],
    )


def _buffers(seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randint(-2, 3, (1, _MAX_M, _MAX_K), device="cuda", generator=rng).to(torch.bfloat16)
    b = torch.randint(-2, 3, (1, _MAX_N, _MAX_K), device="cuda", generator=rng).to(torch.bfloat16)
    out = torch.full((1, _MAX_M, _MAX_N), float("nan"), device="cuda", dtype=torch.bfloat16)
    return dict(zip(_UIDS, (a, b, out)))


def _frame(prepared, buffers):
    return array("Q", (buffers[uid].data_ptr() for uid in prepared.uids))


def _workspace(graph, handle, geometry):
    size = graph.get_workspace_size_plan_at_index(graph._plan_index, handle, *geometry)
    return torch.empty(max(size, 1), device="cuda", dtype=torch.uint8)


def _assert_result(buffers, m=128, n=192, k=64):
    a, b, out = (buffers[uid] for uid in _UIDS)
    # Integer-valued BF16 inputs give an exact reference independent of TF32.
    expected = a[:, :m, :k].double() @ b[:, :n, :k].double().transpose(-1, -2)
    torch.testing.assert_close(out[:, :m, :n], expected.to(out.dtype), atol=0, rtol=0)


def test_prepared_rebinds_addresses_and_copies_override_geometry(backend_graph, cudnn_handle):
    geometry = _geometry()
    prepared = backend_graph._prepare_backend_execution(*geometry)
    assert prepared.uids == tuple(sorted(_UIDS))
    assert prepared.uids is prepared.uids
    workspace = _workspace(backend_graph, cudnn_handle, geometry)
    # Mutating the source lists cannot change an already-prepared descriptor.
    geometry[1][0][1] = 1
    geometry[2][0][1] = 1
    buffers = [_buffers(40), _buffers(41)]
    assert buffers[0][_UIDS[0]].data_ptr() != buffers[1][_UIDS[0]].data_ptr()
    for data in buffers:
        prepared.execute(_frame(prepared, data), workspace.data_ptr(), cudnn_handle.backend_handle)
        _assert_result(data)
    # New geometry belongs to a different descriptor; the old one remains valid.
    changed = _geometry(m=64, n=128, k=96)
    second = backend_graph._prepare_backend_execution(*changed)
    second_workspace = _workspace(backend_graph, cudnn_handle, changed)
    data = _buffers(42)
    second.execute(_frame(second, data), second_workspace.data_ptr(), cudnn_handle.backend_handle)
    _assert_result(data, m=64, n=128, k=96)
    buffers[0][_UIDS[2]].fill_(float("nan"))
    prepared.execute(_frame(prepared, buffers[0]), workspace.data_ptr(), cudnn_handle.backend_handle)
    _assert_result(buffers[0])


def test_prepared_without_overrides(backend_graph, cudnn_handle):
    prepared = backend_graph._prepare_backend_execution()
    workspace = torch.empty(max(backend_graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    data = _buffers(43)
    prepared.execute(_frame(prepared, data), workspace.data_ptr(), cudnn_handle.backend_handle)
    _assert_result(data, m=_MAX_M, n=_MAX_N, k=_MAX_K)


def test_prepared_rejects_malformed_descriptors_and_pointer_buffers(backend_graph):
    geometry = _geometry()
    with pytest.raises(ValueError, match="supplied together"):
        backend_graph._prepare_backend_execution(override_uids=geometry[0])
    with pytest.raises(ValueError, match="lengths"):
        backend_graph._prepare_backend_execution(geometry[0], geometry[1][:-1], geometry[2])
    with pytest.raises(ValueError, match="Duplicate"):
        backend_graph._prepare_backend_execution([_UIDS[0]] * 2, [geometry[1][0]] * 2, [geometry[2][0]] * 2)
    with pytest.raises(ValueError, match="ranks"):
        backend_graph._prepare_backend_execution([_UIDS[0]], [[1, 128]], [[128, 1]])
    with pytest.raises(ValueError, match="positive"):
        backend_graph._prepare_backend_execution([_UIDS[0]], [[1, 0, 64]], [geometry[2][0]])
    with pytest.raises(ValueError, match="nonnegative"):
        backend_graph._prepare_backend_execution([_UIDS[0]], [geometry[1][0]], [[8192, -64, 1]])
    prepared = backend_graph._prepare_backend_execution(*geometry)
    count = len(prepared.uids)
    # These must fail on the host: none of the deliberately fake pointers launch.
    with pytest.raises(ValueError, match="number"):
        prepared.execute(array("Q", [1] * (count + 1)))
    for bad in (array("d", [1] * count), array("I", [1] * count), memoryview(array("Q", [1] * (2 * count)))[::2]):
        with pytest.raises(ValueError, match="native unsigned pointer-width"):
            prepared.execute(bad)
    with pytest.raises(ValueError, match="nonnegative"):
        prepared.execute(array("Q", [1] * count), workspace=-1)


def test_prepared_rejects_foreign_endian_and_unaligned_buffers(backend_graph):
    numpy = pytest.importorskip("numpy")
    prepared = backend_graph._prepare_backend_execution(*_geometry())
    count = len(prepared.uids)
    for bad in (
        numpy.ones(count, dtype=numpy.dtype("uint64").newbyteorder("S")),
        memoryview(bytearray(count * 8 + 1))[1:].cast("Q"),
    ):
        with pytest.raises(ValueError, match="native unsigned pointer-width"):
            prepared.execute(bad)


@pytest.mark.parametrize("mutation", ["rebuild", "deserialize", "replace", "filter"])
def test_prepared_rejects_stale_native_graph(backend_graph, mutation):
    prepared = backend_graph._prepare_backend_execution(*_geometry())
    if mutation == "rebuild":
        backend_graph.build_plan_at_index(backend_graph._plan_index)
    elif mutation == "deserialize":
        backend_graph.deserialize(backend_graph.serialize())
    elif mutation == "replace":
        backend_graph._reset_lowered_state()
    else:
        backend_graph.deselect_workspace_greater_than(0)
    with pytest.raises(RuntimeError, match="stale"):
        prepared.execute(array("Q", [1] * len(prepared.uids)))


def test_prepared_capture_survives_new_geometry_and_replays_changed_input(backend_graph, cudnn_handle):
    geometry = _geometry()
    prepared = backend_graph._prepare_backend_execution(*geometry)
    workspace = _workspace(backend_graph, cudnn_handle, geometry)
    buffers = _buffers(50)
    pointers = _frame(prepared, buffers)
    # Handle stream is queried by backend execution on every invocation.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            cudnn.set_stream(cudnn_handle, stream.cuda_stream)
            prepared.execute(pointers, workspace.data_ptr())
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                prepared.execute(pointers, workspace.data_ptr())
        stream.synchronize()
        second = backend_graph._prepare_backend_execution(*_geometry(m=64, n=128, k=96))
        del prepared, pointers
        gc.collect()
        # The old device bindings remain caller-owned through replay, even after
        # an adapter switches to another descriptor and releases the first one.
        with torch.cuda.stream(stream):
            buffers[_UIDS[0]].fill_(1)
            buffers[_UIDS[2]].fill_(float("nan"))
            captured.replay()
            _assert_result(buffers)
        stream.synchronize()
        assert second.uids == tuple(sorted(_UIDS))
    finally:
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)


def test_prepared_thread_local_handles_and_call_frames(backend_graph):
    geometry = _geometry()
    prepared = backend_graph._prepare_backend_execution(*geometry)

    def run(seed):
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        handle = cudnn.create_handle()
        try:
            with torch.cuda.stream(stream):
                cudnn.set_stream(handle, stream.cuda_stream)
                workspace = _workspace(backend_graph, handle, geometry)
                buffers = _buffers(seed)
                with pytest.raises(ValueError, match="thread-local handle"):
                    prepared.execute(_frame(prepared, buffers), workspace.data_ptr())
                for _ in range(4):
                    buffers[_UIDS[2]].fill_(float("nan"))
                    prepared.execute(_frame(prepared, buffers), workspace.data_ptr(), handle.backend_handle)
                    _assert_result(buffers)
            stream.synchronize()
        finally:
            cudnn.destroy_handle(handle)

    with ThreadPoolExecutor(max_workers=2) as executor:
        for task in (executor.submit(run, 60), executor.submit(run, 61)):
            task.result()
