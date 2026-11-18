# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise replay and batch isolation for supported custom-reduction graphs.

The arg helpers expose a value/index tuple, but these graphs consume the index
projection. Custom reducer callbacks in general make no determinism promise.
There is no custom-reduction backward entry point.
"""

import pytest

import cudnn
from cudnn.engines.engine_ids import is_backend_engine
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

pytestmark = pytest.mark.L0

_VALUE_TYPES = {
    "fp32": ("float32", cudnn.data_type.FLOAT),
    "bf16": ("bfloat16", cudnn.data_type.BFLOAT16),
    "fp16": ("float16", cudnn.data_type.HALF),
}
_EXTENT = 257  # Cross a 256-element boundary with a partial final tile.


def _build_native(graph):
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    backend_plans = [i for i in range(graph.get_execution_plan_count()) if is_backend_engine(graph.get_engine_and_knobs_at_index(i)[0])]
    assert backend_plans, "custom-reduction graph has no native cuDNN backend plan"
    graph.select_plan(backend_plans[0])  # A strict pin; no Python-engine fallback.
    graph.check_support()
    graph.build_plans()


def _execute_on_handle_stream(torch, handle, graph, bindings, workspace):
    # The fixture's handle has its own stream. Order all input/sentinel/snapshot
    # production before it, then its output before current-stream comparisons.
    # Waiting only on these streams keeps assert_replays' noise stream runnable.
    producer = torch.cuda.current_stream()
    execution = torch.cuda.ExternalStream(cudnn.get_stream(handle), device=producer.device)
    execution.wait_stream(producer)
    try:
        with torch.cuda.stream(execution):
            graph.execute(bindings, workspace, handle=handle)
    finally:
        producer.wait_stream(execution)
    # The handle was never rebound, including when graph.execute raises.


def _require_backend(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")
    torch = pytest.importorskip("torch")
    if cudnn._pybind_module._compiled_cudnn_version < 92900 or cudnn.backend_version() < 92900:
        pytest.skip("Custom reduction requires cuDNN 9.29 headers and runtime")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("This suite targets the SM100/SM103 TensorIR MemBound rows")
    return torch


def _inputs(torch, dtype):
    positions = torch.arange(_EXTENT, dtype=torch.int32)
    target = (((positions * 37) % 31) - 15).to(dtype)
    target[5] = target[253] = 20
    target[7] = target[251] = -20
    neighbors = (((torch.arange(32 * _EXTENT).view(32, _EXTENT) * 13) % 39) - 19).to(dtype)
    changed = neighbors.clone()
    changed[3] = -changed[3]
    return {
        "alone": (target.unsqueeze(0).contiguous(), (0,)),
        "packed": (torch.cat((neighbors[:16], target.unsqueeze(0), neighbors[16:])).contiguous(), (16,)),
        "reordered": (torch.cat((target.unsqueeze(0), neighbors)).contiguous(), (0,)),
        "duplicated": (torch.cat((target.unsqueeze(0), neighbors, target.unsqueeze(0))).contiguous(), (0, 33)),
        "changed_neighbor": (torch.cat((changed[:16], target.unsqueeze(0), changed[16:])).contiguous(), (16,)),
    }


def _arg_plan(handle, torch, dtype, value_type, kind, rows):
    graph = cudnn.pygraph(
        handle=handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    shape, stride = [rows, _EXTENT], [_EXTENT, 1]
    x = graph.tensor(dim=shape, stride=stride, data_type=value_type, name="X")
    index_input = graph.tensor(dim=shape, stride=stride, data_type=cudnn.data_type.INT32, name="IDX")
    _, index = graph.custom_reduction(
        [x, index_input],
        reduction_dims=[1],
        name=f"determinism_{kind}_{dtype}",
        **getattr(cudnn.reduction_modifiers, kind)(value_type, rule="torch"),
    )
    index.set_output(True)
    _build_native(graph)
    return graph, x, index_input, index


def _add_plan(handle, rows):
    def add_body(child, lhs, rhs):
        return [child.add(lhs[0], rhs[0], compute_data_type=cudnn.data_type.FLOAT, name="add")]

    graph = cudnn.pygraph(
        handle=handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x = graph.tensor(dim=[rows, _EXTENT], stride=[_EXTENT, 1], data_type=cudnn.data_type.FLOAT, name="X")
    y = graph.custom_reduction(
        [x],
        add_body,
        "test.python.custom_reduction.determinism.direct_add.v1",
        [1],
        [cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)],
    )[0]
    y.set_output(True)
    _build_native(graph)
    return graph, x, y


def _check_scenarios(torch, scenarios, execute):
    target, _ = scenarios["alone"]
    alone = assert_replays(lambda: execute(target), repeats=3)
    for name in ("packed", "reordered", "duplicated", "changed_neighbor"):
        values, target_rows = scenarios[name]
        outputs = snapshot(execute(values))
        for row in target_rows:
            assert_bitwise({key: value[row : row + 1] for key, value in outputs.items()}, alone, f"{name}[{row}]")
    # Every supported dtype/body gets sixteen rows per SM, with the
    # identical target retained at the midpoint of a freshly built batch. This
    # exercises larger scheduling geometry, without assuming rows per CTA.
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    pool = scenarios["packed"][0][:16]
    side_rows = 8 * sm_count
    neighbors = pool.repeat((side_rows + len(pool) - 1) // len(pool), 1)[:side_rows]
    large = torch.cat((neighbors, target, neighbors.flip(0))).contiguous()
    assert large.shape[0] == 16 * sm_count + 1
    large_replay = assert_replays(lambda: execute(large), repeats=3)
    assert_bitwise({key: value[side_rows : side_rows + 1] for key, value in large_replay.items()}, alone, "large-batch target")
    return alone


@pytest.mark.parametrize("dtype_id", ["fp32", "bf16", "fp16"], ids=["fprop-fp32-k257", "fprop-bf16-k257", "fprop-fp16-k257"])
@pytest.mark.parametrize("kind", ["argmax", "argmin"])
def test_custom_reduction_arg_index_replay_and_batch_invariance(cudnn_handle, dtype_id, kind):
    torch = _require_backend(cudnn_handle)
    dtype_name, value_type = _VALUE_TYPES[dtype_id]
    scenarios = _inputs(torch, getattr(torch, dtype_name))
    plans = {}
    iotas = {}

    def execute(values):
        rows = values.shape[0]
        if rows not in plans:
            plans[rows] = _arg_plan(cudnn_handle, torch, dtype_id, value_type, kind, rows)
            iotas[rows] = torch.arange(_EXTENT, dtype=torch.int32).expand(rows, _EXTENT).contiguous()
        graph, x_desc, index_desc, output_desc = plans[rows]
        x = values.cuda()
        index_input = iotas[rows].cuda()
        before = snapshot({"x": x, "index_input": index_input})
        output = torch.full((rows, 1), -1, dtype=torch.int32, device="cuda")
        workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
        _execute_on_handle_stream(torch, cudnn_handle, graph, {x_desc: x, index_desc: index_input, output_desc: output}, workspace)
        assert_bitwise({"x": x, "index_input": index_input}, before, "inputs")
        expected = getattr(torch, kind)(values, dim=1, keepdim=True).to(torch.int32)
        assert torch.equal(output.cpu(), expected), f"{kind} {dtype_id}: independent Torch index reference"
        return {"index": output}

    alone = _check_scenarios(torch, scenarios, execute)
    assert_golden(
        f"custom_reduction_{kind}_index_{dtype_id}_k257",
        {"x": scenarios["alone"][0], "index_input": iotas[1]},
        alone,
    )


def test_custom_reduction_direct_add_fprop_fp32_k257_replay_and_batch_invariance(cudnn_handle):
    torch = _require_backend(cudnn_handle)
    scenarios = _inputs(torch, torch.float32)
    plans = {}

    def execute(values):
        rows = values.shape[0]
        if rows not in plans:
            plans[rows] = _add_plan(cudnn_handle, rows)
        graph, x_desc, output_desc = plans[rows]
        # Multiples of one eighth give an exactly representable FP32 sum.
        values = values / 8
        x = values.cuda()
        before = snapshot({"x": x})
        output = torch.full((rows, 1), float("nan"), dtype=torch.float32, device="cuda")
        workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
        _execute_on_handle_stream(torch, cudnn_handle, graph, {x_desc: x, output_desc: output}, workspace)
        assert_bitwise({"x": x}, before, "inputs")
        expected = values.sum(dim=1, keepdim=True)
        assert torch.equal(output.cpu(), expected), "direct add: independent Torch sum reference"
        return {"sum": output}

    alone = _check_scenarios(torch, scenarios, execute)
    assert_golden("custom_reduction_direct_add_fp32_k257", {"x": scenarios["alone"][0] / 8}, alone)
