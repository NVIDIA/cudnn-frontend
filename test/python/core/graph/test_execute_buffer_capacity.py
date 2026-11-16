# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend input/output bounds survive normalization and every public transport."""

import math

import cudnn
import pytest
import torch

from cudnn.engines.engine_ids import is_backend_engine
from test_execute_workspace import _PythonBuffer, _call

pytestmark = pytest.mark.L0


def _graph(op, dtype, handle, overriding=False):
    graph = cudnn.pygraph(
        handle=handle, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, is_override_shape_enabled=overriding
    )
    shape = (2, 64, 64) if op == "matmul" else (2, 2, 64, 64)
    count = 2 if op == "matmul" else 3
    values = [torch.zeros(shape, device="cuda", dtype=dtype) for _ in range(count)]
    tensors = [graph.tensor_like(value) for value in values]
    if op == "matmul":
        outputs = [graph.matmul(A=tensors[0], B=tensors[1])]
    elif op == "sdpa":
        output, _ = graph.sdpa(q=tensors[0], k=tensors[1], v=tensors[2], generate_stats=False, attn_scale=0.125)
        outputs = [output]
    else:
        values[2].fill_(1)
        values.extend([torch.ones_like(values[0]), torch.ones_like(values[0]), torch.full((2, 2, 64, 1), math.log(64), device="cuda")])
        tensors.extend(graph.tensor_like(value) for value in values[3:])
        outputs = list(graph.sdpa_backward(q=tensors[0], k=tensors[1], v=tensors[2], o=tensors[3], dO=tensors[4], stats=tensors[5], attn_scale=0.125))
    for output in outputs:
        output.set_output(True).set_dim(shape).set_stride(values[0].stride()).set_data_type(
            cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
        )
        tensors.append(output)
        values.append(torch.empty_like(values[0]))
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    errors = []
    for index, config in enumerate(graph.plans):
        if not is_backend_engine(config.engine_id):
            continue
        try:
            graph.build_plan_at_index(index)
        except (cudnn.cudnnGraphNotSupportedError, NotImplementedError) as exc:
            errors.append(str(exc))
            continue
        assert graph.selected_engine is None
        return graph, tensors, values
    pytest.fail(f"No backend {op} plan: {errors}")


@pytest.mark.parametrize("entry", ["mapping", "ordered", "mapping_at_index", "ordered_at_index"])
@pytest.mark.parametrize("op,overriding", [("matmul", False), ("matmul", True), ("sdpa", False), ("sdpa", True), ("sdpa_backward", False)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("port", ["input", "output"])
@pytest.mark.parametrize("producer", ["native", "python"])
def test_short_operand_rejects_before_launch(entry, op, overriding, dtype, port, producer, cudnn_handle):
    graph, tensors, values = _graph(op, dtype, cudnn_handle, overriding)
    if overriding:
        # The storage still covers the declaration: a backend plan may ignore
        # a smaller batch override, even when overriding was enabled (#1570).
        values[-1].zero_()
    overrides = (
        dict(
            override_uids=[t.get_uid() for t in tensors],
            override_shapes=[list(v[:1].shape) for v in values],
            override_strides=[list(v.stride()) for v in values],
        )
        if overriding
        else {}
    )
    required_ws = graph.get_workspace_size_plan_at_index(graph._plan_index, handle=cudnn_handle, **overrides)
    workspace = torch.empty(max(required_ws, 1), device="cuda", dtype=torch.uint8)
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    _call(graph, tensors, values, workspace, cudnn_handle, entry, overrides)
    torch.cuda.synchronize()
    expected = values[-1].clone()
    assert torch.isfinite(expected).all().item()
    slot = 0 if port == "input" else len(values) - 1
    original = values[slot]
    required = original.numel() * original.element_size()
    # Enough owned memory for the full access even before the fix. Only the
    # caller's view is short, so RED cannot touch an unrelated allocation.
    backing = torch.full((required + 256,), 165, device="cuda", dtype=torch.uint8)
    full = backing[:required]
    full.copy_(original.reshape(-1).view(torch.uint8))
    wrap = _PythonBuffer if producer == "python" else lambda value: value
    values[slot] = wrap(full[: required - 1])
    before = backing.clone()
    workspace.fill_(165)
    if port == "input":
        values[-1].fill_(123)
    with pytest.raises(ValueError, match=f"tensor uid {tensors[slot].get_uid()} requires a {required}-byte buffer but received {required - 1} bytes"):
        _call(graph, tensors, values, workspace, cudnn_handle, entry, overrides)
    torch.cuda.synchronize()
    torch.testing.assert_close(backing, before, atol=0, rtol=0)
    assert torch.all(workspace == 165).item(), "Reject before initializing FE scratch"
    if port == "input":
        assert torch.all(values[-1] == 123).item()
    # Flat byte storage, a wider typed carrier, and an unknown-capacity pointer
    # are equivalent valid bindings; shape/dtype equality is not the contract.
    for carrier in (full, full.view(torch.float32), full.data_ptr()):
        values[slot] = wrap(carrier) if not isinstance(carrier, int) else carrier
        _call(graph, tensors, values, workspace, cudnn_handle, entry, overrides)
        torch.cuda.synchronize()
        actual = full.view(dtype).reshape(original.shape) if port == "output" else values[-1]
        torch.testing.assert_close(actual[:1] if overriding else actual, expected[:1] if overriding else expected, atol=0, rtol=0)
        assert torch.all(backing[required:] == 165).item()


@pytest.mark.parametrize("entry", ["mapping", "ordered", "mapping_at_index", "ordered_at_index"])
@pytest.mark.parametrize("overriding", [False, True])
def test_deserialized_graph_uses_loaded_operand_bounds(entry, overriding, cudnn_handle):
    source, tensors, values = _graph("matmul", torch.float16, cudnn_handle, overriding)
    graph = cudnn.pygraph(handle=cudnn_handle)
    graph.deserialize(source.serialize())
    full_required = values[-1].numel() * values[-1].element_size()
    overrides = (
        dict(
            override_uids=[t.get_uid() for t in tensors],
            override_shapes=[list(v[:1].shape) for v in values],
            override_strides=[list(v.stride()) for v in values],
        )
        if overriding
        else {}
    )
    required = values[-1].numel() * values[-1].element_size()
    backing = torch.full((full_required + 256,), 165, device="cuda", dtype=torch.uint8)
    values[-1] = backing[: required - 1]
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match=f"tensor uid {tensors[-1].get_uid()} requires a {required}-byte buffer"):
        _call(graph, tensors, values, workspace, cudnn_handle, entry, overrides)
    torch.cuda.synchronize()
    assert torch.all(backing == 165).item()
    values[-1] = backing[:required]
    _call(graph, tensors, values, workspace, cudnn_handle, entry, overrides)
    torch.cuda.synchronize()
    executed = required // 2 if overriding else required
    assert torch.all(backing[:executed] == 0).item()
    assert torch.all(backing[executed:] == 165).item()


@pytest.mark.parametrize("entry", ["mapping", "ordered", "mapping_at_index", "ordered_at_index"])
@pytest.mark.parametrize("op", ["matmul", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deserialized", [False, True])
def test_overrides_cannot_shrink_fixed_graph_capacity(entry, op, dtype, deserialized, cudnn_handle):
    source, tensors, values = _graph(op, dtype, cudnn_handle, overriding=False)
    graph = source
    if deserialized:
        graph = cudnn.pygraph(handle=cudnn_handle)
        graph.deserialize(source.serialize())
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    required = values[-1].numel() * values[-1].element_size()
    backing = torch.full((required,), 165, device="cuda", dtype=torch.uint8)
    # Override metadata is ignored by a fixed-shape backend graph. It must not
    # weaken its capacity check, including after loading into a fresh container.
    overrides = dict(
        override_uids=[t.get_uid() for t in tensors],
        override_shapes=[list(value[:1].shape) for value in values],
        override_strides=[list(value.stride()) for value in values],
    )
    values[-1] = backing[: required // 2]
    workspace = torch.full((max(graph.get_workspace_size(), 1),), 165, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match=f"tensor uid {tensors[-1].get_uid()} requires a {required}-byte buffer but received {required // 2} bytes"):
        _call(graph, tensors, values, workspace, cudnn_handle, entry, overrides)
    torch.cuda.synchronize()
    assert torch.all(backing == 165).item()
    assert torch.all(workspace == 165).item()
