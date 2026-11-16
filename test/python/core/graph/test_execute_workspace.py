# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All backend execution transports reject missing required workspace before launch."""

import cudnn
import pytest
import torch

from cudnn.engines.engine_ids import is_backend_engine
from cudnn._handle import to_backend_handle

pytestmark = [pytest.mark.L0]


@pytest.fixture(autouse=True)
def _restore_stream(cudnn_handle):
    original = cudnn.get_stream(cudnn_handle)
    try:
        yield
    finally:
        cudnn.set_stream(cudnn_handle, original)


def _build(handle, needs_workspace, dtype=torch.bfloat16):
    if needs_workspace and (cudnn.backend_version() < 92600 or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("This backend ALiBi case requires Hopper+ and cuDNN 9.26+")
    graph = cudnn.pygraph(
        handle=handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=needs_workspace,
    )
    if needs_workspace:
        # ALiBi slopes occupy FE scratch independently of the backend plan's
        # workspace choice, so this is not tied to a heuristic engine ranking.
        rng = torch.Generator(device="cuda").manual_seed(471)
        values = [torch.randn((2, 32, 4, 128), generator=rng, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(3)]
        tensors = [graph.tensor_like(value) for value in values]
        output, _ = graph.sdpa(q=tensors[0], k=tensors[1], v=tensors[2], generate_stats=False, attn_scale=128**-0.5, use_alibi_mask=True, use_causal_mask=True)
        output.set_output(True).set_dim(values[0].shape).set_stride(values[0].stride()).set_data_type(
            cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
        )
        result = torch.empty_like(values[0])
    else:
        # C=1 is rejected by the Hopper pointwise backend. A channels-last
        # C=16 tensor keeps this zero-workspace control valid across SM80+.
        values = [torch.linspace(-1, 1, 256, device="cuda").reshape(1, 16, 4, 4).contiguous(memory_format=torch.channels_last)]
        tensors = [graph.tensor_like(values[0])]
        output = graph.relu(tensors[0]).set_output(True)
        result = torch.empty_like(values[0])
    tensors.append(output)
    values.append(result)
    for uid, tensor in enumerate(tensors, 1):
        tensor.set_uid(uid)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    errors = []
    for index, cfg in enumerate(graph.plans):
        if not is_backend_engine(cfg.engine_id):
            continue
        try:
            graph.build_plan_at_index(index)
        except (cudnn.cudnnGraphNotSupportedError, NotImplementedError) as exc:
            errors.append(str(exc))
            continue
        if (graph.get_workspace_size() > 0) == needs_workspace:
            assert graph.selected_engine is None
            return graph, tensors, values
    pytest.fail(f"No backend plan with the required workspace contract: {errors}")


def _call(graph, tensors, values, workspace, handle, entry, overrides):
    uid_map = {t.get_uid(): value for t, value in zip(tensors, values)}
    uids = tuple(uid_map)
    if entry == "mapping":
        graph.execute(uid_map, workspace, handle=handle, **overrides)
    elif entry == "ordered":
        graph.execute(tuple(values), workspace, handle=handle, tensor_uids=uids, **overrides)
    elif entry == "mapping_at_index":
        graph.execute_plan_at_index(uid_map, workspace, index=graph._plan_index, handle=handle, **overrides)
    elif entry == "ordered_at_index":
        graph.execute_plan_at_index(tuple(values), workspace, index=graph._plan_index, handle=handle, tensor_uids=uids, **overrides)
    elif entry == "raw":
        assert not overrides
        pack = graph._normalize(uid_map, workspace)
        cfg = graph._materialize_backend_plan(graph._plan_index)
        graph._lowered_graph._execute_with_raw_ptrs(pack.address, len(pack), pack.workspace, to_backend_handle(handle), cfg.cpp_index)
    elif entry == "cpp_candidate":
        pointers, ws = graph._native_var_pack(uid_map, workspace)
        graph._lowered_graph._execute(pointers, ws, to_backend_handle(handle), **overrides)
    else:
        raise AssertionError(entry)


def _smaller_geometry(graph, values):
    # Hopper engine 9 honors KV sequence lengths but ignores batch changes.
    # Keep an actual shape change in the capacity tests on both engine families.
    if torch.cuda.get_device_capability() == (9, 0) and graph.get_plan_name_at_index(graph._plan_index).startswith("eng9_"):
        return [values[0], values[1][:, :, :16], values[2][:, :, :16], values[3]]
    return [value[:1] for value in values]


@pytest.mark.parametrize("entry", ["mapping", "ordered", "mapping_at_index", "ordered_at_index", "raw", "cpp_candidate"])
@pytest.mark.parametrize("overriding", [False, True])
def test_required_backend_workspace_rejects_null_and_recovers(entry, overriding, cudnn_handle):
    if overriding and (entry == "raw" or cudnn.backend_version() < 92600):
        pytest.skip("Raw transport has no overrides; this override graph needs cuDNN 9.26+")
    graph, tensors, values = _build(cudnn_handle, True)
    # The smaller geometry still requires this plan's FE ALiBi scratch.
    if overriding:
        values = _smaller_geometry(graph, values)
    overrides = (
        dict(override_uids=[t.get_uid() for t in tensors], override_shapes=[list(t.shape) for t in values], override_strides=[list(t.stride()) for t in values])
        if overriding
        else {}
    )
    required = graph.get_workspace_size_plan_at_index(graph._plan_index, handle=cudnn_handle, **overrides)
    assert required > 0
    workspace = torch.empty(required, device="cuda", dtype=torch.uint8)
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    _call(graph, tensors, values, workspace, cudnn_handle, entry, overrides)
    torch.cuda.synchronize()
    expected = values[-1].clone()
    assert torch.isfinite(expected).all().item()
    for null_workspace in (None, 0, torch.empty(0, device="cuda", dtype=torch.uint8)):
        values[-1].fill_(123)
        with pytest.raises(ValueError, match=f"requires a {required}-byte workspace but received null"):
            _call(graph, tensors, values, null_workspace, cudnn_handle, entry, overrides)
        torch.cuda.synchronize()
        assert torch.all(values[-1] == 123).item()
        _call(graph, tensors, values, workspace, cudnn_handle, entry, overrides)
        torch.cuda.synchronize()
        torch.testing.assert_close(values[-1], expected, atol=0, rtol=0)


@pytest.mark.parametrize("entry", ["mapping", "ordered", "mapping_at_index", "ordered_at_index", "raw", "cpp_candidate"])
def test_zero_workspace_backend_accepts_null(entry, cudnn_handle):
    graph, tensors, values = _build(cudnn_handle, False)
    assert graph.get_workspace_size() == 0
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    for workspace in (None, 0, torch.empty(0, device="cuda", dtype=torch.uint8)):
        values[-1].fill_(float("nan"))
        _call(graph, tensors, values, workspace, cudnn_handle, entry, {})
        torch.cuda.synchronize()
        torch.testing.assert_close(values[-1], torch.relu(values[0]), atol=0, rtol=0)


class _PythonBuffer:
    """A supported producer without the native exchange protocol."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.shape, self.dtype = tensor.shape, tensor.dtype

    def data_ptr(self):
        return self.tensor.data_ptr()

    def stride(self):
        return self.tensor.stride()


@pytest.mark.parametrize("entry", ["mapping", "ordered", "mapping_at_index", "ordered_at_index"])
@pytest.mark.parametrize("overriding", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("producer", ["native", "python"])
def test_short_backend_workspace_rejects_before_any_write(entry, overriding, dtype, producer, cudnn_handle):
    graph, tensors, values = _build(cudnn_handle, True, dtype)
    _assert_short_workspace(graph, tensors, values, entry, overriding, producer, cudnn_handle)


def _assert_short_workspace(graph, tensors, values, entry, overriding, producer, cudnn_handle):
    if overriding:
        values = _smaller_geometry(graph, values)
    overrides = (
        dict(override_uids=[t.get_uid() for t in tensors], override_shapes=[list(t.shape) for t in values], override_strides=[list(t.stride()) for t in values])
        if overriding
        else {}
    )
    required = graph.get_workspace_size_plan_at_index(graph._plan_index, handle=cudnn_handle, **overrides)
    assert required > 1
    # The short view stays inside an owned allocation large enough for the plan:
    # a RED run can expose the missing check without an invalid device access.
    backing = torch.empty(required + 256, dtype=torch.uint8, device="cuda")
    full = backing[:required]
    wrap = _PythonBuffer if producer == "python" else lambda value: value
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    _call(graph, tensors, values, wrap(full), cudnn_handle, entry, overrides)
    torch.cuda.synchronize()
    expected = values[-1].clone()
    assert torch.isfinite(expected).all().item()
    for length in (1, required - 1):
        backing.fill_(165)
        values[-1].fill_(123)
        with pytest.raises(ValueError, match=f"requires a {required}-byte workspace but received a {length}-byte buffer"):
            _call(graph, tensors, values, wrap(full[:length]), cudnn_handle, entry, overrides)
        torch.cuda.synchronize()
        assert torch.all(backing == 165).item(), "Rejected execution must not initialize FE scratch"
        assert torch.all(values[-1] == 123).item(), "Rejected execution must not write the output"
    # Exact capacity, a differently typed byte carrier, and an unknown-capacity
    # integer address remain valid; capacity is in bytes, never element count.
    for workspace in (wrap(full), wrap(backing[: (required + 3) // 4 * 4].view(torch.float32)), full.data_ptr()):
        values[-1].fill_(float("nan"))
        _call(graph, tensors, values, workspace, cudnn_handle, entry, overrides)
        torch.cuda.synchronize()
        torch.testing.assert_close(values[-1], expected, atol=0, rtol=0)


@pytest.mark.parametrize("entry", ["mapping", "ordered", "mapping_at_index", "ordered_at_index"])
def test_deserialized_backend_checks_current_workspace_capacity(entry, cudnn_handle):
    source, tensors, values = _build(cudnn_handle, True)
    graph = cudnn.pygraph(handle=cudnn_handle)
    graph.deserialize(source.serialize())
    required = graph.get_workspace_size()
    assert required > 1
    backing = torch.empty(required + 256, device="cuda", dtype=torch.uint8)
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
    _call(graph, tensors, values, backing[:required], cudnn_handle, entry, {})
    torch.cuda.synchronize()
    expected = values[-1].clone()
    backing.fill_(165)
    values[-1].fill_(123)
    with pytest.raises(ValueError, match=f"requires a {required}-byte workspace but received a 1-byte buffer"):
        _call(graph, tensors, values, backing[:1], cudnn_handle, entry, {})
    torch.cuda.synchronize()
    assert torch.all(backing == 165).item()
    assert torch.all(values[-1] == 123).item()
    _call(graph, tensors, values, backing[:required], cudnn_handle, entry, {})
    torch.cuda.synchronize()
    torch.testing.assert_close(values[-1], expected, atol=0, rtol=0)


@pytest.mark.parametrize("entry", ["mapping", "ordered", "mapping_at_index", "ordered_at_index"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_conv_wgrad_checks_workspace_capacity(entry, dtype, cudnn_handle):
    graph = cudnn.pygraph(handle=cudnn_handle, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    rng = torch.Generator(device="cuda").manual_seed(471)
    values = [
        torch.randn(4, channels, 32, 32, generator=rng, device="cuda", dtype=dtype).contiguous(memory_format=torch.channels_last) for channels in (32, 64)
    ]
    tensors = [graph.tensor_like(value) for value in values]
    result = torch.empty((64, 32, 3, 3), device="cuda", dtype=dtype).contiguous(memory_format=torch.channels_last)
    output = graph.conv_wgrad(image=tensors[0], loss=tensors[1], padding=[1, 1], stride=[1, 1], dilation=[1, 1])
    output.set_output(True).set_dim(result.shape).set_stride(result.stride()).set_data_type(
        cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    )
    tensors.append(output)
    values.append(result)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    errors = []
    for index, cfg in enumerate(graph.plans):
        if not is_backend_engine(cfg.engine_id):
            continue
        try:
            graph.build_plan_at_index(index)
        except (cudnn.cudnnGraphNotSupportedError, NotImplementedError) as exc:
            errors.append(str(exc))
            continue
        if graph.get_workspace_size() > 1:
            _assert_short_workspace(graph, tensors, values, entry, False, "native", cudnn_handle)
            return
    pytest.fail(f"No backend wgrad plan with scratch: {errors}")
