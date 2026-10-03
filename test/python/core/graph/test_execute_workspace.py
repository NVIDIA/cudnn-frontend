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


def _build(handle, needs_workspace):
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
        values = [torch.randn((2, 32, 4, 128), generator=rng, device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(3)]
        tensors = [graph.tensor_like(value) for value in values]
        output, _ = graph.sdpa(q=tensors[0], k=tensors[1], v=tensors[2], generate_stats=False, attn_scale=128**-0.5, use_alibi_mask=True, use_causal_mask=True)
        output.set_output(True).set_dim(values[0].shape).set_stride(values[0].stride()).set_data_type(cudnn.data_type.BFLOAT16)
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


@pytest.mark.parametrize("entry", ["mapping", "ordered", "mapping_at_index", "ordered_at_index", "raw", "cpp_candidate"])
@pytest.mark.parametrize("overriding", [False, True])
def test_required_backend_workspace_rejects_null_and_recovers(entry, overriding, cudnn_handle):
    if overriding and (entry == "raw" or cudnn.backend_version() < 92600):
        pytest.skip("Raw transport has no overrides; this override graph needs cuDNN 9.26+")
    graph, tensors, values = _build(cudnn_handle, True)
    # Exercise an actual effective-shape change rather than echoing the
    # declaration. This plan's FE ALiBi scratch remains required at B=1.
    if overriding:
        values = [value[:1] for value in values]
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
