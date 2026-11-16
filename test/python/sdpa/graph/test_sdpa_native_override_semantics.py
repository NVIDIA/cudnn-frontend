# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native override requests must be honored or rejected, never silently ignored."""

import math

import cudnn
import pytest
import torch

from cudnn._handle import to_backend_handle

pytestmark = pytest.mark.L0


def _case(handle, dtype, *, declared_batch=2, declared_layout="bhsd", physical_layout="bhsd"):
    if cudnn.backend_version() < 92300:
        pytest.skip("Runtime workspace queries require cuDNN 9.23+")
    batch, heads, seq, dim = 2, 4, 64, 64
    rng = torch.Generator(device="cpu").manual_seed(1570)
    values = [torch.randn(batch, heads, seq, dim, generator=rng).to(dtype).cuda() for _ in range(3)]
    if physical_layout == "bshd":
        values = [v.transpose(1, 2).contiguous().transpose(1, 2) for v in values]
    q, k, v = values
    scores = q.double() @ k.double().transpose(-1, -2) / math.sqrt(dim)
    reference = torch.softmax(scores, -1) @ v.double()
    lse = scores.logsumexp(-1).unsqueeze(-1)
    # Both declared and override views fit these allocations, including B=1 -> 2.
    values.extend([torch.full_like(q, float("nan")), torch.full((batch, heads, seq, 1), float("nan"), device="cuda")])
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(
        handle=handle,
        io_data_type=dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=True,
    )
    strides = [heads * seq * dim, seq * dim, dim, 1] if declared_layout == "bhsd" else [seq * heads * dim, dim, heads * dim, 1]
    q_t, k_t, v_t = [graph.tensor(uid=i, dim=[declared_batch, heads, seq, dim], stride=strides, data_type=dt) for i in (1, 2, 3)]
    out, stats = graph.sdpa(q=q_t, k=k_t, v=v_t, generate_stats=True, attn_scale=dim**-0.5)
    out.set_uid(4).set_output(True).set_data_type(dt).set_dim([declared_batch, heads, seq, dim]).set_stride(strides)
    stats.set_uid(5).set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([declared_batch, heads, seq, 1]).set_stride([heads * seq, seq, 1, 1])
    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except cudnn.cudnnGraphNotSupportedError as error:
        pytest.skip(f"No plan supports the declared native control: {error}")
    for index, plan in enumerate(graph.plans):
        if plan.engine_id >= 10000:
            continue
        graph.select_plan(index)
        try:
            graph.check_support()
            graph.build_plans()
        except cudnn.cudnnGraphNotSupportedError:
            continue
        assert graph.selected_engine is None
        return graph, values, reference, lse
    pytest.skip("No executable native plan supports the declared control")


def _run(graph, values, handle, batch, entry="mapping"):
    shapes = [list(v.shape) for v in values]
    for shape in shapes:
        shape[0] = batch
    overrides = dict(override_uids=[1, 2, 3, 4, 5], override_shapes=shapes, override_strides=[list(v.stride()) for v in values])
    # The direct C++ leg must reach execute's own guard without an earlier
    # Python workspace query rejecting the request. Storage covers both views.
    if entry == "cpp":
        workspace = torch.empty(max(graph.get_workspace_size(), 16 * 1024 * 1024), device="cuda", dtype=torch.uint8)
        pointers, ws = graph._native_var_pack(dict(zip(range(1, 6), values)), workspace)
        graph._lowered_graph._execute(pointers, ws, to_backend_handle(handle), **overrides)
    else:
        workspace = torch.empty(max(graph.get_workspace_size(), graph.get_workspace_size(handle=handle, **overrides), 1), device="cuda", dtype=torch.uint8)
        if entry == "mapping":
            graph.execute(dict(zip(range(1, 6), values)), workspace, handle=handle, **overrides)
        elif entry == "ordered":
            graph.execute(tuple(values), workspace, tensor_uids=(1, 2, 3, 4, 5), handle=handle, **overrides)
        elif entry == "indexed":
            graph.execute_plan_at_index(dict(zip(range(1, 6), values)), workspace, index=max(0, graph._plan_index), handle=handle, **overrides)
        else:
            raise AssertionError(entry)
    torch.cuda.synchronize()


def _check(values, reference, lse, batch):
    torch.testing.assert_close(values[3][:batch].double(), reference[:batch], atol=0.008, rtol=0.02)
    torch.testing.assert_close(values[4][:batch].double(), lse[:batch], atol=0.001, rtol=0.001)
    assert torch.isnan(values[3][batch:]).all(), "execution wrote an inactive output batch"
    assert torch.isnan(values[4][batch:]).all(), "execution wrote inactive Stats"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout,batch", [("bhsd", 2), ("bshd", 2), ("bhsd", 1)])
@pytest.mark.parametrize("entry", ["mapping", "ordered", "cpp"])
def test_native_override_legal_geometry(dtype, layout, batch, entry, cudnn_handle):
    graph, values, reference, lse = _case(cudnn_handle, dtype, declared_layout=layout, physical_layout=layout)
    if batch == 1 and not (torch.cuda.get_device_capability() == (10, 0) and 92400 <= cudnn.backend_version() <= 92800):
        pytest.skip("Smaller-batch positive control is qualified on SM100/cuDNN 9.24-9.28 only")
    _run(graph, values, cudnn_handle, batch, entry)
    _check(values, reference, lse, batch)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("change", ["strides", "batch_grow", "batch_shrink_hopper"])
@pytest.mark.parametrize("entry", ["mapping", "ordered", "indexed", "cpp"])
@pytest.mark.parametrize("restored", [False, True])
def test_native_override_is_honored_or_rejected(dtype, change, entry, restored, cudnn_handle):
    graph, values, reference, lse = _case(
        cudnn_handle,
        dtype,
        declared_batch=1 if change == "batch_grow" else 2,
        physical_layout="bshd" if change == "strides" else "bhsd",
    )
    if change == "batch_shrink_hopper":
        if torch.cuda.get_device_capability() != (9, 0) or not graph.get_plan_name_at_index(graph._plan_index).startswith("eng9_"):
            pytest.skip("Smaller-batch failure is qualified on Hopper engine 9")
    batch = 1 if change == "batch_shrink_hopper" else 2
    if restored:
        payload = graph.serialize(as_bytes=True)
        graph = cudnn.pygraph(handle=cudnn_handle)
        graph.deserialize(payload, enforce_precompiled=True)
    try:
        _run(graph, values, cudnn_handle, batch, entry)
    except (ValueError, RuntimeError) as error:
        assert any(word in str(error).lower() for word in ("override", "stride", "shape", "extent")), str(error)
        assert torch.isnan(values[3]).all(), "rejected override wrote output"
        assert torch.isnan(values[4]).all(), "rejected override wrote Stats"
        return
    _check(values, reference, lse, batch)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("restored", [False, True])
def test_non_sdpa_override_remains_supported(dtype, restored, cudnn_handle):
    if cudnn.backend_version() < 92300:
        pytest.skip("Runtime workspace queries require cuDNN 9.23+")
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=True,
    )
    rng = torch.Generator(device="cpu").manual_seed(1570)
    a, b = [torch.randn(2, 32, 32, generator=rng).to(dtype).cuda() for _ in range(2)]
    output = torch.full_like(a, float("nan"))
    at, bt = graph.tensor_like(a).set_uid(1), graph.tensor_like(b).set_uid(2)
    graph.matmul(at, bt).set_uid(3).set_output(True).set_data_type(dt)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    assert graph.selected_engine is None
    if restored:
        payload = graph.serialize(as_bytes=True)
        graph = cudnn.pygraph(handle=cudnn_handle)
        graph.deserialize(payload, enforce_precompiled=True)
    kw = dict(override_uids=[1, 2, 3], override_shapes=[[1, 32, 32]] * 3, override_strides=[list(x.stride()) for x in (a, b, output)])
    workspace = torch.empty(max(graph.get_workspace_size(handle=cudnn_handle, **kw), 1), dtype=torch.uint8, device="cuda")
    graph.execute({1: a, 2: b, 3: output}, workspace, handle=cudnn_handle, **kw)
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:1].double(), a[:1].double() @ b[:1].double(), atol=0.06, rtol=0.02)
    assert torch.isnan(output[1:]).all()


@pytest.mark.parametrize("reserialize", [False, True])
def test_legacy_plan_without_geometry_requires_rebuild_for_overrides(reserialize, cudnn_handle):
    graph, values, reference, lse = _case(cudnn_handle, torch.float16)
    payload = graph.serialize(as_bytes=True)
    field = b"sdpa_override_operands"
    assert payload.count(field) == 1
    # A same-length unknown key preserves the UBJSON encoding and models a
    # cache lacking this field, without adding a Python UBJSON dependency.
    legacy = payload.replace(field, b"x" * len(field))
    graph = cudnn.pygraph(handle=cudnn_handle)
    graph.deserialize(legacy, enforce_precompiled=True)
    if reserialize:
        again = graph.serialize(as_bytes=True)
        assert field not in again, "re-serialization invented a non-SDPA contract"
        graph = cudnn.pygraph(handle=cudnn_handle)
        graph.deserialize(again, enforce_precompiled=True)
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    graph.execute(dict(zip(range(1, 6), values)), workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    _check(values, reference, lse, 2)
    values[3].fill_(float("nan"))
    values[4].fill_(float("nan"))
    with pytest.raises((ValueError, RuntimeError), match="Rebuild this legacy execution-plan cache"):
        _run(graph, values, cudnn_handle, 2, "cpp")
    assert torch.isnan(values[3]).all()
    assert torch.isnan(values[4]).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("restored", [False, True])
def test_native_stats_trailing_singleton_remains_optional(dtype, restored, cudnn_handle):
    graph, values, reference, lse = _case(cudnn_handle, dtype)
    if restored:
        payload = graph.serialize(as_bytes=True)
        graph = cudnn.pygraph(handle=cudnn_handle)
        graph.deserialize(payload, enforce_precompiled=True)
    values[4] = values[4].squeeze(-1)
    _run(graph, values, cudnn_handle, 2)
    _check(values, reference, lse.squeeze(-1), 2)
