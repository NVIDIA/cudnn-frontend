# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered tensor bindings preserve the ordinary graph.execute contract."""

import math

import cudnn
import pytest
import torch

from cudnn.engines.engine_ids import is_backend_engine

_B, _HQ, _HK, _D = 2, 4, 2, 128
_Q, _KV = 32, 64


def _graph(handle, provider, uid_offset=0):
    if cudnn.backend_version() < 92600:
        pytest.skip("This shared ragged-override case requires cuDNN 9.26 or later")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("This BF16 attention case requires Hopper or later")
    if provider == "frost":
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

        installed, version = cutedsl_state()
        if not 100 <= major * 10 + minor <= 106 or not installed or cutedsl_too_old(version):
            pytest.skip("The FROST case requires pre-Rubin SM100 and a supported CuTe DSL")
    cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
    graph = cudnn.pygraph(
        handle=handle,
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=True,
    )
    tensors = {}
    for name, heads, seq in (("q", _HQ, _Q), ("k", _HK, _KV), ("v", _HK, _KV)):
        tensors[name] = graph.tensor(name=name, dim=[_B, heads, seq, _D], stride=[seq * heads * _D, _D, heads * _D, 1])
    for name in ("cu_q", "cu_kv", "off_q", "off_kv", "off_lse"):
        tensors[name] = graph.tensor(name=name, dim=[_B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    tensors["q"].set_ragged_offset(tensors["off_q"])
    for name in ("k", "v"):
        tensors[name].set_ragged_offset(tensors["off_kv"])
    out, stats = graph.sdpa(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        generate_stats=True,
        attn_scale=_D**-0.5,
        use_padding_mask=True,
        cu_seq_len_q=tensors["cu_q"],
        cu_seq_len_kv=tensors["cu_kv"],
        max_total_seq_len_q=_B * _Q,
        max_total_seq_len_kv=_B * _KV,
        implementation=cudnn.attention_implementation.UNIFIED,
    )
    out.set_output(True).set_dim([_B, _HQ, _Q, _D]).set_stride([_Q * _HQ * _D, _D, _HQ * _D, 1]).set_ragged_offset(tensors["off_q"])
    stats.set_output(True).set_dim([_B, _HQ, _Q, 1]).set_stride([_Q * _HQ, 1, _HQ, 1]).set_data_type(cudnn.data_type.FLOAT)
    stats.set_ragged_offset(tensors["off_lse"])
    tensors.update(o=out, lse=stats)
    if uid_offset:
        for index, tensor in enumerate(tensors.values()):
            tensor.set_uid(uid_offset + index)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    _select_provider(graph, provider)
    return graph, tensors


def _select_provider(graph, provider):
    """Select a supported provider explicitly, without asserting heuristic order."""
    candidates = []
    for index, config in enumerate(graph.plans):
        if provider == "backend" and is_backend_engine(config.engine_id):
            candidates.append(index)
        elif provider == "frost" and graph.get_plan_name_at_index(index).startswith("sdpa_fwd_prefill_sm100"):
            candidates.append(index)
    for index in candidates:
        graph.select_plan(index)
        try:
            graph.build_plans()
        except (cudnn.cudnnGraphNotSupportedError, NotImplementedError):
            continue
        assert (graph.selected_engine is None) == (provider == "backend")
        return index
    pytest.skip(f"No {provider} plan supports the representative attention graph")


def _buffers(q_len=_Q, kv_len=_KV, *, seed=0, gaps=False):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    buffers = {}
    for name, tokens, heads in (("q", _B * q_len, _HQ), ("k", _B * kv_len, _HK), ("v", _B * kv_len, _HK)):
        # A gap after each token is legal when the graph receives matching strides.
        value = torch.randn(tokens, heads + int(gaps), _D, generator=rng, device="cuda", dtype=torch.bfloat16)
        buffers[name] = value[:, :heads]
    buffers["o"] = torch.full((_B * q_len, _HQ + int(gaps), _D), float("nan"), device="cuda", dtype=torch.bfloat16)[:, :_HQ]
    buffers["lse"] = torch.full((_B * q_len, _HQ), float("nan"), device="cuda")
    buffers["cu_q"] = torch.arange(_B + 1, device="cuda", dtype=torch.int32) * q_len
    buffers["cu_kv"] = torch.arange(_B + 1, device="cuda", dtype=torch.int32) * kv_len
    buffers["off_q"] = buffers["cu_q"] * buffers["q"].stride(0)
    buffers["off_kv"] = buffers["cu_kv"] * buffers["k"].stride(0)
    buffers["off_lse"] = buffers["cu_q"] * _HQ
    return buffers


def _overrides(tensors, buffers, q_len=_Q, kv_len=_KV):
    names = ("q", "k", "v", "o", "lse")
    shapes, strides = [], []
    for name in names:
        heads = _HK if name in ("k", "v") else _HQ
        seq = kv_len if name in ("k", "v") else q_len
        dim = 1 if name == "lse" else _D
        shape = [_B, heads, seq, dim]
        data = buffers[name]
        stride = [seq * data.stride(0), data.stride(1), data.stride(0), 1]
        shapes.append(shape)
        strides.append(stride)
    return dict(override_uids=[tensors[n].get_uid() for n in names], override_shapes=shapes, override_strides=strides)


def _mapping(tensors, buffers):
    return {tensor: buffers[name] for name, tensor in tensors.items()}


def _ordered(tensors, buffers, reverse=False):
    names = sorted(tensors, reverse=reverse)
    return [buffers[name] for name in names], [tensors[name].get_uid() for name in names]


def _workspace(graph, handle, overrides=None):
    if overrides:
        size = graph.get_workspace_size_plan_at_index(
            graph._plan_index, handle, overrides["override_uids"], overrides["override_shapes"], overrides["override_strides"]
        )
    else:
        size = graph.get_workspace_size()
    return torch.empty(max(size, 1), device="cuda", dtype=torch.uint8)


def _assert_result(buffers, q_len=_Q, kv_len=_KV):
    for batch in range(_B):
        q = buffers["q"][batch * q_len : (batch + 1) * q_len].double().transpose(0, 1)
        k = buffers["k"][batch * kv_len : (batch + 1) * kv_len].double().transpose(0, 1).repeat_interleave(_HQ // _HK, 0)
        v = buffers["v"][batch * kv_len : (batch + 1) * kv_len].double().transpose(0, 1).repeat_interleave(_HQ // _HK, 0)
        logits = q @ k.transpose(1, 2) / math.sqrt(_D)
        expected_o = (logits.softmax(-1) @ v).transpose(0, 1)
        expected_lse = logits.logsumexp(-1).transpose(0, 1)
        torch.testing.assert_close(buffers["o"][batch * q_len : (batch + 1) * q_len].double(), expected_o, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(buffers["lse"][batch * q_len : (batch + 1) * q_len].double(), expected_lse, atol=2e-3, rtol=2e-3)


@pytest.fixture(params=["backend", "frost"])
def attention_case(request, cudnn_handle):
    graph, tensors = _graph(cudnn_handle, request.param)
    yield graph, tensors, cudnn_handle, request.param
    cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)


@pytest.mark.L0
def test_ordered_matches_mapping_with_fresh_buffers_and_uid_order(attention_case):
    graph, tensors, handle, _ = attention_case
    workspace = _workspace(graph, handle)
    for seed in (0, 1):
        buffers = _buffers(seed=seed)
        graph.execute(_mapping(tensors, buffers), workspace, handle=handle)
        torch.cuda.synchronize()
        _assert_result(buffers)
        expected = buffers["o"].clone(), buffers["lse"].clone()
        buffers["o"].fill_(float("nan"))
        buffers["lse"].fill_(float("nan"))
        ordered, uids = _ordered(tensors, buffers, reverse=bool(seed))
        graph.execute(ordered if seed else tuple(ordered), workspace, handle=handle, tensor_uids=uids)
        torch.cuda.synchronize()
        torch.testing.assert_close(buffers["o"], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(buffers["lse"], expected[1], atol=0, rtol=0)
        _assert_result(buffers)


@pytest.mark.L1
def test_ordered_at_index_uses_the_same_provider_contract(attention_case):
    graph, tensors, handle, _ = attention_case
    buffers = _buffers(seed=9)
    values, uids = _ordered(tensors, buffers)
    selected = graph._plan_index
    workspace = _workspace(graph, handle)
    with pytest.raises(ValueError, match="missing a buffer"):
        graph.execute_plan_at_index(values[:-1], workspace, index=selected, handle=handle, tensor_uids=uids[:-1])
    graph.execute(values, workspace, handle=handle, tensor_uids=uids)
    torch.cuda.synchronize()
    _assert_result(buffers)
    buffers["o"].fill_(float("nan"))
    buffers["lse"].fill_(float("nan"))
    graph.execute_plan_at_index(values, workspace, index=selected, handle=handle, tensor_uids=uids)
    torch.cuda.synchronize()
    _assert_result(buffers)
    assert graph._plan_index == selected


@pytest.mark.L1
def test_ordered_deserialized_backend_replaces_the_binding_layout(cudnn_handle):
    restored = cudnn.pygraph(handle=cudnn_handle)
    for uid_offset in (0, 1000):
        source, tensors = _graph(cudnn_handle, "backend", uid_offset=uid_offset)
        restored.deserialize(source.serialize())
        buffers = _buffers(seed=15 + uid_offset)
        workspace = _workspace(restored, cudnn_handle)
        # The old mapping call remains a numerical control for each loaded plan.
        restored.execute(_mapping(tensors, buffers), workspace, handle=cudnn_handle)
        torch.cuda.synchronize()
        _assert_result(buffers)
        expected = buffers["o"].clone(), buffers["lse"].clone()
        values, uids = _ordered(tensors, buffers, reverse=bool(uid_offset))
        for at_index in (False, True):
            buffers["o"].fill_(float("nan"))
            buffers["lse"].fill_(float("nan"))
            if at_index:
                restored.execute_plan_at_index(values, workspace, index=0, handle=cudnn_handle, tensor_uids=uids)
            else:
                restored.execute(values, workspace, handle=cudnn_handle, tensor_uids=uids)
            torch.cuda.synchronize()
            torch.testing.assert_close(buffers["o"], expected[0], atol=0, rtol=0)
            torch.testing.assert_close(buffers["lse"], expected[1], atol=0, rtol=0)


@pytest.mark.L1
def test_ordered_autobound_inputs_caller_precedence_and_unused_uids(attention_case):
    graph, tensors, handle, _ = attention_case
    buffers = _buffers(seed=10)
    # Exercise the binding table populated by tensor-valued operation inputs.
    # An explicit Q must replace its earlier binding; all other inputs may be
    # omitted, and unrelated extra bindings are ignored just as for a mapping.
    previous = dict(graph._data_bindings)
    graph._data_bindings.update({tensors[name].get_uid(): data for name, data in buffers.items() if name not in ("o", "lse")})
    graph._data_bindings[tensors["q"].get_uid()] = torch.zeros_like(buffers["q"])
    supplied = {tensors[name].get_uid(): buffers[name] for name in ("q", "o", "lse")}
    supplied[max(tensor.get_uid() for tensor in tensors.values()) + 12345] = object()
    workspace = _workspace(graph, handle)
    try:
        graph.execute(supplied, workspace, handle=handle)
        torch.cuda.synchronize()
        _assert_result(buffers)
        expected = buffers["o"].clone(), buffers["lse"].clone()
        buffers["o"].fill_(float("nan"))
        buffers["lse"].fill_(float("nan"))
        graph.execute(list(supplied.values()), workspace, handle=handle, tensor_uids=list(supplied))
        torch.cuda.synchronize()
        torch.testing.assert_close(buffers["o"], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(buffers["lse"], expected[1], atol=0, rtol=0)
    finally:
        graph._data_bindings.clear()
        graph._data_bindings.update(previous)


@pytest.mark.L1
def test_ordered_mixes_current_native_and_python_buffer_observation(attention_case):
    class PythonBuffer:
        # A torch-like producer without the native DLPack exchange vtable.
        def __init__(self, tensor):
            self.tensor = tensor
            self.shape, self.dtype, self.device = tensor.shape, tensor.dtype, tensor.device

        def data_ptr(self):
            return self.tensor.data_ptr()

        def stride(self):
            return self.tensor.stride()

        def element_size(self):
            return self.tensor.element_size()

    graph, tensors, handle, _ = attention_case
    workspace = _workspace(graph, handle)
    values, uids = [], []
    for seed in (11, 12):
        buffers = _buffers(seed=seed)
        mixed = dict(buffers, q=PythonBuffer(buffers["q"]), lse=PythonBuffer(buffers["lse"]))
        fresh_values, fresh_uids = _ordered(tensors, mixed)
        values[:], uids[:] = fresh_values, fresh_uids
        graph.execute(values, workspace, handle=handle, tensor_uids=uids)
        torch.cuda.synchronize()
        _assert_result(buffers)


@pytest.mark.L0
def test_ordered_overrides_follow_mutable_values(attention_case):
    graph, tensors, handle, _ = attention_case
    geometry = None
    ordered, uids = [], []
    for iteration, (q_len, kv_len) in enumerate(((_Q, _KV), (16, 32), (_Q, _KV))):
        buffers = _buffers(q_len, kv_len, seed=q_len + iteration)
        current = _overrides(tensors, buffers, q_len, kv_len)
        if iteration == 1:
            for values in current.values():
                values.reverse()
        if geometry is None:
            geometry = current
        else:
            # Same list identities, different contents: caches must compare values.
            geometry["override_uids"][:] = current["override_uids"]
            for old, new in zip(geometry["override_shapes"], current["override_shapes"]):
                old[:] = new
            for old, new in zip(geometry["override_strides"], current["override_strides"]):
                old[:] = new
        workspace = _workspace(graph, handle, geometry)
        graph.execute(_mapping(tensors, buffers), workspace, handle=handle, **geometry)
        torch.cuda.synchronize()
        _assert_result(buffers, q_len, kv_len)
        expected = buffers["o"].clone(), buffers["lse"].clone()
        buffers["o"].fill_(float("nan"))
        buffers["lse"].fill_(float("nan"))
        fresh_ordered, fresh_uids = _ordered(tensors, buffers, reverse=bool(iteration % 2))
        ordered[:], uids[:] = fresh_ordered, fresh_uids
        graph.execute(ordered, workspace, handle=handle, tensor_uids=uids, **geometry)
        torch.cuda.synchronize()
        torch.testing.assert_close(buffers["o"], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(buffers["lse"], expected[1], atol=0, rtol=0)
        _assert_result(buffers, q_len, kv_len)


@pytest.mark.L1
def test_ordered_frost_strided_override_matches_mapping(cudnn_handle):
    # FROST permits token-stride changes on this compiled plan. Exercise that
    # provider capability without assuming every backend plan supports it.
    graph, tensors = _graph(cudnn_handle, "frost")
    buffers = _buffers(16, 32, seed=23, gaps=True)
    geometry = _overrides(tensors, buffers, 16, 32)
    workspace = _workspace(graph, cudnn_handle, geometry)
    graph.execute(_mapping(tensors, buffers), workspace, handle=cudnn_handle, **geometry)
    torch.cuda.synchronize()
    _assert_result(buffers, 16, 32)
    expected = buffers["o"].clone(), buffers["lse"].clone()
    buffers["o"].fill_(float("nan"))
    buffers["lse"].fill_(float("nan"))
    ordered, uids = _ordered(tensors, buffers, reverse=True)
    graph.execute(ordered, workspace, handle=cudnn_handle, tensor_uids=uids, **geometry)
    torch.cuda.synchronize()
    torch.testing.assert_close(buffers["o"], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(buffers["lse"], expected[1], atol=0, rtol=0)
    _assert_result(buffers, 16, 32)


@pytest.mark.L1
def test_ordered_capture_replay_survives_new_bindings_and_workspace(attention_case):
    graph, tensors, handle, _ = attention_case
    buffers = _buffers(seed=12)
    ordered, uids = _ordered(tensors, buffers)
    geometry = _overrides(tensors, buffers)
    workspace = _workspace(graph, handle, geometry)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            cudnn.set_stream(handle, stream.cuda_stream)
            graph.execute(ordered, workspace, handle=handle, tensor_uids=uids, **geometry)
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                graph.execute(ordered, workspace, handle=handle, tensor_uids=uids, **geometry)
        stream.synchronize()
        _assert_result(buffers)
        replacement = _buffers(16, 32, seed=13)
        new_geometry = _overrides(tensors, replacement, 16, 32)
        new_workspace = _workspace(graph, handle, new_geometry)
        new_ordered, new_uids = _ordered(tensors, replacement, reverse=True)
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        graph.execute(new_ordered, new_workspace, handle=handle, tensor_uids=new_uids, **new_geometry)
        torch.cuda.synchronize()
        _assert_result(replacement, 16, 32)
        # New binding/schema calls must not mutate the earlier captured frame.
        with torch.cuda.stream(stream):
            buffers["q"].mul_(0.5)
            buffers["o"].fill_(float("nan"))
            buffers["lse"].fill_(float("nan"))
            capture.replay()
        stream.synchronize()
        _assert_result(buffers)
    finally:
        stream.synchronize()
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)


@pytest.mark.L1
def test_ordered_rebuild_and_selected_plan_do_not_reuse_old_bindings(attention_case):
    graph, tensors, handle, provider = attention_case
    for seed in (20, 21):
        # select_plan invalidates build state even when selecting the same plan.
        # A subsequent build must not retain invocation pointers or old geometry.
        graph.select_plan(graph._plan_index)
        graph.build_plans()
        buffers = _buffers(seed=seed)
        ordered, uids = _ordered(tensors, buffers, reverse=bool(seed % 2))
        workspace = _workspace(graph, handle)
        graph.execute(ordered, workspace, handle=handle, tensor_uids=uids)
        torch.cuda.synchronize()
        _assert_result(buffers)
        assert (graph.selected_engine is None) == (provider == "backend")


@pytest.mark.L1
def test_ordered_provider_selection_changes_on_the_same_graph(cudnn_handle):
    graph, tensors = _graph(cudnn_handle, "frost")
    for seed, provider in enumerate(("frost", "backend", "frost")):
        _select_provider(graph, provider)
        buffers = _buffers(seed=40 + seed)
        ordered, uids = _ordered(tensors, buffers)
        graph.execute(ordered, _workspace(graph, cudnn_handle), handle=cudnn_handle, tensor_uids=uids)
        torch.cuda.synchronize()
        _assert_result(buffers)


@pytest.mark.L1
def test_ordered_handle_and_stream_are_invocation_local(attention_case):
    graph, tensors, _, _ = attention_case
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    handles = [cudnn.create_handle(), cudnn.create_handle()]
    buffers = [_buffers(seed=50 + i) for i in range(2)]
    workspaces = [_workspace(graph, handle) for handle in handles]
    try:
        for stream, handle, data, workspace in zip(streams, handles, buffers, workspaces):
            stream.wait_stream(torch.cuda.current_stream())
            cudnn.set_stream(handle, stream.cuda_stream)
            with torch.cuda.stream(stream):
                data["q"].mul_(0.75)
                ordered, uids = _ordered(tensors, data)
                graph.execute(ordered, workspace, handle=handle, tensor_uids=uids)
        for stream in streams:
            stream.synchronize()
        for data in buffers:
            _assert_result(data)
    finally:
        for stream in streams:
            stream.synchronize()
        for handle in handles:
            cudnn.destroy_handle(handle)


@pytest.mark.parametrize("invalid", ["duplicate", "duplicate_unused", "missing", "count"])
@pytest.mark.L1
def test_ordered_rejects_invalid_uid_bindings_before_launch(attention_case, invalid):
    graph, tensors, handle, _ = attention_case
    buffers = _buffers(seed=25)
    ordered, uids = _ordered(tensors, buffers)
    if invalid == "duplicate":
        uids[-1] = uids[0]
    elif invalid == "duplicate_unused":
        uids.extend([max(uids) + 12345] * 2)
        ordered.extend([object(), object()])
    elif invalid == "missing":
        ordered.pop()
        uids.pop()
    else:
        uids.pop()
    with pytest.raises((ValueError, RuntimeError), match="uid|UID|buffer|length|size|count|operand"):
        graph.execute(ordered, _workspace(graph, handle), handle=handle, tensor_uids=uids)
    assert torch.isnan(buffers["o"]).all()
    assert torch.isnan(buffers["lse"]).all()


@pytest.mark.parametrize("invalid", ["length_storage", "cpu_input", "output_inner_stride"])
@pytest.mark.L1
def test_ordered_preserves_frost_observed_metadata_checks(cudnn_handle, invalid):
    graph, tensors = _graph(cudnn_handle, "frost")
    buffers = _buffers(seed=30)
    if invalid == "length_storage":
        buffers["cu_q"] = buffers["cu_q"][:-1]
    elif invalid == "cpu_input":
        buffers["q"] = buffers["q"].cpu()
    else:
        buffers["o"] = torch.empty(_B * _Q, _HQ, _D * 2, device="cuda", dtype=torch.bfloat16)[..., ::2]
    workspace = _workspace(graph, cudnn_handle)
    # The overload must preserve the existing route's rejection, not accidentally
    # replace observed storage/device/strides with cached declaration metadata.
    for ordered_mode in (False, True):
        values, uids = _ordered(tensors, buffers)
        with pytest.raises(ValueError):
            graph.execute(
                values if ordered_mode else _mapping(tensors, buffers), workspace, handle=cudnn_handle, **({"tensor_uids": uids} if ordered_mode else {})
            )


def _schema():
    native = cudnn._pybind_module
    layout = native.DeclaredLayout(1)
    layout.set(0, [1], [1], 4, 2, 32)
    return native._OrderedBindingSchema([1], layout, False)


@pytest.mark.L0
def test_ordered_schema_reentry_keeps_the_original_snapshot():
    schema = _schema()
    # Retain the old pack even in the buggy version: the regression should
    # raise a deterministic wrong-length error, rather than access freed memory.
    previous = schema.read([None], [1], {}, None, None, None, None)

    class ReentrantUID:
        def __index__(self):
            schema.read([], [], {}, None, None, None, None)
            return 1

    current = schema.read([None], [ReentrantUID()], {}, None, None, None, None)
    assert len(previous[0]) == len(current[0]) == 1


@pytest.mark.L0
def test_ordered_schema_metadata_validation_does_not_depend_on_cache_state():
    schema = _schema()
    for warm in (False, True):
        if warm:
            schema.read([None], [1], {}, None, None, None, None)
        # All three empty lists is valid; supplying only one is a partial
        # override even when it resembles the cached no-override metadata.
        for supplied in range(3):
            overrides = [None, None, None]
            overrides[supplied] = []
            with pytest.raises(ValueError, match="supplied together"):
                schema.read([None], [1], {}, None, *overrides)
        # The accepted row vocabulary must be checked on both misses and hits.
        with pytest.raises(TypeError, match="tuple or list"):
            schema.read([None], [1], {}, None, [1], [range(1, 2)], [[1]])


@pytest.mark.L0
def test_ordered_override_uids_remain_strict():
    schema = _schema()
    for override_uids, shapes, strides in (([999], [[1]], [[1]]), ([1, 1], [[1], [1]], [[1], [1]])):
        with pytest.raises(ValueError, match="operand|Duplicate"):
            schema.read([None], [1], {}, None, override_uids, shapes, strides)
