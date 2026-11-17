# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SDPA harnesses retain the units of their bound ragged offsets."""

import cudnn
import pytest
import torch

from sdpa import fp16, fp8
from sdpa.frost.frost_test_utils import requires_bwd_engine_row, requires_dsl, select_engine
from sdpa.random_config import ExecConfig


def _config(dtype, widths, multiplier, *, train=True, gapped=False, cumulative=False):
    d, dv = widths
    cfg = ExecConfig(
        data_type=dtype,
        rng_data_seed=538,
        rng_geom_seed=538,
        is_infer=not train,
        is_ragged=True,
        is_padding=True,
        is_cu_seq_len=cumulative,
        is_alibi=False,
        is_bias=False,
        is_block_mask=False,
        is_dropout=False,
        is_determin=False,
        batches=3,
        h_q=4,
        h_k=2,
        h_v=2,
        s_q=128,
        s_kv=256,
        d_qk=d,
        d_v=dv,
        seq_len_q=[65, 17, 1],
        seq_len_kv=[193, 33, 1],
        fwd_stats=True,
        with_ragged_offset_multiplier=multiplier,
        with_ragged_token_gap=False,
        with_ragged_head_gap=False,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        implementation=cudnn.attention_implementation.AUTO,
    )
    if gapped:
        # Integral multiples of each width preserve offset-multiplier and
        # vector alignment while giving every port an independent row pitch.
        for port, heads, rows, width, factor in (("q", 4, 128, d, 2), ("k", 2, 256, d, 3), ("v", 2, 256, dv, 4), ("o", 4, 128, dv, 5)):
            pitch = factor * heads * width
            setattr(cfg, "stride_" + port, (rows * pitch, width, pitch, 1))
    cfg.fill_derived_fields()
    return cfg


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("widths", [(64, 128), (128, 64)], ids=["d64-v128", "d128-v64"])
@pytest.mark.parametrize("multiplier", [False, True], ids=["element-offsets", "scaled-offsets"])
def test_half_backward_offset_units_match_forward(monkeypatch, cudnn_handle, dtype, widths, multiplier):
    # Inspect real tensor descriptors, stopping before planning or launch.
    # Incorrect offset units must fail this detector without executing them.
    for method in ("validate", "build_operation_graph", "create_execution_plans", "check_support", "build_plans"):
        monkeypatch.setattr(cudnn.pygraph, method, lambda self, *args, **kwargs: None)
    monkeypatch.setattr(fp16, "note_frost_routing", lambda *args, **kwargs: None)
    cfg = _config(dtype, widths, multiplier, gapped=True)
    stream = cudnn.get_stream(cudnn_handle)
    try:
        forward, _ = fp16.create_forward_graph(cfg, {}, cudnn_handle, plan=False)
        backward, _ = fp16.create_backward_graph(cfg, {}, cudnn_handle, cfg.total_q, cfg.total_kv)
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=stream)

    uid = fp16.TensorUid
    for source, grad, width in ((uid.q, uid.dQ, cfg.d_qk), (uid.k, uid.dK, cfg.d_qk), (uid.v, uid.dV, cfg.d_v), (uid.o, uid.dO, cfg.d_v)):
        expected = width if multiplier else 1
        fwd = forward._tensor_by_uid[int(source)]
        assert fwd.ragged_offset_multiplier == expected
        for role in (source, grad):
            bwd = backward._tensor_by_uid[int(role)]
            assert bwd.ragged_offset.get_uid() == fwd.ragged_offset.get_uid()
            assert bwd.ragged_offset_multiplier == expected, f"{role.name}: backward changed the offset units"
    assert backward._tensor_by_uid[int(uid.stats)].ragged_offset_multiplier == 1


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("gapped", [False, True], ids=["compact", "independent-strides"])
@pytest.mark.parametrize("multiplier", [False, True], ids=["element-offsets", "scaled-offsets"])
def test_half_training_ragged_offset_units(dtype, gapped, multiplier, request, cudnn_handle):
    cfg = _config(dtype, (64, 64), multiplier, gapped=gapped)
    stream = cudnn.get_stream(cudnn_handle)
    try:
        fp16.exec_sdpa(cfg, request, cudnn_handle)
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=stream)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("multiplier", [False, True], ids=["element-offsets", "scaled-offsets"])
def test_fp8_training_ragged_offset_units(monkeypatch, dtype, multiplier, request, cudnn_handle):
    cfg = _config(dtype, (64, 64), multiplier)
    original = fp8.generate_graph_bwd

    def checked_backward(*args, **kwargs):
        graph = original(*args, **kwargs)
        # Check what the execution harness actually passed, before any backward
        # launch. This detects a dropped keyword as well as wrong tensor units.
        for role in ("q", "k", "v", "o", "dO", "dQ", "dK", "dV"):
            tensor = graph._tensor_by_uid[int(getattr(fp8.GraphBwdUid, role))]
            assert tensor.ragged_offset_multiplier == (64 if multiplier else 1), role
        assert graph._tensor_by_uid[int(fp8.GraphBwdUid.stats)].ragged_offset_multiplier == 1
        return graph

    monkeypatch.setattr(fp8, "generate_graph_bwd", checked_backward)
    stream = cudnn.get_stream(cudnn_handle)
    try:
        fp8.exec_sdpa_fp8(cfg, request, cudnn_handle)
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=stream)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("widths", [(64, 128), (128, 64)], ids=["d64-v128", "d128-v64"])
@pytest.mark.parametrize("multiplier", [False, True], ids=["element-offsets", "scaled-offsets"])
def test_fp8_backward_offset_units_match_forward(dtype, widths, multiplier):
    d, dv = widths
    geometry = (dtype, dtype, 3, 4, 2, 2, 128, 256, d, dv, 0.125)
    forward = fp8.generate_graph_fwd(*geometry, block_size=0, is_ragged=True, with_ragged_offset_multiplier=multiplier)
    backward = fp8.generate_graph_bwd(*geometry, deterministic=False, is_ragged=True, with_ragged_offset_multiplier=multiplier)
    for source, grad, width in (("q", "dQ", d), ("k", "dK", d), ("v", "dV", dv), ("o", "dO", dv)):
        expected = width if multiplier else 1
        fwd = forward._tensor_by_uid[int(getattr(fp8.GraphFwdUid, source))]
        assert fwd.ragged_offset_multiplier == expected
        for role in (source, grad):
            bwd = backward._tensor_by_uid[int(getattr(fp8.GraphBwdUid, role))]
            offset_role = "dO" if role == "dO" else source
            assert bwd.ragged_offset.get_uid() == int(getattr(fp8.GraphBwdUid, offset_role + "_ragged_offset"))
            assert bwd.ragged_offset_multiplier == expected, role
    assert backward._tensor_by_uid[int(fp8.GraphBwdUid.stats)].ragged_offset_multiplier == 1


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("gapped", [False, True], ids=["compact", "independent-strides"])
@pytest.mark.parametrize("multiplier", [False, True], ids=["element-offsets", "scaled-offsets"])
def test_half_cumulative_forward_offset_units(dtype, gapped, multiplier, request, cudnn_handle):
    cfg = _config(dtype, (64, 64), multiplier, train=False, gapped=gapped, cumulative=True)
    stream = cudnn.get_stream(cudnn_handle)
    try:
        fp16.exec_sdpa(cfg, request, cudnn_handle)
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=stream)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("declare", [False, True], ids=["implicit-totals", "declared-totals"])
def test_fp8_training_declared_totals(monkeypatch, dtype, declare, request, cudnn_handle):
    cfg = _config(dtype, (64, 64), False)
    cfg.declare_total_seq_len = declare
    original = fp8.generate_graph_bwd

    def checked_backward(*args, **kwargs):
        graph = original(*args, **kwargs)
        assert len(graph.nodes) == 1
        for key, capacity in (("max_total_seq_len_q", cfg.total_q), ("max_total_seq_len_kv", cfg.total_kv)):
            assert graph.nodes[0].params.get(key) == (capacity if declare else None), key
        return graph

    monkeypatch.setattr(fp8, "generate_graph_bwd", checked_backward)
    stream = cudnn.get_stream(cudnn_handle)
    try:
        fp8.exec_sdpa_fp8(cfg, request, cudnn_handle)
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=stream)


@pytest.mark.L0
@requires_dsl
@requires_bwd_engine_row("sdpa_bwd_sm107_fp8")
@pytest.mark.parametrize("widths", [(256, 256), (192, 128)], ids=["d256-v256", "d192-v128"])
@pytest.mark.parametrize("multiplier", [False, True], ids=["element-offsets", "scaled-offsets"])
def test_fp8_ragged_sm107_existing_route(monkeypatch, request, cudnn_handle, widths, multiplier):
    from cudnn.frost.buffers import _cutedsl_has_sm107

    if not _cutedsl_has_sm107():
        pytest.skip("the installed CuTe DSL compiler does not support SM107")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    cfg = _config(torch.float8_e4m3fn, widths, multiplier)
    cfg.declare_total_seq_len = True
    original_builder = fp8.generate_graph_bwd
    original_plans = cudnn.pygraph.create_execution_plans
    original_note = fp8.note_frost_routing
    observed = []

    def builder(*args, **kwargs):
        graph = original_builder(*args, **kwargs)
        for name, expected in [("max_total_seq_len_q", cfg.total_q), ("max_total_seq_len_kv", cfg.total_kv)]:
            assert graph.nodes[0].params[name] == expected
        for role in ("q", "k", "dQ", "dK", "v", "o", "dO", "dV"):
            width = widths[0] if role in ("q", "k", "dQ", "dK") else widths[1]
            assert graph._tensor_by_uid[int(getattr(fp8.GraphBwdUid, role))].ragged_offset_multiplier == (width if multiplier else 1)
        assert graph._tensor_by_uid[int(fp8.GraphBwdUid.stats)].ragged_offset_multiplier == 1
        return graph

    def plans(graph, *args, **kwargs):
        result = original_plans(graph, *args, **kwargs)
        if len(graph.nodes) == 1 and graph.nodes[0].node_type.name == "SDPA_FP8_BWD":
            select_engine(graph, "sdpa_bwd_sm107_fp8")
        return result

    def note(graph, *args, **kwargs):
        if kwargs.get("label") == "fp8-bwd":
            assert graph.selected_engine.name == "sdpa_bwd_sm107_fp8"
            observed.append(graph.get_engine_and_knobs_at_index(graph._plan_index))
        return original_note(graph, *args, **kwargs)

    monkeypatch.setattr(fp8, "generate_graph_bwd", builder)
    monkeypatch.setattr(cudnn.pygraph, "create_execution_plans", plans)
    monkeypatch.setattr(fp8, "note_frost_routing", note)
    stream = cudnn.get_stream(cudnn_handle)
    try:
        fp8.exec_sdpa_fp8(cfg, request, cudnn_handle)
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=stream)
    assert len(observed) == 1
    request.node.user_properties.append(("backward_record", str(observed[0])))
