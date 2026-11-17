# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-owned runtime scalar defaults must remain overridable inputs."""

import ctypes
import struct

import pytest
import torch

import cudnn

pytestmark = pytest.mark.L0


def _graph(raw, constant, record=None):
    cls = cudnn._pybind_module.backend_graph if raw else cudnn.pygraph
    g = cls(io_data_type=cudnn.data_type.FLOAT, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = g.tensor(dim=[1, 1, 16, 16], stride=[256, 256, 16, 1], data_type=cudnn.data_type.FLOAT, name="x").set_uid(3)
    kind = cudnn.scalar_type.COMPILE_TIME_CONST if constant else cudnn.scalar_type.RUNTIME_PARAM
    alpha = g.tensor_scalar(2.0, kind).set_name("alpha").set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT).set_uid(31)
    y = g.mul(a=x, b=alpha, name="scaled_x").set_output(True).set_data_type(cudnn.data_type.FLOAT).set_uid(19)
    g.validate()
    g.build_operation_graph()
    if record is None:
        g.create_execution_plans([cudnn.heur_mode.A])
    else:
        g.create_execution_plan(*record)
    return g, x.get_uid(), alpha.get_uid(), y.get_uid()


@pytest.mark.parametrize("mode", ["raw", "mapping", "ordered"])
@pytest.mark.parametrize("serialized", [False, True], ids=["fresh", "serialized"])
@pytest.mark.parametrize("constant", [False, True], ids=["runtime", "constant"])
def test_scalar_binding_precedence_and_capture(mode, serialized, constant):
    raw = mode == "raw"
    g, x_uid, alpha_uid, y_uid = _graph(raw, constant)
    record = g.get_engine_and_knobs_at_index(0)
    # One pinned native plan makes fresh and serialized executions comparable.
    g, x_uid, alpha_uid, y_uid = _graph(raw, constant, record)
    g.build_plan_at_index(0)
    x = torch.arange(256, device="cuda", dtype=torch.float32).reshape(1, 1, 16, 16)
    y = torch.empty_like(x)
    workspace = torch.empty(max(1, g.get_workspace_size_plan_at_index(0)), device="cuda", dtype=torch.uint8)
    handle = cudnn.create_handle()
    replay = None
    try:
        if serialized:
            payload = g.serialize(as_bytes=True)
            restored = (cudnn._pybind_module.backend_graph if raw else cudnn.pygraph)()
            restored.deserialize(handle.backend_handle if raw else handle, payload, enforce_precompiled=True)
            assert restored.get_execution_plan_count() == 1
            g = restored

        def execute(pack):
            cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
            if raw:
                g._execute_plan_at_index({uid: value.data_ptr() for uid, value in pack.items()}, workspace.data_ptr(), 0, handle.backend_handle)
            elif mode == "ordered":
                g.execute_plan_at_index(tuple(pack.values()), workspace, index=0, tensor_uids=tuple(pack), handle=handle)
            else:
                g.execute_plan_at_index(pack, workspace, index=0, handle=handle)

        for supplied in (None, 3.0, 4.0, None):
            pack = {y_uid: y, x_uid: x}  # Deliberately unsorted; the scalar is last.
            if supplied is not None:
                pack[alpha_uid] = torch.tensor([supplied], dtype=torch.float32)
            expected_scale = 2.0 if constant or supplied is None else supplied
            y.fill_(float("nan"))
            execute(pack)
            torch.testing.assert_close(y, x * expected_scale, rtol=0, atol=0)

        alpha = torch.tensor([3.0], dtype=torch.float32)
        pack = {x_uid: x, alpha_uid: alpha, y_uid: y}
        execute(pack)
        torch.cuda.synchronize()
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):
            execute(pack)
        # Host PBV is captured by value. Device X remains live on replay.
        alpha.fill_(4)
        x.add_(1)
        y.fill_(float("nan"))
        replay.replay()
        torch.testing.assert_close(y, x * (2 if constant else 3), rtol=0, atol=0)
        execute(pack)
        torch.testing.assert_close(y, x * (2 if constant else 4), rtol=0, atol=0)
    finally:
        if replay is not None:
            replay.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize(
    "code,bits,payload",
    [
        (0, 32, struct.pack("=i", -2147483647)),
        (0, 64, struct.pack("=q", 2**53 + 19)),
        (2, 16, struct.pack("=e", 1.5)),
        (4, 16, struct.pack("=H", 0x3FC0)),
        (2, 32, struct.pack("=I", 0x80000000)),
        (2, 64, struct.pack("=d", 1.0 + 2**-40)),
    ],
    ids=["int32", "int64", "half", "bfloat16", "float-negative-zero", "double"],
)
def test_scalar_default_payload_preserves_width_and_bits(code, bits, payload):
    native = cudnn._pybind_module
    # A deserialized graph can have no Python IR layout for this UID.
    schema = native._OrderedBindingSchema([31], native.DeclaredLayout(1), True, {31: (code, bits, payload)})
    pack = native.VariantPackNative(1)
    assert schema.read_mapping(pack, {}) == []
    assert pack.first_unfilled() == -1
    assert ctypes.string_at(pack.pointer(0), bits // 8) == payload
    assert tuple(pack.dtype(0)) == (code, bits)


def test_unread_scalar_override_is_not_replaced_with_default():
    native = cudnn._pybind_module
    schema = native._OrderedBindingSchema([31], native.DeclaredLayout(1), True, {31: (2, 32, struct.pack("=f", 2))})
    pack = native.VariantPackNative(1)
    # A producer without the native exchange vtable goes through the Python
    # observation fallback. Its override must not disappear while unread.
    assert schema.read_mapping(pack, {31: object()}) == [0]
    assert pack.first_unfilled() == 0
