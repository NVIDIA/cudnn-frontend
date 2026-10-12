# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unknown capacity is optional; every measured byte count is a real bound.

Host tests only inspect metadata and intercepted launchers. Their synthetic
addresses are never dereferenced. The normalization tests use real CUDA buffers.
"""

from types import SimpleNamespace

import pytest

import cudnn
from cudnn import _pybind_module
from cudnn.engines.base import ExecutionContext, VariantPack
from cudnn.frost.workspace import Workspace, carve_plan

pytestmark = pytest.mark.L0
_ADDRESS = 0x100000


def _pack(nbytes, pointer=_ADDRESS):
    pack = VariantPack((), _pybind_module.VariantPackNative(0), pointer, nbytes)
    pack._device = 0
    return pack


def _raw_pack():
    pointer, capacity = cudnn.pygraph._workspace_extent_fallback(None, _ADDRESS)
    return _pack(capacity, pointer)


@pytest.mark.parametrize("required", [0, 128, 4096])
def test_normalized_raw_workspace_keeps_unknown_capacity(required):
    pack = _raw_pack()
    ws = Workspace.over(pack, required, "raw workspace")
    assert pack.workspace_bytes is None
    assert ws.nbytes is None
    assert ws.view(0, "uint8", (required,)).data_ptr() == _ADDRESS


@pytest.mark.parametrize("capacity", [0, 1, 127, 128, None, 2**33])
@pytest.mark.parametrize("carver", ["view", "take", "native"])
def test_python_and_native_carvers_obey_the_same_capacity(capacity, carver):
    ws = Workspace.over(_pack(capacity), 0, "capacity test")
    native = carve_plan("capacity test", [(0, "float32", (32,))])

    def carve():
        if carver == "view":
            return ws.view(0, "float32", (32,))
        if carver == "take":
            return ws.take(32, "float32")
        return ws.carve(native)[0]

    if capacity is not None and capacity < 128:
        with pytest.raises(ValueError, match="workspace overrun"):
            carve()
    else:
        assert carve().data_ptr() == _ADDRESS


@pytest.mark.parametrize("capacity", [0, 1, 127])
def test_small_workspace_rejects_before_the_first_carve(capacity):
    with pytest.raises(ValueError, match=f"got {capacity} bytes"):
        Workspace.over(_pack(capacity), 128, "capacity test")


def test_empty_and_exhausted_tails_remain_known_empty():
    empty = Workspace.over(_pack(0), 0, "empty workspace").remaining()
    assert empty.shape == (0,)
    assert empty.data_ptr() == _ADDRESS
    ws = Workspace.over(_pack(1), 1, "short workspace")
    ws.take(1, "uint8")
    tail = ws.remaining()
    assert tail.shape == (0,)
    assert tail.data_ptr() == _ADDRESS + 1
    with pytest.raises(ValueError, match="size is unknown"):
        Workspace.over(_raw_pack(), 128, "raw workspace").remaining()


@pytest.mark.parametrize("capacity", [0, None])
def test_empty_native_region_requires_no_capacity(capacity):
    ws = Workspace.over(_pack(capacity), 0, "empty workspace")
    view = ws.carve(carve_plan("empty workspace", [(0, "uint8", (0,))]))[0]
    assert view.numel() == 0


def test_numeric_unknown_sentinels_are_rejected():
    with pytest.raises(ValueError, match="nonnegative or None"):
        _pack(-1)
    with pytest.raises(ValueError, match="nonnegative or None"):
        carve_plan("capacity test", [(0, "uint8", (1,))]).carve(_ADDRESS, -1, 0)


@pytest.mark.parametrize("engine", ["gemm", "linear_attention"])
def test_non_sdpa_adapters_accept_raw_workspace_and_reject_measured_zero(engine):
    calls = []
    regions = carve_plan("non-SDPA workspace", [(0, "float32", (32,))])

    def launch(operands, workspace, stream):
        assert operands == []
        calls.append((workspace.carve(regions)[0].data_ptr(), stream))

    if engine == "gemm":
        from cudnn.gemm.frost.engine import _FrostGemmPlan

        compiled = SimpleNamespace(
            binding=SimpleNamespace(bound_tensors=lambda: []),
            workspace_bytes=128,
            lowered=lambda operands, graph_order, *, stream, workspace: launch(operands, workspace, stream),
        )
        plan = _FrostGemmPlan(compiled)
    else:
        from cudnn.linear_attention.frost.engine import FrostLaPlan

        compiled = SimpleNamespace(device=0, plan_name="capacity test", workspace_size=128, run=launch)
        plan = FrostLaPlan(compiled)
        plan.ports, plan.indices = {}, []  # Binding is already prepared; exercise the actual execute adapter.

    ctx = ExecutionContext(stream=17)
    plan.execute(None, _raw_pack(), ctx)
    assert calls == [(_ADDRESS, 17)]
    with pytest.raises(ValueError, match="got 0 bytes"):
        plan.execute(None, _pack(0), ctx)
    assert len(calls) == 1
    plan.execute(None, _pack(128), ctx)
    assert calls == [(_ADDRESS, 17), (_ADDRESS, 17)]


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("transport", ["tensor", "exchange", "fallback", "raw", "absent"])
def test_graph_normalization_preserves_optional_workspace_capacity(ordered, transport):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("normalization integration needs a CUDA graph handle")
    from cudnn.frost.buffers import DeviceView

    a = torch.empty((1, 16, 16), device="cuda", dtype=torch.float32)
    storage = torch.empty(256, device="cuda", dtype=torch.uint8)
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    aa = graph.tensor_like(a)
    cc = graph.relu(input=aa)
    cc.set_output(True)
    buffers = {aa.get_uid(): a, cc.get_uid(): a}
    for size in (0, 1, 128, 256):
        if transport == "tensor":
            workspace = storage[:size]
        elif transport == "exchange":
            workspace = _pybind_module.make_operand_buffer(storage.data_ptr(), [size], 1, 8, a.device.index)
        elif transport == "fallback":
            workspace = DeviceView(storage.data_ptr(), (size,), "uint8", a.device.index)
        elif transport == "raw":
            workspace = storage.data_ptr()
        else:
            workspace = None
        if ordered:
            pack = graph._normalize_ordered(tuple(buffers.values()), tuple(buffers), workspace)
        else:
            pack = graph._normalize(buffers, workspace)
        expected = None if transport == "raw" else 0 if transport == "absent" else size
        assert pack.workspace_bytes == expected
        if transport in ("raw", "absent"):
            break
