# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SDPA operands must meet the alignment the frontend declares (#1571).

Every tensor declares a 16-byte alignment by default, and the native SDPA kernels
rely on it for their vector loads and TMA. The backend checks neither the row
strides nor the caller's pointers. Without the frontend checks, a misaligned
operand is silently wrong on SM90/SM100 and faults on SM80/SM89, where
`CUDA error: misaligned address` is sticky and ends the CUDA context. These tests
pin the frontend contract:
- a row stride that breaks the alignment is declined when the graph is lowered;
- a misaligned pointer is refused before anything launches.

The layout is BSHD with a row pitch of ``pitch`` elements, so ``pitch`` != 64 pads
the rows and the base can be shifted by ``off`` elements.
"""

import math

import pytest
import torch

import cudnn
from cudnn._pygraph import pygraph
from cudnn.engines import is_python_engine

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

pytestmark = pytest.mark.L0

B, H, S, D = 2, 4, 128, 64
SCALE = 1.0 / math.sqrt(D)


def _storage(pitch, off=0, fill=None):
    store = torch.randn(B * S * H * pitch + off, dtype=torch.float16, device="cuda")
    view = store[off:].as_strided((B, H, S, D), (S * H * pitch, pitch, H * pitch, 1))
    if fill is not None:
        view.fill_(fill)
    return view


def _strides(pitch):
    return [S * H * pitch, pitch, H * pitch, 1]


def _pin_native(g):
    native = [i for i, plan in enumerate(g.plans) if not is_python_engine(plan.engine_id)]
    if not native:
        pytest.skip("no native backend plan claims this graph on this part")
    g.select_plan(native[0])
    try:
        g.check_support()
    except cudnn.cudnnGraphNotSupportedError as error:
        pytest.skip(f"native plan does not support this graph: {error}")
    g.build_plans()


def _forward_graph(pitch, **graph_kwargs):
    g = pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, **graph_kwargs)
    q, k, v = (g.tensor(name=n, dim=[B, H, S, D], stride=_strides(pitch), data_type=cudnn.data_type.HALF) for n in "qkv")
    o, _ = g.sdpa(q, k, v, is_inference=True, attn_scale=SCALE)
    o.set_output(True).set_data_type(cudnn.data_type.HALF).set_dim([B, H, S, D]).set_stride(_strides(pitch))
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    return g, q, k, v, o


def _backward_graph(pitch):
    g = pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v, o, do = (g.tensor(name=n, dim=[B, H, S, D], stride=_strides(pitch), data_type=cudnn.data_type.HALF) for n in ("q", "k", "v", "o", "do"))
    stats = g.tensor(name="stats", dim=[B, H, S, 1], stride=[H * S, S, 1, 1], data_type=cudnn.data_type.FLOAT)
    dq, dk, dv = g.sdpa_backward(q, k, v, o, do, stats, attn_scale=SCALE)
    for t in (dq, dk, dv):
        t.set_output(True).set_data_type(cudnn.data_type.HALF).set_dim([B, H, S, D]).set_stride(_strides(pitch))
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    return g, (q, k, v, o, do, stats, dq, dk, dv)


def _reference(q, k, v):
    p = torch.softmax((q.double() @ k.double().transpose(-1, -2)) * SCALE, dim=-1)
    return (p @ v.double()).float()


def _run_forward(g, ports, bufs):
    workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
    g.execute(dict(zip(ports, bufs)), workspace)
    torch.cuda.synchronize()


def test_padded_pitch_and_aligned_offset_are_served():
    """Control: a padded pitch and a shifted base that keep 16-byte alignment still run."""
    g, *ports = _forward_graph(72)
    _pin_native(g)
    q, k, v = (_storage(72, off=8) for _ in range(3))
    o = _storage(72, off=8, fill=float("nan"))
    _run_forward(g, ports, (q, k, v, o))
    torch.testing.assert_close(o.float(), _reference(q, k, v), atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_row_pitch_off_the_alignment_is_declined_by_the_native_lowering(direction):
    """A 136-byte row pitch is not a multiple of 16; no native plan may claim it.

    A python engine (FROST on SM100) may still serve the graph; only the native
    plans are judged here.
    """
    try:
        g = _forward_graph(68)[0] if direction == "forward" else _backward_graph(68)[0]
    except Exception as error:  # noqa: BLE001 -- the graph-level error names the native decline
        assert "stride 68 of dimension 1 is 136 bytes" in str(error), str(error)
        return
    assert all(is_python_engine(plan.engine_id) for plan in g.plans), [plan.engine_id for plan in g.plans]


def test_misaligned_base_pointer_is_refused_before_launch():
    g, *ports = _forward_graph(64)
    _pin_native(g)
    q = _storage(64, off=4)  # 8 bytes past a 16-byte boundary
    k, v = _storage(64), _storage(64)
    o = _storage(64, fill=float("nan"))
    with pytest.raises(Exception, match="not aligned to the tensor's 16-byte alignment"):
        _run_forward(g, ports, (q, k, v, o))
    assert torch.isnan(o).all(), "the kernel ran on a refused variant pack"
    # Refused before launch: the context is intact and the same plan still serves aligned buffers.
    q = _storage(64)
    _run_forward(g, ports, (q, k, v, o))
    torch.testing.assert_close(o.float(), _reference(q, k, v), atol=5e-3, rtol=1e-2)


def test_backward_misaligned_pointer_is_refused_before_launch():
    g, ports = _backward_graph(64)
    _pin_native(g)
    q, k, v, o, do = (_storage(64) for _ in range(5))
    stats = torch.zeros(B, H, S, 1, dtype=torch.float32, device="cuda")
    dq, dk = _storage(64), _storage(64)
    dv = _storage(64, off=4)
    workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
    with pytest.raises(Exception, match="not aligned to the tensor's 16-byte alignment"):
        g.execute(dict(zip(ports, (q, k, v, o, do, stats, dq, dk, dv))), workspace)
    torch.cuda.synchronize()


def test_cuda_graph_populate_and_update_refuse_a_misaligned_pointer():
    """populate_cuda_graph()/update_cuda_graph() build their own variant pack; they refuse before recording."""
    cudart = pytest.importorskip("cuda.bindings.runtime")
    g, *ports = _forward_graph(64)
    _pin_native(g)
    k, v = _storage(64), _storage(64)
    o = _storage(64, fill=float("nan"))
    workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")

    def pack(q):
        return {t.get_uid(): buf.data_ptr() for t, buf in zip(ports, (q, k, v, o))}

    handle = cudnn.create_handle()
    err, cuda_graph = cudart.cudaGraphCreate(0)
    assert err == cudart.cudaError_t.cudaSuccess
    graph_exec = None
    try:
        with pytest.raises(Exception, match="not aligned to the tensor's 16-byte alignment"):
            g.populate_cuda_graph(handle, pack(_storage(64, off=4)), workspace.data_ptr(), int(cuda_graph))
        # Nothing was recorded: populate requires an empty graph, and this one still is.
        q = _storage(64)
        g.populate_cuda_graph(handle, pack(q), workspace.data_ptr(), int(cuda_graph))
        with pytest.raises(Exception, match="not aligned to the tensor's 16-byte alignment"):
            g.update_cuda_graph(handle, pack(_storage(64, off=4)), workspace.data_ptr(), int(cuda_graph))
        # The refused update left the aligned recording in place.
        err, graph_exec = cudart.cudaGraphInstantiate(cuda_graph, 0)
        assert err == cudart.cudaError_t.cudaSuccess
        torch.cuda.synchronize()
        assert cudart.cudaGraphLaunch(graph_exec, 0)[0] == cudart.cudaError_t.cudaSuccess
        torch.cuda.synchronize()
        torch.testing.assert_close(o.float(), _reference(q, k, v), atol=5e-3, rtol=1e-2)
    finally:
        if graph_exec is not None:
            cudart.cudaGraphExecDestroy(graph_exec)
        cudart.cudaGraphDestroy(cuda_graph)
        cudnn.destroy_handle(handle)


def test_serialized_graph_keeps_the_pointer_check(cudnn_handle):
    g, *ports = _forward_graph(64)
    _pin_native(g)
    restored = cudnn.pygraph()
    restored.deserialize(cudnn_handle, g.serialize())
    q = _storage(64, off=4)
    k, v = _storage(64), _storage(64)
    o = _storage(64, fill=float("nan"))
    workspace = torch.empty(max(int(restored.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
    with pytest.raises(Exception, match="not aligned to the tensor's 16-byte alignment"):
        restored.execute(dict(zip(ports, (q, k, v, o))), workspace, handle=cudnn_handle)
    torch.cuda.synchronize()


def test_override_stride_off_the_alignment_is_refused_before_launch():
    """The build-time decline only sees the declared strides; an override is checked at execute."""
    g, *ports = _forward_graph(64, is_override_shape_enabled=True)
    _pin_native(g)
    q, k, v = (_storage(68) for _ in range(3))
    o = _storage(68, fill=float("nan"))
    overrides = dict(override_uids=[t.get_uid() for t in ports], override_shapes=[[B, H, S, D]] * 4, override_strides=[_strides(68)] * 4)
    workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
    with pytest.raises(Exception, match="override stride 68 of dimension 1 is 136 bytes"):
        g.execute(dict(zip(ports, (q, k, v, o))), workspace, **overrides)
    torch.cuda.synchronize()
    assert torch.isnan(o).all(), "the kernel ran on a refused variant pack"


_WIDTH = D + 8


def _legacy_slice_graph(start):
    """Q is the slice [start, start + D) of the last dimension of a wider ``q_full``."""
    g = pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q_full = g.tensor(name="q_full", dim=[B, H, S, _WIDTH], stride=_strides(_WIDTH), data_type=cudnn.data_type.HALF)
    q = g.slice(q_full, [slice(None), slice(None), slice(None), slice(start, start + D)])
    k, v = (g.tensor(name=n, dim=[B, H, S, D], stride=_strides(_WIDTH), data_type=cudnn.data_type.HALF) for n in "kv")
    o, _ = g.sdpa(q, k, v, is_inference=True, attn_scale=SCALE)
    o.set_output(True).set_data_type(cudnn.data_type.HALF).set_dim([B, H, S, D]).set_stride(_strides(_WIDTH))
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    _pin_native(g)
    return g, (q_full, k, v, o)


_legacy_slice = pytest.mark.skipif(cudnn.backend_version() >= 92200, reason="Slice is lowered by pointer offset only before cuDNN 9.22")


@_legacy_slice
def test_legacy_slice_offset_into_an_operand_is_checked():
    """Q is a slice starting 4 elements into an aligned buffer: the caller's pointer is aligned, Q's is not."""
    g, ports = _legacy_slice_graph(4)
    q_buf = torch.randn(B, S, H, _WIDTH, dtype=torch.float16, device="cuda").permute(0, 2, 1, 3)
    k_buf, v_buf = (_storage(_WIDTH) for _ in range(2))
    o_buf = _storage(_WIDTH, fill=float("nan"))
    with pytest.raises(Exception, match="not aligned to the tensor's 16-byte alignment"):
        _run_forward(g, ports, (q_buf, k_buf, v_buf, o_buf))
    assert torch.isnan(o_buf).all(), "the kernel ran on a refused variant pack"


@_legacy_slice
def test_legacy_slice_execute_matches_the_reference():
    """execute() hands the backend the sliced pointer only; the source uid is not part of the backend graph."""
    g, ports = _legacy_slice_graph(8)
    q_buf = torch.randn(B, S, H, _WIDTH, dtype=torch.float16, device="cuda").permute(0, 2, 1, 3)
    k_buf, v_buf = (_storage(_WIDTH) for _ in range(2))
    o_buf = _storage(_WIDTH, fill=float("nan"))
    _run_forward(g, ports, (q_buf, k_buf, v_buf, o_buf))
    torch.testing.assert_close(o_buf.float(), _reference(q_buf[..., 8 : 8 + D], k_buf, v_buf), atol=5e-3, rtol=1e-2)


@_legacy_slice
def test_legacy_slice_cuda_graph_binds_and_checks_the_sliced_pointer():
    """populate_cuda_graph() applies the Slice offset to the caller's source pointer, as execute() does."""
    cudart = pytest.importorskip("cuda.bindings.runtime")
    q_buf = torch.randn(B, S, H, _WIDTH, dtype=torch.float16, device="cuda").permute(0, 2, 1, 3)
    k_buf, v_buf = (_storage(_WIDTH) for _ in range(2))
    o_buf = _storage(_WIDTH, fill=float("nan"))
    handle = cudnn.create_handle()
    graphs, graph_exec = [], None
    try:
        for start in (4, 8):
            g, ports = _legacy_slice_graph(start)
            workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
            err, cuda_graph = cudart.cudaGraphCreate(0)
            assert err == cudart.cudaError_t.cudaSuccess
            graphs.append(cuda_graph)
            uid_to_ptr = {t.get_uid(): buf.data_ptr() for t, buf in zip(ports, (q_buf, k_buf, v_buf, o_buf))}
            if start == 4:
                with pytest.raises(Exception, match="not aligned to the tensor's 16-byte alignment"):
                    g.populate_cuda_graph(handle, uid_to_ptr, workspace.data_ptr(), int(cuda_graph))
                continue
            g.populate_cuda_graph(handle, uid_to_ptr, workspace.data_ptr(), int(cuda_graph))
            err, graph_exec = cudart.cudaGraphInstantiate(cuda_graph, 0)
            assert err == cudart.cudaError_t.cudaSuccess
            torch.cuda.synchronize()
            assert cudart.cudaGraphLaunch(graph_exec, 0)[0] == cudart.cudaError_t.cudaSuccess
            torch.cuda.synchronize()
            torch.testing.assert_close(o_buf.float(), _reference(q_buf[..., start : start + D], k_buf, v_buf), atol=5e-3, rtol=1e-2)
    finally:
        if graph_exec is not None:
            cudart.cudaGraphExecDestroy(graph_exec)
        for cuda_graph in graphs:
            cudart.cudaGraphDestroy(cuda_graph)
        cudnn.destroy_handle(handle)
