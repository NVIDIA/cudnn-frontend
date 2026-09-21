# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reclaim unrelated FE resources during capture, then run the native backend again."""

import gc
import weakref

import pytest
import torch

import cudnn
from cudnn.frost.buffers import DeviceBuffer

pytestmark = pytest.mark.L0


class _Cycle:
    def __init__(self, resource):
        self.resource = resource
        self.cycle = self


def _native_relu(handle=None):
    graph = cudnn.pygraph(
        handle=handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    # 3-D: the runtime-fusion engine serves this on every arch. A 4-D [1, 1, 32, 32]
    # tensor is refused before SM100 (getDimA()[1] == 1) and only the SM10x-only
    # TensorIR MemBound engine accepted it.
    x = graph.tensor(dim=[1, 32, 32], stride=[1024, 32, 1], data_type=cudnn.data_type.FLOAT)
    y = graph.relu(x).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    assert graph.selected_engine is None and graph._lowered_graph is not None
    return graph, x, y


@pytest.mark.parametrize("resource", ["device_buffer", "owned_backend_graph"])
def test_collect_unrelated_resources_during_capture(resource, cudnn_handle):
    """A GC-timed free must neither invalidate capture nor poison later cuDNN launches.

    This deliberately collects INSIDE the window, rather than avoiding the failure
    by disabling GC or passing relaxed capture mode to torch.
    """
    from cuda.bindings import driver

    graph, tx, ty = _native_relu(cudnn_handle)
    x = torch.linspace(-2, 2, 1024, device="cuda").reshape(1, 32, 32)
    y = torch.empty_like(x)
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    backend_stream = torch.cuda.ExternalStream(cudnn_handle.stream)
    backend_stream.wait_stream(torch.cuda.current_stream())
    graph.execute({tx: x, ty: y}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.relu())

    capture = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    previous_stream = torch.cuda.current_stream()
    stream.wait_stream(previous_stream)
    copied = torch.empty_like(x)
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()  # Keep this specific victim alive until the explicit collection below.
    try:
        if resource == "device_buffer":
            victim = _Cycle(DeviceBuffer(256, torch.cuda.current_device()))
        else:
            # No explicit handle: its lowered PyGraph owns cudnnDestroy as well
            # as the backend graph / execution-plan descriptors.
            victim = _Cycle(_native_relu()[0])
        victim_ref = weakref.ref(victim)
        del victim
        assert victim_ref() is not None
        with torch.cuda.graph(capture, stream=stream, capture_error_mode="global"):
            gc.collect()
            assert victim_ref() is None
            err, status = driver.cuStreamIsCapturing(driver.CUstream(stream.cuda_stream))
            assert int(err) == 0 and status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE
            copied.copy_(x)
        copied.fill_(float("nan"))
        capture.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(copied, x)

        # cudnnDestroy inside a failed capture can leave later native plans
        # unable to launch. Exercise an existing backend handle after collection.
        x.neg_()
        y.fill_(float("nan"))
        backend_stream.wait_stream(torch.cuda.current_stream())
        graph.execute({tx: x, ty: y}, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, x.relu())
    finally:
        torch.cuda.set_stream(previous_stream)
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("free_raises", [False, True])
def test_device_buffer_finalizer_restores_capture_mode(monkeypatch, free_raises):
    """Restore the caller's mode even if the driver binding raises during free."""
    from cuda.bindings import driver

    initial = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
    relaxed = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
    state, freed = [initial], []

    def exchange(mode):
        old, state[0] = state[0], mode
        return driver.CUresult.CUDA_SUCCESS, old

    def free(ptr):
        assert state[0] == relaxed
        freed.append(ptr)
        if free_raises:
            raise RuntimeError("driver teardown")
        return (driver.CUresult.CUDA_SUCCESS,)

    monkeypatch.setattr(driver, "cuThreadExchangeStreamCaptureMode", exchange)
    monkeypatch.setattr(driver, "cuMemFree", free)
    buf = DeviceBuffer.__new__(DeviceBuffer)
    buf._ptr = 256
    try:
        buf.__del__()
        assert freed == [256]
        assert state[0] == initial
    finally:
        buf._ptr = 0
