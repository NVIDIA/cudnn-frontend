# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fluent ``cudnn.Graph`` wrapper (python/cudnn/wrapper.py)."""

import pytest
import torch

import cudnn

pytestmark = pytest.mark.skipif(cudnn.backend_version() < 91200, reason="fluent Graph requires cuDNN >= 9.12")


def _matmul_graph(**kwargs):
    """A 64x64 half matmul through the fluent wrapper."""
    with cudnn.Graph(
        handle="auto",
        io_data_type=cudnn.data_type.HALF,
        compute_data_type=cudnn.data_type.FLOAT,
        inputs=["X", "W"],
        outputs=["Y"],
        **kwargs,
    ) as graph:
        X = graph.tensor(name="X", dim=[1, 64, 64], stride=[64 * 64, 64, 1])
        W = graph.tensor(name="W", dim=[1, 64, 64], stride=[64 * 64, 64, 1])
        Y = graph.matmul(name="mm", A=X, B=W)
        Y.set_output(True).set_name("Y")
    return graph


@pytest.mark.L0
def test_workspace_alloc_default_allocates():
    """The default path allocates a workspace the caller never has to think about."""
    graph = _matmul_graph()
    assert torch.is_tensor(graph._Graph__workspace)


@pytest.mark.L0
def test_workspace_alloc_false_is_honored():
    """``workspace_alloc=False`` means the CALLER owns the workspace.

    Regression: the sentinel is written as ``self.__workspace`` (mangled to
    ``_Graph__workspace``) but was read back with ``hasattr(self, "__workspace")``
    — a plain string, which is NOT name-mangled. That probe was therefore always
    False, the sentinel was overwritten with a fresh allocation on every
    ``__exit__``, and the "Need to specify workspace" guard below was unreachable.
    """
    graph = _matmul_graph(workspace_alloc=False)
    assert graph._Graph__workspace is False

    x = torch.randn(1, 64, 64, dtype=torch.half, device="cuda")
    w = torch.randn(1, 64, 64, dtype=torch.half, device="cuda")

    with pytest.raises(RuntimeError, match="Need to specify workspace"):
        graph(x, w)

    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    out = graph(x, w, workspace=workspace)
    torch.testing.assert_close(out.float(), (x @ w).float(), atol=1e-2, rtol=1e-2)


@pytest.mark.L0
def test_auto_handle_is_per_device():
    """A second GPU must not execute with the first GPU's cuDNN handle."""
    from cudnn import wrapper

    first = torch.cuda.current_device()
    supported = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(i)[0] >= 8]
    if first not in supported or len(supported) < 2:
        pytest.skip("requires two SM80+ CUDA devices")
    second = next(i for i in supported if i != first)
    handles = {}
    for device in (first, second, first):
        with torch.cuda.device(device):
            graph = _matmul_graph()
            x = torch.randn(1, 64, 64, dtype=torch.half, device="cuda")
            w = torch.randn_like(x)
            out = graph(x, w)
            torch.testing.assert_close(out, x @ w, atol=1e-2, rtol=1e-2)
            handle = wrapper.get_default_handle()
            assert handle.device.ordinal == device
            if device in handles:
                assert handle is handles[device]
            handles[device] = handle
    assert handles[first] is not handles[second]
    assert torch.cuda.current_device() == first


@pytest.mark.L0
def test_auto_handle_is_per_thread():
    """Re-streaming one thread's auto handle must leave another's untouched."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from cudnn import wrapper

    device = torch.cuda.current_device()
    barrier = threading.Barrier(2, timeout=20)

    def worker():
        with torch.cuda.device(device):
            stream = torch.cuda.Stream()
            handle = wrapper.get_default_handle(stream.cuda_stream)
            barrier.wait()
            return handle, stream, cudnn.get_stream(handle)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert results[0][0] is not results[1][0]
    for _handle, stream, observed in results:
        assert observed == stream.cuda_stream


@pytest.mark.L0
def test_auto_handle_cleanup_allows_recreation():
    from cudnn import wrapper

    original = wrapper.get_default_handle()
    wrapper.destroy_default_handle()
    wrapper.destroy_default_handle()  # cleanup is idempotent
    assert original.backend_handle is None
    replacement = wrapper.get_default_handle()
    assert replacement is not original and replacement.backend_handle is not None
    graph = _matmul_graph()
    x = torch.randn(1, 64, 64, dtype=torch.half, device="cuda")
    w = torch.randn_like(x)
    torch.testing.assert_close(graph(x, w), x @ w, atol=1e-2, rtol=1e-2)


@pytest.mark.L0
def test_auto_handle_accepts_stream_object():
    from cudnn import wrapper

    current_device = torch.cuda.current_device()
    device = next((i for i in range(torch.cuda.device_count()) if i != current_device), current_device)
    stream = torch.cuda.Stream(device=device)
    handle = wrapper.get_default_handle(stream)
    assert handle.device.ordinal == device
    assert cudnn.get_stream(handle) == stream.cuda_stream
    assert torch.cuda.current_device() == current_device
    with torch.cuda.device(device):
        assert wrapper.get_default_handle(stream.cuda_stream) is handle


@pytest.mark.L0
def test_auto_handle_cleanup_uses_creation_device(monkeypatch):
    """Cleanup selects each cached handle's GPU and restores the ambient device."""
    from cudnn import wrapper

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    current = torch.cuda.current_device()
    other = next(i for i in range(torch.cuda.device_count()) if i != current)
    handles = [wrapper.get_default_handle()]
    with torch.cuda.device(other):
        handles.append(wrapper.get_default_handle())
    destroy = cudnn.destroy_handle

    def checked_destroy(handle):
        assert torch.cuda.current_device() == handle.device.ordinal
        destroy(handle)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(cudnn, "destroy_handle", checked_destroy)
            wrapper.destroy_default_handle()
        assert all(handle.backend_handle is None for handle in handles)
        assert torch.cuda.current_device() == current
    finally:
        for handle in handles:
            with torch.cuda.device(handle.device.ordinal):
                destroy(handle)


@pytest.mark.L0
def test_auto_handle_cleanup_retries_failed_handle(monkeypatch):
    """A failed destroy remains retryable while other handles are still released."""
    from cudnn import wrapper

    handles = [cudnn.create_handle(), cudnn.create_handle()]
    destroy = cudnn.destroy_handle
    calls = []

    def fail_once(handle):
        calls.append(handle)
        if len(calls) == 1:
            raise RuntimeError("injected destroy failure")
        destroy(handle)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(wrapper, "_default_handle_registry", handles.copy())
            patch.setattr(cudnn, "destroy_handle", fail_once)
            with pytest.raises(RuntimeError, match="injected destroy failure"):
                wrapper.destroy_default_handle()
            assert calls == handles
            assert handles[0].backend_handle is not None
            assert handles[1].backend_handle is None
            wrapper.destroy_default_handle()
            assert calls == [*handles, handles[0]]
            assert handles[0].backend_handle is None
    finally:
        for handle in handles:
            destroy(handle)
