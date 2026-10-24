# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side concurrency invariants for cached linear-attention graphs."""

import contextlib
import threading
from types import SimpleNamespace

import pytest

from cudnn.linear_attention.ops import common

pytestmark = pytest.mark.L0


class _Graph:
    """Minimal built graph exposing only the workspace-size contract."""

    _is_built = True

    def get_workspace_size(self):
        """Return a stable fake workspace requirement."""
        return 256


class _Workspace:
    """Small tensor stand-in used to observe host-side allocations."""

    def __init__(self, size, device):
        """Record the allocation size and device."""
        self._size = size
        self.device = device

    def numel(self):
        """Return the fake allocation size."""
        return self._size


def test_graph_workspace_is_caller_owned_per_execution(monkeypatch):
    """Each invocation receives fresh scratch without retaining it on the graph."""
    device = SimpleNamespace(index=0)
    current = [SimpleNamespace(cuda_stream=11)]
    allocations = []

    monkeypatch.setattr(common.torch.cuda, "current_stream", lambda unused_device: current[0])
    monkeypatch.setattr(common.torch.cuda, "device", lambda unused_device: contextlib.nullcontext())
    monkeypatch.setattr(common.torch.cuda, "stream", lambda unused_stream: contextlib.nullcontext())

    def allocate(size, *, dtype, device):
        """Record each workspace allocation."""
        workspace = _Workspace(size, device)
        allocations.append(workspace)
        return workspace

    monkeypatch.setattr(common.torch, "empty", allocate)

    graph = _Graph()
    first = common.graph_workspace(graph, device)
    again = common.graph_workspace(graph, device)
    current[0] = SimpleNamespace(cuda_stream=22)
    second = common.graph_workspace(graph, device)

    assert first is not again
    assert first is not second
    assert len(allocations) == 3
    assert not hasattr(graph, "_la_ops_workspaces")


def test_cudnn_handles_are_thread_local_and_follow_each_threads_stream(monkeypatch):
    """Concurrent host threads use distinct handles bound to their own streams."""
    device = SimpleNamespace(index=0)
    local = threading.local()
    barrier = threading.Barrier(2)
    creation_lock = threading.Lock()
    created = []
    bindings = []
    results = []

    if hasattr(common, "cudnn_handles"):
        common.cudnn_handles.clear()
    if hasattr(common, "_tls"):
        monkeypatch.setattr(common, "_tls", threading.local())

    monkeypatch.setattr(common.torch.cuda, "current_stream", lambda unused_device: local.stream)
    monkeypatch.setattr(common.torch.cuda, "device", lambda unused_device: contextlib.nullcontext())

    def create_handle():
        """Return a distinct fake handle for each creation."""
        with creation_lock:
            handle = len(created) + 1
            created.append(handle)
            return handle

    monkeypatch.setattr(common.cudnn, "create_handle", create_handle)
    monkeypatch.setattr(common.cudnn, "set_stream", lambda *, handle, stream: bindings.append((handle, stream)))

    def worker(stream):
        """Fetch the same thread-local handle twice on one fake stream."""
        local.stream = SimpleNamespace(cuda_stream=stream)
        barrier.wait()
        first = common.get_handle(device)
        second = common.get_handle(device)
        results.append((stream, first, second))

    threads = [threading.Thread(target=worker, args=(stream,)) for stream in (101, 202)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert all(first == second for _, first, second in results)
    assert len({first for _, first, _ in results}) == 2
    assert {(stream, handle) for stream, handle, _ in results} == {(stream, handle) for handle, stream in bindings}


def test_first_graph_execution_is_serialized(monkeypatch):
    """Only the graph's first lazy execution is protected by its build lock."""
    first_entered = threading.Event()
    second_waiting = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    failures = []

    class TrackingLock:
        """Lock that exposes when a second thread starts waiting."""

        def __init__(self):
            """Initialize the underlying lock and attempt counter."""
            self.lock = threading.Lock()
            self.guard = threading.Lock()
            self.attempts = 0

        def __enter__(self):
            """Record the acquisition attempt before blocking."""
            with self.guard:
                self.attempts += 1
                if self.attempts == 2:
                    second_waiting.set()
            self.lock.acquire()

        def __exit__(self, *unused):
            """Release the underlying lock."""
            self.lock.release()

    class Graph:
        """Fake graph whose first execution blocks until the test releases it."""

        def __init__(self):
            """Initialize execution accounting and the observable lock."""
            self.calls = 0
            self._la_ops_execution_lock = TrackingLock()

        def execute(self, unused_variant_pack, *, workspace, handle):
            """Block the first call and record entry into the second."""
            assert workspace is sentinel_workspace
            assert handle is sentinel_handle
            self.calls += 1
            if self.calls == 1:
                first_entered.set()
                assert release_first.wait(timeout=5)
            else:
                second_entered.set()

    sentinel_workspace = object()
    sentinel_handle = object()

    graph = Graph()

    def worker():
        """Execute the graph and preserve failures from this host thread."""
        try:
            common.execute_graph(graph, {}, sentinel_workspace, sentinel_handle)
        except BaseException as exc:  # noqa: BLE001 -- preserve thread failures
            failures.append(exc)

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    try:
        assert second_waiting.wait(timeout=5)
        assert not second_entered.is_set(), "the second execute overlapped the first graph initialization"
    finally:
        release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert not failures
    assert graph.calls == 2
    assert graph._la_ops_execution_ready
