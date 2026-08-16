# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.set_stream remembers the last stream ON a cudnn.Handle and skips the backend
call (cudnnSetStream, which re-issues several CUDA driver queries every call) when the
stream has not changed. A foreign raw-int handle is not ours to track -- its owner may
call cudnnSetStream out-of-band -- so it is always set through. destroy_handle clears a
Handle's backend handle so a released cudnnHandle_t cannot be handed back to C++. These
tests mock the raw backend calls, so no GPU is needed."""

import pytest

import cudnn

pytestmark = pytest.mark.L0


def test_handle_skips_backend_call_when_unchanged(monkeypatch):
    streams = []
    monkeypatch.setattr(cudnn._pybind_module, "_raw_set_stream", lambda h, s: streams.append(s))

    h = cudnn.Handle(backend_handle=0xABC)  # no GPU: only forwarded to the mocked backend call
    cudnn.set_stream(h, 100)
    cudnn.set_stream(h, 100)  # unchanged -> skipped (fast path lives on the object)
    cudnn.set_stream(h, 200)  # changed -> forwarded

    assert streams == [100, 200]
    assert h.stream == 200
    assert cudnn.get_stream(h) == 200  # read comes from the object, no backend query


def test_foreign_int_always_sets_through(monkeypatch):
    # A foreign raw-int handle has no object we own; its owner may change the stream
    # out-of-band, so we never skip on a cached value -- every call forwards.
    calls = []
    monkeypatch.setattr(cudnn._pybind_module, "_raw_set_stream", lambda h, s: calls.append((h, s)))

    cudnn.set_stream(handle=1, stream=100)
    cudnn.set_stream(handle=1, stream=100)  # same stream, but foreign -> NOT skipped
    cudnn.set_stream(handle=2, stream=100)

    assert calls == [(1, 100), (1, 100), (2, 100)]


def test_destroy_handle_clears_backend_handle(monkeypatch):
    destroyed = []
    monkeypatch.setattr(cudnn._pybind_module, "_raw_set_stream", lambda h, s: None)
    monkeypatch.setattr(cudnn._pybind_module, "_raw_destroy_handle", lambda h: destroyed.append(h))

    h = cudnn.Handle(backend_handle=0x777)
    cudnn.set_stream(h, 100)
    cudnn.destroy_handle(h)

    assert h.backend_handle is None  # cleared so a reused Handle cannot pass a released handle to C++
    assert h.stream is None
    assert destroyed == [0x777]

    # A double-destroy / a later set_stream must not touch the released handle.
    cudnn.destroy_handle(h)
    cudnn.set_stream(h, 200)
    assert destroyed == [0x777]  # no second destroy call


def test_destroy_foreign_int_forwards_to_backend(monkeypatch):
    destroyed = []
    monkeypatch.setattr(cudnn._pybind_module, "_raw_destroy_handle", lambda h: destroyed.append(h))

    cudnn.destroy_handle(99)  # foreign raw-int handle -> destroyed directly

    assert destroyed == [99]


def test_handle_and_foreign_int_are_independent(monkeypatch):
    monkeypatch.setattr(cudnn._pybind_module, "_raw_set_stream", lambda h, s: None)

    h = cudnn.Handle(backend_handle=42)
    cudnn.set_stream(h, 100)  # Handle path -> stored on the object
    cudnn.set_stream(42, 100)  # foreign int 42 -> forwarded, independent of the Handle object

    assert h.stream == 100
    assert cudnn.get_stream(h) == 100  # from the object, not any shared registry
