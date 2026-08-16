# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.set_stream remembers the last stream per handle and skips the backend call
(cudnnSetStream, which re-issues several CUDA driver queries every call) when the
stream has not changed. A cudnn.Handle remembers on itself; a foreign raw-int handle
uses the module registry. These tests mock the raw backend call, so no GPU is needed."""

import pytest

import cudnn

pytestmark = pytest.mark.L0


def test_set_stream_skips_backend_call_when_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(cudnn._pybind_module, "_raw_set_stream", lambda h, s: calls.append((h, s)))
    cudnn._handle_to_stream.clear()

    cudnn.set_stream(handle=1, stream=100)
    cudnn.set_stream(handle=1, stream=100)  # unchanged -> skipped
    cudnn.set_stream(handle=1, stream=200)  # changed -> forwarded
    cudnn.set_stream(handle=2, stream=100)  # different handle -> forwarded

    assert calls == [(1, 100), (1, 200), (2, 100)]


def test_destroy_handle_forgets_cached_stream(monkeypatch):
    calls = []
    monkeypatch.setattr(cudnn._pybind_module, "_raw_set_stream", lambda h, s: calls.append((h, s)))
    monkeypatch.setattr(cudnn._pybind_module, "_raw_destroy_handle", lambda h: None)
    cudnn._handle_to_stream.clear()

    cudnn.set_stream(handle=7, stream=100)
    cudnn.destroy_handle(7)
    cudnn.set_stream(handle=7, stream=100)  # a reused handle address must re-arm the backend

    assert calls == [(7, 100), (7, 100)]


def test_handle_carries_its_own_stream_not_the_registry(monkeypatch):
    # A cudnn.Handle remembers its stream on the object; the module registry stays untouched.
    streams = []
    monkeypatch.setattr(cudnn._pybind_module, "_raw_set_stream", lambda h, s: streams.append(s))
    cudnn._handle_to_stream.clear()

    h = cudnn.Handle(backend_handle=0xABC)  # no GPU: only forwarded to the mocked backend call
    cudnn.set_stream(h, 100)
    cudnn.set_stream(h, 100)  # unchanged -> skipped
    cudnn.set_stream(h, 200)  # changed -> forwarded

    assert streams == [100, 200]
    assert h.stream == 200
    assert cudnn.get_stream(h) == 200  # read comes from the object, no backend query
    assert cudnn._handle_to_stream == {}  # a Handle never touches the foreign-int registry


def test_handle_and_foreign_int_do_not_collide(monkeypatch):
    monkeypatch.setattr(cudnn._pybind_module, "_raw_set_stream", lambda h, s: None)
    monkeypatch.setattr(cudnn._pybind_module, "_raw_destroy_handle", lambda h: None)
    cudnn._handle_to_stream.clear()

    h = cudnn.Handle(backend_handle=42)
    cudnn.set_stream(h, 100)  # Handle path -> stored on the object
    cudnn.set_stream(42, 100)  # foreign int 42 -> registry, independent of the Handle

    assert h.stream == 100
    assert cudnn._handle_to_stream == {42: 100}

    cudnn.destroy_handle(h)
    assert h.stream is None
    assert cudnn._handle_to_stream == {42: 100}  # destroying the Handle leaves the int registry alone
