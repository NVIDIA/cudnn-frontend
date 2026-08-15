# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.set_stream caches the last stream per handle and skips the backend call
(cudnnSetStream, which re-issues several CUDA driver queries every call) when the
stream has not changed. These tests mock the raw backend call, so no GPU is needed."""

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
