# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.set_stream remembers the last stream ON a cudnn.Handle and skips the backend
call (cudnnSetStream, which re-issues several CUDA driver queries every call) when the
stream has not changed. destroy_handle clears a Handle's backend handle so a released
cudnnHandle_t cannot be handed back to C++. The handle APIs take a cudnn.Handle only --
a raw backend int is rejected. These tests mock the raw backend calls, so no GPU is needed."""

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


def test_raw_int_handle_rejected():
    # The Python API creates handles only via cudnn.create_handle() (-> Handle), so a
    # raw backend int is a mistake and is rejected rather than silently opting out of
    # the Handle's device/stream tracking.
    for call in (
        lambda: cudnn.set_stream(handle=1, stream=100),
        lambda: cudnn.get_stream(1),
        lambda: cudnn.destroy_handle(1),
    ):
        with pytest.raises(TypeError, match="cudnn.Handle"):
            call()


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
