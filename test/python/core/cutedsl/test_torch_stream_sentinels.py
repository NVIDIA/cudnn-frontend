# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject PTDS before touching torch's allocator or taking the current-stream shortcut.

A raw value of 2 names different CUDA streams in different host threads. Treating
it as torch's legacy stream (or a process-wide allocator identity) is unsafe.
These are host-side validation tests; native side-stream lifetime coverage lives
in test_torch_stream_staging.py and the DSA API suites.
"""

import pytest
import torch
from cuda.bindings import driver as cuda

from cudnn._torch_stream import as_torch_stream, launch_stream, stream_context

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("stream", [2, cuda.CUstream(2)])
@pytest.mark.parametrize("entrypoint", ["as_torch_stream", "launch_stream", "stream_context", "verified_stream_context"])
def test_per_thread_sentinel_is_rejected_before_torch_stream_queries(monkeypatch, stream, entrypoint):
    def refuse(*args, **kwargs):
        raise AssertionError("PTDS must be rejected before querying or wrapping a torch stream")

    for name in ("default_stream", "current_stream", "ExternalStream"):
        monkeypatch.setattr(torch.cuda, name, refuse)
    # Even when the raw fast path would match, handle 2 must not bypass validation.
    monkeypatch.setattr("cudnn._torch_stream._raw_current_stream", lambda *args: 2)
    with pytest.raises(ValueError, match="cudaStreamPerThread"):
        if entrypoint == "as_torch_stream":
            as_torch_stream(stream, 0)
        elif entrypoint == "launch_stream":
            launch_stream(stream, 0)
        else:
            with stream_context(stream, 0, verify_current=entrypoint == "verified_stream_context"):
                raise AssertionError("PTDS context must not be entered")


def test_implicit_per_thread_current_stream_is_rejected(monkeypatch):
    monkeypatch.setattr("cudnn._torch_stream._raw_current_stream", lambda *args: 2)
    with pytest.raises(ValueError, match="cudaStreamPerThread"):
        launch_stream(None, 0)


def test_implicit_stream_records_source_lifetime(monkeypatch):
    from cudnn._torch_stream import record_streams

    consumer = object()
    recorded = []

    class Tensor:
        is_cuda = True
        device = torch.device("cuda", 0)

        def record_stream(self, stream):
            recorded.append(stream)

    def current_launch_stream(stream, device):
        assert stream is None
        assert device == torch.device("cuda", 0)
        return consumer

    monkeypatch.setattr("cudnn._torch_stream.launch_stream", current_launch_stream)
    record_streams((None, Tensor()), None)
    assert recorded == [consumer]
