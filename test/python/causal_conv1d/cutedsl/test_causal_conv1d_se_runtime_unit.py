# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource-query failures and compatibility paths must not masquerade as full-GPU schedules."""

from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch

pytestmark = pytest.mark.L0


def _runtime():
    from cudnn.causal_conv1d_se import runtime

    return runtime


def _driver(monkeypatch, version=13060):
    from cuda.bindings import driver as cuda
    from cudnn import _device

    monkeypatch.setattr(_device, "execution_context", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=123))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=216))
    monkeypatch.setattr(cuda, "cuDriverGetVersion", lambda: (cuda.CUresult.CUDA_SUCCESS, version))
    return cuda


def test_current_stream_resources_are_not_cached_by_reusable_handle(monkeypatch):
    cuda = _driver(monkeypatch)
    counts = iter([108, 100, 216])

    def resource(stream, kind):
        assert int(stream) == 123
        return cuda.CUresult.CUDA_SUCCESS, SimpleNamespace(sm=SimpleNamespace(smCount=next(counts)))

    monkeypatch.setattr(cuda, "cuStreamGetDevResource", resource, raising=False)
    assert [_runtime().stream_multiprocessors(0) for _ in range(3)] == [108, 100, 216]


def test_older_driver_queries_the_launch_streams_converted_context(monkeypatch):
    cuda = _driver(monkeypatch, 12090)
    monkeypatch.setattr(cuda, "cuStreamGetCtx", lambda stream: (cuda.CUresult.CUDA_SUCCESS, "converted-green"))

    def resource(context, kind):
        assert context == "converted-green"
        return cuda.CUresult.CUDA_SUCCESS, SimpleNamespace(sm=SimpleNamespace(smCount=80))

    monkeypatch.setattr(cuda, "cuCtxGetDevResource", resource)
    assert _runtime().stream_multiprocessors(0) == 80


def test_pre_green_driver_uses_device_count(monkeypatch):
    _driver(monkeypatch, 12030)
    assert _runtime().stream_multiprocessors(0) == 216


def test_bad_stream_resource_query_is_not_silently_ignored(monkeypatch):
    cuda = _driver(monkeypatch)
    monkeypatch.setattr(cuda, "cuStreamGetDevResource", lambda *args: (cuda.CUresult.CUDA_ERROR_INVALID_HANDLE, None), raising=False)
    with pytest.raises(RuntimeError, match="stream resource query failed"):
        _runtime().stream_multiprocessors(0)


def test_empty_resource_partition_is_rejected(monkeypatch):
    cuda = _driver(monkeypatch)
    monkeypatch.setattr(
        cuda, "cuStreamGetDevResource", lambda *args: (cuda.CUresult.CUDA_SUCCESS, SimpleNamespace(sm=SimpleNamespace(smCount=0))), raising=False
    )
    with pytest.raises(RuntimeError, match="no SM resources"):
        _runtime().stream_multiprocessors(0)


@pytest.mark.parametrize("major,minor,accepted", [(12, 9, False), (13, 3, False), (13, 4, True), (13, 6, True)])
def test_sm107_checks_the_dsl_compiler_not_the_torch_runtime(monkeypatch, major, minor, accepted):
    from cutlass.base_dsl import version_info
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "cutedsl_arch_requirement_error", lambda capability: None)
    monkeypatch.setattr(version_info, "CUDA_VERSION", SimpleNamespace(major=major, minor=minor))
    if accepted:
        _runtime().require_arch((10, 7))
    else:
        with pytest.raises(NotImplementedError, match="compiler >= 13.4"):
            _runtime().require_arch((10, 7))
