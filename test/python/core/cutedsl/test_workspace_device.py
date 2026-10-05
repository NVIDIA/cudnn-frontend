# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace storage follows the declared launch device, regardless of ordinal.

These checks need no kernel compilation or architecture-specific plan. The
split-K projection integration test separately verifies rejection before launch.
"""

import pytest

torch = pytest.importorskip("torch")

from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace

pytestmark = pytest.mark.L0


class _DLPackOnly:
    def __init__(self, tensor):
        self.tensor = tensor

    def __dlpack__(self, **kwargs):
        return self.tensor.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


class _CAIOnly:
    def __init__(self, tensor):
        self.tensor = tensor

    @property
    def __cuda_array_interface__(self):
        return self.tensor.__cuda_array_interface__


_PRODUCERS = [pytest.param(lambda tensor: tensor, id="torch"), pytest.param(_DLPackOnly, id="dlpack"), pytest.param(_CAIOnly, id="cai")]


def _check_workspace_device(tensor, producer, wrong_device):
    current = torch.cuda.current_device()
    device = tensor.device.index
    workspace = Workspace(producer(tensor), tensor.numel(), "device test", device=device)
    assert torch.cuda.current_device() == current
    view = workspace.take(tensor.numel(), "uint8")
    assert buffers.probe(view)[0] == tensor.data_ptr()
    assert view.__dlpack_device__() == (2, device)
    with pytest.raises(ValueError, match=rf"workspace must be on cuda:{wrong_device}, got cuda:{device}"):
        Workspace(producer(tensor), tensor.numel(), "device test", device=wrong_device)
    assert torch.cuda.current_device() == current


@pytest.mark.parametrize("producer", _PRODUCERS)
def test_workspace_device_validation(producer):
    tensor = torch.empty(256, dtype=torch.uint8, device="cuda")
    # Only the expected metadata differs. No allocation or launch on another
    # ordinal is needed, so the diagnostic is covered with one visible GPU too.
    _check_workspace_device(tensor, producer, tensor.device.index + 1)


@pytest.mark.parametrize("producer", _PRODUCERS[:2])
@pytest.mark.parametrize("pin_memory", [False, True], ids=["cpu", "pinned_cpu"])
def test_workspace_rejects_host_storage(producer, pin_memory):
    tensor = torch.empty(256, dtype=torch.uint8, device="cpu", pin_memory=pin_memory)
    with pytest.raises(ValueError, match="workspace must be a CUDA device buffer"):
        Workspace(producer(tensor), tensor.numel(), "device test", device=torch.cuda.current_device())


@pytest.mark.parametrize("producer", _PRODUCERS)
def test_workspace_device_is_independent_of_current_device(producer):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible GPUs to exercise a foreign current device")
    current = torch.cuda.current_device()
    other = (current + 1) % torch.cuda.device_count()
    for device, ambient in ((current, other), (other, current)):
        tensor = torch.empty(256, dtype=torch.uint8, device=torch.device("cuda", device))
        with torch.cuda.device(ambient):
            _check_workspace_device(tensor, producer, ambient)
    assert torch.cuda.current_device() == current


def test_dlpack_export_failure_restores_current_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible GPUs to exercise a foreign current device")
    current = torch.cuda.current_device()
    other = (current + 1) % torch.cuda.device_count()
    tensor = torch.empty(256, dtype=torch.uint8, device=torch.device("cuda", other))

    class FailingExport(_DLPackOnly):
        def __dlpack__(self, **kwargs):
            # Exercise the real producer's device check before failing. Restoring
            # the ambient device must also hold when export raises an exception.
            super().__dlpack__(**kwargs)
            raise RuntimeError("producer export failed")

    with pytest.raises(RuntimeError, match="producer export failed"):
        Workspace(FailingExport(tensor), tensor.numel(), "device test", device=other)
    assert torch.cuda.current_device() == current


@pytest.mark.parametrize("producer", _PRODUCERS)
def test_workspace_foreign_device_probe_is_capture_safe(producer):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible GPUs to exercise a foreign current device")
    current = torch.cuda.current_device()
    other = (current + 1) % torch.cuda.device_count()
    with torch.cuda.device(other):
        tensor = torch.zeros(256, dtype=torch.uint8, device=torch.device("cuda", other))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            tensor.add_(1)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with torch.cuda.device(current):
                workspace = Workspace(producer(tensor), tensor.numel(), "device test", device=other)
                view = workspace.take(tensor.numel(), "uint8")
                assert view.data_ptr() == tensor.data_ptr()
                assert view.__dlpack_device__() == (2, other)
                assert torch.cuda.current_device() == current
            tensor.add_(1)
        graph.replay()
        torch.testing.assert_close(tensor, torch.full_like(tensor, 2))
    assert torch.cuda.current_device() == current
