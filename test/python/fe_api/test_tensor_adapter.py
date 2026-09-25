# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
from cuda.bindings import runtime as cudart

from cudnn.tensor_adapter import get_compute_capability

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("device_index", [None, 0])
def test_compute_capability_device_selection(monkeypatch, backend, device_index):
    capabilities = {0: (9, 0), 1: (10, 0)}
    if backend == "torch":
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 1, get_device_capability=lambda index: capabilities[index])
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
    else:
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setattr(cudart, "cudaGetDevice", lambda: (cudart.cudaError_t.cudaSuccess, 1))

        def attribute(attr, index):
            component = 0 if attr == cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor else 1
            return cudart.cudaError_t.cudaSuccess, capabilities[index][component]

        monkeypatch.setattr(cudart, "cudaDeviceGetAttribute", attribute)
    actual = get_compute_capability() if device_index is None else get_compute_capability(device_index)
    assert actual == capabilities[1 if device_index is None else device_index]
