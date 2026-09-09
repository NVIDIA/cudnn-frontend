# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import pytest


def pytest_configure(config):
    for marker in ("L0: smoke tests", "gpu_exclusive: requires exclusive GPU access"):
        config.addinivalue_line("markers", marker)


@pytest.fixture(autouse=True)
def sm100_device(request):
    if request.node.name == "test_import_isolation":
        return
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or str(getattr(devices[0], "compute_capability", "")) != "10.0":
        pytest.skip("BSA JAX tests require one visible SM100 GPU")
