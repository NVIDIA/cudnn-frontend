# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import pytest


def pytest_addoption(parser):
    parser.addoption("--require-sm100", action="store_true", help="Fail qualification if a single SM100 GPU is unavailable")


def sm100_available():
    devices = jax.local_devices()
    return len(devices) == 1 and devices[0].platform == "gpu" and str(getattr(devices[0], "compute_capability", "")) == "10.0"


def pytest_configure(config):
    for marker in ("L0: smoke tests", "gpu_exclusive: requires exclusive GPU access"):
        config.addinivalue_line("markers", marker)
    if config.getoption("--require-sm100") and not sm100_available():
        raise pytest.UsageError("BSA JAX qualification requires one visible SM100 GPU")


@pytest.fixture(autouse=True)
def sm100_device(request):
    if request.node.name == "test_import_isolation":
        return
    if not sm100_available():
        pytest.skip("BSA JAX tests require one visible SM100 GPU")
