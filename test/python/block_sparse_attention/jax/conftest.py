# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest


def pytest_addoption(parser):
    parser.addoption("--require-sm100", action="store_true", help="Fail qualification if the default JAX device is not an SM100 GPU")


def sm100_available():
    import jax

    device = jax.local_devices()[0]  # where uncommitted test arrays land
    return device.platform == "gpu" and str(getattr(device, "compute_capability", "")) == "10.0"


def pytest_configure(config):
    for marker in ("L0: smoke tests", "gpu_exclusive: requires exclusive GPU access"):
        config.addinivalue_line("markers", marker)
    if config.getoption("--require-sm100", default=False) and not sm100_available():
        raise pytest.UsageError("BSA JAX qualification requires the default JAX device to be an SM100 GPU")


@pytest.fixture(autouse=True)
def cuda_device(request):
    if request.node.name == "test_import_isolation":
        return
    if not sm100_available():
        pytest.skip("BSA JAX tests require the default JAX device to be an SM100 GPU")
