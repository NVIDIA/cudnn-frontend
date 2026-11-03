# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest


@pytest.fixture(autouse=True)
def cuda_device():
    import jax

    devices = [device for device in jax.local_devices() if device.platform == "gpu" and str(getattr(device, "compute_capability", "")) in ("10.0", "10.3")]
    if not devices:
        pytest.skip("JAX norm tests require an SM100 or SM103 GPU")
    with jax.default_device(devices[0]):
        yield
