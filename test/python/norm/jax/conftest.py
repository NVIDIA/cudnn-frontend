# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest


@pytest.fixture(autouse=True)
def cuda_device():
    import jax

    device = jax.local_devices()[0]
    if device.platform != "gpu" or str(getattr(device, "compute_capability", "")) != "10.0":
        pytest.skip("JAX norm tests require the default JAX device to be an SM100 GPU")
