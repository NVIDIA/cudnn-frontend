# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def cuda_device():
    import jax

    device = jax.local_devices()[0]  # where uncommitted test arrays land
    if device.platform != "gpu" or str(getattr(device, "compute_capability", "")) not in ("10.0", "10.3"):
        pytest.skip("JAX tests require the default JAX device to be an SM100 or SM103 GPU")
