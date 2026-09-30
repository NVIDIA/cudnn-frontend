# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def cuda_device():
    import jax

    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or str(getattr(devices[0], "compute_capability", "")) != "10.0":
        pytest.skip("JAX tests require one visible SM100 GPU")
