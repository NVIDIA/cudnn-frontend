# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def cuda_device(request):
    if not request.module.__name__.endswith("test_kda_jax"):
        return
    import jax

    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or str(getattr(devices[0], "compute_capability", "")) not in ("10.0", "10.3"):
        pytest.skip("KDA requires one visible SM100/SM103 GPU")
