# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True, scope="module")
def cuda_device(request):
    jax = pytest.importorskip("jax")
    devices = [d for d in jax.devices() if d.platform == "gpu"]
    if not devices:
        pytest.skip("JAX CUDA device required")
    if request.module.__name__.endswith("test_kda"):
        from cudnn.frost.device import compute_capability

        if len(devices) != 1 or compute_capability(devices[0].local_hardware_id) not in ((10, 0), (10, 3)):
            pytest.skip("KDA requires one visible SM100/SM103 GPU")
