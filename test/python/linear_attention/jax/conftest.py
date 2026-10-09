# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def cuda_device(request):
    if not request.module.__name__.endswith("test_kda_jax"):
        return
    import jax

    # Other GPUs may be visible (CI exposes several). The first one is JAX's default
    # device and the GPU CuTeDSL compiles for, so it has to be the supported one.
    first = jax.local_devices()[0]
    if first.platform != "gpu" or str(getattr(first, "compute_capability", "")) not in ("10.0", "10.3"):
        pytest.skip("KDA requires the first visible GPU to be SM100/SM103")
