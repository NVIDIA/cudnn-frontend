# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def cuda_device(request):
    if request.module.__name__.endswith(("test_linear_attention_jax", "test_linear_attention_jax_determinism")):
        jax = pytest.importorskip("jax", minversion="0.9.1")
        from cudnn.frost import cutedsl_runtime_available

        devices = [device for device in jax.local_devices() if device.platform == "gpu" and str(getattr(device, "compute_capability", "")) in ("10.0", "10.3")]
        devices = [device for device in devices if cutedsl_runtime_available(tuple(map(int, str(device.compute_capability).split("."))))]
        if not devices:
            pytest.skip("JAX linear attention requires an SM100/SM103 GPU supported by CuTeDSL")
        with jax.default_device(devices[0]):
            yield devices[0]
        return
    if not request.module.__name__.endswith("test_kda_jax"):
        yield
        return
    import jax

    # Other GPUs may be visible (CI exposes several). The first one is JAX's default
    # device and the GPU CuTeDSL compiles for, so it has to be the supported one.
    first = jax.local_devices()[0]
    if first.platform != "gpu" or str(getattr(first, "compute_capability", "")) not in ("10.0", "10.3"):
        pytest.skip("KDA requires the first visible GPU to be SM100/SM103")
    yield
