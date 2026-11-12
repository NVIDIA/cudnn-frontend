# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject unsafe native buffer references before a Cake launch."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def cuda_device():
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("requires GPU")
    from cudnn._cake.loader import check_toolkit, device_arch

    try:
        check_toolkit(device_arch(0), "Cake JAX bridge tests")
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("offset,size", [(0, 512), (1, 16)])
def test_workspace_bounds_and_alignment(offset, size):
    from cudnn._cake.jax import TensorRef, make_call

    invoke = make_call([], 1, (jax.ShapeDtypeStruct((256,), jnp.uint8),), 0, (TensorRef(1, (size,), "uint8", offset),))
    with pytest.raises(Exception, match="invalid Cake workspace capacity or alignment"):
        jax.block_until_ready(invoke(jax.device_put(np.zeros(16, np.uint8))))


def test_unknown_plan_rejected():
    from cudnn._cake.jax import _TARGET, _library

    _library()
    invoke = jax.ffi.ffi_call(_TARGET, jax.ShapeDtypeStruct((16,), jnp.uint8))
    with pytest.raises(Exception, match="unknown Cake JAX plan"):
        jax.block_until_ready(invoke(plan=np.int64(0)))


def test_external_workspace_lifetime():
    # A rejected side-effecting call leaves a failed JAX runtime token.
    subprocess.run(
        [sys.executable, "-c", "from core.jax.test_cake_ffi import _external_workspace_lifetime; _external_workspace_lifetime()"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _external_workspace_lifetime():
    from cudnn._cake.jax import ExternalTensorRef, LaunchContext, TensorRef, make_call
    from cudnn._device import _ck, _driver

    device = next(d for d in jax.local_devices() if d.platform == "gpu")
    with jax.default_device(device):
        jax.block_until_ready(jax.device_put(np.zeros(1, np.uint8), device))
        driver = _driver()
        pointer = _ck(*driver.cuMemAlloc(256))
        context = LaunchContext(device.local_hardware_id)
        try:
            external = ExternalTensorRef(int(pointer), (256,), "uint8", 256)
            with pytest.raises(ValueError, match="owning launch context"):
                make_call([], 0, (), device.local_hardware_id, (external,))
            _ck(*driver.cuMemsetD8(pointer, 7, 256))
            invoke = jax.jit(
                make_call(
                    [],
                    0,
                    (jax.ShapeDtypeStruct((256,), jnp.uint8),),
                    device.local_hardware_id,
                    (external, TensorRef(0, (256,), "uint8")),
                    context=context,
                )
            )
            jax.block_until_ready(invoke())
            host = np.empty(256, np.uint8)
            _ck(*driver.cuMemcpyDtoH(host.ctypes.data, pointer, host.nbytes))
            np.testing.assert_array_equal(host, 0)
            context.close()
            with pytest.raises(Exception, match="launch context is closed"):
                jax.block_until_ready(invoke())
        finally:
            context.close()
            _ck(*driver.cuMemFree(pointer))
