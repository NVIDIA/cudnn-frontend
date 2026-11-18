# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject unsafe native buffer references before a Cake launch."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def available():
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
        jax.block_until_ready(invoke(jnp.zeros(16, jnp.uint8)))


def test_unknown_plan_rejected():
    from cudnn._cake.jax import _TARGET, _library

    _library()
    invoke = jax.ffi.ffi_call(_TARGET, jax.ShapeDtypeStruct((16,), jnp.uint8))
    with pytest.raises(Exception, match="unknown Cake JAX plan"):
        jax.block_until_ready(invoke(plan=np.int64(0)))
