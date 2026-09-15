# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import cutlass.cute as cute
from cudnn.jax import call, zeros_init

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@cute.kernel
def write_pair(x, y, z):
    i = cute.arch.thread_idx()[0]
    y[i] = x[i] + 3
    z[i] = z[i] + x[i]


@cute.jit
def pair(stream, x, y, z):
    write_pair(x, y, z).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def test_initialized_output_order():
    desc = jax.ShapeDtypeStruct((32,), jnp.float32)
    run = jax.jit(call(pair, output_shape_dtype=(desc, desc), initialized_outputs={1: zeros_init}))
    x = jnp.arange(32, dtype=jnp.float32)
    y, z = run(x)
    np.testing.assert_array_equal(y, x + 3)
    np.testing.assert_array_equal(z, x)


@cute.kernel
def write_three(x, a, b, c):
    i = cute.arch.thread_idx()[0]
    a[i] = x[i] + 1
    b[i] = b[i] + x[i] + 2
    c[i] = x[i] + 3


@cute.jit
def three(stream, x, a, b, c):
    write_three(x, a, b, c).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def test_initialized_middle_output():
    desc = jax.ShapeDtypeStruct((32,), jnp.float32)
    run = jax.jit(call(three, output_shape_dtype=(desc,) * 3, initialized_outputs={1: zeros_init}))
    x = jnp.arange(32, dtype=jnp.float32)
    for i, y in enumerate(run(x), 1):
        np.testing.assert_array_equal(y, x + i)


def test_repeated_input():
    @cute.jit
    def repeated(stream, x, unused, y, z):
        pair(stream, x, y, z)

    desc = jax.ShapeDtypeStruct((32,), jnp.float32)
    run = jax.jit(call(repeated, output_shape_dtype=(desc,) * 2, initialized_outputs={1: zeros_init}))
    x = jnp.arange(32, dtype=jnp.float32)
    y, z = run(x, x)
    np.testing.assert_array_equal(y, x + 3)
    np.testing.assert_array_equal(z, x)
