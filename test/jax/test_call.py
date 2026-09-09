# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import numpy as np
import cutlass.cute as cute

from cudnn.jax import call, zeros_init


@cute.kernel
def mixed_kernel(x, y, accumulator, z):
    i = cute.arch.thread_idx()[0]
    value = x[i]
    y[i] = value + 1
    accumulator[i] = accumulator[i] + value * 2
    z[i] = (value + 3).to(z.element_type)


@cute.jit
def mixed_launch(stream, x, y, accumulator, z):
    mixed_kernel(x, y, accumulator, z).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@pytest.mark.parametrize("initialized", [(1,), (0, 1, 2)])
def test_mixed_initialized_output(initialized):
    spec = jax.ShapeDtypeStruct((32,), jnp.float32)
    run = jax.jit(
        call(
            mixed_launch,
            output_shape_dtype=(spec, spec, jax.ShapeDtypeStruct((32,), jnp.float16)),
            initialized_outputs={i: zeros_init for i in initialized},
        )
    )
    for offset in (1, 5):
        x = jnp.arange(32, dtype=jnp.float32) + offset
        y, accumulator, z = run(x)
        np.testing.assert_array_equal(y, x + 1)
        np.testing.assert_array_equal(accumulator, x * 2)
        np.testing.assert_array_equal(z, x + 3)


@cute.jit
def alias_launch(stream, x, accumulator, z):
    mixed_kernel(x, x, accumulator, z).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def test_explicit_alias_with_initialized_output():
    spec = jax.ShapeDtypeStruct((32,), jnp.float32)
    run = jax.jit(call(alias_launch, output_shape_dtype=(spec, spec, spec), input_output_aliases={0: 0}, initialized_outputs={1: zeros_init}))
    x = jnp.arange(32, dtype=jnp.float32)
    y, accumulator, z = run(x)
    np.testing.assert_array_equal(y, x + 1)
    np.testing.assert_array_equal(accumulator, x * 2)
    np.testing.assert_array_equal(z, x + 3)
