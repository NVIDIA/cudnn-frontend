# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("JAX bridge tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

jax = pytest.importorskip("jax", minversion="0.9.1")
import jax.numpy as jnp

import cutlass.cute as cute
from cutlass.jax import TensorSpec
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


@cute.kernel
def accumulate_kernel(x, y):
    i = cute.arch.thread_idx()[0]
    y[i] = y[i] + x[i]


@cute.jit
def accumulate(stream, x, y):
    accumulate_kernel(x, y).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@pytest.mark.parametrize("container", [None, tuple, list])
@pytest.mark.parametrize("jitted", [False, True])
def test_initialized_single_output(container, jitted):
    desc = jax.ShapeDtypeStruct((32,), jnp.float32)
    spec = TensorSpec(layout=(0,))
    run = call(
        accumulate,
        output_shape_dtype=container([desc]) if container else desc,
        output_spec=container([spec]) if container else spec,
        initialized_outputs={0: zeros_init},
    )
    if jitted:
        run = jax.jit(run)
    x = jnp.arange(32, dtype=jnp.float32)
    result = run(x)
    assert isinstance(result, container or jax.Array)
    np.testing.assert_array_equal(result[0] if container else result, x)


def test_eager_initialized_outputs_reuse_compilation():
    desc = jax.ShapeDtypeStruct((32,), jnp.float32)
    run = call(three, output_shape_dtype=(desc,) * 3, initialized_outputs={1: zeros_init})
    inputs = [jnp.arange(32, dtype=jnp.float32) + offset for offset in range(3)]
    jax.block_until_ready(run(inputs[0]))
    with jax.no_tracing(True):
        results = [run(x) for x in inputs]
        jax.block_until_ready(results)
    for x, outputs in zip(inputs, results):
        for offset, output in enumerate(outputs, 1):
            np.testing.assert_array_equal(output, np.asarray(x) + offset)


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
