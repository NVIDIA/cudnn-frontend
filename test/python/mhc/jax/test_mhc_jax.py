# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cudnn.mhc import jax_api as api

pytestmark = pytest.mark.L0


@pytest.fixture
def data():
    if not any(d.platform == "gpu" for d in jax.local_devices()):
        pytest.skip("CUDA JAX required")
    from cudnn._cake import loader

    try:
        loader.check_toolkit(loader.device_arch(), "JAX mHC tests")
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(str(error))
    rng = np.random.default_rng(74)

    def rand(shape, scale=1):
        return jnp.asarray(rng.standard_normal(shape).astype(np.float32) * scale)

    return (rand((3, 4, 4096)).astype(jnp.bfloat16), rand((24, 16384), 0.01), rand((3,), 0.1) + 1, rand((24,), 0.1), rand((4096,), 0.2))


def _branch(u, weight):
    return jnp.tanh(u.astype(jnp.float32) * weight).astype(jnp.bfloat16)


def _reference(x, fn, scale, base, weight, iters):
    x = x.astype(jnp.float32)
    flat = x.reshape(x.shape[0], -1)
    z = jnp.matmul(flat * jax.lax.rsqrt(jnp.mean(flat * flat, axis=1, keepdims=True) + 1e-5), fn.T, precision=jax.lax.Precision.HIGHEST)
    pre = jax.nn.sigmoid(z[:, :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * jax.nn.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    mix = jax.nn.softmax((z[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4), axis=-1) + 1e-6
    mix /= jnp.sum(mix, axis=1, keepdims=True) + 1e-6
    for _ in range(iters - 1):
        mix /= jnp.sum(mix, axis=2, keepdims=True) + 1e-6
        mix /= jnp.sum(mix, axis=1, keepdims=True) + 1e-6
    u = jnp.sum(pre[..., None] * x, axis=1).astype(jnp.bfloat16)
    return (post[..., None] * _branch(u, weight)[:, None, :] + jnp.einsum("tij,tid->tjd", mix, x)).astype(jnp.bfloat16)


@pytest.mark.parametrize("iters", [7, 20])
def test_connection_jit_grad_reference_replay(data, iters):
    def native(*args):
        return api.mhc_jax(*args[:4], _branch, args[4], sinkhorn_iters=iters)

    def reference(*args):
        return _reference(*args, iters)

    dy = jnp.asarray(np.random.default_rng(16).standard_normal(data[0].shape), dtype=jnp.bfloat16)

    def loss(function, *args):
        return jnp.sum(function(*args).astype(jnp.float32) * dy.astype(jnp.float32))

    native_jit = jax.jit(native)
    native_grad = jax.jit(jax.grad(lambda *a: loss(native, *a), argnums=tuple(range(5))))
    reference_grad = jax.jit(jax.grad(lambda *a: loss(reference, *a), argnums=tuple(range(5))))
    for x in (data[0], -data[0]):
        args = (x, *data[1:])
        result = native_jit(*args)
        np.testing.assert_allclose(np.asarray(result, dtype=np.float32), np.asarray(reference(*args), dtype=np.float32), atol=0.025, rtol=0.015)
        gradients = native_grad(*args)
        expected = reference_grad(*args)
        assert [g.dtype for g in gradients] == [jnp.bfloat16, jnp.float32, jnp.float32, jnp.float32, jnp.float32]
        for actual, want in zip(gradients, expected):
            actual, want = np.asarray(actual, dtype=np.float32), np.asarray(want, dtype=np.float32)
            assert np.linalg.norm(actual - want) / np.linalg.norm(want) < 0.015
        repeat = native_grad(*args)
        for actual, want in zip(gradients, repeat):
            np.testing.assert_array_equal(actual, want)


def test_head_jit_gradient(data):
    x = data[0]
    forward = jax.jit(lambda x: api.mhc_head_forward_jax(x)[0])
    expected = jnp.mean(x.astype(jnp.float32), axis=1).astype(jnp.bfloat16)
    np.testing.assert_array_equal(forward(x), expected)
    derivative = jax.jit(jax.grad(lambda x: jnp.sum(forward(x).astype(jnp.float32))))
    np.testing.assert_array_equal(derivative(x), jnp.full_like(x, 0.25))
    np.testing.assert_array_equal(forward(-x), -expected)


def test_public_exports():
    from cudnn import ops

    for name in (
        "mhc_jax",
        "mhc_prepare_jax",
        "mhc_finish_jax",
        "mhc_backward_prepare_jax",
        "mhc_backward_finish_jax",
        "mhc_head_forward_jax",
        "mhc_head_backward_jax",
    ):
        assert getattr(ops, name) is getattr(api, name)


def test_validation_before_native_load(monkeypatch):
    monkeypatch.setattr(api, "_build", lambda *a: pytest.fail("invalid input reached native loader"))
    with pytest.raises(ValueError, match="x\\[T,4,H\\]"):
        api.mhc_head_forward_jax(jnp.zeros((1, 3, 4096), jnp.bfloat16))
    with pytest.raises(ValueError, match="sinkhorn_iters"):
        api.mhc_jax(None, None, None, None, _branch, sinkhorn_iters=True)
    with pytest.raises(ValueError, match="dtype"):
        api.mhc_head_backward_jax(jnp.zeros((1, 4096), jnp.float32))
