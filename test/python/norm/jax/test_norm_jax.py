# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import numpy as np
import pytest

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.norm._lazy import dsl_unavailable

requirement_error = cutedsl_requirement_error("JAX norm tests", min_version=(4, 7, 1)) or dsl_unavailable("JAX norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

jax = pytest.importorskip("jax", minversion="0.6.0")
import jax.numpy as jnp

from cudnn.ops import layer_norm_forward_jax, rms_norm_forward_jax

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def invoke(kind, args, normalized_shape=None, eps=1e-5):
    x, weight, bias = args
    if kind == "layer":
        result = layer_norm_forward_jax(x, weight, bias, normalized_shape=normalized_shape, eps=eps)
    else:
        result = rms_norm_forward_jax(x, weight, normalized_shape=normalized_shape, eps=eps)
    assert type(result) is tuple and len(result) == 1
    return result[0]


def reference(kind, args, normalized_shape, eps):
    x, weight, bias = args
    axes = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    y = x.astype(jnp.float32)
    if kind == "layer":
        y = y - jnp.mean(y, axis=axes, keepdims=True)
    y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=axes, keepdims=True) + eps)
    if weight is not None:
        y = y * weight.astype(jnp.float32)
    if bias is not None:
        y = y + bias.astype(jnp.float32)
    return y.astype(x.dtype)


def assert_close(actual, expected, dtype):
    tolerance = {jnp.bfloat16: 0.025, jnp.float16: 0.003, jnp.float32: 3e-5}[dtype]
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "kind,shape,dtype,normalized_shape,has_weight,has_bias",
    [
        ("layer", (31,), jnp.float32, None, True, True),
        ("layer", (3, 31), jnp.bfloat16, None, False, False),
        ("layer", (2, 3, 5, 8), jnp.float16, (5, 8), True, True),
        ("layer", (7, 128), jnp.float32, 128, False, True),
        ("layer", (128, 4096), jnp.float16, None, True, True),
        ("layer", (2, 32768), jnp.bfloat16, [32768], True, False),
        ("rms", (3, 31), jnp.float16, None, False, False),
        ("rms", (2, 3, 5, 8), jnp.bfloat16, (5, 8), True, False),
        ("rms", (8, 128), jnp.float32, 128, True, False),
        ("rms", (128, 4096), jnp.bfloat16, None, True, False),
        ("rms", (2, 16384), jnp.float32, [16384], True, False),
    ],
)
def test_forward_and_jitted_gradients(kind, shape, dtype, normalized_shape, has_weight, has_bias):
    normalized = (shape[-1],) if normalized_shape is None else (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    eps = 3e-4
    forward = jax.jit(lambda args: invoke(kind, args, normalized_shape, eps))

    def loss(args, cotangent, reference_mode=False):
        y = reference(kind, args, normalized, eps) if reference_mode else forward(args)
        return jnp.sum(y.astype(jnp.float32) * cotangent.astype(jnp.float32))

    gradients = jax.jit(jax.grad(loss))
    reference_gradients = jax.jit(jax.grad(lambda args, cotangent: loss(args, cotangent, True)))
    rng = np.random.default_rng(42)
    for iteration in range(2):
        args = (
            jnp.asarray(rng.normal(0.3 * iteration, 0.7, shape), dtype),
            jnp.asarray(rng.uniform(0.5, 1.5, normalized), dtype) if has_weight else None,
            jnp.asarray(rng.normal(0, 0.2, normalized), dtype) if has_bias else None,
        )
        cotangent = jnp.asarray(rng.normal(size=shape), dtype)
        output = forward(args)
        assert output.shape == shape and output.dtype == dtype
        assert_close(output, reference(kind, args, normalized, eps), dtype)
        actual, expected = gradients(args, cotangent), reference_gradients(args, cotangent)
        for primal, gradient, reference_gradient in zip(args, actual, expected):
            if primal is None:
                assert gradient is None and reference_gradient is None
            else:
                assert gradient.shape == primal.shape and gradient.dtype == primal.dtype
                assert_close(gradient, reference_gradient, dtype)
        if shape == (3, 31) and iteration == 0:
            assert_close(invoke(kind, args, normalized_shape, eps), output, dtype)


@pytest.mark.parametrize("kind", ["layer", "rms"])
def test_eager_multidimensional_inputs_reach_bridge_unchanged(kind, monkeypatch):
    from cudnn.norm import jax_api

    shape, normalized_shape = (2, 3, 5, 8), (5, 8)
    rng = np.random.default_rng(42)
    x = jnp.asarray(rng.normal(size=shape), jnp.float32)
    weight = jnp.asarray(rng.uniform(0.5, 1.5, normalized_shape), jnp.float32)
    bias = jnp.asarray(rng.normal(size=normalized_shape), jnp.float32) if kind == "layer" else None
    dy = jnp.asarray(rng.normal(size=shape), jnp.float32)

    def check_inputs(factory, expected):
        def bind(config):
            run = factory(config)

            def checked(*arrays):
                for actual, original in zip(arrays, expected):
                    assert actual is original
                return run(*arrays)

            return checked

        return bind

    monkeypatch.setattr(jax_api, "_forward_call", check_inputs(jax_api._forward_call, (x, weight, weight if bias is None else bias)))
    monkeypatch.setattr(jax_api, "_backward_call", check_inputs(jax_api._backward_call, (dy, x, weight)))
    args = (x, weight, bias)
    y, backward = jax.vjp(lambda args: invoke(kind, args, normalized_shape), args)
    expected, reference_backward = jax.vjp(lambda args: reference(kind, args, normalized_shape, 1e-5), args)
    assert_close(y, expected, jnp.float32)
    actual_gradients = backward(dy)[0]
    expected_gradients = reference_backward(dy)[0]
    for actual, expected in zip(actual_gradients, expected_gradients):
        if expected is None:
            assert actual is None
        else:
            assert_close(actual, expected, jnp.float32)


@pytest.mark.parametrize("kind", ["layer", "rms"])
@pytest.mark.parametrize("eps", [0.0, -1e-5, float("nan"), float("inf"), 1e100, 1e-100])
def test_rejects_invalid_epsilon(kind, eps):
    x = jnp.ones((2, 8), jnp.float32)
    with pytest.raises((TypeError, ValueError), match="eps"):
        invoke(kind, (x, None, None), eps=eps)


@pytest.mark.parametrize("kind", ["layer", "rms"])
@pytest.mark.parametrize("normalized_shape", [(), (4, 2), (0,), (2, 2, 8)])
def test_rejects_invalid_normalized_shape(kind, normalized_shape):
    x = jnp.ones((2, 8), jnp.float32)
    with pytest.raises((TypeError, ValueError), match="shape|dimension|positive"):
        invoke(kind, (x, None, None), normalized_shape)


@pytest.mark.parametrize("kind", ["layer", "rms"])
def test_rejects_invalid_operands(kind):
    x = jnp.ones((2, 8), jnp.float32)
    with pytest.raises((TypeError, ValueError), match="dtype"):
        invoke(kind, (x.astype(jnp.int32), None, None))
    with pytest.raises((TypeError, ValueError), match="dtype"):
        invoke(kind, (x, jnp.ones((8,), jnp.float16), None))
    with pytest.raises((TypeError, ValueError), match="shape"):
        invoke(kind, (x, jnp.ones((1, 8), jnp.float32), None))
    with pytest.raises((TypeError, ValueError), match="shape|dimension|rank|positive"):
        invoke(kind, (jnp.ones((), jnp.float32), None, None))
    with pytest.raises((TypeError, ValueError), match="shape|dimension|positive"):
        invoke(kind, (jnp.ones((0, 8), jnp.float32), None, None))
    if kind == "layer":
        with pytest.raises((TypeError, ValueError), match="dtype"):
            invoke(kind, (x, None, jnp.ones((8,), jnp.float16)))
        with pytest.raises((TypeError, ValueError), match="shape"):
            invoke(kind, (x, None, jnp.ones((7,), jnp.float32)))


@pytest.mark.parametrize("kind", ["layer", "rms"])
def test_rejects_cpu_arrays(kind):
    x = jax.device_put(np.ones((2, 8), np.float32), jax.devices("cpu")[0])
    with pytest.raises((ValueError, NotImplementedError), match="GPU|gpu|CUDA|cuda"):
        invoke(kind, (x, None, None))


@pytest.mark.parametrize("kind", ["layer", "rms"])
def test_rejects_unsupported_gpu(kind, monkeypatch):
    monkeypatch.setattr("cudnn.frost.device.compute_capability", lambda device: (9, 0))
    x = jnp.ones((2, 8), jnp.float32)
    with pytest.raises(NotImplementedError, match="SM100"):
        invoke(kind, (x, None, None))


def test_rejects_multiple_processes(monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    with pytest.raises(NotImplementedError, match="single process"):
        jax.jit(lambda x: layer_norm_forward_jax(x)).lower(jax.ShapeDtypeStruct((2, 8), jnp.float32))


@pytest.mark.parametrize(
    "property_name,other_value",
    [("compute_capability", (10, 3)), ("multiprocessor_count", 132), ("shared_memory_per_block_optin", 163840)],
)
def test_tracing_rejects_heterogeneous_cuda_devices_with_closed_over_weight(monkeypatch, property_name, other_value):
    from cudnn.frost import device

    monkeypatch.setattr(device, "device_count", lambda: 2)
    properties = {"compute_capability": (10, 0), "multiprocessor_count": 148, "shared_memory_per_block_optin": 232448}
    for name, value in properties.items():
        monkeypatch.setattr(device, name, lambda ordinal, value=value: value)
    monkeypatch.setattr(device, property_name, lambda ordinal: other_value if ordinal == 1 else properties[property_name])
    weight = jnp.ones((8,), jnp.float32)
    with pytest.raises(NotImplementedError, match="matching compute capability"):
        jax.jit(lambda x: layer_norm_forward_jax(x, weight)).lower(jax.ShapeDtypeStruct((2, 8), jnp.float32))


@pytest.mark.parametrize("kind", ["layer", "rms"])
def test_multiple_gpus_preserve_placement_and_reuse_compilation(kind):
    from cudnn.frost import device

    devices = [d for d in jax.local_devices() if d.platform == "gpu" and str(getattr(d, "compute_capability", "")) == "10.0"]
    if len(devices) < 2:
        pytest.skip("needs two visible SM100 GPUs")
    properties = {(device.compute_capability(i), device.multiprocessor_count(i), device.shared_memory_per_block_optin(i)) for i in range(device.device_count())}
    if len(properties) != 1:
        pytest.skip("needs matching properties across all visible CUDA GPUs")
    shape = (8, 4096)
    forward = jax.jit(lambda args: invoke(kind, args))
    gradients = jax.grad(lambda args, cotangent: jnp.sum(invoke(kind, args) * cotangent))
    compiled_gradients = jax.jit(gradients)
    reference_gradients = jax.grad(lambda args, cotangent: jnp.sum(reference(kind, args, shape[-1:], 1e-5) * cotangent))
    rng = np.random.default_rng(42)
    for iteration, target in enumerate((devices[0], devices[1], devices[0], devices[1])):
        host_args = (
            rng.normal(0.3 * iteration, 0.7, shape).astype(np.float32),
            rng.uniform(0.5, 1.5, shape[-1:]).astype(np.float32),
            rng.normal(0, 0.2, shape[-1:]).astype(np.float32) if kind == "layer" else None,
        )
        args, cotangent = jax.device_put((host_args, rng.normal(size=shape).astype(np.float32)), target)
        expected = reference(kind, args, shape[-1:], 1e-5), reference_gradients(args, cotangent)
        results = [(forward(args), compiled_gradients(args, cotangent))]
        if iteration == 1:
            results.append((invoke(kind, args), gradients(args, cotangent)))
        for result in results:
            for actual, desired in zip(jax.tree.leaves(result), jax.tree.leaves(expected), strict=True):
                assert actual.devices() == {target}
                assert_close(actual, desired, jnp.float32)
        if iteration == 1:
            cache_sizes = forward._cache_size(), compiled_gradients._cache_size()
        elif iteration > 1:
            assert (forward._cache_size(), compiled_gradients._cache_size()) == cache_sizes
    with pytest.raises(ValueError, match="same GPU"):
        invoke(kind, (args[0], jax.device_put(args[1], devices[0]), args[2]))


def test_sm_count_follows_ambient_and_build_devices(monkeypatch):
    from cudnn.frost import device
    from cudnn.norm.utils import sm_count

    counts = {0: 68, 1: 148}
    monkeypatch.setattr(device, "multiprocessor_count", lambda ordinal: counts[ordinal])
    monkeypatch.setattr(device, "ambient_device", lambda: 0)
    assert sm_count() == 68
    with device.build_device(1):
        assert sm_count() == 148
    assert sm_count() == 68
    monkeypatch.setattr(device, "ambient_device", lambda: 1)
    assert sm_count() == 148


def test_torch_free_forward_and_gradients():
    script = """
import sys
class RejectTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("torch import attempted: " + fullname)
sys.meta_path.insert(0, RejectTorch())
import jax
import jax.numpy as jnp
from cudnn.ops import layer_norm_forward_jax, rms_norm_forward_jax
for rows, cols in ((3, 31), (2, 16384)):
    x = jnp.arange(rows * cols, dtype=jnp.float32).reshape(rows, cols) / (rows * cols)
    weight = jnp.linspace(0.5, 1.5, cols, dtype=jnp.float32)
    bias = jnp.linspace(-0.2, 0.2, cols, dtype=jnp.float32)
    for op, args in ((layer_norm_forward_jax, (x, weight, bias)), (rms_norm_forward_jax, (x, weight))):
        run = jax.jit(jax.value_and_grad(lambda args: jnp.square(op(*args)[0]).sum()))
        value, gradients = jax.block_until_ready(run(args))
        assert jnp.isfinite(value) and all(jnp.all(jnp.isfinite(g)) for g in gradients)
assert "torch" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        timeout=300,
    )
