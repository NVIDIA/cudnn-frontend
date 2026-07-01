# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""GPU integration tests for the JAX RMSNorm + RHT + amax wrapper."""

import math

import pytest


def _hadamard_16(jnp):
    values = [[1.0 if ((row & col).bit_count() % 2 == 0) else -1.0 for col in range(16)] for row in range(16)]
    return jnp.asarray(values, dtype=jnp.float32) / math.sqrt(16)


@pytest.mark.L0
def test_jax_rmsnorm_rht_amax_abstract_shape():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("cutlass.jax")

    from cudnn.jax import rmsnorm_rht_amax_sm100

    x = jax.ShapeDtypeStruct((256, 2048), jnp.bfloat16)
    weight = jax.ShapeDtypeStruct((2048,), jnp.bfloat16)
    output, amax = jax.eval_shape(rmsnorm_rht_amax_sm100, x, weight)

    assert output.shape == (256, 2048)
    assert output.dtype == jnp.bfloat16
    assert amax.shape == (128,)
    assert amax.dtype == jnp.float32


@pytest.mark.L0
def test_jax_rmsnorm_rht_amax_jit():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("cutlass.jax")

    gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    if not gpu_devices:
        pytest.skip("JAX CUDA device is not available")

    capable_devices = []
    reported_capabilities = []
    for device in gpu_devices:
        capability = getattr(device, "compute_capability", None)
        if capability is None:
            continue
        major, minor = (int(value) for value in str(capability).split(".", 1))
        reported_capabilities.append(f"SM{major}{minor}")
        if major * 10 + minor >= 100:
            capable_devices.append(device)
    if not capable_devices:
        reported = ", ".join(reported_capabilities) or "unknown capabilities"
        pytest.skip(f"RMSNorm + RHT requires SM100+; local GPUs report {reported}")
    device = capable_devices[0]

    from cudnn.jax import rmsnorm_rht_amax_sm100

    m, n = 256, 2048
    rows_per_cta = 2
    eps = 1e-5
    x = jax.device_put(
        jax.random.normal(jax.random.key(0), (m, n), dtype=jnp.bfloat16),
        device,
    )
    weight = jax.device_put(
        jax.random.normal(jax.random.key(1), (n,), dtype=jnp.bfloat16),
        device,
    )

    @jax.jit
    def run(x, weight):
        return rmsnorm_rht_amax_sm100(
            x,
            weight,
            eps=eps,
            num_threads=128,
            rows_per_cta=rows_per_cta,
        )

    lowered = run.lower(x, weight)
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo

    compiled = lowered.compile()
    output, amax = compiled(x, weight)
    output.block_until_ready()
    second_output, second_amax = compiled(x, weight)
    second_output.block_until_ready()

    x_f32 = x.astype(jnp.float32)
    normalized = x_f32 * jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
    normalized *= weight.astype(jnp.float32)[None, :]
    reference = (normalized.reshape(m, n // 16, 16) @ _hadamard_16(jnp)).reshape(m, n)
    amax_reference = jnp.max(
        jnp.abs(reference).reshape(m // rows_per_cta, rows_per_cta, n),
        axis=(1, 2),
    )

    assert output.shape == (m, n)
    assert output.dtype == jnp.bfloat16
    assert amax.shape == (m // rows_per_cta,)
    assert amax.dtype == jnp.float32
    assert jnp.allclose(output.astype(jnp.float32), reference, atol=4e-2, rtol=1e-2)
    assert jnp.allclose(amax, amax_reference, atol=2e-3, rtol=1e-3)
    assert jnp.array_equal(second_output, output)
    assert jnp.array_equal(second_amax, amax)
