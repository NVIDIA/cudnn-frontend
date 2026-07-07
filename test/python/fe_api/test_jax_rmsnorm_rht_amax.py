# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX integration tests for RMSNorm + RHT + amax."""

import math

import pytest


def _jax_runtime():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    cutlass_jax = pytest.importorskip("cutlass.jax")
    if not cutlass_jax.is_available():
        pytest.skip("Installed JAX version is unsupported by CUTLASS JAX")
    return jax, jnp


def _hadamard_16(jnp):
    values = [[1.0 if ((row & column).bit_count() % 2 == 0) else -1.0 for column in range(16)] for row in range(16)]
    return jnp.asarray(values, dtype=jnp.float32) / math.sqrt(16)


@pytest.mark.L0
def test_jax_rmsnorm_rht_amax_abstract_contract():
    jax, jnp = _jax_runtime()

    from cudnn import OpKernel, TensorDesc
    from cudnn.jax import JaxApiBase, RmsNormRhtAmaxSm100, rmsnorm_rht_amax_sm100
    from cudnn.rmsnorm_rht_amax.kernel import RMSNormRHTAmaxKernel

    sample_x = jax.ShapeDtypeStruct((256, 2048), jnp.bfloat16)
    sample_weight = jax.ShapeDtypeStruct((2048,), jnp.bfloat16)
    api = RmsNormRhtAmaxSm100(sample_x, sample_weight)

    assert isinstance(api, JaxApiBase)
    assert isinstance(api.kernel, OpKernel)
    assert isinstance(api.kernel, RMSNormRHTAmaxKernel)
    assert isinstance(api.x_desc, TensorDesc)
    assert isinstance(api.w_desc, TensorDesc)
    assert api.kernel.x is api.x_desc
    assert api.kernel.weight is api.w_desc
    assert api.x_desc.dtype == jnp.dtype(jnp.bfloat16)
    assert api.x_desc.stride == (2048, 1)
    assert api.x_desc.stride_order == (1, 0)
    assert api.x_desc.name == "sample_x"
    assert all(value is not sample_x and value is not sample_weight for value in vars(api).values())
    output, amax = jax.eval_shape(api, sample_x, sample_weight)
    assert output.shape == (256, 2048)
    assert output.dtype == jnp.bfloat16
    assert amax.shape == (128,)
    assert amax.dtype == jnp.float32

    functional_output, functional_amax = jax.eval_shape(
        lambda x, weight: rmsnorm_rht_amax_sm100(
            x,
            weight,
            eps=1e-4,
            num_threads=128,
            rows_per_cta=2,
        ),
        sample_x,
        sample_weight,
    )
    assert functional_output.shape == (256, 2048)
    assert functional_output.dtype == jnp.bfloat16
    assert functional_amax.shape == (128,)
    assert functional_amax.dtype == jnp.float32

    wrong_x = jax.ShapeDtypeStruct((128, 2048), jnp.bfloat16)
    with pytest.raises(ValueError, match="sample_x tensor shape mismatch"):
        jax.eval_shape(api, wrong_x, sample_weight)

    api.kernel.requested_rows_per_cta = 3
    with pytest.raises(ValueError, match="M must be divisible by rows_per_cta"):
        jax.eval_shape(lambda x, weight: api(x, weight), sample_x, sample_weight)


@pytest.mark.L0
def test_jax_rmsnorm_rht_amax_jit():
    jax, jnp = _jax_runtime()
    gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    capable_devices = []
    for device in gpu_devices:
        capability = getattr(device, "compute_capability", None)
        if capability is None:
            continue
        major, minor = (int(value) for value in str(capability).split(".", 1))
        if major * 10 + minor >= 100:
            capable_devices.append(device)
    if not capable_devices:
        pytest.skip("RMSNorm + RHT requires an SM100+ JAX device")

    from cudnn.jax import rmsnorm_rht_amax_sm100

    device = capable_devices[0]
    m, n = 256, 2048
    rows_per_cta = 2
    eps = 1e-4
    x = jax.device_put(jax.random.normal(jax.random.key(0), (m, n), dtype=jnp.bfloat16), device)
    weight = jax.device_put(jax.random.normal(jax.random.key(1), (n,), dtype=jnp.bfloat16), device)
    lowered = rmsnorm_rht_amax_sm100.lower(
        x,
        weight,
        eps=eps,
        num_threads=128,
        rows_per_cta=rows_per_cta,
    )
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo

    output, amax = lowered.compile()(x, weight)
    output.block_until_ready()
    x_f32 = x.astype(jnp.float32)
    normalized = x_f32 * jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
    normalized *= weight.astype(jnp.float32)[None, :]
    reference = (normalized.reshape(m, n // 16, 16) @ _hadamard_16(jnp)).reshape(m, n)
    amax_reference = jnp.max(
        jnp.abs(reference).reshape(m // rows_per_cta, rows_per_cta, n),
        axis=(1, 2),
    )

    assert jnp.allclose(output.astype(jnp.float32), reference, atol=4e-2, rtol=1e-2)
    assert jnp.allclose(amax, amax_reference, atol=2e-3, rtol=1e-3)
