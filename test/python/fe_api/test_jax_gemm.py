# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX shape and GPU integration tests for dense GEMM fusions."""

from __future__ import annotations

import pytest


def _jax_dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("cutlass.jax")
    return jax, jnp


def _sm100_device(jax):
    for device in jax.local_devices():
        if device.platform != "gpu":
            continue
        capability = getattr(device, "compute_capability", None)
        if capability is None:
            continue
        if isinstance(capability, (tuple, list)):
            major, minor = (int(value) for value in capability[:2])
            capability_number = major * 10 + minor
        elif "." in str(capability):
            major, minor = (int(value) for value in str(capability).split(".", 1))
            capability_number = major * 10 + minor
        else:
            capability_number = int(capability)
            if capability_number < 10:
                capability_number *= 10
        if capability_number >= 100:
            return device
    pytest.skip("JAX SM100+ device is not available")


@pytest.mark.L0
def test_jax_dense_gemm_abstract_shapes():
    jax, jnp = _jax_dependencies()

    from cudnn.jax import (
        gemm_amax_wrapper_sm100,
        gemm_dsrelu_wrapper_sm100,
        gemm_srelu_wrapper_sm100,
        gemm_swiglu_wrapper_sm100,
    )

    a_bf16 = jax.ShapeDtypeStruct((128, 128, 1), jnp.bfloat16)
    b_bf16 = jax.ShapeDtypeStruct((128, 128, 1), jnp.bfloat16)
    swiglu = jax.eval_shape(gemm_swiglu_wrapper_sm100, a_bf16, b_bf16)
    assert swiglu.ab12_tensor.shape == (128, 128, 1)
    assert swiglu.ab12_tensor.dtype == jnp.float32
    assert swiglu.c_tensor.shape == (128, 64, 1)
    assert swiglu.c_tensor.dtype == jnp.float16
    assert swiglu.sfc_tensor is None
    assert swiglu.amax_tensor is None

    a_fp8 = jax.ShapeDtypeStruct((256, 512, 2), jnp.float8_e4m3fn)
    b_fp8 = jax.ShapeDtypeStruct((256, 512, 2), jnp.float8_e4m3fn)
    scales = jax.ShapeDtypeStruct((32, 4, 2, 4, 4, 2), jnp.float8_e8m0fnu)
    prob = jax.ShapeDtypeStruct((256, 1, 2), jnp.float32)
    c = jax.ShapeDtypeStruct((256, 256, 2), jnp.bfloat16)

    amax = jax.eval_shape(
        gemm_amax_wrapper_sm100,
        a_fp8,
        b_fp8,
        scales,
        scales,
    )
    assert amax.c_tensor.shape == (256, 256, 2)
    assert amax.c_tensor.dtype == jnp.float32
    assert amax.amax_tensor.shape == (1, 1, 1)
    assert amax.amax_tensor.dtype == jnp.float32

    srelu = jax.eval_shape(
        gemm_srelu_wrapper_sm100,
        a_fp8,
        b_fp8,
        scales,
        scales,
        prob,
    )
    assert srelu.c_tensor.shape == (256, 256, 2)
    assert srelu.d_tensor.shape == (256, 256, 2)
    assert srelu.c_tensor.dtype == jnp.bfloat16
    assert srelu.d_tensor.dtype == jnp.bfloat16
    assert srelu.amax_tensor is None
    assert srelu.sfd_tensor is None

    dsrelu = jax.eval_shape(
        gemm_dsrelu_wrapper_sm100,
        a_fp8,
        b_fp8,
        c,
        scales,
        scales,
        prob,
    )
    assert dsrelu.d_tensor.shape == (256, 256, 2)
    assert dsrelu.d_tensor.dtype == jnp.bfloat16
    assert dsrelu.dprob_tensor.shape == (256, 1, 2)
    assert dsrelu.dprob_tensor.dtype == jnp.float32
    assert dsrelu.amax_tensor is None
    assert dsrelu.sfd_tensor is None


@pytest.mark.L0
def test_jax_gemm_swiglu_jit():
    jax, jnp = _jax_dependencies()
    device = _sm100_device(jax)

    from cudnn.jax import gemm_swiglu_wrapper_sm100

    m = n = k = 128
    batch = 1
    alpha = 0.5
    a = jax.device_put(
        jax.random.normal(jax.random.key(0), (m, k, batch), dtype=jnp.bfloat16),
        device,
    )
    b = jax.device_put(
        jax.random.normal(jax.random.key(1), (n, k, batch), dtype=jnp.bfloat16),
        device,
    )

    @jax.jit
    def run(a, b):
        return gemm_swiglu_wrapper_sm100(
            a,
            b,
            alpha=alpha,
            ab12_dtype=jnp.float32,
            c_dtype=jnp.bfloat16,
        )

    lowered = run.lower(a, b)
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo
    compiled = lowered.compile()

    def reference(a_value, b_value):
        ab12 = alpha * jnp.einsum(
            "mkl,nkl->mnl",
            a_value.astype(jnp.float32),
            b_value.astype(jnp.float32),
        )
        blocks = ab12.reshape(m, n // 32, 32, batch)
        input_blocks = blocks[:, 0::2].reshape(m, n // 2, batch)
        gate_blocks = blocks[:, 1::2].reshape(m, n // 2, batch)
        c = input_blocks * gate_blocks * jax.nn.sigmoid(gate_blocks)
        return ab12, c

    for a_value in (a, -a):
        result = compiled(a_value, b)
        result.c_tensor.block_until_ready()
        expected_ab12, expected_c = reference(a_value, b)
        assert jnp.allclose(result.ab12_tensor, expected_ab12, atol=5e-2, rtol=2e-2)
        assert jnp.allclose(
            result.c_tensor.astype(jnp.float32),
            expected_c,
            atol=8e-2,
            rtol=3e-2,
        )


@pytest.mark.L0
def test_jax_gemm_amax_jit_reinitializes_reduction():
    jax, jnp = _jax_dependencies()
    device = _sm100_device(jax)

    from cudnn.jax import gemm_amax_wrapper_sm100

    m = n = k = 128
    batch = 1
    fp8_dtype = jnp.float8_e4m3fn
    a = jax.device_put(
        jax.random.uniform(
            jax.random.key(2),
            (m, k, batch),
            dtype=jnp.float32,
            minval=-1.0,
            maxval=1.0,
        ).astype(fp8_dtype),
        device,
    )
    b = jax.device_put(
        jax.random.uniform(
            jax.random.key(3),
            (n, k, batch),
            dtype=jnp.float32,
            minval=-1.0,
            maxval=1.0,
        ).astype(fp8_dtype),
        device,
    )
    sf_k = k // 32

    def pack_scales(canonical):
        return canonical.reshape(1, 4, 32, 1, 4, batch).transpose(2, 1, 0, 4, 3, 5).astype(jnp.float8_e8m0fnu)

    row = jnp.arange(m, dtype=jnp.int32)[:, None, None]
    scale_column = jnp.arange(sf_k, dtype=jnp.int32)[None, :, None]
    sfa_canonical = jnp.where((row + scale_column) % 2, 2.0, 1.0)
    sfb_canonical = jnp.where((row + 2 * scale_column) % 2, 1.0, 0.5)
    sfa = jax.device_put(pack_scales(sfa_canonical), device)
    sfb = jax.device_put(pack_scales(sfb_canonical), device)

    @jax.jit
    def run(a, b, sfa, sfb):
        return gemm_amax_wrapper_sm100(a, b, sfa, sfb)

    lowered = run.lower(a, b, sfa, sfb)
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo
    compiled = lowered.compile()

    result = compiled(a, b, sfa, sfb)
    result.c_tensor.block_until_ready()
    expected = jnp.einsum(
        "mkl,nkl->mnl",
        a.astype(jnp.float32) * jnp.repeat(sfa_canonical, 32, axis=1),
        b.astype(jnp.float32) * jnp.repeat(sfb_canonical, 32, axis=1),
    )
    expected_amax = jnp.max(jnp.abs(expected)).reshape(1, 1, 1)
    assert jnp.allclose(result.c_tensor, expected, atol=2e-1, rtol=3e-2)
    assert jnp.allclose(result.amax_tensor, expected_amax, atol=5e-1, rtol=3e-2)

    zero_result = compiled(a, jnp.zeros_like(b), sfa, sfb)
    zero_result.amax_tensor.block_until_ready()
    assert jnp.array_equal(zero_result.c_tensor, jnp.zeros_like(zero_result.c_tensor))
    assert jnp.array_equal(
        zero_result.amax_tensor,
        jnp.zeros_like(zero_result.amax_tensor),
    )


@pytest.mark.L0
def test_jax_gemm_squared_relu_forward_and_backward_jit():
    jax, jnp = _jax_dependencies()
    device = _sm100_device(jax)

    from cudnn.jax import gemm_dsrelu_wrapper_sm100, gemm_srelu_wrapper_sm100

    m = n = k = 128
    batch = 1
    fp8_dtype = jnp.float8_e4m3fn
    a = jax.device_put(
        jax.random.uniform(
            jax.random.key(4),
            (m, k, batch),
            dtype=jnp.float32,
            minval=-0.5,
            maxval=0.5,
        ).astype(fp8_dtype),
        device,
    )
    b = jax.device_put(
        jax.random.uniform(
            jax.random.key(5),
            (n, k, batch),
            dtype=jnp.float32,
            minval=-0.5,
            maxval=0.5,
        ).astype(fp8_dtype),
        device,
    )
    c_input = jax.device_put(
        jax.random.uniform(
            jax.random.key(6),
            (m, n, batch),
            dtype=jnp.float32,
            minval=-0.5,
            maxval=0.5,
        ).astype(jnp.bfloat16),
        device,
    )
    prob = jax.device_put(
        jax.random.uniform(
            jax.random.key(7),
            (m, 1, batch),
            dtype=jnp.float32,
            minval=0.25,
            maxval=1.0,
        ),
        device,
    )
    scales = jax.device_put(
        jnp.ones((32, 4, 1, 4, 1, batch), dtype=jnp.float32).astype(jnp.float8_e8m0fnu),
        device,
    )

    @jax.jit
    def forward(a, b, sfa, sfb, prob):
        return gemm_srelu_wrapper_sm100(
            a,
            b,
            sfa,
            sfb,
            prob,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )

    @jax.jit
    def backward(a, b, c, sfa, sfb, prob):
        return gemm_dsrelu_wrapper_sm100(
            a,
            b,
            c,
            sfa,
            sfb,
            prob,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )

    forward_lowered = forward.lower(a, b, scales, scales, prob)
    backward_lowered = backward.lower(a, b, c_input, scales, scales, prob)
    for lowered in (forward_lowered, backward_lowered):
        stablehlo = lowered.as_text("stablehlo")
        assert stablehlo.count("stablehlo.custom_call") == 1
        assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo

    x = jnp.einsum(
        "mkl,nkl->mnl",
        a.astype(jnp.float32),
        b.astype(jnp.float32),
    )
    relu_x = jnp.maximum(x, 0.0)

    forward_result = forward_lowered.compile()(a, b, scales, scales, prob)
    forward_result.d_tensor.block_until_ready()
    expected_forward = jnp.square(relu_x) * prob
    assert jnp.allclose(
        forward_result.c_tensor.astype(jnp.float32),
        x,
        atol=2e-1,
        rtol=3e-2,
    )
    assert jnp.allclose(
        forward_result.d_tensor.astype(jnp.float32),
        expected_forward,
        atol=4e-1,
        rtol=5e-2,
    )

    backward_compiled = backward_lowered.compile()
    backward_result = backward_compiled(
        a,
        b,
        c_input,
        scales,
        scales,
        prob,
    )
    backward_result.dprob_tensor.block_until_ready()
    c_f32 = c_input.astype(jnp.float32)
    expected_d = c_f32 * prob * 2 * relu_x
    expected_dprob = jnp.sum(c_f32 * jnp.square(relu_x), axis=1, keepdims=True)
    assert jnp.allclose(
        backward_result.d_tensor.astype(jnp.float32),
        expected_d,
        atol=4e-1,
        rtol=5e-2,
    )
    assert jnp.allclose(
        backward_result.dprob_tensor,
        expected_dprob,
        atol=2.0,
        rtol=5e-2,
    )

    zero_result = backward_compiled(
        a,
        b,
        jnp.zeros_like(c_input),
        scales,
        scales,
        prob,
    )
    zero_result.dprob_tensor.block_until_ready()
    assert jnp.array_equal(
        zero_result.dprob_tensor,
        jnp.zeros_like(zero_result.dprob_tensor),
    )
