# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX integration tests for dense block-scaled sReLU fusions."""

import pytest


def _jax_runtime():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    cutlass_jax = pytest.importorskip("cutlass.jax")
    if not cutlass_jax.is_available():
        pytest.skip("Installed JAX version is unsupported by CUTLASS JAX")
    return jax, jnp


def _compute_capability(device):
    reported = getattr(device, "compute_capability", None)
    if isinstance(reported, (tuple, list)):
        major, minor = int(reported[0]), int(reported[1])
    else:
        text = str(reported)
        if "." in text:
            major_text, minor_text = text.split(".", 1)
            major, minor = int(major_text), int(minor_text)
        else:
            major, minor = divmod(int(text), 10)
    return major * 10 + minor


def _supported_gpu(jax):
    devices = tuple(device for device in jax.local_devices() if device.platform == "gpu")
    try:
        capabilities = {_compute_capability(device) for device in devices}
    except (TypeError, ValueError):
        pytest.skip("JAX GPU compute capability is unavailable")
    if not devices or len(capabilities) != 1:
        pytest.skip("A homogeneous JAX SM100-family GPU configuration is required")
    capability = capabilities.pop()
    if capability not in {100, 103, 107}:
        pytest.skip("A supported JAX SM100-family GPU is not available")
    return devices[0]


@pytest.mark.L0
def test_jax_gemm_relu_abstract_contract(monkeypatch):
    jax, jnp = _jax_runtime()

    from cudnn._jax import JaxApiBase
    from cudnn.jax import GemmDsreluSm100, GemmSreluSm100, gemm_dsrelu_wrapper_sm100, gemm_srelu_wrapper_sm100

    def abstract_call(self, _inputs, *, output_descs, output_spec, **_options):
        return tuple(
            jnp.empty(
                self._materialize_tensor_desc(desc, mode=spec.mode).shape,
                dtype=desc.dtype,
            )
            for desc, spec in zip(output_descs, output_spec)
        )

    monkeypatch.setattr(JaxApiBase, "_call_kernel", abstract_call)
    monkeypatch.setattr(
        JaxApiBase,
        "_resolve_compute_capability",
        staticmethod(lambda _target, _supported, _operation: 100),
    )
    monkeypatch.setattr(JaxApiBase, "_get_max_active_clusters", lambda self, _size, **_options: 8)

    batch, m, n, k = 2, 256, 256, 512
    a = jax.ShapeDtypeStruct((batch, m, k), jnp.float8_e4m3fn)
    b = jax.ShapeDtypeStruct((batch, n, k), jnp.float8_e4m3fn)
    scales = jax.ShapeDtypeStruct((batch, m // 128, k // 128, 32, 4, 4), jnp.float8_e8m0fnu)
    prob = jax.ShapeDtypeStruct((batch, 1, m), jnp.float32)
    c = jax.ShapeDtypeStruct((batch, m, n), jnp.bfloat16)

    forward = jax.eval_shape(
        GemmSreluSm100(a, b, scales, scales, prob, sf_vec_size=32),
        a,
        b,
        scales,
        scales,
        prob,
    )
    backward = jax.eval_shape(
        GemmDsreluSm100(a, b, c, scales, scales, prob, sf_vec_size=32),
        a,
        b,
        c,
        scales,
        scales,
        prob,
    )
    assert tuple(forward.keys()) == ("c_tensor", "d_tensor", "amax_tensor", "sfd_tensor")
    assert forward["c_tensor"].shape == (batch, m, n)
    assert forward["d_tensor"].shape == (batch, m, n)
    assert forward["amax_tensor"] is None
    assert backward["d_tensor"].shape == (batch, m, n)
    assert backward["dprob_tensor"].shape == (batch, 1, m)

    alternate = jax.eval_shape(
        lambda a_value, b_value, scale_a, scale_b, probability: gemm_srelu_wrapper_sm100(
            a_value,
            b_value,
            scale_a,
            scale_b,
            probability,
            c_layout="LNM",
            sf_vec_size=32,
        ),
        jax.ShapeDtypeStruct((batch, k, m), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((batch, k, n), jnp.float8_e4m3fn),
        scales,
        scales,
        prob,
    )
    assert alternate["c_tensor"].shape == (batch, n, m)

    norm_const = jax.ShapeDtypeStruct((1,), jnp.float32)
    with pytest.raises(NotImplementedError, match="does not implement SFD generation"):
        jax.eval_shape(
            lambda *args: gemm_dsrelu_wrapper_sm100(
                *args[:-1],
                d_dtype=jnp.float8_e4m3fn,
                norm_const_tensor=args[-1],
                sf_vec_size=32,
            ),
            a,
            b,
            c,
            scales,
            scales,
            prob,
            norm_const,
        )

    fp4_dtype = getattr(jnp, "float4_e2m1fn", None)
    if fp4_dtype is not None:
        fp4_a = jax.ShapeDtypeStruct((1, 128, 128), fp4_dtype)
        fp4_b = jax.ShapeDtypeStruct((1, 128, 128), fp4_dtype)
        fp4_scales = jax.ShapeDtypeStruct((1, 1, 2, 32, 4, 4), jnp.float8_e8m0fnu)
        fp4_prob = jax.ShapeDtypeStruct((1, 1, 128), jnp.float32)
        fp4_result = jax.eval_shape(
            lambda *args: gemm_srelu_wrapper_sm100(
                *args,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
            ),
            fp4_a,
            fp4_b,
            fp4_scales,
            fp4_scales,
            fp4_prob,
        )
        assert fp4_result["amax_tensor"].shape == (1,)


@pytest.mark.L0
def test_jax_gemm_relu_sm100_jit_and_numerics():
    jax, jnp = _jax_runtime()
    device = _supported_gpu(jax)

    from cudnn.jax import gemm_dsrelu_wrapper_sm100, gemm_srelu_wrapper_sm100

    batch = 1
    m = n = k = 128
    alpha = 0.5
    fp8_dtype = jnp.float8_e4m3fn
    a = jax.device_put(
        jax.random.uniform(jax.random.key(0), (batch, m, k), minval=-0.5, maxval=0.5).astype(fp8_dtype),
        device,
    )
    b = jax.device_put(
        jax.random.uniform(jax.random.key(1), (batch, n, k), minval=-0.5, maxval=0.5).astype(fp8_dtype),
        device,
    )
    c = jax.device_put(
        jax.random.uniform(jax.random.key(2), (batch, m, n), minval=-0.5, maxval=0.5).astype(jnp.bfloat16),
        device,
    )
    prob = jax.device_put(
        jax.random.uniform(jax.random.key(3), (batch, 1, m), minval=0.25, maxval=1.0),
        device,
    )
    scales = jax.device_put(
        jnp.ones((batch, 1, 1, 32, 4, 4), dtype=jnp.float32).astype(jnp.float8_e8m0fnu),
        device,
    )

    forward_lowered = gemm_srelu_wrapper_sm100.lower(
        a,
        b,
        scales,
        scales,
        prob,
        alpha=alpha,
        sf_vec_size=32,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )
    backward_lowered = gemm_dsrelu_wrapper_sm100.lower(
        a,
        b,
        c,
        scales,
        scales,
        prob,
        alpha=alpha,
        sf_vec_size=32,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )
    for lowered in (forward_lowered, backward_lowered):
        stablehlo = lowered.as_text("stablehlo")
        assert stablehlo.count("stablehlo.custom_call") == 1
        assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo

    gemm = alpha * jnp.einsum("lmk,lnk->lmn", a.astype(jnp.float32), b.astype(jnp.float32))
    relu_gemm = jnp.maximum(gemm, 0.0)
    probability = prob.transpose(0, 2, 1)

    forward_result = forward_lowered.compile()(a, b, scales, scales, prob)
    forward_result["d_tensor"].block_until_ready()
    assert jnp.allclose(forward_result["c_tensor"].astype(jnp.float32), gemm, atol=2e-1, rtol=3e-2)
    assert jnp.allclose(
        forward_result["d_tensor"].astype(jnp.float32),
        jnp.square(relu_gemm) * probability,
        atol=4e-1,
        rtol=5e-2,
    )

    backward_compiled = backward_lowered.compile()
    backward_result = backward_compiled(a, b, c, scales, scales, prob)
    backward_result["dprob_tensor"].block_until_ready()
    expected_d = c.astype(jnp.float32) * probability * 2 * relu_gemm
    expected_dprob = jnp.sum(c.astype(jnp.float32) * jnp.square(relu_gemm), axis=2, keepdims=True).transpose(0, 2, 1)
    assert jnp.allclose(backward_result["d_tensor"].astype(jnp.float32), expected_d, atol=4e-1, rtol=5e-2)
    assert jnp.allclose(backward_result["dprob_tensor"], expected_dprob, atol=2.0, rtol=5e-2)

    zero_result = backward_compiled(a, b, jnp.zeros_like(c), scales, scales, prob)
    zero_result["dprob_tensor"].block_until_ready()
    assert jnp.array_equal(zero_result["dprob_tensor"], jnp.zeros_like(zero_result["dprob_tensor"]))


@pytest.mark.L0
def test_jax_gemm_srelu_native_fp4_amax_smoke():
    jax, jnp = _jax_runtime()
    device = _supported_gpu(jax)
    fp4_dtype = getattr(jnp, "float4_e2m1fn", None)
    if fp4_dtype is None:
        pytest.skip("JAX does not provide native float4_e2m1fn")

    from cudnn.jax import gemm_srelu_wrapper_sm100

    m = n = k = 128
    a = jax.device_put(jnp.ones((1, m, k), dtype=jnp.float32).astype(fp4_dtype), device)
    b = jax.device_put(jnp.ones((1, n, k), dtype=jnp.float32).astype(fp4_dtype), device)
    scales = jax.device_put(
        jnp.ones((1, 1, 2, 32, 4, 4), dtype=jnp.float32).astype(jnp.float8_e8m0fnu),
        device,
    )
    prob = jax.device_put(jnp.ones((1, 1, m), dtype=jnp.float32), device)
    lowered = gemm_srelu_wrapper_sm100.lower(
        a,
        b,
        scales,
        scales,
        prob,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )
    result = lowered.compile()(a, b, scales, scales, prob)
    result["amax_tensor"].block_until_ready()
    assert result["amax_tensor"].shape == (1,)
    assert jnp.all(jnp.isfinite(result["amax_tensor"]))
