# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX integration tests for the standard dense GEMM + SwiGLU adapter."""

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
def test_jax_gemm_swiglu_abstract_contract(
    monkeypatch,
):
    jax, jnp = _jax_runtime()

    from cudnn._jax import JaxApiBase
    from cudnn.jax import GemmSwigluSm100, gemm_swiglu_wrapper_sm100

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
        staticmethod(lambda _target, _supported, _operation, **_options: 100),
    )

    sample_a = jax.ShapeDtypeStruct((3, 128, 64), jnp.bfloat16)
    sample_b = jax.ShapeDtypeStruct((3, 192, 64), jnp.bfloat16)
    api = GemmSwigluSm100(
        sample_a,
        sample_b,
    )
    result = jax.eval_shape(api, sample_a, sample_b)
    assert tuple(result.keys()) == (
        "ab12_tensor",
        "c_tensor",
        "sfc_tensor",
        "amax_tensor",
    )
    assert result["ab12_tensor"].shape == (3, 128, 192)
    assert result["ab12_tensor"].dtype == jnp.float32
    assert result["c_tensor"].shape == (3, 128, 96)
    assert result["c_tensor"].dtype == jnp.float16
    assert result["sfc_tensor"] is None
    assert result["amax_tensor"] is None

    alternate_a = jax.ShapeDtypeStruct((3, 64, 128), jnp.bfloat16)
    alternate_b = jax.ShapeDtypeStruct((3, 64, 192), jnp.bfloat16)
    alternate = jax.eval_shape(
        lambda a, b: gemm_swiglu_wrapper_sm100(
            a,
            b,
            a_layout="LKM",
            b_layout="LKN",
            c_layout="LNM",
            c_dtype=jnp.bfloat16,
        ),
        alternate_a,
        alternate_b,
    )
    assert alternate["ab12_tensor"].shape == (3, 192, 128)
    assert alternate["c_tensor"].shape == (3, 96, 128)
    assert alternate["c_tensor"].dtype == jnp.bfloat16

    explicit = GemmSwigluSm100(
        sample_a,
        sample_b,
        sample_ab12=jax.ShapeDtypeStruct((3, 128, 192), jnp.bfloat16),
        sample_c=jax.ShapeDtypeStruct((3, 128, 96), jnp.bfloat16),
    )
    explicit_result = jax.eval_shape(explicit, sample_a, sample_b)
    assert explicit_result["ab12_tensor"].dtype == jnp.bfloat16
    assert explicit_result["c_tensor"].dtype == jnp.bfloat16


@pytest.mark.L0
def test_jax_gemm_swiglu_sm100_jit_and_numerics():
    jax, jnp = _jax_runtime()
    device = _supported_gpu(jax)

    from cudnn.jax import gemm_swiglu_wrapper_sm100

    batch, m, n, k = 1, 128, 128, 128
    alpha = 0.5
    a = jax.device_put(
        jax.random.normal(
            jax.random.key(0),
            (batch, k, m),
            dtype=jnp.bfloat16,
        )
        * jnp.bfloat16(0.1),
        device,
    )
    b = jax.device_put(
        jax.random.normal(
            jax.random.key(1),
            (batch, k, n),
            dtype=jnp.bfloat16,
        )
        * jnp.bfloat16(0.1),
        device,
    )

    lowered = gemm_swiglu_wrapper_sm100.lower(
        a,
        b,
        alpha=alpha,
        a_layout="LKM",
        b_layout="LKN",
        c_layout="LNM",
    )
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo

    compiled = lowered.compile()

    def reference(a_value):
        ab12_lmn = alpha * jnp.einsum(
            "lmk,lnk->lmn",
            a_value.transpose(0, 2, 1).astype(jnp.float32),
            b.transpose(0, 2, 1).astype(jnp.float32),
        )
        blocks = ab12_lmn.reshape(batch, m, n // 32, 32)
        input_blocks = blocks[:, :, 0::2].reshape(batch, m, n // 2)
        gate_blocks = blocks[:, :, 1::2].reshape(batch, m, n // 2)
        c_lmn = input_blocks * jax.nn.silu(gate_blocks)
        return ab12_lmn.transpose(0, 2, 1), c_lmn.transpose(0, 2, 1)

    for a_value in (a, -a):
        result = compiled(a_value, b)
        result["c_tensor"].block_until_ready()
        reference_ab12, reference_c = reference(a_value)
        assert jnp.allclose(
            result["ab12_tensor"],
            reference_ab12,
            atol=5e-2,
            rtol=2e-2,
        )
        assert jnp.allclose(
            result["c_tensor"].astype(jnp.float32),
            reference_c,
            atol=8e-2,
            rtol=3e-2,
        )
