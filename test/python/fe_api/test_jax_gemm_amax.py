# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX integration tests for block-scaled dense GEMM + amax."""

import pytest


def _jax_runtime():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    cutlass_jax = pytest.importorskip("cutlass.jax")
    if not cutlass_jax.is_available():
        pytest.skip("Installed JAX version is unsupported by CUTLASS JAX")
    if not hasattr(jnp, "float8_e8m0fnu"):
        pytest.skip("JAX does not provide the E8M0 scale-factor dtype")
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
    try:
        devices = tuple(jax.local_devices(backend="gpu"))
        capabilities = {_compute_capability(device) for device in devices}
    except RuntimeError:
        pytest.skip("A JAX GPU backend is not available")
    except (TypeError, ValueError):
        pytest.skip("JAX GPU compute capability is unavailable")
    if not devices or len(capabilities) != 1:
        pytest.skip("A homogeneous JAX SM100-family GPU configuration is required")
    capability = capabilities.pop()
    if capability not in {100, 103, 107}:
        pytest.skip("A supported JAX SM100-family GPU is not available")
    return devices[0]


def _scale_shape(batch, rows, k, sf_vec_size):
    return (
        batch,
        (rows + 127) // 128,
        ((k + sf_vec_size - 1) // sf_vec_size + 3) // 4,
        32,
        4,
        4,
    )


@pytest.mark.L0
def test_jax_gemm_amax_abstract_contract(monkeypatch):
    jax, jnp = _jax_runtime()

    from cudnn._jax import JaxApiBase
    from cudnn.jax import GemmAmaxSm100, gemm_amax_wrapper_sm100

    def abstract_call(self, _inputs, *, output_descs, output_spec, **_options):
        values = []
        for desc, spec in zip(output_descs, output_spec):
            shape = self._materialize_tensor_desc(desc, mode=spec.mode).shape
            fill = 0 if desc.init_value is None else desc.init_value
            values.append(jnp.full(shape, fill, dtype=desc.dtype))
        return tuple(values)

    monkeypatch.setattr(JaxApiBase, "_call_kernel", abstract_call)
    monkeypatch.setattr(
        JaxApiBase,
        "_resolve_compute_capability",
        staticmethod(lambda _target, _supported, _operation, **_options: 100),
    )
    monkeypatch.setattr(
        JaxApiBase,
        "_get_max_active_clusters",
        lambda self, _cluster_size, **_options: 8,
    )

    batch, m, n, k = 2, 128, 256, 128
    sample_a = jax.ShapeDtypeStruct((batch, m, k), jnp.float8_e4m3fn)
    sample_b = jax.ShapeDtypeStruct((batch, n, k), jnp.float8_e4m3fn)
    sample_sfa = jax.ShapeDtypeStruct(
        _scale_shape(batch, m, k, 32),
        jnp.float8_e8m0fnu,
    )
    sample_sfb = jax.ShapeDtypeStruct(
        _scale_shape(batch, n, k, 32),
        jnp.float8_e8m0fnu,
    )
    operation = GemmAmaxSm100(sample_a, sample_b, sample_sfa, sample_sfb)
    result = jax.eval_shape(
        operation,
        sample_a,
        sample_b,
        sample_sfa,
        sample_sfb,
    )
    assert tuple(result.keys()) == ("c_tensor", "amax_tensor")
    assert result["c_tensor"].shape == (batch, m, n)
    assert result["c_tensor"].dtype == jnp.float32
    assert result["amax_tensor"].shape == (1, 1, 1)
    assert result["amax_tensor"].dtype == jnp.float32

    alternate_a = jax.ShapeDtypeStruct((batch, k, m), jnp.float8_e4m3fn)
    alternate_b = jax.ShapeDtypeStruct((batch, k, n), jnp.float8_e4m3fn)
    alternate = jax.eval_shape(
        lambda a, b, sfa, sfb: gemm_amax_wrapper_sm100(
            a,
            b,
            sfa,
            sfb,
            a_layout="LKM",
            b_layout="LKN",
            c_layout="LNM",
            c_dtype=jnp.bfloat16,
        ),
        alternate_a,
        alternate_b,
        sample_sfa,
        sample_sfb,
    )
    assert alternate["c_tensor"].shape == (batch, n, m)
    assert alternate["c_tensor"].dtype == jnp.bfloat16


@pytest.mark.L0
def test_jax_gemm_amax_sm100_jit_numerics_and_amax_reset():
    jax, jnp = _jax_runtime()
    device = _supported_gpu(jax)

    from cudnn.jax import gemm_amax_wrapper_sm100

    batch, m, n, k = 1, 128, 128, 128
    a_base = jax.random.uniform(
        jax.random.key(0),
        (batch, m, k),
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float32,
    ).astype(jnp.float8_e4m3fn)
    b = jax.random.uniform(
        jax.random.key(1),
        (batch, n, k),
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float32,
    ).astype(jnp.float8_e4m3fn)
    a_large = (a_base.astype(jnp.float32) * 2).astype(jnp.float8_e4m3fn)
    sfa = jnp.ones(
        _scale_shape(batch, m, k, 32),
        dtype=jnp.float8_e8m0fnu,
    )
    sfb = jnp.ones(
        _scale_shape(batch, n, k, 32),
        dtype=jnp.float8_e8m0fnu,
    )
    a_base, a_large, b, sfa, sfb = (jax.device_put(value, device) for value in (a_base, a_large, b, sfa, sfb))

    lowered = gemm_amax_wrapper_sm100.lower(a_large, b, sfa, sfb)
    stablehlo = lowered.as_text("stablehlo")
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "CuteDSLRT_NvJaxCutlassCall" in stablehlo
    compiled = lowered.compile()

    def reference(a_value):
        c = jnp.einsum(
            "lmk,lnk->lmn",
            a_value.astype(jnp.float32),
            b.astype(jnp.float32),
        )
        return c, jnp.max(jnp.abs(c)).reshape(1, 1, 1)

    # Run the larger input first. The smaller second result proves that the
    # aliased amax seed is reset to -inf for each invocation.
    for a_value in (a_large, a_base):
        result = compiled(a_value, b, sfa, sfb)
        result["amax_tensor"].block_until_ready()
        reference_c, reference_amax = reference(a_value)
        assert jnp.allclose(result["c_tensor"], reference_c, atol=2e-1, rtol=5e-2)
        assert jnp.allclose(
            result["amax_tensor"],
            reference_amax,
            atol=2e-1,
            rtol=5e-2,
        )
