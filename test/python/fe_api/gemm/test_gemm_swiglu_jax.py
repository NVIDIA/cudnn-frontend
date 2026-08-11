# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX tests for the type-erased GemmSwigluSm100 API and the gemm_swiglu_jax_sm100
XLA custom-call entry point. Same JAX contract as gemm_amax (see
test_gemm_amax_jax.py): k-major (mn, k, 1) A/B, n-major outputs, L == 1, SF tensors
in the physical C-contiguous atom shape.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import (
    device_sync,
    make_ab_fp8,
    make_sf_physical,
    skip_unless_sm100,
)


def swiglu_block_ref(ab12_ref, n):
    """c = input * silu(gate) with 32-column block interleaving (even blocks input, odd gate)."""
    cols = np.arange(n).reshape(n // 32, 32)
    input_idx, gate_idx = cols[0::2].reshape(-1), cols[1::2].reshape(-1)
    gate = ab12_ref[:, gate_idx]
    return ab12_ref[:, input_idx] * (gate / (1 + np.exp(-gate)))


def make_ab_bf16(mn, k, rng):
    values = (rng.integers(-4, 5, size=(mn, k, 1)) * 0.25).astype(ml_dtypes.bfloat16)
    return values, values.astype(np.float32)[:, :, 0]


@pytest.mark.L0
def test_gemm_swiglu_jax_wrapper_bf16():
    """Eager wrapper, non-quantized kernel, bf16 inputs."""
    skip_unless_sm100()
    from cudnn import gemm_swiglu_wrapper_sm100

    m, n, k = 256, 256, 256
    alpha = 1.0
    rng = np.random.default_rng(0)
    a_np, a_ref = make_ab_bf16(m, k, rng)
    b_np, b_ref = make_ab_bf16(n, k, rng)
    a_dev, b_dev = jax.device_put(a_np), jax.device_put(b_np)
    jax.block_until_ready((a_dev, b_dev))

    ab12_ref = alpha * (a_ref @ b_ref.T)
    c_ref = swiglu_block_ref(ab12_ref, n)

    for _ in range(2):  # exercise the compile cache path
        ab12, c, sfc, amax = gemm_swiglu_wrapper_sm100(
            a_tensor=a_dev,
            b_tensor=b_dev,
            alpha=alpha,
            c_major="n",
            ab12_dtype=np.float32,
            c_dtype=ml_dtypes.bfloat16,
        )
        device_sync()  # eager path runs on the CUDA legacy default stream
        assert sfc is None and amax is None
        np.testing.assert_allclose(np.asarray(ab12)[:, :, 0], ab12_ref, atol=0.02, rtol=0.02)
        np.testing.assert_allclose(np.asarray(c).astype(np.float32)[:, :, 0], c_ref, atol=0.05, rtol=0.05)


@pytest.mark.L0
def test_gemm_swiglu_jax_wrapper_quant_fp8():
    """Eager wrapper, blockscaled quantized kernel, MXFP8 inputs."""
    skip_unless_sm100()
    from cudnn import gemm_swiglu_wrapper_sm100

    m, n, k = 256, 256, 256
    sf_vec_size = 32
    rng = np.random.default_rng(1)
    a_np, a_ref = make_ab_fp8(m, k, ml_dtypes.float8_e4m3fn, rng)
    b_np, b_ref = make_ab_fp8(n, k, ml_dtypes.float8_e4m3fn, rng)
    sfa_np, sfa_expanded = make_sf_physical(m, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)
    sfb_np, sfb_expanded = make_sf_physical(n, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)
    a_dev, b_dev, sfa_dev, sfb_dev = (jax.device_put(x) for x in (a_np, b_np, sfa_np, sfb_np))
    jax.block_until_ready((a_dev, b_dev, sfa_dev, sfb_dev))

    ab12_ref = (a_ref * sfa_expanded) @ (b_ref * sfb_expanded).T
    c_ref = swiglu_block_ref(ab12_ref, n)

    ab12, c, sfc, amax = gemm_swiglu_wrapper_sm100(
        a_tensor=a_dev,
        b_tensor=b_dev,
        c_major="n",
        ab12_dtype=ml_dtypes.bfloat16,
        c_dtype=ml_dtypes.bfloat16,
        sfa_tensor=sfa_dev,
        sfb_tensor=sfb_dev,
        sf_vec_size=sf_vec_size,
    )
    device_sync()
    assert sfc is None and amax is None
    np.testing.assert_allclose(np.asarray(ab12).astype(np.float32)[:, :, 0], ab12_ref, atol=0.5, rtol=0.05)
    np.testing.assert_allclose(np.asarray(c).astype(np.float32)[:, :, 0], c_ref, atol=1.0, rtol=0.05)


@pytest.mark.L0
def test_gemm_swiglu_jax_jit_sm100():
    """XLA custom-call entry point: jitted, repeated (donation safety), and alpha attr."""
    import cutlass.jax

    if not cutlass.jax.is_available():
        pytest.skip("CuTeDSL JAX extensions unavailable (jax >= 0.5 required)")
    skip_unless_sm100()
    from cudnn import gemm_swiglu_jax_sm100

    m, n, k = 256, 256, 256
    alpha = 2.0
    rng = np.random.default_rng(2)
    a_np, a_ref = make_ab_bf16(m, k, rng)
    b_np, b_ref = make_ab_bf16(n, k, rng)
    a_dev, b_dev = jax.device_put(a_np), jax.device_put(b_np)

    ab12_ref = alpha * (a_ref @ b_ref.T)
    c_ref = swiglu_block_ref(ab12_ref, n)

    jitted = jax.jit(lambda a, b: gemm_swiglu_jax_sm100(a, b, alpha=alpha, ab12_dtype=jnp.float32, c_dtype=jnp.bfloat16))
    for _ in range(2):
        ab12, c = jitted(a_dev, b_dev)
        jax.block_until_ready((ab12, c))  # XLA-stream ordered; no manual device sync needed
        np.testing.assert_allclose(np.asarray(ab12)[:, :, 0], ab12_ref, atol=0.02, rtol=0.02)
        np.testing.assert_allclose(np.asarray(c).astype(np.float32)[:, :, 0], c_ref, atol=0.05, rtol=0.05)

    # Quantized (blockscaled MXFP8) config through the same entry point, under jit
    sf_vec_size = 32
    a2_np, a2_ref = make_ab_fp8(m, k, ml_dtypes.float8_e4m3fn, rng)
    b2_np, b2_ref = make_ab_fp8(n, k, ml_dtypes.float8_e4m3fn, rng)
    sfa_np, sfa_expanded = make_sf_physical(m, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)
    sfb_np, sfb_expanded = make_sf_physical(n, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)
    a2, b2, sfa, sfb = (jax.device_put(x) for x in (a2_np, b2_np, sfa_np, sfb_np))

    quant = jax.jit(
        lambda a, b, sfa, sfb: gemm_swiglu_jax_sm100(
            a, b, ab12_dtype=jnp.bfloat16, c_dtype=jnp.bfloat16, sfa_tensor=sfa, sfb_tensor=sfb, sf_vec_size=sf_vec_size
        )
    )
    ab12q, cq = quant(a2, b2, sfa, sfb)
    jax.block_until_ready((ab12q, cq))
    ab12q_ref = (a2_ref * sfa_expanded) @ (b2_ref * sfb_expanded).T
    np.testing.assert_allclose(np.asarray(ab12q).astype(np.float32)[:, :, 0], ab12q_ref, atol=0.5, rtol=0.05)
    np.testing.assert_allclose(np.asarray(cq).astype(np.float32)[:, :, 0], swiglu_block_ref(ab12q_ref, n), atol=1.0, rtol=0.05)


@pytest.mark.L0
def test_gemm_swiglu_jax_wrapper_errors():
    skip_unless_sm100()
    from cudnn import gemm_swiglu_wrapper_sm100

    m, n, k = 256, 256, 256
    rng = np.random.default_rng(3)
    a_np, _ = make_ab_bf16(m, k, rng)
    b_np, _ = make_ab_bf16(n, k, rng)
    a_dev, b_dev = jax.device_put(a_np), jax.device_put(b_np)

    with pytest.raises(ValueError, match="row-major"):
        gemm_swiglu_wrapper_sm100(a_dev, b_dev, c_major="m")

    a_batched = jax.device_put(np.repeat(a_np, 2, axis=2))
    b_batched = jax.device_put(np.repeat(b_np, 2, axis=2))
    with pytest.raises(ValueError, match="batch dim L == 1"):
        gemm_swiglu_wrapper_sm100(a_batched, b_batched)

    with pytest.raises(ValueError, match="Unsupported tensor framework"):
        gemm_swiglu_wrapper_sm100(a_np, b_np)
