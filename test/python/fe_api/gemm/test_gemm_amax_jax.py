# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX tests for the type-erased GemmAmaxSm100 / gemm_amax_wrapper_sm100 API.

JAX arrays are always row-major, so the JAX contract differs from torch in a few
documented ways: A/B are k-major (mn, k, 1), C is n-major only, batch dim L == 1,
scale factors are passed in the physical C-contiguous atom shape
(L, MN', K', 32, 4, 4), and packed fp4 uses a uint8 container tensor.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
import jax.numpy as jnp

ATOM_M = (32, 4)
ATOM_K = 4


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def skip_unless_sm100():
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("JAX has no CUDA device")
    from cudnn.tensor_adapter import get_compute_capability

    major, _ = get_compute_capability()
    if major < 10:
        pytest.skip(f"Environment not supported: requires compute capability >= 10, found {major}")


def make_sf_physical(mn, k, sf_vec_size, sf_dtype, rng):
    """Build a scale-factor tensor in the physical C-contiguous atom shape
    (1, MN', K', 32, 4, 4), plus its (mn, k) f32 expansion for the reference.

    The logical torch-style view is this allocation permuted by (3, 4, 1, 5, 2, 0),
    with mn = a0 + 32*a1 + 128*rm and sf_k = ak + 4*rk.
    """
    sf_k = ceil_div(k, sf_vec_size)
    mn_div = ceil_div(mn, ATOM_M[0] * ATOM_M[1])
    k_div = ceil_div(sf_k, ATOM_K)

    exponents = rng.integers(-2, 3, size=(1, mn_div, k_div, ATOM_M[0], ATOM_M[1], ATOM_K))
    sf_np = (2.0**exponents).astype(sf_dtype)
    sf_f32 = sf_np.astype(np.float32)

    # (rm, rk, a0, a1, ak) -> (rm, a1, a0, rk, ak) -> (mn_padded, sf_k_padded)
    sf_2d = np.transpose(sf_f32[0], (0, 3, 2, 1, 4)).reshape(mn_div * ATOM_M[0] * ATOM_M[1], k_div * ATOM_K)
    sf_expanded = np.repeat(sf_2d[:mn, :sf_k], sf_vec_size, axis=1)[:, :k]
    return sf_np, sf_expanded


def make_ab_fp8(mn, k, ab_dtype, rng):
    """(mn, k, 1) C-contiguous fp8 tensor with values exactly representable in fp8."""
    values = (rng.integers(-4, 5, size=(mn, k, 1)) * 0.25).astype(np.float32)
    quantized = values.astype(ab_dtype)
    return quantized, quantized.astype(np.float32)[:, :, 0]


FP4_E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def make_ab_fp4_uint8(mn, k, rng):
    """(mn, k // 2, 1) C-contiguous uint8 tensor holding packed fp4 (e2m1) pairs,
    element 0 in the low nibble, plus the (mn, k) f32 decode for the reference."""
    codes = rng.integers(0, 16, size=(mn, k)).astype(np.uint8)
    decoded = FP4_E2M1_VALUES[codes & 0x7] * np.where(codes & 0x8, -1.0, 1.0).astype(np.float32)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)[:, :, None]
    return packed, decoded


def reference_gemm_amax(a_f32, b_f32, sfa_expanded, sfb_expanded):
    c_ref = (a_f32 * sfa_expanded) @ (b_f32 * sfb_expanded).T
    amax_ref = np.max(np.abs(c_ref))
    return c_ref, amax_ref


def device_sync():
    from cuda.bindings import runtime as cudart

    (err,) = cudart.cudaDeviceSynchronize()
    assert err == cudart.cudaError_t.cudaSuccess


def run_wrapper_and_check(a_np, a_ref, b_np, b_ref, sfa_np, sfa_expanded, sfb_np, sfb_expanded, c_dtype, sf_vec_size, m, n):
    from cudnn import gemm_amax_wrapper_sm100

    a_dev = jax.device_put(a_np)
    b_dev = jax.device_put(b_np)
    sfa_dev = jax.device_put(sfa_np)
    sfb_dev = jax.device_put(sfb_np)
    jax.block_until_ready((a_dev, b_dev, sfa_dev, sfb_dev))

    c_ref, amax_ref = reference_gemm_amax(a_ref, b_ref, sfa_expanded, sfb_expanded)

    # Run twice to exercise the compile cache path
    for _ in range(2):
        c_dev, amax_dev = gemm_amax_wrapper_sm100(
            a_tensor=a_dev,
            b_tensor=b_dev,
            sfa_tensor=sfa_dev,
            sfb_tensor=sfb_dev,
            c_major="n",
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
        )
        # The kernel runs on the CUDA legacy default stream, outside XLA's tracking
        device_sync()

        assert c_dev.shape == (m, n, 1)
        c_out = np.asarray(c_dev).astype(np.float32)[:, :, 0]
        amax_out = float(np.asarray(amax_dev).reshape(()))
        np.testing.assert_allclose(c_out, c_ref.astype(np.asarray(c_dev).dtype).astype(np.float32), atol=0.01, rtol=0.01)
        np.testing.assert_allclose(amax_out, amax_ref, atol=1e-1, rtol=1e-1)


@pytest.mark.L0
@pytest.mark.parametrize("c_dtype", [np.float32, ml_dtypes.bfloat16])
def test_gemm_amax_jax_wrapper_fp8(c_dtype):
    skip_unless_sm100()
    m, n, k = 512, 256, 256
    sf_vec_size = 32
    rng = np.random.default_rng(0)

    a_np, a_ref = make_ab_fp8(m, k, ml_dtypes.float8_e5m2, rng)
    b_np, b_ref = make_ab_fp8(n, k, ml_dtypes.float8_e5m2, rng)
    sfa_np, sfa_expanded = make_sf_physical(m, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)
    sfb_np, sfb_expanded = make_sf_physical(n, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)

    run_wrapper_and_check(a_np, a_ref, b_np, b_ref, sfa_np, sfa_expanded, sfb_np, sfb_expanded, c_dtype, sf_vec_size, m, n)


@pytest.mark.L0
@pytest.mark.xfail(
    raises=TypeError,
    strict=False,
    reason="The uint8 packed-fp4 container path fails at kernel construction for torch and JAX alike "
    "(TensorDescs are built before _interpret_uint8_as_fp4x2 is set, so the kernel sees Uint8); "
    "torch.uint8 is likewise disabled in test_gemm_amax_utils.py. Remove this marker when the container path is fixed.",
)
@pytest.mark.parametrize("sf_vec_size", [16, 32])
def test_gemm_amax_jax_wrapper_fp4_uint8(sf_vec_size):
    skip_unless_sm100()
    m, n, k = 512, 256, 256
    rng = np.random.default_rng(0)

    a_np, a_ref = make_ab_fp4_uint8(m, k, rng)
    b_np, b_ref = make_ab_fp4_uint8(n, k, rng)
    sf_dtype = ml_dtypes.float8_e8m0fnu if sf_vec_size == 32 else ml_dtypes.float8_e4m3fn
    sfa_np, sfa_expanded = make_sf_physical(m, k, sf_vec_size, sf_dtype, rng)
    sfb_np, sfb_expanded = make_sf_physical(n, k, sf_vec_size, sf_dtype, rng)

    run_wrapper_and_check(a_np, a_ref, b_np, b_ref, sfa_np, sfa_expanded, sfb_np, sfb_expanded, np.float32, sf_vec_size, m, n)


@pytest.mark.L0
def test_gemm_amax_jax_compile_execute_fp8():
    """Class API with caller-allocated JAX output buffers and an explicit stream."""
    skip_unless_sm100()
    from cuda.bindings import driver as cuda
    from cudnn import GemmAmaxSm100

    m, n, k = 512, 256, 256
    sf_vec_size = 32
    rng = np.random.default_rng(1)

    a_np, a_ref = make_ab_fp8(m, k, ml_dtypes.float8_e5m2, rng)
    b_np, b_ref = make_ab_fp8(n, k, ml_dtypes.float8_e5m2, rng)
    sfa_np, sfa_expanded = make_sf_physical(m, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)
    sfb_np, sfb_expanded = make_sf_physical(n, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)

    a_dev = jax.device_put(a_np)
    b_dev = jax.device_put(b_np)
    sfa_dev = jax.device_put(sfa_np)
    sfb_dev = jax.device_put(sfb_np)
    c_dev = jnp.zeros((m, n, 1), dtype=jnp.float32)
    amax_dev = jnp.full((1, 1, 1), -float("inf"), dtype=jnp.float32)
    jax.block_until_ready((a_dev, b_dev, sfa_dev, sfb_dev, c_dev, amax_dev))

    gemm = GemmAmaxSm100(
        sample_a=a_dev,
        sample_b=b_dev,
        sample_sfa=sfa_dev,
        sample_sfb=sfb_dev,
        sample_c=c_dev,
        sample_amax=amax_dev,
        sf_vec_size=sf_vec_size,
    )
    assert gemm.check_support()
    gemm.compile()
    gemm.execute(
        a_tensor=a_dev,
        b_tensor=b_dev,
        sfa_tensor=sfa_dev,
        sfb_tensor=sfb_dev,
        c_tensor=c_dev,
        amax_tensor=amax_dev,
        current_stream=cuda.CUstream(0),
    )
    device_sync()

    c_ref, amax_ref = reference_gemm_amax(a_ref, b_ref, sfa_expanded, sfb_expanded)
    np.testing.assert_allclose(np.asarray(c_dev)[:, :, 0], c_ref, atol=0.01, rtol=0.01)
    np.testing.assert_allclose(float(np.asarray(amax_dev).reshape(())), amax_ref, atol=1e-1, rtol=1e-1)


@pytest.mark.L0
def test_gemm_amax_jax_wrapper_errors():
    skip_unless_sm100()
    from cudnn import gemm_amax_wrapper_sm100

    m, n, k = 512, 256, 256
    sf_vec_size = 32
    rng = np.random.default_rng(2)

    a_np, _ = make_ab_fp8(m, k, ml_dtypes.float8_e5m2, rng)
    b_np, _ = make_ab_fp8(n, k, ml_dtypes.float8_e5m2, rng)
    sfa_np, _ = make_sf_physical(m, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)
    sfb_np, _ = make_sf_physical(n, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)

    a_dev = jax.device_put(a_np)
    b_dev = jax.device_put(b_np)
    sfa_dev = jax.device_put(sfa_np)
    sfb_dev = jax.device_put(sfb_np)

    with pytest.raises(ValueError, match="row-major"):
        gemm_amax_wrapper_sm100(a_dev, b_dev, sfa_dev, sfb_dev, c_major="m")

    a_batched = jax.device_put(np.repeat(a_np, 2, axis=2))
    b_batched = jax.device_put(np.repeat(b_np, 2, axis=2))
    with pytest.raises(ValueError, match="batch dim L == 1"):
        gemm_amax_wrapper_sm100(a_batched, b_batched, sfa_dev, sfb_dev)

    with pytest.raises(ValueError, match="Unsupported tensor framework"):
        gemm_amax_wrapper_sm100(a_np, b_np, sfa_np, sfb_np)


@pytest.mark.L0
def test_gemm_amax_jax_jit_sm100():
    """XLA custom-call entry point (cudnn.jax.call): eager, jitted, cached, and composed."""
    import cutlass.jax

    if not cutlass.jax.is_available():
        pytest.skip("CuTeDSL JAX extensions unavailable (jax >= 0.5 required)")
    skip_unless_sm100()
    from cudnn import gemm_amax_jax_sm100

    # cudnn.jax is reachable from a bare `import cudnn` (lazy submodule export)
    import cudnn

    assert cudnn.jax.TensorSpec is cutlass.jax.TensorSpec

    m, n, k = 512, 256, 256
    sf_vec_size = 32
    rng = np.random.default_rng(3)

    a_np, a_ref = make_ab_fp8(m, k, ml_dtypes.float8_e5m2, rng)
    b_np, b_ref = make_ab_fp8(n, k, ml_dtypes.float8_e5m2, rng)
    sfa_np, sfa_expanded = make_sf_physical(m, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)
    sfb_np, sfb_expanded = make_sf_physical(n, k, sf_vec_size, ml_dtypes.float8_e8m0fnu, rng)
    c_ref, amax_ref = reference_gemm_amax(a_ref, b_ref, sfa_expanded, sfb_expanded)

    a_dev, b_dev, sfa_dev, sfb_dev = (jax.device_put(x) for x in (a_np, b_np, sfa_np, sfb_np))

    def check(c_dev, amax_dev):
        # XLA orders the custom call on its own stream; block_until_ready is sufficient
        jax.block_until_ready((c_dev, amax_dev))
        np.testing.assert_allclose(np.asarray(c_dev)[:, :, 0], c_ref, atol=0.01, rtol=0.01)
        np.testing.assert_allclose(float(np.asarray(amax_dev).reshape(())), amax_ref, atol=1e-1, rtol=1e-1)

    # Eager
    check(*gemm_amax_jax_sm100(a_dev, b_dev, sfa_dev, sfb_dev, sf_vec_size=sf_vec_size))

    # Under jax.jit, twice (donation safety + compiled-kernel/registration cache)
    jitted = jax.jit(lambda a, b, sfa, sfb: gemm_amax_jax_sm100(a, b, sfa, sfb, sf_vec_size=sf_vec_size))
    check(*jitted(a_dev, b_dev, sfa_dev, sfb_dev))
    check(*jitted(a_dev, b_dev, sfa_dev, sfb_dev))

    # Composed with surrounding jnp ops in one jitted graph
    @jax.jit
    def composed(a, b, sfa, sfb):
        c, amax = gemm_amax_jax_sm100(a, b, sfa, sfb, sf_vec_size=sf_vec_size)
        return jnp.sum(c) / amax[0, 0, 0]

    ratio = composed(a_dev, b_dev, sfa_dev, sfb_dev)
    jax.block_until_ready(ratio)
    np.testing.assert_allclose(float(ratio), c_ref.sum() / amax_ref, rtol=0.02)

    # Batch dim restriction
    with pytest.raises(ValueError, match="batch dim L == 1"):
        gemm_amax_jax_sm100(
            jax.device_put(np.repeat(a_np, 2, axis=2)),
            jax.device_put(np.repeat(b_np, 2, axis=2)),
            sfa_dev,
            sfb_dev,
            sf_vec_size=sf_vec_size,
        )
