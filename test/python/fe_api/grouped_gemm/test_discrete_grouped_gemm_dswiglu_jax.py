# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX coverage for the discrete-weight grouped GEMM dSwiGLU (backward) wrapper.

JAX contract mirrors the SwiGLU forward: FP8 inputs, per-expert B/SFB as packed-uint8
pointer arrays, scale-factor tensors (SFA input and the sfd_row/sfd_col outputs) in the
physical C-contiguous atom shape (1, MN', K', 32, 4, 4) (the kernel rebuilds every SF
layout from the A/D shapes and consumes only the base pointers), plus the
backward-specific C (forward activations), beta, prob, and zero-initialized dprob
inputs. d_row/d_col are checked bit-identical against the torch wrapper on identical
input bytes; dprob accumulates through floating-point atomics whose ordering is not
deterministic, so it is compared with a tight tolerance instead.

Rejected for JAX: packed-fp4 inputs (JAX has no packed fp4 dtype, and uint8 container
arrays are rejected at the kernel entry).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import ceil_div, device_sync, skip_unless_sm100

M, N, K, EXPERTS = 1024, 512, 512, 4
N_OUT = 2 * N
SF_VEC_SIZE = 32


def make_problem():
    rng = np.random.default_rng(20260811)
    a_np = rng.integers(-2, 3, (M, K, 1)).astype(np.float32).astype(ml_dtypes.float8_e4m3fn)  # gradient input, k-major
    b_np = [rng.integers(-2, 3, (N, K)).astype(np.float32).astype(ml_dtypes.float8_e4m3fn) for _ in range(EXPERTS)]  # k-major
    c_np = (rng.standard_normal((M, N_OUT, 1), dtype=np.float32) * 0.5).astype(ml_dtypes.bfloat16)  # forward activations
    rest_k = ceil_div(ceil_div(K, SF_VEC_SIZE), 4)
    sfa_u8 = rng.integers(126, 130, (1, ceil_div(M, 128), rest_k, 32, 4, 4), dtype=np.uint8)
    sfb_u8 = [rng.integers(126, 130, (1, ceil_div(N, 128), rest_k, 32, 4, 4), dtype=np.uint8) for _ in range(EXPERTS)]
    offsets_np = np.arange(M // EXPERTS, M + 1, M // EXPERTS, dtype=np.int32)
    alpha_np = rng.uniform(-1.5, 1.5, (EXPERTS,)).astype(np.float32)
    beta_np = rng.uniform(-1.5, 1.5, (EXPERTS,)).astype(np.float32)
    prob_np = rng.uniform(-1.0, 1.0, (M, 1, 1)).astype(np.float32)
    norm_const_np = np.array([0.01], dtype=np.float32)
    return a_np, b_np, c_np, sfa_u8, sfb_u8, offsets_np, alpha_np, beta_np, prob_np, norm_const_np


MMA_PERMUTE_ORDER = (3, 4, 1, 5, 2, 0)


def run_torch(a_np, b_np, c_np, sfa_u8, sfb_u8, offsets_np, alpha_np, beta_np, prob_np, norm_const_np):
    """Reference run through the established torch contract on identical bytes."""
    from cudnn import discrete_grouped_gemm_dswiglu_wrapper_sm100

    a_t = torch.from_numpy(a_np.view(np.uint8)).view(torch.float8_e4m3fn).reshape(a_np.shape).cuda()
    b_t = [torch.from_numpy(b.view(np.uint8)).view(torch.float8_e4m3fn).reshape(b.shape).cuda() for b in b_np]
    c_t = torch.from_numpy(c_np.view(np.uint8)).view(torch.bfloat16).reshape(c_np.shape).cuda()
    sfa_t = torch.from_numpy(sfa_u8).cuda().view(torch.float8_e8m0fnu).permute(MMA_PERMUTE_ORDER)  # torch atom view
    sfb_t = [torch.from_numpy(sfb).cuda().view(torch.float8_e8m0fnu) for sfb in sfb_u8]
    dprob_t = torch.zeros((M, 1, 1), dtype=torch.float32, device="cuda")
    result = discrete_grouped_gemm_dswiglu_wrapper_sm100(
        a_tensor=a_t,
        b_ptrs=torch.tensor([b.data_ptr() for b in b_t], dtype=torch.int64, device="cuda"),
        c_tensor=c_t,
        sfa_tensor=sfa_t,
        sfb_ptrs=torch.tensor([sfb.data_ptr() for sfb in sfb_t], dtype=torch.int64, device="cuda"),
        padded_offsets=torch.from_numpy(offsets_np).cuda(),
        alpha_tensor=torch.from_numpy(alpha_np).cuda(),
        beta_tensor=torch.from_numpy(beta_np).cuda(),
        prob_tensor=torch.from_numpy(prob_np).cuda(),
        dprob_tensor=dprob_t,
        norm_const_tensor=torch.from_numpy(norm_const_np).cuda(),
        n=N,
        b_dtype=torch.float8_e4m3fn,
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=SF_VEC_SIZE,
        act_func="dswiglu",
    )
    torch.cuda.synchronize()
    return result, (a_t, b_t, c_t, sfa_t, sfb_t)


def packed_ptrs(arrays):
    values = np.array([array.unsafe_buffer_pointer() for array in arrays], dtype=np.int64)
    return jax.block_until_ready(jnp.asarray(values.view(np.uint8)))


def as_bytes(array_or_tensor):
    """Raw little-endian memory bytes as a uint8 numpy array (exact comparison incl. fp8)."""
    if isinstance(array_or_tensor, torch.Tensor):
        # The wrapper outputs are (m, n, 1) with an extent-1 batch dim of arbitrary
        # stride; squeeze it so the row-major bytes match the JAX C-contiguous bytes.
        data = array_or_tensor.squeeze(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes()
    else:
        data = np.asarray(array_or_tensor).tobytes()
    return np.frombuffer(data, dtype=np.uint8)


@pytest.mark.L0
def test_discrete_grouped_gemm_dswiglu_jax_fp8_matches_torch():
    skip_unless_sm100()
    from cudnn import discrete_grouped_gemm_dswiglu_wrapper_sm100

    a_np, b_np, c_np, sfa_u8, sfb_u8, offsets_np, alpha_np, beta_np, prob_np, norm_const_np = make_problem()
    result_t, _torch_keepalive = run_torch(a_np, b_np, c_np, sfa_u8, sfb_u8, offsets_np, alpha_np, beta_np, prob_np, norm_const_np)

    a_j = jnp.asarray(a_np)
    b_j = [jnp.asarray(b) for b in b_np]
    c_j = jnp.asarray(c_np)
    sfa_j = jnp.asarray(sfa_u8.view(ml_dtypes.float8_e8m0fnu))  # physical atom shape
    sfb_j = [jnp.asarray(sfb.view(ml_dtypes.float8_e8m0fnu)) for sfb in sfb_u8]
    offsets_j, alpha_j, beta_j, prob_j, norm_const_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, beta_np, prob_np, norm_const_np))
    dprob_j = jnp.zeros((M, 1, 1), dtype=jnp.float32)  # output accumulator, must be zero-initialized
    jax.block_until_ready((a_j, c_j, sfa_j, offsets_j, alpha_j, beta_j, prob_j, norm_const_j, dprob_j, *b_j, *sfb_j))

    # The per-expert weight/scale arrays must stay alive while the kernel runs.
    result_j = discrete_grouped_gemm_dswiglu_wrapper_sm100(
        a_tensor=a_j,
        b_ptrs=packed_ptrs(b_j),
        c_tensor=c_j,
        sfa_tensor=sfa_j,
        sfb_ptrs=packed_ptrs(sfb_j),
        padded_offsets=offsets_j,
        alpha_tensor=alpha_j,
        beta_tensor=beta_j,
        prob_tensor=prob_j,
        dprob_tensor=dprob_j,
        norm_const_tensor=norm_const_j,
        n=N,
        b_dtype="float8_e4m3fn",
        d_dtype="float8_e4m3fn",
        sf_vec_size=SF_VEC_SIZE,
        act_func="dswiglu",
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream

    # d_row/d_col are deterministic per-tile kernel outputs; compare raw bytes.
    for key in ("d_row_tensor", "d_col_tensor"):
        np.testing.assert_array_equal(
            as_bytes(result_j[key]),
            as_bytes(result_t[key]),
            err_msg=f"dswiglu {key}: JAX output differs from torch output on identical input bytes",
        )
    # dprob accumulates through floating-point atomics; ordering is not deterministic,
    # so compare with a tight tolerance instead of bit equality.
    np.testing.assert_allclose(
        np.asarray(result_j["dprob_tensor"]),
        result_t["dprob_tensor"].float().cpu().numpy(),
        rtol=2e-5,
        atol=1e-4,
        err_msg="dswiglu dprob_tensor: JAX output differs from torch output beyond atomic-ordering tolerance",
    )


@pytest.mark.L0
def test_discrete_grouped_gemm_dswiglu_jax_jit_matches_eager():
    """XLA custom-call entry point (cudnn.jax.call): bit-identical to the eager JAX wrapper.

    d_row/d_col are deterministic per-tile kernel outputs and are compared bitwise;
    dprob accumulates through floating-point atomics whose ordering is not
    deterministic across runs, so it is compared with the same tight tolerance the
    torch-parity test uses.
    """
    import cutlass.jax

    if not cutlass.jax.is_available():
        pytest.skip("CuTeDSL JAX extensions unavailable (jax >= 0.5 required)")
    skip_unless_sm100()
    from cudnn import discrete_grouped_gemm_dswiglu_jax_sm100, discrete_grouped_gemm_dswiglu_wrapper_sm100

    a_np, b_np, c_np, sfa_u8, sfb_u8, offsets_np, alpha_np, beta_np, prob_np, norm_const_np = make_problem()

    a_j = jnp.asarray(a_np)
    b_j = [jnp.asarray(b) for b in b_np]
    c_j = jnp.asarray(c_np)
    sfa_j = jnp.asarray(sfa_u8.view(ml_dtypes.float8_e8m0fnu))  # physical atom shape
    sfb_j = [jnp.asarray(sfb.view(ml_dtypes.float8_e8m0fnu)) for sfb in sfb_u8]
    offsets_j, alpha_j, beta_j, prob_j, norm_const_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, beta_np, prob_np, norm_const_np))
    dprob_j = jnp.zeros((M, 1, 1), dtype=jnp.float32)  # eager wrapper takes dprob as a zeroed input buffer
    jax.block_until_ready((a_j, c_j, sfa_j, offsets_j, alpha_j, beta_j, prob_j, norm_const_j, dprob_j, *b_j, *sfb_j))
    b_ptrs_j = packed_ptrs(b_j)
    sfb_ptrs_j = packed_ptrs(sfb_j)

    # Eager wrapper baseline on the same bytes; the last padded offset covers every
    # row and d_row/d_col are fully overwritten per-row kernel outputs, so
    # full-buffer bitwise comparison against the jit path is well-defined.
    result_eager = discrete_grouped_gemm_dswiglu_wrapper_sm100(
        a_tensor=a_j,
        b_ptrs=b_ptrs_j,
        c_tensor=c_j,
        sfa_tensor=sfa_j,
        sfb_ptrs=sfb_ptrs_j,
        padded_offsets=offsets_j,
        alpha_tensor=alpha_j,
        beta_tensor=beta_j,
        prob_tensor=prob_j,
        dprob_tensor=dprob_j,
        norm_const_tensor=norm_const_j,
        n=N,
        b_dtype="float8_e4m3fn",
        d_dtype="float8_e4m3fn",
        sf_vec_size=SF_VEC_SIZE,
        act_func="dswiglu",
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream
    expected = {key: np.asarray(result_eager[key]).view(np.uint8) for key in ("d_row_tensor", "d_col_tensor")}
    dprob_expected = np.asarray(result_eager["dprob_tensor"])

    def run_jit_entry(a, c, sfa, offsets, alpha, beta, prob, norm_const, b_ptrs, sfb_ptrs):
        # dprob is a donated zero-initialized output of the custom call (the jit
        # entry has no caller-provided dprob buffer, unlike the eager wrapper).
        return discrete_grouped_gemm_dswiglu_jax_sm100(
            a_tensor=a,
            b_ptrs=b_ptrs,
            c_tensor=c,
            sfa_tensor=sfa,
            sfb_ptrs=sfb_ptrs,
            padded_offsets=offsets,
            alpha_tensor=alpha,
            beta_tensor=beta,
            prob_tensor=prob,
            norm_const_tensor=norm_const,
            n=N,
            d_dtype="float8_e4m3fn",
            sf_vec_size=SF_VEC_SIZE,
            act_func="dswiglu",
        )

    def check(result):
        jax.block_until_ready(tuple(value for value in result.values() if value is not None))
        for key in ("d_row_tensor", "d_col_tensor"):
            np.testing.assert_array_equal(
                np.asarray(result[key]).view(np.uint8),
                expected[key],
                err_msg=f"dswiglu {key}: jit output differs from eager wrapper output on identical input bytes",
            )
        np.testing.assert_allclose(
            np.asarray(result["dprob_tensor"]),
            dprob_expected,
            rtol=2e-5,
            atol=1e-4,
            err_msg="dswiglu dprob_tensor: jit output differs from eager wrapper output beyond atomic-ordering tolerance",
        )
        assert result["amax_tensor"] is None  # fp8 d_dtype: no amax, matching the eager wrapper
        assert result["dbias_tensor"] is None  # generate_dbias not requested

    # Eager custom call
    check(run_jit_entry(a_j, c_j, sfa_j, offsets_j, alpha_j, beta_j, prob_j, norm_const_j, b_ptrs_j, sfb_ptrs_j))

    # Under jax.jit, twice (compiled-kernel / registration cache). n stays static.
    jitted = jax.jit(run_jit_entry)
    check(jitted(a_j, c_j, sfa_j, offsets_j, alpha_j, beta_j, prob_j, norm_const_j, b_ptrs_j, sfb_ptrs_j))
    check(jitted(a_j, c_j, sfa_j, offsets_j, alpha_j, beta_j, prob_j, norm_const_j, b_ptrs_j, sfb_ptrs_j))


@pytest.mark.L0
def test_discrete_grouped_gemm_dswiglu_jax_errors():
    skip_unless_sm100()
    from cudnn import discrete_grouped_gemm_dswiglu_wrapper_sm100

    a_np, b_np, c_np, sfa_u8, sfb_u8, offsets_np, alpha_np, beta_np, prob_np, norm_const_np = make_problem()
    b_j = [jnp.asarray(b) for b in b_np]
    sfb_j = [jnp.asarray(sfb.view(ml_dtypes.float8_e8m0fnu)) for sfb in sfb_u8]
    jax.block_until_ready((*b_j, *sfb_j))

    # Packed fp4 has no JAX dtype; uint8 container inputs are rejected.
    a_fp4_j = jnp.asarray(np.zeros((M, K // 2, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        discrete_grouped_gemm_dswiglu_wrapper_sm100(
            a_tensor=a_fp4_j,
            b_ptrs=packed_ptrs(b_j),
            c_tensor=jnp.asarray(c_np),
            sfa_tensor=jnp.asarray(sfa_u8.view(ml_dtypes.float8_e8m0fnu)),
            sfb_ptrs=packed_ptrs(sfb_j),
            padded_offsets=jnp.asarray(offsets_np),
            alpha_tensor=jnp.asarray(alpha_np),
            beta_tensor=jnp.asarray(beta_np),
            prob_tensor=jnp.asarray(prob_np),
            dprob_tensor=jnp.zeros((M, 1, 1), dtype=jnp.float32),
            n=N,
            b_dtype="uint8",
            sf_vec_size=SF_VEC_SIZE,
            act_func="dswiglu",
        )
