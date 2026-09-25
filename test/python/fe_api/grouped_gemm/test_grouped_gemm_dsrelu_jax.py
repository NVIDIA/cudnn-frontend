# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX coverage for the SM100 grouped GEMM dSReLU backward wrapper.

JAX contract: discrete weight mode only (dense mode's expert-outermost strided B
layout is not expressible as row-major JAX arrays) with fp8 inputs (JAX has no
packed fp4 dtype). Pointer arrays are passed as packed uint8 (8 bytes per pointer)
because JAX truncates int64 without x64 mode. Scale-factor tensors are passed in
the physical C-contiguous atom shape (L, MN', K', 32, 4, 4) -- the permuted torch
view of the same bytes is not expressible in JAX, and the kernel rebuilds the SF
layout from the GEMM shapes, consuming only the SF base pointer.

Outputs are checked bit-identical against the torch wrapper run on identical input
bytes, except dprob, which the kernel accumulates with atomic float adds (ordering
is not deterministic), checked with a tight allclose instead.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import device_sync, skip_unless_sm100

MMA_PERMUTE_ORDER = (3, 4, 1, 5, 2, 0)


def ceil_div(a, b):
    return (a + b - 1) // b


def make_problem(m=512, n=256, k=256, experts=2, sf_vec_size=32):
    rng = np.random.default_rng(20260809)
    rk = ceil_div(ceil_div(k, sf_vec_size), 4)
    a_np = (rng.integers(-4, 5, size=(m, k, 1)) * 0.25).astype(ml_dtypes.float8_e4m3fn)
    b_np = (rng.integers(-4, 5, size=(experts, n, k)) * 0.25).astype(ml_dtypes.float8_e4m3fn)  # per-expert (n, k) k-major
    c_np = (rng.standard_normal((m, n, 1), dtype=np.float32) * 0.5).astype(ml_dtypes.bfloat16)
    # SF tensors in physical C-contiguous atom form; e8m0 holds powers of two exactly.
    sfa_np = (2.0 ** rng.integers(-1, 2, size=(1, ceil_div(m, 128), rk, 32, 4, 4))).astype(ml_dtypes.float8_e8m0fnu)
    sfb_np = (2.0 ** rng.integers(-1, 2, size=(experts, 1, ceil_div(n, 128), rk, 32, 4, 4))).astype(ml_dtypes.float8_e8m0fnu)
    offsets_np = np.arange(m // experts, m + 1, m // experts, dtype=np.int32)
    alpha_np = np.array([0.75, 1.25][:experts], dtype=np.float32)
    prob_np = np.linspace(0.25, 1.0, m, dtype=np.float32).reshape(m, 1, 1)
    norm_const_np = np.ones(1, dtype=np.float32)
    return a_np, b_np, c_np, sfa_np, sfb_np, offsets_np, alpha_np, prob_np, norm_const_np


def _torch_from_bytes(arr, torch_dtype, shape):
    return torch.from_numpy(np.ascontiguousarray(arr).view(np.uint8)).view(torch_dtype).reshape(shape).cuda()


def _u8(tensor_or_array):
    """Byte view of an fp8/bf16 tensor or array for exact comparisons."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.view(torch.uint8).cpu().numpy()
    return np.asarray(tensor_or_array).view(np.uint8)


@pytest.mark.L0
def test_grouped_gemm_dsrelu_jax_discrete_fp8_matches_torch():
    skip_unless_sm100()
    from cudnn import grouped_gemm_dsrelu_wrapper_sm100

    m, n, k, experts, sf_vec_size = 512, 256, 256, 2, 32
    a_np, b_np, c_np, sfa_np, sfb_np, offsets_np, alpha_np, prob_np, norm_const_np = make_problem(m, n, k, experts, sf_vec_size)

    # ---- torch run (discrete mode, permuted SF atom views over the same bytes) ----
    a_t = _torch_from_bytes(a_np, torch.float8_e4m3fn, (m, k, 1))
    c_t = _torch_from_bytes(c_np, torch.bfloat16, (m, n, 1))
    sfa_t = _torch_from_bytes(sfa_np, torch.float8_e8m0fnu, sfa_np.shape).permute(MMA_PERMUTE_ORDER)
    b_t = _torch_from_bytes(b_np, torch.float8_e4m3fn, b_np.shape)
    sfb_t = _torch_from_bytes(sfb_np, torch.float8_e8m0fnu, sfb_np.shape)
    b_ptrs_t = torch.tensor([b_t[i].data_ptr() for i in range(experts)], dtype=torch.int64, device="cuda")
    sfb_ptrs_t = torch.tensor([sfb_t[i].data_ptr() for i in range(experts)], dtype=torch.int64, device="cuda")

    result_t = grouped_gemm_dsrelu_wrapper_sm100(
        a_tensor=a_t,
        c_tensor=c_t,
        sfa_tensor=sfa_t,
        padded_offsets=torch.from_numpy(offsets_np).cuda(),
        alpha_tensor=torch.from_numpy(alpha_np).cuda(),
        prob_tensor=torch.from_numpy(prob_np).cuda(),
        b_ptrs=b_ptrs_t,
        sfb_ptrs=sfb_ptrs_t,
        n=n,
        b_dtype=torch.float8_e4m3fn,
        b_major="k",
        norm_const_tensor=torch.from_numpy(norm_const_np).cuda(),
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=sf_vec_size,
    )
    torch.cuda.synchronize()

    # ---- jax run on identical bytes (SF tensors in the physical atom form) ----
    a_j = jnp.asarray(a_np)
    c_j = jnp.asarray(c_np)
    sfa_j = jnp.asarray(sfa_np)
    b_experts_j = [jnp.asarray(b_np[i]) for i in range(experts)]
    sfb_experts_j = [jnp.asarray(sfb_np[i]) for i in range(experts)]
    offsets_j = jnp.asarray(offsets_np)
    alpha_j = jnp.asarray(alpha_np)
    prob_j = jnp.asarray(prob_np)
    norm_const_j = jnp.asarray(norm_const_np)
    jax.block_until_ready((a_j, c_j, sfa_j, offsets_j, alpha_j, prob_j, norm_const_j, *b_experts_j, *sfb_experts_j))

    # Packed uint8 pointer arrays (8 little-endian bytes per pointer): JAX truncates
    # int64 without x64 mode. The weight/SF arrays must stay alive while the kernel runs.
    b_ptr_values = np.array([w.unsafe_buffer_pointer() for w in b_experts_j], dtype=np.int64)
    sfb_ptr_values = np.array([w.unsafe_buffer_pointer() for w in sfb_experts_j], dtype=np.int64)
    b_ptrs_j = jax.block_until_ready(jnp.asarray(b_ptr_values.view(np.uint8)))
    sfb_ptrs_j = jax.block_until_ready(jnp.asarray(sfb_ptr_values.view(np.uint8)))

    result_j = grouped_gemm_dsrelu_wrapper_sm100(
        a_tensor=a_j,
        c_tensor=c_j,
        sfa_tensor=sfa_j,
        padded_offsets=offsets_j,
        alpha_tensor=alpha_j,
        prob_tensor=prob_j,
        b_ptrs=b_ptrs_j,
        sfb_ptrs=sfb_ptrs_j,
        n=n,
        b_dtype="float8_e4m3fn",
        b_major="k",
        norm_const_tensor=norm_const_j,
        d_dtype="float8_e4m3fn",
        sf_vec_size=sf_vec_size,
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream

    # Sanity: the kernels actually ran and produced non-trivial values.
    assert np.count_nonzero(result_t["dprob_tensor"].cpu().numpy()) > 0
    assert np.count_nonzero(_u8(result_t["d_row_tensor"])) > 0

    # Elementwise-quantized outputs must be bit-identical on identical input bytes.
    for key in ("d_row_tensor", "d_col_tensor", "d_srelu_tensor"):
        np.testing.assert_array_equal(
            _u8(result_j[key]),
            _u8(result_t[key]),
            err_msg=f"dsrelu discrete {key}: JAX output differs from torch output on identical input bytes",
        )

    # SF outputs: JAX holds the physical atom form; the torch view of the same logical
    # tensor is its (3, 4, 1, 5, 2, 0) permutation.
    for key in ("sfd_row_tensor", "sfd_col_tensor", "sfd_col_d_srelu_tensor"):
        np.testing.assert_array_equal(
            np.transpose(_u8(result_j[key]), MMA_PERMUTE_ORDER),
            _u8(result_t[key]),
            err_msg=f"dsrelu discrete {key}: JAX output differs from torch output on identical input bytes",
        )

    # dprob is accumulated with atomic float adds; ordering is nondeterministic, so
    # compare with a tight tolerance rather than bit-identically.
    np.testing.assert_allclose(
        np.asarray(result_j["dprob_tensor"]),
        result_t["dprob_tensor"].cpu().numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="dsrelu discrete dprob_tensor: JAX output differs from torch output beyond atomic-add tolerance",
    )

    assert result_j["amax_tensor"] is None and result_t["amax_tensor"] is None
    assert result_j["dbias_tensor"] is None and result_t["dbias_tensor"] is None


@pytest.mark.L0
def test_grouped_gemm_dsrelu_jax_jit_matches_eager():
    """XLA custom-call entry point (cudnn.jax.call): bit-identical to the eager JAX wrapper."""
    import cutlass.jax

    if not cutlass.jax.is_available():
        pytest.skip("CuTeDSL JAX extensions unavailable (jax >= 0.5 required)")
    skip_unless_sm100()
    from cudnn import grouped_gemm_dsrelu_jax_sm100, grouped_gemm_dsrelu_wrapper_sm100

    m, n, k, experts, sf_vec_size = 512, 256, 256, 2, 32
    a_np, b_np, c_np, sfa_np, sfb_np, offsets_np, alpha_np, prob_np, norm_const_np = make_problem(m, n, k, experts, sf_vec_size)

    a_j = jnp.asarray(a_np)
    c_j = jnp.asarray(c_np)
    sfa_j = jnp.asarray(sfa_np)
    b_experts_j = [jnp.asarray(b_np[i]) for i in range(experts)]  # per-expert (n, k) k-major
    sfb_experts_j = [jnp.asarray(sfb_np[i]) for i in range(experts)]
    offsets_j, alpha_j, prob_j, norm_const_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, prob_np, norm_const_np))
    jax.block_until_ready((a_j, c_j, sfa_j, offsets_j, alpha_j, prob_j, norm_const_j, *b_experts_j, *sfb_experts_j))
    b_ptr_values = np.array([w.unsafe_buffer_pointer() for w in b_experts_j], dtype=np.int64)
    sfb_ptr_values = np.array([w.unsafe_buffer_pointer() for w in sfb_experts_j], dtype=np.int64)
    b_ptrs_j = jax.block_until_ready(jnp.asarray(b_ptr_values.view(np.uint8)))
    sfb_ptrs_j = jax.block_until_ready(jnp.asarray(sfb_ptr_values.view(np.uint8)))

    # Eager wrapper baseline on the same bytes; the last padded offset covers every
    # row, so full-buffer bitwise comparison against the jit path is well-defined.
    result_eager = grouped_gemm_dsrelu_wrapper_sm100(
        a_tensor=a_j,
        c_tensor=c_j,
        sfa_tensor=sfa_j,
        padded_offsets=offsets_j,
        alpha_tensor=alpha_j,
        prob_tensor=prob_j,
        b_ptrs=b_ptrs_j,
        sfb_ptrs=sfb_ptrs_j,
        n=n,
        b_dtype="float8_e4m3fn",
        b_major="k",
        norm_const_tensor=norm_const_j,
        d_dtype="float8_e4m3fn",
        sf_vec_size=sf_vec_size,
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream

    exact_keys = ("d_row_tensor", "d_col_tensor", "d_srelu_tensor", "sfd_row_tensor", "sfd_col_tensor", "sfd_col_d_srelu_tensor")
    assert np.count_nonzero(_u8(result_eager["d_row_tensor"])) > 0
    expected = {key: _u8(result_eager[key]) for key in exact_keys}
    expected_dprob = np.asarray(result_eager["dprob_tensor"])

    def check(results):
        # jit entry returns the eager wrapper's key order as a tuple.
        d_row, d_col, d_srelu, dprob, dbias, amax, sfd_row, sfd_col, sfd_col_d_srelu = results
        jax.block_until_ready((d_row, d_col, d_srelu, dprob, sfd_row, sfd_col, sfd_col_d_srelu))
        assert dbias is None and amax is None
        got = {
            "d_row_tensor": d_row,
            "d_col_tensor": d_col,
            "d_srelu_tensor": d_srelu,
            "sfd_row_tensor": sfd_row,
            "sfd_col_tensor": sfd_col,
            "sfd_col_d_srelu_tensor": sfd_col_d_srelu,
        }
        for key in exact_keys:
            np.testing.assert_array_equal(
                _u8(got[key]),
                expected[key],
                err_msg=f"dsrelu jit {key}: output differs from eager wrapper output on identical input bytes",
            )
        # dprob is accumulated with atomic float adds; ordering is nondeterministic.
        np.testing.assert_allclose(
            np.asarray(dprob),
            expected_dprob,
            rtol=1e-4,
            atol=1e-4,
            err_msg="dsrelu jit dprob: output differs from eager wrapper output beyond atomic-add tolerance",
        )

    def run(a, c, sfa, offsets, alpha, prob, b_ptrs, sfb_ptrs, norm_const):
        return grouped_gemm_dsrelu_jax_sm100(
            a_tensor=a,
            c_tensor=c,
            sfa_tensor=sfa,
            padded_offsets=offsets,
            alpha_tensor=alpha,
            prob_tensor=prob,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            n=n,
            norm_const_tensor=norm_const,
            b_dtype="float8_e4m3fn",
            b_major="k",
            d_dtype="float8_e4m3fn",
            sf_vec_size=sf_vec_size,
        )

    args = (a_j, c_j, sfa_j, offsets_j, alpha_j, prob_j, b_ptrs_j, sfb_ptrs_j, norm_const_j)

    # Eager custom call
    check(run(*args))

    # Under jax.jit, twice (compiled-kernel / registration cache). n stays static.
    jitted = jax.jit(run)
    check(jitted(*args))
    check(jitted(*args))


@pytest.mark.L0
def test_grouped_gemm_dsrelu_jax_errors():
    skip_unless_sm100()
    from cudnn import grouped_gemm_dsrelu_wrapper_sm100

    m, n, k, experts, sf_vec_size = 512, 256, 256, 2, 32
    a_np, b_np, c_np, sfa_np, sfb_np, offsets_np, alpha_np, prob_np, norm_const_np = make_problem(m, n, k, experts, sf_vec_size)
    a_j = jnp.asarray(a_np)
    c_j = jnp.asarray(c_np)
    sfa_j = jnp.asarray(sfa_np)
    offsets_j, alpha_j, prob_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, prob_np))

    # Dense weight mode: the expert-outermost strided (n, k, l) B layout is inexpressible.
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        grouped_gemm_dsrelu_wrapper_sm100(
            a_tensor=a_j,
            c_tensor=c_j,
            sfa_tensor=sfa_j,
            padded_offsets=offsets_j,
            alpha_tensor=alpha_j,
            prob_tensor=prob_j,
            b_tensor=jnp.asarray(np.zeros((n, k, experts), dtype=ml_dtypes.float8_e4m3fn)),
            sfb_tensor=jnp.asarray(sfb_np),
            sf_vec_size=sf_vec_size,
        )

    # Packed fp4 (raw uint8) inputs: JAX has no packed fp4 dtype.
    a_u8_j = jnp.asarray(np.zeros((m, k // 2, 1), dtype=np.uint8))
    b_ptrs_j = jnp.asarray(np.zeros(8 * experts, dtype=np.uint8))
    sfb_ptrs_j = jnp.asarray(np.zeros(8 * experts, dtype=np.uint8))
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        grouped_gemm_dsrelu_wrapper_sm100(
            a_tensor=a_u8_j,
            c_tensor=c_j,
            sfa_tensor=sfa_j,
            padded_offsets=offsets_j,
            alpha_tensor=alpha_j,
            prob_tensor=prob_j,
            b_ptrs=b_ptrs_j,
            sfb_ptrs=sfb_ptrs_j,
            n=n,
            b_dtype="uint8",
            sf_vec_size=16,
        )

    # Plain numpy arrays are neither torch nor JAX device tensors.
    with pytest.raises(ValueError, match="Unsupported tensor framework"):
        grouped_gemm_dsrelu_wrapper_sm100(
            a_tensor=np.asarray(a_np),
            c_tensor=np.asarray(c_np),
            sfa_tensor=np.asarray(sfa_np),
            padded_offsets=offsets_np,
            alpha_tensor=alpha_np,
            prob_tensor=prob_np,
            b_ptrs=np.zeros(8 * experts, dtype=np.uint8),
            sfb_ptrs=np.zeros(8 * experts, dtype=np.uint8),
            n=n,
            b_dtype="float8_e4m3fn",
            sf_vec_size=sf_vec_size,
        )
