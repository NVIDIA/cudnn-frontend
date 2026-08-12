# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX coverage for the SM100 grouped GEMM dGLU backward wrapper.

JAX contract: BF16 backend, discrete weight mode only (dense mode's expert-outermost
strided B is not expressible as row-major JAX arrays, and the block-scaled backend's
MMA-interleaved scale-factor layouts are likewise inexpressible), with b_ptrs built
from per-expert weight pointers as a packed uint8 array. dprob is a caller-provided
zero-initialized (m, 1, 1) f32 buffer that the kernel writes; dbias is wrapper-allocated.
Outputs are checked bit-identical against the torch wrapper run on identical input bytes.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import device_sync, skip_unless_sm100


def make_problem(m=512, n_weight=128, k=128, experts=2):
    rng = np.random.default_rng(20260809)
    two_n = 2 * n_weight
    a_np = (rng.standard_normal((m, k, 1), dtype=np.float32) * 0.125).astype(ml_dtypes.bfloat16)
    c_np = (rng.standard_normal((m, two_n, 1), dtype=np.float32) * 0.125).astype(ml_dtypes.bfloat16)
    b_storage_np = (rng.standard_normal((experts, n_weight, k), dtype=np.float32) * 0.125).astype(ml_dtypes.bfloat16)
    group_m = m // experts
    offsets_np = np.arange(group_m, m + 1, group_m, dtype=np.int32)
    alpha_np = np.array([0.75, -1.25][:experts], dtype=np.float32)
    beta_np = np.array([1.5, 0.5][:experts], dtype=np.float32)
    prob_np = np.linspace(0.25, 0.875, m, dtype=np.float32).reshape(m, 1, 1)
    return a_np, c_np, b_storage_np, offsets_np, alpha_np, beta_np, prob_np


def _packed_jax_ptrs(weights):
    ptr_values = np.array([w.unsafe_buffer_pointer() for w in weights], dtype=np.int64)
    return jax.block_until_ready(jnp.asarray(ptr_values.view(np.uint8)))


@pytest.mark.L0
def test_grouped_gemm_dglu_jax_discrete_matches_torch():
    skip_unless_sm100()
    from cudnn import grouped_gemm_dglu_wrapper_sm100

    m, n_weight, k, experts = 512, 128, 128, 2
    two_n = 2 * n_weight
    a_np, c_np, b_storage_np, offsets_np, alpha_np, beta_np, prob_np = make_problem(m, n_weight, k, experts)

    # ---- torch run (discrete BF16 mode) ----
    a_t = torch.from_numpy(a_np.view(np.uint8)).view(torch.bfloat16).reshape(m, k, 1).cuda()
    c_t = torch.from_numpy(c_np.view(np.uint8)).view(torch.bfloat16).reshape(m, two_n, 1).cuda()
    b_storage_t = torch.from_numpy(b_storage_np.view(np.uint8)).view(torch.bfloat16).reshape(experts, n_weight, k).cuda()
    b_ptrs_t = torch.tensor([b_storage_t[i].data_ptr() for i in range(experts)], dtype=torch.int64, device="cuda")
    dprob_t = torch.zeros((m, 1, 1), dtype=torch.float32, device="cuda")
    result_t = grouped_gemm_dglu_wrapper_sm100(
        a_tensor=a_t,
        c_tensor=c_t,
        sfa_tensor=None,
        padded_offsets=torch.from_numpy(offsets_np).cuda(),
        alpha_tensor=torch.from_numpy(alpha_np).cuda(),
        beta_tensor=torch.from_numpy(beta_np).cuda(),
        prob_tensor=torch.from_numpy(prob_np).cuda(),
        dprob_tensor=dprob_t,
        b_ptrs=b_ptrs_t,
        n=n_weight,
        b_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        generate_dbias=True,
    )
    torch.cuda.synchronize()

    # ---- jax run on identical bytes ----
    a_j = jnp.asarray(a_np)
    c_j = jnp.asarray(c_np)
    b_experts_j = [jnp.asarray(b_storage_np[i]) for i in range(experts)]  # per-expert (n_weight, k) k-major
    offsets_j, alpha_j, beta_j, prob_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, beta_np, prob_np))
    # Kernel-written output buffer: zero-initialized, materialized before its pointer is used.
    dprob_j = jax.block_until_ready(jnp.zeros((m, 1, 1), dtype=jnp.float32))
    jax.block_until_ready((a_j, c_j, offsets_j, alpha_j, beta_j, prob_j, *b_experts_j))

    # Packed uint8 pointer array (8 little-endian bytes per pointer): JAX truncates
    # int64 without x64 mode. The weight arrays must stay alive while the kernel runs.
    b_ptrs_j = _packed_jax_ptrs(b_experts_j)

    result_j = grouped_gemm_dglu_wrapper_sm100(
        a_tensor=a_j,
        c_tensor=c_j,
        sfa_tensor=None,
        padded_offsets=offsets_j,
        alpha_tensor=alpha_j,
        beta_tensor=beta_j,
        prob_tensor=prob_j,
        dprob_tensor=dprob_j,
        b_ptrs=b_ptrs_j,
        n=n_weight,
        b_dtype="bfloat16",
        d_dtype="bfloat16",
        generate_dbias=True,
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream

    for key in ("d_row_tensor", "dprob_tensor", "dbias_tensor"):
        np.testing.assert_array_equal(
            np.asarray(result_j[key]).astype(np.float32),
            result_t[key].float().cpu().numpy(),
            err_msg=f"grouped dGLU {key}: JAX output differs from torch output on identical input bytes",
        )


@pytest.mark.L0
def test_grouped_gemm_dglu_jax_jit_matches_eager():
    """XLA custom-call entry point (cudnn.jax.call): bit-identical to the eager JAX wrapper."""
    import cutlass.jax

    if not cutlass.jax.is_available():
        pytest.skip("CuTeDSL JAX extensions unavailable (jax >= 0.5 required)")
    skip_unless_sm100()
    from cudnn import grouped_gemm_dglu_jax_sm100, grouped_gemm_dglu_wrapper_sm100

    m, n_weight, k, experts = 512, 128, 128, 2
    a_np, c_np, b_storage_np, offsets_np, alpha_np, beta_np, prob_np = make_problem(m, n_weight, k, experts)

    a_j = jnp.asarray(a_np)
    c_j = jnp.asarray(c_np)
    b_experts_j = [jnp.asarray(b_storage_np[i]) for i in range(experts)]  # per-expert (n_weight, k) k-major
    offsets_j, alpha_j, beta_j, prob_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, beta_np, prob_np))
    # Eager path only: kernel-written output buffer, zero-initialized and materialized
    # before its pointer is used (the jit entry allocates dprob as a donated output).
    dprob_j = jax.block_until_ready(jnp.zeros((m, 1, 1), dtype=jnp.float32))
    jax.block_until_ready((a_j, c_j, offsets_j, alpha_j, beta_j, prob_j, *b_experts_j))
    b_ptrs_j = _packed_jax_ptrs(b_experts_j)

    # Eager wrapper baseline on the same bytes; the last padded offset covers every
    # row, so full-buffer bitwise comparison against the jit path is well-defined.
    result_eager = grouped_gemm_dglu_wrapper_sm100(
        a_tensor=a_j,
        c_tensor=c_j,
        sfa_tensor=None,
        padded_offsets=offsets_j,
        alpha_tensor=alpha_j,
        beta_tensor=beta_j,
        prob_tensor=prob_j,
        dprob_tensor=dprob_j,
        b_ptrs=b_ptrs_j,
        n=n_weight,
        b_dtype="bfloat16",
        generate_dbias=True,
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream
    expected = {key: np.asarray(result_eager[key]).view(np.uint8) for key in ("d_row_tensor", "dprob_tensor", "dbias_tensor")}

    def check(d_row_tensor, dprob_tensor, dbias_tensor):
        jax.block_until_ready((d_row_tensor, dprob_tensor, dbias_tensor))
        for got, key in ((d_row_tensor, "d_row_tensor"), (dprob_tensor, "dprob_tensor"), (dbias_tensor, "dbias_tensor")):
            np.testing.assert_array_equal(
                np.asarray(got).view(np.uint8),
                expected[key],
                err_msg=f"grouped dGLU {key}: jit output differs from eager wrapper output on identical input bytes",
            )

    # Eager custom call
    check(*grouped_gemm_dglu_jax_sm100(a_j, c_j, offsets_j, alpha_j, beta_j, b_ptrs_j, n_weight, prob_j, generate_dbias=True))

    # Under jax.jit, twice (compiled-kernel / registration cache). n stays static.
    jitted = jax.jit(
        lambda a, c, offsets, alpha, beta, ptrs, prob: grouped_gemm_dglu_jax_sm100(a, c, offsets, alpha, beta, ptrs, n_weight, prob, generate_dbias=True),
    )
    check(*jitted(a_j, c_j, offsets_j, alpha_j, beta_j, b_ptrs_j, prob_j))
    check(*jitted(a_j, c_j, offsets_j, alpha_j, beta_j, b_ptrs_j, prob_j))

    # generate_dbias=False returns (d_row, dprob, None)
    d_row_only, dprob_only, dbias_none = grouped_gemm_dglu_jax_sm100(a_j, c_j, offsets_j, alpha_j, beta_j, b_ptrs_j, n_weight, prob_j)
    jax.block_until_ready((d_row_only, dprob_only))
    assert dbias_none is None
    np.testing.assert_array_equal(np.asarray(d_row_only).view(np.uint8), expected["d_row_tensor"])
    np.testing.assert_array_equal(np.asarray(dprob_only).view(np.uint8), expected["dprob_tensor"])


@pytest.mark.L0
def test_grouped_gemm_dglu_jax_errors():
    skip_unless_sm100()
    from cudnn import grouped_gemm_dglu_wrapper_sm100

    m, n_weight, k, experts = 512, 128, 128, 2
    a_np, c_np, b_storage_np, offsets_np, alpha_np, beta_np, prob_np = make_problem(m, n_weight, k, experts)
    a_j = jnp.asarray(a_np)
    c_j = jnp.asarray(c_np)
    offsets_j, alpha_j, beta_j, prob_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, beta_np, prob_np))
    dprob_j = jnp.zeros((m, 1, 1), dtype=jnp.float32)

    # Dense weight mode: (n, k, experts) expert-outermost strides are inexpressible.
    b_dense_j = jnp.asarray(b_storage_np)  # (experts, n, k): not the dense-mode layout
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        grouped_gemm_dglu_wrapper_sm100(
            a_tensor=a_j,
            c_tensor=c_j,
            sfa_tensor=None,
            padded_offsets=offsets_j,
            alpha_tensor=alpha_j,
            beta_tensor=beta_j,
            prob_tensor=prob_j,
            dprob_tensor=dprob_j,
            b_tensor=b_dense_j,
        )

    # Block-scaled backend (fp4/fp8 dtypes): MMA-interleaved SF layouts are inexpressible.
    b_experts_j = [jnp.asarray(b_storage_np[i]) for i in range(experts)]
    jax.block_until_ready(b_experts_j)
    b_ptrs_j = _packed_jax_ptrs(b_experts_j)
    a_fp4_j = jnp.asarray(np.zeros((m, k, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        grouped_gemm_dglu_wrapper_sm100(
            a_tensor=a_fp4_j,
            c_tensor=c_j,
            sfa_tensor=None,
            padded_offsets=offsets_j,
            alpha_tensor=alpha_j,
            beta_tensor=beta_j,
            prob_tensor=prob_j,
            dprob_tensor=dprob_j,
            b_ptrs=b_ptrs_j,
            n=n_weight,
            b_dtype="uint8",
        )
