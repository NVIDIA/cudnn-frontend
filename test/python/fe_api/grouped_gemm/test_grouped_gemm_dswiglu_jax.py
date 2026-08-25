# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX coverage for the SM100 grouped GEMM dSwiGLU backward wrapper.

JAX contract: this backward API only supports dense weight mode, whose
expert-outermost strided B layout (n, k, l) has no row-major JAX equivalent,
so JAX inputs are rejected with a clear error (and unknown frameworks with
an "Unsupported tensor framework" error).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import skip_unless_sm100


def _make_jax_inputs(m=256, n=128, k=128, l=2):
    rng = np.random.default_rng(20260809)
    a_j = jnp.asarray((rng.integers(-4, 5, size=(m, k, 1)) * 0.25).astype(ml_dtypes.float8_e4m3fn))
    b_j = jnp.asarray((rng.integers(-4, 5, size=(n, k, l)) * 0.25).astype(ml_dtypes.float8_e4m3fn))
    c_j = jnp.asarray(rng.standard_normal((m, n * 2, 1), dtype=np.float32).astype(ml_dtypes.bfloat16))
    rk = ((k + 31) // 32 + 3) // 4
    sfa_j = jnp.asarray((2.0 ** rng.integers(-2, 3, size=(1, (m + 127) // 128, rk, 32, 4, 4))).astype(ml_dtypes.float8_e8m0fnu))
    sfb_j = jnp.asarray((2.0 ** rng.integers(-2, 3, size=(l, (n + 127) // 128, rk, 32, 4, 4))).astype(ml_dtypes.float8_e8m0fnu))
    offsets_j = jnp.asarray(np.arange(m // l, m + 1, m // l, dtype=np.int32))
    alpha_j = jnp.asarray(np.ones(l, dtype=np.float32))
    beta_j = jnp.asarray(np.ones(l, dtype=np.float32))
    prob_j = jnp.asarray(np.ones((m, 1, 1), dtype=np.float32))
    return a_j, b_j, c_j, sfa_j, sfb_j, offsets_j, alpha_j, beta_j, prob_j


@pytest.mark.L0
def test_grouped_gemm_dswiglu_jax_rejected():
    """JAX inputs are rejected: the dense-only B layout is not expressible as JAX arrays."""
    skip_unless_sm100()
    from cudnn import grouped_gemm_dswiglu_wrapper_sm100

    a_j, b_j, c_j, sfa_j, sfb_j, offsets_j, alpha_j, beta_j, prob_j = _make_jax_inputs()
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        grouped_gemm_dswiglu_wrapper_sm100(
            a_tensor=a_j,
            b_tensor=b_j,
            c_tensor=c_j,
            sfa_tensor=sfa_j,
            sfb_tensor=sfb_j,
            padded_offsets=offsets_j,
            alpha_tensor=alpha_j,
            beta_tensor=beta_j,
            prob_tensor=prob_j,
            sf_vec_size=32,
        )


@pytest.mark.L0
def test_grouped_gemm_dswiglu_jax_api_class_rejected():
    """Constructing the API class directly with JAX samples raises the same clear error."""
    skip_unless_sm100()
    from cudnn import GroupedGemmDswigluSm100

    a_j, b_j, c_j, sfa_j, sfb_j, offsets_j, alpha_j, beta_j, prob_j = _make_jax_inputs()
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        GroupedGemmDswigluSm100(
            sample_a=a_j,
            sample_b=b_j,
            sample_c=c_j,
            sample_d_row=c_j,
            sample_d_col=c_j,
            sample_sfa=sfa_j,
            sample_sfb=sfb_j,
            sample_padded_offsets=offsets_j,
            sample_alpha=alpha_j,
            sample_beta=beta_j,
            sample_prob=prob_j,
            sample_dprob=prob_j,
            sf_vec_size=32,
        )


@pytest.mark.L0
def test_grouped_gemm_dswiglu_unknown_framework_rejected():
    """Plain numpy arrays are neither torch nor JAX device tensors: clear unsupported error."""
    skip_unless_sm100()
    from cudnn import grouped_gemm_dswiglu_wrapper_sm100

    m, n, k, l = 256, 128, 128, 2
    a_np = np.zeros((m, k, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported tensor framework"):
        grouped_gemm_dswiglu_wrapper_sm100(
            a_tensor=a_np,
            b_tensor=np.zeros((n, k, l), dtype=np.uint8),
            c_tensor=np.zeros((m, n * 2, 1), dtype=np.float32),
            sfa_tensor=None,
            sfb_tensor=None,
            padded_offsets=np.arange(m // l, m + 1, m // l, dtype=np.int32),
            alpha_tensor=np.ones(l, dtype=np.float32),
            beta_tensor=np.ones(l, dtype=np.float32),
            prob_tensor=np.ones((m, 1, 1), dtype=np.float32),
        )
