# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX coverage for the SM100 grouped GEMM Quant wrapper.

JAX contract: every configuration of this kernel is block-scaled and consumes the
scale-factor tensor sfa (and, in dense mode, sfb; plus the sfd outputs for FP8 configs)
as MMA-tiled (32, 4, m//128, 4, rest_k, l) strided views built via torch .permute().
Those layouts are not expressible as row-major JAX arrays, so JAX inputs are rejected
with a clear ValueError at both the wrapper and the API class — in dense AND discrete
(b_ptrs) weight modes, since sfa is required in both. Torch behavior is unchanged.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
import jax.numpy as jnp


def make_jax_inputs(m=256, n=256, k=128, experts=2):
    """Plausible (physical-layout) inputs; rejection fires before shape validation."""
    rng = np.random.default_rng(20260809)
    a_j = jnp.asarray(rng.integers(0, 255, (m, k, 1), dtype=np.uint8).view(ml_dtypes.float8_e4m3fn))
    rest_k = -(-(-(-k // 32) // 4))  # ceil_div(ceil_div(k, 32), 4)
    sfa_j = jnp.asarray(rng.integers(0, 127, (32, 4, -(-m // 128), 4, rest_k, 1), dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu))
    group_m = m // experts
    offsets_j = jnp.asarray(np.arange(group_m, m + 1, group_m, dtype=np.int32))
    alpha_j = jnp.asarray(np.ones(experts, dtype=np.float32))
    return a_j, sfa_j, offsets_j, alpha_j


@pytest.mark.L0
def test_grouped_gemm_quant_jax_discrete_rejected_with_clear_error():
    from cudnn import grouped_gemm_quant_wrapper_sm100

    m, n, k, experts = 256, 256, 128, 2
    a_j, sfa_j, offsets_j, alpha_j = make_jax_inputs(m, n, k, experts)
    # Discrete mode ships weights as pointer arrays, but sfa is still an MMA-tiled
    # cute tensor argument, so the whole jax config is rejected up front.
    ptrs_j = jnp.asarray(np.zeros(8 * experts, dtype=np.uint8))
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        grouped_gemm_quant_wrapper_sm100(
            a_tensor=a_j,
            sfa_tensor=sfa_j,
            padded_offsets=offsets_j,
            alpha_tensor=alpha_j,
            b_ptrs=ptrs_j,
            sfb_ptrs=ptrs_j,
            n=n,
            b_dtype="float8_e4m3fn",
        )


@pytest.mark.L0
def test_grouped_gemm_quant_jax_api_class_rejected():
    from cudnn.gemm.cutedsl.grouped.quant.api import GroupedGemmQuantSm100

    m, n, k, experts = 256, 256, 128, 2
    a_j, sfa_j, offsets_j, alpha_j = make_jax_inputs(m, n, k, experts)
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        GroupedGemmQuantSm100(
            sample_a=a_j,
            sample_sfa=sfa_j,
            sample_padded_offsets=offsets_j,
            sample_alpha=alpha_j,
            sample_d=None,
            num_experts=experts,
            b_shape=(n, k),
            b_dtype="float8_e4m3fn",
        )


@pytest.mark.L0
def test_grouped_gemm_quant_unknown_framework_rejected():
    from cudnn import grouped_gemm_quant_wrapper_sm100

    a_np = np.zeros((256, 128, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported tensor framework 'numpy'"):
        grouped_gemm_quant_wrapper_sm100(
            a_tensor=a_np,
            sfa_tensor=None,
            padded_offsets=None,
            alpha_tensor=None,
        )
