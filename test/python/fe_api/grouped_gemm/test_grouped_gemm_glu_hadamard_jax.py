# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX coverage for the SM100 grouped GEMM GLU + Hadamard wrapper.

The GLU + Hadamard fusion is block-scaled only: its mandatory scale-factor inputs
(sfa/sfb) use an MMA-interleaved 6-D layout with no row-major equivalent, so no
configuration of this API is expressible as JAX arrays. The wrapper and the API
class must therefore reject JAX inputs with a clear error.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import skip_unless_sm100


@pytest.mark.L0
def test_grouped_gemm_glu_hadamard_jax_rejected():
    skip_unless_sm100()
    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100

    m, k, experts = 512, 128, 2
    a_j = jnp.asarray(np.zeros((m, k // 2, 1), dtype=np.uint8))  # packed fp4 container
    offsets_j = jnp.asarray(np.array([256, 512], dtype=np.int32))
    alpha_j = jnp.asarray(np.ones(experts, dtype=np.float32))
    prob_j = jnp.asarray(np.ones((m, 1, 1), dtype=np.float32))
    sfa_j = jnp.asarray(np.zeros((1,), dtype=np.uint8))  # never inspected: rejection happens first

    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        grouped_gemm_glu_hadamard_wrapper_sm100(
            a_tensor=a_j,
            sfa_tensor=sfa_j,
            padded_offsets=offsets_j,
            alpha_tensor=alpha_j,
            prob_tensor=prob_j,
            b_ptrs=jnp.asarray(np.zeros(8 * experts, dtype=np.uint8)),
            sfb_ptrs=jnp.asarray(np.zeros(8 * experts, dtype=np.uint8)),
            n=256,
            b_dtype="uint8",
        )


@pytest.mark.L0
def test_grouped_gemm_glu_hadamard_class_jax_rejected():
    skip_unless_sm100()
    from cudnn.gemm.cutedsl.grouped.glu_hadamard.api import GroupedGemmGluHadamardSm100

    a_j = jnp.asarray(np.zeros((512, 64, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        GroupedGemmGluHadamardSm100(
            sample_a=a_j,
            sample_c=None,
            sample_d=None,
            sample_sfa=None,
            sample_padded_offsets=None,
            sample_alpha=None,
            sample_prob=None,
            num_experts=2,
            b_shape=(256, 128),
            b_dtype="uint8",
        )
