# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AMAX belongs to a wrapper invocation, even when its compiled plan is cached."""

import pytest
import torch
from cuda.bindings import driver as cuda
from cudnn.gemm.cutedsl.grouped.swiglu import api
from gemm.cutedsl.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors


@pytest.mark.L0
def test_cached_swiglu_amax_is_per_call():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ is required")
    torch.manual_seed(1233)
    api._cache_of_GroupedGemmSwigluSm100Objects.clear()
    inputs = allocate_grouped_gemm_input_tensors(
        n=512,
        k=1024,
        l=4,
        group_m_list=[32] * 4,
        ab_dtype=torch.float8_e4m3fn,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=32,
        m_aligned=256,
    )
    inputs["alpha_tensor"].fill_(1)
    inputs["prob_tensor"].fill_(1)
    args = {key: inputs[key] for key in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "alpha_tensor", "prob_tensor", "norm_const_tensor")}
    args.update(
        padded_offsets=inputs["padded_offsets_tensor"],
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        cd_major="n",
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        vector_f32=False,
        m_aligned=256,
        discrete_col_sfd=False,
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    first = api.grouped_gemm_swiglu_wrapper_sm100(**args)
    before = first["amax_tensor"].clone()
    assert bool((before > 0).all())
    plans = list(api._cache_of_GroupedGemmSwigluSm100Objects.values())
    inputs["alpha_tensor"].zero_()
    second = api.grouped_gemm_swiglu_wrapper_sm100(**args)
    assert list(api._cache_of_GroupedGemmSwigluSm100Objects.values()) == plans
    torch.testing.assert_close(second["amax_tensor"], torch.zeros_like(before), rtol=0, atol=0)
    torch.testing.assert_close(first["amax_tensor"], before, rtol=0, atol=0)
    assert first["amax_tensor"].data_ptr() != second["amax_tensor"].data_ptr()
