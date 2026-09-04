# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Grouped GEMM SReLU Forward Kernel (SM100+)

This module tests the contiguous grouped block-scaled GEMM with SReLU activation
for MoE (Mixture of Experts) workloads.

Reference: continugous_blockscaled_grouped_gemm_srelu_quant_fusion.py
"""

import torch
import pytest
from test_utils import require_cutedsl_version, torch_fork_set_rng

require_cutedsl_version("4.7.0")

from fe_api.grouped_gemm.test_grouped_gemm_srelu_utils import (
    grouped_gemm_srelu_init,
    with_grouped_gemm_srelu_params_fp4,
    with_grouped_gemm_srelu_params_fp8,
    allocate_grouped_gemm_input_tensors,
    allocate_grouped_gemm_output_tensors,
    check_ref_grouped_gemm_srelu,
    run_grouped_gemm_srelu_ref,
)
from fe_api.grouped_gemm.test_discrete_grouped_gemm_swiglu_utils import (
    allocate_discrete_input_tensors,
    discrete_grouped_gemm_init,
)

GROUPED_GEMM_SWIGLU_DYNAMIC_SHAPES_M_VALUES = [64, 320, 576, 832, 1088, 1344, 1600, 1856, 2112, 2368]

DISCRETE_GROUPED_GEMM_SRELU_SUPPORTED_CONFIGS = [
    pytest.param(torch.float4_e2m1fn_x2, torch.bfloat16, torch.bfloat16, "k", id="fp4-k-major"),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, "k", id="fp8-k-major"),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, "n", id="fp8-n-major"),
]


def _dense_ref_inputs_from_discrete(inputs):
    ref_inputs = dict(inputs)
    ref_inputs["b_ref"] = torch.cat(inputs["b_ref_list"], dim=2)
    ref_inputs["sfb_ref"] = torch.cat(inputs["sfb_ref_list"], dim=2)
    return ref_inputs


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_srelu_params_fp4
def test_grouped_gemm_srelu_compile_execute_fp4(
    ab_dtype,
    c_dtype,
    d_dtype,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    _test_grouped_gemm_srelu_compile_execute(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major=cd_major,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_srelu_params_fp8
def test_grouped_gemm_srelu_compile_execute_fp8(
    ab_dtype,
    c_dtype,
    d_dtype,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    _test_grouped_gemm_srelu_compile_execute(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major=cd_major,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_srelu_params_fp4
def test_grouped_gemm_srelu_wrapper_fp4(
    ab_dtype,
    c_dtype,
    d_dtype,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    _test_grouped_gemm_srelu_wrapper(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major=cd_major,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_srelu_params_fp8
def test_grouped_gemm_srelu_wrapper_fp8(
    ab_dtype,
    c_dtype,
    d_dtype,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    _test_grouped_gemm_srelu_wrapper(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major=cd_major,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_srelu_wrapper_cache_partial_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_srelu_wrapper_dynamic_shape_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=False,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=1)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_srelu_wrapper_cache_full_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_srelu_wrapper_dynamic_shape_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=True,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_grouped_gemm_srelu_wrapper_uint8_raw_fp4_smoke(request):
    try:
        from cudnn import grouped_gemm_srelu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = grouped_gemm_srelu_init(
        request=request,
        ab_dtype=torch.uint8,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=16,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=True,
        discrete_col_sfd=False,
    )

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )

    outputs = grouped_gemm_srelu_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        acc_dtype=cfg["acc_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    torch.cuda.synchronize()
    assert torch.isfinite(outputs["c_tensor"].float()).all()
    assert torch.isfinite(outputs["d_tensor"].float()).all()
    assert torch.count_nonzero(outputs["d_tensor"]).item() > 0


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
@pytest.mark.parametrize("vector_f32", [False, True], ids=["scalar_f32", "vector_f32"])
def test_grouped_gemm_srelu_wrapper_uint8_raw_fp4_bias_ref(request, vector_f32):
    try:
        from cudnn import grouped_gemm_srelu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = grouped_gemm_srelu_init(
        request=request,
        ab_dtype=torch.uint8,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float32,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=vector_f32,
        discrete_col_sfd=False,
        enable_bias=True,
    )
    cfg.update(
        {
            "n": 256,
            "k": 256,
            "l": 2,
            "group_m_list": [256, 256],
        }
    )

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        enable_bias=cfg["enable_bias"],
    )
    inputs["alpha_tensor"].copy_(torch.tensor([1.0, -1.25], dtype=torch.float32, device=inputs["alpha_tensor"].device))
    inputs["prob_tensor"].uniform_(0.5, 1.5)

    outputs = grouped_gemm_srelu_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        bias_tensor=inputs["bias_tensor"],
        prob_tensor=inputs["prob_tensor"],
        acc_dtype=cfg["acc_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    torch.cuda.synchronize()
    check_ref_grouped_gemm_srelu(
        inputs,
        outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
@pytest.mark.parametrize("ab_dtype,c_dtype,d_dtype,b_major", DISCRETE_GROUPED_GEMM_SRELU_SUPPORTED_CONFIGS)
def test_grouped_gemm_srelu_discrete_compile_execute(request, ab_dtype, c_dtype, d_dtype, b_major):
    try:
        from cudnn import GroupedGemmSreluSm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = discrete_grouped_gemm_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        b_major=b_major,
    )

    inputs = allocate_discrete_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )
    outputs = allocate_grouped_gemm_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )

    api = GroupedGemmSreluSm100(
        sample_a=inputs["a_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        num_experts=cfg["l"],
        b_shape=(cfg["n"], cfg["k"]),
        b_dtype=inputs["b_list"][0].dtype,
        sample_amax=outputs.get("amax_tensor"),
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        sample_prob=inputs.get("prob_tensor"),
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        b_major=cfg["b_major"],
    )

    try:
        assert api.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    api.compile()
    api.execute(
        a_tensor=inputs["a_tensor"],
        b_ptrs=inputs["b_ptrs_tensor"],
        sfb_ptrs=inputs["sfb_ptrs_tensor"],
        c_tensor=outputs["c_tensor"],
        d_tensor=outputs["d_tensor"],
        d_col_tensor=outputs["d_col_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        sfd_row_tensor=outputs.get("sfd_row_tensor"),
        sfd_col_tensor=outputs.get("sfd_col_tensor"),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        prob_tensor=inputs.get("prob_tensor"),
        amax_tensor=outputs.get("amax_tensor"),
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    torch.cuda.synchronize()
    check_ref_grouped_gemm_srelu(
        _dense_ref_inputs_from_discrete(inputs),
        outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
@pytest.mark.parametrize("ab_dtype,c_dtype,d_dtype,b_major", DISCRETE_GROUPED_GEMM_SRELU_SUPPORTED_CONFIGS)
def test_grouped_gemm_srelu_discrete_wrapper(request, ab_dtype, c_dtype, d_dtype, b_major):
    try:
        from cudnn import grouped_gemm_srelu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = discrete_grouped_gemm_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        b_major=b_major,
    )

    inputs = allocate_discrete_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )

    outputs = grouped_gemm_srelu_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        b_ptrs=inputs["b_ptrs_tensor"],
        sfb_ptrs=inputs["sfb_ptrs_tensor"],
        n=cfg["n"],
        b_dtype=inputs["b_list"][0].dtype,
        b_major=cfg["b_major"],
        norm_const_tensor=inputs.get("norm_const_tensor"),
        prob_tensor=inputs["prob_tensor"],
        acc_dtype=cfg["acc_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    torch.cuda.synchronize()
    check_ref_grouped_gemm_srelu(
        _dense_ref_inputs_from_discrete(inputs),
        outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


"""
GroupedGemmSrelu API with explicit check_support, compile, and execute paths.
Use this method when running one static configuration for each GroupedGemmSrelu object.
"""


def _test_grouped_gemm_srelu_compile_execute(
    ab_dtype,
    c_dtype,
    d_dtype,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    try:
        from cudnn import GroupedGemmSreluSm100
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = grouped_gemm_srelu_init(
        request,
        ab_dtype,
        c_dtype,
        d_dtype,
        cd_major,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        sf_dtype,
        vector_f32,
        discrete_col_sfd,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )

    outputs = allocate_grouped_gemm_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )

    api = GroupedGemmSreluSm100(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_d_col=outputs["d_col_tensor"],
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        sample_prob=inputs.get("prob_tensor"),
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
    )

    try:
        assert api.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    api.compile()
    api.execute(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        c_tensor=outputs["c_tensor"],
        d_tensor=outputs["d_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        d_col_tensor=outputs["d_col_tensor"],
        sfd_row_tensor=outputs.get("sfd_row_tensor"),
        sfd_col_tensor=outputs.get("sfd_col_tensor"),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        prob_tensor=inputs.get("prob_tensor"),
        amax_tensor=outputs.get("amax_tensor"),
        current_stream=stream,
    )

    check_ref_grouped_gemm_srelu(
        inputs,
        outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


"""
GroupedGemmSrelu API with grouped_gemm_srelu_wrapper:
Use the wrapper to directly call GroupedGemmSrelu without explicit setup and compilation.
"""


def _test_grouped_gemm_srelu_wrapper(
    ab_dtype,
    c_dtype,
    d_dtype,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    try:
        from cudnn import grouped_gemm_srelu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = grouped_gemm_srelu_init(
        request,
        ab_dtype,
        c_dtype,
        d_dtype,
        cd_major,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        sf_dtype,
        vector_f32,
        discrete_col_sfd,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )

    try:
        for _ in range(2):  # Run twice to test caching path
            outputs = grouped_gemm_srelu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                prob_tensor=inputs.get("prob_tensor"),
                acc_dtype=cfg["acc_dtype"],
                c_dtype=cfg["c_dtype"],
                d_dtype=cfg["d_dtype"],
                cd_major=cfg["cd_major"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
                m_aligned=cfg["m_aligned"],
                discrete_col_sfd=cfg["discrete_col_sfd"],
                current_stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    check_ref_grouped_gemm_srelu(
        inputs,
        outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


def _test_grouped_gemm_srelu_wrapper_dynamic_shape_cache_behavior(
    request,
    monkeypatch,
    use_full_dynamic,
    ab_dtype,
):
    try:
        from cudnn import grouped_gemm_srelu_wrapper_sm100
        from cudnn.gemm.cutedsl.grouped.srelu import api as grouped_gemm_srelu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    if use_full_dynamic:
        monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    else:
        monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)

    grouped_gemm_srelu_api._cache_of_GroupedGemmSreluSm100Objects.clear()

    compile_count = {"value": 0}
    original_compile = grouped_gemm_srelu_api.GroupedGemmSreluSm100.compile

    def counted_compile(self):
        compile_count["value"] += 1
        return original_compile(self)

    monkeypatch.setattr(grouped_gemm_srelu_api.GroupedGemmSreluSm100, "compile", counted_compile)

    d_dtype = torch.float8_e4m3fn if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16

    cfg = grouped_gemm_srelu_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=torch.bfloat16,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2],
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        for group_m in GROUPED_GEMM_SWIGLU_DYNAMIC_SHAPES_M_VALUES:
            group_m_list = [group_m] * cfg["l"]
            inputs = allocate_grouped_gemm_input_tensors(
                n=cfg["n"],
                k=cfg["k"],
                l=cfg["l"],
                group_m_list=group_m_list,
                ab_dtype=cfg["ab_dtype"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
            )

            wrapper_outputs = grouped_gemm_srelu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                prob_tensor=inputs.get("prob_tensor"),
                acc_dtype=cfg["acc_dtype"],
                c_dtype=cfg["c_dtype"],
                d_dtype=cfg["d_dtype"],
                cd_major=cfg["cd_major"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
                m_aligned=cfg["m_aligned"],
                discrete_col_sfd=cfg["discrete_col_sfd"],
                current_stream=stream,
            )
            torch.cuda.synchronize()

            # check_ref_grouped_gemm_srelu(
            #     inputs,
            #     wrapper_outputs,
            #     cfg,
            #     skip_ref=cfg["skip_ref"],
            # )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    finally:
        cache_entries = len(grouped_gemm_srelu_api._cache_of_GroupedGemmSreluSm100Objects)
        grouped_gemm_srelu_api._cache_of_GroupedGemmSreluSm100Objects.clear()

    return compile_count["value"], cache_entries


# =============================================================================
# Soft-clamped SReLU (tanh_clamp_scale)
# =============================================================================
#
# tanh_clamp_scale=s replaces the epilogue's relu(x)**2 with (s*tanh(relu(x)/s))**2, bounding the
# output by s**2 instead of letting it grow unbounded. Unclamped is the s -> infinity limit,
# which is why it is one kernel with a const_expr gate rather than two kernel families.
#
# Two properties drive the coverage below and are not obvious from the parameter names:
#
# 1. ``s`` is a TRACE-TIME constant -- 1/s is folded into the kernel at compile time. A cache
#    key that omits it is therefore a correctness bug, not a missed optimization: the second
#    tanh_clamp_scale in a process would silently reuse the first one's kernel. Hence
#    test_grouped_gemm_srelu_tanh_clamp_scale_cache_key, and hence two live scales in the matrix.
# 2. The clamped branch is the only place in this epilogue with BOTH a packed and a scalar
#    variant -- the unclamped branch is packed-only regardless of vector_f32. So vector_f32
#    False/True is the only coverage the new scalar fallback gets, and both must be in the
#    matrix.

GROUPED_GEMM_SRELU_CLAMP_CONFIGS = [
    pytest.param(torch.uint8, torch.bfloat16, torch.bfloat16, 16, torch.float8_e8m0fnu, True, False, id="fp4-vector_f32"),
    pytest.param(torch.uint8, torch.bfloat16, torch.bfloat16, 16, torch.float8_e8m0fnu, False, False, id="fp4-scalar_f32"),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, True, True, id="fp8-vector_f32"),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True, id="fp8-scalar_f32"),
]

# 16 and 32 are the two production candidates still under evaluation; both have to work.
# None is kept in the matrix so the clamped and unclamped paths are exercised through
# identical test code, which is what makes a regression in the shared plumbing visible.
GROUPED_GEMM_SRELU_CLAMP_SCALES = [
    pytest.param(None, id="unclamped"),
    pytest.param(16.0, id="clamp16"),
    pytest.param(32.0, id="clamp32"),
]

# fp8_e4m3 keeps 3 mantissa bits, so adjacent representable values are up to 2**-3 apart.
_FP8_E4M3_ULP_RTOL = 2.0**-3


def _assert_close_allowing_quant_boundary(actual, ref, *, rtol, atol, ulp_rtol, max_outlier_frac):
    """assert_close, except that elements sitting on an output-quantization boundary may
    differ by up to one representable step of the output dtype.

    This is NOT a loosened tolerance in disguise. With an fp8 output the representable values
    are ~12.5% apart, so an element whose pre-quantization value lands within the fastmath-tanh
    error of a rounding boundary quantizes to the neighbouring code point in the kernel and to
    this one in the fp32 reference -- a full ulp of difference produced by an arbitrarily small
    cause, which no rtol between 1e-2 and 1.3e-1 can distinguish from a real 12% error.
    Measured on GB300 (sm_103) for this matrix: 4-7 of 524288 elements (1.3e-5) at
    tanh_clamp_scale 16/32, zero unclamped. So instead of widening rtol, bound both the SIZE
    of each exception (one ulp) and the COUNT of them -- a systematic error fails the count,
    a boundary straddle does not.
    """
    a = actual.float()
    r = ref.float()
    close = torch.isclose(a, r, rtol=rtol, atol=atol)
    if bool(close.all()):
        return
    outliers = ~close
    frac = outliers.float().mean().item()
    assert frac <= max_outlier_frac, f"{outliers.sum().item()} / {outliers.numel()} elements ({frac:.3g}) outside tolerance, above {max_outlier_frac:.3g}"
    gap = (a - r).abs()[outliers]
    allowed = (r.abs()[outliers] * ulp_rtol).clamp_min(atol)
    assert bool((gap <= allowed).all()), f"an out-of-tolerance element differs by more than one output ulp: max gap {gap.max().item():.4g}"


def _build_srelu_case(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd):
    cfg = grouped_gemm_srelu_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
    )
    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )
    return inputs, cfg


def _run_srelu_case(case, **wrapper_kwargs):
    """Run the forward wrapper over an already-built case, so repeats share identical inputs."""
    from cudnn import grouped_gemm_srelu_wrapper_sm100
    from cuda.bindings import driver as cuda

    inputs, cfg = case
    wrapper_kwargs.setdefault("current_stream", cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    return grouped_gemm_srelu_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        norm_const_tensor=inputs.get("norm_const_tensor"),
        prob_tensor=inputs.get("prob_tensor"),
        acc_dtype=cfg["acc_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        **wrapper_kwargs,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("tanh_clamp_scale", GROUPED_GEMM_SRELU_CLAMP_SCALES)
@pytest.mark.parametrize(
    "ab_dtype,c_dtype,d_dtype,sf_vec_size,sf_dtype,vector_f32,discrete_col_sfd",
    GROUPED_GEMM_SRELU_CLAMP_CONFIGS,
)
def test_grouped_gemm_srelu_wrapper_tanh_clamp_scale(
    request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd, tanh_clamp_scale
):
    """The fused clamped epilogue agrees with the torch reference across both f32 variants."""
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    case = _build_srelu_case(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd)
    inputs, cfg = case
    # check_ref_grouped_gemm_srelu reads this to pick the clamped branch of the reference.
    cfg["tanh_clamp_scale"] = tanh_clamp_scale

    try:
        outputs = _run_srelu_case(case, tanh_clamp_scale=tanh_clamp_scale)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    if d_dtype is torch.float8_e4m3fn and tanh_clamp_scale is not None:
        # The whole output set at one-ulp rtol, so nothing goes unchecked...
        check_ref_grouped_gemm_srelu(inputs, outputs, cfg, rtol=_FP8_E4M3_ULP_RTOL + 1e-2, skip_ref=cfg["skip_ref"])
        # ...and D, the tensor the epilogue actually produces, still checked sharply.
        if not cfg["skip_ref"]:
            ref_tensors = run_grouped_gemm_srelu_ref(
                a_ref=inputs["a_ref"],
                b_ref=inputs["b_ref"],
                sfa_ref=inputs["sfa_ref"],
                sfb_ref=inputs["sfb_ref"],
                alpha_tensor=inputs["alpha_tensor"],
                prob_tensor=inputs["prob_tensor"],
                aligned_group_m_list=inputs["aligned_group_m_list"],
                valid_m=inputs["valid_m"],
                generate_amax=outputs.get("amax_tensor") is not None,
                generate_sfd=outputs.get("sfd_row_tensor") is not None and outputs.get("sfd_col_tensor") is not None,
                norm_const_tensor=inputs.get("norm_const_tensor"),
                c_dtype=cfg["c_dtype"],
                d_dtype=cfg["d_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                sf_dtype=cfg["sf_dtype"],
                tanh_clamp_scale=tanh_clamp_scale,
            )
            _assert_close_allowing_quant_boundary(
                outputs["d_tensor"],
                ref_tensors["d_ref"],
                rtol=1e-2,
                atol=1e-1,
                ulp_rtol=_FP8_E4M3_ULP_RTOL,
                max_outlier_frac=1e-3,
            )
    else:
        check_ref_grouped_gemm_srelu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
@pytest.mark.parametrize("vector_f32", [False, True], ids=["scalar_f32", "vector_f32"])
def test_grouped_gemm_srelu_tanh_clamp_scale_none_is_bit_identical(request, vector_f32):
    """tanh_clamp_scale=None must be BIT-identical to not passing the argument at all.

    The headline claim of this feature is that it is purely additive: existing callers get
    the same kernel and the same bits as before. Every other test here checks the clamped
    path against a reference, which by construction cannot detect a regression in the
    unclamped one. This compares the two call forms directly, at torch.equal rather than a
    tolerance -- the const_expr guard means both should trace to the same program, so
    anything short of bit-equality is a real behaviour change.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    case = _build_srelu_case(request, torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, vector_f32, True)

    try:
        omitted = _run_srelu_case(case)  # exactly the pre-change call signature
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    explicit_none = _run_srelu_case(case, tanh_clamp_scale=None)
    torch.cuda.synchronize()

    # d_col and the scale factors are only written when the fp8 SFD path is active;
    # otherwise the kernel leaves them at whatever their allocation happened to contain,
    # which differs between two independent calls for reasons unrelated to this flag.
    compared = ["c_tensor", "d_tensor"]
    if omitted.get("sfd_row_tensor") is not None:
        compared += ["d_col_tensor", "sfd_row_tensor", "sfd_col_tensor"]
    if omitted.get("amax_tensor") is not None:
        compared.append("amax_tensor")

    for key in compared:
        a, b = omitted[key], explicit_none[key]
        assert a is not None and b is not None, f"{key} missing from one of the two runs"
        assert torch.equal(a, b), f"{key} differs between omitted and tanh_clamp_scale=None"


@pytest.mark.L0
@torch_fork_set_rng(seed=3)
def test_grouped_gemm_srelu_tanh_clamp_scale_cache_key(request, monkeypatch):
    """Two clamp scales in one process must compile two kernels, not reuse the first.

    ``s`` is folded into the kernel at trace time, so this is a correctness test: if
    tanh_clamp_scale were missing from the cache key, the s=32 call would silently return the
    s=16 kernel's numbers. Both the compile count and the per-scale reference check have to
    hold -- the count alone would pass a kernel that compiled twice and computed the wrong
    thing, and the reference check alone would pass if the cache key were over-specified.
    """
    try:
        import cudnn  # noqa: F401
        from cudnn.gemm.cutedsl.grouped.srelu import api as grouped_gemm_srelu_api
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    monkeypatch.setattr(grouped_gemm_srelu_api, "_cache_of_GroupedGemmSreluSm100Objects", {})

    compile_count = {"value": 0}
    original_compile = grouped_gemm_srelu_api.GroupedGemmSreluSm100.compile

    def counted_compile(self):
        compile_count["value"] += 1
        return original_compile(self)

    monkeypatch.setattr(grouped_gemm_srelu_api.GroupedGemmSreluSm100, "compile", counted_compile)

    args = (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)
    case = _build_srelu_case(request, *args)
    inputs, cfg = case

    results = {}
    for tanh_clamp_scale in (16.0, 32.0):
        try:
            results[tanh_clamp_scale] = _run_srelu_case(case, tanh_clamp_scale=tanh_clamp_scale)
        except (ValueError, NotImplementedError) as e:
            pytest.skip(f"Unsupported testcase: {e}")
        torch.cuda.synchronize()

    assert compile_count["value"] == 2, f"expected one compile per tanh_clamp_scale, got {compile_count['value']}"

    for tanh_clamp_scale, outputs in results.items():
        cfg["tanh_clamp_scale"] = tanh_clamp_scale
        # fp8 D: one-ulp rtol, for the quantization-boundary reason documented on
        # _assert_close_allowing_quant_boundary. The sharp per-element check of D lives in
        # test_grouped_gemm_srelu_wrapper_tanh_clamp_scale; here the point is the cache key.
        check_ref_grouped_gemm_srelu(inputs, outputs, cfg, rtol=_FP8_E4M3_ULP_RTOL + 1e-2, skip_ref=cfg["skip_ref"])

    # Same inputs, different s => different outputs. Without this, a kernel that ignored
    # tanh_clamp_scale entirely (computing plain sqrelu twice) would still pass the count check,
    # and would pass the reference check too if the reference had the same bug.
    if not cfg["skip_ref"]:
        assert not torch.equal(results[16.0]["d_tensor"], results[32.0]["d_tensor"])


@pytest.mark.L0
@torch_fork_set_rng(seed=5)
@pytest.mark.parametrize("vector_f32", [False, True], ids=["scalar_f32", "vector_f32"])
def test_grouped_gemm_srelu_tanh_clamp_scale_saturated(request, vector_f32):
    """The saturated tail: for relu(C) >> s the output must converge to the exact bound s^2*w.

    This is the regime the feature exists for -- it is what bounds the MXFP8 dynamic range --
    and it is the one the default random inputs barely reach, so it needs its own case rather
    than another tanh_clamp_scale against the same distribution. It is also the regime where a
    relative tolerance is the wrong instrument: the reference value is s^2*w and the fused
    value approaches it from below, so the check is on the absolute gap.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    # d_dtype bf16 rather than fp8: fp8 output quantization is coarser than the effect under
    # test and would set the tolerance floor instead of the tanh.
    args = (torch.uint8, torch.bfloat16, torch.bfloat16, 16, torch.float8_e8m0fnu, vector_f32, False)
    case = _build_srelu_case(request, *args)
    inputs, cfg = case

    # s small enough that essentially the whole positive tail is saturated. Chosen from the
    # measured relu(C) distribution of these inputs, not assumed.
    tanh_clamp_scale = 0.5
    cfg["tanh_clamp_scale"] = tanh_clamp_scale

    try:
        outputs = _run_srelu_case(case, tanh_clamp_scale=tanh_clamp_scale)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    if cfg["skip_ref"]:
        pytest.skip("--skip-ref")

    check_ref_grouped_gemm_srelu(inputs, outputs, cfg, skip_ref=False)

    # And the property the reference cannot check for us, because it shares the tanh: the
    # output is bounded, everywhere, with no reliance on the reference.
    #
    # The bound is |d| <= s^2*|w|, NOT d <= s^2*w. This harness draws the routing probability
    # from torch.randint(-2, 2) (srelu_utils:357), so w is one of {-2,-1,0,1} and is negative
    # about half the time; for w < 0 the inequality d = c^2*w <= s^2*w points the other way.
    # The signed form passes only by accident on a non-negative-w harness.
    valid_m = inputs["valid_m"]
    prob = inputs["prob_tensor"].float().expand(-1, cfg["n"], -1)[:valid_m]
    d = outputs["d_tensor"].float()[:valid_m].abs()
    bound = (tanh_clamp_scale * tanh_clamp_scale) * prob.abs()
    # bf16 storage of d rounds up as often as down, so the bound is only exact up to one ulp
    # of the stored value.
    ulp = (bound * 2.0**-8).clamp_min(torch.finfo(torch.bfloat16).tiny)
    assert torch.all(d <= bound + ulp), f"clamped |output| exceeded s^2*|w|: max excess {(d - bound).max().item():.4g}"
    # Non-vacuous: the saturated regime was actually reached, on rows where w != 0.
    assert torch.any(d > 0.5 * bound), "no element reached the saturated regime -- test is vacuous"


@pytest.mark.L0
def test_grouped_gemm_srelu_tanh_clamp_scale_validation(request):
    """tanh_clamp_scale must be finite and positive; the API rejects the rest at construction."""
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    args = (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)
    case = _build_srelu_case(request, *args)

    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tanh_clamp_scale"):
            _run_srelu_case(case, tanh_clamp_scale=bad)
