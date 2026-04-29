"""
Tests for Grouped GEMM SReLU Forward Kernel (SM100+)

This module tests the contiguous grouped block-scaled GEMM with SReLU activation
for MoE (Mixture of Experts) workloads.

Reference: continugous_blockscaled_grouped_gemm_srelu_quant_fusion.py
"""

import torch
import pytest
from test_utils import torch_fork_set_rng
from fe_api.test_grouped_gemm_srelu_utils import (
    grouped_gemm_srelu_init,
    with_grouped_gemm_srelu_params_fp4,
    with_grouped_gemm_srelu_params_fp8,
    allocate_grouped_gemm_input_tensors,
    allocate_grouped_gemm_output_tensors,
    check_ref_grouped_gemm_srelu,
)
from fe_api.test_discrete_grouped_gemm_swiglu_utils import (
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
        from cudnn.grouped_gemm.grouped_gemm_srelu import api as grouped_gemm_srelu_api
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
