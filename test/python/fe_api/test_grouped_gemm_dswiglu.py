"""
Tests for Grouped GEMM dSwiGLU Backward Kernel (SM100+)

This module tests the contiguous grouped block-scaled GEMM backward pass
with dSwiGLU activation gradient for MoE (Mixture of Experts) workloads.
"""

import torch
import pytest
from test_utils import torch_fork_set_rng
from fe_api.test_grouped_gemm_swiglu_utils import (
    grouped_gemm_swiglu_init,
    allocate_grouped_gemm_input_tensors,
)
from fe_api.test_grouped_gemm_dswiglu_utils import (
    with_grouped_gemm_dswiglu_params_fp4,
    with_grouped_gemm_dswiglu_params_fp8,
    allocate_grouped_gemm_dswiglu_tensors,
    check_ref_grouped_gemm_dswiglu,
)

GROUPED_GEMM_DSWIGLU_DYNAMIC_SHAPES_M_VALUES = [64, 320, 576, 832, 1088, 1344, 1600, 1856, 2112, 2368]


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_dswiglu_params_fp4
def test_grouped_gemm_dswiglu_compile_execute_fp4(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
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
    _test_grouped_gemm_dswiglu_compile_execute(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        b_major=b_major,
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
@with_grouped_gemm_dswiglu_params_fp8
def test_grouped_gemm_dswiglu_compile_execute_fp8(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
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
    _test_grouped_gemm_dswiglu_compile_execute(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        b_major=b_major,
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
@with_grouped_gemm_dswiglu_params_fp4
def test_grouped_gemm_dswiglu_wrapper_fp4(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
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
    _test_grouped_gemm_dswiglu_wrapper(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        b_major=b_major,
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
@with_grouped_gemm_dswiglu_params_fp8
def test_grouped_gemm_dswiglu_wrapper_fp8(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
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
    _test_grouped_gemm_dswiglu_wrapper(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        b_major=b_major,
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
def test_grouped_gemm_dswiglu_wrapper_cache_partial_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dswiglu_wrapper_dynamic_shape_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=False,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dswiglu_wrapper_cache_full_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dswiglu_wrapper_dynamic_shape_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=True,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dswiglu_wrapper_cache_zero_m_after_compile_partial_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dswiglu_wrapper_zero_m_after_compile_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=False,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dswiglu_wrapper_cache_zero_m_after_compile_full_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dswiglu_wrapper_zero_m_after_compile_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=True,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dswiglu_wrapper_cache_zero_m_before_compile_partial_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dswiglu_wrapper_zero_m_before_compile_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=False,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dswiglu_wrapper_cache_zero_m_before_compile_full_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dswiglu_wrapper_zero_m_before_compile_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=True,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


"""
GroupedGemmDswiglu API with explicit check_support, compile, and execute paths.
Use this method when running one static configuration for each GroupedGemmDswiglu object.
"""


def _test_grouped_gemm_dswiglu_compile_execute(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
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
        from cudnn import GroupedGemmDswigluSm100
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = grouped_gemm_swiglu_init(
        request=request,
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
        b_major=b_major,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        b_major=cfg["b_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )

    inputs, outputs = allocate_grouped_gemm_dswiglu_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        input_tensors=inputs,
    )

    api = GroupedGemmDswigluSm100(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_c=inputs["c_tensor"],
        sample_d_row=outputs["d_row_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_beta=inputs["beta_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_dprob=outputs["dprob_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
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
        c_tensor=inputs["c_tensor"],
        d_row_tensor=outputs["d_row_tensor"],
        d_col_tensor=outputs["d_col_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=inputs["prob_tensor"],
        dprob_tensor=outputs["dprob_tensor"],
        sfd_row_tensor=outputs.get("sfd_row_tensor"),
        sfd_col_tensor=outputs.get("sfd_col_tensor"),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        amax_tensor=outputs.get("amax_tensor"),
        current_stream=stream,
    )

    torch.cuda.synchronize()
    check_ref_grouped_gemm_dswiglu(
        inputs,
        outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


"""
GroupedGemmDswiglu API with grouped_gemm_dswiglu_wrapper:
Use the wrapper to directly call GroupedGemmDswiglu without explicit setup and compilation.
"""


def _test_grouped_gemm_dswiglu_wrapper(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
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
        from cudnn import grouped_gemm_dswiglu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = grouped_gemm_swiglu_init(
        request=request,
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
        b_major=b_major,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        b_major=cfg["b_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )

    inputs, _ = allocate_grouped_gemm_dswiglu_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        input_tensors=inputs,
    )

    try:
        for _ in range(2):  # Run twice to test caching path
            wrapper_outputs = grouped_gemm_dswiglu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                beta_tensor=inputs["beta_tensor"],
                prob_tensor=inputs["prob_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                acc_dtype=cfg["acc_dtype"],
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

    torch.cuda.synchronize()
    check_ref_grouped_gemm_dswiglu(
        inputs,
        wrapper_outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


def _test_grouped_gemm_dswiglu_wrapper_dynamic_shape_cache_behavior(
    request,
    monkeypatch,
    use_full_dynamic,
    ab_dtype,
):
    try:
        from cudnn import grouped_gemm_dswiglu_wrapper_sm100
        from cudnn.grouped_gemm.grouped_gemm_dswiglu import api as grouped_gemm_dswiglu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    if use_full_dynamic:
        monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    else:
        monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)

    grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects.clear()

    compile_count = {"value": 0}
    original_compile = grouped_gemm_dswiglu_api.GroupedGemmDswigluSm100.compile

    def counted_compile(self):
        compile_count["value"] += 1
        return original_compile(self)

    monkeypatch.setattr(grouped_gemm_dswiglu_api.GroupedGemmDswigluSm100, "compile", counted_compile)

    d_dtype = torch.float8_e4m3fn if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16

    cfg = grouped_gemm_swiglu_init(
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
        discrete_col_sfd=False,
        b_major="k",
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        for group_m in GROUPED_GEMM_DSWIGLU_DYNAMIC_SHAPES_M_VALUES:
            group_m_list = [group_m] * cfg["l"]
            inputs = allocate_grouped_gemm_input_tensors(
                n=cfg["n"],
                k=cfg["k"],
                l=cfg["l"],
                group_m_list=group_m_list,
                ab_dtype=cfg["ab_dtype"],
                b_major=cfg["b_major"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
            )

            inputs, _ = allocate_grouped_gemm_dswiglu_tensors(
                tensor_m=inputs["tensor_m"],
                n=cfg["n"],
                l=cfg["l"],
                ab_dtype=cfg["ab_dtype"],
                c_dtype=cfg["c_dtype"],
                d_dtype=cfg["d_dtype"],
                cd_major=cfg["cd_major"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                input_tensors=inputs,
            )

            wrapper_outputs = grouped_gemm_dswiglu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                beta_tensor=inputs["beta_tensor"],
                prob_tensor=inputs["prob_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                acc_dtype=cfg["acc_dtype"],
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
            # check_ref_grouped_gemm_dswiglu(
            #     inputs,
            #     wrapper_outputs,
            #     cfg,
            #     skip_ref=cfg["skip_ref"],
            # )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    finally:
        cache_entries = len(grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects)
        grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects.clear()

    return compile_count["value"], cache_entries


def _test_grouped_gemm_dswiglu_wrapper_zero_m_after_compile_cache_behavior(
    request,
    monkeypatch,
    use_full_dynamic,
    ab_dtype,
):
    return _test_grouped_gemm_dswiglu_wrapper_zero_m_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=use_full_dynamic,
        ab_dtype=ab_dtype,
        group_m_values=[512, 0],
    )


def _test_grouped_gemm_dswiglu_wrapper_zero_m_before_compile_cache_behavior(
    request,
    monkeypatch,
    use_full_dynamic,
    ab_dtype,
):
    return _test_grouped_gemm_dswiglu_wrapper_zero_m_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=use_full_dynamic,
        ab_dtype=ab_dtype,
        group_m_values=[0, 512],
    )


def _test_grouped_gemm_dswiglu_wrapper_zero_m_cache_behavior(
    request,
    monkeypatch,
    use_full_dynamic,
    ab_dtype,
    group_m_values,
):
    try:
        from cudnn import grouped_gemm_dswiglu_wrapper_sm100
        from cudnn.grouped_gemm.grouped_gemm_dswiglu import api as grouped_gemm_dswiglu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    if use_full_dynamic:
        monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    else:
        monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)

    grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects.clear()

    compile_count = {"value": 0}
    original_compile = grouped_gemm_dswiglu_api.GroupedGemmDswigluSm100.compile

    def counted_compile(self):
        compile_count["value"] += 1
        return original_compile(self)

    monkeypatch.setattr(grouped_gemm_dswiglu_api.GroupedGemmDswigluSm100, "compile", counted_compile)

    d_dtype = torch.float8_e4m3fn if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16

    cfg = grouped_gemm_swiglu_init(
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
        discrete_col_sfd=False,
        b_major="k",
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        for group_m in group_m_values:
            group_m_list = [group_m] * cfg["l"]
            inputs = allocate_grouped_gemm_input_tensors(
                n=cfg["n"],
                k=cfg["k"],
                l=cfg["l"],
                group_m_list=group_m_list,
                ab_dtype=cfg["ab_dtype"],
                b_major=cfg["b_major"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
            )

            inputs, _ = allocate_grouped_gemm_dswiglu_tensors(
                tensor_m=inputs["tensor_m"],
                n=cfg["n"],
                l=cfg["l"],
                ab_dtype=cfg["ab_dtype"],
                c_dtype=cfg["c_dtype"],
                d_dtype=cfg["d_dtype"],
                cd_major=cfg["cd_major"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                input_tensors=inputs,
            )

            grouped_gemm_dswiglu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                beta_tensor=inputs["beta_tensor"],
                prob_tensor=inputs["prob_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                acc_dtype=cfg["acc_dtype"],
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
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    finally:
        cache_entries = len(grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects)
        grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects.clear()

    return compile_count["value"], cache_entries
