"""
Tests for Discrete-weight Grouped GEMM GLU Forward Kernel (SM100+)

This module tests the discrete-weight block-scaled grouped GEMM with GLU
activation (SwiGLU/GeGLU) for MoE (Mixture of Experts) workloads.
"""

import torch
import pytest
from test_utils import torch_fork_set_rng
from fe_api.test_fe_api_utils import DYNAMIC_SHAPES_M_VALUES
from fe_api.test_discrete_grouped_gemm_swiglu_utils import (
    discrete_grouped_gemm_init,
    with_discrete_grouped_gemm_params_fp4,
    with_discrete_grouped_gemm_params_fp8,
    allocate_discrete_input_tensors,
    allocate_discrete_output_tensors,
    check_ref_discrete_grouped_gemm,
)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_discrete_grouped_gemm_params_fp4
def test_discrete_grouped_gemm_compile_execute_fp4(
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
    act_func,
    request,
):
    _test_discrete_grouped_gemm_compile_execute(
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
        act_func=act_func,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_discrete_grouped_gemm_params_fp8
def test_discrete_grouped_gemm_compile_execute_fp8(
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
    act_func,
    b_major,
    request,
):
    _test_discrete_grouped_gemm_compile_execute(
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
        act_func=act_func,
        b_major=b_major,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_discrete_grouped_gemm_params_fp4
def test_discrete_grouped_gemm_wrapper_fp4(
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
    act_func,
    request,
):
    _test_discrete_grouped_gemm_wrapper(
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
        act_func=act_func,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_discrete_grouped_gemm_params_fp8
def test_discrete_grouped_gemm_wrapper_fp8(
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
    act_func,
    b_major,
    request,
):
    _test_discrete_grouped_gemm_wrapper(
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
        act_func=act_func,
        b_major=b_major,
        request=request,
    )


"""
DiscreteGroupedGemmSwigluSm100 API with explicit check_support, compile, and execute paths.
All tests use int64 pointer tensors for execute() (the production/graph-safe path).
"""


def _test_discrete_grouped_gemm_compile_execute(
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
    act_func,
    request,
    b_major="k",
    enable_bias=False,
):
    try:
        from cudnn import DiscreteGroupedGemmSwigluSm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = discrete_grouped_gemm_init(
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
        act_func,
        b_major=b_major,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

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
        enable_bias=enable_bias,
    )

    outputs = allocate_discrete_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        num_experts=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )

    api = DiscreteGroupedGemmSwigluSm100(
        sample_a=inputs["a_tensor"],
        num_experts=len(inputs["b_list"]),
        b_shape=(cfg["n"], cfg["k"]),
        b_dtype=inputs["b_list"][0].dtype,
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_d_col=outputs["d_col_tensor"],
        sample_bias=inputs.get("bias_tensor"),
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
        act_func=cfg["act_func"],
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
        c_tensor=outputs["c_tensor"],
        d_tensor=outputs["d_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_ptrs=inputs["sfb_ptrs_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        bias_tensor=inputs.get("bias_tensor"),
        d_col_tensor=outputs["d_col_tensor"],
        sfd_row_tensor=outputs.get("sfd_row_tensor"),
        sfd_col_tensor=outputs.get("sfd_col_tensor"),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        prob_tensor=inputs.get("prob_tensor"),
        amax_tensor=outputs.get("amax_tensor"),
        current_stream=stream,
    )

    check_ref_discrete_grouped_gemm(
        inputs,
        outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


"""
DiscreteGroupedGemmSwiglu API with wrapper:
Use the wrapper to directly call without explicit setup and compilation.
"""


def _test_discrete_grouped_gemm_wrapper(
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
    act_func,
    request,
    b_major="k",
    enable_bias=False,
):
    try:
        from cudnn import discrete_grouped_gemm_swiglu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = discrete_grouped_gemm_init(
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
        act_func,
        b_major=b_major,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

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
        enable_bias=enable_bias,
    )

    try:
        for _ in range(2):  # Run twice to test caching path
            outputs = discrete_grouped_gemm_swiglu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_ptrs=inputs["b_ptrs_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_ptrs=inputs["sfb_ptrs_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                bias_tensor=inputs.get("bias_tensor"),
                n=cfg["n"],
                b_dtype=inputs["b_list"][0].dtype,
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
                act_func=cfg["act_func"],
                b_major=cfg["b_major"],
                current_stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    check_ref_discrete_grouped_gemm(
        inputs,
        outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_discrete_grouped_gemm_compile_execute_with_bias(request):
    _test_discrete_grouped_gemm_compile_execute(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func="swiglu",
        request=request,
        enable_bias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_discrete_grouped_gemm_wrapper_with_bias(request):
    _test_discrete_grouped_gemm_wrapper(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func="swiglu",
        request=request,
        enable_bias=True,
    )


"""
Cross-kernel comparison: verify discrete and contiguous kernels produce
identical outputs when given the exact same input data.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=42)
@pytest.mark.parametrize(
    "ab_dtype,d_dtype,sf_vec_size,sf_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, torch.bfloat16, 32, torch.float8_e8m0fnu, id="fp4"),
        pytest.param(torch.float8_e4m3fn, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, id="fp8"),
    ],
)
def test_discrete_vs_contiguous_match(ab_dtype, d_dtype, sf_vec_size, sf_dtype, request):
    """Both kernels do the same math -- given identical inputs, outputs must be bitwise equal."""
    try:
        from cudnn import DiscreteGroupedGemmSwigluSm100, GroupedGemmSwigluSm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(f"Requires SM100+, found SM{major}0")

    from test_fe_api_utils import create_and_permute_tensor, create_scale_factor_tensor
    from fe_api.test_discrete_grouped_gemm_swiglu_utils import create_mask
    from cudnn.api_base import ceil_div

    n, k, num_experts = 512, 512, 4
    group_m_list = [256] * num_experts
    c_dtype = torch.bfloat16
    m_aligned = 256
    mma_tiler_mn = (256, 256)
    cluster_shape_mn = (2, 1)

    valid_m, _, padded_offsets_tensor = create_mask(group_m_list, m_aligned)
    tensor_m = valid_m

    # --- Shared input data (same seed -> identical values) ---
    a_ref, a_tensor = create_and_permute_tensor(1, tensor_m, k, False, ab_dtype)
    b_ref, b_stacked = create_and_permute_tensor(num_experts, n, k, False, ab_dtype)
    sfa_ref, sfa_tensor = create_scale_factor_tensor(1, tensor_m, k, sf_vec_size, sf_dtype)
    sfb_ref, sfb_stacked = create_scale_factor_tensor(num_experts, n, k, sf_vec_size, sf_dtype)
    alpha_tensor = torch.randint(-2, 2, (num_experts,), dtype=torch.float32, device="cuda").float()
    prob_tensor = torch.randint(-2, 2, (tensor_m, 1, 1), dtype=torch.float32, device="cuda").float()

    norm_const_tensor = None
    if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sf_dtype in [
        torch.float8_e8m0fnu,
        torch.float8_e4m3fn,
    ]:
        norm_const_tensor = torch.tensor([0.01], dtype=torch.float32, device="cuda")

    # Clone each expert's slice into a truly discrete allocation. This ensures
    # the discrete kernel reads from independent GPU addresses (not views into
    # the stacked tensor), while both kernels see identical data.
    b_list = [b_stacked[:, :, i].clone() for i in range(num_experts)]
    sfb_list = [sfb_stacked[:, :, :, :, :, i].clone() for i in range(num_experts)]

    # --- Allocate output tensors ---
    n_out = n // 2

    def make_outputs():
        c = torch.empty_strided((tensor_m, n, 1), (n, 1, tensor_m * n), dtype=c_dtype, device="cuda")
        d = torch.empty_strided(
            (tensor_m, n_out, 1),
            (n_out, 1, tensor_m * n_out),
            dtype=d_dtype,
            device="cuda",
        )
        d_col = torch.empty_strided(
            (tensor_m, n_out, 1),
            (n_out, 1, tensor_m * n_out),
            dtype=d_dtype,
            device="cuda",
        )
        sfd_row, sfd_col, amax = None, None, None
        if d_dtype in [torch.bfloat16, torch.float16]:
            amax = torch.full((num_experts, 1), float("-inf"), dtype=torch.float32, device="cuda")
        if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sf_dtype in [
            torch.float8_e8m0fnu,
            torch.float8_e4m3fn,
        ]:
            perm = (3, 4, 1, 5, 2, 0)
            sf_k_row = ceil_div(n_out, sf_vec_size)
            sfd_row = torch.empty(
                (1, ceil_div(tensor_m, 128), ceil_div(sf_k_row, 4), 32, 4, 4),
                dtype=sf_dtype,
                device="cuda",
            ).permute(perm)
            sf_k_col = ceil_div(tensor_m, sf_vec_size)
            sfd_col = torch.empty(
                (1, ceil_div(n_out, 128), ceil_div(sf_k_col, 4), 32, 4, 4),
                dtype=sf_dtype,
                device="cuda",
            ).permute(perm)
        return c, d, d_col, sfd_row, sfd_col, amax

    c_contig, d_contig, d_col_contig, sfd_row_contig, sfd_col_contig, amax_contig = make_outputs()
    (
        c_discrete,
        d_discrete,
        d_col_discrete,
        sfd_row_discrete,
        sfd_col_discrete,
        amax_discrete,
    ) = make_outputs()

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # --- Run contiguous kernel ---
    contig_api = GroupedGemmSwigluSm100(
        sample_a=a_tensor,
        sample_b=b_stacked,
        sample_c=c_contig,
        sample_d=d_contig,
        sample_sfa=sfa_tensor,
        sample_sfb=sfb_stacked,
        sample_padded_offsets=padded_offsets_tensor,
        sample_alpha=alpha_tensor,
        sample_d_col=d_col_contig,
        sample_sfd_row=sfd_row_contig,
        sample_sfd_col=sfd_col_contig,
        sample_amax=amax_contig,
        sample_norm_const=norm_const_tensor,
        sample_prob=prob_tensor,
        acc_dtype=torch.float32,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=False,
        m_aligned=m_aligned,
    )
    try:
        assert contig_api.check_support()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Contiguous kernel unsupported: {e}")
    contig_api.compile()
    contig_api.execute(
        a_tensor=a_tensor,
        b_tensor=b_stacked,
        c_tensor=c_contig,
        d_tensor=d_contig,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_stacked,
        padded_offsets=padded_offsets_tensor,
        alpha_tensor=alpha_tensor,
        d_col_tensor=d_col_contig,
        sfd_row_tensor=sfd_row_contig,
        sfd_col_tensor=sfd_col_contig,
        amax_tensor=amax_contig,
        norm_const_tensor=norm_const_tensor,
        prob_tensor=prob_tensor,
        current_stream=stream,
    )

    # --- Run discrete kernel ---
    b_ptrs = torch.tensor([b.data_ptr() for b in b_list], dtype=torch.int64, device="cuda")
    sfb_ptrs = torch.tensor([sfb.data_ptr() for sfb in sfb_list], dtype=torch.int64, device="cuda")
    discrete_api = DiscreteGroupedGemmSwigluSm100(
        sample_a=a_tensor,
        num_experts=num_experts,
        b_shape=(n, k),
        b_dtype=b_list[0].dtype,
        sample_c=c_discrete,
        sample_d=d_discrete,
        sample_sfa=sfa_tensor,
        sample_padded_offsets=padded_offsets_tensor,
        sample_alpha=alpha_tensor,
        sample_d_col=d_col_discrete,
        sample_sfd_row=sfd_row_discrete,
        sample_sfd_col=sfd_col_discrete,
        sample_amax=amax_discrete,
        sample_norm_const=norm_const_tensor,
        sample_prob=prob_tensor,
        acc_dtype=torch.float32,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=False,
        m_aligned=m_aligned,
    )
    try:
        assert discrete_api.check_support()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Discrete kernel unsupported: {e}")
    discrete_api.compile()
    discrete_api.execute(
        a_tensor=a_tensor,
        b_ptrs=b_ptrs,
        c_tensor=c_discrete,
        d_tensor=d_discrete,
        sfa_tensor=sfa_tensor,
        sfb_ptrs=sfb_ptrs,
        padded_offsets=padded_offsets_tensor,
        alpha_tensor=alpha_tensor,
        d_col_tensor=d_col_discrete,
        sfd_row_tensor=sfd_row_discrete,
        sfd_col_tensor=sfd_col_discrete,
        amax_tensor=amax_discrete,
        norm_const_tensor=norm_const_tensor,
        prob_tensor=prob_tensor,
        current_stream=stream,
    )

    torch.cuda.synchronize()

    # --- Compare outputs ---
    # The two kernels use different tile schedulers (StaticPersistentTileScheduler
    # vs MoEPersistentTileScheduler), so tiles may be accumulated in a different
    # order. Floating-point addition is not associative, so results may differ
    # slightly despite identical inputs.
    atol, rtol = 1e-1, 1e-2

    torch.testing.assert_close(
        c_contig[:valid_m].cpu().float(),
        c_discrete[:valid_m].cpu().float(),
        atol=atol,
        rtol=rtol,
        msg="C tensor mismatch between contiguous and discrete kernels",
    )
    torch.testing.assert_close(
        d_contig[:valid_m].cpu().float(),
        d_discrete[:valid_m].cpu().float(),
        atol=atol,
        rtol=rtol,
        msg="D tensor mismatch between contiguous and discrete kernels",
    )
    if amax_contig is not None and amax_discrete is not None:
        torch.testing.assert_close(
            amax_contig.cpu(),
            amax_discrete.cpu(),
            atol=atol,
            rtol=rtol,
            msg="AMAX mismatch between contiguous and discrete kernels",
        )
    if sfd_row_contig is not None and sfd_row_discrete is not None:
        torch.testing.assert_close(
            sfd_row_contig.cpu().float(),
            sfd_row_discrete.cpu().float(),
            atol=atol,
            rtol=rtol,
            msg="SFD_row mismatch between contiguous and discrete kernels",
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_discrete_grouped_gemm_wrapper_rejects_invalid_pointer_tensors(request):
    try:
        from cudnn import discrete_grouped_gemm_swiglu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = discrete_grouped_gemm_init(
        request=request,
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func="swiglu",
        b_major="k",
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

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

    bad_b_ptrs = inputs["b_ptrs_tensor"].to(torch.int32)
    with pytest.raises(ValueError, match="b_ptrs must be int64"):
        discrete_grouped_gemm_swiglu_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            b_ptrs=bad_b_ptrs,
            sfa_tensor=inputs["sfa_tensor"],
            sfb_ptrs=inputs["sfb_ptrs_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            n=cfg["n"],
            b_dtype=inputs["b_list"][0].dtype,
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
            act_func=cfg["act_func"],
            b_major=cfg["b_major"],
            current_stream=stream,
        )

    bad_sfb_ptrs = inputs["sfb_ptrs_tensor"][:-1]
    with pytest.raises(ValueError, match="sfb_ptrs length mismatch"):
        discrete_grouped_gemm_swiglu_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            b_ptrs=inputs["b_ptrs_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_ptrs=bad_sfb_ptrs,
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            n=cfg["n"],
            b_dtype=inputs["b_list"][0].dtype,
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
            act_func=cfg["act_func"],
            b_major=cfg["b_major"],
            current_stream=stream,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
def test_discrete_grouped_gemm_wrapper_cache_dynamic_m_smoke(request, monkeypatch):
    compile_count, cache_entries = _test_discrete_grouped_gemm_wrapper_dynamic_m_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
    )

    assert compile_count == 1
    assert cache_entries == 1


def _test_discrete_grouped_gemm_wrapper_dynamic_m_cache_behavior(request, monkeypatch):
    try:
        from cudnn import discrete_grouped_gemm_swiglu_wrapper_sm100
        from cudnn.discrete_grouped_gemm.discrete_grouped_gemm_swiglu import api as discrete_grouped_gemm_swiglu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    discrete_grouped_gemm_swiglu_api._cache_of_DiscreteGroupedGemmSwigluSm100Objects.clear()

    compile_count = {"value": 0}
    original_compile = discrete_grouped_gemm_swiglu_api.DiscreteGroupedGemmSwigluSm100.compile

    def counted_compile(self):
        compile_count["value"] += 1
        return original_compile(self)

    monkeypatch.setattr(discrete_grouped_gemm_swiglu_api.DiscreteGroupedGemmSwigluSm100, "compile", counted_compile)

    cfg = discrete_grouped_gemm_init(
        request=request,
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func="swiglu",
        b_major="k",
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        for group_m in DYNAMIC_SHAPES_M_VALUES:
            inputs = allocate_discrete_input_tensors(
                n=cfg["n"],
                k=cfg["k"],
                num_experts=cfg["l"],
                group_m_list=[group_m] * cfg["l"],
                ab_dtype=cfg["ab_dtype"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
                b_major=cfg["b_major"],
            )

            discrete_grouped_gemm_swiglu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_ptrs=inputs["b_ptrs_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_ptrs=inputs["sfb_ptrs_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                n=cfg["n"],
                b_dtype=cfg["ab_dtype"],
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
                act_func=cfg["act_func"],
                b_major=cfg["b_major"],
                current_stream=stream,
            )
            torch.cuda.synchronize()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    finally:
        cache_entries = len(discrete_grouped_gemm_swiglu_api._cache_of_DiscreteGroupedGemmSwigluSm100Objects)
        discrete_grouped_gemm_swiglu_api._cache_of_DiscreteGroupedGemmSwigluSm100Objects.clear()

    return compile_count["value"], cache_entries
