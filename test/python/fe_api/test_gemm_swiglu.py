import torch

import pytest
from test_utils import torch_fork_set_rng
from fe_api.test_gemm_swiglu_utils import (
    allocate_input_tensors,
    allocate_output_tensors,
    check_ref_gemm_swiglu,
    with_gemm_swiglu_params,
    gemm_swiglu_init,
    with_gemm_swiglu_quant_params,
    check_ref_gemm_swiglu_quant,
)


"""
GemmSwiglu API with explicit set_params, compile, and execute paths. 
Use this method when running one static configuration for each GemmSwiglu object.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_swiglu_params
def test_gemm_swiglu_compile_execute(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    ab12_dtype,
    acc_dtype,
    c_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    try:
        from cudnn import GemmSwigluSm100
        from cuda.bindings import driver as cuda
    except ImportError as e:
        # raise e
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )
    cfg = gemm_swiglu_init(
        request,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        ab12_dtype,
        acc_dtype,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, _, b_torch, _, _, _, _, _, _ = allocate_input_tensors(
        cfg["m"],
        cfg["n"],
        cfg["k"],
        cfg["l"],
        cfg["ab_dtype"],
        cfg["a_major"],
        cfg["b_major"],
    )
    ab12_torch, c_torch, _, _, _ = allocate_output_tensors(
        cfg["m"], cfg["n"], cfg["l"], cfg["ab12_dtype"], cfg["c_dtype"], cfg["c_major"]
    )

    gemm_swiglu = GemmSwigluSm100(
        sample_a=a_torch,
        sample_b=b_torch,
        sample_ab12=ab12_torch,
        sample_c=c_torch,
        alpha=cfg["alpha"],
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
    )
    try:
        assert gemm_swiglu.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        # raise e
        pytest.skip(f"Unsupported testcase: {e}")
    gemm_swiglu.compile(current_stream=stream)
    gemm_swiglu.execute(
        a_tensor=a_torch,
        b_tensor=b_torch,
        ab12_tensor=ab12_torch,
        c_tensor=c_torch,
        alpha=cfg["alpha"],
        current_stream=stream,
    )

    check_ref_gemm_swiglu(
        a_torch,
        b_torch,
        ab12_torch,
        c_torch,
        alpha=cfg["alpha"],
        skip_ref=cfg["skip_ref"],
    )


"""
GemmSwiglu API with gemm_swiglu_wrapper:
Use the wrapper to directly call GemmSwiglu without explicit setup and compilation.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_swiglu_params
def test_gemm_swiglu_wrapper(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    ab12_dtype,
    acc_dtype,
    c_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    try:
        from cudnn import gemm_swiglu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError as e:
        print(f"ImportError: {e}")
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )
    cfg = gemm_swiglu_init(
        request,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        ab12_dtype,
        acc_dtype,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, _, b_torch, _, _, _, _, _, _ = allocate_input_tensors(
        cfg["m"],
        cfg["n"],
        cfg["k"],
        cfg["l"],
        cfg["ab_dtype"],
        cfg["a_major"],
        cfg["b_major"],
    )

    try:
        for _ in range(2):  # Run twice to test caching path
            ab12_torch, c_torch = gemm_swiglu_wrapper_sm100(
                a_tensor=a_torch,
                b_tensor=b_torch,
                alpha=cfg["alpha"],
                c_major=cfg["c_major"],
                ab12_dtype=cfg["ab12_dtype"],
                c_dtype=cfg["c_dtype"],
                acc_dtype=cfg["acc_dtype"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    check_ref_gemm_swiglu(
        a_torch,
        b_torch,
        ab12_torch,
        c_torch,
        alpha=cfg["alpha"],
        skip_ref=cfg["skip_ref"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_swiglu_quant_params
def test_gemm_swiglu_compile_execute_quantize(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    ab12_dtype,
    c_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    request,
):
    try:
        from cudnn import GemmSwigluSm100
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )
    cfg = gemm_swiglu_init(
        request,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        ab12_dtype,
        acc_dtype,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    (
        a_torch,
        a_ref,
        b_torch,
        b_ref,
        sfa_tensor,
        sfa_ref,
        sfb_tensor,
        sfb_ref,
        norm_const_tensor,
    ) = allocate_input_tensors(
        cfg["m"],
        cfg["n"],
        cfg["k"],
        cfg["l"],
        cfg["ab_dtype"],
        cfg["a_major"],
        cfg["b_major"],
        is_block_scaled=True,
        sf_vec_size=cfg["sf_vec_size"],
        sf_dtype=cfg["sf_dtype"],
        c_dtype=cfg["c_dtype"],
        norm_const=1.0,
    )

    ab12_torch, c_torch, sfc_tensor, sfc_ref, amax_tensor = allocate_output_tensors(
        cfg["m"],
        cfg["n"],
        cfg["l"],
        cfg["ab12_dtype"],
        cfg["c_dtype"],
        cfg["c_major"],
        is_block_scaled=True,
        sf_vec_size=cfg["sf_vec_size"],
        sf_dtype=cfg["sf_dtype"],
    )

    gemm_swiglu = GemmSwigluSm100(
        sample_a=a_torch,
        sample_b=b_torch,
        sample_ab12=ab12_torch,
        sample_c=c_torch,
        alpha=cfg["alpha"],
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sample_sfa=sfa_tensor,
        sample_sfb=sfb_tensor,
        sample_amax=amax_tensor,
        sample_sfc=sfc_tensor,
        sample_norm_const=norm_const_tensor,
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        ab12_stages=4,
    )
    try:
        assert gemm_swiglu.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    gemm_swiglu.compile(current_stream=stream)
    gemm_swiglu.execute(
        a_tensor=a_torch,
        b_tensor=b_torch,
        ab12_tensor=ab12_torch,
        c_tensor=c_torch,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        amax_tensor=amax_tensor,
        sfc_tensor=sfc_tensor,
        norm_const_tensor=norm_const_tensor,
        alpha=cfg["alpha"],
        current_stream=stream,
    )

    check_ref_gemm_swiglu_quant(
        a_torch,
        a_ref,
        b_torch,
        b_ref,
        sfa_ref,
        sfb_ref,
        ab12_torch,
        c_torch,
        sfc_tensor,
        amax_tensor,
        norm_const_tensor,
        cfg["sf_vec_size"],
        alpha=cfg["alpha"],
        skip_ref=cfg["skip_ref"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_swiglu_quant_params
def test_gemm_swiglu_wrapper_quantize(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    ab12_dtype,
    c_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    request,
):
    try:
        from cudnn import gemm_swiglu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )
    cfg = gemm_swiglu_init(
        request,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        ab12_dtype,
        acc_dtype,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    (
        a_torch,
        a_ref,
        b_torch,
        b_ref,
        sfa_tensor,
        sfa_ref,
        sfb_tensor,
        sfb_ref,
        norm_const_tensor,
    ) = allocate_input_tensors(
        cfg["m"],
        cfg["n"],
        cfg["k"],
        cfg["l"],
        cfg["ab_dtype"],
        cfg["a_major"],
        cfg["b_major"],
        is_block_scaled=True,
        sf_vec_size=cfg["sf_vec_size"],
        sf_dtype=cfg["sf_dtype"],
        c_dtype=cfg["c_dtype"],
        norm_const=1.0,
    )

    try:
        for _ in range(2):  # Run twice to test caching path
            ab12_torch, c_torch, sfc_tensor, amax_tensor = gemm_swiglu_wrapper_sm100(
                a_tensor=a_torch,
                b_tensor=b_torch,
                alpha=cfg["alpha"],
                c_major=cfg["c_major"],
                ab12_dtype=cfg["ab12_dtype"],
                c_dtype=cfg["c_dtype"],
                acc_dtype=cfg["acc_dtype"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                sfa_tensor=sfa_tensor,
                sfb_tensor=sfb_tensor,
                norm_const_tensor=norm_const_tensor,
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
                ab12_stages=4,
                stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    check_ref_gemm_swiglu_quant(
        a_torch,
        a_ref,
        b_torch,
        b_ref,
        sfa_ref,
        sfb_ref,
        ab12_torch,
        c_torch,
        sfc_tensor,
        amax_tensor,
        norm_const_tensor,
        cfg["sf_vec_size"],
        alpha=cfg["alpha"],
        skip_ref=cfg["skip_ref"],
    )
