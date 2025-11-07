import torch

import pytest
from test_utils import torch_fork_set_rng
from fe_api.test_gemm_swiglu_utils import (
    allocate_input_tensors,
    allocate_output_tensors,
    check_ref_gemm_swiglu,
    with_gemm_swiglu_params,
    gemm_swiglu_init,
)
from cuda.bindings import driver as cuda


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
    c_dtype,
    acc_dtype,
    glu_dtype,
    use_2cta_instrs,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    try:
        from cudnn import GemmSwigluSm100
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
        c_dtype,
        acc_dtype,
        glu_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, b_torch = allocate_input_tensors(
        cfg["m"],
        cfg["n"],
        cfg["k"],
        cfg["l"],
        cfg["ab_dtype"],
        cfg["a_major"],
        cfg["b_major"],
    )
    c_torch, glu_torch = allocate_output_tensors(
        cfg["m"], cfg["n"], cfg["l"], cfg["c_dtype"], cfg["glu_dtype"], cfg["c_major"]
    )

    gemm_swiglu = GemmSwigluSm100(
        sample_a=a_torch,
        sample_b=b_torch,
        sample_c=c_torch,
        sample_glu=glu_torch,
        alpha=cfg["alpha"],
        acc_dtype=cfg["acc_dtype"],
        use_2cta_instrs=cfg["use_2cta_instrs"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
    )
    try:
        assert gemm_swiglu.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    gemm_swiglu.compile(current_stream=stream)
    gemm_swiglu.execute(
        a_tensor=a_torch,
        b_tensor=b_torch,
        c_tensor=c_torch,
        glu_tensor=glu_torch,
        alpha=cfg["alpha"],
        current_stream=stream,
    )

    check_ref_gemm_swiglu(
        a_torch,
        b_torch,
        c_torch,
        glu_torch,
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
    c_dtype,
    acc_dtype,
    glu_dtype,
    use_2cta_instrs,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    try:
        from cudnn import gemm_swiglu_wrapper_sm100
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
        c_dtype,
        acc_dtype,
        glu_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, b_torch = allocate_input_tensors(
        cfg["m"],
        cfg["n"],
        cfg["k"],
        cfg["l"],
        cfg["ab_dtype"],
        cfg["a_major"],
        cfg["b_major"],
    )

    try:
        c_torch, glu_torch = gemm_swiglu_wrapper_sm100(
            a_tensor=a_torch,
            b_tensor=b_torch,
            alpha=cfg["alpha"],
            c_major=cfg["c_major"],
            c_dtype=cfg["c_dtype"],
            glu_dtype=cfg["glu_dtype"],
            acc_dtype=cfg["acc_dtype"],
            use_2cta_instrs=cfg["use_2cta_instrs"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            stream=stream,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    check_ref_gemm_swiglu(
        a_torch,
        b_torch,
        c_torch,
        glu_torch,
        alpha=cfg["alpha"],
        skip_ref=cfg["skip_ref"],
    )
