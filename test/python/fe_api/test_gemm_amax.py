import torch

import pytest

from test_utils import torch_fork_set_rng
from cuda.bindings import driver as cuda
from fe_api.test_gemm_amax_utils import (
    with_gemm_amax_params,
)

"""
GemmAmax API with explicit set_params, compile, and execute paths. 
Use this method when running one static configuration for each GemmAmax object.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_amax_params
def test_gemm_amax_compile_execute(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    sf_dtype,
    c_dtype,
    acc_dtype,
    sf_vec_size,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    try:
        from cudnn import GemmAmaxSm100
        from fe_api.test_gemm_amax_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_amax,
            gemm_amax_init,
        )
    except ImportError as e:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )
    cfg = gemm_amax_init(
        request,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        sf_dtype,
        c_dtype,
        acc_dtype,
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, a_ref, b_torch, b_ref, sfa_torch, sfa_ref, sfb_torch, sfb_ref = (
        allocate_input_tensors(
            cfg["m"],
            cfg["n"],
            cfg["k"],
            cfg["l"],
            cfg["ab_dtype"],
            cfg["sf_dtype"],
            cfg["sf_vec_size"],
            cfg["a_major"],
            cfg["b_major"],
        )
    )
    c_torch, amax_torch = allocate_output_tensors(
        cfg["m"], cfg["n"], cfg["l"], cfg["c_dtype"], cfg["c_major"]
    )

    gemm = GemmAmaxSm100(
        sample_a=a_torch,
        sample_b=b_torch,
        sample_sfa=sfa_torch,
        sample_sfb=sfb_torch,
        sample_c=c_torch,
        sample_amax=amax_torch,
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
    )
    try:
        assert gemm.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    gemm.compile(current_stream=stream)
    gemm.execute(
        a_tensor=a_torch,
        b_tensor=b_torch,
        sfa_tensor=sfa_torch,
        sfb_tensor=sfb_torch,
        c_tensor=c_torch,
        amax_tensor=amax_torch,
        current_stream=stream,
    )

    check_ref_gemm_amax(
        a_ref, b_ref, sfa_ref, sfb_ref, c_torch, amax_torch, skip_ref=cfg["skip_ref"]
    )


"""
GemmAmax API with gemm_amax_wrapper:
Use the wrapper to directly call GemmAmax without explicit setup and compilation.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_amax_params
def test_gemm_amax_wrapper(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    sf_dtype,
    c_dtype,
    acc_dtype,
    sf_vec_size,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    try:
        from cudnn import gemm_amax_wrapper_sm100
        from fe_api.test_gemm_amax_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_amax,
            gemm_amax_init,
        )
    except ImportError as e:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )
    cfg = gemm_amax_init(
        request,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        sf_dtype,
        c_dtype,
        acc_dtype,
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, a_ref, b_torch, b_ref, sfa_torch, sfa_ref, sfb_torch, sfb_ref = (
        allocate_input_tensors(
            cfg["m"],
            cfg["n"],
            cfg["k"],
            cfg["l"],
            cfg["ab_dtype"],
            cfg["sf_dtype"],
            cfg["sf_vec_size"],
            cfg["a_major"],
            cfg["b_major"],
        )
    )

    try:
        c_torch, amax_torch = gemm_amax_wrapper_sm100(
            a_tensor=a_torch,
            b_tensor=b_torch,
            sfa_tensor=sfa_torch,
            sfb_tensor=sfb_torch,
            c_major=cfg["c_major"],
            c_dtype=cfg["c_dtype"],
            acc_dtype=cfg["acc_dtype"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            stream=stream,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    check_ref_gemm_amax(
        a_ref, b_ref, sfa_ref, sfb_ref, c_torch, amax_torch, skip_ref=cfg["skip_ref"]
    )
