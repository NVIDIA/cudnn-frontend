# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest

from test_utils import torch_fork_set_rng
from fe_api.gemm.test_gemm_amax_utils import (
    with_gemm_amax_params_fp4,
    with_gemm_amax_params_fp8,
)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_amax_params_fp4
def test_gemm_amax_compile_execute_fp4(
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
    _test_gemm_amax_compile_execute(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_amax_params_fp8
def test_gemm_amax_compile_execute_fp8(
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
    _test_gemm_amax_compile_execute(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_amax_params_fp4
def test_gemm_amax_wrapper_fp4(
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
    _test_gemm_amax_wrapper(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_amax_params_fp8
def test_gemm_amax_wrapper_fp8(
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
    _test_gemm_amax_wrapper(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        request=request,
    )


"""
GemmAmax API with explicit set_params, compile, and execute paths. 
Use this method when running one static configuration for each GemmAmax object.
"""


def _test_gemm_amax_compile_execute(
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
        from cuda.bindings import driver as cuda
        from fe_api.gemm.test_gemm_amax_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_amax,
            gemm_amax_init,
        )
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
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
    a_torch, a_ref, b_torch, b_ref, sfa_torch, sfa_ref, sfb_torch, sfb_ref = allocate_input_tensors(
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
    c_torch, amax_torch = allocate_output_tensors(cfg["m"], cfg["n"], cfg["l"], cfg["c_dtype"], cfg["c_major"])

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
    gemm.compile()
    gemm.execute(
        a_tensor=a_torch,
        b_tensor=b_torch,
        sfa_tensor=sfa_torch,
        sfb_tensor=sfb_torch,
        c_tensor=c_torch,
        amax_tensor=amax_torch,
        current_stream=stream,
    )

    check_ref_gemm_amax(a_ref, b_ref, sfa_ref, sfb_ref, c_torch, amax_torch, skip_ref=cfg["skip_ref"])


"""
GemmAmax API with gemm_amax_wrapper:
Use the wrapper to directly call GemmAmax without explicit setup and compilation.
"""


def _test_gemm_amax_wrapper(
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
        from cuda.bindings import driver as cuda
        from fe_api.gemm.test_gemm_amax_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_amax,
            gemm_amax_init,
        )
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
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
    a_torch, a_ref, b_torch, b_ref, sfa_torch, sfa_ref, sfb_torch, sfb_ref = allocate_input_tensors(
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

    try:
        for _ in range(2):  # Run twice to test caching path
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

    check_ref_gemm_amax(a_ref, b_ref, sfa_ref, sfb_ref, c_torch, amax_torch, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
def test_gemm_amax_rejects_noncontiguous_scale_factors():
    """SF tensors are consumed by base pointer only (the kernel rebuilds the layout from
    the GEMM shapes), so a shape-matching but differently-strided tensor must be rejected
    rather than silently producing wrong results."""
    try:
        from cudnn import gemm_amax_wrapper_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires SM100+")

    m, n, k, sf_vec_size = 512, 256, 256, 32
    a = torch.randn(m, k, 1, device="cuda").to(torch.float8_e5m2)
    b = torch.randn(n, k, 1, device="cuda").to(torch.float8_e5m2)
    sf_dtype = torch.float8_e8m0fnu
    sfa = torch.ones(1, m // 128, k // (4 * sf_vec_size), 32, 4, 4, device="cuda", dtype=torch.uint8).view(sf_dtype)
    sfb = torch.ones(1, n // 128, k // (4 * sf_vec_size), 32, 4, 4, device="cuda", dtype=torch.uint8).view(sf_dtype)

    # Valid: the physical form and its (3, 4, 1, 5, 2, 0)-permuted atom view
    gemm_amax_wrapper_sm100(a, b, sfa, sfb, sf_vec_size=sf_vec_size)
    gemm_amax_wrapper_sm100(a, b, sfa.permute(3, 4, 1, 5, 2, 0), sfb.permute(3, 4, 1, 5, 2, 0), sf_vec_size=sf_vec_size)

    # Non-contiguous tensor whose shape matches the physical form
    bad_physical = torch.ones(1, k // (4 * sf_vec_size), m // 128, 32, 4, 4, device="cuda", dtype=torch.uint8).view(sf_dtype).permute(0, 2, 1, 3, 4, 5)
    assert tuple(bad_physical.shape) == tuple(sfa.shape) and not bad_physical.is_contiguous()
    with pytest.raises(ValueError, match="stride"):
        gemm_amax_wrapper_sm100(a, b, bad_physical, sfb, sf_vec_size=sf_vec_size)

    # C-contiguous allocation in the atom-view shape (not a permutation of the physical form)
    bad_atom = torch.ones(32, 4, m // 128, 4, k // (4 * sf_vec_size), 1, device="cuda", dtype=torch.uint8).view(sf_dtype)
    with pytest.raises(ValueError, match="stride"):
        gemm_amax_wrapper_sm100(a, b, bad_atom, sfb, sf_vec_size=sf_vec_size)
