# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest
from test_utils import torch_fork_set_rng
from gemm.cutedsl.test_gemm_swiglu_utils import (
    allocate_input_tensors,
    allocate_output_tensors,
    check_ref_gemm_swiglu,
    with_gemm_swiglu_params,
    gemm_swiglu_init,
    with_gemm_swiglu_quant_params_fp4,
    with_gemm_swiglu_quant_params_fp8,
    check_ref_gemm_swiglu_quant,
    run_gemm_swiglu_ref,
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
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
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
    ab12_torch, c_torch, _, _, _ = allocate_output_tensors(cfg["m"], cfg["n"], cfg["l"], cfg["ab12_dtype"], cfg["c_dtype"], cfg["c_major"])

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
        pytest.skip(f"Unsupported testcase: {e}")
    gemm_swiglu.compile()
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


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("tile", [(256, 256), (128, 128), (128, 64)], ids=["multi_group_tile", "persistent_tiles", "persistent_single_group"])
def test_gemm_swiglu_retained_outputs_replay(dtype, tile):
    """Retain every launch's outputs so a later correct store cannot hide a race."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Requires SM100 or newer")
    from cudnn import GemmSwigluSm100
    from cuda.bindings import driver as cuda

    persistent = tile[0] == 128
    m = n = 2048 if persistent else 256
    cluster = (1, 1) if persistent else (2, 2)
    a, _, b, *_ = allocate_input_tensors(m, n, 512, 2, dtype, "m", "n" if dtype == torch.float8_e4m3fn else "k")
    ab12, c, *_ = allocate_output_tensors(m, n, 2, torch.float32, torch.bfloat16, "n")
    plan = GemmSwigluSm100(
        sample_a=a,
        sample_b=b,
        sample_ab12=ab12,
        sample_c=c,
        alpha=1.0,
        acc_dtype=torch.float32,
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
    )
    try:
        supported = plan.check_support()
    except (ValueError, NotImplementedError) as error:
        pytest.skip(str(error))
    if not supported:
        pytest.skip("GemmSwigluSm100 is unsupported")
    plan.compile()
    slots = 4 if persistent else 16
    outputs = [(ab12, c)] + [
        (
            torch.empty_strided(ab12.shape, ab12.stride(), dtype=ab12.dtype, device=ab12.device),
            torch.empty_strided(c.shape, c.stride(), dtype=c.dtype, device=c.device),
        )
        for _ in range(slots - 1)
    ]

    def run():
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        for intermediate, result in outputs:
            plan.execute(a_tensor=a, b_tensor=b, ab12_tensor=intermediate, c_tensor=result, alpha=1.0, current_stream=stream)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run()
        for phase in range(2):
            if phase:
                a.copy_((a.float() * 0.5).to(dtype))
            ref_ab12, ref_c = run_gemm_swiglu_ref(a.float(), b.float(), 1.0)
            ref_ab12, ref_c = ref_ab12.to(ab12.dtype), ref_c.to(c.dtype)
            for intermediate, result in outputs:
                intermediate.fill_(float("nan"))
                result.fill_(float("nan"))
            for _ in range(8):
                graph.replay()
                for intermediate, result in outputs:
                    torch.testing.assert_close(intermediate.cpu(), ref_ab12, atol=0.01, rtol=9e-3)
                    torch.testing.assert_close(result.cpu(), ref_c, atol=0.01, rtol=9e-3)
    finally:
        graph.reset()


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
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
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
                stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    assert sfc_tensor is None
    assert amax_tensor is None

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
@with_gemm_swiglu_quant_params_fp4
def test_gemm_swiglu_compile_execute_quant_fp4(
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
    _test_gemm_swiglu_compile_execute_quant(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        ab12_dtype=ab12_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_swiglu_quant_params_fp8
def test_gemm_swiglu_compile_execute_quant_fp8(
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
    _test_gemm_swiglu_compile_execute_quant(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        ab12_dtype=ab12_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_swiglu_quant_params_fp4
def test_gemm_swiglu_wrapper_quant_fp4(
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
    _test_gemm_swiglu_wrapper_quant(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        ab12_dtype=ab12_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_swiglu_quant_params_fp8
def test_gemm_swiglu_wrapper_quant_fp8(
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
    _test_gemm_swiglu_wrapper_quant(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        ab12_dtype=ab12_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        request=request,
    )


def _test_gemm_swiglu_compile_execute_quant(
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
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
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
    gemm_swiglu.compile()
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


def _test_gemm_swiglu_wrapper_quant(
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
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
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
