# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest
from test_utils import torch_fork_set_rng
from gemm.cutedsl.test_gemm_glu_utils import (
    allocate_input_tensors,
    allocate_output_tensors,
    check_ref_gemm_glu,
    with_gemm_glu_params,
    gemm_glu_init,
    with_gemm_glu_quant_params_fp4,
    with_gemm_glu_quant_params_fp8,
    check_ref_gemm_glu_quant,
    run_gemm_glu_ref,
)

"""
GemmGlu plan with explicit set_params, compile, and execute paths. 
Use this method when running one static configuration for each GemmGlu object.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_glu_params
def test_gemm_glu_compile_execute(
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
        from cudnn.gemm.cutedsl.dense.glu.api import GemmGlu
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = gemm_glu_init(
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

    gemm_glu = GemmGlu(
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
        assert gemm_glu.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    gemm_glu.compile()
    gemm_glu.execute(
        a_tensor=a_torch,
        b_tensor=b_torch,
        ab12_tensor=ab12_torch,
        c_tensor=c_torch,
        alpha=cfg["alpha"],
        current_stream=stream,
    )

    check_ref_gemm_glu(
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
def test_gemm_glu_retained_outputs_replay(dtype, tile):
    """Retain every launch's outputs so a later correct store cannot hide a race."""
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100 or newer")
    from cudnn.gemm.cutedsl.dense.glu.api import GemmGlu
    from cuda.bindings import driver as cuda

    persistent = tile[0] == 128
    m = n = 2048 if persistent else 256
    cluster = (1, 1) if persistent else (2, 2)
    a, _, b, *_ = allocate_input_tensors(m, n, 512, 2, dtype, "m", "n" if dtype == torch.float8_e4m3fn else "k")
    ab12, c, *_ = allocate_output_tensors(m, n, 2, torch.float32, torch.bfloat16, "n")
    plan = GemmGlu(
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
        pytest.skip("GemmGlu is unsupported")
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
            ref_ab12, ref_c = run_gemm_glu_ref(a.float(), b.float(), 1.0)
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
gemm_glu_forward:
Call the op directly; it caches the compiled plan per configuration.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_glu_params
def test_gemm_glu_forward(
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
        from cudnn.ops import gemm_glu_forward
        from cuda.bindings import driver as cuda
    except ImportError as e:
        print(f"ImportError: {e}")
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = gemm_glu_init(
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
            ab12_torch, c_torch, sfc_tensor, amax_tensor = gemm_glu_forward(
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

    check_ref_gemm_glu(
        a_torch,
        b_torch,
        ab12_torch,
        c_torch,
        alpha=cfg["alpha"],
        skip_ref=cfg["skip_ref"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_glu_quant_params_fp4
def test_gemm_glu_compile_execute_quant_fp4(
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
    _test_gemm_glu_compile_execute_quant(
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
@with_gemm_glu_quant_params_fp8
def test_gemm_glu_compile_execute_quant_fp8(
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
    _test_gemm_glu_compile_execute_quant(
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
@with_gemm_glu_quant_params_fp4
def test_gemm_glu_forward_quant_fp4(
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
    _test_gemm_glu_forward_quant(
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
@with_gemm_glu_quant_params_fp8
def test_gemm_glu_forward_quant_fp8(
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
    _test_gemm_glu_forward_quant(
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


def _test_gemm_glu_compile_execute_quant(
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
        from cudnn.gemm.cutedsl.dense.glu.api import GemmGlu
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = gemm_glu_init(
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

    gemm_glu = GemmGlu(
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
        assert gemm_glu.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    gemm_glu.compile()
    gemm_glu.execute(
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

    check_ref_gemm_glu_quant(
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


def _test_gemm_glu_forward_quant(
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
        from cudnn.ops import gemm_glu_forward
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = gemm_glu_init(
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
            ab12_torch, c_torch, sfc_tensor, amax_tensor = gemm_glu_forward(
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

    check_ref_gemm_glu_quant(
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


"""
GeGLU: act_func="geglu" through cudnn.ops.gemm_glu_forward.
"""

# The second variant reuses the first one's compiled plan: these are runtime parameters.
GEGLU_VARIANTS = [
    dict(),
    dict(linear_offset=0.25, geglu_alpha=1.0, glu_clamp_max=1.5, glu_clamp_min=-1.0),
]


def skip_unless_tcgen05():
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100 or newer")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("ab_dtype", [torch.bfloat16, torch.float32, torch.float8_e4m3fn], ids=["bf16", "fp32", "fp8"])
@pytest.mark.parametrize("tile,cluster", [((128, 128), (1, 1)), ((256, 256), (2, 2))], ids=["1cta", "2cta"])
@pytest.mark.parametrize("c_major", ["m", "n"])
def test_gemm_glu_forward_geglu(ab_dtype, tile, cluster, c_major):
    skip_unless_tcgen05()
    from cudnn.ops import gemm_glu_forward
    from cudnn.gemm.cutedsl.dense.glu import api

    # alpha keeps the pre-activations inside the sigmoid's active range.
    alpha = 0.05
    a, _, b, *_ = allocate_input_tensors(256, 256, 512, 2, ab_dtype, "k", "k")
    plans = None
    for glu in GEGLU_VARIANTS:
        ab12, c, sfc, amax = gemm_glu_forward(
            a,
            b,
            alpha=alpha,
            c_major=c_major,
            ab12_dtype=torch.float32,
            c_dtype=torch.bfloat16,
            mma_tiler_mn=tile,
            cluster_shape_mn=cluster,
            act_func="geglu",
            **glu,
        )
        assert sfc is None and amax is None
        check_ref_gemm_glu(a, b, ab12, c, alpha=alpha, act_func="geglu", **glu)
        assert plans in (None, len(api.glu_plan_cache))
        plans = len(api.glu_plan_cache)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype,sf_dtype,sf_vec_size,c_dtype",
    [
        (torch.float8_e4m3fn, torch.float8_e8m0fnu, 32, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float8_e8m0fnu, 32, torch.float32),
        (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, 32, torch.bfloat16),
        (torch.float4_e2m1fn_x2, torch.float8_e4m3fn, 16, torch.bfloat16),
    ],
    ids=["mxfp8-bf16", "mxfp8-fp32", "mxfp4-bf16-amax", "nvfp4-bf16-amax"],
)
@pytest.mark.parametrize("tile,cluster", [((128, 128), (1, 1)), ((256, 256), (2, 2))], ids=["1cta", "2cta"])
@pytest.mark.parametrize("vector_f32", [True, False])
def test_gemm_glu_forward_geglu_quant(ab_dtype, sf_dtype, sf_vec_size, c_dtype, tile, cluster, vector_f32):
    skip_unless_tcgen05()
    from cudnn.ops import gemm_glu_forward

    alpha = 0.02
    ab12_dtype = torch.float32 if ab_dtype == torch.float4_e2m1fn_x2 else torch.bfloat16
    a, a_ref, b, b_ref, sfa, sfa_ref, sfb, sfb_ref, norm_const = allocate_input_tensors(
        256, 256, 512, 2, ab_dtype, "k", "k", True, sf_vec_size, sf_dtype, c_dtype
    )
    for glu in GEGLU_VARIANTS:
        ab12, c, sfc, amax = gemm_glu_forward(
            a,
            b,
            alpha=alpha,
            ab12_dtype=ab12_dtype,
            c_dtype=c_dtype,
            mma_tiler_mn=tile,
            cluster_shape_mn=cluster,
            sfa_tensor=sfa,
            sfb_tensor=sfb,
            norm_const_tensor=norm_const,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            act_func="geglu",
            **glu,
        )
        check_ref_gemm_glu_quant(a, a_ref, b, b_ref, sfa_ref, sfb_ref, ab12, c, sfc, amax, norm_const, sf_vec_size, alpha=alpha, act_func="geglu", **glu)


@pytest.mark.L0
def test_gemm_glu_forward_rejects_unsupported_activation():
    skip_unless_tcgen05()
    from cudnn.ops import gemm_glu_forward

    a, _, b, *_ = allocate_input_tensors(256, 256, 512, 1, torch.float8_e4m3fn, "k", "k")
    with pytest.raises(NotImplementedError, match="situglu"):
        gemm_glu_forward(a, b, act_func="situglu")
    with pytest.raises(ValueError, match="act_func must be"):
        gemm_glu_forward(a, b, act_func="relu")
    a16, b16 = a.to(torch.float16), b.to(torch.float16)
    with pytest.raises(NotImplementedError, match="float32 acc_dtype"):
        gemm_glu_forward(a16, b16, ab12_dtype=torch.float16, acc_dtype=torch.float16, act_func="geglu")


@pytest.mark.L0
def test_dense_swiglu_only_api_is_removed():
    import importlib

    import cudnn

    for name in ("GemmSwigluSm100", "gemm_swiglu_wrapper_sm100", "gemm_swiglu_jax_sm100"):
        assert not hasattr(cudnn, name)
    with pytest.raises(ImportError):
        importlib.import_module("cudnn.gemm.cutedsl.dense.swiglu")
    assert {"gemm_glu_forward", "gemm_glu_forward_jax"} <= set(cudnn.ops.__all__)
    assert callable(cudnn.ops.gemm_glu_forward)
