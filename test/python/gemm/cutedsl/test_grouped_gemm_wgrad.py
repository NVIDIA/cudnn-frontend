# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for grouped GEMM wgrad FE API."""

import ast
import inspect
from pathlib import Path

import pytest
import torch
import cudnn

from test_utils import torch_fork_set_rng
from core.cutedsl.test_fe_api_utils import reencode_sf_tensor_as_ue5m3
from gemm.cutedsl.test_grouped_gemm_wgrad_utils import (
    _skip_unless_e5m3_supported,
    grouped_gemm_wgrad_init,
    with_grouped_gemm_wgrad_params_fp4,
    with_grouped_gemm_wgrad_params_fp8,
    allocate_grouped_gemm_wgrad_tensors,
    allocate_grouped_gemm_wgrad_output,
    check_ref_grouped_gemm_wgrad,
    wgrad_to_ragged_layout,
)
from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import (
    assert_grouped_gemm_wgrad_close as assert_grouped_gemm_wgrad_bf16_close,
    grouped_gemm_wgrad_bf16_reference,
    make_grouped_gemm_wgrad_bf16_problem,
)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("callable_obj", "legacy_positional_parameters"),
    (
        (
            cudnn.GroupedGemmWgradSm100.execute,
            (
                "self",
                "a_tensor",
                "b_tensor",
                "sfa_tensor",
                "sfb_tensor",
                "offsets_tensor",
                "wgrad_tensor",
                "wgrad_ptrs",
                "global_scale_a",
                "global_scale_b",
                "current_stream",
            ),
        ),
        (
            cudnn.grouped_gemm_wgrad_wrapper_sm100,
            (
                "a_tensor",
                "b_tensor",
                "sfa_tensor",
                "sfb_tensor",
                "offsets_tensor",
                "output_mode",
                "wgrad_tensor",
                "wgrad_ptrs",
                "global_scale_a",
                "global_scale_b",
                "acc_dtype",
                "wgrad_dtype",
                "mma_tiler_mn",
                "cluster_shape_mn",
                "sf_vec_size",
                "sf_fp8_dtype_override",
                "accumulate_on_output",
                "input_order",
                "current_stream",
            ),
        ),
    ),
    ids=("facade-execute", "wrapper"),
)
def test_grouped_gemm_wgrad_preserves_legacy_positional_prefix(callable_obj, legacy_positional_parameters):
    signature = inspect.signature(callable_obj)
    parameters = signature.parameters
    positional_parameters = tuple(
        name for name, parameter in parameters.items() if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    assert positional_parameters == legacy_positional_parameters

    descriptor_workspace = parameters["descriptor_workspace"]
    assert descriptor_workspace.kind is inspect.Parameter.KEYWORD_ONLY
    assert descriptor_workspace.default is None

    sentinels = [object() for _ in legacy_positional_parameters]
    bound = signature.bind(*sentinels)
    for name, sentinel in zip(legacy_positional_parameters, sentinels):
        assert bound.arguments[name] is sentinel
    assert "descriptor_workspace" not in bound.arguments

    workspace = object()
    bound_with_workspace = signature.bind(*sentinels, descriptor_workspace=workspace)
    assert bound_with_workspace.arguments["descriptor_workspace"] is workspace
    with pytest.raises(TypeError):
        signature.bind(*(sentinels + [workspace]))


# ---------------------------------------------------------------------------
# Dense mode: Class API
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["bf16-dense", "bf16-discrete"])
def test_grouped_gemm_wgrad_wrapper_bf16(discrete):
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM WGrad BF16 kernel.")

    problem = make_grouped_gemm_wgrad_bf16_problem(discrete=discrete)
    expected = grouped_gemm_wgrad_bf16_reference(problem)
    kwargs = dict(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=problem["offsets"],
        output_mode="discrete" if discrete else "dense",
        wgrad_tensor=problem["output"],
        wgrad_ptrs=problem["output_ptrs"],
        acc_dtype=torch.float32,
        wgrad_dtype=problem["output_dtype"],
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        input_order=problem["input_order"],
    )
    result = cudnn.grouped_gemm_wgrad_wrapper_sm100(**kwargs)
    assert result["wgrad_tensor"] is problem["output"]
    assert_grouped_gemm_wgrad_bf16_close(result["wgrad_tensor"], expected)


def _test_grouped_gemm_wgrad_dense_compile_execute(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    sf_fp8_dtype_override=None,
):
    cfg = grouped_gemm_wgrad_init(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
    )
    inputs = allocate_grouped_gemm_wgrad_tensors(cfg)

    if sf_fp8_dtype_override == "e5m3":
        # Rewrite the scale bytes as UE5M3 in place; values are exact in both
        # formats so inputs["ref_result"] stays valid.
        reencode_sf_tensor_as_ue5m3(inputs["sfa_tensor"])
        reencode_sf_tensor_as_ue5m3(inputs["sfb_tensor"])
    wgrad_tensor = allocate_grouped_gemm_wgrad_output(cfg)

    op = cudnn.GroupedGemmWgradSm100(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_offsets=inputs["offsets_tensor"],
        sample_wgrad=wgrad_tensor,
        sample_global_scale_a=inputs["global_scale_a"],
        sample_global_scale_b=inputs["global_scale_b"],
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        sf_fp8_dtype_override=sf_fp8_dtype_override,
    )
    try:
        assert op.check_support()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    op.compile()
    op.execute(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        offsets_tensor=inputs["offsets_tensor"],
        wgrad_tensor=wgrad_tensor,
        global_scale_a=inputs["global_scale_a"],
        global_scale_b=inputs["global_scale_b"],
    )
    torch.cuda.synchronize()
    check_ref_grouped_gemm_wgrad(wgrad_tensor, inputs["ref_result"], cfg["tolerance"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_wgrad_params_fp4
def test_grouped_gemm_wgrad_dense_compile_execute_fp4(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    sf_fp8_dtype_override,
):
    _test_grouped_gemm_wgrad_dense_compile_execute(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        sf_fp8_dtype_override=sf_fp8_dtype_override,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_wgrad_params_fp8
def test_grouped_gemm_wgrad_dense_compile_execute_fp8(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
):
    _test_grouped_gemm_wgrad_dense_compile_execute(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
    )


# ---------------------------------------------------------------------------
# Dense mode: Wrapper API
# ---------------------------------------------------------------------------


def _test_grouped_gemm_wgrad_dense_wrapper(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    sf_fp8_dtype_override=None,
):
    cfg = grouped_gemm_wgrad_init(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
    )
    inputs = allocate_grouped_gemm_wgrad_tensors(cfg)

    if sf_fp8_dtype_override == "e5m3":
        # Rewrite the scale bytes as UE5M3 in place; values are exact in both
        # formats so inputs["ref_result"] stays valid.
        reencode_sf_tensor_as_ue5m3(inputs["sfa_tensor"])
        reencode_sf_tensor_as_ue5m3(inputs["sfb_tensor"])
    try:
        for _ in range(2):  # Run twice to test caching path
            result = cudnn.grouped_gemm_wgrad_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                offsets_tensor=inputs["offsets_tensor"],
                output_mode="dense",
                global_scale_a=inputs["global_scale_a"],
                global_scale_b=inputs["global_scale_b"],
                acc_dtype=cfg["acc_dtype"],
                wgrad_dtype=cfg["wgrad_dtype"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                sf_vec_size=cfg["sf_vec_size"],
                sf_fp8_dtype_override=sf_fp8_dtype_override,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()
    check_ref_grouped_gemm_wgrad(result["wgrad_tensor"], inputs["ref_result"], cfg["tolerance"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_wgrad_params_fp4
def test_grouped_gemm_wgrad_dense_wrapper_fp4(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    sf_fp8_dtype_override,
):
    _test_grouped_gemm_wgrad_dense_wrapper(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        sf_fp8_dtype_override=sf_fp8_dtype_override,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_wgrad_params_fp8
def test_grouped_gemm_wgrad_dense_wrapper_fp8(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
):
    _test_grouped_gemm_wgrad_dense_wrapper(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
    )


# ---------------------------------------------------------------------------
# Discrete mode: Class API
# ---------------------------------------------------------------------------


def _test_grouped_gemm_wgrad_discrete_compile_execute(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    accumulate_on_output=False,
    sf_fp8_dtype_override=None,
):
    cfg = grouped_gemm_wgrad_init(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
    )
    inputs = allocate_grouped_gemm_wgrad_tensors(cfg)

    if sf_fp8_dtype_override == "e5m3":
        # Rewrite the scale bytes as UE5M3 in place; values are exact in both
        # formats so inputs["ref_result"] stays valid.
        reencode_sf_tensor_as_ue5m3(inputs["sfa_tensor"])
        reencode_sf_tensor_as_ue5m3(inputs["sfb_tensor"])
    wgrad_tensor = allocate_grouped_gemm_wgrad_output(cfg, accumulate_on_output=accumulate_on_output)
    expected = inputs["ref_result"]
    if accumulate_on_output:
        wgrad_tensor.fill_(1)
        if expected is not None:
            expected = expected + 1

    op = cudnn.GroupedGemmWgradSm100(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_offsets=inputs["offsets_tensor"],
        sample_wgrad_expert=None,
        num_experts=cfg["l"],
        wgrad_shape=(cfg["m"], cfg["n"]),
        wgrad_dtype=cfg["wgrad_dtype"],
        sample_global_scale_a=inputs["global_scale_a"],
        sample_global_scale_b=inputs["global_scale_b"],
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        sf_fp8_dtype_override=sf_fp8_dtype_override,
        accumulate_on_output=accumulate_on_output,
    )
    try:
        assert op.check_support()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    op.compile()
    op.execute(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        offsets_tensor=inputs["offsets_tensor"],
        wgrad_tensor=wgrad_tensor,
        global_scale_a=inputs["global_scale_a"],
        global_scale_b=inputs["global_scale_b"],
    )
    torch.cuda.synchronize()
    check_ref_grouped_gemm_wgrad(wgrad_tensor, expected, cfg["tolerance"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_wgrad_params_fp4
def test_grouped_gemm_wgrad_discrete_compile_execute_fp4(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    sf_fp8_dtype_override,
):
    _test_grouped_gemm_wgrad_discrete_compile_execute(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        sf_fp8_dtype_override=sf_fp8_dtype_override,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_wgrad_params_fp8
def test_grouped_gemm_wgrad_discrete_compile_execute_fp8(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
):
    _test_grouped_gemm_wgrad_discrete_compile_execute(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_wgrad_params_fp4
def test_grouped_gemm_wgrad_discrete_accumulate_compile_execute_fp4(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    sf_fp8_dtype_override,
):
    _test_grouped_gemm_wgrad_discrete_compile_execute(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        sf_fp8_dtype_override=sf_fp8_dtype_override,
        accumulate_on_output=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_wgrad_params_fp8
def test_grouped_gemm_wgrad_discrete_accumulate_compile_execute_fp8(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
):
    _test_grouped_gemm_wgrad_discrete_compile_execute(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        accumulate_on_output=True,
    )


# ---------------------------------------------------------------------------
# Discrete mode: Wrapper API
# ---------------------------------------------------------------------------


def _test_grouped_gemm_wgrad_discrete_wrapper(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    sf_fp8_dtype_override=None,
):
    cfg = grouped_gemm_wgrad_init(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
    )
    inputs = allocate_grouped_gemm_wgrad_tensors(cfg)

    if sf_fp8_dtype_override == "e5m3":
        # Rewrite the scale bytes as UE5M3 in place; values are exact in both
        # formats so inputs["ref_result"] stays valid.
        reencode_sf_tensor_as_ue5m3(inputs["sfa_tensor"])
        reencode_sf_tensor_as_ue5m3(inputs["sfb_tensor"])
    try:
        for _ in range(2):  # Run twice to test caching path
            result = cudnn.grouped_gemm_wgrad_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                offsets_tensor=inputs["offsets_tensor"],
                output_mode="discrete",
                global_scale_a=inputs["global_scale_a"],
                global_scale_b=inputs["global_scale_b"],
                acc_dtype=cfg["acc_dtype"],
                wgrad_dtype=cfg["wgrad_dtype"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                sf_vec_size=cfg["sf_vec_size"],
                sf_fp8_dtype_override=sf_fp8_dtype_override,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()
    check_ref_grouped_gemm_wgrad(result["wgrad_tensor"], inputs["ref_result"], cfg["tolerance"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_wgrad_params_fp4
def test_grouped_gemm_wgrad_discrete_wrapper_fp4(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    sf_fp8_dtype_override,
):
    _test_grouped_gemm_wgrad_discrete_wrapper(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        sf_fp8_dtype_override=sf_fp8_dtype_override,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_wgrad_params_fp8
def test_grouped_gemm_wgrad_discrete_wrapper_fp8(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
):
    _test_grouped_gemm_wgrad_discrete_wrapper(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
    )


def _cfg_with_group_k_list(cfg, group_k_list):
    updated_cfg = dict(cfg)
    updated_cfg["l"] = len(group_k_list)
    updated_cfg["group_k_list"] = list(group_k_list)
    return updated_cfg


def _test_grouped_gemm_wgrad_dynamic_tokens_compile_execute(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    output_mode,
):
    compile_cfg = grouped_gemm_wgrad_init(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
    )
    runtime_group_k_list = [128, 128]
    if torch.cuda.get_device_capability() == (10, 7) and ab_dtype == torch.float4_e2m1fn_x2:
        runtime_group_k_list = [256, 256]
    runtime_cfg = _cfg_with_group_k_list(compile_cfg, runtime_group_k_list)

    compile_inputs = allocate_grouped_gemm_wgrad_tensors(compile_cfg)
    runtime_inputs = allocate_grouped_gemm_wgrad_tensors(runtime_cfg)
    runtime_wgrad = allocate_grouped_gemm_wgrad_output(runtime_cfg)

    if output_mode == "dense":
        op = cudnn.GroupedGemmWgradSm100(
            sample_a=compile_inputs["a_tensor"],
            sample_b=compile_inputs["b_tensor"],
            sample_sfa=compile_inputs["sfa_tensor"],
            sample_sfb=compile_inputs["sfb_tensor"],
            sample_offsets=compile_inputs["offsets_tensor"],
            sample_wgrad=allocate_grouped_gemm_wgrad_output(compile_cfg),
            sample_global_scale_a=compile_inputs["global_scale_a"],
            sample_global_scale_b=compile_inputs["global_scale_b"],
            acc_dtype=compile_cfg["acc_dtype"],
            mma_tiler_mn=compile_cfg["mma_tiler_mn"],
            cluster_shape_mn=compile_cfg["cluster_shape_mn"],
            sf_vec_size=compile_cfg["sf_vec_size"],
        )
    else:
        op = cudnn.GroupedGemmWgradSm100(
            sample_a=compile_inputs["a_tensor"],
            sample_b=compile_inputs["b_tensor"],
            sample_sfa=compile_inputs["sfa_tensor"],
            sample_sfb=compile_inputs["sfb_tensor"],
            sample_offsets=compile_inputs["offsets_tensor"],
            sample_wgrad_expert=None,
            num_experts=compile_cfg["l"],
            wgrad_shape=(compile_cfg["m"], compile_cfg["n"]),
            wgrad_dtype=compile_cfg["wgrad_dtype"],
            sample_global_scale_a=compile_inputs["global_scale_a"],
            sample_global_scale_b=compile_inputs["global_scale_b"],
            acc_dtype=compile_cfg["acc_dtype"],
            mma_tiler_mn=compile_cfg["mma_tiler_mn"],
            cluster_shape_mn=compile_cfg["cluster_shape_mn"],
            sf_vec_size=compile_cfg["sf_vec_size"],
        )

    try:
        assert op.check_support()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    op.compile()
    op.execute(
        a_tensor=runtime_inputs["a_tensor"],
        b_tensor=runtime_inputs["b_tensor"],
        sfa_tensor=runtime_inputs["sfa_tensor"],
        sfb_tensor=runtime_inputs["sfb_tensor"],
        offsets_tensor=runtime_inputs["offsets_tensor"],
        wgrad_tensor=runtime_wgrad,
        global_scale_a=runtime_inputs["global_scale_a"],
        global_scale_b=runtime_inputs["global_scale_b"],
    )
    torch.cuda.synchronize()
    check_ref_grouped_gemm_wgrad(runtime_wgrad, runtime_inputs["ref_result"], runtime_cfg["tolerance"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("output_mode", ["dense", "discrete"])
@with_grouped_gemm_wgrad_params_fp4
def test_grouped_gemm_wgrad_dynamic_tokens_compile_execute_fp4(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    output_mode,
    sf_fp8_dtype_override,  # noqa: ARG001
):
    if sf_fp8_dtype_override is not None:
        pytest.skip("Skip e5m3 test. This test is not for numerical correctness and covering e5m3's gain is marginal.")

    _test_grouped_gemm_wgrad_dynamic_tokens_compile_execute(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        output_mode=output_mode,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("output_mode", ["dense", "discrete"])
@with_grouped_gemm_wgrad_params_fp8
def test_grouped_gemm_wgrad_dynamic_tokens_compile_execute_fp8(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    output_mode,
):
    _test_grouped_gemm_wgrad_dynamic_tokens_compile_execute(
        ab_dtype=ab_dtype,
        wgrad_dtype=wgrad_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        output_mode=output_mode,
    )


def _make_wgrad_wrapper_cache_inputs(group_k_list, sf_vec_size=16):
    hidden = 32
    intermediate = 64
    expert_cnt = len(group_k_list)
    tokens_sum = sum(group_k_list)

    scale_cols = ((tokens_sum + sf_vec_size - 1) // sf_vec_size + 3) // 4 * 4

    return {
        "a_tensor": torch.empty((hidden, tokens_sum), dtype=torch.bfloat16),
        "b_tensor": torch.empty_strided((tokens_sum, intermediate), (1, tokens_sum), dtype=torch.bfloat16),
        "sfa_tensor": torch.empty((128, scale_cols), dtype=torch.bfloat16),
        "sfb_tensor": torch.empty((128, scale_cols), dtype=torch.bfloat16),
        "offsets_tensor": torch.tensor([sum(group_k_list[: i + 1]) for i in range(expert_cnt)], dtype=torch.int32),
    }


@pytest.mark.L0
@pytest.mark.parametrize("output_mode", ["dense", "discrete"])
def test_grouped_gemm_wgrad_wrapper_dynamic_tokens_cache_behavior(monkeypatch, output_mode):
    from cudnn.gemm.cutedsl.grouped.wgrad import api as grouped_gemm_wgrad_api

    # The wrapper memo is consulted before the backend cache, so a prior test's
    # entry could bypass compile(); give this test an empty one.
    monkeypatch.setattr(grouped_gemm_wgrad_api, "_wgrad_wrapper_memo", {})
    grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()

    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "check_support", lambda self: True)
    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "compile", counted_compile)
    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "execute", lambda self, **kwargs: None)
    monkeypatch.setattr(
        grouped_gemm_wgrad_api,
        "select_grouped_gemm_backend",
        lambda **_: grouped_gemm_wgrad_api.GroupedGemmBackend.BLOCK_SCALED,
    )

    first_inputs = _make_wgrad_wrapper_cache_inputs([8, 12])
    second_inputs = _make_wgrad_wrapper_cache_inputs([80, 80])

    try:
        cudnn.grouped_gemm_wgrad_wrapper_sm100(
            **first_inputs,
            output_mode=output_mode,
            acc_dtype=torch.float32,
            wgrad_dtype=torch.bfloat16,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            sf_vec_size=16,
        )
        cudnn.grouped_gemm_wgrad_wrapper_sm100(
            **second_inputs,
            output_mode=output_mode,
            acc_dtype=torch.float32,
            wgrad_dtype=torch.bfloat16,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            sf_vec_size=16,
        )
    finally:
        cache_entries = len(grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects)
        grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()

    assert compile_count["value"] == 1
    assert cache_entries == 1


@pytest.mark.L0
@pytest.mark.parametrize("caller_owned_workspace", [False, True], ids=["internal-workspace", "caller-workspace"])
def test_grouped_gemm_wgrad_wrapper_explicit_dense_output_cache(
    monkeypatch,
    caller_owned_workspace,
):
    """Distinct dense outputs share one compiled kernel, with or without a caller workspace."""
    from cudnn.gemm.cutedsl.grouped.wgrad import api as grouped_gemm_wgrad_api

    # The wrapper memo is consulted before the backend cache, so a prior test's
    # entry could bypass compile(); give this test an empty one.
    monkeypatch.setattr(grouped_gemm_wgrad_api, "_wgrad_wrapper_memo", {})
    grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()
    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "check_support", lambda self: True)
    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "compile", counted_compile)
    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "execute", lambda self, **kwargs: None)
    monkeypatch.setattr(
        grouped_gemm_wgrad_api,
        "select_grouped_gemm_backend",
        lambda **_: grouped_gemm_wgrad_api.GroupedGemmBackend.BLOCK_SCALED,
    )

    inputs = _make_wgrad_wrapper_cache_inputs([8, 12])
    outputs = [torch.empty((2, 32, 64), dtype=torch.bfloat16) for _ in range(2)]
    workspaces = [torch.empty(512, dtype=torch.uint8) for _ in range(2)]
    try:
        for output, workspace in zip(outputs, workspaces):
            workspace_kwargs = {"descriptor_workspace": workspace} if caller_owned_workspace else {}
            cudnn.grouped_gemm_wgrad_wrapper_sm100(
                **inputs,
                **workspace_kwargs,
                output_mode="dense",
                wgrad_tensor=output,
                acc_dtype=torch.float32,
                wgrad_dtype=torch.bfloat16,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                sf_vec_size=16,
            )
    finally:
        cache_entries = len(grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects)
        grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()

    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert compile_count["value"] == 1
    assert cache_entries == 1


@pytest.mark.L0
def test_grouped_gemm_wgrad_wrapper_discrete_accepts_caller_workspace(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.wgrad import api as grouped_gemm_wgrad_api

    grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()
    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(
        grouped_gemm_wgrad_api.GroupedGemmWgradSm100,
        "check_support",
        lambda self: True,
    )
    monkeypatch.setattr(
        grouped_gemm_wgrad_api.GroupedGemmWgradSm100,
        "compile",
        counted_compile,
    )
    monkeypatch.setattr(
        grouped_gemm_wgrad_api.GroupedGemmWgradSm100,
        "execute",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        grouped_gemm_wgrad_api,
        "select_grouped_gemm_backend",
        lambda **_: grouped_gemm_wgrad_api.GroupedGemmBackend.BLOCK_SCALED,
    )

    inputs = _make_wgrad_wrapper_cache_inputs([8, 12])
    outputs = [torch.empty((2, 32, 64), dtype=torch.bfloat16) for _ in range(2)]
    workspaces = [torch.empty(512, dtype=torch.uint8) for _ in range(2)]
    try:
        for output, workspace in zip(outputs, workspaces):
            cudnn.grouped_gemm_wgrad_wrapper_sm100(
                **inputs,
                output_mode="discrete",
                wgrad_tensor=output,
                descriptor_workspace=workspace,
                acc_dtype=torch.float32,
                wgrad_dtype=torch.bfloat16,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                sf_vec_size=16,
            )
    finally:
        cache_entries = len(grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects)
        grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()

    assert compile_count["value"] == 1
    assert cache_entries == 1


@pytest.mark.L0
def test_grouped_gemm_wgrad_workspace_size():
    assert cudnn.get_grouped_gemm_wgrad_workspace_size_sm100(2) == 1024
    assert (
        cudnn.get_grouped_gemm_wgrad_workspace_size_sm100(
            2,
            input_order="tensor_ragged",
        )
        == 1024
    )


@pytest.mark.L0
def test_blockscaled_wgrad_execute_uses_caller_workspace(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.wgrad import _blockscaled_api

    api = object.__new__(_blockscaled_api.GroupedGemmWgradBlockScaledAPI)
    api._workspace_bytes = 16
    api._workspace = torch.empty(16, dtype=torch.uint8)
    api.a_desc = type("TensorDesc", (), {"device": torch.device("cpu")})()
    api.weight_mode = _blockscaled_api.MoEWeightMode.DENSE
    api._get_default_stream = lambda stream: stream
    api._runtime_error_if = lambda condition, message: None
    api._value_error_if = lambda condition, message: None
    monkeypatch.setattr(
        _blockscaled_api,
        "from_dlpack",
        lambda tensor, **kwargs: tensor,
    )

    launch_workspaces = []

    def compiled_kernel(*args):
        launch_workspaces.append(args[6])

    api._compiled_kernel = compiled_kernel
    operand = torch.empty((1, 1))
    offsets = torch.tensor([1], dtype=torch.int32)
    outputs = [torch.empty((2, 32, 64), dtype=torch.bfloat16) for _ in range(2)]
    workspaces = [torch.empty(16, dtype=torch.uint8) for _ in range(2)]
    for output, workspace in (
        (outputs[0], workspaces[0]),
        (outputs[0], workspaces[0]),
        (outputs[1], workspaces[1]),
    ):
        api.execute(
            operand,
            operand,
            operand,
            operand,
            offsets,
            wgrad_tensor=output,
            descriptor_workspace=workspace,
            current_stream=object(),
        )

    assert launch_workspaces[0] is workspaces[0]
    assert launch_workspaces[1] is workspaces[0]
    assert launch_workspaces[2] is workspaces[1]
    assert launch_workspaces[0].data_ptr() != launch_workspaces[2].data_ptr()


@pytest.mark.L0
def test_blockscaled_discrete_wgrad_execute_uses_caller_workspace(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.wgrad import _blockscaled_api

    api = object.__new__(_blockscaled_api.GroupedGemmWgradBlockScaledAPI)
    api._workspace_bytes = 16
    api._workspace_arg = torch.empty(16, dtype=torch.uint8)
    api.a_desc = type("TensorDesc", (), {"device": torch.device("cpu")})()
    api.weight_mode = _blockscaled_api.MoEWeightMode.DISCRETE
    api.expert_cnt = 2
    api._get_default_stream = lambda stream: stream
    api._runtime_error_if = lambda condition, message: None
    api._value_error_if = lambda condition, message: None
    monkeypatch.setattr(
        _blockscaled_api,
        "from_dlpack",
        lambda tensor, **kwargs: tensor,
    )
    monkeypatch.setattr(
        _blockscaled_api,
        "_validate_pointer_tensor",
        lambda tensor, name, count: None,
    )

    launch_workspaces = []

    def compiled_kernel(*args):
        launch_workspaces.append(args[6])

    api._compiled_kernel = compiled_kernel
    operand = torch.empty((1, 1))
    offsets = torch.tensor([1, 2], dtype=torch.int32)
    wgrad_ptrs = torch.empty(2, dtype=torch.int64)
    workspaces = [torch.empty(16, dtype=torch.uint8) for _ in range(2)]
    for workspace in (workspaces[0], workspaces[0], workspaces[1]):
        api.execute(
            operand,
            operand,
            operand,
            operand,
            offsets,
            wgrad_ptrs=wgrad_ptrs,
            descriptor_workspace=workspace,
            current_stream=object(),
        )

    assert launch_workspaces[0] is workspaces[0]
    assert launch_workspaces[1] is workspaces[0]
    assert launch_workspaces[2] is workspaces[1]
    assert launch_workspaces[0].data_ptr() != launch_workspaces[2].data_ptr()


@pytest.mark.L0
def test_grouped_gemm_wgrad_wrapper_input_order_cache_key(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.wgrad import api as grouped_gemm_wgrad_api

    grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()

    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "check_support", lambda self: True)
    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "compile", counted_compile)
    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "execute", lambda self, **kwargs: None)
    monkeypatch.setattr(
        grouped_gemm_wgrad_api,
        "select_grouped_gemm_backend",
        lambda **_: grouped_gemm_wgrad_api.GroupedGemmBackend.BLOCK_SCALED,
    )

    inputs = _make_wgrad_wrapper_cache_inputs([8, 12])

    try:
        for input_order in ("tensor2d", "tensor_ragged"):
            cudnn.grouped_gemm_wgrad_wrapper_sm100(
                **inputs,
                output_mode="dense",
                acc_dtype=torch.float32,
                wgrad_dtype=torch.bfloat16,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                sf_vec_size=16,
                input_order=input_order,
            )
    finally:
        cache_entries = len(grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects)
        grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()

    assert compile_count["value"] == 2
    assert cache_entries == 2


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_wgrad_dense_wrapper_tensor_ragged_fp4():
    cfg = grouped_gemm_wgrad_init(
        ab_dtype=torch.float4_e2m1fn_x2,
        wgrad_dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sf_vec_size=16,
        sf_dtype=torch.float8_e4m3fn,
    )
    inputs = allocate_grouped_gemm_wgrad_tensors(cfg)
    a_ragged = wgrad_to_ragged_layout(inputs["a_tensor"], cfg["group_k_list"], k_dim=1)
    b_ragged = wgrad_to_ragged_layout(inputs["b_tensor"], cfg["group_k_list"], k_dim=0)

    result = cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=a_ragged,
        b_tensor=b_ragged,
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        offsets_tensor=inputs["offsets_tensor"],
        output_mode="dense",
        global_scale_a=inputs["global_scale_a"],
        global_scale_b=inputs["global_scale_b"],
        acc_dtype=cfg["acc_dtype"],
        wgrad_dtype=cfg["wgrad_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        input_order="tensor_ragged",
    )

    torch.cuda.synchronize()
    check_ref_grouped_gemm_wgrad(result["wgrad_tensor"], inputs["ref_result"], cfg["tolerance"])


def _wgrad_nvfp4_inputs(sf_vec_size=16, sf_dtype=torch.float8_e4m3fn, ab_dtype=torch.float4_e2m1fn_x2):
    cfg = grouped_gemm_wgrad_init(
        ab_dtype=ab_dtype,
        wgrad_dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
    )
    return cfg, allocate_grouped_gemm_wgrad_tensors(cfg)


def _run_wgrad_wrapper(cfg, inputs, sf_fp8_dtype_override):
    """Call the wrapper directly; the harnesses turn ValueError into a skip."""
    return cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        offsets_tensor=inputs["offsets_tensor"],
        output_mode="dense",
        global_scale_a=inputs["global_scale_a"],
        global_scale_b=inputs["global_scale_b"],
        acc_dtype=cfg["acc_dtype"],
        wgrad_dtype=cfg["wgrad_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        sf_fp8_dtype_override=sf_fp8_dtype_override,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "sf_fp8_dtype_override,overrides,expected",
    [
        pytest.param("e5m3", dict(sf_vec_size=32, sf_dtype=torch.float8_e8m0fnu), "requires the NVFP4 recipe", id="mxfp4_e8m0_carrier"),
        pytest.param(
            "e5m3",
            dict(ab_dtype=torch.float8_e4m3fn, sf_vec_size=32, sf_dtype=torch.float8_e8m0fnu),
            "requires the NVFP4 recipe",
            id="fp8_ab",
        ),
        pytest.param("e4m3", {}, "sf_fp8_dtype_override must be", id="e4m3_is_not_an_override"),
        pytest.param("e5m2", {}, "sf_fp8_dtype_override must be", id="unknown_format"),
    ],
)
def test_grouped_gemm_wgrad_rejects_unsupported_sf_fp8_dtype(sf_fp8_dtype_override, overrides, expected):
    """e5m3 is only reachable through the Rubin FP4xFP4 atom with e4m3-carried scales."""
    if sf_fp8_dtype_override == "e5m3":
        _skip_unless_e5m3_supported()
    cfg, inputs = _wgrad_nvfp4_inputs(**overrides)
    with pytest.raises(ValueError, match=expected):
        _run_wgrad_wrapper(cfg, inputs, sf_fp8_dtype_override)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_wgrad_e5m3_is_not_cached_as_e4m3():
    """sf_fp8_dtype_override must take part in the compile cache key.

    Identical scale-factor bytes decode differently under E4M3 and UE5M3, so if
    the override were missing from the key the second call would reuse the first
    kernel and silently return E4M3 results.
    """
    _skip_unless_e5m3_supported()
    cfg, inputs = _wgrad_nvfp4_inputs()
    w_e4m3 = _run_wgrad_wrapper(cfg, inputs, None)["wgrad_tensor"].float().clone()
    w_e5m3 = _run_wgrad_wrapper(cfg, inputs, "e5m3")["wgrad_tensor"].float().clone()
    torch.cuda.synchronize()
    assert not torch.equal(
        w_e4m3, w_e5m3
    ), "e5m3 and e4m3 produced identical output from identical scale-factor bytes; sf_fp8_dtype_override is likely missing from the compile cache key"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_wgrad_bf16_rejects_sf_fp8_dtype_override():
    """The BF16 backend has no scale factors, so any explicit override is an error.

    The None case is the important one: wgrad forwards **kwargs to both backends,
    so merely adding the parameter once broke every BF16 call with a TypeError,
    which a rejection-only test would not have caught.
    """
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM WGrad BF16 kernel.")
    problem = make_grouped_gemm_wgrad_bf16_problem(discrete=False)
    kwargs = dict(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=problem["offsets"],
        output_mode="dense",
        wgrad_tensor=problem["output"],
        wgrad_ptrs=problem["output_ptrs"],
        acc_dtype=torch.float32,
        wgrad_dtype=problem["output_dtype"],
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        input_order=problem["input_order"],
    )
    # None is accepted and dispatches to BF16 as usual.
    cudnn.grouped_gemm_wgrad_wrapper_sm100(**kwargs, sf_fp8_dtype_override=None)
    with pytest.raises(ValueError, match="BF16 forbids scale control sf_fp8_dtype_override"):
        cudnn.grouped_gemm_wgrad_wrapper_sm100(**kwargs, sf_fp8_dtype_override="e5m3")


@pytest.mark.L0
@pytest.mark.parametrize("kernel", ["moe_grouped_gemm_wgrad.py", "moe_blockscaled_grouped_gemm_wgrad.py"])
def test_wgrad_scheduler_advances_before_producer_tail(kernel):
    """Without the advance, producer_tail's first wait reuses the END sentinel's parity and can hang the CTA forever."""

    def calls(statement, receiver, method):
        call = statement.value if isinstance(statement, ast.Expr) else None
        return (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == method
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == receiver
        )

    source = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "gemm" / "cutedsl" / "grouped" / "wgrad" / kernel
    tails = 0
    for node in ast.walk(ast.parse(source.read_text())):
        for block in (getattr(node, "body", None), getattr(node, "orelse", None)):
            if not isinstance(block, list):
                continue
            for previous, statement in zip([None] + block, block):
                if calls(statement, "sched_pipeline", "producer_tail"):
                    tails += 1
                    assert previous is not None and calls(previous, "sched_producer_state", "advance"), f"{source}:{statement.lineno}"
    assert tails == 1


# ---------------------------------------------------------------------------
# BF16 wide tiles: 256 x 512 per CTA pair, each A tile shared by both 256-column halves. The API runs the wide kernel for
# dense output, Tensor2D input and n % 512 == 0 at the default 2-CTA 256x256 tile when the dW has enough wide tiles
# for the device (_wide_tiles_pay_off); its dW must be the stock kernel's byte for byte. The discrete output mode always
# runs the stock kernel on the same inputs.
# ---------------------------------------------------------------------------


def _bf16_wgrad_api(m, n, experts, *, output_mode="dense", tile=(256, 256), cluster=(2, 1), wgrad_dtype=torch.bfloat16, **kwargs):
    """A BF16 wgrad API for dW of experts x (m, n) built from CPU tensors: enough for the kernel selection, never compiled."""
    from cudnn.gemm.cutedsl.grouped.wgrad._bf16_api import GroupedGemmWgradBf16API

    tokens = 256
    a = torch.empty((tokens, m), dtype=torch.bfloat16).t()
    b = torch.empty((tokens, n), dtype=torch.bfloat16)
    offsets = torch.zeros((experts,), dtype=torch.int32)
    if output_mode == "dense":
        out = dict(sample_wgrad=torch.empty((experts, m, n), dtype=wgrad_dtype))
    else:
        out = dict(num_experts=experts, wgrad_shape=(m, n), wgrad_dtype=wgrad_dtype)
    return GroupedGemmWgradBf16API(
        sample_a=a, sample_b=b, sample_sfa=None, sample_sfb=None, sample_offsets=offsets, mma_tiler_mn=tile, cluster_shape_mn=cluster, **out, **kwargs
    )


def _min_wide_experts(m, n, clusters, counter):
    """The fewest experts of dW (m, n) for which the API picks the wide kernel on `clusters` resident 2-CTA clusters."""
    from cudnn.gemm.cutedsl.grouped.wgrad._bf16_api import GroupedGemmWgradBf16API

    per_expert = (m // 256) * (n // 512)
    return -(-GroupedGemmWgradBf16API._WIDE_MIN_TILES_PER_CLUSTER[counter] * clusters // per_expert)


# MoE wgrad GEMMs at 8192 tokens per rank, (m, n, local experts) of dW1 (2I x H) and dW2 (H x I), on GB200 (76 resident
# 2-CTA clusters): the API picks the wide kernel where it measured faster than the stock one, per claim mode.
WIDE_RULE_GEMMS = [
    # name, m, n, experts, wide under CLC, wide with the atomic counter
    ("qwen35_122b-ep4-dW1", 2048, 3072, 64, True, True),
    ("qwen35_122b-ep4-dW2", 3072, 1024, 64, True, False),
    ("qwen35_122b-ep8-dW1", 2048, 3072, 32, True, False),
    ("qwen35_122b-ep8-dW2", 3072, 1024, 32, False, False),
    ("qwen35_397b-ep8-dW2", 4096, 1024, 64, True, True),
    ("glm53_flash-ep8-dW2", 4096, 2048, 36, True, True),
    ("deepseek_v3-ep8-dW2", 7168, 2048, 32, True, True),
    ("qwen38_2p4t-ep4-dW1", 4096, 8192, 128, True, True),
]


@pytest.mark.L0
@pytest.mark.parametrize("name, m, n, experts, wide_clc, wide_counter", WIDE_RULE_GEMMS, ids=[g[0] for g in WIDE_RULE_GEMMS])
def test_bf16_wgrad_wide_rule_on_moe_shapes(name, m, n, experts, wide_clc, wide_counter):
    """CPU only: the selection rule on MoE wgrad shapes at GB200's 76 resident 2-CTA clusters."""
    from cudnn.gemm.cutedsl.grouped.wgrad._bf16_api import GroupedGemmWgradBf16API

    assert GroupedGemmWgradBf16API._wide_tiles_pay_off(m, n, experts, 76, False) is wide_clc
    assert GroupedGemmWgradBf16API._wide_tiles_pay_off(m, n, experts, 76, True) is wide_counter


@pytest.mark.L0
@pytest.mark.parametrize("m, n", [(3072, 1024), (2048, 3072), (4096, 2048), (4096, 7168)])
@pytest.mark.parametrize("clusters", [1, 76, 106])
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_bf16_wgrad_wide_rule_boundary(m, n, clusters, counter):
    """CPU only: the wide kernel from exactly _WIDE_MIN_TILES_PER_CLUSTER wide tiles per resident cluster."""
    from cudnn.gemm.cutedsl.grouped.wgrad._bf16_api import GroupedGemmWgradBf16API

    experts = _min_wide_experts(m, n, clusters, counter)
    assert GroupedGemmWgradBf16API._wide_tiles_pay_off(m, n, experts, clusters, counter)
    assert not GroupedGemmWgradBf16API._wide_tiles_pay_off(m, n, experts - 1, clusters, counter)
    assert not GroupedGemmWgradBf16API._wide_tiles_pay_off(m, n, experts, 0, counter)


@pytest.mark.L0
@pytest.mark.parametrize("rubin", [False, True], ids=["sm100", "sm107"])
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_bf16_wgrad_wide_selection(monkeypatch, rubin, counter):
    """CPU only: _wide_applies() at the rule's boundary and for every configuration the wide kernel does not cover. The
    rule is the same on SM107 (with that device's cluster count); _use_wide_tiles = False keeps the stock kernel."""
    from cudnn.gemm.cutedsl.grouped.wgrad._bf16_api import GroupedGemmWgradBf16API

    clusters = 106 if rubin else 76
    monkeypatch.setattr(GroupedGemmWgradBf16API, "_max_active_clusters", lambda self: clusters)
    m, n = 2048, 3072
    experts = _min_wide_experts(m, n, clusters, counter)

    def applies(experts=experts, m=m, n=n, **kwargs):
        api = _bf16_wgrad_api(m, n, experts, atomic_counter_claim=counter, **kwargs)
        api._is_rubin_kernel = rubin
        return api._wide_applies()

    assert applies()
    assert not applies(experts - 1)
    assert not applies(n=n - 512 + 256)  # n % 512 != 0
    assert not applies(output_mode="discrete")
    assert not applies(accumulate_on_output=True)
    assert not applies(input_order="tensor_ragged")
    assert not applies(tile=(128, 256), cluster=(1, 1))
    assert not applies(cluster=(4, 1))
    assert not applies(wgrad_dtype=torch.float32)
    assert applies(wgrad_dtype=torch.float16)
    monkeypatch.setattr(GroupedGemmWgradBf16API, "_use_wide_tiles", False)
    assert not applies()


def _require_bf16_wide_device():
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("The wide BF16 wgrad is SM100/SM103/SM107 only")


def _bf16_wgrad_run(problem, output_mode, *, counter=False):
    output = torch.full_like(problem["output"], float("nan"))
    discrete = output_mode == "discrete"
    cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=problem["offsets"],
        output_mode=output_mode,
        wgrad_tensor=None if discrete else output,
        wgrad_ptrs=torch.tensor([e.data_ptr() for e in output], dtype=torch.int64, device="cuda") if discrete else None,
        acc_dtype=torch.float32,
        wgrad_dtype=output.dtype,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        atomic_counter_claim=counter,
    )
    torch.cuda.synchronize()
    return output


def _record_wide_picks(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.wgrad import _bf16_api

    picked = []
    original = _bf16_api.GroupedGemmWgradBf16API._wide_applies

    def record(self):
        applies = original(self)
        picked.append(applies)
        return applies

    monkeypatch.setattr(_bf16_api.GroupedGemmWgradBf16API, "_wide_applies", record)
    monkeypatch.setattr(cudnn.gemm.cutedsl.grouped.wgrad.api, "_wgrad_wrapper_memo", {})
    monkeypatch.setattr(cudnn.gemm.cutedsl.grouped.wgrad.api, "_cache_of_GroupedGemmWgradSm100Objects", {})
    return picked


@pytest.mark.L0
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_bf16_wgrad_default_picks_wide_at_rule_boundary(monkeypatch, counter):
    """With the library defaults, the fewest experts that reach the rule run the wide kernel and one expert fewer runs
    the stock kernel; both give the stock kernel's dW bytes (the discrete mode)."""
    _require_bf16_wide_device()
    import cutlass
    from determinism_utils import assert_bitwise, snapshot

    picked = _record_wide_picks(monkeypatch)
    m, n = 2048, 4096
    experts = _min_wide_experts(m, n, cutlass.utils.HardwareInfo().get_max_active_clusters(2), counter)
    for count in (experts, experts - 1):
        groups = (256, 0, 512) * (count // 3) + (256,) * (count % 3)
        problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=groups, m=m, n=n, a_major="m", b_major="n")
        dense = _bf16_wgrad_run(problem, "dense", counter=counter)
        stock = _bf16_wgrad_run(problem, "discrete", counter=counter)
        assert_bitwise({"wgrad": dense}, snapshot({"wgrad": stock}), f"{count} experts: default vs stock")
        del problem, dense, stock
    assert picked == [True, False, False, False]


# (groups, m, n): empty experts, uneven groups, and multi-pair clusters (m a multiple of 1024: (8, 2) under CLC).
WIDE_CASES = [
    pytest.param((256, 0, 1024, 256, 0, 512), 512, 1024, id="m512-n1024", marks=pytest.mark.L0),
    pytest.param((512, 0, 256, 2048, 0, 0, 768), 2048, 2048, id="m2048-n2048", marks=pytest.mark.L0),
    pytest.param((0, 0, 0), 1024, 512, id="all-empty", marks=pytest.mark.L1),
    pytest.param((1024, 256, 0, 512), 1024, 3072, id="m1024-n3072", marks=pytest.mark.L1),
]


@pytest.mark.parametrize("groups, m, n", WIDE_CASES)
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
@pytest.mark.parametrize("majors", [("m", "n"), ("k", "k")], ids=["mn-major", "k-major"])
def test_bf16_wgrad_wide_matches_stock(monkeypatch, groups, m, n, counter, majors):
    """The wide kernel, forced on these small shapes, gives the stock kernel's dW bytes."""
    _require_bf16_wide_device()
    from cudnn.gemm.cutedsl.grouped.wgrad import _bf16_api
    from determinism_utils import assert_bitwise, snapshot

    if not sum(groups) and majors != ("m", "n"):
        pytest.skip("an operand without tokens has one layout")
    picked = _record_wide_picks(monkeypatch)
    monkeypatch.setattr(_bf16_api.GroupedGemmWgradBf16API, "_WIDE_MIN_TILES_PER_CLUSTER", {False: 0, True: 0})
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=groups, m=m, n=n, a_major=majors[0], b_major=majors[1])
    wide = _bf16_wgrad_run(problem, "dense", counter=counter)
    stock = _bf16_wgrad_run(problem, "discrete", counter=counter)
    assert picked == [True, False]
    assert_bitwise({"wgrad": wide}, snapshot({"wgrad": stock}), "wide vs stock")
    assert_grouped_gemm_wgrad_bf16_close(wide, grouped_gemm_wgrad_bf16_reference(problem))
