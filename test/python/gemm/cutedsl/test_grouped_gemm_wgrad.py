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
from determinism_utils import assert_bitwise, snapshot
from gemm.cutedsl.test_grouped_gemm_wgrad_utils import (
    _skip_unless_e5m3_supported,
    _wgrad_assemble_scales_2d2d,
    _wgrad_create_scale_tensor,
    _wgrad_independent_reference,
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


# MXFP8 wgrad variants for dense output, Tensor2D input and 2-CTA 256x256 tiles in a (2, 1) cluster.
#
# The API picks one of two variants of the block-scaled wgrad on its own (``_select_kernel``): the A-reuse variant when the
# experts are short on average (the token extent of A over the expert count fits its A ring) and dW has at least 3
# work units of N tiles per M tile, else the global-descriptor variant. SM100/SM103 and SM107 (Rubin) each have their
# own pair. Their dW must be the stock kernel's byte for byte, for every offsets value the stock kernel accepts: an
# expert whose token count is not a multiple of the K tile reads its last K tile through its own descriptors, so tokens
# past its end (the next expert's, or data past the last offset) never enter its sum. The discrete output mode runs the
# stock kernel on the same inputs. The cases below force one variant, except the selection tests.

_GLOBAL_DESC_M, _GLOBAL_DESC_N = 512, 768


def _global_desc_require_sm100():
    from cudnn.api_base import get_device_type
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")
    if torch.cuda.get_device_capability()[0] != 10 or get_device_type() == "rubin":
        pytest.skip("The global-descriptor and A-reuse wgrads are SM100/SM103 only")


def _global_desc_require_rubin():
    from cudnn.api_base import get_device_type

    if not torch.cuda.is_available() or get_device_type() != "rubin":
        pytest.skip("The Rubin global-descriptor and A-reuse wgrads run on SM107 only")


def _global_desc_problem(data_groups, num_experts, *, seed, n=_GLOBAL_DESC_N):
    """MXFP8 operands over ``data_groups`` token groups; the first ``num_experts`` are experts, the rest is nonzero data
    past the last offset. Scale factors are blocked per group when every group is whole K tiles but the last (the
    layout the reference reads), else random bytes of the size the API checks."""
    M, N = _GLOBAL_DESC_M, n
    torch.manual_seed(seed)
    tokens = sum(data_groups)
    groups = list(data_groups[:num_experts])
    offsets = torch.tensor(groups, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    if tokens == 0:
        a = torch.empty_strided((M, 0), (16, 1), dtype=torch.float8_e4m3fn, device="cuda")
        b = torch.empty_strided((0, N), (1, 16), dtype=torch.float8_e4m3fn, device="cuda")
        sfa = torch.empty_strided((M, 0), (16, 1), dtype=torch.float8_e8m0fnu, device="cuda")
        sfb = torch.empty_strided((N, 0), (16, 1), dtype=torch.float8_e8m0fnu, device="cuda")
        return {"a": a, "b": b, "sfa": sfa, "sfb": sfb, "offsets": offsets, "reference": None, "experts": num_experts, "n": N}
    logical_a = torch.randint(-1, 2, (M, tokens), device="cuda").to(torch.bfloat16)
    logical_b = torch.randint(-1, 2, (tokens, N), device="cuda").to(torch.bfloat16)
    logical_a[logical_a == 0] = 1
    logical_b[logical_b == 0] = -1
    blocked = all(k % 128 == 0 for k in data_groups[:-1])
    if blocked:
        raw_sfa = [_wgrad_create_scale_tensor((M, -(-k // 32)), torch.float8_e8m0fnu) for k in data_groups]
        raw_sfb = [_wgrad_create_scale_tensor((N, -(-k // 32)), torch.float8_e8m0fnu) for k in data_groups]
        sfa, sfb = _wgrad_assemble_scales_2d2d(raw_sfa, M), _wgrad_assemble_scales_2d2d(raw_sfb, N)
        reference = _wgrad_independent_reference(logical_a.float(), logical_b.float(), raw_sfa[:num_experts], raw_sfb[:num_experts], groups, 32, torch.bfloat16)
    else:
        cols = -(-(-(-tokens // 32)) // 4) * 4
        sfa = _wgrad_create_scale_tensor((M, cols), torch.float8_e8m0fnu)
        sfb = _wgrad_create_scale_tensor((N, cols), torch.float8_e8m0fnu)
        reference = None
    return {
        "a": logical_a.to(torch.float8_e4m3fn).contiguous(),
        "b": logical_b.to(torch.float8_e4m3fn).t().contiguous().t(),
        "sfa": sfa,
        "sfb": sfb,
        "offsets": offsets,
        "reference": reference,
        "experts": num_experts,
        "n": N,
    }


def _global_desc_run(problem, output_mode, *, counter=False):
    output = torch.full((problem["experts"], _GLOBAL_DESC_M, problem["n"]), float("nan"), dtype=torch.bfloat16, device="cuda")
    discrete = output_mode == "discrete"
    cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=problem["sfa"],
        sfb_tensor=problem["sfb"],
        offsets_tensor=problem["offsets"],
        output_mode=output_mode,
        wgrad_tensor=None if discrete else output,
        wgrad_ptrs=torch.tensor([e.data_ptr() for e in output], dtype=torch.int64, device="cuda") if discrete else None,
        acc_dtype=torch.float32,
        wgrad_dtype=torch.bfloat16,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        atomic_counter_claim=counter,
    )
    torch.cuda.synchronize()
    return output


def _global_desc_force_kernel(monkeypatch, kernel_class, **kernel_options):
    """Make the API build ``kernel_class`` (with ``kernel_options``) wherever it would pick one of the dense-output
    variants (global descriptors or A reuse, SM100/SM103 or Rubin); ``None`` keeps the API's pick. Uses fresh wrapper
    caches and returns the list of kernel classes picked."""
    import functools

    from cudnn.gemm.cutedsl.grouped.wgrad import _blockscaled_api
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel

    picked = []
    original = _blockscaled_api.GroupedGemmWgradBlockScaledAPI._select_kernel

    def select(self):
        kernel = original(self)
        if kernel_class is not None and issubclass(kernel, BlockScaledMoEGroupedGemmWgradDenseKernel):
            kernel = kernel_class
        picked.append(kernel)
        return functools.partial(kernel, **kernel_options) if kernel_options else kernel

    monkeypatch.setattr(_blockscaled_api.GroupedGemmWgradBlockScaledAPI, "_select_kernel", select)
    monkeypatch.setattr(cudnn.gemm.cutedsl.grouped.wgrad.api, "_wgrad_wrapper_memo", {})
    monkeypatch.setattr(cudnn.gemm.cutedsl.grouped.wgrad.api, "_cache_of_GroupedGemmWgradSm100Objects", {})
    return picked


@pytest.mark.L0
def test_dense_mxfp8_wgrad_picks_global_descriptors(monkeypatch):
    _global_desc_require_sm100()
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel

    picked = _global_desc_force_kernel(monkeypatch, None)
    # 832 tokens per expert on average: more than the A-reuse variant's 6 K tiles.
    problem = _global_desc_problem((1024, 640), 2, seed=1)
    _global_desc_run(problem, "dense")
    _global_desc_run(problem, "discrete")
    assert picked[0] is BlockScaledMoEGroupedGemmWgradDenseKernel
    assert picked[1] not in (BlockScaledMoEGroupedGemmWgradDenseKernel, BlockScaledMoEGroupedGemmWgradReuseKernel)


# (data groups, experts): partial last K tiles with nonzero data after them, empty experts, and starts that are not
# multiples of 128 (the stock kernel addresses an expert's scale factors at rows * start / 32 bytes and reads whole
# 512-byte atoms, so each case keeps those reads inside the scale-factor buffer).
_GLOBAL_DESC_CASES = [
    pytest.param((256, 0, 192, 128), 3, id="partial-last-expert-then-data", marks=pytest.mark.L0),
    pytest.param((128, 64, 0, 384, 96, 256), 5, id="partial-experts-then-data", marks=pytest.mark.L0),
    pytest.param((0, 0, 0), 3, id="all-empty", marks=pytest.mark.L1),
    pytest.param((96, 160, 32, 0, 224, 256), 5, id="unaligned-starts", marks=pytest.mark.L1),
]


@pytest.mark.parametrize("groups, experts", _GLOBAL_DESC_CASES)
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_global_descriptor_wgrad_matches_stock(monkeypatch, groups, experts, counter):
    _global_desc_require_sm100()
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel

    _global_desc_force_kernel(monkeypatch, BlockScaledMoEGroupedGemmWgradDenseKernel)
    problem = _global_desc_problem(groups, experts, seed=7)
    dense = _global_desc_run(problem, "dense", counter=counter)
    stock = _global_desc_run(problem, "discrete", counter=counter)
    assert_bitwise({"wgrad": dense}, snapshot({"wgrad": stock}), "global descriptors vs stock")
    if problem["reference"] is not None and all(k % 128 == 0 for k in groups[: experts - 1]):
        # Blocked scale factors match the stock addressing only when every earlier expert is whole K tiles.
        check_ref_grouped_gemm_wgrad(dense, problem["reference"])


def _global_desc_selection_api(
    *, device_type, m=_GLOBAL_DESC_M, n=_GLOBAL_DESC_N, groups=(256, 256), discrete=False, dtype=torch.float8_e4m3fn, wgrad_dtype=torch.bfloat16, **kwargs
):
    """A block-scaled wgrad API object (not compiled) as built on ``device_type`` ("blackwell" or "rubin"), with host
    tensors of the given shapes: ``groups`` are the experts' token counts."""
    from unittest import mock

    from cudnn.gemm.cutedsl.grouped.wgrad._blockscaled_api import GroupedGemmWgradBlockScaledAPI

    tokens, experts = sum(groups), len(groups)
    a = torch.empty((m, tokens), dtype=torch.uint8).view(dtype)
    b = torch.empty((n, tokens), dtype=torch.uint8).view(dtype).t()
    cols = -(-(-(-tokens // 32)) // 4) * 4
    sfa = torch.empty((m, cols), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    sfb = torch.empty((n, cols), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    offsets = torch.tensor(groups, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    if discrete:
        output = dict(num_experts=experts, wgrad_shape=(m, n), wgrad_dtype=wgrad_dtype)
    else:
        output = dict(sample_wgrad=torch.empty((experts, m, n), dtype=wgrad_dtype))
    with mock.patch("cudnn.api_base.get_device_type", return_value=device_type):
        return GroupedGemmWgradBlockScaledAPI(
            sample_a=a, sample_b=b, sample_sfa=sfa, sample_sfb=sfb, sample_offsets=offsets, sf_vec_size=32, **output, **kwargs
        )


@pytest.mark.L0
def test_wgrad_variant_selection_per_architecture():
    """The dense-output variants the API picks on SM100/SM103 and on SM107 (Rubin). Global descriptors wherever their
    conditions hold, else the stock kernel. A reuse where, in addition, the experts are short on average (at most 6 x 128
    tokens on SM100/SM103, 5 x 256 on Rubin) and dW has at least 3 work units of N tiles per M tile. Host-side only:
    the API objects are not compiled."""
    from cudnn.gemm.cutedsl.grouped.wgrad._blockscaled_api import dense_output_token_class
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense_rubin import BlockScaledMoEGroupedGemmWgradDenseRubinKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse_rubin import BlockScaledMoEGroupedGemmWgradReuseRubinKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel

    kernels = {
        "blackwell": (BlockScaledMoEGroupedGemmWgradDenseKernel, BlockScaledMoEGroupedGemmWgradReuseKernel, BlockScaledMoEGroupedGemmWgradKernel),
        "rubin": (BlockScaledMoEGroupedGemmWgradDenseRubinKernel, BlockScaledMoEGroupedGemmWgradReuseRubinKernel, BlockScaledMoEGroupedGemmWgradRubinKernel),
    }
    for device_type, (dense, reuse, stock) in kernels.items():

        def pick(**kwargs):
            return _global_desc_selection_api(device_type=device_type, **kwargs)._select_kernel()

        # Global descriptors: short experts, but one work unit per M tile (N of 3 tiles) or one N tile.
        for kwargs in ({}, dict(dtype=torch.float8_e5m2), dict(n=256), dict(atomic_counter_claim=True, use_pdl=True)):
            assert pick(**kwargs) is dense, (device_type, kwargs)
        for kwargs in (
            dict(discrete=True),
            dict(n=640),
            dict(wgrad_dtype=torch.float32),
            dict(accumulate_on_output=True),
            dict(cluster_shape_mn=(4, 1)),
            dict(mma_tiler_mn=(256, 128)),
            dict(mma_tiler_mn=(128, 256), cluster_shape_mn=(1, 1)),
            dict(n=1536, accumulate_on_output=True),
        ):
            assert pick(**kwargs) is stock, (device_type, kwargs)
        # At the thresholds. Units per M tile: N of 12 tiles is 3 units of 4, N of 9 tiles 3 units of 3, N of 8 tiles and
        # of 6 tiles 2 units. The A ring's capacity in tokens per expert, on average.
        capacity = reuse.A_STAGES * reuse.K_TILE
        assert pick(n=3072, groups=(capacity, capacity)) is reuse, device_type
        assert pick(n=2304, groups=(capacity, capacity)) is reuse, device_type
        assert pick(n=2048, groups=(capacity, capacity)) is dense, device_type
        assert pick(n=1536, groups=(capacity, capacity)) is dense, device_type
        assert pick(n=3072, groups=(capacity + 1, capacity)) is dense, device_type
        assert pick(n=3072, groups=(capacity, 0, capacity)) is reuse, device_type
        assert pick(n=3072, groups=(256, 256), discrete=True) is stock, device_type
        assert dense_output_token_class(2 * capacity + 1, 2, device_type == "rubin") == 0
    assert BlockScaledMoEGroupedGemmWgradReuseKernel.A_STAGES * 128 == 768
    assert BlockScaledMoEGroupedGemmWgradReuseRubinKernel.A_STAGES * 256 == 1280
    # The A stages each is built with: SM100/SM103 a fixed ring, Rubin sized to the mean expert (2 to 5 K tiles).
    assert dense_output_token_class(2 * 768, 2, False) == 6
    assert [dense_output_token_class(2 * rows, 2, True) for rows in (1, 256, 512, 768, 1024, 1280)] == [2, 2, 2, 3, 4, 5]


# The 20 benchmark GEMMs (5 MoE models x EP4 / EP8 x dW1 / dW2, 8192 tokens per rank routed evenly over the local
# experts, each padded to 256 rows): (model, EP, GEMM, m, n, rows per expert) -> A reuse on SM100/SM103, on SM107. A reuse
# is picked only where it measured clearly faster than global descriptors (see prefer_over_dense).
_WGRAD_BENCHMARK_PICKS = [
    ("deepseek_v3", 4, "dW1", 4096, 7168, 1024, False, True),
    ("deepseek_v3", 4, "dW2", 7168, 2048, 1024, False, False),
    ("qwen35_122b", 4, "dW1", 2048, 3072, 1024, False, True),
    ("qwen35_122b", 4, "dW2", 3072, 1024, 1024, False, False),
    ("qwen38_2p4t", 4, "dW1", 4096, 8192, 768, True, True),
    ("qwen38_2p4t", 4, "dW2", 8192, 2048, 768, False, False),
    ("qwen35_397b", 4, "dW1", 2048, 4096, 768, True, True),
    ("qwen35_397b", 4, "dW2", 4096, 1024, 768, False, False),
    ("glm53_flash", 4, "dW1", 4096, 4096, 1024, False, True),
    ("glm53_flash", 4, "dW2", 4096, 2048, 1024, False, False),
    ("deepseek_v3", 8, "dW1", 4096, 7168, 2048, False, False),
    ("deepseek_v3", 8, "dW2", 7168, 2048, 2048, False, False),
    ("qwen35_122b", 8, "dW1", 2048, 3072, 2048, False, False),
    ("qwen35_122b", 8, "dW2", 3072, 1024, 2048, False, False),
    ("qwen38_2p4t", 8, "dW1", 4096, 8192, 1280, False, True),
    ("qwen38_2p4t", 8, "dW2", 8192, 2048, 1280, False, False),
    ("qwen35_397b", 8, "dW1", 2048, 4096, 1280, False, True),
    ("qwen35_397b", 8, "dW2", 4096, 1024, 1280, False, False),
    ("glm53_flash", 8, "dW1", 4096, 4096, 2048, False, False),
    ("glm53_flash", 8, "dW2", 4096, 2048, 2048, False, False),
]


@pytest.mark.L0
@pytest.mark.parametrize("device_type", ["blackwell", "rubin"])
def test_wgrad_variant_selection_at_benchmark_shapes(device_type):
    """The API's pick at each benchmark GEMM (two experts of the GEMM's rows; the rule depends on the mean only).
    Host-side only."""
    from cudnn.gemm.cutedsl.grouped.wgrad._blockscaled_api import _dense_output_variants

    dense, reuse = _dense_output_variants(device_type == "rubin")
    for model, ep, gemm, m, n, rows, reuse_sm100, reuse_sm107 in _WGRAD_BENCHMARK_PICKS:
        expected = reuse if (reuse_sm107 if device_type == "rubin" else reuse_sm100) else dense
        picked = _global_desc_selection_api(device_type=device_type, m=m, n=n, groups=(rows, rows))._select_kernel()
        assert picked is expected, (device_type, model, ep, gemm, picked.__name__)


# Rubin (SM107) global descriptors: K64 MMA, a 256-token K tile with two SF atoms and two disjoint accumulator stages;
# dW must be the Rubin kernel's byte for byte. The cases cover work of 1 to 6 N tiles, experts longer than others,
# experts ending half way into a 256-token K tile (multiples of 128 tokens), a last expert shorter than one SF atom
# (96 tokens), nonzero data past the last offset, and empty experts.
_GLOBAL_DESC_RUBIN_CASES = [
    pytest.param(768, (256, 512, 1280, 0, 256), 5, id="n768-aligned", marks=pytest.mark.L1),
    pytest.param(512, (2048, 256, 1536, 512), 4, id="n512-long-and-short", marks=pytest.mark.L1),
    pytest.param(1024, (128, 384, 0, 640, 256), 4, id="n1024-half-k-tiles-then-data", marks=pytest.mark.L0),
    pytest.param(1536, (256, 0, 192, 128), 3, id="n1536-partial-last-expert-then-data", marks=pytest.mark.L1),
    pytest.param(256, (384, 0, 1024, 96, 160), 4, id="n256-short-last-expert-then-data", marks=pytest.mark.L0),
]


@pytest.mark.parametrize("n, groups, experts", _GLOBAL_DESC_RUBIN_CASES)
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_rubin_global_descriptor_wgrad_matches_rubin_stock(monkeypatch, n, groups, experts, counter):
    _global_desc_require_rubin()
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense_rubin import BlockScaledMoEGroupedGemmWgradDenseRubinKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel

    picked = _global_desc_force_kernel(monkeypatch, BlockScaledMoEGroupedGemmWgradDenseRubinKernel)
    problem = _global_desc_problem(groups, experts, seed=17, n=n)
    dense = _global_desc_run(problem, "dense", counter=counter)
    stock = _global_desc_run(problem, "discrete", counter=counter)
    assert picked[0] is BlockScaledMoEGroupedGemmWgradDenseRubinKernel and picked[-1] is BlockScaledMoEGroupedGemmWgradRubinKernel, picked
    assert_bitwise({"wgrad": dense}, snapshot({"wgrad": stock}), "Rubin global descriptors vs Rubin stock")


# A reuse: a work unit is 2, 3 or 4 N tiles (the largest divisor of dW's N tile count up to 4). An expert whose K tiles
# fit the A ring (6 x 128 tokens on SM100/SM103, up to 5 x 256 on Rubin) loads A once per unit; a longer one reloads A
# per N tile.


@pytest.mark.L0
def test_reuse_selection_rule():
    """Host-side rule: more than one N tile per unit and at most 6 x 128 tokens per expert on average; the API also
    needs at least 3 units per M tile."""
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel as reuse

    assert [reuse.n_group_for(n) for n in (256, 512, 768, 1024, 1280, 1536, 3072)] == [1, 2, 3, 4, 1, 3, 4]
    fits = dict(ab_bits=8, c_bits=16, has_global_scale=False)
    assert reuse.can_implement(512, 768, tokens=768 * 4, experts=4, **fits)
    assert not reuse.can_implement(512, 768, tokens=768 * 4 + 1, experts=4, **fits)
    assert not reuse.can_implement(512, 256, tokens=128, experts=4, **fits)  # one N tile
    assert not reuse.can_implement(512, 1280, tokens=128, experts=4, **fits)  # 5 N tiles: units of one
    assert not reuse.can_implement(512, 768, tokens=0, experts=0, **fits)
    assert reuse.can_implement(512, 768, tokens=0, experts=3, **fits)
    assert not reuse.can_implement(512, 768, tokens=128, experts=4, ab_bits=8, c_bits=32, has_global_scale=False)
    # Fewer than 3 units per M tile: not preferred over global descriptors.
    units = (512, 1), (768, 1), (1024, 1), (1536, 2), (2048, 2), (2304, 3), (3072, 3), (4096, 4), (6144, 6), (7168, 7), (8192, 8)
    for n, count in units:
        assert reuse.units_per_m_tile(n) == count, n
        assert reuse.prefer_over_dense(4096, n, tokens=768 * 4, experts=4, **fits) is (count >= 3), n
    assert not reuse.prefer_over_dense(4096, 4096, tokens=768 * 4 + 1, experts=4, **fits)


@pytest.mark.L0
def test_api_picks_reuse_for_short_experts(monkeypatch):
    _global_desc_require_sm100()
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel

    picked = _global_desc_force_kernel(monkeypatch, None)
    short = _global_desc_problem((256, 0, 384), 3, seed=1, n=3072)
    long = _global_desc_problem((1024, 896), 2, seed=2, n=3072)
    one_tile = _global_desc_problem((256, 0, 384), 3, seed=3, n=256)
    two_units = _global_desc_problem((256, 0, 384), 3, seed=4, n=2048)
    _global_desc_run(short, "dense")
    _global_desc_run(long, "dense")  # same shapes but the token count: a second compiled entry
    _global_desc_run(short, "dense")  # cached
    _global_desc_run(one_tile, "dense")
    _global_desc_run(two_units, "dense")
    _global_desc_run(short, "discrete")
    assert picked[:4] == [
        BlockScaledMoEGroupedGemmWgradReuseKernel,
        BlockScaledMoEGroupedGemmWgradDenseKernel,
        BlockScaledMoEGroupedGemmWgradDenseKernel,
        BlockScaledMoEGroupedGemmWgradDenseKernel,
    ]
    assert picked[4] not in (BlockScaledMoEGroupedGemmWgradDenseKernel, BlockScaledMoEGroupedGemmWgradReuseKernel)


# (dW N, data groups, experts). N 512 / 768 / 1024 / 1536: units of 2, 3, 4 N tiles and two units of 3.
_REUSE_CASES = [
    # whole K tiles, up to the A ring's 6
    pytest.param(768, (256, 128, 768, 0, 640), 5, id="n768-aligned", marks=pytest.mark.L0),
    # experts longer than the A ring (A reloaded per N tile) next to short ones
    pytest.param(512, (1024, 128, 896, 256), 4, id="n512-long-and-short", marks=pytest.mark.L0),
    # partial last K tiles with nonzero data after them, empty experts
    pytest.param(1024, (128, 64, 0, 384, 96, 256), 5, id="n1024-partial-experts-then-data", marks=pytest.mark.L0),
    pytest.param(1536, (256, 0, 192, 128), 3, id="n1536-partial-last-expert-then-data", marks=pytest.mark.L0),
    pytest.param(512, (1000, 24, 512), 2, id="n512-long-partial-then-data", marks=pytest.mark.L1),
    pytest.param(768, (0, 0, 0), 3, id="n768-all-empty", marks=pytest.mark.L1),
    pytest.param(768, (0, 0, 256), 2, id="n768-empty-experts-then-data", marks=pytest.mark.L1),
    # starts that are not multiples of 128 (each case keeps the stock kernel's whole-atom SF reads in the buffer)
    pytest.param(1024, (96, 160, 32, 0, 224, 256), 5, id="n1024-unaligned-starts", marks=pytest.mark.L1),
]


@pytest.mark.parametrize("n, groups, experts", _REUSE_CASES)
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_reuse_wgrad_matches_stock(monkeypatch, n, groups, experts, counter):
    _global_desc_require_sm100()
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel

    picked = _global_desc_force_kernel(monkeypatch, BlockScaledMoEGroupedGemmWgradReuseKernel)
    problem = _global_desc_problem(groups, experts, seed=11, n=n)
    reuse = _global_desc_run(problem, "dense", counter=counter)
    stock = _global_desc_run(problem, "discrete", counter=counter)
    assert picked[0] is BlockScaledMoEGroupedGemmWgradReuseKernel
    assert_bitwise({"wgrad": reuse}, snapshot({"wgrad": stock}), "A reuse vs stock")
    if problem["reference"] is not None and all(k % 128 == 0 for k in groups[: experts - 1]):
        check_ref_grouped_gemm_wgrad(reuse, problem["reference"])


@pytest.mark.L0
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_reuse_wgrad_pdl_options(counter):
    """use_pdl and launch_pdl_dependents_at_entry through the class API: same dW as without them."""
    _global_desc_require_sm100()
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel

    problem = _global_desc_problem((256, 0, 384, 128), 3, seed=5, n=3072)
    outputs = []
    for options in ({}, dict(use_pdl=True), dict(use_pdl=True, launch_pdl_dependents_at_entry=True)):
        output = torch.full((problem["experts"], _GLOBAL_DESC_M, problem["n"]), float("nan"), dtype=torch.bfloat16, device="cuda")
        op = cudnn.GroupedGemmWgradSm100(
            sample_a=problem["a"],
            sample_b=problem["b"],
            sample_sfa=problem["sfa"],
            sample_sfb=problem["sfb"],
            sample_offsets=problem["offsets"],
            sample_wgrad=output,
            acc_dtype=torch.float32,
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            sf_vec_size=32,
            atomic_counter_claim=counter,
            **options,
        )
        assert op.check_support()
        assert op._implementation._select_kernel() is BlockScaledMoEGroupedGemmWgradReuseKernel
        op.compile()
        for _ in range(2):
            op.execute(
                a_tensor=problem["a"],
                b_tensor=problem["b"],
                sfa_tensor=problem["sfa"],
                sfb_tensor=problem["sfb"],
                offsets_tensor=problem["offsets"],
                wgrad_tensor=output,
            )
        torch.cuda.synchronize()
        outputs.append(output)
    stock = _global_desc_run(problem, "discrete", counter=counter)
    for output in outputs:
        assert_bitwise({"wgrad": output}, snapshot({"wgrad": stock}), "A reuse with PDL options vs stock")


# Rubin (SM107) A reuse, against the Rubin kernel: units of 2, 3 and 4 N tiles; experts longer than the A ring; experts
# ending half way into a 256-token K tile (128-token multiples) with data after them; empty experts.
_REUSE_RUBIN_CASES = [
    pytest.param(768, (256, 512, 1280, 0, 256), 5, id="n768-aligned", marks=pytest.mark.L1),
    pytest.param(512, (2048, 256, 1536, 512), 4, id="n512-long-and-short", marks=pytest.mark.L1),
    pytest.param(1024, (128, 384, 0, 640, 256), 4, id="n1024-half-k-tiles-then-data", marks=pytest.mark.L1),
    pytest.param(1536, (256, 0, 192, 128), 3, id="n1536-partial-last-expert-then-data", marks=pytest.mark.L1),
]


@pytest.mark.parametrize("n, groups, experts", _REUSE_RUBIN_CASES)
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
@pytest.mark.parametrize("a_stages", [5, 2], ids=["a5", "a2"])
def test_rubin_reuse_wgrad_matches_rubin_stock(monkeypatch, n, groups, experts, counter, a_stages):
    _global_desc_require_rubin()
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse_rubin import BlockScaledMoEGroupedGemmWgradReuseRubinKernel

    picked = _global_desc_force_kernel(monkeypatch, BlockScaledMoEGroupedGemmWgradReuseRubinKernel, a_stages=a_stages)
    problem = _global_desc_problem(groups, experts, seed=13, n=n)
    reuse = _global_desc_run(problem, "dense", counter=counter)
    stock = _global_desc_run(problem, "discrete", counter=counter)
    assert picked[0] is BlockScaledMoEGroupedGemmWgradReuseRubinKernel
    assert_bitwise({"wgrad": reuse}, snapshot({"wgrad": stock}), "Rubin A reuse vs Rubin stock")


@pytest.mark.L0
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_rubin_api_picks_wgrad_variants(monkeypatch, counter):
    """On Rubin the API picks A reuse for short experts with at least 3 work units per M tile, else global
    descriptors, and either matches the Rubin kernel bitwise."""
    _global_desc_require_rubin()
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense_rubin import BlockScaledMoEGroupedGemmWgradDenseRubinKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse_rubin import BlockScaledMoEGroupedGemmWgradReuseRubinKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel

    picked = _global_desc_force_kernel(monkeypatch, None)
    cases = [
        # (groups, experts, n, kernel): short experts in 3 units of 4 N tiles; experts past the A ring (1408 tokens on
        # average); short experts in 2 units per M tile; one N tile.
        ((256, 0, 384, 128), 3, 3072, BlockScaledMoEGroupedGemmWgradReuseRubinKernel),
        ((1536, 1280), 2, 3072, BlockScaledMoEGroupedGemmWgradDenseRubinKernel),
        ((256, 0, 384, 128), 3, 2048, BlockScaledMoEGroupedGemmWgradDenseRubinKernel),
        ((384, 0, 1024, 96, 160), 4, 256, BlockScaledMoEGroupedGemmWgradDenseRubinKernel),
    ]
    for seed, (groups, experts, n, kernel) in enumerate(cases):
        problem = _global_desc_problem(groups, experts, seed=21 + seed, n=n)
        start = len(picked)
        variant = _global_desc_run(problem, "dense", counter=counter)
        stock = _global_desc_run(problem, "discrete", counter=counter)
        assert picked[start:] == [kernel, BlockScaledMoEGroupedGemmWgradRubinKernel], (n, groups, picked[start:])
        assert_bitwise({"wgrad": variant}, snapshot({"wgrad": stock}), f"Rubin {kernel.__name__} vs Rubin stock")


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


def _min_wide_experts(m, n, clusters):
    """The fewest experts of dW (m, n) for which the API picks the wide kernel on `clusters` resident 2-CTA clusters."""
    from cudnn.gemm.cutedsl.grouped.wgrad._bf16_api import GroupedGemmWgradBf16API

    per_expert = (m // 256) * (n // 512)
    return -(-GroupedGemmWgradBf16API._WIDE_MIN_TILES_PER_CLUSTER * clusters // per_expert)


# The 20 MoE weight gradients at 8192 tokens per rank, (m, n, local experts) of dW1 (2I x H) and dW2 (H x I), and
# whether the API picks the wide kernel on GB200 (76 resident 2-CTA clusters) and VR200 (106).
WIDE_RULE_GEMMS = [
    # name, m, n, experts, wide on GB200, wide on VR200
    ("deepseek_v3-ep4-dW1", 4096, 7168, 64, True, True),
    ("deepseek_v3-ep4-dW2", 7168, 2048, 64, True, True),
    ("qwen35_122b-ep4-dW1", 2048, 3072, 64, True, True),
    ("qwen35_122b-ep4-dW2", 3072, 1024, 64, True, False),
    ("qwen35_397b-ep4-dW1", 2048, 4096, 128, True, True),
    ("qwen35_397b-ep4-dW2", 4096, 1024, 128, True, True),
    ("glm53_flash-ep4-dW1", 4096, 4096, 72, True, True),
    ("glm53_flash-ep4-dW2", 4096, 2048, 72, True, True),
    ("qwen38_2p4t-ep4-dW1", 4096, 8192, 128, True, True),
    ("qwen38_2p4t-ep4-dW2", 8192, 2048, 128, True, True),
    ("deepseek_v3-ep8-dW1", 4096, 7168, 32, True, True),
    ("deepseek_v3-ep8-dW2", 7168, 2048, 32, True, True),
    ("qwen35_122b-ep8-dW1", 2048, 3072, 32, True, False),
    ("qwen35_122b-ep8-dW2", 3072, 1024, 32, False, False),
    ("qwen35_397b-ep8-dW1", 2048, 4096, 64, True, True),
    ("qwen35_397b-ep8-dW2", 4096, 1024, 64, True, True),
    ("glm53_flash-ep8-dW1", 4096, 4096, 36, True, True),
    ("glm53_flash-ep8-dW2", 4096, 2048, 36, True, True),
    ("qwen38_2p4t-ep8-dW1", 4096, 8192, 64, True, True),
    ("qwen38_2p4t-ep8-dW2", 8192, 2048, 64, True, True),
]


@pytest.mark.L0
@pytest.mark.parametrize("name, m, n, experts, wide_gb200, wide_vr200", WIDE_RULE_GEMMS, ids=[g[0] for g in WIDE_RULE_GEMMS])
def test_bf16_wgrad_wide_rule_on_moe_shapes(name, m, n, experts, wide_gb200, wide_vr200):
    """CPU only: the selection rule on the MoE wgrad shapes at the resident cluster counts of GB200 and VR200."""
    from cudnn.gemm.cutedsl.grouped.wgrad._bf16_api import GroupedGemmWgradBf16API

    assert GroupedGemmWgradBf16API._wide_tiles_pay_off(m, n, experts, 76) is wide_gb200
    assert GroupedGemmWgradBf16API._wide_tiles_pay_off(m, n, experts, 106) is wide_vr200


@pytest.mark.L0
@pytest.mark.parametrize("m, n", [(3072, 1024), (2048, 3072), (4096, 2048), (4096, 7168)])
@pytest.mark.parametrize("clusters", [1, 76, 106])
def test_bf16_wgrad_wide_rule_boundary(m, n, clusters):
    """CPU only: the wide kernel from exactly _WIDE_MIN_TILES_PER_CLUSTER wide tiles per resident cluster."""
    from cudnn.gemm.cutedsl.grouped.wgrad._bf16_api import GroupedGemmWgradBf16API

    experts = _min_wide_experts(m, n, clusters)
    assert GroupedGemmWgradBf16API._wide_tiles_pay_off(m, n, experts, clusters)
    assert not GroupedGemmWgradBf16API._wide_tiles_pay_off(m, n, experts - 1, clusters)
    assert not GroupedGemmWgradBf16API._wide_tiles_pay_off(m, n, experts, 0)


@pytest.mark.L0
@pytest.mark.parametrize("rubin", [False, True], ids=["sm100", "sm107"])
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_bf16_wgrad_wide_selection(monkeypatch, rubin, counter):
    """CPU only: _wide_applies() at the rule's boundary and for every configuration the wide kernel does not cover. The
    rule is the same on SM107 (with that device's cluster count) and for both claim modes; _use_wide_tiles = False
    keeps the stock kernel."""
    from cudnn.gemm.cutedsl.grouped.wgrad._bf16_api import GroupedGemmWgradBf16API

    clusters = 106 if rubin else 76
    monkeypatch.setattr(GroupedGemmWgradBf16API, "_max_active_clusters", lambda self: clusters)
    m, n = 2048, 3072
    experts = _min_wide_experts(m, n, clusters)

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
    experts = _min_wide_experts(m, n, cutlass.utils.HardwareInfo().get_max_active_clusters(2))
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
    monkeypatch.setattr(_bf16_api.GroupedGemmWgradBf16API, "_WIDE_MIN_TILES_PER_CLUSTER", 0)
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=groups, m=m, n=n, a_major=majors[0], b_major=majors[1])
    wide = _bf16_wgrad_run(problem, "dense", counter=counter)
    stock = _bf16_wgrad_run(problem, "discrete", counter=counter)
    assert picked == [True, False]
    assert_bitwise({"wgrad": wide}, snapshot({"wgrad": stock}), "wide vs stock")
    assert_grouped_gemm_wgrad_bf16_close(wide, grouped_gemm_wgrad_bf16_reference(problem))
