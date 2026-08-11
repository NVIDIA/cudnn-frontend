# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for grouped GEMM wgrad FE API."""

import pytest
import torch
import cudnn

from test_utils import torch_fork_set_rng
from fe_api.test_fe_api_utils import reencode_sf_tensor_as_ue5m3
from fe_api.grouped_gemm.test_grouped_gemm_wgrad_utils import (
    _skip_unless_e5m3_supported,
    grouped_gemm_wgrad_init,
    with_grouped_gemm_wgrad_params_fp4,
    with_grouped_gemm_wgrad_params_fp8,
    allocate_grouped_gemm_wgrad_tensors,
    allocate_grouped_gemm_wgrad_output,
    check_ref_grouped_gemm_wgrad,
    wgrad_to_ragged_layout,
)
from fe_api.test_grouped_gemm_wgrad_bf16_utils import (
    assert_grouped_gemm_wgrad_close as assert_grouped_gemm_wgrad_bf16_close,
    grouped_gemm_wgrad_bf16_reference,
    make_grouped_gemm_wgrad_bf16_problem,
)

# ---------------------------------------------------------------------------
# Dense mode: Class API
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["bf16-dense", "bf16-discrete"])
def test_grouped_gemm_wgrad_wrapper_bf16(discrete):
    if torch.cuda.get_device_capability()[0] < 10:
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


@pytest.mark.parametrize("output_mode", ["dense", "discrete"])
def test_grouped_gemm_wgrad_wrapper_dynamic_tokens_cache_behavior(monkeypatch, output_mode):
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
    if torch.cuda.get_device_capability()[0] < 10:
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
