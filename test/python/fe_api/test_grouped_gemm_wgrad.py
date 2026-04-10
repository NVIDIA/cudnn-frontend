"""Tests for grouped GEMM wgrad FE API."""

import pytest
import torch
import cudnn

from test_utils import torch_fork_set_rng
from fe_api.test_grouped_gemm_wgrad_utils import (
    grouped_gemm_wgrad_init,
    with_grouped_gemm_wgrad_params_fp4,
    with_grouped_gemm_wgrad_params_fp8,
    allocate_grouped_gemm_wgrad_tensors,
    allocate_grouped_gemm_wgrad_output,
    check_ref_grouped_gemm_wgrad,
)

# ---------------------------------------------------------------------------
# Dense mode: Class API
# ---------------------------------------------------------------------------


def _test_grouped_gemm_wgrad_dense_compile_execute(
    ab_dtype,
    wgrad_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
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
    wgrad_tensor = allocate_grouped_gemm_wgrad_output(cfg)

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
def test_grouped_gemm_wgrad_discrete_compile_execute_fp4(
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
    runtime_cfg = _cfg_with_group_k_list(compile_cfg, [128, 128])

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

    scale_cols = 0
    for group_k in group_k_list:
        scale_cols += ((group_k + sf_vec_size - 1) // sf_vec_size + 3) // 4 * 4

    return {
        "a_tensor": torch.empty((hidden, tokens_sum), dtype=torch.bfloat16),
        "b_tensor": torch.empty_strided((tokens_sum, intermediate), (1, tokens_sum), dtype=torch.bfloat16),
        "sfa_tensor": torch.empty((128, scale_cols), dtype=torch.bfloat16),
        "sfb_tensor": torch.empty((128, scale_cols), dtype=torch.bfloat16),
        "offsets_tensor": torch.tensor([sum(group_k_list[: i + 1]) for i in range(expert_cnt)], dtype=torch.int32),
    }


@pytest.mark.parametrize("output_mode", ["dense", "discrete"])
def test_grouped_gemm_wgrad_wrapper_dynamic_tokens_cache_behavior(monkeypatch, output_mode):
    from cudnn.grouped_gemm.grouped_gemm_wgrad import api as grouped_gemm_wgrad_api

    grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()

    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "check_support", lambda self: True)
    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "compile", counted_compile)
    monkeypatch.setattr(grouped_gemm_wgrad_api.GroupedGemmWgradSm100, "execute", lambda self, **kwargs: None)

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
