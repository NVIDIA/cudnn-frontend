# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""SM90 (Hopper) BF16 grouped GEMM wgrad: grouped_gemm_wgrad_wrapper_sm90 / GroupedGemmWgradSm90."""

import cutlass
import pytest
import torch

from cudnn.tensor_adapter import has_wgmma
import cudnn

from gemm.cutedsl.test_grouped_gemm_sm90 import override_schedule
from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import (
    assert_grouped_gemm_wgrad_close,
    grouped_gemm_wgrad_bf16_reference,
    make_grouped_gemm_wgrad_bf16_problem,
)

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm90():
    if not torch.cuda.is_available() or not has_wgmma(torch.cuda.get_device_capability()[0]):
        pytest.skip("The SM90 grouped GEMM wgrad needs a cc 9.0 device")


def run_wrapper(problem, schedule=None, **overrides):
    """``schedule`` forces the SM90 schedule ("cooperative" / "pingpong"); None keeps the heuristic."""
    kwargs = dict(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=problem["offsets"],
        wgrad_tensor=problem["output"],
        acc_dtype=torch.float32,
        wgrad_dtype=problem["output_dtype"],
    )
    kwargs.update(overrides)
    with override_schedule(schedule):
        return cudnn.grouped_gemm_wgrad_wrapper_sm90(**kwargs)["wgrad_tensor"]


# id: (group_k_list, m, n, a_major, b_major, wgrad dtype, mma_tiler_mn, cluster_shape_mn); None takes the SM90 default.
CASES = {
    "default": ((256, 0, 256), 256, 256, "m", "n", torch.bfloat16, None, None),
    "kmajor-fp32": ((512, 256), 256, 512, "k", "k", torch.float32, None, None),
    "edge-n192": ((256, 512, 0, 256), 384, 192, "m", "n", torch.bfloat16, (128, 192), (2, 1)),
    "fp16-128x128-c11": ((0, 256, 768), 512, 256, "m", "n", torch.float16, (128, 128), (1, 1)),
    "64x256-c12": ((256,) * 8, 320, 512, "m", "n", torch.bfloat16, (64, 256), (1, 2)),
    "experts32": ((0, 256, 512, 1024) * 8, 512, 256, "m", "k", torch.bfloat16, None, None),
    "one-tile-cluster": ((256, 256), 128, 128, "k", "n", torch.bfloat16, None, (2, 1)),
    "long-k": ((4096, 256), 256, 256, "m", "n", torch.bfloat16, None, None),
}


@pytest.mark.parametrize("case", list(CASES))
def test_sm90_wgrad_matches_reference(case):
    group_k_list, m, n, a_major, b_major, dtype, tile, cluster = CASES[case]
    # A nonzero initial value checks that every tile, including empty experts' zeros, is overwritten.
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=group_k_list, m=m, n=n, a_major=a_major, b_major=b_major, output_dtype=dtype, initial_value=3.0)
    actual = run_wrapper(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster)
    torch.cuda.synchronize()
    assert actual is problem["output"]
    assert_grouped_gemm_wgrad_close(actual, grouped_gemm_wgrad_bf16_reference(problem))


def test_sm90_wgrad_class_api_reuses_compile_across_token_counts():
    sample = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(256, 256), m=256, n=256, a_major="m", b_major="n")
    op = cudnn.GroupedGemmWgradSm90(
        sample_a=sample["a"],
        sample_b=sample["b"],
        sample_sfa=None,
        sample_sfb=None,
        sample_offsets=sample["offsets"],
        sample_wgrad=sample["output"],
        acc_dtype=torch.float32,
    )
    assert op.check_support()
    op.compile()
    for group_k_list in ((256, 256), (0, 1024), (768, 0)):
        problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=group_k_list, m=256, n=256, a_major="m", b_major="n", initial_value=-1.0)
        op.execute(
            a_tensor=problem["a"],
            b_tensor=problem["b"],
            sfa_tensor=None,
            sfb_tensor=None,
            offsets_tensor=problem["offsets"],
            wgrad_tensor=problem["output"],
        )
        torch.cuda.synchronize()
        assert_grouped_gemm_wgrad_close(problem["output"], grouped_gemm_wgrad_bf16_reference(problem))


def test_sm90_wgrad_schedule_follows_rows_per_expert(monkeypatch):
    # Output N 1024 is not a multiple of 192: the rule picks pingpong at <= 1024 average rows per expert and cooperative
    # above. The token count is a dynamic dimension of the wrapper's compile cache, so the cache key carries the pick.
    from cudnn.gemm.cutedsl.grouped.wgrad import api as wgrad_api

    monkeypatch.setattr(wgrad_api, "_cache_of_GroupedGemmWgradSm100Objects", {})
    monkeypatch.setattr(wgrad_api, "_wgrad_wrapper_memo", {})
    for group_k_list in ((512, 0, 1024), (2048, 1024, 1024)):
        problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=group_k_list, m=256, n=1024, a_major="m", b_major="n", initial_value=3.0)
        actual = run_wrapper(problem)
        torch.cuda.synchronize()
        assert_grouped_gemm_wgrad_close(actual, grouped_gemm_wgrad_bf16_reference(problem))
    ops = list(wgrad_api._cache_of_GroupedGemmWgradSm100Objects.values())
    assert [(op._implementation.pingpong, op._implementation.mma_tiler_mn) for op in ops] == [(True, (128, 192)), (False, (128, 256))]


def one_cluster_margin(cluster):
    """The CUDNNFE_CLUSTER_OVERLAP_MARGIN that leaves one persistent cluster: every CTA then runs all its slot's tiles."""
    return str(cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1]) - 1)


# 3 M tiles x 1 N tile per expert: 15 tiles per CTA of the one cluster (8 for MMA group 0, 7 for group 1), two
# zero-token experts and one whose 2048 tokens wrap the A/B stages several times.
PINGPONG_GROUPS = (256, 0, 768, 0, 2048)


def test_sm90_wgrad_pingpong_one_cluster(monkeypatch):
    # Pingpong runs one configuration: 128x192 tiles on (2, 1) clusters (M 640: 3 cluster M tiles per expert).
    monkeypatch.setenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", one_cluster_margin((2, 1)))
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=PINGPONG_GROUPS, m=640, n=192, a_major="m", b_major="n", initial_value=3.0)
    actual = run_wrapper(problem, mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1), schedule="pingpong")
    torch.cuda.synchronize()
    assert_grouped_gemm_wgrad_close(actual, grouped_gemm_wgrad_bf16_reference(problem))


@pytest.mark.parametrize("tile,cluster", [((128, 256), (2, 1)), ((128, 192), (1, 2)), ((128, 128), (1, 1))], ids=["128x256-c21", "128x192-c12", "128x128-c11"])
def test_sm90_wgrad_cooperative_one_cluster(monkeypatch, tile, cluster):
    # One cluster makes each CTA reuse its tile_info stages many times (the case racecheck needs).
    monkeypatch.setenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", one_cluster_margin(cluster))
    m = 640 if cluster == (2, 1) else 384
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=PINGPONG_GROUPS, m=m, n=tile[1], a_major="m", b_major="n", initial_value=3.0)
    actual = run_wrapper(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster, schedule="cooperative")
    torch.cuda.synchronize()
    assert_grouped_gemm_wgrad_close(actual, grouped_gemm_wgrad_bf16_reference(problem))


# id: (group_k_list, m, n, a_major, b_major, wgrad dtype, mma_tiler_mn, cluster_shape_mn, one cluster); None takes the
# pingpong tile (128, 192) and the (2, 1) cluster.
PINGPONG_CASES = {
    "default-ragged": ((0, 256, 512, 1024) * 4, 512, 384, "m", "n", torch.bfloat16, None, None, False),
    "kmajor-even-tiles": ((256, 0, 768, 512), 640, 192, "k", "k", torch.bfloat16, None, None, True),
    "fp32": ((512, 0, 256), 256, 576, "m", "k", torch.float32, (128, 192), (2, 1), False),
    "fp16-one-cluster": ((0, 768, 256), 640, 384, "k", "n", torch.float16, (128, 192), (2, 1), True),
}


@pytest.mark.parametrize("case", list(PINGPONG_CASES))
def test_sm90_wgrad_pingpong_matches_reference(monkeypatch, case):
    group_k_list, m, n, a_major, b_major, dtype, tile, cluster, one_cluster = PINGPONG_CASES[case]
    if one_cluster:
        monkeypatch.setenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", one_cluster_margin(cluster or (2, 1)))
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=group_k_list, m=m, n=n, a_major=a_major, b_major=b_major, output_dtype=dtype, initial_value=3.0)
    actual = run_wrapper(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster, schedule="pingpong")
    torch.cuda.synchronize()
    assert_grouped_gemm_wgrad_close(actual, grouped_gemm_wgrad_bf16_reference(problem))


@pytest.mark.parametrize("tokens", [256, 0], ids=["buffer-256", "zero-tokens"])
@pytest.mark.parametrize("schedule", ["cooperative", "pingpong"])
def test_sm90_wgrad_all_experts_empty(schedule, tokens):
    # Every expert has 0 tokens (total K = 0), over a 256-token buffer or true (M, 0) / (0, N) inputs: dW is all zeros.
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(tokens, 0, 0), m=256, n=384, a_major="m", b_major="n", initial_value=3.0)
    problem["offsets"].zero_()
    actual = run_wrapper(problem, schedule=schedule)
    torch.cuda.synchronize()
    assert actual is problem["output"]
    assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize(
    "overrides,error,match",
    [
        (dict(accumulate_on_output=True), NotImplementedError, "accumulate_on_output"),
        (dict(output_mode="discrete"), NotImplementedError, "dense output"),
        (dict(input_order="tensor_ragged"), NotImplementedError, "Tensor2D"),
        (dict(mma_tiler_mn=(256, 128)), ValueError, "Unsupported BF16 grouped GEMM wgrad configuration"),
        (dict(cluster_shape_mn=(4, 4)), ValueError, "Unsupported BF16 grouped GEMM wgrad configuration"),
        (dict(schedule="pingpong", mma_tiler_mn=(128, 256)), ValueError, "pingpong runs only"),
        (dict(schedule="pingpong", mma_tiler_mn=(128, 128)), ValueError, "pingpong runs only"),
        (dict(schedule="pingpong", mma_tiler_mn=(128, 160)), ValueError, "pingpong runs only"),
        (dict(schedule="pingpong", mma_tiler_mn=(64, 256), cluster_shape_mn=(1, 2)), ValueError, "pingpong runs only"),
        (dict(schedule="pingpong", mma_tiler_mn=(128, 192), cluster_shape_mn=(1, 2)), ValueError, "pingpong runs only"),
        (dict(schedule="stream-k"), ValueError, "SM90 schedule override must be"),
    ],
    ids=[
        "accumulate",
        "discrete",
        "ragged",
        "tile-m-256",
        "cluster-16",
        "pingpong-128x256",
        "pingpong-128x128",
        "pingpong-128x160",
        "pingpong-64x256-c12",
        "pingpong-128x192-c12",
        "schedule-name",
    ],
)
def test_sm90_wgrad_rejects_unsupported(overrides, error, match):
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(256, 256), m=256, n=256)
    if overrides.get("output_mode") == "discrete":
        ptrs = torch.tensor([t.data_ptr() for t in problem["output"]], dtype=torch.int64, device="cuda")
        overrides = dict(overrides, wgrad_tensor=None, wgrad_ptrs=ptrs)
    with pytest.raises(error, match=match):
        run_wrapper(problem, **overrides)


def test_sm90_wgrad_names_match_sm100_signatures():
    import inspect

    assert inspect.signature(cudnn.grouped_gemm_wgrad_wrapper_sm90) == inspect.signature(cudnn.grouped_gemm_wgrad_wrapper_sm100)
    for method in ("__init__", "check_support", "compile", "execute"):
        assert inspect.signature(getattr(cudnn.GroupedGemmWgradSm90, method)) == inspect.signature(getattr(cudnn.GroupedGemmWgradSm100, method))


def test_sm90_wgrad_rejects_sm100_names():
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(256, 256), m=256, n=256, a_major="m", b_major="n")
    with pytest.raises(RuntimeError, match="requires SM100"):
        cudnn.grouped_gemm_wgrad_wrapper_sm100(
            a_tensor=problem["a"],
            b_tensor=problem["b"],
            sfa_tensor=None,
            sfb_tensor=None,
            offsets_tensor=problem["offsets"],
            wgrad_tensor=problem["output"],
            acc_dtype=torch.float32,
            wgrad_dtype=problem["output_dtype"],
        )
