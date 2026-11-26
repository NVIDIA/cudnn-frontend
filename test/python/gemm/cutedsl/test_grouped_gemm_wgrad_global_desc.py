# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 wgrad from global descriptors (dense output, Tensor2D input, 2-CTA 256x256 tiles in a (2, 1) cluster).

The API picks this variant of the SM100 block-scaled wgrad on its own when the experts are not short on average (for
short experts it picks the A-reuse variant, test_grouped_gemm_wgrad_reuse.py); the cases here force it. Its dW must be
the stock kernel's byte for byte, for every offsets value the stock kernel accepts: an expert whose token count is not
a multiple of the 128-token K tile reads its last K tile through its own descriptors, so tokens past its end (the next
expert's, or data past the last offset) never enter its sum. The discrete output mode runs the stock kernel on the
same inputs.
"""

import pytest
import torch

import cudnn
from cudnn.gemm.cutedsl.grouped.wgrad import _blockscaled_api
from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel
from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel
from determinism_utils import assert_bitwise, snapshot
from gemm.cutedsl.test_grouped_gemm_wgrad_utils import (
    _wgrad_assemble_scales_2d2d,
    _wgrad_create_scale_tensor,
    _wgrad_independent_reference,
    check_ref_grouped_gemm_wgrad,
)


@pytest.fixture(autouse=True)
def require_sm100():
    from cudnn.api_base import get_device_type
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")
    if torch.cuda.get_device_capability()[0] != 10 or get_device_type() == "rubin":
        pytest.skip("The global-descriptor wgrad is SM100/SM103 only")


M, N = 512, 768


def _problem(data_groups, num_experts, *, seed, n=N):
    """MXFP8 operands over ``data_groups`` token groups; the first ``num_experts`` are experts, the rest is nonzero data
    past the last offset. Scale factors are blocked per group when every group is whole K tiles but the last (the
    layout the reference reads), else random bytes of the size the API checks."""
    torch.manual_seed(seed)
    N = n
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


def _run(problem, output_mode, *, counter=False):
    output = torch.full((problem["experts"], M, problem["n"]), float("nan"), dtype=torch.bfloat16, device="cuda")
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


def force_kernel(monkeypatch, kernel_class):
    """Make the API pick ``kernel_class`` wherever it would pick the global-descriptor or A-reuse variant, on fresh
    wrapper caches; returns the list of kernel classes picked."""
    picked = []
    original = _blockscaled_api.GroupedGemmWgradBlockScaledAPI._select_kernel
    variants = (BlockScaledMoEGroupedGemmWgradDenseKernel, BlockScaledMoEGroupedGemmWgradReuseKernel)

    def select(self):
        kernel = original(self)
        if kernel_class is not None and kernel in variants:
            kernel = kernel_class
        picked.append(kernel)
        return kernel

    monkeypatch.setattr(_blockscaled_api.GroupedGemmWgradBlockScaledAPI, "_select_kernel", select)
    monkeypatch.setattr(cudnn.gemm.cutedsl.grouped.wgrad.api, "_wgrad_wrapper_memo", {})
    monkeypatch.setattr(cudnn.gemm.cutedsl.grouped.wgrad.api, "_cache_of_GroupedGemmWgradSm100Objects", {})
    return picked


@pytest.mark.L0
def test_dense_mxfp8_wgrad_picks_global_descriptors(monkeypatch):
    picked = force_kernel(monkeypatch, None)
    # 832 tokens per expert on average: more than the A-reuse variant's 6 K tiles.
    problem = _problem((1024, 640), 2, seed=1)
    _run(problem, "dense")
    _run(problem, "discrete")
    assert picked[0] is BlockScaledMoEGroupedGemmWgradDenseKernel
    assert picked[1] not in (BlockScaledMoEGroupedGemmWgradDenseKernel, BlockScaledMoEGroupedGemmWgradReuseKernel)


# (data groups, experts): partial last K tiles with nonzero data after them, empty experts, and starts that are not
# multiples of 128 (the stock kernel addresses an expert's scale factors at rows * start / 32 bytes and reads whole
# 512-byte atoms, so each case keeps those reads inside the scale-factor buffer).
CASES = [
    pytest.param((256, 0, 192, 128), 3, id="partial-last-expert-then-data", marks=pytest.mark.L0),
    pytest.param((128, 64, 0, 384, 96, 256), 5, id="partial-experts-then-data", marks=pytest.mark.L0),
    pytest.param((0, 0, 0), 3, id="all-empty", marks=pytest.mark.L1),
    pytest.param((96, 160, 32, 0, 224, 256), 5, id="unaligned-starts", marks=pytest.mark.L1),
]


@pytest.mark.parametrize("groups, experts", CASES)
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_global_descriptor_wgrad_matches_stock(monkeypatch, groups, experts, counter):
    force_kernel(monkeypatch, BlockScaledMoEGroupedGemmWgradDenseKernel)
    problem = _problem(groups, experts, seed=7)
    dense = _run(problem, "dense", counter=counter)
    stock = _run(problem, "discrete", counter=counter)
    assert_bitwise({"wgrad": dense}, snapshot({"wgrad": stock}), "global descriptors vs stock")
    if problem["reference"] is not None and all(k % 128 == 0 for k in groups[: experts - 1]):
        # Blocked scale factors match the stock addressing only when every earlier expert is whole K tiles.
        check_ref_grouped_gemm_wgrad(dense, problem["reference"])
