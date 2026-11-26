# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 wgrad keeping A in shared memory across the N tiles of a work unit (dense output, Tensor2D input, 2-CTA
256x256 tiles in a (2, 1) cluster).

The API picks this variant of the SM100 block-scaled wgrad on its own when dW has more than one N tile per work unit
and the experts are short on average: the token extent of A over the expert count is at most 6 K tiles of 128 tokens.
A work unit is 2, 3 or 4 N tiles (the largest divisor of dW's N tile count up to 4). An expert with at most 6 K tiles
loads A once per unit; a longer one reloads A per N tile. Its dW must be the stock kernel's byte for byte, for every
offsets value the stock kernel accepts; the discrete output mode runs the stock kernel on the same inputs.
"""

import pytest
import torch

import cudnn
from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel
from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel
from determinism_utils import assert_bitwise, snapshot
from gemm.cutedsl.test_grouped_gemm_wgrad import (
    _GLOBAL_DESC_M as M,
    _global_desc_force_kernel as force_kernel,
    _global_desc_problem as _problem,
    _global_desc_run as _run,
)
from gemm.cutedsl.test_grouped_gemm_wgrad_utils import check_ref_grouped_gemm_wgrad


@pytest.fixture(autouse=True)
def require_sm100():
    from cudnn.api_base import get_device_type
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")
    if torch.cuda.get_device_capability()[0] != 10 or get_device_type() == "rubin":
        pytest.skip("The A-reuse wgrad is SM100/SM103 only")


@pytest.mark.L0
def test_selection_rule():
    """Host-side rule: more than one N tile per unit and at most 6 x 128 tokens per expert on average."""
    reuse = BlockScaledMoEGroupedGemmWgradReuseKernel
    assert [reuse.n_group_for(n) for n in (256, 512, 768, 1024, 1280, 1536, 3072)] == [1, 2, 3, 4, 1, 3, 4]
    fits = dict(ab_bits=8, c_bits=16, has_global_scale=False)
    assert reuse.can_implement(512, 768, tokens=768 * 4, experts=4, **fits)
    assert not reuse.can_implement(512, 768, tokens=768 * 4 + 1, experts=4, **fits)
    assert not reuse.can_implement(512, 256, tokens=128, experts=4, **fits)  # one N tile
    assert not reuse.can_implement(512, 1280, tokens=128, experts=4, **fits)  # 5 N tiles: units of one
    assert not reuse.can_implement(512, 768, tokens=0, experts=0, **fits)
    assert reuse.can_implement(512, 768, tokens=0, experts=3, **fits)
    assert not reuse.can_implement(512, 768, tokens=128, experts=4, ab_bits=8, c_bits=32, has_global_scale=False)


@pytest.mark.L0
def test_api_picks_reuse_for_short_experts(monkeypatch):
    picked = force_kernel(monkeypatch, None)
    short = _problem((256, 0, 384), 3, seed=1)
    long = _problem((1024, 896), 2, seed=2)
    one_tile = _problem((256, 0, 384), 3, seed=3, n=256)
    _run(short, "dense")
    _run(long, "dense")  # same shapes but the token count: a second compiled entry
    _run(short, "dense")  # cached
    _run(one_tile, "dense")
    _run(short, "discrete")
    assert picked[:3] == [BlockScaledMoEGroupedGemmWgradReuseKernel, BlockScaledMoEGroupedGemmWgradDenseKernel, BlockScaledMoEGroupedGemmWgradDenseKernel]
    assert picked[3] not in (BlockScaledMoEGroupedGemmWgradDenseKernel, BlockScaledMoEGroupedGemmWgradReuseKernel)


# (dW N, data groups, experts). N 512 / 768 / 1024 / 1536: units of 2, 3, 4 N tiles and two units of 3.
CASES = [
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


@pytest.mark.parametrize("n, groups, experts", CASES)
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_reuse_wgrad_matches_stock(monkeypatch, n, groups, experts, counter):
    picked = force_kernel(monkeypatch, BlockScaledMoEGroupedGemmWgradReuseKernel)
    problem = _problem(groups, experts, seed=11, n=n)
    reuse = _run(problem, "dense", counter=counter)
    stock = _run(problem, "discrete", counter=counter)
    assert picked[0] is BlockScaledMoEGroupedGemmWgradReuseKernel
    assert_bitwise({"wgrad": reuse}, snapshot({"wgrad": stock}), "A reuse vs stock")
    if problem["reference"] is not None and all(k % 128 == 0 for k in groups[: experts - 1]):
        check_ref_grouped_gemm_wgrad(reuse, problem["reference"])


@pytest.mark.L0
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
def test_reuse_wgrad_pdl_options(counter):
    """use_pdl and launch_pdl_dependents_at_entry through the class API: same dW as without them."""
    problem = _problem((256, 0, 384, 128), 3, seed=5, n=768)
    outputs = []
    for options in ({}, dict(use_pdl=True), dict(use_pdl=True, launch_pdl_dependents_at_entry=True)):
        output = torch.full((problem["experts"], M, problem["n"]), float("nan"), dtype=torch.bfloat16, device="cuda")
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
    stock = _run(problem, "discrete", counter=counter)
    for output in outputs:
        assert_bitwise({"wgrad": output}, snapshot({"wgrad": stock}), "A reuse with PDL options vs stock")
