# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rubin (SM107) MXFP8 wgrad keeping A in shared memory across the N tiles of a work unit, against the Rubin kernel.

The wgrad API does not pick this kernel; these tests force it in place of the Rubin kernel for dense output and
compare dW bitwise with the discrete output mode, which runs the Rubin kernel on the same inputs. They run on Rubin
only.
"""

import pytest
import torch

from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode
from cudnn.gemm.cutedsl.grouped.wgrad import _blockscaled_api
from determinism_utils import assert_bitwise, snapshot
from gemm.cutedsl.test_grouped_gemm_wgrad_global_desc import _problem, _run


@pytest.fixture(autouse=True)
def require_rubin():
    from cudnn.api_base import get_device_type

    if not torch.cuda.is_available() or get_device_type() != "rubin":
        pytest.skip("The Rubin A-reuse wgrad runs on SM107 only")


def _force_rubin_reuse(monkeypatch, a_stages=None):
    import cudnn
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse_rubin import BlockScaledMoEGroupedGemmWgradReuseRubinKernel

    picked = []
    original = _blockscaled_api.GroupedGemmWgradBlockScaledAPI._select_kernel

    def select(self):
        kernel = original(self)
        if self.weight_mode == MoEWeightMode.DENSE:
            if a_stages is None:
                kernel = BlockScaledMoEGroupedGemmWgradReuseRubinKernel
            else:

                def kernel(*args, **kwargs):
                    return BlockScaledMoEGroupedGemmWgradReuseRubinKernel(*args, a_stages=a_stages, **kwargs)

        picked.append(kernel)
        return kernel

    monkeypatch.setattr(_blockscaled_api.GroupedGemmWgradBlockScaledAPI, "_select_kernel", select)
    monkeypatch.setattr(cudnn.gemm.cutedsl.grouped.wgrad.api, "_wgrad_wrapper_memo", {})
    monkeypatch.setattr(cudnn.gemm.cutedsl.grouped.wgrad.api, "_cache_of_GroupedGemmWgradSm100Objects", {})
    return picked


# (dW N, data groups, experts): units of 2, 3 and 4 N tiles; experts longer than the A ring; experts ending half way
# into a 256-token K tile (128-token multiples) with data after them; empty experts.
CASES = [
    pytest.param(768, (256, 512, 1280, 0, 256), 5, id="n768-aligned", marks=pytest.mark.L1),
    pytest.param(512, (2048, 256, 1536, 512), 4, id="n512-long-and-short", marks=pytest.mark.L1),
    pytest.param(1024, (128, 384, 0, 640, 256), 4, id="n1024-half-k-tiles-then-data", marks=pytest.mark.L1),
    pytest.param(1536, (256, 0, 192, 128), 3, id="n1536-partial-last-expert-then-data", marks=pytest.mark.L1),
]


@pytest.mark.parametrize("n, groups, experts", CASES)
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
@pytest.mark.parametrize("a_stages", [None, 2], ids=["a5", "a2"])
def test_rubin_reuse_wgrad_matches_rubin_stock(monkeypatch, n, groups, experts, counter, a_stages):
    _force_rubin_reuse(monkeypatch, a_stages)
    problem = _problem(groups, experts, seed=13, n=n)
    reuse = _run(problem, "dense", counter=counter)
    stock = _run(problem, "discrete", counter=counter)
    assert_bitwise({"wgrad": reuse}, snapshot({"wgrad": stock}), "Rubin A reuse vs Rubin stock")
