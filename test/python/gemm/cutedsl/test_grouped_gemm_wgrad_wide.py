# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 wgrad with wide tiles (256 x 512 per CTA pair, each A tile shared by both 256-column halves).

The API picks the wide kernel on its own for dense output, Tensor2D input and n % 512 == 0 at the default 2-CTA
256x256 tile. Its dW must be the stock kernel's byte for byte: the discrete output mode runs the stock kernel on the
same inputs.
"""

import pytest
import torch

import cudnn
from cudnn.gemm.cutedsl.grouped.wgrad import _bf16_api
from determinism_utils import assert_bitwise, snapshot
from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import (
    assert_grouped_gemm_wgrad_close,
    grouped_gemm_wgrad_bf16_reference,
    make_grouped_gemm_wgrad_bf16_problem,
)


@pytest.fixture(autouse=True)
def require_sm100():
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("The wide wgrad is SM100/SM103 only")


def _run(problem, output_mode, *, counter=False, tile=(256, 256), cluster=(2, 1)):
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
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
        atomic_counter_claim=counter,
    )
    torch.cuda.synchronize()
    return output


@pytest.mark.L0
def test_dense_bf16_wgrad_picks_wide_tiles(monkeypatch):
    picked = []
    original = _bf16_api.GroupedGemmWgradBf16API._wide_applies

    def record(self):
        applies = original(self)
        picked.append(applies)
        return applies

    monkeypatch.setattr(_bf16_api.GroupedGemmWgradBf16API, "_wide_applies", record)
    monkeypatch.setattr(cudnn.gemm.cutedsl.grouped.wgrad.api, "_wgrad_wrapper_memo", {})
    monkeypatch.setattr(cudnn.gemm.cutedsl.grouped.wgrad.api, "_cache_of_GroupedGemmWgradSm100Objects", {})
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(256, 512), m=512, n=1024)
    _run(problem, "dense")
    _run(problem, "discrete")
    narrow = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(256, 512), m=512, n=768)
    _run(narrow, "dense")
    assert picked == [True, False, False]


# (groups, m, n): empty experts, uneven groups, and multi-pair clusters (m a multiple of 1024: (8, 2) under CLC).
CASES = [
    pytest.param((256, 0, 1024, 256, 0, 512), 512, 1024, id="m512-n1024", marks=pytest.mark.L0),
    pytest.param((512, 0, 256, 2048, 0, 0, 768), 2048, 2048, id="m2048-n2048", marks=pytest.mark.L0),
    pytest.param((0, 0, 0), 1024, 512, id="all-empty", marks=pytest.mark.L1),
    pytest.param((1024, 256, 0, 512), 1024, 3072, id="m1024-n3072", marks=pytest.mark.L1),
]


@pytest.mark.parametrize("groups, m, n", CASES)
@pytest.mark.parametrize("counter", [False, True], ids=["clc", "counter"])
@pytest.mark.parametrize("majors", [("m", "n"), ("k", "k")], ids=["mn-major", "k-major"])
def test_wide_wgrad_matches_stock(groups, m, n, counter, majors):
    if not sum(groups) and majors != ("m", "n"):
        pytest.skip("an operand without tokens has one layout")
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=groups, m=m, n=n, a_major=majors[0], b_major=majors[1])
    wide = _run(problem, "dense", counter=counter)
    stock = _run(problem, "discrete", counter=counter)
    assert_bitwise({"wgrad": wide}, snapshot({"wgrad": stock}), "wide vs stock")
    assert_grouped_gemm_wgrad_close(wide, grouped_gemm_wgrad_bf16_reference(problem))
