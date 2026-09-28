# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 coverage for the unfused SM100 grouped GEMM wrapper."""

from __future__ import annotations

import pytest
import torch

from test_grouped_gemm_bf16_utils import (
    assert_grouped_gemm_close,
    grouped_gemm_bf16_reference,
    make_grouped_gemm_bf16_problem,
)


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("SM100 is required")


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["bf16-dense", "bf16-discrete"])
def test_grouped_gemm_bf16_wrapper(discrete):
    from cudnn import grouped_gemm_wrapper_sm100

    problem = make_grouped_gemm_bf16_problem(discrete=discrete, enable_bias=True)
    expected_c, expected_d = grouped_gemm_bf16_reference(problem)
    kwargs = dict(
        a_tensor=problem["a"],
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        bias_tensor=problem["bias"],
        prob_tensor=problem["prob"],
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        generate_c=True,
    )
    if discrete:
        kwargs.update(b_ptrs=problem["b_ptrs"], n=problem["n"], b_dtype=torch.bfloat16)
    else:
        kwargs.update(b_tensor=problem["b"])

    result = grouped_gemm_wrapper_sm100(**kwargs)
    assert_grouped_gemm_close(result["c_tensor"], expected_c)
    assert_grouped_gemm_close(result["d_tensor"], expected_d)
