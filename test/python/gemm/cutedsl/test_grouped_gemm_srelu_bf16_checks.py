# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression checks for the BF16 grouped GEMM numerical test oracle."""

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_srelu_bf16_utils import assert_dprob_within_bound


@pytest.mark.L0
@pytest.mark.parametrize("operand", ["candidate", "reference", "bound"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "negative-inf"])
def test_dprob_bound_rejects_nonfinite(operand, value):
    values = {name: torch.zeros(2) for name in ("candidate", "reference", "bound")}
    values[operand][0] = value
    with pytest.raises(AssertionError, match=f"non-finite {operand}"):
        assert_dprob_within_bound("dprob", **values, routed=2)


@pytest.mark.L0
def test_dprob_bound_rejects_negative_allowance():
    with pytest.raises(AssertionError, match="negative summation bound"):
        assert_dprob_within_bound("dprob", torch.ones(1), torch.ones(1), torch.tensor([-0.25]), routed=1)


@pytest.mark.L0
def test_dprob_bound_enforces_finite_error_and_ignores_unrouted_rows():
    reference = torch.tensor([1.0, 0.0, float("nan")])
    bound = torch.tensor([0.25, 0.0, float("nan")])
    candidate = torch.tensor([1.25, 0.0, float("nan")])
    assert_dprob_within_bound("dprob", candidate, reference, bound, routed=2)
    candidate[0] = 1.5
    with pytest.raises(AssertionError, match="summation bound"):
        assert_dprob_within_bound("dprob", candidate, reference, bound, routed=2)
