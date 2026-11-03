# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 backward's mismatch allowance bounds both the count and the magnitude."""

import pytest
import torch

from sdpa.mxfp8 import assert_close_mxfp8_grad, compare_amax

pytestmark = pytest.mark.L0


def _outputs():
    expected = torch.zeros((1, 4, 1, 128))
    expected[0, 3, 0, 0] = 100.0  # A non-maximum error leaves the existing amax check unchanged.
    return expected.clone(), expected


@pytest.mark.parametrize("delta", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("whole_row", [False, True])
def test_bounded_sparse_and_row_deviations_are_accepted(delta, whole_row):
    actual, expected = _outputs()
    actual[0, 0, 0, : (None if whole_row else 1)] = delta
    assert_close_mxfp8_grad(actual, expected, 0.125, 0.20, tag="dK", keys=4096)


@pytest.mark.parametrize("delta", [0.501, 8.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["actual", "expected"])
def test_one_unbounded_nonmaximum_error_is_rejected(delta, side):
    actual, expected = _outputs()
    (actual if side == "actual" else expected)[0, 0, 0, 0] = delta
    if delta == 8.0:
        assert compare_amax(actual, expected, rtol=0.04, tag="dK")
    with pytest.raises(AssertionError):
        assert_close_mxfp8_grad(actual, expected, 0.125, 0.20, tag="dK", keys=4096)


def test_matching_nonfinite_values_are_rejected():
    actual, expected = _outputs()
    actual[0, 0, 0, 0] = expected[0, 0, 0, 0] = float("inf")
    with pytest.raises(AssertionError):
        assert_close_mxfp8_grad(actual, expected, 0.125, 0.20, tag="dK", keys=4096)


def test_broad_bounded_corruption_still_exceeds_the_count_budget():
    actual, expected = _outputs()
    actual[:, :3] = 0.25
    with pytest.raises(AssertionError):
        assert_close_mxfp8_grad(actual, expected, 0.125, 0.20, tag="dK", keys=1)


def test_ordinary_relative_tolerance_is_preserved():
    expected = torch.full((1, 4, 1, 128), 100.0)
    actual = expected + 1.0  # Above the budget cap, but inside the original elementwise tolerance.
    assert_close_mxfp8_grad(actual, expected, 0.125, 0.20, tag="dK", keys=1)
