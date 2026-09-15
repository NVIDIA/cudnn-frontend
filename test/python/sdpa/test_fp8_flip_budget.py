# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-test for ``sdpa.fp8.assert_close_fp8_grad``'s single-flip structure check.

One flipped fp8 intermediate (a P or dS code that the kernel and the reference rounded to
neighbouring e4m3 values) moves ONE output d-row by ``ulp * operand_row`` -- a rank-1 update along
a single Q / K / V / dO row whose step is a power of two times the intermediate's descale.  The
sm107 212-SM CI lane produced exactly that on test_mhas_v2 fp8_bwd test310 (dK row +-0.5 from a
negative-score q row of amplitude 8), above the fixed ``4 * atol`` row cap that was calibrated on
amplitude-4 rows.  These cases pin what the structure check accepts (one flip) and what it must keep
rejecting (a garbage row, a non-power-of-two step, a row fed by two flips, and everything when the
caller does not pass an operand).
"""

import pytest
import torch

from sdpa.fp8 import assert_close_fp8_grad
from sdpa.helpers import create_sparse_int_tensor, inject_negative_score_rows

pytestmark = pytest.mark.L0

ATOL, RTOL = 0.08, 0.2


def _dk_problem(seed=0, b=1, s=512, h=1, d=64):
    """A dK-shaped output built the way the reference builds it: dK[j] = sum_i dS[i, j] * Q[i] * dP_descale.

    Returns (q, dk, dp_descale, negative-score q rows, j) where j is the dK row of smallest magnitude, so the
    perturbations below are judged against ~atol rather than swallowed by rtol * |dK| -- the suite's real dK
    rows are mostly near zero for the same reason (sparse-int data, dS ~ 0 off the diagonal band)."""
    rng = torch.Generator(device="cuda")
    rng.manual_seed(seed)
    q = create_sparse_int_tensor((b, s, h, d), torch.float, rng)
    k = create_sparse_int_tensor((b, s, h, d), torch.float, rng)
    inject_negative_score_rows(q, k, rng, attn_scale=d**-0.5, head_axis=2)
    ds = create_sparse_int_tensor((b, h, s, s), torch.float, rng, sparsity=0.98)  # stands in for the fp8 dS codes
    dp_descale = 2.0
    dk = torch.einsum("bhij,bihd->bjhd", ds, q) * dp_descale
    neg_rows = (q.abs().amax(-1) == q.abs().max()).nonzero()
    j = tuple(int(x) for x in torch.unravel_index(dk.abs().amax(-1).flatten().argmin(), dk.shape[:-1]))
    return q, dk, dp_descale, [tuple(r.tolist()) for r in neg_rows], j


def _flip(dk, q, dp_descale, i, j, log2_ulp):
    out = dk.clone()
    out[j[0], j[1], j[2]] += (2.0**log2_ulp) * dp_descale * q[i]
    return out


def test_one_flipped_ds_code_is_accepted_when_the_operand_is_given():
    q, dk, dp_descale, neg, j = _dk_problem()
    actual = _flip(dk, q, dp_descale, neg[0], j, -3)
    assert (actual - dk).abs().max().item() > 4 * ATOL, "the case must exceed the fixed row cap to exercise the structure check"
    assert_close_fp8_grad(actual, dk, ATOL, RTOL, tag="dK", keys=dk.shape[1], operand=q, flip_unit=dp_descale)


def test_the_fixed_cap_still_governs_without_an_operand():
    q, dk, dp_descale, neg, j = _dk_problem()
    actual = _flip(dk, q, dp_descale, neg[0], j, -3)
    with pytest.raises(AssertionError):
        assert_close_fp8_grad(actual, dk, ATOL, RTOL, tag="dK", keys=dk.shape[1])


def test_a_non_power_of_two_step_is_rejected():
    q, dk, dp_descale, neg, j = _dk_problem()
    actual = dk.clone()
    actual[j] += 0.7 * dp_descale * q[neg[0]]
    with pytest.raises(AssertionError):
        assert_close_fp8_grad(actual, dk, ATOL, RTOL, tag="dK", keys=dk.shape[1], operand=q, flip_unit=dp_descale)


def test_a_row_fed_by_two_operand_rows_is_rejected():
    q, dk, dp_descale, neg, j = _dk_problem()
    i2 = tuple((q.abs().amax(-1) == 2.0).nonzero()[0].tolist())
    actual = _flip(dk, q, dp_descale, neg[0], j, -3)
    actual[j] += (2.0**-1) * dp_descale * q[i2]
    with pytest.raises(AssertionError):
        assert_close_fp8_grad(actual, dk, ATOL, RTOL, tag="dK", keys=dk.shape[1], operand=q, flip_unit=dp_descale)


def test_a_garbage_row_is_rejected():
    q, dk, dp_descale, _, j = _dk_problem()
    actual = dk.clone()
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    actual[j] += torch.randn(dk.shape[-1], generator=g, device="cuda")
    with pytest.raises(AssertionError):
        assert_close_fp8_grad(actual, dk, ATOL, RTOL, tag="dK", keys=dk.shape[1], operand=q, flip_unit=dp_descale)
