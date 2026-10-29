# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU pins of the gated block's shared OUTPUT bound (``gated_block_bounds.py``), the validator the QSA forward suites and the backward
suite hold their bf16 / f16 outputs to.  No GPU and no kernel: the assertion ITSELF is the subject, so a future edit to the validator is
testable on its own -- both dtype boundaries exactly (one ulp beyond refused), empty and all-zero tensors, a non-finite cell in either
operand, a planted single-element excursion whose global cosine still passes, a pure dtype rounding accepted, the chunked fp64 cosine
against one plain fp64 reduction, the typed errors, and noncontiguous views."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gated_block_bounds as gbb  # noqa: E402
from gated_block_bounds import (  # noqa: E402
    OUTPUT_ATOL_FRAC,
    OUTPUT_COS_MIN,
    OUTPUT_RTOL,
    assert_output_close,
    cos64,
    format_output_stats,
    output_bound_stats,
)

pytestmark = pytest.mark.L0

DTYPES = pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


@DTYPES
def test_exact_boundary_passes_and_one_ulp_beyond_fails(dt):
    """|d| == atol_frac * max|ref| + rtol * |ref| is INSIDE the bound (ratio exactly 1.0); the next representable value is outside.  Planted
    at a NEGATIVE cell that is not the max, so both terms contribute (max|ref| = 1 sits elsewhere) and |ref| is read as an absolute."""
    ref = torch.tensor([[1.0, 0.25, -0.5, 0.0], [0.5, -0.25, 0.0, 0.125]])  # powers of two: exact in both dtypes
    e = (0, 2)
    bound_e = OUTPUT_ATOL_FRAC[dt] * 1.0 + OUTPUT_RTOL[dt] * 0.5
    got = ref.to(dt)
    got[e] = ref[e] - bound_e
    assert float(got[e]) == -0.5 - bound_e  # representable, so the ratio below is exactly 1 and not a rounding of it
    s = assert_output_close(got, ref, f"{dt} boundary")
    assert s["bound_ratio"] == 1.0 and s["n_above_bound"] == 0 and s["worst"]["index"] == list(e)
    assert (s["rtol"], s["atol_frac"]) == (OUTPUT_RTOL[dt], OUTPUT_ATOL_FRAC[dt])  # the dtype of GOT picks the bound
    assert "worst cell 1.000 of the bound" in format_output_stats("x", s)
    beyond = got.clone()
    beyond[e] = torch.nextafter(got[e], torch.tensor(-2.0, dtype=dt))
    s = output_bound_stats(beyond, ref)
    assert s["bound_ratio"] > 1.0 and s["n_above_bound"] == 1 and s["worst"]["index"] == list(e)
    with pytest.raises(AssertionError, match="1 of 8 elements above"):
        assert_output_close(beyond, ref, f"{dt} one ulp beyond")


@DTYPES
def test_empty_and_all_zero_tensors(dt):
    """Two empty tensors and two all-zero tensors agree exactly (cos defined as 1.0, ratio 0); an all-zero reference admits NO difference
    (its bound is 0 everywhere: ratio inf), and an all-zero result against a live reference is refused."""
    for shape in ((0,), (0, 8)):
        s = assert_output_close(torch.zeros(shape, dtype=dt), torch.zeros(shape), "empty")
        assert (s["n_elements"], s["cos"], s["bound_ratio"], s["n_above_bound"], s["worst"]) == (0, 1.0, 0.0, 0, None)
    s = assert_output_close(torch.zeros(4, 8, dtype=dt), torch.zeros(4, 8), "all zero")
    assert s["cos"] == 1.0 and s["bound_ratio"] == 0.0 and s["max_ref"] == 0.0
    got = torch.zeros(4, 8, dtype=dt)
    got[1, 2] = 2.0**-10  # one tiny live cell against a dead reference
    s = output_bound_stats(got, torch.zeros(4, 8))
    assert math.isinf(s["bound_ratio"]) and s["n_above_bound"] == 1 and math.isnan(s["cos"])
    with pytest.raises(AssertionError):
        assert_output_close(got, torch.zeros(4, 8), "live cell vs dead ref")
    ref = torch.full((4, 8), 0.75)
    with pytest.raises(AssertionError):
        assert_output_close(torch.zeros(4, 8, dtype=dt), ref, "dead result vs live ref")
    assert output_bound_stats(torch.zeros(4, 8, dtype=dt), ref)["bound_ratio"] == pytest.approx(1.0 / (OUTPUT_ATOL_FRAC[dt] + OUTPUT_RTOL[dt]))


@DTYPES
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "-inf"])
def test_non_finite_values_are_refused(dt, bad):
    """A non-finite cell in the RESULT fails the finite check; one in the REFERENCE poisons the bound and the cosine (both NaN), so the
    assertion fails there too -- never a silent pass, whichever operand carries it."""
    ref = _randn(16, 8)
    got = ref.to(dt)
    bad_got = got.clone()
    bad_got[3, 5] = bad
    assert output_bound_stats(bad_got, ref)["finite"] is False
    with pytest.raises(AssertionError, match="non-finite"):
        assert_output_close(bad_got, ref, "non-finite result")
    bad_ref = ref.clone()
    bad_ref[3, 5] = bad
    s = output_bound_stats(got, bad_ref)
    assert s["finite"] is True and not (s["bound_ratio"] <= 1.0) and not (s["cos"] >= OUTPUT_COS_MIN)
    with pytest.raises(AssertionError):
        assert_output_close(got, bad_ref, "non-finite reference")


@DTYPES
def test_single_element_excursion_is_refused_despite_a_passing_cosine(dt):
    """ONE wrong cell in 2^18 moves the global cosine by ~2e-6 -- far above the 0.999 floor -- and is caught only by the per-element bound:
    the reason the suites hold both."""
    ref = _randn(512, 512, seed=1)
    got = ref.to(dt)
    e = (300, 7)
    got[e] = ref[e] + 1.0  # about 8x the bf16 bound at max|ref| ~ 5
    s = output_bound_stats(got, ref)
    assert s["cos"] > OUTPUT_COS_MIN and s["n_above_bound"] == 1 and s["bound_ratio"] > 1.0 and s["worst"]["index"] == list(e)
    with pytest.raises(AssertionError, match="1 of 262144 elements above"):
        assert_output_close(got, ref, "planted excursion")


@DTYPES
def test_pure_dtype_rounding_of_the_reference_is_accepted(dt):
    """Casting the fp32 oracle to the output dtype is the floor of what any kernel can do: a half ulp per cell is at most 1/6 of the bound
    (rtol = 2 ulp(1), atol_frac = ulp(1) of max|ref|), so the ratio reads WELL below 1 and never sits on the edge."""
    ref = _randn(64, 256, seed=2) * 3.0
    s = assert_output_close(ref.to(dt), ref, "pure rounding")
    assert 0.0 < s["bound_ratio"] < 0.25 and s["n_above_bound"] == 0 and s["cos"] > 0.9999 and s["finite"]


def test_chunked_cosine_matches_an_independent_fp64_reduction():
    """``cos64`` reduces in fp64 chunks (a 32k-token O never materializes as one fp64 copy); on a tensor THREE elements longer than one
    chunk -- the partial last chunk and the cross-chunk accumulation both run -- it agrees with one plain fp64 reduction to 1e-9."""
    n = gbb._CHUNK + 3
    a = torch.linspace(-1.0, 1.0, n).to(torch.bfloat16)
    b = (torch.linspace(-1.0, 1.0, n) * 0.5 + 0.25).to(torch.bfloat16)
    a64, b64 = a.double(), b.double()
    expect = float(a64 @ b64) / math.sqrt(float(a64 @ a64) * float(b64 @ b64))
    del a64, b64
    assert 0.5 < expect < 0.999 and abs(cos64(a, b) - expect) <= 1e-9
    assert abs(cos64(a[:1000], b[:1000]) - float(a[:1000].double() @ b[:1000].double()) / (a[:1000].double().norm() * b[:1000].double().norm())) <= 1e-12


@DTYPES
def test_shape_mismatch_and_unsupported_dtype_raise_value_error(dt):
    """Both are ValueErrors raised BEFORE any statistic (never an AssertionError that reads like a numerics failure): a shape mismatch, and a
    result dtype without a calibrated bound (fp32 / fp64) unless rtol AND atol_frac are passed explicitly."""
    ref = _randn(4, 8)
    with pytest.raises(ValueError, match="shape mismatch"):
        output_bound_stats(torch.zeros(4, 9, dtype=dt), ref)
    with pytest.raises(ValueError, match="shape mismatch"):
        assert_output_close(torch.zeros(8, 4, dtype=dt), ref, "transposed shape")
    for other in (torch.float32, torch.float64):
        with pytest.raises(ValueError, match="no output bound is defined"):
            output_bound_stats(ref.to(other), ref)
        with pytest.raises(ValueError, match="no output bound is defined"):
            output_bound_stats(ref.to(other), ref, rtol=OUTPUT_RTOL[dt])  # one explicit term is not enough
    s = assert_output_close(ref.to(torch.float64), ref, "explicit bounds", rtol=OUTPUT_RTOL[dt], atol_frac=OUTPUT_ATOL_FRAC[dt])
    assert s["bound_ratio"] == 0.0 and (s["rtol"], s["atol_frac"]) == (OUTPUT_RTOL[dt], OUTPUT_ATOL_FRAC[dt])


@DTYPES
def test_noncontiguous_views_give_the_statistics_of_their_contiguous_copies(dt):
    """A transposed view and two strided slices (all non-contiguous) report the SAME statistics -- the worst index in the VIEW's shape, the
    same fp64 cosine -- as their contiguous copies: the helper reads logical elements, never storage order."""
    ref = _randn(32, 48, seed=3)
    got = ref.to(dt)
    got[5, 7] = ref[5, 7] + 0.5  # one planted excursion so the worst cell is well defined
    for g, r in ((got.t(), ref.t()), (got[:, ::3], ref[:, ::3]), (got[1::2, 2:30], ref[1::2, 2:30])):
        assert not g.is_contiguous()
        assert output_bound_stats(g, r) == output_bound_stats(g.contiguous(), r.contiguous())
    assert output_bound_stats(got.t(), ref.t())["worst"]["index"] == [7, 5]
