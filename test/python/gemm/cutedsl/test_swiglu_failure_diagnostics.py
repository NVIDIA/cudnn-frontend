# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed epilogue retains its assertion and identifies the correct GLU pair."""

import json
import pytest
import torch

from gemm.cutedsl.test_gemm_swiglu_utils import _swiglu_failure_details, check_ref_gemm_swiglu, run_gemm_swiglu_ref

pytestmark = pytest.mark.L0


def _case():
    a = torch.arange(1, 13, dtype=torch.float32).view(2, 3, 2) / 10
    b = torch.linspace(-0.5, 0.5, 128 * 3 * 2).view(128, 3, 2)
    intermediate, expected = run_gemm_swiglu_ref(a, b, 0.7)
    return a, b, intermediate, expected


@pytest.mark.parametrize("column", [0, 31, 32, 63])
def test_failure_details_keep_interleaved_input_gate_columns(column):
    a, b, intermediate, expected = _case()
    actual = expected.clone()
    actual[1, column, 1] += 10
    result = _swiglu_failure_details(a, b, intermediate, actual, expected, 0.7, atol=0.01, rtol=9e-3)
    assert result["mismatches"] == 1
    (point,) = result["points"]
    assert point["index"] == [1, column, 1]
    assert point["input_col"] == (column // 32) * 64 + column % 32
    assert point["gate_col"] == point["input_col"] + 32
    assert point["from_stored_ab12"] == pytest.approx(float(expected[1, column, 1]))
    assert point["from_fp64_dot"] == pytest.approx(float(expected[1, column, 1]), abs=1e-6)


def test_failure_diagnostic_is_bounded_and_preserves_assertion(capsys):
    a, b, intermediate, expected = _case()
    check_ref_gemm_swiglu(a, b, intermediate, expected, alpha=0.7)
    assert not capsys.readouterr().out
    with pytest.raises(AssertionError, match="Tensor-likes are not close"):
        check_ref_gemm_swiglu(a, b, intermediate, expected + 10, alpha=0.7)
    output = capsys.readouterr().out
    report = json.loads(output.split("SwiGLU output diagnostic: ", 1)[1])
    assert report["mismatches"] == expected.numel()
    assert len(report["points"]) == 8


def test_diagnostic_error_preserves_original_assertion(monkeypatch, capsys):
    import gemm.cutedsl.test_gemm_swiglu_utils as helpers

    a, b, intermediate, expected = _case()

    def broken(*args, **kwargs):
        raise RuntimeError("diagnostic probe")

    monkeypatch.setattr(helpers, "_swiglu_failure_details", broken)
    with pytest.raises(AssertionError, match="Tensor-likes are not close"):
        check_ref_gemm_swiglu(a, b, intermediate, expected + 10, alpha=0.7)
    assert "diagnostic unavailable: diagnostic probe" in capsys.readouterr().out
