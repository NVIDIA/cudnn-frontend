# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SM110 (Jetson Thor) amax-reduction dispatch for the CuTeDSL GEMM kernels (#1686).

``redux.sync.max.f32`` does not assemble on ``sm_110a`` although the block-scaled
``tcgen05`` MMA does.  Every FP32 warp amax in ``python/cudnn/gemm/cutedsl`` must
therefore go through ``cudnn._cutlass_helpers.warp_reduce.warp_amax_with_nan``
(or the Hadamard-quant helper's own ``warp_redux_sync``, which carries the same
fallback).  The source checks pin the target predicate, and the SM110 device
checks exercise every lane including negative values, NaNs and infinities.
The grouped / dense / discrete GEMM suites cover the enclosing kernels.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import pytest
import torch

pytest.importorskip("cutlass")

import cudnn._cutlass_helpers.warp_reduce as warp_reduce  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402
from cudnn.gemm.cutedsl.grouped.glu_hadamard_quant import moe_kernel_helpers as hadamard  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GEMM_ROOT = _REPO_ROOT / "python" / "cudnn" / "gemm" / "cutedsl"
_HELPER = _REPO_ROOT / "python" / "cudnn" / "_cutlass_helpers" / "warp_reduce.py"
_HADAMARD_HELPER = _GEMM_ROOT / "grouped" / "glu_hadamard_quant" / "moe_kernel_helpers.py"

_DIRECT_FMAX_REDUX = re.compile(r"cute\.arch\.warp_redux_sync\((?:[^)]|\n)*?kind=\"fmax\"", re.MULTILINE)
_DIRECT_REDUX_F32_ASM = re.compile(r"redux\.sync\.[a-z.]*f32")


def _gemm_sources():
    return sorted(p for p in _GEMM_ROOT.rglob("*.py"))


@pytest.mark.L0
def test_no_direct_fp32_redux_outside_the_shared_helper():
    offenders = []
    for path in _gemm_sources():
        text = path.read_text(encoding="utf-8")
        if _DIRECT_FMAX_REDUX.search(text):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], f"FP32 fmax redux must go through warp_amax_with_nan (SM110 has no redux.f32): {offenders}"


@pytest.mark.L0
def test_redux_f32_inline_asm_only_in_the_hadamard_helper():
    offenders = []
    for path in _gemm_sources():
        if path == _HADAMARD_HELPER:
            continue
        if _DIRECT_REDUX_F32_ASM.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], offenders


@pytest.mark.L0
def test_shared_helper_keeps_both_forms():
    text = _HELPER.read_text(encoding="utf-8")
    assert 'kind="fmax"' in text and "shuffle_sync_bfly" in text
    hadamard = _HADAMARD_HELPER.read_text(encoding="utf-8")
    assert "redux.sync.max.abs.NaN.f32" in hadamard and "shuffle_sync_bfly" in hadamard
    assert "target_needs_shuffle_amax()" in hadamard


@pytest.mark.L0
@pytest.mark.parametrize(
    ("major", "minor", "expected"),
    [(11, 0, True), (10, 0, False), (10, 3, False), (10, 7, False), (12, 0, False), (9, 0, False)],
)
def test_target_predicate_selects_the_shuffle_only_on_sm110(major, minor, expected):
    arch = mock.Mock(major=major, minor=minor)
    dsl = mock.Mock()
    dsl.get_arch_enum.return_value = arch
    with mock.patch("cutlass.cutlass_dsl.CuTeDSL._get_dsl", return_value=dsl):
        assert warp_reduce.target_needs_shuffle_amax() is expected
        assert warp_reduce._target_needs_shuffle_amax() is expected


@pytest.mark.L0
def test_grouped_helper_reexports_the_shared_names():
    from cudnn.gemm.cutedsl.grouped import moe_kernel_helpers as grouped

    assert grouped.warp_amax_with_nan is warp_reduce.warp_amax_with_nan
    assert grouped._target_needs_shuffle_amax is warp_reduce.target_needs_shuffle_amax


@cute.kernel
def _amax_kernel(x: cute.Tensor, out: cute.Tensor, absolute: cutlass.Constexpr):
    lane = cute.arch.thread_idx()[0]
    if absolute:
        out[lane] = hadamard.warp_redux_sync(x[lane], hadamard.ReduxKind.MAX, 0xFFFFFFFF, nan=True)
    else:
        out[lane] = warp_reduce.warp_amax_with_nan(x[lane])


@cute.jit
def _amax(x: cute.Tensor, out: cute.Tensor, absolute: cutlass.Constexpr):
    _amax_kernel(x, out, absolute).launch(grid=[1, 1, 1], block=[32, 1, 1])


@pytest.mark.L0
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("case", ["finite", "negative", "nan", "infinity"])
def test_sm110_amax_all_lanes(case, absolute):
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Exercises the native SM110 shuffle target")
    x = torch.arange(32, device="cuda", dtype=torch.float32) - 16
    if case == "negative":
        x = -x.abs() - 1
    elif case == "nan":
        x[17] = float("nan")
    elif case == "infinity":
        x[3], x[29] = -float("inf"), float("inf")
    out = torch.full_like(x, -7)
    x_cute, out_cute = from_dlpack(x), from_dlpack(out)
    compiled = cute.compile(_amax, x_cute, out_cute, absolute)
    compiled(x_cute, out_cute)
    torch.cuda.synchronize()
    expected = (x.abs() if absolute else x).amax().expand_as(out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
