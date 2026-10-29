# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cmp_* epilogue ops keep IEEE comparison semantics for non-finite operands.

The stored 0/1 masks must equal torch's comparisons cell for cell: a NaN operand
compares false (true for cmp_neq), equal infinities compare equal, and finite
operands, subnormals included, stay exact. Covers an aux operand, two chain
operands (the isnan idiom cmp_neq(C, C)) and a scalar operand (the isfinite
idiom cmp_lt(abs(C), inf))."""

from __future__ import annotations

import pytest
import torch

import cudnn
import cudnn.gemm.frost  # noqa: F401  — installs the cudnn.pygraph recorder hook
from cudnn.gemm.frost.tile_config import by_name

from gemm_test_utils import Plan as _plan, requires_sm120

pytestmark = [pytest.mark.L0, requires_sm120]

_M = _N = _K = 128
_INF, _NAN = float("inf"), float("nan")
_SUBNORMAL = 1.401298464324817e-45  # smallest positive fp32 subnormal
_CMP_OPS = {"cmp_eq": torch.eq, "cmp_neq": torch.ne, "cmp_lt": torch.lt, "cmp_le": torch.le, "cmp_gt": torch.gt, "cmp_ge": torch.ge}

# Row bands of C, and what the aux X holds there (even / odd columns).
_BANDS = {
    "C=+inf, X=+inf/3e38": (0, 16),
    "C=-inf, X=-inf/-3e38": (16, 32),
    "C=NaN, X=0": (32, 48),
    "C finite, X=NaN": (48, 64),
    "C finite, X=-inf/+inf": (64, 80),
    "C finite, X=C/C+-d": (80, 112),
    "C=0, X=-sub/+sub": (112, 128),
}


def _operands():
    """A, B (as [N, K]), the exact fp32 accumulator C = A @ B and the aux X."""
    torch.manual_seed(0)
    a = torch.randint(-2, 3, (_M, _K)).float()
    b = torch.randint(-2, 3, (_N, _K)).float()
    b[:, 0] = 1.0  # a non-finite A[r, 0] then fills row r of C with that value
    a[112:] = 0.0
    acc = (a.double() @ b.double().T).float()  # exact: small integers
    for (r0, r1), v in (((0, 16), _INF), ((16, 32), -_INF), ((32, 48), _NAN)):
        a[r0:r1, 0] = v
        acc[r0:r1] = v
    odd = torch.arange(_N) % 2 == 1
    x = acc.clone()
    x[0:16] = torch.where(odd, 3e38, _INF)
    x[16:32] = torch.where(odd, -3e38, -_INF)
    x[32:48] = 0.0
    x[48:64] = _NAN
    x[64:80] = torch.where(odd, _INF, -_INF)
    x[80:112] += torch.tensor([0.0, 0.5, 0.0, -0.5, 0.0, 1.0, 0.0, -1.0]).repeat(_N // 8)
    x[112:128] = torch.where(odd, _SUBNORMAL, -_SUBNORMAL)
    return a, b, acc, x


def test_cmp_ops_follow_ieee_for_nonfinite_operands_sm120():
    """The six cmp ops, the isnan idiom and the isfinite idiom on +-inf, NaN and
    subnormal operands, dense sm120 matmul with an fp32 epilogue."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, _M, _K], stride=[_M * _K, _K, 1])
    B = g.tensor(name="B", dim=[1, _K, _N], stride=[_K * _N, 1, _K])
    C = g.matmul(A=A, B=B, name="mm")
    X = g.tensor(name="x", dim=[1, _M, _N], stride=[_M * _N, _N, 1], data_type=cudnn.data_type.FLOAT)
    inf = g.tensor(name="inf", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    ys = {op: getattr(g, op)(input=C, comparison=X, name=f"y_{op}") for op in _CMP_OPS}
    ys["isnan"] = g.cmp_neq(input=C, comparison=C, name="y_isnan")
    ys["isfinite"] = g.cmp_lt(input=g.abs(input=C, name="abs"), comparison=inf, name="y_isfinite")
    for y in ys.values():
        y.set_data_type(cudnn.data_type.FLOAT).set_output(True)
    compiled = _plan(g, config=by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"))

    a, b, acc, x = _operands()
    outs = {name: torch.full((1, _M, _N), _NAN, device="cuda") for name in ys}
    variant_pack = {
        A: a.to(torch.bfloat16).view(1, _M, _K).cuda(),
        B: b.to(torch.bfloat16).view(1, _N, _K).transpose(1, 2).cuda(),
        X: x.view(1, _M, _N).cuda(),
        inf: torch.full((1, 1, 1), _INF, device="cuda"),
    }
    variant_pack.update({ys[name]: out for name, out in outs.items()})
    compiled(variant_pack)
    torch.cuda.synchronize()

    refs = {op: fn(acc, x) for op, fn in _CMP_OPS.items()}
    refs["isnan"] = torch.isnan(acc)
    refs["isfinite"] = torch.isfinite(acc)
    bad = {}
    for name, ref in refs.items():
        got = outs[name][0].cpu()
        for band, (r0, r1) in _BANDS.items():
            n = int((got[r0:r1] != ref[r0:r1].float()).sum())
            if n:
                bad[f"{name} @ {band}"] = n
    assert not bad, f"cmp masks differ from IEEE comparisons (mismatching cells of {_N} per row): {bad}"
