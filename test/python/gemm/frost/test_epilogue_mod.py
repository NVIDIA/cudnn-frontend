# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The epilogue MOD op is C fmod (cuDNN's floating-point remainder; torch.fmod).

fmod is exact, so the dense SM120 and SM100 templates are checked bitwise, signed zeros
included, on the operands an emission of a - b * trunc(a / b) gets wrong: quotients that
round up to an integer (k / 0.1f), |a / b| >= 2**23, an infinite divisor and a quotient
that overflows. The sm100-family sweep in test_matmul_epilogue_fusion.py draws quotients
below ~10 and compares within a tolerance, so it cannot tell the two forms apart.
"""

from __future__ import annotations

import pytest
import torch

from gemm_test_utils import Plan, requires_sm100, requires_sm120, vp

pytestmark = pytest.mark.L0

import cudnn
import cudnn.gemm.frost  # noqa: F401  — installs the cudnn.pygraph recorder hook
from cudnn.gemm.frost.tile_config import by_name

# The emitter is shared, so each family's dense template runs the same cases.
_CONFIGS = [
    pytest.param("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", marks=requires_sm120, id="sm120"),
    pytest.param("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma", marks=requires_sm100, id="sm100"),
]
_M = _N = _K = 128
_INF, _NAN = float("inf"), float("nan")


def _mod_graph(aux_is_dividend: bool):
    """C = A @ B, then mod(C, x) or mod(x, C) with x a per-element fp32 aux; fp32 output."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, _M, _K], stride=[_M * _K, _K, 1])
    B = g.tensor(name="B", dim=[1, _K, _N], stride=[_K * _N, 1, _K])
    C = g.matmul(A=A, B=B, name="mm")
    x = g.tensor(name="x", dim=[1, _M, _N], stride=[_M * _N, _N, 1], data_type=cudnn.data_type.FLOAT)
    y = g.mod(input0=x, input1=C) if aux_is_dividend else g.mod(input0=C, input1=x)
    y.set_data_type(cudnn.data_type.FLOAT).set_output(True)
    return g


def _bf16(values) -> torch.Tensor:
    """``values`` rounded to bf16, as fp32."""
    return torch.as_tensor(values, dtype=torch.float32).to(torch.bfloat16).float()


def _cases(aux_is_dividend: bool) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """(label, GEMM values, fp32 aux values); the GEMM values are bf16-exact and finite."""
    if aux_is_dividend:  # mod(x, C)
        return [
            ("x mod bf16(2 pi), |x / C| up to 2**24", torch.full((256,), 6.28125), torch.linspace(1e6, 1e8, 256)),
            ("|x / C| >= 2**23, x / C overflows", torch.tensor([-3.0, 0.5, 2.0**-100]), torch.tensor([2.0**25, 3e38, 1e38])),
            ("x = +-inf, NaN, -0", torch.full((4,), 3.0), torch.tensor([_INF, -_INF, _NAN, -0.0])),
            ("small quotients", _bf16(torch.linspace(0.5, 1.5, 128)), torch.linspace(-8, 8, 128)),
        ]
    k = torch.arange(1, 257, dtype=torch.float32)
    big = torch.tensor([2.0**25, 3 * 2.0**24, 5 * 2.0**27, 7 * 2.0**30, 2.0**40, 3 * 2.0**50])
    return [
        ("k / 0.1f rounds up to 10 k", torch.cat([k, -k[:64]]), torch.full((320,), 0.1)),
        ("|C / x| >= 2**23", big.repeat_interleave(4), torch.tensor([-3.0, 0.7, 1.3, 2.9]).repeat(6)),
        ("x = +-inf", torch.tensor([3.0, -2.5, 100.0, 0.0]).repeat_interleave(2), torch.tensor([_INF, -_INF]).repeat(4)),
        ("exact negative multiples give -0", torch.tensor([-4.0, -6.0, -2.0]), torch.tensor([2.0, 3.0, 0.5])),
        ("C / x overflows", _bf16([1e38, -1e38]), torch.full((2,), 2.0**-100)),
        ("C mod 2 pi", _bf16(torch.linspace(1e6, 1e8, 256)), torch.full((256,), 6.2831855)),
        ("x = 0 gives NaN", torch.tensor([5.0, 0.0]), torch.zeros(2)),
        ("small quotients", _bf16(torch.linspace(-8, 8, 128)), torch.linspace(0.5, 1.5, 128)),
    ]


@pytest.mark.parametrize("aux_is_dividend", [False, True], ids=["mod_C_x", "mod_x_C"])
@pytest.mark.parametrize("config", _CONFIGS)
def test_epilogue_mod_is_exact_fmod(config, aux_is_dividend):
    plan = Plan(_mod_graph(aux_is_dividend), config=by_name(config))
    cases = _cases(aux_is_dividend)
    assert all(len(gemm_vals) == len(aux_vals) for _, gemm_vals, aux_vals in cases)
    # The remaining positions hold (1, 1), whose fmod is 0.
    sizes = [(label, len(gemm_vals)) for label, gemm_vals, _ in cases]
    sizes.append(("(1, 1) filler", _M * _N - sum(n for _, n in sizes)))
    c = torch.cat([gemm_vals for _, gemm_vals, _ in cases] + [torch.ones(sizes[-1][1])])
    x = torch.cat([aux_vals for _, _, aux_vals in cases] + [torch.ones(sizes[-1][1])])
    assert torch.equal(c.to(torch.bfloat16).float(), c)
    # With A = I every dot has one nonzero product, so C is exactly B: C[m, n] = Bt[n, m].
    a = torch.eye(_M, _K, device="cuda", dtype=torch.bfloat16).view(1, _M, _K)
    bt = c.view(_M, _N).t().contiguous().to(torch.bfloat16).view(1, _N, _K).cuda()
    out = torch.full((1, _M, _N), -12345.0, device="cuda")
    plan(vp(plan, a, bt, out, x.view(1, _M, _N).cuda()))
    torch.cuda.synchronize()

    dividend, divisor = (x, c) if aux_is_dividend else (c, x)
    ref = torch.fmod(dividend.double(), divisor.double()).float()  # exact: an fp32 fmod is representable
    got = out.flatten().cpu()
    bad = ~((got.view(torch.int32) == ref.view(torch.int32)) | (got.isnan() & ref.isnan()))
    report, start = [], 0
    for label, n in sizes:
        idx = bad[start : start + n].nonzero().flatten() + start
        if len(idx):
            j = int(idx[0])
            report.append(
                f"{label}: {len(idx)} of {n} differ, e.g. mod({dividend[j].item()!r}, {divisor[j].item()!r}) = {got[j].item()!r}, fmod = {ref[j].item()!r}"
            )
        start += n
    assert not report, f"{int(bad.sum())} of {got.numel()} differ from fmod:\n  " + "\n  ".join(report)
