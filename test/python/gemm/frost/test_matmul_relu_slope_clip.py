# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ReLU with a negative slope together with a clip, as cuDNN defines it, through
the epilogue emitter both arch families share:

  RELU_FWD: y = x < lower ? lower + slope * (x - lower) : x, then min(y, upper)
  RELU_BWD: dx = dy * (x <= lower ? slope : (x < upper ? 1 : 0))

with lower defaulting to 0. lower_clip is where the slope starts, not a clamp
applied after it, and the backward slope never reaches x >= upper_clip. The
emitter used to compute max(leaky(x), lower) -- plain ReLU for the natural
relu(negative_slope=s, lower_clip=0.0) -- and gave x >= upper the gradient
slope * dy. One dense config per family; the one the active GPU does not run is
skipped.
"""

from __future__ import annotations

import cudnn
import pytest
import torch

from gemm_test_utils import (
    Plan as _plan,
    requires_matmul_gpu,
    skip_unless_pipeline_active,
    vp as _vp,
)
from cudnn.gemm.frost.tile_config import by_name

pytestmark = [pytest.mark.L0, requires_matmul_gpu]

_CONFIGS = {
    "sm100": "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "sm120": "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
}
_M = _N = _K = 128

_FWD_CASES = (
    dict(negative_slope=0.1, lower_clip=0.0),  # leaky ReLU spelled with its default lower clip
    dict(negative_slope=0.1, lower_clip=-1.0),
    dict(negative_slope=0.1, lower_clip=0.5, upper_clip=2.0),
)
_BWD_CASES = (
    dict(negative_slope=0.1, upper_clip=1.0),
    dict(negative_slope=0.1, lower_clip=-1.0, upper_clip=1.0),
)


def _case_id(attrs):
    return "-".join(f"{k}{v}" for k, v in attrs.items())


def _matmul_graph():
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, _M, _K], stride=[_M * _K, _K, 1])
    B = g.tensor(name="B", dim=[1, _K, _N], stride=[_K * _N, 1, _K])
    return g, g.matmul(A=A, B=B, name="mm")


def _compile(g, family):
    cfg = by_name(_CONFIGS[family])
    skip_unless_pipeline_active(cfg)
    return _plan(g, config=cfg)


def _operands():
    torch.manual_seed(0)
    a = torch.randn(1, _M, _K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(1, _N, _K, device="cuda", dtype=torch.bfloat16)
    return a, b, torch.einsum("bmk,bnk->bmn", a.float(), b.float())


@pytest.mark.parametrize("attrs", _FWD_CASES, ids=_case_id)
@pytest.mark.parametrize("family", sorted(_CONFIGS))
def test_relu_slope_starts_at_lower_clip(family, attrs):
    g, C = _matmul_graph()
    y = g.relu(input=C, name="r", **attrs)
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    compiled = _compile(g, family)

    a, b, acc = _operands()
    out = torch.full((1, _M, _N), float("nan"), device="cuda", dtype=torch.bfloat16)
    compiled(_vp(compiled, a, b, out))
    torch.cuda.synchronize()

    slope, lower, upper = attrs["negative_slope"], attrs.get("lower_clip", 0.0), attrs.get("upper_clip")
    ref = torch.where(acc < lower, lower + slope * (acc - lower), acc)
    if upper is not None:
        ref = torch.clamp(ref, max=upper)
    torch.testing.assert_close(out.float(), ref.to(torch.bfloat16).float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("attrs", _BWD_CASES, ids=_case_id)
@pytest.mark.parametrize("family", sorted(_CONFIGS))
def test_relu_backward_slope_stops_at_upper_clip(family, attrs):
    g, C = _matmul_graph()
    x = g.tensor(name="x", dim=[1, _M, _N], stride=[_M * _N, _N, 1], data_type=cudnn.data_type.FLOAT)
    dx = g.relu_backward(loss=C, input=x, name="rb", **attrs)
    dx.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    compiled = _compile(g, family)

    a, b, acc = _operands()
    xin = torch.randn(1, _M, _N, device="cuda") * 2.0
    out = torch.full((1, _M, _N), float("nan"), device="cuda", dtype=torch.bfloat16)
    compiled(_vp(compiled, a, b, out, xin))
    torch.cuda.synchronize()

    slope, lower, upper = attrs["negative_slope"], attrs.get("lower_clip", 0.0), attrs["upper_clip"]
    ref = acc * torch.where(xin <= lower, slope, (xin < upper).float())
    torch.testing.assert_close(out.float(), ref.to(torch.bfloat16).float(), atol=1e-2, rtol=1e-2)
