# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""relu_backward at cuDNN's default parameters, with an inf or NaN gradient on
dead units, through the SM120 dense FROST template.

The default RELU_BWD (lower clip 0, slope 0, no upper clip) is the threshold
select dx = x <= 0 ? +0 : dy, as torch's threshold_backward is: the native cuDNN
engines return +0 at x <= 0 whatever dy is. The shared epilogue emitter used to
multiply dy by a 0/1 mask, so an inf or NaN dy on a dead unit -- an fp16 GEMM
output above 65504, say -- came out NaN on the FROST plan only. A NaN x passes dy
on both. The clipped and sloped forms multiply on the native engines as well and
are not covered here.
"""

from __future__ import annotations

import cudnn
import pytest
import torch

from gemm_test_utils import Plan as _plan, requires_sm120, vp as _vp
from cudnn.gemm.frost.tile_config import by_name

pytestmark = [pytest.mark.L0, requires_sm120]

_CONFIG = "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"
_M = _N = _K = 128
_INF, _NAN = float("inf"), float("nan")
_DY = (_INF, -_INF, _NAN, -1.5, 1.5)  # row r of the GEMM output is exactly _DY[r % 5]
_X = (-1.0, 0.0, -0.0, 0.5, 2.0, _NAN, -_INF, _INF)  # column n of x is _X[n % 8]

# Spellings of the same default RELU_BWD: for fp32 compute the frontend lowers an unset lower clip, slope and
# upper clip as 0, 0 and FLT_MAX (include/cudnn_frontend/node/pointwise.h).
_DEFAULT_RELU_BWD = {
    "relu_backward": lambda g, dy, x: g.relu_backward(loss=dy, input=x, name="rb"),
    "lower_clip_0": lambda g, dy, x: g.relu_backward(loss=dy, input=x, lower_clip=0.0, name="rb"),
    "negative_slope_0": lambda g, dy, x: g.relu_backward(loss=dy, input=x, negative_slope=0.0, name="rb"),
    "upper_clip_flt_max": lambda g, dy, x: g.relu_backward(loss=dy, input=x, upper_clip=torch.finfo(torch.float32).max, name="rb"),
    "leaky_relu_backward_0": lambda g, dy, x: g.leaky_relu_backward(loss=dy, input=x, negative_slope=0.0, name="rb"),
}


def _graph(build, c_dtype=None):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, _M, _K], stride=[_M * _K, _K, 1])
    B = g.tensor(name="B", dim=[1, _K, _N], stride=[_K * _N, 1, _K])
    C = g.matmul(A=A, B=B, name="mm")
    if c_dtype is not None:
        C.set_data_type(c_dtype)
    x = g.tensor(name="x", dim=[1, _M, _N], stride=[_M * _N, _N, 1], data_type=cudnn.data_type.FLOAT)
    build(g, C, x).set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return _plan(g, config=by_name(_CONFIG))


def _run(compiled, a, b, x):
    out = torch.full((1, _M, _N), 7.0, device="cuda", dtype=torch.bfloat16)  # nonzero, so an unwritten dead unit cannot pass
    compiled(_vp(compiled, a, b, out, x))
    torch.cuda.synchronize()
    return out


def _assert_threshold_select(out, dy, x):
    dead = x <= 0
    got = out[dead].float()
    bad = got[(got != 0) | torch.signbit(got)]
    assert bad.numel() == 0, (
        f"{bad.numel()} of {int(dead.sum())} dead units are not +0: {int(bad.isnan().sum())} NaN, "
        f"{int((bad == 0).sum())} -0, {int(((bad != 0) & ~bad.isnan()).sum())} other"
    )
    ref = torch.where(dead, torch.zeros_like(dy), dy).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("spelling", list(_DEFAULT_RELU_BWD))
def test_sm120_relu_backward_zeroes_nonfinite_dy_on_dead_units(spelling):
    """dy = +-inf, NaN, -1.5 and 1.5 against x = -1, 0, -0, 0.5, 2, NaN, -inf and
    inf: every dead unit is exactly +0 and every live one (NaN x included) is dy."""
    compiled = _graph(_DEFAULT_RELU_BWD[spelling])

    a = torch.zeros(1, _M, _K, device="cuda", dtype=torch.bfloat16)
    a[0, :, 0] = torch.tensor([_DY[r % len(_DY)] for r in range(_M)], device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(1, _N, _K, device="cuda", dtype=torch.bfloat16)
    b[0, :, 0] = 1.0
    x = torch.tensor([_X[n % len(_X)] for n in range(_N)], device="cuda").expand(1, _M, _N).contiguous()
    dy = a[0, :, :1].float().expand(_M, _N).unsqueeze(0)

    _assert_threshold_select(_run(compiled, a, b, x), dy, x)


def test_sm120_relu_backward_fp16_overflow_on_dead_units():
    """dy need not arrive non-finite: with the GEMM output declared fp16, finite
    accumulators of +-90000 round to +-inf before relu_backward."""
    compiled = _graph(_DEFAULT_RELU_BWD["relu_backward"], c_dtype=cudnn.data_type.HALF)

    torch.manual_seed(0)
    a = torch.zeros(1, _M, _K, device="cuda", dtype=torch.bfloat16)
    a[0, :, 0] = 300.0
    a[0, 1::2, 0] = -300.0
    a[0, ::4, 0] = 1.0  # every fourth row stays in fp16 range
    b = torch.zeros(1, _N, _K, device="cuda", dtype=torch.bfloat16)
    b[0, :, 0] = 300.0
    x = torch.randn(1, _M, _N, device="cuda")
    dy = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).half().float()
    assert int(torch.isinf(dy[x <= 0]).sum()) > 0

    _assert_threshold_select(_run(compiled, a, b, x), dy, x)
