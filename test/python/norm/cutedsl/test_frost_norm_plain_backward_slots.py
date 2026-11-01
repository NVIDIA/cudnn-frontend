# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LN/RMS backward on the plain route with its dgamma / dbeta spread over slots.

Rows whose width has no whole-warp vector mapping (C not a multiple of 64 in bf16 / fp16) take the plain one-CTA-per-row
backward, whose dgamma / dbeta are atomics. From 1024 rows those atomics go to a [nslots, C] partial buffer that the
finalize kernel sums; this checks every gradient on both sides of the threshold against an fp64 reference.

    pytest test/python/norm/cutedsl/test_frost_norm_plain_backward_slots.py
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm kernels require a built cudnn frontend")

from cudnn.norm._lazy import dsl_unavailable  # noqa: E402

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402
from cudnn.norm.bprop.kernels.layernorm_sm100 import _bwd_pipe_cfg, _plain_bwd_slots  # noqa: E402

_EB = {torch.bfloat16: 2, torch.float16: 2}


@pytest.mark.L0
@pytest.mark.parametrize("variant", [NormVariant.LAYER_NORM, NormVariant.RMS_NORM], ids=["layer_norm", "rms_norm"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("rows, cols", [(16, 96), (1000, 96), (40000, 96), (5000, 80), (8192, 160), (3000, 1000), (2048, 2000)])
def test_plain_route_gradients(variant, dtype, rows, cols):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the SM100 norm kernels need compute capability 10.0 or 10.3")
    assert _bwd_pipe_cfg(cols, _EB[dtype]) is None or _bwd_pipe_cfg(cols, _EB[dtype])[3] < 2, "not a plain-route width"
    assert bool(_plain_bwd_slots(rows, cols)) == (rows >= 1024), "the case sits on the wrong side of the slot threshold"
    has_beta = variant is NormVariant.LAYER_NORM
    torch.manual_seed(763)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    g = torch.randn(cols, device="cuda", dtype=dtype)
    b = torch.randn(cols, device="cuda", dtype=dtype) if has_beta else None
    dy = torch.randn_like(x)

    _, mean, rstd = norm_fprop(variant, x, g, b, eps=1e-5, normalized_shape=[cols])
    # stale bytes in the caller's buffers: the route must clear them itself
    out = tuple(torch.full(s, 7.0, device="cuda", dtype=dt) for s, dt in (((rows, cols), dtype), ((cols,), torch.float32), ((cols,), torch.float32)))
    dx, dgamma, dbeta = norm_bprop(variant, dy, x, g, mean, rstd, has_beta=has_beta, normalized_shape=[cols], out=out if has_beta else out[:2] + (None,))
    torch.cuda.synchronize()

    xr = x.double().detach().requires_grad_(True)
    gr = g.double().detach().requires_grad_(True)
    br = b.double().detach().requires_grad_(True) if has_beta else None
    yr = F.layer_norm(xr, [cols], gr, br, 1e-5) if has_beta else F.rms_norm(xr, [cols], gr, eps=1e-5)
    ref = torch.autograd.grad(yr, [t for t in (xr, gr, br) if t is not None], dy.double())
    pairs = [("dx", dx, ref[0]), ("dgamma", dgamma, ref[1])] + ([("dbeta", dbeta, ref[2])] if has_beta else [])
    for what, got, want in pairs:
        err = (got.double() - want).abs().max().item() / max(1.0, want.abs().max().item())
        assert err <= 3e-2, f"{what}: rel {err:.2e}"
