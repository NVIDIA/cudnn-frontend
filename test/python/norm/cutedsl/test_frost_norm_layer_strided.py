# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LayerNorm/RMSNorm accepts strided affine and saved-statistic vectors."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")
from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402


def _views(storage, n):
    # The last three slices also exercise input addresses that are not 16-byte aligned.
    return (storage[:n], storage[:1].expand(n), storage[1 : n + 1], storage[1::2][:n], storage[1::3][:n])


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["layer", "rms"])
@pytest.mark.parametrize("width", [128, 32768])
@pytest.mark.parametrize("operand", ["gamma", "beta", "both"])
def test_frost_norm_layer_forward_strided_vectors(kind, width, operand):
    torch.manual_seed(7101)
    variant = NormVariant.LAYER_NORM if kind == "layer" else NormVariant.RMS_NORM
    x = torch.randn((3, width), device="cuda", dtype=torch.bfloat16)
    gs = _views(torch.randn(width * 3 + 1, device="cuda", dtype=x.dtype), width)
    bs = _views(torch.randn(width * 3 + 1, device="cuda", dtype=x.dtype), width)
    for i in (0, 1, 2, 3, 4, 0):
        gamma = gs[i if operand in ("gamma", "both") else 0]
        beta = bs[i if operand in ("beta", "both") else 0]
        y, _, rstd = norm_fprop(variant, x, gamma, beta, normalized_shape=(width,), eps=1e-5)
        if kind == "layer":
            ref = F.layer_norm(x.float(), (width,), gamma.float(), beta.float(), 1e-5)
        else:
            ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5) * gamma.float() + beta.float()
        torch.testing.assert_close(y.float(), ref, rtol=0.02, atol=0.02)
        assert torch.isfinite(rstd).all()


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["layer", "rms"])
@pytest.mark.parametrize("width", [128, 32768])
@pytest.mark.parametrize("operand", ["gamma", "mean", "rstd", "all"])
def test_frost_norm_layer_backward_strided_vectors(kind, width, operand):
    if kind == "rms" and operand == "mean":
        pytest.skip("RMSNorm has no mean input")
    torch.manual_seed(7102)
    variant = NormVariant.LAYER_NORM if kind == "layer" else NormVariant.RMS_NORM
    x = torch.randn((3, width), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    gs = _views(torch.randn(width * 3 + 1, device="cuda", dtype=x.dtype), width)
    ms = _views(torch.randn(10, device="cuda") * 0.3, 3)
    rs = _views(torch.rand(10, device="cuda") * 0.4 + 0.8, 3)
    for i in (0, 1, 2, 3, 4, 0):
        gamma = gs[i if operand in ("gamma", "all") else 0]
        mean = ms[i if operand in ("mean", "all") else 0] if kind == "layer" else None
        rstd = rs[i if operand in ("rstd", "all") else 0]
        dx, dg, db = norm_bprop(variant, dy, x, gamma, mean, rstd, normalized_shape=(width,), has_beta=True)
        centered = x.float() - mean[:, None] if mean is not None else x.float()
        z = centered * rstd[:, None]
        d = dy.float() * gamma.float()
        dx_ref = (d - (d.mean(-1, keepdim=True) if mean is not None else 0) - z * (d * z).mean(-1, keepdim=True)) * rstd[:, None]
        torch.testing.assert_close(dx.float(), dx_ref, rtol=0.02, atol=0.02)
        torch.testing.assert_close(dg.float(), (dy.float() * z).sum(0), rtol=0.01, atol=0.03)
        torch.testing.assert_close(db.float(), dy.float().sum(0), rtol=0.01, atol=0.03)


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["layer", "rms"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_frost_norm_layer_strided_additional_dtypes(kind, dtype):
    torch.manual_seed(7105)
    width = 256
    variant = NormVariant.LAYER_NORM if kind == "layer" else NormVariant.RMS_NORM
    x = torch.randn((2, width), device="cuda", dtype=dtype)
    dy = torch.randn_like(x)
    gs = torch.randn(width * 2 + 1, device="cuda", dtype=dtype)
    bs = torch.randn_like(gs)
    gamma, beta = gs[:1].expand(width), bs[1::2][:width]
    y, _, _ = norm_fprop(variant, x, gamma, beta, normalized_shape=(width,), eps=1e-5)
    xf = x.float()
    mean = xf.mean(-1) if kind == "layer" else None
    rstd = torch.rsqrt((xf.var(-1, unbiased=False) if mean is not None else xf.square().mean(-1)) + 1e-5)
    dx, dg, db = norm_bprop(variant, dy, x, gamma, mean, rstd, normalized_shape=(width,), has_beta=True)
    xr = xf.detach().requires_grad_()
    gr = gamma.float().detach().requires_grad_()
    br = beta.float().detach().requires_grad_()
    ref = F.layer_norm(xr, (width,), gr, br, 1e-5) if mean is not None else xr * torch.rsqrt(xr.square().mean(-1, keepdim=True) + 1e-5) * gr + br
    dx_ref, dg_ref, db_ref = torch.autograd.grad(ref, (xr, gr, br), dy.float())
    for got, expected in ((y.float(), ref), (dx.float(), dx_ref), (dg.float(), dg_ref), (db.float(), db_ref)):
        torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
