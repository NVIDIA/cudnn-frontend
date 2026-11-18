# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tall-row backward fallback accepts strided vectors and uses partial slots."""

import pytest
import torch

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")
from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402
from cudnn.norm.bprop.kernels import layernorm_sm100  # noqa: E402


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["layer", "rms"])
def test_frost_norm_layer_strided_backward_tall_rows(kind):
    torch.manual_seed(7106)
    rows, width = 1024, 128
    variant = NormVariant.LAYER_NORM if kind == "layer" else NormVariant.RMS_NORM
    x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    gamma = torch.randn(1, device="cuda", dtype=x.dtype).expand(width)
    xf = x.float()
    ms = torch.empty(rows * 2 + 1, device="cuda")
    rs = torch.empty_like(ms)
    mean = ms[1::2] if kind == "layer" else None
    rstd = rs[1::2]
    if mean is not None:
        mean.copy_(xf.mean(-1))
    rstd.copy_(torch.rsqrt((xf.var(-1, unbiased=False) if mean is not None else xf.square().mean(-1)) + 1e-5))
    layernorm_sm100._KCACHE.clear()
    dx, dg, db = norm_bprop(variant, dy, x, gamma, mean, rstd, normalized_shape=(width,), has_beta=True)
    z = (xf - mean[:, None] if mean is not None else xf) * rstd[:, None]
    d = dy.float() * gamma.float()
    dx_ref = (d - (d.mean(-1, keepdim=True) if mean is not None else 0) - z * (d * z).mean(-1, keepdim=True)) * rstd[:, None]
    torch.testing.assert_close(dx.float(), dx_ref, rtol=0.02, atol=0.02)
    torch.testing.assert_close(dg.float(), (dy.float() * z).sum(0), rtol=0.01, atol=0.08)
    torch.testing.assert_close(db.float(), dy.float().sum(0), rtol=0.01, atol=0.08)
    assert any(key[0] == "slots" for key in layernorm_sm100._KCACHE)


@pytest.mark.L0
@pytest.mark.parametrize("width", [128, 32768])
@pytest.mark.parametrize("broadcast", [False, True])
def test_frost_norm_rms_no_beta_strided(width, broadcast):
    torch.manual_seed(7107)
    x = torch.randn((2, width), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    storage = torch.randn(width, device="cuda", dtype=x.dtype)
    gamma = storage[:1].expand(width) if broadcast else storage
    y, _, rstd = norm_fprop(NormVariant.RMS_NORM, x, gamma, None, normalized_shape=(width,), eps=1e-5)
    dx, dg, db = norm_bprop(NormVariant.RMS_NORM, dy, x, gamma, None, rstd, normalized_shape=(width,), has_beta=False)
    xf = x.float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
    z = xf * r
    d = dy.float() * gamma.float()
    torch.testing.assert_close(y.float(), z * gamma.float(), rtol=0.02, atol=0.02)
    torch.testing.assert_close(dx.float(), (d - z * (d * z).mean(-1, keepdim=True)) * r, rtol=0.02, atol=0.02)
    torch.testing.assert_close(dg.float(), (dy.float() * z).sum(0), rtol=0.01, atol=0.03)
    assert db is None
