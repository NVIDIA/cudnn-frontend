# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward-save to explicit-backward integration coverage for HSTU LMSD."""

import pytest
import torch

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

from cudnn.hstu_lmsd import hstu_lmsd_backward, hstu_lmsd_forward

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_IS_SM10X = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _backward_reference(dy, x, u, weight, bias, mean, rstd, mask, p):
    d = x.shape[1]
    scale = 1.0 / (1.0 - p)
    xf = x.float()
    uf = u.float()
    wf = weight.float()
    bf = bias.float()
    dy0, dy1, dy2 = (part.float() for part in dy.split(d, dim=1))
    mask_i32 = mask.to(torch.int32)
    zero = torch.zeros((), device=x.device)
    direct_du = torch.where((mask_i32 & 4) != 0, dy0 * scale, zero)
    direct_dx = torch.where((mask_i32 & 2) != 0, dy1 * scale, zero)
    fused_dy = torch.where((mask_i32 & 1) != 0, dy2 * scale, zero)

    xhat = (xf - mean[:, None]) * rstd[:, None]
    ln = xhat * wf + bf
    sigmoid = torch.sigmoid(uf)
    silu = uf * sigmoid
    dsilu = sigmoid + silu * (1.0 - sigmoid)

    dln = fused_dy * silu
    dweight = torch.sum(dln * xhat, dim=0)
    dbias = torch.sum(dln, dim=0)
    weighted = dln * wf
    mean_weighted = torch.mean(weighted, dim=1, keepdim=True)
    mean_xhat_weighted = torch.mean(xhat * weighted, dim=1, keepdim=True)
    dx = direct_dx + (
        weighted - mean_weighted - xhat * mean_xhat_weighted
    ) * rstd[:, None]
    du = (fused_dy * ln + direct_du) * dsilu
    return dx, du, dweight, dbias


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_forward_saved_tensors_feed_explicit_backward():
    torch.manual_seed(2026)
    n, d, p = 513, 512, 0.1
    x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
    u = u_storage[:, :d]
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn((n, 3 * d), device="cuda", dtype=torch.bfloat16)

    forward = hstu_lmsd_forward(
        x, u, weight, bias, eps=1e-6, dropout_ratio=p, seed=29
    )
    y, mean, rstd, mask = forward
    actual = hstu_lmsd_backward(
        dy,
        x,
        u,
        weight,
        bias,
        mean,
        rstd,
        mask,
        dropout_ratio=p,
    )
    expected = _backward_reference(
        dy, x, u, weight, bias, mean, rstd, mask, p
    )
    tolerances = (
        (2.5e-2, 2.5e-2),
        (2.5e-2, 2.5e-2),
        (3.5e-2, 5.0e-1),
        (3.5e-2, 5.0e-1),
    )
    for got, ref, (rtol, atol) in zip(actual, expected, tolerances):
        torch.testing.assert_close(got.float(), ref, rtol=rtol, atol=atol)
