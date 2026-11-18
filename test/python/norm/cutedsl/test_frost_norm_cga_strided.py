# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CGA norm kernels must accept strided read-only vectors through TVM FFI."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")
from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402
from cudnn.norm.bprop.kernels import groupnorm_cga_sm100 as bwd_kernel  # noqa: E402
from cudnn.norm.fprop.kernels import groupnorm_cga_sm100 as fwd_kernel  # noqa: E402


def _setup(kind):
    if kind == "group":
        return NormVariant.GROUP_NORM, (2, 64, 16, 32), 4, {"num_groups": 4}
    return NormVariant.INSTANCE_NORM, (1, 8, 32, 128), 8, {}


def _views(storage, n):
    # Nonzero offsets exercise element-aligned (not necessarily 16-byte-aligned) inputs.
    return (storage[:1].expand(n), storage[1 : n + 1], storage[1::2][:n], storage[1::3][:n])


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["group", "instance"])
@pytest.mark.parametrize("operand", ["gamma", "beta", "both"])
def test_frost_norm_cga_forward_strided_vectors(kind, operand):
    torch.manual_seed(6102)
    variant, shape, groups, kwargs = _setup(kind)
    fwd_kernel._KCACHE.clear()
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gs = _views(torch.randn(shape[1] * 3 + 1, device="cuda", dtype=x.dtype), shape[1])
    bs = _views(torch.randn(shape[1] * 3 + 1, device="cuda", dtype=x.dtype), shape[1])
    for i in (1, 0, 2, 3, 1):
        gamma = gs[i if operand in ("gamma", "both") else 1]
        beta = bs[i if operand in ("beta", "both") else 1]
        y, mean, rstd = norm_fprop(variant, x, gamma, beta, eps=1e-5, **kwargs)
        ref = F.group_norm(x.float(), groups, gamma.float(), beta.float(), 1e-5)
        rows = x.float().reshape(shape[0], groups, -1)
        torch.testing.assert_close(y.float(), ref, rtol=0.02, atol=0.02)
        torch.testing.assert_close(mean.reshape_as(rows.mean(-1)), rows.mean(-1), rtol=0.001, atol=0.002)
        torch.testing.assert_close(rstd.reshape_as(rows.mean(-1)), torch.rsqrt(rows.var(-1, unbiased=False) + 1e-5), rtol=0.001, atol=0.002)
        assert len(fwd_kernel._KCACHE) == 1


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["group", "instance"])
@pytest.mark.parametrize("operand", ["gamma", "mean", "rstd", "all"])
def test_frost_norm_cga_backward_strided_vectors(kind, operand):
    torch.manual_seed(6103)
    variant, shape, groups, kwargs = _setup(kind)
    bwd_kernel._KCACHE.clear()
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    count, rows = shape[1], shape[0] * groups
    gs = _views(torch.randn(count * 3 + 1, device="cuda", dtype=x.dtype), count)
    ms = _views(torch.randn(rows * 3 + 1, device="cuda") * 0.3, rows)
    rs = _views(torch.rand(rows * 3 + 1, device="cuda") * 0.4 + 0.8, rows)
    for i in (1, 0, 2, 3, 1):
        gamma = gs[i if operand in ("gamma", "all") else 1]
        mean = ms[i if operand in ("mean", "all") else 1]
        rstd = rs[i if operand in ("rstd", "all") else 1]
        dx, dg, db = norm_bprop(variant, dy, x, gamma, mean, rstd, has_beta=True, **kwargs)
        z = (x.float().reshape(shape[0], groups, -1) - mean.reshape(shape[0], groups, 1)) * rstd.reshape(shape[0], groups, 1)
        d = (dy.float() * gamma.float().reshape(1, count, 1, 1)).reshape(shape[0], groups, -1)
        dx_ref = (d - d.mean(-1, keepdim=True) - z * (d * z).mean(-1, keepdim=True)) * rstd.reshape(shape[0], groups, 1)
        torch.testing.assert_close(dx.float(), dx_ref.reshape_as(x), rtol=0.02, atol=0.02)
        torch.testing.assert_close(dg.float(), (dy.float() * z.reshape_as(x)).sum((0, 2, 3)), rtol=0.01, atol=0.06)
        torch.testing.assert_close(db.float(), dy.float().sum((0, 2, 3)), rtol=0.01, atol=0.06)
        assert len(bwd_kernel._KCACHE) == 1
