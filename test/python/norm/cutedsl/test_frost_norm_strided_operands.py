# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache reuse must not change the meaning of a strided affine operand."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")
from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402


def _views(t):
    # Both start aligned, while one has a dynamic non-unit stride.
    return t[:1280], t[::2]


@pytest.mark.L0
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("strided", ["gamma", "beta", "both"])
def test_frost_norm_forward_affine_stride_cache(reverse, strided):
    from cudnn.norm.fprop.kernels import groupnorm_sm100

    groupnorm_sm100._KCACHE.clear()
    torch.manual_seed(3151)
    x = torch.randn((2, 1280, 8, 8), device="cuda", dtype=torch.bfloat16)
    gs = _views(torch.randn(2560, device="cuda", dtype=x.dtype))
    bs = _views(torch.randn(2560, device="cuda", dtype=x.dtype))
    for i in ([1, 0] if reverse else [0, 1]):
        gamma = gs[i if strided in ("gamma", "both") else 0]
        beta = bs[i if strided in ("beta", "both") else 0]
        y, _, _ = norm_fprop(NormVariant.GROUP_NORM, x, gamma, beta, num_groups=32, eps=1e-5)
        ref = F.group_norm(x.float(), 32, gamma.float(), beta.float(), 1e-5)
        torch.testing.assert_close(y.float(), ref, rtol=0.02, atol=0.02)


@pytest.mark.L0
@pytest.mark.parametrize("reverse", [False, True])
def test_frost_norm_backward_scale_stride_cache(reverse):
    from cudnn.norm.bprop.kernels import groupnorm_fast_sm100

    groupnorm_fast_sm100._KCACHE.clear()
    torch.manual_seed(3151)
    x = torch.randn((2, 1280, 8, 8), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    gs = _views(torch.randn(2560, device="cuda", dtype=x.dtype))
    grouped = x.float().reshape(2, 32, -1)
    mean = grouped.mean(-1).reshape(-1)
    rstd = torch.rsqrt(grouped.var(-1, unbiased=False) + 1e-5).reshape(-1)
    for i in ([1, 0] if reverse else [0, 1]):
        gamma = gs[i]
        dx, _, _ = norm_bprop(NormVariant.GROUP_NORM, dy, x, gamma, mean, rstd, num_groups=32, has_beta=True)
        xr = x.float().detach().requires_grad_()
        ref = F.group_norm(xr, 32, gamma.float(), None, 1e-5)
        (dx_ref,) = torch.autograd.grad(ref, (xr,), dy.float())
        torch.testing.assert_close(dx.float(), dx_ref, rtol=0.02, atol=0.02)


@pytest.mark.L0
def test_frost_norm_forward_reuses_dynamic_stride_signature():
    from cudnn.norm.fprop.kernels import groupnorm_sm100

    groupnorm_sm100._KCACHE.clear()
    torch.manual_seed(3151)
    x = torch.randn((2, 1280, 8, 8), device="cuda", dtype=torch.bfloat16)
    g_storage = torch.randn(3840, device="cuda", dtype=x.dtype)
    b_storage = torch.randn_like(g_storage)
    for step in (2, 3, 2):
        gamma, beta = g_storage[::step][:1280], b_storage[::step][:1280]
        y, _, _ = norm_fprop(NormVariant.GROUP_NORM, x, gamma, beta, num_groups=32, eps=1e-5)
        ref = F.group_norm(x.float(), 32, gamma.float(), beta.float(), 1e-5)
        torch.testing.assert_close(y.float(), ref, rtol=0.02, atol=0.02)
        assert len(groupnorm_sm100._KCACHE) == 1


@pytest.mark.L0
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("strided", ["mean", "rstd", "both"])
def test_frost_norm_backward_stats_stride_cache(reverse, strided):
    from cudnn.norm.bprop.kernels import groupnorm_fast_sm100

    groupnorm_fast_sm100._KCACHE.clear()
    torch.manual_seed(3151)
    x = torch.randn((2, 1280, 8, 8), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    gamma = torch.randn(1280, device="cuda", dtype=x.dtype)
    grouped = x.float().reshape(2, 32, -1)
    mean = grouped.mean(-1).reshape(-1)
    rstd = torch.rsqrt(grouped.var(-1, unbiased=False) + 1e-5).reshape(-1)
    mean_storage = torch.empty(2 * mean.numel(), device="cuda", dtype=mean.dtype)
    rstd_storage = torch.empty_like(mean_storage)
    mean_storage[::2].copy_(mean)
    rstd_storage[::2].copy_(rstd)
    means, rstds = (mean, mean_storage[::2]), (rstd, rstd_storage[::2])
    xr = x.float().detach().requires_grad_()
    ref = F.group_norm(xr, 32, gamma.float(), None, 1e-5)
    (dx_ref,) = torch.autograd.grad(ref, (xr,), dy.float())
    for i in ([1, 0] if reverse else [0, 1]):
        m = means[i if strided in ("mean", "both") else 0]
        r = rstds[i if strided in ("rstd", "both") else 0]
        dx, _, _ = norm_bprop(NormVariant.GROUP_NORM, dy, x, gamma, m, r, num_groups=32, has_beta=True)
        torch.testing.assert_close(dx.float(), dx_ref, rtol=0.02, atol=0.02)


@pytest.mark.L0
def test_frost_norm_strided_affine_cuda_graph_replay():
    torch.manual_seed(3151)
    x = torch.randn((2, 1280, 8, 8), device="cuda", dtype=torch.bfloat16)
    g_storage = torch.randn(2560, device="cuda", dtype=x.dtype)
    b_storage = torch.randn_like(g_storage)
    gamma, beta = g_storage[::2], b_storage[::2]
    out = (torch.empty_like(x), torch.empty(64, device="cuda"), torch.empty(64, device="cuda"))
    norm_fprop(NormVariant.GROUP_NORM, x, gamma, beta, num_groups=32, eps=1e-5, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        norm_fprop(NormVariant.GROUP_NORM, x, gamma, beta, num_groups=32, eps=1e-5, out=out)
    for _ in range(2):
        gamma.copy_(torch.randn_like(gamma))
        beta.copy_(torch.randn_like(beta))
        graph.replay()
        ref = F.group_norm(x.float(), 32, gamma.float(), beta.float(), 1e-5)
        torch.testing.assert_close(out[0].float(), ref, rtol=0.02, atol=0.02)
