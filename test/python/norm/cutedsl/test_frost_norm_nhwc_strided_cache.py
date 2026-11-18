# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Channels-last norm cache signatures must track unit-stride specialization."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")
from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402


def _case(variant, backward=False):
    if backward:
        from cudnn.norm.bprop.kernels import groupnorm_nhwc_sm100 as gn
        from cudnn.norm.bprop.kernels import instancenorm_nhwc_sm100 as inn
    else:
        from cudnn.norm.fprop.kernels import groupnorm_nhwc_sm100 as gn
        from cudnn.norm.fprop.kernels import instancenorm_nhwc_sm100 as inn

    if variant == "group":
        assert gn.eligible(128, 32, 2)
        return NormVariant.GROUP_NORM, 4, gn
    assert inn.nhwc_cfg(128, 2) is not None
    return NormVariant.INSTANCE_NORM, 128, inn


def _kw(variant, groups):
    return {"num_groups": groups} if variant == "group" else {}


def _views(t, length):
    return t[:length], t[::2][:length]


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["group", "instance"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("strided", ["gamma", "beta", "both"])
def test_frost_norm_nhwc_forward_stride_cache(variant, reverse, strided):
    kind, groups, module = _case(variant)
    module._KCACHE.clear()
    torch.manual_seed(3152)
    x = torch.randn((2, 128, 8, 8), device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    gs = _views(torch.randn(256, device="cuda", dtype=x.dtype), 128)
    bs = _views(torch.randn(256, device="cuda", dtype=x.dtype), 128)
    for i in ([1, 0] if reverse else [0, 1]):
        gamma = gs[i if strided in ("gamma", "both") else 0]
        beta = bs[i if strided in ("beta", "both") else 0]
        y, _, _ = norm_fprop(kind, x, gamma, beta, eps=1e-5, **_kw(variant, groups))
        assert module._KCACHE, "expected channels-last kernel was not used"
        ref = F.group_norm(x.float(), groups, gamma.float(), beta.float(), 1e-5)
        torch.testing.assert_close(y.float(), ref, rtol=0.02, atol=0.02)


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["group", "instance"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("strided", ["gamma", "mean", "rstd", "all"])
def test_frost_norm_nhwc_backward_stride_cache(variant, reverse, strided):
    kind, groups, module = _case(variant, backward=True)
    module._KCACHE.clear()
    torch.manual_seed(3152)
    x = torch.randn((2, 128, 8, 8), device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    dy = torch.randn_like(x)
    gs = _views(torch.randn(256, device="cuda", dtype=x.dtype), 128)
    grouped = x.float().reshape(2, groups, -1)
    mean = grouped.mean(-1).reshape(-1)
    rstd = torch.rsqrt(grouped.var(-1, unbiased=False) + 1e-5).reshape(-1)
    mean_storage = torch.empty(2 * mean.numel(), device="cuda", dtype=mean.dtype)
    rstd_storage = torch.empty_like(mean_storage)
    mean_storage[::2].copy_(mean)
    rstd_storage[::2].copy_(rstd)
    means, rstds = (mean, mean_storage[::2]), (rstd, rstd_storage[::2])
    for i in ([1, 0] if reverse else [0, 1]):
        gamma = gs[i if strided in ("gamma", "all") else 0]
        m = means[i if strided in ("mean", "all") else 0]
        r = rstds[i if strided in ("rstd", "all") else 0]
        dx, _, _ = norm_bprop(kind, dy, x, gamma, m, r, has_beta=True, **_kw(variant, groups))
        assert module._KCACHE, "expected channels-last kernel was not used"
        xr = x.float().detach().requires_grad_()
        ref = F.group_norm(xr, groups, gamma.float(), None, 1e-5)
        (dx_ref,) = torch.autograd.grad(ref, (xr,), dy.float())
        torch.testing.assert_close(dx.float(), dx_ref, rtol=0.02, atol=0.02)


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["group", "instance"])
def test_frost_norm_nhwc_dynamic_stride_cache_reuse(variant):
    kind, groups, module = _case(variant)
    module._KCACHE.clear()
    torch.manual_seed(3152)
    x = torch.randn((2, 128, 8, 8), device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    gs = torch.randn(384, device="cuda", dtype=x.dtype)
    bs = torch.randn_like(gs)
    count = None
    for step in (2, 3, 2):
        gamma, beta = gs[::step][:128], bs[::step][:128]
        y, _, _ = norm_fprop(kind, x, gamma, beta, eps=1e-5, **_kw(variant, groups))
        ref = F.group_norm(x.float(), groups, gamma.float(), beta.float(), 1e-5)
        torch.testing.assert_close(y.float(), ref, rtol=0.02, atol=0.02)
        count = len(module._KCACHE) if count is None else count
        assert count > 0 and len(module._KCACHE) == count


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["group", "instance"])
def test_frost_norm_nhwc_strided_graph_replay(variant):
    from cudnn.norm.fprop.api import fprop_workspace_bytes
    from cudnn.norm.utils import Workspace

    kind, groups, _ = _case(variant)
    torch.manual_seed(3152)
    x = torch.randn((2, 128, 8, 8), device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    gs = torch.randn(256, device="cuda", dtype=x.dtype)
    bs = torch.randn_like(gs)
    gamma, beta = gs[::2], bs[::2]
    out = (torch.empty_like(x), torch.empty(2 * groups, device="cuda"), torch.empty(2 * groups, device="cuda"))
    nbytes = fprop_workspace_bytes(kind, x.shape, "bf16", nhwc=True, **_kw(variant, groups))
    ws = Workspace(torch.empty(nbytes, device="cuda", dtype=torch.uint8))
    norm_fprop(kind, x, gamma, beta, eps=1e-5, out=out, ws=ws, **_kw(variant, groups))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        norm_fprop(kind, x, gamma, beta, eps=1e-5, out=out, ws=ws, **_kw(variant, groups))
    for _ in range(2):
        gamma.copy_(torch.randn_like(gamma))
        beta.copy_(torch.randn_like(beta))
        graph.replay()
        ref = F.group_norm(x.float(), groups, gamma.float(), beta.float(), 1e-5)
        torch.testing.assert_close(out[0].float(), ref, rtol=0.02, atol=0.02)


def _broadcast_and_strided(storage, count):
    return storage[:1].expand(count), storage[::2][:count]


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["group", "instance"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("strided", ["gamma", "beta", "both"])
def test_frost_norm_nhwc_forward_broadcast_cache(variant, reverse, strided):
    kind, groups, module = _case(variant)
    module._KCACHE.clear()
    torch.manual_seed(3152)
    x = torch.randn((2, 128, 8, 8), device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    gs = _broadcast_and_strided(torch.randn(256, device="cuda", dtype=x.dtype), 128)
    bs = _broadcast_and_strided(torch.randn(256, device="cuda", dtype=x.dtype), 128)
    for i in ([1, 0] if reverse else [0, 1]):
        gamma = gs[i if strided in ("gamma", "both") else 1]
        beta = bs[i if strided in ("beta", "both") else 1]
        y, _, _ = norm_fprop(kind, x, gamma, beta, eps=1e-5, **_kw(variant, groups))
        assert module._KCACHE, "expected channels-last kernel was not used"
        ref = F.group_norm(x.float(), groups, gamma.float(), beta.float(), 1e-5)
        torch.testing.assert_close(y.float(), ref, rtol=0.02, atol=0.02)


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["group", "instance"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("strided", ["gamma", "mean", "rstd", "all"])
def test_frost_norm_nhwc_backward_broadcast_cache(variant, reverse, strided):
    kind, groups, module = _case(variant, backward=True)
    module._KCACHE.clear()
    torch.manual_seed(3152)
    x = torch.randn((2, 128, 8, 8), device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    dy = torch.randn_like(x)
    gs = _broadcast_and_strided(torch.randn(256, device="cuda", dtype=x.dtype), 128)
    grouped = x.float().reshape(2, groups, -1)
    # Make individual group values distinct enough to expose a cached
    # broadcast layout when a strided stats view follows.
    mean = grouped.mean(-1).reshape(-1) + torch.linspace(-2, 2, 2 * groups, device=x.device)
    rstd = torch.rsqrt(grouped.var(-1, unbiased=False) + 1e-5).reshape(-1) * torch.linspace(0.5, 2, 2 * groups, device=x.device)
    mean_storage = torch.empty(2 * mean.numel(), device="cuda", dtype=mean.dtype)
    rstd_storage = torch.empty_like(mean_storage)
    mean_storage[::2].copy_(mean)
    rstd_storage[::2].copy_(rstd)
    means = (mean_storage[:1].expand_as(mean), mean_storage[::2])
    rstds = (rstd_storage[:1].expand_as(rstd), rstd_storage[::2])
    for i in ([1, 0] if reverse else [0, 1]):
        gamma = gs[i if strided in ("gamma", "all") else 1]
        m = means[i if strided in ("mean", "all") else 1]
        r = rstds[i if strided in ("rstd", "all") else 1]
        dx, _, _ = norm_bprop(kind, dy, x, gamma, m, r, has_beta=True, **_kw(variant, groups))
        assert module._KCACHE, "expected channels-last kernel was not used"
        # The supplied stats may be broadcast independently of x. Compute the
        # backward formula from those exact operands instead of recomputing stats.
        centered = x.float().reshape(2, groups, -1) - m.reshape(2, groups, 1)
        scaled_dy = (dy.float() * gamma.float().reshape(1, 128, 1, 1)).reshape(2, groups, -1)
        inv = r.reshape(2, groups, 1)
        dx_ref = (scaled_dy - scaled_dy.mean(-1, keepdim=True) - centered * inv.square() * (scaled_dy * centered).mean(-1, keepdim=True)) * inv
        torch.testing.assert_close(dx.float(), dx_ref.reshape_as(x), rtol=0.02, atol=0.02)
