# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BatchNorm compiled signatures must distinguish unit and dynamic vector strides."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")
from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402


def _module(layout, backward):
    if backward:
        from cudnn.norm.bprop.kernels import batchnorm_nchw_sm100 as nchw
        from cudnn.norm.bprop.kernels import batchnorm_nhwc_sm100 as nhwc
        from cudnn.norm.bprop.kernels import batchnorm_sm100 as plain
    else:
        from cudnn.norm.fprop.kernels import batchnorm_nchw_sm100 as nchw
        from cudnn.norm.fprop.kernels import batchnorm_nhwc_sm100 as nhwc
        from cudnn.norm.fprop.kernels import batchnorm_sm100 as plain
    return {"plain": plain, "nchw": nchw, "nhwc": nhwc}[layout]


def _views(t, count):
    return t[:count], t[::2][:count]


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["plain", "nchw", "nhwc"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "phase,field",
    [(phase, field) for phase in ("training", "inference") for field in ("gamma", "beta", "running_mean", "running_var")]
    + [("backward", field) for field in ("gamma", "mean", "rstd")],
)
def test_frost_batchnorm_vector_stride_cache(layout, reverse, phase, field):
    backward = phase == "backward"
    module = _module(layout, backward)
    module._KCACHE.clear()
    torch.manual_seed(422)
    c = 7 if layout == "plain" else 128
    x = torch.randn((2, c, 8, 8), device="cuda", dtype=torch.bfloat16)
    if layout == "nhwc":
        x = x.contiguous(memory_format=torch.channels_last)
    dy = torch.randn_like(x)
    gs = _views(torch.randn(2 * c, device="cuda", dtype=x.dtype), c)
    bs = _views(torch.randn(2 * c, device="cuda", dtype=x.dtype), c)
    m0 = torch.randn(c, device="cuda")
    v0 = torch.rand(c, device="cuda") + 0.5
    saved_mean = x.float().mean((0, 2, 3))
    saved_rstd = torch.rsqrt(x.float().var((0, 2, 3), unbiased=False) + 1e-5)
    for i in ([1, 0] if reverse else [0, 1]):
        gamma = gs[i if field == "gamma" else 0]
        beta = bs[i if field == "beta" else 0]
        mean_storage = torch.full((2 * c,), -71.0, device="cuda")
        var_storage = torch.full((2 * c,), -73.0, device="cuda")
        mean_source = saved_mean if backward else m0
        var_source = saved_rstd if backward else v0
        mean_storage[::2].copy_(mean_source)
        var_storage[::2].copy_(var_source)
        mean = mean_storage[::2] if i and field in ("running_mean", "mean") else mean_source.clone()
        var = var_storage[::2] if i and field in ("running_var", "rstd") else var_source.clone()
        if backward:
            actual, _, _ = norm_bprop(NormVariant.BATCH_NORM, dy, x, gamma, mean, var, has_beta=True)
            xr = x.float().detach().requires_grad_()
            yr = F.batch_norm(xr, None, None, gamma.float(), None, training=True, eps=1e-5)
            (ref,) = torch.autograd.grad(yr, (xr,), dy.float())
        else:
            rm_ref, rv_ref = mean.clone(), var.clone()
            actual, _, _ = norm_fprop(
                NormVariant.BATCH_NORM, x, gamma, beta, training=phase == "training", running_mean=mean, running_var=var, momentum=0.1, eps=1e-5
            )
            ref = F.batch_norm(x.float(), rm_ref, rv_ref, gamma.float(), beta.float(), training=phase == "training", momentum=0.1, eps=1e-5)
            torch.testing.assert_close(mean, rm_ref, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(var, rv_ref, rtol=1e-4, atol=1e-4)
            if i and field == "running_mean":
                assert torch.all(mean_storage[1::2] == -71.0)
            if i and field == "running_var":
                assert torch.all(var_storage[1::2] == -73.0)
        assert module._KCACHE, "expected BatchNorm route was not used"
        torch.testing.assert_close(actual.float(), ref, rtol=0.02, atol=0.02)


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["plain", "nchw", "nhwc"])
def test_frost_batchnorm_dynamic_stride_cache_reuse(layout):
    module = _module(layout, False)
    module._KCACHE.clear()
    torch.manual_seed(422)
    c = 7 if layout == "plain" else 128
    x = torch.randn((2, c, 8, 8), device="cuda", dtype=torch.bfloat16)
    if layout == "nhwc":
        x = x.contiguous(memory_format=torch.channels_last)
    gs = torch.randn(3 * c, device="cuda", dtype=x.dtype)
    bs = torch.randn_like(gs)
    count = None
    for step in (2, 3, 2):
        gamma, beta = gs[::step][:c], bs[::step][:c]
        y, _, _ = norm_fprop(NormVariant.BATCH_NORM, x, gamma, beta, training=True)
        ref = F.batch_norm(x.float(), None, None, gamma.float(), beta.float(), training=True)
        torch.testing.assert_close(y.float(), ref, rtol=0.02, atol=0.02)
        count = len(module._KCACHE) if count is None else count
        assert count > 0 and len(module._KCACHE) == count


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["plain", "nchw", "nhwc"])
def test_frost_batchnorm_strided_graph_replay(layout):
    from cudnn.norm.fprop.api import fprop_workspace_bytes
    from cudnn.norm.utils import Workspace

    torch.manual_seed(422)
    c = 7 if layout == "plain" else 128
    x = torch.randn((2, c, 8, 8), device="cuda", dtype=torch.bfloat16)
    if layout == "nhwc":
        x = x.contiguous(memory_format=torch.channels_last)
    gs = torch.randn(2 * c, device="cuda", dtype=x.dtype)
    bs = torch.randn_like(gs)
    gamma, beta = gs[::2], bs[::2]
    m_storage = torch.full((2 * c,), -71.0, device="cuda")
    v_storage = torch.full((2 * c,), -73.0, device="cuda")
    mean, var = m_storage[::2], v_storage[::2]
    mean.copy_(torch.randn_like(mean))
    var.copy_(torch.rand_like(var) + 0.5)
    out = (torch.empty_like(x), torch.empty(c, device="cuda"), torch.empty(c, device="cuda"))
    nbytes = fprop_workspace_bytes(NormVariant.BATCH_NORM, x.shape, "bf16", nhwc=layout == "nhwc")
    ws = Workspace(torch.empty(nbytes, device="cuda", dtype=torch.uint8))
    norm_fprop(NormVariant.BATCH_NORM, x, gamma, beta, training=True, running_mean=mean, running_var=var, out=out, ws=ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        norm_fprop(NormVariant.BATCH_NORM, x, gamma, beta, training=True, running_mean=mean, running_var=var, out=out, ws=ws)
    for _ in range(2):
        gamma.copy_(torch.randn_like(gamma))
        beta.copy_(torch.randn_like(beta))
        rm_ref, rv_ref = mean.clone(), var.clone()
        graph.replay()
        ref = F.batch_norm(x.float(), rm_ref, rv_ref, gamma.float(), beta.float(), training=True)
        torch.testing.assert_close(out[0].float(), ref, rtol=0.02, atol=0.02)
        torch.testing.assert_close(mean, rm_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(var, rv_ref, rtol=1e-4, atol=1e-4)
        assert torch.all(m_storage[1::2] == -71.0)
        assert torch.all(v_storage[1::2] == -73.0)


def _broadcast_and_strided(storage, count):
    return storage[:1].expand(count), storage[::2][:count]


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["plain", "nchw", "nhwc"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "phase,field",
    [("training", field) for field in ("gamma", "beta")]
    + [("inference", field) for field in ("gamma", "beta", "running_mean", "running_var")]
    + [("backward", field) for field in ("gamma", "mean", "rstd")],
)
def test_frost_batchnorm_broadcast_stride_cache(layout, reverse, phase, field):
    backward = phase == "backward"
    module = _module(layout, backward)
    module._KCACHE.clear()
    torch.manual_seed(422)
    c = 7 if layout == "plain" else 128
    x = torch.randn((2, c, 8, 8), device="cuda", dtype=torch.bfloat16)
    if layout == "nhwc":
        x = x.contiguous(memory_format=torch.channels_last)
    dy = torch.randn_like(x)
    gs = _broadcast_and_strided(torch.randn(2 * c, device="cuda", dtype=x.dtype), c)
    bs = _broadcast_and_strided(torch.randn(2 * c, device="cuda", dtype=x.dtype), c)
    m0 = x.float().mean((0, 2, 3)) + torch.linspace(-2, 2, c, device=x.device)
    r0 = torch.rsqrt(x.float().var((0, 2, 3), unbiased=False) + 1e-5) * torch.linspace(0.5, 2, c, device=x.device)
    v0 = torch.rand(c, device=x.device) + 0.5
    ms = torch.empty(2 * c, device=x.device)
    rs = torch.empty_like(ms)
    vs = torch.empty_like(ms)
    ms[::2].copy_(m0)
    rs[::2].copy_(r0)
    vs[::2].copy_(v0)
    means = (ms[:1].expand(c), ms[::2])
    rstds = (rs[:1].expand(c), rs[::2])
    variances = (vs[:1].expand(c), vs[::2])
    for i in ([1, 0] if reverse else [0, 1]):
        gamma = gs[i if field == "gamma" else 1]
        beta = bs[i if field == "beta" else 1]
        if backward:
            mean = means[i if field == "mean" else 1]
            rstd = rstds[i if field == "rstd" else 1]
            actual, _, _ = norm_bprop(NormVariant.BATCH_NORM, dy, x, gamma, mean, rstd, has_beta=True)
            centered = x.float() - mean.reshape(1, c, 1, 1)
            scaled_dy = dy.float() * gamma.float().reshape(1, c, 1, 1)
            inv = rstd.reshape(1, c, 1, 1)
            ref = (scaled_dy - scaled_dy.mean((0, 2, 3), keepdim=True) - centered * inv.square() * (scaled_dy * centered).mean((0, 2, 3), keepdim=True)) * inv
        else:
            # Training updates these tensors in place, so keep them disjoint.
            mean = means[i if field == "running_mean" else 1] if phase == "inference" else m0.clone()
            var = variances[i if field == "running_var" else 1] if phase == "inference" else v0.clone()
            rm_ref, rv_ref = mean.contiguous().clone(), var.contiguous().clone()
            actual, _, _ = norm_fprop(
                NormVariant.BATCH_NORM, x, gamma, beta, training=phase == "training", running_mean=mean, running_var=var, momentum=0.1, eps=1e-5
            )
            ref = F.batch_norm(x.float(), rm_ref, rv_ref, gamma.float(), beta.float(), training=phase == "training", momentum=0.1, eps=1e-5)
            torch.testing.assert_close(mean, rm_ref, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(var, rv_ref, rtol=1e-4, atol=1e-4)
        assert module._KCACHE, "expected BatchNorm route was not used"
        torch.testing.assert_close(actual.float(), ref, rtol=0.02, atol=0.02)


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["plain", "nchw", "nhwc"])
def test_frost_batchnorm_broadcast_graph_replay(layout):
    from cudnn.norm.fprop.api import fprop_workspace_bytes
    from cudnn.norm.utils import Workspace

    module = _module(layout, False)
    module._KCACHE.clear()
    torch.manual_seed(422)
    c = 7 if layout == "plain" else 128
    x = torch.randn((2, c, 8, 8), device="cuda", dtype=torch.bfloat16)
    if layout == "nhwc":
        x = x.contiguous(memory_format=torch.channels_last)
    gs = torch.randn(2 * c, device="cuda", dtype=x.dtype)
    bs = torch.randn_like(gs)
    mean, var = torch.randn(c, device=x.device), torch.rand(c, device=x.device) + 0.5
    out = (torch.empty_like(x), torch.empty(c, device="cuda"), torch.empty(c, device="cuda"))
    nbytes = fprop_workspace_bytes(NormVariant.BATCH_NORM, x.shape, "bf16", nhwc=layout == "nhwc")
    ws = Workspace(torch.empty(nbytes, device="cuda", dtype=torch.uint8))
    # Warm the dynamic signature before capturing a distinct broadcast signature.
    norm_fprop(NormVariant.BATCH_NORM, x, gs[::2], bs[::2], training=True, running_mean=mean, running_var=var, out=out, ws=ws)
    gamma, beta = gs[:1].expand(c), bs[:1].expand(c)
    norm_fprop(NormVariant.BATCH_NORM, x, gamma, beta, training=True, running_mean=mean, running_var=var, out=out, ws=ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        norm_fprop(NormVariant.BATCH_NORM, x, gamma, beta, training=True, running_mean=mean, running_var=var, out=out, ws=ws)
    for value in (0.5, -1.25):
        gs[:1].fill_(value)
        bs[:1].fill_(-value)
        rm_ref, rv_ref = mean.clone(), var.clone()
        graph.replay()
        ref = F.batch_norm(x.float(), rm_ref, rv_ref, gamma.float(), beta.float(), training=True)
        torch.testing.assert_close(out[0].float(), ref, rtol=0.02, atol=0.02)
        torch.testing.assert_close(mean, rm_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(var, rv_ref, rtol=1e-4, atol=1e-4)
