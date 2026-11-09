# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward cache entries must match the vector strides of the current call."""

import pytest
import torch

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")
from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop  # noqa: E402
from cudnn.norm.bprop.kernels import groupnorm_sm100, instancenorm_warp_sm100  # noqa: E402


def _stride_views(storage, n):
    return {0: storage[:1].expand(n), 1: storage[:n], 2: storage[::2][:n], 3: storage[::3][:n]}


@pytest.mark.L0
@pytest.mark.parametrize("route", ["instance_warp", "group_fallback"])
@pytest.mark.parametrize("operand", ["gamma", "mean", "rstd", "all"])
@pytest.mark.parametrize("order", [(0, 2), (2, 0), (1, 2), (2, 1), (2, 3)])
def test_frost_norm_backward_vector_stride_cache(route, operand, order):
    torch.manual_seed(5103)
    if route == "instance_warp":
        variant, shape, groups, kernel = NormVariant.INSTANCE_NORM, (2, 64, 4, 4), 64, instancenorm_warp_sm100
    else:
        # 67 channels per group makes both optimized GN routes decline.
        variant, shape, groups, kernel = NormVariant.GROUP_NORM, (2, 67, 2, 2), 1, groupnorm_sm100
    kernel._KCACHE.clear()
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    count, rows = shape[1], shape[0] * groups
    gs = _stride_views(torch.randn(count * 3, device="cuda", dtype=x.dtype), count)
    ms = _stride_views(torch.randn(rows * 3, device="cuda") * 0.3, rows)
    rs = _stride_views(torch.rand(rows * 3, device="cuda") * 0.4 + 0.8, rows)
    for stride in order:
        gamma = gs[stride if operand in ("gamma", "all") else 2]
        mean = ms[stride if operand in ("mean", "all") else 2]
        rstd = rs[stride if operand in ("rstd", "all") else 2]
        kwargs = {"num_groups": groups} if variant == NormVariant.GROUP_NORM else {}
        dx, dg, db = norm_bprop(variant, dy, x, gamma, mean, rstd, has_beta=True, **kwargs)
        z = (x.float().reshape(shape[0], groups, -1) - mean.reshape(shape[0], groups, 1)) * rstd.reshape(shape[0], groups, 1)
        d = (dy.float() * gamma.float().reshape(1, count, 1, 1)).reshape(shape[0], groups, -1)
        dx_ref = (d - d.mean(-1, keepdim=True) - z * (d * z).mean(-1, keepdim=True)) * rstd.reshape(shape[0], groups, 1)
        dg_ref = (dy.float() * z.reshape_as(x)).sum((0, 2, 3))
        db_ref = dy.float().sum((0, 2, 3))
        torch.testing.assert_close(dx.float(), dx_ref.reshape_as(x), rtol=0.02, atol=0.02)
        torch.testing.assert_close(dg.float(), dg_ref, rtol=0.01, atol=0.03)
        torch.testing.assert_close(db.float(), db_ref, rtol=0.01, atol=0.03)
    assert len(kernel._KCACHE) == (1 if order == (2, 3) else 2)


@pytest.mark.L0
@pytest.mark.parametrize("route", ["instance_warp", "group_fallback"])
def test_frost_norm_backward_broadcast_graph_replay(route):
    torch.manual_seed(5104)
    if route == "instance_warp":
        variant, shape, groups, kernel = NormVariant.INSTANCE_NORM, (2, 64, 4, 4), 64, instancenorm_warp_sm100
    else:
        variant, shape, groups, kernel = NormVariant.GROUP_NORM, (2, 67, 2, 2), 1, groupnorm_sm100
    kernel._KCACHE.clear()
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    f = x.float().reshape(shape[0], groups, -1)
    mean = f.mean(-1).reshape(-1)
    rstd = torch.rsqrt(f.var(-1, unbiased=False) + 1e-5).reshape(-1)
    gs = torch.randn(shape[1] * 2, device="cuda", dtype=x.dtype)
    gamma = gs[:1].expand(shape[1])
    outputs = (torch.empty_like(x), torch.empty(shape[1], device="cuda"), torch.empty(shape[1], device="cuda"))
    kwargs = {"num_groups": groups} if variant == NormVariant.GROUP_NORM else {}
    norm_bprop(variant, dy, x, gs[::2], mean, rstd, has_beta=True, out=outputs, **kwargs)
    norm_bprop(variant, dy, x, gamma, mean, rstd, has_beta=True, out=outputs, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        norm_bprop(variant, dy, x, gamma, mean, rstd, has_beta=True, out=outputs, **kwargs)
    z = (f - mean.reshape(shape[0], groups, 1)) * rstd.reshape(shape[0], groups, 1)
    for value in (0.5, -1.25):
        gs[:1].fill_(value)
        graph.replay()
        d = (dy.float() * gamma.float().reshape(1, shape[1], 1, 1)).reshape(shape[0], groups, -1)
        dx_ref = (d - d.mean(-1, keepdim=True) - z * (d * z).mean(-1, keepdim=True)) * rstd.reshape(shape[0], groups, 1)
        torch.testing.assert_close(outputs[0].float(), dx_ref.reshape_as(x), rtol=0.02, atol=0.02)
        torch.testing.assert_close(outputs[1], (dy.float() * z.reshape_as(x)).sum((0, 2, 3)), rtol=0.01, atol=0.03)
        torch.testing.assert_close(outputs[2], dy.float().sum((0, 2, 3)), rtol=0.01, atol=0.03)
