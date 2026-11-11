# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CGA norm FFI descriptors retain strided input bindings during graph replay."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")
from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["group", "instance"])
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_frost_norm_cga_strided_graph_replay(kind, phase):
    torch.manual_seed(6104)
    if kind == "group":
        variant, shape, groups, kwargs = NormVariant.GROUP_NORM, (2, 64, 16, 32), 4, {"num_groups": 4}
    else:
        variant, shape, groups, kwargs = NormVariant.INSTANCE_NORM, (1, 8, 32, 128), 8, {}
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    storage = torch.randn(shape[1] * 2 + 1, device="cuda", dtype=x.dtype)
    gamma = storage[:1].expand(shape[1])
    beta_storage = torch.randn_like(storage)
    beta = beta_storage[1::2][: shape[1]]
    grouped = x.float().reshape(shape[0], groups, -1)
    mean_values = grouped.mean(-1).reshape(-1)
    rstd_values = torch.rsqrt(grouped.var(-1, unbiased=False) + 1e-5).reshape(-1)
    ms = torch.empty(mean_values.numel() * 2 + 1, device="cuda")
    rs = torch.empty_like(ms)
    mean, rstd = ms[1::2], rs[1::2]
    mean.copy_(mean_values)
    rstd.copy_(rstd_values)
    if phase == "forward":
        out = (torch.empty_like(x), torch.empty_like(mean_values), torch.empty_like(rstd_values))
    else:
        out = (torch.empty_like(x), torch.empty(shape[1], device="cuda"), torch.empty(shape[1], device="cuda"))

    def call():
        if phase == "forward":
            return norm_fprop(variant, x, gamma, beta, eps=1e-5, out=out, **kwargs)
        return norm_bprop(variant, dy, x, gamma, mean, rstd, has_beta=True, out=out, **kwargs)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -1.25):
        storage[:1].fill_(value)
        beta_storage[1::2].fill_(-value)
        graph.replay()
        if phase == "forward":
            ref = F.group_norm(x.float(), groups, gamma.float(), beta.float(), 1e-5)
        else:
            z = (grouped - mean.reshape(shape[0], groups, 1)) * rstd.reshape(shape[0], groups, 1)
            d = (dy.float() * gamma.float().reshape(1, shape[1], 1, 1)).reshape(shape[0], groups, -1)
            ref = ((d - d.mean(-1, keepdim=True) - z * (d * z).mean(-1, keepdim=True)) * rstd.reshape(shape[0], groups, 1)).reshape_as(x)
        torch.testing.assert_close(out[0].float(), ref, rtol=0.02, atol=0.02)
