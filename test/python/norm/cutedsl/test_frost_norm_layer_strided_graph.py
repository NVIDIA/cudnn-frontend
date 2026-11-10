# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strided LayerNorm/RMSNorm operands remain valid during graph replay."""

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
@pytest.mark.parametrize("kind", ["layer", "rms"])
@pytest.mark.parametrize("width", [128, 32768])
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_frost_norm_layer_strided_graph_replay(kind, width, phase):
    torch.manual_seed(7103)
    variant = NormVariant.LAYER_NORM if kind == "layer" else NormVariant.RMS_NORM
    x = torch.randn((2, width), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    gs = torch.randn(width * 2 + 1, device="cuda", dtype=x.dtype)
    bs = torch.randn_like(gs)
    gamma, beta = gs[:1].expand(width), bs[1::2][:width]
    xf = x.float()
    mean_values = xf.mean(-1)
    rstd_values = torch.rsqrt((xf.var(-1, unbiased=False) if kind == "layer" else xf.square().mean(-1)) + 1e-5)
    ms = torch.empty(5, device="cuda")
    rs = torch.empty_like(ms)
    mean = ms[1::2] if kind == "layer" else None
    rstd = rs[1::2]
    if mean is not None:
        mean.copy_(mean_values)
    rstd.copy_(rstd_values)
    if phase == "forward":
        out = (torch.empty_like(x), torch.empty(2, device="cuda") if mean is not None else None, torch.empty(2, device="cuda"))
    else:
        out = (torch.empty_like(x), torch.empty(width, device="cuda"), torch.empty(width, device="cuda"))

    def call():
        if phase == "forward":
            return norm_fprop(variant, x, gamma, beta, normalized_shape=(width,), eps=1e-5, out=out)
        return norm_bprop(variant, dy, x, gamma, mean, rstd, normalized_shape=(width,), has_beta=True, out=out)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -1.25):
        gs[:1].fill_(value)
        bs[1::2].fill_(-value)
        graph.replay()
        if phase == "forward":
            if mean is not None:
                ref = F.layer_norm(xf, (width,), gamma.float(), beta.float(), 1e-5)
            else:
                ref = xf * rstd[:, None] * gamma.float() + beta.float()
        else:
            z = (xf - mean[:, None] if mean is not None else xf) * rstd[:, None]
            d = dy.float() * gamma.float()
            ref = (d - (d.mean(-1, keepdim=True) if mean is not None else 0) - z * (d * z).mean(-1, keepdim=True)) * rstd[:, None]
        torch.testing.assert_close(out[0].float(), ref, rtol=0.02, atol=0.02)
