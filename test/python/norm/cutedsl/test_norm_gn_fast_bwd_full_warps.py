# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The atomic-free GroupNorm backward (groupnorm_fast_sm100) at channel-per-group counts that do not divide 256.

The CTA width is rounded up to a multiple of CPG. At CPG 20 / 40 (640 / 1280 channels in 32 groups) that gave 260 / 280
threads, a partial last warp: the block reduction dropped it from the row sums and its lane 0 wrote over another warp's slot,
so dx was wrong and changed from run to run."""

import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")

from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop


def _fast():
    from cudnn.norm.bprop.kernels import groupnorm_fast_sm100

    return groupnorm_fast_sm100


@pytest.mark.L0
@pytest.mark.norm_rejects_before_launch
@pytest.mark.parametrize("cpg", [1, 2, 3, 5, 8, 10, 20, 40, 64, 96, 160])
@pytest.mark.parametrize("span", [1, 49, 64, 256])
def test_block_width_is_whole_warps_and_a_multiple_of_cpg(cpg, span):
    spec = SimpleNamespace(channels_per_group=cpg, gamma_inner_span=span, M=cpg * span)
    bt = _fast().block_threads(spec, 2)
    assert bt is not None and bt % 32 == 0 and bt % cpg == 0 and 32 <= bt <= 1024, bt


@pytest.mark.L0
@pytest.mark.norm_rejects_before_launch
def test_no_whole_warp_width_declines_the_route():
    # lcm(33, 32) = 1056 threads: past the CTA limit, so the route declines and the dispatcher falls through.
    spec = SimpleNamespace(channels_per_group=33, gamma_inner_span=64, M=33 * 64)
    assert _fast().block_threads(spec, 2) is None


@pytest.mark.L0
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((16, 1280, 16, 16), torch.bfloat16),
        ((16, 1280, 16, 16), torch.float32),
        ((4, 640, 32, 32), torch.bfloat16),  # in fp32 this shape takes another route
        ((2, 1280, 8, 8), torch.bfloat16),
        ((2, 1280, 8, 8), torch.float32),
    ],
)
def test_cpg_off_256_dx_is_deterministic_and_matches_reference(shape, dtype):
    N, C, H, W = shape
    G = 32
    gen = torch.Generator(device="cuda").manual_seed(N * C + H)
    x = torch.randn(N, C, H, W, device="cuda", dtype=dtype, generator=gen)
    g = torch.randn(C, device="cuda", dtype=dtype, generator=gen)
    b = torch.randn(C, device="cuda", dtype=dtype, generator=gen)
    dy = torch.randn(N, C, H, W, device="cuda", dtype=dtype, generator=gen)
    _, mean, rstd = norm_fprop(NormVariant.GROUP_NORM, x, g, b, eps=1e-5, num_groups=G)

    norm_bprop(NormVariant.GROUP_NORM, dy, x, g, mean, rstd, num_groups=G, has_beta=True)  # loads the route modules
    fast = sys.modules["cudnn.norm.bprop.kernels.groupnorm_fast_sm100"]
    calls = []
    orig = fast.backward

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    fast.backward = spy
    try:
        runs = [norm_bprop(NormVariant.GROUP_NORM, dy, x, g, mean, rstd, num_groups=G, has_beta=True) for _ in range(8)]
    finally:
        fast.backward = orig
    assert len(calls) == 8, "this shape no longer takes the atomic-free backward; pick one that does"
    dx = runs[0][0]
    for r in runs[1:]:
        assert torch.equal(r[0], dx), "dx changed between identical calls"

    xr = x.double().requires_grad_(True)
    gr = g.double().requires_grad_(True)
    br = b.double().requires_grad_(True)
    y = torch.nn.functional.group_norm(xr, G, gr, br, 1e-5)
    dx_ref, dg_ref, db_ref = torch.autograd.grad(y, (xr, gr, br), grad_outputs=dy.double())
    torch.testing.assert_close(dx.double(), dx_ref, rtol=2e-2 if dtype == torch.bfloat16 else 1e-4, atol=2e-2 if dtype == torch.bfloat16 else 1e-4)
    torch.testing.assert_close(runs[0][1].double(), dg_ref, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(runs[0][2].double(), db_ref, rtol=1e-3, atol=1e-2)
