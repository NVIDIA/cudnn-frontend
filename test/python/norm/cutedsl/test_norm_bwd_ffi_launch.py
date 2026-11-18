# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The LN / RMS backward's pipelined, tiled and plain routes launch through tvm-ffi entries compiled on a dynamic row
count: one pipelined or plain artifact serves every row count (the tiled route's key carries its row tiling and grid),
autograd-tape operands are accepted, and the launch lands on torch's current stream.

    pytest test/python/norm/cutedsl/test_norm_bwd_ffi_launch.py
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm kernels require a built cudnn frontend")

from cudnn.norm._lazy import dsl_unavailable  # noqa: E402

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402
from cudnn.norm.bprop.kernels import layernorm_sm100  # noqa: E402

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def _sm100_family():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the SM100 norm kernels need compute capability 10.0 or 10.3")


def _artifacts():
    return len(layernorm_sm100._KPIPE) + len(layernorm_sm100._KTILED) + len(layernorm_sm100._KCACHE)


def _grads(variant, x, gamma, beta, dy):
    cols = x.shape[-1]
    _, mean, rstd = norm_fprop(variant, x, gamma, beta, eps=1e-5, normalized_shape=[cols])
    return norm_bprop(variant, dy, x, gamma, mean, rstd, normalized_shape=[cols], has_beta=beta is not None)


@pytest.mark.parametrize(
    "variant, cols, rows",
    [
        (NormVariant.LAYER_NORM, 1024, (300, 4100, 9000)),  # pipelined
        (NormVariant.LAYER_NORM, 96, (50, 300, 900)),  # plain, below the slot floor
    ],
)
def test_one_artifact_per_route_serves_every_row_count(variant, cols, rows):
    gamma = torch.randn(cols, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(cols, device="cuda", dtype=torch.bfloat16) if variant is NormVariant.LAYER_NORM else None
    x = torch.randn(rows[0], cols, device="cuda", dtype=torch.bfloat16)
    _grads(variant, x, gamma, beta, torch.randn_like(x))
    compiled = _artifacts()
    for r in rows[1:]:
        x = torch.randn(r, cols, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x)
        dx = _grads(variant, x, gamma, beta, dy)[0]
        xr = x.double().requires_grad_()
        y = (
            F.layer_norm(xr, [cols], gamma.double(), None if beta is None else beta.double(), 1e-5)
            if beta is not None
            else F.rms_norm(xr, [cols], gamma.double(), eps=1e-5)
        )
        (ref,) = torch.autograd.grad(y, xr, dy.double())
        torch.testing.assert_close(dx.double(), ref, atol=3e-2, rtol=3e-2)
    assert _artifacts() == compiled, "a new row count recompiled the backward"


def test_autograd_tape_operands_are_accepted():
    x = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    gamma = torch.nn.Parameter(torch.randn(1024, device="cuda", dtype=torch.bfloat16))
    beta = torch.nn.Parameter(torch.randn(1024, device="cuda", dtype=torch.bfloat16))
    got = _grads(NormVariant.LAYER_NORM, x.requires_grad_(), gamma, beta, dy)[0]
    want = _grads(NormVariant.LAYER_NORM, x.detach(), gamma.detach(), beta.detach(), dy)[0]
    assert torch.equal(got, want)


def test_launch_lands_on_the_current_stream():
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    _, mean, rstd = norm_fprop(NormVariant.RMS_NORM, x, gamma, None, eps=1e-6, normalized_shape=[4096])
    ref = norm_bprop(NormVariant.RMS_NORM, dy, x, gamma, mean, rstd, normalized_shape=[4096], has_beta=False)[0]
    dx = torch.zeros_like(x)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(200_000_000)  # holds the side stream well past the snapshot below
        norm_bprop(NormVariant.RMS_NORM, dy, x, gamma, mean, rstd, normalized_shape=[4096], has_beta=False, out=(dx, None, None))
    snapshot = dx.clone()  # on the default stream, which nothing holds: the side-stream launch has not run yet
    torch.cuda.synchronize()
    assert not snapshot.any(), "the backward ran on a stream other than the caller's current one"
    assert torch.equal(dx, ref)
