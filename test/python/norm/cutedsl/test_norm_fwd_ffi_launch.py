# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The warp-per-row LN / RMS forward launches through a tvm-ffi entry compiled once on a dynamic row count: torch
tensors bind directly (no per-call DLPack export), so the compile key must stay row-count free, autograd-tape operands
must still be accepted, and a launch must land on torch's current stream.

    pytest test/python/norm/cutedsl/test_norm_fwd_ffi_launch.py
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm kernels require a built cudnn frontend")

from cudnn.norm._lazy import dsl_unavailable  # noqa: E402

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_fprop  # noqa: E402
from cudnn.norm.fprop.kernels import layernorm_warp_sm100  # noqa: E402

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def _sm100_family():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the SM100 norm kernels need compute capability 10.0 or 10.3")


@pytest.mark.parametrize("cols", [768, 5120])  # the persistent and the pipelined warp launch
def test_one_artifact_serves_every_row_count(cols):
    gamma = torch.randn(cols, device="cuda", dtype=torch.bfloat16)
    norm_fprop(NormVariant.RMS_NORM, torch.randn(7, cols, device="cuda", dtype=torch.bfloat16), gamma, None, eps=1e-6, normalized_shape=[cols])
    compiled = len(layernorm_warp_sm100._KCACHE)
    for rows in (1, 300, 4097, 20000):
        x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
        y = norm_fprop(NormVariant.RMS_NORM, x, gamma, None, eps=1e-6, normalized_shape=[cols])[0]
        torch.testing.assert_close(y.float(), F.rms_norm(x.float(), [cols], gamma.float(), eps=1e-6), atol=2e-2, rtol=2e-2)
    assert len(layernorm_warp_sm100._KCACHE) == compiled, "a new row count recompiled the forward"


def test_autograd_tape_operands_are_accepted():
    x = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16)
    gamma = torch.nn.Parameter(torch.randn(1024, device="cuda", dtype=torch.bfloat16))
    beta = torch.nn.Parameter(torch.randn(1024, device="cuda", dtype=torch.bfloat16))
    got = norm_fprop(NormVariant.LAYER_NORM, x.requires_grad_(), gamma, beta, eps=1e-5, normalized_shape=[1024])
    want = norm_fprop(NormVariant.LAYER_NORM, x.detach(), gamma.detach(), beta.detach(), eps=1e-5, normalized_shape=[1024])
    for a, b in zip(got, want):
        assert torch.equal(a, b)


def test_launch_lands_on_the_current_stream():
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    ref = norm_fprop(NormVariant.RMS_NORM, x, gamma, None, eps=1e-6, normalized_shape=[4096])[0]
    out = torch.zeros_like(x)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(200_000_000)  # holds the side stream well past the snapshot below
        norm_fprop(NormVariant.RMS_NORM, x, gamma, None, eps=1e-6, normalized_shape=[4096], out=(out, None, None))
    snapshot = out.clone()  # on the default stream, which nothing holds: the side-stream launch has not run yet
    torch.cuda.synchronize()
    assert not snapshot.any(), "the forward ran on a stream other than the caller's current one"
    assert torch.equal(out, ref)
