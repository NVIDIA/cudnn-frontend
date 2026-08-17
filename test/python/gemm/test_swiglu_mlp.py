# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for cudnn.gemm.ops.swiglu_mlp (dense bf16 SwiGLU-MLP).

The fused SwiGLU forward runs on cuDNN's runtime-fusion engine, which needs an
SM100 (Blackwell) device; the op must match torch to bf16 noise on the output and
all four gradients.
"""

import pytest
import torch
import torch.nn.functional as F

from cudnn.gemm.ops import swiglu_mlp


def _cc():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _rel_l2(a, b):
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-9)


def _ref(x, Wg, Wu, Wd):
    return (F.silu(x @ Wg.t()) * (x @ Wu.t())) @ Wd.t()


# bf16 noise across three chained GEMMs + the SwiGLU; cuDNN vs torch differ only by
# accumulation order, so a relative-L2 at this level is the meaningful bar.
_TOL = 2e-2


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
@pytest.mark.parametrize(
    "M,H,inter",
    [(512, 512, 1024), (2048, 1024, 2048)],
    ids=["small", "mlp"],
)
def test_swiglu_mlp_parity(M, H, inter):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
    Wg = (torch.randn(inter, H, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wu = (torch.randn(inter, H, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wd = (torch.randn(H, inter, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    do = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16)

    xr, Wgr, Wur, Wdr = (t.detach().clone().requires_grad_(True) for t in (x, Wg, Wu, Wd))

    out = swiglu_mlp(x, Wg, Wu, Wd)
    out.backward(do)
    ref = _ref(xr, Wgr, Wur, Wdr)
    ref.backward(do)

    assert _rel_l2(out, ref) < _TOL, f"fwd rel={_rel_l2(out, ref):.2e}"
    for name, a, b in (("dx", x.grad, xr.grad), ("dWg", Wg.grad, Wgr.grad), ("dWu", Wu.grad, Wur.grad), ("dWd", Wd.grad, Wdr.grad)):
        assert _rel_l2(a, b) < _TOL, f"{name} rel={_rel_l2(a, b):.2e}"


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() >= 100),
    reason="cuDNN SwiGLU-MLP fusion requires SM100 (Blackwell)",
)
def test_swiglu_mlp_forward_is_single_kernel():
    """The gate GEMM + up GEMM + SiLU + mul must fuse into ONE cuDNN launch."""
    from cudnn.gemm.ops.swiglu_mlp import _swiglu_act

    torch.manual_seed(0)
    M, H, inter = 512, 512, 1024
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    Wg = torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02
    Wu = torch.randn(inter, H, device="cuda", dtype=torch.bfloat16) * 0.02
    _swiglu_act(x, Wg, Wu)  # warm/cache/autotune
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        _swiglu_act(x, Wg, Wu)
        torch.cuda.synchronize()
    launches = sum(ev.count for ev in prof.key_averages() if ev.self_device_time_total > 0)
    assert launches == 1, f"expected 1 fused kernel, saw {launches}"
