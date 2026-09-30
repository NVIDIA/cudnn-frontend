# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BatchNorm forward, sm_100, CUTLASS primitives.

BatchNorm reduces across the batch *per channel*: for channel ``c`` the group is
``{x[n, c, s] : all n, all spatial s}``. One CTA owns one channel. The reduction
elements are strided (stride ``C*S`` across ``n``), so there is no contiguous row
to cp.async-stage — the kernel reads X from global and reduces through the shared
:func:`cudnn.norm._common_sm100.block_reduce_sum2`
(``SmemAllocator`` scratch + warp-shuffle). Optionally maintains running
mean/var with momentum (training) or consumes them (inference). Input viewed as
``[N, C, S]``.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator

from cudnn.norm.utils import dyn
from cudnn.norm._common_sm100 import block_reduce_sum2, red_scratch_len


@cute.kernel
def _bn_fwd_kernel(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mSavedMean: cute.Tensor,
    mSavedRstd: cute.Tensor,
    mRunMean: cute.Tensor,
    mRunVar: cute.Tensor,
    N: cutlass.Int32,
    S: cutlass.Int32,
    eps: cutlass.Float32,
    momentum: cutlass.Float32,
    bt: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    training: cutlass.Constexpr,
    update_running: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    c, _, _ = cute.arch.block_idx()
    count = N * S
    countf = cutlass.Float32(count)

    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(red_scratch_len(bt)), byte_alignment=8)

    mean = cutlass.Float32(0.0)
    rstd = cutlass.Float32(0.0)
    if cutlass.const_expr(training):
        s1 = cutlass.Float32(0.0)
        s2 = cutlass.Float32(0.0)
        i = tid
        while i < count:
            n = i // S
            s = i % S
            x = mX[n, c, s].to(cutlass.Float32)
            s1 = s1 + x
            s2 = s2 + x * x
            i = i + bt
        s1, s2 = block_reduce_sum2(s1, s2, tid, red, bt)
        mean = s1 / countf
        var = s2 / countf - mean * mean
        rstd = cute.math.rsqrt(var + eps)
        if tid == 0:
            mSavedMean[c] = mean
            mSavedRstd[c] = rstd
            if cutlass.const_expr(update_running):
                # PyTorch semantics: running_var uses the unbiased estimator.
                unbiased = var * countf / (countf - 1.0)
                mRunMean[c] = (1.0 - momentum) * mRunMean[c] + momentum * mean
                mRunVar[c] = (1.0 - momentum) * mRunVar[c] + momentum * unbiased
    else:
        mean = mRunMean[c].to(cutlass.Float32)
        var = mRunVar[c].to(cutlass.Float32)
        rstd = cute.math.rsqrt(var + eps)

    g = mGamma[c].to(cutlass.Float32)
    b = mBeta[c].to(cutlass.Float32) if cutlass.const_expr(has_beta) else cutlass.Float32(0.0)
    i = tid
    while i < count:
        n = i // S
        s = i % S
        x = mX[n, c, s].to(cutlass.Float32)
        y = g * ((x - mean) * rstd) + b
        mY[n, c, s] = y.to(mY.element_type)
        i = i + bt


@cute.jit
def _bn_fwd_host(
    mX,
    mY,
    mGamma,
    mBeta,
    mSavedMean,
    mSavedRstd,
    mRunMean,
    mRunVar,
    C: cutlass.Int32,
    N: cutlass.Int32,
    S: cutlass.Int32,
    eps: cutlass.Float32,
    momentum: cutlass.Float32,
    bt: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    training: cutlass.Constexpr,
    update_running: cutlass.Constexpr,
) -> None:
    _bn_fwd_kernel(
        mX,
        mY,
        mGamma,
        mBeta,
        mSavedMean,
        mSavedRstd,
        mRunMean,
        mRunVar,
        N,
        S,
        eps,
        momentum,
        bt,
        has_beta,
        training,
        update_running,
    ).launch(grid=(C, 1, 1), block=(bt, 1, 1))


_KCACHE = {}


def forward(spec, x3d, gamma, beta, *, eps, momentum, training, running_mean=None, running_var=None, cfg, params):
    """Launch BatchNorm forward. Returns ``(y, saved_mean, saved_rstd)``.

    When ``training`` and running tensors are supplied, they are updated in place.
    """
    import torch

    has_beta = beta is not None
    if beta is None:
        beta = torch.zeros(spec.C, dtype=x3d.dtype, device=x3d.device)

    update_running = bool(training and running_mean is not None and running_var is not None)
    if running_mean is None:
        running_mean = torch.zeros(spec.C, dtype=torch.float32, device=x3d.device)
    if running_var is None:
        running_var = torch.ones(spec.C, dtype=torch.float32, device=x3d.device)

    y = torch.empty_like(x3d)
    saved_mean = torch.empty(spec.C, dtype=torch.float32, device=x3d.device)
    saved_rstd = torch.empty(spec.C, dtype=torch.float32, device=x3d.device)

    args = (
        dyn(x3d),
        dyn(y),
        dyn(gamma),
        dyn(beta),
        dyn(saved_mean),
        dyn(saved_rstd),
        dyn(running_mean),
        dyn(running_var),
        cutlass.Int32(spec.C),
        cutlass.Int32(spec.N),
        cutlass.Int32(spec.S),
        cutlass.Float32(eps),
        cutlass.Float32(momentum),
    )
    ce = (cfg.block_threads, has_beta, bool(training), update_running)
    key = (params.io_dtype,) + ce
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_bn_fwd_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return y, saved_mean, saved_rstd
