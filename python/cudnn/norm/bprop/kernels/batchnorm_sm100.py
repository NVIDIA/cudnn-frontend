# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BatchNorm backward, sm_100, CUTLASS primitives.

Per channel ``c`` (one CTA), with ``xhat = (x-mean)*rstd``::

    dbeta_c  = sum_{n,s} dy
    dgamma_c = sum_{n,s} dy * xhat
    dx = gamma_c * rstd * (dy - dbeta_c/count - xhat * dgamma_c/count)

Because each channel is reduced by a single CTA, ``dgamma``/``dbeta`` need no
atomics. Reductions go through the shared ``block_reduce_sum2``
(``SmemAllocator`` scratch). Input viewed as ``[N, C, S]``.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator

from cudnn.norm.utils import dyn
from cudnn.norm._common_sm100 import block_reduce_sum2, red_scratch_len


@cute.kernel
def _bn_bwd_kernel(
    mDY: cute.Tensor,
    mX: cute.Tensor,
    mGamma: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    mDX: cute.Tensor,
    mDGamma: cute.Tensor,
    mDBeta: cute.Tensor,
    N: cutlass.Int32,
    S: cutlass.Int32,
    bt: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    c, _, _ = cute.arch.block_idx()
    count = N * S
    countf = cutlass.Float32(count)

    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(red_scratch_len(bt)), byte_alignment=8)

    mean = mMean[c].to(cutlass.Float32)
    rstd = mRstd[c].to(cutlass.Float32)
    g = mGamma[c].to(cutlass.Float32)

    # --- reduce dy and dy*xhat over the channel ---
    s_dy = cutlass.Float32(0.0)
    s_dy_xhat = cutlass.Float32(0.0)
    i = tid
    while i < count:
        n = i // S
        s = i % S
        x = mX[n, c, s].to(cutlass.Float32)
        dy = mDY[n, c, s].to(cutlass.Float32)
        xhat = (x - mean) * rstd
        s_dy = s_dy + dy
        s_dy_xhat = s_dy_xhat + dy * xhat
        i = i + bt
    s_dy, s_dy_xhat = block_reduce_sum2(s_dy, s_dy_xhat, tid, red, bt)

    if tid == 0:
        mDGamma[c] = s_dy_xhat
        if cutlass.const_expr(has_beta):
            mDBeta[c] = s_dy

    a = s_dy / countf
    b = s_dy_xhat / countf
    i = tid
    while i < count:
        n = i // S
        s = i % S
        x = mX[n, c, s].to(cutlass.Float32)
        dy = mDY[n, c, s].to(cutlass.Float32)
        xhat = (x - mean) * rstd
        dx = g * rstd * (dy - a - xhat * b)
        mDX[n, c, s] = dx.to(mDX.element_type)
        i = i + bt


@cute.jit
def _bn_bwd_host(
    mDY,
    mX,
    mGamma,
    mMean,
    mRstd,
    mDX,
    mDGamma,
    mDBeta,
    C: cutlass.Int32,
    N: cutlass.Int32,
    S: cutlass.Int32,
    bt: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    _bn_bwd_kernel(
        mDY,
        mX,
        mGamma,
        mMean,
        mRstd,
        mDX,
        mDGamma,
        mDBeta,
        N,
        S,
        bt,
        has_beta,
    ).launch(grid=(C, 1, 1), block=(bt, 1, 1))


_KCACHE = {}


def backward(spec, dy3d, x3d, gamma, saved_mean, saved_rstd, *, has_beta, cfg, params):
    """Launch BatchNorm backward. Returns ``(dx, dgamma, dbeta)``."""
    import torch

    dx = torch.empty_like(x3d)
    dgamma = torch.zeros(spec.C, dtype=torch.float32, device=x3d.device)
    dbeta = torch.zeros(spec.C, dtype=torch.float32, device=x3d.device)

    args = (
        dyn(dy3d),
        dyn(x3d),
        dyn(gamma),
        dyn(saved_mean),
        dyn(saved_rstd),
        dyn(dx),
        dyn(dgamma),
        dyn(dbeta),
        cutlass.Int32(spec.C),
        cutlass.Int32(spec.N),
        cutlass.Int32(spec.S),
    )
    ce = (cfg.block_threads, has_beta)
    key = (params.io_dtype,) + ce
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_bn_bwd_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return dx, dgamma, (dbeta if has_beta else None)
