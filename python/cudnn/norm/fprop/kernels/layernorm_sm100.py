# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LayerNorm forward, sm_100, CUTLASS primitives.

One CTA owns one normalization group (a row of the ``[R, M]`` view). Two-pass:
reduce ``mean``/``rstd`` over ``M``, then normalize + affine ``y = gamma*xhat + beta``.

Staging (``stage_mode``): the row is brought into shared memory once so X is read
from global only once — via a single ``cp.async.bulk`` (TMA-family, mbarrier;
``STAGE_BULK``, the default when the row fits smem) or per-thread ``cp.async``
(``STAGE_CPASYNC``). When it can't be staged (row too big) the kernel reads X
directly from global (``STAGE_NONE``). Loads/stores are 128-bit vectorized
(``vec``) when ``M % V == 0``: the global Y store (and, unstaged, the X load) go
through register fragments + ``autovec_copy``. Statistics are fp32.

Also serves RMSNorm via ``has_mean=False`` (see :mod:`rmsnorm_sm100`).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator

from cudnn.norm.utils import dyn
from cudnn.norm._common_sm100 import (
    STAGE_BULK,
    STAGE_CPASYNC,
    STAGE_NONE,
    block_reduce_sum2,
    red_scratch_len,
    stage_row,
    stage_row_bulk,
)


@cute.kernel
def _ln_fwd_kernel(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    eps: cutlass.Float32,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    elem_bytes: cutlass.Constexpr,
    stage_mode: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
    staged: cutlass.Constexpr = stage_mode != STAGE_NONE
    nv: cutlass.Constexpr = M // V

    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(red_scratch_len(bt)), byte_alignment=8)
    sX = None
    if cutlass.const_expr(staged):
        if cutlass.const_expr(stage_mode == STAGE_BULK):
            mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), byte_alignment=8)
            sX = smem.allocate_tensor(mX.element_type, cute.make_layout(M), byte_alignment=16)
            stage_row_bulk(sX, mbar, mX[r, None], M * elem_bytes, tid)
        else:
            sX = smem.allocate_tensor(mX.element_type, cute.make_layout(M), byte_alignment=16)
            stage_row(sX, mX[r, None], M, tid, V, bt, elem_bytes)

    Mf = cutlass.Float32(M)

    # --- pass 1: reduce sum(x), sum(x^2) ---
    s1 = cutlass.Float32(0.0)
    s2 = cutlass.Float32(0.0)
    if cutlass.const_expr(staged):
        j = tid
        while j < M:
            x = sX[j].to(cutlass.Float32)
            s1 = s1 + x
            s2 = s2 + x * x
            j = j + bt
    elif cutlass.const_expr(vec):
        xv = cute.zipped_divide(mX[r, None], (V,))
        vi = tid
        while vi < nv:
            xf = cute.make_fragment_like(xv[None, vi])
            cute.autovec_copy(xv[None, vi], xf)
            for e in cutlass.range_constexpr(V):
                x = xf[e].to(cutlass.Float32)
                s1 = s1 + x
                s2 = s2 + x * x
            vi = vi + bt
    else:
        j = tid
        while j < M:
            x = mX[r, j].to(cutlass.Float32)
            s1 = s1 + x
            s2 = s2 + x * x
            j = j + bt
    s1, s2 = block_reduce_sum2(s1, s2, tid, red, bt)

    mean = cutlass.Float32(0.0)
    if cutlass.const_expr(has_mean):
        mean = s1 / Mf
        var = s2 / Mf - mean * mean
    else:
        var = s2 / Mf
    rstd = cute.math.rsqrt(var + eps)
    if tid == 0:
        if cutlass.const_expr(has_mean):
            mMean[r] = mean
        mRstd[r] = rstd

    # --- pass 2: normalize + affine, store Y ---
    if cutlass.const_expr(vec):
        yv = cute.zipped_divide(mY[r, None], (V,))
        xvg = cute.zipped_divide(mX[r, None], (V,)) if cutlass.const_expr(not staged) else None
        vi = tid
        while vi < nv:
            j0 = vi * V
            yf = cute.make_fragment_like(yv[None, vi])
            if cutlass.const_expr(not staged):
                xf = cute.make_fragment_like(xvg[None, vi])
                cute.autovec_copy(xvg[None, vi], xf)
            for e in cutlass.range_constexpr(V):
                x = (sX[j0 + e] if cutlass.const_expr(staged) else xf[e]).to(cutlass.Float32)
                y = mGamma[j0 + e].to(cutlass.Float32) * ((x - mean) * rstd)
                if cutlass.const_expr(has_beta):
                    y = y + mBeta[j0 + e].to(cutlass.Float32)
                yf[e] = y.to(mY.element_type)
            cute.autovec_copy(yf, yv[None, vi])
            vi = vi + bt
    else:
        j = tid
        while j < M:
            x = (sX[j] if cutlass.const_expr(staged) else mX[r, j]).to(cutlass.Float32)
            y = mGamma[j].to(cutlass.Float32) * ((x - mean) * rstd)
            if cutlass.const_expr(has_beta):
                y = y + mBeta[j].to(cutlass.Float32)
            mY[r, j] = y.to(mY.element_type)
            j = j + bt


@cute.jit
def _ln_fwd_host(
    mX,
    mY,
    mGamma,
    mBeta,
    mMean,
    mRstd,
    R: cutlass.Int32,
    eps: cutlass.Float32,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    elem_bytes: cutlass.Constexpr,
    stage_mode: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    _ln_fwd_kernel(
        mX,
        mY,
        mGamma,
        mBeta,
        mMean,
        mRstd,
        eps,
        M,
        V,
        bt,
        elem_bytes,
        stage_mode,
        vec,
        has_mean,
        has_beta,
    ).launch(grid=(R, 1, 1), block=(bt, 1, 1))


_KCACHE = {}


def forward(spec, x2d, gamma, beta, *, eps, cfg, params):
    """Launch LayerNorm/RMSNorm forward. Returns ``(y, mean, rstd)``."""
    import torch

    has_beta = beta is not None
    if beta is None:
        beta = torch.zeros(spec.gamma_len, dtype=x2d.dtype, device=x2d.device)

    y = torch.empty_like(x2d)
    mean = torch.empty(spec.R, dtype=torch.float32, device=x2d.device)
    rstd = torch.empty(spec.R, dtype=torch.float32, device=x2d.device)

    args = (
        dyn(x2d),
        dyn(y),
        dyn(gamma),
        dyn(beta),
        dyn(mean),
        dyn(rstd),
        cutlass.Int32(spec.R),
        cutlass.Float32(eps),
    )
    ce = (spec.M, cfg.V, cfg.block_threads, cfg.elem_bytes, cfg.stage_mode, cfg.vec, spec.has_mean, has_beta)
    key = (params.io_dtype,) + ce
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_ln_fwd_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return y, mean, rstd
