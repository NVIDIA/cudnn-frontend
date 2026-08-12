"""LayerNorm backward, sm_100, CUTLASS primitives.

Given ``dy`` and the forward's saved ``x``, ``gamma``, ``mean``, ``rstd`` this
produces ``dx`` plus ``dgamma``/``dbeta`` (reduced across every group sharing a
channel, via fp32 atomics). Per group ``r`` with ``xhat=(x-mean)*rstd`` and
``dxhat=dy*gamma``::

    dgamma_c += sum_j dy*xhat ;  dbeta_c += sum_j dy      (atomics)
    a = mean_j(dxhat)   (0 for RMSNorm) ;  b = mean_j(dxhat*xhat)
    dx = rstd * (dxhat - a - xhat*b)

X and DY are staged into smem once (``STAGE_BULK`` = two ``cp.async.bulk`` on one
mbarrier; ``STAGE_CPASYNC`` = per-thread cp.async) when they fit; otherwise read
from global (``STAGE_NONE``). The DX store (and, unstaged, the X/DY loads) are
128-bit vectorized when ``M % V == 0``. Also serves RMSNorm (``has_mean=False``).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator

from cudnn.norm.utils import dyn
from cudnn.norm._common_sm100 import (
    STAGE_BULK,
    STAGE_NONE,
    block_reduce_sum2,
    red_scratch_len,
    stage_row,
    stage_two_bulk,
)


@cute.kernel
def _ln_bwd_kernel(
    mDY: cute.Tensor,
    mX: cute.Tensor,
    mGamma: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    mDX: cute.Tensor,
    mDGamma: cute.Tensor,
    mDBeta: cute.Tensor,
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
    sDY = None
    if cutlass.const_expr(staged):
        if cutlass.const_expr(stage_mode == STAGE_BULK):
            mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), byte_alignment=8)
            sX = smem.allocate_tensor(mX.element_type, cute.make_layout(M), byte_alignment=16)
            sDY = smem.allocate_tensor(mDY.element_type, cute.make_layout(M), byte_alignment=16)
            stage_two_bulk(sX, sDY, mbar, mX[r, None], mDY[r, None], M * elem_bytes, tid)
        else:
            sX = smem.allocate_tensor(mX.element_type, cute.make_layout(M), byte_alignment=16)
            sDY = smem.allocate_tensor(mDY.element_type, cute.make_layout(M), byte_alignment=16)
            stage_row(sX, mX[r, None], M, tid, V, bt, elem_bytes)
            stage_row(sDY, mDY[r, None], M, tid, V, bt, elem_bytes)

    Mf = cutlass.Float32(M)
    mean = cutlass.Float32(0.0)
    if cutlass.const_expr(has_mean):
        mean = mMean[r]
    rstd = mRstd[r]

    # --- pass 1: reductions + dgamma/dbeta atomics ---
    s_dxhat = cutlass.Float32(0.0)
    s_dxhat_xhat = cutlass.Float32(0.0)
    if cutlass.const_expr(staged):
        j = tid
        while j < M:
            x = sX[j].to(cutlass.Float32)
            dy = sDY[j].to(cutlass.Float32)
            g = mGamma[j].to(cutlass.Float32)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            s_dxhat = s_dxhat + dxhat
            s_dxhat_xhat = s_dxhat_xhat + dxhat * xhat
            cute.arch.atomic_add(mDGamma.iterator + j, dy * xhat)
            if cutlass.const_expr(has_beta):
                cute.arch.atomic_add(mDBeta.iterator + j, dy)
            j = j + bt
    elif cutlass.const_expr(vec):
        xv = cute.zipped_divide(mX[r, None], (V,))
        dyv = cute.zipped_divide(mDY[r, None], (V,))
        vi = tid
        while vi < nv:
            j0 = vi * V
            xf = cute.make_fragment_like(xv[None, vi])
            dyf = cute.make_fragment_like(dyv[None, vi])
            cute.autovec_copy(xv[None, vi], xf)
            cute.autovec_copy(dyv[None, vi], dyf)
            for e in cutlass.range_constexpr(V):
                x = xf[e].to(cutlass.Float32)
                dy = dyf[e].to(cutlass.Float32)
                g = mGamma[j0 + e].to(cutlass.Float32)
                xhat = (x - mean) * rstd
                dxhat = dy * g
                s_dxhat = s_dxhat + dxhat
                s_dxhat_xhat = s_dxhat_xhat + dxhat * xhat
                cute.arch.atomic_add(mDGamma.iterator + (j0 + e), dy * xhat)
                if cutlass.const_expr(has_beta):
                    cute.arch.atomic_add(mDBeta.iterator + (j0 + e), dy)
            vi = vi + bt
    else:
        j = tid
        while j < M:
            x = mX[r, j].to(cutlass.Float32)
            dy = mDY[r, j].to(cutlass.Float32)
            g = mGamma[j].to(cutlass.Float32)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            s_dxhat = s_dxhat + dxhat
            s_dxhat_xhat = s_dxhat_xhat + dxhat * xhat
            cute.arch.atomic_add(mDGamma.iterator + j, dy * xhat)
            if cutlass.const_expr(has_beta):
                cute.arch.atomic_add(mDBeta.iterator + j, dy)
            j = j + bt

    s_dxhat, s_dxhat_xhat = block_reduce_sum2(s_dxhat, s_dxhat_xhat, tid, red, bt)
    a = cutlass.Float32(0.0)
    if cutlass.const_expr(has_mean):
        a = s_dxhat / Mf
    b = s_dxhat_xhat / Mf

    # --- pass 2: dx ---
    if cutlass.const_expr(vec):
        dxv = cute.zipped_divide(mDX[r, None], (V,))
        xvg = cute.zipped_divide(mX[r, None], (V,)) if cutlass.const_expr(not staged) else None
        dyvg = cute.zipped_divide(mDY[r, None], (V,)) if cutlass.const_expr(not staged) else None
        vi = tid
        while vi < nv:
            j0 = vi * V
            dxf = cute.make_fragment_like(dxv[None, vi])
            if cutlass.const_expr(not staged):
                xf = cute.make_fragment_like(xvg[None, vi])
                dyf = cute.make_fragment_like(dyvg[None, vi])
                cute.autovec_copy(xvg[None, vi], xf)
                cute.autovec_copy(dyvg[None, vi], dyf)
            for e in cutlass.range_constexpr(V):
                x = (sX[j0 + e] if cutlass.const_expr(staged) else xf[e]).to(cutlass.Float32)
                dy = (sDY[j0 + e] if cutlass.const_expr(staged) else dyf[e]).to(cutlass.Float32)
                g = mGamma[j0 + e].to(cutlass.Float32)
                xhat = (x - mean) * rstd
                dxhat = dy * g
                dxf[e] = (rstd * (dxhat - a - xhat * b)).to(mDX.element_type)
            cute.autovec_copy(dxf, dxv[None, vi])
            vi = vi + bt
    else:
        j = tid
        while j < M:
            x = (sX[j] if cutlass.const_expr(staged) else mX[r, j]).to(cutlass.Float32)
            dy = (sDY[j] if cutlass.const_expr(staged) else mDY[r, j]).to(cutlass.Float32)
            g = mGamma[j].to(cutlass.Float32)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            mDX[r, j] = (rstd * (dxhat - a - xhat * b)).to(mDX.element_type)
            j = j + bt


@cute.jit
def _ln_bwd_host(
    mDY, mX, mGamma, mMean, mRstd, mDX, mDGamma, mDBeta,
    R: cutlass.Int32,
    M: cutlass.Constexpr, V: cutlass.Constexpr, bt: cutlass.Constexpr,
    elem_bytes: cutlass.Constexpr, stage_mode: cutlass.Constexpr, vec: cutlass.Constexpr,
    has_mean: cutlass.Constexpr, has_beta: cutlass.Constexpr,
) -> None:
    _ln_bwd_kernel(
        mDY, mX, mGamma, mMean, mRstd, mDX, mDGamma, mDBeta,
        M, V, bt, elem_bytes, stage_mode, vec, has_mean, has_beta,
    ).launch(grid=(R, 1, 1), block=(bt, 1, 1))


_KCACHE = {}


def backward(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, cfg, params):
    """Launch LayerNorm/RMSNorm backward. Returns ``(dx, dgamma, dbeta)``."""
    import torch

    if mean is None:  # RMSNorm has no centering; kernel ignores it when has_mean=False
        mean = torch.empty(spec.R, dtype=torch.float32, device=x2d.device)

    dx = torch.empty_like(x2d)
    dgamma = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dbeta = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x2d.device)

    args = (
        dyn(dy2d), dyn(x2d), dyn(gamma), dyn(mean), dyn(rstd),
        dyn(dx), dyn(dgamma), dyn(dbeta), cutlass.Int32(spec.R),
    )
    ce = (spec.M, cfg.V, cfg.block_threads, cfg.elem_bytes, cfg.stage_mode, cfg.vec, spec.has_mean, has_beta)
    key = (params.io_dtype,) + ce
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_ln_bwd_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return dx, dgamma, (dbeta if has_beta else None)
