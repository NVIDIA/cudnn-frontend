"""GroupNorm forward, sm_100, CUTLASS primitives.

One CTA owns one ``(sample, group)`` — a row of the ``[R, M]`` view with
``R = N*G`` and ``M = (C/G)*HW``. Same staged/vectorized two-pass reduction as
LayerNorm (see :mod:`layernorm_sm100` for the ``stage_mode`` / ``vec`` knobs),
but the affine channel for element ``(r, j)`` is
``(r % groups_per_sample) * channels_per_group + j // gamma_inner_span``
(runtime args). InstanceNorm reuses this kernel as GroupNorm with ``G = C``.
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
    stage_row_bulk,
)


@cute.kernel
def _gn_fwd_kernel(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    gps: cutlass.Int32,
    cpg: cutlass.Int32,
    span: cutlass.Int32,
    eps: cutlass.Float32,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    elem_bytes: cutlass.Constexpr,
    stage_mode: cutlass.Constexpr,
    vec: cutlass.Constexpr,
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
    base_c = (r % gps) * cpg

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

    mean = s1 / Mf
    var = s2 / Mf - mean * mean
    rstd = cute.math.rsqrt(var + eps)
    if tid == 0:
        mMean[r] = mean
        mRstd[r] = rstd

    # --- pass 2: normalize + affine (per-channel gamma/beta), store Y ---
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
                c = base_c + ((j0 + e) // span)
                y = mGamma[c].to(cutlass.Float32) * ((x - mean) * rstd)
                if cutlass.const_expr(has_beta):
                    y = y + mBeta[c].to(cutlass.Float32)
                yf[e] = y.to(mY.element_type)
            cute.autovec_copy(yf, yv[None, vi])
            vi = vi + bt
    else:
        j = tid
        while j < M:
            x = (sX[j] if cutlass.const_expr(staged) else mX[r, j]).to(cutlass.Float32)
            c = base_c + (j // span)
            y = mGamma[c].to(cutlass.Float32) * ((x - mean) * rstd)
            if cutlass.const_expr(has_beta):
                y = y + mBeta[c].to(cutlass.Float32)
            mY[r, j] = y.to(mY.element_type)
            j = j + bt


@cute.jit
def _gn_fwd_host(
    mX, mY, mGamma, mBeta, mMean, mRstd,
    R: cutlass.Int32, gps: cutlass.Int32, cpg: cutlass.Int32, span: cutlass.Int32,
    eps: cutlass.Float32,
    M: cutlass.Constexpr, V: cutlass.Constexpr, bt: cutlass.Constexpr,
    elem_bytes: cutlass.Constexpr, stage_mode: cutlass.Constexpr, vec: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    _gn_fwd_kernel(
        mX, mY, mGamma, mBeta, mMean, mRstd, gps, cpg, span, eps,
        M, V, bt, elem_bytes, stage_mode, vec, has_beta,
    ).launch(grid=(R, 1, 1), block=(bt, 1, 1))


_KCACHE = {}


def forward(spec, x2d, gamma, beta, *, eps, cfg, params):
    """Launch GroupNorm/InstanceNorm forward. Returns ``(y, mean, rstd)``."""
    import torch

    has_beta = beta is not None
    if beta is None:
        beta = torch.zeros(spec.gamma_len, dtype=x2d.dtype, device=x2d.device)

    y = torch.empty_like(x2d)
    mean = torch.empty(spec.R, dtype=torch.float32, device=x2d.device)
    rstd = torch.empty(spec.R, dtype=torch.float32, device=x2d.device)

    args = (
        dyn(x2d), dyn(y), dyn(gamma), dyn(beta), dyn(mean), dyn(rstd),
        cutlass.Int32(spec.R), cutlass.Int32(spec.groups_per_sample),
        cutlass.Int32(spec.channels_per_group), cutlass.Int32(spec.gamma_inner_span),
        cutlass.Float32(eps),
    )
    ce = (spec.M, cfg.V, cfg.block_threads, cfg.elem_bytes, cfg.stage_mode, cfg.vec, has_beta)
    key = (params.io_dtype,) + ce
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_gn_fwd_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return y, mean, rstd
