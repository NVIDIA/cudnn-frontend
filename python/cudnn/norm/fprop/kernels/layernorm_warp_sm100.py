"""LayerNorm/RMSNorm forward, sm_100 — warp-per-row, CUTLASS primitives.

Modeled on cuDNN's ln_fwd kernel. Built on ``cutlass.primitives`` (nvvm):

- ``tpr`` threads own one row. For large C the row is spread over ``wn = tpr//32``
  warps (``THREADS_PER_ROW = wn*32``) so each thread caches only ~``ldgs*V``
  elements (small register footprint — avoids the spill that halves bandwidth at
  large C). For tiny C, ``tpr<32`` (sub-warp) and a CTA packs several rows/warp.
- reduction: ``nvvm.shfl_sync`` butterfly within the ``intra``-lane group, then
  (``wn>1``) a cross-warp combine of the ``wn`` partials through smem.
- X loaded once via ``nvvm.load_ext`` (vectorized; recast->int + bitcast) into
  registers, reduction fused into the load; gamma/beta loaded once into smem.
- FMA-fused affine; vectorized store via ``nvvm.store_ext``. Grid-stride persistent.

Selected by ``make_warp_cfg`` for LN/RMS (norm dim C == gamma length).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.utils import dyn

_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}


@cute.kernel
def _warp_fwd_kernel(
    mXi: cute.Tensor,
    mYi: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    R: cutlass.Int32,
    ctas: cutlass.Int32,
    eps: cutlass.Float32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    tpr: cutlass.Constexpr,
    wn: cutlass.Constexpr,
    intra: cutlass.Constexpr,
    ldgs: cutlass.Constexpr,
    rpc: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
    et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    nsteps: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    rit = tid // tpr          # row within the CTA's tile
    tir = tid % tpr           # thread within the row group
    warp_in_row = tir // 32
    lane = tir % 32
    clamp: cutlass.Constexpr = 0x1F | ((32 - intra) << 8)

    smem = SmemAllocator()
    sG = smem.allocate_tensor(mGamma.element_type, cute.make_layout(C), byte_alignment=16)
    sB = None
    if cutlass.const_expr(has_beta):
        sB = smem.allocate_tensor(mBeta.element_type, cute.make_layout(C), byte_alignment=16)
    red = None
    if cutlass.const_expr(wn > 1):
        red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(rpc * wn * (2 if has_mean else 1)), byte_alignment=8)

    i = tid
    while i < C:
        sG[i] = mGamma[i]
        if cutlass.const_expr(has_beta):
            sB[i] = mBeta[i]
        i = i + block_threads
    cute.arch.sync_threads()

    Cf = cutlass.Float32(C)
    rn = 1.0 / Cf
    stride = ctas * rpc
    row = bid * rpc + rit
    while row < R:
        base = cutlass.Int64(row) * C

        # --- pass 1: load X to registers (vectorized), fused reduce ---
        xs = [cutlass.Float32(0.0)] * (ldgs * V)
        s1 = cutlass.Float32(0.0)
        s2 = cutlass.Float32(0.0)
        for it in cutlass.range_constexpr(ldgs):
            col0 = (it * tpr + tir) * V
            xvec = nvvm.load_ext(mXi.iterator + (base + col0), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xvec[e].to(cutlass.Float32)
                xs[it * V + e] = x
                if cutlass.const_expr(has_mean):
                    s1 = s1 + x
                s2 = s2 + x * x
        for k in cutlass.range_constexpr(nsteps):
            o = intra >> (k + 1)
            if cutlass.const_expr(has_mean):
                s1 = s1 + nvvm.shfl_sync(0xFFFFFFFF, s1, o, clamp, nvvm.Shfl.BFLY)
            s2 = s2 + nvvm.shfl_sync(0xFFFFFFFF, s2, o, clamp, nvvm.Shfl.BFLY)

        if cutlass.const_expr(wn > 1):  # combine the wn warp-partials via smem
            if lane == 0:
                red[rit * wn + warp_in_row] = s2
                if cutlass.const_expr(has_mean):
                    red[rpc * wn + rit * wn + warp_in_row] = s1
            cute.arch.sync_threads()
            s2 = cutlass.Float32(0.0)
            s1 = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(wn):
                s2 = s2 + red[rit * wn + j]
                if cutlass.const_expr(has_mean):
                    s1 = s1 + red[rpc * wn + rit * wn + j]

        mean = cutlass.Float32(0.0)
        if cutlass.const_expr(has_mean):
            mean = s1 * rn
            var = s2 * rn - mean * mean
        else:
            var = s2 * rn
        rstd = cute.math.rsqrt(var + eps)
        neg = -rstd * mean
        if tir == 0:
            if cutlass.const_expr(has_mean):
                mMean[row] = mean
            mRstd[row] = rstd

        # --- pass 2: normalize from registers + affine, vectorized store ---
        for it in cutlass.range_constexpr(ldgs):
            col0 = (it * tpr + tir) * V
            ys = []
            for e in cutlass.range_constexpr(V):
                if cutlass.const_expr(has_mean):
                    y = rstd * xs[it * V + e] + neg
                else:
                    y = rstd * xs[it * V + e]
                y = sG[col0 + e].to(cutlass.Float32) * y
                if cutlass.const_expr(has_beta):
                    y = y + sB[col0 + e].to(cutlass.Float32)
                ys.append(y.to(et))
            yvec = cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty)
            nvvm.store_ext(yvec, mYi.iterator + (base + col0))
        if cutlass.const_expr(wn > 1):
            cute.arch.sync_threads()  # red reused next iteration
        row = row + stride


@cute.jit
def _warp_fwd_host(
    mX, mY, mGamma, mBeta, mMean, mRstd,
    R: cutlass.Int32, ctas: cutlass.Int32, eps: cutlass.Float32,
    C: cutlass.Constexpr, V: cutlass.Constexpr, tpr: cutlass.Constexpr, wn: cutlass.Constexpr,
    intra: cutlass.Constexpr, ldgs: cutlass.Constexpr, rpc: cutlass.Constexpr,
    block_threads: cutlass.Constexpr, et: cutlass.Constexpr, it_ty: cutlass.Constexpr,
    nsteps: cutlass.Constexpr, has_mean: cutlass.Constexpr, has_beta: cutlass.Constexpr,
    mbpm: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _warp_fwd_kernel(
        mXi, mYi, mGamma, mBeta, mMean, mRstd, R, ctas, eps,
        C, V, tpr, wn, intra, ldgs, rpc, block_threads, et, it_ty, nsteps, has_mean, has_beta,
    ).launch(grid=(ctas, 1, 1), block=(block_threads, 1, 1), min_blocks_per_mp=mbpm)


# min CTAs/SM (launch_bounds minnctapersm). 0 = compiler default (measured: forcing
# it spills and hurts, so left off).
_MBPM = 0
# Persistent grid: cap the grid at SM_COUNT * _PERSIST_MULT so each CTA grid-strides
# over many rows, overlapping row i's reduction latency with row i+1's loads (a
# one-tile-per-CTA grid exposes that latency and loses ~30% bandwidth). Measured
# sweet spot ~4x SM count on this GPU. 0 override = one tile per CTA.
_PERSIST_MULT = 4
_CTAS_CAP = 0  # test override; 0 = use _PERSIST_MULT * SM count
_SM_COUNT = None


def _persist_cap():
    global _SM_COUNT
    if _CTAS_CAP:
        return _CTAS_CAP
    if _SM_COUNT is None:
        import torch
        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT * _PERSIST_MULT


_KCACHE = {}


def forward(spec, x2d, gamma, beta, *, eps, wcfg, params):
    """Launch the warp-per-row LN/RMS forward. ``wcfg`` = (tpr, wn, intra, ldgs, rpc, block_threads, V)."""
    import torch

    tpr, wn, intra, ldgs, rpc, block_threads, V = wcfg
    has_beta = beta is not None
    if beta is None:
        beta = torch.zeros(spec.gamma_len, dtype=x2d.dtype, device=x2d.device)

    R, C = spec.R, spec.M
    ctas = min((R + rpc - 1) // rpc, _persist_cap())  # persistent grid
    y = torch.empty_like(x2d)
    mean = torch.empty(R, dtype=torch.float32, device=x2d.device)
    rstd = torch.empty(R, dtype=torch.float32, device=x2d.device)

    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[DTYPE_BYTES[params.io_dtype]]
    nsteps = int(intra).bit_length() - 1

    args = (dyn(x2d), dyn(y), dyn(gamma), dyn(beta), dyn(mean), dyn(rstd),
            cutlass.Int32(R), cutlass.Int32(ctas), cutlass.Float32(eps))
    ce = (C, V, tpr, wn, intra, ldgs, rpc, block_threads, et, it_ty, nsteps, spec.has_mean, has_beta, _MBPM)
    key = (params.io_dtype, C, tpr, wn, intra, ldgs, rpc, block_threads, spec.has_mean, has_beta, _MBPM)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_warp_fwd_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return y, mean, rstd
