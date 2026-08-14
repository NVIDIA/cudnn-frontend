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
    full_warp: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    rit = tid // tpr          # row within the CTA's tile
    tir = tid % tpr           # thread within the row group
    warp_in_row = tir // 32
    lane = tir % 32

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
        # mean/s1/neg are const_expr-guarded so the RMS path carries zero extra
        # live registers (a dead ``neg``/``s1`` tips ldgs=5 into a spill: -23% BW).
        xs = [cutlass.Float32(0.0)] * (ldgs * V)
        s2 = cutlass.Float32(0.0)
        if cutlass.const_expr(has_mean):
            s1 = cutlass.Float32(0.0)
        for it in cutlass.range_constexpr(ldgs):
            col0 = (it * tpr + tir) * V
            xvec = nvvm.load_ext(mXi.iterator + (base + col0), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xvec[e].to(cutlass.Float32)
                xs[it * V + e] = x
                if cutlass.const_expr(has_mean):
                    s1 = s1 + x
                s2 = s2 + x * x
        if cutlass.const_expr(full_warp):
            # wn>1 / full-warp: literal offsets & clamp fold to compile-time constants.
            # (``intra`` is a Constexpr *param* that arrives as a runtime Int32, so
            # ``intra >> k`` would NOT fold — the live intra/clamp/offset registers tip
            # ldgs=5 over the 8-CTA/SM occupancy cliff: -19% BW. Literals avoid that.)
            for k in cutlass.range_constexpr(5):
                off = 32 >> (k + 1)
                if cutlass.const_expr(has_mean):
                    s1 = s1 + nvvm.shfl_sync(0xFFFFFFFF, s1, off, 0x1F, nvvm.Shfl.BFLY)
                s2 = s2 + nvvm.shfl_sync(0xFFFFFFFF, s2, off, 0x1F, nvvm.Shfl.BFLY)
        else:
            clamp: cutlass.Constexpr = 0x1F | ((32 - intra) << 8)  # sub-warp (tiny C)
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
            for j in cutlass.range_constexpr(wn):
                s2 = s2 + red[rit * wn + j]
            if cutlass.const_expr(has_mean):
                s1 = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(wn):
                    s1 = s1 + red[rpc * wn + rit * wn + j]

        if cutlass.const_expr(has_mean):
            mean = s1 * rn
            rstd = cute.math.rsqrt(s2 * rn - mean * mean + eps)
            neg = -rstd * mean
            if tir == 0:
                mMean[row] = mean
                mRstd[row] = rstd
        else:
            rstd = cute.math.rsqrt(s2 * rn + eps)
            if tir == 0:
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
    nsteps: cutlass.Constexpr, full_warp: cutlass.Constexpr,
    has_mean: cutlass.Constexpr, has_beta: cutlass.Constexpr,
    mbpm: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _warp_fwd_kernel(
        mXi, mYi, mGamma, mBeta, mMean, mRstd, R, ctas, eps,
        C, V, tpr, wn, intra, ldgs, rpc, block_threads, et, it_ty, nsteps, full_warp, has_mean, has_beta,
    ).launch(grid=(ctas, 1, 1), block=(block_threads, 1, 1), min_blocks_per_mp=mbpm)


# min CTAs/SM (launch_bounds minnctapersm). 0 = compiler default (measured: forcing
# it spills and hurts, so left off).
_MBPM = 0
# Persistent grid: cap the grid so each CTA grid-strides over many rows, overlapping
# row i's reduction latency with row i+1's loads (a one-tile-per-CTA grid exposes
# that latency and loses ~30% bandwidth). The cap multiple of SM_COUNT is picked to
# hold a *constant working set per SM*: ``mult * (block_threads * ldgs) ~ 4096``, i.e.
# ``mult ~ 4096 / (block_threads*ldgs)`` (== round(32768/C) at rpc=1). This matches
# the measured occupancy sweet spot for every large-C shape and, crucially, lands on
# the safe side of a sharp occupancy cliff: e.g. C=5120 peaks at 6-7x (0.91x) but
# 8x collapses to 0.77x. For wn==1 (small C, cheap 1-2-ldg rows) latency is hidden by
# raw occupancy, so fill the machine (mult=8; clamped to full_ctas for huge-N shapes).
_CTAS_CAP = 0  # test override; 0 = use the formula
_SM_COUNT = None


def _persist_cap(full_ctas, block_threads, wn, ldgs):
    global _SM_COUNT
    if _CTAS_CAP:
        return min(full_ctas, _CTAS_CAP)
    if _SM_COUNT is None:
        import torch
        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    if wn > 1:
        mult = max(2, min(16, round(4096 / (block_threads * ldgs))))
    else:
        mult = 8
    return min(full_ctas, _SM_COUNT * mult)


_KCACHE = {}


def forward(spec, x2d, gamma, beta, *, eps, wcfg, params):
    """Launch the warp-per-row LN/RMS forward. ``wcfg`` = (tpr, wn, intra, ldgs, rpc, block_threads, V)."""
    import torch

    tpr, wn, intra, ldgs, rpc, block_threads, V = wcfg
    has_beta = beta is not None

    R, C = spec.R, spec.M
    ctas = _persist_cap((R + rpc - 1) // rpc, block_threads, wn, ldgs)  # persistent grid
    y = torch.empty_like(x2d)
    rstd = torch.empty(R, dtype=torch.float32, device=x2d.device)
    # Unused kernel operands (beta for RMS, mean for RMS) alias an existing tensor:
    # a distinct dummy alloc adds a per-call memset (counted in device-time) and an
    # extra pointer register that tips ldgs=5 below the 8-CTA/SM occupancy cliff.
    mean = torch.empty(R, dtype=torch.float32, device=x2d.device) if spec.has_mean else rstd
    if beta is None:
        beta = gamma

    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[DTYPE_BYTES[params.io_dtype]]
    nsteps = int(intra).bit_length() - 1
    full_warp = int(intra) == 32

    args = (dyn(x2d), dyn(y), dyn(gamma), dyn(beta), dyn(mean), dyn(rstd),
            cutlass.Int32(R), cutlass.Int32(ctas), cutlass.Float32(eps))
    ce = (C, V, tpr, wn, intra, ldgs, rpc, block_threads, et, it_ty, nsteps, full_warp, spec.has_mean, has_beta, _MBPM)
    key = (params.io_dtype, C, tpr, wn, intra, ldgs, rpc, block_threads, spec.has_mean, has_beta, _MBPM)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_warp_fwd_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return y, mean, rstd
