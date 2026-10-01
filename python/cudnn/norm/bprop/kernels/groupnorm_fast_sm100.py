# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic-free GroupNorm / InstanceNorm backward, sm_100, CUTLASS primitives.

The original backward issued **one global fp32 atomic per element** into a
``gamma_len``-sized accumulator. With gamma_len = C (64..512) that is millions of
atomics colliding on a handful of addresses: 5-130 GB/s, i.e. 0.00-0.04 of
achievable bandwidth and up to 100x slower than PyTorch.

**The fix is a work map that makes the per-channel reduction register-local.**
Row ``r`` of the rowwise view is group ``r % gps`` of sample ``r // gps``, and the
affine channel of element ``(r, j)`` is ``(r % gps)*CPG + j//span``. If a CTA only
ever sees rows with the SAME ``r % gps``, its channel set ``[base_c, base_c+CPG)``
is fixed for the whole launch. Choosing ``ctas`` as a multiple of ``gps`` and
grid-striding by ``ctas`` guarantees exactly that.

So: thread ``t`` owns channel ``base_c + t//tps`` (``tps = bt//CPG`` threads per
channel segment) and accumulates that channel's ``dgamma``/``dbeta`` in two
registers across every row the CTA touches. At the end each CTA flushes
``[2, CPG]`` values -- not ``[2, C]`` -- and a tiny finalize kernel sums the
``ctas/gps`` contributors per channel. **No atomics anywhere.**

The row-local reduction (``sum(dxhat)``, ``sum(dxhat*xhat)``) still spans the whole
row, so it stays a block reduce -- one per row, not one per element.

The row is staged into shared memory on the first read so the dx pass does not go
back to global: traffic is then the minimal 3 units (read dy, read x, write dx).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.utils import dyn

_CTA_SS = nvvm.SharedSpace.shared_cta
_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_FULL = 0xFFFFFFFF
_BFLY_CLAMP = 0x1F  # shfl width == 32
_SMEM_CAP = 160 * 1024
_SM_COUNT = None


def _nsm():
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


def red_scratch_len(block_threads: int) -> int:
    """fp32 scratch for :func:`_block_sum2` -- one slot per warp, per accumulator."""
    return 2 * (block_threads // 32)


@cute.jit
def _block_sum2(v1, v2, tid, red, bt: cutlass.Constexpr):
    """Reduce two fp32 partials across the CTA, broadcast to all threads.

    nvvm only -- a butterfly shuffle inside each warp, then one shared round across
    warps. The shared helper in _common_sm100 uses ``cute.arch.warp_reduction_sum``
    and ``cute.arch.sync_threads``; this kernel must stay on cutlass primitives.
    """
    nwarps: cutlass.Constexpr = bt // 32
    warp = tid // 32
    lane = tid % 32
    for d in cutlass.range_constexpr(5):
        off = 1 << d
        v1 = v1 + nvvm.shfl_sync(_FULL, v1, off, _BFLY_CLAMP, nvvm.Shfl.BFLY)
        v2 = v2 + nvvm.shfl_sync(_FULL, v2, off, _BFLY_CLAMP, nvvm.Shfl.BFLY)
    if cutlass.const_expr(nwarps == 1):
        return v1, v2
    if lane == 0:
        red[warp] = v1
        red[nwarps + warp] = v2
    nvvm.barrier_cta_sync_aligned(0)
    a = cutlass.Float32(0.0)
    b = cutlass.Float32(0.0)
    for w in cutlass.range_constexpr(nwarps):
        a = a + red[w]
        b = b + red[nwarps + w]
    nvvm.barrier_cta_sync_aligned(0)
    return a, b


@cute.kernel
def _gn_bwd_fast_kernel(
    mDY,
    mDYi,
    mX,
    mXi,
    mDX,
    mDXi,
    mG,
    mMean,
    mRstd,
    mDGp,
    mDBp,
    R: cutlass.Int32,
    ctas: cutlass.Int32,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    CPG: cutlass.Constexpr,
    SPAN: cutlass.Constexpr,
    GPS: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    TPS: cutlass.Constexpr = bt // CPG  # threads per channel segment
    NVS: cutlass.Constexpr = SPAN // V  # vectors in one channel's span
    Mf = cutlass.Float32(M)

    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(red_scratch_len(bt)), byte_alignment=8)
    seg_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * bt), byte_alignment=16)
    cty: cutlass.Constexpr = et if V == 1 else it_ty
    sX = smem.allocate_tensor(cty, cute.make_layout(M), byte_alignment=16)
    sD = smem.allocate_tensor(cty, cute.make_layout(M), byte_alignment=16)

    seg = tid // TPS  # which channel of the group this thread serves
    lane = tid % TPS
    base_c = (bid % GPS) * CPG  # FIXED for this CTA -- that is what kills the atomics
    gch = base_c + seg
    g_seg = mG[gch].to(cutlass.Float32)

    dg = cutlass.Float32(0.0)
    db = cutlass.Float32(0.0)

    row = bid
    while row < R:
        mean = cutlass.Float32(0.0)
        if cutlass.const_expr(has_mean):
            mean = mMean[row]
        rstd = mRstd[row]
        base = cutlass.Int64(row) * M

        # ---- pass 1: stage the row, row-wide reduction + this thread's channel sums ----
        s1 = cutlass.Float32(0.0)
        s2 = cutlass.Float32(0.0)
        kv = lane
        while kv < NVS:
            off = seg * SPAN + kv * V
            if cutlass.const_expr(V == 1):
                rx = mX[base + off]
                rd = mDY[base + off]
                sX[off] = rx
                sD[off] = rd
                xs = [rx.to(cutlass.Float32)]
                ds = [rd.to(cutlass.Float32)]
            else:
                rx = nvvm.load_ext(mXi.iterator + (base + off), dtype=it_ty, count=V)
                rd = nvvm.load_ext(mDYi.iterator + (base + off), dtype=it_ty, count=V)
                nvvm.store_ext(rx, sX.iterator + off, shared_space=_CTA_SS)
                nvvm.store_ext(rd, sD.iterator + off, shared_space=_CTA_SS)
                xs = [rx.bitcast(et)[e].to(cutlass.Float32) for e in range(V)]
                ds = [rd.bitcast(et)[e].to(cutlass.Float32) for e in range(V)]
            for e in cutlass.range_constexpr(V):
                xh = (xs[e] - mean) * rstd
                dy = ds[e]
                dxh = dy * g_seg
                s1 = s1 + dxh
                s2 = s2 + dxh * xh
                dg = dg + dy * xh
                db = db + dy
            kv = kv + TPS
        s1, s2 = _block_sum2(s1, s2, tid, red, bt)
        a = s1 / Mf
        b = s2 / Mf

        # ---- pass 2: dx from the staged row (no second global read) ----
        kv = lane
        while kv < NVS:
            off = seg * SPAN + kv * V
            if cutlass.const_expr(V == 1):
                xh = (sX[off].to(cutlass.Float32) - mean) * rstd
                dxh = sD[off].to(cutlass.Float32) * g_seg
                mDX[base + off] = (rstd * (dxh - a - xh * b)).to(et)
            else:
                xv = nvvm.load_ext(sX.iterator + off, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                dv = nvvm.load_ext(sD.iterator + off, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                ys = []
                for e in cutlass.range_constexpr(V):
                    xh = (xv[e].to(cutlass.Float32) - mean) * rstd
                    dxh = dv[e].to(cutlass.Float32) * g_seg
                    ys.append((rstd * (dxh - a - xh * b)).to(et))
                nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (base + off))
            kv = kv + TPS
        nvvm.barrier_cta_sync_aligned(0)
        row = row + ctas

    # ---- flush: reduce each channel segment's TPS threads, write [ctas, 2, CPG] ----
    seg_red[tid] = dg
    seg_red[bt + tid] = db
    nvvm.barrier_cta_sync_aligned(0)
    if tid < CPG:
        agg = cutlass.Float32(0.0)
        aggb = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr(TPS):
            agg = agg + seg_red[tid * TPS + k]
            aggb = aggb + seg_red[bt + tid * TPS + k]
        mDGp[bid * CPG + tid] = agg
        if cutlass.const_expr(has_beta):
            mDBp[bid * CPG + tid] = aggb


@cute.kernel
def _gn_bwd_finalize_kernel(
    mDGp,
    mDBp,
    mDGamma,
    mDBeta,
    ctas: cutlass.Int32,
    CPG: cutlass.Constexpr,
    GPS: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    c = bid * bt + tid
    if c < GPS * CPG:
        gi = c // CPG
        k = c % CPG
        sg = cutlass.Float32(0.0)
        sb = cutlass.Float32(0.0)
        j = cutlass.Int32(gi)  # CTAs with (cta % GPS) == gi
        while j < ctas:
            sg = sg + mDGp[j * CPG + k]
            if cutlass.const_expr(has_beta):
                sb = sb + mDBp[j * CPG + k]
            j = j + GPS
        mDGamma[c] = sg
        if cutlass.const_expr(has_beta):
            mDBeta[c] = sb


@cute.jit
def _gn_bwd_fast_host(
    mDY,
    mX,
    mDX,
    mG,
    mMean,
    mRstd,
    mDGp,
    mDBp,
    mDGamma,
    mDBeta,
    R,
    ctas,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    CPG: cutlass.Constexpr,
    SPAN: cutlass.Constexpr,
    GPS: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
    fgrid: cutlass.Constexpr,
    fbt: cutlass.Constexpr,
) -> None:
    mDYi = cute.recast_tensor(mDY, it_ty)
    mXi = cute.recast_tensor(mX, it_ty)
    mDXi = cute.recast_tensor(mDX, it_ty)
    _gn_bwd_fast_kernel(
        mDY,
        mDYi,
        mX,
        mXi,
        mDX,
        mDXi,
        mG,
        mMean,
        mRstd,
        mDGp,
        mDBp,
        R,
        ctas,
        M,
        V,
        CPG,
        SPAN,
        GPS,
        bt,
        it_ty,
        et,
        has_mean,
        has_beta,
    ).launch(grid=(ctas, 1, 1), block=(bt, 1, 1), smem=smem_bytes)
    _gn_bwd_finalize_kernel(mDGp, mDBp, mDGamma, mDBeta, ctas, CPG, GPS, fbt, has_beta).launch(grid=(fgrid, 1, 1), block=(fbt, 1, 1))


_KCACHE = {}


def _vec_for(span, elem_bytes):
    """Widest <=128-bit vector the channel span divides; 1 (scalar path) for an
    odd span such as 7x7 -> 49."""
    v = 16 // elem_bytes
    while v > 1 and span % v != 0:
        v //= 2
    return v


def eligible(spec, elem_bytes):
    """True when the atomic-free map applies: a channel count per group that
    divides the block, and a row that fits the shared-memory stage."""
    cpg = int(spec.channels_per_group)
    if cpg < 1 or cpg > 1024:
        return False
    if int(spec.gamma_inner_span) < 1:
        return False
    if 2 * int(spec.M) * elem_bytes + 2 * 1024 * 4 + 256 > _SMEM_CAP:
        return False
    return True


def backward(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, cfg, params):
    """Launch the atomic-free GN/IN backward. Returns ``(dx, dgamma, dbeta)``."""
    import torch

    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    M = int(spec.M)
    CPG = int(spec.channels_per_group)
    SPAN = int(spec.gamma_inner_span)
    V = _vec_for(SPAN, eb)
    GPS = int(spec.groups_per_sample)
    R = int(spec.R)
    # Size the block to the ROW, not to a fixed 256. InstanceNorm has CPG=1, so a
    # whole block serves one span of M elements; at 7x7 that is 49 vectors and a
    # 256-thread block leaves 80% of its lanes idle. Round the vector count up to
    # a warp, then up to a multiple of CPG so the segment split stays exact.
    nvec = max(1, M // V)
    bt = min(256, max(32, ((nvec + 31) // 32) * 32))
    if bt < CPG:
        bt = CPG
    if bt % CPG:
        bt = ((bt + CPG - 1) // CPG) * CPG
    bt = min(bt, 1024)

    # ctas MUST be a multiple of GPS so every row a CTA sees shares its group
    # index -- that is what makes the channel set fixed and the partials [ctas,CPG].
    target = max(1, min(4 * _nsm(), R))
    ctas = max(GPS, (target // GPS) * GPS)
    if ctas > R:
        ctas = max(GPS, (R // GPS) * GPS) or GPS

    dx = torch.empty_like(x2d)
    dgamma = torch.empty(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dbeta = torch.empty(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dgp = torch.zeros(ctas * CPG, dtype=torch.float32, device=x2d.device)
    dbp = torch.zeros(ctas * CPG, dtype=torch.float32, device=x2d.device)
    if mean is None:
        mean = rstd

    smem_bytes = 2 * M * eb + 2 * bt * 4 + red_scratch_len(bt) * 4 + 256
    fbt = 128
    fgrid = (GPS * CPG + fbt - 1) // fbt

    # Flat views: the kernel addresses rows by `row*M + off`, so the scalar path's
    # `mX[idx]` must index a 1-D tensor (the vector path uses iterator arithmetic
    # and was unaffected, which is why only V==1 shapes were wrong).
    dyf, xf, dxf = dy2d.reshape(-1), x2d.reshape(-1), dx.reshape(-1)
    args = (dyn(dyf), dyn(xf), dyn(dxf), dyn(gamma), dyn(mean), dyn(rstd), dyn(dgp), dyn(dbp), dyn(dgamma), dyn(dbeta), cutlass.Int32(R), cutlass.Int32(ctas))
    ce = (M, V, CPG, SPAN, GPS, bt, it_ty, et, spec.has_mean, has_beta, smem_bytes, fgrid, fbt)
    key = (params.io_dtype, M, V, CPG, SPAN, GPS, bt, spec.has_mean, has_beta, ctas)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_gn_bwd_fast_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return dx, dgamma, (dbeta if has_beta else None)
