# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BatchNorm backward for NCHW, sm_100, CUTLASS primitives.

Mirrors the forward's work map (``fprop/kernels/batchnorm_nchw_sm100.py``): a CTA
owns ``CT = WPB*CPW`` channels, warp ``w`` owns channels ``c0 + cw*WPB + w`` and
*only* those, so the two per-channel reductions live in **registers** and the
cross-CTA reduce needs **no atomics**. ``mparts`` CTAs split the image axis.

The math, for channel ``c`` over its ``M = N*S`` elements with
``xhat = (x-mean)*rstd``::

    dgamma = sum(dy * xhat)        dbeta = sum(dy)
    dx     = gamma*rstd * (dy - dbeta/M - xhat * dgamma/M)

So pass 1 reduces ``sum(dy)`` and ``sum(dy*xhat)``, and pass 2 needs **both** dy
and x again. Minimal traffic is 3 units (read dy, read x, write dx); an uncached
two-pass moves 5, which caps utilisation at 0.6. ``KSR`` whole image-rows per warp
are therefore cached in shared on the pass-1 read -- twice the forward's footprint
per row, since both streams must be kept.

Pass 2's affine is a **scalar per row** (one channel per row), staged into
registers once, so the normalize is a pure streaming pass.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.fprop.kernels.batchnorm_nchw_sm100 import nchw_cfg
from cudnn.norm.utils import dyn, smem_budget, smem_per_sm

_CTA_SS = nvvm.SharedSpace.shared_cta
_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_FULL = 0xFFFFFFFF
_BFLY_CLAMP = 0x1F
_SMEM_CAP_PREF = 200 * 1024  # measured on sm_100; smem_budget clamps it elsewhere
_SM_COUNT = None


def _nsm():
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


@cute.jit
def _warp_sum2(a, b):
    """Butterfly all-reduce of two fp32 partials across a full warp."""
    for d in cutlass.range_constexpr(5):
        off = 1 << d
        a = a + nvvm.shfl_sync(_FULL, a, off, _BFLY_CLAMP, nvvm.Shfl.BFLY)
        b = b + nvvm.shfl_sync(_FULL, b, off, _BFLY_CLAMP, nvvm.Shfl.BFLY)
    return a, b


@cute.kernel
def _bn_bwd_nchw_kernel(
    mDY,
    mDYi,
    mX,
    mXi,
    mDX,
    mDXi,
    mG,
    mMean,
    mRstd,
    mP,
    mRet,
    mDGamma,
    mDBeta,
    N: cutlass.Int32,
    mparts: cutlass.Int32,
    C: cutlass.Constexpr,
    S: cutlass.Constexpr,
    VS: cutlass.Constexpr,
    UF: cutlass.Constexpr,
    WPB: cutlass.Constexpr,
    CPW: cutlass.Constexpr,
    CT: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    KSR: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cx, my, _ = cute.arch.block_idx()
    w = tid // 32
    lane = tid % 32
    NV: cutlass.Constexpr = S // VS
    STEP: cutlass.Constexpr = UF * 32
    CSZ: cutlass.Constexpr = WPB * CPW * KSR * S

    smem = SmemAllocator()
    stat = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * CT), byte_alignment=16)
    part = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * CT), byte_alignment=16)
    scx = None
    scd = None
    if cutlass.const_expr(KSR > 0):
        cty: cutlass.Constexpr = et if VS == 1 else it_ty
        scx = smem.allocate_tensor(cty, cute.make_layout(CSZ), byte_alignment=16)
        scd = smem.allocate_tensor(cty, cute.make_layout(CSZ), byte_alignment=16)

    c0 = cx * CT
    per = (N + mparts - 1) // mparts
    n0 = my * per
    n1 = n0 + per
    if n1 > N:
        n1 = N
    nc = n0 + KSR
    if nc > n1:
        nc = n1

    # per-warp owned channels: mean/rstd are scalars here, staged once
    mn = [cutlass.Float32(0.0)] * CPW
    rs = [cutlass.Float32(0.0)] * CPW
    for cw in cutlass.range_constexpr(CPW):
        ci = c0 + cw * WPB + w
        mn[cw] = mMean[ci].to(cutlass.Float32)
        rs[cw] = mRstd[ci].to(cutlass.Float32)

    acc = [cutlass.Float32(0.0)] * CPW  # sum(dy)
    accx = [cutlass.Float32(0.0)] * CPW  # sum(dy*xhat)

    # ---- pass 1a: cached images (stash BOTH streams on the single read) ----
    if cutlass.const_expr(KSR > 0):
        n = n0
        r = cutlass.Int32(0)
        while n < nc:
            bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
            cbs = [((w * CPW + cw) * KSR + r) * S for cw in range(CPW)]
            kb = cutlass.Int32(0)
            while kb + STEP <= NV:
                xs = []
                ds = []
                for j in cutlass.range_constexpr(UF):
                    for cw in cutlass.range_constexpr(CPW):
                        o = kb + j * 32 + lane
                        if cutlass.const_expr(VS == 1):
                            xs.append(mX[bs[cw] + o])
                            ds.append(mDY[bs[cw] + o])
                        else:
                            xs.append(nvvm.load_ext(mXi.iterator + (bs[cw] + o * VS), dtype=it_ty, count=VS))
                            ds.append(nvvm.load_ext(mDYi.iterator + (bs[cw] + o * VS), dtype=it_ty, count=VS))
                i = 0
                for j in cutlass.range_constexpr(UF):
                    for cw in cutlass.range_constexpr(CPW):
                        o = kb + j * 32 + lane
                        if cutlass.const_expr(VS == 1):
                            scx[cbs[cw] + o] = xs[i]
                            scd[cbs[cw] + o] = ds[i]
                            xv = [xs[i].to(cutlass.Float32)]
                            dv = [ds[i].to(cutlass.Float32)]
                        else:
                            nvvm.store_ext(xs[i], scx.iterator + (cbs[cw] + o * VS), shared_space=_CTA_SS)
                            nvvm.store_ext(ds[i], scd.iterator + (cbs[cw] + o * VS), shared_space=_CTA_SS)
                            xvv = xs[i].bitcast(et)
                            dvv = ds[i].bitcast(et)
                            xv = [xvv[e].to(cutlass.Float32) for e in range(VS)]
                            dv = [dvv[e].to(cutlass.Float32) for e in range(VS)]
                        for e in cutlass.range_constexpr(VS):
                            acc[cw] = acc[cw] + dv[e]
                            accx[cw] = accx[cw] + dv[e] * ((xv[e] - mn[cw]) * rs[cw])
                        i = i + 1
                kb = kb + STEP
            kv = kb + lane
            while kv < NV:
                for cw in cutlass.range_constexpr(CPW):
                    if cutlass.const_expr(VS == 1):
                        rx = mX[bs[cw] + kv]
                        rd = mDY[bs[cw] + kv]
                        scx[cbs[cw] + kv] = rx
                        scd[cbs[cw] + kv] = rd
                        xv = [rx.to(cutlass.Float32)]
                        dv = [rd.to(cutlass.Float32)]
                    else:
                        rx = nvvm.load_ext(mXi.iterator + (bs[cw] + kv * VS), dtype=it_ty, count=VS)
                        rd = nvvm.load_ext(mDYi.iterator + (bs[cw] + kv * VS), dtype=it_ty, count=VS)
                        nvvm.store_ext(rx, scx.iterator + (cbs[cw] + kv * VS), shared_space=_CTA_SS)
                        nvvm.store_ext(rd, scd.iterator + (cbs[cw] + kv * VS), shared_space=_CTA_SS)
                        xvv = rx.bitcast(et)
                        dvv = rd.bitcast(et)
                        xv = [xvv[e].to(cutlass.Float32) for e in range(VS)]
                        dv = [dvv[e].to(cutlass.Float32) for e in range(VS)]
                    for e in cutlass.range_constexpr(VS):
                        acc[cw] = acc[cw] + dv[e]
                        accx[cw] = accx[cw] + dv[e] * ((xv[e] - mn[cw]) * rs[cw])
                kv = kv + 32
            n = n + 1
            r = r + 1

    # ---- pass 1b: streamed images ----
    n = nc
    while n < n1:
        bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
        kb = cutlass.Int32(0)
        while kb + STEP <= NV:
            xs = []
            ds = []
            for j in cutlass.range_constexpr(UF):
                for cw in cutlass.range_constexpr(CPW):
                    o = kb + j * 32 + lane
                    if cutlass.const_expr(VS == 1):
                        xs.append(mX[bs[cw] + o].to(cutlass.Float32))
                        ds.append(mDY[bs[cw] + o].to(cutlass.Float32))
                    else:
                        xs.append(nvvm.load_ext(mXi.iterator + (bs[cw] + o * VS), dtype=it_ty, count=VS).bitcast(et))
                        ds.append(nvvm.load_ext(mDYi.iterator + (bs[cw] + o * VS), dtype=it_ty, count=VS).bitcast(et))
            i = 0
            for j in cutlass.range_constexpr(UF):
                for cw in cutlass.range_constexpr(CPW):
                    if cutlass.const_expr(VS == 1):
                        acc[cw] = acc[cw] + ds[i]
                        accx[cw] = accx[cw] + ds[i] * ((xs[i] - mn[cw]) * rs[cw])
                    else:
                        for e in cutlass.range_constexpr(VS):
                            xe = xs[i][e].to(cutlass.Float32)
                            de = ds[i][e].to(cutlass.Float32)
                            acc[cw] = acc[cw] + de
                            accx[cw] = accx[cw] + de * ((xe - mn[cw]) * rs[cw])
                    i = i + 1
            kb = kb + STEP
        kv = kb + lane
        while kv < NV:
            for cw in cutlass.range_constexpr(CPW):
                if cutlass.const_expr(VS == 1):
                    xe = mX[bs[cw] + kv].to(cutlass.Float32)
                    de = mDY[bs[cw] + kv].to(cutlass.Float32)
                    acc[cw] = acc[cw] + de
                    accx[cw] = accx[cw] + de * ((xe - mn[cw]) * rs[cw])
                else:
                    xv = nvvm.load_ext(mXi.iterator + (bs[cw] + kv * VS), dtype=it_ty, count=VS).bitcast(et)
                    dv = nvvm.load_ext(mDYi.iterator + (bs[cw] + kv * VS), dtype=it_ty, count=VS).bitcast(et)
                    for e in cutlass.range_constexpr(VS):
                        xe = xv[e].to(cutlass.Float32)
                        de = dv[e].to(cutlass.Float32)
                        acc[cw] = acc[cw] + de
                        accx[cw] = accx[cw] + de * ((xe - mn[cw]) * rs[cw])
            kv = kv + 32
        n = n + 1

    for cw in cutlass.range_constexpr(CPW):
        a, b = _warp_sum2(acc[cw], accx[cw])
        if lane == 0:
            part[cw * WPB + w] = a
            part[CT + cw * WPB + w] = b
    nvvm.barrier_cta_sync_aligned(0)

    if tid < 2 * CT:
        h = tid // CT
        ct = tid % CT
        mP[cx * (2 * mparts * CT) + h * (mparts * CT) + my * CT + ct] = part[tid]

    nvvm.fence_acq_rel(nvvm.MemScope.GPU)
    nvvm.barrier_cta_sync_aligned(0)
    if tid == 0:
        nvvm.atomicrmw(nvvm.AtomicOp.ADD, mRet.iterator + cx, cutlass.Int32(1), mem_order=nvvm.MemOrder.RELEASE, syncscope=nvvm.MemScope.GPU)
        done = False
        while not done:
            v = nvvm.atomicrmw(nvvm.AtomicOp.ADD, mRet.iterator + cx, cutlass.Int32(0), mem_order=nvvm.MemOrder.ACQUIRE, syncscope=nvvm.MemScope.GPU)
            if v >= mparts:
                done = True
    nvvm.barrier_cta_sync_aligned(0)

    # ---- finalize: tiny (mparts*2*CT floats), so redundancy across CTAs is free ----
    if tid < CT:
        sdy = cutlass.Float32(0.0)
        sdx = cutlass.Float32(0.0)
        p = cutlass.Int32(0)
        while p < mparts:
            sdy = sdy + mP[cx * (2 * mparts * CT) + p * CT + tid]
            sdx = sdx + mP[cx * (2 * mparts * CT) + mparts * CT + p * CT + tid]
            p = p + 1
        stat[tid] = sdy / Mf  # a = dbeta / M
        stat[CT + tid] = sdx / Mf  # b = dgamma / M
        if my == 0:
            mDGamma[c0 + tid] = sdx
            if cutlass.const_expr(has_beta):
                mDBeta[c0 + tid] = sdy
    nvvm.barrier_cta_sync_aligned(0)

    # ---- pass 2: dx = g*rstd*(dy - a - xhat*b); all scalars per row ----
    ga = [cutlass.Float32(0.0)] * CPW
    aa = [cutlass.Float32(0.0)] * CPW
    bb = [cutlass.Float32(0.0)] * CPW
    for cw in cutlass.range_constexpr(CPW):
        ci = cw * WPB + w
        ga[cw] = mG[c0 + ci].to(cutlass.Float32) * rs[cw]
        aa[cw] = stat[ci]
        bb[cw] = stat[CT + ci]

    if cutlass.const_expr(KSR > 0):
        n = n0
        r = cutlass.Int32(0)
        while n < nc:
            bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
            cbs = [((w * CPW + cw) * KSR + r) * S for cw in range(CPW)]
            kv = lane
            while kv < NV:
                for cw in cutlass.range_constexpr(CPW):
                    if cutlass.const_expr(VS == 1):
                        xe = scx[cbs[cw] + kv].to(cutlass.Float32)
                        de = scd[cbs[cw] + kv].to(cutlass.Float32)
                        mDX[bs[cw] + kv] = (ga[cw] * (de - aa[cw] - ((xe - mn[cw]) * rs[cw]) * bb[cw])).to(et)
                    else:
                        xv = nvvm.load_ext(scx.iterator + (cbs[cw] + kv * VS), dtype=it_ty, count=VS, shared_space=_CTA_SS).bitcast(et)
                        dv = nvvm.load_ext(scd.iterator + (cbs[cw] + kv * VS), dtype=it_ty, count=VS, shared_space=_CTA_SS).bitcast(et)
                        ys = []
                        for e in cutlass.range_constexpr(VS):
                            xe = xv[e].to(cutlass.Float32)
                            de = dv[e].to(cutlass.Float32)
                            ys.append((ga[cw] * (de - aa[cw] - ((xe - mn[cw]) * rs[cw]) * bb[cw])).to(et))
                        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (bs[cw] + kv * VS))
                kv = kv + 32
            n = n + 1
            r = r + 1

    n = nc
    while n < n1:
        bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
        kb = cutlass.Int32(0)
        while kb + STEP <= NV:
            xs = []
            ds = []
            for j in cutlass.range_constexpr(UF):
                for cw in cutlass.range_constexpr(CPW):
                    o = kb + j * 32 + lane
                    if cutlass.const_expr(VS == 1):
                        xs.append(mX[bs[cw] + o].to(cutlass.Float32))
                        ds.append(mDY[bs[cw] + o].to(cutlass.Float32))
                    else:
                        xs.append(nvvm.load_ext(mXi.iterator + (bs[cw] + o * VS), dtype=it_ty, count=VS).bitcast(et))
                        ds.append(nvvm.load_ext(mDYi.iterator + (bs[cw] + o * VS), dtype=it_ty, count=VS).bitcast(et))
            i = 0
            for j in cutlass.range_constexpr(UF):
                for cw in cutlass.range_constexpr(CPW):
                    o = kb + j * 32 + lane
                    if cutlass.const_expr(VS == 1):
                        mDX[bs[cw] + o] = (ga[cw] * (ds[i] - aa[cw] - ((xs[i] - mn[cw]) * rs[cw]) * bb[cw])).to(et)
                    else:
                        ys = []
                        for e in cutlass.range_constexpr(VS):
                            xe = xs[i][e].to(cutlass.Float32)
                            de = ds[i][e].to(cutlass.Float32)
                            ys.append((ga[cw] * (de - aa[cw] - ((xe - mn[cw]) * rs[cw]) * bb[cw])).to(et))
                        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (bs[cw] + o * VS))
                    i = i + 1
            kb = kb + STEP
        kv = kb + lane
        while kv < NV:
            for cw in cutlass.range_constexpr(CPW):
                if cutlass.const_expr(VS == 1):
                    xe = mX[bs[cw] + kv].to(cutlass.Float32)
                    de = mDY[bs[cw] + kv].to(cutlass.Float32)
                    mDX[bs[cw] + kv] = (ga[cw] * (de - aa[cw] - ((xe - mn[cw]) * rs[cw]) * bb[cw])).to(et)
                else:
                    xv = nvvm.load_ext(mXi.iterator + (bs[cw] + kv * VS), dtype=it_ty, count=VS).bitcast(et)
                    dv = nvvm.load_ext(mDYi.iterator + (bs[cw] + kv * VS), dtype=it_ty, count=VS).bitcast(et)
                    ys = []
                    for e in cutlass.range_constexpr(VS):
                        xe = xv[e].to(cutlass.Float32)
                        de = dv[e].to(cutlass.Float32)
                        ys.append((ga[cw] * (de - aa[cw] - ((xe - mn[cw]) * rs[cw]) * bb[cw])).to(et))
                    nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (bs[cw] + kv * VS))
            kv = kv + 32
        n = n + 1


_bn_bwd_nchw_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _bn_bwd_nchw_host(
    mDY,
    mX,
    mDX,
    mG,
    mMean,
    mRstd,
    mP,
    mRet,
    mDGamma,
    mDBeta,
    N,
    mparts,
    C: cutlass.Constexpr,
    S: cutlass.Constexpr,
    VS: cutlass.Constexpr,
    UF: cutlass.Constexpr,
    WPB: cutlass.Constexpr,
    CPW: cutlass.Constexpr,
    CT: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    KSR: cutlass.Constexpr,
    cparts: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
    mbpm: cutlass.Constexpr,
) -> None:
    mDYi = cute.recast_tensor(mDY, it_ty)
    mXi = cute.recast_tensor(mX, it_ty)
    mDXi = cute.recast_tensor(mDX, it_ty)
    _bn_bwd_nchw_kernel(
        mDY,
        mDYi,
        mX,
        mXi,
        mDX,
        mDXi,
        mG,
        mMean,
        mRstd,
        mP,
        mRet,
        mDGamma,
        mDBeta,
        N,
        mparts,
        C,
        S,
        VS,
        UF,
        WPB,
        CPW,
        CT,
        BT,
        KSR,
        it_ty,
        et,
        Mf,
        has_beta,
    ).launch(grid=(cparts, mparts, 1), block=(BT, 1, 1), smem=smem_bytes, cooperative=True, min_blocks_per_mp=mbpm)


_KCACHE = {}
_OCC = {}


def _ksr(nloc, WPB, CPW, S, eb, occ):
    """Image rows per warp cached in shared. Backward stashes BOTH dy and x, so a
    row costs twice the forward's."""
    per_row = 2 * WPB * CPW * S * eb
    if per_row == 0:
        return 0
    return max(0, min(nloc, (smem_budget(_SMEM_CAP_PREF) // occ) // per_row))


def backward(spec, dy3d, x3d, gamma, saved_mean, saved_rstd, *, has_beta, cfg, params, knobs=None):
    """Launch the NCHW BatchNorm backward. Returns ``(dx, dgamma, dbeta)``."""
    import torch

    N, C, S = int(spec.N), int(spec.C), int(spec.S)
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    geo = nchw_cfg(C, N, S, eb)
    if geo is None:
        raise ValueError(f"NCHW BatchNorm backward: C={C} S={S} unsupported")
    VS, UF, WPB, CPW, CT, cparts, BT = geo

    dx = torch.empty_like(x3d)
    dgamma = torch.empty(C, dtype=torch.float32, device=x3d.device)
    dbeta = torch.empty(C, dtype=torch.float32, device=x3d.device)
    xf, df, gf = x3d.reshape(-1), dy3d.reshape(-1), dx.reshape(-1)
    count = float(N * S)

    okey = (params.io_dtype, C, S, N, has_beta)
    occ0 = _OCC.get(okey, 2)
    for occ in ([2, 1] if occ0 == 2 else [1]):
        mparts = max(1, min(occ * _nsm() // max(1, cparts), N))
        nloc = (N + mparts - 1) // mparts
        KSR = _ksr(nloc, WPB, CPW, S, eb, occ) if knobs is None else int(knobs)
        smem_bytes = 2 * (2 * CT) * 4 + 2 * WPB * CPW * KSR * S * eb + 128
        mbpm = 2 if occ == 2 else 0
        ce = (C, S, VS, UF, WPB, CPW, CT, BT, KSR, cparts, it_ty, et, count, has_beta, smem_bytes, mbpm)
        key = (params.io_dtype, C, S, VS, UF, CPW, KSR, has_beta, mbpm)
        pbuf = torch.empty(cparts * 2 * mparts * CT, dtype=torch.float32, device=x3d.device)
        ret = torch.zeros(cparts, dtype=torch.int32, device=x3d.device)
        args = (
            dyn(df),
            dyn(xf),
            dyn(gf),
            dyn(gamma),
            dyn(saved_mean),
            dyn(saved_rstd),
            dyn(pbuf),
            dyn(ret),
            dyn(dgamma),
            dyn(dbeta),
            cutlass.Int32(N),
            cutlass.Int32(mparts),
        )
        fn = _KCACHE.get(key)
        if fn is None:
            fn = cute.compile(_bn_bwd_nchw_host, *args, *ce)
            _KCACHE[key] = fn
        try:
            fn(*args)
            _OCC[okey] = occ
            return dx, dgamma, (dbeta if has_beta else None)
        except Exception as e:
            if "TOO_LARGE" in str(e) and occ == 2:
                continue
            raise
    return dx, dgamma, (dbeta if has_beta else None)
