# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InstanceNorm backward for NHWC (channels-last), sm_100, CUTLASS primitives.

In NHWC an InstanceNorm group -- one ``(sample, channel)`` pair -- is STRIDED by C,
so the rowwise view the NCHW path uses does not exist and ``.contiguous()`` would
transpose. But image ``n`` on its own is a contiguous ``[H*W, C]`` block, which is
exactly the BatchNorm-NHWC problem with ``M = H*W``. So this is the BN NHWC backward
with an extra grid dimension over images and the statistics kept per ``(n, c)``.

One reduction per ``(n, c)`` yields everything. With ``dxhat = dy*gamma_c``::

    S1 = sum_hw(dy)                 S2 = sum_hw(dy * xhat)
    a  = gamma_c * S1 / M           b  = gamma_c * S2 / M
    dx = rstd * (dy*gamma_c - a - xhat*b)
    dgamma_c = sum_n S2[n, c]       dbeta_c = sum_n S1[n, c]

``dx`` therefore needs only per-image stats, while ``dgamma``/``dbeta`` need a sum
ACROSS images -- so each image's leader CTA parks its ``S1``/``S2`` in an ``[N, C]``
buffer and a tiny finalize reduces the ``N`` contributors. No atomics.

Grid is ``(cblks, mparts, N)``: ``mparts`` CTAs split one image's ``H*W`` and
synchronise on a retired-CTA counter indexed by ``(n, channel_tile)``, so images
never wait on each other.
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
_CPC_MAX = 128
_BT = 256
_SM_COUNT = None


def _nsm():
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


def nhwc_cfg(C, eb, block_threads=_BT):
    """``(V, CPC, TPP, PPL, cblks, BT)`` or None when C is not 128-bit tileable."""
    V = 16 // eb
    if C % V != 0:
        return None
    cpc = 0
    for cand in range(min(C, _CPC_MAX), V - 1, -V):
        if C % cand == 0:
            cpc = cand
            break
    if cpc == 0:
        return None
    tpp = cpc // V
    if tpp > block_threads or block_threads % tpp != 0:
        return None
    return V, cpc, tpp, block_threads // tpp, C // cpc, block_threads


@cute.kernel
def _in_bwd_nhwc_kernel(
    mDYi,
    mXi,
    mDXi,
    mG,
    mMean,
    mRstd,
    mP,
    mRet,
    mS1,
    mS2,
    HW: cutlass.Int32,
    mparts: cutlass.Int32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    KS: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    cblks: cutlass.Constexpr,
    COOP: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cx, my, nz = cute.arch.block_idx()
    NSEG: cutlass.Constexpr = V // 4

    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * BT * V), byte_alignment=16)
    stat = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * CPC), byte_alignment=16)
    scx = None
    scd = None
    if cutlass.const_expr(KS > 0):
        scx = smem.allocate_tensor(it_ty, cute.make_layout(BT * KS * V), byte_alignment=16)
        scd = smem.allocate_tensor(it_ty, cute.make_layout(BT * KS * V), byte_alignment=16)

    lane = tid % TPP
    tp = tid // TPP
    c0 = cx * (TPP * V) + lane * V
    img = cutlass.Int64(nz) * HW  # first pixel of this image
    per = (HW + mparts - 1) // mparts
    r0 = my * per
    r1 = r0 + per
    if r1 > HW:
        r1 = HW

    mn = []
    rsd = []
    for e in cutlass.range_constexpr(V):
        mn.append(mMean[nz * C + c0 + e])
        rsd.append(mRstd[nz * C + c0 + e])

    s = [cutlass.Float32(0.0)] * V
    sq = [cutlass.Float32(0.0)] * V
    xc = [cutlass.Float32(0.0)] * (KC * V)
    dc = [cutlass.Float32(0.0)] * (KC * V)

    for kk in cutlass.range_constexpr(KC):
        rk = r0 + tp + kk * PPL
        if rk < r1:
            xv = nvvm.load_ext(mXi.iterator + ((img + rk) * C + c0), dtype=it_ty, count=V).bitcast(et)
            dv = nvvm.load_ext(mDYi.iterator + ((img + rk) * C + c0), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                d = dv[e].to(cutlass.Float32)
                s[e] = s[e] + d
                sq[e] = sq[e] + d * ((x - mn[e]) * rsd[e])
                xc[kk * V + e] = x
                dc[kk * V + e] = d
    for ks in cutlass.range_constexpr(KS):
        rk = r0 + tp + (KC + ks) * PPL
        if rk < r1:
            rx = nvvm.load_ext(mXi.iterator + ((img + rk) * C + c0), dtype=it_ty, count=V)
            rd = nvvm.load_ext(mDYi.iterator + ((img + rk) * C + c0), dtype=it_ty, count=V)
            xv = rx.bitcast(et)
            dv = rd.bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                d = dv[e].to(cutlass.Float32)
                s[e] = s[e] + d
                sq[e] = sq[e] + d * ((x - mn[e]) * rsd[e])
            nvvm.store_ext(rx, scx.iterator + (tid * KS + ks) * V, shared_space=_CTA_SS)
            nvvm.store_ext(rd, scd.iterator + (tid * KS + ks) * V, shared_space=_CTA_SS)
    row = r0 + tp + (KC + KS) * PPL
    while row < r1:
        xv = nvvm.load_ext(mXi.iterator + ((img + row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        dv = nvvm.load_ext(mDYi.iterator + ((img + row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            d = dv[e].to(cutlass.Float32)
            s[e] = s[e] + d
            sq[e] = sq[e] + d * ((x - mn[e]) * rsd[e])
        row = row + PPL

    def _reduce2(va, vb):
        if cutlass.const_expr(PPL > 1):
            for e in cutlass.range_constexpr(V):
                red[(tp * TPP + lane) * V + e] = va[e]
                red[BT * V + (tp * TPP + lane) * V + e] = vb[e]
            nvvm.barrier_cta_sync_aligned(0)
            if tp == 0:
                for e in cutlass.range_constexpr(V):
                    aa_ = va[e]
                    bb_ = vb[e]
                    for j in cutlass.range_constexpr(PPL - 1):
                        aa_ = aa_ + red[((j + 1) * TPP + lane) * V + e]
                        bb_ = bb_ + red[BT * V + ((j + 1) * TPP + lane) * V + e]
                    va[e] = aa_
                    vb[e] = bb_
            nvvm.barrier_cta_sync_aligned(0)
        return va, vb

    s, sq = _reduce2(s, sq)

    # When mparts == 1 one CTA owns the whole image for its channel tile, so the
    # PPL reduce above is already the complete per-(n,c) sum: no partials, no grid
    # barrier, and no cooperative launch (which would cap the grid at what can
    # co-reside -- cblks*N alone is 512 CTAs at C=2048).
    facc = s
    faccsq = sq
    if cutlass.const_expr(COOP):
        pbase = (cutlass.Int32(nz) * cblks + cx) * (mparts * TPP * V * 2)
        psq = pbase + mparts * TPP * V
        if tp == 0:
            off = (my * TPP + lane) * V
            for h in cutlass.range_constexpr(NSEG):
                sseg = cutlass.Vector.from_elements(tuple(s[h * 4 + j] for j in range(4)), cutlass.Float32)
                qseg = cutlass.Vector.from_elements(tuple(sq[h * 4 + j] for j in range(4)), cutlass.Float32)
                nvvm.store_ext(sseg, mP.iterator + (pbase + off + h * 4))
                nvvm.store_ext(qseg, mP.iterator + (psq + off + h * 4))

        nvvm.fence_acq_rel(nvvm.MemScope.GPU)
        nvvm.barrier_cta_sync_aligned(0)
        rslot = cutlass.Int32(nz) * cblks + cx
        if tid == 0:
            nvvm.atomicrmw(nvvm.AtomicOp.ADD, mRet.iterator + rslot, cutlass.Int32(1), mem_order=nvvm.MemOrder.RELEASE, syncscope=nvvm.MemScope.GPU)
            done = False
            while not done:
                v = nvvm.atomicrmw(nvvm.AtomicOp.ADD, mRet.iterator + rslot, cutlass.Int32(0), mem_order=nvvm.MemOrder.ACQUIRE, syncscope=nvvm.MemScope.GPU)
                if v >= mparts:
                    done = True
        nvvm.barrier_cta_sync_aligned(0)

        facc = [cutlass.Float32(0.0)] * V
        faccsq = [cutlass.Float32(0.0)] * V
        part = tp
        while part < mparts:
            off = (part * TPP + lane) * V
            for h in cutlass.range_constexpr(NSEG):
                sv = nvvm.load_ext(mP.iterator + (pbase + off + h * 4), dtype=cutlass.Float32, count=4)
                qv = nvvm.load_ext(mP.iterator + (psq + off + h * 4), dtype=cutlass.Float32, count=4)
                for j in cutlass.range_constexpr(4):
                    facc[h * 4 + j] = facc[h * 4 + j] + sv[j]
                    faccsq[h * 4 + j] = faccsq[h * 4 + j] + qv[j]
            part = part + PPL
        facc, faccsq = _reduce2(facc, faccsq)

    if tp == 0:
        for e in cutlass.range_constexpr(V):
            stat[lane * V + e] = facc[e]
            stat[CPC + lane * V + e] = faccsq[e]
            if my == 0:
                # per-image S1/S2 for the cross-image dgamma/dbeta finalize
                mS1[nz * C + c0 + e] = facc[e]
                mS2[nz * C + c0 + e] = faccsq[e]
    nvvm.barrier_cta_sync_aligned(0)

    aa = []
    bb = []
    gc = []
    for e in cutlass.range_constexpr(V):
        g_ = mG[c0 + e].to(cutlass.Float32)
        gc.append(g_)
        aa.append(g_ * stat[lane * V + e] / Mf)
        bb.append(g_ * stat[CPC + lane * V + e] / Mf)

    for kk in cutlass.range_constexpr(KC):
        rk = r0 + tp + kk * PPL
        if rk < r1:
            ys = []
            for e in cutlass.range_constexpr(V):
                xh = (xc[kk * V + e] - mn[e]) * rsd[e]
                ys.append((rsd[e] * (dc[kk * V + e] * gc[e] - aa[e] - xh * bb[e])).to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + ((img + rk) * C + c0))
    for ks in cutlass.range_constexpr(KS):
        rk = r0 + tp + (KC + ks) * PPL
        if rk < r1:
            xv = nvvm.load_ext(scx.iterator + (tid * KS + ks) * V, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
            dv = nvvm.load_ext(scd.iterator + (tid * KS + ks) * V, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
            ys = []
            for e in cutlass.range_constexpr(V):
                xh = (xv[e].to(cutlass.Float32) - mn[e]) * rsd[e]
                ys.append((rsd[e] * (dv[e].to(cutlass.Float32) * gc[e] - aa[e] - xh * bb[e])).to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + ((img + rk) * C + c0))
    row = r0 + tp + (KC + KS) * PPL
    while row < r1:
        xv = nvvm.load_ext(mXi.iterator + ((img + row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        dv = nvvm.load_ext(mDYi.iterator + ((img + row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            xh = (xv[e].to(cutlass.Float32) - mn[e]) * rsd[e]
            ys.append((rsd[e] * (dv[e].to(cutlass.Float32) * gc[e] - aa[e] - xh * bb[e])).to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + ((img + row) * C + c0))
        row = row + PPL


@cute.kernel
def _in_bwd_nhwc_finalize(mS1, mS2, mDGamma, mDBeta, N: cutlass.Int32, C: cutlass.Constexpr, bt: cutlass.Constexpr, has_beta: cutlass.Constexpr) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    c = bid * bt + tid
    if c < C:
        sg = cutlass.Float32(0.0)
        sb = cutlass.Float32(0.0)
        n = cutlass.Int32(0)
        while n < N:
            sb = sb + mS1[n * C + c]
            sg = sg + mS2[n * C + c]
            n = n + 1
        mDGamma[c] = sg
        if cutlass.const_expr(has_beta):
            mDBeta[c] = sb


@cute.jit
def _in_bwd_nhwc_host(
    mDY,
    mX,
    mDX,
    mG,
    mMean,
    mRstd,
    mP,
    mRet,
    mS1,
    mS2,
    mDGamma,
    mDBeta,
    HW,
    mparts,
    N,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    KS: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    cblks: cutlass.Constexpr,
    COOP: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
    gridz: cutlass.Constexpr,
    fgrid: cutlass.Constexpr,
    fbt: cutlass.Constexpr,
) -> None:
    mDYi = cute.recast_tensor(mDY, it_ty)
    mXi = cute.recast_tensor(mX, it_ty)
    mDXi = cute.recast_tensor(mDX, it_ty)
    _in_bwd_nhwc_kernel(
        mDYi,
        mXi,
        mDXi,
        mG,
        mMean,
        mRstd,
        mP,
        mRet,
        mS1,
        mS2,
        HW,
        mparts,
        C,
        V,
        TPP,
        PPL,
        BT,
        KC,
        KS,
        CPC,
        cblks,
        COOP,
        it_ty,
        et,
        Mf,
    ).launch(grid=(cblks, mparts, gridz), block=(BT, 1, 1), smem=smem_bytes, cooperative=COOP)
    _in_bwd_nhwc_finalize(mS1, mS2, mDGamma, mDBeta, N, C, fbt, has_beta).launch(grid=(fgrid, 1, 1), block=(fbt, 1, 1))


_KCACHE = {}


def backward(spec, dy4d, x4d, gamma, mean, rstd, *, has_beta, cfg, params, knobs=None):
    """Launch the native NHWC InstanceNorm backward. ``x4d``/``dy4d`` are the
    channels-last tensors viewed as ``[N, H*W, C]``."""
    import torch

    N, HW, C = (int(v) for v in x4d.shape)
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    geo = nhwc_cfg(C, eb)
    if geo is None:
        raise ValueError(f"NHWC InstanceNorm backward: C={C} not tileable")
    V, CPC, TPP, PPL, cblks, BT = geo

    KC, KS = knobs if knobs is not None else (4, 0)
    smem_bytes = 2 * BT * V * 4 + 2 * CPC * 4 + 2 * BT * KS * V * eb + 128
    # cooperative: the whole grid must co-reside
    mparts = max(1, min(_nsm() // max(1, cblks * N), (HW + PPL - 1) // PPL))

    dx = torch.empty_like(x4d)
    dgamma = torch.empty(C, dtype=torch.float32, device=x4d.device)
    dbeta = torch.empty(C, dtype=torch.float32, device=x4d.device)
    pbuf = torch.empty(N * cblks * mparts * TPP * V * 2, dtype=torch.float32, device=x4d.device)
    ret = torch.zeros(N * cblks, dtype=torch.int32, device=x4d.device)
    s1 = torch.empty(N * C, dtype=torch.float32, device=x4d.device)
    s2 = torch.empty(N * C, dtype=torch.float32, device=x4d.device)
    fbt = 128
    fgrid = (C + fbt - 1) // fbt

    args = (
        dyn(dy4d.reshape(-1)),
        dyn(x4d.reshape(-1)),
        dyn(dx.reshape(-1)),
        dyn(gamma),
        dyn(mean),
        dyn(rstd),
        dyn(pbuf),
        dyn(ret),
        dyn(s1),
        dyn(s2),
        dyn(dgamma),
        dyn(dbeta),
        cutlass.Int32(HW),
        cutlass.Int32(mparts),
        cutlass.Int32(N),
    )
    ce = (C, V, TPP, PPL, BT, KC, KS, CPC, cblks, mparts > 1, it_ty, et, float(HW), has_beta, smem_bytes, N, fgrid, fbt)
    key = (params.io_dtype, C, HW, N, KC, KS, CPC, has_beta, mparts, mparts > 1)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_in_bwd_nhwc_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return dx, dgamma, (dbeta if has_beta else None)
