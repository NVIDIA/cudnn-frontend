# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InstanceNorm forward for NHWC (channels-last), sm_100, CUTLASS primitives.

An InstanceNorm group is one ``(sample, channel)`` pair, which in NHWC is strided
by C -- but image ``n`` on its own is a contiguous ``[H*W, C]`` block, so this is
the BatchNorm-NHWC map with the statistics kept per ``(n, c)`` instead of per ``c``.

Simpler than the backward: ``mean``/``rstd`` are per group, so there is NO
cross-image reduction at all. Each ``(image, channel tile)`` reduces its own pixels,
and when one CTA owns the whole image for its tile the reduction is entirely in-CTA
-- no partials, no grid barrier, no cooperative launch.

``KC`` pixels per thread are cached in registers on the pass-1 read, so a slice that
fits is read exactly once (1R + 1W).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.utils import dyn, run_coop, sm_count

_CTA_SS = nvvm.SharedSpace.shared_cta
_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_CPC_MAX = 128
_BT = 256


def nhwc_cfg(C, eb, block_threads=_BT):
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
def _in_fwd_nhwc_kernel(
    mXi,
    mYi,
    mG,
    mB,
    mMean,
    mRstd,
    mP,
    mRet,
    HW: cutlass.Int32,
    mparts: cutlass.Int32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    cblks: cutlass.Constexpr,
    COOP: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cx, my, nz = cute.arch.block_idx()
    NSEG: cutlass.Constexpr = V // 4

    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * BT * V), byte_alignment=16)
    stat = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * CPC), byte_alignment=16)

    lane = tid % TPP
    tp = tid // TPP
    c0 = cx * (TPP * V) + lane * V
    img = cutlass.Int64(nz) * HW
    per = (HW + mparts - 1) // mparts
    r0 = my * per
    r1 = r0 + per
    if r1 > HW:
        r1 = HW

    s = [cutlass.Float32(0.0)] * V
    sq = [cutlass.Float32(0.0)] * V
    xc = [cutlass.Float32(0.0)] * (KC * V)

    for kk in cutlass.range_constexpr(KC):
        p = r0 + tp + kk * PPL
        if p < r1:
            xv = nvvm.load_ext(mXi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                s[e] = s[e] + x
                sq[e] = sq[e] + x * x
                xc[kk * V + e] = x
    p = r0 + tp + KC * PPL
    while p < r1:
        xv = nvvm.load_ext(mXi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            s[e] = s[e] + x
            sq[e] = sq[e] + x * x
        p = p + PPL

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
        s, sq = _reduce2(facc, faccsq)

    if tp == 0:
        for e in cutlass.range_constexpr(V):
            mnv = s[e] / Mf
            rsv = cute.math.rsqrt(sq[e] / Mf - mnv * mnv + eps)
            stat[lane * V + e] = mnv
            stat[CPC + lane * V + e] = rsv
            if my == 0:
                mMean[cutlass.Int32(nz) * C + c0 + e] = mnv
                mRstd[cutlass.Int32(nz) * C + c0 + e] = rsv
    nvvm.barrier_cta_sync_aligned(0)

    sc = []
    sh = []
    for e in cutlass.range_constexpr(V):
        f = stat[CPC + lane * V + e] * mG[c0 + e].to(cutlass.Float32)
        sc.append(f)
        o = -stat[lane * V + e] * f
        if cutlass.const_expr(has_beta):
            o = o + mB[c0 + e].to(cutlass.Float32)
        sh.append(o)

    for kk in cutlass.range_constexpr(KC):
        p = r0 + tp + kk * PPL
        if p < r1:
            ys = []
            for e in cutlass.range_constexpr(V):
                ys.append((xc[kk * V + e] * sc[e] + sh[e]).to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + ((img + p) * C + c0))
    p = r0 + tp + KC * PPL
    while p < r1:
        xv = nvvm.load_ext(mXi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            ys.append((xv[e].to(cutlass.Float32) * sc[e] + sh[e]).to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + ((img + p) * C + c0))
        p = p + PPL


_in_fwd_nhwc_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _in_fwd_nhwc_host(
    mX,
    mY,
    mG,
    mB,
    mMean,
    mRstd,
    mP,
    mRet,
    HW,
    mparts,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    cblks: cutlass.Constexpr,
    COOP: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
    gridm: cutlass.Constexpr,
    gridn: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _in_fwd_nhwc_kernel(
        mXi,
        mYi,
        mG,
        mB,
        mMean,
        mRstd,
        mP,
        mRet,
        HW,
        mparts,
        C,
        V,
        TPP,
        PPL,
        BT,
        KC,
        CPC,
        cblks,
        COOP,
        it_ty,
        et,
        Mf,
        eps,
        has_beta,
    ).launch(grid=(cblks, gridm, gridn), block=(BT, 1, 1), smem=smem_bytes, cooperative=COOP)


_KCACHE = {}


def forward(spec, x3, gamma, beta, *, eps, cfg, params, knobs=None):
    """Launch the native NHWC InstanceNorm forward. ``x3`` is ``[N, H*W, C]``."""
    import torch

    N, HW, C = (int(v) for v in x3.shape)
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    geo = nhwc_cfg(C, eb)
    if geo is None:
        raise ValueError(f"NHWC InstanceNorm forward: C={C} not tileable")
    V, CPC, TPP, PPL, cblks, BT = geo
    has_beta = beta is not None
    if beta is None:
        beta = gamma
    KC = 4 if knobs is None else int(knobs)

    smem_bytes = 2 * BT * V * 4 + 2 * CPC * 4 + 128

    def _run(occ):
        # Split H*W only as far as needed to fill the machine `occ` CTAs deep, and
        # never past the point where a part stops filling the register cache -- a
        # thinner slice buys no bandwidth and still pays the whole grid barrier.
        mparts = max(1, min(sm_count() * occ // max(1, cblks * N), HW // (PPL * KC)))
        COOP = mparts > 1
        y = torch.empty_like(x3)
        mean = torch.empty(N * C, dtype=torch.float32, device=x3.device)
        rstd = torch.empty(N * C, dtype=torch.float32, device=x3.device)
        pbuf = torch.empty(N * cblks * mparts * TPP * V * 2, dtype=torch.float32, device=x3.device)
        ret = torch.zeros(N * cblks, dtype=torch.int32, device=x3.device)
        args = (
            dyn(x3.reshape(-1)),
            dyn(y.reshape(-1)),
            dyn(gamma),
            dyn(beta),
            dyn(mean),
            dyn(rstd),
            dyn(pbuf),
            dyn(ret),
            cutlass.Int32(HW),
            cutlass.Int32(mparts),
        )
        ce = (C, V, TPP, PPL, BT, KC, CPC, cblks, COOP, it_ty, et, float(HW), float(eps), has_beta, smem_bytes, mparts, N)
        key = (params.io_dtype, C, HW, N, KC, CPC, has_beta, mparts, COOP)
        fn = _KCACHE.get(key)
        if fn is None:
            fn = cute.compile(_in_fwd_nhwc_host, *args, *ce)
            _KCACHE[key] = fn
        fn(*args)
        return y, mean, rstd

    return run_coop(("in_fwd", params.io_dtype, C, HW, N, KC, has_beta), _run)
