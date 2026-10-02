# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GroupNorm backward for NHWC (channels-last), sm_100, CUTLASS primitives.

A GroupNorm group is ``cpg`` ADJACENT channels over all of H*W. In NHWC those
channels are contiguous *within a pixel*, so group ``g`` of image ``n`` is the set
``{ (n*HW + p)*C + g*cpg + k }`` -- strided by C across pixels, contiguous in k.
That gives a clean tile: let the channel tile BE the group.

The reduction differs from BatchNorm/InstanceNorm in one important way: a group
spans ``cpg`` channels, so the group statistics reduce across the channel tile as
well as across pixels -- a FULL block reduce, not a per-lane one. The parameter
gradients stay per-CHANNEL, so they remain per-lane accumulators. With
``dxhat = dy*gamma_c`` and ``M = HW*cpg``::

    a = sum_{hw,k}(dxhat) / M        b = sum_{hw,k}(dxhat*xhat) / M   (per (n,g))
    dx = rstd * (dxhat - a - xhat*b)
    dgamma_c = sum_{n,hw}(dy*xhat)   dbeta_c = sum_{n,hw}(dy)         (per channel)

**Coalescing drives the tile size, not the group.** Setting the channel tile equal
to the group gives ``TPP = cpg/V``, which at ``cpg=2`` is ONE thread per pixel -- so
consecutive threads land on consecutive pixels, ``C*eb`` bytes apart, and each reads
4 bytes per 128-byte stride. Measured 0.07 of achievable. Instead the tile is a fixed
``CPC`` channels wide so ``TPP = CPC/V`` threads span it contiguously, and the tile
holds ``CPC/cpg`` whole groups. A lane's ``V`` channels then sit inside ONE group
(``V <= cpg``), and ``cpg/V`` consecutive lanes share a group -- so the group
reduction is a SEGMENTED reduce over those lanes plus all ``PPL`` pixel groups.

One CTA owns one ``(image, channel tile)``. The per-channel gradients go to an
``[N, C]`` buffer that a tiny finalize reduces over images, exactly as the NHWC
InstanceNorm backward does.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.utils import dyn

_SM_COUNT = None


def _nsm():
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


_CTA_SS = nvvm.SharedSpace.shared_cta
_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_FULL = 0xFFFFFFFF
_BFLY_CLAMP = 0x1F
_BT = 256


@cute.jit
def _block_sum2(v1, v2, tid, red, bt: cutlass.Constexpr):
    """Reduce two fp32 partials across the CTA; nvvm only."""
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
def _gn_bwd_nhwc_kernel(
    mDYi,
    mXi,
    mDXi,
    mG,
    mMean,
    mRstd,
    mS1,
    mS2,
    mP,
    mRet,
    HW: cutlass.Int32,
    mparts: cutlass.Int32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    CPG: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    G: cutlass.Constexpr,
    GPT: cutlass.Constexpr,
    LPG: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    TILES: cutlass.Constexpr,
    COOP: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    gx, my, nz = cute.arch.block_idx()  # gx = channel TILE, my = H*W split

    smem = SmemAllocator()
    seg = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * BT), byte_alignment=16)
    gst = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * GPT), byte_alignment=16)
    acc = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * BT * V), byte_alignment=16)

    lane = tid % TPP  # which V-wide slice of the channel TILE
    tp = tid // TPP  # which of the PPL parallel pixels
    c0 = gx * CPC + lane * V  # absolute channel of this thread's first element
    gi = (lane * V) // CPG  # which group within the tile this lane belongs to
    grp = gx * GPT + gi  # absolute group index
    img = cutlass.Int64(nz) * HW
    per = (HW + mparts - 1) // mparts
    p0 = my * per
    p1 = p0 + per
    if p1 > HW:
        p1 = HW
    mean = mMean[cutlass.Int32(nz) * G + grp]
    rstd = mRstd[cutlass.Int32(nz) * G + grp]

    gc = []
    for e in cutlass.range_constexpr(V):
        gc.append(mG[c0 + e].to(cutlass.Float32))

    s1 = cutlass.Float32(0.0)  # sum(dxhat)      -- group-wide
    s2 = cutlass.Float32(0.0)  # sum(dxhat*xhat) -- group-wide
    dg = [cutlass.Float32(0.0)] * V  # per-channel
    db = [cutlass.Float32(0.0)] * V
    xc = [cutlass.Float32(0.0)] * (KC * V)
    dc = [cutlass.Float32(0.0)] * (KC * V)

    # ---- pass 1 ----
    for kk in cutlass.range_constexpr(KC):
        p = p0 + tp + kk * PPL
        if p < p1:
            xv = nvvm.load_ext(mXi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
            dv = nvvm.load_ext(mDYi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                d = dv[e].to(cutlass.Float32)
                xh = (x - mean) * rstd
                dxh = d * gc[e]
                s1 = s1 + dxh
                s2 = s2 + dxh * xh
                dg[e] = dg[e] + d * xh
                db[e] = db[e] + d
                xc[kk * V + e] = x
                dc[kk * V + e] = d
    p = p0 + tp + KC * PPL
    while p < p1:
        xv = nvvm.load_ext(mXi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
        dv = nvvm.load_ext(mDYi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            d = dv[e].to(cutlass.Float32)
            xh = (x - mean) * rstd
            dxh = d * gc[e]
            s1 = s1 + dxh
            s2 = s2 + dxh * xh
            dg[e] = dg[e] + d * xh
            db[e] = db[e] + d
        p = p + PPL

    # group statistics: SEGMENTED reduce -- a group owns LPG consecutive lanes across
    # all PPL pixel groups, so this is not a whole-block sum.
    seg[tid] = s1
    seg[BT + tid] = s2
    nvvm.barrier_cta_sync_aligned(0)
    if tid < GPT:
        t1 = cutlass.Float32(0.0)
        t2 = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(PPL):
            for l in cutlass.range_constexpr(LPG):
                idx = j * TPP + tid * LPG + l
                t1 = t1 + seg[idx]
                t2 = t2 + seg[BT + idx]
        gst[tid] = t1
        gst[GPT + tid] = t2
    nvvm.barrier_cta_sync_aligned(0)
    if cutlass.const_expr(COOP):
        # mparts CTAs share this (image, tile): park partials, spin, re-reduce.
        pb = ((cutlass.Int32(nz) * TILES + gx) * mparts + my) * (2 * GPT)
        if tid < GPT:
            mP[pb + tid] = gst[tid]
            mP[pb + GPT + tid] = gst[GPT + tid]
        nvvm.fence_acq_rel(nvvm.MemScope.GPU)
        nvvm.barrier_cta_sync_aligned(0)
        rslot = cutlass.Int32(nz) * TILES + gx
        if tid == 0:
            nvvm.atomicrmw(nvvm.AtomicOp.ADD, mRet.iterator + rslot, cutlass.Int32(1), mem_order=nvvm.MemOrder.RELEASE, syncscope=nvvm.MemScope.GPU)
            done = False
            while not done:
                v = nvvm.atomicrmw(nvvm.AtomicOp.ADD, mRet.iterator + rslot, cutlass.Int32(0), mem_order=nvvm.MemOrder.ACQUIRE, syncscope=nvvm.MemScope.GPU)
                if v >= mparts:
                    done = True
        nvvm.barrier_cta_sync_aligned(0)
        if tid < GPT:
            t1 = cutlass.Float32(0.0)
            t2 = cutlass.Float32(0.0)
            q = cutlass.Int32(0)
            base = (cutlass.Int32(nz) * TILES + gx) * mparts * (2 * GPT)
            while q < mparts:
                t1 = t1 + mP[base + q * (2 * GPT) + tid]
                t2 = t2 + mP[base + q * (2 * GPT) + GPT + tid]
                q = q + 1
            gst[tid] = t1
            gst[GPT + tid] = t2
        nvvm.barrier_cta_sync_aligned(0)
    if tid < GPT:
        gst[tid] = gst[tid] / Mf
        gst[GPT + tid] = gst[GPT + tid] / Mf
    nvvm.barrier_cta_sync_aligned(0)
    a = gst[gi]
    b = gst[GPT + gi]

    # ---- pass 2 ----
    for kk in cutlass.range_constexpr(KC):
        p = p0 + tp + kk * PPL
        if p < p1:
            ys = []
            for e in cutlass.range_constexpr(V):
                xh = (xc[kk * V + e] - mean) * rstd
                ys.append((rstd * (dc[kk * V + e] * gc[e] - a - xh * b)).to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + ((img + p) * C + c0))
    p = p0 + tp + KC * PPL
    while p < p1:
        xv = nvvm.load_ext(mXi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
        dv = nvvm.load_ext(mDYi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            xh = (xv[e].to(cutlass.Float32) - mean) * rstd
            ys.append((rstd * (dv[e].to(cutlass.Float32) * gc[e] - a - xh * b)).to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + ((img + p) * C + c0))
        p = p + PPL

    # ---- per-channel gradients: reduce across the PPL pixel groups, park in [N, C] ----
    for e in cutlass.range_constexpr(V):
        acc[(tp * TPP + lane) * V + e] = dg[e]
        acc[BT * V + (tp * TPP + lane) * V + e] = db[e]
    nvvm.barrier_cta_sync_aligned(0)
    if tp == 0:
        for e in cutlass.range_constexpr(V):
            ag = dg[e]
            ab = db[e]
            for j in cutlass.range_constexpr(PPL - 1):
                ag = ag + acc[((j + 1) * TPP + lane) * V + e]
                ab = ab + acc[BT * V + ((j + 1) * TPP + lane) * V + e]
            slot = (cutlass.Int32(nz) * mparts + my) * C + c0 + e
            mS2[slot] = ag
            mS1[slot] = ab


@cute.kernel
def _gn_bwd_nhwc_finalize(mS1, mS2, mDGamma, mDBeta, NM: cutlass.Int32, C: cutlass.Constexpr, bt: cutlass.Constexpr, has_beta: cutlass.Constexpr) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    c = bid * bt + tid
    if c < C:
        sg = cutlass.Float32(0.0)
        sb = cutlass.Float32(0.0)
        n = cutlass.Int32(0)
        while n < NM:
            sg = sg + mS2[n * C + c]
            sb = sb + mS1[n * C + c]
            n = n + 1
        mDGamma[c] = sg
        if cutlass.const_expr(has_beta):
            mDBeta[c] = sb


@cute.jit
def _gn_bwd_nhwc_host(
    mDY,
    mX,
    mDX,
    mG,
    mMean,
    mRstd,
    mS1,
    mS2,
    mP,
    mRet,
    mDGamma,
    mDBeta,
    HW,
    mparts,
    NM,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    CPG: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    G: cutlass.Constexpr,
    GPT: cutlass.Constexpr,
    LPG: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    TILES: cutlass.Constexpr,
    COOP: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
    gridm: cutlass.Constexpr,
    gridn: cutlass.Constexpr,
    fgrid: cutlass.Constexpr,
    fbt: cutlass.Constexpr,
) -> None:
    mDYi = cute.recast_tensor(mDY, it_ty)
    mXi = cute.recast_tensor(mX, it_ty)
    mDXi = cute.recast_tensor(mDX, it_ty)
    _gn_bwd_nhwc_kernel(
        mDYi,
        mXi,
        mDXi,
        mG,
        mMean,
        mRstd,
        mS1,
        mS2,
        mP,
        mRet,
        HW,
        mparts,
        C,
        V,
        CPG,
        CPC,
        TPP,
        PPL,
        BT,
        G,
        GPT,
        LPG,
        KC,
        TILES,
        COOP,
        it_ty,
        et,
        Mf,
    ).launch(grid=(TILES, gridm, gridn), block=(BT, 1, 1), smem=smem_bytes, cooperative=COOP)
    _gn_bwd_nhwc_finalize(mS1, mS2, mDGamma, mDBeta, NM, C, fbt, has_beta).launch(grid=(fgrid, 1, 1), block=(fbt, 1, 1))


_KCACHE = {}


_CPC_MAX = 128


def _geom(C, cpg, eb, bt=_BT):
    """``(V, CPC, TPP, PPL, GPT, LPG)`` or None.

    The tile is sized for COALESCING (TPP = CPC/V threads span it contiguously) and
    must hold a whole number of groups. A lane's V channels sit inside one group, and
    LPG = cpg/V consecutive lanes share a group.
    """
    V = min(16 // eb, cpg)
    while V > 1 and cpg % V != 0:
        V //= 2
    if cpg % V != 0:
        return None
    cpc = 0
    for cand in range(min(C, _CPC_MAX), 0, -1):
        if C % cand == 0 and cand % cpg == 0 and (cand // V) <= bt and bt % (cand // V) == 0:
            cpc = cand
            break
    if cpc == 0:
        return None
    tpp = cpc // V
    return V, cpc, tpp, bt // tpp, cpc // cpg, cpg // V


def eligible(C, cpg, eb):
    return _geom(C, cpg, eb) is not None


def backward(spec, dy3, x3, gamma, mean, rstd, *, has_beta, cfg, params, knobs=None):
    """Launch the native NHWC GroupNorm backward. ``x3``/``dy3`` are ``[N, H*W, C]``."""
    import torch

    N, HW, C = (int(v) for v in x3.shape)
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    CPG = int(spec.channels_per_group)
    G = int(spec.groups_per_sample)
    BT = _BT
    V, CPC, TPP, PPL, GPT, LPG = _geom(C, CPG, eb, BT)
    KC = 0 if knobs is None else int(knobs)

    TILES = C // CPC
    # Split H*W only when the (tile, image) grid alone cannot fill the machine.
    # Cooperative launch requires the whole grid co-resident, so mparts is bounded.
    mparts = max(1, min(_nsm() // max(1, TILES * N), (HW + PPL - 1) // PPL))
    COOP = mparts > 1

    dx = torch.empty_like(x3)
    dgamma = torch.empty(C, dtype=torch.float32, device=x3.device)
    dbeta = torch.empty(C, dtype=torch.float32, device=x3.device)
    NM = N * mparts
    s1 = torch.zeros(NM * C, dtype=torch.float32, device=x3.device)
    s2 = torch.zeros(NM * C, dtype=torch.float32, device=x3.device)
    pbuf = torch.empty(N * TILES * mparts * 2 * GPT, dtype=torch.float32, device=x3.device)
    ret = torch.zeros(N * TILES, dtype=torch.int32, device=x3.device)
    smem_bytes = 2 * BT * 4 + 2 * GPT * 4 + 2 * BT * V * 4 + 128
    fbt = 128
    fgrid = (C + fbt - 1) // fbt

    args = (
        dyn(dy3.reshape(-1)),
        dyn(x3.reshape(-1)),
        dyn(dx.reshape(-1)),
        dyn(gamma),
        dyn(mean),
        dyn(rstd),
        dyn(s1),
        dyn(s2),
        dyn(pbuf),
        dyn(ret),
        dyn(dgamma),
        dyn(dbeta),
        cutlass.Int32(HW),
        cutlass.Int32(mparts),
        cutlass.Int32(NM),
    )
    ce = (C, V, CPG, CPC, TPP, PPL, BT, G, GPT, LPG, KC, TILES, COOP, it_ty, et, float(HW * CPG), has_beta, smem_bytes, mparts, N, fgrid, fbt)
    key = (params.io_dtype, C, HW, N, CPG, G, V, CPC, KC, has_beta, mparts, COOP)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_gn_bwd_nhwc_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return dx, dgamma, (dbeta if has_beta else None)
