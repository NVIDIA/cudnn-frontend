# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BatchNorm backward for NHWC (channels-last), sm_100, CUTLASS primitives.

The forward's fused cooperative kernel (``fprop/kernels/batchnorm_nhwc_sm100.py``)
with a SECOND cached stream. NHWC flattens to ``[M = N*H*W, C]`` and the reduction
runs down ``M`` per channel, so ``C`` is the fast, 128-bit-vectorised axis and the
thread map is unchanged: ``TPP = C_tile/V`` threads span the contiguous channel
tile, ``PPL = BT/TPP`` pixels are in flight, and ``mparts`` CTAs split ``M``.

For channel ``c`` over its ``M`` elements with ``xhat = (x-mean)*rstd``::

    dgamma = sum(dy * xhat)        dbeta = sum(dy)
    dx     = gamma*rstd * (dy - dbeta/M - xhat * dgamma/M)

Pass 2 needs both ``dy`` and ``x``, so BOTH are cached on the single pass-1 read --
``KC`` pixels per thread in registers and ``KS`` more in shared. That is the one
structural difference from the forward, and it halves the per-stream cache depth
for a given budget.

Everything else mirrors the forward: partials to a ``[cblks, 2, mparts, C_tile]``
buffer with plain vector stores (no atomics), a gpu-scope fence plus atomic-spin
grid barrier, then a REDUNDANT finalize (every part-CTA re-reduces its channel
tile; the partials are L2-resident and it needs no second barrier).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.fprop.kernels.batchnorm_nhwc_sm100 import nhwc_cfg as _fwd_nhwc_cfg
from cudnn.norm.utils import dyn

_CTA_SS = nvvm.SharedSpace.shared_cta
_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_SM_COUNT = None

# Channel tile for the BACKWARD. The forward caps at 256, giving TPP = CPC/V = 32 --
# so a warp is exactly one pixel group and the tp-reduction is entirely cross-warp
# (shared memory, two barriers a round). Capping lower puts 32/TPP pixel groups
# INSIDE a warp, where the reduction is a butterfly shuffle.
_CPC_BWD = 128


def nhwc_cfg(C, eb, block_threads=512):
    """Backward geometry: like the forward's but with its own channel-tile cap."""
    V = 16 // eb
    if C % V != 0:
        return None
    cpc = 0
    for cand in range(min(C, _CPC_BWD), V - 1, -V):
        if C % cand == 0:
            cpc = cand
            break
    if cpc == 0:
        return None
    tpp = cpc // V
    if tpp > block_threads or block_threads % tpp != 0:
        return None
    return V, cpc, tpp, block_threads // tpp, C // cpc, block_threads


def _nsm():
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


@cute.kernel
def _bn_bwd_nhwc_kernel(
    mDYi,
    mXi,
    mDXi,
    mG,
    mMean,
    mRstd,
    mP,
    mRet,
    mDGamma,
    mDBeta,
    M: cutlass.Int32,
    mparts: cutlass.Int32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    KS: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cx, my, _ = cute.arch.block_idx()
    NSEG: cutlass.Constexpr = V // 4

    smem = SmemAllocator()
    # Two staging planes so BOTH accumulators reduce in ONE shared round: the
    # tp-reduction is cross-WARP (TPP=32 means a warp is exactly one pixel group),
    # so it cannot use shuffles and every round costs two barriers. Fusing the
    # pairs takes the kernel from 8 barriers to 4.
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
    per = (M + mparts - 1) // mparts
    r0 = my * per
    r1 = r0 + per
    if r1 > M:
        r1 = M

    mn = []
    rs = []
    for e in cutlass.range_constexpr(V):
        mn.append(mMean[c0 + e])
        rs.append(mRstd[c0 + e])

    s = [cutlass.Float32(0.0)] * V  # sum(dy)
    sq = [cutlass.Float32(0.0)] * V  # sum(dy*xhat)
    xc = [cutlass.Float32(0.0)] * (KC * V)
    dc = [cutlass.Float32(0.0)] * (KC * V)

    # ---- pass 1: reduce; cache BOTH streams, KC pixels in regs then KS in smem ----
    for kk in cutlass.range_constexpr(KC):
        rk = r0 + tp + kk * PPL
        if rk < r1:
            xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(rk) * C + c0), dtype=it_ty, count=V).bitcast(et)
            dv = nvvm.load_ext(mDYi.iterator + (cutlass.Int64(rk) * C + c0), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                d = dv[e].to(cutlass.Float32)
                s[e] = s[e] + d
                sq[e] = sq[e] + d * ((x - mn[e]) * rs[e])
                xc[kk * V + e] = x
                dc[kk * V + e] = d
    for ks in cutlass.range_constexpr(KS):
        rk = r0 + tp + (KC + ks) * PPL
        if rk < r1:
            rx = nvvm.load_ext(mXi.iterator + (cutlass.Int64(rk) * C + c0), dtype=it_ty, count=V)
            rd = nvvm.load_ext(mDYi.iterator + (cutlass.Int64(rk) * C + c0), dtype=it_ty, count=V)
            xv = rx.bitcast(et)
            dv = rd.bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                d = dv[e].to(cutlass.Float32)
                s[e] = s[e] + d
                sq[e] = sq[e] + d * ((x - mn[e]) * rs[e])
            nvvm.store_ext(rx, scx.iterator + (tid * KS + ks) * V, shared_space=_CTA_SS)
            nvvm.store_ext(rd, scd.iterator + (tid * KS + ks) * V, shared_space=_CTA_SS)
    row = r0 + tp + (KC + KS) * PPL
    while row < r1:
        xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        dv = nvvm.load_ext(mDYi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            d = dv[e].to(cutlass.Float32)
            s[e] = s[e] + d
            sq[e] = sq[e] + d * ((x - mn[e]) * rs[e])
        row = row + PPL

    def _reduce2(va, vb):
        """Reduce BOTH accumulators across the PPL pixel groups in one shared round."""
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

    psum = cx * (mparts * TPP * V * 2)
    psq = psum + mparts * TPP * V
    if tp == 0:
        off = (my * TPP + lane) * V
        for h in cutlass.range_constexpr(NSEG):
            sseg = cutlass.Vector.from_elements(tuple(s[h * 4 + j] for j in range(4)), cutlass.Float32)
            qseg = cutlass.Vector.from_elements(tuple(sq[h * 4 + j] for j in range(4)), cutlass.Float32)
            nvvm.store_ext(sseg, mP.iterator + (psum + off + h * 4))
            nvvm.store_ext(qseg, mP.iterator + (psq + off + h * 4))

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

    # ---- redundant finalize across all part-CTAs of this channel tile ----
    facc = [cutlass.Float32(0.0)] * V
    faccsq = [cutlass.Float32(0.0)] * V
    part = tp
    while part < mparts:
        off = (part * TPP + lane) * V
        for h in cutlass.range_constexpr(NSEG):
            sv = nvvm.load_ext(mP.iterator + (psum + off + h * 4), dtype=cutlass.Float32, count=4)
            qv = nvvm.load_ext(mP.iterator + (psq + off + h * 4), dtype=cutlass.Float32, count=4)
            for j in cutlass.range_constexpr(4):
                facc[h * 4 + j] = facc[h * 4 + j] + sv[j]
                faccsq[h * 4 + j] = faccsq[h * 4 + j] + qv[j]
        part = part + PPL
    facc, faccsq = _reduce2(facc, faccsq)

    if tp == 0:
        for e in cutlass.range_constexpr(V):
            stat[lane * V + e] = facc[e] / Mf  # a = dbeta / M
            stat[CPC + lane * V + e] = faccsq[e] / Mf  # b = dgamma / M
            if my == 0:
                mDGamma[c0 + e] = faccsq[e]
                if cutlass.const_expr(has_beta):
                    mDBeta[c0 + e] = facc[e]
    nvvm.barrier_cta_sync_aligned(0)

    aa = []
    bb = []
    ga = []
    for e in cutlass.range_constexpr(V):
        aa.append(stat[lane * V + e])
        bb.append(stat[CPC + lane * V + e])
        ga.append(mG[c0 + e].to(cutlass.Float32) * rs[e])

    # ---- pass 2: dx from the register cache, then smem, then re-read ----
    for kk in cutlass.range_constexpr(KC):
        rk = r0 + tp + kk * PPL
        if rk < r1:
            ys = []
            for e in cutlass.range_constexpr(V):
                xh = (xc[kk * V + e] - mn[e]) * rs[e]
                ys.append((ga[e] * (dc[kk * V + e] - aa[e] - xh * bb[e])).to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (cutlass.Int64(rk) * C + c0))
    for ks in cutlass.range_constexpr(KS):
        rk = r0 + tp + (KC + ks) * PPL
        if rk < r1:
            xv = nvvm.load_ext(scx.iterator + (tid * KS + ks) * V, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
            dv = nvvm.load_ext(scd.iterator + (tid * KS + ks) * V, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
            ys = []
            for e in cutlass.range_constexpr(V):
                xh = (xv[e].to(cutlass.Float32) - mn[e]) * rs[e]
                ys.append((ga[e] * (dv[e].to(cutlass.Float32) - aa[e] - xh * bb[e])).to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (cutlass.Int64(rk) * C + c0))
    row = r0 + tp + (KC + KS) * PPL
    while row < r1:
        xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        dv = nvvm.load_ext(mDYi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            xh = (xv[e].to(cutlass.Float32) - mn[e]) * rs[e]
            ys.append((ga[e] * (dv[e].to(cutlass.Float32) - aa[e] - xh * bb[e])).to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (cutlass.Int64(row) * C + c0))
        row = row + PPL


@cute.jit
def _bn_bwd_nhwc_host(
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
    M,
    mparts,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    KS: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    cblks: cutlass.Constexpr,
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
    _bn_bwd_nhwc_kernel(
        mDYi,
        mXi,
        mDXi,
        mG,
        mMean,
        mRstd,
        mP,
        mRet,
        mDGamma,
        mDBeta,
        M,
        mparts,
        C,
        V,
        TPP,
        PPL,
        BT,
        KC,
        KS,
        CPC,
        it_ty,
        et,
        Mf,
        has_beta,
    ).launch(grid=(cblks, mparts, 1), block=(BT, 1, 1), smem=smem_bytes, cooperative=True, min_blocks_per_mp=mbpm)


_KCACHE = {}
_OCC = {}

# (min_blocks_per_mp, KC regs, KS smem pixels), measured per shape on the RN50 set.
# The backward caches TWO streams, so for a given budget each is half the forward's
# depth -- which is why the winning configs here are shallower than the forward's.
#   large M  -> occ=2 with a small smem cache (the slice is long, occupancy wins)
#   small M  -> occ=1 with a 2-pixel register cache and a big L1
_KNOB_BIG = (2, 0, 3)
_KNOB_SMALL = (0, 2, 0)
_BIG_M = 65536


def _knobs_for(M, C):
    return _KNOB_BIG if M > _BIG_M else _KNOB_SMALL


def _smem_bytes(BT, V, CPC, KS, eb):
    return 2 * BT * V * 4 + 2 * CPC * 4 + 2 * BT * KS * V * eb + 128


def backward(spec, dy2d, x2d, gamma, saved_mean, saved_rstd, *, has_beta, cfg, params, knobs=None):
    """Launch the NHWC BatchNorm backward on views of shape ``[M, C]``."""
    import torch

    M, C = int(x2d.shape[0]), int(x2d.shape[1])
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    geo = nhwc_cfg(C, eb)
    if geo is None:
        raise ValueError(f"NHWC BatchNorm backward: C={C} is not 128-bit tileable")
    V, CPC, TPP, PPL, cblks, BT = geo

    dx = torch.empty_like(x2d)
    dgamma = torch.empty(C, dtype=torch.float32, device=x2d.device)
    dbeta = torch.empty(C, dtype=torch.float32, device=x2d.device)

    mbpm, KC, KS = knobs if knobs is not None else _knobs_for(M, C)
    smem_bytes = _smem_bytes(BT, V, CPC, KS, eb)

    ce = (C, V, TPP, PPL, BT, KC, KS, CPC, cblks, it_ty, et, float(M), has_beta, smem_bytes, mbpm)
    key = (params.io_dtype, C, KC, KS, CPC, has_beta, mbpm)
    okey = key + (M,)
    occ0 = _OCC.get(okey, 2 if mbpm != 1 else 1)
    can2 = smem_bytes * 2 <= 224 * 1024
    fn = _KCACHE.get(key)
    for occ in ([2, 1] if (occ0 == 2 and can2) else [1]):
        mparts = max(1, min(occ * _nsm() // max(1, cblks), (M + PPL - 1) // PPL))
        pbuf = torch.empty(cblks * mparts * TPP * V * 2, dtype=torch.float32, device=x2d.device)
        ret = torch.zeros(cblks, dtype=torch.int32, device=x2d.device)
        args = (
            dyn(dy2d),
            dyn(x2d),
            dyn(dx),
            dyn(gamma),
            dyn(saved_mean),
            dyn(saved_rstd),
            dyn(pbuf),
            dyn(ret),
            dyn(dgamma),
            dyn(dbeta),
            cutlass.Int32(M),
            cutlass.Int32(mparts),
        )
        if fn is None:
            fn = cute.compile(_bn_bwd_nhwc_host, *args, *ce)
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
