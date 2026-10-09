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
holds ``GPT = CPC/cpg`` whole groups.

The vector width is ALWAYS the full 128 bits; the lane-to-group relation then falls
into one of two cases, and the kernel handles both with one set of indices:

* ``cpg >= V`` -- a lane sits inside ONE group and ``LPG = cpg/V`` consecutive lanes
  share it (``GPL = 1`` accumulator per lane).
* ``cpg < V``  -- a lane spans ``GPL = V/cpg`` WHOLE groups (``LPG = 1``), so it keeps
  ``GPL`` accumulators and element ``e`` belongs to sub-group ``e // cpg``.

Since ``e // cpg`` is 0 for every ``e < V <= cpg``, the same expression covers both.
Tying ``V`` to ``cpg`` instead would put small groups back on 4-byte loads.

Either way the group statistic is a SEGMENTED reduce over the sharing lanes and all
``PPL`` pixel groups. A group's contributions are laid out CONTIGUOUSLY in shared
memory (``g*SEGLEN + tp*LPG + lane%LPG``), which turns that into an ordinary
power-of-two tree -- ``log2(SEGLEN)`` parallel steps instead of one thread per group
walking ``SEGLEN`` entries, which at large ``cpg`` leaves 255 of 256 threads idle.

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
from cudnn.norm.utils import dyn, run_coop, sm_count

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
    GPL: cutlass.Constexpr,
    SEGLEN: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    TILES: cutlass.Constexpr,
    COOP: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    gx, my, nz = cute.arch.block_idx()  # gx = channel TILE, my = H*W split

    NS: cutlass.Constexpr = BT * GPL  # smem slots per group statistic

    smem = SmemAllocator()
    seg = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * NS), byte_alignment=16)
    acc = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * BT * V), byte_alignment=16)

    lane = tid % TPP  # which V-wide slice of the channel TILE
    tp = tid // TPP  # which of the PPL parallel pixels
    c0 = gx * CPC + lane * V  # absolute channel of this thread's first element
    gb = (lane * V) // CPG  # first group within the tile this lane touches
    pos = tp * LPG + (lane % LPG)  # position inside the group's contiguous segment
    img = cutlass.Int64(nz) * HW
    per = (HW + mparts - 1) // mparts
    p0 = my * per
    p1 = p0 + per
    if p1 > HW:
        p1 = HW
    mn = []
    rsd = []
    for sub in cutlass.range_constexpr(GPL):
        mn.append(mMean[cutlass.Int32(nz) * G + gx * GPT + gb + sub])
        rsd.append(mRstd[cutlass.Int32(nz) * G + gx * GPT + gb + sub])

    gc = []
    for e in cutlass.range_constexpr(V):
        gc.append(mG[c0 + e].to(cutlass.Float32))

    a1 = [cutlass.Float32(0.0)] * GPL  # sum(dxhat)      -- group-wide
    a2 = [cutlass.Float32(0.0)] * GPL  # sum(dxhat*xhat) -- group-wide
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
                xh = (x - mn[e // CPG]) * rsd[e // CPG]
                dxh = d * gc[e]
                a1[e // CPG] = a1[e // CPG] + dxh
                a2[e // CPG] = a2[e // CPG] + dxh * xh
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
            xh = (x - mn[e // CPG]) * rsd[e // CPG]
            dxh = d * gc[e]
            a1[e // CPG] = a1[e // CPG] + dxh
            a2[e // CPG] = a2[e // CPG] + dxh * xh
            dg[e] = dg[e] + d * xh
            db[e] = db[e] + d
        p = p + PPL

    # group statistics: SEGMENTED reduce -- a group owns LPG consecutive lanes across
    # all PPL pixel groups, so this is not a whole-block sum. The segment is laid out
    # contiguously, so it reduces as a plain power-of-two tree.
    for sub in cutlass.range_constexpr(GPL):
        seg[(gb + sub) * SEGLEN + pos] = a1[sub]
        seg[NS + (gb + sub) * SEGLEN + pos] = a2[sub]
    st = SEGLEN // 2
    while st >= 1:  # trace-time unroll: SEGLEN is constexpr
        nvvm.barrier_cta_sync_aligned(0)
        if pos < st:
            for sub in cutlass.range_constexpr(GPL):
                sb = (gb + sub) * SEGLEN + pos
                seg[sb] = seg[sb] + seg[sb + st]
                seg[NS + sb] = seg[NS + sb] + seg[NS + sb + st]
        st //= 2
    nvvm.barrier_cta_sync_aligned(0)
    if cutlass.const_expr(COOP):
        # mparts CTAs share this (image, tile): park partials, spin, re-reduce.
        pb = ((cutlass.Int32(nz) * TILES + gx) * mparts + my) * (2 * GPT)
        if pos == 0:
            for sub in cutlass.range_constexpr(GPL):
                mP[pb + gb + sub] = seg[(gb + sub) * SEGLEN]
                mP[pb + GPT + gb + sub] = seg[NS + (gb + sub) * SEGLEN]
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
        if pos == 0:
            base = (cutlass.Int32(nz) * TILES + gx) * mparts * (2 * GPT)
            for sub in cutlass.range_constexpr(GPL):
                t1 = cutlass.Float32(0.0)
                t2 = cutlass.Float32(0.0)
                q = cutlass.Int32(0)
                while q < mparts:
                    t1 = t1 + mP[base + q * (2 * GPT) + gb + sub]
                    t2 = t2 + mP[base + q * (2 * GPT) + GPT + gb + sub]
                    q = q + 1
                seg[(gb + sub) * SEGLEN] = t1
                seg[NS + (gb + sub) * SEGLEN] = t2
        nvvm.barrier_cta_sync_aligned(0)
    av = []
    bv = []
    for sub in cutlass.range_constexpr(GPL):
        av.append(seg[(gb + sub) * SEGLEN] / Mf)
        bv.append(seg[NS + (gb + sub) * SEGLEN] / Mf)

    # ---- pass 2 ----
    for kk in cutlass.range_constexpr(KC):
        p = p0 + tp + kk * PPL
        if p < p1:
            ys = []
            for e in cutlass.range_constexpr(V):
                sb = e // CPG
                xh = (xc[kk * V + e] - mn[sb]) * rsd[sb]
                ys.append((rsd[sb] * (dc[kk * V + e] * gc[e] - av[sb] - xh * bv[sb])).to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + ((img + p) * C + c0))
    p = p0 + tp + KC * PPL
    while p < p1:
        xv = nvvm.load_ext(mXi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
        dv = nvvm.load_ext(mDYi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            sb = e // CPG
            xh = (xv[e].to(cutlass.Float32) - mn[sb]) * rsd[sb]
            ys.append((rsd[sb] * (dv[e].to(cutlass.Float32) * gc[e] - av[sb] - xh * bv[sb])).to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + ((img + p) * C + c0))
        p = p + PPL

    # ---- per-channel gradients: reduce across the PPL pixel groups, park in [N, C] ----
    # A WALK by the tp==0 row, not a tree: each thread carries V values here (unlike
    # the scalar group statistic above), so a log2(PPL) tree touches more shared
    # memory in total than the walk does -- measured 0.31 util against 0.32.
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


_gn_bwd_nhwc_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _gn_bwd_nhwc_finalize(
    mS1,
    mS2,
    mDGamma,
    mDBeta,
    NM: cutlass.Int32,
    C: cutlass.Constexpr,
    CW: cutlass.Constexpr,
    R: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    """Reduce the per-(image, H*W-part) channel gradients down to ``[C]``.

    Mapped as ``CW`` channels x ``R`` rows rather than one thread per channel: with a
    small C the channel dimension alone is a single CTA, and the ``N*mparts`` rows it
    then walks serially become a long tail -- which grows with exactly the grid depth
    the main kernel wants. ``CW`` is a warp wide so each row read stays coalesced.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    cl = tid % CW
    r = tid // CW
    c = bid * CW + cl

    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * CW * R), byte_alignment=16)

    sg = cutlass.Float32(0.0)
    sb = cutlass.Float32(0.0)
    if c < C:
        n = cutlass.Int32(r)
        while n < NM:
            sg = sg + mS2[n * C + c]
            sb = sb + mS1[n * C + c]
            n = n + R
    red[r * CW + cl] = sg
    red[CW * R + r * CW + cl] = sb
    nvvm.barrier_cta_sync_aligned(0)
    if r == 0 and c < C:
        for j in cutlass.range_constexpr(R - 1):
            sg = sg + red[(j + 1) * CW + cl]
            sb = sb + red[CW * R + (j + 1) * CW + cl]
        mDGamma[c] = sg
        if cutlass.const_expr(has_beta):
            mDBeta[c] = sb


_gn_bwd_nhwc_finalize.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
    GPL: cutlass.Constexpr,
    SEGLEN: cutlass.Constexpr,
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
    fcw: cutlass.Constexpr,
    fr: cutlass.Constexpr,
    fsmem: cutlass.Constexpr,
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
        GPL,
        SEGLEN,
        KC,
        TILES,
        COOP,
        it_ty,
        et,
        Mf,
    ).launch(grid=(TILES, gridm, gridn), block=(BT, 1, 1), smem=smem_bytes, cooperative=COOP)
    _gn_bwd_nhwc_finalize(mS1, mS2, mDGamma, mDBeta, NM, C, fcw, fr, has_beta).launch(grid=(fgrid, 1, 1), block=(fcw * fr, 1, 1), smem=fsmem)


_KCACHE = {}


_CPC_MAX = 128


def _p2(n):
    return n > 0 and (n & (n - 1)) == 0


def _geom(C, cpg, eb, bt=_BT):
    """``(V, CPC, TPP, PPL, GPT, LPG, GPL)`` or None when the tile does not exist.

    The tile is sized for COALESCING (TPP = CPC/V threads span it contiguously) and
    must hold a whole number of groups. Mirrors the forward's geometry exactly, so a
    channels-last fprop -> bprop chain stays native on both halves.
    """
    V = 16 // eb
    while V > 1 and cpg % V and V % cpg:  # a lane must nest in a group, or hold whole ones
        V //= 2
    if V < 2 or (cpg % V and V % cpg):
        return None
    lpg = max(1, cpg // V)
    gpl = max(1, V // cpg)

    def ok(cand):
        if C % cand or cand % cpg or cand % V:
            return False
        tpp = cand // V
        if tpp > bt or bt % tpp or tpp % lpg:
            return False
        return _p2((bt // tpp) * lpg)  # SEGLEN must be a power of two for the tree

    if cpg <= _CPC_MAX:
        # Small groups: the widest tile under the cap, so one CTA covers many groups.
        cands = range(min(C, _CPC_MAX), 0, -1)
    else:
        # A group already exceeds the cap, so the tile must grow to hold one whole
        # group; take the SMALLEST such tile to keep the channel grid as wide as possible.
        cands = range(cpg, min(C, bt * V) + 1)
    for cand in cands:
        if ok(cand):
            tpp = cand // V
            return V, cand, tpp, bt // tpp, cand // cpg, lpg, gpl
    return None


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
    V, CPC, TPP, PPL, GPT, LPG, GPL = _geom(C, CPG, eb, BT)
    SEGLEN = PPL * LPG
    KC = 0 if knobs is None else int(knobs)

    TILES = C // CPC
    smem_bytes = 2 * BT * GPL * 4 + 2 * BT * V * 4 + 128
    fcw, fr = 32, 8  # a warp of channels x 8 rows, so small C still gets 8-way depth
    fgrid = (C + fcw - 1) // fcw
    fsmem = 2 * fcw * fr * 4 + 128

    def _run(occ):
        # Split H*W only when the (tile, image) grid alone cannot fill the machine
        # `occ` CTAs deep, and never past the point where a part stops filling the
        # register cache. Cooperative launch requires the whole grid co-resident, so
        # `occ` is probed rather than assumed (see cudnn.norm.utils.run_coop).
        mparts = max(1, min(sm_count() * occ // max(1, TILES * N), HW // (PPL * max(1, KC))))
        COOP = mparts > 1
        return _launch(mparts, COOP)

    def _launch(mparts, COOP):
        dx = torch.empty_like(x3)
        dgamma = torch.empty(C, dtype=torch.float32, device=x3.device)
        dbeta = torch.empty(C, dtype=torch.float32, device=x3.device)
        NM = N * mparts
        s1 = torch.zeros(NM * C, dtype=torch.float32, device=x3.device)
        s2 = torch.zeros(NM * C, dtype=torch.float32, device=x3.device)
        pbuf = torch.empty(N * TILES * mparts * 2 * GPT, dtype=torch.float32, device=x3.device)
        ret = torch.zeros(N * TILES, dtype=torch.int32, device=x3.device)

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
        ce = (
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
            GPL,
            SEGLEN,
            KC,
            TILES,
            COOP,
            it_ty,
            et,
            float(HW * CPG),
            has_beta,
            smem_bytes,
            mparts,
            N,
            fgrid,
            fcw,
            fr,
            fsmem,
        )
        key = (params.io_dtype, C, HW, N, CPG, G, V, CPC, KC, has_beta, mparts, COOP)
        fn = _KCACHE.get(key)
        if fn is None:
            fn = cute.compile(_gn_bwd_nhwc_host, *args, *ce)
            _KCACHE[key] = fn
        fn(*args)
        return dx, dgamma, (dbeta if has_beta else None)

    return run_coop(("gn_bwd", params.io_dtype, C, HW, N, CPG, KC, has_beta), _run)
