# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GroupNorm forward for NHWC (channels-last), sm_100, CUTLASS primitives.

A group is ``cpg`` ADJACENT channels over all of H*W -- contiguous within a pixel in
NHWC -- so the channel tile is sized for COALESCING (``TPP = CPC/V`` threads span it)
rather than set equal to the group, and it holds ``GPT = CPC/cpg`` whole groups.

The vector width is ALWAYS the full 128 bits; the lane-to-group relation then falls
into one of two cases, and the kernel handles both with one set of indices:

* ``cpg >= V`` -- a lane sits inside ONE group and ``LPG = cpg/V`` consecutive lanes
  share it (``GPL = 1`` accumulator per lane).
* ``cpg < V``  -- a lane spans ``GPL = V/cpg`` WHOLE groups (``LPG = 1``), so it keeps
  ``GPL`` accumulators and element ``e`` belongs to sub-group ``e // cpg``.

Since ``e // cpg`` is 0 for every ``e < V <= cpg``, the same expression covers both.
Tying ``V`` to the dtype rather than to ``cpg`` is what keeps small groups off 4-byte
loads: at ``cpg = 2`` the old lane-inside-a-group rule gave ``V = 2``.

A group's contributions are laid out CONTIGUOUSLY in shared memory
(``g*SEGLEN + tp*LPG + lane%LPG``), which turns the segmented reduce into an ordinary
power-of-two tree -- ``log2(SEGLEN)`` parallel steps instead of one thread per group
walking ``SEGLEN`` entries. That matters at large ``cpg``, where ``GPT`` falls to 1
and a serial reduce would leave 255 of 256 threads idle.

Simpler than the backward in one respect: ``mean``/``rstd`` are per ``(n, g)``, so
there is no cross-image reduction. ``mparts`` CTAs split H*W only when the
``(tile, image)`` grid alone cannot fill the machine; then they rendezvous on a
retired-CTA counter indexed by ``(n, tile)`` so images never wait on each other.

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
_BT = 256
_CPC_MAX = 128


def _p2(n):
    return n > 0 and (n & (n - 1)) == 0


def _geom(C, cpg, eb, bt=_BT):
    """``(V, CPC, TPP, PPL, GPT, LPG, GPL)`` or None when the tile does not exist."""
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


@cute.kernel
def _gn_fwd_nhwc_kernel(
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
    eps: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    gx, my, nz = cute.arch.block_idx()
    NS: cutlass.Constexpr = BT * GPL  # smem slots per statistic

    smem = SmemAllocator()
    seg = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * NS), byte_alignment=16)

    lane = tid % TPP
    tp = tid // TPP
    c0 = gx * CPC + lane * V
    gb = (lane * V) // CPG  # first group of this tile the lane touches
    pos = tp * LPG + (lane % LPG)  # position inside the group's contiguous segment
    img = cutlass.Int64(nz) * HW
    per = (HW + mparts - 1) // mparts
    p0 = my * per
    p1 = p0 + per
    if p1 > HW:
        p1 = HW

    a1 = [cutlass.Float32(0.0)] * GPL
    a2 = [cutlass.Float32(0.0)] * GPL
    xc = [cutlass.Float32(0.0)] * (KC * V)

    for kk in cutlass.range_constexpr(KC):
        p = p0 + tp + kk * PPL
        if p < p1:
            xv = nvvm.load_ext(mXi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                a1[e // CPG] = a1[e // CPG] + x
                a2[e // CPG] = a2[e // CPG] + x * x
                xc[kk * V + e] = x
    p = p0 + tp + KC * PPL
    while p < p1:
        xv = nvvm.load_ext(mXi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            a1[e // CPG] = a1[e // CPG] + x
            a2[e // CPG] = a2[e // CPG] + x * x
        p = p + PPL

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

    mn = []
    rs = []
    for sub in cutlass.range_constexpr(GPL):
        m_ = seg[(gb + sub) * SEGLEN] / Mf
        mn.append(m_)
        rs.append(cute.math.rsqrt(seg[NS + (gb + sub) * SEGLEN] / Mf - m_ * m_ + eps))
    if pos == 0 and my == 0:
        for sub in cutlass.range_constexpr(GPL):
            mMean[cutlass.Int32(nz) * G + gx * GPT + gb + sub] = mn[sub]
            mRstd[cutlass.Int32(nz) * G + gx * GPT + gb + sub] = rs[sub]

    sc = []
    sh = []
    for e in cutlass.range_constexpr(V):
        f = rs[e // CPG] * mG[c0 + e].to(cutlass.Float32)
        sc.append(f)
        o = -mn[e // CPG] * f
        if cutlass.const_expr(has_beta):
            o = o + mB[c0 + e].to(cutlass.Float32)
        sh.append(o)

    for kk in cutlass.range_constexpr(KC):
        p = p0 + tp + kk * PPL
        if p < p1:
            ys = []
            for e in cutlass.range_constexpr(V):
                ys.append((xc[kk * V + e] * sc[e] + sh[e]).to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + ((img + p) * C + c0))
    p = p0 + tp + KC * PPL
    while p < p1:
        xv = nvvm.load_ext(mXi.iterator + ((img + p) * C + c0), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            ys.append((xv[e].to(cutlass.Float32) * sc[e] + sh[e]).to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + ((img + p) * C + c0))
        p = p + PPL


_gn_fwd_nhwc_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _gn_fwd_nhwc_host(
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
    eps: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
    gridm: cutlass.Constexpr,
    gridn: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _gn_fwd_nhwc_kernel(
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
        eps,
        has_beta,
    ).launch(grid=(TILES, gridm, gridn), block=(BT, 1, 1), smem=smem_bytes, cooperative=COOP)


_KCACHE = {}


def forward(spec, x3, gamma, beta, *, eps, cfg, params, knobs=None):
    """Launch the native NHWC GroupNorm forward. ``x3`` is ``[N, H*W, C]``."""
    import torch

    N, HW, C = (int(v) for v in x3.shape)
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    CPG = int(spec.channels_per_group)
    G = int(spec.groups_per_sample)
    BT = _BT
    geo = _geom(C, CPG, eb, BT)
    if geo is None:
        raise ValueError(f"NHWC GroupNorm forward: C={C} cpg={CPG} not tileable")
    V, CPC, TPP, PPL, GPT, LPG, GPL = geo
    SEGLEN = PPL * LPG
    has_beta = beta is not None
    if beta is None:
        beta = gamma
    KC = 4 if knobs is None else int(knobs)

    TILES = C // CPC
    smem_bytes = 2 * BT * GPL * 4 + 128

    def _run(occ):
        # Split H*W only as far as needed to fill the machine `occ` CTAs deep, and
        # never past the point where a part stops filling the register cache -- a
        # thinner slice buys no bandwidth and still pays the whole grid barrier.
        mparts = max(1, min(sm_count() * occ // max(1, TILES * N), HW // (PPL * KC)))
        COOP = mparts > 1
        y = torch.empty_like(x3)
        mean = torch.empty(N * G, dtype=torch.float32, device=x3.device)
        rstd = torch.empty(N * G, dtype=torch.float32, device=x3.device)
        pbuf = torch.empty(N * TILES * mparts * 2 * GPT, dtype=torch.float32, device=x3.device)
        ret = torch.zeros(N * TILES, dtype=torch.int32, device=x3.device)
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
        ce = (C, V, CPG, CPC, TPP, PPL, BT, G, GPT, LPG, GPL, SEGLEN, KC, TILES, COOP, it_ty, et, float(HW * CPG), float(eps), has_beta, smem_bytes, mparts, N)
        key = (params.io_dtype, C, HW, N, CPG, V, CPC, KC, has_beta, mparts, COOP)
        fn = _KCACHE.get(key)
        if fn is None:
            fn = cute.compile(_gn_fwd_nhwc_host, *args, *ce)
            _KCACHE[key] = fn
        fn(*args)
        return y, mean, rstd

    return run_coop(("gn_fwd", params.io_dtype, C, HW, N, CPG, KC, has_beta), _run)
