# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-support-pass SSA sparse-attention + teacher forward.

One CTA owns one query row and *all* Hq heads, so each gathered K/V row feeds
every head instead of being re-read once per 16-head group.  Because the
teacher distribution needs the final per-head normaliser before any support
slot can be written, the support is walked twice:

  pass A  gather K, QK MMA, stable online (max, sum) update in registers;
  pass B  gather K and V, QK MMA again, exp/normalise into a BF16 P^T tile,
          teacher reduction, PV MMA into the FP32 TMEM accumulator.

That is three gathered K/V streams per query instead of ``2 * Hq/16``, and it
keeps the TMEM/register footprint independent of TopK, so the same schedule
covers TopK 512/1024/2048 at Hq 32 and 96.
"""

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack, make_fake_stream

I32 = cutlass.Int32

D = 256
MT = 128  # support rows per QK MMA tile
DCH = 128  # depth elements per gathered K granule
KSUP = 128  # support rows per PV chunk
NW = 12
NT = NW * 32  # 384 threads
NHG = NT // MT  # 3 head groups
STG = 4  # TMEM score stages
P_GRP = 32  # MN swizzle atom width for BF16 P^T

SCALE = 0.0625
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
C1 = SCALE * LOG2E
NEG = float("-inf")  # exact identity: never clips a representable score
NEGC = -1.0e30  # clamp: -inf minus -inf is NaN for an all-masked warp


def _tmem_cols(hq):
    need = (STG + 1) * hq
    cols = 32
    while cols < need:
        cols *= 2
    return cols


def _smem_bytes(hq, topk):
    return (
        2 * (2 * MT * DCH) * 2  # two K granule buffers (V^T aliases the 2nd)
        + hq * D * 2  # Q
        + hq * KSUP * 2  # P^T
        + 2 * topk * 4  # selected indices (2 rows)
        + 2 * hq * 4 * 4  # sRedM + sRedS
        + 2 * hq * 4  # sMx + sRl
        + 2 * NHG * MT * 4  # teacher partials (double buffered)
        + 3 * (topk // MT) * 8  # mbarriers
        + 2048
    )


@cute.jit
def _gather_k(cpa, mK8, sI, sK8, base, tidx, ncp: cutlass.Constexpr):
    """Issue both 128-deep granules of a gathered K row from one loop body.

    The two granules share the support row, hence the same shared-memory
    index read and the same ``idx * k_token_stride`` row base, so folding
    them together halves the index traffic and the address arithmetic on the
    gather path without changing the issued copies.
    """
    for i in cutlass.range_constexpr(ncp):
        c = i * NT + tidx
        if c < MT * 8:
            r = c >> 3
            ch = c & 7
            idx = sI[base + r]
            for q in cutlass.range_constexpr(4):
                cute.copy(cpa, mK8[idx, q * 8 + ch, None], sK8[q >> 1, q & 1, r, ch, None])


@cute.jit
def _gather_v(cpa, mV8, sI, sV8, base, tidx, ncp: cutlass.Constexpr):
    """Both depth halves of a gathered V row from one index read."""
    for i in cutlass.range_constexpr(ncp):
        c = i * NT + tidx
        if c < MT * 8:
            r = c >> 3
            ch = c & 7
            idx = sI[base + r]
            for q in cutlass.range_constexpr(4):
                cute.copy(cpa, mV8[idx, q * 8 + ch, None], sV8[q >> 1, q & 1, ch, r, None])


@cute.jit
def _online8(mp, fl, frg, base: cutlass.Constexpr):
    """Stable online (max, shifted-sum) update with one exp2 per score.

    ``(mo - mn) + (x - mn)`` with ``mn = max(mo, x)`` is exactly
    ``min(mo, x) - max(mo, x)``: one FMNMX plus one packed subtract replaces
    two subtracts and an add, and the packed multiply folds the log2(e) scale
    for a lane pair at once.  Bit-identical, two fewer ALU ops per score.
    """
    for j in cutlass.range_constexpr(0, 8, 2):
        x0 = frg[j]
        x1 = frg[j + 1]
        o0 = mp[base + j]
        o1 = mp[base + j + 1]
        n0 = cute.arch.fmax(o0, x0)
        n1 = cute.arch.fmax(o1, x1)
        d0, d1 = cute.arch.sub_packed_f32x2((cute.arch.fmin(o0, x0), cute.arch.fmin(o1, x1)), (n0, n1))
        d0, d1 = cute.arch.mul_packed_f32x2((d0, d1), (C1, C1))
        e0 = cute.math.exp2(d0, fastmath=True)
        e1 = cute.math.exp2(d1, fastmath=True)
        l0 = fl[base + j]
        l1 = fl[base + j + 1]
        lo0, lo1 = cute.arch.add_packed_f32x2((l0, l1), (e0, e1))
        hi0, hi1 = cute.arch.fma_packed_f32x2((l0, l1), (e0, e1), (cutlass.Float32(1.0), cutlass.Float32(1.0)))
        fl[base + j] = hi0 if x0 > o0 else lo0
        fl[base + j + 1] = hi1 if x1 > o1 else lo1
        mp[base + j] = n0
        mp[base + j + 1] = n1


@cute.jit
def _fold_tile(tld, thr_ld, tptr, frg, mp, fl, s, vlen, h0g, scol, HQ: cutlass.Constexpr, NSUB: cutlass.Constexpr):
    """TMEM loads stay warp-collective; only the arithmetic is masked."""
    for hsub in cutlass.range_constexpr(NSUB):
        tv = cute.make_tensor(tptr + (scol + h0g + hsub * 8), cute.make_layout((MT, 8), stride=(65536, 1)))
        cute.copy(tld, thr_ld.partition_S(tv), frg)
        if s < vlen:
            _online8(mp, fl, frg, hsub * 8)


@cute.jit
def _exp_norm8(frg, neg_max, base: cutlass.Constexpr, nr, ob, tsum):
    """p = exp2((s - max) * C1 + log2(1/sum)).

    ``s - max`` keeps its own rounding step: the shift cancels exactly even
    for the adversarial raw logits (|s| ~ 2^100) the semantic probes inject,
    which a single fused ``s*C1 + bias`` cannot represent.  The log2 of the
    reciprocal sum then rides along in the FMA addend, so the separate
    normalising multiply disappears: three packed ALU ops per lane pair
    instead of four, and one fewer rounding step on the probability.
    """
    for j in cutlass.range_constexpr(0, 8, 2):
        x0, x1 = cute.arch.add_packed_f32x2((frg[j], frg[j + 1]), (neg_max[base + j], neg_max[base + j + 1]))
        x0, x1 = cute.arch.fma_packed_f32x2((x0, x1), (C1, C1), (nr[base + j], nr[base + j + 1]))
        p0 = cute.math.exp2(x0, fastmath=True)
        p1 = cute.math.exp2(x1, fastmath=True)
        ob[j] = p0.to(cutlass.BFloat16)
        ob[j + 1] = p1.to(cutlass.BFloat16)
        tsum[0], tsum[1] = cute.arch.add_packed_f32x2((tsum[0], tsum[1]), (p0, p1))


@cute.kernel
def _ssa(
    mQ8,
    mK8,
    mV8,
    mIdx,
    mVlen,
    mO,
    mLse,
    mTea,
    mSoftmaxMax,
    mSoftmaxInvSum,
    tmma_qk,
    tmma_pv,
    TQ: I32,
    NBLK: I32,
    HQ: cutlass.Constexpr,
    TOPK: cutlass.Constexpr,
):
    NQK = TOPK // MT
    # Support tiles pass A leaves resident in TMEM for pass B.  A third stage
    # fits in the 512-column allocation, but at TopK1024 the shorter pass-B
    # walk cannot amortise the extra ramp, so it is taken only at TopK2048.
    # Four support tiles stay resident in TMEM for pass B.  Only one of the
    # two PV accumulators needs its own columns: the other aliases the stage
    # holding support tile 0, which pass B drains in iteration 0 and the first
    # PV MMA does not touch until iteration 1.  That buys a fourth residual
    # stage inside the same 512-column allocation, so pass B re-gathers (and
    # re-multiplies) four fewer support tiles per query row.
    RES = 4 if NQK >= 4 else NQK
    # For TopK1024, launch V^T tile 0 before the inter-pass reduction so its
    # 64 KiB gather overlaps the final softmax-state combine.  At TopK2048 the
    # extra in-flight group lengthens the deeper pass-B pipeline, so retain the
    # established drain-then-gather cadence there.
    EV0 = TOPK <= 1024
    STGV = 4
    HPG = HQ // NHG
    NSUB = HPG // 8
    NCP = (MT * 8 + NT - 1) // NT
    NCQ = HQ * D // 8 // NT
    NCI = (TOPK + NT - 1) // NT
    NCI4 = (TOPK // 4 + NT - 1) // NT
    INV_H = 1.0 / float(HQ)

    tidx, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    warp_id = tidx // 32
    lane = tidx % 32
    sub = tidx % MT
    hg = tidx // MT
    lg = sub // 32
    h0g = hg * HPG

    q_beg = (blk * TQ) // NBLK
    q_end = ((blk + 1) * TQ) // NBLK
    nloc = q_end - q_beg

    # ---------------- static layouts ----------------
    a_l = sm100_utils.make_smem_layout_a(tmma_qk, (MT, HQ, DCH), cutlass.BFloat16, 2)
    b_l = sm100_utils.make_smem_layout_b(tmma_qk, (MT, HQ, D), cutlass.BFloat16, 1)
    va_l = sm100_utils.make_smem_layout_a(tmma_pv, (MT, HQ, KSUP), cutlass.BFloat16, 2)
    pb_l = sm100_utils.make_smem_layout_b(tmma_pv, (MT, HQ, KSUP), cutlass.BFloat16, 1)

    # ---------------- shared memory ----------------
    smem = utils.SmemAllocator()
    sKa = smem.allocate_tensor(cutlass.BFloat16, a_l.outer, byte_alignment=1024, swizzle=a_l.inner)
    sKb = smem.allocate_tensor(cutlass.BFloat16, a_l.outer, byte_alignment=1024, swizzle=a_l.inner)
    sQ = smem.allocate_tensor(cutlass.BFloat16, b_l.outer, byte_alignment=1024, swizzle=b_l.inner)
    sP = smem.allocate_tensor(cutlass.BFloat16, pb_l.outer, byte_alignment=1024, swizzle=pb_l.inner)
    sIdx = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2, TOPK), stride=(TOPK, 1)), byte_alignment=16)
    sIdx4 = cute.make_tensor(
        sIdx.iterator,
        cute.make_layout((2, TOPK // 4, 4), stride=(TOPK, 4, 1)),
    )
    mIdx4 = cute.make_tensor(
        mIdx.iterator,
        cute.make_layout((TQ, TOPK // 4, 4), stride=(TOPK, 4, 1)),
    )
    sRedM = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HQ, 4), stride=(4, 1)), byte_alignment=16)
    sRedS = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HQ, 4), stride=(4, 1)), byte_alignment=16)
    sMx = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HQ), byte_alignment=16)
    sRl = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HQ), byte_alignment=16)
    sTea = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((2, NHG, MT), stride=(NHG * MT, MT, 1)),
        byte_alignment=16,
    )
    mbart = smem.allocate_tensor(cutlass.Int64, cute.make_layout(3 * NQK), byte_alignment=16)
    mbar = mbart.iterator

    kap = sKa.iterator
    kbp = sKb.iterator
    sK8a = cute.make_tensor(kap, cute.make_layout((2, 2, MT, 8, 8), stride=(MT * DCH, 8192, 64, 8, 1)))
    sK8b = cute.make_tensor(kbp, cute.make_layout((2, 2, MT, 8, 8), stride=(MT * DCH, 8192, 64, 8, 1)))
    # V^T stages alias the second K granule buffer exactly.
    sV = cute.make_tensor(kbp, va_l.outer)
    sV8 = cute.make_tensor(kbp, cute.make_layout((2, 2, 8, KSUP, 8), stride=(MT * KSUP, KSUP * 64, 8, 64, 1)))
    qp = sQ.iterator
    sQ8 = cute.make_tensor(qp, cute.make_layout((4, HQ, 8, 8), stride=(HQ * 64, 64, 8, 1)))
    sPv = cute.make_tensor(sP.iterator, cute.make_layout((HQ * KSUP // 8, 8), stride=(8, 1)))

    cpa = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), cutlass.BFloat16, num_bits_per_copy=128)
    cpai = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), cutlass.Int32, num_bits_per_copy=128)

    if tidx == 0:
        for i in cutlass.range_constexpr(3 * NQK):
            cute.arch.mbarrier_init(mbar + i, 1)
    cute.arch.mbarrier_init_fence()

    # ---------------- TMEM ----------------
    tbar = pipeline.NamedBarrier(barrier_id=1, num_threads=NT)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tbar)
    tmem.allocate(_tmem_cols(HQ))
    tmem.wait_for_alloc()
    tptr = tmem.retrieve_ptr(cutlass.Float32)

    accS_shape = tmma_qk.partition_shape_C((MT, HQ))
    fakeS = tmma_qk.make_fragment_C(cute.append(accS_shape, STGV))
    tS = cute.make_tensor(tptr, fakeS.layout)
    fakeO = tmma_pv.make_fragment_C(cute.append(accS_shape, 1))
    tO0 = cute.make_tensor(tptr + STGV * HQ, fakeO.layout)
    tO1 = cute.make_tensor(tptr + HQ, fakeO.layout)

    tCrKa = tmma_qk.make_fragment_A(sKa)
    tCrKb = tmma_qk.make_fragment_A(sKb)
    tCrQ = tmma_qk.make_fragment_B(sQ)
    tCrV = tmma_pv.make_fragment_A(sV)
    tCrP = tmma_pv.make_fragment_B(sP)

    ldop = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(8)), cutlass.Float32)
    tprobe = cute.make_tensor(tptr, cute.make_layout((MT, 8), stride=(65536, 1)))
    tld = tcgen05.make_tmem_copy(ldop, tprobe)
    thr_ld = tld.get_slice(sub)
    cId = cute.make_identity_tensor((MT, 8))
    tCrd = thr_ld.partition_D(cId)
    frg = cute.make_fragment_like(tCrd, cutlass.Float32)
    row = tCrd[0][0]

    pv4 = cute.make_rmem_tensor((4,), cutlass.Float32)
    pl4 = cute.make_rmem_tensor((4,), cutlass.Float32)
    ob = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    mp = cute.make_rmem_tensor((HPG,), cutlass.Float32)
    fl = cute.make_rmem_tensor((HPG,), cutlass.Float32)
    tsum = cute.make_rmem_tensor((2,), cutlass.Float32)

    # ---------------- prime the first query row ----------------
    for i in cutlass.range_constexpr(NCI):
        c = i * NT + tidx
        if c < TOPK:
            sIdx[0, c] = mIdx[q_beg, c]
    cute.arch.barrier()
    # Put the compact next-row index transfer ahead of Q and the two large K
    # granules.  The first pass-A wait then retires indices/Q/K[last] without
    # unnecessarily waiting for K[last-1].
    if cutlass.const_expr(EV0):
        t1 = q_beg + 1
        if t1 >= TQ:
            t1 = TQ - 1
        for i in cutlass.range_constexpr(NCI4):
            c = i * NT + tidx
            if c < TOPK // 4:
                cute.copy(cpai, mIdx4[t1, c, None], sIdx4[1, c, None])
    for i in cutlass.range_constexpr(NCQ):
        c = i * NT + tidx
        dc = c % 32
        cute.copy(cpa, mQ8[q_beg, c, None], sQ8[dc // 8, c // 32, dc % 8, None])
    if cutlass.const_expr(EV0):
        cute.arch.cp_async_commit_group()
    sI0 = cute.make_tensor(sIdx.iterator, cute.make_layout(TOPK))
    _gather_k(cpa, mK8, sI0, sK8a, I32((NQK - 1) * MT), tidx, NCP)
    cute.arch.cp_async_commit_group()
    _gather_k(cpa, mK8, sI0, sK8b, I32((NQK - 2) * MT), tidx, NCP)
    cute.arch.cp_async_commit_group()

    for qi in cutlass.range(nloc):
        ph = I32(qi % 2)
        cur = qi % 2
        t = q_beg + qi
        tn = t + 1
        if tn >= TQ:
            tn = TQ - 1
        vlen = mVlen[t]
        sIc = cute.make_tensor(sIdx.iterator + cur * TOPK, cute.make_layout(TOPK))
        sIn = cute.make_tensor(sIdx.iterator + (1 - cur) * TOPK, cute.make_layout(TOPK))

        if cutlass.const_expr(not EV0):
            for i in cutlass.range_constexpr(NCI4):
                c = i * NT + tidx
                if c < TOPK // 4:
                    cute.copy(cpai, mIdx4[tn, c, None], sIdx4[1 - cur, c, None])
            cute.arch.cp_async_commit_group()

        for j in cutlass.range_constexpr(HPG):
            mp[j] = cutlass.Float32(NEG)
            fl[j] = cutlass.Float32(0.0)

        # ---------------- pass A: online (max, sum) ----------------
        # The support is walked backwards so the four score tiles pass A
        # leaves resident in TMEM are exactly the four tiles pass B consumes
        # first: tiles 0..3 need neither a re-gather nor a second QK MMA,
        # removing 4 of the 3*NQK gathered K/V tiles and 4 of the 2*NQK QK
        # MMAs per query row.  (max, sum) is order independent up to FP32
        # rounding, so the reversed walk is numerically equivalent.
        for ntp in cutlass.range(NQK // 2, unroll=4):
            for u in cutlass.range_constexpr(2):
                nt = 2 * ntp + u
                # Support tiles 3 and 2 are diverted to stages 3 and 2 so
                # the walk ends with tiles 3/2/1/0 resident in stages 3/2/0/1.
                last2 = ntp == NQK // 2 - 2
                if cutlass.const_expr(u == 0):
                    wstg = I32(3) if last2 else I32(0)
                else:
                    wstg = I32(2) if last2 else I32(1)
                rcol = I32(2 * HQ) if ntp == NQK // 2 - 1 else I32(HQ)
                rcol1 = I32(3 * HQ) if last2 else I32(0)
                cute.arch.cp_async_wait_group(1)
                cute.arch.barrier()
                cute.arch.fence_view_async_shared()
                if warp_id == 0:
                    for st in cutlass.range_constexpr(2):
                        for kb in cutlass.range_constexpr(DCH // 16):
                            tmma_qk.set(tcgen05.Field.ACCUMULATE, not (st == 0 and kb == 0))
                            cute.gemm(
                                tmma_qk,
                                tS[None, None, None, wstg],
                                (tCrKa if u == 0 else tCrKb)[None, None, kb, st],
                                tCrQ[None, None, st * (DCH // 16) + kb, 0],
                                tS[None, None, None, wstg],
                            )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + nt)

                # Fold the previous tile (already retired, other TMEM stage)
                # while this tile's QK MMA is still in flight.
                if u == 1:
                    _fold_tile(tld, thr_ld, tptr, frg, mp, fl, (NQK - 1 - 2 * ntp) * MT + row, vlen, h0g, rcol1, HQ, NSUB)
                else:
                    if ntp >= 1:
                        _fold_tile(tld, thr_ld, tptr, frg, mp, fl, (NQK - 2 * ntp) * MT + row, vlen, h0g, rcol, HQ, NSUB)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_wait(mbar + nt, ph)

                # Hand buffer A to pass B with score tile 2's K: tiles 0 and 1
                # are already resident in TMEM stages 1 and 0.  V tile 0 is
                # issued only after the walk so K stays the older cp.async
                # group and can be waited independently in steady state.
                if cutlass.const_expr(RES < NQK):
                    if ntp == NQK // 2 - 1 and u == 0:
                        _gather_k(cpa, mK8, sIc, sK8a, I32(RES * MT), tidx, NCP)
                if ntp < NQK // 2 - 1:
                    nb = (NQK - 3 - nt) * MT
                    if u == 0:
                        _gather_k(cpa, mK8, sIc, sK8a, nb, tidx, NCP)
                    else:
                        _gather_k(cpa, mK8, sIc, sK8b, nb, tidx, NCP)
                cute.arch.cp_async_commit_group()

        if cutlass.const_expr(EV0):
            # sKb held the tile pass A just consumed; its QK MMA is retired,
            # so V^T tile 0 can overlap the final score fold and reduction.
            _gather_v(cpa, mV8, sIc, sV8, I32(0), tidx, NCP)
            cute.arch.cp_async_commit_group()

        # Drain the final support tile (tile 0, the last one walked).
        _fold_tile(tld, thr_ld, tptr, frg, mp, fl, row, vlen, h0g, I32(HQ), HQ, NSUB)
        cute.arch.fence_view_async_tmem_load()

        # ---------------- softmax state ----------------
        for j in cutlass.range_constexpr(HPG):
            mw = cute.arch.warp_redux_sync(mp[j], "fmax", nan=True)
            dd = cute.arch.fmax(mp[j] - mw, cutlass.Float32(NEGC))
            fl[j] = fl[j] * cute.math.exp2(dd * C1, fastmath=True)
            mp[j] = mw
        # HPG lanes-wide sums folded as one butterfly instead of HPG separate
        # five-step reductions: at every step a lane forwards the half of its
        # head range it will not own and keeps the other, so the block costs
        # 31 shuffles and ends with lane j holding head j's denominator.
        nred = HPG
        offr = 16
        while nred > 1:
            hhf = nred // 2
            sel = (lane & offr) != 0
            for i in range(hhf):
                a = fl[i]
                b = fl[i + hhf]
                snd = a if sel else b
                kep = b if sel else a
                fl[i] = kep + cute.arch.shuffle_sync_bfly(snd, offset=offr, mask=-1, mask_and_clamp=31)
            nred = hhf
            offr = offr // 2
        if lane == 0:
            for j in cutlass.range_constexpr(HPG):
                sRedM[h0g + j, lg] = mp[j]
        sRedS[h0g + lane, lg] = fl[0]
        cute.arch.barrier()
        if tidx < HQ:
            cute.autovec_copy(sRedM[tidx, None], pv4)
            cute.autovec_copy(sRedS[tidx, None], pl4)
            m = cute.arch.fmax(cute.arch.fmax(pv4[0], pv4[1]), cute.arch.fmax(pv4[2], pv4[3]))
            l = (
                pl4[0] * cute.math.exp2((pv4[0] - m) * C1, fastmath=True)
                + pl4[1] * cute.math.exp2((pv4[1] - m) * C1, fastmath=True)
                + pl4[2] * cute.math.exp2((pv4[2] - m) * C1, fastmath=True)
                + pl4[3] * cute.math.exp2((pv4[3] - m) * C1, fastmath=True)
            )
            lgv = cute.log2(l, approx=True, ftz=True)
            sMx[tidx] = m
            sRl[tidx] = cutlass.Float32(0.0) - lgv
            mSoftmaxMax[t, tidx] = m
            mSoftmaxInvSum[t, tidx] = cute.rcp(l, approx=True, ftz=True)
            mLse[t, tidx] = m * SCALE + lgv * LN2
        cute.arch.barrier()
        for j in cutlass.range_constexpr(HPG):
            mp[j] = cutlass.Float32(0.0) - sMx[h0g + j]
            fl[j] = sRl[h0g + j]

        # ---------------- pass B: probabilities, teacher, PV ----------------
        # Keep K as the older and V as the younger cp.async group.  Waiting for
        # one group exposes K to QK while the previous V tail is still in
        # flight; after PV is queued, next-K gathering overlaps that PV.
        for nt in cutlass.range(NQK + 1, unroll=4):
            # tiles 0..3 live in stages 1, 0, 2, 3; from tile 4 on the QK
            # MMA rotates between stages 2 (even nt) and 0 (odd nt), both long
            # drained by then.  Stage 1 belongs to the aliased PV accumulator.
            pcol = I32(HQ) if nt == 0 else (I32(3 * HQ) if nt == 3 else (1 - nt % 2) * I32(2 * HQ))
            # Pass A leaves the first RES score tiles resident in TMEM, so
            # their pass-B prefix consumes neither the gathered-K buffers nor
            # a new QK result.  Preserve the readiness bracket only for tiles
            # that actually issue QK; the independent V/PV bracket below still
            # orders probability stores and the value operand for every tile.
            if nt >= RES:
                if cutlass.const_expr(EV0):
                    cute.arch.cp_async_wait_group(1)
                else:
                    if nt == 0:
                        cute.arch.cp_async_wait_group(0)
                    else:
                        cute.arch.cp_async_wait_group(1)
                cute.arch.barrier()
                cute.arch.fence_view_async_shared()
            if nt >= RES and nt < NQK:
                if warp_id == 0:
                    for st in cutlass.range_constexpr(2):
                        for kb in cutlass.range_constexpr(DCH // 16):
                            tmma_qk.set(tcgen05.Field.ACCUMULATE, not (st == 0 and kb == 0))
                            cute.gemm(
                                tmma_qk,
                                tS[None, None, None, (1 - nt % 2) * 2],
                                tCrKa[None, None, kb, st],
                                tCrQ[None, None, st * (DCH // 16) + kb, 0],
                                tS[None, None, None, (1 - nt % 2) * 2],
                            )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + NQK + nt)
            if nt >= 1:
                # V is the sole younger group after wait_group(1).
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                cute.arch.fence_view_async_shared()
                pb = (nt - 1) % 2
                if warp_id == 0:
                    for kb in cutlass.range_constexpr(KSUP // 16):
                        tmma_pv.set(tcgen05.Field.ACCUMULATE, not (nt == 1 and kb == 0))
                        cute.gemm(tmma_pv, tO0[None, None, None, 0], tCrV[None, None, kb, 0], tCrP[None, None, kb, 0], tO0[None, None, None, 0])
                        cute.gemm(tmma_pv, tO1[None, None, None, 0], tCrV[None, None, kb, 1], tCrP[None, None, kb, 0], tO1[None, None, None, 0])
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + 2 * NQK + nt - 1)
                else:
                    # Keep the MMA-issuing warp's scalar path short.  The
                    # next four warps reduce the completed teacher tile while
                    # warp 0 queues PV, overlapping independent issue pipes.
                    tr = tidx - 32
                    if tr < MT:
                        tea = sTea[pb, 0, tr]
                        for g in cutlass.range_constexpr(1, NHG):
                            tea = tea + sTea[pb, g, tr]
                        mTea[t, (nt - 1) * MT + tr] = tea * INV_H
            if nt >= RES and nt < NQK:
                cute.arch.mbarrier_wait(mbar + NQK + nt, ph)
                if nt + 1 < NQK:
                    nb = (nt + 1) * MT
                    _gather_k(cpa, mK8, sIc, sK8a, nb, tidx, NCP)
                    cute.arch.cp_async_commit_group()

            if nt >= 1:
                cute.arch.mbarrier_wait(mbar + 2 * NQK + nt - 1, ph)
                if nt < NQK:
                    nb = nt * KSUP
                    _gather_v(cpa, mV8, sIc, sV8, nb, tidx, NCP)
                    cute.arch.cp_async_commit_group()
            elif cutlass.const_expr(not EV0):
                _gather_v(cpa, mV8, sIc, sV8, I32(0), tidx, NCP)
                cute.arch.cp_async_commit_group()

            if nt < NQK:
                s = nt * MT + row
                tsum[0] = cutlass.Float32(0.0)
                tsum[1] = cutlass.Float32(0.0)
                for hsub in cutlass.range_constexpr(NSUB):
                    tv = cute.make_tensor(tptr + (pcol + h0g + hsub * 8), cute.make_layout((MT, 8), stride=(65536, 1)))
                    cute.copy(tld, thr_ld.partition_S(tv), frg)
                    if s < vlen:
                        _exp_norm8(frg, mp, hsub * 8, fl, ob, tsum)
                    else:
                        for j in cutlass.range_constexpr(8):
                            ob[j] = cutlass.BFloat16(0.0)
                    h = h0g + hsub * 8
                    pidx = ((h // P_GRP) * (P_GRP * KSUP) + row * P_GRP + (h % P_GRP)) // 8
                    cute.autovec_copy(ob, sPv[pidx, None])
                sTea[nt % 2, hg, row] = tsum[0] + tsum[1]
                cute.arch.fence_view_async_tmem_load()

        # Q and both K buffers are retired after the final QK/PV.  Issue the
        # next row's prologue first so these transfers overlap the TMEM output
        # drain instead of following it on the serial epilogue tail.
        if cutlass.const_expr(EV0):
            t2 = t + 2
            if t2 >= TQ:
                t2 = TQ - 1
            for i in cutlass.range_constexpr(NCI4):
                c = i * NT + tidx
                if c < TOPK // 4:
                    cute.copy(cpai, mIdx4[t2, c, None], sIdx4[cur, c, None])
        for i in cutlass.range_constexpr(NCQ):
            c = i * NT + tidx
            dc = c % 32
            cute.copy(cpa, mQ8[tn, c, None], sQ8[dc // 8, c // 32, dc % 8, None])
        if cutlass.const_expr(EV0):
            cute.arch.cp_async_commit_group()
        _gather_k(cpa, mK8, sIn, sK8a, I32((NQK - 1) * MT), tidx, NCP)
        cute.arch.cp_async_commit_group()
        _gather_k(cpa, mK8, sIn, sK8b, I32((NQK - 2) * MT), tidx, NCP)
        cute.arch.cp_async_commit_group()

        # ---------------- epilogue ----------------
        for hsub in cutlass.range_constexpr(NSUB):
            h = h0g + hsub * 8
            for ai in cutlass.range_constexpr(2):
                acol = STGV * HQ if ai == 0 else HQ
                tv = cute.make_tensor(tptr + (acol + h), cute.make_layout((MT, 8), stride=(65536, 1)))
                cute.copy(tld, thr_ld.partition_S(tv), frg)
                dv = ai * MT + row
                for j in cutlass.range_constexpr(8):
                    mO[t, h + j, dv] = frg[j].to(cutlass.BFloat16)
        cute.arch.fence_view_async_tmem_load()

        cute.arch.barrier()

    cute.arch.cp_async_wait_group(0)
    tmem.relinquish_alloc_permit()
    cute.arch.barrier()
    tmem.free(tptr)


@cute.jit
def _launch(
    rawQ8,
    rawK8,
    rawV8,
    rawIdx,
    rawVlen,
    rawO,
    rawLse,
    rawTea,
    rawSoftmaxMax,
    rawSoftmaxInvSum,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    TQ: I32,
    TK: I32,
    NBLK: I32,
    HQ: cutlass.Constexpr,
    TOPK: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    q_token_stride = cute.assume(q_token_stride, 8)
    k_token_stride = cute.assume(k_token_stride, 8)
    v_token_stride = cute.assume(v_token_stride, 8)

    mQ8 = cute.make_tensor(
        rawQ8.iterator,
        cute.make_layout((TQ, HQ * D // 8, 8), stride=(q_token_stride, 8, 1)),
    )
    mK8 = cute.make_tensor(
        rawK8.iterator,
        cute.make_layout((TK, D // 8, 8), stride=(k_token_stride, 8, 1)),
    )
    mV8 = cute.make_tensor(
        rawV8.iterator,
        cute.make_layout((TK, D // 8, 8), stride=(v_token_stride, 8, 1)),
    )
    mIdx = cute.make_tensor(rawIdx.iterator, cute.make_layout((TQ, TOPK), stride=(TOPK, 1)))
    mVlen = cute.make_tensor(rawVlen.iterator, cute.make_layout((TQ,), stride=(1,)))
    mO = cute.make_tensor(rawO.iterator, cute.make_layout((TQ, HQ, D), stride=(HQ * D, D, 1)))
    mLse = cute.make_tensor(rawLse.iterator, cute.make_layout((TQ, HQ), stride=(HQ, 1)))
    mTea = cute.make_tensor(rawTea.iterator, cute.make_layout((TQ, TOPK), stride=(TOPK, 1)))
    mSoftmaxMax = cute.make_tensor(rawSoftmaxMax.iterator, cute.make_layout((TQ, HQ), stride=(HQ, 1)))
    mSoftmaxInvSum = cute.make_tensor(rawSoftmaxInvSum.iterator, cute.make_layout((TQ, HQ), stride=(HQ, 1)))

    tmma_qk = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K, OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE, (MT, HQ)
    )
    tmma_pv = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.MN, OperandMajorMode.MN, cutlass.Float32, tcgen05.CtaGroup.ONE, (MT, HQ)
    )
    _ssa(mQ8, mK8, mV8, mIdx, mVlen, mO, mLse, mTea, mSoftmaxMax, mSoftmaxInvSum, tmma_qk, tmma_pv, TQ, NBLK, HQ, TOPK).launch(
        grid=[NBLK, 1, 1],
        block=[NT, 1, 1],
        smem=_smem_bytes(HQ, TOPK),
        stream=stream,
    )


_cache = {}


def _dynamic_tensor(tensor, alignment):
    return from_dlpack(tensor.detach(), assumed_align=alignment, enable_tvm_ffi=True).mark_layout_dynamic()


@torch.no_grad()
def run_group(q, k, v, selected_indices, valid_lengths, out, lse, teacher, softmax_max, softmax_inv_sum):
    tq, hq, _ = q.shape
    tk = k.shape[0]
    topk = selected_indices.shape[-1]

    q8 = q.as_strided((tq, hq * D // 8, 8), (q.stride(0), 8, 1))
    k8 = k.as_strided((tk, D // 8, 8), (k.stride(0), 8, 1))
    v8 = v.as_strided((tk, D // 8, 8), (v.stride(0), 8, 1))
    idx = selected_indices.view(tq, topk)
    vl = valid_lengths.view(tq)
    ov = out.view(tq, hq, D)
    ls = lse.view(tq, hq)
    te = teacher.view(tq, topk)
    mx = softmax_max.view(tq, hq)
    iv = softmax_inv_sum.view(tq, hq)

    nblk = min(torch.cuda.get_device_properties(q.device).multi_processor_count, tq)
    runtime_args = (q8, k8, v8, idx, vl, ov, ls, te, mx, iv)

    key = (q.device.index, hq, topk)
    compiled = _cache.get(key)
    if compiled is None:
        args = [_dynamic_tensor(tensor, alignment) for tensor, alignment in zip(runtime_args, (16, 16, 16, 16, 4, 16, 16, 16, 16, 16))]
        compiled = cute.compile(
            _launch,
            *args,
            I32(q8.stride(0)),
            I32(k8.stride(0)),
            I32(v8.stride(0)),
            I32(tq),
            I32(tk),
            I32(nblk),
            hq,
            topk,
            stream=make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _cache[key] = compiled

    compiled(
        *(t.detach() for t in runtime_args),
        I32(q8.stride(0)),
        I32(k8.stride(0)),
        I32(v8.stride(0)),
        I32(tq),
        I32(tk),
        I32(nblk),
    )
