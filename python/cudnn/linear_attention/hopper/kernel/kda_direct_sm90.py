# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi Delta Attention (KDA) chunked linear-attention prefill for Hopper sm90.

CuTe DSL, two-kernel schedule.

Per token t, per head (D = 128):
    alpha_t = exp(g_t) in (0,1]^D          (per-key-channel decay)
    S_t = (I - beta_t k_t^T k_t) Diag(alpha_t) S_{t-1} + beta_t k_t^T v_t
    o_t = q_t S_t                          (q pre-scaled by 1/sqrt(D))

Chunked (UT/WY) form, chunk BT = 64, cs_i = sum_{s<=i} g_s, mid-chunk anchor
r = cs_31 (keeps every exp argument inside +-40 so bf16 never overflows):

    Wn = -k*exp(cs-r)   U = beta*k*exp(r-cs)   Qt = q*scale*exp(cs-r)
    Sh = Diag(exp(r)) S                       (anchored state)
    M    = tril(-(Wn U^T), -1)                Tinv = (I+M)^{-1}
    wh   = Tinv Wn        uh = Tinv V
    R    = uh + wh Sh
    O    = Qt Sh + tril(Qt U^T, 0) R
    S   <- Diag(exp(cs63-r)) (Sh + U^T R)

Everything down to wh/uh depends only on the chunk's own (q,k,v,g,beta), so it
is hoisted into a fully chunk-parallel PREP kernel (grid = chunks x heads x
sequences).  The sequential SCAN kernel then carries the [128,128] state
through only three dependent stages per chunk instead of eight.

Tinv uses 16x16 block forward-substitution for the diagonal blocks (fp32, in
registers) plus a block-Neumann expansion:
    (I+M) = (I+Md)(I+G), G = Dinv Mo, G^4 = 0, (I+G)^-1 = (I-G)(I+G^2)
"""

import math

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

BF = cutlass.BFloat16
F32 = cutlass.Float32
I32 = cutlass.Int32

D = 128
BT = 64
NTHR = 256
NTOK = 8
NG = 8
DV_SINGLE = 16
DV_BATCHED = 64
LOG2E = 1.4426950408889634
PKS = NTHR * 8


def _smem_layout(dtype, shape, contig_mode, swizzle_bytes=128):
    if contig_mode == 1:
        kind = (
            warpgroup.SmemLayoutAtomKind.K_SW32
            if swizzle_bytes == 32
            else warpgroup.SmemLayoutAtomKind.K_SW64 if swizzle_bytes == 64 else warpgroup.SmemLayoutAtomKind.K_SW128
        )
        order = (0, 1)
    else:
        kind = (
            warpgroup.SmemLayoutAtomKind.MN_SW32
            if swizzle_bytes == 32
            else warpgroup.SmemLayoutAtomKind.MN_SW64 if swizzle_bytes == 64 else warpgroup.SmemLayoutAtomKind.MN_SW128
        )
        order = (1, 0)
    return cute.tile_to_shape(warpgroup.make_smem_layout_atom(kind, dtype), shape, order=order)


def _alloc(smem, dtype, lay):
    return smem.allocate_tensor(dtype, lay.outer, 1024, swizzle=lay.inner)


def _tview(a):
    return cute.composition(a, cute.make_ordered_layout((a.shape[1], a.shape[0]), order=(1, 0)))


def _mkatoms(tiled_mma):
    a0 = cute.make_mma_atom(tiled_mma.op)
    a0.set(warpgroup.Field.ACCUMULATE, False)
    a1 = cute.make_mma_atom(tiled_mma.op)
    a1.set(warpgroup.Field.ACCUMULATE, True)
    return a0, a1


def _gemm(atoms, acc, rA, rB, zero_init, fence=True, wait=0):
    a0, a1 = atoms
    if fence:
        warpgroup.fence()
    for k in range(cute.size(rA.shape[2])):
        cute.gemm(a0 if (k == 0 and zero_init) else a1, acc, rA[None, None, k], rB[None, None, k], acc)
    warpgroup.commit_group()
    if wait >= 0:
        warpgroup.wait_group(wait)


def _fence_bar():
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()


def _mma_ab():
    MK = cute.nvgpu.OperandMajorMode.K
    MM = cute.nvgpu.OperandMajorMode.MN
    return (
        sm90_utils.make_trivial_tiled_mma(BF, BF, MK, MK, F32, (1, 2, 1), tiler_mn=(64, 32)),
        sm90_utils.make_trivial_tiled_mma(BF, BF, MK, MM, F32, (1, 2, 1), tiler_mn=(64, 64)),
        sm90_utils.make_trivial_tiled_mma(BF, BF, MK, MM, F32, (1, 2, 1), tiler_mn=(64, 32)),
        sm90_utils.make_trivial_tiled_mma(BF, BF, MM, MM, F32, (2, 1, 1), tiler_mn=(64, 128)),
    )


def _mma_scan(dv):
    """One-WG token GEMMs plus a two-WG M-split state update for a DV slab."""
    MK = cute.nvgpu.OperandMajorMode.K
    MM = cute.nvgpu.OperandMajorMode.MN
    return (
        sm90_utils.make_trivial_tiled_mma(BF, BF, MK, MK, F32, (1, 2, 1), tiler_mn=(64, 32)),
        sm90_utils.make_trivial_tiled_mma(BF, BF, MK, MM, F32, (1, 1, 1), tiler_mn=(64, dv)),
        sm90_utils.make_trivial_tiled_mma(BF, BF, MM, MM, F32, (2, 1, 1), tiler_mn=(64, dv)),
    )


@cute.jit
def _fetch3(atomG, lay, tg, mWh, mQt, mUt, toff, nr, vw, vq, vu):
    """Stage one chunk of the prep-kernel outputs into registers.

    Rows past the sequence end are zero-filled: the scan's state update
    contracts U^T R over all 64 rows, so a stale U row would leak the next
    packed sequence into this sequence's state."""
    gW = cute.make_tensor((mWh.iterator + toff).align(8), lay)
    gQ = cute.make_tensor((mQt.iterator + toff).align(8), lay)
    gU = cute.make_tensor((mUt.iterator + toff).align(8), lay)
    if nr == BT:
        for i in cutlass.range_constexpr(NTOK):
            cute.copy(atomG, gW[(None, i)], vw[(None, i)])
            cute.copy(atomG, gQ[(None, i)], vq[(None, i)])
            cute.copy(atomG, gU[(None, i)], vu[(None, i)])
    else:
        for i in cutlass.range_constexpr(NTOK):
            if (tg * NTOK + i) < nr:
                cute.copy(atomG, gW[(None, i)], vw[(None, i)])
                cute.copy(atomG, gQ[(None, i)], vq[(None, i)])
                cute.copy(atomG, gU[(None, i)], vu[(None, i)])
            else:
                for c in cutlass.range_constexpr(4):
                    vw[c, i] = BF(0.0)
                    vq[c, i] = BF(0.0)
                    vu[c, i] = BF(0.0)


@cute.jit
def _fetchv(mVec, nseq, gc, h, r0, r1, fv):
    """Gather the four decay scalars needed by this thread's state rows."""
    fv[0] = mVec[nseq, gc, h, 0, r0]
    fv[1] = mVec[nseq, gc, h, 0, r1]
    fv[2] = mVec[nseq, gc, h, 1, r0]
    fv[3] = mVec[nseq, gc, h, 1, r1]


@cute.jit
def _fetchl(atom128, mUl, mPl, uoff, poff, vh, vp):
    """Stage the wgmma-C-layout uh seed and the packed P tile (fully coalesced)."""
    gU = cute.make_tensor((mUl.iterator + uoff).align(16), cute.make_layout((8, 4), stride=(1, PKS)))
    gP = cute.make_tensor((mPl.iterator + poff).align(16), cute.make_layout((8, 2), stride=(1, PKS)))
    for j in cutlass.range_constexpr(4):
        cute.copy(atom128, gU[(None, j)], vh[(None, j)])
    for j in cutlass.range_constexpr(2):
        cute.copy(atom128, gP[(None, j)], vp[(None, j)])


@cute.jit
def _fetchl_scan16(atom128, mUl, mPl, uoff, poff, tidx, vh, vp):
    """Load one DV=16 quarter-fragment of uh and the unchanged two-WG P fragment."""
    if tidx < 128:
        gU = cute.make_tensor((mUl.iterator + uoff).align(16), cute.make_layout((8, 1), stride=(1, PKS)))
        cute.copy(atom128, gU[(None, 0)], vh[(None, 0)])
    gP = cute.make_tensor((mPl.iterator + poff).align(16), cute.make_layout((8, 2), stride=(1, PKS)))
    for j in cutlass.range_constexpr(2):
        cute.copy(atom128, gP[(None, j)], vp[(None, j)])


@cute.jit
def _fetchl_scan32(atom128, mUl, mPl, uoff, poff, tidx, vh, vp):
    """Load one DV=32 half-fragment of uh and the unchanged two-WG P fragment."""
    if tidx < 128:
        gU = cute.make_tensor((mUl.iterator + uoff).align(16), cute.make_layout((8, 2), stride=(1, PKS)))
        for j in cutlass.range_constexpr(2):
            cute.copy(atom128, gU[(None, j)], vh[(None, j)])
    gP = cute.make_tensor((mPl.iterator + poff).align(16), cute.make_layout((8, 2), stride=(1, PKS)))
    for j in cutlass.range_constexpr(2):
        cute.copy(atom128, gP[(None, j)], vp[(None, j)])


@cute.jit
def _fetchl_scan64(atom128, mUl, mPl, uoff, poff, tidx, vh, vp):
    """Load one DV=64 uh fragment and the unchanged two-WG P fragment."""
    if tidx < 128:
        gU = cute.make_tensor((mUl.iterator + uoff).align(16), cute.make_layout((8, 4), stride=(1, PKS)))
        for j in cutlass.range_constexpr(4):
            cute.copy(atom128, gU[(None, j)], vh[(None, j)])
    gP = cute.make_tensor((mPl.iterator + poff).align(16), cute.make_layout((8, 2), stride=(1, PKS)))
    for j in cutlass.range_constexpr(2):
        cute.copy(atom128, gP[(None, j)], vp[(None, j)])


def _elemwise_copies(atomG):
    cpKQ = cute.make_tiled_copy_tv(atomG, cute.make_ordered_layout((NG, 32), order=(1, 0)), cute.make_ordered_layout((NTOK, 4), order=(1, 0)))
    cpT = cute.make_tiled_copy_tv(atomG, cute.make_ordered_layout((32, NG), order=(0, 1)), cute.make_ordered_layout((4, NTOK), order=(0, 1)))
    return cpKQ, cpT


def _output_copy(atomG, dv, ntok_o):
    return cute.make_tiled_copy_tv(atomG, cute.make_ordered_layout((BT // ntok_o, dv // 4), order=(1, 0)), cute.make_ordered_layout((ntok_o, 4), order=(1, 0)))


# ===========================================================================
#  PREP: one CTA per (chunk, head, sequence) -- fully parallel
# ===========================================================================
class KDAPrep:
    def __init__(self, H, N):
        self.H = H
        self.N = N
        self.strt = H * D

    @cute.jit
    def __call__(self, mQ, mK, mV, mG, mBeta, mCu, mWh, mUl, mUt, mQt, mPl, mVec, stream: cuda.CUstream):
        mmaA, mmaB, mmaC, _ = _mma_ab()
        lW = _smem_layout(BF, (BT, D), 1)
        lT = _smem_layout(BF, (D, BT), 0)
        l64k = _smem_layout(BF, (64, 64), 1)
        l64m = _smem_layout(BF, (64, 64), 0)
        self.kernel(mQ, mK, mV, mG, mBeta, mCu, mWh, mUl, mUt, mQt, mPl, mVec, mmaA, mmaB, mmaC, lW, lT, l64k, l64m).launch(
            # Sum_n ceil(seq_len[n]/BT) <= ceil(T/BT)+N-1.  Flatten useful
            # chunks so packed batches do not launch an N-fold grid of no-ops.
            grid=(cute.ceil_div(mQ.shape[0], BT) + self.N - 1, self.H, 1),
            block=[NTHR, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mG: cute.Tensor,
        mBeta: cute.Tensor,
        mCu: cute.Tensor,
        mWh: cute.Tensor,
        mUl: cute.Tensor,
        mUt: cute.Tensor,
        mQt: cute.Tensor,
        mPl: cute.Tensor,
        mVec: cute.Tensor,
        mmaA: cute.TiledMma,
        mmaB: cute.TiledMma,
        mmaC: cute.TiledMma,
        lW: cute.ComposedLayout,
        lT: cute.ComposedLayout,
        l64k: cute.ComposedLayout,
        l64m: cute.ComposedLayout,
    ):
        strt = self.strt
        tidx, _, _ = cute.arch.thread_idx()
        gcx, h, _ = cute.arch.block_idx()

        nseq = I32(0)
        cx = gcx
        chunk_prefix = I32(0)
        for ns in cutlass.range_constexpr(self.N):
            slen = mCu[ns + 1] - mCu[ns]
            ncs = (slen + (BT - 1)) // BT
            if gcx >= chunk_prefix:
                nseq = I32(ns)
                cx = gcx - chunk_prefix
            chunk_prefix = chunk_prefix + ncs

        seq_beg = mCu[nseq]
        seq_end = mCu[nseq + 1]
        t0 = seq_beg + cx * BT
        if gcx < chunk_prefix:
            nrows = seq_end - t0
            if nrows > BT:
                nrows = I32(BT)

            wg = tidx // 128
            ct = tidx % 32
            tg = tidx // 32
            mbase = ((tidx % 128) // 32) * 16 + (tidx % 32) // 4
            nbase = 2 * (tidx % 4)
            scale = cutlass.Float32(1.0 / math.sqrt(D))

            smem = utils.SmemAllocator()
            sWn = _alloc(smem, BF, lW)
            sU = _alloc(smem, BF, lW)
            sQh = _alloc(smem, BF, lW)
            sMo = _alloc(smem, BF, l64m)
            sGk = _alloc(smem, BF, l64k)
            sGn = _alloc(smem, BF, l64m)
            sGm = _alloc(smem, BF, l64k)
            # These lifetimes are disjoint.  Reuse the dead Neumann tiles and
            # re-view Wn/U storage as the closing MN-major wh/uh operands.
            # Together with Qt/Dinv reuse below this cuts PREP to ~90 KiB,
            # allowing two CTAs/SM once register use is also below 128.
            sG2 = sMo
            # Qt is dead after P = Qt U^T; its 16 KiB tile then holds the two
            # 8 KiB diagonal-inverse views used by the Neumann chain.
            sDk = cute.make_tensor(cute.recast_ptr(sQh.iterator, l64k.inner, dtype=BF), l64k.outer)
            sDn = cute.make_tensor(cute.recast_ptr(sQh.iterator + 64 * 64, l64m.inner, dtype=BF), l64m.outer)
            sZk = sDk
            sTi = sGk
            sWt = cute.make_tensor(cute.recast_ptr(sWn.iterator, lT.inner, dtype=BF), lT.outer)
            sVt = cute.make_tensor(cute.recast_ptr(sU.iterator, lT.inner, dtype=BF), lT.outer)
            sMd = smem.allocate_tensor(F32, cute.make_layout((4, 16, 16), stride=(256, 16, 1)), 16)
            sEye = smem.allocate_tensor(F32, cute.make_layout((16, 16), stride=(16, 1)), 16)
            sScan = smem.allocate_tensor(F32, cute.make_layout((NG, D), stride=(D, 1)), 16)
            sBeta = smem.allocate_tensor(F32, cute.make_layout(BT), 16)

            sEye[tidx // 16, tidx % 16] = F32(0.0)
            cute.arch.sync_threads()
            if tidx < 16:
                sEye[tidx, tidx] = F32(1.0)

            atomG = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF, num_bits_per_copy=64)
            atomL32 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), F32, num_bits_per_copy=128)
            atom128 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF, num_bits_per_copy=128)
            cpKQ, cpT = _elemwise_copies(atomG)
            tk = cpKQ.get_slice(tidx)
            tt = cpT.get_slice(tidx)
            tWs = tk.partition_D(sWn)
            tUs = tk.partition_D(sU)
            tQs = tk.partition_D(sQh)
            tWts = tt.partition_D(sWt)
            tVts = tt.partition_D(sVt)
            tRdW = tk.partition_S(sWn)

            _ar = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF, num_bits_per_copy=32)
            tcA = cute.make_tiled_copy_C(_ar, mmaA)
            tcB = cute.make_tiled_copy_C(_ar, mmaB)
            tcC = cute.make_tiled_copy_C(_ar, mmaC)
            rA = tcA.get_slice(tidx)
            rB = tcB.get_slice(tidx)
            rC = tcC.get_slice(tidx)
            dMo = rA.partition_D(_tview(sMo))
            dGk = rC.partition_D(sGk)
            dGn = rC.partition_D(_tview(sGn))
            dGm = rC.partition_D(sGm)
            dG2 = rC.partition_D(_tview(sG2))
            dZk = rC.partition_D(sZk)
            dTi = rC.partition_D(sTi)
            dOw = rB.partition_D(sWn)

            thrA = mmaA.get_slice(wg * 128)
            thrB = mmaB.get_slice(wg * 128)
            thrC = mmaC.get_slice(wg * 128)
            atA = _mkatoms(mmaA)
            atB = _mkatoms(mmaB)
            atC = _mkatoms(mmaC)
            shpA = thrA.partition_C(cute.make_identity_tensor((64, 64))).shape
            shpB = thrB.partition_C(cute.make_identity_tensor((64, D))).shape
            nA = cute.size(shpA)

            rA_W = mmaA.make_fragment_A(thrA.partition_A(sWn))
            rB_U = mmaA.make_fragment_B(thrA.partition_B(sU))
            rA_Qp = mmaA.make_fragment_A(thrA.partition_A(sQh))
            rA_Dk = mmaC.make_fragment_A(thrC.partition_A(sDk))
            rB_Mo = mmaC.make_fragment_B(thrC.partition_B(sMo))
            rA_Gk = mmaC.make_fragment_A(thrC.partition_A(sGk))
            rB_Gn = mmaC.make_fragment_B(thrC.partition_B(sGn))
            rA_Gm = mmaC.make_fragment_A(thrC.partition_A(sGm))
            rB_G2 = mmaC.make_fragment_B(thrC.partition_B(sG2))
            rA_Zk = mmaC.make_fragment_A(thrC.partition_A(sZk))
            rB_Dn = mmaC.make_fragment_B(thrC.partition_B(sDn))
            rA_Ti = mmaB.make_fragment_A(thrB.partition_A(sTi))
            rB_Wt = mmaB.make_fragment_B(thrB.partition_B(sWt))
            rB_Vt = mmaB.make_fragment_B(thrB.partition_B(sVt))

            mb0 = mbase
            mb1 = mbase + 8
            jb = wg * 32 + nbase
            sb = tidx // 16
            sc = tidx % 16

            if tidx < BT:
                bv = F32(0.0)
                if tidx < nrows:
                    bv = mBeta[t0 + tidx, h]
                sBeta[tidx] = bv

            eoff = t0 * strt + h * D + tg * NTOK * strt + ct * 4
            lay = cute.make_layout((4, NTOK), stride=(1, strt))
            tGg = cute.make_tensor((mG.iterator + eoff).align(16), lay)
            tKg = cute.make_tensor((mK.iterator + eoff).align(8), lay)
            tQg = cute.make_tensor((mQ.iterator + eoff).align(8), lay)
            tVg = cute.make_tensor((mV.iterator + eoff).align(8), lay)
            # Keep q/v out of the peak live set.  q is reloaded after Wn/U
            # retire; v is loaded only before the closing uh GEMM.
            fg = cute.make_rmem_tensor((4, NTOK), F32)
            fk = cute.make_rmem_tensor((4, NTOK), BF)
            if nrows == BT:
                for i in cutlass.range_constexpr(NTOK):
                    cute.copy(atomL32, tGg[(None, i)], fg[(None, i)])
                    cute.copy(atomG, tKg[(None, i)], fk[(None, i)])
            else:
                for i in cutlass.range_constexpr(NTOK):
                    if (tg * NTOK + i) < nrows:
                        cute.copy(atomL32, tGg[(None, i)], fg[(None, i)])
                        cute.copy(atomG, tKg[(None, i)], fk[(None, i)])
                    else:
                        for c in cutlass.range_constexpr(4):
                            fg[c, i] = F32(0.0)
                            fk[c, i] = BF(0.0)

            for c in cutlass.range_constexpr(4):
                acc = F32(0.0)
                for i in cutlass.range_constexpr(NTOK):
                    acc = acc + fg[c, i] * LOG2E
                    fg[c, i] = acc
                sScan[tg, ct * 4 + c] = acc
            cute.arch.sync_threads()

            if tidx < D:
                pv = cute.make_rmem_tensor(NG, F32)
                for gi in cutlass.range_constexpr(NG):
                    pv[gi] = sScan[gi, tidx]
                p = F32(0.0)
                rr = F32(0.0)
                for gi in cutlass.range_constexpr(NG):
                    tv = pv[gi]
                    pv[gi] = p
                    p = p + tv
                    if gi == (NG // 2 - 1):
                        rr = p
                for gi in cutlass.range_constexpr(NG):
                    sScan[gi, tidx] = pv[gi] - rr
                mVec[nseq, cx, h, 0, tidx] = cute.math.exp2(rr, fastmath=True)
                mVec[nseq, cx, h, 1, tidx] = cute.math.exp2(p - rr, fastmath=True)
            cute.arch.sync_threads()
            dlt = cute.make_rmem_tensor(4, F32)
            cute.copy(atomL32, cute.make_tensor((sScan.iterator + (tg * D + ct * 4)).align(16), cute.make_layout(4, stride=1)), dlt)

            frgW = cute.make_rmem_tensor(tWs.shape, BF)
            frgU = cute.make_rmem_tensor(tWs.shape, BF)
            for i in cutlass.range_constexpr(NTOK):
                bt = sBeta[tg * NTOK + i]
                for c in cutlass.range_constexpr(4):
                    x = dlt[c] + fg[c, i]
                    e = cute.math.exp2(x, fastmath=True)
                    f = cute.math.exp2(-x, fastmath=True)
                    kk = fk[c, i].to(F32)
                    frgW[c + 4 * i] = BF(-kk * e)
                    frgU[c + 4 * i] = BF(bt * kk * f)
            cute.arch.sync_threads()
            cute.copy(atomG, frgW, tWs)
            cute.copy(atomG, frgU, tUs)
            # U streams straight out; q/Qt are materialized only after the
            # Wn/U staging fragments above have retired.
            oQ = cute.make_tensor((mQt.iterator + eoff).align(8), lay)
            oU = cute.make_tensor((mUt.iterator + eoff).align(8), lay)
            fU = cute.make_tensor(frgU.iterator, cute.make_layout((4, NTOK), stride=(1, 4)))
            for i in cutlass.range_constexpr(NTOK):
                if (tg * NTOK + i) < nrows:
                    cute.copy(atomG, fU[(None, i)], oU[(None, i)])

            fq = cute.make_rmem_tensor((4, NTOK), BF)
            frgQ = cute.make_rmem_tensor(tWs.shape, BF)
            fQ = cute.make_tensor(frgQ.iterator, cute.make_layout((4, NTOK), stride=(1, 4)))
            if nrows == BT:
                for i in cutlass.range_constexpr(NTOK):
                    cute.copy(atomG, tQg[(None, i)], fq[(None, i)])
            else:
                for i in cutlass.range_constexpr(NTOK):
                    if (tg * NTOK + i) < nrows:
                        cute.copy(atomG, tQg[(None, i)], fq[(None, i)])
                    else:
                        for c in cutlass.range_constexpr(4):
                            fq[c, i] = BF(0.0)
            for i in cutlass.range_constexpr(NTOK):
                for c in cutlass.range_constexpr(4):
                    e = cute.math.exp2(dlt[c] + fg[c, i], fastmath=True)
                    frgQ[c + 4 * i] = BF(fq[c, i].to(F32) * (scale * e))
            cute.copy(atomG, frgQ, tQs)
            for i in cutlass.range_constexpr(NTOK):
                if (tg * NTOK + i) < nrows:
                    cute.copy(atomG, fQ[(None, i)], oQ[(None, i)])
            plo = nseq * mPl.stride[0] + cx * mPl.stride[1] + h * mPl.stride[2] + tidx * 8
            ulo = nseq * mUl.stride[0] + cx * mUl.stride[1] + h * mUl.stride[2] + tidx * 8
            _fence_bar()

            frgV = cute.make_rmem_tensor(tVts.shape, BF)
            fv = cute.make_tensor(frgV.iterator, cute.make_layout((4, NTOK), stride=(1, 4)))
            if nrows == BT:
                for i in cutlass.range_constexpr(NTOK):
                    cute.copy(atomG, tVg[(None, i)], fv[(None, i)])
            else:
                for i in cutlass.range_constexpr(NTOK):
                    if (tg * NTOK + i) < nrows:
                        cute.copy(atomG, tVg[(None, i)], fv[(None, i)])
                    else:
                        for c in cutlass.range_constexpr(4):
                            fv[c, i] = BF(0.0)

            # ---- M = -(Wn U^T) ----
            accM = cute.make_rmem_tensor(shpA, F32)
            _gemm(atA, accM, rA_W, rB_U, True, wait=0)
            frgMo = cute.make_rmem_tensor(shpA, BF)
            for v in cutlass.range_constexpr(nA):
                ii = mb1 if ((v // 2) % 2) else mb0
                jj = jb + ((v % 2) + 8 * (v // 4))
                bi = ii // 16
                bj = jj // 16
                mv = -accM[v]
                frgMo[v] = BF(0.0)
                if bi > bj:
                    frgMo[v] = BF(mv)
                if bi == bj:
                    dvv = F32(0.0)
                    if ii > jj:
                        dvv = mv
                    sMd[bi, ii % 16, jj % 16] = dvv
            cute.copy(tcA, tcA.retile(frgMo), dMo)

            # Serialize P after M so their two fp32 accumulator fragments do
            # not overlap at the register high-water mark.
            accP = cute.make_rmem_tensor(shpA, F32)
            _gemm(atA, accP, rA_Qp, rB_U, True, wait=0)
            frgP = cute.make_rmem_tensor(shpA, BF)
            for v in cutlass.range_constexpr(nA):
                ii = mb1 if ((v // 2) % 2) else mb0
                jj = jb + ((v % 2) + 8 * (v // 4))
                frgP[v] = BF(0.0)
                if ii >= jj:
                    frgP[v] = BF(accP[v])
            gPl = cute.make_tensor((mPl.iterator + plo).align(16), cute.make_layout((8, 2), stride=(1, PKS)))
            vPl = cute.make_tensor(frgP.iterator, cute.make_layout((8, 2), stride=(1, 8)))
            for j in cutlass.range_constexpr(2):
                cute.copy(atom128, vPl[(None, j)], gPl[(None, j)])
            # M/P drained Wn/U; publish the aliased closing operands now.
            cute.arch.sync_threads()
            cute.copy(atomG, frgW, tWts)
            cute.copy(atomG, frgV, tVts)
            for j in cutlass.range_constexpr(16):
                sDk[tidx // 4, (tidx % 4) * 16 + j] = BF(0.0)
                sDn[(tidx % 4) * 16 + j, tidx // 4] = BF(0.0)
            _fence_bar()

            if tidx < 64:
                inv = cute.make_rmem_tensor(16, F32)
                for i in cutlass.range_constexpr(16):
                    s = sEye[i, sc]
                    for j in cutlass.range_constexpr(i):
                        s = s - sMd[sb, i, j] * inv[j]
                    inv[i] = s
                for i in cutlass.range_constexpr(16):
                    bfv = inv[i].to(BF)
                    sDk[sb * 16 + i, sb * 16 + sc] = bfv
                    sDn[sb * 16 + sc, sb * 16 + i] = bfv
            _fence_bar()

            # ---- G = Dinv Mo ----
            accG = cute.make_rmem_tensor(shpA, F32)
            _gemm(atC, accG, rA_Dk, rB_Mo, True, wait=0)
            frgGk = cute.make_rmem_tensor(shpA, BF)
            frgGm = cute.make_rmem_tensor(shpA, BF)
            frgGk.store(accG.load().to(BF))
            for v in cutlass.range_constexpr(nA):
                ii = mb1 if ((v // 2) % 2) else mb0
                jj = jb + ((v % 2) + 8 * (v // 4))
                mval = -accG[v]
                if ii == jj:
                    mval = F32(1.0)
                frgGm[v] = mval.to(BF)
            rfGk = tcC.retile(frgGk)
            cute.copy(tcC, rfGk, dGk)
            cute.copy(tcC, rfGk, dGn)
            cute.copy(tcC, tcC.retile(frgGm), dGm)
            _fence_bar()

            # ---- G2 = G G ----
            accG2 = cute.make_rmem_tensor(shpA, F32)
            _gemm(atC, accG2, rA_Gk, rB_Gn, True, wait=0)
            frgG2 = cute.make_rmem_tensor(shpA, BF)
            for v in cutlass.range_constexpr(nA):
                ii = mb1 if ((v // 2) % 2) else mb0
                jj = jb + ((v % 2) + 8 * (v // 4))
                gv = accG2[v]
                if ii == jj:
                    gv = F32(1.0)
                frgG2[v] = gv.to(BF)
            cute.copy(tcC, tcC.retile(frgG2), dG2)
            _fence_bar()

            # ---- Z = (I-G)(I+G2) ----
            accZ = cute.make_rmem_tensor(shpA, F32)
            _gemm(atC, accZ, rA_Gm, rB_G2, True, wait=0)
            frgZ = cute.make_rmem_tensor(shpA, BF)
            frgZ.store(accZ.load().to(BF))
            cute.copy(tcC, tcC.retile(frgZ), dZk)
            _fence_bar()

            # ---- Tinv = Z Dinv ----
            accT = cute.make_rmem_tensor(shpA, F32)
            _gemm(atC, accT, rA_Zk, rB_Dn, True, wait=0)
            frgT = cute.make_rmem_tensor(shpA, BF)
            frgT.store(accT.load().to(BF))
            cute.copy(tcC, tcC.retile(frgT), dTi)
            _fence_bar()

            # ---- wh = Tinv Wn ; uh = Tinv V  (staged through smem for coalescing)
            accW = cute.make_rmem_tensor(shpB, F32)
            accU = cute.make_rmem_tensor(shpB, F32)
            _gemm(atB, accW, rA_Ti, rB_Wt, True, wait=-1)
            _gemm(atB, accU, rA_Ti, rB_Vt, True, fence=False, wait=0)
            # Drain uh before materializing wh so only one bf16 epilogue
            # fragment contributes to the register high-water mark.
            fU2 = cute.make_rmem_tensor(shpB, BF)
            fU2.store(accU.load().to(BF))
            # uh stays in the wgmma C layout: the scan seeds its R accumulator
            # straight from these registers, so no I@uh gemm is needed there.
            gUl = cute.make_tensor((mUl.iterator + ulo).align(16), cute.make_layout((8, 4), stride=(1, PKS)))
            vUl = cute.make_tensor(fU2.iterator, cute.make_layout((8, 4), stride=(1, 8)))
            for j in cutlass.range_constexpr(4):
                cute.copy(atom128, vUl[(None, j)], gUl[(None, j)])
            fW2 = cute.make_rmem_tensor(shpB, BF)
            fW2.store(accW.load().to(BF))
            cute.arch.sync_threads()
            cute.copy(tcB, tcB.retile(fW2), dOw)
            cute.arch.sync_threads()
            oW = cute.make_tensor((mWh.iterator + eoff).align(8), lay)
            rw = cute.make_rmem_tensor(tRdW.shape, BF)
            cute.copy(atomG, tRdW, rw)
            vrw = cute.make_tensor(rw.iterator, cute.make_layout((4, NTOK), stride=(1, 4)))
            for i in cutlass.range_constexpr(NTOK):
                if (tg * NTOK + i) < nrows:
                    cute.copy(atomG, vrw[(None, i)], oW[(None, i)])


# ===========================================================================
#  SCAN: DV=16 for N=1, DV=64 for batched varlen; sequential over chunks
# ===========================================================================
class KDAScan:
    def __init__(self, H, N):
        self.H = H
        self.N = N
        self.strt = H * D
        self.dv = DV_SINGLE if N == 1 else DV_BATCHED
        self.nv = D // self.dv
        self.ntok_o = self.dv // 16
        self.swizzle_bytes = 32 if self.dv == 16 else 64 if self.dv == 32 else 128

    @cute.jit
    def __call__(self, mWh, mUl, mUt, mQt, mPl, mVec, mCu, mO, mS, stream: cuda.CUstream):
        mmaA, mmaB, mmaS = _mma_scan(self.dv)
        lW = _smem_layout(BF, (BT, D), 1)
        lT = _smem_layout(BF, (D, BT), 0)
        lSt = _smem_layout(BF, (self.dv, D), 0, swizzle_bytes=self.swizzle_bytes)
        lR = _smem_layout(BF, (self.dv, BT), 0, swizzle_bytes=self.swizzle_bytes)
        lO = _smem_layout(BF, (BT, self.dv), 1, swizzle_bytes=self.swizzle_bytes)
        l64k = _smem_layout(BF, (64, 64), 1)
        self.kernel(mWh, mUl, mUt, mQt, mPl, mVec, mCu, mO, mS, mmaA, mmaB, mmaS, lW, lT, lSt, lR, lO, l64k).launch(
            grid=(self.H * self.nv, self.N, 1), block=[NTHR, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mWh: cute.Tensor,
        mUl: cute.Tensor,
        mUt: cute.Tensor,
        mQt: cute.Tensor,
        mPl: cute.Tensor,
        mVec: cute.Tensor,
        mCu: cute.Tensor,
        mO: cute.Tensor,
        mS: cute.Tensor,
        mmaA: cute.TiledMma,
        mmaB: cute.TiledMma,
        mmaS: cute.TiledMma,
        lW: cute.ComposedLayout,
        lT: cute.ComposedLayout,
        lSt: cute.ComposedLayout,
        lR: cute.ComposedLayout,
        lO: cute.ComposedLayout,
        l64k: cute.ComposedLayout,
    ):
        strt = self.strt
        tidx, _, _ = cute.arch.thread_idx()
        hv, nseq, _ = cute.arch.block_idx()
        h = hv // self.nv
        vb = hv % self.nv
        wg = tidx // 128
        ct = tidx % 32
        tg = tidx // 32
        mbase = ((tidx % 128) // 32) * 16 + (tidx % 32) // 4
        nbase = 2 * (tidx % 4)

        smem = utils.SmemAllocator()
        sWh = _alloc(smem, BF, lW)  # (tok,dk) K-major
        sQt = _alloc(smem, BF, lW)  # (tok,dk) K-major
        sUt = _alloc(smem, BF, lT)  # (dk,tok) MN-major A of U^T R
        sSt = _alloc(smem, BF, lSt)  # (DV,dk) MN-major B
        sR = _alloc(smem, BF, lR)  # (DV,tok) MN-major B
        sO = _alloc(smem, BF, lO)  # (tok,DV) coalesced output staging
        sP = _alloc(smem, BF, l64k)  # (tok,tok) K-major A

        atomG = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF, num_bits_per_copy=64)
        atom128 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF, num_bits_per_copy=128)
        cpKQ, cpT = _elemwise_copies(atomG)
        cpO = _output_copy(atomG, self.dv, self.ntok_o)
        tk = cpKQ.get_slice(tidx)
        tt = cpT.get_slice(tidx)
        to = cpO.get_slice(tidx)
        tWs = tk.partition_D(sWh)
        tQs = tk.partition_D(sQt)
        tUts = tt.partition_D(sUt)
        tOs = to.partition_S(sO)

        _ar = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF, num_bits_per_copy=32)
        tcA = cute.make_tiled_copy_C(_ar, mmaA)
        tcB = cute.make_tiled_copy_C(_ar, mmaB)
        tcS = cute.make_tiled_copy_C(_ar, mmaS)
        rAs = tcA.get_slice(tidx)
        rBs = tcB.get_slice(tidx % 128)
        dR = rBs.partition_D(_tview(sR))
        dO = rBs.partition_D(sO)
        dP = rAs.partition_D(sP)
        if cutlass.const_expr(self.dv == 64):
            rSs = tcS.get_slice(tidx)
            dSt = rSs.partition_D(_tview(sSt))

        thrA = mmaA.get_slice(wg * 128)
        thrB = mmaB.get_slice(0)
        thrS = mmaS.get_slice(wg * 128)
        atA = _mkatoms(mmaA)
        atB = _mkatoms(mmaB)
        atS = _mkatoms(mmaS)
        cA = thrA.partition_C(cute.make_identity_tensor((64, 64)))
        cB = thrB.partition_C(cute.make_identity_tensor((64, self.dv)))
        cS = thrS.partition_C(cute.make_identity_tensor((D, self.dv)))
        shpA = cA.shape
        shpB = cB.shape
        shpS = cS.shape
        nA = cute.size(shpA)
        nB = cute.size(shpB)
        nS = cute.size(shpS)
        accSt = cute.make_rmem_tensor(shpS, F32)
        accSt.fill(0.0)

        rA_Wh = mmaB.make_fragment_A(thrB.partition_A(sWh))
        rB_St = mmaB.make_fragment_B(thrB.partition_B(sSt))
        rA_Qt = mmaB.make_fragment_A(thrB.partition_A(sQt))
        rA_P = mmaB.make_fragment_A(thrB.partition_A(sP))
        rB_R = mmaB.make_fragment_B(thrB.partition_B(sR))
        rA_Ut = mmaS.make_fragment_A(thrS.partition_A(sUt))
        rB_RS = mmaS.make_fragment_B(thrS.partition_B(sR))

        mb0 = cA[0][0] + mbase
        mb1 = mb0 + 8
        jb = cA[0][1] + nbase
        om0 = cB[0][0] + mbase
        om1 = om0 + 8
        on0 = cB[0][1] + nbase
        srow0 = cS[0][0] + mbase
        srow1 = srow0 + 8
        ncol0 = cS[0][1] + nbase
        pend0 = F32(1.0)
        pend1 = F32(1.0)

        seq_beg = mCu[nseq]
        seq_end = mCu[nseq + 1]
        nchunk = (seq_end - seq_beg + (BT - 1)) // BT
        eb = h * D + tg * NTOK * strt + ct * 4
        lay = cute.make_layout((4, NTOK), stride=(1, strt))
        cto = tidx % (self.dv // 4)
        tgo = tidx // (self.dv // 4)
        eout = h * D + vb * self.dv + tgo * self.ntok_o * strt + cto * 4
        layO = cute.make_layout((4, self.ntok_o), stride=(1, strt))

        # register-resident prefetch buffers: chunk ic+1's operands are issued
        # right after chunk ic's operands land in smem, so the ~600-cycle global
        # load latency overlaps the chunk's own matmuls instead of stalling it.
        fw = cute.make_rmem_tensor(tWs.shape, BF)
        fq = cute.make_rmem_tensor(tWs.shape, BF)
        fu = cute.make_rmem_tensor(tUts.shape, BF)
        fh = cute.make_rmem_tensor(shpB, BF)
        fp = cute.make_rmem_tensor(shpA, BF)
        fv = cute.make_rmem_tensor(4, F32)
        vw = cute.make_tensor(fw.iterator, cute.make_layout((4, NTOK), stride=(1, 4)))
        vq = cute.make_tensor(fq.iterator, cute.make_layout((4, NTOK), stride=(1, 4)))
        vu = cute.make_tensor(fu.iterator, cute.make_layout((4, NTOK), stride=(1, 4)))
        if cutlass.const_expr(self.dv == 16):
            vh = cute.make_tensor(fh.iterator, cute.make_layout((8, 1), stride=(1, 8)))
        elif cutlass.const_expr(self.dv == 32):
            vh = cute.make_tensor(fh.iterator, cute.make_layout((8, 2), stride=(1, 8)))
        else:
            vh = cute.make_tensor(fh.iterator, cute.make_layout((8, 4), stride=(1, 8)))
        vp = cute.make_tensor(fp.iterator, cute.make_layout((8, 2), stride=(1, 8)))
        # PREP packs uh as two m64n64 WGMMA fragments.  Narrow scan slabs select
        # a contiguous quarter/half of the record; DV=64 consumes it in full.
        ul_tid = (tidx % 128) + ((vb * self.dv) // 64) * 128
        ulb = nseq * mUl.stride[0] + h * mUl.stride[2] + ul_tid * 8 + (((vb * self.dv) % 64) // 16) * PKS
        plb = nseq * mPl.stride[0] + h * mPl.stride[2] + tidx * 8

        nr0 = seq_end - seq_beg
        if nr0 > BT:
            nr0 = I32(BT)
        if nchunk > 0:
            _fetch3(atomG, lay, tg, mWh, mQt, mUt, eb + seq_beg * strt, nr0, vw, vq, vu)
            if cutlass.const_expr(self.dv == 16):
                _fetchl_scan16(atom128, mUl, mPl, ulb, plb, tidx, vh, vp)
            elif cutlass.const_expr(self.dv == 32):
                _fetchl_scan32(atom128, mUl, mPl, ulb, plb, tidx, vh, vp)
            else:
                _fetchl_scan64(atom128, mUl, mPl, ulb, plb, tidx, vh, vp)
            _fetchv(mVec, nseq, 0, h, srow0, srow1, fv)

        for ic in cutlass.range(nchunk, unroll=1):
            t0 = seq_beg + ic * BT
            nrows = seq_end - t0
            if nrows > BT:
                nrows = I32(BT)
            o0 = eb + t0 * strt
            oo0 = eout + t0 * strt
            sr0 = fv[0] * pend0
            sr1 = fv[1] * pend1
            npd0 = fv[2]
            npd1 = fv[3]

            cute.arch.sync_threads()
            cute.copy(atomG, fw, tWs)
            cute.copy(atomG, fq, tQs)
            cute.copy(atomG, fu, tUts)
            cute.copy(tcA, tcA.retile(fp), dP)
            accR = cute.make_rmem_tensor(shpB, F32)
            if wg == 0:
                for vi in cutlass.range_constexpr(nB):
                    accR[vi] = fh[vi].to(F32)

            if (t0 + BT) < seq_end:
                nrn = seq_end - (t0 + BT)
                if nrn > BT:
                    nrn = I32(BT)
                _fetch3(atomG, lay, tg, mWh, mQt, mUt, o0 + BT * strt, nrn, vw, vq, vu)
                if cutlass.const_expr(self.dv == 16):
                    _fetchl_scan16(atom128, mUl, mPl, ulb + (ic + 1) * mUl.stride[1], plb + (ic + 1) * mPl.stride[1], tidx, vh, vp)
                elif cutlass.const_expr(self.dv == 32):
                    _fetchl_scan32(atom128, mUl, mPl, ulb + (ic + 1) * mUl.stride[1], plb + (ic + 1) * mPl.stride[1], tidx, vh, vp)
                else:
                    _fetchl_scan64(atom128, mUl, mPl, ulb + (ic + 1) * mUl.stride[1], plb + (ic + 1) * mPl.stride[1], tidx, vh, vp)
                _fetchv(mVec, nseq, ic + 1, h, srow0, srow1, fv)

            if cutlass.const_expr(self.dv == 64):
                frgSt = cute.make_rmem_tensor(shpS, BF)
            for v in cutlass.range_constexpr(nS):
                accSt[v] = accSt[v] * (sr1 if ((v // 2) % 2) else sr0)
                if cutlass.const_expr(self.dv < 64):
                    dk = srow1 if ((v // 2) % 2) else srow0
                    dv = ncol0 + ((v % 2) + 8 * (v // 4))
                    sSt[dv, dk] = accSt[v].to(BF)
            if cutlass.const_expr(self.dv == 64):
                frgSt.store(accSt.load().to(BF))
                cute.copy(tcS, tcS.retile(frgSt), dSt)
            _fence_bar()

            # ---- R = uh + wh Sh  (accumulator seeded from the uh registers) ----
            if wg == 0:
                _gemm(atB, accR, rA_Wh, rB_St, False, wait=0)
                frgR = cute.make_rmem_tensor(shpB, BF)
                frgR.store(accR.load().to(BF))
                cute.copy(tcB, tcB.retile(frgR), dR)
            _fence_bar()

            # ---- O = Qt Sh + P R ;  S = Diag(expc)(Sh + U^T R) ----
            accO = cute.make_rmem_tensor(shpB, F32)
            if wg == 0:
                _gemm(atB, accO, rA_Qt, rB_St, True, wait=-1)
                _gemm(atB, accO, rA_P, rB_R, False, fence=False, wait=-1)
            _gemm(atS, accSt, rA_Ut, rB_RS, False, fence=True, wait=0)
            pend0 = npd0
            pend1 = npd1

            if wg == 0:
                frgO = cute.make_rmem_tensor(shpB, BF)
                frgO.store(accO.load().to(BF))
                cute.copy(tcB, tcB.retile(frgO), dO)
            cute.arch.sync_threads()
            frgOut = cute.make_rmem_tensor(tOs.shape, BF)
            fo = cute.make_tensor(frgOut.iterator, cute.make_layout((4, self.ntok_o), stride=(1, 4)))
            cute.copy(atomG, tOs, frgOut)
            gO = cute.make_tensor((mO.iterator + oo0).align(8), layO)
            if nrows == BT:
                for i in cutlass.range_constexpr(self.ntok_o):
                    cute.copy(atomG, fo[(None, i)], gO[(None, i)])
            else:
                for i in cutlass.range_constexpr(self.ntok_o):
                    if (tgo * self.ntok_o + i) < nrows:
                        cute.copy(atomG, fo[(None, i)], gO[(None, i)])
            cute.arch.sync_threads()

        for v in cutlass.range_constexpr(nS):
            dk = srow1 if ((v // 2) % 2) else srow0
            dv = vb * self.dv + ncol0 + ((v % 2) + 8 * (v // 4))
            mS[nseq, h, dv, dk] = accSt[v] * (pend1 if ((v // 2) % 2) else pend0)


_cache = {}
# scratch buffers, fully rewritten by the prep kernel on every call
_ws = {}


def _fake(dtype, shape, leading_dim, div):
    stride = tuple(cute.sym_int64(divisibility=div) if i != leading_dim else 1 for i in range(len(shape)))
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=max(div * dtype.width // 8, 4))


def _cvt(t, ld):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=ld)


@torch.no_grad()
def run(q, k, v, g, beta, cu_seqlens, o, final_state):
    T, H, _ = q.shape
    N = cu_seqlens.shape[0] - 1
    NC = (T + BT - 1) // BT
    wkey = (T, H, N, q.device.index)
    w = _ws.get(wkey)
    if w is None:
        dev = q.device
        w = (
            torch.empty(T, H, D, dtype=torch.bfloat16, device=dev),
            torch.empty(N, NC, H, 4, PKS, dtype=torch.bfloat16, device=dev),
            torch.empty(T, H, D, dtype=torch.bfloat16, device=dev),
            torch.empty(T, H, D, dtype=torch.bfloat16, device=dev),
            torch.empty(N, NC, H, 2, PKS, dtype=torch.bfloat16, device=dev),
            torch.empty(N, NC, H, 2, D, dtype=torch.float32, device=dev),
        )
        _ws[wkey] = w
    wh, ul, ut, qt, pl, vec = w

    key = (H, N)
    ent = _cache.get(key)
    if ent is None:
        Tsym = cute.sym_int()
        NCsym = cute.sym_int()
        fq = _fake(BF, (Tsym, H, D), 2, 8)
        fk = _fake(BF, (Tsym, H, D), 2, 8)
        fv = _fake(BF, (Tsym, H, D), 2, 8)
        fgg = _fake(F32, (Tsym, H, D), 2, 4)
        fb = _fake(F32, (Tsym, H), 1, 1)
        fc = _fake(cutlass.Int32, (N + 1,), 0, 1)
        fo = _fake(BF, (Tsym, H, D), 2, 8)
        fs = _fake(F32, (N, H, D, D), 3, 4)
        fwh = _fake(BF, (Tsym, H, D), 2, 8)
        ful = _fake(BF, (N, NCsym, H, 4, PKS), 4, 8)
        fut = _fake(BF, (Tsym, H, D), 2, 8)
        fqt = _fake(BF, (Tsym, H, D), 2, 8)
        fpl = _fake(BF, (N, NCsym, H, 2, PKS), 4, 8)
        fvec = _fake(F32, (N, NCsym, H, 2, D), 4, 4)
        fst = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        cprep = cute.compile(KDAPrep(H, N), fq, fk, fv, fgg, fb, fc, fwh, ful, fut, fqt, fpl, fvec, fst, options="--enable-tvm-ffi")
        cscan = cute.compile(KDAScan(H, N), fwh, ful, fut, fqt, fpl, fvec, fc, fo, fs, fst, options="--enable-tvm-ffi")
        ent = (cprep, cscan)
        _cache[key] = ent

    cprep, cscan = ent
    cprep(q, k, v, g, beta, cu_seqlens, wh, ul, ut, qt, pl, vec)
    cscan(wh, ul, ut, qt, pl, vec, cu_seqlens, o, final_state)
