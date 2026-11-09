# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 SSA supplied-LSE dense teacher / indexer KL — CuTe DSL, Blackwell.

Fast path: one fused CTA per query tile of BQ rows.
  * The *teacher* score GEMM runs on 5th-gen tensor cores (tcgen05 / UMMA,
    CtaGroup.ONE).  Per key tile of BK=128 keys we compute, for every query of
    the tile, D[key, head] = K[key, :] . Q[query, head, :] with a single
    128 x HQ x 256 UMMA (16 k-steps) into a TMEM accumulator, then reduce
    u[query][key] = sum_h exp2(D*TSC2 - lse2[h][query]) in the TMEM->RMEM
    epilogue and publish it into a shared-memory tile sU (BQ x BK, fp32).
  * The *student* (indexer) GEMM still uses bf16 mma.sync.m16n8k16 warp MMA
    with FP32 accumulation and ldmatrix smem operand staging.
  * Phase A streams the legal key range once, reconstructing the head-summed
    teacher numerator u = sum_h exp(s_h - lse_h) directly from the *supplied*
    teacher LSE (no online max needed: s_h - lse_h is bounded above), and
    accumulates the aggregate row L1 mass S = sum_j u plus an online student
    softmax (max, sumexp).
  * Phase B re-streams the same range (recomputing the teacher numerator from
    Q/K on the fly -- nothing proportional to TQ*TK is ever materialised in
    HBM), rebuilds t = u / S and the student p, accumulates KL and forms
    dS = (p - t)/denom.
  * The indexer gradients are two more bf16 mma.sync GEMMs rather than FP32
    SIMT FMA: dq = dS @ Ki and dk = dS^T @ Qi, with dS carried as a bf16
    hi + lo pair (two accumulating FP32 passes) so the products keep ~17
    mantissa bits.  `ldmatrix.trans` supplies the MN-major operands.

Working state is O(TQ*HQ + TQ + TK): the only HBM scratch is the O(NSPLIT*TQ)
row-statistics buffer used to join the two split-key launches.

Oracle path (tiny shapes, matching the reference's own FP64 switch): scalar
FP64 kernel, one CTA per query row.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream

f32 = cutlass.Float32
f64 = cutlass.Float64
bf16 = cutlass.BFloat16
i32 = cutlass.Int32
i64 = cutlass.Int64
u8 = cutlass.Uint8

DQ = 256
DI = 128
DQ8 = DQ // 8
DI8 = DI // 8

BQ = 32
BK = 128
# Per-HQ choice of the teacher MMA CTA group (1 = 1-SM m128, 2 = 2-SM m256).
_NC_BY_HQ = {16: 2, 32: 2, 96: 2}
# Per-HQ epilogue schedule.  DUAL=True drains each accumulator stage with
# BOTH warpgroups (half the N columns each) and issues the next stage's UMMA
# from warp 0 before it joins its own drain.  That doubles the warps feeding
# the exp2 pipe, but it also halves the lead time the teacher-Q TMA gets, so
# it only pays off where a stage stages few enough bytes: HQ=96 stages one
# query per CTA under the 2-SM MMA, HQ=16/32 stage a full 64 KiB group.
_DUAL_BY_HQ = {16: False, 32: True, 96: True}


def _dual(HQ):
    return _DUAL_BY_HQ.get(HQ, False)


NWARP = 8
NTHR = NWARP * 32
PAD = 8
PQ = DQ + PAD
PI = DI + PAD
PK = BK + PAD
NKI = DI // 16
NKQ = BK // 16
NKK = BQ // 16
BKH = BK // 2

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
TSCALE = 1.0 / 16.0
SSCALE = 1.0 / math.sqrt(128.0)
TSC2 = TSCALE * LOG2E
SSC2 = SSCALE * LOG2E
NEG = -3.0e38
BIG = 1.0e30

# ---- tcgen05 teacher GEMM configuration ----------------------------------
# D[M, N] = A[M, K] @ B[K, N] with M = BK keys, N = HQ heads, K = DQ.
# A = the teacher K tile (loaded once per key tile, reused by all BQ queries).
# B = the teacher Q of ONE query (N = head, K = dim, L = query token).
ASTAGE = 1  # A (key tile) SMEM stages
BSTAGE = 2  # B (query-group) SMEM stages == TMEM accumulator stages
KCH = DQ // 64  # TMA sub-chunks along the contraction dim


def _ncta(HQ):
    """CTAs per teacher MMA: 2 uses a 2-SM (CtaGroup.TWO) m256 MMA."""
    return _NC_BY_HQ.get(HQ, 1)


def _qgroup(HQ):
    """Queries batched into one UMMA so N = G*HQ is wide enough that the
    A-operand SMEM re-reads amortise (N=16 needs ~4.5x the SMEM->tensor-core
    bandwidth the path provides; N>=64 is math bound)."""
    if HQ <= 32:
        return 128 // HQ
    return _ncta(HQ)


# sA and the indexer key tile sKi share one raw SMEM region: sA is dead as soon
# as the last teacher UMMA of the key tile has retired (guaranteed by the
# accumulator drain + the CTA barrier that follows the teacher block).
NRAW = max(BK * DQ * ASTAGE, BK * PI)

# mbarrier slot map (raw mbarriers, one Int64 each)
B_A = 0  # 1 : teacher-K tile TMA arrival
B_BF = 1  # 2 : teacher-Q stage full (TMA arrival)
B_BE = 3  # 2 : teacher-Q stage empty (UMMA consumed the operand)
B_MM = 5  # 2 : UMMA into accumulator stage complete
B_AE = 7  # 2 : accumulator stage drained by the epilogue warps
NBAR = 16

TMEM_BAR = 1  # named-barrier id used by TmemAllocator
CMASK = 3  # cluster mask covering both CTAs of the 2-SM pair

# SMEM budget, worst case HQ = 96 (see report):
#   sRaw 65536 + sB 98304 + sQib 8704(->9216) + sU/sDS 16880(->17408)
#   + sLs 12288 + sRow 256 + sWS/sSA/sSB 3072 + sInv/sNrm 256 + sRed 1024
#   + mbarriers 128  ==  ~206976 B; rounded up with slack.
SMEM_FAST = 212992
SMEM_ORACLE = 163840

OMAX = 128
OTHR = 128


def _row_groups(lay, n):
    """Trace-time helper: accumulator element index -> row offset."""
    groups = {}
    for e in range(n):
        crd = cute.crd2idx(e, lay)
        groups.setdefault(int(crd[0]), []).append(e)
    keys = sorted(groups.keys())
    return [groups[k] for k in keys]


def _epi_split(HQ):
    """TMEM epilogue sub-tiling: (num subtiles, columns per subtile).

    `Ld32x32bOp` repetition count == subtile width, so the width must be a
    legal `Repetition` (power of two).  HQ=96 -> 3 x 32, HQ=32 -> 1 x 32,
    HQ=16 -> 1 x 16.
    """
    if HQ % 32 == 0:
        return HQ // 32, 32
    return 1, HQ


# ---------------------------------------------------------------------------
@cute.kernel
def _fused(
    gQi: cute.Tensor,
    gKi: cute.Tensor,
    gLse: cute.Tensor,
    mSt: cute.Tensor,
    mEn: cute.Tensor,
    mVa: cute.Tensor,
    mDen: cute.Tensor,
    mLoss: cute.Tensor,
    pDq: cute.Tensor,
    pDk: cute.Tensor,
    pStats: cute.Tensor,
    tiled_mma: cute.TiledMma,
    mma_k: cute.TiledMma,
    tc_mma: cute.TiledMma,
    tc_mma1: cute.TiledMma,
    tma_a: cpasync.TmaInfo,
    tma_b: cpasync.TmaInfo,
    tq: i32,
    tk: i32,
    ntile: i32,
    HQ: cutlass.Constexpr,
    NC: cutlass.Constexpr,
    NSPLIT: cutlass.Constexpr,
    START: cutlass.Constexpr,
    STOP: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    # Split index varies fastest so the CTAs resident at any moment are the
    # NSPLIT siblings of a few query tiles rather than NSPLIT-times as many
    # distinct ones.  The teacher-Q tile of a query group is re-read once per
    # key tile, so shrinking the set of live query tiles by NSPLIT keeps that
    # traffic inside L2 instead of going back to DRAM (at HQ=96 a query tile
    # is 1.5 MiB, so 148 distinct live tiles would be ~222 MiB).
    bx, bidx, _ = cute.arch.block_idx()
    BKM = BK * NC
    CG = tcgen05.CtaGroup.TWO if NC == 2 else tcgen05.CtaGroup.ONE
    MMASK = i32(3) if NC == 2 else None
    if cutlass.const_expr(NC == 2):
        cv = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader = cv == i32(0)
    else:
        cv = 0
        is_leader = True
    sid = bx // NC
    warp = tidx // 32
    lane = tidx % 32
    wu = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    qt = ntile - 1 - bidx
    q0 = qt * BQ

    # Split-key phase A performs no global atomics.  Use its whole grid to
    # initialize the phase-B atomic destinations, eliminating three separate
    # framework fill launches while retaining a kernel boundary as the global
    # ordering point before phase B.
    if cutlass.const_expr(STOP == 1):
        linear = (bx * ntile + bidx) * NTHR + tidx
        stride = ntile * NSPLIT * NC * NTHR
        if linear == 0:
            mLoss[0] = f32(0.0)
        nfill = tq * DI
        if tk * DI > nfill:
            nfill = tk * DI
        zi = linear
        while zi < nfill:
            if zi < tq * DI:
                pDq[zi] = f32(0.0)
            if zi < tk * DI:
                pDk[zi] = f32(0.0)
            zi = zi + stride

    G = _qgroup(HQ)
    NB = G * HQ
    GH = G // NC  # queries staged per CTA (B is split along N)
    NBC = NB // NC  # B columns owned by this CTA
    NGRP = BQ // G
    NIT = NGRP // BSTAGE
    NSUB, REP = _epi_split(NB)
    # Both warpgroups drain every accumulator stage: warpgroup w owns the
    # N-column half [w*NB/2, (w+1)*NB/2).  NB/2 is always a whole number of
    # heads (HQ16 -> 64 = 4 queries, HQ32 -> 64 = 2 queries, HQ96/2-SM ->
    # 96 = 1 query), so the per-column (query, head) arithmetic inside a half
    # is identical for both halves up to a constant query offset.
    DUAL = _dual(HQ)
    NSUBH = NSUB // 2 if DUAL else NSUB
    GH2 = G // 2 if DUAL else G
    NF = 2 if NSUBH > 1 else 1
    mma_tiler = (BKM, NB, DQ)
    mA_mk = tma_a.tma_tensor
    mB_nkl = tma_b.tma_tensor
    bfull_layout = sm100_utils.make_smem_layout_b(tc_mma, (BKM, NB, DQ), bf16, BSTAGE)

    # ---- shared memory ----------------------------------------------------
    smem = utils.SmemAllocator()
    # Raw region shared by the UMMA A operand (teacher K tile) and, later in the
    # same key-tile iteration, by the indexer K tile.
    assert cute.cosize(cute.get_nonswizzle_portion(tma_a.smem_layout)) <= NRAW
    sRaw = smem.allocate_tensor(bf16, cute.make_layout(NRAW), byte_alignment=1024)
    sA = cute.make_tensor(
        cute.recast_ptr(
            sRaw.iterator,
            swizzle_=cute.get_swizzle_portion(tma_a.smem_layout),
            dtype=bf16,
        ),
        cute.get_nonswizzle_portion(tma_a.smem_layout),
    )
    # One raw region carries both views of the teacher-Q staging buffer: the
    # TMA writes (HQ, 64) chunks (view sB1, 4*G*BSTAGE slots) and the UMMA
    # reads it as (G*HQ, DQ) x BSTAGE.  The two address maps coincide exactly.
    sBraw = smem.allocate_tensor(bf16, cute.make_layout(NBC * DQ * BSTAGE), byte_alignment=1024)
    sB1 = cute.make_tensor(
        cute.recast_ptr(
            sBraw.iterator,
            swizzle_=cute.get_swizzle_portion(tma_b.smem_layout),
            dtype=bf16,
        ),
        cute.get_nonswizzle_portion(tma_b.smem_layout),
    )
    sB = cute.make_tensor(
        cute.recast_ptr(
            sBraw.iterator,
            swizzle_=cute.get_swizzle_portion(bfull_layout),
            dtype=bf16,
        ),
        cute.get_nonswizzle_portion(bfull_layout),
    )
    sQib = smem.allocate_tensor(bf16, cute.make_layout(BQ * PI), byte_alignment=128)
    # One fp32-sized region holds the head-summed teacher numerator sU during
    # the teacher stage and, afterwards, the bf16 hi/lo split of the dS tile
    # (disjoint lifetimes, separated by CTA barriers).
    sUb = smem.allocate_tensor(f32, cute.make_layout(BQ * PK), byte_alignment=128)
    sLs = smem.allocate_tensor(f32, cute.make_layout((BQ, HQ), stride=(HQ, 1)), byte_alignment=128)
    sRow = smem.allocate_tensor(i32, cute.make_layout((2, BQ), stride=(BQ, 1)), byte_alignment=128)
    sWS = smem.allocate_tensor(f32, cute.make_layout((NWARP, BQ), stride=(BQ, 1)), byte_alignment=128)
    sSA = smem.allocate_tensor(f32, cute.make_layout((NWARP, BQ), stride=(BQ, 1)), byte_alignment=128)
    sSB = smem.allocate_tensor(f32, cute.make_layout((NWARP, BQ), stride=(BQ, 1)), byte_alignment=128)
    sInv = smem.allocate_tensor(f32, cute.make_layout(BQ), byte_alignment=128)
    sNrm = smem.allocate_tensor(f32, cute.make_layout(BQ), byte_alignment=128)
    sRed = smem.allocate_tensor(f32, cute.make_layout(NWARP), byte_alignment=128)
    mbar = smem.allocate_array(element_type=i64, num_elems=NBAR)

    sKi2 = cute.make_tensor(sRaw.iterator, cute.make_layout((BK, DI), stride=(PI, 1)))
    sKi3 = cute.make_tensor(sRaw.iterator, cute.make_layout((BK, DI8, 8), stride=(PI, 8, 1)))
    sQi2 = cute.make_tensor(sQib.iterator, cute.make_layout((BQ, DI), stride=(PI, 1)))
    sQi3 = cute.make_tensor(sQib.iterator, cute.make_layout((BQ, DI8, 8), stride=(PI, 8, 1)))
    sQiT = cute.make_tensor(sQib.iterator, cute.make_layout((DI, BQ), stride=(1, PI)))
    sKiT = cute.make_tensor(sRaw.iterator, cute.make_layout((DI, BK), stride=(1, PI)))
    sU = cute.make_tensor(sUb.iterator, cute.make_layout((BQ, BK), stride=(PK, 1)))
    # bf16 hi/lo split of dS, aliasing the (already consumed) sU storage.
    _dh = cute.recast_ptr(sUb.iterator, dtype=bf16)
    _dl = cute.recast_ptr(sUb.iterator + (BQ * PK) // 2, dtype=bf16)
    sDh = cute.make_tensor(_dh, cute.make_layout((BQ, BK), stride=(PK, 1)))
    sDl = cute.make_tensor(_dl, cute.make_layout((BQ, BK), stride=(PK, 1)))
    sDhT = [cute.make_tensor(_dh + BKH * h, cute.make_layout((BKH, BQ), stride=(1, PK))) for h in range(2)]
    sDlT = [cute.make_tensor(_dl + BKH * h, cute.make_layout((BKH, BQ), stride=(1, PK))) for h in range(2)]

    # ---- raw mbarriers ----------------------------------------------------
    if wu == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + B_A, 1)
            for j in cutlass.range_constexpr(BSTAGE):
                cute.arch.mbarrier_init(mbar + B_BF + j, 1)
                cute.arch.mbarrier_init(mbar + B_BE + j, 1)
                cute.arch.mbarrier_init(mbar + B_MM + j, 1)
                cute.arch.mbarrier_init(
                    mbar + B_AE + j,
                    (NTHR if DUAL else 128) * NC,
                )
        cpasync.prefetch_descriptor(tma_a.atom)
        cpasync.prefetch_descriptor(tma_b.atom)
    cute.arch.mbarrier_init_fence()
    if cutlass.const_expr(NC == 2):
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    # ---- TMEM allocation (top level, never inside a dynamic branch) -------
    tmem_bar = pipeline.NamedBarrier(barrier_id=TMEM_BAR, num_threads=NTHR)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar, is_two_cta=(NC == 2))
    acc_shape = tc_mma.partition_shape_C((BKM, NB))
    tAccFake = tc_mma.make_fragment_C(cute.append(acc_shape, BSTAGE))
    num_tmem_cols = utils.get_num_tmem_alloc_cols(tAccFake)
    tmem.allocate(num_tmem_cols)

    # ---- row metadata -----------------------------------------------------
    if tidx < BQ:
        r = q0 + tidx
        st = i32(0)
        en = i32(0)
        if r < tq:
            if mVa[r] != u8(0):
                a = mSt[r]
                b = mEn[r]
                if a < 0:
                    a = i32(0)
                if b > tk:
                    b = tk
                if b > a:
                    st = a
                    en = b
        sRow[0, tidx] = st
        sRow[1, tidx] = en
    for u in cutlass.range_constexpr(NWARP * BQ // NTHR):
        sWS[(tidx + NTHR * u) // BQ, (tidx + NTHR * u) % BQ] = f32(0.0)
        sSA[(tidx + NTHR * u) // BQ, (tidx + NTHR * u) % BQ] = f32(NEG)
        sSB[(tidx + NTHR * u) // BQ, (tidx + NTHR * u) % BQ] = f32(0.0)
    for u in cutlass.range_constexpr((HQ * BQ + NTHR - 1) // NTHR):
        idx = tidx + NTHR * u
        if idx < HQ * BQ:
            rr = idx // HQ
            hh = idx % HQ
            gr = q0 + rr
            v = f32(0.0)
            if gr < tq:
                v = gLse[gr, hh] * f32(-LOG2E)
            sLs[rr, hh] = v
    ld8 = cute.make_rmem_tensor(8, bf16)
    NPRE = BK * DI8 // NTHR
    kpre = [cute.make_rmem_tensor(8, bf16) for _ in range(NPRE)]
    for u in cutlass.range_constexpr(BQ * DI8 // NTHR):
        idx = tidx + NTHR * u
        rr = idx // DI8
        cc = idx % DI8
        gr = q0 + rr
        if gr >= tq:
            gr = tq - 1
        cute.autovec_copy(gQi[gr, cc, None], ld8)
        cute.autovec_copy(ld8, sQi3[rr, cc, None])

    # ---- TMEM pointer -----------------------------------------------------
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(f32)
    tmem.relinquish_alloc_permit()
    # (MMA, MMA_M, MMA_N, STAGE)
    tAccAll = cute.make_tensor(tmem_ptr, tAccFake.layout)
    tAccS = [tAccAll[(None, None, None, j)] for j in range(BSTAGE)]

    # ---- tcgen05 operand / TMA plumbing -----------------------------------
    # (bM, bK, RestM, RestK)
    gA = cute.local_tile(mA_mk, cute.slice_(mma_tiler, (None, 0, None)), (None, None))
    # (bN, bK, RestN, RestK, RestL) for the (HQ, 64) TMA tile
    gB = cute.local_tile(mB_nkl, cute.slice_((BK, HQ, 64), (0, None, None)), (None, None, None))
    thr_tc = tc_mma.get_slice(cv)
    tCgB = tc_mma1.get_slice(0).partition_B(gB)
    tCgA = thr_tc.partition_A(gA)
    # SMEM-descriptor views (never load()/store() these)
    tCrA = tc_mma.make_fragment_A(sA)
    tCrB = tc_mma.make_fragment_B(sB)
    tAsA, tAgA = cpasync.tma_partition(
        tma_a.atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_b.atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sB1, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    tAgAm = tAgA[(None, None, 0)]
    tBgBq = tBgB[(None, 0, 0, None)]
    TXA = cute.size_in_bytes(bf16, cute.select(tma_a.smem_layout, mode=[0, 1, 2]))
    TXB = cute.size_in_bytes(bf16, cute.select(tma_b.smem_layout, mode=[0, 1, 2]))
    TXBT = TXB * GH * KCH

    # ---- TMEM -> RMEM epilogue plumbing (warps 4..7) ----------------------
    epi_tiler = (
        (
            cute.size(tAccS[0], mode=[0, 0]),
            cute.size(tAccS[0], mode=[0, 1]) // NSUB,
        ),
    )
    tAccEpi = [cute.zipped_divide(tAccS[j], epi_tiler) for j in range(BSTAGE)]
    ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(REP)), f32)
    tiled_ld = tcgen05.make_tmem_copy(ld_atom, tAccEpi[0][None, 0])
    etid = tidx % 128
    thr_ld = tiled_ld.get_slice(etid)
    tTR = [thr_ld.partition_S(tAccEpi[j]) for j in range(BSTAGE)]
    # Register-fragment shape comes from the DESTINATION side of the tmem copy.
    # `mRef` is a layout donor only -- it is never dereferenced.
    mRef = cute.make_tensor(sUb.iterator, cute.make_layout((BK, NB), stride=(NB, 1)))
    cRef = cute.zipped_divide(thr_tc.partition_C(mRef), epi_tiler)
    fshape = thr_ld.partition_D(cRef)[(None, None, 0)].shape
    frg = [cute.make_rmem_tensor(fshape, f32) for _ in range(NF)]
    usum = cute.make_rmem_tensor(GH2, f32)
    usm2 = cute.make_rmem_tensor(GH2, f32)
    wg1 = tidx >= 128

    cute.arch.sync_threads()

    kmin = i32(1 << 30)
    kmax = i32(0)
    for u in cutlass.range_constexpr(BQ):
        st = sRow[0, u]
        en = sRow[1, u]
        if en > st:
            if st < kmin:
                kmin = st
            if en > kmax:
                kmax = en
    kt0 = kmin // BKM
    kt1 = (kmax + BKM - 1) // BKM
    nkt = kt1 - kt0
    kt0_full = kt0
    kt0 = kt0_full + (nkt * sid) // NSPLIT
    kt1 = kt0_full + (nkt * (sid + 1)) // NSPLIT

    # ---- warp-level mma.sync plumbing (student GEMM) ----------------------
    thr_mma = tiled_mma.get_slice(tidx)
    cC = cute.make_identity_tensor((BQ, BK))
    tCcC = thr_mma.partition_C(cC)
    acc = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((BQ, BK)))
    NE = cute.size(acc.layout)
    RG = _row_groups(tCcC.layout, NE)
    NRG = len(RG)
    NPG = len(RG[0])
    tac = cute.make_rmem_tensor(NE, f32)

    rows = cute.make_rmem_tensor(NRG, i32)
    for g in cutlass.range_constexpr(NRG):
        rows[g] = tCcC[RG[g][0]][0]
    ks0 = cute.make_rmem_tensor(NRG, i32)
    ke0 = cute.make_rmem_tensor(NRG, i32)
    for g in cutlass.range_constexpr(NRG):
        ks0[g] = sRow[0, rows[g]]
        ke0[g] = sRow[1, rows[g]]

    ldm = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), bf16)
    ldmT = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), bf16)
    tcA = cute.make_tiled_copy_A(ldm, tiled_mma)
    tcB = cute.make_tiled_copy_B(ldm, tiled_mma)
    thA = tcA.get_slice(tidx)
    thB = tcB.get_slice(tidx)
    tCsQi = thA.partition_S(sQi2)
    tCsKi = thB.partition_S(sKi2)
    shA = thr_mma.partition_A(sQi2)[(None, None, 0)].shape
    shB = thr_mma.partition_B(sKi2)[(None, None, 0)].shape
    frA = tiled_mma.make_fragment_A((shA[0], shA[1], 1))
    frB = tiled_mma.make_fragment_B((shB[0], shB[1], 1))
    rA = thA.retile(frA)
    rB = thB.retile(frB)

    # ---- indexer gradient GEMMs (tensor cores) ---------------------------
    # dq[BQ, DI] += dS[BQ, BK] @ Ki[BK, DI]   (A k-major, B mn-major)
    # dk[BK, DI] += dS^T[BK, BQ] @ Qi[BQ, DI] (A mn-major, B mn-major)
    # dS is carried in bf16 hi + lo halves so the products keep >=16 mantissa
    # bits -- a single bf16 pass would be ~1e-3 relative, well over tolerance.
    tcBq = cute.make_tiled_copy_B(ldmT, tiled_mma)
    thBq = tcBq.get_slice(tidx)
    tqsAh = thA.partition_S(sDh)
    tqsAl = thA.partition_S(sDl)
    tqsB = thBq.partition_S(sKiT)
    rBq = thBq.retile(frB)
    accq = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((BQ, DI)))
    accq.fill(0.0)

    thr_mk = mma_k.get_slice(tidx)
    tcAk = cute.make_tiled_copy_A(ldmT, mma_k)
    tcBk = cute.make_tiled_copy_B(ldmT, mma_k)
    thAk = tcAk.get_slice(tidx)
    thBk = tcBk.get_slice(tidx)
    tksAh = [thAk.partition_S(sDhT[h]) for h in range(2)]
    tksAl = [thAk.partition_S(sDlT[h]) for h in range(2)]
    tksB = thBk.partition_S(sQiT)
    shAk = thr_mk.partition_A(sDhT[0])[(None, None, 0)].shape
    shBk = thr_mk.partition_B(sQiT)[(None, None, 0)].shape
    frAk = mma_k.make_fragment_A((shAk[0], shAk[1], 1))
    frBk = mma_k.make_fragment_B((shBk[0], shBk[1], 1))
    rAk = thAk.retile(frAk)
    rBk = thBk.retile(frBk)
    acck = mma_k.make_fragment_C(mma_k.partition_shape_C((BKH, DI)))
    NEK = cute.size(acck.layout)
    cCk = cute.make_identity_tensor((BKH, DI))
    tCcCk = thr_mk.partition_C(cCk)

    ssum = cute.make_rmem_tensor(NRG, f32)
    smax = cute.make_rmem_tensor(NRG, f32)
    slsum = cute.make_rmem_tensor(NRG, f32)
    for g in cutlass.range_constexpr(NRG):
        ssum[g] = f32(0.0)
        smax[g] = f32(NEG)
        slsum[g] = f32(0.0)

    mkv = cute.make_rmem_tensor(NE, f32)
    dk2 = cute.make_rmem_tensor(2, f32)

    inv_den = f32(1.0) / mDen[0]
    lsum = f32(0.0)
    aph = i32(0)
    invs = cute.make_rmem_tensor(NRG, f32)
    nrm = cute.make_rmem_tensor(NRG, f32)

    if cutlass.const_expr(START == 1):
        if tidx < BQ:
            gr = q0 + tidx
            tot = f32(0.0)
            gm = f32(NEG)
            if gr < tq:
                for sp in cutlass.range_constexpr(NC * NSPLIT):
                    tot = tot + pStats[(sp * tq + gr) * 3]
                    mv = pStats[(sp * tq + gr) * 3 + 1]
                    if mv > gm:
                        gm = mv
            sl = f32(0.0)
            if gr < tq:
                for sp in cutlass.range_constexpr(NC * NSPLIT):
                    mv = pStats[(sp * tq + gr) * 3 + 1]
                    lv = pStats[(sp * tq + gr) * 3 + 2]
                    sl = sl + lv * cute.math.exp2(mv - gm, fastmath=True)
            iv = f32(0.0)
            if tot > f32(0.0):
                iv = f32(1.0) / tot
            nv = f32(0.0)
            if sl > f32(0.0):
                nv = gm + cute.math.log2(sl, fastmath=True)
            sInv[tidx] = iv
            sNrm[tidx] = nv
        cute.arch.sync_threads()
        for g in cutlass.range_constexpr(NRG):
            invs[g] = sInv[rows[g]]
            nrm[g] = sNrm[rows[g]]

    for phase in cutlass.range_constexpr(START, STOP):
        kt = kt0
        while kt < kt1:
            kbase = kt * BKM + cv * BK
            # Issue the indexer-K global loads for THIS key tile up front so
            # their HBM/L2 latency is covered by the teacher UMMA block; the
            # SMEM landing zone aliases the UMMA A operand, so the stores wait.
            for u in cutlass.range_constexpr(NPRE):
                idx = tidx + NTHR * u
                jj = kbase + idx // DI8
                if jj >= tk:
                    jj = tk - 1
                cute.autovec_copy(gKi[jj, idx % DI8, None], kpre[u])
            cute.arch.sync_threads()

            # ---- teacher scores: tcgen05 UMMA -> TMEM -> sU ---------------
            # Both phases run this block: the teacher numerator is recomputed
            # from Q/K rather than persisted, so no HBM state grows with
            # TQ * TK.
            if cutlass.const_expr(DUAL):
                if wu == 0:
                    # Every CTA of the pair TMAs its own 128-key A slice and its own
                    # GH-query half of the B operand; the 2-SM copy routes both
                    # CTAs' complete_tx to the leader's mbarrier, so only the leader
                    # arms (with double the byte count) and only the leader waits.
                    if is_leader:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(mbar + B_A, NC * TXA)
                    cute.copy(
                        tma_a.atom,
                        tAgAm[(None, kt)],
                        tAsA[(None, 0)],
                        tma_bar_ptr=mbar + B_A,
                    )
                    for j in cutlass.range_constexpr(BSTAGE):
                        if is_leader:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(mbar + B_BF + j, NC * TXBT)
                        for g in cutlass.range_constexpr(GH):
                            gqp = cutlass.min(q0 + j * G + cv * i32(GH) + g, tq - i32(1))
                            for c in cutlass.range_constexpr(KCH):
                                cute.copy(
                                    tma_b.atom,
                                    tBgBq[(None, (c, gqp))],
                                    tBsB[(None, g + GH * (c + KCH * j))],
                                    tma_bar_ptr=mbar + B_BF + j,
                                )
                    if is_leader:
                        cute.arch.mbarrier_wait(mbar + B_A, aph)
                        # Prologue UMMA for group 0.  Each accumulator stage
                        # completes exactly NIT times per key tile and NIT is even
                        # for every supported (HQ, G), so every mbarrier's phase
                        # parity is back to its initial value at the tile boundary
                        # and this wait is a constant phase 0.
                        cute.arch.mbarrier_wait(mbar + B_BF, 0)
                        tc_mma.set(tcgen05.Field.ACCUMULATE, False)
                        cute.gemm(
                            tc_mma,
                            tAccS[0],
                            tCrA[(None, None, None, 0)],
                            tCrB[(None, None, None, 0)],
                            tAccS[0],
                        )
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar + B_MM, MMASK, CG)
                            tcgen05.commit(mbar + B_BE, MMASK, CG)

                for it in cutlass.range(NIT, unroll=1):
                    par = it % 2
                    parn = (it + 1) % 2
                    for j in cutlass.range_constexpr(BSTAGE):
                        jn = 1 - j
                        qb = (BSTAGE * it + i32(j)) * G
                        # Warp 0 launches the UMMA of the NEXT group into the other
                        # accumulator stage *before* it joins its own warpgroup's
                        # drain, so the tensor core and the SFU pipe overlap.
                        if wu == 0:
                            if is_leader:
                                if cutlass.const_expr(j == 0):
                                    # group 2*it+1 -> stage 1; always exists
                                    cute.arch.mbarrier_wait(mbar + B_BF + jn, par)
                                    if it > i32(0):
                                        cute.arch.mbarrier_wait(mbar + B_AE + jn, parn)
                                    tc_mma.set(tcgen05.Field.ACCUMULATE, False)
                                    cute.gemm(
                                        tc_mma,
                                        tAccS[jn],
                                        tCrA[(None, None, None, 0)],
                                        tCrB[(None, None, None, jn)],
                                        tAccS[jn],
                                    )
                                    with cute.arch.elect_one():
                                        tcgen05.commit(mbar + B_MM + jn, MMASK, CG)
                                        tcgen05.commit(mbar + B_BE + jn, MMASK, CG)
                                else:
                                    # group 2*it+2 -> stage 0; only if it remains
                                    if it < i32(NIT - 1):
                                        cute.arch.mbarrier_wait(mbar + B_BF + jn, parn)
                                        cute.arch.mbarrier_wait(mbar + B_AE + jn, par)
                                        tc_mma.set(tcgen05.Field.ACCUMULATE, False)
                                        cute.gemm(
                                            tc_mma,
                                            tAccS[jn],
                                            tCrA[(None, None, None, 0)],
                                            tCrB[(None, None, None, jn)],
                                            tAccS[jn],
                                        )
                                        with cute.arch.elect_one():
                                            tcgen05.commit(mbar + B_MM + jn, MMASK, CG)
                                            tcgen05.commit(mbar + B_BE + jn, MMASK, CG)
                        # Refill sB stage j for the group BSTAGE ahead, issued
                        # before the drain so the copy has a full drain of latency
                        # to land.  Warp 4 owns it rather than warp 0: warp 0 is
                        # already on the critical path with the UMMA launch, and
                        # every instruction it spends here is added to the drain it
                        # then has to join.
                        if wu == 4:
                            if it < i32(NIT - 1):
                                cute.arch.mbarrier_wait(mbar + B_BE + j, par)
                                gq0 = q0 + (BSTAGE * (it + i32(1)) + i32(j)) * G
                                if is_leader:
                                    with cute.arch.elect_one():
                                        cute.arch.mbarrier_arrive_and_expect_tx(mbar + B_BF + j, NC * TXBT)
                                for g in cutlass.range_constexpr(GH):
                                    gqn = cutlass.min(gq0 + cv * i32(GH) + i32(g), tq - i32(1))
                                    for c in cutlass.range_constexpr(KCH):
                                        cute.copy(
                                            tma_b.atom,
                                            tBgBq[(None, (c, gqn))],
                                            tBsB[(None, g + GH * (c + KCH * j))],
                                            tma_bar_ptr=mbar + B_BF + j,
                                        )
                        cute.arch.mbarrier_wait(mbar + B_MM + j, par)
                        # Each warpgroup reduces its own N-column half of the
                        # accumulator: two warps per SM sub-partition feed the
                        # exp2 pipe instead of one.  The two halves are emitted
                        # inline (a helper would either capture dynamic scope or
                        # escape the DSL preprocessor).
                        if wg1:
                            for gl in cutlass.range_constexpr(GH2):
                                usum[gl] = f32(0.0)
                                usm2[gl] = f32(0.0)
                            for cch in cutlass.range_constexpr((NSUBH + NF - 1) // NF):
                                ch = cch * NF
                                nf = min(NF, NSUBH - ch)
                                for f in cutlass.range_constexpr(nf):
                                    cute.copy(tiled_ld, tTR[j][None, None, NSUBH + ch + f], frg[f])
                                cute.arch.fence_view_async_tmem_load()
                                for f in cutlass.range_constexpr(nf):
                                    for c2 in cutlass.range_constexpr(REP // 2):
                                        c = 2 * c2
                                        col = (ch + f) * REP + c
                                        ql = col // HQ
                                        h0 = col % HQ
                                        qr = qb + i32(GH2 + ql)
                                        x0, x1 = cute.arch.fma_packed_f32x2(
                                            (frg[f][c], frg[f][c + 1]),
                                            (f32(TSC2), f32(TSC2)),
                                            (sLs[qr, h0], sLs[qr, h0 + 1]),
                                            rnd="rn",
                                        )
                                        usum[ql], usm2[ql] = cute.arch.add_packed_f32x2(
                                            (usum[ql], usm2[ql]),
                                            (
                                                cute.math.exp2(x0, fastmath=True),
                                                cute.math.exp2(x1, fastmath=True),
                                            ),
                                            rnd="rn",
                                        )
                            if lane == 0:
                                if cutlass.const_expr(NC == 2):
                                    cute.arch.mbarrier_arrive(
                                        mbar + B_AE + j,
                                        peer_cta_rank_in_cluster=i32(0),
                                        arrive_count=32,
                                    )
                                else:
                                    cute.arch.mbarrier_arrive(mbar + B_AE + j, arrive_count=32)
                            kk = kbase + etid
                            for gl in cutlass.range_constexpr(GH2):
                                qq = qb + i32(GH2 + gl)
                                usum[gl] = usum[gl] + usm2[gl]
                                uv = f32(0.0)
                                if kk >= sRow[0, qq]:
                                    if kk < sRow[1, qq]:
                                        uv = usum[gl]
                                sU[qq, etid] = uv
                        else:
                            for gl in cutlass.range_constexpr(GH2):
                                usum[gl] = f32(0.0)
                                usm2[gl] = f32(0.0)
                            for cch in cutlass.range_constexpr((NSUBH + NF - 1) // NF):
                                ch = cch * NF
                                nf = min(NF, NSUBH - ch)
                                for f in cutlass.range_constexpr(nf):
                                    cute.copy(tiled_ld, tTR[j][None, None, 0 + ch + f], frg[f])
                                cute.arch.fence_view_async_tmem_load()
                                for f in cutlass.range_constexpr(nf):
                                    for c2 in cutlass.range_constexpr(REP // 2):
                                        c = 2 * c2
                                        col = (ch + f) * REP + c
                                        ql = col // HQ
                                        h0 = col % HQ
                                        qr = qb + i32(0 + ql)
                                        x0, x1 = cute.arch.fma_packed_f32x2(
                                            (frg[f][c], frg[f][c + 1]),
                                            (f32(TSC2), f32(TSC2)),
                                            (sLs[qr, h0], sLs[qr, h0 + 1]),
                                            rnd="rn",
                                        )
                                        usum[ql], usm2[ql] = cute.arch.add_packed_f32x2(
                                            (usum[ql], usm2[ql]),
                                            (
                                                cute.math.exp2(x0, fastmath=True),
                                                cute.math.exp2(x1, fastmath=True),
                                            ),
                                            rnd="rn",
                                        )
                            if lane == 0:
                                if cutlass.const_expr(NC == 2):
                                    cute.arch.mbarrier_arrive(
                                        mbar + B_AE + j,
                                        peer_cta_rank_in_cluster=i32(0),
                                        arrive_count=32,
                                    )
                                else:
                                    cute.arch.mbarrier_arrive(mbar + B_AE + j, arrive_count=32)
                            kk = kbase + etid
                            for gl in cutlass.range_constexpr(GH2):
                                qq = qb + i32(0 + gl)
                                usum[gl] = usum[gl] + usm2[gl]
                                uv = f32(0.0)
                                if kk >= sRow[0, qq]:
                                    if kk < sRow[1, qq]:
                                        uv = usum[gl]
                                sU[qq, etid] = uv
            else:
                if wu == 0:
                    # Every CTA of the pair TMAs its own 128-key A slice and its own
                    # GH-query half of the B operand; the 2-SM copy routes both
                    # CTAs' complete_tx to the leader's mbarrier, so only the leader
                    # arms (with double the byte count) and only the leader waits.
                    if is_leader:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(mbar + B_A, NC * TXA)
                    cute.copy(
                        tma_a.atom,
                        tAgAm[(None, kt)],
                        tAsA[(None, 0)],
                        tma_bar_ptr=mbar + B_A,
                    )
                    for j in cutlass.range_constexpr(BSTAGE):
                        if is_leader:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(mbar + B_BF + j, NC * TXBT)
                        for g in cutlass.range_constexpr(GH):
                            gqp = cutlass.min(q0 + j * G + cv * i32(GH) + g, tq - i32(1))
                            for c in cutlass.range_constexpr(KCH):
                                cute.copy(
                                    tma_b.atom,
                                    tBgBq[(None, (c, gqp))],
                                    tBsB[(None, g + GH * (c + KCH * j))],
                                    tma_bar_ptr=mbar + B_BF + j,
                                )
                    if is_leader:
                        cute.arch.mbarrier_wait(mbar + B_A, aph)
                    for it in cutlass.range(NIT - 1, unroll=1):
                        par = it % 2
                        parn = (it + 1) % 2
                        for j in cutlass.range_constexpr(BSTAGE):
                            if is_leader:
                                cute.arch.mbarrier_wait(mbar + B_BF + j, par)
                                cute.arch.mbarrier_wait(mbar + B_AE + j, parn)
                                tc_mma.set(tcgen05.Field.ACCUMULATE, False)
                                cute.gemm(
                                    tc_mma,
                                    tAccS[j],
                                    tCrA[(None, None, None, 0)],
                                    tCrB[(None, None, None, j)],
                                    tAccS[j],
                                )
                                with cute.arch.elect_one():
                                    tcgen05.commit(mbar + B_MM + j, MMASK, CG)
                                    tcgen05.commit(mbar + B_BE + j, MMASK, CG)
                            cute.arch.mbarrier_wait(mbar + B_BE + j, par)
                            gq0 = q0 + (BSTAGE * (it + i32(1)) + i32(j)) * G
                            if is_leader:
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive_and_expect_tx(mbar + B_BF + j, NC * TXBT)
                            for g in cutlass.range_constexpr(GH):
                                gqn = cutlass.min(gq0 + cv * i32(GH) + i32(g), tq - i32(1))
                                for c in cutlass.range_constexpr(KCH):
                                    cute.copy(
                                        tma_b.atom,
                                        tBgBq[(None, (c, gqn))],
                                        tBsB[(None, g + GH * (c + KCH * j))],
                                        tma_bar_ptr=mbar + B_BF + j,
                                    )
                    if is_leader:
                        for j in cutlass.range_constexpr(BSTAGE):
                            cute.arch.mbarrier_wait(mbar + B_BF + j, (NIT - 1) % 2)
                            cute.arch.mbarrier_wait(mbar + B_AE + j, NIT % 2)
                            tc_mma.set(tcgen05.Field.ACCUMULATE, False)
                            cute.gemm(
                                tc_mma,
                                tAccS[j],
                                tCrA[(None, None, None, 0)],
                                tCrB[(None, None, None, j)],
                                tAccS[j],
                            )
                            with cute.arch.elect_one():
                                tcgen05.commit(mbar + B_MM + j, MMASK, CG)
                                tcgen05.commit(mbar + B_BE + j, MMASK, CG)

                if wu >= 4:
                    for it in cutlass.range(NIT, unroll=1):
                        par = it % 2
                        for j in cutlass.range_constexpr(BSTAGE):
                            qb = (BSTAGE * it + i32(j)) * G
                            cute.arch.mbarrier_wait(mbar + B_MM + j, par)
                            for g in cutlass.range_constexpr(G):
                                usum[g] = f32(0.0)
                                usm2[g] = f32(0.0)
                            for cch in cutlass.range_constexpr((NSUB + NF - 1) // NF):
                                ch = cch * NF
                                nf = min(NF, NSUB - ch)
                                for f in cutlass.range_constexpr(nf):
                                    cute.copy(tiled_ld, tTR[j][None, None, ch + f], frg[f])
                                cute.arch.fence_view_async_tmem_load()
                                for f in cutlass.range_constexpr(nf):
                                    for c2 in cutlass.range_constexpr(REP // 2):
                                        c = 2 * c2
                                        col = (ch + f) * REP + c
                                        ql = col // HQ
                                        h0 = col % HQ
                                        qr = qb + i32(ql)
                                        x0, x1 = cute.arch.fma_packed_f32x2(
                                            (frg[f][c], frg[f][c + 1]),
                                            (f32(TSC2), f32(TSC2)),
                                            (sLs[qr, h0], sLs[qr, h0 + 1]),
                                            rnd="rn",
                                        )
                                        usum[ql], usm2[ql] = cute.arch.add_packed_f32x2(
                                            (usum[ql], usm2[ql]),
                                            (
                                                cute.math.exp2(x0, fastmath=True),
                                                cute.math.exp2(x1, fastmath=True),
                                            ),
                                            rnd="rn",
                                        )
                            if lane == 0:
                                if cutlass.const_expr(NC == 2):
                                    cute.arch.mbarrier_arrive(
                                        mbar + B_AE + j,
                                        peer_cta_rank_in_cluster=i32(0),
                                        arrive_count=32,
                                    )
                                else:
                                    cute.arch.mbarrier_arrive(mbar + B_AE + j, arrive_count=32)
                            kk = kbase + etid
                            for g in cutlass.range_constexpr(G):
                                qq = qb + i32(g)
                                usum[g] = usum[g] + usm2[g]
                                uv = f32(0.0)
                                if kk >= sRow[0, qq]:
                                    if kk < sRow[1, qq]:
                                        uv = usum[g]
                                sU[qq, etid] = uv
            aph = i32(1) - aph
            cute.arch.sync_threads()

            for e in cutlass.range_constexpr(NE):
                mkv[e] = f32(NEG)
            for g in cutlass.range_constexpr(NRG):
                for ii in cutlass.range_constexpr(NPG):
                    e = RG[g][ii]
                    j = kbase + tCcC[e][1]
                    if j >= ks0[g]:
                        if j < ke0[g]:
                            mkv[e] = f32(0.0)

            if cutlass.const_expr(phase == 0):
                for g in cutlass.range_constexpr(NRG):
                    for ii in cutlass.range_constexpr(NPG):
                        e = RG[g][ii]
                        ssum[g] = ssum[g] + sU[tCcC[e][0], tCcC[e][1]]
            else:
                for e in cutlass.range_constexpr(NE):
                    tac[e] = sU[tCcC[e][0], tCcC[e][1]]

            # ---- indexer K tile (aliases the teacher-K / UMMA-A storage) --
            cute.arch.sync_threads()
            for u in cutlass.range_constexpr(NPRE):
                idx = tidx + NTHR * u
                cute.autovec_copy(kpre[u], sKi3[idx // DI8, idx % DI8, None])
            cute.arch.sync_threads()

            acc.fill(0.0)
            for kb in cutlass.range_constexpr(NKI):
                cute.copy(tcA, tCsQi[None, None, kb], rA[None, None, 0])
                cute.copy(tcB, tCsKi[None, None, kb], rB[None, None, 0])
                cute.gemm(tiled_mma, acc, frA[None, None, 0], frB[None, None, 0], acc)

            if cutlass.const_expr(phase == 0):
                for g in cutlass.range_constexpr(NRG):
                    tmax = f32(NEG)
                    for ii in cutlass.range_constexpr(NPG):
                        e = RG[g][ii]
                        v = acc[e] * f32(SSC2) + mkv[e]
                        acc[e] = v
                        tmax = cute.arch.fmax(tmax, v)
                    nm = cute.arch.fmax(smax[g], tmax)
                    s = f32(0.0)
                    for ii in cutlass.range_constexpr(NPG):
                        e = RG[g][ii]
                        s = s + cute.math.exp2(acc[e] - nm, fastmath=True)
                    slsum[g] = slsum[g] * cute.math.exp2(smax[g] - nm, fastmath=True) + s
                    smax[g] = nm
            else:
                sm_all = f32(0.0)
                for g in cutlass.range_constexpr(NRG):
                    iv = invs[g]
                    nv = nrm[g]
                    keep = (ke0[g] - ks0[g]) > 1
                    for ii in cutlass.range_constexpr(NPG):
                        e = RG[g][ii]
                        lp = acc[e] * f32(SSC2) - nv
                        tv = tac[e] * iv
                        ds = f32(0.0)
                        if mkv[e] == f32(0.0):
                            if keep:
                                pv = cute.math.exp2(lp, fastmath=True)
                                ds = (pv - tv) * inv_den
                                if tv > f32(0.0):
                                    sm_all = sm_all + tv * (cute.math.log2(tv, fastmath=True) - lp) * f32(LN2)
                        dh = ds.to(bf16)
                        sDh[tCcC[e][0], tCcC[e][1]] = dh
                        sDl[tCcC[e][0], tCcC[e][1]] = (ds - dh.to(f32)).to(bf16)
                lsum = lsum + sm_all
                cute.arch.sync_threads()

                # dq += dS @ Ki  (hi and lo bf16 passes share the B fragment)
                for kb in cutlass.range_constexpr(NKQ):
                    cute.copy(tcBq, tqsB[None, None, kb], rBq[None, None, 0])
                    cute.copy(tcA, tqsAh[None, None, kb], rA[None, None, 0])
                    cute.gemm(tiled_mma, accq, frA[None, None, 0], frB[None, None, 0], accq)
                    cute.copy(tcA, tqsAl[None, None, kb], rA[None, None, 0])
                    cute.gemm(tiled_mma, accq, frA[None, None, 0], frB[None, None, 0], accq)

                # dk += dS^T @ Qi, two 64-key halves to cap accumulator regs
                for h in cutlass.range_constexpr(2):
                    acck.fill(0.0)
                    for kb in cutlass.range_constexpr(NKK):
                        cute.copy(tcBk, tksB[None, None, kb], rBk[None, None, 0])
                        cute.copy(tcAk, tksAh[h][None, None, kb], rAk[None, None, 0])
                        cute.gemm(mma_k, acck, frAk[None, None, 0], frBk[None, None, 0], acck)
                        cute.copy(tcAk, tksAl[h][None, None, kb], rAk[None, None, 0])
                        cute.gemm(mma_k, acck, frAk[None, None, 0], frBk[None, None, 0], acck)
                    # The m16n8k16 accumulator gives each thread its columns in
                    # adjacent pairs, so consecutive fragment elements are
                    # contiguous in dk: reduce with .v2 atomics, halving the
                    # number of L2 atomic operations.
                    for ep in cutlass.range_constexpr(NEK // 2):
                        e = 2 * ep
                        v0 = acck[e] * f32(SSCALE)
                        v1 = acck[e + 1] * f32(SSCALE)
                        nz = i32(0)
                        if v0 != f32(0.0):
                            nz = i32(1)
                        if v1 != f32(0.0):
                            nz = i32(1)
                        if nz != i32(0):
                            j = kbase + BKH * h + tCcCk[e][0]
                            if j < tk:
                                dk2[0] = v0
                                dk2[1] = v1
                                cute.arch.atomic_add(
                                    pDk.iterator + (j * DI + tCcCk[e][1]),
                                    dk2.load(),
                                    scope="gpu",
                                )
            kt = kt + 1

        if cutlass.const_expr(phase == 0):
            for g in cutlass.range_constexpr(NRG):
                sv = ssum[g]
                mv = smax[g]
                lv = slsum[g]
                for off in cutlass.range_constexpr(2):
                    sv = sv + cute.arch.shuffle_sync_bfly(sv, 1 << off)
                    m2 = cute.arch.shuffle_sync_bfly(mv, 1 << off)
                    l2 = cute.arch.shuffle_sync_bfly(lv, 1 << off)
                    nm = mv
                    if m2 > nm:
                        nm = m2
                    lv = lv * cute.math.exp2(mv - nm, fastmath=True) + l2 * cute.math.exp2(m2 - nm, fastmath=True)
                    mv = nm
                if lane % 4 == 0:
                    sWS[warp, rows[g]] = sv
                    sSA[warp, rows[g]] = mv
                    sSB[warp, rows[g]] = lv
            cute.arch.sync_threads()
            if tidx < BQ:
                gr = q0 + tidx
                tot = f32(0.0)
                gm = f32(NEG)
                for w in cutlass.range_constexpr(NWARP):
                    tot = tot + sWS[w, tidx]
                    if sSA[w, tidx] > gm:
                        gm = sSA[w, tidx]
                sl = f32(0.0)
                for w in cutlass.range_constexpr(NWARP):
                    sl = sl + sSB[w, tidx] * cute.math.exp2(sSA[w, tidx] - gm, fastmath=True)
                if cutlass.const_expr(STOP == 1):
                    if gr < tq:
                        pStats[(bx * tq + gr) * 3] = tot
                        pStats[(bx * tq + gr) * 3 + 1] = gm
                        pStats[(bx * tq + gr) * 3 + 2] = sl
                else:
                    iv = f32(0.0)
                    if tot > f32(0.0):
                        iv = f32(1.0) / tot
                    nv = f32(0.0)
                    if sl > f32(0.0):
                        nv = gm + cute.math.log2(sl, fastmath=True)
                    sInv[tidx] = iv
                    sNrm[tidx] = nv
            if cutlass.const_expr(STOP == 2):
                cute.arch.sync_threads()
                for g in cutlass.range_constexpr(NRG):
                    invs[g] = sInv[rows[g]]
                    nrm[g] = sNrm[rows[g]]

    # ---- epilogue ---------------------------------------------------------
    if cutlass.const_expr(STOP == 2):
        # m16n8k16 assigns adjacent output columns to consecutive fragment
        # elements, so one f32x2 atomic deposits each pair of dQ values.
        for ep in cutlass.range_constexpr(NE // 2):
            e = 2 * ep
            gr = q0 + tCcC[e][0]
            if gr < tq:
                v0 = accq[e] * f32(SSCALE)
                v1 = accq[e + 1] * f32(SSCALE)
                nz = i32(0)
                if v0 != f32(0.0):
                    nz = i32(1)
                if v1 != f32(0.0):
                    nz = i32(1)
                if nz != i32(0):
                    dk2[0] = v0
                    dk2[1] = v1
                    cute.arch.atomic_add(
                        pDq.iterator + gr * DI + tCcC[e][1],
                        dk2.load(),
                        scope="gpu",
                    )

        wsum = cute.arch.warp_reduction_sum(lsum)
        if lane == 0:
            sRed[warp] = wsum
        cute.arch.sync_threads()
        if tidx == 0:
            tot = f32(0.0)
            for w in cutlass.range_constexpr(NWARP):
                tot = tot + sRed[w]
            cute.arch.atomic_add(mLoss.iterator, tot * inv_den, scope="gpu")

    # ---- release TMEM -----------------------------------------------------
    pipeline.sync(barrier_id=TMEM_BAR)
    tmem.free(tmem_ptr)


# ---------------------------------------------------------------------------
@cute.kernel
def _oracle(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gQi: cute.Tensor,
    gKi: cute.Tensor,
    gLse: cute.Tensor,
    mSt: cute.Tensor,
    mEn: cute.Tensor,
    mVa: cute.Tensor,
    mDen: cute.Tensor,
    mLoss: cute.Tensor,
    pDq: cute.Tensor,
    pDk: cute.Tensor,
    tq: i32,
    tk: i32,
    HQ: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    i, _, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    sQ = smem.allocate_tensor(bf16, cute.make_layout(HQ * DQ), byte_alignment=128)
    sK = smem.allocate_tensor(bf16, cute.make_layout((OMAX, DQ), stride=(DQ, 1)), byte_alignment=128)
    sKi = smem.allocate_tensor(bf16, cute.make_layout((OMAX, DI), stride=(DI, 1)), byte_alignment=128)
    sQi = smem.allocate_tensor(bf16, cute.make_layout(DI), byte_alignment=128)
    sLse = smem.allocate_tensor(f64, cute.make_layout(HQ), byte_alignment=128)
    sS = smem.allocate_tensor(f64, cute.make_layout(OMAX), byte_alignment=128)
    sT = smem.allocate_tensor(f64, cute.make_layout(OMAX), byte_alignment=128)
    sRed = smem.allocate_tensor(f64, cute.make_layout(OTHR), byte_alignment=128)

    st = i32(0)
    en = i32(0)
    if mVa[i] != u8(0):
        a = mSt[i]
        b = mEn[i]
        if a < 0:
            a = i32(0)
        if b > tk:
            b = tk
        if b > a:
            st = a
            en = b
    n = en - st
    den = mDen[0].to(f64)

    for h in cutlass.range_constexpr(HQ):
        for u in cutlass.range_constexpr(DQ // OTHR):
            sQ[h * DQ + tidx + OTHR * u] = gQ[i, h, tidx + OTHR * u]
    if tidx < DI:
        sQi[tidx] = gQi[i, tidx]
    if tidx < HQ:
        sLse[tidx] = gLse[i, tidx].to(f64)
    for r in cutlass.range(n, unroll=1):
        for u in cutlass.range_constexpr(DQ // OTHR):
            sK[r, tidx + OTHR * u] = gK[st + r, tidx + OTHR * u]
        sKi[r, tidx] = gKi[st + r, tidx]
    cute.arch.sync_threads()

    ac = cute.make_rmem_tensor(8, f64)
    if tidx < n:
        uu = f64(0.0)
        for hc in cutlass.range_constexpr(HQ // 8):
            for z in cutlass.range_constexpr(8):
                ac[z] = f64(0.0)
            for d in cutlass.range(DQ, unroll=1):
                kv = sK[tidx, d].to(f64)
                for z in cutlass.range_constexpr(8):
                    ac[z] = ac[z] + sQ[(hc * 8 + z) * DQ + d].to(f64) * kv
            for z in cutlass.range_constexpr(8):
                uu = uu + cute.math.exp(ac[z] * f64(TSCALE) - sLse[hc * 8 + z])
        sT[tidx] = uu
        av = f64(0.0)
        for d in cutlass.range(DI, unroll=1):
            av = av + sQi[d].to(f64) * sKi[tidx, d].to(f64)
        sS[tidx] = av * f64(SSCALE)
    else:
        sT[tidx] = f64(0.0)
        sS[tidx] = f64(-1.0e300)
    cute.arch.sync_threads()

    sRed[tidx] = sT[tidx]
    cute.arch.sync_threads()
    s = OTHR // 2
    while s > 0:
        if tidx < s:
            sRed[tidx] = sRed[tidx] + sRed[tidx + s]
        cute.arch.sync_threads()
        s = s // 2
    tot = sRed[0]
    cute.arch.sync_threads()

    sRed[tidx] = sS[tidx]
    cute.arch.sync_threads()
    s = OTHR // 2
    while s > 0:
        if tidx < s:
            if sRed[tidx + s] > sRed[tidx]:
                sRed[tidx] = sRed[tidx + s]
        cute.arch.sync_threads()
        s = s // 2
    sm = sRed[0]
    cute.arch.sync_threads()

    ev = f64(0.0)
    if tidx < n:
        ev = cute.math.exp(sS[tidx] - sm)
    sRed[tidx] = ev
    cute.arch.sync_threads()
    s = OTHR // 2
    while s > 0:
        if tidx < s:
            sRed[tidx] = sRed[tidx] + sRed[tidx + s]
        cute.arch.sync_threads()
        s = s // 2
    sl = sRed[0]
    cute.arch.sync_threads()
    lsl = f64(0.0)
    if sl > f64(0.0):
        lsl = cute.math.log(sl)

    ls = f64(0.0)
    dsv = f64(0.0)
    if tidx < n:
        if tot > f64(0.0):
            tv = sT[tidx] / tot
            lp = sS[tidx] - sm - lsl
            pv = cute.math.exp(lp)
            if tv > f64(0.0):
                ls = tv * (cute.math.log(tv) - lp)
            dsv = (pv - tv) / den
    sRed[tidx] = ls
    sT[tidx] = dsv
    cute.arch.sync_threads()
    s = OTHR // 2
    while s > 0:
        if tidx < s:
            sRed[tidx] = sRed[tidx] + sRed[tidx + s]
        cute.arch.sync_threads()
        s = s // 2
    if tidx == 0:
        cute.arch.atomic_add(mLoss.iterator, (sRed[0] / den).to(f32), scope="gpu")
    cute.arch.sync_threads()

    av = f64(0.0)
    for jj in cutlass.range(n, unroll=1):
        av = av + sT[jj] * sKi[jj, tidx].to(f64)
    pDq[i * DI + tidx] = (av * f64(SSCALE)).to(f32)
    qv = sQi[tidx].to(f64) * f64(SSCALE)
    for jj in cutlass.range(n, unroll=1):
        v = (sT[jj] * qv).to(f32)
        if v != f32(0.0):
            cute.arch.atomic_add(pDk.iterator + ((st + jj) * DI + tidx), v, scope="gpu")


# ---------------------------------------------------------------------------
@cute.jit
def _launch_fast(
    pQ: cute.Tensor,
    pK: cute.Tensor,
    pQi: cute.Tensor,
    pKi: cute.Tensor,
    pLse: cute.Tensor,
    mSt: cute.Tensor,
    mEn: cute.Tensor,
    mVa: cute.Tensor,
    mDen: cute.Tensor,
    mLoss: cute.Tensor,
    pDq: cute.Tensor,
    pDk: cute.Tensor,
    pStats: cute.Tensor,
    tq: i32,
    tk: i32,
    qs: i32,
    ks: i32,
    ntile: i32,
    HQ: cutlass.Constexpr,
    NC: cutlass.Constexpr,
    NSPLIT: cutlass.Constexpr,
    START: cutlass.Constexpr,
    STOP: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    gQi = cute.make_tensor(pQi.iterator, cute.make_layout((tq, DI8, 8), stride=(DI, 8, 1)))
    gKi = cute.make_tensor(pKi.iterator, cute.make_layout((tk, DI8, 8), stride=(DI, 8, 1)))
    gLse = cute.make_tensor(pLse.iterator, cute.make_layout((tq, HQ), stride=(HQ, 1)))

    # warp-level mma.sync tiled MMA (student / indexer GEMM)
    op = cute.nvgpu.warp.MmaF16BF16Op(bf16, f32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(op, cute.make_layout((2, 4, 1)), permutation_mnk=(BQ, 64, 16))
    # dk GEMM: C = (64 keys, 128 dims) over 8 warps arranged 4 x 2.
    mma_k = cute.make_tiled_mma(op, cute.make_layout((4, 2, 1)))

    # tcgen05 tiled MMA (teacher score GEMM): D[BK, N] = K[BK, DQ] @ Q[DQ, N]
    # with N = G*HQ, i.e. G whole queries batched into one UMMA.
    G = _qgroup(HQ)
    NB = G * HQ
    GH = G // NC
    NBC = NB // NC
    BKM = BK * NC
    CG = tcgen05.CtaGroup.TWO if NC == 2 else tcgen05.CtaGroup.ONE
    mma_tiler = (BKM, NB, DQ)
    tc_op = tcgen05.MmaF16BF16Op(
        bf16,
        f32,
        (BKM, NB, 16),
        CG,
        tcgen05.OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
    )
    tc_mma = cute.make_tiled_mma(tc_op)
    # Per-query, per-K-chunk TMA tile: (HQ, 64).
    tiler1 = (BK, HQ, 64)
    tc_op1 = tcgen05.MmaF16BF16Op(
        bf16,
        f32,
        (BK, HQ, 16),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
    )
    tc_mma1 = cute.make_tiled_mma(tc_op1)

    # A: teacher keys (M = key, K = dim).  B: teacher queries (N = head,
    # K = 64-dim chunk, L = (chunk, query token)).
    aT = cute.make_tensor(pK.iterator, cute.make_layout((tk, DQ), stride=(ks, 1)))
    bT = cute.make_tensor(
        pQ.iterator,
        cute.make_layout((HQ, 64, (KCH, tq)), stride=(DQ, 1, (64, qs))),
    )
    a_smem_layout = sm100_utils.make_smem_layout_a(tc_mma, mma_tiler, bf16, ASTAGE)
    b1_layout = sm100_utils.make_smem_layout_b(tc_mma1, tiler1, bf16, KCH * GH * BSTAGE)
    # 2-SM copies: the complete_tx of BOTH CTAs is routed to the leader's
    # mbarrier, which is what lets a single leader-side wait cover the pair.
    tma_op = cpasync.CopyBulkTensorTileG2SOp(CG)
    cta_layout_vmnk = cute.tiled_divide(cute.make_layout((NC, 1, 1)), (tc_mma.thr_id,))
    tma_a = cute.nvgpu.make_tiled_tma_atom_A(tma_op, aT, a_smem_layout, mma_tiler, tc_mma, cta_layout_vmnk.shape)
    tma_b = cute.nvgpu.make_tiled_tma_atom_B(tma_op, bT, b1_layout, tiler1, tc_mma1, cta_layout_vmnk.shape)

    smem_bytes = (
        BK * DQ * ASTAGE * 2
        + NBC * DQ * BSTAGE * 2
        + BQ * PI * 2
        + BQ * PK * 4
        + HQ * BQ * 4
        + 2 * BQ * 4
        + 3 * NWARP * BQ * 4
        + 2 * BQ * 4
        + NWARP * 4
        + NBAR * 8
    )
    # The TMEM allocator parks its 32-bit result slot at an aligned address
    # above the SMEM high-water mark, so the launch must reserve headroom past
    # the last named allocation (clamped to the 227 KiB per-CTA maximum).
    smem_bytes = min(smem_bytes + 3072, 232064)
    kn = _fused(
        gQi,
        gKi,
        gLse,
        mSt,
        mEn,
        mVa,
        mDen,
        mLoss,
        pDq,
        pDk,
        pStats,
        tiled_mma,
        mma_k,
        tc_mma,
        tc_mma1,
        tma_a,
        tma_b,
        tq,
        tk,
        ntile,
        HQ,
        NC,
        NSPLIT,
        START,
        STOP,
    )
    if cutlass.const_expr(NC == 2):
        kn.launch(
            grid=(NC * NSPLIT, ntile, 1),
            block=(NTHR, 1, 1),
            cluster=(NC, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )
    else:
        kn.launch(
            grid=(NSPLIT, ntile, 1),
            block=(NTHR, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )


@cute.jit
def _launch_oracle(
    pQ: cute.Tensor,
    pK: cute.Tensor,
    pQi: cute.Tensor,
    pKi: cute.Tensor,
    pLse: cute.Tensor,
    mSt: cute.Tensor,
    mEn: cute.Tensor,
    mVa: cute.Tensor,
    mDen: cute.Tensor,
    mLoss: cute.Tensor,
    pDq: cute.Tensor,
    pDk: cute.Tensor,
    tq: i32,
    tk: i32,
    qs: i32,
    ks: i32,
    HQ: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    gQ = cute.make_tensor(pQ.iterator, cute.make_layout((tq, HQ, DQ), stride=(qs, DQ, 1)))
    gK = cute.make_tensor(pK.iterator, cute.make_layout((tk, DQ), stride=(ks, 1)))
    gQi = cute.make_tensor(pQi.iterator, cute.make_layout((tq, DI), stride=(DI, 1)))
    gKi = cute.make_tensor(pKi.iterator, cute.make_layout((tk, DI), stride=(DI, 1)))
    gLse = cute.make_tensor(pLse.iterator, cute.make_layout((tq, HQ), stride=(HQ, 1)))
    _oracle(gQ, gK, gQi, gKi, gLse, mSt, mEn, mVa, mDen, mLoss, pDq, pDk, tq, tk, HQ).launch(
        grid=(tq, 1, 1),
        block=(OTHR, 1, 1),
        smem=SMEM_ORACLE,
        stream=stream,
    )


_CACHE = {}


def _dyn(t, align=16):
    return from_dlpack(t, assumed_align=align).mark_compact_shape_dynamic(mode=0, divisibility=1)


@torch.no_grad()
def run(
    q,
    k,
    teacher_lse,
    q_idx,
    k_idx,
    key_starts,
    key_ends,
    query_valid,
    loss_denominator,
    loss,
    dq_idx,
    dk_idx,
):
    TQ, HQ, _ = q.shape
    TK = k.shape[0]
    qs = q.stride(0)
    ks = k.stride(0)

    nq = (TQ - 1) * qs + HQ * DQ
    nk = (TK - 1) * ks + DQ
    qf = torch.as_strided(q, (nq,), (1,), q.storage_offset())
    kf = torch.as_strided(k, (nk,), (1,), k.storage_offset())
    qif = q_idx.reshape(TQ * DI)
    kif = k_idx.reshape(TK * DI)
    lsf = teacher_lse.reshape(TQ * HQ)
    dqf = dq_idx.reshape(TQ * DI)
    dkf = dk_idx.reshape(TK * DI)
    va = query_valid.view(torch.uint8)
    den1 = loss_denominator.reshape(1)
    loss1 = loss.reshape(1)

    oracle = TQ <= 64 and TK <= OMAX
    ntile = (TQ + BQ - 1) // BQ
    nc = _ncta(HQ)
    nktile = (TK + BK * nc - 1) // (BK * nc)
    # One CTA per query tile underfills the machine whenever the query extent
    # is small next to the key extent, and packed/causal bounds make the
    # per-tile key ranges very uneven.  Split the key range far enough to fill
    # the grid; splitting also shrinks the set of query tiles live at once,
    # which keeps the teacher-Q tile each CTA re-reads per key tile in L2.
    # The cap is per-shape, not per-head: causal square bounds already spread
    # unequal key ranges across the query tiles, so a second split is all the
    # balance they need and every further split only adds redundant teacher-Q
    # re-reads, dq atomics and empty early-causal CTAs; the rectangular
    # post-all-gather shapes have uniformly long rows and need four splits to
    # fill the grid.  Each split keeps at least four key tiles so no split ever
    # runs empty.
    nsplit = 1
    if not oracle:
        if TQ == TK:
            cap = 2
        else:
            cap = 4
        nsplit = max(1, min(cap, nktile // 4, 2048 // max(1, ntile)))
    # NSPLIT is a compile-time argument and must be part of the code-cache key.
    key = (HQ, oracle, nsplit, nc)
    fn = _CACHE.get(key)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    args = (
        _dyn(qf),
        _dyn(kf),
        _dyn(qif),
        _dyn(kif),
        _dyn(lsf, 4),
        _dyn(key_starts, 4),
        _dyn(key_ends, 4),
        _dyn(va, 1),
        from_dlpack(den1, assumed_align=4),
        from_dlpack(loss1, assumed_align=4),
        _dyn(dqf, 4),
        _dyn(dkf, 4),
    )
    if oracle:
        loss.zero_()
        dk_idx.zero_()
        if fn is None:
            fn = cute.compile(
                _launch_oracle,
                *args,
                i32(TQ),
                i32(TK),
                i32(qs),
                i32(ks),
                HQ,
                make_fake_stream(),
            )
            _CACHE[key] = fn
        fn(qf, kf, qif, kif, lsf, key_starts, key_ends, va, den1, loss1, dqf, dkf, i32(TQ), i32(TK), i32(qs), i32(ks), stream)
    else:
        stats = torch.empty(3 * nc * nsplit * TQ, device=q.device, dtype=torch.float32)
        fast_args = (*args, _dyn(stats, 4))
        if fn is None:
            fn = (
                cute.compile(
                    _launch_fast,
                    *fast_args,
                    i32(TQ),
                    i32(TK),
                    i32(qs),
                    i32(ks),
                    i32(ntile),
                    HQ,
                    nc,
                    nsplit,
                    0,
                    1,
                    make_fake_stream(),
                ),
                cute.compile(
                    _launch_fast,
                    *fast_args,
                    i32(TQ),
                    i32(TK),
                    i32(qs),
                    i32(ks),
                    i32(ntile),
                    HQ,
                    nc,
                    nsplit,
                    1,
                    2,
                    make_fake_stream(),
                ),
            )
            _CACHE[key] = fn
        call_args = (
            qf,
            kf,
            qif,
            kif,
            lsf,
            key_starts,
            key_ends,
            va,
            den1,
            loss1,
            dqf,
            dkf,
            stats,
            i32(TQ),
            i32(TK),
            i32(qs),
            i32(ks),
            i32(ntile),
            stream,
        )
        fn[0](*call_args)
        fn[1](*call_args)
