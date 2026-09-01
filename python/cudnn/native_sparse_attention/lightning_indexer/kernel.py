# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact CuTe DSL decode kernels for the MiniMax Lightning Indexer."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import SmemAllocator

BLOCK_SIZE = 128
TOP_K = 16
HEAD_DIM = 128


@cute.kernel
def _dec(
    mQ: cute.Tensor,  # (B, 4, 16, 8) bf16
    mK: cute.Tensor,  # (B, S_K, 16, 8) bf16
    mW: cute.Tensor,  # (B, 4, NB) i32 monotone score keys
    mC: cute.Tensor,  # (B,) i32 arrival counters
    mO: cute.Tensor,  # (B, 4, 16) i32
    mN: cute.Tensor,  # (B, 4) i32 valid output counts
    mP: cute.Tensor,  # (B, 1) i64 explicit query positions
    S_K: cutlass.Constexpr,
    NMAX: cutlass.Constexpr,  # candidate blocks below the envelope's last block
    NB: cutlass.Constexpr,
    NS: cutlass.Constexpr,  # CTAs per batch
    KPT: cutlass.Constexpr,  # key blocks handled per CTA
    HD: cutlass.Constexpr,  # dims per SMEM slice (64 or 128)
    NGRP: cutlass.Constexpr,  # cp.async commit groups per SMEM slice
    HS: cutlass.Constexpr,  # CTAs the four heads are split across
):
    NT = 128
    HPC = 4 // HS  # heads per CTA
    BIG = cutlass.Int32(0x7FFFFFF)
    MINI = cutlass.Int32(-0x7FFFFFFF)
    tidx, _, _ = cute.arch.thread_idx()
    sp, hy, bz = cute.arch.block_idx()
    h0 = hy * HPC
    lane = cute.arch.lane_idx()
    wi = cute.arch.warp_idx()
    position = mP[bz, 0]
    valid_position = (position >= cutlass.Int64(0)) & (position < cutlass.Int64(S_K))
    cur_blk = cutlass.Int32(position // BLOCK_SIZE)

    LD = HD + 8  # padded slice stride -> conflict-free 128-bit LDS
    NC = HD // 8  # 8-element chunks per slice
    NG = 128 // HD  # slices per key row
    NCH = NC // NGRP  # chunks staged per commit group
    alloc = SmemAllocator()
    sK = alloc.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((KPT * 128, (8, NC)), stride=(LD, (1, 8))),
        byte_alignment=128,
    )
    # Q staged as BF16, not FP32: every thread re-reads the same q chunk for all
    # four heads, so these broadcasts are the kernel's dominant L1 consumer
    # (NCU: L1/TEX 78%).  BF16 halves the LDS instruction count for them; the
    # extra cvt lands on the ALU, which is only ~49% issued.
    sQ = alloc.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((HPC, (8, 16)), stride=(128, (1, 8))),
        byte_alignment=128,
    )
    sR = alloc.allocate_tensor(
        cutlass.Int32,
        cute.make_layout((4, KPT * HPC), stride=(KPT * HPC, 1)),
        byte_alignment=16,
    )
    sF = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(4), byte_alignment=16)

    cpa = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
    # ---- stage Q (4 heads x 128 dims) ----
    # cp.async, not ld.global+sts: a blocking load here put a full HBM round trip
    # in front of the first key cp.async, serialising the CTA's two cold fetches.
    # Issued into the same commit group as the first key half, both now fly
    # together and the CTA pays one memory latency instead of two.
    if tidx < 16 * HPC:  # HPC heads x 16 dim-chunks of 8
        cute.copy(
            cpa,
            mQ[bz, h0 + tidx // 16, tidx % 16, None],
            sQ[tidx // 16, (None, tidx % 16)],
        )

    kbase = sp * (KPT * BLOCK_SIZE)
    klast = cutlass.Int32(NMAX * BLOCK_SIZE - 1)

    acc = cute.make_rmem_tensor(cute.make_layout(KPT * HPC), cutlass.Float32)
    for i in cutlass.range_constexpr(KPT * HPC):
        acc[i] = cutlass.Float32(0.0)
    qv = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
    qsr = cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)
    kb = cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)
    kf = cute.make_rmem_tensor(cute.make_layout((KPT, 8)), cutlass.Float32)

    # Small grids (B=1) leave most SMs idle, so CTA residency is worthless and
    # the only thing that matters is the dependent-latency chain: stage the whole
    # 128-dim key row (NG==1) and split its issue into NGRP commit groups so the
    # dot product starts on the first arriving slice.  Large grids keep the
    # 64-dim slice: there the halved SMEM footprint buys CTAs per SM, which is
    # worth more than the extra round trip.
    for g in cutlass.range_constexpr(NG):
        for gg in cutlass.range_constexpr(NGRP):
            for s in cutlass.range_constexpr(KPT * NCH):
                ci = tidx + s * NT
                gr = kbase + ci // NCH
                gr = gr if gr < klast else klast
                cute.copy(
                    cpa,
                    mK[bz, gr, g * NC + gg * NCH + ci % NCH, None],
                    sK[ci // NCH, (None, gg * NCH + ci % NCH)],
                )
            cute.arch.cp_async_commit_group()
        for gg in cutlass.range_constexpr(NGRP):
            cute.arch.cp_async_wait_group(NGRP - 1 - gg)
            cute.arch.barrier()
            # head loop outside the key loop: one q chunk load + convert now
            # feeds every key this thread owns instead of being redone per key
            for j in cutlass.range_constexpr(gg * NCH, (gg + 1) * NCH):
                for kk in cutlass.range_constexpr(KPT):
                    cute.autovec_copy(sK[tidx + kk * 128, (None, j)], kb)
                    kf[kk, None].store(kb.load().to(cutlass.Float32))
                for h in cutlass.range_constexpr(HPC):
                    cute.autovec_copy(sQ[h, (None, g * NC + j)], qsr)
                    qv.store(qsr.load().to(cutlass.Float32))
                    for kk in cutlass.range_constexpr(KPT):
                        a = acc[kk * HPC + h]
                        for e in cutlass.range_constexpr(8):
                            a = a + qv[e] * kf[kk, e]
                        acc[kk * HPC + h] = a
        if cutlass.const_expr(NG > 1):
            cute.arch.barrier()

    # ---- monotone int key + block max reduction ----
    kr = cute.make_rmem_tensor(cute.make_layout(KPT * HPC), cutlass.Int32)
    kr.store(acc.load().bitcast(cutlass.Int32))
    for i in cutlass.range_constexpr(KPT * HPC):
        v = kr[i]
        v = v if v >= 0 else (v ^ cutlass.Int32(0x7FFFFFFF))
        kr[i] = cute.arch.warp_redux_sync(v, "max")
    if lane == 0:
        for i in cutlass.range_constexpr(KPT * HPC):
            sR[wi, i] = kr[i]
    cute.arch.barrier()
    if tidx < KPT * HPC:
        m = sR[0, tidx]
        for w in cutlass.range_constexpr(3):
            o = sR[w + 1, tidx]
            m = o if o > m else m
        nb = sp * KPT + (tidx // HPC)
        if nb < NMAX:
            mW[bz, h0 + (tidx % HPC), nb] = m
    cute.arch.barrier()

    # ---- recipe D: last arriving CTA merges ----
    if tidx == 0:
        old = cute.arch.atomic_add(mC.iterator + bz, cutlass.Int32(1), scope="gpu", sem="acq_rel")
        sF[0] = cutlass.Int32(1) if old == NS * HS - 1 else cutlass.Int32(0)
    cute.arch.barrier()

    if sF[0] == 1:
        # The current stream cannot begin the next invocation until this kernel
        # exits, so reset now and overlap the atomic with the top-k merge.
        if tidx == 0:
            cute.arch.atomic_exch(mC.iterator + bz, cutlass.Int32(0), scope="gpu")
        SLOTS = (NB + 31) // 32
        vals = cute.make_rmem_tensor(cute.make_layout(SLOTS), cutlass.Int32)
        ids = cute.make_rmem_tensor(cute.make_layout(SLOTS), cutlass.Int32)
        for e in cutlass.range_constexpr(SLOTS):
            nn = lane + 32 * e
            ok = valid_position & (nn < cur_blk) & (nn < NMAX)
            ids[e] = nn if ok else BIG
            vals[e] = mW[bz, wi, nn if ok else 0] if ok else MINI
        # Sort each lane's small queue once.  The global top-15 is then a
        # 32-way merge of queue heads instead of 15 repeated local scans.
        for ii in cutlass.range_constexpr(SLOTS):
            for jj in cutlass.range_constexpr(SLOTS - 1 - ii):
                av = vals[jj]
                ai = ids[jj]
                bv = vals[jj + 1]
                bi = ids[jj + 1]
                take = (bv > av) | ((bv == av) & (bi < ai))
                vals[jj] = bv if take else av
                ids[jj] = bi if take else ai
                vals[jj + 1] = av if take else bv
                ids[jj + 1] = ai if take else bi
        if lane == 0:
            mO[bz, wi, 0] = cur_blk if valid_position else cutlass.Int32(-1)
            count = cur_blk + cutlass.Int32(1)
            count = count if count < TOP_K else cutlass.Int32(TOP_K)
            mN[bz, wi] = count if valid_position else cutlass.Int32(0)
        for it in cutlass.range_constexpr(15):
            lv = vals[0]
            gv = cute.arch.warp_redux_sync(lv, "max")
            li = ids[0] if lv == gv else BIG
            gi = cute.arch.warp_redux_sync(li, "min")
            if lane == 0:
                mO[bz, wi, it + 1] = gi if gi < BIG else cutlass.Int32(-1)
            if ids[0] == gi:
                for e in cutlass.range_constexpr(SLOTS - 1):
                    vals[e] = vals[e + 1]
                    ids[e] = ids[e + 1]
                vals[SLOTS - 1] = MINI
                ids[SLOTS - 1] = BIG


@cute.kernel
def _short(
    mP: cute.Tensor,
    mO: cute.Tensor,
    mN: cute.Tensor,
    S_K: cutlass.Constexpr,
):
    """Emit all visible blocks when Top-K covers the complete causal prefix."""
    tid, _, _ = cute.arch.thread_idx()
    bz, _, _ = cute.arch.block_idx()
    if tid < 4:
        position = mP[bz, 0]
        valid_position = (position >= cutlass.Int64(0)) & (position < cutlass.Int64(S_K))
        cur_blk = cutlass.Int32(position // BLOCK_SIZE)
        count = cur_blk + cutlass.Int32(1)
        count = count if count < TOP_K else cutlass.Int32(TOP_K)
        mN[bz, tid] = count if valid_position else cutlass.Int32(0)
        mO[bz, tid, 0] = cur_blk if valid_position else cutlass.Int32(-1)
        for slot in cutlass.range_constexpr(1, TOP_K):
            block = cutlass.Int32(slot - 1)
            mO[bz, tid, slot] = block if valid_position & (block < cur_blk) else cutlass.Int32(-1)


@cute.jit
def decode_host(
    pq: cutlass.Int64,
    pk: cutlass.Int64,
    pp: cutlass.Int64,
    po: cutlass.Int64,
    pn: cutlass.Int64,
    pw: cutlass.Int64,
    pc: cutlass.Int64,
    stream: cuda.CUstream,
    B: cutlass.Constexpr,
    S_K: cutlass.Constexpr,
    P_B_STRIDE: cutlass.Constexpr,
    NMAX: cutlass.Constexpr,
    NB: cutlass.Constexpr,
    NS: cutlass.Constexpr,
    KPT: cutlass.Constexpr,
    HD: cutlass.Constexpr,
    NGRP: cutlass.Constexpr,
    HS: cutlass.Constexpr,
):
    gQ = cute.make_tensor(
        cute.make_ptr(cutlass.BFloat16, pq, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((B, 4, 16, 8), stride=(512, 128, 8, 1))
    )
    gK = cute.make_tensor(
        cute.make_ptr(cutlass.BFloat16, pk, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((B, S_K, 16, 8), stride=(S_K * 128, 128, 8, 1))
    )
    gP = cute.make_tensor(
        cute.make_ptr(cutlass.Int64, pp, cute.AddressSpace.gmem, assumed_align=8),
        cute.make_layout((B, 1), stride=(P_B_STRIDE, 1)),
    )
    gO = cute.make_tensor(cute.make_ptr(cutlass.Int32, po, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((B, 4, 16), stride=(64, 16, 1)))
    gN = cute.make_tensor(cute.make_ptr(cutlass.Int32, pn, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((B, 4), stride=(4, 1)))
    gW = cute.make_tensor(cute.make_ptr(cutlass.Int32, pw, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((B, 4, NB), stride=(4 * NB, NB, 1)))
    gC = cute.make_tensor(cute.make_ptr(cutlass.Int32, pc, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout(B))
    _dec(gQ, gK, gW, gC, gO, gN, gP, S_K, NMAX, NB, NS, KPT, HD, NGRP, HS).launch(grid=(NS, HS, B), block=(128, 1, 1), stream=stream)


@cute.jit
def short_host(
    pp: cutlass.Int64,
    po: cutlass.Int64,
    pn: cutlass.Int64,
    stream: cuda.CUstream,
    B: cutlass.Constexpr,
    S_K: cutlass.Constexpr,
    P_B_STRIDE: cutlass.Constexpr,
):
    gP = cute.make_tensor(
        cute.make_ptr(cutlass.Int64, pp, cute.AddressSpace.gmem, assumed_align=8),
        cute.make_layout((B, 1), stride=(P_B_STRIDE, 1)),
    )
    gO = cute.make_tensor(cute.make_ptr(cutlass.Int32, po, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((B, 4, 16), stride=(64, 16, 1)))
    gN = cute.make_tensor(cute.make_ptr(cutlass.Int32, pn, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((B, 4), stride=(4, 1)))
    _short(gP, gO, gN, S_K).launch(grid=(B, 1, 1), block=(32, 1, 1), stream=stream)
