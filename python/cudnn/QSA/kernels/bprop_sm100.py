# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen Sparse Attention (QSA) backward for SM100 / SM103, written with the CuTe DSL primitives.

The launch sequence is a dense key-tile kernel for the queries below the crossover, and for the remaining queries a
query-major kernel (S, dP, dQ and the P / dS records of every selected block) followed by a key-major kernel that
accumulates dK / dV from those records. The selection arrives compacted per query (``ids`` block keys, ``counts``,
``first`` block key -> first token) with the inclusive key bounds ``lo`` / ``hi`` of ``cudnn.ops.qsa_backward``; keys
after the query are never attended. The dense kernel masks per token through a bitmap, so a block may start anywhere
and the bounds may cut it. A query past the dense prefix holds at most SPARSE_CAP blocks: a longer list contributes
nothing and its dQ is NaN. A nonfinite K / V value of a masked token adds exact zeros (P and dS are selected, the dQ
MMAs read a sanitized copy of K), one of a selected token poisons its queries.
"""

import math
from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cuda.bindings import driver as cuda_driver
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

from cudnn.api_base import ws_align

I32 = cutlass.Int32
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

# Geometry: GQ query heads per KV head, HKV in (1, 2) KV heads. The kernels take HKV as a compile-time parameter.
GQ = 12
D = 256
SPARSE_CAP = 513  # blocks per query past the dense prefix: top-512 complete blocks and the causal tail block
LOG2E = 1.4426950408889634

NTHREADS = 512
# Blocks per chunk fill 8 * CHB of the 128 MMA rows. With the K and V tiles copied on to TMEM, 12 blocks leave room
# for a 3-stage K ring. Rows past 8 * CHB alias the next region and are masked out. With one KV head every K^T atom
# of the dQ^T MMA holds 4 alien rows (the next 512 B, cancelled by zero dS^T rows), and 0 * NaN would poison every
# column: the softmax warps zero the rows of the chunk the MMA must not see, and the 512 B after a stage, which its
# last K-step reads the same way, never hold K or V (see the layout below).
CHB = 12
NKS = 3  # K stages, so K(c+1) streams in while S(c), dP(c) and dQ(c-1) run

# shared memory of the query-major kernel (bytes from a 1024-aligned base): [K0][pad][K1][dO][K2][Q][V][dS]
# A chunk slot is 8 MMA rows in 4 KB: one block of one query with two KV heads (row = 2 * token + head, one SW128 box)
# or the same block number of two consecutive queries with one KV head (row = 4 * query + token, two 2 KB boxes).
BLK_SMEM = 4096
SLAB = 1024
HALF_BLK = 2048
K_STAGE = CHB * BLK_SMEM  # blocks sit at the 8-row-group stride of the MMA operand
# A zero pad, dO and Q separate the K stages, so the alien rows of a stage's last K-step never hold the next chunk's
# K (a nonfinite block there would reach the previous chunk's pair). Every region stays 1024-aligned: a stage that
# starts 512 B into its swizzle span costs 5% on the kernel, and the budget holds one aligned pad, not three.
QSLAB = 3072
Q_BYTES = 4 * QSLAB  # [4 slabs][24 rows x 128 B] with SW128: rows 0-11 are KV head 0 / the first query of the pair
OFF_K = 0
GAP01 = 1024  # the zero pad between stages 0 and 1
GAP12 = Q_BYTES  # dO sits between stages 1 and 2
OFF_DO = OFF_K + 2 * K_STAGE + GAP01
OFF_Q = OFF_K + 3 * K_STAGE + GAP01 + GAP12
OFF_V = OFF_Q + Q_BYTES
# dS^T as [row groups][3 column chunks][8 rows x 16 B], the MN-major B operand of dQ^T. One KV head needs two row groups
# per slot (rows 4-7 of each stay zero and mask the other query's tokens that share the K^T atom).
OFF_DS = OFF_V + K_STAGE
DS_BYTES = 2 * CHB * 384
SMEM_BYTES = OFF_DS + DS_BYTES

# tensor memory columns of the query-major kernel
TM_S = 0  # S^T, double buffered at a 64-column pitch
TM_DP = 32  # dP^T, interleaved with S^T
TM_DQ = 128  # dQ^T halves at 128 and 160
TM_KA = 192  # the chunk's K tile as the TMEM A operand of S^T (128 columns of packed bf16 pairs)
TM_VA = 320  # the chunk's V tile as the TMEM A operand of dP^T
TM_COLS = 512
L2_EVICT_FIRST = 0x12F0000000000000  # createpolicy encoding of the L2 evict-first priority

Q_BOX_BYTES = 24 * 128


@cute.jit
def _wait(bar, parity):
    while not prims.mbarrier_try_wait_parity(bar, parity):
        pass


@cute.jit
def _st_async16(dst, v4, mbar):
    # st.async stores 16 B from registers into a peer CTA's shared memory and completes them on the peer's mbarrier.
    prims.inline_ptx(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}}, [{$r5}];",
        read_only_args=[dst, v4[0], v4[1], v4[2], v4[3], mbar],
    )


def _desc(addr, lbo, sbo, swz):
    return prims.Tcgen05SmemDesc.build(start_address=addr, leading_byte_offset=lbo, stride_byte_offset=sbo, layout=swz)


def _pad(hkv):
    # The tail block of the last queries may reduce exact zeros up to 3 tokens past T, so the sparse dK/dV
    # accumulators carry this many fp32 elements of slack.
    return 4 * hkv * D


def _rec_w(hkv):
    # int32 words per (query, block) record: P[4 tokens][12 * hkv heads] bf16, then dS in the same layout
    return 48 * hkv


@cute.jit
def _item_meta(p_off, j, pp):
    # (segment start, block count) of sparse query j in KV pass pp
    prow = p_off + (j * POS + pp)
    o0 = prow.load()
    return o0, (prow + 1).load() - o0


@cute.jit
def _pair_meta(p_off, j, pp, nsq, PAIR: cutlass.Constexpr[bool]):
    # the same for the pair partner j + 1 of a one-KV-head item, an empty segment when it does not exist
    o0 = I32(0)
    nblk = I32(0)
    if cutlass.const_expr(PAIR):
        if j + 1 < nsq:
            prow = p_off + ((j + 1) * POS + pp)
            o0 = prow.load()
            nblk = (prow + 1).load() - o0
    return o0, nblk


@cute.jit
def _lane_meta(o0_0, nblk_0, o0_1, nblk_1, h):
    # the item's chunk count follows its longer list, the lane follows the list of its query h
    nblk = nblk_0
    if nblk_1 > nblk:
        nblk = nblk_1
    o0 = o0_0
    nblk_l = nblk_0
    if h == 1:
        o0 = o0_1
        nblk_l = nblk_1
    return nblk, o0, nblk_l


@cute.jit
def _chunk_valid(nblk, c):
    # valid blocks of chunk c of a list of nblk blocks
    n = nblk - c * CHB
    if n > CHB:
        n = I32(CHB)
    if n < 0:
        n = I32(0)
    return n


@cute.jit
def _chunk_bytes(nblk_0, nblk_1, c, PAIR: cutlass.Constexpr[bool]):
    # transaction bytes of chunk c: every valid block of every query of the item
    if cutlass.const_expr(PAIR):
        return (_chunk_valid(nblk_0, c) + _chunk_valid(nblk_1, c)) * HALF_BLK
    else:
        return _chunk_valid(nblk_0, c) * BLK_SMEM


@cute.jit
def _koff(st):
    # byte offset of K stage st from OFF_K: the regions between the stages differ in size
    return st * K_STAGE + ((st + 1) >> 1) * GAP01 + (st >> 1) * GAP12


@cute.jit
def _load_block(tmap, dst, tok, bar, PAIR: cutlass.Constexpr[bool]):
    # the 4-token block starting at token tok (any alignment: sequences start anywhere in the packed buffer)
    if cutlass.const_expr(PAIR):
        prims.cp_async_bulk_tensor_shared_cta_global(dst, tmap.get_ptr(), (0, tok, 0), bar)
    else:
        prims.cp_async_bulk_tensor_shared_cta_global(dst, tmap.get_ptr(), (0, 0, tok, 0), bar)


@cute.jit
def _prefetch_block(tmap, tok, PAIR: cutlass.Constexpr[bool]):
    if cutlass.const_expr(PAIR):
        prims.cp_async_bulk_tensor_prefetch(tmap.get_ptr(), [0, tok, 0], [])
    else:
        prims.cp_async_bulk_tensor_prefetch(tmap.get_ptr(), [0, 0, tok, 0], [])


# ----------------------------------------------------------------------------- query-major kernel (dQ, P / dS records)


@cute.jit
def _load_lse4(lse: cute.Tensor, flat: I32, HQ: cutlass.Constexpr[int]):
    # four heads of one query from the native [Hq, T] LSE
    qi = flat // HQ
    head = flat % HQ
    values = []
    for i in cutlass.range_constexpr(4):
        values.append(lse[head + i, qi])
    return cutlass.Vector.from_elements(values, F32)


@cute.kernel
def qsa_bwd_sparse_query_kernel(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_do: cutlass.GridConstant[cuda.TensorMap],
    tma_k: cutlass.GridConstant[cuda.TensorMap],
    tma_v: cutlass.GridConstant[cuda.TensorMap],
    g_lse: cute.Tensor,
    g_delta: cute.Tensor,
    g_dq: cute.Tensor,
    g_sparse_blocks: cute.Tensor,
    g_sparse_pass_offsets: cute.Tensor,
    g_sparse_block_slots: cute.Tensor,
    g_pair_records: cute.Tensor,
    g_sparse_queries: cute.Tensor,
    g_lo: cute.Tensor,
    g_hi: cute.Tensor,
    g_sparse_count: cute.Tensor,
    nctas: I32,
    NP: I32,
    scale: F32,
    HKV: cutlass.Constexpr[int],
):
    # An item is one sparse query (two KV heads) or two consecutive ones of the compacted list (one KV head), see the
    # SMEM layout note.
    HQ = GQ * HKV
    PAIR = HKV == 1
    QSTEP = 2 if PAIR else 1
    REC_W = _rec_w(HKV)
    nsq = g_sparse_count.iterator.raw_ptr().load()
    p_sparse_queries = g_sparse_queries.iterator.raw_ptr()
    p_lo = g_lo.iterator.raw_ptr()
    p_hi = g_hi.iterator.raw_ptr()
    # Q and dO of an item are read once per KV pass, while the pass's K/V blocks are reused by many queries.
    hint_q = cutlass.Int64(L2_EVICT_FIRST)
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % 32

    smem = cutlass.Array(I32, SMEM_BYTES // 4, space=cutlass.AddressSpace.smem, alignment=1024)
    b_qf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_qe = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_kf = cutlass.Array(cutlass.Int64, NKS, space=cutlass.AddressSpace.smem, alignment=8)
    b_ke = cutlass.Array(cutlass.Int64, NKS, space=cutlass.AddressSpace.smem, alignment=8)
    b_vf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_ve = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_sf = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_pf = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_dqf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_dqe = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_sde = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_dsf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_ktf = cutlass.Array(cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8)
    b_kte = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_vtf = cutlass.Array(cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8)
    b_vte = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    # zero the whole operand region once, so padding rows and stale rows stay finite
    zero4 = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
    sbase = smem.data_ptr()
    for it in cutlass.range(tidx, SMEM_BYTES // 16, NTHREADS, unroll=1):
        (sbase + it * 4).store(zero4, alignment=16)

    if warp == 0:
        if prims.elect_sync():
            prims.prefetch_tensormap(tma_q.get_ptr())
            prims.prefetch_tensormap(tma_do.get_ptr())
            prims.prefetch_tensormap(tma_k.get_ptr())
            prims.prefetch_tensormap(tma_v.get_ptr())
            prims.mbarrier_init(b_qf, 1)
            prims.mbarrier_init(b_qe, 1)
            for s in cutlass.range_constexpr(NKS):
                prims.mbarrier_init(b_kf.subview(s), 1)
                prims.mbarrier_init(b_ke.subview(s), 1)
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_pf.subview(s), 4)
            prims.mbarrier_init(b_sf.subview(0), 1)
            prims.mbarrier_init(b_sf.subview(1), 1)
            prims.mbarrier_init(b_vf, 1)
            prims.mbarrier_init(b_ve, 8)
            prims.mbarrier_init(b_dqf, 1)
            prims.mbarrier_init(b_dqe, 4)
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_sde.subview(s), 4)
            prims.mbarrier_init(b_dsf, 1)
            for s in cutlass.range_constexpr(4):
                prims.mbarrier_init(b_ktf.subview(s), 8)
                prims.mbarrier_init(b_vtf.subview(s), 8)
            prims.mbarrier_init(b_kte, 1)
            prims.mbarrier_init(b_vte, 1)
    if warp == 2:
        prims.tcgen05_alloc(tmem_slot, TM_COLS, group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    prims.fence_mbarrier_init()
    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
    prims.barrier_cta_sync(0)

    tmem_raw = tmem_slot.load()
    tmem_col0 = tmem_raw & I32(0xFFFF)
    tmem_row0 = tmem_raw >> I32(16)

    p_delta = g_delta.iterator.raw_ptr()
    p_dq = g_dq.iterator.raw_ptr()
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()
    p_sparse_block_slots = g_sparse_block_slots.iterator.raw_ptr()
    p_pair_records = g_pair_records.iterator.raw_ptr()

    # warpgroup dispatch (all roles fit the 128-register budget of a 512-thread CTA)
    wg = warp // 4
    if wg == 0:
        # ================================================================ producer warp (TMA gathers)
        # Lane j < CHB owns block slot j of every chunk, one 4 KB box per block and tensor (one KV head: lane 2j + h
        # owns query h's block of slot j, a 2 KB box). Tokens past T are zero-filled by the TMA unit.
        if cutlass.const_expr(PAIR):
            jl = lane >> 1
            hl = lane & 1
            lane_ok = lane < 2 * CHB
        else:
            jl = lane
            hl = I32(0)
            lane_ok = lane < CHB
        if warp == 0:
            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                    o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                    o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                    nblk, o0_l, nblk_l = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, hl)
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((j + hl) * PART_W + o0_l)
                        nchunks = (nblk + (CHB - 1)) // CHB
                        _wait(b_qe, (nq & 1) ^ 1)
                        if prims.elect_sync():
                            # Q / dO rows of the item: the query's 24 head rows, or 12 of each query of the pair (the
                            # partner's rows land 1.5 KB into every slab)
                            nqq = I32(1)
                            if cutlass.const_expr(PAIR):
                                if j + 1 < nsq:
                                    nqq = I32(2)
                            prims.mbarrier_arrive_expect_tx(b_qf, nqq * (8 * Q_BOX_BYTES // QSTEP))
                            for h in cutlass.range_constexpr(QSTEP):
                                if h < nqq:
                                    qh = (p_sparse_queries + (j + h)).load()
                                    for s in cutlass.range_constexpr(4):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sbase + (OFF_Q + s * QSLAB + h * (QSLAB // 2)) // 4,
                                            tma_q.get_ptr(),
                                            (s * 64, 0, qh),
                                            b_qf,
                                            l2_cache_hint=hint_q,
                                        )
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sbase + (OFF_DO + s * QSLAB + h * (QSLAB // 2)) // 4,
                                            tma_do.get_ptr(),
                                            (s * 64, 0, qh),
                                            b_qf,
                                            l2_cache_hint=hint_q,
                                        )
                        # this lane's block slot of the next chunk, loaded one chunk ahead with the column clamped to a
                        # valid one (entries past nsel are replaced by the tail block below)
                        m0 = jl
                        if m0 >= nblk_l:
                            m0 = I32(0)
                        xnext = (lrow + m0).load()
                        for c in cutlass.range(nchunks, unroll=1):
                            cgc = gc + c
                            st = cgc % NKS
                            kpar = (cgc // NKS) & 1
                            xcur = xnext
                            if c + 1 < nchunks:
                                m1 = (c + 1) * CHB + jl
                                if m1 >= nblk_l:
                                    m1 = I32(0)
                                xnext = (lrow + m1).load()
                            m = c * CHB + jl
                            ok = (m < nblk_l) & lane_ok
                            tbytes = _chunk_bytes(nblk_0, nblk_1, c, PAIR)
                            # ---- K
                            _wait(b_ke.subview(st), kpar ^ 1)
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(b_kf.subview(st), tbytes)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if ok:
                                _load_block(tma_k, sbase + (OFF_K + _koff(st) + jl * BLK_SMEM + hl * HALF_BLK) // 4, xcur, b_kf.subview(st), PAIR)
                            if c + 1 < nchunks:
                                okn = ((c + 1) * CHB + jl < nblk_l) & lane_ok
                                if okn:
                                    _prefetch_block(tma_k, xnext, PAIR)
                        gc = gc + nchunks
                        nq = nq + 1

        # V has only one SMEM stage.  Giving it to the otherwise-idle warp 1 prevents its empty-barrier wait from
        # serialising warp 0's three-stage K producer, so K can run up to the actual ring depth ahead of the MMA.
        if warp == 1:
            gc = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                    o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                    o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                    nblk, o0_l, nblk_l = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, hl)
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((j + hl) * PART_W + o0_l)
                        nchunks = (nblk + (CHB - 1)) // CHB
                        m0 = jl
                        if m0 >= nblk_l:
                            m0 = I32(0)
                        xnext = (lrow + m0).load()
                        for c in cutlass.range(nchunks, unroll=1):
                            cgc = gc + c
                            xcur = xnext
                            if c + 1 < nchunks:
                                m1 = (c + 1) * CHB + jl
                                if m1 >= nblk_l:
                                    m1 = I32(0)
                                xnext = (lrow + m1).load()
                            m = c * CHB + jl
                            ok = (m < nblk_l) & lane_ok
                            tbytes = _chunk_bytes(nblk_0, nblk_1, c, PAIR)
                            _wait(b_ve, (cgc & 1) ^ 1)
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(b_vf, tbytes)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if ok:
                                _load_block(tma_v, sbase + (OFF_V + jl * BLK_SMEM + hl * HALF_BLK) // 4, xcur, b_vf, PAIR)
                            if c + 1 < nchunks:
                                okn = ((c + 1) * CHB + jl < nblk_l) & lane_ok
                                if okn:
                                    _prefetch_block(tma_v, xnext, PAIR)
                        gc = gc + nchunks

        # ================================================================ score MMA warp (S^T and dP^T)
        if warp == 2:
            idesc_s = prims.Tcgen05InstrDesc.build(
                c_dtype=F32,
                a_dtype=BF16,
                b_dtype=BF16,
                n_dim=24,
                m_dim=128,
                a_major=0,
                b_major=0,
            )
            SW128 = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
            # Q/dO as the K-major B operand of S^T / dP^T
            d_q_kmaj = _desc(sbase + OFF_Q // 4, 16, 1024, SW128)
            d_do_kmaj = _desc(sbase + OFF_DO // 4, 16, 1024, SW128)

            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                    o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                    o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                    nblk, _, _ = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, I32(0))
                    if nblk > 0:
                        nchunks = (nblk + (CHB - 1)) // CHB
                        _wait(b_qf, nq & 1)
                        for c in cutlass.range(nchunks + 1, unroll=1):
                            # S(c) and dP(c) only need softmax(c-1) to have read S/dP (signalled early through sde),
                            # so softmax(c) overlaps dQ(c-1) and S/dP(c+1) on the tensor pipe.
                            cgc = gc + c
                            st = cgc % NKS
                            if c < nchunks:
                                _wait(b_kf.subview(st), (cgc // NKS) & 1)
                                # softmax(c-1) has loaded S/dP into registers
                                _wait(b_sde.subview(cgc & 1), ((cgc >> 1) & 1) ^ 1)
                                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                                # A = K(c) from TMEM (copied there slab by slab by warps 8-15), because the SMEM operand
                                # fetch, not the math, bounds these small-N MMAs
                                for pub in cutlass.range_constexpr(2):
                                    _wait(b_ktf.subview(pub), cgc & 1)
                                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                                    stop = 1 if pub == 0 else 4
                                    for sl4 in cutlass.range(pub, stop, unroll=1):
                                        for k4 in cutlass.range_constexpr(4):
                                            if prims.elect_sync():
                                                prims.tcgen05_mma(
                                                    prims.Tcgen05MMAKind.F16,
                                                    prims.CTAGroup.CTA_1,
                                                    prims.make_tmem_ptr(tmem_raw + (TM_S + (cgc & 1) * 64), F32),
                                                    prims.make_tmem_ptr(tmem_raw + (TM_KA + sl4 * 32 + k4 * 8), BF16),
                                                    d_q_kmaj.advance_start_address(sl4 * QSLAB + k4 * 32),
                                                    idesc_s,
                                                    (sl4 * 4 + k4) > 0,
                                                )
                                if prims.elect_sync():
                                    prims.tcgen05_commit(b_kte, group=prims.CTAGroup.CTA_1)
                            if c < nchunks:
                                # A = V(c) from TMEM (copied by warps 8-15, which also release the V SMEM stage)
                                for pub in cutlass.range_constexpr(2):
                                    _wait(b_vtf.subview(pub), cgc & 1)
                                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                                    stop = 1 if pub == 0 else 4
                                    for sl4 in cutlass.range(pub, stop, unroll=1):
                                        for k4 in cutlass.range_constexpr(4):
                                            if prims.elect_sync():
                                                prims.tcgen05_mma(
                                                    prims.Tcgen05MMAKind.F16,
                                                    prims.CTAGroup.CTA_1,
                                                    prims.make_tmem_ptr(tmem_raw + (TM_DP + (cgc & 1) * 64), F32),
                                                    prims.make_tmem_ptr(tmem_raw + (TM_VA + sl4 * 32 + k4 * 8), BF16),
                                                    d_do_kmaj.advance_start_address(sl4 * QSLAB + k4 * 32),
                                                    idesc_s,
                                                    (sl4 * 4 + k4) > 0,
                                                )
                                if prims.elect_sync():
                                    prims.tcgen05_commit(b_vte, group=prims.CTAGroup.CTA_1)
                                    prims.tcgen05_commit(b_sf.subview(cgc & 1), group=prims.CTAGroup.CTA_1)
                                    if c == nchunks - 1:
                                        prims.tcgen05_commit(b_qe, group=prims.CTAGroup.CTA_1)
                        gc = gc + nchunks
                        nq = nq + 1

        # ================================================================ dQ MMA warp
        # A second MMA-issuing warp: dQ(c) is issued as soon as softmax(c) has published dS^T, and S/dP of later
        # chunks no longer queue behind it in warp 2 (tcgen05.commit tracks the MMAs of the issuing thread only).
        if warp == 3:
            idesc_q3 = prims.Tcgen05InstrDesc.build(
                c_dtype=F32,
                a_dtype=BF16,
                b_dtype=BF16,
                n_dim=24,
                m_dim=128,
                a_major=1,
                b_major=1,
            )
            # K tile read as K^T (MN-major SW128) and dS^T (MN-major, no swizzle), as in warp 2. With one KV head a
            # K-step is one slot, whose atoms each hold 4 alien rows of the next slab (zero dS^T rows), 512 B apart.
            if cutlass.const_expr(PAIR):
                d_k_mn3 = _desc(sbase + OFF_K // 4, 512, HALF_BLK, prims.Tcgen05SmemSwizzle.SWIZZLE_128B)
            else:
                d_k_mn3 = _desc(sbase + OFF_K // 4, SLAB, BLK_SMEM, prims.Tcgen05SmemSwizzle.SWIZZLE_128B)
            d_ds_mn3 = _desc(sbase + OFF_DS // 4, 384, 128, prims.Tcgen05SmemSwizzle.NONE)
            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                    o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                    o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                    nblk, _, _ = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, I32(0))
                    if nblk > 0:
                        nchunks = (nblk + (CHB - 1)) // CHB
                        for c in cutlass.range(nchunks, unroll=1):
                            cgb = gc + c
                            pb = cgb & 1
                            kb = cgb % NKS
                            nv3 = _chunk_valid(nblk, c)
                            # softmax(c) has written dS^T to SMEM
                            _wait(b_pf.subview(pb), (cgb >> 1) & 1)
                            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            if c == 0:
                                # the previous item's dQ has been drained from TMEM
                                _wait(b_dqe, (nq & 1) ^ 1)
                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            acc_q = c > 0  # the first chunk of an item initialises the dQ accumulator
                            dk_mn = d_k_mn3.advance_start_address(_koff(kb))
                            # K-steps over the chunk's valid slots only
                            if cutlass.const_expr(PAIR):
                                nkk = nv3
                            else:
                                nkk = (nv3 + 1) >> 1
                            for hh in cutlass.range_constexpr(2):
                                for kk in cutlass.range(nkk, unroll=1):
                                    if cutlass.const_expr(PAIR):
                                        a_k = dk_mn.advance_start_address(hh * 2 * 512 + kk * BLK_SMEM)
                                    else:
                                        a_k = dk_mn.advance_start_address(hh * 2 * SLAB + kk * 2 * BLK_SMEM)
                                    if prims.elect_sync():
                                        prims.tcgen05_mma(
                                            prims.Tcgen05MMAKind.F16,
                                            prims.CTAGroup.CTA_1,
                                            prims.make_tmem_ptr(tmem_raw + (TM_DQ + hh * 32), F32),
                                            a_k,
                                            d_ds_mn3.advance_start_address(kk * 768),
                                            idesc_q3,
                                            acc_q | (kk > 0),
                                        )
                            if prims.elect_sync():
                                prims.tcgen05_commit(b_ke.subview(kb), group=prims.CTAGroup.CTA_1)
                                prims.tcgen05_commit(b_dsf, group=prims.CTAGroup.CTA_1)
                                if c == nchunks - 1:
                                    prims.tcgen05_commit(b_dqf, group=prims.CTAGroup.CTA_1)
                            prims.bar_warp_sync(0xFFFFFFFF)
                        gc = gc + nchunks
                        nq = nq + 1

    elif wg == 1:
        # ================================================================ softmax warps (P, dS, records, dQ drain)
        qd = warp % 4
        row = qd * 32 + lane
        if cutlass.const_expr(PAIR):
            hrow = (lane >> 2) & 1  # query of the pair of this row
            trow = lane & 3  # token within block
        else:
            hrow = lane & 1  # kv head of this row
            trow = (lane & 7) >> 1  # token within block
        jrow = row >> 3  # block slot within chunk
        c_s = scale * F32(LOG2E)
        gc = I32(0)
        nq = I32(0)
        if cutlass.const_expr(PAIR):
            ds_off = (OFF_DS + (2 * jrow + hrow) * 384 + trow * 16) // 4
        else:
            ds_off = (OFF_DS + jrow * 384 + (row & 7) * 16) // 4
        for pp in cutlass.range(NP, unroll=1):
            for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                if cutlass.const_expr(PAIR):
                    nblk, o0, nblk_r = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, hrow)
                else:
                    nblk, o0, nblk_r = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, I32(0))
                if nblk > 0:
                    # this row's query: sparse query j, or j + hrow of the pair (a missing partner has no rows)
                    jr = j
                    row_ok = jrow < CHB
                    if cutlass.const_expr(PAIR):
                        jr = j + hrow
                        row_ok = row_ok & (jr < nsq)
                        if jr >= nsq:
                            jr = j
                    qr = (p_sparse_queries + jr).load()
                    if cutlass.const_expr(PAIR):
                        lbase = qr * HQ
                    else:
                        lbase = qr * HQ + hrow * GQ
                    # the inclusive key bounds of the query: a block's tokens outside them have zero P / dS
                    lo_q = (p_lo + qr).load()
                    hi_q = (p_hi + qr).load()
                    if hi_q > qr:
                        hi_q = qr
                    nchunks = (nblk + (CHB - 1)) // CHB
                    # record slot and first token of this row's block entry, loaded one chunk ahead
                    prow_pp = p_sparse_block_slots + (jr * PART_W + o0)
                    trow_pp = p_sparse_blocks + (jr * PART_W + o0)
                    mpf = jrow
                    if mpf >= nblk_r:
                        mpf = I32(0)
                    pnx = (prow_pp + mpf).load()
                    tnx = (trow_pp + mpf).load()
                    lse_a = _load_lse4(g_lse, lbase, HQ)
                    lse_b = _load_lse4(g_lse, lbase + 4, HQ)
                    lse_c = _load_lse4(g_lse, lbase + 8, HQ)
                    del_a = (p_delta + lbase).load(count=4, alignment=16)
                    del_b = (p_delta + lbase + 4).load(count=4, alignment=16)
                    del_c = (p_delta + lbase + 8).load(count=4, alignment=16)
                    # The 12-head softmax state stays at vector level across the chunks, built once per query.
                    lse_v = cutlass.Vector.from_elements(
                        [
                            lse_a[0],
                            lse_a[1],
                            lse_a[2],
                            lse_a[3],
                            lse_b[0],
                            lse_b[1],
                            lse_b[2],
                            lse_b[3],
                            lse_c[0],
                            lse_c[1],
                            lse_c[2],
                            lse_c[3],
                        ],
                        F32,
                    )
                    del_v = cutlass.Vector.from_elements(
                        [
                            del_a[0],
                            del_a[1],
                            del_a[2],
                            del_a[3],
                            del_b[0],
                            del_b[1],
                            del_b[2],
                            del_b[3],
                            del_c[0],
                            del_c[1],
                            del_c[2],
                            del_c[3],
                        ],
                        F32,
                    )
                    zero12 = cutlass.Vector.from_elements([F32(0.0)] * GQ, F32)
                    for c in cutlass.range(nchunks, unroll=1):
                        cgc = gc + c
                        st = cgc & 1
                        _wait(b_sf.subview(cgc & 1), (cgc >> 1) & 1)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        t_s = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_S + (cgc & 1) * 64).as_ptr(F32)
                        t_dp = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_DP + (cgc & 1) * 64).as_ptr(F32)
                        sv = prims.tcgen05_ld("32x32b", t_s, num=32)
                        dpv = prims.tcgen05_ld("32x32b", t_dp, num=32)
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.bar_warp_sync(0xFFFFFFFF)
                        if prims.elect_sync():
                            prims.mbarrier_arrive(b_sde.subview(cgc & 1))

                        m = c * CHB + jrow
                        valid = (m < nblk_r) & row_ok
                        slot_ok = valid
                        pcur = pnx
                        tcur = tnx
                        if c + 1 < nchunks:
                            mpf = (c + 1) * CHB + jrow
                            if mpf >= nblk_r:
                                mpf = I32(0)
                            pnx = (prow_pp + mpf).load()
                            tnx = (trow_pp + mpf).load()
                        tk = tcur + trow
                        score_v = sv[0:GQ]
                        dot_v = dpv[0:GQ]
                        if hrow == 1:
                            score_v = sv[GQ : 2 * GQ]
                            dot_v = dpv[GQ : 2 * GQ]
                        # a nonfinite K row makes every head's score nonfinite, so one head tells
                        bad = valid & (~cute.math.isfinite(score_v[0]))
                        if cute.arch.vote_any_sync(bad):
                            # The dQ^T MMA reads this token's K row from SMEM: zero it when its K is nonfinite, so
                            # nothing nonfinite reaches the paired query through the zero dS^T rows (a masked finite
                            # row contributes 0 x finite). S^T read the TMEM copy, so the selecting query is still
                            # poisoned. Rows a slot keeps from an earlier chunk were zeroed the same way back then or
                            # are finite. The dS publish below fences and releases these stores. The warp vote keeps
                            # the branch uniform on the hot path.
                            if bad:
                                st_k = cgc % NKS
                                if cutlass.const_expr(PAIR):
                                    krow = sbase + (OFF_K + _koff(st_k) + jrow * BLK_SMEM + hrow * HALF_BLK + trow * 128) // 4
                                    for s4 in cutlass.range_constexpr(4):
                                        for u in cutlass.range_constexpr(8):
                                            (krow + (s4 * 512) // 4 + u * 4).store(zero4, alignment=16)
                                else:
                                    krow = sbase + (OFF_K + _koff(st_k) + jrow * BLK_SMEM + (2 * trow + hrow) * 128) // 4
                                    for s4 in cutlass.range_constexpr(4):
                                        for u in cutlass.range_constexpr(8):
                                            (krow + (s4 * SLAB) // 4 + u * 4).store(zero4, alignment=16)
                        valid = valid & (tk >= lo_q) & (tk <= hi_q)
                        prob_v = zero12
                        ds_v = zero12
                        if valid:
                            prob_v = cute.math.exp2(score_v * c_s - lse_v * F32(LOG2E), fastmath=True)
                            ds_v = prob_v * (dot_v - del_v) * scale

                        # dS^T in block-diagonal placement (own head group or own query, zeros elsewhere), built in
                        # packed bf16 words
                        prw = prob_v.to(BF16).bitcast(I32)
                        drw = ds_v.to(BF16).bitcast(I32)
                        dwords = []
                        for n in cutlass.range_constexpr(16):
                            dv_ = I32(0)
                            if cutlass.const_expr(n < 6):
                                if hrow == 0:
                                    dv_ = drw[n]
                            elif cutlass.const_expr(n < 12):
                                if hrow == 1:
                                    dv_ = drw[n - 6]
                            dwords.append(dv_)
                        dvec = cutlass.Vector.from_elements(dwords, I32)

                        # dQ(c-1) has finished reading the single dS^T SMEM buffer
                        _wait(b_dsf, (cgc & 1) ^ 1)
                        # dS^T row -> SMEM (MN-major core matrices: 3 n-chunks of 8 at 128 B). Slots past CHB have no
                        # row group.
                        if jrow < CHB:
                            for nc in cutlass.range_constexpr(3):
                                (sbase + ds_off + nc * 32).store(dvec[nc * 4 : nc * 4 + 4], alignment=16)
                        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                        prims.bar_warp_sync(0xFFFFFFFF)
                        if prims.elect_sync():
                            prims.mbarrier_arrive(b_pf.subview(st))
                        # The key-major kernel runs after this one, so dS is published to the dQ MMA first and the pair
                        # record streams out after the ready signal.
                        if slot_ok:
                            # this row's record word: token trow, kv head hrow (one record per query of a pair)
                            if cutlass.const_expr(PAIR):
                                roff = cutlass.Int64(pcur) * REC_W + cutlass.Int64(trow * 6)
                            else:
                                roff = cutlass.Int64(pcur) * REC_W + cutlass.Int64(trow * 12 + hrow * 6)
                            rbase = p_pair_records + roff
                            for u in cutlass.range_constexpr(3):
                                prims.store_ext(prw[2 * u : 2 * u + 2], rbase + 2 * u, cache_modifier=prims.StoreCacheModifier.CS)
                                prims.store_ext(drw[2 * u : 2 * u + 2], rbase + REC_W // 2 + 2 * u, cache_modifier=prims.StoreCacheModifier.CS)

                    # ---- dQ of the item: dQ^T[dim, 12 * query + head] from TMEM -> dq[query row, head, dim], which
                    # for two KV heads is dq[row, kv head * 12 + head, dim]
                    _wait(b_dqf, nq & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    dqv = []
                    for hh in cutlass.range_constexpr(2):
                        t_q = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_DQ + hh * 32).as_ptr(F32)
                        dqv.append(prims.tcgen05_ld("32x32b", t_q, num=32))
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_dqe)
                    for h in cutlass.range_constexpr(2 if PAIR else 1):
                        # The query's first non-empty KV pass stores dQ, later passes add to it. A query without blocks
                        # (or a missing pair partner) keeps the row the partition kernel wrote.
                        o0_h = o0_0
                        write_h = (p_sparse_pass_offsets + (j * POS + NP)).load() > 0
                        if cutlass.const_expr(h == 1):
                            o0_h = o0_1
                            write_h = False
                            if j + 1 < nsq:
                                write_h = (p_sparse_pass_offsets + ((j + 1) * POS + NP)).load() > 0
                        if write_h:
                            qrow = (p_sparse_queries + (j + h)).load()
                            for hh in cutlass.range_constexpr(2):
                                dim = hh * 128 + qd * 32 + lane
                                qb = p_dq + (qrow * (HQ * D) + dim)
                                if o0_h == 0:
                                    for n in cutlass.range_constexpr(GQ if PAIR else HQ):
                                        (qb + n * D).store(dqv[hh][h * GQ + n])
                                else:
                                    for n in cutlass.range_constexpr(GQ if PAIR else HQ):
                                        prims.red(
                                            prims.ReductionOp.ADD,
                                            prims.ReductionType.F32,
                                            qb + n * D,
                                            dqv[hh][h * GQ + n],
                                            mem_order=prims.MemOrder.RELAXED,
                                            mem_scope=prims.MemScope.GPU,
                                            cache_hint=cutlass.Int64(L2_EVICT_FIRST),
                                        )
                    gc = gc + nchunks
                    nq = nq + 1

    else:
        # ================================================================ K/V copy warps (SMEM -> TMEM)
        # The K and V tiles of every chunk become TMEM A operands, slab by slab so S can start on the first slab.
        # Lane = chunk row, column = packed bf16 dim pair. Warp pair hh of a lane quadrant moves 16-byte chunks
        # 4hh..4hh+3 of each 128-byte SW128 row segment (chunk cc of a row at 128 B index y of its atom sits at cc ^ y).
        qd = warp % 4
        hh = (warp - 8) >> 2
        r = qd * 32 + lane
        if cutlass.const_expr(PAIR):
            # row = 8 * slot + 4 * query + token at slot + query * 2 KB + slab * 512 B + token * 128 B
            yr = r & 3
            roff = (OFF_K + (r >> 3) * BLK_SMEM + ((r >> 2) & 1) * HALF_BLK + yr * 128) // 4
            slab_w = 512 // 4
        else:
            # row = 8 * block + 2 * token + head at block + slab * 1 KB + row * 128 B
            yr = r & 7
            roff = (OFF_K + (r >> 3) * BLK_SMEM + yr * 128) // 4
            slab_w = SLAB // 4
        # zero every quadrant's K / V operand rows once, so rows a chunk does not copy keep finite data
        zw = []
        for e in cutlass.range_constexpr(16):
            zw.append(I32(0))
        for zb in cutlass.range_constexpr(2):
            for sl4 in cutlass.range_constexpr(4):
                prims.tcgen05_st(
                    "32x32b",
                    prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_KA + zb * 128 + sl4 * 32 + hh * 16).as_ptr(I32),
                    cutlass.Vector.from_elements(zw, I32),
                )
        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
        gc = I32(0)
        for pp in cutlass.range(NP, unroll=1):
            for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                nblk, _, _ = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, I32(0))
                if nblk > 0:
                    nchunks = (nblk + (CHB - 1)) // CHB
                    for c in cutlass.range(nchunks, unroll=1):
                        cgc = gc + c
                        st = cgc % NKS
                        # the last chunk of an item may be partial, and quadrant qd holds slots 4qd..4qd+3
                        nval_c = _chunk_valid(nblk, c)
                        do_cp = qd * 4 < nval_c
                        # K(c): SMEM stage st -> TMEM (single buffer, free once S(c-1) has completed)
                        _wait(b_kf.subview(st), (cgc // NKS) & 1)
                        _wait(b_kte, (cgc & 1) ^ 1)
                        for sl4 in cutlass.range_constexpr(4):
                            # with one KV head a slab's rows start 512 B x slab in, so odd slabs sit in the upper half
                            # of their swizzle span
                            ys = yr
                            if cutlass.const_expr(PAIR and sl4 % 2 == 1):
                                ys = yr + 4
                            if do_cp:
                                off = roff + _koff(st) // 4 + sl4 * slab_w
                                wv = []
                                for i4 in cutlass.range_constexpr(4):
                                    cc = hh * 4 + i4
                                    v4 = (sbase + (off + ((ys ^ cc) * 4))).load(count=4, alignment=16)
                                    for e in cutlass.range_constexpr(4):
                                        wv.append(v4[e])
                                prims.tcgen05_st(
                                    "32x32b",
                                    prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_KA + sl4 * 32 + hh * 16).as_ptr(I32),
                                    cutlass.Vector.from_elements(wv, I32),
                                )
                            if cutlass.const_expr(sl4 == 0):
                                if do_cp:
                                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                prims.bar_warp_sync(0xFFFFFFFF)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive(b_ktf.subview(0))
                            elif cutlass.const_expr(sl4 == 3):
                                # One wait retires the three outstanding stores for slabs 1-3.
                                if do_cp:
                                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                prims.bar_warp_sync(0xFFFFFFFF)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive(b_ktf.subview(1))
                        # V(c): SMEM -> TMEM (free once dP(c-1) has completed), then release the V SMEM stage
                        _wait(b_vf, cgc & 1)
                        _wait(b_vte, (cgc & 1) ^ 1)
                        vb = sbase + (roff - OFF_K // 4 + OFF_V // 4)
                        for sl4 in cutlass.range_constexpr(4):
                            ys = yr
                            if cutlass.const_expr(PAIR and sl4 % 2 == 1):
                                ys = yr + 4
                            if do_cp:
                                wv = []
                                for i4 in cutlass.range_constexpr(4):
                                    cc = hh * 4 + i4
                                    v4 = (vb + (sl4 * slab_w + ((ys ^ cc) * 4))).load(count=4, alignment=16)
                                    for e in cutlass.range_constexpr(4):
                                        wv.append(v4[e])
                                prims.tcgen05_st(
                                    "32x32b",
                                    prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_VA + sl4 * 32 + hh * 16).as_ptr(I32),
                                    cutlass.Vector.from_elements(wv, I32),
                                )
                            if cutlass.const_expr(sl4 == 0):
                                if do_cp:
                                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                prims.bar_warp_sync(0xFFFFFFFF)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive(b_vtf.subview(0))
                            elif cutlass.const_expr(sl4 == 3):
                                if do_cp:
                                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                prims.bar_warp_sync(0xFFFFFFFF)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive(b_vtf.subview(1))
                        if prims.elect_sync():
                            prims.mbarrier_arrive(b_ve)
                    gc = gc + nchunks

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == 2:
        prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw, F32), TM_COLS, group="cta_1")


qsa_bwd_sparse_query_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _sparse_query_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    sparse_blocks: cute.Tensor,
    sparse_pass_offsets: cute.Tensor,
    sparse_block_slots: cute.Tensor,
    pair_records: cute.Tensor,
    sparse_queries: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    sparse_count: cute.Tensor,
    nctas: I32,
    NP: I32,
    scale: F32,
    HKV: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = q.shape[0]
    qbox = (1, GQ * HKV, 64)
    tma_q = cuda.create_tensor_map_tiled_from_view(q, box_dims=qbox, stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_do = cuda.create_tensor_map_tiled_from_view(do, box_dims=qbox, stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    if cutlass.const_expr(HKV == 1):
        # a block as [4 slabs][4 tokens x 128 B]
        kv_layout = cute.make_layout((T, 4, 64), stride=(D, 64, 1))
        kv_box = (4, 4, 64)
        kv_order = (2, 0, 1)
    else:
        # a block as [4 slabs][4 tokens x 2 heads x 128 B]
        kv_layout = cute.make_layout((T, HKV, 4, 64), stride=(HKV * D, D, 64, 1))
        kv_box = (4, HKV, 4, 64)
        kv_order = (3, 1, 0, 2)
    tma_k = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(k.iterator, kv_layout), box_dims=kv_box, stride_order=kv_order, swizzle=cuda.TensorMapSwizzle.s128b
    )
    tma_v = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(v.iterator, kv_layout), box_dims=kv_box, stride_order=kv_order, swizzle=cuda.TensorMapSwizzle.s128b
    )
    qsa_bwd_sparse_query_kernel(
        tma_q,
        tma_do,
        tma_k,
        tma_v,
        lse,
        delta,
        dq,
        sparse_blocks,
        sparse_pass_offsets,
        sparse_block_slots,
        pair_records,
        sparse_queries,
        lo,
        hi,
        sparse_count,
        nctas,
        NP,
        scale,
        HKV,
    ).launch(grid=(nctas, 1, 1), block=(NTHREADS, 1, 1), stream=stream)


# ----------------------------------------------------------------------------- key-major kernel (dK, dV)
# One work item per key (range r, block b): the n queries of range r that selected block b. Their Q / dO rows are
# gathered a few queries per stage and, with the P / dS records expanded into MN-major B tiles, accumulated on the
# tensor core:
#   dV^T [256 dims, 16] += dO^T [256, K] P [K, 16]      dK^T [256 dims, 16] += Q^T [256, K] dS [K, 16]
# (two KV heads: two queries per stage, K = 48, column n < 8 = kv head n // 4, token n % 4. One KV head: four queries
# of 16 K-rows each, 12 heads and 4 zero rows, column n < 4 = token n. Other columns are zero). The item's dV / dK
# (fp32, TMEM double buffered) are then reduced once into the fp32 accumulators.
PB_NST = 4  # stages: one landing, one being expanded, one on the tensor core, one refilling
PB_THREADS = 256  # w0 producer, w1 MMA, w2-3 record expansion, w4-7 dK/dV drain


class _PbLayout:
    """Shared memory layout of the key-major kernel for one geometry (bytes from a 1024-aligned base)."""

    def __init__(self, hkv):
        self.qps = 4 // hkv  # queries per stage
        self.rec_w = _rec_w(hkv)
        if hkv == 2:
            # Q and dO tiles as [4 slabs][48 rows x 128 B] (SW128), one 3 KB box per query and slab
            self.slab = 48 * 128
            self.q_bytes = 4 * self.slab  # per tensor and stage
            self.k_groups = 6
        else:
            # one 6 KB box per query and tensor as [4 slabs][12 rows x 128 B]. A query's K-step of 16 rows reads its
            # 12 heads and the first 4 rows of the next slab, which meet zero P / dS rows.
            self.slab = 12 * 128
            self.q_bytes = 4 * 4 * self.slab
            self.k_groups = 8
        self.off_q = 0
        self.off_do = self.q_bytes
        self.off_rec = 2 * self.q_bytes
        self.off_p = self.off_rec + 1024  # P tile: [k_groups][2 N-chunks][8 rows x 16 B]
        self.tile = self.k_groups * 256
        self.off_ds = self.off_p + self.tile
        self.stage = self.off_ds + self.tile  # every stage starts on a 1 KB boundary, which the SW128 slabs require
        self.off_st = PB_NST * self.stage  # dK/dV staging: 4 drain warps x 4 boxes x [4 tokens][hkv][32 dims] fp32
        self.box_w = 4 * hkv * 32  # words per staging box
        self.st_w = 4 * self.box_w
        self.smem = self.off_st + 4 * self.st_w * 4
        self.tx_per_query = 2 * (self.q_bytes // self.qps) + self.rec_w * 4


@cute.jit
def _active_key(p_active, n0, KR, i):
    # item i of the key-major kernel: range 0's active keys come first, range 1's start at KR
    off = i
    if i >= n0:
        off = i - n0 + KR
    return (p_active + off).load()


@cute.kernel
def qsa_bwd_sparse_key_kernel(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_do: cutlass.GridConstant[cuda.TensorMap],
    tma_rk: cutlass.GridConstant[cuda.TensorMap],
    tma_rv: cutlass.GridConstant[cuda.TensorMap],
    g_pair_records: cute.Tensor,
    g_key_slot_offsets: cute.Tensor,
    g_key_pair_counts: cute.Tensor,
    g_slot_queries: cute.Tensor,
    g_active_keys: cute.Tensor,
    g_active_count: cute.Tensor,
    KR: I32,
    SK: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
):
    PB = _PbLayout(HKV)
    # the items: the keys with at least one record, compacted by the index scan
    n0 = g_active_count.iterator.raw_ptr().load()
    Ktot = n0 + (g_active_count.iterator.raw_ptr() + 1).load()
    p_active = g_active_keys.iterator.raw_ptr()
    QPS = PB.qps
    REC_W = PB.rec_w
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nctas, _, _ = cute.arch.grid_dim()
    warp = cute.arch.warp_idx()
    lane = tidx % 32

    smem = cutlass.Array(I32, PB.smem // 4, space=cutlass.AddressSpace.smem, alignment=1024)
    b_full = cutlass.Array(cutlass.Int64, PB_NST, space=cutlass.AddressSpace.smem, alignment=8)
    b_exp = cutlass.Array(cutlass.Int64, PB_NST, space=cutlass.AddressSpace.smem, alignment=8)
    b_emp = cutlass.Array(cutlass.Int64, PB_NST, space=cutlass.AddressSpace.smem, alignment=8)
    b_af = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_ae = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    # the dense pair grid is a programmatic dependent launch (PDL) secondary, so it backfills the SMs this grid leaves
    prims.griddepcontrol("launch_dependents")

    sbase = smem.data_ptr()
    zero4 = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
    for it in cutlass.range(tidx, PB.smem // 16, PB_THREADS, unroll=1):
        (sbase + it * 4).store(zero4, alignment=16)
    if warp == 0:
        if prims.elect_sync():
            prims.prefetch_tensormap(tma_q.get_ptr())
            prims.prefetch_tensormap(tma_do.get_ptr())
            prims.prefetch_tensormap(tma_rk.get_ptr())
            prims.prefetch_tensormap(tma_rv.get_ptr())
            for s in cutlass.range_constexpr(PB_NST):
                prims.mbarrier_init(b_full.subview(s), 1)
                prims.mbarrier_init(b_exp.subview(s), 2)
                prims.mbarrier_init(b_emp.subview(s), 1)
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_af.subview(s), 1)
                prims.mbarrier_init(b_ae.subview(s), 4)
    if warp == 1:
        prims.tcgen05_alloc(tmem_slot, 128, group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    prims.fence_mbarrier_init()
    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
    prims.barrier_cta_sync(0)

    tmem_raw = tmem_slot.load()
    tmem_col0 = tmem_raw & I32(0xFFFF)
    tmem_row0 = tmem_raw >> I32(16)
    p_pair_records = g_pair_records.iterator.raw_ptr()
    p_key_slot_offsets = g_key_slot_offsets.iterator.raw_ptr()
    p_key_pair_counts = g_key_pair_counts.iterator.raw_ptr()
    p_slot_queries = g_slot_queries.iterator.raw_ptr()

    if warp == 0:
        # ---------------------------------------------------------------- producer
        gs = I32(0)
        # key counts / offsets two keys ahead and the first 32 query ids one key ahead keep the slot_queries latency off
        # the start of every key
        n_nx = I32(0)
        o_nx = I32(0)
        q_nx = I32(0)
        n_n2 = I32(0)
        o_n2 = I32(0)
        if bid < Ktot:
            w0 = _active_key(p_active, n0, KR, bid)
            n_nx = (p_key_pair_counts + w0).load()
            o_nx = (p_key_slot_offsets + w0).load()
            if lane < n_nx:
                q_nx = (p_slot_queries + o_nx + lane).load()
        if bid + nctas < Ktot:
            w1 = _active_key(p_active, n0, KR, bid + nctas)
            n_n2 = (p_key_pair_counts + w1).load()
            o_n2 = (p_key_slot_offsets + w1).load()
        for i in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            o = o_nx
            q0v = q_nx
            n_nx = n_n2
            o_nx = o_n2
            q_nx = I32(0)
            if lane < n_nx:
                q_nx = (p_slot_queries + o_nx + lane).load()
            i2 = i + 2 * nctas
            if i2 < Ktot:
                w2 = _active_key(p_active, n0, KR, i2)
                n_n2 = (p_key_pair_counts + w2).load()
                o_n2 = (p_key_slot_offsets + w2).load()
            for qb in cutlass.range(0, n, 32, unroll=1):
                qv = q0v
                if qb > 0:
                    qv = I32(0)
                    if qb + lane < n:
                        qv = (p_slot_queries + (o + qb + lane)).load()
                nloc = n - qb
                if nloc > 32:
                    nloc = I32(32)
                for j2 in cutlass.range(0, nloc, QPS, unroll=1):
                    qids = []
                    for i in cutlass.range_constexpr(QPS):
                        qids.append(prims.shfl_sync(0xFFFFFFFF, qv, j2 + i, 0x1F, prims.Shfl.IDX))
                    nq = nloc - j2
                    if nq > QPS:
                        nq = I32(QPS)
                    slot = gs % PB_NST
                    _wait(b_emp.subview(slot), ((gs // PB_NST) & 1) ^ 1)
                    if prims.elect_sync():
                        sb = sbase + (slot * PB.stage) // 4
                        prims.mbarrier_arrive_expect_tx(b_full.subview(slot), nq * PB.tx_per_query)
                        for i in cutlass.range_constexpr(QPS):
                            if i < nq:
                                if cutlass.const_expr(HKV == 2):
                                    for s in cutlass.range_constexpr(4):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sb + (PB.off_q + s * PB.slab + i * 3072) // 4, tma_q.get_ptr(), (s * 64, 0, qids[i]), b_full.subview(slot)
                                        )
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sb + (PB.off_do + s * PB.slab + i * 3072) // 4, tma_do.get_ptr(), (s * 64, 0, qids[i]), b_full.subview(slot)
                                        )
                                else:
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sb + (PB.off_q + i * 4 * PB.slab) // 4, tma_q.get_ptr(), (0, 0, 0, qids[i]), b_full.subview(slot)
                                    )
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sb + (PB.off_do + i * 4 * PB.slab) // 4, tma_do.get_ptr(), (0, 0, 0, qids[i]), b_full.subview(slot)
                                    )
                        prims.cp_async_bulk_shared_cluster_global(
                            sb + PB.off_rec // 4,
                            p_pair_records + cutlass.Int64(o + qb + j2) * REC_W,
                            b_full.subview(slot),
                            nq * (REC_W * 4),
                            l2_cache_hint=cutlass.Int64(L2_EVICT_FIRST),
                        )
                    prims.bar_warp_sync(0xFFFFFFFF)
                    gs = gs + 1
    elif warp == 1:
        # ---------------------------------------------------------------- MMA issuer
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=F32,
            a_dtype=BF16,
            b_dtype=BF16,
            n_dim=16,
            m_dim=128,
            a_major=1,
            b_major=1,
        )
        SW128 = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
        SWN = prims.Tcgen05SmemSwizzle.NONE
        # Q / dO tiles read as Q^T / dO^T (MN-major SW128): LBO = next 64-dim slab, SBO = next 8 rows
        d_q = _desc(sbase + PB.off_q // 4, PB.slab, 1024, SW128)
        d_do = _desc(sbase + PB.off_do // 4, PB.slab, 1024, SW128)
        # P / dS tiles (MN-major, no swizzle): LBO = next 8 K-rows (256 B), SBO = next 8 columns (128 B)
        d_p = _desc(sbase + PB.off_p // 4, 256, 128, SWN)
        d_ds = _desc(sbase + PB.off_ds // 4, 256, 128, SWN)
        gs = I32(0)
        ia = I32(0)
        n_nx = I32(0)
        k_nx = I32(0)
        if bid < Ktot:
            k_nx = _active_key(p_active, n0, KR, bid)
            n_nx = (p_key_pair_counts + k_nx).load()
        for i in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            w = k_nx
            if i + nctas < Ktot:
                k_nx = _active_key(p_active, n0, KR, i + nctas)
                n_nx = (p_key_pair_counts + k_nx).load()
            if n > 0:
                ab = ia & 1
                _wait(b_ae.subview(ab), ((ia >> 1) & 1) ^ 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                nst = (n + (QPS - 1)) // QPS
                for si in cutlass.range(nst, unroll=1):
                    slot = gs % PB_NST
                    ph = (gs // PB_NST) & 1
                    _wait(b_full.subview(slot), ph)
                    _wait(b_exp.subview(slot), ph)
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    soff = slot * PB.stage
                    if cutlass.const_expr(HKV == 1):
                        # one K-step of 16 rows per query of the stage, over the stage's queries only
                        nq_s = n - QPS * si
                        if nq_s > QPS:
                            nq_s = I32(QPS)
                        if prims.elect_sync():
                            for qq in cutlass.range(nq_s, unroll=1):
                                accf = (si > 0) | (qq > 0)
                                for hh in cutlass.range_constexpr(2):
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        prims.make_tmem_ptr(tmem_raw + (ab * 64 + hh * 16), F32),
                                        d_do.advance_start_address(soff + qq * 4 * PB.slab + hh * 2 * PB.slab),
                                        d_p.advance_start_address(soff + qq * 512),
                                        idesc,
                                        accf,
                                    )
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        prims.make_tmem_ptr(tmem_raw + (ab * 64 + 32 + hh * 16), F32),
                                        d_q.advance_start_address(soff + qq * 4 * PB.slab + hh * 2 * PB.slab),
                                        d_ds.advance_start_address(soff + qq * 512),
                                        idesc,
                                        accf,
                                    )
                            prims.tcgen05_commit(b_emp.subview(slot), group=prims.CTAGroup.CTA_1)
                    elif cutlass.const_expr(SK > 0):
                        # a single-query stage (odd n) has zero P / dS rows 24-47, so K-step 2 (rows 32-47) is skipped
                        nk3 = I32(2)
                        if n - 2 * si > 1:
                            nk3 = I32(3)
                        if prims.elect_sync():
                            for hh in cutlass.range_constexpr(2):
                                for kk in cutlass.range_constexpr(3):
                                    accf = (si > 0) | (kk > 0)
                                    if kk < nk3:
                                        prims.tcgen05_mma(
                                            prims.Tcgen05MMAKind.F16,
                                            prims.CTAGroup.CTA_1,
                                            prims.make_tmem_ptr(tmem_raw + (ab * 64 + hh * 16), F32),
                                            d_do.advance_start_address(soff + hh * 2 * PB.slab + kk * 2048),
                                            d_p.advance_start_address(soff + kk * 512),
                                            idesc,
                                            accf,
                                        )
                                        prims.tcgen05_mma(
                                            prims.Tcgen05MMAKind.F16,
                                            prims.CTAGroup.CTA_1,
                                            prims.make_tmem_ptr(tmem_raw + (ab * 64 + 32 + hh * 16), F32),
                                            d_q.advance_start_address(soff + hh * 2 * PB.slab + kk * 2048),
                                            d_ds.advance_start_address(soff + kk * 512),
                                            idesc,
                                            accf,
                                        )
                            prims.tcgen05_commit(b_emp.subview(slot), group=prims.CTAGroup.CTA_1)
                    else:
                        if prims.elect_sync():
                            for hh in cutlass.range_constexpr(2):
                                for kk in cutlass.range_constexpr(3):
                                    accf = (si > 0) | (kk > 0)
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        prims.make_tmem_ptr(tmem_raw + (ab * 64 + hh * 16), F32),
                                        d_do.advance_start_address(soff + hh * 2 * PB.slab + kk * 2048),
                                        d_p.advance_start_address(soff + kk * 512),
                                        idesc,
                                        accf,
                                    )
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        prims.make_tmem_ptr(tmem_raw + (ab * 64 + 32 + hh * 16), F32),
                                        d_q.advance_start_address(soff + hh * 2 * PB.slab + kk * 2048),
                                        d_ds.advance_start_address(soff + kk * 512),
                                        idesc,
                                        accf,
                                    )
                            prims.tcgen05_commit(b_emp.subview(slot), group=prims.CTAGroup.CTA_1)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    gs = gs + 1
                if prims.elect_sync():
                    prims.tcgen05_commit(b_af.subview(ab), group=prims.CTAGroup.CTA_1)
                prims.bar_warp_sync(0xFFFFFFFF)
                ia = ia + 1
    elif warp < 4:
        # ---------------------------------------------------------------- record expansion into MN-major B tiles
        t = tidx - 64
        if cutlass.const_expr(HKV == 2):
            jq = I32(0)
            if t >= 24:
                jq = I32(1)
            hq = t - 24 * jq
            hkv = I32(0)
            if hq >= 12:
                hkv = I32(1)
            row_on = t < 48
        else:
            jq = t >> 4
            hq = t & 15
            hkv = I32(0)
            row_on = hq < 12
        rw = hq >> 1
        rs = (hq & 1) * 16
        toff = ((t >> 3) * 256 + (t & 7) * 16) // 4
        gs = I32(0)
        n_nx = I32(0)
        k_nx = I32(0)
        if bid < Ktot:
            k_nx = _active_key(p_active, n0, KR, bid)
            n_nx = (p_key_pair_counts + k_nx).load()
        for i in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            w = k_nx
            if i + nctas < Ktot:
                k_nx = _active_key(p_active, n0, KR, i + nctas)
                n_nx = (p_key_pair_counts + k_nx).load()
            nst = (n + (QPS - 1)) // QPS
            for si in cutlass.range(nst, unroll=1):
                slot = gs % PB_NST
                ph = (gs // PB_NST) & 1
                nq = n - QPS * si
                _wait(b_full.subview(slot), ph)
                if row_on:
                    sb = sbase + (slot * PB.stage) // 4
                    pw0 = I32(0)
                    pw1 = I32(0)
                    dw0 = I32(0)
                    dw1 = I32(0)
                    if jq < nq:
                        rb = sb + (PB.off_rec // 4 + jq * REC_W + rw)
                        pv = []
                        dv_ = []
                        for tt in cutlass.range_constexpr(4):
                            pv.append(((rb + tt * (6 * HKV)).load() >> rs) & I32(0xFFFF))
                            dv_.append(((rb + REC_W // 2 + tt * (6 * HKV)).load() >> rs) & I32(0xFFFF))
                        pw0 = pv[0] | (pv[1] << 16)
                        pw1 = pv[2] | (pv[3] << 16)
                        dw0 = dv_[0] | (dv_[1] << 16)
                        dw1 = dv_[2] | (dv_[3] << 16)
                    e0 = pw0
                    e1 = pw1
                    e2 = I32(0)
                    e3 = I32(0)
                    f0 = dw0
                    f1 = dw1
                    f2 = I32(0)
                    f3 = I32(0)
                    if hkv == 1:
                        e2 = pw0
                        e3 = pw1
                        e0 = I32(0)
                        e1 = I32(0)
                        f2 = dw0
                        f3 = dw1
                        f0 = I32(0)
                        f1 = I32(0)
                    vp = cutlass.Vector.from_elements([e0, e1, e2, e3], I32)
                    vd = cutlass.Vector.from_elements([f0, f1, f2, f3], I32)
                    (sb + (PB.off_p // 4 + toff)).store(vp, alignment=16)
                    (sb + (PB.off_ds // 4 + toff)).store(vd, alignment=16)
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.bar_warp_sync(0xFFFFFFFF)
                if prims.elect_sync():
                    prims.mbarrier_arrive(b_exp.subview(slot))
                gs = gs + 1
    else:
        # ---------------------------------------------------------------- dK / dV drain (one TMEM lane quadrant per warp)
        qd = warp - 4
        stg = sbase + (PB.off_st + qd * PB.st_w * 4) // 4
        ia = I32(0)
        n_nx = I32(0)
        k_nx = I32(0)
        if bid < Ktot:
            k_nx = _active_key(p_active, n0, KR, bid)
            n_nx = (p_key_pair_counts + k_nx).load()
        for i in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            w = k_nx
            if i + nctas < Ktot:
                k_nx = _active_key(p_active, n0, KR, i + nctas)
                n_nx = (p_key_pair_counts + k_nx).load()
            if n > 0:
                tok = w % KR
                ab = ia & 1
                _wait(b_af.subview(ab), (ia >> 1) & 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                va = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + ab * 64).as_ptr(F32), num=32)
                vb = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + ab * 64 + 32).as_ptr(F32), num=32)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.bar_warp_sync(0xFFFFFFFF)
                if prims.elect_sync():
                    prims.mbarrier_arrive(b_ae.subview(ab))
                # The key's dV / dK slice of this warp's 32 dims is staged as four [4 tokens][hkv][32 dims] boxes and
                # added with one full-line TMA tensor reduce each. The previous key's reduces must have read the boxes.
                prims.cp_async_bulk_wait_group(0, read=True)
                prims.bar_warp_sync(0xFFFFFFFF)
                for bx in cutlass.range_constexpr(4):
                    # box bx: tensor (0 dV, 1 dK) = bx >> 1, d-half = bx & 1
                    for kv in cutlass.range_constexpr(HKV):
                        for tt in cutlass.range_constexpr(4):
                            col = 4 * kv + tt + 16 * (bx & 1)
                            if cutlass.const_expr((bx >> 1) == 0):
                                val = va[col]
                            else:
                                val = vb[col]
                            (stg + (bx * PB.box_w + (tt * HKV + kv) * 32 + lane)).store(prims.mov_b32(val, target_type=I32))
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.bar_warp_sync(0xFFFFFFFF)
                if lane == 0:
                    for bx in cutlass.range_constexpr(4):
                        if cutlass.const_expr((bx >> 1) == 0):
                            prims.cp_async_bulk_tensor_reduce(tma_rv.get_ptr(), stg + bx * PB.box_w, prims.TMARedux.ADD, [(bx & 1) * 128 + qd * 32, 0, tok])
                        else:
                            prims.cp_async_bulk_tensor_reduce(tma_rk.get_ptr(), stg + bx * PB.box_w, prims.TMARedux.ADD, [(bx & 1) * 128 + qd * 32, 0, tok])
                prims.cp_async_bulk_commit_group()
                ia = ia + 1
        prims.cp_async_bulk_wait_group(0)

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == 1:
        prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw, F32), 128, group="cta_1")


qsa_bwd_sparse_key_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _sparse_key_host(
    q: cute.Tensor,
    do: cute.Tensor,
    pair_records: cute.Tensor,
    key_slot_offsets: cute.Tensor,
    key_pair_counts: cute.Tensor,
    slot_queries: cute.Tensor,
    sparse_dk_acc: cute.Tensor,
    sparse_dv_acc: cute.Tensor,
    active_keys: cute.Tensor,
    active_count: cute.Tensor,
    nctas: I32,
    KR: I32,
    SK: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = q.shape[0]
    if cutlass.const_expr(HKV == 2):
        tma_q = cuda.create_tensor_map_tiled_from_view(q, box_dims=(1, 24, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
        tma_do = cuda.create_tensor_map_tiled_from_view(do, box_dims=(1, 24, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    else:
        # a query's 12 head rows of all 4 slabs in one box, slab-major in SMEM
        q_layout = cute.make_layout((T, GQ, 4, 64), stride=(GQ * D, D, 64, 1))
        tma_q = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(q.iterator, q_layout), box_dims=(1, GQ, 4, 64), stride_order=(3, 1, 2, 0), swizzle=cuda.TensorMapSwizzle.s128b
        )
        tma_do = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(do.iterator, q_layout), box_dims=(1, GQ, 4, 64), stride_order=(3, 1, 2, 0), swizzle=cuda.TensorMapSwizzle.s128b
        )
    TP = sparse_dk_acc.shape[0] // (HKV * D)
    tma_rk = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(sparse_dk_acc.iterator, cute.make_layout((TP, HKV, D), stride=(HKV * D, D, 1))),
        box_dims=(4, HKV, 32),
        stride_order=(2, 1, 0),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_rv = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(sparse_dv_acc.iterator, cute.make_layout((TP, HKV, D), stride=(HKV * D, D, 1))),
        box_dims=(4, HKV, 32),
        stride_order=(2, 1, 0),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    qsa_bwd_sparse_key_kernel(
        tma_q, tma_do, tma_rk, tma_rv, pair_records, key_slot_offsets, key_pair_counts, slot_queries, active_keys, active_count, KR, SK, HKV
    ).launch(grid=(nctas, 1, 1), block=(PB_THREADS, 1, 1), stream=stream)


# ----------------------------------------------------------------------------- prep kernel
PREP_THREADS = 256
PREP_ROWS = 4  # Delta rows in flight per warp, so all eight vector loads issue before the dot products


@cute.kernel
def qsa_bwd_prep_kernel(
    g_o: cute.Tensor,
    g_do: cute.Tensor,
    g_delta: cute.Tensor,
    g_sparse_dk_acc: cute.Tensor,
    g_sparse_dv_acc: cute.Tensor,
    g_dense_block_bitmap: cute.Tensor,
    g_key_pair_counts: cute.Tensor,
    g_dq_acc: cute.Tensor,
    g_dense_dk_acc: cute.Tensor,
    g_dense_dv_acc: cute.Tensor,
    g_dense_hi: cute.Tensor,
    T: I32,
    nz: I32,
    nsel: I32,
    n_key_pair_counts: I32,
    n_dense_dkv: I32,
    HKV: cutlass.Constexpr[int],
):
    HQ = GQ * HKV
    # Delta and the zero fill of every accumulator share one launch.
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    lane = tidx % 32
    wib = tidx // 32
    p_o = g_o.iterator.raw_ptr()
    p_do = g_do.iterator.raw_ptr()
    p_delta = g_delta.iterator.raw_ptr()
    nrows = T * (GQ * HKV)
    nwarps = nb * (PREP_THREADS // 32)
    for r0 in cutlass.range(bid * (PREP_THREADS // 32) + wib, nrows, nwarps * PREP_ROWS, unroll=1):
        o_vecs = []
        do_vecs = []
        for u in cutlass.range_constexpr(PREP_ROWS):
            r = r0 + u * nwarps
            if r >= nrows:
                r = r0
            o_vecs.append((p_o + r * D + lane * 8).load(count=8, alignment=16))
            do_vecs.append((p_do + r * D + lane * 8).load(count=8, alignment=16))
        for u in cutlass.range_constexpr(PREP_ROWS):
            r = r0 + u * nwarps
            ov = o_vecs[u].to(F32)
            dv = do_vecs[u].to(F32)
            acc = F32(0.0)
            for j in cutlass.range_constexpr(8):
                acc = acc + ov[j] * dv[j]
            for sh in cutlass.range_constexpr(5):
                acc = acc + prims.shfl_sync(0xFFFFFFFF, acc, 16 >> sh, 0x1F, prims.Shfl.BFLY)
            if lane == 0:
                if r < nrows:
                    (p_delta + r).store(acc)
    p_sparse_dk_acc = g_sparse_dk_acc.iterator.raw_ptr()
    p_sparse_dv_acc = g_sparse_dv_acc.iterator.raw_ptr()
    zf = cutlass.Vector.from_elements([F32(0.0)] * 8, F32)
    zi = cutlass.Vector.from_elements([I32(0)] * 4, I32)
    for it in cutlass.range(bid * PREP_THREADS + tidx, nz // 8, nb * PREP_THREADS, unroll=1):
        (p_sparse_dk_acc + it * 8).store(zf, alignment=16)
        (p_sparse_dv_acc + it * 8).store(zf, alignment=16)
    p_dense_block_bitmap = g_dense_block_bitmap.iterator.raw_ptr()
    for it in cutlass.range(bid * PREP_THREADS + tidx, nsel // 4, nb * PREP_THREADS, unroll=1):
        (p_dense_block_bitmap + it * 4).store(zi, alignment=16)
    p_key_pair_counts = g_key_pair_counts.iterator.raw_ptr()
    for it in cutlass.range(bid * PREP_THREADS + tidx, n_key_pair_counts // 4, nb * PREP_THREADS, unroll=1):
        (p_key_pair_counts + it * 4).store(zi, alignment=16)
    # the dense queries' dQ rows start at zero, the query-major kernel initializes the sparse rows itself
    p_dense_hi = g_dense_hi.iterator.raw_ptr()
    p_dq_acc = g_dq_acc.iterator.raw_ptr()
    for r in cutlass.range(bid * (PREP_THREADS // 32) + wib, T, nwarps, unroll=1):
        if r < (p_dense_hi + r).load():
            for v in cutlass.range(lane, HQ * D // 8, 32, unroll=1):
                (p_dq_acc + (r * (HQ * D) + v * 8)).store(zf, alignment=16)
    p_dense_dk_acc = g_dense_dk_acc.iterator.raw_ptr()
    p_dense_dv_acc = g_dense_dv_acc.iterator.raw_ptr()
    for it in cutlass.range(bid * PREP_THREADS + tidx, n_dense_dkv // 8, nb * PREP_THREADS, unroll=1):
        (p_dense_dk_acc + it * 8).store(zf, alignment=16)
        (p_dense_dv_acc + it * 8).store(zf, alignment=16)


qsa_bwd_prep_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _prep_host(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    sparse_dk_acc: cute.Tensor,
    sparse_dv_acc: cute.Tensor,
    dense_block_bitmap: cute.Tensor,
    key_pair_counts: cute.Tensor,
    dq_acc: cute.Tensor,
    dense_dk_acc: cute.Tensor,
    dense_dv_acc: cute.Tensor,
    dense_hi: cute.Tensor,
    nblocks: I32,
    HKV: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = o.shape[0]
    nz = sparse_dk_acc.shape[0]
    qsa_bwd_prep_kernel(
        o,
        do,
        delta,
        sparse_dk_acc,
        sparse_dv_acc,
        dense_block_bitmap,
        key_pair_counts,
        dq_acc,
        dense_dk_acc,
        dense_dv_acc,
        dense_hi,
        I32(T),
        I32(nz),
        I32(dense_block_bitmap.shape[0]),
        I32(key_pair_counts.shape[0]),
        I32(dense_dk_acc.shape[0]),
        HKV,
    ).launch(grid=(nblocks, 1, 1), block=(PREP_THREADS, 1, 1), stream=stream)


# ----------------------------------------------------------------------------- sequences and the dense / sparse split
# cu_seqlens [B + 1] bounds the packed sequences. Positions, blocks and the tail are sequence-relative, token ids are
# positions in the packed buffer. A sequence's first nd queries run on the dense key-tile kernel, the rest on the
# query-major / key-major pair. seq_start gives every token its sequence, sparse_queries lists the sparse
# queries of all sequences in order, seq_sparse_base[d] is the first entry of sequence d and sparse_count their total.
SCAN_T = 1024  # the largest CTA, so one CTA scans a whole range's keys or all sequences with a serial per-thread prefix


@cute.jit
def _cta_max(sp, tidx, value):
    # maximum of value over the SCAN_T threads, in every thread
    value = cute.arch.warp_reduction_max(value)
    if tidx % 32 == 0:
        (sp + tidx // 32).store(value)
    prims.barrier_cta_sync(0)
    acc = I32(0)
    for w in cutlass.range_constexpr(SCAN_T // 32):
        x = (sp + w).load()
        if x > acc:
            acc = x
    prims.barrier_cta_sync(0)
    return acc


@cute.jit
def _dense_unit(max_len, HKV, num_sms):
    # The dense pair kernel's query groups per (key tile, unit) item for a dense-only plan: one wave of the longest
    # sequence's causal work per pair slot (key tile t sees the groups from its first token up), so the fixed cost of
    # a pair is paid once per slot and the waves stay balanced.
    nd = _dense_queries(max_len)
    gd = (nd + (DP_G - 1)) // DP_G
    tiles = (nd + (DP_TT - 1)) // DP_TT
    work = HKV * (tiles * gd - (DP_TT // DP_G) * ((tiles * (tiles - 1)) // 2))
    slots = num_sms // 2
    unit = (work + slots - 1) // slots
    if unit < 1:
        unit = I32(1)
    if unit > DP_U_MAX:
        unit = I32(DP_U_MAX)
    return unit


@cute.jit
def _cta_scan(sp, tidx, value):
    # exclusive prefix of value over the SCAN_T threads and their total, in every thread
    lane = tidx % 32
    wid = tidx // 32
    incl = value
    for offset in (1, 2, 4, 8, 16):
        prior = cute.arch.shuffle_sync_up(incl, offset)
        if lane >= offset:
            incl = incl + prior
    if lane == 31:
        (sp + wid).store(incl)
    prims.barrier_cta_sync(0)
    base = I32(0)
    total = I32(0)
    for w in cutlass.range_constexpr(SCAN_T // 32):
        x = (sp + w).load()
        if w < wid:
            base = base + x
        total = total + x
    prims.barrier_cta_sync(0)
    return base + incl - value, total


@cute.jit
def _cta_scan2(sp, tidx, a, b):
    # exclusive prefixes of a and b over the SCAN_T threads and their totals, in every thread: a warp scan, then the
    # 32 warp totals scanned by one warp-wide shuffle pass
    lane = tidx % 32
    wid = tidx // 32
    ia = a
    ib = b
    for offset in (1, 2, 4, 8, 16):
        pa = cute.arch.shuffle_sync_up(ia, offset)
        pb = cute.arch.shuffle_sync_up(ib, offset)
        if lane >= offset:
            ia = ia + pa
            ib = ib + pb
    if lane == 31:
        (sp + wid).store(ia)
        (sp + 32 + wid).store(ib)
    prims.barrier_cta_sync(0)
    wa = (sp + lane).load()
    wb = (sp + 32 + lane).load()
    for offset in (1, 2, 4, 8, 16):
        pa = cute.arch.shuffle_sync_up(wa, offset)
        pb = cute.arch.shuffle_sync_up(wb, offset)
        if lane >= offset:
            wa = wa + pa
            wb = wb + pb
    ta = cute.arch.shuffle_sync(wa, 31)
    tb = cute.arch.shuffle_sync(wb, 31)
    ba = cute.arch.shuffle_sync(wa, wid) - (sp + wid).load()
    bb = cute.arch.shuffle_sync(wb, wid) - (sp + 32 + wid).load()
    prims.barrier_cta_sync(0)
    return ba + ia - a, ta, bb + ib - b, tb


@cute.jit
def _dense_queries(L):
    # dense queries of a sequence of L tokens, the crossover rule of DENSE_QUERIES_MAX
    nd = L
    if L - DENSE_QUERIES_MAX >= SPARSE_FOLD:
        nd = I32(DENSE_QUERIES_MAX)
    return nd


@cute.kernel
def qsa_bwd_seq_scan_kernel(
    g_cu: cute.Tensor,
    g_tok_base: cute.Tensor,
    g_nseq: cute.Tensor,
    g_dense_unit: cute.Tensor,
    g_seq_sparse_base: cute.Tensor,
    g_sparse_count: cute.Tensor,
    T: I32,
    B_in: I32,
    mode: I32,
    unit_fixed: I32,
    HKV: I32,
    num_sms: I32,
):
    # one CTA: the sequence bounds (mode 0 one sequence, 1 the caller's cu_seqlens, 2 derived from the per-token
    # origins tok_base, a sequence starting wherever the origin changes), the dense unit (unit_fixed, or from the
    # longest sequence when 0), then the exclusive scan of the sequences' sparse query counts and their total
    tidx, _, _ = cute.arch.thread_idx()
    sm = cutlass.Array(I32, SCAN_T, space=cutlass.AddressSpace.smem, alignment=16)
    sp = sm.data_ptr()
    p_cu = g_cu.iterator.raw_ptr()
    p_base = g_seq_sparse_base.iterator.raw_ptr()
    p_nseq = g_nseq.iterator.raw_ptr()
    if mode == 2:
        p_tb = g_tok_base.iterator.raw_ptr()
        ch_t = (T + (SCAN_T - 1)) // SCAN_T
        t0 = tidx * ch_t
        t1 = t0 + ch_t
        if t1 > T:
            t1 = T
        cnt = I32(0)
        for t in cutlass.range(t0, t1, unroll=1):
            st = True
            if t > 0:
                st = (p_tb + t).load() != (p_tb + t - 1).load()
            if st:
                cnt = cnt + 1
        idx, total = _cta_scan(sp, tidx, cnt)
        if tidx == 0:
            p_nseq.store(total)
            (p_cu + total).store(T)
        for t in cutlass.range(t0, t1, unroll=1):
            st = True
            if t > 0:
                st = (p_tb + t).load() != (p_tb + t - 1).load()
            if st:
                (p_cu + idx).store(t)
                idx = idx + 1
        prims.barrier_cta_sync(0)
    else:
        if tidx == 0:
            if mode == 0:
                p_cu.store(I32(0))
                (p_cu + 1).store(T)
                p_nseq.store(I32(1))
            else:
                p_nseq.store(B_in)
        prims.barrier_cta_sync(0)
    B = p_nseq.load()
    ch = (B + (SCAN_T - 1)) // SCAN_T
    e0 = tidx * ch
    e1 = e0 + ch
    if e1 > B:
        e1 = B
    s = I32(0)
    max_len = I32(0)
    for d in cutlass.range(e0, e1, unroll=1):
        L = (p_cu + d + 1).load() - (p_cu + d).load()
        s = s + (L - _dense_queries(L))
        if L > max_len:
            max_len = L
    max_len = _cta_max(sp, tidx, max_len)
    acc, total = _cta_scan(sp, tidx, s)
    if tidx == 0:
        g_sparse_count.iterator.raw_ptr().store(total)
        unit = unit_fixed
        if unit == 0:
            unit = _dense_unit(max_len, HKV, num_sms)
        g_dense_unit.iterator.raw_ptr().store(unit)
    for d in cutlass.range(e0, e1, unroll=1):
        (p_base + d).store(acc)
        L = (p_cu + d + 1).load() - (p_cu + d).load()
        acc = acc + (L - _dense_queries(L))


qsa_bwd_seq_scan_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _seq_scan_host(
    cu: cute.Tensor,
    tok_base: cute.Tensor,
    nseq: cute.Tensor,
    dense_unit: cute.Tensor,
    seq_sparse_base: cute.Tensor,
    sparse_count: cute.Tensor,
    T: I32,
    B: I32,
    mode: I32,
    unit_fixed: I32,
    HKV: I32,
    num_sms: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_seq_scan_kernel(cu, tok_base, nseq, dense_unit, seq_sparse_base, sparse_count, T, B, mode, unit_fixed, HKV, num_sms).launch(
        grid=(1, 1, 1), block=(SCAN_T, 1, 1), stream=stream
    )


@cute.kernel
def qsa_bwd_seq_fill_kernel(
    g_cu: cute.Tensor,
    g_nseq: cute.Tensor,
    g_seq_sparse_base: cute.Tensor,
    g_seq_start: cute.Tensor,
    g_dense_hi: cute.Tensor,
    g_sparse_queries: cute.Tensor,
    T: I32,
):
    # one thread per token: its sequence by binary search, the first row past its sequence's dense queries, then its
    # entry in the compacted sparse list
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    t = bid * 256 + tidx
    p_cu = g_cu.iterator.raw_ptr()
    B = g_nseq.iterator.raw_ptr().load()
    if t < T:
        lo = I32(0)
        hi = B
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if (p_cu + mid).load() <= t:
                lo = mid
            else:
                hi = mid
        s = (p_cu + lo).load()
        e = (p_cu + lo + 1).load()
        (g_seq_start.iterator.raw_ptr() + t).store(s)
        p = t - s
        nd = _dense_queries(e - s)
        (g_dense_hi.iterator.raw_ptr() + t).store(s + nd)
        if p >= nd:
            (g_sparse_queries.iterator.raw_ptr() + ((g_seq_sparse_base.iterator.raw_ptr() + lo).load() + (p - nd))).store(t)


qsa_bwd_seq_fill_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _seq_fill_host(
    cu: cute.Tensor,
    nseq: cute.Tensor,
    seq_sparse_base: cute.Tensor,
    seq_start: cute.Tensor,
    dense_hi: cute.Tensor,
    sparse_queries: cute.Tensor,
    T: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_seq_fill_kernel(cu, nseq, seq_sparse_base, seq_start, dense_hi, sparse_queries, T).launch(
        grid=((T + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )


@cute.kernel
def qsa_bwd_window_kernel(total_count: cute.Tensor, window_count: cute.Tensor, key_counts: cute.Tensor, offset: I32, capacity: I32):
    # The window's query count is rebuilt on the device for every call and replay, and the key counts of the inverted
    # index start from zero.
    tid = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
    if tid < cute.size(key_counts):
        key_counts[tid] = I32(0)
    if tid == 0:
        count = total_count[0] - offset
        if count < 0:
            count = I32(0)
        if count > capacity:
            count = capacity
        window_count[0] = count


qsa_bwd_window_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _window_host(total_count: cute.Tensor, window_count: cute.Tensor, key_counts: cute.Tensor, offset: I32, capacity: I32, stream: cuda_driver.CUstream):
    qsa_bwd_window_kernel(total_count, window_count, key_counts, offset, capacity).launch(
        grid=(cute.ceil_div(cute.size(key_counts), 256), 1, 1), block=(256, 1, 1), stream=stream
    )


# ----------------------------------------------------------------------------- KV-pass partition and inverted index
# The query-major kernel walks (KV pass, query) items pass-major so that one pass's K/V blocks stay L2 resident.
# sparse_blocks[j] holds the first token of every selected block of sparse query j (and of its tail block) grouped by
# pass p = block // RB, and sparse_pass_offsets[j, p] .. sparse_pass_offsets[j, p+1] bounds pass p's segment.
PART_W = 516  # list row stride in int32: 512 blocks plus the tail block, rounded up to a multiple of 4 words
POS = 64  # offset row stride in int32, a power of two well above the 11 passes of the longest sequence, so NP < POS

# Sparse queries are split into ranges of QR queries. Every (query, selected block) pair gets a slot in a key-major
# order: key = range * T + the block's first token, so every block of the packed buffer has its own key, and the slots
# of a key are contiguous (key_slot_offsets[key] .. + key_pair_counts[key]). The query-major kernel writes the pair's
# P / dS record (bf16, [4 tokens][12 * HKV heads] each, see _rec_w) into its slot, and the key-major kernel walks the
# keys holding a record, compacted into active_keys by the index scan.
SPARSE_WINDOW_QUERIES = 4096  # Reuse at most two ranges of P/dS records, including for packed inputs.
QR = 2048  # queries per range: a range's Q / dO rows stay L2 resident while the key-major kernel walks its keys
QR_SH = 11  # log2(QR)
RANGE_PAIRS = 513 * QR  # slots per range: every query has 512 blocks and at most one tail block


@cute.jit
def _block_tok(p_first, lrow, m):
    # first token of the m-th block key of a query's compacted list
    return (p_first + (lrow + m).load()).load()


@cute.kernel
def qsa_bwd_partition_kernel(
    g_ids: cute.Tensor,
    g_counts: cute.Tensor,
    g_first: cute.Tensor,
    g_dq_acc: cute.Tensor,
    g_sparse_queries: cute.Tensor,
    g_sparse_count: cute.Tensor,
    g_sparse_blocks: cute.Tensor,
    g_sparse_pass_offsets: cute.Tensor,
    g_key_pair_counts: cute.Tensor,
    cap: I32,
    RB: I32,
    NP: I32,
    KR: I32,
    HKV: cutlass.Constexpr[int],
):
    # per-query block lists split by KV pass, plus the per-key pair counts of the inverted index, whose key is the
    # block's first token within the query's range. An empty row only needs its dQ zeroed. A row past SPARSE_CAP
    # blocks exceeds the slot budget of a range: it contributes nothing and its dQ is NaN.
    HQ = GQ * HKV
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    j = bid * 8 + tidx // 32
    p_ids = g_ids.iterator.raw_ptr()
    p_first = g_first.iterator.raw_ptr()
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()
    p_key_pair_counts = g_key_pair_counts.iterator.raw_ptr()
    if j < g_sparse_count.iterator.raw_ptr().load():
        qi = (g_sparse_queries.iterator.raw_ptr() + j).load()
        n = (g_counts.iterator.raw_ptr() + qi).load()
        lrow = p_ids + cutlass.Int64(qi) * cutlass.Int64(cap)
        p_dq_acc = g_dq_acc.iterator.raw_ptr()
        if n > SPARSE_CAP:
            nanv = cutlass.Vector.from_elements([F32(float("nan"))] * 8, F32)
            for v in cutlass.range(lane, HQ * D // 8, 32, unroll=1):
                (p_dq_acc + (qi * (HQ * D) + v * 8)).store(nanv, alignment=16)
            n = I32(0)
        elif n == 0:
            zf = cutlass.Vector.from_elements([F32(0.0)] * 8, F32)
            for v in cutlass.range(lane, HQ * D // 8, 32, unroll=1):
                (p_dq_acc + (qi * (HQ * D) + v * 8)).store(zf, alignment=16)
        kb = (j >> QR_SH) * KR
        lt = (I32(1) << lane) - I32(1)
        off = I32(0)
        if lane == 0:
            (p_sparse_pass_offsets + j * POS).store(I32(0))
        for pp in cutlass.range(NP, unroll=1):
            lo = pp * RB
            hi = lo + RB
            for m0 in cutlass.range(0, n, 32, unroll=1):
                m = m0 + lane
                inp = m < n
                tok = I32(0)
                if inp:
                    tok = _block_tok(p_first, lrow, m)
                if tok < lo:
                    inp = False
                if tok >= hi:
                    inp = False
                bal = prims.vote_sync(0xFFFFFFFF, inp, prims.VoteSync.BALLOT)
                if inp:
                    (p_sparse_blocks + (j * PART_W + off + cute.arch.popc(bal & lt))).store(tok)
                    prims.red(
                        prims.ReductionOp.ADD,
                        prims.ReductionType.U32,
                        p_key_pair_counts + (kb + tok),
                        I32(1),
                        mem_order=prims.MemOrder.RELAXED,
                        mem_scope=prims.MemScope.GPU,
                    )
                off = off + cute.arch.popc(bal)
            if lane == 0:
                (p_sparse_pass_offsets + (j * POS + pp + 1)).store(off)


qsa_bwd_partition_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _part_host(
    ids: cute.Tensor,
    counts: cute.Tensor,
    first: cute.Tensor,
    dq_acc: cute.Tensor,
    sparse_queries: cute.Tensor,
    sparse_count: cute.Tensor,
    sparse_blocks: cute.Tensor,
    sparse_pass_offsets: cute.Tensor,
    key_pair_counts: cute.Tensor,
    nsq_cap: I32,
    RB: I32,
    NP: I32,
    KR: I32,
    HKV: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    cap = ids.shape[1]
    qsa_bwd_partition_kernel(
        ids,
        counts,
        first,
        dq_acc,
        sparse_queries,
        sparse_count,
        sparse_blocks,
        sparse_pass_offsets,
        key_pair_counts,
        I32(cap),
        RB,
        NP,
        KR,
        HKV,
    ).launch(grid=((nsq_cap + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)


@cute.kernel
def qsa_bwd_index_scan_kernel(
    g_window_count: cute.Tensor,
    g_key_pair_counts: cute.Tensor,
    g_key_slot_offsets: cute.Tensor,
    g_key_slot_cursors: cute.Tensor,
    g_active_keys: cute.Tensor,
    g_active_count: cute.Tensor,
    KR: I32,
    NR: I32,
):
    # one CTA per range over tiles of 8 * SCAN_T keys, a thread holding 8 consecutive keys (two 16-byte vectors, so
    # a warp's loads coalesce): slot offsets = the range's slot base + exclusive scan of its key counts, cursors
    # start there, and the keys holding a record are compacted into the range's segment of the key-major kernel's
    # item list (active_keys[r * KR ..], active_count[r])
    tidx, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
    sm = cutlass.Array(I32, 64, space=cutlass.AddressSpace.smem, alignment=16)
    sp = sm.data_ptr()
    p_key_pair_counts = g_key_pair_counts.iterator.raw_ptr()
    p_key_slot_offsets = g_key_slot_offsets.iterator.raw_ptr()
    p_key_slot_cursors = g_key_slot_cursors.iterator.raw_ptr()
    p_active = g_active_keys.iterator.raw_ptr()
    zero4 = cutlass.Vector.from_elements([I32(0)] * 4, I32)
    kb = r * KR
    pos = kb
    if g_window_count.iterator.raw_ptr().load() > 0:
        acc = r * RANGE_PAIRS
        for t0 in cutlass.range(0, KR, 8 * SCAN_T, unroll=1):
            e = t0 + 8 * tidx
            inr = e < KR
            ca = zero4
            cb = zero4
            if inr:
                ca = (p_key_pair_counts + (kb + e)).load(count=4, alignment=16)
                cb = (p_key_pair_counts + (kb + e + 4)).load(count=4, alignment=16)
            s = ca[0] + ca[1] + ca[2] + ca[3] + cb[0] + cb[1] + cb[2] + cb[3]
            live = I32(0)
            for u in cutlass.range_constexpr(4):
                if ca[u] > 0:
                    live = live + 1
                if cb[u] > 0:
                    live = live + 1
            off, tot_s, lpos, tot_l = _cta_scan2(sp, tidx, s, live)
            if inr:
                a = acc + off
                p = pos + lpos
                offs = []
                for u in cutlass.range_constexpr(4):
                    offs.append(a)
                    a = a + ca[u]
                    if ca[u] > 0:
                        (p_active + p).store(kb + e + u)
                        p = p + 1
                ov = cutlass.Vector.from_elements(offs, I32)
                (p_key_slot_offsets + (kb + e)).store(ov, alignment=16)
                (p_key_slot_cursors + (kb + e)).store(ov, alignment=16)
                offs = []
                for u in cutlass.range_constexpr(4):
                    offs.append(a)
                    a = a + cb[u]
                    if cb[u] > 0:
                        (p_active + p).store(kb + e + 4 + u)
                        p = p + 1
                ov = cutlass.Vector.from_elements(offs, I32)
                (p_key_slot_offsets + (kb + e + 4)).store(ov, alignment=16)
                (p_key_slot_cursors + (kb + e + 4)).store(ov, alignment=16)
            acc = acc + tot_s
            pos = pos + tot_l
    if tidx == 0:
        p_count = g_active_count.iterator.raw_ptr()
        (p_count + r).store(pos - kb)
        if NR == 1:
            (p_count + 1).store(I32(0))


qsa_bwd_index_scan_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _scan_host(
    window_count: cute.Tensor,
    key_pair_counts: cute.Tensor,
    key_slot_offsets: cute.Tensor,
    key_slot_cursors: cute.Tensor,
    active_keys: cute.Tensor,
    active_count: cute.Tensor,
    KR: I32,
    NR: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_index_scan_kernel(window_count, key_pair_counts, key_slot_offsets, key_slot_cursors, active_keys, active_count, KR, NR).launch(
        grid=(NR, 1, 1), block=(SCAN_T, 1, 1), stream=stream
    )


@cute.kernel
def qsa_bwd_index_fill_kernel(
    g_sparse_queries: cute.Tensor,
    g_sparse_count: cute.Tensor,
    g_sparse_blocks: cute.Tensor,
    g_sparse_pass_offsets: cute.Tensor,
    g_key_slot_cursors: cute.Tensor,
    g_sparse_block_slots: cute.Tensor,
    g_slot_queries: cute.Tensor,
    NP: I32,
    KR: I32,
):
    # one warp per sparse query claims a slot for every pair: sparse_block_slots (aligned with sparse_blocks) -> slot, slot_queries[slot] = query
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    j = bid * 8 + tidx // 32
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()
    p_key_slot_cursors = g_key_slot_cursors.iterator.raw_ptr()
    p_sparse_block_slots = g_sparse_block_slots.iterator.raw_ptr()
    p_slot_queries = g_slot_queries.iterator.raw_ptr()
    if j < g_sparse_count.iterator.raw_ptr().load():
        qi = (g_sparse_queries.iterator.raw_ptr() + j).load()
        kb = (j >> QR_SH) * KR
        tot = (p_sparse_pass_offsets + (j * POS + NP)).load()
        for L in cutlass.range(lane, tot, 32, unroll=1):
            blk = (p_sparse_blocks + (j * PART_W + L)).load()
            pos = prims.atomicrmw("add", p_key_slot_cursors + (kb + blk), I32(1), mem_order=prims.MemOrder.RELAXED, syncscope=prims.MemScope.GPU)
            (p_sparse_block_slots + (j * PART_W + L)).store(pos)
            (p_slot_queries + pos).store(qi)


qsa_bwd_index_fill_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _fill_host(
    sparse_queries: cute.Tensor,
    sparse_count: cute.Tensor,
    sparse_blocks: cute.Tensor,
    sparse_pass_offsets: cute.Tensor,
    key_slot_cursors: cute.Tensor,
    sparse_block_slots: cute.Tensor,
    slot_queries: cute.Tensor,
    nsq_cap: I32,
    NP: I32,
    KR: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_index_fill_kernel(
        sparse_queries, sparse_count, sparse_blocks, sparse_pass_offsets, key_slot_cursors, sparse_block_slots, slot_queries, NP, KR
    ).launch(grid=((nsq_cap + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)


# ----------------------------------------------------------------------------- selection bitmap of the dense queries
@cute.kernel
def qsa_bwd_bitmap_kernel(
    g_ids: cute.Tensor,
    g_counts: cute.Tensor,
    g_first: cute.Tensor,
    g_lo: cute.Tensor,
    g_hi: cute.Tensor,
    g_seq_start: cute.Tensor,
    g_dense_hi: cute.Tensor,
    g_dense_block_bitmap: cute.Tensor,
    T: I32,
    cap: I32,
    NW: I32,
):
    # Row qi of the bitmap holds one bit per sequence-relative key token a dense query attends: the tokens of its
    # selected blocks within [lo, min(hi, query)], so the dense kernel tests the bit alone and a block may start
    # anywhere.
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    qi = bid * 8 + tidx // 32
    p_ids = g_ids.iterator.raw_ptr()
    p_first = g_first.iterator.raw_ptr()
    p_dense_block_bitmap = g_dense_block_bitmap.iterator.raw_ptr()
    if qi < T:
        if qi < (g_dense_hi.iterator.raw_ptr() + qi).load():
            s = (g_seq_start.iterator.raw_ptr() + qi).load()
            n = (g_counts.iterator.raw_ptr() + qi).load()
            floor = (g_lo.iterator.raw_ptr() + qi).load()
            if floor < s:
                floor = s
            ceil = (g_hi.iterator.raw_ptr() + qi).load()
            if ceil > qi:
                ceil = qi
            lrow = p_ids + cutlass.Int64(qi) * cutlass.Int64(cap)
            prow = p_dense_block_bitmap + qi * NW
            for m in cutlass.range(lane, n, 32, unroll=1):
                tok = _block_tok(p_first, lrow, m)
                b0 = tok
                if b0 < floor:
                    b0 = floor
                b1 = tok + 3
                if b1 > ceil:
                    b1 = ceil
                if b0 <= b1:
                    # bits r0 .. r1 of the row, in one word or two adjacent ones
                    r0 = b0 - s
                    r1 = b1 - s
                    m0 = ~((I32(1) << (r0 & 31)) - I32(1))
                    m1 = (I32(1) << (r1 & 31)) | ((I32(1) << (r1 & 31)) - I32(1))
                    if (r0 >> 5) == (r1 >> 5):
                        m0 = m0 & m1
                    prims.red(
                        prims.ReductionOp.OR,
                        prims.ReductionType.B32,
                        prow + (r0 >> 5),
                        m0,
                        mem_order=prims.MemOrder.RELAXED,
                        mem_scope=prims.MemScope.GPU,
                    )
                    if (r0 >> 5) != (r1 >> 5):
                        prims.red(
                            prims.ReductionOp.OR,
                            prims.ReductionType.B32,
                            prow + (r1 >> 5),
                            m1,
                            mem_order=prims.MemOrder.RELAXED,
                            mem_scope=prims.MemScope.GPU,
                        )


qsa_bwd_bitmap_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _bitmap_host(
    ids: cute.Tensor,
    counts: cute.Tensor,
    first: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    seq_start: cute.Tensor,
    dense_hi: cute.Tensor,
    dense_block_bitmap: cute.Tensor,
    NW: I32,
    stream: cuda_driver.CUstream,
):
    T = counts.shape[0]
    cap = ids.shape[1]
    qsa_bwd_bitmap_kernel(ids, counts, first, lo, hi, seq_start, dense_hi, dense_block_bitmap, I32(T), I32(cap), NW).launch(
        grid=((I32(T) + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream
    )


FIN_THREADS = 256


@cute.kernel
def qsa_bwd_finalize_kernel(
    g_sparse_dk_acc: cute.Tensor,
    g_sparse_dv_acc: cute.Tensor,
    g_dense_dk_acc: cute.Tensor,
    g_dense_dv_acc: cute.Tensor,
    g_dq_acc: cute.Tensor,
    g_dk: cute.Tensor,
    g_dv: cute.Tensor,
    g_dq: cute.Tensor,
    T: I32,
    HAS_SPARSE: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
):
    HQ = GQ * HKV
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    p_sparse_dk_acc = g_sparse_dk_acc.iterator.raw_ptr()
    p_sparse_dv_acc = g_sparse_dv_acc.iterator.raw_ptr()
    p_dense_dk_acc = g_dense_dk_acc.iterator.raw_ptr()
    p_dense_dv_acc = g_dense_dv_acc.iterator.raw_ptr()
    p_dq_acc = g_dq_acc.iterator.raw_ptr()
    p_dk = g_dk.iterator.raw_ptr()
    p_dv = g_dv.iterator.raw_ptr()
    p_dq = g_dq.iterator.raw_ptr()
    nitems = T * (HKV * D // 8)
    for it in cutlass.range(bid * FIN_THREADS + tidx, nitems, nb * FIN_THREADS, unroll=1):
        eo = it * 8
        ak = (p_dense_dk_acc + eo).load(count=8, alignment=16)
        av = (p_dense_dv_acc + eo).load(count=8, alignment=16)
        if cutlass.const_expr(HAS_SPARSE > 0):
            ak = ak + (p_sparse_dk_acc + eo).load(count=8, alignment=16)
            av = av + (p_sparse_dv_acc + eo).load(count=8, alignment=16)
        (p_dk + eo).store(ak.to(BF16), alignment=16)
        (p_dv + eo).store(av.to(BF16), alignment=16)
    nq8 = T * (HQ * D // 8)
    for it in cutlass.range(bid * FIN_THREADS + tidx, nq8, nb * FIN_THREADS, unroll=1):
        (p_dq + it * 8).store((p_dq_acc + it * 8).load(count=8, alignment=16).to(F32).to(BF16), alignment=16)


qsa_bwd_finalize_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _fin_host(
    sparse_dk_acc: cute.Tensor,
    sparse_dv_acc: cute.Tensor,
    dense_dk_acc: cute.Tensor,
    dense_dv_acc: cute.Tensor,
    dq_acc: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dq: cute.Tensor,
    nblocks: I32,
    HAS_SPARSE: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = dk.shape[0]
    qsa_bwd_finalize_kernel(sparse_dk_acc, sparse_dv_acc, dense_dk_acc, dense_dv_acc, dq_acc, dk, dv, dq, I32(T), HAS_SPARSE, HKV).launch(
        grid=(nblocks, 1, 1), block=(FIN_THREADS, 1, 1), stream=stream
    )


@cute.jit
def _dense_sel(p_bitmap, p_seq_start, qc, tok, NW):
    # query qc's selection word holding key token tok (zero when the token lies outside the row's span, below the
    # query's sequence or past its dense prefix) and the token's position rel in the row
    rel = tok - (p_seq_start + qc).load()
    wi = rel >> 5
    oob = (rel < 0) | (wi >= NW)
    if oob:
        wi = I32(0)
    sw = (p_bitmap + (qc * NW + wi)).load()
    if oob:
        sw = I32(0)
    return sw, rel


@cute.jit
def _rot(k, off, nk):
    r = k + off
    if r >= nk:
        r = r - nk
    return r


# Key-tile-major path for the queries below num_dense_queries. The two CTAs of a cluster split each group's work by operand, so every
# MMA keeps M=128 and each CTA holds one [128 x 256] fp32 dV or dK accumulator in TMEM across all groups.
DP_TT = 128
DP_G = 4  # queries per group: with 12 heads N = 48, the widest N whose S^T / dP^T, dQ^T and P / dS slices fit the TMEM columns
DP_N = DP_G * GQ
DP_THREADS = 512
DP_KSLAB = DP_TT * 128
DP_QSLAB = DP_N * 128
DP_OFF_T = 0
DP_NE = 2  # early ring (B of the score MMA: Q on rank 0, dO on rank 1), released right after S^T / dP^T
DP_NL = 2  # late ring (B of dV / dK: dO on rank 0, Q on rank 1), held until the backward MMAs of the group
DP_HS = 4 * DP_QSLAB
DP_OFF_E = 2 * DP_KSLAB
DP_OFF_L = DP_OFF_E + DP_NE * DP_HS
DP_CM = 16 * 6 * 128
DP_NX = 3  # rank 0's dQ^T holds an exchange buffer for DQL = 2 more groups, so a third buffer keeps the exchange moving
DP_OFF_P = DP_OFF_L + DP_NL * DP_HS
DP_OFF_DS = DP_OFF_P + DP_NX * DP_CM
DP_ST_W = DP_N * 128
DP_OFF_ST = DP_OFF_DS + DP_NX * DP_CM
DP_SMEM = DP_OFF_ST + 4 * DP_ST_W
DP_TM_SD = 0
DP_TM_DQ = 64
DP_TM_A = 128
DP_TM_KV = 256
DP_TM_PA = (48, 56, 112)  # P^T / dS^T slices reuse the TMEM columns left free after S^T (0-47) and dQ^T (64-111)


@cute.kernel
def qsa_bwd_dense_pair_kernel(
    tma_kd: cutlass.GridConstant[cuda.TensorMap],
    tma_qd: cutlass.GridConstant[cuda.TensorMap],
    tma_dod: cutlass.GridConstant[cuda.TensorMap],
    tma_dq: cutlass.GridConstant[cuda.TensorMap],
    g_k: cute.Tensor,
    g_v: cute.Tensor,
    g_lse: cute.Tensor,
    g_delta: cute.Tensor,
    g_dense_block_bitmap: cute.Tensor,
    g_dense_dk_acc: cute.Tensor,
    g_dense_dv_acc: cute.Tensor,
    g_dense_done_count: cute.Tensor,
    g_seq_start: cute.Tensor,
    g_dense_hi: cute.Tensor,
    g_unit: cute.Tensor,
    T: I32,
    NW: I32,
    ntiles: I32,
    U: I32,
    nparts: I32,
    adapt: I32,
    scale: F32,
    PDL: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
):
    HQ = GQ * HKV
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % 32
    rank = bid & 1
    pr = bid >> 1
    if cutlass.const_expr(HKV == 2):
        hk = pr & 1
        rest = pr >> 1
    else:
        hk = I32(0)
        rest = pr
    # the pair's key tile and unit. Key tokens and query rows are positions in the packed buffer, the per-token tables
    # give every query its sequence start and dense limit.
    part = rest // ntiles
    tile = rest - part * ntiles
    a0 = tile * DP_TT
    p_seq_start = g_seq_start.iterator.raw_ptr()
    p_dense_hi = g_dense_hi.iterator.raw_ptr()
    # the dense queries this key tile can reach end at the largest dense limit among the sequences holding its tokens
    dh = I32(0)
    for u in cutlass.range_constexpr(DP_TT // 32):
        tt = a0 + u * 32 + lane
        if tt < T:
            x = (p_dense_hi + tt).load()
            if x > dh:
                dh = x
    dh = cute.arch.warp_reduction_max(dh)
    g_cap = (T + (DP_G - 1)) // DP_G
    g_top = (dh + (DP_G - 1)) // DP_G
    if g_top > g_cap:
        g_top = g_cap
    g_lo = a0 // DP_G
    # Without a sparse grid to backfill, the unit is one wave of the longest sequence's causal work per pair slot
    # (decided on the device, where the sequences are known), raised when the tile's parts could not cover its groups
    # otherwise. With a sparse grid, the tuned unit backfills the key-major grid.
    Ut = U
    if adapt != 0:
        Ut = g_unit.iterator.raw_ptr().load()
        need = (g_top - g_lo + nparts - 1) // nparts
        if need > Ut:
            Ut = need
        if Ut > U:
            Ut = U
    g_top = g_top - part * Ut
    g_bot = g_top - Ut
    if g_bot > g_lo:
        g_lo = g_bot
    nk = g_top - g_lo
    if a0 >= dh:
        nk = I32(0)
    if nk > 0:
        off = (tile * 3) % nk
        peer = rank ^ 1

        smem = cutlass.Array(I32, DP_SMEM // 4, space=cutlass.AddressSpace.smem, alignment=1024)
        b_tile = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_vt = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_ef = cutlass.Array(cutlass.Int64, DP_NE, space=cutlass.AddressSpace.smem, alignment=8)
        b_ee = cutlass.Array(cutlass.Int64, DP_NE, space=cutlass.AddressSpace.smem, alignment=8)
        b_lf = cutlass.Array(cutlass.Int64, DP_NL, space=cutlass.AddressSpace.smem, alignment=8)
        b_le = cutlass.Array(cutlass.Int64, DP_NL, space=cutlass.AddressSpace.smem, alignment=8)
        b_sf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_sr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_pf = cutlass.Array(cutlass.Int64, DP_NX, space=cutlass.AddressSpace.smem, alignment=8)
        b_pe = cutlass.Array(cutlass.Int64, DP_NX, space=cutlass.AddressSpace.smem, alignment=8)
        b_xr = cutlass.Array(cutlass.Int64, DP_NX, space=cutlass.AddressSpace.smem, alignment=8)
        b_xe = cutlass.Array(cutlass.Int64, DP_NX, space=cutlass.AddressSpace.smem, alignment=8)
        b_dqf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_dqe = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_kvd = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_ks = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

        sbase = smem.data_ptr()
        if warp == 0:
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_kd.get_ptr())
                prims.prefetch_tensormap(tma_qd.get_ptr())
                prims.prefetch_tensormap(tma_dod.get_ptr())
                prims.prefetch_tensormap(tma_dq.get_ptr())
                prims.mbarrier_init(b_tile, 1)
                prims.mbarrier_init(b_vt, 8)
                for s in cutlass.range_constexpr(DP_NE):
                    prims.mbarrier_init(b_ef.subview(s), 1)
                    prims.mbarrier_init(b_ee.subview(s), 1)
                for s in cutlass.range_constexpr(DP_NL):
                    prims.mbarrier_init(b_lf.subview(s), 1)
                    prims.mbarrier_init(b_le.subview(s), 1)
                prims.mbarrier_init(b_sf, 1)
                prims.mbarrier_init(b_sr, 8)
                for s in cutlass.range_constexpr(DP_NX):
                    prims.mbarrier_init(b_pf.subview(s), 8)
                    prims.mbarrier_init(b_pe.subview(s), 1)
                    prims.mbarrier_init(b_xr.subview(s), 1)
                    prims.mbarrier_init(b_xe.subview(s), 1)
                prims.fence_mbarrier_init()
                # receiver-armed exchange: each rank arms its own "received" barriers for the first DP_NX groups
                for s in cutlass.range_constexpr(DP_NX):
                    if s < nk:
                        prims.mbarrier_arrive_expect_tx(b_xr.subview(s), DP_CM)
                prims.mbarrier_init(b_dqf, 1)
                prims.mbarrier_init(b_dqe, 4)
                prims.mbarrier_init(b_kvd, 1)
                prims.mbarrier_init(b_ks, 4)
        if warp == 1:
            prims.tcgen05_alloc(tmem_slot, 512, group="cta_1")
            prims.tcgen05_relinquish_alloc_permit(group="cta_1")
        prims.fence_mbarrier_init()
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        prims.barrier_cta_sync(0)
        # the peer's barriers must be initialised before any remote arrive / bulk copy
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

        tmem_raw = tmem_slot.load()
        tmem_col0 = tmem_raw & I32(0xFFFF)
        tmem_row0 = tmem_raw >> I32(16)
        p_delta = g_delta.iterator.raw_ptr()
        p_dense_block_bitmap = g_dense_block_bitmap.iterator.raw_ptr()
        p_dense_dk_acc = g_dense_dk_acc.iterator.raw_ptr()
        p_dense_dv_acc = g_dense_dv_acc.iterator.raw_ptr()
        p_k = g_k.iterator.raw_ptr()
        p_v = g_v.iterator.raw_ptr()

        if warp == 0:
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(b_tile, 2 * DP_KSLAB)
                for s in cutlass.range_constexpr(2):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sbase + (DP_OFF_T + s * DP_KSLAB) // 4, tma_kd.get_ptr(), ((rank * 2 + s) * 64, hk, a0), b_tile
                    )
            for k in cutlass.range(nk, unroll=1):
                g = g_top - 1 - _rot(k, off, nk)
                se = k % DP_NE
                _wait(b_ee.subview(se), ((k // DP_NE) & 1) ^ 1)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(b_ef.subview(se), DP_HS)
                    for s in cutlass.range_constexpr(4):
                        if rank == 0:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sbase + (DP_OFF_E + se * DP_HS + s * DP_QSLAB) // 4, tma_qd.get_ptr(), (s * 64, hk * GQ, g * DP_G), b_ef.subview(se)
                            )
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sbase + (DP_OFF_E + se * DP_HS + s * DP_QSLAB) // 4, tma_dod.get_ptr(), (s * 64, hk * GQ, g * DP_G), b_ef.subview(se)
                            )
        elif warp == 3:
            for k in cutlass.range(nk, unroll=1):
                g = g_top - 1 - _rot(k, off, nk)
                sl = k % DP_NL
                _wait(b_le.subview(sl), ((k // DP_NL) & 1) ^ 1)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(b_lf.subview(sl), DP_HS)
                    for s in cutlass.range_constexpr(4):
                        if rank == 0:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sbase + (DP_OFF_L + sl * DP_HS + s * DP_QSLAB) // 4, tma_dod.get_ptr(), (s * 64, hk * GQ, g * DP_G), b_lf.subview(sl)
                            )
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sbase + (DP_OFF_L + sl * DP_HS + s * DP_QSLAB) // 4, tma_qd.get_ptr(), (s * 64, hk * GQ, g * DP_G), b_lf.subview(sl)
                            )
        elif warp == 1:
            idesc_s = prims.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=BF16, b_dtype=BF16, n_dim=DP_N, m_dim=128, a_major=0, b_major=0)
            idesc_q = prims.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=BF16, b_dtype=BF16, n_dim=DP_N, m_dim=128, a_major=1, b_major=1)
            idesc_kv = prims.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=BF16, b_dtype=BF16, n_dim=256, m_dim=128, a_major=0, b_major=1)
            SW128 = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
            SWN = prims.Tcgen05SmemSwizzle.NONE
            d_kmn = _desc(sbase + DP_OFF_T // 4, DP_KSLAB, 1024, SW128)
            d_e = _desc(sbase + DP_OFF_E // 4, 16, 1024, SW128)
            d_lmn = _desc(sbase + DP_OFF_L // 4, DP_QSLAB, 1024, SW128)
            d_dsb = _desc(sbase + DP_OFF_DS // 4, 768, 128, SWN)
            t_sd = prims.make_tmem_ptr(tmem_raw + DP_TM_SD, F32)
            t_dq = prims.make_tmem_ptr(tmem_raw + DP_TM_DQ, F32)
            t_kv = prims.make_tmem_ptr(tmem_raw + DP_TM_KV, F32)
            _wait(b_tile, 0)
            _wait(b_vt, 0)
            _wait(b_ks, 0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            # rank 0 issues dQ^T(k - DQL) after the backward MMAs of k - 1, so the dS round trip through rank 1
            # overlaps two groups of rank-0 work
            DQL = 2
            for k in cutlass.range(nk + DQL, unroll=1):
                se = k % DP_NE
                if k < nk:
                    if k >= 1:
                        _wait(b_sr, (k - 1) & 1)
                    _wait(b_ef.subview(se), (k // DP_NE) & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    qoff = se * DP_HS
                    if prims.elect_sync():
                        for ks in cutlass.range_constexpr(16):
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16,
                                prims.CTAGroup.CTA_1,
                                t_sd,
                                prims.make_tmem_ptr(tmem_raw + (DP_TM_A + ks * 8), BF16),
                                d_e.advance_start_address(qoff + (ks >> 2) * DP_QSLAB + (ks & 3) * 32),
                                idesc_s,
                                ks > 0,
                            )
                        prims.tcgen05_commit(b_sf, group=prims.CTAGroup.CTA_1)
                        prims.tcgen05_commit(b_ee.subview(se), group=prims.CTAGroup.CTA_1)
                if k >= 1 and k <= nk:
                    kb = k - 1
                    sl = kb % DP_NL
                    qob = sl * DP_HS
                    xb = kb % DP_NX
                    acc0 = kb > 0
                    _wait(b_lf.subview(sl), (kb // DP_NL) & 1)
                    _wait(b_pf.subview(xb), (kb // DP_NX) & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    if rank == 0:
                        if prims.elect_sync():
                            for kk in cutlass.range_constexpr(DP_N // 16):
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F16,
                                    prims.CTAGroup.CTA_1,
                                    t_kv,
                                    prims.make_tmem_ptr(tmem_raw + DP_TM_PA[kk], BF16),
                                    d_lmn.advance_start_address(qob + kk * 2048),
                                    idesc_kv,
                                    acc0 | (kk > 0),
                                )
                            prims.tcgen05_commit(b_pe.subview(xb), group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_le.subview(sl), group=prims.CTAGroup.CTA_1)
                            if kb == nk - 1:
                                prims.tcgen05_commit(b_kvd, group=prims.CTAGroup.CTA_1)
                    else:
                        if kb >= 1:
                            _wait(b_dqe, (kb - 1) & 1)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if prims.elect_sync():
                            for kk in cutlass.range_constexpr(DP_N // 16):
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F16,
                                    prims.CTAGroup.CTA_1,
                                    t_kv,
                                    prims.make_tmem_ptr(tmem_raw + DP_TM_PA[kk], BF16),
                                    d_lmn.advance_start_address(qob + kk * 2048),
                                    idesc_kv,
                                    acc0 | (kk > 0),
                                )
                            for kk in cutlass.range_constexpr(8):
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F16,
                                    prims.CTAGroup.CTA_1,
                                    t_dq,
                                    d_kmn.advance_start_address(kk * 2048),
                                    d_dsb.advance_start_address(xb * DP_CM + kk * 1536),
                                    idesc_q,
                                    kk > 0,
                                )
                            prims.tcgen05_commit(b_pe.subview(xb), group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_le.subview(sl), group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_dqf, group=prims.CTAGroup.CTA_1)
                            if kb == nk - 1:
                                prims.tcgen05_commit(b_kvd, group=prims.CTAGroup.CTA_1)
                if rank == 0:
                    if k >= DQL:
                        # Rank 0's half of dQ^T for group k - DQL needs the dS that rank 1 sends back.
                        kq = k - DQL
                        xq = kq % DP_NX
                        _wait(b_xr.subview(xq), (kq // DP_NX) & 1)
                        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                        if kq + DP_NX < nk:
                            if prims.elect_sync():
                                # arm the buffer for dS(kq + DP_NX). Rank 1 only sends it after P(kq + DP_NX),
                                # which our softmax sends only after the dQ^T MMAs below have completed.
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(xq), DP_CM)
                        if kq >= 1:
                            _wait(b_dqe, (kq - 1) & 1)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if prims.elect_sync():
                            for kk in cutlass.range_constexpr(8):
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F16,
                                    prims.CTAGroup.CTA_1,
                                    t_dq,
                                    d_kmn.advance_start_address(kk * 2048),
                                    d_dsb.advance_start_address(xq * DP_CM + kk * 1536),
                                    idesc_q,
                                    kk > 0,
                                )
                            prims.tcgen05_commit(b_dqf, group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_xe.subview(xq), group=prims.CTAGroup.CTA_1)
        elif warp >= 4 and warp < 12:
            hf = (warp - 4) >> 2
            qd = warp & 3
            r = qd * 32 + lane
            tok = a0 + r
            row_ok = tok < T
            c_s = scale * F32(LOG2E)
            xoff = (DP_OFF_P + (r >> 3) * 768 + (r & 7) * 16) // 4 + hf * 96
            xoffd = (DP_OFF_DS + (r >> 3) * 768 + (r & 7) * 16) // 4 + hf * 96
            # Lane r writes key row tok of K (rank 0) or V (rank 1) to TMEM as packed bf16 pairs, the score MMA's A operand.
            vok = tok < T
            asrc = p_k
            if rank == 1:
                asrc = p_v
            vrow = asrc + (tok * HKV + hk) * D
            for c2 in cutlass.range_constexpr(2):
                c4 = hf * 2 + c2
                stw = []
                for u in cutlass.range_constexpr(8):
                    ldv = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
                    if vok:
                        ldv = (vrow + c4 * 64 + u * 8).load(count=8, alignment=16).bitcast(I32)
                    for ee in cutlass.range_constexpr(4):
                        stw.append(ldv[ee])
                prims.tcgen05_st(
                    "32x32b",
                    prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_A + c4 * 32).as_ptr(I32),
                    cutlass.Vector.from_elements(stw, I32),
                )
            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(b_vt)
            scol = tmem_col0 + DP_TM_SD + hf * 24
            if rank == 0:
                # Selection words, block indices and lse load one group ahead to hide their latency.
                g_n = g_top - 1 - _rot(I32(0), off, nk)
                sw_n = []
                bq_n = []
                av_n = []
                for jj in cutlass.range_constexpr(2):
                    qc = g_n * DP_G + hf * 2 + jj
                    if qc >= T:
                        qc = T - 1
                    sw_j, bq_j = _dense_sel(p_dense_block_bitmap, p_seq_start, qc, tok, NW)
                    sw_n.append(sw_j)
                    bq_n.append(bq_j)
                    ab = qc * HQ + hk * GQ
                    av_n.append(
                        [
                            _load_lse4(g_lse, ab, HQ),
                            _load_lse4(g_lse, ab + 4, HQ),
                            _load_lse4(g_lse, ab + 8, HQ),
                        ]
                    )
                for k in cutlass.range(nk, unroll=1):
                    g = g_top - 1 - _rot(k, off, nk)
                    sw = sw_n
                    bqs = bq_n
                    av = av_n
                    kn = k + 1
                    if kn >= nk:
                        kn = nk - 1
                    g_n = g_top - 1 - _rot(kn, off, nk)
                    sw_n = []
                    bq_n = []
                    av_n = []
                    for jj in cutlass.range_constexpr(2):
                        qc = g_n * DP_G + hf * 2 + jj
                        if qc >= T:
                            qc = T - 1
                        sw_j, bq_j = _dense_sel(p_dense_block_bitmap, p_seq_start, qc, tok, NW)
                        sw_n.append(sw_j)
                        bq_n.append(bq_j)
                        ab = qc * HQ + hk * GQ
                        av_n.append(
                            [
                                _load_lse4(g_lse, ab, HQ),
                                _load_lse4(g_lse, ab + 4, HQ),
                                _load_lse4(g_lse, ab + 8, HQ),
                            ]
                        )
                    _wait(b_sf, k & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    xv0 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol).as_ptr(F32), num=16)
                    xv1 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol + 16).as_ptr(F32), num=8)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_sr)
                    opk = []
                    for jj in cutlass.range_constexpr(2):
                        qi = g * DP_G + hf * 2 + jj
                        sel = (sw[jj] >> (bqs[jj] & 31)) & 1
                        # The token is visible when its bit is set: the bitmap kernel applied the sequence, the key
                        # bounds and causality per token, and a sparse query's row has no bits.
                        valid = row_ok & (qi < T) & (sel != 0)
                        ow = []
                        for hh in cutlass.range_constexpr(GQ):
                            n = jj * GQ + hh
                            a = av[jj][hh // 4][hh % 4]
                            if cutlass.const_expr(n < 16):
                                x = xv0[n]
                            else:
                                x = xv1[n - 16]
                            e = cute.math.exp2(x * c_s - a * F32(LOG2E), fastmath=True)
                            pj = F32(0.0)
                            if valid:
                                pj = e
                            ow.append(pj)
                        pk = cutlass.Vector.from_elements(ow, F32).to(BF16).bitcast(I32)
                        for u in cutlass.range_constexpr(GQ // 2):
                            opk.append(pk[u])
                    ovec = cutlass.Vector.from_elements(opk, I32)
                    xb = k % DP_NX
                    if k >= 1:
                        # dV(k-1) has read the TMEM P^T slot
                        kp = k - 1
                        _wait(b_pe.subview(kp % DP_NX), (kp // DP_NX) & 1)
                    if k >= DP_NX:
                        # dQ^T(k - DP_NX) has completed: rank 1 has read P(k - DP_NX) (dS(k - DP_NX) arrived) and may receive into its
                        # buffer again, and our dS buffer is free for dS(k)
                        _wait(b_xe.subview(xb), ((k // DP_NX) & 1) ^ 1)
                    if hf == 0:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[0]).as_ptr(I32), ovec[0:8])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1]).as_ptr(I32), ovec[8:12])
                    else:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1] + 4).as_ptr(I32), ovec[0:4])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[2]).as_ptr(I32), ovec[4:12])
                    rb = prims.mapa(b_xr.subview(xb), peer)
                    for nc in cutlass.range_constexpr(3):
                        _st_async16(prims.mapa(sbase + xoff + xb * (DP_CM // 4) + nc * 32, peer), ovec[nc * 4 : nc * 4 + 4], rb)
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_pf.subview(xb))
            else:
                g_n = g_top - 1 - _rot(I32(0), off, nk)
                av_n = []
                for jj in cutlass.range_constexpr(2):
                    qc = g_n * DP_G + hf * 2 + jj
                    if qc >= T:
                        qc = T - 1
                    ab = qc * HQ + hk * GQ
                    av_n.append(
                        [
                            (p_delta + ab).load(count=4, alignment=16),
                            (p_delta + ab + 4).load(count=4, alignment=16),
                            (p_delta + ab + 8).load(count=4, alignment=16),
                        ]
                    )
                for k in cutlass.range(nk, unroll=1):
                    av = av_n
                    kn = k + 1
                    if kn >= nk:
                        kn = nk - 1
                    g_n = g_top - 1 - _rot(kn, off, nk)
                    av_n = []
                    for jj in cutlass.range_constexpr(2):
                        qc = g_n * DP_G + hf * 2 + jj
                        if qc >= T:
                            qc = T - 1
                        ab = qc * HQ + hk * GQ
                        av_n.append(
                            [
                                (p_delta + ab).load(count=4, alignment=16),
                                (p_delta + ab + 4).load(count=4, alignment=16),
                                (p_delta + ab + 8).load(count=4, alignment=16),
                            ]
                        )
                    _wait(b_sf, k & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    xv0 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol).as_ptr(F32), num=16)
                    xv1 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol + 16).as_ptr(F32), num=8)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_sr)
                    xb = k % DP_NX
                    _wait(b_xr.subview(xb), (k // DP_NX) & 1)
                    pin = []
                    for nc in cutlass.range_constexpr(3):
                        pin.append((sbase + xoff + xb * (DP_CM // 4) + nc * 32).load(count=4, alignment=16))
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if k + DP_NX < nk:
                        if warp == 4:
                            if prims.elect_sync():
                                # arm this buffer for P(k + DP_NX): rank 0 sends it only after dS(k) (whose quadrant-0
                                # copy this warp issues after this arm) has been consumed by its dQ^T
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(xb), DP_CM)
                    dpk = []
                    for jj in cutlass.range_constexpr(2):
                        dw = []
                        for hh in cutlass.range_constexpr(GQ):
                            n = jj * GQ + hh
                            a = av[jj][hh // 4][hh % 4]
                            if cutlass.const_expr(n < 16):
                                x = xv0[n]
                            else:
                                x = xv1[n - 16]
                            dw.append(x - a)
                        for hp in cutlass.range_constexpr(GQ // 2):
                            n0 = jj * GQ + 2 * hp
                            pw = pin[n0 // 8][(n0 % 8) // 2]
                            pv = cutlass.Vector.from_elements([pw], I32).bitcast(BF16).to(F32)
                            # a masked token has P = 0 and must give dS = 0 even when its dP is nonfinite
                            d0 = F32(0.0)
                            d1 = F32(0.0)
                            if pv[0] != 0.0:
                                d0 = pv[0] * dw[2 * hp] * scale
                            if pv[1] != 0.0:
                                d1 = pv[1] * dw[2 * hp + 1] * scale
                            dpk.append(cutlass.Vector.from_elements([d0, d1], F32).to(BF16).bitcast(I32)[0])
                    dvec = cutlass.Vector.from_elements(dpk, I32)
                    if k >= 1:
                        # dK / dQ^T(k-1) have read the TMEM dS^T slot (and k - DP_NX this SMEM buffer). The copy of dS(k - DP_NX)
                        # has landed and rank 0's dQ^T(k - DP_NX) has consumed it (both implied by the arrival of P(k)).
                        kp = k - 1
                        _wait(b_pe.subview(kp % DP_NX), (kp // DP_NX) & 1)
                    if hf == 0:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[0]).as_ptr(I32), dvec[0:8])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1]).as_ptr(I32), dvec[8:12])
                    else:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1] + 4).as_ptr(I32), dvec[0:4])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[2]).as_ptr(I32), dvec[4:12])
                    rb = prims.mapa(b_xr.subview(xb), peer)
                    for nc in cutlass.range_constexpr(3):
                        (sbase + xoffd + xb * (DP_CM // 4) + nc * 32).store(dvec[nc * 4 : nc * 4 + 4], alignment=16)
                        _st_async16(prims.mapa(sbase + xoffd + xb * (DP_CM // 4) + nc * 32, peer), dvec[nc * 4 : nc * 4 + 4], rb)
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_pf.subview(xb))
            _wait(b_kvd, 0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            obase = (tok * HKV + hk) * D + hf * 128
            pdst = p_dense_dv_acc
            if rank == 1:
                pdst = p_dense_dk_acc
            for c8 in cutlass.range_constexpr(4):
                vv = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_KV + hf * 128 + c8 * 32).as_ptr(F32), num=32)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                if row_ok:
                    for u in cutlass.range_constexpr(8):
                        prims.red(
                            prims.ReductionOp.ADD,
                            prims.ReductionType.F32,
                            pdst + (obase + c8 * 32 + u * 4),
                            vv[u * 4 : u * 4 + 4],
                            mem_order=prims.MemOrder.RELAXED,
                            mem_scope=prims.MemScope.GPU,
                        )
        elif warp >= 12:
            qd = warp - 12
            # Zero the nonfinite values of the SMEM K tile, the A operand of dQ^T: a masked token then adds exact
            # zeros, while a selected one still poisons its queries through S^T, which reads the TMEM copy of K.
            _wait(b_tile, 0)
            kt = sbase + DP_OFF_T // 4
            for it in cutlass.range_constexpr((2 * DP_KSLAB) // (16 * 128)):
                koff = (it * 128 + (tidx - 384)) * 4
                w4 = (kt + koff).load(count=4, alignment=16)
                ws = []
                nbad = I32(0)
                for e in cutlass.range_constexpr(4):
                    w = w4[e]
                    if (w & I32(0x7F80)) == I32(0x7F80):
                        w = w & I32(-65536)
                        nbad = nbad + 1
                    if (w & I32(0x7F800000)) == I32(0x7F800000):
                        w = w & I32(0xFFFF)
                        nbad = nbad + 1
                    ws.append(w)
                if nbad > 0:
                    (kt + koff).store(cutlass.Vector.from_elements(ws, I32), alignment=16)
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(b_ks)
            stg0 = sbase + (DP_OFF_ST + qd * DP_ST_W) // 4
            # rank 0 sends P straight into the peer (st.async) and never touches its own P buffers, so they hold a
            # second staging box per drain warp: the reduce of group k-1 may still be reading while k is staged
            stg1 = sbase + (DP_OFF_P + qd * DP_ST_W) // 4
            for k in cutlass.range(nk, unroll=1):
                g = g_top - 1 - _rot(k, off, nk)
                _wait(b_dqf, k & 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                vh0 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_DQ).as_ptr(F32), num=32)
                vh1 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_DQ + 32).as_ptr(F32), num=DP_N - 32)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.bar_warp_sync(0xFFFFFFFF)
                if prims.elect_sync():
                    prims.mbarrier_arrive(b_dqe)
                # the reduce that last used this staging box must have finished reading it. Rank 0 has two boxes,
                # so one reduce may stay in flight, and rank 1 has one.
                stg = stg0
                if rank == 0:
                    if (k & 1) == 1:
                        stg = stg1
                if rank == 0:
                    prims.cp_async_bulk_wait_group(1, read=True)
                else:
                    prims.cp_async_bulk_wait_group(0, read=True)
                prims.bar_warp_sync(0xFFFFFFFF)
                for n in cutlass.range_constexpr(DP_N):
                    if cutlass.const_expr(n < 32):
                        val = vh0[n]
                    else:
                        val = vh1[n - 32]
                    (stg + n * 32 + lane).store(prims.mov_b32(val, target_type=I32))
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.bar_warp_sync(0xFFFFFFFF)
                if lane == 0:
                    prims.cp_async_bulk_tensor_reduce(tma_dq.get_ptr(), stg, prims.TMARedux.ADD, [rank * 128 + qd * 32, hk * GQ, g * DP_G])
                prims.cp_async_bulk_commit_group()
            prims.cp_async_bulk_wait_group(0)

        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        # nobody leaves while the peer may still touch our shared memory
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
        if warp == 1:
            prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw, F32), 512, group="cta_1")

    if cutlass.const_expr(PDL > 0):
        # Only the last dependent CTA waits for the sparse primary. Dense-grid completion then
        # orders the normally launched final combine after both accumulator producers.
        if tidx == 0:
            nb, _, _ = cute.arch.grid_dim()
            old = prims.atomicrmw("add", g_dense_done_count.iterator.raw_ptr(), I32(1), mem_order=prims.MemOrder.RELAXED, syncscope=prims.MemScope.GPU)
            if old == nb - 1:
                prims.griddepcontrol("wait")


qsa_bwd_dense_pair_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _dpair_host(
    k: cute.Tensor,
    v: cute.Tensor,
    q: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dense_block_bitmap: cute.Tensor,
    dq_acc: cute.Tensor,
    dense_dk_acc: cute.Tensor,
    dense_dv_acc: cute.Tensor,
    dense_done_count: cute.Tensor,
    seq_start: cute.Tensor,
    dense_hi: cute.Tensor,
    unit: cute.Tensor,
    NW: I32,
    ntiles: I32,
    U: I32,
    nparts: I32,
    adapt: I32,
    scale: F32,
    PDL: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    HQ = GQ * HKV
    T = k.shape[0]
    tma_kd = cuda.create_tensor_map_tiled_from_view(k, box_dims=(DP_TT, 1, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_qd = cuda.create_tensor_map_tiled_from_view(q, box_dims=(DP_G, GQ, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_dod = cuda.create_tensor_map_tiled_from_view(do, box_dims=(DP_G, GQ, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_dq = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(dq_acc.iterator, cute.make_layout((T, HQ, D), stride=(HQ * D, D, 1))),
        box_dims=(DP_G, GQ, 32),
        stride_order=(2, 1, 0),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    qsa_bwd_dense_pair_kernel(
        tma_kd,
        tma_qd,
        tma_dod,
        tma_dq,
        k,
        v,
        lse,
        delta,
        dense_block_bitmap,
        dense_dk_acc,
        dense_dv_acc,
        dense_done_count,
        seq_start,
        dense_hi,
        unit,
        I32(T),
        NW,
        ntiles,
        U,
        nparts,
        adapt,
        scale,
        PDL,
        HKV,
    ).launch(grid=(nparts * ntiles * 2 * HKV, 1, 1), block=(DP_THREADS, 1, 1), cluster=(2, 1, 1), stream=stream, use_pdl=(PDL == 1))


# Tuned dense/sparse crossover: below it row i selects 2048 of its i + 1 tokens, dense enough that keeping dK/dV on
# chip beats the record round trip of a (query, block) pair. A sparse remainder under SPARSE_FOLD queries is folded in
# too.
DENSE_QUERIES_MAX = 20480
SPARSE_FOLD = 2048
KV_PASS = 32768  # KV pass of the query-major kernel (tokens), so one pass's K/V blocks stay L2 resident
# From this length on a key holds few enough queries that single-query stages are common, and skipping their empty
# K-step pays for the per-stage branch.
KEY_SKIP_MIN_T = 196608
DP_U = 320  # tuned query groups per dense work unit, small enough to backfill the key-major grid's tail evenly
DP_U_MAX = 448  # query groups per dense work unit beyond which a larger unit no longer amortizes the key tile load and dK / dV flush

# where the sequence bounds come from
SEQ_SINGLE = 0  # the T rows are one sequence, or documents the caller did not describe
SEQ_CU = 1  # the caller's cu_seqlens
SEQ_ORIGIN = 2  # derived from the per-token sequence origins (tok_base of the bitmap API)


def max_packed_len(hkv: int) -> int:
    # The kernels address the [T, 12 * hkv, 256] tensors with 32-bit element offsets.
    return (2**31 - 1) // (GQ * hkv * D)


def _dense_cap(max_len: int, single: bool) -> int:
    # the most dense queries a sequence can have under the device-side _dense_queries rule: the one sequence of
    # max_len tokens, or any packed sequence up to max_len tokens (dense throughout just below the crossover)
    if single:
        return max_len if max_len - DENSE_QUERIES_MAX < SPARSE_FOLD else DENSE_QUERIES_MAX
    return min(max_len, DENSE_QUERIES_MAX + SPARSE_FOLD - 1)


class LaunchPlan:
    """Launch geometry and scratch layout for one packed length, KV head count and sequence source.

    ``seq_mode`` SEQ_SINGLE treats the T rows as one sequence, SEQ_CU takes the caller's ``cu_seqlens`` of ``num_seqs``
    sequences, SEQ_ORIGIN derives the sequences on the device from the per-token origins (at most ``num_seqs`` of
    them). The per-sequence dense / sparse split is decided on the device.
    """

    def __init__(self, seq_len: int, num_sms: int, hkv: int, seq_mode: int, num_seqs: int = 1, max_seq_len: Optional[int] = None, arch: int = 100):
        T = int(seq_len)
        assert hkv in (1, 2), f"{hkv} KV heads"
        assert 1 <= T <= max_packed_len(hkv), f"sequence length {T} outside [1, {max_packed_len(hkv)}]"
        HKV = hkv
        HQ = GQ * hkv
        REC_W = _rec_w(hkv)
        PAD = _pad(hkv)
        self.hkv = hkv
        self.arch = int(arch)
        self.seq_len = T
        self.seq_mode = seq_mode
        self.seq_count = max(int(num_seqs), 1)
        max_len = T if max_seq_len is None else min(int(max_seq_len), T)
        dense_cap = _dense_cap(max_len, seq_mode == SEQ_SINGLE)
        self.has_sparse = dense_cap < max_len
        # one selection bit per key token of the dense prefix, a row per dense query: the first dense_cap rows of one
        # sequence, any row of packed ones
        self.nw = (dense_cap + 31) // 32
        bitmap_rows = min(T, dense_cap) if seq_mode == SEQ_SINGLE else T
        self.nsel = (bitmap_rows * self.nw + 64 + 3) // 4 * 4
        # Key tiles of the dense pair kernel cover the buffer, or only its dense prefix when the rows are one sequence.
        dense_tokens = T if seq_mode != SEQ_SINGLE else min(T, dense_cap)
        self.dense_tiles = (dense_tokens + DP_TT - 1) // DP_TT
        gd = (dense_cap + DP_G - 1) // DP_G
        # A (key tile, unit) pair of CTAs walks its query groups serially behind one key tile load and one dK / dV flush.
        # Without a sparse grid to backfill, the unit hands every pair slot one wave of a sequence's causal work (key
        # tile t sees the groups from its first token up), so the fixed cost is paid once per slot and the waves stay
        # balanced.
        causal_work = HKV * sum(gd - t * (DP_TT // DP_G) for t in range((dense_cap + DP_TT - 1) // DP_TT))
        self.dense_unit = DP_U if self.has_sparse else max(1, min(DP_U_MAX, -(-causal_work // (num_sms // 2))))
        self.num_sms = num_sms
        # the groups a key tile can reach: the dense queries of the sequences holding its tokens
        reach = (DP_TT + dense_cap + DP_G - 1) // DP_G
        self.dense_parts = (reach + self.dense_unit - 1) // self.dense_unit
        self.pass_tokens = KV_PASS
        self.num_passes = (T + self.pass_tokens - 1) // self.pass_tokens
        assert self.num_passes < POS, f"sequence length {T} needs {self.num_passes} KV passes, the offset rows hold {POS - 1}"
        # Sparse queries: exactly T - dense_cap of a single sequence, up to T of packed ones (counted on the device).
        nsq = 0 if not self.has_sparse else (T - dense_cap if seq_mode == SEQ_SINGLE else T)
        self.nsq_cap = nsq
        self.window_capacity = min(nsq, SPARSE_WINDOW_QUERIES)
        self.query_windows = tuple((lo, min(lo + SPARSE_WINDOW_QUERIES, nsq)) for lo in range(0, nsq, SPARSE_WINDOW_QUERIES))
        # the query-major kernel's item is a query, or a pair of queries with one KV head
        self.sparse_ctas = min(num_sms, (self.window_capacity + 1) // 2 if hkv == 1 else self.window_capacity)
        # inverted index of the sparse queries: ranges of QR queries, one key per (range, block), one slot per pair
        self.num_ranges = (self.window_capacity + QR - 1) // QR
        assert self.num_ranges <= 2, "the key-major kernel walks the active keys of at most two ranges"
        # a range's keys are the first tokens of the buffer, padded to the index scan's 8 keys per thread
        self.keys_per_range = (T + 7) // 8 * 8
        self.num_keys = self.num_ranges * self.keys_per_range
        self.dense_cap = dense_cap
        self.num_pairs = self.num_ranges * RANGE_PAIRS
        self.key_name = f"key_skip{1 if T >= KEY_SKIP_MIN_T else 0}"
        self.key_ctas = num_sms
        nz = T * HKV * D if self.has_sparse else 0
        dense_dkv_size = T * HKV * D + 64
        keys = (self.num_keys + 31) // 16 * 16
        prep_vecs = max(T * HQ // PREP_ROWS, nz // 8, (self.nsel + 4) // 4, keys // 4, T * HQ * D // 8, dense_dkv_size // 8)
        self.prep_blocks = min(num_sms * 8, (prep_vecs + 255) // 256)
        self.fin_blocks = min(num_sms * 8, (T * HKV * D // 8 + 255) // 256)
        sp = self.has_sparse
        cu_len = {SEQ_SINGLE: 4, SEQ_CU: 0, SEQ_ORIGIN: (T + 1 + 3) // 4 * 4}[seq_mode]
        T4 = (T + 3) // 4 * 4
        # Without the sparse path its buffers shrink to nothing, except the dK/dV accumulators, which keep one
        # never-read PAD chunk that both kernels alias.
        self.chunks = [
            ("delta", (T, HQ), torch.float32),
            ("sparse_dk_acc", (nz + PAD,), torch.float32),
            ("sparse_dv_acc", (nz + PAD if sp else 0,), torch.float32),
            ("dense_block_bitmap", (self.nsel + 4,), torch.int32),
            ("dq_acc", (T * HQ * D,), torch.float32),
            ("dense_dk_acc", (dense_dkv_size,), torch.float32),
            ("dense_dv_acc", (dense_dkv_size,), torch.float32),
            ("seq_lens", (cu_len,), torch.int32),
            ("seq_sparse_base", ((self.seq_count + 3) // 4 * 4,), torch.int32),
            ("sparse_count", (4,), torch.int32),
            ("window_count", (4 if sp else 0,), torch.int32),
            ("seq_start", (T4,), torch.int32),
            ("dense_hi", (T4,), torch.int32),
            ("nseq", (4,), torch.int32),
            ("dense_unit", (4,), torch.int32),
            ("active_count", (4 if sp else 0,), torch.int32),
            ("sparse_queries", ((nsq + 3) // 4 * 4 if sp else 0,), torch.int32),
            ("sparse_blocks", (self.window_capacity * PART_W + 16 if sp else 0,), torch.int32),
            ("sparse_pass_offsets", (self.window_capacity * POS + 16 if sp else 0,), torch.int32),
            ("sparse_block_slots", (self.window_capacity * PART_W + 16 if sp else 0,), torch.int32),
            ("key_pair_counts", (keys if sp else 0,), torch.int32),
            ("key_slot_offsets", (keys if sp else 0,), torch.int32),
            ("key_slot_cursors", (keys if sp else 0,), torch.int32),
            ("active_keys", (keys if sp else 0,), torch.int32),
            ("slot_queries", ((self.num_pairs + 31) // 16 * 16 if sp else 0,), torch.int32),
            ("pair_records", (self.num_pairs * REC_W + 16 if sp else 0,), torch.int32),
        ]

    def workspace_bytes(self) -> int:
        return sum(ws_align(math.prod(shape) * dtype.itemsize) for _, shape, dtype in self.chunks)

    def carve(self, carver) -> dict:
        """Scratch views from a ``cudnn.api_base.WorkspaceCarver`` over the caller's workspace."""
        scratch = {name: carver.take(math.prod(shape), dtype).view(shape) for name, shape, dtype in self.chunks if math.prod(shape)}
        if not self.has_sparse:
            scratch["sparse_dv_acc"] = scratch["sparse_dk_acc"]
            scratch["key_pair_counts"] = scratch["dense_block_bitmap"][self.nsel :]
            scratch["sparse_queries"] = scratch["seq_sparse_base"]
        return scratch


_PLANS: dict = {}


def plan_for(seq_len: int, device: torch.device, hkv: int, seq_mode: int, num_seqs: int = 1) -> LaunchPlan:
    """The plan of a geometry, built once per process."""
    key = (int(seq_len), hkv, seq_mode, int(num_seqs), device.index)
    plan = _PLANS.get(key)
    if plan is None:
        cc = torch.cuda.get_device_capability(device)
        plan = LaunchPlan(seq_len, torch.cuda.get_device_properties(device).multi_processor_count, hkv, seq_mode, num_seqs, arch=cc[0] * 10 + cc[1])
        _PLANS[key] = plan
    return plan


_COMPILED = {}


def _fake(dtype, shape, sym, align=16):
    full = tuple(sym if s is None else s for s in shape)
    order = tuple(range(len(shape) - 1, -1, -1))
    return make_fake_compact_tensor(dtype, full, stride_order=order, assumed_align=align)


def compile_kernels(plan: LaunchPlan) -> dict:
    """Compile the launch sequence of the plan's path once per process and KV head count.

    Every extent is symbolic, so one set of artifacts serves all sequence lengths of a path on one architecture.
    """
    names = ["seq_scan", "seq_fill", "prep", "bitmap"]
    if plan.has_sparse:
        names += ["window", "part", "scan", "fill", "query", plan.key_name, "dense_pdl", "fin_sparse"]
    else:
        names += ["dense", "fin"]
    hkv = plan.hkv
    HQ = GQ * hkv
    cache = _COMPILED.setdefault((hkv, plan.arch), {})
    missing = [name for name in names if name not in cache]
    if missing:
        sym_t = cute.sym_int64()
        f_q = _fake(BF16, (None, HQ, D), sym_t)
        f_kv = _fake(BF16, (None, hkv, D), sym_t)
        f_out = _fake(F32, (None, HQ, D), sym_t)
        f_lse = _fake(F32, (HQ, None), sym_t, align=4)
        f_delta = _fake(F32, (None, HQ), sym_t)
        f_h = _fake(F32, (None,), cute.sym_int64(divisibility=8))
        f_dq_acc = _fake(F32, (None,), cute.sym_int64(divisibility=8))
        f_dense_dkv = _fake(F32, (None,), cute.sym_int64(divisibility=8))

        def f_ids():
            # the compacted block keys [T, capacity]
            return make_fake_compact_tensor(I32, (sym_t, cute.sym_int64()), stride_order=(1, 0), assumed_align=4)

        def f_i():
            # scratch int32 vectors, padded to 4 entries
            return _fake(I32, (None,), cute.sym_int64(divisibility=4))

        def f_c():
            # caller int32 vectors of any length
            return _fake(I32, (None,), cute.sym_int64(), align=4)

        stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        dense_fakes = (
            f_kv,
            f_kv,
            f_q,
            f_q,
            f_lse,
            f_delta,
            f_i(),
            f_dq_acc,
            f_dense_dkv,
            f_dense_dkv,
            f_i(),
            f_i(),
            f_i(),
            f_i(),
            I32(1),
            I32(1),
            I32(1),
            I32(1),
            I32(0),
            F32(1),
        )
        fin_fakes = (f_h, f_h, f_dense_dkv, f_dense_dkv, f_dq_acc, f_kv, f_kv, f_q, I32(1))
        builders = {
            "seq_scan": lambda: (_seq_scan_host, f_c(), f_c(), f_i(), f_i(), f_i(), f_i(), I32(1), I32(1), I32(0), I32(0), I32(1), I32(1), stream),
            "seq_fill": lambda: (_seq_fill_host, f_c(), f_i(), f_i(), f_i(), f_i(), f_i(), I32(1), stream),
            "prep": lambda: (
                _prep_host,
                f_out,
                f_q,
                f_delta,
                f_h,
                f_h,
                f_i(),
                f_i(),
                f_dq_acc,
                f_dense_dkv,
                f_dense_dkv,
                f_i(),
                I32(1),
                hkv,
                stream,
            ),
            "bitmap": lambda: (_bitmap_host, f_ids(), f_c(), f_c(), f_c(), f_c(), f_i(), f_i(), f_i(), I32(1), stream),
            "window": lambda: (_window_host, f_i(), f_i(), f_i(), I32(0), I32(1), stream),
            "part": lambda: (_part_host, f_ids(), f_c(), f_c(), f_dq_acc, f_i(), f_i(), f_i(), f_i(), f_i(), I32(1), I32(1), I32(1), I32(1), hkv, stream),
            "scan": lambda: (_scan_host, f_i(), f_i(), f_i(), f_i(), f_i(), f_i(), I32(1), I32(1), stream),
            "fill": lambda: (_fill_host, f_i(), f_i(), f_i(), f_i(), f_i(), f_i(), f_i(), I32(1), I32(1), I32(1), stream),
            "query": lambda: (
                _sparse_query_host,
                f_q,
                f_kv,
                f_kv,
                f_q,
                f_lse,
                f_delta,
                f_dq_acc,
                f_i(),
                f_i(),
                f_i(),
                f_i(),
                f_i(),
                f_c(),
                f_c(),
                f_i(),
                I32(1),
                I32(1),
                F32(1),
                hkv,
                stream,
            ),
            "dense": lambda: (_dpair_host, *dense_fakes, 0, hkv, stream),
            "dense_pdl": lambda: (_dpair_host, *dense_fakes, 1, hkv, stream),
            "fin": lambda: (_fin_host, *fin_fakes, 0, hkv, stream),
            "fin_sparse": lambda: (_fin_host, *fin_fakes, 1, hkv, stream),
        }
        for skip in (0, 1):
            builders[f"key_skip{skip}"] = lambda skip=skip: (
                _sparse_key_host,
                f_q,
                f_q,
                f_i(),
                f_i(),
                f_i(),
                f_i(),
                f_h,
                f_h,
                f_i(),
                f_i(),
                I32(1),
                I32(1),
                skip,
                hkv,
                stream,
            )
        for name in missing:
            cache[name] = cute.compile(*builders[name](), options=f"--enable-tvm-ffi --gpu-arch=sm_{plan.arch}a")
    return {name: cache[name] for name in names}


def launch(plan, fns, q, k, v, out, dout, lse, ids, counts, first, lo, hi, dq, dk, dv, scratch, scale, stream, cu_seqlens=None, tok_base=None):
    """Enqueue the launch sequence on ``stream``. Inputs, outputs and scratch are caller-owned.

    ``ids`` [T, capacity] holds every query's block keys, ``counts`` [T] their number, ``first`` [NB] the first token
    of a block key, ``lo`` / ``hi`` [T] the inclusive key bounds. ``cu_seqlens`` (SEQ_CU) or ``tok_base`` (SEQ_ORIGIN)
    describe the packed sequences.
    """
    s = scratch
    T = plan.seq_len
    HKV = plan.hkv
    scale = F32(scale)
    dense_done_count = s["dense_block_bitmap"][plan.nsel :]
    cu = cu_seqlens if plan.seq_mode == SEQ_CU else s["seq_lens"]
    origins = tok_base if plan.seq_mode == SEQ_ORIGIN else s["nseq"]
    num_seqs = plan.seq_count if plan.seq_mode == SEQ_CU else 1
    fns["seq_scan"](
        cu,
        origins,
        s["nseq"],
        s["dense_unit"],
        s["seq_sparse_base"],
        s["sparse_count"],
        T,
        num_seqs,
        plan.seq_mode,
        plan.dense_unit if plan.has_sparse else 0,
        HKV,
        plan.num_sms,
        stream,
    )
    fns["seq_fill"](cu, s["nseq"], s["seq_sparse_base"], s["seq_start"], s["dense_hi"], s["sparse_queries"], T, stream)
    fns["prep"](
        out,
        dout,
        s["delta"],
        s["sparse_dk_acc"],
        s["sparse_dv_acc"],
        s["dense_block_bitmap"],
        s["key_pair_counts"],
        s["dq_acc"],
        s["dense_dk_acc"],
        s["dense_dv_acc"],
        s["dense_hi"],
        plan.prep_blocks,
        stream,
    )
    fns["bitmap"](ids, counts, first, lo, hi, s["seq_start"], s["dense_hi"], s["dense_block_bitmap"], plan.nw, stream)
    dense_args = (
        k,
        v,
        q,
        dout,
        lse,
        s["delta"],
        s["dense_block_bitmap"],
        s["dq_acc"],
        s["dense_dk_acc"],
        s["dense_dv_acc"],
        dense_done_count,
        s["seq_start"],
        s["dense_hi"],
        s["dense_unit"],
        plan.nw,
        plan.dense_tiles,
        plan.dense_unit,
        plan.dense_parts,
        0 if plan.has_sparse else 1,
        scale,
    )
    fin_args = (
        s["sparse_dk_acc"],
        s["sparse_dv_acc"],
        s["dense_dk_acc"],
        s["dense_dv_acc"],
        s["dq_acc"],
        dk,
        dv,
        dq,
        plan.fin_blocks,
        stream,
    )
    if plan.has_sparse:
        window_count = s["window_count"]
        KR = plan.keys_per_range
        for qlo, qhi in plan.query_windows:
            queries = s["sparse_queries"][qlo : (qhi + 3) // 4 * 4]
            fns["window"](s["sparse_count"], window_count, s["key_pair_counts"], qlo, qhi - qlo, stream)
            fns["part"](
                ids,
                counts,
                first,
                s["dq_acc"],
                queries,
                window_count,
                s["sparse_blocks"],
                s["sparse_pass_offsets"],
                s["key_pair_counts"],
                qhi - qlo,
                plan.pass_tokens,
                plan.num_passes,
                KR,
                stream,
            )
            fns["scan"](
                window_count,
                s["key_pair_counts"],
                s["key_slot_offsets"],
                s["key_slot_cursors"],
                s["active_keys"],
                s["active_count"],
                KR,
                plan.num_ranges,
                stream,
            )
            fns["fill"](
                queries,
                window_count,
                s["sparse_blocks"],
                s["sparse_pass_offsets"],
                s["key_slot_cursors"],
                s["sparse_block_slots"],
                s["slot_queries"],
                qhi - qlo,
                plan.num_passes,
                KR,
                stream,
            )
            fns["query"](
                q,
                k,
                v,
                dout,
                lse,
                s["delta"],
                s["dq_acc"],
                s["sparse_blocks"],
                s["sparse_pass_offsets"],
                s["sparse_block_slots"],
                s["pair_records"],
                queries,
                lo,
                hi,
                window_count,
                plan.sparse_ctas,
                plan.num_passes,
                scale,
                stream,
            )
            # Consume the window before its records and inverted index are reused.
            fns[plan.key_name](
                q,
                dout,
                s["pair_records"],
                s["key_slot_offsets"],
                s["key_pair_counts"],
                s["slot_queries"],
                s["sparse_dk_acc"],
                s["sparse_dv_acc"],
                s["active_keys"],
                s["active_count"],
                plan.key_ctas,
                KR,
                stream,
            )
        # The dense grid backfills only the final key-major grid; compute it once.
        fns["dense_pdl"](*dense_args, stream)
        fns["fin_sparse"](*fin_args)
    else:
        fns["dense"](*dense_args, stream)
        fns["fin"](*fin_args)


def dsl_cuda_major():
    # the toolkit flavor of the DSL compiler, which torch.version.cuda and CUDA_HOME do not identify
    from cutlass.base_dsl.version_info import CUDA_VERSION

    return CUDA_VERSION.major


def backward(grad_out, q, k, v, out_fp32, lse, sel_blocks, lo, hi, tok_base, block_first, block_last, scale, arch):
    """``cudnn.ops.qsa_backward`` on SM100 / SM103: the compacted selection through the launch sequence above."""
    from cudnn.api_base import TupleDict, WorkspaceCarver
    from .. import _metadata

    total, hkv = q.shape[0], k.shape[1]
    if hkv == 1 and dsl_cuda_major() < 13:
        # the 12Q/1KV key-major kernel crashes the CUDA 12 DSL compiler
        raise NotImplementedError(
            "12Q/1KV QSA backward requires the CUDA 13 build of CuTe DSL: install 'nvidia-cutlass-dsl[cu13]>=4.7.0' with a CUDA 13 compatible driver"
        )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    ids, counts = _metadata.compact(sel_blocks, lo, hi, block_first, block_last, arch, stream)
    # the sequences come from tok_base on the device; there are at most as many as blocks
    plan = plan_for(total, q.device, hkv, SEQ_ORIGIN, num_seqs=min(total, block_first.shape[0]))
    fns = compile_kernels(plan)
    workspace = torch.empty(plan.workspace_bytes(), dtype=torch.uint8, device=q.device)
    scratch = plan.carve(WorkspaceCarver(workspace, plan.workspace_bytes(), "qsa_backward"))
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    launch(plan, fns, q, k, v, out_fp32, grad_out, lse, ids, counts, block_first, lo, hi, dq, dk, dv, scratch, float(scale), stream, tok_base=tok_base)
    return TupleDict(dq=dq, dk=dk, dv=dv)
