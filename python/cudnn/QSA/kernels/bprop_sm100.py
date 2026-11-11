# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen Sparse Attention (QSA) backward written with the CuTe DSL primitives.

The launch sequence is a dense key-tile kernel for the queries below the crossover, and for the remaining queries a
query-major kernel (S, dP, dQ and the P / dS records of every selected block) followed by a key-major kernel that
accumulates dK / dV from those records. The selection arrives compacted per query (``ids`` block keys, ``counts``,
``first`` block key -> first token) with the inclusive key bounds ``lo`` / ``hi`` of ``cudnn.ops.qsa_backward``. A
query sees every selected key within its bounds, keys after itself included. The dense kernel masks per token through
a bitmap, so a block may start anywhere and the bounds may cut it. A query past the dense prefix holds at most
SPARSE_CAP blocks: a longer list contributes nothing and its dQ is NaN. A nonfinite K / V value of a masked token
adds exact zeros (P and dS are selected, the dQ MMAs read a sanitized copy of K), one of a selected token poisons its
queries.

``cudnn.QSA._backward`` plans the launch sequence, compiles these kernels and launches them.
"""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cuda.bindings import driver as cuda_driver
from cutlass.experimental import primitives as prims

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

I32 = cutlass.Int32
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

# dS^T reaches the dQ MMAs in bf16 parts of its fp32 value: three keep the cancellation of nearly equal dS rows (the
# default), one rounds dS to bf16 (ds_precision="bf16", the faster option).
DQ_PARTS_DEFAULT = 3

# Geometry: GQ head rows per query in the MMA tiles, HKV in (1, 2) KV heads. The kernels take HKV and GH, the real
# query heads per KV head (12, or 6 for the four-way tensor-parallel shard), as compile-time parameters: with GH < GQ
# the tile rows past GH read zero Q / dO, lse = +inf and delta = 0, so their P and dS are zero and dQ skips them.
GQ = 12
D = 256
SPARSE_CAP = 513  # blocks per query past the dense prefix: top-512 complete blocks and the causal tail block
LOG2E = 1.4426950408889634

NTHREADS = 512
# Blocks per chunk fill 8 * CHB of the 128 MMA rows. With the K and V tiles copied on to TMEM, 12 blocks leave room
# for a 3-stage K ring with one dS^T part (2 stages with three). Rows past 8 * CHB alias the next region and are
# masked out. With one KV head every K^T atom of the dQ^T MMA holds 4 alien rows (the next 512 B, cancelled by zero
# dS^T rows), and 0 * NaN would poison every column: the softmax warps zero the rows of the chunk the MMA must not
# see, and the 512 B after a stage, which its last K-step reads the same way, never hold K or V (see the layout below).
CHB = 12

# shared memory of the query-major kernel (bytes from a 1024-aligned base): [K0][pad][K1][dO][K2][Q][V][dS] with one
# dS^T part, [K0][pad][K1][dO][Q][V][dS x 3] with three (the third K stage, which lets K(c+1) stream in while S(c),
# dP(c) and dQ(c-1) run, pays for the parts).
# A chunk slot is 8 MMA rows in 4 KB: one block of one query with two KV heads (row = 2 * token + head, one SW128 box)
# or the same block number of two consecutive queries with one KV head (row = 4 * query + token, two 2 KB boxes).
BLK_SMEM = 4096
SLAB = 1024
HALF_BLK = 2048
K_STAGE = CHB * BLK_SMEM  # blocks sit at the 8-row-group stride of the MMA operand
PAIR_HALF = CHB * HALF_BLK + 512  # one query's boxes of a K / V stage plus 512 B: the two queries' rows take complementary SW128 phases
# A zero pad, dO and Q separate the K stages, so the alien rows of a stage's last K-step never hold the next chunk's
# K (a nonfinite block there would reach the previous chunk's pair). Every region stays 1024-aligned: a stage that
# starts 512 B into its swizzle span is slower, and the budget holds one aligned pad, not three.
QSLAB = 3072
Q_BYTES = 4 * QSLAB  # [4 slabs][24 rows x 128 B] with SW128: rows 0-11 are KV head 0 / the first query of the pair
OFF_K = 0
GAP01 = 1024  # the zero pad between stages 0 and 1
GAP12 = Q_BYTES  # dO sits between stages 1 and 2
DS_BYTES = CHB * 384  # one bf16 part of dS^T, the B operand of dQ^T: 8 key rows x 48 B per slot, rows in the key order of the K^T columns


def _bf16_parts(x, n, parts):
    """The n fp32 lanes of x as `parts` bf16 vectors that sum to x. After the round-to-nearest part 0 the residual fits
    16 bits, so the other parts are exact truncations and a pair costs one cvt."""
    words = []
    res = x
    for part in range(parts):
        if part == 0 or parts == 2:
            bp = res.to(BF16)
            words.append(bp.bitcast(I32))
            if part + 1 < parts:
                res = res - bp.to(F32)
        else:
            hw = res.bitcast(I32)
            hs = []
            for i in range(n):
                hs.append(hw[i] & I32(-65536))
            pk = []
            for i in range(n // 2):
                pk.append(prims.prmt(hs[2 * i], 0x7632, prims.PermuteMode.DEFAULT, hi=hs[2 * i + 1]))
            words.append(cutlass.Vector.from_elements(pk, I32))
            if part + 1 < parts:
                res = res - cutlass.Vector.from_elements(hs, I32).bitcast(F32)
    return words


def _k_stage(pair):
    return 2 * PAIR_HALF if pair else K_STAGE


def _query_layout(parts, pair):
    # K stage count and size, and the dO, Q, V and dS offsets of the query-major kernel
    nks = 3 if parts == 1 else 2
    ks = _k_stage(pair)
    off_do = OFF_K + 2 * ks + GAP01
    off_q = off_do + GAP12 + (nks - 2) * ks
    off_v = off_q + Q_BYTES
    return nks, ks, off_do, off_q, off_v, off_v + ks


# tensor memory columns of the query-major kernel
TM_S = 0  # S^T, double buffered at a 48-column pitch
TM_DP = 24  # dP^T, interleaved with S^T
TM_DQ = 96  # dQ^T halves at 96 and 128
TM_KA = 160  # the chunk's K tile as the TMEM A operand of S^T (128 columns of packed bf16 pairs)
TM_VA = 288  # the chunk's V tile as the TMEM A operand of dP^T
TM_KT = 416  # the chunk's K tile transposed (lane = dim of a half, column = packed key pair), the A operand of dQ^T
TM_COLS = 512
L2_EVICT_FIRST = 0x12F0000000000000  # createpolicy encoding of the L2 evict-first priority


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


@cute.jit
def _st_async8(dst, v2, mbar):
    prims.inline_ptx(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [{$r0}], {{$r1}, {$r2}}, [{$r3}];",
        read_only_args=[dst, v2[0], v2[1], mbar],
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
def _koff(st, ks):
    # byte offset of K stage st (of ks bytes) from OFF_K: the regions between the stages differ in size
    return st * ks + ((st + 1) >> 1) * GAP01 + (st >> 1) * GAP12


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
def _lse_rows(lse: cute.Tensor, qi, h0, GH: cutlass.Constexpr[int]):
    # the GQ head rows of one query from the native [Hq, T] LSE, from head h0. Rows past the GH real heads read +inf.
    values = []
    for i in cutlass.range_constexpr(GQ):
        if cutlass.const_expr(i < GH):
            values.append(lse[h0 + i, qi])
        else:
            values.append(F32(float("inf")))
    return cutlass.Vector.from_elements(values, F32)


@cute.jit
def _delta_rows(p_delta, base, GH: cutlass.Constexpr[int]):
    # the GQ head rows of one query's delta from flat index base (16-byte aligned when every row is a real head). Rows
    # past the GH real heads read 0.
    values = []
    if cutlass.const_expr(GH == GQ):
        for c in cutlass.range_constexpr(GQ // 4):
            v4 = (p_delta + base + c * 4).load(count=4, alignment=16)
            for i in cutlass.range_constexpr(4):
                values.append(v4[i])
    else:
        for i in cutlass.range_constexpr(GQ):
            if cutlass.const_expr(i < GH):
                values.append((p_delta + base + i).load())
            else:
                values.append(F32(0.0))
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
    GH: cutlass.Constexpr[int],
    DQ_PARTS: cutlass.Constexpr[int],
):
    # An item is one sparse query (two KV heads) or two consecutive ones of the compacted list (one KV head), see the
    # SMEM layout note.
    NKS, KS, OFF_DO, OFF_Q, OFF_V, OFF_DS = _query_layout(DQ_PARTS, HKV == 1)
    # slot stride and second-query offset inside a K / V stage
    SLOT_B = cutlass.const_expr(HALF_BLK if HKV == 1 else BLK_SMEM)
    HALF_B = cutlass.const_expr(PAIR_HALF if HKV == 1 else HALF_BLK)
    HQ = GH * HKV
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

    smem = cutlass.Array(I32, (OFF_DS + DQ_PARTS * DS_BYTES) // 4, space=cutlass.AddressSpace.smem, alignment=1024)
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
    b_kzf = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)  # softmax -> copy warps: K rows final, nonfinite ones zeroed
    b_ktrf = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)  # copy warps -> dQ warp: the chunk's K^T is in TMEM
    tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    # zero the whole operand region once, so padding rows and stale rows stay finite
    zero4 = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
    sbase = smem.data_ptr()
    for it in cutlass.range(tidx, (OFF_DS + DQ_PARTS * DS_BYTES) // 16, NTHREADS, unroll=1):
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
                prims.mbarrier_init(b_ke.subview(s), 4)
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_pf.subview(s), 4)
            prims.mbarrier_init(b_sf.subview(0), 1)
            prims.mbarrier_init(b_sf.subview(1), 1)
            prims.mbarrier_init(b_vf, 1)
            prims.mbarrier_init(b_ve, 4)
            prims.mbarrier_init(b_dqf, 1)
            prims.mbarrier_init(b_dqe, 4)
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_sde.subview(s), 4)
            prims.mbarrier_init(b_dsf, 1)
            for s in cutlass.range_constexpr(4):
                prims.mbarrier_init(b_ktf.subview(s), 4)
                prims.mbarrier_init(b_vtf.subview(s), 4)
            prims.mbarrier_init(b_kte, 1)
            prims.mbarrier_init(b_vte, 1)
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_kzf.subview(s), 4)
                prims.mbarrier_init(b_ktrf.subview(s), 4)
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
                            prims.mbarrier_arrive_expect_tx(b_qf, nqq * (2 * 4 * GH * HKV * 128))
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
                        # valid one
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
                                _load_block(tma_k, sbase + (OFF_K + _koff(st, KS) + jl * SLOT_B + hl * HALF_B) // 4, xcur, b_kf.subview(st), PAIR)
                            if c + 1 < nchunks:
                                okn = ((c + 1) * CHB + jl < nblk_l) & lane_ok
                                if okn:
                                    _prefetch_block(tma_k, xnext, PAIR)
                        gc = gc + nchunks
                        nq = nq + 1

        # V has only one SMEM stage.  Giving it to the otherwise-idle warp 1 prevents its empty-barrier wait from
        # serialising warp 0's K producer, so K can run up to the actual ring depth ahead of the MMA.
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
                                _load_block(tma_v, sbase + (OFF_V + jl * SLOT_B + hl * HALF_B) // 4, xcur, b_vf, PAIR)
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
                            # S(c) and dP(c) only need softmax(c-2), the last reader of this S/dP buffer, to have read it
                            # (signalled early through sde), so softmax(c) overlaps dQ(c-1) and S/dP(c+1) on the tensor pipe.
                            cgc = gc + c
                            st = cgc % NKS
                            if c < nchunks:
                                _wait(b_kf.subview(st), (cgc // NKS) & 1)
                                # softmax(c-2) has loaded this S/dP buffer into registers
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
                                                    prims.make_tmem_ptr(tmem_raw + (TM_S + (cgc & 1) * 48), F32),
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
                                                    prims.make_tmem_ptr(tmem_raw + (TM_DP + (cgc & 1) * 48), F32),
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
                a_major=0,
                b_major=1,
            )
            # A = K^T(c) from TMEM (transposed there by the copy warps), B = the dS^T parts in SMEM. A K-step covers the
            # 16 keys of two slots.
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
                            nv3 = _chunk_valid(nblk, c)
                            # softmax(c) has written dS^T to SMEM and warps 8-15 K^T(c) to TMEM
                            _wait(b_pf.subview(pb), (cgb >> 1) & 1)
                            _wait(b_ktrf.subview(pb), (cgb >> 1) & 1)
                            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            if c == 0:
                                # the previous item's dQ has been drained from TMEM
                                _wait(b_dqe, (nq & 1) ^ 1)
                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            acc_q = c > 0  # the first chunk of an item initialises the dQ accumulator
                            # K-steps over the chunk's valid slots only, two slots per step
                            nkk = (nv3 + 1) >> 1
                            # one pass per bf16 part of dS^T
                            for hh in cutlass.range_constexpr(2):
                                for part in cutlass.range_constexpr(DQ_PARTS):
                                    for kk in cutlass.range(nkk, unroll=1):
                                        if prims.elect_sync():
                                            prims.tcgen05_mma(
                                                prims.Tcgen05MMAKind.F16,
                                                prims.CTAGroup.CTA_1,
                                                prims.make_tmem_ptr(tmem_raw + (TM_DQ + hh * 32), F32),
                                                prims.make_tmem_ptr(tmem_raw + (TM_KT + hh * 48 + kk * 8), BF16),
                                                d_ds_mn3.advance_start_address(part * DS_BYTES + kk * 768),
                                                idesc_q3,
                                                acc_q | (part > 0) | (kk > 0),
                                            )
                            if prims.elect_sync():
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
        # dS^T row of this key inside its K-step, in the key order of the TMEM K^T columns (see the copy warps)
        if cutlass.const_expr(PAIR):
            ds_off = (OFF_DS + (jrow >> 1) * 768 + hrow * 384 + ((trow >> 1) * 4 + (jrow & 1) * 2 + (trow & 1)) * 16) // 4
        else:
            ds_off = (OFF_DS + (jrow >> 1) * 768 + (trow >> 1) * 384 + ((trow & 1) * 4 + (jrow & 1) * 2 + hrow) * 16) // 4
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
                    h0 = I32(0)
                    if cutlass.const_expr(not PAIR):
                        h0 = hrow * GH
                    lbase = qr * HQ + h0
                    # the inclusive key bounds of the query: a block's tokens outside them have zero P / dS
                    lo_q = (p_lo + qr).load()
                    hi_q = (p_hi + qr).load()
                    nchunks = (nblk + (CHB - 1)) // CHB
                    # record slot and first token of this row's block entry, loaded one chunk ahead
                    prow_pp = p_sparse_block_slots + (jr * PART_W + o0)
                    trow_pp = p_sparse_blocks + (jr * PART_W + o0)
                    mpf = jrow
                    if mpf >= nblk_r:
                        mpf = I32(0)
                    pnx = (prow_pp + mpf).load()
                    tnx = (trow_pp + mpf).load()
                    # The 12-head softmax state stays at vector level across the chunks, built once per query. The rows
                    # past the GH real heads carry lse = +inf and delta = 0, so their P and dS are zero.
                    lse_v = _lse_rows(g_lse, qr, h0, GH)
                    del_v = _delta_rows(p_delta, lbase, GH)
                    zero12 = cutlass.Vector.from_elements([F32(0.0)] * GQ, F32)
                    for c in cutlass.range(nchunks, unroll=1):
                        cgc = gc + c
                        st = cgc & 1
                        _wait(b_sf.subview(cgc & 1), (cgc >> 1) & 1)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        t_s = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_S + (cgc & 1) * 48).as_ptr(F32)
                        t_dp = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_DP + (cgc & 1) * 48).as_ptr(F32)
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
                            # Zero a nonfinite K row before the copy warps transpose it into the dQ^T operand: the paired query then sees
                            # 0 x finite, while S^T already read the TMEM copy and keeps the selecting query poisoned.
                            if bad:
                                st_k = cgc % NKS
                                if cutlass.const_expr(PAIR):
                                    krow = sbase + (OFF_K + _koff(st_k, KS) + jrow * HALF_BLK + hrow * PAIR_HALF + trow * 128) // 4
                                    for s4 in cutlass.range_constexpr(4):
                                        for u in cutlass.range_constexpr(8):
                                            (krow + (s4 * 512) // 4 + u * 4).store(zero4, alignment=16)
                                else:
                                    krow = sbase + (OFF_K + _koff(st_k, KS) + jrow * BLK_SMEM + (2 * trow + hrow) * 128) // 4
                                    for s4 in cutlass.range_constexpr(4):
                                        for u in cutlass.range_constexpr(8):
                                            (krow + (s4 * SLAB) // 4 + u * 4).store(zero4, alignment=16)
                        # the K rows of the chunk are final: warps 8-15 may transpose them into TMEM
                        prims.bar_warp_sync(0xFFFFFFFF)
                        if prims.elect_sync():
                            prims.mbarrier_arrive(b_kzf.subview(cgc & 1))
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
                        # dS^T for the dQ MMA in DQ_PARTS bf16 parts of its fp32 value (three keep the cancellation of
                        # nearly equal rows). The record keeps the first.
                        dvecs = []
                        bpws = _bf16_parts(ds_v, GQ, DQ_PARTS)
                        for part in cutlass.range_constexpr(DQ_PARTS):
                            bpw = bpws[part]
                            dwords = []
                            for n in cutlass.range_constexpr(16):
                                dv_ = I32(0)
                                if cutlass.const_expr(n < 6):
                                    if hrow == 0:
                                        dv_ = bpw[n]
                                elif cutlass.const_expr(n < 12):
                                    if hrow == 1:
                                        dv_ = bpw[n - 6]
                                dwords.append(dv_)
                            dvecs.append(cutlass.Vector.from_elements(dwords, I32))

                        # dQ(c-1) has finished reading the dS^T SMEM buffers
                        _wait(b_dsf, (cgc & 1) ^ 1)
                        # dS^T row -> SMEM (MN-major core matrices: 3 n-chunks of 8 at 128 B). Slots past CHB have no
                        # row group.
                        if jrow < CHB:
                            for part in cutlass.range_constexpr(DQ_PARTS):
                                for nc in cutlass.range_constexpr(3):
                                    (sbase + ds_off + part * (DS_BYTES // 4) + nc * 32).store(dvecs[part][nc * 4 : nc * 4 + 4], alignment=16)
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
                                    for n in cutlass.range_constexpr(GH if PAIR else HQ):
                                        (qb + n * D).store(dqv[hh][h * GQ + n])
                                else:
                                    for n in cutlass.range_constexpr(GH if PAIR else HQ):
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
        # Warps 8-11 (group 0): K and V of every chunk as TMEM A operands, lane = chunk row, column = packed bf16 dim
        # pair, all eight 16-byte chunks of each 128-byte SW128 row segment (chunk cc of a row at 128 B index y of its
        # atom sits at cc ^ y). Warps 12-15 (group 1): K^T, lane = dim, column = packed key pair, by ldmatrix.trans.
        qd = warp % 4
        group = (warp - 8) >> 2
        r = qd * 32 + lane
        if cutlass.const_expr(PAIR):
            # row = 8 * slot + 4 * query + token at query * PAIR_HALF + slot * 2 KB + slab * 512 B + token * 128 B
            yr = r & 3
            yq = ((r >> 2) & 1) * 4
            roff = (OFF_K + (r >> 3) * HALF_BLK + ((r >> 2) & 1) * PAIR_HALF + yr * 128) // 4
            slab_w = 512 // 4
        else:
            # row = 8 * block + 2 * token + head at block + slab * 1 KB + row * 128 B
            yr = r & 7
            roff = (OFF_K + (r >> 3) * BLK_SMEM + yr * 128) // 4
            slab_w = SLAB // 4
        zw = []
        for e in cutlass.range_constexpr(32):
            zw.append(I32(0))
        zero32 = cutlass.Vector.from_elements(zw, I32)
        if group == 0:
            # zero the quadrant's K / V operand rows once, so rows a chunk does not copy keep finite data
            for zb in cutlass.range_constexpr(2):
                for sl4 in cutlass.range_constexpr(4):
                    prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_KA + zb * 128 + sl4 * 32).as_ptr(I32), zero32)
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
                                # odd slabs and the second query's rows sit 512 B into their swizzle span, which flips the phase bit
                                ys = yr
                                if cutlass.const_expr(PAIR):
                                    ys = yr + (yq ^ (4 * (sl4 % 2)))
                                if do_cp:
                                    off = roff + _koff(st, KS) // 4 + sl4 * slab_w
                                    wv = []
                                    for cc in cutlass.range_constexpr(8):
                                        v4 = (sbase + (off + ((ys ^ cc) * 4))).load(count=4, alignment=16)
                                        for e in cutlass.range_constexpr(4):
                                            wv.append(v4[e])
                                    prims.tcgen05_st(
                                        "32x32b",
                                        prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_KA + sl4 * 32).as_ptr(I32),
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
                                if cutlass.const_expr(PAIR):
                                    ys = yr + (yq ^ (4 * (sl4 % 2)))
                                if do_cp:
                                    wv = []
                                    for cc in cutlass.range_constexpr(8):
                                        v4 = (vb + (sl4 * slab_w + ((ys ^ cc) * 4))).load(count=4, alignment=16)
                                        for e in cutlass.range_constexpr(4):
                                            wv.append(v4[e])
                                    prims.tcgen05_st(
                                        "32x32b",
                                        prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_VA + sl4 * 32).as_ptr(I32),
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
        else:
            # ldmatrix.x4.trans: lane L feeds row L & 7 of tile L >> 3, and a tile is one slot's eight rows (eight SW128 bank
            # groups). The softmax writes the dS^T rows in the same key order.
            ldm_tile = lane >> 3
            ldm_row = lane & 7
            kstep_row = 8 * (ldm_tile & 1) + ldm_row
            ldm_off = []
            for h2 in cutlass.range_constexpr(2):
                dim_slab = 2 * h2 + (qd >> 1)
                for half in cutlass.range_constexpr(2):
                    dim_chunk = (qd & 1) * 4 + half * 2 + (ldm_tile >> 1)
                    if cutlass.const_expr(PAIR):
                        row_phase = (kstep_row & 3) + 4 * ((dim_slab + ((kstep_row >> 2) & 1)) & 1)
                        row_off = (
                            (kstep_row >> 3) * HALF_BLK
                            + ((kstep_row >> 2) & 1) * PAIR_HALF
                            + dim_slab * 512
                            + (kstep_row & 3) * 128
                            + ((dim_chunk ^ row_phase) << 4)
                        )
                    else:
                        row_off = (kstep_row >> 3) * BLK_SMEM + dim_slab * SLAB + (kstep_row & 7) * 128 + ((dim_chunk ^ (kstep_row & 7)) << 4)
                    ldm_off.append(row_off // 4)
            # zero the quadrant's K^T rows once: the columns of invalid slots meet zero dS^T rows and must be finite
            for z3 in cutlass.range_constexpr(3):
                prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_KT + z3 * 32).as_ptr(I32), zero32)
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
                            # K^T(c): once softmax(c) has zeroed the chunk's nonfinite K rows in SMEM and dQ(c-1) has
                            # finished reading the previous K^T. The K SMEM stage is free after this copy.
                            _wait(b_kzf.subview(cgc & 1), (cgc >> 1) & 1)
                            _wait(b_dsf, (cgc & 1) ^ 1)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            kbase = sbase + (OFF_K + _koff(st, KS)) // 4
                            for kk in cutlass.range_constexpr(CHB // 2):
                                for h2 in cutlass.range_constexpr(2):
                                    for half in cutlass.range_constexpr(2):
                                        frag = prims.ldmatrix(kbase + ((kk * 2 * SLOT_B) // 4 + ldm_off[h2 * 2 + half]), 4, prims.MMALayout.COL)
                                        prims.tcgen05_st(
                                            "16x256b",
                                            prims.TmemAddr.from_row_col(tmem_row0 + qd * 32 + half * 16, tmem_col0 + TM_KT + h2 * 48 + kk * 8).as_ptr(I32),
                                            frag,
                                        )
                            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if prims.elect_sync():
                                prims.mbarrier_arrive(b_ktrf.subview(cgc & 1))
                                prims.mbarrier_arrive(b_ke.subview(st))
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
    GH: cutlass.Constexpr[int],
    DQ_PARTS: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = q.shape[0]
    qbox = (1, GH * HKV, 64)
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
        GH,
        DQ_PARTS,
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

    def __init__(self, hkv, gh=GQ):
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
        self.tx_per_query = 2 * 4 * gh * hkv * 128 + self.rec_w * 4  # the real head rows of Q and dO, plus the record


@cute.jit
def _active_key(p_active, n0, KR, i):
    # Item i of the key-major kernel: the n0 active tokens as keys of range 0, then the same tokens as keys of range 1.
    # n0 is a multiple of the grid, so a CTA walks the same tokens in both ranges and their adds keep one order.
    off = i
    base = I32(0)
    if i >= n0:
        off = i - n0
        base = KR
    return (p_active + off).load() + base


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
    GH: cutlass.Constexpr[int],
):
    PB = _PbLayout(HKV, GH)
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
                                elif cutlass.const_expr(GH == GQ):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sb + (PB.off_q + i * 4 * PB.slab) // 4, tma_q.get_ptr(), (0, 0, 0, qids[i]), b_full.subview(slot)
                                    )
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sb + (PB.off_do + i * 4 * PB.slab) // 4, tma_do.get_ptr(), (0, 0, 0, qids[i]), b_full.subview(slot)
                                    )
                                else:
                                    # GH rows per slab into the 12-row slab slots (the rows past GH stay zero)
                                    for s in cutlass.range_constexpr(4):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sb + (PB.off_q + (i * 4 + s) * PB.slab) // 4, tma_q.get_ptr(), (0, 0, s, qids[i]), b_full.subview(slot)
                                        )
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sb + (PB.off_do + (i * 4 + s) * PB.slab) // 4, tma_do.get_ptr(), (0, 0, s, qids[i]), b_full.subview(slot)
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
            if i == n0 + bid:
                # range 1 adds to the rows range 0's reduces of this warp wrote: those must have completed
                prims.cp_async_bulk_wait_group(0)
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
    GH: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = q.shape[0]
    if cutlass.const_expr(HKV == 2):
        tma_q = cuda.create_tensor_map_tiled_from_view(q, box_dims=(1, 24, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
        tma_do = cuda.create_tensor_map_tiled_from_view(do, box_dims=(1, 24, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    else:
        # a query's 12 head rows of all 4 slabs in one box, slab-major in SMEM
        q_layout = cute.make_layout((T, GH, 4, 64), stride=(GH * D, D, 64, 1))
        if cutlass.const_expr(GH == GQ):
            qbox4 = (1, GH, 4, 64)
        else:
            qbox4 = (1, GH, 1, 64)  # one box per slab into the 12-row slab slots
        tma_q = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(q.iterator, q_layout), box_dims=qbox4, stride_order=(3, 1, 2, 0), swizzle=cuda.TensorMapSwizzle.s128b
        )
        tma_do = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(do.iterator, q_layout), box_dims=qbox4, stride_order=(3, 1, 2, 0), swizzle=cuda.TensorMapSwizzle.s128b
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
        tma_q, tma_do, tma_rk, tma_rv, pair_records, key_slot_offsets, key_pair_counts, slot_queries, active_keys, active_count, KR, SK, HKV, GH
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
    g_dense_sync: cute.Tensor,
    T: I32,
    nz: I32,
    nsel: I32,
    n_key_pair_counts: I32,
    n_dense_dkv: I32,
    n_dense_sync: I32,
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
):
    HQ = GH * HKV
    # Delta and the zero fill of every accumulator share one launch.
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    lane = tidx % 32
    wib = tidx // 32
    p_o = g_o.iterator.raw_ptr()
    p_do = g_do.iterator.raw_ptr()
    p_delta = g_delta.iterator.raw_ptr()
    nrows = T * HQ
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
    p_dense_sync = g_dense_sync.iterator.raw_ptr()
    for it in cutlass.range(bid * PREP_THREADS + tidx, n_dense_sync // 4, nb * PREP_THREADS, unroll=1):
        (p_dense_sync + it * 4).store(zi, alignment=16)


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
    dense_sync: cute.Tensor,
    nblocks: I32,
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
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
        dense_sync,
        I32(T),
        I32(nz),
        I32(dense_block_bitmap.shape[0]),
        I32(key_pair_counts.shape[0]),
        I32(dense_dk_acc.shape[0]),
        I32(dense_sync.shape[0]),
        HKV,
        GH,
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
    nparts: I32,
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
            if nparts > 0:
                # Ordered bands are one unit in every key tile, so nparts of them must cover the groups a tile reaches.
                reach = (DP_TT + _dense_queries(max_len) + (DP_G - 1)) // DP_G
                cover = (reach + (nparts - 1)) // nparts
                if cover > unit:
                    unit = cover
                # A tile's bands add their dK / dV one after another, so a band is no shorter than the bands are many.
                side = I32(1)
                while side * side < reach:
                    side = side + 1
                if side > unit:
                    unit = side
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
    nparts: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_seq_scan_kernel(cu, tok_base, nseq, dense_unit, seq_sparse_base, sparse_count, T, B, mode, unit_fixed, HKV, num_sms, nparts).launch(
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
    g_first_group: cute.Tensor,
    T: I32,
):
    # one thread per token: its sequence by binary search, the first row past its sequence's dense queries, then its
    # entry in the compacted sparse list. The thread of a key tile's first token also resets the tile's first-group
    # entry, which the bitmap kernel lowers for queries that see keys after themselves.
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
        if t % DP_TT == 0:
            (g_first_group.iterator.raw_ptr() + t // DP_TT).store(I32(2**31 - 1))


qsa_bwd_seq_fill_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _seq_fill_host(
    cu: cute.Tensor,
    nseq: cute.Tensor,
    seq_sparse_base: cute.Tensor,
    seq_start: cute.Tensor,
    dense_hi: cute.Tensor,
    sparse_queries: cute.Tensor,
    first_group: cute.Tensor,
    T: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_seq_fill_kernel(cu, nseq, seq_sparse_base, seq_start, dense_hi, sparse_queries, first_group, T).launch(
        grid=((T + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )


@cute.kernel
def qsa_bwd_window_kernel(
    total_count: cute.Tensor, window_count: cute.Tensor, key_counts: cute.Tensor, key_query_bits: cute.Tensor, offset: I32, capacity: I32
):
    # The window's query count is rebuilt on the device for every call and replay, and the key counts of the inverted
    # index start from zero. So do the query bits, which only the keys the previous window counted hold.
    tid = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
    if tid < cute.size(key_counts):
        if (offset == 0) | (key_counts[tid] != 0):
            row = key_query_bits.iterator.raw_ptr() + tid * QB
            zero4 = cutlass.Vector.from_elements([I32(0)] * 4, I32)
            for w in cutlass.range_constexpr(QB // 4):
                (row + w * 4).store(zero4, alignment=16)
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
def _window_host(
    total_count: cute.Tensor,
    window_count: cute.Tensor,
    key_counts: cute.Tensor,
    key_query_bits: cute.Tensor,
    offset: I32,
    capacity: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_window_kernel(total_count, window_count, key_counts, key_query_bits, offset, capacity).launch(
        grid=(cute.ceil_div(cute.size(key_counts), 256), 1, 1), block=(256, 1, 1), stream=stream
    )


# ----------------------------------------------------------------------------- KV-pass partition and inverted index
# The query-major kernel walks (KV pass, query) items pass-major so that one pass's K/V blocks stay L2 resident.
# sparse_blocks[j] holds the first token of every selected block of sparse query j (and of its tail block) grouped by
# pass p = block // RB, and sparse_pass_offsets[j, p] .. sparse_pass_offsets[j, p+1] bounds pass p's segment.
PART_W = 516  # list row stride in int32: 512 blocks plus the tail block, rounded up to a multiple of 4 words
POS = 64  # offset row stride in int32, a power of two above the KV passes of the longest sequence, so NP < POS

# Sparse queries are split into ranges of QR queries. Every (query, selected block) pair gets a slot in a key-major
# order: key = range * T + the block's first token, so every block of the packed buffer has its own key, and the slots
# of a key are contiguous (key_slot_offsets[key] .. + key_pair_counts[key]) in the order of its queries. The query-major
# kernel writes the pair's P / dS record (bf16, [4 tokens][12 * HKV heads] each, see _rec_w) into its slot, and the
# key-major kernel walks the tokens holding a record in either range, compacted into active_keys by the index scan.
SPARSE_WINDOW_QUERIES = 4096  # Reuse at most two ranges of P/dS records, including for packed inputs.
QR = 2048  # queries per range: a range's Q / dO rows stay L2 resident while the key-major kernel walks its keys
QR_SH = 11  # log2(QR)
QB = QR // 32  # int32 words of a key's query bits
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
    g_key_query_bits: cute.Tensor,
    cap: I32,
    RB: I32,
    NP: I32,
    KR: I32,
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
):
    # per-query block lists split by KV pass, plus the per-key pair counts and query bits of the inverted index, whose
    # key is the block's first token within the query's range. An empty row only needs its dQ zeroed. A row past
    # SPARSE_CAP blocks exceeds the slot budget of a range: it contributes nothing and its dQ is NaN.
    HQ = GH * HKV
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    j = bid * 8 + tidx // 32
    p_ids = g_ids.iterator.raw_ptr()
    p_first = g_first.iterator.raw_ptr()
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()
    p_key_pair_counts = g_key_pair_counts.iterator.raw_ptr()
    p_key_query_bits = g_key_query_bits.iterator.raw_ptr()
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
        jr = j & (QR - 1)
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
                    bit = I32(1) << (jr & 31)
                    seen = prims.atomicrmw(
                        "or", p_key_query_bits + ((kb + tok) * QB + (jr >> 5)), bit, mem_order=prims.MemOrder.RELAXED, syncscope=prims.MemScope.GPU
                    )
                    # a row repeating a first token (no valid block layout does) counts it once, so every slot of the key is written
                    if (seen & bit) == 0:
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
    key_query_bits: cute.Tensor,
    nsq_cap: I32,
    RB: I32,
    NP: I32,
    KR: I32,
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
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
        key_query_bits,
        I32(cap),
        RB,
        NP,
        KR,
        HKV,
        GH,
    ).launch(grid=((nsq_cap + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)


@cute.kernel
def qsa_bwd_index_scan_kernel(
    g_window_count: cute.Tensor,
    g_key_pair_counts: cute.Tensor,
    g_key_slot_offsets: cute.Tensor,
    g_active_keys: cute.Tensor,
    g_active_count: cute.Tensor,
    KR: I32,
    NR: I32,
    nctas: I32,
):
    # one CTA per range over tiles of 8 * SCAN_T keys, a thread holding 8 consecutive keys (two 16-byte vectors, so
    # a warp's loads coalesce): slot offsets = the range's slot base + exclusive scan of its key counts. Range 0's CTA
    # also compacts the tokens holding a record in either range into the key-major kernel's item list, padded with
    # the never-selected token KR - 1 to a multiple of that kernel's grid (active_count[0], and [1] with two ranges).
    tidx, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
    sm = cutlass.Array(I32, 64, space=cutlass.AddressSpace.smem, alignment=16)
    sp = sm.data_ptr()
    p_key_pair_counts = g_key_pair_counts.iterator.raw_ptr()
    p_key_slot_offsets = g_key_slot_offsets.iterator.raw_ptr()
    p_active = g_active_keys.iterator.raw_ptr()
    zero4 = cutlass.Vector.from_elements([I32(0)] * 4, I32)
    kb = r * KR
    ko = (NR - 1 - r) * KR  # the other range, or this one again when it is alone
    pos = I32(0)
    if g_window_count.iterator.raw_ptr().load() > 0:
        acc = r * RANGE_PAIRS
        for t0 in cutlass.range(0, KR, 8 * SCAN_T, unroll=1):
            e = t0 + 8 * tidx
            inr = e < KR
            ca = zero4
            cb = zero4
            ua = zero4
            ub = zero4
            if inr:
                ca = (p_key_pair_counts + (kb + e)).load(count=4, alignment=16)
                cb = (p_key_pair_counts + (kb + e + 4)).load(count=4, alignment=16)
                ua = (p_key_pair_counts + (ko + e)).load(count=4, alignment=16)
                ub = (p_key_pair_counts + (ko + e + 4)).load(count=4, alignment=16)
            s = ca[0] + ca[1] + ca[2] + ca[3] + cb[0] + cb[1] + cb[2] + cb[3]
            live = I32(0)
            for u in cutlass.range_constexpr(4):
                if (ca[u] | ua[u]) > 0:
                    live = live + 1
                if (cb[u] | ub[u]) > 0:
                    live = live + 1
            off, tot_s, lpos, tot_l = _cta_scan2(sp, tidx, s, live)
            if inr:
                a = acc + off
                p = pos + lpos
                offs = []
                for u in cutlass.range_constexpr(4):
                    offs.append(a)
                    a = a + ca[u]
                    if (ca[u] | ua[u]) > 0:
                        if r == 0:
                            (p_active + p).store(e + u)
                        p = p + 1
                (p_key_slot_offsets + (kb + e)).store(cutlass.Vector.from_elements(offs, I32), alignment=16)
                offs = []
                for u in cutlass.range_constexpr(4):
                    offs.append(a)
                    a = a + cb[u]
                    if (cb[u] | ub[u]) > 0:
                        if r == 0:
                            (p_active + p).store(e + 4 + u)
                        p = p + 1
                (p_key_slot_offsets + (kb + e + 4)).store(cutlass.Vector.from_elements(offs, I32), alignment=16)
            acc = acc + tot_s
            pos = pos + tot_l
    if r == 0:
        padded = (pos + nctas - 1) // nctas * nctas
        if tidx < padded - pos:
            (p_active + (pos + tidx)).store(KR - 1)
        if tidx == 0:
            p_count = g_active_count.iterator.raw_ptr()
            p_count.store(padded)
            second = I32(0)
            if NR == 2:
                second = padded
            (p_count + 1).store(second)


qsa_bwd_index_scan_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _scan_host(
    window_count: cute.Tensor,
    key_pair_counts: cute.Tensor,
    key_slot_offsets: cute.Tensor,
    active_keys: cute.Tensor,
    active_count: cute.Tensor,
    KR: I32,
    NR: I32,
    nctas: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_index_scan_kernel(window_count, key_pair_counts, key_slot_offsets, active_keys, active_count, KR, NR, nctas).launch(
        grid=(NR, 1, 1), block=(SCAN_T, 1, 1), stream=stream
    )


@cute.kernel
def qsa_bwd_index_fill_kernel(
    g_sparse_queries: cute.Tensor,
    g_sparse_count: cute.Tensor,
    g_sparse_blocks: cute.Tensor,
    g_sparse_pass_offsets: cute.Tensor,
    g_key_slot_offsets: cute.Tensor,
    g_key_query_bits: cute.Tensor,
    g_sparse_block_slots: cute.Tensor,
    g_slot_queries: cute.Tensor,
    NP: I32,
    KR: I32,
):
    # one warp per sparse query gives every pair its slot: sparse_block_slots (aligned with sparse_blocks) -> slot, slot_queries[slot] = query.
    # The slot is the key's first plus its set query bits below the query's, so a key's slots follow its queries on every run.
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    j = bid * 8 + tidx // 32
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()
    p_key_slot_offsets = g_key_slot_offsets.iterator.raw_ptr()
    p_key_query_bits = g_key_query_bits.iterator.raw_ptr()
    p_sparse_block_slots = g_sparse_block_slots.iterator.raw_ptr()
    p_slot_queries = g_slot_queries.iterator.raw_ptr()
    if j < g_sparse_count.iterator.raw_ptr().load():
        qi = (g_sparse_queries.iterator.raw_ptr() + j).load()
        kb = (j >> QR_SH) * KR
        jw = (j & (QR - 1)) >> 5
        below = (I32(1) << (j & 31)) - I32(1)
        tot = (p_sparse_pass_offsets + (j * POS + NP)).load()
        for L in cutlass.range(lane, tot, 32, unroll=1):
            key = kb + (p_sparse_blocks + (j * PART_W + L)).load()
            row = p_key_query_bits + key * QB
            pos = (p_key_slot_offsets + key).load()
            for w4 in cutlass.range(0, jw // 4 + 1, unroll=1):
                bits = (row + w4 * 4).load(count=4, alignment=16)
                for e in cutlass.range_constexpr(4):
                    word = w4 * 4 + e
                    mask = I32(-1)
                    if word == jw:
                        mask = below
                    if word > jw:
                        mask = I32(0)
                    pos = pos + cute.arch.popc(bits[e] & mask)
            (p_sparse_block_slots + (j * PART_W + L)).store(pos)
            (p_slot_queries + pos).store(qi)


qsa_bwd_index_fill_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _fill_host(
    sparse_queries: cute.Tensor,
    sparse_count: cute.Tensor,
    sparse_blocks: cute.Tensor,
    sparse_pass_offsets: cute.Tensor,
    key_slot_offsets: cute.Tensor,
    key_query_bits: cute.Tensor,
    sparse_block_slots: cute.Tensor,
    slot_queries: cute.Tensor,
    nsq_cap: I32,
    NP: I32,
    KR: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_index_fill_kernel(
        sparse_queries, sparse_count, sparse_blocks, sparse_pass_offsets, key_slot_offsets, key_query_bits, sparse_block_slots, slot_queries, NP, KR
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
    g_first_group: cute.Tensor,
    g_dq_acc: cute.Tensor,
    T: I32,
    cap: I32,
    NW: I32,
    HQD: I32,
):
    # Row qi of the bitmap holds one bit per sequence-relative key token a dense query attends: the tokens of its
    # selected blocks within [lo, hi] and the dense prefix, so the dense kernel tests the bit alone and a block may
    # start anywhere. A query that sees keys after itself lowers the first query group of the key tiles up to hi, so
    # the dense kernel walks those groups. Keys past the dense prefix have no bits, and selecting one leaves the
    # query's dQ NaN (the sparse path alone reaches them).
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    qi = bid * 8 + tidx // 32
    p_ids = g_ids.iterator.raw_ptr()
    p_first = g_first.iterator.raw_ptr()
    p_dense_block_bitmap = g_dense_block_bitmap.iterator.raw_ptr()
    if qi < T:
        dense_end = (g_dense_hi.iterator.raw_ptr() + qi).load()
        if qi < dense_end:
            s = (g_seq_start.iterator.raw_ptr() + qi).load()
            n = (g_counts.iterator.raw_ptr() + qi).load()
            floor = (g_lo.iterator.raw_ptr() + qi).load()
            if floor < s:
                floor = s
            hi_q = (g_hi.iterator.raw_ptr() + qi).load()
            ceil = hi_q
            if ceil > dense_end - 1:
                ceil = dense_end - 1
            lrow = p_ids + cutlass.Int64(qi) * cutlass.Int64(cap)
            prow = p_dense_block_bitmap + qi * NW
            beyond = False
            for m in cutlass.range(lane, n, 32, unroll=1):
                tok = _block_tok(p_first, lrow, m)
                # a token of the block past the dense prefix and within the bounds (a whole-sequence prefix ends at
                # the sequence, which hi never reaches)
                if (hi_q >= dense_end) & (tok + 3 >= dense_end) & (tok <= hi_q) & (tok + 3 >= floor):
                    beyond = True
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
            if (hi_q > qi) & (ceil > qi):
                # the key tiles after the query's own, up to its bound, must walk down to this query's group
                p_fg = g_first_group.iterator.raw_ptr()
                for tile in cutlass.range(qi // DP_TT + 1 + lane, ceil // DP_TT + 1, 32, unroll=1):
                    prims.red(
                        prims.ReductionOp.MIN,
                        prims.ReductionType.S32,
                        p_fg + tile,
                        qi // DP_G,
                        mem_order=prims.MemOrder.RELAXED,
                        mem_scope=prims.MemScope.GPU,
                    )
            if prims.vote_sync(0xFFFFFFFF, beyond, prims.VoteSync.ANY):
                nanv = cutlass.Vector.from_elements([F32(float("nan"))] * 8, F32)
                p_dq_acc = g_dq_acc.iterator.raw_ptr()
                for v in cutlass.range(lane, HQD // 8, 32, unroll=1):
                    (p_dq_acc + (qi * HQD + v * 8)).store(nanv, alignment=16)


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
    first_group: cute.Tensor,
    dq_acc: cute.Tensor,
    NW: I32,
    HQD: I32,
    stream: cuda_driver.CUstream,
):
    T = counts.shape[0]
    cap = ids.shape[1]
    qsa_bwd_bitmap_kernel(ids, counts, first, lo, hi, seq_start, dense_hi, dense_block_bitmap, first_group, dq_acc, I32(T), I32(cap), NW, HQD).launch(
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
    GH: cutlass.Constexpr[int],
):
    HQ = GH * HKV
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
    GH: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = dk.shape[0]
    qsa_bwd_finalize_kernel(sparse_dk_acc, sparse_dv_acc, dense_dk_acc, dense_dv_acc, dq_acc, dk, dv, dq, I32(T), HAS_SPARSE, HKV, GH).launch(
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


# Key-tile-major path for the dense-prefix queries of each sequence. The two CTAs of a cluster split each group's work
# by operand, so every MMA keeps M=128 and each CTA holds one [128 x 256] fp32 dV or dK accumulator in TMEM across all
# groups.
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
DP_NX = 3  # rank 1's P ring spans DP_NX * DP_CM bytes (R0 slots of PXB). Rank 0 never writes its P ring, so its dS
# slots (R0 groups of DQ_PARTS parts) span both rings
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


@cute.jit
def _dense_rows_load(dst, tmap, s, hk, g, bar, GH: cutlass.Constexpr[int]):
    # slab s of the group's DP_G queries: one box of GQ head rows per query, or one GH-row box per query into its
    # 12-row slot when the tile has more rows than the geometry has heads
    if cutlass.const_expr(GH == GQ):
        prims.cp_async_bulk_tensor_shared_cta_global(dst, tmap.get_ptr(), (s * 64, hk * GQ, g * DP_G), bar)
    else:
        for qq in cutlass.range_constexpr(DP_G):
            prims.cp_async_bulk_tensor_shared_cta_global(dst + (qq * GQ * 128) // 4, tmap.get_ptr(), (s * 64, hk * GH, g * DP_G + qq), bar)


# In the deterministic mode (DET) the pairs that add to one fp32 row add in a fixed order. A pair runs the unit of the
# ticket it draws at its start and waits only for lower tickets, which already run, so no dispatch order deadlocks.
DP_SYNC = 32  # dense_sync words per unit after the ticket counter's line, 16 per rank
DP_SYNC_DQ = 0  # + drain warp: the groups whose dQ^T reduce has completed
DP_SYNC_KV = 4  # + softmax warp: set once the warp's dK / dV rows are flushed
DP_SYNC_TICKET = 15  # of the physical pair: the ticket it drew
DP_KV_ORDER = 4  # words of a unit's flush order in shared memory


@cute.jit
def _unit_ticket(band, tile, hk, ntiles, HKV: cutlass.Constexpr[int]):
    return (band * ntiles + tile) * HKV + hk


@cute.jit
def _dense_ticket(g_dense_sync, bid, tidx):
    # One ticket per pair: rank 0 draws it and the cluster barrier orders its store before the peer's load.
    p_count = g_dense_sync.iterator.raw_ptr()
    p_ticket = p_count + (DP_SYNC + (bid >> 1) * DP_SYNC + DP_SYNC_TICKET)
    if ((bid & 1) == 0) & (tidx == 0):
        p_ticket.store(prims.atomicrmw("add", p_count, I32(1), mem_order=prims.MemOrder.RELAXED, syncscope=prims.MemScope.GPU))
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()
    return p_ticket.load()


@cute.jit
def _tile_groups(p_dense_hi, p_first_group, tile, ntiles, T, lane):
    # The dense query groups [g_lo, g_top) of a key tile, empty outside the grid. They end at the largest dense limit
    # among the sequences holding its tokens, and a bound past the query lowers the first one.
    a0 = tile * DP_TT
    dh = I32(0)
    g_lo = a0 // DP_G
    if (tile >= 0) & (tile < ntiles):
        for u in cutlass.range_constexpr(DP_TT // 32):
            tt = a0 + u * 32 + lane
            if tt < T:
                x = (p_dense_hi + tt).load()
                if x > dh:
                    dh = x
        fg = (p_first_group + tile).load()
        if fg < g_lo:
            g_lo = fg
    dh = cute.arch.warp_reduction_max(dh)
    g_top = (dh + (DP_G - 1)) // DP_G
    g_cap = (T + (DP_G - 1)) // DP_G
    if g_top > g_cap:
        g_top = g_cap
    if a0 >= dh:
        g_top = g_lo
    return g_lo, g_top


@cute.jit
def _band_groups(t_lo, t_top, band, Ut, nparts):
    # Band `band` of a tile's groups [t_lo, t_top): Ut groups each from the top. The lowest band takes the rest, which
    # only a tile shared by two sequences with bounds past the query has (the plan's bands cover a causal tile).
    hi = t_top - band * Ut
    lo = t_lo
    if band < nparts - 1:
        if hi - Ut > lo:
            lo = hi - Ut
    return lo, hi


@cute.jit
def _wait_above(ptr, value):
    # The word only grows, so one acquire load after the relaxed polls still orders the writer's adds before ours.
    x = prims.load_ext(ptr, order=prims.MemOrder.RELAXED, scope=prims.MemScope.GPU)
    while x <= value:
        x = prims.load_ext(ptr, order=prims.MemOrder.RELAXED, scope=prims.MemScope.GPU)
    return prims.load_ext(ptr, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.GPU)


@cute.jit
def _publish(ptr, value):
    # The release store orders this thread's adds, and those of the threads it met at a barrier, before the acquire
    # load of the waiting pair.
    prims.store_ext(value.ir_value(), ptr, order=prims.MemOrder.RELEASE, scope=prims.MemScope.GPU)


@cute.jit
def _kv_order(p_order, band, tile, hk, rank, pr, g_lo, t_lo, ntiles, HKV: cutlass.Constexpr[int]):
    # The two top bands add at once (two adds onto zero commute), the next ones follow the band above, and the lowest,
    # which may be short, adds to the sparse accumulator alone. Shared memory keeps the order out of the group loop's registers.
    off = rank * 16 + DP_SYNC_KV
    after = I32(-1)
    first = I32(-1)
    low = I32(0)
    if band > 1:
        if g_lo > t_lo:
            after = _unit_ticket(band - 1, tile, hk, ntiles, HKV) * DP_SYNC + off
            if band == 2:
                first = _unit_ticket(I32(0), tile, hk, ntiles, HKV) * DP_SYNC + off
        else:
            low = I32(1)
    p_order.store(after)
    (p_order + 1).store(first)
    (p_order + 2).store(low)
    (p_order + 3).store(pr * DP_SYNC + off)


@cute.jit
def _kv_turn(p_order, p_sync, slot, lane, p_dense, p_sparse, PDL: cutlass.Constexpr[int]):
    after = p_order.load()
    if after >= 0:
        # the warp follows its twin of the band above, the third band both of the first two
        if lane == 0:
            _wait_above(p_sync + (after + slot), I32(0))
            first = (p_order + 1).load()
            if first >= 0:
                _wait_above(p_sync + (first + slot), I32(0))
        prims.bar_warp_sync(0xFFFFFFFF)
    pdst = p_dense
    if (p_order + 2).load() != 0:
        pdst = p_sparse
        if cutlass.const_expr(PDL > 0):
            # the key-major grid this launch backfills adds to the sparse accumulator first
            if lane == 0:
                prims.griddepcontrol("wait")
            prims.bar_warp_sync(0xFFFFFFFF)
    return pdst


@cute.jit
def _kv_done(p_order, p_sync, slot, lane):
    prims.bar_warp_sync(0xFFFFFFFF)
    if lane == 0:
        _publish(p_sync + ((p_order + 3).load() + slot), I32(1))


@cute.jit
def _dq_turn(p_sync, slot, g, n_lo, n_top, Ut, nparts, tile, hk, pr, nb_seen, np_seen, ntiles, HKV: cutlass.Constexpr[int]):
    # The previous tile adds group g to dQ first, as step kn of its band bn, whose progress as last read is np_seen.
    if (g >= n_lo) & (g < n_top):
        bn = (n_top - 1 - g) // Ut
        if bn > nparts - 1:
            bn = nparts - 1
        if bn != nb_seen:
            nb_seen = bn
            np_seen = I32(0)
        kn = n_top - bn * Ut - 1 - g
        nu = _unit_ticket(bn, tile - 1, hk, ntiles, HKV)
        # A lower ticket already runs. Tables no valid input gives must not end in a wait on a unit that never starts.
        if (np_seen <= kn) & (nu < pr):
            np_seen = _wait_above(p_sync + (nu * DP_SYNC + slot), kn)
        # the acquire load ordered that reduce before this thread, the fence orders this thread's reduce after it
        prims.fence_proxy(prims.Proxy.ASYNC_GLOBAL)
    return nb_seen, np_seen


@cute.jit
def _dq_done(p_sync, pr, slot, k, lane):
    # A completed reduce is visible to this thread alone. The fence and the release store order it before the acquire
    # load of the next tile.
    prims.cp_async_bulk_wait_group(0)
    if lane == 0:
        prims.fence_proxy(prims.Proxy.ASYNC_GLOBAL)
        _publish(p_sync + (pr * DP_SYNC + slot), k + 1)


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
    g_first_group: cute.Tensor,
    g_dense_sync: cute.Tensor,
    g_sparse_dk_acc: cute.Tensor,
    g_sparse_dv_acc: cute.Tensor,
    T: I32,
    NW: I32,
    ntiles: I32,
    U: I32,
    nparts: I32,
    adapt: I32,
    scale: F32,
    PDL: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
    DQ_PARTS: cutlass.Constexpr[int],
    DET: cutlass.Constexpr[int],
):
    HQ = GH * HKV
    # Rank 1 forms dS from the P that rank 0 sends. With dS in parts, a bf16 P would cap its precision, so P crosses
    # as the top 24 bits of its fp32 value (bf16 high halves, then one byte per value).
    P_LO = DQ_PARTS > 1
    PXB = DP_CM + DP_CM // 2 if P_LO else DP_CM
    # Rank 0 keeps the dS parts of R0 groups in flight and issues dQ^T DQL groups late, so the dS round trip through
    # rank 1 overlaps DQL groups of its work. Rank 1 writes dS once dQ^T of the previous group has read it, so one slot
    # holds its parts. Rank 0 sends P(k) only after its dQ^T(k - R0), so rank 1's P ring of R0 slots is free by then.
    R0 = min(DP_NX, (2 * DP_NX) // DQ_PARTS, (DP_NX * DP_CM) // PXB)
    DQL = R0 - 1
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % 32
    rank = bid & 1
    pr = bid >> 1
    if cutlass.const_expr(DET):
        pr = _dense_ticket(g_dense_sync, bid, tidx)
    if cutlass.const_expr(HKV == 2):
        hk = pr & 1
        rest = pr >> 1
    else:
        hk = I32(0)
        rest = pr
    # the pair's key tile and unit. Key tokens and query rows are positions in the packed buffer, the per-token tables
    # give every query its sequence start and dense limit.
    band = rest // ntiles
    tile = rest - band * ntiles
    a0 = tile * DP_TT
    p_seq_start = g_seq_start.iterator.raw_ptr()
    p_dense_hi = g_dense_hi.iterator.raw_ptr()
    p_first_group = g_first_group.iterator.raw_ptr()
    t_lo, t_top = _tile_groups(p_dense_hi, p_first_group, tile, ntiles, T, lane)
    # Without a sparse grid to backfill, the unit is one wave of the longest sequence's causal work per pair slot
    # (decided on the device, where the sequences are known). With a sparse grid, the tuned unit backfills the
    # key-major grid.
    Ut = U
    if adapt != 0:
        Ut = g_unit.iterator.raw_ptr().load()
        if Ut > U:
            Ut = U
    if cutlass.const_expr(not DET):
        # The parts must cover every group of the tile. A tile shared by two packed sequences whose first group a bound
        # past the query lowered spans more groups than the plan's reach, so its unit grows.
        need = (t_top - t_lo + nparts - 1) // nparts
        if need > Ut:
            Ut = need
    g_lo, g_top = _band_groups(t_lo, t_top, band, Ut, nparts)
    nk = g_top - g_lo
    if nk > 0:
        off = (tile * 3) % nk
        peer = rank ^ 1
        if cutlass.const_expr(DET):
            # tiles add to a group's dQ in order, so every tile walks its groups from the top
            off = I32(0)
            p_sync = g_dense_sync.iterator.raw_ptr() + DP_SYNC
            n_lo, n_top = _tile_groups(p_dense_hi, p_first_group, tile - 1, ntiles, T, lane)

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
        if cutlass.const_expr(DET):
            kv_order = cutlass.Array(I32, DP_KV_ORDER, space=cutlass.AddressSpace.smem, alignment=16)

        sbase = smem.data_ptr()
        if cutlass.const_expr(GH < GQ):
            # the tile rows the GH-row boxes never write must read as zero Q / dO
            zq = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
            for it in cutlass.range(tidx, (DP_OFF_P - DP_OFF_E) // 16, DP_THREADS, unroll=1):
                (sbase + DP_OFF_E // 4 + it * 4).store(zq, alignment=16)
        if warp == 0:
            if prims.elect_sync():
                if cutlass.const_expr(DET):
                    _kv_order(kv_order.data_ptr(), band, tile, hk, rank, pr, g_lo, t_lo, ntiles, HKV)
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
                # receiver-armed exchange: each rank arms its own "received" barriers for the first groups
                for s in cutlass.range_constexpr(DP_NX):
                    if s < nk:
                        if rank == 0:
                            if cutlass.const_expr(s < R0):
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(s), DQ_PARTS * DP_CM)
                        else:
                            if cutlass.const_expr(s < R0):
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(s), PXB)
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
                    prims.mbarrier_arrive_expect_tx(b_ef.subview(se), 4 * DP_G * GH * 128)
                    for s in cutlass.range_constexpr(4):
                        if rank == 0:
                            _dense_rows_load(sbase + (DP_OFF_E + se * DP_HS + s * DP_QSLAB) // 4, tma_qd, s, hk, g, b_ef.subview(se), GH)
                        else:
                            _dense_rows_load(sbase + (DP_OFF_E + se * DP_HS + s * DP_QSLAB) // 4, tma_dod, s, hk, g, b_ef.subview(se), GH)
        elif warp == 3:
            for k in cutlass.range(nk, unroll=1):
                g = g_top - 1 - _rot(k, off, nk)
                sl = k % DP_NL
                _wait(b_le.subview(sl), ((k // DP_NL) & 1) ^ 1)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(b_lf.subview(sl), 4 * DP_G * GH * 128)
                    for s in cutlass.range_constexpr(4):
                        if rank == 0:
                            _dense_rows_load(sbase + (DP_OFF_L + sl * DP_HS + s * DP_QSLAB) // 4, tma_dod, s, hk, g, b_lf.subview(sl), GH)
                        else:
                            _dense_rows_load(sbase + (DP_OFF_L + sl * DP_HS + s * DP_QSLAB) // 4, tma_qd, s, hk, g, b_lf.subview(sl), GH)
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
            d_x0 = _desc(sbase + DP_OFF_P // 4, 768, 128, SWN)
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
                            # the dQ^T MMAs of a K-step share their A operand K^T, which the collector keeps across the parts
                            for kk in cutlass.range_constexpr(8):
                                for part in cutlass.range_constexpr(DQ_PARTS):
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        t_dq,
                                        d_kmn.advance_start_address(kk * 2048),
                                        d_dsb.advance_start_address(part * DP_CM + kk * 1536),
                                        idesc_q,
                                        (part > 0) | (kk > 0),
                                        collector_op=None if DQ_PARTS == 1 else "fill" if part == 0 else "lastuse" if part == DQ_PARTS - 1 else "use",
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
                        xq = kq % R0
                        _wait(b_xr.subview(xq), (kq // R0) & 1)
                        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                        if kq + R0 < nk:
                            if prims.elect_sync():
                                # arm the slot for dS(kq + R0). Rank 1 only sends it after P(kq + R0), which our
                                # softmax sends only after the dQ^T MMAs below have completed.
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(xq), DQ_PARTS * DP_CM)
                        if kq >= 1:
                            _wait(b_dqe, (kq - 1) & 1)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if prims.elect_sync():
                            for kk in cutlass.range_constexpr(8):
                                for part in cutlass.range_constexpr(DQ_PARTS):
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        t_dq,
                                        d_kmn.advance_start_address(kk * 2048),
                                        d_x0.advance_start_address((xq * DQ_PARTS + part) * DP_CM + kk * 1536),
                                        idesc_q,
                                        (part > 0) | (kk > 0),
                                        collector_op=None if DQ_PARTS == 1 else "fill" if part == 0 else "lastuse" if part == DQ_PARTS - 1 else "use",
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
            # a P slot's low bytes follow its high halves: 16 B, then 8 B, per (half, row)
            xlo4 = (DP_OFF_P + DP_CM) // 4 + (hf * 128 + r) * 4
            xlo2 = (DP_OFF_P + DP_CM + 4096) // 4 + (hf * 128 + r) * 2
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
                    ab = qc * HQ + hk * GH
                    av_n.append(_lse_rows(g_lse, qc, hk * GH, GH))
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
                        ab = qc * HQ + hk * GH
                        av_n.append(_lse_rows(g_lse, qc, hk * GH, GH))
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
                    pall = []
                    for jj in cutlass.range_constexpr(2):
                        qi = g * DP_G + hf * 2 + jj
                        sel = (sw[jj] >> (bqs[jj] & 31)) & 1
                        # The token is visible when its bit is set: the bitmap kernel applied the sequence, the key
                        # bounds and causality per token, and a sparse query's row has no bits.
                        valid = row_ok & (qi < T) & (sel != 0)
                        ow = []
                        for hh in cutlass.range_constexpr(GQ):
                            n = jj * GQ + hh
                            a = av[jj][hh]
                            if cutlass.const_expr(n < 16):
                                x = xv0[n]
                            else:
                                x = xv1[n - 16]
                            e = cute.math.exp2(x * c_s - a * F32(LOG2E), fastmath=True)
                            pj = F32(0.0)
                            if valid:
                                pj = e
                            ow.append(pj)
                            pall.append(pj)
                        pk = cutlass.Vector.from_elements(ow, F32).to(BF16).bitcast(I32)
                        for u in cutlass.range_constexpr(GQ // 2):
                            opk.append(pk[u])
                    ovec = cutlass.Vector.from_elements(opk, I32)
                    xvec = ovec
                    if cutlass.const_expr(P_LO):
                        # truncation keeps NaN and inf intact and P = 0 exact
                        pb = cutlass.Vector.from_elements(pall, F32).bitcast(I32)
                        xw = []
                        for w in cutlass.range_constexpr(GQ):
                            xw.append(prims.prmt(pb[2 * w], 0x7632, prims.PermuteMode.DEFAULT, hi=pb[2 * w + 1]))
                        lw = []
                        for w in cutlass.range_constexpr(GQ // 2):
                            t0 = prims.prmt(pb[4 * w], 0x0051, prims.PermuteMode.DEFAULT, hi=pb[4 * w + 1])
                            t1 = prims.prmt(pb[4 * w + 2], 0x5100, prims.PermuteMode.DEFAULT, hi=pb[4 * w + 3])
                            lw.append(prims.prmt(t0, 0x7610, prims.PermuteMode.DEFAULT, hi=t1))
                        xvec = cutlass.Vector.from_elements(xw, I32)
                        lvec = cutlass.Vector.from_elements(lw, I32)
                    xb = k % DP_NX
                    xp = k % R0
                    if k >= 1:
                        # dV(k-1) has read the TMEM P^T slot
                        kp = k - 1
                        _wait(b_pe.subview(kp % DP_NX), (kp // DP_NX) & 1)
                    if k >= R0:
                        # dQ^T(k - R0) has completed: rank 1 has read P(k - R0) (dS(k - R0) arrived) and may receive into
                        # its buffer again, and our dS slot is free for dS(k)
                        _wait(b_xe.subview(k % R0), ((k // R0) & 1) ^ 1)
                    if hf == 0:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[0]).as_ptr(I32), ovec[0:8])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1]).as_ptr(I32), ovec[8:12])
                    else:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1] + 4).as_ptr(I32), ovec[0:4])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[2]).as_ptr(I32), ovec[4:12])
                    rb = prims.mapa(b_xr.subview(xp), peer)
                    for nc in cutlass.range_constexpr(3):
                        _st_async16(prims.mapa(sbase + xoff + xp * (PXB // 4) + nc * 32, peer), xvec[nc * 4 : nc * 4 + 4], rb)
                    if cutlass.const_expr(P_LO):
                        _st_async16(prims.mapa(sbase + xlo4 + xp * (PXB // 4), peer), lvec[0:4], rb)
                        _st_async8(prims.mapa(sbase + xlo2 + xp * (PXB // 4), peer), lvec[4:6], rb)
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
                    ab = qc * HQ + hk * GH
                    av_n.append(_delta_rows(p_delta, ab, GH))
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
                        ab = qc * HQ + hk * GH
                        av_n.append(_delta_rows(p_delta, ab, GH))
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
                    xp = k % R0
                    _wait(b_xr.subview(xp), (k // R0) & 1)
                    pin = []
                    for nc in cutlass.range_constexpr(3):
                        pin.append((sbase + xoff + xp * (PXB // 4) + nc * 32).load(count=4, alignment=16))
                    if cutlass.const_expr(P_LO):
                        lo4 = (sbase + xlo4 + xp * (PXB // 4)).load(count=4, alignment=16)
                        lo2 = (sbase + xlo2 + xp * (PXB // 4)).load(count=2, alignment=8)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if k + R0 < nk:
                        if warp == 4:
                            if prims.elect_sync():
                                # arm this buffer for P(k + R0): rank 0 sends it only after dS(k) (whose quadrant-0
                                # copy this warp issues after this arm) has been consumed by its dQ^T
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(xp), PXB)
                    dpk = []
                    for jj in cutlass.range_constexpr(2):
                        dw = []
                        for hh in cutlass.range_constexpr(GQ):
                            n = jj * GQ + hh
                            a = av[jj][hh]
                            if cutlass.const_expr(n < 16):
                                x = xv0[n]
                            else:
                                x = xv1[n - 16]
                            dw.append(x - a)
                        for hp in cutlass.range_constexpr(GQ // 2):
                            n0 = jj * GQ + 2 * hp
                            pw = pin[n0 // 8][(n0 % 8) // 2]
                            if cutlass.const_expr(P_LO):
                                if cutlass.const_expr(n0 < 16):
                                    lw = lo4[n0 // 4]
                                else:
                                    lw = lo2[n0 // 4 - 4]
                                # the value's low byte repeats its own extra byte, so P = 0 stays exact
                                sb = (n0 % 4) * 0x11
                                p0 = prims.prmt(lw, 0x5400 + sb, prims.PermuteMode.DEFAULT, hi=pw)
                                p1 = prims.prmt(lw, 0x7611 + sb, prims.PermuteMode.DEFAULT, hi=pw)
                                pv = cutlass.Vector.from_elements([p0, p1], I32).bitcast(F32)
                            else:
                                pv = cutlass.Vector.from_elements([pw], I32).bitcast(BF16).to(F32)
                            # a masked token has P = 0 and must give dS = 0 even when its dP is nonfinite
                            d0 = F32(0.0)
                            d1 = F32(0.0)
                            if pv[0] != 0.0:
                                d0 = pv[0] * dw[2 * hp] * scale
                            if pv[1] != 0.0:
                                d1 = pv[1] * dw[2 * hp + 1] * scale
                            dpk.append(d0)
                            dpk.append(d1)
                    # dS^T for the dQ^T MMAs in DQ_PARTS bf16 parts of its fp32 value. dK takes the first.
                    dparts = _bf16_parts(cutlass.Vector.from_elements(dpk, F32), len(dpk), DQ_PARTS)
                    dvec = dparts[0]
                    if k >= 1:
                        # dK / dQ^T(k-1) have read the TMEM dS^T slot and the SMEM parts. The copy of dS(k - R0) has landed
                        # and rank 0's dQ^T(k - R0) has consumed it (both implied by the arrival of P(k)).
                        kp = k - 1
                        _wait(b_pe.subview(kp % DP_NX), (kp // DP_NX) & 1)
                    if hf == 0:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[0]).as_ptr(I32), dvec[0:8])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1]).as_ptr(I32), dvec[8:12])
                    else:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1] + 4).as_ptr(I32), dvec[0:4])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[2]).as_ptr(I32), dvec[4:12])
                    # the parts go to our slot and to rank 0's slot of the group, whose region starts at the P ring
                    x0 = k % R0
                    rb = prims.mapa(b_xr.subview(x0), peer)
                    for part in cutlass.range_constexpr(DQ_PARTS):
                        for nc in cutlass.range_constexpr(3):
                            (sbase + xoffd + part * (DP_CM // 4) + nc * 32).store(dparts[part][nc * 4 : nc * 4 + 4], alignment=16)
                            _st_async16(prims.mapa(sbase + xoff + (x0 * DQ_PARTS + part) * (DP_CM // 4) + nc * 32, peer), dparts[part][nc * 4 : nc * 4 + 4], rb)
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_pf.subview(xb))
            pdst = p_dense_dv_acc
            if rank == 1:
                pdst = p_dense_dk_acc
            if cutlass.const_expr(DET):
                p_sparse = g_sparse_dv_acc.iterator.raw_ptr()
                if rank == 1:
                    p_sparse = g_sparse_dk_acc.iterator.raw_ptr()
                pdst = _kv_turn(kv_order.data_ptr(), p_sync, warp - 4, lane, pdst, p_sparse, PDL)
            _wait(b_kvd, 0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            obase = (tok * HKV + hk) * D + hf * 128
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
            if cutlass.const_expr(DET):
                _kv_done(kv_order.data_ptr(), p_sync, warp - 4, lane)
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
            # rank 0 sends P straight into the peer (st.async) and never touches its own P buffers. With one dS part the
            # room its dS slots leave holds a second staging box per drain warp, so the reduce of group k-1 may still be
            # reading while k is staged.
            STG2 = R0 * DQ_PARTS * DP_CM + 4 * DP_ST_W <= 2 * DP_NX * DP_CM
            stg1 = sbase + (DP_OFF_P + R0 * DQ_PARTS * DP_CM + qd * DP_ST_W) // 4
            if cutlass.const_expr(DET):
                dq_slot = rank * 16 + DP_SYNC_DQ + qd
                nb_seen = I32(-1)
                np_seen = I32(0)
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
                # the reduce that last used this staging box must have finished reading it. With STG2 rank 0 has two
                # boxes, so one reduce may stay in flight. Otherwise each rank has one.
                stg = stg0
                if cutlass.const_expr(STG2):
                    if rank == 0:
                        if (k & 1) == 1:
                            stg = stg1
                        prims.cp_async_bulk_wait_group(1, read=True)
                    else:
                        prims.cp_async_bulk_wait_group(0, read=True)
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
                    if cutlass.const_expr(DET):
                        nb_seen, np_seen = _dq_turn(p_sync, dq_slot, g, n_lo, n_top, Ut, nparts, tile, hk, pr, nb_seen, np_seen, ntiles, HKV)
                    if cutlass.const_expr(GH == GQ):
                        prims.cp_async_bulk_tensor_reduce(tma_dq.get_ptr(), stg, prims.TMARedux.ADD, [rank * 128 + qd * 32, hk * GQ, g * DP_G])
                    else:
                        # the GH real head rows of every query of the group
                        for qq in cutlass.range_constexpr(DP_G):
                            prims.cp_async_bulk_tensor_reduce(
                                tma_dq.get_ptr(), stg + qq * GQ * 32, prims.TMARedux.ADD, [rank * 128 + qd * 32, hk * GH, g * DP_G + qq]
                            )
                prims.cp_async_bulk_commit_group()
                if cutlass.const_expr(DET):
                    _dq_done(p_sync, pr, dq_slot, k, lane)
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
    first_group: cute.Tensor,
    dense_sync: cute.Tensor,
    sparse_dk_acc: cute.Tensor,
    sparse_dv_acc: cute.Tensor,
    NW: I32,
    ntiles: I32,
    U: I32,
    nparts: I32,
    adapt: I32,
    scale: F32,
    PDL: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
    DQ_PARTS: cutlass.Constexpr[int],
    DET: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    HQ = GH * HKV
    T = k.shape[0]
    tma_kd = cuda.create_tensor_map_tiled_from_view(k, box_dims=(DP_TT, 1, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    if cutlass.const_expr(GH == GQ):
        qbox = (DP_G, GQ, 64)
        dqbox = (DP_G, GQ, 32)
    else:
        qbox = (1, GH, 64)  # one box per query into its 12-row slot
        dqbox = (1, GH, 32)
    tma_qd = cuda.create_tensor_map_tiled_from_view(q, box_dims=qbox, stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_dod = cuda.create_tensor_map_tiled_from_view(do, box_dims=qbox, stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_dq = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(dq_acc.iterator, cute.make_layout((T, HQ, D), stride=(HQ * D, D, 1))),
        box_dims=dqbox,
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
        first_group,
        dense_sync,
        sparse_dk_acc,
        sparse_dv_acc,
        I32(T),
        NW,
        ntiles,
        U,
        nparts,
        adapt,
        scale,
        PDL,
        HKV,
        GH,
        DQ_PARTS,
        DET,
    ).launch(grid=(nparts * ntiles * 2 * HKV, 1, 1), block=(DP_THREADS, 1, 1), cluster=(2, 1, 1), stream=stream, use_pdl=(PDL == 1))


# Tuned dense/sparse crossover: below it row i selects 2048 of its i + 1 tokens, dense enough that keeping dK/dV on
# chip beats the record round trip of a (query, block) pair. A sparse remainder under SPARSE_FOLD queries is folded in
# too.
DENSE_QUERIES_MAX = 20480
SPARSE_FOLD = 2048
DP_U_MAX = 448  # query groups per dense work unit beyond which a larger unit no longer amortizes the key tile load and dK / dV flush
