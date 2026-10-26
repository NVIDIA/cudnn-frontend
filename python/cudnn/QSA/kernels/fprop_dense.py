# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense 128-query tiles with exact masks and paired-key online softmax."""

import hashlib
import math
from pathlib import Path
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
from cutlass.pipeline import Agent, CooperativeGroup
from cutlass.cute.runtime import make_fake_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32, Int64
from cuda.bindings import driver as cuda
from cudnn.frost.compiled_cache import compile_cached, template_key

DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FROST_SOURCE_DIGEST = DIGEST
LOG2E, LN2 = math.log2(math.e), math.log(2.0)
TILE_M, TILE_N, HEAD_DIM = 128, 64, 256
TPC, WPC, CHUNK_KEYS, WARP, WG = 2, 4, 128, 32, 128
GROUP = 2
NEG_BIG = cutlass.Float32(-3.0e38)


@cute.kernel
def frost_qsa_fprop_dense(
    mma_tile_qk: cute.Tile,
    mma_tile_pv: cute.Tile,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    tiled_mma_pv64: cute.TiledMma,
    tma_q: cute.nvgpu.cpasync.TmaInfo,
    tma_k: cute.nvgpu.cpasync.TmaInfo,
    tma_v: cute.nvgpu.cpasync.TmaInfo,
    mBitsF: cute.Tensor,  # flat (T*W,) int32
    mRanges: cute.Tensor,
    mMode: cute.Tensor,
    mOrder: cute.Tensor,
    mOF: cute.Tensor,  # flat (T*Hq*D,) bf16
    mO32F: cute.Tensor,  # flat (T*Hq*D,) f32
    mLSEF: cute.Tensor,  # flat (Hq*T,) f32
    scale_log2: Float32,
    t_total: Int32,
    nw: Int32,
    h_r: cutlass.Constexpr,
    n_heads: cutlass.Constexpr,
):
    head, slot, _ = cute.arch.block_idx()
    tile = mOrder[slot]
    if mMode[tile] == 0:
        acc_dtype = Float32
        mma_dtype = cutlass.BFloat16

        coord_q = tile
        head_kv = head // h_r
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // WARP)
        wg_idx = cute.arch.make_warp_uniform(tidx // WG)
        wg_tidx = tidx % WG

        mma_warp_qk_id = 0
        mma_warp_pv_id = 1
        tma_warp_qk_id = 2
        tma_warp_vo_id = 3
        softmax_wg_id = 1

        t0 = coord_q * TILE_M
        k_begin = mRanges[coord_q, 0]
        n_chunks = mRanges[coord_q, 1]
        n_tiles = n_chunks * TPC
        tile0 = k_begin // TILE_N
        chunk0 = k_begin // CHUNK_KEYS

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        mcast_coord = (0, 0, 0, 0)
        mcast_layout = cute.make_layout((1, 1, 1, 1))

        smem = utils.SmemAllocator()
        tmem_ptr_smem_ptr = smem.allocate_array(Int32, 1)
        tmem_alloc_cols = 512
        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)

        elect_one = CooperativeGroup(Agent.Thread)
        wg_coop = CooperativeGroup(Agent.Thread, WG)

        q_smem_bytes = cute.size_in_bytes(mma_dtype, tma_q.smem_layout)
        q_mbar_ptr = smem.allocate_array(Int64, 1)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_mbar_ptr, 1)

        k_stages = tma_k.smem_layout.shape[-1]
        k_stage_bytes = cute.size_in_bytes(mma_dtype, tma_k.smem_layout) // k_stages
        k_pipe_ptr = smem.allocate_array(Int64, k_stages * 2)
        k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=k_pipe_ptr,
            num_stages=k_stages,
            producer_group=elect_one,
            consumer_group=elect_one,
            tx_count=k_stage_bytes,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()

        v_stages = tma_v.smem_layout.shape[-1]
        v_stage_bytes = cute.size_in_bytes(mma_dtype, tma_v.smem_layout) // v_stages
        v_pipe_ptr = smem.allocate_array(Int64, v_stages * 2)
        v_producer, v_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=v_pipe_ptr,
            num_stages=v_stages,
            producer_group=elect_one,
            consumer_group=elect_one,
            tx_count=v_stage_bytes,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()

        s_stages = 2
        s_pipe_ptr = smem.allocate_array(Int64, s_stages * 2)
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=s_pipe_ptr,
            num_stages=s_stages,
            producer_group=elect_one,
            consumer_group=wg_coop,
            defer_sync=True,
        ).make_participants()

        p_stages = 2
        p_pipe_ptr = smem.allocate_array(Int64, p_stages * 2)
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=p_stages,
            producer_group=wg_coop,
            consumer_group=elect_one,
            barrier_storage=p_pipe_ptr,
            defer_sync=True,
        ).make_participants()

        o_stages = 1
        o_pipe_ptr = smem.allocate_array(Int64, o_stages * 2)
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=o_stages,
            producer_group=elect_one,
            consumer_group=wg_coop,
            barrier_storage=o_pipe_ptr,
            defer_sync=True,
        ).make_participants()

        tAsQ_layout = tma_q.smem_layout
        tAsQ = smem.allocate_tensor(mma_dtype, tAsQ_layout.outer, 128, tAsQ_layout.inner)
        tBsK_layout = tma_k.smem_layout
        tBsK = smem.allocate_tensor(mma_dtype, tBsK_layout.outer, 128, tBsK_layout.inner)
        tBsV_layout = tma_v.smem_layout
        tBsV = smem.allocate_tensor(mma_dtype, tBsV_layout.outer, 128, tBsV_layout.inner)

        cute.arch.sync_threads()

        tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
        tmem = utils.TmemBufferPool(tmem_ptr, tmem_alloc_cols)

        tCtS_shape = tiled_mma_qk.partition_shape_C((TILE_M, TILE_N, s_stages))
        tCtS = tiled_mma_qk.make_fragment_C(tCtS_shape)
        tCtS = tmem.allocate_tensor(tCtS.layout, acc_dtype)

        tAtP_shape = tiled_mma_pv.partition_shape_A((TILE_M, TILE_N, p_stages))
        tAtP = tiled_mma_pv.make_fragment_A(tAtP_shape)
        tAtP = tmem.allocate_tensor(tAtP.layout, mma_dtype)

        tCtO_shape = tiled_mma_pv.partition_shape_C((TILE_M, HEAD_DIM))
        tCtO = tiled_mma_pv.make_fragment_C(tCtO_shape)
        tCtO = tmem.allocate_tensor(tCtO.layout, acc_dtype)
        o_chunk_layout = tiled_mma_pv64.make_fragment_C(tiled_mma_pv64.partition_shape_C((TILE_M, TILE_N, HEAD_DIM // TILE_N))).layout
        tCtO4 = cute.make_tensor(tCtO.iterator, o_chunk_layout)

        mma_dice = (None, None, None)
        cpy_dice = (None, *mma_dice)

        # Load Q and K.
        if warp_idx == tma_warp_qk_id:
            gQ = cute.local_tile(tma_q.tma_tensor, tiler=(TILE_M, HEAD_DIM), coord=(coord_q, 0, head))
            tAgQ = thr_mma_qk.partition_A(gQ)
            tAsQ_tma, tAgQ_tma = cute.nvgpu.cpasync.tma_partition(
                tma_q.atom, mcast_coord, mcast_layout, cute.group_modes(tAsQ, 0, 3), cute.group_modes(tAgQ, 0, 3)
            )

            gK = cute.local_tile(tma_k.tma_tensor, tiler=(TILE_N, HEAD_DIM), coord=(None, 0, head_kv))
            tBgK = thr_mma_qk.partition_B(gK)
            tBsK_tma, tBgK_tma = cute.nvgpu.cpasync.tma_partition(
                tma_k.atom, mcast_coord, mcast_layout, cute.group_modes(tBsK, 0, 3), cute.group_modes(tBgK, 0, 3)
            )

            cute.copy(tma_q.atom, tAgQ_tma, tAsQ_tma[None, 0], tma_bar_ptr=q_mbar_ptr)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(q_mbar_ptr, q_smem_bytes)

            for s in cutlass.range(n_tiles):
                h = k_producer.acquire_and_advance()
                cute.copy(tma_k.atom, tBgK_tma[None, tile0 + s], tBsK_tma[None, h.index], tma_bar_ptr=h.barrier)

        # Load V.
        elif warp_idx == tma_warp_vo_id:
            gV = cute.local_tile(tma_v.tma_tensor, tiler=(HEAD_DIM, TILE_N), coord=(0, None, head_kv))
            tBgV = thr_mma_pv.partition_B(gV)
            tBsV_tma, tBgV_tma = cute.nvgpu.cpasync.tma_partition(
                tma_v.atom, mcast_coord, mcast_layout, smem_tensor=cute.group_modes(tBsV, 0, 3), gmem_tensor=cute.group_modes(tBgV, 0, 3)
            )
            for s in cutlass.range(n_tiles):
                h = v_producer.acquire_and_advance()
                cute.copy(tma_v.atom, tBgV_tma[None, tile0 + s], tBsV_tma[None, h.index], tma_bar_ptr=h.barrier)

        # Compute QK.
        elif warp_idx == mma_warp_qk_id:
            tAsQ_desc = thr_mma_qk.make_fragment_A(tAsQ)
            tBsK_desc = thr_mma_qk.make_fragment_B(tBsK)
            cute.arch.mbarrier_wait(q_mbar_ptr, phase=0)
            num_mma_k = cute.size(tBsK.shape[2])
            for s in cutlass.range(n_tiles):
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                kh = k_consumer.wait_and_advance()
                sh = s_producer.acquire_and_advance()
                tAsQ_s = tAsQ_desc[mma_dice + (0,)]
                tBsK_s = tBsK_desc[mma_dice + (kh.index,)]
                tCtS_s = tCtS[mma_dice + (sh.index,)]
                for mk in cutlass.range_constexpr(num_mma_k):
                    cute.gemm(tiled_mma_qk, tCtS_s, tAsQ_s[None, None, mk], tBsK_s[None, None, mk], tCtS_s)
                    if mk == 0:
                        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                kh.release()
                sh.commit()

        # Compute PV.
        elif warp_idx == mma_warp_pv_id:
            tBsV_desc = thr_mma_pv.make_fragment_B(tBsV)
            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
            num_mma_k = cute.size(tBsV.shape[2])
            for s in cutlass.range(n_tiles // GROUP):
                oh = o_producer.acquire_and_advance()
                for g in cutlass.range_constexpr(GROUP):
                    vh = v_consumer.wait_and_advance()
                    ph = p_consumer.wait_and_advance()
                    tAtP_s = tAtP[mma_dice + (ph.index,)]
                    tBsV_s = tBsV_desc[mma_dice + (vh.index,)]
                    for mk in cutlass.range_constexpr(num_mma_k):
                        cute.gemm(tiled_mma_pv, tCtO, tAtP_s[None, None, mk], tBsV_s[None, None, mk], tCtO)
                        if mk == 0:
                            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                    vh.release()
                    ph.release()
                oh.commit()
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # Softmax, correction, and output.
        elif wg_idx == softmax_wg_id:
            rep = tcgen05.Repetition(TILE_N)
            tmem_ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(rep), acc_dtype)
            tmem_st_o_atom = cute.make_copy_atom(tcgen05.St32x32bOp(rep), acc_dtype)
            tmem_ld = tcgen05.make_tmem_copy(tmem_ld_atom, tCtS[mma_dice + (0,)])
            thr_ld = tmem_ld.get_slice(wg_tidx)

            rep_p = tcgen05.Repetition(TILE_N * mma_dtype.width // 32)
            tmem_st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(rep_p), mma_dtype)
            tmem_st = tcgen05.make_tmem_copy(tmem_st_atom, tAtP[mma_dice + (0,)])
            thr_st = tmem_st.get_slice(wg_tidx)

            tStS = thr_ld.partition_S(tCtS)
            tSrS_shape = thr_ld.partition_D(tCtS).shape[:-1]
            tPtP = thr_st.partition_D(tAtP)
            tPrP_shape = thr_st.partition_S(tAtP).shape[:-1]

            tmem_ld_o = tcgen05.make_tmem_copy(tmem_ld_atom, tCtO4[mma_dice + (0,)])
            thr_ld_o = tmem_ld_o.get_slice(wg_tidx)
            tOtO = thr_ld_o.partition_S(tCtO4)
            tOrO_shape = thr_ld_o.partition_D(tCtO4).shape[:-1]

            row = t0 + wg_tidx
            row_safe = row
            if row_safe > t_total - 1:
                row_safe = t_total - 1

            prev_max = NEG_BIG
            row_sum = Float32(0.0)
            rWords = cute.make_rmem_tensor(cute.make_layout(WPC), Int32)
            tPrP_layout = cute.make_layout(tPrP_shape)

            bits_base = Int64(row_safe) * Int64(nw) + Int64(chunk0) * WPC
            rAhead = cute.make_rmem_tensor(cute.make_layout(WPC), Int32)
            cute.autovec_copy(cute.make_tensor(mBitsF.iterator + bits_base, cute.make_layout(WPC)), rWords)
            for ch in cutlass.range(n_chunks):
                ahead = ch + 1
                if ahead >= n_chunks:
                    ahead = n_chunks - 1
                gW = cute.make_tensor(mBitsF.iterator + (bits_base + ahead * WPC), cute.make_layout(WPC))
                cute.autovec_copy(gW, rAhead)
                for pair in cutlass.range_constexpr(TPC // GROUP):
                    slots = []
                    for g in cutlass.range_constexpr(GROUP):
                        sh = s_consumer.wait_and_advance()
                        tSrS = cute.make_rmem_tensor(tSrS_shape, acc_dtype)
                        cute.copy(tmem_ld_atom, tStS[cpy_dice + (sh.index,)], tSrS)
                        cute.arch.fence_view_async_tmem_load()
                        sh.release()
                        tPrS = cute.composition(tSrS, tPrP_layout)
                        for half in cutlass.range_constexpr(TILE_N // 32):
                            wd = rWords[(TILE_N // 32) * (GROUP * pair + g) + half]
                            for i in cutlass.range_constexpr(32):
                                bit = (wd >> i) & 1
                                j = half * 32 + i
                                scaled = tPrS[j] * scale_log2
                                tPrS[j] = Float32(arith.select((bit != 0).ir_value(), scaled.ir_value(), Float32(-float("inf")).ir_value()))
                        slots.append(tPrS.load())

                    new_max = slots[0].reduce(cute.ReductionOp.MAX, prev_max, 0)
                    for g in cutlass.range_constexpr(1, GROUP):
                        new_max = cute.math.max(new_max, slots[g].reduce(cute.ReductionOp.MAX, prev_max, 0))
                    m_use = cute.math.max(new_max, NEG_BIG)

                    psum = Float32(0.0)
                    for g in cutlass.range_constexpr(GROUP):
                        p_ssa = cute.math.exp2(slots[g] - m_use, fastmath=True)
                        psum = psum + p_ssa.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                        tPrP = cute.make_rmem_tensor(tPrP_layout, mma_dtype)
                        tPrP.store(p_ssa.to(mma_dtype))
                        ph = p_producer.acquire_and_advance()
                        cute.copy(tmem_st_atom, tPrP, tPtP[cpy_dice + (ph.index,)])
                        cute.arch.fence_view_async_tmem_store()
                        ph.commit()

                    corr = Float32(1.0)
                    do_corr = cutlass.Boolean(True)
                    if pair == 0:
                        do_corr = ch > 0
                    if do_corr:
                        corr = cute.math.exp2(prev_max - m_use, fastmath=True)
                        oh = o_consumer.wait_and_advance()
                        if cutlass.Uint32(cute.arch.vote_ballot_sync(corr != Float32(1.0))) != cutlass.Uint32(0):
                            for c in cutlass.range_constexpr(HEAD_DIM // TILE_N):
                                tOrO = cute.make_rmem_tensor(tOrO_shape, acc_dtype)
                                cute.copy(tmem_ld_atom, tOtO[cpy_dice + (c,)], tOrO)
                                cute.arch.fence_view_async_tmem_load()
                                tOrO.store(corr * tOrO.load())
                                cute.copy(tmem_st_o_atom, tOrO, tOtO[cpy_dice + (c,)])
                            cute.arch.fence_view_async_tmem_store()
                        oh.release()
                    prev_max = m_use
                    row_sum = row_sum * corr + psum
                rWords.store(rAhead.load())

            oh = o_consumer.wait_and_advance()
            denom = cute.math.max(row_sum, Float32(1e-20))
            rcp = cute.arch.rcp_approx(denom)
            for c in cutlass.range_constexpr(HEAD_DIM // TILE_N):
                tOrO = cute.make_rmem_tensor(tOrO_shape, acc_dtype)
                cute.copy(tmem_ld_atom, tOtO[cpy_dice + (c,)], tOrO)
                cute.arch.fence_view_async_tmem_load()
                vals = tOrO.load() * rcp
                tOrO.store(vals)
                if row < t_total:
                    ooff = (Int64(row) * n_heads + head) * HEAD_DIM + c * TILE_N
                    g32 = cute.make_tensor(mO32F.iterator + ooff, cute.make_layout(TILE_N))
                    cute.autovec_copy(tOrO, g32)
                    tOb = cute.make_rmem_tensor(tOrO_shape, cutlass.BFloat16)
                    tOb.store(vals.to(cutlass.BFloat16))
                    gbf = cute.make_tensor(mOF.iterator + ooff, cute.make_layout(TILE_N))
                    cute.autovec_copy(tOb, gbf)
            oh.release()

            if row < t_total:
                lse = (prev_max + cute.math.log2(row_sum, fastmath=True)) * Float32(LN2)
                mLSEF[Int64(head) * Int64(t_total) + row] = lse


frost_qsa_fprop_dense.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _attn_launch(
    mQ,
    mK,
    mV,
    mBitsF,
    mRanges,
    mMode,
    mOrder,
    mOF,
    mO32F,
    mLSEF,
    scale: Float32,
    nw: Int32,
    stream: cuda.CUstream,
    h_r: cutlass.Constexpr,
    n_heads: cutlass.Constexpr,
):
    mma_dtype = cutlass.BFloat16
    acc_dtype = Float32

    mma_tile_qk = (TILE_M, TILE_N, HEAD_DIM)
    mma_tile_pv = (TILE_M, HEAD_DIM, TILE_N)

    tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
        mma_dtype, mma_dtype, OperandMajorMode.K, OperandMajorMode.K, acc_dtype, tcgen05.CtaGroup.ONE, (TILE_M, TILE_N), tcgen05.OperandSource.SMEM
    )
    tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
        mma_dtype, mma_dtype, OperandMajorMode.K, OperandMajorMode.MN, acc_dtype, tcgen05.CtaGroup.ONE, (TILE_M, HEAD_DIM), tcgen05.OperandSource.TMEM
    )
    tiled_mma_pv64 = sm100_utils.make_trivial_tiled_mma(
        mma_dtype, mma_dtype, OperandMajorMode.K, OperandMajorMode.MN, acc_dtype, tcgen05.CtaGroup.ONE, (TILE_M, TILE_N), tcgen05.OperandSource.TMEM
    )
    smem_layout_q = sm100_utils.make_smem_layout_a(tiled_mma_qk, mma_tile_qk, mma_dtype, 1)
    smem_layout_k = sm100_utils.make_smem_layout_b(tiled_mma_qk, mma_tile_qk, mma_dtype, 2)
    smem_layout_v = sm100_utils.make_smem_layout_b(tiled_mma_pv, mma_tile_pv, mma_dtype, 2)

    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    tma_q = cute.nvgpu.make_tiled_tma_atom_A(op, mQ, smem_layout_q, mma_tile_qk, tiled_mma_qk)
    tma_k = cute.nvgpu.make_tiled_tma_atom_B(op, mK, smem_layout_k, mma_tile_qk, tiled_mma_qk)
    tma_v = cute.nvgpu.make_tiled_tma_atom_B(op, mV, smem_layout_v, mma_tile_pv, tiled_mma_pv)

    t_total = mQ.shape[0]
    frost_qsa_fprop_dense(
        mma_tile_qk,
        mma_tile_pv,
        tiled_mma_qk,
        tiled_mma_pv,
        tiled_mma_pv64,
        tma_q,
        tma_k,
        tma_v,
        mBitsF,
        mRanges,
        mMode,
        mOrder,
        mOF,
        mO32F,
        mLSEF,
        scale * Float32(LOG2E),
        Int32(t_total),
        nw,
        h_r,
        n_heads,
    ).launch(
        grid=[n_heads, cute.ceil_div(t_total, TILE_M), 1],
        block=[2 * WG, 1, 1],
        cluster=[1, 1, 1],
        min_blocks_per_mp=1,
        stream=stream,
    )


def compile_kernel(hq, hkv, arch):
    total, bits, tiles = cute.sym_int64(), cute.sym_int64(), cute.sym_int64()
    q = make_fake_tensor(cutlass.BFloat16, (total, 256, hq), (hq * 256, 1, 256), assumed_align=16)
    k = make_fake_tensor(cutlass.BFloat16, (total, 256, hkv), (hkv * 256, 1, 256), assumed_align=16)
    v = make_fake_tensor(cutlass.BFloat16, (256, total, hkv), (1, hkv * 256, 256), assumed_align=16)

    def flat(dtype, length):
        return make_fake_tensor(dtype, (length,), (1,), assumed_align=16)

    return compile_cached(
        _attn_launch,
        q,
        k,
        v,
        flat(Int32, bits),
        make_fake_tensor(Int32, (tiles, 2), (2, 1), assumed_align=4),
        flat(Int32, tiles),
        flat(Int32, tiles),
        flat(cutlass.BFloat16, total * hq * 256),
        flat(Float32, total * hq * 256),
        flat(Float32, total * hq),
        Float32(1),
        Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        hq // hkv,
        hq,
        cache_key=template_key(globals(), {"hq": hq, "hkv": hkv, "arch": arch, "source": DIGEST}, "dense"),
        symbol="qsa_hybrid_dense",
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
    )
