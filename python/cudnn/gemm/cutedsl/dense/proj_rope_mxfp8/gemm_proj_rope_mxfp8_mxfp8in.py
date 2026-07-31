# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused projection GEMM + per-head YARN RoPE + dual-direction MXFP8 quantize (Blackwell / SM100)."""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir import ir as _mlir_ir
from cutlass._mlir.dialects import llvm as _llvm_dialect
from cutlass.cutlass_dsl import Float32 as _Float32Sym, Int32 as _Int32Sym

# ---- DSv3 constants ----
# NUM_HEADS inferred
QK_NOPE = 128
QK_ROPE = 64
HALF = 32
HEAD_DIM = 192  # 128 + 64
BLOCK = 32
FP8_MAX = 448.0

# ---- tile config ----
io_dtype = cutlass.Float8E4M3FN
sf_dtype = cutlass.Float8E8M0FNU
SF_VEC = 32
acc_dtype = cutlass.Float32
TILE_M = 128
TILE_N = HEAD_DIM  # 192, one head per CTA
K_TILE = 128
COLBLK = TILE_M // BLOCK  # col blocks per CTA along tokens
stage_dtype = cutlass.BFloat16  # SMEM staging dtype for post-rope tile (matches TE BF16 RoPE precision)
# Compact-scale row stride (uint32 words per row) is K // 128. It is NOT hardcoded: the class deduces
# it from the contraction dim (weight K) and threads it as a Constexpr (`k_scale_words`) into the host
# -> kernel -> _relay_scale_words, so any K (not just DSv3's 1536) is indexed with the correct stride.
SF_WORDS_PER_K_TILE = K_TILE // (SF_VEC * 4)
SFB_RELAY_ROWS = 256

mma_tiler_mnk = (TILE_M, TILE_N, K_TILE)

ab_stages = 3
acc_stages = 2
NUM_ACCUMULATOR_TMEM_COLS = TILE_N * acc_stages
NUM_SFA_TMEM_COLS = (TILE_M // 32) * 4
NUM_SFB_TMEM_COLS = ((256) // 32) * 4
NUM_TMEM_COLS = 512
NUM_EPI_WARPS = 12  # epilogue warps (rope + quant); 4 of them do T2R staging
T2R_WARPS = 4  # warps that drain TMEM->SMEM (fixed by 128-thread T2R)
threads_in_epilogue = NUM_EPI_WARPS * 32
SACC_STRIDE = 196  # marginally better than 200 in testing
FEATCELL = 64  # VEC2: feature-cell width; lane owns 2 contiguous feats
N_FEATCELL = HEAD_DIM // FEATCELL  # 3
HALFW = 16  # lanes per 32-feature row-block within a featcell

# ---- Structural specialization (compile-time) ----
# Like the SDPA kernels' fixed head_dim, this kernel is specialized for the DeepSeek-V3 Q up-proj
# head geometry: HEAD_DIM=192 (QK_NOPE 128 + QK_ROPE 64), MXFP8 BLOCK=32, TILE_M=128. The epilogue's
# VEC2 feature-cell layout, warp specialization, and single trailing rope cell depend on these exact
# values -- changing them requires reworking the epilogue, not just editing a constant. These asserts
# fail loudly at import if the constants are set to an unsupported combination.
assert FEATCELL == 64, "FEATCELL is warp-size (32) x VEC2 (2); must be 64"
assert QK_NOPE + QK_ROPE == HEAD_DIM, "QK_NOPE + QK_ROPE must equal HEAD_DIM"
assert HEAD_DIM % FEATCELL == 0, "HEAD_DIM must be a whole number of 64-wide feature cells"
assert QK_ROPE == FEATCELL, "the rope occupies exactly one feature cell; QK_ROPE must equal FEATCELL"
assert HALF == QK_ROPE // 2, "HALF must be QK_ROPE // 2"
assert TILE_M % BLOCK == 0, "TILE_M must be a whole number of MXFP8 blocks"
assert NUM_EPI_WARPS == COLBLK * N_FEATCELL, "epilogue warp count must equal COLBLK x N_FEATCELL"


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32


@cute.jit
def _e8m0_inv(sbyte32):
    """E8M0 byte -> its reciprocal scale in fp32 (the shared PTX snippet)."""
    inv_bits = _llvm_dialect.inline_asm(
        _mlir_ir.F32Type.get(),
        [_Int32Sym(sbyte32).ir_value()],
        "{ .reg .s32 t; sub.s32 t, 254, $1; shl.b32 t, t, 23; mov.b32 $0, t; }",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm_dialect.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(inv_bits)


@cute.jit
def _e8m0(amax):
    scaled = amax * cutlass.Float32(1.0 / FP8_MAX)
    packed_i16 = _llvm_dialect.inline_asm(
        _mlir_ir.IntegerType.get_signless(16),
        [_Float32Sym(scaled).ir_value()],
        "cvt.rp.satfinite.ue8m0x2.f32 $0, 0f00000000, $1;",
        "=h,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm_dialect.AsmDialect.AD_ATT,
    )
    sbyte32 = cutlass.Int32(packed_i16) & cutlass.Int32(0xFF)
    return _e8m0_inv(sbyte32), sbyte32


@cute.jit
def _e8m0_pair(amax0, amax1):
    s0 = amax0 * cutlass.Float32(1.0 / FP8_MAX)
    s1 = amax1 * cutlass.Float32(1.0 / FP8_MAX)
    packed_i16 = _llvm_dialect.inline_asm(
        _mlir_ir.IntegerType.get_signless(16),
        [_Float32Sym(s0).ir_value(), _Float32Sym(s1).ir_value()],
        "cvt.rp.satfinite.ue8m0x2.f32 $0, $1, $2;",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm_dialect.AsmDialect.AD_ATT,
    )
    p = cutlass.Int32(packed_i16)
    sbyte0 = (p >> 8) & cutlass.Int32(0xFF)
    sbyte1 = p & cutlass.Int32(0xFF)
    return _e8m0_inv(sbyte0), sbyte0, _e8m0_inv(sbyte1), sbyte1


@cute.jit
def _pack_e4m3x2(vlo, vhi):
    packed_i16 = _llvm_dialect.inline_asm(
        _mlir_ir.IntegerType.get_signless(16),
        [_Float32Sym(vhi).ir_value(), _Float32Sym(vlo).ir_value()],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm_dialect.AsmDialect.AD_ATT,
    )
    return packed_i16


@cute.jit
def _relay_scale_words(mSFA_compact, mSFB_compact, sSFA, sSFB, m_idx, n_idx, k_tile_idx, stage_idx, lane, k_scale_words: cutlass.Constexpr):
    # k_scale_words = K // 128 (uint32 words per compact-scale row); deduced from the contraction dim.
    gSFA_words = cute.make_tensor(
        cute.recast_ptr(mSFA_compact.iterator, dtype=cutlass.Uint32),
        cute.make_layout((mSFA_compact.shape[0], k_scale_words), stride=(k_scale_words, 1)),
    )
    gSFB_words = cute.make_tensor(
        cute.recast_ptr(mSFB_compact.iterator, dtype=cutlass.Uint32),
        cute.make_layout((mSFB_compact.shape[0], k_scale_words), stride=(k_scale_words, 1)),
    )
    sSFA_words = cute.make_tensor(
        cute.recast_ptr(sSFA.iterator, dtype=cutlass.Uint32),
        cute.make_layout(((32, 4), SF_WORDS_PER_K_TILE, ab_stages), stride=((4, 1), 128, 128 * SF_WORDS_PER_K_TILE)),
    )
    sSFB_words = cute.make_tensor(
        cute.recast_ptr(sSFB.iterator, dtype=cutlass.Uint32),
        cute.make_layout(
            (((32, 4), SFB_RELAY_ROWS // 128), SF_WORDS_PER_K_TILE, ab_stages),
            stride=(((4, 1), 128 * SF_WORDS_PER_K_TILE), 128, SFB_RELAY_ROWS * SF_WORDS_PER_K_TILE),
        ),
    )

    sfa_row_base = m_idx * TILE_M
    sfb_row_base = n_idx * TILE_N - (n_idx % 2) * QK_ROPE
    src_word = k_tile_idx * SF_WORDS_PER_K_TILE

    for row_blk in cutlass.range_constexpr(TILE_M // 32):
        src_row = sfa_row_base + row_blk * 32 + lane
        for katom in cutlass.range_constexpr(SF_WORDS_PER_K_TILE):
            sSFA_words[(lane, row_blk), katom, stage_idx] = gSFA_words[src_row, src_word + katom]

    for row_blk in cutlass.range_constexpr(SFB_RELAY_ROWS // 32):
        row_atom = row_blk // 4
        atom_row_blk = row_blk - row_atom * 4
        src_row = sfb_row_base + row_blk * 32 + lane
        for katom in cutlass.range_constexpr(SF_WORDS_PER_K_TILE):
            sSFB_words[((lane, atom_row_blk), row_atom), katom, stage_idx] = gSFB_words[src_row, src_word + katom]


@cute.kernel
def gemm_proj_rope_mxfp8_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    b_smem_layout: cute.ComposedLayout,
    mSFA_compact: cute.Tensor,
    sfa_smem_layout: cute.Layout,
    mSFB_compact: cute.Tensor,
    sfb_smem_layout: cute.Layout,
    mCos: cute.Tensor,
    mSin: cute.Tensor,
    mQrow: cute.Tensor,
    mSrow: cute.Tensor,
    mQcol: cute.Tensor,
    mScol: cute.Tensor,
    epi_tile: cute.Tile,
    cta_layout_vmnk: cute.Layout,
    tile_sched_params: utils.PersistentTileSchedulerParams,
    num_tmem_cols: cutlass.Constexpr,
    num_heads: cutlass.Constexpr,
    t2r_x8: cutlass.Constexpr,
    k_scale_words: cutlass.Constexpr,
):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32

    epilogue_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    mma_warp_id = 12
    tma_warp_id = 13

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(element_type=io_dtype, layout=a_smem_layout.outer, byte_alignment=128, swizzle=a_smem_layout.inner)
    sB = smem.allocate_tensor(element_type=io_dtype, layout=b_smem_layout.outer, byte_alignment=128, swizzle=b_smem_layout.inner)
    sSFA = smem.allocate_tensor(element_type=sf_dtype, layout=sfa_smem_layout, byte_alignment=128)
    sSFB = smem.allocate_tensor(element_type=sf_dtype, layout=sfb_smem_layout, byte_alignment=128)
    sACC = smem.allocate_tensor(element_type=stage_dtype, layout=cute.make_layout((TILE_M, TILE_N), stride=(SACC_STRIDE, 1)), byte_alignment=128)

    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

    gA = cute.local_tile(mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None))
    gB = cute.local_tile(mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None))

    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)

    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))

    epilogue_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_in_epilogue)
    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * len((mma_warp_id, *epilogue_warp_ids)))
    tmem = utils.TmemAllocator(storage.tmem_holding_buffer.ptr, barrier_for_retrieve=tmem_alloc_barrier, allocator_warp_id=epilogue_warp_ids[0], is_two_cta=False)

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        cta_in_cluster_coord_vmnk[2],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        cta_in_cluster_coord_vmnk[1],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    num_tma_copy_bytes = cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])) + cute.size_in_bytes(
        io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
    )

    mainloop_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    mainloop_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=ab_stages,
        producer_group=mainloop_producer_group,
        consumer_group=mainloop_consumer_group,
        tx_count=num_tma_copy_bytes,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    acc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    acc_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=T2R_WARPS)
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=acc_stages,
        producer_group=acc_producer_group,
        consumer_group=acc_consumer_group,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    num_k_tiles = cute.size(tCgA, mode=[4])

    tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
    work_tile = tile_sched.initial_work_tile_info()

    # ================= TMA load warp =================
    if warp_idx == tma_warp_id:
        while work_tile.is_valid_tile:
            coord = work_tile.tile_idx
            m_idx = coord[0]
            n_idx = coord[1]
            tAgA_slice = tAgA[(None, m_idx, None)]
            tBgB_slice = tBgB[(None, n_idx, None)]
            for k_tile_idx in cutlass.range(num_k_tiles):
                handle = ab_producer.acquire_and_advance()
                _relay_scale_words(mSFA_compact, mSFB_compact, sSFA, sSFB, m_idx, n_idx, k_tile_idx, handle.index, lane, k_scale_words)
                cute.arch.sync_warp()
                cute.arch.fence_view_async_shared()
                cute.copy(tma_atom_a, tAgA_slice[(None, k_tile_idx)], tAsA[(None, handle.index)], tma_bar_ptr=handle.barrier)
                cute.copy(tma_atom_b, tBgB_slice[(None, k_tile_idx)], tBsB[(None, handle.index)], tma_bar_ptr=handle.barrier)
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        ab_producer.tail()

    # ================= MMA warp =================
    elif warp_idx == mma_warp_id:
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        sfa_tmem_ptr = cute.recast_ptr(tmem_ptr + NUM_ACCUMULATOR_TMEM_COLS, dtype=sf_dtype)
        sfb_tmem_ptr = cute.recast_ptr(tmem_ptr + NUM_ACCUMULATOR_TMEM_COLS + NUM_SFA_TMEM_COLS, dtype=sf_dtype)
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            SF_VEC,
            cute.slice_(sfa_smem_layout, (None, None, None, 0)),
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            SF_VEC,
            cute.slice_(sfb_smem_layout, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
        tCsSFA_compact = cute.filter_zeros(sSFA)
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        tCsSFB_compact = cute.filter_zeros(sSFB)
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype)
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        tCsSFA_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
        tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfa, tCsSFA_s2t_)
        tCtSFA_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        tCsSFB_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
        tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfb, tCsSFB_s2t_)
        tCtSFB_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)
        while work_tile.is_valid_tile:
            acc_empty = acc_producer.acquire_and_advance()
            coord = work_tile.tile_idx
            n_idx = coord[1]
            tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]
            offset = cutlass.Int32((n_idx % 2) * 2)
            shifted_sfb_ptr = cute.recast_ptr(
                tmem_ptr + NUM_ACCUMULATOR_TMEM_COLS + NUM_SFA_TMEM_COLS + offset,
                dtype=sf_dtype,
            )
            tCtSFB_mma = cute.make_tensor(shifted_sfb_ptr, tCtSFB_layout)
            for k_tile_idx in cutlass.range(num_k_tiles):
                handle = ab_consumer.wait_and_advance()
                s2t_stage_coord = (None, None, None, None, handle.index)
                cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t[s2t_stage_coord], tCtSFA_s2t)
                cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t[s2t_stage_coord], tCtSFB_s2t)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_idx != 0)
                tile_crd = (None, None, None, handle.index)
                cute.gemm(tiled_mma, tCtAcc, [tCrA[tile_crd], tCtSFA], [tCrB[tile_crd], tCtSFB_mma], tCtAcc)
                handle.release()
            acc_empty.commit()
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        acc_producer.tail()

    # ================= Epilogue warps =================
    elif warp_idx < mma_warp_id:
        tmem.allocate(num_tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        t2r_rep = tcgen05.Repetition.x8 if t2r_x8 else tcgen05.Repetition.x32
        copy_atom_t2r = cute.make_copy_atom(tcgen05.Ld32x32bOp(t2r_rep), cutlass.Float32)

        sACC_epi = cute.flat_divide(sACC, epi_tile)
        buf0 = cute.make_rmem_tensor((BLOCK,), cutlass.Float32)
        buf1 = cute.make_rmem_tensor((BLOCK,), cutlass.Float32)
        rPr = cute.make_rmem_tensor((1,), cutlass.Uint16)
        rPc = cute.make_rmem_tensor((1,), cutlass.Uint16)
        st16 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Uint16, num_bits_per_copy=16)
        wid = warp_idx  # epilogue warp id 0..NUM_EPI_WARPS-1

        while work_tile.is_valid_tile:
            coord = work_tile.tile_idx
            m_idx = coord[0]
            head = coord[1]
            token_base = m_idx * TILE_M

            if wid < T2R_WARPS:
                acc_full = acc_consumer.wait_and_advance()
                tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]
                tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
                tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)])
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
                tTR_sACC = thr_copy_t2r.partition_D(sACC_epi)
                tTR_rAcc = cute.make_rmem_tensor(tTR_sACC[(None, None, None, 0, 0)].shape, cutlass.Float32)
                tTR_rStg = cute.make_rmem_tensor(tTR_sACC[(None, None, None, 0, 0)].shape, stage_dtype)
                tTR_tAcc_g = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tTR_sACC_g = cute.group_modes(tTR_sACC, 3, cute.rank(tTR_sACC))
                subtile_cnt = cute.size(tTR_tAcc_g.shape, mode=[3])

                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                    cute.copy(tiled_copy_t2r, tTR_tAcc_g[(None, None, None, subtile_idx)], tTR_rAcc)
                    tTR_rStg.store(tTR_rAcc.load().to(stage_dtype))
                    cute.autovec_copy(tTR_rStg, tTR_sACC_g[(None, None, None, subtile_idx)])

                with cute.arch.elect_one():
                    acc_full.release()

            epilogue_sync_barrier.arrive_and_wait()

            # ---- VEC2: lane owns 2 contiguous features; 12 cells = 12 warps ----
            cell = wid
            cb = cell // N_FEATCELL  # token-block 0..3
            fc = cell % N_FEATCELL  # feature-cell 0..2
            tok0 = cb * BLOCK
            f0 = fc * FEATCELL + 2 * lane  # global feature (even)
            f1 = f0 + 1
            b = fc * 2 + (lane // HALFW)  # 32-feature row-block 0..5
            col_amax0 = cutlass.Float32(0.0)
            col_amax1 = cutlass.Float32(0.0)
            # TE weight layout: NOPE at features 0..127, ROPE at features 128..191 (last cell)
            is_rope = fc == (N_FEATCELL - 1)
            if is_rope:
                lib = lane % HALFW
                pcol0 = QK_NOPE + 4 * lib  # sACC columns 128, 132, ..., 188
                pcol1 = pcol0 + 2
                is_second_half = (lane // HALFW) == 1
                roff = HALF * (lane // HALFW)
                cidx0 = 2 * lib + roff

                # BF16 pre-computed cos/sin tables, shape (T, QK_ROPE); matches TE precision
                cs_row_bytes = QK_ROPE * 2  # BF16 = 2 bytes per element
                cos_base = mCos[token_base + tok0, None].iterator.toint() + cidx0 * 2
                sin_base = mSin[token_base + tok0, None].iterator.toint() + cidx0 * 2
                if is_second_half:
                    # lanes 16..31: v = p*s + q*c
                    for r in cutlass.range_constexpr(BLOCK):
                        p0 = sACC[tok0 + r, pcol0].to(cutlass.Float32)
                        q0 = sACC[tok0 + r, pcol0 + 1].to(cutlass.Float32)
                        p1 = sACC[tok0 + r, pcol1].to(cutlass.Float32)
                        q1 = sACC[tok0 + r, pcol1 + 1].to(cutlass.Float32)
                        tc = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, cos_base + r * cs_row_bytes, cute.AddressSpace.gmem, assumed_align=2), (2,))
                        ts = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, sin_base + r * cs_row_bytes, cute.AddressSpace.gmem, assumed_align=2), (2,))
                        c0 = tc[0].to(cutlass.Float32)
                        s0 = ts[0].to(cutlass.Float32)
                        c1 = tc[1].to(cutlass.Float32)
                        s1 = ts[1].to(cutlass.Float32)
                        ps0, ps1 = cute.arch.mul_packed_f32x2((p0, p1), (s0, s1))
                        v0, v1 = cute.arch.fma_packed_f32x2((q0, q1), (c0, c1), (ps0, ps1))
                        buf0[r] = v0
                        buf1[r] = v1
                        col_amax0 = cute.arch.fmax(col_amax0, cute.arch.fmax(v0, -v0))
                        col_amax1 = cute.arch.fmax(col_amax1, cute.arch.fmax(v1, -v1))
                else:
                    # lanes 0..15: v = p*c - q*s
                    for r in cutlass.range_constexpr(BLOCK):
                        p0 = sACC[tok0 + r, pcol0].to(cutlass.Float32)
                        q0 = sACC[tok0 + r, pcol0 + 1].to(cutlass.Float32)
                        p1 = sACC[tok0 + r, pcol1].to(cutlass.Float32)
                        q1 = sACC[tok0 + r, pcol1 + 1].to(cutlass.Float32)
                        tc = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, cos_base + r * cs_row_bytes, cute.AddressSpace.gmem, assumed_align=2), (2,))
                        ts = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, sin_base + r * cs_row_bytes, cute.AddressSpace.gmem, assumed_align=2), (2,))
                        c0 = tc[0].to(cutlass.Float32)
                        s0 = ts[0].to(cutlass.Float32)
                        c1 = tc[1].to(cutlass.Float32)
                        s1 = ts[1].to(cutlass.Float32)
                        pc0, pc1 = cute.arch.mul_packed_f32x2((p0, p1), (c0, c1))
                        qs0, qs1 = cute.arch.mul_packed_f32x2((q0, q1), (s0, s1))
                        v0, v1 = cute.arch.fma_packed_f32x2((qs0, qs1), (cutlass.Float32(-1.0), cutlass.Float32(-1.0)), (pc0, pc1))
                        buf0[r] = v0
                        buf1[r] = v1
                        col_amax0 = cute.arch.fmax(col_amax0, cute.arch.fmax(v0, -v0))
                        col_amax1 = cute.arch.fmax(col_amax1, cute.arch.fmax(v1, -v1))
            else:
                for r in cutlass.range_constexpr(BLOCK):
                    v0 = sACC[tok0 + r, f0].to(cutlass.Float32)
                    v1 = sACC[tok0 + r, f1].to(cutlass.Float32)
                    buf0[r] = v0
                    buf1[r] = v1
                    col_amax0 = cute.arch.fmax(col_amax0, cute.arch.fmax(v0, -v0))
                    col_amax1 = cute.arch.fmax(col_amax1, cute.arch.fmax(v1, -v1))
            invc0, sbc0, invc1, sbc1 = _e8m0_pair(col_amax0, col_amax1)
            scol_row = m_idx * COLBLK + cb
            mScol[scol_row, head, f0] = cutlass.Uint8(sbc0)
            mScol[scol_row, head, f1] = cutlass.Uint8(sbc1)
            is_leader = (lane % HALFW) == 0
            vchunk = f0 // 2  # VEC=2 chunk index within HEAD_DIM

            row_bytes = num_heads * HEAD_DIM
            pr_base = mQrow[token_base + tok0, head, None].iterator.toint()
            pc_base = mQcol[token_base + tok0, head, None].iterator.toint()
            # First pass: compute all row_amax for this warp's tokens
            inv_r_buf = cute.make_rmem_tensor((BLOCK,), cutlass.Float32)
            sbyte_r_buf = cute.make_rmem_tensor((BLOCK,), cutlass.Int32)
            for r in cutlass.range_constexpr(BLOCK):
                v0 = buf0[r]
                v1 = buf1[r]
                m = cute.arch.fmax(cute.arch.fmax(v0, -v0), cute.arch.fmax(v1, -v1))
                o8 = cute.arch.shuffle_sync_bfly(m, 8)
                m = cute.arch.fmax(m, o8)
                o4 = cute.arch.shuffle_sync_bfly(m, 4)
                m = cute.arch.fmax(m, o4)
                o2 = cute.arch.shuffle_sync_bfly(m, 2)
                m = cute.arch.fmax(m, o2)
                o1 = cute.arch.shuffle_sync_bfly(m, 1)
                row_amax = cute.arch.fmax(m, o1)
                inv_r, sbyte_r = _e8m0(row_amax)
                inv_r_buf[r] = inv_r
                sbyte_r_buf[r] = sbyte_r
            # Second pass: scale, pack, store
            for r in cutlass.range_constexpr(BLOCK):
                token = token_base + tok0 + r
                v0 = buf0[r]
                v1 = buf1[r]
                inv_r = inv_r_buf[r]
                sbyte_r = sbyte_r_buf[r]
                if is_leader:
                    mSrow[token, head, b] = cutlass.Uint8(sbyte_r)
                vr0, vr1 = cute.arch.mul_packed_f32x2((v0, v1), (inv_r, inv_r))
                vc0, vc1 = cute.arch.mul_packed_f32x2((v0, v1), (invc0, invc1))
                rPr[0] = cutlass.Uint16(_pack_e4m3x2(vr0, vr1))
                rPc[0] = cutlass.Uint16(_pack_e4m3x2(vc0, vc1))
                pr = cute.make_ptr(cutlass.Uint16, pr_base + r * row_bytes, cute.AddressSpace.gmem, assumed_align=16)
                pc = cute.make_ptr(cutlass.Uint16, pc_base + r * row_bytes, cute.AddressSpace.gmem, assumed_align=16)
                gr = cute.tiled_divide(cute.make_tensor(pr, (HEAD_DIM // 2,)), (1,))
                gc = cute.tiled_divide(cute.make_tensor(pc, (HEAD_DIM // 2,)), (1,))
                cute.copy(st16, rPr, gr[None, vchunk])
                cute.copy(st16, rPc, gc[None, vchunk])

            epilogue_sync_barrier.arrive_and_wait()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def gemm_proj_rope_mxfp8_host(
    mA: cute.Tensor,
    mSFA_compact: cute.Tensor,
    mB: cute.Tensor,
    mSFB_compact: cute.Tensor,
    mCos: cute.Tensor,
    mSin: cute.Tensor,
    mQrow: cute.Tensor,
    mSrow: cute.Tensor,
    mQcol: cute.Tensor,
    mScol: cute.Tensor,
    grid_m: cutlass.Constexpr,
    num_heads: cutlass.Constexpr,
    max_active_clusters: cutlass.Constexpr,
    swizzle_size: cutlass.Constexpr,
    t2r_x8: cutlass.Constexpr,
    k_scale_words: cutlass.Constexpr,
    stream,
):
    a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
    b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()

    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        io_dtype,
        io_dtype,
        a_major,
        b_major,
        sf_dtype,
        SF_VEC,
        tcgen05.CtaGroup.ONE,
        (TILE_M, TILE_N),
    )

    a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, mA.element_type, ab_stages)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, mB.element_type, ab_stages)
    sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, SF_VEC, ab_stages)
    sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, SF_VEC, ab_stages)

    cluster_shape_mnk = (1, 1, 1)
    cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

    a_smem_layout_1 = cute.slice_(a_smem_layout, (None, None, None, 0))
    b_smem_layout_1 = cute.slice_(b_smem_layout, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mA, a_smem_layout_1, mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape)
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mB, b_smem_layout_1, mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape)

    cta_tile_shape_mnk = (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_tiler_mnk[2])
    c_layout_kind = utils.LayoutEnum.ROW_MAJOR
    epi_tile = utils.compute_epilogue_tile_shape(cta_tile_shape_mnk, False, c_layout_kind, cutlass.Float32)

    # TMEM budget is fixed for this tile config (accumulator + SFA/SFB scale factors); the kernel
    # computes the accumulator fragment layout itself, so no host-side fake fragment is needed here.
    num_tmem_cols = NUM_TMEM_COLS

    num_ctas_mnl = (grid_m, num_heads, 1)
    tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnk, swizzle_size, True)
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

    gemm_proj_rope_mxfp8_kernel(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        a_smem_layout,
        tma_atom_b,
        tma_tensor_b,
        b_smem_layout,
        mSFA_compact,
        sfa_smem_layout,
        mSFB_compact,
        sfb_smem_layout,
        mCos,
        mSin,
        mQrow,
        mSrow,
        mQcol,
        mScol,
        epi_tile,
        cta_layout_vmnk,
        tile_sched_params,
        num_tmem_cols,
        num_heads,
        t2r_x8,
        k_scale_words,
    ).launch(
        grid=grid,
        block=[(NUM_EPI_WARPS + 2) * 32, 1, 1],
        cluster=cluster_shape_mnk,
        stream=stream,
    )


def _as_e8m0(t):
    ct = from_dlpack(t.detach(), assumed_align=16)
    ct.element_type = cutlass.Float8E8M0FNU
    return ct.mark_layout_dynamic(leading_dim=1)
