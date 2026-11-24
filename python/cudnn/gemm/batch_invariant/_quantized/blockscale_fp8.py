# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted for cuDNN Frontend from FlashInfer's
# flashinfer/gemm/batch_invariant/blocks/blockscale_fp8.py (e2d163fb3d46).
"""Groupwise FP8 GEMM. Fold 128-element K blocks with FP32 scale multiplication and FMA."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils.blackwell_helpers import get_smem_store_op, get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from cudnn.frost.tile_dsl.barrier import named_barrier_sync_unaligned

from .arithmetic import CuteF32
from .arithmetic import Knobs, MainloopBlock
from .arithmetic import BLOCK_K, SCALE_BLOCK_N, EpilogueChain
from .epilogue import apply_chain_pre_cvt
from .mainloop import mma_fold
from .operands import (
    ACC_DTYPE,
    BAR_EPILOGUE,
    BAR_TMEM_ALLOC,
    IN_DT,
    OUT_DT,
    StaticPlan,
    TmaSetup,
)
from .scheduler import SchedParams, TileScheduler, build_grid

IN_FMT = "e4m3_bs"
TMEM_COLS = 512  # the accumulator ring's budget: the dense body's 512 TMEM columns
MAX_ACC_STAGES = 8  # the ring's depth is capped: beyond it the hand-off barriers cost more than the latency they hide (empirical)
NUM_PROMO_GROUPS = 2  # the promotion warps: two warpgroups, each owning every other epilogue subtile (a column half) of the tile
NUM_PROMO_WARPS = 4 * NUM_PROMO_GROUPS
PROMO_WARPS = tuple(range(NUM_PROMO_WARPS))
MMA_WARP_BS = NUM_PROMO_WARPS
TMA_WARP_BS = NUM_PROMO_WARPS + 1
NUM_WARPS_BS = -(-(NUM_PROMO_WARPS + 2) // 4) * 4  # whole warpgroups (setmaxnreg is a warpgroup instruction); the last warps idle
THREADS_BS = 32 * NUM_WARPS_BS
# register budgets per thread (setmaxnreg): the promotion warps hold the running sum and two blocks' partials, the rest little;
# the sum over the warps stays within the register file
PROMO_REGS = 224
UNIFORM_REGS = 56
BAR_GROUP = (
    BAR_EPILOGUE,
    BAR_EPILOGUE + 2,
)  # one named barrier per promotion warpgroup (the C stage hand-off to its TMA store)
BAR_PROMO_JOIN = BAR_GROUP[1] + 1  # both promotion warpgroups once, after their last TMEM loads, before the TMEM free
NUM_B_BLOCKS = 2  # B block scales per ring stage: a tile up to 256 wide spans two 128-column weight scale blocks (one per promotion group)
PROMO_BATCH = 2  # subtiles of a group's half loaded from TMEM before their fmas: the register budget's bound at the 256-wide tile
NUM_SCALE_STAGES = 16  # the scale ring's depth in blocks: blocks whose scales wait in smem ahead of the promotion; deeper than the operand
# ring plus the accumulator ring, so the operand stage is always the producer's binding wait (empirical)
SCALE_GROUP_NATIVE_K = 4  # the K-native staging (a few rows, K-layout scales): blocks per ring stage, a row's four block scales one 16-byte copy


def scale_ring_bytes(knobs: Knobs) -> int:
    """Smem of the scale ring: per stage the CTA tile's A row scales and one B block scale, fp32."""
    return NUM_SCALE_STAGES * (knobs.cta_tile_mn[0] + NUM_B_BLOCKS) * (ACC_DTYPE.width // 8)  # the same bytes in either staging


def acc_stages_for(knobs: Knobs) -> int:
    """The accumulator ring's depth for the record: one stage per K block in flight between the MMA warp and the promotion, as many
    as the TMEM holds for the tile (`TMEM_COLS // tile_n`, at most `MAX_ACC_STAGES`): 4 at tile N 128, 8 below. A placement: the
    stage a block lands in never enters the fold (Lean `promoteLoop`: the stage index is erased)."""
    return max(2, min(MAX_ACC_STAGES, TMEM_COLS // knobs.tile_n))


@cute.jit
def blockscale_mma_loop(
    lowering,
    num_acc_stage: cutlass.Constexpr,
    k_tile_cnt: cutlass.Int32,
    is_leader_cta,
    tile_sched,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
):
    """Per tile, per k-tile (blocks ascending): the dense ``mma_fold`` of ONE k-tile -- the lowering's ``inner`` steps from a reset
    accumulator into the next accumulator stage, committed to the epilogue -- so every block's partial is the spec's ``P_kb``."""
    assert cute.size(tCrA, mode=[2]) == lowering.inner, "one smem stage must hold exactly the lowering's inner steps (one scale block)"
    acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, num_acc_stage)
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        for _kb in range(k_tile_cnt):  # blocks ascending, each a fold from +0 into the next stage
            ab_consumer, acc_state = mma_fold(
                lowering,
                1,
                is_leader_cta,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                acc_state,
                tCrA,
                tCrB,
                tCtAcc_base,
            )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)


def _elem(t: cute.Tensor, coord):
    """A one-element view of ``t`` at ``coord`` (the copy atoms move tensors)."""
    return cute.make_tensor(t.iterator + cute.crd2idx(coord, t.layout), cute.make_layout((1,)))


@cute.jit
def blockscale_producer_loop(
    atom_thr_size: cutlass.Constexpr,
    k_tile_cnt: cutlass.Int32,
    tile_sched,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tAgA: cute.Tensor,
    tBgB: cute.Tensor,
    tAsA: cute.Tensor,
    tBsB: cute.Tensor,
    a_mcast_mask,
    b_mcast_mask,
    scale_pipeline,
    mSFA: cute.Tensor,
    mSFB: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    tCcC: cute.Tensor,
    native_k: cutlass.Constexpr,
):
    """Stream operands and scales in ascending K-block order. In each scale ring stage the warp copies
    one FP32 scale per A row and one per 128-column B block, clamping edge coordinates to the last valid index."""
    lane = cute.arch.lane_idx()
    M = mSFA.shape[0]
    NB = mSFB.shape[0]
    rows_per_lane = cute.size(sSFA, mode=[0]) // 32
    atom_g2s = cute.make_copy_atom(cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=ACC_DTYPE.width)
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        NUM_SCALE_STAGES,
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        coord = work_tile.tile_idx
        mnl = (coord[0] // atom_thr_size, coord[1], coord[2])
        tAgA_slice = tAgA[(None, mnl[0], None, coord[2])]  # ((atom_v, rest_v), RestK): this tile's A rows, all k-tiles
        tBgB_slice = tBgB[(None, coord[1], None, coord[2])]
        c0 = tCcC[(None, None, None, mnl[0], mnl[1], mnl[2])][0]  # this CTA tile's first output coordinate (row, column)
        nb = cutlass.min(c0[1] // SCALE_BLOCK_N, NB - 1)
        ab_producer.reset()
        peek_empty = ab_producer.try_acquire()
        for _ in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(peek_empty)  # operand slot empty; the leader arms the tx barrier
            cute.copy(
                tma_atom_a,
                tAgA_slice[(None, handle.count)],
                tAsA[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=a_mcast_mask,
            )
            cute.copy(
                tma_atom_b,
                tBgB_slice[(None, handle.count)],
                tBsB[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=b_mcast_mask,
            )
            scale_pipeline.producer_acquire(scale_state)  # ring stage free (the promotion released it)
            for r in cutlass.range_constexpr(rows_per_lane):
                i = r * 32 + lane
                row = cutlass.min(c0[0] + i, M - 1)
                cute.copy(
                    atom_g2s,
                    _elem(mSFA, (row, handle.count)),
                    _elem(sSFA, (i, scale_state.index)),
                )
            if lane < NUM_B_BLOCKS:  # the tile's B block scales, one thread each (clamped to the tensor)
                nb_l = cutlass.min(nb + lane, NB - 1)
                cute.copy(
                    atom_g2s,
                    _elem(mSFB, (nb_l, handle.count)),
                    _elem(sSFB, (lane, scale_state.index)),
                )
            scale_pipeline.producer_commit(scale_state)  # arrives when this thread's copies have landed
            scale_state.advance()
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = ab_producer.try_acquire()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    ab_producer.tail()
    scale_pipeline.producer_tail(scale_state)


@cute.jit
def blockscale_epilogue_loop(
    st,
    chain,
    atom_thr_size: cutlass.Constexpr,
    tidx: cutlass.Int32,
    warp_idx: cutlass.Int32,
    tile_sched,
    tma_atom_c: cute.CopyAtom,
    tCtAcc_base: cute.Tensor,
    sC: cute.Tensor,
    tCgC_base: cute.Tensor,
    tCcC_base: cute.Tensor,
    epi_tile: cute.Tile,
    acc_pipeline,
    scale_pipeline,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    k_tile_cnt: cutlass.Int32,
    native_k: cutlass.Constexpr,
):
    """The promotion and the store (module docstring) by two warpgroups, each owning one contiguous column half of the tile (both
    read the same TMEM lane quadrants -- the 32x32b atom -- at different columns; a tile of one epilogue subtile is group 0's
    alone). Per block each thread loads the group's subtiles of the block partial back to back (the register budget of a half
    leaves the compiler free to issue them early), folds them with packed fmas and releases the stage. The group index is a
    runtime value that enters addresses only (TMEM columns, the C smem stage, the subtile stored), never the fold. ``sSFA``
    ``(cta_tile_m, STAGE)`` and ``sSFB`` ``(STAGE,)`` are the scale ring's smem; ``tCcC_base`` the output coordinates partitioned
    like C (each thread finds its row within the CTA tile from them)."""
    g = warp_idx // 4  # this warp's promotion group
    tidx_g = tidx % 128  # the thread within its group
    # ((ATOM_M, MMA_M), (ATOM_N, MMA_N), ...)
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCcC = transform_partitioned_tensor_layout(tCcC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    # the epilogue subtiles (the TMA store's unit): the 128-thread tiled copy of one subtile, shared by both groups
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk,
        st.c_layout,
        st.c_dtype,
        ACC_DTYPE,
        epi_tile,
        st.use_2cta_instrs,
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
    thr_t2r = tiled_copy_t2r.get_slice(tidx_g)
    tCgC_epi = cute.flat_divide(tCgC, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, RestM, RestN, RestL)
    tTR_gC = thr_t2r.partition_D(tCgC_epi)
    tCcC_epi = cute.flat_divide(tCcC, epi_tile)
    tTR_cC = thr_t2r.partition_D(tCcC_epi)  # the output coordinates in the accumulator's partition
    frag_shape = tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape
    tTR_rP = cute.make_rmem_tensor(frag_shape, ACC_DTYPE)  # one subtile's values on the way to the store
    tTR_rC = cute.make_rmem_tensor(frag_shape, st.c_dtype)
    V = cute.size(tTR_rP)
    sub_m, sub_n = cute.size(tAcc_epi, mode=[2]), cute.size(tAcc_epi, mode=[3])
    assert sub_m == 1, "the epilogue subtiles run along N only (the CTA tile's 128 rows are one subtile row)"
    subtile_cnt = sub_n
    groups = NUM_PROMO_GROUPS if subtile_cnt >= NUM_PROMO_GROUPS else 1  # a one-subtile tile is group 0's alone
    per_group = subtile_cnt // groups  # subtiles per group: contiguous column halves
    assert per_group * groups == subtile_cnt, (subtile_cnt, groups)
    cols_g = per_group * cute.size(epi_tile[1])  # this group's columns
    assert cols_g % 2 == 0, cols_g  # the promotion packs element pairs
    tTR_tAcc_base = thr_t2r.partition_S(tAcc_epi)  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
    # the static shape of the fragment: every element of a fragment, and of every subtile of the tile, lies on ONE output row (the
    # 32x32b TMEM atom of the 128-row CTA tile), so one A scale serves the thread for a block; the columns of a tile lie in one
    # 128-column weight scale block (tile N divides 128), so one B scale does too
    offs0 = tTR_cC[(None, None, None, 0, 0, 0, 0, 0)]
    offs = [offs0.layout(e) for e in range(V)]
    assert all(o[0] == offs[0][0] for o in offs), "the fragment spans more than one output row"
    for s in range(subtile_cnt):
        assert tTR_cC[(None, None, None, 0, s, 0, 0, 0)].layout(0)[0] == offs[0][0], "a subtile changes the output row"
    assert cols_g <= SCALE_BLOCK_N and SCALE_BLOCK_N % cols_g == 0, cols_g  # a group's columns lie in one weight scale block
    assert subtile_cnt * cute.size(epi_tile[1]) <= NUM_B_BLOCKS * SCALE_BLOCK_N
    blk_per_group = cols_g // SCALE_BLOCK_N  # 1 at the 256-wide tile (group g = block g), else 0
    assert per_group % PROMO_BATCH == 0 or per_group < PROMO_BATCH, (
        per_group,
        PROMO_BATCH,
    )
    batch = min(per_group, PROMO_BATCH)
    VW = per_group * V  # this group's elements per thread
    bufs = [cute.make_rmem_tensor(frag_shape, ACC_DTYPE) for _ in range(batch)]  # the block partial of a batch of the group's subtiles
    tTR_rSum = cute.make_rmem_tensor((VW,), ACC_DTYPE)  # the running sum of this group's half, in registers
    # registers -> smem -> global: each group stores through its own C stage and its own TMA-store pipeline
    copy_atom_r2s = get_smem_store_op(st.c_layout, st.c_dtype, ACC_DTYPE, tiled_copy_t2r)
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
    thr_r2s = tiled_copy_r2s.get_slice(tidx_g)
    tRS_sC = thr_r2s.partition_D(sC)
    tRS_rC = tiled_copy_r2s.retile(tTR_rC)
    bSG_sC, bSG_gC_all = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )
    assert st.num_c_stage >= NUM_PROMO_GROUPS, st.num_c_stage
    acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, st.num_acc_stage)
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        NUM_SCALE_STAGES,
    )
    c_pipeline0 = pipeline.PipelineTmaStore.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
    )
    c_pipeline1 = pipeline.PipelineTmaStore.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
    )
    gsel = g if groups == 2 else cutlass.Int32(0)
    active = (g == 0) if groups == 1 else (g >= 0)  # a one-subtile tile: group 1 has no columns
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        coord = work_tile.tile_idx
        mnl = (coord[0] // atom_thr_size, coord[1], coord[2])
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        bSG_gC = bSG_gC_all[(None, None, None, mnl[0], mnl[1], mnl[2])]
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        # this thread's row within the CTA tile: the slot of its A scale in a ring stage
        irow = tTR_cC[(None, None, None, 0, 0, mnl[0], mnl[1], mnl[2])][0][0] - tCcC[(None, None, mnl[0], mnl[1], mnl[2])][0][0]
        for i in cutlass.range_constexpr(VW):
            tTR_rSum[i] = CuteF32.const(0.0)  # acc_0 = +0
        for _kb in range(k_tile_cnt):  # blocks ascending: the block fold digit
            scale_pipeline.consumer_wait(scale_state)  # block kb's scales are in the ring
            scale = CuteF32.mul(
                sSFA[(irow, scale_state.index)],
                sSFB[(gsel * blk_per_group, scale_state.index)],
            )  # mul.rn(s_a[m, kb], s_b[n // 128, kb]), once
            scale_pipeline.consumer_release(scale_state)  # read: the stage may take a later block
            scale_state.advance()
            acc_pipeline.consumer_wait(acc_state)  # block kb's fold is complete
            if active:
                tTR_tAcc = cute.group_modes(
                    tTR_tAcc_base[(None, None, None, None, None, acc_state.index)],
                    3,
                    5,
                )
                for b0 in cutlass.range_constexpr(0, per_group, batch):  # the group's subtiles in batches: a batch's loads first
                    for j in cutlass.range_constexpr(batch):
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, gsel * per_group + b0 + j)],
                            bufs[j],
                        )
                    for j in cutlass.range_constexpr(batch):
                        k = b0 + j
                        for e in cutlass.range_constexpr(0, V, 2):
                            # acc = fma(scale, P_kb, acc) for two elements per fma.rn.f32x2: each component the same correctly rounded fma on
                            # the same operands (a packing placement, as the attention body's)
                            r0, r1 = CuteF32.fma2(
                                (scale, scale),
                                (bufs[j][e], bufs[j][e + 1]),
                                (tTR_rSum[k * V + e], tTR_rSum[k * V + e + 1]),
                            )
                            tTR_rSum[k * V + e] = r0
                            tTR_rSum[k * V + e + 1] = r1
            cute.arch.fence_view_async_tmem_load()  # this warp's reads of the partial are complete
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_state)  # one arrive per warp: the stage is free once all arrived
            acc_state.advance()
        # the chain (one cvt.rn) and the store of this group's half of the tile's fold, subtile by subtile
        if active:
            for k in cutlass.range_constexpr(per_group):
                for e in cutlass.range_constexpr(V):
                    tTR_rP[e] = tTR_rSum[k * V + e]
                apply_chain_pre_cvt(chain, tTR_rP, None)  # the chain's fp32 ops (none: the scales are in the fold)
                acc_vec = tiled_copy_r2s.retile(tTR_rP).load()
                tRS_rC.store(acc_vec.to(st.c_dtype))  # the chain's cvt.rn <out>, exactly once
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, gsel)])
                cute.arch.fence_proxy("async.shared", space="cta")
                if gsel == 0:
                    named_barrier_sync_unaligned(BAR_GROUP[0], 128)
                else:
                    named_barrier_sync_unaligned(BAR_GROUP[1], 128)
                if warp_idx == 4 * gsel:  # the group's leader warp issues the TMA store
                    cute.copy(
                        tma_atom_c,
                        bSG_sC[(None, gsel)],
                        bSG_gC[(None, gsel * per_group + k)],
                    )
                    if gsel == 0:
                        c_pipeline0.producer_commit()
                        c_pipeline0.producer_acquire()
                    else:
                        c_pipeline1.producer_commit()
                        c_pipeline1.producer_acquire()
                if gsel == 0:
                    named_barrier_sync_unaligned(BAR_GROUP[0], 128)
                else:
                    named_barrier_sync_unaligned(BAR_GROUP[1], 128)
    if gsel == 0:
        c_pipeline0.producer_tail()
    else:
        c_pipeline1.producer_tail()


class BlockScaleFp8Gemm:
    """``C[m, n] = cvt( fold_{kb} fma(mul(SFA[m, kb], SFB[n // 128, kb]), P_kb, acc) )`` for the knobs; M, N, K dynamic, K a whole
    number of 128-element blocks. Operands: A ``(M, K, 1)`` and B ``(N, K, 1)`` e4m3 with K contiguous, SFA ``(M, K / 128)`` and
    SFB ``(ceil(N / 128), K / 128)`` fp32 with any strides, C ``(M, N, 1)`` row-major in the output format (the chain is the
    conversion alone)."""

    def __init__(self, knobs: Knobs, out_fmt: str, native_k: bool = False):
        self.knobs = knobs
        self.native_k = native_k  # the K-native scale staging (the API's choice at a few rows in the K layout)
        self.cluster_shape_mn = knobs.cluster_shape_mn
        self.chain = EpilogueChain.for_class(out_fmt, False)
        assert not self.chain.scaled and self.chain.out_fmt == out_fmt, self.chain.describe()

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        assert a.element_type is IN_DT[IN_FMT] and b.element_type is IN_DT[IN_FMT], (
            a.element_type,
            b.element_type,
        )
        assert OUT_DT[self.chain.out_fmt] is c.element_type, (
            self.chain.describe(),
            c.element_type,
        )
        assert sfa.element_type is ACC_DTYPE and sfb.element_type is ACC_DTYPE
        st = StaticPlan(
            self.knobs,
            a.element_type,
            b.element_type,
            c.element_type,
            LayoutEnum.from_tensor(a).mma_major_mode(),
            LayoutEnum.from_tensor(b).mma_major_mode(),
            LayoutEnum.from_tensor(c),
            in_fmt=IN_FMT,
            acc_stages=acc_stages_for(self.knobs),
            reserve_smem_bytes=scale_ring_bytes(self.knobs),
        )
        self.st = st
        self.block = MainloopBlock(IN_FMT)
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block == BLOCK_K, (
            st.mma_tiler,
            self.block.k_block,
        )  # one stage = one scale block
        tma = TmaSetup(st, a, b, c)
        self.num_tma_load_bytes = tma.tx_bytes
        params, grid = build_grid(st, c, max_active_clusters)
        self.kernel(
            st.tiled_mma,
            tma.atom_a,
            tma.tensor_a,
            tma.atom_b,
            tma.tensor_b,
            tma.atom_c,
            tma.tensor_c,
            sfa,
            sfb,
            st.cluster_layout_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.c_smem_layout_staged,
            st.epi_tile,
            params,
        ).launch(
            grid=grid,
            block=[THREADS_BS, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFA: cute.Tensor,
        mSFB: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged,
        epi_tile: cute.Tile,
        tile_sched_params: SchedParams,
    ):
        st = self.st
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP_BS:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        use_2cta_instrs = atom_thr_size == 2
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % atom_thr_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # --- scaffolding: barriers, pipelines, TMEM allocator (the dense body's)
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_acc_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, NUM_SCALE_STAGES * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=st.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=st.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, len(PROMO_WARPS) * (2 if use_2cta_instrs else 1)),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=NUM_SCALE_STAGES // (SCALE_GROUP_NATIVE_K if self.native_k else 1),
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(PROMO_WARPS)),
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=BAR_TMEM_ALLOC, num_threads=THREADS_BS)
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=PROMO_WARPS[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # --- operand staging: smem rings, multicast masks, global tiles and partitions
        sA = smem.allocate_tensor(
            element_type=st.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=st.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # the scale ring: a stage holds the CTA tile's A row scales and the tile's B block scale
        tm_cta = st.cta_tile_shape_mnk[0]
        sSFA = smem.allocate_tensor(
            element_type=ACC_DTYPE,
            layout=cute.make_layout((tm_cta, NUM_SCALE_STAGES)),
            byte_alignment=16,
        )
        sSFB = smem.allocate_tensor(
            element_type=ACC_DTYPE,
            layout=cute.make_layout((NUM_B_BLOCKS, NUM_SCALE_STAGES)),
            byte_alignment=16,
        )
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(st.is_a_mcast or st.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None))  # (bM, bK, RestM, RestK, RestL)
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None))  # (bN, bK, RestN, RestK, RestL)
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(st.mma_tiler, (None, None, 0)), (None, None, None))  # (bM, bN, RestM, RestN, RestL)
        # the output coordinates, tiled and partitioned like C: the epilogue reads each thread's row and the tile's column block
        cC_mnl = cute.local_tile(
            cute.make_identity_tensor(mC_mnl.shape),
            cute.slice_(st.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])  # = K / BLOCK_K: the block count
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCcC = thr_mma.partition_C(cC_mnl)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, MMA_M, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)  # (MMA, MMA_N, MMA_K, STAGE)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, st.num_acc_stage))
        # the scheduler (its first tile decoded here) before the barrier init completes: the decode overlaps the wait
        tile_sched = TileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        tmem.allocate(st.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)

        # --- roles
        if warp_idx == TMA_WARP_BS:
            cute.arch.setmaxregister_decrease(UNIFORM_REGS)
            blockscale_producer_loop(
                atom_thr_size,
                k_tile_cnt,
                tile_sched,
                ab_producer,
                tma_atom_a,
                tma_atom_b,
                tAgA,
                tBgB,
                tAsA,
                tBsB,
                a_full_mcast_mask,
                b_full_mcast_mask,
                scale_pipeline,
                mSFA,
                mSFB,
                sSFA,
                sSFB,
                tCcC,
                self.native_k,
            )
        if warp_idx == MMA_WARP_BS:
            cute.arch.setmaxregister_decrease(UNIFORM_REGS)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            blockscale_mma_loop(
                self.lowering,
                st.num_acc_stage,
                k_tile_cnt,
                is_leader_cta,
                tile_sched,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                tCrA,
                tCrB,
                tCtAcc_base,
            )
        sC = smem.allocate_tensor(
            element_type=st.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )
        if warp_idx < NUM_PROMO_WARPS:
            cute.arch.setmaxregister_increase(PROMO_REGS)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            blockscale_epilogue_loop(
                st,
                self.chain,
                atom_thr_size,
                tidx,
                warp_idx,
                tile_sched,
                tma_atom_c,
                tCtAcc_base,
                sC,
                tCgC,
                tCcC,
                epi_tile,
                acc_pipeline,
                scale_pipeline,
                sSFA,
                sSFB,
                k_tile_cnt,
                self.native_k,
            )
            # the allocator warp alone frees the columns (`TmemAllocator.free`): the other warpgroup's last TMEM loads must be
            # complete first; its store barriers are its own, so the groups join here (the dense bodies' one epilogue group is
            # joined by its store barrier already)
            named_barrier_sync_unaligned(BAR_PROMO_JOIN, 32 * NUM_PROMO_WARPS)
        if warp_idx > TMA_WARP_BS:
            cute.arch.setmaxregister_decrease(UNIFORM_REGS)  # the warps padding the last warpgroup: no role

        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
