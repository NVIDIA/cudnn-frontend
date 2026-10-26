# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from NVIDIA-JerryChen/Block-Sparse-Attention, commit f953680ff93a2ef2321ad7dc4291451eea4a9c3c.
# Shared transfer, scale, correction and epilogue helpers for FP8 blk128.
# The selected launch and compute pipeline lives in bsa_fwd_sm100_fp8_fused_pv.
import types
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Optional, TypeAlias, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as cutlass_pipeline
from cutlass import Boolean, Float32, Int32, Uint32, const_expr, pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic

from cudnn.block_sparse_attention.csrc.utils import copy_utils, kernel_utils as utils, layout_utils, mma_sm100_desc as sm100_desc
from cudnn.block_sparse_attention.csrc.utils.tcgen05_mma_helpers import i64_to_i32x2
from cudnn.block_sparse_attention.csrc.utils.block_sparse_tile_scheduler import TileSchedulerProtocol


class BlockSparseAttnForwardFp8Sm100Blk128:
    """Shared helpers; the optimized subclass owns configuration and launch."""

    tile_m = 128
    tile_n = 128

    @cute.jit
    def clc_scheduler_warp(
        self,
        clc_pipeline: cutlass_pipeline.PipelineClcFetchAsync,
        tile_scheduler: "TileSchedulerProtocol",
    ):
        """Runs on leader CTA's scheduler warp — produces CLC work queries."""
        clc_producer_state = cutlass_pipeline.make_pipeline_state(cutlass_pipeline.PipelineUserType.Producer, self.sched_stages)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            clc_pipeline.producer_acquire(clc_producer_state)
            mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
            tile_scheduler.advance_to_next_work(mbarrier_addr=mbarrier_addr)
            clc_producer_state.advance()

            work_tile = tile_scheduler.consumer_advance()
        clc_pipeline.producer_tail(clc_producer_state)

    @cute.jit
    def _split_block_range(
        self,
        mBlockNums: Optional[cute.Tensor],
        block_sparse_num: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        split_idx: Int32,
    ):
        """Partition the live sparse-index prefix without host metadata work."""
        count = mBlockNums[batch_idx, head_idx, m_block] if const_expr(mBlockNums is not None) else block_sparse_num
        begin = count * split_idx // self.num_splits
        end = count * (split_idx + 1) // self.num_splits
        return begin, end - begin

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: "BlockInfo",
        tile_scheduler: "TileSchedulerProtocol",
        mBlockIndex: cute.Tensor,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            mQ_cur = mQ[None, None, None, batch_idx][None, None, head_idx]
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))  # (128, 128)
            gQ = layout_utils.select(cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1])  # (128, 128, 1)

            head_idx_kv = head_idx
            mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ)

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )

            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
            )

            # n_block(i): maps logical index i to actual KV block index via q2k_block_index
            # When mBlockNums is provided, raw count may be odd; round up to even for kernel loops.
            # max_i clamps phantom block indices to the last valid entry.
            if const_expr(self.num_splits > 1):
                split_begin, raw_block_count = self._split_block_range(mBlockNums, block_sparse_num, batch_idx, head_idx, m_block, split_idx)
                process_tile = raw_block_count > Int32(0)
                block_iter_count = (raw_block_count + 1) & ~1
                mBlockIndex_split = cute.make_tensor(mBlockIndex.iterator + split_begin, mBlockIndex.layout)
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex_split,
                    batch_idx,
                    head_idx,
                    m_block,
                    max_i=cutlass.max(raw_block_count - 1, Int32(0)),
                )
            elif const_expr(mBlockNums is not None):
                raw_block_count = mBlockNums[batch_idx, head_idx, m_block]
                process_tile = raw_block_count > Int32(0) if const_expr(self.allow_empty_block_nums) else True
                block_iter_count = (raw_block_count + 1) & ~1
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex,
                    batch_idx,
                    head_idx,
                    m_block,
                    max_i=cutlass.max(raw_block_count - 1, Int32(0)),
                )
            else:
                process_tile = True
                block_iter_count = (block_sparse_num + 1) & ~1
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex,
                    batch_idx,
                    head_idx,
                    m_block,
                    max_i=cutlass.max(block_sparse_num - 1, Int32(0)),
                )

            if process_tile:
                load_K(
                    block=n_block(block_iter_count - 1),
                    producer_state=kv_producer_state,
                    page_idx=None,
                )  # K0
                if const_expr(len(self.load_warp_ids) == 1) or warp_idx == self.load_warp_ids[0]:
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(0)
                    load_Q_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
                kv_producer_state.advance()

                # q_stage=1 intra-warp overlap
                # Load order: K[N-1], Q, K[N-2], {V[N-1-i], K[N-3-i]}x(N-2), V[1], V[0]
                block_loop_count = block_iter_count - 2
                q_producer_phase ^= 1

                # Prologue: K[N-2]
                load_K(
                    block=n_block(block_iter_count - 2),
                    producer_state=kv_producer_state,
                    page_idx=None,
                )
                kv_producer_state.advance()

                # Flat main loop: N-2 iterations, each loads V then K
                for i in cutlass.range(block_loop_count, unroll=1):
                    # V[N-1-i]: V for the block whose S was computed earlier
                    load_V(
                        block=n_block(block_iter_count - 1 - i),
                        producer_state=kv_producer_state,
                        page_idx=None,
                    )
                    kv_producer_state.advance()
                    # K[N-3-i]: K for the next QK GEMM
                    load_K(
                        block=n_block(block_iter_count - 3 - i),
                        producer_state=kv_producer_state,
                        page_idx=None,
                    )
                    kv_producer_state.advance()

                # Epilogue: last 2 V loads
                load_V(block=n_block(1), producer_state=kv_producer_state, page_idx=None)
                kv_producer_state.advance()
                load_V(block=n_block(0), producer_state=kv_producer_state, page_idx=None)
                kv_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.consumer_advance()
            # End of persistent scheduler loop

        pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail
        if const_expr(len(self.load_warp_ids) == 1) or warp_idx == self.load_warp_ids[0]:
            pipeline_q.producer_acquire_w_index_phase(self.q_stage - 1, q_producer_phase)

    @cute.jit
    def load_k_scale(
        self,
        mKScale: cute.Tensor,
        key_block: Int32,
        batch_idx: Int32,
        head_idx: Int32,
    ) -> Float32:
        lane = cute.arch.thread_idx()[0] % 32
        value = Float32(0.0)
        if lane < 8:
            scale_idx = cutlass.min(key_block * 8 + lane, mKScale.shape[2] - 1)
            value = mKScale[batch_idx, head_idx, scale_idx]
        return value

    # for both softmax0 and softmax1 warp group

    @cute.jit
    def valid_block_size(
        self,
        mBlockSizes: Optional[cute.Tensor],
        batch_idx: Int32,
        head_idx: Int32,
        key_block: Int32,
        seqlen_k: Int32,
    ) -> Int32:
        size = cutlass.min(cutlass.max(seqlen_k - key_block * 128, Int32(0)), Int32(128))
        if const_expr(self.has_block_sizes):
            if const_expr(self.block_sizes_mode == 1):
                size = cutlass.min(size, mBlockSizes[key_block])
            elif const_expr(self.block_sizes_mode == 2):
                size = cutlass.min(size, mBlockSizes[batch_idx, key_block])
            else:
                size = cutlass.min(size, mBlockSizes[batch_idx, head_idx, key_block])
        return size

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = self.correction_tile_size
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)), self.pv_acc_dtype)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_rmem_tensor((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2((tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale))
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue_combine(
        self,
        thr_mma: cute.ThrMma,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        tidx: Int32,
        scale0: Float32,
        scale1: Float32,
        sO: cute.Tensor,
        mVScale: cute.Tensor,
        head_idx: Int32,
    ):
        """Combine two partial O accumulators (O0, O1) from intra-warp overlap and write to smem/gmem.

        For q_stage=1 intra-warp overlap, O0 and O1 accumulate results from even and odd
        n_blocks respectively. This method reads both from TMEM, applies their respective
        scales (derived from combined softmax stats), adds them, and writes the result.
        """
        corr_tile_size = self.epilogue_tile_size
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO0_i = cute.logical_divide(tOtO0, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOtO1_i = cute.logical_divide(tOtO1, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO0_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load)
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO0_t2r = thr_tmem_load.partition_S(tOtO0_i[(None, None), None])
        tOtO1_t2r = thr_tmem_load.partition_S(tOtO1_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        # Each warp shares the same channel scales. Coalesced loads followed
        # by register shuffles avoid one scalar global load per output value.
        # SM103's larger correction fragments leave less register headroom,
        # so load its scales one output fragment at a time.
        lane = tidx % 32
        if const_expr(self.prefetch_all_v_scales):
            tVScales = cute.make_rmem_tensor((4,), Float32)
            for chunk in cutlass.range_constexpr(4):
                tVScales[chunk] = mVScale[head_idx, chunk * 32 + lane]
        for i in cutlass.range(self.head_dim_v_padded // corr_tile_size, unroll_full=True):
            if const_expr(not self.prefetch_all_v_scales):
                v_scale = mVScale[head_idx, i * corr_tile_size + lane]
            tOtO0_t2r_i = tOtO0_t2r[None, 0, 0, i]
            tOtO1_t2r_i = tOtO1_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            frg_shape = tOcO_t2r[None, 0, 0, i].shape
            tOrO0_frg = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            tOrO1_frg = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            # When both scales are 0 (empty tile), skip tmem reads to avoid 0*NaN=NaN.
            is_zero_output = scale0 == Float32(0.0) and scale1 == Float32(0.0)
            if not is_zero_output:
                cute.copy(tiled_tmem_load, tOtO0_t2r_i, tOrO0_frg)
                cute.copy(tiled_tmem_load, tOtO1_t2r_i, tOrO1_frg)
                # Combined: O = O0 * scale0 + O1 * scale1
                for j in cutlass.range(0, cute.size(tOrO0_frg), 2, unroll_full=True):
                    o0_a, o0_b = cute.arch.mul_packed_f32x2((tOrO0_frg[j], tOrO0_frg[j + 1]), (scale0, scale0))
                    o1_a, o1_b = cute.arch.mul_packed_f32x2((tOrO1_frg[j], tOrO1_frg[j + 1]), (scale1, scale1))
                    tOrO0_frg[j], tOrO0_frg[j + 1] = cute.arch.add_packed_f32x2((o0_a, o0_b), (o1_a, o1_b))
            else:
                tOrO0_frg.fill(Float32(0.0))
            for j in cutlass.range_constexpr(cute.size(tOrO0_frg)):
                channel = tOcO_t2r[None, 0, 0, i][j][1]
                if const_expr(self.prefetch_all_v_scales):
                    value_scale = cute.arch.shuffle_sync(tVScales[channel // 32], channel % 32)
                else:
                    value_scale = cute.arch.shuffle_sync(v_scale, channel % 32)
                tOrO0_frg[j] *= value_scale
            copy_utils.cvt_copy(tiled_smem_store, tOrO0_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        pipeline_o_epi: pipeline.PipelineAsync,
        tile_scheduler: "TileSchedulerProtocol",
    ):
        epi_consumer_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            output_head = head_idx
            if const_expr(self.num_splits > 1):
                output_head += split_idx * (mO.shape[2] // self.num_splits)
            mO_cur = mO[None, None, None, batch_idx][None, None, output_head]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))  # (128, 128)
            gO = layout_utils.select(cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1])  # (128, 128, 1)
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0],))[None, 0, None, None]

            store_O, _, _ = copy_utils.tma_get_copy_fn(tma_atom_O, 0, cute.make_layout(1), sO, gO)
            for stage in cutlass.range(self.q_stage, unroll_full=True):
                # wait from corr, issue tma store on smem
                # 1. wait for O0 final
                pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                # 2. copy O0 to gmem
                store_O(src_idx=stage, dst_idx=stage)
                cute.arch.cp_async_bulk_commit_group()
            for stage in cutlass.range_constexpr(self.q_stage):
                # Ensure O0 buffer is ready to be released
                cute.arch.cp_async_bulk_wait_group(self.q_stage - 1 - stage, read=True)
                pipeline_o_epi.consumer_release_w_index(stage)

            epi_consumer_phase ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()

    @cute.jit
    def load_KV(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        extra_tx_count: Optional[Int32] = None,
    ):
        assert K_or_V in ("K", "V")
        stage = producer_state.index
        extra_tx_count_kv = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
        extra_tx_count = extra_tx_count_kv + (extra_tx_count if extra_tx_count is not None else 0)
        extra_kwargs = {"extra_tx_count": extra_tx_count}
        pipeline_kv.producer_acquire(producer_state, **extra_kwargs)

        tXsX_cur = tXsX[None, stage]
        tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
        cute.copy(
            tma_atom,
            tXgX_cur,
            tXsX_cur,
            tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
        )


# =============================================================================
# Sequence Metadata
# =============================================================================
@dataclass(frozen=True)
class SeqlenInfoQK:
    seqlen_q: Int32
    seqlen_k: Int32

    @staticmethod
    def create(
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
    ):
        return SeqlenInfoQK(
            seqlen_q_static,
            seqlen_k_static,
        )


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]

    @cute.jit
    def get_n_block_idx(
        self,
        block_index: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        i: Int32,
        max_i: Optional[Int32] = None,
    ) -> Int32:
        idx = cutlass.min(i, max_i) if const_expr(max_i is not None) else i
        return block_index[batch_idx, head_idx, m_block, idx]


@cute.jit
def declare_ptx_smem_desc(
    smem_desc_start_a: Int32,  # If TS, then this is the tmem start address for A
    smem_desc_base_a: Optional[int],
    tCrA_layout: cute.Layout,
    var_name_prefix: str = "smem_desc",
) -> None:
    is_ts = const_expr(smem_desc_base_a is None)
    num_k_tile = cute.size(tCrA_layout.shape[2])
    smem_desc_base_a_lo, smem_desc_a_hi = None, None
    if const_expr(not is_ts):
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
    tCrA_layout = (
        tCrA_layout
        if const_expr(not is_ts)
        # currently hard-coding the width to 16
        else cute.recast_layout(32, 16, tCrA_layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(num_k_tile)]
    smem_desc_start_a_lo = None
    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | smem_desc_start_a)
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value()],
            f".reg .b32 {var_name_prefix}_lo;\n\t"
            f".reg .b64 {var_name_prefix}_<{num_k_tile}>;\n\t"
            f"mov.b64 {var_name_prefix}_0, {{$0, {hex(smem_desc_a_hi)}}};\n\t"
            + "".join(
                (
                    f"add.s32 {var_name_prefix}_lo, $0, {hex(offset_a[k])};\n\t"
                    f"mov.b64 {var_name_prefix}_{k}, {{{var_name_prefix}_lo, {hex(smem_desc_a_hi)}}};\n\t"
                )
                for k in range(1, num_k_tile)
            ),
            "r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp, var_name: str = "idesc") -> None:
    idesc = const_expr(sm100_desc.mma_op_to_idesc(op))
    llvm.inline_asm(
        None,
        [],
        f".reg .b32 {var_name};\n\t"  # noqa
        f"mov.b32 {var_name}, {hex(idesc)};\n\t",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_precomputed_varname(
    acc_tmem_addr: Int32,
    smem_desc_start_b: Int32,
    smem_desc_base_b: int,
    tCrB_layout: cute.Layout,
    smem_var_name_prefix: str,
    idesc_var_name: str,
    smem_offset: int,
    zero_init: bool | Boolean = False,
) -> None:
    is_ts = False
    num_k_tile = cute.size(tCrB_layout.shape[2])
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    offset_b = [cute.crd2idx((0, 0, k), tCrB_layout) for k in range(num_k_tile)]

    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | smem_desc_start_b)
    zero_init_is_dynamic = isinstance(zero_init, Boolean)
    pred_str = "p" if zero_init_is_dynamic else "0" if zero_init else "1"
    pred_input = zero_init if zero_init_is_dynamic else not zero_init
    pred_setp = "setp.eq.b32" if zero_init_is_dynamic else "setp.ne.b32"
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(pred_input).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            f".reg .b64 smem_desc_b_<{num_k_tile}>;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 tmem_acc, $2;\n\t"
            "mov.b32 smem_desc_b_lo_start, $0;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_0;\n\t"
            f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
            f"mov.b64 {smem_var_name_prefix}_0, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b_0, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            + "".join(
                (
                    f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_{k};\n\t"
                    f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
                    f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 {smem_var_name_prefix}_{k}, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b_{k}, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + f"{pred_setp} p, $1, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f8f6f4 [tmem_acc], {smem_var_name_prefix}_0, smem_desc_b_0, {idesc_var_name}, {pred_str};\n\t"
            + "".join(
                (f"@leader_thread tcgen05.mma.cta_group::1.kind::f8f6f4 [tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b_{k}, {idesc_var_name}, 1;\n\t")
                for k in range(1, num_k_tile)
            )
            + "}\n",
            "r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


sm100_utils = types.SimpleNamespace(
    declare_ptx_smem_desc=declare_ptx_smem_desc,
    declare_ptx_idesc=declare_ptx_idesc,
    gemm_ptx_precomputed_varname=gemm_ptx_precomputed_varname,
)


# =============================================================================
# Mask Helpers
# =============================================================================
MASK_R2P_CHUNK_SIZE: int = 32

MaskGenFn: TypeAlias = Callable[[int], Uint32]


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions < limit (exclusive upper bound)."""
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
) -> None:
    """Apply R2P masking with a custom bitmask generator to a rank-1 tensor."""
    ncol = const_expr(cute.size(X.shape))
    CHUNK_SIZE = MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            X[c] = X[c] if in_bound else -Float32.inf


@cute.jit
def _max16_tree(values: cute.TensorSSA) -> Float32:
    """Reduce one K-scale group with eight maxima and three dependent levels."""
    tValues = cute.make_rmem_tensor((16,), Float32)
    tValues.store(values)
    max0 = utils.fmax(tValues[0], tValues[1], tValues[2])
    max1 = utils.fmax(tValues[3], tValues[4], tValues[5])
    max2 = utils.fmax(tValues[6], tValues[7], tValues[8])
    max3 = utils.fmax(tValues[9], tValues[10], tValues[11])
    max4 = utils.fmax(tValues[12], tValues[13], tValues[14])
    return utils.fmax(utils.fmax(max0, max1, max2), utils.fmax(max3, max4, tValues[15]))


@cute.jit
def apply_block_size_mask(
    acc_S: cute.Tensor,
    block_size: Int32,
    n_block_size: cutlass.Constexpr[int] = 128,
) -> None:
    """Apply R2P bitmask masking positions >= block_size within a tile."""
    if block_size < n_block_size:
        mask_r2p_lambda(
            acc_S,
            lambda s: r2p_bitmask_below(block_size, s),
        )


@cute.jit
def tmem_load_red_max_32dp32b16x(
    tmem_addr: Int32,
) -> Tuple[Tuple[Float32, ...], Float32]:
    """Load 16 FP32 TMEM values and return their hardware-reduced maximum."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 17),
        [Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value()],
        "tcgen05.ld.red.sync.aligned.32x32b.x16.max.f32 " "{" "$0, $1, $2, $3, $4, $5, $6, $7, " "$8, $9, $10, $11, $12, $13, $14, $15" "}, $16, [$17];",
        ",".join(["=r"] * 17 + ["r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    values = tuple(Float32(llvm.extractvalue(T.f32(), out, [i])) for i in range(16))
    row_max = Float32(llvm.extractvalue(T.f32(), out, [16]))
    return values, row_max
