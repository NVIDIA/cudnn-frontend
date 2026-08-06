# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as hopper_helpers
from cutlass._mlir.dialects import math as _math
from cudnn.block_sparse_attention.csrc.utils.batched_static_scheduler import (
    BatchedStaticSchedulerMixin,
    BatchedStaticWorkDesc,
)

SM90_FWD_BLOCK_SIZE = 64


@dataclass
class SplitBatchedStaticWorkDesc(BatchedStaticWorkDesc):
    split_idx: int


class SplitBatchedStaticSchedulerMixin(BatchedStaticSchedulerMixin):
    def get_split_grid_config(self, seqlen_q, num_qo_heads, batch_size):
        tile_size_m = self.tile_shape_qk[0]
        num_q_tiles = cute.ceil_div(seqlen_q, tile_size_m)
        return (num_q_tiles, self.num_splits, num_qo_heads * batch_size)

    def get_split_work_desc(self, num_qo_heads):
        qo_tile_idx, split_idx, batch_head_idx = cute.arch.block_idx()
        qo_head_idx = batch_head_idx % num_qo_heads
        batch_idx = batch_head_idx // num_qo_heads
        kv_head_idx = qo_head_idx // self.gqa_ratio
        return SplitBatchedStaticWorkDesc(qo_tile_idx, qo_head_idx, kv_head_idx, batch_idx, split_idx)


# =============================================================================
# Public kernel class
# =============================================================================
class BlockSparseAttnForwardSm90Blk64(SplitBatchedStaticSchedulerMixin):
    def __init__(
        self,
        gqa_ratio: int = 1,
        head_dim: int = 128,
        value_dim: int = 128,
        blocksparse_blocksize_q: int = 64,
        blocksparse_blocksize_k: int = 64,
        dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        has_block_sizes: bool = True,
        num_splits: cutlass.Constexpr[int] = 1,
        allow_empty_block_nums: bool = False,
    ):
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        assert self.dtype in [cutlass.Float16, cutlass.BFloat16], "SM90 blk64 fwd supports fp16/bf16"
        assert self.acc_dtype in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]

        self.tile_size = 64

        assert blocksparse_blocksize_q == 64, "Only block_size_m=64 is supported in this kernel."
        assert blocksparse_blocksize_k in [64], "block_size_n should be one of [64]"
        assert gqa_ratio >= 1
        assert head_dim in (64, 96, 128), "SM90 blk64 fwd supports QK dim 64, 96, or 128"
        assert value_dim in (64, 96, 128), "SM90 blk64 fwd supports value dim 64, 96, or 128"
        self.gqa_ratio = gqa_ratio
        self.qk_dim = head_dim
        self.value_dim = value_dim
        self.tile_shape_qk = (self.tile_size, self.tile_size, self.qk_dim)
        self.tile_shape_pv = (self.tile_size, self.value_dim, self.tile_size)

        assert num_splits >= 1, "num_splits must be >= 1"
        self.num_splits = num_splits
        self.use_tma_o = self.value_dim <= head_dim and self.num_splits == 1
        self.has_block_sizes = has_block_sizes
        # Compile-time specialization: only the empty-enabled variant pays for the
        # runtime num_n_tiles > 0 guard in the non-split kernel.
        self.allow_empty_block_nums = allow_empty_block_nums

    def check_dim(self, tensor: cute.Tensor | list[cute.Tensor], mode: int):
        if isinstance(tensor, list):
            for t in tensor:
                self.check_dim(t, mode)
            return
        assert tensor.shape[mode] in [64, 96, 128], f"dim must be one of [64, 96, 128] in mode {mode}."
        assert tensor.stride[mode] == 1, f"dim must be contiguous in mode {mode}."

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        blocksparse_indices_q2k: cute.Tensor,
        blocksparse_num_blocks_q2k: cute.Tensor,
        blocksparse_varblk: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        Q_smem_layout: cute.ComposedLayout,
        K_smem_layout: cute.ComposedLayout,
        V_smem_layout: cute.ComposedLayout,
        O_smem_layout: cute.ComposedLayout,
        scale_softmax_log2e: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        work_desc = self.get_work_desc()
        seqlen = mK.shape[0]
        num_compute_tiles = cute.ceil_div(seqlen, self.tile_size)

        shared_storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage_t)

        if warp_idx == 0 and lane_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)

            if cutlass.const_expr(self.use_tma_o):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_O)

        op = pipeline.PipelineOp.TmaLoad
        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Initialize single stage barrier for Q/K/V
        Q_barrier = (pipeline.MbarrierArray(shared_storage.Q_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)
        K_barrier = (pipeline.MbarrierArray(shared_storage.K_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)
        V_barrier = (pipeline.MbarrierArray(shared_storage.V_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)

        # partition tensors
        sQ = shared_storage.Q_smem.get_tensor(Q_smem_layout.outer, swizzle=Q_smem_layout.inner)
        sK = shared_storage.K_smem.get_tensor(K_smem_layout.outer, swizzle=K_smem_layout.inner)
        sV = shared_storage.V_smem.get_tensor(V_smem_layout.outer, swizzle=V_smem_layout.inner)

        mO_slice = mO[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mLSE_slice = mLSE[None, work_desc.qo_head_idx, work_desc.batch_idx]
        mQ_slice = mQ[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mK_slice = mK[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        mV_slice = mV[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        gO = cute.local_tile(mO_slice, (self.tile_shape_pv[0], self.tile_shape_pv[1]), coord=(work_desc.qo_tile_idx, 0))
        gQ = cute.local_tile(mQ_slice, (self.tile_shape_qk[0], self.tile_shape_qk[2]), coord=(work_desc.qo_tile_idx, 0))
        gK = cute.local_tile(mK_slice, (self.tile_shape_qk[1], self.tile_shape_qk[2]), coord=(None, 0))
        gV = cute.local_tile(mV_slice, (self.tile_shape_pv[1], self.tile_shape_pv[2]), coord=(0, None))

        gIndices = blocksparse_indices_q2k[None, work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        num_n_tiles = blocksparse_num_blocks_q2k[work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        if cutlass.const_expr(self.has_block_sizes):
            gBSZ = blocksparse_varblk[None, work_desc.qo_head_idx, work_desc.batch_idx]

        cta_coord_layout = (0, cute.make_layout(1))  # CTA coord layout for TMA multicasting, effectively no multicast
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, *cta_coord_layout, cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(tma_atom_K, *cta_coord_layout, cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(tma_atom_V, *cta_coord_layout, cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

        cS = cute.make_identity_tensor(self.tile_shape_qk[:2])

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        tSrQ = tiled_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
        tSrS = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((self.tile_shape_qk[0], self.tile_shape_qk[1])), self.acc_dtype)
        tScS = thr_mma_qk.partition_C(cS)

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tOrV = tiled_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sV))
        tOrO = cute.make_rmem_tensor(thr_mma_pv.partition_shape_C((self.tile_shape_pv[0], self.tile_shape_pv[1])), self.acc_dtype)

        max_m_layout = cute.make_layout(cute.size(layout_acc_mn(tiled_mma_pv, tOrO.layout), mode=[0]))
        max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)

        tOrO.store(cute.full_like(tOrO, 0.0, self.acc_dtype))
        max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
        sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

        if cutlass.const_expr(self.allow_empty_block_nums):
            # Variable counts may contain empty rows. Keep the branch CTA-uniform
            # and avoid reading gIndices[-1]. Empty rows fall through to the
            # empty-safe finalizer and produce O=0, LSE=-inf.
            process_tile = num_n_tiles > 0
        else:
            process_tile = True
        if process_tile:
            n_tile_ind_ = num_n_tiles - 1  # indirect tile index (logical), needs to be looked up
            n_tile_idx_ = gIndices[n_tile_ind_]  # direct tile index (physical)

            if warp_idx == 0:
                Q_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.Q_dtype, Q_smem_layout))
                cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=Q_barrier[0].get_barrier(0))

                K_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.K_dtype, K_smem_layout))
                cute.copy(tma_atom_K, tKgK[None, n_tile_idx_], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))

            cute.arch.sync_threads()

            Q_barrier[0].wait(index=0, phase=0)
            n_tile_idx_next = n_tile_idx_
            for n_tile_ind in cutlass.range(num_n_tiles - 1, -1, -1):
                n_tile_idx = n_tile_idx_next
                n_tile_idx_next = -1
                if n_tile_ind > 0:
                    n_tile_idx_next = gIndices[n_tile_ind - 1]
                if cutlass.const_expr(self.has_block_sizes):
                    varblk = gBSZ[n_tile_idx]
                else:
                    varblk = cutlass.Int32(self.tile_size)
                    if n_tile_idx == num_compute_tiles - 1:
                        varblk = seqlen - n_tile_idx * self.tile_size

                # load this V block
                if warp_idx == 0:
                    V_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.V_dtype, V_smem_layout))
                    cute.copy(tma_atom_V, tVgV[None, n_tile_idx], tVsV, tma_bar_ptr=V_barrier[0].get_barrier(0))

                # compute Q@K
                K_barrier[0].wait(index=0, phase=(num_n_tiles - n_tile_ind - 1) % 2)
                cute.nvgpu.warpgroup.fence()  # implicit sync WG
                gemm_zero_acc(tiled_mma_qk, tSrQ, tSrK, tSrS)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                # load next K block
                if warp_idx == 0 and n_tile_idx_next >= 0:
                    K_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.K_dtype, K_smem_layout))
                    cute.copy(tma_atom_K, tKgK[None, n_tile_idx_next], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))

                mask(tiled_mma_qk, tSrS, tScS, varblk)
                prev_ratio = get_prev_ratio_and_update_max_and_rescale_sum(tiled_mma_qk, tSrS, max_m, sum_m, scale_softmax_log2e)
                inc_softmax(tiled_mma_qk, tSrS, max_m, sum_m, scale_softmax_log2e)

                # compute P@V
                rescale_o_for_next_acc(tiled_mma_pv, tOrO, prev_ratio)
                tOrP = make_acc_into_op(tSrS, tiled_mma_pv.tv_layout_A, self.K_dtype)
                V_barrier[0].wait(index=0, phase=(num_n_tiles - n_tile_ind - 1) % 2)
                cute.nvgpu.warpgroup.fence()  # implicit sync WG
                tiled_mma_pv.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma_pv, tOrO, tOrP, tOrV, tOrO)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

        if cutlass.const_expr(self.allow_empty_block_nums):
            final_ratio, lse = get_final_ratio_and_lse_empty_safe(max_m, sum_m, scale_softmax_log2e)
        else:
            final_ratio, lse = get_final_ratio_and_lse(max_m, sum_m, scale_softmax_log2e)
        rescale_o_for_next_acc(tiled_mma_pv, tOrO, final_ratio)
        tScS_mn = cute.make_tensor(tScS.iterator, layout_acc_mn(tiled_mma_qk, tScS.layout))
        for m in cutlass.range_constexpr(cute.size(lse)):
            row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
            if row_idx < mQ.shape[0]:
                mLSE_slice[row_idx] = lse[m]

        tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
        tOrO_cvt.store(tOrO.load().to(self.O_dtype))

        if cutlass.const_expr(self.use_tma_o):
            # R2S
            sO = shared_storage.Q_smem.get_tensor(O_smem_layout.outer, swizzle=O_smem_layout.inner)
            tiled_copy_o_r2s = cute.make_tiled_copy_C(
                cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4), self.O_dtype),
                tiled_mma_pv,
            )
            tOrO_cv = tiled_copy_o_r2s.retile(tOrO_cvt)
            tOsO = tiled_copy_o_r2s.get_slice(tidx).partition_D(sO)
            cute.copy(tiled_copy_o_r2s, tOrO_cv, tOsO)

            # S2G
            # Publish every warp's R2S writes to the TMA async proxy.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp_idx == 0:
                tma_tOsO, tma_tOgO = cute.nvgpu.cpasync.tma_partition(tma_atom_O, *cta_coord_layout, cute.group_modes(sO, 0, 2), cute.group_modes(gO, 0, 2))
                cute.copy(tma_atom_O, tma_tOsO, tma_tOgO)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            tiled_copy_o_r2g = cute.make_tiled_copy_C(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.O_dtype, num_bits_per_copy=32),
                tiled_mma_pv,
            )
            tOrO_cv = tiled_copy_o_r2g.retile(tOrO_cvt)
            tOgO = tiled_copy_o_r2g.get_slice(tidx).partition_D(gO)
            cute.autovec_copy(tOrO_cv, tOgO)

    @cute.kernel
    def kernel_split(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        blocksparse_indices_q2k: cute.Tensor,
        blocksparse_varblk: cute.Tensor,
        blocksparse_split_offsets: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        Q_smem_layout: cute.ComposedLayout,
        K_smem_layout: cute.ComposedLayout,
        V_smem_layout: cute.ComposedLayout,
        scale_softmax_log2e: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        work_desc = self.get_split_work_desc(mQ.shape[2])
        seqlen = mK.shape[0]
        num_compute_tiles = cute.ceil_div(seqlen, self.tile_size)

        shared_storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage_t)

        if warp_idx == 0 and lane_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)

        op = pipeline.PipelineOp.TmaLoad
        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Initialize single stage barrier for Q/K/V
        Q_barrier = (pipeline.MbarrierArray(shared_storage.Q_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)
        K_barrier = (pipeline.MbarrierArray(shared_storage.K_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)
        V_barrier = (pipeline.MbarrierArray(shared_storage.V_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)

        # partition tensors
        sQ = shared_storage.Q_smem.get_tensor(Q_smem_layout.outer, swizzle=Q_smem_layout.inner)
        sK = shared_storage.K_smem.get_tensor(K_smem_layout.outer, swizzle=K_smem_layout.inner)
        sV = shared_storage.V_smem.get_tensor(V_smem_layout.outer, swizzle=V_smem_layout.inner)

        out_head_idx = cutlass.Int64(work_desc.qo_head_idx) + cutlass.Int64(work_desc.split_idx) * cutlass.Int64(mQ.shape[2])
        out_batch_idx = cutlass.Int64(work_desc.batch_idx)
        mO_slice = mO[None, None, out_head_idx, out_batch_idx]
        mLSE_slice = mLSE[None, out_head_idx, out_batch_idx]
        mQ_slice = mQ[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mK_slice = mK[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        mV_slice = mV[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        gO = cute.local_tile(mO_slice, (self.tile_shape_pv[0], self.tile_shape_pv[1]), coord=(work_desc.qo_tile_idx, 0))
        gQ = cute.local_tile(mQ_slice, (self.tile_shape_qk[0], self.tile_shape_qk[2]), coord=(work_desc.qo_tile_idx, 0))
        gK = cute.local_tile(mK_slice, (self.tile_shape_qk[1], self.tile_shape_qk[2]), coord=(None, 0))
        gV = cute.local_tile(mV_slice, (self.tile_shape_pv[1], self.tile_shape_pv[2]), coord=(0, None))

        gIndices = blocksparse_indices_q2k[None, work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        split_start = blocksparse_split_offsets[
            work_desc.split_idx,
            work_desc.qo_tile_idx,
            work_desc.qo_head_idx,
            work_desc.batch_idx,
        ]
        split_end = blocksparse_split_offsets[
            work_desc.split_idx + 1,
            work_desc.qo_tile_idx,
            work_desc.qo_head_idx,
            work_desc.batch_idx,
        ]
        num_n_tiles = split_end - split_start
        gIndices = cute.domain_offset((split_start,), gIndices)
        if cutlass.const_expr(self.has_block_sizes):
            gBSZ = blocksparse_varblk[None, work_desc.qo_head_idx, work_desc.batch_idx]

        cta_coord_layout = (0, cute.make_layout(1))  # CTA coord layout for TMA multicasting, effectively no multicast
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, *cta_coord_layout, cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(tma_atom_K, *cta_coord_layout, cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(tma_atom_V, *cta_coord_layout, cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

        cS = cute.make_identity_tensor(self.tile_shape_qk[:2])

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        tSrQ = tiled_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
        tSrS = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((self.tile_shape_qk[0], self.tile_shape_qk[1])), self.acc_dtype)
        tScS = thr_mma_qk.partition_C(cS)

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tOrV = tiled_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sV))
        tOrO = cute.make_rmem_tensor(thr_mma_pv.partition_shape_C((self.tile_shape_pv[0], self.tile_shape_pv[1])), self.acc_dtype)

        max_m_layout = cute.make_layout(cute.size(layout_acc_mn(tiled_mma_pv, tOrO.layout), mode=[0]))
        max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)

        tOrO.store(cute.full_like(tOrO, 0.0, self.acc_dtype))
        max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
        sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

        # Split offsets can produce empty work for tiles with a short runtime
        # block count. Keep the branch CTA-uniform and avoid reading gIndices[-1].
        if num_n_tiles > 0:
            n_tile_ind_ = num_n_tiles - 1
            n_tile_idx_ = gIndices[n_tile_ind_]

            if warp_idx == 0:
                Q_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.Q_dtype, Q_smem_layout))
                cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=Q_barrier[0].get_barrier(0))

                K_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.K_dtype, K_smem_layout))
                cute.copy(tma_atom_K, tKgK[None, n_tile_idx_], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))

            cute.arch.sync_threads()

            Q_barrier[0].wait(index=0, phase=0)
            n_tile_idx_next = n_tile_idx_
            for n_tile_ind in cutlass.range(num_n_tiles - 1, -1, -1):
                n_tile_idx = n_tile_idx_next
                n_tile_idx_next = -1
                if n_tile_ind > 0:
                    n_tile_idx_next = gIndices[n_tile_ind - 1]
                if cutlass.const_expr(self.has_block_sizes):
                    varblk = gBSZ[n_tile_idx]
                else:
                    varblk = cutlass.Int32(self.tile_size)
                    if n_tile_idx == num_compute_tiles - 1:
                        varblk = seqlen - n_tile_idx * self.tile_size

                # load this V block
                if warp_idx == 0:
                    V_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.V_dtype, V_smem_layout))
                    cute.copy(tma_atom_V, tVgV[None, n_tile_idx], tVsV, tma_bar_ptr=V_barrier[0].get_barrier(0))

                # compute Q@K
                K_barrier[0].wait(index=0, phase=(num_n_tiles - n_tile_ind - 1) % 2)
                cute.nvgpu.warpgroup.fence()  # implicit sync WG
                gemm_zero_acc(tiled_mma_qk, tSrQ, tSrK, tSrS)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                # load next K block
                if warp_idx == 0 and n_tile_idx_next >= 0:
                    K_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.K_dtype, K_smem_layout))
                    cute.copy(tma_atom_K, tKgK[None, n_tile_idx_next], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))

                mask(tiled_mma_qk, tSrS, tScS, varblk)
                prev_ratio = get_prev_ratio_and_update_max_and_rescale_sum(tiled_mma_qk, tSrS, max_m, sum_m, scale_softmax_log2e)
                inc_softmax(tiled_mma_qk, tSrS, max_m, sum_m, scale_softmax_log2e)

                # compute P@V
                rescale_o_for_next_acc(tiled_mma_pv, tOrO, prev_ratio)
                tOrP = make_acc_into_op(tSrS, tiled_mma_pv.tv_layout_A, self.K_dtype)
                V_barrier[0].wait(index=0, phase=(num_n_tiles - n_tile_ind - 1) % 2)
                cute.nvgpu.warpgroup.fence()  # implicit sync WG
                tiled_mma_pv.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma_pv, tOrO, tOrP, tOrV, tOrO)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

        final_ratio, lse = get_final_ratio_and_lse_empty_safe(max_m, sum_m, scale_softmax_log2e)
        rescale_o_for_next_acc(tiled_mma_pv, tOrO, final_ratio)
        tScS_mn = cute.make_tensor(tScS.iterator, layout_acc_mn(tiled_mma_qk, tScS.layout))
        for m in cutlass.range_constexpr(cute.size(lse)):
            row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
            if row_idx < mQ.shape[0]:
                mLSE_slice[row_idx] = lse[m]

        tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
        tOrO_cvt.store(tOrO.load().to(self.O_dtype))

        tiled_copy_o_r2g = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.O_dtype, num_bits_per_copy=32),
            tiled_mma_pv,
        )
        tOrO_cv = tiled_copy_o_r2g.retile(tOrO_cvt)
        tOgO = tiled_copy_o_r2g.get_slice(tidx).partition_D(gO)
        cute.autovec_copy(tOrO_cv, tOgO)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (head_dim, seqlen, nheads, batch)
        mK: cute.Tensor,  # (head_dim, seqlen, nheads, batch)
        mV: cute.Tensor,  # (value_dim, seqlen, nheads, batch)
        mO: cute.Tensor,  # (value_dim, seqlen, nheads, batch)
        mLSE: cute.Tensor,  # (seqlen, nheads, batch)
        blocksparse_indices_q2k: cute.Tensor,  # (k, q, nheads, batch)
        blocksparse_num_blocks_q2k: cute.Tensor,  # (q, nheads, batch)
        blocksparse_varblk: cute.Tensor,
        blocksparse_split_offsets: cute.Tensor,  # (num_splits + 1, q, nheads, batch)
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Restore compile-time head dimensions while keeping sequence, head,
        # batch shapes, and their strides dynamic.
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[0], self.qk_dim, mQ.shape[2], mQ.shape[3]),
                stride=mQ.stride,
            ),
        )
        mK = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (mK.shape[0], self.qk_dim, mK.shape[2], mK.shape[3]),
                stride=mK.stride,
            ),
        )
        mV = cute.make_tensor(
            mV.iterator,
            cute.make_layout(
                (self.value_dim, mV.shape[1], mV.shape[2], mV.shape[3]),
                stride=mV.stride,
            ),
        )
        mO = cute.make_tensor(
            mO.iterator,
            cute.make_layout(
                (mO.shape[0], self.value_dim, mO.shape[2], mO.shape[3]),
                stride=mO.stride,
            ),
        )
        self.check_dim([mQ, mK, mO], 1)
        self.check_dim(mV, 0)

        Q_layout = utils.LayoutEnum.from_tensor(mQ)
        K_layout = utils.LayoutEnum.from_tensor(mK)
        V_layout = utils.LayoutEnum.from_tensor(mV)
        O_layout = utils.LayoutEnum.from_tensor(mO)

        self.Q_dtype = mQ.element_type
        self.K_dtype = mK.element_type
        self.V_dtype = mV.element_type
        self.O_dtype = mO.element_type

        # major mode is K for Query and Key, MN for Value
        Q_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper_helpers.get_smem_layout_atom(Q_layout, self.Q_dtype, self.tile_shape_qk[2]), self.Q_dtype
        )
        K_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper_helpers.get_smem_layout_atom(K_layout, self.K_dtype, self.tile_shape_qk[2]), self.K_dtype
        )
        V_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper_helpers.get_smem_layout_atom(V_layout, self.V_dtype, self.tile_shape_pv[1]), self.V_dtype
        )

        self.Q_smem_layout = cute.tile_to_shape(Q_smem_layout_atom, (self.tile_shape_qk[0], self.tile_shape_qk[2]), order=(0, 1))
        self.K_smem_layout = cute.tile_to_shape(K_smem_layout_atom, (self.tile_shape_qk[1], self.tile_shape_qk[2]), order=(0, 1))
        self.V_smem_layout = cute.tile_to_shape(V_smem_layout_atom, (self.tile_shape_pv[1], self.tile_shape_pv[2]), order=(1, 0))

        self.O_smem_layout = self.Q_smem_layout  # dummy for non TMA O
        if cutlass.const_expr(self.num_splits == 1):
            O_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                hopper_helpers.get_smem_layout_atom(O_layout, self.O_dtype, self.tile_shape_pv[1]), self.O_dtype
            )
            self.O_smem_layout = cute.tile_to_shape(O_smem_layout_atom, (self.tile_shape_pv[0], self.tile_shape_pv[1]), order=(0, 1))

        @cute.struct
        class SharedStorage:
            Q_barrier: cute.struct.MemRange[cutlass.Int64, 1]
            K_barrier: cute.struct.MemRange[cutlass.Int64, 1]
            V_barrier: cute.struct.MemRange[cutlass.Int64, 1]

            Q_smem: cute.struct.Align[cute.struct.MemRange[self.Q_dtype, cute.cosize(self.Q_smem_layout)], 128]
            K_smem: cute.struct.Align[cute.struct.MemRange[self.K_dtype, cute.cosize(self.K_smem_layout)], 128]
            V_smem: cute.struct.Align[cute.struct.MemRange[self.V_dtype, cute.cosize(self.V_smem_layout)], 128]

        self.shared_storage_t = SharedStorage

        atom_layout_mnk = (1, 1, 1)
        tiled_mma_qk = hopper_helpers.make_trivial_tiled_mma(
            self.Q_dtype,
            self.K_dtype,
            Q_layout.sm90_mma_major_mode(),
            K_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            atom_layout_mnk,
            tiler_mn=self.tile_shape_qk[:2],
        )
        tiled_mma_pv = hopper_helpers.make_trivial_tiled_mma(
            self.K_dtype,
            self.V_dtype,
            K_layout.sm90_mma_major_mode(),
            V_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            atom_layout_mnk,
            tiler_mn=self.tile_shape_pv[:2],
            a_source=cute.nvgpu.warpgroup.OperandSource.RMEM,
        )

        tma_copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mQ,
            self.Q_smem_layout,
            (self.tile_shape_qk[0], self.tile_shape_qk[2]),
            num_multicast=1,
        )
        tma_atom_K, tma_tensor_K = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mK,
            self.K_smem_layout,
            (self.tile_shape_qk[1], self.tile_shape_qk[2]),
            num_multicast=1,
        )
        tma_atom_V, tma_tensor_V = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mV,
            self.V_smem_layout,
            (self.tile_shape_pv[1], self.tile_shape_pv[2]),
            num_multicast=1,
        )

        tma_copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_O, tma_tensor_O = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_copy_op,
                mO,
                self.O_smem_layout,
                (self.tile_shape_pv[0], self.tile_shape_pv[1]),
                num_multicast=1,
            )
            if self.use_tma_o
            else (tma_atom_Q, None)
        )  # use tma_atom_Q as dummy

        log2_e = 1.44269504088896340736

        block_config = (128, 1, 1)

        if cutlass.const_expr(self.num_splits == 1):
            grid_config = self.get_grid_config(mQ.shape[0], mQ.shape[2], mQ.shape[3])
            self.kernel(
                tma_tensor_Q,
                tma_tensor_K,
                tma_tensor_V,
                tma_tensor_O if self.use_tma_o else mO,
                mLSE,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_O,
                blocksparse_indices_q2k,
                blocksparse_num_blocks_q2k,
                blocksparse_varblk,
                tiled_mma_qk,
                tiled_mma_pv,
                self.Q_smem_layout,
                self.K_smem_layout,
                self.V_smem_layout,
                self.O_smem_layout,
                softmax_scale * log2_e,
            ).launch(
                grid=grid_config,
                block=block_config,
                smem=self.shared_storage_t.size_in_bytes(),
                stream=stream,
                min_blocks_per_mp=4,
            )
        else:
            grid_config = self.get_split_grid_config(mQ.shape[0], mQ.shape[2], mQ.shape[3])
            self.kernel_split(
                tma_tensor_Q,
                tma_tensor_K,
                tma_tensor_V,
                mO,
                mLSE,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                blocksparse_indices_q2k,
                blocksparse_varblk,
                blocksparse_split_offsets,
                tiled_mma_qk,
                tiled_mma_pv,
                self.Q_smem_layout,
                self.K_smem_layout,
                self.V_smem_layout,
                softmax_scale * log2_e,
            ).launch(
                grid=grid_config,
                block=block_config,
                smem=self.shared_storage_t.size_in_bytes(),
                stream=stream,
                min_blocks_per_mp=4,
            )


# =============================================================================
# Local CuTe helpers
# =============================================================================


def convert_c_layout_to_a_layout(c, a):
    return cute.make_layout(
        (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
        stride=(
            c.stride[0],
            c.stride[1],
            (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
        ),
    )


@cute.jit
def make_acc_into_op(acc, operand_layout_tv, Element):
    operand = cute.make_rmem_tensor_like(
        convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
        Element,
    )
    operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
    acc_vec = acc.load()
    operand_as_acc.store(acc_vec.to(Element))

    if cutlass.const_expr(Element.width == 8 and True):
        tidx, _, _ = cute.arch.thread_idx()
        tid = tidx % 4
        values_u32 = cute.recast_tensor(operand, cutlass.Uint32)
        for n in cutlass.range_constexpr(cute.size(values_u32, mode=[1])):
            for k in cutlass.range_constexpr(cute.size(values_u32, mode=[2])):
                for ii in cutlass.range_constexpr(0, 8, 4):
                    values_tmp_0 = values_u32[ii // 2 + 0, n, k]
                    values_tmp_1 = values_u32[ii // 2 + 1, n, k]

                    v_to_send = 1
                    if tid == 1 or tid == 2:
                        v_to_send = 0

                    v_to_recv = v_to_send
                    t_to_recv_from = (0x3021 >> (tid * 4)) & 0xF

                    values_tmp_a = values_tmp_1
                    if v_to_send == 0:
                        values_tmp_a = values_tmp_0

                    values_tmp_a = cute.arch.shuffle_sync_op(values_tmp_a, t_to_recv_from, 0xFFFFFFFF, 7199)

                    v_to_send = 1 - v_to_send
                    v_to_recv = 1 - v_to_recv
                    t_to_recv_from = (0x2130 >> (tid * 4)) & 0xF

                    values_tmp_b = values_tmp_1
                    if v_to_send == 0:
                        values_tmp_b = values_tmp_0

                    values_tmp_b = cute.arch.shuffle_sync_op(values_tmp_b, t_to_recv_from, 0xFFFFFFFF, 7199)

                    order = 0x5410
                    if v_to_send == 0:
                        order = 0x1054

                    values_u32[ii // 2 + 0, n, k] = cute.arch.prmt(
                        values_tmp_a,
                        values_tmp_b,
                        order,
                    )

                    order = 0x7632
                    if v_to_send == 0:
                        order = 0x3276
                    values_u32[ii // 2 + 1, n, k] = cute.arch.prmt(values_tmp_a, values_tmp_b, order)
    return operand


@cute.jit
def gemm_zero_acc(tiled_mma, A, B, C):
    rA = cute.rank(A)
    rB = cute.rank(B)
    rC = cute.rank(C)
    if cutlass.const_expr(rA == 2 and rB == 2 and rC == 1):
        for k_block_idx in range(cute.size(A, mode=[1]), unroll_full=True):
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
            cute.gemm(
                tiled_mma,
                C,
                A[None, k_block_idx],
                B[None, k_block_idx],
                C,
            )
    elif cutlass.const_expr(rA == 3 and rB == 3 and rC == 3):
        for k_block_idx in range(cute.size(A, mode=[2]), unroll_full=True):
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
            cute.gemm(
                tiled_mma,
                C,
                A[None, None, k_block_idx],
                B[None, None, k_block_idx],
                C,
            )
    else:
        assert 0, "unreachable"


@cute.jit
def reduce_max(
    tSrS_mn: cute.Tensor,
    max_m: cute.Tensor,
):
    for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
        for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
            max_m[m] = cute.arch.fmax(max_m[m], tSrS_mn[m, n])

    for m in cutlass.range_constexpr(cute.size(max_m, mode=[0])):
        max_m[m] = cute.arch.warp_reduction_max(max_m[m], threads_in_group=4)


@cute.jit
def thread_reduce_sum(
    tSrS_mn: cute.Tensor,
    sum_m: cute.Tensor,
):
    for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
        for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
            sum_m[m] += tSrS_mn[m, n]


@cute.jit
def warp_reduce_sum(sum_m: cute.Tensor):
    for m in cutlass.range_constexpr(cute.size(sum_m, mode=[0])):
        sum_m[m] = cute.arch.warp_reduction_sum(sum_m[m], threads_in_group=4)


@cute.jit
def mask(
    qk_tiled_mma: cute.TiledMma,
    tSrS: cute.ThrMma,
    tScS: cute.Tensor,
    varblk: cutlass.Int32,
):
    tSrS_mn = cute.make_tensor(tSrS.iterator, layout_acc_mn(qk_tiled_mma, tSrS.layout))
    tScS_mn = cute.make_tensor(tScS.iterator, layout_acc_mn(qk_tiled_mma, tScS.layout))

    for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
        should_mask = tScS_mn[0, n][1] >= varblk
        for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
            tSrS_mn[m, n] = -cutlass.Float32.inf if should_mask else tSrS_mn[m, n]


@cute.jit
def get_prev_ratio_and_update_max_and_rescale_sum(
    qk_tiled_mma: cute.TiledMma,
    tSrS: cute.ThrMma,
    max_m: cute.Tensor,
    sum_m: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
) -> cute.Tensor:
    tSrS_mn = cute.make_tensor(tSrS.iterator, layout_acc_mn(qk_tiled_mma, tSrS.layout))

    prev_ratio_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
    prev_max_m = cute.make_rmem_tensor_like(max_m, max_m._dtype)
    cute.autovec_copy(max_m, prev_max_m)
    reduce_max(tSrS_mn, max_m)
    for m in cutlass.range_constexpr(cute.size(max_m, mode=[0])):
        prev_max = prev_max_m[m]
        new_max = max_m[m]
        if new_max == -cutlass.Float32.inf:
            new_max = 0.0

        prev_ratio = cute.math.exp2((prev_max - new_max) * softmax_scale_log2e, fastmath=True)
        prev_ratio_m[m] = prev_ratio
        sum_m[m] *= prev_ratio

    return prev_ratio_m


@cute.jit
def inc_softmax(
    tiled_mma_qk: cute.TiledMma,
    tSrS: cute.ThrMma,
    max_m: cute.Tensor,
    sum_m: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
):
    tSrS_mn = cute.make_tensor(tSrS.iterator, layout_acc_mn(tiled_mma_qk, tSrS.layout))

    for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
        new_max = max_m[m]
        if new_max == -cutlass.Float32.inf:
            new_max = 0.0

        for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
            tSrS_mn[m, n] = cute.math.exp2((tSrS_mn[m, n] - new_max) * softmax_scale_log2e, fastmath=True)

    thread_reduce_sum(tSrS_mn, sum_m)


@cute.jit
def rescale_o_for_next_acc(
    pv_tiled_mma: cute.TiledMma,
    tOrO: cute.ThrMma,
    prev_ratio_m: cute.Tensor,
):
    tOrO_mn = cute.make_tensor(tOrO.iterator, layout_acc_mn(pv_tiled_mma, tOrO.layout))
    for m in cutlass.range_constexpr(cute.size(tOrO_mn, mode=[0])):
        for n in cutlass.range_constexpr(cute.size(tOrO_mn, mode=[1])):
            tOrO_mn[m, n] *= prev_ratio_m[m]


@cute.jit
def get_final_ratio_and_lse(
    max_m: cute.Tensor,
    sum_m: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
) -> cute.Tensor:
    warp_reduce_sum(sum_m)
    final_ratio = cute.make_rmem_tensor_like(sum_m, cutlass.Float32)
    lse = cute.make_rmem_tensor_like(sum_m, cutlass.Float32)

    for m in cutlass.range_constexpr(cute.size(sum_m, mode=[0])):
        final_sum = sum_m[m]
        final_ratio[m] = cute.arch.rcp_approx(final_sum)

        ln2 = 0.693147180559945309417
        lse[m] = -cutlass.Float32.inf if final_sum == 0.0 else (max_m[m] * softmax_scale_log2e * ln2 + _math.log(final_sum))

    return final_ratio, lse


@cute.jit
def get_final_ratio_and_lse_empty_safe(
    max_m: cute.Tensor,
    sum_m: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
) -> cute.Tensor:
    warp_reduce_sum(sum_m)
    final_ratio = cute.make_rmem_tensor_like(sum_m, cutlass.Float32)
    lse = cute.make_rmem_tensor_like(sum_m, cutlass.Float32)

    for m in cutlass.range_constexpr(cute.size(sum_m, mode=[0])):
        final_sum = sum_m[m]
        final_ratio[m] = cutlass.Float32(0.0) if final_sum == 0.0 else cute.arch.rcp_approx(final_sum)

        ln2 = 0.693147180559945309417
        lse[m] = -cutlass.Float32.inf if final_sum == 0.0 else (max_m[m] * softmax_scale_log2e * ln2 + _math.log(final_sum))

    return final_ratio, lse


@cute.jit
def layout_acc_mn(tiled_mma, acc):
    separated = layout_separate(tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1])

    V_M = separated[0]
    V_N = separated[1]
    if cutlass.const_expr(cute.rank(V_M) == 1):
        V_M1 = cute.append(V_M, acc[1])
    else:
        V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])

    if cutlass.const_expr(cute.rank(V_N) == 1):
        V_N1 = cute.append(V_N, acc[2])
    else:
        V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
    if cutlass.const_expr(cute.rank(V_M1) == 1):
        return cute.append(V_M1, V_N1)
    return cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)


def layout_separate(thr, src, ref):
    lt = cute.make_layout(())
    ge = cute.make_layout(())

    for k, v in enumerate(ref):
        if cutlass.const_expr(v < thr):
            lt = cute.append(lt, src[k])
        else:
            ge = cute.append(ge, src[k])

    if cutlass.const_expr(cute.rank(lt) == 1):
        return cute.append(lt, ge)
    return cute.append(cute.append(cute.make_layout(()), lt), ge)
