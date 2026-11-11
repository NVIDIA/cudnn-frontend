# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hopper Sage FP8 blk64 forward with pretransposed BHDS V."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as hopper_helpers

from cudnn.block_sparse_attention.csrc.utils.batched_static_scheduler import BatchedStaticSchedulerMixin
from .bsa_fwd_sm90 import (
    _tree,
    gemm_zero_acc,
    get_final_ratio_and_lse,
    layout_acc_mn,
    make_acc_into_op,
    rescale_o_for_next_acc,
)


class BlockSparseAttnForwardFp8Sm90Blk64:
    """Choose the schedule from dynamic launch geometry, without recompiling."""

    def __init__(self, num_sms: int):
        # Retain the existing small-grid/large-grid schedule selection.
        self.pipeline_cta_limit = 2 * num_sms
        self.pipelined = _Fp8Sm90Schedule(pipelined=True)
        self.compact = _Fp8Sm90Schedule(pipelined=False)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        blocksparse_indices_q2k: cute.Tensor,
        block_sparse_num: cutlass.Int32,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        num_ctas = (mQ.shape[0] // 64) * mQ.shape[2] * mQ.shape[3]
        if num_ctas <= self.pipeline_cta_limit:
            self.pipelined(mQ, mK, mV, mO, mLSE, blocksparse_indices_q2k, block_sparse_num, mQScale, mKScale, mVScale, softmax_scale, stream)
        else:
            self.compact(mQ, mK, mV, mO, mLSE, blocksparse_indices_q2k, block_sparse_num, mQScale, mKScale, mVScale, softmax_scale, stream)


class _Fp8Sm90Schedule(BatchedStaticSchedulerMixin):
    def __init__(self, pipelined: bool):
        self.pipelined = pipelined
        self.k_stages = 2 if pipelined else 1
        self.gqa_ratio = 1
        self.qk_dim = self.value_dim = 128
        self.tile_size = 64
        self.tile_shape_qk = (64, 64, 128)
        self.tile_shape_pv = (64, 128, 64)
        self.acc_dtype = cutlass.Float32

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
        block_sparse_num: cutlass.Int32,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
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

        shared_storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage_t)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_O)

        op = pipeline.PipelineOp.TmaLoad
        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Small grids use two K/V stages with joint completion. Large grids
        # retain single-stage K to keep register usage and occupancy stable.
        Q_barrier = (pipeline.MbarrierArray(shared_storage.Q_barrier.data_ptr(), num_stages=1, agent=(op, cg)),)
        K_barrier = (pipeline.MbarrierArray(shared_storage.K_barrier.data_ptr(), num_stages=self.k_stages, agent=(op, cg)),)
        if cutlass.const_expr(not self.pipelined):
            V_barrier = (pipeline.MbarrierArray(shared_storage.V_barrier.data_ptr(), num_stages=2, agent=(op, cg)),)

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
        num_n_tiles = block_sparse_num
        gQScale = mQScale[None, work_desc.qo_head_idx, work_desc.batch_idx]
        gKScale = mKScale[None, work_desc.kv_head_idx, work_desc.batch_idx]
        gVScale = mVScale[None, work_desc.kv_head_idx]

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

        tOcO = thr_mma_pv.partition_C(cute.make_identity_tensor(self.tile_shape_pv[:2]))

        max_m_layout = cute.make_layout(cute.size(layout_acc_mn(tiled_mma_pv, tOrO.layout), mode=[0]))
        max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)

        tOrO.store(cute.full_like(tOrO, 0.0, self.acc_dtype))
        max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
        sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

        # Q descales are invariant across the sparse loop. Keep them in registers.
        q_factors = cute.make_rmem_tensor_like(max_m)
        tScS_mn = cute.make_tensor(tScS.iterator, layout_acc_mn(tiled_mma_qk, tScS.layout))
        for m in cutlass.range_constexpr(cute.size(max_m)):
            q_factors[m] = gQScale[work_desc.qo_tile_idx * 64 + tScS_mn[m, 0][0]] * scale_softmax_log2e
        k_factors = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
        if cutlass.const_expr(not self.pipelined):
            k_factors_next = cute.make_rmem_tensor_like(k_factors)
            k_factors_next.fill(0.0)

        # The private launcher validates a positive fixed count.
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        n_tile_ind_ = num_n_tiles - 1  # indirect tile index (logical), needs to be looked up
        n_tile_idx_ = gIndices[n_tile_ind_]  # direct tile index (physical)

        if warp_idx == 0:
            Q_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.Q_dtype, Q_smem_layout))
            cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=Q_barrier[0].get_barrier(0))

            if cutlass.const_expr(self.pipelined):
                K_barrier[0].arrive_and_expect_tx(
                    index=0, tx_count=(cute.size_in_bytes(self.K_dtype, K_smem_layout) + cute.size_in_bytes(self.V_dtype, V_smem_layout)) // 2
                )
                cute.copy(tma_atom_K, tKgK[None, n_tile_idx_], tKsK[None, 0], tma_bar_ptr=K_barrier[0].get_barrier(0))
                cute.copy(tma_atom_V, tVgV[None, n_tile_idx_], tVsV[None, 0], tma_bar_ptr=K_barrier[0].get_barrier(0))
            else:
                K_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.K_dtype, K_smem_layout))
                cute.copy(tma_atom_K, tKgK[None, n_tile_idx_], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))
                V_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.V_dtype, V_smem_layout) // 2)
                cute.copy(tma_atom_V, tVgV[None, n_tile_idx_], tVsV[None, 0], tma_bar_ptr=V_barrier[0].get_barrier(0))

        cute.arch.sync_threads()

        Q_barrier[0].wait(index=0, phase=0)
        for k in cutlass.range_constexpr(4):
            k_factors[k] = gKScale[n_tile_idx_ * 4 + k]
        n_tile_idx_next = n_tile_idx_
        for n_tile_ind in cutlass.range(num_n_tiles - 1, -1, -1):
            n_tile_idx_next = -1
            if n_tile_ind > 0:
                n_tile_idx_next = gIndices[n_tile_ind - 1]
            if cutlass.const_expr(not self.pipelined):
                if n_tile_ind > 0:
                    for k in cutlass.range_constexpr(4):
                        k_factors_next[k] = gKScale[n_tile_idx_next * 4 + k]
            iteration = num_n_tiles - n_tile_ind - 1
            stage = iteration % 2

            if cutlass.const_expr(self.pipelined):
                # The previous iteration's final CTA barrier released the
                # alternate stage. Start its TMA before current QK/softmax/PV.
                if warp_idx == 0 and n_tile_idx_next >= 0:
                    K_barrier[0].arrive_and_expect_tx(
                        index=1 - stage, tx_count=(cute.size_in_bytes(self.K_dtype, K_smem_layout) + cute.size_in_bytes(self.V_dtype, V_smem_layout)) // 2
                    )
                    cute.copy(tma_atom_K, tKgK[None, n_tile_idx_next], tKsK[None, 1 - stage], tma_bar_ptr=K_barrier[0].get_barrier(1 - stage))
                    cute.copy(tma_atom_V, tVgV[None, n_tile_idx_next], tVsV[None, 1 - stage], tma_bar_ptr=K_barrier[0].get_barrier(1 - stage))
                K_barrier[0].wait(index=stage, phase=(iteration // 2) % 2)
            else:
                K_barrier[0].wait(index=0, phase=iteration % 2)

            # compute Q@K
            cute.nvgpu.warpgroup.fence()  # implicit sync WG
            if cutlass.const_expr(self.pipelined):
                gemm_zero_acc(tiled_mma_qk, tSrQ, tSrK[None, None, None, stage], tSrS)
            else:
                gemm_zero_acc(tiled_mma_qk, tSrQ, tSrK, tSrS)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            if cutlass.const_expr(not self.pipelined):
                cute.arch.sync_threads()

                # Prefetch next K and the alternate V stage before softmax/PV.
                if warp_idx == 0 and n_tile_idx_next >= 0:
                    K_barrier[0].arrive_and_expect_tx(index=0, tx_count=cute.size_in_bytes(self.K_dtype, K_smem_layout))
                    cute.copy(tma_atom_K, tKgK[None, n_tile_idx_next], tKsK, tma_bar_ptr=K_barrier[0].get_barrier(0))
                    V_barrier[0].arrive_and_expect_tx(index=1 - stage, tx_count=cute.size_in_bytes(self.V_dtype, V_smem_layout) // 2)
                    cute.copy(tma_atom_V, tVgV[None, n_tile_idx_next], tVsV[None, 1 - stage], tma_bar_ptr=V_barrier[0].get_barrier(1 - stage))

            prev_ratio = _sage_softmax(
                tiled_mma_qk,
                tSrS,
                tScS,
                max_m,
                sum_m,
                q_factors,
                k_factors,
            )

            if cutlass.const_expr(self.pipelined):
                # Current softmax has consumed these scales; reuse their
                # registers for the next sparse block, whose index is physical.
                if n_tile_ind > 0:
                    for k in cutlass.range_constexpr(4):
                        k_factors[k] = gKScale[n_tile_idx_next * 4 + k]
            # PV reads and writes running O, so rescale before issuing it.
            rescale_o_for_next_acc(tiled_mma_pv, tOrO, prev_ratio)
            tOrP = make_acc_into_op(tSrS, tiled_mma_pv.tv_layout_A, self.K_dtype)
            if cutlass.const_expr(not self.pipelined):
                V_barrier[0].wait(index=stage, phase=(iteration // 2) % 2)
            cute.nvgpu.warpgroup.fence()  # implicit sync WG
            tiled_mma_pv.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma_pv, tOrO, tOrP, tOrV[None, None, None, stage], tOrO)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            if cutlass.const_expr(not self.pipelined):
                k_factors.store(k_factors_next.load())
            cute.arch.sync_threads()

        final_ratio, lse = get_final_ratio_and_lse(max_m, sum_m, cutlass.Float32(1.0))
        _normalize_o(tiled_mma_pv, tOrO, tOcO, final_ratio, gVScale)
        tScS_mn = cute.make_tensor(tScS.iterator, layout_acc_mn(tiled_mma_qk, tScS.layout))
        if lane_idx % 4 == 0:
            for m in cutlass.range_constexpr(cute.size(lse)):
                row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
                mLSE_slice[row_idx] = lse[m] - cutlass.Float32(8.0 * 0.6931471805599453)

        tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
        tOrO_cvt.store(tOrO.load().to(self.O_dtype))

        # R2S
        sO = shared_storage.O_smem.get_tensor(O_smem_layout.outer, swizzle=O_smem_layout.inner)
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

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (seqlen, head_dim, nheads, batch)
        mK: cute.Tensor,  # (seqlen, head_dim, nheads, batch)
        mV: cute.Tensor,  # (value_dim, seqlen, nheads, batch)
        mO: cute.Tensor,  # (seqlen, value_dim, nheads, batch)
        mLSE: cute.Tensor,  # (seqlen, nheads, batch)
        blocksparse_indices_q2k: cute.Tensor,  # (k, q, nheads, batch)
        block_sparse_num: cutlass.Int32,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
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
        assert mQ.stride[1] == mK.stride[1] == mO.stride[1] == mV.stride[1] == 1

        Q_layout = utils.LayoutEnum.from_tensor(mQ)
        K_layout = utils.LayoutEnum.from_tensor(mK)
        V_layout = utils.LayoutEnum.from_tensor(mV)
        O_layout = utils.LayoutEnum.from_tensor(mO)

        self.Q_dtype = mQ.element_type
        self.K_dtype = mK.element_type
        self.V_dtype = mV.element_type
        self.O_dtype = mO.element_type

        # Q/K reduce along D; BHDS V reduces along sequence. All are K-major.
        Q_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper_helpers.get_smem_layout_atom(Q_layout, self.Q_dtype, self.tile_shape_qk[2]), self.Q_dtype
        )
        K_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper_helpers.get_smem_layout_atom(K_layout, self.K_dtype, self.tile_shape_qk[2]), self.K_dtype
        )
        V_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper_helpers.get_smem_layout_atom(V_layout, self.V_dtype, self.tile_shape_pv[2]), self.V_dtype
        )

        self.Q_smem_layout = cute.tile_to_shape(Q_smem_layout_atom, (self.tile_shape_qk[0], self.tile_shape_qk[2]), order=(0, 1))
        if cutlass.const_expr(self.pipelined):
            self.K_smem_layout = cute.tile_to_shape(K_smem_layout_atom, (self.tile_shape_qk[1], self.tile_shape_qk[2], 2), order=(0, 1, 2))
        else:
            self.K_smem_layout = cute.tile_to_shape(K_smem_layout_atom, (self.tile_shape_qk[1], self.tile_shape_qk[2]), order=(0, 1))
        self.V_smem_layout = cute.tile_to_shape(V_smem_layout_atom, (self.tile_shape_pv[1], self.tile_shape_pv[2], 2), order=(0, 1, 2))

        O_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper_helpers.get_smem_layout_atom(O_layout, self.O_dtype, self.value_dim), self.O_dtype
        )
        self.O_smem_layout = cute.tile_to_shape(O_smem_layout_atom, self.tile_shape_pv[:2], order=(0, 1))

        @cute.struct
        class SharedStorage:
            Q_barrier: cute.struct.MemRange[cutlass.Int64, 1]
            K_barrier: cute.struct.MemRange[cutlass.Int64, self.k_stages]
            V_barrier: cute.struct.MemRange[cutlass.Int64, 2]

            Q_smem: cute.struct.Align[cute.struct.MemRange[self.Q_dtype, cute.cosize(self.Q_smem_layout)], 128]
            K_smem: cute.struct.Align[cute.struct.MemRange[self.K_dtype, cute.cosize(self.K_smem_layout)], 128]
            V_smem: cute.struct.Align[cute.struct.MemRange[self.V_dtype, cute.cosize(self.V_smem_layout)], 128]

            O_smem: cute.struct.Align[cute.struct.MemRange[self.O_dtype, cute.cosize(self.O_smem_layout)], 128]

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
        tma_atom_O, tma_tensor_O = cute.nvgpu.cpasync.make_tiled_tma_atom(tma_copy_op, mO, self.O_smem_layout, self.tile_shape_pv[:2], num_multicast=1)

        log2_e = 1.44269504088896340736

        block_config = (128, 1, 1)

        grid_config = self.get_grid_config(mQ.shape[0], mQ.shape[2], mQ.shape[3])
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            blocksparse_indices_q2k,
            block_sparse_num,
            mQScale,
            mKScale,
            mVScale,
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
            min_blocks_per_mp=1,
        )


@cute.jit
def _sage_softmax(tiled_mma, scores, coords, row_max, row_sum, q_factors, k_factors, is_first: cutlass.Constexpr = False):
    s = cute.make_tensor(scores.iterator, layout_acc_mn(tiled_mma, scores.layout))
    c = cute.make_tensor(coords.iterator, layout_acc_mn(tiled_mma, coords.layout))
    if cutlass.const_expr(cute.size(k_factors) == 8):
        # In the Q128/N128 two-WG layout, lanes only change a column's low
        # three bits. Its 16-token scale group is therefore the same as in
        # thread zero. Use those static coordinates: a dynamic subscript of
        # the eight-register array otherwise lowers to local-memory traffic.
        c0 = tiled_mma.get_slice(0).partition_C(cute.make_identity_tensor((128, 128)))
        c0 = cute.make_tensor(c0.iterator, layout_acc_mn(tiled_mma, c0.layout))
    if cutlass.const_expr(cute.size(k_factors) == 8):
        # The quantizer produces nonnegative scales. Keep a runtime fallback
        # for signed scales accepted by the internal prequantized launcher.
        nonnegative_k = k_factors[0] >= 0.0
        for group in cutlass.range_constexpr(1, 8):
            nonnegative_k = nonnegative_k and k_factors[group] >= 0.0
    alpha = cute.make_rmem_tensor_like(row_max)
    for m in cutlass.range_constexpr(cute.size(row_max)):
        q_factor = q_factors[m]
        maximum = -cutlass.Float32.inf if cutlass.const_expr(is_first) else row_max[m]
        if cutlass.const_expr(cute.size(k_factors) == 8):
            # For nonnegative factors, max(a*x) = a*max(x) within each
            # 16-token K scale group. Retain raw QK for the probability FMA.
            if q_factor >= 0.0 and nonnegative_k:
                group_max = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
                for group in cutlass.range_constexpr(8):
                    values = [s[m, n] for n in range(cute.size(s, mode=[1])) if c0[m, n][1] // 16 == group]
                    group_max[group] = _tree(values, cute.arch.fmax) * (q_factor * k_factors[group])
                maximum = cute.arch.fmax(maximum, _tree([group_max[g] for g in range(8)], cute.arch.fmax))
            else:
                # Keep arbitrary signed scales and negative softmax scales valid.
                scaled = [s[m, n] * (q_factor * k_factors[c0[m, n][1] // 16]) for n in range(cute.size(s, mode=[1]))]
                maximum = cute.arch.fmax(maximum, _tree(scaled, cute.arch.fmax))
        else:
            for n in cutlass.range_constexpr(cute.size(s, mode=[1])):
                k_factor = k_factors[c[m, n][1] // 16]
                s[m, n] *= q_factor * k_factor
                maximum = cute.arch.fmax(maximum, s[m, n])
        maximum = cute.arch.warp_reduction_max(maximum, threads_in_group=4)
        if cutlass.const_expr(is_first):
            alpha[m] = 1.0
        else:
            alpha[m] = cute.math.exp2(row_max[m] - maximum, fastmath=True)
        row_max[m] = maximum
        # The wider blk128 fragment benefits from exposing all exponentials
        # before the serial sum, as in BF16. Keep blk64's measured schedule.
        # Both paths preserve the summation order and Sage +8 probability shift.
        if cutlass.const_expr(cute.size(k_factors) == 8):
            shift = cutlass.Float32(8.0) - maximum
            for n in cutlass.range_constexpr(cute.size(s, mode=[1])):
                factor = q_factor * k_factors[c0[m, n][1] // 16]
                s[m, n] = cute.math.exp2(cute.math.fma(s[m, n], factor, shift), fastmath=True)
        total = cutlass.Float32(0.0) if cutlass.const_expr(is_first) else row_sum[m] * alpha[m]
        for n in cutlass.range_constexpr(cute.size(s, mode=[1])):
            if cutlass.const_expr(cute.size(k_factors) != 8):
                s[m, n] = cute.math.exp2(s[m, n] - maximum + 8.0, fastmath=True)
            if cutlass.const_expr(is_first and n == 0):
                total = s[m, n]
            else:
                total += s[m, n]
        row_sum[m] = total
    return alpha


@cute.jit
def _normalize_o(tiled_mma, output, coords, inv_sum, v_scale):
    o = cute.make_tensor(output.iterator, layout_acc_mn(tiled_mma, output.layout))
    c = cute.make_tensor(coords.iterator, layout_acc_mn(tiled_mma, coords.layout))
    for m in cutlass.range_constexpr(cute.size(inv_sum)):
        for n in cutlass.range_constexpr(cute.size(o, mode=[1])):
            o[m, n] *= inv_sum[m] * v_scale[c[m, n][1]]
