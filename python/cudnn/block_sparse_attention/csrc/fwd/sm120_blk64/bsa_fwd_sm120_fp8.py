# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 Sage FP8 block-sparse attention forward kernel."""

import operator

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cuda.bindings.driver as cuda
from cudnn.block_sparse_attention.csrc.utils import kernel_utils
from cudnn.block_sparse_attention.csrc.utils import layout_utils
from cudnn.block_sparse_attention.csrc.utils.batched_static_scheduler import (
    BatchedStaticSchedulerMixin,
)

SM120_FWD_BLOCK_SIZE = 64


# =============================================================================
# Public kernel class
# =============================================================================
class BlockSparseAttnForwardFp8Sm120Blk64(BatchedStaticSchedulerMixin):
    def __init__(
        self,
        gqa_ratio: int = 1,
        head_dim: int = 128,
        value_dim: int = 128,
        blocksparse_blocksize_q: int = 64,
        blocksparse_blocksize_k: int = 64,
        dtype: type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        has_block_sizes: bool = True,
        has_block_nums: bool = True,
        block_sizes_mode: int = 0,
    ):
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        assert self.dtype is cutlass.Float8E4M3FN, "SM120 FP8 blk64 fwd requires fp8_e4m3fn"
        assert self.acc_dtype is cutlass.Float32
        self.softmax_p_scale_log2 = 8.0

        self.tile_size = 64

        assert blocksparse_blocksize_q == 64, "Only block_size_m=64 is supported in this kernel."
        assert blocksparse_blocksize_k in [64], "block_size_n should be one of [64]"
        self.num_threads = 128
        self.num_mma_warps = 4
        self.kv_stage = 1
        self.q_stage = 1
        self.q_in_regs = True

        assert gqa_ratio >= 1
        assert head_dim == 128, "SM120 blk64 fwd currently requires QK dim 128"
        assert value_dim == 128, "SM120 blk64 fwd currently requires value dim 128"
        self.gqa_ratio = gqa_ratio
        self.qk_dim = head_dim
        self.value_dim = value_dim
        self.tile_shape_qk = (self.tile_size, self.tile_size, self.qk_dim)
        self.tile_shape_pv = (self.tile_size, self.value_dim, self.tile_size)

        self.scheduler = None

        self.use_tma_o = True
        self.has_block_sizes = has_block_sizes
        self.has_block_nums = has_block_nums
        self.block_sizes_mode = block_sizes_mode

    def check_dim(self, tensor: cute.Tensor | list[cute.Tensor], mode: int):
        if isinstance(tensor, list):
            for t in tensor:
                self.check_dim(t, mode)
            return
        assert tensor.shape[mode] == 128, f"dim must be 128 in mode {mode}."
        assert tensor.stride[mode] == 1, f"dim must be contiguous in mode {mode}."

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        blocksparse_indices_q2k: cute.Tensor,
        blocksparse_num_blocks_q2k: cute.Tensor,
        block_sparse_num: cutlass.Int32,
        blocksparse_varblk: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        Q_smem_layout: cute.ComposedLayout,
        K_smem_layout: cute.ComposedLayout,
        V_smem_layout: cute.ComposedLayout,
        Vt_smem_layout: cute.ComposedLayout,
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

        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # TMA load barriers. K/V use a 2-stage ring buffer.
        Q_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads // 32),
            tx_count=cute.size_in_bytes(self.Q_dtype, cute.select(Q_smem_layout, mode=[0, 1])),
            barrier_storage=shared_storage.Q_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        Q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        Q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        K_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads // 32),
            tx_count=cute.size_in_bytes(self.K_dtype, cute.select(K_smem_layout, mode=[0, 1])),
            barrier_storage=shared_storage.K_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        V_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads // 32),
            tx_count=cute.size_in_bytes(self.V_dtype, cute.select(V_smem_layout, mode=[0, 1])),
            barrier_storage=shared_storage.V_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        K_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        K_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        V_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        V_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)

        # partition tensors
        sQ = shared_storage.Q_smem.get_tensor(
            Q_smem_layout.outer,
            swizzle=Q_smem_layout.inner,
        )
        sK = shared_storage.K_smem.get_tensor(K_smem_layout.outer, swizzle=K_smem_layout.inner)
        sV = shared_storage.V_smem.get_tensor(V_smem_layout.outer, swizzle=V_smem_layout.inner)
        # This K-major view only defines the FP8 B-fragment layout. The direct
        # transpose load below sources data from the N-major sV.
        sO_buffer = shared_storage.O_smem.get_tensor(
            O_smem_layout.outer,
            swizzle=O_smem_layout.inner,
        )
        sVt = cute.make_tensor(
            cute.recast_ptr(
                sO_buffer.iterator,
                Vt_smem_layout.inner,
                dtype=self.V_dtype,
            ),
            Vt_smem_layout.outer,
        )

        mO_slice = mO[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mLSE_slice = mLSE[None, work_desc.qo_head_idx, work_desc.batch_idx]
        mQ_slice = mQ[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mK_slice = mK[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        mV_slice = mV[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        gQScale = mQScale[None, work_desc.qo_head_idx, work_desc.batch_idx]
        gKScale = mKScale[None, work_desc.kv_head_idx, work_desc.batch_idx]
        gVScale = mVScale[None, work_desc.kv_head_idx]
        gO = cute.local_tile(mO_slice, (self.tile_shape_pv[0], self.tile_shape_pv[1]), coord=(work_desc.qo_tile_idx, 0))
        gQ = cute.local_tile(mQ_slice, (self.tile_shape_qk[0], self.tile_shape_qk[2]), coord=(work_desc.qo_tile_idx, 0))
        gK = cute.local_tile(mK_slice, (self.tile_shape_qk[1], self.tile_shape_qk[2]), coord=(None, 0))
        gV = cute.local_tile(mV_slice, (self.tile_shape_pv[1], self.tile_shape_pv[2]), coord=(0, None))

        gIndices = blocksparse_indices_q2k[None, work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        if cutlass.const_expr(self.has_block_nums):
            num_n_tiles = blocksparse_num_blocks_q2k[work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        else:
            num_n_tiles = block_sparse_num
        if cutlass.const_expr(self.has_block_sizes):
            if cutlass.const_expr(self.block_sizes_mode == 1):
                gBSZ = blocksparse_varblk
            elif cutlass.const_expr(self.block_sizes_mode == 2):
                gBSZ = blocksparse_varblk[None, work_desc.batch_idx]
            else:
                gBSZ = blocksparse_varblk[None, work_desc.qo_head_idx, work_desc.batch_idx]

        cta_coord_layout = (0, cute.make_layout(1))  # CTA coord layout for TMA multicasting, effectively no multicast
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            *cta_coord_layout,
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            *cta_coord_layout,
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            *cta_coord_layout,
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )

        cS = cute.make_identity_tensor(self.tile_shape_qk[:2])

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        tSsQ = thr_mma_qk.partition_A(sQ)
        tSsK = thr_mma_qk.partition_B(sK)
        tSrQ = tiled_mma_qk.make_fragment_A(tSsQ[None, None, None, 0])
        tSrK = tiled_mma_qk.make_fragment_B(tSsK[None, None, None, 0])
        tSrS = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((self.tile_shape_qk[0], self.tile_shape_qk[1])), self.acc_dtype)
        tScS = thr_mma_qk.partition_C(cS)

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tOsV = thr_mma_pv.partition_B(sVt)
        tOrV = tiled_mma_pv.make_fragment_B(tOsV[None, None, None, 0])
        tOrO = cute.make_rmem_tensor(thr_mma_pv.partition_shape_C((self.tile_shape_pv[0], self.tile_shape_pv[1])), self.acc_dtype)
        cO = cute.make_identity_tensor(self.tile_shape_pv[:2])
        tOcO = thr_mma_pv.partition_C(cO)

        atom_copy_ldmatrix_Q = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False,
                num_matrices=4,
            ),
            self.Q_dtype,
        )
        atom_copy_ldmatrix_K = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False,
                num_matrices=4,
            ),
            self.K_dtype,
        )
        atom_copy_ldmatrix_V = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix16x16x8bOp(
                transpose=True,
                num_matrices=2,
            ),
            self.V_dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(atom_copy_ldmatrix_Q, tiled_mma_qk)
        smem_tiled_copy_K = cute.make_tiled_copy_B(atom_copy_ldmatrix_K, tiled_mma_qk)
        smem_tiled_copy_V = cute.make_tiled_copy_B(
            atom_copy_ldmatrix_V,
            tiled_mma_pv,
        )

        thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
        tSsQ_copy = thr_copy_Q.partition_S(sQ)
        tSrQ_copy = thr_copy_Q.retile(tSrQ)
        tSsK_copy = thr_copy_K.partition_S(sK)
        tOsV_copy = thr_copy_V.partition_S(sVt)

        max_m_layout = cute.make_layout(cute.size(layout_utils.reshape_acc_to_mn(tOrO).layout, mode=[0]))
        max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
        q_softmax_scale_log2e_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
        tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
        for m in cutlass.range_constexpr(cute.size(q_softmax_scale_log2e_m)):
            row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
            q_softmax_scale_log2e_m[m] = gQScale[row_idx] * scale_softmax_log2e if row_idx < mQ.shape[0] else cutlass.Float32(0.0)

        tOrO.store(cute.full_like(tOrO, 0.0, self.acc_dtype))
        max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
        sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

        if warp_idx == 0:
            Q_pipeline.producer_acquire(Q_producer_state)
            cute.copy(
                tma_atom_Q,
                tQgQ,
                tQsQ[None, 0],
                tma_bar_ptr=Q_pipeline.producer_get_barrier(Q_producer_state),
            )
            Q_pipeline.producer_commit(Q_producer_state)
            Q_producer_state.advance()

            preload_count = cutlass.Int32(0)
            if preload_count < num_n_tiles:
                logical_idx = num_n_tiles - 1 - preload_count
                physical_idx = gIndices[logical_idx]
                K_pipeline.producer_acquire(K_producer_state)
                cute.copy(
                    tma_atom_K,
                    tKgK[None, physical_idx],
                    tKsK[None, K_producer_state.index],
                    tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer_state),
                )
                K_pipeline.producer_commit(K_producer_state)
                K_producer_state.advance()
                V_pipeline.producer_acquire(V_producer_state)
                cute.copy(
                    tma_atom_V,
                    tVgV[None, physical_idx],
                    tVsV[None, V_producer_state.index],
                    tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer_state),
                )
                V_pipeline.producer_commit(V_producer_state)
                V_producer_state.advance()
            if cutlass.const_expr(self.kv_stage > 1):
                preload_count = cutlass.Int32(1)
                if preload_count < num_n_tiles:
                    logical_idx = num_n_tiles - 1 - preload_count
                    physical_idx = gIndices[logical_idx]
                    K_pipeline.producer_acquire(K_producer_state)
                    cute.copy(
                        tma_atom_K,
                        tKgK[None, physical_idx],
                        tKsK[None, K_producer_state.index],
                        tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer_state),
                    )
                    K_pipeline.producer_commit(K_producer_state)
                    K_producer_state.advance()
                    V_pipeline.producer_acquire(V_producer_state)
                    cute.copy(
                        tma_atom_V,
                        tVgV[None, physical_idx],
                        tVsV[None, V_producer_state.index],
                        tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer_state),
                    )
                    V_pipeline.producer_commit(V_producer_state)
                    V_producer_state.advance()

        cute.arch.sync_threads()

        Q_wait_status = Q_pipeline.consumer_try_wait(Q_consumer_state)
        Q_pipeline.consumer_wait(Q_consumer_state, Q_wait_status)
        if cutlass.const_expr(self.q_in_regs):
            tQsQ_p = tSsQ_copy[None, None, None, 0]
            for k_block_idx in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
                tQsQ_k = tQsQ_p[None, None, k_block_idx]
                tQsQ_k = cute.make_tensor(
                    tQsQ_k.iterator.align(16),
                    tQsQ_k.layout,
                )
                cute.copy(
                    smem_tiled_copy_Q,
                    tQsQ_k,
                    tSrQ_copy[None, None, k_block_idx],
                )
        Q_pipeline.consumer_release(Q_consumer_state)
        Q_consumer_state.advance()

        for load_count in cutlass.range(0, num_n_tiles, 1, unroll=1):
            n_tile_ind = num_n_tiles - 1 - load_count
            n_tile_idx = gIndices[n_tile_ind]
            if cutlass.const_expr(self.has_block_sizes):
                varblk = gBSZ[n_tile_idx]
            else:
                varblk = cutlass.Int32(self.tile_size)
                if n_tile_idx == num_compute_tiles - 1:
                    varblk = seqlen - n_tile_idx * self.tile_size

            K_wait_status = K_pipeline.consumer_try_wait(K_consumer_state)
            K_pipeline.consumer_wait(K_consumer_state, K_wait_status)

            k_stage = K_consumer_state.index

            _gemm_smem_zero_acc_fp8(
                tiled_mma_qk,
                tSrS,
                tSrQ,
                tSrK,
                tSsQ_copy[None, None, None, 0],
                tSsK_copy[None, None, None, k_stage],
                smem_tiled_copy_Q,
                smem_tiled_copy_K,
                A_in_regs=self.q_in_regs,
                B_uses_fp8_ldsm=True,
            )

            tKScale = _load_sage_k_scales_fp8(
                tScS,
                gKScale,
                n_tile_idx,
                self.tile_size,
            )

            K_pipeline.consumer_release(K_consumer_state)
            K_consumer_state.advance()

            preload_count = load_count + self.kv_stage
            preload_physical = cutlass.Int32(0)
            if warp_idx == 0 and preload_count < num_n_tiles:
                preload_logical = num_n_tiles - 1 - preload_count
                preload_physical = gIndices[preload_logical]
                K_pipeline.producer_acquire(K_producer_state)
                cute.copy(
                    tma_atom_K,
                    tKgK[None, preload_physical],
                    tKsK[None, K_producer_state.index],
                    tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer_state),
                )
                K_pipeline.producer_commit(K_producer_state)
                K_producer_state.advance()

            if varblk < self.tile_size:
                _mask_fp8(tiled_mma_qk, tSrS, tScS, varblk)
            row_scale = _online_softmax_fp8(
                tiled_mma_qk,
                tSrS,
                max_m,
                sum_m,
                q_softmax_scale_log2e_m,
                self.softmax_p_scale_log2,
                tKScale,
            )

            # Compute P @ V.
            _rescale_o_for_next_acc_fp8(tiled_mma_pv, tOrO, row_scale)
            tOrP = _make_acc_into_fp8_op(
                tSrS,
                tiled_mma_pv.tv_layout_A,
                self.V_dtype,
            )

            V_wait_status = V_pipeline.consumer_try_wait(V_consumer_state)
            V_pipeline.consumer_wait(V_consumer_state, V_wait_status)

            # Load each 16x32 N-major V subtile with the native 8-bit
            # transpose path. Hopper WGMMA writes the transposed values back
            # to K-major SMEM, but SM120 warp MMA consumes B from registers,
            # so the LDSM.T result can directly form the FP8 B fragment.
            destination_layout = cute.make_layout(
                ((4, (2, 2)),),
                stride=((1, (16, 4)),),
            )
            for k_block in cutlass.range_constexpr(2):
                for n_block in cutlass.range_constexpr(8):
                    source_ptr = sV.iterator + cute.crd2idx(
                        (n_block * 16, k_block * 32 + lane_idx, 0),
                        sV.layout,
                    )
                    source = cute.make_tensor(
                        source_ptr.align(16),
                        cute.make_layout(16),
                    )
                    block_idx = k_block * 8 + n_block
                    destination_base = (block_idx % 8) * 8 + (block_idx // 8) * 2
                    destination = cute.make_tensor(
                        tOrV.iterator + destination_base * 4,
                        destination_layout,
                    )
                    cute.copy(
                        atom_copy_ldmatrix_V,
                        source,
                        destination,
                    )

            _gemm_rs_fp8(
                tiled_mma_pv,
                tOrO,
                tOrP,
                tOrV,
                tOsV_copy[None, None, None, 0],
                smem_tiled_copy_V,
                B_in_regs=True,
            )
            V_pipeline.consumer_release(V_consumer_state)
            V_consumer_state.advance()

            if warp_idx == 0 and preload_count < num_n_tiles:
                V_pipeline.producer_acquire(V_producer_state)
                cute.copy(
                    tma_atom_V,
                    tVgV[None, preload_physical],
                    tVsV[None, V_producer_state.index],
                    tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer_state),
                )
                V_pipeline.producer_commit(V_producer_state)
                V_producer_state.advance()

        final_ratio, lse = _finalize_softmax_fp8(
            max_m,
            sum_m,
            q_softmax_scale_log2e_m,
            self.softmax_p_scale_log2,
        )
        _rescale_o_with_sage_v_scale_fp8(
            tOrO,
            tOcO,
            final_ratio,
            gVScale,
        )
        tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
        for m in cutlass.range_constexpr(cute.size(lse)):
            row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
            if tScS_mn[m, 0][1] == 0:
                if row_idx < mQ.shape[0]:
                    mLSE_slice[row_idx] = lse[m]

        tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
        tOrO_cvt.store(tOrO.load().to(self.O_dtype))

        # FP8 Q needs a separate, larger buffer for the BF16 output tile.
        sO = shared_storage.O_smem.get_tensor(O_smem_layout.outer, swizzle=O_smem_layout.inner)
        tiled_copy_o_r2s = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.O_layout.is_m_major_c(), 4),
                self.O_dtype,
            ),
            tiled_mma_pv,
        )
        tOrO_cv = tiled_copy_o_r2s.retile(tOrO_cvt)
        tOsO = tiled_copy_o_r2s.get_slice(tidx).partition_D(sO)
        cute.copy(tiled_copy_o_r2s, tOrO_cv, tOsO)

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # S2G with explicit TMA store wait.
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_O,
            *cta_coord_layout,
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )
        if warp_idx == 0:
            cute.copy(tma_atom_O, tOsO, tOgO)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (head_dim, seqlen, nheads, batch)
        mK: cute.Tensor,  # (head_dim, seqlen, nheads, batch)
        mV: cute.Tensor,  # (value_dim, seqlen, nheads, batch)
        mO: cute.Tensor,  # (value_dim, seqlen, nheads, batch)
        mLSE: cute.Tensor,  # (seqlen, nheads, batch)
        mQScale: cute.Tensor,  # (seqlen_q, nheads_q, batch)
        mKScale: cute.Tensor,  # (ceil_div(seqlen_k, 16), nheads_kv, batch)
        mVScale: cute.Tensor,  # (value_dim, nheads_kv)
        blocksparse_indices_q2k: cute.Tensor,  # (k, q, nheads, batch)
        blocksparse_num_blocks_q2k: cute.Tensor,  # (q, nheads, batch)
        block_sparse_num: cutlass.Int32,
        blocksparse_varblk: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Restore compile-time head dimensions while keeping runtime tensor
        # modes dynamic for reusable JIT callables.
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
        mVScale = cute.make_tensor(
            mVScale.iterator,
            cute.make_layout(
                (self.value_dim, mVScale.shape[1]),
                stride=mVScale.stride,
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
        assert self.Q_dtype == self.K_dtype == self.V_dtype == cutlass.Float8E4M3FN
        assert self.O_dtype is cutlass.BFloat16
        self.Q_layout = Q_layout
        self.K_layout = K_layout
        self.V_layout = V_layout
        self.O_layout = O_layout

        atom_layout_mnk = (4, 1, 1)
        mma_inst_mnk = (16, 8, 32)
        permutation_mnk = (
            atom_layout_mnk[0] * mma_inst_mnk[0],
            atom_layout_mnk[1] * mma_inst_mnk[1] * 2,
            atom_layout_mnk[2] * mma_inst_mnk[2],
        )
        mma_op_qk = cute.nvgpu.warp.MmaFP8Op(
            self.Q_dtype,
            self.acc_dtype,
            mma_inst_mnk,
        )
        mma_op_pv = cute.nvgpu.warp.MmaFP8Op(
            self.V_dtype,
            self.acc_dtype,
            mma_inst_mnk,
        )
        tiled_mma_qk = cute.make_tiled_mma(
            mma_op_qk,
            cute.make_layout(atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )
        tiled_mma_pv = cute.make_tiled_mma(
            mma_op_pv,
            cute.make_layout(atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

        self.Q_smem_layout = sm90_utils.make_smem_layout_a(
            Q_layout,
            self.tile_shape_qk,
            self.Q_dtype,
            self.q_stage,
        )
        self.K_smem_layout = sm90_utils.make_smem_layout_b(
            K_layout,
            self.tile_shape_qk,
            self.K_dtype,
            self.kv_stage,
        )
        self.V_smem_layout = sm90_utils.make_smem_layout_b(
            V_layout,
            self.tile_shape_pv,
            self.V_dtype,
            self.kv_stage,
        )
        vt_smem_layout_atom = cute.nvgpu.tcgen05.make_smem_layout_atom(
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW64,
            self.V_dtype,
        )
        self.Vt_smem_layout = cute.tile_to_shape(
            vt_smem_layout_atom,
            (*self.tile_shape_pv[1:], self.kv_stage),
            order=(0, 1, 2),
        )
        O_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.O_dtype,
            O_layout,
            self.tile_shape_pv[:2],
            1,
        )
        self.O_smem_layout = cute.select(O_smem_layout_staged, mode=[0, 1])

        @cute.struct
        class SharedStorage:
            Q_barrier: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            K_barrier: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            V_barrier: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]

            Q_smem: cute.struct.Align[cute.struct.MemRange[self.Q_dtype, cute.cosize(self.Q_smem_layout)], 128]
            K_smem: cute.struct.Align[cute.struct.MemRange[self.K_dtype, cute.cosize(self.K_smem_layout)], 128]
            V_smem: cute.struct.Align[cute.struct.MemRange[self.V_dtype, cute.cosize(self.V_smem_layout)], 128]
            O_smem: cute.struct.Align[cute.struct.MemRange[self.O_dtype, cute.cosize(self.O_smem_layout)], 128]

        self.shared_storage_t = SharedStorage

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
        tma_atom_O, tma_tensor_O = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mO,
            self.O_smem_layout,
            (self.tile_shape_pv[0], self.tile_shape_pv[1]),
            num_multicast=1,
        )

        log2_e = 1.44269504088896340736

        grid_config = self.get_grid_config(mQ.shape[0], mQ.shape[2], mQ.shape[3])
        block_config = (self.num_threads, 1, 1)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O,
            mLSE,
            mQScale,
            mKScale,
            mVScale,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            blocksparse_indices_q2k,
            blocksparse_num_blocks_q2k,
            block_sparse_num,
            blocksparse_varblk,
            tiled_mma_qk,
            tiled_mma_pv,
            self.Q_smem_layout,
            self.K_smem_layout,
            self.V_smem_layout,
            self.Vt_smem_layout,
            self.O_smem_layout,
            softmax_scale * log2_e,
        ).launch(
            grid=grid_config,
            block=block_config,
            cluster=(1, 1, 1),
            smem=self.shared_storage_t.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )


# =============================================================================
# Local CuTe helpers
# =============================================================================


def _convert_c_layout_to_a_layout_fp8(c: cute.Layout, a: cute.Layout) -> cute.Layout:
    """Reinterpret a C fragment with the logical shape of an A operand."""
    a_layout = cute.make_layout(a)
    c_atom_size = cute.size(c, mode=[0])
    expansion = cute.size(a) // c_atom_size
    return cute.make_layout(
        (a, c.shape[1], c.shape[2] // expansion),
        stride=(
            a_layout.stride,
            0,
            cute.size(a),
        ),
    )


@cute.jit
def _make_acc_into_fp8_op(
    acc: cute.Tensor,
    operand_layout_tv: cute.Layout,
    element: type[cutlass.Numeric],
) -> cute.Tensor:
    """Apply FA3's C-layout reinterpretation, E4M3 cast, and A-reg permutation."""
    operand = cute.make_rmem_tensor_like(
        _convert_c_layout_to_a_layout_fp8(acc.layout, operand_layout_tv.shape[1]),
        element,
    )
    operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
    operand_as_acc.store(acc.load().to(element))

    # The C and FP8 A layouts distribute adjacent bytes differently across
    # each quad. Exchange packed groups of four FP8 values in registers.
    tidx, _, _ = cute.arch.thread_idx()
    tid = tidx % 4
    values_u32 = cute.recast_tensor(operand, cutlass.Uint32)
    for n in cutlass.range_constexpr(cute.size(values_u32, mode=[1])):
        for k in cutlass.range_constexpr(cute.size(values_u32, mode=[2])):
            for ii in cutlass.range_constexpr(0, 8, 4):
                values_tmp_0 = values_u32[ii // 2, n, k]
                values_tmp_1 = values_u32[ii // 2 + 1, n, k]

                use_first = tid == 1 or tid == 2
                source_lane = (0x3021 >> (tid * 4)) & 0xF
                values_tmp_a = values_tmp_0 if use_first else values_tmp_1
                values_tmp_a = cute.arch.shuffle_sync_op(values_tmp_a, source_lane, 0xFFFFFFFF, 7199)

                use_first = not use_first
                source_lane = (0x2130 >> (tid * 4)) & 0xF
                values_tmp_b = values_tmp_0 if use_first else values_tmp_1
                values_tmp_b = cute.arch.shuffle_sync_op(values_tmp_b, source_lane, 0xFFFFFFFF, 7199)

                selector = 0x1054 if use_first else 0x5410
                values_u32[ii // 2, n, k] = cute.arch.prmt(values_tmp_a, values_tmp_b, selector)
                selector = 0x3276 if use_first else 0x7632
                values_u32[ii // 2 + 1, n, k] = cute.arch.prmt(values_tmp_a, values_tmp_b, selector)
    return operand


@cute.jit
def _gemm_smem_zero_acc_fp8(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_tiled_copy_A: cute.TiledCopy,
    smem_tiled_copy_B: cute.TiledCopy,
    A_in_regs: bool = False,
    B_uses_fp8_ldsm: bool = False,
) -> None:
    acc.fill(0.0)
    tCrA_copy_view = smem_tiled_copy_A.retile(tCrA)
    tCrB_copy_view = smem_tiled_copy_B.retile(tCrB)
    if cutlass.const_expr(not A_in_regs):
        cute.copy(smem_tiled_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
    tCsB_cur = tCsB[None, None, 0]
    if cutlass.const_expr(B_uses_fp8_ldsm):
        tCsB_cur = cute.make_tensor(
            tCsB_cur.iterator.align(16),
            tCsB_cur.layout,
        )
    cute.copy(smem_tiled_copy_B, tCsB_cur, tCrB_copy_view[None, None, 0])
    for k_block_idx in cutlass.range_constexpr(cute.size(tCsB.shape[2])):
        if k_block_idx < cute.size(tCsB.shape[2]) - 1:
            if cutlass.const_expr(not A_in_regs):
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA[None, None, k_block_idx + 1],
                    tCrA_copy_view[None, None, k_block_idx + 1],
                )
            tCsB_next = tCsB[None, None, k_block_idx + 1]
            if cutlass.const_expr(B_uses_fp8_ldsm):
                tCsB_next = cute.make_tensor(
                    tCsB_next.iterator.align(16),
                    tCsB_next.layout,
                )
            cute.copy(
                smem_tiled_copy_B,
                tCsB_next,
                tCrB_copy_view[None, None, k_block_idx + 1],
            )
        cute.gemm(
            tiled_mma,
            acc,
            tCrA[None, None, k_block_idx],
            tCrB[None, None, k_block_idx],
            acc,
        )


@cute.jit
def _gemm_rs_fp8(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_tiled_copy_B: cute.TiledCopy,
    B_in_regs: bool = False,
) -> None:
    if cutlass.const_expr(not B_in_regs):
        tCrB_copy_view = smem_tiled_copy_B.retile(tCrB)
        tCsB_cur = tCsB[None, None, 0]
        cute.copy(smem_tiled_copy_B, tCsB_cur, tCrB_copy_view[None, None, 0])
    for k_block_idx in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if cutlass.const_expr(not B_in_regs):
            if k_block_idx < cute.size(tCrA.shape[2]) - 1:
                tCsB_next = tCsB[None, None, k_block_idx + 1]
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_next,
                    tCrB_copy_view[None, None, k_block_idx + 1],
                )
        cute.gemm(
            tiled_mma,
            acc,
            tCrA[None, None, k_block_idx],
            tCrB[None, None, k_block_idx],
            acc,
        )


@cute.jit
def _load_sage_k_scales_fp8(
    tScS: cute.Tensor,
    gKScale: cute.Tensor,
    k_tile_idx: cutlass.Int32,
    tile_size: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Load the four per-token-group K descales for one KV tile."""
    tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
    num_k_scales = tile_size // 16
    tKScale = cute.make_rmem_tensor(
        cute.make_layout(num_k_scales),
        cutlass.Float32,
    )
    k_scale_base = k_tile_idx * num_k_scales
    scores_per_k_scale = cute.size(tScS_mn, mode=[1]) // num_k_scales
    for k_scale_group in cutlass.range_constexpr(num_k_scales):
        tile_scale_idx = tScS_mn[0, k_scale_group * scores_per_k_scale][1] // 16
        scale_idx = k_scale_base + tile_scale_idx
        scale_idx = scale_idx if scale_idx < gKScale.shape[0] else gKScale.shape[0] - 1
        tKScale[k_scale_group] = gKScale[scale_idx]
    return tKScale


@cute.jit
def _mask_fp8(
    qk_tiled_mma: cute.TiledMma,
    tSrS: cute.ThrMma,
    tScS: cute.Tensor,
    varblk: cutlass.Int32,
):
    tSrS_mn = layout_utils.reshape_acc_to_mn(tSrS)
    tScS_mn = layout_utils.reshape_acc_to_mn(tScS)

    for n in cutlass.range(cute.size(tSrS_mn, mode=[1]), unroll_full=True):
        should_mask = tScS_mn[0, n][1] >= varblk
        for m in cutlass.range(cute.size(tSrS_mn, mode=[0]), unroll_full=True):
            tSrS_mn[m, n] = -cutlass.Float32.inf if should_mask else tSrS_mn[m, n]


@cute.jit
def _online_softmax_fp8(
    tiled_mma_qk: cute.TiledMma,
    tSrS: cute.ThrMma,
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    softmax_scale_log2e_m: cute.Tensor,
    exp_scale_log2: cutlass.Float32,
    tKScale: cute.Tensor,
) -> cute.Tensor:
    tSrS_mn = layout_utils.reshape_acc_to_mn(tSrS)
    row_scale = cute.make_rmem_tensor_like(row_max, cutlass.Float32)
    num_k_scales = cute.size(tKScale)
    scores_per_k_scale = cute.size(tSrS_mn, mode=[1]) // num_k_scales

    for m in cutlass.range(cute.size(row_max), unroll_full=True):
        softmax_scale_log2e = softmax_scale_log2e_m[m]
        row_max_local = row_max[m]
        # K descales are positive and shared by each score group, so reduce
        # unscaled scores first and apply one descale to the group maximum.
        for k_scale_group in cutlass.range_constexpr(num_k_scales):
            k_scale = tKScale[k_scale_group]
            group_begin = k_scale_group * scores_per_k_scale
            group_max = tSrS_mn[m, group_begin]
            for n in cutlass.range_constexpr(
                group_begin + 1,
                (k_scale_group + 1) * scores_per_k_scale,
            ):
                group_max = cute.arch.fmax(group_max, tSrS_mn[m, n])
            row_max_local = cute.arch.fmax(
                row_max_local,
                group_max * k_scale,
            )
        row_max_cur = cute.arch.warp_reduction_max(
            row_max_local,
            threads_in_group=4,
        )

        row_max_prev = row_max[m]
        row_max[m] = row_max_cur
        row_max_safe = cutlass.Float32(0.0) if row_max_cur == -cutlass.Float32.inf else row_max_cur
        # Shift the exponent by log2(P scale) before FP8 conversion. Keeping
        # row_sum in the same scale removes a vector multiply from every tile.
        row_max_scaled = row_max_safe * softmax_scale_log2e - exp_scale_log2

        for k_scale_group in cutlass.range_constexpr(num_k_scales):
            score_scale_log2e = tKScale[k_scale_group] * softmax_scale_log2e
            for n in cutlass.range_constexpr(
                k_scale_group * scores_per_k_scale,
                (k_scale_group + 1) * scores_per_k_scale,
            ):
                tSrS_mn[m, n] = cute.math.exp2(
                    tSrS_mn[m, n] * score_scale_log2e - row_max_scaled,
                    fastmath=True,
                )
        acc_S_row_exp = tSrS_mn[m, None].load()
        row_scale[m] = cute.math.exp2(
            (row_max_prev - row_max_safe) * softmax_scale_log2e,
            fastmath=True,
        )
        row_sum[m] = kernel_utils.fadd_reduce(
            acc_S_row_exp,
            init_val=row_sum[m] * row_scale[m],
            arch=80,
        )
        tSrS_mn[m, None].store(acc_S_row_exp)

    return row_scale


@cute.jit
def _finalize_softmax_fp8(
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    softmax_scale_log2e_m: cute.Tensor,
    exp_scale_log2: cutlass.Float32,
) -> cute.Tensor:
    row_sum.store(kernel_utils.warp_reduce(row_sum.load(), operator.add, width=4))
    final_ratio = cute.make_rmem_tensor_like(row_sum, cutlass.Float32)
    lse = cute.make_rmem_tensor_like(row_sum, cutlass.Float32)

    for m in cutlass.range(cute.size(row_sum), unroll_full=True):
        softmax_scale_log2e = softmax_scale_log2e_m[m]
        final_sum = row_sum[m]
        is_zero_or_nan = final_sum == 0.0 or final_sum != final_sum
        final_ratio[m] = cute.arch.rcp_approx(final_sum if not is_zero_or_nan else 1.0)

        ln2 = 0.693147180559945309417
        lse[m] = (
            -cutlass.Float32.inf if is_zero_or_nan else (row_max[m] * softmax_scale_log2e + cute.math.log2(final_sum, fastmath=True) - exp_scale_log2) * ln2
        )

    return final_ratio, lse


@cute.jit
def _rescale_o_for_next_acc_fp8(
    pv_tiled_mma: cute.TiledMma,
    tOrO: cute.ThrMma,
    prev_ratio_m: cute.Tensor,
):
    tOrO_mn = layout_utils.reshape_acc_to_mn(tOrO)
    for m in cutlass.range(cute.size(prev_ratio_m), unroll_full=True):
        tOrO_mn[m, None].store(tOrO_mn[m, None].load() * prev_ratio_m[m])


@cute.jit
def _rescale_o_with_sage_v_scale_fp8(
    tOrO: cute.Tensor,
    tOcO: cute.Tensor,
    final_ratio_m: cute.Tensor,
    gVScale: cute.Tensor,
) -> None:
    """Normalize O and apply the per-channel Sage V descale."""
    tOrO_mn = layout_utils.reshape_acc_to_mn(tOrO)
    tOcO_mn = layout_utils.reshape_acc_to_mn(tOcO)
    for n in cutlass.range(cute.size(tOrO_mn, mode=[1]), unroll_full=True):
        v_scale = gVScale[tOcO_mn[0, n][1]]
        for m in cutlass.range(cute.size(final_ratio_m), unroll_full=True):
            tOrO_mn[m, n] *= final_ratio_m[m] * v_scale
