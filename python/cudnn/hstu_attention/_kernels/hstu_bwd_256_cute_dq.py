# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Type, Tuple, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, Int64, Float32

from .hstu_bwd_256_cute_utils import (
    compute_sm100_fmha_grid as compute_grid,
    HSTUFusedMask as FusedMask,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
    SM100_TMEM_CAPACITY_COLUMNS,
    Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler,
    Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
)
from .utils import (
    fma_packed_f32x2,
    mul_packed_f32x2,
    sub_packed_f32x2,
    tanhf,
)
from .block_sparsity import (
    HSTUBlockSparseTensors,
    get_q2k_block_for_iteration,
    get_q2k_block_sparse_consumer_row,
)


class BlackwellFusedMultiHeadAttentionBackwardDQKernel:
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        is_causal: bool,
        window_size_left: int | None,
        window_size_right: int | None,
        skip_residual_mask: bool = False,
        is_arbitrary: bool = False,
        func_num: int = 0,
        use_auto_block_metadata: bool = False,
    ):
        self.acc_dtype = acc_dtype
        self.is_causal = is_causal
        # Normalize negative window sizes to the unbounded-window sentinel.
        window_size_left = None if (window_size_left is None or window_size_left < 0) else cutlass.Int32(window_size_left)
        window_size_right = None if (window_size_right is None or window_size_right < 0) else cutlass.Int32(window_size_right)
        self.window_size_left = None if self.is_causal else window_size_left
        self.window_size_right = cutlass.Int32(0) if self.is_causal else window_size_right
        self.is_local = (not self.is_causal) and (self.window_size_left is not None or self.window_size_right is not None)
        self.skip_residual_mask = skip_residual_mask
        self.is_arbitrary = is_arbitrary
        self.func_num = func_num
        self.use_auto_block_metadata = use_auto_block_metadata
        assert not self.is_arbitrary or not (self.is_causal or self.is_local)
        assert not self.is_arbitrary or (self.func_num > 0 and self.func_num % 2 == 1)
        assert not self.use_auto_block_metadata or self.is_arbitrary
        assert mma_tiler[0] == 128 and mma_tiler[1] == 128, "Only 128x128 tile impl is supported"
        assert mma_tiler[2] == 256, "Only 256 is supported for 128x128 tile impl"
        self.cta_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = (
            2 * mma_tiler[0],
            mma_tiler[1],
            self.cta_tiler[2],
        )
        self.dov_mma_tiler = self.qk_mma_tiler
        self.dsk_mma_tiler = (
            2 * mma_tiler[0],
            self.cta_tiler[2],
            mma_tiler[1],
        )

        self.dsk_block_tiler = (
            self.dsk_mma_tiler[0] // 2,
            self.dsk_mma_tiler[1],
            self.dsk_mma_tiler[2],
        )
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_dov = self.cta_tiler[2] // self.dov_mma_tiler[2]
        self.iterations_dsk = self.cta_tiler[2] // self.dsk_mma_tiler[1]
        self.cluster_shape_mn = (2, 1)
        self.tmem_warp_shape_mn = (4, 1)
        self.use_semantic_trip_range = self.is_causal or self.is_local

        self.compute_warp_ids = (0, 1, 2, 3)
        self.epilogue_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_id = (10, 11)
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.num_compute_warps = len(self.compute_warp_ids)

        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.compute_sync_bar_id = 2

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.compute_warp_ids,  # this is to get a round num threads
                *self.epilogue_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.empty_warp_id,
            )
        )

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )

        self.tmem_s_offset = 0
        self.tmem_dp_offset = 128
        self.tmem_dq_offset = 256

        self.num_regs_compute = 256
        self.num_regs_epilogue = 160
        self.num_regs_other = 32

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.q_stage = self.iterations_qk
        self.k_stage = self.iterations_qk
        self.do_stage = self.iterations_dov
        self.v_stage = self.iterations_dov
        self.kt_stage = 1
        self.qk_acc_stage = 1
        self.dov_acc_stage = 1
        self.dsk_acc_stage = 1
        self.epi_stage = 1

    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        dq_tensor: cute.Tensor,
        do_tensor: cute.Tensor,
        cum_seqlen_q: cute.Tensor,
        cum_seqlen_k: cute.Tensor,
        func: Optional[cute.Tensor],
        max_seqlen_q: Int32,
        scale_softmax: cutlass.Float32,
        scale_gradient: cutlass.Float32,
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        stream: cuda.CUstream,
    ):
        # Infer shape metadata from normalized 5D tensors (B, S, H_k, H_r, D),
        # similar to the dedicated hd256 forward path.
        # Packed tensors expose total tokens in shape[1].  Grid geometry must
        # use the maximum per-sequence length or it over-launches by roughly B.
        s_q = max_seqlen_q
        d = q_tensor.shape[4]
        h_k = q_tensor.shape[2]
        h_r = q_tensor.shape[3]
        b = cum_seqlen_q.shape[0] - 1
        d64 = cute.assume(Int64(d), divby=128)
        h_r64 = Int64(h_r)
        h_k64 = Int64(h_k)
        # Packed-varlen representation uses batch-dim = 1 and sequence-dim = total_{q,k}.
        # Keep the *physical* sequence extent in the tensor layouts so that applying
        # `cuseqlen_*` offsets stays within the tensor domain.
        s_q_total = q_tensor.shape[1]
        s_k_total = k_tensor.shape[1]
        stride_b_qo = 0
        stride_b_kv = 0

        # (s, d, ((h_r, h_k), b))
        q_layout = cute.make_layout(
            (s_q_total, d, ((h_r, h_k), b)),
            stride=(d64 * h_r64 * h_k64, 1, ((d64, d64 * h_r64), stride_b_qo)),
        )
        q = cute.make_tensor(q_tensor.iterator, q_layout)
        # (s, d, ((h_r, h_k), b))
        do_layout = cute.make_layout(
            (s_q_total, d, ((h_r, h_k), b)),
            stride=(d64 * h_r64 * h_k64, 1, ((d64, d64 * h_r64), stride_b_qo)),
        )
        do = cute.make_tensor(do_tensor.iterator, do_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k_layout = cute.make_layout(
            (s_k_total, d, ((h_r, h_k), b)),
            stride=(d64 * h_k64, 1, ((0, d64), stride_b_kv)),
        )
        k = cute.make_tensor(k_tensor.iterator, k_layout)
        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        kt_layout = cute.make_layout(
            (d, s_k_total, ((h_r, h_k), b)),
            stride=(1, d64 * h_k64, ((0, d64), stride_b_kv)),
        )
        kt = cute.make_tensor(k_tensor.iterator, kt_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v_layout = cute.make_layout(
            (s_k_total, d, ((h_r, h_k), b)),
            stride=(d64 * h_k64, 1, ((0, d64), stride_b_kv)),
        )
        v = cute.make_tensor(v_tensor.iterator, v_layout)
        # (s, d, ((h_r, h_k), b))
        dq_layout = cute.make_layout(
            (s_q_total, d, ((h_r, h_k), b)),
            stride=(d64 * h_r64 * h_k64, 1, ((d64, d64 * h_r64), stride_b_qo)),
        )
        dq = cute.make_tensor(dq_tensor.iterator, dq_layout)

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.do_dtype = do.element_type
        if cutlass.const_expr(self.use_auto_block_metadata):
            assert isinstance(block_sparse_tensors, HSTUBlockSparseTensors)
            assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16)

        self.tile_sched_params, grid = compute_grid(
            (s_q, dq.shape[1], dq.shape[2]),
            self.cta_tiler,
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.do_major_mode = utils.LayoutEnum.from_tensor(do).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.dq_layout = utils.LayoutEnum.from_tensor(dq)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.do_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        if cutlass.const_expr(self.q_dtype != self.do_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.do_dtype}")

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        # the intermediate tensor p is from tmem & k-major
        ds_source = tcgen05.OperandSource.TMEM
        ds_major_mode = tcgen05.OperandMajorMode.K
        k_trans_major_mode = tcgen05.OperandMajorMode.MN
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        dov_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.do_dtype,
            self.do_major_mode,
            self.v_major_mode,
            self.acc_dtype,
            cta_group,
            self.dov_mma_tiler[:2],
        )
        dsk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            ds_major_mode,
            k_trans_major_mode,
            self.acc_dtype,
            cta_group,
            self.dsk_mma_tiler[:2],
            ds_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.dsk_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )
        do_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dov_tiled_mma,
            self.dov_mma_tiler,
            self.do_dtype,
            self.do_stage,
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dov_tiled_mma,
            self.dov_mma_tiler,
            self.v_dtype,
            self.v_stage,
        )
        ds_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            dsk_tiled_mma,
            self.dsk_mma_tiler,
            self.q_dtype,
            self.qk_acc_stage,
        )
        ds_tmem_layout = cute.select(ds_tmem_layout_staged, mode=[0, 1, 2])
        kt_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dsk_tiled_mma,
            self.dsk_mma_tiler,
            self.k_dtype,
            self.dsk_acc_stage,
        )

        # TMA load for Q
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for dO
        do_smem_layout = cute.select(do_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            do,
            do_smem_layout,
            self.dov_mma_tiler,
            dov_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.dov_mma_tiler,
            dov_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for KT
        kt_smem_layout = cute.select(kt_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            kt,
            kt_smem_layout,
            self.dsk_mma_tiler,
            dsk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        do_copy_size = cute.size_in_bytes(self.do_dtype, do_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        kt_copy_size = cute.size_in_bytes(self.k_dtype, kt_smem_layout)
        self.tma_copy_q_bytes = q_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_k_bytes = k_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_do_bytes = do_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_v_bytes = v_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_kt_bytes = kt_copy_size * cute.size(qk_tiled_mma.thr_id.shape)

        @cute.struct
        class SharedStorage:
            # TMA G2S load barriers: LOAD warp (producer) -> MMA warp (consumer)
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]  # load_q_{producer,consumer}
            load_do_mbar_ptr: cute.struct.MemRange[Int64, self.do_stage * 2]  # load_do_{producer,consumer}
            load_k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2]  # load_k_{producer,consumer}
            load_kt_mbar_ptr: cute.struct.MemRange[Int64, self.kt_stage * 2]  # load_kt_{producer,consumer}
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2]  # load_v_{producer,consumer}
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            mma_dp_mbar_ptr: cute.struct.MemRange[Int64, self.dov_acc_stage * 2]
            mma_dq_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2]
            ds_mma_mbar_ptr: cute.struct.MemRange[Int64, self.dsk_acc_stage * 2]
            # A CTA-wide "TMEM lifetime" barrier used to safely deallocate TMEM after all users finish.
            tmem_dealloc_mbar_ptr: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage

        grid = cute.round_up(grid, self.cluster_shape_mnk)
        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            dov_tiled_mma,
            dsk_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_do,
            tma_tensor_do,
            tma_atom_kt,
            tma_tensor_kt,
            dq,
            cum_seqlen_q,
            cum_seqlen_k,
            func,
            block_sparse_tensors,
            scale_softmax,
            scale_gradient,
            self.window_size_left,
            self.window_size_right,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            v_smem_layout_staged,
            do_smem_layout_staged,
            kt_smem_layout_staged,
            ds_tmem_layout,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        dov_tiled_mma: cute.TiledMma,
        dsk_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        mdO_qdl: cute.Tensor,
        tma_atom_kt: cute.CopyAtom,
        mK_dkl: cute.Tensor,
        mdQ_qdl: cute.Tensor,
        cum_seqlen_q: cute.Tensor,
        cum_seqlen_k: cute.Tensor,
        func: Optional[cute.Tensor],
        block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        scale_softmax: Float32,
        scale_gradient: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        do_smem_layout_staged: cute.ComposedLayout,
        kt_smem_layout_staged: cute.ComposedLayout,
        ds_tmem_layout_staged: cute.ComposedLayout,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        # llvm.inline_asm(
        #     None,
        #     [],
        #     '.pragma "global knob CommonIntoMultiBlockLoop=1";',
        #     "",
        #     has_side_effects=True,
        #     is_align_stack=False,
        #     asm_dialect=llvm.AsmDialect.AD_ATT,
        # )

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_do)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kt)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_do_producer, load_do_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.do_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_do_bytes,
            barrier_storage=storage.load_do_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_kt_producer, load_kt_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kt_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kt_bytes,
            barrier_storage=storage.load_kt_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_dp_producer, mma_dp_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.dov_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_dp_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        ds_mma_producer, ds_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.dsk_acc_stage,
            producer_group=make_thread_cooperative_group(
                len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.ds_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_dq_producer, mma_dq_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.epilogue_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_dq_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        tmem.allocate(self.tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        # K and V now use separate memory since we removed the transform stage
        sV = smem.allocate_tensor(
            element_type=self.v_dtype,
            layout=v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sdO = smem.allocate_tensor(
            element_type=self.do_dtype,
            layout=do_smem_layout_staged.outer,
            swizzle=do_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sKT = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=kt_smem_layout_staged.outer,
            swizzle=kt_smem_layout_staged.inner,
            byte_alignment=128,
        )
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        dov_thr_mma = dov_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        dsk_thr_mma = dsk_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tdPrdO = dov_thr_mma.make_fragment_A(sdO)
        tdPrV = dov_thr_mma.make_fragment_B(sV)
        tdQrKT = dsk_thr_mma.make_fragment_B(sKT)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        dov_acc_shape = dov_thr_mma.partition_shape_C((self.dov_mma_tiler[0], self.dov_mma_tiler[1]))
        tdPtdP = dov_thr_mma.make_fragment_C(cute.append(dov_acc_shape, self.dov_acc_stage))
        dsk_acc_shape = dsk_thr_mma.partition_shape_C((self.dsk_mma_tiler[0], self.dsk_mma_tiler[1]))
        tdQtdQ = dsk_thr_mma.make_fragment_C(dsk_acc_shape)
        tdQtdQ_layout = cute.append(
            tdQtdQ.layout,
            cute.make_layout(
                self.iterations_dsk,
                stride=self.dsk_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tdPtdP = cute.make_tensor(tdPtdP.iterator + self.tmem_dp_offset, tdPtdP.layout)
        tdQtdQ_staged = cute.make_tensor(tdQtdQ.iterator + self.tmem_dq_offset, tdQtdQ_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for _i in cutlass.range_constexpr(len(self.empty_warp_id)):
            if warp_idx == self.empty_warp_id[_i]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        blk_idx = cute.arch.block_idx()
        tile_sched = FmhaStaticTileScheduler(tile_sched_params, blk_idx[0], blk_idx, cute.arch.grid_dim())
        work_tile = tile_sched.initial_work_tile_info()

        # Cluster wait
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                batch_coord = curr_block_coord[2][1]
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = mK_kdl.shape[0]
                cuseqlen_q = Int32(0)
                cuseqlen_k = Int32(0)
                seqlen_kv_loop_start = Int32(0)
                seqlen_kv_loop_steps = Int32(0)
                mask_block_cnt = None
                mask_block_idx = None
                full_block_cnt = None
                full_block_idx = None
                block_offset = (
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    ((Int32(0), Int32(0)), Int32(0)),
                )
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    block_offset = (
                        cuseqlen_q,
                        cuseqlen_k,
                        Int32(0),
                        ((Int32(0), Int32(0)), Int32(0)),
                    )
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                if cutlass.const_expr(self.use_auto_block_metadata):
                    assert isinstance(block_sparse_tensors, HSTUBlockSparseTensors)
                    (
                        seqlen_kv_loop_steps,
                        mask_block_cnt,
                        mask_block_idx,
                        full_block_cnt,
                        full_block_idx,
                    ) = get_q2k_block_sparse_consumer_row(
                        block_sparse_tensors,
                        batch_coord,
                        mma_block_coord[0],
                    )
                elif not continue_cond:
                    (
                        seqlen_kv_loop_start,
                        seqlen_kv_loop_steps,
                    ) = FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                    )
                has_work = not continue_cond and seqlen_kv_loop_steps > 0
                if has_work:
                    mQ_qdl_ = cute.domain_offset(cute.select(block_offset, mode=[0, 2, 3]), mQ_qdl)
                    mK_kdl_ = cute.domain_offset(cute.select(block_offset, mode=[1, 2, 3]), mK_kdl)
                    mdO_qdl_ = cute.domain_offset(cute.select(block_offset, mode=[0, 2, 3]), mdO_qdl)
                    mV_dkl_ = cute.domain_offset(cute.select(block_offset, mode=[1, 2, 3]), mV_dkl)
                    mK_dkl_ = cute.domain_offset(cute.select(block_offset, mode=[2, 1, 3]), mK_dkl)
                    # Local tile partition global tensors
                    q_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                    # (bM, bK, loopM, loopK, loopL)
                    gQ_qdl = cute.flat_divide(mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        block_in_cluster_coord_vmnk[2],
                        q_cta_layout,
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    k_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                    gK_kdl = cute.flat_divide(mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2]))
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        block_in_cluster_coord_vmnk[1],
                        k_cta_layout,
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK_kdl, 0, 3),
                    )
                    do_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                    # (bM, bK, loopM, loopK, loopL)
                    gdO_qdl = cute.flat_divide(mdO_qdl_, cute.select(self.dov_mma_tiler, mode=[0, 2]))
                    tdPgdO_qdl = dov_thr_mma.partition_A(gdO_qdl)
                    tdOsdO, tdOgdO_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_do,
                        block_in_cluster_coord_vmnk[2],
                        do_cta_layout,
                        cute.group_modes(sdO, 0, 3),
                        cute.group_modes(tdPgdO_qdl, 0, 3),
                    )
                    v_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                    gV_dkl = cute.flat_divide(mV_dkl_, cute.select(self.dov_mma_tiler, mode=[1, 2]))
                    tSgV_dkl = dov_thr_mma.partition_B(gV_dkl)
                    tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        block_in_cluster_coord_vmnk[1],
                        v_cta_layout,
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tSgV_dkl, 0, 3),
                    )
                    # kt layout
                    kt_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                    gK_dkl = cute.flat_divide(mK_dkl_, cute.select(self.dsk_mma_tiler, mode=[1, 2]))
                    tdQgK_dkl = dsk_thr_mma.partition_B(gK_dkl)
                    tKTsKT, tKgK_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_kt,
                        block_in_cluster_coord_vmnk[1],
                        kt_cta_layout,
                        cute.group_modes(sKT, 0, 3),
                        cute.group_modes(tdQgK_dkl, 0, 3),
                    )
                    # ((atom_v, rest_v), RestK)
                    tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestK)
                    tdOgdO = tdOgdO_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tKTgKT = tKgK_dkl[None, None, None, mma_block_coord[2]]

                    # Q
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        q_handle = load_q_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[None, iter],
                            tQsQ[None, q_handle.index],
                            tma_bar_ptr=q_handle.barrier,
                        )
                    # dO
                    for iter in cutlass.range(self.iterations_dov, unroll=1):
                        do_handle = load_do_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_do,
                            tdOgdO[None, iter],
                            tdOsdO[None, do_handle.index],
                            tma_bar_ptr=do_handle.barrier,
                        )

                    for i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        kv_coord = seqlen_kv_loop_start + i
                        if cutlass.const_expr(self.use_auto_block_metadata):
                            kv_coord, _ = get_q2k_block_for_iteration(
                                mask_block_cnt,
                                mask_block_idx,
                                full_block_cnt,
                                full_block_idx,
                                i,
                            )
                        # Ki
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            k_handle = load_k_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, kv_coord, iter],
                                tKsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                        # Vi
                        for iter in cutlass.range(self.iterations_dov, unroll=1):
                            v_handle = load_v_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tVgV[None, kv_coord, iter],
                                tVsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )
                        # KTi
                        for iter in cutlass.range(self.iterations_dsk, unroll=1):
                            kt_handle = load_kt_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_kt,
                                tKTgKT[None, iter, kv_coord],
                                tKTsKT[None, kt_handle.index],
                                tma_bar_ptr=kt_handle.barrier,
                            )
                work_tile = tile_sched.advance_to_next_work()
                # End of static tile scheduler loop
            load_k_producer.tail()
            load_v_producer.tail()
            load_kt_producer.tail()
            load_q_producer.tail()
            load_do_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = mK_kdl.shape[0]
                seqlen_kv_loop_start = Int32(0)
                seqlen_kv_loop_steps = Int32(0)
                batch_coord = curr_block_coord[2][1]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )

                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    if cutlass.const_expr(self.use_auto_block_metadata):
                        assert isinstance(block_sparse_tensors, HSTUBlockSparseTensors)
                        seqlen_kv_loop_steps = get_q2k_block_sparse_consumer_row(
                            block_sparse_tensors,
                            batch_coord,
                            mma_block_coord[0],
                        )[0]
                    else:
                        (
                            seqlen_kv_loop_start,
                            seqlen_kv_loop_steps,
                        ) = FusedMask.get_trip_start_count_via_block_info(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            self.is_local,
                            window_size_left,
                            window_size_right,
                        )

                has_work = not continue_cond and seqlen_kv_loop_steps > 0
                if has_work:
                    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                    is_leader_cta = cta_rank_in_cluster % 2 == 0
                    # dq_handle = mma_dq_producer.acquire_and_advance()
                    load_q_releaser = load_q_consumer.clone()
                    load_do_releaser = load_do_consumer.clone()
                    dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    num_innerloop = 8

                    if is_leader_cta:
                        dq_handle = mma_dq_producer.acquire_and_advance()
                        if seqlen_kv_loop_steps > 1:
                            # QK0
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter]

                                k_handle = load_k_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                qk_tiled_mma,
                                                tStS_slice,
                                                tSrQ_slice[kphase_coord],
                                                tSrK_trans_slice[kphase_coord],
                                                tStS_slice,
                                            )
                                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                            cute.arch.fence_view_async_tmem_store()
                            s_handle.commit()

                            # dOV0
                            dp_handle = mma_dp_producer.acquire_and_advance()
                            tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_dov, unroll=1):
                                load_do_consumer.wait_and_advance()
                                tdPrdO_slice = tdPrdO[None, None, None, iter]
                                v_handle = load_v_consumer.wait_and_advance()
                                tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tdPrdO_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                dov_tiled_mma,
                                                tdPtdP_slice,
                                                tdPrdO_slice[kphase_coord],
                                                tdPrV_trans_slice[kphase_coord],
                                                tdPtdP_slice,
                                            )
                                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dov_tiled_mma,
                                            tdPtdP_slice,
                                            tdPrdO_slice[kphase_coord],
                                            tdPrV_trans_slice[kphase_coord],
                                            tdPtdP_slice,
                                        )
                                        dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                            cute.arch.fence_view_async_tmem_store()
                            dp_handle.commit()

                            for i in cutlass.range(1, seqlen_kv_loop_steps - 1, 1, unroll=1):
                                # QKi
                                s_handle = mma_s_producer.acquire_and_advance()

                                tStS_slice = tStS[None, None, None, s_handle.index]
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                for iter in cutlass.range(self.iterations_qk, unroll=1):
                                    tSrQ_slice = tSrQ[None, None, None, iter]
                                    k_handle = load_k_consumer.wait_and_advance()
                                    tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                    num_kphases = cute.size(tSrQ_slice, mode=[2])
                                    if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                        num_outer_iter = num_kphases // num_innerloop
                                        for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                            for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                                kphase_coord = (
                                                    None,
                                                    None,
                                                    outer_iter * num_innerloop + kphase_idx,
                                                )
                                                cute.gemm(
                                                    qk_tiled_mma,
                                                    tStS_slice,
                                                    tSrQ_slice[kphase_coord],
                                                    tSrK_trans_slice[kphase_coord],
                                                    tStS_slice,
                                                )
                                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    else:
                                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                            kphase_coord = (None, None, kphase_idx)
                                            cute.gemm(
                                                qk_tiled_mma,
                                                tStS_slice,
                                                tSrQ_slice[kphase_coord],
                                                tSrK_trans_slice[kphase_coord],
                                                tStS_slice,
                                            )
                                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    k_handle.release()
                                s_handle.commit()

                                # dSKTi
                                ds_handle = ds_mma_consumer.wait_and_advance()
                                dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                                for iter in cutlass.range(self.iterations_dsk, unroll=1):
                                    kt_handle = load_kt_consumer.wait_and_advance()
                                    dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                                    tdQtdQ_slice = tdQtdQ_staged[None, None, None, iter]
                                    tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                                    tdS = cute.make_tensor(tdStdS_slice.iterator, ds_tmem_layout_staged.outer)
                                    tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                                    tdQrdS_slice = cute.make_tensor(
                                        cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                        tdQrdS.layout,
                                    )

                                    tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                                    num_kphases = cute.size(tdQrKT_slice, mode=[2])
                                    if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                        num_outer_iter = num_kphases // num_innerloop
                                        for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                            for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                                kphase_coord = (
                                                    None,
                                                    None,
                                                    outer_iter * num_innerloop + kphase_idx,
                                                )
                                                cute.gemm(
                                                    dsk_tiled_mma,
                                                    tdQtdQ_slice,
                                                    tdQrdS_slice[kphase_coord],
                                                    tdQrKT_slice[kphase_coord],
                                                    tdQtdQ_slice,
                                                )
                                                dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    else:
                                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                            kphase_coord = (None, None, kphase_idx)
                                            cute.gemm(
                                                dsk_tiled_mma,
                                                tdQtdQ_slice,
                                                tdQrdS_slice[kphase_coord],
                                                tdQrKT_slice[kphase_coord],
                                                tdQtdQ_slice,
                                            )
                                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    kt_handle.release()
                                ds_handle.release()

                                # dOVi
                                dp_handle = mma_dp_producer.acquire_and_advance()
                                tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                                dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                for iter in cutlass.range(self.iterations_dov, unroll=1):
                                    tdPrdO_slice = tdPrdO[None, None, None, iter]
                                    v_handle = load_v_consumer.wait_and_advance()
                                    tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                                    num_kphases = cute.size(tdPrdO_slice, mode=[2])
                                    if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                        num_outer_iter = num_kphases // num_innerloop
                                        for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                            for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                                kphase_coord = (
                                                    None,
                                                    None,
                                                    outer_iter * num_innerloop + kphase_idx,
                                                )
                                                cute.gemm(
                                                    dov_tiled_mma,
                                                    tdPtdP_slice,
                                                    tdPrdO_slice[kphase_coord],
                                                    tdPrV_trans_slice[kphase_coord],
                                                    tdPtdP_slice,
                                                )
                                                dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    else:
                                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                            kphase_coord = (None, None, kphase_idx)
                                            cute.gemm(
                                                dov_tiled_mma,
                                                tdPtdP_slice,
                                                tdPrdO_slice[kphase_coord],
                                                tdPrV_trans_slice[kphase_coord],
                                                tdPtdP_slice,
                                            )
                                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    v_handle.release()
                                dp_handle.commit()

                            # QKend
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_k_consumer.wait_and_advance()

                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]

                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                qk_tiled_mma,
                                                tStS_slice,
                                                tSrQ_slice[kphase_coord],
                                                tSrK_trans_slice[kphase_coord],
                                                tStS_slice,
                                            )
                                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                            # dSKTend - 1
                            ds_handle = ds_mma_consumer.wait_and_advance()
                            dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            for iter in cutlass.range(self.iterations_dsk, unroll=1):
                                kt_handle = load_kt_consumer.wait_and_advance()
                                dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                                tdQtdQ_slice = tdQtdQ_staged[None, None, None, iter]
                                tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                                tdS = cute.make_tensor(tdStdS_slice.iterator, ds_tmem_layout_staged.outer)
                                tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                                tdQrdS_slice = cute.make_tensor(
                                    cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                    tdQrdS.layout,
                                )

                                tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                                num_kphases = cute.size(tdQrKT_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                dsk_tiled_mma,
                                                tdQtdQ_slice,
                                                tdQrdS_slice[kphase_coord],
                                                tdQrKT_slice[kphase_coord],
                                                tdQtdQ_slice,
                                            )
                                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dsk_tiled_mma,
                                            tdQtdQ_slice,
                                            tdQrdS_slice[kphase_coord],
                                            tdQrKT_slice[kphase_coord],
                                            tdQtdQ_slice,
                                        )
                                        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                kt_handle.release()
                            ds_handle.release()

                            # dOVend
                            dp_handle = mma_dp_producer.acquire_and_advance()
                            tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_dov, unroll=1):
                                tdPrdO_slice = tdPrdO[None, None, None, iter]
                                v_handle = load_v_consumer.wait_and_advance()
                                tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tdPrdO_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                dov_tiled_mma,
                                                tdPtdP_slice,
                                                tdPrdO_slice[kphase_coord],
                                                tdPrV_trans_slice[kphase_coord],
                                                tdPtdP_slice,
                                            )
                                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dov_tiled_mma,
                                            tdPtdP_slice,
                                            tdPrdO_slice[kphase_coord],
                                            tdPrV_trans_slice[kphase_coord],
                                            tdPtdP_slice,
                                        )
                                        dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                                load_do_releaser.release()
                                load_do_releaser.advance()
                            dp_handle.commit()
                            # dSKTend
                            ds_handle = ds_mma_consumer.wait_and_advance()
                            dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            for iter in cutlass.range(self.iterations_dsk, unroll=1):
                                kt_handle = load_kt_consumer.wait_and_advance()
                                dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                                tdQtdQ_slice = tdQtdQ_staged[None, None, None, iter]
                                tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                                tdS = cute.make_tensor(tdStdS_slice.iterator, ds_tmem_layout_staged.outer)
                                tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                                tdQrdS_slice = cute.make_tensor(
                                    cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                    tdQrdS.layout,
                                )

                                tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                                num_kphases = cute.size(tdQrKT_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                dsk_tiled_mma,
                                                tdQtdQ_slice,
                                                tdQrdS_slice[kphase_coord],
                                                tdQrKT_slice[kphase_coord],
                                                tdQtdQ_slice,
                                            )
                                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dsk_tiled_mma,
                                            tdQtdQ_slice,
                                            tdQrdS_slice[kphase_coord],
                                            tdQrKT_slice[kphase_coord],
                                            tdQtdQ_slice,
                                        )
                                        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                kt_handle.release()
                            ds_handle.release()
                        else:
                            # QK0
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_k_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                qk_tiled_mma,
                                                tStS_slice,
                                                tSrQ_slice[kphase_coord],
                                                tSrK_trans_slice[kphase_coord],
                                                tStS_slice,
                                            )
                                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                            # dOV0
                            dp_handle = mma_dp_producer.acquire_and_advance()
                            tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_dov, unroll=1):
                                load_do_consumer.wait_and_advance()
                                tdPrdO_slice = tdPrdO[None, None, None, iter]
                                v_handle = load_v_consumer.wait_and_advance()
                                tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tdPrdO_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                dov_tiled_mma,
                                                tdPtdP_slice,
                                                tdPrdO_slice[kphase_coord],
                                                tdPrV_trans_slice[kphase_coord],
                                                tdPtdP_slice,
                                            )
                                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dov_tiled_mma,
                                            tdPtdP_slice,
                                            tdPrdO_slice[kphase_coord],
                                            tdPrV_trans_slice[kphase_coord],
                                            tdPtdP_slice,
                                        )
                                        dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                                load_do_releaser.release()
                                load_do_releaser.advance()
                            dp_handle.commit()

                            # dSKT0
                            ds_handle = ds_mma_consumer.wait_and_advance()
                            dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            for iter in cutlass.range(self.iterations_dsk, unroll=1):
                                kt_handle = load_kt_consumer.wait_and_advance()
                                dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                                tdQtdQ_slice = tdQtdQ_staged[None, None, None, iter]
                                tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                                tdS = cute.make_tensor(tdStdS_slice.iterator, ds_tmem_layout_staged.outer)
                                tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                                tdQrdS_slice = cute.make_tensor(
                                    cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                    tdQrdS.layout,
                                )

                                tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                                num_kphases = cute.size(tdQrKT_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(num_innerloop, unroll_full=True):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                dsk_tiled_mma,
                                                tdQtdQ_slice,
                                                tdQrdS_slice[kphase_coord],
                                                tdQrKT_slice[kphase_coord],
                                                tdQtdQ_slice,
                                            )
                                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dsk_tiled_mma,
                                            tdQtdQ_slice,
                                            tdQrdS_slice[kphase_coord],
                                            tdQrKT_slice[kphase_coord],
                                            tdQtdQ_slice,
                                        )
                                        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                kt_handle.release()
                            ds_handle.release()
                        dq_handle.commit()
                work_tile = tile_sched.advance_to_next_work()
            # End of static tile scheduler loop
            mma_s_producer.tail()
            mma_dp_producer.tail()
            mma_dq_producer.tail()

        # Softmax and dSoftmax warp
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_compute)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = mK_kdl.shape[0]
                cuseqlen_q = Int32(0)
                start_count = Int32(0)
                trip_count = Int32(0)
                mask_block_cnt = None
                mask_block_idx = None
                full_block_cnt = None
                full_block_idx = None
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                    self.qk_mma_tiler[0],
                    mma_block_coord[0],
                    seqlen_q,
                )
                if not continue_cond:
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                if cutlass.const_expr(self.use_auto_block_metadata):
                    assert isinstance(block_sparse_tensors, HSTUBlockSparseTensors)
                    (
                        trip_count,
                        mask_block_cnt,
                        mask_block_idx,
                        full_block_cnt,
                        full_block_idx,
                    ) = get_q2k_block_sparse_consumer_row(
                        block_sparse_tensors,
                        batch_coord,
                        mma_block_coord[0],
                    )
                elif not continue_cond:
                    start_count, trip_count = FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                    )
                has_work = not continue_cond and trip_count > 0
                if has_work:
                    if cutlass.const_expr(self.use_semantic_trip_range):
                        n_block_min_causal_local_mask, n_block_min_before_local_mask = FusedMask.get_trip_mask_bounds_via_block_info(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            self.is_local,
                            window_size_left,
                            window_size_right,
                        )

                    cS_base = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                    cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base)
                    cdP_base = cute.make_identity_tensor((self.dov_mma_tiler[0], self.dov_mma_tiler[1]))
                    cdP = cute.domain_offset((mma_block_coord[0] * self.dov_mma_tiler[0], 0), cdP_base)
                    for iteration in cutlass.range(0, trip_count, 1, unroll=1):
                        step = start_count + iteration
                        is_mask_block = False
                        if cutlass.const_expr(self.use_auto_block_metadata):
                            step, is_mask_block = get_q2k_block_for_iteration(
                                mask_block_cnt,
                                mask_block_idx,
                                full_block_cnt,
                                full_block_idx,
                                iteration,
                            )
                        cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
                        tScS_iter = qk_thr_mma.partition_C(cS_iter)

                        cdP_iter = cute.domain_offset((0, step * self.dov_mma_tiler[1]), cdP)

                        tdPcdP_iter = dov_thr_mma.partition_C(cdP_iter)

                        mask_args = (window_size_left, window_size_right)
                        value_args = (
                            seqlen_q,
                            seqlen_k,
                            cuseqlen_q,
                            scale_softmax,
                            scale_gradient,
                        )
                        tensor_args = (tStS, tScS_iter, tdPtdP, tdPcdP_iter, func)
                        pipeline_args = (
                            mma_s_consumer,
                            mma_dp_consumer,
                            ds_mma_producer,
                        )

                        # Si, dPi -> dSi.  Separate compile-time call sites keep
                        # FULL interior free of predicate fragments and func
                        # loads, while FULL tails retain only the seqlen guard.
                        mask_kind = Int32(0)
                        if cutlass.const_expr(self.use_auto_block_metadata):
                            residual_q = (mma_block_coord[0] + 1) * self.qk_mma_tiler[0] > seqlen_q
                            residual_k = (step + 1) * self.qk_mma_tiler[1] > seqlen_k
                            if is_mask_block:
                                mask_kind = Int32(2)
                            elif residual_q or residual_k:
                                mask_kind = Int32(1)
                        else:
                            if cutlass.const_expr(self.is_arbitrary):
                                need_apply_mask = True
                            elif cutlass.const_expr(self.use_semantic_trip_range):
                                need_apply_mask = step >= n_block_min_causal_local_mask or step < n_block_min_before_local_mask
                            else:
                                residual_q = (mma_block_coord[0] + 1) * self.qk_mma_tiler[0] > seqlen_q
                                residual_k = (step + 1) * self.qk_mma_tiler[1] > seqlen_k
                                need_apply_mask = residual_q or residual_k
                            if need_apply_mask:
                                mask_kind = Int32(2)
                        mma_s_consumer, mma_dp_consumer, ds_mma_producer = self.compute_step(
                            mask_args,
                            value_args,
                            tensor_args,
                            pipeline_args,
                            mask_kind,
                        )
                work_tile = tile_sched.advance_to_next_work()
            ds_mma_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_epilogue)

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                # cute.printf("batch_coord={}", batch_coord)
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = mK_kdl.shape[0]
                continue_cond = False
                cuseqlen_q = Int32(0)
                trip_count = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )

                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    if cutlass.const_expr(self.use_auto_block_metadata):
                        assert isinstance(block_sparse_tensors, HSTUBlockSparseTensors)
                        trip_count = get_q2k_block_sparse_consumer_row(
                            block_sparse_tensors,
                            batch_coord,
                            mma_block_coord[0],
                        )[0]
                    else:
                        _, trip_count = FusedMask.get_trip_start_count_via_block_info(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            self.is_local,
                            window_size_left,
                            window_size_right,
                        )

                    mdQ_qdl_eff = mdQ_qdl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        block_offset_dQ = (
                            cuseqlen_q,
                            Int32(0),
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )
                        mdQ_qdl_eff = cute.domain_offset(cute.select(block_offset_dQ, mode=[0, 2, 3]), mdQ_qdl)

                    # (bM, bN, loopM, loopN, loopL)
                    gdQ_qdl = cute.flat_divide(mdQ_qdl_eff, cute.select(self.dsk_block_tiler, mode=[0, 1]))
                    cdQ_qdl = cute.flat_divide(
                        cute.make_identity_tensor(mdQ_qdl_eff.shape),
                        cute.select(self.dsk_block_tiler, mode=[0, 1]),
                    )

                    gdQ_staged = gdQ_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                    cdQ_staged = cdQ_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]

                    if trip_count > 0:
                        # dQ TMEM to GMEM
                        mma_dq_consumer = self.dQ_epilogue(
                            (seqlen_q, cuseqlen_q, mQ_qdl.shape[0], batch_coord),
                            (mma_dq_consumer, gdQ_staged, cdQ_staged, tdQtdQ_staged),
                            self.epi_tile,
                        )
                    else:
                        # This physical Q128 CTA owns the output tile.  The
                        # sparse row has neither MASK nor FULL blocks, so no
                        # dQ pipeline handoff exists and the owner writes zero
                        # directly to global memory.
                        self.dQ_epilogue_write_zero(
                            seqlen_q,
                            (gdQ_staged, cdQ_staged, tdQtdQ_staged),
                            self.epi_tile,
                        )
                work_tile = tile_sched.advance_to_next_work()
            # TMEM is released after every warp in this CTA leaves the work loop.

        # ///////////////////////////////////////////////////////////////////////////////
        #  Empty warps reg dealloc
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx > self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cooperative TMEM Deallocation (2CTA)
        # ///////////////////////////////////////////////////////////////////////////////
        # All warps in this CTA, including the scheduler warp, must finish before
        # the allocator performs its cross-CTA deallocation handshake.
        cute.arch.barrier(
            barrier_id=self.cta_sync_bar_id,
            number_of_threads=self.threads_per_cta,
        )
        # Empty metadata rows intentionally have no dQ pipeline handoff.  A
        # cluster rendezvous keeps the paired CTAs in the same TMEM lifetime
        # epoch before the two-CTA allocator performs its deallocation.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

        return

    @cute.jit
    def is_auto_score_valid(
        self,
        index_qk: cute.Coord,
        seqlen_q: Int32,
        seqlen_k: Int32,
        query_offset: Int32,
        apply_func: cutlass.Constexpr[bool],
        func: cute.Tensor,
    ) -> cutlass.Boolean:
        """Validate a tail/MASK score with compile-time func elimination."""

        index_q, index_k = index_qk
        valid = index_q < seqlen_q and index_k < seqlen_k
        if cutlass.const_expr(apply_func):
            if valid:
                func_row = query_offset + index_q
                value_valid = index_k < func[0, 0, func_row]
                for interval_idx in cutlass.range(self.func_num // 2, unroll_full=True):
                    interval_start = func[0, 2 * interval_idx + 1, func_row]
                    interval_end = func[0, 2 * interval_idx + 2, func_row]
                    value_valid = value_valid | ((index_k >= interval_start) & (index_k < interval_end))
                valid = valid and value_valid
        return valid

    @cute.jit
    def apply_auto_silu_derivative(
        self,
        scores: cute.Tensor,
        score_coords: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        query_offset: Int32,
        scale_softmax: Float32,
        func: cute.Tensor,
        mask_kind: cutlass.Constexpr[int],
    ) -> None:
        """Apply an unmasked, seqlen-only, or exact SiLU derivative."""

        endpoint_cache = None
        q_in_bounds = cutlass.Boolean(False)
        if cutlass.const_expr(mask_kind == 2 and self.func_num <= 7):
            # The D256 dQ TMEM load maps one thread to one Q row and all 128
            # K columns.  Prefetch that row's endpoints once instead of
            # reloading them for every score element.  Larger odd func counts
            # retain the scalar exact path to avoid an unbounded GPR cache.
            endpoint_cache = cute.make_rmem_tensor(
                (self.func_num,),
                cutlass.Int32,
            )
            for endpoint in cutlass.range_constexpr(
                self.func_num,
                unroll_full=True,
            ):
                endpoint_cache[endpoint] = Int32(0)
            index_q, _ = score_coords[0]
            q_in_bounds = index_q < seqlen_q
            if q_in_bounds:
                func_row = query_offset + index_q
                for endpoint in cutlass.range_constexpr(
                    self.func_num,
                    unroll_full=True,
                ):
                    endpoint_cache[endpoint] = func[0, endpoint, func_row]

        for k in cutlass.range(0, cute.size(scores), 2, unroll_full=True):
            x0, x1 = mul_packed_f32x2(
                (scores[k], scores[k + 1]),
                (scale_softmax, scale_softmax),
            )
            tanh0 = tanhf(x0 * 0.5)
            tanh1 = tanhf(x1 * 0.5)
            sigmoid0, sigmoid1 = fma_packed_f32x2(
                (0.5, 0.5),
                (tanh0, tanh1),
                (0.5, 0.5),
            )
            if cutlass.const_expr(mask_kind != 0):
                if cutlass.const_expr(mask_kind == 2):
                    if cutlass.const_expr(self.func_num <= 7):
                        assert isinstance(endpoint_cache, cute.Tensor)
                        _, index_k0 = score_coords[k]
                        _, index_k1 = score_coords[k + 1]
                        valid0 = q_in_bounds and index_k0 < seqlen_k
                        valid1 = q_in_bounds and index_k1 < seqlen_k
                        if valid0:
                            value_valid0 = index_k0 < endpoint_cache[0]
                            for interval_idx in cutlass.range_constexpr(
                                self.func_num // 2,
                                unroll_full=True,
                            ):
                                interval_start = endpoint_cache[2 * interval_idx + 1]
                                interval_end = endpoint_cache[2 * interval_idx + 2]
                                value_valid0 = value_valid0 | ((index_k0 >= interval_start) & (index_k0 < interval_end))
                            valid0 = valid0 and value_valid0
                        if valid1:
                            value_valid1 = index_k1 < endpoint_cache[0]
                            for interval_idx in cutlass.range_constexpr(
                                self.func_num // 2,
                                unroll_full=True,
                            ):
                                interval_start = endpoint_cache[2 * interval_idx + 1]
                                interval_end = endpoint_cache[2 * interval_idx + 2]
                                value_valid1 = value_valid1 | ((index_k1 >= interval_start) & (index_k1 < interval_end))
                            valid1 = valid1 and value_valid1
                    else:
                        valid0 = self.is_auto_score_valid(
                            score_coords[k],
                            seqlen_q,
                            seqlen_k,
                            query_offset,
                            True,
                            func,
                        )
                        valid1 = self.is_auto_score_valid(
                            score_coords[k + 1],
                            seqlen_q,
                            seqlen_k,
                            query_offset,
                            True,
                            func,
                        )
                else:
                    valid0 = self.is_auto_score_valid(
                        score_coords[k],
                        seqlen_q,
                        seqlen_k,
                        query_offset,
                        False,
                        func,
                    )
                    valid1 = self.is_auto_score_valid(
                        score_coords[k + 1],
                        seqlen_q,
                        seqlen_k,
                        query_offset,
                        False,
                        func,
                    )
                sigmoid0 = sigmoid0 if valid0 else self.acc_dtype(0)
                sigmoid1 = sigmoid1 if valid1 else self.acc_dtype(0)
            one_minus0, one_minus1 = sub_packed_f32x2(
                (1.0, 1.0),
                (sigmoid0, sigmoid1),
            )
            inner0, inner1 = fma_packed_f32x2(
                (x0, x1),
                (one_minus0, one_minus1),
                (1.0, 1.0),
            )
            scores[k], scores[k + 1] = mul_packed_f32x2(
                (sigmoid0, sigmoid1),
                (inner0, inner1),
            )

    @cute.jit
    def compute_step(
        self,
        mask_args: Tuple,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        mask_kind: Int32,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        window_size_left, window_size_right = mask_args
        (
            seqlen_q,
            seqlen_k,
            cuseqlen_q,
            scale_softmax,
            scale_gradient,
        ) = value_args
        tStS, tScS, tdPtdP, tdPcdP, func = tensor_args
        mma_s_consumer, mma_dp_consumer, ds_mma_producer = pipeline_args

        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.compute_warp_ids))
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), self.acc_dtype)
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
        cute.arch.fence_view_async_tmem_load()
        s_handle.release()
        score_predicates = None
        if cutlass.const_expr(not self.use_auto_block_metadata and not self.skip_residual_mask):
            score_predicates = cute.make_rmem_tensor(tTMEM_LOADrS.shape, cutlass.Boolean)
            for k in cutlass.range(cute.size(score_predicates), unroll_full=True):
                score_predicates[k] = True
            if mask_kind != 0:
                FusedMask.apply_mask_via_causal_local(
                    score_predicates,
                    tTMEM_LOADcS,
                    seqlen_q,
                    seqlen_k,
                    self.use_semantic_trip_range,
                    self.is_causal,
                    self.is_local,
                    window_size_left,
                    window_size_right,
                    self.is_arbitrary,
                    self.func_num,
                    func,
                    cuseqlen_q,
                )
        if cutlass.const_expr(self.use_auto_block_metadata):
            assert isinstance(func, cute.Tensor)

        # HSTU uses P=silu(alpha*S), not softmax. Keep the SiLU derivative in
        # the score fragment so it can be multiplied with dP below.
        if cutlass.const_expr(self.use_auto_block_metadata):
            if mask_kind == 2:
                self.apply_auto_silu_derivative(
                    tTMEM_LOADrS,
                    tTMEM_LOADcS,
                    seqlen_q,
                    seqlen_k,
                    cuseqlen_q,
                    scale_softmax,
                    func,
                    2,
                )
            elif mask_kind == 1:
                self.apply_auto_silu_derivative(
                    tTMEM_LOADrS,
                    tTMEM_LOADcS,
                    seqlen_q,
                    seqlen_k,
                    cuseqlen_q,
                    scale_softmax,
                    func,
                    1,
                )
            else:
                self.apply_auto_silu_derivative(
                    tTMEM_LOADrS,
                    tTMEM_LOADcS,
                    seqlen_q,
                    seqlen_k,
                    cuseqlen_q,
                    scale_softmax,
                    func,
                    0,
                )
        else:
            for k in cutlass.range(0, cute.size(tTMEM_LOADrS), 2, unroll_full=True):
                x0, x1 = mul_packed_f32x2(
                    (tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1]),
                    (scale_softmax, scale_softmax),
                )
                tanh0 = tanhf(x0 * 0.5)
                tanh1 = tanhf(x1 * 0.5)
                sigmoid0, sigmoid1 = fma_packed_f32x2(
                    (0.5, 0.5),
                    (tanh0, tanh1),
                    (0.5, 0.5),
                )
                if cutlass.const_expr(score_predicates is not None):
                    sigmoid0 = sigmoid0 if score_predicates[k] else self.acc_dtype(0)
                    sigmoid1 = sigmoid1 if score_predicates[k + 1] else self.acc_dtype(0)
                one_minus0, one_minus1 = sub_packed_f32x2(
                    (1.0, 1.0),
                    (sigmoid0, sigmoid1),
                )
                inner0, inner1 = fma_packed_f32x2(
                    (x0, x1),
                    (one_minus0, one_minus1),
                    (1.0, 1.0),
                )
                tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1] = mul_packed_f32x2(
                    (sigmoid0, sigmoid1),
                    (inner0, inner1),
                )

        dp_handle = mma_dp_consumer.wait_and_advance()
        tdPtdP_slice = tdPtdP[(None, None), 0, 0, dp_handle.index]
        tdPcdP_slice = tdPcdP[(None, None), 0, 0]
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtdP = thr_load.partition_S(tdPtdP_slice)
        tTMEM_LOADcdP = thr_load.partition_D(tdPcdP_slice)
        tTMEM_LOADrdP = cute.make_rmem_tensor(tTMEM_LOADcdP.shape, self.acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtdP, tTMEM_LOADrdP)
        cute.arch.fence_view_async_tmem_load()
        dp_handle.release()
        tTMEM_STORErdP = cute.make_rmem_tensor(tTMEM_LOADrdP.shape, self.q_dtype)

        for k in cutlass.range(0, cute.size(tTMEM_LOADrdP), 2, unroll_full=True):
            tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = mul_packed_f32x2(
                (tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]),
                (tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1]),
            )
            tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = mul_packed_f32x2(
                (tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]),
                (scale_gradient, scale_gradient),
            )
        dp_vec = tTMEM_LOADrdP.load()
        tTMEM_STORErdP.store(dp_vec.to(self.q_dtype))

        ds_handle = ds_mma_producer.acquire_and_advance()
        tmem_store_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.acc_dtype)
        tilePlikeFP32 = tdPtdP_slice.shape[1] // Float32.width * self.q_dtype.width
        tdPtdP_dS_layout = cute.composition(tdPtdP_slice.layout, cute.make_layout((tdPtdP_slice.shape[0], tilePlikeFP32)))
        tdPtdP_dS = cute.make_tensor(tdPtdP_slice.iterator, tdPtdP_dS_layout)
        tdPcdP_dS_layout = cute.composition(tdPcdP_slice.layout, cute.make_layout((tdPcdP_slice.shape[0], tilePlikeFP32)))
        tdPcdP_dS = cute.make_tensor(tdPcdP_slice.iterator, tdPcdP_dS_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tdPtdP_dS)

        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtdS = thr_store.partition_D(tdPtdP_dS)
        tTMEM_STOREcdP = thr_store.partition_S(tdPcdP_dS)
        tTMEM_STORErdS_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErdP.iterator, dtype=self.acc_dtype),
            tTMEM_STOREcdP.shape,
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErdS_, tTMEM_STOREtdS)
        cute.arch.fence_view_async_tmem_store()
        ds_handle.commit()
        return mma_s_consumer, mma_dp_consumer, ds_mma_producer

    @cute.jit
    def dQ_epilogue(
        self,
        value_args: Tuple,
        dq_args: Tuple,
        epi_tile: cute.Tile,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        seqlen_q, cuseqlen_q, total_q, batch_coord = value_args
        mma_dq_consumer, gdQ_staged, cdQ_staged, tdQtdQ_staged = dq_args
        dq_handle = mma_dq_consumer.wait_and_advance()
        cute.arch.fence_view_async_shared()

        for iter in cutlass.range(self.iterations_dsk):
            gdQ = gdQ_staged[None, None, iter]
            cdQ = cdQ_staged[None, None, iter]
            tdQtdQ = tdQtdQ_staged[(None, None), 0, 0, iter]
            tdQtdQ_epi = cute.zipped_divide(tdQtdQ, epi_tile)
            cdQ_epi = cute.zipped_divide(cdQ, epi_tile)
            gdQ_epi = cute.zipped_divide(gdQ, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.epilogue_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tdQtdQ_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtdQ = thr_tmem_load.partition_S(tdQtdQ_epi)
            tTMEM_LOADgdQ = thr_tmem_load.partition_D(gdQ_epi)
            tTMEM_LOADcdQ = thr_tmem_load.partition_D(cdQ_epi)

            for i in cutlass.range(cute.size(tTMEM_LOADtdQ, mode=[1]), unroll_full=True):
                tTMEM_LOADtdQ_i = tTMEM_LOADtdQ[None, i, 0]
                tTMEM_LOADgdQ_i = tTMEM_LOADgdQ[None, i, 0]
                tTMEM_LOADcdQ_i = tTMEM_LOADcdQ[None, i, 0]
                tTMrdQ = cute.make_rmem_tensor(tTMEM_LOADcdQ[None, 0, i].shape, self.acc_dtype)
                cute.copy(tiled_tmem_load, tTMEM_LOADtdQ_i, tTMrdQ)
                tSMrdQ = cute.make_rmem_tensor(tTMrdQ.shape, self.q_dtype)
                dq_vec = tTMrdQ.load()
                tSMrdQ.store(dq_vec.to(self.q_dtype))
                if cute.elem_less(tTMEM_LOADcdQ_i[0][0], seqlen_q):
                    cute.autovec_copy(tSMrdQ, tTMEM_LOADgdQ_i)
        dq_handle.release()
        return mma_dq_consumer

    @cute.jit
    def dQ_epilogue_write_zero(
        self,
        seqlen_q: Int32,
        dq_args: Tuple,
        epi_tile: cute.Tile,
    ) -> None:
        """Write zeros for an EMPTY Q-owner tile without touching dQ phases."""
        gdQ_staged, cdQ_staged, tdQtdQ_staged = dq_args

        for iter in cutlass.range(self.iterations_dsk):
            gdQ = gdQ_staged[None, None, iter]
            cdQ = cdQ_staged[None, None, iter]
            tdQtdQ = tdQtdQ_staged[(None, None), 0, 0, iter]
            tdQtdQ_epi = cute.zipped_divide(tdQtdQ, epi_tile)
            cdQ_epi = cute.zipped_divide(cdQ, epi_tile)
            gdQ_epi = cute.zipped_divide(gdQ, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.epilogue_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tdQtdQ_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADgdQ = thr_tmem_load.partition_D(gdQ_epi)
            tTMEM_LOADcdQ = thr_tmem_load.partition_D(cdQ_epi)

            for i in cutlass.range(cute.size(tTMEM_LOADgdQ, mode=[1]), unroll_full=True):
                tTMEM_LOADgdQ_i = tTMEM_LOADgdQ[None, i, 0]
                tTMEM_LOADcdQ_i = tTMEM_LOADcdQ[None, i, 0]
                zero_dq = cute.make_rmem_tensor(tTMEM_LOADcdQ_i.shape, self.q_dtype)
                zero_dq.fill(0)
                if cute.elem_less(tTMEM_LOADcdQ_i[0][0], seqlen_q):
                    cute.autovec_copy(zero_dq, tTMEM_LOADgdQ_i)
