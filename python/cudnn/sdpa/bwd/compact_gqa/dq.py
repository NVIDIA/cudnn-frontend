# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Optional, Type, Tuple, Union
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.cute.testing as testing
import cutlass.utils.blackwell_helpers as sm100_utils


class PersistentDenseGemmKernel:

    def __init__(
        self, acc_dtype: Type[cutlass.Numeric], use_2cta_instrs: bool, mma_tiler_mn: Tuple[int, int], cluster_shape_mn: Tuple[int, int], use_tma_store: bool
    ):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store
        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(self.a_dtype, self.a_major_mode, self.b_major_mode, self.acc_dtype, self.cta_group, self.mma_tiler[:2])
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * mma_inst_tile_k)
        self.cta_tile_shape_mnk = (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape), self.mma_tiler[1], self.mma_tiler[2])
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,))
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype)
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
        )
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.c_smem_layout_staged = (
            sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage) if cutlass.const_expr(self.use_tma_store) else None
        )
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(tiled_mma, self.mma_tiler, self.num_acc_stage)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        query_start: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        self._setup_attributes()
        tiled_mma = sm100_utils.make_trivial_tiled_mma(self.a_dtype, self.a_major_mode, self.b_major_mode, self.acc_dtype, self.cta_group, self.mma_tiler[:2])
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.TFloat32 if a.element_type is cutlass.Float32 else None,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.TFloat32 if b.element_type is cutlass.Float32 else None,
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            c_cta_v_layout = cute.composition(cute.make_identity_layout(c.shape), self.epi_tile)
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, c_cta_v_layout)
        self.tile_sched_params, grid = self._compute_grid(c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters)
        self.buffer_align_bytes = 1024
        c_smem_size = cute.cosize(self.c_smem_layout_staged.outer) if cutlass.const_expr(self.use_tma_store) else 0

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[cute.struct.MemRange[self.c_dtype, c_smem_size], self.buffer_align_bytes]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if cutlass.const_expr(self.use_tma_store) else c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            query_start,
            epilogue_op,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(*self.cluster_shape_mn, 1), smem=self.shared_storage.size_in_bytes(), stream=stream)
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        query_start: cutlass.Int32,
        epilogue_op: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)
        cute.arch.mbarrier_init_fence()
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner) if cutlass.const_expr(self.use_tma_store) else None
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_block_cnt = cute.size(gA_mkl, mode=[3])
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta)
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape), cur_tile_coord[1], cur_tile_coord[2])
                causal_end = cutlass.min(mA_mkl.shape[1], query_start + (mma_tile_coord_mnl[0] + 1) * 128)
                k_block_cnt = cute.ceil_div(causal_end, self.mma_tiler[2])
                tAgA_slice = tAgA[None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]]
                tBgB_slice = tBgB[None, mma_tile_coord_mnl[1], None, 0]
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_block_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                for k_block in cutlass.range(0, k_block_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[None, ab_producer_state.count],
                        tAsA[None, ab_producer_state.index],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[None, ab_producer_state.count],
                        tBsB[None, ab_producer_state.index],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_block_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)
        if warp_idx == self.mma_warp_id:
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(barrier_id=self.tmem_ptr_sync_bar_id, number_of_threads=tmem_ptr_read_threads)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape), cur_tile_coord[1], cur_tile_coord[2])
                causal_end = cutlass.min(mA_mkl.shape[1], query_start + (mma_tile_coord_mnl[0] + 1) * 128)
                k_block_cnt = cute.ceil_div(causal_end, self.mma_tiler[2])
                tCtAcc = tCtAcc_base[None, None, None, acc_producer_state.index]
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in range(k_block_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx, ab_consumer_state.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kphase_coord], tCrB[kphase_coord], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_block_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)
        if warp_idx < self.mma_warp_id:
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=use_2cta_instrs)
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(barrier_id=self.tmem_ptr_sync_bar_id, number_of_threads=tmem_ptr_read_threads)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs)
            tTR_rC = None
            tiled_copy_r2s = None
            simt_atom = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_gC_partitioned = None
            if cutlass.const_expr(self.use_tma_store):
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, epi_tidx, sC)
                tma_atom_c, bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, epi_tile, sC)
            else:
                simt_atom, tTR_rC, tTR_gC_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tiled_copy_t2r, tCgC, epi_tile, sC)
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            c_pipeline = None
            if cutlass.const_expr(self.use_tma_store):
                c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilog_warp_id))
                c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape), cur_tile_coord[1], cur_tile_coord[2])
                bSG_gC = None
                tTR_gC = None
                if cutlass.const_expr(self.use_tma_store):
                    bSG_gC = bSG_gC_partitioned[None, None, None, *mma_tile_coord_mnl]
                else:
                    tTR_gC = tTR_gC_partitioned[None, None, None, None, None, *mma_tile_coord_mnl]
                tTR_tAcc = tTR_tAcc_base[None, None, None, None, None, acc_consumer_state.index]
                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                if cutlass.const_expr(self.use_tma_store):
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                else:
                    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(0, int(subtile_cnt)):
                    tTR_tAcc_mn = tTR_tAcc[None, None, None, subtile_idx]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    if cutlass.const_expr(self.use_tma_store):
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        acc_vec = acc_vec.to(self.c_dtype)
                        tRS_rC.store(acc_vec)
                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[None, None, None, c_buffer])
                        cute.arch.fence_view_async_shared()
                        epilog_threads = 32 * len(self.epilog_warp_id)
                        cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads)
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(tma_atom_c, bSG_sC[None, c_buffer], bSG_gC[None, subtile_idx])
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads)
                    else:
                        acc_vec = tTR_rAcc.load()
                        acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                        tTR_rC.store(acc_vec)
                        cute.copy(simt_atom, tTR_rC, tTR_gC[None, None, None, subtile_idx])
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads)
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs)
            if cutlass.const_expr(self.use_tma_store):
                c_pipeline.producer_tail()

    def epilog_tmem_copy_and_partition(
        self, tidx: cutlass.Int32, tAcc: cute.Tensor, gC_mnl: cute.Tensor, epi_tile: cute.Tile, use_2cta_instrs: Union[cutlass.Boolean, bool]
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, use_2cta_instrs)
        tAcc_epi = cute.flat_divide(tAcc[(None, None), 0, 0, None], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[None, None, 0, 0, 0])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(gC_mnl[(None, None), 0, 0, None, None, None], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[None, None, None, 0, 0, 0, 0, 0].shape, self.acc_dtype)
        return (tiled_copy_t2r, tTR_tAcc, tTR_rAcc)

    def epilog_smem_copy_and_partition(
        self, tiled_copy_t2r: cute.TiledCopy, tTR_rC: cute.Tensor, tidx: cutlass.Int32, sC: cute.Tensor
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return (tiled_copy_r2s, tRS_rC, tRS_sC)

    def epilog_gmem_copy_and_partition(
        self, tidx: cutlass.Int32, atom: Union[cute.CopyAtom, cute.TiledCopy], gC_mnl: cute.Tensor, epi_tile: cute.Tile, sC: cute.Tensor
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        gC_epi = cute.flat_divide(gC_mnl[(None, None), 0, 0, None, None, None], epi_tile)
        if cutlass.const_expr(self.use_tma_store):
            tma_atom_c = atom
            sC_for_tma_partition = cute.group_modes(sC, 0, 2)
            gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
            bSG_sC, bSG_gC = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), sC_for_tma_partition, gC_for_tma_partition)
            return (tma_atom_c, bSG_sC, bSG_gC)
        else:
            tiled_copy_t2r = atom
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_gC = thr_copy_t2r.partition_D(gC_epi)
            tTR_rC = cute.make_rmem_tensor(tTR_gC[None, None, None, 0, 0, 0, 0, 0].shape, self.c_dtype)
            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
            return (simt_atom, tTR_rC, tTR_gC)

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        smem_capacity: int,
        occupancy: int,
        use_tma_store: bool,
    ) -> Tuple[int, int, int]:
        num_acc_stage = 2
        num_c_stage = 2 if use_tma_store else 0
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1) if use_tma_store else None
        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one) if use_tma_store else 0
        c_bytes = c_bytes_per_stage * num_c_stage
        num_ab_stage = (smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)) // ab_bytes_per_stage
        if use_tma_store:
            num_c_stage += (smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy * (mbar_helpers_bytes + c_bytes)) // (
                occupancy * c_bytes_per_stage
            )
        return (num_acc_stage, num_ab_stage, num_c_stage)

    @staticmethod
    def _compute_grid(
        c: cute.Tensor, cta_tile_shape_mnk: Tuple[int, int, int], cluster_shape_mn: Tuple[int, int], max_active_clusters: cutlass.Constexpr
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[0, (None, None, None)].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)
        return (tile_sched_params, grid)

    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma: cute.TiledMma, mma_tiler: Tuple[int, int, int], num_acc_stage: int) -> int:
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        return num_tmem_alloc_cols


class DirectCausalDq:
    def __init__(self):
        self.gemm = PersistentDenseGemmKernel(cutlass.Float32, True, (128, 256), (2, 1), True)

    @cute.jit
    def __call__(
        self, ds: cute.Tensor, k: cute.Tensor, dq: cute.Tensor, query_start: cutlass.Int32, max_active_clusters: cutlass.Constexpr, stream: cuda.CUstream
    ):
        rows = cutlass.min(ds.shape[2], k.shape[0] - query_start)
        keys = cutlass.min(k.shape[0], query_start + ds.shape[2])
        a = cute.make_tensor(ds.iterator, cute.make_layout((rows, keys, 8), stride=(ds.shape[3], 1, cutlass.Int64(ds.shape[2]) * ds.shape[3])))
        b = cute.make_tensor(k.iterator, cute.make_layout((256, keys, 1), stride=(1, 256, cutlass.Int64(k.shape[0]) * 256)))
        c = cute.make_tensor(dq.iterator + cutlass.Int64(query_start) * 2048, cute.make_layout((rows, 256, 8), stride=(2048, 1, 256)))
        self.gemm(a, b, c, query_start, max_active_clusters, stream)
