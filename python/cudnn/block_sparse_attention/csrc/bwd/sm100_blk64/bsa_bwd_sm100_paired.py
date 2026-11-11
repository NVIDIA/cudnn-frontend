# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import LoadCacheMode, OperandMajorMode, cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils

from typing import Tuple


from .bsa_bwd_sm100 import BlockSparseAttnBackwardSm100Blk64, _cp_async_copy_or_zero_fill
from cudnn._cutlass_compat import SmemAllocator, TmemAllocator


class MixedKvBackwardSm100:
    """Run residual KV tasks and paired intersections in one shared workspace."""

    def __init__(self, sparse_block_size, has_block_sizes, bucket_size_blocks):
        from ..local_pair import PairedKvNeighborhoods, LocalPairMatch8, LocalPairPartition8

        from ..local_pair_refine_batch_wide import LocalPairRefineBatchWide

        self.bucket_size = bucket_size_blocks
        self.classifier = PairedKvNeighborhoods()
        self.matcher = LocalPairMatch8()
        self.refiner = LocalPairRefineBatchWide(32, 32, 128)
        self.partition = LocalPairPartition8()
        self.regular = BlockSparseAttnBackwardSm100Blk64(sparse_block_size, has_block_sizes, skip_convert=True)
        self.paired = LocalPairedBackwardSm100(
            sparse_block_size,
            has_block_sizes,
            skip_preprocess=True,
        )

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        dO: cute.Tensor,
        O: cute.Tensor,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        LSE: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        offsets: cute.Tensor,
        indices: cute.Tensor,
        sizes: cute.Tensor,
        workspace: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
        q2k: cute.Tensor,
        counts: cute.Tensor,
        bits: cute.Tensor,
        duplicates: cute.Tensor,
        flags: cute.Tensor,
        residuals: cute.Tensor,
    ):
        b, h, q = counts.shape
        self.classifier.clear(bits, duplicates).launch(grid=(cute.ceil_div(cute.size(bits), 256), 1, 1), block=(256, 1, 1), stream=stream)
        self.classifier.scatter(q2k, counts, bits, duplicates).launch(grid=(q, h, b), block=(128, 1, 1), stream=stream)
        owners = cute.make_tensor(
            flags.iterator + bits.shape[2],
            cute.make_layout((b, h, cute.ceil_div(bits.shape[2], 2)), stride=flags.stride),
        )
        self.matcher(bits, duplicates, flags, owners, residuals[None, None, 0, None], stream)
        # The fixed shared tile and device-side word limit permit compile reuse.
        self.refiner(bits, duplicates, flags, owners, residuals[None, None, 0, None], stream)
        self.partition(bits, flags, offsets, indices, residuals, self.bucket_size, stream)
        self.regular(
            problem_shape,
            dO,
            O,
            Q,
            K,
            V,
            LSE,
            dQ,
            dK,
            dV,
            offsets,
            indices,
            sizes,
            workspace,
            scale,
            stream,
            flags,
            residuals,
        )
        self.paired(
            problem_shape,
            dO,
            O,
            Q,
            K,
            V,
            LSE,
            dQ,
            dK,
            dV,
            offsets,
            indices,
            sizes,
            workspace,
            scale,
            stream,
            flags,
            residuals,
        )


class LocalPairedBackwardSm100(BlockSparseAttnBackwardSm100Blk64):
    """Reuse Q, dO and dQ reduction across matched KV neighborhoods."""

    def __init__(self, sparse_block_size, has_block_sizes, skip_preprocess=False):
        super().__init__(sparse_block_size, has_block_sizes, skip_preprocess=skip_preprocess, kv_group_size=2)
        self.local_pairs = True
        self.num_regs_load = self.num_regs_mma = self.num_regs_empty = 104

    @cute.jit
    def _compute_bwd_grid(self, problem_shape, bucketed_k2q_offsets):
        _, _, _, HB = problem_shape
        H, B = HB
        tasks = cute.ceil_div(cute.size(bucketed_k2q_offsets.shape[0]) - 1, 2)
        return (tasks * bucketed_k2q_offsets.shape[1], H, B)

    @cute.kernel
    def bwd(
        self,
        QK_tiled_mma: cute.TiledMma,
        fake_QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        fake_dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        dQ_acc: cute.Tensor,
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        LSE: cute.Tensor,
        scale_softmax: Float32,
        sum_OdO: cute.Tensor,
        bucketed_k2q_offsets: cute.Tensor,
        bucketed_k2q_indices: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        variable_block_sizes: cute.Tensor,
        Q_smem_layout_staged: cute.ComposedLayout,
        K_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_staged: cute.ComposedLayout,
        KT_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        dST_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        dQ_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
        pair_flags: cute.Tensor | None,
        pair_residuals: cute.Tensor | None,
    ):
        bidx, bidy, bidz = self.logical_block_coord(problem_shape[0])
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        seqlen_q, seqlen_k, head_dim, HB = problem_shape
        num_heads, batch_size = HB

        tasks_per_group = cute.ceil_div(cute.size(bucketed_k2q_offsets.shape[0]) - 1, 2)
        task_linear = bidx
        q_group = task_linear // tasks_per_group
        task_idx = task_linear - q_group * tasks_per_group

        nk = cute.size(bucketed_k2q_offsets.shape[0]) - 1
        first = pair_flags[bidz, bidy, nk + task_idx]
        owner_valid = first < nk
        first = cute.min(first, nk - 1)
        second = pair_flags[bidz, bidy, first]
        owner_valid = owner_valid and second > first
        kv_block_idx = (first, cute.max(second, 0))
        k2q_begin = bucketed_k2q_offsets[first, q_group, (bidy, bidz)] + pair_residuals[bidz, bidy, q_group, first]
        k2q_end = bucketed_k2q_offsets[first + 1, q_group, (bidy, bidz)]
        iter_count = k2q_end - k2q_begin
        iter_index = Int32(0)
        load_iter_count = iter_count
        mma_iter_count = cute.ceil_div(iter_count, 2)
        compute_iter_count = mma_iter_count
        reduce_iter_count = iter_count

        task_has_work = iter_count > 0
        task_has_work = task_has_work and owner_valid

        if task_has_work:
            if warp_idx == self.load_warp_id:
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)

            smem = SmemAllocator()
            storage = smem.allocate(self.shared_storage)

            load_mma_Q_pipeline = self.make_and_init_load_mma_Q_pipeline(storage.load_mma_Q_mbar_ptr.data_ptr())
            load_mma_dO_pipeline = self.make_and_init_load_mma_dO_pipeline(storage.load_mma_dO_mbar_ptr.data_ptr())
            load_compute_LSE_pipeline = self.make_and_init_load_compute_LSE_pipeline(storage.load_compute_lse_mbar_ptr.data_ptr())
            load_compute_sum_OdO_pipeline = self.make_and_init_load_compute_sum_OdO_pipeline(storage.load_compute_sum_OdO_mbar_ptr.data_ptr())
            mma_compute_S_pipeline = self.make_and_init_mma_compute_S_pipeline(storage.mma_compute_S_mbar_ptr.data_ptr())
            mma_compute_dP_pipeline = self.make_and_init_mma_compute_dP_pipeline(storage.mma_compute_dP_mbar_ptr.data_ptr())
            mma_reduce_dQ_pipeline = self.make_and_init_mma_reduce_dQ_pipeline(storage.mma_reduce_dQ_mbar_ptr.data_ptr())
            compute_mma_P_pipeline = self.make_and_init_compute_mma_P_pipeline(storage.compute_mma_P_mbar_ptr.data_ptr())
            compute_mma_dS_pipeline = self.make_and_init_compute_mma_dS_pipeline(storage.compute_mma_dS_mbar_ptr.data_ptr())
            mma_compute_dKdV_pipeline = self.make_and_init_mma_compute_dKdV_pipeline(storage.mma_compute_dKdV_mbar_ptr.data_ptr())
            reduce_tma_store_pipeline = self.make_and_init_reduce_tma_store_pipeline()

            self.cta_sync_barrier.arrive_and_wait()

            sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
            sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
            sV = storage.sV.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
            sP = storage.sP.get_tensor(P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner)
            sdO = storage.sdO.get_tensor(dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner)
            if cutlass.const_expr(self.alias_ps):
                sdS = storage.sP.get_tensor(dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner)
            else:
                sdS = storage.sdS.get_tensor(dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner)
            sdQ = storage.sdQ.get_tensor(dQ_smem_layout_staged.outer, swizzle=dQ_smem_layout_staged.inner)
            sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
            sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)

            tmem_holding_buf = storage.tmem_holding_buf.ptr
            tmem = TmemAllocator(
                tmem_holding_buf,
                barrier_for_retrieve=self.tmem_alloc_barrier,
                allocator_warp_id=self.mma_warp_id,
            )

            sQT_ptr = cute.recast_ptr(sQ.iterator, QT_smem_layout_staged.inner)
            sQT = cute.make_tensor(sQT_ptr, QT_smem_layout_staged.outer)
            sKT_ptr = cute.recast_ptr(sK.iterator, KT_smem_layout_staged.inner)
            sKT = cute.make_tensor(sKT_ptr, KT_smem_layout_staged.outer)
            sdST_ptr = cute.recast_ptr(sdS.iterator, dST_smem_layout_staged.inner)
            sdST = cute.make_tensor(sdST_ptr, dST_smem_layout_staged.outer)
            sdOT_ptr = cute.recast_ptr(sdO.iterator, dOT_smem_layout_staged.inner)
            sdOT = cute.make_tensor(sdOT_ptr, dOT_smem_layout_staged.outer)

            # (MMA, MMA_M, MMA_K, STAGE)
            tSrQ = QK_tiled_mma.make_fragment_A(sQ)
            # (MMA, MMA_N, MMA_K, STAGE)
            tSrK = QK_tiled_mma.make_fragment_B(sK)

            tdPrdO = dOV_tiled_mma.make_fragment_A(sdO)
            tdPrV = dOV_tiled_mma.make_fragment_B(sV)

            tdKTrQT = QdS_tiled_mma.make_fragment_A(sQT)
            tdKTrdST = QdS_tiled_mma.make_fragment_B(sdST)

            tdVTrdOT = dOP_tiled_mma.make_fragment_A(sdOT)
            tdVTrP = dOP_tiled_mma.make_fragment_B(sP)

            tdQrdS = dSK_tiled_mma.make_fragment_A(sdS)
            tdQrKT = dSK_tiled_mma.make_fragment_B(sKT)

            if warp_idx == self.load_warp_id or warp_idx == self.load_warp_id + 1:
                cute.arch.setmaxregister_decrease(self.num_regs_load)
                for loader_role in cutlass.range_constexpr(2):
                    if warp_idx == self.load_warp_id + loader_role:
                        self.load(
                            Q_in,
                            K_in,
                            V_in,
                            dO_in,
                            LSE,
                            sum_OdO,
                            sQ,
                            sK,
                            sV,
                            sdO,
                            sLSE,
                            sSum_OdO,
                            bucketed_k2q_indices,
                            k2q_begin,
                            kv_block_idx,
                            fake_QK_tiled_mma,
                            fake_dOV_tiled_mma,
                            tma_atom_Q,
                            tma_atom_K,
                            tma_atom_V,
                            tma_atom_dO,
                            problem_shape,
                            load_iter_count,
                            iter_index,
                            (
                                load_mma_Q_pipeline,
                                load_compute_LSE_pipeline,
                                load_mma_dO_pipeline,
                                load_compute_sum_OdO_pipeline,
                            ),
                            loader_role,
                        )
            elif warp_idx == self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_mma)

                tmem.allocate(self.tmem_alloc_cols)
                # Barrier before retrieve tensor memory ptr from shared memory
                self.tmem_alloc_barrier.arrive_and_wait()
                # Retrieve tmem ptr
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

                tStS_shape = QK_tiled_mma.partition_shape_C(cute.select(self.QK_mma_tiler, mode=[0, 1]))
                tStS = QK_tiled_mma.make_fragment_C(tStS_shape)
                tStS = cute.make_tensor(tmem_ptr_base + self.tmem_S_offset, tStS.layout)

                tdPtdP_shape = dOV_tiled_mma.partition_shape_C(cute.select(self.dOV_mma_tiler, mode=[0, 1]))
                tdPtdP = dOV_tiled_mma.make_fragment_C(tdPtdP_shape)
                tdPtdP = cute.make_tensor(tmem_ptr_base + self.tmem_dP_offset, tdPtdP.layout)

                tdQtdQ_shape = dSK_tiled_mma.partition_shape_C(cute.select(self.dSK_mma_tiler, mode=[0, 1]))
                tdQtdQ = dSK_tiled_mma.make_fragment_C(tdQtdQ_shape)
                tdQtdQ = cute.make_tensor(tmem_ptr_base + self.tmem_dQ_offset, tdQtdQ.layout)

                tdKTtdKT_shape = QdS_tiled_mma.partition_shape_C(cute.select(self.QdS_mma_tiler, mode=[0, 1]))
                tdKTtdKT = QdS_tiled_mma.make_fragment_C(tdKTtdKT_shape)
                tdKTtdKT = cute.make_tensor(tmem_ptr_base + self.tmem_dK_offset, tdKTtdKT.layout)

                tdVTtdVT_shape = dOP_tiled_mma.partition_shape_C(cute.select(self.dOP_mma_tiler, mode=[0, 1]))
                tdVTtdVT = dOP_tiled_mma.make_fragment_C(tdVTtdVT_shape)
                tdVTtdVT = cute.make_tensor(tmem_ptr_base + self.tmem_dV_offset, tdVTtdVT.layout)

                self.mma(
                    QK_tiled_mma,
                    dOV_tiled_mma,
                    dOP_tiled_mma,
                    QdS_tiled_mma,
                    dSK_tiled_mma,
                    tStS,
                    tSrQ,
                    tSrK,
                    tdPtdP,
                    tdPrdO,
                    tdPrV,
                    tdVTtdVT,
                    tdVTrdOT,
                    tdVTrP,
                    tdQtdQ,
                    tdQrdS,
                    tdQrKT,
                    tdKTtdKT,
                    tdKTrQT,
                    tdKTrdST,
                    mma_iter_count,
                    (
                        load_mma_Q_pipeline,
                        mma_compute_S_pipeline,
                        load_mma_dO_pipeline,
                        mma_compute_dP_pipeline,
                        mma_reduce_dQ_pipeline,
                        compute_mma_P_pipeline,
                        compute_mma_dS_pipeline,
                        mma_compute_dKdV_pipeline,
                    ),
                )
            elif warp_idx in self.compute_warp_id:
                cute.arch.setmaxregister_increase(self.num_regs_compute)
                self.tmem_alloc_barrier.arrive_and_wait()
                # Retrieve tmem ptr
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

                tStS_shape = QK_tiled_mma.partition_shape_C(cute.select(self.QK_mma_tiler, mode=[0, 1]))
                tStS = QK_tiled_mma.make_fragment_C(tStS_shape)
                tStS = cute.make_tensor(tmem_ptr_base + self.tmem_S_offset, tStS.layout)

                tdPtdP_shape = dOV_tiled_mma.partition_shape_C(cute.select(self.dOV_mma_tiler, mode=[0, 1]))
                tdPtdP = dOV_tiled_mma.make_fragment_C(tdPtdP_shape)
                tdPtdP = cute.make_tensor(tmem_ptr_base + self.tmem_dP_offset, tdPtdP.layout)

                tdKTtdKT_shape = QdS_tiled_mma.partition_shape_C(cute.select(self.QdS_mma_tiler, mode=[0, 1]))
                tdKTtdKT = QdS_tiled_mma.make_fragment_C(tdKTtdKT_shape)
                tdKTtdKT = cute.make_tensor(tmem_ptr_base + self.tmem_dK_offset, tdKTtdKT.layout)

                tdVTtdVT_shape = dOP_tiled_mma.partition_shape_C(cute.select(self.dOP_mma_tiler, mode=[0, 1]))
                tdVTtdVT = dOP_tiled_mma.make_fragment_C(tdVTtdVT_shape)
                tdVTtdVT = cute.make_tensor(tmem_ptr_base + self.tmem_dV_offset, tdVTtdVT.layout)
                self.compute(
                    tStS,
                    tdPtdP,
                    sLSE,
                    sdS,
                    sP,
                    sSum_OdO,
                    dK_acc,
                    dV_acc,
                    tdKTtdKT,
                    tdVTtdVT,
                    kv_block_idx,
                    variable_block_sizes,
                    problem_shape,
                    compute_iter_count,
                    scale_softmax,
                    (
                        mma_compute_S_pipeline,
                        compute_mma_P_pipeline,
                        load_compute_LSE_pipeline,
                        load_compute_sum_OdO_pipeline,
                        mma_compute_dP_pipeline,
                        compute_mma_dS_pipeline,
                        mma_compute_dKdV_pipeline,
                    ),
                )

                self.epilogue_sync_barrier.arrive_and_wait()
                if warp_idx % self.num_compute_warps == 0:
                    tmem_ptr = cute.arch.retrieve_tmem_ptr(
                        Float32,
                        alignment=16,
                        ptr_to_buffer_holding_addr=tmem_holding_buf,
                    )
                    cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
            elif warp_idx in self.reduce_warp_id:
                cute.arch.setmaxregister_increase(self.num_regs_reduce)

                self.tmem_alloc_barrier.arrive_and_wait()
                # Retrieve tmem ptr
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

                tdQtdQ_shape = dSK_tiled_mma.partition_shape_C(cute.select(self.dSK_mma_tiler, mode=[0, 1]))
                tdQtdQ = dSK_tiled_mma.make_fragment_C(tdQtdQ_shape)
                tdQtdQ = cute.make_tensor(tmem_ptr_base + self.tmem_dQ_offset, tdQtdQ.layout)

                self.reduce(
                    problem_shape,
                    tdQtdQ,
                    bucketed_k2q_indices,
                    k2q_begin,
                    tma_atom_dQ_acc,
                    dQ_acc,
                    sdQ,
                    reduce_iter_count,
                    (mma_reduce_dQ_pipeline, reduce_tma_store_pipeline),
                )
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_empty)

    @cute.jit
    def load(
        self,
        Q_in: cute.Tensor,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        dO_in: cute.Tensor,
        LSE: cute.Tensor,
        sum_OdO: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        bucketed_k2q_indices: cute.Tensor,
        k2q_begin: Int32,
        kv_block_idx: tuple,
        fake_QK_tiled_mma: cute.TiledMma,
        fake_dOV_tiled_mma: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        iter_count: Int32,
        iter_index: Int32,
        pipeline_args: tuple,
        loader_role: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % self.threads_per_warp
        async_copy_num_elts = self.sparse_block_size // self.threads_per_warp
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=LoadCacheMode.ALWAYS),
            self.acc_dtype,
            num_bits_per_copy=self.acc_dtype.width,
        )
        _, blk_coord_h, blk_coord_b = self.logical_block_coord(problem_shape[0])
        seqlen_q, seqlen_k, head_dim, HB = problem_shape
        num_heads, batch_size = HB
        (
            load_mma_Q_pipeline,
            load_compute_LSE_pipeline,
            load_mma_dO_pipeline,
            load_compute_sum_OdO_pipeline,
        ) = pipeline_args

        total_iter_count = iter_count

        # (bM, bK, RestM, RestK, (H, B))
        gQ = cute.local_tile(Q_in, cute.select(self.fake_QK_mma_tiler, mode=[0, 2]), (None, None, None))
        # (bN, bK, RestN, RestK, (H, B))
        # (bM, bK, RestM, RestK, (H, B))
        gdO = cute.local_tile(dO_in, cute.select(self.fake_dOV_mma_tiler, mode=[0, 2]), (None, None, None))
        # (bN, bK, RestN, RestK, (H, B))

        QK_thr_mma = fake_QK_tiled_mma.get_slice(0)
        dOV_thr_mma = fake_dOV_tiled_mma.get_slice(0)

        tSgQ = QK_thr_mma.partition_A(gQ)
        tdPgdO = dOV_thr_mma.partition_A(gdO)

        load_mma_Q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_Q_stage)
        load_compute_LSE_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_compute_LSE_stage)
        load_mma_dO_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_dO_stage)
        load_compute_sum_OdO_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_compute_sum_OdO_stage)

        sQ = cute.make_tensor(sQ.iterator, cute.make_layout(((64, 16), 2, (4, 2), 2), stride=((64, 1), 4096, (16, 8192), 16384)))
        sQ_0 = sQ[None, 0, None, load_mma_Q_producer_state.index]
        sQ_1 = sQ[None, 1, None, load_mma_Q_producer_state.index]

        sdO = cute.make_tensor(sdO.iterator, cute.make_layout(((64, 16), 2, (4, 2), 2), stride=((64, 1), 4096, (16, 8192), 16384)))
        sdO_0 = sdO[None, 0, None, load_mma_dO_producer_state.index]
        sdO_1 = sdO[None, 1, None, load_mma_dO_producer_state.index]
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tQsQ_0, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_0, 0, 2), cute.group_modes(tSgQ, 0, 3))
        tQsQ_1, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_1, 0, 2), cute.group_modes(tSgQ, 0, 3))
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tdOsdO_0, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO, 0, cute.make_layout(1), cute.group_modes(sdO_0, 0, 2), cute.group_modes(tdPgdO, 0, 3)
        )
        tdOsdO_1, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_dO, 0, cute.make_layout(1), cute.group_modes(sdO_1, 0, 2), cute.group_modes(tdPgdO, 0, 3))
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))

        q_block_idx_0 = bucketed_k2q_indices[k2q_begin + iter_index, (blk_coord_h, blk_coord_b)]
        iter_index += 1
        q_block_idx_1 = cute.ceil_div(seqlen_q, self.sparse_block_size)  # First block beyond the final partial Q block.
        if iter_index < total_iter_count:
            q_block_idx_1 = bucketed_k2q_indices[k2q_begin + iter_index, (blk_coord_h, blk_coord_b)]
        q_block_0_full = (q_block_idx_0 + 1) * self.sparse_block_size <= seqlen_q

        if cutlass.const_expr(loader_role == 0):
            load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
            tma_barrier = load_mma_Q_pipeline.producer_get_barrier(load_mma_Q_producer_state)
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_Q_bytes * (self.kv_group_size + 1))

            # Load the two physical KV blocks into the original KV128 shared tile.
            self.load_pair(tma_atom_K, K_in, sK, kv_block_idx, blk_coord_h, blk_coord_b, tma_barrier)

            # Load Q0
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
                tQsQ_0,
                tma_bar_ptr=tma_barrier,
            )

            # Load Q1
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
                tQsQ_1,
                tma_bar_ptr=tma_barrier,
            )

            load_mma_Q_producer_state.advance()

        if cutlass.const_expr(loader_role == 1):
            load_compute_LSE_pipeline.producer_acquire(load_compute_LSE_producer_state)

            # Load LSE
            # 32 threads load 64 values, each thread loads 2 values
            sLSE_for_copy = cute.flat_divide(sLSE, (1,))
            LSE_for_copy = cute.flat_divide(LSE, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                LSE_idx = q_block_idx_0 * self.sparse_block_size + thread_idx * async_copy_num_elts
                if q_block_0_full:
                    cute.copy(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[
                            None,
                            thread_idx * async_copy_num_elts + i,
                            load_compute_LSE_producer_state.index,
                        ],
                    )
                else:
                    _cp_async_copy_or_zero_fill(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[
                            None,
                            thread_idx * async_copy_num_elts + i,
                            load_compute_LSE_producer_state.index,
                        ],
                        cute.elem_less(LSE_idx + i, seqlen_q),
                    )

            for i in cutlass.range_constexpr(async_copy_num_elts):
                LSE_idx = q_block_idx_1 * self.sparse_block_size + thread_idx * async_copy_num_elts
                _cp_async_copy_or_zero_fill(
                    atom_async_copy,
                    LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                    sLSE_for_copy[
                        None,
                        self.sparse_block_size + thread_idx * async_copy_num_elts + i,
                        load_compute_LSE_producer_state.index,
                    ],
                    cute.elem_less(LSE_idx + i, seqlen_q),
                )

            load_compute_LSE_pipeline.producer_commit(load_compute_LSE_producer_state)
            load_compute_LSE_producer_state.advance()

            load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
            tma_barrier = load_mma_dO_pipeline.producer_get_barrier(load_mma_dO_producer_state)
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_dO_bytes * (self.kv_group_size + 1))

            # Load dO0
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO_0,
                tma_bar_ptr=tma_barrier,
            )
            # Load dO1
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO_1,
                tma_bar_ptr=tma_barrier,
            )

            # Load the two physical KV blocks into the original KV128 shared tile.
            self.load_pair(tma_atom_V, V_in, sV, kv_block_idx, blk_coord_h, blk_coord_b, tma_barrier)

            load_mma_dO_producer_state.advance()

            load_compute_sum_OdO_pipeline.producer_acquire(load_compute_sum_OdO_producer_state)

            sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
            sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                sum_OdO_idx = q_block_idx_0 * self.sparse_block_size + thread_idx * async_copy_num_elts
                if q_block_0_full:
                    cute.copy(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[
                            None,
                            thread_idx * async_copy_num_elts + i,
                            load_compute_sum_OdO_producer_state.index,
                        ],
                    )
                else:
                    _cp_async_copy_or_zero_fill(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[
                            None,
                            thread_idx * async_copy_num_elts + i,
                            load_compute_sum_OdO_producer_state.index,
                        ],
                        cute.elem_less(sum_OdO_idx + i, seqlen_q),
                    )
            for i in cutlass.range_constexpr(async_copy_num_elts):
                sum_OdO_idx = q_block_idx_1 * self.sparse_block_size + thread_idx * async_copy_num_elts
                _cp_async_copy_or_zero_fill(
                    atom_async_copy,
                    sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                    sSum_OdO_for_copy[
                        None,
                        self.sparse_block_size + thread_idx * async_copy_num_elts + i,
                        load_compute_sum_OdO_producer_state.index,
                    ],
                    cute.elem_less(sum_OdO_idx + i, seqlen_q),
                )

            load_compute_sum_OdO_pipeline.producer_commit(load_compute_sum_OdO_producer_state)
            load_compute_sum_OdO_producer_state.advance()

        iter_count -= 2
        iter_index += 1

        while iter_count > 0:

            sQ = cute.make_tensor(sQ.iterator, cute.make_layout(((64, 16), 2, (4, 2), 2), stride=((64, 1), 4096, (16, 8192), 16384)))
            sQ_0 = sQ[None, 0, None, load_mma_Q_producer_state.index]
            sQ_1 = sQ[None, 1, None, load_mma_Q_producer_state.index]

            sdO = cute.make_tensor(sdO.iterator, cute.make_layout(((64, 16), 2, (4, 2), 2), stride=((64, 1), 4096, (16, 8192), 16384)))
            sdO_0 = sdO[None, 0, None, load_mma_dO_producer_state.index]
            sdO_1 = sdO[None, 1, None, load_mma_dO_producer_state.index]

            # ((atom_v, rest_v), STAGE)
            # ((atom_v, rest_v), RestM, RestK, (H, B))
            tQsQ_0, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_0, 0, 2), cute.group_modes(tSgQ, 0, 3))
            tQsQ_1, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_1, 0, 2), cute.group_modes(tSgQ, 0, 3))
            # ((atom_v, rest_v), STAGE)
            # ((atom_v, rest_v), RestM, RestK, (H, B))
            tdOsdO_0, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_dO, 0, cute.make_layout(1), cute.group_modes(sdO_0, 0, 2), cute.group_modes(tdPgdO, 0, 3))
            tdOsdO_1, _ = cute.nvgpu.cpasync.tma_partition(tma_atom_dO, 0, cute.make_layout(1), cute.group_modes(sdO_1, 0, 2), cute.group_modes(tdPgdO, 0, 3))

            q_block_idx_0 = bucketed_k2q_indices[k2q_begin + iter_index, (blk_coord_h, blk_coord_b)]
            iter_index += 1
            q_block_idx_1 = cute.ceil_div(seqlen_q, self.sparse_block_size)  # First block beyond the final partial Q block.
            if iter_index < total_iter_count:
                q_block_idx_1 = bucketed_k2q_indices[k2q_begin + iter_index, (blk_coord_h, blk_coord_b)]
            q_block_0_full = (q_block_idx_0 + 1) * self.sparse_block_size <= seqlen_q

            if cutlass.const_expr(loader_role == 0):
                load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
                tma_barrier = load_mma_Q_pipeline.producer_get_barrier(load_mma_Q_producer_state)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_Q_bytes)

                # Load Q0
                cute.copy(
                    tma_atom_Q,
                    tQgQ_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
                    tQsQ_0,
                    tma_bar_ptr=tma_barrier,
                )

                # Load Q1
                cute.copy(
                    tma_atom_Q,
                    tQgQ_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
                    tQsQ_1,
                    tma_bar_ptr=tma_barrier,
                )

                load_mma_Q_producer_state.advance()

            if cutlass.const_expr(loader_role == 1):
                load_compute_LSE_pipeline.producer_acquire(load_compute_LSE_producer_state)

                # Load LSE
                # 32 threads load 64 values, each thread loads 2 values
                sLSE_for_copy = cute.flat_divide(sLSE, (1,))
                LSE_for_copy = cute.flat_divide(LSE, (1,))
                for i in cutlass.range_constexpr(async_copy_num_elts):
                    LSE_idx = q_block_idx_0 * self.sparse_block_size + thread_idx * async_copy_num_elts
                    if q_block_0_full:
                        cute.copy(
                            atom_async_copy,
                            LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                            sLSE_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                load_compute_LSE_producer_state.index,
                            ],
                        )
                    else:
                        _cp_async_copy_or_zero_fill(
                            atom_async_copy,
                            LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                            sLSE_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                load_compute_LSE_producer_state.index,
                            ],
                            cute.elem_less(LSE_idx + i, seqlen_q),
                        )

                for i in cutlass.range_constexpr(async_copy_num_elts):
                    LSE_idx = q_block_idx_1 * self.sparse_block_size + thread_idx * async_copy_num_elts
                    _cp_async_copy_or_zero_fill(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx + i, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[
                            None,
                            self.sparse_block_size + thread_idx * async_copy_num_elts + i,
                            load_compute_LSE_producer_state.index,
                        ],
                        cute.elem_less(LSE_idx + i, seqlen_q),
                    )

                load_compute_LSE_pipeline.producer_commit(load_compute_LSE_producer_state)
                load_compute_LSE_producer_state.advance()

                load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
                tma_barrier = load_mma_dO_pipeline.producer_get_barrier(load_mma_dO_producer_state)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_dO_bytes)

                # Load dO0
                cute.copy(
                    tma_atom_dO,
                    tdOgdO_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
                    tdOsdO_0,
                    tma_bar_ptr=tma_barrier,
                )
                # Load dO1
                cute.copy(
                    tma_atom_dO,
                    tdOgdO_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
                    tdOsdO_1,
                    tma_bar_ptr=tma_barrier,
                )

                load_mma_dO_producer_state.advance()

                load_compute_sum_OdO_pipeline.producer_acquire(load_compute_sum_OdO_producer_state)

                sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
                sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
                for i in cutlass.range_constexpr(async_copy_num_elts):
                    sum_OdO_idx = q_block_idx_0 * self.sparse_block_size + thread_idx * async_copy_num_elts
                    if q_block_0_full:
                        cute.copy(
                            atom_async_copy,
                            sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                            sSum_OdO_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                load_compute_sum_OdO_producer_state.index,
                            ],
                        )
                    else:
                        _cp_async_copy_or_zero_fill(
                            atom_async_copy,
                            sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                            sSum_OdO_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                load_compute_sum_OdO_producer_state.index,
                            ],
                            cute.elem_less(sum_OdO_idx + i, seqlen_q),
                        )
                for i in cutlass.range_constexpr(async_copy_num_elts):
                    sum_OdO_idx = q_block_idx_1 * self.sparse_block_size + thread_idx * async_copy_num_elts
                    _cp_async_copy_or_zero_fill(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx + i, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[
                            None,
                            self.sparse_block_size + thread_idx * async_copy_num_elts + i,
                            load_compute_sum_OdO_producer_state.index,
                        ],
                        cute.elem_less(sum_OdO_idx + i, seqlen_q),
                    )

                load_compute_sum_OdO_pipeline.producer_commit(load_compute_sum_OdO_producer_state)
                load_compute_sum_OdO_producer_state.advance()

            iter_count -= 2
            iter_index += 1

    @cute.jit
    def load_pair(self, atom, tensor, shared, pair, head, batch, barrier):
        layout = cute.make_layout((64, 64), stride=(64, 1))
        source = tensor[None, None, (head, batch)]
        for plane in cutlass.range_constexpr(2):
            for half in cutlass.range_constexpr(2):
                dest = cute.make_tensor(shared.iterator + plane * 8192 + half * 4096, layout)
                tile = cute.local_tile(cute.domain_offset((pair[half] * 64, plane * 64), source), (64, 64), (0, 0))
                dst, src = cpasync.tma_partition(atom, 0, cute.make_layout(1), cute.group_modes(dest, 0, 2), cute.group_modes(tile, 0, 2))
                cute.copy(atom, src, dst, tma_bar_ptr=barrier)

    @cute.jit
    def compute(
        self,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        sLSE: cute.Tensor,
        sdS: cute.Tensor,
        sP: cute.Tensor,
        sSum_OdO: cute.Tensor,
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        tdKTtdKT: cute.Tensor,
        tdVTtdVT: cute.Tensor,
        kv_block_idx: tuple,
        variable_block_sizes: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        iter_count: Int32,
        scale_softmax: Float32,
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, blk_coord_h, blk_coord_b = self.logical_block_coord(problem_shape[0])
        seqlen_q, seqlen_k, head_dim, HB = problem_shape
        num_heads, batch_size = HB
        (
            mma_compute_S_pipeline,
            compute_mma_P_pipeline,
            load_compute_LSE_pipeline,
            load_compute_sum_OdO_pipeline,
            mma_compute_dP_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args

        mma_compute_S_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage)
        compute_mma_P_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_P_stage)
        load_compute_LSE_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_compute_LSE_stage)
        load_compute_sum_OdO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_compute_sum_OdO_stage)
        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage)
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage)
        mma_compute_dKdV_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dKdV_stage)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )

        # (128, 64)
        tStS = tStS[(None, None), 0, 0]
        # (128, 64)
        tdPtdP = tdPtdP[(None, None), 0, 0]

        cS = cute.make_identity_tensor(cute.select(self.QK_mma_tiler, mode=[0, 1]))
        cdP = cute.make_identity_tensor(cute.select(self.dOV_mma_tiler, mode=[0, 1]))

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cS_p = thr_t2r.partition_D(cS)
        tTR_cS = self.split_wg(tTR_cS_p, num_warp_groups, wg_idx)
        tTR_rS = cute.make_rmem_tensor(tTR_cS.shape, self.acc_dtype)

        tTR_tS = thr_t2r.partition_S(tStS)
        tTR_tS = self.split_wg(tTR_tS, num_warp_groups, wg_idx)

        tTR_cdP_p = thr_t2r.partition_D(cdP)
        tTR_cdP = self.split_wg(tTR_cdP_p, num_warp_groups, wg_idx)
        tTR_rdP = cute.make_rmem_tensor(tTR_cdP.shape, self.acc_dtype)

        tTR_tdP = thr_t2r.partition_S(tdPtdP)
        tTR_tdP = self.split_wg(tTR_tdP, num_warp_groups, wg_idx)

        tP_packed = cute.make_tensor(tStS.iterator, cute.make_layout((128, 64), stride=(65536, 1)))
        p_store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32)
        p_store_copy = tcgen05.make_tmem_copy(p_store_atom, tP_packed)
        p_store_thr = p_store_copy.get_slice(dp_idx)
        p_store_dest = self.split_wg(p_store_thr.partition_D(tP_packed), num_warp_groups, wg_idx)

        tDS_packed = cute.make_tensor(tdPtdP.iterator, cute.make_layout((128, 64), stride=(65536, 1)))
        ds_store_dest = self.split_wg(p_store_thr.partition_D(tDS_packed), num_warp_groups, wg_idx)

        block_size_k = Int32(self.sparse_block_size)
        if cutlass.const_expr(self.has_block_sizes):
            block_size_k = variable_block_sizes[blk_coord_b, kv_block_idx[0]]

        valid_k_0, valid_k_1 = Int32(64), Int32(64)
        if cutlass.const_expr(self.has_block_sizes):
            valid_k_0 = variable_block_sizes[blk_coord_b, kv_block_idx[0]]
            valid_k_1 = variable_block_sizes[blk_coord_b, kv_block_idx[1]]
        while iter_count > 0:
            # Wait for S and P
            mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state)
            compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state)
            # Wait for LSE
            load_compute_LSE_pipeline.consumer_wait(load_compute_LSE_consumer_state)

            # Compute P = softmax(S, LSE)
            cute.copy(tiled_t2r, tTR_tS, tTR_rS)

            if cutlass.const_expr(self.has_block_sizes and self.kv_group_size == 2):
                for i in cutlass.range_constexpr(cute.size(tTR_rS)):
                    index_k, _ = tTR_cS[i]
                    valid_k = valid_k_0 if index_k < 64 else valid_k_1
                    tTR_rS[i] = tTR_rS[i] if index_k % 64 < valid_k else -Float32.inf
            else:
                if cutlass.const_expr(self.has_block_sizes):
                    # block_sizes describes valid K positions; Q rows stay full-sized.
                    if block_size_k < self.sparse_block_size:
                        for i in cutlass.range_constexpr(cute.size(tTR_rS)):
                            index_k, _ = tTR_cS[i]
                            is_valid = index_k < block_size_k
                            tTR_rS[i] = tTR_rS[i] if is_valid else -Float32.inf

            self.compute_probabilities(tTR_rS, tTR_cS, sLSE, load_compute_LSE_consumer_state.index, scale_softmax, 1)

            # convert fp32 P to bf16 P which will be used in the dOP
            tRS_rP = self.quantize(tTR_rS, 4)

            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()
            cute.arch.fence_view_async_tmem_load()

            # P overwrites S only after every compute warp has finished its TMEM load.
            packed_p = cute.recast_tensor(tRS_rP, Float32)
            cute.copy(p_store_copy, packed_p, p_store_dest)
            cute.arch.fence_view_async_tmem_store()

            # Fence for shared memory
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )

            # Notify for P
            compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
            compute_mma_P_producer_state.advance()

            # Release S
            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
            mma_compute_S_consumer_state.advance()

            # Release LSE
            load_compute_LSE_pipeline.consumer_release(load_compute_LSE_consumer_state)
            load_compute_LSE_consumer_state.advance()

            # Wait for OdO
            load_compute_sum_OdO_pipeline.consumer_wait(load_compute_sum_OdO_consumer_state)
            # Wait for dP
            mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state)

            # Wait for dS
            compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state)

            # Compute dS = dsoftmax(P, dP, sum_OdO)
            cute.copy(tiled_t2r, tTR_tdP, tTR_rdP)

            for i in cutlass.range(0, cute.size(tTR_rdP), 2, unroll_full=True):
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]),
                    (
                        sSum_OdO[
                            cute.get(tTR_cdP[i], mode=[1]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                        sSum_OdO[
                            cute.get(tTR_cdP[i + 1], mode=[1]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                    ),
                )
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.mul_packed_f32x2((tTR_rdP[i], tTR_rdP[i + 1]), (tTR_rS[i], tTR_rS[i + 1]))

            # convert fp32 dS to bf16 dS which will be used in the computation of dK and dQ
            tTR_rdS = self.quantize(tTR_rdP, 4)

            # Every warp must finish reading dP before packed dS overwrites it.
            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()
            packed_ds = cute.recast_tensor(tTR_rdS, Float32)
            cute.copy(p_store_copy, packed_ds, ds_store_dest)
            cute.arch.fence_view_async_tmem_store()

            # Release dP
            cute.arch.fence_view_async_tmem_load()
            mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
            mma_compute_dP_consumer_state.advance()

            sdS_slice = sdS[None, None, None, compute_mma_dS_producer_state.index]

            thread_layout = cute.make_ordered_layout(self.QK_mma_tiler[:2], (1, 0))
            sdS_slice_tmp = cute.composition(sdS_slice, thread_layout)
            sdS_slice_p = cute.composition(sdS_slice_tmp[dp_idx, None], cute.make_layout(tTR_cdP_p.shape))
            sdS_slice = self.split_wg(sdS_slice_p, num_warp_groups, wg_idx)

            cute.autovec_copy(tTR_rdS, sdS_slice)

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
            compute_mma_dS_producer_state.advance()

            # Release OdO
            load_compute_sum_OdO_pipeline.consumer_release(load_compute_sum_OdO_consumer_state)
            load_compute_sum_OdO_consumer_state.advance()

            iter_count -= 1

        self.epilogue(
            problem_shape,
            dK_acc,
            dV_acc,
            tdKTtdKT,
            tdVTtdVT,
            kv_block_idx,
            (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state),
        )

    @cute.jit
    def epilogue(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        tdKTtdKT: cute.Tensor,
        tdVTtdVT: cute.Tensor,
        kv_block_idx: tuple,
        # (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state)
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        Q, K, D, HB = problem_shape
        H, B = HB
        _, blk_coord_h, blk_coord_b = self.logical_block_coord(problem_shape[0])
        mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state = pipeline_args

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )

        tdKTtdKT = tdKTtdKT[(None, None), 0, 0]
        tdVTtdVT = tdVTtdVT[(None, None), 0, 0]

        dK_acc = cute.make_tensor(dK_acc.iterator, cute.select(dK_acc.layout, mode=[1, 0, 2]))
        dV_acc = cute.make_tensor(dV_acc.iterator, cute.select(dV_acc.layout, mode=[1, 0, 2]))

        cdK = cute.domain_offset((0, 0), cute.make_identity_tensor((self.QdS_mma_tiler[0], self.QdS_mma_tiler[1])))
        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        tiled_t2r_dK = tcgen05.make_tmem_copy(load_op, tdKTtdKT)
        thr_t2r_dK = tiled_t2r_dK.get_slice(dp_idx)

        tTR_cdK = thr_t2r_dK.partition_D(cdK)
        tTR_cdK = self.split_wg(tTR_cdK, num_warp_groups, wg_idx)
        tTR_rdK = cute.make_rmem_tensor(tTR_cdK.shape, self.acc_dtype)
        tTR_tdK = thr_t2r_dK.partition_S(tdKTtdKT)
        tTR_tdK = self.split_wg(tTR_tdK, num_warp_groups, wg_idx)

        cdV = cute.domain_offset((0, 0), cute.make_identity_tensor((self.dOP_mma_tiler[0], self.dOP_mma_tiler[1])))

        tiled_t2r_dV = tcgen05.make_tmem_copy(load_op, tdVTtdVT)
        thr_t2r_dV = tiled_t2r_dV.get_slice(dp_idx)

        tTR_cdV = thr_t2r_dV.partition_D(cdV)
        tTR_cdV = self.split_wg(tTR_cdV, num_warp_groups, wg_idx)
        tTR_rdV = cute.make_rmem_tensor(tTR_cdV.shape, self.acc_dtype)
        tTR_tdV = thr_t2r_dV.partition_S(tdVTtdVT)
        tTR_tdV = self.split_wg(tTR_tdV, num_warp_groups, wg_idx)

        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)

        # Load tdVtdVT
        cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)

        self.store_pair(dV_acc, tTR_rdV, tTR_cdV, kv_block_idx, blk_coord_h, blk_coord_b, K, D)

        cute.arch.fence_view_async_tmem_load()

        mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)

        cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)

        self.store_pair(dK_acc, tTR_rdK, tTR_cdK, kv_block_idx, blk_coord_h, blk_coord_b, K, D)

        cute.arch.fence_view_async_tmem_load()
        mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

    @cute.jit
    def store_pair(self, output, regs, coords, kv, head, batch, K, D):
        values = cute.make_rmem_tensor((2,), Float32)
        for pair_idx in cutlass.range_constexpr(cute.size(regs) // 2):
            i = pair_idx * 2
            row, d = coords[i]
            next_row, next_d = coords[i + 1]
            k = (kv[0] if row < 64 else kv[1]) * 64 + row % 64
            if next_row == row and next_d == d + 1 and d % 2 == 0 and d + 1 < D and k < K:
                ptr = output.iterator + cute.crd2idx((k, d, (head, batch)), output.layout)
                values[0], values[1] = regs[i], regs[i + 1]
                cute.arch.atomic_add(ptr.llvm_ptr, values.load(), sem="relaxed", scope="gpu")
            else:
                for j in cutlass.range_constexpr(2):
                    row_j, d_j = coords[i + j]
                    k_j = (kv[0] if row_j < 64 else kv[1]) * 64 + row_j % 64
                    if k_j < K and d_j < D:
                        ptr = output.iterator + cute.crd2idx((k_j, d_j, (head, batch)), output.layout)
                        cute.arch.atomic_add(ptr.llvm_ptr, regs[i + j], sem="relaxed", scope="gpu")

    @cute.jit
    def mma(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        tStS: cute.Tensor,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdPrdO: cute.Tensor,
        tdPrV: cute.Tensor,
        tdVTtdVT: cute.Tensor,
        tdVTrdOT: cute.Tensor,
        tdVTrP: cute.Tensor,
        tdQtdQ: cute.Tensor,
        tdQrdS: cute.Tensor,
        tdQrKT: cute.Tensor,
        tdKTtdKT: cute.Tensor,
        tdKTrQT: cute.Tensor,
        tdKTrdST: cute.Tensor,
        iter_count: Int32,
        pipeline_args: tuple,
    ):
        (
            load_mma_Q_pipeline,
            mma_compute_S_pipeline,
            load_mma_dO_pipeline,
            mma_compute_dP_pipeline,
            mma_reduce_dQ_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args

        load_mma_Q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_Q_stage)
        load_mma_Q_release_state = load_mma_Q_consumer_state.clone()
        mma_compute_S_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_S_stage)
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage)
        mma_compute_dP_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage)
        mma_reduce_dQ_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_reduce_dQ_stage)
        load_mma_dO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_dO_stage)
        compute_mma_P_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_P_stage)
        mma_compute_dKdV_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dKdV_stage)

        dOP_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.dOP_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        p_layout = sm100_utils.make_smem_layout_a(dOP_tiled_mma, self.dOP_mma_tiler, self.element_dtype, 1)
        p_tensor = cute.make_tensor(cute.recast_ptr(tStS.iterator, dtype=self.element_dtype), p_layout.outer)
        tdVTrP = dOP_tiled_mma.make_fragment_A(p_tensor)
        tdVTrP = cute.make_tensor(p_tensor.iterator, tdVTrP.layout)

        QdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.QdS_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        ds_layout = sm100_utils.make_smem_layout_a(QdS_tiled_mma, self.QdS_mma_tiler, self.element_dtype, 1)
        ds_tensor = cute.make_tensor(cute.recast_ptr(tdPtdP.iterator, dtype=self.element_dtype), ds_layout.outer)
        tdKTrdST = QdS_tiled_mma.make_fragment_A(ds_tensor)
        tdKTrdST = cute.make_tensor(ds_tensor.iterator, tdKTrdST.layout)

        load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
        mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)

        # S = Q * K
        QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll_full=True):
            cute.gemm(
                QK_tiled_mma,
                tStS,
                tSrK[None, None, k_block, 0],
                tSrQ[None, None, k_block, load_mma_Q_consumer_state.index],
                tStS,
            )
            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        load_mma_Q_consumer_state.advance()
        mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
        mma_compute_S_producer_state.advance()

        load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

        mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)
        mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)

        # dP = dO * V
        dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tdPrdO, mode=[2]), unroll_full=True):
            cute.gemm(
                dOV_tiled_mma,
                tdPtdP,
                tdPrV[None, None, k_block, 0],
                tdPrdO[None, None, k_block, load_mma_dO_consumer_state.index],
                tdPtdP,
            )
            dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
        mma_compute_dP_producer_state.advance()

        compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)

        # dV = dO * P
        dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tdVTrdOT, mode=[2]), unroll_full=True):
            cute.gemm(
                dOP_tiled_mma,
                tdVTtdVT,
                tdVTrP[None, None, k_block, 0],
                tdVTrdOT[None, None, k_block, load_mma_dO_consumer_state.index],
                tdVTtdVT,
            )
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
        compute_mma_P_consumer_state.advance()

        load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
        load_mma_dO_consumer_state.advance()

        iter_count -= 1

        while iter_count > 0:
            load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
            mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)

            # S = Q * K
            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll_full=True):
                cute.gemm(
                    QK_tiled_mma,
                    tStS,
                    tSrK[None, None, k_block, 0],
                    tSrQ[None, None, k_block, load_mma_Q_consumer_state.index],
                    tStS,
                )
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            load_mma_Q_consumer_state.advance()
            mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()

            compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

            mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)

            # dK = Q * dS
            for k_block in cutlass.range(0, cute.size(tdKTrQT, mode=[2]), unroll_full=True):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKTtdKT,
                    tdKTrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKTrQT[None, None, k_block, load_mma_Q_release_state.index],
                    tdKTtdKT,
                )
                QdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ = dS * K
            dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdQrdS, mode=[2]), unroll_full=True):
                cute.gemm(
                    dSK_tiled_mma,
                    tdQtdQ,
                    tdQrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQrKT[None, None, k_block, 0],
                    tdQtdQ,
                )
                dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
            mma_reduce_dQ_producer_state.advance()

            load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
            load_mma_Q_release_state.advance()

            compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()

            mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
            load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

            # dP = dO * V
            dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdPrdO, mode=[2]), unroll_full=True):
                cute.gemm(
                    dOV_tiled_mma,
                    tdPtdP,
                    tdPrV[None, None, k_block, 0],
                    tdPrdO[None, None, k_block, load_mma_dO_consumer_state.index],
                    tdPtdP,
                )
                dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)

            # dV = dO * P
            for k_block in cutlass.range(0, cute.size(tdVTrdOT, mode=[2]), unroll_full=True):
                cute.gemm(
                    dOP_tiled_mma,
                    tdVTtdVT,
                    tdVTrP[None, None, k_block, 0],
                    tdVTrdOT[None, None, k_block, load_mma_dO_consumer_state.index],
                    tdVTtdVT,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
            compute_mma_P_consumer_state.advance()

            load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
            load_mma_dO_consumer_state.advance()

            iter_count -= 1

        mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_producer_state)
        mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
        mma_compute_dKdV_producer_state.advance()

        mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_producer_state)

        compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

        # dK = Q * dS
        for k_block in cutlass.range(0, cute.size(tdKTrQT, mode=[2]), unroll_full=True):
            cute.gemm(
                QdS_tiled_mma,
                tdKTtdKT,
                tdKTrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                tdKTrQT[None, None, k_block, load_mma_Q_release_state.index],
                tdKTtdKT,
            )
            QdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        # dQ = dS * K
        dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tdQrdS, mode=[2]), unroll_full=True):
            cute.gemm(
                dSK_tiled_mma,
                tdQtdQ,
                tdQrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                tdQrKT[None, None, k_block, 0],
                tdQtdQ,
            )
            dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
        mma_reduce_dQ_producer_state.advance()

        # dK epilogue permits TMEM deallocation: wait for the final dQ reader first.
        mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
        mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
        mma_compute_dKdV_producer_state.advance()

        load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
        load_mma_Q_release_state.advance()

        compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
        compute_mma_dS_consumer_state.advance()
