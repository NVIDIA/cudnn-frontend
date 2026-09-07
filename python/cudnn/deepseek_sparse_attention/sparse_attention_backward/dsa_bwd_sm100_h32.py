# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05

from .dsa_bwd_sm100_h16 import FlashAttentionDSABackwardSm100H16


class FlashAttentionDSABackwardSm100H32(FlashAttentionDSABackwardSm100H16):
    """H32/D576 single-query specialization with an M64 sparse-row tile.

    P and dS stream through one 64-row shared-memory shuttle. Keeping the
    class separate prevents H32 scheduling changes from perturbing the tuned
    H16 and generic H64 code generation.
    """

    def __init__(
        self,
        element_dtype,
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
    ):
        """Configure H32/D576 M64 tiling, TMEM offsets, and register budgets."""
        # Initialize the shared H16 machinery with its native K128 shape;
        # every shape-dependent tensor below is then rebuilt for this class.
        super().__init__(
            element_dtype,
            head_dim,
            head_dim_v,
            128,
            max_topk,
        )
        if head_dim != 576 or head_dim_v != 512 or block_tile != 64:
            raise ValueError("H32 M64 requires head_dim=576, head_dim_v=512, and block_tile=64")

        self.block_tile = block_tile

        self.h_tile = 32
        self.kv_subtile = 64
        self.num_kv_subtiles = block_tile // self.kv_subtile
        self.reduce_rows_per_thread = self.kv_subtile // self.num_reduce_warps

        # Score/dP follow the M64 sparse-row tile. The later GEMMs consume the
        # same 64-row subtile, so P/dS remain 4 KiB each.
        self.QK_mma_tiler = (block_tile, self.h_tile, head_dim)
        self.dOV_mma_tiler = (block_tile, self.h_tile, head_dim_v)
        self.dOP_mma_tiler = (128, self.kv_subtile, self.h_tile)
        self.dOP_cta_tiler = (head_dim_v, self.kv_subtile, self.h_tile)
        self.KdS_mma_tiler = (128, self.h_tile, self.kv_subtile)
        # Keep the 64-row tile addressable through the K operand view; each
        # UMMA instruction reduces over one K64 subtile.
        self.KdS_cta_tiler = (self.head_dim_main, self.h_tile, block_tile)
        self.QdS_mma_tiler = (128, self.kv_subtile, self.h_tile)
        self.QdS_cta_tiler = (self.head_dim_main, self.kv_subtile, self.h_tile)
        self.dQ4_mma_tiler = (64, self.h_tile, self.kv_subtile)
        self.dKV4_mma_tiler = (64, self.kv_subtile, self.h_tile)

        # S/dP remain live through their downstream consumers. dKV4 therefore
        # gets a disjoint region instead of aliasing their columns.
        self.tmem_S_offset = 0
        self.tmem_dP_offset = 32
        self.tmem_dKV0_offset = 64
        self.tmem_dKV1_offset = 128
        self.tmem_dKV2_offset = self.tmem_dKV0_offset
        self.tmem_dKV3_offset = self.tmem_dKV1_offset
        self.tmem_dQ0_offset = 192
        self.tmem_dQ1_offset = 224
        self.tmem_dQ2_offset = 256
        self.tmem_dQ3_offset = 288
        self.tmem_dQ4_offset = 320
        self.tmem_dKV4_offset = 352

        self.num_regs_load_KV = 32
        self.num_regs_compute = 160

    @cute.jit
    def mma(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        KdS_tiled_mma: cute.TiledMma,
        dKV4_tiled_mma: cute.TiledMma,
        dQ4_tiled_mma: cute.TiledMma,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tdPrdO: cute.Tensor,
        tdPrV: cute.Tensor,
        tdKVrdOT: cute.Tensor,
        tdKVrP: cute.Tensor,
        tdQrK: cute.Tensor,
        tdQrdST: cute.Tensor,
        tdKVrQT: cute.Tensor,
        tdKVrdS: cute.Tensor,
        tdQrK_tail: cute.Tensor,
        tdKVrQT_tail: cute.Tensor,
        tdKVrdS_4: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdKVtdKV: tuple,
        tdQtdQ: tuple,
        tile_count: Int32,
        sdS: cute.Tensor,
        pipelines,
    ):
        """One sparse-row load/score followed by its M64 generations."""
        (
            load_mma_QdO_pipeline,
            load_mma_K_pipeline,
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            mma_compute_dQ_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_reduce_dKV_pipeline,
        ) = pipelines
        tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4 = tdKVtdKV
        tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4 = tdQtdQ

        load_mma_QdO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_QdO_stage)
        load_mma_K_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_K_stage)
        mma_compute_S_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_S_stage)
        mma_compute_dP_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage)
        mma_compute_dQ_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dQ_stage)
        compute_mma_P_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_P_stage)
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage)
        mma_reduce_dKV_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_reduce_dKV_stage)

        load_mma_QdO_pipeline.consumer_wait(load_mma_QdO_consumer_state)
        mma_compute_dQ_pipeline.producer_acquire(mma_compute_dQ_producer_state)
        tile_index = tile_count - 1
        is_first_generation = True
        is_first_dkv_half = True
        while tile_index >= 0:
            load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state)

            mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)
            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll=4):
                cute.gemm(
                    QK_tiled_mma,
                    tStS,
                    tSrQ[None, None, k_block, load_mma_QdO_consumer_state.index],
                    tSrK[None, None, k_block, load_mma_K_consumer_state.index],
                    tStS,
                )
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()

            mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)
            dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdPrdO, mode=[2]), unroll=4):
                cute.gemm(
                    dOV_tiled_mma,
                    tdPtdP,
                    tdPrdO[None, None, k_block, load_mma_QdO_consumer_state.index],
                    tdPrV[None, None, k_block, load_mma_K_consumer_state.index],
                    tdPtdP,
                )
                dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            k_blocks_per_half = cute.size(tdQrdST, mode=[2])
            # Preserve the generic M64 kernel's descending sparse-tile order
            # inside each K128 tile so FP32 dQ accumulation is not needlessly
            # reordered by the optimization.
            for half_iter in cutlass.range_constexpr(self.num_kv_subtiles):
                kv_half = self.num_kv_subtiles - 1 - half_iter
                compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)
                mma_reduce_dKV_pipeline.producer_acquire(mma_reduce_dKV_producer_state)
                if not is_first_dkv_half:
                    # dKV2/3 alias dKV0/1.  A two-stage ring otherwise waits
                    # on the intervening tail generation instead of the
                    # aliased TMEM columns from the preceding half.
                    self.t2r_dKV23_done_barrier.arrive_and_wait()

                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                    cute.gemm(
                        dOP_tiled_mma,
                        tdKVtdKV0,
                        tdKVrdOT[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                        tdKVtdKV0,
                    )
                    dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                    cute.gemm(
                        dOP_tiled_mma,
                        tdKVtdKV1,
                        tdKVrdOT[None, None, 1, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                        tdKVtdKV1,
                    )
                    dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)
                QdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                for k_block in cutlass.range(0, cute.size(tdKVrdS, mode=[2]), unroll=2):
                    cute.gemm(
                        QdS_tiled_mma,
                        tdKVtdKV0,
                        tdKVrQT[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdKVtdKV0,
                    )
                    cute.gemm(
                        QdS_tiled_mma,
                        tdKVtdKV1,
                        tdKVrQT[None, None, 1, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdKVtdKV1,
                    )
                mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
                mma_reduce_dKV_producer_state.advance()

                # On a non-final subtile, retain the original reducer overlap.
                # On the final subtile, defer the K-independent dKV tail until after dQ
                # so the K buffer can be released one MMA group earlier.
                if half_iter != self.num_kv_subtiles - 1:
                    mma_reduce_dKV_pipeline.producer_acquire(mma_reduce_dKV_producer_state)
                    dKV4_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_block in cutlass.range(0, cute.size(tdKVrdS_4, mode=[2]), unroll=2):
                        cute.gemm(
                            dKV4_tiled_mma,
                            tdKVtdKV4,
                            tdKVrQT_tail[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                            tdKVrdS_4[None, None, k_block, compute_mma_dS_consumer_state.index],
                            tdKVtdKV4,
                        )
                        dKV4_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
                    mma_reduce_dKV_producer_state.advance()

                accumulate_dq = not is_first_generation
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, accumulate_dq)
                for k_block in cutlass.range(0, k_blocks_per_half, unroll=2):
                    k_half_block = kv_half * k_blocks_per_half + k_block
                    cute.gemm(
                        KdS_tiled_mma,
                        tdQtdQ0,
                        tdQrK[None, None, 0, k_half_block, load_mma_K_consumer_state.index],
                        tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdQtdQ0,
                    )
                    KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, accumulate_dq)
                for k_block in cutlass.range(0, k_blocks_per_half, unroll=2):
                    k_half_block = kv_half * k_blocks_per_half + k_block
                    cute.gemm(
                        KdS_tiled_mma,
                        tdQtdQ1,
                        tdQrK[None, None, 1, k_half_block, load_mma_K_consumer_state.index],
                        tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdQtdQ1,
                    )
                    KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, accumulate_dq)
                for k_block in cutlass.range(0, k_blocks_per_half, unroll=2):
                    k_half_block = kv_half * k_blocks_per_half + k_block
                    cute.gemm(
                        KdS_tiled_mma,
                        tdQtdQ2,
                        tdQrK[None, None, 2, k_half_block, load_mma_K_consumer_state.index],
                        tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdQtdQ2,
                    )
                    KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, accumulate_dq)
                for k_block in cutlass.range(0, k_blocks_per_half, unroll=2):
                    k_half_block = kv_half * k_blocks_per_half + k_block
                    cute.gemm(
                        KdS_tiled_mma,
                        tdQtdQ3,
                        tdQrK[None, None, 3, k_half_block, load_mma_K_consumer_state.index],
                        tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdQtdQ3,
                    )
                    KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                dQ4_tiled_mma.set(tcgen05.Field.ACCUMULATE, accumulate_dq)
                for k_block in cutlass.range(0, k_blocks_per_half, unroll=2):
                    k_half_block = kv_half * k_blocks_per_half + k_block
                    cute.gemm(
                        dQ4_tiled_mma,
                        tdQtdQ4,
                        tdQrK_tail[None, None, 0, k_half_block, load_mma_K_consumer_state.index],
                        tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdQtdQ4,
                    )
                    dQ4_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                # The final M64 subtile is the last consumer of this K gather.
                # Release the single-stage K buffer before dKV2/3,
                # which only consume Q/dO and P/dS, so the loaders can gather
                # the next sparse tile under the remaining MMA and atomics.
                if half_iter == self.num_kv_subtiles - 1:
                    load_mma_K_pipeline.consumer_release(load_mma_K_consumer_state)
                    load_mma_K_consumer_state.advance()

                    mma_reduce_dKV_pipeline.producer_acquire(mma_reduce_dKV_producer_state)
                    dKV4_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_block in cutlass.range(0, cute.size(tdKVrdS_4, mode=[2]), unroll=2):
                        cute.gemm(
                            dKV4_tiled_mma,
                            tdKVtdKV4,
                            tdKVrQT_tail[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                            tdKVrdS_4[None, None, k_block, compute_mma_dS_consumer_state.index],
                            tdKVtdKV4,
                        )
                        dKV4_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
                    mma_reduce_dKV_producer_state.advance()

                mma_reduce_dKV_pipeline.producer_acquire(mma_reduce_dKV_producer_state)
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                    cute.gemm(
                        dOP_tiled_mma,
                        tdKVtdKV2,
                        tdKVrdOT[None, None, 2, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                        tdKVtdKV2,
                    )
                    dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                    cute.gemm(
                        dOP_tiled_mma,
                        tdKVtdKV3,
                        tdKVrdOT[None, None, 3, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                        tdKVtdKV3,
                    )
                    dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
                compute_mma_P_consumer_state.advance()

                for k_block in cutlass.range(0, cute.size(tdKVrdS, mode=[2]), unroll=2):
                    cute.gemm(
                        QdS_tiled_mma,
                        tdKVtdKV2,
                        tdKVrQT[None, None, 2, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdKVtdKV2,
                    )
                    cute.gemm(
                        QdS_tiled_mma,
                        tdKVtdKV3,
                        tdKVrQT[None, None, 3, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdKVtdKV3,
                    )
                mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
                mma_reduce_dKV_producer_state.advance()
                compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
                compute_mma_dS_consumer_state.advance()
                is_first_generation = False
                is_first_dkv_half = False

            tile_index -= 1

        # Balance the reducer's final one-way dKV2/3 T2R notification.
        self.t2r_dKV23_done_barrier.arrive_and_wait()
        mma_compute_dQ_pipeline.producer_commit(mma_compute_dQ_producer_state)
        mma_compute_dQ_producer_state.advance()
        load_mma_QdO_pipeline.consumer_release(load_mma_QdO_consumer_state)
        load_mma_QdO_consumer_state.advance()

    @cute.jit
    def compute(
        self,
        tma_atom_dQ: cute.CopyAtom,
        tma_tensor_dQ: cute.Tensor,
        tma_atom_dQ_64: cute.CopyAtom,
        tma_tensor_dQ_64: cute.Tensor,
        dQ4_tiled_mma: cute.TiledMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdQtdQ: tuple,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        sP: cute.Tensor,
        sP_store: cute.Tensor,
        sdS: cute.Tensor,
        sdS_store: cute.Tensor,
        sdQ: cute.Tensor,
        sdQ4: cute.Tensor,
        scale_softmax: Float32,
        tile_count: Int32,
        pipelines,
    ):
        """Stream each M64 score/dP result through one physical M64 slot."""
        (
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            load_compute_LSE_pipeline,
            load_compute_sum_OdO_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dQ_pipeline,
            compute_tmastore_dQ_pipeline,
        ) = pipelines
        tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4 = tdQtdQ

        tidx, _, _ = cute.arch.thread_idx()
        token_idx, head_block_idx, batch_idx = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        mma_compute_S_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage)
        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage)
        mma_compute_dQ_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dQ_stage)
        compute_mma_P_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_P_stage)
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage)
        load_compute_LSE_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_compute_LSE_stage)
        load_compute_sum_OdO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_compute_sum_OdO_stage)
        compute_tmastore_dQ_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_tmastore_dQ_stage)

        # N32 requires four 256-bit repetitions.  Keep the native TMEM lane
        # mapping for the selected M{128,64} score tile.
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tStS = tStS[(None, None), 0, 0]
        tdPtdP = tdPtdP[(None, None), 0, 0]
        cS = cute.make_identity_tensor((self.block_tile, self.h_tile))
        cdP = cute.make_identity_tensor((self.block_tile, self.h_tile))

        tiled_t2r_S = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        tiled_t2r_dP = tcgen05.make_tmem_copy(tmem_load_atom, tdPtdP)
        thr_t2r_S = tiled_t2r_S.get_slice(tidx % 128)
        thr_t2r_dP = tiled_t2r_dP.get_slice(tidx % 128)
        tTR_cS = thr_t2r_S.partition_D(cS)
        tTR_tS = thr_t2r_S.partition_S(tStS)
        tTR_cdP = thr_t2r_dP.partition_D(cdP)
        tTR_tdP = thr_t2r_dP.partition_S(tdPtdP)
        tTR_rS = cute.make_rmem_tensor(tTR_cS.shape, self.acc_dtype)
        tTR_rdP = cute.make_rmem_tensor(tTR_cdP.shape, self.acc_dtype)
        tTR_rS_f16 = cute.make_rmem_tensor(tTR_cS.shape, self.element_dtype)
        tTR_rdP_f16 = cute.make_rmem_tensor(tTR_cdP.shape, self.element_dtype)

        load_compute_LSE_pipeline.consumer_wait(load_compute_LSE_consumer_state)
        load_compute_sum_OdO_pipeline.consumer_wait(load_compute_sum_OdO_consumer_state)
        log2_e = Float32(math.log2(math.e))
        softmax_scale_log2_e = scale_softmax * log2_e

        tile_index = tile_count - 1
        while tile_index >= 0:
            mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state)
            mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state)

            cute.copy(tiled_t2r_S, tTR_tS, tTR_rS)
            for i in cutlass.range(0, cute.size(tTR_rS), 2, unroll_full=True):
                lse = (
                    sLSE[cute.get(tTR_cS[i], mode=[1]), load_compute_LSE_consumer_state.index],
                    sLSE[cute.get(tTR_cS[i + 1], mode=[1]), load_compute_LSE_consumer_state.index],
                )
                tTR_rS[i], tTR_rS[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTR_rS[i], tTR_rS[i + 1]),
                    (softmax_scale_log2_e, softmax_scale_log2_e),
                    lse,
                )
                tTR_rS[i] = cute.math.exp2(tTR_rS[i], fastmath=True)
                tTR_rS[i + 1] = cute.math.exp2(tTR_rS[i + 1], fastmath=True)
            tTR_rS_f16 = self.quantize(tTR_rS, 2)

            cute.copy(tiled_t2r_dP, tTR_tdP, tTR_rdP)
            for i in cutlass.range(0, cute.size(tTR_rdP), 2, unroll_full=True):
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]),
                    (
                        sSum_OdO[cute.get(tTR_cdP[i], mode=[1]), load_compute_sum_OdO_consumer_state.index],
                        sSum_OdO[cute.get(tTR_cdP[i + 1], mode=[1]), load_compute_sum_OdO_consumer_state.index],
                    ),
                )
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.mul_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]),
                    (tTR_rS[i], tTR_rS[i + 1]),
                )
            tTR_rdP_f16 = self.quantize(tTR_rdP, 2, scale_softmax)

            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()

            for half_iter in cutlass.range_constexpr(self.num_kv_subtiles):
                kv_half = self.num_kv_subtiles - 1 - half_iter

                compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state)
                p_stage = 0 if self.compute_mma_P_stage == 1 else compute_mma_P_producer_state.index
                for i in cutlass.range_constexpr(cute.size(tTR_rS_f16)):
                    global_row = cute.get(tTR_cS[i], mode=[0])
                    if global_row // self.kv_subtile == kv_half:
                        row = global_row - kv_half * self.kv_subtile
                        col = cute.get(tTR_cS[i], mode=[1])
                        sP[(row, col), 0, 0, p_stage] = tTR_rS_f16[i]
                cute.arch.fence_proxy("async.shared", space="cta")
                compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
                compute_mma_P_producer_state.advance()

                compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state)
                ds_stage = 0 if self.compute_mma_dS_stage == 1 else compute_mma_dS_producer_state.index
                for i in cutlass.range_constexpr(cute.size(tTR_rdP_f16)):
                    global_row = cute.get(tTR_cdP[i], mode=[0])
                    if global_row // self.kv_subtile == kv_half:
                        row = global_row - kv_half * self.kv_subtile
                        col = cute.get(tTR_cdP[i], mode=[1])
                        sdS[(row, col), 0, 0, ds_stage] = tTR_rdP_f16[i]
                cute.arch.fence_proxy("async.shared", space="cta")
                compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
                compute_mma_dS_producer_state.advance()

            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
            mma_compute_S_consumer_state.advance()
            mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
            mma_compute_dP_consumer_state.advance()
            tile_index -= 1

        load_compute_LSE_pipeline.consumer_release(load_compute_LSE_consumer_state)
        load_compute_sum_OdO_pipeline.consumer_release(load_compute_sum_OdO_consumer_state)

        # Persistent dQ epilogue; identical ordering to H16, but every tile is
        # M{128,64} x N32 and therefore occupies 32 TMEM columns.
        tdQtdQ0 = tdQtdQ0[(None, None), 0, 0]
        tdQtdQ1 = tdQtdQ1[(None, None), 0, 0]
        tdQtdQ2 = tdQtdQ2[(None, None), 0, 0]
        tdQtdQ3 = tdQtdQ3[(None, None), 0, 0]
        gdQ = cute.local_tile(tma_tensor_dQ, cute.select(self.KdS_mma_tiler, mode=[0, 1]), (None, None, (token_idx, batch_idx)))
        gdQ0 = gdQ[None, None, 0, head_block_idx]
        gdQ1 = gdQ[None, None, 1, head_block_idx]
        gdQ2 = gdQ[None, None, 2, head_block_idx]
        gdQ3 = gdQ[None, None, 3, head_block_idx]
        sdQ_slice = sdQ[None, None, mma_compute_dQ_consumer_state.index]
        tdQsdQ0, tdQgdQ0_mkl = cpasync.tma_partition(tma_atom_dQ, 0, cute.make_layout(1), cute.group_modes(sdQ_slice, 0, 2), cute.group_modes(gdQ0, 0, 2))
        tdQsdQ1, tdQgdQ1_mkl = cpasync.tma_partition(tma_atom_dQ, 0, cute.make_layout(1), cute.group_modes(sdQ_slice, 0, 2), cute.group_modes(gdQ1, 0, 2))
        tdQsdQ2, tdQgdQ2_mkl = cpasync.tma_partition(tma_atom_dQ, 0, cute.make_layout(1), cute.group_modes(sdQ_slice, 0, 2), cute.group_modes(gdQ2, 0, 2))
        tdQsdQ3, tdQgdQ3_mkl = cpasync.tma_partition(tma_atom_dQ, 0, cute.make_layout(1), cute.group_modes(sdQ_slice, 0, 2), cute.group_modes(gdQ3, 0, 2))

        tdQtdQ4 = tdQtdQ4[(None, None), 0, 0]
        gdQ4 = cute.local_tile(tma_tensor_dQ_64, cute.select(self.dQ4_mma_tiler, mode=[0, 1]), (None, None, (token_idx, batch_idx)))
        gdQ4 = gdQ4[None, None, 8, head_block_idx]
        sdQ4_slice = sdQ4[None, None, mma_compute_dQ_consumer_state.index]
        tdQsdQ4, tdQgdQ4_mkl = cpasync.tma_partition(tma_atom_dQ_64, 0, cute.make_layout(1), cute.group_modes(sdQ4_slice, 0, 2), cute.group_modes(gdQ4, 0, 2))

        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
        mma_compute_dQ_pipeline.consumer_wait(mma_compute_dQ_consumer_state)

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()
        self.store_dQ(tma_atom_dQ, sdQ_slice, tdQsdQ0, tdQgdQ0_mkl, tdQtdQ0, dp_idx, warp_idx)
        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()
        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()
        self.store_dQ(tma_atom_dQ, sdQ_slice, tdQsdQ1, tdQgdQ1_mkl, tdQtdQ1, dp_idx, warp_idx)
        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()
        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()
        self.store_dQ(tma_atom_dQ, sdQ_slice, tdQsdQ2, tdQgdQ2_mkl, tdQtdQ2, dp_idx, warp_idx)
        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()
        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()
        self.store_dQ(tma_atom_dQ, sdQ_slice, tdQsdQ3, tdQgdQ3_mkl, tdQtdQ3, dp_idx, warp_idx)
        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()
        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()
        self.store_dQ_64(tma_atom_dQ_64, sdQ4_slice, tdQsdQ4, tdQgdQ4_mkl, tdQtdQ4, dp_idx, wg_idx, warp_idx)
        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()
        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        mma_compute_dQ_pipeline.consumer_release(mma_compute_dQ_consumer_state)
        mma_compute_dQ_consumer_state.advance()
        compute_tmastore_dQ_pipeline.producer_tail()

    @cute.jit
    def store_dQ(
        self,
        tma_atom_dQ: cute.CopyAtom,
        sdQ: cute.Tensor,
        tdQsdQ: cute.Tensor,
        tdQgdQ_mkl: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dp_idx: Int32,
        warp_idx: Int32,
    ):
        """Store one M128xN32 main dQ accumulator."""
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        cdQ = cute.make_identity_tensor(cute.select(self.KdS_mma_tiler, mode=[0, 1]))
        tiled_t2r_dQ = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_t2r_dQ = tiled_t2r_dQ.get_slice(dp_idx)
        tTR_cdQ = thr_t2r_dQ.partition_D(cdQ)
        tTR_rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)
        tTR_tdQ = thr_t2r_dQ.partition_S(tdQtdQ)

        cute.copy(tiled_t2r_dQ, tTR_tdQ, tTR_rdQ)
        tTR_rdQ_f16 = self.quantize(tTR_rdQ, 2)
        cute.arch.fence_view_async_tmem_load()

        # TMEM's register fragment is swizzled.  Compose it with the physical
        # M128xN32 epilogue layout instead of treating its coordinate tensor
        # as a scalar-store map.
        thread_layout = cute.make_ordered_layout((128, self.h_tile), (0, 1))
        sdQ_slice_tmp = cute.composition(sdQ, thread_layout)
        sdQ_slice = cute.composition(sdQ_slice_tmp[dp_idx, None], cute.make_layout(tTR_cdQ.shape))
        cute.autovec_copy(tTR_rdQ_f16, sdQ_slice)

        self.compute_sync_barrier.arrive_and_wait()
        cute.arch.fence_proxy("async.shared", space="cta")
        self.compute_sync_barrier.arrive_and_wait()
        if warp_idx == self.compute_warp_id[0]:
            cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ_mkl)

    @cute.jit
    def store_dQ_64(
        self,
        tma_atom_dQ: cute.CopyAtom,
        sdQ: cute.Tensor,
        tdQsdQ: cute.Tensor,
        tdQgdQ_mkl: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dp_idx: Int32,
        wg_idx: Int32,
        warp_idx: Int32,
    ):
        """Store the M64xN32 dQ tail accumulator."""
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(2)),
            self.acc_dtype,
        )
        cdQ = cute.make_identity_tensor(cute.select(self.dQ4_mma_tiler, mode=[0, 1]))
        tiled_t2r_dQ = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_t2r_dQ = tiled_t2r_dQ.get_slice(dp_idx)
        tTR_cdQ = thr_t2r_dQ.partition_D(cdQ)
        tTR_rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)
        tTR_tdQ = thr_t2r_dQ.partition_S(tdQtdQ)

        cute.copy(tiled_t2r_dQ, tTR_tdQ, tTR_rdQ)
        cute.arch.fence_view_async_tmem_load()

        for i in cutlass.range_constexpr(cute.size(tTR_rdQ)):
            row = cute.get(tTR_cdQ[i], mode=[0])
            col = cute.get(tTR_cdQ[i], mode=[1])
            sdQ[row, col] = self.element_dtype(tTR_rdQ[i])

        self.compute_sync_barrier.arrive_and_wait()
        cute.arch.fence_proxy("async.shared", space="cta")
        self.compute_sync_barrier.arrive_and_wait()
        if warp_idx == self.compute_warp_id[0]:
            cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ_mkl)

    @cute.jit
    def _t2r_dKV_main(self, tdKVtdKV: cute.Tensor):
        """Load one M128xN64 dKV fragment into reducer registers."""
        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r = tiled_t2r.get_slice(dp_idx)
        cdKV = cute.make_identity_tensor((self.dOP_mma_tiler[0], self.dOP_mma_tiler[1]))
        tTR_cdKV = self.split_wg(thr_t2r.partition_D(cdKV), num_warp_groups, wg_idx)
        tTR_rdKV = cute.make_rmem_tensor(tTR_cdKV.shape, self.acc_dtype)
        tTR_tdKV = self.split_wg(thr_t2r.partition_S(tdKVtdKV), num_warp_groups, wg_idx)
        cute.copy(tiled_t2r, tTR_tdKV, tTR_rdKV)
        return tTR_rdKV

    @cute.jit
    def _reduce_dKV_main_from_reg(
        self,
        dKV_acc: cute.Tensor,
        tTR_rdKV: cute.Tensor,
        rTopkIdx: cute.Tensor,
        sub_tile_idx: int,
    ):
        """Atomically accumulate one M128xN64 register fragment into global dKV."""
        tidx, _, _ = cute.arch.thread_idx()
        _, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        for i in cutlass.range_constexpr(self.reduce_rows_per_thread):
            coord_base = i * 2 - i % 2
            rdKV_frg = cute.make_rmem_tensor((4,), self.acc_dtype)
            rdKV_frg[0] = tTR_rdKV[coord_base]
            rdKV_frg[1] = tTR_rdKV[coord_base + 2]
            rdKV_frg[2] = tTR_rdKV[coord_base + 16]
            rdKV_frg[3] = tTR_rdKV[coord_base + 18]
            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                dKV_row = dKV_acc[None, topk_idx, (0, batch_idx)]
                tile_dKV_row = cute.flat_divide(dKV_row, (128,))[None, sub_tile_idx]
                cur_dKV_frg = cute.flat_divide(tile_dKV_row, (4,))[None, dp_idx // 4]
                cute.arch.atomic_add(cur_dKV_frg.iterator.llvm_ptr, rdKV_frg.load(), sem="relaxed", scope="gpu")

    @cute.jit
    def reduce_dKV(
        self,
        tdKVtdKV: tuple,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        max_seqlen_kv: Int32,
        tile_count: Int32,
        topk: Int32,
        mma_reduce_dKV_pipeline,
    ):
        """Consume three reducer generations for each M64 tile."""
        tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4 = tdKVtdKV
        tdKVtdKV0 = tdKVtdKV0[(None, None), 0, 0]
        tdKVtdKV1 = tdKVtdKV1[(None, None), 0, 0]
        tdKVtdKV2 = tdKVtdKV2[(None, None), 0, 0]
        tdKVtdKV3 = tdKVtdKV3[(None, None), 0, 0]
        tdKVtdKV4 = tdKVtdKV4[(None, None), 0, 0]

        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        cdKV = cute.make_identity_tensor((self.dOP_mma_tiler[0], self.dOP_mma_tiler[1]))
        thr_main = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV0).get_slice(dp_idx)
        tTR_cdKV = self.split_wg(thr_main.partition_D(cdKV), num_warp_groups, wg_idx)
        cdKV_64 = cute.make_identity_tensor((self.dKV4_mma_tiler[0], self.dKV4_mma_tiler[1]))
        thr_tail = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV4).get_slice(dp_idx)
        tTR_cdKV_64 = self.split_wg(thr_tail.partition_D(cdKV_64), num_warp_groups, wg_idx)

        consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_reduce_dKV_stage)
        rTopkIdx = cute.make_rmem_tensor((self.reduce_rows_per_thread,), cutlass.Int32)
        rTopkIdx_64 = cute.make_rmem_tensor((self.reduce_rows_per_thread,), cutlass.Int32)
        full_tiles = (topk % self.block_tile) == 0

        tile_index = tile_count - 1
        while tile_index >= 0:
            for half_iter in cutlass.range_constexpr(self.num_kv_subtiles):
                kv_half = self.num_kv_subtiles - 1 - half_iter
                row_base = tile_index * self.block_tile + kv_half * self.kv_subtile
                for i in cutlass.range_constexpr(self.reduce_rows_per_thread):
                    coord_base = i * 2 - i % 2
                    local_row_idx = cute.get(tTR_cdKV[coord_base], mode=[1])
                    global_row_idx = row_base + local_row_idx
                    if full_tiles:
                        rTopkIdx[i] = mTopkIdxs[global_row_idx, (token_idx, batch_idx)]
                    else:
                        if global_row_idx < topk:
                            rTopkIdx[i] = mTopkIdxs[global_row_idx, (token_idx, batch_idx)]
                        else:
                            rTopkIdx[i] = Int32(-1)
                    local_row_idx_64 = cute.get(tTR_cdKV_64[coord_base], mode=[1])
                    global_row_idx_64 = row_base + local_row_idx_64
                    if full_tiles:
                        rTopkIdx_64[i] = mTopkIdxs[global_row_idx_64, (token_idx, batch_idx)]
                    else:
                        if global_row_idx_64 < topk:
                            rTopkIdx_64[i] = mTopkIdxs[global_row_idx_64, (token_idx, batch_idx)]
                        else:
                            rTopkIdx_64[i] = Int32(-1)

                mma_reduce_dKV_pipeline.consumer_wait(consumer_state)
                rdKV0 = self._t2r_dKV_main(tdKVtdKV0)
                cute.arch.fence_view_async_tmem_load()
                self._reduce_dKV_main_from_reg(mdKV_acc, rdKV0, rTopkIdx, 0)
                rdKV1 = self._t2r_dKV_main(tdKVtdKV1)
                cute.arch.fence_view_async_tmem_load()
                mma_reduce_dKV_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
                self._reduce_dKV_main_from_reg(mdKV_acc, rdKV1, rTopkIdx, 1)
                mma_reduce_dKV_pipeline.consumer_wait(consumer_state)
                rdKV4 = self.t2r_dKV_64(tdKVtdKV4)
                cute.arch.fence_view_async_tmem_load()
                mma_reduce_dKV_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
                self.reduce_dKV_64_from_reg(mdKV_acc, rdKV4, rTopkIdx_64)
                mma_reduce_dKV_pipeline.consumer_wait(consumer_state)
                rdKV2 = self._t2r_dKV_main(tdKVtdKV2)
                cute.arch.fence_view_async_tmem_load()
                self._reduce_dKV_main_from_reg(mdKV_acc, rdKV2, rTopkIdx, 2)
                rdKV3 = self._t2r_dKV_main(tdKVtdKV3)
                cute.arch.fence_view_async_tmem_load()
                # T2R has detached dKV2/3 from TMEM.  Publish the lifetime
                # boundary without waiting for the MMA warp, so rdKV3 global
                # atomics overlap the next tile's dKV0/1 production.
                self.t2r_dKV23_done_barrier.arrive()
                mma_reduce_dKV_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
                self._reduce_dKV_main_from_reg(mdKV_acc, rdKV3, rTopkIdx, 3)
            tile_index -= 1
