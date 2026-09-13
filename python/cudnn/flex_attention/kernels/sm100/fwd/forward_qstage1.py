# SPDX-License-Identifier: BSD-3-Clause
# SM100/SM103 generic forward pipeline implementation.

import math
from functools import partial
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import Boolean, Float32, Int32, const_expr

from cudnn.flex_attention.kernels.sm100 import blackwell_helpers as sm100_utils
from cudnn.flex_attention.kernels.sm100 import mma_desc as sm100_desc
from cudnn.flex_attention.kernels.sm100.fwd.forward_config import (
    SM100_FWD_MASK_PAYLOAD_WORDS,
)
from cudnn.flex_attention.kernels.sm100.fwd.named_barrier import NamedBarrierFwdSm100
from cudnn.flex_attention.plan.kernels import BlockSparseTensors
from cudnn.flex_attention.plan.kernels.packed_mask import (
    get_total_arbitrary_block_count_fwd_sm100,
    produce_arbitrary_forward_loads_qstage1_n_direction_sm100,
)

from cudnn.flex_attention._compat import copy_utils, layout_utils

from .forward import _FlexAttentionForwardSm100Base

# Static tuning for the generic qstage1 2CTA candidates.
# Keys are (head_dim_padded, is_sm103).
_QSTAGE1_2CTA_TUNING_CONFIG = {
    (128, False): {
        "ex2_emu_freq": 10,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 176,
        "num_regs_correction": 88,
    },
    (192, False): {
        "ex2_emu_freq": 16,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 184,
        "num_regs_correction": 80,
    },
    (128, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 80,
    },
    (192, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 64,
    },
}


class FlexAttentionForwardQStage1Sm100(_FlexAttentionForwardSm100Base):
    """Generic one-Q-stage SM100/SM103 forward kernel for 1CTA or 2CTA."""

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        pack_gqa: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        overlap_pv_with_k_wait: bool = False,
    ):
        if type(overlap_pv_with_k_wait) is not bool:
            raise TypeError("overlap_pv_with_k_wait must be a bool")
        if use_2cta_instrs and overlap_pv_with_k_wait:
            raise ValueError("overlap_pv_with_k_wait is only supported by qstage1 1CTA")
        self.overlap_pv_with_k_wait = overlap_pv_with_k_wait
        super().__init__(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
            q_stage=1,
            is_varlen_q=is_varlen_q,
            use_2cta_instrs=use_2cta_instrs,
        )
        # All generic 2CTA kernels share the same packed-mask pipeline.
        self.use_smem_mask_pipeline = self.use_2cta_instrs
        if self.use_2cta_instrs:
            self._tune = _QSTAGE1_2CTA_TUNING_CONFIG.get((self.head_dim_padded, self.is_sm103), {})
            if "ex2_emu_freq" in self._tune:
                self.enable_ex2_emu = self._tune["ex2_emu_freq"] > 0
            if "num_regs_softmax" in self._tune:
                self.num_regs_softmax = self._tune["num_regs_softmax"]
                self.num_regs_correction = self._tune["num_regs_correction"]
                self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction
        if self.head_dim_padded == 128 and self.head_dim_v_padded == 128:
            if self.use_2cta_instrs and self.is_sm103:
                self.num_regs_softmax = 192
                self.num_regs_correction = 80
            elif not self.use_2cta_instrs and self.is_sm103:
                self.num_regs_softmax = 200
                self.num_regs_correction = 64
            else:
                self.num_regs_softmax = 184
                self.num_regs_correction = 88
            self.num_regs_other = 48

    @cute.jit
    def _produce_loads(
        self,
        blocksparse_tensors,
        batch_idx,
        head_idx,
        m_block,
        seqlen,
        kv_producer_state,
        load_Q,
        load_K,
        load_V,
        q_producer_phase,
        mma_thread_idx,
        pipeline_mask_s0=None,
        pipeline_mask_s1=None,
        mask_s0_producer_state=None,
        mask_s1_producer_state=None,
        sMask=None,
    ):
        return produce_arbitrary_forward_loads_qstage1_n_direction_sm100(
            blocksparse_tensors,
            batch_idx,
            head_idx,
            m_block,
            seqlen,
            kv_producer_state,
            load_Q,
            load_K,
            load_V,
            q_producer_phase,
            self.cta_tiler[0] * self.cta_group_size,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            payload_subtile_idx=mma_thread_idx,
            payload_groups=self.m_block_size,
            payload_words=SM100_FWD_MASK_PAYLOAD_WORDS,
            pipeline_mask_s0=pipeline_mask_s0,
            pipeline_mask_s1=pipeline_mask_s1,
            mask_s0_producer_state=mask_s0_producer_state,
            mask_s1_producer_state=mask_s1_producer_state,
            sMask=sMask,
        )

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: Optional[pipeline.PipelineAsync],
        pipeline_load_epi: Optional[pipeline.PipelineAsync],
        gmem_tiled_copy_O: cute.TiledCopy,
        softmax_scale_log2: Float32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: BlockSparseTensors,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mma_tile_coord_v = thr_mma_qk.thr_idx

        for stage in cutlass.range_constexpr(self.score_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        o_corr_phase_s0 = Int32(0)
        o_corr_phase_s1 = Int32(0)
        corr_epi_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.output_stage)
        load_epi_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = None
            if const_expr(self.use_tma_O or not self.pack_gqa):
                tiler_gO = (
                    self.mma_tiler_pv[0] * self.q_stage,
                    self.head_dim_v_padded,
                )
                gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
                gO = layout_utils.select(
                    cute.flat_divide(gO, (self.mma_tiler_pv[0],)),
                    mode=[0, 2, 1],
                )
                gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]

            total_block_count = get_total_arbitrary_block_count_fwd_sm100(
                blocksparse_tensors,
                batch_idx,
                head_idx,
                m_block,
                seqlen,
                self.cta_tiler[0] * self.cta_group_size,
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            )
            stage_count_s0 = (total_block_count + Int32(1)) // Int32(2)
            stage_count_s1 = total_block_count // Int32(2)
            stats = [
                (Float32(0.0), -Float32.inf, Boolean(True)),
                (Float32(0.0), -Float32.inf, Boolean(True)),
            ]

            for stage in cutlass.range_constexpr(self.score_stage):
                stage_count = stage_count_s0 if const_expr(stage == 0) else stage_count_s1
                sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(stage)
                if stage_count == Int32(0):
                    stats[stage] = (
                        Float32(0.0),
                        -Float32.inf,
                        Boolean(True),
                    )

            max_stream_count = stage_count_s0
            for stream_ordinal in cutlass.range(Int32(1), max_stream_count, unroll=1):
                if stream_ordinal < stage_count_s0:
                    sm_stats_barrier.arrive_and_wait_w_index(index=warp_idx)
                    scale = sScale[tidx]
                    if cute.arch.vote_ballot_sync(scale < Float32(1.0)) != 0:
                        self.correction_rescale(
                            thr_mma_pv,
                            tOtO[None, None, None, 0],
                            tidx,
                            scale,
                        )
                    pipeline_s_p_o.consumer_release_w_index(0)
                    pipeline_sm_stats.consumer_release_w_index(0)
                if stream_ordinal < stage_count_s1:
                    sm_stats_barrier.arrive_and_wait_w_index(index=4 + warp_idx)
                    scale = sScale[tidx + self.m_block_size]
                    if cute.arch.vote_ballot_sync(scale < Float32(1.0)) != 0:
                        self.correction_rescale(
                            thr_mma_pv,
                            tOtO[None, None, None, 1],
                            tidx,
                            scale,
                        )
                    pipeline_s_p_o.consumer_release_w_index(1)
                    pipeline_sm_stats.consumer_release_w_index(1)

            for stage in cutlass.range_constexpr(self.score_stage):
                stage_count = stage_count_s0 if const_expr(stage == 0) else stage_count_s1
                if stage_count > Int32(0):
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    row_max = sScale[tidx + stage * self.m_block_size + self.score_stage * self.m_block_size]
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    invalid = row_sum <= Float32(0.0) or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, invalid)

            row_sum0, row_max0, invalid0 = stats[0]
            row_sum1, row_max1, invalid1 = stats[1]
            row_max0 = row_max0 if not invalid0 else -Float32.inf
            row_max1 = row_max1 if not invalid1 else -Float32.inf
            row_max_combined = cutlass.max(row_max0, row_max1)
            row_max_safe = row_max_combined if row_max_combined != -Float32.inf else Float32(0.0)
            scale0 = (
                cute.math.exp2(
                    (row_max0 - row_max_safe) * softmax_scale_log2,
                    fastmath=True,
                )
                if not invalid0
                else Float32(0.0)
            )
            scale1 = (
                cute.math.exp2(
                    (row_max1 - row_max_safe) * softmax_scale_log2,
                    fastmath=True,
                )
                if not invalid1
                else Float32(0.0)
            )
            row_sum_combined = row_sum0 * scale0 + row_sum1 * scale1
            combined_invalid = row_sum_combined <= Float32(0.0) or row_sum_combined != row_sum_combined
            inv_row_sum = cute.arch.rcp_approx(row_sum_combined if not combined_invalid else Float32(1.0))
            final_scale0 = scale0 * inv_row_sum
            final_scale1 = scale1 * inv_row_sum

            if stage_count_s0 > Int32(0):
                pipeline_o_acc.consumer_wait_w_index_phase(0, o_corr_phase_s0)
            if stage_count_s1 > Int32(0):
                pipeline_o_acc.consumer_wait_w_index_phase(1, o_corr_phase_s1)
            output_stage = Int32(0)
            if const_expr(not self.use_correction_warps_for_epi):
                pipeline_o_epi.producer_acquire(corr_epi_producer_state)
                output_stage = corr_epi_producer_state.index
            self.correction_epilogue_combine_qstage1_n_direction(
                thr_mma_pv,
                tOtO[None, None, None, 0],
                tOtO[None, None, None, 1],
                tidx,
                final_scale0,
                final_scale1,
                sO[None, None, output_stage],
            )
            if stage_count_s0 > Int32(0):
                pipeline_s_p_o.consumer_release_w_index(0)
                o_corr_phase_s0 ^= 1
            if stage_count_s1 > Int32(0):
                pipeline_s_p_o.consumer_release_w_index(1)
                o_corr_phase_s1 ^= 1

            if const_expr(not self.use_correction_warps_for_epi):
                pipeline_o_epi.producer_commit(corr_epi_producer_state)
                corr_epi_producer_state.advance()
            else:
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                    number_of_threads=(len(self.correction_warp_ids) * cute.arch.WARP_SIZE),
                )
                m_tile_idx = m_block * self.cta_group_size + mma_tile_coord_v
                gO_stage = gO[None, None, 0] if const_expr(gO is not None) else None
                self._store_O_to_gmem(
                    sO[None, None, output_stage],
                    gO_stage,
                    mO_cur,
                    gmem_tiled_copy_O,
                    tidx,
                    seqlen.seqlen_q,
                    m_tile_idx,
                )

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                m_tile_idx = m_block * self.cta_group_size + mma_tile_coord_v
                lse = (
                    (row_max_safe * softmax_scale_log2 + cute.math.log2(row_sum_combined, fastmath=True)) * math.log(2.0)
                    if not combined_invalid
                    else -Float32.inf
                )
                seqlen_q = seqlen.seqlen_q if const_expr(not self.pack_gqa) else seqlen.seqlen_q * self.qhead_per_kvhead
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                    gLSE[tidx] = lse

            if const_expr(pipeline_load_epi is not None and self.use_correction_warps_for_epi):
                pipeline_load_epi.producer_acquire(load_epi_producer_state)
                with cute.arch.elect_one():
                    pipeline_load_epi.producer_commit(load_epi_producer_state)
                load_epi_producer_state.advance()

            work_tile = tile_scheduler.advance_to_next_work()

        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi.producer_tail(corr_epi_producer_state)

    @cute.jit
    def correction_epilogue_combine_qstage1_n_direction(
        self,
        thr_mma: cute.ThrMma,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        tidx: Int32,
        scale0: Float32,
        scale1: Float32,
        sO: cute.Tensor,
    ):
        """Merge two FP32 online-softmax streams and stage one output tile."""

        corr_tile_size = 8 * 32 // self.o_dtype.width
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
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO0_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            tiled_tmem_load,
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
        tOtO0_t2r = thr_tmem_load.partition_S(tOtO0_i[(None, None), None])
        tOtO1_t2r = thr_tmem_load.partition_S(tOtO1_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        for i in cutlass.range(self.head_dim_v_padded // corr_tile_size, unroll_full=True):
            frg_shape = tOcO_t2r[None, 0, 0, i].shape
            tOrO = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            tOrO0 = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            tOrO1 = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            # TMEM loads are warp-collective and must not be guarded by a
            # per-row scale predicate. Invalid streams are zeroed only after
            # both collective loads have completed.
            cute.copy(
                tiled_tmem_load,
                tOtO0_t2r[None, 0, 0, i],
                tOrO0,
            )
            cute.copy(
                tiled_tmem_load,
                tOtO1_t2r[None, 0, 0, i],
                tOrO1,
            )
            if scale0 != Float32(0.0):
                for j in cutlass.range(0, cute.size(tOrO0), 2, unroll_full=True):
                    tOrO[j], tOrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tOrO0[j], tOrO0[j + 1]),
                        (scale0, scale0),
                    )
            else:
                tOrO.fill(Float32(0.0))
            if scale1 != Float32(0.0):
                for j in cutlass.range(0, cute.size(tOrO1), 2, unroll_full=True):
                    o1_a, o1_b = cute.arch.mul_packed_f32x2(
                        (tOrO1[j], tOrO1[j + 1]),
                        (scale1, scale1),
                    )
                    tOrO[j], tOrO[j + 1] = cute.arch.add_packed_f32x2(
                        (tOrO[j], tOrO[j + 1]),
                        (o1_a, o1_b),
                    )
            copy_utils.cvt_copy(
                tiled_smem_store,
                tOrO,
                tOsO_s2r[None, 0, 0, i],
            )
        cute.arch.fence_view_async_shared()

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        is_leader_cta: Boolean,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: BlockSparseTensors,
        tile_scheduler=None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        q_smem_start = sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, 0].iterator)
        sm100_utils.declare_ptx_smem_desc(
            q_smem_start,
            q_smem_base,
            tSrQ[None, None, None, 0].layout,
            var_name_prefix="flex_fwd_qstage1_q_smem_desc",
        )
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="flex_fwd_qstage1_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="flex_fwd_qstage1_pv_mma_idesc")

        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="flex_fwd_qstage1_q_smem_desc",
                idesc_var_name="flex_fwd_qstage1_qk_mma_idesc",
                smem_offset=0,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.score_stage)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrP[None, None, None, stage],
                sA=None,
                split_arrive=self.split_P_arrive,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.score_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        phase_s0 = Int32(0)
        phase_s1 = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            block_iter_count = get_total_arbitrary_block_count_fwd_sm100(
                blocksparse_tensors,
                batch_idx,
                head_idx,
                m_block,
                seqlen,
                self.cta_tiler[0] * self.cta_group_size,
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            )
            if block_iter_count > Int32(0) and is_leader_cta:
                pipeline_q.consumer_wait_w_index_phase(0, mma_q_consumer_phase)
                mma_q_consumer_phase ^= 1

                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                k_index, k_phase = (
                    mma_kv_consumer_state.index,
                    mma_kv_consumer_state.phase,
                )
                sK_cur = sK[None, None, None, k_index]
                if const_expr(self.uneven_kv_smem):
                    sK_cur = self.offset_kv_smem(sK_cur, k_index, k_phase)
                gemm_Si[0](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
                pipeline_s_p_o.producer_commit_w_index(0)
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                if block_iter_count > Int32(1):
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    k_index, k_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    sK_cur = sK[None, None, None, k_index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, k_index, k_phase)
                    gemm_Si[1](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
                    pipeline_s_p_o.producer_commit_w_index(1)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()

                o_acc_s0 = Boolean(False)
                o_acc_s1 = Boolean(False)
                middle_count = cutlass.max(block_iter_count - Int32(2), Int32(0))
                pair_count = middle_count // Int32(2)
                for _ in cutlass.range(pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.score_stage):
                        phase_cur = phase_s0 if const_expr(stage == 0) else phase_s1
                        o_acc_cur = o_acc_s0 if const_expr(stage == 0) else o_acc_s1
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        v_release_state = mma_kv_consumer_state.clone()
                        v_index, v_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        tOrVi = tOrV[None, None, None, v_index]
                        sV_cur = sV[None, None, None, v_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, v_index, v_phase)
                        mma_kv_consumer_state.advance()

                        k_index, k_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        sK_cur = sK[None, None, None, k_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, k_index, k_phase)
                        if const_expr(not self.overlap_pv_with_k_wait):
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)

                        pipeline_s_p_o.producer_acquire_w_index_phase(stage, phase_cur)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not o_acc_cur,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage),
                            mbar_phase=phase_cur,
                        )
                        if const_expr(self.overlap_pv_with_k_wait):
                            pipeline_kv.consumer_release(v_release_state)
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        gemm_Si[stage](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        if const_expr(stage == 0):
                            phase_s0 ^= 1
                            o_acc_s0 = Boolean(True)
                        else:
                            phase_s1 ^= 1
                            o_acc_s1 = Boolean(True)
                        if const_expr(not self.overlap_pv_with_k_wait):
                            pipeline_kv.consumer_release(v_release_state)
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()

                if middle_count % Int32(2) != Int32(0):
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    v_release_state = mma_kv_consumer_state.clone()
                    v_index, v_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tOrVi = tOrV[None, None, None, v_index]
                    sV_cur = sV[None, None, None, v_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, v_index, v_phase)
                    mma_kv_consumer_state.advance()

                    k_index, k_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    sK_cur = sK[None, None, None, k_index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, k_index, k_phase)
                    if const_expr(not self.overlap_pv_with_k_wait):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)

                    pipeline_s_p_o.producer_acquire_w_index_phase(0, phase_s0)
                    gemm_Pi[0](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=not o_acc_s0,
                        mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(0),
                        mbar_phase=phase_s0,
                    )
                    if const_expr(self.overlap_pv_with_k_wait):
                        pipeline_kv.consumer_release(v_release_state)
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    gemm_Si[0](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
                    pipeline_s_p_o.producer_commit_w_index(0)
                    phase_s0 ^= 1
                    o_acc_s0 = Boolean(True)
                    if const_expr(not self.overlap_pv_with_k_wait):
                        pipeline_kv.consumer_release(v_release_state)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()

                pipeline_q.consumer_release_w_index(0)

                epilogue_count = cutlass.min(block_iter_count, Int32(2))
                epilogue_begin = block_iter_count - epilogue_count
                for epilogue_ordinal in cutlass.range(epilogue_begin, block_iter_count, unroll=1):
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    v_index, v_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tOrVi = tOrV[None, None, None, v_index]
                    sV_cur = sV[None, None, None, v_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, v_index, v_phase)

                    if epilogue_ordinal % Int32(2) == Int32(0):
                        pipeline_s_p_o.producer_acquire_w_index_phase(0, phase_s0)
                        gemm_Pi[0](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not o_acc_s0,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(0),
                            mbar_phase=phase_s0,
                        )
                        pipeline_o_acc.producer_commit_w_index(0)
                        phase_s0 ^= 1
                        o_acc_s0 = Boolean(True)
                    else:
                        pipeline_s_p_o.producer_acquire_w_index_phase(1, phase_s1)
                        gemm_Pi[1](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not o_acc_s1,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(1),
                            mbar_phase=phase_s1,
                        )
                        pipeline_o_acc.producer_commit_w_index(1)
                        phase_s1 ^= 1
                        o_acc_s1 = Boolean(True)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()

            work_tile = tile_scheduler.advance_to_next_work()
