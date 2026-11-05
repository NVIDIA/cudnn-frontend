# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 CuTeDSL dense attention-score recompute kernel for MXFP8 Q/K.

Computes ``out[q, k] = sum_h exp(Q_h . K_k * scale - LSE[q, h])`` over the ratio-causal key range
(``-inf`` elsewhere in the row) and ``denom[q] = sum_k out[q, k]`` over the valid keys.

One CTA owns one packed m tile (128 (token, head) rows: ``128 // H`` query tokens, H in {64, 128})
and walks its whole ratio-causal K range under CLC persistent scheduling.  SwapAB: K is the MMA's A
operand (128 keys), packed Q the B operand, so a 128-key block's scores land in TMEM as 128 lanes
(keys) x 128 columns (token, head).  The head dim is consumed in 128-wide chunks (four K=32 MMAs,
one E8M0 scale field each).

* warp 0 loads LSE (pre-scaled to ``-lse * log2(e)``) through a 2-stage ``PipelineAsync``, Q / Q
  scales (``Q_STAGES``-deep ring) and, per key block and chunk, the K chunk plus its K-scale tile
  into a ``num_kv_stages``-deep chunk ring (one TMA barrier per chunk).
* warp 3 copies each chunk's K scales SMEM -> TMEM into a ``NUM_SF_SLOTS``-deep SFA ring.
* warp 1 copies the tile's Q scales into one SFB field per chunk, then issues every block's MMAs
  into TMEM accumulator ``b % NUM_ACC`` (block counter ``b`` runs across tiles).
* math warpgroup ``w`` owns accumulator ``w``: every NUM_ACC-th block, all 128 columns, one key row
  per thread.  It streams TMEM in 32-column slices (the next slice loading while the current one is
  reduced), so no cross-warpgroup head reduce and no atomics are needed; masked keys are written
  as ``-inf`` in-kernel and each tile's ``denom`` is reduced across the warpgroups through SMEM.
"""

from __future__ import annotations

import math
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineTmaUmma,
    PipelineUserType,
    make_pipeline_state,
    pipeline_init_arrive,
    pipeline_init_wait,
    PipelineClcFetchAsync,
)
import cutlass.utils as utils
from cutlass.utils import blockscaled_layout as _blockscaled_layout
from cudnn._cutlass_compat import SmemAllocator

import cuda.bindings.driver as cuda

from cudnn.deepseek_sparse_attention.utils import copy as copy_utils
from cudnn.deepseek_sparse_attention.utils.sm100.gemm import (
    gemm_ptx_partial_mxfp8 as _gemm_ptx_partial_mxfp8,
)
from cudnn.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK
from .score_kernel_sm100 import (
    BlockScaledScoreSm100Mixin,
    ScoreKernelSm100,
    add_packed_f32x2,
    fma_packed_f32x2,
    tcgen05_fence_after as _tcgen05_fence_after,
    tcgen05_fence_before as _tcgen05_fence_before,
)

_SMEM_BYTES = 232448  # SM100 dynamic SMEM per CTA
_SMEM_RESERVE = 8192  # barriers, LSE ring, reduce scratch, Q scales, alignment


class BwdDenseAttnScoreSm100Mxfp8(BlockScaledScoreSm100Mixin, ScoreKernelSm100):
    """SM100 dense attention-score recompute for MXFP8 Q/K (head_dim 128 or 512, 64 or 128 heads)."""

    format_name = "MXFP8"
    CHUNK_K = 128  # head-dim chunk: four K=32 MMAs reading one E8M0 scale field
    NUM_ACC = 3  # TMEM accumulators == math warpgroups
    NUM_SF_SLOTS = 4  # K-scale (SFA) TMEM ring, in chunk fields
    NUM_LSE_STAGES = 2
    Q_STAGES = 2  # Q / Q-scale ring: double-buffered so the next tile's Q lands under this tile's MMAs
    MIN_KV_CHUNKS = 4  # Q_STAGES falls back to 1 below this many K chunks
    MAX_KV_CHUNKS = 12

    def __init__(
        self,
        head_dim: int,
        qhead_per_kvhead: int = 128,
        m_block_size: int = 128,
        n_block_size: int = 128,
        ratio: int = 1,
        is_varlen: bool = False,
        sf_vec_size: int = 32,
    ):
        if head_dim not in (128, 512):
            raise ValueError(f"SM100 dense attention score {self.format_name} requires head_dim in (128, 512), got {head_dim}")
        if qhead_per_kvhead not in (64, 128):
            raise ValueError(f"SM100 dense attention score {self.format_name} requires qhead_per_kvhead in (64, 128), got {qhead_per_kvhead}")
        if sf_vec_size != 32:
            raise ValueError(f"SM100 dense attention score {self.format_name} requires sf_vec_size=32")
        if m_block_size != 128 or n_block_size != 128:
            raise ValueError(f"SM100 dense attention score {self.format_name} requires m_block_size=n_block_size=128")
        nkc = head_dim // self.CHUNK_K
        q_chunk_bytes = m_block_size * self.CHUNK_K  # E4M3
        k_chunk_bytes = n_block_size * self.CHUNK_K
        budget = _SMEM_BYTES - _SMEM_RESERVE
        q_stage = self.Q_STAGES
        if q_stage == 2 and (budget - 2 * nkc * q_chunk_bytes) // (k_chunk_bytes + 512) < self.MIN_KV_CHUNKS:
            q_stage = 1
        num_kv_stages = min(self.MAX_KV_CHUNKS, (budget - q_stage * nkc * q_chunk_bytes) // (k_chunk_bytes + 512))
        assert num_kv_stages >= 2, "SMEM plan holds fewer than two K chunks"
        super().__init__(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            kv_stage=num_kv_stages,
            k_block_size=self.CHUNK_K,
            ratio=ratio,
            is_varlen=is_varlen,
        )
        self.sf_vec_size = sf_vec_size
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.q_stage = q_stage
        self.num_kv_stages = num_kv_stages

        # Warps: 0 load, 1 MMA, 2 CLC scheduler, 3 K-scale S2T, then NUM_ACC math warpgroups.
        self.sf_warp_id = 3
        self._set_register_split(self.NUM_ACC)

        # TMEM: NUM_ACC accumulators, then the SFA ring, then one SFB field per Q chunk.
        self.tmem_s_stride = self.m_block_size
        self.tmem_sf_field_cols = (self.n_block_size // 32) * 4  # one 128-wide K chunk of scales
        self.tmem_k_sfa_base_offset = self.NUM_ACC * self.m_block_size
        self.tmem_q_sfb_offset = self.tmem_k_sfa_base_offset + self.NUM_SF_SLOTS * self.tmem_sf_field_cols
        self.tmem_alloc_cols = 512
        assert self.tmem_q_sfb_offset + nkc * self.tmem_sf_field_cols <= self.tmem_alloc_cols

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPerHead: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mOut: cute.Tensor,
        mDenom: cute.Tensor,
        softmax_scale: Float32 | float,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mCuSeqlensQScalePadded: cute.Tensor | None,
        mCuSeqlensKScalePadded: cute.Tensor | None,
        mQCausalOffsets: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        is_varlen = mCuSeqlensQ is not None
        if const_expr(is_varlen):
            assert self.is_varlen
            assert mCuSeqlensQ is not None and mCuSeqlensK is not None
            assert mCuSeqlensQScalePadded is not None and mCuSeqlensKScalePadded is not None
        else:
            assert not self.is_varlen
            assert mCuSeqlensQ is None and mCuSeqlensK is None
            assert mCuSeqlensQScalePadded is None and mCuSeqlensKScalePadded is None

        NKC = self.num_k_chunks
        ops = self._block_scaled_operands(
            mQ,
            mK,
            mQScale,
            mKScale,
            is_varlen,
            scale_tile_k=self.CHUNK_K,
            q_data_stages=NKC * self.q_stage,
            q_scale_stages=NKC * self.q_stage,
            k_data_stages=self.num_kv_stages,
            k_scale_stages=self.num_kv_stages,
        )
        b = self.tma_bytes
        self.tma_copy_bytes = {"Q": NKC * (b["q_data"] + b["q_scale"]), "KV": b["k_data"] + b["k_scale"]}
        seqlen_q_static = max_seqlen_q if const_expr(is_varlen) else ops["seqlen_q"]
        batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1 if const_expr(is_varlen) else cute.size(ops["mQ"].shape[3])
        tile_sched_params, grid_dim = self._tile_scheduler(seqlen_q_static, batch_size)
        self.kernel(
            ops["mQ"],
            ops["mK"],
            self._rows_first(mPerHead, is_varlen, 3),
            ops["mQScale"],
            ops["mKScale"],
            self._rows_first(mOut, is_varlen, 3),
            self._rows_first(mDenom, is_varlen, 2),
            softmax_scale,
            ops["tma_atom_Q"],
            ops["tma_atom_K"],
            ops["tma_atom_QScale"],
            ops["tma_atom_KScale"],
            ops["tiled_mma_qk"],
            ops["blockscaled_tiled_mma_qk"],
            ops["sQ_layout"],
            ops["sK_layout"],
            ops["sQScale_layout"],
            ops["sKScale_layout"],
            tile_sched_params,
            max_seqlen_q,
            max_seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuSeqlensQScalePadded,
            mCuSeqlensKScalePadded,
            mQCausalOffsets,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ,
        mK,
        mPerHead,
        mQScale,
        mKScale,
        mOut,
        mDenom,
        softmax_scale: Float32 | float,
        tma_atom_Q,
        tma_atom_K,
        tma_atom_QScale,
        tma_atom_KScale,
        tiled_mma_qk,
        blockscaled_tiled_mma_qk,
        sQ_layout,
        sK_layout,
        sQScale_layout,
        sKScale_layout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mCuSeqlensQScalePadded: cute.Tensor | None,
        mCuSeqlensKScalePadded: cute.Tensor | None,
        mQCausalOffsets: cute.Tensor | None,
    ):
        is_varlen = mCuSeqlensQ is not None
        NKC = self.num_k_chunks
        NUM_ACC = self.NUM_ACC
        NSF = self.NUM_SF_SLOTS
        seqlen_q_static = max_seqlen_q if const_expr(is_varlen) else cute.size(mQ.shape[0]) // self.qhead_per_kvhead
        seqlen_k_static = max_seqlen_k if const_expr(is_varlen) else cute.size(mK.shape[0])
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=seqlen_q_static,
            seqlen_k_static=seqlen_k_static,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            tile_m=self.m_block_size,
            tile_n=self.n_block_size,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_QScale)
            cpasync.prefetch_descriptor(tma_atom_KScale)

        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)
        sQScale_size = cute.cosize(sQScale_layout)
        sKScale_size = cute.cosize(sKScale_layout)
        sLSE_size = self.m_block_size * self.NUM_LSE_STAGES
        sRed_size = 2 * self.q_tokens_per_tile * 4 * self.num_math_wg

        @cute.struct
        class SharedStorage:
            lse_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.NUM_LSE_STAGES]
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.q_stage]
            KV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.num_kv_stages]
            SF_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.NUM_SF_SLOTS]
            T_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.NUM_ACC]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, 4]
            sLSE: cute.struct.Align[cute.struct.MemRange[Float32, sLSE_size], 128]
            sRed: cute.struct.Align[cute.struct.MemRange[Float32, sRed_size], 128]
            sQScale: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, sQScale_size], self.buffer_align_bytes]
            sKScale: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, sKScale_size], self.buffer_align_bytes]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.k_dtype, sK_size], self.buffer_align_bytes]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        Q_mbar_ptr = storage.Q_mbar_ptr.data_ptr()
        KV_mbar_ptr = storage.KV_mbar_ptr.data_ptr()
        SF_mbar_ptr = storage.SF_mbar_ptr.data_ptr()  # [full x NSF, empty x NSF]
        T_mbar_ptr = storage.T_mbar_ptr.data_ptr()  # [full x NUM_ACC, empty x NUM_ACC]
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf.ptr
        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQScale = storage.sQScale.get_tensor(sQScale_layout)
        sKScale = storage.sKScale.get_tensor(sKScale_layout)
        sLSE = storage.sLSE.get_tensor(cute.make_layout((sLSE_size,), stride=(1,)))
        sRed = storage.sRed.get_tensor(cute.make_layout((sRed_size,), stride=(1,)))

        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))
        # LSE: all 32 load lanes publish, every math lane releases at the end of the tile.
        pipeline_lse = PipelineAsync.create(
            barrier_storage=storage.lse_mbar_ptr.data_ptr(),
            num_stages=self.NUM_LSE_STAGES,
            producer_group=CooperativeGroup(Agent.Thread, self.WARP_SIZE),
            consumer_group=CooperativeGroup(Agent.Thread, self.WARPGROUP_SIZE * self.num_math_wg),
            defer_sync=True,
        )
        pipeline_Q = PipelineTmaUmma.create(
            barrier_storage=Q_mbar_ptr,
            num_stages=self.q_stage,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_KV = PipelineTmaUmma.create(
            barrier_storage=KV_mbar_ptr,
            num_stages=self.num_kv_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_bytes["KV"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        if warp_idx == 1:
            cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, self.num_math_wg)
        if warp_idx == 0:
            for _s in cutlass.range_constexpr(NSF):
                cute.arch.mbarrier_init(SF_mbar_ptr + _s, 1)  # full: SF warp commit
                cute.arch.mbarrier_init(SF_mbar_ptr + NSF + _s, 1)  # empty: MMA commit
            for _a in cutlass.range_constexpr(NUM_ACC):
                cute.arch.mbarrier_init(T_mbar_ptr + _a, 1)  # full: MMA commit
                cute.arch.mbarrier_init(T_mbar_ptr + NUM_ACC + _a, self.WARPGROUP_SIZE)  # empty: the owning math warpgroup
        num_clc_consumer_threads = self.WARP_SIZE * (4 + 4 * self.num_math_wg)
        clc_pipeline = PipelineClcFetchAsync.create(
            barrier_storage=clc_mbar_ptr,
            num_stages=self.num_clc_stage,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, num_clc_consumer_threads),
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
        cute.arch.sync_threads()
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        clc_consumer_state = make_pipeline_state(PipelineUserType.Consumer, self.num_clc_stage)
        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim(), clc_response_ptr)
        work_tile = tile_sched.initial_work_tile_info()
        lse_producer, lse_consumer = pipeline_lse.make_participants()
        Q_producer, Q_consumer = pipeline_Q.make_participants()

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        tStS_ref = cute.make_tensor(cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16), tStS_fake.layout)
        sf_tmem_ptr = cute.recast_ptr(tStS_ref.iterator, dtype=self.sf_dtype)
        tmem_sf_offset_scale = self.qk_acc_dtype.width // self.sf_dtype.width
        k_scale_smem_layout = cute.slice_(sKScale_layout, (None, None, None, 0))
        q_scale_smem_layout = cute.slice_(sQScale_layout, (None, None, None, 0))
        sf_tiler = (self.n_block_size, self.m_block_size, self.CHUNK_K)
        tCtKSFA_layout = _blockscaled_layout.make_tmem_layout_sfa(blockscaled_tiled_mma_qk, sf_tiler, self.sf_vec_size, k_scale_smem_layout)
        tCtQSFB_layout = _blockscaled_layout.make_tmem_layout_sfb(blockscaled_tiled_mma_qk, sf_tiler, self.sf_vec_size, q_scale_smem_layout)

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.WARPGROUP_SIZE)
        if warp_group_idx == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_load)

            # ---------------- load warp: LSE, Q / Q scales, then K chunks + K-scale tiles ----------------
            if warp_idx == self.load_warp_id:
                lane_id = tidx % self.WARP_SIZE
                scale_batch_idx = Int32(-1)
                q_scale_m_block = Int32(0)
                k_scale_n_block = Int32(0)
                KV_producer_state = make_pipeline_state(PipelineUserType.Producer, self.num_kv_stages)
                while work_tile.is_valid_tile:
                    valid, m_block, seqlen, batch_idx, q_causal_offset = self._tile_info(work_tile, SeqlenInfoCls, mQCausalOffsets)
                    scale_l_idx, scale_batch_idx, q_scale_m_block, k_scale_n_block = self._scale_offsets(
                        batch_idx, scale_batch_idx, q_scale_m_block, k_scale_n_block, mCuSeqlensQScalePadded, mCuSeqlensKScalePadded
                    )
                    if valid:
                        num_n_blocks = self._dense_compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)
                        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, 0]
                        mQScale_cur = mQScale[None, None, scale_l_idx]
                        handle_Q = Q_producer.acquire_and_advance()
                        for _kc in cutlass.range_constexpr(NKC):
                            gQ = cute.local_tile(mQ_cur, (self.m_block_size, self.CHUNK_K), (m_block, _kc))
                            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_Q, 0, cute.make_layout(1), gQ, sQ[None, None, None, handle_Q.index * NKC + _kc], single_stage=True
                            )
                            load_Q_fn(tma_bar_ptr=handle_Q.barrier)
                            gQScale = cute.local_tile(mQScale_cur, (self.m_block_size, self.CHUNK_K), (q_scale_m_block + m_block, _kc))
                            load_QScale_fn, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_QScale,
                                0,
                                cute.make_layout(1),
                                gQScale,
                                sQScale[None, None, None, handle_Q.index * NKC + _kc],
                                filter_zeros=True,
                                single_stage=True,
                            )
                            load_QScale_fn(tma_bar_ptr=handle_Q.barrier)

                        lse = lse_producer.acquire_and_advance()
                        # staged as -lse * log2(e): the math warpgroups evaluate exp2(s * scale * log2(e) - lse * log2(e))
                        mLSE_cur = seqlen.offset_batch_Q(mPerHead, batch_idx, dim=2)
                        self._stage_per_head(sLSE, lse.index * self.m_block_size, mLSE_cur, m_block, seqlen.seqlen_q, lane_id, -math.log2(math.e))
                        lse.commit()

                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, 0]
                        mKScale_cur = mKScale[None, None, scale_l_idx]
                        nb = Int32(0)
                        while nb < num_n_blocks:
                            for _kc in cutlass.range_constexpr(NKC):
                                pipeline_KV.producer_acquire(KV_producer_state)
                                kv_bar = pipeline_KV.producer_get_barrier(KV_producer_state)
                                gKScale = cute.local_tile(mKScale_cur, (self.n_block_size, self.CHUNK_K), (k_scale_n_block + nb, _kc))
                                load_KScale_fn, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_KScale,
                                    0,
                                    cute.make_layout(1),
                                    gKScale,
                                    sKScale[None, None, None, KV_producer_state.index],
                                    filter_zeros=True,
                                    single_stage=True,
                                )
                                load_KScale_fn(tma_bar_ptr=kv_bar)
                                gK = cute.local_tile(mK_cur, (self.n_block_size, self.CHUNK_K), (nb, _kc))
                                load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_K, 0, cute.make_layout(1), gK, sK[None, None, None, KV_producer_state.index], single_stage=True
                                )
                                load_K_fn(tma_bar_ptr=kv_bar)
                                KV_producer_state.advance()
                            nb = nb + 1
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                lse_producer.tail()
                Q_producer.tail()
                pipeline_KV.producer_tail(KV_producer_state)

            # ---------------- SF warp: K scales SMEM -> TMEM SFA ring ----------------
            if warp_idx == self.sf_warp_id:
                cute.arch.barrier(barrier_id=1, number_of_threads=2 * self.WARP_SIZE)  # TMEM allocated
                KV_sf_state = make_pipeline_state(PipelineUserType.Consumer, self.num_kv_stages)
                sf_ctr = Int32(0)
                sf_empty_bits = Int32((1 << NSF) - 1)
                while work_tile.is_valid_tile:
                    valid, m_block, seqlen, batch_idx, q_causal_offset = self._tile_info(work_tile, SeqlenInfoCls, mQCausalOffsets)
                    if valid:
                        num_n_blocks = self._dense_compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)
                        n_chunks = num_n_blocks * NKC
                        c = Int32(0)
                        while c < n_chunks:
                            pipeline_KV.consumer_wait(KV_sf_state)
                            st = sf_ctr
                            cute.arch.mbarrier_wait(SF_mbar_ptr + NSF + st, (sf_empty_bits >> st) & Int32(1))
                            sf_empty_bits = sf_empty_bits ^ (Int32(1) << st)
                            _tcgen05_fence_after()
                            col = Int32(self.tmem_k_sfa_base_offset) + st * Int32(self.tmem_sf_field_cols)
                            tCtKSFA = cute.make_tensor(sf_tmem_ptr + Int32(tmem_sf_offset_scale) * col, tCtKSFA_layout)
                            tiled_copy_s2t_ksfa, tCsKSFA_s2t, tCtKSFA_s2t = self._s2t_copy_and_partition(sKScale, tCtKSFA)
                            cute.copy(tiled_copy_s2t_ksfa, tCsKSFA_s2t[(None, None, None, None, KV_sf_state.index)], tCtKSFA_s2t)
                            with cute.arch.elect_one():
                                tcgen05.commit(SF_mbar_ptr + st)
                            KV_sf_state.advance()
                            sf_ctr = sf_ctr + Int32(1) if sf_ctr < Int32(NSF - 1) else Int32(0)
                            c = c + 1
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

            # ---------------- MMA warp ----------------
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(Int32(self.tmem_alloc_cols), tmem_holding_buf)
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=1, number_of_threads=2 * self.WARP_SIZE)
                KV_mma_state = make_pipeline_state(PipelineUserType.Consumer, self.num_kv_stages)
                sf_ctr = Int32(0)
                sf_full_bits = Int32(0)
                t_empty_bits = Int32((1 << NUM_ACC) - 1)
                acc = Int32(0)  # block counter mod NUM_ACC, carried across tiles
                tSrK = tiled_mma_qk.make_fragment_A(sK)
                tSrQ = tiled_mma_qk.make_fragment_B(sQ)
                while work_tile.is_valid_tile:
                    valid, m_block, seqlen, batch_idx, q_causal_offset = self._tile_info(work_tile, SeqlenInfoCls, mQCausalOffsets)
                    if valid:
                        num_n_blocks = self._dense_compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)
                        handle_Q = Q_consumer.wait_and_advance()
                        for _kc in cutlass.range_constexpr(NKC):
                            col = Int32(self.tmem_q_sfb_offset + _kc * self.tmem_sf_field_cols)
                            tCtQSFB_c = cute.make_tensor(sf_tmem_ptr + Int32(tmem_sf_offset_scale) * col, tCtQSFB_layout)
                            tiled_copy_s2t_qsfb, tCsQSFB_s2t, tCtQSFB_s2t = self._s2t_copy_and_partition(sQScale, tCtQSFB_c)
                            cute.copy(tiled_copy_s2t_qsfb, tCsQSFB_s2t[(None, None, None, None, handle_Q.index * NKC + _kc)], tCtQSFB_s2t)
                        nb = Int32(0)
                        while nb < num_n_blocks:
                            cute.arch.mbarrier_wait(T_mbar_ptr + NUM_ACC + acc, (t_empty_bits >> acc) & Int32(1))
                            t_empty_bits = t_empty_bits ^ (Int32(1) << acc)
                            _tcgen05_fence_after()
                            tStS_a = cute.make_tensor(
                                cute.make_ptr(Float32, acc * self.tmem_s_stride, mem_space=cute.AddressSpace.tmem, assumed_align=16), tStS_ref.layout
                            )
                            for _kc in cutlass.range_constexpr(NKC):
                                pipeline_KV.consumer_wait(KV_mma_state)
                                st = sf_ctr
                                cute.arch.mbarrier_wait(SF_mbar_ptr + st, (sf_full_bits >> st) & Int32(1))
                                sf_full_bits = sf_full_bits ^ (Int32(1) << st)
                                _tcgen05_fence_after()
                                col_a = Int32(self.tmem_k_sfa_base_offset) + st * Int32(self.tmem_sf_field_cols)
                                tCtKSFA = cute.make_tensor(sf_tmem_ptr + Int32(tmem_sf_offset_scale) * col_a, tCtKSFA_layout)
                                col_b = Int32(self.tmem_q_sfb_offset + _kc * self.tmem_sf_field_cols)
                                tCtQSFB = cute.make_tensor(sf_tmem_ptr + Int32(tmem_sf_offset_scale) * col_b, tCtQSFB_layout)
                                k_idx = KV_mma_state.index
                                _gemm_ptx_partial_mxfp8(
                                    blockscaled_tiled_mma_qk.op,
                                    tStS_a.iterator.toint(),
                                    tSrK[None, None, None, k_idx],
                                    tSrQ[None, None, None, handle_Q.index * NKC + _kc],
                                    sK[None, None, None, k_idx],
                                    sQ[None, None, None, handle_Q.index * NKC + _kc],
                                    tCtKSFA,
                                    tCtQSFB,
                                    zero_init=const_expr(_kc == 0),
                                    kphase_offset=0,
                                )
                                with cute.arch.elect_one():
                                    tcgen05.commit(SF_mbar_ptr + NSF + st)
                                pipeline_KV.consumer_release(KV_mma_state)
                                KV_mma_state.advance()
                                sf_ctr = sf_ctr + Int32(1) if sf_ctr < Int32(NSF - 1) else Int32(0)
                            with cute.arch.elect_one():
                                tcgen05.commit(T_mbar_ptr + acc)
                            acc = acc + Int32(1) if acc < Int32(NUM_ACC - 1) else Int32(0)
                            nb = nb + 1
                        handle_Q.release()
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                tmem_ptr_dealloc = cute.arch.retrieve_tmem_ptr(Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf)
                cute.arch.dealloc_tmem(tmem_ptr_dealloc, Int32(self.tmem_alloc_cols))

            # ---------------- CLC scheduler warp ----------------
            if warp_idx == self.sched_warp_id:
                self._sched_warp(work_tile, tile_sched, clc_pipeline, clc_consumer_state)

        # ---------------- math warpgroups ----------------
        if warp_group_idx > 0:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue)
            self._math_warpgroup(
                warp_group_idx - 1,
                tidx,
                tiled_mma_qk,
                tStS_ref,
                sLSE,
                lse_consumer,
                sRed,
                T_mbar_ptr,
                mOut,
                mDenom,
                softmax_scale,
                SeqlenInfoCls,
                mQCausalOffsets,
                work_tile,
                tile_sched,
                clc_pipeline,
                clc_consumer_state,
                tmem_dealloc_mbar_ptr,
            )

    @cute.jit
    def _math_warpgroup(
        self,
        wg,
        tidx,
        tiled_mma_qk,
        tStS_ref,
        sLSE,
        lse_consumer,
        sRed,
        T_mbar_ptr,
        mOut,
        mDenom,
        softmax_scale,
        SeqlenInfoCls,
        mQCausalOffsets,
        work_tile,
        tile_sched,
        clc_pipeline,
        clc_consumer_state,
        tmem_dealloc_mbar_ptr,
    ):
        """Math warpgroup ``wg`` (runtime value, one code copy): every block whose counter is ``wg`` mod NUM_ACC."""
        NUM_ACC = self.NUM_ACC
        T = self.q_tokens_per_tile
        H = self.qhead_per_kvhead
        tidx_wg = tidx % self.WARPGROUP_SIZE
        warp = tidx_wg // self.WARP_SIZE
        ratio = Int32(self.ratio)
        scale_log2_e = Float32(softmax_scale) * Float32(math.log2(math.e))
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_ref).get_slice(tidx_wg)
        thr_mma = tiled_mma_qk.get_slice(tidx_wg)
        tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2])))
        kv_offset = tScS[0][0]  # this thread's key row inside a block
        slice_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape))[None, 0, None, None].shape
        bufs = [cute.make_rmem_tensor(slice_shape, Float32), cute.make_rmem_tensor(slice_shape, Float32)]
        rL = cute.make_rmem_tensor((32,), Float32)
        t_full_phase = Int32(0)
        acc_base = Int32(0)  # counter mod NUM_ACC of this tile's first block
        tile_parity = Int32(0)
        while work_tile.is_valid_tile:
            valid, m_block, seqlen, batch_idx, q_causal_offset = self._tile_info(work_tile, SeqlenInfoCls, mQCausalOffsets)
            if valid:
                num_n_blocks = self._dense_compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)
                mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=2)
                mDenom_cur = seqlen.offset_batch_Q(mDenom, batch_idx, dim=1)
                row_len = cute.size(mOut_cur.shape[1])
                lse = lse_consumer.wait_and_advance()
                lse_off = lse.index * self.m_block_size
                q_toks = [m_block * T + Int32(u) for u in range(T)]
                lims = []
                rows = []
                for u in cutlass.range_constexpr(T):
                    lim = (q_causal_offset + q_toks[u] + Int32(1)) // ratio
                    lim = lim if lim < seqlen.seqlen_k else seqlen.seqlen_k
                    lims.append(lim)
                    q_row = q_toks[u] if q_toks[u] < seqlen.seqlen_q else Int32(0)
                    rows.append(cute.make_tensor(mOut_cur.iterator + cute.crd2idx((q_row, 0), mOut_cur.layout), cute.make_layout((row_len,), stride=(1,))))
                dsum = [Float32(0.0) for _ in range(T)]
                # first block of this tile owned by this warpgroup
                j = wg - acc_base
                j = j + Int32(NUM_ACC) if j < Int32(0) else j
                while j < num_n_blocks:
                    cute.arch.mbarrier_wait(T_mbar_ptr + wg, t_full_phase)
                    t_full_phase = t_full_phase ^ Int32(1)
                    _tcgen05_fence_after()
                    tStS_w = cute.make_tensor(
                        cute.make_ptr(Float32, wg * self.tmem_s_stride, mem_space=cute.AddressSpace.tmem, assumed_align=16), tStS_ref.layout
                    )
                    tStS_t2r = thr_tmem_load.partition_S(tStS_w)
                    cute.copy(thr_tmem_load, tStS_t2r[None, 0, None, None], bufs[0])
                    cute.arch.fence_view_async_tmem_load()
                    accs = [(Float32(0.0), Float32(0.0)) for _ in range(2 * T)]
                    for k in cutlass.range_constexpr(4):
                        if const_expr(k < 3):
                            cute.copy(thr_tmem_load, tStS_t2r[None, k + 1, None, None], bufs[(k + 1) % 2])
                        cute.autovec_copy(cute.make_tensor(sLSE.iterator + lse_off + k * 32, cute.make_layout((32,), stride=(1,))), rL)
                        u = (k * 32) // H
                        for p in cutlass.range_constexpr(16):
                            s0, s1 = fma_packed_f32x2((bufs[k % 2][2 * p], bufs[k % 2][2 * p + 1]), (scale_log2_e, scale_log2_e), (rL[2 * p], rL[2 * p + 1]))
                            e = (cute.math.exp2(s0, fastmath=True), cute.math.exp2(s1, fastmath=True))
                            a = 2 * u + (p % 2)
                            accs[a] = add_packed_f32x2(accs[a], e)
                        if const_expr(k < 3):
                            cute.arch.fence_view_async_tmem_load()
                        if const_expr(k == 2):
                            # every slice of this block is in registers: hand the accumulator back
                            _tcgen05_fence_before()
                            cute.arch.mbarrier_arrive(T_mbar_ptr + NUM_ACC + wg)
                    pos = kv_offset + j * self.n_block_size
                    for u in cutlass.range_constexpr(T):
                        s2 = add_packed_f32x2(accs[2 * u], accs[2 * u + 1])
                        score = s2[0] + s2[1]
                        row_u = rows[u]
                        valid = pos < lims[u]
                        dsum[u] = dsum[u] + (score if valid else Float32(0.0))
                        out_val = score if valid else -Float32.inf
                        if q_toks[u] < seqlen.seqlen_q and pos < row_len:
                            row_u[pos] = out_val
                    j = j + Int32(NUM_ACC)
                # -inf past the computed blocks, split across the warpgroups the same way
                n_row_blocks = cute.ceil_div(row_len, self.n_block_size)
                jt = num_n_blocks + wg
                while jt < n_row_blocks:
                    pos = kv_offset + jt * self.n_block_size
                    for u in cutlass.range_constexpr(T):
                        row_u = rows[u]
                        if q_toks[u] < seqlen.seqlen_q and pos < row_len:
                            row_u[pos] = -Float32.inf
                    jt = jt + Int32(NUM_ACC)
                lse.release()
                # denom: per token, reduce the warpgroups' partial sums through SMEM (double-buffered by tile parity)
                red_off = tile_parity * (T * 4 * self.num_math_wg)
                for u in cutlass.range_constexpr(T):
                    wsum = cute.arch.warp_reduction_sum(dsum[u])
                    if tidx_wg % self.WARP_SIZE == 0:
                        sRed[red_off + u * 4 * self.num_math_wg + wg * 4 + warp] = wsum
                cute.arch.barrier(barrier_id=2, number_of_threads=self.WARPGROUP_SIZE * self.num_math_wg)
                if wg == 0:
                    if tidx_wg < T:
                        total = Float32(0.0)
                        for i in cutlass.range_constexpr(4 * self.num_math_wg):
                            total = total + sRed[red_off + tidx_wg * 4 * self.num_math_wg + i]
                        q_tok = m_block * T + tidx_wg
                        if q_tok < seqlen.seqlen_q:
                            mDenom_cur[q_tok] = total
                tile_parity = tile_parity ^ Int32(1)
                acc_base = (acc_base + num_n_blocks) % Int32(NUM_ACC)
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
        if tidx_wg == 0:
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
