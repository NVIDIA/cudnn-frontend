# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dense BF16 attention-score recompute — SM100 Cute-DSL implementation.

Dense variant: iterates over the full KV sequence via TMA (no topk_indices).

    P[b,q,h,t] = exp(Q_h · K_t^T · scale - LSE[b,q,h])
    S[b,q,t] = sum_h P[b,q,h,t]
    out[b,q,t] = S[b,q,t]   (written per n_block)
    denom[b,q] = sum_t S[b,q,t]

The indexer scores (BF16 / MXFP8 / MXFP4) and the MXFP8 attention score run the newer kernels built on
``score_kernel_sm100.ScoreKernelSm100``.

K is loaded via TMA (dense sequential, not sparse gather).

Design: SwapAB (K as A, Q_packed as B), PackGQA,
        1 epilogue warpgroup.
        CLC persistent scheduling.
        Optional k_block_size splits head_dim into chunks for reduced sK SMEM.

Multi-q_token support (q_tokens_per_tile = m_block_size // qhead_per_kvhead):
  When m_block_size = qhead_per_kvhead * 2 (e.g. qhpkv=64, m=128), the MMA
  tile packs 2 query tokens.  TMEM columns [0, qhpkv) belong to q_token_0
  and [qhpkv, 2*qhpkv) to q_token_1.  The epilogue performs independent
  head-reduce, score write, and denom computation for each q_token.
"""

import math
from functools import partial

import cuda.bindings.driver as cuda

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
from cutlass.utils.blackwell_helpers import (
    make_trivial_tiled_mma as _make_trivial_tiled_mma,
    make_smem_layout_a as _make_smem_layout_a,
    make_smem_layout_b as _make_smem_layout_b,
)
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator

from cudnn.deepseek_sparse_attention.utils.sm100.gemm import gemm_ptx_partial as _gemm_ptx_partial
from cudnn.deepseek_sparse_attention.utils import copy as copy_utils
from cudnn.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK
from .score_kernel_sm100 import ScoreKernelSm100

from .score_kernel_sm100 import add_packed_f32x2, fma_packed_f32x2  # noqa: E402

mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd="rn")


class DenseScoreRecomputeSm100(ScoreKernelSm100):
    """
    SM100 Cute-DSL kernel for the dense BF16 attention score: exp(QK*scale - LSE) -> head reduce -> scores + L1 denom.

    Supports q_tokens_per_tile = 1 or 2 (set via m_block_size = qhpkv * {1,2}).
    With 2 q_tokens, TMEM columns [0,qhpkv) = q_token_0, [qhpkv,2*qhpkv) = q_token_1.

    SwapAB design:
      - A = K (n_tile x head_dim), loaded via TMA
      - B = Q_packed (m_tile x head_dim), loaded via TMA
      - C = S^T in TMEM: (n_tile, m_tile)

    Warp layout (8 warps total):
      - Warp 0:     Load (TMA Q, TMA K, per-head LSE)
      - Warp 1:     MMA  (QK GEMM via TCGen05, swapAB)
      - Warp 2:     CLC scheduler (producer)
      - Warp 3:     Idle
      - Warps 4-7:  Epilogue warpgroup (score write + denom accumulation)
    """

    max_q_tokens_per_tile = 2

    def __init__(
        self,
        head_dim: int,
        qhead_per_kvhead: int = 32,
        m_block_size: int = 32,
        n_block_size: int = 128,
        kv_stage: int = 4,
        k_block_size: int | None = None,
        ratio: int = 1,
        is_varlen: bool = False,
    ):
        super().__init__(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            kv_stage=kv_stage,
            k_block_size=k_block_size,
            ratio=ratio,
            is_varlen=is_varlen,
        )
        # 8 warps (2 warpgroups), no WG2 for K loading
        self.epilogue_wg_warp_ids = (4, 5, 6, 7)
        self.num_warps = 8
        self.threads_per_cta = self.WARP_SIZE * self.num_warps  # 256

        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_s_stride = self.m_block_size
        # One accumulator slot per m_block_size TMEM columns, at most 16: the
        # per-slot phase bits live in one Int32 and a 16-deep S ring already
        # decouples the MMA from the epilogue (a 16-column tile would otherwise
        # ask for 32 slots and overflow the phase word).
        self.num_tmem_slots = min(16, SM100_TMEM_CAPACITY_COLUMNS // self.m_block_size)
        if qhead_per_kvhead in (64, 128):
            self.num_tmem_slots = 2
        self.tmem_total = self.tmem_s_stride * self.num_tmem_slots
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.Q_mbar_size = 2
        self.K_mbar_size = 2 * self.kv_stage
        self.S_mbar_size = 2 * self.num_tmem_slots
        self.s_empty_arrive_count = self.WARPGROUP_SIZE

        self.reduce_sync_mbar_size = 2
        self.reduce_sync_arrive_count = self.WARPGROUP_SIZE

        self.num_warps_in_epi_wg = len(self.epilogue_wg_warp_ids)
        # Cross-warp denom scratch, double-buffered by the reduce phase: a warp that races ahead to the next
        # tile writes the other half, so it never overwrites a partial another warp is still reading (a warp
        # cannot get two tiles ahead without every warp passing the intermediate reduce barrier).
        self.sScoreAll_phase_size = self.num_warps_in_epi_wg * 2
        self.sScoreAll_size = 2 * self.sScoreAll_phase_size

    # =========================================================================
    # Host-side entry
    # =========================================================================
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # BSHD (bs, seqlen_q, n_heads_q, head_dim) or THD (total_q, n_heads_q, head_dim) BF16
        mK: cute.Tensor,  # BSHD (bs, seqlen_k, n_heads_kv, head_dim) or THD (total_k, n_heads_kv, head_dim) BF16
        mPerHead: cute.Tensor,  # BSH (bs, seqlen_q, n_heads_q) or TH (total_q, n_heads_q) LSE (FP32)
        mOut: cute.Tensor,  # BSS (bs, seqlen_q, seqlen_k) or TS (total_q, max_seqlen_k) FP32
        mDenom: cute.Tensor,  # BS (bs, seqlen_q) or T (total_q,) FP32
        softmax_scale: Float32 | float,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mQCausalOffsets: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        """Host-side: layout transpose, PackGQA, TMA creation, kernel launch."""

        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        is_varlen = mCuSeqlensQ is not None

        self.tma_copy_bytes = {
            "Q": self.m_block_size * self.head_dim_padded * (self.q_dtype.width // 8),
            "K": self.n_block_size * self.k_block_size * (self.k_dtype.width // 8),
        }

        if const_expr(is_varlen):
            assert self.is_varlen
            assert mCuSeqlensQ is not None and mCuSeqlensK is not None
        else:
            assert not self.is_varlen
            assert mCuSeqlensQ is None and mCuSeqlensK is None

        # --- Layout transpose: sequence dim first for both BSHD and THD ---
        # BSHD: (bs, seqlen, heads, hdim) -> (seqlen, hdim, heads, bs)
        # THD:  (total, heads, hdim)      -> (total, hdim, heads)
        Q_layout_transpose = [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0]
        K_layout_transpose = [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=K_layout_transpose))

        # --- PackGQA: reshape Q to pack qhpkv heads into the seqlen dim ---
        # BSHD (rank 4 after transpose):  ((qhpkv, sq), hd, 1, bs)
        # THD  (rank 3 after transpose):  ((qhpkv, total_q), hd, 1)
        shape_Q_packed = (
            (self.qhead_per_kvhead, mQ.shape[0]),
            mQ.shape[1],
            1,
            *mQ.shape[3:],
        )
        stride_Q_packed = (
            (mQ.stride[2], mQ.stride[0]),
            mQ.stride[1],
            mQ.stride[2] * self.qhead_per_kvhead,
            *mQ.stride[3:],
        )
        mQ = cute.make_tensor(mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed))

        cta_group = tcgen05.CtaGroup.ONE
        self.q_major_mode = LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = LayoutEnum.from_tensor(mK).mma_major_mode()

        tiled_mma_qk = _make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.k_major_mode,
            self.q_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # --- SMEM layouts ---
        sK_layout = _make_smem_layout_a(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,
        )
        sQ_layout = _make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.num_k_chunks,
        )

        # --- TMA atoms for both Q and K ---
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        self.sQ_layout = sQ_layout
        self.sK_layout = sK_layout

        # --- PerHead layout: BSH (bs, sq, nh) -> (sq, nh, bs) ; TH (total_q, nh) stays ---
        PerHead_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
        mPerHead = cute.make_tensor(mPerHead.iterator, cute.select(mPerHead.layout, mode=PerHead_transpose))

        # --- Output layout: BSS (bs, sq, sk) -> (sq, sk, bs) ; TS (total_q, max_sk) stays ---
        Out_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
        mOut = cute.make_tensor(mOut.iterator, cute.select(mOut.layout, mode=Out_transpose))

        # --- Denom layout: BS (bs, sq) -> (sq, bs) ; T (total_q,) stays ---
        Denom_transpose = [0] if const_expr(is_varlen) else [1, 0]
        mDenom = cute.make_tensor(mDenom.iterator, cute.select(mDenom.layout, mode=Denom_transpose))

        # --- Grid and kernel dispatch (CLC persistent scheduling) ---
        # Grid sized using max_seqlen_q for varlen so persistent scheduling
        # can iterate over per-batch tiles up to the longest sequence.
        seqlen_q_static = max_seqlen_q if const_expr(is_varlen) else cute.size(mQ.shape[0]) // self.qhead_per_kvhead
        num_m_blocks = cute.ceil_div(seqlen_q_static * self.qhead_per_kvhead, self.m_block_size)
        batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1 if const_expr(is_varlen) else cute.size(mQ.shape[3])
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (num_m_blocks, 1, batch_size),
            (*self.cluster_shape_mn, 1),
        )
        grid_dim = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
        self.kernel(
            mQ,
            mK,
            mPerHead,
            mOut,
            mDenom,
            softmax_scale,
            tma_atom_Q,
            tma_atom_K,
            tiled_mma_qk,
            sQ_layout,
            sK_layout,
            tile_sched_params,
            max_seqlen_q,
            max_seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mQCausalOffsets,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            # One CTA of two warpgroups per SM: every thread can hold up to 255
            # registers, so no setmaxnreg reallocation is needed.
            min_blocks_per_mp=1,
        )

    # =========================================================================
    # Device-side kernel
    # =========================================================================
    @cute.kernel
    def kernel(
        self,
        mQ,
        mK,
        mPerHead,
        mOut,
        mDenom,
        softmax_scale: Float32 | float,
        tma_atom_Q,
        tma_atom_K,
        tiled_mma_qk,
        sQ_layout,
        sK_layout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mQCausalOffsets: cute.Tensor | None,
    ):
        """Device-side kernel entry with CLC persistent scheduling."""
        is_varlen = mCuSeqlensQ is not None
        if const_expr(is_varlen):
            assert self.is_varlen
        else:
            assert not self.is_varlen

        # Static seqlens used as fallback when varlen is disabled.
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
        neg_log2_e = Float32(-math.log2(math.e))

        # --- TMA descriptor prefetch ---
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)

        # =====================================================================
        # Shared memory allocation
        # =====================================================================
        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)
        sPerHead_size = self.m_block_size * 2

        @cute.struct
        class SharedStorage:
            per_head_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.Q_mbar_size]
            K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.K_mbar_size]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.S_mbar_size]
            reduce_sync_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.reduce_sync_mbar_size]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, 4]
            sPerHead: cute.struct.Align[
                cute.struct.MemRange[Float32, sPerHead_size],
                128,
            ]
            sScoreAll: cute.struct.Align[
                cute.struct.MemRange[Float32, self.sScoreAll_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sK_size],
                self.buffer_align_bytes,
            ]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)

        Q_mbar_ptr = storage.Q_mbar_ptr.data_ptr()
        K_mbar_ptr = storage.K_mbar_ptr.data_ptr()
        S_mbar_ptr = storage.S_mbar_ptr.data_ptr()
        reduce_sync_mbar_ptr = storage.reduce_sync_mbar_ptr.data_ptr()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf.ptr
        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sPerHead = storage.sPerHead.get_tensor(cute.make_layout((self.m_block_size,), stride=(1,)))
        sScoreAll = storage.sScoreAll.get_tensor(cute.make_layout((self.sScoreAll_size,), stride=(1,)))

        # =====================================================================
        # Pipeline setup
        # =====================================================================
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0

        # Per-head LSE (2 stages): all 32 load lanes publish after their stores; every epilogue lane
        # releases a slot only after the tile's epilogue has read it.
        pipeline_head = PipelineAsync.create(
            barrier_storage=storage.per_head_mbar_ptr.data_ptr(),
            num_stages=2,
            producer_group=CooperativeGroup(Agent.Thread, self.WARP_SIZE),
            consumer_group=CooperativeGroup(Agent.Thread, self.WARPGROUP_SIZE),
            defer_sync=True,
        )

        # Q pipeline (1 stage)
        pipeline_Q = PipelineTmaUmma.create(
            barrier_storage=Q_mbar_ptr,
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # K pipeline (TMA, load warp produces, MMA warp consumes)
        pipeline_K = PipelineTmaUmma.create(
            barrier_storage=K_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # S barriers: paired (s_full, s_empty) per TMEM slot
        if warp_idx == 1:
            cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, 1)
        if warp_idx == 0:
            for _si in cutlass.range_constexpr(self.num_tmem_slots):
                cute.arch.mbarrier_init(S_mbar_ptr + 2 * _si, 1)
                cute.arch.mbarrier_init(S_mbar_ptr + 2 * _si + 1, self.s_empty_arrive_count)
            cute.arch.mbarrier_init(reduce_sync_mbar_ptr, self.reduce_sync_arrive_count)

        # CLC persistent scheduling pipeline
        cluster_size = cute.size(self.cluster_shape_mn)
        num_clc_consumer_threads = self.WARP_SIZE * (1 + cluster_size * (1 + 1 + 4))
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
        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
        work_tile = tile_sched.initial_work_tile_info()

        head_producer, head_consumer = pipeline_head.make_participants()
        Q_producer, Q_consumer = pipeline_Q.make_participants()
        K_producer, K_consumer = pipeline_K.make_participants()

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS_ref = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        # Each warp belongs to one warpgroup. Preserve that uniformity across
        # the load/MMA and epilogue branches to avoid redundant collectives.
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.WARPGROUP_SIZE)

        # =====================================================================
        # Warpgroup 0 (warps 0-3): Load + MMA + Scheduler
        # =====================================================================
        if warp_group_idx == 0:
            # -----------------------------------------------------------------
            # Load warp (warp 0): PerHead data + TMA Q + TMA K
            # -----------------------------------------------------------------
            if warp_idx == self.load_warp_id:
                rows_per_thread = cute.ceil_div(self.m_block_size, self.WARP_SIZE)
                lane_id = tidx % self.WARP_SIZE

                while work_tile.is_valid_tile:
                    m_block_sched = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                    num_m_blocks_cur = cute.ceil_div(
                        seqlen.seqlen_q * self.qhead_per_kvhead,
                        self.m_block_size,
                    )
                    is_valid_m_block = m_block_sched < num_m_blocks_cur

                    if is_valid_m_block:
                        m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                        num_n_blocks_compute = self._dense_compute_n_blocks(
                            m_block,
                            seqlen.seqlen_q,
                            seqlen.seqlen_k,
                            q_causal_offset,
                        )
                        # PerHead data load (double-buffered)
                        head = head_producer.acquire_and_advance()
                        per_head_buf_off = head.index * self.m_block_size
                        mPerHead_cur = seqlen.offset_batch_Q(mPerHead, batch_idx, dim=2)
                        for ri in cutlass.range_constexpr(rows_per_thread):
                            row = ri * self.WARP_SIZE + lane_id
                            if row < self.m_block_size:
                                m_packed_idx = m_block * self.m_block_size + row
                                m_idx = m_packed_idx // self.qhead_per_kvhead
                                h_idx = m_packed_idx - m_idx * self.qhead_per_kvhead
                                if m_idx < seqlen.seqlen_q:
                                    sPerHead[per_head_buf_off + row] = Float32(mPerHead_cur[m_idx, h_idx]) * neg_log2_e
                                else:
                                    sPerHead[per_head_buf_off + row] = Float32(0.0)

                        head.commit()
                        cute.arch.fence_view_async_shared()

                        # TMA Q load (1 barrier, num_k_chunks TMA copies)
                        Q_producer.reset()
                        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, 0]
                        handle_Q = Q_producer.acquire_and_advance()
                        for _kc in cutlass.range_constexpr(self.num_k_chunks):
                            gQ = cute.local_tile(
                                mQ_cur,
                                (self.m_block_size, self.k_block_size),
                                (m_block, _kc),
                            )
                            sQ_cur = sQ[None, None, None, _kc]
                            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_Q,
                                0,
                                cute.make_layout(1),
                                gQ,
                                sQ_cur,
                                single_stage=True,
                            )
                            load_Q_fn(tma_bar_ptr=handle_Q.barrier)

                        # TMA K load (reverse order, matching MMA direction)
                        K_producer.reset()
                        mK_src = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, 0]
                        n_block_k = num_n_blocks_compute - Int32(1)
                        while n_block_k >= Int32(0):
                            for _kc in cutlass.range_constexpr(self.num_k_chunks):
                                handle_K = K_producer.acquire_and_advance()
                                gK = cute.local_tile(
                                    mK_src,
                                    (self.n_block_size, self.k_block_size),
                                    (n_block_k, _kc),
                                )
                                sK_stage = sK[None, None, None, handle_K.index]
                                load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_K,
                                    0,
                                    cute.make_layout(1),
                                    gK,
                                    sK_stage,
                                    single_stage=True,
                                )
                                load_K_fn(tma_bar_ptr=handle_K.barrier)
                            n_block_k = n_block_k - 1

                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                head_producer.tail()
                Q_producer.tail()
                K_producer.tail()

            # -----------------------------------------------------------------
            # MMA warp (warp 1): QK GEMM via TCGen05
            # -----------------------------------------------------------------
            if warp_idx == self.mma_warp_id:
                tmem_alloc_cols = Int32(self.tmem_alloc_cols)
                cute.arch.alloc_tmem(tmem_alloc_cols, tmem_holding_buf)
                cute.arch.sync_warp()

                s_empty_phase_bits = Int32((1 << self.num_tmem_slots) - 1)
                K_mma_state = make_pipeline_state(PipelineUserType.Consumer, self.kv_stage)
                while work_tile.is_valid_tile:
                    m_block_sched = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                    num_m_blocks_cur = cute.ceil_div(
                        seqlen.seqlen_q * self.qhead_per_kvhead,
                        self.m_block_size,
                    )
                    is_valid_m_block = m_block_sched < num_m_blocks_cur

                    if is_valid_m_block:
                        m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                        num_n_blocks_compute = self._dense_compute_n_blocks(
                            m_block,
                            seqlen.seqlen_q,
                            seqlen.seqlen_k,
                            q_causal_offset,
                        )
                        Q_consumer.reset()
                        handle_Q = Q_consumer.wait_and_advance()

                        tSrK = tiled_mma_qk.make_fragment_A(sK)
                        tSrQ = tiled_mma_qk.make_fragment_B(sQ)
                        qk_mma_op = tiled_mma_qk.op

                        n_block = num_n_blocks_compute - Int32(1)

                        while n_block >= Int32(0):
                            slot = n_block % Int32(self.num_tmem_slots)
                            s_empty_phase = self._phase_for_slot(s_empty_phase_bits, slot)
                            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot + 1, s_empty_phase)

                            for _kc in cutlass.range_constexpr(self.num_k_chunks):
                                pipeline_K.consumer_wait(K_mma_state)

                                tSrKi = tSrK[None, None, None, K_mma_state.index]
                                sK_cur = sK[None, None, None, K_mma_state.index]
                                tSrQ_kc = tSrQ[None, None, None, _kc]
                                sQ_kc = sQ[None, None, None, _kc]
                                _gemm_ptx_partial(
                                    qk_mma_op,
                                    slot * self.tmem_s_stride,
                                    tSrKi,
                                    tSrQ_kc,
                                    sA=sK_cur,
                                    sB=sQ_kc,
                                    zero_init=const_expr(_kc == 0),
                                )

                                pipeline_K.consumer_release(K_mma_state)
                                K_mma_state.advance()

                            with cute.arch.elect_one():
                                tcgen05.commit(S_mbar_ptr + 2 * slot)

                            s_empty_phase_bits = self._toggle_phase_for_slot(
                                s_empty_phase_bits,
                                slot,
                            )
                            n_block = n_block - 1

                        handle_Q.release()

                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

                # TMEM dealloc
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                tmem_ptr_dealloc = cute.arch.retrieve_tmem_ptr(
                    Float32,
                    alignment=16,
                    ptr_to_buffer_holding_addr=tmem_holding_buf,
                )
                cute.arch.dealloc_tmem(tmem_ptr_dealloc, Int32(self.tmem_alloc_cols))

            # -----------------------------------------------------------------
            # Scheduler warp (warp 2): CLC producer loop
            # -----------------------------------------------------------------
            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                clc_producer_state = make_pipeline_state(PipelineUserType.ProducerConsumer, self.num_clc_stage)
                while work_tile.is_valid_tile:
                    clc_pipeline.producer_acquire(clc_producer_state)
                    mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                    tile_sched.advance_to_next_work(mbarrier_addr)
                    clc_producer_state.advance()
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                clc_pipeline.producer_tail(clc_producer_state)

        # =====================================================================
        # Warpgroup 1 (warps 4-7): Epilogue — score write + denom accumulation
        # =====================================================================
        if warp_group_idx == 1:
            s_full_phase_bits = Int32(0)
            reduce_phase = Int32(0)
            while work_tile.is_valid_tile:
                m_block_sched = work_tile.tile_idx[0]
                batch_idx = work_tile.tile_idx[2]
                seqlen = SeqlenInfoCls(batch_idx)
                q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                num_m_blocks_cur = cute.ceil_div(
                    seqlen.seqlen_q * self.qhead_per_kvhead,
                    self.m_block_size,
                )
                is_valid_m_block = m_block_sched < num_m_blocks_cur

                if is_valid_m_block:
                    m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                    num_n_blocks_compute = self._dense_compute_n_blocks(
                        m_block,
                        seqlen.seqlen_q,
                        seqlen.seqlen_k,
                        q_causal_offset,
                    )
                    # The epilogue copies the tile's LSE into registers at its first block and hands the slot
                    # back to the load warp right there; a tile with no key block releases it here.
                    head = head_consumer.wait_and_advance()
                    sLSE_cur = cute.make_tensor(
                        cute.make_ptr(Float32, (sPerHead.iterator + head.index * self.m_block_size).llvm_ptr, cute.AddressSpace.smem, assumed_align=16),
                        cute.make_layout((self.m_block_size,)),
                    )
                    if num_n_blocks_compute == Int32(0):
                        head.release()
                    mDenom_cur = seqlen.offset_batch_Q(mDenom, batch_idx, dim=1)
                    mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=2)
                    s_full_phase_bits, reduce_phase = self._epilogue_attention_dense(
                        tiled_mma_qk,
                        tStS_ref,
                        sLSE_cur,
                        head,
                        sScoreAll,
                        S_mbar_ptr,
                        reduce_sync_mbar_ptr,
                        mOut_cur,
                        mDenom_cur,
                        num_n_blocks_compute,
                        seqlen.seqlen_k,
                        seqlen.seqlen_q,
                        max_seqlen_k,
                        q_causal_offset,
                        m_block,
                        tidx,
                        s_full_phase_bits,
                        reduce_phase,
                        softmax_scale,
                    )
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            if warp_idx == self.epilogue_wg_warp_ids[-1]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    @cute.jit
    def _phase_for_slot(self, phase_bits, slot):
        return (phase_bits >> slot) & Int32(1)

    @cute.jit
    def _toggle_phase_for_slot(self, phase_bits, slot):
        return phase_bits ^ (Int32(1) << slot)

    # =========================================================================
    # Cross-warp reduce helpers (epilogue warpgroup)
    # =========================================================================
    @cute.jit
    def _inter_warp_sync_sum(
        self,
        sScoreAll,
        reduce_sync_mbar_ptr,
        reduce_sync_phase,
        warp_id_in_wg,
        warp_sum,
    ):
        """Cross-warp sum sync with pre-computed warp-level sum (scratch half selected by the reduce phase)."""
        red = reduce_sync_phase * Int32(self.sScoreAll_phase_size)
        with cute.arch.elect_one():
            sScoreAll[red + warp_id_in_wg] = warp_sum
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
        cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_sync_phase)
        reduce_sync_phase = reduce_sync_phase ^ 1

        global_sum = sScoreAll[red]
        for wi in cutlass.range_constexpr(self.num_warps_in_epi_wg - 1):
            global_sum = global_sum + sScoreAll[red + wi + 1]

        return global_sum, reduce_sync_phase

    # =========================================================================
    # Epilogue: dense attention — exp(QK*scale - LSE), head reduce, write + L1 denom
    # =========================================================================
    @cute.jit
    def _epilogue_attention_dense(
        self,
        tiled_mma_qk,
        tStS_ref,
        sLSE_1d,
        lse_handle,
        sScoreAll,
        S_mbar_ptr,
        reduce_sync_mbar_ptr,
        mOut,
        mDenom,
        num_n_blocks_compute,
        seqlen_k,
        seqlen_q,
        max_seqlen_k,
        q_causal_offset,
        m_block,
        tidx,
        s_full_phase_bits,
        reduce_phase,
        softmax_scale,
    ):
        """Dense attention epilogue: TMEM->RF, exp2(S*scale - LSE), head reduce.  ``sLSE_1d`` is the tile's
        ``-lse * log2(e)`` SMEM slot: copied to registers at the first block, then handed back (``lse_handle``).

        Writes score per n_block to GMEM, accumulates L1-norm denom.
        Supports q_tokens_per_tile = 1 or 2: each q_token's heads occupy
        a contiguous [qi*qhpkv, (qi+1)*qhpkv) slice of the m_block_size
        columns in tSrS and rLSE_all.
        """
        tidx_wg = tidx % self.WARPGROUP_SIZE

        qhpkv = self.qhead_per_kvhead
        q_tokens_per_tile = self.q_tokens_per_tile
        ratio = Int32(self.ratio)

        LSE_ILP = 4
        rLSE_all = cute.make_rmem_tensor((self.m_block_size,), Float32)

        kv_offset = Int32(0)
        warp_id_in_wg = tidx_wg // self.WARP_SIZE

        log2_e = Float32(math.log2(math.e))
        scale_log2_e = Float32(softmax_scale) * log2_e

        q_token_base = m_block * q_tokens_per_tile
        # Per-thread L1-norm denom accumulators. Reduce once at tile end
        # instead of doing a warp reduction for every K block.
        local_sum_acc = [Float32(0.0) for _ in range(q_tokens_per_tile)]
        first_block = Int32(1)

        n_blk = num_n_blocks_compute - 1
        while n_blk >= Int32(0):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_repetition)),
                Float32,
            )
            thr_tmem_load = tcgen05.make_tmem_copy(
                tmem_load_atom,
                tStS_ref,
            ).get_slice(tidx_wg)

            thr_mma = tiled_mma_qk.get_slice(tidx_wg)
            cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
            tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))

            tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
            tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

            slot = n_blk % Int32(self.num_tmem_slots)
            s_full_phase = self._phase_for_slot(s_full_phase_bits, slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot, s_full_phase)

            if first_block == Int32(1):
                cute.autovec_copy(sLSE_1d, rLSE_all)
                lse_handle.release()
                first_block = Int32(0)

            tmem_ptr_cur = cute.make_ptr(
                Float32,
                slot * self.tmem_s_stride,
                mem_space=cute.AddressSpace.tmem,
                assumed_align=16,
            )
            tStS_cur = cute.make_tensor(tmem_ptr_cur, tStS_ref.layout)
            tStS_t2r_cur = thr_tmem_load.partition_S(tStS_cur)

            cute.copy(thr_tmem_load, tStS_t2r_cur, tSrS)
            cute.arch.fence_view_async_tmem_load()

            cute.arch.mbarrier_arrive(S_mbar_ptr + 2 * slot + 1)
            s_full_phase_bits = self._toggle_phase_for_slot(
                s_full_phase_bits,
                slot,
            )

            kv_offset = tScS[0][0]
            pos = kv_offset + n_blk * self.n_block_size

            # Head reduce per q_token. rLSE_all already holds
            # -log2(e) * LSE, loaded by the producer warp.
            for qi in cutlass.range_constexpr(q_tokens_per_tile):
                local_sum = (Float32(0.0), Float32(0.0))
                for ho in cutlass.range_constexpr(qhpkv // 2 // LSE_ILP):
                    for ci in cutlass.range_constexpr(LSE_ILP):
                        idx0 = qi * qhpkv + (ho * LSE_ILP + ci) * 2
                        idx1 = idx0 + 1
                        lse_pair = (rLSE_all[idx0], rLSE_all[idx1])

                        val0 = tSrS[idx0]
                        val1 = tSrS[idx1]

                        val0, val1 = fma_packed_f32x2(
                            (val0, val1),
                            (scale_log2_e, scale_log2_e),
                            lse_pair,
                        )
                        if cutlass.const_expr(qhpkv == 64):
                            val0 = cute.math.exp2(val0, fastmath=True)
                            val1 = cute.math.exp2(val1, fastmath=True)
                        else:
                            val0 = val0 if val0 > Float32(-126.0) else Float32(-126.0)
                            val1 = val1 if val1 > Float32(-126.0) else Float32(-126.0)
                            val0 = cute.math.exp2(val0, fastmath=True)
                            val1 = cute.math.exp2(val1, fastmath=True)

                        local_sum = add_packed_f32x2(local_sum, (val0, val1))

                score = local_sum[0] + local_sum[1]

                # Ratio causal mask + tail OOB. Keep the accumulation value at
                # zero for invalid positions, but expose -inf in the score output.
                q_token_idx = q_token_base + qi
                col_limit = (q_causal_offset + q_token_idx + 1) // ratio
                score_out = score
                if pos >= col_limit or pos >= seqlen_k or q_token_idx >= seqlen_q:
                    score = Float32(0.0)
                    score_out = -Float32.inf

                if q_token_idx < seqlen_q and pos < max_seqlen_k:
                    mOut[q_token_idx, pos] = score_out

                local_sum_acc[qi] = local_sum_acc[qi] + score

            n_blk = n_blk - 1

        # Fill wholly skipped future/tail output blocks with the score default.
        # The main
        # loop only visits K blocks that can contain valid causal columns; keep
        # this local to skipped blocks instead of doing a full host-side
        # memset before launch.
        n_blk_zero = cute.ceil_div(max_seqlen_k, self.n_block_size) - Int32(1)
        if cutlass.const_expr(q_tokens_per_tile == 1):
            while n_blk_zero >= num_n_blocks_compute:
                pos = kv_offset + n_blk_zero * self.n_block_size
                if q_token_base < seqlen_q and pos < max_seqlen_k:
                    mOut[q_token_base, pos] = -Float32.inf
                n_blk_zero = n_blk_zero - 1
        else:
            while n_blk_zero >= num_n_blocks_compute:
                pos = kv_offset + n_blk_zero * self.n_block_size
                for qi in cutlass.range_constexpr(q_tokens_per_tile):
                    q_token_idx = q_token_base + qi
                    if q_token_idx < seqlen_q and pos < max_seqlen_k:
                        mOut[q_token_idx, pos] = -Float32.inf
                n_blk_zero = n_blk_zero - 1

        # Cross-warp reduce for global L1 denom. The 2q path writes both
        # per-warp partials before one barrier and then reduces both q tokens,
        # avoiding the second mbarrier round-trip in the flash configuration.
        if cutlass.const_expr(q_tokens_per_tile == 2):
            warp_sum0 = cute.arch.warp_reduction_sum(local_sum_acc[0])
            warp_sum1 = cute.arch.warp_reduction_sum(local_sum_acc[1])
            red = reduce_phase * Int32(self.sScoreAll_phase_size)
            with cute.arch.elect_one():
                sScoreAll[red + warp_id_in_wg] = warp_sum0
                sScoreAll[red + warp_id_in_wg + self.num_warps_in_epi_wg] = warp_sum1
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
            cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_phase)
            reduce_phase = reduce_phase ^ 1

            global_sum0 = sScoreAll[red]
            global_sum1 = sScoreAll[red + self.num_warps_in_epi_wg]
            for wi in cutlass.range_constexpr(self.num_warps_in_epi_wg - 1):
                global_sum0 = global_sum0 + sScoreAll[red + wi + 1]
                global_sum1 = global_sum1 + sScoreAll[red + self.num_warps_in_epi_wg + wi + 1]

            if q_token_base < seqlen_q:
                with cute.arch.elect_one():
                    mDenom[q_token_base] = global_sum0
            q_token_idx1 = q_token_base + Int32(1)
            if q_token_idx1 < seqlen_q:
                with cute.arch.elect_one():
                    mDenom[q_token_idx1] = global_sum1
        else:
            warp_sum = cute.arch.warp_reduction_sum(local_sum_acc[0])
            global_sum, reduce_phase = self._inter_warp_sync_sum(
                sScoreAll,
                reduce_sync_mbar_ptr,
                reduce_phase,
                warp_id_in_wg,
                warp_sum,
            )

            if q_token_base < seqlen_q:
                with cute.arch.elect_one():
                    mDenom[q_token_base] = global_sum

        return s_full_phase_bits, reduce_phase
