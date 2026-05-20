"""
Indexer QK Forward Kernel — SM100 Cute-DSL Implementation.

Computes: S_sum(b,q,k) = sum_h [ReLU(Q_h · K_{g(h)}^T) · W_{b,q,h}]
with bottom-right aligned ratio causal mask, output FP32.

Design: SwapAB (K as A, Q_packed as B), PackGQA,
        Ld32x32bOp head reduction, 2 epilogue warpgroups for 2 q_stages.
        SingleTile LPT scheduling for causal load balancing.
"""

import math
from typing import Optional, Tuple
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

from cudnn.deepseek_sparse_attention.utils.sm100.gemm import gemm_ptx_partial as _gemm_ptx_partial
from cudnn.deepseek_sparse_attention.utils import copy as copy_ops
from cudnn.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK

mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd="rn")
add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd="rn")


class IndexerForwardSm100:
    """
    SM100 Cute-DSL kernel for Indexer QK scoring.

    SwapAB design:
      - A = K (n_tile x head_dim), loaded via TMA as A operand
        # For sparse GEMM: use sparse load KV via cp.async instead of TMA.
      - B = Q_packed (m_tile x head_dim), loaded via TMA as B operand
      - C = S^T in TMEM: (n_tile, m_tile), m_tile = q_tokens x qhpkv

    Warp layout (12 warps total):
      - Warp 0:     Load (TMA K/Q, regular W load)
      - Warp 1:     MMA  (QK GEMM via TCGen05, swapAB)
      - Warp 2:     CLC sched (producer, LPT tile order)
      - Warp 3:     TMA Score store
      - Warps 4-7:  Epilogue warpgroup 0 (q_stage=0)
      - Warps 8-11: Epilogue warpgroup 1 (q_stage=1)
    """

    arch = 100

    def __init__(
        self,
        head_dim: int,
        qhead_per_kvhead: int = 32,
        ratio: int = 4,
        is_varlen: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        kv_stage: int = 4,
    ):
        self.head_dim = head_dim
        self.qhead_per_kvhead = qhead_per_kvhead
        self.ratio = ratio
        self.is_varlen = is_varlen
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = q_stage
        self.kv_stage = kv_stage
        self.sched_warp_id = 2
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16
        assert self.num_clc_stage == 1, "Only single-stage CLC pipeline is supported"

        # Pad head_dim to multiple of 16
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)

        # q_tokens per q_stage tile
        self.q_tokens_per_tile = m_block_size // qhead_per_kvhead

        # swapAB: K is A (M-dim), Q_packed is B (N-dim)
        # After swap: M=n_block_size, N=m_block_size, K=head_dim
        self.mma_tiler_qk = (n_block_size, m_block_size, self.head_dim_padded)

        # CTA tiler: (q_stage * m_block, n_block, hdim)
        self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)

        # Dtypes: set from input tensors in __call__ (no hardcode; compile per dtype like flash_attn)
        self.qk_acc_dtype = Float32

        # Warp config: 12 warps
        self.load_warp_id = 0
        self.mma_warp_id = 1
        self.tma_store_warp_id = 3  # warp 2 is CLC sched
        self.epilogue_wg0_warp_ids = (4, 5, 6, 7)
        self.epilogue_wg1_warp_ids = (8, 9, 10, 11)
        self.num_warps = 12
        self.threads_per_cta = 32 * self.num_warps  # 384

        # Register allocation
        self.num_regs_load = 48
        self.num_regs_mma = 48
        self.num_regs_epilogue = 224

        # TMEM: q_stage * n_tile * m_tile columns
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_s_offset = [i * self.n_block_size for i in range(self.q_stage)]
        self.tmem_total = self.q_stage * self.n_block_size  # 2*128 = 256
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        # Pipeline barrier counts
        self.Q_mbar_size = 2 * self.q_stage
        self.K_mbar_size = 2 * self.kv_stage
        self.S_mbar_size = 2 * self.q_stage
        # S barriers: s_full count=1 (MMA commit), s_empty count=128 (epilogue warpgroup arrive)
        self.s_empty_arrive_count = 128

        # Score store barriers: full/empty for each q_stage
        self.Score_store_mbar_size = 2 * self.q_stage
        self.score_store_full_arrive_count = 128  # EPI WG (128 threads) arrive
        self.score_store_empty_arrive_count = 32  # TMA store warp (32 threads) arrive

        # Score smem: (q_tokens_per_tile, n_block_size) per q_stage
        self.sScore_size = self.q_tokens_per_tile * self.n_block_size * self.q_stage

        # SMEM alignment
        self.buffer_align_bytes = 1024

        # Cluster shape (no multi-CTA clustering for simplicity)
        self.cluster_shape_mn = (1, 1)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (bs, seqlen_q, n_heads_q, head_dim) BF16
        mK: cute.Tensor,  # (bs, seqlen_k, n_heads_kv, head_dim) BF16
        mW: cute.Tensor,  # (bs, seqlen_q, n_heads_q) BF16
        mOut: cute.Tensor,  # (bs, seqlen_q, seqlen_k) FP32
        n_heads_kv: Int32,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        sm_scale: Float32 | float,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        """Host-side: layout transpose, PackGQA, TMA creation, kernel launch.

        sm_scale is applied to the fp32 head-reduced score inside the kernel
        (post head-reduce, pre causal mask) so that callers can pass the raw
        bf16 W tensor without losing precision to a host-side scalar multiply.
        """

        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.w_dtype = mW.element_type
        is_varlen = mCuSeqlensQ is not None

        # TMA copy bytes per tile (for pipeline tx_count)
        self.tma_copy_bytes = {
            "Q": self.m_block_size * self.head_dim_padded * (self.q_dtype.width // 8),
            "K": self.n_block_size * self.head_dim_padded * (self.k_dtype.width // 8),
        }

        if const_expr(is_varlen):
            assert self.is_varlen
            assert mCuSeqlensQ is not None and mCuSeqlensK is not None
        else:
            assert not self.is_varlen
            assert mCuSeqlensQ is None and mCuSeqlensK is None

        # --- Layout transpose: sequence dimension first for both fixed and THD inputs ---
        Q_layout_transpose = [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0]
        K_layout_transpose = [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=K_layout_transpose))

        # --- PackGQA: reshape Q to pack qhpkv heads into seqlen_q dim ---
        shape_Q_packed = (
            (self.qhead_per_kvhead, mQ.shape[0]),  # packed M dim
            mQ.shape[1],  # head_dim
            mK.shape[2],  # n_heads_kv
            *mQ.shape[3:],  # bs (and beyond)
        )
        stride_Q_packed = (
            (mQ.stride[2], mQ.stride[0]),  # (stride_head, stride_seqlen)
            mQ.stride[1],  # stride_hdim
            mQ.stride[2] * self.qhead_per_kvhead,  # stride across kv_head groups
            *mQ.stride[3:],  # stride_bs
        )
        mQ = cute.make_tensor(mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed))

        cta_group = tcgen05.CtaGroup.ONE
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()

        tiled_mma_qk = _make_trivial_tiled_mma(
            self.q_dtype,
            self.k_major_mode,  # A operand major mode (K)
            self.q_major_mode,  # B operand major mode (Q)
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],  # (n_block, m_block) after swap
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
            self.q_stage,
        )

        # --- TMA atoms ---
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        self.sQ_layout = sQ_layout
        self.sK_layout = sK_layout
        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)

        # --- W layout transpose: sequence dimension first ---
        W_layout_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
        mW = cute.make_tensor(mW.iterator, cute.select(mW.layout, mode=W_layout_transpose))

        # --- Output layout transpose: sequence dimension first ---
        Out_layout_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
        mOut = cute.make_tensor(mOut.iterator, cute.select(mOut.layout, mode=Out_layout_transpose))
        mOut_scalar = mOut

        # --- TMA store descriptor for Score (smem -> gmem) ---
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        score_tile = (self.q_tokens_per_tile, self.n_block_size)
        score_cta_v_layout = cute.composition(cute.make_identity_layout(mOut.shape), score_tile)
        sScore_layout = cute.make_layout(
            (self.q_tokens_per_tile, self.n_block_size, self.q_stage),
            stride=(self.n_block_size, 1, self.q_tokens_per_tile * self.n_block_size),
        )
        sScore_single = cute.select(sScore_layout, mode=[0, 1])
        tma_atom_Score, mOut = cpasync.make_tiled_tma_atom(
            tma_store_op,
            mOut,
            sScore_single,
            score_cta_v_layout,
        )
        self.sScore_layout = sScore_layout

        # --- Grid and kernel dispatch (CLC persistent scheduling) ---
        # Grid tiles only in M and batch — each CTA iterates all n_blocks
        # internally, so N-dim of grid must be 1 (NOT tiled by n_block_size).
        seqlen_q_static = max_seqlen_q if const_expr(is_varlen) else cute.size(mQ.shape[0]) // self.qhead_per_kvhead
        seqlen_q_packed = seqlen_q_static * self.qhead_per_kvhead
        num_m_blocks = cute.ceil_div(seqlen_q_packed, self.q_stage * self.m_block_size)
        batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1 if const_expr(is_varlen) else cute.size(mQ.shape[3])
        cluster_shape_mnl = (*self.cluster_shape_mn, 1)
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams((num_m_blocks, 1, batch_size), cluster_shape_mnl)
        grid_dim = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
        self.kernel(
            mQ,
            mK,
            mW,
            mOut,
            mOut_scalar,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_Score,
            tiled_mma_qk,
            sQ_layout,
            sK_layout,
            sScore_layout,
            tile_sched_params,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
            mCuSeqlensQ,
            mCuSeqlensK,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ,
        mK,
        mW,
        mOut,
        mOut_scalar,
        tma_atom_Q,
        tma_atom_K,
        tma_atom_Score,
        tiled_mma_qk,
        sQ_layout,
        sK_layout,
        sScore_layout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        sm_scale: Float32 | float,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
    ):
        """Device-side kernel entry with CLC persistent scheduling.

        seqlen_q_packed, seqlen_k, num_n_blocks are derived at runtime
        from the dynamic tensor shapes (mark_layout_dynamic) so that
        a single compilation works for any sequence length.
        """
        is_varlen = mCuSeqlensQ is not None
        if const_expr(is_varlen):
            assert self.is_varlen
        else:
            assert not self.is_varlen

        # --- Runtime seqlen from dynamic tensor shapes ---
        seqlen_q = max_seqlen_q if const_expr(is_varlen) else cute.size(mQ.shape[0]) // self.qhead_per_kvhead
        seqlen_k = max_seqlen_k if const_expr(is_varlen) else cute.size(mK.shape[0])
        seqlen_q_packed = seqlen_q * self.qhead_per_kvhead
        num_m_blocks = cute.ceil_div(seqlen_q_packed, self.q_stage * self.m_block_size)
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=seqlen_q,
            seqlen_k_static=seqlen_k,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            tile_m=self.q_stage * self.m_block_size,
            tile_n=self.n_block_size,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]

        # --- TMA descriptor prefetch ---
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_Score)

        # --- Allocate SMEM ---
        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)
        sW_single = self.m_block_size * self.q_stage
        sW_size = sW_single * 2  # double-buffer to avoid LOAD/EPI race
        sScore_size = self.sScore_size

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.Q_mbar_size]
            K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.K_mbar_size]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.S_mbar_size]
            Score_store_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.Score_store_mbar_size]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, 4]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.w_dtype, sW_size],
                128,
            ]
            sScore: cute.struct.Align[
                cute.struct.MemRange[Float32, sScore_size],
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

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Extract all pointers/tensors from storage so no 'if' body references storage
        # (avoids DSLRuntimeError: SharedStorage as user-defined Python object in if)
        Q_mbar_ptr = storage.Q_mbar_ptr.data_ptr()
        K_mbar_ptr = storage.K_mbar_ptr.data_ptr()
        S_mbar_ptr = storage.S_mbar_ptr.data_ptr()
        Score_store_mbar_ptr = storage.Score_store_mbar_ptr.data_ptr()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf
        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sW_layout_1d = cute.make_layout((self.m_block_size * self.q_stage,), stride=(1,))
        sW = storage.sW.get_tensor(sW_layout_1d)
        sScore = storage.sScore.get_tensor(sScore_layout)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0

        # TMA producer (load warp) and UMMA consumer (mma warp) pipelines
        pipeline_producer_group = CooperativeGroup(Agent.Thread, len([self.load_warp_id]))
        pipeline_consumer_group = CooperativeGroup(Agent.Thread, len([self.mma_warp_id]))
        pipeline_Q = PipelineTmaUmma.create(
            barrier_storage=Q_mbar_ptr,
            num_stages=self.q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_K = PipelineTmaUmma.create(
            barrier_storage=K_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # S barriers and tmem_dealloc: manual init (S has custom s_empty count=128)
        if warp_idx == 1:
            cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, 2)  # both epilogue WG0 and WG1 must arrive
        if warp_idx == 0:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(S_mbar_ptr + i, 1)  # s_full
                cute.arch.mbarrier_init(S_mbar_ptr + self.q_stage + i, self.s_empty_arrive_count)  # s_empty
        if warp_idx == self.tma_store_warp_id:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(Score_store_mbar_ptr + i, self.score_store_full_arrive_count)  # Score_full
                cute.arch.mbarrier_init(
                    Score_store_mbar_ptr + self.q_stage + i,
                    self.score_store_empty_arrive_count,
                )  # Score_empty

        cluster_size = cute.size(self.cluster_shape_mn)
        num_clc_consumer_threads = 32 * (1 + cluster_size * (1 + 1 + 8 + 1))  # load(1) + mma(1) + tma_store(1) + epilogue(8) + sched(1) warps
        clc_pipeline_producer_group = CooperativeGroup(Agent.Thread)
        clc_pipeline_consumer_group = CooperativeGroup(Agent.Thread, num_clc_consumer_threads)
        clc_pipeline = PipelineClcFetchAsync.create(
            barrier_storage=clc_mbar_ptr,
            num_stages=self.num_clc_stage,
            producer_group=clc_pipeline_producer_group,
            consumer_group=clc_pipeline_consumer_group,
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

        Q_producer, Q_consumer = pipeline_Q.make_participants()
        K_producer, K_consumer = pipeline_K.make_participants()

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)

        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        tStSs = tuple(cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout) for stage in range(self.q_stage))

        warp_group_idx = tidx // 128
        # --- Warpgroup 0 (warps 0-3): Load + MMA + Sched + Empty ---
        if warp_group_idx == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_load)

            # Load warp (warp 0): CLC persistent loop
            if warp_idx == self.load_warp_id:
                warp_threads_load = 32
                rows_per_thread_load = cute.ceil_div(self.m_block_size, warp_threads_load)
                load_tile_count = Int32(0)
                while work_tile.is_valid_tile:
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    num_m_blocks_cur = cute.ceil_div(
                        seqlen.seqlen_q * self.qhead_per_kvhead,
                        self.q_stage * self.m_block_size,
                    )
                    is_valid_m_block = work_tile.tile_idx[0] < num_m_blocks_cur
                    if is_valid_m_block:
                        m_block = num_m_blocks_cur - 1 - work_tile.tile_idx[0]
                        num_n_blocks = self._causal_num_n_blocks(m_block, seqlen.seqlen_k, seqlen.seqlen_q)
                        # W load (double-buffered: even tiles → first half, odd tiles → second half)
                        sW_buf_off = (load_tile_count % 2) * self.q_stage * self.m_block_size
                        mW_cur = seqlen.offset_batch_Q(mW, batch_idx, dim=2)
                        for qs in cutlass.range_constexpr(self.q_stage):
                            for ri in cutlass.range_constexpr(rows_per_thread_load):
                                row = ri * warp_threads_load + (tidx % warp_threads_load)
                                if row < self.m_block_size:
                                    m_packed_idx = (self.q_stage * m_block + qs) * self.m_block_size + row
                                    m_idx = m_packed_idx // self.qhead_per_kvhead
                                    h_idx = m_packed_idx - m_idx * self.qhead_per_kvhead
                                    sW_idx = sW_buf_off + qs * self.m_block_size + row
                                    if m_idx < seqlen.seqlen_q:
                                        sW[sW_idx] = mW_cur[m_idx, h_idx]
                                    else:
                                        sW[sW_idx] = self.w_dtype(0)
                        cute.arch.fence_view_async_shared()
                        # TMA Q (make_participants API — reset count, preserve index/phase)
                        Q_producer.reset()
                        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, 0]
                        for qs in cutlass.range_constexpr(self.q_stage):
                            handle_Q = Q_producer.acquire_and_advance()
                            gQ = cute.local_tile(
                                mQ_cur,
                                (self.m_block_size, self.head_dim_padded),
                                (self.q_stage * m_block + qs, 0),
                            )
                            sQ_stage = sQ[None, None, None, qs]
                            load_Q_fn, _, _ = copy_ops.tma_get_copy_fn(
                                tma_atom_Q,
                                0,
                                cute.make_layout(1),
                                gQ,
                                sQ_stage,
                                single_stage=True,
                            )
                            load_Q_fn(tma_bar_ptr=handle_Q.barrier)
                        # TMA K (same inline pattern)
                        K_producer.reset()
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, 0]
                        peek_K = K_producer.try_acquire()
                        local_count = num_n_blocks if num_n_blocks < Int32(3) else Int32(3)
                        for iter_idx in cutlass.range(num_n_blocks, unroll=1):
                            n_block = iter_idx - local_count
                            if iter_idx < local_count:
                                n_block = num_n_blocks - Int32(1) - iter_idx
                            handle_K = K_producer.acquire_and_advance(peek_K)
                            gK = cute.local_tile(
                                mK_cur,
                                (self.n_block_size, self.head_dim_padded),
                                (n_block, 0),
                            )
                            sK_stage = sK[None, None, None, handle_K.index]
                            load_K_fn, _, _ = copy_ops.tma_get_copy_fn(
                                tma_atom_K,
                                0,
                                cute.make_layout(1),
                                gK,
                                sK_stage,
                                single_stage=True,
                            )
                            load_K_fn(tma_bar_ptr=handle_K.barrier)
                            peek_K = cutlass.Boolean(1)
                            if handle_K.count + 1 < num_n_blocks:
                                peek_K = K_producer.try_acquire()
                        cute.arch.fence_view_async_shared()
                        load_tile_count = load_tile_count + 1
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                Q_producer.tail()
                K_producer.tail()

            # MMA warp (warp 1): CLC persistent loop
            if warp_idx == self.mma_warp_id:
                tmem_alloc_cols = Int32(self.tmem_alloc_cols)
                cute.arch.alloc_tmem(tmem_alloc_cols, tmem_holding_buf)
                cute.arch.sync_warp()

                mma_s_empty_phase_0 = Int32(0)
                mma_s_empty_phase_1 = Int32(0)
                mma_tile_count = Int32(0)
                while work_tile.is_valid_tile:
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    num_m_blocks_cur = cute.ceil_div(
                        seqlen.seqlen_q * self.qhead_per_kvhead,
                        self.q_stage * self.m_block_size,
                    )
                    is_valid_m_block = work_tile.tile_idx[0] < num_m_blocks_cur
                    if is_valid_m_block:
                        m_block = num_m_blocks_cur - 1 - work_tile.tile_idx[0]
                        num_n_blocks = self._causal_num_n_blocks(m_block, seqlen.seqlen_k, seqlen.seqlen_q)
                        Q_consumer.reset()
                        handle_Q0 = Q_consumer.wait_and_advance()
                        handle_Q1 = Q_consumer.wait_and_advance()

                        tSrK = tiled_mma_qk.make_fragment_A(sK)
                        tSrQ = tiled_mma_qk.make_fragment_B(sQ)
                        qk_mma_op = tiled_mma_qk.op
                        tSrQs = tuple(tSrQ[None, None, None, stage] for stage in range(self.q_stage))
                        sQs = tuple(sQ[None, None, None, stage] for stage in range(self.q_stage))

                        K_consumer.reset()
                        peek_K_full = K_consumer.try_wait()
                        for iter_idx in cutlass.range(num_n_blocks, unroll=1):
                            handle_K = K_consumer.wait_and_advance(peek_K_full)
                            kv_stage = handle_K.index

                            if iter_idx > 0:
                                cute.arch.mbarrier_wait(S_mbar_ptr + self.q_stage + 0, mma_s_empty_phase_0)
                                mma_s_empty_phase_0 ^= 1
                            if iter_idx > 0:
                                cute.arch.mbarrier_wait(S_mbar_ptr + self.q_stage + 1, mma_s_empty_phase_1)
                                mma_s_empty_phase_1 ^= 1

                            tSrKi = tSrK[None, None, None, kv_stage]
                            sK_cur = sK[None, None, None, kv_stage]
                            _gemm_ptx_partial(
                                qk_mma_op,
                                self.tmem_s_offset[0],
                                tSrKi,
                                tSrQs[0],
                                sA=sK_cur,
                                sB=sQs[0],
                                zero_init=True,
                            )
                            with cute.arch.elect_one():
                                tcgen05.commit(S_mbar_ptr + 0)

                            _gemm_ptx_partial(
                                qk_mma_op,
                                self.tmem_s_offset[1],
                                tSrKi,
                                tSrQs[1],
                                sA=sK_cur,
                                sB=sQs[1],
                                zero_init=True,
                            )
                            with cute.arch.elect_one():
                                tcgen05.commit(S_mbar_ptr + 1)

                            handle_K.release()
                            peek_K_full = cutlass.Boolean(1)
                            if handle_K.count + 1 < num_n_blocks:
                                peek_K_full = K_consumer.try_wait()

                        if num_n_blocks > 0:
                            cute.arch.mbarrier_wait(S_mbar_ptr + self.q_stage + 0, mma_s_empty_phase_0)
                            mma_s_empty_phase_0 ^= 1
                            cute.arch.mbarrier_wait(S_mbar_ptr + self.q_stage + 1, mma_s_empty_phase_1)
                            mma_s_empty_phase_1 ^= 1

                        handle_Q0.release()
                        handle_Q1.release()
                        mma_tile_count = mma_tile_count + 1
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

            # Sched warp (warp 2): CLC producer loop
            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                clc_producer_state = make_pipeline_state(PipelineUserType.ProducerConsumer, self.num_clc_stage)
                sched_tile_count = Int32(0)
                while work_tile.is_valid_tile:
                    clc_pipeline.producer_acquire(clc_producer_state)
                    mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                    tile_sched.advance_to_next_work(mbarrier_addr)
                    clc_producer_state.advance()
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    sched_tile_count = sched_tile_count + 1
                    clc_consumer_state.advance()
                clc_pipeline.producer_tail(clc_producer_state)

            # TMA Score store (warp 3): CLC persistent loop
            if warp_idx == self.tma_store_warp_id:
                score_full_phase_0 = Int32(0)
                score_full_phase_1 = Int32(0)
                score_tile = (self.q_tokens_per_tile, self.n_block_size)
                while work_tile.is_valid_tile:
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    num_m_blocks_cur = cute.ceil_div(
                        seqlen.seqlen_q * self.qhead_per_kvhead,
                        self.q_stage * self.m_block_size,
                    )
                    is_valid_m_block = work_tile.tile_idx[0] < num_m_blocks_cur
                    if is_valid_m_block:
                        m_block = num_m_blocks_cur - 1 - work_tile.tile_idx[0]
                        num_n_blocks = self._causal_num_n_blocks(m_block, seqlen.seqlen_k, seqlen.seqlen_q)
                        local_count = num_n_blocks if num_n_blocks < Int32(3) else Int32(3)

                        if const_expr(is_varlen):
                            # Aligned tile = entire (q_stage * q_tokens_per_tile)
                            # range stays within seqlen_q_b → safe to TMA-bulk.
                            # Misaligned (last m_block of a batch) falls back
                            # to per-q-token scalar store with OOB guard.
                            q_token_end_tile = self.q_stage * (m_block + 1) * self.q_tokens_per_tile
                            both_aligned = q_token_end_tile <= seqlen.seqlen_q
                            mOut_thd = cute.domain_offset((seqlen.offset_q, Int32(0)), mOut)
                            if both_aligned:
                                # Fast path: TMA bulk store, mirrors BSHD path.
                                for iter_idx in cutlass.range(num_n_blocks, unroll=1):
                                    _n_block = iter_idx - local_count
                                    if iter_idx < local_count:
                                        _n_block = num_n_blocks - Int32(1) - iter_idx
                                    cute.arch.mbarrier_wait(Score_store_mbar_ptr + 0, score_full_phase_0)
                                    score_full_phase_0 = score_full_phase_0 ^ 1
                                    gScore_0 = cute.local_tile(
                                        mOut_thd,
                                        score_tile,
                                        (self.q_stage * m_block + 0, _n_block),
                                    )
                                    sScore_0 = sScore[None, None, 0]
                                    store_fn_0, _, _ = copy_ops.tma_get_copy_fn(
                                        tma_atom_Score,
                                        0,
                                        cute.make_layout(1),
                                        sScore_0,
                                        gScore_0,
                                        single_stage=True,
                                    )
                                    store_fn_0()
                                    cute.arch.cp_async_bulk_commit_group()

                                    cute.arch.mbarrier_wait(Score_store_mbar_ptr + 1, score_full_phase_1)
                                    score_full_phase_1 = score_full_phase_1 ^ 1
                                    gScore_1 = cute.local_tile(
                                        mOut_thd,
                                        score_tile,
                                        (self.q_stage * m_block + 1, _n_block),
                                    )
                                    sScore_1 = sScore[None, None, 1]
                                    store_fn_1, _, _ = copy_ops.tma_get_copy_fn(
                                        tma_atom_Score,
                                        0,
                                        cute.make_layout(1),
                                        sScore_1,
                                        gScore_1,
                                        single_stage=True,
                                    )
                                    store_fn_1()
                                    cute.arch.cp_async_bulk_commit_group()

                                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                                    cute.arch.mbarrier_arrive(Score_store_mbar_ptr + self.q_stage + 0)
                                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                                    cute.arch.mbarrier_arrive(Score_store_mbar_ptr + self.q_stage + 1)
                            else:
                                lane_id = tidx % cute.arch.WARP_SIZE
                                for iter_idx in cutlass.range(num_n_blocks, unroll=1):
                                    _n_block = iter_idx - local_count
                                    if iter_idx < local_count:
                                        _n_block = num_n_blocks - Int32(1) - iter_idx
                                    cute.arch.mbarrier_wait(Score_store_mbar_ptr + 0, score_full_phase_0)
                                    score_full_phase_0 = score_full_phase_0 ^ 1
                                    sScore_0 = sScore[None, None, 0]
                                    for idx in cutlass.range(
                                        lane_id,
                                        self.q_tokens_per_tile * self.n_block_size,
                                        cute.arch.WARP_SIZE,
                                        unroll=1,
                                    ):
                                        qi = idx // self.n_block_size
                                        kj = idx - qi * self.n_block_size
                                        q_local = (self.q_stage * m_block + 0) * self.q_tokens_per_tile + qi
                                        k_local = _n_block * self.n_block_size + kj
                                        if q_local < seqlen.seqlen_q and k_local < cute.size(mOut_scalar.shape[1]):
                                            mOut_scalar[seqlen.offset_q + q_local, k_local] = sScore_0[qi, kj]
                                    cute.arch.sync_warp()
                                    cute.arch.mbarrier_arrive(Score_store_mbar_ptr + self.q_stage + 0)

                                    cute.arch.mbarrier_wait(Score_store_mbar_ptr + 1, score_full_phase_1)
                                    score_full_phase_1 = score_full_phase_1 ^ 1
                                    sScore_1 = sScore[None, None, 1]
                                    for idx in cutlass.range(
                                        lane_id,
                                        self.q_tokens_per_tile * self.n_block_size,
                                        cute.arch.WARP_SIZE,
                                        unroll=1,
                                    ):
                                        qi = idx // self.n_block_size
                                        kj = idx - qi * self.n_block_size
                                        q_local = (self.q_stage * m_block + 1) * self.q_tokens_per_tile + qi
                                        k_local = _n_block * self.n_block_size + kj
                                        if q_local < seqlen.seqlen_q and k_local < cute.size(mOut_scalar.shape[1]):
                                            mOut_scalar[seqlen.offset_q + q_local, k_local] = sScore_1[qi, kj]
                                    cute.arch.sync_warp()
                                    cute.arch.mbarrier_arrive(Score_store_mbar_ptr + self.q_stage + 1)
                        else:
                            mOut_cur = mOut[None, None, batch_idx]
                            for iter_idx in cutlass.range(num_n_blocks, unroll=1):
                                _n_block = iter_idx - local_count
                                if iter_idx < local_count:
                                    _n_block = num_n_blocks - Int32(1) - iter_idx
                                # Wait for both stages, issue TMA store for each
                                cute.arch.mbarrier_wait(Score_store_mbar_ptr + 0, score_full_phase_0)
                                score_full_phase_0 = score_full_phase_0 ^ 1
                                gScore_0 = cute.local_tile(
                                    mOut_cur,
                                    score_tile,
                                    (self.q_stage * m_block + 0, _n_block),
                                )
                                sScore_0 = sScore[None, None, 0]
                                store_fn_0, _, _ = copy_ops.tma_get_copy_fn(
                                    tma_atom_Score,
                                    0,
                                    cute.make_layout(1),
                                    sScore_0,
                                    gScore_0,
                                    single_stage=True,
                                )
                                store_fn_0()
                                cute.arch.cp_async_bulk_commit_group()

                                cute.arch.mbarrier_wait(Score_store_mbar_ptr + 1, score_full_phase_1)
                                score_full_phase_1 = score_full_phase_1 ^ 1
                                gScore_1 = cute.local_tile(
                                    mOut_cur,
                                    score_tile,
                                    (self.q_stage * m_block + 1, _n_block),
                                )
                                sScore_1 = sScore[None, None, 1]
                                store_fn_1, _, _ = copy_ops.tma_get_copy_fn(
                                    tma_atom_Score,
                                    0,
                                    cute.make_layout(1),
                                    sScore_1,
                                    gScore_1,
                                    single_stage=True,
                                )
                                store_fn_1()
                                cute.arch.cp_async_bulk_commit_group()

                                # Wait for stores to complete, then signal smem empty
                                cute.arch.cp_async_bulk_wait_group(1, read=True)
                                cute.arch.mbarrier_arrive(Score_store_mbar_ptr + self.q_stage + 0)
                                cute.arch.cp_async_bulk_wait_group(0, read=True)
                                cute.arch.mbarrier_arrive(Score_store_mbar_ptr + self.q_stage + 1)

                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

        # --- Warpgroup 1 (warps 4-7): Epilogue q_stage=0, CLC persistent loop ---
        if warp_group_idx == 1:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue)
            epi_s_full_phase_0 = Int32(0)
            score_empty_phase_0 = Int32(0)
            epi0_tile_count = Int32(0)
            sScore_0_layout_1d = cute.make_layout((self.q_tokens_per_tile * self.n_block_size,), stride=(1,))
            sScore_0 = cute.make_tensor(sScore[None, None, 0].iterator, sScore_0_layout_1d)
            while work_tile.is_valid_tile:
                batch_idx = work_tile.tile_idx[2]
                seqlen = SeqlenInfoCls(batch_idx)
                num_m_blocks_cur = cute.ceil_div(
                    seqlen.seqlen_q * self.qhead_per_kvhead,
                    self.q_stage * self.m_block_size,
                )
                is_valid_m_block = work_tile.tile_idx[0] < num_m_blocks_cur
                if is_valid_m_block:
                    m_block = num_m_blocks_cur - 1 - work_tile.tile_idx[0]
                    num_n_blocks = self._causal_num_n_blocks(m_block, seqlen.seqlen_k, seqlen.seqlen_q)
                    epi0_sW_base = (epi0_tile_count % 2) * self.q_stage * self.m_block_size
                    epi_s_full_phase_0, score_empty_phase_0 = self._epilogue_warp(
                        0,
                        tiled_mma_qk,
                        tStSs[0],
                        sW,
                        sScore_0,
                        S_mbar_ptr,
                        Score_store_mbar_ptr,
                        m_block,
                        batch_idx,
                        num_n_blocks,
                        seqlen.seqlen_k,
                        seqlen.seqlen_q,
                        tidx,
                        epi_s_full_phase_0,
                        score_empty_phase_0,
                        sm_scale,
                        sW_base=epi0_sW_base,
                    )
                    epi0_tile_count = epi0_tile_count + 1
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            if warp_idx == self.epilogue_wg0_warp_ids[-1]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # --- Warpgroup 2 (warps 8-11): Epilogue q_stage=1, CLC persistent loop ---
        if warp_group_idx == 2:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue)
            epi_s_full_phase_1 = Int32(0)
            score_empty_phase_1 = Int32(0)
            epi1_tile_count = Int32(0)
            sScore_1_layout_1d = cute.make_layout((self.q_tokens_per_tile * self.n_block_size,), stride=(1,))
            sScore_1 = cute.make_tensor(sScore[None, None, 1].iterator, sScore_1_layout_1d)
            while work_tile.is_valid_tile:
                batch_idx = work_tile.tile_idx[2]
                seqlen = SeqlenInfoCls(batch_idx)
                num_m_blocks_cur = cute.ceil_div(
                    seqlen.seqlen_q * self.qhead_per_kvhead,
                    self.q_stage * self.m_block_size,
                )
                is_valid_m_block = work_tile.tile_idx[0] < num_m_blocks_cur
                if is_valid_m_block:
                    m_block = num_m_blocks_cur - 1 - work_tile.tile_idx[0]
                    num_n_blocks = self._causal_num_n_blocks(m_block, seqlen.seqlen_k, seqlen.seqlen_q)
                    epi1_sW_base = (epi1_tile_count % 2) * self.q_stage * self.m_block_size
                    epi_s_full_phase_1, score_empty_phase_1 = self._epilogue_warp(
                        1,
                        tiled_mma_qk,
                        tStSs[1],
                        sW,
                        sScore_1,
                        S_mbar_ptr,
                        Score_store_mbar_ptr,
                        m_block,
                        batch_idx,
                        num_n_blocks,
                        seqlen.seqlen_k,
                        seqlen.seqlen_q,
                        tidx,
                        epi_s_full_phase_1,
                        score_empty_phase_1,
                        sm_scale,
                        sW_base=epi1_sW_base,
                    )
                    epi1_tile_count = epi1_tile_count + 1
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            if warp_idx == self.epilogue_wg1_warp_ids[-1]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    # =========================================================================
    # Causal n-block computation
    # =========================================================================
    @cute.jit
    def _causal_num_n_blocks(self, m_block, seqlen_k, seqlen_q):
        """Compute causal-aware number of KV n-blocks for a given m_block.

        Under bottom-right ratio causal masking, kv_token >=
        (q_global_start + q_token + 1) // ratio is masked, where
        q_global_start = seqlen_k * ratio - seqlen_q.
        The largest q_token in this CTA tile determines the upper bound on kv_token,
        and therefore how many n-blocks are actually needed.
        """
        q_global_start = seqlen_k * self.ratio - seqlen_q
        max_q_plus_1 = self.q_stage * (m_block + 1) * self.q_tokens_per_tile
        max_kv_needed = (q_global_start + max_q_plus_1) // self.ratio
        max_kv_needed = max_kv_needed if max_kv_needed < seqlen_k else seqlen_k
        num_n_blocks = cute.ceil_div(max_kv_needed, self.n_block_size)
        return num_n_blocks

    # =========================================================================
    # Epilogue Warpgroup: TMEM→RF, ReLU, ×W, per-element accumulate to output
    # =========================================================================
    @cute.jit
    def _epilogue_warp(
        self,
        q_stage_idx,
        tiled_mma_qk,
        tStS_stage,
        sW,
        sScore_stage,
        S_mbar_ptr,
        Score_store_mbar_ptr,
        m_block,
        batch_idx,
        num_n_blocks,
        seqlen_k,
        seqlen_q,
        tidx,
        epi_s_full_phase,
        score_empty_phase,
        sm_scale,
        sW_base=None,
    ):
        """Epilogue: TMEM->RF, ReLU, W from SMEM (sW), write Score to smem for TMA store."""
        tidx_wg = tidx % (cute.arch.WARP_SIZE * 4)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_stage).get_slice(tidx_wg)
        tStS_t2r = thr_tmem_load.partition_S(tStS_stage)

        # Coordinate tensor: MMA partition first, then remap to TMEM copy's
        # DESTINATION layout so tScS[i] matches tSrS[i] after TMEM→RF copy.
        thr_mma = tiled_mma_qk.get_slice(tidx_wg)
        cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
        tScS_mma = thr_mma.partition_C(cS)
        tScS = thr_tmem_load.partition_D(tScS_mma)

        # RF fragment for TMEM→RF copy (destination partition)
        tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_stage.shape)).shape
        tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

        _sW_base = Int32(0) if sW_base is None else sW_base
        qhpkv = self.qhead_per_kvhead
        ratio = Int32(self.ratio)
        q_global_start = seqlen_k * ratio - seqlen_q

        rW_ILP = 4
        sW_f32_ptr = cute.make_ptr(
            Float32,
            (sW.iterator + (_sW_base + q_stage_idx * self.m_block_size)).llvm_ptr,
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        sW_1d_f32 = cute.make_tensor(
            sW_f32_ptr,
            cute.make_layout((self.m_block_size // 2,)),
        )
        tSsW_f32_tiled = cute.logical_divide(
            sW_1d_f32,
            cute.make_layout((rW_ILP,)),
        )
        rW_f32 = cute.make_rmem_tensor((self.m_block_size // 2,), Float32)
        tRsW_f32_tiled = cute.logical_divide(
            rW_f32,
            cute.make_layout((rW_ILP,)),
        )
        rW_buf = cute.make_rmem_tensor((2 * rW_ILP,), self.w_dtype)
        rW_buf_f32 = cute.recast_tensor(rW_buf, Float32)

        local_count = num_n_blocks if num_n_blocks < Int32(3) else Int32(3)
        for iter_idx in cutlass.range(num_n_blocks, unroll=1):
            _n_block = iter_idx - local_count
            if iter_idx < local_count:
                _n_block = num_n_blocks - Int32(1) - iter_idx

            cute.arch.mbarrier_wait(S_mbar_ptr + q_stage_idx, epi_s_full_phase)
            epi_s_full_phase = epi_s_full_phase ^ 1

            if iter_idx == 0:
                for wi in cutlass.range_constexpr((self.m_block_size // 2) // rW_ILP):
                    cute.autovec_copy(tSsW_f32_tiled[None, wi], tRsW_f32_tiled[None, wi])

            cute.copy(thr_tmem_load, tStS_t2r, tSrS)
            cute.arch.fence_view_async_tmem_load()

            cute.arch.mbarrier_arrive(S_mbar_ptr + self.q_stage + q_stage_idx)

            local_sum = [(Float32(0.0), Float32(0.0)) for _ in range(self.q_tokens_per_tile)]

            for qi in cutlass.range_constexpr(self.q_tokens_per_tile):
                for ho in cutlass.range_constexpr(qhpkv // 2 // rW_ILP):
                    cute.autovec_copy(
                        tRsW_f32_tiled[None, qi * (qhpkv // 2 // rW_ILP) + ho],
                        rW_buf_f32,
                    )

                    for ci in cutlass.range_constexpr(rW_ILP):
                        w_pair = (Float32(rW_buf[2 * ci]), Float32(rW_buf[2 * ci + 1]))
                        idx0 = qi * qhpkv + (ho * rW_ILP + ci) * 2
                        idx1 = idx0 + 1

                        val0 = tSrS[idx0]
                        val0 = val0 if val0 > Float32(0.0) else Float32(0.0)  # relu
                        val1 = tSrS[idx1]
                        val1 = val1 if val1 > Float32(0.0) else Float32(0.0)

                        prod = mul_packed_f32x2((val0, val1), w_pair)
                        local_sum[qi] = add_packed_f32x2(local_sum[qi], prod)

            # Apply sm_scale onto the fp32 head-reduced score (post-reduce,
            # pre causal mask). Preserves precision vs pre-multiplying onto
            # bf16 W on the host.
            rAcc = cute.make_rmem_tensor((self.q_tokens_per_tile,), Float32)
            for qi in cutlass.range_constexpr(self.q_tokens_per_tile):
                rAcc[qi] = (local_sum[qi][0] + local_sum[qi][1]) * Float32(sm_scale)

            # write Score to smem for TMA store warp
            if iter_idx > 0:
                cute.arch.mbarrier_wait(
                    Score_store_mbar_ptr + self.q_stage + q_stage_idx,
                    score_empty_phase,
                )
                score_empty_phase = score_empty_phase ^ 1

            kv_offset = tScS[0][0]
            q_token_base = (self.q_stage * m_block + q_stage_idx) * self.q_tokens_per_tile

            # Causal mask only on the rightmost local_count blocks (the
            # right-to-left iters). Every other iter's n_block is guaranteed
            # by num_n_blocks definition + col_limit_max - col_limit_min ≤
            # (q_tokens_per_tile - 1) / ratio to lie strictly inside the
            # valid kv range — no per-element check needed there.
            if iter_idx < local_count:
                kv_token = kv_offset + _n_block * self.n_block_size
                for qi in cutlass.range(self.q_tokens_per_tile, unroll_full=True):
                    q_token = q_token_base + qi
                    val = -Float32.inf
                    if q_token < seqlen_q and kv_token < seqlen_k:
                        col_limit = (q_global_start + q_token + 1) // ratio
                        if kv_token < col_limit:
                            val = rAcc[qi]
                    rAcc[qi] = val

            sScore_dst_ptr = cute.make_ptr(
                Float32,
                (sScore_stage.iterator + kv_offset).llvm_ptr,
                cute.AddressSpace.smem,
            )
            sScore_dst = cute.make_tensor(
                sScore_dst_ptr,
                cute.make_layout(
                    (self.q_tokens_per_tile,),
                    stride=(self.n_block_size,),
                ),
            )
            cute.autovec_copy(rAcc, sScore_dst)

            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(Score_store_mbar_ptr + q_stage_idx)

        # Trailing Score_empty wait: consume the last TMA store arrival so
        # phase stays in sync across tiles (mirrors MMA's trailing S_empty wait).
        if num_n_blocks > 0:
            cute.arch.mbarrier_wait(
                Score_store_mbar_ptr + self.q_stage + q_stage_idx,
                score_empty_phase,
            )
            score_empty_phase = score_empty_phase ^ 1

        return epi_s_full_phase, score_empty_phase
