# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 CuTeDSL unified indexer-score kernel (BF16).

One kernel serves the indexer forward (dense scores or the compact ratio-causal candidate buffer) and
the dense indexer score recompute (scores + LSE): ``compute_lse`` / ``is_compressed_logits`` are
compile-time keys.  One CTA owns one packed m tile of 128 (token, head) rows -- for every head group,
so a small group packs ``128 // H`` tokens and reuses each K block across them -- and walks its whole
ratio-causal K range under CLC persistent scheduling.  SwapAB: K is the MMA's A operand (128 keys),
packed Q the B operand, so a block's scores land in TMEM as 128 lanes (keys) x 128 columns.

* SMEM (``smem_plan``): Q is double buffered when it fits, so the next work tile's Q lands while
  this tile computes.  The K ring holds ``kv_group`` (128 x k_block) chunks per stage and barrier: a
  whole n block when two blocks fit next to Q, else one chunk (large head dims).  Paged K streams each
  chunk as page-sized row boxes.
* TMEM: ``NUM_ACC`` accumulators of 128 columns; accumulator ``j`` holds block ``r * NUM_ACC + j`` of
  the tile, guarded by its own full (MMA commit) / empty (every math thread) barrier.
* math warpgroups (2 for H64, else 4) split the 128 columns: warpgroup ``t`` owns ``cols / H`` whole
  tokens of every block.  TMEM is read in ``CHUNK``-column chunks, the next chunk (across block
  boundaries too) loading while the current one is reduced; ptxas sizes this region against the launch
  register budget (168 at 384 threads, 96 at 640), so W plus two chunks must fit it.  The reduce is
  unconditional, only the store is predicated.  ``relu(s) * w`` is ``(s + |s|) * w / 2`` (FADD2 with
  an |abs| operand) for every pair below H64 and for odd pairs at H64.
* W (per-head weights) is handed off through a 2-stage ``PipelineAsync``: all load lanes publish,
  every math lane releases after copying.

:class:`IndexerScoreSm100Base` carries the indexer-score contract (compact output, LSE flag, compact
row offsets) that the block-scaled kernels share.
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
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
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator

import cuda.bindings.driver as cuda
from functools import partial

from cudnn.deepseek_sparse_attention.utils import copy as copy_utils
from cudnn.deepseek_sparse_attention.utils.sm100.gemm import gemm_ptx_partial as _gemm_ptx_partial
from cudnn.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK
from .dense_score_recompute_sm100 import DenseScoreRecomputeSm100, add_packed_f32x2, fma_packed_f32x2


@dsl_user_op
def _tcgen05_fence_after(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _tcgen05_fence_before(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


_SMEM_BYTES = 232448  # SM100 dynamic SMEM per CTA
_SMEM_RESERVE = 4096  # barriers, W ring, LSE scratch, alignment
_MAX_KV_CHUNKS = 8  # K ring cap in (128 x k_block) chunks (raised to two n blocks for large head dims)


def smem_plan(head_dim: int, k_block_size: int | None = None, *, tile: int = 128, elem_bytes: int = 2):
    """``(q_stage, kv_group, num_kv_stages)`` of the BF16 kernel's SMEM, or ``None`` if ``head_dim`` does not fit.

    ``kv_group`` is the (tile x k_block) chunks one K ring stage (and barrier) holds: the whole n block when
    two blocks fit next to the Q ring, else one chunk.  The ring needs at least two stages.
    """
    hd_padded = (head_dim + 15) // 16 * 16
    if k_block_size is None:
        k_block_size = 64 if hd_padded % 64 == 0 else hd_padded
    if k_block_size <= 0 or hd_padded % k_block_size != 0:
        return None
    nkc = hd_padded // k_block_size
    q_bytes = tile * hd_padded * elem_bytes
    chunk_bytes = tile * k_block_size * elem_bytes
    budget = _SMEM_BYTES - _SMEM_RESERVE
    q_stage = 2 if 2 * q_bytes + 2 * nkc * chunk_bytes <= budget else 1
    kv_group = nkc if q_stage * q_bytes + 2 * nkc * chunk_bytes <= budget else 1
    num_kv_stages = min(max(_MAX_KV_CHUNKS, 2 * nkc) // kv_group, (budget - q_stage * q_bytes) // (kv_group * chunk_bytes))
    if num_kv_stages < 2:
        return None
    return q_stage, kv_group, num_kv_stages


class IndexerScoreSm100Base(DenseScoreRecomputeSm100):
    """Indexer-score contract shared by the SM100 indexer kernels (BF16 here, block-scaled subclasses).

    The packed M tile holds ``q_tokens_per_tile = m_block_size // qhead_per_kvhead`` query tokens; every
    token's ReLU(QK) * W head reduce writes one score per key, densely or into the compact ratio-causal
    candidate buffer (``is_compressed_logits``), with an optional per-token LSE (``compute_lse``).
    """

    max_q_tokens_per_tile = 8

    def __init__(
        self,
        *args,
        compute_lse: bool = False,
        is_compressed_logits: bool = False,
        cand_2d: bool = False,
        **kwargs,
    ):
        # Reuse the dense-score base mainloop, but force its epilogue contract
        # to dense indexer score.
        kwargs["score_type"] = "indexer"
        super().__init__(*args, **kwargs)
        # Dense FE entry points remain the default. The compressed forward path
        # opts in and writes only ratio-causal candidates into a compact buffer.
        self.is_compressed_logits = is_compressed_logits
        # BSHD BF16 may expose that compact buffer as (batch, per_batch_floats)
        # so the hot epilogue carries an Int32 column offset instead of a flat
        # Int64 address. Varlen and non-uniform batches retain the flat path.
        self.cand_2d = cand_2d
        # LSE is independent of output layout: dense recompute requests it, and
        # compressed BF16 can request the per-row LSE.
        self.compute_lse = compute_lse

    @cute.jit
    def _g_prefix_i64(self, n):
        """Prefix sum of ratio-causal candidate counts through row ``n``."""
        ratio = Int64(self.ratio)
        quotient = n // ratio
        remainder = n - quotient * ratio
        return ratio * quotient * (quotient - Int64(1)) // Int64(2) + quotient * (remainder + Int64(1))

    @cute.jit
    def _row_offset_i64(self, row, q_global_start):
        return self._g_prefix_i64(q_global_start + row) - self._g_prefix_i64(q_global_start)


class IndexerScoreUnifiedSm100(IndexerScoreSm100Base):
    """BF16 dense / compact indexer score with optional LSE (128-row tile, multi-warpgroup epilogue)."""

    max_q_tokens_per_tile = 32
    NUM_ACC = 4  # TMEM accumulators (128 columns each)
    NUM_MATH_WG = 0  # 0: 2 for 2-token tiles (H64), else 4
    CHUNK = 16  # TMEM load chunk width in columns (16 beat 32 / 64 at H64 in every mode; 64 spills)
    NUM_W_STAGES = 2  # W ring depth

    def __init__(
        self,
        head_dim: int,
        qhead_per_kvhead: int = 64,
        m_block_size: int = 128,
        n_block_size: int = 128,
        k_block_size: int | None = None,
        ratio: int = 1,
        is_varlen: bool = False,
        compute_lse: bool = False,
        is_compressed_logits: bool = False,
        cand_2d: bool = False,
        page_size: int = 0,
    ):
        if m_block_size != 128 or n_block_size != 128:
            raise ValueError(f"SM100 BF16 indexer requires m_block_size=n_block_size=128, got {m_block_size} / {n_block_size}")
        if qhead_per_kvhead not in (4, 8, 16, 32, 64):
            raise ValueError(f"SM100 BF16 indexer requires qhead_per_kvhead in (4, 8, 16, 32, 64), got {qhead_per_kvhead}")
        hd_padded = (head_dim + 15) // 16 * 16
        if k_block_size is None:
            k_block_size = 64 if hd_padded % 64 == 0 else hd_padded
        plan = smem_plan(head_dim, k_block_size)
        if plan is None:
            raise ValueError(f"SM100 BF16 indexer: head_dim={head_dim} (k_block_size={k_block_size}) does not fit the SMEM plan")
        q_stage, self.kv_group, self.num_kv_stages = plan
        super().__init__(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            kv_stage=self.num_kv_stages * self.kv_group,  # the SMEM plan's K ring, in (128 x k_block) chunks
            k_block_size=k_block_size,
            ratio=ratio,
            is_varlen=is_varlen,
            compute_lse=compute_lse,
            is_compressed_logits=is_compressed_logits,
            cand_2d=cand_2d,
            page_size=page_size,
        )
        self.q_stage = q_stage

        # Warps: 0 load, 1 MMA, 2 CLC scheduler, 3 idle, then the math warpgroups.
        self.load_warp_id, self.mma_warp_id, self.sched_warp_id = 0, 1, 2
        self.num_math_wg = self.NUM_MATH_WG if self.NUM_MATH_WG else (2 if self.q_tokens_per_tile == 2 else 4)
        assert self.q_tokens_per_tile % self.num_math_wg == 0
        self.num_warps = 4 + 4 * self.num_math_wg
        self.threads_per_cta = self.WARP_SIZE * self.num_warps
        # setmaxnreg only redistributes the CTA's launch allocation (floor8(64K / threads) per thread).
        self.num_regs_load = 56
        launch_regs = (65536 // self.threads_per_cta) // 8 * 8
        self.num_regs_epilogue = min(
            256,
            ((launch_regs * self.threads_per_cta - self.WARPGROUP_SIZE * self.num_regs_load) // (self.WARPGROUP_SIZE * self.num_math_wg)) // 8 * 8,
        )
        self.relu_fadd_all = self.qhead_per_kvhead != 64
        # TMEM -> register chunk width: a thread holds W (cols) plus two chunks (load next / reduce current).
        # ptxas sizes the math region against the launch budget (168 regs at 384 threads, 96 at 640).
        self.math_cols = self.m_block_size // self.num_math_wg
        self.chunk_cols = self.CHUNK
        self.chunk_cols = min(self.chunk_cols, self.math_cols)
        assert self.math_cols % self.chunk_cols == 0

        self.tmem_s_stride = self.m_block_size
        self.tmem_alloc_cols = 512
        assert self.NUM_ACC * self.tmem_s_stride <= self.tmem_alloc_cols

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPerHead: cute.Tensor,
        mOut: cute.Tensor,
        mDenom: cute.Tensor | None,
        softmax_scale: Float32 | float,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mQCausalOffsets: cute.Tensor | None,
        stream: cuda.CUstream,
        mCandBatchOffsets: cute.Tensor | None = None,
        mBlockTable: cute.Tensor | None = None,
        mKvLens: cute.Tensor | None = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        is_varlen = mCuSeqlensQ is not None
        is_paged = mBlockTable is not None
        if const_expr(is_varlen):
            assert self.is_varlen
        else:
            assert not self.is_varlen
        if const_expr(is_paged):
            assert self.page_size > 0 and mKvLens is not None and mCuSeqlensK is None
        else:
            assert self.page_size == 0 and mKvLens is None

        # BSHD (bs, s, h, d) -> (s, d, h, bs); THD (t, h, d) -> (t, d, h); paged K pool (pages, page, h, d) -> (page, d, h, pages).
        Q_layout_transpose = [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0]
        K_layout_transpose = [0, 2, 1] if const_expr(is_varlen and not is_paged) else [1, 3, 2, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=K_layout_transpose))
        shape_Q_packed = ((self.qhead_per_kvhead, mQ.shape[0]), mQ.shape[1], 1, *mQ.shape[3:])
        stride_Q_packed = ((mQ.stride[2], mQ.stride[0]), mQ.stride[1], mQ.stride[2] * self.qhead_per_kvhead, *mQ.stride[3:])
        mQ = cute.make_tensor(mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed))

        cta_group = tcgen05.CtaGroup.ONE
        self.q_major_mode = LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = LayoutEnum.from_tensor(mK).mma_major_mode()
        tiled_mma_qk = _make_trivial_tiled_mma(
            self.q_dtype, self.q_dtype, self.k_major_mode, self.q_major_mode, self.qk_acc_dtype, cta_group, self.mma_tiler_qk[:2]
        )
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))

        sK_layout = _make_smem_layout_a(tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage)
        sQ_layout = _make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.num_k_chunks * self.q_stage)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mQ, cute.select(sQ_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk, self.cluster_layout_vmnk.shape
        )
        if const_expr(is_paged):
            # One box per page row range: a (box_rows, k_block) swizzled tile at a 1 KiB-aligned row offset of the
            # chunk, so stacked boxes equal the MMA's full tile.
            k_box_rows = min(self.page_size, self.n_block_size)
            k_box_layout = cute.make_composed_layout(sK_layout.inner, 0, cute.make_layout((k_box_rows, self.k_block_size), stride=(self.k_block_size, 1)))
            tma_atom_K, mK = cpasync.make_tiled_tma_atom(tma_load_op, mK, k_box_layout, (k_box_rows, self.k_block_size))
        else:
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op, mK, cute.select(sK_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk, self.cluster_layout_vmnk.shape
            )
        q_chunk_bytes = cute.size_in_bytes(self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2]))
        k_chunk_bytes = cute.size_in_bytes(self.k_dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        self.tma_copy_bytes = {"Q": self.num_k_chunks * q_chunk_bytes, "K": self.kv_group * k_chunk_bytes}

        PerHead_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
        mPerHead = cute.make_tensor(mPerHead.iterator, cute.select(mPerHead.layout, mode=PerHead_transpose))
        if const_expr(not self.is_compressed_logits):
            Out_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
            mOut = cute.make_tensor(mOut.iterator, cute.select(mOut.layout, mode=Out_transpose))
        if const_expr(mDenom is not None):
            Denom_transpose = [0] if const_expr(is_varlen) else [1, 0]
            mDenom = cute.make_tensor(mDenom.iterator, cute.select(mDenom.layout, mode=Denom_transpose))
        else:
            assert not self.compute_lse

        seqlen_q_static = max_seqlen_q if const_expr(is_varlen) else cute.size(mQ.shape[0]) // self.qhead_per_kvhead
        num_m_blocks = cute.ceil_div(seqlen_q_static * self.qhead_per_kvhead, self.m_block_size)
        batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1 if const_expr(is_varlen) else cute.size(mQ.shape[3])
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams((num_m_blocks, 1, batch_size), (*self.cluster_shape_mn, 1))
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
            mCandBatchOffsets,
            mBlockTable,
            mKvLens,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            # Launch bounds let ptxas size the entry register file; without them it ignores setmaxnreg
            # (C7508) and may squeeze the whole kernel into the load warpgroup's budget.
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ,
        mK,
        mPerHead,
        mOut,
        mDenom: cute.Tensor | None,
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
        mCandBatchOffsets: cute.Tensor | None = None,
        mBlockTable: cute.Tensor | None = None,
        mKvLens: cute.Tensor | None = None,
    ):
        is_varlen = mCuSeqlensQ is not None
        is_paged = mBlockTable is not None
        NKC = self.num_k_chunks
        G = self.kv_group
        NUM_ACC = self.NUM_ACC
        seqlen_q_static = max_seqlen_q if const_expr(is_varlen) else cute.size(mQ.shape[0]) // self.qhead_per_kvhead
        seqlen_k_static = max_seqlen_k if const_expr(is_varlen or is_paged) else cute.size(mK.shape[0])
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=seqlen_q_static,
            seqlen_k_static=seqlen_k_static,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedK=mKvLens,
            tile_m=self.m_block_size,
            tile_n=self.n_block_size,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)

        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)
        sPerHead_size = self.m_block_size * self.NUM_W_STAGES

        @cute.struct
        class SharedStorage:
            per_head_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.NUM_W_STAGES]
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.q_stage]
            KV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.num_kv_stages]
            T_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.NUM_ACC]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, 4]
            sPerHead: cute.struct.Align[cute.struct.MemRange[Float32, sPerHead_size], 128]
            sRed: cute.struct.Align[cute.struct.MemRange[Float32, 8 * self.num_math_wg], 128]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.k_dtype, sK_size], self.buffer_align_bytes]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        T_mbar_ptr = storage.T_mbar_ptr.data_ptr()  # [full_acc0..N-1, empty_acc0..N-1]
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf.ptr
        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sPerHead = storage.sPerHead.get_tensor(cute.make_layout((sPerHead_size,), stride=(1,)))

        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))
        pipeline_head = PipelineAsync.create(
            barrier_storage=storage.per_head_mbar_ptr.data_ptr(),
            num_stages=self.NUM_W_STAGES,
            producer_group=CooperativeGroup(Agent.Thread, self.WARP_SIZE),
            consumer_group=CooperativeGroup(Agent.Thread, self.WARPGROUP_SIZE * self.num_math_wg),
            defer_sync=True,
        )
        pipeline_Q = PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_KV = PipelineTmaUmma.create(
            barrier_storage=storage.KV_mbar_ptr.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        if warp_idx == 1:
            cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, self.num_math_wg)
        if warp_idx == 0:
            for _w in cutlass.range_constexpr(NUM_ACC):
                cute.arch.mbarrier_init(T_mbar_ptr + _w, 1)  # full: MMA commit
                cute.arch.mbarrier_init(T_mbar_ptr + NUM_ACC + _w, self.WARPGROUP_SIZE * self.num_math_wg)  # empty: every math thread
        num_clc_consumer_threads = self.WARP_SIZE * (3 + 4 * self.num_math_wg)
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
        head_producer, head_consumer = pipeline_head.make_participants()
        Q_producer, Q_consumer = pipeline_Q.make_participants()

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS_ref = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.WARPGROUP_SIZE)

        if warp_group_idx == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_load)

            # ---------------- load warp: W, Q, then K blocks left to right ----------------
            if warp_idx == self.load_warp_id:
                lane_id = tidx % self.WARP_SIZE
                KV_producer_state = make_pipeline_state(PipelineUserType.Producer, self.num_kv_stages)
                while work_tile.is_valid_tile:
                    m_block_sched = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                    num_m_blocks_cur = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.m_block_size)
                    if m_block_sched < num_m_blocks_cur:
                        m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                        num_n_blocks = self._dense_compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)
                        head = head_producer.acquire_and_advance()
                        per_head_buf_off = head.index * self.m_block_size
                        mPerHead_cur = seqlen.offset_batch_Q(mPerHead, batch_idx, dim=2)
                        for ri in cutlass.range_constexpr(self.m_block_size // self.WARP_SIZE):
                            row = ri * self.WARP_SIZE + lane_id
                            m_packed_idx = m_block * self.m_block_size + row
                            m_idx = m_packed_idx // self.qhead_per_kvhead
                            h_idx = m_packed_idx - m_idx * self.qhead_per_kvhead
                            if m_idx < seqlen.seqlen_q:
                                sPerHead[per_head_buf_off + row] = Float32(mPerHead_cur[m_idx, h_idx])
                            else:
                                sPerHead[per_head_buf_off + row] = Float32(0.0)
                        head.commit()
                        cute.arch.fence_view_async_shared()

                        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, 0]
                        handle_Q = Q_producer.acquire_and_advance()
                        for _kc in cutlass.range_constexpr(NKC):
                            gQ = cute.local_tile(mQ_cur, (self.m_block_size, self.k_block_size), (m_block, _kc))
                            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_Q, 0, cute.make_layout(1), gQ, sQ[None, None, None, handle_Q.index * NKC + _kc], single_stage=True
                            )
                            load_Q_fn(tma_bar_ptr=handle_Q.barrier)

                        # K blocks left to right; a ring stage holds G of the block's num_k_chunks chunks.
                        if const_expr(is_paged):
                            k_box_rows = min(self.page_size, self.n_block_size)
                            k_boxes = self.n_block_size // k_box_rows
                            k_box_layout = cute.make_composed_layout(
                                sK_layout.inner, 0, cute.make_layout((k_box_rows, self.k_block_size), stride=(self.k_block_size, 1))
                            )
                            sK_ptr = storage.sK.data_ptr()
                            last_slot = cute.size(mBlockTable.shape[1]) - Int32(1)
                        else:
                            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, 0]
                        nb = Int32(0)
                        while nb < num_n_blocks:
                            if const_expr(is_paged):
                                # One page lookup per row box; the n-block count is clamped by this batch's key
                                # count, and the slot is clamped into the table row so the read never leaves it.
                                mK_boxes = []
                                box_rows = []
                                for _j in cutlass.range_constexpr(k_boxes):
                                    row = nb * Int32(self.n_block_size) + Int32(_j * k_box_rows)
                                    page_slot = row // Int32(self.page_size)
                                    page_slot = page_slot if page_slot < last_slot else last_slot
                                    page = mBlockTable[batch_idx, page_slot]
                                    mK_boxes.append(mK[None, None, None, page][None, None, 0])
                                    box_rows.append((row - page_slot * Int32(self.page_size)) // Int32(k_box_rows))
                            for _kc in cutlass.range_constexpr(NKC):
                                if const_expr(_kc % G == 0):
                                    pipeline_KV.producer_acquire(KV_producer_state)
                                    kv_bar = pipeline_KV.producer_get_barrier(KV_producer_state)
                                slot = KV_producer_state.index * G + _kc % G
                                if const_expr(is_paged):
                                    for _j in cutlass.range_constexpr(k_boxes):
                                        gK = cute.local_tile(mK_boxes[_j], (k_box_rows, self.k_block_size), (box_rows[_j], _kc))
                                        sK_box = cute.make_tensor(
                                            sK_ptr + slot * Int32(self.n_block_size * self.k_block_size) + _j * k_box_rows * self.k_block_size, k_box_layout
                                        )
                                        load_K_fn, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, cute.make_layout(1), gK, sK_box, single_stage=True)
                                        load_K_fn(tma_bar_ptr=kv_bar)
                                else:
                                    gK = cute.local_tile(mK_cur, (self.n_block_size, self.k_block_size), (nb, _kc))
                                    load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                                        tma_atom_K, 0, cute.make_layout(1), gK, sK[None, None, None, slot], single_stage=True
                                    )
                                    load_K_fn(tma_bar_ptr=kv_bar)
                                if const_expr(_kc % G == G - 1):
                                    KV_producer_state.advance()
                            nb = nb + 1
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                head_producer.tail()
                Q_producer.tail()
                pipeline_KV.producer_tail(KV_producer_state)

            # ---------------- MMA warp ----------------
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(Int32(self.tmem_alloc_cols), tmem_holding_buf)
                cute.arch.sync_warp()
                KV_mma_state = make_pipeline_state(PipelineUserType.Consumer, self.num_kv_stages)
                t_empty_bits = Int32((1 << NUM_ACC) - 1)
                tSrK = tiled_mma_qk.make_fragment_A(sK)
                tSrQ = tiled_mma_qk.make_fragment_B(sQ)
                qk_mma_op = tiled_mma_qk.op
                while work_tile.is_valid_tile:
                    m_block_sched = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                    num_m_blocks_cur = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.m_block_size)
                    if m_block_sched < num_m_blocks_cur:
                        m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                        num_n_blocks = self._dense_compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)
                        handle_Q = Q_consumer.wait_and_advance()
                        nb = Int32(0)
                        acc = Int32(0)
                        while nb < num_n_blocks:
                            cute.arch.mbarrier_wait(T_mbar_ptr + NUM_ACC + acc, (t_empty_bits >> acc) & Int32(1))
                            t_empty_bits = t_empty_bits ^ (Int32(1) << acc)
                            for _kc in cutlass.range_constexpr(NKC):
                                if const_expr(_kc % G == 0):
                                    pipeline_KV.consumer_wait(KV_mma_state)
                                    _tcgen05_fence_after()
                                k_idx = KV_mma_state.index * G + _kc % G
                                q_idx = handle_Q.index * NKC + _kc
                                _gemm_ptx_partial(
                                    qk_mma_op,
                                    acc * self.tmem_s_stride,
                                    tSrK[None, None, None, k_idx],
                                    tSrQ[None, None, None, q_idx],
                                    sA=sK[None, None, None, k_idx],
                                    sB=sQ[None, None, None, q_idx],
                                    zero_init=const_expr(_kc == 0),
                                )
                                if const_expr(_kc % G == G - 1):
                                    pipeline_KV.consumer_release(KV_mma_state)
                                    KV_mma_state.advance()
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

        # ---------------- math warpgroups ----------------
        if warp_group_idx > 0:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue)
            self._math_warpgroup(
                warp_group_idx - 1,
                tidx,
                tiled_mma_qk,
                tStS_ref,
                sPerHead,
                head_consumer,
                storage.sRed.get_tensor(cute.make_layout((8 * self.num_math_wg,), stride=(1,))),
                T_mbar_ptr,
                mOut,
                mDenom,
                softmax_scale,
                SeqlenInfoCls,
                mQCausalOffsets,
                mCandBatchOffsets,
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
        sPerHead,
        head_consumer,
        sRed,
        T_mbar_ptr,
        mOut,
        mDenom,
        softmax_scale,
        SeqlenInfoCls,
        mQCausalOffsets,
        mCandBatchOffsets,
        work_tile,
        tile_sched,
        clc_pipeline,
        clc_consumer_state,
        tmem_dealloc_mbar_ptr,
    ):
        """One math warpgroup: owns packed columns ``[wg * cols, (wg + 1) * cols)`` of every block (one code copy, runtime ``wg``)."""
        tidx_wg = tidx % self.WARPGROUP_SIZE
        ratio = Int32(self.ratio)
        tpw = self.q_tokens_per_tile // self.num_math_wg
        cols = self.m_block_size // self.num_math_wg
        t_full_bits = Int32(0)
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.chunk_cols)), Float32)
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_ref).get_slice(tidx_wg)
        thr_mma = tiled_mma_qk.get_slice(tidx_wg)
        tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2])))
        tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
        kv_offset = tScS[0][0]
        rW = cute.make_rmem_tensor((cols,), Float32)
        while work_tile.is_valid_tile:
            m_block_sched = work_tile.tile_idx[0]
            batch_idx = work_tile.tile_idx[2]
            seqlen = SeqlenInfoCls(batch_idx)
            q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
            num_m_blocks_cur = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.m_block_size)
            if m_block_sched < num_m_blocks_cur:
                m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                num_n_blocks = self._dense_compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)
                if const_expr(self.is_compressed_logits):
                    mOut_cur = mOut
                else:
                    mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=2)
                mDenom_cur = seqlen.offset_batch_Q(mDenom, batch_idx, dim=1) if const_expr(self.compute_lse) else None
                head = head_consumer.wait_and_advance()
                sW = cute.make_tensor(sPerHead.iterator + head.index * self.m_block_size + wg * cols, cute.make_layout((cols,), stride=(1,)))
                q_toks = [m_block * self.q_tokens_per_tile + wg * tpw + Int32(u) for u in range(tpw)]
                col_limits = [(q_causal_offset + q_toks[u] + Int32(1)) // ratio for u in range(tpw)]
                if num_n_blocks > Int32(0):
                    cute.autovec_copy(sW, rW)
                head.release()
                t_full_bits = self._math_tile(
                    wg,
                    tpw,
                    cols,
                    num_n_blocks,
                    q_toks,
                    col_limits,
                    seqlen,
                    mOut_cur,
                    mDenom_cur,
                    rW,
                    softmax_scale,
                    thr_tmem_load,
                    tStS_ref,
                    tSrS_shape,
                    kv_offset,
                    T_mbar_ptr,
                    t_full_bits,
                    sRed,
                    tidx_wg,
                    batch_idx,
                    q_causal_offset,
                    mCandBatchOffsets,
                )
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
        if tidx_wg == 0:
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    def _load_chunk(self, thr_tmem_load, tStS_ref, buf, wg, cols: int, j: int, c: int):
        """Issue the TMEM load of chunk ``c`` of accumulator ``j`` (this warpgroup's columns) into ``buf``."""
        col = j * self.tmem_s_stride + c * self.chunk_cols
        tStS_c = cute.make_tensor(cute.make_ptr(Float32, wg * cols + col, mem_space=cute.AddressSpace.tmem, assumed_align=16), tStS_ref.layout)
        cute.copy(thr_tmem_load, thr_tmem_load.partition_S(tStS_c)[None, 0, None, None], buf[None, 0, None, None])

    @cute.jit
    def _reduce_chunk(self, buf, rW, c: cutlass.Constexpr[int], accs):
        """Accumulate chunk ``c`` (columns ``[c * CH, (c + 1) * CH)`` of this warpgroup) into the per-token
        (acc0, acc1) pairs of ``accs``; returns the sums of the local tokens the chunk completes, in token order.

        Pair ``p`` of a token goes to acc ``p % 2`` in pair order whatever the chunking, so the result is
        bitwise the whole-token reduce.  The all-FADD tree yields 2x the sum (the caller folds the 1/2)."""
        H = self.qhead_per_kvhead
        CH = self.chunk_cols
        done = []
        for q in cutlass.range_constexpr(CH // 2):
            col = c * CH + 2 * q
            u = col // H
            p = (col % H) // 2
            if const_expr(p == 0):
                accs[u] = [(Float32(0.0), Float32(0.0)), (Float32(0.0), Float32(0.0))]
            v0 = buf[2 * q]
            v1 = buf[2 * q + 1]
            if const_expr(self.relu_fadd_all or p % 2 == 1):
                # relu(v) * w == (v + |v|) * w / 2 exactly
                v0, v1 = add_packed_f32x2((v0, v1), (cute.math.absf(v0), cute.math.absf(v1)))
            else:
                v0 = v0 if v0 > Float32(0.0) else Float32(0.0)
                v1 = v1 if v1 > Float32(0.0) else Float32(0.0)
            accs[u][p % 2] = fma_packed_f32x2((v0, v1), (rW[col], rW[col + 1]), accs[u][p % 2])
            if const_expr(p == H // 2 - 1):
                acc0, acc1 = accs[u]
                if const_expr(self.relu_fadd_all):
                    acc = add_packed_f32x2(acc0, acc1)
                else:
                    acc = fma_packed_f32x2(acc1, (Float32(0.5), Float32(0.5)), acc0)
                done.append(acc[0] + acc[1])
        return done

    @cute.jit
    def _math_tile(
        self,
        wg,
        tpw: cutlass.Constexpr[int],
        cols: cutlass.Constexpr[int],
        num_n_blocks,
        q_toks,
        col_limits,
        seqlen,
        mOut_cur,
        mDenom_cur,
        rW,
        softmax_scale,
        thr_tmem_load,
        tStS_ref,
        tSrS_shape,
        kv_offset,
        T_mbar_ptr,
        t_full_bits,
        sRed,
        tidx_wg,
        batch_idx,
        q_causal_offset,
        mCandBatchOffsets,
    ):
        """Every n block of one work tile for this warpgroup's tokens, then their LSE.  Returns the updated full-barrier phase bits."""
        NUM_ACC = self.NUM_ACC
        out_scale = Float32(softmax_scale) * Float32(0.5) if const_expr(self.relu_fadd_all) else Float32(softmax_scale)
        log2_e = Float32(math.log2(math.e))
        lims = []
        rows = []
        row_base = None
        for u in cutlass.range_constexpr(tpw):
            lim = col_limits[u] if col_limits[u] < seqlen.seqlen_k else seqlen.seqlen_k
            lim = lim if q_toks[u] < seqlen.seqlen_q else Int32(0)
            lims.append(lim)
            q_row = q_toks[u] if q_toks[u] < seqlen.seqlen_q else Int32(0)
            if const_expr(self.is_compressed_logits):
                # Compact ratio-causal layout: batch slab base + prefix count of the earlier rows.
                q_global_start = Int64(q_causal_offset)
                row_off = self._row_offset_i64(Int64(q_row), q_global_start)
                if const_expr(u == 0):
                    row_off0 = row_off
                    if const_expr(self.cand_2d):
                        base_idx = cute.crd2idx((batch_idx, Int32(row_off0)), mOut_cur.layout)
                    elif const_expr(mCandBatchOffsets is not None):
                        base_idx = Int64(mCandBatchOffsets[batch_idx]) + row_off0
                    else:
                        base_idx = Int64(batch_idx) * self._row_offset_i64(Int64(seqlen.seqlen_q), q_global_start) + row_off0
                    row_base = cute.make_tensor(mOut_cur.iterator + base_idx, cute.make_layout((tpw * seqlen.seqlen_k,), stride=(1,)))
                rows.append(Int32(row_off - row_off0))
            else:
                rows.append(
                    cute.make_tensor(mOut_cur.iterator + cute.crd2idx((q_row, 0), mOut_cur.layout), cute.make_layout((mOut_cur.shape[1],), stride=(1,)))
                )
        lmax = [-Float32.inf for _ in range(tpw)]
        lsum = [Float32(0.0) for _ in range(tpw)]
        bufs = [cute.make_rmem_tensor(tSrS_shape, Float32), cute.make_rmem_tensor(tSrS_shape, Float32)]
        CH = self.chunk_cols
        ncs = cols // CH

        # Prologue: block 0 chunk 0.  Every later chunk is loaded one step ahead of its reduce (across block
        # boundaries too); a block's accumulator is released once its last chunk is in registers.
        if num_n_blocks > Int32(0):
            cute.arch.mbarrier_wait(T_mbar_ptr, t_full_bits & Int32(1))
            t_full_bits = t_full_bits ^ Int32(1)
            _tcgen05_fence_after()
            self._load_chunk(thr_tmem_load, tStS_ref, bufs[0], wg, cols, 0, 0)
            cute.arch.fence_view_async_tmem_load()
        blk0 = Int32(0)
        while blk0 < num_n_blocks:
            for j in cutlass.range_constexpr(NUM_ACC):
                if blk0 + j < num_n_blocks:
                    pos = kv_offset + (blk0 + j) * self.n_block_size
                    accs = {}
                    for c in cutlass.range_constexpr(ncs):
                        b = (j * ncs + c) % 2
                        if const_expr(c + 1 < ncs):
                            self._load_chunk(thr_tmem_load, tStS_ref, bufs[1 - b], wg, cols, j, c + 1)
                        else:
                            _tcgen05_fence_before()
                            cute.arch.mbarrier_arrive(T_mbar_ptr + NUM_ACC + j)
                            jn = (j + 1) % NUM_ACC
                            if blk0 + (j + 1) < num_n_blocks:
                                cute.arch.mbarrier_wait(T_mbar_ptr + jn, (t_full_bits >> jn) & Int32(1))
                                t_full_bits = t_full_bits ^ Int32(1 << jn)
                                _tcgen05_fence_after()
                                self._load_chunk(thr_tmem_load, tStS_ref, bufs[1 - b], wg, cols, jn, 0)
                        done = self._reduce_chunk(bufs[b], rW, c, accs)
                        done_toks = [u for u in range(tpw) if ((u + 1) * self.qhead_per_kvhead - 1) // CH == c]
                        for u in cutlass.range_constexpr(tpw):
                            if const_expr(u in done_toks):
                                val = done[u - done_toks[0]] * out_scale
                                if pos < lims[u]:
                                    if const_expr(self.is_compressed_logits):
                                        row_base[rows[u] + pos] = val
                                    else:
                                        rows[u][pos] = val
                                    if const_expr(self.compute_lse):
                                        new_max = val if val > lmax[u] else lmax[u]
                                        lsum[u] = lsum[u] * cute.math.exp2((lmax[u] - new_max) * log2_e, fastmath=True) + cute.math.exp2(
                                            (val - new_max) * log2_e, fastmath=True
                                        )
                                        lmax[u] = new_max
                        cute.arch.fence_view_async_tmem_load()
            blk0 = blk0 + NUM_ACC
        if const_expr(self.compute_lse):
            warp = tidx_wg // self.WARP_SIZE
            bar_id = 2 + wg  # named barriers: 0 = CTA sync
            inv_log2_e = Float32(1.0 / math.log2(math.e))
            for u in cutlass.range_constexpr(tpw):
                wmax = cute.arch.warp_redux_sync(lmax[u], "fmax")
                if tidx_wg % self.WARP_SIZE == 0:
                    sRed[wg * 8 + warp] = wmax
                cute.arch.barrier(barrier_id=bar_id, number_of_threads=self.WARPGROUP_SIZE)
                gmax = sRed[wg * 8]
                for wi in cutlass.range_constexpr(1, 4):
                    v = sRed[wg * 8 + wi]
                    gmax = v if v > gmax else gmax
                adj = Float32(0.0)
                if gmax > Float32(-1e30):
                    adj = lsum[u] * cute.math.exp2((lmax[u] - gmax) * log2_e, fastmath=True)
                wsum = cute.arch.warp_reduction_sum(adj)
                if tidx_wg % self.WARP_SIZE == 0:
                    sRed[wg * 8 + 4 + warp] = wsum
                cute.arch.barrier(barrier_id=bar_id, number_of_threads=self.WARPGROUP_SIZE)
                if tidx_wg == 0:
                    gsum = sRed[wg * 8 + 4] + sRed[wg * 8 + 5] + sRed[wg * 8 + 6] + sRed[wg * 8 + 7]
                    lse = -Float32.inf
                    if gmax > Float32(-1e30):
                        lse = gmax + cute.math.log2(gsum) * inv_log2_e
                    if q_toks[u] < seqlen.seqlen_q:
                        mDenom_cur[q_toks[u]] = lse
                cute.arch.barrier(barrier_id=bar_id, number_of_threads=self.WARPGROUP_SIZE)
        return t_full_bits
