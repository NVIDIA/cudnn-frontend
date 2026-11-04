# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 CuTeDSL unified MXFP8 indexer-score kernel (MXFP4 subclasses it).

One CTA owns one packed m tile (128 (token, head) rows: 128 // H query tokens) and
walks its whole ratio-causal K range under CLC persistent scheduling.  SwapAB: K is
the MMA's A operand (128 keys), packed Q the B operand, so a 128-key block's scores
land in TMEM as 128 lanes (keys) x 128 columns (token, head).

In-CTA pipeline (DeepGEMM mqa_logits style):

* the K range is consumed in *splits* of ``SPLIT`` (=3) n blocks (384 keys).  One TMA
  barrier per split covers its K data (``num_k_chunks`` chunks per n block) and its
  ``SPLIT`` K-scale tiles; the split ring is ``NUM_SPLIT_STAGES`` deep.  Q / Q scale /
  W are double buffered so the next work tile's Q lands while this tile computes; W has
  its own 2-stage async pipeline (all 32 load lanes publish, every math lane releases).
* warp 3 copies a split's K scales SMEM -> TMEM into a ``NUM_SF_STAGES``-deep SFA ring
  (sf_full / sf_empty barriers), so the MMA warp only issues MMAs: per split it waits
  the KV and SF barriers once, then issues every block's MMAs into TMEM accumulator
  ``j`` (columns ``[128 j, 128 j + 128)``), each guarded by its own full/empty pair.
* math warpgroups split the 128 columns: warpgroup ``t`` of ``num_math_wg`` owns columns
  ``[t * cols, (t + 1) * cols)`` -- ``cols / H`` whole query tokens -- of every block.
  A block's TMEM is released once every warpgroup has loaded its columns; the next
  block's TMEM load is issued before the previous block's head reduce.  The head
  reduce is unconditional (only the store is predicated) and stores go through
  per-token row tensors (dense scores or the compact ratio-causal candidate buffer).
  The optional LSE is an online (max, sum) per token, reduced across the warpgroup.

``relu(s) * w`` is evaluated as ``(s + |s|) * w / 2`` (FADD2 with an |abs| operand on
the FMA pipe) for every pair when H < 64 and for odd pairs at H64; the halving is
applied once per score.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Tuple

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
    make_blockscaled_trivial_tiled_mma as _make_blockscaled_trivial_tiled_mma,
    cluster_shape_to_tma_atom_A as _cluster_shape_to_tma_atom_A,
    cluster_shape_to_tma_atom_SFB as _cluster_shape_to_tma_atom_SFB,
)
from cutlass.utils import blockscaled_layout as _blockscaled_layout
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator

import cuda.bindings.driver as cuda

from cudnn.deepseek_sparse_attention.utils import copy as copy_utils
from cudnn.deepseek_sparse_attention.utils.sm100.gemm import (
    gemm_ptx_partial_mxfp8 as _gemm_ptx_partial_mxfp8,
)
from cudnn.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK
from .indexer_score_unified_sm100 import IndexerScoreSm100Base, _tcgen05_fence_after, _tcgen05_fence_before
from .dense_score_recompute_sm100 import (
    add_packed_f32x2,
    fma_packed_f32x2,
)


class IndexerScoreUnifiedSm100Mxfp8(IndexerScoreSm100Base):
    """Block-scaled (MXFP8) dense / compact indexer score with optional LSE."""

    arch = 100
    # The tile stays 128 packed rows wide for every group (the scale atom), so the
    # smallest group (4 heads) packs 32 tokens.
    max_q_tokens_per_tile = 32
    # Block-scaled format of Q/K, named in errors.  E4M3 Q/K run tcgen05
    # kind::mxf8f6f4, whose MMA reads K=32 values and one E8M0 scale per row.
    format_name = "MXFP8"
    mma_inst_k = 32

    SPLIT = 3  # n blocks (= TMEM accumulators) per KV split
    NUM_SPLIT_STAGES = 3  # KV split ring depth (SMEM)
    NUM_SF_STAGES = 2  # K-scale TMEM ring depth
    Q_STAGES = 2  # Q / Q scale ring depth
    NUM_W_STAGES = 2  # W ring depth (deeper measured no faster)

    def __init__(
        self,
        head_dim: int,
        qhead_per_kvhead: int = 64,
        m_block_size: int = 128,
        n_block_size: int = 128,
        k_block_size: int | None = None,
        ratio: int = 1,
        is_varlen: bool = False,
        sf_vec_size: int = 32,
        compute_lse: bool = False,
        is_compressed_logits: bool = False,
    ):
        if head_dim != 128:
            raise ValueError(f"SM100 unified indexer {self.format_name} currently requires head_dim=128")
        if sf_vec_size != 32:
            raise ValueError(f"SM100 dense score {self.format_name} currently requires sf_vec_size=32")
        if m_block_size != 128 or n_block_size != 128:
            raise ValueError(f"SM100 dense score {self.format_name} currently requires m_block_size=n_block_size=128")
        if qhead_per_kvhead not in (4, 8, 16, 32, 64):
            raise ValueError(f"SM100 unified indexer {self.format_name} requires qhead_per_kvhead in (4, 8, 16, 32, 64), got {qhead_per_kvhead}")
        k_block_size = 64 if k_block_size is None else k_block_size
        # k_block_size is an internal K-split knob (num_k_chunks = head_dim_padded // k_block_size;
        # the per-chunk scale phase derives from it): any multiple of sf_vec_size and of the MMA K
        # that divides head_dim_padded is valid (default 64).
        _hd_padded = ((head_dim + 15) // 16) * 16
        if k_block_size <= 0 or k_block_size % sf_vec_size != 0 or _hd_padded % k_block_size != 0:
            raise ValueError(
                f"SM100 unified indexer {self.format_name} requires k_block_size to be a "
                f"positive multiple of sf_vec_size ({sf_vec_size}) that divides "
                f"head_dim_padded ({_hd_padded}); got {k_block_size}"
            )
        if k_block_size % self.mma_inst_k != 0:
            raise ValueError(
                f"SM100 unified indexer {self.format_name} requires k_block_size to be a multiple of the MMA K ({self.mma_inst_k}); got {k_block_size}"
            )
        super().__init__(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            kv_stage=self.NUM_SPLIT_STAGES * self.SPLIT * (_hd_padded // k_block_size),
            k_block_size=k_block_size,
            ratio=ratio,
            is_varlen=is_varlen,
            compute_lse=compute_lse,
            is_compressed_logits=is_compressed_logits,
        )
        self.sf_vec_size = sf_vec_size
        self.sf_dtype = cutlass.Float8E8M0FNU
        # One full-D128 scale tile per n block (K) / per work tile (Q).
        self.scale_tile_k = self.head_dim_padded

        # SMEM rings, in the base layouts' units: K data in (128 x k_block) chunks, K scales in
        # 128-row tiles, Q in k chunks per Q stage, Q scales one tile per Q stage. The K ring
        # (kv_stage = NUM_SPLIT_STAGES * SPLIT * num_k_chunks) is passed to the base class above.
        self.q_stage = self.Q_STAGES
        self.num_q_scale_stages = self.q_stage
        self.k_scale_stage = self.NUM_SPLIT_STAGES * self.SPLIT

        # Warps: 0 load, 1 MMA, 2 CLC scheduler, 3 K-scale S2T, then the math warpgroups.
        # H64 tiles hold 2 tokens (one per math warpgroup); smaller groups use 4 warpgroups so a
        # thread's resident W / score columns stay small and 4 warps per SMSP hide TMEM latency.
        self.load_warp_id, self.mma_warp_id, self.sched_warp_id, self.sf_warp_id = 0, 1, 2, 3
        self.num_math_wg = 2 if self.q_tokens_per_tile == 2 else 4
        assert self.q_tokens_per_tile % self.num_math_wg == 0
        self.num_warps = 4 + 4 * self.num_math_wg
        self.threads_per_cta = self.WARP_SIZE * self.num_warps
        # setmaxnreg only redistributes the CTA's launch allocation (floor8(64K / threads) per thread);
        # asking for more than that pool hangs the increasing warpgroups.
        self.num_regs_load = 56
        launch_regs = (65536 // self.threads_per_cta) // 8 * 8
        self.num_regs_epilogue = min(
            256,
            ((launch_regs * self.threads_per_cta - self.WARPGROUP_SIZE * self.num_regs_load) // (self.WARPGROUP_SIZE * self.num_math_wg)) // 8 * 8,
        )
        # relu via (s + |s|) for every head pair below H64 (the reduce is issue bound there); H64 keeps
        # the ALU/FMA split (FMNMX for even pairs, FADD2 for odd pairs).
        self.relu_fadd_all = self.qhead_per_kvhead != 64

        # TMEM: SPLIT accumulators of m_block columns, then a NUM_SF_STAGES ring of SPLIT K-scale
        # (SFA) fields, then the Q-scale (SFB) field.
        self.tmem_s_stride = self.m_block_size
        self.tmem_sfa_cols = (self.n_block_size // 32) * 4
        self.tmem_sfb_cols = (self.m_block_size // 32) * 4
        self.tmem_k_sfa_base_offset = self.SPLIT * self.m_block_size
        self.sfa_stage_cols = self.SPLIT * self.tmem_sfa_cols
        self.tmem_q_sfb_offset = self.tmem_k_sfa_base_offset + self.NUM_SF_STAGES * self.sfa_stage_cols
        self.tmem_alloc_cols = 512
        assert self.tmem_q_sfb_offset + self.tmem_sfb_cols <= self.tmem_alloc_cols

    @staticmethod
    def _qk_tensor(t: cute.Tensor) -> cute.Tensor:
        """Q or K as this kernel reads it, from the tensor it is launched with (the same tensor here)."""
        return t

    def _make_data_tiled_mma(self, blockscaled_tiled_mma_qk, cta_group):
        """Tiled MMA that lays out and partitions the Q/K data tiles: the unscaled E4M3 MMA of the QK tile."""
        return _make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.k_major_mode,
            self.q_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPerHead: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mOut: cute.Tensor,
        mDenom: cute.Tensor | None,
        softmax_scale: Float32 | float,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mCuSeqlensQScalePadded: cute.Tensor | None,
        mCuSeqlensKScalePadded: cute.Tensor | None,
        mQCausalOffsets: cute.Tensor | None,
        stream: cuda.CUstream,
        mCandBatchOffsets: cute.Tensor | None = None,
    ):
        # mDenom is None when compute_lse is off: the slot is compiled out (R3).
        # Runtime tensor metadata is needed to build CuTe layouts and compile
        # the matching kernel.  The element dtypes become compile-time constants
        # inside cute.compile.
        mQ = self._qk_tensor(mQ)
        mK = self._qk_tensor(mK)
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.sf_dtype = mQScale.element_type
        is_varlen = mCuSeqlensQ is not None

        # The API accepts either BSHD or packed THD.  The compile key records
        # is_varlen, and these asserts make accidental mismatches fail during
        # compilation instead of silently interpreting strides incorrectly.
        if const_expr(is_varlen):
            assert self.is_varlen
            assert mCuSeqlensQ is not None and mCuSeqlensK is not None
            assert mCuSeqlensQScalePadded is not None
            assert mCuSeqlensKScalePadded is not None
        else:
            assert not self.is_varlen
            assert mCuSeqlensQ is None and mCuSeqlensK is None
            assert mCuSeqlensQScalePadded is None
            assert mCuSeqlensKScalePadded is None

        # Normalize BSHD and THD tensors into the K-major/Q-major shapes expected
        # by the SM100 TMA helpers.  THD keeps batch in cu_seqlens; BSHD exposes
        # the batch as the final tensor mode.
        Q_layout_transpose = [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0]
        K_layout_transpose = [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=K_layout_transpose))

        seqlen_q_static = max_seqlen_q if const_expr(is_varlen) else cute.size(mQ.shape[0])
        seqlen_q_packed = seqlen_q_static * self.qhead_per_kvhead

        # The packed scale tensors expose their caller-defined, atom-aligned MN
        # address extent directly in the second mode.
        scale_l_size = cute.size(mQScale.shape[0])
        q_scale_layout = _blockscaled_layout.tile_atom_to_shape_SF(
            (cute.size(mQScale.shape[1]), self.head_dim_padded, scale_l_size),
            self.sf_vec_size,
        )
        k_scale_layout = _blockscaled_layout.tile_atom_to_shape_SF(
            (cute.size(mKScale.shape[1]), self.head_dim_padded, scale_l_size),
            self.sf_vec_size,
        )
        mQScale = cute.make_tensor(mQScale.iterator, q_scale_layout)
        mKScale = cute.make_tensor(mKScale.iterator, k_scale_layout)

        # Fold qhead_per_kvhead into the M dimension so one m tile covers two
        # QH64 tokens or four QH32 tokens. The epilogue reverses this packed
        # index when writing outputs.
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

        # blockscaled_tiled_mma_qk is the UMMA op with SFA/SFB fields.  tiled_mma_qk
        # (``_make_data_tiled_mma``) lays out and partitions the data tensors; it may
        # be the block-scaled op itself.
        blockscaled_tiled_mma_qk = _make_blockscaled_trivial_tiled_mma(
            self.k_dtype,
            self.k_dtype,
            self.k_major_mode,
            self.q_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_qk = self._make_data_tiled_mma(blockscaled_tiled_mma_qk, cta_group)

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # Data SMEM follows the data tiled MMA's layouts; scale SMEM follows CUTLASS
        # blockscaled SFA/SFB layouts so tcgen05 can consume E8M0 factors from
        # TMEM fields rather than doing explicit dequant math in registers.
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
            self.num_k_chunks * self.q_stage,
        )
        scale_tile_k = self.scale_tile_k
        scale_mma_tiler_qk = (self.n_block_size, self.m_block_size, scale_tile_k)
        sKScale_layout = _blockscaled_layout.make_smem_layout_sfa(
            blockscaled_tiled_mma_qk,
            scale_mma_tiler_qk,
            self.sf_vec_size,
            self.k_scale_stage,
        )
        sQScale_layout = _blockscaled_layout.make_smem_layout_sfb(
            blockscaled_tiled_mma_qk,
            scale_mma_tiler_qk,
            self.sf_vec_size,
            self.num_q_scale_stages,
        )
        self.sQ_layout = sQ_layout
        self.sK_layout = sK_layout
        self.sQScale_layout = sQScale_layout
        self.sKScale_layout = sKScale_layout

        # Data TMA atoms load the Q/K tiles into SMEM.  Scale TMA atoms load
        # packed E8M0 factors as Int16 internally because CUTLASS blockscaled
        # layouts store two E8M0 bytes per logical compact element.
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

        k_scale_tma_op = _cluster_shape_to_tma_atom_A(self.cluster_shape_mn, blockscaled_tiled_mma_qk.thr_id)
        q_scale_tma_op = _cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, blockscaled_tiled_mma_qk.thr_id)
        k_scale_smem_layout = cute.slice_(sKScale_layout, (None, None, None, 0))
        q_scale_smem_layout = cute.slice_(sQScale_layout, (None, None, None, 0))
        tma_atom_KScale, mKScale = cute.nvgpu.make_tiled_tma_atom_A(
            k_scale_tma_op,
            mKScale,
            k_scale_smem_layout,
            scale_mma_tiler_qk,
            blockscaled_tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_QScale, mQScale = cute.nvgpu.make_tiled_tma_atom_B(
            q_scale_tma_op,
            mQScale,
            q_scale_smem_layout,
            scale_mma_tiler_qk,
            blockscaled_tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        q_data_tma_bytes = cute.size_in_bytes(self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2]))
        k_data_tma_bytes = cute.size_in_bytes(self.k_dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        q_scale_tma_bytes = cute.size_in_bytes(self.sf_dtype, q_scale_smem_layout)
        k_scale_tma_bytes = cute.size_in_bytes(self.sf_dtype, k_scale_smem_layout)
        # KScale has its own pipeline and barrier accounting.  One full-D128
        # scale tile is reused by the two k64 data chunks for the same n block.
        self.tma_copy_bytes = {
            "Q": self.num_k_chunks * q_data_tma_bytes + q_scale_tma_bytes,
            "K": k_data_tma_bytes,
            "KScale": k_scale_tma_bytes,
        }

        PerHead_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
        mPerHead = cute.make_tensor(mPerHead.iterator, cute.select(mPerHead.layout, mode=PerHead_transpose))

        if const_expr(not self.is_compressed_logits):
            Out_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
            mOut = cute.make_tensor(mOut.iterator, cute.select(mOut.layout, mode=Out_transpose))

        if const_expr(mDenom is not None):
            Denom_transpose = [0] if const_expr(is_varlen) else [1, 0]
            mDenom = cute.make_tensor(mDenom.iterator, cute.select(mDenom.layout, mode=Denom_transpose))
        else:
            assert not self.compute_lse, "mDenom is None but compute_lse=True writes the denominator"

        # Persistent scheduler grid: one logical work tile per m block and
        # batch.  The kernel itself walks all dense K blocks for that m tile.
        num_m_blocks = cute.ceil_div(seqlen_q_packed, self.m_block_size)
        batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1 if const_expr(is_varlen) else cute.size(mQ.shape[3])
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams((num_m_blocks, 1, batch_size), (*self.cluster_shape_mn, 1))
        grid_dim = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
        # Launch signature keeps Q/K scales explicit so the same unified object
        # can serve forward score-only and backward score+LSE dispatches.
        self.kernel(
            mQ,
            mK,
            mPerHead,
            mQScale,
            mKScale,
            mOut,
            mDenom,
            softmax_scale,
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
            tile_sched_params,
            max_seqlen_q,
            max_seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuSeqlensQScalePadded,
            mCuSeqlensKScalePadded,
            mQCausalOffsets,
            mCandBatchOffsets,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
        )

    @cute.jit
    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        # Scale factors are loaded by TMA into SMEM and then copied into TMEM
        # SFA/SFB fields with tcgen05 S2T copies.  filter_zeros keeps the copy
        # compact for the blockscaled layout's sparse logical strides.
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def _gemm_blockscaled_qk(
        self,
        blockscaled_tiled_mma_qk,
        tStS_stage,
        tSrK_stage,
        tSrQ_stage,
        sK_stage,
        sQ_stage,
        tCtKSFA,
        tCtQSFB,
        accumulate_first: cutlass.Constexpr[bool] = False,
        kphase_offset: cutlass.Constexpr[int] = 0,
    ):
        """Blockscaled QK MMAs of one K chunk with explicit SFA/SFB TMEM operands.

        The direct PTX wrapper keeps every descriptor a compile-time base plus a uniform offset;
        the generic ``cute.gemm`` path re-materialized descriptors through ELECT / R2UR per MMA,
        which made the MMA warp issue bound.  ``kphase_offset`` is the index of this chunk's first
        MMA along K and selects its E8M0 scale group.
        """
        _gemm_ptx_partial_mxfp8(
            blockscaled_tiled_mma_qk.op,
            tStS_stage.iterator.toint(),
            tSrK_stage,
            tSrQ_stage,
            sK_stage,
            sQ_stage,
            tCtKSFA,
            tCtQSFB,
            zero_init=const_expr(not accumulate_first),
            kphase_offset=kphase_offset,
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
        mDenom: cute.Tensor | None,
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
        mCandBatchOffsets: cute.Tensor | None = None,
    ):
        is_varlen = mCuSeqlensQ is not None
        SPLIT = self.SPLIT
        NKC = self.num_k_chunks
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
        sPerHead_size = self.m_block_size * self.NUM_W_STAGES

        @cute.struct
        class SharedStorage:
            per_head_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.NUM_W_STAGES]
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.q_stage]
            KV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.NUM_SPLIT_STAGES]
            SF_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.NUM_SF_STAGES]
            T_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.SPLIT]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, 4]
            sPerHead: cute.struct.Align[cute.struct.MemRange[Float32, sPerHead_size], 128]
            sRed: cute.struct.Align[cute.struct.MemRange[Float32, 8 * self.num_math_wg], 128]  # LSE cross-warp scratch
            sQScale: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, sQScale_size], self.buffer_align_bytes]
            sKScale: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, sKScale_size], self.buffer_align_bytes]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.k_dtype, sK_size], self.buffer_align_bytes]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        Q_mbar_ptr = storage.Q_mbar_ptr.data_ptr()
        KV_mbar_ptr = storage.KV_mbar_ptr.data_ptr()
        SF_mbar_ptr = storage.SF_mbar_ptr.data_ptr()  # [full0, full1, empty0, empty1]
        T_mbar_ptr = storage.T_mbar_ptr.data_ptr()  # [full_blk0..2, empty_blk0..2]
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf.ptr
        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQScale = storage.sQScale.get_tensor(sQScale_layout)
        sKScale = storage.sKScale.get_tensor(sKScale_layout)
        sPerHead = storage.sPerHead.get_tensor(cute.make_layout((sPerHead_size,), stride=(1,)))

        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))

        # Publish W from all load lanes to the math warpgroups; reuse a slot only after every math
        # lane has copied it out (tensor-core completion orders neither direction).
        pipeline_head = PipelineAsync.create(
            barrier_storage=storage.per_head_mbar_ptr.data_ptr(),
            num_stages=self.NUM_W_STAGES,
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
            num_stages=self.NUM_SPLIT_STAGES,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=SPLIT * (NKC * self.tma_copy_bytes["K"] + self.tma_copy_bytes["KScale"]),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        if warp_idx == 1:
            cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, self.num_math_wg)
        if warp_idx == 0:
            for _s in cutlass.range_constexpr(self.NUM_SF_STAGES):
                cute.arch.mbarrier_init(SF_mbar_ptr + _s, 1)  # full: SF warp commit
                cute.arch.mbarrier_init(SF_mbar_ptr + self.NUM_SF_STAGES + _s, 1)  # empty: MMA commit
            for _w in cutlass.range_constexpr(SPLIT):
                cute.arch.mbarrier_init(T_mbar_ptr + _w, 1)  # full: MMA commit
                cute.arch.mbarrier_init(T_mbar_ptr + SPLIT + _w, self.WARPGROUP_SIZE * self.num_math_wg)  # empty: every math WG
        num_clc_consumer_threads = self.WARP_SIZE * (1 + 3 + 4 * self.num_math_wg)
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

        sf_tmem_ptr = cute.recast_ptr(tStS_ref.iterator, dtype=self.sf_dtype)
        tmem_sf_offset_scale = self.qk_acc_dtype.width // self.sf_dtype.width
        k_scale_smem_layout = cute.slice_(sKScale_layout, (None, None, None, 0))
        q_scale_smem_layout = cute.slice_(sQScale_layout, (None, None, None, 0))
        tCtKSFA_layout = _blockscaled_layout.make_tmem_layout_sfa(
            blockscaled_tiled_mma_qk, (self.n_block_size, self.m_block_size, self.head_dim_padded), self.sf_vec_size, k_scale_smem_layout
        )
        tCtQSFB_layout = _blockscaled_layout.make_tmem_layout_sfb(
            blockscaled_tiled_mma_qk, (self.n_block_size, self.m_block_size, self.head_dim_padded), self.sf_vec_size, q_scale_smem_layout
        )
        tCtQSFB = cute.make_tensor(sf_tmem_ptr + tmem_sf_offset_scale * self.tmem_q_sfb_offset, tCtQSFB_layout)
        tiled_copy_s2t_qsfb, tCsQSFB_compact_s2t, tCtQSFB_compact_s2t = self.mainloop_s2t_copy_and_partition(sQScale, tCtQSFB)

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.WARPGROUP_SIZE)

        if warp_group_idx == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_load)

            # ---------------- load warp: W, Q/QScale, then KV splits left to right ----------------
            if warp_idx == self.load_warp_id:
                lane_id = tidx % self.WARP_SIZE
                scale_batch_idx = Int32(-1)
                q_scale_m_block = Int32(0)
                k_scale_n_block = Int32(0)
                KV_producer_state = make_pipeline_state(PipelineUserType.Producer, self.NUM_SPLIT_STAGES)
                while work_tile.is_valid_tile:
                    m_block_sched = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                    if const_expr(is_varlen):
                        scale_l_idx = 0
                        if batch_idx != scale_batch_idx:
                            q_scale_m_block = mCuSeqlensQScalePadded[batch_idx] * self.qhead_per_kvhead // self.m_block_size
                            k_scale_n_block = mCuSeqlensKScalePadded[batch_idx] // self.n_block_size
                            scale_batch_idx = batch_idx
                    else:
                        scale_l_idx = batch_idx
                    num_m_blocks_cur = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.m_block_size)
                    if m_block_sched < num_m_blocks_cur:
                        m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                        num_n_blocks = self._dense_compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)
                        num_splits = cute.ceil_div(num_n_blocks, SPLIT)
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
                        mQScale_cur = mQScale[None, None, scale_l_idx]
                        handle_Q = Q_producer.acquire_and_advance()
                        for _kc in cutlass.range_constexpr(NKC):
                            gQ = cute.local_tile(mQ_cur, (self.m_block_size, self.k_block_size), (m_block, _kc))
                            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_Q, 0, cute.make_layout(1), gQ, sQ[None, None, None, handle_Q.index * NKC + _kc], single_stage=True
                            )
                            load_Q_fn(tma_bar_ptr=handle_Q.barrier)
                        gQScale = cute.local_tile(mQScale_cur, (self.m_block_size, self.scale_tile_k), (q_scale_m_block + m_block, 0))
                        load_QScale_fn, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_QScale, 0, cute.make_layout(1), gQScale, sQScale[None, None, None, handle_Q.index], filter_zeros=True, single_stage=True
                        )
                        load_QScale_fn(tma_bar_ptr=handle_Q.barrier)

                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, 0]
                        mKScale_cur = mKScale[None, None, scale_l_idx]
                        split = Int32(0)
                        while split < num_splits:
                            pipeline_KV.producer_acquire(KV_producer_state)
                            kv_bar = pipeline_KV.producer_get_barrier(KV_producer_state)
                            kv_idx = KV_producer_state.index
                            for j in cutlass.range_constexpr(SPLIT):
                                nb = split * SPLIT + j
                                gKScale = cute.local_tile(mKScale_cur, (self.n_block_size, self.head_dim_padded), (k_scale_n_block + nb, 0))
                                load_KScale_fn, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_KScale,
                                    0,
                                    cute.make_layout(1),
                                    gKScale,
                                    sKScale[None, None, None, kv_idx * SPLIT + j],
                                    filter_zeros=True,
                                    single_stage=True,
                                )
                                load_KScale_fn(tma_bar_ptr=kv_bar)
                                for _kc in cutlass.range_constexpr(NKC):
                                    gK = cute.local_tile(mK_cur, (self.n_block_size, self.k_block_size), (nb, _kc))
                                    load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                                        tma_atom_K, 0, cute.make_layout(1), gK, sK[None, None, None, (kv_idx * SPLIT + j) * NKC + _kc], single_stage=True
                                    )
                                    load_K_fn(tma_bar_ptr=kv_bar)
                            KV_producer_state.advance()
                            split = split + 1
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                head_producer.tail()
                Q_producer.tail()
                pipeline_KV.producer_tail(KV_producer_state)

            # ---------------- SF warp: K scales SMEM -> TMEM (2-deep SFA ring) ----------------
            if warp_idx == self.sf_warp_id:
                # TMEM is allocated by the MMA warp; wait for it before the first S2T.
                cute.arch.barrier(barrier_id=1, number_of_threads=2 * self.WARP_SIZE)
                KV_sf_state = make_pipeline_state(PipelineUserType.Consumer, self.NUM_SPLIT_STAGES)
                sf_ctr = Int32(0)
                sf_empty_bits = Int32((1 << self.NUM_SF_STAGES) - 1)
                while work_tile.is_valid_tile:
                    m_block_sched = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                    num_m_blocks_cur = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.m_block_size)
                    if m_block_sched < num_m_blocks_cur:
                        m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                        num_n_blocks = self._dense_compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)
                        num_splits = cute.ceil_div(num_n_blocks, SPLIT)
                        split = Int32(0)
                        while split < num_splits:
                            pipeline_KV.consumer_wait(KV_sf_state)
                            st = sf_ctr
                            cute.arch.mbarrier_wait(SF_mbar_ptr + self.NUM_SF_STAGES + st, (sf_empty_bits >> st) & Int32(1))
                            sf_empty_bits = sf_empty_bits ^ (Int32(1) << st)
                            _tcgen05_fence_after()
                            for j in cutlass.range_constexpr(SPLIT):
                                col = Int32(self.tmem_k_sfa_base_offset) + st * Int32(self.sfa_stage_cols) + Int32(j * self.tmem_sfa_cols)
                                tCtKSFA = cute.make_tensor(sf_tmem_ptr + Int32(tmem_sf_offset_scale) * col, tCtKSFA_layout)
                                tiled_copy_s2t_ksfa, tCsKSFA_s2t, tCtKSFA_s2t = self.mainloop_s2t_copy_and_partition(sKScale, tCtKSFA)
                                cute.copy(tiled_copy_s2t_ksfa, tCsKSFA_s2t[(None, None, None, None, KV_sf_state.index * SPLIT + j)], tCtKSFA_s2t)
                            with cute.arch.elect_one():
                                tcgen05.commit(SF_mbar_ptr + st)
                            KV_sf_state.advance()
                            sf_ctr = sf_ctr + Int32(1) if sf_ctr < Int32(self.NUM_SF_STAGES - 1) else Int32(0)
                            split = split + 1
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

            # ---------------- MMA warp ----------------
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(Int32(self.tmem_alloc_cols), tmem_holding_buf)
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=1, number_of_threads=2 * self.WARP_SIZE)
                KV_mma_state = make_pipeline_state(PipelineUserType.Consumer, self.NUM_SPLIT_STAGES)
                sf_ctr = Int32(0)
                sf_full_bits = Int32(0)
                t_empty_bits = Int32((1 << SPLIT) - 1)
                tSrK = tiled_mma_qk.make_fragment_A(sK)
                tSrQ = tiled_mma_qk.make_fragment_B(sQ)
                while work_tile.is_valid_tile:
                    m_block_sched = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                    num_m_blocks_cur = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.m_block_size)
                    if m_block_sched < num_m_blocks_cur:
                        m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                        num_n_blocks = self._dense_compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)
                        num_splits = cute.ceil_div(num_n_blocks, SPLIT)
                        handle_Q = Q_consumer.wait_and_advance()
                        cute.copy(tiled_copy_s2t_qsfb, tCsQSFB_compact_s2t[(None, None, None, None, handle_Q.index)], tCtQSFB_compact_s2t)
                        split = Int32(0)
                        while split < num_splits:
                            pipeline_KV.consumer_wait(KV_mma_state)
                            st = sf_ctr
                            cute.arch.mbarrier_wait(SF_mbar_ptr + st, (sf_full_bits >> st) & Int32(1))
                            sf_full_bits = sf_full_bits ^ (Int32(1) << st)
                            _tcgen05_fence_after()
                            for w in cutlass.range_constexpr(SPLIT):
                                cute.arch.mbarrier_wait(T_mbar_ptr + SPLIT + w, (t_empty_bits >> w) & Int32(1))
                                t_empty_bits = t_empty_bits ^ Int32(1 << w)
                                _tcgen05_fence_after()
                                col = Int32(self.tmem_k_sfa_base_offset) + st * Int32(self.sfa_stage_cols) + Int32(w * self.tmem_sfa_cols)
                                tCtKSFA = cute.make_tensor(sf_tmem_ptr + Int32(tmem_sf_offset_scale) * col, tCtKSFA_layout)
                                tStS_w = cute.make_tensor(
                                    cute.make_ptr(Float32, w * self.tmem_s_stride, mem_space=cute.AddressSpace.tmem, assumed_align=16), tStS_ref.layout
                                )
                                for _kc in cutlass.range_constexpr(NKC):
                                    k_idx = (KV_mma_state.index * SPLIT + w) * NKC + _kc
                                    self._gemm_blockscaled_qk(
                                        blockscaled_tiled_mma_qk,
                                        tStS_w,
                                        tSrK[None, None, None, k_idx],
                                        tSrQ[None, None, None, handle_Q.index * NKC + _kc],
                                        sK[None, None, None, k_idx],
                                        sQ[None, None, None, handle_Q.index * NKC + _kc],
                                        tCtKSFA,
                                        tCtQSFB,
                                        accumulate_first=const_expr(_kc != 0),
                                        kphase_offset=const_expr(_kc * (self.k_block_size // self.mma_inst_k)),
                                    )
                                with cute.arch.elect_one():
                                    tcgen05.commit(T_mbar_ptr + w)
                            with cute.arch.elect_one():
                                tcgen05.commit(SF_mbar_ptr + self.NUM_SF_STAGES + st)
                            pipeline_KV.consumer_release(KV_mma_state)
                            KV_mma_state.advance()
                            sf_ctr = sf_ctr + Int32(1) if sf_ctr < Int32(self.NUM_SF_STAGES - 1) else Int32(0)
                            split = split + 1
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
        """One math warpgroup: owns packed columns ``[wg * cols, (wg + 1) * cols)`` of every block.

        ``wg`` is a runtime value: every warpgroup runs this one code copy (separate per-warpgroup
        copies overflowed the instruction cache for the small head groups).
        """
        SPLIT = self.SPLIT
        tidx_wg = tidx % self.WARPGROUP_SIZE
        ratio = Int32(self.ratio)
        tpw = self.q_tokens_per_tile // self.num_math_wg
        cols = self.m_block_size // self.num_math_wg
        t_full_bits = Int32(0)
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_repetition)), Float32)
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
                num_splits = cute.ceil_div(num_n_blocks, SPLIT)
                if const_expr(self.is_compressed_logits):
                    mOut_cur = mOut  # flat compact candidate buffer, addressed per row
                else:
                    mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=2)
                mDenom_cur = seqlen.offset_batch_Q(mDenom, batch_idx, dim=1) if const_expr(self.compute_lse) else None
                head = head_consumer.wait_and_advance()
                sW = cute.make_tensor(sPerHead.iterator + head.index * self.m_block_size + wg * cols, cute.make_layout((cols,), stride=(1,)))
                q_toks = [m_block * self.q_tokens_per_tile + wg * tpw + Int32(u) for u in range(tpw)]
                col_limits = [(q_causal_offset + q_toks[u] + Int32(1)) // ratio for u in range(tpw)]
                if num_splits > Int32(0):
                    cute.autovec_copy(sW, rW)
                head.release()
                self._math_tile(
                    wg,
                    tpw,
                    cols,
                    num_splits,
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
                # Every block barrier toggled once per split.
                t_full_bits = t_full_bits ^ ((num_splits & Int32(1)) * Int32((1 << SPLIT) - 1))
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
        if tidx_wg == 0:
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    @cute.jit
    def _reduce_token(self, tSrS, rW, u: cutlass.Constexpr[int]):
        """sum_h relu(s_h) * w_h of local token ``u`` (its qhpkv columns in ``tSrS`` / ``rW``), times 2 for
        the all-FADD tree (the caller folds the 1/2 into the output scale)."""
        qhpkv = self.qhead_per_kvhead
        acc0 = (Float32(0.0), Float32(0.0))
        acc1 = (Float32(0.0), Float32(0.0))
        for p in cutlass.range_constexpr(qhpkv // 2):
            wi = u * qhpkv + 2 * p
            v0 = tSrS[wi]
            v1 = tSrS[wi + 1]
            if const_expr(self.relu_fadd_all or p % 2 == 1):
                # relu(v) * w == (v + |v|) * w / 2 exactly
                v0, v1 = add_packed_f32x2((v0, v1), (cute.math.absf(v0), cute.math.absf(v1)))
            else:
                v0 = v0 if v0 > Float32(0.0) else Float32(0.0)
                v1 = v1 if v1 > Float32(0.0) else Float32(0.0)
            if const_expr(p % 2 == 0):
                acc0 = fma_packed_f32x2((v0, v1), (rW[wi], rW[wi + 1]), acc0)
            else:
                acc1 = fma_packed_f32x2((v0, v1), (rW[wi], rW[wi + 1]), acc1)
        if const_expr(self.relu_fadd_all):
            acc = add_packed_f32x2(acc0, acc1)
        else:
            acc = fma_packed_f32x2(acc1, (Float32(0.5), Float32(0.5)), acc0)
        return acc[0] + acc[1]

    @cute.jit
    def _math_tile(
        self,
        wg,
        tpw: cutlass.Constexpr[int],
        cols: cutlass.Constexpr[int],
        num_splits,
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
        """All splits of one work tile for this warpgroup's tokens, then their LSE."""
        SPLIT = self.SPLIT
        nsl = 4 // self.num_math_wg  # 32-column TMEM load slices owned by this warpgroup
        out_scale = Float32(softmax_scale) * Float32(0.5) if const_expr(self.relu_fadd_all) else Float32(softmax_scale)
        log2_e = Float32(math.log2(math.e))
        lims = []
        rows = []
        row_base = None  # compressed: one 64-bit row pointer + per-token Int32 offsets (fewer live registers)
        for u in cutlass.range_constexpr(tpw):
            lim = col_limits[u] if col_limits[u] < seqlen.seqlen_k else seqlen.seqlen_k
            lim = lim if q_toks[u] < seqlen.seqlen_q else Int32(0)
            lims.append(lim)
            q_row = q_toks[u] if q_toks[u] < seqlen.seqlen_q else Int32(0)
            if const_expr(self.is_compressed_logits):
                # Compact ratio-causal layout: batch slab base + prefix count of the earlier rows.
                q_global_start = Int64(q_causal_offset)
                if const_expr(mCandBatchOffsets is not None):
                    cand_batch_base = Int64(mCandBatchOffsets[batch_idx])
                else:
                    cand_batch_base = Int64(batch_idx) * self._row_offset_i64(Int64(seqlen.seqlen_q), q_global_start)
                row_off = cand_batch_base + self._row_offset_i64(Int64(q_row), q_global_start)
                if const_expr(u == 0):
                    row_off0 = row_off
                    row_base = cute.make_tensor(mOut_cur.iterator + row_off0, cute.make_layout((tpw * seqlen.seqlen_k,), stride=(1,)))
                # The tile's rows lie within tpw * seqlen_k entries of each other.
                rows.append(Int32(row_off - row_off0))
            else:
                rows.append(
                    cute.make_tensor(mOut_cur.iterator + cute.crd2idx((q_row, 0), mOut_cur.layout), cute.make_layout((mOut_cur.shape[1],), stride=(1,)))
                )
        lmax = [-Float32.inf for _ in range(tpw)]
        lsum = [Float32(0.0) for _ in range(tpw)]
        bufs = [cute.make_rmem_tensor(tSrS_shape, Float32), cute.make_rmem_tensor(tSrS_shape, Float32)]
        split = Int32(0)
        while split < num_splits:
            for j in cutlass.range_constexpr(SPLIT + 1):
                if const_expr(j < SPLIT):
                    cute.arch.mbarrier_wait(T_mbar_ptr + j, ((t_full_bits >> j) ^ split) & Int32(1))
                    _tcgen05_fence_after()
                    tStS_j = cute.make_tensor(
                        cute.make_ptr(Float32, j * self.tmem_s_stride + wg * cols, mem_space=cute.AddressSpace.tmem, assumed_align=16), tStS_ref.layout
                    )
                    tStS_t2r = thr_tmem_load.partition_S(tStS_j)
                    for k in cutlass.range_constexpr(nsl):
                        cute.copy(thr_tmem_load, tStS_t2r[None, k, None, None], bufs[j % 2][None, k, None, None])
                if const_expr(j > 0):
                    # Block j - 1 is already in registers (waited by the previous step's fence).
                    pos = kv_offset + (split * SPLIT + (j - 1)) * self.n_block_size
                    for u in cutlass.range_constexpr(tpw):
                        val = self._reduce_token(bufs[(j - 1) % 2], rW, u) * out_scale
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
                if const_expr(j < SPLIT):
                    cute.arch.fence_view_async_tmem_load()
                    _tcgen05_fence_before()
                    cute.arch.mbarrier_arrive(T_mbar_ptr + SPLIT + j)
            split = split + 1
        if const_expr(self.compute_lse):
            # Reduce each owned token's online (max, sum) across the warpgroup's 128 key rows.
            warp = tidx_wg // self.WARP_SIZE
            bar_id = 2 + wg  # named barriers: 0 = CTA sync, 1 = MMA/SF warps
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
                # sRed is reused by the next token / tile.
                cute.arch.barrier(barrier_id=bar_id, number_of_threads=self.WARPGROUP_SIZE)
