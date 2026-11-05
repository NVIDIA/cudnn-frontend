# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scaffolding shared by the SM100 warp-specialized score kernels.

The BF16 / block-scaled indexer scorers and the MXFP8 dense attention scorer run the same skeleton:
one CTA owns one packed m tile of 128 (token, head) rows (``128 // H`` query tokens) and walks its
ratio-causal K range under CLC persistent scheduling, with K as the MMA's A operand (128 keys) and
packed Q as the B operand.  A load warp streams Q / K by TMA and stages the tile's per-head vector
(W or the scaled LSE) in SMEM, an MMA warp fills TMEM accumulators, a CLC warp fetches the next
tile, and math warpgroups reduce the heads.  :class:`ScoreKernelSm100` holds that common
configuration and the host / device helpers; :class:`BlockScaledScoreSm100Mixin` adds the E8M0
scale plumbing (scale TMA, SMEM -> TMEM S2T copies) of the MXFP8 / MXFP4 kernels.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.pipeline import PipelineUserType, make_pipeline_state
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
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cudnn._cutlass_compat import LayoutEnum

add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd="rn")
fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd="rn")


@dsl_user_op
def tcgen05_fence_after(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tcgen05_fence_before(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


class ScoreKernelSm100:
    """Configuration and helpers of the SM100 packed-tile score kernels (SwapAB, CLC persistent)."""

    arch = 100
    WARP_SIZE = 32
    WARPGROUP_SIZE = 128
    max_q_tokens_per_tile = 32

    def __init__(
        self,
        head_dim: int,
        qhead_per_kvhead: int,
        m_block_size: int,
        n_block_size: int,
        kv_stage: int,
        k_block_size: int | None = None,
        ratio: int = 1,
        is_varlen: bool = False,
        page_size: int = 0,
    ):
        assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
        # Paged K streams each n block as min(page, n_block) row boxes, so a box never straddles two pages.
        # Boxes stack in 8-row swizzle atoms, hence the multiple of 8.
        assert page_size % 8 == 0 and (
            page_size % n_block_size == 0 or n_block_size % page_size == 0
        ), f"page_size ({page_size}) must be a multiple of 8 that divides n_block_size ({n_block_size}) or is a multiple of it"
        self.head_dim = head_dim
        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.kv_stage = kv_stage
        self.ratio = ratio
        self.is_varlen = is_varlen
        self.page_size = page_size
        self.head_dim_padded = int(math.ceil(head_dim / 16) * 16)
        self.k_block_size = k_block_size if k_block_size is not None else self.head_dim_padded
        assert (
            self.head_dim_padded % self.k_block_size == 0
        ), f"head_dim_padded ({self.head_dim_padded}) must be a multiple of k_block_size ({self.k_block_size})"
        self.num_k_chunks = self.head_dim_padded // self.k_block_size
        self.q_tokens_per_tile = m_block_size // qhead_per_kvhead
        assert (
            self.q_tokens_per_tile <= self.max_q_tokens_per_tile
        ), f"q_tokens_per_tile ({self.q_tokens_per_tile}) must not exceed {self.max_q_tokens_per_tile}"
        self.tmem_repetition = self.m_block_size // 4
        # SwapAB: K is A (M = n_block keys), packed Q is B (N = m_block (token, head) rows).
        self.mma_tiler_qk = (n_block_size, m_block_size, self.k_block_size)
        self.qk_acc_dtype = Float32
        self.load_warp_id, self.mma_warp_id, self.sched_warp_id = 0, 1, 2
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)

    def _set_register_split(self, num_math_wg: int):
        """Warps 0-3 (load / MMA / CLC / aux) plus ``num_math_wg`` math warpgroups, and their setmaxnreg split.

        setmaxnreg only redistributes the CTA's launch allocation (floor8(64K / threads) per thread);
        asking for more than that pool hangs the increasing warpgroups.
        """
        self.num_math_wg = num_math_wg
        self.num_warps = 4 + 4 * num_math_wg
        self.threads_per_cta = self.WARP_SIZE * self.num_warps
        self.num_regs_load = 56
        launch_regs = (65536 // self.threads_per_cta) // 8 * 8
        self.num_regs_epilogue = min(
            256,
            ((launch_regs * self.threads_per_cta - self.WARPGROUP_SIZE * self.num_regs_load) // (self.WARPGROUP_SIZE * num_math_wg)) // 8 * 8,
        )

    # ------------------------------------------------------------------ host helpers
    def _pack_qk(self, mQ: cute.Tensor, mK: cute.Tensor, is_varlen: bool, is_paged: bool = False):
        """BSHD (bs, s, h, d) -> (s, d, h, bs), THD (t, h, d) -> (t, d, h), paged K pool -> (page, d, h, pages);
        then fold the heads into Q's M mode: packed row = token * H + head."""
        Q_layout_transpose = [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0]
        K_layout_transpose = [0, 2, 1] if const_expr(is_varlen and not is_paged) else [1, 3, 2, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=K_layout_transpose))
        seqlen_q = cute.size(mQ.shape[0])
        shape_Q_packed = ((self.qhead_per_kvhead, mQ.shape[0]), mQ.shape[1], 1, *mQ.shape[3:])
        stride_Q_packed = ((mQ.stride[2], mQ.stride[0]), mQ.stride[1], mQ.stride[2] * self.qhead_per_kvhead, *mQ.stride[3:])
        mQ = cute.make_tensor(mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed))
        self.q_major_mode = LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = LayoutEnum.from_tensor(mK).mma_major_mode()
        return mQ, mK, seqlen_q

    @staticmethod
    def _rows_first(t: cute.Tensor | None, is_varlen: bool, rank: int):
        """Per-token tensors: BSH (bs, s, h) -> (s, h, bs) / BS (bs, s) -> (s, bs); THD layouts stay as they are."""
        if const_expr(t is None or is_varlen):
            return t
        mode = [1, 2, 0] if const_expr(rank == 3) else [1, 0]
        return cute.make_tensor(t.iterator, cute.select(t.layout, mode=mode))

    def _tile_scheduler(self, seqlen_q, batch_size):
        """One CLC work tile per packed m block and batch."""
        num_m_blocks = cute.ceil_div(seqlen_q * self.qhead_per_kvhead, self.m_block_size)
        params = utils.ClcDynamicPersistentTileSchedulerParams((num_m_blocks, 1, batch_size), (*self.cluster_shape_mn, 1))
        return params, utils.ClcDynamicPersistentTileScheduler.get_grid_shape(params)

    # ------------------------------------------------------------------ device helpers
    @cute.jit
    def _dense_compute_n_blocks(self, m_block, seqlen_q, seqlen_k, q_causal_offset):
        """Dense K blocks that can contain at least one valid KV column of the tile's tokens."""
        ratio = Int32(self.ratio)
        q_token_base = m_block * self.q_tokens_per_tile
        q_token_last = q_token_base + Int32(self.q_tokens_per_tile - 1)
        q_token_last = q_token_last if q_token_last < seqlen_q else seqlen_q - Int32(1)
        max_kv_needed = (q_causal_offset + q_token_last + Int32(1)) // ratio
        max_kv_needed = max_kv_needed if max_kv_needed > Int32(0) else Int32(0)
        max_kv_needed = max_kv_needed if max_kv_needed < seqlen_k else seqlen_k
        return cute.ceil_div(max_kv_needed, self.n_block_size)

    @cute.jit
    def _tile_info(self, work_tile, SeqlenInfoCls, mQCausalOffsets):
        """``(valid, m_block, seqlen, batch_idx, q_causal_offset)`` of a work tile.  Tiles are walked longest first
        (the last m block of a sequence sees the most keys); a valid tile's key-block count is
        ``_dense_compute_n_blocks`` (left to the caller's ``if valid`` so invalid tiles skip it)."""
        m_block_sched = work_tile.tile_idx[0]
        batch_idx = work_tile.tile_idx[2]
        seqlen = SeqlenInfoCls(batch_idx)
        q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
        num_m_blocks_cur = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.m_block_size)
        valid = m_block_sched < num_m_blocks_cur
        m_block = num_m_blocks_cur - Int32(1) - m_block_sched
        return valid, m_block, seqlen, batch_idx, q_causal_offset

    @cute.jit
    def _sched_warp(self, work_tile, tile_sched, clc_pipeline, clc_consumer_state):
        """CLC scheduler warp: request the next work tile while the current one runs."""
        clc_producer_state = make_pipeline_state(PipelineUserType.ProducerConsumer, self.num_clc_stage)
        while work_tile.is_valid_tile:
            clc_pipeline.producer_acquire(clc_producer_state)
            tile_sched.advance_to_next_work(clc_pipeline.producer_get_barrier(clc_producer_state))
            clc_producer_state.advance()
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
        clc_pipeline.producer_tail(clc_producer_state)

    @cute.jit
    def _stage_per_head(self, sBuf, off, mPerHead_cur, m_block, seqlen_q, lane_id, scale: cutlass.Constexpr = None):
        """One warp stages the tile's per-head vector (packed row order, ``scale``-multiplied, 0 past the sequence)."""
        for ri in cutlass.range_constexpr(self.m_block_size // self.WARP_SIZE):
            row = ri * self.WARP_SIZE + lane_id
            m_packed_idx = m_block * self.m_block_size + row
            m_idx = m_packed_idx // self.qhead_per_kvhead
            h_idx = m_packed_idx - m_idx * self.qhead_per_kvhead
            if m_idx < seqlen_q:
                v = Float32(mPerHead_cur[m_idx, h_idx])
                if const_expr(scale is not None):
                    v = v * Float32(scale)
                sBuf[off + row] = v
            else:
                sBuf[off + row] = Float32(0.0)


class BlockScaledScoreSm100Mixin:
    """E8M0 block-scale plumbing of the MXFP8 / MXFP4 score kernels (mixed into a :class:`ScoreKernelSm100`)."""

    sf_vec_size = 32

    @staticmethod
    def _qk_tensor(t: cute.Tensor) -> cute.Tensor:
        """Q or K as the kernel reads it, from the tensor it is launched with (the same tensor by default)."""
        return t

    def _make_data_tiled_mma(self, blockscaled_tiled_mma_qk, cta_group):
        """Tiled MMA that lays out and partitions the Q/K data tiles (the unscaled MMA of the QK tile by default)."""
        return _make_trivial_tiled_mma(self.q_dtype, self.q_dtype, self.k_major_mode, self.q_major_mode, self.qk_acc_dtype, cta_group, self.mma_tiler_qk[:2])

    def _block_scaled_operands(self, mQ, mK, mQScale, mKScale, is_varlen, *, scale_tile_k, q_data_stages, q_scale_stages, k_data_stages, k_scale_stages):
        """Host side: packed Q/K, the block-scaled and data tiled MMAs, SMEM layouts and TMA atoms of Q, K and their
        E8M0 scale tiles (``scale_tile_k`` wide along the head dim).  Returns a dict; also sets ``self.tma_bytes``."""
        mQ = self._qk_tensor(mQ)
        mK = self._qk_tensor(mK)
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.sf_dtype = mQScale.element_type
        mQ, mK, seqlen_q = self._pack_qk(mQ, mK, is_varlen)

        # Packed scale tensors: the second mode is the caller's atom-aligned MN extent.
        scale_l_size = cute.size(mQScale.shape[0])
        mQScale = cute.make_tensor(
            mQScale.iterator,
            _blockscaled_layout.tile_atom_to_shape_SF((cute.size(mQScale.shape[1]), self.head_dim_padded, scale_l_size), self.sf_vec_size),
        )
        mKScale = cute.make_tensor(
            mKScale.iterator,
            _blockscaled_layout.tile_atom_to_shape_SF((cute.size(mKScale.shape[1]), self.head_dim_padded, scale_l_size), self.sf_vec_size),
        )

        cta_group = tcgen05.CtaGroup.ONE
        blockscaled_tiled_mma_qk = _make_blockscaled_trivial_tiled_mma(
            self.k_dtype, self.k_dtype, self.k_major_mode, self.q_major_mode, self.sf_dtype, self.sf_vec_size, cta_group, self.mma_tiler_qk[:2]
        )
        tiled_mma_qk = self._make_data_tiled_mma(blockscaled_tiled_mma_qk, cta_group)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))

        sK_layout = _make_smem_layout_a(tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, k_data_stages)
        sQ_layout = _make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, q_data_stages)
        scale_mma_tiler_qk = (self.n_block_size, self.m_block_size, scale_tile_k)
        sKScale_layout = _blockscaled_layout.make_smem_layout_sfa(blockscaled_tiled_mma_qk, scale_mma_tiler_qk, self.sf_vec_size, k_scale_stages)
        sQScale_layout = _blockscaled_layout.make_smem_layout_sfb(blockscaled_tiled_mma_qk, scale_mma_tiler_qk, self.sf_vec_size, q_scale_stages)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mQ, cute.select(sQ_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk, self.cluster_layout_vmnk.shape
        )
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mK, cute.select(sK_layout, mode=[0, 1, 2]), self.mma_tiler_qk, tiled_mma_qk, self.cluster_layout_vmnk.shape
        )
        # Scale TMA moves the packed E8M0 bytes as Int16 (the blockscaled layouts store two per compact element).
        k_scale_smem_layout = cute.slice_(sKScale_layout, (None, None, None, 0))
        q_scale_smem_layout = cute.slice_(sQScale_layout, (None, None, None, 0))
        tma_atom_KScale, mKScale = cute.nvgpu.make_tiled_tma_atom_A(
            _cluster_shape_to_tma_atom_A(self.cluster_shape_mn, blockscaled_tiled_mma_qk.thr_id),
            mKScale,
            k_scale_smem_layout,
            scale_mma_tiler_qk,
            blockscaled_tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_QScale, mQScale = cute.nvgpu.make_tiled_tma_atom_B(
            _cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, blockscaled_tiled_mma_qk.thr_id),
            mQScale,
            q_scale_smem_layout,
            scale_mma_tiler_qk,
            blockscaled_tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        self.tma_bytes = {
            "q_data": cute.size_in_bytes(self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2])),
            "k_data": cute.size_in_bytes(self.k_dtype, cute.select(sK_layout, mode=[0, 1, 2])),
            "q_scale": cute.size_in_bytes(self.sf_dtype, q_scale_smem_layout),
            "k_scale": cute.size_in_bytes(self.sf_dtype, k_scale_smem_layout),
        }
        return dict(
            mQ=mQ,
            mK=mK,
            mQScale=mQScale,
            mKScale=mKScale,
            seqlen_q=seqlen_q,
            tiled_mma_qk=tiled_mma_qk,
            blockscaled_tiled_mma_qk=blockscaled_tiled_mma_qk,
            sQ_layout=sQ_layout,
            sK_layout=sK_layout,
            sQScale_layout=sQScale_layout,
            sKScale_layout=sKScale_layout,
            tma_atom_Q=tma_atom_Q,
            tma_atom_K=tma_atom_K,
            tma_atom_QScale=tma_atom_QScale,
            tma_atom_KScale=tma_atom_KScale,
        )

    @cute.jit
    def _scale_offsets(self, batch_idx, scale_batch_idx, q_scale_m_block, k_scale_n_block, mCuSeqlensQScalePadded, mCuSeqlensKScalePadded):
        """``(scale_l_idx, scale_batch_idx, q_scale_m_block, k_scale_n_block)`` of a work tile: BSHD indexes the scale
        tensors by batch; THD shares one packed scale slab and offsets it by the padded scale prefixes (re-read only
        when the batch changes)."""
        scale_l_idx = batch_idx
        if const_expr(mCuSeqlensQScalePadded is not None):
            scale_l_idx = 0
            if batch_idx != scale_batch_idx:
                q_scale_m_block = mCuSeqlensQScalePadded[batch_idx] * self.qhead_per_kvhead // self.m_block_size
                k_scale_n_block = mCuSeqlensKScalePadded[batch_idx] // self.n_block_size
                scale_batch_idx = batch_idx
        return scale_l_idx, scale_batch_idx, q_scale_m_block, k_scale_n_block

    @cute.jit
    def _s2t_copy_and_partition(self, sSF: cute.Tensor, tSF: cute.Tensor) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """tcgen05 S2T copy of a packed E8M0 scale tile from SMEM into a TMEM SF field (zeros of the blockscaled
        layout's sparse strides filtered out)."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, thr_copy_s2t.partition_S(tCsSF_compact))
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t
