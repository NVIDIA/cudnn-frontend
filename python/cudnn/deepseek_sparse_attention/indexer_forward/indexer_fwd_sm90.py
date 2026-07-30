# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 indexer forward kernel - direct CuTeDSL port of the old C++ path.

This kernel intentionally keeps the SM90 C++ Q/K staging topology:
  * 256 threads / 2 warpgroups
  * WG0 warp0 issues Q/K TMA loads and loads weights
  * WG1 runs two 64x64x16 WGMMA stages and writes the reduced scores

The algorithm computes the same score tensor as the C++ kernel:
    sm_scale * sum_h relu(Q_h @ K^T) * W_h
with ratio causal masking against compressed-KV positions. Reduced BF16 scores
are written directly from registers to global memory.
"""

import math
import operator
from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum

from cudnn.deepseek_sparse_attention.utils import copy as copy_utils
from cudnn.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK
from cudnn.deepseek_sparse_attention.utils.sm90 import mma as sm90_mma
from cudnn.deepseek_sparse_attention.utils.sm90 import primitives as sm90_ops
from cudnn.deepseek_sparse_attention.utils.sm90.mma import gemm_w_idx


def _mma_partition_fragment_AB(
    thr_mma: cute.ThrMma,
    sA: cute.Tensor,
    sB: cute.Tensor,
    swap_AB: bool,
):
    if const_expr(not swap_AB):
        return thr_mma.make_fragment_A(thr_mma.partition_A(sA)), thr_mma.make_fragment_B(thr_mma.partition_B(sB))
    return thr_mma.make_fragment_B(thr_mma.partition_B(sA)), thr_mma.make_fragment_A(thr_mma.partition_A(sB))


class IndexerForwardSm90:
    """Direct CuTe DSL translation of the legacy SM90 forward kernel."""

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int = 128,
        qhead_per_kvhead: int = 64,
        ratio: int = 4,
        is_varlen: bool = False,
        clean_logits: bool = True,
    ):
        assert head_dim == 128, f"SM90 direct forward supports head_dim=128, got {head_dim}"
        assert qhead_per_kvhead in (16, 32, 64), f"SM90 direct forward supports qhpkv in (16, 32, 64), got {qhead_per_kvhead}"
        assert ratio >= 1, f"ratio must be >=1, got {ratio}"
        self.dtype = dtype
        self.head_dim = head_dim
        self.tile_hdim = int(math.ceil(head_dim / 16) * 16)
        self.qhead_per_kvhead = qhead_per_kvhead
        self.ratio = ratio
        self.is_varlen = is_varlen
        self.clean_logits = clean_logits

        self.tile_m = 128
        self.tile_n = 64
        self.q_stages = 2
        self.kv_stages = 2
        self.q_per_stage = self.tile_m // self.q_stages
        self.q_tokens_per_tile = self.tile_m // self.qhead_per_kvhead
        self.q_tokens_per_stage = self.q_tokens_per_tile // self.q_stages
        assert self.q_per_stage == 64
        assert self.q_tokens_per_stage * self.qhead_per_kvhead == self.q_per_stage
        # qh16 packs eight query tokens into one CTA. Their ratio-causal limits
        # can straddle a 64-column boundary, so both rightmost N blocks need
        # elementwise masking. Keep the existing one-block path unchanged for
        # qh32/qh64 so those specializations lower exactly as before.
        self.mask_two_rightmost_nblocks = self.qhead_per_kvhead == 16

        self.num_threads = 256
        self.num_threads_per_warp_group = 128
        self.swap_AB = True
        self.num_acc_n_rows_per_thread = self.tile_n // 32

    def _setup_attributes(self):
        self.sQ_layout_single = sm90_mma.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.q_per_stage, self.tile_hdim),
            stage=None,
        )
        self.sQ_layout_staged = sm90_mma.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.q_per_stage, self.tile_hdim),
            stage=self.q_stages,
        )
        self.sKV_layout_single = sm90_mma.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), stage=None)
        self.sKV_layout_staged = sm90_mma.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim),
            stage=self.kv_stages,
        )

    def _get_tiled_mma(self):
        return sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.tile_n, self.q_per_stage),
        )

    def _get_shared_storage_cls(self):
        sQ_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout_staged)], 1024]
        sKV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sKV_layout_staged)], 1024]
        sW_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, self.tile_m], 128]

        @cute.struct
        class SharedStorage:
            mbar_Q0: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_Q1: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_KV0: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_KV1: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_KVEmpty0: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_KVEmpty1: cute.struct.MemRange[cutlass.Int64, 2]
            sW: sW_struct
            sQ: sQ_struct
            sKV: sKV_struct

        return SharedStorage

    @cute.jit
    def _compute_n_blocks(self, m_block: Int32, seqlen_q: Int32, seqlen_k: Int32, q_causal_offset: Int32) -> Int32:
        last_q_token = (m_block + Int32(1)) * Int32(self.q_tokens_per_tile) - Int32(1)
        kv_limit = (q_causal_offset + last_q_token + Int32(1)) // Int32(self.ratio)
        kv_limit = kv_limit if kv_limit < seqlen_k else seqlen_k
        kv_limit = kv_limit if kv_limit > Int32(0) else Int32(0)
        return cute.ceil_div(kv_limit, self.tile_n)

    @cute.jit
    def _iter_n_block(self, iter_idx: Int32, n_block_max: Int32) -> Int32:
        local_count = n_block_max if n_block_max < Int32(3) else Int32(3)
        return n_block_max - Int32(1) - iter_idx if iter_idx < local_count else iter_idx - local_count

    @cute.jit
    def _issue_tma(
        self,
        gmem_cur: cute.Tensor,
        smem_tile: cute.Tensor,
        block_idx: Int32,
        tma_atom: cute.CopyAtom,
        mbar_ptr,
        copy_bytes: cutlass.Constexpr[int],
        tile_rows: cutlass.Constexpr[int],
    ):
        g_tile = cute.local_tile(gmem_cur, (tile_rows, self.tile_hdim), (block_idx, 0))
        load_fn, _, _ = copy_utils.tma_get_copy_fn(tma_atom, 0, cute.make_layout(1), g_tile, smem_tile, single_stage=True)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, copy_bytes)
        load_fn(tma_bar_ptr=mbar_ptr)

    @cute.jit
    def _load_weights(
        self,
        mW_cur: cute.Tensor,
        sW: cute.Tensor,
        m_block: Int32,
        seqlen_q: Int32,
        lane_id: Int32,
    ):
        rows_per_thread = (self.tile_m + 31) // 32
        for ri in cutlass.range_constexpr(rows_per_thread):
            row = ri * 32 + lane_id
            if row < self.tile_m:
                m_packed = m_block * self.tile_m + row
                q_idx = m_packed // self.qhead_per_kvhead
                if q_idx < seqlen_q:
                    sW[row] = mW_cur[m_packed]
                else:
                    sW[row] = self.dtype(0.0)

    @cute.jit
    def _epilogue_store_to_gmem(
        self,
        q_stage_idx: cutlass.Constexpr[int],
        acc_S: cute.Tensor,
        tWeights: cute.Tensor,
        mOut: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        apply_causal_mask: Boolean,
        seqlen_q: Int32,
        seqlen_k: Int32,
        max_seqlen_k: Int32,
        q_batch_offset: Int32,
        batch_idx: Int32,
        q_causal_offset: Int32,
        sm_scale: Float32,
    ):
        acc_mn = sm90_ops.make_acc_tensor_mn_view(acc_S)
        kNRows = cute.size(acc_mn, mode=[0])
        kNCols = cute.size(acc_mn, mode=[1])
        kColsPerQToken = kNCols // self.q_tokens_per_stage

        lane = cute.arch.lane_idx()
        t0 = lane % 4
        t1 = lane // 4
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        m_base = t1 + warp_idx_in_wg * 16
        q_token_base = (self.q_stages * m_block + q_stage_idx) * self.q_tokens_per_stage

        ps = cute.make_rmem_tensor((kNRows, self.q_tokens_per_stage), Float32)
        ps.fill(Float32(0.0))

        for qi in cutlass.range_constexpr(self.q_tokens_per_stage):
            for hi in cutlass.range(kColsPerQToken, unroll_full=True):
                ni = qi * kColsPerQToken + hi
                w = Float32(tWeights[ni])
                for mi in cutlass.range(kNRows, unroll_full=True):
                    a = acc_mn[mi, ni]
                    ps[mi, qi] = ps[mi, qi] + cute.arch.fmax(a, Float32(0.0)) * w

        for mi in cutlass.range(kNRows, unroll_full=True):
            for qi in cutlass.range_constexpr(self.q_tokens_per_stage):
                ps[mi, qi] = sm90_ops.warp_reduce(ps[mi, qi], operator.add, width=4) * sm_scale

        if t0 == 0:
            for qi in cutlass.range_constexpr(self.q_tokens_per_stage):
                for mi in cutlass.range(kNRows, unroll_full=True):
                    kv_m = m_base + mi * 8
                    q_local = q_token_base + qi
                    k_local = n_block * self.tile_n + kv_m
                    score = ps[mi, qi]
                    should_store = Boolean(True)
                    if apply_causal_mask:
                        col_lim = (q_causal_offset + q_local + Int32(1)) // Int32(self.ratio)
                        if k_local >= seqlen_k or k_local >= col_lim:
                            if const_expr(self.clean_logits):
                                score = -Float32.inf
                            else:
                                should_store = Boolean(False)
                    elif k_local >= seqlen_k:
                        should_store = Boolean(False)

                    if should_store and q_local < seqlen_q and k_local < max_seqlen_k:
                        if const_expr(self.is_varlen):
                            mOut[q_batch_offset + q_local, k_local] = score
                        else:
                            mOut[batch_idx, q_local, k_local] = score

    @cute.jit
    def producer(
        self,
        mQ: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        mW: cute.Tensor,
        sQ_0: cute.Tensor,
        sQ_1: cute.Tensor,
        sKV_0: cute.Tensor,
        sKV_1: cute.Tensor,
        sW: cute.Tensor,
        mbar_Q0_ptr,
        mbar_Q1_ptr,
        mbar_KV0_ptr,
        mbar_KV1_ptr,
        mbar_KVEmpty0_ptr,
        mbar_KVEmpty1_ptr,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mQCausalOffsets: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx % 4
        lane_id = cute.arch.lane_idx()
        block_x, head_idx, batch_idx = cute.arch.block_idx()

        seqlen = SeqlenInfoQK.create(
            batch_idx,
            max_seqlen_q,
            max_seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            None,
            None,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
        num_m_blocks = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.tile_m)
        if block_x < num_m_blocks:
            m_block = num_m_blocks - Int32(1) - block_x
            n_block_max = self._compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)

            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx]
            mW_cur = seqlen.offset_batch_Q(mW, batch_idx, dim=2)[None, head_idx]

            q0_phase = Int32(0)
            q1_phase = Int32(0)
            kve0_phase = Int32(0)
            kve1_phase = Int32(0)

            if warp_idx_in_wg == 0:
                self._load_weights(mW_cur, sW, m_block, seqlen.seqlen_q, lane_id)
                cute.arch.fence_view_async_shared()
                self._issue_tma(
                    mQ_cur,
                    sQ_0,
                    self.q_stages * m_block,
                    tma_atom_Q,
                    mbar_Q0_ptr,
                    self.tma_copy_bytes_Q,
                    self.q_per_stage,
                )
                self._issue_tma(
                    mQ_cur,
                    sQ_1,
                    self.q_stages * m_block + Int32(1),
                    tma_atom_Q,
                    mbar_Q1_ptr,
                    self.tma_copy_bytes_Q,
                    self.q_per_stage,
                )

                iter_idx = Int32(0)
                while iter_idx < n_block_max:
                    stage = iter_idx & Int32(1)
                    n_block = self._iter_n_block(iter_idx, n_block_max)
                    if iter_idx >= Int32(2):
                        if stage == Int32(0):
                            cute.arch.mbarrier_wait(mbar_KVEmpty0_ptr, kve0_phase)
                            kve0_phase = kve0_phase ^ Int32(1)
                        else:
                            cute.arch.mbarrier_wait(mbar_KVEmpty1_ptr, kve1_phase)
                            kve1_phase = kve1_phase ^ Int32(1)
                    if stage == Int32(0):
                        self._issue_tma(
                            mK_cur,
                            sKV_0,
                            n_block,
                            tma_atom_K,
                            mbar_KV0_ptr,
                            self.tma_copy_bytes_K,
                            self.tile_n,
                        )
                    else:
                        self._issue_tma(
                            mK_cur,
                            sKV_1,
                            n_block,
                            tma_atom_K,
                            mbar_KV1_ptr,
                            self.tma_copy_bytes_K,
                            self.tile_n,
                        )
                    iter_idx = iter_idx + Int32(1)

                if n_block_max >= Int32(1):
                    cute.arch.mbarrier_wait(mbar_KVEmpty0_ptr, kve0_phase)
                if n_block_max >= Int32(2):
                    cute.arch.mbarrier_wait(mbar_KVEmpty1_ptr, kve1_phase)

    @cute.jit
    def consumer(
        self,
        tiled_mma_QK: cute.TiledMma,
        sQ_0: cute.Tensor,
        sQ_1: cute.Tensor,
        sKV_staged: cute.Tensor,
        sW: cute.Tensor,
        mOut: cute.Tensor,
        mbar_Q0_ptr,
        mbar_Q1_ptr,
        mbar_KV0_ptr,
        mbar_KV1_ptr,
        mbar_KVEmpty0_ptr,
        mbar_KVEmpty1_ptr,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mQCausalOffsets: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        sm_scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        wg_tidx = tidx % self.num_threads_per_warp_group
        block_x, head_idx, batch_idx = cute.arch.block_idx()

        seqlen = SeqlenInfoQK.create(
            batch_idx,
            max_seqlen_q,
            max_seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            None,
            None,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
        num_m_blocks = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.tile_m)
        if block_x < num_m_blocks:
            m_block = num_m_blocks - Int32(1) - block_x
            n_block_max = self._compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)

            thr_mma = tiled_mma_QK.get_slice(wg_tidx)
            tSrQ0, tSrK = _mma_partition_fragment_AB(thr_mma, sQ_0, sKV_staged, self.swap_AB)
            tSrQ1, _ = _mma_partition_fragment_AB(thr_mma, sQ_1, sKV_staged, self.swap_AB)
            acc_shape = tiled_mma_QK.partition_shape_C((self.tile_n, self.q_per_stage))

            sW0_mma = cute.make_tensor(
                sW.iterator,
                cute.make_layout((self.q_per_stage, self.tile_n), stride=(1, 0)),
            )
            sW1_mma = cute.make_tensor(
                sW.iterator + self.q_per_stage,
                cute.make_layout((self.q_per_stage, self.tile_n), stride=(1, 0)),
            )
            if const_expr(self.swap_AB):
                sW0_mma = sm90_ops.transpose_view(sW0_mma)
                sW1_mma = sm90_ops.transpose_view(sW1_mma)
            weights_slice = (None, 0) if const_expr(not self.swap_AB) else (0, None)
            tWeights0 = sm90_ops.make_acc_tensor_mn_view(thr_mma.partition_C(sW0_mma))[weights_slice]
            tWeights1 = sm90_ops.make_acc_tensor_mn_view(thr_mma.partition_C(sW1_mma))[weights_slice]

            q0_phase = Int32(0)
            q1_phase = Int32(0)
            kv0_phase = Int32(0)
            kv1_phase = Int32(0)
            warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

            cute.arch.mbarrier_wait(mbar_Q0_ptr, q0_phase)
            cute.arch.mbarrier_wait(mbar_Q1_ptr, q1_phase)

            iter_idx = Int32(0)
            while iter_idx < n_block_max:
                stage = iter_idx & Int32(1)
                n_block = self._iter_n_block(iter_idx, n_block_max)
                if stage == Int32(0):
                    cute.arch.mbarrier_wait(mbar_KV0_ptr, kv0_phase)
                    kv0_phase = kv0_phase ^ Int32(1)
                else:
                    cute.arch.mbarrier_wait(mbar_KV1_ptr, kv1_phase)
                    kv1_phase = kv1_phase ^ Int32(1)

                acc0 = cute.make_rmem_tensor(acc_shape, Float32)
                acc1 = cute.make_rmem_tensor(acc_shape, Float32)
                gemm_w_idx(
                    tiled_mma_QK,
                    acc0,
                    tSrQ0,
                    tSrK,
                    zero_init=Boolean(True),
                    B_idx=stage,
                    wg_wait=-1,
                    swap_AB=self.swap_AB,
                )
                gemm_w_idx(
                    tiled_mma_QK,
                    acc1,
                    tSrQ1,
                    tSrK,
                    zero_init=Boolean(True),
                    B_idx=stage,
                    wg_wait=-1,
                    swap_AB=self.swap_AB,
                )
                warpgroup.wait_group(0)

                if stage == Int32(0):
                    if warp_idx_in_wg == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar_KVEmpty0_ptr, arrive_count=128)
                else:
                    if warp_idx_in_wg == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar_KVEmpty1_ptr, arrive_count=128)

                apply_causal_mask = Boolean(iter_idx < Int32(2)) if const_expr(self.mask_two_rightmost_nblocks) else Boolean(iter_idx == Int32(0))
                self._epilogue_store_to_gmem(
                    0,
                    acc0,
                    tWeights0,
                    mOut,
                    m_block,
                    n_block,
                    apply_causal_mask,
                    seqlen.seqlen_q,
                    seqlen.seqlen_k,
                    max_seqlen_k,
                    seqlen.offset_q,
                    batch_idx,
                    q_causal_offset,
                    sm_scale,
                )
                self._epilogue_store_to_gmem(
                    1,
                    acc1,
                    tWeights1,
                    mOut,
                    m_block,
                    n_block,
                    apply_causal_mask,
                    seqlen.seqlen_q,
                    seqlen.seqlen_k,
                    max_seqlen_k,
                    seqlen.offset_q,
                    batch_idx,
                    q_causal_offset,
                    sm_scale,
                )

                iter_idx = iter_idx + Int32(1)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mW: cute.Tensor,
        mOut: cute.Tensor,
        n_heads_kv: Int32,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        sm_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mQCausalOffsets: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        is_varlen = mCuSeqlensQ is not None
        if const_expr(is_varlen):
            assert self.is_varlen
        else:
            assert not self.is_varlen
        assert mQ.element_type == self.dtype
        assert mK.element_type == self.dtype
        assert mW.element_type == self.dtype

        def _assume_strides(t):
            divby = 128 // t.element_type.width
            new_strides = []
            for s in t.stride[:-1]:
                if const_expr(isinstance(s, int)):
                    new_strides.append(s)
                else:
                    new_strides.append(cute.assume(s, divby=divby))
            new_strides.append(t.stride[-1])
            return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=tuple(new_strides)))

        mQ = _assume_strides(mQ)
        mK = _assume_strides(mK)
        mW = _assume_strides(mW)
        mOut = _assume_strides(mOut)

        mQ = sm90_ops.select(mQ, [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0])
        mK = sm90_ops.select(mK, [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0])
        mW = sm90_ops.select(mW, [0, 1] if const_expr(is_varlen) else [1, 2, 0])

        qhpkv = self.qhead_per_kvhead
        num_head_kv = n_heads_kv
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                ((qhpkv, mQ.shape[0]), mQ.shape[1], num_head_kv, *mQ.shape[3:]),
                stride=(
                    (mQ.stride[2], mQ.stride[0]),
                    mQ.stride[1],
                    mQ.stride[2] * qhpkv,
                    *mQ.stride[3:],
                ),
            ),
        )
        mW = cute.make_tensor(
            mW.iterator,
            cute.make_layout(
                ((qhpkv, mW.shape[0]), num_head_kv, *mW.shape[2:]),
                stride=(
                    (mW.stride[1], mW.stride[0]),
                    mW.stride[1] * qhpkv,
                    *mW.stride[2:],
                ),
            ),
        )

        self._setup_attributes()
        tiled_mma_QK = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes_Q = cute.size_in_bytes(self.dtype, cute.select(self.sQ_layout_single, mode=[0, 1]))
        self.tma_copy_bytes_K = cute.size_in_bytes(self.dtype, cute.select(self.sKV_layout_single, mode=[0, 1]))

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            self.sQ_layout_single,
            (self.q_per_stage, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            self.sKV_layout_single,
            (self.tile_n, self.tile_hdim),
        )

        batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1 if const_expr(is_varlen) else cute.size(mQ.shape[3])
        grid_x = cute.ceil_div(max_seqlen_q * self.qhead_per_kvhead, self.tile_m)
        grid = (grid_x, num_head_kv, batch_size)

        self.kernel(
            tma_tensor_Q,
            tma_atom_Q,
            tma_tensor_K,
            tma_atom_K,
            mW,
            mOut,
            self.sQ_layout_staged,
            self.sKV_layout_staged,
            tiled_mma_QK,
            SharedStorage,
            mCuSeqlensQ,
            mCuSeqlensK,
            mQCausalOffsets,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        mW: cute.Tensor,
        mOut: cute.Tensor,
        sQ_layout_staged: cute.ComposedLayout,
        sKV_layout_staged: cute.ComposedLayout,
        tiled_mma_QK: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mQCausalOffsets: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        sm_scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        mbar_Q0_ptr = storage.mbar_Q0.data_ptr()
        mbar_Q1_ptr = storage.mbar_Q1.data_ptr()
        mbar_KV0_ptr = storage.mbar_KV0.data_ptr()
        mbar_KV1_ptr = storage.mbar_KV1.data_ptr()
        mbar_KVEmpty0_ptr = storage.mbar_KVEmpty0.data_ptr()
        mbar_KVEmpty1_ptr = storage.mbar_KVEmpty1.data_ptr()

        if warp_idx == 0:
            cute.arch.mbarrier_init(mbar_Q0_ptr, 1)
            cute.arch.mbarrier_init(mbar_Q1_ptr, 1)
            cute.arch.mbarrier_init(mbar_KV0_ptr, 1)
            cute.arch.mbarrier_init(mbar_KV1_ptr, 1)
            cute.arch.mbarrier_init(mbar_KVEmpty0_ptr, 128)
            cute.arch.mbarrier_init(mbar_KVEmpty1_ptr, 128)
        cute.arch.sync_threads()

        sQ_staged = storage.sQ.get_tensor(sQ_layout_staged.outer, swizzle=sQ_layout_staged.inner)
        sKV_staged = storage.sKV.get_tensor(sKV_layout_staged.outer, swizzle=sKV_layout_staged.inner)
        sQ_0 = sQ_staged[None, None, 0]
        sQ_1 = sQ_staged[None, None, 1]
        sKV_0 = sKV_staged[None, None, 0]
        sKV_1 = sKV_staged[None, None, 1]
        sW = storage.sW.get_tensor(cute.make_layout((self.tile_m,), stride=(1,)))

        if warp_group_idx == 0:
            if warp_idx == 0:
                self.producer(
                    mQ,
                    tma_atom_Q,
                    mK,
                    tma_atom_K,
                    mW,
                    sQ_0,
                    sQ_1,
                    sKV_0,
                    sKV_1,
                    sW,
                    mbar_Q0_ptr,
                    mbar_Q1_ptr,
                    mbar_KV0_ptr,
                    mbar_KV1_ptr,
                    mbar_KVEmpty0_ptr,
                    mbar_KVEmpty1_ptr,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mQCausalOffsets,
                    max_seqlen_q,
                    max_seqlen_k,
                )
        else:
            self.consumer(
                tiled_mma_QK,
                sQ_0,
                sQ_1,
                sKV_staged,
                sW,
                mOut,
                mbar_Q0_ptr,
                mbar_Q1_ptr,
                mbar_KV0_ptr,
                mbar_KV1_ptr,
                mbar_KVEmpty0_ptr,
                mbar_KVEmpty1_ptr,
                mCuSeqlensQ,
                mCuSeqlensK,
                mQCausalOffsets,
                max_seqlen_q,
                max_seqlen_k,
                sm_scale,
            )
