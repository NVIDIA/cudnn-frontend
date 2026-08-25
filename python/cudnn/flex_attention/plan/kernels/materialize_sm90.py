# SPDX-License-Identifier: BSD-3-Clause
"""SM90 packed-predicate materialization kernels."""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Boolean, Int32, Uint32, const_expr
import cuda.bindings.driver as cuda

from cudnn.flex_attention.kernels.sm90.bwd.backward_config import make_sm90_bwd_tiled_mma_sdp
from cudnn.flex_attention.kernels.sm90.fwd.forward_config import (
    _sm90_fwd_mask_payload_representative_tidx,
    make_sm90_fwd_tiled_mma_qk,
)
from cudnn.flex_attention.plan.kernels.common import (
    _shr_u32,
)
from cudnn.flex_attention.plan.kernels.compact import (
    _ArbitraryPlanK2QCompact,
    _ArbitraryPlanQ2KCompact,
)


class _ArbitraryPlanMaterializeSm90(_ArbitraryPlanQ2KCompact):
    """Materialize stable CSR lists and MMA-thread-native payloads."""

    @cute.jit
    def __call__(
        self,
        mArbitraryFunc: cute.Tensor,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialIndices: cute.Tensor,
        mPartialMasks: cute.Tensor,
        mPartialWorkDesc: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullIndices: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_m_blocks: Int32,
        max_n_blocks: Int32,
        nfunc: Int32,
        stream: cuda.CUstream = None,
    ):
        upper_total_m_blocks = mVisibleBits.shape[1]
        hmask = mArbitraryFunc.shape[0]
        tiled_mma_qk = make_sm90_fwd_tiled_mma_qk(
            self.dtype,
            self.tile_m,
            self.tile_n,
        )
        self.compact_kernel(
            mVisibleBits,
            mFullBits,
            mPartialOffsets,
            mPartialIndices,
            mPartialWorkDesc,
            mFullOffsets,
            mFullIndices,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            max_m_blocks,
            max_n_blocks,
        ).launch(
            grid=(upper_total_m_blocks, hmask, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )
        if mPartialMasks.shape[0] > 0:
            self.payload_kernel(
                tiled_mma_qk,
                mArbitraryFunc,
                mPartialWorkDesc,
                mPartialMasks,
                mCuSeqlensQ,
                mCuSeqlensK,
                mCuTotalMBlocks,
                batch_size,
                seqlen_q_fixed,
                seqlen_k_fixed,
                total_q,
                total_k,
                max_m_blocks,
                nfunc,
            ).launch(
                grid=(mPartialMasks.shape[0], 1, 1),
                block=(self.num_threads, 1, 1),
                stream=stream,
            )

    @cute.kernel
    def payload_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        mArbitraryFunc: cute.Tensor,
        mPartialWorkDesc: cute.Tensor,
        mPartialMasks: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_m_blocks: Int32,
        nfunc: Int32,
    ):
        planner_tidx, _, _ = cute.arch.thread_idx()
        payload_idx, _, _ = cute.arch.block_idx()
        mask_head = mPartialWorkDesc[payload_idx, 0]
        batch_idx = mPartialWorkDesc[payload_idx, 1]
        local_m_block = mPartialWorkDesc[payload_idx, 2]
        local_n_block = mPartialWorkDesc[payload_idx, 3]
        upper_outer_row = batch_idx * max_m_blocks + local_m_block
        (
            _,
            _,
            _,
            q_begin,
            q_len,
            _,
            k_len,
            valid_m_block,
        ) = self._sample_info(
            upper_outer_row,
            max_m_blocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
        )
        canonical_words = (self.tile_n + 31) // 32
        canonical_stride = canonical_words + 1
        smem = cutlass_utils.SmemAllocator()
        sMask = smem.allocate_tensor(
            element_type=Uint32,
            layout=cute.make_layout(
                (self.tile_m, canonical_stride),
                stride=(canonical_stride, 1),
            ),
            byte_alignment=16,
        )
        if planner_tidx < Int32(self.tile_m):
            row_in_tile = planner_tidx
            q_global, _, q_valid = self._physical_q_info(
                local_m_block,
                row_in_tile,
                q_begin,
                q_len,
            )
            rCanonical = cute.make_rmem_tensor((canonical_words,), Uint32)
            rCanonical.fill(Uint32(0))
            if valid_m_block and q_valid:
                num_intervals = (nfunc + Int32(1)) // Int32(2)
                for interval_idx in cutlass.range(num_intervals, unroll=1):
                    _, _, local_begin, local_end = self._safe_interval(
                        mArbitraryFunc,
                        mask_head,
                        interval_idx,
                        q_global,
                        k_len,
                    )
                    for word in cutlass.range_constexpr(canonical_words):
                        word_begin = local_n_block * Int32(self.tile_n) + Int32(word * 32)
                        lo = cutlass.max(Int32(0), local_begin - word_begin)
                        hi = cutlass.max(Int32(0), local_end - word_begin)
                        lo = cutlass.min(Int32(32), lo)
                        hi = cutlass.min(Int32(32), hi)
                        low_hi = _shr_u32(
                            Uint32(0xFFFF_FFFF),
                            Uint32(Int32(32) - hi),
                        )
                        low_lo = _shr_u32(
                            Uint32(0xFFFF_FFFF),
                            Uint32(Int32(32) - lo),
                        )
                        rCanonical[word] |= low_hi & ~low_lo
            for word in cutlass.range_constexpr(canonical_words):
                sMask[row_in_tile, word] = rCanonical[word]
        cute.arch.sync_threads()

        payload_group_idx = planner_tidx
        while payload_group_idx < Int32(self.num_mask_payload_groups):
            consumer_tidx = _sm90_fwd_mask_payload_representative_tidx(
                payload_group_idx,
                self.payload_qhead_per_kvhead,
            )
            thr_mma_qk = tiled_mma_qk.get_slice(consumer_tidx)
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS = thr_mma_qk.partition_C(cS)
            rMask = cute.make_rmem_tensor((self.payload_padded_words,), Uint32)
            rMask.fill(Uint32(0))
            for word_idx in cutlass.range_constexpr(self.payload_valid_words):
                mask_word = Uint32(0)
                for bit_idx in cutlass.range_constexpr(32):
                    value_idx = word_idx * 32 + bit_idx
                    keep = Boolean(False)
                    if value_idx < self.payload_values_per_thread:
                        coord = tScS[value_idx]
                        row_in_tile = Int32(coord[0])
                        col_in_tile = Int32(coord[1])
                        source_word = col_in_tile // Int32(32)
                        source_bit = col_in_tile - source_word * Int32(32)
                        keep = (sMask[row_in_tile, source_word] & (Uint32(1) << Uint32(source_bit))) != Uint32(0)
                    if keep:
                        mask_word |= Uint32(1) << Uint32(bit_idx)
                rMask[word_idx] = mask_word
            mask_iter = mPartialMasks.iterator + cute.crd2idx(
                (payload_idx, Int32(0), payload_group_idx, Int32(0)),
                mPartialMasks.layout,
            )
            mask_ptr = cute.make_ptr(
                Uint32,
                mask_iter.toint(),
                cute.AddressSpace.gmem,
                assumed_align=min(
                    16,
                    4 * (self.payload_padded_words & -self.payload_padded_words),
                ),
            )
            gMask = cute.make_tensor(mask_ptr, (self.payload_padded_words,))
            cute.autovec_copy(rMask, gMask)
            payload_group_idx += Int32(self.num_threads)


class _ArbitraryPlanK2QMaterializeSm90(_ArbitraryPlanK2QCompact):
    """Materialize stable K2Q CSR, native backward payload, and dQ ranks."""

    @cute.jit
    def __call__(
        self,
        mArbitraryFunc: cute.Tensor,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mQPartialCounts: cute.Tensor,
        mQFullCounts: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialIndices: cute.Tensor,
        mPartialMasks: cute.Tensor,
        mPartialWorkDesc: cute.Tensor,
        mPartialDQOrder: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullIndices: cute.Tensor,
        mFullDQOrder: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalQBlocks: Optional[cute.Tensor],
        mCuTotalKBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_n_blocks: Int32,
        nfunc: Int32,
        stream: cuda.CUstream = None,
    ):
        # The kernel decodes an upper-bound row as batch_idx * max_n_blocks +
        # local_n_block. Launching only the compact K-block count would skip
        # samples after an interior zero-length sample.
        upper_total_n_blocks = batch_size * max_n_blocks
        hmask = mArbitraryFunc.shape[0]
        tiled_mma_sdp = make_sm90_bwd_tiled_mma_sdp(
            self.dtype,
            self.consumer_tile_m,
            self.tile_n,
            self.num_wg_mma,
            self.atom_layout_m_sdp,
            self.sdp_swap_ab,
        )
        self.compact_kernel(
            mVisibleBits,
            mFullBits,
            mQPartialCounts,
            mQFullCounts,
            mPartialOffsets,
            mPartialIndices,
            mPartialWorkDesc,
            mPartialDQOrder,
            mFullOffsets,
            mFullIndices,
            mFullDQOrder,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalQBlocks,
            mCuTotalKBlocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            max_n_blocks,
        ).launch(
            grid=(upper_total_n_blocks, hmask, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )
        if mPartialMasks.shape[0] > 0:
            self.payload_kernel(
                tiled_mma_sdp,
                mArbitraryFunc,
                mPartialWorkDesc,
                mPartialMasks,
                mCuSeqlensQ,
                mCuSeqlensK,
                mCuTotalQBlocks,
                mCuTotalKBlocks,
                batch_size,
                seqlen_q_fixed,
                seqlen_k_fixed,
                total_q,
                total_k,
                max_n_blocks,
                nfunc,
            ).launch(
                grid=(mPartialMasks.shape[0], 1, 1),
                block=(self.num_threads, 1, 1),
                stream=stream,
            )

    @cute.kernel
    def payload_kernel(
        self,
        tiled_mma_sdp: cute.TiledMma,
        mArbitraryFunc: cute.Tensor,
        mPartialWorkDesc: cute.Tensor,
        mPartialMasks: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalQBlocks: Optional[cute.Tensor],
        mCuTotalKBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_n_blocks: Int32,
        nfunc: Int32,
    ):
        planner_tidx, _, _ = cute.arch.thread_idx()
        payload_idx, _, _ = cute.arch.block_idx()
        mask_head = mPartialWorkDesc[payload_idx, 0]
        batch_idx = mPartialWorkDesc[payload_idx, 1]
        local_q_block = mPartialWorkDesc[payload_idx, 2]
        local_n_block = mPartialWorkDesc[payload_idx, 3]
        upper_n_row = batch_idx * max_n_blocks + local_n_block
        (
            _,
            _,
            _,
            _,
            _,
            q_begin,
            q_len,
            _,
            k_len,
            valid_n_block,
        ) = self._sample_info_k(
            upper_n_row,
            max_n_blocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalQBlocks,
            mCuTotalKBlocks,
        )
        canonical_words = (self.tile_n + 31) // 32
        canonical_stride = canonical_words + 1
        smem = cutlass_utils.SmemAllocator()
        sMask = smem.allocate_tensor(
            element_type=Uint32,
            layout=cute.make_layout(
                (self.tile_m, canonical_stride),
                stride=(canonical_stride, 1),
            ),
            byte_alignment=16,
        )
        if planner_tidx < Int32(self.tile_m):
            q_row = planner_tidx
            q_local = local_q_block * Int32(self.tile_m) + q_row
            rCanonical = cute.make_rmem_tensor((canonical_words,), Uint32)
            rCanonical.fill(Uint32(0))
            if valid_n_block and q_local < q_len:
                q_global = q_begin + q_local
                num_intervals = (nfunc + Int32(1)) // Int32(2)
                for interval_idx in cutlass.range(num_intervals, unroll=1):
                    _, _, local_begin, local_end = self._safe_interval(
                        mArbitraryFunc,
                        mask_head,
                        interval_idx,
                        q_global,
                        k_len,
                    )
                    for word in cutlass.range_constexpr(canonical_words):
                        word_begin = local_n_block * Int32(self.tile_n) + Int32(word * 32)
                        lo = cutlass.max(Int32(0), local_begin - word_begin)
                        hi = cutlass.max(Int32(0), local_end - word_begin)
                        lo = cutlass.min(Int32(32), lo)
                        hi = cutlass.min(Int32(32), hi)
                        low_hi = _shr_u32(
                            Uint32(0xFFFF_FFFF),
                            Uint32(Int32(32) - hi),
                        )
                        low_lo = _shr_u32(
                            Uint32(0xFFFF_FFFF),
                            Uint32(Int32(32) - lo),
                        )
                        rCanonical[word] |= low_hi & ~low_lo
            for word in cutlass.range_constexpr(canonical_words):
                sMask[q_row, word] = rCanonical[word]
        cute.arch.sync_threads()

        consumer_tidx = planner_tidx
        while consumer_tidx < Int32(self.num_mma_threads):
            thr_mma_sdp = tiled_mma_sdp.get_slice(consumer_tidx)
            acc_shape = (self.consumer_tile_m, self.tile_n)
            cS = cute.make_identity_tensor(acc_shape if const_expr(not self.sdp_swap_ab) else acc_shape[::-1])
            tScS = thr_mma_sdp.partition_C(cS)
            row_coord = 0 if const_expr(not self.sdp_swap_ab) else 1
            col_coord = 1 if const_expr(not self.sdp_swap_ab) else 0
            for subtile_idx in cutlass.range_constexpr(self.subtile_factor):
                rMask = cute.make_rmem_tensor((self.payload_padded_words,), Uint32)
                rMask.fill(Uint32(0))
                for word_idx in cutlass.range_constexpr(self.payload_valid_words):
                    mask_word = Uint32(0)
                    for bit_idx in cutlass.range_constexpr(32):
                        value_idx = word_idx * 32 + bit_idx
                        keep = Boolean(False)
                        if value_idx < self.payload_values_per_thread:
                            coord = tScS[value_idx]
                            q_row = Int32(subtile_idx * self.consumer_tile_m) + Int32(coord[row_coord])
                            k_col = Int32(coord[col_coord])
                            source_word = k_col // Int32(32)
                            source_bit = k_col - source_word * Int32(32)
                            keep = (sMask[q_row, source_word] & (Uint32(1) << Uint32(source_bit))) != Uint32(0)
                        if keep:
                            mask_word |= Uint32(1) << Uint32(bit_idx)
                    rMask[word_idx] = mask_word
                mask_iter = mPartialMasks.iterator + cute.crd2idx(
                    (payload_idx, Int32(subtile_idx), consumer_tidx, Int32(0)),
                    mPartialMasks.layout,
                )
                mask_ptr = cute.make_ptr(
                    Uint32,
                    mask_iter.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=min(
                        16,
                        4 * (self.payload_padded_words & -self.payload_padded_words),
                    ),
                )
                gMask = cute.make_tensor(mask_ptr, (self.payload_padded_words,))
                cute.autovec_copy(rMask, gMask)
            consumer_tidx += Int32(self.num_threads)
