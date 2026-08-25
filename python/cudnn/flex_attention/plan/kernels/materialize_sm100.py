# SPDX-License-Identifier: BSD-3-Clause
"""SM100 packed-predicate materialization kernels."""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Int32, Uint32, const_expr
import cuda.bindings.driver as cuda

from cudnn.flex_attention.kernels.sm100.bwd.backward_config import (
    _ResolvedSm100BwdConsumerConfig,
    make_sm100_bwd_tiled_mma_sdp,
    make_sm100_bwd_tmem_load,
)
from cudnn.flex_attention.kernels.sm100.bwd.backward_config_hd256 import (
    _ResolvedSm100Hd256DkdvConsumerConfig,
    _ResolvedSm100Hd256DqConsumerConfig,
    make_sm100_hd256_dkdv_score_ownership,
    make_sm100_hd256_dkdv_tiled_mma_kq,
)
from cudnn.flex_attention.kernels.sm100.fwd.forward_config import (
    _ResolvedSm100FwdConsumerConfig,
)
from cudnn.flex_attention.kernels.sm100.fwd.forward_config_hd256 import (
    _ResolvedSm100Hd256FwdConsumerConfig,
)
from cudnn.flex_attention.plan.kernels.common import (
    _shr_u32,
)
from cudnn.flex_attention.plan.kernels.materialize_sm90 import (
    _ArbitraryPlanK2QMaterializeSm90,
    _ArbitraryPlanMaterializeSm90,
)
from cudnn.flex_attention.plan.topology import (
    _ResolvedSm100FwdTopologyConfig,
    _ResolvedSm100Hd256DqTopologyConfig,
    _ResolvedSm100Hd256FwdTopologyConfig,
)


class _ArbitraryPlanMaterializeSm100(_ArbitraryPlanMaterializeSm90):
    """Materialize SM100 Q-major CSR and payloads with split kernels."""

    def __init__(
        self,
        config: _ResolvedSm100FwdConsumerConfig | _ResolvedSm100Hd256FwdConsumerConfig | _ResolvedSm100Hd256DqConsumerConfig,
    ):
        self.is_hd256_fwd = isinstance(config, _ResolvedSm100Hd256FwdConsumerConfig)
        self.is_hd256_dq = isinstance(config, _ResolvedSm100Hd256DqConsumerConfig)
        self.is_hd256 = self.is_hd256_fwd or self.is_hd256_dq
        if self.is_hd256_fwd:
            topology_config = _ResolvedSm100Hd256FwdTopologyConfig(config)
        elif self.is_hd256_dq:
            topology_config = _ResolvedSm100Hd256DqTopologyConfig(config)
        else:
            topology_config = _ResolvedSm100FwdTopologyConfig(config)
        super().__init__(topology_config)
        self.consumer_tile_m = config.tile_m
        self.consumer_tile_n = config.tile_n
        self.physical_subtiles = config.physical_subtiles
        self.cta_group_size = config.cta_group_size

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
        block_id = mPartialWorkDesc[payload_idx, 3]
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

        if planner_tidx < Int32(self.physical_subtiles * self.num_mask_payload_groups):
            payload_subtile_idx = planner_tidx // Int32(self.num_mask_payload_groups)
            payload_group_idx = planner_tidx - payload_subtile_idx * Int32(self.num_mask_payload_groups)
            row_in_tile = payload_subtile_idx * Int32(self.consumer_tile_m) + payload_group_idx
            q_global, _, q_valid = self._physical_q_info(
                local_m_block,
                row_in_tile,
                q_begin,
                q_len,
            )
            rMask = cute.make_rmem_tensor((self.payload_padded_words,), Uint32)
            rMask.fill(Uint32(0))
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
                    for word_idx in cutlass.range_constexpr(self.payload_valid_words):
                        word_begin = block_id * Int32(self.tile_n) + Int32(word_idx * 32)
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
                        rMask[word_idx] |= low_hi & ~low_lo

            mask_iter = mPartialMasks.iterator + cute.crd2idx(
                (
                    payload_idx,
                    payload_subtile_idx,
                    payload_group_idx,
                    Int32(0),
                ),
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


class _ArbitraryPlanK2QMaterializeSm100(_ArbitraryPlanK2QMaterializeSm90):
    """Materialize SM100 K-major CSR and payloads with split kernels."""

    def __init__(
        self,
        config: _ResolvedSm100BwdConsumerConfig | _ResolvedSm100Hd256DkdvConsumerConfig,
    ):
        self.is_hd256_dkdv = isinstance(config, _ResolvedSm100Hd256DkdvConsumerConfig)
        super().__init__(config)

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
        upper_total_n_blocks = batch_size * max_n_blocks
        hmask = mArbitraryFunc.shape[0]
        if const_expr(self.is_hd256_dkdv):
            tiled_mma_sdp = make_sm100_hd256_dkdv_tiled_mma_kq(self.dtype)
        else:
            tiled_mma_sdp = make_sm100_bwd_tiled_mma_sdp(
                self.dtype,
                self.consumer_tile_m,
                self.tile_n,
                self.cta_group_size,
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

        canonical_words = self.tile_n // 32
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
        if consumer_tidx < Int32(self.num_mma_threads):
            if const_expr(self.is_hd256_dkdv):
                tScS_t2r = make_sm100_hd256_dkdv_score_ownership(
                    tiled_mma_sdp,
                    consumer_tidx,
                )
                for q_subtile in cutlass.range_constexpr(self.subtile_factor):
                    for cta_rank in cutlass.range_constexpr(self.cta_group_size):
                        rMask = cute.make_rmem_tensor(
                            (self.payload_padded_words,),
                            Uint32,
                        )
                        rMask.fill(Uint32(0))
                        for word_idx in cutlass.range_constexpr(self.payload_valid_words):
                            mask_word = Uint32(0)
                            for bit_idx in cutlass.range_constexpr(32):
                                value_idx = word_idx * 32 + bit_idx
                                if value_idx < self.payload_values_per_thread:
                                    coord = tScS_t2r[value_idx]
                                    q_row = Int32(q_subtile * self.consumer_tile_m) + Int32(coord[1])
                                    k_col = Int32(cta_rank * self.consumer_tile_n) + Int32(coord[0])
                                    source_word = k_col // Int32(32)
                                    source_bit = k_col - source_word * Int32(32)
                                    keep = (sMask[q_row, source_word] & (Uint32(1) << Uint32(source_bit))) != Uint32(0)
                                    if keep:
                                        mask_word |= Uint32(1) << Uint32(bit_idx)
                            rMask[word_idx] = mask_word
                        payload_subtile_idx = q_subtile * self.cta_group_size + cta_rank
                        mask_iter = mPartialMasks.iterator + cute.crd2idx(
                            (
                                payload_idx,
                                Int32(payload_subtile_idx),
                                consumer_tidx,
                                Int32(0),
                            ),
                            mPartialMasks.layout,
                        )
                        mask_ptr = cute.make_ptr(
                            Uint32,
                            mask_iter.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        gMask = cute.make_tensor(mask_ptr, (self.payload_padded_words,))
                        cute.autovec_copy(rMask, gMask)
            if const_expr(not self.is_hd256_dkdv):
                thr_tmem_load = make_sm100_bwd_tmem_load(
                    consumer_tidx,
                    self.num_wg_mma,
                )
                for subtile_idx in cutlass.range_constexpr(self.subtile_factor):
                    for cta_rank in cutlass.range_constexpr(self.cta_group_size):
                        thr_mma_sdp = tiled_mma_sdp.get_slice(cta_rank)
                        cS = cute.make_identity_tensor((self.tile_n, self.consumer_tile_m))
                        tScS = thr_mma_sdp.partition_C(cS)
                        tScS_t2r = thr_tmem_load.partition_D(tScS)
                        rMask = cute.make_rmem_tensor((self.payload_padded_words,), Uint32)
                        rMask.fill(Uint32(0))
                        for word_idx in cutlass.range_constexpr(self.payload_valid_words):
                            mask_word = Uint32(0)
                            for bit_idx in cutlass.range_constexpr(32):
                                value_idx = word_idx * 32 + bit_idx
                                if value_idx < self.payload_values_per_thread:
                                    coord = tScS_t2r[value_idx]
                                    q_row = Int32(subtile_idx * self.consumer_tile_m) + Int32(coord[1])
                                    k_col = Int32(coord[0])
                                    source_word = k_col // Int32(32)
                                    source_bit = k_col - source_word * Int32(32)
                                    keep = (sMask[q_row, source_word] & (Uint32(1) << Uint32(source_bit))) != Uint32(0)
                                    if keep:
                                        mask_word |= Uint32(1) << Uint32(bit_idx)
                            rMask[word_idx] = mask_word
                        payload_subtile_idx = subtile_idx * self.cta_group_size + cta_rank
                        mask_iter = mPartialMasks.iterator + cute.crd2idx(
                            (
                                payload_idx,
                                Int32(payload_subtile_idx),
                                consumer_tidx,
                                Int32(0),
                            ),
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
