# SPDX-License-Identifier: BSD-3-Clause
"""Shared device geometry and stable-compaction helpers for plan kernels."""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import torch

from cudnn.flex_attention.kernels.sm90.bwd.backward_config import _ResolvedSm90BwdConsumerConfig
from cudnn.flex_attention.kernels.sm90.fwd.forward_config import _ResolvedSm90FwdConsumerConfig
from cudnn.flex_attention.kernels.sm100.bwd.backward_config import _ResolvedSm100BwdConsumerConfig
from cudnn.flex_attention.kernels.sm100.bwd.backward_config_hd256 import (
    _ResolvedSm100Hd256DkdvConsumerConfig,
)
from cudnn.flex_attention.plan.topology import (
    _ResolvedSm90BwdTopologyConfig,
    _ResolvedSm100BwdTopologyConfig,
    _ResolvedSm100FwdTopologyConfig,
    _ResolvedSm100Hd256DkdvTopologyConfig,
    _ResolvedSm100Hd256DqTopologyConfig,
    _ResolvedSm100Hd256FwdTopologyConfig,
    _consumer_plan_signature,
)


@dsl_user_op
def _shr_u32(val: Uint32, shift: Uint32, *, loc=None, ip=None) -> Uint32:
    """Perform a defined PTX unsigned shift, including a shift by 32."""

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [val.ir_value(loc=loc, ip=ip), shift.ir_value(loc=loc, ip=ip)],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _load_endpoint(
    mArbitraryFunc: cute.Tensor,
    mask_head: Int32,
    endpoint_idx: Int32,
    q_global: Int32,
) -> Int32:
    return mArbitraryFunc[(mask_head, endpoint_idx, q_global)]


class _ArbitraryPlanCommonSm90:
    def __init__(
        self,
        config: (
            _ResolvedSm90FwdConsumerConfig
            | _ResolvedSm90BwdTopologyConfig
            | _ResolvedSm100BwdTopologyConfig
            | _ResolvedSm100FwdTopologyConfig
            | _ResolvedSm100Hd256FwdTopologyConfig
            | _ResolvedSm100Hd256DqTopologyConfig
            | _ResolvedSm100Hd256DkdvTopologyConfig
        ),
    ):
        self.dtype = cutlass.BFloat16 if config.dtype == torch.bfloat16 else cutlass.Float16
        self.tile_m = config.tile_m
        self.tile_n = config.tile_n
        self.pack_gqa = config.pack_gqa
        self.qhead_per_kvhead = config.qhead_per_kvhead
        self.payload_qhead_per_kvhead = config.qhead_per_kvhead if config.pack_gqa else 1
        self.num_mma_threads = config.num_mma_threads
        self.num_mask_payload_groups = config.num_mask_payload_groups
        self.payload_values_per_thread = config.payload_values_per_thread
        self.payload_valid_words = config.payload_valid_words
        self.payload_padded_words = config.payload_padded_words
        self.is_varlen = config.is_varlen
        self.num_warps = 8
        self.num_threads = cute.arch.WARP_SIZE * self.num_warps

    @cute.jit
    def _sample_info(
        self,
        upper_outer_row: Int32,
        max_m_blocks: Int32,
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
    ):
        batch_idx = upper_outer_row // max_m_blocks
        local_m_block = upper_outer_row - batch_idx * max_m_blocks
        q_begin = batch_idx * seqlen_q_fixed
        q_end = q_begin + seqlen_q_fixed
        k_begin = batch_idx * seqlen_k_fixed
        k_end = k_begin + seqlen_k_fixed
        compact_outer_row = upper_outer_row
        if const_expr(self.is_varlen):
            q_begin = mCuSeqlensQ[batch_idx]
            q_end = mCuSeqlensQ[batch_idx + 1]
            k_begin = mCuSeqlensK[batch_idx]
            k_end = mCuSeqlensK[batch_idx + 1]
            q_begin = cutlass.max(Int32(0), cutlass.min(q_begin, total_q))
            q_end = cutlass.max(q_begin, cutlass.min(q_end, total_q))
            k_begin = cutlass.max(Int32(0), cutlass.min(k_begin, total_k))
            k_end = cutlass.max(k_begin, cutlass.min(k_end, total_k))
            compact_outer_row = mCuTotalMBlocks[batch_idx] + local_m_block
        q_len = q_end - q_begin
        k_len = k_end - k_begin
        physical_q_len = q_len * Int32(self.qhead_per_kvhead) if const_expr(self.pack_gqa) else q_len
        num_m_blocks = cute.ceil_div(physical_q_len, self.tile_m)
        valid_m_block = (batch_idx < batch_size) & (local_m_block < num_m_blocks)
        return (
            batch_idx,
            local_m_block,
            compact_outer_row,
            q_begin,
            q_len,
            k_begin,
            k_len,
            valid_m_block,
        )

    @cute.jit
    def _physical_q_info(
        self,
        local_m_block: Int32,
        row_in_tile: Int32,
        q_begin: Int32,
        q_len: Int32,
    ):
        physical_q = local_m_block * Int32(self.tile_m) + row_in_tile
        if const_expr(self.pack_gqa):
            q_local = physical_q // Int32(self.qhead_per_kvhead)
        else:
            q_local = physical_q
        q_valid = q_local < q_len
        return q_begin + q_local, q_local, q_valid

    @cute.jit
    def _safe_interval(
        self,
        mArbitraryFunc: cute.Tensor,
        mask_head: Int32,
        interval_idx: Int32,
        q_global: Int32,
        k_len: Int32,
    ):
        endpoint_begin = Int32(0)
        if interval_idx > Int32(0):
            endpoint_begin = _load_endpoint(mArbitraryFunc, mask_head, interval_idx * Int32(2) - Int32(1), q_global)
        endpoint_end = _load_endpoint(
            mArbitraryFunc,
            mask_head,
            interval_idx * Int32(2),
            q_global,
        )
        safe_begin = cutlass.max(Int32(0), cutlass.min(endpoint_begin, k_len))
        safe_end = cutlass.max(Int32(0), cutlass.min(endpoint_end, k_len))
        if safe_end < safe_begin:
            safe_end = safe_begin
        return endpoint_begin, endpoint_end, safe_begin, safe_end

    @cute.jit
    def _row_block_state(
        self,
        mArbitraryFunc: cute.Tensor,
        mask_head: Int32,
        q_global: Int32,
        block_id: Int32,
        nfunc: Int32,
        k_len: Int32,
    ):
        block_begin = block_id * Int32(self.tile_n)
        block_end = block_begin + Int32(self.tile_n)
        covered_end = block_begin
        visible = Boolean(False)
        num_intervals = (nfunc + Int32(1)) // Int32(2)
        for interval_idx in cutlass.range(num_intervals, unroll=1):
            _, _, local_begin, local_end = self._safe_interval(
                mArbitraryFunc,
                mask_head,
                interval_idx,
                q_global,
                k_len,
            )
            lo = cutlass.max(local_begin, block_begin)
            hi = cutlass.min(local_end, block_end)
            if hi > lo:
                visible = Boolean(True)
                if lo <= covered_end and hi > covered_end:
                    covered_end = hi
        return visible, covered_end >= block_end


class _ArbitraryPlanK2QCommonSm90(_ArbitraryPlanCommonSm90):
    def __init__(
        self,
        config: _ResolvedSm90BwdConsumerConfig | _ResolvedSm100BwdConsumerConfig | _ResolvedSm100Hd256DkdvConsumerConfig,
    ):
        if isinstance(config, _ResolvedSm100BwdConsumerConfig):
            topology_config = _ResolvedSm100BwdTopologyConfig(config)
        elif isinstance(config, _ResolvedSm100Hd256DkdvConsumerConfig):
            topology_config = _ResolvedSm100Hd256DkdvTopologyConfig(config)
        else:
            topology_config = _ResolvedSm90BwdTopologyConfig(config)
        super().__init__(topology_config)
        self.consumer_tile_m = config.tile_m
        self.consumer_tile_n = config.tile_n
        self.subtile_factor = config.subtile_factor
        self.sdp_swap_ab = config.sdp_swap_ab if isinstance(config, _ResolvedSm90BwdConsumerConfig) else True
        self.atom_layout_m_sdp = config.atom_layout_m_sdp if isinstance(config, _ResolvedSm90BwdConsumerConfig) else 1
        self.num_wg_mma = config.num_wg
        self.cta_group_size = (
            config.cta_group_size
            if isinstance(
                config,
                (_ResolvedSm100BwdConsumerConfig, _ResolvedSm100Hd256DkdvConsumerConfig),
            )
            else 1
        )
        self.spt = config.spt
        dq_order_format = _consumer_plan_signature(config).dq_order_format
        self.store_dq_order = dq_order_format != "none"

    @cute.jit
    def _sample_info_k(
        self,
        upper_n_row: Int32,
        max_n_blocks: Int32,
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalQBlocks: Optional[cute.Tensor],
        mCuTotalKBlocks: Optional[cute.Tensor],
    ):
        batch_idx = upper_n_row // max_n_blocks
        local_n_block = upper_n_row - batch_idx * max_n_blocks
        q_begin = batch_idx * seqlen_q_fixed
        q_end = q_begin + seqlen_q_fixed
        k_begin = batch_idx * seqlen_k_fixed
        k_end = k_begin + seqlen_k_fixed
        compact_q_begin = batch_idx * cute.ceil_div(seqlen_q_fixed, self.tile_m)
        compact_n_row = upper_n_row
        if const_expr(self.is_varlen):
            q_begin = cutlass.max(Int32(0), cutlass.min(mCuSeqlensQ[batch_idx], total_q))
            q_end = cutlass.max(q_begin, cutlass.min(mCuSeqlensQ[batch_idx + 1], total_q))
            k_begin = cutlass.max(Int32(0), cutlass.min(mCuSeqlensK[batch_idx], total_k))
            k_end = cutlass.max(k_begin, cutlass.min(mCuSeqlensK[batch_idx + 1], total_k))
            compact_q_begin = mCuTotalQBlocks[batch_idx]
            compact_n_row = mCuTotalKBlocks[batch_idx] + local_n_block
        q_len = q_end - q_begin
        k_len = k_end - k_begin
        num_q_blocks = cute.ceil_div(q_len, self.tile_m)
        num_k_blocks = cute.ceil_div(k_len, self.tile_n)
        valid_n_block = (batch_idx < batch_size) & (local_n_block < num_k_blocks)
        return (
            batch_idx,
            local_n_block,
            compact_n_row,
            compact_q_begin,
            num_q_blocks,
            q_begin,
            q_len,
            k_begin,
            k_len,
            valid_n_block,
        )

    @cute.jit
    def _dq_write_rank(
        self,
        mVisibleBits: cute.Tensor,
        mask_head: Int32,
        compact_q_row: Int32,
        local_n_block: Int32,
        contributor_count: Int32,
    ) -> Int32:
        rank = Int32(0)
        word_limit = local_n_block // Int32(32)
        word_idx = Int32(0)
        while word_idx < word_limit:
            rank += Int32(cute.arch.popc(mVisibleBits[mask_head, compact_q_row, word_idx]))
            word_idx += Int32(1)
        bit_idx = local_n_block - word_limit * Int32(32)
        low_bits = _shr_u32(Uint32(0xFFFF_FFFF), Uint32(Int32(32) - bit_idx))
        rank += Int32(cute.arch.popc(mVisibleBits[mask_head, compact_q_row, word_limit] & low_bits))
        if const_expr(self.spt):
            rank = contributor_count - Int32(1) - rank
        return rank
