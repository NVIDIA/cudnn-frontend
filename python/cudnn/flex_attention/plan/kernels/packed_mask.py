# SPDX-License-Identifier: BSD-3-Clause
"""
Packed-mask materialization and runtime consumption helpers.

This module contains runtime execution functions for block-sparse attention kernels.
These utilities are used by CUTE DSL kernels to produce and consume block-sparse loads.
"""

from functools import partial
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

# Import data structures from block_sparsity
from cudnn.flex_attention.kernels.common import barrier
from cudnn.flex_attention.kernels.common.seqlen_info import SeqlenInfoQK
from cudnn.flex_attention.kernels.sm90.bwd.named_barrier import NamedBarrierBwd
from cudnn.flex_attention.kernels.sm90.fwd.forward_config import _sm90_fwd_mask_payload_group_idx
from cudnn.flex_attention.plan.kernels import BlockSparseTensors
from cudnn.flex_attention.runtime.dsl_utils import bulk_copy
from cudnn.flex_attention._compat import copy_utils


@dsl_user_op
def _prefetch_global_l1(ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Prefetch one global cache line into the local L1 data cache."""
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [ptr_i64],
        "prefetch.global.L1 [$0];",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def get_curr_arbitrary_blocksparse_tensors(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    seqlen_info: SeqlenInfoQK,
    tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
):
    """Extract one compact arbitrary-plan row and its mask-payload base."""
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    mask_block_idx = blocksparse_tensors.mask_block_idx
    full_block_cnt = blocksparse_tensors.full_block_cnt
    full_block_idx = blocksparse_tensors.full_block_idx
    mask_block_offset = blocksparse_tensors.mask_block_offset
    full_block_offset = blocksparse_tensors.full_block_offset
    assert mask_block_offset is not None
    assert full_block_cnt is not None
    assert full_block_idx is not None
    assert full_block_offset is not None

    total_m_blocks = mask_block_cnt.shape[1]
    plan_head = Int32(0)
    if mask_block_cnt.shape[0] != 1:
        plan_head = head_idx
    if const_expr(seqlen_info.has_cu_seqlens_q):
        outer_row = seqlen_info.m_block_offset + m_block
    else:
        physical_q_len = seqlen_info.seqlen_q * qhead_per_kvhead
        m_blocks_per_sample = cute.ceil_div(physical_q_len, tile_m)
        outer_row = batch_idx * m_blocks_per_sample + m_block
    plan_row = plan_head * total_m_blocks + outer_row

    partial_base = mask_block_offset[plan_row]
    full_base = full_block_offset[plan_row]
    curr_mask_block_cnt = mask_block_cnt[plan_head, outer_row]
    curr_mask_block_idx = cute.domain_offset(partial_base, mask_block_idx)
    curr_full_block_cnt = full_block_cnt[plan_head, outer_row]
    curr_full_block_idx = cute.domain_offset(full_base, full_block_idx)
    return (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
        partial_base,
    )


@cute.jit
def get_curr_arbitrary_block_counts_fwd_sm90(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    seqlen_info: SeqlenInfoQK,
    tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
):
    """Load only the arbitrary row metadata needed by SM90 MMA consumers."""
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    full_block_cnt = blocksparse_tensors.full_block_cnt
    mask_block_offset = blocksparse_tensors.mask_block_offset
    assert full_block_cnt is not None
    assert mask_block_offset is not None

    total_m_blocks = mask_block_cnt.shape[1]
    plan_head = Int32(0)
    if mask_block_cnt.shape[0] != 1:
        plan_head = head_idx
    if const_expr(seqlen_info.has_cu_seqlens_q):
        outer_row = seqlen_info.m_block_offset + m_block
    else:
        physical_q_len = seqlen_info.seqlen_q * qhead_per_kvhead
        m_blocks_per_sample = cute.ceil_div(physical_q_len, tile_m)
        outer_row = batch_idx * m_blocks_per_sample + m_block
    return (
        mask_block_cnt[plan_head, outer_row],
        full_block_cnt[plan_head, outer_row],
        mask_block_offset[plan_head * total_m_blocks + outer_row],
    )


@cute.jit
def get_curr_arbitrary_blocksparse_tensors_bwd(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    n_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    n_blocks_per_sample: cutlass.Int32,
):
    """Extract one compact arbitrary K2Q row and its mask-payload base."""

    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    mask_block_idx = blocksparse_tensors.mask_block_idx
    full_block_cnt = blocksparse_tensors.full_block_cnt
    full_block_idx = blocksparse_tensors.full_block_idx
    mask_block_offset = blocksparse_tensors.mask_block_offset
    full_block_offset = blocksparse_tensors.full_block_offset
    cu_total_k_blocks = blocksparse_tensors.cu_total_m_blocks
    assert mask_block_offset is not None
    assert full_block_cnt is not None
    assert full_block_idx is not None
    assert full_block_offset is not None

    total_n_blocks = mask_block_cnt.shape[1]
    plan_head = Int32(0)
    if mask_block_cnt.shape[0] != 1:
        plan_head = head_idx
    outer_row = batch_idx * n_blocks_per_sample + n_block
    if const_expr(cu_total_k_blocks is not None):
        outer_row = cu_total_k_blocks[batch_idx] + n_block
    plan_row = plan_head * total_n_blocks + outer_row

    partial_base = mask_block_offset[plan_row]
    full_base = full_block_offset[plan_row]
    curr_mask_block_cnt = mask_block_cnt[plan_head, outer_row]
    curr_mask_block_idx = cute.domain_offset(partial_base, mask_block_idx)
    curr_full_block_cnt = full_block_cnt[plan_head, outer_row]
    curr_full_block_idx = cute.domain_offset(full_base, full_block_idx)
    return (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
        partial_base,
        full_base,
    )


@cute.jit
def get_curr_arbitrary_block_counts_bwd_sm90(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    n_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    n_blocks_per_sample: cutlass.Int32,
):
    """Load the K2Q counts and partial-payload base needed by the MMA consumer."""
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    full_block_cnt = blocksparse_tensors.full_block_cnt
    mask_block_offset = blocksparse_tensors.mask_block_offset
    cu_total_k_blocks = blocksparse_tensors.cu_total_m_blocks
    assert full_block_cnt is not None
    assert mask_block_offset is not None

    total_n_blocks = mask_block_cnt.shape[1]
    plan_head = Int32(0)
    if mask_block_cnt.shape[0] != 1:
        plan_head = head_idx
    outer_row = batch_idx * n_blocks_per_sample + n_block
    if const_expr(cu_total_k_blocks is not None):
        outer_row = cu_total_k_blocks[batch_idx] + n_block
    return (
        mask_block_cnt[plan_head, outer_row],
        full_block_cnt[plan_head, outer_row],
        mask_block_offset[plan_head * total_n_blocks + outer_row],
    )


@cute.jit
def load_mask_payload(
    mask_payloads: Optional[cute.Tensor],
    payload_idx: Int32,
    payload_group_idx: Int32,
    subtile_idx: Int32 = Int32(0),
    payload_words: cutlass.Constexpr[int] = 4,
):
    """Load one compact consumer-native arbitrary-mask payload."""
    if const_expr(mask_payloads is None):
        return None
    payload_alignment = min(16, 4 * (payload_words & -payload_words))
    mask_iter = mask_payloads.iterator + cute.crd2idx(
        (payload_idx, subtile_idx, payload_group_idx, Int32(0)),
        mask_payloads.layout,
    )
    mask_ptr = cute.make_ptr(
        Uint32,
        mask_iter.toint(),
        cute.AddressSpace.gmem,
        assumed_align=payload_alignment,
    )
    g_mask = cute.make_tensor(mask_ptr, (payload_words,))
    r_mask = cute.make_rmem_tensor_like(g_mask, Uint32)
    cute.autovec_copy(g_mask, r_mask)
    return r_mask


@cute.jit
def load_mask_payload_to_smem(
    mask_payloads: cute.Tensor,
    payload_idx: Int32,
    subtile_idx: Int32,
    s_mask: cute.Tensor,
    mask_pipeline,
    producer_state,
    payload_groups: cutlass.Constexpr[int],
    payload_words: cutlass.Constexpr[int],
):
    """Stage one CTA-native mask payload with a bulk G2S copy."""

    mask_iter = mask_payloads.iterator + cute.crd2idx(
        (payload_idx, subtile_idx, Int32(0), Int32(0)),
        mask_payloads.layout,
    )
    mask_ptr = cute.make_ptr(
        Uint32,
        mask_iter.toint(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    # Keep this view one-dimensional. A nested (group, word) layout lowers to
    # one 16-byte bulk copy per group and can exhaust the async-copy queue.
    mask_layout = cute.make_layout((payload_groups * payload_words,))
    g_mask = cute.make_tensor(mask_ptr, mask_layout)
    s_mask_linear = cute.make_tensor(s_mask.iterator, mask_layout)
    copy_atom_mask = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Uint32)

    mask_pipeline.producer_acquire(producer_state)
    bulk_copy(
        copy_atom_mask,
        g_mask,
        s_mask_linear,
        mbar_ptr=mask_pipeline.producer_get_barrier(producer_state),
    )
    producer_state.advance()
    return producer_state


@cute.jit
def consume_mask_payload_from_smem(
    s_mask: cute.Tensor,
    payload_group_idx: Int32,
    mask_pipeline,
    consumer_state,
):
    """Wait for one staged payload and copy this thread's words to registers."""

    mask_pipeline.consumer_wait(consumer_state)
    s_mask_thread = s_mask[payload_group_idx, None]
    r_mask = cute.make_rmem_tensor_like(s_mask_thread, Uint32)
    cute.autovec_copy(s_mask_thread, r_mask)
    mask_pipeline.consumer_release(consumer_state)
    consumer_state.advance()
    return r_mask, consumer_state


# NOTE [SM100 block-sparse empty tiles: mbarrier contract]
#
# For block-sparse SM100 forward, a given (m_block, stage) Q tile can have zero active
# KV blocks (total_block_cnt == 0). In that case there is no seqlen_kv iteration, so
# the softmax warp-group has no row stats to publish.
#
# The correction warp-group seeds fully-masked-row stats and runs the usual correction
# epilogue so output/LSE have well-defined values. Both warp-groups must still perform
# the softmax<->correction mbarrier handshake so phases advance correctly across
# empty->empty and empty->non-empty tile sequences.
#
# This follows the usual fully-masked-row convention:
# output is zero and LSE is -inf.
#
# Barrier contract (each is `mbar_ptr + <offset> + stage`):
#
# Producer/consumer pairs:
# - `mbar_softmax_corr_full`    : softmax arrive        -> correction wait
# - `mbar_softmax_corr_empty`   : correction arrive     -> softmax wait
# - `mbar_P_full_O_rescaled`    : softmax arrive (+ correction arrive) -> MMA wait
# - `mbar_P_full_2`             : softmax arrive        -> MMA wait
# - `mbar_corr_epi_full_/empty` : correction <-> epilogue (only when epilogue is separate)
#
# Empty tile (`total_block_cnt == 0`):
# - Softmax: skips the seqlen_kv softmax path entirely (no P stores, no `mbar_P_full_*`).
#   It only arrives `mbar_softmax_corr_full` once per stage as a synthetic "no work" signal.
#   At the `softmax_loop` level, softmax unconditionally waits `mbar_softmax_corr_empty`
#   before each tile (when block-sparse) to drain a prior correction arrival and keep
#   phases aligned across non-empty -> empty transitions.
# - Correction: waits `mbar_softmax_corr_full`, seeds stats + runs `correction_epilogue(scale=0)`,
#   and arrives `mbar_softmax_corr_empty` (and `mbar_corr_epi_full_/empty` when applicable).
# - No `mbar_P_full_*` barriers are arrived (no P, no MMA O); only the softmax<->correction
#   (and correction<->epilogue) handshakes advance phases.
#
# Non-empty tile:
# - Softmax: runs `softmax_step` (produces P) and uses `mbar_softmax_corr_full/empty` to
#   publish row_max (during seqlen_kv) and final row stats (once per tile), and to advance phases;
#   arrives `mbar_P_full_*` when P is stored.
# - Correction: waits `mbar_softmax_corr_full`, may rescale/release O, arrives `mbar_softmax_corr_empty`
#   to ack/advance, and arrives `mbar_P_full_O_rescaled` when MMA can proceed.
#
# Backward (SM100):
# - Empty KV tile: for a given `n_block`, `total_m_block_cnt == 0` means no Q tiles contribute.
# - Both the load and compute loops guard all pipeline work on `process_tile`, so empty tiles
#   skip producer/consumer operations entirely (no per-tile mbarrier phase handshake like forward).
# - In the `not dKV_postprocess` path, dK/dV for empty KV tiles are explicitly written as zeros
#   even when `process_tile == False` (see `flash_bwd_sm100.py` `should_zero_dKV`).
@cute.jit
def prefetch_arbitrary_forward_block_index(
    block_idx: cute.Tensor,
    list_idx: Int32,
    valid,
):
    """Prefetch one indirect block index without selecting a CSR list."""
    if valid:
        block_iter = block_idx.iterator + cute.crd2idx((list_idx,), block_idx.layout)
        with cute.arch.elect_one():
            block_ptr = cute.make_ptr(
                Int32,
                block_iter.toint(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            _prefetch_global_l1(block_ptr)


@cute.jit
def prefetch_arbitrary_forward_mask(
    mask_payloads: cute.Tensor,
    payload_idx: Int32,
    is_partial,
    payload_words: cutlass.Constexpr[int],
    subtile_idx: Int32 = Int32(0),
):
    """Use the TMA producer warp to stage a partial payload in L1."""
    if is_partial:
        lane_idx = cute.arch.lane_idx()
        cache_line_words = 128 // 4
        payload_total_words = mask_payloads.shape[2] * payload_words
        word_offset = lane_idx * cache_line_words
        if word_offset < payload_total_words:
            mask_iter = mask_payloads.iterator + cute.crd2idx(
                (payload_idx, subtile_idx, Int32(0), Int32(0)),
                mask_payloads.layout,
            )
            mask_ptr = cute.make_ptr(
                Uint32,
                mask_iter.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            _prefetch_global_l1(mask_ptr + word_offset)


@cute.jit
def apply_arbitrary_forward_mask(
    acc_S: cute.Tensor,
    n_block: Int32,
    mask_payloads: cute.Tensor,
    payload_idx: Int32,
    payload_group_idx: Int32,
    payload_words: cutlass.Constexpr[int],
    mask_seqlen: cutlass.Constexpr[bool] = True,
    r_bitmask: Optional[cute.Tensor] = None,
):
    """Apply one partial-block payload."""
    # The packed payload already encodes the sequence boundary.
    if const_expr(r_bitmask is None):
        r_bitmask = load_mask_payload(
            mask_payloads,
            payload_idx,
            payload_group_idx,
            payload_words=payload_words,
        )
    apply_loaded_arbitrary_mask(acc_S, n_block, r_bitmask, payload_words)


@cute.jit
def apply_loaded_arbitrary_mask(
    acc_S: cute.Tensor,
    n_block: Int32,
    r_bitmask: cute.Tensor,
    payload_valid_words: cutlass.Constexpr[int],
):
    """Apply valid mask words without retaining generic mask state."""

    # The payload is already laid out in the exact TMEM-to-register score
    # order, so arbitrary masking needs no sequence coordinates or
    # AttentionMask object.  Keeping this leaf specialized also makes the
    # consumer-native mapping explicit.
    for word_idx in cutlass.range_constexpr(payload_valid_words):
        col_start = 32 * word_idx
        mask_word = r_bitmask[word_idx]
        for bit_idx in cutlass.range_constexpr(32):
            col = col_start + bit_idx
            # The last payload word may extend past a non-word-aligned fragment.
            if const_expr(col < cute.size(acc_S)):
                keep = cutlass.Boolean((mask_word >> bit_idx) & 1)
                acc_S[col] = acc_S[col] if keep else -Float32.inf


@cute.jit
def apply_loaded_arbitrary_forward_mask(
    acc_S: cute.Tensor,
    n_block: Int32,
    r_bitmask: cute.Tensor,
):
    """Apply every word of a forward consumer-native payload."""

    apply_loaded_arbitrary_mask(
        acc_S,
        n_block,
        r_bitmask,
        cute.size(r_bitmask.shape[0]),
    )


@cute.jit
def produce_arbitrary_forward_nonoverlap(
    partial_block_cnt,
    partial_block_idx: cute.Tensor,
    full_block_cnt,
    full_block_idx: cute.Tensor,
    partial_payload_base: Int32,
    mask_payloads: cute.Tensor,
    payload_words: cutlass.Constexpr[int],
    kv_producer_state,
    load_K: Callable,
    load_V: Callable,
    pipeline_k,
    pipeline_v,
    o_empty_mbar_ptr: Optional[cute.Pointer] = None,
    o_empty_phase: Optional[Int32] = None,
    pipeline_mask=None,
    mask_producer_state=None,
    sMask: Optional[cute.Tensor] = None,
    payload_groups: cutlass.Constexpr[int] = 0,
):
    """Produce anchored partial/full CSR loops without K/V overlap."""
    if const_expr(o_empty_mbar_ptr is not None):
        assert o_empty_phase is not None
        cute.arch.mbarrier_wait(o_empty_mbar_ptr, phase=o_empty_phase)

    for iteration in cutlass.range(partial_block_cnt, unroll=1):
        partial_list_idx = partial_block_cnt - Int32(1) - iteration
        n_block = partial_block_idx[partial_list_idx]
        payload_idx = partial_payload_base + partial_list_idx
        if const_expr(pipeline_mask is not None):
            assert mask_producer_state is not None
            assert sMask is not None
            mask_producer_state = load_mask_payload_to_smem(
                mask_payloads,
                payload_idx,
                Int32(0),
                sMask[None, None, mask_producer_state.index],
                pipeline_mask,
                mask_producer_state,
                payload_groups,
                payload_words,
            )
        else:
            prefetch_arbitrary_forward_mask(mask_payloads, payload_idx, True, payload_words)
        prefetch_arbitrary_forward_block_index(
            partial_block_idx,
            partial_list_idx - Int32(1),
            iteration + Int32(1) < partial_block_cnt,
        )
        pipeline_k.producer_acquire(kv_producer_state)
        load_K(src_idx=n_block, producer_state=kv_producer_state)
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block, producer_state=kv_producer_state)
        kv_producer_state.advance()

    for iteration in cutlass.range(full_block_cnt, unroll=1):
        full_list_idx = full_block_cnt - Int32(1) - iteration
        n_block = full_block_idx[full_list_idx]
        prefetch_arbitrary_forward_block_index(
            full_block_idx,
            full_list_idx - Int32(1),
            iteration + Int32(1) < full_block_cnt,
        )
        pipeline_k.producer_acquire(kv_producer_state)
        load_K(src_idx=n_block, producer_state=kv_producer_state)
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block, producer_state=kv_producer_state)
        kv_producer_state.advance()
    return kv_producer_state, mask_producer_state


@cute.jit
def produce_arbitrary_forward_overlap(
    partial_block_cnt,
    partial_block_idx: cute.Tensor,
    full_block_cnt,
    full_block_idx: cute.Tensor,
    partial_payload_base: Int32,
    mask_payloads: cute.Tensor,
    payload_words: cutlass.Constexpr[int],
    kv_producer_state,
    load_K: Callable,
    load_V: Callable,
    pipeline_k,
    pipeline_v,
    o_empty_mbar_ptr: Optional[cute.Pointer] = None,
    o_empty_phase: Optional[Int32] = None,
    pipeline_mask=None,
    mask_producer_state=None,
    sMask: Optional[cute.Tensor] = None,
    payload_groups: cutlass.Constexpr[int] = 0,
):
    """Produce partial/full CSR loops with overlapped K/V loads."""
    total_block_cnt = partial_block_cnt + full_block_cnt
    n_block_prev = Int32(0)
    full_start = Int32(0)
    if total_block_cnt > Int32(0):
        if partial_block_cnt > Int32(0):
            partial_list_idx = partial_block_cnt - Int32(1)
            n_block_prev = partial_block_idx[partial_list_idx]
            payload_idx = partial_payload_base + partial_list_idx
            if const_expr(pipeline_mask is not None):
                assert mask_producer_state is not None
                assert sMask is not None
                mask_producer_state = load_mask_payload_to_smem(
                    mask_payloads,
                    payload_idx,
                    Int32(0),
                    sMask[None, None, mask_producer_state.index],
                    pipeline_mask,
                    mask_producer_state,
                    payload_groups,
                    payload_words,
                )
            else:
                prefetch_arbitrary_forward_mask(mask_payloads, payload_idx, True, payload_words)
                prefetch_arbitrary_forward_mask(
                    mask_payloads,
                    payload_idx - Int32(1),
                    partial_block_cnt > Int32(1),
                    payload_words,
                )
            prefetch_arbitrary_forward_block_index(
                partial_block_idx,
                partial_list_idx - Int32(1),
                partial_block_cnt > Int32(1),
            )
            prefetch_arbitrary_forward_block_index(
                full_block_idx,
                full_block_cnt - Int32(1),
                full_block_cnt > Int32(0),
            )
        else:
            full_list_idx = full_block_cnt - Int32(1)
            n_block_prev = full_block_idx[full_list_idx]
            prefetch_arbitrary_forward_block_index(
                full_block_idx,
                full_list_idx - Int32(1),
                full_block_cnt > Int32(1),
            )
        pipeline_k.producer_acquire(kv_producer_state)
        load_K(src_idx=n_block_prev, producer_state=kv_producer_state)

    # K uses independent shared storage. Issue the first K TMA before waiting
    # for the previous O epilogue to release the aliased V/O buffer.
    if const_expr(o_empty_mbar_ptr is not None):
        assert o_empty_phase is not None
        cute.arch.mbarrier_wait(o_empty_mbar_ptr, phase=o_empty_phase)

    if total_block_cnt > Int32(0):
        if partial_block_cnt > Int32(0):
            for iteration in cutlass.range(1, partial_block_cnt, unroll=1):
                partial_list_idx = partial_block_cnt - Int32(1) - iteration
                n_block = partial_block_idx[partial_list_idx]
                payload_idx = partial_payload_base + partial_list_idx
                if const_expr(pipeline_mask is not None):
                    assert mask_producer_state is not None
                    assert sMask is not None
                    mask_producer_state = load_mask_payload_to_smem(
                        mask_payloads,
                        payload_idx,
                        Int32(0),
                        sMask[None, None, mask_producer_state.index],
                        pipeline_mask,
                        mask_producer_state,
                        payload_groups,
                        payload_words,
                    )
                else:
                    prefetch_arbitrary_forward_mask(
                        mask_payloads,
                        payload_idx - Int32(1),
                        iteration + Int32(1) < partial_block_cnt,
                        payload_words,
                    )
                prefetch_arbitrary_forward_block_index(
                    partial_block_idx,
                    partial_list_idx - Int32(1),
                    iteration + Int32(1) < partial_block_cnt,
                )
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_prev, producer_state=kv_producer_state_prev)
                n_block_prev = n_block
            full_start = Int32(0)
        else:
            full_start = Int32(1)

        for iteration in cutlass.range(full_start, full_block_cnt, unroll=1):
            full_list_idx = full_block_cnt - Int32(1) - iteration
            n_block = full_block_idx[full_list_idx]
            prefetch_arbitrary_forward_block_index(
                full_block_idx,
                full_list_idx - Int32(1),
                iteration + Int32(1) < full_block_cnt,
            )
            kv_producer_state_prev = kv_producer_state.clone()
            kv_producer_state.advance()
            pipeline_k.producer_acquire(kv_producer_state)
            load_K(src_idx=n_block, producer_state=kv_producer_state)
            pipeline_v.producer_acquire(kv_producer_state_prev)
            load_V(src_idx=n_block_prev, producer_state=kv_producer_state_prev)
            n_block_prev = n_block

    if total_block_cnt > Int32(0):
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block_prev, producer_state=kv_producer_state)
        kv_producer_state.advance()
    return kv_producer_state, mask_producer_state


@cute.jit
def consume_arbitrary_forward_nonoverlap(
    partial_block_cnt,
    partial_block_idx: Optional[cute.Tensor],
    full_block_cnt,
    full_block_idx: Optional[cute.Tensor],
    partial_payload_base: Int32,
    mask_payloads: cute.Tensor,
    kv_consumer_state,
    mma_pv_fn: Callable,
    mma_one_n_block: Callable,
    payload_group_idx: Int32,
    payload_words: cutlass.Constexpr[int],
    warp_scheduler_barrier_sync: Callable,
    warp_scheduler_barrier_arrive: Callable,
    pipeline_mask=None,
    mask_consumer_state=None,
    sMask: Optional[cute.Tensor] = None,
):
    """Consume partial/full CSR loops without K/V overlap."""
    total_block_cnt = partial_block_cnt + full_block_cnt
    processed_any = total_block_cnt > Int32(0)
    full_start = Int32(0)
    if processed_any:
        warp_scheduler_barrier_sync()
        if partial_block_cnt > Int32(0):
            partial_list_idx = partial_block_cnt - Int32(1)
            payload_idx = partial_payload_base + partial_list_idx
            n_block = Int32(0)
            if const_expr(partial_block_idx is not None):
                n_block = partial_block_idx[partial_list_idx]
            r_bitmask = None
            if const_expr(pipeline_mask is not None):
                assert mask_consumer_state is not None
                assert sMask is not None
                r_bitmask, mask_consumer_state = consume_mask_payload_from_smem(
                    sMask[None, None, mask_consumer_state.index],
                    payload_group_idx,
                    pipeline_mask,
                    mask_consumer_state,
                )
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=n_block,
                mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                mask_fn=partial(
                    apply_arbitrary_forward_mask,
                    mask_payloads=mask_payloads,
                    payload_idx=payload_idx,
                    payload_group_idx=payload_group_idx,
                    payload_words=payload_words,
                    r_bitmask=r_bitmask,
                ),
                is_first_n_block=True,
            )
            for iteration in cutlass.range(1, partial_block_cnt, unroll=1):
                partial_list_idx = partial_block_cnt - Int32(1) - iteration
                payload_idx = partial_payload_base + partial_list_idx
                n_block = Int32(0)
                if const_expr(partial_block_idx is not None):
                    n_block = partial_block_idx[partial_list_idx]
                r_bitmask = None
                if const_expr(pipeline_mask is not None):
                    assert mask_consumer_state is not None
                    assert sMask is not None
                    r_bitmask, mask_consumer_state = consume_mask_payload_from_smem(
                        sMask[None, None, mask_consumer_state.index],
                        payload_group_idx,
                        pipeline_mask,
                        mask_consumer_state,
                    )
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                    mask_fn=partial(
                        apply_arbitrary_forward_mask,
                        mask_payloads=mask_payloads,
                        payload_idx=payload_idx,
                        payload_group_idx=payload_group_idx,
                        payload_words=payload_words,
                        r_bitmask=r_bitmask,
                    ),
                    is_first_n_block=False,
                )
            full_start = Int32(0)
        else:
            full_list_idx = full_block_cnt - Int32(1)
            n_block = Int32(0)
            if const_expr(full_block_idx is not None):
                n_block = full_block_idx[full_list_idx]
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=n_block,
                mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                mask_fn=None,
                is_first_n_block=True,
            )
            full_start = Int32(1)
        for iteration in cutlass.range(full_start, full_block_cnt, unroll=1):
            full_list_idx = full_block_cnt - Int32(1) - iteration
            n_block = Int32(0)
            if const_expr(full_block_idx is not None):
                n_block = full_block_idx[full_list_idx]
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=n_block,
                mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                mask_fn=None,
                is_first_n_block=False,
            )
        warp_scheduler_barrier_arrive()
    return kv_consumer_state, processed_any, mask_consumer_state


@cute.jit
def consume_arbitrary_forward_overlap(
    partial_block_cnt,
    partial_block_idx: Optional[cute.Tensor],
    full_block_cnt,
    full_block_idx: Optional[cute.Tensor],
    partial_payload_base: Int32,
    mask_payloads: cute.Tensor,
    seqlen_info,
    kv_consumer_state,
    mma_pv_fn: Callable,
    mma_one_n_block: Callable,
    process_first_half_block: Callable,
    process_last_half_block: Callable,
    payload_group_idx: Int32,
    payload_words: cutlass.Constexpr[int],
    pipeline_mask=None,
    mask_consumer_state=None,
    sMask: Optional[cute.Tensor] = None,
):
    """Consume partial/full CSR loops with K/V overlap."""
    processed_any = partial_block_cnt + full_block_cnt > Int32(0)
    O_should_accumulate = False
    full_start = Int32(0)
    if processed_any:
        if partial_block_cnt > Int32(0):
            partial_list_idx = partial_block_cnt - Int32(1)
            payload_idx = partial_payload_base + partial_list_idx
            n_block = Int32(0)
            if const_expr(partial_block_idx is not None):
                n_block = partial_block_idx[partial_list_idx]
            if const_expr(pipeline_mask is not None):
                assert mask_consumer_state is not None
                assert sMask is not None
                r_bitmask, mask_consumer_state = consume_mask_payload_from_smem(
                    sMask[None, None, mask_consumer_state.index],
                    payload_group_idx,
                    pipeline_mask,
                    mask_consumer_state,
                )
                kv_consumer_state = process_first_half_block(
                    n_block=n_block,
                    seqlen=seqlen_info,
                    kv_consumer_state=kv_consumer_state,
                    mask_fn=partial(
                        apply_arbitrary_forward_mask,
                        mask_payloads=mask_payloads,
                        payload_idx=payload_idx,
                        payload_group_idx=payload_group_idx,
                        payload_words=payload_words,
                        r_bitmask=r_bitmask,
                    ),
                    mask_prefetch_fn=None,
                    is_first_block=True,
                )
            else:
                kv_consumer_state = process_first_half_block(
                    n_block=n_block,
                    seqlen=seqlen_info,
                    kv_consumer_state=kv_consumer_state,
                    mask_fn=partial(
                        apply_arbitrary_forward_mask,
                        mask_payloads=mask_payloads,
                        payload_idx=payload_idx,
                        payload_group_idx=payload_group_idx,
                        payload_words=payload_words,
                    ),
                    mask_prefetch_fn=partial(
                        load_mask_payload,
                        mask_payloads,
                        payload_idx,
                        payload_group_idx,
                        payload_words=payload_words,
                    ),
                    is_first_block=True,
                )
            for iteration in cutlass.range(1, partial_block_cnt, unroll=1):
                partial_list_idx = partial_block_cnt - Int32(1) - iteration
                payload_idx = partial_payload_base + partial_list_idx
                n_block = Int32(0)
                if const_expr(partial_block_idx is not None):
                    n_block = partial_block_idx[partial_list_idx]
                if const_expr(pipeline_mask is not None):
                    assert mask_consumer_state is not None
                    assert sMask is not None
                    r_bitmask, mask_consumer_state = consume_mask_payload_from_smem(
                        sMask[None, None, mask_consumer_state.index],
                        payload_group_idx,
                        pipeline_mask,
                        mask_consumer_state,
                    )
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block,
                        seqlen=seqlen_info,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(
                            apply_arbitrary_forward_mask,
                            mask_payloads=mask_payloads,
                            payload_idx=payload_idx,
                            payload_group_idx=payload_group_idx,
                            payload_words=payload_words,
                            r_bitmask=r_bitmask,
                        ),
                        mask_prefetch_fn=None,
                    )
                else:
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block,
                        seqlen=seqlen_info,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(
                            apply_arbitrary_forward_mask,
                            mask_payloads=mask_payloads,
                            payload_idx=payload_idx,
                            payload_group_idx=payload_group_idx,
                            payload_words=payload_words,
                        ),
                        mask_prefetch_fn=partial(
                            load_mask_payload,
                            mask_payloads,
                            payload_idx,
                            payload_group_idx,
                            payload_words=payload_words,
                        ),
                    )
                O_should_accumulate = True
            full_start = Int32(0)
        else:
            full_list_idx = full_block_cnt - Int32(1)
            n_block = Int32(0)
            if const_expr(full_block_idx is not None):
                n_block = full_block_idx[full_list_idx]
            kv_consumer_state = process_first_half_block(
                n_block=n_block,
                seqlen=seqlen_info,
                kv_consumer_state=kv_consumer_state,
                mask_fn=None,
                mask_prefetch_fn=None,
                is_first_block=True,
            )
            full_start = Int32(1)

        for iteration in cutlass.range(full_start, full_block_cnt, unroll=1):
            full_list_idx = full_block_cnt - Int32(1) - iteration
            n_block = Int32(0)
            if const_expr(full_block_idx is not None):
                n_block = full_block_idx[full_list_idx]
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=n_block,
                seqlen=seqlen_info,
                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                mask_fn=None,
                mask_prefetch_fn=None,
            )
            O_should_accumulate = True

        kv_consumer_state = process_last_half_block(
            kv_consumer_state=kv_consumer_state,
            zero_init=not O_should_accumulate,
        )
        O_should_accumulate = True
    return kv_consumer_state, O_should_accumulate, processed_any, mask_consumer_state


@cute.jit
def produce_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen_info: SeqlenInfoQK,
    kv_producer_state,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    intra_wg_overlap: cutlass.Constexpr,
    tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    o_empty_mbar_ptr: Optional[cute.Pointer] = None,
    o_empty_phase: Optional[Int32] = None,
    pipeline_mask=None,
    mask_producer_state=None,
    sMask: Optional[cute.Tensor] = None,
    payload_groups: cutlass.Constexpr[int] = 0,
    payload_words: cutlass.Constexpr[int] = 4,
):
    """Produce K/V loads for one arbitrary-mask plan row on SM90."""
    mask_payloads = blocksparse_tensors.mask_block_masks
    assert mask_payloads is not None
    (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
        mask_payload_base,
    ) = get_curr_arbitrary_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block,
        blocksparse_tensors,
        seqlen_info,
        tile_m,
        qhead_per_kvhead,
    )
    if const_expr(not intra_wg_overlap):
        kv_producer_state, mask_producer_state = produce_arbitrary_forward_nonoverlap(
            curr_mask_block_cnt,
            curr_mask_block_idx,
            curr_full_block_cnt,
            curr_full_block_idx,
            mask_payload_base,
            mask_payloads,
            payload_words,
            kv_producer_state,
            load_K,
            load_V,
            pipeline_k,
            pipeline_v,
            o_empty_mbar_ptr,
            o_empty_phase,
            pipeline_mask,
            mask_producer_state,
            sMask,
            payload_groups,
        )
    else:
        kv_producer_state, mask_producer_state = produce_arbitrary_forward_overlap(
            curr_mask_block_cnt,
            curr_mask_block_idx,
            curr_full_block_cnt,
            curr_full_block_idx,
            mask_payload_base,
            mask_payloads,
            payload_words,
            kv_producer_state,
            load_K,
            load_V,
            pipeline_k,
            pipeline_v,
            o_empty_mbar_ptr,
            o_empty_phase,
            pipeline_mask,
            mask_producer_state,
            sMask,
            payload_groups,
        )
    if const_expr(pipeline_mask is not None):
        return kv_producer_state, mask_producer_state
    return kv_producer_state


@cute.jit
def consume_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen_info,
    kv_consumer_state,
    mma_pv_fn,
    mma_one_n_block,
    process_first_half_block,
    process_last_half_block,
    intra_wg_overlap: cutlass.Constexpr,
    warp_scheduler_barrier_sync: Callable,
    warp_scheduler_barrier_arrive: Callable,
    tile_m: cutlass.Constexpr[int],
    consumer_tidx: Int32,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    payload_words: cutlass.Constexpr[int] = 4,
    pipeline_mask=None,
    mask_consumer_state=None,
    sMask: Optional[cute.Tensor] = None,
):
    """Consume one arbitrary-mask plan row on the SM90 MMA warp group."""
    mask_payloads = blocksparse_tensors.mask_block_masks
    assert mask_payloads is not None
    (
        curr_mask_block_cnt,
        curr_full_block_cnt,
        mask_payload_base,
    ) = get_curr_arbitrary_block_counts_fwd_sm90(
        batch_idx,
        head_idx,
        m_block,
        blocksparse_tensors,
        seqlen_info,
        tile_m,
        qhead_per_kvhead,
    )
    payload_group_idx = _sm90_fwd_mask_payload_group_idx(
        consumer_tidx,
        qhead_per_kvhead,
    )
    if const_expr(not intra_wg_overlap):
        kv_consumer_state, processed_any, mask_consumer_state = consume_arbitrary_forward_nonoverlap(
            curr_mask_block_cnt,
            None,
            curr_full_block_cnt,
            None,
            mask_payload_base,
            mask_payloads,
            kv_consumer_state,
            mma_pv_fn,
            mma_one_n_block,
            payload_group_idx,
            payload_words,
            warp_scheduler_barrier_sync,
            warp_scheduler_barrier_arrive,
            pipeline_mask,
            mask_consumer_state,
            sMask,
        )
        if const_expr(pipeline_mask is not None):
            return kv_consumer_state, processed_any, processed_any, mask_consumer_state
        return kv_consumer_state, processed_any, processed_any
    kv_consumer_state, O_should_accumulate, processed_any, mask_consumer_state = consume_arbitrary_forward_overlap(
        curr_mask_block_cnt,
        None,
        curr_full_block_cnt,
        None,
        mask_payload_base,
        mask_payloads,
        seqlen_info,
        kv_consumer_state,
        mma_pv_fn,
        mma_one_n_block,
        process_first_half_block,
        process_last_half_block,
        payload_group_idx,
        payload_words,
        pipeline_mask,
        mask_consumer_state,
        sMask,
    )
    if const_expr(pipeline_mask is not None):
        return kv_consumer_state, O_should_accumulate, processed_any, mask_consumer_state
    return kv_consumer_state, O_should_accumulate, processed_any


@cute.jit
def produce_arbitrary_forward_loads_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx: Int32,
    head_idx: Int32,
    m_block: Int32,
    seqlen_info: SeqlenInfoQK,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    q_stage: cutlass.Constexpr,
    payload_cta_rank: Int32,
    cta_group_size: cutlass.Constexpr[int],
    q_producer_phase: Int32,
    sparse_tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
    payload_words: cutlass.Constexpr[int],
):
    """Produce compact Q2K payload loads for generic SM100 forward."""

    (
        partial_count,
        partial_indices,
        full_count,
        full_indices,
        payload_base,
    ) = get_curr_arbitrary_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block,
        blocksparse_tensors,
        seqlen_info,
        sparse_tile_m,
        qhead_per_kvhead,
    )
    q_phase_flipped = False
    if partial_count > 0:
        # Keep the callable producer closures at this single JIT boundary.
        # CuTe DSL 4.6.1 cannot re-stage a captured load_Q/K/V callable through
        # a second decorated helper (it rejects the resulting code object for
        # having free variables).
        load_Q(block=0, stage=0)
        if const_expr(q_stage == 2):
            load_Q(block=1, stage=1)

        current_ordinal = partial_count - Int32(1)
        prefetch_arbitrary_forward_block_index(
            partial_indices,
            current_ordinal,
            True,
        )
        for stage_idx in cutlass.range_constexpr(q_stage):
            payload_subtile_idx = Int32(stage_idx) * cta_group_size + payload_cta_rank
            prefetch_arbitrary_forward_mask(
                blocksparse_tensors.mask_block_masks,
                payload_base + current_ordinal,
                True,
                payload_words,
                payload_subtile_idx,
            )
        for offset in cutlass.range(partial_count, unroll=1):
            partial_ordinal = partial_count - Int32(1) - offset
            n_block = partial_indices[partial_ordinal]
            if offset + Int32(1) < partial_count:
                next_ordinal = partial_ordinal - Int32(1)
                prefetch_arbitrary_forward_block_index(
                    partial_indices,
                    next_ordinal,
                    True,
                )
                for stage_idx in cutlass.range_constexpr(q_stage):
                    payload_subtile_idx = Int32(stage_idx) * cta_group_size + payload_cta_rank
                    prefetch_arbitrary_forward_mask(
                        blocksparse_tensors.mask_block_masks,
                        payload_base + next_ordinal,
                        True,
                        payload_words,
                        payload_subtile_idx,
                    )
            load_K(block=n_block, producer_state=kv_producer_state)
            kv_producer_state.advance()
            load_V(block=n_block, producer_state=kv_producer_state)
            kv_producer_state.advance()

        q_phase_flipped = True
    elif full_count > 0:
        load_Q(block=0, stage=0)
        if const_expr(q_stage == 2):
            load_Q(block=1, stage=1)
        q_phase_flipped = True

    if full_count > 0:
        current_ordinal = full_count - Int32(1)
        prefetch_arbitrary_forward_block_index(
            full_indices,
            current_ordinal,
            True,
        )
        for offset in cutlass.range(full_count, unroll=1):
            full_ordinal = full_count - Int32(1) - offset
            n_block = full_indices[full_ordinal]
            if offset + Int32(1) < full_count:
                prefetch_arbitrary_forward_block_index(
                    full_indices,
                    full_ordinal - Int32(1),
                    True,
                )
            load_K(block=n_block, producer_state=kv_producer_state)
            kv_producer_state.advance()
            load_V(block=n_block, producer_state=kv_producer_state)
            kv_producer_state.advance()

    if q_phase_flipped:
        q_producer_phase ^= 1
    return kv_producer_state, q_producer_phase


@cute.jit
def _get_arbitrary_forward_block_by_traversal_ordinal(
    traversal_ordinal: Int32,
    partial_count: Int32,
    partial_indices: cute.Tensor,
    full_count: Int32,
    full_indices: cute.Tensor,
) -> Int32:
    """Map the forward traversal ordinal to its compact plan block index."""

    n_block = Int32(0)
    if traversal_ordinal < partial_count:
        n_block = partial_indices[partial_count - Int32(1) - traversal_ordinal]
    else:
        full_traversal_ordinal = traversal_ordinal - partial_count
        n_block = full_indices[full_count - Int32(1) - full_traversal_ordinal]
    return n_block


@cute.jit
def _produce_arbitrary_forward_mask_by_traversal_ordinal(
    traversal_ordinal: Int32,
    partial_count: Int32,
    payload_base: Int32,
    mask_payloads: cute.Tensor,
    payload_subtile_idx: Int32,
    payload_groups: cutlass.Constexpr[int],
    payload_words: cutlass.Constexpr[int],
    pipeline_mask_s0,
    pipeline_mask_s1,
    mask_s0_producer_state,
    mask_s1_producer_state,
    sMask: cute.Tensor,
):
    """Issue the mask copy owned by one qstage1 traversal ordinal."""

    if traversal_ordinal < partial_count:
        partial_ordinal = partial_count - Int32(1) - traversal_ordinal
        if (traversal_ordinal & Int32(1)) == Int32(0):
            mask_s0_producer_state = load_mask_payload_to_smem(
                mask_payloads,
                payload_base + partial_ordinal,
                payload_subtile_idx,
                sMask[None, None, 0],
                pipeline_mask_s0,
                mask_s0_producer_state,
                payload_groups,
                payload_words,
            )
        else:
            mask_s1_producer_state = load_mask_payload_to_smem(
                mask_payloads,
                payload_base + partial_ordinal,
                payload_subtile_idx,
                sMask[None, None, 1],
                pipeline_mask_s1,
                mask_s1_producer_state,
                payload_groups,
                payload_words,
            )
    return mask_s0_producer_state, mask_s1_producer_state


@cute.jit
def produce_arbitrary_forward_loads_qstage1_n_direction_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx: Int32,
    head_idx: Int32,
    m_block: Int32,
    seqlen_info: SeqlenInfoQK,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    q_producer_phase: Int32,
    sparse_tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
    payload_subtile_idx: Int32 = Int32(0),
    payload_groups: cutlass.Constexpr[int] = 128,
    payload_words: cutlass.Constexpr[int] = 4,
    pipeline_mask_s0=None,
    pipeline_mask_s1=None,
    mask_s0_producer_state=None,
    mask_s1_producer_state=None,
    sMask: Optional[cute.Tensor] = None,
):
    """Produce K/K/V interleaving for the generic qstage1 N-direction mainloop."""

    (
        partial_count,
        partial_indices,
        full_count,
        full_indices,
        payload_base,
    ) = get_curr_arbitrary_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block,
        blocksparse_tensors,
        seqlen_info,
        sparse_tile_m,
        qhead_per_kvhead,
    )
    total_count = partial_count + full_count
    mask_payloads = blocksparse_tensors.mask_block_masks
    if const_expr(pipeline_mask_s0 is not None):
        assert mask_payloads is not None
        assert pipeline_mask_s1 is not None
        assert sMask is not None
    if total_count > Int32(0):
        load_Q(block=0, stage=0)
        n_block_0 = _get_arbitrary_forward_block_by_traversal_ordinal(
            Int32(0),
            partial_count,
            partial_indices,
            full_count,
            full_indices,
        )
        if const_expr(pipeline_mask_s0 is not None):
            mask_s0_producer_state, mask_s1_producer_state = _produce_arbitrary_forward_mask_by_traversal_ordinal(
                Int32(0),
                partial_count,
                payload_base,
                mask_payloads,
                payload_subtile_idx,
                payload_groups,
                payload_words,
                pipeline_mask_s0,
                pipeline_mask_s1,
                mask_s0_producer_state,
                mask_s1_producer_state,
                sMask,
            )
        load_K(block=n_block_0, producer_state=kv_producer_state)
        kv_producer_state.advance()

        if total_count == Int32(1):
            load_V(block=n_block_0, producer_state=kv_producer_state)
            kv_producer_state.advance()
        else:
            n_block_1 = _get_arbitrary_forward_block_by_traversal_ordinal(
                Int32(1),
                partial_count,
                partial_indices,
                full_count,
                full_indices,
            )
            if const_expr(pipeline_mask_s0 is not None):
                mask_s0_producer_state, mask_s1_producer_state = _produce_arbitrary_forward_mask_by_traversal_ordinal(
                    Int32(1),
                    partial_count,
                    payload_base,
                    mask_payloads,
                    payload_subtile_idx,
                    payload_groups,
                    payload_words,
                    pipeline_mask_s0,
                    pipeline_mask_s1,
                    mask_s0_producer_state,
                    mask_s1_producer_state,
                    sMask,
                )
            load_K(block=n_block_1, producer_state=kv_producer_state)
            kv_producer_state.advance()

            for traversal_ordinal in cutlass.range(Int32(2), total_count, unroll=1):
                v_block = _get_arbitrary_forward_block_by_traversal_ordinal(
                    traversal_ordinal - Int32(2),
                    partial_count,
                    partial_indices,
                    full_count,
                    full_indices,
                )
                load_V(block=v_block, producer_state=kv_producer_state)
                kv_producer_state.advance()

                k_block = _get_arbitrary_forward_block_by_traversal_ordinal(
                    traversal_ordinal,
                    partial_count,
                    partial_indices,
                    full_count,
                    full_indices,
                )
                if const_expr(pipeline_mask_s0 is not None):
                    mask_s0_producer_state, mask_s1_producer_state = _produce_arbitrary_forward_mask_by_traversal_ordinal(
                        traversal_ordinal,
                        partial_count,
                        payload_base,
                        mask_payloads,
                        payload_subtile_idx,
                        payload_groups,
                        payload_words,
                        pipeline_mask_s0,
                        pipeline_mask_s1,
                        mask_s0_producer_state,
                        mask_s1_producer_state,
                        sMask,
                    )
                load_K(block=k_block, producer_state=kv_producer_state)
                kv_producer_state.advance()

            for traversal_ordinal in cutlass.range(total_count - Int32(2), total_count, unroll=1):
                v_block = _get_arbitrary_forward_block_by_traversal_ordinal(
                    traversal_ordinal,
                    partial_count,
                    partial_indices,
                    full_count,
                    full_indices,
                )
                load_V(block=v_block, producer_state=kv_producer_state)
                kv_producer_state.advance()

        q_producer_phase ^= 1
    if const_expr(pipeline_mask_s0 is not None):
        return (
            kv_producer_state,
            q_producer_phase,
            mask_s0_producer_state,
            mask_s1_producer_state,
        )
    return kv_producer_state, q_producer_phase


# SM100-specific tile processor using SM100 helpers
@cute.jit
def get_total_arbitrary_block_count_fwd_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx: Int32,
    head_idx: Int32,
    m_block: Int32,
    seqlen_info: SeqlenInfoQK,
    sparse_tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
):
    """Count one compact Q2K plan row without using fixed-linear shapes."""

    partial_count, full_count, _ = get_curr_arbitrary_block_counts_fwd_sm90(
        batch_idx,
        head_idx,
        m_block,
        blocksparse_tensors,
        seqlen_info,
        sparse_tile_m,
        qhead_per_kvhead,
    )
    return partial_count + full_count


@cute.jit
def handle_block_sparse_empty_tile_correction_sm100(
    tidx: Int32,
    q_stage: cutlass.Constexpr,
    m_block_size: cutlass.Constexpr,
    mLSE,
    seqlen_info,
    m_block: Int32,
    sScale: cute.Tensor,
    stats: list,
    correction_epilogue: Callable,
    thr_mma_pv: cute.ThrMma,
    tOtO: cute.Tensor,
    sO: cute.Tensor,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    pipeline_o_epi: cutlass.pipeline.PipelineAsync,
    sm_stats_consumer_phase: Int32,
    o_corr_consumer_phase: Int32,
    corr_epi_producer_phase: Int32,
    mO_cur: Optional[cute.Tensor] = None,
    gO: Optional[cute.Tensor] = None,
    gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
):
    """Handle SM100 forward block-sparse tiles with no active KV blocks.

    This path is taken when `total_block_cnt == 0`. The softmax warp-group still
    arrives `mbar_softmax_corr_full` (synthetic "no work") so the correction
    warp-group can:

    - seed fully-masked-row stats (row_sum=1; row_max=-inf when tracked) for LSE
    - run `correction_epilogue` with `scale=0` so the output tile is written as zeros
      (independent of any prior tmem contents)
    - wait on `mbar_softmax_corr_full` and arrive `mbar_softmax_corr_empty`
      (and `mbar_corr_epi_*` when applicable) so phases stay aligned across tiles

    This helper intentionally does not touch `mbar_P_full_*` since no P is produced.
    See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
    """
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

    for stage in cutlass.range_constexpr(q_stage):
        row_sum_value = Float32(1.0)
        row_max_value = -Float32.inf if const_expr(mLSE is not None) else None
        if tidx < m_block_size:
            scale_row_idx = tidx + stage * m_block_size
            sScale[scale_row_idx] = row_sum_value
            if const_expr(mLSE is not None):
                sScale[scale_row_idx + q_stage * m_block_size] = row_max_value
        acc_flag = row_sum_value == Float32(0.0) or row_sum_value != row_sum_value
        stats[stage] = (row_sum_value, row_max_value, acc_flag)

        # See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
        pipeline_sm_stats.consumer_release_w_index(stage)

        if const_expr(gmem_tiled_copy_O is None):
            pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)

        gO_stage = gO[None, None, stage] if const_expr(gO is not None) else None
        correction_epilogue(
            thr_mma_pv,
            tOtO[None, None, None, stage],
            tidx,
            stage,
            m_block,
            seqlen_info.seqlen_q,
            Float32(0.0),  # zero scale ensures empty tile writes zeros into staged outputs
            sO[None, None, stage],
            mO_cur,
            gO_stage,
            gmem_tiled_copy_O,
        )
        if const_expr(gmem_tiled_copy_O is None):
            pipeline_o_epi.producer_commit_w_index(stage)

    sm_stats_consumer_phase ^= 1
    corr_epi_producer_phase ^= 1

    return (
        sm_stats_consumer_phase,
        o_corr_consumer_phase,
        corr_epi_producer_phase,
    )


@cute.jit
def softmax_arbitrary_forward_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx: Int32,
    head_idx: Int32,
    m_block: Int32,
    seqlen_info: SeqlenInfoQK,
    softmax_step: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    s0_s1_sequence_phase: Int32,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    stage_idx: Int32,
    payload_subtile_idx: Int32,
    payload_group_idx: Int32,
    sparse_tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
    payload_words: cutlass.Constexpr[int],
):
    """Consume one compact Q2K row with a TMEM-native mask payload."""

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
    (
        partial_count,
        partial_indices,
        full_count,
        full_indices,
        payload_base,
    ) = get_curr_arbitrary_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block,
        blocksparse_tensors,
        seqlen_info,
        sparse_tile_m,
        qhead_per_kvhead,
    )
    total_count = partial_count + full_count
    mask_payloads = blocksparse_tensors.mask_block_masks
    assert mask_payloads is not None

    if total_count == 0:
        sm_stats_barrier.arrive_w_index(index=stage_idx * 4 + warp_idx)
    else:
        if partial_count > 0:
            partial_ordinal = partial_count - Int32(1)
            n_block = partial_indices[partial_ordinal]
            payload_idx = payload_base + partial_ordinal
            # This global-to-register load intentionally precedes the score
            # pipeline wait inside softmax_step. The four words remain live
            # while QK completes, then mask the score.
            r_bitmask = load_mask_payload(
                mask_payloads,
                payload_idx,
                payload_group_idx,
                subtile_idx=payload_subtile_idx,
                payload_words=payload_words,
            )
            (
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
            ) = softmax_step(
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
                n_block,
                is_first=True,
                mask_fn=partial(
                    apply_loaded_arbitrary_forward_mask,
                    r_bitmask=r_bitmask,
                ),
            )
            for offset in cutlass.range(1, partial_count, unroll=1):
                partial_ordinal = partial_count - Int32(1) - offset
                n_block = partial_indices[partial_ordinal]
                payload_idx = payload_base + partial_ordinal
                r_bitmask = load_mask_payload(
                    mask_payloads,
                    payload_idx,
                    payload_group_idx,
                    subtile_idx=payload_subtile_idx,
                    payload_words=payload_words,
                )
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    n_block,
                    mask_fn=partial(
                        apply_loaded_arbitrary_forward_mask,
                        r_bitmask=r_bitmask,
                    ),
                )

        if full_count > 0:
            full_ordinal = full_count - Int32(1)
            n_block = full_indices[full_ordinal]
            if partial_count == 0:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    n_block,
                    is_first=True,
                    mask_fn=None,
                )
            else:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    n_block,
                    mask_fn=None,
                )
            for offset in cutlass.range(1, full_count, unroll=1):
                full_ordinal = full_count - Int32(1) - offset
                n_block = full_indices[full_ordinal]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    n_block,
                    mask_fn=None,
                )

    return (
        mma_si_consumer_phase,
        si_corr_producer_phase,
        s0_s1_sequence_phase,
        total_count == 0,
    )


@cute.jit
def softmax_arbitrary_forward_qstage1_n_direction_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx: Int32,
    head_idx: Int32,
    m_block: Int32,
    seqlen_info: SeqlenInfoQK,
    softmax_step: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    stage_idx: cutlass.Constexpr[int],
    payload_subtile_idx: Int32,
    payload_group_idx: Int32,
    sparse_tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
    payload_words: cutlass.Constexpr[int],
    pipeline_mask=None,
    mask_consumer_state=None,
    sMask: Optional[cute.Tensor] = None,
):
    """Consume one parity stream while preserving partial-then-full order."""

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
    (
        partial_count,
        partial_indices,
        full_count,
        full_indices,
        payload_base,
    ) = get_curr_arbitrary_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block,
        blocksparse_tensors,
        seqlen_info,
        sparse_tile_m,
        qhead_per_kvhead,
    )
    total_count = partial_count + full_count
    stream_has_work = total_count > Int32(stage_idx)
    mask_payloads = blocksparse_tensors.mask_block_masks
    assert mask_payloads is not None

    if stream_has_work:
        first_traversal_ordinal = Int32(stage_idx)
        if first_traversal_ordinal < partial_count:
            partial_ordinal = partial_count - Int32(1) - first_traversal_ordinal
            n_block = partial_indices[partial_ordinal]
            if const_expr(pipeline_mask is not None):
                assert sMask is not None
                r_bitmask, mask_consumer_state = consume_mask_payload_from_smem(
                    sMask,
                    payload_group_idx,
                    pipeline_mask,
                    mask_consumer_state,
                )
            else:
                r_bitmask = load_mask_payload(
                    mask_payloads,
                    payload_base + partial_ordinal,
                    payload_group_idx,
                    subtile_idx=payload_subtile_idx,
                    payload_words=payload_words,
                )
            mma_si_consumer_phase, si_corr_producer_phase, _ = softmax_step(
                mma_si_consumer_phase,
                si_corr_producer_phase,
                Int32(0),
                n_block,
                is_first=True,
                mask_fn=partial(
                    apply_loaded_arbitrary_forward_mask,
                    r_bitmask=r_bitmask,
                ),
            )

            partial_stream_count = (partial_count - Int32(stage_idx) + Int32(1)) // Int32(2)
            for stream_ordinal in cutlass.range(Int32(1), partial_stream_count, unroll=1):
                traversal_ordinal = Int32(stage_idx) + stream_ordinal * Int32(2)
                partial_ordinal = partial_count - Int32(1) - traversal_ordinal
                n_block = partial_indices[partial_ordinal]
                if const_expr(pipeline_mask is not None):
                    r_bitmask, mask_consumer_state = consume_mask_payload_from_smem(
                        sMask,
                        payload_group_idx,
                        pipeline_mask,
                        mask_consumer_state,
                    )
                else:
                    r_bitmask = load_mask_payload(
                        mask_payloads,
                        payload_base + partial_ordinal,
                        payload_group_idx,
                        subtile_idx=payload_subtile_idx,
                        payload_words=payload_words,
                    )
                mma_si_consumer_phase, si_corr_producer_phase, _ = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    Int32(0),
                    n_block,
                    mask_fn=partial(
                        apply_loaded_arbitrary_forward_mask,
                        r_bitmask=r_bitmask,
                    ),
                )

        full_start = (Int32(stage_idx) - partial_count) & Int32(1)
        full_is_first = first_traversal_ordinal >= partial_count
        if full_is_first:
            full_ordinal = full_count - Int32(1) - full_start
            n_block = full_indices[full_ordinal]
            mma_si_consumer_phase, si_corr_producer_phase, _ = softmax_step(
                mma_si_consumer_phase,
                si_corr_producer_phase,
                Int32(0),
                n_block,
                is_first=True,
                mask_fn=None,
            )
            full_start += Int32(2)

        if full_start < full_count:
            full_stream_count = (full_count - full_start + Int32(1)) // Int32(2)
            for stream_ordinal in cutlass.range(full_stream_count, unroll=1):
                full_traversal_ordinal = full_start + stream_ordinal * Int32(2)
                full_ordinal = full_count - Int32(1) - full_traversal_ordinal
                n_block = full_indices[full_ordinal]
                mma_si_consumer_phase, si_corr_producer_phase, _ = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    Int32(0),
                    n_block,
                    mask_fn=None,
                )
    else:
        sm_stats_barrier.arrive_w_index(index=stage_idx * 4 + warp_idx)

    if const_expr(pipeline_mask is not None):
        return (
            mma_si_consumer_phase,
            si_corr_producer_phase,
            not stream_has_work,
            mask_consumer_state,
        )
    return mma_si_consumer_phase, si_corr_producer_phase, not stream_has_work


# =============================================================================
# Backward-specific block-sparse helpers (SM100)
# =============================================================================
#
# In backward, iteration is transposed compared to forward:
# - Forward: outer loop over m_blocks (Q tiles), inner loop over n_blocks (KV tiles)
# - Backward: outer loop over n_blocks (KV tiles), inner loop over m_blocks (Q tiles)
#
# The backward block-sparse tensors use "Q direction" indexing:
# - q_block_cnt[batch, head, n_block] → count of m_blocks to process for this KV tile
# - q_block_idx[batch, head, n_block, :] → indices of m_blocks to process
#


@cute.jit
def get_total_q_block_count_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    n_blocks_per_sample: int = 0,
):
    """Count total tile iterations for given n_block (KV tile) in backward."""
    _, _, _, _, _, _, total = get_block_sparse_iteration_info_bwd(
        blocksparse_tensors,
        batch_idx,
        head_idx,
        n_block,
        subtile_factor=subtile_factor,
        m_block_max=m_block_max,
        n_blocks_per_sample=n_blocks_per_sample,
    )
    return total


@cute.jit
def produce_block_sparse_q_loads_bwd_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    # Pipeline states (will be returned after advancing)
    producer_state_Q_LSE,
    producer_state_dO_dPsum,
    # Pipelines
    pipeline_Q,
    pipeline_LSE,
    pipeline_dO,
    pipeline_dPsum,
    # Load functions
    load_K,
    load_V,
    load_Q,
    load_dO,
    copy_stats,
    # Global tensors for LSE/dPsum
    gLSE,
    sLSE,
    gdPsum,
    sdPsum,
    # TMA copy bytes for extra_tx_count
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    # Flags for which loads to perform
    should_load_Q: cutlass.Constexpr,
    should_load_dO: cutlass.Constexpr,
    # Subtiling factor and bounds
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    n_blocks_per_sample: int = 0,
):
    """SM100 backward block sparse loading with subtiling.

    Returns updated (producer_state_Q_LSE, producer_state_dO_dPsum).
    First iteration loads K/V alongside Q/dO; subsequent iterations load only Q/dO.
    """
    (
        curr_q_cnt,
        curr_q_idx,
        curr_full_cnt,
        curr_full_idx,
        _,
        _,
        loop_count,
    ) = get_block_sparse_iteration_info_bwd(
        blocksparse_tensors,
        batch_idx,
        head_idx,
        n_block,
        subtile_factor,
        m_block_max,
        n_blocks_per_sample,
    )

    split_sparse_blocks = const_expr(curr_full_idx is not None)
    block_group_count: cutlass.Constexpr[int] = 2 if const_expr(split_sparse_blocks) else 1
    for block_group in cutlass.range_constexpr(block_group_count):
        group_loop_count = loop_count
        iter_offset = Int32(0)
        if const_expr(split_sparse_blocks):
            group_loop_count = curr_q_cnt * subtile_factor
            if const_expr(block_group == 1):
                group_loop_count = curr_full_cnt * subtile_factor
                iter_offset = curr_q_cnt * subtile_factor

        for group_iter_idx in cutlass.range(group_loop_count, unroll=1):
            iter_idx = group_iter_idx + iter_offset
            if const_expr(split_sparse_blocks):
                sparse_iter_idx = group_iter_idx // subtile_factor
                subtile_offset = group_iter_idx % subtile_factor
                if const_expr(block_group == 0):
                    m_block = curr_q_idx[sparse_iter_idx] * subtile_factor + subtile_offset
                else:
                    assert curr_full_idx is not None
                    m_block = curr_full_idx[sparse_iter_idx] * subtile_factor + subtile_offset
            else:
                m_block, _ = get_m_block_from_iter_bwd(
                    iter_idx,
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    subtile_factor,
                    m_block_max,
                )
            m_block_safe = m_block
            if m_block_max > 0:
                m_block_safe = cutlass.min(m_block, m_block_max - 1)

            if iter_idx == 0:
                # First block: load K/V alongside Q/dO
                if const_expr(should_load_Q):
                    pipeline_Q.producer_acquire(producer_state_Q_LSE, extra_tx_count=tma_copy_bytes_K)
                    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                    load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)
                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                    copy_stats(
                        gLSE[None, m_block_safe],
                        sLSE[None, producer_state_Q_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                    )
                    producer_state_Q_LSE.advance()
                if const_expr(should_load_dO):
                    pipeline_dO.producer_acquire(producer_state_dO_dPsum, extra_tx_count=tma_copy_bytes_V)
                    load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                    load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)
                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                    copy_stats(
                        gdPsum[None, m_block_safe],
                        sdPsum[None, producer_state_dO_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                    )
                    producer_state_dO_dPsum.advance()
            else:
                # Subsequent blocks: just load Q/dO (K/V already loaded)
                if const_expr(should_load_Q):
                    pipeline_Q.producer_acquire(producer_state_Q_LSE)
                    load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)
                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                    copy_stats(
                        gLSE[None, m_block_safe],
                        sLSE[None, producer_state_Q_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                    )
                    producer_state_Q_LSE.advance()
                if const_expr(should_load_dO):
                    pipeline_dO.producer_acquire(producer_state_dO_dPsum)
                    load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)
                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                    copy_stats(
                        gdPsum[None, m_block_safe],
                        sdPsum[None, producer_state_dO_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                    )
                    producer_state_dO_dPsum.advance()

    return producer_state_Q_LSE, producer_state_dO_dPsum


@cute.jit
def get_block_sparse_iteration_info_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    n_blocks_per_sample: int = 0,
):
    """Extract block-sparse iteration info for backward pass.

    Returns partial/full counts and indices, their compact payload bases, and
    the total physical iteration count.  Every SM100 role uses this accessor
    so fixed and varlen K2Q rows cannot drift between producer and consumers.
    """
    assert blocksparse_tensors.mask_block_masks is not None
    (
        curr_q_cnt,
        curr_q_idx,
        curr_full_cnt,
        curr_full_idx,
        partial_base,
        full_base,
    ) = get_curr_arbitrary_blocksparse_tensors_bwd(
        batch_idx,
        head_idx,
        n_block,
        blocksparse_tensors,
        n_blocks_per_sample,
    )

    sparse_block_count = curr_q_cnt
    if const_expr(curr_full_idx is not None):
        sparse_block_count = sparse_block_count + curr_full_cnt
    total_count = sparse_block_count * subtile_factor

    return (
        curr_q_cnt,
        curr_q_idx,
        curr_full_cnt,
        curr_full_idx,
        partial_base,
        full_base,
        total_count,
    )


@cute.jit
def get_curr_dq_write_order_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    n_blocks_per_sample: int = 0,
):
    if const_expr(blocksparse_tensors.dq_write_order is None):
        return None, None
    assert blocksparse_tensors.dq_write_order is not None
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    mask_block_offset = blocksparse_tensors.mask_block_offset
    assert mask_block_offset is not None
    plan_head = Int32(0) if mask_block_cnt.shape[0] == 1 else head_idx
    outer_row = batch_idx * n_blocks_per_sample + n_block
    cu_total_k_blocks = blocksparse_tensors.cu_total_m_blocks
    if const_expr(cu_total_k_blocks is not None):
        outer_row = cu_total_k_blocks[batch_idx] + n_block
    offset_idx = plan_head * mask_block_cnt.shape[1] + outer_row
    curr_dq_write_order = cute.domain_offset(mask_block_offset[offset_idx], blocksparse_tensors.dq_write_order)
    curr_dq_write_order_full = None
    if const_expr(blocksparse_tensors.dq_write_order_full is not None):
        assert blocksparse_tensors.dq_write_order_full is not None
        full_block_offset = blocksparse_tensors.full_block_offset
        assert full_block_offset is not None
        curr_dq_write_order_full = cute.domain_offset(full_block_offset[offset_idx], blocksparse_tensors.dq_write_order_full)
    return curr_dq_write_order, curr_dq_write_order_full


@cute.jit
def get_m_block_from_iter_bwd(
    iter_idx,
    curr_q_cnt,
    curr_q_idx: cute.Tensor,
    curr_full_cnt,
    curr_full_idx: Optional[cute.Tensor],
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    full_first: cutlass.Constexpr[bool] = False,
):
    """Derive m_block index and is_full_block flag from iteration index.

    Returns (m_block, is_full_block):
        - m_block: The actual Q-tile block index
        - is_full_block: True if this is a full block (no packed mask needed)
    """
    sparse_iter_idx = iter_idx // subtile_factor
    subtile_offset = iter_idx % subtile_factor

    sparse_m_block = Int32(0)
    is_full_block = False
    if const_expr(curr_full_idx is not None):
        if const_expr(full_first):
            if sparse_iter_idx < curr_full_cnt:
                sparse_m_block = curr_full_idx[sparse_iter_idx]
                is_full_block = True
            else:
                sparse_m_block = curr_q_idx[sparse_iter_idx - curr_full_cnt]
        else:
            if sparse_iter_idx < curr_q_cnt:
                sparse_m_block = curr_q_idx[sparse_iter_idx]
            else:
                sparse_m_block = curr_full_idx[sparse_iter_idx - curr_q_cnt]
                is_full_block = True
    else:
        sparse_m_block = curr_q_idx[sparse_iter_idx]

    return sparse_m_block * subtile_factor + subtile_offset, is_full_block


@cute.jit
def get_physical_subtile_count_bwd_sm90(
    sparse_block_count,
    sparse_block_indices: cute.Tensor,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
):
    """Return the exact physical-Q iteration count for one sorted K2Q list."""
    if const_expr(subtile_factor == 1):
        return sparse_block_count
    physical_count = sparse_block_count * subtile_factor
    if sparse_block_count > Int32(0):
        last_sparse_m_block = sparse_block_indices[sparse_block_count - Int32(1)]
        tail_excess = (last_sparse_m_block + Int32(1)) * subtile_factor - m_block_max
        if tail_excess > Int32(0):
            physical_count -= tail_excess
    return physical_count


@cute.jit
def _load_q_do_block_sm90(
    m_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    sQ_block_metadata: Optional[cute.Tensor],
    producer_tidx: Int32,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    load_kv: bool,
):
    """Load one Q/dO block, optionally loading K/V on first iteration."""
    if load_kv:
        pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=tma_copy_bytes_K)
    else:
        pipeline_Q.producer_acquire(producer_state_Q)
    if const_expr(sQ_block_metadata is not None):
        if producer_tidx == Int32(0):
            stage = producer_state_Q.index
            sQ_block_metadata[0, stage] = m_block
            cute.arch.fence_view_async_shared()
    if load_kv:
        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
    load_Q(m_block, producer_state=producer_state_Q)
    load_LSE(m_block, producer_state=producer_state_Q)

    producer_state_dO_cur = producer_state_dO if const_expr(not Q_stage_eq_dO_stage) else producer_state_Q
    if load_kv:
        pipeline_dO.producer_acquire(producer_state_dO_cur, extra_tx_count=tma_copy_bytes_V)
        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
    else:
        pipeline_dO.producer_acquire(producer_state_dO_cur)
    load_dO(m_block, producer_state=producer_state_dO_cur)
    load_dPsum(m_block, producer_state=producer_state_dO_cur)

    producer_state_Q.advance()
    producer_state_dO.advance()
    return producer_state_Q, producer_state_dO


@cute.jit
def produce_block_sparse_q_loads_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    sQ_block_metadata: Optional[cute.Tensor],
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
    n_blocks_per_sample: int,
    producer_tidx: Int32,
):
    """SM90 backward block sparse loading with separate partial/full loops.

    K/V are loaded with the first valid block. Iterates partial blocks first,
    then full blocks, matching consumer order.

    Returns updated (producer_state_Q, producer_state_dO).
    """
    assert blocksparse_tensors.mask_block_masks is not None
    curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, _, _ = get_curr_arbitrary_blocksparse_tensors_bwd(
        batch_idx,
        head_idx,
        n_block,
        blocksparse_tensors,
        n_blocks_per_sample,
    )

    kv_loaded = False

    for sparse_idx in cutlass.range(curr_q_cnt, unroll=1):
        sparse_m_block = curr_q_idx[sparse_idx] * subtile_factor
        for subtile_offset in cutlass.range(subtile_factor, unroll=1):
            m_block = sparse_m_block + subtile_offset

            if m_block < m_block_max:
                producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                    m_block,
                    producer_state_Q,
                    producer_state_dO,
                    pipeline_Q,
                    pipeline_dO,
                    load_K,
                    load_V,
                    load_Q,
                    load_dO,
                    load_LSE,
                    load_dPsum,
                    sQ_block_metadata,
                    producer_tidx,
                    tma_copy_bytes_K,
                    tma_copy_bytes_V,
                    Q_stage_eq_dO_stage,
                    load_kv=not kv_loaded,
                )
                kv_loaded = True

    if const_expr(curr_full_idx is not None):
        for sparse_idx in cutlass.range(curr_full_cnt, unroll=1):
            sparse_m_block = curr_full_idx[sparse_idx] * subtile_factor
            for subtile_offset in cutlass.range(subtile_factor, unroll=1):
                m_block = sparse_m_block + subtile_offset

                if m_block < m_block_max:
                    producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                        m_block,
                        producer_state_Q,
                        producer_state_dO,
                        pipeline_Q,
                        pipeline_dO,
                        load_K,
                        load_V,
                        load_Q,
                        load_dO,
                        load_LSE,
                        load_dPsum,
                        sQ_block_metadata,
                        producer_tidx,
                        tma_copy_bytes_K,
                        tma_copy_bytes_V,
                        Q_stage_eq_dO_stage,
                        load_kv=not kv_loaded,
                    )
                    kv_loaded = True

    return producer_state_Q, producer_state_dO


@cute.jit
def consume_block_sparse_mma_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    consumer_state_Q,
    consumer_state_dO,
    mma_one_m_block_fn,
    consumer_tidx: Int32,
    sQ_block_metadata: Optional[cute.Tensor],
    pipeline_Q,
    n_blocks_per_sample: int,
    payload_words: cutlass.Constexpr[int] = 4,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Consume partial/full arbitrary-plan rows in SM90 backward."""
    mask_payloads = blocksparse_tensors.mask_block_masks
    assert mask_payloads is not None
    mask_payload_base = Int32(0)
    if const_expr(subtile_factor == 1 and sQ_block_metadata is not None):
        curr_q_cnt, curr_full_cnt, mask_payload_base = get_curr_arbitrary_block_counts_bwd_sm90(
            batch_idx,
            head_idx,
            n_block,
            blocksparse_tensors,
            n_blocks_per_sample,
        )
        curr_q_idx = None
        curr_full_idx = None
    else:
        (
            curr_q_cnt,
            curr_q_idx,
            curr_full_cnt,
            curr_full_idx,
            mask_payload_base,
            _,
        ) = get_curr_arbitrary_blocksparse_tensors_bwd(
            batch_idx,
            head_idx,
            n_block,
            blocksparse_tensors,
            n_blocks_per_sample,
        )

    dKV_accumulate = False
    if const_expr(sQ_block_metadata is not None):
        partial_iter_count = get_physical_subtile_count_bwd_sm90(curr_q_cnt, curr_q_idx, subtile_factor, m_block_max)
        full_iter_count = get_physical_subtile_count_bwd_sm90(curr_full_cnt, curr_full_idx, subtile_factor, m_block_max)
        for iter_idx in cutlass.range(partial_iter_count, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            packed_mask_fn = partial(
                load_mask_payload,
                mask_payloads,
                mask_payload_base + sparse_idx,
                consumer_tidx,
                subtile_idx=subtile_offset,
                payload_words=payload_words,
            )
            pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
            stage = consumer_state_Q.index
            m_block = sQ_block_metadata[0, stage]
            consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                m_block,
                consumer_state_Q,
                consumer_state_dO,
                packed_mask_fn=packed_mask_fn,
                dKV_accumulate=dKV_accumulate,
                q_pipeline_already_waited=True,
            )
            dKV_accumulate = True

        for _ in cutlass.range(full_iter_count, unroll=1):
            pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
            stage = consumer_state_Q.index
            m_block = sQ_block_metadata[0, stage]
            consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                m_block,
                consumer_state_Q,
                consumer_state_dO,
                mask_fn=None,
                dKV_accumulate=dKV_accumulate,
                q_pipeline_already_waited=True,
            )
            dKV_accumulate = True
        return consumer_state_Q, consumer_state_dO, dKV_accumulate
    else:
        for sparse_idx in cutlass.range(curr_q_cnt, unroll=1):
            sparse_m_block = curr_q_idx[sparse_idx] * subtile_factor
            for subtile_offset in cutlass.range(subtile_factor, unroll=1):
                m_block = sparse_m_block + subtile_offset
                if m_block < m_block_max:
                    packed_mask_fn = partial(
                        load_mask_payload,
                        mask_payloads,
                        mask_payload_base + sparse_idx,
                        consumer_tidx,
                        subtile_idx=subtile_offset,
                        payload_words=payload_words,
                    )
                    consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                        m_block,
                        consumer_state_Q,
                        consumer_state_dO,
                        packed_mask_fn=packed_mask_fn,
                        dKV_accumulate=dKV_accumulate,
                    )
                    dKV_accumulate = True

        for sparse_idx in cutlass.range(curr_full_cnt, unroll=1):
            sparse_m_block = curr_full_idx[sparse_idx] * subtile_factor
            for subtile_offset in cutlass.range(subtile_factor, unroll=1):
                m_block = sparse_m_block + subtile_offset
                if m_block < m_block_max:
                    consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                        m_block,
                        consumer_state_Q,
                        consumer_state_dO,
                        mask_fn=None,
                        dKV_accumulate=dKV_accumulate,
                    )
                    dKV_accumulate = True

    return consumer_state_Q, consumer_state_dO, dKV_accumulate


@cute.jit
def _store_one_dQaccum_sm90(
    m_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
    accum_row_major: cutlass.Constexpr[bool] = False,
    deterministic: cutlass.Constexpr[bool] = False,
    mdQ_semaphore_cur: Optional[cute.Tensor] = None,
    warp_local_tidx: Int32 = Int32(0),
    lock_value: Int32 = Int32(0),
):
    """Store dQaccum for a single m_block."""
    if const_expr(accum_row_major and not deterministic):
        # A row-major copy chunk contains columns produced by every warp
        # group. Do not release any writer for the next iteration until all
        # previous chunks have finished reading shared memory.
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
        if const_expr(not deterministic and not accum_row_major):
            cute.arch.cp_async_bulk_wait_group(num_dQ_warp_groups - 1 - warp_group_idx, read=True)
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
            number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )

    if const_expr(deterministic):
        assert mdQ_semaphore_cur is not None
        barrier.wait_eq(
            mdQ_semaphore_cur[(m_block, None)].iterator,
            warp_local_tidx,
            0,  # flag_offset
            lock_value,
        )

    if const_expr(accum_row_major):
        # A contiguous store chunk spans rows and therefore contains columns
        # produced by every dQ warp group. Wait for all writers before reading
        # any chunk from the row-major shared-memory matrix.
        for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
        for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
            with cute.arch.elect_one():
                copy_utils.cpasync_reduce_bulk_add_f32(
                    sdQaccum[None, warp_group_idx].iterator,
                    gdQaccum[(None, warp_group_idx), m_block].iterator,
                    tma_copy_bytes_dQ,
                )
            cute.arch.cp_async_bulk_commit_group()
    else:
        for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            with cute.arch.elect_one():
                copy_utils.cpasync_reduce_bulk_add_f32(
                    sdQaccum[None, warp_group_idx].iterator,
                    gdQaccum[(None, warp_group_idx), m_block].iterator,
                    tma_copy_bytes_dQ,
                )
            cute.arch.cp_async_bulk_commit_group()

    if const_expr(deterministic):
        assert mdQ_semaphore_cur is not None
        # The next contributor must not start until every dQ chunk from this
        # CTA has completed its global-memory reduction.
        cute.arch.cp_async_bulk_wait_group(0, read=False)
        barrier.arrive_inc(
            mdQ_semaphore_cur[(m_block, None)].iterator,
            warp_local_tidx,
            0,  # flag_offset
            1,
        )


@cute.jit
def dQaccum_store_block_sparse_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
    n_blocks_per_sample: int,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
    deterministic: cutlass.Constexpr[bool] = False,
    accum_row_major: cutlass.Constexpr[bool] = False,
    mdQ_semaphore_cur: Optional[cute.Tensor] = None,
    warp_local_tidx: Int32 = Int32(0),
):
    """SM90 backward block sparse dQaccum store with separate partial/full loops.

    Iterates partial blocks first, then full blocks, matching producer/consumer order.
    """
    assert blocksparse_tensors.mask_block_masks is not None
    (
        curr_q_cnt,
        curr_q_idx,
        curr_full_cnt,
        curr_full_idx,
        partial_base,
        full_base,
    ) = get_curr_arbitrary_blocksparse_tensors_bwd(
        batch_idx,
        head_idx,
        n_block,
        blocksparse_tensors,
        n_blocks_per_sample,
    )
    curr_dq_write_order = None
    curr_dq_write_order_full = None
    if const_expr(deterministic):
        assert blocksparse_tensors.dq_write_order is not None
        assert blocksparse_tensors.dq_write_order_full is not None
        curr_dq_write_order = cute.domain_offset(partial_base, blocksparse_tensors.dq_write_order)
        curr_dq_write_order_full = cute.domain_offset(full_base, blocksparse_tensors.dq_write_order_full)
        assert curr_dq_write_order is not None

    for sparse_idx in cutlass.range(curr_q_cnt, unroll=1):
        sparse_m_block = curr_q_idx[sparse_idx] * subtile_factor
        lock_value = Int32(0)
        if const_expr(deterministic):
            assert curr_dq_write_order is not None
            lock_value = curr_dq_write_order[sparse_idx]
        for subtile_offset in cutlass.range(subtile_factor, unroll=1):
            m_block = sparse_m_block + subtile_offset

            if m_block < m_block_max:
                _store_one_dQaccum_sm90(
                    m_block,
                    sdQaccum,
                    gdQaccum,
                    num_dQ_warp_groups,
                    num_threads_per_warp_group,
                    tma_copy_bytes_dQ,
                    accum_row_major=accum_row_major,
                    deterministic=deterministic,
                    mdQ_semaphore_cur=mdQ_semaphore_cur,
                    warp_local_tidx=warp_local_tidx,
                    lock_value=lock_value,
                )

    if const_expr(curr_full_idx is not None):
        if const_expr(deterministic):
            assert curr_dq_write_order_full is not None
        for sparse_idx in cutlass.range(curr_full_cnt, unroll=1):
            sparse_m_block = curr_full_idx[sparse_idx] * subtile_factor
            lock_value = Int32(0)
            if const_expr(deterministic):
                assert curr_dq_write_order_full is not None
                lock_value = curr_dq_write_order_full[sparse_idx]
            for subtile_offset in cutlass.range(subtile_factor, unroll=1):
                m_block = sparse_m_block + subtile_offset

                if m_block < m_block_max:
                    _store_one_dQaccum_sm90(
                        m_block,
                        sdQaccum,
                        gdQaccum,
                        num_dQ_warp_groups,
                        num_threads_per_warp_group,
                        tma_copy_bytes_dQ,
                        accum_row_major=accum_row_major,
                        deterministic=deterministic,
                        mdQ_semaphore_cur=mdQ_semaphore_cur,
                        warp_local_tidx=warp_local_tidx,
                        lock_value=lock_value,
                    )
