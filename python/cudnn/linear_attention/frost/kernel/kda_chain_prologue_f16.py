# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunked Kimi Delta Attention (KDA) piece-chain prologue for SM100 / SM103 / SM107 (Cutlass primitives): the one launch that
builds every table the chain's kernels read, so each consumer keeps its body and skips its own prologue.

Phases (two blocks: both build the piece table, block 0 the work-item tables, block 1 the descriptor arrays):
  piece table        : the flat piece slots of every sequence from cu_seqlens
  work-item tables   : the LPT order of the main and summary work-item tables, the scheduler rings, the
                       checkpoint-seeded series items where the backward reads a coarse series
  descriptor arrays  : the per-piece TMA descriptor arrays of every kernel the chain launches

Warp assignments (descriptor phase, one warp per array; a launch is forward or backward, never both):
  warps 0-2     : fused summary
  warps 0-3     : recompute H
  warps 4-7     : recompute M (V read from k)
  warps 8-11    : series recompute
  warps 12-17   : prefill, or warps 12-21 bprop
  warps 22-25   : bprop summary
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda.tensor_map as tma
import cutlass.experimental.primitives as nvvm

from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids

from ..common.piece_chain import piece_table_body
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, expanded_cu_seqlen, gen_interval_items, order_body
from . import kda_bprop_f16, kda_bprop_summary_f16, kda_prefill_f16, kda_recompute_f16, kda_summary_f16

USE_PDL = True


@cute.kernel
def frost_kda_chain_prologue(
    pieces: cutlass.Int32,
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    num_ctas: cutlass.Constexpr[int],
    heads_out: cutlass.Int32,
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_gate: cutlass.GridConstant[tma.TensorMap],
    base_o: cutlass.GridConstant[tma.TensorMap],
    base_do: cutlass.GridConstant[tma.TensorMap],
    base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_dq: cutlass.GridConstant[tma.TensorMap],
    base_dk: cutlass.GridConstant[tma.TensorMap],
    base_dv: cutlass.GridConstant[tma.TensorMap],
    base_dgate: cutlass.GridConstant[tma.TensorMap],
    num_seqs: cutlass.Int32,
    series_span_chunks: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: Optional[cute.Tensor],
    scheduler: cute.Tensor,
    series_items: Optional[cute.Tensor],
    series_count: Optional[cute.Tensor],
    summary_words: Optional[cute.Tensor],
    recompute_h_words: Optional[cute.Tensor],
    recompute_m_words: Optional[cute.Tensor],
    series_words: Optional[cute.Tensor],
    prefill_words: Optional[cute.Tensor],
    bprop_summary_words: Optional[cute.Tensor],
    bprop_words: Optional[cute.Tensor],
    q: Optional[cute.Tensor],
    k: cute.Tensor,
    v: Optional[cute.Tensor],
    gate: cute.Tensor,
    o: Optional[cute.Tensor],
    do_: Optional[cute.Tensor],
    checkpoints: Optional[cute.Tensor],
    dq: Optional[cute.Tensor],
    dk: Optional[cute.Tensor],
    dv: Optional[cute.Tensor],
    dgate: Optional[cute.Tensor],
    checkpoint_after_block: cutlass.Constexpr[bool] = False,
    tail: cutlass.Constexpr[bool] = False,
    first_head_lo: cutlass.Int32 = 0,
    cu_full: Optional[cute.Tensor] = None,
    batch_base: cutlass.Int32 = 0,
    seed_every_n: cutlass.Int32 = 0,
    seed_row_base: Optional[cute.Tensor] = None,
    tail_enable: Optional[cute.Tensor] = None,
) -> None:
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    n_heads_out = cutlass.Int32(heads_out)
    head_floor = None
    if cutlass.const_expr(tail):
        head_floor = first_head_lo

    # ---- piece table -----------------------------------------------------------------
    piece_table_body(
        ORDER_THREADS,
        pieces,
        unit_chunks,
        b_t,
        1,
        length_rule,
        heads_out,
        tidx,
        num_seqs,
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
    )
    if tidx == cutlass.Int32(0):
        main_count[0] = cutlass.Int32(main_rows[num_seqs]) - first_head_lo * (cutlass.Int32(main_rows[1]) // n_heads_out)
        summary_count[0] = cutlass.Int32(summary_rows[num_seqs]) - first_head_lo * (cutlass.Int32(summary_rows[1]) // n_heads_out)
    nvvm.barrier_cta_sync()
    n_pieces = num_seqs * cutlass.Int32(pieces)
    if cutlass.const_expr(tail_enable is not None):
        n_pieces = n_pieces if cutlass.Int32(tail_enable[0]) != cutlass.Int32(0) else cutlass.Int32(0)
        if tidx == cutlass.Int32(0):
            if cutlass.Int32(tail_enable[0]) == cutlass.Int32(0):
                main_count[0] = cutlass.Int32(0)
                summary_count[0] = cutlass.Int32(0)
        nvvm.barrier_cta_sync()
    n_items = n_heads_out * n_pieces
    if cutlass.const_expr(tail and seed_row_base is not None):
        if bidx == cutlass.Int32(1) and tidx == cutlass.Int32(0):
            seed_rows = cutlass.Int32(0)
            seed_b = cutlass.Int32(0)
            while seed_b < batch_base:
                seed_len = expanded_cu_seqlen(1, cu_full, seed_b + 1) - expanded_cu_seqlen(1, cu_full, seed_b)
                seed_rows = seed_rows + (seed_len + seed_every_n - cutlass.Int32(1)) // seed_every_n
                seed_b = seed_b + cutlass.Int32(1)
            seed_row_base[0] = seed_rows

    if bidx == cutlass.Int32(0):
        # ---- work-item tables --------------------------------------------------------
        sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        order_body(
            False,
            b_t,
            ORDER_THREADS,
            ORDER_ELEMENTS,
            tidx,
            n_heads_out,
            n_items,
            cu_pieces,
            None,
            main_count,
            work_items,
            scheduler,
            sKey,
            sIdx,
            sSpread,
            num_ctas=num_ctas,
            chain=True,
            mRowBase=main_rows,
            first_head_lo=head_floor,
        )
        if cutlass.const_expr(work_items_summary is not None):
            nvvm.barrier_cta_sync()
            order_body(
                False,
                b_t,
                ORDER_THREADS,
                ORDER_ELEMENTS,
                tidx,
                n_heads_out,
                n_items,
                cu_pieces,
                None,
                summary_count,
                work_items_summary,
                None,
                sKey,
                sIdx,
                sSpread,
                num_ctas=num_ctas,
                chain=True,
                mRowBase=summary_rows,
                mSlotRows=main_rows,
                first_head_lo=head_floor,
            )
        if cutlass.const_expr(series_items is not None):
            gen_interval_items(
                b_t,
                ORDER_THREADS,
                tidx,
                n_heads_out,
                n_items,
                series_span_chunks,
                cu_pieces,
                series_count,
                series_items,
                None,
                first_head_lo=head_floor,
                first_head_slots=cutlass.Int32(main_rows[1]) // n_heads_out if cutlass.const_expr(tail) else None,
            )
    else:
        # ---- descriptor arrays -------------------------------------------------------
        checkpoint_row_base = None
        if cutlass.const_expr(tail and checkpoints is not None):
            checkpoint_row_base = cutlass.Int32(0)
            seq_b = cutlass.Int32(0)
            while seq_b < batch_base:
                seq_len = expanded_cu_seqlen(1, cu_full, seq_b + 1) - expanded_cu_seqlen(1, cu_full, seq_b)
                if cutlass.const_expr(checkpoint_after_block):
                    checkpoint_row_base = checkpoint_row_base + seq_len // checkpoint_every_n
                else:
                    checkpoint_row_base = checkpoint_row_base + (seq_len + checkpoint_every_n - cutlass.Int32(1)) // checkpoint_every_n
                seq_b = seq_b + cutlass.Int32(1)
        if cutlass.const_expr(summary_words is not None):
            kda_summary_f16.build_descs_body(widx, base_k, base_v, base_gate, summary_words, cu_pieces, k, v, gate, n_pieces)
        if cutlass.const_expr(recompute_h_words is not None):
            kda_recompute_f16.build_descs_body(
                widx, base_k, base_v, base_gate, base_k, recompute_h_words, cu_pieces, k, v, gate, None, n_pieces, cutlass.Int32(0)
            )
        if cutlass.const_expr(recompute_m_words is not None):
            kda_recompute_f16.build_descs_body(
                widx - cutlass.Int32(4), base_k, base_k, base_gate, base_k, recompute_m_words, cu_pieces, k, k, gate, None, n_pieces, cutlass.Int32(0)
            )
        if cutlass.const_expr(series_words is not None):
            kda_recompute_f16.build_descs_body(
                widx - cutlass.Int32(8),
                base_k,
                base_v,
                base_gate,
                base_checkpoint,
                series_words,
                cu_pieces,
                k,
                v,
                gate,
                checkpoints,
                n_pieces,
                checkpoint_every_n,
                checkpoint_row_base=checkpoint_row_base,
            )
        if cutlass.const_expr(prefill_words is not None):
            kda_prefill_f16.build_descs_body(
                widx - cutlass.Int32(12),
                base_q,
                base_k,
                base_v,
                base_gate,
                base_o,
                base_checkpoint,
                prefill_words,
                cu_pieces,
                q,
                k,
                v,
                gate,
                o,
                checkpoints,
                n_pieces,
                checkpoint_every_n,
                checkpoint_after_block=checkpoint_after_block,
                checkpoint_row_base=checkpoint_row_base,
            )
        if cutlass.const_expr(bprop_words is not None):
            kda_bprop_f16.build_descs_body(
                widx - cutlass.Int32(12),
                base_q,
                base_k,
                base_v,
                base_gate,
                base_do,
                base_dq,
                base_dk,
                base_dv,
                base_dgate,
                base_checkpoint,
                bprop_words,
                cu_pieces,
                q,
                k,
                v,
                gate,
                do_,
                dq,
                dk,
                dv,
                dgate,
                checkpoints,
                n_pieces,
                checkpoint_every_n,
                checkpoint_row_base=checkpoint_row_base,
            )
        if cutlass.const_expr(bprop_summary_words is not None):
            kda_bprop_summary_f16.build_descs_body(
                widx - cutlass.Int32(22), base_q, base_k, base_gate, base_do, bprop_summary_words, cu_pieces, q, k, gate, do_, n_pieces
            )


@cute.jit
def chain_prologue(
    pieces: cutlass.Int32,
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    num_ctas: cutlass.Constexpr[int],
    heads_out: cutlass.Int32,
    series_span_chunks: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: Optional[cute.Tensor],
    scheduler: cute.Tensor,
    series_items: Optional[cute.Tensor],
    series_count: Optional[cute.Tensor],
    summary_words: Optional[cute.Tensor],
    recompute_h_words: Optional[cute.Tensor],
    recompute_m_words: Optional[cute.Tensor],
    series_words: Optional[cute.Tensor],
    prefill_words: Optional[cute.Tensor],
    bprop_summary_words: Optional[cute.Tensor],
    bprop_words: Optional[cute.Tensor],
    q: Optional[cute.Tensor],
    k: cute.Tensor,
    v: Optional[cute.Tensor],
    gate: cute.Tensor,
    o: Optional[cute.Tensor],
    do_: Optional[cute.Tensor],
    checkpoints: Optional[cute.Tensor],
    dq: Optional[cute.Tensor],
    dk: Optional[cute.Tensor],
    dv: Optional[cute.Tensor],
    dgate: Optional[cute.Tensor],
    stream: cuda.CUstream,
    checkpoint_after_block: cutlass.Constexpr[bool] = False,
    checkpoint_box_elems: cutlass.Constexpr[int] = 0,
    tail: cutlass.Constexpr[bool] = False,
    first_head_lo=None,
    cu_full=None,
    batch_base=None,
    seed_every_n=None,
    seed_row_base=None,
    tail_enable=None,
) -> None:
    swizzle_128b = tma.TensorMapSwizzle.s128b
    head_floor = cutlass.Int32(0)
    base_seq = cutlass.Int32(0)
    seed_n = cutlass.Int32(0)
    if cutlass.const_expr(tail):
        head_floor = first_head_lo
        base_seq = batch_base
        if cutlass.const_expr(seed_every_n is not None):
            seed_n = seed_every_n
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((k.shape[2], k.shape[1], k.shape[0]), stride=(1, k.stride[1], k.stride[0])))
    base_k = tma.create_tensor_map_tiled_from_view(
        k_headed, box_dims=(128 // (k.element_type.width // 8), 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle_128b
    )
    gate_headed = cute.make_tensor(gate.iterator, cute.make_layout((gate.shape[2], gate.shape[1], gate.shape[0]), stride=(1, gate.stride[1], gate.stride[0])))
    base_gate = tma.create_tensor_map_tiled_from_view(
        gate_headed, box_dims=(128 // (gate.element_type.width // 8), 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle_128b
    )
    base_q = base_k
    base_v = base_k
    base_o = base_k
    base_do = base_k
    base_checkpoint = base_k
    base_dq = base_k
    base_dk = base_k
    base_dv = base_k
    base_dgate = base_k
    if cutlass.const_expr(q is not None):
        q_headed = cute.make_tensor(q.iterator, cute.make_layout((q.shape[2], q.shape[1], q.shape[0]), stride=(1, q.stride[1], q.stride[0])))
        base_q = tma.create_tensor_map_tiled_from_view(
            q_headed, box_dims=(128 // (q.element_type.width // 8), 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle_128b
        )
    if cutlass.const_expr(v is not None):
        v_headed = cute.make_tensor(v.iterator, cute.make_layout((v.shape[2], v.shape[1], v.shape[0]), stride=(1, v.stride[1], v.stride[0])))
        base_v = tma.create_tensor_map_tiled_from_view(
            v_headed, box_dims=(128 // (v.element_type.width // 8), 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle_128b
        )
    if cutlass.const_expr(o is not None):
        o_headed = cute.make_tensor(o.iterator, cute.make_layout((o.shape[2], o.shape[1], o.shape[0]), stride=(1, o.stride[1], o.stride[0])))
        base_o = tma.create_tensor_map_tiled_from_view(
            o_headed, box_dims=(128 // (o.element_type.width // 8), 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle_128b
        )
    if cutlass.const_expr(do_ is not None):
        do_headed = cute.make_tensor(do_.iterator, cute.make_layout((do_.shape[2], do_.shape[1], do_.shape[0]), stride=(1, do_.stride[1], do_.stride[0])))
        base_do = tma.create_tensor_map_tiled_from_view(
            do_headed, box_dims=(128 // (do_.element_type.width // 8), 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle_128b
        )
    if cutlass.const_expr(checkpoints is not None):
        checkpoint_view = cute.make_tensor(
            checkpoints.iterator,
            cute.make_layout(
                (checkpoints.shape[3], checkpoints.shape[2], checkpoints.shape[0], checkpoints.shape[1]),
                stride=(checkpoints.stride[3], checkpoints.stride[2], checkpoints.stride[0], checkpoints.stride[1]),
            ),
        )
        checkpoint_elem_bytes = checkpoints.element_type.width // 8
        checkpoint_box_width = checkpoint_box_elems
        if cutlass.const_expr(checkpoint_box_elems == 0):
            checkpoint_box_width = 128 // checkpoint_elem_bytes
        if cutlass.const_expr(checkpoint_box_width * checkpoint_elem_bytes == 64):
            checkpoint_swizzle_mode = tma.TensorMapSwizzle.s64b
        else:
            checkpoint_swizzle_mode = swizzle_128b
        base_checkpoint = tma.create_tensor_map_tiled_from_view(
            checkpoint_view,
            box_dims=(checkpoint_box_width, checkpoints.shape[2], 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=checkpoint_swizzle_mode,
        )
    if cutlass.const_expr(dq is not None):
        dq_headed = cute.make_tensor(dq.iterator, cute.make_layout((dq.shape[2], dq.shape[1], dq.shape[0]), stride=(1, dq.stride[1], dq.stride[0])))
        base_dq = tma.create_tensor_map_tiled_from_view(
            dq_headed, box_dims=(128 // (dq.element_type.width // 8), 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle_128b
        )
    if cutlass.const_expr(dk is not None):
        dk_headed = cute.make_tensor(dk.iterator, cute.make_layout((dk.shape[2], dk.shape[1], dk.shape[0]), stride=(1, dk.stride[1], dk.stride[0])))
        base_dk = tma.create_tensor_map_tiled_from_view(
            dk_headed, box_dims=(128 // (dk.element_type.width // 8), 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle_128b
        )
    if cutlass.const_expr(dv is not None):
        dv_headed = cute.make_tensor(dv.iterator, cute.make_layout((dv.shape[2], dv.shape[1], dv.shape[0]), stride=(1, dv.stride[1], dv.stride[0])))
        base_dv = tma.create_tensor_map_tiled_from_view(
            dv_headed, box_dims=(128 // (dv.element_type.width // 8), 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle_128b
        )
    if cutlass.const_expr(dgate is not None):
        dgate_headed = cute.make_tensor(
            dgate.iterator, cute.make_layout((dgate.shape[2], dgate.shape[1], dgate.shape[0]), stride=(1, dgate.stride[1], dgate.stride[0]))
        )
        base_dgate = tma.create_tensor_map_tiled_from_view(
            dgate_headed, box_dims=(128 // (dgate.element_type.width // 8), 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle_128b
        )
    frost_kda_chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        num_ctas,
        heads_out,
        base_q,
        base_k,
        base_v,
        base_gate,
        base_o,
        base_do,
        base_checkpoint,
        base_dq,
        base_dk,
        base_dv,
        base_dgate,
        cutlass.Int32(cu_seqlens.shape[0] - 1),
        series_span_chunks,
        checkpoint_every_n,
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        scheduler,
        series_items,
        series_count,
        summary_words,
        recompute_h_words,
        recompute_m_words,
        series_words,
        prefill_words,
        bprop_summary_words,
        bprop_words,
        q,
        k,
        v,
        gate,
        o,
        do_,
        checkpoints,
        dq,
        dk,
        dv,
        dgate,
        checkpoint_after_block,
        tail,
        head_floor,
        cu_full,
        base_seq,
        seed_n,
        seed_row_base,
        tail_enable,
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)
