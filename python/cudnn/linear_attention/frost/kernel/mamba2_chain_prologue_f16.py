# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mamba-2 piece-chain prologue for SM100 / SM103 / SM107 (Cutlass primitives): the one launch that builds every table the
chain's kernels read, so each consumer keeps its body and skips its own prologue.

Phases (two blocks: both build the piece table, block 0 the work-item tables, block 1 the descriptor arrays):
  piece table        : the flat piece slots of every sequence from cu_seqlens
  work-item tables   : the LPT order of the main and summary work-item tables and the scheduler rings
  descriptor arrays  : the per-piece TMA descriptor arrays of the prefill (its K, V, checkpoint arrays read by the
                       recompute kernel's summary and regenerate passes, all seven by the main forward pass) and of the
                       bprop (its summary and main passes)

Warp assignments (descriptor phase, one warp per array; a launch builds the prefill's, the bprop's or both):
  warps 0-6     : prefill (Q, K, V, O, checkpoints, z, pre-gate O)
  warps 7-14    : bprop (Q, K, V, dO, checkpoints, dQ, dK, dV)
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda.tensor_map as tma
import cutlass.experimental.primitives as nvvm

from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids

from ..common.piece_chain import piece_table_body
from ..common.gate_bwd import exit_slot_table_body
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, order_body
from . import mamba2_bprop_f16, mamba2_prefill_f16

USE_PDL = True


@cute.kernel
def frost_mamba2_chain_prologue(
    pieces: cutlass.Int32,
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    num_ctas: cutlass.Constexpr[int],
    heads_out: cutlass.Int32,
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_o: cutlass.GridConstant[tma.TensorMap],
    base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_z: cutlass.GridConstant[tma.TensorMap],
    base_pre: cutlass.GridConstant[tma.TensorMap],
    base_do: cutlass.GridConstant[tma.TensorMap],
    base_dq: cutlass.GridConstant[tma.TensorMap],
    base_dk: cutlass.GridConstant[tma.TensorMap],
    base_dv: cutlass.GridConstant[tma.TensorMap],
    num_seqs: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor,
    scheduler: cute.Tensor,
    desc_workspace: Optional[cute.Tensor],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: Optional[cute.Tensor],
    checkpoints: Optional[cute.Tensor],
    z: Optional[cute.Tensor],
    o_pre: Optional[cute.Tensor],
    bprop_words: Optional[cute.Tensor],
    do_: Optional[cute.Tensor],
    dq: Optional[cute.Tensor],
    dk: Optional[cute.Tensor],
    dv: Optional[cute.Tensor],
    exit_slots: Optional[cute.Tensor],
) -> None:
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    n_heads_out = cutlass.Int32(heads_out)

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
    nvvm.barrier_cta_sync()
    n_pieces = num_seqs * cutlass.Int32(pieces)
    if cutlass.const_expr(exit_slots is not None):
        # ---- the gate backward's exit-slot table over the piece boundaries, by the descriptor CTA -------------
        if bidx == cutlass.Int32(1):
            exit_slot_table_body(b_t, tidx, ORDER_THREADS, n_pieces, cu_pieces, exit_slots)

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
            n_heads_out * n_pieces,
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
        )
        nvvm.barrier_cta_sync()
        order_body(
            False,
            b_t,
            ORDER_THREADS,
            ORDER_ELEMENTS,
            tidx,
            n_heads_out,
            n_heads_out * n_pieces,
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
        )
    else:
        # ---- descriptor arrays -------------------------------------------------------
        if cutlass.const_expr(desc_workspace is not None):
            mamba2_prefill_f16.build_descs_body(
                widx,
                base_q,
                base_k,
                base_v,
                base_o,
                base_checkpoint,
                base_z,
                base_pre,
                desc_workspace,
                cu_pieces,
                q,
                k,
                v,
                o,
                checkpoints,
                z,
                o_pre,
                n_pieces,
                checkpoint_every_n,
                b_t,
            )
        if cutlass.const_expr(bprop_words is not None):
            mamba2_bprop_f16.build_descs_body(
                widx - cutlass.Int32(mamba2_prefill_f16.TENSORMAP_DESC_ARRAYS),
                base_q,
                base_k,
                base_v,
                base_do,
                base_checkpoint,
                base_dq,
                base_dk,
                base_dv,
                bprop_words,
                cu_pieces,
                q,
                k,
                v,
                do_,
                checkpoints,
                dq,
                dk,
                dv,
                n_pieces,
                b_t,
            )


@cute.jit
def chain_prologue(
    pieces: cutlass.Int32,
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    num_ctas: cutlass.Constexpr[int],
    heads_out: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor,
    scheduler: cute.Tensor,
    desc_workspace: Optional[cute.Tensor],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: Optional[cute.Tensor],
    checkpoints: Optional[cute.Tensor],
    z: Optional[cute.Tensor],
    o_pre: Optional[cute.Tensor],
    bprop_words: Optional[cute.Tensor],
    do_: Optional[cute.Tensor],
    dq: Optional[cute.Tensor],
    dk: Optional[cute.Tensor],
    dv: Optional[cute.Tensor],
    exit_slots: Optional[cute.Tensor],
    stream: cuda.CUstream,
) -> None:
    """One-launch chain prologue: the piece table, the LPT-ordered work-item tables, the per-piece TMA-descriptor
    arrays of the prefill and the bprop, and in the backward the gate's exit-slot table."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    heads_out_q = h_q if h_q >= h_v else h_v
    d_v = v.shape[2]
    bytes_per_element = q.element_type.width // 8
    box_elements = 128 // bytes_per_element
    bt = b_t
    seqlen = q.shape[0]
    d_k = q.shape[2]
    q_headed = cute.make_tensor(q.iterator, cute.make_layout((seqlen, h_q, d_k), stride=(q.stride[0], q.stride[1], 1)))
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((seqlen, h_k, d_k), stride=(k.stride[0], k.stride[1], 1)))
    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v.stride[1], v.stride[0])))
    swizzle_128b = tma.TensorMapSwizzle.s128b
    base_desc_q = tma.create_tensor_map_tiled_from_view(q_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    base_desc_k = tma.create_tensor_map_tiled_from_view(k_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    base_desc_v = tma.create_tensor_map_tiled_from_view(v_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)
    base_desc_o = base_desc_v
    if cutlass.const_expr(o is not None):
        o_headed = cute.make_tensor(o.iterator, cute.make_layout((d_v, heads_out_q, o.shape[0]), stride=(1, o.stride[1], o.stride[0])))
        base_desc_o = tma.create_tensor_map_tiled_from_view(o_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)
    base_desc_checkpoint = base_desc_v
    if cutlass.const_expr(checkpoints is not None):
        d_v_state = checkpoints.shape[2]
        d_k_state = checkpoints.shape[3]
        checkpoint_box_elements = 128 // (checkpoints.element_type.width // 8)
        checkpoint_view = cute.make_tensor(
            checkpoints.iterator,
            cute.make_layout(
                (d_k_state, d_v_state, checkpoints.shape[0], heads_out_q),
                stride=(checkpoints.stride[3], checkpoints.stride[2], checkpoints.stride[0], checkpoints.stride[1]),
            ),
        )
        base_desc_checkpoint = tma.create_tensor_map_tiled_from_view(
            checkpoint_view, box_dims=(checkpoint_box_elements, d_v_state, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swizzle_128b
        )
    base_desc_z = base_desc_v
    if cutlass.const_expr(z is not None):
        z_headed = cute.make_tensor(z.iterator, cute.make_layout((d_v, heads_out_q, z.shape[0]), stride=(1, z.stride[1], z.stride[0])))
        base_desc_z = tma.create_tensor_map_tiled_from_view(z_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)
    base_desc_pre = base_desc_o
    if cutlass.const_expr(o_pre is not None):
        pre_headed = cute.make_tensor(o_pre.iterator, cute.make_layout((d_v, heads_out_q, o_pre.shape[0]), stride=(1, o_pre.stride[1], o_pre.stride[0])))
        base_desc_pre = tma.create_tensor_map_tiled_from_view(pre_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)
    base_desc_do = base_desc_v
    base_desc_dq = base_desc_q
    base_desc_dk = base_desc_k
    base_desc_dv = base_desc_v
    if cutlass.const_expr(bprop_words is not None):
        do_headed = cute.make_tensor(do_.iterator, cute.make_layout((d_v, heads_out_q, seqlen), stride=(1, do_.stride[1], do_.stride[0])))
        base_desc_do = tma.create_tensor_map_tiled_from_view(do_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)
        dq_headed = cute.make_tensor(dq.iterator, cute.make_layout((seqlen, dq.shape[1], d_k), stride=(dq.stride[0], dq.stride[1], 1)))
        base_desc_dq = tma.create_tensor_map_tiled_from_view(dq_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
        dk_headed = cute.make_tensor(dk.iterator, cute.make_layout((seqlen, dk.shape[1], d_k), stride=(dk.stride[0], dk.stride[1], 1)))
        base_desc_dk = tma.create_tensor_map_tiled_from_view(dk_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
        dv_headed = cute.make_tensor(dv.iterator, cute.make_layout((d_v, heads_out_q, seqlen), stride=(1, dv.stride[1], dv.stride[0])))
        base_desc_dv = tma.create_tensor_map_tiled_from_view(dv_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)
    frost_mamba2_chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        num_ctas,
        heads_out,
        base_desc_q,
        base_desc_k,
        base_desc_v,
        base_desc_o,
        base_desc_checkpoint,
        base_desc_z,
        base_desc_pre,
        base_desc_do,
        base_desc_dq,
        base_desc_dk,
        base_desc_dv,
        cutlass.Int32(cu_seqlens.shape[0] - 1),
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
        desc_workspace,
        q,
        k,
        v,
        o,
        checkpoints,
        z,
        o_pre,
        bprop_words,
        do_,
        dq,
        dk,
        dv,
        exit_slots,
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


frost_mamba2_chain_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
