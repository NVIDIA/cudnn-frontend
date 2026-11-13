# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One compiled launch for the GDN warmup and uncut backward: the T pass (with its own
descriptor prologue), the split-K table (warmup only), the recompute prologue and the checkpoint-series recompute (unless
the forward's per-chunk series is passed back), the bprop prologue and the bprop, all at ``--opt-level 2`` like their
standalone builds, so the call is one crossing into the DSL.  Every kernel, its
host and the tensor placeholder each host was compiled with are the standalone modules' own; a buffer two hosts read
through different placeholder types is passed twice (the table's 4-byte compact views of work_items, work_count and
item_scratch; the bprop prologue's 4-byte cu_seqlens).  The bprop module is a constexpr argument: GDP at d_v = 64 runs
the gdp_bprop_v64_f16 fork, whose prologue takes no T-pass tiles."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cutlass.experimental.cuda.tensor_map as tma
from cudnn.frost.buffers import DeviceView
from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids

from ..common import expand, gate_bwd, head_reduce, l2norm, split_k
from ..common.host import get_dtype, family_slots
from ..common.piece_chain import chain_rows_per_cta, dtype_name, launch_state_chain
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, gen_interval_items, order_body
from ..common.tail import gate_body
from . import gdn_bprop_f16, gdn_bprop_summary_f16, gdn_chain_backward_f16, gdn_recompute_f16, gdn_tinv_f16, gdp_bprop_v64_f16

USE_PDL = True
warmup_backward_cache = {}

TAIL_BUFFERS = (
    "cu_pieces",
    "main_rows",
    "summary_rows",
    "main_count",
    "summary_count",
    "work_items",
    "work_items_summary",
    "series_items",
    "series_count",
    "recompute_items",
    "recompute_count",
    "scheduler_all",
    "scheduler_recompute",
    "scheduler_m",
    "scheduler_series",
    "scheduler_summary",
    "scheduler_bwd",
    "tinv_words",
    "tinv_rows",
    "tinv_row_count",
    "summary_words",
    "recompute_m_words",
    "series_words",
    "bprop_summary_words",
    "bprop_words",
    "summary_q",
    "summary_do",
    "tinv",
    "state_h_summary",
    "state_m_main",
    "state_h_chain",
    "state_m_chain",
    "state_x_chain",
    "seed",
    "state_x_series",
    "state_g",
    "state_dx_end",
    "dseed",
    "seed_row_base",
)


@cute.kernel
def frost_gdn_regenerate_prologue(
    compact_qdo: cutlass.Constexpr[bool],
    recompute_orders: cutlass.Constexpr[bool],
    recompute_order_gen: cutlass.Constexpr[bool],
    coarse: cutlass.Constexpr[bool],
    bwd_orders: cutlass.Constexpr[bool],
    bwd_order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    recompute_base_k: cutlass.GridConstant[tma.TensorMap],
    recompute_base_v: cutlass.GridConstant[tma.TensorMap],
    recompute_base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    recompute_base_tinv: cutlass.GridConstant[tma.TensorMap],
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_do: cutlass.GridConstant[tma.TensorMap],
    base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_dq: cutlass.GridConstant[tma.TensorMap],
    base_dk: cutlass.GridConstant[tma.TensorMap],
    base_dv: cutlass.GridConstant[tma.TensorMap],
    base_tinv: cutlass.GridConstant[tma.TensorMap],
    recompute_workspace: cute.Tensor,
    bprop_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_seqlens_bprop: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do_: cute.Tensor,
    gate: cute.Tensor,
    checkpoints: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    tinv: Optional[cute.Tensor],
    staging_recompute: Optional[cute.Tensor],
    series_count: cute.Tensor,
    series_items: cute.Tensor,
    scheduler_recompute: Optional[cute.Tensor],
    staging_bprop: Optional[cute.Tensor],
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_bprop: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    seed_span_chunks: cutlass.Int32,
    k_slots: cutlass.Constexpr[int] = 1,
    v_slots: cutlass.Constexpr[int] = 1,
    mItemLimit: Optional[cute.Tensor] = None,
    gate_sms: cutlass.Constexpr[int] = 0,
) -> None:
    """Four-CTA prologue of the regenerate path, the recompute's and the bprop's prologues in one launch: block 0 the
    recompute's item phase (the interval items under ``coarse``, the LPT order under ``recompute_orders``), block 1 the
    bprop's (under ``bwd_orders``), block 2 the recompute's per-batch TMA-descriptor arrays, block 3 the bprop's, one
    warp per array."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    n_items = cutlass.Int32(gate.shape[1]) * n_batch
    if cutlass.const_expr(mItemLimit is not None):
        if cutlass.const_expr(gate_sms > 0):
            gate_body(ORDER_THREADS, tidx, cutlass.Int32(gate.shape[1]), gate_sms, cu_seqlens, mItemLimit)
        n_items = cutlass.Int32(mItemLimit[1])
    if bidx == cutlass.Int32(0):
        n_heads_out = cutlass.Int32(gate.shape[1])
        if cutlass.const_expr(coarse):
            gen_interval_items(
                b_t,
                ORDER_THREADS,
                tidx,
                n_heads_out,
                n_items,
                seed_span_chunks,
                cu_seqlens,
                series_count,
                series_items,
                scheduler_recompute,
                expand_num,
            )
        if cutlass.const_expr(recompute_orders):
            sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
            order_body(
                recompute_order_gen,
                b_t,
                ORDER_THREADS,
                ORDER_ELEMENTS,
                tidx,
                n_heads_out,
                n_items,
                cu_seqlens,
                staging_recompute,
                series_count,
                series_items,
                scheduler_recompute,
                sKey,
                sIdx,
                sSpread,
                expand_num=expand_num,
            )
    elif bidx == cutlass.Int32(1):
        if cutlass.const_expr(bwd_orders):
            sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
            n_heads_out = cutlass.Int32(do_.shape[1])
            order_body(
                bwd_order_gen,
                b_t,
                ORDER_THREADS,
                ORDER_ELEMENTS,
                tidx,
                n_heads_out,
                n_items,
                cu_seqlens_bprop,
                staging_bprop,
                work_count,
                work_items,
                scheduler_bprop,
                sKey,
                sIdx,
                sSpread,
                expand_num=expand_num,
            )
    elif bidx == cutlass.Int32(2):
        gdn_recompute_f16.build_descs_body(
            widx,
            recompute_base_k,
            recompute_base_v,
            recompute_base_checkpoint,
            recompute_base_tinv,
            recompute_workspace,
            cu_seqlens,
            k,
            v,
            checkpoints,
            tinv,
            n_batch,
            checkpoint_every_n,
            b_t,
            expand_num,
            k_slots=k_slots,
            v_slots=v_slots,
        )
    else:
        if cutlass.const_expr(compact_qdo):
            gdp_bprop_v64_f16.build_descs_body(
                widx,
                base_q,
                base_k,
                base_v,
                base_do,
                base_checkpoint,
                base_dq,
                base_dk,
                base_dv,
                bprop_workspace,
                cu_seqlens_bprop,
                q,
                k,
                v,
                do_,
                checkpoints,
                dq,
                dk,
                dv,
                n_batch,
                cutlass.Int32(b_t),
                expand_num,
                k_slots=k_slots,
                v_slots=v_slots,
            )
        else:
            gdn_bprop_f16.build_descs_body(
                widx,
                base_q,
                base_k,
                base_v,
                base_do,
                base_checkpoint,
                base_dq,
                base_dk,
                base_dv,
                base_tinv,
                bprop_workspace,
                cu_seqlens_bprop,
                q,
                k,
                v,
                do_,
                checkpoints,
                dq,
                dk,
                dv,
                tinv,
                n_batch,
                cutlass.Int32(b_t),
                b_t,
                expand_num,
                k_slots=k_slots,
                v_slots=v_slots,
            )


@cute.jit
def warmup_backward_host(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    overhead_chunks: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    bprop: cutlass.Constexpr,
    compact_qdo: cutlass.Constexpr[bool],
    tinv_pass: cutlass.Constexpr[bool],
    recompute: cutlass.Constexpr[bool],
    recompute_orders: cutlass.Constexpr[bool],
    recompute_order_gen: cutlass.Constexpr[bool],
    coarse: cutlass.Constexpr[bool],
    bwd_orders: cutlass.Constexpr[bool],
    bwd_order_gen: cutlass.Constexpr[bool],
    tinv_cfg: cutlass.Constexpr,
    recompute_cfg: cutlass.Constexpr,
    bprop_cfg: cutlass.Constexpr,
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    seed_span_chunks: cutlass.Int32,
    seed_every_n: cutlass.Int32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    beta: cute.Tensor,
    a_log: Optional[cute.Tensor],
    a_log_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    cu_seqlens_bprop: cute.Tensor,
    tinv: Optional[cute.Tensor],
    tinv_words: Optional[cute.Tensor],
    tinv_rows: Optional[cute.Tensor],
    tinv_row_count: Optional[cute.Tensor],
    checkpoints: cute.Tensor,
    seed_checkpoints: Optional[cute.Tensor],
    state_in: Optional[cute.Tensor],
    composed: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_items_table: cute.Tensor,
    work_count: cute.Tensor,
    work_count_table: cute.Tensor,
    series_items: Optional[cute.Tensor],
    series_count: Optional[cute.Tensor],
    staging_recompute: Optional[cute.Tensor],
    staging_bprop: Optional[cute.Tensor],
    item_scratch: Optional[cute.Tensor],
    chunk_scratch: Optional[cute.Tensor],
    scheduler_all: cute.Tensor,
    scheduler_all_recompute: Optional[cute.Tensor],
    scheduler_all_bprop: Optional[cute.Tensor],
    scheduler_recompute: cute.Tensor,
    recompute_words: Optional[cute.Tensor],
    bprop_words: cute.Tensor,
    dgate: cute.Tensor,
    dbeta: cute.Tensor,
    dstate0: Optional[cute.Tensor],
    dstate_in: Optional[cute.Tensor],
    dcomposed: Optional[cute.Tensor],
    inv_q: Optional[cute.Tensor],
    inv_k: Optional[cute.Tensor],
    scheduler_bwd: cute.Tensor,
    scale: cutlass.Float32,
    stream: cuda.CUstream,
    k_slots: cutlass.Constexpr[int] = 1,
    v_slots: cutlass.Constexpr[int] = 1,
    item_limit_word: Optional[cute.Tensor] = None,
    fold_rows: cutlass.Constexpr[int] = 0,
    fold_update: Optional[cute.Tensor] = None,
    fold_transition: Optional[cute.Tensor] = None,
    fold_seed: Optional[cute.Tensor] = None,
    fold_composed: Optional[cute.Tensor] = None,
    pack_q: Optional[cute.Tensor] = None,
    pack_q_x: Optional[cute.Tensor] = None,
    pack_do: Optional[cute.Tensor] = None,
    pack_do_x: Optional[cute.Tensor] = None,
    l2norm_k_slots: cutlass.Constexpr[int] = 1,
    l2norm_k_expand_num: cutlass.Constexpr[int] = 1,
    l2norm_q: Optional[cute.Tensor] = None,
    l2norm_k: Optional[cute.Tensor] = None,
    l2norm_inv_q: Optional[cute.Tensor] = None,
    l2norm_inv_k: Optional[cute.Tensor] = None,
    gate_blocks: cutlass.Int32 = 0,
    gate_bwd_dgate: Optional[cute.Tensor] = None,
    gate_bwd_gate: Optional[cute.Tensor] = None,
    gate_bwd_a_log: Optional[cute.Tensor] = None,
    gate_bwd_dt_bias: Optional[cute.Tensor] = None,
    gate_part_a: Optional[cute.Tensor] = None,
    gate_part_dt: Optional[cute.Tensor] = None,
    d_a_log: Optional[cute.Tensor] = None,
    d_dt_bias: Optional[cute.Tensor] = None,
    reduce_dq_src: Optional[cute.Tensor] = None,
    reduce_dq: Optional[cute.Tensor] = None,
    reduce_dk_src: Optional[cute.Tensor] = None,
    reduce_dk: Optional[cute.Tensor] = None,
    reduce_dv_src: Optional[cute.Tensor] = None,
    reduce_dv: Optional[cute.Tensor] = None,
    gather_dq_src: Optional[cute.Tensor] = None,
    gather_dq: Optional[cute.Tensor] = None,
    tail_unit_chunks: cutlass.Constexpr[int] = 0,
    tail_chain_rows: cutlass.Constexpr[int] = 0,
    tail_summary_q_step: cutlass.Constexpr[int] = 1,
    tail_tinv_cfg: cutlass.Constexpr = None,
    tail_summary_cfg: cutlass.Constexpr = None,
    tail_transition_cfg: cutlass.Constexpr = None,
    tail_series_cfg: cutlass.Constexpr = None,
    tail_bwd_summary_cfg: cutlass.Constexpr = None,
    tail_bprop_cfg: cutlass.Constexpr = None,
    tail_pieces: cutlass.Int32 = 0,
    tail_num_seqs: cutlass.Int32 = 0,
    tail_series_span_chunks: cutlass.Int32 = 0,
    tail_batch_base: cutlass.Int32 = 0,
    tail_first_head_lo: cutlass.Int32 = 0,
    tail_cu_pieces: Optional[cute.Tensor] = None,
    tail_main_rows: Optional[cute.Tensor] = None,
    tail_summary_rows: Optional[cute.Tensor] = None,
    tail_main_count: Optional[cute.Tensor] = None,
    tail_summary_count: Optional[cute.Tensor] = None,
    tail_work_items: Optional[cute.Tensor] = None,
    tail_work_items_summary: Optional[cute.Tensor] = None,
    tail_series_items: Optional[cute.Tensor] = None,
    tail_series_count: Optional[cute.Tensor] = None,
    tail_recompute_items: Optional[cute.Tensor] = None,
    tail_recompute_count: Optional[cute.Tensor] = None,
    tail_scheduler_all: Optional[cute.Tensor] = None,
    tail_scheduler_recompute: Optional[cute.Tensor] = None,
    tail_scheduler_m: Optional[cute.Tensor] = None,
    tail_scheduler_series: Optional[cute.Tensor] = None,
    tail_scheduler_summary: Optional[cute.Tensor] = None,
    tail_scheduler_bwd: Optional[cute.Tensor] = None,
    tail_tinv_words: Optional[cute.Tensor] = None,
    tail_tinv_rows: Optional[cute.Tensor] = None,
    tail_tinv_row_count: Optional[cute.Tensor] = None,
    tail_summary_words: Optional[cute.Tensor] = None,
    tail_recompute_m_words: Optional[cute.Tensor] = None,
    tail_series_words: Optional[cute.Tensor] = None,
    tail_bprop_summary_words: Optional[cute.Tensor] = None,
    tail_bprop_words: Optional[cute.Tensor] = None,
    tail_summary_q: Optional[cute.Tensor] = None,
    tail_summary_do: Optional[cute.Tensor] = None,
    tail_tinv: Optional[cute.Tensor] = None,
    tail_state_h_summary: Optional[cute.Tensor] = None,
    tail_state_m_main: Optional[cute.Tensor] = None,
    tail_state_h_chain: Optional[cute.Tensor] = None,
    tail_state_m_chain: Optional[cute.Tensor] = None,
    tail_state_x_chain: Optional[cute.Tensor] = None,
    tail_seed: Optional[cute.Tensor] = None,
    tail_state_x_series: Optional[cute.Tensor] = None,
    tail_state_g: Optional[cute.Tensor] = None,
    tail_state_dx_end: Optional[cute.Tensor] = None,
    tail_dseed: Optional[cute.Tensor] = None,
    tail_seed_row_base: Optional[cute.Tensor] = None,
) -> None:
    if cutlass.const_expr(fold_update is not None):
        if cutlass.const_expr(fold_seed is not None):
            fold_seed = cute.make_tensor(
                fold_seed.iterator + (cutlass.Int32(fold_seed.shape[0]) - cutlass.Int32(1)) * fold_seed.stride[0],
                cute.make_layout((fold_composed.shape[0], fold_seed.shape[1], fold_seed.shape[2], fold_seed.shape[3]), stride=fold_seed.stride),
            )
        launch_state_chain(
            n_heads_out,
            bprop_cfg.d_v,
            bprop_cfg.d_k,
            fold_rows,
            cutlass.Int32(fold_update.shape[0]),
            True,
            False,
            fold_seed is not None,
            True,
            False,
            False,
            cutlass.Int32(1),
            fold_update,
            fold_transition,
            None,
            fold_seed,
            fold_composed,
            None,
            None,
            None,
            None,
            stream,
        )
    if cutlass.const_expr(pack_do is not None):
        pack_chunks = cutlass.Int64(pack_do_x.shape[0]) * cutlass.Int64(pack_do.shape[1]) * cutlass.Int64(bprop_cfg.d_v // 2) // cutlass.Int64(4)
        if cutlass.const_expr(pack_q is not None):
            q_chunks = cutlass.Int64(pack_q_x.shape[0]) * cutlass.Int64(pack_q.shape[1]) * cutlass.Int64(bprop_cfg.d_k // 2) // cutlass.Int64(4)
            pack_chunks = pack_chunks + q_chunks
        expand.pack_bwd_launch(
            pack_q is not None,
            expand_num,
            cutlass.Int32(pack_q.shape[1]) if cutlass.const_expr(pack_q is not None) else cutlass.Int32(1),
            bprop_cfg.d_k // 2 if cutlass.const_expr(pack_q is not None) else 4,
            cutlass.Int32(pack_do.shape[1]),
            bprop_cfg.d_v // 2,
            pack_q if cutlass.const_expr(pack_q is not None) else pack_do,
            pack_q_x if cutlass.const_expr(pack_q is not None) else pack_do_x,
            pack_do,
            pack_do_x,
            cutlass.Int32((pack_chunks + cutlass.Int64(expand.BLOCK - 1)) // cutlass.Int64(expand.BLOCK)),
            stream,
        )
    if cutlass.const_expr(l2norm_q is not None):
        cta_rows = l2norm.THREADS_PER_CTA // l2norm.FWD_LANES * l2norm.FWD_ROWS_PER_GROUP
        q_rows = cutlass.Int32(q.shape[0]) * cutlass.Int32(q.shape[1])
        qk_rows = q_rows + cutlass.Int32(k.shape[0]) * cutlass.Int32(k.shape[1])
        l2norm.l2norm_qk_launch(
            l2norm_q,
            l2norm_k,
            q,
            k,
            l2norm_inv_q,
            l2norm_inv_k,
            q_rows,
            qk_rows,
            cutlass.Int32(q.shape[1]),
            cutlass.Int32(k.shape[1]),
            (qk_rows + cutlass.Int32(cta_rows - 1)) // cutlass.Int32(cta_rows),
            bprop_cfg.d_k,
            expand_num if cutlass.const_expr(pack_do is not None) else 1,
            expand_num - 1 if cutlass.const_expr(pack_do is not None) else 0,
            pack_do is not None,
            l2norm_k_slots,
            l2norm_k_expand_num,
            stream,
        )
    gate_sms = num_sms if cutlass.const_expr(item_limit_word is not None) else 0
    if cutlass.const_expr(tinv_pass):
        gdn_tinv_f16.host(tinv_cfg, True, k, tinv_words, gate, a_log, dt_bias, beta, cu_seqlens, tinv, tinv_rows, tinv_row_count, stream)
    if cutlass.const_expr(split):
        split_k.launch(
            split,
            b_t,
            log_gate,
            safe_gate,
            True,
            overhead_chunks,
            expand_num,
            warmup_cap,
            n_heads_out,
            num_sms,
            n_tiles,
            ideal_chunks,
            batch_size,
            log2_thresh,
            gate_table,
            a_log_table,
            dt_bias_table,
            cu_seqlens_table,
            chunk_scratch,
            item_scratch,
            work_items_table,
            work_count_table,
            scheduler_all,
            n_scan_ctas,
            n_walk_ctas,
            stream,
            item_limit_word,
            gate_sms=gate_sms,
        )
        gate_sms = 0
    if cutlass.const_expr(recompute):
        recompute_base_k, recompute_base_v, recompute_base_checkpoint, recompute_base_tinv = gdn_recompute_f16.base_descriptors(
            io_dtype, b_t, k, v, checkpoints, tinv, expand_num=expand_num, k_slots=k_slots, v_slots=v_slots
        )
        if cutlass.const_expr(compact_qdo):
            base_q, base_k, base_v, base_do, base_checkpoint, base_dq, base_dk, base_dv = bprop.base_descriptors(
                io_dtype, b_t, q, k, v, do, dq, dk, dv, checkpoints, expand_num=expand_num, k_slots=k_slots, v_slots=v_slots
            )
            base_tinv = base_k
        else:
            base_q, base_k, base_v, base_do, base_checkpoint, base_dq, base_dk, base_dv, base_tinv = bprop.base_descriptors(
                io_dtype, b_t, q, k, v, do, dq, dk, dv, checkpoints, tinv, expand_num=expand_num, k_slots=k_slots, v_slots=v_slots
            )
        frost_gdn_regenerate_prologue(
            compact_qdo,
            recompute_orders,
            recompute_order_gen,
            coarse,
            bwd_orders,
            bwd_order_gen,
            b_t,
            expand_num,
            recompute_base_k,
            recompute_base_v,
            recompute_base_checkpoint,
            recompute_base_tinv,
            base_q,
            base_k,
            base_v,
            base_do,
            base_checkpoint,
            base_dq,
            base_dk,
            base_dv,
            base_tinv,
            recompute_words,
            bprop_words,
            cu_seqlens,
            cu_seqlens_bprop,
            q,
            k,
            v,
            do,
            gate,
            checkpoints,
            dq,
            dk,
            dv,
            tinv,
            staging_recompute,
            series_count,
            series_items,
            scheduler_all_recompute,
            staging_bprop,
            work_count,
            work_items,
            scheduler_all_bprop,
            batch_size,
            checkpoint_every_n,
            seed_span_chunks,
            k_slots,
            v_slots,
            item_limit_word,
            gate_sms,
        ).launch(grid=(4, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)
        gdn_recompute_f16.host(
            recompute_cfg,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            cu_seqlens,
            state_in,
            composed,
            None,
            seed_checkpoints,
            tinv,
            series_items,
            series_count,
            scheduler_recompute,
            checkpoint_every_n,
            seed_every_n,
            recompute_words,
            stream,
        )
    elif cutlass.const_expr(compact_qdo):
        bprop.prologue(
            io_dtype,
            b_t,
            bwd_orders,
            bwd_order_gen,
            expand_num,
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            checkpoints,
            cu_seqlens_bprop,
            staging_bprop,
            work_count,
            work_items,
            scheduler_all_bprop,
            bprop_words,
            stream,
            k_slots,
            v_slots,
            item_limit_word=item_limit_word,
            gate_sms=gate_sms,
        )
    else:
        bprop.prologue(
            io_dtype,
            b_t,
            bwd_orders,
            bwd_order_gen,
            expand_num,
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            checkpoints,
            cu_seqlens_bprop,
            staging_bprop,
            work_count,
            work_items,
            scheduler_all_bprop,
            tinv,
            bprop_words,
            stream,
            k_slots,
            v_slots,
            item_limit_word=item_limit_word,
            gate_sms=gate_sms,
        )
    if cutlass.const_expr(compact_qdo):
        bprop.host(
            bprop_cfg,
            q,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            beta,
            dgate,
            dbeta,
            do,
            dq,
            dk,
            dv,
            cu_seqlens_bprop,
            dstate0,
            dstate_in,
            dcomposed,
            inv_q,
            inv_k,
            work_items,
            work_count,
            scheduler_bwd,
            scale,
            bprop_words,
            stream,
        )
    else:
        bprop.host(
            bprop_cfg,
            q,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            beta,
            dgate,
            dbeta,
            do,
            dq,
            dk,
            dv,
            cu_seqlens_bprop,
            dstate0,
            dstate_in,
            dcomposed,
            inv_q,
            inv_k,
            tinv,
            work_items,
            work_count,
            scheduler_bwd,
            scale,
            bprop_words,
            stream,
        )
    if cutlass.const_expr(tail_bprop_cfg is not None):
        gdn_chain_backward_f16.chain_backward_head_host(
            tail_unit_chunks,
            b_t,
            expand_num,
            False,
            compact_qdo,
            tail_summary_q_step,
            tail_summary_cfg is not None,
            tail_series_cfg is not None,
            tinv_pass,
            False,
            tail_tinv_cfg,
            tail_summary_cfg,
            tail_transition_cfg,
            tail_series_cfg,
            tail_bprop_cfg.d_v,
            tail_bprop_cfg.d_k,
            tail_chain_rows,
            tail_seed is not None,
            True,
            tail_pieces,
            n_heads_out,
            tail_num_seqs,
            tail_series_span_chunks,
            checkpoint_every_n,
            seed_every_n,
            tail_batch_base,
            tail_first_head_lo,
            q,
            k,
            v,
            do,
            gate,
            beta,
            a_log,
            dt_bias,
            cu_seqlens,
            tail_cu_pieces,
            tail_main_rows,
            tail_summary_rows,
            tail_main_count,
            tail_summary_count,
            tail_work_items,
            tail_work_items_summary,
            tail_series_items,
            tail_series_count,
            tail_recompute_items,
            tail_recompute_count,
            tail_scheduler_all,
            tail_scheduler_recompute,
            tail_scheduler_m,
            tail_scheduler_series,
            tail_tinv_words,
            tail_tinv_rows,
            tail_tinv_row_count,
            tail_summary_words,
            tail_recompute_m_words,
            tail_series_words,
            tail_bprop_summary_words,
            tail_bprop_words,
            checkpoints,
            dq,
            dk,
            dv,
            tail_summary_q,
            tail_summary_do,
            tail_tinv,
            tail_state_h_summary,
            tail_state_m_main,
            tail_state_h_chain,
            tail_state_m_chain,
            tail_state_x_chain,
            tail_seed,
            None,
            tail_state_x_series,
            seed_checkpoints,
            tail_seed_row_base,
            stream,
            k_slots,
            v_slots,
            item_limit_word,
            bprop,
            False,
            tail_bwd_summary_cfg,
            tail_bprop_cfg,
            tail_dseed is not None,
            scale,
            tail_state_g,
            tail_state_dx_end,
            tail_dseed,
            dcomposed,
            dgate,
            dbeta,
            dstate0,
            inv_q,
            inv_k,
            tail_scheduler_summary,
            tail_scheduler_bwd,
        )
    if cutlass.const_expr(gate_bwd_dgate is not None):
        gate_tokens = cutlass.Int32(gate_bwd_dgate.shape[0])
        gate_heads = cutlass.Int32(gate_bwd_dgate.shape[1])
        gate_bwd.scalar_gate_bwd_launch(
            gate_bwd_dgate,
            gate_bwd_gate,
            gate_bwd_a_log,
            gate_bwd_dt_bias,
            gate_part_a,
            gate_part_dt,
            d_a_log,
            d_dt_bias,
            gate_tokens,
            gate_heads,
            (gate_tokens + gate_blocks - cutlass.Int32(1)) // gate_blocks,
            gate_blocks,
            (gate_heads + cutlass.Int32(gate_bwd.SCALAR_HEAD_TILE - 1)) // cutlass.Int32(gate_bwd.SCALAR_HEAD_TILE),
            stream,
        )
    if cutlass.const_expr(reduce_dq is not None):
        dq_words = cutlass.Int64(reduce_dq.shape[0]) * cutlass.Int64(reduce_dq.shape[1]) * cutlass.Int64(bprop_cfg.d_k // 2)
        head_reduce.launch(
            reduce_dq_src,
            reduce_dq,
            dq_words,
            cutlass.Int64(reduce_dq.stride[0]) // cutlass.Int64(2),
            cutlass.Int64(reduce_dq.stride[1]) // cutlass.Int64(2),
            cutlass.Int32((dq_words + cutlass.Int64(head_reduce.BLOCK - 1)) // cutlass.Int64(head_reduce.BLOCK)),
            cutlass.Int32(reduce_dq.shape[1]),
            cutlass.Int32(reduce_dq_src.shape[1]) // cutlass.Int32(reduce_dq.shape[1]),
            bprop_cfg.d_k // 2,
            bprop_cfg.io_dtype,
            stream,
        )
    if cutlass.const_expr(reduce_dk is not None):
        dk_words = cutlass.Int64(reduce_dk.shape[0]) * cutlass.Int64(reduce_dk.shape[1]) * cutlass.Int64(bprop_cfg.d_k // 2)
        head_reduce.launch(
            reduce_dk_src,
            reduce_dk,
            dk_words,
            cutlass.Int64(reduce_dk.stride[0]) // cutlass.Int64(2),
            cutlass.Int64(reduce_dk.stride[1]) // cutlass.Int64(2),
            cutlass.Int32((dk_words + cutlass.Int64(head_reduce.BLOCK - 1)) // cutlass.Int64(head_reduce.BLOCK)),
            cutlass.Int32(reduce_dk.shape[1]),
            cutlass.Int32(reduce_dk_src.shape[1]) // cutlass.Int32(reduce_dk.shape[1]),
            bprop_cfg.d_k // 2,
            bprop_cfg.io_dtype,
            stream,
        )
    if cutlass.const_expr(reduce_dv is not None):
        dv_words = cutlass.Int64(reduce_dv.shape[0]) * cutlass.Int64(reduce_dv.shape[1]) * cutlass.Int64(bprop_cfg.d_v // 2)
        head_reduce.launch(
            reduce_dv_src,
            reduce_dv,
            dv_words,
            cutlass.Int64(reduce_dv.stride[0]) // cutlass.Int64(2),
            cutlass.Int64(reduce_dv.stride[1]) // cutlass.Int64(2),
            cutlass.Int32((dv_words + cutlass.Int64(head_reduce.BLOCK - 1)) // cutlass.Int64(head_reduce.BLOCK)),
            cutlass.Int32(reduce_dv.shape[1]),
            cutlass.Int32(reduce_dv_src.shape[1]) // cutlass.Int32(reduce_dv.shape[1]),
            bprop_cfg.d_v // 2,
            bprop_cfg.io_dtype,
            stream,
        )
    if cutlass.const_expr(gather_dq is not None):
        gather_chunks = cutlass.Int64(gather_dq.shape[0]) * cutlass.Int64(gather_dq.shape[1]) * cutlass.Int64(bprop_cfg.d_k // 2) // cutlass.Int64(4)
        expand.gather_dq_launch(
            expand_num,
            cutlass.Int32(gather_dq.shape[1]),
            bprop_cfg.d_k // 2,
            gather_dq_src,
            gather_dq,
            cutlass.Int32((gather_chunks + cutlass.Int64(expand.BLOCK - 1)) // cutlass.Int64(expand.BLOCK)),
            stream,
        )


def build_warmup_backward(
    *,
    bprop_module,
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    gate,
    beta,
    a_log,
    dt_bias,
    cu_seqlens,
    tinv,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    checkpoints,
    seed_checkpoints,
    state_in,
    composed=None,
    work_items,
    work_count,
    series_items,
    series_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    recompute_words,
    bprop_words,
    dgate,
    dbeta,
    dstate0,
    dstate_in,
    dcomposed,
    inv_q,
    inv_k,
    scheduler_bwd,
    scale,
    use_initial_state,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    expand_num,
    tinv_pass,
    recompute,
    recompute_orders,
    coarse,
    bwd_orders,
    compact_qdo,
    seed_span_tokens,
    seed_every_n_tokens,
    log_gate,
    safe_gate,
    use_beta_sigmoid,
    allow_neg_eigval,
    device,
    stream,
    item_limit_word=None,
    tail=None,
    tail_chain_rows=0,
    tail_summary_q_step=1,
    tail_scalars=(),
    tail_buffers=(),
    lead_buffers=(None,) * 12,
    gate_blocks=0,
    post_buffers=(None,) * 16,
):
    """Compile (cached per static config) the warmup or uncut backward launch over the buffers of one plan.  The
    placeholders repeat the marks of the standalone builds so every kernel compiles as it does there."""
    DK = q.shape[2]
    DV = v.shape[-1]
    if not safe_gate:
        a_log = None
        dt_bias = None
    facts = split_k.split_table_facts(
        gate,
        cu_seqlens,
        split=split,
        n_tiles=n_tiles,
        ideal_chunks=ideal_chunks,
        num_sms=num_sm,
        b_t=b_t,
        log2_threshold=None,
        log_gate=log_gate,
        safe_gate=safe_gate,
        expand_num=expand_num,
    )
    io_dtype = get_dtype(q.dtype)
    recompute_order_gen = not (recompute_orders and split)
    bwd_order_gen = bwd_orders and not split
    cu_align = 8 if str(cu_seqlens.dtype).endswith("int64") else 4
    k_slots = family_slots(k, expand_num, 3)
    v_slots = family_slots(v, expand_num, 3)
    beta_slots = family_slots(beta, expand_num, 2)
    fold_update, fold_transition, fold_seed, fold_composed = lead_buffers[:4]
    l2norm_k = lead_buffers[9]
    fold_rows = chain_rows_per_cta(DV, DK, 1, gate.shape[1], num_sm) if fold_update is not None else 0
    l2norm_k_slots = family_slots(l2norm_k, expand_num, 3) if l2norm_k is not None else 1
    l2norm_k_expand_num = int(l2norm_k.shape[1]) if l2norm_k is not None and len(l2norm_k.shape) == 4 else 1
    key = (
        str(q.dtype),
        str(seed_checkpoints.dtype) if coarse else "none",
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(beta.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(state_in.dtype) if state_in is not None else "none",
        composed is not None,
        int(device),
        int(num_sm),
        DK,
        DV,
        int(b_t),
        int(expand_num),
        bool(split),
        bool(tinv_pass),
        bool(recompute),
        bool(recompute_orders),
        bool(coarse),
        bool(bwd_orders),
        bool(compact_qdo),
        bool(log_gate),
        bool(safe_gate),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        bprop_module.__name__,
        len(k.shape),
        len(v.shape),
        len(beta.shape),
        k_slots,
        v_slots,
        beta_slots,
        item_limit_word is not None,
        (
            (
                int(tail.unit_chunks),
                int(tail_chain_rows),
                int(tail_summary_q_step),
                *(str(buffer.dtype) if buffer is not None else None for buffer in tail_buffers),
                *(buffer.shape[1] for name, buffer in zip(TAIL_BUFFERS, tail_buffers) if name in ("summary_q", "summary_do")),
            )
            if tail is not None
            else None
        ),
        str(dstate0.dtype) if dstate0 is not None else "none",
        str(dstate_in.dtype) if dstate_in is not None else "none",
        dcomposed is not None,
        inv_q is not None,
        bool(use_initial_state),
        tuple(str(buffer.dtype) if buffer is not None else None for buffer in (*lead_buffers, *post_buffers)),
        fold_rows,
        len(l2norm_k.shape) if l2norm_k is not None else 0,
        l2norm_k_slots,
        l2norm_k_expand_num,
    )
    if key not in warmup_backward_cache:
        tinv_cfg = None
        if tinv_pass:
            tinv_cfg = gdn_tinv_f16.build_cfg(
                io_dtype,
                num_sm=num_sm,
                log_gate=log_gate,
                safe_gate=safe_gate,
                beta_sigmoid=use_beta_sigmoid,
                allow_neg_eigval=allow_neg_eigval,
                d_k=DK,
                expand_num=expand_num,
                k_slots=k_slots,
                beta_slots=beta_slots,
            )
        recompute_cfg = None
        if recompute:
            recompute_cfg = gdn_recompute_f16.build_cfg(
                io_dtype,
                get_dtype(state_in.dtype) if state_in is not None else cutlass.Float32,
                max_active_clusters=num_sm,
                use_initial_state=state_in is not None or composed is not None,
                state_table=state_in is not None,
                has_composed=composed is not None,
                store_final_state=False,
                enable_checkpoints=True,
                seed_checkpoints=coarse,
                log_gate=log_gate,
                safe_gate=safe_gate,
                d_k=DK,
                d_v=DV,
                expand_num=expand_num,
                k_slots=k_slots,
                v_slots=v_slots,
            )
        bprop_cfg = bprop_module.build_cfg(
            io_dtype,
            max_active_clusters=num_sm,
            use_initial_state=use_initial_state,
            use_dstate_in=dstate_in is not None or dcomposed is not None,
            use_dstate0=dstate0 is not None,
            log_gate=log_gate,
            safe_gate=safe_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            fused_l2norm=inv_q is not None,
            **({} if compact_qdo else dict(tinv_source="gmem" if tinv_pass else "compute")),
            d_k=DK,
            d_v=DV,
            expand_num=expand_num,
            k_slots=k_slots,
            v_slots=v_slots,
            beta_slots=beta_slots,
            dstate_table=dstate_in is not None,
            has_composed_dstate=dcomposed is not None,
        )

        gate_table_placeholder = None
        dt_bias_table_placeholder = None
        staging_placeholder = None
        item_scratch_placeholder = None
        chunk_scratch_placeholder = None
        if split:
            gate_table_placeholder = from_dlpack(gate, assumed_align=8 if facts.gate_elem_bytes == 2 else 4).mark_layout_dynamic(
                leading_dim=len(gate.shape) - 1
            )
            staging_placeholder = from_dlpack(item_scratch, assumed_align=16)
            staging_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            item_scratch_placeholder = from_dlpack(item_scratch, assumed_align=4)
            item_scratch_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            chunk_scratch_placeholder = from_dlpack(chunk_scratch, assumed_align=4)
            chunk_scratch_placeholder.mark_layout_dynamic(leading_dim=1)
        if dt_bias is not None:
            dt_bias_table_placeholder = from_dlpack(dt_bias, assumed_align=4)
            dt_bias_table_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(len(dt_bias.shape))), divisibility=1)
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_table_placeholder = from_dlpack(work_items, assumed_align=4)
        work_items_table_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_table_placeholder = from_dlpack(work_count, assumed_align=4)
        work_count_table_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0,), divisibility=1)
        series_items_placeholder = None
        if recompute:
            series_items_placeholder = from_dlpack(series_items, assumed_align=16)
            series_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        state_in_placeholder = None
        if state_in is not None:
            state_in_placeholder = from_dlpack(state_in, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        composed_placeholder = None
        if composed is not None:
            composed_placeholder = from_dlpack(composed, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        tail_constexprs = ()
        tail_placeholders = ()
        if tail is not None:
            buffers = dict(zip(TAIL_BUFFERS, tail_buffers))
            HO = gate.shape[1]
            tail_cfgs = gdn_chain_backward_f16.build_configs(
                bprop_module,
                io_dtype,
                get_dtype(buffers["state_h_summary"].dtype) if buffers["state_h_summary"] is not None else None,
                get_dtype(buffers["state_m_main"].dtype),
                get_dtype(buffers["state_x_series"].dtype) if buffers["state_x_series"] is not None else cutlass.Float32,
                fused_h_m=buffers["state_h_summary"] is not None,
                series=buffers["recompute_items"] is not None,
                coarse=coarse,
                compact_qdo=compact_qdo,
                use_dstate0=dstate0 is not None,
                fused_l2norm=inv_q is not None,
                num_sm=num_sm,
                log_gate=log_gate,
                safe_gate=safe_gate,
                beta_sigmoid=use_beta_sigmoid,
                allow_neg_eigval=allow_neg_eigval,
                d_k=DK,
                d_v=DV,
                expand_num=expand_num,
                k_slots=k_slots,
                v_slots=v_slots,
                beta_slots=beta_slots,
            )
            tail_constexprs = (int(tail.unit_chunks), int(tail_chain_rows), int(tail_summary_q_step), *tail_cfgs)
            tail_items = {}
            for name in ("work_items", "work_items_summary", "series_items"):
                tail_items[name] = None
                if buffers[name] is not None:
                    tail_items[name] = from_dlpack(buffers[name], assumed_align=16)
                    tail_items[name].mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_states = {}
            for name in ("state_h_summary", "state_m_main", "state_x_series"):
                tail_states[name] = None
                if buffers[name] is not None:
                    tail_states[name] = from_dlpack(buffers[name], assumed_align=16)
                    tail_states[name].mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
            chain_states = [
                from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3) for _ in range(2)
            ]
            tail_placeholders = [
                *(cutlass.Int32(int(value)) for value in tail_scalars),
                from_dlpack(buffers["cu_pieces"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["main_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["summary_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["main_count"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["summary_count"], assumed_align=4).mark_layout_dynamic(),
                tail_items["work_items"],
                tail_items["work_items_summary"],
                tail_items["series_items"],
                from_dlpack(buffers["series_count"], assumed_align=4).mark_layout_dynamic() if buffers["series_count"] is not None else None,
                (tail_items["series_items"] if coarse else tail_items["work_items"]) if buffers["recompute_items"] is not None else None,
                from_dlpack(buffers["recompute_count"], assumed_align=4).mark_layout_dynamic() if buffers["recompute_count"] is not None else None,
                *(
                    from_dlpack(buffers[name], assumed_align=4).mark_layout_dynamic()
                    for name in ("scheduler_all", "scheduler_recompute", "scheduler_m", "scheduler_series", "scheduler_summary", "scheduler_bwd")
                ),
                from_dlpack(buffers["tinv_words"], assumed_align=128).mark_layout_dynamic() if buffers["tinv_words"] is not None else None,
                from_dlpack(buffers["tinv_rows"], assumed_align=16).mark_layout_dynamic(leading_dim=1) if buffers["tinv_rows"] is not None else None,
                from_dlpack(buffers["tinv_row_count"], assumed_align=4).mark_layout_dynamic() if buffers["tinv_row_count"] is not None else None,
                *(
                    from_dlpack(buffers[name], assumed_align=128).mark_layout_dynamic() if buffers[name] is not None else None
                    for name in ("summary_words", "recompute_m_words", "series_words", "bprop_summary_words", "bprop_words")
                ),
                from_dlpack(buffers["summary_q"], assumed_align=16).mark_layout_dynamic(leading_dim=2),
                from_dlpack(buffers["summary_do"], assumed_align=16).mark_layout_dynamic(leading_dim=2),
                from_dlpack(buffers["tinv"], assumed_align=128).mark_layout_dynamic(leading_dim=3),
                tail_states["state_h_summary"],
                tail_states["state_m_main"],
                chain_states[0] if buffers["state_h_chain"] is not None else None,
                from_dlpack(DeviceView(256, (1, HO, DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                chain_states[1] if buffers["state_x_chain"] is not None else None,
                (
                    from_dlpack(DeviceView(256, (1, HO, DV, DK), dtype_name(buffers["seed"].dtype), int(device)), assumed_align=4).mark_layout_dynamic(
                        leading_dim=3
                    )
                    if buffers["seed"] is not None
                    else None
                ),
                tail_states["state_x_series"],
                from_dlpack(buffers["state_g"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(buffers["state_dx_end"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                (
                    from_dlpack(DeviceView(256, (1, HO, DV, DK), dtype_name(buffers["dseed"].dtype), int(device)), assumed_align=4).mark_layout_dynamic(
                        leading_dim=3
                    )
                    if buffers["dseed"] is not None
                    else None
                ),
                from_dlpack(buffers["seed_row_base"], assumed_align=4).mark_layout_dynamic() if buffers["seed_row_base"] is not None else None,
            ]
        fold_placeholders = [None] * 4
        if fold_update is not None:
            fold_placeholders = [
                from_dlpack(DeviceView(256, (1, gate.shape[1], DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(DeviceView(256, (1, gate.shape[1], DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                (
                    from_dlpack(DeviceView(256, (1, gate.shape[1], DV, DK), dtype_name(fold_seed.dtype), int(device)), assumed_align=4).mark_layout_dynamic(
                        leading_dim=3
                    )
                    if fold_seed is not None
                    else None
                ),
                from_dlpack(DeviceView(256, (1, gate.shape[1], DV, DK), dtype_name(fold_composed.dtype), int(device)), assumed_align=4).mark_layout_dynamic(
                    leading_dim=3
                ),
            ]
        pack_placeholders = [
            from_dlpack(buffer, assumed_align=4).mark_layout_dynamic(leading_dim=len(buffer.shape) - 1) if buffer is not None else None
            for buffer in lead_buffers[4:8]
        ]
        l2norm_placeholders = [
            from_dlpack(buffer, assumed_align=16).mark_layout_dynamic(leading_dim=len(buffer.shape) - 1) if buffer is not None else None
            for buffer in lead_buffers[8:]
        ]
        post_placeholders = [
            from_dlpack(buffer, assumed_align=4).mark_layout_dynamic(leading_dim=len(buffer.shape) - 1) if buffer is not None else None
            for buffer in post_buffers
        ]
        warmup_backward_cache[key] = cute.compile(
            warmup_backward_host,
            facts.split,
            facts.b_t,
            facts.log_gate,
            facts.safe_gate,
            facts.overhead_chunks,
            facts.expand_num,
            facts.warmup_cap,
            cutlass.Int32(facts.n_heads_out),
            facts.num_sms,
            io_dtype,
            bprop_module,
            bool(compact_qdo),
            bool(tinv_pass),
            bool(recompute),
            bool(recompute_orders),
            recompute_order_gen,
            bool(coarse),
            bool(bwd_orders),
            bwd_order_gen,
            tinv_cfg,
            recompute_cfg,
            bprop_cfg,
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(facts.ideal_chunks),
            cutlass.Int32(facts.batch_size),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_walk_ctas),
            cutlass.Int32(int(b_t)),
            cutlass.Int32(int(seed_span_tokens or seed_every_n_tokens) // int(b_t)),
            cutlass.Int32(int(seed_every_n_tokens)),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=len(k.shape) - 1),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=len(v.shape) - 1),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dq, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dk, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dv, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            gate_table_placeholder,
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=len(beta.shape) - 1),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            dt_bias_table_placeholder,
            from_dlpack(cu_seqlens, assumed_align=cu_align).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(tinv, assumed_align=128).mark_layout_dynamic(leading_dim=3) if tinv_pass else None,
            from_dlpack(tinv_words, assumed_align=128).mark_layout_dynamic() if tinv_pass else None,
            from_dlpack(tinv_rows, assumed_align=16).mark_layout_dynamic(leading_dim=1) if tinv_pass else None,
            from_dlpack(tinv_row_count, assumed_align=4).mark_layout_dynamic() if tinv_pass else None,
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(seed_checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if coarse else None,
            state_in_placeholder,
            composed_placeholder,
            work_items_placeholder,
            work_items_table_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            work_count_table_placeholder,
            series_items_placeholder,
            from_dlpack(series_count, assumed_align=4).mark_layout_dynamic() if recompute else None,
            staging_placeholder if recompute and recompute_orders else None,
            staging_placeholder if bwd_orders else None,
            item_scratch_placeholder,
            chunk_scratch_placeholder,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic() if recompute and (recompute_orders or coarse) else None,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic() if bwd_orders else None,
            from_dlpack(scheduler_recompute, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(recompute_words, assumed_align=128).mark_layout_dynamic() if recompute else None,
            from_dlpack(bprop_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(dgate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dbeta, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate0 is not None else None,
            from_dlpack(dstate_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate_in is not None else None,
            from_dlpack(dcomposed, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dcomposed is not None else None,
            from_dlpack(inv_q, assumed_align=4).mark_layout_dynamic(leading_dim=1) if inv_q is not None else None,
            from_dlpack(inv_k, assumed_align=4).mark_layout_dynamic(leading_dim=1) if inv_k is not None else None,
            from_dlpack(scheduler_bwd, assumed_align=4).mark_layout_dynamic(),
            float(scale),
            cuda.CUstream(int(stream)),
            k_slots,
            v_slots,
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            fold_rows,
            *fold_placeholders,
            *pack_placeholders,
            l2norm_k_slots,
            l2norm_k_expand_num,
            *l2norm_placeholders,
            cutlass.Int32(int(gate_blocks)),
            *post_placeholders,
            *tail_constexprs,
            *tail_placeholders,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return warmup_backward_cache[key], facts


def run_warmup_backward(
    compiled,
    facts,
    *,
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    gate,
    beta,
    a_log,
    dt_bias,
    cu_seqlens,
    tinv,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    checkpoints,
    seed_checkpoints,
    state_in,
    composed=None,
    work_items,
    work_count,
    series_items,
    series_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    recompute_words,
    bprop_words,
    dgate,
    dbeta,
    dstate0,
    dstate_in,
    dcomposed,
    inv_q,
    inv_k,
    scheduler_bwd,
    scale,
    b_t,
    tinv_pass,
    recompute,
    recompute_orders,
    coarse,
    bwd_orders,
    compact_qdo,
    seed_span_tokens,
    seed_every_n_tokens,
    stream,
    item_limit_word=None,
    tail_scalars=(),
    tail_buffers=(),
    lead_buffers=(None,) * 12,
    gate_blocks=0,
    post_buffers=(None,) * 16,
) -> None:
    """Replay the warmup or uncut backward: one crossing into the DSL.  The plan validated the contract at
    build, so nothing here raises."""
    compiled(
        facts.n_heads_out,
        facts.n_tiles,
        facts.ideal_chunks,
        facts.batch_size,
        facts.log2_threshold,
        facts.n_scan_ctas,
        facts.n_walk_ctas,
        int(b_t),
        int(seed_span_tokens or seed_every_n_tokens) // int(b_t),
        int(seed_every_n_tokens),
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        gate,
        gate if facts.split else None,
        beta,
        a_log if facts.safe_gate else None,
        a_log if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        cu_seqlens,
        cu_seqlens,
        cu_seqlens,
        tinv if tinv_pass else None,
        tinv_words if tinv_pass else None,
        tinv_rows if tinv_pass else None,
        tinv_row_count if tinv_pass else None,
        checkpoints,
        seed_checkpoints if coarse else None,
        state_in,
        composed,
        work_items,
        work_items,
        work_count,
        work_count,
        series_items if recompute else None,
        series_count if recompute else None,
        item_scratch if facts.split and recompute and recompute_orders else None,
        item_scratch if facts.split and bwd_orders else None,
        item_scratch if facts.split else None,
        chunk_scratch if facts.split else None,
        scheduler_all,
        scheduler_all if recompute and (recompute_orders or coarse) else None,
        scheduler_all if bwd_orders else None,
        scheduler_recompute,
        recompute_words if recompute else None,
        bprop_words,
        dgate,
        dbeta,
        dstate0,
        dstate_in,
        dcomposed,
        inv_q,
        inv_k,
        scheduler_bwd,
        float(scale),
        cuda.CUstream(int(stream)),
        item_limit_word,
        *lead_buffers,
        int(gate_blocks),
        *post_buffers,
        *tail_scalars,
        *tail_buffers,
    )


SUMMARY_TAIL_BUFFERS = (
    "cu_pieces",
    "main_rows",
    "summary_rows",
    "main_count",
    "summary_count",
    "work_items",
    "scheduler_all",
    "scheduler_m",
    "scheduler_summary",
    "recompute_m_words",
    "bprop_summary_words",
    "state_m_main",
    "state_m_chain",
    "state_g",
    "state_g_chain",
    "state_x_chain",
    "dstate_in",
    "dstate0",
)

warmup_summary_backward_cache = {}


@cute.jit
def warmup_summary_backward_host(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    overhead_chunks: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    order_gen: cutlass.Constexpr[bool],
    tinv_cfg: cutlass.Constexpr,
    bprop_summary_cfg: cutlass.Constexpr,
    transition_cfg: cutlass.Constexpr,
    transition_rows: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    do: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    beta: cute.Tensor,
    a_log: Optional[cute.Tensor],
    a_log_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    cu_seqlens_bprop: cute.Tensor,
    dstate0: cute.Tensor,
    dstate_in: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_items_table: cute.Tensor,
    work_count: cute.Tensor,
    work_count_table: cute.Tensor,
    staging: Optional[cute.Tensor],
    item_scratch: Optional[cute.Tensor],
    chunk_scratch: Optional[cute.Tensor],
    scheduler: cute.Tensor,
    scheduler_all: cute.Tensor,
    workspace: cute.Tensor,
    tinv: cute.Tensor,
    tinv_words: cute.Tensor,
    tinv_rows: cute.Tensor,
    tinv_row_count: cute.Tensor,
    transition_m: Optional[cute.Tensor],
    transition: Optional[cute.Tensor],
    scheduler_transition: Optional[cute.Tensor],
    transition_words: Optional[cute.Tensor],
    stream: cuda.CUstream,
    k_slots: cutlass.Constexpr[int] = 1,
    item_limit_word: Optional[cute.Tensor] = None,
    l2norm_k_slots: cutlass.Constexpr[int] = 1,
    l2norm_k_expand_num: cutlass.Constexpr[int] = 1,
    l2norm_q: Optional[cute.Tensor] = None,
    l2norm_k: Optional[cute.Tensor] = None,
    l2norm_inv_q: Optional[cute.Tensor] = None,
    l2norm_inv_k: Optional[cute.Tensor] = None,
    tail_unit_chunks: cutlass.Constexpr[int] = 0,
    tail_chain_rows: cutlass.Constexpr[int] = 0,
    tail_transition_cfg: cutlass.Constexpr = None,
    tail_bprop_summary_cfg: cutlass.Constexpr = None,
    tail_pieces: cutlass.Int32 = 0,
    tail_num_seqs: cutlass.Int32 = 0,
    tail_batch_base: cutlass.Int32 = 0,
    tail_first_head_lo: cutlass.Int32 = 0,
    tail_cu_pieces: Optional[cute.Tensor] = None,
    tail_main_rows: Optional[cute.Tensor] = None,
    tail_summary_rows: Optional[cute.Tensor] = None,
    tail_main_count: Optional[cute.Tensor] = None,
    tail_summary_count: Optional[cute.Tensor] = None,
    tail_work_items: Optional[cute.Tensor] = None,
    tail_scheduler_all: Optional[cute.Tensor] = None,
    tail_scheduler_m: Optional[cute.Tensor] = None,
    tail_scheduler_summary: Optional[cute.Tensor] = None,
    tail_recompute_m_words: Optional[cute.Tensor] = None,
    tail_bprop_summary_words: Optional[cute.Tensor] = None,
    tail_state_m_main: Optional[cute.Tensor] = None,
    tail_state_m_chain: Optional[cute.Tensor] = None,
    tail_state_g: Optional[cute.Tensor] = None,
    tail_state_g_chain: Optional[cute.Tensor] = None,
    tail_state_x_chain: Optional[cute.Tensor] = None,
    tail_dstate_in: Optional[cute.Tensor] = None,
    tail_dstate0: Optional[cute.Tensor] = None,
) -> None:
    """One compiled launch of the backward summary node without a chain of its own."""
    gate_sms = num_sms if cutlass.const_expr(item_limit_word is not None) else 0
    if cutlass.const_expr(l2norm_q is not None):
        cta_rows = l2norm.THREADS_PER_CTA // l2norm.FWD_LANES * l2norm.FWD_ROWS_PER_GROUP
        q_rows = cutlass.Int32(q.shape[0]) * cutlass.Int32(q.shape[1])
        qk_rows = q_rows + cutlass.Int32(k.shape[0]) * cutlass.Int32(k.shape[1])
        l2norm.l2norm_qk_launch(
            l2norm_q,
            l2norm_k,
            q,
            k,
            l2norm_inv_q,
            l2norm_inv_k,
            q_rows,
            qk_rows,
            cutlass.Int32(q.shape[1]),
            cutlass.Int32(k.shape[1]),
            (qk_rows + cutlass.Int32(cta_rows - 1)) // cutlass.Int32(cta_rows),
            bprop_summary_cfg.d_k,
            1,
            0,
            False,
            l2norm_k_slots,
            l2norm_k_expand_num,
            stream,
        )
    gdn_tinv_f16.host(tinv_cfg, True, k, tinv_words, gate, a_log, dt_bias, beta, cu_seqlens, tinv, tinv_rows, tinv_row_count, stream)
    if cutlass.const_expr(split):
        split_k.launch(
            split,
            b_t,
            log_gate,
            safe_gate,
            True,
            overhead_chunks,
            expand_num,
            warmup_cap,
            n_heads_out,
            num_sms,
            n_tiles,
            ideal_chunks,
            batch_size,
            log2_thresh,
            gate_table,
            a_log_table,
            dt_bias_table,
            cu_seqlens_table,
            chunk_scratch,
            item_scratch,
            work_items_table,
            work_count_table,
            scheduler_all,
            n_scan_ctas,
            n_walk_ctas,
            stream,
            item_limit_word,
            gate_sms=gate_sms,
        )
        gate_sms = 0
    gdn_bprop_summary_f16.prologue(
        io_dtype,
        b_t,
        True,
        order_gen,
        expand_num,
        1,
        q,
        k,
        do,
        cu_seqlens_bprop,
        staging,
        work_count,
        work_items,
        scheduler_all,
        tinv,
        workspace,
        stream,
        k_slots,
        item_limit_word,
        gate_sms,
    )
    gdn_bprop_summary_f16.host(
        bprop_summary_cfg,
        q,
        k,
        gate,
        a_log,
        dt_bias,
        do,
        cu_seqlens_bprop,
        dstate0,
        dstate_in,
        None,
        tinv,
        work_items,
        work_count,
        scheduler,
        scale,
        workspace,
        stream,
    )
    if cutlass.const_expr(transition_m is not None):
        gdn_recompute_f16.prologue(
            io_dtype,
            b_t,
            True,
            order_gen,
            False,
            expand_num,
            k,
            k,
            gate,
            cu_seqlens,
            None,
            staging,
            work_count,
            work_items,
            scheduler_all,
            cutlass.Int32(0),
            cutlass.Int32(0),
            tinv,
            transition_words,
            stream,
            1,
            k_slots,
            k_slots,
            item_limit_word,
        )
        gdn_recompute_f16.host(
            transition_cfg,
            k,
            k,
            gate,
            a_log,
            dt_bias,
            cu_seqlens,
            None,
            None,
            transition_m,
            None,
            tinv,
            work_items,
            work_count,
            scheduler_transition,
            cutlass.Int32(0),
            cutlass.Int32(0),
            transition_words,
            stream,
        )
        launch_state_chain(
            n_heads_out,
            bprop_summary_cfg.d_v,
            bprop_summary_cfg.d_k,
            transition_rows,
            cutlass.Int32(1),
            True,
            True,
            False,
            False,
            True,
            True,
            batch_size,
            None,
            transition_m,
            None,
            None,
            None,
            transition,
            None,
            None,
            None,
            stream,
            None,
            item_limit_word,
        )
    if cutlass.const_expr(tail_bprop_summary_cfg is not None):
        gdn_chain_backward_f16.chain_summary_backward_host(
            tail_unit_chunks,
            b_t,
            expand_num,
            False,
            None,
            tail_transition_cfg,
            tail_bprop_summary_cfg,
            tail_bprop_summary_cfg.d_v,
            tail_bprop_summary_cfg.d_k,
            tail_chain_rows,
            tail_dstate_in is not None,
            True,
            transition is not None,
            tail_pieces,
            n_heads_out,
            tail_num_seqs,
            scale,
            tail_batch_base,
            tail_first_head_lo,
            q,
            k,
            do,
            gate,
            beta,
            a_log,
            dt_bias,
            cu_seqlens,
            tail_cu_pieces,
            tail_main_rows,
            tail_summary_rows,
            tail_main_count,
            tail_summary_count,
            tail_work_items,
            tail_scheduler_all,
            tail_scheduler_m,
            tail_scheduler_summary,
            None,
            None,
            None,
            tail_recompute_m_words,
            tail_bprop_summary_words,
            tinv,
            tail_state_m_main,
            tail_state_m_chain,
            tail_state_g,
            tail_state_g_chain,
            tail_state_x_chain,
            tail_dstate_in,
            tail_dstate0,
            transition,
            stream,
            k_slots,
            item_limit_word,
        )


def build_warmup_summary_backward(
    *,
    q,
    k,
    do,
    gate,
    beta,
    a_log,
    dt_bias,
    cu_seqlens,
    dstate0,
    dstate_in,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    scheduler_all,
    workspace,
    tinv,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    transition_m,
    transition,
    scheduler_transition,
    transition_words,
    transition_rows,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    log_gate,
    safe_gate,
    use_beta_sigmoid,
    allow_neg_eigval,
    expand_num,
    scale,
    device,
    stream,
    item_limit_word=None,
    l2norm_buffers=(None,) * 4,
    tail=None,
    tail_chain_rows=0,
    tail_scalars=(),
    tail_buffers=(),
):
    """Compile (cached per static config) the backward summary launch over the buffers of one plan."""
    DK = q.shape[2]
    DV = do.shape[2]
    HO = gate.shape[1]
    if not safe_gate:
        a_log = None
        dt_bias = None
    facts = split_k.split_table_facts(
        gate,
        cu_seqlens,
        split=split,
        n_tiles=n_tiles,
        ideal_chunks=ideal_chunks,
        num_sms=num_sm,
        b_t=b_t,
        log2_threshold=None,
        log_gate=log_gate,
        safe_gate=safe_gate,
        expand_num=expand_num,
    )
    io_dtype = get_dtype(q.dtype)
    k_slots = family_slots(k, expand_num, 3)
    beta_slots = family_slots(beta, expand_num, 2)
    l2norm_k = l2norm_buffers[1]
    l2norm_k_slots = family_slots(l2norm_k, expand_num, 3) if l2norm_k is not None else 1
    l2norm_k_expand_num = int(l2norm_k.shape[1]) if l2norm_k is not None and len(l2norm_k.shape) == 4 else 1
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(beta.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        str(dstate0.dtype),
        str(dstate_in.dtype) if dstate_in is not None else "none",
        str(transition.dtype) if transition is not None else "none",
        transition_m is not None,
        int(transition_rows),
        int(device),
        int(num_sm),
        DK,
        DV,
        int(b_t),
        int(expand_num),
        bool(split),
        bool(log_gate),
        bool(safe_gate),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        len(k.shape),
        len(beta.shape),
        k_slots,
        beta_slots,
        item_limit_word is not None,
        tuple(str(buffer.dtype) if buffer is not None else None for buffer in l2norm_buffers),
        len(l2norm_k.shape) if l2norm_k is not None else 0,
        l2norm_k_slots,
        l2norm_k_expand_num,
        (int(tail.unit_chunks), int(tail_chain_rows)) if tail is not None else None,
    )
    if key not in warmup_summary_backward_cache:
        tinv_cfg = gdn_tinv_f16.build_cfg(
            io_dtype,
            num_sm=num_sm,
            log_gate=log_gate,
            safe_gate=safe_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            d_k=DK,
            expand_num=expand_num,
            k_slots=k_slots,
            beta_slots=beta_slots,
        )
        summary_flags = dict(max_active_clusters=num_sm, log_gate=log_gate, safe_gate=safe_gate, d_k=DK, d_v=DV, expand_num=expand_num, k_slots=k_slots)
        bprop_summary_cfg = gdn_bprop_summary_f16.build_cfg(io_dtype, use_dstate_in=dstate_in is not None, dstate_table=dstate_in is not None, **summary_flags)
        transition_cfg = None
        if transition_m is not None:
            transition_cfg = gdn_recompute_f16.build_cfg(
                io_dtype,
                get_dtype(transition_m.dtype),
                max_active_clusters=num_sm,
                use_initial_state=False,
                store_final_state=True,
                enable_checkpoints=False,
                seed_checkpoints=False,
                log_gate=log_gate,
                safe_gate=safe_gate,
                seed_identity=True,
                v_is_zero=True,
                d_k=DK,
                d_v=DK,
                expand_num=expand_num,
                k_slots=k_slots,
                v_slots=k_slots,
            )
        gate_table_placeholder = None
        dt_bias_table_placeholder = None
        staging_placeholder = None
        item_scratch_placeholder = None
        chunk_scratch_placeholder = None
        if split:
            gate_table_placeholder = from_dlpack(gate, assumed_align=8 if facts.gate_elem_bytes == 2 else 4).mark_layout_dynamic(
                leading_dim=len(gate.shape) - 1
            )
            staging_placeholder = from_dlpack(item_scratch, assumed_align=16)
            staging_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            item_scratch_placeholder = from_dlpack(item_scratch, assumed_align=4)
            item_scratch_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            chunk_scratch_placeholder = from_dlpack(chunk_scratch, assumed_align=4)
            chunk_scratch_placeholder.mark_layout_dynamic(leading_dim=1)
        if dt_bias is not None:
            dt_bias_table_placeholder = from_dlpack(dt_bias, assumed_align=4)
            dt_bias_table_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(len(dt_bias.shape))), divisibility=1)
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_table_placeholder = from_dlpack(work_items, assumed_align=4)
        work_items_table_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_table_placeholder = from_dlpack(work_count, assumed_align=4)
        work_count_table_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0,), divisibility=1)
        tail_constexprs = ()
        tail_placeholders = ()
        if tail is not None:
            buffers = dict(zip(SUMMARY_TAIL_BUFFERS, tail_buffers))
            tail_transition_cfg = gdn_recompute_f16.build_cfg(
                io_dtype,
                get_dtype(buffers["state_m_main"].dtype),
                max_active_clusters=num_sm,
                use_initial_state=False,
                store_final_state=True,
                enable_checkpoints=False,
                seed_checkpoints=False,
                log_gate=log_gate,
                safe_gate=safe_gate,
                seed_identity=True,
                v_is_zero=True,
                d_k=DK,
                d_v=DK,
                expand_num=expand_num,
                k_slots=k_slots,
                v_slots=k_slots,
            )
            tail_bprop_summary_cfg = gdn_bprop_summary_f16.build_cfg(io_dtype, use_dstate_in=False, **summary_flags)
            tail_items = from_dlpack(buffers["work_items"], assumed_align=16)
            tail_items.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_state_m = from_dlpack(buffers["state_m_main"], assumed_align=16)
            tail_state_m.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
            tail_constexprs = (int(tail.unit_chunks), int(tail_chain_rows), tail_transition_cfg, tail_bprop_summary_cfg)
            tail_placeholders = [
                *(cutlass.Int32(int(value)) for value in tail_scalars),
                from_dlpack(buffers["cu_pieces"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["main_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["summary_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["main_count"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["summary_count"], assumed_align=4).mark_layout_dynamic(),
                tail_items,
                from_dlpack(buffers["scheduler_all"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_m"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_summary"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["recompute_m_words"], assumed_align=128).mark_layout_dynamic(),
                from_dlpack(buffers["bprop_summary_words"], assumed_align=128).mark_layout_dynamic(),
                tail_state_m,
                from_dlpack(DeviceView(256, (1, HO, DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(buffers["state_g"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                (
                    from_dlpack(DeviceView(256, (1, HO, DV, DK), dtype_name(buffers["dstate_in"].dtype), int(device)), assumed_align=4).mark_layout_dynamic(
                        leading_dim=3
                    )
                    if buffers["dstate_in"] is not None
                    else None
                ),
                from_dlpack(DeviceView(256, (1, HO, DV, DK), dtype_name(buffers["dstate0"].dtype), int(device)), assumed_align=4).mark_layout_dynamic(
                    leading_dim=3
                ),
            ]
        warmup_summary_backward_cache[key] = cute.compile(
            warmup_summary_backward_host,
            facts.split,
            facts.b_t,
            facts.log_gate,
            facts.safe_gate,
            facts.overhead_chunks,
            facts.expand_num,
            facts.warmup_cap,
            cutlass.Int32(facts.n_heads_out),
            facts.num_sms,
            io_dtype,
            not split,
            tinv_cfg,
            bprop_summary_cfg,
            transition_cfg,
            int(transition_rows),
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(facts.ideal_chunks),
            cutlass.Int32(facts.batch_size),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_walk_ctas),
            float(scale),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=len(k.shape) - 1),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            gate_table_placeholder,
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=len(beta.shape) - 1),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            dt_bias_table_placeholder,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(dstate_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate_in is not None else None,
            work_items_placeholder,
            work_items_table_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            work_count_table_placeholder,
            staging_placeholder,
            item_scratch_placeholder,
            chunk_scratch_placeholder,
            from_dlpack(scheduler, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(tinv, assumed_align=128).mark_layout_dynamic(leading_dim=3),
            from_dlpack(tinv_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(tinv_rows, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(tinv_row_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(transition_m, assumed_align=16).mark_layout_dynamic(leading_dim=3) if transition_m is not None else None,
            (
                from_dlpack(DeviceView(256, (1, HO, DK, DK), dtype_name(transition.dtype), int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3)
                if transition is not None
                else None
            ),
            from_dlpack(scheduler_transition, assumed_align=4).mark_layout_dynamic() if scheduler_transition is not None else None,
            from_dlpack(transition_words, assumed_align=128).mark_layout_dynamic() if transition_words is not None else None,
            cuda.CUstream(int(stream)),
            k_slots,
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            l2norm_k_slots,
            l2norm_k_expand_num,
            *(
                from_dlpack(buffer, assumed_align=16).mark_layout_dynamic(leading_dim=len(buffer.shape) - 1) if buffer is not None else None
                for buffer in l2norm_buffers
            ),
            *tail_constexprs,
            *tail_placeholders,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return warmup_summary_backward_cache[key], facts


def run_warmup_summary_backward(
    compiled,
    facts,
    *,
    q,
    k,
    do,
    gate,
    beta,
    a_log,
    dt_bias,
    cu_seqlens,
    dstate0,
    dstate_in,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    scheduler_all,
    workspace,
    tinv,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    transition_m,
    transition,
    scheduler_transition,
    transition_words,
    scale,
    stream,
    item_limit_word=None,
    l2norm_buffers=(None,) * 4,
    tail_scalars=(),
    tail_buffers=(),
) -> None:
    """Replay the backward summary launch: one crossing into the DSL."""
    compiled(
        facts.n_heads_out,
        facts.n_tiles,
        facts.ideal_chunks,
        facts.batch_size,
        facts.log2_threshold,
        facts.n_scan_ctas,
        facts.n_walk_ctas,
        float(scale),
        q,
        k,
        do,
        gate,
        gate if facts.split else None,
        beta,
        a_log if facts.safe_gate else None,
        a_log if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        cu_seqlens,
        cu_seqlens,
        cu_seqlens,
        dstate0,
        dstate_in,
        work_items,
        work_items,
        work_count,
        work_count,
        item_scratch if facts.split else None,
        item_scratch if facts.split else None,
        chunk_scratch if facts.split else None,
        scheduler,
        scheduler_all,
        workspace,
        tinv,
        tinv_words,
        tinv_rows,
        tinv_row_count,
        transition_m,
        transition,
        scheduler_transition,
        transition_words,
        cuda.CUstream(int(stream)),
        item_limit_word,
        *l2norm_buffers,
        *tail_scalars,
        *tail_buffers,
    )
