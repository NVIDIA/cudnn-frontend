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

"""One compiled launch for the GDN-2 uncut backward: the recompute prologue and the checkpoint-series recompute (unless
the forward's per-chunk series is passed back), the bprop prologue and the bprop, issued from a single host at
``--opt-level 2``, the level of every GDN-2 module and its prologues, so every nested kernel is the standalone one.  Every
kernel, its host and the tensor placeholder each host was compiled with are the standalone modules' own; a buffer two
hosts read through different placeholder types is passed twice (``beta`` at 16 bytes in the prologues and 4 in the
recompute and the bprop's guarded read; the gate when the bprop reads it as a linear alpha)."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cutlass.experimental.cuda.tensor_map as tma
from cudnn.frost.buffers import DeviceView
from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids

from ..common import gate_bwd, head_reduce
from ..common.host import get_dtype
from ..common.piece_chain import chain_rows_per_cta, dtype_name, launch_state_chain
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, gen_interval_items, order_body
from ..common.tail import gate_body
from ..common.state_snapshot import snapshot_states
from . import gdn2_bprop_f16, gdn2_bprop_summary_f16, gdn2_chain_backward_f16, gdn2_recompute_f16, gdn2_summary_f16

USE_PDL = True

TAIL_BUFFERS = (
    "beta_recompute",
    "cu_pieces",
    "cu_pieces_main",
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
    "summary_words",
    "recompute_m_words",
    "series_words",
    "bprop_summary_words",
    "bprop_words",
    "state_h_summary",
    "state_m_main",
    "state_h_chain",
    "state_m_chain",
    "state_x_chain",
    "seed",
    "state_x_series",
    "state_g",
    "state_g_chain",
    "state_dx_end_chain",
    "state_dx_end",
    "dseed",
    "dcomposed",
    "dstate0",
    "seed_row_base",
)

uncut_backward_cache = {}


@cute.kernel
def frost_gdn2_regenerate_prologue(
    recompute_orders: cutlass.Constexpr[bool],
    coarse: cutlass.Constexpr[bool],
    bwd_orders: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    recompute_base_k: cutlass.GridConstant[tma.TensorMap],
    recompute_base_v: cutlass.GridConstant[tma.TensorMap],
    recompute_base_gate: cutlass.GridConstant[tma.TensorMap],
    recompute_base_beta: cutlass.GridConstant[tma.TensorMap],
    recompute_base_w: cutlass.GridConstant[tma.TensorMap],
    recompute_base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_gate: cutlass.GridConstant[tma.TensorMap],
    base_do: cutlass.GridConstant[tma.TensorMap],
    base_beta: cutlass.GridConstant[tma.TensorMap],
    base_w: cutlass.GridConstant[tma.TensorMap],
    base_dq: cutlass.GridConstant[tma.TensorMap],
    base_dk: cutlass.GridConstant[tma.TensorMap],
    base_dv: cutlass.GridConstant[tma.TensorMap],
    base_dgate: cutlass.GridConstant[tma.TensorMap],
    base_dwo: cutlass.GridConstant[tma.TensorMap],
    base_dbo: cutlass.GridConstant[tma.TensorMap],
    base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    recompute_workspace: cute.Tensor,
    bprop_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dgate: cute.Tensor,
    dw: cute.Tensor,
    dbeta: cute.Tensor,
    checkpoints: cute.Tensor,
    series_count: cute.Tensor,
    series_items: cute.Tensor,
    scheduler_recompute: Optional[cute.Tensor],
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_bprop: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    seed_span_chunks: cutlass.Int32,
    mItemLimit: Optional[cute.Tensor] = None,
    gate_sms: cutlass.Constexpr[int] = 0,
) -> None:
    """Four-CTA prologue of the regenerate path, the recompute's and the bprop's prologues in one launch: block 0 the
    recompute's item phase (the interval items under ``coarse``, the uncut table under ``recompute_orders``), block 1
    the bprop's (under ``bwd_orders``), block 2 the recompute's per-batch TMA-descriptor arrays, block 3 the bprop's,
    one warp per array."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    n_heads_out = cutlass.Int32(gate.shape[1])
    n_items = n_heads_out * n_batch
    if cutlass.const_expr(mItemLimit is not None):
        if cutlass.const_expr(gate_sms > 0):
            gate_body(ORDER_THREADS, tidx, n_heads_out, gate_sms, cu_seqlens, mItemLimit)
        n_items = cutlass.Int32(mItemLimit[1])
    if bidx == cutlass.Int32(0):
        if cutlass.const_expr(coarse):
            gen_interval_items(b_t, ORDER_THREADS, tidx, n_heads_out, n_items, seed_span_chunks, cu_seqlens, series_count, series_items, scheduler_recompute)
        if cutlass.const_expr(recompute_orders):
            sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
            order_body(
                True,
                b_t,
                ORDER_THREADS,
                ORDER_ELEMENTS,
                tidx,
                n_heads_out,
                n_items,
                cu_seqlens,
                None,
                series_count,
                series_items,
                scheduler_recompute,
                sKey,
                sIdx,
                sSpread,
            )
    elif bidx == cutlass.Int32(1):
        if cutlass.const_expr(bwd_orders):
            sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
            order_body(
                True,
                b_t,
                ORDER_THREADS,
                ORDER_ELEMENTS,
                tidx,
                n_heads_out,
                n_items,
                cu_seqlens,
                None,
                work_count,
                work_items,
                scheduler_bprop,
                sKey,
                sIdx,
                sSpread,
            )
    elif bidx == cutlass.Int32(2):
        gdn2_recompute_f16.build_descs_body(
            widx,
            recompute_base_k,
            recompute_base_v,
            recompute_base_gate,
            recompute_base_beta,
            recompute_base_w,
            recompute_base_checkpoint,
            recompute_workspace,
            cu_seqlens,
            k,
            v,
            gate,
            beta,
            w,
            checkpoints,
            n_batch,
            checkpoint_every_n,
        )
    else:
        gdn2_bprop_f16.build_descs_body(
            widx,
            base_q,
            base_k,
            base_v,
            base_gate,
            base_do,
            base_beta,
            base_w,
            base_dq,
            base_dk,
            base_dv,
            base_dgate,
            base_dwo,
            base_dbo,
            base_checkpoint,
            bprop_workspace,
            cu_seqlens,
            q,
            k,
            v,
            gate,
            do,
            beta,
            w,
            dq,
            dk,
            dv,
            dgate,
            dw,
            dbeta,
            checkpoints,
            n_batch,
            cutlass.Int32(b_t),
        )


@cute.jit
def uncut_backward_host(
    b_t: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    recompute: cutlass.Constexpr[bool],
    recompute_orders: cutlass.Constexpr[bool],
    coarse: cutlass.Constexpr[bool],
    bwd_orders: cutlass.Constexpr[bool],
    recompute_cfg: cutlass.Constexpr,
    bprop_cfg: cutlass.Constexpr,
    checkpoint_every_n: cutlass.Int32,
    seed_span_chunks: cutlass.Int32,
    seed_every_n: cutlass.Int32,
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    gate: cute.Tensor,
    gate_main: Optional[cute.Tensor],
    beta: cute.Tensor,
    beta_recompute: Optional[cute.Tensor],
    beta_bprop: Optional[cute.Tensor],
    w: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    checkpoints: cute.Tensor,
    seed_checkpoints: Optional[cute.Tensor],
    state_in: Optional[cute.Tensor],
    composed: Optional[cute.Tensor],
    dgate: cute.Tensor,
    dw: cute.Tensor,
    dbeta: cute.Tensor,
    dstate0: Optional[cute.Tensor],
    dstate_in: Optional[cute.Tensor],
    dcomposed: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    series_items: Optional[cute.Tensor],
    series_count: Optional[cute.Tensor],
    scheduler_all: cute.Tensor,
    scheduler_all_recompute: Optional[cute.Tensor],
    scheduler_all_bprop: Optional[cute.Tensor],
    scheduler_recompute: cute.Tensor,
    scheduler_bwd: cute.Tensor,
    recompute_words: Optional[cute.Tensor],
    bprop_words: cute.Tensor,
    dstate_scratch: cute.Tensor,
    dstate_snapshot: Optional[cute.Tensor],
    stream: cuda.CUstream,
    item_limit_word: Optional[cute.Tensor] = None,
    fold_rows: cutlass.Constexpr[int] = 0,
    fold_update: Optional[cute.Tensor] = None,
    fold_transition: Optional[cute.Tensor] = None,
    fold_seed: Optional[cute.Tensor] = None,
    fold_composed: Optional[cute.Tensor] = None,
    gate_lower_bound: cutlass.Constexpr[float] = 0.0,
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
    tail_unit_chunks: cutlass.Constexpr[int] = 0,
    tail_chain_rows: cutlass.Constexpr[int] = 0,
    tail_summary_cfg: cutlass.Constexpr = None,
    tail_transition_cfg: cutlass.Constexpr = None,
    tail_bwd_summary_cfg: cutlass.Constexpr = None,
    tail_series_cfg: cutlass.Constexpr = None,
    tail_bprop_cfg: cutlass.Constexpr = None,
    tail_pieces: cutlass.Int32 = 0,
    tail_num_seqs: cutlass.Int32 = 0,
    tail_series_span_chunks: cutlass.Int32 = 0,
    tail_batch_base: cutlass.Int32 = 0,
    tail_first_head_lo: cutlass.Int32 = 0,
    tail_beta_recompute: Optional[cute.Tensor] = None,
    tail_cu_pieces: Optional[cute.Tensor] = None,
    tail_cu_pieces_main: Optional[cute.Tensor] = None,
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
    tail_summary_words: Optional[cute.Tensor] = None,
    tail_recompute_m_words: Optional[cute.Tensor] = None,
    tail_series_words: Optional[cute.Tensor] = None,
    tail_bprop_summary_words: Optional[cute.Tensor] = None,
    tail_bprop_words: Optional[cute.Tensor] = None,
    tail_state_h_summary: Optional[cute.Tensor] = None,
    tail_state_m_main: Optional[cute.Tensor] = None,
    tail_state_h_chain: Optional[cute.Tensor] = None,
    tail_state_m_chain: Optional[cute.Tensor] = None,
    tail_state_x_chain: Optional[cute.Tensor] = None,
    tail_seed: Optional[cute.Tensor] = None,
    tail_state_x_series: Optional[cute.Tensor] = None,
    tail_state_g: Optional[cute.Tensor] = None,
    tail_state_g_chain: Optional[cute.Tensor] = None,
    tail_state_dx_end_chain: Optional[cute.Tensor] = None,
    tail_state_dx_end: Optional[cute.Tensor] = None,
    tail_dseed: Optional[cute.Tensor] = None,
    tail_dcomposed: Optional[cute.Tensor] = None,
    tail_dstate0: Optional[cute.Tensor] = None,
    tail_seed_row_base: Optional[cute.Tensor] = None,
) -> None:
    if cutlass.const_expr(fold_update is not None):
        if cutlass.const_expr(fold_seed is not None):
            fold_seed = cute.make_tensor(
                fold_seed.iterator + (cutlass.Int32(fold_seed.shape[0]) - cutlass.Int32(1)) * fold_seed.stride[0],
                cute.make_layout((fold_composed.shape[0], fold_seed.shape[1], fold_seed.shape[2], fold_seed.shape[3]), stride=fold_seed.stride),
            )
        launch_state_chain(
            cutlass.Int32(gate.shape[1]),
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
    gate_sms = bprop_cfg.max_active_clusters if cutlass.const_expr(item_limit_word is not None) else 0
    dstate_seed = dstate_in
    if cutlass.const_expr(dstate_snapshot is not None):
        # d_initial_state overwrites the d_final_state the bprop's exact twin would reread: read it from a copy.
        snapshot_states(dstate_in, dstate_snapshot, None, stream)
        dstate_seed = dstate_snapshot
    heads_out = cutlass.Int32(gate.shape[1])
    q_ratio = heads_out // cutlass.Int32(q.shape[1])
    k_ratio = heads_out // cutlass.Int32(k.shape[1])
    v_ratio = heads_out // cutlass.Int32(v.shape[1])
    if cutlass.const_expr(recompute):
        recompute_base_k, recompute_base_v, recompute_base_gate, recompute_base_beta, recompute_base_w, recompute_base_checkpoint = (
            gdn2_recompute_f16.base_descriptors(io_dtype, b_t, k, v, gate, beta, w, checkpoints)
        )
        base_q, base_k, base_v, base_gate, base_do, base_beta, base_w, base_dq, base_dk, base_dv, base_dgate, base_dwo, base_dbo, base_checkpoint = (
            gdn2_bprop_f16.base_descriptors(io_dtype, b_t, q, k, v, gate, do, beta, w, dq, dk, dv, dgate, dw, dbeta, checkpoints)
        )
        frost_gdn2_regenerate_prologue(
            recompute_orders,
            coarse,
            bwd_orders,
            b_t,
            recompute_base_k,
            recompute_base_v,
            recompute_base_gate,
            recompute_base_beta,
            recompute_base_w,
            recompute_base_checkpoint,
            base_q,
            base_k,
            base_v,
            base_gate,
            base_do,
            base_beta,
            base_w,
            base_dq,
            base_dk,
            base_dv,
            base_dgate,
            base_dwo,
            base_dbo,
            base_checkpoint,
            recompute_words,
            bprop_words,
            cu_seqlens,
            q,
            k,
            v,
            gate,
            do,
            beta,
            w,
            dq,
            dk,
            dv,
            dgate,
            dw,
            dbeta,
            checkpoints,
            series_count,
            series_items,
            scheduler_all_recompute,
            work_count,
            work_items,
            scheduler_all_bprop,
            cutlass.Int32(cu_seqlens.shape[0] - 1),
            checkpoint_every_n,
            seed_span_chunks,
            item_limit_word,
            gate_sms,
        ).launch(grid=(4, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)
        gdn2_recompute_f16.host(
            recompute_cfg,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            beta_recompute,
            w,
            cu_seqlens,
            state_in,
            composed,
            None,
            seed_checkpoints,
            series_items,
            series_count,
            scheduler_recompute,
            recompute_words,
            checkpoint_every_n,
            seed_every_n,
            stream,
        )
    else:
        gdn2_bprop_f16.prologue(
            io_dtype,
            b_t,
            bwd_orders,
            q,
            k,
            v,
            gate,
            do,
            beta,
            w,
            dq,
            dk,
            dv,
            dgate,
            dw,
            dbeta,
            checkpoints,
            cu_seqlens,
            work_count,
            work_items,
            scheduler_all_bprop,
            bprop_words,
            stream,
            item_limit_word=item_limit_word,
            gate_sms=gate_sms,
        )
    gdn2_bprop_f16.host(
        bprop_cfg,
        q_ratio,
        k_ratio,
        v_ratio,
        checkpoints,
        a_log,
        dt_bias,
        beta_bprop,
        gate_main,
        dgate,
        dbeta,
        dw,
        cu_seqlens,
        dstate0,
        dstate_seed,
        dcomposed,
        work_items,
        work_count,
        scheduler_bwd,
        bprop_words,
        dstate_scratch,
        scale,
        stream,
    )
    if cutlass.const_expr(tail_bprop_cfg is not None):
        gdn2_chain_backward_f16.chain_backward_host(
            tail_unit_chunks,
            b_t,
            False,
            tail_summary_cfg is not None,
            tail_series_cfg is not None,
            False,
            False,
            tail_summary_cfg,
            tail_transition_cfg,
            tail_bwd_summary_cfg,
            tail_series_cfg,
            tail_bprop_cfg,
            tail_bprop_cfg.d_v,
            tail_bprop_cfg.d_k,
            tail_chain_rows,
            tail_seed is not None,
            tail_dseed is not None,
            True,
            tail_pieces,
            heads_out,
            tail_num_seqs,
            tail_series_span_chunks,
            checkpoint_every_n,
            seed_every_n,
            scale,
            tail_batch_base,
            tail_first_head_lo,
            q,
            k,
            v,
            do,
            gate,
            gate_main,
            beta,
            tail_beta_recompute,
            beta_bprop,
            w,
            a_log,
            dt_bias,
            cu_seqlens,
            tail_cu_pieces,
            tail_cu_pieces_main,
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
            tail_scheduler_summary,
            tail_scheduler_bwd,
            tail_summary_words,
            tail_recompute_m_words,
            tail_series_words,
            tail_bprop_summary_words,
            tail_bprop_words,
            dstate_scratch,
            checkpoints,
            seed_checkpoints,
            dq,
            dk,
            dv,
            dgate,
            dw,
            dbeta,
            tail_state_h_summary,
            tail_state_m_main,
            tail_state_h_chain,
            tail_state_m_chain,
            tail_state_x_chain,
            tail_seed,
            None,
            tail_state_x_series,
            tail_state_g,
            tail_state_g_chain,
            tail_state_dx_end_chain,
            tail_state_dx_end,
            tail_dseed,
            tail_dcomposed,
            tail_dstate0,
            tail_seed_row_base,
            stream,
            item_limit_word,
        )
    if cutlass.const_expr(gate_bwd_dgate is not None):
        gate_tokens = cutlass.Int32(gate_bwd_dgate.shape[0])
        gate_bwd.channel_gate_bwd_launch(
            bprop_cfg.d_k,
            gate_bwd_dgate,
            gate_bwd_gate,
            gate_bwd_a_log,
            gate_bwd_dt_bias,
            gate_part_a,
            gate_part_dt,
            d_a_log,
            d_dt_bias,
            gate_tokens,
            cutlass.Int32(gate_bwd_dgate.shape[1]),
            (gate_tokens + cutlass.Int32(gate_bwd.GATE_BWD_BLOCKS - 1)) // cutlass.Int32(gate_bwd.GATE_BWD_BLOCKS),
            cutlass.Float32(gate_lower_bound),
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


def build_uncut_backward(
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
    w,
    a_log,
    dt_bias,
    cu_seqlens,
    checkpoints,
    seed_checkpoints,
    state_in,
    composed=None,
    use_initial_state,
    dgate,
    dw,
    dbeta,
    dstate0,
    dstate_in,
    dcomposed=None,
    work_items,
    work_count,
    series_items,
    series_count,
    scheduler_all,
    scheduler_recompute,
    scheduler_bwd,
    recompute_words,
    bprop_words,
    dstate_scratch,
    num_sm,
    b_t,
    recompute,
    recompute_orders,
    coarse,
    bwd_orders,
    seed_span_tokens,
    seed_every_n_tokens,
    log_gate,
    safe_gate,
    gate_lower_bound,
    use_qk_l2norm,
    use_beta_sigmoid,
    allow_neg_eigval,
    beta_guard,
    scale,
    device,
    stream,
    dstate_snapshot=None,
    item_limit_word=None,
    tail=None,
    tail_chain_rows=0,
    tail_scalars=(),
    tail_buffers=(),
    lead_buffers=(None,) * 4,
    post_buffers=(None,) * 14,
):
    """Compile (cached per static config) the uncut backward launch over the buffers of one plan.  The placeholders
    repeat the marks of the standalone builds so every kernel compiles as it does there; the recompute or the bprop
    prologue orders the uncut table (``recompute_orders`` / ``bwd_orders``)."""
    _HQ, DK = q.shape[1], q.shape[2]
    k.shape[1]
    _HV, DV = v.shape[1], v.shape[2]
    gate.shape[1]
    if not safe_gate:
        a_log = None
        dt_bias = None
    io_dtype = get_dtype(q.dtype)
    gate_dtype = get_dtype(gate.dtype)
    gate_scale_log2 = float(gate_lower_bound) * gdn2_summary_f16.LOG2_E
    gate_main = not log_gate and not safe_gate
    beta_main = beta_guard and use_beta_sigmoid
    fold_update, fold_transition, fold_seed, fold_composed = lead_buffers
    fold_rows = chain_rows_per_cta(DV, DK, 1, gate.shape[1], num_sm) if fold_update is not None else 0
    key = (
        str(q.dtype),
        str(seed_checkpoints.dtype) if coarse else "none",
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(beta.dtype),
        str(w.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(state_in.dtype) if state_in is not None else "none",
        str(dstate0.dtype) if dstate0 is not None else "none",
        str(dstate_in.dtype) if dstate_in is not None else "none",
        composed is not None,
        dcomposed is not None,
        int(device),
        int(num_sm),
        DK,
        DV,
        int(b_t),
        bool(recompute),
        bool(recompute_orders),
        bool(coarse),
        bool(bwd_orders),
        bool(use_initial_state),
        bool(log_gate),
        bool(safe_gate),
        float(gate_lower_bound),
        bool(use_qk_l2norm),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        bool(beta_guard),
        dstate_snapshot is not None,
        item_limit_word is not None,
        (
            (int(tail.unit_chunks), int(tail_chain_rows), *(str(buffer.dtype) if buffer is not None else None for buffer in tail_buffers))
            if tail is not None
            else None
        ),
        tuple(str(buffer.dtype) if buffer is not None else None for buffer in (*lead_buffers, *post_buffers)),
        fold_rows,
    )
    if key not in uncut_backward_cache:
        flags = dict(
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=gate_scale_log2,
            log_gate=log_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            beta_guard=beta_guard,
            max_active_clusters=num_sm,
            d_k=DK,
        )
        recompute_cfg = None
        if recompute:
            recompute_cfg = gdn2_recompute_f16.build_cfg(
                io_dtype,
                get_dtype(state_in.dtype) if state_in is not None else cutlass.Float32,
                gate_dtype,
                use_initial_state=state_in is not None or composed is not None,
                state_table=state_in is not None,
                has_composed=composed is not None,
                store_final_state=False,
                enable_checkpoints=True,
                seed_checkpoints=coarse,
                d_v=DV,
                **flags,
            )
        bprop_cfg = gdn2_bprop_f16.build_cfg(
            io_dtype,
            gate_dtype,
            use_dstate_in=dstate_in is not None or dcomposed is not None,
            dstate_table=dstate_in is not None,
            has_composed_dstate=dcomposed is not None,
            use_dstate0=dstate0 is not None,
            use_initial_state=use_initial_state,
            d_v=DV,
            **flags,
        )
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        series_items_placeholder = None
        if recompute:
            series_items_placeholder = from_dlpack(series_items, assumed_align=16)
            series_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        tail_constexprs = ()
        tail_placeholders = ()
        if tail is not None:
            buffers = dict(zip(TAIL_BUFFERS, tail_buffers))
            HO = gate.shape[1]
            tail_summary_cfg, tail_transition_cfg, tail_series_cfg = None, None, None
            if buffers["state_h_summary"] is not None:
                tail_summary_cfg = gdn2_summary_f16.build_cfg(io_dtype, gate_dtype, use_initial_state=False, d_v=DV, **flags)
            else:
                tail_transition_cfg = gdn2_recompute_f16.build_cfg(
                    io_dtype,
                    get_dtype(buffers["state_m_main"].dtype),
                    gate_dtype,
                    use_initial_state=False,
                    store_final_state=True,
                    enable_checkpoints=False,
                    seed_checkpoints=False,
                    seed_identity=True,
                    v_is_zero=True,
                    d_v=DK,
                    **flags,
                )
            if buffers["recompute_items"] is not None:
                tail_series_cfg = gdn2_recompute_f16.build_cfg(
                    io_dtype,
                    get_dtype(buffers["state_x_series"].dtype) if buffers["state_x_series"] is not None else cutlass.Float32,
                    gate_dtype,
                    use_initial_state=not coarse,
                    store_final_state=False,
                    enable_checkpoints=True,
                    seed_checkpoints=coarse,
                    d_v=DV,
                    **flags,
                )
            tail_bwd_summary_cfg = gdn2_bprop_summary_f16.build_cfg(io_dtype, gate_dtype, use_dstate_in=False, use_dstate0=True, d_v=do.shape[2], **flags)
            tail_bprop_cfg = gdn2_bprop_f16.build_cfg(
                io_dtype, gate_dtype, use_dstate_in=True, use_dstate0=buffers["dstate0"] is not None, use_initial_state=True, d_v=DV, **flags
            )
            tail_constexprs = (
                int(tail.unit_chunks),
                int(tail_chain_rows),
                tail_summary_cfg,
                tail_transition_cfg,
                tail_bwd_summary_cfg,
                tail_series_cfg,
                tail_bprop_cfg,
            )
            tail_items = {}
            for name in ("work_items", "work_items_summary", "series_items"):
                tail_items[name] = None
                if buffers[name] is not None:
                    tail_items[name] = from_dlpack(buffers[name], assumed_align=16)
                    tail_items[name].mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            chain_states = [
                from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3) for _ in range(5)
            ]
            tail_placeholders = [
                *(cutlass.Int32(int(value)) for value in tail_scalars),
                from_dlpack(buffers["beta_recompute"], assumed_align=4).mark_layout_dynamic(leading_dim=2),
                from_dlpack(buffers["cu_pieces"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["cu_pieces_main"], assumed_align=8).mark_layout_dynamic(),
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
                *(
                    from_dlpack(buffers[name], assumed_align=128).mark_layout_dynamic() if buffers[name] is not None else None
                    for name in ("summary_words", "recompute_m_words", "series_words", "bprop_summary_words", "bprop_words")
                ),
                (
                    from_dlpack(buffers["state_h_summary"], assumed_align=16).mark_layout_dynamic(leading_dim=3)
                    if buffers["state_h_summary"] is not None
                    else None
                ),
                from_dlpack(buffers["state_m_main"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
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
                from_dlpack(buffers["state_x_series"], assumed_align=16).mark_layout_dynamic(leading_dim=3) if buffers["state_x_series"] is not None else None,
                from_dlpack(buffers["state_g"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                chain_states[2],
                chain_states[3],
                from_dlpack(buffers["state_dx_end"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                (
                    from_dlpack(DeviceView(256, (1, HO, DV, DK), dtype_name(buffers["dseed"].dtype), int(device)), assumed_align=4).mark_layout_dynamic(
                        leading_dim=3
                    )
                    if buffers["dseed"] is not None
                    else None
                ),
                chain_states[4] if buffers["dcomposed"] is not None else None,
                from_dlpack(buffers["dstate0"], assumed_align=16).mark_layout_dynamic(leading_dim=3) if buffers["dstate0"] is not None else None,
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
        gate_align = 4 * (gate_dtype.width // 8)
        post_placeholders = [
            from_dlpack(buffer, assumed_align=gate_align).mark_layout_dynamic(leading_dim=2) if buffer is not None else None for buffer in post_buffers[:2]
        ] + [
            from_dlpack(buffer, assumed_align=4).mark_layout_dynamic(leading_dim=len(buffer.shape) - 1) if buffer is not None else None
            for buffer in post_buffers[2:]
        ]
        uncut_backward_cache[key] = cute.compile(
            uncut_backward_host,
            int(b_t),
            io_dtype,
            bool(recompute),
            bool(recompute_orders),
            bool(coarse),
            bool(bwd_orders),
            recompute_cfg,
            bprop_cfg,
            cutlass.Int32(int(b_t)),
            cutlass.Int32(int(seed_span_tokens or seed_every_n_tokens) // int(b_t)),
            cutlass.Int32(int(seed_every_n_tokens)),
            float(scale),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dq, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dk, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dv, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=4).mark_layout_dynamic(leading_dim=2) if gate_main else None,
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=2) if recompute else None,
            from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=2) if beta_main else None,
            from_dlpack(w, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(seed_checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if coarse else None,
            from_dlpack(state_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if state_in is not None else None,
            from_dlpack(composed, assumed_align=16).mark_layout_dynamic(leading_dim=3) if composed is not None else None,
            from_dlpack(dgate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dw, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dbeta, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate0 is not None else None,
            from_dlpack(dstate_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate_in is not None else None,
            from_dlpack(dcomposed, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dcomposed is not None else None,
            work_items_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            series_items_placeholder,
            from_dlpack(series_count, assumed_align=4).mark_layout_dynamic() if recompute else None,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic() if recompute and (recompute_orders or coarse) else None,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic() if bwd_orders else None,
            from_dlpack(scheduler_recompute, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_bwd, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(recompute_words, assumed_align=128).mark_layout_dynamic() if recompute else None,
            from_dlpack(bprop_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(dstate_scratch, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(dstate_snapshot, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate_snapshot is not None else None,
            cuda.CUstream(int(stream)),
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            fold_rows,
            *fold_placeholders,
            float(gate_lower_bound),
            *post_placeholders,
            *tail_constexprs,
            *tail_placeholders,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return uncut_backward_cache[key]


def run_uncut_backward(
    compiled,
    *,
    log_gate,
    safe_gate,
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    gate,
    beta,
    w,
    a_log,
    dt_bias,
    cu_seqlens,
    checkpoints,
    seed_checkpoints,
    state_in,
    composed=None,
    dgate,
    dw,
    dbeta,
    dstate0,
    dstate_in,
    dcomposed=None,
    work_items,
    work_count,
    series_items,
    series_count,
    scheduler_all,
    scheduler_recompute,
    scheduler_bwd,
    recompute_words,
    bprop_words,
    dstate_scratch,
    b_t,
    recompute,
    recompute_orders,
    coarse,
    bwd_orders,
    seed_span_tokens,
    seed_every_n_tokens,
    use_beta_sigmoid,
    beta_guard,
    scale,
    stream,
    dstate_snapshot=None,
    item_limit_word=None,
    tail_scalars=(),
    tail_buffers=(),
    lead_buffers=(None,) * 4,
    post_buffers=(None,) * 14,
) -> None:
    """Replay the uncut backward: one crossing into the DSL.  The plan validated the contract at build, so nothing here
    raises."""
    compiled(
        int(b_t),
        int(seed_span_tokens or seed_every_n_tokens) // int(b_t),
        int(seed_every_n_tokens),
        float(scale),
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        gate,
        gate if not log_gate and not safe_gate else None,
        beta,
        beta if recompute else None,
        beta if beta_guard and use_beta_sigmoid else None,
        w,
        a_log,
        dt_bias,
        cu_seqlens,
        checkpoints,
        seed_checkpoints if coarse else None,
        state_in,
        composed,
        dgate,
        dw,
        dbeta,
        dstate0,
        dstate_in,
        dcomposed,
        work_items,
        work_count,
        series_items if recompute else None,
        series_count if recompute else None,
        scheduler_all,
        scheduler_all if recompute and (recompute_orders or coarse) else None,
        scheduler_all if bwd_orders else None,
        scheduler_recompute,
        scheduler_bwd,
        recompute_words if recompute else None,
        bprop_words,
        dstate_scratch,
        dstate_snapshot,
        cuda.CUstream(int(stream)),
        item_limit_word,
        *lead_buffers,
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

uncut_summary_backward_cache = {}


@cute.jit
def uncut_summary_backward_host(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    bprop_summary_cfg: cutlass.Constexpr,
    transition_cfg: cutlass.Constexpr,
    transition_rows: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    do: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    beta_recompute: Optional[cute.Tensor],
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    dstate0: cute.Tensor,
    dstate_in: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    scheduler: cute.Tensor,
    scheduler_all: cute.Tensor,
    workspace: cute.Tensor,
    dstate_scratch: cute.Tensor,
    transition_m: Optional[cute.Tensor],
    transition_m_chain: Optional[cute.Tensor],
    transition: Optional[cute.Tensor],
    scheduler_transition: Optional[cute.Tensor],
    transition_words: Optional[cute.Tensor],
    stream: cuda.CUstream,
    item_limit_word: Optional[cute.Tensor] = None,
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
    heads_out = cutlass.Int32(gate.shape[1])
    gdn2_bprop_summary_f16.prologue(
        io_dtype,
        b_t,
        True,
        q,
        k,
        gate,
        do,
        beta,
        cu_seqlens,
        work_count,
        work_items,
        scheduler_all,
        workspace,
        stream,
        item_limit_word,
        gate_sms,
    )
    gdn2_bprop_summary_f16.host(
        bprop_summary_cfg,
        heads_out // cutlass.Int32(q.shape[1]),
        heads_out // cutlass.Int32(k.shape[1]),
        a_log,
        dt_bias,
        cu_seqlens,
        dstate0,
        dstate_in,
        None,
        work_items,
        work_count,
        scheduler,
        workspace,
        dstate_scratch,
        scale,
        stream,
    )
    if cutlass.const_expr(transition_m is not None):
        gdn2_recompute_f16.prologue(
            io_dtype,
            b_t,
            True,
            False,
            k,
            k,
            gate,
            beta,
            k,
            None,
            cu_seqlens,
            work_count,
            work_items,
            scheduler_all,
            transition_words,
            cutlass.Int32(0),
            cutlass.Int32(0),
            stream,
            item_limit_word,
        )
        gdn2_recompute_f16.host(
            transition_cfg,
            k,
            k,
            gate,
            a_log,
            dt_bias,
            beta_recompute,
            k,
            cu_seqlens,
            None,
            None,
            transition_m,
            None,
            work_items,
            work_count,
            scheduler_transition,
            transition_words,
            cutlass.Int32(0),
            cutlass.Int32(0),
            stream,
        )
        launch_state_chain(
            heads_out,
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
            cutlass.Int32(cu_seqlens.shape[0] - 1),
            None,
            transition_m_chain,
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
        gdn2_chain_backward_f16.chain_summary_backward_host(
            tail_unit_chunks,
            b_t,
            False,
            tail_transition_cfg,
            tail_bprop_summary_cfg,
            tail_bprop_summary_cfg.d_v,
            tail_bprop_summary_cfg.d_k,
            tail_chain_rows,
            tail_dstate_in is not None,
            transition is not None,
            True,
            tail_pieces,
            heads_out,
            tail_num_seqs,
            scale,
            tail_batch_base,
            tail_first_head_lo,
            q,
            k,
            do,
            gate,
            beta,
            beta_recompute,
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
            tail_recompute_m_words,
            tail_bprop_summary_words,
            dstate_scratch,
            tail_state_m_main,
            tail_state_m_chain,
            tail_state_g,
            tail_state_g_chain,
            tail_state_x_chain,
            tail_dstate_in,
            tail_dstate0,
            transition,
            stream,
            item_limit_word,
        )


def build_uncut_summary_backward(
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
    scheduler,
    scheduler_all,
    workspace,
    dstate_scratch,
    transition_m,
    transition,
    scheduler_transition,
    transition_words,
    transition_rows,
    num_sm,
    b_t,
    log_gate,
    safe_gate,
    gate_lower_bound,
    use_qk_l2norm,
    use_beta_sigmoid,
    allow_neg_eigval,
    beta_guard,
    scale,
    device,
    stream,
    item_limit_word=None,
    tail=None,
    tail_chain_rows=0,
    tail_scalars=(),
    tail_buffers=(),
):
    """Compile (cached per static config) the uncut backward summary launch over the buffers of one plan."""
    DK, DV = q.shape[2], do.shape[2]
    HO = gate.shape[1]
    if not safe_gate:
        a_log = None
        dt_bias = None
    io_dtype = get_dtype(q.dtype)
    gate_dtype = get_dtype(gate.dtype)
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
        bool(log_gate),
        bool(safe_gate),
        float(gate_lower_bound),
        bool(use_qk_l2norm),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        bool(beta_guard),
        item_limit_word is not None,
        (int(tail.unit_chunks), int(tail_chain_rows)) if tail is not None else None,
    )
    if key not in uncut_summary_backward_cache:
        flags = dict(
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=float(gate_lower_bound) * gdn2_summary_f16.LOG2_E,
            log_gate=log_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            beta_guard=beta_guard,
            max_active_clusters=num_sm,
            d_k=DK,
        )
        bprop_summary_cfg = gdn2_bprop_summary_f16.build_cfg(
            io_dtype, gate_dtype, use_dstate_in=dstate_in is not None, dstate_table=dstate_in is not None, use_dstate0=True, d_v=DV, **flags
        )
        transition_cfg = gdn2_recompute_f16.build_cfg(
            io_dtype,
            cutlass.Float32,
            gate_dtype,
            use_initial_state=False,
            store_final_state=True,
            enable_checkpoints=False,
            seed_checkpoints=False,
            seed_identity=True,
            v_is_zero=True,
            d_v=DK,
            **flags,
        )
        transition_placeholders = [None] * 5
        if transition_m is not None:
            transition_placeholders = [
                from_dlpack(transition_m, assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(DeviceView(256, (1, HO, DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(DeviceView(256, (1, HO, DK, DK), dtype_name(transition.dtype), int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(scheduler_transition, assumed_align=4).mark_layout_dynamic(),
                from_dlpack(transition_words, assumed_align=128).mark_layout_dynamic(),
            ]
        tail_constexprs = ()
        tail_placeholders = ()
        if tail is not None:
            buffers = dict(zip(SUMMARY_TAIL_BUFFERS, tail_buffers))
            tail_items = from_dlpack(buffers["work_items"], assumed_align=16)
            tail_items.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_bprop_summary_cfg = gdn2_bprop_summary_f16.build_cfg(io_dtype, gate_dtype, use_dstate_in=False, use_dstate0=True, d_v=DV, **flags)
            tail_constexprs = (int(tail.unit_chunks), int(tail_chain_rows), transition_cfg, tail_bprop_summary_cfg)
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
                from_dlpack(buffers["state_m_main"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
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
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        uncut_summary_backward_cache[key] = cute.compile(
            uncut_summary_backward_host,
            io_dtype,
            int(b_t),
            int(num_sm),
            bprop_summary_cfg,
            transition_cfg,
            int(transition_rows),
            float(scale),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=2) if transition_m is not None or tail is not None else None,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(dstate_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate_in is not None else None,
            work_items_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(dstate_scratch, assumed_align=16).mark_layout_dynamic(),
            *transition_placeholders,
            cuda.CUstream(int(stream)),
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            *tail_constexprs,
            *tail_placeholders,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return uncut_summary_backward_cache[key]


def run_uncut_summary_backward(
    compiled,
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
    scheduler,
    scheduler_all,
    workspace,
    dstate_scratch,
    transition_m,
    transition,
    scheduler_transition,
    transition_words,
    scale,
    stream,
    item_limit_word=None,
    tail_scalars=(),
    tail_buffers=(),
) -> None:
    """Replay the uncut backward summary: one crossing into the DSL."""
    compiled(
        float(scale),
        q,
        k,
        do,
        gate,
        beta,
        beta if transition_m is not None or tail_buffers else None,
        a_log,
        dt_bias,
        cu_seqlens,
        dstate0,
        dstate_in,
        work_items,
        work_count,
        scheduler,
        scheduler_all,
        workspace,
        dstate_scratch,
        transition_m,
        transition_m,
        transition,
        scheduler_transition,
        transition_words,
        cuda.CUstream(int(stream)),
        item_limit_word,
        *tail_scalars,
        *tail_buffers,
    )
