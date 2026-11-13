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

"""One compiled launch for the GDN warmup, uncut and dv_split forwards: the split-K table (plan, scan and walk, warmup only), the
prefill prologue and the prefill issued from a single host, the way ``split_k.launch`` already sequences its three kernels.
Every kernel, its host and the tensor placeholder each host was compiled with are the standalone modules' own; this host
only sequences the launches, so the kernels' SASS is unchanged and the Python side crosses into the DSL once per call
instead of two or three times.  A buffer that two hosts read through different placeholder types is passed twice, once per
type: the table marks the gate at its element alignment along its last mode, a_log fully dynamic, dt_bias, work_items,
work_count and item_scratch as 4-byte compact views, cu_seqlens at 4 bytes; the prologue and prefill mark the same buffers
as the standalone prefill wrapper does."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..common import l2norm, split_k
from cudnn.frost.buffers import DeviceView
from cudnn.frost.compiled_cache import compile_cached

from ..common.host import get_dtype, family_slots
from ..common.piece_chain import DV_SPLIT_TILES, chain_rows_per_cta, dtype_name, launch_state_chain
from . import gdn_chain_forward_f16, gdn_prefill_f16, gdn_recompute_f16, gdn_summary_f16, gdn_tinv_f16

TAIL_BUFFERS = (
    "cu_pieces",
    "main_rows",
    "summary_rows",
    "summary_count",
    "work_items_summary",
    "scheduler_all",
    "scheduler_summary",
    "tinv_words",
    "tinv_rows",
    "tinv_row_count",
    "summary_words",
    "tinv",
    "state_h_summary",
    "state_m_summary",
    "state_h_chain",
    "state_m_chain",
    "state_x_chain",
    "seed",
    "state_x_prefill",
    "work_items",
    "work_count",
    "scheduler",
    "workspace",
)

warmup_forward_cache = {}


@cute.jit
def warmup_forward_host(
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
    prefill_cfg: cutlass.Constexpr,
    tiles_per_head: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    item_limit: cutlass.Int32,
    item_base: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    a_log: Optional[cute.Tensor],
    a_log_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    beta: Optional[cute.Tensor],
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    state_in: Optional[cute.Tensor],
    composed: Optional[cute.Tensor],
    state_out: Optional[cute.Tensor],
    seed_indices: Optional[cute.Tensor],
    final_indices: Optional[cute.Tensor],
    checkpoints: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_items_table: cute.Tensor,
    work_count: cute.Tensor,
    work_count_table: cute.Tensor,
    staging: Optional[cute.Tensor],
    item_scratch: Optional[cute.Tensor],
    chunk_scratch: Optional[cute.Tensor],
    scheduler: cute.Tensor,
    workspace: cute.Tensor,
    stream: cuda.CUstream,
    k_slots: cutlass.Constexpr[int] = 1,
    v_slots: cutlass.Constexpr[int] = 1,
    item_limit_word: Optional[cute.Tensor] = None,
    tail_enable: Optional[cute.Tensor] = None,
    fold_rows: cutlass.Constexpr[int] = 0,
    fold_update: Optional[cute.Tensor] = None,
    fold_transition: Optional[cute.Tensor] = None,
    fold_seed: Optional[cute.Tensor] = None,
    fold_composed: Optional[cute.Tensor] = None,
    l2norm_k_slots: cutlass.Constexpr[int] = 1,
    l2norm_k_expand_num: cutlass.Constexpr[int] = 1,
    l2norm_skip_k: cutlass.Constexpr[bool] = False,
    l2norm_q: Optional[cute.Tensor] = None,
    l2norm_k: Optional[cute.Tensor] = None,
    l2norm_inv_q: Optional[cute.Tensor] = None,
    l2norm_inv_k: Optional[cute.Tensor] = None,
    tail_unit_chunks: cutlass.Constexpr[int] = 0,
    tail_chain_rows: cutlass.Constexpr[int] = 0,
    tail_tinv_cfg: cutlass.Constexpr = None,
    tail_summary_cfg: cutlass.Constexpr = None,
    tail_prefill_cfg: cutlass.Constexpr = None,
    tail_pieces: cutlass.Int32 = 0,
    tail_num_seqs: cutlass.Int32 = 0,
    tail_batch_base: cutlass.Int32 = 0,
    tail_first_head_lo: cutlass.Int32 = 0,
    tail_item_limit: cutlass.Int32 = 0,
    tail_item_base: cutlass.Int32 = 0,
    tail_cu_pieces: Optional[cute.Tensor] = None,
    tail_main_rows: Optional[cute.Tensor] = None,
    tail_summary_rows: Optional[cute.Tensor] = None,
    tail_summary_count: Optional[cute.Tensor] = None,
    tail_work_items_summary: Optional[cute.Tensor] = None,
    tail_scheduler_all: Optional[cute.Tensor] = None,
    tail_scheduler_summary: Optional[cute.Tensor] = None,
    tail_tinv_words: Optional[cute.Tensor] = None,
    tail_tinv_rows: Optional[cute.Tensor] = None,
    tail_tinv_row_count: Optional[cute.Tensor] = None,
    tail_summary_words: Optional[cute.Tensor] = None,
    tail_tinv: Optional[cute.Tensor] = None,
    tail_state_h_summary: Optional[cute.Tensor] = None,
    tail_state_m_summary: Optional[cute.Tensor] = None,
    tail_state_h_chain: Optional[cute.Tensor] = None,
    tail_state_m_chain: Optional[cute.Tensor] = None,
    tail_state_x_chain: Optional[cute.Tensor] = None,
    tail_seed: Optional[cute.Tensor] = None,
    tail_state_x_prefill: Optional[cute.Tensor] = None,
    tail_work_items: Optional[cute.Tensor] = None,
    tail_work_count: Optional[cute.Tensor] = None,
    tail_scheduler: Optional[cute.Tensor] = None,
    tail_workspace: Optional[cute.Tensor] = None,
) -> None:
    if cutlass.const_expr(fold_update is not None):
        launch_state_chain(
            n_heads_out,
            prefill_cfg.d_v * tiles_per_head,
            prefill_cfg.d_k,
            fold_rows,
            cutlass.Int32(fold_update.shape[0]),
            False,
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
    if cutlass.const_expr(l2norm_q is not None):
        cta_rows = l2norm.THREADS_PER_CTA // l2norm.FWD_LANES * l2norm.FWD_ROWS_PER_GROUP
        q_rows = cutlass.Int32(q.shape[0]) * cutlass.Int32(q.shape[1])
        qk_rows = q_rows if cutlass.const_expr(l2norm_skip_k) else q_rows + cutlass.Int32(k.shape[0]) * cutlass.Int32(k.shape[1])
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
            prefill_cfg.d_k,
            1,
            0,
            False,
            l2norm_k_slots,
            l2norm_k_expand_num,
            stream,
        )
    gate_sms = num_sms if cutlass.const_expr(item_limit_word is not None) else 0
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
            scheduler,
            n_scan_ctas,
            n_walk_ctas,
            stream,
            item_limit_word,
            gate_sms=gate_sms,
        )
        gate_sms = 0
    gdn_prefill_f16.prologue(
        io_dtype,
        b_t,
        order_gen,
        expand_num,
        q,
        k,
        v,
        o,
        cu_seqlens,
        checkpoints,
        staging,
        work_count,
        work_items,
        scheduler,
        checkpoint_every_n,
        None,
        workspace,
        stream,
        tiles_per_head,
        k_slots,
        v_slots,
        checkpoint_after_block=prefill_cfg.checkpoint_after_block,
        checkpoint_box_elems=prefill_cfg.checkpoint_box_elems,
        item_limit=item_limit,
        item_base=item_base,
        item_limit_word=item_limit_word,
        tail_enable=tail_enable,
        gate_sms=gate_sms,
    )
    gdn_prefill_f16.host(
        prefill_cfg,
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        o,
        cu_seqlens,
        state_in,
        composed,
        state_out,
        seed_indices,
        final_indices,
        None,
        work_items,
        work_count,
        scheduler,
        checkpoint_every_n,
        scale,
        workspace,
        stream,
    )
    if cutlass.const_expr(tail_summary_cfg is not None):
        gdn_chain_forward_f16.chain_forward_host(
            tail_unit_chunks,
            b_t,
            expand_num,
            False,
            tail_tinv_cfg,
            tail_summary_cfg,
            tail_prefill_cfg,
            tail_prefill_cfg.d_v,
            tail_prefill_cfg.d_k,
            tail_chain_rows,
            tail_seed is not None,
            True,
            tail_pieces,
            n_heads_out,
            tail_num_seqs,
            checkpoint_every_n,
            scale,
            tail_batch_base,
            tail_first_head_lo,
            q,
            k,
            v,
            gate,
            beta,
            a_log,
            dt_bias,
            o,
            cu_seqlens,
            tail_cu_pieces,
            tail_main_rows,
            tail_summary_rows,
            tail_work_count,
            tail_summary_count,
            tail_work_items,
            tail_work_items_summary,
            tail_scheduler_all,
            tail_scheduler_summary,
            tail_scheduler,
            tail_tinv_words,
            tail_tinv_rows,
            tail_tinv_row_count,
            tail_summary_words,
            tail_workspace,
            tail_tinv,
            tail_state_h_summary,
            tail_state_m_summary,
            tail_state_h_chain,
            tail_state_m_chain,
            tail_state_x_chain,
            tail_seed,
            None,
            seed_indices,
            tail_state_x_prefill,
            state_out,
            final_indices,
            checkpoints,
            stream,
            k_slots,
            v_slots,
            item_limit_word,
        )
    if cutlass.const_expr(tail_summary_cfg is None and tail_prefill_cfg is not None):
        warmup_forward_host(
            False,
            b_t,
            log_gate,
            safe_gate,
            overhead_chunks,
            expand_num,
            warmup_cap,
            n_heads_out,
            num_sms,
            io_dtype,
            True,
            tail_prefill_cfg,
            tail_prefill_cfg.tiles_per_head,
            n_tiles,
            tail_item_limit,
            tail_item_base,
            ideal_chunks,
            batch_size,
            log2_thresh,
            n_scan_ctas,
            n_walk_ctas,
            checkpoint_every_n,
            scale,
            q,
            k,
            v,
            gate,
            None,
            a_log,
            None,
            dt_bias,
            None,
            beta,
            o,
            cu_seqlens,
            cu_seqlens_table,
            state_in,
            None,
            state_out,
            seed_indices,
            final_indices,
            checkpoints,
            tail_work_items,
            work_items_table,
            tail_work_count,
            work_count_table,
            None,
            None,
            None,
            tail_scheduler,
            tail_workspace,
            stream,
            k_slots,
            v_slots,
            None,
            item_limit_word,
        )


def build_warmup_forward(
    *,
    q,
    k,
    v,
    gate,
    beta,
    a_log,
    dt_bias,
    o,
    cu_seqlens,
    state_in,
    composed=None,
    state_out,
    seed_indices,
    final_indices,
    checkpoints,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    workspace,
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
    checkpoint_every_n_tokens,
    checkpoint_after_block=False,
    scale,
    device,
    stream,
    tiles_per_head=1,
    item_limit_word=None,
    tail_enable=None,
    tail=None,
    tail_chain_rows=0,
    tail_scalars=(),
    tail_buffers=(),
    lead_buffers=(None,) * 8,
    l2norm_skip_k=False,
):
    """Compile (cached per static config: dtypes, heads, dims, gate flags, the split-K geometry, the d_v split, state and checkpoint
    presence, device) the warmup or uncut forward launch over the buffers of one plan.  The placeholders repeat the marks
    of the standalone split-table and prefill builds so every kernel compiles as it does there."""
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
    state_src = state_in if state_in is not None else state_out
    state_dtype = get_dtype(state_src.dtype) if state_src is not None else cutlass.Float32
    k_slots = family_slots(k, expand_num, 3)
    v_slots = family_slots(v, expand_num, 3)
    beta_slots = family_slots(beta, expand_num, 2) if beta is not None else 1
    fold_update, fold_transition, fold_seed, fold_composed, l2norm_q, l2norm_k, l2norm_inv_q, l2norm_inv_k = lead_buffers
    fold_rows = chain_rows_per_cta(DV, DK, 1, gate.shape[1], num_sm) if fold_update is not None else 0
    l2norm_k_slots = family_slots(l2norm_k, expand_num, 3) if l2norm_k is not None else 1
    l2norm_k_expand_num = int(l2norm_k.shape[1]) if l2norm_k is not None and len(l2norm_k.shape) == 4 else 1
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(beta.dtype) if beta is not None else "none",
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        str(state_src.dtype) if state_src is not None else "none",
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
        state_in is not None,
        composed is not None,
        state_out is not None,
        str(seed_indices.dtype) if seed_indices is not None else "none",
        str(final_indices.dtype) if final_indices is not None else "none",
        int(checkpoint_every_n_tokens) > 0,
        str(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else "none",
        bool(checkpoint_after_block),
        int(tiles_per_head),
        len(k.shape),
        len(v.shape),
        len(beta.shape) if beta is not None else 0,
        k_slots,
        v_slots,
        beta_slots,
        item_limit_word is not None,
        tail_enable is not None,
        (int(tail.unit_chunks), int(tail_chain_rows), bool(tail.dv_split)) if tail is not None else None,
        tuple(str(buffer.dtype) if buffer is not None else None for buffer in lead_buffers),
        fold_rows,
        len(l2norm_k.shape) if l2norm_k is not None else 0,
        l2norm_k_slots,
        l2norm_k_expand_num,
        bool(l2norm_skip_k),
    )
    if key not in warmup_forward_cache:
        prefill_cfg = gdn_prefill_f16.build_cfg(
            io_dtype,
            state_dtype,
            max_active_clusters=num_sm,
            use_initial_state=state_in is not None or composed is not None,
            state_table=state_in is not None,
            has_composed=composed is not None,
            store_final_state=state_out is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            checkpoint_dtype=get_dtype(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else None,
            checkpoint_after_block=bool(checkpoint_after_block),
            log_gate=log_gate,
            safe_gate=safe_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            tinv_source="compute",
            d_k=DK,
            d_v=DV // tiles_per_head,
            expand_num=expand_num,
            tiles_per_head=tiles_per_head,
            k_slots=k_slots,
            v_slots=v_slots,
            beta_slots=beta_slots,
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
        state_in_placeholder = None
        if state_in is not None:
            state_in_placeholder = from_dlpack(state_in, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        composed_placeholder = None
        if composed is not None:
            composed_placeholder = from_dlpack(composed, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        state_out_placeholder = None
        if state_out is not None:
            state_out_placeholder = from_dlpack(state_out, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        tail_constexprs = ()
        tail_placeholders = ()
        if tail is not None:
            buffers = dict(zip(TAIL_BUFFERS, tail_buffers))
            tail_flags = dict(
                store_final_state=state_out is not None,
                enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
                checkpoint_dtype=get_dtype(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else None,
                checkpoint_after_block=bool(checkpoint_after_block),
                log_gate=log_gate,
                safe_gate=safe_gate,
                beta_sigmoid=use_beta_sigmoid,
                allow_neg_eigval=allow_neg_eigval,
                d_k=DK,
                expand_num=expand_num,
                k_slots=k_slots,
                v_slots=v_slots,
                beta_slots=beta_slots,
            )
            tail_tinv_cfg, tail_summary_cfg = None, None
            tail_views = [None] * 19
            if tail.dv_split:
                tail_prefill_cfg = gdn_prefill_f16.build_cfg(
                    io_dtype,
                    state_dtype,
                    max_active_clusters=num_sm,
                    use_initial_state=state_in is not None,
                    state_table=state_in is not None,
                    tinv_source="compute",
                    d_v=DV // DV_SPLIT_TILES,
                    tiles_per_head=DV_SPLIT_TILES,
                    **tail_flags,
                )
            else:
                HO = gate.shape[1]
                seed_name = dtype_name(state_in.dtype) if state_in is not None else "float32"
                tail_tinv_cfg, tail_summary_cfg, tail_prefill_cfg = gdn_chain_forward_f16.build_configs(
                    io_dtype, get_dtype(buffers["state_h_summary"].dtype), get_dtype(buffers["state_x_prefill"].dtype), num_sm=num_sm, d_v=DV, **tail_flags
                )
                tail_items_summary = from_dlpack(buffers["work_items_summary"], assumed_align=16)
                tail_items_summary.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
                tail_states = {}
                for name in ("state_h_summary", "state_m_summary", "state_x_prefill"):
                    tail_states[name] = from_dlpack(buffers[name], assumed_align=16)
                    tail_states[name].mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
                tail_views = [
                    from_dlpack(buffers["cu_pieces"], assumed_align=4).mark_layout_dynamic(),
                    from_dlpack(buffers["main_rows"], assumed_align=16).mark_layout_dynamic(),
                    from_dlpack(buffers["summary_rows"], assumed_align=16).mark_layout_dynamic(),
                    from_dlpack(buffers["summary_count"], assumed_align=4).mark_layout_dynamic(),
                    tail_items_summary,
                    from_dlpack(buffers["scheduler_all"], assumed_align=4).mark_layout_dynamic(),
                    from_dlpack(buffers["scheduler_summary"], assumed_align=4).mark_layout_dynamic(),
                    from_dlpack(buffers["tinv_words"], assumed_align=128).mark_layout_dynamic(),
                    from_dlpack(buffers["tinv_rows"], assumed_align=16).mark_layout_dynamic(leading_dim=1),
                    from_dlpack(buffers["tinv_row_count"], assumed_align=4).mark_layout_dynamic(),
                    from_dlpack(buffers["summary_words"], assumed_align=128).mark_layout_dynamic(),
                    from_dlpack(buffers["tinv"], assumed_align=128).mark_layout_dynamic(leading_dim=3),
                    tail_states["state_h_summary"],
                    tail_states["state_m_summary"],
                    from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                    from_dlpack(DeviceView(256, (1, HO, DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                    from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                    (
                        from_dlpack(DeviceView(256, (1, HO, DV, DK), seed_name, int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=3)
                        if state_in is not None
                        else None
                    ),
                    tail_states["state_x_prefill"],
                ]
            tail_items = from_dlpack(buffers["work_items"], assumed_align=16)
            tail_items.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_constexprs = (int(tail.unit_chunks), int(tail_chain_rows), tail_tinv_cfg, tail_summary_cfg, tail_prefill_cfg)
            tail_placeholders = [
                *(cutlass.Int32(int(value)) for value in tail_scalars),
                *tail_views,
                tail_items,
                from_dlpack(buffers["work_count"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["workspace"], assumed_align=128).mark_layout_dynamic(),
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
        warmup_forward_cache[key] = compile_cached(
            warmup_forward_host,
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
            prefill_cfg,
            int(tiles_per_head),
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(0),
            cutlass.Int32(facts.ideal_chunks),
            cutlass.Int32(facts.batch_size),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_walk_ctas),
            cutlass.Int32(int(checkpoint_every_n_tokens)),
            float(scale),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=len(k.shape) - 1),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=len(v.shape) - 1),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            gate_table_placeholder,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            dt_bias_table_placeholder,
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=len(beta.shape) - 1) if beta is not None else None,
            from_dlpack(o, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            state_in_placeholder,
            composed_placeholder,
            state_out_placeholder,
            from_dlpack(seed_indices, assumed_align=4).mark_layout_dynamic() if seed_indices is not None else None,
            from_dlpack(final_indices, assumed_align=4).mark_layout_dynamic() if final_indices is not None else None,
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if int(checkpoint_every_n_tokens) > 0 else None,
            work_items_placeholder,
            work_items_table_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            work_count_table_placeholder,
            staging_placeholder,
            item_scratch_placeholder,
            chunk_scratch_placeholder,
            from_dlpack(scheduler, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            cuda.CUstream(int(stream)),
            k_slots,
            v_slots,
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            from_dlpack(tail_enable, assumed_align=16).mark_layout_dynamic() if tail_enable is not None else None,
            fold_rows,
            *fold_placeholders,
            l2norm_k_slots,
            l2norm_k_expand_num,
            bool(l2norm_skip_k),
            from_dlpack(l2norm_q, assumed_align=16).mark_layout_dynamic(leading_dim=2) if l2norm_q is not None else None,
            from_dlpack(l2norm_k, assumed_align=16).mark_layout_dynamic(leading_dim=len(l2norm_k.shape) - 1) if l2norm_k is not None else None,
            from_dlpack(l2norm_inv_q, assumed_align=16).mark_layout_dynamic(leading_dim=1) if l2norm_inv_q is not None else None,
            from_dlpack(l2norm_inv_k, assumed_align=16).mark_layout_dynamic(leading_dim=1) if l2norm_inv_k is not None else None,
            *tail_constexprs,
            *tail_placeholders,
            cache_key=repr((__name__, key)),
            symbol="frost_la_gdn_warmup_forward_f16",
            options="--enable-tvm-ffi --opt-level 2",
        )
    compiled = warmup_forward_cache[key]
    return compiled, facts


def run_warmup_forward(
    compiled,
    facts,
    *,
    q,
    k,
    v,
    gate,
    beta,
    a_log,
    dt_bias,
    o,
    cu_seqlens,
    state_in,
    composed=None,
    state_out,
    seed_indices,
    final_indices,
    checkpoints,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    workspace,
    checkpoint_every_n_tokens,
    scale,
    stream,
    item_limit=None,
    item_base=None,
    item_limit_word=None,
    tail_enable=None,
    tail_scalars=(),
    tail_buffers=(),
    lead_buffers=(None,) * 8,
) -> None:
    """Replay the warmup or uncut forward: one crossing into the DSL for the table, prologue and prefill launches.  The
    plan validated the contract at build, so nothing here raises."""
    compiled(
        facts.n_heads_out,
        facts.n_tiles,
        facts.n_tiles if item_limit is None else int(item_limit),
        0 if item_base is None else int(item_base),
        facts.ideal_chunks,
        facts.batch_size,
        facts.log2_threshold,
        facts.n_scan_ctas,
        facts.n_walk_ctas if item_limit is None else min(int(facts.n_walk_ctas), int(item_limit)),
        int(checkpoint_every_n_tokens),
        float(scale),
        q,
        k,
        v,
        gate,
        gate if facts.split else None,
        a_log if facts.safe_gate else None,
        a_log if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        beta,
        o,
        cu_seqlens,
        cu_seqlens,
        state_in,
        composed,
        state_out,
        seed_indices,
        final_indices,
        checkpoints if int(checkpoint_every_n_tokens) > 0 else None,
        work_items,
        work_items,
        work_count,
        work_count,
        item_scratch if facts.split else None,
        item_scratch if facts.split else None,
        chunk_scratch if facts.split else None,
        scheduler,
        workspace,
        cuda.CUstream(int(stream)),
        item_limit_word,
        tail_enable,
        *lead_buffers,
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
    "scheduler_summary",
    "summary_words",
    "state_h_summary",
    "state_m_summary",
    "state_h_chain",
    "state_m_chain",
    "state_x_chain",
    "seed",
    "final_state",
    "transition",
)

warmup_summary_cache = {}


@cute.jit
def warmup_summary_host(
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
    summary_cfg: cutlass.Constexpr,
    recompute_cfg: cutlass.Constexpr,
    tiles_per_head: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    a_log: Optional[cute.Tensor],
    a_log_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    state_in: Optional[cute.Tensor],
    state_out: cute.Tensor,
    transition: Optional[cute.Tensor],
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
    stream: cuda.CUstream,
    k_slots: cutlass.Constexpr[int] = 1,
    v_slots: cutlass.Constexpr[int] = 1,
    item_limit_word: Optional[cute.Tensor] = None,
    l2norm_k_slots: cutlass.Constexpr[int] = 1,
    l2norm_k_expand_num: cutlass.Constexpr[int] = 1,
    l2norm_k: Optional[cute.Tensor] = None,
    tail_unit_chunks: cutlass.Constexpr[int] = 0,
    tail_chain_rows: cutlass.Constexpr[int] = 0,
    tail_summary_cfg: cutlass.Constexpr = None,
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
    tail_scheduler_summary: Optional[cute.Tensor] = None,
    tail_summary_words: Optional[cute.Tensor] = None,
    tail_state_h_summary: Optional[cute.Tensor] = None,
    tail_state_m_summary: Optional[cute.Tensor] = None,
    tail_state_h_chain: Optional[cute.Tensor] = None,
    tail_state_m_chain: Optional[cute.Tensor] = None,
    tail_state_x_chain: Optional[cute.Tensor] = None,
    tail_seed: Optional[cute.Tensor] = None,
    tail_final_state: Optional[cute.Tensor] = None,
    tail_transition: Optional[cute.Tensor] = None,
) -> None:
    """One compiled launch of the summary node without a chain of its own."""
    gate_sms = num_sms if cutlass.const_expr(item_limit_word is not None) else 0
    if cutlass.const_expr(l2norm_k is not None):
        cta_rows = l2norm.THREADS_PER_CTA // l2norm.FWD_LANES * l2norm.FWD_ROWS_PER_GROUP
        k_rows = cutlass.Int32(k.shape[0]) * cutlass.Int32(k.shape[1])
        l2norm.l2norm_qk_launch(
            k,
            l2norm_k,
            k,
            k,
            None,
            None,
            cutlass.Int32(0),
            k_rows,
            cutlass.Int32(k.shape[1]),
            cutlass.Int32(k.shape[1]),
            (k_rows + cutlass.Int32(cta_rows - 1)) // cutlass.Int32(cta_rows),
            tinv_cfg.d_k,
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
    if cutlass.const_expr(summary_cfg is not None):
        gdn_summary_f16.prologue(
            io_dtype,
            b_t,
            True,
            order_gen,
            expand_num,
            k,
            v,
            gate,
            cu_seqlens,
            staging,
            work_count,
            work_items,
            scheduler_all,
            tinv,
            workspace,
            stream,
            tiles_per_head,
            k_slots,
            v_slots,
            item_limit_word,
            gate_sms,
        )
        gdn_summary_f16.host(
            summary_cfg,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            cu_seqlens,
            tinv,
            state_in,
            None,
            state_out,
            transition,
            work_items,
            work_count,
            scheduler,
            workspace,
            stream,
        )
    if cutlass.const_expr(recompute_cfg is not None):
        gdn_recompute_f16.prologue(
            io_dtype,
            b_t,
            True,
            order_gen,
            False,
            expand_num,
            k,
            v,
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
            workspace,
            stream,
            tiles_per_head,
            k_slots,
            v_slots,
        )
        gdn_recompute_f16.host(
            recompute_cfg,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            cu_seqlens,
            state_in,
            None,
            state_out,
            None,
            tinv,
            work_items,
            work_count,
            scheduler,
            cutlass.Int32(0),
            cutlass.Int32(0),
            workspace,
            stream,
        )
    if cutlass.const_expr(tail_summary_cfg is not None):
        gdn_chain_forward_f16.chain_summary_host(
            tail_unit_chunks,
            b_t,
            expand_num,
            False,
            None,
            tail_summary_cfg,
            tail_summary_cfg.d_v,
            tail_summary_cfg.d_k,
            tail_chain_rows,
            tail_seed is not None,
            True,
            tail_transition is not None,
            tail_pieces,
            n_heads_out,
            tail_num_seqs,
            tail_batch_base,
            tail_first_head_lo,
            k,
            v,
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
            tail_scheduler_summary,
            None,
            None,
            None,
            tail_summary_words,
            tinv,
            tail_state_h_summary,
            tail_state_m_summary,
            tail_state_h_chain,
            tail_state_m_chain,
            tail_state_x_chain,
            tail_seed,
            tail_final_state,
            tail_transition,
            stream,
            k_slots,
            v_slots,
            item_limit_word,
        )


def build_warmup_summary(
    *,
    k,
    v,
    gate,
    beta,
    a_log,
    dt_bias,
    cu_seqlens,
    state_in,
    state_out,
    transition,
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
    device,
    stream,
    tiles_per_head=1,
    item_limit_word=None,
    l2norm_k=None,
    tail=None,
    tail_chain_rows=0,
    tail_scalars=(),
    tail_buffers=(),
):
    """Compile (cached per static config) the summary launch over the buffers of one plan."""
    DK = k.shape[-1]
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
    io_dtype = get_dtype(k.dtype)
    k_slots = family_slots(k, expand_num, 3)
    v_slots = family_slots(v, expand_num, 3)
    beta_slots = family_slots(beta, expand_num, 2)
    l2norm_k_slots = family_slots(l2norm_k, expand_num, 3) if l2norm_k is not None else 1
    l2norm_k_expand_num = int(l2norm_k.shape[1]) if l2norm_k is not None and len(l2norm_k.shape) == 4 else 1
    key = (
        str(k.dtype),
        str(gate.dtype),
        str(beta.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        str(state_in.dtype) if state_in is not None else "none",
        str(state_out.dtype),
        str(transition.dtype) if transition is not None else "none",
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
        int(tiles_per_head),
        len(k.shape),
        len(v.shape),
        len(beta.shape),
        k_slots,
        v_slots,
        beta_slots,
        item_limit_word is not None,
        str(l2norm_k.dtype) if l2norm_k is not None else None,
        len(l2norm_k.shape) if l2norm_k is not None else 0,
        l2norm_k_slots,
        l2norm_k_expand_num,
        (int(tail.unit_chunks), int(tail_chain_rows)) if tail is not None else None,
    )
    if key not in warmup_summary_cache:
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
        flags = dict(
            max_active_clusters=num_sm,
            use_initial_state=state_in is not None,
            log_gate=log_gate,
            safe_gate=safe_gate,
            d_k=DK,
            d_v=DV // tiles_per_head,
            expand_num=expand_num,
            k_slots=k_slots,
            v_slots=v_slots,
            tiles_per_head=tiles_per_head,
            state_table=state_in is not None,
        )
        summary_cfg, recompute_cfg = None, None
        if transition is None:
            state_dtype = get_dtype((state_in if state_in is not None else state_out).dtype)
            recompute_cfg = gdn_recompute_f16.build_cfg(
                io_dtype, state_dtype, store_final_state=True, enable_checkpoints=False, seed_checkpoints=False, **flags
            )
        else:
            summary_cfg = gdn_summary_f16.build_cfg(io_dtype, get_dtype(state_out.dtype), **flags)
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
            HO = gate.shape[1]
            tail_summary_cfg = gdn_summary_f16.build_cfg(
                io_dtype,
                get_dtype(buffers["state_h_summary"].dtype),
                max_active_clusters=num_sm,
                use_initial_state=False,
                log_gate=log_gate,
                safe_gate=safe_gate,
                d_k=DK,
                d_v=DV,
                expand_num=expand_num,
                k_slots=k_slots,
                v_slots=v_slots,
            )
            tail_items = from_dlpack(buffers["work_items"], assumed_align=16)
            tail_items.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_states = {}
            for name in ("state_h_summary", "state_m_summary"):
                tail_states[name] = from_dlpack(buffers[name], assumed_align=16)
                tail_states[name].mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
            tail_constexprs = (int(tail.unit_chunks), int(tail_chain_rows), tail_summary_cfg)
            tail_placeholders = [
                *(cutlass.Int32(int(value)) for value in tail_scalars),
                from_dlpack(buffers["cu_pieces"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["main_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["summary_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["main_count"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["summary_count"], assumed_align=4).mark_layout_dynamic(),
                tail_items,
                from_dlpack(buffers["scheduler_all"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_summary"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["summary_words"], assumed_align=128).mark_layout_dynamic(),
                tail_states["state_h_summary"],
                tail_states["state_m_summary"],
                from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(DeviceView(256, (1, HO, DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                (
                    from_dlpack(DeviceView(256, (1, HO, DV, DK), dtype_name(buffers["seed"].dtype), int(device)), assumed_align=4).mark_layout_dynamic(
                        leading_dim=3
                    )
                    if buffers["seed"] is not None
                    else None
                ),
                from_dlpack(DeviceView(256, (1, HO, DV, DK), dtype_name(buffers["final_state"].dtype), int(device)), assumed_align=4).mark_layout_dynamic(
                    leading_dim=3
                ),
                from_dlpack(DeviceView(256, (1, HO, DK, DK), dtype_name(buffers["transition"].dtype), int(device)), assumed_align=16).mark_layout_dynamic(
                    leading_dim=3
                ),
            ]
        warmup_summary_cache[key] = compile_cached(
            warmup_summary_host,
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
            summary_cfg,
            recompute_cfg,
            int(tiles_per_head),
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(facts.ideal_chunks),
            cutlass.Int32(facts.batch_size),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_walk_ctas),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=len(k.shape) - 1),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=len(v.shape) - 1),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            gate_table_placeholder,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            dt_bias_table_placeholder,
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=len(beta.shape) - 1),
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(state_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if state_in is not None else None,
            from_dlpack(state_out, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(transition, assumed_align=16).mark_layout_dynamic(leading_dim=3) if transition is not None else None,
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
            cuda.CUstream(int(stream)),
            k_slots,
            v_slots,
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            l2norm_k_slots,
            l2norm_k_expand_num,
            from_dlpack(l2norm_k, assumed_align=16).mark_layout_dynamic(leading_dim=len(l2norm_k.shape) - 1) if l2norm_k is not None else None,
            *tail_constexprs,
            *tail_placeholders,
            cache_key=repr((__name__, "summary", key)),
            symbol="frost_la_gdn_warmup_summary_f16",
            options="--enable-tvm-ffi --opt-level 2",
        )
    return warmup_summary_cache[key], facts


def run_warmup_summary(
    compiled,
    facts,
    *,
    k,
    v,
    gate,
    beta,
    a_log,
    dt_bias,
    cu_seqlens,
    state_in,
    state_out,
    transition,
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
    stream,
    item_limit_word=None,
    l2norm_k=None,
    tail_scalars=(),
    tail_buffers=(),
) -> None:
    """Replay the summary launch: one crossing into the DSL."""
    compiled(
        facts.n_heads_out,
        facts.n_tiles,
        facts.ideal_chunks,
        facts.batch_size,
        facts.log2_threshold,
        facts.n_scan_ctas,
        facts.n_walk_ctas,
        k,
        v,
        gate,
        gate if facts.split else None,
        a_log if facts.safe_gate else None,
        a_log if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        beta,
        cu_seqlens,
        cu_seqlens,
        state_in,
        state_out,
        transition,
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
        cuda.CUstream(int(stream)),
        item_limit_word,
        l2norm_k,
        *tail_scalars,
        *tail_buffers,
    )
