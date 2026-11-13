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

"""One compiled launch for the KDA uncut and d_v-split forwards: the prefill prologue and the prefill issued from a single
host.  Every kernel, its host and the tensor placeholder each host was compiled with are the standalone modules' own; this
host only sequences the launches, so the kernels' SASS is unchanged and the Python side crosses into the DSL once per call
instead of twice; the prologue and prefill mark the same buffers as the standalone prefill wrapper does."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.buffers import DeviceView
from cudnn.frost.compiled_cache import compile_cached

from ..common.host import get_dtype
from ..common.piece_chain import DV_SPLIT_TILES, chain_rows_per_cta, dtype_name, launch_state_chain
from ..common.state_snapshot import snapshot_states
from . import kda_chain_forward_f16, kda_prefill_f16, kda_prep_f16, kda_prep_prefill_f16, kda_prep_summary_f16, kda_recompute_f16, kda_summary_f16

TAIL_BUFFERS = (
    "cu_pieces",
    "cu_pieces_main",
    "main_rows",
    "summary_rows",
    "summary_count",
    "work_items_summary",
    "scheduler_all",
    "scheduler_summary",
    "summary_words",
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
    "prep_k_decay",
    "prep_q_decay",
    "prep_t",
    "prep_a",
    "prep_diag",
    "prep_words",
    "prep_rows",
    "prep_row_count",
)

uncut_forward_cache = {}


@cute.jit
def uncut_forward_host(
    b_t: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    prefill_cfg: cutlass.Constexpr,
    tiles_per_head: cutlass.Constexpr[int],
    checkpoint_every_n: cutlass.Int32,
    scale: cutlass.Float32,
    item_limit: cutlass.Int32,
    item_base: cutlass.Int32,
    row_seq_lo: cutlass.Int32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    beta: cute.Tensor,
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    state_in: Optional[cute.Tensor],
    composed: Optional[cute.Tensor],
    state_out: Optional[cute.Tensor],
    seed_indices: Optional[cute.Tensor],
    final_indices: Optional[cute.Tensor],
    checkpoints: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    scheduler: cute.Tensor,
    workspace: cute.Tensor,
    prep: cutlass.Constexpr[bool],
    prep_cfg: cutlass.Constexpr,
    prep_k_decay: Optional[cute.Tensor],
    prep_q_decay: Optional[cute.Tensor],
    prep_t: Optional[cute.Tensor],
    prep_a: Optional[cute.Tensor],
    prep_diag: Optional[cute.Tensor],
    prep_words: Optional[cute.Tensor],
    prep_rows: Optional[cute.Tensor],
    prep_row_count: Optional[cute.Tensor],
    state_snapshot: Optional[cute.Tensor],
    stream: cuda.CUstream,
    item_limit_word: Optional[cute.Tensor] = None,
    tail_enable: Optional[cute.Tensor] = None,
    fold_rows: cutlass.Constexpr[int] = 0,
    fold_update: Optional[cute.Tensor] = None,
    fold_transition: Optional[cute.Tensor] = None,
    fold_seed: Optional[cute.Tensor] = None,
    fold_composed: Optional[cute.Tensor] = None,
    tail_unit_chunks: cutlass.Constexpr[int] = 0,
    tail_chain_rows: cutlass.Constexpr[int] = 0,
    tail_summary_cfg: cutlass.Constexpr = None,
    tail_prefill_cfg: cutlass.Constexpr = None,
    tail_prep_cfg: cutlass.Constexpr = None,
    tail_pieces: cutlass.Int32 = 0,
    tail_num_seqs: cutlass.Int32 = 0,
    tail_batch_base: cutlass.Int32 = 0,
    tail_first_head_lo: cutlass.Int32 = 0,
    tail_item_limit: cutlass.Int32 = 0,
    tail_item_base: cutlass.Int32 = 0,
    tail_cu_pieces: Optional[cute.Tensor] = None,
    tail_cu_pieces_main: Optional[cute.Tensor] = None,
    tail_main_rows: Optional[cute.Tensor] = None,
    tail_summary_rows: Optional[cute.Tensor] = None,
    tail_summary_count: Optional[cute.Tensor] = None,
    tail_work_items_summary: Optional[cute.Tensor] = None,
    tail_scheduler_all: Optional[cute.Tensor] = None,
    tail_scheduler_summary: Optional[cute.Tensor] = None,
    tail_summary_words: Optional[cute.Tensor] = None,
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
    tail_prep_k_decay: Optional[cute.Tensor] = None,
    tail_prep_q_decay: Optional[cute.Tensor] = None,
    tail_prep_t: Optional[cute.Tensor] = None,
    tail_prep_a: Optional[cute.Tensor] = None,
    tail_prep_diag: Optional[cute.Tensor] = None,
    tail_prep_words: Optional[cute.Tensor] = None,
    tail_prep_rows: Optional[cute.Tensor] = None,
    tail_prep_row_count: Optional[cute.Tensor] = None,
) -> None:
    if cutlass.const_expr(fold_update is not None):
        launch_state_chain(
            cutlass.Int32(gate.shape[1]),
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
    gate_sms = num_sms if cutlass.const_expr(item_limit_word is not None) else 0
    seed_state, seed_rows = state_in, seed_indices
    if cutlass.const_expr(state_snapshot is not None):
        # The prefill's final state overwrites the seeds its exact twin would reread: seed both from a copy.
        snapshot_states(state_in, state_snapshot, seed_indices, stream)
        seed_state, seed_rows = state_snapshot, None
    if cutlass.const_expr(prep):
        kda_prep_prefill_f16.prologue(
            io_dtype,
            b_t,
            num_sms,
            q,
            k,
            v,
            gate,
            o,
            checkpoints,
            cu_seqlens,
            work_count,
            work_items,
            scheduler,
            workspace,
            checkpoint_every_n,
            stream,
            tiles_per_head,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            prep_a,
            prep_diag,
            prep_cfg,
            prep_words,
            prep_rows,
            prep_row_count,
            checkpoint_after_block=prefill_cfg.checkpoint_after_block,
            checkpoint_box_elems=prefill_cfg.checkpoint_box_elems,
            item_limit=item_limit,
            item_base=item_base,
            row_seq_lo=row_seq_lo,
            item_limit_word=item_limit_word,
            tail_enable=tail_enable,
            gate_sms=gate_sms,
        )
        kda_prep_f16.host(
            prep_cfg,
            q,
            k,
            prep_words,
            gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            prep_a,
            prep_diag,
            prep_rows,
            prep_row_count,
            scheduler,
            stream,
        )
    else:
        kda_prefill_f16.prologue(
            io_dtype,
            b_t,
            num_sms,
            q,
            k,
            v,
            gate,
            o,
            checkpoints,
            cu_seqlens,
            work_count,
            work_items,
            scheduler,
            workspace,
            checkpoint_every_n,
            stream,
            tiles_per_head,
            checkpoint_after_block=prefill_cfg.checkpoint_after_block,
            checkpoint_box_elems=prefill_cfg.checkpoint_box_elems,
            item_limit=item_limit,
            item_base=item_base,
            item_limit_word=item_limit_word,
            tail_enable=tail_enable,
            gate_sms=gate_sms,
        )
    prefill_host = kda_prep_prefill_f16.host if cutlass.const_expr(prep) else kda_prefill_f16.host
    prefill_host(
        prefill_cfg,
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        seed_state,
        composed,
        o,
        state_out,
        seed_rows,
        final_indices,
        work_items,
        work_count,
        scheduler,
        workspace,
        checkpoint_every_n,
        scale,
        stream,
    )
    if cutlass.const_expr(tail_summary_cfg is not None):
        kda_chain_forward_f16.chain_forward_host(
            tail_unit_chunks,
            b_t,
            False,
            tail_summary_cfg,
            tail_prefill_cfg,
            tail_prefill_cfg.d_v,
            tail_prefill_cfg.d_k,
            tail_chain_rows,
            tail_seed is not None,
            True,
            tail_pieces,
            cutlass.Int32(gate.shape[1]),
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
            tail_cu_pieces_main,
            tail_main_rows,
            tail_summary_rows,
            tail_work_count,
            tail_summary_count,
            tail_work_items,
            tail_work_items_summary,
            tail_scheduler_all,
            tail_scheduler_summary,
            tail_scheduler,
            tail_summary_words,
            tail_workspace,
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
            item_limit_word,
        )
    if cutlass.const_expr(tail_summary_cfg is None and tail_prefill_cfg is not None):
        uncut_forward_host(
            b_t,
            num_sms,
            io_dtype,
            tail_prefill_cfg,
            tail_prefill_cfg.tiles_per_head,
            checkpoint_every_n,
            scale,
            tail_item_limit,
            tail_item_base,
            tail_batch_base,
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
            None,
            state_out,
            seed_indices,
            final_indices,
            checkpoints,
            tail_work_items,
            tail_work_count,
            tail_scheduler,
            tail_workspace,
            tail_prep_cfg is not None,
            tail_prep_cfg,
            tail_prep_k_decay,
            tail_prep_q_decay,
            tail_prep_t,
            tail_prep_a,
            tail_prep_diag,
            tail_prep_words,
            tail_prep_rows,
            tail_prep_row_count,
            None if cutlass.const_expr(tail_prep_cfg is not None) else state_snapshot,
            stream,
            None,
            item_limit_word,
        )


def build_configs(
    io_dtype,
    state_dtype,
    gate_dtype,
    *,
    use_initial_state,
    store_final_state,
    enable_checkpoints,
    checkpoint_dtype=None,
    checkpoint_after_block=False,
    tiles_per_head,
    prep,
    state_table=None,
    has_composed=False,
    **flags,
):
    prefill_module = kda_prep_prefill_f16 if prep else kda_prefill_f16
    prefill_cfg = prefill_module.build_cfg(
        io_dtype,
        state_dtype,
        gate_dtype,
        use_initial_state=use_initial_state,
        state_table=state_table,
        has_composed=has_composed,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        checkpoint_dtype=checkpoint_dtype,
        checkpoint_after_block=checkpoint_after_block,
        tiles_per_head=tiles_per_head,
        **dict(flags, d_v=flags["d_v"] // tiles_per_head),
    )
    prep_cfg = None
    if prep:
        prep_cfg = kda_prep_f16.build_cfg(
            io_dtype,
            gate_dtype,
            num_sm=flags["max_active_clusters"],
            **{name: value for name, value in flags.items() if name not in ("max_active_clusters", "d_v")},
        )
    return prefill_cfg, prep_cfg


def build_uncut_forward(
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
    scheduler,
    workspace,
    num_sm,
    b_t,
    log_gate,
    safe_gate,
    gate_lower_bound,
    use_qk_l2norm,
    use_beta_sigmoid,
    allow_neg_eigval,
    checkpoint_every_n_tokens,
    checkpoint_after_block=False,
    scale,
    device,
    stream,
    tiles_per_head=1,
    prep=False,
    prep_k_decay=None,
    prep_q_decay=None,
    prep_t=None,
    prep_a=None,
    prep_diag=None,
    prep_words=None,
    prep_rows=None,
    prep_row_count=None,
    item_limit_word=None,
    tail_enable=None,
    state_snapshot=None,
    tail=None,
    tail_chain_rows=0,
    tail_prep=False,
    tail_scalars=(),
    tail_buffers=(),
    fold_buffers=(None,) * 4,
):
    """Compile (cached per static config: dtypes, heads, dims, gate flags and bound, the d_v split, state and checkpoint
    presence, device) the uncut forward launch over the buffers of one plan.  The placeholders repeat the marks of the
    standalone prefill build so every kernel compiles as it does there."""
    HQ, DK = q.shape[1], q.shape[2]
    k.shape[1]
    HV, DV = v.shape[1], v.shape[2]
    max(HQ, HV)
    if not safe_gate:
        a_log = None
        dt_bias = None
    io_dtype = get_dtype(q.dtype)
    gate_dtype = get_dtype(gate.dtype)
    state_src = state_in if state_in is not None else state_out
    state_dtype = get_dtype(state_src.dtype) if state_src is not None else cutlass.Float32
    fold_rows = chain_rows_per_cta(DV, DK, 1, gate.shape[1], num_sm) if fold_buffers[0] is not None else 0
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(beta.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        str(state_src.dtype) if state_src is not None else "none",
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
        state_in is not None,
        composed is not None,
        state_out is not None,
        str(seed_indices.dtype) if seed_indices is not None else "none",
        str(final_indices.dtype) if final_indices is not None else "none",
        int(checkpoint_every_n_tokens) > 0,
        str(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else "none",
        bool(checkpoint_after_block),
        int(tiles_per_head),
        bool(prep),
        item_limit_word is not None,
        tail_enable is not None,
        state_snapshot is not None,
        (int(tail.unit_chunks), int(tail_chain_rows), bool(tail.dv_split), bool(tail_prep)) if tail is not None else None,
        (int(fold_rows), tuple(str(buffer.dtype) if buffer is not None else None for buffer in fold_buffers)),
    )
    if key not in uncut_forward_cache:
        prefill_cfg, prep_cfg = build_configs(
            io_dtype,
            state_dtype,
            gate_dtype,
            use_initial_state=state_in is not None or composed is not None,
            state_table=state_in is not None,
            has_composed=composed is not None,
            store_final_state=state_out is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            checkpoint_dtype=get_dtype(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else None,
            checkpoint_after_block=bool(checkpoint_after_block),
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=float(gate_lower_bound) * kda_prefill_f16.LOG2_E,
            log_gate=log_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            max_active_clusters=num_sm,
            d_k=DK,
            d_v=DV,
            tiles_per_head=tiles_per_head,
            prep=prep,
        )
        prep_placeholders = [None] * 8
        if prep:
            prep_placeholders = [from_dlpack(rec, assumed_align=128).mark_layout_dynamic(leading_dim=3) for rec in (prep_k_decay, prep_q_decay, prep_t)]
            prep_placeholders.append(from_dlpack(prep_a, assumed_align=16).mark_layout_dynamic(leading_dim=2))
            prep_placeholders.append(from_dlpack(prep_diag, assumed_align=16).mark_layout_dynamic(leading_dim=2))
            prep_placeholders.append(from_dlpack(prep_words, assumed_align=128).mark_layout_dynamic())
            prep_placeholders.append(from_dlpack(prep_rows, assumed_align=16).mark_layout_dynamic(leading_dim=1))
            prep_placeholders.append(from_dlpack(prep_row_count, assumed_align=4).mark_layout_dynamic())

        tail_constexprs = ()
        tail_placeholders = ()
        if tail is not None:
            buffers = dict(zip(TAIL_BUFFERS, tail_buffers))
            tail_flags = dict(
                store_final_state=state_out is not None,
                enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
                checkpoint_dtype=get_dtype(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else None,
                checkpoint_after_block=bool(checkpoint_after_block),
                l2norm=use_qk_l2norm,
                safe_gate=safe_gate,
                gate_scale_log2=float(gate_lower_bound) * kda_prefill_f16.LOG2_E,
                log_gate=log_gate,
                beta_sigmoid=use_beta_sigmoid,
                allow_neg_eigval=allow_neg_eigval,
                max_active_clusters=num_sm,
                d_k=DK,
                d_v=DV,
            )
            tail_summary_cfg, tail_prep_cfg = None, None
            tail_views = [None] * 16
            tail_prep_placeholders = [None] * 8
            if tail.dv_split:
                tail_prefill_cfg, tail_prep_cfg = build_configs(
                    io_dtype,
                    state_dtype,
                    gate_dtype,
                    use_initial_state=state_in is not None,
                    state_table=state_in is not None,
                    tiles_per_head=DV_SPLIT_TILES,
                    prep=tail_prep,
                    **tail_flags,
                )
                if tail_prep:
                    tail_prep_placeholders = [
                        from_dlpack(buffers[name], assumed_align=128).mark_layout_dynamic(leading_dim=3) for name in ("prep_k_decay", "prep_q_decay", "prep_t")
                    ]
                    tail_prep_placeholders.append(from_dlpack(buffers["prep_a"], assumed_align=16).mark_layout_dynamic(leading_dim=2))
                    tail_prep_placeholders.append(from_dlpack(buffers["prep_diag"], assumed_align=16).mark_layout_dynamic(leading_dim=2))
                    tail_prep_placeholders.append(from_dlpack(buffers["prep_words"], assumed_align=128).mark_layout_dynamic())
                    tail_prep_placeholders.append(from_dlpack(buffers["prep_rows"], assumed_align=16).mark_layout_dynamic(leading_dim=1))
                    tail_prep_placeholders.append(from_dlpack(buffers["prep_row_count"], assumed_align=4).mark_layout_dynamic())
            else:
                HO = gate.shape[1]
                seed_name = dtype_name(state_in.dtype) if state_in is not None else "float32"
                tail_summary_cfg, tail_prefill_cfg = kda_chain_forward_f16.build_configs(
                    io_dtype, get_dtype(buffers["state_x_prefill"].dtype), gate_dtype, **tail_flags
                )
                tail_items_summary = from_dlpack(buffers["work_items_summary"], assumed_align=16)
                tail_items_summary.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
                tail_views = [
                    from_dlpack(buffers["cu_pieces"], assumed_align=4).mark_layout_dynamic(),
                    from_dlpack(buffers["cu_pieces_main"], assumed_align=8).mark_layout_dynamic(),
                    from_dlpack(buffers["main_rows"], assumed_align=16).mark_layout_dynamic(),
                    from_dlpack(buffers["summary_rows"], assumed_align=16).mark_layout_dynamic(),
                    from_dlpack(buffers["summary_count"], assumed_align=4).mark_layout_dynamic(),
                    tail_items_summary,
                    from_dlpack(buffers["scheduler_all"], assumed_align=4).mark_layout_dynamic(),
                    from_dlpack(buffers["scheduler_summary"], assumed_align=4).mark_layout_dynamic(),
                    from_dlpack(buffers["summary_words"], assumed_align=128).mark_layout_dynamic(),
                    from_dlpack(buffers["state_h_summary"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                    from_dlpack(buffers["state_m_summary"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                    from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                    from_dlpack(DeviceView(256, (1, HO, DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                    from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                    (
                        from_dlpack(DeviceView(256, (1, HO, DV, DK), seed_name, int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=3)
                        if state_in is not None
                        else None
                    ),
                    from_dlpack(buffers["state_x_prefill"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                ]
            tail_items = from_dlpack(buffers["work_items"], assumed_align=16)
            tail_items.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_constexprs = (int(tail.unit_chunks), int(tail_chain_rows), tail_summary_cfg, tail_prefill_cfg, tail_prep_cfg)
            tail_placeholders = [
                *(cutlass.Int32(int(value)) for value in tail_scalars),
                *tail_views,
                tail_items,
                from_dlpack(buffers["work_count"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["workspace"], assumed_align=128).mark_layout_dynamic(),
                *tail_prep_placeholders,
            ]
        fold_update, fold_transition, fold_seed, fold_composed = fold_buffers
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
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        uncut_forward_cache[key] = compile_cached(
            uncut_forward_host,
            int(b_t),
            int(num_sm),
            io_dtype,
            prefill_cfg,
            int(tiles_per_head),
            cutlass.Int32(int(checkpoint_every_n_tokens)),
            float(scale),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(o, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(state_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if state_in is not None else None,
            from_dlpack(composed, assumed_align=16).mark_layout_dynamic(leading_dim=3) if composed is not None else None,
            from_dlpack(state_out, assumed_align=16).mark_layout_dynamic(leading_dim=3) if state_out is not None else None,
            from_dlpack(seed_indices, assumed_align=4).mark_layout_dynamic() if seed_indices is not None else None,
            from_dlpack(final_indices, assumed_align=4).mark_layout_dynamic() if final_indices is not None else None,
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if int(checkpoint_every_n_tokens) > 0 else None,
            work_items_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            bool(prep),
            prep_cfg,
            *prep_placeholders,
            from_dlpack(state_snapshot, assumed_align=16).mark_layout_dynamic(leading_dim=3) if state_snapshot is not None else None,
            cuda.CUstream(int(stream)),
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            from_dlpack(tail_enable, assumed_align=16).mark_layout_dynamic() if tail_enable is not None else None,
            fold_rows,
            *fold_placeholders,
            *tail_constexprs,
            *tail_placeholders,
            cache_key=repr((__name__, key)),
            symbol="frost_la_kda_uncut_forward_f16",
            options="--enable-tvm-ffi --opt-level 2",
        )
    return uncut_forward_cache[key]


def run_uncut_forward(
    compiled,
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
    scheduler,
    workspace,
    checkpoint_every_n_tokens,
    scale,
    stream,
    prep_k_decay=None,
    prep_q_decay=None,
    prep_t=None,
    prep_a=None,
    prep_diag=None,
    prep_words=None,
    prep_rows=None,
    prep_row_count=None,
    item_limit=None,
    item_base=None,
    row_seq_lo=None,
    item_limit_word=None,
    tail_enable=None,
    state_snapshot=None,
    tail_scalars=(),
    tail_buffers=(),
    fold_buffers=(None,) * 4,
) -> None:
    """Replay the uncut forward: one crossing into the DSL for the prologue and prefill launches.  The plan validated the
    contract at build, so nothing here raises."""
    compiled(
        int(checkpoint_every_n_tokens),
        float(scale),
        int(work_items.shape[0]) if item_limit is None else int(item_limit),
        0 if item_base is None else int(item_base),
        0 if row_seq_lo is None else int(row_seq_lo),
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
        checkpoints if int(checkpoint_every_n_tokens) > 0 else None,
        work_items,
        work_count,
        scheduler,
        workspace,
        prep_k_decay,
        prep_q_decay,
        prep_t,
        prep_a,
        prep_diag,
        prep_words,
        prep_rows,
        prep_row_count,
        state_snapshot,
        cuda.CUstream(int(stream)),
        item_limit_word,
        tail_enable,
        *fold_buffers,
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

uncut_summary_cache = {}


@cute.jit
def uncut_summary_host(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    summary_cfg: cutlass.Constexpr,
    prep_cfg: cutlass.Constexpr,
    recompute_cfg: cutlass.Constexpr,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    state_in: Optional[cute.Tensor],
    state_out: cute.Tensor,
    transition: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    scheduler: cute.Tensor,
    scheduler_all: cute.Tensor,
    workspace: cute.Tensor,
    prep_k_decay: Optional[cute.Tensor],
    prep_t: Optional[cute.Tensor],
    prep_diag: Optional[cute.Tensor],
    prep_words: Optional[cute.Tensor],
    prep_rows: Optional[cute.Tensor],
    prep_row_count: Optional[cute.Tensor],
    stream: cuda.CUstream,
    item_limit_word: Optional[cute.Tensor] = None,
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
    if cutlass.const_expr(prep_cfg is not None):
        kda_prep_summary_f16.prep_summary_host(
            summary_cfg,
            prep_cfg,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            state_in,
            None,
            state_out,
            transition,
            work_items,
            work_count,
            scheduler,
            scheduler_all,
            workspace,
            prep_k_decay,
            prep_t,
            prep_diag,
            prep_words,
            prep_rows,
            prep_row_count,
            stream,
        )
    if cutlass.const_expr(prep_cfg is None and summary_cfg is not None):
        kda_summary_f16.prologue(
            io_dtype,
            b_t,
            True,
            k,
            v,
            gate,
            cu_seqlens,
            work_count,
            work_items,
            scheduler_all,
            workspace,
            stream,
            summary_cfg.tiles_per_head,
            item_limit_word,
            num_sms if cutlass.const_expr(item_limit_word is not None) else 0,
        )
        kda_summary_f16.host(
            summary_cfg,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
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
        kda_recompute_f16.prologue(
            io_dtype,
            b_t,
            True,
            False,
            k,
            v,
            gate,
            None,
            cu_seqlens,
            work_count,
            work_items,
            scheduler_all,
            workspace,
            cutlass.Int32(0),
            cutlass.Int32(0),
            stream,
        )
        kda_recompute_f16.host(
            recompute_cfg,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            state_in,
            None,
            state_out,
            None,
            work_items,
            work_count,
            scheduler,
            workspace,
            cutlass.Int32(0),
            cutlass.Int32(0),
            stream,
        )
    if cutlass.const_expr(tail_summary_cfg is not None):
        kda_chain_forward_f16.chain_summary_host(
            tail_unit_chunks,
            b_t,
            False,
            tail_summary_cfg,
            tail_summary_cfg.d_v,
            tail_summary_cfg.d_k,
            tail_chain_rows,
            tail_seed is not None,
            tail_transition is not None,
            True,
            tail_pieces,
            cutlass.Int32(gate.shape[1]),
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
            tail_summary_words,
            tail_state_h_summary,
            tail_state_m_summary,
            tail_state_h_chain,
            tail_state_m_chain,
            tail_state_x_chain,
            tail_seed,
            tail_final_state,
            tail_transition,
            stream,
            item_limit_word,
        )


def build_uncut_summary(
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
    scheduler,
    scheduler_all,
    workspace,
    num_sm,
    b_t,
    log_gate,
    safe_gate,
    gate_lower_bound,
    use_qk_l2norm,
    use_beta_sigmoid,
    allow_neg_eigval,
    device,
    stream,
    tiles_per_head=1,
    prep_k_decay=None,
    prep_t=None,
    prep_diag=None,
    prep_words=None,
    prep_rows=None,
    prep_row_count=None,
    item_limit_word=None,
    tail=None,
    tail_chain_rows=0,
    tail_scalars=(),
    tail_buffers=(),
):
    """Compile (cached per static config) the uncut summary launch over the buffers of one plan."""
    DK, DV = k.shape[2], v.shape[2]
    if not safe_gate:
        a_log = None
        dt_bias = None
    io_dtype = get_dtype(k.dtype)
    gate_dtype = get_dtype(gate.dtype)
    prep = prep_words is not None
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
        bool(log_gate),
        bool(safe_gate),
        float(gate_lower_bound),
        bool(use_qk_l2norm),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        int(tiles_per_head),
        prep,
        item_limit_word is not None,
        (int(tail.unit_chunks), int(tail_chain_rows)) if tail is not None else None,
    )
    if key not in uncut_summary_cache:
        flags = dict(
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=float(gate_lower_bound) * kda_summary_f16.LOG2_E,
            log_gate=log_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
        )
        summary_cfg, prep_cfg, recompute_cfg = None, None, None
        if transition is None:
            recompute_cfg = kda_recompute_f16.build_cfg(
                io_dtype,
                get_dtype(state_in.dtype if state_in is not None else state_out.dtype),
                gate_dtype,
                use_initial_state=state_in is not None,
                store_final_state=True,
                enable_checkpoints=False,
                seed_checkpoints=False,
                max_active_clusters=num_sm,
                d_k=DK,
                d_v=DV,
                **flags,
            )
        else:
            summary_cfg = (kda_prep_summary_f16 if prep else kda_summary_f16).build_cfg(
                io_dtype,
                gate_dtype,
                use_initial_state=state_in is not None,
                max_active_clusters=num_sm,
                d_k=DK,
                d_v=DV // tiles_per_head,
                tiles_per_head=tiles_per_head,
                **flags,
            )
        prep_placeholders = [None] * 6
        if prep:
            prep_cfg = kda_prep_f16.build_cfg(io_dtype, gate_dtype, num_sm=num_sm, d_k=DK, summary_records=True, **flags)
            prep_placeholders = [
                from_dlpack(prep_k_decay, assumed_align=128).mark_layout_dynamic(leading_dim=3),
                from_dlpack(prep_t, assumed_align=128).mark_layout_dynamic(leading_dim=3),
                from_dlpack(prep_diag, assumed_align=16).mark_layout_dynamic(leading_dim=2),
                from_dlpack(prep_words, assumed_align=128).mark_layout_dynamic(),
                from_dlpack(prep_rows, assumed_align=16).mark_layout_dynamic(leading_dim=1),
                from_dlpack(prep_row_count, assumed_align=4).mark_layout_dynamic(),
            ]
        tail_constexprs = ()
        tail_placeholders = ()
        if tail is not None:
            buffers = dict(zip(SUMMARY_TAIL_BUFFERS, tail_buffers))
            HO = gate.shape[1]
            tail_summary_cfg = kda_summary_f16.build_cfg(io_dtype, gate_dtype, use_initial_state=False, max_active_clusters=num_sm, d_k=DK, d_v=DV, **flags)
            tail_items = from_dlpack(buffers["work_items"], assumed_align=16)
            tail_items.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
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
                from_dlpack(buffers["state_h_summary"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(buffers["state_m_summary"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
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
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        uncut_summary_cache[key] = compile_cached(
            uncut_summary_host,
            io_dtype,
            int(b_t),
            int(num_sm),
            summary_cfg,
            prep_cfg,
            recompute_cfg,
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(state_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if state_in is not None else None,
            from_dlpack(state_out, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(transition, assumed_align=16).mark_layout_dynamic(leading_dim=3) if transition is not None else None,
            work_items_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            *prep_placeholders,
            cuda.CUstream(int(stream)),
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            *tail_constexprs,
            *tail_placeholders,
            cache_key=repr((__name__, "summary", key)),
            symbol="frost_la_kda_uncut_summary_f16",
            options="--enable-tvm-ffi --opt-level 2",
        )
    return uncut_summary_cache[key]


def run_uncut_summary(
    compiled,
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
    scheduler,
    scheduler_all,
    workspace,
    stream,
    prep_k_decay=None,
    prep_t=None,
    prep_diag=None,
    prep_words=None,
    prep_rows=None,
    prep_row_count=None,
    item_limit_word=None,
    tail_scalars=(),
    tail_buffers=(),
) -> None:
    """Replay the uncut summary: one crossing into the DSL."""
    compiled(
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        state_in,
        state_out,
        transition,
        work_items,
        work_count,
        scheduler,
        scheduler_all,
        workspace,
        prep_k_decay,
        prep_t,
        prep_diag,
        prep_words,
        prep_rows,
        prep_row_count,
        cuda.CUstream(int(stream)),
        item_limit_word,
        *tail_scalars,
        *tail_buffers,
    )
