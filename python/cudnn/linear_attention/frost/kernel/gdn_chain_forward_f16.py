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

"""One compiled launch for the GDN chain forward: chain prologue, T pass, fused summary, fp32 state chain and prefill issued
from a single host, the way ``split_k.run_table`` launches plan, scan and walk.  Every kernel, its host and the tensor
placeholder each host was compiled with are the standalone modules' own; this host only sequences the five launches, so the
kernels' SASS is unchanged and the Python side crosses into the DSL once per call instead of five times.  A buffer that two
hosts read through different placeholder types (the summary's ``state_out`` is the state chain's ``H``) is passed twice, once
per type."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.buffers import DeviceView

from ..common import l2norm
from ..common.host import get_dtype, family_slots
from ..common.piece_chain import chain_rows_per_cta, dtype_name, launch_state_chain
from . import gdn_chain_prologue_f16, gdn_prefill_f16, gdn_summary_f16, gdn_tinv_f16

CHAIN_FORWARD_BUFFERS = (
    "q",
    "k",
    "v",
    "gate",
    "beta",
    "a_log",
    "dt_bias",
    "o",
    "cu_seqlens",
    "cu_pieces",
    "main_rows",
    "summary_rows",
    "main_count",
    "summary_count",
    "work_items",
    "work_items_summary",
    "scheduler_all",
    "scheduler_summary",
    "scheduler_prefill",
    "tinv_words",
    "tinv_rows",
    "tinv_row_count",
    "summary_words",
    "prefill_words",
    "tinv",
    "state_h",
    "state_m",
    "state_h",
    "state_m",
    "state_x",
    "seed",
    "composed",
    "seed_indices",
    "state_x",
    "final_state",
    "final_indices",
    "checkpoints",
)

chain_forward_cache = {}


@cute.jit
def chain_forward_host(
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    tinv_cfg: cutlass.Constexpr,
    summary_cfg: cutlass.Constexpr,
    prefill_cfg: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    has_seed: cutlass.Constexpr[bool],
    tail: cutlass.Constexpr[bool],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    scale: cutlass.Float32,
    batch_base: cutlass.Int32,
    first_head_lo: cutlass.Int32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor,
    scheduler_all: cute.Tensor,
    scheduler_summary: cute.Tensor,
    scheduler_prefill: cute.Tensor,
    tinv_words: cute.Tensor,
    tinv_rows: cute.Tensor,
    tinv_row_count: cute.Tensor,
    summary_words: cute.Tensor,
    prefill_words: cute.Tensor,
    tinv: cute.Tensor,
    state_h_summary: cute.Tensor,
    state_m_summary: cute.Tensor,
    state_h_chain: cute.Tensor,
    state_m_chain: cute.Tensor,
    state_x_chain: cute.Tensor,
    seed: Optional[cute.Tensor],
    composed: Optional[cute.Tensor],
    seed_indices: Optional[cute.Tensor],
    state_x_prefill: cute.Tensor,
    final_state: Optional[cute.Tensor],
    final_indices: Optional[cute.Tensor],
    checkpoints: Optional[cute.Tensor],
    stream: cuda.CUstream,
    k_slots: cutlass.Constexpr[int] = 1,
    v_slots: cutlass.Constexpr[int] = 1,
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
) -> None:
    if cutlass.const_expr(fold_update is not None):
        launch_state_chain(
            heads_out,
            dim_v,
            dim_k,
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
            dim_k,
            1,
            0,
            False,
            l2norm_k_slots,
            l2norm_k_expand_num,
            stream,
        )
    cu_full = None
    if cutlass.const_expr(tail):
        cu_full = cu_seqlens
        cu_seqlens = cute.make_tensor(cu_seqlens.iterator + batch_base, cute.make_layout((num_seqs + cutlass.Int32(1),), stride=(1,)))
        if cutlass.const_expr(seed is not None and seed_indices is None):
            seed = cute.make_tensor(
                seed.iterator + batch_base * seed.stride[0], cute.make_layout((num_seqs, seed.shape[1], seed.shape[2], seed.shape[3]), stride=seed.stride)
            )
        if cutlass.const_expr(seed_indices is not None):
            seed_indices = cute.make_tensor(seed_indices.iterator + batch_base, cute.make_layout((num_seqs,), stride=(1,)))
        if cutlass.const_expr(final_state is not None and final_indices is None):
            final_state = cute.make_tensor(
                final_state.iterator + batch_base * final_state.stride[0],
                cute.make_layout((num_seqs, final_state.shape[1], final_state.shape[2], final_state.shape[3]), stride=final_state.stride),
            )
        if cutlass.const_expr(final_indices is not None):
            final_indices = cute.make_tensor(final_indices.iterator + batch_base, cute.make_layout((num_seqs,), stride=(1,)))
    gdn_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        expand_num,
        length_rule,
        prefill_cfg.max_active_clusters,
        heads_out,
        False,
        1,
        cutlass.Int32(0),
        checkpoint_every_n,
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        scheduler_all,
        None,
        None,
        tinv_words,
        tinv_rows,
        tinv_row_count,
        summary_words,
        None,
        None,
        None,
        prefill_words,
        None,
        None,
        q,
        k,
        v,
        o,
        None,
        checkpoints,
        None,
        None,
        None,
        None,
        None,
        tinv,
        stream,
        k_slots,
        v_slots,
        prefill_cfg.checkpoint_after_block,
        prefill_cfg.checkpoint_box_elems,
        tail=tail,
        first_head_lo=first_head_lo,
        cu_full=cu_full,
        batch_base=batch_base,
        tail_enable=tail_enable,
    )
    gdn_tinv_f16.host(tinv_cfg, False, k, tinv_words, gate, a_log, dt_bias, beta, cu_pieces, tinv, tinv_rows, tinv_row_count, stream)
    gdn_summary_f16.host(
        summary_cfg,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        cu_pieces,
        tinv,
        None,
        None,
        state_h_summary,
        state_m_summary,
        work_items_summary,
        summary_count,
        scheduler_summary,
        summary_words,
        stream,
    )
    launch_state_chain(
        heads_out,
        dim_v,
        dim_k,
        chain_rows,
        pieces,
        False,
        False,
        has_seed,
        False,
        False,
        True,
        num_seqs,
        state_h_chain,
        state_m_chain,
        state_x_chain,
        seed,
        None,
        None,
        main_rows,
        seed_indices,
        composed,
        stream,
        mTailEnable=tail_enable,
        first_head_lo=first_head_lo,
    )
    gdn_prefill_f16.host(
        prefill_cfg,
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        None,
        o,
        cu_pieces,
        state_x_prefill,
        None,
        final_state,
        None,
        final_indices,
        tinv,
        work_items,
        main_count,
        scheduler_prefill,
        checkpoint_every_n,
        scale,
        prefill_words,
        stream,
    )


def build_configs(
    io_dtype,
    summary_dtype,
    state_dtype,
    *,
    store_final_state,
    enable_checkpoints,
    checkpoint_dtype=None,
    checkpoint_after_block=False,
    num_sm,
    log_gate,
    safe_gate,
    beta_sigmoid,
    allow_neg_eigval,
    d_k,
    d_v,
    expand_num,
    k_slots,
    v_slots,
    beta_slots,
):
    tinv_cfg = gdn_tinv_f16.build_cfg(
        io_dtype,
        num_sm=num_sm,
        log_gate=log_gate,
        safe_gate=safe_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        d_k=d_k,
        expand_num=expand_num,
        k_slots=k_slots,
        beta_slots=beta_slots,
    )
    summary_cfg = gdn_summary_f16.build_cfg(
        io_dtype,
        summary_dtype,
        max_active_clusters=num_sm,
        use_initial_state=False,
        log_gate=log_gate,
        safe_gate=safe_gate,
        d_k=d_k,
        d_v=d_v,
        expand_num=expand_num,
        k_slots=k_slots,
        v_slots=v_slots,
    )
    prefill_cfg = gdn_prefill_f16.build_cfg(
        io_dtype,
        state_dtype,
        max_active_clusters=num_sm,
        use_initial_state=True,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        checkpoint_dtype=checkpoint_dtype,
        checkpoint_after_block=checkpoint_after_block,
        log_gate=log_gate,
        safe_gate=safe_gate,
        tinv_source="gmem",
        d_k=d_k,
        d_v=d_v,
        expand_num=expand_num,
        k_slots=k_slots,
        v_slots=v_slots,
        beta_slots=beta_slots,
    )
    return tinv_cfg, summary_cfg, prefill_cfg


def build_chain_forward(
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
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    work_items_summary,
    scheduler_all,
    scheduler_summary,
    scheduler_prefill,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    summary_words,
    prefill_words,
    tinv,
    state_h,
    state_m,
    state_x,
    seed,
    composed=None,
    seed_indices,
    final_state,
    final_indices,
    checkpoints,
    pieces,
    heads_out,
    num_seqs,
    unit_chunks,
    b_t,
    expand_num,
    length_rule,
    log_gate,
    safe_gate,
    use_beta_sigmoid,
    allow_neg_eigval,
    checkpoint_every_n_tokens,
    checkpoint_after_block=False,
    scale,
    chain_rows,
    device,
    num_sm,
    stream,
    tail=False,
    tail_enable=None,
    lead_buffers=(None,) * 8,
    l2norm_skip_k=False,
):
    """Compile (cached per static config: dtypes, heads, dims, gate flags, checkpoint and final-state presence, seed dtype,
    chain rows, device) the chain forward launch over the buffers of one plan; ``pieces``, ``heads_out`` and ``num_seqs`` are
    launch arguments.  The placeholders repeat the marks of the standalone modules' builds so every kernel compiles as it
    does there."""
    _HQ, DK = q.shape[1], q.shape[2]
    DV = v.shape[-1]
    HO = gate.shape[1]
    if not safe_gate:
        a_log = None
        dt_bias = None
    io_dtype = get_dtype(q.dtype)
    state_dtype = get_dtype(state_x.dtype)
    has_seed = seed is not None
    seed_name = dtype_name(seed.dtype) if has_seed else "float32"
    k_slots = family_slots(k, expand_num, 3)
    v_slots = family_slots(v, expand_num, 3)
    beta_slots = family_slots(beta, expand_num, 2)
    fold_update, fold_transition, fold_seed, fold_composed, l2norm_q, l2norm_k, l2norm_inv_q, l2norm_inv_k = lead_buffers
    fold_rows = chain_rows_per_cta(DV, DK, 1, gate.shape[1], num_sm) if fold_update is not None else 0
    l2norm_k_slots = family_slots(l2norm_k, expand_num, 3) if l2norm_k is not None else 1
    l2norm_k_expand_num = int(l2norm_k.shape[1]) if l2norm_k is not None and len(l2norm_k.shape) == 4 else 1
    key = (
        str(q.dtype),
        str(state_x.dtype),
        str(final_state.dtype) if final_state is not None else "none",
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(beta.dtype),
        seed_name,
        composed is not None,
        int(device),
        int(num_sm),
        DK,
        DV,
        int(unit_chunks),
        int(b_t),
        int(expand_num),
        bool(length_rule),
        bool(log_gate),
        bool(safe_gate),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        int(checkpoint_every_n_tokens) > 0,
        str(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else "none",
        bool(checkpoint_after_block),
        final_state is not None,
        has_seed,
        str(seed_indices.dtype) if seed_indices is not None else "none",
        str(final_indices.dtype) if final_indices is not None else "none",
        int(chain_rows),
        len(k.shape),
        len(v.shape),
        len(beta.shape),
        k_slots,
        v_slots,
        beta_slots,
        bool(tail),
        tail_enable is not None,
        tuple(str(buffer.dtype) if buffer is not None else None for buffer in lead_buffers),
        fold_rows,
        len(l2norm_k.shape) if l2norm_k is not None else 0,
        l2norm_k_slots,
        l2norm_k_expand_num,
        bool(l2norm_skip_k),
    )
    if key not in chain_forward_cache:
        tinv_cfg, summary_cfg, prefill_cfg = build_configs(
            io_dtype,
            get_dtype(state_h.dtype),
            state_dtype,
            store_final_state=final_state is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            checkpoint_dtype=get_dtype(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else None,
            checkpoint_after_block=bool(checkpoint_after_block),
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

        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_summary_placeholder = from_dlpack(work_items_summary, assumed_align=16)
        work_items_summary_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        state_h_placeholder = from_dlpack(state_h, assumed_align=16)
        state_h_placeholder.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
        state_m_placeholder = from_dlpack(state_m, assumed_align=16)
        state_m_placeholder.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
        state_x_placeholder = from_dlpack(state_x, assumed_align=16)
        state_x_placeholder.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
        final_state_placeholder = None
        if final_state is not None:
            final_state_placeholder = from_dlpack(final_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        fold_placeholders = [None] * 4
        if fold_update is not None:
            fold_placeholders = [
                from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(DeviceView(256, (1, HO, DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                (
                    from_dlpack(DeviceView(256, (1, HO, DV, DK), dtype_name(fold_seed.dtype), int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=3)
                    if fold_seed is not None
                    else None
                ),
                from_dlpack(DeviceView(256, (1, HO, DV, DK), dtype_name(fold_composed.dtype), int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=3),
            ]
        chain_forward_cache[key] = cute.compile(
            chain_forward_host,
            int(unit_chunks),
            int(b_t),
            int(expand_num),
            bool(length_rule),
            tinv_cfg,
            summary_cfg,
            prefill_cfg,
            DV,
            DK,
            int(chain_rows),
            has_seed,
            bool(tail),
            cutlass.Int32(int(pieces)),
            cutlass.Int32(int(heads_out)),
            cutlass.Int32(int(num_seqs)),
            cutlass.Int32(int(checkpoint_every_n_tokens)),
            float(scale),
            cutlass.Int32(0),
            cutlass.Int32(0),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=len(k.shape) - 1),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=len(v.shape) - 1),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=len(beta.shape) - 1),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(o, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_pieces, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(main_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(summary_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(main_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_count, assumed_align=4).mark_layout_dynamic(),
            work_items_placeholder,
            work_items_summary_placeholder,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_summary, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_prefill, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(tinv_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(tinv_rows, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(tinv_row_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(prefill_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(tinv, assumed_align=128).mark_layout_dynamic(leading_dim=3),
            state_h_placeholder,
            state_m_placeholder,
            from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(DeviceView(256, (1, HO, DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(DeviceView(256, (1, HO, DV, DK), seed_name, int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=3) if has_seed else None,
            (
                from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3)
                if composed is not None
                else None
            ),
            from_dlpack(seed_indices, assumed_align=4).mark_layout_dynamic() if seed_indices is not None else None,
            state_x_placeholder,
            final_state_placeholder,
            from_dlpack(final_indices, assumed_align=4).mark_layout_dynamic() if final_indices is not None else None,
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if checkpoints is not None else None,
            cuda.CUstream(int(stream)),
            k_slots,
            v_slots,
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
            options="--enable-tvm-ffi --opt-level 2",
        )
    return chain_forward_cache[key]


def run_chain_forward(
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
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    work_items_summary,
    scheduler_all,
    scheduler_summary,
    scheduler_prefill,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    summary_words,
    prefill_words,
    tinv,
    state_h,
    state_m,
    state_x,
    seed,
    composed=None,
    seed_indices,
    final_state,
    final_indices,
    checkpoints,
    pieces,
    heads_out,
    num_seqs,
    checkpoint_every_n_tokens,
    scale,
    stream,
    batch_base=0,
    first_head_lo=0,
    tail_enable=None,
    lead_buffers=(None,) * 8,
) -> None:
    """Replay the chain forward: one crossing into the DSL for the five launches.  The plan validated the contract at build,
    so nothing here raises."""
    buffers = locals()
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        int(checkpoint_every_n_tokens),
        float(scale),
        int(batch_base),
        int(first_head_lo),
        *(buffers[name] for name in CHAIN_FORWARD_BUFFERS),
        cuda.CUstream(int(stream)),
        tail_enable,
        *lead_buffers,
    )


CHAIN_SUMMARY_BUFFERS = (
    "k",
    "v",
    "gate",
    "beta",
    "a_log",
    "dt_bias",
    "cu_seqlens",
    "cu_pieces",
    "main_rows",
    "summary_rows",
    "main_count",
    "summary_count",
    "work_items",
    "scheduler_all",
    "scheduler_summary",
    "tinv_words",
    "tinv_rows",
    "tinv_row_count",
    "summary_words",
    "tinv",
    "state_h",
    "state_m",
    "state_h",
    "state_m",
    "state_x",
    "seed",
    "final_state",
    "transition",
)

chain_summary_cache = {}


@cute.jit
def chain_summary_host(
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    tinv_cfg: cutlass.Constexpr,
    summary_cfg: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    has_seed: cutlass.Constexpr[bool],
    tail: cutlass.Constexpr[bool],
    emit_summary: cutlass.Constexpr[bool],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    batch_base: cutlass.Int32,
    first_head_lo: cutlass.Int32,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_all: cute.Tensor,
    scheduler_summary: cute.Tensor,
    tinv_words: Optional[cute.Tensor],
    tinv_rows: Optional[cute.Tensor],
    tinv_row_count: Optional[cute.Tensor],
    summary_words: cute.Tensor,
    tinv: cute.Tensor,
    state_h_summary: cute.Tensor,
    state_m_summary: cute.Tensor,
    state_h_chain: cute.Tensor,
    state_m_chain: cute.Tensor,
    state_x_chain: cute.Tensor,
    seed: Optional[cute.Tensor],
    final_state: cute.Tensor,
    transition: Optional[cute.Tensor],
    stream: cuda.CUstream,
    k_slots: cutlass.Constexpr[int] = 1,
    v_slots: cutlass.Constexpr[int] = 1,
    tail_enable: Optional[cute.Tensor] = None,
    l2norm_k_slots: cutlass.Constexpr[int] = 1,
    l2norm_k_expand_num: cutlass.Constexpr[int] = 1,
    l2norm_k: Optional[cute.Tensor] = None,
) -> None:
    """The chain of the summary node: the chain forward without its main pass."""
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
            dim_k,
            1,
            0,
            False,
            l2norm_k_slots,
            l2norm_k_expand_num,
            stream,
        )
    cu_full = None
    if cutlass.const_expr(tail):
        cu_full = cu_seqlens
        cu_seqlens = cute.make_tensor(cu_seqlens.iterator + batch_base, cute.make_layout((num_seqs + cutlass.Int32(1),), stride=(1,)))
        if cutlass.const_expr(seed is not None):
            seed = cute.make_tensor(
                seed.iterator + batch_base * seed.stride[0], cute.make_layout((num_seqs, seed.shape[1], seed.shape[2], seed.shape[3]), stride=seed.stride)
            )
        final_state = cute.make_tensor(
            final_state.iterator + batch_base * final_state.stride[0],
            cute.make_layout((num_seqs, final_state.shape[1], final_state.shape[2], final_state.shape[3]), stride=final_state.stride),
        )
        if cutlass.const_expr(transition is not None):
            transition = cute.make_tensor(
                transition.iterator + batch_base * transition.stride[0],
                cute.make_layout((num_seqs, transition.shape[1], transition.shape[2], transition.shape[3]), stride=transition.stride),
            )
    gdn_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        expand_num,
        length_rule,
        summary_cfg.max_active_clusters,
        heads_out,
        False,
        1,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        None,
        scheduler_all,
        None,
        None,
        tinv_words,
        tinv_rows,
        tinv_row_count,
        summary_words,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        k,
        v,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        tinv,
        stream,
        k_slots,
        v_slots,
        tail=tail,
        first_head_lo=first_head_lo,
        cu_full=cu_full,
        batch_base=batch_base,
        tail_enable=tail_enable,
    )
    if cutlass.const_expr(tinv_rows is not None):
        gdn_tinv_f16.host(tinv_cfg, False, k, tinv_words, gate, a_log, dt_bias, beta, cu_pieces, tinv, tinv_rows, tinv_row_count, stream)
    gdn_summary_f16.host(
        summary_cfg,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        cu_pieces,
        tinv,
        None,
        None,
        state_h_summary,
        state_m_summary,
        work_items,
        main_count,
        scheduler_summary,
        summary_words,
        stream,
    )
    head_floor = None
    if cutlass.const_expr(tail):
        head_floor = first_head_lo
    launch_state_chain(
        heads_out,
        dim_v,
        dim_k,
        chain_rows,
        pieces,
        False,
        False,
        has_seed,
        True,
        emit_summary,
        True,
        num_seqs,
        state_h_chain,
        state_m_chain,
        state_x_chain,
        seed,
        final_state,
        transition,
        main_rows,
        None,
        None,
        stream,
        mTailEnable=tail_enable,
        first_head_lo=head_floor,
    )


def build_chain_summary(
    *,
    k,
    v,
    gate,
    beta,
    a_log,
    dt_bias,
    cu_seqlens,
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    scheduler_all,
    scheduler_summary,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    summary_words,
    tinv,
    state_h,
    state_m,
    state_x,
    seed,
    final_state,
    transition,
    pieces,
    heads_out,
    num_seqs,
    unit_chunks,
    b_t,
    expand_num,
    length_rule,
    log_gate,
    safe_gate,
    use_beta_sigmoid,
    allow_neg_eigval,
    chain_rows,
    device,
    num_sm,
    stream,
    tail=False,
    tail_enable=None,
    l2norm_k=None,
):
    """Compile (cached per static config) the summary chain launch over the buffers of one plan; ``pieces``, ``heads_out``
    and ``num_seqs`` are launch arguments."""
    DK = k.shape[-1]
    DV = v.shape[-1]
    HO = gate.shape[1]
    if not safe_gate:
        a_log = None
        dt_bias = None
    io_dtype = get_dtype(k.dtype)
    has_seed = seed is not None
    emit_summary = transition is not None
    seed_name = dtype_name(seed.dtype) if has_seed else "float32"
    tail_name = dtype_name(final_state.dtype)
    summary_name = dtype_name(transition.dtype) if emit_summary else "float32"
    k_slots = family_slots(k, expand_num, 3)
    v_slots = family_slots(v, expand_num, 3)
    beta_slots = family_slots(beta, expand_num, 2)
    l2norm_k_slots = family_slots(l2norm_k, expand_num, 3) if l2norm_k is not None else 1
    l2norm_k_expand_num = int(l2norm_k.shape[1]) if l2norm_k is not None and len(l2norm_k.shape) == 4 else 1
    key = (
        str(k.dtype),
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(beta.dtype),
        seed_name,
        tail_name,
        summary_name,
        int(device),
        int(num_sm),
        DK,
        DV,
        int(unit_chunks),
        int(b_t),
        int(expand_num),
        bool(length_rule),
        bool(log_gate),
        bool(safe_gate),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        has_seed,
        emit_summary,
        int(chain_rows),
        len(k.shape),
        len(v.shape),
        len(beta.shape),
        k_slots,
        v_slots,
        beta_slots,
        bool(tail),
        tail_enable is not None,
        str(l2norm_k.dtype) if l2norm_k is not None else None,
        len(l2norm_k.shape) if l2norm_k is not None else 0,
        l2norm_k_slots,
        l2norm_k_expand_num,
    )
    if key not in chain_summary_cache:
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
        summary_cfg = gdn_summary_f16.build_cfg(
            io_dtype,
            get_dtype(state_h.dtype),
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
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        state_h_placeholder = from_dlpack(state_h, assumed_align=16)
        state_h_placeholder.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
        state_m_placeholder = from_dlpack(state_m, assumed_align=16)
        state_m_placeholder.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
        chain_summary_cache[key] = cute.compile(
            chain_summary_host,
            int(unit_chunks),
            int(b_t),
            int(expand_num),
            bool(length_rule),
            tinv_cfg,
            summary_cfg,
            DV,
            DK,
            int(chain_rows),
            has_seed,
            bool(tail),
            emit_summary,
            cutlass.Int32(int(pieces)),
            cutlass.Int32(int(heads_out)),
            cutlass.Int32(int(num_seqs)),
            cutlass.Int32(0),
            cutlass.Int32(0),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=len(k.shape) - 1),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=len(v.shape) - 1),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=len(beta.shape) - 1),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_pieces, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(main_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(summary_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(main_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_count, assumed_align=4).mark_layout_dynamic(),
            work_items_placeholder,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_summary, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(tinv_words, assumed_align=128).mark_layout_dynamic() if tinv_words is not None else None,
            from_dlpack(tinv_rows, assumed_align=16).mark_layout_dynamic(leading_dim=1) if tinv_rows is not None else None,
            from_dlpack(tinv_row_count, assumed_align=4).mark_layout_dynamic() if tinv_row_count is not None else None,
            from_dlpack(summary_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(tinv, assumed_align=128).mark_layout_dynamic(leading_dim=3),
            state_h_placeholder,
            state_m_placeholder,
            from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(DeviceView(256, (1, HO, DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(DeviceView(256, (1, HO, DV, DK), seed_name, int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=3) if has_seed else None,
            from_dlpack(DeviceView(256, (1, HO, DV, DK), tail_name, int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=3),
            (
                from_dlpack(DeviceView(256, (1, HO, DK, DK), summary_name, int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3)
                if emit_summary
                else None
            ),
            cuda.CUstream(int(stream)),
            k_slots,
            v_slots,
            from_dlpack(tail_enable, assumed_align=16).mark_layout_dynamic() if tail_enable is not None else None,
            l2norm_k_slots,
            l2norm_k_expand_num,
            from_dlpack(l2norm_k, assumed_align=16).mark_layout_dynamic(leading_dim=len(l2norm_k.shape) - 1) if l2norm_k is not None else None,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return chain_summary_cache[key]


def run_chain_summary(
    compiled,
    *,
    k,
    v,
    gate,
    beta,
    a_log,
    dt_bias,
    cu_seqlens,
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    scheduler_all,
    scheduler_summary,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    summary_words,
    tinv,
    state_h,
    state_m,
    state_x,
    seed,
    final_state,
    transition,
    pieces,
    heads_out,
    num_seqs,
    stream,
    batch_base=0,
    first_head_lo=0,
    tail_enable=None,
    l2norm_k=None,
) -> None:
    """Replay the summary chain: one crossing into the DSL for its launches."""
    buffers = locals()
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        int(batch_base),
        int(first_head_lo),
        *(buffers[name] for name in CHAIN_SUMMARY_BUFFERS),
        cuda.CUstream(int(stream)),
        tail_enable,
        l2norm_k,
    )
