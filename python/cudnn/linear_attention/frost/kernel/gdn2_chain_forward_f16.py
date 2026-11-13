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

"""One compiled launch for the GDN-2 chain forward: chain prologue, fused summary, fp32 state chain and prefill issued from
a single host, the way ``split_k.run_table`` launches plan, scan and walk.  Every kernel, its host and the tensor placeholder
each host was compiled with are the standalone modules' own; this host only sequences the four launches, so the kernels'
SASS is unchanged and the Python side crosses into the DSL once per call instead of four times.  A buffer that two hosts
read through different placeholder types is passed twice, once per type: H, M and X (the summary's and prefill's torch views,
the state chain's ``(1, HO, V, K)`` device views), ``cu_pieces`` (element alignment in the prologue, 8 bytes in the summary and
prefill) and ``beta`` (16 bytes in the prologue and summary, 4 in the prefill).  Compiled at ``--opt-level 2`` like the
summary and prefill it nests (their standalone builds' option); the state chain's standalone build takes the DSL default 3,
so its nested copy carries 16 more header instructions, one runtime division per CTA."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.buffers import DeviceView

from cudnn.frost.compiled_cache import compile_cached

from ..common.host import get_dtype
from ..common.piece_chain import chain_rows_per_cta, dtype_name, launch_state_chain
from . import gdn2_chain_prologue_f16, gdn2_prefill_f16, gdn2_summary_f16

CHAIN_FORWARD_BUFFERS = (
    "q",
    "k",
    "v",
    "gate",
    "beta",
    "beta",
    "w",
    "a_log",
    "dt_bias",
    "o",
    "cu_seqlens",
    "cu_pieces",
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
    "summary_words",
    "prefill_words",
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
    length_rule: cutlass.Constexpr[bool],
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
    beta_prefill: cute.Tensor,
    w: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    cu_pieces_main: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor,
    scheduler_all: cute.Tensor,
    scheduler_summary: cute.Tensor,
    scheduler_prefill: cute.Tensor,
    summary_words: cute.Tensor,
    prefill_words: cute.Tensor,
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
    tail_enable: Optional[cute.Tensor] = None,
    fold_rows: cutlass.Constexpr[int] = 0,
    fold_update: Optional[cute.Tensor] = None,
    fold_transition: Optional[cute.Tensor] = None,
    fold_seed: Optional[cute.Tensor] = None,
    fold_composed: Optional[cute.Tensor] = None,
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
    gdn2_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        prefill_cfg.max_active_clusters,
        heads_out,
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
        gate,
        beta,
        w,
        o,
        None,
        checkpoints,
        None,
        None,
        None,
        None,
        None,
        None,
        stream,
        checkpoint_after_block=prefill_cfg.checkpoint_after_block,
        checkpoint_box_elems=prefill_cfg.checkpoint_box_elems,
        tail=tail,
        first_head_lo=first_head_lo,
        cu_full=cu_full,
        batch_base=batch_base,
        tail_enable=tail_enable,
    )
    gdn2_summary_f16.host(
        summary_cfg,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        w,
        cu_pieces_main,
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
    gdn2_prefill_f16.host(
        prefill_cfg,
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta_prefill,
        w,
        cu_pieces_main,
        state_x_prefill,
        None,
        o,
        final_state,
        None,
        final_indices,
        work_items,
        main_count,
        scheduler_prefill,
        prefill_words,
        checkpoint_every_n,
        scale,
        stream,
    )


def build_chain_forward(
    *,
    q,
    k,
    v,
    gate,
    beta,
    w,
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
    summary_words,
    prefill_words,
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
    length_rule,
    log_gate,
    safe_gate,
    gate_lower_bound,
    use_qk_l2norm,
    use_beta_sigmoid,
    allow_neg_eigval,
    beta_guard,
    checkpoint_every_n_tokens,
    checkpoint_after_block=False,
    scale,
    chain_rows,
    device,
    num_sm,
    stream,
    tail=False,
    tail_enable=None,
    lead_buffers=(None,) * 4,
):
    """Compile (cached per static config: dtypes, heads, dims, gate flags and bound, beta guard, checkpoint and final-state
    presence, seed dtype, chain rows, device) the chain forward launch over the buffers of one plan; ``pieces``,
    ``heads_out`` and ``num_seqs`` are launch arguments.  The placeholders repeat the marks of the standalone modules'
    builds so every kernel compiles as it does there."""
    _HQ, DK = q.shape[1], q.shape[2]
    k.shape[1]
    _HV, DV = v.shape[1], v.shape[2]
    HO = gate.shape[1]
    if not safe_gate:
        a_log = None
        dt_bias = None
    io_dtype = get_dtype(q.dtype)
    gate_dtype = get_dtype(gate.dtype)
    state_dtype = get_dtype(state_x.dtype)
    has_seed = seed is not None
    seed_name = dtype_name(seed.dtype) if has_seed else "float32"
    gate_scale_log2 = float(gate_lower_bound) * gdn2_summary_f16.LOG2_E
    fold_update, fold_transition, fold_seed, fold_composed = lead_buffers
    fold_rows = chain_rows_per_cta(DV, DK, 1, gate.shape[1], num_sm) if fold_update is not None else 0
    key = (
        str(q.dtype),
        str(state_x.dtype),
        str(final_state.dtype) if final_state is not None else "none",
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(beta.dtype),
        str(w.dtype),
        seed_name,
        int(device),
        int(num_sm),
        DK,
        DV,
        int(unit_chunks),
        int(b_t),
        bool(length_rule),
        bool(log_gate),
        bool(safe_gate),
        float(gate_lower_bound),
        bool(use_qk_l2norm),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        bool(beta_guard),
        int(checkpoint_every_n_tokens) > 0,
        str(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else "none",
        bool(checkpoint_after_block),
        final_state is not None,
        has_seed,
        composed is not None,
        str(seed_indices.dtype) if seed_indices is not None else "none",
        str(final_indices.dtype) if final_indices is not None else "none",
        int(chain_rows),
        bool(tail),
        tail_enable is not None,
        tuple(str(buffer.dtype) if buffer is not None else None for buffer in lead_buffers),
        fold_rows,
    )
    if key not in chain_forward_cache:
        summary_cfg = gdn2_summary_f16.build_cfg(
            io_dtype,
            gate_dtype,
            use_initial_state=False,
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=gate_scale_log2,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            beta_guard=beta_guard,
            max_active_clusters=num_sm,
            d_k=DK,
            d_v=DV,
            log_gate=log_gate,
        )
        prefill_cfg = gdn2_prefill_f16.build_cfg(
            io_dtype,
            state_dtype,
            gate_dtype,
            use_initial_state=True,
            store_final_state=final_state is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            checkpoint_dtype=get_dtype(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else None,
            checkpoint_after_block=bool(checkpoint_after_block),
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=gate_scale_log2,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            beta_guard=beta_guard,
            max_active_clusters=num_sm,
            d_k=DK,
            d_v=DV,
            log_gate=log_gate,
        )

        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_summary_placeholder = from_dlpack(work_items_summary, assumed_align=16)
        work_items_summary_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
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
        chain_forward_cache[key] = compile_cached(
            chain_forward_host,
            int(unit_chunks),
            int(b_t),
            bool(length_rule),
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
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=2),
            from_dlpack(w, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(o, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_pieces, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(cu_pieces, assumed_align=8).mark_layout_dynamic(),
            from_dlpack(main_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(summary_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(main_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_count, assumed_align=4).mark_layout_dynamic(),
            work_items_placeholder,
            work_items_summary_placeholder,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_summary, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_prefill, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(prefill_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(state_h, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(state_m, assumed_align=16).mark_layout_dynamic(leading_dim=3),
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
            from_dlpack(state_x, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(final_state, assumed_align=16).mark_layout_dynamic(leading_dim=3) if final_state is not None else None,
            from_dlpack(final_indices, assumed_align=4).mark_layout_dynamic() if final_indices is not None else None,
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if checkpoints is not None else None,
            cuda.CUstream(int(stream)),
            from_dlpack(tail_enable, assumed_align=16).mark_layout_dynamic() if tail_enable is not None else None,
            fold_rows,
            *fold_placeholders,
            cache_key=repr((__name__, key)),
            symbol="frost_la_gdn2_chain_forward_f16",
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
    w,
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
    summary_words,
    prefill_words,
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
    lead_buffers=(None,) * 4,
) -> None:
    """Replay the chain forward: one crossing into the DSL for the four launches.  The plan validated the contract at build,
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
    "w",
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
    "summary_words",
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
    length_rule: cutlass.Constexpr[bool],
    summary_cfg: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    has_seed: cutlass.Constexpr[bool],
    emit_summary: cutlass.Constexpr[bool],
    tail: cutlass.Constexpr[bool],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    batch_base: cutlass.Int32,
    first_head_lo: cutlass.Int32,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
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
    summary_words: cute.Tensor,
    state_h_summary: cute.Tensor,
    state_m_summary: cute.Tensor,
    state_h_chain: cute.Tensor,
    state_m_chain: cute.Tensor,
    state_x_chain: cute.Tensor,
    seed: Optional[cute.Tensor],
    final_state: cute.Tensor,
    transition: Optional[cute.Tensor],
    stream: cuda.CUstream,
    tail_enable: Optional[cute.Tensor] = None,
) -> None:
    """The chain of the summary node: the chain forward without its main pass."""
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
    gdn2_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        summary_cfg.max_active_clusters,
        heads_out,
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
        gate,
        beta,
        w,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        stream,
        tail=tail,
        first_head_lo=first_head_lo,
        cu_full=cu_full,
        batch_base=batch_base,
        tail_enable=tail_enable,
    )
    gdn2_summary_f16.host(
        summary_cfg,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        w,
        cu_pieces,
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
    w,
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
    summary_words,
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
    length_rule,
    log_gate,
    safe_gate,
    gate_lower_bound,
    use_qk_l2norm,
    use_beta_sigmoid,
    allow_neg_eigval,
    beta_guard,
    chain_rows,
    device,
    num_sm,
    stream,
    tail=False,
    tail_enable=None,
):
    """Compile (cached per static config) the summary chain launch over the buffers of one plan; ``pieces``, ``heads_out``
    and ``num_seqs`` are launch arguments."""
    DK = k.shape[2]
    DV = v.shape[2]
    HO = gate.shape[1]
    if not safe_gate:
        a_log = None
        dt_bias = None
    io_dtype = get_dtype(k.dtype)
    gate_dtype = get_dtype(gate.dtype)
    has_seed = seed is not None
    emit_summary = transition is not None
    seed_name = dtype_name(seed.dtype) if has_seed else "float32"
    tail_name = dtype_name(final_state.dtype)
    summary_name = dtype_name(transition.dtype) if emit_summary else "float32"
    gate_scale_log2 = float(gate_lower_bound) * gdn2_summary_f16.LOG2_E
    key = (
        str(k.dtype),
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(beta.dtype),
        str(w.dtype),
        seed_name,
        tail_name,
        summary_name,
        int(device),
        int(num_sm),
        DK,
        DV,
        int(unit_chunks),
        int(b_t),
        bool(length_rule),
        bool(log_gate),
        bool(safe_gate),
        float(gate_lower_bound),
        bool(use_qk_l2norm),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        bool(beta_guard),
        has_seed,
        emit_summary,
        int(chain_rows),
        bool(tail),
        tail_enable is not None,
    )
    if key not in chain_summary_cache:
        summary_cfg = gdn2_summary_f16.build_cfg(
            io_dtype,
            gate_dtype,
            use_initial_state=False,
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=gate_scale_log2,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            beta_guard=beta_guard,
            max_active_clusters=num_sm,
            d_k=DK,
            d_v=DV,
            log_gate=log_gate,
        )
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        chain_summary_cache[key] = compile_cached(
            chain_summary_host,
            int(unit_chunks),
            int(b_t),
            bool(length_rule),
            summary_cfg,
            DV,
            DK,
            int(chain_rows),
            has_seed,
            emit_summary,
            bool(tail),
            cutlass.Int32(int(pieces)),
            cutlass.Int32(int(heads_out)),
            cutlass.Int32(int(num_seqs)),
            cutlass.Int32(0),
            cutlass.Int32(0),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(w, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_pieces, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(main_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(summary_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(main_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_count, assumed_align=4).mark_layout_dynamic(),
            work_items_placeholder,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_summary, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(state_h, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(state_m, assumed_align=16).mark_layout_dynamic(leading_dim=3),
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
            from_dlpack(tail_enable, assumed_align=16).mark_layout_dynamic() if tail_enable is not None else None,
            cache_key=repr((__name__, key)),
            symbol="frost_la_gdn2_chain_summary_f16",
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
    w,
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
    summary_words,
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
) -> None:
    """Replay the summary chain: one crossing into the DSL for the three launches."""
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
    )
