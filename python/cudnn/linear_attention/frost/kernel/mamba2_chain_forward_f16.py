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

"""One compiled launch for the Mamba-2 chain forward: chain prologue, summary pass, scalar state chain and prefill issued
from a single host, the way ``mamba2_warmup_forward_f16`` sequences the split-K table and the prefill.  The summary pass
is the recompute kernel over the piece slots from a zero state (every slot's end state H and decay product gamma), the
chain is the per-head axpy X_{j+1} = gamma_j X_j + H_j seeded with the initial state, and the main pass is the prefill
over the same slots seeded with X; the two passes share the prologue's descriptor arrays (the recompute reads K, V).
Every kernel, its host and the tensor placeholder each host was compiled with are the standalone modules' own; this host
only sequences the launches."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.buffers import DeviceView

from ..common.host import get_dtype
from ..common.piece_chain import CHAIN_THREADS, dtype_name, launch_scalar_state_chain
from . import mamba2_chain_prologue_f16, mamba2_prefill_f16, mamba2_recompute_f16

CHAIN_FORWARD_BUFFERS = (
    "q",
    "k",
    "v",
    "gate",
    "a",
    "dt_bias",
    "d",
    "z",
    "o",
    "o_pre",
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
    "words",
    "state_h",
    "transition",
    "state_x",
    "seed",
    "composed",
    "seed_indices",
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
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    batch_base: cutlass.Int32,
    first_head_lo: cutlass.Int32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    a: cute.Tensor,
    dt_bias: Optional[cute.Tensor],
    d: Optional[cute.Tensor],
    z: Optional[cute.Tensor],
    o: cute.Tensor,
    o_pre: Optional[cute.Tensor],
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
    words: cute.Tensor,
    state_h: cute.Tensor,
    transition: cute.Tensor,
    state_x: cute.Tensor,
    seed: Optional[cute.Tensor],
    composed: Optional[cute.Tensor],
    seed_indices: Optional[cute.Tensor],
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
        launch_scalar_state_chain(
            heads_out,
            dim_v,
            dim_k,
            fold_rows,
            cutlass.Int32(fold_update.shape[0]),
            fold_seed is not None,
            False,
            True,
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
    mamba2_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        prefill_cfg.max_active_clusters,
        heads_out,
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
        words,
        q,
        k,
        v,
        o,
        checkpoints,
        z,
        o_pre,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        stream,
        prefill_cfg.checkpoint_after_block,
        tail=tail,
        first_head_lo=first_head_lo,
        cu_full=cu_full,
        batch_base=batch_base,
        tail_enable=tail_enable,
    )
    mamba2_recompute_f16.host(
        summary_cfg,
        1,
        2,
        4,
        k,
        v,
        gate,
        a,
        dt_bias,
        cu_pieces,
        None,
        None,
        state_h,
        None,
        None,
        work_items_summary,
        summary_count,
        scheduler_summary,
        transition,
        cutlass.Int32(0),
        cutlass.Int32(0),
        dt_min,
        dt_max,
        words,
        stream,
    )
    launch_scalar_state_chain(
        heads_out,
        dim_v,
        dim_k,
        chain_rows,
        pieces,
        has_seed,
        False,
        False,
        True,
        num_seqs,
        state_h,
        transition,
        state_x,
        seed,
        None,
        main_rows,
        seed_indices,
        composed,
        stream,
        mTailEnable=tail_enable,
        first_head_lo=first_head_lo,
    )
    mamba2_prefill_f16.host(
        prefill_cfg,
        q,
        k,
        v,
        gate,
        a,
        dt_bias,
        d,
        z,
        o,
        o_pre,
        cu_pieces,
        state_x,
        None,
        final_state,
        None,
        final_indices,
        work_items,
        main_count,
        scheduler_prefill,
        None,
        checkpoint_every_n,
        dt_min,
        dt_max,
        words,
        stream,
    )


def build_configs(
    io_dtype,
    *,
    store_final_state,
    enable_checkpoints,
    dt_softplus,
    has_d,
    d_per_channel,
    has_z,
    store_pre_gate,
    max_active_clusters,
    d_k,
    d_v,
    checkpoint_dtype=None,
    checkpoint_after_block=False,
):
    """The summary pass's recompute config (zero state in, every slot's end state out; the prefill's K, V descriptor
    arrays) and the main pass's config (seeded from the fp32 chain state)."""
    summary_cfg = mamba2_recompute_f16.build_cfg(
        io_dtype,
        cutlass.Float32,
        max_active_clusters=max_active_clusters,
        use_initial_state=False,
        store_final_state=True,
        enable_checkpoints=False,
        dt_softplus=dt_softplus,
        d_k=d_k,
        d_v=d_v,
    )
    prefill_cfg = mamba2_prefill_f16.build_cfg(
        io_dtype,
        cutlass.Float32,
        max_active_clusters=max_active_clusters,
        use_initial_state=True,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        checkpoint_dtype=checkpoint_dtype,
        checkpoint_after_block=checkpoint_after_block,
        dt_softplus=dt_softplus,
        has_d=has_d,
        d_per_channel=d_per_channel,
        has_z=has_z,
        store_pre_gate=store_pre_gate,
        d_k=d_k,
        d_v=d_v,
    )
    return summary_cfg, prefill_cfg


def build_chain_forward(
    *,
    q,
    k,
    v,
    gate,
    a,
    dt_bias,
    d,
    z,
    o,
    o_pre,
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
    words,
    state_h,
    transition,
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
    dt_softplus,
    dt_min,
    dt_max,
    checkpoint_every_n_tokens,
    checkpoint_after_block=False,
    chain_rows,
    device,
    num_sm,
    stream,
    tail=False,
    tail_enable=None,
    fold_buffers=(None,) * 4,
):
    """Compile (cached per static config) the chain forward launch over the buffers of one plan; ``pieces``,
    ``heads_out`` and ``num_seqs`` are launch arguments."""
    _HQ, DK = q.shape[1], q.shape[2]
    _HV, DV = v.shape[1], v.shape[2]
    io_dtype = get_dtype(q.dtype)
    has_seed = seed is not None
    seed_name = dtype_name(seed.dtype) if has_seed else "float32"
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(a.dtype),
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(d.dtype) if d is not None else "none",
        len(d.shape) if d is not None else 0,
        z is not None,
        o_pre is not None,
        str(cu_seqlens.dtype),
        str(final_state.dtype) if final_state is not None else "none",
        seed_name,
        int(device),
        int(num_sm),
        DK,
        DV,
        int(unit_chunks),
        int(b_t),
        bool(length_rule),
        bool(dt_softplus),
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
        tuple(str(buffer.dtype) if buffer is not None else None for buffer in fold_buffers),
    )
    if key not in chain_forward_cache:
        summary_cfg, prefill_cfg = build_configs(
            io_dtype,
            store_final_state=final_state is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            checkpoint_dtype=get_dtype(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else None,
            checkpoint_after_block=bool(checkpoint_after_block),
            dt_softplus=dt_softplus,
            has_d=d is not None,
            d_per_channel=d is not None and len(d.shape) == 2,
            has_z=z is not None,
            store_pre_gate=o_pre is not None,
            max_active_clusters=num_sm,
            d_k=DK,
            d_v=DV,
        )
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_summary_placeholder = from_dlpack(work_items_summary, assumed_align=16)
        work_items_summary_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        d_placeholder = None
        if d is not None:
            d_placeholder = from_dlpack(d, assumed_align=4).mark_layout_dynamic(leading_dim=len(d.shape) - 1)
        fold_update, fold_transition, fold_seed, fold_composed = fold_buffers
        fold_rows = 4 * CHAIN_THREADS // DK if fold_update is not None else 0
        fold_placeholders = [None] * 4
        if fold_update is not None:
            fold_shape = (1, int(heads_out), DV, DK)
            fold_placeholders = [
                from_dlpack(DeviceView(256, fold_shape, "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(DeviceView(256, fold_shape[:2], "float32", int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=1),
                (
                    from_dlpack(DeviceView(256, fold_shape, dtype_name(fold_seed.dtype), int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=3)
                    if fold_seed is not None
                    else None
                ),
                from_dlpack(DeviceView(256, fold_shape, dtype_name(fold_composed.dtype), int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=3),
            ]
        chain_forward_cache[key] = cute.compile(
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
            cutlass.Float32(float(dt_min)),
            cutlass.Float32(float(dt_max)),
            cutlass.Int32(0),
            cutlass.Int32(0),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(a, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            d_placeholder,
            from_dlpack(z, assumed_align=16).mark_layout_dynamic(leading_dim=2) if z is not None else None,
            from_dlpack(o, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(o_pre, assumed_align=16).mark_layout_dynamic(leading_dim=2) if o_pre is not None else None,
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
            from_dlpack(words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(state_h, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(transition, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(state_x, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(seed, assumed_align=4).mark_layout_dynamic(leading_dim=3) if has_seed else None,
            from_dlpack(composed, assumed_align=16).mark_layout_dynamic(leading_dim=3) if composed is not None else None,
            from_dlpack(seed_indices, assumed_align=4).mark_layout_dynamic() if seed_indices is not None else None,
            from_dlpack(final_state, assumed_align=16).mark_layout_dynamic(leading_dim=3) if final_state is not None else None,
            from_dlpack(final_indices, assumed_align=4).mark_layout_dynamic() if final_indices is not None else None,
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if int(checkpoint_every_n_tokens) > 0 else None,
            cuda.CUstream(int(stream)),
            from_dlpack(tail_enable, assumed_align=16).mark_layout_dynamic() if tail_enable is not None else None,
            fold_rows,
            *fold_placeholders,
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
    a,
    dt_bias,
    d,
    z,
    o,
    o_pre,
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
    words,
    state_h,
    transition,
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
    dt_min,
    dt_max,
    stream,
    batch_base=0,
    first_head_lo=0,
    tail_enable=None,
    fold_buffers=(None,) * 4,
) -> None:
    """Replay the chain forward: one crossing into the DSL for the four launches."""
    buffers = dict(locals())
    buffers["checkpoints"] = checkpoints if int(checkpoint_every_n_tokens) > 0 else None
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        int(checkpoint_every_n_tokens),
        float(dt_min),
        float(dt_max),
        int(batch_base),
        int(first_head_lo),
        *(buffers[name] for name in CHAIN_FORWARD_BUFFERS),
        cuda.CUstream(int(stream)),
        tail_enable,
        *fold_buffers,
    )


CHAIN_SUMMARY_BUFFERS = (
    "k",
    "v",
    "gate",
    "a",
    "dt_bias",
    "cu_seqlens",
    "cu_pieces",
    "main_rows",
    "summary_rows",
    "main_count",
    "summary_count",
    "work_items",
    "work_items_summary",
    "scheduler",
    "words",
    "state_h",
    "slot_transition",
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
    tail: cutlass.Constexpr[bool],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    batch_base: cutlass.Int32,
    first_head_lo: cutlass.Int32,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    a: cute.Tensor,
    dt_bias: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor,
    scheduler: cute.Tensor,
    words: cute.Tensor,
    state_h: cute.Tensor,
    slot_transition: cute.Tensor,
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
                transition.iterator + batch_base * transition.stride[0], cute.make_layout((num_seqs, transition.shape[1]), stride=transition.stride)
            )
    mamba2_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        summary_cfg.max_active_clusters,
        heads_out,
        cutlass.Int32(0),
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        scheduler,
        words,
        k,
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
    mamba2_recompute_f16.host(
        summary_cfg,
        1,
        2,
        4,
        k,
        v,
        gate,
        a,
        dt_bias,
        cu_pieces,
        None,
        None,
        state_h,
        None,
        None,
        work_items,
        main_count,
        scheduler,
        slot_transition,
        cutlass.Int32(0),
        cutlass.Int32(0),
        dt_min,
        dt_max,
        words,
        stream,
    )
    head_floor = None
    if cutlass.const_expr(tail):
        head_floor = first_head_lo
    launch_scalar_state_chain(
        heads_out,
        dim_v,
        dim_k,
        chain_rows,
        pieces,
        has_seed,
        False,
        True,
        False,
        num_seqs,
        state_h,
        slot_transition,
        None,
        seed,
        final_state,
        main_rows,
        None,
        None,
        stream,
        mTailEnable=tail_enable,
        mTransitionOut=transition,
        first_head_lo=head_floor,
    )


def build_chain_summary(
    *,
    k,
    v,
    gate,
    a,
    dt_bias,
    cu_seqlens,
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    work_items_summary,
    scheduler,
    words,
    state_h,
    slot_transition,
    seed,
    final_state,
    transition,
    pieces,
    heads_out,
    num_seqs,
    unit_chunks,
    b_t,
    length_rule,
    dt_softplus,
    dt_min,
    dt_max,
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
    io_dtype = get_dtype(k.dtype)
    has_seed = seed is not None
    key = (
        str(k.dtype),
        str(gate.dtype),
        str(a.dtype),
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        str(final_state.dtype),
        str(seed.dtype) if has_seed else "none",
        str(transition.dtype) if transition is not None else "none",
        int(device),
        int(num_sm),
        DK,
        DV,
        int(unit_chunks),
        int(b_t),
        bool(length_rule),
        bool(dt_softplus),
        int(chain_rows),
        bool(tail),
        tail_enable is not None,
    )
    if key not in chain_summary_cache:
        summary_cfg = mamba2_recompute_f16.build_cfg(
            io_dtype,
            cutlass.Float32,
            max_active_clusters=num_sm,
            use_initial_state=False,
            store_final_state=True,
            enable_checkpoints=False,
            dt_softplus=dt_softplus,
            d_k=DK,
            d_v=DV,
        )
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_summary_placeholder = from_dlpack(work_items_summary, assumed_align=16)
        work_items_summary_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        chain_summary_cache[key] = cute.compile(
            chain_summary_host,
            int(unit_chunks),
            int(b_t),
            bool(length_rule),
            summary_cfg,
            DV,
            DK,
            int(chain_rows),
            has_seed,
            bool(tail),
            cutlass.Int32(int(pieces)),
            cutlass.Int32(int(heads_out)),
            cutlass.Int32(int(num_seqs)),
            cutlass.Float32(float(dt_min)),
            cutlass.Float32(float(dt_max)),
            cutlass.Int32(0),
            cutlass.Int32(0),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(a, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_pieces, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(main_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(summary_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(main_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_count, assumed_align=4).mark_layout_dynamic(),
            work_items_placeholder,
            work_items_summary_placeholder,
            from_dlpack(scheduler, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(state_h, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(slot_transition, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(seed, assumed_align=4).mark_layout_dynamic(leading_dim=3) if has_seed else None,
            from_dlpack(final_state, assumed_align=4).mark_layout_dynamic(leading_dim=3),
            from_dlpack(transition, assumed_align=4).mark_layout_dynamic(leading_dim=1) if transition is not None else None,
            cuda.CUstream(int(stream)),
            from_dlpack(tail_enable, assumed_align=16).mark_layout_dynamic() if tail_enable is not None else None,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return chain_summary_cache[key]


def run_chain_summary(
    compiled,
    *,
    k,
    v,
    gate,
    a,
    dt_bias,
    cu_seqlens,
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    work_items_summary,
    scheduler,
    words,
    state_h,
    slot_transition,
    seed,
    final_state,
    transition,
    pieces,
    heads_out,
    num_seqs,
    dt_min,
    dt_max,
    stream,
    batch_base=0,
    first_head_lo=0,
    tail_enable=None,
) -> None:
    """Replay the summary chain: one crossing into the DSL for the three launches."""
    buffers = dict(locals())
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        float(dt_min),
        float(dt_max),
        int(batch_base),
        int(first_head_lo),
        *(buffers[name] for name in CHAIN_SUMMARY_BUFFERS),
        cuda.CUstream(int(stream)),
        tail_enable,
    )
