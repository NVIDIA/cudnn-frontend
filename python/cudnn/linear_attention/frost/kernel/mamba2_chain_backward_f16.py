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

"""One compiled launch for the Mamba-2 chain backward: chain prologue, the forward summary when the graph carries no
checkpoint series (the forward chain's own three steps: the recompute kernel over the summary table for every slot's end
state H and decay product from zero, the scalar chain X_{j+1} = gamma_j X_j + H_j seeded with the initial state, then the
recompute kernel over the main table seeded with X writing the series), the backward summary (the summary kernel: every
slot's start gradient G_j from a zero end gradient, and the decay products when the series was passed back; the forward
summary writes them otherwise), the reverse
scalar chain dX_{j-1} = gamma_j dX_j + G_j seeded with d_final_state, and the bprop over the pieces with S from the
series and dS_end = dX; the two bprop passes share the prologue's descriptor arrays, the summary pass has its own.
Every kernel, its host and the tensor placeholder each host was compiled with are the standalone modules' own; this host
only sequences the launches."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.buffers import DeviceView

from ..common import gate_bwd, head_reduce
from ..common.host import get_dtype
from ..common.piece_chain import CHAIN_THREADS, dtype_name, launch_scalar_state_chain
from . import mamba2_bprop_f16, mamba2_bprop_summary_f16, mamba2_chain_prologue_f16, mamba2_prefill_f16, mamba2_recompute_f16

chain_backward_cache = {}


@cute.jit
def chain_backward_host(
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    regenerate: cutlass.Constexpr[bool],
    coarse: cutlass.Constexpr[bool],
    recompute_summary_cfg: cutlass.Constexpr,
    recompute_series_cfg: cutlass.Constexpr,
    bprop_summary_cfg: cutlass.Constexpr,
    bprop_cfg: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    has_seed: cutlass.Constexpr[bool],
    has_dfs: cutlass.Constexpr[bool],
    tail: cutlass.Constexpr[bool],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    seed_every_n_tokens: cutlass.Int32,
    batch_base: cutlass.Int32,
    first_head_lo: cutlass.Int32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    gate: cute.Tensor,
    a: cute.Tensor,
    dt_bias: Optional[cute.Tensor],
    d: Optional[cute.Tensor],
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor,
    scheduler_all: cute.Tensor,
    scheduler_summary_fwd: cute.Tensor,
    scheduler_summary_bwd: cute.Tensor,
    scheduler_bprop: cute.Tensor,
    scheduler_regenerate: cute.Tensor,
    prefill_words: Optional[cute.Tensor],
    bprop_words: cute.Tensor,
    series: cute.Tensor,
    state_h: Optional[cute.Tensor],
    transition: cute.Tensor,
    state_x: Optional[cute.Tensor],
    seed_checkpoints: Optional[cute.Tensor],
    state_g: cute.Tensor,
    state_dx: cute.Tensor,
    seed: Optional[cute.Tensor],
    composed: Optional[cute.Tensor],
    dfs: Optional[cute.Tensor],
    dcomposed: Optional[cute.Tensor],
    dstate0: Optional[cute.Tensor],
    exit_dots: cute.Tensor,
    dot_q: cute.Tensor,
    dot_k: cute.Tensor,
    skip_bwd_part: Optional[cute.Tensor],
    exit_slots: cute.Tensor,
    exit_base: cute.Tensor,
    seed_row_base: Optional[cute.Tensor],
    stream: cuda.CUstream,
    tail_enable: Optional[cute.Tensor] = None,
    fold_rows: cutlass.Constexpr[int] = 0,
    fold_update: Optional[cute.Tensor] = None,
    fold_transition: Optional[cute.Tensor] = None,
    fold_seed: Optional[cute.Tensor] = None,
    fold_composed: Optional[cute.Tensor] = None,
    gate_skip_channels: cutlass.Constexpr[int] = 1,
    zgate_do: Optional[cute.Tensor] = None,
    zgate_z: Optional[cute.Tensor] = None,
    zgate_o_pre: Optional[cute.Tensor] = None,
    zgate_dz: Optional[cute.Tensor] = None,
    gate_bwd_gate: Optional[cute.Tensor] = None,
    gate_bwd_a: Optional[cute.Tensor] = None,
    gate_bwd_dt_bias: Optional[cute.Tensor] = None,
    gate_bwd_dgate: Optional[cute.Tensor] = None,
    gate_part_a: Optional[cute.Tensor] = None,
    gate_part_dt: Optional[cute.Tensor] = None,
    gate_skip_part: Optional[cute.Tensor] = None,
    d_a: Optional[cute.Tensor] = None,
    d_dt_bias: Optional[cute.Tensor] = None,
    d_d: Optional[cute.Tensor] = None,
    reduce_dq_src: Optional[cute.Tensor] = None,
    reduce_dq: Optional[cute.Tensor] = None,
    reduce_dk_src: Optional[cute.Tensor] = None,
    reduce_dk: Optional[cute.Tensor] = None,
) -> None:
    if cutlass.const_expr(fold_update is not None):
        if cutlass.const_expr(fold_seed is not None):
            fold_seed = cute.make_tensor(
                fold_seed.iterator + (cutlass.Int32(fold_seed.shape[0]) - cutlass.Int32(1)) * fold_seed.stride[0],
                cute.make_layout((cutlass.Int32(1), fold_seed.shape[1], fold_seed.shape[2], fold_seed.shape[3]), stride=fold_seed.stride),
            )
        launch_scalar_state_chain(
            heads_out,
            dim_v,
            dim_k,
            fold_rows,
            cutlass.Int32(fold_update.shape[0]),
            fold_seed is not None,
            True,
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
    if cutlass.const_expr(zgate_z is not None):
        zgate_heads = cutlass.Int32(cutlass.Int64(zgate_z.shape[1]))
        zgate_vectors = cutlass.Int32(cutlass.Int64(zgate_z.shape[0]) * cutlass.Int64(bprop_cfg.d_v // (16 // (bprop_cfg.io_dtype.width // 8))))
        zgate_cap = cutlass.Int32(4096) // zgate_heads
        zgate_cap = zgate_cap if zgate_cap > cutlass.Int32(1) else cutlass.Int32(1)
        zgate_blocks = (zgate_vectors + cutlass.Int32(gate_bwd.MAMBA2_ZGATE_THREADS - 1)) // cutlass.Int32(gate_bwd.MAMBA2_ZGATE_THREADS)
        zgate_blocks = zgate_blocks if zgate_blocks < zgate_cap else zgate_cap
        gate_bwd.mamba2_zgate_bwd_launch(
            zgate_do, zgate_z, zgate_o_pre, do, zgate_dz, zgate_vectors, bprop_cfg.d_v, bprop_cfg.io_dtype, zgate_heads, zgate_blocks, stream
        )
    cu_full = None
    if cutlass.const_expr(tail):
        cu_full = cu_seqlens
        cu_seqlens = cute.make_tensor(cu_seqlens.iterator + batch_base, cute.make_layout((num_seqs + cutlass.Int32(1),), stride=(1,)))
        if cutlass.const_expr(seed is not None):
            seed = cute.make_tensor(
                seed.iterator + batch_base * seed.stride[0], cute.make_layout((num_seqs, seed.shape[1], seed.shape[2], seed.shape[3]), stride=seed.stride)
            )
        if cutlass.const_expr(dfs is not None):
            dfs = cute.make_tensor(
                dfs.iterator + batch_base * dfs.stride[0], cute.make_layout((num_seqs, dfs.shape[1], dfs.shape[2], dfs.shape[3]), stride=dfs.stride)
            )
        if cutlass.const_expr(dstate0 is not None):
            dstate0 = cute.make_tensor(
                dstate0.iterator + batch_base * dstate0.stride[0],
                cute.make_layout((num_seqs, dstate0.shape[1], dstate0.shape[2], dstate0.shape[3]), stride=dstate0.stride),
            )
    mamba2_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        bprop_cfg.max_active_clusters,
        heads_out,
        cutlass.Int32(b_t),
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        scheduler_all,
        prefill_words,
        q,
        k,
        v,
        None,
        series,
        None,
        None,
        bprop_words,
        do,
        dq,
        dk,
        dv,
        exit_slots,
        exit_base,
        stream,
        tail=tail,
        first_head_lo=first_head_lo,
        cu_full=cu_full,
        batch_base=batch_base,
        seed_every_n=seed_every_n_tokens,
        seed_row_base=seed_row_base,
        tail_enable=tail_enable,
    )
    if cutlass.const_expr(regenerate and not coarse):
        # the forward chain's own first two steps: zero-state summaries and the scalar chain
        mamba2_recompute_f16.host(
            recompute_summary_cfg,
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
            scheduler_summary_fwd,
            transition,
            cutlass.Int32(0),
            cutlass.Int32(0),
            dt_min,
            dt_max,
            prefill_words,
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
            None,
            composed,
            stream,
            mTailEnable=tail_enable,
            first_head_lo=first_head_lo,
        )
    if cutlass.const_expr(regenerate):
        # the series pass: every piece seeded from its chain state, or under coarse from its checkpoint row
        mamba2_recompute_f16.host(
            recompute_series_cfg,
            1,
            2,
            4,
            k,
            v,
            gate,
            a,
            dt_bias,
            cu_pieces,
            seed_checkpoints if coarse else state_x,
            None,
            None,
            None,
            None,
            work_items,
            main_count,
            scheduler_regenerate,
            None,
            cutlass.Int32(b_t),
            seed_every_n_tokens if coarse else cutlass.Int32(0),
            dt_min,
            dt_max,
            prefill_words,
            stream,
            seed_row_base,
        )
    mamba2_bprop_summary_f16.host(
        bprop_summary_cfg,
        0,
        3,
        q,
        do,
        gate,
        a,
        dt_bias,
        cu_pieces,
        None,
        None,
        state_g,
        work_items_summary,
        summary_count,
        scheduler_summary_bwd,
        None if (regenerate and not coarse) else transition,
        dt_min,
        dt_max,
        bprop_words,
        stream,
    )
    launch_scalar_state_chain(
        heads_out,
        dim_v,
        dim_k,
        chain_rows,
        pieces,
        has_dfs,
        True,
        False,
        True,
        num_seqs,
        state_g,
        transition,
        state_dx,
        dfs,
        None,
        main_rows,
        None,
        dcomposed,
        stream,
        mTailEnable=tail_enable,
        first_head_lo=first_head_lo,
    )
    mamba2_bprop_f16.host(
        bprop_cfg,
        q,
        k,
        v,
        do,
        gate,
        a,
        dt_bias,
        d,
        dq,
        dk,
        dv,
        series,
        exit_dots,
        dot_q,
        dot_k,
        skip_bwd_part,
        exit_base,
        cu_pieces,
        state_dx,
        None,
        dstate0,
        work_items,
        main_count,
        scheduler_bprop,
        None,
        dt_min,
        dt_max,
        bprop_words,
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
    if cutlass.const_expr(gate_bwd_dgate is not None):
        gate_bwd.mamba2_gate_bwd_launch(
            dot_q,
            dot_k,
            exit_dots,
            gate_bwd_gate,
            gate_bwd_a,
            gate_bwd_dt_bias,
            exit_slots,
            gate_bwd_dgate,
            gate_part_a,
            gate_part_dt,
            gate_skip_part,
            d_a,
            d_dt_bias,
            d_d,
            cutlass.Int32(cutlass.Int64(exit_dots.shape[1])),
            cutlass.Int32(cutlass.Int64(exit_dots.shape[0])),
            dt_min,
            dt_max,
            b_t,
            bprop_cfg.dt_softplus,
            gate_skip_channels,
            stream,
        )


def build_configs(
    io_dtype, *, regenerate, coarse, seed_dtype, use_dstate0, dt_softplus, has_d, d_per_channel, skip_bwd_partials, max_active_clusters, d_k, d_v
):
    """The configs of the chain backward: the two recompute passes of the series regeneration (the zero-state summary and
    the series pass seeded from the chain states, the forward chain's own pair; under ``coarse`` the series pass alone,
    seeded from the checkpoint rows of ``seed_dtype``), the backward summary and the main bprop."""
    recompute_summary_cfg = None
    recompute_series_cfg = None
    if regenerate and not coarse:
        recompute_summary_cfg = mamba2_recompute_f16.build_cfg(
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
    if regenerate:
        recompute_series_cfg = mamba2_recompute_f16.build_cfg(
            io_dtype,
            seed_dtype if coarse else cutlass.Float32,
            max_active_clusters=max_active_clusters,
            use_initial_state=True,
            store_final_state=False,
            enable_checkpoints=True,
            dt_softplus=dt_softplus,
            seed_checkpoints=coarse,
            d_k=d_k,
            d_v=d_v,
        )
    bprop_summary_cfg = mamba2_bprop_summary_f16.build_cfg(
        io_dtype,
        cutlass.Float32,
        max_active_clusters=max_active_clusters,
        use_dstate_in=False,
        dt_softplus=dt_softplus,
        d_k=d_k,
        d_v=d_v,
    )
    bprop_cfg = mamba2_bprop_f16.build_cfg(
        io_dtype,
        cutlass.Float32,
        max_active_clusters=max_active_clusters,
        use_dstate_in=True,
        use_dstate0=use_dstate0,
        dt_softplus=dt_softplus,
        has_d=has_d,
        d_per_channel=d_per_channel,
        skip_bwd_partials=skip_bwd_partials,
        d_k=d_k,
        d_v=d_v,
    )
    return recompute_summary_cfg, recompute_series_cfg, bprop_summary_cfg, bprop_cfg


def build_chain_backward(
    *,
    q,
    k,
    v,
    do,
    gate,
    a,
    dt_bias,
    d,
    dq,
    dk,
    dv,
    cu_seqlens,
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    work_items_summary,
    scheduler_all,
    scheduler_summary_fwd,
    scheduler_summary_bwd,
    scheduler_bprop,
    scheduler_regenerate,
    prefill_words,
    bprop_words,
    series,
    state_h,
    transition,
    state_x,
    state_g,
    state_dx,
    seed,
    composed=None,
    dfs,
    dcomposed=None,
    dstate0,
    exit_dots,
    dot_q,
    dot_k,
    skip_bwd_part,
    exit_slots,
    exit_base,
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
    seed_checkpoints=None,
    seed_every_n_tokens=0,
    tail=False,
    seed_row_base=None,
    tail_enable=None,
    post_buffers=(None,) * 18,
    fold_buffers=(None,) * 4,
):
    """Compile (cached per static config) the chain backward launch over the buffers of one plan; ``pieces``,
    ``heads_out`` and ``num_seqs`` are launch arguments; ``prefill_words`` is None when the graph passes the series back
    at the chunk cadence; ``seed_checkpoints`` (io or fp32 rows ``seed_every_n_tokens`` apart) seeds the series pass per
    piece instead of the forward summary and the scalar chain."""
    _HQ, DK = q.shape[1], q.shape[2]
    _HV, DV = v.shape[1], v.shape[2]
    io_dtype = get_dtype(q.dtype)
    regenerate = prefill_words is not None
    coarse = seed_checkpoints is not None
    has_seed = seed is not None
    has_dfs = dfs is not None
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(a.dtype),
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(d.dtype) if d is not None else "none",
        len(d.shape) if d is not None else 0,
        str(cu_seqlens.dtype),
        str(series.dtype),
        dtype_name(seed.dtype) if has_seed else "none",
        dtype_name(dfs.dtype) if has_dfs else "none",
        composed is not None,
        dcomposed is not None,
        str(dstate0.dtype) if dstate0 is not None else "none",
        int(device),
        int(num_sm),
        DK,
        DV,
        int(unit_chunks),
        int(b_t),
        bool(length_rule),
        bool(dt_softplus),
        regenerate,
        bool(coarse),
        str(seed_checkpoints.dtype) if coarse else "none",
        int(chain_rows),
        skip_bwd_part is not None,
        bool(tail),
        seed_row_base is not None,
        tail_enable is not None,
        tuple((str(buffer.dtype), len(buffer.shape)) if buffer is not None else None for buffer in post_buffers),
        tuple(str(buffer.dtype) if buffer is not None else None for buffer in fold_buffers),
    )
    if key not in chain_backward_cache:
        recompute_summary_cfg, recompute_series_cfg, bprop_summary_cfg, bprop_cfg = build_configs(
            io_dtype,
            regenerate=regenerate,
            coarse=coarse,
            seed_dtype=get_dtype(seed_checkpoints.dtype) if coarse else cutlass.Float32,
            use_dstate0=dstate0 is not None,
            dt_softplus=dt_softplus,
            has_d=d is not None,
            d_per_channel=d is not None and len(d.shape) == 2,
            skip_bwd_partials=skip_bwd_part is not None,
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
        gate_skip_channels = int(post_buffers[13].shape[1]) if post_buffers[13] is not None and len(post_buffers[13].shape) == 2 else 1
        post_placeholders = [
            from_dlpack(buffer, assumed_align=16).mark_layout_dynamic(leading_dim=2) if buffer is not None else None for buffer in post_buffers[:4]
        ] + [
            from_dlpack(buffer, assumed_align=4).mark_layout_dynamic(leading_dim=len(buffer.shape) - 1) if buffer is not None else None
            for buffer in post_buffers[4:]
        ]
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
        chain_backward_cache[key] = cute.compile(
            chain_backward_host,
            int(unit_chunks),
            int(b_t),
            bool(length_rule),
            regenerate,
            coarse,
            recompute_summary_cfg,
            recompute_series_cfg,
            bprop_summary_cfg,
            bprop_cfg,
            DV,
            DK,
            int(chain_rows),
            has_seed,
            has_dfs,
            bool(tail),
            cutlass.Int32(int(pieces)),
            cutlass.Int32(int(heads_out)),
            cutlass.Int32(int(num_seqs)),
            cutlass.Float32(float(dt_min)),
            cutlass.Float32(float(dt_max)),
            cutlass.Int32(int(seed_every_n_tokens)),
            cutlass.Int32(0),
            cutlass.Int32(0),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(a, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            d_placeholder,
            from_dlpack(dq, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dk, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dv, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_pieces, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(main_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(summary_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(main_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_count, assumed_align=4).mark_layout_dynamic(),
            work_items_placeholder,
            work_items_summary_placeholder,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_summary_fwd, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_summary_bwd, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_bprop, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_regenerate, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(prefill_words, assumed_align=128).mark_layout_dynamic() if regenerate else None,
            from_dlpack(bprop_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(series, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(state_h, assumed_align=16).mark_layout_dynamic(leading_dim=3) if regenerate and not coarse else None,
            from_dlpack(transition, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(state_x, assumed_align=16).mark_layout_dynamic(leading_dim=3) if regenerate and not coarse else None,
            from_dlpack(seed_checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if coarse else None,
            from_dlpack(state_g, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(state_dx, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(seed, assumed_align=4).mark_layout_dynamic(leading_dim=3) if has_seed else None,
            from_dlpack(composed, assumed_align=16).mark_layout_dynamic(leading_dim=3) if composed is not None else None,
            from_dlpack(dfs, assumed_align=4).mark_layout_dynamic(leading_dim=3) if has_dfs else None,
            from_dlpack(dcomposed, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dcomposed is not None else None,
            from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate0 is not None else None,
            from_dlpack(exit_dots, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dot_q, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dot_k, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(skip_bwd_part, assumed_align=4).mark_layout_dynamic(leading_dim=len(skip_bwd_part.shape) - 1) if skip_bwd_part is not None else None,
            from_dlpack(exit_slots, assumed_align=8).mark_layout_dynamic(leading_dim=1),
            from_dlpack(exit_base, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(seed_row_base, assumed_align=4).mark_layout_dynamic() if seed_row_base is not None else None,
            cuda.CUstream(int(stream)),
            from_dlpack(tail_enable, assumed_align=16).mark_layout_dynamic() if tail_enable is not None else None,
            fold_rows,
            *fold_placeholders,
            gate_skip_channels,
            *post_placeholders,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return chain_backward_cache[key]


def run_chain_backward(
    compiled,
    *,
    q,
    k,
    v,
    do,
    gate,
    a,
    dt_bias,
    d,
    dq,
    dk,
    dv,
    cu_seqlens,
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    work_items_summary,
    scheduler_all,
    scheduler_summary_fwd,
    scheduler_summary_bwd,
    scheduler_bprop,
    scheduler_regenerate,
    prefill_words,
    bprop_words,
    series,
    state_h,
    transition,
    state_x,
    state_g,
    state_dx,
    seed,
    composed=None,
    dfs,
    dcomposed=None,
    dstate0,
    exit_dots,
    dot_q,
    dot_k,
    skip_bwd_part,
    exit_slots,
    exit_base,
    pieces,
    heads_out,
    num_seqs,
    dt_min,
    dt_max,
    stream,
    seed_checkpoints=None,
    seed_every_n_tokens=0,
    batch_base=0,
    first_head_lo=0,
    seed_row_base=None,
    tail_enable=None,
    post_buffers=(None,) * 18,
    fold_buffers=(None,) * 4,
) -> None:
    """Replay the chain backward: one crossing into the DSL for its launches."""
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        float(dt_min),
        float(dt_max),
        int(seed_every_n_tokens),
        int(batch_base),
        int(first_head_lo),
        q,
        k,
        v,
        do,
        gate,
        a,
        dt_bias,
        d,
        dq,
        dk,
        dv,
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        scheduler_all,
        scheduler_summary_fwd,
        scheduler_summary_bwd,
        scheduler_bprop,
        scheduler_regenerate,
        prefill_words,
        bprop_words,
        series,
        state_h,
        transition,
        state_x,
        seed_checkpoints,
        state_g,
        state_dx,
        seed,
        composed,
        dfs,
        dcomposed,
        dstate0,
        exit_dots,
        dot_q,
        dot_k,
        skip_bwd_part,
        exit_slots,
        exit_base,
        seed_row_base,
        cuda.CUstream(int(stream)),
        tail_enable,
        *fold_buffers,
        *post_buffers,
    )


CHAIN_SUMMARY_BACKWARD_BUFFERS = (
    "q",
    "do",
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
    "state_g",
    "slot_transition",
    "dstate_in",
    "dstate0",
    "transition",
)

chain_summary_backward_cache = {}


@cute.jit
def chain_summary_backward_host(
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    bprop_summary_cfg: cutlass.Constexpr,
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
    q: cute.Tensor,
    do: cute.Tensor,
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
    state_g: cute.Tensor,
    slot_transition: cute.Tensor,
    dstate_in: Optional[cute.Tensor],
    dstate0: cute.Tensor,
    transition: Optional[cute.Tensor],
    stream: cuda.CUstream,
    tail_enable: Optional[cute.Tensor] = None,
) -> None:
    """The chain of the backward summary node."""
    cu_full = None
    if cutlass.const_expr(tail):
        cu_full = cu_seqlens
        cu_seqlens = cute.make_tensor(cu_seqlens.iterator + batch_base, cute.make_layout((num_seqs + cutlass.Int32(1),), stride=(1,)))
        if cutlass.const_expr(dstate_in is not None):
            dstate_in = cute.make_tensor(
                dstate_in.iterator + batch_base * dstate_in.stride[0],
                cute.make_layout((num_seqs, dstate_in.shape[1], dstate_in.shape[2], dstate_in.shape[3]), stride=dstate_in.stride),
            )
        dstate0 = cute.make_tensor(
            dstate0.iterator + batch_base * dstate0.stride[0],
            cute.make_layout((num_seqs, dstate0.shape[1], dstate0.shape[2], dstate0.shape[3]), stride=dstate0.stride),
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
        bprop_summary_cfg.max_active_clusters,
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
        q,
        q,
        do,
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
    mamba2_bprop_summary_f16.host(
        bprop_summary_cfg,
        0,
        2,
        q,
        do,
        gate,
        a,
        dt_bias,
        cu_pieces,
        None,
        None,
        state_g,
        work_items,
        main_count,
        scheduler,
        slot_transition,
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
        True,
        True,
        False,
        num_seqs,
        state_g,
        slot_transition,
        None,
        dstate_in,
        dstate0,
        main_rows,
        None,
        None,
        stream,
        mTailEnable=tail_enable,
        mTransitionOut=transition,
        first_head_lo=head_floor,
    )


def build_chain_summary_backward(
    *,
    q,
    do,
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
    state_g,
    slot_transition,
    dstate_in,
    dstate0,
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
    """Compile (cached per static config) the backward summary chain launch over the buffers of one plan; ``pieces``,
    ``heads_out`` and ``num_seqs`` are launch arguments."""
    DK = q.shape[2]
    DV = do.shape[2]
    io_dtype = get_dtype(q.dtype)
    has_seed = dstate_in is not None
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(a.dtype),
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        str(dstate0.dtype),
        str(dstate_in.dtype) if has_seed else "none",
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
    if key not in chain_summary_backward_cache:
        bprop_summary_cfg = mamba2_bprop_summary_f16.build_cfg(
            io_dtype,
            cutlass.Float32,
            max_active_clusters=num_sm,
            use_dstate_in=False,
            dt_softplus=dt_softplus,
            d_k=DK,
            d_v=DV,
        )
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_summary_placeholder = from_dlpack(work_items_summary, assumed_align=16)
        work_items_summary_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        chain_summary_backward_cache[key] = cute.compile(
            chain_summary_backward_host,
            int(unit_chunks),
            int(b_t),
            bool(length_rule),
            bprop_summary_cfg,
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
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
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
            from_dlpack(state_g, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(slot_transition, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dstate_in, assumed_align=4).mark_layout_dynamic(leading_dim=3) if has_seed else None,
            from_dlpack(dstate0, assumed_align=4).mark_layout_dynamic(leading_dim=3),
            from_dlpack(transition, assumed_align=4).mark_layout_dynamic(leading_dim=1) if transition is not None else None,
            cuda.CUstream(int(stream)),
            from_dlpack(tail_enable, assumed_align=16).mark_layout_dynamic() if tail_enable is not None else None,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return chain_summary_backward_cache[key]


def run_chain_summary_backward(
    compiled,
    *,
    q,
    do,
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
    state_g,
    slot_transition,
    dstate_in,
    dstate0,
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
    """Replay the backward summary chain: one crossing into the DSL for the three launches."""
    buffers = dict(locals())
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        float(dt_min),
        float(dt_max),
        int(batch_base),
        int(first_head_lo),
        *(buffers[name] for name in CHAIN_SUMMARY_BACKWARD_BUFFERS),
        cuda.CUstream(int(stream)),
        tail_enable,
    )
