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

"""One compiled launch for the Mamba-2 warmup and uncut backwards: the split-K table (plan, scan and walk, warmup
only), the bprop prologue and the bprop issued from a single host, the way ``split_k.launch`` already sequences its
three kernels, and the backward summary (its prologue and the summary kernel) the same way.  The checkpoint series the
bprop reads is the forward's own or the one the engine regenerates with the recompute kernel ahead of this launch.  Every kernel, its host and the tensor placeholder each host was compiled with are the
standalone modules' own; this host only sequences the launches.  The decay-warmup table reads the Mamba-2 gate through
the GDN safe-gate transform (raw dt gate, ``a`` (A), ``dt_bias``), the same per-token decay the kernel's gate
warp builds."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cutlass.experimental.cuda.tensor_map as tma
from cudnn.frost.buffers import DeviceView
from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids

from ..common import gate_bwd, head_reduce, split_k
from ..common.host import get_dtype
from ..common.gate_bwd import exit_slot_table_body
from ..common.piece_chain import CHAIN_THREADS, dtype_name, launch_scalar_state_chain
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, gen_interval_items, order_body
from ..common.tail import gate_body
from . import mamba2_bprop_f16, mamba2_bprop_summary_f16, mamba2_chain_backward_f16, mamba2_recompute_f16

USE_PDL = True

TAIL_BUFFERS = (
    "cu_pieces",
    "main_rows",
    "summary_rows",
    "main_count",
    "summary_count",
    "work_items",
    "work_items_summary",
    "scheduler_all",
    "scheduler_summary_fwd",
    "scheduler_summary_bwd",
    "scheduler_bprop",
    "scheduler_regenerate",
    "prefill_words",
    "bprop_words",
    "state_h",
    "transition",
    "state_x",
    "state_g",
    "state_dx",
    "seed",
    "dfs",
    "exit_base",
    "seed_row_base",
)
SUMMARY_TAIL_BUFFERS = (
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
)

warmup_backward_cache = {}
regenerate_backward_cache = {}
summary_backward_cache = {}


@cute.jit
def warmup_backward_host(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    order_gen: cutlass.Constexpr[bool],
    bprop_cfg: cutlass.Constexpr,
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    a: cute.Tensor,
    a_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    d: Optional[cute.Tensor],
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    checkpoints: cute.Tensor,
    exit_dots: cute.Tensor,
    dot_q: cute.Tensor,
    dot_k: cute.Tensor,
    skip_bwd_part: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    dstate_in: Optional[cute.Tensor],
    dcomposed: Optional[cute.Tensor],
    dstate0: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_items_table: cute.Tensor,
    work_count: cute.Tensor,
    work_count_table: cute.Tensor,
    staging: Optional[cute.Tensor],
    item_scratch: Optional[cute.Tensor],
    chunk_scratch: Optional[cute.Tensor],
    scheduler: cute.Tensor,
    transition: Optional[cute.Tensor],
    workspace: cute.Tensor,
    exit_slots: cute.Tensor,
    stream: cuda.CUstream,
    item_limit_word: Optional[cute.Tensor] = None,
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
    tail_unit_chunks: cutlass.Constexpr[int] = 0,
    tail_chain_rows: cutlass.Constexpr[int] = 0,
    tail_recompute_summary_cfg: cutlass.Constexpr = None,
    tail_recompute_series_cfg: cutlass.Constexpr = None,
    tail_bprop_summary_cfg: cutlass.Constexpr = None,
    tail_bprop_cfg: cutlass.Constexpr = None,
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
    tail_work_items_summary: Optional[cute.Tensor] = None,
    tail_scheduler_all: Optional[cute.Tensor] = None,
    tail_scheduler_summary_fwd: Optional[cute.Tensor] = None,
    tail_scheduler_summary_bwd: Optional[cute.Tensor] = None,
    tail_scheduler_bprop: Optional[cute.Tensor] = None,
    tail_scheduler_regenerate: Optional[cute.Tensor] = None,
    tail_prefill_words: Optional[cute.Tensor] = None,
    tail_bprop_words: Optional[cute.Tensor] = None,
    tail_state_h: Optional[cute.Tensor] = None,
    tail_transition: Optional[cute.Tensor] = None,
    tail_state_x: Optional[cute.Tensor] = None,
    tail_state_g: Optional[cute.Tensor] = None,
    tail_state_dx: Optional[cute.Tensor] = None,
    tail_seed: Optional[cute.Tensor] = None,
    tail_dfs: Optional[cute.Tensor] = None,
    tail_exit_base: Optional[cute.Tensor] = None,
    tail_seed_row_base: Optional[cute.Tensor] = None,
) -> None:
    if cutlass.const_expr(fold_update is not None):
        if cutlass.const_expr(fold_seed is not None):
            fold_seed = cute.make_tensor(
                fold_seed.iterator + (cutlass.Int32(fold_seed.shape[0]) - cutlass.Int32(1)) * fold_seed.stride[0],
                cute.make_layout((cutlass.Int32(1), fold_seed.shape[1], fold_seed.shape[2], fold_seed.shape[3]), stride=fold_seed.stride),
            )
        launch_scalar_state_chain(
            n_heads_out,
            bprop_cfg.d_v,
            bprop_cfg.d_k,
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
    gate_sms = num_sms if cutlass.const_expr(item_limit_word is not None) else 0
    if cutlass.const_expr(split):
        split_k.launch(
            split,
            b_t,
            True,
            True,
            False,
            overhead_chunks,
            1,
            warmup_cap,
            n_heads_out,
            num_sms,
            n_tiles,
            ideal_chunks,
            batch_size,
            log2_thresh,
            gate_table,
            a_table,
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
    mamba2_bprop_f16.prologue(
        io_dtype,
        b_t,
        order_gen,
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        cu_seqlens,
        checkpoints,
        staging,
        work_count,
        work_items,
        scheduler,
        workspace,
        exit_slots,
        stream,
        item_limit_word=item_limit_word,
        gate_sms=gate_sms,
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
        checkpoints,
        exit_dots,
        dot_q,
        dot_k,
        skip_bwd_part,
        None,
        cu_seqlens,
        dstate_in,
        dcomposed,
        dstate0,
        work_items,
        work_count,
        scheduler,
        transition,
        dt_min,
        dt_max,
        workspace,
        stream,
    )
    if cutlass.const_expr(tail_bprop_cfg is not None):
        mamba2_chain_backward_f16.chain_backward_host(
            tail_unit_chunks,
            b_t,
            False,
            tail_recompute_series_cfg is not None,
            False,
            tail_recompute_summary_cfg,
            tail_recompute_series_cfg,
            tail_bprop_summary_cfg,
            tail_bprop_cfg,
            tail_bprop_cfg.d_v,
            tail_bprop_cfg.d_k,
            tail_chain_rows,
            tail_seed is not None,
            tail_dfs is not None,
            True,
            tail_pieces,
            n_heads_out,
            tail_num_seqs,
            dt_min,
            dt_max,
            cutlass.Int32(0),
            tail_batch_base,
            tail_first_head_lo,
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
            tail_cu_pieces,
            tail_main_rows,
            tail_summary_rows,
            tail_main_count,
            tail_summary_count,
            tail_work_items,
            tail_work_items_summary,
            tail_scheduler_all,
            tail_scheduler_summary_fwd,
            tail_scheduler_summary_bwd,
            tail_scheduler_bprop,
            tail_scheduler_regenerate,
            tail_prefill_words,
            tail_bprop_words,
            checkpoints,
            tail_state_h,
            tail_transition,
            tail_state_x,
            None,
            tail_state_g,
            tail_state_dx,
            tail_seed,
            None,
            tail_dfs,
            dcomposed,
            dstate0,
            exit_dots,
            dot_q,
            dot_k,
            skip_bwd_part,
            exit_slots,
            tail_exit_base,
            tail_seed_row_base,
            stream,
            item_limit_word,
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


def build_warmup_backward(
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
    checkpoints,
    exit_dots,
    dot_q,
    dot_k,
    skip_bwd_part,
    cu_seqlens,
    dstate_in,
    dcomposed=None,
    dstate0,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    workspace,
    exit_slots,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    dt_softplus,
    dt_min,
    dt_max,
    device,
    stream,
    transition=None,
    item_limit_word=None,
    tail=None,
    tail_chain_rows=0,
    tail_scalars=(),
    tail_buffers=(),
    post_buffers=(None,) * 18,
    fold_buffers=(None,) * 4,
):
    """Compile (cached per static config) the warmup or uncut backward launch over the buffers of one plan."""
    _HQ, DK = q.shape[1], q.shape[2]
    _HV, DV = do.shape[1], do.shape[2]
    facts = split_k.split_table_facts(
        gate,
        cu_seqlens,
        split=split,
        n_tiles=n_tiles,
        ideal_chunks=ideal_chunks,
        num_sms=num_sm,
        b_t=b_t,
        log2_threshold=None,
        log_gate=True,
        safe_gate=True,
        expand_num=1,
    )
    io_dtype = get_dtype(q.dtype)
    state_source = dstate_in if dstate_in is not None else dstate0
    state_dtype = get_dtype(state_source.dtype) if state_source is not None else cutlass.Float32
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(a.dtype),
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(d.dtype) if d is not None else "none",
        len(d.shape) if d is not None else 0,
        str(cu_seqlens.dtype),
        str(state_source.dtype) if state_source is not None else "none",
        str(checkpoints.dtype),
        int(device),
        int(num_sm),
        DK,
        DV,
        int(b_t),
        bool(split),
        bool(dt_softplus),
        dstate_in is not None,
        dcomposed is not None,
        dstate0 is not None,
        transition is not None,
        skip_bwd_part is not None,
        item_limit_word is not None,
        (
            (int(tail.unit_chunks), int(tail_chain_rows), *(str(buffer.dtype) if buffer is not None else None for buffer in tail_buffers))
            if tail is not None
            else None
        ),
        tuple((str(buffer.dtype), len(buffer.shape)) if buffer is not None else None for buffer in post_buffers),
        tuple(str(buffer.dtype) if buffer is not None else None for buffer in fold_buffers),
    )
    if key not in warmup_backward_cache:
        bprop_cfg = mamba2_bprop_f16.build_cfg(
            io_dtype,
            state_dtype,
            max_active_clusters=num_sm,
            use_dstate_in=dstate_in is not None or dcomposed is not None,
            dstate_table=dstate_in is not None,
            has_composed_dstate=dcomposed is not None,
            use_dstate0=dstate0 is not None,
            dt_softplus=dt_softplus,
            has_d=d is not None,
            d_per_channel=d is not None and len(d.shape) == 2,
            skip_bwd_partials=skip_bwd_part is not None,
            d_k=DK,
            d_v=DV,
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
        dstate_in_placeholder = None
        if dstate_in is not None:
            dstate_in_placeholder = from_dlpack(dstate_in, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        dcomposed_placeholder = None
        if dcomposed is not None:
            dcomposed_placeholder = from_dlpack(dcomposed, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        dstate0_placeholder = None
        if dstate0 is not None:
            dstate0_placeholder = from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        d_placeholder = None
        if d is not None:
            d_placeholder = from_dlpack(d, assumed_align=4).mark_layout_dynamic(leading_dim=len(d.shape) - 1)
        tail_constexprs = ()
        tail_placeholders = ()
        if tail is not None:
            buffers = dict(zip(TAIL_BUFFERS, tail_buffers))
            tail_cfgs = mamba2_chain_backward_f16.build_configs(
                io_dtype,
                regenerate=False,
                coarse=False,
                seed_dtype=cutlass.Float32,
                use_dstate0=dstate0 is not None,
                dt_softplus=dt_softplus,
                has_d=d is not None,
                d_per_channel=d is not None and len(d.shape) == 2,
                skip_bwd_partials=skip_bwd_part is not None,
                max_active_clusters=num_sm,
                d_k=DK,
                d_v=DV,
            )
            tail_items = from_dlpack(buffers["work_items"], assumed_align=16)
            tail_items.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_items_summary = from_dlpack(buffers["work_items_summary"], assumed_align=16)
            tail_items_summary.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_constexprs = (int(tail.unit_chunks), int(tail_chain_rows), *tail_cfgs)
            tail_placeholders = [
                *(cutlass.Int32(int(value)) for value in tail_scalars),
                from_dlpack(buffers["cu_pieces"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["main_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["summary_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["main_count"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["summary_count"], assumed_align=4).mark_layout_dynamic(),
                tail_items,
                tail_items_summary,
                from_dlpack(buffers["scheduler_all"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_summary_fwd"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_summary_bwd"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_bprop"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_regenerate"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["prefill_words"], assumed_align=128).mark_layout_dynamic() if buffers["prefill_words"] is not None else None,
                from_dlpack(buffers["bprop_words"], assumed_align=128).mark_layout_dynamic(),
                from_dlpack(buffers["state_h"], assumed_align=16).mark_layout_dynamic(leading_dim=3) if buffers["state_h"] is not None else None,
                from_dlpack(buffers["transition"], assumed_align=4).mark_layout_dynamic(leading_dim=1),
                from_dlpack(buffers["state_x"], assumed_align=16).mark_layout_dynamic(leading_dim=3) if buffers["state_x"] is not None else None,
                from_dlpack(buffers["state_g"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(buffers["state_dx"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(buffers["seed"], assumed_align=4).mark_layout_dynamic(leading_dim=3) if buffers["seed"] is not None else None,
                from_dlpack(buffers["dfs"], assumed_align=4).mark_layout_dynamic(leading_dim=3) if buffers["dfs"] is not None else None,
                from_dlpack(buffers["exit_base"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["seed_row_base"], assumed_align=4).mark_layout_dynamic() if buffers["seed_row_base"] is not None else None,
            ]
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
            fold_shape = (1, gate.shape[1], DV, DK)
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
        warmup_backward_cache[key] = cute.compile(
            warmup_backward_host,
            facts.split,
            facts.b_t,
            facts.overhead_chunks,
            facts.warmup_cap,
            cutlass.Int32(facts.n_heads_out),
            facts.num_sms,
            io_dtype,
            not split,
            bprop_cfg,
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(facts.ideal_chunks),
            cutlass.Int32(facts.batch_size),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_walk_ctas),
            cutlass.Float32(float(dt_min)),
            cutlass.Float32(float(dt_max)),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            gate_table_placeholder,
            from_dlpack(a, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(a, assumed_align=4).mark_layout_dynamic() if split else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            dt_bias_table_placeholder,
            d_placeholder,
            from_dlpack(dq, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dk, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dv, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(exit_dots, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dot_q, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dot_k, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(skip_bwd_part, assumed_align=4).mark_layout_dynamic(leading_dim=len(skip_bwd_part.shape) - 1) if skip_bwd_part is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            dstate_in_placeholder,
            dcomposed_placeholder,
            dstate0_placeholder,
            work_items_placeholder,
            work_items_table_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            work_count_table_placeholder,
            staging_placeholder,
            item_scratch_placeholder,
            chunk_scratch_placeholder,
            from_dlpack(scheduler, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(transition, assumed_align=4).mark_layout_dynamic(leading_dim=1) if transition is not None else None,
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(exit_slots, assumed_align=8).mark_layout_dynamic(leading_dim=1),
            cuda.CUstream(int(stream)),
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            fold_rows,
            *fold_placeholders,
            gate_skip_channels,
            *post_placeholders,
            *tail_constexprs,
            *tail_placeholders,
            options="--enable-tvm-ffi --opt-level 2",
        )
    compiled = warmup_backward_cache[key]
    return compiled, facts


def run_warmup_backward(
    compiled,
    facts,
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
    checkpoints,
    exit_dots,
    dot_q,
    dot_k,
    skip_bwd_part,
    cu_seqlens,
    dstate_in,
    dcomposed=None,
    dstate0,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    workspace,
    exit_slots,
    dt_min,
    dt_max,
    stream,
    transition=None,
    item_limit_word=None,
    tail_scalars=(),
    tail_buffers=(),
    post_buffers=(None,) * 18,
    fold_buffers=(None,) * 4,
) -> None:
    """Replay the warmup or uncut backward: one crossing into the DSL for the table, prologue and bprop launches."""
    compiled(
        facts.n_heads_out,
        facts.n_tiles,
        facts.ideal_chunks,
        facts.batch_size,
        facts.log2_threshold,
        facts.n_scan_ctas,
        facts.n_walk_ctas,
        float(dt_min),
        float(dt_max),
        q,
        k,
        v,
        do,
        gate,
        gate if facts.split else None,
        a,
        a if facts.split else None,
        dt_bias,
        dt_bias,
        d,
        dq,
        dk,
        dv,
        checkpoints,
        exit_dots,
        dot_q,
        dot_k,
        skip_bwd_part,
        cu_seqlens,
        cu_seqlens,
        dstate_in,
        dcomposed,
        dstate0,
        work_items,
        work_items,
        work_count,
        work_count,
        item_scratch if facts.split else None,
        item_scratch if facts.split else None,
        chunk_scratch if facts.split else None,
        scheduler,
        transition,
        workspace,
        exit_slots,
        cuda.CUstream(int(stream)),
        item_limit_word,
        *fold_buffers,
        *post_buffers,
        *tail_scalars,
        *tail_buffers,
    )


@cute.kernel
def frost_mamba2_regenerate_prologue(
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    recompute_base_k: cutlass.GridConstant[tma.TensorMap],
    recompute_base_v: cutlass.GridConstant[tma.TensorMap],
    recompute_base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_do: cutlass.GridConstant[tma.TensorMap],
    base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_dq: cutlass.GridConstant[tma.TensorMap],
    base_dk: cutlass.GridConstant[tma.TensorMap],
    base_dv: cutlass.GridConstant[tma.TensorMap],
    recompute_workspace: cute.Tensor,
    bprop_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    series: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    mStaging: Optional[cute.Tensor],
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mSchedulerRecompute: cute.Tensor,
    mSchedulerBprop: cute.Tensor,
    n_batch: cutlass.Int32,
    mSlots: cute.Tensor,
    coarse: cutlass.Constexpr[bool],
    seed_span_chunks: cutlass.Int32,
    mSeriesItems: Optional[cute.Tensor],
    mSeriesCount: Optional[cute.Tensor],
    mItemLimit: Optional[cute.Tensor] = None,
    gate_sms: cutlass.Constexpr[int] = 0,
) -> None:
    """Three-CTA prologue of the regenerate path: block 0 LPT-orders the work-item table both kernels consume (under
    ``coarse`` the bprop's alone; the recompute then runs over checkpoint-seeded interval items of ``seed_span_chunks``)
    and zeroes both scheduler rings, block 1 builds the recompute's per-batch TMA-descriptor arrays (K, V, the series),
    block 2 the bprop's (Q, K, V, dO, the series, dQ, dK, dV), one warp per array, then the gate backward's exit-slot
    table."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    if bidx == cutlass.Int32(0):
        if tidx == 0:
            for cell in cutlass.range_constexpr(2):
                mSchedulerRecompute[cell] = cutlass.Int32(0)
                mSchedulerBprop[cell] = cutlass.Int32(0)
        sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        n_heads_out = cutlass.Int32(q.shape[1] if q.shape[1] >= do.shape[1] else do.shape[1])
        n_items = n_heads_out * n_batch
        if cutlass.const_expr(mItemLimit is not None):
            if cutlass.const_expr(gate_sms > 0):
                gate_body(ORDER_THREADS, tidx, n_heads_out, gate_sms, cu_seqlens, mItemLimit)
            n_items = cutlass.Int32(mItemLimit[1])
        order_body(
            order_gen,
            b_t,
            ORDER_THREADS,
            ORDER_ELEMENTS,
            tidx,
            n_heads_out,
            n_items,
            cu_seqlens,
            mStaging,
            mCount,
            mWorkItems,
            None,
            sKey,
            sIdx,
            sSpread,
        )
        if cutlass.const_expr(coarse):
            gen_interval_items(b_t, ORDER_THREADS, tidx, n_heads_out, n_items, seed_span_chunks, cu_seqlens, mSeriesCount, mSeriesItems, None)
    elif bidx == cutlass.Int32(1):
        mamba2_recompute_f16.build_descs_body(
            widx,
            recompute_base_k,
            recompute_base_v,
            recompute_base_checkpoint,
            recompute_workspace,
            cu_seqlens,
            k,
            v,
            series,
            n_batch,
            cutlass.Int32(b_t),
            b_t,
        )
    else:
        mamba2_bprop_f16.build_descs_body(
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
            cu_seqlens,
            q,
            k,
            v,
            do,
            series,
            dq,
            dk,
            dv,
            n_batch,
            b_t,
        )
        exit_slot_table_body(b_t, tidx, ORDER_THREADS, n_batch, cu_seqlens, mSlots)


@cute.jit
def regenerate_backward_host(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    order_gen: cutlass.Constexpr[bool],
    coarse: cutlass.Constexpr[bool],
    recompute_cfg: cutlass.Constexpr,
    bprop_cfg: cutlass.Constexpr,
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    a: cute.Tensor,
    a_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    d: Optional[cute.Tensor],
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    series: cute.Tensor,
    exit_dots: cute.Tensor,
    dot_q: cute.Tensor,
    dot_k: cute.Tensor,
    skip_bwd_part: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    state_in: Optional[cute.Tensor],
    composed: Optional[cute.Tensor],
    dstate_in: Optional[cute.Tensor],
    dcomposed: Optional[cute.Tensor],
    dstate0: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_items_table: cute.Tensor,
    work_count: cute.Tensor,
    work_count_table: cute.Tensor,
    staging: Optional[cute.Tensor],
    item_scratch: Optional[cute.Tensor],
    chunk_scratch: Optional[cute.Tensor],
    scheduler_recompute: cute.Tensor,
    scheduler_bprop: cute.Tensor,
    transition: Optional[cute.Tensor],
    recompute_workspace: cute.Tensor,
    bprop_workspace: cute.Tensor,
    exit_slots: cute.Tensor,
    seed_checkpoints: Optional[cute.Tensor],
    series_items: Optional[cute.Tensor],
    series_count: Optional[cute.Tensor],
    seed_span_chunks: cutlass.Int32,
    seed_every_n_tokens: cutlass.Int32,
    stream: cuda.CUstream,
    item_limit_word: Optional[cute.Tensor] = None,
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
    tail_unit_chunks: cutlass.Constexpr[int] = 0,
    tail_chain_rows: cutlass.Constexpr[int] = 0,
    tail_recompute_summary_cfg: cutlass.Constexpr = None,
    tail_recompute_series_cfg: cutlass.Constexpr = None,
    tail_bprop_summary_cfg: cutlass.Constexpr = None,
    tail_bprop_cfg: cutlass.Constexpr = None,
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
    tail_work_items_summary: Optional[cute.Tensor] = None,
    tail_scheduler_all: Optional[cute.Tensor] = None,
    tail_scheduler_summary_fwd: Optional[cute.Tensor] = None,
    tail_scheduler_summary_bwd: Optional[cute.Tensor] = None,
    tail_scheduler_bprop: Optional[cute.Tensor] = None,
    tail_scheduler_regenerate: Optional[cute.Tensor] = None,
    tail_prefill_words: Optional[cute.Tensor] = None,
    tail_bprop_words: Optional[cute.Tensor] = None,
    tail_state_h: Optional[cute.Tensor] = None,
    tail_transition: Optional[cute.Tensor] = None,
    tail_state_x: Optional[cute.Tensor] = None,
    tail_state_g: Optional[cute.Tensor] = None,
    tail_state_dx: Optional[cute.Tensor] = None,
    tail_seed: Optional[cute.Tensor] = None,
    tail_dfs: Optional[cute.Tensor] = None,
    tail_exit_base: Optional[cute.Tensor] = None,
    tail_seed_row_base: Optional[cute.Tensor] = None,
) -> None:
    """The backward without a saved series: the split-K table (warmup only), one prologue for both kernels, the
    recompute writing the chunk-cadence series (under ``coarse`` over interval items seeded from ``seed_checkpoints``,
    io or fp32 rows ``seed_every_n_tokens`` apart), the bprop reading it."""
    if cutlass.const_expr(fold_update is not None):
        if cutlass.const_expr(fold_seed is not None):
            fold_seed = cute.make_tensor(
                fold_seed.iterator + (cutlass.Int32(fold_seed.shape[0]) - cutlass.Int32(1)) * fold_seed.stride[0],
                cute.make_layout((cutlass.Int32(1), fold_seed.shape[1], fold_seed.shape[2], fold_seed.shape[3]), stride=fold_seed.stride),
            )
        launch_scalar_state_chain(
            n_heads_out,
            bprop_cfg.d_v,
            bprop_cfg.d_k,
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
    gate_sms = num_sms if cutlass.const_expr(item_limit_word is not None) else 0
    if cutlass.const_expr(split):
        split_k.launch(
            split,
            b_t,
            True,
            True,
            False,
            overhead_chunks,
            1,
            warmup_cap,
            n_heads_out,
            num_sms,
            n_tiles,
            ideal_chunks,
            batch_size,
            log2_thresh,
            gate_table,
            a_table,
            dt_bias_table,
            cu_seqlens_table,
            chunk_scratch,
            item_scratch,
            work_items_table,
            work_count_table,
            scheduler_bprop,
            n_scan_ctas,
            n_walk_ctas,
            stream,
            item_limit_word,
            gate_sms=gate_sms,
        )
        gate_sms = 0
    recompute_base_k, recompute_base_v, recompute_base_checkpoint = mamba2_recompute_f16.base_descriptors(io_dtype, b_t, k, v, series)
    base_q, base_k, base_v, base_do, base_checkpoint, base_dq, base_dk, base_dv = mamba2_bprop_f16.base_descriptors(
        io_dtype, b_t, q, k, v, do, dq, dk, dv, series
    )
    frost_mamba2_regenerate_prologue(
        order_gen,
        b_t,
        recompute_base_k,
        recompute_base_v,
        recompute_base_checkpoint,
        base_q,
        base_k,
        base_v,
        base_do,
        base_checkpoint,
        base_dq,
        base_dk,
        base_dv,
        recompute_workspace,
        bprop_workspace,
        cu_seqlens,
        q,
        k,
        v,
        do,
        series,
        dq,
        dk,
        dv,
        staging,
        work_count,
        work_items,
        scheduler_recompute,
        scheduler_bprop,
        batch_size,
        exit_slots,
        coarse,
        seed_span_chunks,
        series_items,
        series_count,
        item_limit_word,
        gate_sms,
    ).launch(grid=(3, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)
    mamba2_recompute_f16.host(
        recompute_cfg,
        0,
        1,
        2,
        k,
        v,
        gate,
        a,
        dt_bias,
        cu_seqlens,
        seed_checkpoints if coarse else state_in,
        composed,
        None,
        None,
        None,
        series_items if coarse else work_items,
        series_count if coarse else work_count,
        scheduler_recompute,
        None,
        cutlass.Int32(b_t),
        seed_every_n_tokens if coarse else cutlass.Int32(0),
        dt_min,
        dt_max,
        recompute_workspace,
        stream,
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
        None,
        cu_seqlens,
        dstate_in,
        dcomposed,
        dstate0,
        work_items,
        work_count,
        scheduler_bprop,
        transition,
        dt_min,
        dt_max,
        bprop_workspace,
        stream,
    )
    if cutlass.const_expr(tail_bprop_cfg is not None):
        mamba2_chain_backward_f16.chain_backward_host(
            tail_unit_chunks,
            b_t,
            False,
            tail_recompute_series_cfg is not None,
            coarse,
            tail_recompute_summary_cfg,
            tail_recompute_series_cfg,
            tail_bprop_summary_cfg,
            tail_bprop_cfg,
            tail_bprop_cfg.d_v,
            tail_bprop_cfg.d_k,
            tail_chain_rows,
            tail_seed is not None,
            tail_dfs is not None,
            True,
            tail_pieces,
            n_heads_out,
            tail_num_seqs,
            dt_min,
            dt_max,
            seed_every_n_tokens,
            tail_batch_base,
            tail_first_head_lo,
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
            tail_cu_pieces,
            tail_main_rows,
            tail_summary_rows,
            tail_main_count,
            tail_summary_count,
            tail_work_items,
            tail_work_items_summary,
            tail_scheduler_all,
            tail_scheduler_summary_fwd,
            tail_scheduler_summary_bwd,
            tail_scheduler_bprop,
            tail_scheduler_regenerate,
            tail_prefill_words,
            tail_bprop_words,
            series,
            tail_state_h,
            tail_transition,
            tail_state_x,
            seed_checkpoints,
            tail_state_g,
            tail_state_dx,
            tail_seed,
            None,
            tail_dfs,
            dcomposed,
            dstate0,
            exit_dots,
            dot_q,
            dot_k,
            skip_bwd_part,
            exit_slots,
            tail_exit_base,
            tail_seed_row_base,
            stream,
            item_limit_word,
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


def build_regenerate_backward(
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
    series,
    exit_dots,
    dot_q,
    dot_k,
    skip_bwd_part,
    cu_seqlens,
    state_in,
    composed=None,
    dstate_in,
    dcomposed=None,
    dstate0,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler_recompute,
    scheduler_bprop,
    recompute_workspace,
    bprop_workspace,
    exit_slots,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    dt_softplus,
    dt_min,
    dt_max,
    device,
    stream,
    transition=None,
    seed_checkpoints=None,
    series_items=None,
    series_count=None,
    seed_span_tokens=0,
    seed_every_n_tokens=0,
    item_limit_word=None,
    tail=None,
    tail_chain_rows=0,
    tail_scalars=(),
    tail_buffers=(),
    post_buffers=(None,) * 18,
    fold_buffers=(None,) * 4,
):
    """Compile (cached per static config) the regenerate backward launch (table, one prologue, recompute, bprop) over
    the buffers of one plan; ``seed_checkpoints`` (io or fp32 rows ``seed_every_n_tokens`` apart) makes the recompute run
    over interval items of ``seed_span_tokens`` seeded from the rows."""
    _HQ, DK = q.shape[1], q.shape[2]
    _HV, DV = do.shape[1], do.shape[2]
    facts = split_k.split_table_facts(
        gate,
        cu_seqlens,
        split=split,
        n_tiles=n_tiles,
        ideal_chunks=ideal_chunks,
        num_sms=num_sm,
        b_t=b_t,
        log2_threshold=None,
        log_gate=True,
        safe_gate=True,
        expand_num=1,
    )
    io_dtype = get_dtype(q.dtype)
    dstate_source = dstate_in if dstate_in is not None else dstate0
    dstate_dtype = get_dtype(dstate_source.dtype) if dstate_source is not None else cutlass.Float32
    coarse = seed_checkpoints is not None
    state_source = seed_checkpoints if coarse else state_in
    state_dtype = get_dtype(state_source.dtype) if state_source is not None else cutlass.Float32
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(a.dtype),
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(d.dtype) if d is not None else "none",
        len(d.shape) if d is not None else 0,
        str(cu_seqlens.dtype),
        str(state_source.dtype) if state_source is not None else "none",
        bool(coarse),
        composed is not None,
        str(dstate_source.dtype) if dstate_source is not None else "none",
        str(series.dtype),
        int(device),
        int(num_sm),
        DK,
        DV,
        int(b_t),
        bool(split),
        bool(dt_softplus),
        dstate_in is not None,
        dcomposed is not None,
        dstate0 is not None,
        transition is not None,
        skip_bwd_part is not None,
        item_limit_word is not None,
        (
            (int(tail.unit_chunks), int(tail_chain_rows), *(str(buffer.dtype) if buffer is not None else None for buffer in tail_buffers))
            if tail is not None
            else None
        ),
        tuple((str(buffer.dtype), len(buffer.shape)) if buffer is not None else None for buffer in post_buffers),
        tuple(str(buffer.dtype) if buffer is not None else None for buffer in fold_buffers),
    )
    if key not in regenerate_backward_cache:
        recompute_cfg = mamba2_recompute_f16.build_cfg(
            io_dtype,
            state_dtype,
            max_active_clusters=num_sm,
            use_initial_state=state_source is not None or composed is not None,
            state_table=state_source is not None,
            has_composed=composed is not None,
            store_final_state=False,
            enable_checkpoints=True,
            dt_softplus=dt_softplus,
            seed_checkpoints=coarse,
            d_k=DK,
            d_v=DV,
        )
        bprop_cfg = mamba2_bprop_f16.build_cfg(
            io_dtype,
            dstate_dtype,
            max_active_clusters=num_sm,
            use_dstate_in=dstate_in is not None or dcomposed is not None,
            dstate_table=dstate_in is not None,
            has_composed_dstate=dcomposed is not None,
            use_dstate0=dstate0 is not None,
            dt_softplus=dt_softplus,
            has_d=d is not None,
            d_per_channel=d is not None and len(d.shape) == 2,
            skip_bwd_partials=skip_bwd_part is not None,
            d_k=DK,
            d_v=DV,
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
        seed_checkpoints_placeholder = None
        series_items_placeholder = None
        if coarse:
            seed_checkpoints_placeholder = from_dlpack(seed_checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3)
            series_items_placeholder = from_dlpack(series_items, assumed_align=16)
            series_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        dstate_in_placeholder = None
        if dstate_in is not None:
            dstate_in_placeholder = from_dlpack(dstate_in, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        dcomposed_placeholder = None
        if dcomposed is not None:
            dcomposed_placeholder = from_dlpack(dcomposed, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        dstate0_placeholder = None
        if dstate0 is not None:
            dstate0_placeholder = from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        d_placeholder = None
        if d is not None:
            d_placeholder = from_dlpack(d, assumed_align=4).mark_layout_dynamic(leading_dim=len(d.shape) - 1)
        tail_constexprs = ()
        tail_placeholders = ()
        if tail is not None:
            buffers = dict(zip(TAIL_BUFFERS, tail_buffers))
            tail_cfgs = mamba2_chain_backward_f16.build_configs(
                io_dtype,
                regenerate=True,
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
            tail_items = from_dlpack(buffers["work_items"], assumed_align=16)
            tail_items.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_items_summary = from_dlpack(buffers["work_items_summary"], assumed_align=16)
            tail_items_summary.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_constexprs = (int(tail.unit_chunks), int(tail_chain_rows), *tail_cfgs)
            tail_placeholders = [
                *(cutlass.Int32(int(value)) for value in tail_scalars),
                from_dlpack(buffers["cu_pieces"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["main_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["summary_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["main_count"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["summary_count"], assumed_align=4).mark_layout_dynamic(),
                tail_items,
                tail_items_summary,
                from_dlpack(buffers["scheduler_all"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_summary_fwd"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_summary_bwd"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_bprop"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["scheduler_regenerate"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["prefill_words"], assumed_align=128).mark_layout_dynamic() if buffers["prefill_words"] is not None else None,
                from_dlpack(buffers["bprop_words"], assumed_align=128).mark_layout_dynamic(),
                from_dlpack(buffers["state_h"], assumed_align=16).mark_layout_dynamic(leading_dim=3) if buffers["state_h"] is not None else None,
                from_dlpack(buffers["transition"], assumed_align=4).mark_layout_dynamic(leading_dim=1),
                from_dlpack(buffers["state_x"], assumed_align=16).mark_layout_dynamic(leading_dim=3) if buffers["state_x"] is not None else None,
                from_dlpack(buffers["state_g"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(buffers["state_dx"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(buffers["seed"], assumed_align=4).mark_layout_dynamic(leading_dim=3) if buffers["seed"] is not None else None,
                from_dlpack(buffers["dfs"], assumed_align=4).mark_layout_dynamic(leading_dim=3) if buffers["dfs"] is not None else None,
                from_dlpack(buffers["exit_base"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["seed_row_base"], assumed_align=4).mark_layout_dynamic() if buffers["seed_row_base"] is not None else None,
            ]
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
            fold_shape = (1, gate.shape[1], DV, DK)
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
        regenerate_backward_cache[key] = cute.compile(
            regenerate_backward_host,
            facts.split,
            facts.b_t,
            facts.overhead_chunks,
            facts.warmup_cap,
            cutlass.Int32(facts.n_heads_out),
            facts.num_sms,
            io_dtype,
            not split,
            bool(coarse),
            recompute_cfg,
            bprop_cfg,
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(facts.ideal_chunks),
            cutlass.Int32(facts.batch_size),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_walk_ctas),
            cutlass.Float32(float(dt_min)),
            cutlass.Float32(float(dt_max)),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            gate_table_placeholder,
            from_dlpack(a, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(a, assumed_align=4).mark_layout_dynamic() if split else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            dt_bias_table_placeholder,
            d_placeholder,
            from_dlpack(dq, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dk, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dv, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(series, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(exit_dots, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dot_q, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dot_k, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(skip_bwd_part, assumed_align=4).mark_layout_dynamic(leading_dim=len(skip_bwd_part.shape) - 1) if skip_bwd_part is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            state_in_placeholder,
            composed_placeholder,
            dstate_in_placeholder,
            dcomposed_placeholder,
            dstate0_placeholder,
            work_items_placeholder,
            work_items_table_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            work_count_table_placeholder,
            staging_placeholder,
            item_scratch_placeholder,
            chunk_scratch_placeholder,
            from_dlpack(scheduler_recompute, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_bprop, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(transition, assumed_align=4).mark_layout_dynamic(leading_dim=1) if transition is not None else None,
            from_dlpack(recompute_workspace, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(bprop_workspace, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(exit_slots, assumed_align=8).mark_layout_dynamic(leading_dim=1),
            seed_checkpoints_placeholder,
            series_items_placeholder,
            from_dlpack(series_count, assumed_align=4).mark_layout_dynamic() if coarse else None,
            cutlass.Int32(int(seed_span_tokens or seed_every_n_tokens) // int(b_t)),
            cutlass.Int32(int(seed_every_n_tokens)),
            cuda.CUstream(int(stream)),
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            fold_rows,
            *fold_placeholders,
            gate_skip_channels,
            *post_placeholders,
            *tail_constexprs,
            *tail_placeholders,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return regenerate_backward_cache[key], facts


def run_regenerate_backward(
    compiled,
    facts,
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
    series,
    exit_dots,
    dot_q,
    dot_k,
    skip_bwd_part,
    cu_seqlens,
    state_in,
    composed=None,
    dstate_in,
    dcomposed=None,
    dstate0,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler_recompute,
    scheduler_bprop,
    recompute_workspace,
    bprop_workspace,
    exit_slots,
    dt_min,
    dt_max,
    stream,
    transition=None,
    seed_checkpoints=None,
    series_items=None,
    series_count=None,
    seed_span_tokens=0,
    seed_every_n_tokens=0,
    item_limit_word=None,
    tail_scalars=(),
    tail_buffers=(),
    post_buffers=(None,) * 18,
    fold_buffers=(None,) * 4,
) -> None:
    """Replay the regenerate backward: one crossing into the DSL for the table, prologue, recompute and bprop launches."""
    compiled(
        facts.n_heads_out,
        facts.n_tiles,
        facts.ideal_chunks,
        facts.batch_size,
        facts.log2_threshold,
        facts.n_scan_ctas,
        facts.n_walk_ctas,
        float(dt_min),
        float(dt_max),
        q,
        k,
        v,
        do,
        gate,
        gate if facts.split else None,
        a,
        a if facts.split else None,
        dt_bias,
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
        cu_seqlens,
        cu_seqlens,
        state_in,
        composed,
        dstate_in,
        dcomposed,
        dstate0,
        work_items,
        work_items,
        work_count,
        work_count,
        item_scratch if facts.split else None,
        item_scratch if facts.split else None,
        chunk_scratch if facts.split else None,
        scheduler_recompute,
        scheduler_bprop,
        transition,
        recompute_workspace,
        bprop_workspace,
        exit_slots,
        seed_checkpoints,
        series_items,
        series_count,
        int(seed_span_tokens or seed_every_n_tokens) // int(facts.b_t),
        int(seed_every_n_tokens),
        cuda.CUstream(int(stream)),
        item_limit_word,
        *fold_buffers,
        *post_buffers,
        *tail_scalars,
        *tail_buffers,
    )


@cute.jit
def summary_backward_host(
    b_t: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    summary_cfg: cutlass.Constexpr,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    q: cute.Tensor,
    do: cute.Tensor,
    gate: cute.Tensor,
    a: cute.Tensor,
    dt_bias: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    dstate_in: Optional[cute.Tensor],
    dcomposed: Optional[cute.Tensor],
    dstate0: cute.Tensor,
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    scheduler: cute.Tensor,
    transition: Optional[cute.Tensor],
    workspace: cute.Tensor,
    stream: cuda.CUstream,
    item_limit_word: Optional[cute.Tensor] = None,
    tail_unit_chunks: cutlass.Constexpr[int] = 0,
    tail_chain_rows: cutlass.Constexpr[int] = 0,
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
    tail_work_items_summary: Optional[cute.Tensor] = None,
    tail_scheduler: Optional[cute.Tensor] = None,
    tail_words: Optional[cute.Tensor] = None,
    tail_state_g: Optional[cute.Tensor] = None,
    tail_slot_transition: Optional[cute.Tensor] = None,
    tail_dstate_in: Optional[cute.Tensor] = None,
    tail_dstate0: Optional[cute.Tensor] = None,
) -> None:
    """The backward summary over every sequence: the summary prologue (one uncut item per sequence and head) and the
    summary kernel."""
    gate_sms = summary_cfg.max_active_clusters if cutlass.const_expr(item_limit_word is not None) else 0
    mamba2_bprop_summary_f16.prologue(
        io_dtype,
        b_t,
        True,
        q,
        do,
        cu_seqlens,
        None,
        work_count,
        work_items,
        scheduler,
        workspace,
        stream,
        item_limit_word=item_limit_word,
        gate_sms=gate_sms,
    )
    mamba2_bprop_summary_f16.host(
        summary_cfg,
        0,
        1,
        q,
        do,
        gate,
        a,
        dt_bias,
        cu_seqlens,
        dstate_in,
        dcomposed,
        dstate0,
        work_items,
        work_count,
        scheduler,
        transition,
        dt_min,
        dt_max,
        workspace,
        stream,
    )
    if cutlass.const_expr(tail_bprop_summary_cfg is not None):
        mamba2_chain_backward_f16.chain_summary_backward_host(
            tail_unit_chunks,
            b_t,
            False,
            tail_bprop_summary_cfg,
            tail_bprop_summary_cfg.d_v,
            tail_bprop_summary_cfg.d_k,
            tail_chain_rows,
            tail_dstate_in is not None,
            True,
            tail_pieces,
            cutlass.Int32(gate.shape[1]),
            tail_num_seqs,
            dt_min,
            dt_max,
            tail_batch_base,
            tail_first_head_lo,
            q,
            do,
            gate,
            a,
            dt_bias,
            cu_seqlens,
            tail_cu_pieces,
            tail_main_rows,
            tail_summary_rows,
            tail_main_count,
            tail_summary_count,
            tail_work_items,
            tail_work_items_summary,
            tail_scheduler,
            tail_words,
            tail_state_g,
            tail_slot_transition,
            tail_dstate_in,
            tail_dstate0,
            transition,
            stream,
            item_limit_word,
        )


def build_summary_backward(
    *,
    q,
    do,
    gate,
    a,
    dt_bias,
    cu_seqlens,
    dstate_in,
    dcomposed=None,
    dstate0,
    work_items,
    work_count,
    scheduler,
    workspace,
    num_sm,
    b_t,
    dt_softplus,
    dt_min,
    dt_max,
    device,
    stream,
    transition=None,
    item_limit_word=None,
    tail=None,
    tail_chain_rows=0,
    tail_scalars=(),
    tail_buffers=(),
):
    """Compile (cached per static config) the backward summary launch over the buffers of one plan."""
    _HQ, DK = q.shape[1], q.shape[2]
    _HV, DV = do.shape[1], do.shape[2]
    io_dtype = get_dtype(q.dtype)
    state_source = dstate_in if dstate_in is not None else dstate0
    state_dtype = get_dtype(state_source.dtype)
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(a.dtype),
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        str(state_source.dtype),
        int(device),
        int(num_sm),
        DK,
        DV,
        int(b_t),
        bool(dt_softplus),
        dstate_in is not None,
        dcomposed is not None,
        str(transition.dtype) if transition is not None else "none",
        item_limit_word is not None,
        (
            (int(tail.unit_chunks), int(tail_chain_rows), *(str(buffer.dtype) if buffer is not None else None for buffer in tail_buffers))
            if tail is not None
            else None
        ),
    )
    if key not in summary_backward_cache:
        summary_cfg = mamba2_bprop_summary_f16.build_cfg(
            io_dtype,
            state_dtype,
            max_active_clusters=num_sm,
            use_dstate_in=dstate_in is not None or dcomposed is not None,
            dstate_table=dstate_in is not None,
            has_composed_dstate=dcomposed is not None,
            dt_softplus=dt_softplus,
            d_k=DK,
            d_v=DV,
        )
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        tail_constexprs = ()
        tail_placeholders = ()
        if tail is not None:
            buffers = dict(zip(SUMMARY_TAIL_BUFFERS, tail_buffers))
            tail_bprop_summary_cfg = mamba2_bprop_summary_f16.build_cfg(
                io_dtype,
                cutlass.Float32,
                max_active_clusters=num_sm,
                use_dstate_in=False,
                dt_softplus=dt_softplus,
                d_k=DK,
                d_v=DV,
            )
            tail_items = from_dlpack(buffers["work_items"], assumed_align=16)
            tail_items.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_items_summary = from_dlpack(buffers["work_items_summary"], assumed_align=16)
            tail_items_summary.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            tail_constexprs = (int(tail.unit_chunks), int(tail_chain_rows), tail_bprop_summary_cfg)
            tail_placeholders = [
                *(cutlass.Int32(int(value)) for value in tail_scalars),
                from_dlpack(buffers["cu_pieces"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["main_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["summary_rows"], assumed_align=16).mark_layout_dynamic(),
                from_dlpack(buffers["main_count"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["summary_count"], assumed_align=4).mark_layout_dynamic(),
                tail_items,
                tail_items_summary,
                from_dlpack(buffers["scheduler"], assumed_align=4).mark_layout_dynamic(),
                from_dlpack(buffers["words"], assumed_align=128).mark_layout_dynamic(),
                from_dlpack(buffers["state_g"], assumed_align=16).mark_layout_dynamic(leading_dim=3),
                from_dlpack(buffers["slot_transition"], assumed_align=4).mark_layout_dynamic(leading_dim=1),
                from_dlpack(buffers["dstate_in"], assumed_align=4).mark_layout_dynamic(leading_dim=3) if buffers["dstate_in"] is not None else None,
                from_dlpack(buffers["dstate0"], assumed_align=4).mark_layout_dynamic(leading_dim=3),
            ]
        summary_backward_cache[key] = cute.compile(
            summary_backward_host,
            int(b_t),
            io_dtype,
            summary_cfg,
            cutlass.Float32(float(dt_min)),
            cutlass.Float32(float(dt_max)),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(a, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(dstate_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate_in is not None else None,
            from_dlpack(dcomposed, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dcomposed is not None else None,
            from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            work_items_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(transition, assumed_align=4).mark_layout_dynamic(leading_dim=1) if transition is not None else None,
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            cuda.CUstream(int(stream)),
            from_dlpack(item_limit_word, assumed_align=16).mark_layout_dynamic() if item_limit_word is not None else None,
            *tail_constexprs,
            *tail_placeholders,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return summary_backward_cache[key]


def run_summary_backward(
    compiled,
    *,
    q,
    do,
    gate,
    a,
    dt_bias,
    cu_seqlens,
    dstate_in,
    dcomposed=None,
    dstate0,
    work_items,
    work_count,
    scheduler,
    workspace,
    dt_min,
    dt_max,
    stream,
    transition=None,
    item_limit_word=None,
    tail_scalars=(),
    tail_buffers=(),
) -> None:
    """Replay the backward summary: one crossing into the DSL for the prologue and kernel launches."""
    compiled(
        float(dt_min),
        float(dt_max),
        q,
        do,
        gate,
        a,
        dt_bias,
        cu_seqlens,
        dstate_in,
        dcomposed,
        dstate0,
        work_items,
        work_count,
        scheduler,
        transition,
        workspace,
        cuda.CUstream(int(stream)),
        item_limit_word,
        *tail_scalars,
        *tail_buffers,
    )


frost_mamba2_regenerate_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
