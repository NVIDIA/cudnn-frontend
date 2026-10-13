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
from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids

from ..common import split_k
from ..common.host import get_dtype
from ..common.gate_bwd import exit_slot_table_body
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, order_body
from . import mamba2_bprop_f16, mamba2_bprop_summary_f16, mamba2_recompute_f16

USE_PDL = True
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
    skip_bwd_part: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    dstate_in: Optional[cute.Tensor],
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
) -> None:
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
        )
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
        skip_bwd_part,
        cu_seqlens,
        dstate_in,
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
    skip_bwd_part,
    cu_seqlens,
    dstate_in,
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
        dstate0 is not None,
        transition is not None,
        skip_bwd_part is not None,
    )
    if key not in warmup_backward_cache:
        bprop_cfg = mamba2_bprop_f16.build_cfg(
            io_dtype,
            state_dtype,
            max_active_clusters=num_sm,
            use_dstate_in=dstate_in is not None,
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
        dstate0_placeholder = None
        if dstate0 is not None:
            dstate0_placeholder = from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        d_placeholder = None
        if d is not None:
            d_placeholder = from_dlpack(d, assumed_align=4).mark_layout_dynamic(leading_dim=len(d.shape) - 1)
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
            from_dlpack(skip_bwd_part, assumed_align=4).mark_layout_dynamic(leading_dim=len(skip_bwd_part.shape) - 1) if skip_bwd_part is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            dstate_in_placeholder,
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
    skip_bwd_part,
    cu_seqlens,
    dstate_in,
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
        skip_bwd_part,
        cu_seqlens,
        cu_seqlens,
        dstate_in,
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
) -> None:
    """Three-CTA prologue of the regenerate path: block 0 LPT-orders the work-item table both kernels consume and zeroes
    both scheduler rings, block 1 builds the recompute's per-batch TMA-descriptor arrays (K, V, the series), block 2 the
    bprop's (Q, K, V, dO, the series, dQ, dK, dV), one warp per array, then the gate backward's exit-slot table."""
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
        order_body(
            order_gen,
            b_t,
            ORDER_THREADS,
            ORDER_ELEMENTS,
            tidx,
            n_heads_out,
            n_heads_out * n_batch,
            cu_seqlens,
            mStaging,
            mCount,
            mWorkItems,
            None,
            sKey,
            sIdx,
            sSpread,
        )
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
    skip_bwd_part: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    state_in: Optional[cute.Tensor],
    dstate_in: Optional[cute.Tensor],
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
    stream: cuda.CUstream,
) -> None:
    """The backward without a saved series: the split-K table (warmup only), one prologue for both kernels, the
    recompute writing the chunk-cadence series, the bprop reading it."""
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
        )
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
        state_in,
        None,
        None,
        None,
        work_items,
        work_count,
        scheduler_recompute,
        None,
        cutlass.Int32(b_t),
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
        skip_bwd_part,
        cu_seqlens,
        dstate_in,
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
    skip_bwd_part,
    cu_seqlens,
    state_in,
    dstate_in,
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
):
    """Compile (cached per static config) the regenerate backward launch (table, one prologue, recompute, bprop) over
    the buffers of one plan."""
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
    state_dtype = get_dtype(state_in.dtype) if state_in is not None else cutlass.Float32
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(a.dtype),
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(d.dtype) if d is not None else "none",
        len(d.shape) if d is not None else 0,
        str(cu_seqlens.dtype),
        str(state_in.dtype) if state_in is not None else "none",
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
        dstate0 is not None,
        transition is not None,
        skip_bwd_part is not None,
    )
    if key not in regenerate_backward_cache:
        recompute_cfg = mamba2_recompute_f16.build_cfg(
            io_dtype,
            state_dtype,
            max_active_clusters=num_sm,
            use_initial_state=state_in is not None,
            store_final_state=False,
            enable_checkpoints=True,
            dt_softplus=dt_softplus,
            d_k=DK,
            d_v=DV,
        )
        bprop_cfg = mamba2_bprop_f16.build_cfg(
            io_dtype,
            dstate_dtype,
            max_active_clusters=num_sm,
            use_dstate_in=dstate_in is not None,
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
        dstate_in_placeholder = None
        if dstate_in is not None:
            dstate_in_placeholder = from_dlpack(dstate_in, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        dstate0_placeholder = None
        if dstate0 is not None:
            dstate0_placeholder = from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        d_placeholder = None
        if d is not None:
            d_placeholder = from_dlpack(d, assumed_align=4).mark_layout_dynamic(leading_dim=len(d.shape) - 1)
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
            from_dlpack(skip_bwd_part, assumed_align=4).mark_layout_dynamic(leading_dim=len(skip_bwd_part.shape) - 1) if skip_bwd_part is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            state_in_placeholder,
            dstate_in_placeholder,
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
            cuda.CUstream(int(stream)),
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
    skip_bwd_part,
    cu_seqlens,
    state_in,
    dstate_in,
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
        skip_bwd_part,
        cu_seqlens,
        cu_seqlens,
        state_in,
        dstate_in,
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
        cuda.CUstream(int(stream)),
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
    dstate0: cute.Tensor,
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    scheduler: cute.Tensor,
    transition: Optional[cute.Tensor],
    workspace: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """The backward summary over every sequence: the summary prologue (one uncut item per sequence and head) and the
    summary kernel."""
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


def build_summary_backward(
    *,
    q,
    do,
    gate,
    a,
    dt_bias,
    cu_seqlens,
    dstate_in,
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
        str(transition.dtype) if transition is not None else "none",
    )
    if key not in summary_backward_cache:
        summary_cfg = mamba2_bprop_summary_f16.build_cfg(
            io_dtype,
            state_dtype,
            max_active_clusters=num_sm,
            use_dstate_in=dstate_in is not None,
            dt_softplus=dt_softplus,
            d_k=DK,
            d_v=DV,
        )
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
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
            from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            work_items_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(transition, assumed_align=4).mark_layout_dynamic(leading_dim=1) if transition is not None else None,
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            cuda.CUstream(int(stream)),
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
    dstate0,
    work_items,
    work_count,
    scheduler,
    workspace,
    dt_min,
    dt_max,
    stream,
    transition=None,
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
        dstate0,
        work_items,
        work_count,
        scheduler,
        transition,
        workspace,
        cuda.CUstream(int(stream)),
    )


frost_mamba2_regenerate_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
