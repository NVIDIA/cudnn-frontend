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

"""One compiled launch for the Mamba-2 warmup and uncut forwards: the split-K table (plan, scan and walk, warmup
only), the prefill prologue and the prefill issued from a single host, the way ``split_k.launch`` already sequences its
three kernels.  Every kernel, its host and the tensor placeholder each host was compiled with are the standalone
modules' own; this host only sequences the launches.  The decay-warmup table reads the Mamba-2 gate through the GDN
safe-gate transform: the raw dt is the gate, ``a`` (A) and ``dt_bias`` are its parameters, so the table and
the kernel's gate warp see the same per-token decay."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..common import split_k
from ..common.host import get_dtype
from . import mamba2_prefill_f16, mamba2_recompute_f16

warmup_forward_cache = {}
recompute_cache = {}


@cute.jit
def warmup_forward_host(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    order_gen: cutlass.Constexpr[bool],
    prefill_cfg: cutlass.Constexpr,
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    a: cute.Tensor,
    a_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    d: Optional[cute.Tensor],
    z: Optional[cute.Tensor],
    o: Optional[cute.Tensor],
    o_pre: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    state_in: Optional[cute.Tensor],
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
    transition: Optional[cute.Tensor],
    workspace: cute.Tensor,
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
    mamba2_prefill_f16.prologue(
        io_dtype,
        b_t,
        order_gen,
        q,
        k,
        v,
        o,
        cu_seqlens,
        checkpoints,
        z,
        o_pre,
        staging,
        work_count,
        work_items,
        scheduler,
        checkpoint_every_n,
        workspace,
        stream,
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
        cu_seqlens,
        state_in,
        state_out,
        seed_indices,
        final_indices,
        work_items,
        work_count,
        scheduler,
        transition,
        checkpoint_every_n,
        dt_min,
        dt_max,
        workspace,
        stream,
    )


def build_warmup_forward(
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
    state_in,
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
    dt_softplus,
    dt_min,
    dt_max,
    checkpoint_every_n_tokens,
    device,
    stream,
    transition=None,
):
    """Compile (cached per static config) the warmup or uncut forward launch over the buffers of one plan."""
    _HQ, DK = q.shape[1], q.shape[2]
    _HV, DV = v.shape[1], v.shape[2]
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
    state_source = state_in if state_in is not None else state_out
    state_dtype = get_dtype(state_source.dtype) if state_source is not None else cutlass.Float32
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
        str(state_source.dtype) if state_source is not None else "none",
        int(device),
        int(num_sm),
        DK,
        DV,
        int(b_t),
        bool(split),
        bool(dt_softplus),
        state_in is not None,
        state_out is not None,
        seed_indices is not None,
        final_indices is not None,
        int(checkpoint_every_n_tokens) > 0,
        transition is not None,
    )
    if key not in warmup_forward_cache:
        prefill_cfg = mamba2_prefill_f16.build_cfg(
            io_dtype,
            state_dtype,
            max_active_clusters=num_sm,
            use_initial_state=state_in is not None,
            store_final_state=state_out is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            dt_softplus=dt_softplus,
            has_d=d is not None,
            d_per_channel=d is not None and len(d.shape) == 2,
            has_z=z is not None,
            store_pre_gate=o_pre is not None,
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
        state_out_placeholder = None
        if state_out is not None:
            state_out_placeholder = from_dlpack(state_out, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        d_placeholder = None
        if d is not None:
            d_placeholder = from_dlpack(d, assumed_align=4).mark_layout_dynamic(leading_dim=len(d.shape) - 1)
        warmup_forward_cache[key] = cute.compile(
            warmup_forward_host,
            facts.split,
            facts.b_t,
            facts.overhead_chunks,
            facts.warmup_cap,
            cutlass.Int32(facts.n_heads_out),
            facts.num_sms,
            io_dtype,
            not split,
            prefill_cfg,
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(facts.ideal_chunks),
            cutlass.Int32(facts.batch_size),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_walk_ctas),
            cutlass.Int32(int(checkpoint_every_n_tokens)),
            cutlass.Float32(float(dt_min)),
            cutlass.Float32(float(dt_max)),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            gate_table_placeholder,
            from_dlpack(a, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(a, assumed_align=4).mark_layout_dynamic() if split else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            dt_bias_table_placeholder,
            d_placeholder,
            from_dlpack(z, assumed_align=16).mark_layout_dynamic(leading_dim=2) if z is not None else None,
            from_dlpack(o, assumed_align=16).mark_layout_dynamic(leading_dim=2) if o is not None else None,
            from_dlpack(o_pre, assumed_align=16).mark_layout_dynamic(leading_dim=2) if o_pre is not None else None,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            state_in_placeholder,
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
            from_dlpack(transition, assumed_align=4).mark_layout_dynamic(leading_dim=1) if transition is not None else None,
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            cuda.CUstream(int(stream)),
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
    a,
    dt_bias,
    d,
    z,
    o,
    o_pre,
    cu_seqlens,
    state_in,
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
    dt_min,
    dt_max,
    stream,
    transition=None,
) -> None:
    """Replay the warmup or uncut forward: one crossing into the DSL for the table, prologue and prefill launches."""
    compiled(
        facts.n_heads_out,
        facts.n_tiles,
        facts.ideal_chunks,
        facts.batch_size,
        facts.log2_threshold,
        facts.n_scan_ctas,
        facts.n_walk_ctas,
        int(checkpoint_every_n_tokens),
        float(dt_min),
        float(dt_max),
        q,
        k,
        v,
        gate,
        gate if facts.split else None,
        a,
        a if facts.split else None,
        dt_bias,
        dt_bias,
        d,
        z,
        o,
        o_pre,
        cu_seqlens,
        cu_seqlens,
        state_in,
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
        transition,
        workspace,
        cuda.CUstream(int(stream)),
    )


@cute.jit
def recompute_host(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    order_gen: cutlass.Constexpr[bool],
    recompute_cfg: cutlass.Constexpr,
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    a: cute.Tensor,
    a_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    state_in: Optional[cute.Tensor],
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
    transition: Optional[cute.Tensor],
    workspace: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """The state chain alone: the split-K table (warmup only), the recompute prologue and the recompute kernel."""
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
    mamba2_recompute_f16.prologue(
        io_dtype,
        b_t,
        order_gen,
        k,
        v,
        cu_seqlens,
        checkpoints,
        staging,
        work_count,
        work_items,
        scheduler,
        checkpoint_every_n,
        workspace,
        stream,
    )
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
        state_out,
        seed_indices,
        final_indices,
        work_items,
        work_count,
        scheduler,
        transition,
        checkpoint_every_n,
        dt_min,
        dt_max,
        workspace,
        stream,
    )


def build_recompute(
    *,
    k,
    v,
    gate,
    a,
    dt_bias,
    cu_seqlens,
    state_in,
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
    dt_softplus,
    dt_min,
    dt_max,
    checkpoint_every_n_tokens,
    device,
    stream,
    transition=None,
):
    """Compile (cached per static config) the recompute launch (the state chain alone) over the buffers of one plan."""
    _HK, DK = k.shape[1], k.shape[2]
    _HV, DV = v.shape[1], v.shape[2]
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
    io_dtype = get_dtype(k.dtype)
    state_source = state_in if state_in is not None else state_out
    state_dtype = get_dtype(state_source.dtype) if state_source is not None else cutlass.Float32
    key = (
        str(k.dtype),
        str(gate.dtype),
        str(a.dtype),
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        str(state_source.dtype) if state_source is not None else "none",
        int(device),
        int(num_sm),
        DK,
        DV,
        int(b_t),
        bool(split),
        bool(dt_softplus),
        state_in is not None,
        state_out is not None,
        seed_indices is not None,
        final_indices is not None,
        int(checkpoint_every_n_tokens) > 0,
        str(transition.dtype) if transition is not None else "none",
    )
    if key not in recompute_cache:
        recompute_cfg = mamba2_recompute_f16.build_cfg(
            io_dtype,
            state_dtype,
            max_active_clusters=num_sm,
            use_initial_state=state_in is not None,
            store_final_state=state_out is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            dt_softplus=dt_softplus,
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
        state_out_placeholder = None
        if state_out is not None:
            state_out_placeholder = from_dlpack(state_out, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        recompute_cache[key] = cute.compile(
            recompute_host,
            facts.split,
            facts.b_t,
            facts.overhead_chunks,
            facts.warmup_cap,
            cutlass.Int32(facts.n_heads_out),
            facts.num_sms,
            io_dtype,
            not split,
            recompute_cfg,
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(facts.ideal_chunks),
            cutlass.Int32(facts.batch_size),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_walk_ctas),
            cutlass.Int32(int(checkpoint_every_n_tokens)),
            cutlass.Float32(float(dt_min)),
            cutlass.Float32(float(dt_max)),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            gate_table_placeholder,
            from_dlpack(a, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(a, assumed_align=4).mark_layout_dynamic() if split else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            dt_bias_table_placeholder,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            state_in_placeholder,
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
            from_dlpack(transition, assumed_align=4).mark_layout_dynamic(leading_dim=1) if transition is not None else None,
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            cuda.CUstream(int(stream)),
            options="--enable-tvm-ffi --opt-level 2",
        )
    compiled = recompute_cache[key]
    return compiled, facts


def run_recompute(
    compiled,
    facts,
    *,
    k,
    v,
    gate,
    a,
    dt_bias,
    cu_seqlens,
    state_in,
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
    dt_min,
    dt_max,
    stream,
    transition=None,
) -> None:
    """Replay the recompute: one crossing into the DSL for the table, prologue and kernel launches."""
    compiled(
        facts.n_heads_out,
        facts.n_tiles,
        facts.ideal_chunks,
        facts.batch_size,
        facts.log2_threshold,
        facts.n_scan_ctas,
        facts.n_walk_ctas,
        int(checkpoint_every_n_tokens),
        float(dt_min),
        float(dt_max),
        k,
        v,
        gate,
        gate if facts.split else None,
        a,
        a if facts.split else None,
        dt_bias,
        dt_bias,
        cu_seqlens,
        cu_seqlens,
        state_in,
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
        transition,
        workspace,
        cuda.CUstream(int(stream)),
    )
