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
from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids

from ..common.host import get_dtype
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, gen_interval_items, order_body
from ..common.state_snapshot import snapshot_states
from . import gdn2_bprop_f16, gdn2_recompute_f16, gdn2_summary_f16

USE_PDL = True

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
    if bidx == cutlass.Int32(0):
        if cutlass.const_expr(coarse):
            gen_interval_items(
                b_t, ORDER_THREADS, tidx, n_heads_out, n_heads_out * n_batch, seed_span_chunks, cu_seqlens, series_count, series_items, scheduler_recompute
            )
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
                n_heads_out * n_batch,
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
                n_heads_out * n_batch,
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
    dgate: cute.Tensor,
    dw: cute.Tensor,
    dbeta: cute.Tensor,
    dstate0: Optional[cute.Tensor],
    dstate_in: Optional[cute.Tensor],
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
) -> None:
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
        work_items,
        work_count,
        scheduler_bwd,
        bprop_words,
        dstate_scratch,
        scale,
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
    use_initial_state,
    dgate,
    dw,
    dbeta,
    dstate0,
    dstate_in,
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
                use_initial_state=state_in is not None,
                store_final_state=False,
                enable_checkpoints=True,
                seed_checkpoints=coarse,
                d_v=DV,
                **flags,
            )
        bprop_cfg = gdn2_bprop_f16.build_cfg(
            io_dtype,
            gate_dtype,
            use_dstate_in=dstate_in is not None,
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
            from_dlpack(dgate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dw, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dbeta, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate0 is not None else None,
            from_dlpack(dstate_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate_in is not None else None,
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
    dgate,
    dw,
    dbeta,
    dstate0,
    dstate_in,
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
        dgate,
        dw,
        dbeta,
        dstate0,
        dstate_in,
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
    )
