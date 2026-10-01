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

"""One compiled launch for the GDN-2 uncut and d_v-split forwards: the prefill prologue and the prefill issued from a
single host.  Every kernel, its host and the tensor placeholder each host was compiled with are the standalone modules'
own; this host only sequences the launches, so the kernels' SASS is unchanged and the Python side crosses into the DSL
once per call instead of twice.  A buffer that two hosts read through different placeholder types is passed twice, once
per type: the prologue marks beta at 16 bytes and the prefill at 4; the prologue and prefill otherwise mark the same
buffers as the standalone prefill wrapper does."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..common.host import get_dtype
from . import gdn2_prefill_f16, gdn2_prep_f16, gdn2_prep_prefill_f16

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
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    beta: cute.Tensor,
    beta_prefill: cute.Tensor,
    w: cute.Tensor,
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    state_in: Optional[cute.Tensor],
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
    stream: cuda.CUstream,
) -> None:
    if cutlass.const_expr(prep):
        gdn2_prep_prefill_f16.prologue(
            io_dtype,
            b_t,
            q,
            k,
            v,
            gate,
            beta,
            w,
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
            beta,
            prep_words,
            prep_rows,
            prep_row_count,
        )
        gdn2_prep_f16.host(
            prep_cfg,
            q,
            k,
            prep_words,
            gate,
            a_log,
            dt_bias,
            cu_seqlens,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            prep_a,
            prep_diag,
            prep_rows,
            prep_row_count,
            stream,
        )
    else:
        gdn2_prefill_f16.prologue(
            io_dtype,
            b_t,
            q,
            k,
            v,
            gate,
            beta,
            w,
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
        )
    prefill_host = gdn2_prep_prefill_f16.host if cutlass.const_expr(prep) else gdn2_prefill_f16.host
    prefill_host(
        prefill_cfg,
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta_prefill,
        w,
        cu_seqlens,
        state_in,
        o,
        state_out,
        seed_indices,
        final_indices,
        work_items,
        work_count,
        scheduler,
        workspace,
        checkpoint_every_n,
        scale,
        stream,
    )


def build_uncut_forward(
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
    state_in,
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
    beta_guard,
    checkpoint_every_n_tokens,
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
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(beta.dtype),
        str(w.dtype),
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
        bool(beta_guard),
        state_in is not None,
        state_out is not None,
        seed_indices is not None,
        final_indices is not None,
        int(checkpoint_every_n_tokens) > 0,
        int(tiles_per_head),
        bool(prep),
    )
    if key not in uncut_forward_cache:
        prefill_module = gdn2_prep_prefill_f16 if prep else gdn2_prefill_f16
        prefill_cfg = prefill_module.build_cfg(
            io_dtype,
            state_dtype,
            gate_dtype,
            use_initial_state=state_in is not None,
            store_final_state=state_out is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=float(gate_lower_bound) * gdn2_prefill_f16.LOG2_E,
            log_gate=log_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            beta_guard=beta_guard,
            max_active_clusters=num_sm,
            d_k=DK,
            d_v=DV // tiles_per_head,
            tiles_per_head=tiles_per_head,
        )
        prep_cfg = None
        if prep:
            prep_cfg = gdn2_prep_f16.build_cfg(
                io_dtype,
                gate_dtype,
                num_sm=num_sm,
                l2norm=use_qk_l2norm,
                safe_gate=safe_gate,
                gate_scale_log2=float(gate_lower_bound) * gdn2_prefill_f16.LOG2_E,
                log_gate=log_gate,
                beta_sigmoid=use_beta_sigmoid,
                allow_neg_eigval=allow_neg_eigval,
                d_k=DK,
                beta_guard=beta_guard,
            )
        prep_placeholders = [None] * 8
        if prep:
            prep_placeholders = [from_dlpack(rec, assumed_align=128).mark_layout_dynamic(leading_dim=3) for rec in (prep_k_decay, prep_q_decay, prep_t)]
            prep_placeholders.append(from_dlpack(prep_a, assumed_align=16).mark_layout_dynamic(leading_dim=2))
            prep_placeholders.append(from_dlpack(prep_diag, assumed_align=16).mark_layout_dynamic(leading_dim=2))
            prep_placeholders.append(from_dlpack(prep_words, assumed_align=128).mark_layout_dynamic())
            prep_placeholders.append(from_dlpack(prep_rows, assumed_align=16).mark_layout_dynamic(leading_dim=1))
            prep_placeholders.append(from_dlpack(prep_row_count, assumed_align=4).mark_layout_dynamic())

        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        uncut_forward_cache[key] = cute.compile(
            uncut_forward_host,
            int(b_t),
            int(num_sm),
            io_dtype,
            prefill_cfg,
            int(tiles_per_head),
            cutlass.Int32(int(checkpoint_every_n_tokens)),
            float(scale),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=2),
            from_dlpack(w, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(o, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(state_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if state_in is not None else None,
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
            cuda.CUstream(int(stream)),
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
    w,
    a_log,
    dt_bias,
    o,
    cu_seqlens,
    state_in,
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
) -> None:
    """Replay the uncut forward: one crossing into the DSL for the prologue and prefill launches.  The plan validated the
    contract at build, so nothing here raises."""
    compiled(
        int(checkpoint_every_n_tokens),
        float(scale),
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        beta,
        w,
        o,
        cu_seqlens,
        state_in,
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
        cuda.CUstream(int(stream)),
    )
