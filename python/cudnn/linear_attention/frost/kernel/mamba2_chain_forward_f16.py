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

from ..common.host import get_dtype
from ..common.piece_chain import dtype_name, launch_scalar_state_chain
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
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
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
    seed_indices: Optional[cute.Tensor],
    final_state: Optional[cute.Tensor],
    final_indices: Optional[cute.Tensor],
    checkpoints: Optional[cute.Tensor],
    stream: cuda.CUstream,
) -> None:
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
        stream,
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
        state_h,
        None,
        None,
        work_items_summary,
        summary_count,
        scheduler_summary,
        transition,
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
        has_seed,
        False,
        num_seqs,
        state_h,
        transition,
        state_x,
        seed,
        main_rows,
        seed_indices,
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
        cu_pieces,
        state_x,
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


def build_configs(io_dtype, *, store_final_state, enable_checkpoints, dt_softplus, has_d, d_per_channel, has_z, store_pre_gate, max_active_clusters, d_k, d_v):
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
    chain_rows,
    device,
    num_sm,
    stream,
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
        final_state is not None,
        has_seed,
        seed_indices is not None,
        final_indices is not None,
        int(chain_rows),
    )
    if key not in chain_forward_cache:
        summary_cfg, prefill_cfg = build_configs(
            io_dtype,
            store_final_state=final_state is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
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
            cutlass.Int32(int(pieces)),
            cutlass.Int32(int(heads_out)),
            cutlass.Int32(int(num_seqs)),
            cutlass.Int32(int(checkpoint_every_n_tokens)),
            cutlass.Float32(float(dt_min)),
            cutlass.Float32(float(dt_max)),
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
            from_dlpack(seed_indices, assumed_align=4).mark_layout_dynamic() if seed_indices is not None else None,
            from_dlpack(final_state, assumed_align=16).mark_layout_dynamic(leading_dim=3) if final_state is not None else None,
            from_dlpack(final_indices, assumed_align=4).mark_layout_dynamic() if final_indices is not None else None,
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if int(checkpoint_every_n_tokens) > 0 else None,
            cuda.CUstream(int(stream)),
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
        *(buffers[name] for name in CHAIN_FORWARD_BUFFERS),
        cuda.CUstream(int(stream)),
    )
