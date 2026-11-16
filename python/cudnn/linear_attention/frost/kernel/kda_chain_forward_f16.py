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

"""One compiled KDA forward host: prologue, H/M summary, FP32 chain and prefill.

The prologue constructs immutable piece metadata once. Bounded independent-head
waves reuse private H/M/X storage; each wave regenerates its queues and owns two
counts separate from the row tables. The next queue launch waits for completion
of the preceding prefill before reusing storage. Caller seeds, final states and
checkpoints retain their full-head layout and per-head arithmetic is unchanged.

Buffers consumed through different placeholder types are passed twice: H/M/X as
summary/prefill views and state-chain device views, and the piece prefixes with
the alignments required by their consumers. Compile keys include the bounded
head divisor; head count, piece count, addresses and stream remain runtime data.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.buffers import DeviceView

from cudnn.frost.compiled_cache import compile_cached

from ..common.host import get_dtype
from ..common.head_wave import state_wave_items
from ..common.piece_chain import dtype_name, launch_state_chain
from . import kda_chain_prologue_f16, kda_prefill_f16, kda_summary_f16

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
    "state_wave_counts",
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
    state_head_divisor: cutlass.Constexpr[int],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    scale: cutlass.Float32,
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
    state_wave_counts: Optional[cute.Tensor],
    stream: cuda.CUstream,
) -> None:
    kda_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
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
        o,
        None,
        checkpoints,
        None,
        None,
        None,
        None,
        stream,
    )
    for state_phase in cutlass.range(state_head_divisor):
        active_main_count = main_count
        active_summary_count = summary_count
        if cutlass.const_expr(state_head_divisor > 1):
            active_main_count = cute.make_tensor(state_wave_counts.iterator, cute.make_layout((1,)))
            active_summary_count = cute.make_tensor(state_wave_counts.iterator + 1, cute.make_layout((1,)))
            state_wave_items(
                state_head_divisor,
                b_t,
                state_phase,
                heads_out,
                cu_pieces,
                main_rows,
                summary_rows,
                work_items,
                work_items_summary,
                active_main_count,
                active_summary_count,
                scheduler_all,
            ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream, use_pdl=True)
        kda_summary_f16.host(
            summary_cfg,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            beta,
            cu_pieces_main,
            None,
            None,
            state_h_summary,
            state_m_summary,
            work_items_summary,
            active_summary_count,
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
            head_divisor=state_head_divisor,
            head_phase=state_phase,
        )
        kda_prefill_f16.host(
            prefill_cfg,
            q,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            beta,
            cu_pieces_main,
            state_x_prefill,
            None,
            o,
            final_state,
            None,
            final_indices,
            work_items,
            active_main_count,
            scheduler_prefill,
            prefill_words,
            checkpoint_every_n,
            scale,
            stream,
        )


def build_configs(io_dtype, state_dtype, gate_dtype, *, store_final_state, enable_checkpoints, checkpoint_dtype=None, **flags):
    summary_cfg = kda_summary_f16.build_cfg(io_dtype, gate_dtype, use_initial_state=False, **flags)
    prefill_cfg = kda_prefill_f16.build_cfg(
        io_dtype,
        state_dtype,
        gate_dtype,
        use_initial_state=True,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        checkpoint_dtype=checkpoint_dtype,
        **flags,
    )
    return summary_cfg, prefill_cfg


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
    checkpoint_every_n_tokens,
    scale,
    chain_rows,
    device,
    num_sm,
    stream,
    state_head_divisor=1,
    state_wave_counts=None,
):
    """Compile a forward chain for static dtypes, dimensions, gates, output presence and head divisor.

    Piece/head/sequence counts and buffer addresses are runtime data. A divisor
    greater than one requires compact H/M/X views and two caller-owned wave
    counters; public states and checkpoints retain the original head count.
    The prepared host launches on the supplied stream without allocating.
    """
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
    gate_scale_log2 = float(gate_lower_bound) * kda_summary_f16.LOG2_E
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
        int(checkpoint_every_n_tokens) > 0,
        str(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else "none",
        final_state is not None,
        has_seed,
        composed is not None,
        seed_indices is not None,
        final_indices is not None,
        int(chain_rows),
        int(state_head_divisor),
    )
    if key not in chain_forward_cache:
        summary_cfg, prefill_cfg = build_configs(
            io_dtype,
            state_dtype,
            gate_dtype,
            store_final_state=final_state is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            checkpoint_dtype=get_dtype(checkpoints.dtype) if int(checkpoint_every_n_tokens) > 0 else None,
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=gate_scale_log2,
            log_gate=log_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            max_active_clusters=num_sm,
            d_k=DK,
            d_v=DV,
        )

        summary_cfg.state_head_divisor = state_head_divisor
        prefill_cfg.state_head_divisor = state_head_divisor

        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_summary_placeholder = from_dlpack(work_items_summary, assumed_align=16)
        work_items_summary_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
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
            int(state_head_divisor),
            cutlass.Int32(int(pieces)),
            cutlass.Int32(int(heads_out)),
            cutlass.Int32(int(num_seqs)),
            cutlass.Int32(int(checkpoint_every_n_tokens)),
            float(scale),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=1),
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
            from_dlpack(state_wave_counts, assumed_align=4).mark_layout_dynamic() if state_wave_counts is not None else None,
            cuda.CUstream(int(stream)),
            cache_key=repr((__name__, key)),
            symbol="frost_la_kda_chain_forward_f16",
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
    state_wave_counts=None,
) -> None:
    """Launch the compiled chain with the current buffers and stream."""
    buffers = locals()
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        int(checkpoint_every_n_tokens),
        float(scale),
        *(buffers[name] for name in CHAIN_FORWARD_BUFFERS),
        cuda.CUstream(int(stream)),
    )
