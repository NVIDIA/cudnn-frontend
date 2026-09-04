# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host side of the CAKE C16 recurrent-KDA training route: work-item planning,
TMA descriptors, workspace carve and the launch sequences for the KDA and
KDA_BWD nodes.

This is the FlashInfer ``_kda_training_impl`` / ``flashkda_training_paired_binding``
host logic for the C16 route, restated over ``cuda.bindings`` and the FE
engine contract. The kernels are the frozen bodies in ``kernels/``; nothing
here changes their arithmetic.

Route facts the code below relies on:

* Forward writes the per-chunk (16-token) bf16 state checkpoints the backward
  consumes; they ride the op's ``state_checkpoints`` port, which is why the
  engine requires ``checkpoint_every_n_tokens == 16``. Forward also produces a
  ``beta_active`` tape (sigmoid(beta) in bf16) and a possibly gate-refined
  ``work_items`` table; the backward regenerates both from the inputs instead of
  saving them (``fe_cake_kda_beta_active``, and the same copy + refine kernel).
* Work items are planned on the host from the sequence lengths, so every
  execute reads ``cu_seqlens`` back (one stream sync). Plans are cached per
  length tuple.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from cudnn.frost.device import compute_capability, current_device, device_context, multiprocessor_count, shared_memory_per_block_optin
from cudnn.frost.workspace import WorkspaceLayout, carve_plan

from . import compiler

HEAD_DIM = 128
CHUNK = 16
WORK_ITEM_FIELDS = 8
MAX_PIECES = 2048
BLOCK = 512
FWD_SMEM = 230016
BWD_SMEM = 230400
REFINE_SMEM = 128
REDUCE_SMEM = 4224
LOG2_THRESHOLD = -14.426950408889635
LOWER_BOUND = -5.0
SCALE = 1.0 / math.sqrt(HEAD_DIM)
STATE_WORDS = HEAD_DIM * HEAD_DIM  # one head's fp32 recurrent state, in 32-bit words

BODIES = {
    "c16": "flashkda_training_c16.cu",
    "fwd_aligned": "cake_flashkda_training_c16_aligned_forward_36075669f2.cu",
    "bwd_aligned": "cake_flashkda_training_c16_aligned_backward_0f8187e742.cu",
    "reduce_aligned": "cake_flashkda_training_c16_aligned_param_reduce_be120a1e72.cu",
    "aux": "flashkda_training_aux.cu",
    "helpers": "fe_cake_kda_helpers.cu",
}
KERNEL_BODY = {
    "kernel_flashkda_forward_checkpoint_c16": "c16",
    "kernel_flashkda_backward_persistent_c16": "c16",
    "kernel_cake_flashkda_forward_checkpoint_c16_aligned": "fwd_aligned",
    "kernel_cake_flashkda_backward_persistent_c16_aligned": "bwd_aligned",
    "kernel_cake_flashkda_backward_param_reduce_c16_aligned": "reduce_aligned",
    "kernel_flashkda_refine_forgetting_horizons": "aux",
    "kernel_flashkda_backward_param_reduce_c16_partial": "aux",
    "kernel_flashkda_grouped_qk_reduce": "aux",
    "fe_cake_kda_beta_active": "helpers",
}


# ---------------------------------------------------------------------------
# Planning (pure Python, no device access)
# ---------------------------------------------------------------------------


def _grid_cost(groups: Sequence[Tuple[int, int]], resident: int, item_setup: float, chunk_compute: float, chunk_memory: float) -> float:
    service = max(chunk_compute, chunk_memory)
    total_items = 0
    total_work = 0.0
    largest = 0.0
    for items, chunks_per_item in groups:
        item_work = item_setup + chunks_per_item * service
        total_items += items
        total_work += items * item_work
        largest = max(largest, item_work)
    waves = total_items / resident
    tail = math.ceil(waves) - waves
    return total_work / resident + tail * largest


def c16_use_split(seq_lens: Sequence[int], num_v_heads: int, resident_sms: int) -> bool:
    """The C16 route's own split decision (the cost model FlashInfer's dispatcher
    evaluates for this template)."""
    counts = [(length + CHUNK - 1) // CHUNK for length in seq_lens]
    unsplit = _grid_cost([(num_v_heads, count) for count in counts], resident_sms, 16.0, 1.0, 0.75)
    target = max(1, math.ceil(sum(counts) * num_v_heads / resident_sms))
    groups = []
    boundaries = 0
    for count in counts:
        pieces = min(count, max(1, math.ceil(count / target)))
        groups.append((num_v_heads * pieces, math.ceil(count / pieces)))
        boundaries += num_v_heads * (pieces - 1)
    split = _grid_cost(groups, resident_sms, 16.0, 1.0, 0.75) + boundaries * 8.0 / resident_sms
    return split < unsplit


@dataclass(frozen=True)
class C16Plan:
    seq_lens: Tuple[int, ...]
    offsets: Tuple[int, ...]
    total_tokens: int
    total_chunks: int
    aligned: bool
    checkpoint_starts: Tuple[int, ...]
    work_rows: Tuple[int, ...]  # WORK_ITEM_FIELDS ints per item
    boundaries: Tuple[int, ...]  # pairs of item indices; (0, 0) when there are none
    total_work_items: int
    uniform_work_items: int
    boundary_count: int


def plan_c16(seq_lens: Sequence[int], num_v_heads: int, resident_sms: int) -> C16Plan:
    """Work items, split boundaries and checkpoint offsets for one length tuple."""
    seq_lens = tuple(int(length) for length in seq_lens)
    if not seq_lens or min(seq_lens) <= 0:
        raise ValueError(f"kda_cake: every sequence must be non-empty, got lengths {seq_lens}")
    chunk_counts = tuple((length + CHUNK - 1) // CHUNK for length in seq_lens)
    offsets = [0]
    checkpoint_starts = [0]
    for length, count in zip(seq_lens, chunk_counts):
        offsets.append(offsets[-1] + length)
        checkpoint_starts.append(checkpoint_starts[-1] + count)
    total_tokens = offsets[-1]
    split = c16_use_split(seq_lens, num_v_heads, resident_sms)
    n_tiles = len(seq_lens) * num_v_heads
    ideal_tokens = (total_tokens * num_v_heads + resident_sms - 1) // resident_sms
    ideal_chunks = max(1, (ideal_tokens + CHUNK - 1) // CHUNK)

    rows = []  # (sequence, head, piece, write_start, write_end, compute_start, compute_end, bos, eos)
    pieces_per_sequence = []
    for sequence, count in enumerate(chunk_counts):
        piece_count = 1
        if split:
            piece_cap = max(1, min(count, MAX_PIECES))
            piece_count = max(1, min((count + ideal_chunks - 1) // ideal_chunks, piece_cap))
            if n_tiles < 2 * resident_sms:
                piece_start = max(1, piece_count - 8)
                best_cost = 2**31 - 1
                for delta in range(16):
                    candidate = min(piece_start + delta, piece_cap)
                    span = (count + candidate - 1) // candidate
                    waves = (n_tiles * candidate + resident_sms - 1) // resident_sms
                    cost = waves * (span + 16)
                    if cost < best_cost:
                        best_cost, piece_count = cost, candidate
                unsplit_cost = ((n_tiles + resident_sms - 1) // resident_sms) * (count + 16)
                if 4 * best_cost > 3 * unsplit_cost:
                    piece_count = 1
        span = (count + piece_count - 1) // piece_count
        num_pieces = (count + span - 1) // span
        pieces_per_sequence.append(num_pieces)
        for head in range(num_v_heads):
            for piece in range(num_pieces):
                write_start = piece * span
                write_end = min(count, write_start + span)
                rows.append(
                    (
                        sequence,
                        head,
                        piece,
                        write_start,
                        write_end,
                        0 if piece == 0 else write_start,
                        count if piece + 1 == num_pieces else write_end,
                        offsets[sequence],
                        offsets[sequence + 1],
                    )
                )
    rows.sort(key=lambda row: row[4] - row[3], reverse=True)
    row_index = {(row[0], row[1], row[2]): index for index, row in enumerate(rows)}
    boundaries = []
    for sequence, pieces in enumerate(pieces_per_sequence):
        for head in range(num_v_heads):
            for piece in range(1, pieces):
                boundaries.append(row_index[(sequence, head, piece - 1)])
                boundaries.append(row_index[(sequence, head, piece)])
    work_rows = []
    for sequence, head, _piece, write_start, write_end, compute_start, compute_end, bos, eos in rows:
        work_rows.extend((sequence, head, write_start, write_end, compute_start, compute_end, bos, eos))
    boundary_count = len(boundaries) // 2
    return C16Plan(
        seq_lens=seq_lens,
        offsets=tuple(offsets),
        total_tokens=total_tokens,
        total_chunks=checkpoint_starts[-1],
        aligned=checkpoint_starts[-1] * CHUNK == total_tokens,
        checkpoint_starts=tuple(checkpoint_starts),
        work_rows=tuple(work_rows),
        boundaries=tuple(boundaries) if boundaries else (0, 0),
        total_work_items=len(rows),
        uniform_work_items=int(boundary_count == 0 and len(set(chunk_counts)) == 1),
        boundary_count=boundary_count,
    )


# ---------------------------------------------------------------------------
# Compiled plans
# ---------------------------------------------------------------------------


def _is_int64(view) -> bool:
    return "64" in str(view.dtype)


class _CakeKda:
    """State shared by the forward and backward plans: node geometry, device,
    the workspace carve and the per-length-tuple planning cache."""

    max_cached_plans = 64

    def __init__(self, node, is_bwd: bool):
        self.node = node
        self.is_bwd = is_bwd
        self.plan_name = "KdaCakeEngine (KDA_BWD)" if is_bwd else "KdaCakeEngine (KDA)"
        self.device = current_device()
        self.arch = compiler.arch_for_device(self.device)
        self.num_sm = multiprocessor_count(self.device)
        smem_needed = BWD_SMEM if is_bwd else FWD_SMEM
        if shared_memory_per_block_optin(self.device) < smem_needed:
            raise RuntimeError(f"{self.plan_name}: device exposes less than {smem_needed} bytes of opt-in shared memory")
        q, v, cu = node.inputs["q"], node.inputs["v"], node.inputs["cu_seqlens"]
        self.total = int(q.dim[0])
        self.h_qk = int(q.dim[1])
        self.h_v = int(v.dim[1])
        self.n_seq = int(cu.dim[0]) - 1
        self.grouped = self.h_v != self.h_qk
        self.beta_active_stride = max(self.h_v, 8)
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else SCALE
        lower_bound = node.params.get("gate_lower_bound")
        self.lower_bound = float(lower_bound) if lower_bound is not None else LOWER_BOUND
        # Bounds for the runtime-planned tables: every sequence has at most
        # len // 16 + 1 chunks, and never more work items than chunks per head.
        self.max_chunks = self.total // CHUNK + self.n_seq
        self.max_work_items = self.h_v * self.max_chunks
        self.state_words = self.n_seq * self.h_v * STATE_WORDS
        self._plans: Dict[Tuple[int, ...], C16Plan] = {}
        self._regions = []
        self._layout = WorkspaceLayout()

    # -- workspace ---------------------------------------------------------

    def _region(self, name: str, numel: int, dtype: str) -> None:
        itemsize = {"int32": 4, "int64": 8, "float32": 4, "bfloat16": 2}[dtype]
        offset = self._layout.add(max(int(numel), 1) * itemsize)
        self._regions.append((name, offset, dtype, (max(int(numel), 1),)))

    def _common_regions(self) -> None:
        self._region("cu_i64", self.n_seq + 1, "int64")
        self._region("ckpt_starts", self.n_seq + 1, "int64")
        self._region("base_items", self.max_work_items * WORK_ITEM_FIELDS, "int32")
        self._region("work_items", self.max_work_items * WORK_ITEM_FIELDS, "int32")
        self._region("boundaries", max(1, self.max_work_items) * 2, "int32")
        self._region("counters", self.h_v + 2, "int32")
        self._region("beta_active", self.total * self.beta_active_stride, "bfloat16")

    def _finish_layout(self) -> None:
        self.workspace_size = self._layout.size
        self.carve_names = [name for name, _offset, _dtype, _shape in self._regions]
        self.carve = carve_plan(self.plan_name, [(offset, dtype, shape) for _name, offset, dtype, shape in self._regions])

    def workspace_bytes(self) -> int:
        return self.workspace_size

    # -- planning ------------------------------------------------------------

    def _plan(self, cu_view, stream: int) -> C16Plan:
        itemsize = 8 if _is_int64(cu_view) else 4
        offsets = compiler.read_device_ints(cu_view.data_ptr(), self.n_seq + 1, itemsize, stream)
        seq_lens = tuple(right - left for left, right in zip(offsets, offsets[1:]))
        plan = self._plans.get(seq_lens)
        if plan is None:
            if offsets[0] != 0 or offsets[-1] != self.total:
                raise ValueError(f"{self.plan_name}: cu_seqlens must run from 0 to total_tokens ({self.total}), got {offsets[0]}..{offsets[-1]}")
            plan = plan_c16(seq_lens, self.h_v, self.num_sm)
            if len(self._plans) >= self.max_cached_plans:
                self._plans.clear()
            self._plans[seq_lens] = plan
        return plan

    def _upload_plan(self, plan: C16Plan, region: dict, stream: int) -> None:
        compiler.memcpy_htod(region["cu_i64"].data_ptr(), array("q", plan.offsets), stream)
        compiler.memcpy_htod(region["ckpt_starts"].data_ptr(), array("q", plan.checkpoint_starts), stream)
        rows = array("i", plan.work_rows)
        if plan.boundary_count:
            compiler.memcpy_htod(region["base_items"].data_ptr(), rows, stream)
            compiler.memcpy_htod(region["boundaries"].data_ptr(), array("i", plan.boundaries), stream)
        else:
            compiler.memcpy_htod(region["work_items"].data_ptr(), rows, stream)

    def _refine_work_items(self, plan: C16Plan, region: dict, g_ptr: int, a_log_ptr: int, dt_bias_ptr: int, stream: int) -> None:
        """Forward's work-item preparation, repeated verbatim by the backward:
        with split boundaries the table is refined by the gate's forgetting
        horizons, otherwise only the dynamic counter is reset."""
        counters = region["counters"].data_ptr()
        if plan.boundary_count:
            compiler.memcpy_dtod(region["work_items"].data_ptr(), region["base_items"].data_ptr(), plan.total_work_items * WORK_ITEM_FIELDS * 4, stream)
            compiler.memset_d32(counters, 0, self.h_v + 2, stream)
            params = compiler.Params()
            params.ptr(g_ptr).ptr(a_log_ptr).ptr(dt_bias_ptr).ptr(region["work_items"].data_ptr()).ptr(region["boundaries"].data_ptr()).ptr(counters)
            params.i32(self.h_v).f32(self.lower_bound).f32(LOG2_THRESHOLD)
            compiler.launch(
                self._kernel("kernel_flashkda_refine_forgetting_horizons", REFINE_SMEM),
                (plan.boundary_count, 1, 1),
                (128, 1, 1),
                REFINE_SMEM,
                stream,
                params,
                "forgetting-horizon refinement",
            )
        elif plan.uniform_work_items == 0:
            compiler.memset_d32(counters, 0, 1, stream)

    # -- kernels and descriptors ----------------------------------------------

    def _kernel(self, name: str, dynamic_smem: int):
        return compiler.library(BODIES[KERNEL_BODY[name]], self.device).function(name, dynamic_smem)

    def _token_map(self, address: int, heads: int, what: str):
        row = HEAD_DIM * 2
        return compiler.encode_tiled(address, (HEAD_DIM, heads, self.total), (row, row * heads), (64, 1, CHUNK), what=what)

    def _checkpoint_map(self, address: int, total_chunks: int):
        row = HEAD_DIM * 2
        state = row * HEAD_DIM
        return compiler.encode_tiled(
            address, (HEAD_DIM, HEAD_DIM, self.h_v, total_chunks), (row, state, state * self.h_v), (64, HEAD_DIM, 1, 1), what="state_checkpoints"
        )

    def _beta_map(self, address: int):
        return compiler.encode_tiled(
            address, (self.beta_active_stride, self.total), (self.beta_active_stride * 2,), (8, CHUNK), swizzle=compiler.SWIZZLE_NONE, what="beta_active"
        )

    def _grid(self, plan: C16Plan) -> Tuple[int, int, int]:
        return (min(plan.total_work_items, self.num_sm), 1, 1)

    def _prepare(self, stream: int):
        compiler.check_not_capturing(stream, self.plan_name)


class CompiledCakeKda(_CakeKda):
    """KDA node: the C16 forward with per-chunk checkpoints."""

    def __init__(self, node):
        super().__init__(node, is_bwd=False)
        self.has_initial_state = "initial_state" in node.inputs
        self.has_final_state = "final_state" in node.outputs
        self._common_regions()
        if not self.has_initial_state:
            self._region("state0_zero", self.state_words, "float32")
        if not self.has_final_state:
            self._region("final_scratch", self.state_words, "float32")
        self._finish_layout()

    def bind(self, names) -> None:
        pos = {name: index for index, name in enumerate(names)}
        self.index_q = pos["q"]
        self.index_k = pos["k"]
        self.index_v = pos["v"]
        self.index_g = pos["g"]
        self.index_beta = pos["beta"]
        self.index_cu = pos["cu_seqlens"]
        self.index_a_log = pos["a_log"]
        self.index_dt_bias = pos["dt_bias"]
        self.index_initial_state = pos.get("initial_state")
        self.index_o = pos["O"]
        self.index_final_state = pos.get("final_state")
        self.index_state_checkpoints = pos["state_checkpoints"]

    def run(self, views, workspace, stream) -> None:
        stream = int(stream) if stream is not None else 0
        with device_context(self.device):
            self._prepare(stream)
            q, k, v, g, beta = (views[i] for i in (self.index_q, self.index_k, self.index_v, self.index_g, self.index_beta))
            cu = views[self.index_cu]
            a_log, dt_bias = views[self.index_a_log], views[self.index_dt_bias]
            o = views[self.index_o]
            state_checkpoints = views[self.index_state_checkpoints]
            region = dict(zip(self.carve_names, workspace.carve(self.carve)))
            plan = self._plan(cu, stream)
            if plan.total_chunks > int(state_checkpoints.shape[0]):
                raise ValueError(f"{self.plan_name}: state_checkpoints has {state_checkpoints.shape[0]} rows, the route needs {plan.total_chunks}")
            self._upload_plan(plan, region, stream)
            self._refine_work_items(plan, region, g.data_ptr(), a_log.data_ptr(), dt_bias.data_ptr(), stream)

            if self.index_initial_state is not None:
                state0_ptr = views[self.index_initial_state].data_ptr()
            else:
                state0_ptr = region["state0_zero"].data_ptr()
                compiler.memset_d32(state0_ptr, 0, self.state_words, stream)
            final_ptr = views[self.index_final_state].data_ptr() if self.index_final_state is not None else region["final_scratch"].data_ptr()

            q_map = self._token_map(q.data_ptr(), self.h_qk, "q")
            k_map = self._token_map(k.data_ptr(), self.h_qk, "k")
            v_map = self._token_map(v.data_ptr(), self.h_v, "v")
            g_map = self._token_map(g.data_ptr(), self.h_v, "g")
            o_map = self._token_map(o.data_ptr(), self.h_v, "O")
            checkpoint_map = self._checkpoint_map(state_checkpoints.data_ptr(), plan.total_chunks)

            params = compiler.Params()
            params.ptr(region["counters"].data_ptr())
            params.tensor_map(q_map).tensor_map(k_map).tensor_map(v_map).tensor_map(g_map)
            params.ptr(g.data_ptr())
            params.tensor_map(o_map)
            if not plan.aligned:
                params.ptr(o.data_ptr())
            params.tensor_map(checkpoint_map)
            params.ptr(beta.data_ptr()).ptr(region["beta_active"].data_ptr())
            params.ptr(a_log.data_ptr()).ptr(dt_bias.data_ptr())
            params.ptr(region["cu_i64"].data_ptr()).ptr(region["ckpt_starts"].data_ptr()).ptr(region["work_items"].data_ptr())
            params.ptr(state0_ptr).ptr(final_ptr)
            params.i32(plan.total_work_items).i32(plan.uniform_work_items).i32(self.h_qk).i32(self.h_v).i32(self.beta_active_stride).i32(CHUNK)
            params.f32(self.scale).f32(self.lower_bound)
            name = "kernel_cake_flashkda_forward_checkpoint_c16_aligned" if plan.aligned else "kernel_flashkda_forward_checkpoint_c16"
            compiler.launch(self._kernel(name, FWD_SMEM), self._grid(plan), (BLOCK, 1, 1), FWD_SMEM, stream, params, name)


class CompiledCakeKdaBwd(_CakeKda):
    """KDA_BWD node: the persistent C16 backward, the parameter-gradient
    reduction, and the grouped-head reduction of dQ/dK."""

    def __init__(self, node):
        super().__init__(node, is_bwd=True)
        self.has_d_final_state = "d_final_state" in node.inputs
        self.wants_d_initial_state = "d_initial_state" in node.outputs
        self._common_regions()
        self._region("dlog_decay", self.total * self.h_v * HEAD_DIM, "float32")
        self._region("dlog_boundary", self.max_chunks * self.h_v * HEAD_DIM, "float32")
        self._region("dbeta_active", self.total * self.h_v, "float32")
        self._region("gate_part_a", 128 * self.h_v * HEAD_DIM, "float32")
        self._region("gate_part_dt", 128 * self.h_v * HEAD_DIM, "float32")
        self._region("dummy_u32", 1, "int32")
        self._region("dummy_f32", 1, "float32")
        if self.grouped:
            self._region("dq_value", self.total * self.h_v * HEAD_DIM, "bfloat16")
            self._region("dk_value", self.total * self.h_v * HEAD_DIM, "bfloat16")
        if not self.has_d_final_state:
            self._region("dfinal_zero", self.state_words, "float32")
        if not self.wants_d_initial_state:
            self._region("dinit_scratch", self.state_words, "float32")
        self._finish_layout()

    def bind(self, names) -> None:
        pos = {name: index for index, name in enumerate(names)}
        self.index_q = pos["q"]
        self.index_k = pos["k"]
        self.index_v = pos["v"]
        self.index_g = pos["g"]
        self.index_beta = pos["beta"]
        self.index_cu = pos["cu_seqlens"]
        self.index_do = pos["dO"]
        self.index_state_checkpoints = pos["state_checkpoints"]
        self.index_d_final_state = pos.get("d_final_state")
        self.index_a_log = pos["a_log"]
        self.index_dt_bias = pos["dt_bias"]
        self.index_dq = pos["dQ"]
        self.index_dk = pos["dK"]
        self.index_dv = pos["dV"]
        self.index_dg = pos["dG"]
        self.index_dbeta = pos["dBeta"]
        self.index_d_initial_state = pos.get("d_initial_state")
        self.index_d_a_log = pos["d_a_log"]
        self.index_d_dt_bias = pos["d_dt_bias"]

    def run(self, views, workspace, stream) -> None:
        stream = int(stream) if stream is not None else 0
        with device_context(self.device):
            self._prepare(stream)
            q, k, v, g, beta = (views[i] for i in (self.index_q, self.index_k, self.index_v, self.index_g, self.index_beta))
            cu = views[self.index_cu]
            do = views[self.index_do]
            state_checkpoints = views[self.index_state_checkpoints]
            a_log, dt_bias = views[self.index_a_log], views[self.index_dt_bias]
            dq, dk, dv, dg, dbeta = (views[i] for i in (self.index_dq, self.index_dk, self.index_dv, self.index_dg, self.index_dbeta))
            d_a_log, d_dt_bias = views[self.index_d_a_log], views[self.index_d_dt_bias]
            region = dict(zip(self.carve_names, workspace.carve(self.carve)))
            plan = self._plan(cu, stream)
            if plan.total_chunks > int(state_checkpoints.shape[0]):
                raise ValueError(f"{self.plan_name}: state_checkpoints has {state_checkpoints.shape[0]} rows, the route needs {plan.total_chunks}")
            self._upload_plan(plan, region, stream)
            self._refine_work_items(plan, region, g.data_ptr(), a_log.data_ptr(), dt_bias.data_ptr(), stream)

            # The forward's beta_active tape, regenerated from beta.
            beta_active = region["beta_active"].data_ptr()
            count = self.total * self.beta_active_stride
            params = compiler.Params()
            params.ptr(beta.data_ptr()).ptr(beta_active).i64(self.total).i32(self.h_v).i32(self.beta_active_stride)
            compiler.launch(
                self._kernel("fe_cake_kda_beta_active", 0), ((count + 255) // 256, 1, 1), (256, 1, 1), 0, stream, params, "beta_active regeneration"
            )

            if self.index_d_final_state is not None:
                dfinal_ptr = views[self.index_d_final_state].data_ptr()
            else:
                dfinal_ptr = region["dfinal_zero"].data_ptr()
                compiler.memset_d32(dfinal_ptr, 0, self.state_words, stream)
            dinit_ptr = views[self.index_d_initial_state].data_ptr() if self.index_d_initial_state is not None else region["dinit_scratch"].data_ptr()
            dq_value_ptr = region["dq_value"].data_ptr() if self.grouped else dq.data_ptr()
            dk_value_ptr = region["dk_value"].data_ptr() if self.grouped else dk.data_ptr()
            counters = region["counters"].data_ptr()
            compiler.memset_d32(counters + 4, 0, self.h_v + 1, stream)

            q_map = self._token_map(q.data_ptr(), self.h_qk, "q")
            k_map = self._token_map(k.data_ptr(), self.h_qk, "k")
            g_map = self._token_map(g.data_ptr(), self.h_v, "g")
            do_map = self._token_map(do.data_ptr(), self.h_v, "dO")
            v_map = self._token_map(v.data_ptr(), self.h_v, "v")
            state_map = self._checkpoint_map(state_checkpoints.data_ptr(), plan.total_chunks)
            dv_map = self._token_map(dv.data_ptr(), self.h_v, "dV")
            beta_map = self._beta_map(beta_active)

            diagnostic = region["dummy_f32"].data_ptr()
            params = compiler.Params()
            params.ptr(counters + 4)
            params.tensor_map(q_map).tensor_map(k_map).tensor_map(g_map).tensor_map(do_map).tensor_map(v_map).tensor_map(state_map)
            params.ptr(dfinal_ptr)
            params.tensor_map(dv_map)
            if not plan.aligned:
                params.ptr(dv.data_ptr())
            params.ptr(dq_value_ptr).ptr(dk_value_ptr)
            params.ptr(region["dlog_decay"].data_ptr()).ptr(region["dlog_boundary"].data_ptr())
            params.ptr(dinit_ptr).ptr(a_log.data_ptr()).ptr(dt_bias.data_ptr())
            params.tensor_map(beta_map)
            params.ptr(region["dbeta_active"].data_ptr())
            params.ptr(region["cu_i64"].data_ptr()).ptr(region["ckpt_starts"].data_ptr()).ptr(region["work_items"].data_ptr())
            params.ptr(region["dummy_u32"].data_ptr())
            for _ in range(14):
                params.ptr(diagnostic)
            params.i32(plan.total_work_items).i32(plan.uniform_work_items).i32(plan.total_chunks).i32(self.h_qk).i32(self.h_v).i32(1).i32(1)
            params.f32(self.scale).f32(self.lower_bound)
            name = "kernel_cake_flashkda_backward_persistent_c16_aligned" if plan.aligned else "kernel_flashkda_backward_persistent_c16"
            compiler.launch(self._kernel(name, BWD_SMEM), self._grid(plan), (BLOCK, 1, 1), BWD_SMEM, stream, params, name)

            params = compiler.Params()
            params.ptr(g.data_ptr()).ptr(beta_active).ptr(a_log.data_ptr()).ptr(dt_bias.data_ptr())
            params.ptr(region["dlog_decay"].data_ptr()).ptr(region["dlog_boundary"].data_ptr()).ptr(region["dbeta_active"].data_ptr())
            params.ptr(dg.data_ptr()).ptr(dbeta.data_ptr())
            params.ptr(region["gate_part_a"].data_ptr()).ptr(region["gate_part_dt"].data_ptr())
            params.ptr(counters + 8)
            params.ptr(d_a_log.data_ptr()).ptr(d_dt_bias.data_ptr())
            params.i32(self.total).i32(self.h_v).i32(self.beta_active_stride).i32((self.total + 127) // 128)
            params.f32(self.lower_bound)
            name = "kernel_cake_flashkda_backward_param_reduce_c16_aligned" if plan.aligned else "kernel_flashkda_backward_param_reduce_c16_partial"
            compiler.launch(self._kernel(name, REDUCE_SMEM), (128, self.h_v, 1), (128, 1, 1), REDUCE_SMEM, stream, params, name)

            if self.grouped:
                params = compiler.Params()
                params.ptr(dq_value_ptr).ptr(dk_value_ptr).ptr(dq.data_ptr()).ptr(dk.data_ptr())
                params.i32(self.total).i32(self.h_qk).i32(self.h_v)
                compiler.launch(
                    self._kernel("kernel_flashkda_grouped_qk_reduce", 0),
                    ((self.total + 15) // 16, self.h_qk, 1),
                    (128, 1, 1),
                    0,
                    stream,
                    params,
                    "grouped dQ/dK reduction",
                )


def build_kda_cake(graph):
    """The compiled plan for a single-node KDA / KDA_BWD graph."""
    nodes = list(graph.nodes)
    if len(nodes) != 1 or getattr(nodes[0].node_type, "name", None) not in ("KDA", "KDA_BWD"):
        raise ValueError("build_kda_cake: graph does not contain exactly one KDA/KDA_BWD node")
    node = nodes[0]
    if node.node_type.name == "KDA_BWD":
        return CompiledCakeKdaBwd(node)
    return CompiledCakeKda(node)


__all__ = ["C16Plan", "CompiledCakeKda", "CompiledCakeKdaBwd", "build_kda_cake", "c16_use_split", "plan_c16"]
