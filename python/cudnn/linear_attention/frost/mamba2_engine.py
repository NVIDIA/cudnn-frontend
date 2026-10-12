# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST Mamba-2 engine: MAMBA2 / MAMBA2_BWD nodes on the chunk-128 prefill and backward kernels (SM100 / SM103 / SM107,
bf16/fp16, d_qk = 128, d_v = 64).  Mamba-2 is the gated delta rule without the erase term: the gate is dt, the write
strength is dt, and the decay is exp(A dt) with ``a`` = A on the node; the kernels reuse the GDN safe-gate
transform.  Long sequences on few (sequence, head) tiles run the decay-warmup split-K; the piece chain follows."""

from __future__ import annotations

import math
from operator import itemgetter

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost.device import build_device, current_device, multiprocessor_count
from cudnn.frost.workspace import WorkspaceLayout, carve_plan
from ..graph_analyzer import analyze
from .engine import FrostLaPlan, frost_la_gate, summary_support_gates


def build_mamba2(graph):
    """Import the kernel module and wrap the single node; cute.compile runs on first execute."""
    nodes = list(graph.nodes)
    if len(nodes) != 1 or getattr(nodes[0].node_type, "name", None) not in ("MAMBA2", "MAMBA2_BWD"):
        raise ValueError("build_mamba2: graph does not contain exactly one MAMBA2/MAMBA2_BWD node")
    node = nodes[0]
    if node.node_type.name == "MAMBA2_BWD":
        from .kernel import mamba2_bprop_f16 as bwd_module
        from .kernel import mamba2_prefill_f16 as prefill_module
        from .kernel import mamba2_recompute_f16 as recompute_module

        return CompiledMamba2Bwd(node, bwd_module, prefill_module, recompute_module)
    from .kernel import mamba2_prefill_f16 as kernel_module

    return CompiledMamba2(node, kernel_module)


def mamba2_support_gates(engine: str, facts) -> None:
    """The Mamba-2 kernel family's dtype/attribute gates."""
    import cudnn

    checkpoint = facts.checkpoint_every_n_tokens
    if checkpoint and checkpoint % 128 != 0:
        raise NotImplementedError(f"{engine}: checkpoint_every_n_tokens must be a positive multiple of 128 (got {checkpoint})")
    if not facts.gates_at_ho:
        raise NotImplementedError(f"{engine}: dt must carry HO = max(q, v) heads ({facts.h_o})")
    if facts.h_v != facts.h_o:
        raise NotImplementedError(f"{engine}: x (v) must carry every head ({facts.h_o}); B / C (k / q) may be grouped")
    if facts.has_state_indices:
        if facts.state_indices_dtype != cudnn.data_type.INT32:
            raise NotImplementedError(f"{engine}: 'state_indices' must be int32, got {facts.state_indices_dtype}")
        if facts.is_bwd:
            raise NotImplementedError(f"{engine}: 'state_indices' is a forward-only pool addressing mode")
        if checkpoint:
            raise NotImplementedError(f"{engine}: 'state_indices' cannot be combined with checkpoint_every_n_tokens")
    io = (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
    state_dtypes = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16)
    param_dtypes = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
    for port, got in (("a", facts.a_dtype), ("dt_bias", facts.dt_bias_dtype), ("d", facts.d_dtype)):
        if got not in param_dtypes + (None,):
            raise NotImplementedError(f"{engine}: '{port}' must be fp32/bf16/fp16, got {got}")
    if facts.has_z and facts.z_dtype != facts.io_dtype:
        raise NotImplementedError(f"{engine}: 'z' must match the io dtype ({facts.io_dtype}), got {facts.z_dtype}")
    if facts.wants_pre_gate and facts.o_pre_dtype not in (facts.io_dtype, None):
        raise NotImplementedError(f"{engine}: 'O_pre' must match the io dtype")
    for port, got in (("initial_state", facts.state_dtype), ("final_state", facts.final_state_dtype)):
        if got not in state_dtypes + (None,):
            raise NotImplementedError(f"{engine}: '{port}' must be fp32/bf16, got {got}")
    if not facts.state_pair_match:
        raise NotImplementedError(f"{engine}: initial_state and final_state dtypes must match")
    if facts.is_bwd:
        for port, got, want in (("dA", facts.da_dtype, facts.a_dtype), ("d_dt_bias", facts.d_dt_bias_dtype, facts.dt_bias_dtype)):
            if got not in (want, None):
                raise NotImplementedError(f"{engine}: '{port}' must match its parameter dtype ({want}), got {got}")
        if facts.dd_dtype not in (facts.d_dtype, None):
            raise NotImplementedError(f"{engine}: 'dD' must match 'd' ({facts.d_dtype}), got {facts.dd_dtype}")
        if facts.dz_dtype not in (facts.z_dtype, None):
            raise NotImplementedError(f"{engine}: 'dZ' must match 'z' ({facts.z_dtype}), got {facts.dz_dtype}")
        state_grad_want = facts.state_dtype if facts.state_dtype is not None else cudnn.data_type.FLOAT
        for port, got in (("d_final_state", facts.d_final_state_dtype), ("d_initial_state", facts.d_initial_state_dtype)):
            if got not in (state_grad_want, None):
                raise NotImplementedError(f"{engine}: '{port}' must match the state dtype ({state_grad_want}), got {got}")
        if facts.wants_d_initial_state and facts.d_initial_state_dtype is None:
            raise NotImplementedError(f"{engine}: 'd_initial_state' must mirror initial_state ({state_grad_want}), got an unset dtype")
        if facts.dg_dtype not in (facts.g_dtype, None):
            raise NotImplementedError(f"{engine}: 'dG' must match 'g' ({facts.g_dtype}), got {facts.dg_dtype}")
        if facts.do_dtype not in io + (None,):
            raise NotImplementedError(f"{engine}: 'dO' must be fp16/bf16, got {facts.do_dtype}")
        if facts.io_dtype is not None and facts.state_checkpoints_dtype not in (facts.io_dtype, None):
            raise NotImplementedError(f"{engine}: 'state_checkpoints' must match the io dtype")
    elif facts.io_dtype is not None and facts.state_checkpoints_out_dtype not in (facts.io_dtype, None):
        raise NotImplementedError(f"{engine}: 'state_checkpoints' must match the io dtype")


class Mamba2FrostEngine(BaseEngine):
    """FROST chunked-kernel backend for single-node MAMBA2 graphs (THD layout) on SM100 / SM103 / SM107."""

    name = "mamba2_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        facts = graph._facts_for(analyze)
        frost_la_gate("Mamba2FrostEngine", facts, "MAMBA2")
        if facts.d_qk != 128:
            raise NotImplementedError(f"Mamba2FrostEngine: the state dimension (B / C head dim) must be 128, got {facts.d_qk}")
        if facts.d_v != 64:
            raise NotImplementedError(f"Mamba2FrostEngine: the head dim (x head dim) must be 64, got {facts.d_v}")
        mamba2_support_gates("Mamba2FrostEngine", facts)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_mamba2(graph))


def warmup_split_applies(dt_softplus: bool, dt_min: float, dt_max: float) -> bool:
    """The decay-warmup split applies to the plain softplus dt; a clamped or linear dt runs uncut."""
    return bool(dt_softplus) and float(dt_min) <= 0.0 and math.isinf(float(dt_max))


class CompiledMamba2:
    """Compiled FROST Mamba-2 forward plan over the resolved node buffers.  ``choose_pieces`` fixes the scheme at
    build: ``uncut`` (one item per sequence and head), ``warmup`` (decay-warmup split-K) or ``chain`` (per-piece
    summaries and an fp32 state chain seeding every piece)."""

    def __init__(self, node, kernel_module):
        from .common.piece_chain import CHAIN_THREADS, LENGTH_RULE_PIECE_TOKENS, choose_pieces, piece_table_layout
        from .common.split_k import WORK_ITEM_FIELDS, compute_ideal_chunks, max_work_items, split_scratch_rows
        from cuda.bindings.driver import CUstream

        from .kernel.mamba2_chain_forward_f16 import CHAIN_FORWARD_BUFFERS, build_chain_forward, run_chain_forward
        from .kernel.mamba2_warmup_forward_f16 import build_warmup_forward, run_warmup_forward

        self.node = node
        self.warmup_launch = None
        self.chain_launch = None
        self.chain_buffer_names = CHAIN_FORWARD_BUFFERS
        self.stream_type = CUstream
        self.build_warmup_forward = build_warmup_forward
        self.run_warmup_forward = run_warmup_forward
        self.build_chain_forward = build_chain_forward
        self.run_chain_forward = run_chain_forward
        self.plan_name = "Mamba2FrostEngine (MAMBA2)"
        self.device = current_device()
        self.dt_softplus = bool(node.params.get("dt_softplus", True) if node.params.get("dt_softplus") is not None else True)
        self.dt_min = float(node.params.get("dt_min", 0.0) or 0.0)
        self.dt_max = float(node.params.get("dt_max")) if node.params.get("dt_max") is not None else float("inf")

        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self.b_t = kernel_module.CFG.B_T
        total = node.inputs["k"].dim[0]
        HO = g.dim[1]
        K = q.dim[2]
        V = v.dim[2]
        self.io_name = "float16" if q.get_data_type().name == "HALF" else "bfloat16"
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.has_final_state = "final_state" in node.outputs
        self.checkpoint = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.has_state_checkpoints = "state_checkpoints" in node.outputs
        self.has_pre_gate = "O_pre" in node.outputs
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.overwrite_initial_state = bool(node.params.get("overwrite_initial_state", False))
        self.num_sm = multiprocessor_count(self.device)
        self.n_heads_out = HO
        self.dim_v, self.dim_k = V, K
        self.num_seqs = B
        self.pieces, self.unit_chunks = choose_pieces(
            num_seqs=B,
            heads_out=HO,
            num_sm=self.num_sm,
            total_tokens=total,
            b_t=self.b_t,
            cadence_tokens=self.checkpoint,
            batch_invariant=self.batch_invariant,
            expand_num=1,
        )
        self.chain = self.pieces > 0
        self.length_rule = self.chain and self.batch_invariant
        self.split = (
            not self.chain
            and not self.batch_invariant
            and not self.overwrite_initial_state
            and total >= LENGTH_RULE_PIECE_TOKENS  # the split table (plan, scan, walk) costs a short batch more than its cuts save
            and warmup_split_applies(self.dt_softplus, self.dt_min, self.dt_max)
        )

        layout = WorkspaceLayout()
        from .common.host import tensormap_workspace_bytes

        regions = []
        if self.chain:
            self.num_pieces = B * self.pieces
            self.n_tiles = self.num_pieces * HO
            self.work_item_rows = self.n_tiles
            self.ideal = None
            off_scheduler = layout.add(16)
            regions += [
                ("scheduler_prefill", off_scheduler, "int32", (2,)),
                ("scheduler_summary", off_scheduler + 8, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (4,)),
                ("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)),
            ]
            table = piece_table_layout(B, self.pieces, HO)
            off_piece_table = layout.add(table.nbytes, align=256)
            regions += [
                ("main_rows", off_piece_table + table.main_rows, "int32", (B + 1,)),
                ("summary_rows", off_piece_table + table.summary_rows, "int32", (B + 1,)),
                ("cu_pieces", off_piece_table + table.cu_pieces, "int32", (self.num_pieces + 1,)),
                ("main_count", off_piece_table + table.main_count, "int32", (1,)),
                ("summary_count", off_piece_table + table.summary_count, "int32", (1,)),
                ("work_items_summary", layout.add(table.item_rows * WORK_ITEM_FIELDS * 4), "int32", (table.item_rows, WORK_ITEM_FIELDS)),
                ("state_h", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
                ("transition", layout.add(self.num_pieces * HO * 4), "float32", (self.num_pieces, HO)),
                ("state_x", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
            ]
            # one 16-byte state vector per thread per slot: the chain prefetches every slot's vector before its walk
            self.chain_rows = 4 * CHAIN_THREADS // K
        else:
            self.num_pieces = B
            regions.append(("scheduler", layout.add(8), "int32", (2,)))
            self.n_tiles = B * HO
            if self.split:
                self.ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
                self.work_item_rows = max_work_items(total, B, HO, self.ideal, self.b_t, self.num_sm)
            else:
                self.ideal = None
                self.work_item_rows = self.n_tiles
            regions.append(("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
            regions.append(("work_count", layout.add(4), "int32", (1,)))
            if self.split:
                self.chunk_scratch_rows = split_scratch_rows(total, B, HO, self.b_t)
                regions.append(("item_scratch", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
                regions.append(("chunk_scratch", layout.add(self.chunk_scratch_rows * HO * 4), "float32", (self.chunk_scratch_rows, HO)))
        self.tensormap_words = tensormap_workspace_bytes(kernel_module, self.num_pieces) // 8
        regions.append(("tensormaps", layout.add(self.tensormap_words * 8, align=128), "int64", (self.tensormap_words,)))
        self.workspace_size = layout.size
        self.carve_names = [name for name, off, dt, shape in regions]
        self.carve = carve_plan(self.plan_name, [(off, dt, shape) for name, off, dt, shape in regions])

    def workspace_bytes(self) -> int:
        return self.workspace_size

    def bind(self, names) -> None:
        pos = {name: i for i, name in enumerate(names)}
        self.index_q = pos["q"]
        self.index_k = pos["k"]
        self.index_v = pos["v"]
        self.index_g = pos["g"]
        self.index_a = pos["a"]
        self.index_cu_seqlens = pos["cu_seqlens"]
        self.index_initial_state = pos.get("initial_state")
        self.index_dt_bias = pos.get("dt_bias")
        self.index_d = pos.get("d")
        self.index_z = pos.get("z")
        self.index_state_indices = pos.get("state_indices")
        self.index_o = pos["O"]
        self.index_final_state = pos.get("final_state")
        self.index_state_checkpoints = pos.get("state_checkpoints")
        self.index_o_pre = pos.get("O_pre")
        if self.chain:
            positions = pos | {name: len(names) + i for i, name in enumerate(self.carve_names)}
            aliases = {
                "gate": "g",
                "o": "O",
                "seed": "initial_state",
                "seed_indices": "state_indices",
                "final_indices": "state_indices",
                "checkpoints": "state_checkpoints",
                "o_pre": "O_pre",
                "words": "tensormaps",
            }
            absent = len(names) + len(self.carve_names)
            if not (self.has_state_checkpoints and self.checkpoint):
                positions.pop("state_checkpoints", None)
            indices = [positions.get(aliases.get(name, name), absent) for name in self.chain_buffer_names]
            self.chain_buffers = itemgetter(*indices)
            self.chain_scalars = (
                int(self.pieces),
                int(self.n_heads_out),
                int(self.num_seqs),
                int(self.checkpoint if self.has_state_checkpoints else 0),
                float(self.dt_min),
                float(self.dt_max),
            )

    def run(self, views, workspace, stream) -> None:
        q = views[self.index_q]
        if self.chain and self.chain_launch is not None:
            carved = workspace.carve(self.carve)
            self.chain_launch(*self.chain_scalars, *self.chain_buffers((*views, *carved, None)), self.stream_type(int(stream or 0)))
            return
        k = views[self.index_k]
        v = views[self.index_v]
        g = views[self.index_g]
        a = views[self.index_a]
        cu = views[self.index_cu_seqlens]
        state0 = views[self.index_initial_state] if self.index_initial_state is not None else None
        dt_bias = views[self.index_dt_bias] if self.index_dt_bias is not None else None
        d = views[self.index_d] if self.index_d is not None else None
        z = views[self.index_z] if self.index_z is not None else None
        state_indices = views[self.index_state_indices] if self.index_state_indices is not None else None
        o = views[self.index_o]
        final_state = views[self.index_final_state] if self.index_final_state is not None else None
        state_checkpoints = views[self.index_state_checkpoints] if self.index_state_checkpoints is not None else None
        o_pre = views[self.index_o_pre] if self.index_o_pre is not None else None
        stream = stream if stream is not None else 0
        region = dict(zip(self.carve_names, workspace.carve(self.carve)))
        if self.chain:
            self.run_chain(q, k, v, g, a, dt_bias, d, z, cu, state0, state_indices, o, final_state, state_checkpoints, o_pre, region, stream)
            return

        buffers = dict(
            q=q,
            k=k,
            v=v,
            gate=g,
            a=a,
            dt_bias=dt_bias,
            d=d,
            z=z,
            o=o,
            o_pre=o_pre,
            cu_seqlens=cu,
            state_in=state0,
            state_out=final_state,
            seed_indices=state_indices,
            final_indices=state_indices,
            checkpoints=state_checkpoints,
            work_items=region["work_items"],
            work_count=region["work_count"],
            item_scratch=region.get("item_scratch"),
            chunk_scratch=region.get("chunk_scratch"),
            scheduler=region["scheduler"],
            workspace=region["tensormaps"],
        )
        if self.warmup_launch is None:
            self.warmup_launch = self.build_warmup_forward(
                **buffers,
                split=self.split,
                n_tiles=self.n_tiles,
                ideal_chunks=self.ideal,
                num_sm=self.num_sm,
                b_t=self.b_t,
                dt_softplus=self.dt_softplus,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                checkpoint_every_n_tokens=self.checkpoint,
                device=self.device,
                stream=stream,
            )
        self.run_warmup_forward(
            *self.warmup_launch,
            **buffers,
            checkpoint_every_n_tokens=self.checkpoint,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            stream=stream,
        )

    def run_chain(self, q, k, v, g, a, dt_bias, d, z, cu, state0, state_indices, o, final_state, state_checkpoints, o_pre, region, stream) -> None:
        """Chain prologue, recompute summary, scalar state chain and the prefill over the pieces, launched from one compiled
        host (kernel/mamba2_chain_forward_f16.py)."""
        buffers = dict(
            q=q,
            k=k,
            v=v,
            gate=g,
            a=a,
            dt_bias=dt_bias,
            d=d,
            z=z,
            o=o,
            o_pre=o_pre,
            cu_seqlens=cu,
            cu_pieces=region["cu_pieces"],
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=region["main_count"],
            summary_count=region["summary_count"],
            work_items=region["work_items"],
            work_items_summary=region["work_items_summary"],
            scheduler_all=region["scheduler_all"],
            scheduler_summary=region["scheduler_summary"],
            scheduler_prefill=region["scheduler_prefill"],
            words=region["tensormaps"],
            state_h=region["state_h"],
            transition=region["transition"],
            state_x=region["state_x"],
            seed=state0,
            seed_indices=state_indices,
            final_state=final_state,
            final_indices=state_indices,
            checkpoints=state_checkpoints if self.has_state_checkpoints and self.checkpoint else None,
        )
        checkpoint = self.checkpoint if self.has_state_checkpoints else 0
        if self.chain_launch is None:
            self.chain_launch = self.build_chain_forward(
                **buffers,
                pieces=self.pieces,
                heads_out=self.n_heads_out,
                num_seqs=self.num_seqs,
                unit_chunks=self.unit_chunks,
                b_t=self.b_t,
                length_rule=self.length_rule,
                dt_softplus=self.dt_softplus,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                checkpoint_every_n_tokens=checkpoint,
                chain_rows=self.chain_rows,
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
            )
        self.run_chain_forward(
            self.chain_launch,
            **buffers,
            pieces=self.pieces,
            heads_out=self.n_heads_out,
            num_seqs=self.num_seqs,
            checkpoint_every_n_tokens=checkpoint,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            stream=stream,
        )


class CompiledMamba2Bwd:
    """Compiled FROST Mamba-2 backward plan over the resolved node buffers: the nine gradients from the forward's
    checkpoint series (regenerated by the recompute kernel when absent or coarser than a chunk), the z gate ahead of
    the bprop, the dt-gate and parameter sums behind it, the per-head dQ / dK folded to the groups."""

    def __init__(self, node, bwd_module, prefill_module, recompute_module):
        from .common.host import tensormap_workspace_bytes
        from .common.gate_bwd import mamba2_gate_bwd, mamba2_zgate_bwd
        from .common.head_reduce import head_group_reduce_dot_product
        from .common.piece_chain import CHAIN_THREADS, LENGTH_RULE_PIECE_TOKENS, choose_pieces, piece_table_layout
        from .common.split_k import WORK_ITEM_FIELDS, compute_ideal_chunks, max_work_items, split_scratch_rows
        from .kernel.mamba2_chain_backward_f16 import build_chain_backward, run_chain_backward
        from .kernel.mamba2_warmup_backward_f16 import build_regenerate_backward, build_warmup_backward, run_regenerate_backward, run_warmup_backward

        self.node = node
        self.warmup_launch = None
        self.regenerate_launch = None
        self.chain_launch = None
        self.build_warmup_backward = build_warmup_backward
        self.run_warmup_backward = run_warmup_backward
        self.build_regenerate_backward = build_regenerate_backward
        self.run_regenerate_backward = run_regenerate_backward
        self.build_chain_backward = build_chain_backward
        self.run_chain_backward = run_chain_backward
        self.head_group_reduce_dot_product = head_group_reduce_dot_product
        self.zgate_bwd = mamba2_zgate_bwd
        self.gate_bwd = mamba2_gate_bwd
        self.plan_name = "Mamba2FrostEngine (MAMBA2_BWD)"
        self.device = current_device()
        self.dt_softplus = bool(node.params.get("dt_softplus", True) if node.params.get("dt_softplus") is not None else True)
        self.dt_min = float(node.params.get("dt_min", 0.0) or 0.0)
        self.dt_max = float(node.params.get("dt_max")) if node.params.get("dt_max") is not None else float("inf")

        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self.b_t = bwd_module.CFG.B_T
        total = node.inputs["k"].dim[0]
        HQ, HK, HV = q.dim[1], node.inputs["k"].dim[1], v.dim[1]
        HO = g.dim[1]
        K = q.dim[2]
        V = v.dim[2]
        self.io_name = "float16" if q.get_data_type().name == "HALF" else "bfloat16"
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.checkpoint = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.has_state_checkpoints = "state_checkpoints" in node.inputs
        self.reuse_series = self.has_state_checkpoints and self.checkpoint == self.b_t
        self.has_initial_state = "initial_state" in node.inputs
        self.use_dstate_in = "d_final_state" in node.inputs
        self.use_dstate0 = "d_initial_state" in node.outputs
        self.has_d = "d" in node.inputs
        self.d_per_channel = self.has_d and len(node.inputs["d"].dim) == 2
        self.has_z = "z" in node.inputs
        self.wants_d_dt_bias = "d_dt_bias" in node.outputs
        self.wants_dd = "dD" in node.outputs
        self.skip_bwd_in_kernel = self.wants_dd
        self.wants_dz = "dZ" in node.outputs
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.num_sm = multiprocessor_count(self.device)
        self.n_heads_out = HO
        self.dim_v, self.dim_k = V, K
        self.num_seqs = B
        self.pieces, self.unit_chunks = choose_pieces(
            num_seqs=B,
            heads_out=HO,
            num_sm=self.num_sm,
            total_tokens=total,
            b_t=self.b_t,
            cadence_tokens=self.checkpoint if self.has_state_checkpoints else 0,
            batch_invariant=self.batch_invariant,
            expand_num=1,
            reverse=True,
        )
        self.chain = self.pieces > 0
        self.length_rule = self.chain and self.batch_invariant
        self.split = (
            not self.chain
            and not self.batch_invariant
            and total >= LENGTH_RULE_PIECE_TOKENS  # the split table (plan, scan, walk) costs a short batch more than its cuts save
            and warmup_split_applies(self.dt_softplus, self.dt_min, self.dt_max)
        )
        self.fold_dq = HQ < HO
        self.fold_dk = HK < HO

        layout = WorkspaceLayout()
        regions = []
        if self.chain:
            self.num_pieces = B * self.pieces
            off_scheduler = layout.add(32)
            regions += [
                ("scheduler_summary_fwd", off_scheduler, "int32", (2,)),
                ("scheduler_summary_bwd", off_scheduler + 8, "int32", (2,)),
                ("scheduler_bwd", off_scheduler + 16, "int32", (2,)),
                ("scheduler_regenerate", off_scheduler + 24, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (8,)),
            ]
            table = piece_table_layout(B, self.pieces, HO)
            off_piece_table = layout.add(table.nbytes, align=256)
            regions += [
                ("main_rows", off_piece_table + table.main_rows, "int32", (B + 1,)),
                ("summary_rows", off_piece_table + table.summary_rows, "int32", (B + 1,)),
                ("cu_pieces", off_piece_table + table.cu_pieces, "int32", (self.num_pieces + 1,)),
                ("main_count", off_piece_table + table.main_count, "int32", (1,)),
                ("summary_count", off_piece_table + table.summary_count, "int32", (1,)),
                ("work_items_summary", layout.add(table.item_rows * WORK_ITEM_FIELDS * 4), "int32", (table.item_rows, WORK_ITEM_FIELDS)),
                ("state_g", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
                ("state_dx", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
                ("transition", layout.add(self.num_pieces * HO * 4), "float32", (self.num_pieces, HO)),
            ]
            if not self.reuse_series:
                regions += [
                    ("state_h", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
                    ("state_x", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
                ]
            # one 16-byte state vector per thread per slot: the chain prefetches every slot's vector before its walk
            self.chain_rows = 4 * CHAIN_THREADS // K
        else:
            self.num_pieces = B
            off_scheduler = layout.add(16)
            regions += [("scheduler_recompute", off_scheduler, "int32", (2,)), ("scheduler_bwd", off_scheduler + 8, "int32", (2,))]
        self.tensormap_words = tensormap_workspace_bytes(bwd_module, self.num_pieces) // 8
        regions.append(("tensormaps", layout.add(self.tensormap_words * 8, align=128), "int64", (self.tensormap_words,)))
        if not self.reuse_series:
            # the recompute reads the prefill's K, V and checkpoint descriptor arrays
            self.recompute_tensormap_words = tensormap_workspace_bytes(prefill_module if self.chain else recompute_module, self.num_pieces) // 8
            regions.append(("recompute_tensormaps", layout.add(self.recompute_tensormap_words * 8, align=128), "int64", (self.recompute_tensormap_words,)))
            self.series_rows = max(total // self.b_t + B, 1)
            regions.append(("state_checkpoints", layout.add(self.series_rows * HO * K * V * 2), self.io_name, (self.series_rows, HO, V, K)))
        if self.has_z:
            regions.append(("do_pre", layout.add(total * HO * V * 2), self.io_name, (total, HO, V)))
        if self.fold_dq:
            regions.append(("dq_ho", layout.add(total * HO * K * 2), self.io_name, (total, HO, K)))
        if self.fold_dk:
            regions.append(("dk_ho", layout.add(total * HO * K * 2), self.io_name, (total, HO, K)))
        # exit-dot rows: one slot per chunk plus one past every sequence, under the slot formula
        self.n_slots = total // self.b_t + 2 * self.num_pieces + 1
        regions.append(("exit_dots", layout.add(self.n_slots * HO * 4), "float32", (self.n_slots, HO)))
        regions.append(("gate_slots", layout.add(self.n_slots * 8), "int32", (self.n_slots, 2)))
        if self.skip_bwd_in_kernel and self.d_per_channel:
            regions.append(("skip_bwd_part", layout.add(4 * self.n_slots * HO * V * 4), "float32", (4 * self.n_slots, HO, V)))
        elif self.skip_bwd_in_kernel:
            regions.append(("skip_bwd_part", layout.add(4 * self.n_slots * HO * 4), "float32", (4 * self.n_slots, HO)))
        regions.append(("dot_q", layout.add(total * HO * 4), "float32", (total, HO)))
        regions.append(("dot_k", layout.add(total * HO * 4), "float32", (total, HO)))
        regions.append(("gate_part_a", layout.add(self.n_slots * HO * 4), "float32", (self.n_slots * HO,)))
        if self.wants_d_dt_bias:
            regions.append(("gate_part_dt", layout.add(self.n_slots * HO * 4), "float32", (self.n_slots * HO,)))
        self.n_tiles = self.num_pieces * HO
        if self.split:
            self.ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
            self.work_item_rows = max_work_items(total, B, HO, self.ideal, self.b_t, self.num_sm)
        else:
            self.ideal = None
            self.work_item_rows = self.n_tiles
        regions.append(("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
        if not self.chain:
            regions.append(("work_count", layout.add(4), "int32", (1,)))
        if self.split:
            self.chunk_scratch_rows = split_scratch_rows(total, B, HO, self.b_t)
            regions.append(("item_scratch", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
            regions.append(("chunk_scratch", layout.add(self.chunk_scratch_rows * HO * 4), "float32", (self.chunk_scratch_rows, HO)))
        self.workspace_size = layout.size
        self.carve_names = [name for name, off, dt, shape in regions]
        self.carve = carve_plan(self.plan_name, [(off, dt, shape) for name, off, dt, shape in regions])

    def workspace_bytes(self) -> int:
        return self.workspace_size

    def bind(self, names) -> None:
        pos = {name: i for i, name in enumerate(names)}
        self.index_q = pos["q"]
        self.index_k = pos["k"]
        self.index_v = pos["v"]
        self.index_g = pos["g"]
        self.index_a = pos["a"]
        self.index_cu_seqlens = pos["cu_seqlens"]
        self.index_do = pos["dO"]
        self.index_state_checkpoints = pos.get("state_checkpoints")
        self.index_initial_state = pos.get("initial_state")
        self.index_d_final_state = pos.get("d_final_state")
        self.index_dt_bias = pos.get("dt_bias")
        self.index_d = pos.get("d")
        self.index_z = pos.get("z")
        self.index_o_pre = pos.get("O_pre")
        self.index_dq = pos["dQ"]
        self.index_dk = pos["dK"]
        self.index_dv = pos["dV"]
        self.index_dg = pos["dG"]
        self.index_da = pos["dA"]
        self.index_d_initial_state = pos.get("d_initial_state")
        self.index_d_dt_bias = pos.get("d_dt_bias")
        self.index_dd = pos.get("dD")
        self.index_dz = pos.get("dZ")

    def run(self, views, workspace, stream) -> None:
        q = views[self.index_q]
        k = views[self.index_k]
        v = views[self.index_v]
        g = views[self.index_g]
        a = views[self.index_a]
        cu = views[self.index_cu_seqlens]
        do = views[self.index_do]
        state_checkpoints = views[self.index_state_checkpoints] if self.index_state_checkpoints is not None else None
        state0 = views[self.index_initial_state] if self.index_initial_state is not None else None
        dstate_in = views[self.index_d_final_state] if self.index_d_final_state is not None else None
        dt_bias = views[self.index_dt_bias] if self.index_dt_bias is not None else None
        d = views[self.index_d] if self.index_d is not None else None
        z = views[self.index_z] if self.index_z is not None else None
        o_pre = views[self.index_o_pre] if self.index_o_pre is not None else None
        dq = views[self.index_dq]
        dk = views[self.index_dk]
        dv = views[self.index_dv]
        dg = views[self.index_dg]
        da = views[self.index_da]
        dstate0 = views[self.index_d_initial_state] if self.index_d_initial_state is not None else None
        d_dt_bias = views[self.index_d_dt_bias] if self.index_d_dt_bias is not None else None
        dd = views[self.index_dd] if self.index_dd is not None else None
        dz = views[self.index_dz] if self.index_dz is not None else None
        stream = stream if stream is not None else 0
        region = dict(zip(self.carve_names, workspace.carve(self.carve)))

        # ---- z gate: dO_pre = dO silu(z), dZ --------------------------------------------
        if self.has_z:
            self.zgate_bwd(do, z, o_pre, region["do_pre"], dz, stream=stream)
            do = region["do_pre"]

        dq_out = region["dq_ho"] if self.fold_dq else dq
        dk_out = region["dk_ho"] if self.fold_dk else dk
        skip_bwd_part = region["skip_bwd_part"] if self.skip_bwd_in_kernel else None
        if self.chain:
            self.run_chain(q, k, v, g, a, dt_bias, d, do, cu, state0, dstate_in, state_checkpoints, dq_out, dk_out, dv, dstate0, skip_bwd_part, region, stream)
        else:
            self.run_uncut(q, k, v, g, a, dt_bias, d, do, cu, state0, dstate_in, state_checkpoints, dq_out, dk_out, dv, dstate0, skip_bwd_part, region, stream)

        # ---- the dt gate, the parameters, the D skip, the group folds -------------------------
        self.head_group_reduce_dot_product(dq_out, q, dq if self.fold_dq else None, region["dot_q"], stream=stream)
        self.head_group_reduce_dot_product(dk_out, k, dk if self.fold_dk else None, region["dot_k"], stream=stream)
        self.gate_bwd(
            region["dot_q"],
            region["dot_k"],
            region["exit_dots"],
            g,
            a,
            dt_bias,
            region["gate_slots"],
            dg,
            da,
            d_dt_bias,
            region["gate_part_a"],
            region.get("gate_part_dt"),
            b_t=self.b_t,
            dt_softplus=self.dt_softplus,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            stream=stream,
            skip_bwd_part=skip_bwd_part.view(-1) if self.skip_bwd_in_kernel else None,
            dd=dd if self.skip_bwd_in_kernel else None,
        )
        return None

    def run_chain(
        self, q, k, v, g, a, dt_bias, d, do, cu, state0, dstate_in, state_checkpoints, dq_out, dk_out, dv, dstate0, skip_bwd_part, region, stream
    ) -> None:
        """Chain prologue, the forward chain's three steps when regenerating (zero-state summaries, scalar chain, the series
        from the chain states), backward summary, reverse scalar chain and the bprop over the pieces, launched from one
        compiled host (kernel/mamba2_chain_backward_f16.py)."""
        regenerate = not self.reuse_series
        buffers = dict(
            q=q,
            k=k,
            v=v,
            do=do,
            gate=g,
            a=a,
            dt_bias=dt_bias,
            d=d,
            dq=dq_out,
            dk=dk_out,
            dv=dv,
            cu_seqlens=cu,
            cu_pieces=region["cu_pieces"],
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=region["main_count"],
            summary_count=region["summary_count"],
            work_items=region["work_items"],
            work_items_summary=region["work_items_summary"],
            scheduler_all=region["scheduler_all"],
            scheduler_summary_fwd=region["scheduler_summary_fwd"],
            scheduler_summary_bwd=region["scheduler_summary_bwd"],
            scheduler_bprop=region["scheduler_bwd"],
            scheduler_regenerate=region["scheduler_regenerate"],
            prefill_words=region["recompute_tensormaps"] if regenerate else None,
            bprop_words=region["tensormaps"],
            series=region["state_checkpoints"] if regenerate else state_checkpoints,
            state_h=region["state_h"] if regenerate else None,
            transition=region["transition"],
            state_x=region["state_x"] if regenerate else None,
            state_g=region["state_g"],
            state_dx=region["state_dx"],
            seed=state0,
            dfs=dstate_in,
            dstate0=dstate0 if self.use_dstate0 else None,
            exit_dots=region["exit_dots"],
            skip_bwd_part=skip_bwd_part,
            exit_slots=region["gate_slots"],
        )
        if self.chain_launch is None:
            self.chain_launch = self.build_chain_backward(
                **buffers,
                pieces=self.pieces,
                heads_out=self.n_heads_out,
                num_seqs=self.num_seqs,
                unit_chunks=self.unit_chunks,
                b_t=self.b_t,
                length_rule=self.length_rule,
                dt_softplus=self.dt_softplus,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                chain_rows=self.chain_rows,
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
            )
        self.run_chain_backward(
            self.chain_launch,
            **buffers,
            pieces=self.pieces,
            heads_out=self.n_heads_out,
            num_seqs=self.num_seqs,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            stream=stream,
        )

    def run_uncut(
        self, q, k, v, g, a, dt_bias, d, do, cu, state0, dstate_in, state_checkpoints, dq_out, dk_out, dv, dstate0, skip_bwd_part, region, stream
    ) -> None:
        """The uncut or warmup-split bprop over the forward's series, or the regenerate path (one host: table, one
        prologue, the recompute writing the chunk-cadence series, the bprop) when the graph passed none."""
        if not self.reuse_series:
            buffers = dict(
                q=q,
                k=k,
                v=v,
                do=do,
                gate=g,
                a=a,
                dt_bias=dt_bias,
                d=d,
                dq=dq_out,
                dk=dk_out,
                dv=dv,
                series=region["state_checkpoints"],
                exit_dots=region["exit_dots"],
                skip_bwd_part=skip_bwd_part,
                cu_seqlens=cu,
                state_in=state0,
                dstate_in=dstate_in,
                dstate0=dstate0,
                work_items=region["work_items"],
                work_count=region["work_count"],
                item_scratch=region.get("item_scratch"),
                chunk_scratch=region.get("chunk_scratch"),
                scheduler_recompute=region["scheduler_recompute"],
                scheduler_bprop=region["scheduler_bwd"],
                recompute_workspace=region["recompute_tensormaps"],
                bprop_workspace=region["tensormaps"],
                exit_slots=region["gate_slots"],
            )
            if self.regenerate_launch is None:
                self.regenerate_launch = self.build_regenerate_backward(
                    **buffers,
                    split=self.split,
                    n_tiles=self.n_tiles,
                    ideal_chunks=self.ideal,
                    num_sm=self.num_sm,
                    b_t=self.b_t,
                    dt_softplus=self.dt_softplus,
                    dt_min=self.dt_min,
                    dt_max=self.dt_max,
                    device=self.device,
                    stream=stream,
                )
            self.run_regenerate_backward(*self.regenerate_launch, **buffers, dt_min=self.dt_min, dt_max=self.dt_max, stream=stream)
            return

        # ---- the bprop ------------------------------------------------------------------
        buffers = dict(
            q=q,
            k=k,
            v=v,
            do=do,
            gate=g,
            a=a,
            dt_bias=dt_bias,
            d=d,
            dq=dq_out,
            dk=dk_out,
            dv=dv,
            checkpoints=state_checkpoints,
            exit_dots=region["exit_dots"],
            skip_bwd_part=skip_bwd_part,
            cu_seqlens=cu,
            dstate_in=dstate_in,
            dstate0=dstate0,
            work_items=region["work_items"],
            work_count=region["work_count"],
            item_scratch=region.get("item_scratch"),
            chunk_scratch=region.get("chunk_scratch"),
            scheduler=region["scheduler_bwd"],
            workspace=region["tensormaps"],
            exit_slots=region["gate_slots"],
        )
        if self.warmup_launch is None:
            self.warmup_launch = self.build_warmup_backward(
                **buffers,
                split=self.split,
                n_tiles=self.n_tiles,
                ideal_chunks=self.ideal,
                num_sm=self.num_sm,
                b_t=self.b_t,
                dt_softplus=self.dt_softplus,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                device=self.device,
                stream=stream,
            )
        self.run_warmup_backward(*self.warmup_launch, **buffers, dt_min=self.dt_min, dt_max=self.dt_max, stream=stream)


def build_mamba2_summary(graph):
    """Wrap the single MAMBA2_SUMMARY / MAMBA2_SUMMARY_BWD node on the recompute kernel or the backward summary kernel."""
    nodes = list(graph.nodes)
    if len(nodes) != 1 or getattr(nodes[0].node_type, "name", None) not in ("MAMBA2_SUMMARY", "MAMBA2_SUMMARY_BWD"):
        raise ValueError("build_mamba2_summary: graph does not contain exactly one MAMBA2_SUMMARY/MAMBA2_SUMMARY_BWD node")
    node = nodes[0]
    if node.node_type.name == "MAMBA2_SUMMARY_BWD":
        from .kernel import mamba2_bprop_summary_f16 as summary_module

        return CompiledMamba2SummaryBwd(node, summary_module)
    from .kernel import mamba2_recompute_f16 as recompute_module

    return CompiledMamba2Summary(node, recompute_module)


class CompiledMamba2Summary:
    """Compiled summary plan: the recompute kernel over every sequence (one uncut item per sequence and head) writes
    ``final_state`` (``initial_state`` honored) and, when bound, ``transition``, the log2 of every sequence's decay
    product."""

    def __init__(self, node, kernel_module):
        from .common.host import tensormap_workspace_bytes
        from .common.split_k import WORK_ITEM_FIELDS
        from .kernel.mamba2_warmup_forward_f16 import build_recompute, run_recompute

        self.node = node
        self.launch = None
        self.build_recompute = build_recompute
        self.run_recompute = run_recompute
        self.plan_name = "Mamba2SummaryFrostEngine (MAMBA2_SUMMARY)"
        self.device = current_device()
        self.dt_softplus = bool(node.params.get("dt_softplus", True) if node.params.get("dt_softplus") is not None else True)
        self.dt_min = float(node.params.get("dt_min", 0.0) or 0.0)
        self.dt_max = float(node.params.get("dt_max")) if node.params.get("dt_max") is not None else float("inf")
        self.has_transition = "transition" in node.outputs
        self.b_t = kernel_module.CFG.B_T
        HO = node.inputs["g"].dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.num_sm = multiprocessor_count(self.device)
        self.n_tiles = B * HO

        layout = WorkspaceLayout()
        regions = []
        self.tensormap_words = tensormap_workspace_bytes(kernel_module, B) // 8
        regions.append(("tensormaps", layout.add(self.tensormap_words * 8, align=128), "int64", (self.tensormap_words,)))
        regions.append(("scheduler", layout.add(8), "int32", (2,)))
        regions.append(("work_items", layout.add(self.n_tiles * WORK_ITEM_FIELDS * 4), "int32", (self.n_tiles, WORK_ITEM_FIELDS)))
        regions.append(("work_count", layout.add(4), "int32", (1,)))
        self.workspace_size = layout.size
        self.carve_names = [name for name, off, dt, shape in regions]
        self.carve = carve_plan(self.plan_name, [(off, dt, shape) for name, off, dt, shape in regions])

    def workspace_bytes(self) -> int:
        return self.workspace_size

    def bind(self, names) -> None:
        pos = {name: i for i, name in enumerate(names)}
        self.index_k = pos["k"]
        self.index_v = pos["v"]
        self.index_g = pos["g"]
        self.index_a = pos["a"]
        self.index_cu_seqlens = pos["cu_seqlens"]
        self.index_initial_state = pos.get("initial_state")
        self.index_dt_bias = pos.get("dt_bias")
        self.index_final_state = pos["final_state"]
        self.index_transition = pos.get("transition")

    def run(self, views, workspace, stream) -> None:
        k = views[self.index_k]
        v = views[self.index_v]
        g = views[self.index_g]
        a = views[self.index_a]
        cu = views[self.index_cu_seqlens]
        state0 = views[self.index_initial_state] if self.index_initial_state is not None else None
        dt_bias = views[self.index_dt_bias] if self.index_dt_bias is not None else None
        final_state = views[self.index_final_state]
        transition = views[self.index_transition] if self.index_transition is not None else None
        stream = stream if stream is not None else 0
        region = dict(zip(self.carve_names, workspace.carve(self.carve)))

        buffers = dict(
            k=k,
            v=v,
            gate=g,
            a=a,
            dt_bias=dt_bias,
            cu_seqlens=cu,
            state_in=state0,
            state_out=final_state,
            seed_indices=None,
            final_indices=None,
            checkpoints=None,
            work_items=region["work_items"],
            work_count=region["work_count"],
            item_scratch=None,
            chunk_scratch=None,
            scheduler=region["scheduler"],
            workspace=region["tensormaps"],
        )
        if self.launch is None:
            self.launch = self.build_recompute(
                **buffers,
                split=False,
                n_tiles=self.n_tiles,
                ideal_chunks=None,
                num_sm=self.num_sm,
                b_t=self.b_t,
                dt_softplus=self.dt_softplus,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                checkpoint_every_n_tokens=0,
                device=self.device,
                stream=stream,
                transition=transition,
            )
        self.run_recompute(
            *self.launch,
            **buffers,
            checkpoint_every_n_tokens=0,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            stream=stream,
            transition=transition,
        )


class CompiledMamba2SummaryBwd:
    """Compiled gradient summary: the backward summary kernel over every sequence (one uncut item per sequence and
    head), the reverse state-gradient recurrence seeded by ``d_final_state`` (zero when unbound) writing
    ``d_initial_state`` and, when bound, ``transition``, the log2 of every sequence's decay product."""

    def __init__(self, node, kernel_module):
        from .common.host import tensormap_workspace_bytes
        from .common.split_k import WORK_ITEM_FIELDS
        from .kernel.mamba2_warmup_backward_f16 import build_summary_backward, run_summary_backward

        self.node = node
        self.launch = None
        self.build_summary_backward = build_summary_backward
        self.run_summary_backward = run_summary_backward
        self.plan_name = "Mamba2SummaryFrostEngine (MAMBA2_SUMMARY_BWD)"
        self.device = current_device()
        self.dt_softplus = bool(node.params.get("dt_softplus", True) if node.params.get("dt_softplus") is not None else True)
        self.dt_min = float(node.params.get("dt_min", 0.0) or 0.0)
        self.dt_max = float(node.params.get("dt_max")) if node.params.get("dt_max") is not None else float("inf")
        self.has_transition = "transition" in node.outputs
        self.b_t = kernel_module.CFG.B_T
        HO = node.inputs["g"].dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.num_sm = multiprocessor_count(self.device)
        self.n_tiles = B * HO

        layout = WorkspaceLayout()
        regions = []
        self.tensormap_words = tensormap_workspace_bytes(kernel_module, B) // 8
        regions.append(("tensormaps", layout.add(self.tensormap_words * 8, align=128), "int64", (self.tensormap_words,)))
        regions.append(("scheduler", layout.add(8), "int32", (2,)))
        regions.append(("work_items", layout.add(self.n_tiles * WORK_ITEM_FIELDS * 4), "int32", (self.n_tiles, WORK_ITEM_FIELDS)))
        regions.append(("work_count", layout.add(4), "int32", (1,)))
        self.workspace_size = layout.size
        self.carve_names = [name for name, off, dt, shape in regions]
        self.carve = carve_plan(self.plan_name, [(off, dt, shape) for name, off, dt, shape in regions])

    def workspace_bytes(self) -> int:
        return self.workspace_size

    def bind(self, names) -> None:
        pos = {name: i for i, name in enumerate(names)}
        self.index_q = pos["q"]
        self.index_g = pos["g"]
        self.index_a = pos["a"]
        self.index_cu_seqlens = pos["cu_seqlens"]
        self.index_do = pos["dO"]
        self.index_d_final_state = pos.get("d_final_state")
        self.index_dt_bias = pos.get("dt_bias")
        self.index_d_initial_state = pos["d_initial_state"]
        self.index_transition = pos.get("transition")

    def run(self, views, workspace, stream) -> None:
        q = views[self.index_q]
        g = views[self.index_g]
        a = views[self.index_a]
        cu = views[self.index_cu_seqlens]
        do = views[self.index_do]
        dstate_in = views[self.index_d_final_state] if self.index_d_final_state is not None else None
        dt_bias = views[self.index_dt_bias] if self.index_dt_bias is not None else None
        dstate0 = views[self.index_d_initial_state]
        transition = views[self.index_transition] if self.index_transition is not None else None
        stream = stream if stream is not None else 0
        region = dict(zip(self.carve_names, workspace.carve(self.carve)))

        buffers = dict(
            q=q,
            do=do,
            gate=g,
            a=a,
            dt_bias=dt_bias,
            cu_seqlens=cu,
            dstate_in=dstate_in,
            dstate0=dstate0,
            work_items=region["work_items"],
            work_count=region["work_count"],
            scheduler=region["scheduler"],
            workspace=region["tensormaps"],
        )
        if self.launch is None:
            self.launch = self.build_summary_backward(
                **buffers,
                num_sm=self.num_sm,
                b_t=self.b_t,
                dt_softplus=self.dt_softplus,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                device=self.device,
                stream=stream,
                transition=transition,
            )
        self.run_summary_backward(self.launch, **buffers, dt_min=self.dt_min, dt_max=self.dt_max, stream=stream, transition=transition)


class Mamba2SummaryFrostEngine(BaseEngine):
    """FROST summary backend for single-node MAMBA2_SUMMARY and MAMBA2_SUMMARY_BWD graphs (THD layout)."""

    name = "mamba2_summary_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        facts = graph._facts_for(analyze)
        frost_la_gate("Mamba2SummaryFrostEngine", facts, "MAMBA2_SUMMARY")
        if facts.d_qk != 128:
            raise NotImplementedError(f"Mamba2SummaryFrostEngine: the state dimension must be 128, got {facts.d_qk}")
        if facts.d_v != 64:
            raise NotImplementedError(f"Mamba2SummaryFrostEngine: the head dim must be 64, got {facts.d_v}")
        summary_support_gates("Mamba2SummaryFrostEngine", facts, graph)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_mamba2_summary(graph))
