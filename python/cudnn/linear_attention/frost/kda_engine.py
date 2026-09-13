# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST KDA engine: KDA / KDA_BWD nodes on the chunked prefill and backward kernels (SM100/SM103/SM107, bf16/fp16, BT=16);
the backward regenerates the checkpoint series with the recompute kernel when the graph carries none.  Long sequences on
few (sequence, head) tiles run as an exact piece chain (``common/piece_chain.py``) or as the decay-warmup split-K."""

from __future__ import annotations

import math

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost.device import build_device, current_device, multiprocessor_count
from cudnn.frost.workspace import WorkspaceLayout, carve_plan
from ..graph_analyzer import analyze
from .engine import FrostLaPlan, frost_la_gate, summary_support_gates


def build_kda(graph):
    """Import the kernel module (pulls in the Cutlass primitives) and wrap the
    single node; cute.compile is cached inside the kernel per static config and
    runs on first execute, when the real buffers are known."""
    nodes = list(graph.nodes)
    if len(nodes) != 1 or getattr(nodes[0].node_type, "name", None) not in ("KDA", "KDA_BWD"):
        raise ValueError("build_kda: graph does not contain exactly one KDA/KDA_BWD node")
    node = nodes[0]
    if node.node_type.name == "KDA_BWD":
        from .kernel import kda_bprop_f16 as bwd_module
        from .kernel import kda_recompute_f16 as recompute_module

        return CompiledKdaBwd(node, bwd_module, recompute_module)
    from .kernel import kda_prefill_f16 as kernel_module

    return CompiledKda(node, kernel_module)


class KdaFrostEngine(BaseEngine):
    """FROST chunked-kernel backend for single-node KDA graphs (THD layout).

    Default KDA engine on SM100/SM103/SM107 (lowest KDA engine_id); serves KDA
    forward and KDA_BWD (with a forward checkpoint recompute when ``state_checkpoints`` is absent)."""

    name = "kda_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        import cudnn

        facts = graph._facts_for(analyze)
        frost_la_gate("KdaFrostEngine", facts, "KDA")
        if facts.d_qk not in (64, 128):
            raise NotImplementedError(f"KdaFrostEngine: q/k head dim must be 64 or 128, got {facts.d_qk}")
        if facts.d_v not in (64, 128):
            raise NotImplementedError(f"KdaFrostEngine: v head dim must be 64 or 128, got {facts.d_v}")
        if facts.gate_channels != facts.d_qk:
            raise NotImplementedError(f"KdaFrostEngine: g must carry d_qk = {facts.d_qk} channels, got {facts.gate_channels}")
        checkpoint = facts.checkpoint_every_n_tokens
        if checkpoint and checkpoint % 16 != 0:
            raise NotImplementedError(f"KdaFrostEngine: checkpoint_every_n_tokens must be a positive multiple of 16 (got {checkpoint})")
        if not facts.gates_at_ho:
            raise NotImplementedError(f"KdaFrostEngine: g/beta must carry HO = max(q, v) heads ({facts.h_o})")
        beta_wants = (cudnn.data_type.FLOAT, facts.io_dtype)
        if facts.beta_dtype not in beta_wants + (None,):
            raise NotImplementedError(f"KdaFrostEngine: 'beta' must be {' or '.join(str(w) for w in beta_wants)}, got {facts.beta_dtype}")
        state_dtypes = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16)
        for port, got in (("initial_state", facts.state_dtype), ("final_state", facts.final_state_dtype)):
            if got not in state_dtypes + (None,):
                raise NotImplementedError(f"KdaFrostEngine: '{port}' must be fp32/bf16, got {got}")
        if not facts.state_pair_match:
            raise NotImplementedError("KdaFrostEngine: initial_state and final_state dtypes must match")
        gate_param_dtypes = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
        for port, got in (("a_log", facts.a_log_dtype), ("dt_bias", facts.dt_bias_dtype)):
            if got not in gate_param_dtypes + (None,):
                raise NotImplementedError(f"KdaFrostEngine: '{port}' must be fp32/bf16/fp16, got {got}")
        if facts.is_bwd:
            for port, got, want in (("d_a_log", facts.d_a_log_dtype, facts.a_log_dtype), ("d_dt_bias", facts.d_dt_bias_dtype, facts.dt_bias_dtype)):
                if got is not None and want is None:
                    raise NotImplementedError(f"KdaFrostEngine: '{port}' requires its parameter input")
                if got not in (want, None):
                    raise NotImplementedError(f"KdaFrostEngine: '{port}' must match its parameter dtype ({want}), got {got}")
            state_grad_want = facts.state_dtype if facts.state_dtype is not None else cudnn.data_type.FLOAT
            for port, got in (("d_final_state", facts.d_final_state_dtype), ("d_initial_state", facts.d_initial_state_dtype)):
                if got not in (state_grad_want, None):
                    raise NotImplementedError(f"KdaFrostEngine: '{port}' must match the state dtype ({state_grad_want}), got {got}")
            if facts.wants_d_initial_state and facts.d_initial_state_dtype is None:
                raise NotImplementedError(f"KdaFrostEngine: 'd_initial_state' must mirror initial_state ({state_grad_want}), got an unset dtype")
            if facts.dg_dtype not in (facts.g_dtype, None):
                raise NotImplementedError(f"KdaFrostEngine: 'dG' must match 'g' ({facts.g_dtype}), got {facts.dg_dtype}")
            for port, got in (("dO", facts.do_dtype), ("state_checkpoints", facts.state_checkpoints_dtype)):
                if got not in (facts.io_dtype, None):
                    raise NotImplementedError(f"KdaFrostEngine: '{port}' must match the io dtype")
            if facts.dbeta_dtype not in (facts.beta_dtype, None):
                raise NotImplementedError(f"KdaFrostEngine: 'dBeta' must match 'beta' ({facts.beta_dtype}), got {facts.dbeta_dtype}")
        elif facts.io_dtype is not None and facts.state_checkpoints_out_dtype not in (facts.io_dtype, None):
            raise NotImplementedError("KdaFrostEngine: 'state_checkpoints' must match the io dtype")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_kda(graph))


class CompiledKda:
    """Compiled FROST KDA plan over the resolved node buffers.  ``choose_pieces`` fixes the scheme at build: ``uncut``
    (one item per sequence and head), ``warmup`` (decay-warmup split-K) or ``chain`` (per-piece H and M from the fused
    summary, an fp32 state chain seeding every piece, the prefill over the pieces storing every piece's final state, the
    last filled piece gathered into ``final_state``).  A rectangular state chains like a square one: the transition
    summary runs k in place of v, so M is (DK, DK) while H and X are (DV, DK)."""

    def __init__(self, node, kernel_module):
        from .common.host import tensormap_workspace_bytes
        from .common.piece_chain import build_state_chain, chain_rows_per_cta, choose_pieces, piece_table_layout, run_state_chain
        from .kernel.kda_chain_prologue_f16 import run_chain_prologue
        from .common.split_k import WORK_ITEM_FIELDS, build_split_table, chunk_scratch_rows, compute_ideal_chunks, max_work_items, run_table

        self.node = node
        self.kernel = kernel_module
        self.build_split_table = build_split_table
        self.run_table = run_table
        self.build_state_chain = build_state_chain
        self.chain_rows_per_cta = chain_rows_per_cta
        self.run_state_chain = run_state_chain
        self.table = None
        self.kernel_cache = None
        self.run_chain_prologue = run_chain_prologue
        self.chain_prologue = {}
        self.chain_forward = None
        self.plan_name = "KdaFrostEngine (KDA)"
        self.device = current_device()
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))
        self.allow_neg_eigval = bool(node.params.get("allow_neg_eigval", False))
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.log_gate = (node.params.get("gate_domain") or "log") == "log"
        gate_lower_bound = node.params.get("gate_lower_bound")
        self.gate_lower_bound = float(gate_lower_bound) if gate_lower_bound is not None else kernel_module.DEFAULT_GATE_LOWER_BOUND
        self.has_final_state = "final_state" in node.outputs
        self.has_state_checkpoints = "state_checkpoints" in node.outputs
        self.checkpoint = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"

        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self.b_t = kernel_module.CFG.B_T
        total = q.dim[0]
        HO = g.dim[1]
        K, V = q.dim[2], v.dim[2]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.num_sm = multiprocessor_count(self.device)
        self.n_heads_out = HO
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
        )
        self.chain = self.pieces > 0
        self.split = not self.chain and not self.batch_invariant
        self.length_rule = self.chain and self.batch_invariant

        layout = WorkspaceLayout()
        regions = []
        if self.chain:
            self.num_pieces = B * self.pieces
            self.n_tiles = self.num_pieces * HO
            self.work_item_rows = self.n_tiles
            self.ideal = None
            off_scheduler = layout.add(24)
            regions += [
                ("scheduler_prefill", off_scheduler, "int32", (2,)),
                ("scheduler_h", off_scheduler + 8, "int32", (2,)),
                ("scheduler_m", off_scheduler + 16, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (6,)),
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
                ("state_m", layout.add(self.num_pieces * HO * K * K * 4), "float32", (self.num_pieces, HO, K, K)),
                ("state_x", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
            ]
            from .kernel import kda_summary_f16 as summary_module

            self.fused_summary = summary_module
            self.fused_cache = None
            self.fused_tensormap_bytes = tensormap_workspace_bytes(summary_module, self.num_pieces)
            regions.append(("fused_tensormaps", layout.add(self.fused_tensormap_bytes, align=128), "int64", (self.fused_tensormap_bytes // 8,)))
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
                self.chunk_scratch_rows = chunk_scratch_rows(total, B, self.b_t)
                regions.append(("item_scratch", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
                regions.append(("chunk_scratch", layout.add(self.chunk_scratch_rows * HO * 4), "float32", (self.chunk_scratch_rows, HO)))
        self.tensormap_bytes = tensormap_workspace_bytes(kernel_module, self.num_pieces)
        regions.append(("tensormaps", layout.add(self.tensormap_bytes, align=128), "int64", (self.tensormap_bytes // 8,)))
        self.needs_table = self.split
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
        self.index_beta = pos["beta"]
        self.index_cu_seqlens = pos["cu_seqlens"]
        self.index_initial_state = pos.get("initial_state")
        self.index_o = pos["O"]
        self.index_final_state = pos.get("final_state")
        self.index_state_checkpoints = pos.get("state_checkpoints")
        self.index_a_log = pos.get("a_log")
        self.index_dt_bias = pos.get("dt_bias")

    def run(self, views, workspace, stream) -> None:
        q = views[self.index_q]
        k = views[self.index_k]
        v = views[self.index_v]
        g = views[self.index_g]
        beta = views[self.index_beta]
        cu = views[self.index_cu_seqlens]
        state0 = views[self.index_initial_state] if self.index_initial_state is not None else None
        o = views[self.index_o]
        final_state = views[self.index_final_state] if self.index_final_state is not None else None
        state_checkpoints = views[self.index_state_checkpoints] if self.index_state_checkpoints is not None else None
        a_log = views[self.index_a_log] if self.index_a_log is not None else None
        dt_bias = views[self.index_dt_bias] if self.index_dt_bias is not None else None
        stream = stream if stream is not None else 0

        region = dict(zip(self.carve_names, workspace.carve(self.carve)))
        if self.chain:
            self.run_chain(q, k, v, g, beta, cu, state0, o, final_state, state_checkpoints, a_log, dt_bias, region, stream)
            return
        scheduler_counter = region["scheduler"]
        work_items = region["work_items"]
        work_count = region["work_count"]
        item_scratch = region.get("item_scratch")
        chunk_scratch = region.get("chunk_scratch")
        tensormaps = region["tensormaps"]

        if self.kernel_cache is not None and (self.table is not None or not self.needs_table):
            if self.needs_table:
                self.run_table(self.table, g, a_log, dt_bias, cu, chunk_scratch, item_scratch, work_items, work_count, scheduler_counter, stream)
            self.kernel.run_prefill(
                self.kernel_cache,
                q,
                k,
                v,
                g,
                a_log if self.safe_gate else None,
                dt_bias if self.safe_gate else None,
                beta,
                cu,
                state0,
                o,
                final_state,
                state_checkpoints,
                work_items,
                work_count,
                scheduler_counter,
                item_scratch,
                tensormaps,
                self.checkpoint if self.has_state_checkpoints else 0,
                self.scale,
                stream,
            )
            return

        if not self.needs_table:
            self.table = None
        else:
            self.table = self.build_split_table(
                g,
                cu,
                work_items,
                work_count,
                ideal_chunks=self.ideal,
                n_tiles=self.n_tiles,
                num_sms=self.num_sm,
                b_t=self.b_t,
                chunk_scratch=chunk_scratch,
                item_scratch=item_scratch,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                gate_lower_bound=self.gate_lower_bound if self.safe_gate else None,
                scheduler_counter=scheduler_counter,
                split=self.split,
                stream=stream,
            )

        checkpoint_kwargs = {}
        if self.has_state_checkpoints:
            checkpoint_kwargs = dict(checkpoint_every_n_tokens=self.checkpoint, output_state_checkpoints=state_checkpoints)
        self.kernel_cache = self.kernel.chunk_kda_sm100(
            q,
            k,
            v,
            g,
            beta,
            o,
            cu,
            state0,
            final_state,
            self.scale,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            use_beta_sigmoid_in_kernel=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            safe_gate=self.safe_gate,
            gate_lower_bound=self.gate_lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
            work_items=work_items,
            work_count=work_count,
            scheduler_counter=scheduler_counter,
            work_item_scratch=item_scratch,
            tensormap_workspace=tensormaps,
            **checkpoint_kwargs,
            device=self.device,
            num_sm=self.num_sm,
            stream=stream,
            log_gate=self.log_gate,
        )
        return None

    def run_chain(self, q, k, v, g, beta, cu, state0, o, final_state, state_checkpoints, a_log, dt_bias, region, stream) -> None:
        """Chain prologue, fused H and M summaries, fp32 state chain seeded with initial_state, prefill over the pieces seeded
        with X; the checkpoint series keeps its unsplit layout."""
        cu_pieces = region["cu_pieces"]
        work_items, work_count = region["work_items"], region["main_count"]
        summary_items, summary_count = region["work_items_summary"], region["summary_count"]
        state_h, state_m, state_x = region["state_h"], region["state_m"], region["state_x"]
        gate_a_log = a_log if self.safe_gate else None
        gate_dt_bias = dt_bias if self.safe_gate else None
        checkpoint = self.checkpoint if self.has_state_checkpoints else 0
        checkpoints = state_checkpoints if checkpoint else None
        HO, V, K = state_h.shape[1], state_h.shape[2], state_h.shape[3]
        warm = self.kernel_cache is not None
        self.run_chain_prologue(
            self.chain_prologue,
            pieces=self.pieces,
            unit_chunks=self.unit_chunks,
            b_t=self.b_t,
            length_rule=self.length_rule,
            heads_out=self.n_heads_out,
            checkpoint_every_n_tokens=checkpoint,
            cu_seqlens=cu,
            cu_pieces=cu_pieces,
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=work_count,
            summary_count=summary_count,
            work_items=work_items,
            work_items_summary=summary_items,
            scheduler=region["scheduler_all"],
            summary_words=region.get("fused_tensormaps"),
            prefill_words=region["tensormaps"],
            q=q,
            k=k,
            v=v,
            gate=g,
            o=o,
            checkpoints=checkpoints,
            stream=stream,
        )
        if warm:
            self.fused_summary.run_summary(
                self.fused_cache,
                k,
                v,
                g,
                gate_a_log,
                gate_dt_bias,
                beta,
                cu_pieces,
                None,
                state_h,
                state_m,
                summary_items,
                summary_count,
                region["scheduler_h"],
                region["scheduler_all"],
                None,
                region["fused_tensormaps"],
                stream,
                own_prologue=False,
            )
            self.run_state_chain(self.chain_forward, self.num_seqs, state_h, state_m, state_x, state0, None, None, stream, main_rows=region["main_rows"])
            self.kernel.run_prefill(
                self.kernel_cache,
                q,
                k,
                v,
                g,
                gate_a_log,
                gate_dt_bias,
                beta,
                cu_pieces,
                state_x,
                o,
                final_state,
                state_checkpoints,
                work_items,
                work_count,
                region["scheduler_prefill"],
                None,
                region["tensormaps"],
                checkpoint,
                self.scale,
                stream,
                own_prologue=False,
            )
            return

        summary_common = dict(
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            safe_gate=self.safe_gate,
            gate_lower_bound=self.gate_lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            work_items=summary_items,
            work_count=summary_count,
            scheduler_all=region["scheduler_all"],
            device=self.device,
            num_sm=self.num_sm,
            stream=stream,
            own_prologue=False,
        )
        self.fused_cache = self.fused_summary.chunk_kda_summary_sm100(
            k,
            v,
            g,
            beta,
            cu_pieces,
            None,
            state_h,
            state_m,
            scheduler_counter=region["scheduler_h"],
            tensormap_workspace=region["fused_tensormaps"],
            **summary_common,
            log_gate=self.log_gate,
        )
        self.chain_forward = self.build_state_chain(
            heads_out=HO,
            dim_v=V,
            dim_k=K,
            pieces=self.pieces,
            rows_per_cta=self.chain_rows_per_cta(V, K, self.num_seqs, HO, self.num_sm),
            transpose=False,
            has_seed=state0 is not None,
            has_tail=False,
            emit_summary=False,
            filled_only=True,
            seed_dtype=str(state0.dtype) if state0 is not None else "float32",
            device=self.device,
        )
        self.run_state_chain(self.chain_forward, self.num_seqs, state_h, state_m, state_x, state0, None, None, stream, main_rows=region["main_rows"])
        self.kernel_cache = self.kernel.chunk_kda_sm100(
            q,
            k,
            v,
            g,
            beta,
            o,
            cu_pieces,
            state_x,
            final_state,
            self.scale,
            checkpoint_every_n_tokens=checkpoint,
            output_state_checkpoints=checkpoints,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            use_beta_sigmoid_in_kernel=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            safe_gate=self.safe_gate,
            gate_lower_bound=self.gate_lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
            work_items=work_items,
            work_count=work_count,
            scheduler_counter=region["scheduler_prefill"],
            tensormap_workspace=region["tensormaps"],
            device=self.device,
            num_sm=self.num_sm,
            stream=stream,
            own_prologue=False,
            log_gate=self.log_gate,
        )


class CompiledKdaBwd:
    """Compiled FROST KDA backward plan: one recompute pass regenerates the checkpoint series into workspace when the
    graph carries none; GVA/GQA gradients land in HO-head scratch and reduce to the native head counts.  ``chain`` runs
    the pieces as independent sequences: H, M, X from the fused summary and the forward chain (M alone when the series
    is passed back), G from the bprop summary, a reverse fp32 chain seeding every piece's outgoing gradient, piece 0 of
    every sequence gathered into ``d_initial_state``."""

    def __init__(self, node, bwd_module, recompute_module):
        from .common.gate_bwd import GATE_BWD_BLOCKS, channel_gate_bwd
        from .common.head_reduce import head_group_reduce
        from .common.host import tensormap_workspace_bytes
        from .common.piece_chain import build_state_chain, chain_rows_per_cta, choose_pieces, piece_table_layout, run_state_chain
        from .kernel.kda_chain_prologue_f16 import run_chain_prologue
        from .common.split_k import WORK_ITEM_FIELDS, build_split_table, chunk_scratch_rows, compute_ideal_chunks, max_work_items, run_table

        self.node = node
        self.bwd = bwd_module
        self.recompute = recompute_module
        self.build_split_table = build_split_table
        self.run_table = run_table
        self.build_state_chain = build_state_chain
        self.chain_rows_per_cta = chain_rows_per_cta
        self.run_state_chain = run_state_chain
        self.table = None
        self.kernel_cache = None
        self.recompute_cache = None
        self.run_chain_prologue = run_chain_prologue
        self.chain_prologue = {}
        self.state_m_cache = None
        self.state_g_cache = None
        self.chain_forward = None
        self.chain_reverse = None
        self.plan_name = "KdaFrostEngine (KDA_BWD)"
        self.device = current_device()
        self.head_group_reduce = head_group_reduce
        self.channel_gate_bwd = channel_gate_bwd
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.log_gate = (node.params.get("gate_domain") or "log") == "log"
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))
        self.allow_neg_eigval = bool(node.params.get("allow_neg_eigval", False))
        gate_lower_bound = node.params.get("gate_lower_bound")
        self.gate_lower_bound = float(gate_lower_bound) if gate_lower_bound is not None else bwd_module.DEFAULT_GATE_LOWER_BOUND
        self.gate_bwd_blocks = GATE_BWD_BLOCKS
        self.has_state_checkpoints = "state_checkpoints" in node.inputs
        self.has_dstate0 = "d_initial_state" in node.outputs
        self.has_a_log = "a_log" in node.inputs
        self.has_dt_bias = "dt_bias" in node.inputs
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"

        q, g, v = node.inputs["q"], node.inputs["g"], node.inputs["v"]
        self.b_t = bwd_module.CFG.B_T
        self.checkpoint_cadence = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.coarse_checkpoints = self.has_state_checkpoints and self.checkpoint_cadence > self.b_t
        self.needs_recompute = not self.has_state_checkpoints or self.coarse_checkpoints
        total = q.dim[0]
        HQ, HV = q.dim[1], v.dim[1]
        HK = node.inputs["k"].dim[1]
        HO = g.dim[1]
        K, V = q.dim[-1], v.dim[-1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.io_name = "float16" if node.inputs["q"].get_data_type().name == "HALF" else "bfloat16"
        self.n_heads_out, self.total = HO, total
        self.num_sm = multiprocessor_count(self.device)
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.num_seqs = B
        self.pieces, self.unit_chunks = choose_pieces(
            num_seqs=B,
            heads_out=HO,
            num_sm=self.num_sm,
            total_tokens=total,
            b_t=self.b_t,
            cadence_tokens=self.checkpoint_cadence,
            batch_invariant=self.batch_invariant,
            reverse=True,
            expand_num=1,
        )
        self.chain = self.pieces > 0
        self.split = not self.chain and not self.batch_invariant
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        self.fused_h_m = self.chain and not self.has_state_checkpoints

        layout = WorkspaceLayout()
        regions = []
        off_scheduler = layout.add(40 if self.chain else 16)
        regions += [
            ("scheduler_recompute", off_scheduler, "int32", (2,)),
            ("scheduler_bwd", off_scheduler + 8, "int32", (2,)),
        ]
        if self.fused_h_m:
            from .kernel import kda_summary_f16 as fused_module

            self.fused_summary = fused_module
            self.fused_cache = None
            self.fused_tensormap_bytes = tensormap_workspace_bytes(fused_module, self.num_pieces)
            regions.append(("fused_tensormaps", layout.add(self.fused_tensormap_bytes, align=128), "int64", (self.fused_tensormap_bytes // 8,)))
        if self.chain:
            from .kernel import kda_bprop_summary_f16 as summary_module

            self.summary = summary_module
            self.summary_tensormap_bytes = tensormap_workspace_bytes(summary_module, self.num_pieces)
            regions += [
                ("scheduler_summary", off_scheduler + 16, "int32", (2,)),
                ("scheduler_m", off_scheduler + 24, "int32", (2,)),
                ("scheduler_series", off_scheduler + 32, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (10,)),
                ("summary_tensormaps", layout.add(self.summary_tensormap_bytes, align=128), "int64", (self.summary_tensormap_bytes // 8,)),
            ]
        else:
            regions.append(("scheduler_all", off_scheduler, "int32", (4,)))
        self.n_tiles = self.num_pieces * HO
        if self.split:
            self.ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
            self.work_item_rows = max_work_items(total, B, HO, self.ideal, self.b_t, self.num_sm)
        else:
            self.ideal = None
            self.work_item_rows = self.n_tiles
        regions.append(("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
        regions.append(("work_count", layout.add(4), "int32", (1,)))
        if self.split:
            self.chunk_scratch_rows = chunk_scratch_rows(total, B, self.b_t)
            regions.append(("item_scratch", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
            regions.append(("chunk_scratch", layout.add(self.chunk_scratch_rows * HO * 4), "float32", (self.chunk_scratch_rows, HO)))
        if self.needs_recompute:
            self.state_checkpoints_rows = max(total // self.b_t + self.num_pieces, 1)
            regions.append(
                ("state_checkpoints", layout.add(self.state_checkpoints_rows * HO * K * V * 2), self.io_name, (self.state_checkpoints_rows, HO, V, K))
            )
        self.recompute_tensormap_bytes = tensormap_workspace_bytes(recompute_module, self.num_pieces)
        if self.needs_recompute and not self.chain:
            regions.append(("recompute_tensormaps", layout.add(self.recompute_tensormap_bytes, align=128), "int64", (self.recompute_tensormap_bytes // 8,)))
        if self.chain:
            regions.append(("recompute_tensormaps_m", layout.add(self.recompute_tensormap_bytes, align=128), "int64", (self.recompute_tensormap_bytes // 8,)))
            if self.needs_recompute:
                regions.append(
                    ("recompute_tensormaps_series", layout.add(self.recompute_tensormap_bytes, align=128), "int64", (self.recompute_tensormap_bytes // 8,))
                )
        if self.coarse_checkpoints:
            interval_chunks = self.checkpoint_cadence // self.b_t
            span_chunks = interval_chunks
            if not self.batch_invariant:
                ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
                span_chunks = interval_chunks * max(1, ideal // interval_chunks)
            self.recompute_span_tokens = span_chunks * self.b_t
            self.recompute_item_rows = max((total // (self.b_t * span_chunks) + 2 * self.num_pieces) * HO, 1)
            regions.append(
                ("work_items_recompute", layout.add(self.recompute_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.recompute_item_rows, WORK_ITEM_FIELDS))
            )
            regions.append(("work_count_recompute", layout.add(4), "int32", (1,)))
        self.fold_dq = HQ < HO
        self.fold_dk = HK < HO
        self.fold_dv = HV < HO
        if self.fold_dq:
            regions.append(("dq_ho", layout.add(total * HO * K * 2), self.io_name, (total, HO, K)))
        if self.fold_dk:
            regions.append(("dk_ho", layout.add(total * HO * K * 2), self.io_name, (total, HO, K)))
        if self.fold_dv:
            regions.append(("dv_ho", layout.add(total * HO * V * 2), self.io_name, (total, HO, V)))
        if self.safe_gate and self.has_a_log:
            regions.append(("gate_part_a", layout.add(self.gate_bwd_blocks * HO * K * 4), "float32", (self.gate_bwd_blocks * HO * K,)))
        if self.safe_gate and self.has_dt_bias:
            regions.append(("gate_part_dt", layout.add(self.gate_bwd_blocks * HO * K * 4), "float32", (self.gate_bwd_blocks * HO * K,)))
        if self.chain:
            table = piece_table_layout(B, self.pieces, HO)
            off_piece_table = layout.add(table.nbytes, align=256)
            state_shape = (self.num_pieces, HO, V, K)
            regions += [
                ("main_rows", off_piece_table + table.main_rows, "int32", (B + 1,)),
                ("summary_rows", off_piece_table + table.summary_rows, "int32", (B + 1,)),
                ("cu_pieces", off_piece_table + table.cu_pieces, "int32", (self.num_pieces + 1,)),
                ("main_count", off_piece_table + table.main_count, "int32", (1,)),
                ("summary_count", off_piece_table + table.summary_count, "int32", (1,)),
                ("work_items_summary", layout.add(table.item_rows * WORK_ITEM_FIELDS * 4), "int32", (table.item_rows, WORK_ITEM_FIELDS)),
                ("state_m", layout.add(self.num_pieces * HO * K * K * 4), "float32", (self.num_pieces, HO, K, K)),
                ("state_g", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape),
                ("state_dx_end", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape),
            ]
            if not self.has_state_checkpoints:
                regions.append(("state_h", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape))
                regions.append(("state_x", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape))
        self.bwd_tensormap_bytes = tensormap_workspace_bytes(bwd_module, self.num_pieces)
        regions.append(("bwd_tensormaps", layout.add(self.bwd_tensormap_bytes, align=128), "int64", (self.bwd_tensormap_bytes // 8,)))
        self.needs_table = self.split
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
        self.index_beta = pos["beta"]
        self.index_cu_seqlens = pos["cu_seqlens"]
        self.index_do = pos["dO"]
        self.index_state_checkpoints = pos.get("state_checkpoints")
        self.index_initial_state = pos.get("initial_state")
        self.index_d_final_state = pos.get("d_final_state")
        self.index_dq = pos["dQ"]
        self.index_dk = pos["dK"]
        self.index_dv = pos["dV"]
        self.index_dg = pos["dG"]
        self.index_dbeta = pos["dBeta"]
        self.index_d_initial_state = pos.get("d_initial_state")
        self.index_a_log = pos.get("a_log")
        self.index_dt_bias = pos.get("dt_bias")
        self.index_d_a_log = pos.get("d_a_log")
        self.index_d_dt_bias = pos.get("d_dt_bias")

    def run(self, views, workspace, stream) -> None:
        q = views[self.index_q]
        k = views[self.index_k]
        v = views[self.index_v]
        g = views[self.index_g]
        beta = views[self.index_beta]
        cu = views[self.index_cu_seqlens]
        do = views[self.index_do]
        state_checkpoints = views[self.index_state_checkpoints] if self.index_state_checkpoints is not None else None
        state0 = views[self.index_initial_state] if self.index_initial_state is not None else None
        dstate_in = views[self.index_d_final_state] if self.index_d_final_state is not None else None
        dq = views[self.index_dq]
        dk = views[self.index_dk]
        dv = views[self.index_dv]
        dg = views[self.index_dg]
        dbeta = views[self.index_dbeta]
        dstate0 = views[self.index_d_initial_state] if self.index_d_initial_state is not None else None
        a_log = views[self.index_a_log] if self.index_a_log is not None else None
        dt_bias = views[self.index_dt_bias] if self.index_dt_bias is not None else None
        d_a_log = views[self.index_d_a_log] if self.index_d_a_log is not None else None
        d_dt_bias = views[self.index_d_dt_bias] if self.index_d_dt_bias is not None else None
        stream = stream if stream is not None else 0

        region = dict(zip(self.carve_names, workspace.carve(self.carve)))
        scheduler_recompute = region["scheduler_recompute"]
        scheduler_bwd = region["scheduler_bwd"]
        work_items = region["work_items"]
        work_count = region["work_count"]
        dq_out = region["dq_ho"] if self.fold_dq else dq
        dk_out = region["dk_ho"] if self.fold_dk else dk
        dv_out = region["dv_ho"] if self.fold_dv else dv

        if self.chain:
            self.run_chain(
                q, k, v, g, beta, do, cu, state_checkpoints, state0, dstate_in, dq_out, dk_out, dv_out, dg, dbeta, dstate0, a_log, dt_bias, region, stream
            )
        elif self.kernel_cache is not None and (self.table is not None or not self.needs_table):
            if self.needs_table:
                self.run_table(
                    self.table,
                    g,
                    a_log,
                    dt_bias,
                    cu,
                    region.get("chunk_scratch"),
                    region.get("item_scratch"),
                    work_items,
                    work_count,
                    region["scheduler_all"],
                    stream,
                )
            if self.has_state_checkpoints and not self.coarse_checkpoints:
                checkpoint_series = state_checkpoints
            else:
                checkpoint_series = region["state_checkpoints"]
                self.recompute.run_recompute(
                    self.recompute_cache,
                    k,
                    v,
                    g,
                    a_log if self.safe_gate else None,
                    dt_bias if self.safe_gate else None,
                    beta,
                    cu,
                    None if self.coarse_checkpoints else state0,
                    None,
                    checkpoint_series,
                    region["work_items_recompute"] if self.coarse_checkpoints else work_items,
                    region["work_count_recompute"] if self.coarse_checkpoints else work_count,
                    scheduler_recompute,
                    region["scheduler_all"],
                    None if self.coarse_checkpoints else region.get("item_scratch"),
                    region["recompute_tensormaps"],
                    self.b_t,
                    stream,
                    seed_state_checkpoints=state_checkpoints if self.coarse_checkpoints else None,
                    seed_every_n_tokens=self.checkpoint_cadence if self.coarse_checkpoints else 0,
                    seed_span_tokens=self.recompute_span_tokens if self.coarse_checkpoints else 0,
                )
            self.bwd.run_bwd(
                self.kernel_cache,
                q,
                k,
                v,
                g,
                beta,
                do,
                checkpoint_series,
                dq_out,
                dk_out,
                dv_out,
                dg,
                dbeta,
                cu,
                dstate0 if self.has_dstate0 else None,
                dstate_in,
                work_items,
                work_count,
                scheduler_bwd,
                region["scheduler_all"] if self.has_state_checkpoints else None,
                region.get("item_scratch") if self.has_state_checkpoints else None,
                region["bwd_tensormaps"],
                self.scale,
                stream,
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
            )
        else:
            if not self.needs_table:
                self.table = None
            else:
                self.table = self.build_split_table(
                    g,
                    cu,
                    work_items,
                    work_count,
                    ideal_chunks=self.ideal,
                    n_tiles=self.n_tiles,
                    num_sms=self.num_sm,
                    b_t=self.b_t,
                    chunk_scratch=region.get("chunk_scratch"),
                    item_scratch=region.get("item_scratch"),
                    log_gate=self.log_gate,
                    safe_gate=self.safe_gate,
                    a_log=a_log,
                    dt_bias=dt_bias,
                    gate_lower_bound=self.gate_lower_bound if self.safe_gate else None,
                    scheduler_counter=region["scheduler_all"],
                    split=self.split,
                    stream=stream,
                )

            if self.has_state_checkpoints and not self.coarse_checkpoints:
                checkpoint_series = state_checkpoints
            else:
                checkpoint_series = region["state_checkpoints"]
                self.recompute_cache = self.recompute.chunk_kda_recompute_sm100(
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    None if self.coarse_checkpoints else state0,
                    None,
                    checkpoint_every_n_tokens=self.b_t,
                    output_state_checkpoints=checkpoint_series,
                    seed_state_checkpoints=state_checkpoints if self.coarse_checkpoints else None,
                    seed_every_n_tokens=self.checkpoint_cadence if self.coarse_checkpoints else 0,
                    seed_span_tokens=self.recompute_span_tokens if self.coarse_checkpoints else 0,
                    use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                    safe_gate=self.safe_gate,
                    gate_lower_bound=self.gate_lower_bound,
                    a_log=a_log,
                    dt_bias=dt_bias,
                    use_beta_sigmoid=self.use_beta_sigmoid,
                    allow_neg_eigval=self.allow_neg_eigval,
                    work_items=region["work_items_recompute"] if self.coarse_checkpoints else work_items,
                    work_count=region["work_count_recompute"] if self.coarse_checkpoints else work_count,
                    scheduler_counter=scheduler_recompute,
                    scheduler_all=region["scheduler_all"],
                    work_item_scratch=None if self.coarse_checkpoints else region.get("item_scratch"),
                    order_in_prologue=not self.coarse_checkpoints,
                    tensormap_workspace=region["recompute_tensormaps"],
                    device=self.device,
                    num_sm=self.num_sm,
                    stream=stream,
                    log_gate=self.log_gate,
                )

            self.kernel_cache = self.bwd.chunk_kda_bwd_sm100(
                q,
                k,
                v,
                g,
                beta,
                do,
                checkpoint_series,
                dq_out,
                dk_out,
                dv_out,
                dg,
                dbeta,
                cu,
                self.scale,
                use_initial_state=state0 is not None,
                d_initial_state=dstate0 if self.has_dstate0 else None,
                d_final_state=dstate_in,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=scheduler_bwd,
                scheduler_all=region["scheduler_all"] if self.has_state_checkpoints else None,
                work_item_scratch=region.get("item_scratch") if self.has_state_checkpoints else None,
                order_in_prologue=self.has_state_checkpoints,
                tensormap_workspace=region["bwd_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                log_gate=self.log_gate,
            )

        if self.safe_gate:
            self.channel_gate_bwd(
                dg, g, a_log, dt_bias, d_a_log, d_dt_bias, region.get("gate_part_a"), region.get("gate_part_dt"), self.gate_lower_bound, stream=stream
            )
        if self.fold_dq or self.fold_dk or self.fold_dv:
            for src_ho, dst in ((dq_out, dq), (dk_out, dk), (dv_out, dv)):
                if src_ho is not dst:
                    self.head_group_reduce(src_ho, dst, stream=stream)
        return None

    def run_chain(
        self, q, k, v, g, beta, do, cu, state_checkpoints, state0, dstate_in, dq_out, dk_out, dv_out, dg, dbeta, dstate0, a_log, dt_bias, region, stream
    ) -> None:
        """Chain prologue, H and M summaries (M alone when the series is passed back), forward chain X, G summary, reverse
        chain seeded with d_final_state, dense series (the forward's or the seeded recompute), bprop over the pieces."""
        cu_pieces = region["cu_pieces"]
        work_items, work_count = region["work_items"], region["main_count"]
        summary_items, summary_count = region["work_items_summary"], region["summary_count"]
        state_m, state_g, state_dx_end = region["state_m"], region["state_g"], region["state_dx_end"]
        gate_a_log = a_log if self.safe_gate else None
        gate_dt_bias = dt_bias if self.safe_gate else None
        HO, V, K = state_g.shape[1], state_g.shape[2], state_g.shape[3]
        warm = self.kernel_cache is not None
        if self.has_state_checkpoints and not self.coarse_checkpoints:
            checkpoint_series = state_checkpoints
        else:
            checkpoint_series = region["state_checkpoints"]
        self.run_chain_prologue(
            self.chain_prologue,
            pieces=self.pieces,
            unit_chunks=self.unit_chunks,
            b_t=self.b_t,
            length_rule=self.length_rule,
            heads_out=self.n_heads_out,
            series_span_tokens=self.recompute_span_tokens if self.coarse_checkpoints else 0,
            checkpoint_every_n_tokens=self.b_t,
            cu_seqlens=cu,
            cu_pieces=cu_pieces,
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=work_count,
            summary_count=summary_count,
            work_items=work_items,
            work_items_summary=summary_items,
            scheduler=region["scheduler_all"],
            series_items=region.get("work_items_recompute"),
            series_count=region.get("work_count_recompute"),
            summary_words=region.get("fused_tensormaps"),
            recompute_m_words=region["recompute_tensormaps_m"],
            series_words=region.get("recompute_tensormaps_series"),
            bprop_summary_words=region["summary_tensormaps"],
            bprop_words=region["bwd_tensormaps"],
            q=q,
            k=k,
            v=v,
            gate=g,
            do=do,
            checkpoints=checkpoint_series,
            dq=dq_out,
            dk=dk_out,
            dv=dv_out,
            dgate=dg,
            stream=stream,
        )

        summary_common = dict(
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            safe_gate=self.safe_gate,
            gate_lower_bound=self.gate_lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            work_items=summary_items,
            work_count=summary_count,
            scheduler_counter=region["scheduler_recompute"],
            scheduler_all=region["scheduler_all"],
            device=self.device,
            num_sm=self.num_sm,
            stream=stream,
            own_prologue=False,
        )
        if self.fused_h_m and warm:
            self.fused_summary.run_summary(
                self.fused_cache,
                k,
                v,
                g,
                gate_a_log,
                gate_dt_bias,
                beta,
                cu_pieces,
                None,
                region["state_h"],
                state_m,
                summary_items,
                summary_count,
                region["scheduler_recompute"],
                region["scheduler_all"],
                None,
                region["fused_tensormaps"],
                stream,
                own_prologue=False,
            )
        elif self.fused_h_m:
            self.fused_cache = self.fused_summary.chunk_kda_summary_sm100(
                k,
                v,
                g,
                beta,
                cu_pieces,
                None,
                region["state_h"],
                state_m,
                tensormap_workspace=region["fused_tensormaps"],
                **summary_common,
                log_gate=self.log_gate,
            )
        else:
            if warm:
                self.recompute.run_recompute(
                    self.state_m_cache,
                    k,
                    k,
                    g,
                    gate_a_log,
                    gate_dt_bias,
                    beta,
                    cu_pieces,
                    None,
                    state_m,
                    None,
                    summary_items,
                    summary_count,
                    region["scheduler_m"],
                    region["scheduler_all"],
                    None,
                    region["recompute_tensormaps_m"],
                    0,
                    stream,
                    own_prologue=False,
                )
            else:
                self.state_m_cache = self.recompute.chunk_kda_recompute_sm100(
                    k,
                    k,
                    g,
                    beta,
                    cu_pieces,
                    None,
                    state_m,
                    seed_identity=True,
                    v_is_zero=True,
                    tensormap_workspace=region["recompute_tensormaps_m"],
                    **dict(summary_common, scheduler_counter=region["scheduler_m"]),
                    log_gate=self.log_gate,
                )
        if not self.has_state_checkpoints:
            state_h, state_x = region["state_h"], region["state_x"]
            if not warm:
                self.chain_forward = self.build_state_chain(
                    heads_out=HO,
                    dim_v=V,
                    dim_k=K,
                    pieces=self.pieces,
                    rows_per_cta=self.chain_rows_per_cta(V, K, self.num_seqs, HO, self.num_sm),
                    transpose=False,
                    has_seed=state0 is not None,
                    has_tail=False,
                    emit_summary=False,
                    filled_only=True,
                    seed_dtype=str(state0.dtype) if state0 is not None else "float32",
                    device=self.device,
                )
            self.run_state_chain(self.chain_forward, self.num_seqs, state_h, state_m, state_x, state0, None, None, stream, main_rows=region["main_rows"])

        if warm:
            self.summary.run_bwd_summary(
                self.state_g_cache,
                q,
                k,
                g,
                beta,
                do,
                cu_pieces,
                state_g,
                None,
                summary_items,
                summary_count,
                region["scheduler_summary"],
                region["scheduler_all"],
                None,
                region["summary_tensormaps"],
                self.scale,
                stream,
                a_log=gate_a_log,
                dt_bias=gate_dt_bias,
                own_prologue=False,
            )
        else:
            self.state_g_cache = self.summary.chunk_kda_bwd_summary_sm100(
                q,
                k,
                g,
                beta,
                do,
                state_g,
                cu_pieces,
                self.scale,
                d_final_state=None,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                work_items=summary_items,
                work_count=summary_count,
                scheduler_counter=region["scheduler_summary"],
                scheduler_all=region["scheduler_all"],
                tensormap_workspace=region["summary_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
                log_gate=self.log_gate,
            )
        if not warm:
            self.chain_reverse = self.build_state_chain(
                heads_out=HO,
                dim_v=V,
                dim_k=K,
                pieces=self.pieces,
                rows_per_cta=self.chain_rows_per_cta(V, K, self.num_seqs, HO, self.num_sm),
                transpose=True,
                has_seed=dstate_in is not None,
                has_tail=False,
                emit_summary=False,
                filled_only=True,
                seed_dtype=str(dstate_in.dtype) if dstate_in is not None else "float32",
                device=self.device,
            )
        self.run_state_chain(self.chain_reverse, self.num_seqs, state_g, state_m, state_dx_end, dstate_in, None, None, stream, main_rows=region["main_rows"])

        if not (self.has_state_checkpoints and not self.coarse_checkpoints):
            series_items = region["work_items_recompute"] if self.coarse_checkpoints else work_items
            series_count = region["work_count_recompute"] if self.coarse_checkpoints else work_count
            series_seed = None if self.coarse_checkpoints else region["state_x"]
            seed_kw = dict(
                seed_state_checkpoints=state_checkpoints if self.coarse_checkpoints else None,
                seed_every_n_tokens=self.checkpoint_cadence if self.coarse_checkpoints else 0,
                seed_span_tokens=self.recompute_span_tokens if self.coarse_checkpoints else 0,
            )
            if warm:
                self.recompute.run_recompute(
                    self.recompute_cache,
                    k,
                    v,
                    g,
                    gate_a_log,
                    gate_dt_bias,
                    beta,
                    cu_pieces,
                    series_seed,
                    None,
                    checkpoint_series,
                    series_items,
                    series_count,
                    region["scheduler_series"],
                    region["scheduler_all"],
                    None,
                    region["recompute_tensormaps_series"],
                    self.b_t,
                    stream,
                    own_prologue=False,
                    **seed_kw,
                )
            else:
                self.recompute_cache = self.recompute.chunk_kda_recompute_sm100(
                    k,
                    v,
                    g,
                    beta,
                    cu_pieces,
                    series_seed,
                    None,
                    checkpoint_every_n_tokens=self.b_t,
                    output_state_checkpoints=checkpoint_series,
                    use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                    safe_gate=self.safe_gate,
                    gate_lower_bound=self.gate_lower_bound,
                    a_log=a_log,
                    dt_bias=dt_bias,
                    use_beta_sigmoid=self.use_beta_sigmoid,
                    allow_neg_eigval=self.allow_neg_eigval,
                    work_items=series_items,
                    work_count=series_count,
                    scheduler_counter=region["scheduler_series"],
                    scheduler_all=region["scheduler_all"],
                    tensormap_workspace=region["recompute_tensormaps_series"],
                    device=self.device,
                    num_sm=self.num_sm,
                    stream=stream,
                    own_prologue=False,
                    **seed_kw,
                    log_gate=self.log_gate,
                )

        if warm:
            self.bwd.run_bwd(
                self.kernel_cache,
                q,
                k,
                v,
                g,
                beta,
                do,
                checkpoint_series,
                dq_out,
                dk_out,
                dv_out,
                dg,
                dbeta,
                cu_pieces,
                dstate0,
                state_dx_end,
                work_items,
                work_count,
                region["scheduler_bwd"],
                region["scheduler_all"],
                None,
                region["bwd_tensormaps"],
                self.scale,
                stream,
                a_log=gate_a_log,
                dt_bias=gate_dt_bias,
                own_prologue=False,
            )
        else:
            self.kernel_cache = self.bwd.chunk_kda_bwd_sm100(
                q,
                k,
                v,
                g,
                beta,
                do,
                checkpoint_series,
                dq_out,
                dk_out,
                dv_out,
                dg,
                dbeta,
                cu_pieces,
                self.scale,
                use_initial_state=True,
                d_initial_state=dstate0,
                d_final_state=state_dx_end,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=region["scheduler_bwd"],
                scheduler_all=region["scheduler_all"],
                tensormap_workspace=region["bwd_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
                log_gate=self.log_gate,
            )


def build_kda_summary(graph):
    """Import the summary kernel module (recompute for the forward summary,
    bprop summary for the gradient pass) and wrap the single node."""
    nodes = list(graph.nodes)
    if len(nodes) != 1 or getattr(nodes[0].node_type, "name", None) not in ("KDA_SUMMARY", "KDA_SUMMARY_BWD"):
        raise ValueError("build_kda_summary: graph does not contain exactly one KDA_SUMMARY/KDA_SUMMARY_BWD node")
    node = nodes[0]
    if node.node_type.name == "KDA_SUMMARY_BWD":
        from .kernel import kda_bprop_summary_f16 as summary_module

        return CompiledKdaSummaryBwd(node, summary_module)
    from .kernel import kda_recompute_f16 as recompute_module

    return CompiledKdaSummary(node, recompute_module)


class KdaSummaryFrostEngine(BaseEngine):
    """FROST summary backend for single-node KDA_SUMMARY (final state and span transition through the recompute kernel)
    and KDA_SUMMARY_BWD (d_initial_state through the bprop summary kernel) graphs."""

    name = "kda_summary_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        facts = graph._facts_for(analyze)
        frost_la_gate("KdaSummaryFrostEngine", facts, "KDA_SUMMARY")
        if facts.d_qk not in (64, 128):
            raise NotImplementedError(f"KdaSummaryFrostEngine: k head dim must be 64 or 128, got {facts.d_qk}")
        if facts.d_v not in (64, 128):
            raise NotImplementedError(f"KdaSummaryFrostEngine: v head dim must be 64 or 128, got {facts.d_v}")
        if facts.gate_channels != facts.d_qk:
            raise NotImplementedError(f"KdaSummaryFrostEngine: g must carry d_qk = {facts.d_qk} channels, got {facts.gate_channels}")
        summary_support_gates("KdaSummaryFrostEngine", facts, graph)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_kda_summary(graph))


class CompiledKdaSummary:
    """Compiled summary plan over the state-only recompute kernel: the final-state call (initial_state honored) and, when
    ``transition`` is bound, the identity-seeded zero-value call whose buffer holds ``M_buf = M^T`` (``X_final = X_init @
    M_buf + X_H``).  ``chain`` summarizes every filled piece (H, M) and composes them with one fp32 state chain whose tail
    is ``final_state`` and whose running product is ``transition``."""

    def __init__(self, node, recompute_module):
        from .common.host import tensormap_workspace_bytes
        from .common.piece_chain import build_state_chain, chain_rows_per_cta, choose_pieces, piece_table_layout, run_state_chain
        from .kernel.kda_chain_prologue_f16 import run_chain_prologue
        from .common.split_k import WORK_ITEM_FIELDS, build_split_table, chunk_scratch_rows, compute_ideal_chunks, max_work_items, run_table

        self.node = node
        self.recompute = recompute_module
        self.build_split_table = build_split_table
        self.run_table = run_table
        self.build_state_chain = build_state_chain
        self.chain_rows_per_cta = chain_rows_per_cta
        self.run_state_chain = run_state_chain
        self.table = None
        self.final_cache = None
        self.transition_cache = None
        self.run_chain_prologue = run_chain_prologue
        self.chain_prologue = {}
        self.fused_cache = None
        self.chain_summary = None
        self.plan_name = "KdaSummaryFrostEngine (KDA_SUMMARY)"
        self.device = current_device()

        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.log_gate = (node.params.get("gate_domain") or "log") == "log"
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))
        self.allow_neg_eigval = bool(node.params.get("allow_neg_eigval", False))
        gate_lower_bound = node.params.get("gate_lower_bound")
        self.gate_lower_bound = float(gate_lower_bound) if gate_lower_bound is not None else recompute_module.DEFAULT_GATE_LOWER_BOUND
        self.has_transition = "transition" in node.outputs

        k, g = node.inputs["k"], node.inputs["g"]
        self.b_t = recompute_module.CFG.B_T
        total = k.dim[0]
        HO = g.dim[1]
        K = k.dim[2]
        V = node.inputs["v"].dim[2]
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.num_sm = multiprocessor_count(self.device)
        self.n_tiles = B * HO
        self.n_heads_out = HO
        self.num_seqs = B
        self.dim_k, self.dim_v = K, V
        self.pieces, self.unit_chunks = choose_pieces(
            num_seqs=B,
            heads_out=HO,
            num_sm=self.num_sm,
            total_tokens=total,
            b_t=self.b_t,
            cadence_tokens=0,
            batch_invariant=self.batch_invariant,
            compose_tail=True,
            expand_num=1,
        )
        self.chain = self.pieces > 0
        self.split = not self.chain and not self.batch_invariant
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B

        layout = WorkspaceLayout()
        regions = []
        if self.chain:
            off_scheduler = layout.add(24)
            regions += [
                ("scheduler_h", off_scheduler, "int32", (2,)),
                ("scheduler_m", off_scheduler + 8, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (6,)),
            ]
            table = piece_table_layout(B, self.pieces, HO)
            off_piece_table = layout.add(table.nbytes, align=256)
            regions += [
                ("main_rows", off_piece_table + table.main_rows, "int32", (B + 1,)),
                ("summary_rows", off_piece_table + table.summary_rows, "int32", (B + 1,)),
                ("cu_pieces", off_piece_table + table.cu_pieces, "int32", (self.num_pieces + 1,)),
                ("main_count", off_piece_table + table.main_count, "int32", (1,)),
                ("summary_count", off_piece_table + table.summary_count, "int32", (1,)),
                ("work_items", layout.add(table.item_rows * WORK_ITEM_FIELDS * 4), "int32", (table.item_rows, WORK_ITEM_FIELDS)),
                ("state_h", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
                ("state_m", layout.add(self.num_pieces * HO * K * K * 4), "float32", (self.num_pieces, HO, K, K)),
                ("state_x", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
            ]
            from .kernel import kda_summary_f16 as summary_module

            self.fused_summary = summary_module
            fused_bytes = tensormap_workspace_bytes(summary_module, self.num_pieces)
            regions.append(("fused_tensormaps", layout.add(fused_bytes, align=128), "int64", (fused_bytes // 8,)))
        else:
            off_scheduler = layout.add(16)
            regions += [
                ("scheduler_final", off_scheduler, "int32", (2,)),
                ("scheduler_transition", off_scheduler + 8, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (4,)),
            ]
            if self.split:
                self.ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
                self.work_item_rows = max_work_items(total, B, HO, self.ideal, self.b_t, self.num_sm)
            else:
                self.ideal = None
                self.work_item_rows = self.n_tiles
            regions.append(("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
            regions.append(("work_count", layout.add(4), "int32", (1,)))
            if self.split:
                self.chunk_scratch_rows = chunk_scratch_rows(total, B, self.b_t)
                regions.append(("item_scratch", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
                regions.append(("chunk_scratch", layout.add(self.chunk_scratch_rows * HO * 4), "float32", (self.chunk_scratch_rows, HO)))
            tensormap_bytes = tensormap_workspace_bytes(recompute_module, B)
            regions.append(("tensormaps", layout.add(tensormap_bytes, align=128), "int64", (tensormap_bytes // 8,)))
        self.needs_table = self.split
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
        self.index_beta = pos["beta"]
        self.index_cu_seqlens = pos["cu_seqlens"]
        self.index_initial_state = pos.get("initial_state")
        self.index_a_log = pos.get("a_log")
        self.index_dt_bias = pos.get("dt_bias")
        self.index_final_state = pos["final_state"]
        self.index_transition = pos.get("transition")

    def run(self, views, workspace, stream) -> None:
        k = views[self.index_k]
        v = views[self.index_v]
        g = views[self.index_g]
        beta = views[self.index_beta]
        cu = views[self.index_cu_seqlens]
        state0 = views[self.index_initial_state] if self.index_initial_state is not None else None
        a_log = views[self.index_a_log] if self.index_a_log is not None else None
        dt_bias = views[self.index_dt_bias] if self.index_dt_bias is not None else None
        final_state = views[self.index_final_state]
        transition = views[self.index_transition] if self.index_transition is not None else None
        stream = stream if stream is not None else 0

        region = dict(zip(self.carve_names, workspace.carve(self.carve)))
        if self.chain:
            self.run_chain(k, v, g, beta, cu, state0, final_state, transition, a_log, dt_bias, region, stream)
            return
        work_items = region["work_items"]
        work_count = region["work_count"]
        item_scratch = region.get("item_scratch")

        if self.final_cache is not None and (self.table is not None or not self.needs_table):
            if self.needs_table:
                self.run_table(
                    self.table,
                    g,
                    a_log,
                    dt_bias,
                    cu,
                    region.get("chunk_scratch"),
                    item_scratch,
                    work_items,
                    work_count,
                    region["scheduler_all"],
                    stream,
                )
            self.recompute.run_recompute(
                self.final_cache,
                k,
                v,
                g,
                a_log if self.safe_gate else None,
                dt_bias if self.safe_gate else None,
                beta,
                cu,
                state0,
                final_state,
                None,
                work_items,
                work_count,
                region["scheduler_final"],
                region["scheduler_all"],
                item_scratch,
                region["tensormaps"],
                0,
                stream,
            )
            if self.has_transition:
                self.recompute.run_recompute(
                    self.transition_cache,
                    k,
                    k,
                    g,
                    a_log if self.safe_gate else None,
                    dt_bias if self.safe_gate else None,
                    beta,
                    cu,
                    None,
                    transition,
                    None,
                    work_items,
                    work_count,
                    region["scheduler_transition"],
                    region["scheduler_all"],
                    item_scratch,
                    region["tensormaps"],
                    0,
                    stream,
                )
            return

        if not self.needs_table:
            self.table = None
        else:
            self.table = self.build_split_table(
                g,
                cu,
                work_items,
                work_count,
                ideal_chunks=self.ideal,
                n_tiles=self.n_tiles,
                num_sms=self.num_sm,
                b_t=self.b_t,
                chunk_scratch=region.get("chunk_scratch"),
                item_scratch=item_scratch,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                gate_lower_bound=self.gate_lower_bound if self.safe_gate else None,
                scheduler_counter=region["scheduler_all"],
                split=self.split,
                stream=stream,
            )

        self.final_cache = self.recompute.chunk_kda_recompute_sm100(
            k,
            v,
            g,
            beta,
            cu,
            state0,
            final_state,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            safe_gate=self.safe_gate,
            gate_lower_bound=self.gate_lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            work_items=work_items,
            work_count=work_count,
            scheduler_counter=region["scheduler_final"],
            scheduler_all=region["scheduler_all"],
            work_item_scratch=item_scratch,
            order_in_prologue=True,
            tensormap_workspace=region["tensormaps"],
            device=self.device,
            num_sm=self.num_sm,
            stream=stream,
            log_gate=self.log_gate,
        )
        if self.has_transition:
            self.transition_cache = self.recompute.chunk_kda_recompute_sm100(
                k,
                k,
                g,
                beta,
                cu,
                None,
                transition,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                seed_identity=True,
                v_is_zero=True,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=region["scheduler_transition"],
                scheduler_all=region["scheduler_all"],
                work_item_scratch=item_scratch,
                order_in_prologue=True,
                tensormap_workspace=region["tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                log_gate=self.log_gate,
            )
        return None

    def run_chain(self, k, v, g, beta, cu, state0, final_state, transition, a_log, dt_bias, region, stream) -> None:
        """Chain prologue, fused H and M summaries of every filled piece, one fp32 state chain seeded with initial_state whose
        tail is final_state and whose running product is transition."""
        cu_pieces = region["cu_pieces"]
        work_items, work_count = region["work_items"], region["main_count"]
        state_h, state_m, state_x = region["state_h"], region["state_m"], region["state_x"]
        gate_a_log = a_log if self.safe_gate else None
        gate_dt_bias = dt_bias if self.safe_gate else None
        warm = self.chain_summary is not None
        self.run_chain_prologue(
            self.chain_prologue,
            pieces=self.pieces,
            unit_chunks=self.unit_chunks,
            b_t=self.b_t,
            length_rule=self.length_rule,
            heads_out=self.n_heads_out,
            cu_seqlens=cu,
            cu_pieces=cu_pieces,
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=work_count,
            summary_count=region["summary_count"],
            work_items=work_items,
            scheduler=region["scheduler_all"],
            summary_words=region.get("fused_tensormaps"),
            k=k,
            v=v,
            gate=g,
            stream=stream,
        )
        if warm:
            self.fused_summary.run_summary(
                self.fused_cache,
                k,
                v,
                g,
                gate_a_log,
                gate_dt_bias,
                beta,
                cu_pieces,
                None,
                state_h,
                state_m,
                work_items,
                work_count,
                region["scheduler_h"],
                region["scheduler_all"],
                None,
                region["fused_tensormaps"],
                stream,
                own_prologue=False,
            )
        else:
            summary_common = dict(
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                work_items=work_items,
                work_count=work_count,
                scheduler_all=region["scheduler_all"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
            )
            self.fused_cache = self.fused_summary.chunk_kda_summary_sm100(
                k,
                v,
                g,
                beta,
                cu_pieces,
                None,
                state_h,
                state_m,
                scheduler_counter=region["scheduler_h"],
                tensormap_workspace=region["fused_tensormaps"],
                **summary_common,
                log_gate=self.log_gate,
            )
            self.chain_summary = self.build_state_chain(
                heads_out=self.n_heads_out,
                dim_v=self.dim_v,
                dim_k=self.dim_k,
                pieces=self.pieces,
                rows_per_cta=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                transpose=False,
                has_seed=state0 is not None,
                has_tail=True,
                emit_summary=self.has_transition,
                filled_only=True,
                seed_dtype=str(state0.dtype) if state0 is not None else "float32",
                tail_dtype=str(final_state.dtype),
                summary_dtype=str(transition.dtype) if transition is not None else "float32",
                device=self.device,
            )
        self.run_state_chain(
            self.chain_summary, self.num_seqs, state_h, state_m, state_x, state0, final_state, transition, stream, main_rows=region["main_rows"]
        )


class CompiledKdaSummaryBwd:
    """Compiled gradient summary over the bprop summary kernel: the reverse state-gradient recurrence seeded by
    ``d_final_state`` (zero when unbound) writes ``d_initial_state``; a bound ``transition`` receives ``M^T`` from the
    recompute's transition arm.  ``chain`` summarizes every filled piece (G, M) and composes them with one reverse fp32
    state chain whose tail is ``d_initial_state`` and whose running product is ``transition``."""

    def __init__(self, node, summary_module):
        from .common.host import tensormap_workspace_bytes
        from .common.piece_chain import build_state_chain, chain_rows_per_cta, choose_pieces, piece_table_layout, run_state_chain
        from .kernel.kda_chain_prologue_f16 import run_chain_prologue
        from .common.split_k import WORK_ITEM_FIELDS, build_split_table, chunk_scratch_rows, compute_ideal_chunks, max_work_items, run_table

        self.node = node
        self.summary = summary_module
        self.build_split_table = build_split_table
        self.run_table = run_table
        self.build_state_chain = build_state_chain
        self.chain_rows_per_cta = chain_rows_per_cta
        self.run_state_chain = run_state_chain
        self.table = None
        self.kernel_cache = None
        self.run_chain_prologue = run_chain_prologue
        self.chain_prologue = {}
        self.state_g_cache = None
        self.state_m_cache = None
        self.chain_reverse = None
        self.transition_cache = None
        self.transition_chain = None
        self.plan_name = "KdaSummaryFrostEngine (KDA_SUMMARY_BWD)"
        self.device = current_device()

        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.log_gate = (node.params.get("gate_domain") or "log") == "log"
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))
        self.allow_neg_eigval = bool(node.params.get("allow_neg_eigval", False))
        gate_lower_bound = node.params.get("gate_lower_bound")
        self.gate_lower_bound = float(gate_lower_bound) if gate_lower_bound is not None else summary_module.DEFAULT_GATE_LOWER_BOUND
        self.has_transition = "transition" in node.outputs

        q, do, g = node.inputs["q"], node.inputs["dO"], node.inputs["g"]
        self.b_t = summary_module.CFG.B_T
        total = q.dim[0]
        HO = g.dim[1]
        K, V = q.dim[-1], do.dim[-1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"
        self.n_heads_out, self.total = HO, total
        self.num_seqs = B
        self.dim_k, self.dim_v = K, V
        self.num_sm = multiprocessor_count(self.device)
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.pieces, self.unit_chunks = choose_pieces(
            num_seqs=B,
            heads_out=HO,
            num_sm=self.num_sm,
            total_tokens=total,
            b_t=self.b_t,
            cadence_tokens=0,
            batch_invariant=self.batch_invariant,
            compose_tail=True,
            reverse=True,
            expand_num=1,
        )
        self.chain = self.pieces > 0
        self.split = not self.chain and not self.batch_invariant
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        layout = WorkspaceLayout()
        regions = []
        if self.chain:
            from .kernel import kda_recompute_f16 as recompute_module

            self.recompute = recompute_module
            off_scheduler = layout.add(24)
            regions += [
                ("scheduler_g", off_scheduler, "int32", (2,)),
                ("scheduler_m", off_scheduler + 8, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (6,)),
            ]
            table = piece_table_layout(B, self.pieces, HO)
            off_piece_table = layout.add(table.nbytes, align=256)
            regions += [
                ("main_rows", off_piece_table + table.main_rows, "int32", (B + 1,)),
                ("summary_rows", off_piece_table + table.summary_rows, "int32", (B + 1,)),
                ("cu_pieces", off_piece_table + table.cu_pieces, "int32", (self.num_pieces + 1,)),
                ("main_count", off_piece_table + table.main_count, "int32", (1,)),
                ("summary_count", off_piece_table + table.summary_count, "int32", (1,)),
                ("work_items", layout.add(table.item_rows * WORK_ITEM_FIELDS * 4), "int32", (table.item_rows, WORK_ITEM_FIELDS)),
                ("state_g", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
                ("state_m", layout.add(self.num_pieces * HO * K * K * 4), "float32", (self.num_pieces, HO, K, K)),
                ("state_x", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
            ]
            summary_bytes = tensormap_workspace_bytes(summary_module, self.num_pieces)
            regions.append(("summary_tensormaps", layout.add(summary_bytes, align=128), "int64", (summary_bytes // 8,)))
            recompute_bytes = tensormap_workspace_bytes(recompute_module, self.num_pieces)
            regions.append(("recompute_tensormaps_m", layout.add(recompute_bytes, align=128), "int64", (recompute_bytes // 8,)))
        else:
            off_scheduler = layout.add(16)
            self.n_tiles = B * HO
            if self.split:
                self.ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
                self.work_item_rows = max_work_items(total, B, HO, self.ideal, self.b_t, self.num_sm)
            else:
                self.ideal = None
                self.work_item_rows = self.n_tiles
            regions += [
                ("scheduler_main", off_scheduler, "int32", (2,)),
                ("scheduler_transition", off_scheduler + 8, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (4,)),
                ("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)),
                ("work_count", layout.add(4), "int32", (1,)),
            ]
            if self.split:
                self.chunk_scratch_rows = chunk_scratch_rows(total, B, self.b_t)
                regions.append(("item_scratch", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
                regions.append(("chunk_scratch", layout.add(self.chunk_scratch_rows * HO * 4), "float32", (self.chunk_scratch_rows, HO)))
            tensormap_bytes = tensormap_workspace_bytes(summary_module, B)
            regions.append(("tensormaps", layout.add(tensormap_bytes, align=128), "int64", (tensormap_bytes // 8,)))
            if self.has_transition:
                from .kernel import kda_recompute_f16 as recompute_module

                self.recompute = recompute_module
                recompute_bytes = tensormap_workspace_bytes(recompute_module, B)
                regions.append(("recompute_tensormaps", layout.add(recompute_bytes, align=128), "int64", (recompute_bytes // 8,)))
                regions.append(("transition_m", layout.add(B * HO * K * K * 4), "float32", (B, HO, K, K)))
        self.needs_table = self.split
        self.workspace_size = layout.size
        self.carve_names = [name for name, off, dt, shape in regions]
        self.carve = carve_plan(self.plan_name, [(off, dt, shape) for name, off, dt, shape in regions])

    def workspace_bytes(self) -> int:
        return self.workspace_size

    def bind(self, names) -> None:
        pos = {name: i for i, name in enumerate(names)}
        self.index_q = pos["q"]
        self.index_k = pos["k"]
        self.index_g = pos["g"]
        self.index_beta = pos["beta"]
        self.index_cu_seqlens = pos["cu_seqlens"]
        self.index_do = pos["dO"]
        self.index_d_final_state = pos.get("d_final_state")
        self.index_a_log = pos.get("a_log")
        self.index_dt_bias = pos.get("dt_bias")
        self.index_d_initial_state = pos["d_initial_state"]
        self.index_transition = pos.get("transition")

    def run(self, views, workspace, stream) -> None:
        q = views[self.index_q]
        k = views[self.index_k]
        g = views[self.index_g]
        beta = views[self.index_beta]
        cu = views[self.index_cu_seqlens]
        do = views[self.index_do]
        dstate_in = views[self.index_d_final_state] if self.index_d_final_state is not None else None
        a_log = views[self.index_a_log] if self.index_a_log is not None else None
        dt_bias = views[self.index_dt_bias] if self.index_dt_bias is not None else None
        dstate0 = views[self.index_d_initial_state]
        transition = views[self.index_transition] if self.index_transition is not None else None
        stream = stream if stream is not None else 0

        region = dict(zip(self.carve_names, workspace.carve(self.carve)))
        if self.chain:
            self.run_chain(q, k, g, beta, do, cu, dstate_in, dstate0, transition, a_log, dt_bias, region, stream)
            return
        work_items = region["work_items"]
        work_count = region["work_count"]

        if self.kernel_cache is not None and (self.table is not None or not self.needs_table):
            if self.needs_table:
                self.run_table(
                    self.table,
                    g,
                    a_log,
                    dt_bias,
                    cu,
                    region.get("chunk_scratch"),
                    region.get("item_scratch"),
                    work_items,
                    work_count,
                    region["scheduler_all"],
                    stream,
                )
            self.summary.run_bwd_summary(
                self.kernel_cache,
                q,
                k,
                g,
                beta,
                do,
                cu,
                dstate0,
                dstate_in,
                work_items,
                work_count,
                region["scheduler_main"],
                region["scheduler_all"],
                region.get("item_scratch"),
                region["tensormaps"],
                self.scale,
                stream,
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
            )
            if self.has_transition:
                self.run_transition(k, g, beta, cu, transition, a_log, dt_bias, region, stream)
            return

        if not self.needs_table:
            self.table = None
        else:
            self.table = self.build_split_table(
                g,
                cu,
                work_items,
                work_count,
                ideal_chunks=self.ideal,
                n_tiles=self.n_tiles,
                num_sms=self.num_sm,
                b_t=self.b_t,
                chunk_scratch=region.get("chunk_scratch"),
                item_scratch=region.get("item_scratch"),
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                gate_lower_bound=self.gate_lower_bound if self.safe_gate else None,
                scheduler_counter=region["scheduler_all"],
                split=self.split,
                stream=stream,
            )

        self.kernel_cache = self.summary.chunk_kda_bwd_summary_sm100(
            q,
            k,
            g,
            beta,
            do,
            dstate0,
            cu,
            self.scale,
            d_final_state=dstate_in,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            safe_gate=self.safe_gate,
            gate_lower_bound=self.gate_lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            work_items=work_items,
            work_count=work_count,
            scheduler_counter=region["scheduler_main"],
            scheduler_all=region["scheduler_all"],
            work_item_scratch=region.get("item_scratch"),
            order_in_prologue=True,
            tensormap_workspace=region["tensormaps"],
            device=self.device,
            num_sm=self.num_sm,
            stream=stream,
            log_gate=self.log_gate,
        )
        if self.has_transition:
            self.run_transition(k, g, beta, cu, transition, a_log, dt_bias, region, stream)
        return None

    def run_transition(self, k, g, beta, cu, transition, a_log, dt_bias, region, stream) -> None:
        """The recompute's transition arm over the work-item table writes M into
        ``transition_m``; a one-slot reverse state chain emits its product,
        ``M^T``, into ``transition``."""
        transition_m = region["transition_m"]
        if self.transition_cache is not None:
            self.recompute.run_recompute(
                self.transition_cache,
                k,
                k,
                g,
                a_log if self.safe_gate else None,
                dt_bias if self.safe_gate else None,
                beta,
                cu,
                None,
                transition_m,
                None,
                region["work_items"],
                region["work_count"],
                region["scheduler_transition"],
                region["scheduler_all"],
                region.get("item_scratch"),
                region["recompute_tensormaps"],
                0,
                stream,
            )
        else:
            self.transition_cache = self.recompute.chunk_kda_recompute_sm100(
                k,
                k,
                g,
                beta,
                cu,
                None,
                transition_m,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                seed_identity=True,
                v_is_zero=True,
                work_items=region["work_items"],
                work_count=region["work_count"],
                scheduler_counter=region["scheduler_transition"],
                scheduler_all=region["scheduler_all"],
                work_item_scratch=region.get("item_scratch"),
                order_in_prologue=True,
                tensormap_workspace=region["recompute_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                log_gate=self.log_gate,
            )
            self.transition_chain = self.build_state_chain(
                heads_out=self.n_heads_out,
                dim_v=self.dim_v,
                dim_k=self.dim_k,
                pieces=1,
                rows_per_cta=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                transpose=True,
                has_seed=False,
                has_tail=False,
                emit_summary=True,
                summary_dtype=str(transition.dtype),
                device=self.device,
            )
        self.run_state_chain(self.transition_chain, self.num_seqs, None, transition_m, None, None, None, transition, stream)

    def run_chain(self, q, k, g, beta, do, cu, dstate_in, dstate0, transition, a_log, dt_bias, region, stream) -> None:
        """Chain prologue, M (recompute transition arm) and G (zero-seeded bprop summary) of every filled piece, one reverse
        fp32 state chain seeded with d_final_state whose tail is d_initial_state and whose running product is transition."""
        cu_pieces = region["cu_pieces"]
        work_items, work_count = region["work_items"], region["main_count"]
        state_g, state_m, state_x = region["state_g"], region["state_m"], region["state_x"]
        gate_a_log = a_log if self.safe_gate else None
        gate_dt_bias = dt_bias if self.safe_gate else None
        warm = self.chain_reverse is not None
        self.run_chain_prologue(
            self.chain_prologue,
            pieces=self.pieces,
            unit_chunks=self.unit_chunks,
            b_t=self.b_t,
            length_rule=self.length_rule,
            heads_out=self.n_heads_out,
            cu_seqlens=cu,
            cu_pieces=cu_pieces,
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=work_count,
            summary_count=region["summary_count"],
            work_items=work_items,
            scheduler=region["scheduler_all"],
            recompute_m_words=region["recompute_tensormaps_m"],
            bprop_summary_words=region["summary_tensormaps"],
            q=q,
            k=k,
            gate=g,
            do=do,
            stream=stream,
        )
        if warm:
            self.recompute.run_recompute(
                self.state_m_cache,
                k,
                k,
                g,
                gate_a_log,
                gate_dt_bias,
                beta,
                cu_pieces,
                None,
                state_m,
                None,
                work_items,
                work_count,
                region["scheduler_m"],
                region["scheduler_all"],
                None,
                region["recompute_tensormaps_m"],
                0,
                stream,
                own_prologue=False,
            )
            self.summary.run_bwd_summary(
                self.state_g_cache,
                q,
                k,
                g,
                beta,
                do,
                cu_pieces,
                state_g,
                None,
                work_items,
                work_count,
                region["scheduler_g"],
                region["scheduler_all"],
                None,
                region["summary_tensormaps"],
                self.scale,
                stream,
                a_log=gate_a_log,
                dt_bias=gate_dt_bias,
                own_prologue=False,
            )
        else:
            self.state_m_cache = self.recompute.chunk_kda_recompute_sm100(
                k,
                k,
                g,
                beta,
                cu_pieces,
                None,
                state_m,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                seed_identity=True,
                v_is_zero=True,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=region["scheduler_m"],
                scheduler_all=region["scheduler_all"],
                tensormap_workspace=region["recompute_tensormaps_m"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
                log_gate=self.log_gate,
            )
            self.state_g_cache = self.summary.chunk_kda_bwd_summary_sm100(
                q,
                k,
                g,
                beta,
                do,
                state_g,
                cu_pieces,
                self.scale,
                d_final_state=None,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=region["scheduler_g"],
                scheduler_all=region["scheduler_all"],
                tensormap_workspace=region["summary_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
                log_gate=self.log_gate,
            )
            self.chain_reverse = self.build_state_chain(
                heads_out=self.n_heads_out,
                dim_v=self.dim_v,
                dim_k=self.dim_k,
                pieces=self.pieces,
                rows_per_cta=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                transpose=True,
                has_seed=dstate_in is not None,
                has_tail=True,
                emit_summary=self.has_transition,
                filled_only=True,
                seed_dtype=str(dstate_in.dtype) if dstate_in is not None else "float32",
                tail_dtype=str(dstate0.dtype),
                summary_dtype=str(transition.dtype) if transition is not None else "float32",
                device=self.device,
            )
        self.run_state_chain(
            self.chain_reverse, self.num_seqs, state_g, state_m, state_x, dstate_in, dstate0, transition, stream, main_rows=region["main_rows"]
        )
