# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDN engine: GDN / GDN_BWD nodes on the chunked prefill and backward kernels (SM100/SM103/SM107, bf16/fp16); the
backward regenerates the checkpoint series with the recompute kernel when the graph carries none.  Long sequences on few
(sequence, head) tiles run as an exact piece chain (``common/piece_chain.py``) or as the decay-warmup split-K.
GDP/GDP_BWD nodes run on the ``num_householder``-expanded timeline."""

from __future__ import annotations

import math

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost.device import build_device, current_device, multiprocessor_count
from cudnn.frost.workspace import WorkspaceLayout, carve_plan
from ..graph_analyzer import analyze
from .engine import FrostLaPlan, frost_la_gate, summary_support_gates


def build_gdn(graph):
    """Import the kernel module (pulls in the Cutlass primitives) and wrap the
    single node; cute.compile is cached inside the kernel per static config and
    runs on first execute, when the real buffers are known."""
    nodes = list(graph.nodes)
    if len(nodes) != 1 or getattr(nodes[0].node_type, "name", None) not in ("GDN", "GDN_BWD", "GDP", "GDP_BWD"):
        raise ValueError("build_gdn: graph does not contain exactly one GDN/GDN_BWD/GDP/GDP_BWD node")
    node = nodes[0]
    gdp_v64_bwd = node.node_type.name == "GDP_BWD" and int(node.inputs["v"].dim[-1]) == 64 and int(node.params.get("num_householder", 1) or 1) > 1
    if node.node_type.name in ("GDN_BWD", "GDP_BWD"):
        from .kernel import gdn_recompute_f16 as recompute_module

        if gdp_v64_bwd:
            from .kernel import gdp_bprop_v64_f16 as bwd_module
        else:
            from .kernel import gdn_bprop_f16 as bwd_module

        return CompiledGdnBwd(node, bwd_module, recompute_module)
    from .kernel import gdn_prefill_f16 as kernel_module

    return CompiledGdn(node, kernel_module)


def gdn_support_gates(engine: str, facts) -> None:
    """The GDN kernel family's dtype/attribute gates, shared with the GDP engine."""
    import cudnn

    checkpoint = facts.checkpoint_every_n_tokens
    if checkpoint and checkpoint % 64 != 0:
        raise NotImplementedError(f"{engine}: checkpoint_every_n_tokens must be a positive multiple of 64 (got {checkpoint})")
    if not facts.gates_at_ho:
        raise NotImplementedError(f"{engine}: g/beta must carry HO = max(q, v) heads ({facts.h_o})")
    io = (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
    state_dtypes = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16)
    beta_wants = (cudnn.data_type.FLOAT, facts.io_dtype)
    if facts.beta_dtype not in beta_wants + (None,):
        raise NotImplementedError(f"{engine}: 'beta' must be {' or '.join(str(w) for w in beta_wants)}, got {facts.beta_dtype}")
    gate_param_dtypes = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
    for port, got in (("a_log", facts.a_log_dtype), ("dt_bias", facts.dt_bias_dtype)):
        if got not in gate_param_dtypes + (None,):
            raise NotImplementedError(f"{engine}: '{port}' must be fp32/bf16/fp16, got {got}")
    for port, got in (("initial_state", facts.state_dtype), ("final_state", facts.final_state_dtype)):
        if got not in state_dtypes + (None,):
            raise NotImplementedError(f"{engine}: '{port}' must be fp32/bf16, got {got}")
    if not facts.state_pair_match:
        raise NotImplementedError(f"{engine}: initial_state and final_state dtypes must match")
    if facts.is_bwd:
        for port, got, want in (("d_a_log", facts.d_a_log_dtype, facts.a_log_dtype), ("d_dt_bias", facts.d_dt_bias_dtype, facts.dt_bias_dtype)):
            if got not in (want, None):
                raise NotImplementedError(f"{engine}: '{port}' must match its parameter dtype ({want}), got {got}")
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
        if facts.dbeta_dtype not in (facts.beta_dtype, None):
            raise NotImplementedError(f"{engine}: 'dBeta' must match 'beta' ({facts.beta_dtype}), got {facts.dbeta_dtype}")
    elif facts.io_dtype is not None and facts.state_checkpoints_out_dtype not in (facts.io_dtype, None):
        raise NotImplementedError(f"{engine}: 'state_checkpoints' must match the io dtype")


class GdnFrostEngine(BaseEngine):
    """FROST chunked-kernel backend for single-node GDN graphs (THD layout); the default GDN engine on SM100/SM103/SM107,
    declining elsewhere so ranking falls back to ``GdnCuTileEngine``."""

    name = "gdn_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        facts = graph._facts_for(analyze)
        frost_la_gate("GdnFrostEngine", facts, "GDN")
        if facts.d_qk not in (64, 128):
            raise NotImplementedError(f"GdnFrostEngine: q/k head dim must be 64 or 128, got {facts.d_qk}")
        if facts.d_v not in (64, 128):
            raise NotImplementedError(f"GdnFrostEngine: v head dim must be 64 or 128, got {facts.d_v}")
        gdn_support_gates("GdnFrostEngine", facts)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_gdn(graph))


def chunk_factor_pass(
    cache,
    k,
    g,
    beta,
    cu,
    tinv,
    workspace,
    *,
    log_gate,
    safe_gate,
    a_log,
    dt_bias,
    use_beta_sigmoid,
    allow_neg_eigval,
    expand_num,
    device,
    stream,
    tinv_rows,
    tinv_row_count,
    publish_desc=True,
):
    """Run the T pass over ``cu`` (compile on the first call; ``workspace`` holds its per-batch K and tinv
    descriptor arrays, emitted by the pass itself or, ``publish_desc=False``, by the caller's prologue) and return its
    compiled cache."""
    from .kernel import gdn_tinv_f16 as tinv_module

    if cache is None:
        return tinv_module.chunk_gdn_tinv_sm100(
            k,
            g,
            beta,
            cu,
            tinv,
            log_gate=log_gate,
            safe_gate=safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            expand_num=expand_num,
            workspace=workspace,
            row_table=tinv_rows,
            row_count=tinv_row_count,
            device=device,
            stream=stream,
            publish_desc=publish_desc,
        )
    tinv_module.run_tinv(
        cache,
        k,
        g,
        beta,
        cu,
        tinv,
        workspace,
        stream,
        a_log=a_log if safe_gate else None,
        dt_bias=dt_bias if safe_gate else None,
        row_table=tinv_rows,
        row_count=tinv_row_count,
    )
    return cache


class CompiledGdn:
    """Compiled FROST GDN / GDP plan over the resolved node buffers.  ``choose_pieces`` fixes the scheme at build: ``uncut``
    (one item per sequence and head), ``warmup`` (decay-warmup split-K) or ``chain`` (per-piece H and M from the fused
    summary, an fp32 state chain seeding every piece, the prefill over the pieces as independent sequences).  The chunk
    factor T comes from the T pass when several kernels consume it or the plan fills under half the SMs."""

    def __init__(self, node, kernel_module):
        from .common.piece_chain import build_state_chain, chain_rows_per_cta, choose_pieces, piece_table_layout, run_state_chain
        from .kernel.gdn_chain_prologue_f16 import run_chain_prologue
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
        self.tinv_cache = None
        self.run_chain_prologue = run_chain_prologue
        self.chain_prologue = {}
        self.chain_forward = None
        self.plan_name = "GdpFrostEngine (GDP)" if node.node_type.name == "GDP" else "GdnFrostEngine (GDN)"
        self.device = current_device()
        from .common.l2norm import build_l2norm_qk, run_l2norm_qk

        self.build_l2norm_qk = build_l2norm_qk
        self.run_l2norm_qk = run_l2norm_qk
        self.l2norm = None
        self.num_householder = int(node.params.get("num_householder", 1) or 1)
        self.expand_num = self.num_householder
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.log_gate = (node.params.get("gate_domain") or "log") == "log"
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))
        self.allow_neg_eigval = bool(node.params.get("allow_neg_eigval", False))

        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self.b_t = kernel_module.CFG.B_T
        total = node.inputs["k"].dim[0]
        HO = g.dim[1]
        HQ, HK = q.dim[1], node.inputs["k"].dim[1]
        K = q.dim[2]
        V = v.dim[2]
        self.io_name = "float16" if q.get_data_type().name == "HALF" else "bfloat16"
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.has_final_state = "final_state" in node.outputs
        self.checkpoint = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.has_state_checkpoints = "state_checkpoints" in node.outputs
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.num_sm = multiprocessor_count(self.device)
        self.n_heads_out = HO
        self.dim_v, self.dim_k = V, K
        self.num_seqs = B
        self.q_rows = total // self.expand_num
        self.pieces, self.unit_chunks = choose_pieces(
            num_seqs=B,
            heads_out=HO,
            num_sm=self.num_sm,
            total_tokens=self.q_rows,
            b_t=self.b_t,
            cadence_tokens=self.checkpoint,
            batch_invariant=self.batch_invariant,
            expand_num=self.expand_num,
        )
        self.chain = self.pieces > 0
        self.split = not self.chain and not self.batch_invariant
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        self.tinv_pass = self.chain or 2 * self.num_pieces * HO < self.num_sm

        layout = WorkspaceLayout()
        from .common.host import tensormap_workspace_bytes

        regions = []
        if self.chain:
            self.n_tiles = self.num_pieces * HO
            self.work_item_rows = self.n_tiles
            self.ideal = None
            self.tensormap_words = tensormap_workspace_bytes(kernel_module, self.num_pieces) // 8
            off_scheduler = layout.add(24)
            regions += [
                ("scheduler_prefill", off_scheduler, "int32", (2,)),
                ("scheduler_h", off_scheduler + 8, "int32", (2,)),
                ("scheduler_m", off_scheduler + 16, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (6,)),
                ("tensormaps", layout.add(self.tensormap_words * 8), "int64", (self.tensormap_words,)),
                ("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)),
            ]
            from .kernel import gdn_summary_f16 as summary_module

            self.fused_summary = summary_module
            self.fused_cache = None
            self.fused_tensormap_words = tensormap_workspace_bytes(summary_module, self.num_pieces) // 8
            regions.append(("fused_tensormaps", layout.add(self.fused_tensormap_words * 8), "int64", (self.fused_tensormap_words,)))
            table = piece_table_layout(B, self.pieces, HO)
            off_piece_table = layout.add(table.nbytes)
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
        else:
            self.tensormap_words = tensormap_workspace_bytes(kernel_module, B) // 8
            regions.append(("tensormaps", layout.add(self.tensormap_words * 8), "int64", (self.tensormap_words,)))
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
        if self.tinv_pass:
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(self.q_rows, self.num_pieces, self.expand_num, self.b_t)
            regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, self.num_pieces) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
        if self.use_qk_l2norm:
            regions.append(("q_n", layout.add(self.q_rows * HQ * K * 2), self.io_name, (self.q_rows, HQ, K)))
            regions.append(("k_n", layout.add(total * HK * K * 2), self.io_name, (total, HK, K)))
            regions.append(("inv_q", layout.add(self.q_rows * HQ * 4), "float32", (self.q_rows, HQ)))
            regions.append(("inv_k", layout.add(total * HK * 4), "float32", (total, HK)))
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
        if self.use_qk_l2norm:
            q_n, k_n, inv_q, inv_k = region["q_n"], region["k_n"], region["inv_q"], region["inv_k"]
            if self.l2norm is None:
                self.l2norm = self.build_l2norm_qk(q, k, q_n, k_n, inv_q, inv_k, stream=stream)
            else:
                self.run_l2norm_qk(self.l2norm, q, k, q_n, k_n, inv_q, inv_k, stream)
            q, k = q_n, k_n
        if self.chain:
            self.run_chain(q, k, v, g, beta, cu, state0, o, final_state, state_checkpoints, a_log, dt_bias, region, stream)
            return

        tensormaps = region["tensormaps"]
        scheduler_counter = region["scheduler"]
        work_items = region["work_items"]
        work_count = region["work_count"]
        item_scratch = region.get("item_scratch")
        chunk_scratch = region.get("chunk_scratch")
        factor = dict(tinv=None, beta=beta, use_beta_sigmoid=self.use_beta_sigmoid, allow_neg_eigval=self.allow_neg_eigval)
        if self.kernel_cache is not None and (self.table is not None or not self.needs_table):
            if self.needs_table:
                self.run_table(
                    self.table,
                    g,
                    a_log,
                    dt_bias,
                    cu,
                    chunk_scratch,
                    item_scratch,
                    work_items,
                    work_count,
                    scheduler_counter,
                    stream,
                    expand_num=self.expand_num,
                )
            self.kernel.run_prefill(
                self.kernel_cache,
                q,
                k,
                v,
                g,
                o,
                cu,
                state0,
                final_state,
                state_checkpoints,
                work_items,
                work_count,
                scheduler_counter,
                item_scratch,
                tensormaps,
                self.checkpoint,
                self.scale,
                stream,
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
                tinv=None,
                beta=beta,
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
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
                scheduler_counter=scheduler_counter,
                split=self.split,
                expand_num=self.expand_num,
                stream=stream,
            )

        self.kernel_cache = self.kernel.chunk_gdn_sm100(
            q,
            k,
            v,
            g,
            o,
            cu,
            state0,
            final_state,
            self.scale,
            work_items=work_items,
            work_count=work_count,
            scheduler_counter=scheduler_counter,
            checkpoint_every_n_tokens=self.checkpoint,
            output_state_checkpoints=state_checkpoints,
            log_gate=self.log_gate,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            work_item_scratch=item_scratch,
            **factor,
            expand_num=self.expand_num,
            workspace=tensormaps,
            device=self.device,
            num_sm=self.num_sm,
            stream=stream,
        )
        return None

    def run_chain(self, q, k, v, g, beta, cu, state0, o, final_state, state_checkpoints, a_log, dt_bias, region, stream) -> None:
        """Chain prologue, T pass, fused H and M summaries, fp32 state chain seeded with initial_state, prefill over the
        pieces seeded with X; the checkpoint series keeps its unsplit layout."""
        cu_pieces = region["cu_pieces"]
        work_items, work_count = region["work_items"], region["main_count"]
        summary_items, summary_count = region["work_items_summary"], region["summary_count"]
        state_h, state_m, state_x = region["state_h"], region["state_m"], region["state_x"]
        tinv = region["tinv"]
        gate_params = dict(a_log=a_log if self.safe_gate else None, dt_bias=dt_bias if self.safe_gate else None)
        warm = self.kernel_cache is not None
        self.run_chain_prologue(
            self.chain_prologue,
            pieces=self.pieces,
            unit_chunks=self.unit_chunks,
            b_t=self.b_t,
            expand_num=self.expand_num,
            length_rule=self.length_rule,
            heads_out=self.n_heads_out,
            checkpoint_every_n_tokens=self.checkpoint,
            cu_seqlens=cu,
            cu_pieces=cu_pieces,
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=work_count,
            summary_count=summary_count,
            work_items=work_items,
            work_items_summary=summary_items,
            scheduler=region["scheduler_all"],
            tinv_words=region["tinv_tensormaps"],
            tinv=tinv,
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            summary_words=region.get("fused_tensormaps"),
            prefill_words=region["tensormaps"],
            q=q,
            k=k,
            v=v,
            o=o,
            checkpoints=state_checkpoints,
            stream=stream,
        )
        self.tinv_cache = chunk_factor_pass(
            self.tinv_cache,
            k,
            g,
            beta,
            cu_pieces,
            tinv,
            region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            log_gate=self.log_gate,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            expand_num=self.expand_num,
            device=self.device,
            stream=stream,
            publish_desc=False,
        )
        if warm:
            self.fused_summary.run_summary(
                self.fused_cache,
                k,
                v,
                g,
                cu_pieces,
                tinv,
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
                **gate_params,
            )
        else:
            self.fused_cache = self.fused_summary.chunk_gdn_summary_sm100(
                k,
                v,
                g,
                cu_pieces,
                tinv,
                None,
                state_h,
                state_m,
                work_items=summary_items,
                work_count=summary_count,
                scheduler_counter=region["scheduler_h"],
                scheduler_all=region["scheduler_all"],
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                expand_num=self.expand_num,
                workspace=region["fused_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
            )
        if not warm:
            self.chain_forward = self.build_state_chain(
                heads_out=self.n_heads_out,
                dim_v=self.dim_v,
                dim_k=self.dim_k,
                pieces=self.pieces,
                rows_per_cta=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
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
            self.kernel.run_prefill(
                self.kernel_cache,
                q,
                k,
                v,
                g,
                o,
                cu_pieces,
                state_x,
                final_state,
                state_checkpoints,
                work_items,
                work_count,
                region["scheduler_prefill"],
                None,
                region["tensormaps"],
                self.checkpoint,
                self.scale,
                stream,
                tinv=tinv,
                own_prologue=False,
                **gate_params,
            )
        else:
            self.kernel_cache = self.kernel.chunk_gdn_sm100(
                q,
                k,
                v,
                g,
                o,
                cu_pieces,
                state_x,
                final_state,
                self.scale,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=region["scheduler_prefill"],
                checkpoint_every_n_tokens=self.checkpoint,
                output_state_checkpoints=state_checkpoints,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                tinv=tinv,
                expand_num=self.expand_num,
                workspace=region["tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
            )


class CompiledGdnBwd:
    """Compiled FROST GDN / GDP bprop plan over the resolved node buffers: dQ/dK/dV/dG/dBeta from the forward checkpoint
    series, regenerated by the recompute kernel when the port is absent.  ``chain`` runs the pieces as independent
    sequences: H, M, X from the fused summary and the forward chain (M alone when the series is passed back), G from the
    bprop summary, a reverse fp32 chain seeding every piece's outgoing gradient, the gather returning piece 0's
    d_initial_state.  The GDP bprop summary reads compact q / dO at every d_v; only the d_v = 128 main bprop reads the
    expanded pack."""

    def __init__(self, node, bwd_module, recompute_module):
        from .common.piece_chain import build_state_chain, chain_rows_per_cta, choose_pieces, piece_table_layout, run_state_chain
        from .kernel.gdn_chain_prologue_f16 import run_chain_prologue
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
        self.tinv_cache = None
        self.recompute_cache = None
        self.run_chain_prologue = run_chain_prologue
        self.chain_prologue = {}
        self.state_m_cache = None
        self.state_g_cache = None
        self.chain_forward = None
        self.chain_reverse = None
        self.plan_name = "GdpFrostEngine (GDP_BWD)" if node.node_type.name == "GDP_BWD" else "GdnFrostEngine (GDN_BWD)"
        self.device = current_device()
        from .common.gate_bwd import scalar_gate_bwd, scalar_gate_blocks
        from .common.head_reduce import head_group_reduce
        from .common.host import tensormap_workspace_bytes
        from .common.l2norm import build_l2norm_qk, run_l2norm_qk

        self.head_group_reduce = head_group_reduce
        self.scalar_gate_bwd = scalar_gate_bwd
        self.build_l2norm_qk = build_l2norm_qk
        self.run_l2norm_qk = run_l2norm_qk
        self.l2norm = None
        self.num_householder = int(node.params.get("num_householder", 1) or 1)
        self.expand_num = self.num_householder
        self.compact_qdo = self.num_householder > 1 and int(node.inputs["v"].dim[-1]) == 64
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.log_gate = (node.params.get("gate_domain") or "log") == "log"
        self.wants_d_a_log = "d_a_log" in node.outputs
        self.wants_d_dt_bias = "d_dt_bias" in node.outputs
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))
        self.allow_neg_eigval = bool(node.params.get("allow_neg_eigval", False))

        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self.b_t = bwd_module.CFG.B_T
        total = node.inputs["k"].dim[0]
        self.q_rows = total // self.expand_num
        self.gate_bwd_blocks = scalar_gate_blocks(self.q_rows)
        K, V = q.dim[-1], v.dim[-1]
        HQ, HV = q.dim[1], v.dim[1]
        HK = node.inputs["k"].dim[1]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.has_state_checkpoints = "state_checkpoints" in node.inputs
        self.checkpoint_cadence = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.coarse_checkpoints = self.has_state_checkpoints and self.checkpoint_cadence > self.b_t
        self.needs_recompute = not self.has_state_checkpoints or self.coarse_checkpoints
        self.io_name = "float16" if node.inputs["q"].get_data_type().name == "HALF" else "bfloat16"
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"

        self.num_sm = multiprocessor_count(self.device)
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.num_seqs = B
        self.n_heads_out, self.total = HO, total
        self.dim_v, self.dim_k = V, K
        self.pieces, self.unit_chunks = choose_pieces(
            num_seqs=B,
            heads_out=HO,
            num_sm=self.num_sm,
            total_tokens=self.q_rows,
            b_t=self.b_t,
            cadence_tokens=self.checkpoint_cadence,
            batch_invariant=self.batch_invariant,
            expand_num=self.expand_num,
            reverse=True,
        )
        self.chain = self.pieces > 0
        self.split = not self.chain and not self.batch_invariant
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        self.fused_h_m = self.chain and not self.has_state_checkpoints
        self.pack_bwd_needed = self.num_householder > 1 and not self.compact_qdo
        self.summary_q_step = self.num_householder if self.pack_bwd_needed and self.use_qk_l2norm else 1
        self.tinv_pass = self.chain or self.needs_recompute
        if self.pack_bwd_needed:
            from .common.expand import build_gather_dq, build_pack_bwd, run_gather_dq, run_pack_bwd

            self.build_pack_bwd = build_pack_bwd
            self.run_pack_bwd = run_pack_bwd
            self.build_gather_dq = build_gather_dq
            self.run_gather_dq = run_gather_dq
            self.pack_bwd = None
            self.gather_dq = None

        layout = WorkspaceLayout()
        regions = []
        off_scheduler = layout.add(40 if self.chain else 16)
        regions += [
            ("scheduler_recompute", off_scheduler, "int32", (2,)),
            ("scheduler_bwd", off_scheduler + 8, "int32", (2,)),
        ]
        if self.chain:
            from .kernel import gdn_bprop_summary_f16 as summary_module

            self.summary = summary_module
            self.summary_tensormap_words = tensormap_workspace_bytes(summary_module, self.num_pieces) // 8
            regions += [
                ("scheduler_summary", off_scheduler + 16, "int32", (2,)),
                ("scheduler_m", off_scheduler + 24, "int32", (2,)),
                ("scheduler_series", off_scheduler + 32, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (10,)),
                ("summary_tensormaps", layout.add(self.summary_tensormap_words * 8), "int64", (self.summary_tensormap_words,)),
            ]
        else:
            regions.append(("scheduler_all", off_scheduler, "int32", (4,)))
        self.tensormap_words = tensormap_workspace_bytes(bwd_module, self.num_pieces) // 8
        regions.append(("tensormaps", layout.add(self.tensormap_words * 8), "int64", (self.tensormap_words,)))
        if self.needs_recompute:
            self.state_checkpoints_rows = max(total // self.b_t + self.num_pieces, 1)
            regions.append(
                ("state_checkpoints", layout.add(self.state_checkpoints_rows * HO * K * V * 2), self.io_name, (self.state_checkpoints_rows, HO, V, K))
            )
        self.recompute_tensormap_words = tensormap_workspace_bytes(recompute_module, self.num_pieces) // 8
        if self.needs_recompute and not self.chain:
            regions.append(("recompute_tensormaps", layout.add(self.recompute_tensormap_words * 8), "int64", (self.recompute_tensormap_words,)))
        if self.chain:
            regions.append(("recompute_tensormaps_m", layout.add(self.recompute_tensormap_words * 8), "int64", (self.recompute_tensormap_words,)))
            if self.needs_recompute:
                regions.append(("recompute_tensormaps_series", layout.add(self.recompute_tensormap_words * 8), "int64", (self.recompute_tensormap_words,)))
        if self.fused_h_m:
            from .kernel import gdn_summary_f16 as fused_module

            self.fused_summary = fused_module
            self.fused_cache = None
            self.fused_tensormap_words = tensormap_workspace_bytes(fused_module, self.num_pieces) // 8
            regions.append(("fused_tensormaps", layout.add(self.fused_tensormap_words * 8), "int64", (self.fused_tensormap_words,)))
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
        dq_rows = self.q_rows if self.compact_qdo else total
        if self.fold_dq:
            regions.append(("dq_ho", layout.add(dq_rows * HO * K * 2), self.io_name, (dq_rows, HO, K)))
        if self.fold_dk:
            regions.append(("dk_ho", layout.add(total * HO * K * 2), self.io_name, (total, HO, K)))
        if self.fold_dv:
            regions.append(("dv_ho", layout.add(total * HO * V * 2), self.io_name, (total, HO, V)))
        l2norm_rows = self.q_rows if self.compact_qdo else total
        if self.use_qk_l2norm:
            regions.append(("q_n", layout.add(l2norm_rows * HQ * K * 2), self.io_name, (l2norm_rows, HQ, K)))
            regions.append(("k_n", layout.add(total * HK * K * 2), self.io_name, (total, HK, K)))
            regions.append(("inv_q", layout.add(l2norm_rows * HQ * 4), "float32", (l2norm_rows, HQ)))
            regions.append(("inv_k", layout.add(total * HK * 4), "float32", (total, HK)))
        if self.wants_d_a_log:
            regions.append(("gate_part_a", layout.add(self.gate_bwd_blocks * HO * 4), "float32", (self.gate_bwd_blocks * HO,)))
        if self.wants_d_dt_bias:
            regions.append(("gate_part_dt", layout.add(self.gate_bwd_blocks * HO * 4), "float32", (self.gate_bwd_blocks * HO,)))
        self.pack_expands_q = self.pack_bwd_needed and not self.use_qk_l2norm
        if self.pack_bwd_needed:
            if self.pack_expands_q:
                regions.append(("q_x", layout.add(total * HQ * K * 2), self.io_name, (total, HQ, K)))
            regions.append(("do_x", layout.add(total * HO * V * 2), self.io_name, (total, HO, V)))
            regions.append(("dq_x", layout.add(total * HQ * K * 2), self.io_name, (total, HQ, K)))
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
        if self.chain:
            table = piece_table_layout(B, self.pieces, HO)
            off_piece_table = layout.add(table.nbytes)
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
        if self.tinv_pass:
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(self.q_rows, self.num_pieces, self.expand_num, self.b_t)
            regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, self.num_pieces) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
        self.needs_table = self.split
        self.recompute_orders = self.needs_recompute and not self.coarse_checkpoints
        self.bwd_orders = not self.recompute_orders
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
        dq_node = None
        q_compact, do_compact = q, do
        l2norm_kw = {}
        if self.pack_bwd_needed:
            pack_q = q if self.pack_expands_q else None
            pack_q_x = region["q_x"] if self.pack_expands_q else None
            if self.pack_bwd is None:
                self.pack_bwd = self.build_pack_bwd(pack_q, pack_q_x, do, region["do_x"], self.num_householder, stream)
            else:
                self.run_pack_bwd(self.pack_bwd, pack_q, pack_q_x, do, region["do_x"], stream)
            do = region["do_x"]
            if self.pack_expands_q:
                q = region["q_x"]
            else:
                l2norm_kw = dict(expand_num=self.num_householder, expand_phase=self.num_householder - 1, expand_fill=True)
            dq_node = dq
            dq = region["dq_x"]
        if self.use_qk_l2norm:
            if self.l2norm is None:
                self.l2norm = self.build_l2norm_qk(q, k, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], **l2norm_kw, stream=stream)
            else:
                self.run_l2norm_qk(self.l2norm, q, k, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream)
            q, k = region["q_n"], region["k_n"]
        summary_q = q_compact if self.pack_expands_q else q
        summary_do = do_compact
        dq_out = region["dq_ho"] if self.fold_dq else dq
        dk_out = region["dk_ho"] if self.fold_dk else dk
        dv_out = region["dv_ho"] if self.fold_dv else dv
        tinv = region.get("tinv")
        bwd_tinv = {} if self.compact_qdo else dict(tinv=tinv)

        if self.chain:
            self.run_chain(
                q,
                k,
                v,
                g,
                beta,
                do,
                summary_q,
                summary_do,
                cu,
                state_checkpoints,
                state0,
                dstate_in,
                dq_out,
                dk_out,
                dv_out,
                dg,
                dbeta,
                dstate0,
                a_log,
                dt_bias,
                region,
                stream,
            )
        elif self.kernel_cache is not None and (self.table is not None or not self.needs_table):
            if self.tinv_pass:
                self.tinv_cache = chunk_factor_pass(
                    self.tinv_cache,
                    k,
                    g,
                    beta,
                    cu,
                    tinv,
                    region["tinv_tensormaps"],
                    tinv_rows=region["tinv_rows"],
                    tinv_row_count=region["tinv_row_count"],
                    log_gate=self.log_gate,
                    safe_gate=self.safe_gate,
                    a_log=a_log,
                    dt_bias=dt_bias,
                    use_beta_sigmoid=self.use_beta_sigmoid,
                    allow_neg_eigval=self.allow_neg_eigval,
                    expand_num=self.expand_num,
                    device=self.device,
                    stream=stream,
                )
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
                    expand_num=self.expand_num,
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
                    cu,
                    None if self.coarse_checkpoints else state0,
                    None,
                    checkpoint_series,
                    region["work_items_recompute"] if self.coarse_checkpoints else work_items,
                    region["work_count_recompute"] if self.coarse_checkpoints else work_count,
                    scheduler_recompute,
                    region["scheduler_all"] if (self.recompute_orders or self.coarse_checkpoints) else None,
                    region.get("item_scratch") if self.recompute_orders else None,
                    region["recompute_tensormaps"],
                    self.b_t,
                    stream,
                    a_log=a_log if self.safe_gate else None,
                    dt_bias=dt_bias if self.safe_gate else None,
                    seed_state_checkpoints=state_checkpoints if self.coarse_checkpoints else None,
                    seed_every_n_tokens=self.checkpoint_cadence if self.coarse_checkpoints else 0,
                    seed_span_tokens=self.recompute_span_tokens if self.coarse_checkpoints else 0,
                    tinv=tinv,
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
                dstate0,
                dstate_in,
                work_items,
                work_count,
                scheduler_bwd,
                region["scheduler_all"] if self.bwd_orders else None,
                region.get("item_scratch") if self.bwd_orders else None,
                region["tensormaps"],
                self.scale,
                stream,
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
                inv_q=region["inv_q"] if self.use_qk_l2norm else None,
                inv_k=region["inv_k"] if self.use_qk_l2norm else None,
                **bwd_tinv,
            )
        else:
            if self.tinv_pass:
                self.tinv_cache = chunk_factor_pass(
                    self.tinv_cache,
                    k,
                    g,
                    beta,
                    cu,
                    tinv,
                    region["tinv_tensormaps"],
                    tinv_rows=region["tinv_rows"],
                    tinv_row_count=region["tinv_row_count"],
                    log_gate=self.log_gate,
                    safe_gate=self.safe_gate,
                    a_log=a_log,
                    dt_bias=dt_bias,
                    use_beta_sigmoid=self.use_beta_sigmoid,
                    allow_neg_eigval=self.allow_neg_eigval,
                    expand_num=self.expand_num,
                    device=self.device,
                    stream=stream,
                )
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
                    a_log=a_log if self.safe_gate else None,
                    dt_bias=dt_bias if self.safe_gate else None,
                    scheduler_counter=region["scheduler_all"],
                    split=self.split,
                    expand_num=self.expand_num,
                    stream=stream,
                )

            if self.has_state_checkpoints and not self.coarse_checkpoints:
                checkpoint_series = state_checkpoints
            else:
                checkpoint_series = region["state_checkpoints"]
                self.recompute_cache = self.recompute.chunk_gdn_recompute_sm100(
                    k,
                    v,
                    g,
                    cu,
                    None if self.coarse_checkpoints else state0,
                    None,
                    checkpoint_every_n_tokens=self.b_t,
                    output_state_checkpoints=checkpoint_series,
                    seed_state_checkpoints=state_checkpoints if self.coarse_checkpoints else None,
                    seed_every_n_tokens=self.checkpoint_cadence if self.coarse_checkpoints else 0,
                    seed_span_tokens=self.recompute_span_tokens if self.coarse_checkpoints else 0,
                    safe_gate=self.safe_gate,
                    a_log=a_log if self.safe_gate else None,
                    dt_bias=dt_bias if self.safe_gate else None,
                    work_items=region["work_items_recompute"] if self.coarse_checkpoints else work_items,
                    work_count=region["work_count_recompute"] if self.coarse_checkpoints else work_count,
                    scheduler_counter=scheduler_recompute,
                    scheduler_all=region["scheduler_all"] if (self.recompute_orders or self.coarse_checkpoints) else None,
                    work_item_scratch=region.get("item_scratch") if self.recompute_orders else None,
                    order_in_prologue=self.recompute_orders,
                    log_gate=self.log_gate,
                    tinv=tinv,
                    expand_num=self.expand_num,
                    workspace=region["recompute_tensormaps"],
                    device=self.device,
                    num_sm=self.num_sm,
                    stream=stream,
                )

            self.kernel_cache = self.bwd.chunk_gdn_bwd_sm100(
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
                d_initial_state=dstate0,
                d_final_state=dstate_in,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=scheduler_bwd,
                scheduler_all=region["scheduler_all"] if self.bwd_orders else None,
                work_item_scratch=region.get("item_scratch") if self.bwd_orders else None,
                order_in_prologue=self.bwd_orders,
                log_gate=self.log_gate,
                inv_q=region["inv_q"] if self.use_qk_l2norm else None,
                inv_k=region["inv_k"] if self.use_qk_l2norm else None,
                expand_num=self.expand_num,
                workspace=region["tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                **bwd_tinv,
            )

        if self.safe_gate:
            self.scalar_gate_bwd(dg, g, a_log, dt_bias, d_a_log, d_dt_bias, region.get("gate_part_a"), region.get("gate_part_dt"), stream=stream)
        if self.fold_dq or self.fold_dk or self.fold_dv:
            for src_ho, dst in ((dq_out, dq), (dk_out, dk), (dv_out, dv)):
                if src_ho is not dst:
                    self.head_group_reduce(src_ho, dst, stream=stream)
        if dq_node is not None:
            if self.gather_dq is None:
                self.gather_dq = self.build_gather_dq(dq, dq_node, self.num_householder, stream)
            else:
                self.run_gather_dq(self.gather_dq, dq, dq_node, stream)
        return None

    def run_chain(
        self,
        q,
        k,
        v,
        g,
        beta,
        do,
        summary_q,
        summary_do,
        cu,
        state_checkpoints,
        state0,
        dstate_in,
        dq_out,
        dk_out,
        dv_out,
        dg,
        dbeta,
        dstate0,
        a_log,
        dt_bias,
        region,
        stream,
    ) -> None:
        """Chain prologue, T pass, H and M summaries (M alone when the series is passed back), forward chain X, G summary,
        reverse chain seeded with d_final_state, dense series (the forward's or the seeded recompute), bprop over the
        pieces."""
        cu_pieces = region["cu_pieces"]
        work_items, work_count = region["work_items"], region["main_count"]
        summary_items, summary_count = region["work_items_summary"], region["summary_count"]
        state_m, state_g, state_dx_end = region["state_m"], region["state_g"], region["state_dx_end"]
        tinv = region["tinv"]
        gate_params = dict(a_log=a_log if self.safe_gate else None, dt_bias=dt_bias if self.safe_gate else None)
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
            expand_num=self.expand_num,
            length_rule=self.length_rule,
            heads_out=self.n_heads_out,
            compact_qdo=self.compact_qdo,
            summary_q_step=self.summary_q_step,
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
            tinv_words=region["tinv_tensormaps"],
            tinv=tinv,
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            summary_words=region.get("fused_tensormaps"),
            recompute_m_words=region["recompute_tensormaps_m"],
            series_words=region.get("recompute_tensormaps_series"),
            bprop_summary_words=region["summary_tensormaps"],
            bprop_words=region["tensormaps"],
            q=q,
            k=k,
            v=v,
            do=do,
            checkpoints=checkpoint_series,
            dq=dq_out,
            dk=dk_out,
            dv=dv_out,
            summary_q=summary_q,
            summary_do=summary_do,
            stream=stream,
        )
        self.tinv_cache = chunk_factor_pass(
            self.tinv_cache,
            k,
            g,
            beta,
            cu_pieces,
            tinv,
            region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            log_gate=self.log_gate,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            expand_num=self.expand_num,
            device=self.device,
            stream=stream,
            publish_desc=False,
        )

        summary_common = dict(
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            work_items=summary_items,
            work_count=summary_count,
            scheduler_counter=region["scheduler_recompute"],
            scheduler_all=region["scheduler_all"],
            log_gate=self.log_gate,
            tinv=tinv,
            expand_num=self.expand_num,
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
                cu_pieces,
                tinv,
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
                **gate_params,
            )
        elif self.fused_h_m:
            self.fused_cache = self.fused_summary.chunk_gdn_summary_sm100(
                k,
                v,
                g,
                cu_pieces,
                tinv,
                None,
                region["state_h"],
                state_m,
                work_items=summary_items,
                work_count=summary_count,
                scheduler_counter=region["scheduler_recompute"],
                scheduler_all=region["scheduler_all"],
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                expand_num=self.expand_num,
                workspace=region["fused_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
            )
        else:
            if warm:
                self.recompute.run_recompute(
                    self.state_m_cache,
                    k,
                    k,
                    g,
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
                    **gate_params,
                    tinv=tinv,
                )
            else:
                self.state_m_cache = self.recompute.chunk_gdn_recompute_sm100(
                    k,
                    k,
                    g,
                    cu_pieces,
                    None,
                    state_m,
                    seed_identity=True,
                    v_is_zero=True,
                    workspace=region["recompute_tensormaps_m"],
                    **dict(summary_common, scheduler_counter=region["scheduler_m"]),
                )
        if not self.has_state_checkpoints:
            state_h, state_x = region["state_h"], region["state_x"]
            if not warm:
                self.chain_forward = self.build_state_chain(
                    heads_out=self.n_heads_out,
                    dim_v=self.dim_v,
                    dim_k=self.dim_k,
                    pieces=self.pieces,
                    rows_per_cta=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
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
                summary_q,
                k,
                g,
                summary_do,
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
                tinv=tinv,
                own_prologue=False,
                **gate_params,
            )
        else:
            self.state_g_cache = self.summary.chunk_gdn_bwd_summary_sm100(
                summary_q,
                k,
                g,
                summary_do,
                state_g,
                cu_pieces,
                self.scale,
                d_final_state=None,
                work_items=summary_items,
                work_count=summary_count,
                scheduler_counter=region["scheduler_summary"],
                scheduler_all=region["scheduler_all"],
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                tinv=tinv,
                expand_num=self.expand_num,
                q_step=self.summary_q_step,
                workspace=region["summary_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
            )
            self.chain_reverse = self.build_state_chain(
                heads_out=self.n_heads_out,
                dim_v=self.dim_v,
                dim_k=self.dim_k,
                pieces=self.pieces,
                rows_per_cta=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
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
                    tinv=tinv,
                    own_prologue=False,
                    **gate_params,
                    **seed_kw,
                )
            else:
                self.recompute_cache = self.recompute.chunk_gdn_recompute_sm100(
                    k,
                    v,
                    g,
                    cu_pieces,
                    series_seed,
                    None,
                    checkpoint_every_n_tokens=self.b_t,
                    output_state_checkpoints=checkpoint_series,
                    safe_gate=self.safe_gate,
                    a_log=a_log if self.safe_gate else None,
                    dt_bias=dt_bias if self.safe_gate else None,
                    work_items=series_items,
                    work_count=series_count,
                    scheduler_counter=region["scheduler_series"],
                    scheduler_all=region["scheduler_all"],
                    log_gate=self.log_gate,
                    tinv=tinv,
                    expand_num=self.expand_num,
                    workspace=region["recompute_tensormaps_series"],
                    device=self.device,
                    num_sm=self.num_sm,
                    stream=stream,
                    own_prologue=False,
                    **seed_kw,
                )

        bwd_tinv = {} if self.compact_qdo else dict(tinv=tinv)
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
                region["tensormaps"],
                self.scale,
                stream,
                inv_q=region["inv_q"] if self.use_qk_l2norm else None,
                inv_k=region["inv_k"] if self.use_qk_l2norm else None,
                own_prologue=False,
                **bwd_tinv,
                **gate_params,
            )
        else:
            self.kernel_cache = self.bwd.chunk_gdn_bwd_sm100(
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
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=region["scheduler_bwd"],
                scheduler_all=region["scheduler_all"],
                log_gate=self.log_gate,
                inv_q=region["inv_q"] if self.use_qk_l2norm else None,
                inv_k=region["inv_k"] if self.use_qk_l2norm else None,
                expand_num=self.expand_num,
                workspace=region["tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
                **bwd_tinv,
            )


def build_gdn_summary(graph):
    """Import the summary kernel module (recompute for the forward summary,
    bprop summary for the gradient pass) and wrap the single node."""
    nodes = list(graph.nodes)
    allowed = ("GDN_SUMMARY", "GDP_SUMMARY", "GDN_SUMMARY_BWD", "GDP_SUMMARY_BWD")
    if len(nodes) != 1 or getattr(nodes[0].node_type, "name", None) not in allowed:
        raise ValueError("build_gdn_summary: graph does not contain exactly one GDN_SUMMARY/GDP_SUMMARY/GDN_SUMMARY_BWD/GDP_SUMMARY_BWD node")
    node = nodes[0]
    if node.node_type.name in ("GDN_SUMMARY_BWD", "GDP_SUMMARY_BWD"):
        from .kernel import gdn_bprop_summary_f16 as summary_module

        return CompiledGdnSummaryBwd(node, summary_module)
    from .kernel import gdn_recompute_f16 as recompute_module

    return CompiledGdnSummary(node, recompute_module)


class GdnSummaryFrostEngine(BaseEngine):
    """FROST summary backend for single-node GDN_SUMMARY (final state and span transition through the recompute kernel)
    and GDN_SUMMARY_BWD (d_initial_state through the bprop summary kernel) graphs."""

    name = "gdn_summary_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        facts = graph._facts_for(analyze)
        frost_la_gate("GdnSummaryFrostEngine", facts, "GDN_SUMMARY")
        if facts.d_qk not in (64, 128):
            raise NotImplementedError(f"GdnSummaryFrostEngine: k head dim must be 64 or 128, got {facts.d_qk}")
        if facts.d_v not in (64, 128):
            raise NotImplementedError(f"GdnSummaryFrostEngine: v head dim must be 64 or 128, got {facts.d_v}")
        summary_support_gates("GdnSummaryFrostEngine", facts, graph)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_gdn_summary(graph))


class CompiledGdnSummary:
    """Compiled summary plan over the state-only recompute kernel: the final-state call (initial_state honored) and, when
    ``transition`` is bound, the identity-seeded zero-value call whose buffer holds ``M_buf = M^T`` (``X_final = X_init @
    M_buf + X_H``).  ``chain`` summarizes every filled piece (H, M) and composes them with one fp32 state chain whose tail
    is ``final_state`` and whose running product is ``transition``.  The chunk factor T comes from the T pass."""

    def __init__(self, node, recompute_module):
        from .common.host import tensormap_workspace_bytes
        from .common.l2norm import build_l2norm_qk, run_l2norm_qk
        from .common.piece_chain import build_state_chain, chain_rows_per_cta, choose_pieces, piece_table_layout, run_state_chain
        from .kernel.gdn_chain_prologue_f16 import run_chain_prologue
        from .common.split_k import WORK_ITEM_FIELDS, build_split_table, chunk_scratch_rows, compute_ideal_chunks, max_work_items, run_table

        self.node = node
        self.recompute = recompute_module
        self.build_split_table = build_split_table
        self.run_table = run_table
        self.build_state_chain = build_state_chain
        self.chain_rows_per_cta = chain_rows_per_cta
        self.run_state_chain = run_state_chain
        self.build_l2norm_qk = build_l2norm_qk
        self.run_l2norm_qk = run_l2norm_qk
        self.table = None
        self.final_cache = None
        self.transition_cache = None
        self.l2norm = None
        self.tinv_cache = None
        self.run_chain_prologue = run_chain_prologue
        self.chain_prologue = {}
        self.fused_cache = None
        self.chain_summary = None
        self.plan_name = "GdpSummaryFrostEngine (GDP_SUMMARY)" if node.node_type.name == "GDP_SUMMARY" else "GdnSummaryFrostEngine (GDN_SUMMARY)"
        self.device = current_device()
        self.num_householder = int(node.params.get("num_householder", 1) or 1)
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.log_gate = (node.params.get("gate_domain") or "log") == "log"
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))
        self.allow_neg_eigval = bool(node.params.get("allow_neg_eigval", False))
        self.has_transition = "transition" in node.outputs

        k, g = node.inputs["k"], node.inputs["g"]
        self.b_t = recompute_module.CFG.B_T
        total = k.dim[0]
        HO = g.dim[1]
        HK = k.dim[1]
        K = k.dim[2]
        V = node.inputs["v"].dim[2]
        self.io_name = "float16" if k.get_data_type().name == "HALF" else "bfloat16"
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
            total_tokens=total // self.num_householder,
            b_t=self.b_t,
            cadence_tokens=0,
            batch_invariant=self.batch_invariant,
            compose_tail=True,
            expand_num=self.num_householder,
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
            from .kernel import gdn_summary_f16 as summary_module

            self.fused_summary = summary_module
            words = tensormap_workspace_bytes(summary_module, self.num_pieces) // 8
            regions.append(("fused_tensormaps", layout.add(words * 8, align=128), "int64", (words,)))
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(total // self.num_householder, self.num_pieces, self.num_householder, self.b_t)
            regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, self.num_pieces) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
        else:
            off_scheduler = layout.add(16)
            tensormap_words = tensormap_workspace_bytes(recompute_module, B) // 8
            regions += [
                ("scheduler_final", off_scheduler, "int32", (2,)),
                ("scheduler_transition", off_scheduler + 8, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (4,)),
                ("tensormaps", layout.add(tensormap_words * 8), "int64", (tensormap_words,)),
            ]
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(total // self.num_householder, B, self.num_householder, self.b_t)
            regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, B) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
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
        if self.use_qk_l2norm:
            regions.append(("q_n", layout.add(total * HK * K * 2), self.io_name, (total, HK, K)))
            regions.append(("k_n", layout.add(total * HK * K * 2), self.io_name, (total, HK, K)))
            regions.append(("inv_q", layout.add(total * HK * 4), "float32", (total, HK)))
            regions.append(("inv_k", layout.add(total * HK * 4), "float32", (total, HK)))
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
        if self.use_qk_l2norm:
            # the recompute takes no q; the k rows ride the q slots of the fused pass
            if self.l2norm is None:
                self.l2norm = self.build_l2norm_qk(k, k, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream=stream)
            else:
                self.run_l2norm_qk(self.l2norm, k, k, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream)
            k = region["k_n"]
        if self.chain:
            self.run_chain(k, v, g, beta, cu, state0, final_state, transition, a_log, dt_bias, region, stream)
            return
        tinv = region["tinv"]
        self.tinv_cache = chunk_factor_pass(
            self.tinv_cache,
            k,
            g,
            beta,
            cu,
            tinv,
            region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            log_gate=self.log_gate,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            expand_num=self.num_householder,
            device=self.device,
            stream=stream,
        )
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
                    expand_num=self.num_householder,
                )
            self.recompute.run_recompute(
                self.final_cache,
                k,
                v,
                g,
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
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
                tinv=tinv,
            )
            if self.has_transition:
                self.recompute.run_recompute(
                    self.transition_cache,
                    k,
                    k,
                    g,
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
                    a_log=a_log if self.safe_gate else None,
                    dt_bias=dt_bias if self.safe_gate else None,
                    tinv=tinv,
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
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
                scheduler_counter=region["scheduler_all"],
                split=self.split,
                expand_num=self.num_householder,
                stream=stream,
            )

        self.final_cache = self.recompute.chunk_gdn_recompute_sm100(
            k,
            v,
            g,
            cu,
            state0,
            final_state,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            work_items=work_items,
            work_count=work_count,
            scheduler_counter=region["scheduler_final"],
            scheduler_all=region["scheduler_all"],
            work_item_scratch=item_scratch,
            order_in_prologue=True,
            log_gate=self.log_gate,
            expand_num=self.num_householder,
            workspace=region["tensormaps"],
            device=self.device,
            num_sm=self.num_sm,
            stream=stream,
            tinv=tinv,
        )
        if self.has_transition:
            self.transition_cache = self.recompute.chunk_gdn_recompute_sm100(
                k,
                k,
                g,
                cu,
                None,
                transition,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                seed_identity=True,
                v_is_zero=True,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=region["scheduler_transition"],
                scheduler_all=region["scheduler_all"],
                work_item_scratch=item_scratch,
                order_in_prologue=True,
                log_gate=self.log_gate,
                expand_num=self.num_householder,
                workspace=region["tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                tinv=tinv,
            )
        return None

    def run_chain(self, k, v, g, beta, cu, state0, final_state, transition, a_log, dt_bias, region, stream) -> None:
        """Chain prologue, T pass, fused H and M summaries of every filled piece, one fp32 state chain seeded with
        initial_state whose tail is final_state and whose running product is transition."""
        cu_pieces = region["cu_pieces"]
        work_items, work_count = region["work_items"], region["main_count"]
        state_h, state_m, state_x = region["state_h"], region["state_m"], region["state_x"]
        tinv = region["tinv"]
        gate_params = dict(a_log=a_log if self.safe_gate else None, dt_bias=dt_bias if self.safe_gate else None)
        warm = self.chain_summary is not None
        self.run_chain_prologue(
            self.chain_prologue,
            pieces=self.pieces,
            unit_chunks=self.unit_chunks,
            b_t=self.b_t,
            expand_num=self.num_householder,
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
            tinv_words=region["tinv_tensormaps"],
            tinv=tinv,
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            summary_words=region.get("fused_tensormaps"),
            k=k,
            v=v,
            stream=stream,
        )
        self.tinv_cache = chunk_factor_pass(
            self.tinv_cache,
            k,
            g,
            beta,
            cu_pieces,
            tinv,
            region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            log_gate=self.log_gate,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            expand_num=self.num_householder,
            device=self.device,
            stream=stream,
            publish_desc=False,
        )
        if warm:
            self.fused_summary.run_summary(
                self.fused_cache,
                k,
                v,
                g,
                cu_pieces,
                tinv,
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
                **gate_params,
            )
        else:
            self.fused_cache = self.fused_summary.chunk_gdn_summary_sm100(
                k,
                v,
                g,
                cu_pieces,
                tinv,
                None,
                state_h,
                state_m,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=region["scheduler_h"],
                scheduler_all=region["scheduler_all"],
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                expand_num=self.num_householder,
                workspace=region["fused_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
            )
        if not warm:
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


class CompiledGdnSummaryBwd:
    """Compiled gradient summary over the bprop summary kernel: the reverse state-gradient recurrence seeded by
    ``d_final_state`` (zero when unbound) writes ``d_initial_state``; a bound ``transition`` receives ``M^T`` from the
    recompute's transition arm.  ``chain`` summarizes every filled piece (G, M) and composes them with one reverse fp32
    state chain whose tail is ``d_initial_state`` and whose running product is ``transition``."""

    def __init__(self, node, summary_module):
        from .common.host import tensormap_workspace_bytes
        from .common.l2norm import build_l2norm_qk, run_l2norm_qk
        from .common.piece_chain import build_state_chain, chain_rows_per_cta, choose_pieces, piece_table_layout, run_state_chain
        from .kernel.gdn_chain_prologue_f16 import run_chain_prologue
        from .common.split_k import WORK_ITEM_FIELDS, build_split_table, chunk_scratch_rows, compute_ideal_chunks, max_work_items, run_table

        self.node = node
        self.summary = summary_module
        self.build_split_table = build_split_table
        self.run_table = run_table
        self.build_state_chain = build_state_chain
        self.chain_rows_per_cta = chain_rows_per_cta
        self.run_state_chain = run_state_chain
        self.build_l2norm_qk = build_l2norm_qk
        self.run_l2norm_qk = run_l2norm_qk
        self.table = None
        self.kernel_cache = None
        self.l2norm = None
        self.tinv_cache = None
        self.run_chain_prologue = run_chain_prologue
        self.chain_prologue = {}
        self.state_g_cache = None
        self.state_m_cache = None
        self.chain_reverse = None
        self.transition_cache = None
        self.transition_chain = None
        self.plan_name = "GdpSummaryFrostEngine (GDP_SUMMARY_BWD)" if node.node_type.name == "GDP_SUMMARY_BWD" else "GdnSummaryFrostEngine (GDN_SUMMARY_BWD)"
        self.device = current_device()
        self.num_householder = int(node.params.get("num_householder", 1) or 1)
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.log_gate = (node.params.get("gate_domain") or "log") == "log"
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))
        self.allow_neg_eigval = bool(node.params.get("allow_neg_eigval", False))
        self.has_transition = "transition" in node.outputs

        q, do, g = node.inputs["q"], node.inputs["dO"], node.inputs["g"]
        self.b_t = summary_module.CFG.B_T
        total = node.inputs["k"].dim[0]
        K, V = q.dim[-1], do.dim[-1]
        HQ, HK = q.dim[1], node.inputs["k"].dim[1]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.io_name = "float16" if q.get_data_type().name == "HALF" else "bfloat16"
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"

        self.num_sm = multiprocessor_count(self.device)
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.n_heads_out, self.total = HO, total
        self.num_seqs = B
        self.dim_k, self.dim_v = K, V
        self.pieces, self.unit_chunks = choose_pieces(
            num_seqs=B,
            heads_out=HO,
            num_sm=self.num_sm,
            total_tokens=total // self.num_householder,
            b_t=self.b_t,
            cadence_tokens=0,
            batch_invariant=self.batch_invariant,
            compose_tail=True,
            expand_num=self.num_householder,
            reverse=True,
        )
        self.chain = self.pieces > 0
        self.split = not self.chain and not self.batch_invariant
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        layout = WorkspaceLayout()
        regions = []
        if self.chain:
            from .kernel import gdn_recompute_f16 as recompute_module

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
            summary_words = tensormap_workspace_bytes(summary_module, self.num_pieces) // 8
            regions.append(("summary_tensormaps", layout.add(summary_words * 8, align=128), "int64", (summary_words,)))
            recompute_words = tensormap_workspace_bytes(recompute_module, self.num_pieces) // 8
            regions.append(("recompute_tensormaps_m", layout.add(recompute_words * 8, align=128), "int64", (recompute_words,)))
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(total // self.num_householder, self.num_pieces, self.num_householder, self.b_t)
            regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, self.num_pieces) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
        else:
            off_scheduler = layout.add(16)
            tensormap_words = tensormap_workspace_bytes(summary_module, B) // 8
            regions += [
                ("scheduler_main", off_scheduler, "int32", (2,)),
                ("scheduler_transition", off_scheduler + 8, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (4,)),
                ("tensormaps", layout.add(tensormap_words * 8), "int64", (tensormap_words,)),
            ]
            if self.has_transition:
                from .kernel import gdn_recompute_f16 as recompute_module

                self.recompute = recompute_module
                recompute_words = tensormap_workspace_bytes(recompute_module, B) // 8
                regions.append(("recompute_tensormaps", layout.add(recompute_words * 8, align=128), "int64", (recompute_words,)))
                regions.append(("transition_m", layout.add(B * HO * K * K * 4), "float32", (B, HO, K, K)))
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(total // self.num_householder, B, self.num_householder, self.b_t)
            regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, B) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
        if self.use_qk_l2norm:
            q_rows = total // self.num_householder
            regions.append(("q_n", layout.add(q_rows * HQ * K * 2), self.io_name, (q_rows, HQ, K)))
            regions.append(("k_n", layout.add(total * HK * K * 2), self.io_name, (total, HK, K)))
            regions.append(("inv_q", layout.add(q_rows * HQ * 4), "float32", (q_rows, HQ)))
            regions.append(("inv_k", layout.add(total * HK * 4), "float32", (total, HK)))
        if not self.chain:
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
        if self.use_qk_l2norm:
            if self.l2norm is None:
                self.l2norm = self.build_l2norm_qk(q, k, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream=stream)
            else:
                self.run_l2norm_qk(self.l2norm, q, k, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream)
            q, k = region["q_n"], region["k_n"]
        if self.chain:
            self.run_chain(q, k, g, beta, do, cu, dstate_in, dstate0, transition, a_log, dt_bias, region, stream)
            return
        tinv = region["tinv"]
        self.tinv_cache = chunk_factor_pass(
            self.tinv_cache,
            k,
            g,
            beta,
            cu,
            tinv,
            region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            log_gate=self.log_gate,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            expand_num=self.num_householder,
            device=self.device,
            stream=stream,
        )
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
                    expand_num=self.num_householder,
                )
            self.summary.run_bwd_summary(
                self.kernel_cache,
                q,
                k,
                g,
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
                tinv=tinv,
            )
            if self.has_transition:
                self.run_transition(k, g, cu, transition, a_log, dt_bias, tinv, region, stream)
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
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
                scheduler_counter=region["scheduler_all"],
                split=self.split,
                expand_num=self.num_householder,
                stream=stream,
            )

        self.kernel_cache = self.summary.chunk_gdn_bwd_summary_sm100(
            q,
            k,
            g,
            do,
            dstate0,
            cu,
            self.scale,
            d_final_state=dstate_in,
            work_items=work_items,
            work_count=work_count,
            scheduler_counter=region["scheduler_main"],
            scheduler_all=region["scheduler_all"],
            work_item_scratch=region.get("item_scratch"),
            order_in_prologue=True,
            log_gate=self.log_gate,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            expand_num=self.num_householder,
            workspace=region["tensormaps"],
            device=self.device,
            num_sm=self.num_sm,
            stream=stream,
            tinv=tinv,
        )
        if self.has_transition:
            self.run_transition(k, g, cu, transition, a_log, dt_bias, tinv, region, stream)
        return None

    def run_transition(self, k, g, cu, transition, a_log, dt_bias, tinv, region, stream) -> None:
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
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
                tinv=tinv,
            )
        else:
            self.transition_cache = self.recompute.chunk_gdn_recompute_sm100(
                k,
                k,
                g,
                cu,
                None,
                transition_m,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                seed_identity=True,
                v_is_zero=True,
                work_items=region["work_items"],
                work_count=region["work_count"],
                scheduler_counter=region["scheduler_transition"],
                scheduler_all=region["scheduler_all"],
                work_item_scratch=region.get("item_scratch"),
                order_in_prologue=True,
                log_gate=self.log_gate,
                expand_num=self.num_householder,
                workspace=region["recompute_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                tinv=tinv,
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
        """Chain prologue, T pass, M (recompute transition arm) and G (zero-seeded bprop summary) of every filled piece, one
        reverse fp32 state chain seeded with d_final_state whose tail is d_initial_state and whose running product is
        transition."""
        cu_pieces = region["cu_pieces"]
        work_items, work_count = region["work_items"], region["main_count"]
        state_g, state_m, state_x = region["state_g"], region["state_m"], region["state_x"]
        tinv = region["tinv"]
        gate_params = dict(a_log=a_log if self.safe_gate else None, dt_bias=dt_bias if self.safe_gate else None)
        warm = self.chain_reverse is not None
        self.run_chain_prologue(
            self.chain_prologue,
            pieces=self.pieces,
            unit_chunks=self.unit_chunks,
            b_t=self.b_t,
            expand_num=self.num_householder,
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
            tinv_words=region["tinv_tensormaps"],
            tinv=tinv,
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            recompute_m_words=region["recompute_tensormaps_m"],
            bprop_summary_words=region["summary_tensormaps"],
            k=k,
            summary_q=q,
            summary_do=do,
            stream=stream,
        )
        self.tinv_cache = chunk_factor_pass(
            self.tinv_cache,
            k,
            g,
            beta,
            cu_pieces,
            tinv,
            region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            log_gate=self.log_gate,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            expand_num=self.num_householder,
            device=self.device,
            stream=stream,
            publish_desc=False,
        )
        if warm:
            self.recompute.run_recompute(
                self.state_m_cache,
                k,
                k,
                g,
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
                **gate_params,
                tinv=tinv,
            )
            self.summary.run_bwd_summary(
                self.state_g_cache,
                q,
                k,
                g,
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
                tinv=tinv,
                own_prologue=False,
                **gate_params,
            )
        else:
            self.state_m_cache = self.recompute.chunk_gdn_recompute_sm100(
                k,
                k,
                g,
                cu_pieces,
                None,
                state_m,
                seed_identity=True,
                v_is_zero=True,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=region["scheduler_m"],
                scheduler_all=region["scheduler_all"],
                log_gate=self.log_gate,
                expand_num=self.num_householder,
                workspace=region["recompute_tensormaps_m"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
                tinv=tinv,
            )
            self.state_g_cache = self.summary.chunk_gdn_bwd_summary_sm100(
                q,
                k,
                g,
                do,
                state_g,
                cu_pieces,
                self.scale,
                d_final_state=None,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=region["scheduler_g"],
                scheduler_all=region["scheduler_all"],
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                tinv=tinv,
                expand_num=self.num_householder,
                workspace=region["summary_tensormaps"],
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
                own_prologue=False,
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
