# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDN engine: GDN / GDN_BWD nodes on the chunked prefill and backward kernels (SM100 / SM103 / SM107, bf16/fp16); the
backward regenerates the checkpoint series with the recompute kernel when the graph carries none.  Long sequences on few
(sequence, head) tiles run as an exact piece chain (``common/piece_chain.py``) or as the decay-warmup split-K.
GDP/GDP_BWD nodes run on the ``num_householder``-expanded timeline."""

from __future__ import annotations

import math
from operator import itemgetter

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost.device import build_device, current_device, multiprocessor_count
from cudnn.frost.workspace import WorkspaceLayout, carve_plan
from ..graph_analyzer import analyze
from .common.context import backward_context, forward_apply_context, forward_context
from .engine import FrostLaPlan, context_gate, frost_la_gate, summary_support_gates


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

    if facts.is_bwd and facts.input_context and not facts.context_from_forward:
        context_gate(engine, facts, b_t=64, expand_num=facts.num_householder)
    checkpoint = facts.checkpoint_every_n_tokens
    checkpoint_dtypes = (cudnn.data_type.FLOAT, facts.io_dtype)
    if not facts.is_bwd:
        checkpoint_dtypes = checkpoint_dtypes + (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2)
    if checkpoint and checkpoint % 64 != 0:
        raise NotImplementedError(f"{engine}: checkpoint_every_n_tokens must be a positive multiple of 64 (got {checkpoint})")
    if not facts.gates_at_ho:
        raise NotImplementedError(f"{engine}: g/beta must carry HO = max(q, v) heads ({facts.h_o})")
    if facts.has_state_indices:
        if facts.state_indices_dtype not in (cudnn.data_type.INT32, cudnn.data_type.INT64):
            raise NotImplementedError(f"{engine}: 'state_indices' must be int32 or int64, got {facts.state_indices_dtype}")
        if facts.is_bwd:
            raise NotImplementedError(f"{engine}: 'state_indices' is a forward-only pool addressing mode")
    io = (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
    state_dtypes = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
    beta_wants = (cudnn.data_type.FLOAT, facts.io_dtype)
    if facts.beta_dtype not in beta_wants + (None,):
        raise NotImplementedError(f"{engine}: 'beta' must be {' or '.join(str(w) for w in beta_wants)}, got {facts.beta_dtype}")
    gate_param_dtypes = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
    for port, got in (("a_log", facts.a_log_dtype), ("dt_bias", facts.dt_bias_dtype)):
        if got not in gate_param_dtypes + (None,):
            raise NotImplementedError(f"{engine}: '{port}' must be fp32/bf16/fp16, got {got}")
    fp8 = (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2)
    forward_state_dtypes = state_dtypes if facts.is_bwd else state_dtypes + fp8
    for port, got in (("initial_state", facts.state_dtype), ("final_state", facts.final_state_dtype)):
        if got not in forward_state_dtypes + (None,):
            raise NotImplementedError(f"{engine}: '{port}' must be fp32/bf16/fp16 (fp8 in a forward graph), got {got}")
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
        if facts.io_dtype is not None and facts.state_checkpoints_dtype not in checkpoint_dtypes + (None,):
            raise NotImplementedError(
                f"{engine}: 'state_checkpoints' must be {' or '.join(str(w) for w in checkpoint_dtypes)}, got {facts.state_checkpoints_dtype}"
            )
        if facts.dbeta_dtype not in (facts.beta_dtype, None):
            raise NotImplementedError(f"{engine}: 'dBeta' must match 'beta' ({facts.beta_dtype}), got {facts.dbeta_dtype}")
    elif facts.io_dtype is not None and facts.state_checkpoints_out_dtype not in checkpoint_dtypes + (None,):
        raise NotImplementedError(
            f"{engine}: 'state_checkpoints' must be {' or '.join(str(w) for w in checkpoint_dtypes)}, got {facts.state_checkpoints_out_dtype}"
        )


class GdnFrostEngine(BaseEngine):
    """FROST chunked-kernel backend for single-node GDN graphs (THD layout); the default GDN engine on SM100 / SM103 / SM107,
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


class CompiledGdn:
    """Compiled FROST GDN / GDP plan over the resolved node buffers.  ``choose_pieces`` and ``is_dv_split`` fix the scheme at build: ``uncut``
    (one item per sequence and head), ``warmup`` (decay-warmup split-K), ``dv_split`` (two uncut items per sequence and head,
    each owning half of d_v) or ``chain`` (per-piece H and M from the fused summary, an fp32 state chain seeding every piece,
    the prefill over the pieces as independent sequences).  The chunk factor T comes from the T pass in chain
    plans."""

    def __init__(self, node, kernel_module):
        from .common.piece_chain import (
            DV_SPLIT_TILES,
            chain_rows_per_cta,
            choose_pieces,
            is_dv_split,
            piece_table_layout,
        )
        from cuda.bindings.driver import CUstream

        from .kernel.gdn_chain_forward_f16 import CHAIN_FORWARD_BUFFERS, build_chain_forward, run_chain_forward
        from .kernel.gdn_warmup_forward_f16 import TAIL_BUFFERS, build_warmup_forward, run_warmup_forward
        from .common.split_k import WORK_ITEM_FIELDS, compute_ideal_chunks, max_work_items, split_scratch_rows
        from .common.tail import GATE_WORDS, choose_tail

        self.node = node
        self.chain_rows_per_cta = chain_rows_per_cta
        self.warmup_launch = None
        self.tail_buffer_names = TAIL_BUFFERS
        self.build_warmup_forward = build_warmup_forward
        self.run_warmup_forward = run_warmup_forward
        self.build_chain_forward = build_chain_forward
        self.run_chain_forward = run_chain_forward
        self.chain_launch = None
        self.chain_buffer_names = CHAIN_FORWARD_BUFFERS
        self.stream_type = CUstream
        self.plan_name = "GdpFrostEngine (GDP)" if node.node_type.name == "GDP" else "GdnFrostEngine (GDN)"
        self.device = current_device()
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
        k_dim = node.inputs["k"].dim
        total = k_dim[0] * k_dim[1] if len(k_dim) == 4 else k_dim[0]
        HK = k_dim[-2]
        HO = g.dim[1]
        HQ = q.dim[1]
        K = q.dim[2]
        V = v.dim[-1]
        self.io_name = "float16" if q.get_data_type().name == "HALF" else "bfloat16"
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.has_final_state = "final_state" in node.outputs
        self.checkpoint = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.checkpoint_after_block = bool(node.params.get("checkpoint_after_block", False))
        self.has_state_checkpoints = "state_checkpoints" in node.outputs
        self.has_context = bool(node.params.get("input_context", False))
        self.context = (
            forward_context(self.plan_name, total=total, h_k=HK, d_k=K, io_name=self.io_name, use_qk_l2norm=self.use_qk_l2norm) if self.has_context else None
        )
        self.context_size = self.context.size if self.has_context else 0
        self.has_k_normalized = self.has_context
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.overwrite_initial_state = bool(node.params.get("overwrite_initial_state", False))
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
        self.dv_split = (
            not self.chain
            and not self.batch_invariant
            and is_dv_split(
                num_seqs=B,
                heads_out=HO,
                dim_v=V,
                num_sm=self.num_sm,
                total_tokens=self.q_rows,
                b_t=self.b_t,
                expand_num=self.expand_num,
                unit_chunks=self.unit_chunks,
            )
        )
        self.split = not self.chain and not self.dv_split and not self.batch_invariant and not self.overwrite_initial_state
        self.tiles_per_head = DV_SPLIT_TILES if self.dv_split else 1
        self.tail = None
        if not self.chain and not self.dv_split and not self.batch_invariant:
            self.tail = choose_tail(
                num_seqs=B,
                heads_out=HO,
                num_sm=self.num_sm,
                total_tokens=self.q_rows,
                b_t=self.b_t,
                cadence_tokens=self.checkpoint,
                expand_num=self.expand_num,
                dim_v=V,
            )
            if self.tail is not None and self.tail.batch_base == 0 and ("preceding_state_update" in node.inputs or "composed_initial_state" in node.inputs):
                self.tail = None
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        self.tinv_pass = self.chain
        self.output_context = bool(node.params.get("output_context", False))
        if self.output_context:
            from .kernel import gdn_tinv_f16 as tinv_module

            self.context = forward_apply_context(
                self.plan_name,
                total=total,
                num_householder=self.num_householder,
                num_seqs=B,
                h_q=HQ,
                h_k=HK,
                h_o=HO,
                d_k=K,
                d_v=V,
                b_t=self.b_t,
                io_name=self.io_name,
                num_sm=self.num_sm,
                cadence_tokens=self.checkpoint,
                batch_invariant=self.batch_invariant,
                use_qk_l2norm=self.use_qk_l2norm,
                tinv_rows=tinv_module.tinv_rows,
            )
            self.context_size = self.context.size
            self.has_context = self.context_size > 0
        self.output_normalized_rows = self.output_context and self.context.has_rows
        self.output_chunk_factors = self.output_context and self.context.has_tables

        layout = WorkspaceLayout()
        from .common.host import tensormap_workspace_bytes

        regions = []
        if self.chain:
            self.n_tiles = self.num_pieces * HO
            self.work_item_rows = self.n_tiles
            self.ideal = None
            self.tensormap_words = tensormap_workspace_bytes(kernel_module, self.num_pieces, self.num_householder) // 8
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

            self.fused_tensormap_words = tensormap_workspace_bytes(summary_module, self.num_pieces, self.num_householder) // 8
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
            self.tensormap_words = tensormap_workspace_bytes(kernel_module, B, self.num_householder) // 8
            regions.append(("tensormaps", layout.add(self.tensormap_words * 8), "int64", (self.tensormap_words,)))
            regions.append(("scheduler", layout.add(8), "int32", (2,)))
            self.n_tiles = B * HO * self.tiles_per_head
            if self.split:
                self.ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
                self.work_item_rows = max_work_items(total, B, HO, self.ideal, self.b_t, self.num_sm)
            else:
                self.ideal = None
                self.work_item_rows = self.n_tiles
            regions.append(("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
            regions.append(("work_count", layout.add(4), "int32", (1,)))
            if self.tail is not None:
                regions.append(("gate", layout.add(4 * GATE_WORDS, align=16), "int32", (GATE_WORDS,)))
            if self.split:
                self.chunk_scratch_rows = split_scratch_rows(total, B, HO, self.b_t)
                regions.append(("item_scratch", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
                regions.append(("chunk_scratch", layout.add(self.chunk_scratch_rows * HO * 4), "float32", (self.chunk_scratch_rows, HO)))
        if self.tail is not None and self.tail.dv_split:
            tail = self.tail
            self.tail_ctas = tail.tiles * DV_SPLIT_TILES
            self.tail_item_base = tail.prefix_tiles * DV_SPLIT_TILES
            tail_words = tensormap_workspace_bytes(kernel_module, B, self.num_householder) // 8
            regions += [
                ("tail_tensormaps", layout.add(tail_words * 8), "int64", (tail_words,)),
                ("tail_scheduler", layout.add(8), "int32", (2,)),
                ("tail_work_items", layout.add(self.tail_ctas * WORK_ITEM_FIELDS * 4), "int32", (self.tail_ctas, WORK_ITEM_FIELDS)),
                ("tail_work_count", layout.add(4), "int32", (1,)),
            ]
        elif self.tail is not None:
            from .kernel import gdn_summary_f16 as tail_summary_module
            from .kernel import gdn_tinv_f16 as tail_tinv_module

            tail = self.tail
            self.tail_pieces_total = tail.num_seqs * tail.pieces
            tail_tiles = self.tail_pieces_total * HO
            tail_words = tensormap_workspace_bytes(kernel_module, self.tail_pieces_total, self.num_householder) // 8
            tail_fused_words = tensormap_workspace_bytes(tail_summary_module, self.tail_pieces_total, self.num_householder) // 8
            off_tail_scheduler = layout.add(24)
            regions += [
                ("tail_scheduler_prefill", off_tail_scheduler, "int32", (2,)),
                ("tail_scheduler_h", off_tail_scheduler + 8, "int32", (2,)),
                ("tail_scheduler_m", off_tail_scheduler + 16, "int32", (2,)),
                ("tail_scheduler_all", off_tail_scheduler, "int32", (6,)),
                ("tail_tensormaps", layout.add(tail_words * 8), "int64", (tail_words,)),
                ("tail_work_items", layout.add(tail_tiles * WORK_ITEM_FIELDS * 4), "int32", (tail_tiles, WORK_ITEM_FIELDS)),
                ("tail_fused_tensormaps", layout.add(tail_fused_words * 8), "int64", (tail_fused_words,)),
            ]
            table = piece_table_layout(tail.num_seqs, tail.pieces, HO)
            off_tail_table = layout.add(table.nbytes)
            regions += [
                ("tail_main_rows", off_tail_table + table.main_rows, "int32", (tail.num_seqs + 1,)),
                ("tail_summary_rows", off_tail_table + table.summary_rows, "int32", (tail.num_seqs + 1,)),
                ("tail_cu_pieces", off_tail_table + table.cu_pieces, "int32", (self.tail_pieces_total + 1,)),
                ("tail_main_count", off_tail_table + table.main_count, "int32", (1,)),
                ("tail_summary_count", off_tail_table + table.summary_count, "int32", (1,)),
                ("tail_work_items_summary", layout.add(table.item_rows * WORK_ITEM_FIELDS * 4), "int32", (table.item_rows, WORK_ITEM_FIELDS)),
                ("tail_state_h", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", (self.tail_pieces_total, HO, V, K)),
                ("tail_state_m", layout.add(self.tail_pieces_total * HO * K * K * 4), "float32", (self.tail_pieces_total, HO, K, K)),
                ("tail_state_x", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", (self.tail_pieces_total, HO, V, K)),
            ]
            rows = tail_tinv_module.tinv_rows(self.q_rows, self.tail_pieces_total, self.expand_num, self.b_t)
            regions.append(("tail_tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tail_tinv_words = tensormap_workspace_bytes(tail_tinv_module, self.tail_pieces_total, self.num_householder) // 8
            regions.append(("tail_tinv_tensormaps", layout.add(tail_tinv_words * 8, align=128), "int64", (tail_tinv_words,)))
            regions.append(("tail_tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tail_tinv_row_count", layout.add(4), "int32", (1,)))
        if self.tinv_pass:
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(self.q_rows, self.num_pieces, self.expand_num, self.b_t)
            if not self.output_chunk_factors:
                regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, self.num_pieces, self.num_householder) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
        if self.use_qk_l2norm and not self.output_normalized_rows:
            regions.append(("q_n", layout.add(self.q_rows * HQ * K * 2), self.io_name, (self.q_rows, HQ, K)))
            if not self.has_k_normalized:
                regions.append(("k_n", layout.add(total * HK * K * 2), self.io_name, (total, HK, K)))
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
        self.index_state_indices = pos.get("state_indices")
        self.index_state_checkpoints = pos.get("state_checkpoints")
        self.index_a_log = pos.get("a_log")
        self.index_dt_bias = pos.get("dt_bias")
        self.index_preceding_state_update = pos.get("preceding_state_update")
        self.index_preceding_transition = pos.get("preceding_transition")
        self.index_composed = pos.get("composed_initial_state")
        if self.chain:
            positions = pos | {name: len(names) + i for i, name in enumerate(self.carve_names)}
            if self.has_context:
                positions |= {"context_" + name: len(names) + len(self.carve_names) + i for i, name in enumerate(self.context.names)}
            aliases = {
                "gate": "g",
                "o": "O",
                "seed": "initial_state",
                "composed": "composed_initial_state",
                "seed_indices": "state_indices",
                "final_indices": "state_indices",
                "checkpoints": "state_checkpoints",
                "scheduler_summary": "scheduler_h",
                "tinv_words": "tinv_tensormaps",
                "summary_words": "fused_tensormaps",
                "prefill_words": "tensormaps",
            }
            if self.use_qk_l2norm:
                aliases |= {"q": "q_n", "k": "context_k_n" if self.has_k_normalized else "k_n"}
            if self.output_normalized_rows:
                aliases |= {"q": "context_q_n", "k": "context_k_n"}
            if self.output_chunk_factors:
                aliases |= {"tinv": "context_tinv"}
            absent = len(names) + len(self.carve_names) + (len(self.context.names) if self.has_context else 0)
            if not self.safe_gate:
                positions.pop("a_log", None)
                positions.pop("dt_bias", None)
            indices = [positions.get(aliases.get(name, name), absent) for name in self.chain_buffer_names]
            self.chain_buffers = itemgetter(*indices)
            fold = "preceding_state_update" in pos
            lead = ("preceding_state_update", "preceding_transition", "initial_state", "composed_initial_state") if fold else (None,) * 4
            lead += ("q", "k") if self.use_qk_l2norm else (None, None)
            lead += ("context_inv_q", "context_inv_k") if self.output_normalized_rows else (None, None)
            self.chain_lead = itemgetter(*(positions.get(name, absent) for name in lead))
            self.chain_scalars = (int(self.pieces), int(self.n_heads_out), int(self.num_seqs), int(self.checkpoint or 0), float(self.scale), 0, 0)
        if self.tail is not None:
            positions = pos | {name: len(names) + i for i, name in enumerate(self.carve_names)}
            aliases = {
                "scheduler_summary": "tail_scheduler_h",
                "tinv_words": "tail_tinv_tensormaps",
                "summary_words": "tail_fused_tensormaps",
                "state_h_summary": "tail_state_h",
                "state_m_summary": "tail_state_m",
                "state_h_chain": "tail_state_h",
                "state_m_chain": "tail_state_m",
                "state_x_chain": "tail_state_x",
                "state_x_prefill": "tail_state_x",
                "seed": "initial_state",
                "work_count": "tail_work_count" if self.tail.dv_split else "tail_main_count",
                "scheduler": "tail_scheduler" if self.tail.dv_split else "tail_scheduler_prefill",
                "workspace": "tail_tensormaps",
            }
            absent = len(names) + len(self.carve_names)
            if self.tail.dv_split:
                positions.pop("initial_state", None)
            indices = [positions.get(aliases.get(name, "tail_" + name), absent) for name in self.tail_buffer_names]
            self.tail_buffers = itemgetter(*indices)
            self.tail_scalars = (
                int(self.tail.pieces),
                int(self.tail.num_seqs),
                int(self.tail.batch_base),
                int(self.tail.first_head_lo),
                self.tail_ctas if self.tail.dv_split else 0,
                self.tail_item_base if self.tail.dv_split else 0,
            )

    def run(self, views, workspace, stream, context=None) -> None:
        composed = views[self.index_composed] if self.index_composed is not None else None
        q = views[self.index_q]
        if self.chain and self.chain_launch is not None:
            carved = workspace.carve(self.carve)
            context_views = context.carve(self.context.carve) if self.has_context else ()
            merged = (*views, *carved, *context_views, None)
            self.chain_launch(*self.chain_scalars, *self.chain_buffers(merged), self.stream_type(int(stream or 0)), None, *self.chain_lead(merged))
            return
        k = views[self.index_k]
        v = views[self.index_v]
        g = views[self.index_g]
        beta = views[self.index_beta]
        cu = views[self.index_cu_seqlens]
        state0 = views[self.index_initial_state] if self.index_initial_state is not None else None
        o = views[self.index_o]
        final_state = views[self.index_final_state] if self.index_final_state is not None else None
        state_indices = views[self.index_state_indices] if self.index_state_indices is not None else None
        state_checkpoints = views[self.index_state_checkpoints] if self.index_state_checkpoints is not None else None
        a_log = views[self.index_a_log] if self.index_a_log is not None else None
        dt_bias = views[self.index_dt_bias] if self.index_dt_bias is not None else None
        stream = stream if stream is not None else 0
        region = dict(zip(self.carve_names, workspace.carve(self.carve)))
        context_region = self.context.regions(context) if self.has_context else {}
        fold = self.index_preceding_state_update is not None
        lead_buffers = (
            views[self.index_preceding_state_update] if fold else None,
            views[self.index_preceding_transition] if fold else None,
            state0 if fold else None,
            composed if fold else None,
            q if self.use_qk_l2norm else None,
            k if self.use_qk_l2norm else None,
            context_region["inv_q"] if self.output_normalized_rows else None,
            context_region["inv_k"] if self.output_normalized_rows else None,
        )
        if self.use_qk_l2norm:
            rows = context_region if self.output_normalized_rows else region
            q, k = rows["q_n"], (context_region["k_n"] if self.has_k_normalized else rows["k_n"])
        if self.output_chunk_factors:
            region["tinv"] = context_region["tinv"]
        if self.chain:
            self.run_chain(
                q, k, v, g, beta, cu, state0, composed, o, final_state, state_checkpoints, a_log, dt_bias, state_indices, region, stream, lead_buffers
            )
            return

        buffers = dict(
            q=q,
            k=k,
            v=v,
            gate=g,
            beta=beta,
            a_log=a_log if self.safe_gate else None,
            dt_bias=dt_bias if self.safe_gate else None,
            o=o,
            cu_seqlens=cu,
            state_in=state0,
            composed=composed,
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
            item_limit_word=region.get("gate"),
            tail_scalars=self.tail_scalars if self.tail is not None else (),
            tail_buffers=self.tail_buffers((*views, *region.values(), None)) if self.tail is not None else (),
            lead_buffers=lead_buffers,
        )
        if self.warmup_launch is None:
            tail = self.tail
            self.warmup_launch = self.build_warmup_forward(
                **buffers,
                l2norm_skip_k=self.has_k_normalized,
                tail=tail,
                tail_chain_rows=(
                    self.chain_rows_per_cta(self.dim_v, self.dim_k, tail.num_seqs, self.n_heads_out, self.num_sm)
                    if tail is not None and not tail.dv_split
                    else 0
                ),
                split=self.split,
                tiles_per_head=self.tiles_per_head,
                n_tiles=self.n_tiles,
                ideal_chunks=self.ideal,
                num_sm=self.num_sm,
                b_t=self.b_t,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                expand_num=self.expand_num,
                checkpoint_every_n_tokens=self.checkpoint,
                checkpoint_after_block=self.checkpoint_after_block,
                scale=self.scale,
                device=self.device,
                stream=stream,
            )
        self.run_warmup_forward(*self.warmup_launch, **buffers, checkpoint_every_n_tokens=self.checkpoint, scale=self.scale, stream=stream)

    def run_chain(
        self, q, k, v, g, beta, cu, state0, composed, o, final_state, state_checkpoints, a_log, dt_bias, state_indices, region, stream, lead_buffers
    ) -> None:
        """Chain prologue, T pass, fused H and M summaries, fp32 state chain seeded with initial_state, prefill over the
        pieces seeded with X, all five launched from one compiled host (kernel/gdn_chain_forward_f16.py); the checkpoint
        series keeps its unsplit layout."""
        buffers = dict(
            q=q,
            k=k,
            v=v,
            gate=g,
            beta=beta,
            a_log=a_log if self.safe_gate else None,
            dt_bias=dt_bias if self.safe_gate else None,
            o=o,
            cu_seqlens=cu,
            cu_pieces=region["cu_pieces"],
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=region["main_count"],
            summary_count=region["summary_count"],
            work_items=region["work_items"],
            work_items_summary=region["work_items_summary"],
            scheduler_all=region["scheduler_all"],
            scheduler_summary=region["scheduler_h"],
            scheduler_prefill=region["scheduler_prefill"],
            tinv_words=region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            summary_words=region["fused_tensormaps"],
            prefill_words=region["tensormaps"],
            tinv=region["tinv"],
            state_h=region["state_h"],
            state_m=region["state_m"],
            state_x=region["state_x"],
            seed=state0,
            composed=composed,
            seed_indices=state_indices,
            final_state=final_state,
            final_indices=state_indices,
            checkpoints=state_checkpoints,
            lead_buffers=lead_buffers,
        )
        if self.chain_launch is None:
            self.chain_launch = self.build_chain_forward(
                **buffers,
                l2norm_skip_k=self.has_k_normalized,
                pieces=self.pieces,
                heads_out=self.n_heads_out,
                num_seqs=self.num_seqs,
                unit_chunks=self.unit_chunks,
                b_t=self.b_t,
                expand_num=self.expand_num,
                length_rule=self.length_rule,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                checkpoint_every_n_tokens=self.checkpoint,
                checkpoint_after_block=self.checkpoint_after_block,
                scale=self.scale,
                chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
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
            checkpoint_every_n_tokens=self.checkpoint,
            scale=self.scale,
            stream=stream,
        )


class CompiledGdnBwd:
    """Compiled FROST GDN / GDP bprop plan over the resolved node buffers: dQ/dK/dV/dG/dBeta from the forward checkpoint
    series, regenerated by the recompute kernel when the port is absent.  ``chain`` runs the pieces as independent
    sequences: H, M, X from the fused summary and the forward chain (M alone when the series is passed back), G from the
    bprop summary, a reverse fp32 chain seeding every piece's outgoing gradient, the gather returning piece 0's
    d_initial_state.  The GDP bprop summary reads compact q / dO at every d_v; only the d_v = 128 main bprop reads the
    expanded pack."""

    def __init__(self, node, bwd_module, recompute_module):
        from .common.piece_chain import chain_rows_per_cta, choose_pieces, piece_table_layout
        from .kernel.gdn_chain_backward_f16 import build_chain_backward, run_chain_backward
        from .kernel.gdn_warmup_backward_f16 import TAIL_BUFFERS, build_warmup_backward, run_warmup_backward
        from .common.split_k import WORK_ITEM_FIELDS, compute_ideal_chunks, max_work_items, split_scratch_rows
        from .common.tail import GATE_WORDS, choose_tail

        self.node = node
        self.bwd = bwd_module
        self.chain_rows_per_cta = chain_rows_per_cta
        self.build_chain_backward = build_chain_backward
        self.run_chain_backward = run_chain_backward
        self.build_warmup_backward = build_warmup_backward
        self.run_warmup_backward = run_warmup_backward
        self.chain_launch = None
        self.tail_buffer_names = TAIL_BUFFERS
        self.warmup_launch = None
        self.plan_name = "GdpFrostEngine (GDP_BWD)" if node.node_type.name == "GDP_BWD" else "GdnFrostEngine (GDN_BWD)"
        self.device = current_device()
        from .common.gate_bwd import scalar_gate_blocks
        from .common.host import tensormap_workspace_bytes

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
        k_dim = node.inputs["k"].dim
        total = k_dim[0] * k_dim[1] if len(k_dim) == 4 else k_dim[0]
        HK = k_dim[-2]
        self.q_rows = total // self.expand_num
        self.gate_bwd_blocks = scalar_gate_blocks(self.q_rows)
        K, V = q.dim[-1], v.dim[-1]
        HQ, HV = q.dim[1], v.dim[-2]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        after_block = bool(node.params.get("checkpoint_after_block", False))
        self.has_state_checkpoints = "state_checkpoints" in node.inputs and not after_block
        self.checkpoint_cadence = 0 if after_block else int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.coarse_checkpoints = self.has_state_checkpoints and (
            self.checkpoint_cadence > self.b_t or node.inputs["state_checkpoints"].get_data_type().name == "FLOAT"
        )
        self.needs_recompute = not self.has_state_checkpoints or self.coarse_checkpoints
        self.io_name = "float16" if node.inputs["q"].get_data_type().name == "HALF" else "bfloat16"
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"

        self.num_sm = multiprocessor_count(self.device)
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.overwrite_initial_state = bool(node.params.get("overwrite_initial_state", False))
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
        self.tail = None
        if not self.chain and not self.batch_invariant:
            self.tail = choose_tail(
                num_seqs=B,
                heads_out=HO,
                num_sm=self.num_sm,
                total_tokens=self.q_rows,
                b_t=self.b_t,
                cadence_tokens=self.checkpoint_cadence,
                expand_num=self.expand_num,
            )
            if self.tail is not None and self.tail.batch_base == 0 and ("preceding_state_update" in node.inputs or "composed_initial_state" in node.inputs):
                self.tail = None
        self.split = not self.chain and not self.batch_invariant and not self.overwrite_initial_state
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        self.fused_h_m = self.chain and not self.has_state_checkpoints
        self.pack_bwd_needed = self.num_householder > 1 and not self.compact_qdo
        self.summary_q_step = self.num_householder if self.pack_bwd_needed and self.use_qk_l2norm else 1
        self.tinv_pass = self.chain or self.needs_recompute
        self.has_context = bool(node.params.get("input_context", False))
        self.context_from_forward = self.has_context and bool(node.params.get("context_from_forward", False))
        self.context = None
        if self.context_from_forward:
            from .kernel import gdn_tinv_f16 as tinv_module

            self.context = forward_apply_context(
                self.plan_name,
                total=total,
                num_householder=self.num_householder,
                num_seqs=B,
                h_q=HQ,
                h_k=HK,
                h_o=HO,
                d_k=K,
                d_v=V,
                b_t=self.b_t,
                io_name=self.io_name,
                num_sm=self.num_sm,
                cadence_tokens=int(node.params.get("checkpoint_every_n_tokens", 0) or 0),
                batch_invariant=self.batch_invariant,
                use_qk_l2norm=self.use_qk_l2norm,
                tinv_rows=tinv_module.tinv_rows,
            )
        elif self.has_context:
            from .kernel import gdn_tinv_f16 as tinv_module

            self.context = backward_context(
                self.plan_name,
                full=True,
                total=total,
                num_householder=self.num_householder,
                h_q=HQ,
                h_k=HK,
                h_o=HO,
                d_k=K,
                d_v=V,
                b_t=self.b_t,
                io_name=self.io_name,
                num_sm=self.num_sm,
                cadence_tokens=self.checkpoint_cadence,
                batch_invariant=self.batch_invariant,
                use_qk_l2norm=self.use_qk_l2norm,
                tinv_rows=tinv_module.tinv_rows,
            )
        self.context_size = self.context.size if self.has_context else 0
        same_cut = (
            self.has_context
            and self.context.has_tables
            and (B == 1 or self.context_from_forward)
            and (self.pieces, self.unit_chunks) == (self.context.pieces, self.context.unit_chunks)
        )
        self.has_normalized_rows = self.has_context and self.context.has_rows
        self.has_chunk_factors = same_cut and self.chain
        self.has_piece_tables = same_cut and self.chain and not self.context_from_forward
        self.has_piece_transitions = self.has_piece_tables and not self.fused_h_m

        layout = WorkspaceLayout()
        regions = []
        off_scheduler = layout.add(40 if self.chain else 16)
        regions += [
            ("scheduler_recompute", off_scheduler, "int32", (2,)),
            ("scheduler_bwd", off_scheduler + 8, "int32", (2,)),
        ]
        if self.chain:
            from .kernel import gdn_bprop_summary_f16 as summary_module

            self.summary_tensormap_words = tensormap_workspace_bytes(summary_module, self.num_pieces, self.num_householder) // 8
            regions += [
                ("scheduler_summary", off_scheduler + 16, "int32", (2,)),
                ("scheduler_m", off_scheduler + 24, "int32", (2,)),
                ("scheduler_series", off_scheduler + 32, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (10,)),
                ("summary_tensormaps", layout.add(self.summary_tensormap_words * 8), "int64", (self.summary_tensormap_words,)),
            ]
        else:
            regions.append(("scheduler_all", off_scheduler, "int32", (4,)))
        self.tensormap_words = tensormap_workspace_bytes(bwd_module, self.num_pieces, self.num_householder) // 8
        regions.append(("tensormaps", layout.add(self.tensormap_words * 8), "int64", (self.tensormap_words,)))
        if "following_state_grad" in node.inputs:
            regions.append(("composed_dstate", layout.add(HO * V * K * 4, align=16), "float32", (1, HO, V, K)))
        if self.needs_recompute:
            self.state_checkpoints_rows = max(total // self.b_t + self.num_pieces, 1)
            regions.append(
                ("state_checkpoints", layout.add(self.state_checkpoints_rows * HO * K * V * 2), self.io_name, (self.state_checkpoints_rows, HO, V, K))
            )
        self.recompute_tensormap_words = tensormap_workspace_bytes(recompute_module, self.num_pieces, self.num_householder) // 8
        if self.needs_recompute and not self.chain:
            regions.append(("recompute_tensormaps", layout.add(self.recompute_tensormap_words * 8), "int64", (self.recompute_tensormap_words,)))
        if self.chain:
            regions.append(("recompute_tensormaps_m", layout.add(self.recompute_tensormap_words * 8), "int64", (self.recompute_tensormap_words,)))
            if self.needs_recompute:
                regions.append(("recompute_tensormaps_series", layout.add(self.recompute_tensormap_words * 8), "int64", (self.recompute_tensormap_words,)))
        if self.fused_h_m:
            from .kernel import gdn_summary_f16 as fused_module

            self.fused_tensormap_words = tensormap_workspace_bytes(fused_module, self.num_pieces, self.num_householder) // 8
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
        if self.use_qk_l2norm and not self.has_normalized_rows:
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
        if self.tail is not None:
            regions.append(("gate", layout.add(4 * GATE_WORDS, align=16), "int32", (GATE_WORDS,)))
        if self.split:
            self.chunk_scratch_rows = split_scratch_rows(total, B, HO, self.b_t)
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
                ("state_dx_end", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape),
            ]
            if not self.has_piece_transitions:
                regions.append(("state_m", layout.add(self.num_pieces * HO * K * K * 4), "float32", (self.num_pieces, HO, K, K)))
            if not self.has_piece_tables:
                regions.append(("state_g", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape))
            if not self.has_state_checkpoints:
                regions.append(("state_h", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape))
                regions.append(("state_x", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape))
        if self.tinv_pass:
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(self.q_rows, self.num_pieces, self.expand_num, self.b_t)
            if not self.has_chunk_factors:
                regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, self.num_pieces, self.num_householder) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
        if self.tail is not None:
            from .kernel import gdn_bprop_summary_f16 as tail_summary_module
            from .kernel import gdn_tinv_f16 as tail_tinv_module

            tail = self.tail
            self.tail_pieces_total = tail.num_seqs * tail.pieces
            tail_tiles = self.tail_pieces_total * HO
            off_tail_scheduler = layout.add(40)
            regions += [
                ("tail_scheduler_recompute", off_tail_scheduler, "int32", (2,)),
                ("tail_scheduler_bwd", off_tail_scheduler + 8, "int32", (2,)),
                ("tail_scheduler_summary", off_tail_scheduler + 16, "int32", (2,)),
                ("tail_scheduler_m", off_tail_scheduler + 24, "int32", (2,)),
                ("tail_scheduler_series", off_tail_scheduler + 32, "int32", (2,)),
                ("tail_scheduler_all", off_tail_scheduler, "int32", (10,)),
            ]
            words = tensormap_workspace_bytes(tail_summary_module, self.tail_pieces_total, self.num_householder) // 8
            regions.append(("tail_summary_tensormaps", layout.add(words * 8), "int64", (words,)))
            words = tensormap_workspace_bytes(bwd_module, self.tail_pieces_total, self.num_householder) // 8
            regions.append(("tail_tensormaps", layout.add(words * 8), "int64", (words,)))
            words = tensormap_workspace_bytes(recompute_module, self.tail_pieces_total, self.num_householder) // 8
            regions.append(("tail_recompute_tensormaps_m", layout.add(words * 8), "int64", (words,)))
            if self.needs_recompute:
                regions.append(("tail_recompute_tensormaps_series", layout.add(words * 8), "int64", (words,)))
            if not self.has_state_checkpoints:
                from .kernel import gdn_summary_f16 as tail_fused_module

                words = tensormap_workspace_bytes(tail_fused_module, self.tail_pieces_total, self.num_householder) // 8
                regions.append(("tail_fused_tensormaps", layout.add(words * 8), "int64", (words,)))
            if self.coarse_checkpoints:
                rows = max((total // self.recompute_span_tokens + 2 * self.tail_pieces_total) * HO, 1)
                regions.append(("tail_work_items_recompute", layout.add(rows * WORK_ITEM_FIELDS * 4), "int32", (rows, WORK_ITEM_FIELDS)))
                regions.append(("tail_work_count_recompute", layout.add(4), "int32", (1,)))
                regions.append(("tail_seed_row_base", layout.add(4), "int32", (1,)))
            regions.append(("tail_work_items", layout.add(tail_tiles * WORK_ITEM_FIELDS * 4), "int32", (tail_tiles, WORK_ITEM_FIELDS)))
            table = piece_table_layout(tail.num_seqs, tail.pieces, HO)
            off_tail_table = layout.add(table.nbytes)
            state_shape = (self.tail_pieces_total, HO, V, K)
            regions += [
                ("tail_main_rows", off_tail_table + table.main_rows, "int32", (tail.num_seqs + 1,)),
                ("tail_summary_rows", off_tail_table + table.summary_rows, "int32", (tail.num_seqs + 1,)),
                ("tail_cu_pieces", off_tail_table + table.cu_pieces, "int32", (self.tail_pieces_total + 1,)),
                ("tail_main_count", off_tail_table + table.main_count, "int32", (1,)),
                ("tail_summary_count", off_tail_table + table.summary_count, "int32", (1,)),
                ("tail_work_items_summary", layout.add(table.item_rows * WORK_ITEM_FIELDS * 4), "int32", (table.item_rows, WORK_ITEM_FIELDS)),
                ("tail_state_m", layout.add(self.tail_pieces_total * HO * K * K * 4), "float32", (self.tail_pieces_total, HO, K, K)),
                ("tail_state_g", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", state_shape),
                ("tail_state_dx_end", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", state_shape),
            ]
            if not self.has_state_checkpoints:
                regions.append(("tail_state_h", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", state_shape))
                regions.append(("tail_state_x", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", state_shape))
            if not self.tinv_pass:
                rows = tail_tinv_module.tinv_rows(self.q_rows, self.tail_pieces_total, self.expand_num, self.b_t)
                regions.append(("tail_tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
                words = tensormap_workspace_bytes(tail_tinv_module, self.tail_pieces_total, self.num_householder) // 8
                regions.append(("tail_tinv_tensormaps", layout.add(words * 8, align=128), "int64", (words,)))
                regions.append(("tail_tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
                regions.append(("tail_tinv_row_count", layout.add(4), "int32", (1,)))
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
        self.index_following_state_grad = pos.get("following_state_grad")
        self.index_following_transition = pos.get("following_transition")
        self.index_composed = pos.get("composed_initial_state")
        if self.tail is not None:
            positions = pos | {name: len(names) + i for i, name in enumerate(self.carve_names)}
            if self.has_context:
                positions |= {"context_" + name: len(names) + len(self.carve_names) + i for i, name in enumerate(self.context.names)}
            coarse = self.coarse_checkpoints
            rows = "context_" if self.has_normalized_rows else ""
            aliases = {
                "series_items": "tail_work_items_recompute",
                "series_count": "tail_work_count_recompute",
                "recompute_items": ("tail_work_items_recompute" if coarse else "tail_work_items") if self.needs_recompute else None,
                "recompute_count": ("tail_work_count_recompute" if coarse else "tail_main_count") if self.needs_recompute else None,
                "tinv_words": "tail_tinv_tensormaps",
                "summary_words": "tail_fused_tensormaps",
                "recompute_m_words": "tail_recompute_tensormaps_m",
                "series_words": "tail_recompute_tensormaps_series",
                "bprop_summary_words": "tail_summary_tensormaps",
                "bprop_words": "tail_tensormaps",
                "summary_q": "q" if self.pack_expands_q or not self.use_qk_l2norm else rows + "q_n",
                "summary_do": "dO",
                "tinv": "tinv" if self.tinv_pass else "tail_tinv",
                "state_h_summary": "tail_state_h",
                "state_m_main": "tail_state_m",
                "state_h_chain": "tail_state_h",
                "state_m_chain": "tail_state_m",
                "state_x_chain": "tail_state_x",
                "seed": "initial_state",
                "state_x_series": "tail_state_x",
                "dseed": "d_final_state",
            }
            absent = len(names) + len(self.carve_names) + (len(self.context.names) if self.has_context else 0)
            indices = [positions.get(aliases.get(name, "tail_" + name), absent) for name in self.tail_buffer_names]
            self.tail_buffers = itemgetter(*indices)
            self.tail_scalars = (
                int(self.tail.pieces),
                int(self.tail.num_seqs),
                int(self.recompute_span_tokens if coarse else 0) // self.b_t,
                int(self.tail.batch_base),
                int(self.tail.first_head_lo),
            )

    def run(self, views, workspace, stream, context=None) -> None:
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
        composed = views[self.index_composed] if self.index_composed is not None else None
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
        context_region = self.context.regions(context) if self.has_context else {}
        q_n = context_region["q_n"] if self.has_normalized_rows else region.get("q_n")
        k_n = context_region["k_n"] if self.has_normalized_rows else region.get("k_n")
        inv_q = context_region["inv_q"] if self.has_normalized_rows else region.get("inv_q")
        inv_k = context_region["inv_k"] if self.has_normalized_rows else region.get("inv_k")
        tinv = context_region["tinv"] if self.has_chunk_factors else region.get("tinv")
        state_m = context_region["state_m"] if self.has_piece_transitions else region.get("state_m")
        state_g = context_region["state_g"] if self.has_piece_tables else region.get("state_g")
        fold = self.index_following_state_grad is not None
        dcomposed = region["composed_dstate"] if fold else None
        scheduler_recompute = region["scheduler_recompute"]
        scheduler_bwd = region["scheduler_bwd"]
        work_items = region["work_items"]
        work_count = region["work_count"]
        dq_node = None
        q_compact, do_compact = q, do
        normalize = self.use_qk_l2norm and not self.has_normalized_rows
        lead_buffers = (
            views[self.index_following_state_grad] if fold else None,
            views[self.index_following_transition] if fold else None,
            dstate_in if fold else None,
            dcomposed,
            q if self.pack_expands_q else None,
            region["q_x"] if self.pack_expands_q else None,
            do if self.pack_bwd_needed else None,
            region["do_x"] if self.pack_bwd_needed else None,
            q if normalize else None,
            k if normalize else None,
            inv_q if normalize else None,
            inv_k if normalize else None,
        )
        if self.pack_bwd_needed:
            do = region["do_x"]
            if self.pack_expands_q:
                q = region["q_x"]
            dq_node = dq
            dq = region["dq_x"]
        if self.use_qk_l2norm:
            q, k = q_n, k_n
        summary_q = q_compact if self.pack_expands_q else q
        summary_do = do_compact
        dq_out = region["dq_ho"] if self.fold_dq else dq
        dk_out = region["dk_ho"] if self.fold_dk else dk
        dv_out = region["dv_ho"] if self.fold_dv else dv
        post_buffers = (
            *((dg, g, a_log, dt_bias, region.get("gate_part_a"), region.get("gate_part_dt"), d_a_log, d_dt_bias) if self.safe_gate else (None,) * 8),
            *((dq_out, dq) if self.fold_dq else (None, None)),
            *((dk_out, dk) if self.fold_dk else (None, None)),
            *((dv_out, dv) if self.fold_dv else (None, None)),
            *((dq, dq_node) if dq_node is not None else (None, None)),
        )

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
                composed,
                dcomposed,
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
                tinv,
                state_m,
                state_g,
                inv_q,
                inv_k,
                lead_buffers,
                post_buffers,
            )
        else:
            checkpoint_series = state_checkpoints if (self.has_state_checkpoints and not self.coarse_checkpoints) else region["state_checkpoints"]
            coarse = self.coarse_checkpoints
            buffers = dict(
                q=q,
                k=k,
                v=v,
                do=do,
                dq=dq_out,
                dk=dk_out,
                dv=dv_out,
                gate=g,
                beta=beta,
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
                cu_seqlens=cu,
                tinv=tinv,
                tinv_words=region.get("tinv_tensormaps"),
                tinv_rows=region.get("tinv_rows"),
                tinv_row_count=region.get("tinv_row_count"),
                checkpoints=checkpoint_series,
                seed_checkpoints=state_checkpoints if coarse else None,
                state_in=state0 if self.needs_recompute and not coarse else None,
                composed=composed if self.needs_recompute and not coarse else None,
                work_items=work_items,
                work_count=work_count,
                series_items=(region["work_items_recompute"] if coarse else work_items) if self.needs_recompute else None,
                series_count=(region["work_count_recompute"] if coarse else work_count) if self.needs_recompute else None,
                item_scratch=region.get("item_scratch"),
                chunk_scratch=region.get("chunk_scratch"),
                scheduler_all=region["scheduler_all"],
                scheduler_recompute=scheduler_recompute,
                recompute_words=region.get("recompute_tensormaps"),
                bprop_words=region["tensormaps"],
                dgate=dg,
                dbeta=dbeta,
                dstate0=dstate0,
                dstate_in=dstate_in,
                dcomposed=dcomposed,
                inv_q=inv_q,
                inv_k=inv_k,
                scheduler_bwd=scheduler_bwd,
                item_limit_word=region.get("gate"),
                tail_scalars=self.tail_scalars if self.tail is not None else (),
                tail_buffers=self.tail_buffers((*views, *region.values(), *context_region.values(), None)) if self.tail is not None else (),
                lead_buffers=lead_buffers,
                gate_blocks=self.gate_bwd_blocks,
                post_buffers=post_buffers,
            )
            seeds = dict(
                seed_span_tokens=self.recompute_span_tokens if coarse else 0,
                seed_every_n_tokens=self.checkpoint_cadence if coarse else 0,
            )
            if self.warmup_launch is None:
                tail = self.tail
                self.warmup_launch = self.build_warmup_backward(
                    bprop_module=self.bwd,
                    **buffers,
                    **seeds,
                    scale=self.scale,
                    use_initial_state=state0 is not None or composed is not None,
                    tail=tail,
                    tail_chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, tail.num_seqs, self.n_heads_out, self.num_sm) if tail is not None else 0,
                    tail_summary_q_step=self.summary_q_step,
                    split=self.split,
                    n_tiles=self.n_tiles,
                    ideal_chunks=self.ideal,
                    num_sm=self.num_sm,
                    b_t=self.b_t,
                    expand_num=self.expand_num,
                    tinv_pass=self.tinv_pass,
                    recompute=self.needs_recompute,
                    recompute_orders=self.recompute_orders,
                    coarse=coarse,
                    bwd_orders=self.bwd_orders,
                    compact_qdo=self.compact_qdo,
                    log_gate=self.log_gate,
                    safe_gate=self.safe_gate,
                    use_beta_sigmoid=self.use_beta_sigmoid,
                    allow_neg_eigval=self.allow_neg_eigval,
                    device=self.device,
                    stream=stream,
                )
            self.run_warmup_backward(
                *self.warmup_launch,
                **buffers,
                **seeds,
                scale=self.scale,
                b_t=self.b_t,
                tinv_pass=self.tinv_pass,
                recompute=self.needs_recompute,
                recompute_orders=self.recompute_orders,
                coarse=coarse,
                bwd_orders=self.bwd_orders,
                compact_qdo=self.compact_qdo,
                stream=stream,
            )
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
        composed,
        dcomposed,
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
        tinv,
        state_m,
        state_g,
        inv_q,
        inv_k,
        lead_buffers,
        post_buffers,
    ) -> None:
        """Chain prologue, T pass, H and M summaries (M alone when the series is passed back), forward chain X, G summary,
        reverse chain seeded with d_final_state, dense series (the forward's or the seeded recompute), bprop over the
        pieces."""
        series = not (self.has_state_checkpoints and not self.coarse_checkpoints)
        coarse = self.coarse_checkpoints
        buffers = dict(
            q=q,
            k=k,
            v=v,
            do=do,
            gate=g,
            beta=beta,
            a_log=a_log if self.safe_gate else None,
            dt_bias=dt_bias if self.safe_gate else None,
            cu_seqlens=cu,
            cu_pieces=region["cu_pieces"],
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=region["main_count"],
            summary_count=region["summary_count"],
            work_items=region["work_items"],
            work_items_summary=region["work_items_summary"],
            series_items=region.get("work_items_recompute"),
            series_count=region.get("work_count_recompute"),
            scheduler_all=region["scheduler_all"],
            scheduler_recompute=region["scheduler_recompute"],
            scheduler_m=region["scheduler_m"],
            scheduler_series=region["scheduler_series"],
            scheduler_summary=region["scheduler_summary"],
            scheduler_bwd=region["scheduler_bwd"],
            tinv_words=region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            summary_words=region.get("fused_tensormaps"),
            recompute_m_words=region["recompute_tensormaps_m"],
            series_words=region.get("recompute_tensormaps_series"),
            bprop_summary_words=region["summary_tensormaps"],
            bprop_words=region["tensormaps"],
            checkpoints=region["state_checkpoints"] if series else state_checkpoints,
            seed_checkpoints=state_checkpoints if coarse else None,
            dq=dq_out,
            dk=dk_out,
            dv=dv_out,
            dgate=dg,
            dbeta=dbeta,
            summary_q=summary_q,
            summary_do=summary_do,
            tinv=tinv,
            state_h=region.get("state_h"),
            state_m=state_m,
            state_x=region.get("state_x"),
            state_g=state_g,
            state_dx_end=region["state_dx_end"],
            seed=state0,
            dseed=dstate_in,
            composed=composed,
            dcomposed=dcomposed,
            dstate0=dstate0,
            inv_q=inv_q,
            inv_k=inv_k,
            lead_buffers=lead_buffers,
            gate_blocks=self.gate_bwd_blocks,
            post_buffers=post_buffers,
        )
        schedule = dict(
            pieces=self.pieces,
            heads_out=self.n_heads_out,
            num_seqs=self.num_seqs,
            b_t=self.b_t,
            compact_qdo=self.compact_qdo,
            fused_h_m=self.fused_h_m,
            series=series,
            coarse=coarse,
            series_span_tokens=self.recompute_span_tokens if coarse else 0,
            seed_every_n_tokens=self.checkpoint_cadence if coarse else 0,
            scale=self.scale,
            has_chunk_factors=self.has_chunk_factors,
            has_piece_transitions=self.has_piece_transitions,
            has_piece_state_grads=self.has_piece_tables,
        )
        if self.chain_launch is None:
            self.chain_launch = self.build_chain_backward(
                bprop_module=self.bwd,
                **buffers,
                **schedule,
                unit_chunks=self.unit_chunks,
                expand_num=self.expand_num,
                length_rule=self.length_rule,
                summary_q_step=self.summary_q_step,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
            )
        self.run_chain_backward(self.chain_launch, **buffers, **schedule, stream=stream)


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
    """Compiled summary plan.  With ``transition`` bound the fused H + M summary kernel writes ``final_state`` (initial_state
    honored) and ``M_buf = M^T`` (``X_final = X_init @ M_buf + X_H``), each in its own dtype, in one pass over K: on the uncut,
    split or ``dv_split`` table (two items per sequence and head, each owning half of d_v of H and, with a square state, half
    of the rows of M; otherwise M is stored once), or in ``chain``
    plans per filled piece, composed by one fp32 state chain whose tail is ``final_state`` and whose running product is
    ``transition``.  Without ``transition`` the state-only recompute kernel writes ``final_state`` alone.  The chunk factor T
    comes from the T pass."""

    def __init__(self, node, recompute_module):
        from .common.host import tensormap_workspace_bytes
        from .common.piece_chain import DV_SPLIT_TILES, chain_rows_per_cta, choose_pieces, is_dv_split, piece_table_layout
        from .kernel.gdn_chain_forward_f16 import build_chain_summary, run_chain_summary
        from .kernel.gdn_warmup_forward_f16 import SUMMARY_TAIL_BUFFERS, build_warmup_summary, run_warmup_summary
        from .common.split_k import WORK_ITEM_FIELDS, compute_ideal_chunks, max_work_items, split_scratch_rows
        from .common.tail import GATE_WORDS, choose_tail

        self.node = node
        self.build_chain_summary = build_chain_summary
        self.run_chain_summary = run_chain_summary
        self.build_warmup_summary = build_warmup_summary
        self.run_warmup_summary = run_warmup_summary
        self.chain_rows_per_cta = chain_rows_per_cta
        self.chain_launch = None
        self.warmup_launch = None
        self.tail_buffer_names = SUMMARY_TAIL_BUFFERS
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
        total = k.dim[0] * k.dim[1] if len(k.dim) == 4 else k.dim[0]
        HK = k.dim[-2]
        K = k.dim[-1]
        HO = g.dim[1]
        V = node.inputs["v"].dim[-1]
        self.io_name = "float16" if k.get_data_type().name == "HALF" else "bfloat16"
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.overwrite_initial_state = bool(node.params.get("overwrite_initial_state", False))
        self.num_sm = multiprocessor_count(self.device)
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
            expand_num=self.num_householder,
            compose_tail=True,
        )
        self.chain = self.pieces > 0
        self.dv_split = (
            not self.chain
            and not self.batch_invariant
            and is_dv_split(
                num_seqs=B,
                heads_out=HO,
                dim_v=V,
                num_sm=self.num_sm,
                total_tokens=total // self.num_householder,
                b_t=self.b_t,
                expand_num=self.num_householder,
                unit_chunks=self.unit_chunks,
            )
        )
        self.split = not self.chain and not self.dv_split and not self.batch_invariant and not self.overwrite_initial_state
        self.tiles_per_head = DV_SPLIT_TILES if self.dv_split else 1
        self.n_tiles = B * HO * self.tiles_per_head
        self.tail = None
        if not self.chain and not self.dv_split and not self.batch_invariant and self.has_transition:
            self.tail = choose_tail(
                num_seqs=B,
                heads_out=HO,
                num_sm=self.num_sm,
                total_tokens=total // self.num_householder,
                b_t=self.b_t,
                cadence_tokens=0,
                expand_num=self.num_householder,
            )
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        from .kernel import gdn_summary_f16 as summary_module

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
            words = tensormap_workspace_bytes(summary_module, self.num_pieces, self.num_householder) // 8
            regions.append(("fused_tensormaps", layout.add(words * 8, align=128), "int64", (words,)))
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(total // self.num_householder, self.num_pieces, self.num_householder, self.b_t)
            regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, self.num_pieces, self.num_householder) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
        else:
            off_scheduler = layout.add(8)
            tensormap_words = (
                max(tensormap_workspace_bytes(recompute_module, B, self.num_householder), tensormap_workspace_bytes(summary_module, B, self.num_householder))
                // 8
            )
            regions += [
                ("scheduler_final", off_scheduler, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (2,)),
                ("tensormaps", layout.add(tensormap_words * 8), "int64", (tensormap_words,)),
            ]
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(total // self.num_householder, B, self.num_householder, self.b_t)
            regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, B, self.num_householder) // 8
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
            if self.tail is not None:
                regions.append(("gate", layout.add(4 * GATE_WORDS, align=16), "int32", (GATE_WORDS,)))
            if self.split:
                self.chunk_scratch_rows = split_scratch_rows(total, B, HO, self.b_t)
                regions.append(("item_scratch", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
                regions.append(("chunk_scratch", layout.add(self.chunk_scratch_rows * HO * 4), "float32", (self.chunk_scratch_rows, HO)))
        if self.tail is not None:
            tail = self.tail
            self.tail_pieces_total = tail.num_seqs * tail.pieces
            off_tail_scheduler = layout.add(24)
            regions += [
                ("tail_scheduler_h", off_tail_scheduler, "int32", (2,)),
                ("tail_scheduler_m", off_tail_scheduler + 8, "int32", (2,)),
                ("tail_scheduler_all", off_tail_scheduler, "int32", (6,)),
            ]
            table = piece_table_layout(tail.num_seqs, tail.pieces, HO)
            off_tail_table = layout.add(table.nbytes, align=256)
            regions += [
                ("tail_main_rows", off_tail_table + table.main_rows, "int32", (tail.num_seqs + 1,)),
                ("tail_summary_rows", off_tail_table + table.summary_rows, "int32", (tail.num_seqs + 1,)),
                ("tail_cu_pieces", off_tail_table + table.cu_pieces, "int32", (self.tail_pieces_total + 1,)),
                ("tail_main_count", off_tail_table + table.main_count, "int32", (1,)),
                ("tail_summary_count", off_tail_table + table.summary_count, "int32", (1,)),
                ("tail_work_items", layout.add(table.item_rows * WORK_ITEM_FIELDS * 4), "int32", (table.item_rows, WORK_ITEM_FIELDS)),
                ("tail_state_h", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", (self.tail_pieces_total, HO, V, K)),
                ("tail_state_m", layout.add(self.tail_pieces_total * HO * K * K * 4), "float32", (self.tail_pieces_total, HO, K, K)),
                ("tail_state_x", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", (self.tail_pieces_total, HO, V, K)),
            ]
            words = tensormap_workspace_bytes(summary_module, self.tail_pieces_total, self.num_householder) // 8
            regions.append(("tail_fused_tensormaps", layout.add(words * 8, align=128), "int64", (words,)))
        self.has_context = bool(node.params.get("output_context", False))
        self.context = (
            forward_context(self.plan_name, total=total, h_k=HK, d_k=K, io_name=self.io_name, use_qk_l2norm=self.use_qk_l2norm) if self.has_context else None
        )
        self.context_size = self.context.size if self.has_context else 0
        self.has_k_normalized = self.has_context and self.use_qk_l2norm
        if self.use_qk_l2norm and not self.has_k_normalized:
            regions.append(("k_n", layout.add(total * HK * K * 2), self.io_name, (total, HK, K)))
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
        if self.tail is not None:
            positions = pos | {name: len(names) + i for i, name in enumerate(self.carve_names)}
            aliases = {
                "scheduler_summary": "tail_scheduler_h",
                "summary_words": "tail_fused_tensormaps",
                "state_h_summary": "tail_state_h",
                "state_m_summary": "tail_state_m",
                "state_h_chain": "tail_state_h",
                "state_m_chain": "tail_state_m",
                "state_x_chain": "tail_state_x",
                "seed": "initial_state",
                "final_state": "final_state",
                "transition": "transition",
            }
            absent = len(names) + len(self.carve_names)
            indices = [positions.get(aliases.get(name, "tail_" + name), absent) for name in self.tail_buffer_names]
            self.tail_buffers = itemgetter(*indices)
            self.tail_scalars = (int(self.tail.pieces), int(self.tail.num_seqs), int(self.tail.batch_base), int(self.tail.first_head_lo))

    def run(self, views, workspace, stream, context=None) -> None:
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
        context_region = self.context.regions(context) if self.has_k_normalized else {}
        l2norm_k = None
        if self.use_qk_l2norm:
            l2norm_k = k
            k = context_region["k_n"] if self.has_k_normalized else region["k_n"]
        if self.chain:
            self.run_chain(k, v, g, beta, cu, state0, final_state, transition, a_log, dt_bias, region, stream, l2norm_k)
            return
        tail = self.tail
        buffers = dict(
            k=k,
            v=v,
            gate=g,
            beta=beta,
            a_log=a_log if self.safe_gate else None,
            dt_bias=dt_bias if self.safe_gate else None,
            cu_seqlens=cu,
            state_in=state0,
            state_out=final_state,
            transition=transition,
            work_items=region["work_items"],
            work_count=region["work_count"],
            item_scratch=region.get("item_scratch"),
            chunk_scratch=region.get("chunk_scratch"),
            scheduler=region["scheduler_final"],
            scheduler_all=region["scheduler_all"],
            workspace=region["tensormaps"],
            tinv=region["tinv"],
            tinv_words=region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            item_limit_word=region.get("gate"),
            l2norm_k=l2norm_k,
            tail_scalars=self.tail_scalars if tail is not None else (),
            tail_buffers=self.tail_buffers((*views, *region.values(), None)) if tail is not None else (),
        )
        if self.warmup_launch is None:
            self.warmup_launch = self.build_warmup_summary(
                **buffers,
                tail=tail,
                tail_chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, tail.num_seqs, self.n_heads_out, self.num_sm) if tail is not None else 0,
                split=self.split,
                tiles_per_head=self.tiles_per_head,
                n_tiles=self.n_tiles,
                ideal_chunks=self.ideal,
                num_sm=self.num_sm,
                b_t=self.b_t,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                expand_num=self.num_householder,
                device=self.device,
                stream=stream,
            )
        self.run_warmup_summary(*self.warmup_launch, **buffers, stream=stream)

    def run_chain(self, k, v, g, beta, cu, state0, final_state, transition, a_log, dt_bias, region, stream, l2norm_k) -> None:
        """Chain prologue, T pass, fused H and M summaries of every filled piece, one fp32 state chain seeded with
        initial_state whose tail is final_state and whose running product is transition."""
        buffers = dict(
            k=k,
            v=v,
            gate=g,
            beta=beta,
            a_log=a_log if self.safe_gate else None,
            dt_bias=dt_bias if self.safe_gate else None,
            cu_seqlens=cu,
            cu_pieces=region["cu_pieces"],
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=region["main_count"],
            summary_count=region["summary_count"],
            work_items=region["work_items"],
            scheduler_all=region["scheduler_all"],
            scheduler_summary=region["scheduler_h"],
            tinv_words=region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            summary_words=region["fused_tensormaps"],
            tinv=region["tinv"],
            state_h=region["state_h"],
            state_m=region["state_m"],
            state_x=region["state_x"],
            seed=state0,
            final_state=final_state,
            transition=transition,
            l2norm_k=l2norm_k,
        )
        if self.chain_launch is None:
            self.chain_launch = self.build_chain_summary(
                **buffers,
                pieces=self.pieces,
                heads_out=self.n_heads_out,
                num_seqs=self.num_seqs,
                unit_chunks=self.unit_chunks,
                b_t=self.b_t,
                expand_num=self.num_householder,
                length_rule=self.length_rule,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
            )
        self.run_chain_summary(
            self.chain_launch,
            **buffers,
            pieces=self.pieces,
            heads_out=self.n_heads_out,
            num_seqs=self.num_seqs,
            stream=stream,
        )


class CompiledGdnSummaryBwd:
    """Compiled gradient summary over the bprop summary kernel: the reverse state-gradient recurrence seeded by
    ``d_final_state`` (zero when unbound) writes ``d_initial_state``; a bound ``transition`` receives ``M^T`` from the
    recompute's transition arm.  ``chain`` summarizes every filled piece (G, M) and composes them with one reverse fp32
    state chain whose tail is ``d_initial_state`` and whose running product is ``transition``."""

    def __init__(self, node, summary_module):
        from .common.host import tensormap_workspace_bytes
        from .common.piece_chain import chain_rows_per_cta, choose_pieces, piece_table_layout
        from .kernel.gdn_chain_backward_f16 import build_chain_summary_backward, run_chain_summary_backward
        from .kernel.gdn_warmup_backward_f16 import SUMMARY_TAIL_BUFFERS, build_warmup_summary_backward, run_warmup_summary_backward
        from .common.split_k import WORK_ITEM_FIELDS, compute_ideal_chunks, max_work_items, split_scratch_rows
        from .common.tail import GATE_WORDS, choose_tail

        self.node = node
        self.chain_rows_per_cta = chain_rows_per_cta
        self.build_chain_summary_backward = build_chain_summary_backward
        self.run_chain_summary_backward = run_chain_summary_backward
        self.build_warmup_summary_backward = build_warmup_summary_backward
        self.run_warmup_summary_backward = run_warmup_summary_backward
        self.chain_launch = None
        self.warmup_launch = None
        self.tail_buffer_names = SUMMARY_TAIL_BUFFERS
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
        k_dim = node.inputs["k"].dim
        total = k_dim[0] * k_dim[1] if len(k_dim) == 4 else k_dim[0]
        HK = k_dim[-2]
        K, V = q.dim[-1], do.dim[-1]
        HQ = q.dim[1]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.io_name = "float16" if q.get_data_type().name == "HALF" else "bfloat16"
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"

        self.num_sm = multiprocessor_count(self.device)
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.overwrite_initial_state = bool(node.params.get("overwrite_initial_state", False))
        self.n_heads_out, self.total = HO, total
        self.num_seqs = B
        self.dim_k, self.dim_v = K, V
        self.checkpoint_cadence = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.pieces, self.unit_chunks = choose_pieces(
            num_seqs=B,
            heads_out=HO,
            num_sm=self.num_sm,
            total_tokens=total // self.num_householder,
            b_t=self.b_t,
            cadence_tokens=self.checkpoint_cadence,
            batch_invariant=self.batch_invariant,
            compose_tail=True,
            expand_num=self.num_householder,
            reverse=True,
        )
        self.has_context = bool(node.params.get("output_context", False))
        self.context = None
        if self.has_context:
            from .kernel import gdn_tinv_f16 as tinv_module

            self.context = backward_context(
                self.plan_name,
                full=True,
                total=total,
                num_householder=self.num_householder,
                h_q=HQ,
                h_k=HK,
                h_o=HO,
                d_k=K,
                d_v=V,
                b_t=self.b_t,
                io_name=self.io_name,
                num_sm=self.num_sm,
                cadence_tokens=self.checkpoint_cadence,
                batch_invariant=self.batch_invariant,
                use_qk_l2norm=self.use_qk_l2norm,
                tinv_rows=tinv_module.tinv_rows,
            )
        self.context_size = self.context.size if self.has_context else 0
        self.chain = self.pieces > 0
        self.tail = None
        if not self.chain and not self.batch_invariant:
            self.tail = choose_tail(
                num_seqs=B,
                heads_out=HO,
                num_sm=self.num_sm,
                total_tokens=total // self.num_householder,
                b_t=self.b_t,
                cadence_tokens=0,
                expand_num=self.num_householder,
            )
        self.split = not self.chain and not self.batch_invariant and not self.overwrite_initial_state
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        self.has_normalized_rows = self.has_context and self.context.has_rows
        self.has_chunk_factors = self.has_context and self.context.has_tables
        self.has_piece_tables = self.has_context and self.context.has_tables
        layout = WorkspaceLayout()
        regions = []
        if self.chain:
            from .kernel import gdn_recompute_f16 as recompute_module

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
                ("state_x", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)),
            ]
            if not self.has_piece_tables:
                regions.append(("state_g", layout.add(self.num_pieces * HO * V * K * 4), "float32", (self.num_pieces, HO, V, K)))
                regions.append(("state_m", layout.add(self.num_pieces * HO * K * K * 4), "float32", (self.num_pieces, HO, K, K)))
            summary_words = tensormap_workspace_bytes(summary_module, self.num_pieces, self.num_householder) // 8
            regions.append(("summary_tensormaps", layout.add(summary_words * 8, align=128), "int64", (summary_words,)))
            recompute_words = tensormap_workspace_bytes(recompute_module, self.num_pieces, self.num_householder) // 8
            regions.append(("recompute_tensormaps_m", layout.add(recompute_words * 8, align=128), "int64", (recompute_words,)))
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(total // self.num_householder, self.num_pieces, self.num_householder, self.b_t)
            if not self.has_chunk_factors:
                regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, self.num_pieces, self.num_householder) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
        else:
            off_scheduler = layout.add(16)
            tensormap_words = tensormap_workspace_bytes(summary_module, B, self.num_householder) // 8
            regions += [
                ("scheduler_main", off_scheduler, "int32", (2,)),
                ("scheduler_transition", off_scheduler + 8, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (4,)),
                ("tensormaps", layout.add(tensormap_words * 8), "int64", (tensormap_words,)),
            ]
            if self.has_transition:
                from .kernel import gdn_recompute_f16 as recompute_module

                recompute_words = tensormap_workspace_bytes(recompute_module, B, self.num_householder) // 8
                regions.append(("recompute_tensormaps", layout.add(recompute_words * 8, align=128), "int64", (recompute_words,)))
                regions.append(("transition_m", layout.add(B * HO * K * K * 4), "float32", (B, HO, K, K)))
            from .kernel import gdn_tinv_f16 as tinv_module

            rows = tinv_module.tinv_rows(total // self.num_householder, B, self.num_householder, self.b_t)
            regions.append(("tinv", layout.add(rows * HO * self.b_t * self.b_t * 2), self.io_name, (rows, HO, self.b_t, self.b_t)))
            tinv_words = tensormap_workspace_bytes(tinv_module, B, self.num_householder) // 8
            regions.append(("tinv_tensormaps", layout.add(tinv_words * 8, align=128), "int64", (tinv_words,)))
            regions.append(("tinv_rows", layout.add(rows * 16), "int32", (rows, 4)))
            regions.append(("tinv_row_count", layout.add(4), "int32", (1,)))
        if self.use_qk_l2norm and not self.has_normalized_rows:
            q_rows = total // self.num_householder
            regions.append(("q_n", layout.add(q_rows * HQ * K * 2), self.io_name, (q_rows, HQ, K)))
            regions.append(("k_n", layout.add(total * HK * K * 2), self.io_name, (total, HK, K)))
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
            if self.tail is not None:
                regions.append(("gate", layout.add(4 * GATE_WORDS, align=16), "int32", (GATE_WORDS,)))
            if self.split:
                self.chunk_scratch_rows = split_scratch_rows(total, B, HO, self.b_t)
                regions.append(("item_scratch", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
                regions.append(("chunk_scratch", layout.add(self.chunk_scratch_rows * HO * 4), "float32", (self.chunk_scratch_rows, HO)))
        if self.tail is not None:
            from .kernel import gdn_recompute_f16 as tail_recompute_module

            tail = self.tail
            self.tail_pieces_total = tail.num_seqs * tail.pieces
            off_tail_scheduler = layout.add(24)
            regions += [
                ("tail_scheduler_g", off_tail_scheduler, "int32", (2,)),
                ("tail_scheduler_m", off_tail_scheduler + 8, "int32", (2,)),
                ("tail_scheduler_all", off_tail_scheduler, "int32", (6,)),
            ]
            table = piece_table_layout(tail.num_seqs, tail.pieces, HO)
            off_tail_table = layout.add(table.nbytes, align=256)
            regions += [
                ("tail_main_rows", off_tail_table + table.main_rows, "int32", (tail.num_seqs + 1,)),
                ("tail_summary_rows", off_tail_table + table.summary_rows, "int32", (tail.num_seqs + 1,)),
                ("tail_cu_pieces", off_tail_table + table.cu_pieces, "int32", (self.tail_pieces_total + 1,)),
                ("tail_main_count", off_tail_table + table.main_count, "int32", (1,)),
                ("tail_summary_count", off_tail_table + table.summary_count, "int32", (1,)),
                ("tail_work_items", layout.add(table.item_rows * WORK_ITEM_FIELDS * 4), "int32", (table.item_rows, WORK_ITEM_FIELDS)),
                ("tail_state_x", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", (self.tail_pieces_total, HO, V, K)),
                ("tail_state_g", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", (self.tail_pieces_total, HO, V, K)),
                ("tail_state_m", layout.add(self.tail_pieces_total * HO * K * K * 4), "float32", (self.tail_pieces_total, HO, K, K)),
            ]
            words = tensormap_workspace_bytes(summary_module, self.tail_pieces_total, self.num_householder) // 8
            regions.append(("tail_summary_tensormaps", layout.add(words * 8, align=128), "int64", (words,)))
            words = tensormap_workspace_bytes(tail_recompute_module, self.tail_pieces_total, self.num_householder) // 8
            regions.append(("tail_recompute_tensormaps_m", layout.add(words * 8, align=128), "int64", (words,)))
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
        if self.tail is not None:
            positions = pos | {name: len(names) + i for i, name in enumerate(self.carve_names)}
            aliases = {
                "scheduler_summary": "tail_scheduler_g",
                "recompute_m_words": "tail_recompute_tensormaps_m",
                "bprop_summary_words": "tail_summary_tensormaps",
                "state_m_main": "tail_state_m",
                "state_m_chain": "tail_state_m",
                "state_g_chain": "tail_state_g",
                "state_x_chain": "tail_state_x",
                "dstate_in": "d_final_state",
                "dstate0": "d_initial_state",
            }
            absent = len(names) + len(self.carve_names)
            indices = [positions.get(aliases.get(name, "tail_" + name), absent) for name in self.tail_buffer_names]
            self.tail_buffers = itemgetter(*indices)
            self.tail_scalars = (int(self.tail.pieces), int(self.tail.num_seqs), int(self.tail.batch_base), int(self.tail.first_head_lo))

    def run(self, views, workspace, stream, context=None) -> None:
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
        context_region = self.context.regions(context) if self.has_context else {}
        q_n = context_region["q_n"] if self.has_normalized_rows else region.get("q_n")
        k_n = context_region["k_n"] if self.has_normalized_rows else region.get("k_n")
        inv_q = context_region["inv_q"] if self.has_normalized_rows else None
        inv_k = context_region["inv_k"] if self.has_normalized_rows else None
        tinv = context_region["tinv"] if self.has_chunk_factors else region.get("tinv")
        state_m = context_region["state_m"] if self.has_piece_tables else region.get("state_m")
        state_g = context_region["state_g"] if self.has_piece_tables else region.get("state_g")
        l2norm_buffers = (q, k, inv_q, inv_k) if self.use_qk_l2norm else (None,) * 4
        if self.use_qk_l2norm:
            q, k = q_n, k_n
        if self.chain:
            self.run_chain(q, k, g, beta, do, cu, dstate_in, dstate0, transition, a_log, dt_bias, region, stream, tinv, state_m, state_g, l2norm_buffers)
            return
        tail = self.tail
        buffers = dict(
            q=q,
            k=k,
            do=do,
            gate=g,
            beta=beta,
            a_log=a_log if self.safe_gate else None,
            dt_bias=dt_bias if self.safe_gate else None,
            cu_seqlens=cu,
            dstate0=dstate0,
            dstate_in=dstate_in,
            work_items=region["work_items"],
            work_count=region["work_count"],
            item_scratch=region.get("item_scratch"),
            chunk_scratch=region.get("chunk_scratch"),
            scheduler=region["scheduler_main"],
            scheduler_all=region["scheduler_all"],
            workspace=region["tensormaps"],
            tinv=tinv,
            tinv_words=region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            transition_m=region.get("transition_m"),
            transition=transition,
            scheduler_transition=region["scheduler_transition"] if self.has_transition else None,
            transition_words=region.get("recompute_tensormaps"),
            item_limit_word=region.get("gate"),
            l2norm_buffers=l2norm_buffers,
            tail_scalars=self.tail_scalars if tail is not None else (),
            tail_buffers=self.tail_buffers((*views, *region.values(), None)) if tail is not None else (),
        )
        if self.warmup_launch is None:
            self.warmup_launch = self.build_warmup_summary_backward(
                **buffers,
                tail=tail,
                tail_chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, tail.num_seqs, self.n_heads_out, self.num_sm) if tail is not None else 0,
                transition_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                split=self.split,
                n_tiles=self.n_tiles,
                ideal_chunks=self.ideal,
                num_sm=self.num_sm,
                b_t=self.b_t,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                expand_num=self.num_householder,
                scale=self.scale,
                device=self.device,
                stream=stream,
            )
        self.run_warmup_summary_backward(*self.warmup_launch, **buffers, scale=self.scale, stream=stream)

    def run_chain(self, q, k, g, beta, do, cu, dstate_in, dstate0, transition, a_log, dt_bias, region, stream, tinv, state_m, state_g, l2norm_buffers) -> None:
        """Chain prologue, T pass, M (recompute transition arm) and G (zero-seeded bprop summary) of every filled piece, one
        reverse fp32 state chain seeded with d_final_state whose tail is d_initial_state and whose running product is
        transition."""
        buffers = dict(
            q=q,
            k=k,
            do=do,
            gate=g,
            beta=beta,
            a_log=a_log if self.safe_gate else None,
            dt_bias=dt_bias if self.safe_gate else None,
            cu_seqlens=cu,
            cu_pieces=region["cu_pieces"],
            main_rows=region["main_rows"],
            summary_rows=region["summary_rows"],
            main_count=region["main_count"],
            summary_count=region["summary_count"],
            work_items=region["work_items"],
            scheduler_all=region["scheduler_all"],
            scheduler_m=region["scheduler_m"],
            scheduler_summary=region["scheduler_g"],
            tinv_words=region["tinv_tensormaps"],
            tinv_rows=region["tinv_rows"],
            tinv_row_count=region["tinv_row_count"],
            recompute_m_words=region["recompute_tensormaps_m"],
            bprop_summary_words=region["summary_tensormaps"],
            tinv=tinv,
            state_m=state_m,
            state_g=state_g,
            state_x=region["state_x"],
            dstate_in=dstate_in,
            dstate0=dstate0,
            transition=transition,
            l2norm_buffers=l2norm_buffers,
        )
        if self.chain_launch is None:
            self.chain_launch = self.build_chain_summary_backward(
                **buffers,
                pieces=self.pieces,
                heads_out=self.n_heads_out,
                num_seqs=self.num_seqs,
                unit_chunks=self.unit_chunks,
                b_t=self.b_t,
                expand_num=self.num_householder,
                length_rule=self.length_rule,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                scale=self.scale,
                chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
            )
        self.run_chain_summary_backward(
            self.chain_launch,
            **buffers,
            pieces=self.pieces,
            heads_out=self.n_heads_out,
            num_seqs=self.num_seqs,
            scale=self.scale,
            stream=stream,
        )
