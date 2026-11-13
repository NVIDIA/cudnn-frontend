# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST KDA engine: KDA / KDA_BWD nodes on the chunked prefill and backward kernels (SM100 / SM103 / SM107, bf16/fp16, BT=16);
the backward regenerates the checkpoint series with the recompute kernel when the graph carries none.  Long sequences on
few (sequence, head) tiles run as an exact piece chain (``common/piece_chain.py``)."""

from __future__ import annotations

import math
from operator import itemgetter


from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost.device import build_device, current_device, multiprocessor_count
from cudnn.frost.workspace import WorkspaceLayout, carve_plan
from ..graph_analyzer import analyze
from .common.context import backward_context
from .engine import FrostLaPlan, context_gate, frost_la_gate, summary_support_gates

# the d_v split takes the prep path up to this fraction of the SM count in (sequence, head) tiles (GB200 fit; Rubin retune pending)
PREP_TILE_FRACTION = 0.475


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

    Default KDA engine on SM100 / SM103 / SM107 (lowest KDA engine_id); serves KDA
    forward and KDA_BWD (with a forward checkpoint recompute when ``state_checkpoints`` is absent)."""

    name = "kda_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        import cudnn

        facts = graph._facts_for(analyze)
        frost_la_gate("KdaFrostEngine", facts, "KDA")
        if facts.is_bwd and facts.input_context:
            context_gate("KdaFrostEngine", facts, b_t=16)
        if facts.d_qk not in (64, 128):
            raise NotImplementedError(f"KdaFrostEngine: q/k head dim must be 64 or 128, got {facts.d_qk}")
        if facts.d_v not in (64, 128):
            raise NotImplementedError(f"KdaFrostEngine: v head dim must be 64 or 128, got {facts.d_v}")
        if facts.gate_channels != facts.d_qk:
            raise NotImplementedError(f"KdaFrostEngine: g must carry d_qk = {facts.d_qk} channels, got {facts.gate_channels}")
        checkpoint = facts.checkpoint_every_n_tokens
        checkpoint_dtypes = (cudnn.data_type.FLOAT, facts.io_dtype)
        if not facts.is_bwd:
            checkpoint_dtypes = checkpoint_dtypes + (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2)
        if checkpoint and checkpoint % 16 != 0:
            raise NotImplementedError(f"KdaFrostEngine: checkpoint_every_n_tokens must be a positive multiple of 16 (got {checkpoint})")
        if not facts.gates_at_ho:
            raise NotImplementedError(f"KdaFrostEngine: g/beta must carry HO = max(q, v) heads ({facts.h_o})")
        if facts.has_state_indices:
            if facts.state_indices_dtype not in (cudnn.data_type.INT32, cudnn.data_type.INT64):
                raise NotImplementedError(f"KdaFrostEngine: 'state_indices' must be int32 or int64, got {facts.state_indices_dtype}")
            if facts.is_bwd:
                raise NotImplementedError(f"KdaFrostEngine: 'state_indices' is a forward-only pool addressing mode")
        beta_wants = (cudnn.data_type.FLOAT, facts.io_dtype)
        if facts.beta_dtype not in beta_wants + (None,):
            raise NotImplementedError(f"KdaFrostEngine: 'beta' must be {' or '.join(str(w) for w in beta_wants)}, got {facts.beta_dtype}")
        state_dtypes = (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
        fp8 = (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2)
        forward_state_dtypes = state_dtypes if facts.is_bwd else state_dtypes + fp8
        for port, got in (("initial_state", facts.state_dtype), ("final_state", facts.final_state_dtype)):
            if got not in forward_state_dtypes + (None,):
                raise NotImplementedError(f"KdaFrostEngine: '{port}' must be fp32/bf16/fp16 (fp8 in a forward graph), got {got}")
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
            if facts.do_dtype not in (facts.io_dtype, None):
                raise NotImplementedError("KdaFrostEngine: 'dO' must match the io dtype")
            if facts.state_checkpoints_dtype not in checkpoint_dtypes + (None,):
                raise NotImplementedError(
                    f"KdaFrostEngine: 'state_checkpoints' must be {' or '.join(str(w) for w in checkpoint_dtypes)}, got {facts.state_checkpoints_dtype}"
                )
            if facts.dbeta_dtype not in (facts.beta_dtype, None):
                raise NotImplementedError(f"KdaFrostEngine: 'dBeta' must match 'beta' ({facts.beta_dtype}), got {facts.dbeta_dtype}")
        elif facts.io_dtype is not None and facts.state_checkpoints_out_dtype not in checkpoint_dtypes + (None,):
            raise NotImplementedError(
                f"KdaFrostEngine: 'state_checkpoints' must be {' or '.join(str(w) for w in checkpoint_dtypes)}, got {facts.state_checkpoints_out_dtype}"
            )

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_kda(graph))


class CompiledKda:
    """Compiled FROST KDA plan over the resolved node buffers.  ``choose_pieces`` and ``is_dv_split`` fix the scheme at build: ``uncut``
    (one item per sequence and head), ``dv_split`` (two uncut items per sequence and head, on the prep path up to ``PREP_TILE_FRACTION`` of the SM count in tiles,
    each owning half of d_v) or ``chain`` (per-piece H and M from the fused
    summary, an fp32 state chain seeding every piece, the prefill over the pieces storing every piece's final state, the
    last filled piece gathered into ``final_state``).  A rectangular state chains like a square one: the transition
    summary runs k in place of v, so M is (DK, DK) while H and X are (DV, DK)."""

    def __init__(self, node, kernel_module):
        """Resolve graph attributes and select the forward scheduling family."""
        from cuda.bindings.driver import CUstream

        from .common.host import tensormap_workspace_bytes
        from .common.piece_chain import (
            DV_SPLIT_TILES,
            chain_rows_per_cta,
            choose_pieces,
            is_dv_split,
            piece_table_layout,
        )
        from .kernel.kda_chain_forward_f16 import CHAIN_FORWARD_BUFFERS, build_chain_forward, run_chain_forward
        from .kernel.kda_uncut_forward_f16 import TAIL_BUFFERS, build_uncut_forward, run_uncut_forward
        from .common.split_k import WORK_ITEM_FIELDS
        from .common.tail import GATE_WORDS, choose_tail

        self.node = node
        self.stream_type = CUstream
        self.chain_rows_per_cta = chain_rows_per_cta
        self.uncut_launch = None
        self.tail_buffer_names = TAIL_BUFFERS
        self.build_uncut_forward = build_uncut_forward
        self.run_uncut_forward = run_uncut_forward
        self.build_chain_forward = build_chain_forward
        self.run_chain_forward = run_chain_forward
        self.chain_launch = None
        self.chain_buffer_names = CHAIN_FORWARD_BUFFERS
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
        self.checkpoint_after_block = bool(node.params.get("checkpoint_after_block", False))
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.overwrite_initial_state = bool(node.params.get("overwrite_initial_state", False))
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
        self.dv_split = (
            not self.chain
            and not self.batch_invariant
            and is_dv_split(
                num_seqs=B,
                heads_out=HO,
                dim_v=V,
                num_sm=self.num_sm,
                total_tokens=total,
                b_t=self.b_t,
                expand_num=1,
                unit_chunks=self.unit_chunks,
            )
        )
        self.tiles_per_head = DV_SPLIT_TILES if self.dv_split else 1
        self.prep = self.dv_split and B * HO <= int(PREP_TILE_FRACTION * self.num_sm)
        self.tail = None
        if not self.chain and not self.dv_split and not self.batch_invariant:
            self.tail = choose_tail(
                num_seqs=B,
                heads_out=HO,
                num_sm=self.num_sm,
                total_tokens=total,
                b_t=self.b_t,
                cadence_tokens=self.checkpoint if self.has_state_checkpoints else 0,
                expand_num=1,
                dim_v=V,
            )
            if self.tail is not None and self.tail.batch_base == 0 and ("preceding_state_update" in node.inputs or "composed_initial_state" in node.inputs):
                self.tail = None
        self.tail_prep = self.tail is not None and self.tail.dv_split and self.tail.tiles <= int(PREP_TILE_FRACTION * self.num_sm)
        self.io_name = "float16" if q.get_data_type().name == "HALF" else "bfloat16"
        self.length_rule = self.chain and self.batch_invariant

        from .common.host import get_dtype
        from .common.steep_decay import STEEP_OFF, steep_mode_for

        self.steep_mode = steep_mode_for(get_dtype(self.io_name), self.use_qk_l2norm, self.safe_gate, self.gate_lower_bound * kernel_module.LOG2_E, self.b_t)
        layout = WorkspaceLayout()
        regions = []
        if self.chain:
            self.num_pieces = B * self.pieces
            self.n_tiles = self.num_pieces * HO
            self.work_item_rows = self.n_tiles
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

            self.fused_tensormap_bytes = tensormap_workspace_bytes(summary_module, self.num_pieces)
            regions.append(("fused_tensormaps", layout.add(self.fused_tensormap_bytes, align=128), "int64", (self.fused_tensormap_bytes // 8,)))
        else:
            self.num_pieces = B
            regions.append(("scheduler", layout.add(8), "int32", (2,)))
            self.n_tiles = B * HO * self.tiles_per_head
            self.work_item_rows = self.n_tiles
            regions.append(("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
            regions.append(("work_count", layout.add(4), "int32", (1,)))
            if self.tail is not None:
                regions.append(("gate", layout.add(4 * GATE_WORDS, align=16), "int32", (GATE_WORDS,)))
            if self.prep:
                from .kernel import kda_prep_f16 as prep_module
                from .kernel.gdn_tinv_f16 import tinv_rows

                rows = tinv_rows(total, B, 1, self.b_t)
                for name in ("prep_k_decay", "prep_q_decay", "prep_t"):
                    regions.append((name, layout.add(rows * HO * self.b_t * K * 2, align=128), self.io_name, (rows, HO, self.b_t, K)))
                regions.append(("prep_a", layout.add(rows * HO * self.b_t * self.b_t * 2, align=128), "int32", (rows, HO, self.b_t * self.b_t // 2)))
                regions.append(("prep_diag", layout.add(rows * HO * K * 4, align=128), "float32", (rows, HO, K)))
                prep_words = tensormap_workspace_bytes(prep_module, B) // 8
                regions.append(("prep_tensormaps", layout.add(prep_words * 8, align=128), "int64", (prep_words,)))
                regions.append(("prep_rows", layout.add(rows * 16), "int32", (rows, 4)))
                regions.append(("prep_row_count", layout.add(4), "int32", (1,)))
        if self.prep:
            from .kernel import kda_prep_prefill_f16 as prefill_module
        else:
            prefill_module = kernel_module
        self.tensormap_bytes = tensormap_workspace_bytes(prefill_module, self.num_pieces)
        regions.append(("tensormaps", layout.add(self.tensormap_bytes, align=128), "int64", (self.tensormap_bytes // 8,)))
        if self.tail is not None and self.tail.dv_split:
            tail = self.tail
            self.tail_ctas = tail.tiles * DV_SPLIT_TILES
            self.tail_item_base = tail.prefix_tiles * DV_SPLIT_TILES
            regions.append(("tail_scheduler", layout.add(8), "int32", (2,)))
            regions.append(("tail_work_items", layout.add(self.tail_ctas * WORK_ITEM_FIELDS * 4), "int32", (self.tail_ctas, WORK_ITEM_FIELDS)))
            regions.append(("tail_work_count", layout.add(4), "int32", (1,)))
            if self.tail_prep:
                from .kernel import kda_prep_f16 as tail_prep_module
                from .kernel import kda_prep_prefill_f16 as tail_prefill_module
                from .kernel.gdn_tinv_f16 import tinv_rows

                rows = tinv_rows(total, B, 1, self.b_t)
                for name in ("tail_prep_k_decay", "tail_prep_q_decay", "tail_prep_t"):
                    regions.append((name, layout.add(rows * HO * self.b_t * K * 2, align=128), self.io_name, (rows, HO, self.b_t, K)))
                regions.append(("tail_prep_a", layout.add(rows * HO * self.b_t * self.b_t * 2, align=128), "int32", (rows, HO, self.b_t * self.b_t // 2)))
                regions.append(("tail_prep_diag", layout.add(rows * HO * K * 4, align=128), "float32", (rows, HO, K)))
                tail_prep_words = tensormap_workspace_bytes(tail_prep_module, B) // 8
                regions.append(("tail_prep_tensormaps", layout.add(tail_prep_words * 8, align=128), "int64", (tail_prep_words,)))
                regions.append(("tail_prep_rows", layout.add(rows * 16), "int32", (rows, 4)))
                regions.append(("tail_prep_row_count", layout.add(4), "int32", (1,)))
            else:
                tail_prefill_module = kernel_module
            tail_bytes = tensormap_workspace_bytes(tail_prefill_module, B)
            regions.append(("tail_tensormaps", layout.add(tail_bytes, align=128), "int64", (tail_bytes // 8,)))
        elif self.tail is not None:
            from .kernel import kda_summary_f16 as tail_summary_module

            tail = self.tail
            self.tail_pieces_total = tail.num_seqs * tail.pieces
            tail_tiles = self.tail_pieces_total * HO
            off_tail_scheduler = layout.add(24)
            regions += [
                ("tail_scheduler_prefill", off_tail_scheduler, "int32", (2,)),
                ("tail_scheduler_h", off_tail_scheduler + 8, "int32", (2,)),
                ("tail_scheduler_m", off_tail_scheduler + 16, "int32", (2,)),
                ("tail_scheduler_all", off_tail_scheduler, "int32", (6,)),
                ("tail_work_items", layout.add(tail_tiles * WORK_ITEM_FIELDS * 4), "int32", (tail_tiles, WORK_ITEM_FIELDS)),
            ]
            table = piece_table_layout(tail.num_seqs, tail.pieces, HO)
            off_tail_table = layout.add(table.nbytes, align=256)
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
            tail_fused_bytes = tensormap_workspace_bytes(tail_summary_module, self.tail_pieces_total)
            regions.append(("tail_fused_tensormaps", layout.add(tail_fused_bytes, align=128), "int64", (tail_fused_bytes // 8,)))
            tail_bytes = tensormap_workspace_bytes(kernel_module, self.tail_pieces_total)
            regions.append(("tail_tensormaps", layout.add(tail_bytes, align=128), "int64", (tail_bytes // 8,)))
        seed = node.inputs.get("initial_state")
        if (
            self.steep_mode != STEEP_OFF
            and not self.chain
            and self.has_final_state
            and not self.prep
            and seed is not None
            and (self.overwrite_initial_state or "state_indices" in node.inputs)
        ):
            # The final state may overwrite the seeds (the graph contract, not this call's pointers: buffers can alias
            # after warmup), so the prefill seeds from a copy its exact twin can reread (common/state_snapshot.py).
            seed_name, seed_bytes = {"BFLOAT16": ("bfloat16", 2), "HALF": ("float16", 2)}.get(seed.get_data_type().name, ("float32", 4))
            regions.append(("state_snapshot", layout.add(B * HO * V * K * seed_bytes, align=128), seed_name, (B, HO, V, K)))
        self.workspace_size = layout.size
        self.workspace_regions = tuple(regions)
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
            aliases = {
                "gate": "g",
                "o": "O",
                "seed": "initial_state",
                "composed": "composed_initial_state",
                "seed_indices": "state_indices",
                "final_indices": "state_indices",
                "checkpoints": "state_checkpoints",
                "summary_words": "fused_tensormaps",
                "prefill_words": "tensormaps",
                "scheduler_summary": "scheduler_h",
            }
            absent = len(names) + len(self.carve_names)
            if not self.safe_gate:
                positions.pop("a_log", None)
                positions.pop("dt_bias", None)
            if not (self.has_state_checkpoints and self.checkpoint):
                positions.pop("state_checkpoints", None)
            indices = [positions.get(aliases.get(name, name), absent) for name in self.chain_buffer_names]
            self.chain_buffers = itemgetter(*indices)
            self.chain_scalars = (
                int(self.pieces),
                int(self.n_heads_out),
                int(self.num_seqs),
                int(self.checkpoint if self.has_state_checkpoints else 0),
                float(self.scale),
                0,
                0,
            )
        if self.tail is not None:
            positions = pos | {name: len(names) + i for i, name in enumerate(self.carve_names)}
            aliases = {
                "cu_pieces_main": "tail_cu_pieces",
                "scheduler_summary": "tail_scheduler_h",
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
                "prep_words": "tail_prep_tensormaps",
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

    def host_config(self):
        """The jit host of the plan's scheme and its constexpr arguments by parameter name, built from the node's tensors for a
        caller that traces the host itself (every data type set); the compiled launch derives them from its buffers."""
        from .common.host import get_dtype
        from .kernel import kda_chain_forward_f16, kda_prefill_f16, kda_uncut_forward_f16

        node = self.node
        dtype_names = {"BFLOAT16": "bfloat16", "HALF": "float16"}
        io_dtype = get_dtype(self.io_name)
        gate_dtype = get_dtype(dtype_names.get(node.inputs["g"].get_data_type().name, "float32"))
        dim_k, dim_v = node.inputs["q"].dim[2], node.inputs["v"].dim[2]
        checkpoint = self.checkpoint if self.has_state_checkpoints else 0
        flags = dict(
            store_final_state=self.index_final_state is not None,
            enable_checkpoints=checkpoint > 0,
            checkpoint_dtype=get_dtype(dtype_names.get(node.outputs["state_checkpoints"].get_data_type().name, "float32")) if checkpoint > 0 else None,
            checkpoint_after_block=self.checkpoint_after_block,
            l2norm=self.use_qk_l2norm,
            safe_gate=self.safe_gate,
            gate_scale_log2=self.gate_lower_bound * kda_prefill_f16.LOG2_E,
            log_gate=self.log_gate,
            beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            max_active_clusters=self.num_sm,
            d_k=dim_k,
            d_v=dim_v,
        )
        if self.chain:
            summary_cfg, prefill_cfg = kda_chain_forward_f16.build_configs(io_dtype, get_dtype("float32"), gate_dtype, **flags)
            return kda_chain_forward_f16.chain_forward_host, dict(
                unit_chunks=self.unit_chunks,
                b_t=self.b_t,
                length_rule=self.length_rule,
                summary_cfg=summary_cfg,
                prefill_cfg=prefill_cfg,
                dim_v=dim_v,
                dim_k=dim_k,
                chain_rows=self.chain_rows_per_cta(dim_v, dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                has_seed=self.index_initial_state is not None,
                tail=False,
            )
        state = node.inputs.get("initial_state", node.outputs.get("final_state"))
        prefill_cfg, prep_cfg = kda_uncut_forward_f16.build_configs(
            io_dtype,
            get_dtype(dtype_names.get(state.get_data_type().name, "float32") if state is not None else "float32"),
            gate_dtype,
            use_initial_state=self.index_initial_state is not None or self.index_composed is not None,
            state_table=self.index_initial_state is not None,
            has_composed=self.index_composed is not None,
            tiles_per_head=self.tiles_per_head,
            prep=self.prep,
            **flags,
        )
        return kda_uncut_forward_f16.uncut_forward_host, dict(
            b_t=self.b_t,
            num_sms=self.num_sm,
            io_dtype=io_dtype,
            prefill_cfg=prefill_cfg,
            tiles_per_head=self.tiles_per_head,
            prep=self.prep,
            prep_cfg=prep_cfg,
        )

    def run(self, views, workspace, stream) -> None:
        composed = views[self.index_composed] if self.index_composed is not None else None
        fold_buffers = (None,) * 4
        if self.index_preceding_state_update is not None:
            fold_buffers = (
                views[self.index_preceding_state_update],
                views[self.index_preceding_transition],
                views[self.index_initial_state] if self.index_initial_state is not None else None,
                composed,
            )
        if self.chain and self.chain_launch is not None:
            buffers = self.chain_buffers((*views, *workspace.carve(self.carve), None))
            self.chain_launch(*self.chain_scalars, *buffers, self.stream_type(int(stream or 0)), None, *fold_buffers)
            return
        q = views[self.index_q]
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
        if self.chain:
            self.run_chain(
                q, k, v, g, beta, cu, state0, composed, o, final_state, state_checkpoints, a_log, dt_bias, state_indices, region, stream, fold_buffers
            )
            return
        checkpoint = self.checkpoint if self.has_state_checkpoints else 0
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
            checkpoints=state_checkpoints if self.has_state_checkpoints else None,
            work_items=region["work_items"],
            work_count=region["work_count"],
            scheduler=region["scheduler"],
            workspace=region["tensormaps"],
            prep_k_decay=region.get("prep_k_decay"),
            prep_q_decay=region.get("prep_q_decay"),
            prep_t=region.get("prep_t"),
            prep_a=region.get("prep_a"),
            prep_diag=region.get("prep_diag"),
            prep_words=region.get("prep_tensormaps"),
            prep_rows=region.get("prep_rows"),
            prep_row_count=region.get("prep_row_count"),
            state_snapshot=region.get("state_snapshot"),
            item_limit_word=region.get("gate"),
            tail_scalars=self.tail_scalars if self.tail is not None else (),
            tail_buffers=self.tail_buffers((*views, *region.values(), None)) if self.tail is not None else (),
            fold_buffers=fold_buffers,
        )
        tail = self.tail
        if self.uncut_launch is None:
            self.uncut_launch = self.build_uncut_forward(
                **buffers,
                tail=tail,
                tail_chain_rows=(
                    self.chain_rows_per_cta(v.shape[2], q.shape[2], tail.num_seqs, self.n_heads_out, self.num_sm)
                    if tail is not None and not tail.dv_split
                    else 0
                ),
                tail_prep=self.tail_prep,
                tiles_per_head=self.tiles_per_head,
                prep=self.prep,
                num_sm=self.num_sm,
                b_t=self.b_t,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                use_qk_l2norm=self.use_qk_l2norm,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                checkpoint_every_n_tokens=checkpoint,
                checkpoint_after_block=self.checkpoint_after_block,
                scale=self.scale,
                device=self.device,
                stream=stream,
            )
        self.run_uncut_forward(self.uncut_launch, **buffers, checkpoint_every_n_tokens=checkpoint, scale=self.scale, stream=stream)

    def run_chain(
        self, q, k, v, g, beta, cu, state0, composed, o, final_state, state_checkpoints, a_log, dt_bias, state_indices, region, stream, fold_buffers
    ) -> None:
        """Chain prologue, fused H and M summaries, fp32 state chain seeded with initial_state, prefill over the pieces seeded
        with X, all four launched from one compiled host (kernel/kda_chain_forward_f16.py); the checkpoint series keeps its
        unsplit layout."""
        state_h = region["state_h"]
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
            summary_words=region["fused_tensormaps"],
            prefill_words=region["tensormaps"],
            state_h=state_h,
            state_m=region["state_m"],
            state_x=region["state_x"],
            seed=state0,
            composed=composed,
            seed_indices=state_indices,
            final_state=final_state,
            final_indices=state_indices,
            checkpoints=state_checkpoints if self.has_state_checkpoints and self.checkpoint else None,
            fold_buffers=fold_buffers,
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
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                use_qk_l2norm=self.use_qk_l2norm,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                checkpoint_every_n_tokens=checkpoint,
                checkpoint_after_block=self.checkpoint_after_block,
                scale=self.scale,
                chain_rows=self.chain_rows_per_cta(state_h.shape[2], state_h.shape[3], self.num_seqs, self.n_heads_out, self.num_sm),
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
            scale=self.scale,
            stream=stream,
        )


class CompiledKdaBwd:
    """Compiled FROST KDA backward plan: one recompute pass regenerates the checkpoint series into workspace when the
    graph carries none; GVA/GQA gradients land in HO-head scratch and reduce to the native head counts.  ``chain`` runs
    the pieces as independent sequences: H, M, X from the fused summary and the forward chain (M alone when the series
    is passed back), G from the bprop summary, a reverse fp32 chain seeding every piece's outgoing gradient, piece 0 of
    every sequence gathered into ``d_initial_state``."""

    def __init__(self, node, bwd_module, recompute_module):
        """Resolve graph attributes and select the backward scheduling family."""
        from .common.gate_bwd import GATE_BWD_BLOCKS
        from .common.host import tensormap_workspace_bytes
        from .common.piece_chain import chain_rows_per_cta, choose_pieces, piece_table_layout
        from .kernel.kda_chain_backward_f16 import build_chain_backward, run_chain_backward
        from .kernel.kda_uncut_backward_f16 import TAIL_BUFFERS, build_uncut_backward, run_uncut_backward
        from .common.split_k import WORK_ITEM_FIELDS, compute_ideal_chunks
        from .common.tail import GATE_WORDS, choose_tail

        self.node = node
        self.chain_rows_per_cta = chain_rows_per_cta
        self.build_chain_backward = build_chain_backward
        self.run_chain_backward = run_chain_backward
        self.build_uncut_backward = build_uncut_backward
        self.run_uncut_backward = run_uncut_backward
        self.chain_launch = None
        self.uncut_launch = None
        self.tail_buffer_names = TAIL_BUFFERS
        self.plan_name = "KdaFrostEngine (KDA_BWD)"
        self.device = current_device()
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
        after_block = bool(node.params.get("checkpoint_after_block", False))
        self.has_state_checkpoints = "state_checkpoints" in node.inputs and not after_block
        self.has_dstate0 = "d_initial_state" in node.outputs
        self.has_a_log = "a_log" in node.inputs
        self.has_dt_bias = "dt_bias" in node.inputs
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"

        q, g, v = node.inputs["q"], node.inputs["g"], node.inputs["v"]
        self.b_t = bwd_module.CFG.B_T
        self.checkpoint_cadence = 0 if after_block else int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.coarse_checkpoints = self.has_state_checkpoints and (
            self.checkpoint_cadence > self.b_t or node.inputs["state_checkpoints"].get_data_type().name == "FLOAT"
        )
        self.needs_recompute = not self.has_state_checkpoints or self.coarse_checkpoints
        total = q.dim[0]
        HQ, HV = q.dim[1], v.dim[1]
        HK = node.inputs["k"].dim[1]
        HO = g.dim[1]
        K, V = q.dim[-1], v.dim[-1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.io_name = "float16" if node.inputs["q"].get_data_type().name == "HALF" else "bfloat16"
        self.n_heads_out, self.total = HO, total
        self.dim_k, self.dim_v = K, V
        self.num_sm = multiprocessor_count(self.device)
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.overwrite_initial_state = bool(node.params.get("overwrite_initial_state", False))
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
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        self.fused_h_m = self.chain and not self.has_state_checkpoints
        self.has_context = bool(node.params.get("input_context", False))
        self.context = (
            backward_context(
                self.plan_name,
                full=False,
                total=total,
                num_householder=1,
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
            )
            if self.has_context
            else None
        )
        self.context_size = self.context.size if self.has_context else 0
        same_cut = (
            self.has_context and self.context.has_tables and B == 1 and (self.pieces, self.unit_chunks) == (self.context.pieces, self.context.unit_chunks)
        )
        self.has_piece_tables = same_cut and self.chain
        self.has_piece_transitions = self.has_piece_tables and not self.fused_h_m

        from .common.host import get_dtype
        from .common.steep_decay import STEEP_OFF, steep_mode_for

        self.tail = None
        if not self.chain and not self.batch_invariant and not self.overwrite_initial_state:
            self.tail = choose_tail(
                num_seqs=B,
                heads_out=HO,
                num_sm=self.num_sm,
                total_tokens=total,
                b_t=self.b_t,
                cadence_tokens=self.checkpoint_cadence,
                expand_num=1,
            )
            if self.tail is not None and self.tail.batch_base == 0 and ("preceding_state_update" in node.inputs or "composed_initial_state" in node.inputs):
                self.tail = None

        layout = WorkspaceLayout()
        regions = []
        # One FP32 state and bounded-key ring per persistent CTA; sequential backward kernels share the arena.
        self.steep_mode = steep_mode_for(get_dtype(self.io_name), self.use_qk_l2norm, self.safe_gate, self.gate_lower_bound * bwd_module.LOG2_E, self.b_t)
        dstate_scratch_elements = (
            max(1, min(self.num_sm, self.num_pieces * HO)) * (V * K + bwd_module.CFG.SMEM_DECAY_STAGES * self.b_t * K) if self.steep_mode != STEEP_OFF else 1
        )
        regions.append(("dstate_scratch", layout.add(dstate_scratch_elements * 4), "float32", (dstate_scratch_elements,)))
        off_scheduler = layout.add(40 if self.chain else 16)
        regions += [
            ("scheduler_recompute", off_scheduler, "int32", (2,)),
            ("scheduler_bwd", off_scheduler + 8, "int32", (2,)),
        ]
        if self.fused_h_m:
            from .kernel import kda_summary_f16 as fused_module

            fused_tensormap_bytes = tensormap_workspace_bytes(fused_module, self.num_pieces)
            regions.append(("fused_tensormaps", layout.add(fused_tensormap_bytes, align=128), "int64", (fused_tensormap_bytes // 8,)))
        if self.chain:
            from .kernel import kda_bprop_summary_f16 as summary_module

            summary_tensormap_bytes = tensormap_workspace_bytes(summary_module, self.num_pieces)
            regions += [
                ("scheduler_summary", off_scheduler + 16, "int32", (2,)),
                ("scheduler_m", off_scheduler + 24, "int32", (2,)),
                ("scheduler_series", off_scheduler + 32, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (10,)),
                ("summary_tensormaps", layout.add(summary_tensormap_bytes, align=128), "int64", (summary_tensormap_bytes // 8,)),
            ]
        else:
            regions.append(("scheduler_all", off_scheduler, "int32", (4,)))
        self.n_tiles = self.num_pieces * HO
        self.work_item_rows = self.n_tiles
        regions.append(("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
        regions.append(("work_count", layout.add(4), "int32", (1,)))
        if self.tail is not None:
            regions.append(("gate", layout.add(4 * GATE_WORDS, align=16), "int32", (GATE_WORDS,)))
        if "following_state_grad" in node.inputs:
            regions.append(("composed_dstate", layout.add(HO * V * K * 4, align=16), "float32", (1, HO, V, K)))
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
                ("state_dx_end", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape),
            ]
            if not self.has_piece_transitions:
                regions.append(("state_m", layout.add(self.num_pieces * HO * K * K * 4), "float32", (self.num_pieces, HO, K, K)))
            if not self.has_piece_tables:
                regions.append(("state_g", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape))
            if not self.has_state_checkpoints:
                regions.append(("state_h", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape))
                regions.append(("state_x", layout.add(self.num_pieces * HO * V * K * 4), "float32", state_shape))
        if self.tail is not None:
            from .kernel import kda_bprop_summary_f16 as tail_summary_module
            from .kernel import kda_summary_f16 as tail_fused_module

            tail = self.tail
            self.tail_pieces_total = tail.num_seqs * tail.pieces
            tail_tiles = self.tail_pieces_total * HO
            tail_state_shape = (self.tail_pieces_total, HO, V, K)
            off_tail_scheduler = layout.add(40)
            regions += [
                ("tail_scheduler_recompute", off_tail_scheduler, "int32", (2,)),
                ("tail_scheduler_bwd", off_tail_scheduler + 8, "int32", (2,)),
                ("tail_scheduler_summary", off_tail_scheduler + 16, "int32", (2,)),
                ("tail_scheduler_m", off_tail_scheduler + 24, "int32", (2,)),
                ("tail_scheduler_series", off_tail_scheduler + 32, "int32", (2,)),
                ("tail_scheduler_all", off_tail_scheduler, "int32", (10,)),
                ("tail_work_items", layout.add(tail_tiles * WORK_ITEM_FIELDS * 4), "int32", (tail_tiles, WORK_ITEM_FIELDS)),
            ]
            tail_summary_bytes = tensormap_workspace_bytes(tail_summary_module, self.tail_pieces_total)
            regions.append(("tail_summary_tensormaps", layout.add(tail_summary_bytes, align=128), "int64", (tail_summary_bytes // 8,)))
            if not self.has_state_checkpoints:
                tail_fused_bytes = tensormap_workspace_bytes(tail_fused_module, self.tail_pieces_total)
                regions.append(("tail_fused_tensormaps", layout.add(tail_fused_bytes, align=128), "int64", (tail_fused_bytes // 8,)))
            tail_recompute_bytes = tensormap_workspace_bytes(recompute_module, self.tail_pieces_total)
            regions.append(("tail_recompute_tensormaps_m", layout.add(tail_recompute_bytes, align=128), "int64", (tail_recompute_bytes // 8,)))
            if self.needs_recompute:
                regions.append(("tail_recompute_tensormaps_series", layout.add(tail_recompute_bytes, align=128), "int64", (tail_recompute_bytes // 8,)))
            if self.coarse_checkpoints:
                tail_recompute_item_rows = max((total // self.recompute_span_tokens + 2 * self.tail_pieces_total) * HO, 1)
                regions.append(
                    (
                        "tail_work_items_recompute",
                        layout.add(tail_recompute_item_rows * WORK_ITEM_FIELDS * 4),
                        "int32",
                        (tail_recompute_item_rows, WORK_ITEM_FIELDS),
                    )
                )
                regions.append(("tail_work_count_recompute", layout.add(4), "int32", (1,)))
                regions.append(("tail_seed_row_base", layout.add(4), "int32", (1,)))
            table = piece_table_layout(tail.num_seqs, tail.pieces, HO)
            off_tail_table = layout.add(table.nbytes, align=256)
            regions += [
                ("tail_main_rows", off_tail_table + table.main_rows, "int32", (tail.num_seqs + 1,)),
                ("tail_summary_rows", off_tail_table + table.summary_rows, "int32", (tail.num_seqs + 1,)),
                ("tail_cu_pieces", off_tail_table + table.cu_pieces, "int32", (self.tail_pieces_total + 1,)),
                ("tail_main_count", off_tail_table + table.main_count, "int32", (1,)),
                ("tail_summary_count", off_tail_table + table.summary_count, "int32", (1,)),
                ("tail_work_items_summary", layout.add(table.item_rows * WORK_ITEM_FIELDS * 4), "int32", (table.item_rows, WORK_ITEM_FIELDS)),
                ("tail_state_m", layout.add(self.tail_pieces_total * HO * K * K * 4), "float32", (self.tail_pieces_total, HO, K, K)),
                ("tail_state_g", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", tail_state_shape),
                ("tail_state_dx_end", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", tail_state_shape),
            ]
            if not self.has_state_checkpoints:
                regions.append(("tail_state_h", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", tail_state_shape))
                regions.append(("tail_state_x", layout.add(self.tail_pieces_total * HO * V * K * 4), "float32", tail_state_shape))
            tail_bwd_bytes = tensormap_workspace_bytes(bwd_module, self.tail_pieces_total)
            regions.append(("tail_bwd_tensormaps", layout.add(tail_bwd_bytes, align=128), "int64", (tail_bwd_bytes // 8,)))
        self.bwd_tensormap_bytes = tensormap_workspace_bytes(bwd_module, self.num_pieces)
        regions.append(("bwd_tensormaps", layout.add(self.bwd_tensormap_bytes, align=128), "int64", (self.bwd_tensormap_bytes // 8,)))
        d_final = node.inputs.get("d_final_state")
        if self.steep_mode != STEEP_OFF and not self.chain and self.overwrite_initial_state and d_final is not None:
            # d_initial_state overwrites d_final_state: the bprop reads a copy so its exact twin can redo it.
            d_name, d_bytes = {"BFLOAT16": ("bfloat16", 2), "HALF": ("float16", 2)}.get(d_final.get_data_type().name, ("float32", 4))
            regions.append(("dstate_snapshot", layout.add(B * HO * V * K * d_bytes, align=128), d_name, (B, HO, V, K)))
        self.workspace_size = layout.size
        self.workspace_regions = tuple(regions)
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
        self.index_following_state_grad = pos.get("following_state_grad")
        self.index_following_transition = pos.get("following_transition")
        self.index_composed = pos.get("composed_initial_state")
        self.index_d_a_log = pos.get("d_a_log")
        self.index_d_dt_bias = pos.get("d_dt_bias")
        if self.tail is not None:
            positions = pos | {name: len(names) + i for i, name in enumerate(self.carve_names)}
            coarse = self.coarse_checkpoints
            aliases = {
                "cu_pieces_main": "tail_cu_pieces",
                "series_items": "tail_work_items_recompute",
                "series_count": "tail_work_count_recompute",
                "recompute_items": ("tail_work_items_recompute" if coarse else "tail_work_items") if self.needs_recompute else None,
                "recompute_count": ("tail_work_count_recompute" if coarse else "tail_main_count") if self.needs_recompute else None,
                "summary_words": "tail_fused_tensormaps",
                "recompute_m_words": "tail_recompute_tensormaps_m",
                "series_words": "tail_recompute_tensormaps_series",
                "bprop_summary_words": "tail_summary_tensormaps",
                "bprop_words": "tail_bwd_tensormaps",
                "state_h_summary": "tail_state_h",
                "state_m_main": "tail_state_m",
                "state_h_chain": "tail_state_h",
                "state_m_chain": "tail_state_m",
                "state_x_chain": "tail_state_x",
                "seed": "initial_state",
                "state_x_series": "tail_state_x",
                "state_g_chain": "tail_state_g",
                "state_dx_end_chain": "tail_state_dx_end",
                "dseed": "d_final_state",
                "dcomposed": "composed_dstate",
                "dstate0": "d_initial_state",
            }
            absent = len(names) + len(self.carve_names)
            indices = [positions.get(aliases.get(name, "tail_" + name), absent) for name in self.tail_buffer_names]
            self.tail_buffers = itemgetter(*indices)
            self.tail_scalars = (
                int(self.tail.pieces),
                int(self.tail.num_seqs),
                int(self.recompute_span_tokens if coarse else 0) // self.b_t,
                int(self.tail.batch_base),
                int(self.tail.first_head_lo),
            )

    def host_config(self):
        """The jit host of the plan's scheme and its constexpr arguments by parameter name, built from the node's tensors for a
        caller that traces the host itself (every data type set); the compiled launch derives them from its buffers."""
        from .common.host import get_dtype
        from .kernel import kda_bprop_f16, kda_chain_backward_f16, kda_uncut_backward_f16

        node = self.node
        dtype_names = {"BFLOAT16": "bfloat16", "HALF": "float16"}
        io_dtype = get_dtype(self.io_name)
        gate_dtype = get_dtype(dtype_names.get(node.inputs["g"].get_data_type().name, "float32"))
        coarse = self.coarse_checkpoints
        flags = dict(
            coarse=coarse,
            use_dstate0=self.index_d_initial_state is not None,
            l2norm=self.use_qk_l2norm,
            safe_gate=self.safe_gate,
            gate_scale_log2=self.gate_lower_bound * kda_bprop_f16.LOG2_E,
            log_gate=self.log_gate,
            beta_sigmoid=self.use_beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            max_active_clusters=self.num_sm,
            d_k=self.dim_k,
            d_v=self.dim_v,
        )
        if self.chain:
            summary_cfg, transition_cfg, series_cfg, bwd_summary_cfg, bprop_cfg = kda_chain_backward_f16.build_configs(
                io_dtype, get_dtype("float32"), get_dtype("float32"), gate_dtype, fused_h_m=self.fused_h_m, series=self.needs_recompute, **flags
            )
            return kda_chain_backward_f16.chain_backward_host, dict(
                unit_chunks=self.unit_chunks,
                b_t=self.b_t,
                length_rule=self.length_rule,
                fused_h_m=self.fused_h_m,
                series=self.needs_recompute,
                has_piece_transitions=self.has_piece_transitions,
                has_piece_state_grads=self.has_piece_tables,
                summary_cfg=summary_cfg,
                transition_cfg=transition_cfg,
                bwd_summary_cfg=bwd_summary_cfg,
                series_cfg=series_cfg,
                bprop_cfg=bprop_cfg,
                dim_v=self.dim_v,
                dim_k=self.dim_k,
                chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                has_seed=self.index_initial_state is not None,
                has_dseed=self.index_d_final_state is not None,
                tail=False,
                gate_lower_bound=self.gate_lower_bound,
            )
        seeded = self.needs_recompute and not coarse
        state = node.inputs["initial_state"] if seeded and self.index_initial_state is not None else None
        recompute_cfg, bprop_cfg = kda_uncut_backward_f16.build_configs(
            io_dtype,
            get_dtype(dtype_names.get(state.get_data_type().name, "float32") if state is not None else "float32"),
            gate_dtype,
            recompute=self.needs_recompute,
            has_state_in=state is not None,
            has_composed=seeded and self.index_composed is not None,
            use_dstate_in=self.index_d_final_state is not None or self.index_following_state_grad is not None,
            dstate_table=self.index_d_final_state is not None,
            has_composed_dstate=self.index_following_state_grad is not None,
            use_initial_state=self.index_initial_state is not None or self.index_composed is not None,
            **flags,
        )
        return kda_uncut_backward_f16.uncut_backward_host, dict(
            b_t=self.b_t,
            io_dtype=io_dtype,
            recompute=self.needs_recompute,
            recompute_orders=not coarse,
            coarse=coarse,
            bwd_orders=self.has_state_checkpoints,
            recompute_cfg=recompute_cfg,
            bprop_cfg=bprop_cfg,
            gate_lower_bound=self.gate_lower_bound,
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
        state_m = context_region["state_m"] if self.has_piece_transitions else region.get("state_m")
        state_g = context_region["state_g"] if self.has_piece_tables else region.get("state_g")
        dcomposed = None
        fold_buffers = (None,) * 4
        if self.index_following_state_grad is not None:
            dcomposed = region["composed_dstate"]
            fold_buffers = (views[self.index_following_state_grad], views[self.index_following_transition], dstate_in, dcomposed)
        work_items = region["work_items"]
        work_count = region["work_count"]
        dq_out = region["dq_ho"] if self.fold_dq else dq
        dk_out = region["dk_ho"] if self.fold_dk else dk
        dv_out = region["dv_ho"] if self.fold_dv else dv
        coarse = self.coarse_checkpoints
        checkpoint_series = state_checkpoints if (self.has_state_checkpoints and not coarse) else region["state_checkpoints"]
        post_buffers = (
            *((dg, g, a_log, dt_bias, region.get("gate_part_a"), region.get("gate_part_dt"), d_a_log, d_dt_bias) if self.safe_gate else (None,) * 8),
            *((dq_out, dq) if self.fold_dq else (None, None)),
            *((dk_out, dk) if self.fold_dk else (None, None)),
            *((dv_out, dv) if self.fold_dv else (None, None)),
        )

        if self.chain:
            self.run_chain(
                q,
                k,
                v,
                g,
                beta,
                do,
                cu,
                state_checkpoints,
                checkpoint_series,
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
                state_m,
                state_g,
                post_buffers,
                fold_buffers,
            )
        else:
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
                checkpoints=checkpoint_series,
                seed_checkpoints=state_checkpoints if coarse else None,
                state_in=state0 if self.needs_recompute and not coarse else None,
                composed=composed if self.needs_recompute and not coarse else None,
                dgate=dg,
                dbeta=dbeta,
                dstate0=dstate0 if self.has_dstate0 else None,
                dstate_in=dstate_in,
                dcomposed=dcomposed,
                work_items=work_items,
                work_count=work_count,
                series_items=(region["work_items_recompute"] if coarse else work_items) if self.needs_recompute else None,
                series_count=(region["work_count_recompute"] if coarse else work_count) if self.needs_recompute else None,
                scheduler_all=region["scheduler_all"],
                scheduler_recompute=region["scheduler_recompute"],
                scheduler_bwd=region["scheduler_bwd"],
                recompute_words=region.get("recompute_tensormaps"),
                bprop_words=region["bwd_tensormaps"],
                dstate_scratch=region["dstate_scratch"],
                dstate_snapshot=region.get("dstate_snapshot"),
                item_limit_word=region.get("gate"),
                tail_scalars=self.tail_scalars if self.tail is not None else (),
                tail_buffers=self.tail_buffers((*views, *region.values(), None)) if self.tail is not None else (),
                post_buffers=post_buffers,
                fold_buffers=fold_buffers,
            )
            schedule = dict(
                b_t=self.b_t,
                recompute=self.needs_recompute,
                recompute_orders=not coarse,
                coarse=coarse,
                bwd_orders=self.has_state_checkpoints,
                seed_span_tokens=self.recompute_span_tokens if coarse else 0,
                seed_every_n_tokens=self.checkpoint_cadence if coarse else 0,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                scale=self.scale,
            )
            tail = self.tail
            if self.uncut_launch is None:
                self.uncut_launch = self.build_uncut_backward(
                    **buffers,
                    **schedule,
                    tail=tail,
                    tail_chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, tail.num_seqs, self.n_heads_out, self.num_sm) if tail is not None else 0,
                    use_initial_state=state0 is not None or composed is not None,
                    num_sm=self.num_sm,
                    gate_lower_bound=self.gate_lower_bound,
                    use_qk_l2norm=self.use_qk_l2norm,
                    use_beta_sigmoid=self.use_beta_sigmoid,
                    allow_neg_eigval=self.allow_neg_eigval,
                    device=self.device,
                    stream=stream,
                )
            self.run_uncut_backward(self.uncut_launch, **buffers, **schedule, stream=stream)

        return None

    def run_chain(
        self,
        q,
        k,
        v,
        g,
        beta,
        do,
        cu,
        state_checkpoints,
        checkpoint_series,
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
        state_m,
        state_g,
        post_buffers,
        fold_buffers,
    ) -> None:
        """Chain prologue, H and M summaries (M alone when the series is passed back), forward chain X, G summary, reverse
        chain seeded with d_final_state, dense series (the forward's or the seeded recompute), bprop over the pieces: one
        compiled launch."""
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
            summary_words=region.get("fused_tensormaps"),
            recompute_m_words=region["recompute_tensormaps_m"],
            series_words=region.get("recompute_tensormaps_series"),
            bprop_summary_words=region["summary_tensormaps"],
            bprop_words=region["bwd_tensormaps"],
            dstate_scratch=region["dstate_scratch"],
            checkpoints=checkpoint_series,
            seed_checkpoints=state_checkpoints if coarse else None,
            dq=dq_out,
            dk=dk_out,
            dv=dv_out,
            dgate=dg,
            dbeta=dbeta,
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
            post_buffers=post_buffers,
            fold_buffers=fold_buffers,
        )
        schedule = dict(
            pieces=self.pieces,
            heads_out=self.n_heads_out,
            num_seqs=self.num_seqs,
            b_t=self.b_t,
            fused_h_m=self.fused_h_m,
            series=series,
            coarse=coarse,
            series_span_tokens=self.recompute_span_tokens if coarse else 0,
            seed_every_n_tokens=self.checkpoint_cadence if coarse else 0,
            log_gate=self.log_gate,
            safe_gate=self.safe_gate,
            scale=self.scale,
            has_piece_transitions=self.has_piece_transitions,
            has_piece_state_grads=self.has_piece_tables,
        )
        if self.chain_launch is None:
            self.chain_launch = self.build_chain_backward(
                **buffers,
                **schedule,
                unit_chunks=self.unit_chunks,
                length_rule=self.length_rule,
                gate_lower_bound=self.gate_lower_bound,
                use_qk_l2norm=self.use_qk_l2norm,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                device=self.device,
                num_sm=self.num_sm,
                stream=stream,
            )
        self.run_chain_backward(self.chain_launch, **buffers, **schedule, stream=stream)


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
    """Compiled summary plan.  With ``transition`` bound the fused H + M summary kernel writes ``final_state`` (initial_state
    honored) and ``M_buf = M^T`` (``X_final = X_init @ M_buf + X_H``), each in its own dtype, in one pass over K: on the uncut
    or ``dv_split`` table, or in ``chain`` plans per filled piece, composed by one fp32 state chain whose tail is
    ``final_state`` and whose running product is ``transition``.  Without ``transition`` the state-only recompute kernel
    writes ``final_state`` alone."""

    def __init__(self, node, recompute_module):
        """Resolve graph attributes and select the forward-summary schedule."""
        from .common.host import tensormap_workspace_bytes
        from .common.piece_chain import (
            DV_SPLIT_TILES,
            chain_rows_per_cta,
            choose_pieces,
            is_dv_split,
            piece_table_layout,
        )
        from .kernel.kda_chain_forward_f16 import build_chain_summary, run_chain_summary
        from .kernel.kda_uncut_forward_f16 import SUMMARY_TAIL_BUFFERS, build_uncut_summary, run_uncut_summary
        from .common.split_k import WORK_ITEM_FIELDS
        from .common.tail import GATE_WORDS, choose_tail

        self.node = node
        self.chain_rows_per_cta = chain_rows_per_cta
        self.build_chain_summary = build_chain_summary
        self.run_chain_summary = run_chain_summary
        self.chain_launch = None
        self.build_uncut_summary = build_uncut_summary
        self.run_uncut_summary = run_uncut_summary
        self.uncut_launch = None
        self.tail_buffer_names = SUMMARY_TAIL_BUFFERS
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
        self.overwrite_initial_state = bool(node.params.get("overwrite_initial_state", False))
        self.num_sm = multiprocessor_count(self.device)
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
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        from .kernel import kda_summary_f16 as summary_module

        self.fused_direct = self.has_transition and not self.chain
        self.dv_split = (
            self.fused_direct
            and not self.batch_invariant
            and is_dv_split(
                num_seqs=B,
                heads_out=HO,
                dim_v=V,
                num_sm=self.num_sm,
                total_tokens=total,
                b_t=self.b_t,
                expand_num=1,
                unit_chunks=self.unit_chunks,
            )
        )
        self.tiles_per_head = DV_SPLIT_TILES if self.dv_split else 1
        self.prep = self.dv_split and B * HO <= int(PREP_TILE_FRACTION * self.num_sm)
        if self.prep:
            from .kernel import kda_prep_summary_f16 as summary_module
        self.n_tiles = B * HO * self.tiles_per_head
        self.tail = None
        if not self.chain and not self.batch_invariant and self.has_transition:
            self.tail = choose_tail(
                num_seqs=B,
                heads_out=HO,
                num_sm=self.num_sm,
                total_tokens=total,
                b_t=self.b_t,
                cadence_tokens=0,
                expand_num=1,
            )

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
            fused_bytes = tensormap_workspace_bytes(summary_module, self.num_pieces)
            regions.append(("fused_tensormaps", layout.add(fused_bytes, align=128), "int64", (fused_bytes // 8,)))
        else:
            off_scheduler = layout.add(8)
            regions += [
                ("scheduler_final", off_scheduler, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (2,)),
            ]
            self.work_item_rows = self.n_tiles
            regions.append(("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
            regions.append(("work_count", layout.add(4), "int32", (1,)))
            if self.tail is not None:
                regions.append(("gate", layout.add(4 * GATE_WORDS, align=16), "int32", (GATE_WORDS,)))
            tensormap_bytes = max(tensormap_workspace_bytes(recompute_module, B), tensormap_workspace_bytes(summary_module, B))
            regions.append(("tensormaps", layout.add(tensormap_bytes, align=128), "int64", (tensormap_bytes // 8,)))
            if self.prep:
                from .kernel import kda_prep_f16 as prep_module
                from .kernel.gdn_tinv_f16 import tinv_rows

                io_name = "float16" if k.get_data_type().name == "HALF" else "bfloat16"
                rows = tinv_rows(total, B, 1, self.b_t)
                for name in ("prep_k_decay", "prep_t"):
                    regions.append((name, layout.add(rows * HO * self.b_t * K * 2, align=128), io_name, (rows, HO, self.b_t, K)))
                regions.append(("prep_diag", layout.add(rows * HO * K * 4, align=128), "float32", (rows, HO, K)))
                prep_words = tensormap_workspace_bytes(prep_module, B) // 8
                regions.append(("prep_tensormaps", layout.add(prep_words * 8, align=128), "int64", (prep_words,)))
                regions.append(("prep_rows", layout.add(rows * 16), "int32", (rows, 4)))
                regions.append(("prep_row_count", layout.add(4), "int32", (1,)))
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
            tail_fused_bytes = tensormap_workspace_bytes(summary_module, self.tail_pieces_total)
            regions.append(("tail_fused_tensormaps", layout.add(tail_fused_bytes, align=128), "int64", (tail_fused_bytes // 8,)))
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
            scheduler=region["scheduler_final"],
            scheduler_all=region["scheduler_all"],
            workspace=region["tensormaps"],
            prep_k_decay=region.get("prep_k_decay"),
            prep_t=region.get("prep_t"),
            prep_diag=region.get("prep_diag"),
            prep_words=region.get("prep_tensormaps"),
            prep_rows=region.get("prep_rows"),
            prep_row_count=region.get("prep_row_count"),
            item_limit_word=region.get("gate"),
            tail_scalars=self.tail_scalars if tail is not None else (),
            tail_buffers=self.tail_buffers((*views, *region.values(), None)) if tail is not None else (),
        )
        if self.uncut_launch is None:
            self.uncut_launch = self.build_uncut_summary(
                **buffers,
                tail=tail,
                tail_chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, tail.num_seqs, self.n_heads_out, self.num_sm) if tail is not None else 0,
                tiles_per_head=self.tiles_per_head,
                num_sm=self.num_sm,
                b_t=self.b_t,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                use_qk_l2norm=self.use_qk_l2norm,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                device=self.device,
                stream=stream,
            )
        self.run_uncut_summary(self.uncut_launch, **buffers, stream=stream)

    def run_chain(self, k, v, g, beta, cu, state0, final_state, transition, a_log, dt_bias, region, stream) -> None:
        """Chain prologue, fused H and M summaries of every filled piece, one fp32 state chain seeded with initial_state whose
        tail is final_state and whose running product is transition."""
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
            summary_words=region["fused_tensormaps"],
            state_h=region["state_h"],
            state_m=region["state_m"],
            state_x=region["state_x"],
            seed=state0,
            final_state=final_state,
            transition=transition,
        )
        if self.chain_launch is None:
            self.chain_launch = self.build_chain_summary(
                **buffers,
                pieces=self.pieces,
                heads_out=self.n_heads_out,
                num_seqs=self.num_seqs,
                unit_chunks=self.unit_chunks,
                b_t=self.b_t,
                length_rule=self.length_rule,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                use_qk_l2norm=self.use_qk_l2norm,
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


class CompiledKdaSummaryBwd:
    """Compiled gradient summary over the bprop summary kernel: the reverse state-gradient recurrence seeded by
    ``d_final_state`` (zero when unbound) writes ``d_initial_state``; a bound ``transition`` receives ``M^T`` from the
    recompute's transition arm.  ``chain`` summarizes every filled piece (G, M) and composes them with one reverse fp32
    state chain whose tail is ``d_initial_state`` and whose running product is ``transition``."""

    def __init__(self, node, summary_module):
        """Resolve graph attributes and select the backward-summary schedule."""
        from .common.host import tensormap_workspace_bytes
        from .common.piece_chain import chain_rows_per_cta, choose_pieces, piece_table_layout
        from .kernel.kda_chain_backward_f16 import build_chain_summary_backward, run_chain_summary_backward
        from .kernel.kda_uncut_backward_f16 import SUMMARY_TAIL_BUFFERS, build_uncut_summary_backward, run_uncut_summary_backward
        from .common.split_k import WORK_ITEM_FIELDS
        from .common.tail import GATE_WORDS, choose_tail

        self.node = node
        self.chain_rows_per_cta = chain_rows_per_cta
        self.build_chain_summary_backward = build_chain_summary_backward
        self.run_chain_summary_backward = run_chain_summary_backward
        self.chain_launch = None
        self.build_uncut_summary_backward = build_uncut_summary_backward
        self.run_uncut_summary_backward = run_uncut_summary_backward
        self.uncut_launch = None
        self.tail_buffer_names = SUMMARY_TAIL_BUFFERS
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
        self.overwrite_initial_state = bool(node.params.get("overwrite_initial_state", False))
        self.checkpoint_cadence = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.pieces, self.unit_chunks = choose_pieces(
            num_seqs=B,
            heads_out=HO,
            num_sm=self.num_sm,
            total_tokens=total,
            b_t=self.b_t,
            cadence_tokens=self.checkpoint_cadence,
            batch_invariant=self.batch_invariant,
            compose_tail=True,
            reverse=True,
            expand_num=1,
        )
        self.has_context = bool(node.params.get("output_context", False))
        self.context = None
        if self.has_context:
            self.context = backward_context(
                self.plan_name,
                full=False,
                total=total,
                num_householder=1,
                h_q=q.dim[1],
                h_k=node.inputs["k"].dim[1],
                h_o=HO,
                d_k=K,
                d_v=V,
                b_t=self.b_t,
                io_name="float16" if q.get_data_type().name == "HALF" else "bfloat16",
                num_sm=self.num_sm,
                cadence_tokens=self.checkpoint_cadence,
                batch_invariant=self.batch_invariant,
                use_qk_l2norm=self.use_qk_l2norm,
            )
        self.context_size = self.context.size if self.has_context else 0
        self.chain = self.pieces > 0
        self.length_rule = self.chain and self.batch_invariant
        self.num_pieces = B * self.pieces if self.chain else B
        self.has_piece_tables = self.has_context and self.context.has_tables
        self.tail = None
        if not self.chain and not self.batch_invariant:
            self.tail = choose_tail(
                num_seqs=B,
                heads_out=HO,
                num_sm=self.num_sm,
                total_tokens=total,
                b_t=self.b_t,
                cadence_tokens=0,
                expand_num=1,
            )
        layout = WorkspaceLayout()
        regions = []
        # One FP32 state and bounded-key ring per persistent CTA; sequential backward kernels share the arena.
        from .common.host import get_dtype
        from .common.steep_decay import STEEP_OFF, steep_mode_for

        io_name = "float16" if node.inputs["q"].get_data_type().name == "HALF" else "bfloat16"
        self.steep_mode = steep_mode_for(get_dtype(io_name), self.use_qk_l2norm, self.safe_gate, self.gate_lower_bound * summary_module.LOG2_E, self.b_t)
        dstate_scratch_elements = (
            max(1, min(self.num_sm, self.num_pieces * HO)) * (V * K + summary_module.CFG.SMEM_DECAY_STAGES * self.b_t * K)
            if self.steep_mode != STEEP_OFF
            else 1
        )
        regions.append(("dstate_scratch", layout.add(dstate_scratch_elements * 4), "float32", (dstate_scratch_elements,)))
        if self.chain:
            from .kernel import kda_recompute_f16 as recompute_module

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
            summary_bytes = tensormap_workspace_bytes(summary_module, self.num_pieces)
            regions.append(("summary_tensormaps", layout.add(summary_bytes, align=128), "int64", (summary_bytes // 8,)))
            recompute_bytes = tensormap_workspace_bytes(recompute_module, self.num_pieces)
            regions.append(("recompute_tensormaps_m", layout.add(recompute_bytes, align=128), "int64", (recompute_bytes // 8,)))
        else:
            off_scheduler = layout.add(16)
            self.n_tiles = B * HO
            self.work_item_rows = self.n_tiles
            regions += [
                ("scheduler_main", off_scheduler, "int32", (2,)),
                ("scheduler_transition", off_scheduler + 8, "int32", (2,)),
                ("scheduler_all", off_scheduler, "int32", (4,)),
                ("work_items", layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4), "int32", (self.work_item_rows, WORK_ITEM_FIELDS)),
                ("work_count", layout.add(4), "int32", (1,)),
            ]
            if self.tail is not None:
                regions.append(("gate", layout.add(4 * GATE_WORDS, align=16), "int32", (GATE_WORDS,)))
            tensormap_bytes = tensormap_workspace_bytes(summary_module, B)
            regions.append(("tensormaps", layout.add(tensormap_bytes, align=128), "int64", (tensormap_bytes // 8,)))
            if self.has_transition:
                from .kernel import kda_recompute_f16 as recompute_module

                recompute_bytes = tensormap_workspace_bytes(recompute_module, B)
                regions.append(("recompute_tensormaps", layout.add(recompute_bytes, align=128), "int64", (recompute_bytes // 8,)))
                regions.append(("transition_m", layout.add(B * HO * K * K * 4), "float32", (B, HO, K, K)))
        if self.tail is not None:
            from .kernel import kda_recompute_f16 as tail_recompute_module

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
            tail_summary_bytes = tensormap_workspace_bytes(summary_module, self.tail_pieces_total)
            regions.append(("tail_summary_tensormaps", layout.add(tail_summary_bytes, align=128), "int64", (tail_summary_bytes // 8,)))
            tail_recompute_bytes = tensormap_workspace_bytes(tail_recompute_module, self.tail_pieces_total)
            regions.append(("tail_recompute_tensormaps_m", layout.add(tail_recompute_bytes, align=128), "int64", (tail_recompute_bytes // 8,)))
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
        state_m = context_region["state_m"] if self.has_piece_tables else region.get("state_m")
        state_g = context_region["state_g"] if self.has_piece_tables else region.get("state_g")
        if self.chain:
            self.run_chain(q, k, g, beta, do, cu, dstate_in, dstate0, transition, a_log, dt_bias, region, stream, state_m, state_g)
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
            scheduler=region["scheduler_main"],
            scheduler_all=region["scheduler_all"],
            workspace=region["tensormaps"],
            dstate_scratch=region["dstate_scratch"],
            transition_m=region.get("transition_m"),
            transition=transition,
            scheduler_transition=region["scheduler_transition"] if self.has_transition else None,
            transition_words=region.get("recompute_tensormaps"),
            item_limit_word=region.get("gate"),
            tail_scalars=self.tail_scalars if tail is not None else (),
            tail_buffers=self.tail_buffers((*views, *region.values(), None)) if tail is not None else (),
        )
        if self.uncut_launch is None:
            self.uncut_launch = self.build_uncut_summary_backward(
                **buffers,
                tail=tail,
                tail_chain_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, tail.num_seqs, self.n_heads_out, self.num_sm) if tail is not None else 0,
                transition_rows=self.chain_rows_per_cta(self.dim_v, self.dim_k, self.num_seqs, self.n_heads_out, self.num_sm),
                num_sm=self.num_sm,
                b_t=self.b_t,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                use_qk_l2norm=self.use_qk_l2norm,
                use_beta_sigmoid=self.use_beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                scale=self.scale,
                device=self.device,
                stream=stream,
            )
        self.run_uncut_summary_backward(self.uncut_launch, **buffers, scale=self.scale, stream=stream)

    def run_chain(self, q, k, g, beta, do, cu, dstate_in, dstate0, transition, a_log, dt_bias, region, stream, state_m, state_g) -> None:
        """Chain prologue, M (recompute transition arm) and G (zero-seeded bprop summary) of every filled piece, one reverse
        fp32 state chain seeded with d_final_state whose tail is d_initial_state and whose running product is transition."""
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
            recompute_m_words=region["recompute_tensormaps_m"],
            bprop_summary_words=region["summary_tensormaps"],
            dstate_scratch=region["dstate_scratch"],
            state_m=state_m,
            state_g=state_g,
            state_x=region["state_x"],
            dstate_in=dstate_in,
            dstate0=dstate0,
            transition=transition,
        )
        if self.chain_launch is None:
            self.chain_launch = self.build_chain_summary_backward(
                **buffers,
                pieces=self.pieces,
                heads_out=self.n_heads_out,
                num_seqs=self.num_seqs,
                unit_chunks=self.unit_chunks,
                b_t=self.b_t,
                length_rule=self.length_rule,
                log_gate=self.log_gate,
                safe_gate=self.safe_gate,
                gate_lower_bound=self.gate_lower_bound,
                use_qk_l2norm=self.use_qk_l2norm,
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
