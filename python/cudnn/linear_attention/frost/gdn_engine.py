# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDN engine: GDN nodes on the chunked prefill kernel
(``kernel/gdn_prefill_f16.py``) and GDN_BWD nodes on the chunked backward
kernel (``kernel/gdn_bprop_f16.py``), SM100/SM103/SM107, bf16/fp16.
The backward regenerates the per-chunk state checkpoints with the recompute kernel
(``kernel/gdn_recompute_f16.py``) when the graph does not provide one.
GDP/GDP_BWD nodes run the same compiled plans on the ``num_householder``-expanded
timeline: q/g/dO scatter into expanded workspace copies, O/dQ/dG gather back
(``common/expand.py``)."""

from __future__ import annotations

import math

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost.device import build_device, current_device, multiprocessor_count
from cudnn.frost.workspace import WorkspaceLayout, carve_plan
from ..graph_analyzer import analyze
from .engine import FrostLaPlan, frost_la_gate


def build_gdn(graph):
    """The expensive step: import the kernel module (pulls in the Cutlass
    primitives; the cute.compile itself is cached inside the kernel per static
    config and runs on first execute, when the real buffers are known)."""
    nodes = list(graph.nodes)
    if len(nodes) != 1 or getattr(nodes[0].node_type, "name", None) not in ("GDN", "GDN_BWD", "GDP", "GDP_BWD"):
        raise ValueError("build_gdn: graph does not contain exactly one GDN/GDN_BWD/GDP/GDP_BWD node")
    node = nodes[0]
    if node.node_type.name in ("GDN_BWD", "GDP_BWD"):
        from .kernel import gdn_bprop_f16 as bwd_module
        from .kernel import gdn_recompute_f16 as recompute_module

        return CompiledGdnBwd(node, bwd_module, recompute_module)
    from .kernel import gdn_prefill_f16 as kernel_module

    return CompiledGdn(node, kernel_module)


def gdn_support_gates(engine: str, facts) -> None:
    """The GDN kernel family's dtype/attribute gates, shared with the GDP engine."""
    import cudnn

    checkpoint = facts.checkpoint_every_n_tokens
    if checkpoint and (facts.is_bwd or checkpoint % 64 != 0):
        raise NotImplementedError(f"{engine}: checkpoint_every_n_tokens must be a positive multiple of 64 on the forward node (got {checkpoint})")
    if not facts.gates_at_ho:
        raise NotImplementedError(f"{engine}: g/beta must carry HO = max(q, v) heads ({facts.h_o})")
    fp32 = cudnn.data_type.FLOAT
    io = (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
    state_dtypes = (fp32, cudnn.data_type.BFLOAT16)
    beta_wants = (facts.io_dtype,) if facts.use_beta_sigmoid else (fp32, facts.io_dtype)
    if facts.beta_dtype not in beta_wants + (None,):
        raise NotImplementedError(f"{engine}: 'beta' must be {' or '.join(str(w) for w in beta_wants)}, got {facts.beta_dtype}")
    for port, got in (("a_log", facts.a_log_dtype), ("dt_bias", facts.dt_bias_dtype)):
        if got not in (fp32, None):
            raise NotImplementedError(f"{engine}: '{port}' must be fp32, got {got}")
    for port, got in (("initial_state", facts.state_dtype), ("final_state", facts.final_state_dtype)):
        if got not in state_dtypes + (None,):
            raise NotImplementedError(f"{engine}: '{port}' must be fp32/bf16, got {got}")
    if not facts.state_pair_match:
        raise NotImplementedError(f"{engine}: initial_state and final_state dtypes must match")
    if facts.is_bwd:
        for port, got in (("d_a_log", facts.d_a_log_dtype), ("d_dt_bias", facts.d_dt_bias_dtype)):
            if got not in (fp32, None):
                raise NotImplementedError(f"{engine}: '{port}' must be fp32, got {got}")
        state_grad_want = facts.state_dtype if facts.state_dtype is not None else fp32
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
    """FROST chunked-kernel backend for single-node GDN graphs (THD layout).

    Default GDN engine on SM100/SM103/SM107 (lowest GDN engine_id); declines
    elsewhere so ranking falls back to ``GdnCuTileEngine``."""

    name = "gdn_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        facts = graph._facts_for(analyze)
        frost_la_gate("GdnFrostEngine", facts, "GDN", d_v_allowed=(64, 128))
        gdn_support_gates("GdnFrostEngine", facts)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_gdn(graph))


class CompiledGdn:
    """Compiled FROST GDN plan: a callable over the resolved node buffers."""

    def __init__(self, node, kernel_module):
        from .common.split_k import WORK_ITEM_FIELDS, build_split_table, chunk_scratch_rows, compute_ideal_chunks, max_work_items, run_table

        self.node = node
        self.kernel = kernel_module
        self.build_split_table = build_split_table
        self.run_table = run_table
        self.table = None
        self.kcache = None
        self.plan_name = "GdpFrostEngine (GDP)" if node.node_type.name == "GDP" else "GdnFrostEngine (GDN)"
        from .common.l2norm import build_l2norm_qk, run_l2norm_qk

        self.build_l2norm_qk = build_l2norm_qk
        self.run_l2norm_qk = run_l2norm_qk
        self.l2norm = None
        self.num_householder = int(node.params.get("num_householder", 1) or 1)
        if self.num_householder > 1:
            from .common.expand import build_gather_o, build_pack_fwd, run_gather_o, run_pack_fwd

            self.build_pack_fwd = build_pack_fwd
            self.run_pack_fwd = run_pack_fwd
            self.build_gather_o = build_gather_o
            self.run_gather_o = run_gather_o
            self.pack_fwd = None
            self.gather_o = None
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))

        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self.b_t = kernel_module.CFG.B_T
        total = node.inputs["k"].dim[0]
        HO = g.dim[1]
        HQ, HK = q.dim[1], node.inputs["k"].dim[1]
        V = v.dim[2]
        self.io_name = "float16" if q.get_data_type().name == "HALF" else "bfloat16"
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.has_final_state = "final_state" in node.outputs
        self.checkpoint = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.has_state_checkpoints = "state_checkpoints" in node.outputs
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.split = self.checkpoint % self.b_t == 0 and not self.batch_invariant

        layout = WorkspaceLayout()
        from .common.host import tensormap_workspace_bytes

        self.tensormap_words = tensormap_workspace_bytes(kernel_module, B) // 8
        self.off_tensormaps = layout.add(self.tensormap_words * 8)
        self.off_scheduler = layout.add(8)
        self.num_sm = multiprocessor_count(current_device())
        self.n_tiles = B * HO
        self.n_heads_out = HO
        if self.split:
            self.ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
            self.work_item_rows = max_work_items(total, B, HO, self.ideal, self.b_t, self.num_sm)
        else:
            self.ideal = None
            self.work_item_rows = self.n_tiles
        self.off_work_items = layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4)
        self.off_work_count = layout.add(4)
        if self.split:
            self.off_item_scratch = layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4)
            self.chunk_scratch_rows = chunk_scratch_rows(total, B, self.b_t)
            self.off_chunk_scratch = layout.add(self.chunk_scratch_rows * HO * 4)
        if self.use_qk_l2norm:
            self.off_q_n = layout.add(total * HQ * 128 * 2)
            self.off_k_n = layout.add(total * HK * 128 * 2)
            self.off_inv_q = layout.add(total * HQ * 4)
            self.off_inv_k = layout.add(total * HK * 4)
        if self.num_householder > 1:
            self.off_q_x = layout.add(total * HQ * 128 * 2)
            self.off_g_x = layout.add(total * HO * 4)
            self.off_o_x = layout.add(total * HO * V * 2)
        self.needs_table = self.split
        self.workspace_size = layout.size
        regions = [
            (self.off_tensormaps, "int64", (self.tensormap_words,)),
            (self.off_scheduler, "int32", (2,)),
            (self.off_work_items, "int32", (self.work_item_rows, WORK_ITEM_FIELDS)),
            (self.off_work_count, "int32", (1,)),
        ]
        if self.split:
            regions += [
                (self.off_item_scratch, "int32", (self.work_item_rows, WORK_ITEM_FIELDS)),
                (self.off_chunk_scratch, "float32", (self.chunk_scratch_rows, self.n_heads_out)),
            ]
        if self.use_qk_l2norm:
            regions += [
                (self.off_q_n, self.io_name, (total, HQ, 128)),
                (self.off_k_n, self.io_name, (total, HK, 128)),
                (self.off_inv_q, "float32", (total, HQ)),
                (self.off_inv_k, "float32", (total, HK)),
            ]
        if self.num_householder > 1:
            regions += [
                (self.off_q_x, self.io_name, (total, HQ, 128)),
                (self.off_g_x, "float32", (total, HO)),
                (self.off_o_x, self.io_name, (total, HO, V)),
            ]
        self.carve = carve_plan(self.plan_name, regions)

    def workspace_bytes(self) -> int:
        return self.workspace_size

    def bind(self, names) -> None:
        pos = {name: i for i, name in enumerate(names)}
        self.iq = pos["q"]
        self.ik = pos["k"]
        self.iv = pos["v"]
        self.ig = pos["g"]
        self.ibeta = pos["beta"]
        self.icu = pos["cu_seqlens"]
        self.is0 = pos.get("initial_state")
        self.io_ = pos["O"]
        self.ifs = pos.get("final_state")
        self.ick = pos.get("state_checkpoints")
        self.ia_log = pos.get("a_log")
        self.idt_bias = pos.get("dt_bias")

    def run(self, views, workspace, stream) -> None:
        q = views[self.iq]
        k = views[self.ik]
        v = views[self.iv]
        g = views[self.ig]
        beta = views[self.ibeta]
        cu = views[self.icu]
        state0 = views[self.is0] if self.is0 is not None else None
        o = views[self.io_]
        final_state = views[self.ifs] if self.ifs is not None else None
        state_checkpoints = views[self.ick] if self.ick is not None else None
        a_log = views[self.ia_log] if self.ia_log is not None else None
        dt_bias = views[self.idt_bias] if self.idt_bias is not None else None
        stream = stream if stream is not None else 0
        carved = workspace.carve(self.carve)
        if self.split:
            tensormaps, scheduler_counter, work_items, work_count, item_scratch, chunk_scratch, *rest = carved
        else:
            tensormaps, scheduler_counter, work_items, work_count, *rest = carved
            item_scratch = chunk_scratch = None
        o_x = None
        if self.num_householder > 1:
            n = self.num_householder
            *rest, q_x, g_x, o_x = rest
            if self.use_qk_l2norm:
                q_n, k_n, inv_q, inv_k = rest
                if self.pack_fwd is None:
                    self.pack_fwd = self.build_pack_fwd(g, g_x, None, None, self.num_householder, stream)
                else:
                    self.run_pack_fwd(self.pack_fwd, g, g_x, None, None, stream)
                if self.l2norm is None:
                    self.l2norm = self.build_l2norm_qk(q, k, q_n, k_n, inv_q, inv_k, expand_num=n, expand_phase=n - 1, stream=stream)
                else:
                    self.run_l2norm_qk(self.l2norm, q, k, q_n, k_n, inv_q, inv_k, stream)
                q, k = q_n, k_n
            else:
                if self.pack_fwd is None:
                    self.pack_fwd = self.build_pack_fwd(g, g_x, q, q_x, self.num_householder, stream)
                else:
                    self.run_pack_fwd(self.pack_fwd, g, g_x, q, q_x, stream)
                q = q_x
            g = g_x
        elif self.use_qk_l2norm:
            q_n, k_n, inv_q, inv_k = rest
            if self.l2norm is None:
                self.l2norm = self.build_l2norm_qk(q, k, q_n, k_n, inv_q, inv_k, stream=stream)
            else:
                self.run_l2norm_qk(self.l2norm, q, k, q_n, k_n, inv_q, inv_k, stream)
            q, k = q_n, k_n

        if self.kcache is not None and (self.table is not None or not self.needs_table):
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
                    expand_num=self.num_householder,
                )
            self.kernel.run_prefill(
                self.kcache,
                q,
                k,
                v,
                g,
                beta,
                o if o_x is None else o_x,
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
            )
            if o_x is not None:
                if self.gather_o is None:
                    self.gather_o = self.build_gather_o(o_x, o, self.num_householder, stream)
                else:
                    self.run_gather_o(self.gather_o, o_x, o, stream)
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
                log_gate=True,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                scheduler_counter=scheduler_counter,
                split=self.split,
                expand_num=self.num_householder,
                stream=stream,
            )

        self.kcache = self.kernel.chunk_gdn_sm100(
            q,
            k,
            v,
            g,
            beta,
            o if o_x is None else o_x,
            cu,
            state0,
            final_state,
            self.scale,
            work_items=work_items,
            work_count=work_count,
            scheduler_counter=scheduler_counter,
            checkpoint_every_n_tokens=self.checkpoint,
            output_state_checkpoints=state_checkpoints,
            log_gate=True,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            work_item_scratch=item_scratch,
            expand_num=self.num_householder,
            workspace=tensormaps,
            stream=stream,
        )
        if o_x is not None:
            if self.gather_o is None:
                self.gather_o = self.build_gather_o(o_x, o, self.num_householder, stream)
            else:
                self.run_gather_o(self.gather_o, o_x, o, stream)
        return None


class CompiledGdnBwd:
    """Compiled FROST GDN bprop plan: a callable over the resolved node buffers.

    Produces dQ/dK/dV/dG/dBeta; consumes the forward per-chunk states through
    the node's ``state_checkpoints`` input, or regenerates them with the recompute
    (checkpoint-only) kernel when the port is absent."""

    def __init__(self, node, bwd_module, recompute_module):
        from .common.split_k import WORK_ITEM_FIELDS, build_split_table, chunk_scratch_rows, compute_ideal_chunks, max_work_items, run_table

        self.node = node
        self.bwd = bwd_module
        self.recompute = recompute_module
        self.build_split_table = build_split_table
        self.run_table = run_table
        self.table = None
        self.kcache = None
        self.recompute_cache = None
        self.plan_name = "GdpFrostEngine (GDP_BWD)" if node.node_type.name == "GDP_BWD" else "GdnFrostEngine (GDN_BWD)"
        from .common.gate_bwd import scalar_gate_bwd, scalar_gate_blocks
        from .common.head_reduce import head_group_reduce
        from .common.host import tensormap_workspace_bytes
        from .common.l2norm import build_l2norm_qk, build_l2norm_qk_bwd, run_l2norm_qk, run_l2norm_qk_bwd

        self.head_group_reduce = head_group_reduce
        self.scalar_gate_bwd = scalar_gate_bwd
        self.build_l2norm_qk = build_l2norm_qk
        self.run_l2norm_qk = run_l2norm_qk
        self.build_l2norm_qk_bwd = build_l2norm_qk_bwd
        self.run_l2norm_qk_bwd = run_l2norm_qk_bwd
        self.l2norm = None
        self.l2norm_bwd = None
        self.num_householder = int(node.params.get("num_householder", 1) or 1)
        if self.num_householder > 1:
            from .common.expand import build_gather_bwd, build_pack_bwd, run_gather_bwd, run_pack_bwd

            self.build_pack_bwd = build_pack_bwd
            self.run_pack_bwd = run_pack_bwd
            self.build_gather_bwd = build_gather_bwd
            self.run_gather_bwd = run_gather_bwd
            self.pack_bwd = None
            self.gather_bwd = None
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))

        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self.b_t = bwd_module.CFG.B_T
        total = node.inputs["k"].dim[0]
        self.gate_bwd_blocks = scalar_gate_blocks(total)
        K, V = q.dim[-1], v.dim[-1]
        HQ, HV = q.dim[1], v.dim[1]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.has_state_checkpoints = "state_checkpoints" in node.inputs
        self.io_name = "float16" if node.inputs["q"].get_data_type().name == "HALF" else "bfloat16"
        self.cu_name = "int32" if node.inputs["cu_seqlens"].get_data_type().name == "INT32" else "int64"

        self.num_sm = multiprocessor_count(current_device())
        self.bwd_dynamic_scheduling = True
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        self.split = not self.batch_invariant
        layout = WorkspaceLayout()
        self.off_scheduler = layout.add(16)
        self.tensormap_words = tensormap_workspace_bytes(bwd_module, B) // 8
        self.off_tensormaps = layout.add(self.tensormap_words * 8)
        if not self.has_state_checkpoints:
            self.state_checkpoints_rows = max(total // self.b_t + B, 1)
            self.off_state_checkpoints = layout.add(self.state_checkpoints_rows * HO * K * V * 2)
            self.recompute_tensormap_words = tensormap_workspace_bytes(recompute_module, B) // 8
            self.off_recompute_tensormaps = layout.add(self.recompute_tensormap_words * 8)
        HK = node.inputs["k"].dim[1]
        self.fold_dq = HQ < HO
        self.fold_dk = HK < HO
        self.fold_dv = HV < HO
        if self.fold_dq:
            self.off_dq_ho = layout.add(total * HO * K * 2)
        if self.fold_dk:
            self.off_dk_ho = layout.add(total * HO * K * 2)
        if self.fold_dv:
            self.off_dv_ho = layout.add(total * HO * V * 2)
        if self.use_qk_l2norm:
            self.off_q_n = layout.add(total * HQ * 128 * 2)
            self.off_k_n = layout.add(total * HK * 128 * 2)
            self.off_inv_q = layout.add(total * HQ * 4)
            self.off_inv_k = layout.add(total * HK * 4)
        if self.safe_gate:
            self.off_gate_part_a = layout.add(self.gate_bwd_blocks * HO * 4)
            self.off_gate_part_dt = layout.add(self.gate_bwd_blocks * HO * 4)
        if self.num_householder > 1:
            self.off_q_x = layout.add(total * HQ * 128 * 2)
            self.off_g_x = layout.add(total * HO * 4)
            self.off_do_x = layout.add(total * HO * V * 2)
            self.off_dq_x = layout.add(total * HQ * K * 2)
            self.off_dg_x = layout.add(total * HO * 4)
        self.n_tiles = B * HO
        if self.split:
            self.ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
            self.work_item_rows = max_work_items(total, B, HO, self.ideal, self.b_t, self.num_sm)
        else:
            self.ideal = None
            self.work_item_rows = self.n_tiles
        self.off_work_items = layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4)
        self.off_work_count = layout.add(4)
        if self.split:
            self.off_item_scratch = layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4)
            self.chunk_scratch_rows = chunk_scratch_rows(total, B, self.b_t)
            self.off_chunk_scratch = layout.add(self.chunk_scratch_rows * HO * 4)
        self.needs_table = self.split
        self.recompute_orders = not self.has_state_checkpoints
        self.bwd_orders = self.has_state_checkpoints
        self.n_heads_out, self.total = HO, total
        self.workspace_size = layout.size
        regions = [
            ("scheduler_recompute", self.off_scheduler, "int32", (2,)),
            ("scheduler_bwd", self.off_scheduler + 8, "int32", (2,)),
            ("scheduler_all", self.off_scheduler, "int32", (4,)),
            ("tensormaps", self.off_tensormaps, "int64", (self.tensormap_words,)),
            ("work_items", self.off_work_items, "int32", (self.work_item_rows, WORK_ITEM_FIELDS)),
            ("work_count", self.off_work_count, "int32", (1,)),
        ]
        if self.split:
            regions.append(("item_scratch", self.off_item_scratch, "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
            regions.append(("chunk_scratch", self.off_chunk_scratch, "float32", (self.chunk_scratch_rows, HO)))
        if not self.has_state_checkpoints:
            regions.append(("state_checkpoints", self.off_state_checkpoints, self.io_name, (self.state_checkpoints_rows, HO, V, K)))
            regions.append(("recompute_tensormaps", self.off_recompute_tensormaps, "int64", (self.recompute_tensormap_words,)))
        if self.fold_dq:
            regions.append(("dq_ho", self.off_dq_ho, self.io_name, (total, HO, K)))
        if self.fold_dk:
            regions.append(("dk_ho", self.off_dk_ho, self.io_name, (total, HO, K)))
        if self.fold_dv:
            regions.append(("dv_ho", self.off_dv_ho, self.io_name, (total, HO, V)))
        if self.use_qk_l2norm:
            regions.append(("q_n", self.off_q_n, self.io_name, (total, HQ, 128)))
            regions.append(("k_n", self.off_k_n, self.io_name, (total, HK, 128)))
            regions.append(("inv_q", self.off_inv_q, "float32", (total, HQ)))
            regions.append(("inv_k", self.off_inv_k, "float32", (total, HK)))
        if self.safe_gate:
            regions.append(("gate_part_a", self.off_gate_part_a, "float32", (self.gate_bwd_blocks * HO,)))
            regions.append(("gate_part_dt", self.off_gate_part_dt, "float32", (self.gate_bwd_blocks * HO,)))
        if self.num_householder > 1:
            regions.append(("q_x", self.off_q_x, self.io_name, (total, HQ, 128)))
            regions.append(("g_x", self.off_g_x, "float32", (total, HO)))
            regions.append(("do_x", self.off_do_x, self.io_name, (total, HO, V)))
            regions.append(("dq_x", self.off_dq_x, self.io_name, (total, HQ, K)))
            regions.append(("dg_x", self.off_dg_x, "float32", (total, HO)))
        self.carve_names = [name for name, _off, _dt, _shape in regions]
        self.carve = carve_plan(self.plan_name, [(off, dt, shape) for _name, off, dt, shape in regions])

    def workspace_bytes(self) -> int:
        return self.workspace_size

    def bind(self, names) -> None:
        pos = {name: i for i, name in enumerate(names)}
        self.iq = pos["q"]
        self.ik = pos["k"]
        self.iv = pos["v"]
        self.ig = pos["g"]
        self.ibeta = pos["beta"]
        self.icu = pos["cu_seqlens"]
        self.ido = pos["dO"]
        self.ick = pos.get("state_checkpoints")
        self.is0 = pos.get("initial_state")
        self.idfs = pos.get("d_final_state")
        self.idq = pos["dQ"]
        self.idk = pos["dK"]
        self.idv = pos["dV"]
        self.idg = pos["dG"]
        self.idb = pos["dBeta"]
        self.ids0 = pos.get("d_initial_state")
        self.ia_log = pos.get("a_log")
        self.idt_bias = pos.get("dt_bias")
        self.ida_log = pos.get("d_a_log")
        self.iddt_bias = pos.get("d_dt_bias")

    def run(self, views, workspace, stream) -> None:
        q = views[self.iq]
        k = views[self.ik]
        v = views[self.iv]
        g = views[self.ig]
        beta = views[self.ibeta]
        cu = views[self.icu]
        do = views[self.ido]
        state_checkpoints = views[self.ick] if self.ick is not None else None
        state0 = views[self.is0] if self.is0 is not None else None
        dstate_in = views[self.idfs] if self.idfs is not None else None
        dq = views[self.idq]
        dk = views[self.idk]
        dv = views[self.idv]
        dg = views[self.idg]
        db = views[self.idb]
        dstate0 = views[self.ids0] if self.ids0 is not None else None
        a_log = views[self.ia_log] if self.ia_log is not None else None
        dt_bias = views[self.idt_bias] if self.idt_bias is not None else None
        d_a_log = views[self.ida_log] if self.ida_log is not None else None
        d_dt_bias = views[self.iddt_bias] if self.iddt_bias is not None else None
        stream = stream if stream is not None else 0

        region = dict(zip(self.carve_names, workspace.carve(self.carve)))
        scheduler_recompute = region["scheduler_recompute"]
        scheduler_bwd = region["scheduler_bwd"]
        work_items = region["work_items"]
        work_count = region["work_count"]
        dq_node = dg_node = None
        if self.num_householder > 1:
            if self.pack_bwd is None:
                self.pack_bwd = self.build_pack_bwd(q, region["q_x"], g, region["g_x"], do, region["do_x"], self.num_householder, stream)
            else:
                self.run_pack_bwd(self.pack_bwd, q, region["q_x"], g, region["g_x"], do, region["do_x"], stream)
            q, g, do = region["q_x"], region["g_x"], region["do_x"]
            dq_node, dg_node = dq, dg
            dq, dg = region["dq_x"], region["dg_x"]
        if self.use_qk_l2norm:
            if self.l2norm is None:
                self.l2norm = self.build_l2norm_qk(q, k, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream=stream)
            else:
                self.run_l2norm_qk(self.l2norm, q, k, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream)
            q, k = region["q_n"], region["k_n"]

        if self.kcache is not None and (self.table is not None or not self.needs_table):
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
            if self.has_state_checkpoints:
                checkpoint_series = state_checkpoints
            else:
                checkpoint_series = region["state_checkpoints"]
                self.recompute.run_recompute(
                    self.recompute_cache,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    state0,
                    None,
                    checkpoint_series,
                    work_items,
                    work_count,
                    scheduler_recompute,
                    region["scheduler_all"] if self.recompute_orders else None,
                    region.get("item_scratch") if self.recompute_orders else None,
                    region["recompute_tensormaps"],
                    self.b_t,
                    stream,
                    a_log=a_log if self.safe_gate else None,
                    dt_bias=dt_bias if self.safe_gate else None,
                )
            dq_out = region["dq_ho"] if self.fold_dq else dq
            dk_out = region["dk_ho"] if self.fold_dk else dk
            dv_out = region["dv_ho"] if self.fold_dv else dv
            self.bwd.run_bwd(
                self.kcache,
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
                db,
                cu,
                dstate0,
                dstate_in,
                work_items,
                work_count,
                scheduler_bwd if self.bwd_dynamic_scheduling else None,
                region["scheduler_all"] if self.bwd_orders else None,
                region.get("item_scratch") if self.bwd_orders else None,
                region["tensormaps"],
                self.scale,
                stream,
                a_log=a_log if self.safe_gate else None,
                dt_bias=dt_bias if self.safe_gate else None,
            )
            if self.safe_gate:
                self.scalar_gate_bwd(dg, g, a_log, dt_bias, d_a_log, d_dt_bias, region["gate_part_a"], region["gate_part_dt"], stream=stream)
            if self.fold_dq or self.fold_dk or self.fold_dv:
                for src_ho, dst in ((dq_out, dq), (dk_out, dk), (dv_out, dv)):
                    if src_ho is not dst:
                        self.head_group_reduce(src_ho, dst, stream=stream)
            if self.use_qk_l2norm:
                if self.l2norm_bwd is None:
                    self.l2norm_bwd = self.build_l2norm_qk_bwd(dq, dk, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream=stream)
                else:
                    self.run_l2norm_qk_bwd(self.l2norm_bwd, dq, dk, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream)
            if dq_node is not None:
                if self.gather_bwd is None:
                    self.gather_bwd = self.build_gather_bwd(dq, dq_node, dg, dg_node, self.num_householder, stream)
                else:
                    self.run_gather_bwd(self.gather_bwd, dq, dq_node, dg, dg_node, stream)
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
                log_gate=True,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                scheduler_counter=region["scheduler_all"],
                split=self.split,
                expand_num=self.num_householder,
                stream=stream,
            )

        if self.has_state_checkpoints:
            checkpoint_series = state_checkpoints
        else:
            checkpoint_series = region["state_checkpoints"]
            self.recompute_cache = self.recompute.chunk_gdn_recompute_sm100(
                k,
                v,
                g,
                beta,
                cu,
                state0,
                None,
                checkpoint_every_n_tokens=self.b_t,
                output_state_checkpoints=checkpoint_series,
                safe_gate=self.safe_gate,
                a_log=a_log,
                dt_bias=dt_bias,
                use_beta_sigmoid=self.use_beta_sigmoid,
                work_items=work_items,
                work_count=work_count,
                scheduler_counter=scheduler_recompute,
                scheduler_all=region["scheduler_all"] if self.recompute_orders else None,
                work_item_scratch=region.get("item_scratch") if self.recompute_orders else None,
                order_in_prologue=self.recompute_orders,
                log_gate=True,
                expand_num=self.num_householder,
                workspace=region["recompute_tensormaps"],
                stream=stream,
            )

        dq_out, dk_out, dv_out = dq, dk, dv
        if self.fold_dq:
            dq_out = region["dq_ho"]
        if self.fold_dk:
            dk_out = region["dk_ho"]
        if self.fold_dv:
            dv_out = region["dv_ho"]
        self.kcache = self.bwd.chunk_gdn_bwd_sm100(
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
            db,
            cu,
            self.scale,
            use_initial_state=state0 is not None,
            d_initial_state=dstate0,
            d_final_state=dstate_in,
            safe_gate=self.safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=self.use_beta_sigmoid,
            work_items=work_items,
            work_count=work_count,
            scheduler_counter=scheduler_bwd if self.bwd_dynamic_scheduling else None,
            scheduler_all=region["scheduler_all"] if self.bwd_orders else None,
            work_item_scratch=region.get("item_scratch") if self.bwd_orders else None,
            order_in_prologue=self.bwd_orders,
            log_gate=True,
            expand_num=self.num_householder,
            workspace=region["tensormaps"],
            stream=stream,
        )
        if self.safe_gate:
            self.scalar_gate_bwd(dg, g, a_log, dt_bias, d_a_log, d_dt_bias, region["gate_part_a"], region["gate_part_dt"], stream=stream)
        if dq_out is not dq or dk_out is not dk or dv_out is not dv:
            for src_ho, dst in ((dq_out, dq), (dk_out, dk), (dv_out, dv)):
                if src_ho is not dst:
                    self.head_group_reduce(src_ho, dst, stream=stream)
        if self.use_qk_l2norm:
            if self.l2norm_bwd is None:
                self.l2norm_bwd = self.build_l2norm_qk_bwd(dq, dk, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream=stream)
            else:
                self.run_l2norm_qk_bwd(self.l2norm_bwd, dq, dk, region["q_n"], region["k_n"], region["inv_q"], region["inv_k"], stream)
        if dq_node is not None:
            if self.gather_bwd is None:
                self.gather_bwd = self.build_gather_bwd(dq, dq_node, dg, dg_node, self.num_householder, stream)
            else:
                self.run_gather_bwd(self.gather_bwd, dq, dq_node, dg, dg_node, stream)
        return None
