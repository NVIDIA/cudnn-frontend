# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDN engine: GDN nodes on the chunked prefill kernel
(``kernel/gdn_prefill_f16.py``) and GDN_BWD nodes on the chunked backward
kernel (``kernel/gdn_bprop_f16.py``), Blackwell SM100/SM103, bf16/fp16.
The backward regenerates the per-chunk state checkpoints with the recompute kernel
(``kernel/gdn_recompute_f16.py``) when the graph does not provide one."""

from __future__ import annotations

import math
from typing import Any

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost import buffers
from cudnn.frost.buffers import current_device_id
from cudnn.frost.device import multiprocessor_count
from cudnn.frost.workspace import WorkspaceLayout, carve_plan
from ..graph_analyzer import FrostLaPlan, frost_la_gate, require, analyze


def build_gdn(graph):
    """The expensive step: import the kernel module (pulls in the Cutlass
    primitives; the cute.compile itself is cached inside the kernel per static
    config and runs on first execute, when the real buffers are known)."""
    nodes = list(graph.nodes)
    if len(nodes) != 1 or getattr(nodes[0].node_type, "name", None) not in ("GDN", "GDN_BWD"):
        raise ValueError("build_gdn: graph does not contain exactly one GDN/GDN_BWD node")
    node = nodes[0]
    if node.node_type.name == "GDN_BWD":
        from .kernel import gdn_bprop_f16 as bwd_mod
        from .kernel import gdn_recompute_f16 as regen_mod

        return CompiledGdnBwd(node, bwd_mod, regen_mod)
    from .kernel import gdn_prefill_f16 as kernel_mod

    return CompiledGdn(node, kernel_mod)


class GdnFrostEngine(BaseEngine):
    """FROST chunked-kernel backend for single-node GDN graphs (THD layout).

    Default GDN engine on SM100/SM103 (lowest GDN engine_id); declines
    elsewhere so ranking falls back to ``GdnCuTileEngine``."""

    name = "gdn_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

    def check_support(self, graph) -> None:
        import cudnn

        facts = graph._facts_for(analyze)
        frost_la_gate("GdnFrostEngine", facts, "GDN")
        if facts.use_qk_l2norm:
            raise NotImplementedError("GdnFrostEngine: use_qk_l2norm is not supported (the kernel takes q/k as given)")
        if facts.safe_gate:
            raise NotImplementedError("GdnFrostEngine: safe_gate is not supported (no gate-activation path; the cuTile GDN engine serves it)")
        ckpt = facts.checkpoint_every_n_tokens
        if ckpt and (facts.is_bwd or ckpt % 64 != 0):
            raise NotImplementedError(f"GdnFrostEngine: checkpoint_every_n_tokens must be a positive multiple of 64 on the GDN node (got {ckpt})")
        if not facts.gates_at_ho:
            raise NotImplementedError(f"GdnFrostEngine: g/beta must carry HO = max(q, v) heads ({facts.h_o})")
        fp32 = cudnn.data_type.FLOAT
        io = (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
        state_dtypes = (fp32, cudnn.data_type.BFLOAT16)
        require("GdnFrostEngine", "beta", facts.beta_dtype, fp32)
        require("GdnFrostEngine", "initial_state", facts.state_dtype, state_dtypes)
        require("GdnFrostEngine", "final_state", facts.final_state_dtype, state_dtypes)
        if not facts.state_pair_match:
            raise NotImplementedError("GdnFrostEngine: initial_state and final_state dtypes must match")
        if facts.is_bwd:
            require("GdnFrostEngine", "dO", facts.do_dtype, io)
            if facts.io_dtype is not None:
                require("GdnFrostEngine", "state_checkpoints", facts.state_checkpoints_dtype, facts.io_dtype)
            require("GdnFrostEngine", "d_final_state", facts.d_final_state_dtype, fp32)
            require("GdnFrostEngine", "d_initial_state", facts.d_initial_state_dtype, fp32)
            require("GdnFrostEngine", "dG", facts.dg_dtype, fp32)
            require("GdnFrostEngine", "dBeta", facts.dbeta_dtype, fp32)
        elif facts.io_dtype is not None:
            require("GdnFrostEngine", "state_checkpoints", facts.state_checkpoints_out_dtype, facts.io_dtype)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return FrostLaPlan(build_gdn(graph))


class CompiledGdn:
    """Compiled FROST GDN plan: a callable over the resolved node buffers."""

    def __init__(self, node, kernel_mod):
        from .common.split_k import WORK_ITEM_FIELDS, build_split_table, chunk_scratch_rows, compute_ideal_chunks, max_work_items

        self.node = node
        self.kernel = kernel_mod
        self.build_split_table = build_split_table
        self.plan_name = "GdnFrostEngine (GDN)"
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])

        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self.b_t = kernel_mod.CFG.B_T
        total = q.dim[0]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.has_final_state = "final_state" in node.outputs
        self.ckpt = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self.has_state_checkpoints = "state_checkpoints" in node.outputs
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        # cuts only for chunk-granular checkpoint cadences, never in batch-invariant mode
        self.split = self.ckpt in (0, self.b_t) and not self.batch_invariant

        layout = WorkspaceLayout()
        from .common.host import tensormap_workspace_bytes

        self.tensormap_words = tensormap_workspace_bytes(kernel_mod, B) // 8
        self.off_tensormaps = layout.add(self.tensormap_words * 8)
        self.off_sched = layout.add(8)  # [ticket, done] for the dynamic scheduler
        self.num_sm = multiprocessor_count(current_device_id())
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
        self.ws_bytes = layout.size
        regions = [
            (self.off_tensormaps, "int64", (self.tensormap_words,)),
            (self.off_sched, "int32", (2,)),
            (self.off_work_items, "int32", (self.work_item_rows, WORK_ITEM_FIELDS)),
            (self.off_work_count, "int32", (1,)),
        ]
        if self.split:
            regions += [
                (self.off_item_scratch, "int32", (self.work_item_rows, WORK_ITEM_FIELDS)),
                (self.off_chunk_scratch, "float32", (self.chunk_scratch_rows, self.n_heads_out)),
            ]
        self.carve = carve_plan("GdnFrostEngine (GDN)", regions)

    def workspace_bytes(self) -> int:
        return self.ws_bytes

    def __call__(self, node_buffers, *, workspace=None, stream=None) -> Any:
        nb = node_buffers[self.node]
        q = nb.inputs["q"]
        k = nb.inputs["k"]
        v = nb.inputs["v"]
        g = nb.inputs["g"]
        beta = nb.inputs["beta"]
        cu = nb.inputs["cu_seqlens"]
        state0 = nb.inputs.get("initial_state")
        o = nb.outputs["O"]
        final_state = nb.outputs["final_state"] if self.has_final_state else None
        state_checkpoints = nb.outputs["state_checkpoints"] if self.has_state_checkpoints else None
        stream = stream if stream is not None else 0
        if self.split:
            tensormaps, sched_ctr, work_items, work_count, item_scratch, chunk_scratch = workspace.carve(self.carve)
        else:
            tensormaps, sched_ctr, work_items, work_count = workspace.carve(self.carve)
            item_scratch = chunk_scratch = None
        self.build_split_table(
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
            sched_ctr=sched_ctr,
            split=self.split,
            stream=stream,
        )

        self.kernel.chunk_gdn_sm100(
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
            work_items=work_items,
            work_count=work_count,
            sched_ctr=sched_ctr,
            checkpoint_every_n_tokens=self.ckpt,
            output_state_checkpoints=state_checkpoints,
            log_gate=True,
            workspace=tensormaps,
            stream=stream,
        )
        return None


class CompiledGdnBwd:
    """Compiled FROST GDN bprop plan: a callable over the resolved node buffers.

    Produces dQ/dK/dV/dG/dBeta; consumes the forward per-chunk states through
    the node's ``state_checkpoints`` input, or regenerates them with the recompute
    (checkpoint-only) kernel when the port is absent."""

    def __init__(self, node, bwd_mod, regen_mod):
        from .common.split_k import WORK_ITEM_FIELDS, build_split_table, chunk_scratch_rows, compute_ideal_chunks, max_work_items

        self.node = node
        self.bwd = bwd_mod
        self.regen = regen_mod
        self.build_split_table = build_split_table
        self.plan_name = "GdnFrostEngine (GDN_BWD)"
        from .common.downcast import downcast_state
        from .common.host import tensormap_workspace_bytes

        self.downcast_state = downcast_state
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])

        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self.b_t = bwd_mod.CFG.B_T
        total = q.dim[0]
        K, V = q.dim[-1], v.dim[-1]
        HQ, HV = q.dim[1], v.dim[1]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.has_state_checkpoints = "state_checkpoints" in node.inputs
        self.has_state0 = "initial_state" in node.inputs
        self.io_name = "float16" if node.inputs["q"].get_data_type().name == "HALF" else "bfloat16"

        self.num_sm = multiprocessor_count(current_device_id())
        self.bwd_dyn_sched = B * HO <= self.num_sm
        self.batch_invariant = bool(node.params.get("batch_invariant", False))
        # cuts never in batch-invariant mode: whole-sequence items keep each
        # sequence's math independent of the batch composition
        self.split = not self.batch_invariant
        layout = WorkspaceLayout()
        self.off_sched = layout.add(16)  # one [ticket, done] ring each for the regen and bwd kernels
        self.tensormap_words = tensormap_workspace_bytes(bwd_mod, B) // 8
        self.off_tensormaps = layout.add(self.tensormap_words * 8)
        # chunk-0 entering state, io dtype (downcast initial_state; absent = in-kernel zeros)
        self.off_state0_io = layout.add(B * HO * K * V * 2) if self.has_state0 else None
        if not self.has_state_checkpoints:
            self.state_checkpoints_rows = max(total // self.b_t + B, 1)
            self.off_state_checkpoints = layout.add(self.state_checkpoints_rows * HO * K * V * 2)
            self.regen_tensormap_words = tensormap_workspace_bytes(regen_mod, B) // 8
            self.off_regen_tensormaps = layout.add(self.regen_tensormap_words * 8)
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
        self.n_heads_out, self.total = HO, total
        self.ws_bytes = layout.size
        regions = [
            ("sched_regen", self.off_sched, "int32", (2,)),
            ("sched_bwd", self.off_sched + 8, "int32", (2,)),
            ("sched_all", self.off_sched, "int32", (4,)),
            ("tensormaps", self.off_tensormaps, "int64", (self.tensormap_words,)),
            ("work_items", self.off_work_items, "int32", (self.work_item_rows, WORK_ITEM_FIELDS)),
            ("work_count", self.off_work_count, "int32", (1,)),
        ]
        if self.split:
            regions.append(("item_scratch", self.off_item_scratch, "int32", (self.work_item_rows, WORK_ITEM_FIELDS)))
            regions.append(("chunk_scratch", self.off_chunk_scratch, "float32", (self.chunk_scratch_rows, HO)))
        if self.has_state0:
            regions.append(("state0_io", self.off_state0_io, self.io_name, (B, HO, K, V)))
        if not self.has_state_checkpoints:
            regions.append(("state_checkpoints", self.off_state_checkpoints, self.io_name, (self.state_checkpoints_rows, HO, K, V)))
            regions.append(("regen_tensormaps", self.off_regen_tensormaps, "int64", (self.regen_tensormap_words,)))
        if self.fold_dq:
            regions.append(("dq_ho", self.off_dq_ho, self.io_name, (total, HO, K)))
        if self.fold_dk:
            regions.append(("dk_ho", self.off_dk_ho, self.io_name, (total, HO, K)))
        if self.fold_dv:
            regions.append(("dv_ho", self.off_dv_ho, self.io_name, (total, HO, V)))
        self.carve_names = [name for name, _off, _dt, _shape in regions]
        self.carve = carve_plan("GdnFrostEngine (GDN_BWD)", [(off, dt, shape) for _name, off, dt, shape in regions])

    def workspace_bytes(self) -> int:
        return self.ws_bytes

    def __call__(self, node_buffers, *, workspace=None, stream=None) -> Any:
        nb = node_buffers[self.node]
        q = nb.inputs["q"]
        k = nb.inputs["k"]
        v = nb.inputs["v"]
        g = nb.inputs["g"]
        beta = nb.inputs["beta"]
        cu = nb.inputs["cu_seqlens"]
        do = nb.inputs["dO"]
        state_checkpoints = nb.inputs.get("state_checkpoints")
        state0 = nb.inputs.get("initial_state")
        dstate_in = nb.inputs.get("d_final_state")
        dq = nb.outputs["dQ"]
        dk = nb.outputs["dK"]
        dv = nb.outputs["dV"]
        dg = nb.outputs["dG"]
        db = nb.outputs["dBeta"]
        dstate0 = nb.outputs.get("d_initial_state")
        HO, total = self.n_heads_out, self.total
        K, V = q.shape[-1], v.shape[-1]
        B = cu.shape[0] - 1

        stream = stream if stream is not None else 0
        region = dict(zip(self.carve_names, workspace.carve(self.carve)))
        sched_regen = region["sched_regen"]
        sched_bwd = region["sched_bwd"]
        work_items = region["work_items"]
        work_count = region["work_count"]
        self.build_split_table(
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
            sched_ctr=region["sched_all"],
            split=self.split,
            stream=stream,
        )

        state0_io = None
        if state0 is not None:
            if state0.dtype == self.io_name and buffers.is_contiguous(tuple(state0.shape), state0.stride()) and state0.data_ptr() % 16 == 0:
                # already the staged form: io dtype, compact, aligned — bind directly
                state0_io = state0
            else:
                state0_io = region["state0_io"]
                self.downcast_state(state0, state0_io, stream=stream)
        if self.has_state_checkpoints:
            checkpoint_series = state_checkpoints
        else:
            checkpoint_series = region["state_checkpoints"]
            self.regen.chunk_gdn_recompute_sm100(
                k,
                v,
                g,
                beta,
                cu,
                state0,
                None,
                checkpoint_every_n_tokens=self.b_t,
                output_state_checkpoints=checkpoint_series,
                work_items=work_items,
                work_count=work_count,
                sched_ctr=sched_regen,
                log_gate=True,
                workspace=region["regen_tensormaps"],
                stream=stream,
            )

        dq_out, dk_out, dv_out = dq, dk, dv
        if self.fold_dq:
            dq_out = region["dq_ho"]
        if self.fold_dk:
            dk_out = region["dk_ho"]
        if self.fold_dv:
            dv_out = region["dv_ho"]
        self.bwd.chunk_gdn_bwd_sm100(
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
            initial_state=state0_io,
            d_initial_state=dstate0,
            d_final_state=dstate_in,
            work_items=work_items,
            work_count=work_count,
            sched_ctr=sched_bwd if self.bwd_dyn_sched else None,
            log_gate=True,
            workspace=region["tensormaps"],
            stream=stream,
        )
        if dq_out is not dq or dk_out is not dk or dv_out is not dv:
            from .common.head_reduce import head_group_reduce

            for src_ho, dst in ((dq_out, dq), (dk_out, dk), (dv_out, dv)):
                if src_ho is not dst:
                    head_group_reduce(src_ho, dst, stream=stream)
        return None
