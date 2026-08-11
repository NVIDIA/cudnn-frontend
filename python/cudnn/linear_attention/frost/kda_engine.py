# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST KDA engine: KDA nodes on the chunked prefill kernel
(``kernel/kda_prefill_f16.py``) and KDA_BWD nodes on the chunked backward
kernel (``kernel/kda_bprop_f16.py``), Blackwell SM100/SM103, bf16/fp16,
BT=16.  The backward regenerates the per-chunk state checkpoints with the recompute
kernel (``kernel/kda_recompute_f16.py``) when the graph does not provide
one."""

from __future__ import annotations

import math
from typing import Any

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost.buffers import current_device_id
from cudnn.frost.device import multiprocessor_count
from cudnn.frost.workspace import Workspace, WorkspaceLayout
from ..graph_analyzer import FrostLaPlan, check_layouts, frost_la_gate, require, analyze, expect_table

KDA_ALIGN = {"a_log": 4, "beta": 4, "cu_seqlens": 8}
KDA_COMPACT = ("dt_bias",)


def build_kda(graph):
    """The expensive step: import the kernel module (pulls in the Cutlass
    primitives; the cute.compile itself is cached inside the kernel per static
    config and runs on first execute, when the real buffers are known)."""
    nodes = list(graph.nodes)
    if len(nodes) != 1 or getattr(nodes[0].node_type, "name", None) not in ("KDA", "KDA_BWD"):
        raise ValueError("build_kda: graph does not contain exactly one KDA/KDA_BWD node")
    node = nodes[0]
    if node.node_type.name == "KDA_BWD":
        from .kernel import kda_bprop_f16 as bwd_mod
        from .kernel import kda_recompute_f16 as regen_mod

        return CompiledKdaBwd(node, bwd_mod, regen_mod)
    from .kernel import kda_prefill_f16 as kernel_mod

    return CompiledKda(node, kernel_mod)


class KdaFrostEngine(BaseEngine):
    """FROST chunked-kernel backend for single-node KDA graphs (THD layout).

    Default KDA engine on SM100/SM103 (lowest KDA engine_id); serves KDA
    forward and KDA_BWD (with a forward checkpoint recompute when ``state_checkpoints`` is absent)."""

    name = "kda_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

    def check_support(self, graph) -> None:
        import cudnn

        facts = graph._facts_for(analyze)
        frost_la_gate("KdaFrostEngine", facts, "KDA")
        ckpt = facts.checkpoint_every_n_tokens
        if ckpt and (facts.is_bwd or ckpt != 16):
            raise NotImplementedError(f"KdaFrostEngine: checkpoint_every_n_tokens must be 16 (per-chunk) on the forward node (got {ckpt})")
        if facts.is_bwd and (facts.use_beta_sigmoid or facts.safe_gate):
            raise NotImplementedError("KdaFrostEngine: use_beta_sigmoid/safe_gate are forward-node attributes")
        fp32 = cudnn.data_type.FLOAT
        if not facts.is_bwd and facts.use_beta_sigmoid:
            # in-kernel sigmoid: Beta arrives as io-dtype logits
            if facts.beta_dtype not in (facts.io_dtype, None):
                raise NotImplementedError("KdaFrostEngine: use_beta_sigmoid takes io-dtype beta logits")
        else:
            require("KdaFrostEngine", "beta", facts.beta_dtype, fp32)
        require("KdaFrostEngine", "a_log", facts.a_log_dtype, fp32)
        require("KdaFrostEngine", "dt_bias", facts.dt_bias_dtype, fp32)
        if facts.is_bwd:
            for port, got in (("dO", facts.do_dtype), ("state_checkpoints", facts.state_checkpoints_dtype)):
                if got not in (facts.io_dtype, None):
                    raise NotImplementedError(f"KdaFrostEngine: '{port}' must match the io dtype")
            require("KdaFrostEngine", "initial_state", facts.state_dtype, fp32)
            require("KdaFrostEngine", "d_final_state", facts.d_final_state_dtype, fp32)
            require("KdaFrostEngine", "d_initial_state", facts.d_initial_state_dtype, fp32)
            require("KdaFrostEngine", "dG", facts.dg_dtype, fp32)
            require("KdaFrostEngine", "dBeta", facts.dbeta_dtype, fp32)
        else:
            state_dtypes = (fp32, cudnn.data_type.BFLOAT16)
            require("KdaFrostEngine", "initial_state", facts.state_dtype, state_dtypes)
            if facts.io_dtype is not None:
                require("KdaFrostEngine", "state_checkpoints", facts.state_checkpoints_out_dtype, facts.io_dtype)
            require("KdaFrostEngine", "final_state", facts.final_state_dtype, state_dtypes)
            if not facts.state_pair_match:
                raise NotImplementedError("KdaFrostEngine: initial_state and final_state dtypes must match")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return FrostLaPlan(build_kda(graph))


class CompiledKda:
    """Compiled FROST KDA plan: a callable over the resolved node buffers."""

    def __init__(self, node, kernel_mod):
        from .common.split_k import WORK_ITEM_FIELDS, chunk_scratch_rows, compute_ideal_chunks, max_work_items

        self.node = node
        self.kernel = kernel_mod
        self.expect = expect_table(node, KDA_ALIGN)
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self.use_beta_sigmoid = bool(node.params.get("use_beta_sigmoid", False))
        self.safe_gate = bool(node.params.get("safe_gate", False))
        glb = node.params.get("gate_lower_bound")
        self.gate_lower_bound = float(glb) if glb is not None else kernel_mod.DEFAULT_GATE_LOWER_BOUND
        self.has_final_state = "final_state" in node.outputs
        self.has_state_checkpoints = "state_checkpoints" in node.outputs

        q, g = node.inputs["q"], node.inputs["g"]
        self.b_t = kernel_mod.CFG.B_T
        total = q.dim[0]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        layout = WorkspaceLayout()
        self.off_sched = layout.add(8)  # [ticket, done] for the dynamic scheduler
        self.num_sm = multiprocessor_count(current_device_id())
        self.ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
        self.n_tiles = B * HO
        self.work_item_rows = max_work_items(total, B, HO, self.ideal, self.b_t, self.num_sm)
        self.n_heads_out = HO
        self.off_work_items = layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4)
        self.off_item_scratch = layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4)
        self.off_work_count = layout.add(4)
        self.chunk_scratch_rows = chunk_scratch_rows(total, B, self.b_t)
        self.off_chunk_scratch = layout.add(self.chunk_scratch_rows * HO * 4)
        from .common.host import tensormap_workspace_bytes

        self.tensormap_bytes = tensormap_workspace_bytes(kernel_mod, B)
        self.off_tensormaps = layout.add(self.tensormap_bytes, align=128)
        self.ws_bytes = layout.size

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
        a_log = nb.inputs.get("a_log")
        dt_bias = nb.inputs.get("dt_bias")
        check_layouts(
            "KdaFrostEngine (KDA)",
            expect=self.expect,
            compact=KDA_COMPACT,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu,
            initial_state=state0,
            O=o,
            final_state=final_state,
            state_checkpoints=state_checkpoints,
            a_log=a_log,
            dt_bias=dt_bias,
        )

        stream = stream if stream is not None else 0

        ws = Workspace(workspace, self.ws_bytes, "KdaFrostEngine (KDA)")
        sched_ctr = ws.view(self.off_sched, "int32", (2,))
        from .common.split_k import WORK_ITEM_FIELDS

        work_items = ws.view(self.off_work_items, "int32", (self.work_item_rows, WORK_ITEM_FIELDS))
        work_count = ws.view(self.off_work_count, "int32", (1,))
        item_scratch = ws.view(self.off_item_scratch, "int32", (self.work_item_rows, WORK_ITEM_FIELDS))
        chunk_scratch = ws.view(self.off_chunk_scratch, "float32", (self.chunk_scratch_rows, self.n_heads_out))
        from .common.split_k import build_split_table

        build_split_table(
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
            gate_lower_bound=self.gate_lower_bound if self.safe_gate else None,
            sched_ctr=sched_ctr,
            stream=stream,
        )

        ckpt_kwargs = {}
        if self.has_state_checkpoints:
            # the kernel derives the per-sequence checkpoint entry offsets on device
            ckpt_kwargs = dict(checkpoint_every_n_tokens=self.b_t, output_state_checkpoints=state_checkpoints)
        self.kernel.chunk_kda_sm100(
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
            safe_gate=self.safe_gate,
            gate_lower_bound=self.gate_lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
            work_items=work_items,
            work_count=work_count,
            sched_ctr=sched_ctr,
            tensormap_workspace=ws.view(self.off_tensormaps, "int64", (self.tensormap_bytes // 8,)),
            **ckpt_kwargs,
            stream=stream,
        )
        return None


class CompiledKdaBwd:
    """Compiled FROST KDA backward plan.  When the graph carries no ``state_checkpoints``
    input, the workspace holds a regenerated per-chunk checkpoint series (entry-0
    initial-state slots seeded on device, then the recompute kernel re-run
    with shifted checkpoint bounds).  GVA/GQA gradients land in HO-head
    scratch and are reduced back to the native head counts."""

    def __init__(self, node, bwd_mod, regen_mod):
        from .common.split_k import WORK_ITEM_FIELDS, chunk_scratch_rows, compute_ideal_chunks, max_work_items

        self.node = node
        self.bwd = bwd_mod
        self.regen = regen_mod
        self.expect = expect_table(node, KDA_ALIGN)
        from .common.downcast import downcast_state
        from .common.host import tensormap_workspace_bytes

        self.downcast_state = downcast_state
        scale = node.params.get("scale")
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self.use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self.has_state_checkpoints = "state_checkpoints" in node.inputs
        self.has_state0 = "initial_state" in node.inputs
        self.has_dstate0 = "d_initial_state" in node.outputs

        q, g, v = node.inputs["q"], node.inputs["g"], node.inputs["v"]
        self.b_t = bwd_mod.CFG.B_T
        total = q.dim[0]
        HQ, HV = q.dim[1], v.dim[1]
        HO = g.dim[1]
        K, V = q.dim[-1], v.dim[-1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self.io_name = "float16" if node.inputs["q"].get_data_type().name == "HALF" else "bfloat16"
        self.n_heads_out, self._total = HO, total
        layout = WorkspaceLayout()
        self.off_sched = layout.add(16)  # one [ticket, done] ring each for the regen and bwd kernels
        self.num_sm = multiprocessor_count(current_device_id())
        self.bwd_dyn_sched = B * HO <= self.num_sm
        self.ideal = compute_ideal_chunks(total, HO, self.num_sm, self.b_t)
        self.n_tiles = B * HO
        self.work_item_rows = max_work_items(total, B, HO, self.ideal, self.b_t, self.num_sm)
        self.off_work_items = layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4)
        self.off_item_scratch = layout.add(self.work_item_rows * WORK_ITEM_FIELDS * 4)
        self.off_work_count = layout.add(4)
        self.chunk_scratch_rows = chunk_scratch_rows(total, B, self.b_t)
        self.off_chunk_scratch = layout.add(self.chunk_scratch_rows * HO * 4)
        # chunk-0 entering state, io dtype (downcast initial_state; absent = in-kernel zeros)
        self.off_state0_io = layout.add(B * HO * K * V * 2) if self.has_state0 else None
        if not self.has_state_checkpoints:
            self.state_checkpoints_rows = max(total // self.b_t, 1)
            self.off_state_checkpoints = layout.add(self.state_checkpoints_rows * HO * K * V * 2)
            self.regen_tm_bytes = tensormap_workspace_bytes(regen_mod, B)
            self.off_regen_tensormaps = layout.add(self.regen_tm_bytes, align=128)
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
        self.bwd_tm_bytes = tensormap_workspace_bytes(bwd_mod, B)
        self.off_bwd_tensormaps = layout.add(self.bwd_tm_bytes, align=128)
        self.ws_bytes = layout.size

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
        check_layouts(
            "KdaFrostEngine (KDA_BWD)",
            expect=self.expect,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu,
            dO=do,
            state_checkpoints=state_checkpoints,
            initial_state=state0,
            d_final_state=dstate_in,
            dQ=dq,
            dK=dk,
            dV=dv,
            dG=dg,
            dBeta=db,
            d_initial_state=dstate0,
        )
        stream = stream if stream is not None else 0

        HO, total = self.n_heads_out, self._total
        K, V = q.shape[-1], v.shape[-1]
        B = cu.shape[0] - 1
        ws = Workspace(workspace, self.ws_bytes, "KdaFrostEngine (KDA_BWD)")
        sched_regen = ws.view(self.off_sched, "int32", (2,))
        sched_bwd = ws.view(self.off_sched + 8, "int32", (2,))
        from .common.split_k import WORK_ITEM_FIELDS

        work_items = ws.view(self.off_work_items, "int32", (self.work_item_rows, WORK_ITEM_FIELDS))
        work_count = ws.view(self.off_work_count, "int32", (1,))
        item_scratch = ws.view(self.off_item_scratch, "int32", (self.work_item_rows, WORK_ITEM_FIELDS))
        chunk_scratch = ws.view(self.off_chunk_scratch, "float32", (self.chunk_scratch_rows, HO))
        from .common.split_k import build_split_table

        build_split_table(
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
            sched_ctr=ws.view(self.off_sched, "int32", (4,)),
            stream=stream,
        )

        state0_io = None
        if state0 is not None:
            state0_io = ws.view(self.off_state0_io, self.io_name, (B, HO, K, V))
            self.downcast_state(state0, state0_io, stream=stream)
        if self.has_state_checkpoints:
            checkpoint_series = state_checkpoints
        else:
            checkpoint_series = ws.view(self.off_state_checkpoints, self.io_name, (self.state_checkpoints_rows, HO, K, V))
            self.regen.chunk_kda_recompute_sm100(
                k,
                v,
                g,
                beta,
                cu,
                state0,
                None,
                checkpoint_every_n_tokens=self.b_t,
                output_state_checkpoints=checkpoint_series,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                work_items=work_items,
                work_count=work_count,
                sched_ctr=sched_regen,
                tensormap_workspace=ws.view(self.off_regen_tensormaps, "int64", (self.regen_tm_bytes // 8,)),
                stream=stream,
            )

        dq_out, dk_out, dv_out = dq, dk, dv
        if self.fold_dq:
            dq_out = ws.view(self.off_dq_ho, self.io_name, (total, HO, K))
        if self.fold_dk:
            dk_out = ws.view(self.off_dk_ho, self.io_name, (total, HO, K))
        if self.fold_dv:
            dv_out = ws.view(self.off_dv_ho, self.io_name, (total, HO, V))

        self.bwd.chunk_kda_bwd_sm100(
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
            d_initial_state=dstate0 if self.has_dstate0 else None,
            d_final_state=dstate_in,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            work_items=work_items,
            work_count=work_count,
            sched_ctr=sched_bwd if self.bwd_dyn_sched else None,
            tensormap_workspace=ws.view(self.off_bwd_tensormaps, "int64", (self.bwd_tm_bytes // 8,)),
            stream=stream,
        )
        if dq_out is not dq or dk_out is not dk or dv_out is not dv:
            from .common.head_reduce import head_group_reduce

            for src_ho, dst in ((dq_out, dq), (dk_out, dk), (dv_out, dv)):
                if src_ho is not dst:
                    head_group_reduce(src_ho, dst, stream=stream)
        return None
