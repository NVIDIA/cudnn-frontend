# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDN-2 engine: GDN2 nodes on the chunked prefill kernel
(``kernel/gdn2_prefill_f16.py``, Blackwell SM100/SM103, bf16/fp16, BT=16).
Forward only (GDN2_BWD declines); the only GDN-2 engine — no cuTile
fallback."""

from __future__ import annotations

import math
from typing import Any

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace, WorkspaceLayout, carve_plan
from ..engine_utils import _FrostPlan, _require_dtype, _require_state_pair


def _the_gdn2_node(graph):
    nodes = list(graph.nodes)
    if len(nodes) != 1:
        return None
    node = nodes[0]
    if getattr(node.node_type, "name", None) not in ("GDN2", "GDN2_BWD"):
        return None
    return node


def _check_common(node) -> None:
    """Shape/dtype gates for the prefill kernel."""
    import cudnn

    q, k, v = (node.inputs[p] for p in ("q", "k", "v"))
    io_dtypes = {q.get_data_type(), k.get_data_type(), v.get_data_type()} - {None}
    if len(io_dtypes) > 1:
        raise NotImplementedError(f"Gdn2FrostEngine: q/k/v dtypes must match, got {io_dtypes}")
    for p, t in (("q", q), ("k", k), ("v", v)):
        if t.get_data_type() not in (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF, None):
            raise NotImplementedError(f"Gdn2FrostEngine: '{p}' must be bf16 or fp16, got {t.get_data_type()}")
        if not t.dim or len(t.dim) != 3:
            raise NotImplementedError(f"Gdn2FrostEngine: '{p}' must be THD [total_T, heads, dim]")
    if q.dim[-1] != 128 or v.dim[-1] != 128:
        raise NotImplementedError(f"Gdn2FrostEngine: head dims must be 128 (the recurrent state is 128x128), got K={q.dim[-1]} V={v.dim[-1]}")
    if k.dim[1] != q.dim[1]:
        raise NotImplementedError(f"Gdn2FrostEngine: q and k head counts differ ({q.dim[1]} vs {k.dim[1]})")
    if v.dim[1] % q.dim[1] != 0:
        raise NotImplementedError(f"Gdn2FrostEngine: v heads ({v.dim[1]}) must be a multiple of q heads ({q.dim[1]})")

    # kernel-native operand dtypes: buffers pass through without staging
    fp32 = cudnn.data_type.FLOAT
    io = q.get_data_type()
    _require_dtype("Gdn2FrostEngine", node, "g", fp32)
    if io is not None:
        _require_dtype("Gdn2FrostEngine", node, "beta", io)
        _require_dtype("Gdn2FrostEngine", node, "w", io)
    _require_dtype("Gdn2FrostEngine", node, "cu_seqlens", cudnn.data_type.INT32)
    _require_dtype("Gdn2FrostEngine", node, "initial_state", (fp32, cudnn.data_type.BFLOAT16))
    _require_dtype("Gdn2FrostEngine", node, "final_state", (fp32, cudnn.data_type.BFLOAT16), out=True)
    _require_state_pair("Gdn2FrostEngine", node)


def build_gdn2(graph):
    """The expensive step: import the kernel module (pulls in the Cutlass
    primitives; the cute.compile itself is cached inside the kernel per static
    config and runs on first execute, when the real buffers are known)."""
    node = _the_gdn2_node(graph)
    if node is None or node.node_type.name != "GDN2":
        raise ValueError("build_gdn2: graph does not contain exactly one GDN2 node")
    from .kernel import gdn2_prefill_f16 as kernel_mod

    return CompiledGdn2(node, kernel_mod)


class Gdn2FrostEngine(BaseEngine):
    """FROST chunked-kernel backend for single-node GDN-2 graphs (THD layout).

    The only GDN-2 engine (SM100/SM103, forward only); declines ``GDN2_BWD``
    (the FROST backward kernel is a stub)."""

    name = "gdn2_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

    def check_support(self, graph) -> None:
        node = _the_gdn2_node(graph)
        if node is None:
            raise NotImplementedError("Gdn2FrostEngine supports exactly one GDN2 node")
        if node.node_type.name == "GDN2_BWD":
            raise NotImplementedError("Gdn2FrostEngine: the FROST GDN-2 backward kernel is a stub")
        sm = buffers.current_sm()
        if sm is None or not (100 <= sm <= 103):
            raise NotImplementedError(f"Gdn2FrostEngine requires SM100-SM103 (found {sm})")
        try:
            import cutlass.experimental.primitives  # noqa: F401 — availability probe: ImportError = decline
        except ImportError as exc:
            raise NotImplementedError(f"Gdn2FrostEngine requires the Cutlass DSL with cutlass.experimental.primitives: {exc}") from exc
        for port in ("q", "k", "v", "g", "beta", "w", "cu_seqlens"):
            if port not in node.inputs:
                raise NotImplementedError(f"Gdn2FrostEngine: GDN2 node '{node.name}' is missing input '{port}'")
        if int(node.params.get("checkpoint_every_n_tokens", 0) or 0) or "H" in node.outputs:
            raise NotImplementedError("Gdn2FrostEngine: the per-chunk H output lands with the backward kernel (jopark/kda_gdn2_bprop)")
        if node.node_type.name != "GDN2" and node.params.get("use_beta_w_sigmoid", False):
            raise NotImplementedError("Gdn2FrostEngine: use_beta_w_sigmoid is a forward-node attribute")
        _check_common(node)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return _FrostPlan(build_gdn2(graph))


class CompiledGdn2:
    """Compiled FROST GDN-2 plan: a callable over the resolved node buffers."""

    def __init__(self, node, kernel_mod):
        from .common.split_k import WORK_ITEM_FIELDS, chunk_scratch_rows, compute_ideal_chunks, max_work_items

        self._node = node
        self._kernel = kernel_mod
        scale = node.params.get("scale")
        self._scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])
        self._use_qk_l2norm = bool(node.params.get("use_qk_l2norm", False))
        self._use_beta_w_sigmoid = bool(node.params.get("use_beta_w_sigmoid", False))
        self._has_fs = "final_state" in node.outputs

        q, g = node.inputs["q"], node.inputs["g"]
        self._b_t = kernel_mod.CFG.B_T
        total = q.dim[0]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        layout = WorkspaceLayout()
        self._off_sched = layout.add(8)  # [ticket, done] for the dynamic scheduler
        self._num_sm = kernel_mod._device_sm_count()
        self._ideal = compute_ideal_chunks(total, HO, self._num_sm, self._b_t)
        self._n_tiles = B * HO
        self._work_item_rows = max_work_items(total, B, HO, self._ideal, self._b_t, self._num_sm)
        self._n_heads_out = HO
        self._off_work_items = layout.add(self._work_item_rows * WORK_ITEM_FIELDS * 4)
        self._off_item_scratch = layout.add(self._work_item_rows * WORK_ITEM_FIELDS * 4)
        self._off_work_count = layout.add(4)
        self._chunk_scratch_rows = chunk_scratch_rows(total, B, self._b_t)
        self._off_chunk_scratch = layout.add(self._chunk_scratch_rows * HO * 4)
        self._tensormap_bytes = kernel_mod.get_workspace_size(B, HO, HO)
        self._off_tensormaps = layout.add(self._tensormap_bytes, align=128)
        self._ws_bytes = layout.size

        self._carve = carve_plan(
            "gdn2",
            [
                (self._off_sched, "int32", (2,)),
                (self._off_work_items, "int32", (self._work_item_rows, WORK_ITEM_FIELDS)),
                (self._off_item_scratch, "int32", (self._work_item_rows, WORK_ITEM_FIELDS)),
                (self._off_work_count, "int32", (1,)),
                (self._off_chunk_scratch, "float32", (self._chunk_scratch_rows, HO)),
                (self._off_tensormaps, "int64", (self._tensormap_bytes // 8,)),
            ],
        )

    def workspace_bytes(self) -> int:
        return self._ws_bytes

    def __call__(self, node_buffers, *, workspace=None, stream=None) -> Any:
        nb = node_buffers[self._node]
        q = nb.inputs["q"]
        k = nb.inputs["k"]
        v = nb.inputs["v"]
        g = nb.inputs["g"]
        beta = nb.inputs["beta"]
        w = nb.inputs["w"]
        cu = nb.inputs["cu_seqlens"]
        s0 = nb.inputs.get("initial_state")
        o = nb.outputs["O"]
        fs = nb.outputs["final_state"] if self._has_fs else None

        stream = stream if stream is not None else 0

        ws = workspace
        sched_ctr, work_items, item_scratch, work_count, chunk_scratch, tensormaps = ws.carve(self._carve)
        from .common.split_k import build_split_table

        build_split_table(
            g,
            cu,
            work_items,
            work_count,
            ideal_chunks=self._ideal,
            n_tiles=self._n_tiles,
            num_sms=self._num_sm,
            b_t=self._b_t,
            chunk_scratch=chunk_scratch,
            item_scratch=item_scratch,
            log_gate=True,
            sched_ctr=sched_ctr,
            stream=stream,
        )

        self._kernel.chunk_gdn2_sm100(
            q,
            k,
            v,
            g,
            beta,
            w,
            o,
            cu,
            s0,
            fs,
            self._scale,
            use_qk_l2norm_in_kernel=self._use_qk_l2norm,
            use_beta_w_sigmoid_in_kernel=self._use_beta_w_sigmoid,
            work_items=work_items,
            work_count=work_count,
            sched_ctr=sched_ctr,
            tensormap_workspace=tensormaps,
            stream=stream,
        )
        return None
