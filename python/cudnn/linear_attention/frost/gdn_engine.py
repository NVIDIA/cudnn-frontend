# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDN engine: GDN / GDN_BWD nodes on the chunked prefill kernels
(``kernel/gdn_prefill_f16.py`` / ``kernel/gdn_bprop_f16.py``, Blackwell
SM100/SM103, bf16/fp16)."""

from __future__ import annotations

import math
from typing import Any

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace, WorkspaceLayout
from ..engine_utils import _FrostPlan, _check_contiguous, _require_dtype, _require_state_pair


def _the_gdn_node(graph):
    nodes = list(graph.nodes)
    if len(nodes) != 1:
        return None
    node = nodes[0]
    if getattr(node.node_type, "name", None) not in ("GDN", "GDN_BWD"):
        return None
    return node


def _io_dtype_name(node) -> str:
    """The io dtype's buffer-level name ('bfloat16' / 'float16')."""
    import cudnn

    return "bfloat16" if node.inputs["q"].get_data_type() == cudnn.data_type.BFLOAT16 else "float16"


def _check_common(node) -> None:
    """Shape/dtype gates shared by the fwd and bwd kernels."""
    import cudnn

    q, k, v = (node.inputs[p] for p in ("q", "k", "v"))
    io_dtypes = {q.get_data_type(), k.get_data_type(), v.get_data_type()} - {None}
    if len(io_dtypes) > 1:
        raise NotImplementedError(f"GdnFrostEngine: q/k/v dtypes must match, got {io_dtypes}")
    for p, t in (("q", q), ("k", k), ("v", v)):
        if t.get_data_type() not in (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF, None):
            raise NotImplementedError(f"GdnFrostEngine: '{p}' must be bf16 or fp16, got {t.get_data_type()}")
        if not t.dim or len(t.dim) != 3:
            raise NotImplementedError(f"GdnFrostEngine: '{p}' must be THD [total_T, heads, dim]")
    if q.dim[-1] != 128 or v.dim[-1] != 128:
        raise NotImplementedError(f"GdnFrostEngine: head dims must be 128 (the recurrent state is 128x128), got K={q.dim[-1]} V={v.dim[-1]}")
    hq, hk, hv = q.dim[1], k.dim[1], v.dim[1]
    if hq != hk:
        raise NotImplementedError(f"GdnFrostEngine: q and k head counts differ ({hq} vs {hk})")
    if hv != hq and max(hq, hv) % min(hq, hv) != 0:
        # GVA (v-heads grouped over q-heads) or GQA (q-heads grouped over
        # v-heads, the kernel's opposite direction)
        raise NotImplementedError(f"GdnFrostEngine: q heads ({hq}) and v heads ({hv}) must be equal or one a multiple of the other")
    ho = hq if hq >= hv else hv
    for p in ("g", "beta"):
        t = node.inputs[p]
        if t.dim and t.dim[1] != ho:
            raise NotImplementedError(f"GdnFrostEngine: '{p}' must carry HO = max(q, v) heads ({ho}), got {t.dim[1]}")

    # kernel-native operand dtypes: buffers pass through without staging
    fp32 = cudnn.data_type.FLOAT
    io = (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
    state = (fp32, cudnn.data_type.BFLOAT16)
    _require_dtype("GdnFrostEngine", node, "g", fp32)
    _require_dtype("GdnFrostEngine", node, "beta", fp32)
    _require_dtype("GdnFrostEngine", node, "cu_seqlens", cudnn.data_type.INT32)
    _require_dtype("GdnFrostEngine", node, "initial_state", state)
    _require_dtype("GdnFrostEngine", node, "final_state", state, out=True)
    _require_state_pair("GdnFrostEngine", node)
    if node.node_type.name == "GDN_BWD":
        _require_dtype("GdnFrostEngine", node, "dO", io)
        _require_dtype("GdnFrostEngine", node, "h", io)
        _require_dtype("GdnFrostEngine", node, "d_final_state", fp32)
        _require_dtype("GdnFrostEngine", node, "d_initial_state", fp32, out=True)
        _require_dtype("GdnFrostEngine", node, "dG", fp32, out=True)
        _require_dtype("GdnFrostEngine", node, "dBeta", fp32, out=True)


def build_gdn(graph):
    """The expensive step: import the kernel module (pulls in the Cutlass
    primitives; the cute.compile itself is cached inside the kernel per static
    config and runs on first execute, when the real buffers are known)."""
    node = _the_gdn_node(graph)
    if node is None:
        raise ValueError("build_gdn: graph does not contain exactly one GDN/GDN_BWD node")
    if node.node_type.name == "GDN_BWD":
        from .kernel import gdn_bprop_f16 as kernel_mod

        return CompiledGdnBwd(node, kernel_mod)
    from .kernel import gdn_prefill_f16 as kernel_mod

    return CompiledGdn(node, kernel_mod)


class GdnFrostEngine(BaseEngine):
    """FROST chunked-kernel backend for single-node GDN graphs (THD layout).

    Default GDN engine on SM100/SM103 (lowest GDN engine_id); declines
    elsewhere so the router falls back to ``GdnCuTileEngine``."""

    name = "gdn_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

    def check_support(self, graph) -> None:
        node = _the_gdn_node(graph)
        if node is None:
            raise NotImplementedError("GdnFrostEngine supports exactly one GDN/GDN_BWD node")
        sm = buffers.current_sm()
        if sm is None or not (100 <= sm <= 103):
            raise NotImplementedError(f"GdnFrostEngine requires SM100-SM103 (found {sm})")
        try:
            import cutlass.experimental.primitives  # noqa: F401 — availability probe: ImportError = decline
        except ImportError as exc:
            raise NotImplementedError(f"GdnFrostEngine requires the Cutlass DSL with cutlass.experimental.primitives: {exc}") from exc
        if node.params.get("use_qk_l2norm", False):
            raise NotImplementedError("GdnFrostEngine: use_qk_l2norm is not supported (the kernel takes q/k as given)")
        ports = ("q", "k", "v", "g", "beta", "cu_seqlens")
        if node.node_type.name == "GDN_BWD":
            ports += ("dO",)
        for port in ports:
            if port not in node.inputs:
                raise NotImplementedError(f"GdnFrostEngine: {node.node_type.name} node '{node.name}' is missing input '{port}'")
        ckpt = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        if ckpt and (node.node_type.name != "GDN" or ckpt % 64 != 0):
            raise NotImplementedError(f"GdnFrostEngine: checkpoint_every_n_tokens must be a positive multiple of 64 on the GDN node (got {ckpt})")
        _check_common(node)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return _FrostPlan(build_gdn(graph))


def _splits_for(gate, cu_i32, work_items, item_scratch, chunk_scratch, work_count, ideal_chunks, n_tiles, num_sms, b_t, stream, log_gate=False, sched_ctr=None):
    """Launch the split pipeline (stream-ordered; the scan zeroes the count
    and the scheduler ticket ring)."""
    from .common.split_k import build_split_table

    build_split_table(
        gate,
        cu_i32,
        work_items,
        work_count,
        ideal_chunks=ideal_chunks,
        n_tiles=n_tiles,
        num_sms=num_sms,
        b_t=b_t,
        chunk_scratch=chunk_scratch,
        item_scratch=item_scratch,
        log_gate=log_gate,
        sched_ctr=sched_ctr,
        stream=stream,
    )
    return work_items, work_count


class CompiledGdn:
    """Compiled FROST GDN plan: a callable over the resolved node buffers."""

    def __init__(self, node, kernel_mod):
        from .common.split_k import WORK_ITEM_FIELDS, chunk_scratch_rows, compute_ideal_chunks, max_work_items

        self._node = node
        self._kernel = kernel_mod
        scale = node.params.get("scale")
        self._scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])

        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self._b_t = kernel_mod.CFG.B_T
        total, K, V = q.dim[0], q.dim[2], v.dim[2]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self._has_fs = "final_state" in node.outputs
        self._ckpt = int(node.params.get("checkpoint_every_n_tokens", 0) or 0)
        self._has_h = "H" in node.outputs
        # split work items assume chunk-granular state boundaries; other
        # checkpoint cadences keep the serial per-(b,h) walk
        self._split = self._ckpt in (0, self._b_t)

        layout = WorkspaceLayout()
        self._off_tensormaps = layout.add(kernel_mod.get_workspace_size(B, q.dim[1], v.dim[1]))
        self._tensormap_words = kernel_mod.get_workspace_size(B, q.dim[1], v.dim[1]) // 8
        self._off_sched = layout.add(8)  # [ticket, done] for the dynamic scheduler
        if self._split:
            self._num_sm = kernel_mod._device_sm_count()
            self._ideal = compute_ideal_chunks(total, HO, self._num_sm, self._b_t)
            self._n_tiles = B * HO
            self._n_heads_out = HO
            self._work_item_rows = max_work_items(total, B, HO, self._ideal, self._b_t, self._num_sm)
            self._off_work_items = layout.add(self._work_item_rows * WORK_ITEM_FIELDS * 4)
            self._off_item_scratch = layout.add(self._work_item_rows * WORK_ITEM_FIELDS * 4)
            self._off_work_count = layout.add(4)
            self._chunk_scratch_rows = chunk_scratch_rows(total, B, self._b_t)
            self._off_chunk_scratch = layout.add(self._chunk_scratch_rows * HO * 4)
        self._ws_bytes = layout.size

    def workspace_bytes(self) -> int:
        return self._ws_bytes

    def __call__(self, node_buffers, *, workspace=None, stream=None) -> Any:
        nb = node_buffers[self._node]
        q = nb.inputs["q"]
        k = nb.inputs["k"]
        v = nb.inputs["v"]
        g = nb.inputs["g"]
        beta = nb.inputs["beta"]
        cu = nb.inputs["cu_seqlens"]
        s0 = nb.inputs.get("initial_state")
        o = nb.outputs["O"]
        fs = nb.outputs["final_state"] if self._has_fs else None
        h = nb.outputs["H"] if self._has_h else None
        _check_contiguous("GdnFrostEngine (GDN)", q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu, initial_state=s0, O=o, final_state=fs)

        ws = Workspace(workspace, self._ws_bytes, "GdnFrostEngine (GDN)")
        stream = stream if stream is not None else 0
        sched_ctr = ws.view(self._off_sched, "int32", (2,))
        work_items = work_count = None
        if self._split:
            from .common.split_k import WORK_ITEM_FIELDS

            work_items = ws.view(self._off_work_items, "int32", (self._work_item_rows, WORK_ITEM_FIELDS))
            item_scratch = ws.view(self._off_item_scratch, "int32", (self._work_item_rows, WORK_ITEM_FIELDS))
            work_count = ws.view(self._off_work_count, "int32", (1,))
            chunk_scratch = ws.view(self._off_chunk_scratch, "float32", (self._chunk_scratch_rows, self._n_heads_out))
            _splits_for(
                g,
                cu,
                work_items,
                item_scratch,
                chunk_scratch,
                work_count,
                self._ideal,
                self._n_tiles,
                self._num_sm,
                self._b_t,
                stream,
                log_gate=True,
                sched_ctr=sched_ctr,
            )
        else:
            buffers.memset_zero_async(sched_ctr.data_ptr(), sched_ctr.nbytes, stream)

        self._kernel.chunk_gdn_sm100(
            q,
            k,
            v,
            g,
            beta,
            o,
            cu,
            s0,
            fs,
            self._scale,
            work_items=work_items,
            work_count=work_count,
            sched_ctr=sched_ctr,
            checkpoint_every_n_tokens=self._ckpt,
            output_h=h,
            log_gate=True,
            workspace=ws.view(self._off_tensormaps, "int64", (self._tensormap_words,)),
            stream=stream,
        )
        return None


class CompiledGdnBwd:
    """Compiled FROST GDN bprop plan: a callable over the resolved node buffers.

    Produces dQ/dK/dV/dG/dBeta; consumes the forward per-chunk states through
    the node's ``h`` input, or regenerates them with a forward H-dump run when
    the port is absent."""

    def __init__(self, node, kernel_mod):
        from .common.split_k import WORK_ITEM_FIELDS, chunk_scratch_rows, compute_ideal_chunks, max_work_items

        self._node = node
        self._kernel = kernel_mod
        scale = node.params.get("scale")
        self._scale = float(scale) if scale is not None else 1.0 / math.sqrt(node.inputs["q"].dim[-1])

        from .kernel import gdn_prefill_f16 as fwd_mod

        self._fwd = fwd_mod
        q, v, g = node.inputs["q"], node.inputs["v"], node.inputs["g"]
        self._b_t = kernel_mod.CFG.B_T
        total, K, V = q.dim[0], q.dim[2], v.dim[2]
        HQ, HV = q.dim[1], v.dim[1]
        HO = g.dim[1]
        B = node.inputs["cu_seqlens"].dim[0] - 1
        self._has_h = "h" in node.inputs
        self._has_s0 = "initial_state" in node.inputs
        self._has_dht = "d_final_state" in node.inputs
        self._has_ds0 = "d_initial_state" in node.outputs
        self._io_name = _io_dtype_name(node)

        # multi-wave grids pay ~2-3% for the ticket ring in the BACKWARD
        # (its TMA-LDG publisher drives five load streams); static stride there
        self._bwd_dyn_sched = B * HO <= kernel_mod._device_sm_count()
        layout = WorkspaceLayout()
        self._off_sched = layout.add(16)  # one [ticket, done] ring each for the regen and bwd kernels
        self._tensormap_words = kernel_mod.get_workspace_size(B, HQ, HV) // 8
        self._off_tensormaps = layout.add(self._tensormap_words * 8)
        # regenerated H series (io dtype [n, HO, K, V]); the entry count is
        # bounded statically so no device sync sizes it
        self._h_rows = 0 if self._has_h else max(total // self._b_t + B, 1)
        self._off_h = layout.add(self._h_rows * HO * K * V * 2) if self._h_rows else None
        # io-downcast initial state (chunk 0's S reads it via its own
        # descriptor set)
        self._off_s0io = layout.add(B * HO * K * V * 2) if self._has_s0 else None
        if not self._has_h:
            self._fwd_tensormap_words = fwd_mod.get_workspace_size(B, HO, HO) // 8
            self._off_fwd_tensormaps = layout.add(self._fwd_tensormap_words * 8)
        self._is_gva = HV > HQ
        self._is_gqa = HQ > HV
        if self._is_gva:
            self._off_dq_ho = layout.add(total * HV * K * 2)
            self._off_dk_ho = layout.add(total * HV * K * 2)
        if self._is_gqa:
            self._off_dv_ho = layout.add(total * HQ * V * 2)
        self._num_sm = kernel_mod._device_sm_count()
        self._ideal = compute_ideal_chunks(total, HO, self._num_sm, self._b_t)
        self._n_tiles = B * HO
        self._work_item_rows = max_work_items(total, B, HO, self._ideal, self._b_t, self._num_sm)
        self._off_work_items = layout.add(self._work_item_rows * WORK_ITEM_FIELDS * 4)
        self._off_item_scratch = layout.add(self._work_item_rows * WORK_ITEM_FIELDS * 4)
        self._off_work_count = layout.add(4)
        self._chunk_scratch_rows = chunk_scratch_rows(total, B, self._b_t)
        self._off_chunk_scratch = layout.add(self._chunk_scratch_rows * HO * 4)
        self._shapes = (total, HQ, HV, HO, K, V, B)
        self._ws_bytes = layout.size

    def workspace_bytes(self) -> int:
        return self._ws_bytes

    def __call__(self, node_buffers, *, workspace=None, stream=None):
        nb = node_buffers[self._node]
        q = nb.inputs["q"]
        k = nb.inputs["k"]
        v = nb.inputs["v"]
        g = nb.inputs["g"]
        beta = nb.inputs["beta"]
        cu = nb.inputs["cu_seqlens"]
        do = nb.inputs["dO"]
        h_in = nb.inputs.get("h") if self._has_h else None
        s0 = nb.inputs.get("initial_state") if self._has_s0 else None
        dht = nb.inputs.get("d_final_state") if self._has_dht else None
        ds0 = nb.outputs.get("d_initial_state")
        dq = nb.outputs["dQ"]
        dk = nb.outputs["dK"]
        dv = nb.outputs["dV"]
        dg = nb.outputs["dG"]
        dbeta = nb.outputs["dBeta"]
        _check_contiguous(
            "GdnFrostEngine (GDN_BWD)",
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu,
            dO=do,
            h=h_in,
            initial_state=s0,
            d_final_state=dht,
            d_initial_state=ds0,
            dQ=dq,
            dK=dk,
            dV=dv,
            dG=dg,
            dBeta=dbeta,
        )

        ws = Workspace(workspace, self._ws_bytes, "GdnFrostEngine (GDN_BWD)")
        total, HQ, HV, HO, K, V, B = self._shapes

        stream = stream if stream is not None else 0
        sched_fwd = ws.view(self._off_sched, "int32", (2,))
        sched_bwd = ws.view(self._off_sched + 8, "int32", (2,))
        from .common.split_k import WORK_ITEM_FIELDS

        work_items = ws.view(self._off_work_items, "int32", (self._work_item_rows, WORK_ITEM_FIELDS))
        item_scratch = ws.view(self._off_item_scratch, "int32", (self._work_item_rows, WORK_ITEM_FIELDS))
        work_count = ws.view(self._off_work_count, "int32", (1,))
        chunk_scratch = ws.view(self._off_chunk_scratch, "float32", (self._chunk_scratch_rows, HO))
        _splits_for(
            g,
            cu,
            work_items,
            item_scratch,
            chunk_scratch,
            work_count,
            self._ideal,
            self._n_tiles,
            self._num_sm,
            self._b_t,
            stream,
            log_gate=True,
            sched_ctr=ws.view(self._off_sched, "int32", (4,)),
        )

        s0_io = None
        if s0 is not None:
            s0_io = ws.view(self._off_s0io, self._io_name, (B, HO, K, V))
            self._fwd.downcast_state(s0, s0_io, stream=stream)
        if h_in is not None:
            h = h_in
        else:
            h = ws.view(self._off_h, self._io_name, (self._h_rows, HO, K, V))
            self._fwd.chunk_gdn_sm100(
                q,
                k,
                v,
                g,
                beta,
                None,
                cu,
                s0,
                None,
                self._scale,
                checkpoint_every_n_tokens=self._b_t,
                output_h=h,
                work_items=work_items,
                work_count=work_count,
                sched_ctr=sched_fwd,
                log_gate=True,
                workspace=ws.view(self._off_fwd_tensormaps, "int64", (self._fwd_tensormap_words,)),
                stream=stream,
            )

        dq_out, dk_out, dv_out = dq, dk, dv
        if self._is_gva:
            dq_out = ws.view(self._off_dq_ho, self._io_name, (total, HV, K))
            dk_out = ws.view(self._off_dk_ho, self._io_name, (total, HV, K))
        if self._is_gqa:
            dv_out = ws.view(self._off_dv_ho, self._io_name, (total, HQ, V))
        self._kernel.chunk_gdn_bwd_sm100(
            q,
            k,
            v,
            g,
            beta,
            do,
            h,
            dq_out,
            dk_out,
            dv_out,
            dg,
            dbeta,
            cu,
            self._scale,
            initial_state=s0_io,
            d_initial_state=ds0,
            d_final_state=dht,
            work_items=work_items,
            work_count=work_count,
            sched_ctr=sched_bwd if self._bwd_dyn_sched else None,
            log_gate=True,
            workspace=ws.view(self._off_tensormaps, "int64", (self._tensormap_words,)),
            stream=stream,
        )
        if self._is_gva:
            from .common.head_reduce import head_group_reduce

            head_group_reduce(dq_out, dq, stream=stream)
            head_group_reduce(dk_out, dk, stream=stream)
        if self._is_gqa:
            from .common.head_reduce import head_group_reduce

            head_group_reduce(dv_out, dv, stream=stream)
        return None
