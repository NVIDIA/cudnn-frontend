# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN (Gated DeltaNet) execution backend: GDN / GDN_BWD nodes on the
chunked cuTile kernels (``kernels/gdn_chunk_cutile``)."""

from typing import TYPE_CHECKING, Any, Dict

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan, resolve_node_buffers
from cudnn.graph_types import NodeType
from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace
from cudnn.linear_attention.engine_utils import _dtype_name

if TYPE_CHECKING:
    from cudnn.pygraph import pygraph

_REQUIRED_PORTS = {
    NodeType.GDN: ("q", "k", "v", "g", "beta", "cu_seqlens"),
    NodeType.GDN_BWD: ("q", "k", "v", "g", "beta", "cu_seqlens", "dO"),
}


def _node_ws_layout(node):
    """Static carve plan for one node's pipeline intermediates: name ->
    (offset, dtype-name, shape). The chunk count is data-dependent (varlen),
    so chunk-indexed entries are sized and SHAPED at the bound
    ``cdiv(total, 64) + N`` — the device-built table's sentinel tail keeps
    bound-gridded launches inert past the real count.  Terminal pipeline
    buffers (``o``/``final_state``; the backward's ``dq``/``dk`` finals,
    ``wy_dv``, ``dg_cum``, ``db``, ``dh0``) are NOT carved — execute plants
    the caller's output buffers under those names."""
    from .kernels.gdn_chunk_cutile import _BT, _cdiv, _next_power_of_2

    q, v, cu = (node.inputs[p] for p in ("q", "v", "cu_seqlens"))
    total, H, K = q.dim
    HV, V = v.dim[1], v.dim[2]
    N = cu.dim[0] - 1
    io = _dtype_name(q.get_data_type())
    f32 = "float32"
    NT_bound = _cdiv(total, _BT) + N
    l2norm = bool(node.params.get("use_qk_l2norm", False))

    size = 0
    table = {}

    def add(name, dtype, shape):
        nonlocal size
        nbytes = buffers.DTYPE_ITEMSIZE[dtype]
        for s in shape:
            nbytes *= int(s)
        table[name] = (size, dtype, tuple(int(s) for s in shape))
        size += (nbytes + 127) & ~127  # 128B-aligned sequential carve

    add("chunk_table", "int32", (NT_bound, 2))
    add("chunk_count", "int32", (1,))
    add("chunk_offsets", "int32", (N + 1,))
    add("dummy", "int32", (4,))  # inert stub backing for absent optional kernel args
    add("g_cum", f32, (total, HV))
    add("A", io, (total, HV, _BT))
    add("w", io, (total, HV, K))
    add("u", io, (total, HV, V))
    add("h", io, (NT_bound, HV, K, V))
    add("v_new", io, (total, HV, V))
    if l2norm:
        add("q_norm", io, (total * H, K))
        add("q_rstd", f32, (total * H,))
        add("k_norm", io, (total * H, K))
        add("k_rstd", f32, (total * H,))
    if node.node_type == NodeType.GDN_BWD:
        add("dv", io, (total, HV, V))
        add("dh", io, (NT_bound, HV, K, V))
        add("dv2", io, (total, HV, V))
        NK = _cdiv(K, min(max(_next_power_of_2(K), 16), 64))
        add("dg_nk", f32, (NK, total, HV))
        add("dw", io, (total, HV, K))
        if HV != H or l2norm:
            # dq/dk are finals only without l2norm on an MHA config; every
            # other combination keeps them (or their head-reduced pair) as
            # pipeline intermediates
            add("dq", io, (total, HV, K))
            add("dk", io, (total, HV, K))
        if HV != H:
            add("wy_dk_hred", io, (total, H, K))
            if l2norm:
                add("dq_hred", io, (total, H, K))
                add("dk_hred", io, (total, H, K))
        add("dg", f32, (total, HV))
        add("wy_dk", io, (total, HV, K))
        add("wy_dg", f32, (total, HV))
    return size, table


class _CuTilePlan(CompiledPlan):
    """Carve plan over the caller's workspace: the layout is static per node;
    the buffer arrives with every execute (the explicit-workspace convention)."""

    def __init__(self, engine, graph):
        self._engine = engine
        self._layouts = [(node, *_node_ws_layout(node)) for node in graph.nodes]
        # nodes execute sequentially, each re-carving the same buffer
        self._ws_bytes = max(nbytes for _, nbytes, _ in self._layouts)

    def get_workspace_size(self) -> int:
        return self._ws_bytes

    def execute(self, graph, uid_to_data, ctx) -> None:
        self._engine._execute(resolve_node_buffers(graph, uid_to_data), self, ctx)


class GdnCuTileEngine(BaseEngine):
    """cuTile chunked-kernel backend for single-node GDN graphs (THD layout)."""

    name = "gdn_cutile"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

    def check_support(self, graph: "pygraph") -> None:
        if buffers.current_sm() is None:
            raise NotImplementedError("GdnCuTileEngine requires a CUDA device")
        try:
            from cuda.bindings import runtime as _rt

            err, _cudart_version = _rt.cudaRuntimeGetVersion()
            if int(err) != 0:
                raise NotImplementedError(f"GdnCuTileEngine: cudaRuntimeGetVersion failed ({err})")
        except ImportError as e:
            raise NotImplementedError(f"GdnCuTileEngine requires cuda.bindings: {e}")
        if _cudart_version < 13030:
            raise NotImplementedError(f"GdnCuTileEngine requires CUDA 13.3+ (found {_cudart_version})")
        try:
            from .kernels.gdn_chunk_cutile import (  # noqa: F401
                chunk_gated_delta_rule,
            )
        except ImportError as e:
            raise NotImplementedError(f"GdnCuTileEngine requires the cuda.tile runtime: {e}")

        import cudnn

        supported_dtypes = (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, None)
        if not graph.nodes:
            raise NotImplementedError("GdnCuTileEngine: empty graph")
        for node in graph.nodes:
            required = _REQUIRED_PORTS.get(node.node_type)
            if required is None:
                raise NotImplementedError(f"GdnCuTileEngine only supports GDN/GDN_BWD nodes, got {node.node_type.name}")
            for port in required:
                if port not in node.inputs:
                    raise NotImplementedError(f"GdnCuTileEngine: {node.node_type.name} node '{node.name}' is missing input '{port}'")
            if int(node.params.get("checkpoint_every_n_tokens", 0) or 0) > 0 or "H" in node.outputs:
                raise NotImplementedError("GdnCuTileEngine: per-chunk H output is not supported")
            q, k, v = (node.inputs[p] for p in ("q", "k", "v"))
            for p in ("q", "k", "v"):
                t = node.inputs[p]
                if t.get_data_type() not in supported_dtypes:
                    raise NotImplementedError(f"GdnCuTileEngine: '{p}' must be fp16/bf16, got {t.get_data_type()}")
                if t.dim and len(t.dim) != 3:
                    raise NotImplementedError(f"GdnCuTileEngine: '{p}' must be THD [total_T, heads, dim], got rank {len(t.dim)}")
            if q.dim and k.dim and q.dim[1] != k.dim[1]:
                raise NotImplementedError(f"GdnCuTileEngine: q and k head counts differ ({q.dim[1]} vs {k.dim[1]})")
            if q.dim and v.dim and v.dim[1] % q.dim[1] != 0:
                raise NotImplementedError(
                    f"GdnCuTileEngine: v heads ({v.dim[1]}) must be a multiple of q heads ({q.dim[1]}; GQA-style v broadcast is FROST-only)"
                )
            if q.dim and q.dim[-1] > 256:
                raise NotImplementedError(f"GdnCuTileEngine: head dim K must be <= 256, got {q.dim[-1]}")
            if node.inputs["cu_seqlens"].get_data_type() not in (cudnn.data_type.INT32, None):
                raise NotImplementedError("GdnCuTileEngine: cu_seqlens must be int32 (the device-side table builder reads it directly)")
            io = q.get_data_type()
            f32 = cudnn.data_type.FLOAT
            if node.node_type == NodeType.GDN:
                out_dtypes = {"O": io, "final_state": f32}
                required_out = ("O",)
            else:
                beta_dt = node.inputs["beta"].get_data_type()
                out_dtypes = {"dQ": io, "dK": io, "dV": io, "dG": f32, "dBeta": beta_dt, "d_initial_state": f32}
                required_out = ("dQ", "dK", "dV", "dG", "dBeta")
                if ("initial_state" in node.inputs) != ("d_initial_state" in node.outputs):
                    raise NotImplementedError("GdnCuTileEngine: d_initial_state output must be requested iff initial_state is given")
            for port in required_out:
                if port not in node.outputs:
                    raise NotImplementedError(f"GdnCuTileEngine: {node.node_type.name} node '{node.name}' is missing output '{port}'")
            for port, want in out_dtypes.items():
                t = node.outputs.get(port)
                if t is not None and t.get_data_type() not in (want, None):
                    raise NotImplementedError(f"GdnCuTileEngine: output '{port}' must be {want} (written in place), got {t.get_data_type()}")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return _CuTilePlan(self, graph)

    def _execute(self, node_buffers, plan, ctx) -> None:
        from .kernels.common import build_chunk_table
        from .kernels.gdn_chunk_cutile import _BT

        stream = getattr(ctx, "stream", None)
        stream = 0 if stream is None else stream
        ws = Workspace(getattr(ctx, "workspace", None), plan._ws_bytes, "GdnCuTileEngine")
        for node, _nbytes, table in plan._layouts:
            nb = node_buffers[node]
            cu_seqlens = nb.inputs["cu_seqlens"]
            N = node.inputs["cu_seqlens"].dim[0] - 1
            bufs = {name: ws.view(off, dt, shape) for name, (off, dt, shape) in table.items()}
            bound = bufs["chunk_table"].shape[0]
            build_chunk_table(bufs["chunk_table"], bufs["chunk_count"], bufs["chunk_offsets"], cu_seqlens, N, _BT, bound, stream=stream)
            if node.node_type == NodeType.GDN:
                self._execute_fwd(node, nb, bufs, stream)
            else:
                self._execute_bwd(node, nb, bufs, stream)

    def _execute_fwd(self, node, nb, bufs, stream) -> None:
        from .kernels.gdn_chunk_cutile import chunk_gated_delta_rule_fwd, l2norm_fwd

        want_state = "final_state" in node.outputs
        K = node.inputs["q"].dim[-1]
        q, k, v = nb.inputs["q"], nb.inputs["k"], nb.inputs["v"]
        g, beta = nb.inputs["g"], nb.inputs["beta"]
        scale = node.params.get("scale") or K**-0.5
        if node.params.get("use_qk_l2norm", False):
            q, _ = l2norm_fwd(q, out=bufs["q_norm"], rstd_out=bufs["q_rstd"], stream=stream)
            k, _ = l2norm_fwd(k, out=bufs["k_norm"], rstd_out=bufs["k_rstd"], stream=stream)
        # terminal pipeline buffers = the caller's output buffers
        bufs["o"] = nb.outputs["O"]
        if want_state:
            bufs["final_state"] = nb.outputs["final_state"]
        chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=nb.inputs.get("initial_state"),
            output_final_state=want_state,
            cu_seqlens=nb.inputs["cu_seqlens"],
            chunk_indices=bufs["chunk_table"],
            bufs=bufs,
            stream=stream,
        )

    def _execute_bwd(self, node, nb, bufs, stream) -> None:
        from .kernels.gdn_chunk_cutile import (
            RCP_LN2,
            _BT,
            chunk_gated_delta_rule_bwd,
            chunk_gated_delta_rule_fwd_intra,
            chunk_local_cumsum,
            l2norm_bwd,
            l2norm_fwd,
        )
        from .kernels.common import add_inplace, reshaped

        H, K = node.inputs["q"].dim[1], node.inputs["q"].dim[-1]
        HV = node.inputs["v"].dim[1]
        q, k, v = nb.inputs["q"], nb.inputs["k"], nb.inputs["v"]
        g, beta, do = nb.inputs["g"], nb.inputs["beta"], nb.inputs["dO"]
        cu_seqlens = nb.inputs["cu_seqlens"]
        initial_state = nb.inputs.get("initial_state")
        dht = nb.inputs.get("d_final_state")
        chunk_indices = bufs["chunk_table"]
        scale = node.params.get("scale") or K**-0.5
        l2norm = bool(node.params.get("use_qk_l2norm", False))
        if l2norm:
            q, q_rstd = l2norm_fwd(q, out=bufs["q_norm"], rstd_out=bufs["q_rstd"], stream=stream)
            k, k_rstd = l2norm_fwd(k, out=bufs["k_norm"], rstd_out=bufs["k_rstd"], stream=stream)

        # terminal pipeline buffers = the caller's output buffers (with
        # l2norm, dq/dk stay carves and l2norm_bwd writes the caller's)
        if not l2norm:
            bufs["dq" if HV == H else "dq_hred"] = nb.outputs["dQ"]
            bufs["dk" if HV == H else "dk_hred"] = nb.outputs["dK"]
        bufs["wy_dv"] = nb.outputs["dV"]
        bufs["dg_cum"] = nb.outputs["dG"]
        bufs["db"] = nb.outputs["dBeta"]
        if initial_state is not None:
            bufs["dh0"] = nb.outputs["d_initial_state"]

        # recompute the forward's cumulative gate and intra-chunk WY matrix
        # (the backward entry expects them)
        g_cum = chunk_local_cumsum(g, chunk_size=_BT, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, out=bufs["g_cum"], stream=stream)
        _, _, A = chunk_gated_delta_rule_fwd_intra(
            k=k, v=v, g=g_cum, beta=beta, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, bufs=bufs, compute_wu=False, stream=stream
        )
        dq, dk, dk2, _, _, _, _, _, _ = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g_cum,
            beta=beta,
            A=A,
            scale=scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            bufs=bufs,
            stream=stream,
        )
        if l2norm:
            l2norm_bwd(q, q_rstd, dq, out=nb.outputs["dQ"], bufs=bufs, stream=stream)
            l2norm_bwd(k, k_rstd, dk, dy2=dk2, out=nb.outputs["dK"], bufs=bufs, stream=stream)
        else:
            # dk/dk2 are the head-reduced finals for GVA, HV-head for MHA
            n_dk = 1
            for s_ in dk.shape:
                n_dk *= int(s_)
            add_inplace(reshaped(dk, (n_dk,)), reshaped(dk2, (n_dk,)), n_dk, stream=stream)
