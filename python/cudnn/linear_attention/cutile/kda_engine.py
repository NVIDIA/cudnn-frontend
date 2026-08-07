# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA (Kimi Delta Attention) execution backend: KDA / KDA_BWD nodes on the
chunked cuTile kernels (``kernels/kda_chunk_cutile``)."""

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
    NodeType.KDA: ("q", "k", "v", "g", "beta", "cu_seqlens"),
    NodeType.KDA_BWD: ("q", "k", "v", "g", "beta", "cu_seqlens", "dO"),
}


def _node_ws_layout(node):
    """Static carve plan for one node's ``chunk_kda`` pipeline intermediates:
    name -> (offset, dtype, shape). The chunk count is data-dependent
    (varlen), so the 'h'/'dh' entries are SIZED with the upper bound
    cdiv(total,64)+N and their shape carries None in the NT slot, substituted
    at execute. A KDA_BWD node carries the union of the forward re-run's
    intermediates and the backward temporaries, so both live disjointly in
    one carve.  Terminal pipeline buffers (the forward's ``o``/``fs``; the
    backward's boundary casts / l2norm outputs, ``dv2`` and ``dh0``) are NOT
    carved — execute plants the caller's output buffers under those names."""
    from .kernels.kda_chunk_cutile import _BT as BT, _cdiv, _next_power_of_2

    q, v, g, cu = (node.inputs[p] for p in ("q", "v", "g", "cu_seqlens"))
    total, H, K = q.dim
    HV, V = v.dim[1], v.dim[2]
    N = cu.dim[0] - 1
    io = _dtype_name(q.get_data_type())
    f32 = "float32"
    BC = 32 if K >= 64 else 16  # fwd_intra sub-chunk (see chunk_kda_fwd_intra)
    NK = _cdiv(K, min(64, _next_power_of_2(K)))  # bwd_intra K-split (see chunk_kda_bwd_intra)
    NT_bound = _cdiv(total, BT) + N
    l2norm = bool(node.params.get("use_qk_l2norm", False))

    size = 0
    table = {}

    def add(name, dtype, shape):
        nonlocal size
        nbytes = buffers.DTYPE_ITEMSIZE[dtype]
        shape = tuple(NT_bound if s is None else int(s) for s in shape)
        for s in shape:
            nbytes *= s
        table[name] = (size, dtype, shape)
        size += (nbytes + 127) & ~127  # 128B-aligned sequential carve

    add("chunk_table", "int32", (NT_bound, 2))
    add("chunk_count", "int32", (1,))
    add("chunk_offsets", "int32", (N + 1,))
    add("dummy", "int32", (4,))  # inert stub backing for absent optional kernel args

    # chunk_kda forward (also re-run inside KDA_BWD)
    add("g_cum", f32, (total, HV, K))
    add("Aqk", io, (total, HV, BT))
    add("Akk", io, (total, HV, BT))
    add("Akkd", f32, (total, HV, BC))
    add("w", io, (total, HV, K))
    add("u", io, (total, HV, V))
    add("qg", io, (total, HV, K))
    add("kg", io, (total, HV, K))
    add("h", io, (None, HV, K, V))
    add("v_new", io, (total, HV, V))
    if node.node_type == NodeType.KDA_BWD:
        add("o", io, (total, HV, V))  # discarded output of the forward re-run
    if l2norm:
        add("q_norm", io, (total * H, K))
        add("q_rstd", f32, (total * H,))
        add("k_norm", io, (total * H, K))
        add("k_rstd", f32, (total * H,))
    if node.node_type == NodeType.KDA_BWD:
        add("dAqk", f32, (total, HV, BT))
        add("dv_dAv", io, (total, HV, V))
        add("dh", io, (None, HV, K, V))
        add("dv_dhu", io, (total, HV, V))
        add("dq", f32, (total, HV, K))
        add("dk", f32, (total, HV, K))
        add("dg", f32, (total, HV, K))
        if _dtype_name(node.inputs["beta"].get_data_type()) != f32:
            add("db", f32, (total, HV))
        add("dAkk", f32, (total, HV, BT))
        add("dq2", f32, (total, HV, K))
        add("dk2", f32, (total, HV, K))
        if HV != H:
            add("dq_hred", f32, (total, H, K))
            add("dk_hred", f32, (total, H, K))
        add("db2", f32, (NK, total, HV))
        add("dg2", f32, (total, HV, K))
        if _dtype_name(g.get_data_type()) != f32:
            add("dg_cum", f32, (total, HV, K))
    return size, table


class _KdaCuTilePlan(CompiledPlan):
    """Carve plan over the caller's workspace (see GdnCuTileEngine's
    ``_CuTilePlan``): static per-node layout; the buffer arrives with every
    execute (the explicit-workspace convention)."""

    def __init__(self, engine, graph):
        self._engine = engine
        self._layouts = [(node, *_node_ws_layout(node)) for node in graph.nodes]
        # nodes execute sequentially, each re-carving the same buffer
        self._ws_bytes = max(nbytes for _, nbytes, _ in self._layouts)

    def get_workspace_size(self) -> int:
        return self._ws_bytes

    def execute(self, graph, uid_to_data, ctx) -> None:
        self._engine._execute(resolve_node_buffers(graph, uid_to_data), self, ctx)


class KdaCuTileEngine(BaseEngine):
    """cuTile chunked-kernel backend for single-node KDA graphs (THD layout)."""

    name = "kda_cutile"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

    def check_support(self, graph: "pygraph") -> None:
        if buffers.current_sm() is None:
            raise NotImplementedError("KdaCuTileEngine requires a CUDA device")
        try:
            from cuda.bindings import runtime as _rt

            err, _cudart_version = _rt.cudaRuntimeGetVersion()
            if int(err) != 0:
                raise NotImplementedError(f"KdaCuTileEngine: cudaRuntimeGetVersion failed ({err})")
        except ImportError as e:
            raise NotImplementedError(f"KdaCuTileEngine requires cuda.bindings: {e}")
        if _cudart_version < 13030:
            raise NotImplementedError(f"KdaCuTileEngine requires CUDA 13.3+ (found {_cudart_version})")
        try:
            from .kernels.kda_chunk_cutile import (  # noqa: F401
                chunk_kda,
            )
        except ImportError as e:
            raise NotImplementedError(f"KdaCuTileEngine requires the cuda.tile runtime: {e}")

        import cudnn

        supported_dtypes = (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, None)
        if not graph.nodes:
            raise NotImplementedError("KdaCuTileEngine: empty graph")
        for node in graph.nodes:
            required = _REQUIRED_PORTS.get(node.node_type)
            if required is None:
                raise NotImplementedError(f"KdaCuTileEngine only supports KDA/KDA_BWD nodes, got {node.node_type.name}")
            for port in required:
                if port not in node.inputs:
                    raise NotImplementedError(f"KdaCuTileEngine: {node.node_type.name} node '{node.name}' is missing input '{port}'")
            if int(node.params.get("checkpoint_every_n_tokens", 0) or 0) > 0 or "H" in node.outputs:
                raise NotImplementedError("KdaCuTileEngine: per-chunk H output is not supported")
            q, k, v = (node.inputs[p] for p in ("q", "k", "v"))
            for p in ("q", "k", "v"):
                t = node.inputs[p]
                if t.get_data_type() not in supported_dtypes:
                    raise NotImplementedError(f"KdaCuTileEngine: '{p}' must be fp16/bf16, got {t.get_data_type()}")
                if t.dim and len(t.dim) != 3:
                    raise NotImplementedError(f"KdaCuTileEngine: '{p}' must be THD [total_T, heads, dim], got rank {len(t.dim)}")
            if q.dim and k.dim and q.dim[1] != k.dim[1]:
                raise NotImplementedError(f"KdaCuTileEngine: q and k head counts differ ({q.dim[1]} vs {k.dim[1]})")
            if q.dim and v.dim and v.dim[1] % q.dim[1] != 0:
                raise NotImplementedError(
                    f"KdaCuTileEngine: v heads ({v.dim[1]}) must be a multiple of q heads ({q.dim[1]}; GQA-style v broadcast is FROST-only)"
                )
            if q.dim and q.dim[-1] > 256:
                raise NotImplementedError(f"KdaCuTileEngine: head dim K must be <= 256, got {q.dim[-1]}")
            if node.inputs["cu_seqlens"].get_data_type() not in (cudnn.data_type.INT32, None):
                raise NotImplementedError("KdaCuTileEngine: cu_seqlens must be int32 (the device-side table builder reads it directly)")
            io = q.get_data_type()
            f32 = cudnn.data_type.FLOAT
            if node.node_type == NodeType.KDA:
                out_dtypes = {"O": io, "final_state": f32}
                required_out = ("O",)
            else:
                out_dtypes = {
                    "dQ": io,
                    "dK": io,
                    "dV": io,
                    "dG": node.inputs["g"].get_data_type(),
                    "dBeta": node.inputs["beta"].get_data_type(),
                    "d_initial_state": f32,
                }
                required_out = ("dQ", "dK", "dV", "dG", "dBeta")
                if ("initial_state" in node.inputs) != ("d_initial_state" in node.outputs):
                    raise NotImplementedError("KdaCuTileEngine: d_initial_state output must be requested iff initial_state is given")
            for port in required_out:
                if port not in node.outputs:
                    raise NotImplementedError(f"KdaCuTileEngine: {node.node_type.name} node '{node.name}' is missing output '{port}'")
            for port, want in out_dtypes.items():
                t = node.outputs.get(port)
                if t is not None and t.get_data_type() not in (want, None):
                    raise NotImplementedError(f"KdaCuTileEngine: output '{port}' must be {want} (written in place), got {t.get_data_type()}")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return _KdaCuTilePlan(self, graph)

    def _execute(self, node_buffers, plan, ctx) -> None:
        from .kernels.common import build_chunk_table
        from .kernels.kda_chunk_cutile import _BT

        stream = getattr(ctx, "stream", None)
        stream = 0 if stream is None else stream
        ws = Workspace(getattr(ctx, "workspace", None), plan._ws_bytes, "KdaCuTileEngine")
        for node, _nbytes, table in plan._layouts:
            nb = node_buffers[node]
            cu_seqlens = nb.inputs["cu_seqlens"]
            N = node.inputs["cu_seqlens"].dim[0] - 1
            bufs = {name: ws.view(off, dt, shape) for name, (off, dt, shape) in table.items()}
            bound = bufs["chunk_table"].shape[0]
            build_chunk_table(bufs["chunk_table"], bufs["chunk_count"], bufs["chunk_offsets"], cu_seqlens, N, _BT, bound, stream=stream)
            if node.node_type == NodeType.KDA:
                self._execute_fwd(node, nb, bufs, stream)
            else:
                self._execute_bwd(node, nb, bufs, stream)

    @staticmethod
    def _state_f32(s0):
        # the kernel wants the recurrent state in fp32; callers convert
        if s0 is not None and not str(s0.dtype).endswith("float32"):
            raise ValueError("KdaCuTileEngine: state ports must be fp32 (callers convert)")
        return s0

    def _execute_fwd(self, node, nb, bufs, stream) -> None:
        from .kernels.kda_chunk_cutile import chunk_kda

        want_state = "final_state" in node.outputs
        # terminal pipeline buffers = the caller's output buffers
        bufs["o"] = nb.outputs["O"]
        if want_state:
            bufs["fs"] = nb.outputs["final_state"]
        chunk_kda(
            nb.inputs["q"],
            nb.inputs["k"],
            nb.inputs["v"],
            nb.inputs["g"],
            nb.inputs["beta"],
            scale=node.params.get("scale"),
            initial_state=self._state_f32(nb.inputs.get("initial_state")),
            output_final_state=want_state,
            use_qk_l2norm_in_kernel=bool(node.params.get("use_qk_l2norm", False)),
            cu_seqlens=nb.inputs["cu_seqlens"],
            chunk_indices=bufs["chunk_table"],
            bufs=bufs,
            stream=stream,
        )

    def _execute_bwd(self, node, nb, bufs, stream) -> None:
        from .kernels.common import reshaped
        from .kernels.kda_chunk_cutile import chunk_kda_grad

        total, H, K = node.inputs["q"].dim
        cu_seqlens = nb.inputs["cu_seqlens"]
        initial_state = nb.inputs.get("initial_state")
        dht = nb.inputs.get("d_final_state")
        do, q, k, v = nb.inputs["dO"], nb.inputs["q"], nb.inputs["k"], nb.inputs["v"]
        g, beta = nb.inputs["g"], nb.inputs["beta"]
        scale = node.params.get("scale") or K**-0.5

        # terminal pipeline buffers = the caller's output buffers
        dQ_out = nb.outputs["dQ"]
        dK_out = nb.outputs["dK"]
        if node.params.get("use_qk_l2norm", False):
            bufs["dq_l2"] = reshaped(dQ_out, (total * H, K))
            bufs["dk_l2"] = reshaped(dK_out, (total * H, K))
        bufs["dq_cast"] = dQ_out
        bufs["dk_cast"] = dK_out
        bufs["dv2"] = nb.outputs["dV"]
        bufs["dg_cast" if "dg_cum" in bufs else "dg_cum"] = nb.outputs["dG"]
        bufs["db_cast" if "db" in bufs else "db"] = nb.outputs["dBeta"]
        if initial_state is not None:
            bufs["dh0"] = nb.outputs["d_initial_state"]

        chunk_kda_grad(
            q,
            k,
            v,
            g,
            beta,
            do,
            dht=self._state_f32(dht),
            scale=scale,
            initial_state=self._state_f32(initial_state),
            use_qk_l2norm_in_kernel=bool(node.params.get("use_qk_l2norm", False)),
            cu_seqlens=cu_seqlens,
            chunk_indices=bufs["chunk_table"],
            bufs=bufs,
            stream=stream,
        )
