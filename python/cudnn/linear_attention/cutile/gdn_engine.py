# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuTile GDN engine: GDN / GDN_BWD nodes on the chunked cuTile kernels
(``kernels/gdn_chunk_cutile``)."""

from typing import TYPE_CHECKING

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan, resolve_node_buffers
from cudnn.graph_types import NodeType

from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace
from ..graph_analyzer import check_layouts_compact, analyze, expect_table, to_buffer_dtype

# entry base-alignment expectations; ports not listed assume 16
CUTILE_ALIGN = {"cu_seqlens": 4, "a_log": 4, "beta": 4}

if TYPE_CHECKING:
    from cudnn._pygraph import pygraph


def node_ws_layout(node):
    """Static carve plan for one node's pipeline intermediates: name ->
    (offset, dtype-name, shape). The chunk count is data-dependent (varlen),
    so chunk-indexed entries are sized and SHAPED at the bound
    ``cdiv(total, 64) + N`` — the device-built table's sentinel tail keeps
    bound-gridded launches inert past the real count.  Terminal pipeline
    buffers (``o``/``final_state``; the backward's ``dq``/``dk`` finals,
    ``wy_dv``, ``dg_cum``, ``db``, ``dstate0``) are NOT carved — execute plants
    the caller's output buffers under those names."""
    from .kernels.gdn_chunk_cutile import BT_CHUNK, cdiv, next_power_of_2

    q, v, cu = (node.inputs[p] for p in ("q", "v", "cu_seqlens"))
    total, H, K = q.dim
    HV, V = v.dim[1], v.dim[2]
    N = cu.dim[0] - 1
    io = to_buffer_dtype(q.get_data_type())
    f32 = "float32"
    NT_bound = cdiv(total, BT_CHUNK) + N
    l2norm = bool(node.params.get("use_qk_l2norm", False))

    size = 0
    table = {}

    def carve(name, dtype, shape):
        nonlocal size
        nbytes = buffers.DTYPE_ITEMSIZE[dtype]
        for s in shape:
            nbytes *= int(s)
        table[name] = (size, dtype, tuple(int(s) for s in shape))
        size += (nbytes + 127) & ~127  # 128B-aligned sequential carve

    carve("chunk_table", "int32", (NT_bound, 2))
    carve("chunk_count", "int32", (1,))
    carve("chunk_offsets", "int32", (N + 1,))
    carve("dummy", "int32", (4,))  # inert stub backing for absent optional kernel args
    carve("g_cum", f32, (total, HV))
    carve("A", io, (total, HV, BT_CHUNK))
    carve("w", io, (total, HV, K))
    carve("u", io, (total, HV, V))
    carve("state_checkpoints", io, (NT_bound, HV, K, V))
    carve("v_new", io, (total, HV, V))
    if l2norm:
        carve("q_norm", io, (total * H, K))
        carve("q_rstd", f32, (total * H,))
        carve("k_norm", io, (total * H, K))
        carve("k_rstd", f32, (total * H,))
    if node.node_type == NodeType.GDN_BWD:
        carve("dv", io, (total, HV, V))
        carve("dstate", io, (NT_bound, HV, K, V))
        carve("dv2", io, (total, HV, V))
        NK = cdiv(K, min(max(next_power_of_2(K), 16), 64))
        carve("dg_nk", f32, (NK, total, HV))
        carve("dw", io, (total, HV, K))
        if HV != H or l2norm:
            # dQ/dK are finals only without l2norm on an MHA config; every
            # other combination keeps them (or their head-reduced pair) as
            # pipeline intermediates
            carve("dq", io, (total, HV, K))
            carve("dk", io, (total, HV, K))
        if HV != H:
            carve("wy_dk_hred", io, (total, H, K))
            if l2norm:
                carve("dq_hred", io, (total, H, K))
                carve("dk_hred", io, (total, H, K))
        carve("dg", f32, (total, HV))
        carve("wy_dk", io, (total, HV, K))
        carve("wy_dg", f32, (total, HV))
    return size, table


class GdnCuTilePlan(CompiledPlan):
    """Carve plan over the caller's workspace: the layout is static per node;
    the buffer arrives with every execute."""

    def __init__(self, graph):
        self.layouts = [(node, *node_ws_layout(node)) for node in graph.nodes]
        self.expects = {node: expect_table(node, CUTILE_ALIGN) for node in graph.nodes}
        # nodes execute sequentially, each re-carving the same buffer
        self.ws_bytes = max(nbytes for _, nbytes, _ in self.layouts)

    def get_workspace_size(self) -> int:
        return self.ws_bytes

    def execute(self, graph, uid_to_data, ctx) -> None:
        from .kernels.common import build_chunk_table, ensure_cuda_context
        from .kernels.gdn_chunk_cutile import BT_CHUNK

        node_buffers = resolve_node_buffers(graph, uid_to_data)
        stream = ctx.stream if ctx.stream is not None else 0
        ensure_cuda_context(stream)
        ws = Workspace(ctx.workspace, self.ws_bytes, "GdnCuTileEngine")
        for node, _nbytes, table in self.layouts:
            nb = node_buffers[node]
            check_layouts_compact("GdnCuTileEngine", self.expects[node], nb)
            cu_seqlens = nb.inputs["cu_seqlens"]
            N = node.inputs["cu_seqlens"].dim[0] - 1
            bufs = {name: ws.view(off, dt, shape) for name, (off, dt, shape) in table.items()}
            bound = bufs["chunk_table"].shape[0]
            build_chunk_table(bufs["chunk_table"], bufs["chunk_count"], bufs["chunk_offsets"], cu_seqlens, N, BT_CHUNK, bound, stream=stream)
            if node.node_type == NodeType.GDN:
                self.execute_fwd(node, nb, bufs, stream)
            else:
                self.execute_bwd(node, nb, bufs, stream)

    def execute_fwd(self, node, nb, bufs, stream) -> None:
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
        gate_kwargs = {}
        if node.params.get("safe_gate", False):
            gate_kwargs = dict(use_gate_in_kernel=True, A_log=nb.inputs["a_log"], dt_bias=nb.inputs["dt_bias"])
        chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=nb.inputs.get("initial_state"),
            **gate_kwargs,
            output_final_state=want_state,
            cu_seqlens=nb.inputs["cu_seqlens"],
            chunk_indices=bufs["chunk_table"],
            bufs=bufs,
            stream=stream,
        )

    def execute_bwd(self, node, nb, bufs, stream) -> None:
        from .kernels.common import add_inplace, reshaped
        from .kernels.gdn_chunk_cutile import (
            RCP_LN2,
            BT_CHUNK,
            chunk_gated_delta_rule_bwd,
            chunk_gated_delta_rule_fwd_intra,
            chunk_local_cumsum,
            l2norm_bwd,
            l2norm_fwd,
        )

        H, K = node.inputs["q"].dim[1], node.inputs["q"].dim[-1]
        HV = node.inputs["v"].dim[1]
        q, k, v = nb.inputs["q"], nb.inputs["k"], nb.inputs["v"]
        g, beta, do = nb.inputs["g"], nb.inputs["beta"], nb.inputs["dO"]
        cu_seqlens = nb.inputs["cu_seqlens"]
        initial_state = nb.inputs.get("initial_state")
        dstate_in = nb.inputs.get("d_final_state")
        chunk_indices = bufs["chunk_table"]
        scale = node.params.get("scale") or K**-0.5
        l2norm = bool(node.params.get("use_qk_l2norm", False))
        if l2norm:
            q, q_rstd = l2norm_fwd(q, out=bufs["q_norm"], rstd_out=bufs["q_rstd"], stream=stream)
            k, k_rstd = l2norm_fwd(k, out=bufs["k_norm"], rstd_out=bufs["k_rstd"], stream=stream)

        # terminal pipeline buffers = the caller's output buffers
        if not l2norm:
            bufs["dq" if HV == H else "dq_hred"] = nb.outputs["dQ"]
            bufs["dk" if HV == H else "dk_hred"] = nb.outputs["dK"]
        bufs["wy_dv"] = nb.outputs["dV"]
        bufs["dg_cum"] = nb.outputs["dG"]
        bufs["db"] = nb.outputs["dBeta"]
        if initial_state is not None:
            bufs["dstate0"] = nb.outputs["d_initial_state"]

        # recompute the forward's cumulative gate and intra-chunk WY matrix
        g_cum = chunk_local_cumsum(g, chunk_size=BT_CHUNK, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, out=bufs["g_cum"], stream=stream)
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
            dstate_in=dstate_in,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            bufs=bufs,
            stream=stream,
        )
        if l2norm:
            l2norm_bwd(q, q_rstd, dq, out=nb.outputs["dQ"], bufs=bufs, stream=stream)
            l2norm_bwd(k, k_rstd, dk, dy2=dk2, out=nb.outputs["dK"], bufs=bufs, stream=stream)
        else:
            # dK/dK2 are the head-reduced finals for GVA, HV-head for MHA
            n_dk = 1
            for s in dk.shape:
                n_dk *= int(s)
            add_inplace(reshaped(dk, (n_dk,)), reshaped(dk2, (n_dk,)), n_dk, stream=stream)


class GdnCuTileEngine(BaseEngine):
    """cuTile chunked-kernel backend for single-node GDN graphs (THD layout)."""

    name = "gdn_cutile"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled + autotuned on first execute per shape

    def check_support(self, graph: "pygraph") -> None:
        import cudnn

        if buffers.current_sm() is None:
            raise NotImplementedError("GdnCuTileEngine requires a CUDA device")
        try:
            from cuda.bindings import runtime

            err, cudart_version = runtime.cudaRuntimeGetVersion()
            if int(err) != 0:
                raise NotImplementedError(f"GdnCuTileEngine: cudaRuntimeGetVersion failed ({err})")
        except ImportError as exc:
            raise NotImplementedError(f"GdnCuTileEngine requires cuda.bindings: {exc}") from exc
        if cudart_version < 13030:
            raise NotImplementedError(f"GdnCuTileEngine requires CUDA 13.3+ (found {cudart_version})")
        try:
            from .kernels.gdn_chunk_cutile import chunk_gated_delta_rule  # noqa: F401 — availability probe: ImportError = decline
        except ImportError as exc:
            raise NotImplementedError(f"GdnCuTileEngine requires the cuda.tile runtime: {exc}") from exc

        facts = graph._facts_for(analyze)
        if facts is None or facts.op != "GDN":
            raise NotImplementedError("GdnCuTileEngine supports exactly one GDN/GDN_BWD node")
        if facts.invalid:
            raise NotImplementedError(f"GdnCuTileEngine: {facts.invalid}")
        if facts.checkpoint_every_n_tokens > 0 or facts.wants_state_checkpoints:
            raise NotImplementedError("GdnCuTileEngine: per-chunk state_checkpoints output is not supported")
        if facts.is_bwd and facts.safe_gate:
            raise NotImplementedError("GdnCuTileEngine: safe_gate is forward-only")
        f32 = cudnn.data_type.FLOAT
        for port, got in (
            ("initial_state", facts.state_dtype),
            ("final_state", facts.final_state_dtype),
            ("d_final_state", facts.d_final_state_dtype),
            ("d_initial_state", facts.d_initial_state_dtype),
            ("a_log", facts.a_log_dtype),
            ("dt_bias", facts.dt_bias_dtype),
        ):
            if got not in (f32, None):
                raise NotImplementedError(f"GdnCuTileEngine: '{port}' must be fp32 (callers convert), got {got}")
        if not facts.uniform_io:
            raise NotImplementedError("GdnCuTileEngine: q/k/v dtypes must match")
        if facts.io_dtype not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, None):
            raise NotImplementedError(f"GdnCuTileEngine: q/k/v must be fp16/bf16, got {facts.io_dtype}")
        if not facts.thd_layout:
            raise NotImplementedError("GdnCuTileEngine: q/k/v must be THD [total_T, heads, dim]")
        if facts.h_k != facts.h_q:
            raise NotImplementedError(f"GdnCuTileEngine: q and k head counts differ ({facts.h_q} vs {facts.h_k})")
        if facts.h_q and facts.h_v % facts.h_q != 0:
            raise NotImplementedError(
                f"GdnCuTileEngine: v heads ({facts.h_v}) must be a multiple of q heads ({facts.h_q}; GQA-style v broadcast is FROST-only)"
            )
        if facts.d_qk > 256:
            raise NotImplementedError(f"GdnCuTileEngine: head dim K must be <= 256, got {facts.d_qk}")
        if facts.cu_dtype not in (cudnn.data_type.INT32, None):
            raise NotImplementedError("GdnCuTileEngine: cu_seqlens must be int32 (the device-side table builder reads it directly)")
        io = facts.io_dtype
        f32 = cudnn.data_type.FLOAT
        if not facts.is_bwd:
            out_dtypes = {"O": (facts.o_dtype, io), "final_state": (facts.final_state_dtype, f32)}
        else:
            out_dtypes = {
                "dQ": (facts.dq_dtype, io),
                "dK": (facts.dk_dtype, io),
                "dV": (facts.dv_dtype, io),
                "dG": (facts.dg_dtype, f32),
                "dBeta": (facts.dbeta_dtype, facts.beta_dtype),
                "d_initial_state": (facts.d_initial_state_dtype, f32),
            }
            if facts.has_initial_state != facts.wants_d_initial_state:
                raise NotImplementedError("GdnCuTileEngine: d_initial_state output must be requested iff initial_state is given")
        for port, (got, want) in out_dtypes.items():
            if got is not None and got not in (want, None):
                raise NotImplementedError(f"GdnCuTileEngine: output '{port}' must be {want} (written in place), got {got}")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return GdnCuTilePlan(graph)
