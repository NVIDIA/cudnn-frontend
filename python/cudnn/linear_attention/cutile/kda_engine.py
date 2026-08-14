# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuTile KDA engine: KDA / KDA_BWD nodes on the chunked cuTile kernels
(``kernels/kda_chunk_cutile``)."""

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
    """Static carve plan for one node's ``chunk_kda`` pipeline intermediates:
    name -> (offset, dtype-name, shape). The chunk count is data-dependent
    (varlen), so chunk-indexed entries are sized and SHAPED at the bound
    ``cdiv(total, 64) + N`` — the device-built table's sentinel tail keeps
    bound-gridded launches inert past the real count. A KDA_BWD node carries
    the union of the forward re-run's intermediates and the backward
    temporaries, so both live disjointly in one carve.  Terminal pipeline
    buffers (the forward's ``o``/``final_state``; the backward's boundary casts /
    l2norm outputs, ``dv2`` and ``dstate0``) are NOT carved — execute plants the
    caller's output buffers under those names."""
    from .kernels.kda_chunk_cutile import BT_CHUNK, cdiv, next_power_of_2

    q, v, g, cu = (node.inputs[p] for p in ("q", "v", "g", "cu_seqlens"))
    total, H, K = q.dim
    HV, V = v.dim[1], v.dim[2]
    N = cu.dim[0] - 1
    io = to_buffer_dtype(q.get_data_type())
    f32 = "float32"
    BC = 32 if K >= 64 else 16  # fwd_intra sub-chunk (see chunk_kda_fwd_intra)
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

    # chunk_kda forward (also re-run inside KDA_BWD)
    carve("g_cum", f32, (total, HV, K))
    carve("Aqk", io, (total, HV, BT_CHUNK))
    carve("Akk", io, (total, HV, BT_CHUNK))
    carve("Akkd", f32, (total, HV, BC))
    carve("w", io, (total, HV, K))
    carve("u", io, (total, HV, V))
    carve("qg", io, (total, HV, K))
    carve("kg", io, (total, HV, K))
    carve("state_checkpoints", io, (NT_bound, HV, K, V))
    carve("v_new", io, (total, HV, V))
    if node.params.get("use_beta_sigmoid", False):
        carve("beta_sig", f32, (total, HV))
    if node.node_type == NodeType.KDA_BWD:
        carve("o", io, (total, HV, V))  # discarded output of the forward re-run
    if l2norm:
        carve("q_norm", io, (total * H, K))
        carve("q_rstd", f32, (total * H,))
        carve("k_norm", io, (total * H, K))
        carve("k_rstd", f32, (total * H,))
    if node.node_type == NodeType.KDA_BWD:
        carve("dAqk", f32, (total, HV, BT_CHUNK))
        carve("dv_dAv", io, (total, HV, V))
        carve("dstate", io, (NT_bound, HV, K, V))
        carve("dv_dstate_u", io, (total, HV, V))
        carve("dq", f32, (total, HV, K))
        carve("dk", f32, (total, HV, K))
        carve("dg", f32, (total, HV, K))
        if to_buffer_dtype(node.inputs["beta"].get_data_type()) != f32:
            carve("db", f32, (total, HV))
        carve("dAkk", f32, (total, HV, BT_CHUNK))
        carve("dq2", f32, (total, HV, K))
        carve("dk2", f32, (total, HV, K))
        if HV != H:
            carve("dq_hred", f32, (total, H, K))
            carve("dk_hred", f32, (total, H, K))
        NK = cdiv(K, min(64, next_power_of_2(K)))  # bwd_intra K-split (see chunk_kda_bwd_intra)
        carve("db2", f32, (NK, total, HV))
        carve("dg2", f32, (total, HV, K))
        if to_buffer_dtype(g.get_data_type()) != f32:
            carve("dg_cum", f32, (total, HV, K))
    return size, table


class KdaCuTilePlan(CompiledPlan):
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
        from .kernels.kda_chunk_cutile import BT_CHUNK

        node_buffers = resolve_node_buffers(graph, uid_to_data)
        stream = ctx.stream if ctx.stream is not None else 0
        ensure_cuda_context(stream)
        ws = Workspace(ctx.workspace, self.ws_bytes, "KdaCuTileEngine")
        for node, _nbytes, table in self.layouts:
            nb = node_buffers[node]
            check_layouts_compact("KdaCuTileEngine", self.expects[node], nb)
            cu_seqlens = nb.inputs["cu_seqlens"]
            N = node.inputs["cu_seqlens"].dim[0] - 1
            bufs = {name: ws.view(off, dt, shape) for name, (off, dt, shape) in table.items()}
            bound = bufs["chunk_table"].shape[0]
            build_chunk_table(bufs["chunk_table"], bufs["chunk_count"], bufs["chunk_offsets"], cu_seqlens, N, BT_CHUNK, bound, stream=stream)
            if node.node_type == NodeType.KDA:
                self.execute_fwd(node, nb, bufs, stream)
            else:
                self.execute_bwd(node, nb, bufs, stream)

    def execute_fwd(self, node, nb, bufs, stream) -> None:
        from .kernels.kda_chunk_cutile import chunk_kda

        want_state = "final_state" in node.outputs
        q, k, v = nb.inputs["q"], nb.inputs["k"], nb.inputs["v"]
        g, beta = nb.inputs["g"], nb.inputs["beta"]
        # terminal pipeline buffers = the caller's output buffers
        bufs["o"] = nb.outputs["O"]
        if want_state:
            bufs["final_state"] = nb.outputs["final_state"]
        raw_gate_kwargs = {}
        if node.params.get("use_beta_sigmoid", False):
            raw_gate_kwargs["use_beta_sigmoid_in_kernel"] = True
        if node.params.get("safe_gate", False):
            raw_gate_kwargs.update(
                safe_gate=True,
                use_gate_in_kernel=True,
                lower_bound=float(node.params.get("gate_lower_bound") or -5.0),
                A_log=nb.inputs["a_log"],
                dt_bias=nb.inputs["dt_bias"],
            )
        chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            scale=node.params.get("scale"),
            initial_state=nb.inputs.get("initial_state"),
            output_final_state=want_state,
            use_qk_l2norm_in_kernel=bool(node.params.get("use_qk_l2norm", False)),
            cu_seqlens=nb.inputs["cu_seqlens"],
            chunk_indices=bufs["chunk_table"],
            bufs=bufs,
            stream=stream,
            **raw_gate_kwargs,
        )

    def execute_bwd(self, node, nb, bufs, stream) -> None:
        from .kernels.common import reshaped
        from .kernels.kda_chunk_cutile import chunk_kda_grad

        total, H, K = node.inputs["q"].dim
        cu_seqlens = nb.inputs["cu_seqlens"]
        initial_state = nb.inputs.get("initial_state")
        dstate_in = nb.inputs.get("d_final_state")
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
            bufs["dstate0"] = nb.outputs["d_initial_state"]

        chunk_kda_grad(
            q,
            k,
            v,
            g,
            beta,
            do,
            dstate_in=dstate_in,
            scale=scale,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=bool(node.params.get("use_qk_l2norm", False)),
            cu_seqlens=cu_seqlens,
            chunk_indices=bufs["chunk_table"],
            bufs=bufs,
            stream=stream,
        )


class KdaCuTileEngine(BaseEngine):
    """cuTile chunked-kernel backend for single-node KDA graphs (THD layout)."""

    name = "kda_cutile"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled + autotuned on first execute per shape

    def check_support(self, graph: "pygraph") -> None:
        import cudnn

        if buffers.current_sm() is None:
            raise NotImplementedError("KdaCuTileEngine requires a CUDA device")
        try:
            from cuda.bindings import runtime

            err, cudart_version = runtime.cudaRuntimeGetVersion()
            if int(err) != 0:
                raise NotImplementedError(f"KdaCuTileEngine: cudaRuntimeGetVersion failed ({err})")
        except ImportError as exc:
            raise NotImplementedError(f"KdaCuTileEngine requires cuda.bindings: {exc}") from exc
        if cudart_version < 13030:
            raise NotImplementedError(f"KdaCuTileEngine requires CUDA 13.3+ (found {cudart_version})")
        try:
            from .kernels.kda_chunk_cutile import chunk_kda  # noqa: F401 — availability probe: ImportError = decline
        except ImportError as exc:
            raise NotImplementedError(f"KdaCuTileEngine requires the cuda.tile runtime: {exc}") from exc

        facts = graph._facts_for(analyze)
        if facts is None or facts.op != "KDA":
            raise NotImplementedError("KdaCuTileEngine supports exactly one KDA/KDA_BWD node")
        if facts.invalid:
            raise NotImplementedError(f"KdaCuTileEngine: {facts.invalid}")
        if facts.checkpoint_every_n_tokens > 0 or facts.wants_state_checkpoints:
            raise NotImplementedError("KdaCuTileEngine: per-chunk state_checkpoints output is not supported")
        if facts.is_bwd and (facts.safe_gate or facts.use_beta_sigmoid):
            raise NotImplementedError("KdaCuTileEngine: raw-logit gate modes (safe_gate / use_beta_sigmoid) are forward-only")
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
                raise NotImplementedError(f"KdaCuTileEngine: '{port}' must be fp32 (callers convert), got {got}")
        node = next(iter(graph.nodes), None)
        glb = node.params.get("gate_lower_bound") if node is not None else None
        if glb is not None and glb is not False and not (-5.0 <= float(glb) < 0):
            raise NotImplementedError(f"KdaCuTileEngine: gate_lower_bound must be in [-5, 0) (chunk_kda log-gate floor), got {glb}")
        if not facts.uniform_io:
            raise NotImplementedError("KdaCuTileEngine: q/k/v dtypes must match")
        if facts.io_dtype not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, None):
            raise NotImplementedError(f"KdaCuTileEngine: q/k/v must be fp16/bf16, got {facts.io_dtype}")
        if not facts.thd_layout:
            raise NotImplementedError("KdaCuTileEngine: q/k/v must be THD [total_T, heads, dim]")
        if facts.h_k != facts.h_q:
            raise NotImplementedError(f"KdaCuTileEngine: q and k head counts differ ({facts.h_q} vs {facts.h_k})")
        if facts.h_q and facts.h_v % facts.h_q != 0:
            raise NotImplementedError(
                f"KdaCuTileEngine: v heads ({facts.h_v}) must be a multiple of q heads ({facts.h_q}; GQA-style v broadcast is FROST-only)"
            )
        if facts.d_qk > 256:
            raise NotImplementedError(f"KdaCuTileEngine: head dim K must be <= 256, got {facts.d_qk}")
        if facts.cu_dtype not in (cudnn.data_type.INT32, None):
            raise NotImplementedError("KdaCuTileEngine: cu_seqlens must be int32 (the device-side table builder reads it directly)")
        io = facts.io_dtype
        f32 = cudnn.data_type.FLOAT
        if not facts.is_bwd:
            out_dtypes = {"O": (facts.o_dtype, io), "final_state": (facts.final_state_dtype, f32)}
        else:
            out_dtypes = {
                "dQ": (facts.dq_dtype, io),
                "dK": (facts.dk_dtype, io),
                "dV": (facts.dv_dtype, io),
                "dG": (facts.dg_dtype, facts.g_dtype),
                "dBeta": (facts.dbeta_dtype, facts.beta_dtype),
                "d_initial_state": (facts.d_initial_state_dtype, f32),
            }
            if facts.has_initial_state != facts.wants_d_initial_state:
                raise NotImplementedError("KdaCuTileEngine: d_initial_state output must be requested iff initial_state is given")
        for port, (got, want) in out_dtypes.items():
            if got is not None and got not in (want, None):
                raise NotImplementedError(f"KdaCuTileEngine: output '{port}' must be {want} (written in place), got {got}")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return KdaCuTilePlan(graph)
