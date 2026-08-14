# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-agnostic linear-attention graph analysis: ``graph.nodes`` -> :class:`LaGraphFacts`.

One analyzer serves the three LA families (gdn / kda / gdn2 — single dedicated
nodes sharing the THD port vocabulary). :func:`analyze` is the callable each
family names in ``engines/manifest.py``; PLANNING runs it once per frozen
graph and attaches the record, so the family's engines read that same record
back instead of each parsing the node.

Also hosts the engine-side helpers shared by the LA engines: the
check_support gates over the facts record, the execute-time buffer-layout
gate, and the compiled-plan wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cudnn
from cudnn.engines.base import CompiledPlan, NodeBuffers, bind_ports
from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace

BUFFER_NAME_FROM_CUDNN = {
    cudnn.data_type.HALF: "float16",
    cudnn.data_type.BFLOAT16: "bfloat16",
    cudnn.data_type.FLOAT: "float32",
    cudnn.data_type.INT32: "int32",
    cudnn.data_type.INT64: "int64",
}


def to_buffer_dtype(dt) -> str:
    """cudnn.data_type -> the buffer-level dtype name (``buffers.DTYPES`` vocabulary)."""
    return BUFFER_NAME_FROM_CUDNN[dt]


# node type -> (op family, is_bwd)
LA_NODE_OPS = {
    cudnn.NodeType.GDN: ("GDN", False),
    cudnn.NodeType.GDN_BWD: ("GDN", True),
    cudnn.NodeType.KDA: ("KDA", False),
    cudnn.NodeType.KDA_BWD: ("KDA", True),
    cudnn.NodeType.GDN2: ("GDN2", False),
    cudnn.NodeType.GDN2_BWD: ("GDN2", True),
}


@dataclass(frozen=True)
class LaGraphFacts:
    """What a single-LA graph asks for. Pure description — no support
    judgment: head-dim limits, dtype sets, and feature coverage are per-engine
    knowledge, matched in each engine's ``check_support``. ``invalid`` is the
    one exception: a graph-consistency error (malformed regardless of which
    kernel would run — a missing required port, ``d_initial_state`` without
    ``initial_state``, safe-gate inputs without the attribute); when set,
    every engine is ineligible."""

    invalid: Optional[str] = None

    op: str = ""  # "GDN" | "KDA" | "GDN2"
    is_bwd: bool = False

    # geometry (THD; zeros when the port ranks are not declared)
    thd_layout: bool = True  # Q/K/V are rank-3 [total_T, heads, dim]
    h_q: int = 0
    h_k: int = 0
    h_v: int = 0
    h_o: int = 0
    d_qk: int = 0
    d_v: int = 0
    gates_at_ho: bool = True  # Gate/Beta(/W) carry HO = max(h_q, h_v) heads

    # dtypes (cudnn.data_type vocabulary; None = unset/inferred)
    io_dtype: Any = None  # Q's dtype
    uniform_io: bool = True  # Q/K/V dtypes agree
    g_dtype: Any = None
    beta_dtype: Any = None
    w_dtype: Any = None
    cu_dtype: Any = None
    a_log_dtype: Any = None
    dt_bias_dtype: Any = None
    do_dtype: Any = None
    state_checkpoints_dtype: Any = None
    state_checkpoints_out_dtype: Any = None  # fwd checkpoint OUTPUT port dtype
    d_final_state_dtype: Any = None
    state_dtype: Any = None
    final_state_dtype: Any = None
    state_pair_match: bool = True
    o_dtype: Any = None
    dq_dtype: Any = None
    dk_dtype: Any = None
    dv_dtype: Any = None
    dg_dtype: Any = None
    dbeta_dtype: Any = None
    dw_dtype: Any = None
    d_initial_state_dtype: Any = None

    # ports present / requested
    has_initial_state: bool = False
    wants_d_initial_state: bool = False
    wants_state_checkpoints: bool = False  # fwd checkpoint-series output

    # attributes
    scale: Optional[float] = None
    use_qk_l2norm: bool = False
    safe_gate: bool = False
    use_beta_sigmoid: bool = False
    checkpoint_every_n_tokens: int = 0
    batch_invariant: bool = False


def analyze(graph: "cudnn.pygraph") -> Optional[LaGraphFacts]:
    """Facts for a single-LA graph, or None if the graph is anything else.

    Pure: attaching and caching is the graph's job (create_execution_plans
    -> _attach_facts). This is the callable the LA families name in their
    manifest ``analyzer`` entries."""
    nodes = list(graph.nodes)
    if len(nodes) != 1:
        return None
    node = nodes[0]
    kind = LA_NODE_OPS.get(node.node_type)
    if kind is None:
        return None
    op, is_bwd = kind
    ins, outs, params = node.inputs, node.outputs, node.params

    required_in = ["q", "k", "v", "g", "beta", "cu_seqlens"]
    if op == "GDN2":
        required_in.append("w")
    if is_bwd:
        required_in.append("dO")
    required_out = (["dQ", "dK", "dV", "dG", "dBeta"] + (["dW"] if op == "GDN2" else [])) if is_bwd else ["O"]

    safe_gate = bool(params.get("safe_gate", False))
    ckpt = int(params.get("checkpoint_every_n_tokens", 0) or 0)
    invalid = None
    missing_in = [p for p in required_in if p not in ins]
    missing_out = [p for p in required_out if p not in outs]
    if missing_in:
        invalid = f"{node.node_type.name} node '{node.name}' is missing input(s) {missing_in}"
    elif missing_out:
        invalid = f"{node.node_type.name} node '{node.name}' is missing output(s) {missing_out}"
    elif "d_initial_state" in outs and "initial_state" not in ins:
        invalid = "d_initial_state requires initial_state"
    elif safe_gate and ("a_log" not in ins or "dt_bias" not in ins):
        invalid = "safe_gate requires a_log and dt_bias inputs"
    elif not safe_gate and ("a_log" in ins or "dt_bias" in ins):
        invalid = "a_log/dt_bias require safe_gate=True"
    elif params.get("gate_lower_bound") is not None and not safe_gate:
        invalid = "gate_lower_bound requires safe_gate=True"
    elif ckpt < 0:
        invalid = "checkpoint_every_n_tokens must be non-negative"
    elif not is_bwd and ckpt > 0 and "state_checkpoints" not in outs:
        invalid = "checkpoint_every_n_tokens > 0 requires the state_checkpoints output"
    elif not is_bwd and ckpt == 0 and "state_checkpoints" in outs:
        invalid = "state_checkpoints output requires checkpoint_every_n_tokens > 0"
    if invalid is not None:
        return LaGraphFacts(invalid=invalid, op=op, is_bwd=is_bwd)

    in_dt = {name: t.get_data_type() for name, t in ins.items()}
    out_dt = {name: t.get_data_type() for name, t in outs.items()}
    q, k, v = ins["q"], ins["k"], ins["v"]

    thd_layout = all(t.dim and len(t.dim) == 3 for t in (q, k, v))
    if thd_layout:
        _, h_q, d_qk = (int(d) for d in q.dim)
        h_k, h_v, d_v = int(k.dim[1]), int(v.dim[1]), int(v.dim[2])
    else:
        h_q = h_k = h_v = d_qk = d_v = 0
    h_o = max(h_q, h_v)
    gates_at_ho = all(t is None or not t.dim or (len(t.dim) > 1 and int(t.dim[1]) == h_o) for t in (ins["g"], ins["beta"], ins.get("w")))
    io_dtypes = {in_dt["q"], in_dt["k"], in_dt["v"]} - {None}
    state_dtypes = {in_dt.get("initial_state"), out_dt.get("final_state")} - {None}
    scale = params.get("scale")

    return LaGraphFacts(
        op=op,
        is_bwd=is_bwd,
        thd_layout=thd_layout,
        h_q=h_q,
        h_k=h_k,
        h_v=h_v,
        h_o=h_o,
        d_qk=d_qk,
        d_v=d_v,
        gates_at_ho=gates_at_ho,
        io_dtype=in_dt["q"],
        uniform_io=len(io_dtypes) <= 1,
        g_dtype=in_dt["g"],
        beta_dtype=in_dt["beta"],
        w_dtype=in_dt.get("w"),
        cu_dtype=in_dt["cu_seqlens"],
        a_log_dtype=in_dt.get("a_log"),
        dt_bias_dtype=in_dt.get("dt_bias"),
        do_dtype=in_dt.get("dO"),
        state_checkpoints_dtype=in_dt.get("state_checkpoints"),
        state_checkpoints_out_dtype=out_dt.get("state_checkpoints"),
        d_final_state_dtype=in_dt.get("d_final_state"),
        state_dtype=in_dt.get("initial_state"),
        final_state_dtype=out_dt.get("final_state"),
        state_pair_match=len(state_dtypes) <= 1,
        o_dtype=out_dt.get("O"),
        dq_dtype=out_dt.get("dQ"),
        dk_dtype=out_dt.get("dK"),
        dv_dtype=out_dt.get("dV"),
        dg_dtype=out_dt.get("dG"),
        dbeta_dtype=out_dt.get("dBeta"),
        dw_dtype=out_dt.get("dW"),
        d_initial_state_dtype=out_dt.get("d_initial_state"),
        has_initial_state="initial_state" in ins,
        wants_d_initial_state="d_initial_state" in outs,
        wants_state_checkpoints="state_checkpoints" in outs,
        scale=float(scale) if scale is not None else None,
        use_qk_l2norm=bool(params.get("use_qk_l2norm", False)),
        safe_gate=safe_gate,
        use_beta_sigmoid=bool(params.get("use_beta_sigmoid", False)),
        checkpoint_every_n_tokens=ckpt,
        batch_invariant=bool(params.get("batch_invariant", False)),
    )


# ---------------------------------------------------------------------------
# Engine-side helpers shared by the LA engines
# ---------------------------------------------------------------------------


def require(engine: str, port: str, got, want) -> None:
    """check_support dtype gate over a facts field: unset passes (the kernel
    validates the buffer), anything else must be the kernel-native dtype."""
    if got is None:
        return
    wanted = want if isinstance(want, tuple) else (want,)
    if got not in wanted:
        names = "/".join(w.name for w in wanted)
        raise NotImplementedError(f"{engine}: '{port}' must be {names} (the kernel-native dtype; no staging), got {got}")


def frost_la_gate(engine: str, facts, op: str) -> None:
    """The FROST LA engines' shared check_support core: the analyzer record,
    the device/DSL environment, and the gates common to all three kernels."""
    if facts is None or facts.op != op:
        raise NotImplementedError(f"{engine} supports exactly one {op}/{op}_BWD node")
    if facts.invalid:
        raise NotImplementedError(f"{engine}: {facts.invalid}")
    sm = buffers.current_sm()
    if sm is None or not (100 <= sm <= 103):
        raise NotImplementedError(f"{engine} requires SM100-SM103 (found {sm})")
    installed, version = buffers.cutedsl_state()
    if not installed:
        raise NotImplementedError(f"{engine} requires the cutedsl extra (nvidia-cutlass-dsl), which is not installed")
    if buffers.cutedsl_too_old(version):
        want = ".".join(str(v) for v in buffers.CUTEDSL_MIN_VERSION)
        raise NotImplementedError(f"{engine} requires nvidia-cutlass-dsl >= {want}; found {version[1]}")
    if not facts.uniform_io:
        raise NotImplementedError(f"{engine}: q/k/v dtypes must match")
    require(engine, "q/k/v", facts.io_dtype, (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF))
    if not facts.thd_layout:
        raise NotImplementedError(f"{engine}: q/k/v must be THD [total_T, heads, dim]")
    if facts.d_qk != 128 or facts.d_v != 128:
        raise NotImplementedError(f"{engine}: head dims must be 128 (the recurrent state is 128x128), got K={facts.d_qk} V={facts.d_v}")
    if facts.h_k not in (facts.h_q, facts.h_v):
        raise NotImplementedError(f"{engine}: k heads ({facts.h_k}) must match q's ({facts.h_q}) or v's ({facts.h_v}; canonical GQA shares grouped k/v heads)")
    if facts.h_v != facts.h_q and max(facts.h_q, facts.h_v) % min(facts.h_q, facts.h_v) != 0:
        raise NotImplementedError(f"{engine}: q heads ({facts.h_q}) and v heads ({facts.h_v}) must be equal or one a multiple of the other")
    require(engine, "g", facts.g_dtype, cudnn.data_type.FLOAT)
    require(engine, "cu_seqlens", facts.cu_dtype, (cudnn.data_type.INT32, cudnn.data_type.INT64))


class FrostLaPlan(CompiledPlan):
    """A compiled LA executor, driven from the normalized variant pack: the
    port-to-slot join is a property of the graph, so it happens once and is
    kept; between executes only the buffer addresses move."""

    takes_variant_pack = True

    def __init__(self, compiled):
        self.compiled = compiled
        self.ports = None

    def get_workspace_size(self) -> int:
        return self.compiled.workspace_bytes()

    def execute(self, graph, variant_pack, ctx) -> None:
        ports = self.ports
        if ports is None:
            ports = self.ports = bind_ports(graph, variant_pack)
        ok, offender = variant_pack.all_dense_layout()
        if not ok:
            raise ValueError(dense_layout_message(self.compiled.plan_name, ports, offender))
        node_buffers = {}
        for node, slots in ports.items():
            names = list(slots.inputs) + list(slots.outputs)
            views = variant_pack.operands(list(slots.inputs.values()) + list(slots.outputs.values()))
            split = len(slots.inputs)
            node_buffers[node] = NodeBuffers(dict(zip(names[:split], views[:split])), dict(zip(names[split:], views[split:])))
        workspace = Workspace.over(variant_pack, self.compiled.workspace_bytes(), type(self.compiled).__name__)
        self.compiled(node_buffers, workspace=workspace, stream=ctx.stream)


def expect_table(node, align) -> dict:
    """Build-time ``{port: (dims, dtype_name, align_bytes)}`` for
    :func:`check_layouts`: bound buffers must match the node's frozen
    geometry exactly (one graph per shape), and base pointers must satisfy
    the kernel entry's ``assumed_align`` claim. ``align`` maps port name ->
    bytes (family table read from the entry's from_dlpack calls; absent
    ports default to 16)."""
    table = {}
    for ports in (node.inputs, node.outputs):
        for name, t in ports.items():
            if t is None:
                continue
            dims = tuple(int(d) for d in t.dim) if t.dim else None
            table[name] = (dims, BUFFER_NAME_FROM_CUDNN.get(t.get_data_type()), align.get(name, 16))
    return table


def check_layouts_compact(plan_name: str, expect, nb) -> None:
    """Execute-time gate for the cuTile backend: every bound buffer must be
    CONTIGUOUS (the kernels stage rank-merged views and whole-buffer zero
    fills), and must match the node's build-time dims/dtype and base
    alignment per ``expect`` (see :func:`expect_table`)."""
    for ports in (nb.inputs, nb.outputs):
        for name, b in ports.items():
            if b is None:
                continue
            ptr, shape, strides, dtype, _dev = buffers.probe(b)
            exp = expect.get(name) if expect else None
            if exp is not None:
                dims, dtype_name, align = exp
                if dims is not None and tuple(shape) != dims:
                    raise ValueError(f"{plan_name}: buffer for {name!r} must match the graph's build-time dims {dims}; got {tuple(shape)}")
                if dtype_name is not None and dtype != dtype_name:
                    raise ValueError(f"{plan_name}: buffer for {name!r} must be {dtype_name} (the node's declared dtype); got {dtype}")
                if align and ptr % align != 0:
                    raise ValueError(f"{plan_name}: buffer for {name!r} base pointer must be {align}-byte aligned; got 0x{ptr:x}")
            if not buffers.is_contiguous(shape, strides):
                raise ValueError(
                    f"{plan_name}: buffer for {name!r} must be contiguous (the cuTile backend stages rank-merged views); got shape {shape} strides {strides}"
                )


def dense_layout_message(plan_name, ports, offender) -> str:
    """Name the port behind ``all_dense_layout``'s failing slot. Buffers pass
    straight to the stride-plumbed kernels, so the one execute-time rule is a
    stride-1 innermost dim; this walk only runs on the way to raising."""
    for slots in ports.values():
        for direction in (slots.inputs, slots.outputs):
            for port, slot in direction.items():
                if slot == offender:
                    return f"{plan_name}: buffer for {port!r} must have a stride-1 innermost dim (buffers pass straight to the kernel)"
    return f"{plan_name}: the buffer at variant-pack slot {offender} must have a stride-1 innermost dim"
