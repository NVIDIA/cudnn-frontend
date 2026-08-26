# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-agnostic linear-attention graph analysis: ``graph.nodes`` -> :class:`LaGraphFacts`.

One analyzer serves the three LA families (gdn / kda / gdn2 — single dedicated
nodes sharing the THD port vocabulary). :func:`analyze` is the callable each
family names in ``engines/manifest.py``; PLANNING runs it once per frozen
graph and attaches the record, so the family's engines read that same record
back instead of each parsing the node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cudnn

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
    total_t: int = 0  # packed token count
    n_seq: int = 0
    state_checkpoint_rows: int = 0  # rows declared on the checkpoint port (0 = absent or undeclared)
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
    d_a_log_dtype: Any = None
    d_dt_bias_dtype: Any = None

    # ports present / requested
    has_initial_state: bool = False
    wants_d_initial_state: bool = False
    wants_state_checkpoints: bool = False  # fwd checkpoint-series output

    # attributes
    scale: Optional[float] = None
    use_qk_l2norm: bool = False
    safe_gate: bool = False
    use_beta_sigmoid: bool = False
    beta_guard: bool = False
    gate_lower_bound: Optional[float] = None
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
    checkpoint = int(params.get("checkpoint_every_n_tokens", 0) or 0)
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
    elif ("d_a_log" in outs or "d_dt_bias" in outs) and not (is_bwd and safe_gate):
        invalid = "d_a_log/d_dt_bias require safe_gate=True on a bwd node"
    elif is_bwd and safe_gate and ("d_a_log" not in outs or "d_dt_bias" not in outs):
        invalid = "safe_gate on a bwd node requires the d_a_log and d_dt_bias outputs"
    elif is_bwd and safe_gate and any(list(outs[d].dim or []) != list(ins[p].dim or []) for d, p in (("d_a_log", "a_log"), ("d_dt_bias", "dt_bias"))):
        invalid = "d_a_log/d_dt_bias dims must match a_log/dt_bias"
    elif params.get("gate_lower_bound") is not None and not safe_gate:
        invalid = "gate_lower_bound requires safe_gate=True"
    elif checkpoint < 0:
        invalid = "checkpoint_every_n_tokens must be non-negative"
    elif not is_bwd and checkpoint > 0 and "state_checkpoints" not in outs:
        invalid = "checkpoint_every_n_tokens > 0 requires the state_checkpoints output"
    elif not is_bwd and checkpoint == 0 and "state_checkpoints" in outs:
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
    total_t = int(q.dim[0]) if thd_layout else 0
    cu = ins["cu_seqlens"]
    n_seq = int(cu.dim[0]) - 1 if cu.dim else 0
    checkpoint_port = ins.get("state_checkpoints")
    if checkpoint_port is None:
        checkpoint_port = outs.get("state_checkpoints")
    state_checkpoint_rows = int(checkpoint_port.dim[0]) if checkpoint_port is not None and checkpoint_port.dim else 0
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
        total_t=total_t,
        n_seq=n_seq,
        state_checkpoint_rows=state_checkpoint_rows,
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
        d_a_log_dtype=out_dt.get("d_a_log"),
        d_dt_bias_dtype=out_dt.get("d_dt_bias"),
        has_initial_state="initial_state" in ins,
        wants_d_initial_state="d_initial_state" in outs,
        wants_state_checkpoints="state_checkpoints" in outs,
        scale=float(scale) if scale is not None else None,
        use_qk_l2norm=bool(params.get("use_qk_l2norm", False)),
        safe_gate=safe_gate,
        use_beta_sigmoid=bool(params.get("use_beta_sigmoid", False)),
        beta_guard=bool(params.get("beta_guard", False)),
        gate_lower_bound=float(params["gate_lower_bound"]) if params.get("gate_lower_bound") is not None else None,
        checkpoint_every_n_tokens=checkpoint,
        batch_invariant=bool(params.get("batch_invariant", False)),
    )
