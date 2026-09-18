# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata for the dense Mamba-2 SSD graph family (no framework imports)."""

from dataclasses import dataclass, field
from typing import Any

import cudnn


@dataclass(frozen=True)
class Mamba2GraphFacts:
    invalid: str | None = None
    is_bwd: bool = False
    batch: int = 0
    length: int = 0
    heads: int = 0
    groups: int = 0
    head_dim: int = 0
    state_dim: int = 0
    chunk_size: int = 32
    dt_softplus: bool = True
    intermediate_dtype: str = "float32"
    tensors: dict[str, Any] = field(default_factory=dict)


def analyze(graph):
    """Describe exactly one MAMBA2 or MAMBA2_BWD node."""
    nodes = list(graph.nodes)
    if len(nodes) != 1 or nodes[0].node_type not in (cudnn.NodeType.MAMBA2, cudnn.NodeType.MAMBA2_BWD):
        return None
    node = nodes[0]
    inputs, outputs = node.inputs, node.outputs
    is_bwd = node.node_type == cudnn.NodeType.MAMBA2_BWD
    required = {"x", "dt", "A", "B", "C"} | ({"dO"} if is_bwd else set())
    missing = required - inputs.keys()
    if missing:
        return Mamba2GraphFacts(invalid=f"missing inputs {sorted(missing)}", is_bwd=is_bwd)
    x, b = inputs["x"], inputs["B"]
    if len(x.dim) != 4 or len(b.dim) != 4:
        return Mamba2GraphFacts(invalid="x and B must be rank four", is_bwd=is_bwd)
    batch, length, heads, head_dim = x.dim
    groups, state_dim = b.dim[2:]
    geometry = dict(batch=batch, length=length, heads=heads, groups=groups, head_dim=head_dim, state_dim=state_dim)
    if min(geometry.values()) <= 0 or heads % groups:
        return Mamba2GraphFacts(invalid="dimensions must be positive and heads divisible by groups", is_bwd=is_bwd)
    if is_bwd and ("z" in inputs) != ("ungated_out" in inputs):
        return Mamba2GraphFacts(invalid="z and ungated_out must be provided together for backward", is_bwd=True)
    state = [batch, heads, head_dim, state_dim]
    shapes = {
        "x": list(x.dim),
        "dt": [batch, length, heads],
        "A": [heads],
        "B": [batch, length, groups, state_dim],
        "C": [batch, length, groups, state_dim],
        "D": [heads],
        "dt_bias": [heads],
        "z": list(x.dim),
        "initial_state": state,
        "dO": list(x.dim),
        "d_final_state": state,
        "ungated_out": list(x.dim),
    }
    chunk = node.params.get("chunk_size", 32)
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        return Mamba2GraphFacts(invalid="chunk_size must be a positive integer", is_bwd=is_bwd)
    shapes["state_checkpoints"] = [batch, heads, (length + chunk - 1) // chunk, head_dim, state_dim]
    out_sources = {"dX": "x", "dDt": "dt", "dA": "A", "dB": "B", "dC": "C", "dD": "D", "d_dt_bias": "dt_bias", "dZ": "z", "d_initial_state": "initial_state"}
    required_out = {dst for dst, src in out_sources.items() if src in inputs} if is_bwd else {"O"}
    if not is_bwd:
        if "z" in inputs:
            required_out.add("ungated_out")
        if node.params.get("output_final_state", False):
            required_out.add("final_state")
        if node.params.get("save_state_checkpoints", False):
            required_out.add("state_checkpoints")
    if set(outputs) != required_out:
        return Mamba2GraphFacts(invalid=f"outputs must be {sorted(required_out)}, got {sorted(outputs)}", is_bwd=is_bwd)
    shapes.update({dst: shapes[src] for dst, src in out_sources.items()})
    shapes.update(O=list(x.dim), final_state=state)
    for name, tensor in (*inputs.items(), *outputs.items()):
        if name not in shapes or list(tensor.dim) != shapes[name]:
            return Mamba2GraphFacts(invalid=f"{name}: expected shape {shapes.get(name)}, got {tensor.dim}", is_bwd=is_bwd)
    return Mamba2GraphFacts(
        is_bwd=is_bwd,
        **geometry,
        chunk_size=chunk,
        dt_softplus=node.params.get("dt_softplus", True),
        intermediate_dtype=node.params.get("intermediate_dtype", "float32"),
        tensors={**inputs, **outputs},
    )
