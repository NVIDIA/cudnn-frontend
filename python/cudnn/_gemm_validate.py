# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-native semantic validation for the GEMM node family (issue #704).

The frost_gemm family's ``EngineFamily.validator``: when a python engine is a
candidate for a GEMM graph, ``pygraph.validate()`` runs these rules instead of
eagerly lowering to C++ and asking the backend -- which couples a graph the
FROST GEMM engine fully serves to the installed backend's version (the MoE
grouped-matmul node needs cuDNN >= 9.15 forward / 9.22 backward *to validate*,
attributes the backend would never execute for a FROST-served graph). The
backend's own verdict is deferred to planning, exactly as for SDPA.

Only the version- and arch-agnostic subset of the classic checks lives here:
the C++ matmul node has no semantic pre-validation of its own (the backend
judges shapes at plan time), so this module carries the structural facts a
matmul must satisfy to mean anything -- and the MoE grouped-matmul node's
ATTRIBUTE_NOT_SET checks with their classic messages. Nothing about device
arch or backend version: those are support-surface answers each engine gives
at planning.

Error types match the classic surface: ``ValueError`` for ATTRIBUTE_NOT_SET /
INVALID_VALUE parity, ``cudnn.cudnnGraphNotSupportedError`` for a graph the
backend would decline as GRAPH_NOT_SUPPORTED.
"""

from __future__ import annotations

from typing import Any

from ._ir import NodeType

_MOE_FWD_INPUTS = ("token", "weight", "first_token_offset")
_MOE_BWD_INPUTS = ("doutput", "token", "first_token_offset")

COVERED_NODE_TYPES = frozenset({NodeType.MATMUL, NodeType.MOE_GROUPED_MATMUL, NodeType.MOE_GROUPED_MATMUL_BWD})


def _not_supported(message: str) -> Exception:
    """Classic GRAPH_NOT_SUPPORTED parity: the error type callers catch to skip a config."""
    import cudnn

    return cudnn.cudnnGraphNotSupportedError(message)


def _dims(t: Any):
    d = t.get_dim() if t is not None else None
    return list(d) if d else None


def validate_graph(graph) -> bool:
    """Validate every node when all of them are GEMM-family (MATMUL / MoE grouped
    matmul fwd+bwd); return False without raising when the graph holds a node
    this module does not cover (a pointwise epilogue, a reshape, ...), so the
    caller falls back to the classic eager C++ lowering."""
    nodes = list(graph._nodes)
    if not nodes or any(n.node_type not in COVERED_NODE_TYPES for n in nodes):
        return False
    for node in nodes:
        validate_node(node)
    return True


def validate_node(node) -> None:
    """Dispatch one GEMM-family node to its rules."""
    if node.node_type == NodeType.MATMUL:
        _validate_matmul(node)
    elif node.node_type == NodeType.MOE_GROUPED_MATMUL:
        _validate_required(node, "MoeGroupedMatmul", _MOE_FWD_INPUTS, ("OUT_0",))
    elif node.node_type == NodeType.MOE_GROUPED_MATMUL_BWD:
        _validate_required(node, "MoeGroupedMatmulBwd", _MOE_BWD_INPUTS, ("dweight",))


def _validate_matmul(node) -> None:
    """C = A @ B with batch broadcasting: both operands bound, same rank >= 2,
    the contraction extent agrees, batch extents equal or 1, and a user-declared
    C carries the (M, N) the operands imply."""
    a, b, c = node.inputs.get("A"), node.inputs.get("B"), node.outputs.get("C")
    if a is None or b is None:
        raise ValueError("Matmul inputs A and B must both be set.")
    if c is None:
        raise ValueError("Matmul output C not set.")
    ad, bd = _dims(a), _dims(b)
    if not ad or not bd:
        raise ValueError("Matmul inputs A and B must have their dims set.")
    if len(ad) != len(bd) or len(ad) < 2:
        raise _not_supported(f"Matmul requires A and B of equal rank >= 2; got {ad} and {bd}")
    if ad[-1] != bd[-2]:
        raise _not_supported(f"Matmul contraction mismatch: A's last dim {ad[-1]} != B's second-to-last dim {bd[-2]}")
    for i, (x, y) in enumerate(zip(ad[:-2], bd[:-2])):
        if x != y and x != 1 and y != 1:
            raise _not_supported(f"Matmul batch dim {i} not broadcastable: {x} vs {y}")
    cd = _dims(c)
    if cd:
        if len(cd) != len(ad):
            raise _not_supported(f"Matmul output rank {len(cd)} != operand rank {len(ad)}")
        if cd[-2] != ad[-2] or cd[-1] != bd[-1]:
            raise _not_supported(f"Matmul output dims {cd} do not match (M, N) = ({ad[-2]}, {bd[-1]})")


def _validate_required(node, label: str, inputs, outputs) -> None:
    """Classic ATTRIBUTE_NOT_SET parity for the MoE grouped-matmul nodes."""
    for port in inputs:
        if node.inputs.get(port) is None:
            raise ValueError(f"{label} input {port} not set.")
    for port in outputs:
        if node.outputs.get(port) is None:
            raise ValueError(f"{label} output {port} not set.")
