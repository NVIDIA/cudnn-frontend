"""Pure-PyTorch reference backend — a correctness baseline with no GPU/JIT deps.

This backend exists so the NativeGraph + BaseEngine + Router contract can be
exercised in CI on CPU, and so every future DSL backend has a numerical oracle
to diff against. It supports MATMUL plus a small set of POINTWISE ops; anything
else is declined (the Router then tries another backend or falls back to cuDNN).

It runs wherever the input tensors live (CPU or CUDA) via ``torch.matmul`` /
elementwise ops, and writes results into the caller-provided output buffers.
"""

from typing import TYPE_CHECKING, Any, Dict

try:
    import torch
except ImportError:
    torch = None

from .base import BaseEngine
from .engine_ids import PYTHON_ENGINE_ID_BASE
from ..graph_types import NodeType

if TYPE_CHECKING:
    from ..graph_native import NativeGraph

# POINTWISE modes this reference understands, keyed by cuDNN pointwise_mode name.
_UNARY = {
    "RELU_FWD": lambda x: x.clamp_min(0),
    "GELU_FWD": lambda x: torch.nn.functional.gelu(x),
    "SIGMOID_FWD": lambda x: torch.sigmoid(x),
    "TANH_FWD": lambda x: torch.tanh(x),
    "EXP": lambda x: torch.exp(x),
    "IDENTITY": lambda x: x,
}
_BINARY = {
    "ADD": lambda a, b: a + b,
    "MUL": lambda a, b: a * b,
    "SUB": lambda a, b: a - b,
    "DIV": lambda a, b: a / b,
}


def _mode_name(mode: Any) -> str:
    return getattr(mode, "name", str(mode)).upper()


class ReferenceMatmulEngine(BaseEngine):
    """CPU/GPU PyTorch reference for MATMUL + basic POINTWISE fusions."""

    name = "reference_matmul"
    engine_id = PYTHON_ENGINE_ID_BASE + 0  # stable id (a correctness oracle)

    def check_support(self, graph: "NativeGraph") -> None:
        if torch is None:
            raise RuntimeError("ReferenceMatmulEngine requires PyTorch")
        for node in graph.nodes:
            if node.node_type == NodeType.MATMUL:
                continue
            if node.node_type == NodeType.POINTWISE:
                mode = _mode_name(node.params.get("mode"))
                if mode not in _UNARY and mode not in _BINARY:
                    raise NotImplementedError(f"ReferenceMatmulEngine: unsupported pointwise mode {mode!r}")
                continue
            raise NotImplementedError(f"ReferenceMatmulEngine only supports MATMUL / basic POINTWISE, got {node.node_type.name}")

    def execute(self, graph: "NativeGraph", tensor_data: Dict[int, Any]) -> None:
        # Nodes are already in build (topological) order. Compute each node into
        # a scratch map, then copy declared outputs into the caller's buffers.
        values: Dict[int, Any] = dict(tensor_data)

        for node in graph.nodes:
            if node.node_type == NodeType.MATMUL:
                a = values[node.inputs["A"].uid]
                b = values[node.inputs["B"].uid]
                out = torch.matmul(a, b)
            elif node.node_type == NodeType.POINTWISE:
                mode = _mode_name(node.params.get("mode"))
                ins = [values[t.uid] for t in node.inputs.values()]
                out = _UNARY[mode](ins[0]) if mode in _UNARY else _BINARY[mode](ins[0], ins[1])
            else:  # pragma: no cover — guarded by check_support
                raise NotImplementedError(node.node_type.name)

            out_t = next(iter(node.outputs.values()))
            dst = values.get(out_t.uid)
            if dst is not None and hasattr(dst, "copy_"):
                dst.copy_(out)  # caller-provided output buffer
            values[out_t.uid] = out
