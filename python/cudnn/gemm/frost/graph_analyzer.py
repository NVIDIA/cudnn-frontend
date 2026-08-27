# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Analyze a user-built ``cudnn.pygraph`` and produce a ``FusionChain``.

``cudnn.pygraph`` is the Python-native graph IR: it records its op DAG directly,
exposed via ``graph.nodes`` / ``graph.tensors``. ``analyze(g)`` reads that IR, so a
graph is analyzable whenever it is built — no construction-time hook required.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import cudnn

_LOG = logging.getLogger(__name__)

from .fusion_ir import (
    out_major_of,
    AMajor,
    BMajor,
    BlockQuantizeSpec,
    OutputSpec,
    Dtype,
    FusionChain,
    FusionOp,
    MatmulSpec,
    ReductionSpec,
    TensorRef,
    gemm_source,
)

# Dtype + op tables

from .dtypes import CUDNN_FROM_DTYPE as _CUDNN_FROM_DTYPE
from .dtypes import DTYPE_FROM_CUDNN as _DTYPE_FROM_CUDNN


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


_UNARY_OP_MAP: dict[str, str] = {
    "relu": "relu",
    "gelu": "gelu",
    "gelu_approx_tanh": "gelu_tanh",
    "swish": "swish",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "exp": "exp",
    "abs": "abs",
    "neg": "neg",
    "cos": "cos",
    "sin": "sin",
    "ceil": "ceil",
    "floor": "floor",
    "erf": "erf",
    "log": "log",
    "reciprocal": "reciprocal",
    "rsqrt": "rsqrt",
    "sqrt": "sqrt",
    "identity": "identity",
    "elu": "elu",
    "softplus": "softplus",
    "tan": "tan",
    "logical_not": "logical_not",
    "leaky_relu": "leaky_relu",
    "gen_index": "gen_index",
}
_BINARY_OP_MAP: dict[str, str] = {
    "add": "add",
    "mul": "mul",
    "sub": "sub",
    "div": "div",
    "max": "max",
    "min": "min",
    "pow": "pow",
    "add_square": "add_square",
    "bias": "add",  # cuDNN's `bias(input, bias)` is just `input + bias`
    "scale": "mul",
    "mod": "mod",
    "logical_and": "logical_and",
    "logical_or": "logical_or",
    "cmp_eq": "cmp_eq",
    "cmp_neq": "cmp_neq",
    "cmp_gt": "cmp_gt",
    "cmp_ge": "cmp_ge",
    "cmp_lt": "cmp_lt",
    "cmp_le": "cmp_le",
    "relu_backward": "relu_backward",
    "leaky_relu_backward": "leaky_relu_backward",
    "swish_backward": "swish_backward",
    "sigmoid_backward": "sigmoid_backward",
    "tanh_backward": "tanh_backward",
    "elu_backward": "elu_backward",
    "gelu_backward": "gelu_backward",
    "gelu_approx_tanh_backward": "gelu_tanh_backward",
    "softplus_backward": "softplus_backward",
}

_POINTWISE_ATTR_KEYS = ("negative_slope", "lower_clip", "upper_clip", "swish_beta", "axis")


# Internal recording state, attached to each cudnn.pygraph instance


@dataclass
class _RecordedOp:
    cudnn_name: str
    op_name: str
    inputs: list[int]
    output: int
    output_tensor: Any  # strong ref so id() stays valid
    compute_dtype: Dtype | None = None  # per-op compute_data_type override (None → graph default)
    # block_scale_dequantize: the [non-K, K] block size (e.g. [1,16] for A). None otherwise.
    block_size: tuple[int, ...] | None = None
    is_negative_scale: bool = False
    # block_scale_quantize: quantized output + the SF side-output from cuDNN.
    scale_output: int | None = None
    scale_output_tensor: Any = None
    quant_axis: int | None = None
    quant_transpose: bool = False
    moe_mode: str | None = None  # moe_grouped_matmul mode; None otherwise
    op_attrs: tuple = ()  # pointwise scalar attrs (negative_slope/clips/swish_beta/axis)
    reduction_mode: str | None = None  # "add"/"amax"/"max"/"min"; None otherwise
    # Optional groupOffset input for grouped reductions / grouped col quant
    # (MoE: == first_token_offset).
    group_offset: int | None = None


@dataclass
class _TensorMeta:
    name: str
    dim: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Dtype
    is_input: bool = False
    # SF reorder layout name (e.g. "F8_128x4") or None for the default (NONE).
    reordering: str | None = None
    # Strong ref to the cuDNN tensor object, used to bind each role for the
    # variant-pack dict (uid / name / object) instead of positional args.
    tensor: Any = None


# The native tensor mirrors set_output / set_data_type / set_dim / set_reordering_type
# onto its own attributes; these side tables (keyed by id(tensor)) carry them into
# the analyzer, repopulated from graph.nodes each analyze (see _state_from_graph).
_TENSOR_OUTPUT_FLAG: dict[int, bool] = {}
_TENSOR_EXPLICIT_DTYPE: dict[int, Any] = {}
_TENSOR_DIM_OVERRIDE: dict[int, tuple[int, ...]] = {}
_TENSOR_REORDERING_OVERRIDE: dict[int, str | None] = {}

# The side tables above are module-global; serialize the analyze path around them.
_ANALYZE_LOCK = threading.RLock()


# Variant-pack binding — maps each graph role to its cuDNN tensor


@dataclass
class GemmBinding:
    """Maps each graph role to its cuDNN tensor so a compiled kernel takes a
    variant-pack dict (keyed by tensor object / uid / name) not positional args.

    Operand lists are in kernel distinct-slot order; ``outputs`` in
    :pyattr:`FusionChain.outputs` slot order (recorder order); ``aux`` in
    :pyattr:`FusionChain.aux_tensors` order. Block-scale fills ``sfa/sfb_operands``
    parallel to ``a/b_operands``; MoE fills ``first_token_offset``."""

    a_operands: list[Any] = field(default_factory=list)
    b_operands: list[Any] = field(default_factory=list)
    outputs: list[Any] = field(default_factory=list)
    aux: list[Any] = field(default_factory=list)
    sfa_operands: list[Any] = field(default_factory=list)
    sfb_operands: list[Any] = field(default_factory=list)
    first_token_offset: Any = None

    def bound_tensors(self) -> list[Any]:
        ts = [
            *self.a_operands,
            *self.b_operands,
            *self.outputs,
            *self.aux,
            *self.sfa_operands,
            *self.sfb_operands,
        ]
        if self.first_token_offset is not None:
            ts.append(self.first_token_offset)
        return [t for t in ts if t is not None]


def _make_multi_binding(
    meta: dict,
    a_ids,
    b_ids,
    a_caps,
    b_caps,
    output_objs,
    aux_objs,
    block_scale: bool,
    first_token_offset=None,
) -> GemmBinding:
    """Build a GemmBinding for the multi-operand builders (multi-GEMM / MoE):
    the cuDNN tensor per distinct A/B slot (+ its SF for block-scale) from ``meta``."""

    def _sf(caps: dict, ids) -> list:
        objs = []
        for i in ids:
            sid = caps[i].get("sf_id")
            objs.append(meta[sid].tensor if sid is not None else None)
        return objs

    return GemmBinding(
        a_operands=[meta[i].tensor for i in a_ids],
        b_operands=[meta[i].tensor for i in b_ids],
        outputs=list(output_objs),
        aux=list(aux_objs),
        sfa_operands=_sf(a_caps, a_ids) if block_scale else [],
        sfb_operands=_sf(b_caps, b_ids) if block_scale else [],
        first_token_offset=first_token_offset,
    )


def _safe_name(t: Any) -> str | None:
    try:
        nm = t.get_name()
    except Exception:  # noqa: BLE001 — unbuilt tensors can throw
        return None
    return nm or None


def _safe_uid(t: Any) -> int | None:
    try:
        uid = t.get_uid()
    except Exception:  # noqa: BLE001
        return None
    # uid is -1/0 until build_operation_graph(); only a positive uid is a valid key.
    return uid if isinstance(uid, int) and uid > 0 else None


def resolve_variant_pack(variant_pack: dict, binding: GemmBinding) -> dict[int, Any]:
    """Resolve a ``{key: buffer}`` variant pack to ``{id(bound_tensor): buffer}``.

    Keys may be the cuDNN tensor object, its uid (once positive), or its name.
    Raises on an unknown / unmatched key."""
    if not isinstance(variant_pack, dict):
        raise TypeError("compiled kernels are called with a variant-pack dict " "{cudnn_tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}")
    bound = binding.bound_tensors()
    by_obj = {id(t): t for t in bound}

    name_counts: dict[str, int] = {}
    uid_counts: dict[int, int] = {}
    for t in bound:
        nm = _safe_name(t)
        if nm is not None:
            name_counts[nm] = name_counts.get(nm, 0) + 1
        uid = _safe_uid(t)
        if uid is not None:
            uid_counts[uid] = uid_counts.get(uid, 0) + 1
    by_name = {_safe_name(t): t for t in bound if name_counts.get(_safe_name(t)) == 1}
    by_uid = {_safe_uid(t): t for t in bound if uid_counts.get(_safe_uid(t)) == 1}
    by_name.pop(None, None)
    by_uid.pop(None, None)

    resolved: dict[int, Any] = {}
    for key, buf in variant_pack.items():
        if id(key) in by_obj:
            t = by_obj[id(key)]
        elif isinstance(key, int) and key in by_uid:
            t = by_uid[key]
        elif isinstance(key, str) and key in by_name:
            t = by_name[key]
        else:
            kdesc = key if isinstance(key, (int, str)) else _safe_name(key)
            raise KeyError(
                f"variant-pack key {kdesc!r} does not match any input / output "
                "tensor of this graph (key by the cuDNN tensor object, its uid, "
                "or its name)"
            )
        resolved[id(t)] = buf
    return resolved


# MoE / reduction mode maps (cuDNN enum -> our literal)

_MOE_MODE_FROM_CUDNN: dict[Any, str] = {
    cudnn.moe_grouped_matmul_mode.NONE: "none",
    cudnn.moe_grouped_matmul_mode.GATHER: "gather",
    cudnn.moe_grouped_matmul_mode.SCATTER: "scatter",
}

_REDUCTION_MODE_FROM_CUDNN: dict[Any, str] = {
    cudnn.reduction_mode.ADD: "add",
    cudnn.reduction_mode.AMAX: "amax",
    cudnn.reduction_mode.MAX: "max",
    cudnn.reduction_mode.MIN: "min",
    cudnn.reduction_mode.AVG: "avg",
    cudnn.reduction_mode.MUL: "mul",
    cudnn.reduction_mode.MUL_NO_ZEROS: "mul_no_zeros",
    cudnn.reduction_mode.NORM1: "norm1",
    cudnn.reduction_mode.NORM2: "norm2",
}


# Reading the native cudnn.pygraph IR (graph.nodes) into analyzer state


def _map_dtype(dt: Any) -> "Dtype | None":
    """cuDNN data_type enum (or None) -> our Dtype literal (or None)."""
    if dt is None:
        return None
    return _DTYPE_FROM_CUDNN.get(dt)


def _reordering_name(t: Any) -> "str | None":
    """SF reorder layout name (e.g. ``F8_128x4``) of a tensor, or None (default)."""
    rt = getattr(t, "reordering_type", None)
    name = getattr(rt, "name", None) if rt is not None else None
    return name if (name and name != "NONE") else None


def _node_to_recorded_op(node: Any) -> "_RecordedOp | None":
    """Translate one native-IR ``Node`` into a :class:`_RecordedOp`, or None when
    this family has no lowering for that node type. The caller must DECLINE on
    None — never skip the node."""
    node_type = node.node_type.name
    name = node.name
    compute = _map_dtype(node.compute_data_type)
    if node_type == "MATMUL":
        A, B = node.inputs["A"], node.inputs["B"]
        out = node.outputs["C"]
        return _RecordedOp("matmul", name, [id(A), id(B)], id(out), out, compute_dtype=compute)
    if node_type == "MOE_GROUPED_MATMUL":
        tok = node.inputs["token"]
        weight = node.inputs["weight"]
        fto = node.inputs["first_token_offset"]
        out = node.outputs["OUT_0"]
        mode = node.params.get("mode", cudnn.moe_grouped_matmul_mode.NONE)
        return _RecordedOp(
            "moe_grouped_matmul",
            name,
            [id(tok), id(weight), id(fto)],
            id(out),
            out,
            compute_dtype=compute,
            moe_mode=_MOE_MODE_FROM_CUDNN.get(mode, "none"),
        )
    if node_type == "REDUCTION":
        inp = node.inputs["input"]
        out = node.outputs["OUT_0"]
        group_offset = node.inputs.get("group_offset")
        return _RecordedOp(
            "reduction",
            name,
            [id(inp)],
            id(out),
            out,
            compute_dtype=compute,
            reduction_mode=_REDUCTION_MODE_FROM_CUDNN.get(node.params.get("mode")),
            group_offset=(id(group_offset) if group_offset is not None else None),
        )
    if node_type == "BLOCK_SCALE_DEQUANTIZE":
        inp = node.inputs["input"]
        descale = node.inputs["descale"]
        out = node.outputs["OUT_0"]
        block_size = node.params.get("block_size")
        return _RecordedOp(
            "block_scale_dequantize",
            name,
            [id(inp), id(descale)],
            id(out),
            out,
            compute_dtype=compute,
            block_size=tuple(block_size) if block_size is not None else None,
            is_negative_scale=bool(node.params.get("is_negative_scale", False)),
        )
    if node_type == "BLOCK_SCALE_QUANTIZE":
        inp = node.inputs["input"]
        quantized = node.outputs["Y"]
        scale = node.outputs["scale"]
        block_size = node.params.get("block_size")
        if isinstance(block_size, (list, tuple)):
            if len(block_size) != 1:
                raise NotImplementedError(f"block_scale_quantize expects a scalar block_size in cudnn.gemm.frost; got {block_size!r}")
            block_size_i = int(block_size[0])
        else:
            block_size_i = int(block_size)
        axis = node.params.get("axis")
        group_offset = node.inputs.get("group_offset")
        return _RecordedOp(
            "block_scale_quantize",
            name,
            [id(inp)],
            id(quantized),
            quantized,
            compute_dtype=compute,
            block_size=(block_size_i,),
            scale_output=id(scale),
            scale_output_tensor=scale,
            quant_axis=-1 if axis is None else int(axis),
            quant_transpose=bool(node.params.get("transpose", False)),
            group_offset=(id(group_offset) if group_offset is not None else None),
        )
    if node_type == "POINTWISE":
        out = node.outputs["OUT_0"]
        attrs = tuple(sorted((k, float(v)) for k, v in node.params.items() if k in _POINTWISE_ATTR_KEYS and v is not None))
        return _RecordedOp(
            node.params.get("mode"),
            name,
            [id(t) for t in node.inputs.values()],
            id(out),
            out,
            compute_dtype=compute,
            op_attrs=attrs,
        )
    return None


def _state_from_graph(graph: cudnn.pygraph) -> dict:
    """Read a Python-native ``cudnn.pygraph`` into the analyzer's working state:
    the op list + per-tensor metadata + graph dtype defaults, plus the tensor-flag
    side tables. The native graph exposes its op DAG directly via ``graph.nodes``,
    so nothing is recorded at construction time."""
    _TENSOR_OUTPUT_FLAG.clear()
    _TENSOR_EXPLICIT_DTYPE.clear()
    _TENSOR_DIM_OVERRIDE.clear()
    _TENSOR_REORDERING_OVERRIDE.clear()

    ctx = graph.context
    raw_io = getattr(ctx, "io_data_type", None)
    raw_intermediate = getattr(ctx, "intermediate_data_type", None)
    raw_compute = getattr(ctx, "compute_data_type", None)
    io_dtype = _map_dtype(raw_io)
    intermediate_dtype = _map_dtype(raw_intermediate)
    compute_dtype = _map_dtype(raw_compute)
    for _raw, _mapped, _field in (
        (raw_io, io_dtype, "io_data_type"),
        (raw_intermediate, intermediate_dtype, "intermediate_data_type"),
        (raw_compute, compute_dtype, "compute_data_type"),
    ):
        if _raw is not None and _mapped is None:
            raise ValueError(f"unsupported {_field}: {_raw!r}")
    io_dtype = io_dtype or "bf16"
    intermediate_dtype = intermediate_dtype or "fp32"
    compute_dtype = compute_dtype or "fp32"

    nodes = list(graph.nodes)
    produced: set[int] = set()
    for node in nodes:
        for out in node.outputs.values():
            if out is not None:
                produced.add(id(out))

    tensor_meta: dict[int, _TensorMeta] = {}

    def _register(t: Any) -> None:
        if t is None or id(t) in tensor_meta:
            return
        reordering = _reordering_name(t)
        tensor_meta[id(t)] = _TensorMeta(
            name=t.get_name(),
            dim=tuple(t.dim),
            stride=tuple(t.stride),
            dtype=_map_dtype(t.get_data_type()),
            is_input=id(t) not in produced,
            reordering=reordering,
            tensor=t,
        )
        if getattr(t, "data_type", None) is not None:
            _TENSOR_EXPLICIT_DTYPE[id(t)] = t.get_data_type()
        if getattr(t, "dim_assigned", False) and id(t) in produced:
            _TENSOR_DIM_OVERRIDE[id(t)] = tuple(t.dim)
        if reordering is not None:
            _TENSOR_REORDERING_OVERRIDE[id(t)] = reordering

    for node in nodes:
        for t in node.inputs.values():
            _register(t)
        for t in node.outputs.values():
            _register(t)

    # Materialized (non-virtual) op outputs, in node order.
    for node in nodes:
        for out in node.outputs.values():
            if out is not None and not out.is_virtual:
                _TENSOR_OUTPUT_FLAG[id(out)] = True

    ops: list[_RecordedOp] = []
    for node in nodes:
        recorded = _node_to_recorded_op(node)
        if recorded is None:
            # Declining beats ignoring. Dropping an unrecognized node compiled a
            # SUBGRAPH and then asked the caller for buffers it never bound
            # (matmul -> reshape died in execute with "missing buffers for
            # ['mm::C']"), and because the engine had already claimed the graph
            # there was no backend left to fall back to.
            raise NotImplementedError(f"frost_gemm: no lowering for node type {node.node_type.name}")
        ops.append(recorded)

    for op in ops:
        if op.cudnn_name == "block_scale_dequantize":
            in_meta = tensor_meta.get(op.inputs[0])
            out_meta = tensor_meta.get(op.output)
            if in_meta is not None and out_meta is not None:
                out_meta.dim = in_meta.dim
                out_meta.stride = in_meta.stride

    return {
        "ops": ops,
        "tensor_meta": tensor_meta,
        "io_dtype": io_dtype,
        "intermediate_dtype": intermediate_dtype,
        "compute_dtype": compute_dtype,
    }


def _graph_has_gemm(graph: cudnn.pygraph) -> bool:
    """True if the graph has any matmul / MoE grouped-matmul node (GEMM candidate)."""
    try:
        for node in graph.nodes:
            if node.node_type.name in ("MATMUL", "MOE_GROUPED_MATMUL"):
                return True
    except Exception:  # noqa: BLE001 — a probe must never break the native path
        return False
    return False


# The two halves the GEMM engine (cudnn/gemm/frost/engine.py) is built from:
# probe_gemm_plan = eligibility, no compile; build_gemm_plan = the JIT, run when
# the plan walk reaches this engine. Forced-config callers use
# jit_from_cudnn_graph directly.


def probe_gemm_plan(graph: cudnn.pygraph) -> bool:
    """Cheap eligibility check for the GEMM engine (analyze + support gates, NO
    ``cute.compile``). Never raises (a probe must not break the native path)."""
    if not _graph_has_gemm(graph):
        return False
    from .compiler import probe_supported

    try:
        probe_supported(graph)
    except (NotImplementedError, ValueError):
        return False
    except Exception:  # noqa: BLE001
        _LOG.debug(
            "cudnn.gemm.frost: probe_supported raised unexpectedly; ineligible",
            exc_info=True,
        )
        return False
    return True


def build_gemm_plan(graph: cudnn.pygraph):
    """Analyze + JIT the graph into a compiled GEMM plan.

    Returns a callable :class:`CompiledFusedGemm`; raises ``NotImplementedError`` /
    ``ValueError`` (type + message preserved) on rejection."""
    if not _graph_has_gemm(graph):
        raise ValueError("cudnn.gemm.frost: graph has no matmul / moe_grouped_matmul node; nothing to compile")
    from .compiler import jit_from_cudnn_graph, plan_config

    config, cta_group = plan_config(analyze(graph))
    return jit_from_cudnn_graph(graph, config=config, cta_group=cta_group)


# Analyzer


def _infer_bcast_mode(matmul_out_dim: tuple[int, ...], aux_dim: tuple[int, ...]) -> str:
    """Infer how an aux tensor broadcasts onto the matmul output."""
    if len(aux_dim) > len(matmul_out_dim):
        raise ValueError(f"aux dim rank {len(aux_dim)} exceeds matmul-out rank " f"{len(matmul_out_dim)}: aux={aux_dim} out={matmul_out_dim}")
    aux_norm = (1,) * (len(matmul_out_dim) - len(aux_dim)) + tuple(aux_dim)
    for aux_extent, out_extent in zip(aux_norm, matmul_out_dim):
        if aux_extent not in (1, out_extent):
            raise ValueError(f"aux dim {aux_dim} is not broadcast-compatible with " f"matmul output dim {matmul_out_dim}")
    bcast_m = aux_norm[-2] == 1 and matmul_out_dim[-2] != 1
    bcast_n = aux_norm[-1] == 1 and matmul_out_dim[-1] != 1
    if bcast_m and bcast_n:
        return "scalar"
    if bcast_m and not bcast_n:
        return "per_col"
    if bcast_n and not bcast_m:
        return "per_row"
    return "per_elem"


def _infer_a_major(dim: tuple[int, ...], stride: tuple[int, ...]) -> AMajor:
    if stride[-1] == 1:
        return "k"
    if stride[-2] == 1:
        return "m"
    raise ValueError(f"A must be K-major or M-major in the inner (M,K) plane; " f"got dim={dim} stride={stride}")


def _infer_b_major(dim: tuple[int, ...], stride: tuple[int, ...]) -> BMajor:
    if stride[-2] == 1:
        return "k"
    if stride[-1] == 1:
        return "n"
    raise ValueError(f"B must be K-major or N-major in the inner (K,N) plane; " f"got dim={dim} stride={stride}")


def _resolve_out_dtype(
    out_id: int,
    output_tensor: Any,
    io_dtype: Dtype,
    intermediate_dtype: Dtype,
) -> Dtype:
    """Declared data_type of a chain tensor: explicit set_data_type, else io_dtype
    if a materialized output, else intermediate_dtype.

    The running value is rounded to this dtype before downstream ops read it, so a
    narrow declared dtype loses precision on purpose (matches cuDNN, even virtual)."""
    explicit = _TENSOR_EXPLICIT_DTYPE.get(out_id)
    if explicit is not None and explicit in _DTYPE_FROM_CUDNN:
        return _DTYPE_FROM_CUDNN[explicit]
    if output_tensor is not None:
        try:
            dt = output_tensor.get_data_type()
        except Exception:  # noqa: BLE001 — defensive: unbuilt tensors vary
            dt = None
        if dt is not None and dt != cudnn.data_type.NOT_SET and dt in _DTYPE_FROM_CUDNN:
            return _DTYPE_FROM_CUDNN[dt]
    if _TENSOR_OUTPUT_FLAG.get(out_id, False):
        return io_dtype
    return intermediate_dtype


def _collect_quants(
    reachable_quant_ops: list["_RecordedOp"],
    op_position_by_id: dict[int, int],
    gemm_idx_by_output: dict[int, int],
    allow_gemm_source: bool,
    meta: dict[int, "_TensorMeta"],
    io_dtype: Dtype,
    intermediate_dtype: Dtype,
    compute_dtype: Dtype,
    batch: int,
    M: int,
    N: int,
    err_ctx: str,
    fto_id: int | None = None,
) -> tuple[list[BlockQuantizeSpec], list["_RecordedOp"], list[Dtype], list[Any]]:
    """Fold every reachable ``block_scale_quantize`` whose data output is
    materialized into a :class:`BlockQuantizeSpec`. Returns (specs, recorded
    ops, data dtypes, scale output tensors), all in graph-node order."""
    quants: list[BlockQuantizeSpec] = []
    recs: list[_RecordedOp] = []
    data_dtypes: list[Dtype] = []
    scale_objs: list[Any] = []
    for qop in reachable_quant_ops:
        if not _TENSOR_OUTPUT_FLAG.get(qop.output, False):
            continue
        (input_id,) = qop.inputs
        if input_id in op_position_by_id:
            source_ref = op_position_by_id[input_id]
        elif allow_gemm_source and input_id in gemm_idx_by_output:
            source_ref = gemm_source(gemm_idx_by_output[input_id])
        else:
            raise ValueError(f"block_scale_quantize {qop.op_name!r} input must be a fusion-op output " f"of the shared {err_ctx} epilogue")
        if qop.scale_output is None or qop.scale_output_tensor is None:
            raise AssertionError("block_scale_quantize recorded without scale output")
        if not _TENSOR_OUTPUT_FLAG.get(qop.scale_output, False):
            raise ValueError("block_scale_quantize scale output must be materialized with set_output(True)")
        scale_dtype = _resolve_out_dtype(qop.scale_output, qop.scale_output_tensor, io_dtype, intermediate_dtype)
        scale_reorder = _TENSOR_REORDERING_OVERRIDE.get(qop.scale_output)
        if scale_reorder is None:
            scale_meta = meta.get(qop.scale_output)
            scale_reorder = scale_meta.reordering if scale_meta is not None else None
        bs = int(qop.block_size[0]) if qop.block_size else 0
        if bs <= 0:
            raise ValueError(f"block_scale_quantize block_size must be positive; got {bs}")
        axis = -1 if qop.quant_axis is None else qop.quant_axis
        if qop.quant_transpose:
            axis = 1
        grouped_by_moe = False
        if qop.group_offset is not None:
            if fto_id is None or qop.group_offset != fto_id:
                raise ValueError(f"block_scale_quantize {qop.op_name!r} groupOffset must be the MoE " "first_token_offset tensor")
            if axis != 1:
                raise ValueError(
                    f"block_scale_quantize {qop.op_name!r} with groupOffset supports only "
                    "the M axis (axis=1, col quant); row scales are already per-group "
                    "contiguous in the global layout"
                )
            if scale_reorder != "F8_128x4":
                raise ValueError(f"block_scale_quantize {qop.op_name!r} with groupOffset requires " "F8_128x4 scale reordering")
            grouped_by_moe = True
        if axis == 1:
            if int(M) % bs != 0:
                raise ValueError(f"col block_scale_quantize requires M divisible by block_size; " f"got M={M}, block_size={bs}")
            logical_scale_dim = (int(batch), int(M) // bs, int(N))
            expected_scale_dim = logical_scale_dim
            if scale_reorder == "F8_128x4":
                expected_scale_dim = (
                    logical_scale_dim[0],
                    _round_up(int(N), 128),
                    _round_up(int(M) // bs, 4),
                )
        else:
            if int(N) % bs != 0:
                raise ValueError(f"block_scale_quantize requires N divisible by block_size; got N={N}, block_size={bs}")
            logical_scale_dim = (int(batch), int(M), int(N) // bs)
            expected_scale_dim = logical_scale_dim
            if scale_reorder == "F8_128x4":
                expected_scale_dim = (
                    logical_scale_dim[0],
                    _round_up(logical_scale_dim[1], 128),
                    _round_up(logical_scale_dim[2], 4),
                )
        scale_dim = _TENSOR_DIM_OVERRIDE.get(qop.scale_output)
        if scale_dim is None:
            scale_meta = meta.get(qop.scale_output)
            scale_dim = scale_meta.dim if scale_meta is not None else ()
        if not scale_dim:
            scale_dim = expected_scale_dim
        if len(scale_dim) != 3:
            raise ValueError(f"block_scale_quantize scale output must be rank-3; got {scale_dim}")
        if tuple(scale_dim) != expected_scale_dim:
            raise ValueError(f"block_scale_quantize scale dim must be {expected_scale_dim}; got {scale_dim}")
        compute = qop.compute_dtype if qop.compute_dtype is not None else compute_dtype
        quants.append(
            BlockQuantizeSpec(
                source_ref=source_ref,
                block_size=bs,
                axis=axis,
                transpose=qop.quant_transpose,
                scale_dtype=scale_dtype,
                scale_dim=tuple(scale_dim),
                scale_reorder=scale_reorder,
                compute_dtype=compute,
                grouped_by_moe=grouped_by_moe,
            )
        )
        recs.append(qop)
        dt: Dtype = io_dtype
        explicit = qop.output_tensor.get_data_type()
        if explicit != cudnn.data_type.NOT_SET and explicit in _DTYPE_FROM_CUDNN:
            dt = _DTYPE_FROM_CUDNN[explicit]
        data_dtypes.append(dt)
        scale_objs.append(qop.scale_output_tensor)
    return quants, recs, data_dtypes, scale_objs


def _build_multi_moe_chain(
    moe_ops: list[_RecordedOp],
    ops: list[_RecordedOp],
    meta: dict[int, _TensorMeta],
    io_dtype: Dtype,
    intermediate_dtype: Dtype,
    compute_dtype: Dtype,
) -> FusionChain:
    """Build a FusionChain for K >= 1 MoE grouped matmuls sharing one
    ``first_token_offset`` and one pointwise epilogue DAG (the unified builder
    for every MoE graph; e.g. grouped SwiGLU).

    All GEMMs must share the routed-group layout (same fto), shape / major / dtype,
    and expert count. Operands deduped by tensor id (shared token → one A operand).
    K == 1 additionally supports: no epilogue at all (raw MoE output alone)
    and the raw output as quant source. POC scope: mode=="none", no mainloop
    fusion; for K > 1 every output must be a fusion op or a block_scale_quantize
    fed by one. Block-scale supported (dequant folds into a shared
    :class:`BlockScaleSpec`)."""
    from .fusion_ir import BlockQuantizeSpec, BlockScaleSpec, MoeSpec

    if any(op.cudnn_name == "matmul" for op in ops):
        raise ValueError("a MoE grouped matmul graph cannot also contain a plain matmul; " "mixed MoE + matmul graphs are out of POC scope")
    for moe in moe_ops:
        if moe.moe_mode != "none":
            raise NotImplementedError(
                f"MoE grouped matmul mode {moe.moe_mode!r} is out of POC scope; " "only mode=NONE is supported (gather / scatter rejected)"
            )

    # All GEMMs must share the SAME first_token_offset (identical routed-group layout).
    fto_id = moe_ops[0].inputs[2]
    for moe in moe_ops[1:]:
        if moe.inputs[2] != fto_id:
            raise ValueError("parallel MoE grouped matmuls must share the same " "first_token_offset tensor")
    fto_meta = meta.get(fto_id)
    offset_dtype = fto_meta.dtype if fto_meta is not None else "int32"
    num_groups = int(fto_meta.dim[0]) if fto_meta is not None and fto_meta.dim else 1

    # Resolve each moe operand through any dequant, then dedup by PACKED data
    # tensor id (shared dequant → one distinct operand; SF travels with its data).
    dequant_by_output = {op.output: op for op in ops if op.cudnn_name == "block_scale_dequantize"}

    def _capture_side(operand_id: int) -> dict:
        deq = dequant_by_output.get(operand_id)
        if deq is None:
            return dict(
                data_id=operand_id,
                data_dtype=meta[operand_id].dtype,
                block_size_2d=None,
                sf_dtype=None,
                sf_reorder=None,
                deq_compute=None,
                deq_out=None,
                sf_id=None,
            )
        data_id, sf_id = deq.inputs
        sf_meta = meta[sf_id]
        deq_compute = deq.compute_dtype if deq.compute_dtype is not None else compute_dtype
        deq_out = _resolve_out_dtype(deq.output, deq.output_tensor, io_dtype, intermediate_dtype)
        return dict(
            data_id=data_id,
            data_dtype=meta[data_id].dtype,
            block_size_2d=(tuple(deq.block_size) if deq.block_size else None),
            sf_dtype=sf_meta.dtype,
            sf_reorder=sf_meta.reordering,
            deq_compute=deq_compute,
            deq_out=deq_out,
            sf_id=sf_id,
        )

    a_ids: list[int] = []  # distinct PACKED token (A) data ids
    b_ids: list[int] = []  # distinct PACKED weight (B) data ids
    a_caps: dict[int, dict] = {}
    b_caps: dict[int, dict] = {}
    gemm_operands: list[tuple[int, int]] = []
    for moe in moe_ops:
        a_cap = _capture_side(moe.inputs[0])
        b_cap = _capture_side(moe.inputs[1])
        a_pid, b_pid = a_cap["data_id"], b_cap["data_id"]
        if a_pid not in a_ids:
            a_ids.append(a_pid)
            a_caps[a_pid] = a_cap
        if b_pid not in b_ids:
            b_ids.append(b_pid)
            b_caps[b_pid] = b_cap
        gemm_operands.append((a_ids.index(a_pid), b_ids.index(b_pid)))

    is_block_scale = any(c["sf_dtype"] is not None for c in (*a_caps.values(), *b_caps.values()))

    def _moe_geometry(token_id: int, weight_id: int):
        token_meta = meta[token_id]
        weight_meta = meta[weight_id]
        if len(token_meta.dim) != 3 or len(weight_meta.dim) != 3:
            raise ValueError(f"moe operands must be 3D; got token={token_meta.dim} " f"weight={weight_meta.dim}")
        _bt, M, Ka = token_meta.dim  # token [1, T, H]
        E, Kb, N = weight_meta.dim  # weight [E, H, N]
        if Ka != Kb:
            raise ValueError(f"moe K mismatch: token K={Ka} vs weight K={Kb}")
        return (
            int(M),
            int(N),
            int(Ka),
            int(E),
            _infer_a_major(token_meta.dim, token_meta.stride),
            _infer_b_major(weight_meta.dim, weight_meta.stride),
            token_meta.dtype,
            weight_meta.dtype,
        )

    geom0 = _moe_geometry(a_ids[gemm_operands[0][0]], b_ids[gemm_operands[0][1]])
    for ai, bi in gemm_operands[1:]:
        if _moe_geometry(a_ids[ai], b_ids[bi]) != geom0:
            raise ValueError("parallel MoE grouped matmuls must share shape / layout / dtype " "/ expert count; heterogeneous GEMMs are out of POC scope")
    M, N, K, E, a_major, b_major, a_dtype, b_dtype = geom0
    matmul_out_dim = (1, M, N)

    # Shared BlockScaleSpec (every distinct operand must match GEMM 0's combo).
    block_scale_spec = None
    if is_block_scale:
        a0 = a_caps[a_ids[gemm_operands[0][0]]]
        b0 = b_caps[b_ids[gemm_operands[0][1]]]

        def _combo_key(cap):
            return (
                cap["data_dtype"],
                cap["block_size_2d"],
                cap["sf_dtype"],
                cap["sf_reorder"],
                cap["deq_compute"],
                cap["deq_out"],
            )

        for cap in a_caps.values():
            if _combo_key(cap) != _combo_key(a0):
                raise ValueError("all token operands of a block-scale multi-MoE must share the same SF combo")
        for cap in b_caps.values():
            if _combo_key(cap) != _combo_key(b0):
                raise ValueError("all weight operands of a block-scale multi-MoE must share the same SF combo")
        block_scale_spec = BlockScaleSpec(
            a_dtype=a0["data_dtype"],
            b_dtype=b0["data_dtype"],
            block_size_a=a0["block_size_2d"],
            block_size_b=b0["block_size_2d"],
            sf_dtype_a=a0["sf_dtype"],
            sf_dtype_b=b0["sf_dtype"],
            sfa_reorder=a0["sf_reorder"],
            sfb_reorder=b0["sf_reorder"],
            dequant_compute_a=a0["deq_compute"],
            dequant_compute_b=b0["deq_compute"],
            dequant_out_a=a0["deq_out"],
            dequant_out_b=b0["deq_out"],
        )
    mm_compute = moe_ops[0].compute_dtype if moe_ops[0].compute_dtype is not None else compute_dtype

    # Epilogue DAG over multiple roots (each MoE GEMM output).
    gemm_idx_by_output: dict[int, int] = {mm.output: g for g, mm in enumerate(moe_ops)}
    consumers_by_input: dict[int, list[_RecordedOp]] = {}
    for op in ops:
        for inp in op.inputs:
            consumers_by_input.setdefault(inp, []).append(op)

    reachable_op_ids: set[int] = set()
    bfs_queue: list[int] = [mm.output for mm in moe_ops]
    visited_tensors: set[int] = set()
    while bfs_queue:
        tid = bfs_queue.pop(0)
        if tid in visited_tensors:
            continue
        visited_tensors.add(tid)
        for op in consumers_by_input.get(tid, []):
            if op.cudnn_name == "moe_grouped_matmul":
                continue
            if op.output not in reachable_op_ids:
                reachable_op_ids.add(op.output)
                bfs_queue.append(op.output)
    pointwise_producer = {op.output: op for op in ops if op.cudnn_name in _UNARY_OP_MAP or op.cudnn_name in _BINARY_OP_MAP or op.cudnn_name == "binary_select"}
    demand: list[int] = [tid for tid, flagged in _TENSOR_OUTPUT_FLAG.items() if flagged]
    for op in ops:
        if op.output in reachable_op_ids:
            demand.extend(op.inputs)
        elif op.cudnn_name in ("reduction", "block_scale_quantize") and _TENSOR_OUTPUT_FLAG.get(op.output, False):
            demand.extend(op.inputs)
    while demand:
        tid = demand.pop()
        prod = pointwise_producer.get(tid)
        if prod is not None and prod.output not in reachable_op_ids:
            reachable_op_ids.add(prod.output)
            demand.extend(prod.inputs)
    reachable_ops = [
        op for op in ops if op.output in reachable_op_ids and op.cudnn_name not in {"moe_grouped_matmul", "matmul", "reduction", "block_scale_quantize"}
    ]
    reachable_quant_ops = [
        op for op in ops if op.cudnn_name == "block_scale_quantize" and (op.output in reachable_op_ids or any(i in reachable_op_ids for i in op.inputs))
    ]

    def _is_in_chain(tid: int) -> bool:
        return tid in gemm_idx_by_output or tid in reachable_op_ids

    in_chain_deps: dict[int, list[int]] = {op.output: [inp for inp in op.inputs if _is_in_chain(inp)] for op in reachable_ops}
    placed: set[int] = set(gemm_idx_by_output)
    remaining = list(reachable_ops)
    ordered_ops: list[_RecordedOp] = []
    while remaining:
        ready_idx = next(
            (i for i, op in enumerate(remaining) if all(d in placed for d in in_chain_deps[op.output])),
            None,
        )
        if ready_idx is None:
            raise AssertionError(f"cycle / unsatisfiable deps: {[op.op_name for op in remaining]}")
        op = remaining.pop(ready_idx)
        ordered_ops.append(op)
        placed.add(op.output)

    aux_tensors: list[TensorRef] = []
    aux_objs: list[Any] = []
    aux_seen: set[int] = set()
    op_position_by_id: dict[int, int] = {}
    pending_ops: list[tuple[FusionOp, int]] = []

    def _register_aux(aux_id: int, op_name: str) -> str:
        aux_meta = meta[aux_id]
        if not aux_meta.is_input:
            raise ValueError(f"aux input {aux_meta.name!r} of op {op_name!r} is " "not a graph input — POC supports only graph-input aux")
        if aux_id not in aux_seen:
            aux_seen.add(aux_id)
            aux_dim = tuple(aux_meta.dim)
            grouped_aux = len(aux_dim) == 3 and aux_dim[0] != 1
            if grouped_aux:
                if aux_dim[0] != num_groups:
                    raise ValueError(f"aux {aux_meta.name!r} leading dim {aux_dim[0]} must be 1 or " f"num_groups ({num_groups}) for a MoE epilogue aux")
                bcast = _infer_bcast_mode(matmul_out_dim, (1,) + aux_dim[1:])
            else:
                bcast = _infer_bcast_mode(matmul_out_dim, aux_dim)
            aux_tensors.append(
                TensorRef(
                    name=aux_meta.name,
                    dim=aux_meta.dim,
                    stride=aux_meta.stride,
                    dtype=aux_meta.dtype,
                    bcast_mode=bcast,
                    grouped_by_moe=grouped_aux,
                )
            )
            aux_objs.append(aux_meta.tensor)
        return aux_meta.name

    def _aux_root_ref(tid: int, op_name: str) -> int:
        if tid in op_position_by_id:
            return op_position_by_id[tid]
        name = _register_aux(tid, op_name)
        pending_ops.append((FusionOp(op="aux_load", aux=name), tid))
        op_position_by_id[tid] = len(pending_ops) - 1
        return op_position_by_id[tid]

    def _operand_ref(tid: int, op_name: str = "") -> int:
        if tid in gemm_idx_by_output:
            return gemm_source(gemm_idx_by_output[tid])
        if tid in op_position_by_id:
            return op_position_by_id[tid]
        m = meta.get(tid)
        if m is not None and m.is_input:
            return _aux_root_ref(tid, op_name)
        raise ValueError(f"op {op_name!r} input is not produced by this epilogue chain and is " "not a graph input")

    for next_op in ordered_ops:
        if next_op.cudnn_name in _UNARY_OP_MAP:
            (parent_id,) = next_op.inputs
            if next_op.cudnn_name == "gen_index" and dict(next_op.op_attrs).get("axis") not in (1, 2):
                raise NotImplementedError(f"gen_index {next_op.op_name!r}: only axis 1 (M) or 2 (N) is supported " "in cudnn.gemm.frost")
            fop = FusionOp(op=_UNARY_OP_MAP[next_op.cudnn_name], parent_idx=_operand_ref(parent_id, next_op.op_name), attrs=next_op.op_attrs)
        elif next_op.cudnn_name in _BINARY_OP_MAP:
            inp0, inp1 = next_op.inputs
            in0, in1 = _is_in_chain(inp0), _is_in_chain(inp1)
            if in0 and in1:
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=None,
                    aux_on_rhs=True,
                    parent_idx=_operand_ref(inp0),
                    parent_idx_b=_operand_ref(inp1),
                    attrs=next_op.op_attrs,
                )
            elif in0 or in1:
                if in0:
                    chain_id, aux_id, aux_on_rhs = inp0, inp1, True
                else:
                    chain_id, aux_id, aux_on_rhs = inp1, inp0, False
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=_register_aux(aux_id, next_op.op_name),
                    aux_on_rhs=aux_on_rhs,
                    parent_idx=_operand_ref(chain_id, next_op.op_name),
                    attrs=next_op.op_attrs,
                )
            else:
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=_register_aux(inp1, next_op.op_name),
                    aux_on_rhs=True,
                    parent_idx=_operand_ref(inp0, next_op.op_name),
                    attrs=next_op.op_attrs,
                )
        elif next_op.cudnn_name == "binary_select":
            i0, i1, im = next_op.inputs
            fop = FusionOp(
                op="binary_select",
                parent_idx=_operand_ref(i0, next_op.op_name),
                parent_idx_b=_operand_ref(i1, next_op.op_name),
                parent_idx_c=_operand_ref(im, next_op.op_name),
            )
        else:
            raise ValueError(f"op {next_op.cudnn_name!r} (name={next_op.op_name!r}) is not in " "the POC pointwise subset; out-of-scope")
        pending_ops.append((fop, next_op.output))
        op_position_by_id[next_op.output] = len(pending_ops) - 1

    if not pending_ops and len(moe_ops) > 1:
        raise ValueError("multi-MoE graph has no fusion op; parallel grouped matmuls must " "share a pointwise epilogue (the no-epilogue case is out of scope)")

    set_output_ids_in_order = [tid for tid in _TENSOR_OUTPUT_FLAG if _TENSOR_OUTPUT_FLAG[tid]]

    from dataclasses import replace as _replace

    recorded_by_out = {op.output: op for op in ordered_ops}
    fusion_ops: list[FusionOp] = []
    for fop, out_id in pending_ops:
        if fop.op == "aux_load":
            fusion_ops.append(fop)
            continue
        recorded = recorded_by_out[out_id]
        op_compute = recorded.compute_dtype if recorded.compute_dtype is not None else compute_dtype
        op_out_dtype = _resolve_out_dtype(out_id, recorded.output_tensor, io_dtype, intermediate_dtype)
        fusion_ops.append(_replace(fop, compute_dtype=op_compute, out_dtype=op_out_dtype))

    quants, quant_recs, quant_dtypes, quant_scale_objs = _collect_quants(
        reachable_quant_ops,
        op_position_by_id,
        gemm_idx_by_output,
        not pending_ops,
        meta,
        io_dtype,
        intermediate_dtype,
        compute_dtype,
        1,
        M,
        N,
        "multi-MoE",
        fto_id=fto_id,
    )

    # Dense outputs in plain recorder (set_output) order — no output position
    # carries semantics or capability; specs[0] merely binds first.
    dense_entries: list[tuple[OutputSpec, Any]] = []
    for tid in set_output_ids_in_order:
        qi = next((i for i, rec in enumerate(quant_recs) if rec.output == tid), None)
        if qi is not None:
            dense_entries.append(
                (
                    OutputSpec(source_ref=quants[qi].source_ref, dtype=quant_dtypes[qi], quant_idx=qi),
                    quant_recs[qi].output_tensor,
                )
            )
        elif tid in op_position_by_id and _TENSOR_OUTPUT_FLAG.get(tid, False):
            pos = op_position_by_id[tid]
            dense_entries.append(
                (
                    OutputSpec(source_ref=pos, dtype=fusion_ops[pos].out_dtype),
                    recorded_by_out[tid].output_tensor,
                )
            )

    def _reduction_output(red: _RecordedOp) -> tuple[tuple[int, int, int], bool]:
        dim = _TENSOR_DIM_OVERRIDE.get(red.output)
        if dim is None:
            try:
                dim = tuple(red.output_tensor.get_dim())
            except Exception:  # noqa: BLE001
                dim = ()
        if len(dim) != 3:
            raise ValueError(f"reduction {red.op_name!r} must set a rank-3 output dim; got {dim}")
        full = (1, int(M), int(N))
        grouped_by_moe = False
        if red.group_offset is not None:
            if red.group_offset != fto_id:
                raise ValueError(f"reduction {red.op_name!r} groupOffset must be the MoE " "first_token_offset tensor")
            if int(dim[0]) != num_groups:
                raise ValueError(f"reduction {red.op_name!r} with groupOffset must use " f"output dim[0] == num_groups ({num_groups}); got {dim}")
            grouped_by_moe = True
        axis0_extent = num_groups if grouped_by_moe else full[0]
        compat_full = (axis0_extent, full[1], full[2])
        for axis, (out_extent, full_extent) in enumerate(zip(dim, compat_full)):
            if out_extent not in (1, full_extent):
                raise ValueError(
                    f"reduction {red.op_name!r} output dim {dim} is not compatible " f"with moe output {full}: axis {axis} must be 1 or {full_extent}"
                )
        if all(out_extent == full_extent for out_extent, full_extent in zip(dim, compat_full)):
            raise ValueError(f"reduction {red.op_name!r} output dim {dim} does not reduce any axis")
        return (int(dim[0]), int(dim[1]), int(dim[2])), grouped_by_moe

    reductions: list[ReductionSpec] = []
    reduction_objs: list[Any] = []
    for red in ops:
        if red.cudnn_name != "reduction":
            continue
        if not _TENSOR_OUTPUT_FLAG.get(red.output, False):
            continue
        (input_id,) = red.inputs
        if input_id in gemm_idx_by_output or input_id in op_position_by_id:
            source_ref = _operand_ref(input_id)
        else:
            raise ValueError(f"reduction {red.op_name!r} input is not produced by this " "multi-MoE epilogue chain")
        compute = red.compute_dtype if red.compute_dtype is not None else compute_dtype
        dtype = _resolve_out_dtype(red.output, red.output_tensor, io_dtype, intermediate_dtype)
        if red.reduction_mode is None:
            raise NotImplementedError(
                f"reduction {red.op_name!r} mode is not supported by cudnn.gemm.frost; "
                "supported modes are ADD, AMAX, MAX, MIN, AVG, MUL, MUL_NO_ZEROS, NORM1, and NORM2"
            )
        red_dim, grouped_by_moe = _reduction_output(red)
        reductions.append(
            ReductionSpec(
                mode=red.reduction_mode,  # type: ignore[arg-type]
                source_ref=source_ref,
                dim=red_dim,
                dtype=dtype,
                compute_dtype=compute,
                grouped_by_moe=grouped_by_moe,
            )
        )
        reduction_objs.append(red.output_tensor)

    matmul_out_dtype = _resolve_out_dtype(moe_ops[0].output, moe_ops[0].output_tensor, io_dtype, intermediate_dtype)
    implicit_raw = False
    if not dense_entries:
        if not pending_ops:
            # Raw MoE output (explicit set_output, or implicit when nothing at
            # all was requested).
            implicit_raw = not _TENSOR_OUTPUT_FLAG.get(moe_ops[0].output, False)
            dense_entries.append(
                (
                    OutputSpec(source_ref=gemm_source(0), dtype=matmul_out_dtype),
                    moe_ops[0].output_tensor,
                )
            )
        elif any(_TENSOR_OUTPUT_FLAG.get(mm.output, False) for mm in moe_ops):
            raise ValueError("materializing the raw MoE grouped matmul output alongside pointwise " "ops is not supported; tap a fusion-op output instead")
        elif not reductions:
            raise ValueError("graph materializes no output; mark at least one tensor " "set_output(True)")

    if dense_entries:
        from dataclasses import replace as _spec_replace

        for _di in range(len(dense_entries)):
            _spec_i, _obj_i = dense_entries[_di]
            _d_i = tuple(_obj_i.get_dim()) if _obj_i is not None else ()
            _s_i = tuple(_obj_i.get_stride()) if _obj_i is not None else ()
            if (not _d_i or not _s_i) and _spec_i.quant_idx is not None:
                _meta_i = meta.get(quant_recs[_spec_i.quant_idx].output)
                if _meta_i is not None:
                    _d_i, _s_i = _meta_i.dim, _meta_i.stride
            # Recorded independently — a derived tensor carries its stride long
            # before cuDNN fills its dim (only at build_operation_graph time).
            _layout = {}
            if _d_i:
                _layout["dim"] = tuple(_d_i)
            if _s_i:
                _layout["stride"] = tuple(_s_i)
            dense_entries[_di] = (_spec_replace(_spec_i, **_layout), _obj_i)
    matmul_spec = MatmulSpec(
        M=M,
        N=N,
        K=K,
        batch=1,
        a_batch=1,
        b_batch=1,
        a_major=a_major,
        b_major=b_major,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        accum_dtype=mm_compute,
        out_dtype=matmul_out_dtype,
    )
    # An implicit (never set_output) raw-MoE output is kept only when nothing
    # else was requested; with reductions present it is dropped (no phantom C).
    if implicit_raw and reductions:
        dense_entries = []
    output_specs: list[OutputSpec] = [spec for spec, _obj in dense_entries]
    output_objs: list[Any] = [obj for _spec, obj in dense_entries]
    chain = FusionChain(
        matmul=matmul_spec,
        aux_tensors=aux_tensors,
        ops=fusion_ops,
        output_specs=output_specs,
        num_a_operands=len(a_ids),
        num_b_operands=len(b_ids),
        gemm_operands=gemm_operands,
        moe=MoeSpec(num_experts=int(E), mode=moe_ops[0].moe_mode, offset_dtype=offset_dtype, num_groups=num_groups),
        block_scale=block_scale_spec,
        reductions=reductions,
        quants=quants,
    )
    output_objs.extend(reduction_objs)
    output_objs.extend(quant_scale_objs)
    binding = _make_multi_binding(
        meta,
        a_ids,
        b_ids,
        a_caps,
        b_caps,
        output_objs,
        aux_objs,
        block_scale_spec is not None,
        first_token_offset=meta[fto_id].tensor,
    )
    return chain, binding


def _walk_mainloop_chain(
    operand_id: int,
    label: str,
    op_by_output: dict[int, _RecordedOp],
    meta: dict[int, _TensorMeta],
    aux_tensors: list[TensorRef],
    aux_objs: list[Any],
    aux_seen: set[int],
) -> tuple[int, list[FusionOp]]:
    """Walk backwards from a matmul operand through pointwise ops (unary, or
    binary with one SCALAR graph-input aux) to the root graph input. Returns
    (root_tensor_id, mainloop_ops) in graph-input -> operand' order; registers
    scalar auxes in aux_tensors."""

    def _is_scalar_input(tid: int) -> bool:
        m = meta.get(tid)
        return m is not None and m.is_input and len(m.dim) > 0 and all(d == 1 for d in m.dim)

    # Each entry: (cudnn_name, is_binary, aux_name_or_None, aux_on_rhs).
    steps: list[tuple] = []
    cur = operand_id
    while cur in op_by_output:
        producer = op_by_output[cur]
        if producer.cudnn_name in _UNARY_OP_MAP:
            if producer.cudnn_name == "gen_index" or producer.op_attrs:
                raise ValueError(
                    f"matmul {label} operand producer {producer.op_name!r}: gen_index and " "pointwise scalar attrs are not supported in mainloop fusion"
                )
            steps.append((producer.cudnn_name, False, None, True))
            (cur,) = producer.inputs
        elif producer.cudnn_name in _BINARY_OP_MAP:
            if producer.op_attrs:
                raise ValueError(f"matmul {label} operand producer {producer.op_name!r}: pointwise scalar " "attrs are not supported in mainloop fusion")
            i0, i1 = producer.inputs
            s0, s1 = _is_scalar_input(i0), _is_scalar_input(i1)
            if s0 and not s1:
                aux_id, chain_id, aux_on_rhs = i0, i1, False
            elif s1 and not s0:
                aux_id, chain_id, aux_on_rhs = i1, i0, True
            else:
                raise ValueError(
                    f"matmul {label} operand is produced by binary op "
                    f"{producer.op_name!r} ({producer.cudnn_name!r}) — mainloop "
                    "fusion only supports a binary op with exactly one SCALAR "
                    "graph-input aux (e.g. A * alpha); per-row/col/elem aux on an "
                    "operand is out of POC scope (needs swizzle-aware indexing)"
                )
            aux_meta = meta[aux_id]
            if aux_id not in aux_seen:
                aux_seen.add(aux_id)
                aux_tensors.append(
                    TensorRef(
                        name=aux_meta.name,
                        dim=aux_meta.dim,
                        stride=aux_meta.stride,
                        dtype=aux_meta.dtype,
                        bcast_mode="scalar",
                    )
                )
                aux_objs.append(aux_meta.tensor)
            steps.append((producer.cudnn_name, True, aux_meta.name, aux_on_rhs))
            cur = chain_id
        else:
            raise ValueError(
                f"matmul {label} operand is produced by op {producer.op_name!r} "
                f"({producer.cudnn_name!r}) which is not a pointwise op — mainloop "
                "fusion supports unary ops and scalar-aux binary ops in the POC"
            )
    steps.reverse()
    fops: list[FusionOp] = []
    for idx, (cudnn_name, is_binary, aux_name, aux_on_rhs) in enumerate(steps):
        parent = idx - 1 if idx > 0 else -1
        if is_binary:
            fops.append(
                FusionOp(
                    op=_BINARY_OP_MAP[cudnn_name],
                    aux=aux_name,
                    aux_on_rhs=aux_on_rhs,
                    parent_idx=parent,
                )
            )
        else:
            fops.append(FusionOp(op=_UNARY_OP_MAP[cudnn_name], parent_idx=parent))
    return cur, fops


def _build_multi_gemm_chain(
    matmuls: list[_RecordedOp],
    ops: list[_RecordedOp],
    meta: dict[int, _TensorMeta],
    io_dtype: Dtype,
    intermediate_dtype: Dtype,
    compute_dtype: Dtype,
) -> FusionChain:
    """Build a FusionChain for K >= 1 parallel GEMMs sharing one pointwise
    epilogue DAG (the unified builder for every plain-matmul graph).

    All GEMMs share shape / layout / dtype but may use shared or distinct A / B
    operands (deduped by tensor id). GEMM outputs are the DAG roots; an op refs a
    GEMM output via a negative ``parent_idx`` (``gemm_source(g)``). Block-scale
    supported (dequant folds into the packed tensor; shared dequant → one operand).
    K == 1 additionally supports: mainloop fusion, the matmul-output tap, the
    raw matmul output / quant source. K > 1 POC scope: no mainloop
    fusion, no per-GEMM taps; outputs must be fusion ops or a
    block_scale_quantize fed by one (no-epilogue K > 1: one output per GEMM)."""
    from .fusion_ir import BlockQuantizeSpec, BlockScaleSpec

    # Resolve each matmul operand through any dequant, then dedup by PACKED data
    # tensor id (shared dequant → one distinct operand), matching the runtime dedup.
    dequant_by_output = {op.output: op for op in ops if op.cudnn_name == "block_scale_dequantize"}

    def _capture_side(operand_id: int) -> dict:
        """Packed data id + block-scale fields for a matmul operand (all None for
        a non-dequantized side)."""
        deq = dequant_by_output.get(operand_id)
        if deq is None:
            return dict(
                data_id=operand_id,
                data_dtype=meta[operand_id].dtype,
                block_size_2d=None,
                sf_dtype=None,
                sf_reorder=None,
                deq_compute=None,
                deq_out=None,
                sf_id=None,
            )
        data_id, sf_id = deq.inputs
        sf_meta = meta[sf_id]
        deq_compute = deq.compute_dtype if deq.compute_dtype is not None else compute_dtype
        deq_out = _resolve_out_dtype(deq.output, deq.output_tensor, io_dtype, intermediate_dtype)
        return dict(
            data_id=data_id,
            data_dtype=meta[data_id].dtype,
            block_size_2d=(tuple(deq.block_size) if deq.block_size else None),
            sf_dtype=sf_meta.dtype,
            sf_reorder=sf_meta.reordering,
            deq_compute=deq_compute,
            deq_out=deq_out,
            sf_id=sf_id,
        )

    aux_tensors: list[TensorRef] = []
    aux_objs: list[Any] = []
    aux_seen: set[int] = set()

    # Mainloop fusion (K == 1, non-block-scale only): walk backwards from each
    # operand through unary / scalar-aux ops to the root graph input; the MMA
    # dtype is resolved from the tensor feeding the matmul, the LOAD dtype from
    # the root. Scalar mainloop auxes register FIRST (ahead of epilogue auxes).
    mainloop_a_ops: list[FusionOp] = []
    mainloop_b_ops: list[FusionOp] = []
    mainloop_a_load_dtype: "Dtype | None" = None
    mainloop_b_load_dtype: "Dtype | None" = None
    mma_dtype_override: "tuple[Dtype, Dtype] | None" = None
    operand_ids_by_mm = {mm.output: (mm.inputs[0], mm.inputs[1]) for mm in matmuls}
    if len(matmuls) == 1 and matmuls[0].inputs[0] not in dequant_by_output and matmuls[0].inputs[1] not in dequant_by_output:
        mm0 = matmuls[0]
        op_by_output = {op.output: op for op in ops if op.cudnn_name != "matmul"}
        A_id, B_id = mm0.inputs
        root_a, mainloop_a_ops = _walk_mainloop_chain(A_id, "A", op_by_output, meta, aux_tensors, aux_objs, aux_seen)
        root_b, mainloop_b_ops = _walk_mainloop_chain(B_id, "B", op_by_output, meta, aux_tensors, aux_objs, aux_seen)

        def _mma_operand_dtype(operand_id: int) -> Dtype:
            om = meta.get(operand_id)
            return _resolve_out_dtype(operand_id, om.tensor if om else None, io_dtype, intermediate_dtype)

        mma_a_dtype = _mma_operand_dtype(A_id)
        mma_b_dtype = _mma_operand_dtype(B_id)
        mainloop_a_load_dtype = meta[root_a].dtype if meta[root_a].dtype != mma_a_dtype else None
        mainloop_b_load_dtype = meta[root_b].dtype if meta[root_b].dtype != mma_b_dtype else None
        mma_dtype_override = (mma_a_dtype, mma_b_dtype)
        operand_ids_by_mm = {mm0.output: (root_a, root_b)}

    a_ids: list[int] = []  # distinct PACKED A data ids
    b_ids: list[int] = []
    a_caps: dict[int, dict] = {}
    b_caps: dict[int, dict] = {}
    gemm_operands: list[tuple[int, int]] = []
    for mm in matmuls:
        mm_a_id, mm_b_id = operand_ids_by_mm[mm.output]
        a_cap = _capture_side(mm_a_id)
        b_cap = _capture_side(mm_b_id)
        a_pid, b_pid = a_cap["data_id"], b_cap["data_id"]
        if a_pid not in a_ids:
            a_ids.append(a_pid)
            a_caps[a_pid] = a_cap
        if b_pid not in b_ids:
            b_ids.append(b_pid)
            b_caps[b_pid] = b_cap
        gemm_operands.append((a_ids.index(a_pid), b_ids.index(b_pid)))

    is_block_scale = any(c["sf_dtype"] is not None for c in (*a_caps.values(), *b_caps.values()))

    # Validate every GEMM shares shape / layout / dtype.
    def _gemm_geometry(a_pid: int, b_pid: int):
        A_meta = meta[a_pid]
        B_meta = meta[b_pid]
        if len(A_meta.dim) != 3 or len(B_meta.dim) != 3:
            raise ValueError(f"matmul operands must be 3D; got A={A_meta.dim} B={B_meta.dim}")
        Ba, M, Ka = A_meta.dim
        Bb, Kb, N = B_meta.dim
        if Ka != Kb:
            raise ValueError(f"K dim mismatch: A={A_meta.dim} B={B_meta.dim}")
        batch = max(Ba, Bb)
        if Ba not in (1, batch) or Bb not in (1, batch):
            raise ValueError(f"batch dims must match or broadcast from 1; got A={A_meta.dim} B={B_meta.dim}")
        return (
            int(M),
            int(N),
            int(Ka),
            int(batch),
            int(Ba),
            int(Bb),
            _infer_a_major(A_meta.dim, A_meta.stride),
            _infer_b_major(B_meta.dim, B_meta.stride),
            A_meta.dtype,
            B_meta.dtype,
        )

    geom0 = _gemm_geometry(a_ids[gemm_operands[0][0]], b_ids[gemm_operands[0][1]])
    for ai, bi in gemm_operands[1:]:
        if _gemm_geometry(a_ids[ai], b_ids[bi]) != geom0:
            raise ValueError("parallel GEMMs must share shape / layout / dtype; multi-GEMM " "with heterogeneous GEMMs is out of POC scope")
    M, N, K, batch, Ba, Bb, a_major, b_major, a_dtype, b_dtype = geom0
    if mma_dtype_override is not None:
        a_dtype, b_dtype = mma_dtype_override

    # Shared BlockScaleSpec (every distinct operand must match GEMM 0's combo).
    block_scale_spec = None
    if is_block_scale:
        a0 = a_caps[a_ids[gemm_operands[0][0]]]
        b0 = b_caps[b_ids[gemm_operands[0][1]]]

        def _combo_key(cap):
            return (
                cap["data_dtype"],
                cap["block_size_2d"],
                cap["sf_dtype"],
                cap["sf_reorder"],
                cap["deq_compute"],
                cap["deq_out"],
            )

        for cap in a_caps.values():
            if _combo_key(cap) != _combo_key(a0):
                raise ValueError("all A operands of a block-scale multi-GEMM must share the same SF combo")
        for cap in b_caps.values():
            if _combo_key(cap) != _combo_key(b0):
                raise ValueError("all B operands of a block-scale multi-GEMM must share the same SF combo")
        block_scale_spec = BlockScaleSpec(
            a_dtype=a0["data_dtype"],
            b_dtype=b0["data_dtype"],
            block_size_a=a0["block_size_2d"],
            block_size_b=b0["block_size_2d"],
            sf_dtype_a=a0["sf_dtype"],
            sf_dtype_b=b0["sf_dtype"],
            sfa_reorder=a0["sf_reorder"],
            sfb_reorder=b0["sf_reorder"],
            dequant_compute_a=a0["deq_compute"],
            dequant_compute_b=b0["deq_compute"],
            dequant_out_a=a0["deq_out"],
            dequant_out_b=b0["deq_out"],
        )
    mm_compute = matmuls[0].compute_dtype if matmuls[0].compute_dtype is not None else compute_dtype
    matmul_out_dim = (batch, M, N)

    # Epilogue DAG over multiple roots (each GEMM output).
    gemm_idx_by_output: dict[int, int] = {mm.output: g for g, mm in enumerate(matmuls)}
    consumers_by_input: dict[int, list[_RecordedOp]] = {}
    for op in ops:
        for inp in op.inputs:
            consumers_by_input.setdefault(inp, []).append(op)

    # Pass 1: reachable op set (BFS from all GEMM outputs).
    reachable_op_ids: set[int] = set()
    bfs_queue: list[int] = [mm.output for mm in matmuls]
    visited_tensors: set[int] = set()
    while bfs_queue:
        tid = bfs_queue.pop(0)
        if tid in visited_tensors:
            continue
        visited_tensors.add(tid)
        for op in consumers_by_input.get(tid, []):
            if op.cudnn_name == "matmul":
                continue
            if op.output not in reachable_op_ids:
                reachable_op_ids.add(op.output)
                bfs_queue.append(op.output)
    pointwise_producer = {op.output: op for op in ops if op.cudnn_name in _UNARY_OP_MAP or op.cudnn_name in _BINARY_OP_MAP or op.cudnn_name == "binary_select"}
    demand: list[int] = [tid for tid, flagged in _TENSOR_OUTPUT_FLAG.items() if flagged]
    for op in ops:
        if op.output in reachable_op_ids:
            demand.extend(op.inputs)
        elif op.cudnn_name in ("reduction", "block_scale_quantize") and _TENSOR_OUTPUT_FLAG.get(op.output, False):
            demand.extend(op.inputs)
    while demand:
        tid = demand.pop()
        prod = pointwise_producer.get(tid)
        if prod is not None and prod.output not in reachable_op_ids:
            reachable_op_ids.add(prod.output)
            demand.extend(prod.inputs)
    reachable_ops = [op for op in ops if op.output in reachable_op_ids and op.cudnn_name not in {"matmul", "reduction", "block_scale_quantize"}]
    reachable_quant_ops = [
        op for op in ops if op.cudnn_name == "block_scale_quantize" and (op.output in reachable_op_ids or any(i in reachable_op_ids for i in op.inputs))
    ]

    def _is_in_chain(tid: int) -> bool:
        return tid in gemm_idx_by_output or tid in reachable_op_ids

    in_chain_deps: dict[int, list[int]] = {op.output: [inp for inp in op.inputs if _is_in_chain(inp)] for op in reachable_ops}

    # Pass 2: Kahn topo sort (placed seeded with all GEMM outputs).
    placed: set[int] = set(gemm_idx_by_output)
    remaining = list(reachable_ops)
    ordered_ops: list[_RecordedOp] = []
    while remaining:
        ready_idx = next(
            (i for i, op in enumerate(remaining) if all(d in placed for d in in_chain_deps[op.output])),
            None,
        )
        if ready_idx is None:
            raise AssertionError(f"cycle / unsatisfiable deps: {[op.op_name for op in remaining]}")
        op = remaining.pop(ready_idx)
        ordered_ops.append(op)
        placed.add(op.output)

    op_position_by_id: dict[int, int] = {}
    pending_ops: list[tuple[FusionOp, int]] = []

    def _register_aux(aux_id: int, op_name: str) -> str:
        aux_meta = meta[aux_id]
        if not aux_meta.is_input:
            raise ValueError(f"aux input {aux_meta.name!r} of op {op_name!r} is " "not a graph input — POC supports only graph-input aux")
        if aux_id not in aux_seen:
            aux_seen.add(aux_id)
            bcast = _infer_bcast_mode(matmul_out_dim, aux_meta.dim)
            aux_tensors.append(
                TensorRef(
                    name=aux_meta.name,
                    dim=aux_meta.dim,
                    stride=aux_meta.stride,
                    dtype=aux_meta.dtype,
                    bcast_mode=bcast,
                )
            )
            aux_objs.append(aux_meta.tensor)
        return aux_meta.name

    def _aux_root_ref(tid: int, op_name: str) -> int:
        if tid in op_position_by_id:
            return op_position_by_id[tid]
        name = _register_aux(tid, op_name)
        pending_ops.append((FusionOp(op="aux_load", aux=name), tid))
        op_position_by_id[tid] = len(pending_ops) - 1
        return op_position_by_id[tid]

    def _operand_ref(tid: int, op_name: str = "") -> int:
        """In-chain operand id → producing-op ref: ``gemm_source(g)`` (<0) for a
        GEMM output, an op index, or a synthesized aux_load root for a graph input."""
        if tid in gemm_idx_by_output:
            return gemm_source(gemm_idx_by_output[tid])
        if tid in op_position_by_id:
            return op_position_by_id[tid]
        m = meta.get(tid)
        if m is not None and m.is_input:
            return _aux_root_ref(tid, op_name)
        raise ValueError(f"op {op_name!r} input is not produced by this epilogue chain and is " "not a graph input")

    for next_op in ordered_ops:
        if next_op.cudnn_name in _UNARY_OP_MAP:
            (parent_id,) = next_op.inputs
            if next_op.cudnn_name == "gen_index" and dict(next_op.op_attrs).get("axis") not in (1, 2):
                raise NotImplementedError(f"gen_index {next_op.op_name!r}: only axis 1 (M) or 2 (N) is supported " "in cudnn.gemm.frost")
            fop = FusionOp(op=_UNARY_OP_MAP[next_op.cudnn_name], parent_idx=_operand_ref(parent_id, next_op.op_name), attrs=next_op.op_attrs)
        elif next_op.cudnn_name in _BINARY_OP_MAP:
            inp0, inp1 = next_op.inputs
            in0, in1 = _is_in_chain(inp0), _is_in_chain(inp1)
            if in0 and in1:
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=None,
                    aux_on_rhs=True,
                    parent_idx=_operand_ref(inp0),
                    parent_idx_b=_operand_ref(inp1),
                    attrs=next_op.op_attrs,
                )
            elif in0 or in1:
                if in0:
                    chain_id, aux_id, aux_on_rhs = inp0, inp1, True
                else:
                    chain_id, aux_id, aux_on_rhs = inp1, inp0, False
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=_register_aux(aux_id, next_op.op_name),
                    aux_on_rhs=aux_on_rhs,
                    parent_idx=_operand_ref(chain_id, next_op.op_name),
                    attrs=next_op.op_attrs,
                )
            else:
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=_register_aux(inp1, next_op.op_name),
                    aux_on_rhs=True,
                    parent_idx=_operand_ref(inp0, next_op.op_name),
                    attrs=next_op.op_attrs,
                )
        elif next_op.cudnn_name == "binary_select":
            i0, i1, im = next_op.inputs
            fop = FusionOp(
                op="binary_select",
                parent_idx=_operand_ref(i0, next_op.op_name),
                parent_idx_b=_operand_ref(i1, next_op.op_name),
                parent_idx_c=_operand_ref(im, next_op.op_name),
            )
        else:
            raise ValueError(f"op {next_op.cudnn_name!r} (name={next_op.op_name!r}) is not in " "the POC pointwise subset; out-of-scope")
        pending_ops.append((fop, next_op.output))
        op_position_by_id[next_op.output] = len(pending_ops) - 1

    # No fusion epilogue: each GEMM output materializes directly to its own GMEM
    # buffer. Each must set_output(True); outputs bind in GEMM order.
    # (K == 1 falls through: the matmul output binds like any chain output.)
    if not pending_ops and len(matmuls) > 1:
        per_gemm_dtypes: list[Dtype] = []
        for mm in matmuls:
            if not _TENSOR_OUTPUT_FLAG.get(mm.output, False):
                raise ValueError("no-epilogue multi-GEMM: every GEMM output must be " "set_output(True) (no fusion op materializes it)")
            per_gemm_dtypes.append(_resolve_out_dtype(mm.output, mm.output_tensor, io_dtype, intermediate_dtype))
        matmul_spec = MatmulSpec(
            M=M,
            N=N,
            K=K,
            batch=batch,
            a_batch=Ba,
            b_batch=Bb,
            a_major=a_major,
            b_major=b_major,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=mm_compute,
            out_dtype=per_gemm_dtypes[0],
        )
        chain = FusionChain(
            matmul=matmul_spec,
            aux_tensors=aux_tensors,
            ops=[],
            output_specs=[OutputSpec(source_ref=gemm_source(g), dtype=dt) for g, dt in enumerate(per_gemm_dtypes)],
            num_a_operands=len(a_ids),
            num_b_operands=len(b_ids),
            gemm_operands=gemm_operands,
            block_scale=block_scale_spec,
        )
        # No-epilogue outputs = each GEMM's own buffer, in GEMM order.
        binding = _make_multi_binding(
            meta,
            a_ids,
            b_ids,
            a_caps,
            b_caps,
            [mm.output_tensor for mm in matmuls],
            aux_objs,
            block_scale_spec is not None,
        )
        return chain, binding

    set_output_ids_in_order = [tid for tid in _TENSOR_OUTPUT_FLAG if _TENSOR_OUTPUT_FLAG[tid]]

    from dataclasses import replace as _replace

    recorded_by_out = {op.output: op for op in ordered_ops}
    fusion_ops: list[FusionOp] = []
    for fop, out_id in pending_ops:
        if fop.op == "aux_load":
            fusion_ops.append(fop)
            continue
        recorded = recorded_by_out[out_id]
        op_compute = recorded.compute_dtype if recorded.compute_dtype is not None else compute_dtype
        op_out_dtype = _resolve_out_dtype(out_id, recorded.output_tensor, io_dtype, intermediate_dtype)
        fusion_ops.append(_replace(fop, compute_dtype=op_compute, out_dtype=op_out_dtype))

    quants, quant_recs, quant_dtypes, quant_scale_objs = _collect_quants(
        reachable_quant_ops,
        op_position_by_id,
        gemm_idx_by_output,
        len(matmuls) == 1,
        meta,
        io_dtype,
        intermediate_dtype,
        compute_dtype,
        batch,
        M,
        N,
        "multi-GEMM",
    )

    # GEMM output declared dtype — the epilogue rounds each accumulator to it
    # before the op chain. All GEMMs share it; resolve from GEMM 0.
    matmul_out_dtype = _resolve_out_dtype(matmuls[0].output, matmuls[0].output_tensor, io_dtype, intermediate_dtype)

    # Dense outputs in plain recorder (set_output) order — no output position
    # carries semantics or capability; specs[0] merely binds first. The raw
    # matmul output participates for K == 1 (per-GEMM raw taps stay out of
    # scope for K > 1).
    dense_entries: list[tuple[OutputSpec, Any]] = []
    for tid in set_output_ids_in_order:
        qi = next((i for i, rec in enumerate(quant_recs) if rec.output == tid), None)
        if qi is not None:
            dense_entries.append(
                (
                    OutputSpec(source_ref=quants[qi].source_ref, dtype=quant_dtypes[qi], quant_idx=qi),
                    quant_recs[qi].output_tensor,
                )
            )
        elif tid in op_position_by_id and _TENSOR_OUTPUT_FLAG.get(tid, False):
            pos = op_position_by_id[tid]
            dense_entries.append(
                (
                    OutputSpec(source_ref=pos, dtype=fusion_ops[pos].out_dtype),
                    recorded_by_out[tid].output_tensor,
                )
            )
        elif len(matmuls) == 1 and tid == matmuls[0].output and _TENSOR_OUTPUT_FLAG.get(tid, False):
            dense_entries.append(
                (
                    OutputSpec(source_ref=gemm_source(0), dtype=matmul_out_dtype),
                    matmuls[0].output_tensor,
                )
            )

    def _reduction_output_dim(red: _RecordedOp) -> tuple[int, int, int]:
        dim = _TENSOR_DIM_OVERRIDE.get(red.output)
        if dim is None:
            try:
                dim = tuple(red.output_tensor.get_dim())
            except Exception:  # noqa: BLE001
                dim = ()
        if len(dim) != 3:
            raise ValueError(f"reduction {red.op_name!r} must set a rank-3 output dim; got {dim}")
        full = (int(batch), int(M), int(N))
        for axis, (out_extent, full_extent) in enumerate(zip(dim, full)):
            if out_extent not in (1, full_extent):
                raise ValueError(
                    f"reduction {red.op_name!r} output dim {dim} is not compatible " f"with matmul output {full}: axis {axis} must be 1 or {full_extent}"
                )
        if all(out_extent == full_extent for out_extent, full_extent in zip(dim, full)):
            raise ValueError(f"reduction {red.op_name!r} output dim {dim} does not reduce any axis")
        return (int(dim[0]), int(dim[1]), int(dim[2]))

    reductions: list[ReductionSpec] = []
    reduction_objs: list[Any] = []
    for red in ops:
        if red.cudnn_name != "reduction":
            continue
        if not _TENSOR_OUTPUT_FLAG.get(red.output, False):
            continue
        (input_id,) = red.inputs
        if input_id in gemm_idx_by_output or input_id in op_position_by_id:
            source_ref = _operand_ref(input_id)
        else:
            raise ValueError(f"reduction {red.op_name!r} input is not produced by this " "multi-GEMM epilogue chain")
        compute = red.compute_dtype if red.compute_dtype is not None else compute_dtype
        dtype = _resolve_out_dtype(red.output, red.output_tensor, io_dtype, intermediate_dtype)
        if red.reduction_mode is None:
            raise NotImplementedError(
                f"reduction {red.op_name!r} mode is not supported by cudnn.gemm.frost; "
                "supported modes are ADD, AMAX, MAX, MIN, AVG, MUL, MUL_NO_ZEROS, NORM1, and NORM2"
            )
        reductions.append(
            ReductionSpec(
                mode=red.reduction_mode,  # type: ignore[arg-type]
                source_ref=source_ref,
                dim=_reduction_output_dim(red),
                dtype=dtype,
                compute_dtype=compute,
            )
        )
        reduction_objs.append(red.output_tensor)

    implicit_raw = False
    if not dense_entries:
        if len(matmuls) == 1:
            implicit_raw = not _TENSOR_OUTPUT_FLAG.get(matmuls[0].output, False)
            dense_entries.append(
                (
                    OutputSpec(source_ref=gemm_source(0), dtype=matmul_out_dtype),
                    matmuls[0].output_tensor,
                )
            )
        elif not reductions:
            raise ValueError("graph materializes no output; mark at least one tensor " "set_output(True)")

    if dense_entries:
        from dataclasses import replace as _spec_replace

        for _di in range(len(dense_entries)):
            _spec_i, _obj_i = dense_entries[_di]
            _d_i = tuple(_obj_i.get_dim()) if _obj_i is not None else ()
            _s_i = tuple(_obj_i.get_stride()) if _obj_i is not None else ()
            if (not _d_i or not _s_i) and _spec_i.quant_idx is not None:
                _meta_i = meta.get(quant_recs[_spec_i.quant_idx].output)
                if _meta_i is not None:
                    _d_i, _s_i = _meta_i.dim, _meta_i.stride
            # Recorded independently — a derived tensor carries its stride long
            # before cuDNN fills its dim (only at build_operation_graph time).
            _layout = {}
            if _d_i:
                _layout["dim"] = tuple(_d_i)
            if _s_i:
                _layout["stride"] = tuple(_s_i)
            dense_entries[_di] = (_spec_replace(_spec_i, **_layout), _obj_i)
    matmul_spec = MatmulSpec(
        M=M,
        N=N,
        K=K,
        batch=batch,
        a_batch=Ba,
        b_batch=Bb,
        a_major=a_major,
        b_major=b_major,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        accum_dtype=mm_compute,
        out_dtype=matmul_out_dtype,
    )
    # An implicit (never set_output) raw-matmul output is kept only when
    # nothing else was requested; with reductions present it is dropped (no
    # phantom C).
    if implicit_raw and reductions:
        dense_entries = []
    output_specs: list[OutputSpec] = [spec for spec, _obj in dense_entries]
    output_objs: list[Any] = [obj for _spec, obj in dense_entries]
    chain = FusionChain(
        matmul=matmul_spec,
        aux_tensors=aux_tensors,
        ops=fusion_ops,
        output_specs=output_specs,
        num_a_operands=len(a_ids),
        num_b_operands=len(b_ids),
        gemm_operands=gemm_operands,
        mainloop_a_ops=mainloop_a_ops,
        mainloop_b_ops=mainloop_b_ops,
        mainloop_a_load_dtype=mainloop_a_load_dtype,
        mainloop_b_load_dtype=mainloop_b_load_dtype,
        block_scale=block_scale_spec,
        reductions=reductions,
        quants=quants,
    )
    output_objs.extend(reduction_objs)
    output_objs.extend(quant_scale_objs)
    binding = _make_multi_binding(
        meta,
        a_ids,
        b_ids,
        a_caps,
        b_caps,
        output_objs,
        aux_objs,
        block_scale_spec is not None,
    )
    return chain, binding


def _build_chain(
    ops: list[_RecordedOp],
    meta: dict[int, _TensorMeta],
    io_dtype: Dtype,
    intermediate_dtype: Dtype = "fp32",
    compute_dtype: Dtype = "fp32",
) -> FusionChain:
    # MoE grouped matmul (own graph type): K >= 1 grouped matmuls sharing one
    # fto + one epilogue DAG (K == 1 with or without epilogue degenerates).
    moe_ops = [op for op in ops if op.cudnn_name == "moe_grouped_matmul"]
    if moe_ops:
        return _build_multi_moe_chain(moe_ops, ops, meta, io_dtype, intermediate_dtype, compute_dtype)

    matmuls = [op for op in ops if op.cudnn_name == "matmul"]
    if len(matmuls) == 0:
        raise ValueError("POC scope is >=1 matmul per graph; found 0")
    # K >= 1 parallel GEMMs sharing one epilogue DAG (K == 1 degenerates to the
    # single-GEMM kernel: matmul tap / mainloop fusion / TMA-store all apply).
    return _build_multi_gemm_chain(matmuls, ops, meta, io_dtype, intermediate_dtype, compute_dtype)


def analyze_with_binding(
    graph: cudnn.pygraph,
) -> "tuple[FusionChain, GemmBinding | None]":
    """Build the FusionChain AND a variant-pack binding (role -> cuDNN tensor).
    See :class:`GemmBinding`."""
    with _ANALYZE_LOCK:
        state = _state_from_graph(graph)
        if not state["ops"]:
            raise ValueError("graph has no ops; nothing to compile")
        return _build_chain(
            state["ops"],
            state["tensor_meta"],
            state["io_dtype"],
            state["intermediate_dtype"],
            state["compute_dtype"],
        )


def analyze(graph: cudnn.pygraph) -> FusionChain:
    """Build a FusionChain from a cudnn.pygraph constructed AFTER cudnn.gemm.frost import."""
    chain, _ = analyze_with_binding(graph)
    return chain
