# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Engine-agnostic norm graph analysis: ``graph.nodes`` -> :class:`NormGraphFacts`.

Mirrors ``cudnn.sdpa.graph_analyzer``: extract *facts* (what the graph asks for)
from a single norm node without judging supportedness; each registered engine
matches the facts against its own :class:`Capabilities` row (see
``cudnn.norm.fprop.engines`` / ``cudnn.norm.bprop.engines``). Shared by both
passes — the facts carry ``phase`` ("fprop"/"bprop"). The parse is cached per
graph.

Also hosts the graph-side runtime helpers every norm engine shares: variant-pack
resolution and TensorDesc construction.

cuDNN-node contract (validate against a built cuDNN frontend): a norm forward
node exposes ``X`` and ``scale`` (gamma), optional ``bias`` (beta), scalar
``epsilon``/``momentum`` params and a ``norm_mode``/phase; outputs ``Y`` and
(training) ``mean``/``inv_variance``. A backward node additionally exposes
``DY`` and produces ``DX``/``Dscale``/``Dbias``. Port/param spellings are
resolved defensively below.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass, replace
from typing import Any, Optional

import cudnn
import torch

from .config_sm100 import HAS_MEAN, NormVariant

_LOG = logging.getLogger(__name__)

_DTYPE_FROM_CUDNN = {
    cudnn.data_type.HALF: torch.float16,
    cudnn.data_type.BFLOAT16: torch.bfloat16,
    cudnn.data_type.FLOAT: torch.float32,
}

# cuDNN norm-mode / node-type spelling -> internal NormVariant. Matched by the
# upper-cased name so it tolerates either a generic NORM node carrying a
# ``norm_mode`` param or dedicated per-flavor node types.
_VARIANT_FROM_NAME = {
    "LAYER_NORM": NormVariant.LAYER_NORM,
    "LAYERNORM": NormVariant.LAYER_NORM,
    "RMS_NORM": NormVariant.RMS_NORM,
    "RMSNORM": NormVariant.RMS_NORM,
    "GROUP_NORM": NormVariant.GROUP_NORM,
    "GROUPNORM": NormVariant.GROUP_NORM,
    "INSTANCE_NORM": NormVariant.INSTANCE_NORM,
    "INSTANCENORM": NormVariant.INSTANCE_NORM,
    "BATCH_NORM": NormVariant.BATCH_NORM,
    "BATCHNORM": NormVariant.BATCH_NORM,
}

# Node types that denote a backward (gradient) norm op, by name substring.
_BWD_MARKERS = ("BACKWARD", "BPROP", "BWD", "GRAD")


def _device_cc() -> Optional[tuple]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(torch.cuda.current_device())


def _stride_order(dim: tuple, stride: tuple) -> tuple:
    return tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], dim[x[0]])))


@dataclass(frozen=True)
class NormGraphFacts:
    """What a single-norm graph asks for. Pure description — no support judgment.

    ``invalid`` is the one exception: a graph-consistency error (malformed
    regardless of which kernel would run); when set, every engine is ineligible.
    """

    invalid: Optional[str] = None

    variant: Optional[NormVariant] = None
    phase: str = "fprop"  # "fprop" | "bprop"

    # layout
    x_shape: tuple = ()
    normalized_shape: Optional[tuple] = None  # LayerNorm/RMSNorm
    num_groups: Optional[int] = None  # GroupNorm
    dtype: Optional[torch.dtype] = None  # X dtype
    compute_dtype: Any = None
    uniform_dtype: bool = True  # scale/bias (and DY) share X's dtype; stats are always fp32

    # requested features / scalars
    has_beta: bool = False
    has_mean: bool = True
    training: bool = True
    eps: float = 1e-5
    eps_t: Any = None
    momentum: float = 0.1
    momentum_t: Any = None
    wants_running_stats: bool = False
    adaptive: bool = False
    affine_batches: int = 1
    affine_row_divisor: int = 1
    zero_centered_gamma: bool = False
    scale_add_dtype: Any = None
    scale_one_t: Any = None

    @property
    def extended_affine(self):
        return self.adaptive or self.zero_centered_gamma

    device_cc: Optional[tuple] = None

    # IR tensor refs for binding
    x_t: Any = None
    scale_t: Any = None  # gamma
    bias_t: Any = None  # beta
    y_t: Any = None
    mean_t: Any = None
    inv_var_t: Any = None
    run_mean_t: Any = None  # next_running_mean OUTPUT
    run_var_t: Any = None  # next_running_var OUTPUT
    prev_run_mean_t: Any = None  # in_running_mean INPUT -- a DIFFERENT buffer
    prev_run_var_t: Any = None
    # backward
    dy_t: Any = None
    dx_t: Any = None
    dscale_t: Any = None
    dbias_t: Any = None


def _norm_node(graph: "cudnn.pygraph") -> Optional[Any]:
    """The graph's sole norm node, or None if the graph is anything else."""
    try:
        nodes = graph.nodes
    except Exception:  # noqa: BLE001 — non-IR graph objects
        return None
    norms = [node for node in nodes if _node_variant(node) is not None]
    if len(norms) != 1 or len(nodes) > 2:
        return None
    return norms[0]


def _node_type_name(node: Any) -> str:
    nt = getattr(node, "node_type", None)
    return getattr(nt, "name", str(nt)).upper()


def _node_variant(node: Any) -> Optional[NormVariant]:
    """Map a norm node to a :class:`NormVariant` via its ``norm_mode`` param or
    node-type name; None if the node is not a norm op."""
    params = dict(getattr(node, "params", {}) or {})
    mode = params.get("norm_mode") or params.get("mode")
    if mode is not None:
        name = getattr(mode, "name", str(mode)).upper()
        if name in _VARIANT_FROM_NAME:
            return _VARIANT_FROM_NAME[name]
    name = _node_type_name(node)
    for key, variant in _VARIANT_FROM_NAME.items():
        if key in name:
            return variant
    return None


def _node_phase(node: Any) -> str:
    name = _node_type_name(node)
    if any(m in name for m in _BWD_MARKERS):
        return "bprop"
    params = dict(getattr(node, "params", {}) or {})
    phase = params.get("norm_forward_phase") or params.get("phase")
    pname = getattr(phase, "name", str(phase)).upper() if phase is not None else ""
    if any(m in pname for m in _BWD_MARKERS):
        return "bprop"
    return "fprop"


def _port(node_ports: dict, *candidates: str) -> Any:
    for c in candidates:
        if c in node_ports:
            return node_ports[c]
    # case-insensitive fallback
    lower = {k.lower(): v for k, v in node_ports.items()}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _invalid(reason: str) -> NormGraphFacts:
    return NormGraphFacts(invalid=f"cudnn.norm: {reason}")


def _extract_facts(node: Any) -> NormGraphFacts:
    variant = _node_variant(node)
    if variant is None:
        return _invalid("node is not a recognized norm op")
    phase = _node_phase(node)
    params = dict(getattr(node, "params", {}) or {})
    inputs = dict(getattr(node, "inputs", {}) or {})
    outputs = dict(getattr(node, "outputs", {}) or {})

    x = _port(inputs, "X", "input", "x")
    scale = _port(inputs, "scale", "gamma", "weight")
    bias = _port(inputs, "bias", "beta")
    if x is None:
        return _invalid("missing X input on the norm node")

    x_shape = tuple(x.get_dim())
    x_dtype = _DTYPE_FROM_CUDNN.get(x.get_data_type())
    if x_dtype is None:
        return _invalid(f"X dtype {x.get_data_type()} not in {{fp16, bf16, fp32}}")

    num_groups = params.get("num_groups") or params.get("group_count")
    if variant == NormVariant.GROUP_NORM and num_groups is None:
        return _invalid("GroupNorm node missing num_groups")

    adaptive = "ADA" in _node_type_name(node) or bool(params.get("adaptive", False))
    affine_batches, affine_row_divisor = 1, 1
    if adaptive:
        if scale is None:
            return _invalid("adaptive normalization requires scale")
        ss = tuple(scale.get_dim())
        if variant == NormVariant.INSTANCE_NORM:
            if len(x_shape) not in (3, 4, 5):
                return _invalid("adaptive InstanceNorm requires NCL, NCHW, or NCDHW input")
            if ss != (*x_shape[:2], *((1,) * (len(x_shape) - 2))):
                return _invalid("adaptive InstanceNorm scale must be per sample/channel with singleton spatial dimensions")
            affine_batches = x_shape[0] * x_shape[1]
        elif variant in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM):
            if len(x_shape) != 3:
                return _invalid("adaptive LayerNorm/RMSNorm requires rank-3 X")
            a, b, d = x_shape
            if ss == (a, 1, d):
                affine_batches, affine_row_divisor = a, b
            elif ss == (1, b, d):
                affine_batches, affine_row_divisor = b, 1
            else:
                return _invalid("adaptive scale must be [B,1,D] or [1,B,D] matching X")
        else:
            return _invalid(f"adaptive affine is not supported for {variant.value}")
        if bias is not None and tuple(bias.get_dim()) != ss:
            return _invalid("adaptive bias must have the scale shape")

    # LayerNorm/RMSNorm normalize over scale's trailing feature dimensions.
    normalized_shape = None
    if variant in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM) and scale is not None:
        sshape = tuple(d for d in tuple(scale.get_dim()) if d != 1)
        normalized_shape = x_shape[-1:] if adaptive else (sshape or x_shape[-1:])
    elif variant == NormVariant.INSTANCE_NORM and adaptive:
        normalized_shape = x_shape[2:]

    # epsilon is an input TENSOR on a cuDNN graph, not a node param: reading it from
    # params alone silently yields the 1e-5 default for every graph that passes a
    # different value, and the normalization is then quietly wrong everywhere
    # (eps=1e-3 showed up as a ~0.5% error across 89% of elements). The tensor is
    # carried through to the lowering, which reads its value at execute time.
    eps_t = _port(inputs, "epsilon", "EPSILON", "eps")
    eps = params.get("epsilon")
    eps = float(eps) if eps is not None else 1e-5
    # momentum is an input TENSOR too, and reading it from params alone pins every
    # graph to the 0.1 default -- the running statistics then track at the wrong
    # rate with nothing to show for it. Same treatment as epsilon above.
    momentum_t = _port(inputs, "momentum", "MOMENTUM", "exp_avg_factor")
    momentum = params.get("momentum")
    momentum = float(momentum) if momentum is not None else 0.1

    dy = _port(inputs, "grad", "DY", "dy") if phase == "bprop" else None
    if phase == "bprop" and dy is None:
        return _invalid("backward norm node missing grad (DY) input")
    # "uniform" must cover every operand the kernels read at the I/O dtype, not just
    # DY. cuDNN lets a caller pass fp32 scale/bias alongside a bf16 input -- the torch
    # custom ops do exactly that -- and these kernels read gamma/beta AS the I/O dtype,
    # so an fp32 buffer read as bf16 is garbage (measured 14.6 absolute error, 91% of
    # elements). mean/inv_var are excluded: they are fp32 by definition.
    uniform = all(_DTYPE_FROM_CUDNN.get(t.get_data_type()) == x_dtype for t in (scale, bias) if t is not None)
    if dy is not None:
        uniform = uniform and _DTYPE_FROM_CUDNN.get(dy.get_data_type()) == x_dtype

    # mean / inv_variance are node OUTPUTS in forward (training) and node INPUTS
    # in backward. cuDNN spells the fwd output ``inv_var`` and the bwd input
    # ``inv_variance``; the DScale/DBias grads are ``DScale``/``DBias`` outputs.
    mean_t = _port(outputs, "mean") or _port(inputs, "mean")
    inv_var_t = _port(outputs, "inv_var", "inv_variance", "rstd") or _port(inputs, "inv_variance", "inv_var")

    # Inference is signalled by a BATCHNORM_INFERENCE node or norm_forward_phase.
    fphase = params.get("norm_forward_phase") or params.get("phase")
    fphase_name = getattr(fphase, "name", str(fphase)).upper() if fphase is not None else ""
    is_inference = "INFERENCE" in _node_type_name(node) or "INFERENCE" in fphase_name or bool(params.get("is_inference", False))
    training = phase == "fprop" and not is_inference
    run_mean = _port(inputs, "in_running_mean", "prev_running_mean", "running_mean")
    run_var = _port(inputs, "in_running_var", "prev_running_var", "running_var")

    # A backward graph has no bias INPUT -- beta only shows up as the DBias
    # OUTPUT -- so deriving has_beta from the input alone reports False for every
    # bprop graph. The lowering passes has_beta straight to norm_bprop, which then
    # returns dbeta=None and the DBias buffer is never written: silent, because the
    # engine simply skips a copy it thinks it has nothing for.
    dbias_out = _port(outputs, "DBias", "Dbias", "dbias", "dbeta")

    return NormGraphFacts(
        variant=variant,
        phase=phase,
        adaptive=adaptive,
        affine_batches=affine_batches,
        affine_row_divisor=affine_row_divisor,
        x_shape=x_shape,
        normalized_shape=normalized_shape,
        num_groups=int(num_groups) if num_groups is not None else None,
        dtype=x_dtype,
        compute_dtype=getattr(node, "compute_data_type", None),
        uniform_dtype=uniform,
        has_beta=(bias is not None) if phase == "fprop" else (dbias_out is not None),
        has_mean=HAS_MEAN[variant],
        training=training,
        eps=eps,
        eps_t=eps_t,
        momentum=momentum,
        momentum_t=momentum_t,
        wants_running_stats=(run_mean is not None and run_var is not None),
        device_cc=_device_cc(),
        x_t=x,
        scale_t=scale,
        bias_t=bias,
        y_t=_port(outputs, "Y", "output", "y"),
        mean_t=mean_t,
        inv_var_t=inv_var_t,
        run_mean_t=_port(outputs, "next_running_mean", "running_mean"),
        run_var_t=_port(outputs, "next_running_var", "running_var"),
        prev_run_mean_t=run_mean,
        prev_run_var_t=run_var,
        dy_t=dy,
        dx_t=_port(outputs, "DX", "dx", "grad_input"),
        dscale_t=_port(outputs, "DScale", "Dscale", "dscale", "dgamma"),
        dbias_t=dbias_out,
    )


def _analyze(graph: "cudnn.pygraph") -> Optional[NormGraphFacts]:
    """Describe a norm and its optional scalar-add scale prologue.

    Mutable graphs are analyzed afresh. The graph lifecycle caches facts after
    freezing the declaration, alongside the other engine families.
    """
    node = _norm_node(graph)
    if node is None:
        return None
    facts = _extract_facts(node)
    if facts.invalid or len(graph.nodes) == 1:
        return facts
    add = next(n for n in graph.nodes if n is not node)
    mode = getattr(add, "params", {}).get("mode")
    if _node_type_name(add) != "POINTWISE" or str(getattr(mode, "name", mode)).lower() != "add":
        return None
    if facts.variant not in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM) and not (facts.variant == NormVariant.INSTANCE_NORM and facts.adaptive):
        return None
    if getattr(add, "compute_data_type", None) not in (None, cudnn.data_type.FLOAT):
        return None
    outputs = list(add.outputs.values())
    if len(outputs) != 1 or outputs[0] is not facts.scale_t or not outputs[0].get_is_virtual():
        return None
    operands = list(add.inputs.values())
    if len(operands) != 2:
        return None
    for gamma, one in (operands, operands[::-1]):
        if tuple(gamma.get_dim()) != tuple(facts.scale_t.get_dim()):
            continue
        if not one.get_is_pass_by_value() or any(d != 1 for d in one.get_dim()):
            continue
        value = getattr(one, "pass_by_value", None)
        if value is not None and value != 1.0:
            continue
        return replace(facts, scale_t=gamma, scale_one_t=one, zero_centered_gamma=True, scale_add_dtype=outputs[0].get_data_type())
    return None


_FACTS_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def analyze(graph: "cudnn.pygraph") -> Optional[NormGraphFacts]:
    if not getattr(graph, "_frozen", False):
        return _analyze(graph)
    if graph not in _FACTS_CACHE:
        _FACTS_CACHE[graph] = _analyze(graph)
    return _FACTS_CACHE[graph]


# ---------------------------------------------------------------------------
# Graph-side runtime helpers shared by norm engines
# ---------------------------------------------------------------------------


@dataclass
class NormBinding:
    tensors: dict  # role -> IR tensor

    def bound_tensors(self) -> list:
        return [t for t in self.tensors.values() if t is not None]


def _safe_name(t: Any) -> Optional[str]:
    try:
        nm = t.get_name()
    except Exception:  # noqa: BLE001
        return None
    return nm or None


def _safe_uid(t: Any) -> Optional[int]:
    try:
        uid = t.get_uid()
    except Exception:  # noqa: BLE001
        return None
    return uid if isinstance(uid, int) and uid > 0 else None


def resolve_variant_pack(variant_pack: dict, binding: NormBinding) -> dict:
    """Map a cuDNN variant-pack dict (keyed by IR tensor / uid / name) to buffers
    keyed by ``id(ir_tensor)``. Mirrors ``cudnn.sdpa.graph_analyzer``."""
    if not isinstance(variant_pack, dict):
        raise TypeError(
            "cudnn.norm: compiled plans are called with a variant-pack dict " f"{{cudnn_tensor | uid | name: buffer}}; got {type(variant_pack).__name__}"
        )
    bound = binding.bound_tensors()
    by_obj = {id(t): t for t in bound}

    name_counts: dict = {}
    uid_counts: dict = {}
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

    resolved: dict = {}
    for key, buf in variant_pack.items():
        if id(key) in by_obj:
            t = by_obj[id(key)]
        elif isinstance(key, int) and key in by_uid:
            t = by_uid[key]
        elif isinstance(key, str) and key in by_name:
            t = by_name[key]
        else:
            continue
        resolved[id(t)] = buf
    return resolved


def tensor_desc_from_ir(t: Any, name: str = "") -> "TensorDesc":
    from cudnn.api_base import TensorDesc

    shape = tuple(t.get_dim())
    stride = tuple(t.get_stride())
    dtype = _DTYPE_FROM_CUDNN[t.get_data_type()]
    return TensorDesc(
        dtype=dtype,
        shape=shape,
        stride=stride,
        stride_order=_stride_order(shape, stride),
        device=torch.device("cuda", torch.cuda.current_device()),
        name=name,
    )
