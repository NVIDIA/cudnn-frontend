# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental prepared norm -> pointwise graph lowering.

Consumes existing public graph nodes, without a new fused-operation API.
Not registered for engine discovery yet. Fixed dense LN/RMSNorm declarations,
FP32 pointwise computation, typed add/mul edges and arbitrary-rank broadcasting.
Unsupported graphs/layouts decline before compilation; execute owns no storage.
"""

from dataclasses import dataclass
from math import prod

import cudnn
import torch

from .affine import _dense, _scalar
from .config_sm100 import NormVariant
from .graph_analyzer import NormBinding, _DTYPE_FROM_CUDNN, _extract_facts, _node_type_name, _node_variant, resolve_variant_pack


@dataclass(frozen=True)
class PointwiseOp:
    mode: str
    inputs: tuple
    compute_dtype: object
    output_dtype: object


@dataclass(frozen=True)
class NormEpilogue:
    norm: object
    inputs: tuple
    layouts: tuple
    ops: tuple
    output: object


def analyze(graph):
    """Prepare a topologically ordered SSA program and right-aligned broadcasts."""
    norms = [n for n in graph.nodes if _node_variant(n) is not None]
    if len(norms) != 1:
        raise NotImplementedError("epilogue requires exactly one norm")
    node = norms[0]
    f = _extract_facts(node)
    if f.invalid or f.phase != "fprop" or f.variant not in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM) or f.extended_affine:
        raise NotImplementedError("epilogue prototype supports ordinary forward LayerNorm/RMSNorm")
    shape = f.x_shape
    if not shape or any(d <= 0 for d in shape) or not f.normalized_shape:
        raise NotImplementedError("epilogue requires positive fixed normalization geometry")
    d = prod(f.normalized_shape)
    if d >= 2**31 or prod(shape) // d >= 2**31:
        raise NotImplementedError("epilogue row geometry must fit Int32")
    if f.compute_dtype not in (None, cudnn.data_type.FLOAT) or f.y_t.get_data_type() not in _DTYPE_FROM_CUDNN:
        raise NotImplementedError("epilogue requires FP32 norm compute and FP16/BF16/FP32 output")
    if f.y_t.get_data_type() != f.x_t.get_data_type():
        raise NotImplementedError("norm output dtype must match X")
    if not f.y_t.get_is_virtual() or tuple(f.y_t.get_dim()) != shape:
        raise NotImplementedError("intermediate norm output must be virtual and match X")
    scale_shape = (1,) * (len(shape) - len(f.scale_t.get_dim())) + tuple(f.scale_t.get_dim()) if f.scale_t is not None else ()
    # Require a trailing dense normalization region, not merely an equal count.
    split, tail = len(shape), 1
    while split and tail < d:
        split -= 1
        tail *= shape[split]
    if tail != d or len(scale_shape) != len(shape) or any(v != 1 for v in scale_shape[:split]) or scale_shape[split:] != shape[split:]:
        raise NotImplementedError("scale must span trailing normalization axes")
    for name, t in (("X", f.x_t), ("scale", f.scale_t), ("bias", f.bias_t)):
        if t is not None and (t.get_is_virtual() or t.get_data_type() not in _DTYPE_FROM_CUDNN or not _dense(t)):
            raise NotImplementedError(f"{name} must be a bound dense floating-point tensor")
    if f.bias_t is not None and tuple(f.bias_t.get_dim()) != tuple(f.scale_t.get_dim()):
        raise NotImplementedError("bias shape must match scale")
    if f.eps_t is not None and (not f.eps_t.get_is_pass_by_value() or prod(f.eps_t.get_dim()) != 1):
        raise NotImplementedError("epsilon must be a host pass-by-value scalar")
    for t in (f.mean_t, f.inv_var_t):
        if t is not None and not t.get_is_virtual():
            if t.get_data_type() != cudnn.data_type.FLOAT or not _dense(t) or prod(t.get_dim()) != prod(shape) // d:
                raise NotImplementedError("norm statistics require dense FP32 row storage")
    pointwise = [n for n in graph.nodes if n is not node]
    if not pointwise:
        raise NotImplementedError("epilogue requires at least one pointwise consumer")
    aux, layouts, ops = [], [], []
    # First collect external inputs, then assign stable SSA slots.
    pointwise_outputs = {id(t) for n in pointwise for t in n.outputs.values()}
    stats_outputs = {id(t) for t in node.outputs.values() if t is not f.y_t}
    for n in pointwise:
        for t in n.inputs.values():
            if id(t) in stats_outputs:
                raise NotImplementedError("pointwise consumption of saved statistics is not lowered")
            if id(t) == id(f.y_t) or id(t) in pointwise_outputs or any(t is a for a in aux):
                continue
            if t.get_is_virtual() or t.get_is_pass_by_value() or t.get_data_type() not in _DTYPE_FROM_CUDNN or not _dense(t):
                raise NotImplementedError("pointwise auxiliaries require bound dense floating-point storage")
            dims = tuple(t.get_dim())
            if len(dims) > len(shape):
                raise NotImplementedError("auxiliary rank exceeds norm output rank")
            padded = (1,) * (len(shape) - len(dims)) + dims
            strides = (0,) * (len(shape) - len(dims)) + tuple(t.get_stride())
            if any(a not in (1, b) for a, b in zip(padded, shape)):
                raise NotImplementedError("auxiliary shape is not broadcastable to norm output")
            layout = tuple((prod(shape[i + 1 :]), shape[i], s) for i, (a, s) in enumerate(zip(padded, strides)) if a != 1)
            aux.append(t)
            layouts.append(layout)
    slots = {id(f.y_t): 0, **{id(t): i + 1 for i, t in enumerate(aux)}}
    dependent = {id(f.y_t)}
    output = None
    for n in pointwise:
        mode = getattr(n, "params", {}).get("mode")
        mode = str(getattr(mode, "name", mode)).lower()
        ins, outs = list(n.inputs.values()), list(n.outputs.values())
        if _node_type_name(n) != "POINTWISE" or mode not in ("add", "mul") or len(ins) != 2 or len(outs) != 1:
            raise NotImplementedError("epilogue supports binary add/mul nodes")
        t = outs[0]
        if any(id(a) not in slots for a in ins) or not any(id(a) in dependent for a in ins):
            raise NotImplementedError("pointwise DAG must be topologically ordered and depend on norm Y")
        compute = getattr(n, "compute_data_type", None) or cudnn.data_type.FLOAT
        if compute != cudnn.data_type.FLOAT or t.get_data_type() not in _DTYPE_FROM_CUDNN or tuple(t.get_dim()) != shape:
            raise NotImplementedError("pointwise outputs must retain norm shape and use FP32 compute")
        if not t.get_is_virtual():
            if n is not pointwise[-1] or not _dense(t):
                raise NotImplementedError("only the final dense pointwise output may be observable")
            output = t
        ops.append(PointwiseOp(mode, tuple(slots[id(a)] for a in ins), compute, t.get_data_type()))
        slots[id(t)] = 1 + len(aux) + len(ops) - 1
        dependent.add(id(t))
    if output is None:
        raise NotImplementedError("pointwise DAG requires a bound final output")
    # Every virtual result must contribute to the final output, not be discarded.
    used = {ops[-1].inputs[0], ops[-1].inputs[1]}
    for i in reversed(range(len(ops) - 1)):
        if 1 + len(aux) + i not in used:
            raise NotImplementedError("epilogue has an unused pointwise branch")
        used.update(ops[i].inputs)
    return NormEpilogue(f, tuple(aux), tuple(layouts), tuple(ops), output)


def prepare(graph):
    """Compile once from declarations; return a zero-workspace variant-pack callable."""
    p = analyze(graph)
    f = p.norm
    from cutlass.cute.runtime import make_fake_tensor
    import cutlass
    from cudnn.frost.compiled_cache import compile_cached
    from .epilogue_codegen import forward
    from .config_sm100 import TemplateParams, make_cfg
    from .dtypes import DTYPE_TO_CUTLASS, torch_dtype_to_str
    from .utils import cur_stream, fake_stream

    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)):
        raise NotImplementedError("row norm epilogue requires SM90/SM100/SM103")
    device = torch.cuda.current_device()
    d, r = prod(f.normalized_shape), prod(f.x_shape) // prod(f.normalized_shape)
    cfg = make_cfg(TemplateParams(variant=f.variant, io_dtype=torch_dtype_to_str(f.dtype), has_beta=f.has_beta), d, staged_rows=1)
    roles = dict(
        x=f.x_t,
        y=p.output,
        gamma=f.scale_t,
        beta=f.bias_t,
        mean=f.mean_t if f.has_mean and f.mean_t is not None and not f.mean_t.get_is_virtual() else None,
        rstd=f.inv_var_t if f.inv_var_t is not None and not f.inv_var_t.get_is_virtual() else None,
    )
    roles.update({f"aux{i}": t for i, t in enumerate(p.inputs)})
    shapes = {name: ((r, d) if name in ("x", "y") else (prod(t.get_dim()),)) for name, t in roles.items() if t is not None}

    def dtype(dt):
        return DTYPE_TO_CUTLASS[torch_dtype_to_str(_DTYPE_FROM_CUDNN[dt])]

    spec = (dtype(f.y_t.get_data_type()), p.layouts, tuple((op.mode, *op.inputs, dtype(op.compute_dtype), dtype(op.output_dtype)) for op in p.ops))
    fake = {
        name: make_fake_tensor(dtype(t.get_data_type()), shapes[name], (d, 1) if name in ("x", "y") else (1,), assumed_align=16) if t is not None else None
        for name, t in roles.items()
    }
    ce = (r, d, cfg.V, cfg.block_threads, cfg.elem_bytes, cfg.stage_mode, cfg.vec, f.has_mean, f.has_beta, spec)
    args = [fake[name] for name in ("x", "y", "gamma", "beta", "mean", "rstd")]
    aux = tuple(fake[f"aux{i}"] for i in range(len(p.inputs)))
    key = repr((ce, tuple((name, shapes.get(name), t.get_data_type() if t is not None else None) for name, t in roles.items())))
    fn = compile_cached(
        forward, *args, aux, cutlass.Float32(0), *ce, fake_stream(), cache_key="norm_epilogue:" + key, symbol="norm_epilogue", options="--enable-tvm-ffi"
    )
    # An input can be both norm X and a pointwise residual. Repeated roles do
    # not make that same tensor's UID/name ambiguous.
    unique_roles, seen = {}, set()
    for name, t in dict(roles, epsilon=f.eps_t).items():
        if t is not None and id(t) not in seen:
            unique_roles[name] = t
            seen.add(id(t))
    binding = NormBinding(unique_roles)
    outputs = {"y", "mean", "rstd"}
    declarations = {name: (tuple(t.get_dim()), _DTYPE_FROM_CUDNN[t.get_data_type()]) for name, t in roles.items() if t is not None}

    def execute(variant_pack, workspace=None):
        resolved = resolve_variant_pack(variant_pack, binding)
        eps = _scalar(f.eps_t, resolved, f.eps)
        views = {}
        for name, t in roles.items():
            if t is None:
                views[name] = None
                continue
            buf = resolved.get(id(t))
            if (
                not isinstance(buf, torch.Tensor)
                or buf.device != torch.device("cuda", device)
                or buf.dtype != declarations[name][1]
                or tuple(buf.shape) != declarations[name][0]
                or not buf.is_contiguous()
                or buf.data_ptr() % 16
            ):
                raise ValueError(f"norm epilogue: {name} binding must match declared dense aligned shape/dtype/device")
            views[name] = buf.detach().view(shapes[name])
        spans = [(name, t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for name, t in views.items() if t is not None]
        for i, (name, lo, hi) in enumerate(spans):
            for other, olo, ohi in spans[i + 1 :]:
                if (name in outputs or other in outputs) and lo < ohi and olo < hi:
                    raise ValueError(f"norm epilogue: overlapping storage for {name}/{other}")
        with torch.cuda.device(device):
            fn(
                *(views[name] for name in ("x", "y", "gamma", "beta", "mean", "rstd")),
                tuple(views[f"aux{i}"] for i in range(len(p.inputs))),
                cutlass.Float32(eps),
                cur_stream(),
            )

    execute.workspace_bytes = 0
    return execute
