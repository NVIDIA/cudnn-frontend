# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph lowering for adaptive LayerNorm/RMSNorm/InstanceNorm and gamma + 1.

The ordinary norm routes retain their tuned schedules. These declarations use
the shared row kernels with batch-aware affine indexing and typed scale loads.
All compilation uses declarations; execution binds and validates current storage.
"""

from math import prod

import torch
import cudnn

from .graph_analyzer import NormBinding, _DTYPE_FROM_CUDNN, resolve_variant_pack
from .config_sm100 import NormVariant


def _dense(t):
    stride = 1
    dims, strides = t.get_dim(), t.get_stride()
    if not dims or len(dims) != len(strides):
        return False
    for d, s in zip(reversed(dims), reversed(strides)):
        if d < 1 or (d != 1 and s != stride):
            return False
        stride *= d
    return True


def _roles(f):
    if f.phase == "fprop":
        return dict(x=f.x_t, gamma=f.scale_t, beta=f.bias_t, y=f.y_t, mean=f.mean_t if f.has_mean else None, rstd=f.inv_var_t)
    return dict(dy=f.dy_t, x=f.x_t, gamma=f.scale_t, mean=f.mean_t if f.has_mean else None, rstd=f.inv_var_t, dx=f.dx_t, dgamma=f.dscale_t, dbeta=f.dbias_t)


def mismatch(f):
    """Check declarations before importing or compiling kernels."""
    if f.compute_dtype not in (None, cudnn.data_type.FLOAT):
        return "adaptive/zero-centered norm requires fp32 computation"
    if not f.normalized_shape or not f.scale_t:
        return "adaptive/zero-centered norm requires a scale and normalization dimensions"
    d = prod(f.normalized_shape)
    if d <= 0 or prod(f.x_shape) % d:
        return "invalid normalization dimensions"
    r = prod(f.x_shape) // d
    if not f.adaptive:
        shape = tuple(f.scale_t.get_dim())
        if len(shape) > len(f.x_shape):
            return "scale rank exceeds X rank"
        shape = (1,) * (len(f.x_shape) - len(shape)) + shape
        tail = 1
        split = len(f.x_shape)
        while split and tail < d:
            split -= 1
            tail *= f.x_shape[split]
        if tail != d or any(v != 1 for v in shape[:split]) or shape[split:] != f.x_shape[split:]:
            return "scale must span contiguous trailing normalization dimensions"
    if r >= 2**31 or d >= 2**31:
        return "norm row count and width must fit Int32"
    affine_width = 1 if f.variant == NormVariant.INSTANCE_NORM else d
    expected = dict(
        x=r * d,
        y=r * d,
        dy=r * d,
        dx=r * d,
        gamma=f.affine_batches * affine_width,
        beta=f.affine_batches * affine_width,
        dgamma=f.affine_batches * affine_width,
        dbeta=f.affine_batches * affine_width,
        mean=r,
        rstd=r,
    )
    required = ("x", "gamma", "y") if f.phase == "fprop" else ("dy", "x", "gamma", "rstd", "dx", "dgamma")
    roles = _roles(f)
    if f.phase == "bprop" and f.has_mean and roles["mean"] is None:
        return "centered normalization backward requires saved mean"
    for name in required:
        if roles[name] is None or roles[name].get_is_virtual():
            return f"{name} must be a bound tensor"
    for name, t in roles.items():
        if t is None:
            continue
        dtype = _DTYPE_FROM_CUDNN.get(t.get_data_type())
        if dtype is None:
            return f"unsupported {name} dtype"
        if name in ("mean", "rstd", "dgamma", "dbeta") and dtype != torch.float32:
            return f"{name} must be fp32"
        if name in ("y", "dx", "dy") and dtype != f.dtype:
            return f"{name} dtype must match X"
        if not _dense(t) or prod(t.get_dim()) != expected[name]:
            return f"{name} must have dense row-major storage with {expected[name]} elements; got {t.get_dim()}, strides {t.get_stride()}"
        if name in ("y", "dx", "dy") and tuple(t.get_dim()) != f.x_shape:
            return f"{name} shape must match X"
        if name in ("beta", "dgamma", "dbeta") and tuple(t.get_dim()) != tuple(f.scale_t.get_dim()):
            return f"{name} shape must match scale"
    if f.zero_centered_gamma and f.scale_add_dtype not in _DTYPE_FROM_CUDNN:
        return "unsupported gamma + 1 output dtype"
    for name, t in (("epsilon", f.eps_t), ("gamma offset", f.scale_one_t)):
        if t is not None and (not t.get_is_pass_by_value() or prod(t.get_dim()) != 1):
            return f"{name} must be a host pass-by-value scalar"
    return None


def _scalar(t, resolved, default):
    if t is None:
        return default
    value = getattr(t, "pass_by_value", None)
    if value is not None:
        return float(value)
    buf = resolved.get(id(t))
    if not isinstance(buf, torch.Tensor) or buf.device.type != "cpu" or buf.numel() != 1:
        raise ValueError("cudnn.norm: bind a one-element host tensor for the scalar")
    if buf.dtype != _DTYPE_FROM_CUDNN.get(t.get_data_type()):
        raise ValueError("cudnn.norm: scalar dtype does not match its declaration")
    return float(buf.item())


def validate_graph(graph):
    """Defer backend planning for extended affine graphs we can validate.

    The backend can reject this fusion while finalizing its operation graph.
    Defer that verdict to plan selection, as the GEMM and SDPA families do.
    Other norm graphs retain their existing validation path.
    """
    from .graph_analyzer import analyze, _norm_node

    f = analyze(graph)
    if f is None or f.invalid or not f.extended_affine or mismatch(f) is not None:
        return False
    if f.phase == "fprop":
        phase = _norm_node(graph).params.get("norm_forward_phase")
        if getattr(phase, "name", None) not in ("TRAINING", "INFERENCE"):
            return False
    return True


def lower(f):
    from cutlass.cute.runtime import from_dlpack, make_fake_tensor
    import cutlass
    import cutlass.cute as cute

    from .affine_kernels import forward, backward
    from .config_sm100 import TemplateParams, make_cfg
    from .dtypes import DTYPE_TO_CUTLASS, torch_dtype_to_str
    from .utils import cur_stream, fake_stream

    reason = mismatch(f)
    if reason:
        raise NotImplementedError(reason)
    d = prod(f.normalized_shape)
    r = prod(f.x_shape) // d
    roles = _roles(f)
    if f.phase == "fprop":
        roles = {name: t if t is not None and not t.get_is_virtual() else None for name, t in roles.items()}
    device = torch.cuda.current_device()
    params = TemplateParams(variant=f.variant, io_dtype=torch_dtype_to_str(f.dtype), has_beta=f.has_beta)
    cfg = make_cfg(params, d, staged_rows=1 if f.phase == "fprop" else 2)
    round_type = DTYPE_TO_CUTLASS[torch_dtype_to_str(_DTYPE_FROM_CUDNN[f.scale_add_dtype])] if f.zero_centered_gamma else None
    # AdaIN broadcasts one affine value across each spatial row. LN/RMSNorm
    # retain a separate affine value for every feature in a row.
    per_row = f.variant == NormVariant.INSTANCE_NORM
    affine = (f.affine_batches, f.affine_row_divisor, round_type, per_row)
    shapes = {name: ((r, d) if name in ("x", "y", "dy", "dx") else (prod(t.get_dim()),)) for name, t in roles.items() if t is not None}
    fake = []
    for name, t in roles.items():
        if t is None:
            fake.append(None)
        else:
            dt = DTYPE_TO_CUTLASS[torch_dtype_to_str(_DTYPE_FROM_CUDNN[t.get_data_type()])]
            shape = shapes[name]
            fake.append(make_fake_tensor(dt, shape, (d, 1) if len(shape) == 2 else (1,), assumed_align=16))
    ce = (r, d, cfg.V, cfg.block_threads, cfg.elem_bytes, cfg.stage_mode, cfg.vec, f.has_mean, f.has_beta, affine)
    host = forward if f.phase == "fprop" else backward
    scalars = (cutlass.Float32(0),) if f.phase == "fprop" else ()
    from cudnn.frost.compiled_cache import compile_cached

    # Every baked shape, dtype and affine choice participates; the shared cache
    # also keys the source tree and CUDA context.
    key = repr((f.phase, ce, [(name, shapes.get(name), t.get_data_type() if t is not None else None) for name, t in roles.items()]))
    with torch.cuda.device(device):
        fn = compile_cached(
            host, *fake, *scalars, *ce, fake_stream(), cache_key="norm_affine:" + key, symbol="norm_affine_" + f.phase, options="--enable-tvm-ffi"
        )
    binding = NormBinding(dict(roles, epsilon=f.eps_t, one=f.scale_one_t))

    def execute(variant_pack, ws=None):
        resolved = resolve_variant_pack(variant_pack, binding)
        eps = _scalar(f.eps_t, resolved, f.eps)
        if f.zero_centered_gamma and _scalar(f.scale_one_t, resolved, 1.0) != 1.0:
            raise ValueError("cudnn.norm: zero-centered gamma requires an offset of 1")
        views = {}
        # Validate every current operand before the backward clears any output.
        for name, t in roles.items():
            if t is None:
                views[name] = None
                continue
            buf = resolved.get(id(t))
            if not isinstance(buf, torch.Tensor):
                raise ValueError(f"cudnn.norm: missing tensor binding for {name}")
            if (
                buf.device != torch.device("cuda", device)
                or buf.dtype != _DTYPE_FROM_CUDNN[t.get_data_type()]
                or tuple(buf.shape) != tuple(t.get_dim())
                or not buf.is_contiguous()
                or buf.data_ptr() % 16
            ):
                raise ValueError(f"cudnn.norm: {name} storage does not match its declared dtype, device, shape, or dense aligned layout")
            views[name] = buf.detach().view(shapes[name])
        outputs = {"y", "mean", "rstd"} if f.phase == "fprop" else {"dx", "dgamma", "dbeta"}
        spans = [(name, t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for name, t in views.items() if t is not None]
        for i, (name, lo, hi) in enumerate(spans):
            for other, other_lo, other_hi in spans[i + 1 :]:
                if (name in outputs or other in outputs) and lo < other_hi and other_lo < hi:
                    raise ValueError(f"cudnn.norm: overlapping storage for {name} and {other}")
        with torch.cuda.device(device):
            if f.phase == "bprop" and not per_row:
                views["dgamma"].zero_()
                if views["dbeta"] is not None:
                    views["dbeta"].zero_()
            args = [from_dlpack(t, assumed_align=16, enable_tvm_ffi=True) if t is not None else None for t in views.values()]
            runtime = (cutlass.Float32(eps),) if f.phase == "fprop" else ()
            fn(*args, *runtime, cur_stream())

    execute.workspace_bytes = 0
    return execute
