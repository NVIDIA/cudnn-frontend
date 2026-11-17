# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-time validation and workspace-backed norm/block-quantize composition."""

from math import prod

import torch
import cudnn

from . import affine
from .config_sm100 import NormVariant
from .graph_analyzer import NormBinding, _DTYPE_FROM_CUDNN, resolve_variant_pack

_FP8 = {cudnn.data_type.FP8_E4M3: torch.float8_e4m3fn, cudnn.data_type.FP8_E5M2: torch.float8_e5m2}


def _geometry(f):
    # The backend also accepts a trailing singleton in its rank-4 descriptors.
    feature = max(i for i, d in enumerate(f.x_shape) if d != 1)
    return feature, (prod(f.x_shape[: feature - 1]), f.x_shape[feature - 1], f.x_shape[feature])


def _plain(t):
    return getattr(t.get_reordering_type(), "name", None) in (None, "NONE") and getattr(t, "ragged_offset", None) is None


def _strides(shape, axis, transpose):
    order = list(reversed(range(len(shape))))
    if transpose:
        i = order.index(axis)
        order = order[i:] + order[:i]
    strides, running = [0] * len(shape), 1
    for i in order:
        strides[i], running = running, running * shape[i]
    return tuple(strides)


def mismatch(f):
    """Decline unsupported declarations before compiling or touching GPU storage."""
    if f.device_cc not in ((10, 0), (10, 3), (10, 7)):
        return "norm block scaling requires SM100, SM103, or SM107"
    if f.variant not in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM):
        return "block scaling supports LayerNorm, RMSNorm, AdaLN, and AdaRMSNorm"
    if not f.x_shape or len(f.x_shape) < 2 or not f.normalized_shape or not any(d > 1 for d in f.x_shape):
        return "block scaling requires a matrix of norm rows"
    source = f.y_t if f.phase == "fprop" else f.dx_t
    if source is None or not source.get_is_virtual():
        return "block-scaled norm requires a virtual Y/DX intermediate"
    reason = affine.mismatch(f, staged_output=True)
    if reason:
        return reason
    feature, (b, m, n) = _geometry(f)
    if feature < 1 or n != prod(f.normalized_shape) or n % 32:
        return "block-scaled norm requires one trailing feature axis divisible by 32"
    if len(f.quantize_nodes) not in (1, 2):
        return "expected rowwise quantization and optional columnwise quantization"
    tile_n = 128 if n % 128 == 0 else (64 if n % 64 == 0 else 32)
    axes = []
    output_type = None
    for q in f.quantize_nodes:
        if q.inputs.get("input") is not source or q.inputs.get("group_offset") is not None:
            return "quantizers must consume the norm intermediate without group offsets"
        if q.compute_data_type not in (None, cudnn.data_type.FLOAT):
            return "block quantization requires fp32 computation"
        axis = q.params.get("axis")
        axis = len(f.x_shape) - 1 if axis in (None, -1) else axis
        trans = bool(q.params.get("transpose", False))
        if axis not in (feature, feature - 1) or axis in axes:
            return "expected distinct rowwise/columnwise quantization axes"
        axes.append(axis)
        y, sf = q.outputs.get("Y"), q.outputs.get("scale")
        if y is None or sf is None or y.get_is_virtual() or sf.get_is_virtual():
            return "quantized payloads and scales must be bound outputs"
        fp4 = y.get_data_type() == cudnn.data_type.FP4_E2M1
        tile_m = 4 if fp4 else 32
        if not 0 < b * ((m + tile_m - 1) // tile_m) * (n // tile_n) < 2**31:
            return "block quantization grid exceeds the CUDA grid-x limit"
        block_size = 16 if fp4 else 32
        scale_type = cudnn.data_type.FP8_E4M3 if fp4 else cudnn.data_type.FP8_E8M0
        if (not fp4 and y.get_data_type() not in _FP8) or sf.get_data_type() != scale_type or q.params.get("block_size") != block_size:
            return "expected MXFP8/E8M0 block-32 or NVFP4/E4M3 block-16 quantization"
        if fp4 and (f.phase != "fprop" or f.training or len(f.quantize_nodes) != 1 or axis != feature or trans):
            return "NVFP4 supports rowwise forward inference only"
        if output_type is not None and output_type != y.get_data_type():
            return "paired quantizers must use the same payload dtype"
        output_type = y.get_data_type()
        if trans and (axis != feature - 1 or f.phase == "bprop"):
            return "only forward columnwise quantization supports transpose"
        if axis == feature - 1 and (m % 32 or (f.phase == "fprop" and not f.training)):
            return "columnwise quantization requires training and complete 32-row blocks"
        for t, shape in ((y, f.x_shape), (sf, tuple(d // block_size if i == axis else d for i, d in enumerate(f.x_shape)))):
            expected_stride = _strides(shape, axis, trans)
            if tuple(t.get_dim()) != shape or any(d > 1 and s != e for d, s, e in zip(shape, t.get_stride(), expected_stride)):
                return "quantized output shape/strides do not match the declared axis and transpose"
            if not _plain(t) or t.get_is_pass_by_value():
                return "block scaling requires plain device payload/scale storage"
    if feature not in axes or (f.phase == "bprop" and len(axes) != 2):
        return "forward requires a row output; backward requires row and column outputs"
    for t in affine._roles(f).values():
        if t is not None and not _plain(t):
            return "norm block scaling requires plain dense norm operands"
    return None


def lower(f):
    reason = mismatch(f)
    if reason:
        raise NotImplementedError(reason)
    import cutlass
    from cutlass.cute.runtime import from_dlpack, make_fake_tensor
    from cudnn.frost.compiled_cache import compile_cached
    from .block_scale_kernels import mxfp8, nvfp4
    from .utils import cur_stream, fake_stream

    source = f.y_t if f.phase == "fprop" else f.dx_t
    feature, shape = _geometry(f)
    b, m, n = shape
    qs = sorted(f.quantize_nodes, key=lambda q: feature if q.params.get("axis") in (None, -1) else q.params["axis"], reverse=True)
    row, row_sf = qs[0].outputs["Y"], qs[0].outputs["scale"]
    col, col_sf = (qs[1].outputs["Y"], qs[1].outputs["scale"]) if len(qs) == 2 else (None, None)
    fp4 = row.get_data_type() == cudnn.data_type.FP4_E2M1
    block_size = 16 if fp4 else 32
    tensors = (source, row, row_sf, col, col_sf)
    source_dtype = _DTYPE_FROM_CUDNN[source.get_data_type()]
    dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}[source_dtype]
    shapes = (shape, (b, m, n // 2) if fp4 else shape, (b, m, n // block_size), shape if col else None, (b, m // 32, n) if col else None)
    # Preserve the column output's declared physical order, including the
    # backend's transpose rotation across a non-singleton batch dimension.
    strides = []
    for i, (t, dims) in enumerate(zip(tensors, shapes)):
        if t is None:
            strides.append(None)
        else:
            q = qs[0] if i < 3 else qs[1]
            axis = feature if i < 3 else feature - 1
            ss = _strides(t.get_dim(), axis, i > 0 and bool(q.params.get("transpose", False)))
            strides.append((ss[feature - 2] if feature >= 2 else prod(dims), ss[feature - 1], ss[feature]))
    if fp4:
        strides[1] = (m * n // 2, n // 2, 1)
    fake = [
        None if t is None else make_fake_tensor(dtype if i == 0 else cutlass.Uint8, dims, stride, assumed_align=16 if i == 0 else 1)
        for i, (t, dims, stride) in enumerate(zip(tensors, shapes, strides))
    ]
    e5m2 = row.get_data_type() == cudnn.data_type.FP8_E5M2
    device = torch.cuda.current_device()
    with torch.cuda.device(device):
        norm = affine.lower(f, staged_output=True)
        key = repr((source_dtype, shapes, strides, e5m2, fp4))
        quant = compile_cached(
            nvfp4 if fp4 else mxfp8, *fake, e5m2, fake_stream(), cache_key="norm_block_scale:" + key, symbol="norm_block_scale", options="--enable-tvm-ffi"
        )
    roles = affine._roles(f)
    roles.update(row=row, row_sf=row_sf, col=col, col_sf=col_sf, epsilon=f.eps_t, one=f.scale_one_t)
    binding = NormBinding(roles)
    scratch_bytes = prod(shape) * (4 if source_dtype == torch.float32 else 2)

    def execute(variant_pack, ws=None):
        resolved = resolve_variant_pack(variant_pack, binding)
        scratch = ws.take(scratch_bytes, torch.uint8) if ws is not None else None
        if (
            not isinstance(scratch, torch.Tensor)
            or scratch.dtype != torch.uint8
            or scratch.device != torch.device("cuda", device)
            or not scratch.is_contiguous()
            or scratch.numel() < scratch_bytes
            or scratch.data_ptr() % 16
        ):
            raise ValueError("cudnn.norm: block scaling requires the reported aligned CUDA uint8 workspace")
        views = []
        output_dtypes = (
            source_dtype,
            torch.uint8 if fp4 else _FP8[row.get_data_type()],
            torch.float8_e4m3fn if fp4 else torch.float8_e8m0fnu,
            _FP8[col.get_data_type()] if col else None,
            torch.float8_e8m0fnu,
        )
        for i, (t, dims, stride, expected_dtype) in enumerate(zip(tensors, shapes, strides, output_dtypes)):
            if t is None:
                views.append(None)
                continue
            if i == 0:
                buf = scratch.reshape(-1)[:scratch_bytes].view(source_dtype).view(f.x_shape)
            else:
                buf = resolved.get(id(t))
                packed = fp4 and i == 1
                layout_ok = isinstance(buf, torch.Tensor) and (
                    (buf.is_contiguous() and buf.numel() == prod(shape) // 2)
                    if packed
                    else (
                        tuple(buf.shape) == tuple(t.get_dim())
                        and all(d == 1 or actual == declared for d, actual, declared in zip(buf.shape, buf.stride(), t.get_stride()))
                    )
                )
                if not layout_ok or buf.device != torch.device("cuda", device) or buf.dtype not in (expected_dtype, torch.uint8):
                    raise ValueError("cudnn.norm: quantized storage does not match its declared dtype, device, shape, or strides")
            views.append((buf if i == 0 else buf.view(torch.uint8)).as_strided(dims, stride))
        # Reject every output/input/workspace overlap before norm clears dparams
        # or writes stats. The intermediate is the workspace's sole occupant.
        spans = [("workspace", scratch.data_ptr(), scratch.data_ptr() + scratch.numel())]
        norm_outputs = {"y", "mean", "rstd"} if f.phase == "fprop" else {"dx", "dgamma", "dbeta"}
        outputs = norm_outputs | {"row", "row_sf", "col", "col_sf", "workspace"}
        for name, t in roles.items():
            buf = resolved.get(id(t))
            if t is source or not isinstance(buf, torch.Tensor) or buf.device.type != "cuda":
                continue
            spans.append((name, buf.data_ptr(), buf.data_ptr() + buf.numel() * buf.element_size()))
        for i, (name, lo, hi) in enumerate(spans):
            for other, lo2, hi2 in spans[i + 1 :]:
                if (name in outputs or other in outputs) and lo < hi2 and lo2 < hi:
                    raise ValueError(f"cudnn.norm: overlapping storage for {name} and {other}")
        pack = dict(variant_pack)
        pack[source] = views[0].view(f.x_shape)
        norm(pack)
        with torch.cuda.device(device):
            args = [None if v is None else from_dlpack(v, assumed_align=16 if i == 0 else 1, enable_tvm_ffi=True) for i, v in enumerate(views)]
            quant(*args, cur_stream())

    execute.workspace_bytes = scratch_bytes
    return execute
