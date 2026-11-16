# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape-generic, unsplit convolution weight gradient for SM100/SM103."""

from math import prod

import cudnn
import cutlass
from cuda.bindings import driver as cuda
from cudnn.engines.base import BaseEngine, CompiledPlan
from cudnn.frost.buffers import reordering_name as _reordering_name
from cudnn.frost.device import build_device, current_device
from .checks import (
    DENSE_CUTEDSL_MIN_VERSION,
    _check_cutedsl,
    _check_sm100,
    _check_5d_inputs,
    _check_channels_last_3d_layout,
    _check_conv_mode,
    _check_3d_conv_geometry,
    _check_matching_channels,
    _check_fp32_compute_type,
    _check_output_shapes,
)
from .graph_analyzer import analyze_wgrad
from .utils import _tuple_param, _ZERO, _ONE, _channels_last_3d_stride, _storage_dtype_name, _CUTLASS_STORAGE_DTYPES


def fixed_knobs():
    k = cudnn.knob_type
    return {k.TILE_M: 128, k.TILE_N: 128, k.TILEK: 128, k.CTA_GROUP: 1, k.STAGES: 3}


class _WgradPlan(CompiledPlan):
    takes_variant_pack = True

    def __init__(self, artifact, analysis, device):
        self._compiled = artifact
        self._tensors = (analysis.image, analysis.loss, analysis.weight)
        self._shapes = tuple(tuple(t.dim) for t in self._tensors)
        self._device = device
        self._indices = None
        self._dtypes = tuple(_storage_dtype_name(t.data_type) for t in self._tensors)
        self._geometry = tuple(
            tuple(cutlass.Int32(v) for v in _tuple_param(analysis.conv_node, name, default))
            for name, default in (("pre_padding", _ZERO), ("post_padding", _ZERO), ("stride", _ONE), ("dilation", _ONE))
        )

    def execute(self, graph, pack, ctx):
        if current_device() != self._device:
            raise ValueError("frost_conv_wgrad: execution device differs from plan device")
        if self._indices is None:
            self._indices = [pack.index_of(t.uid) for t in self._tensors]
        operands = pack.operands(self._indices)
        for role, operand, shape, dtype in zip(("X", "dY", "dW"), operands, self._shapes, self._dtypes):
            if tuple(operand.shape) != shape or tuple(operand.stride()) != _channels_last_3d_stride(shape):
                raise ValueError(f"frost_conv_wgrad: {role} must match the declared compact channels-last shape and stride")
            if operand.dtype != dtype or operand.data_ptr() % 16:
                raise ValueError(f"frost_conv_wgrad: {role} requires {dtype} storage and 16-byte alignment")
        self._compiled(*(v.permute(0, 2, 3, 4, 1) for v in operands), cuda.CUstream(ctx.stream or 0), *self._geometry)


class FrostConvWgradEngine(BaseEngine):
    name = "frost_conv_wgrad"
    behavior_notes = (cudnn.behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph):
        a = analyze_wgrad(graph)
        _check_cutedsl(self.name, DENSE_CUTEDSL_MIN_VERSION)
        _check_sm100()
        tensors = (("X", a.image), ("dY", a.loss), ("dW", a.weight))

        _check_5d_inputs(tensors)
        _check_channels_last_3d_layout(tensors)
        _check_matching_channels(a.image, a.weight)
        _check_conv_mode(a.conv_node)
        _check_fp32_compute_type(a.context)
        if a.conv_node.compute_data_type not in (None, cudnn.data_type.FLOAT):
            raise NotImplementedError("frost_conv_wgrad requires FP32 accumulation")
        names = tuple(_storage_dtype_name(t.data_type) for _, t in tensors)
        if any(name is None for name in names) or a.image.data_type != a.loss.data_type:
            raise NotImplementedError("frost_conv_wgrad requires matching FP16/BF16/FP32/FP8 X and dY and a supported dW dtype")
        if any(_reordering_name(t) is not None for _, t in tensors):
            raise NotImplementedError("frost_conv_wgrad does not support reordered storage")
        for (role, tensor), name in zip(tensors, names):
            bits = _CUTLASS_STORAGE_DTYPES[name].width
            if tensor.dim[1] % (128 // bits):
                raise NotImplementedError(f"frost_conv_wgrad: {role} channel-row pitch must be 16-byte aligned")
        if a.epilogue_node is not None:
            if a.context.intermediate_data_type != cudnn.data_type.FLOAT or a.conv_node.outputs["DW"].data_type != cudnn.data_type.FLOAT:
                raise NotImplementedError("frost_conv_wgrad requires an FP32 intermediate for a unary epilogue")
            if a.epilogue_node.compute_data_type not in (None, cudnn.data_type.FLOAT):
                raise NotImplementedError("frost_conv_wgrad requires FP32 epilogue compute")
        from .templates.epilogue import get_epilogue_op
        from cutlass.testing import CantImplementError

        try:
            get_epilogue_op(a.epilogue, a.epilogue_attrs)
        except (CantImplementError, KeyError) as exc:
            raise NotImplementedError(f"frost_conv_wgrad: {exc}") from exc
        dilation = _tuple_param(a.conv_node, "dilation", _ONE)
        expected = _check_3d_conv_geometry(a.conv_node, a.image, a.weight)
        if any((extent - 1) * value > 31 for extent, value in zip(a.weight.dim[2:], dilation)):
            raise NotImplementedError("frost_conv_wgrad requires dilated filter offsets (filter_extent - 1) * dilation at most 31")
        _check_output_shapes(expected, (("dY", a.loss),))
        pre = _tuple_param(a.conv_node, "pre_padding", _ZERO)
        post = _tuple_param(a.conv_node, "post_padding", _ZERO)
        if any(v >= 2**31 for v in (*dilation, *pre, *post)) or any(extent + pad >= 2**31 for extent, pad in zip(a.image.dim[2:], post)):
            raise NotImplementedError("frost_conv_wgrad exceeds Int32 convolution-coordinate bounds")
        length = prod((expected[0], *expected[2:]))
        # Device coordinates and the scheduler use signed Int32. TMA dimensions
        # and byte strides also have finite descriptor fields; reject before JIT.
        if length >= 2**31 or prod(a.weight.dim[2:]) * ((a.weight.dim[0] + 63) // 64) * ((a.image.dim[1] + 31) // 32) >= 2**31:
            raise NotImplementedError("frost_conv_wgrad exceeds Int32 reduction/scheduler bounds")
        for (_, tensor), name in zip(tensors, names):
            if any(v >= 2**31 for v in tensor.dim) or any(v * (_CUTLASS_STORAGE_DTYPES[name].width // 8) >= 2**40 for v in tensor.stride):
                raise NotImplementedError("frost_conv_wgrad exceeds tensor-map dimensions or strides")

    def knobs_to_public(self, knobs):
        return self.knobs_from_public(knobs or {})

    def knobs_from_public(self, public):
        k = cudnn.knob_type
        domains = {k.TILE_M: (64, 128), k.TILE_N: range(32, 257, 32), k.TILEK: (64, 128), k.CTA_GROUP: (1, 2), k.STAGES: (3,)}
        if any(key not in domains or type(value) is not int or value not in domains[key] for key, value in public.items()):
            raise ValueError("frost_conv_wgrad: unsupported configuration or knob")
        return fixed_knobs() | public

    def build_plan(self, graph, plan, ctx=None):
        knobs = self.knobs_to_public(plan.knobs)
        a = analyze_wgrad(graph)
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            self.check_support(graph)
            # Load compiler primitives only after the DSL version/support gate.
            from .templates.sm100_conv_wgrad import compile
            from .tile_config import ConvTileConfig

            k = cudnn.knob_type
            config = ConvTileConfig(knobs[k.TILE_M], knobs[k.TILE_N], knobs[k.TILEK], knobs[k.CTA_GROUP])
            ab_dtype = _CUTLASS_STORAGE_DTYPES[_storage_dtype_name(a.image.data_type)]
            output_dtype = _CUTLASS_STORAGE_DTYPES[_storage_dtype_name(a.weight.data_type)]
            return _WgradPlan(compile(ab_dtype, output_dtype, config, a.epilogue, a.epilogue_attrs), a, current_device())


def FrostConvWgradEngines(ids: dict[str, int]) -> list[BaseEngine]:
    """The weight-gradient engines the manifest asked for, with assigned ids."""
    out = []
    for cls in (FrostConvWgradEngine,):
        if cls.name in ids:
            engine = cls()
            engine.engine_id = ids[cls.name]
            out.append(engine)
    return out
