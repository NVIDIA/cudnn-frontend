# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FROST convolution engine: JIT-compiled convolution for cuDNN graphs.

This first version intentionally serves one small envelope: a conv_fprop,
optionally followed by one unary epilogue, with FP32 accumulation and
channels-last 3D layouts. Input/output dtypes, input shape, and convolution
parameters are specialized at build time.
"""

from math import prod
from typing import Optional, Sequence

import cudnn
import cutlass
from cuda.bindings import driver as _cuda
from cudnn import behavior_note, GraphContext, data_type
from cudnn.datatypes import _cudnn_to_frost_dtype_name
from cudnn.engines.base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig
from cudnn.frost import buffers
from cudnn.frost.device import build_device, current_device
from cudnn.conv.frost.graph_analyzer import analyze, ConvGraph
from cudnn._pygraph import pygraph

from ._cutedsl import BLOCK_SCALE_CUTEDSL_MIN_VERSION, DENSE_CUTEDSL_MIN_VERSION, requirement_error as cutedsl_requirement_error

_ZERO = (0, 0, 0)
_ONE = (1, 1, 1)
_CUTLASS_STORAGE_DTYPES = {
    "float16": cutlass.Float16,
    "bfloat16": cutlass.BFloat16,
    "float32": cutlass.Float32,
    "float8_e4m3fn": cutlass.Float8E4M3FN,
    "float8_e5m2": cutlass.Float8E5M2,
}
_FP4_RUNTIME_DTYPES = frozenset(("code17_4", "float4_e2m1fn_x2"))
_BLOCK_SCALE_OUTPUT_SPECS = {
    cudnn.data_type.FLOAT: ("float32", 32),
    cudnn.data_type.HALF: ("float16", 16),
    cudnn.data_type.BFLOAT16: ("bfloat16", 16),
    cudnn.data_type.FP8_E4M3: ("float8_e4m3fn", 8),
    cudnn.data_type.FP8_E5M2: ("float8_e5m2", 8),
    cudnn.data_type.FP4_E2M1: ("float4_e2m1fn_x2", 4),
}


def _storage_dtype_name(data_type) -> Optional[str]:
    name = _cudnn_to_frost_dtype_name(data_type)
    return name if name in _CUTLASS_STORAGE_DTYPES else None


def _input_channel_tile(dtype_name: str) -> int:
    """Channel elements covered by the narrowest supported mainloop K tile."""
    return 64 // buffers.DTYPE_ITEMSIZE[dtype_name]


def _tuple_param(node, name: str, default: Sequence[int]) -> tuple[int, ...]:
    value = node.params.get(name)
    return tuple(default if value is None else value)


def _channels_last_3d_stride(dim: Sequence[int]) -> tuple[int, ...]:
    """Compact NCDHW/KCTRS stride with channel (axis 1) contiguous."""
    _n, c, d, h, w = (int(x) for x in dim)
    return (c * d * h * w, 1, h * w * c, w * c, c)


def _expected_output_shape(image, weight, pre_padding, post_padding, stride, dilation) -> tuple[int, ...]:
    spatial = tuple((image[i] + pre_padding[i - 2] + post_padding[i - 2] - dilation[i - 2] * (weight[i] - 1) - 1) // stride[i - 2] + 1 for i in range(2, 5))
    return (image[0], weight[0], *spatial)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _reordering_name(tensor) -> Optional[str]:
    reordering = getattr(tensor, "reordering_type", None)
    name = getattr(reordering, "name", None)
    if name in (None, "NONE"):
        return None
    return name


def _dense_span(shape, stride) -> int:
    return 1 + sum((int(extent) - 1) * int(step) for extent, step in zip(shape, stride))


def _logical_storage_bytes(shape: Sequence[int], bits: int) -> int:
    """Storage bytes for a compact logical tensor, including sub-byte types."""
    total_bits = prod(int(extent) for extent in shape) * int(bits)
    if total_bits % 8:
        raise ValueError(f"frost_conv: logical shape {tuple(shape)} at {bits} bits per element is not byte-packable")
    return total_bits // 8


def _physical_channel_shape(shape: Sequence[int], bits: int) -> tuple[int, ...]:
    """Translate a logical channels-first graph shape to its runtime carrier."""
    shape = tuple(int(extent) for extent in shape)
    if bits != 4:
        return shape
    if shape[1] % 2:
        raise ValueError(f"frost_conv: packed FP4 channel extent must be even; got {shape[1]}")
    return (shape[0], shape[1] // 2, *shape[2:])


def _runtime_dtype_names(dtype_name: str, bits: int) -> frozenset[str]:
    return _FP4_RUNTIME_DTYPES if bits == 4 else frozenset((dtype_name,))


def _operand_storage_bytes(operand) -> int:
    """Runtime span in bytes without consulting the byte-only shared dtype map."""
    if operand.dtype in _FP4_RUNTIME_DTYPES:
        # Native DLPack FP4 reports a four-bit element whose C++ element_size()
        # is zero. Runtime shapes are already packed two logical values/byte.
        return int(operand.numel())
    return int(operand.numel()) * int(operand.element_size())


def _check_sm100():
    sm = buffers.current_sm()
    if sm != 100:
        raise NotImplementedError(f"frost_conv requires SM100, but found {sm}")


def _check_cutedsl(what: str, minimum_version: tuple[int, int]) -> None:
    error = cutedsl_requirement_error(what, minimum_version)
    if error is not None:
        raise NotImplementedError(error)


def _cutlass_block_scale_output_dtype(data_type):
    if data_type == cudnn.data_type.FP4_E2M1:
        return cutlass.Float4E2M1FN

    dtype_name = _storage_dtype_name(data_type)
    if dtype_name is None:
        raise NotImplementedError(f"frost_conv: unsupported block-scaled convolution output dtype {data_type}")
    return _CUTLASS_STORAGE_DTYPES[dtype_name]


def _check_5d_inputs(tensors) -> None:
    """Require positive rank-5 logical tensors, naming every offender."""
    bad_rank = tuple(role for role, tensor in tensors if len(tensor.dim) != 5)
    if bad_rank:
        raise NotImplementedError(f"frost_conv: tensors {bad_rank} must be rank-5")
    bad_extents = tuple(role for role, tensor in tensors if any(int(extent) <= 0 for extent in tensor.dim))
    if bad_extents:
        raise NotImplementedError(f"frost_conv: tensors {bad_extents} must have positive extents")


def _check_matching_channels(image, weight) -> None:
    if image.dim[1] != weight.dim[1]:
        raise NotImplementedError(f"frost_conv: image/filter channels differ ({image.dim[1]} vs {weight.dim[1]})")


def _check_conv_mode(node, expected_mode=cudnn._pybind_module.convolution_mode.CROSS_CORRELATION):
    """Checks if the convolution mode of node matches expected_mode."""
    conv_mode = node.params.get("convolution_mode", cudnn._pybind_module.convolution_mode.CROSS_CORRELATION)
    if conv_mode != expected_mode:
        raise NotImplementedError(f"frost_conv: expect convolution mode to be {expected_mode} but got {conv_mode}")


def _check_3d_conv_geometry(node, image, weight):
    """Validate shared 3D im2col limits and return the output shape."""
    pre_padding = _tuple_param(node, "pre_padding", _ZERO)
    post_padding = _tuple_param(node, "post_padding", _ZERO)
    stride = _tuple_param(node, "stride", _ONE)
    dilation = _tuple_param(node, "dilation", _ONE)
    params = (pre_padding, post_padding, stride, dilation)
    if any(len(value) != 3 for value in params):
        raise NotImplementedError("frost_conv requires three spatial convolution parameters")
    if any(value < 0 for value in (*pre_padding, *post_padding)):
        raise NotImplementedError("frost_conv requires non-negative padding")
    if any(value <= 0 for value in (*stride, *dilation)):
        raise NotImplementedError("frost_conv requires positive stride and dilation")
    if any(value > 8 for value in stride):
        raise NotImplementedError("frost_conv requires spatial strides at most 8")
    for filter_extent, lower_pad, upper_pad, dilation_extent in zip(weight.dim[2:], pre_padding, post_padding, dilation):
        lower_corner = -lower_pad
        upper_corner = upper_pad - (filter_extent - 1) * dilation_extent
        if not (-16 <= lower_corner <= 15 and -16 <= upper_corner <= 15):
            raise NotImplementedError("frost_conv requires rank-5 im2col TMA filter-window corners in [-16, 15]; " f"got ({lower_corner}, {upper_corner})")

    output_shape = _expected_output_shape(image.dim, weight.dim, pre_padding, post_padding, stride, dilation)
    if any(extent <= 0 for extent in output_shape):
        raise NotImplementedError(f"frost_conv: convolution parameters produce invalid output shape {output_shape}")
    return output_shape


def _check_output_shapes(expected_shape, tensors) -> None:
    for role, tensor in tensors:
        if tuple(tensor.dim) != expected_shape:
            raise NotImplementedError(f"frost_conv: {role} shape must be {expected_shape}; got {tuple(tensor.dim)}")


def _check_alignment(image, weight, *, input_bits: int, output_bits: int, input_dtype: str, output_dtype: str) -> None:
    """Check the 16-byte contiguous channel units required by TMA."""
    input_multiple = 128 // input_bits
    output_multiple = 128 // output_bits
    if image.dim[1] % input_multiple:
        raise NotImplementedError(f"frost_conv requires input channels to be a multiple of {input_multiple} for {input_dtype}; got {image.dim[1]}")
    if weight.dim[0] % output_multiple:
        raise NotImplementedError(f"frost_conv requires output channels to be a multiple of {output_multiple} for {output_dtype}; got {weight.dim[0]}")


def _check_channels_last_3d_layout(tensors, *, logical: bool = False) -> None:
    qualifier = "logical " if logical else ""
    for role, tensor in tensors:
        expected_stride = _channels_last_3d_stride(tensor.dim)
        if tuple(tensor.stride) != expected_stride:
            raise NotImplementedError(
                f"frost_conv: {role} must use compact {qualifier}channels-last layout; expected stride {expected_stride}, got {tuple(tensor.stride)}"
            )


def _check_fp32_compute_type(context: GraphContext):
    if context.compute_data_type != cudnn.data_type.FLOAT:
        raise NotImplementedError("frost conv: compute type must be FP32")


class _Sm100FrostConvPlan(CompiledPlan):
    """A shape-specialized convolution callable plus its graph port binding. Correspond to templates.sm100_conv.py"""

    takes_variant_pack = True

    @staticmethod
    def check_support(analysis: ConvGraph) -> None:
        _check_cutedsl("frost_conv dense convolution", DENSE_CUTEDSL_MIN_VERSION)
        _check_sm100()

        node = analysis.conv_node
        image, weight, output = analysis.image, analysis.weight, analysis.output
        tensors = (image, weight, output)
        _check_5d_inputs(tuple(zip(("image", "weight", "Y"), tensors)))
        _check_matching_channels(image, weight)
        _check_conv_mode(node)
        expected_output = _check_3d_conv_geometry(node, image, weight)

        image_dtype = _storage_dtype_name(image.data_type)
        weight_dtype = _storage_dtype_name(weight.data_type)
        output_dtype = _storage_dtype_name(output.data_type)
        supported = ", ".join(sorted(_CUTLASS_STORAGE_DTYPES.keys()))
        if image_dtype is None:
            raise NotImplementedError(f"frost_conv: unsupported image dtype {image.data_type}; supported dtypes: {supported}")
        if weight_dtype is None:
            raise NotImplementedError(f"frost_conv: unsupported weight dtype {weight.data_type}; supported dtypes: {supported}")
        if image.data_type != weight.data_type:
            raise NotImplementedError(f"frost_conv requires image and weight to have the same dtype; got {image.data_type} and {weight.data_type}")
        if output_dtype is None:
            raise NotImplementedError(f"frost_conv: unsupported output dtype {output.data_type}; supported dtypes: {supported}")

        _check_alignment(
            image,
            weight,
            input_bits=8 * buffers.DTYPE_ITEMSIZE[image_dtype],
            output_bits=8 * buffers.DTYPE_ITEMSIZE[output_dtype],
            input_dtype=image_dtype,
            output_dtype=output_dtype,
        )
        input_channel_tile = _input_channel_tile(image_dtype)
        if image.dim[1] % input_channel_tile:
            raise NotImplementedError(
                f"frost_conv requires input channels to be a multiple of the {input_channel_tile}-element mainloop tile for {image_dtype}; got {image.dim[1]}"
            )

        _check_output_shapes(expected_output, (("output", output),))

        if node.compute_data_type not in (None, cudnn.data_type.FLOAT):
            raise NotImplementedError("frost_conv requires FP32 compute")
        if analysis.epilogue_node is not None and analysis.epilogue_node.compute_data_type not in (None, cudnn.data_type.FLOAT):
            raise NotImplementedError("frost_conv requires FP32 unary epilogue compute")
        _check_fp32_compute_type(analysis.context)
        if analysis.epilogue_node is not None and analysis.context.intermediate_data_type != cudnn.data_type.FLOAT:
            raise NotImplementedError(
                "frost_conv requires a FLOAT convolution intermediate for unary fusion because the backend applies the epilogue "
                f"directly to its FP32 accumulator; got {analysis.context.intermediate_data_type}"
            )

        _check_channels_last_3d_layout(tuple(zip(("image", "weight", "Y"), tensors)))

    def __init__(self, compiled, tensors):
        self._compiled = compiled
        self._tensors = tuple(tensors)
        self._shapes = tuple(tuple(t.dim) for t in tensors)
        self._dtypes = tuple(_storage_dtype_name(t.data_type) for t in tensors)
        self._indices = None

    def execute(self, graph, variant_pack, ctx: ExecutionContext) -> None:
        if self._indices is None:
            try:
                self._indices = [variant_pack.index_of(tensor.uid) for tensor in self._tensors]
            except KeyError as exc:
                raise ValueError(f"frost_conv: tensor uid {exc} is bound by the kernel but is not an operand of this graph") from exc

        operands = variant_pack.operands(self._indices)
        for role, operand, shape, dtype in zip(("image", "weight", "Y"), operands, self._shapes, self._dtypes):
            actual_shape = tuple(operand.shape)
            actual_stride = tuple(operand.stride())
            expected_stride = _channels_last_3d_stride(shape)
            if actual_shape != shape or actual_stride != expected_stride:
                raise ValueError(
                    f"frost_conv: {role} must have shape {shape} and compact channels-last stride {expected_stride}; "
                    f"got shape {actual_shape}, stride {actual_stride}"
                )
            if operand.dtype != dtype:
                raise ValueError(f"frost_conv: {role} must have dtype {dtype}; got {operand.dtype}")
            if operand.data_ptr() % 16:
                raise ValueError(f"frost_conv: {role} must be 16-byte aligned")

        image, weight, output = operands
        # The graph follows PyTorch/cuDNN's NCDHW/KCTRS convention, while the
        # im2col template consumes zero-copy NDHWC/KTRSC/NZPQK views.
        self._compiled(
            image.permute(0, 2, 3, 4, 1),
            weight.permute(0, 2, 3, 4, 1),
            output.permute(0, 2, 3, 4, 1),
            _cuda.CUstream(ctx.stream or 0),
        )


class _Sm100FrostBlockScaleConvPlan(CompiledPlan):
    """A graph-specialized NVFP4 convolution and its bound buffers. Corresponds to templates.sm100_block_scale_conv.py."""

    takes_variant_pack = True

    def __init__(self, compiled, analysis: ConvGraph, geometry, device: int):
        self._compiled = compiled
        block_scale = analysis.block_scale_data
        if block_scale is None:
            raise ValueError("frost_conv: block-scale plan requires block-scale graph metadata")
        output_quantize = analysis.output_quantize_data
        self._tensors = (analysis.image, analysis.weight, block_scale.sfa, block_scale.sfb, analysis.output)
        if output_quantize is not None:
            self._tensors += (output_quantize.scale,)
        self._indices = None
        self._device = int(device)

        image_shape = tuple(int(x) for x in analysis.image.dim)
        weight_shape = tuple(int(x) for x in analysis.weight.dim)
        output_shape = tuple(int(x) for x in analysis.output.dim)
        n, c, d, h, w = image_shape
        k, _c, t, r, s = weight_shape
        sf_c = _ceil_div(c, 16)
        sf_c_padded = _ceil_div(sf_c, 4) * 4
        sfb_k = _ceil_div(c * t * r * s, 16)
        sfb_bytes = 512 * _ceil_div(k, 128) * _ceil_div(sfb_k, 4)

        output_dtype_name, output_bits = _BLOCK_SCALE_OUTPUT_SPECS[analysis.output.data_type]
        physical_output_shape = _physical_channel_shape(output_shape, output_bits)
        output_dtypes = _runtime_dtype_names(output_dtype_name, output_bits)
        output_bytes = _logical_storage_bytes(output_shape, output_bits)

        self._expected = (
            (
                "A",
                _physical_channel_shape(image_shape, 4),
                _channels_last_3d_stride(_physical_channel_shape(image_shape, 4)),
                _FP4_RUNTIME_DTYPES,
                _logical_storage_bytes(image_shape, 4),
            ),
            (
                "B",
                _physical_channel_shape(weight_shape, 4),
                _channels_last_3d_stride(_physical_channel_shape(weight_shape, 4)),
                _FP4_RUNTIME_DTYPES,
                _logical_storage_bytes(weight_shape, 4),
            ),
            ("SFA", (n * d * h * w, sf_c_padded, 1), (sf_c_padded, 1, 1), frozenset(("float8_e4m3fn",)), n * d * h * w * sf_c_padded),
            ("SFB", (sfb_bytes,), (1,), frozenset(("float8_e4m3fn",)), sfb_bytes),
            (
                "D",
                physical_output_shape,
                _channels_last_3d_stride(physical_output_shape),
                output_dtypes,
                output_bytes,
            ),
        )
        if output_bits == 4:
            output_pixels = output_shape[0] * output_shape[2] * output_shape[3] * output_shape[4]
            output_scale_channels = output_shape[1] // 16
            self._expected += (
                (
                    "SFD",
                    (output_pixels, output_scale_channels, 1),
                    (output_scale_channels, 1, 1),
                    frozenset(("float8_e4m3fn",)),
                    output_pixels * output_scale_channels,
                ),
            )
        self._upper_padding, self._lower_padding, self._stride, self._dilation = geometry

    def execute(self, graph, variant_pack, ctx: ExecutionContext) -> None:
        launch_device = current_device()
        if launch_device != self._device:
            raise ValueError(f"frost_conv: plan was built for cuda:{self._device}, but execute would launch on cuda:{launch_device}")

        if self._indices is None:
            try:
                self._indices = [variant_pack.index_of(tensor.uid) for tensor in self._tensors]
            except KeyError as exc:
                raise ValueError(f"frost_conv: tensor uid {exc} is bound by the kernel but is not an operand of this graph") from exc

        operands = variant_pack.operands(self._indices)
        for operand, (role, shape, stride, dtypes, required_bytes) in zip(operands, self._expected):
            actual_shape = tuple(int(x) for x in operand.shape)
            actual_stride = tuple(int(x) for x in operand.stride())
            if actual_shape != shape or actual_stride != stride:
                raise ValueError(
                    f"frost_conv: {role} must have physical shape {shape} and stride {stride}; " f"got shape {actual_shape}, stride {actual_stride}"
                )
            if operand.dtype not in dtypes:
                expected_dtype = "native float4_e2m1fn_x2" if "float4_e2m1fn_x2" in dtypes else next(iter(dtypes))
                raise ValueError(f"frost_conv: {role} must have dtype {expected_dtype}; got {operand.dtype}")
            if int(operand.numel()) != _dense_span(actual_shape, actual_stride):
                raise ValueError(f"frost_conv: {role} must occupy one dense canonical span")
            actual_bytes = _operand_storage_bytes(operand)
            if actual_bytes != required_bytes:
                raise ValueError(f"frost_conv: {role} must contain exactly {required_bytes} bytes; got {actual_bytes}")
            if int(operand.data_ptr()) % 16:
                raise ValueError(f"frost_conv: {role} must be 16-byte aligned")

        image, weight, sfa, sfb, output, *output_scales = operands
        output_scale = output_scales[0] if output_scales else None
        self._compiled(
            image.permute(0, 2, 3, 4, 1),
            weight.permute(0, 2, 3, 4, 1),
            output.permute(0, 2, 3, 4, 1),
            sfa,
            sfb,
            cutlass.Float32(1.0),
            output_scale,
            cutlass.Float32(1.0),
            None,
            None,
            *(cutlass.Int32(value) for value in (*self._upper_padding, *self._lower_padding, *self._stride, *self._dilation)),
            _cuda.CUstream(ctx.stream or 0),
        )

    @staticmethod
    def _check_graph_support(analysis) -> None:
        node = analysis.conv_node
        block_scale = analysis.block_scale_data
        if block_scale is None:
            raise NotImplementedError("frost_conv: missing block-scale graph metadata")
        image, weight, sfa, sfb, output = analysis.image, analysis.weight, block_scale.sfa, block_scale.sfb, analysis.output
        output_quantize = analysis.output_quantize_data
        output_scale = output_quantize.scale if output_quantize is not None else None
        conv_output = node.outputs["Y"]
        epilogue_output = analysis.epilogue_node.outputs["OUT_0"] if analysis.epilogue_node is not None else conv_output
        graph_tensors = (
            ("A", image),
            ("B", weight),
            ("SFA", sfa),
            ("SFB", sfb),
            ("dequantized A", block_scale.dequantized_image),
            ("dequantized B", block_scale.dequantized_weight),
            ("convolution result", conv_output),
            ("epilogue result", epilogue_output),
            ("D", output),
        ) + ((("SFD", output_scale),) if output_scale is not None else ())

        if analysis.epilogue_node is not None and analysis.epilogue_node.compute_data_type not in (None, cudnn.data_type.FLOAT):
            raise NotImplementedError("frost_conv: NVFP4 convolution requires FP32 pointwise compute")

        _check_5d_inputs(graph_tensors)
        _check_matching_channels(image, weight)
        c = int(image.dim[1])
        if c % 64 or (c > 256 and c % 256):
            raise NotImplementedError(f"frost_conv: unsupported NVFP4 input channels C={c}; C must be 64, 128, 192, 256, or a positive multiple of 256")
        cta_tile_k = min(c, 256)
        if (cta_tile_k == 256 and c % 256) or (cta_tile_k != 256 and c != cta_tile_k):
            raise NotImplementedError(f"frost_conv: illegal CTA-K={cta_tile_k} for C={c}")
        _check_fp32_compute_type(analysis.context)
        if image.data_type != cudnn.data_type.FP4_E2M1 or weight.data_type != cudnn.data_type.FP4_E2M1:
            raise NotImplementedError("frost_conv: block-scale A and B must use FP4_E2M1 storage")
        if sfa.data_type != cudnn.data_type.FP8_E4M3 or sfb.data_type != cudnn.data_type.FP8_E4M3:
            raise NotImplementedError("frost_conv: block-scale SFA and SFB must use FP8_E4M3 storage")
        if output_quantize is None:
            if output.data_type == cudnn.data_type.FP4_E2M1:
                raise NotImplementedError("frost_conv: FP4_E2M1 D requires a terminal block_scale_quantize and E4M3 SFD")
            if output.data_type not in _BLOCK_SCALE_OUTPUT_SPECS:
                supported = "FLOAT, HALF, BFLOAT16, FP8_E4M3, FP8_E5M2, or FP4_E2M1 with block_scale_quantize"
                raise NotImplementedError(f"frost_conv: unsupported block-scaled convolution output dtype {output.data_type}; expected {supported}")
        else:
            quantize = output_quantize.quantize_node
            quant_block_size = quantize.params.get("block_size")
            if quant_block_size not in (16, (16,), [16]):
                raise NotImplementedError(f"frost_conv: NVFP4 output quantization requires block_size=16; got {quant_block_size}")
            if quantize.params.get("axis") != 1:
                raise NotImplementedError("frost_conv: NVFP4 output quantization requires axis=1 (the output-channel axis)")
            if quantize.params.get("transpose") not in (None, False):
                raise NotImplementedError("frost_conv: NVFP4 output quantization does not support transpose=True")
            if output_quantize.group_offset is not None:
                raise NotImplementedError("frost_conv: NVFP4 output quantization does not support group_offset")
            if quantize.compute_data_type not in (None, cudnn.data_type.FLOAT):
                raise NotImplementedError("frost_conv: NVFP4 output quantization requires FP32 compute")
            if output_quantize.input.data_type != cudnn.data_type.FLOAT:
                raise NotImplementedError("frost_conv: NVFP4 output quantization requires a FLOAT convolution intermediate")
            if output.data_type != cudnn.data_type.FP4_E2M1 or output_scale.data_type != cudnn.data_type.FP8_E4M3:
                raise NotImplementedError("frost_conv: NVFP4 output requires FP4_E2M1 D and FP8_E4M3 SFD")
            if output.dim[1] % 64:
                raise NotImplementedError(f"frost_conv: NVFP4 output channels must be a multiple of 64; got {output.dim[1]}")

        expected_block_size = (1, 16, 1, 1, 1)
        for role, dequant in (("A", block_scale.image_dequant_node), ("B", block_scale.weight_dequant_node)):
            block_size = tuple(dequant.params.get("block_size") or ())
            if block_size != expected_block_size:
                raise NotImplementedError(f"frost_conv: {role} block_scale_dequantize requires block_size={expected_block_size}; got {block_size}")
            if dequant.params.get("is_negative_scale", False):
                raise NotImplementedError(f"frost_conv: {role} block_scale_dequantize does not support negative scale encoding")
            if dequant.compute_data_type not in (None, cudnn.data_type.FLOAT):
                raise NotImplementedError("frost_conv: block-scale dequantization requires FP32 compute")

        if node.compute_data_type not in (None, cudnn.data_type.FLOAT):
            raise NotImplementedError("frost_conv: NVFP4 convolution requires FP32 compute")
        if block_scale.dequantized_image.data_type != cudnn.data_type.FLOAT or block_scale.dequantized_weight.data_type != cudnn.data_type.FLOAT:
            raise NotImplementedError("frost_conv: block-scale dequantized A and B must be FP32 virtual intermediates")
        if tuple(block_scale.dequantized_image.dim) != tuple(image.dim) or tuple(block_scale.dequantized_weight.dim) != tuple(weight.dim):
            raise NotImplementedError("frost_conv: block-scale dequantized A and B must preserve their input dimensions")
        if analysis.epilogue_node is not None and conv_output.data_type != cudnn.data_type.FLOAT:
            raise NotImplementedError("frost_conv: an explicit unary epilogue requires a FLOAT convolution intermediate")

        _check_conv_mode(node)
        expected_output = _check_3d_conv_geometry(node, image, weight)
        _check_output_shapes(expected_output, (("convolution result", conv_output), ("epilogue result", epilogue_output), ("D", output)))

        output_dtype_name, output_bits = _BLOCK_SCALE_OUTPUT_SPECS[output.data_type]
        _check_alignment(
            image,
            weight,
            input_bits=4,
            output_bits=output_bits,
            input_dtype="float4_e2m1fn_x2",
            output_dtype=output_dtype_name,
        )

        expected_sfa = (image.dim[0], image.dim[1] // 16, *image.dim[2:])
        expected_sfb = (weight.dim[0], weight.dim[1] // 16, *weight.dim[2:])
        if tuple(sfa.dim) != expected_sfa or tuple(sfb.dim) != expected_sfb:
            raise NotImplementedError(
                f"frost_conv: scale shapes must be SFA={expected_sfa} and SFB={expected_sfb}; " f"got SFA={tuple(sfa.dim)}, SFB={tuple(sfb.dim)}"
            )

        if output_scale is not None:
            expected_sfd = (output.dim[0], output.dim[1] // 16, *output.dim[2:])
            if tuple(output_scale.dim) != expected_sfd:
                raise NotImplementedError(f"frost_conv: NVFP4 output scale shape must be SFD={expected_sfd}; got {tuple(output_scale.dim)}")

        layout_tensors = (
            ("A", image),
            ("B", weight),
            ("SFA", sfa),
            ("SFB", sfb),
            ("D", output),
        )
        if output_scale is not None:
            layout_tensors += (("SFD", output_scale),)
        _check_channels_last_3d_layout(layout_tensors, logical=True)
        if _reordering_name(sfa) is not None:
            raise NotImplementedError("frost_conv: SFA must use the unswizzled NONE layout")
        if _reordering_name(sfb) != "F8_128x4":
            raise NotImplementedError("frost_conv: SFB must declare F8_128x4 reordering")
        if output_scale is not None and _reordering_name(output_scale) is not None:
            raise NotImplementedError("frost_conv: SFD must use the unswizzled NONE layout")

    @classmethod
    def check_support(cls, analysis: ConvGraph) -> None:
        _check_cutedsl("frost_conv block-scale convolution", BLOCK_SCALE_CUTEDSL_MIN_VERSION)
        _check_sm100()
        cls._check_graph_support(analysis)


class FrostConvEngine(BaseEngine):
    """Dispatch graph analysis to a plan-specific Frost convolution contract."""

    name = "frost_conv"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph: "pygraph") -> None:
        analysis = analyze(graph)
        plan_type = _Sm100FrostBlockScaleConvPlan if analysis.block_scale_data is not None else _Sm100FrostConvPlan
        plan_type.check_support(analysis)

    def build_plan(self, graph: "pygraph", plan: PlanConfig, ctx: ExecutionContext = None) -> CompiledPlan:
        analysis = analyze(graph)
        node = analysis.conv_node
        image, weight, output = analysis.image, analysis.weight, analysis.output
        ncdhw = tuple(image.dim)
        ktrs = (weight.dim[0], weight.dim[2], weight.dim[3], weight.dim[4])
        pre_padding = _tuple_param(node, "pre_padding", _ZERO)
        post_padding = _tuple_param(node, "post_padding", _ZERO)
        stride = _tuple_param(node, "stride", _ONE)
        dilation = _tuple_param(node, "dilation", _ONE)
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            if analysis.block_scale_data is not None:
                _check_cutedsl("frost_conv block-scale convolution", BLOCK_SCALE_CUTEDSL_MIN_VERSION)
                # Keep compiler-template imports lazy so engine discovery does not load them.
                from .templates.sm100_block_scale_conv import compile

                compiled = compile(
                    ncdhw=ncdhw,
                    ktrs=ktrs,
                    upper_padding_dhw=post_padding,
                    lower_padding_dhw=pre_padding,
                    stride_dhw=stride,
                    dilation_dhw=dilation,
                    epilogue=analysis.epilogue,
                    d_dtype=_cutlass_block_scale_output_dtype(output.data_type),
                    epilogue_attrs=analysis.epilogue_attrs,
                )
                return _Sm100FrostBlockScaleConvPlan(
                    compiled,
                    analysis,
                    (post_padding, pre_padding, stride, dilation),
                    current_device(),
                )
            else:
                _check_cutedsl("frost_conv dense convolution", DENSE_CUTEDSL_MIN_VERSION)
                # Keep compiler-template imports lazy so engine discovery does not load them.
                from .templates.sm100_conv import compile

                ab_dtype = _CUTLASS_STORAGE_DTYPES[_storage_dtype_name(image.data_type)]
                c_dtype = _CUTLASS_STORAGE_DTYPES[_storage_dtype_name(output.data_type)]
                compiled = compile(
                    ncdhw=ncdhw,
                    ktrs=ktrs,
                    upper_padding_dhw=post_padding,
                    lower_padding_dhw=pre_padding,
                    stride_dhw=stride,
                    dilation_dhw=dilation,
                    epilogue=analysis.epilogue,
                    epilogue_attrs=analysis.epilogue_attrs,
                    ab_dtype=ab_dtype,
                    c_dtype=c_dtype,
                )
                return _Sm100FrostConvPlan(compiled, (image, weight, output))


def FrostConvEngines(ids: dict[str, int]) -> list[BaseEngine]:
    """The convolution engines the manifest asked for, with assigned ids."""
    out = []
    for cls in (FrostConvEngine,):
        if cls.name in ids:
            engine = cls()
            engine.engine_id = ids[cls.name]
            out.append(engine)
    return out
