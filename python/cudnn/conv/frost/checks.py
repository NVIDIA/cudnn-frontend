# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared admission checks for FROST convolution engines."""

import cudnn
from cudnn import GraphContext
from cudnn.frost import buffers

from .utils import _ONE, _ZERO, _channels_last_3d_stride, _expected_output_shape, _tuple_param

BLOCK_SCALE_CUTEDSL_MIN_VERSION = (4, 8)
DENSE_CUTEDSL_MIN_VERSION = (4, 9)


def _check_sm100():
    sm = buffers.current_sm()
    if sm not in (100, 103):  # the SM100-family kernels compile for the device's own target
        raise NotImplementedError(f"frost_conv requires SM100 or SM103, but found {sm}")


def _check_cutedsl(what: str, minimum_version: tuple[int, int]) -> None:
    error = buffers.cutedsl_requirement_error(what, minimum_version, require_installed=True)
    if error is not None:
        raise NotImplementedError(error)


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
