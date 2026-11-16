# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared convolution geometry, layout, and storage helpers."""

from typing import Optional, Sequence

import cudnn
import cutlass
from cudnn.datatypes import _cudnn_to_frost_dtype_name
from cudnn.frost import buffers

_ZERO = (0, 0, 0)
_ONE = (1, 1, 1)
_CUTLASS_STORAGE_DTYPES = {
    "float16": cutlass.Float16,
    "bfloat16": cutlass.BFloat16,
    "float32": cutlass.Float32,
    "float8_e4m3fn": cutlass.Float8E4M3FN,
    "float8_e5m2": cutlass.Float8E5M2,
}
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
    return 128 // buffers.DTYPE_ITEMSIZE[dtype_name]


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


def _physical_channel_shape(shape: Sequence[int], bits: int) -> tuple[int, ...]:
    """Translate a logical channels-first graph shape to its runtime carrier."""
    shape = tuple(int(extent) for extent in shape)
    if bits != 4:
        return shape
    if shape[1] % 2:
        raise ValueError(f"frost_conv: packed FP4 channel extent must be even; got {shape[1]}")
    return (shape[0], shape[1] // 2, *shape[2:])


def _cutlass_block_scale_output_dtype(data_type):
    if data_type == cudnn.data_type.FP4_E2M1:
        return cutlass.Float4E2M1FN

    dtype_name = _storage_dtype_name(data_type)
    if dtype_name is None:
        raise NotImplementedError(f"frost_conv: unsupported block-scaled convolution output dtype {data_type}")
    return _CUTLASS_STORAGE_DTYPES[dtype_name]
