# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A terminal block-scale quantize is only fusable by the block-scaled plan."""

import cudnn
import pytest

from cudnn.conv.frost.engine import FrostConvEngine
from utils import requires_dense_cutedsl, requires_sm100

pytestmark = [pytest.mark.L0, requires_sm100, requires_dense_cutedsl]

_BLOCK_SIZE = 16


def _channels_last_stride(shape) -> tuple[int, ...]:
    _n, c, d, h, w = shape
    return (c * d * h * w, 1, h * w * c, w * c, c)


@pytest.mark.parametrize("epilogue", (False, True), ids=("no-epilogue", "relu"))
def test_dense_conv_rejects_output_block_scale_quantize(epilogue) -> None:
    """A quantize fused onto a graph with dense (non-dequantized) inputs must be declined.

    The dense template writes its epilogue result straight to Y and cannot emit a
    scale tensor, so accepting this graph would silently drop the quantization and
    leave the scale output unwritten.
    """
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    image_shape = (1, 128, 1, 8, 8)
    weight_shape = (128, 128, 3, 3, 3)
    output_shape = (1, 128, 1, 8, 8)

    image = graph.tensor(name="image", dim=image_shape, stride=_channels_last_stride(image_shape), data_type=cudnn.data_type.BFLOAT16)
    weight = graph.tensor(name="weight", dim=weight_shape, stride=_channels_last_stride(weight_shape), data_type=cudnn.data_type.BFLOAT16)

    result = graph.conv_fprop(
        image,
        weight,
        pre_padding=(1, 1, 1),
        post_padding=(1, 1, 1),
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
    )
    if epilogue:
        result = graph.relu(result, compute_data_type=cudnn.data_type.FLOAT, name="relu")
    result.set_dim(output_shape).set_stride(_channels_last_stride(output_shape)).set_data_type(cudnn.data_type.FLOAT)

    output, output_scale = graph.block_scale_quantize(input=result, block_size=_BLOCK_SIZE, axis=1, name="quantize_D")
    output.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3).set_dim(output_shape).set_stride(_channels_last_stride(output_shape))
    scale_shape = (output_shape[0], output_shape[1] // _BLOCK_SIZE, *output_shape[2:])
    output_scale.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3).set_dim(scale_shape).set_stride(_channels_last_stride(scale_shape))
    graph.validate()

    with pytest.raises(NotImplementedError, match=r"fused block-scale quantize .* only supported when both convolution inputs"):
        FrostConvEngine().check_support(graph)
