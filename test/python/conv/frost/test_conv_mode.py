# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convolution mode support for dense and block-scaled Frost graphs."""

from contextlib import nullcontext

import cudnn
import pytest

from cudnn.conv.frost.engine import FrostConvEngine
from utils import requires_block_scale_cutedsl, requires_dense_cutedsl, requires_sm100

pytestmark = [pytest.mark.L0, requires_sm100]


@pytest.mark.parametrize(
    "block_scaled",
    (
        pytest.param(False, marks=requires_dense_cutedsl, id="dense"),
        pytest.param(True, marks=requires_block_scale_cutedsl, id="block-scaled"),
    ),
)
@pytest.mark.parametrize("mode", (None, "CROSS_CORRELATION", "CONVOLUTION"), ids=("default", "cross-correlation", "convolution"))
def test_frost_conv_convolution_mode(block_scaled, mode) -> None:
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def tensor(name, shape, dtype, **kwargs):
        n, c, d, h, w = shape
        return graph.tensor(name=name, dim=shape, stride=(c * d * h * w, 1, h * w * c, w * c, c), data_type=dtype, **kwargs)

    input_dtype = cudnn.data_type.FP4_E2M1 if block_scaled else cudnn.data_type.BFLOAT16
    image = tensor("image", (1, 128, 4, 8, 8), input_dtype)
    weight = tensor("weight", (128, 128, 3, 3, 3), input_dtype)
    if block_scaled:
        sfa = tensor("SFA", (1, 8, 4, 8, 8), cudnn.data_type.FP8_E4M3)
        sfb = tensor("SFB", (128, 8, 3, 3, 3), cudnn.data_type.FP8_E4M3, reordering_type=cudnn.tensor_reordering.F8_128x4)
        image = graph.block_scale_dequantize(input=image, descale=sfa, block_size=(1, 16, 1, 1, 1))
        weight = graph.block_scale_dequantize(input=weight, descale=sfb, block_size=(1, 16, 1, 1, 1))

    mode_kwargs = {} if mode is None else {"convolution_mode": getattr(cudnn._pybind_module.convolution_mode, mode)}
    output = graph.conv_fprop(
        image,
        weight,
        pre_padding=(1, 1, 1),
        post_padding=(1, 1, 1),
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        **mode_kwargs,
    )
    output.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim((1, 128, 4, 8, 8)).set_stride((32768, 1, 8192, 1024, 128))
    graph.validate()

    expectation = (
        pytest.raises(NotImplementedError, match=r"frost_conv: expect convolution mode to be .*CROSS_CORRELATION but got .*CONVOLUTION")
        if mode == "CONVOLUTION"
        else nullcontext()
    )
    with expectation:
        FrostConvEngine().check_support(graph)
