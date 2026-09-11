# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

import test_conv_fuzzer as conv_fuzzer
from sdpa.helpers import create_sparse_int_tensor


@dataclass(frozen=True)
class FixedConvCase:
    config: conv_fuzzer.ConvConfig
    rtol: float = 1e-2
    atol: float = 1e-2
    accumulate_output: bool = False


@pytest.fixture
def num_diffs(request):
    return request.config.getoption("--diffs")


def _fixed_conv_case(
    *,
    spatial_dims: int,
    batch: int,
    in_channels: int,
    out_channels: int,
    input_spatial,
    filter_spatial,
    padding,
    stride,
    dilation,
    conv_type: conv_fuzzer.ConvType,
    dtype: torch.dtype,
    epilogue: conv_fuzzer.EpilogueType = conv_fuzzer.EpilogueType.NONE,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    accumulate_output: bool = False,
    rng_seed: int,
) -> FixedConvCase:
    config = conv_fuzzer.ConvConfig(
        spatial_dims=spatial_dims,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        groups=1,
        input_spatial=input_spatial,
        filter_spatial=filter_spatial,
        padding=padding,
        stride=stride,
        dilation=dilation,
        conv_type=conv_type,
        x_dtype=dtype,
        w_dtype=dtype,
        y_dtype=dtype,
        epilogue=epilogue,
        rng_seed=rng_seed,
    )
    return FixedConvCase(config=config, rtol=rtol, atol=atol, accumulate_output=accumulate_output)


FIXED_CONV_CASES_L0 = tuple(
    pytest.param(
        _fixed_conv_case(
            spatial_dims=2,
            batch=2,
            in_channels=8,
            out_channels=8,
            input_spatial=(15, 20),
            filter_spatial=(3, 3),
            padding=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            conv_type=conv_type,
            dtype=torch.float32,
            accumulate_output=True,
            rng_seed=100 + int(conv_type),
        ),
        id=f"conv2d_{conv_fuzzer.conv_type_name(conv_type)}_accum_f32",
    )
    for conv_type in (conv_fuzzer.ConvType.FPROP, conv_fuzzer.ConvType.DGRAD, conv_fuzzer.ConvType.WGRAD)
) + (
    pytest.param(
        _fixed_conv_case(
            spatial_dims=2,
            batch=5,
            in_channels=32,
            out_channels=32,
            input_spatial=(32, 32),
            filter_spatial=(3, 3),
            padding=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            conv_type=conv_fuzzer.ConvType.FPROP,
            dtype=torch.float16,
            epilogue=conv_fuzzer.EpilogueType.BIAS,
            rtol=2.5e-3,
            atol=2.5e-3,
            rng_seed=103,
        ),
        id="conv2d_bias_f16",
    ),
)


FIXED_CONV_CASES_L1 = tuple(
    pytest.param(
        _fixed_conv_case(
            spatial_dims=3,
            batch=2,
            in_channels=8,
            out_channels=8,
            input_spatial=(15, 20, 25),
            filter_spatial=(3, 3, 3),
            padding=(1, 1, 1),
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            conv_type=conv_type,
            dtype=torch.float32,
            accumulate_output=True,
            rng_seed=104 + int(conv_type),
        ),
        id=f"conv3d_{conv_fuzzer.conv_type_name(conv_type)}_accum_f32",
    )
    for conv_type in (conv_fuzzer.ConvType.FPROP, conv_fuzzer.ConvType.DGRAD, conv_fuzzer.ConvType.WGRAD)
) + tuple(
    pytest.param(
        _fixed_conv_case(
            spatial_dims=3,
            batch=batch,
            in_channels=3,
            out_channels=out_channels,
            input_spatial=(32, 32, 32),
            filter_spatial=(3, 3, 3),
            padding=(3, 0, 0),
            stride=(7, 1, 1),
            dilation=(1, 1, 1),
            conv_type=conv_fuzzer.ConvType.FPROP,
            dtype=torch.float16,
            rtol=2.5e-3,
            atol=2.5e-3,
            rng_seed=107 + batch_index * 3 + channel_index,
        ),
        id=f"conv3d_n{batch}_k{out_channels}_f16",
    )
    for batch_index, batch in enumerate((4, 24, 31))
    for channel_index, out_channels in enumerate((32, 24, 9))
)


def _create_prior_output(conv_case: FixedConvCase, X: torch.Tensor, W: torch.Tensor, Y: torch.Tensor) -> Optional[torch.Tensor]:
    """Create an independently seeded initial value for output accumulation."""
    if not conv_case.accumulate_output:
        return None

    config = conv_case.config
    if config.conv_type == conv_fuzzer.ConvType.FPROP:
        destination = Y
    elif config.conv_type == conv_fuzzer.ConvType.DGRAD:
        destination = X
    else:
        destination = W

    generator = torch.Generator(device="cuda")
    generator.manual_seed(config.rng_seed + 10_000)
    memory_format = torch.channels_last if config.spatial_dims == 2 else torch.channels_last_3d
    return create_sparse_int_tensor(destination.size(), destination.dtype, generator, memory_format=memory_format)


def _run_fixed_conv_test(conv_case: FixedConvCase, cudnn_handle, num_diffs: int) -> None:
    config = conv_case.config
    X = W = Y = bias = prior_output = reference = None
    try:
        X, W, Y, bias = conv_fuzzer.create_tensors(config, random.Random(config.rng_seed))
        prior_output = _create_prior_output(conv_case, X, W, Y)

        execution_succeeded, execution_message = conv_fuzzer.run_cudnn_conv(config, X, W, Y, bias, cudnn_handle, prior_output=prior_output)
        if not execution_succeeded:
            pytest.fail(execution_message)

        reference = conv_fuzzer.compute_reference(config, X, W, Y, bias, prior_output=prior_output)
        if config.conv_type == conv_fuzzer.ConvType.FPROP:
            actual_output, output_dtype, output_name = Y, config.y_dtype, "Y"
        elif config.conv_type == conv_fuzzer.ConvType.DGRAD:
            actual_output, output_dtype, output_name = X, config.x_dtype, "dX"
        else:
            actual_output, output_dtype, output_name = W, config.w_dtype, "dW"

        comparison_passed, comparison_message = conv_fuzzer.compare_results(
            actual_output,
            reference,
            output_dtype,
            num_diffs,
            rtol=conv_case.rtol,
            atol=conv_case.atol,
        )
        assert comparison_passed, f"{output_name} numerical mismatch: {comparison_message}"
    finally:
        del X, W, Y, bias, prior_output, reference
        torch.cuda.empty_cache()


@pytest.mark.L0
@pytest.mark.parametrize("conv_case", FIXED_CONV_CASES_L0)
def test_conv_fixed_L0(conv_case: FixedConvCase, cudnn_handle, num_diffs):
    _run_fixed_conv_test(conv_case, cudnn_handle, num_diffs)


@pytest.mark.L1
@pytest.mark.parametrize("conv_case", FIXED_CONV_CASES_L1)
def test_conv_fixed_L1(conv_case: FixedConvCase, cudnn_handle, num_diffs):
    _run_fixed_conv_test(conv_case, cudnn_handle, num_diffs)
