# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from dataclasses import dataclass, replace
from typing import Optional

import pytest
import torch
import cudnn

import test_conv_fuzzer as conv_fuzzer
from sdpa.helpers import create_sparse_int_tensor


@dataclass(frozen=True)
class FixedConvCase:
    config: conv_fuzzer.ConvConfig
    rtol: float = 1e-2
    atol: float = 1e-2
    accumulate_output: bool = False
    intermediate_dtype: Optional[torch.dtype] = None
    materialize_intermediate: bool = False
    min_device_capability: tuple = (0, 0)


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
    intermediate_dtype: Optional[torch.dtype] = None,
    materialize_intermediate: bool = False,
    min_device_capability: tuple = (0, 0),
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
    return FixedConvCase(
        config=config,
        rtol=rtol,
        atol=atol,
        accumulate_output=accumulate_output,
        intermediate_dtype=intermediate_dtype,
        materialize_intermediate=materialize_intermediate,
        min_device_capability=min_device_capability,
    )


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


# Name, N, C, K, input spatial, filter spatial, padding, stride, dilation, seed base.
WGRAD_RESIDUAL_SHAPES = (
    ("n64_c256_k16_7x7", 64, 256, 16, (7, 7), (1, 1), (0, 0), (1, 1), (1, 1), 212),
    ("n76_c80_k240_34x34", 76, 80, 240, (34, 34), (1, 1), (0, 0), (1, 1), (1, 1), 218),
    ("n17_c16_k80_28x28", 17, 16, 80, (28, 28), (3, 3), (1, 1), (1, 1), (1, 1), 224),
    ("n31_c32_k96_500x1", 31, 32, 96, (500, 1), (5, 1), (2, 0), (1, 1), (1, 1), 230),
    ("n65_c32_k112_1x700", 65, 32, 112, (1, 700), (1, 7), (0, 3), (1, 1), (1, 1), 236),
    ("n15_c16_k1040_28x28", 15, 16, 1040, (28, 28), (1, 1), (0, 0), (1, 1), (1, 1), 242),
    ("n30_c102_k16_7x7", 30, 102, 16, (7, 7), (1, 1), (0, 0), (1, 1), (1, 1), 248),
    ("n35_c64_k22_100x50", 35, 64, 22, (100, 50), (1, 1), (0, 0), (1, 1), (1, 1), 254),
)


FIXED_CONV_EXPANSION_CASES_L1 = tuple(
    pytest.param(
        _fixed_conv_case(
            spatial_dims=2,
            batch=8,
            in_channels=64,
            out_channels=64,
            input_spatial=(64, 64),
            filter_spatial=(3, 3),
            padding=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            conv_type=conv_type,
            dtype=dtype,
            epilogue=epilogue,
            accumulate_output=accumulate_output,
            intermediate_dtype=dtype if topology != "plain" else None,
            materialize_intermediate=materialize_intermediate,
            min_device_capability=min_device_capability,
            rtol=5e-3,
            atol=5e-3,
            rng_seed=seed_base + dtype_seed_offset,
        ),
        id=f"conv2d_{conv_fuzzer.conv_type_name(conv_type)}_{topology}_{dtype_name}",
    )
    for dtype, dtype_name, min_device_capability, dtype_seed_offset in ((torch.float16, "f16", (7, 0), 0), (torch.float32, "f32", (0, 0), 6))
    for conv_type, epilogue, accumulate_output, materialize_intermediate, topology, seed_base in (
        (conv_fuzzer.ConvType.FPROP, conv_fuzzer.EpilogueType.NONE, False, False, "plain", 200),
        (conv_fuzzer.ConvType.FPROP, conv_fuzzer.EpilogueType.RELU, False, False, "relu", 201),
        (conv_fuzzer.ConvType.WGRAD, conv_fuzzer.EpilogueType.RELU, False, True, "relu_exposed", 202),
        (conv_fuzzer.ConvType.FPROP, conv_fuzzer.EpilogueType.NONE, True, True, "add_exposed", 203),
        (conv_fuzzer.ConvType.DGRAD, conv_fuzzer.EpilogueType.NONE, True, True, "add_exposed", 204),
        (conv_fuzzer.ConvType.WGRAD, conv_fuzzer.EpilogueType.NONE, True, True, "add_exposed", 205),
    )
) + tuple(
    pytest.param(
        _fixed_conv_case(
            spatial_dims=2,
            batch=batch,
            in_channels=in_channels,
            out_channels=out_channels,
            input_spatial=input_spatial,
            filter_spatial=filter_spatial,
            padding=padding,
            stride=stride,
            dilation=dilation,
            conv_type=conv_fuzzer.ConvType.WGRAD,
            dtype=dtype,
            accumulate_output=True,
            intermediate_dtype=torch.float32,
            materialize_intermediate=materialize_intermediate,
            min_device_capability=min_device_capability,
            rtol=rtol,
            atol=atol,
            rng_seed=seed_base + dtype_seed_offset + int(materialize_intermediate),
        ),
        id=f"conv2d_wgrad_residual_{'exposed' if materialize_intermediate else 'virtual'}_{dtype_name}_{shape_name}",
    )
    for shape_name, batch, in_channels, out_channels, input_spatial, filter_spatial, padding, stride, dilation, seed_base in WGRAD_RESIDUAL_SHAPES
    for dtype, dtype_name, min_device_capability, rtol, atol, dtype_seed_offset in (
        (torch.float16, "f16", (7, 0), 5e-3, 5e-3, 0),
        # Integer inputs accumulate exactly in FP32; the reference includes BF16 output rounding.
        (torch.bfloat16, "bf16", (8, 0), 0, 0, 2),
        (torch.float32, "f32", (0, 0), 5e-3, 5e-3, 4),
    )
    for materialize_intermediate in (False, True)
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
    if torch.cuda.get_device_capability() < conv_case.min_device_capability:
        pytest.skip(f"Requires device capability {conv_case.min_device_capability}")
    X = W = Y = bias = prior_output = reference = actual_output = None
    intermediate_output = intermediate_reference = destination = None
    intermediate_dtype = conv_case.intermediate_dtype if conv_case.intermediate_dtype is not None else torch.float32
    original_stream = cudnn.get_stream(handle=cudnn_handle)
    try:
        X, W, Y, bias = conv_fuzzer.create_tensors(config, random.Random(config.rng_seed))
        prior_output = _create_prior_output(conv_case, X, W, Y)
        if conv_case.materialize_intermediate:
            if config.conv_type == conv_fuzzer.ConvType.FPROP:
                destination = Y
            elif config.conv_type == conv_fuzzer.ConvType.DGRAD:
                destination = X
            else:
                destination = W
            intermediate_output = torch.empty_like(destination, dtype=intermediate_dtype)
            conv_fuzzer.fill_with_garbage(intermediate_output)

        try:
            execution_succeeded, execution_message = conv_fuzzer.run_cudnn_conv(
                config,
                X,
                W,
                Y,
                bias,
                cudnn_handle,
                prior_output=prior_output,
                raise_on_unsupported=True,
                intermediate_dtype=conv_case.intermediate_dtype,
                intermediate_output=intermediate_output,
            )
        except cudnn.cudnnGraphNotSupportedError as error:
            pytest.skip(
                f"Unsupported convolution for cuDNN {cudnn.backend_version()}, "
                f"device capability {torch.cuda.get_device_capability()}, "
                f"dtypes {config.x_dtype}/{config.w_dtype}/{config.y_dtype}: {error}"
            )
        if not execution_succeeded:
            pytest.fail(execution_message)

        reference = conv_fuzzer.compute_reference(
            config,
            X,
            W,
            Y,
            bias,
            prior_output=prior_output,
            intermediate_dtype=conv_case.intermediate_dtype,
            return_intermediate=conv_case.materialize_intermediate,
        )
        if conv_case.materialize_intermediate:
            reference, intermediate_reference = reference
        if config.conv_type == conv_fuzzer.ConvType.FPROP:
            actual_output, output_dtype, output_name = Y, config.y_dtype, "Y"
        elif config.conv_type == conv_fuzzer.ConvType.DGRAD:
            actual_output, output_dtype, output_name = X, config.x_dtype, "dX"
        else:
            actual_output, output_dtype, output_name = W, config.w_dtype, "dW"

        assert torch.isfinite(actual_output).all().item(), f"{output_name} contains non-finite values"
        assert torch.isfinite(reference).all().item(), f"{output_name} reference contains non-finite values"
        comparison_passed, comparison_message = conv_fuzzer.compare_results(
            actual_output,
            reference,
            output_dtype,
            num_diffs,
            rtol=conv_case.rtol,
            atol=conv_case.atol,
        )
        assert comparison_passed, f"{output_name} numerical mismatch: {comparison_message}"
        if intermediate_output is not None:
            assert torch.isfinite(intermediate_output).all().item(), "Convolution intermediate contains non-finite values"
            assert torch.isfinite(intermediate_reference).all().item(), "Convolution intermediate reference contains non-finite values"
            comparison_passed, comparison_message = conv_fuzzer.compare_results(
                intermediate_output,
                intermediate_reference,
                intermediate_dtype,
                num_diffs,
                rtol=conv_case.rtol,
                atol=conv_case.atol,
            )
            assert comparison_passed, f"Convolution intermediate numerical mismatch: {comparison_message}"
    finally:
        try:
            cudnn.set_stream(handle=cudnn_handle, stream=original_stream)
        finally:
            del X, W, Y, bias, prior_output, reference, actual_output
            del intermediate_output, intermediate_reference, destination
            torch.cuda.empty_cache()


@pytest.mark.L0
@pytest.mark.parametrize("conv_case", FIXED_CONV_CASES_L0)
def test_conv_fixed_L0(conv_case: FixedConvCase, cudnn_handle, num_diffs):
    _run_fixed_conv_test(conv_case, cudnn_handle, num_diffs)


@pytest.mark.L1
@pytest.mark.parametrize("conv_case", FIXED_CONV_CASES_L1)
def test_conv_fixed_L1(conv_case: FixedConvCase, cudnn_handle, num_diffs):
    _run_fixed_conv_test(conv_case, cudnn_handle, num_diffs)


@pytest.mark.L1
@pytest.mark.parametrize("conv_case", FIXED_CONV_EXPANSION_CASES_L1)
def test_conv_fixed_expansion_L1(conv_case: FixedConvCase, cudnn_handle, num_diffs):
    _run_fixed_conv_test(conv_case, cudnn_handle, num_diffs)


@pytest.mark.L0
@pytest.mark.parametrize("conv_type", tuple(conv_fuzzer.ConvType))
@pytest.mark.parametrize("intermediate_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("relu", (False, True))
def test_fixed_reference_intermediate(conv_type, intermediate_dtype, relu):
    config = _fixed_conv_case(
        spatial_dims=2,
        batch=1,
        in_channels=1,
        out_channels=1,
        input_spatial=(2, 2),
        filter_spatial=(1, 1),
        padding=(0, 0),
        stride=(1, 1),
        dilation=(1, 1),
        conv_type=conv_type,
        dtype=torch.float32,
        epilogue=conv_fuzzer.EpilogueType.RELU if relu else conv_fuzzer.EpilogueType.NONE,
        rng_seed=0,
    ).config
    X = torch.tensor([1.0003, -1.0003, 0.3333, -0.3333]).reshape(1, 1, 2, 2)
    W = torch.tensor([1.3]).reshape(1, 1, 1, 1)
    Y = torch.tensor([0.7003, -0.9002, 0.1, -0.2]).reshape(1, 1, 2, 2)
    config.x_shape, config.w_shape = tuple(X.shape), tuple(W.shape)
    if conv_type == conv_fuzzer.ConvType.FPROP:
        raw_convolution = X * W
    elif conv_type == conv_fuzzer.ConvType.DGRAD:
        raw_convolution = Y * W
    else:
        raw_convolution = (X * Y).sum().reshape_as(W)
    prior_output = torch.full_like(raw_convolution, 0.00013)
    expected_intermediate = raw_convolution.to(intermediate_dtype)
    expected = expected_intermediate.float() + prior_output
    if relu:
        expected = expected.relu()
    actual, intermediate = conv_fuzzer.compute_reference(
        config, X, W, Y, None, prior_output=prior_output, intermediate_dtype=intermediate_dtype, return_intermediate=True
    )
    assert intermediate.dtype == intermediate_dtype
    torch.testing.assert_close(intermediate, expected_intermediate, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.L0
@pytest.mark.parametrize("conv_type", tuple(conv_fuzzer.ConvType))
def test_fixed_reference_defaults(conv_type):
    config = _fixed_conv_case(
        spatial_dims=2,
        batch=1,
        in_channels=1,
        out_channels=1,
        input_spatial=(2, 2),
        filter_spatial=(1, 1),
        padding=(0, 0),
        stride=(1, 1),
        dilation=(1, 1),
        conv_type=conv_type,
        dtype=torch.float32,
        rng_seed=0,
    ).config
    X = torch.tensor([1.0, -2.0, 3.0, -4.0]).reshape(1, 1, 2, 2)
    W = torch.tensor([2.0]).reshape(1, 1, 1, 1)
    Y = torch.tensor([3.0, 4.0, 5.0, 6.0]).reshape(1, 1, 2, 2)
    config.x_shape, config.w_shape = tuple(X.shape), tuple(W.shape)
    if conv_type == conv_fuzzer.ConvType.FPROP:
        expected = X * W
    elif conv_type == conv_fuzzer.ConvType.DGRAD:
        expected = Y * W
    else:
        expected = (X * Y).sum().reshape_as(W)
    actual = conv_fuzzer.compute_reference(config, X, W, Y, None)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.parametrize("path", ("reference", "graph"))
@pytest.mark.parametrize(
    "conv_type,epilogue,intermediate_dtype,expose",
    (
        (conv_fuzzer.ConvType.DGRAD, conv_fuzzer.EpilogueType.BIAS, None, False),
        (conv_fuzzer.ConvType.DGRAD, conv_fuzzer.EpilogueType.BIAS_RELU, None, False),
        (conv_fuzzer.ConvType.WGRAD, conv_fuzzer.EpilogueType.BIAS, None, False),
        (conv_fuzzer.ConvType.WGRAD, conv_fuzzer.EpilogueType.BIAS_RELU, None, False),
        (conv_fuzzer.ConvType.FPROP, conv_fuzzer.EpilogueType.NONE, torch.float16, False),
        (conv_fuzzer.ConvType.FPROP, conv_fuzzer.EpilogueType.NONE, None, True),
    ),
)
def test_fixed_invalid_fusion_arguments(path, conv_type, epilogue, intermediate_dtype, expose, cudnn_handle):
    config = replace(FIXED_CONV_CASES_L0[0].values[0].config, conv_type=conv_type, epilogue=epilogue)
    tensor = torch.ones(1)
    message = "Backward convolution does not support bias" if conv_type != conv_fuzzer.ConvType.FPROP else "requires an epilogue or residual add"
    original_stream = cudnn.get_stream(handle=cudnn_handle)
    try:
        with pytest.raises(ValueError, match=message):
            if path == "reference":
                conv_fuzzer.compute_reference(config, tensor, tensor, tensor, None, intermediate_dtype=intermediate_dtype, return_intermediate=expose)
            else:
                conv_fuzzer.run_cudnn_conv(
                    config,
                    tensor,
                    tensor,
                    tensor,
                    None,
                    cudnn_handle,
                    intermediate_dtype=intermediate_dtype,
                    intermediate_output=tensor if expose else None,
                )
        assert cudnn.get_stream(handle=cudnn_handle) == original_stream
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=original_stream)


@pytest.mark.L0
def test_fixed_intermediate_buffer_dtype(cudnn_handle):
    config = FIXED_CONV_CASES_L0[0].values[0].config
    tensor = torch.ones(1)
    original_stream = cudnn.get_stream(handle=cudnn_handle)
    try:
        with pytest.raises(ValueError, match="Intermediate buffer dtype"):
            conv_fuzzer.run_cudnn_conv(
                config,
                tensor,
                tensor,
                tensor,
                None,
                cudnn_handle,
                prior_output=tensor,
                intermediate_dtype=torch.float16,
                intermediate_output=tensor,
            )
        assert cudnn.get_stream(handle=cudnn_handle) == original_stream
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=original_stream)


@pytest.mark.L0
@pytest.mark.parametrize("epilogue", (conv_fuzzer.EpilogueType.BIAS, conv_fuzzer.EpilogueType.BIAS_RELU))
def test_fixed_reference_bias_order(epilogue):
    config = replace(
        FIXED_CONV_CASES_L0[0].values[0].config,
        batch=1,
        in_channels=1,
        out_channels=1,
        input_spatial=(1, 2),
        filter_spatial=(1, 1),
        padding=(0, 0),
        epilogue=epilogue,
    )
    X = torch.tensor([1.0003, -1.0003]).reshape(1, 1, 1, 2)
    W = torch.full((1, 1, 1, 1), 1.3)
    Y = torch.empty_like(X)
    prior_output = torch.full_like(X, 0.00013)
    bias = torch.full((1, 1, 1, 1), 0.00017)
    expected = (X * W).half().float() + prior_output + bias
    if epilogue == conv_fuzzer.EpilogueType.BIAS_RELU:
        expected = expected.relu()
    actual = conv_fuzzer.compute_reference(config, X, W, Y, bias, prior_output, intermediate_dtype=torch.float16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


@pytest.mark.L0
@pytest.mark.parametrize("path", ("reference", "graph"))
@pytest.mark.parametrize("epilogue", (conv_fuzzer.EpilogueType.BIAS, conv_fuzzer.EpilogueType.BIAS_RELU))
def test_fixed_missing_bias(path, epilogue, cudnn_handle):
    config = replace(FIXED_CONV_CASES_L0[0].values[0].config, epilogue=epilogue)
    tensor = torch.ones(1)
    original_stream = cudnn.get_stream(handle=cudnn_handle)
    try:
        with pytest.raises(ValueError, match="requires a bias tensor"):
            if path == "reference":
                conv_fuzzer.compute_reference(config, tensor, tensor, tensor, None)
            else:
                conv_fuzzer.run_cudnn_conv(config, tensor, tensor, tensor, None, cudnn_handle)
        assert cudnn.get_stream(handle=cudnn_handle) == original_stream
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=original_stream)


@pytest.mark.L0
def test_fixed_default_intermediate_buffer(cudnn_handle, num_diffs):
    conv_case = next(p.values[0] for p in FIXED_CONV_EXPANSION_CASES_L1 if p.id == "conv2d_wgrad_residual_exposed_f16_n64_c256_k16_7x7")
    _run_fixed_conv_test(replace(conv_case, intermediate_dtype=None), cudnn_handle, num_diffs)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("materialize", (False, True), ids=("virtual", "exposed"))
@pytest.mark.parametrize("fp32_edge", (False, True), ids=("low_precision_edge", "fp32_edge"))
def test_wgrad_intermediate_rounding(dtype, materialize, fp32_edge, cudnn_handle, num_diffs):
    if torch.cuda.get_device_capability() < ((8, 0) if dtype == torch.bfloat16 else (7, 0)):
        pytest.skip(f"Unsupported convolution input dtype: {dtype}")
    config = _fixed_conv_case(
        spatial_dims=2,
        batch=2,
        in_channels=32,
        out_channels=32,
        input_spatial=(4, 4),
        filter_spatial=(1, 1),
        padding=(0, 0),
        stride=(1, 1),
        dilation=(1, 1),
        conv_type=conv_fuzzer.ConvType.WGRAD,
        dtype=dtype,
        rng_seed=300,
    ).config
    X = W = Y = prior_output = intermediate = reference = raw_reference = None
    expected = opposite_precision = None
    edge_dtype = torch.float32 if fp32_edge else dtype
    original_stream = cudnn.get_stream(handle=cudnn_handle)
    try:
        X, W, Y, _ = conv_fuzzer.create_tensors(config, random.Random(config.rng_seed))
        X.zero_()
        Y.zero_()
        X[0, 0, 0, 0] = 1 + 2**-7
        Y[0, 0, 0, 0] = 1 + 2**-7
        prior_output = torch.zeros_like(W)
        prior_output[0, 0, 0, 0] = -(1 + 2**-6)
        if materialize:
            intermediate = torch.empty_like(W, dtype=edge_dtype)
            conv_fuzzer.fill_with_garbage(intermediate)

        # The exact product is 1 + 2^-6 + 2^-14. Rounding drops the last term;
        # cancellation after the edge makes that rounding visible in the output.
        expected = torch.zeros_like(W)
        opposite_precision = torch.zeros_like(W)
        expected[0, 0, 0, 0] = 2**-14 if fp32_edge else 0
        opposite_precision[0, 0, 0, 0] = 0 if fp32_edge else 2**-14
        assert not conv_fuzzer.compare_results(opposite_precision, expected, dtype, num_diffs, rtol=0, atol=1e-6)[0]
        try:
            success, message = conv_fuzzer.run_cudnn_conv(
                config,
                X,
                W,
                Y,
                None,
                cudnn_handle,
                prior_output,
                raise_on_unsupported=True,
                intermediate_dtype=edge_dtype,
                intermediate_output=intermediate,
            )
            assert success, message
        except cudnn.cudnnGraphNotSupportedError as error:
            pytest.skip(f"Unsupported convolution rounding control: {error}")
        reference, raw_reference = conv_fuzzer.compute_reference(
            config,
            X,
            W,
            Y,
            None,
            prior_output,
            intermediate_dtype=edge_dtype,
            return_intermediate=True,
        )
        assert torch.isfinite(W).all() and torch.isfinite(reference).all()
        passed, message = conv_fuzzer.compare_results(W, reference, dtype, num_diffs, rtol=0, atol=1e-6)
        assert passed, message
        torch.testing.assert_close(reference, expected, rtol=0, atol=0)
        if intermediate is not None:
            assert torch.isfinite(intermediate).all() and torch.isfinite(raw_reference).all()
            torch.testing.assert_close(intermediate, raw_reference, rtol=0, atol=0)
    finally:
        try:
            cudnn.set_stream(handle=cudnn_handle, stream=original_stream)
        finally:
            del X, W, Y, prior_output, intermediate, reference, raw_reference, expected, opposite_precision
            torch.cuda.empty_cache()
