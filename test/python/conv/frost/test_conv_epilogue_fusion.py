# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for every Frost convolution unary epilogue."""

from __future__ import annotations

import cudnn
import pytest
import torch

from cudnn.conv.frost.engine import FrostConvEngine
from cudnn.conv.frost.graph_analyzer import SUPPORTED_UNARY_EPILOGUES
from epilogue_cases import EPILOGUE_CASES, EpilogueCase
from utils import make_input, requires_dense_cutedsl, requires_sm100, select_and_build_frost_plan

_INPUT_SHAPE = (1, 64, 4, 8, 8)
_WEIGHT_SHAPE = (256, 64, 1, 1, 1)
_ATOL = 2.0e-2
_RTOL = 2.0e-2


pytestmark = [pytest.mark.L1, requires_sm100, requires_dense_cutedsl]


def _canonical_channels_last(tensor: torch.Tensor) -> torch.Tensor:
    output = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device, memory_format=torch.channels_last_3d)
    return output.copy_(tensor)


def _make_operands(input_profile: str) -> tuple[torch.Tensor, torch.Tensor]:
    image = make_input(_INPUT_SHAPE, torch.bfloat16, seed=1234)
    weight = make_input(_WEIGHT_SHAPE, torch.bfloat16, seed=5678)

    if input_profile == "positive":
        # Keep log/sqrt/reciprocal inputs strictly positive and well away from
        # zero. The small scale also keeps every transcendental in a benign
        # numerical range.
        image = _canonical_channels_last((image.abs() + 0.25) * 0.125)
        weight = _canonical_channels_last((weight.abs() + 0.25) * 0.125)
    elif input_profile == "logical":
        # Make the convolution an exact copy of image channel zero. This
        # guarantees that logical_not observes both exact zeros and nonzeros.
        image = (image * 0.125).contiguous(memory_format=torch.channels_last_3d)
        weight = torch.zeros_like(weight)
        weight[:, 0, 0, 0, 0] = 1.0
    else:
        # Small signed results exercise both activation branches and avoid tan
        # poles or exponential overflow.
        image = (image * 0.125).contiguous(memory_format=torch.channels_last_3d)

    return image, weight


def test_epilogue_cases_match_the_engine_contract() -> None:
    """Adding an engine mode must also add an end-to-end case here."""
    from cudnn.conv.frost.templates.epilogue import SUPPORTED_EPILOGUES

    assert {case.mode for case in EPILOGUE_CASES} == set(SUPPORTED_UNARY_EPILOGUES) == set(SUPPORTED_EPILOGUES)


def test_plain_conv_does_not_require_a_float_intermediate() -> None:
    image_gpu, weight_gpu = _make_operands("signed")
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, compute_data_type=cudnn.data_type.FLOAT)
    image = graph.tensor_like(image_gpu, name="image")
    weight = graph.tensor_like(weight_gpu, name="weight")
    output = graph.conv_fprop(
        image,
        weight,
        name="conv",
        pre_padding=(0, 0, 0),
        post_padding=(0, 0, 0),
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
    )
    output_gpu = torch.empty((_INPUT_SHAPE[0], _WEIGHT_SHAPE[0], *_INPUT_SHAPE[2:]), dtype=torch.bfloat16, device="cuda", memory_format=torch.channels_last_3d)
    output.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(output_gpu.shape).set_stride(output_gpu.stride())
    graph.validate()

    FrostConvEngine().check_support(graph)


@pytest.mark.parametrize(
    "intermediate_dtype,compute_dtype,error",
    (
        (cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT, "requires a FLOAT convolution intermediate"),
        (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, "requires FP32 compute"),
    ),
    ids=("non-fp32-intermediate", "non-fp32-compute"),
)
def test_frost_conv_rejects_unsupported_precision_contracts(intermediate_dtype, compute_dtype, error) -> None:
    image_gpu, weight_gpu = _make_operands("signed")
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=intermediate_dtype,
        compute_data_type=compute_dtype,
    )
    image = graph.tensor_like(image_gpu, name="image")
    weight = graph.tensor_like(weight_gpu, name="weight")
    conv = graph.conv_fprop(
        image,
        weight,
        name="conv",
        pre_padding=(0, 0, 0),
        post_padding=(0, 0, 0),
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
    )
    output = graph.relu(input=conv, name="relu")
    output.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    graph.validate()

    with pytest.raises(NotImplementedError, match=error):
        FrostConvEngine().check_support(graph)


@pytest.mark.parametrize("case", EPILOGUE_CASES, ids=lambda case: case.name)
def test_frost_conv_supported_unary_epilogues(monkeypatch, case: EpilogueCase) -> None:
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

    image_gpu, weight_gpu = _make_operands(case.input_profile)
    conv_reference = torch.nn.functional.conv3d(image_gpu.float(), weight_gpu.float())
    if case.input_profile == "positive":
        assert torch.all(conv_reference > 0)
    elif case.input_profile == "logical":
        assert torch.any(conv_reference == 0)
        assert torch.any(conv_reference != 0)
    else:
        assert torch.any(conv_reference < 0)
        assert torch.any(conv_reference > 0)
    expected = case.reference_fn(conv_reference).to(torch.bfloat16).float()

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    image = graph.tensor_like(image_gpu, name="image")
    weight = graph.tensor_like(weight_gpu, name="weight")
    conv = graph.conv_fprop(
        image,
        weight,
        name="conv",
        pre_padding=(0, 0, 0),
        post_padding=(0, 0, 0),
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
    )
    output = case.graph_fn(graph, conv)

    output_gpu = torch.empty(
        conv_reference.shape,
        dtype=torch.bfloat16,
        device="cuda",
        memory_format=torch.channels_last_3d,
    )
    output.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(output_gpu.shape).set_stride(output_gpu.stride())
    select_and_build_frost_plan(graph, "frost_conv")

    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    graph.execute({image: image_gpu, weight: weight_gpu, output: output_gpu}, workspace)
    torch.cuda.synchronize()

    torch.testing.assert_close(output_gpu.float(), expected, atol=_ATOL, rtol=_RTOL)
