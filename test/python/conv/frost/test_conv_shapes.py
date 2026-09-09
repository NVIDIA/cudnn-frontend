# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Frost convolution shape and geometry coverage."""

from __future__ import annotations

from dataclasses import dataclass

import cudnn
import pytest
import torch

from utils import make_input, requires_dense_cutedsl, requires_sm100, select_and_build_frost_plan


@dataclass(frozen=True)
class ConvCase:
    name: str
    image_shape: tuple[int, int, int, int, int]
    weight_shape: tuple[int, int, int, int, int]
    pre_padding: tuple[int, int, int] = (0, 0, 0)
    post_padding: tuple[int, int, int] = (0, 0, 0)
    stride: tuple[int, int, int] = (1, 1, 1)
    dilation: tuple[int, int, int] = (1, 1, 1)


def _input_extent(output_extent: int, filter_extent: int, stride: int, dilation: int) -> int:
    """Return the unpadded input extent that produces output_extent."""
    return (output_extent - 1) * stride + dilation * (filter_extent - 1) + 1


def _geometry_cases() -> tuple[ConvCase, ...]:
    """Cover every supported stride value and dilation value on every axis.

    Eight cyclic tuples give marginal coverage of 1..8 in D, H, and W for
    both parameters without an impractical 8**6 Cartesian product. Input
    extents are derived so every case has a small 2x2x2 output.
    """
    cases = []
    output_spatial = (2, 2, 2)
    filter_spatial = (2, 2, 2)
    for index in range(8):
        stride = tuple(((index + offset) % 8) + 1 for offset in (0, 3, 5))
        dilation = tuple(((7 - index + offset) % 8) + 1 for offset in (0, 2, 4))
        input_spatial = tuple(
            _input_extent(output_extent, filter_extent, stride_extent, dilation_extent)
            for output_extent, filter_extent, stride_extent, dilation_extent in zip(output_spatial, filter_spatial, stride, dilation)
        )
        cases.append(
            ConvCase(
                name=f"stride-{stride[0]}{stride[1]}{stride[2]}-dilation-{dilation[0]}{dilation[1]}{dilation[2]}",
                image_shape=(1, 64, *input_spatial),
                weight_shape=(32, 64, *filter_spatial),
                stride=stride,
                dilation=dilation,
            )
        )
    return tuple(cases)


_SHAPE_CASES = (
    # A 64-byte BF16 channel row forces the automatic selector onto K64.
    ConvCase("automatic-k64-tile", (1, 32, 1, 1, 1), (32, 32, 1, 1, 1)),
    # Smallest aligned C/K rows and a single output element exercise both M
    # and N tail handling.
    ConvCase("minimum-aligned-single-output", (1, 64, 1, 1, 1), (8, 64, 1, 1, 1)),
    # Non-power-of-two batch/spatial sizes and a partial output-channel tile.
    ConvCase("partial-m-and-n-tiles", (3, 64, 2, 3, 5), (40, 64, 1, 1, 1)),
    # A complete 256x256 implicit-GEMM output tile, matching the basic example.
    ConvCase("complete-output-tile", (1, 64, 6, 10, 10), (256, 64, 3, 3, 3)),
    # More than one complete mainloop channel tile.
    ConvCase("multiple-input-channel-tiles", (1, 128, 2, 3, 4), (64, 128, 1, 1, 1)),
    # Filter extents, padding, and dilation are deliberately different on all
    # three spatial axes to catch D/H/W ordering mistakes.
    ConvCase(
        "anisotropic-filter-and-padding",
        (2, 64, 3, 5, 7),
        (32, 64, 1, 2, 3),
        pre_padding=(0, 1, 0),
        post_padding=(0, 0, 1),
        dilation=(1, 2, 1),
    ),
    ConvCase(
        "symmetric-padding",
        (1, 64, 3, 4, 5),
        (32, 64, 3, 3, 3),
        pre_padding=(1, 1, 1),
        post_padding=(1, 1, 1),
    ),
    ConvCase(
        "asymmetric-padding",
        (1, 64, 5, 9, 8),
        (32, 64, 3, 3, 3),
        pre_padding=(0, 1, 2),
        post_padding=(1, 0, 0),
    ),
    # Exercise both inclusive im2col corner limits: -pre_padding == -16 and
    # post_padding == 15 for a unit filter.
    ConvCase(
        "im2col-padding-corner-bounds",
        (1, 64, 1, 1, 1),
        (32, 64, 1, 1, 1),
        pre_padding=(16, 0, 1),
        post_padding=(0, 15, 0),
    ),
    # With a size-three filter, dilation eight reaches the other -16 corner
    # through -(filter_extent - 1) * dilation.
    ConvCase(
        "im2col-dilation-corner-bound",
        (1, 64, 17, 3, 3),
        (32, 64, 3, 1, 1),
        dilation=(8, 1, 1),
    ),
) + _geometry_cases()


pytestmark = [pytest.mark.L1, requires_sm100, requires_dense_cutedsl]


def _output_spatial(case: ConvCase) -> tuple[int, int, int]:
    return tuple(
        (image_extent + lower_pad + upper_pad - dilation * (filter_extent - 1) - 1) // stride + 1
        for image_extent, filter_extent, lower_pad, upper_pad, stride, dilation in zip(
            case.image_shape[2:],
            case.weight_shape[2:],
            case.pre_padding,
            case.post_padding,
            case.stride,
            case.dilation,
        )
    )


def test_shape_cases_obey_frost_alignment_and_geometry_contract() -> None:
    """Keep generated cases within the kernel's advertised support envelope."""
    for case in _SHAPE_CASES:
        assert case.image_shape[1] == case.weight_shape[1]
        assert case.image_shape[1] % 32 == 0  # Narrowest BF16 mainloop channel tile (64 bytes)
        assert case.weight_shape[0] % 8 == 0  # 16-byte BF16 output rows
        assert all(1 <= value <= 8 for value in case.stride)
        assert all(value > 0 for value in case.dilation)
        assert all(value > 0 for value in _output_spatial(case))
        for filter_extent, lower_pad, upper_pad, dilation in zip(case.weight_shape[2:], case.pre_padding, case.post_padding, case.dilation):
            assert -16 <= -lower_pad <= 15
            assert -16 <= upper_pad - (filter_extent - 1) * dilation <= 15

    for axis in range(3):
        assert {case.stride[axis] for case in _geometry_cases()} == set(range(1, 9))
        assert {case.dilation[axis] for case in _geometry_cases()} == set(range(1, 9))


@pytest.mark.parametrize("case", _SHAPE_CASES, ids=lambda case: case.name)
def test_frost_conv_supported_shapes_and_geometry(monkeypatch, case: ConvCase) -> None:
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

    image_gpu = make_input(case.image_shape, torch.bfloat16, seed=1234)
    weight_gpu = make_input(case.weight_shape, torch.bfloat16, seed=5678)

    pre_d, pre_h, pre_w = case.pre_padding
    post_d, post_h, post_w = case.post_padding
    padded_image = torch.nn.functional.pad(image_gpu.float(), (pre_w, post_w, pre_h, post_h, pre_d, post_d))
    reference = torch.nn.functional.conv3d(
        padded_image,
        weight_gpu.float(),
        stride=case.stride,
        dilation=case.dilation,
    )
    assert tuple(reference.shape[2:]) == _output_spatial(case)
    expected = reference.to(torch.bfloat16).float()

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    image = graph.tensor_like(image_gpu, name="image")
    weight = graph.tensor_like(weight_gpu, name="weight")
    output = graph.conv_fprop(
        image,
        weight,
        name="conv",
        pre_padding=case.pre_padding,
        post_padding=case.post_padding,
        stride=case.stride,
        dilation=case.dilation,
    )

    output_gpu = torch.empty(
        reference.shape,
        dtype=torch.bfloat16,
        device="cuda",
        memory_format=torch.channels_last_3d,
    )
    output.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(output_gpu.shape).set_stride(output_gpu.stride())
    select_and_build_frost_plan(graph, "frost_conv")

    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    graph.execute({image: image_gpu, weight: weight_gpu, output: output_gpu}, workspace)
    torch.cuda.synchronize()

    epsilon = float(torch.finfo(torch.bfloat16).eps)
    torch.testing.assert_close(output_gpu.float(), expected, atol=2 * epsilon, rtol=2 * epsilon)
