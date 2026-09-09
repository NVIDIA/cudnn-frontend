# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Frost 3D-convolution coverage for every storage dtype pair."""

from __future__ import annotations

import itertools

import cudnn
import pytest
import torch

from utils import make_input, requires_dense_cutedsl, requires_sm100, select_and_build_frost_plan

_DTYPES = (
    ("fp16", torch.float16, cudnn.data_type.HALF),
    ("bf16", torch.bfloat16, cudnn.data_type.BFLOAT16),
    ("fp32", torch.float32, cudnn.data_type.FLOAT),
    ("fp8_e4m3", torch.float8_e4m3fn, cudnn.data_type.FP8_E4M3),
    ("fp8_e5m2", torch.float8_e5m2, cudnn.data_type.FP8_E5M2),
)
_DTYPE_COMBINATIONS = tuple(itertools.product(_DTYPES, repeat=2))
_DTYPE_COMBINATION_IDS = tuple(f"{input_dtype[0]}-to-{output_dtype[0]}" for input_dtype, output_dtype in _DTYPE_COMBINATIONS)


pytestmark = [pytest.mark.L0, requires_sm100, requires_dense_cutedsl]


@pytest.mark.parametrize("input_dtype,output_dtype", _DTYPE_COMBINATIONS, ids=_DTYPE_COMBINATION_IDS)
def test_frost_conv_supported_dtype_combinations(monkeypatch, input_dtype, output_dtype) -> None:
    """The five supported A/B dtypes combine independently with all five C dtypes."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

    _input_name, input_torch_dtype, input_cudnn_dtype = input_dtype
    _output_name, output_torch_dtype, output_cudnn_dtype = output_dtype

    # Z*P*Q = 4*8*8 = 256 and K=256 match the kernel's fixed output tile;
    # C=128 is a complete mainloop channel tile for every input dtype.
    image_gpu = make_input((1, 128, 6, 10, 10), input_torch_dtype, seed=1234)
    weight_gpu = make_input((256, 128, 3, 3, 3), input_torch_dtype, seed=5678)
    reference = torch.nn.functional.conv3d(image_gpu.float(), weight_gpu.float())
    expected = reference.to(output_torch_dtype).float()

    graph = cudnn.pygraph(
        io_data_type=input_cudnn_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
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

    output_gpu = torch.empty(reference.shape, dtype=output_torch_dtype, device="cuda", memory_format=torch.channels_last_3d)
    output.set_output(True).set_data_type(output_cudnn_dtype).set_dim(output_gpu.shape).set_stride(output_gpu.stride())
    select_and_build_frost_plan(graph, "frost_conv")

    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    graph.execute({image: image_gpu, weight: weight_gpu, output: output_gpu}, workspace)
    torch.cuda.synchronize()

    # Compare after applying the output dtype's quantization to the reference.
    # Two output epsilons allow for a one-step conversion difference at a bin
    # boundary while remaining dtype-scaled across FP32 through FP8.
    epsilon = float(torch.finfo(output_torch_dtype).eps)
    torch.testing.assert_close(output_gpu.float(), expected, atol=2 * epsilon, rtol=2 * epsilon)
