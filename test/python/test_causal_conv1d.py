# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudnn

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "binding_name,args,kernel_size_index,api_name,argument_name,max_kernel_size",
    [
        (
            "causal_conv1d_forward",
            (0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0),
            8,
            "causal_conv1d",
            "kernel_size",
            256,
        ),
        (
            "causal_conv1d_backward",
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
            11,
            "causal_conv1d",
            "kernel_size",
            256,
        ),
        (
            "causal_conv1d_nwh_forward",
            (0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0),
            8,
            "causal_conv1d_nwh",
            "kernel_size",
            128,
        ),
        (
            "causal_conv1d_nwh_backward",
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
            11,
            "causal_conv1d_nwh",
            "kernel_size",
            128,
        ),
        (
            "b2b_causal_conv1d_forward",
            (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 0),
            10,
            "b2b_causal_conv1d",
            "kernel_size_proj",
            32,
        ),
        (
            "b2b_causal_conv1d_backward",
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0),
            14,
            "b2b_causal_conv1d",
            "kernel_size_proj",
            32,
        ),
        (
            "b2b_causal_conv1d_forward",
            (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0),
            11,
            "b2b_causal_conv1d",
            "kernel_size_mixer",
            256,
        ),
        (
            "b2b_causal_conv1d_backward",
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0),
            15,
            "b2b_causal_conv1d",
            "kernel_size_mixer",
            256,
        ),
    ],
)
@pytest.mark.parametrize("boundary", ["below", "above"])
def test_causal_conv1d_rejects_unsupported_kernel_size(binding_name, args, kernel_size_index, api_name, argument_name, max_kernel_size, boundary):
    binding = getattr(cudnn, binding_name, None)
    if binding is None:
        pytest.skip(f"{binding_name} is unavailable in this cuDNN frontend build")

    kernel_size = 1 if boundary == "below" else max_kernel_size + 1
    args = list(args)
    args[kernel_size_index] = kernel_size

    message = rf"{api_name} {argument_name} must be between 2 and {max_kernel_size}, inclusive; got {kernel_size}"
    with pytest.raises(ValueError, match=message):
        binding(*args)
