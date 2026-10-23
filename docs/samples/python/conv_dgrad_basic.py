# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/conv_dgrad.md
API: conv_dgrad
Use case: Basic
Source: adapted from `test/python/test_conv_fuzzer.py`.
"""

import cudnn
import torch


def sample():
    torch.manual_seed(0)
    weight = torch.randn((4, 3, 3, 3), device="cuda", dtype=torch.float32)
    dy = torch.randn((1, 4, 8, 8), device="cuda", dtype=torch.float32)
    with cudnn.Graph(
        io_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=[dy, weight],
        outputs=["dX"],
        handle="auto",
    ) as graph:
        dx_desc = graph.conv_dgrad(
            loss=dy,
            filter=weight,
            padding=[1, 1],
            stride=[1, 1],
            dilation=[1, 1],
            compute_data_type=torch.float32,
            name="conv_dgrad",
        )
        dx_desc.set_output(True).set_name("dX").set_dim((1, 3, 8, 8)).set_stride(
            (192, 64, 8, 1)
        ).set_data_type(torch.float32)

    dx = graph(dy, weight)
    expected = torch.nn.grad.conv2d_input((1, 3, 8, 8), weight, dy, padding=1)
    torch.testing.assert_close(dx, expected, rtol=1e-4, atol=1e-4)
    print(f"conv_dgrad result matches reference: {tuple(dx.shape)}")


if __name__ == "__main__":
    sample()
