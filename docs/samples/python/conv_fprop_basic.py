# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/conv_fprop.md
API: conv_fprop
Use case: Basic
Source: adapted from `test/python/test_conv_fprop.py`.
"""

import cudnn
import torch


def sample():
    torch.manual_seed(0)
    x = torch.randn((1, 3, 8, 8), device="cuda", dtype=torch.float32)
    weight = torch.randn((4, 3, 3, 3), device="cuda", dtype=torch.float32)
    with cudnn.Graph(
        io_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=[x, weight],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        y_desc = graph.conv_fprop(
            image=x,
            weight=weight,
            padding=[1, 1],
            stride=[1, 1],
            dilation=[1, 1],
            compute_data_type=torch.float32,
            name="conv_fprop",
        )
        y_desc.set_output(True).set_name("Y").set_data_type(torch.float32)

    y = graph(x, weight)
    expected = torch.nn.functional.conv2d(x, weight, padding=1)
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)
    print(f"conv_fprop result matches reference: {tuple(y.shape)}")


if __name__ == "__main__":
    sample()
