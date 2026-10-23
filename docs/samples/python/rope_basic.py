# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/rope.md
API: rope
Use case: Basic
Source: adapted from `test/python/test_oss_rope.py`.
"""

import cudnn
import torch


def sample():
    B, H, S, D = 1, 2, 16, 64
    torch.manual_seed(0)
    x = torch.randn((B, H, S, D), device="cuda", dtype=torch.bfloat16)
    freqs = torch.randn((S, 1, 1, D), device="cuda", dtype=torch.float32)
    with cudnn.Graph(
        io_data_type=torch.bfloat16,
        compute_data_type=torch.float32,
        inputs=[x, freqs],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        y_desc = graph.rope(input=x, freqs=freqs, name="rope")
        y_desc.set_output(True).set_name("Y").set_data_type(torch.bfloat16)

    y = graph(x, freqs)
    angles = freqs[:, 0, 0, : D // 2].reshape(1, 1, S, D // 2)
    x1, x2 = x.float().chunk(2, dim=-1)
    expected = torch.cat(
        (x1 * angles.cos() - x2 * angles.sin(), x2 * angles.cos() + x1 * angles.sin()),
        dim=-1,
    ).to(torch.bfloat16)
    torch.testing.assert_close(y, expected, rtol=3e-2, atol=3e-2)
    print(f"rope result matches reference: {tuple(y.shape)}")


if __name__ == "__main__":
    sample()
