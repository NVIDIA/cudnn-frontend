# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/add.md
API: add
Use case: Basic
Source: adapted from `python/pygraph/pointwise.cpp`.
"""

import cudnn
import torch


def sample():
    a = torch.rand((2, 3, 4), device="cuda", dtype=torch.float32)
    b = torch.rand((2, 3, 4), device="cuda", dtype=torch.float32)
    with cudnn.Graph(
        io_data_type=torch.float32,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=[a, b],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        Y = graph.add(a=a, b=b, compute_data_type=torch.float32, name="add")
        Y.set_output(True).set_name("Y")
    y = graph(a, b)
    torch.testing.assert_close(y, a + b)
    print(f"add output: shape={tuple(y.shape)}, dtype={y.dtype}")


if __name__ == "__main__":
    sample()
