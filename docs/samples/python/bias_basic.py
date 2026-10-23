# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/bias.md
API: bias
Use case: Basic
Source: adapted from `python/pygraph/pointwise.cpp`.
"""

import sys

try:
    import cudnn
    import torch
except Exception as exc:
    print(f"SKIP: could not import cudnn or torch: {exc}")
    raise SystemExit(0)


def run_sample(sample):
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available to PyTorch")
        return
    try:
        sample()
    except Exception as exc:
        if "--strict" in sys.argv:
            raise
        print(f"SKIP: sample could not run on this installation: {exc}")


def sample():
    x = torch.randn((2, 3, 4), device="cuda", dtype=torch.float32)
    bias = torch.randn((1, 3, 1), device="cuda", dtype=torch.float32)
    with cudnn.Graph(
        io_data_type=torch.float32, inputs=["X", "bias"], outputs=["Y"], handle="auto"
    ) as graph:
        X = graph.tensor(name="X", dim=x.shape, stride=x.stride())
        B = graph.tensor(name="bias", dim=bias.shape, stride=bias.stride())
        Y = graph.bias(input=X, bias=B, compute_data_type=cudnn.data_type.FLOAT, name="bias")
        Y.set_output(True).set_name("Y")
    y = graph(x, bias)
    print(f"bias output: shape={tuple(y.shape)}, dtype={y.dtype}")


if __name__ == "__main__":
    run_sample(sample)
