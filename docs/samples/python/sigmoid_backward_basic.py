# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/sigmoid_backward.md
API: sigmoid_backward
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
    dY_data = torch.randn((2, 3, 4), device="cuda", dtype=torch.float32)
    x = torch.randn((2, 3, 4), device="cuda", dtype=torch.float32)
    with cudnn.Graph(
        io_data_type=torch.float32,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=["dY", "X"],
        outputs=["dX"],
        handle="auto",
    ) as graph:
        dY = graph.tensor(name="dY", dim=dY_data.shape, stride=dY_data.stride())
        X = graph.tensor(name="X", dim=x.shape, stride=x.stride())
        dX = graph.sigmoid_backward(loss=dY, input=X, name="sigmoid_backward")
        dX.set_output(True).set_name("dX")
    out = graph(dY_data, x)
    print(f"sigmoid_backward output: shape={tuple(out.shape)}, dtype={out.dtype}")


if __name__ == "__main__":
    run_sample(sample)
