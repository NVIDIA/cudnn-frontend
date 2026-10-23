# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/matmul.md
API: matmul
Use case: Example 2
Source: adapted from `test/python/test_matmul_bias_relu.py`.
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
    x = torch.randn((2, 4, 16), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((2, 16, 8), device="cuda", dtype=torch.bfloat16)
    with cudnn.Graph(
        io_data_type=torch.bfloat16,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=["X", "W"],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        X = graph.tensor(name="X", dim=x.shape, stride=x.stride())
        W = graph.tensor(name="W", dim=w.shape, stride=w.stride())
        Y = graph.matmul(A=X, B=W, compute_data_type=cudnn.data_type.FLOAT, name="matmul")
        Y.set_output(True).set_name("Y").set_data_type(cudnn.data_type.BFLOAT16)
    y = graph(x, w)
    print(f"matmul output: shape={tuple(y.shape)}, dtype={y.dtype}")


if __name__ == "__main__":
    run_sample(sample)
