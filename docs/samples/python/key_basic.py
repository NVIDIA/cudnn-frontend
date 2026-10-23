# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/key.md
API: key
Use case: Basic
Source: adapted from `samples/cpp/sdpa/fp16_cached.cpp translated to Python form`.
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
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    a = graph.tensor(
        name="A",
        dim=(1, 4, 16),
        stride=(64, 16, 1),
        data_type=cudnn.data_type.BFLOAT16,
    )
    b = graph.tensor(
        name="B",
        dim=(1, 16, 8),
        stride=(128, 8, 1),
        data_type=cudnn.data_type.BFLOAT16,
    )
    c = graph.matmul(A=a, B=b, compute_data_type=cudnn.data_type.FLOAT)
    c.set_output(True).set_dim((1, 4, 8)).set_stride((32, 8, 1))
    graph.validate()
    graph.build_operation_graph()
    print(f"key={graph.key()}")


if __name__ == "__main__":
    run_sample(sample)
