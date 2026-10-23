# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/tensor_like.md
API: tensor_like
Use case: Example 2
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
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x = torch.randn((2, 3), device="cuda", dtype=torch.float32)
    X = graph.tensor_like(x, name="X")
    Y = graph.identity(input=X, name="identity")
    Y.set_output(True).set_name("Y").set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    print("tensor_like graph validated")


if __name__ == "__main__":
    run_sample(sample)
