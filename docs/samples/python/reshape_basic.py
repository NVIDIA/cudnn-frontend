# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/reshape.md
API: reshape
Use case: Basic
Source: adapted from `samples/cpp/membound/reshape.cpp` and the Python binding.
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
    X = graph.tensor(name="X", dim=(2, 3, 4), stride=(12, 4, 1), data_type=cudnn.data_type.FLOAT)
    Y = graph.reshape(input=X, name="reshape")
    Y.set_dim((6, 4)).set_stride((4, 1))
    Y.set_output(True).set_name("Y").set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    print("reshape graph validated")


if __name__ == "__main__":
    run_sample(sample)
