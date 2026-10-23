# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/tensor.md
API: tensor
Use case: Example 2
Source: adapted from `test/python/test_norm_fuzzer.py`.
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
    X = graph.tensor(name="X", dim=(2, 3), stride=(3, 1), data_type=cudnn.data_type.FLOAT)
    Y = graph.identity(input=X, name="identity")
    Y.set_output(True).set_name("Y").set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    print("tensor graph validated")


if __name__ == "__main__":
    run_sample(sample)
