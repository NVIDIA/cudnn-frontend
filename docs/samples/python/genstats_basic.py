# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/genstats.md
API: genstats
Use case: Basic
Source: adapted from `test/python/test_conv_genstats.py`.
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
    sum_out, sq_sum = graph.genstats(
        input=X, compute_data_type=cudnn.data_type.FLOAT, name="genstats"
    )
    sum_out.set_output(True).set_name("sum").set_data_type(cudnn.data_type.FLOAT)
    sq_sum.set_output(True).set_name("sq_sum").set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    print("genstats graph validated")


if __name__ == "__main__":
    run_sample(sample)
