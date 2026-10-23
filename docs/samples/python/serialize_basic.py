# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/serialize.md
API: serialize
Use case: Basic
Source: adapted from `test/python/test_kernel_cache.py`.
"""

import cudnn


def sample():
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = graph.tensor(
        name="A",
        dim=(2, 4, 16),
        stride=(64, 16, 1),
        data_type=cudnn.data_type.BFLOAT16,
    )
    B = graph.tensor(
        name="B",
        dim=(2, 16, 8),
        stride=(128, 8, 1),
        data_type=cudnn.data_type.BFLOAT16,
    )
    C = graph.matmul(A=A, B=B, compute_data_type=cudnn.data_type.FLOAT, name="matmul")
    C.set_output(True).set_name("C").set_dim((2, 4, 8)).set_stride((32, 8, 1)).set_data_type(
        cudnn.data_type.BFLOAT16
    )
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    print("serialized_bytes", len(graph.serialize()))


if __name__ == "__main__":
    sample()
