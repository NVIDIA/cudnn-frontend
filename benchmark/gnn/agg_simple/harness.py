# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch

from executor import benchmark
from model_shapes import MODEL_SHAPES

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cuDNN GNN simple aggregation")
    parser.add_argument("--shape", choices=MODEL_SHAPES, default="medium")
    parser.add_argument("--dtype", choices=_DTYPES, default="float32")
    parser.add_argument("--aggr", choices=("sum", "mean", "max", "min"), default="sum")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    result = benchmark(
        MODEL_SHAPES[args.shape],
        dtype=_DTYPES[args.dtype],
        aggr=args.aggr,
        warmup=args.warmup,
        iterations=args.iterations,
        include_backward=args.backward,
    )
    print(f"forward_ms={result.forward_ms:.4f}")
    print(f"edges_per_second={result.edges_per_second:.3e}")
    if result.backward_ms is not None:
        print(f"backward_ms={result.backward_ms:.4f}")


if __name__ == "__main__":
    main()
