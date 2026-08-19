# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import dataclass
from typing import Optional

import torch

from cudnn.gnn import CscGraph, agg_simple


@dataclass(frozen=True)
class BenchmarkShape:
    num_src_nodes: int
    num_dst_nodes: int
    degree: int
    feature_dim: int

    @property
    def num_edges(self) -> int:
        return self.num_dst_nodes * self.degree


_SHAPES = {
    "small": BenchmarkShape(num_src_nodes=4_096, num_dst_nodes=4_096, degree=16, feature_dim=64),
    "medium": BenchmarkShape(num_src_nodes=65_536, num_dst_nodes=65_536, degree=32, feature_dim=128),
    "large": BenchmarkShape(num_src_nodes=262_144, num_dst_nodes=262_144, degree=64, feature_dim=256),
}

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class BenchmarkResult:
    forward_ms: float
    backward_ms: Optional[float]
    edges_per_second: float


def _make_inputs(shape: BenchmarkShape, dtype: torch.dtype) -> tuple[CscGraph, torch.Tensor]:
    device = torch.device("cuda")
    offsets = torch.arange(
        0,
        shape.num_edges + 1,
        shape.degree,
        device=device,
        dtype=torch.int32,
    )
    generator = torch.Generator(device=device).manual_seed(1234)
    indices = torch.randint(
        shape.num_src_nodes,
        (shape.num_edges,),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    features = torch.randn(
        (shape.num_src_nodes, shape.feature_dim),
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    return CscGraph(offsets, indices, shape.num_src_nodes), features


def _time_cuda(callable_, warmup: int, iterations: int) -> float:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark(
    shape: BenchmarkShape,
    *,
    dtype: torch.dtype,
    aggr: str,
    warmup: int,
    iterations: int,
    include_backward: bool,
) -> BenchmarkResult:
    graph, features = _make_inputs(shape, dtype)

    forward_ms = _time_cuda(lambda: agg_simple(graph, node_features=features, aggr=aggr), warmup, iterations)
    backward_ms = None
    if include_backward:
        grad_output = torch.randn((shape.num_dst_nodes, shape.feature_dim), device="cuda", dtype=dtype)

        def forward_backward() -> None:
            features.grad = None
            agg_simple(graph, node_features=features, aggr=aggr).backward(grad_output)

        total_ms = _time_cuda(forward_backward, warmup, iterations)
        backward_ms = max(total_ms - forward_ms, 0.0)

    edges_per_second = shape.num_edges / (forward_ms * 1.0e-3)
    return BenchmarkResult(forward_ms, backward_ms, edges_per_second)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cuDNN GNN simple aggregation")
    parser.add_argument("--shape", choices=_SHAPES, default="medium")
    parser.add_argument("--dtype", choices=_DTYPES, default="float32")
    parser.add_argument("--aggr", choices=("sum", "mean", "max", "min"), default="sum")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    result = benchmark(
        _SHAPES[args.shape],
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
