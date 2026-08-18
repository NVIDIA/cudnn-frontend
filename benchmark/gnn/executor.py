# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import torch

from cudnn.gnn import CscGraph, agg_simple_n2n

from model_shapes import GnnBenchmarkShape


@dataclass(frozen=True)
class BenchmarkResult:
    forward_ms: float
    backward_ms: Optional[float]
    edges_per_second: float


def _make_inputs(shape: GnnBenchmarkShape, dtype: torch.dtype) -> tuple[CscGraph, torch.Tensor]:
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
    shape: GnnBenchmarkShape,
    *,
    dtype: torch.dtype,
    aggr: str,
    warmup: int,
    iterations: int,
    include_backward: bool,
) -> BenchmarkResult:
    graph, features = _make_inputs(shape, dtype)

    forward_ms = _time_cuda(lambda: agg_simple_n2n(features, graph, aggr=aggr), warmup, iterations)
    backward_ms = None
    if include_backward:
        grad_output = torch.randn((shape.num_dst_nodes, shape.feature_dim), device="cuda", dtype=dtype)

        def forward_backward() -> None:
            features.grad = None
            agg_simple_n2n(features, graph, aggr=aggr).backward(grad_output)

        total_ms = _time_cuda(forward_backward, warmup, iterations)
        backward_ms = max(total_ms - forward_ms, 0.0)

    edges_per_second = shape.num_edges / (forward_ms * 1.0e-3)
    return BenchmarkResult(forward_ms, backward_ms, edges_per_second)
