# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI harness for HSTU LMSD micro and forward-to-backward benchmarks.

Examples:
    python -m benchmark.hstu.hstu_lmsd.harness --shape smoke --mode all
    python -m benchmark.hstu.hstu_lmsd.harness --shape hstu_production --mode all
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
from typing import Callable, Optional

import torch

import cudnn

from .executor import HSTULMSDExecutor
from .model_shapes import DEFAULT_SHAPE, MODEL_SHAPES, get_model_shape


@dataclass(frozen=True)
class TimingResult:
    mode: str
    p50_ms: float
    mean_ms: float
    minimum_ms: float
    p95_ms: float
    logical_gbps: float
    warmup: int
    repeats: int


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def time_cuda_callable(
    operation: Callable[[], None],
    *,
    mode: str,
    logical_bytes: int,
    warmup: int,
    repeats: int,
) -> TimingResult:
    """Measure a precompiled, allocation-free callable with CUDA events."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    samples_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    p50_ms = statistics.median(samples_ms)
    return TimingResult(
        mode=mode,
        p50_ms=p50_ms,
        mean_ms=statistics.fmean(samples_ms),
        minimum_ms=min(samples_ms),
        p95_ms=_percentile(samples_ms, 0.95),
        logical_gbps=logical_bytes / (p50_ms * 1e-3) / 1e9,
        warmup=warmup,
        repeats=repeats,
    )


def run_micro_benchmarks(
    executor: HSTULMSDExecutor,
    *,
    warmup: int,
    repeats: int,
) -> list[TimingResult]:
    """Benchmark forward and backward independently."""

    return [
        time_cuda_callable(
            executor.forward,
            mode="forward",
            logical_bytes=executor.logical_bytes("forward"),
            warmup=warmup,
            repeats=repeats,
        ),
        time_cuda_callable(
            executor.backward,
            mode="backward",
            logical_bytes=executor.logical_bytes("backward"),
            warmup=warmup,
            repeats=repeats,
        ),
    ]


def run_e2e_benchmark(
    executor: HSTULMSDExecutor,
    *,
    warmup: int,
    repeats: int,
) -> TimingResult:
    """Benchmark forward plus explicit backward as one training dataflow."""

    return time_cuda_callable(
        executor.e2e,
        mode="e2e",
        logical_bytes=executor.logical_bytes("e2e"),
        warmup=warmup,
        repeats=repeats,
    )


def run_benchmarks(
    *,
    shape_name: str,
    mode: str,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict:
    shape = get_model_shape(shape_name)
    executor = HSTULMSDExecutor(shape, seed=seed)
    if mode == "all":
        timings = run_micro_benchmarks(executor, warmup=warmup, repeats=repeats)
        timings.append(run_e2e_benchmark(executor, warmup=warmup, repeats=repeats))
    elif mode == "e2e":
        timings = [run_e2e_benchmark(executor, warmup=warmup, repeats=repeats)]
    else:
        operation = executor.forward if mode == "forward" else executor.backward
        timings = [
            time_cuda_callable(
                operation,
                mode=mode,
                logical_bytes=executor.logical_bytes(mode),
                warmup=warmup,
                repeats=repeats,
            )
        ]

    return {
        "operation": "hstu_lmsd",
        "shape": asdict(shape),
        "dtype": "bfloat16",
        "seed": seed,
        "backward_chunks": executor.backward_chunks,
        "workspace_bytes": executor.workspace_bytes,
        "device": torch.cuda.get_device_name(executor.device),
        "compute_capability": list(torch.cuda.get_device_capability(executor.device)),
        "torch_version": torch.__version__,
        "cudnn_frontend_version": getattr(cudnn, "__version__", "unknown"),
        "cudnn_backend_version": cudnn.backend_version(),
        "timings": [asdict(timing) for timing in timings],
    }


def _print_results(result: dict) -> None:
    shape = result["shape"]
    print(f"HSTU LMSD shape={shape['name']} N={shape['num_rows']:,} D={shape['hidden_size']} u_stride={shape['u_storage_width']} device={result['device']}")
    print("mode       p50(ms)  mean(ms)   min(ms)   p95(ms)  logical GB/s")
    for timing in result["timings"]:
        print(
            f"{timing['mode']:<10} "
            f"{timing['p50_ms']:>8.3f} "
            f"{timing['mean_ms']:>9.3f} "
            f"{timing['minimum_ms']:>9.3f} "
            f"{timing['p95_ms']:>9.3f} "
            f"{timing['logical_gbps']:>13.1f}"
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        choices=sorted(MODEL_SHAPES),
        default=DEFAULT_SHAPE,
    )
    parser.add_argument(
        "--mode",
        choices=("forward", "backward", "e2e", "all"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional path for the complete machine-readable result",
    )
    parser.add_argument(
        "--list-shapes",
        action="store_true",
        help="print named workloads and exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_shapes:
        for name, shape in MODEL_SHAPES.items():
            print(f"{name}: N={shape.num_rows:,}, D={shape.hidden_size}, u_stride={shape.u_storage_width}, p={shape.dropout_ratio}")
        return 0
    result = run_benchmarks(
        shape_name=args.shape,
        mode=args.mode,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    _print_results(result)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
