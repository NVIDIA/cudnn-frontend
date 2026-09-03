# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interleaved forward benchmark for qlen=1 tile experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import types

import torch

repo = Path(__file__).resolve().parents[2]
cudnn = types.ModuleType("cudnn")
cudnn.__path__ = [str(repo / "python" / "cudnn")]
cudnn.__file__ = str(repo / "python" / "cudnn" / "__init__.py")
cudnn.HSTUBwdSm100 = object
cudnn.HSTUFwdSm100 = object
sys.modules["cudnn"] = cudnn

hstu = types.ModuleType("cudnn.hstu_attention")
hstu.__path__ = [str(repo / "python" / "cudnn" / "hstu_attention")]
sys.modules["cudnn.hstu_attention"] = hstu

benchmark_path = Path(__file__).with_name("benchmark_hstu_qlen1.py")
spec = importlib.util.spec_from_file_location("hstu_q1_benchmark", benchmark_path)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impls", nargs="+", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(64, 512, 1024))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--groups", type=int, default=9)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    for case_index, batch_size in enumerate(args.batch_sizes):
        tensors = benchmark._make_inputs(batch_size, 4, 128, 2048, torch.bfloat16, device, seed=1234 + case_index)
        k_lengths = tensors["k_lengths"]
        runs = {}
        for impl in args.impls:
            _, run, _ = benchmark._compile_forward(tensors, max(k_lengths), (-1, 0), 0.7, 2048.0, impl)
            runs[impl] = run

        for _ in range(args.warmup):
            for run in runs.values():
                run()
        torch.cuda.synchronize()

        samples = {impl: [] for impl in args.impls}
        for group in range(args.groups):
            order = list(args.impls)
            shift = group % len(order)
            order = order[shift:] + order[:shift]
            if (group // len(args.impls)) % 2:
                order.reverse()
            for impl in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    runs[impl]()
                end.record()
                end.synchronize()
                samples[impl].append(start.elapsed_time(end) / args.iterations)

        result = {
            "batch_size": batch_size,
            "device": torch.cuda.get_device_name(device),
            "results": {
                impl: {
                    "median_ms": statistics.median(values),
                    "min_ms": min(values),
                    "samples_ms": values,
                }
                for impl, values in samples.items()
            },
        }
        print("INTERLEAVED " + json.dumps(result, sort_keys=True), flush=True)
        del tensors, runs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
