# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep qlen=1 HSTU implementations over a representative workload matrix.

The default matrix varies one important workload axis at a time around the
customer target (BF16, causal, H=4, D=128, average KV=2048).  Multiple
implementations are measured against the same allocated inputs for each case.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics

import torch

import cudnn
from benchmark_hstu_qlen1 import (
    _compile_backward,
    _compile_forward,
    _make_inputs,
    _measure_ms,
)


def _default_cases() -> list[tuple[int, int, int]]:
    """Return (batch, heads, average_kv) cases without duplicates."""
    cases: list[tuple[int, int, int]] = []

    def add(batch: int, heads: int, average_kv: int) -> None:
        case = (batch, heads, average_kv)
        if case not in cases:
            cases.append(case)

    # Batch/CTA-grid sweep around the requested BS=64/512/1024 points.
    for batch in (16, 32, 64, 128, 256, 512, 1024, 2048):
        add(batch, 4, 2048)

    # Head count also scales the launch grid, independently of batch size.
    for batch in (64, 512, 1024):
        for heads in (1, 2, 4, 8):
            add(batch, heads, 2048)

    # KV length controls useful work per CTA and the split setup tradeoff.
    for batch in (64, 512, 1024):
        for average_kv in (128, 256, 512, 1024, 2048, 4096):
            add(batch, 4, average_kv)

    # Opposite grid/work corners catch policies that overfit one axis.
    for case in ((64, 1, 4096), (64, 8, 128), (1024, 1, 4096), (1024, 8, 128)):
        add(*case)
    return cases


def _parse_case(value: str) -> tuple[int, int, int]:
    try:
        batch, heads, average_kv = (int(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("cases must use BATCH:HEADS:AVERAGE_KV") from exc
    if min(batch, heads, average_kv) <= 0:
        raise argparse.ArgumentTypeError("case values must be positive")
    return batch, heads, average_kv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("forward", "backward"), required=True)
    parser.add_argument("--implementations", nargs="+", default=("auto",))
    parser.add_argument(
        "--cases",
        type=_parse_case,
        nargs="*",
        help="optional explicit BATCH:HEADS:AVERAGE_KV cases",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--mask", choices=("causal", "local", "full"), default="causal")
    parser.add_argument("--window-left", type=int, default=255)
    parser.add_argument("--window-right", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="compile all implementations first, then rotate their timing order for every group",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("sweep_hstu_qlen1.py requires a CUDA GPU")

    cases = list(args.cases) if args.cases else _default_cases()
    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    window_size = (-1, 0) if args.mask == "causal" else (args.window_left, args.window_right) if args.mask == "local" else (-1, -1)
    alpha = 0.7
    scaling_seqlen = 2048.0
    payload: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cudnn_module": cudnn.__file__,
        "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
        "direction": args.direction,
        "implementations": args.implementations,
        "dtype": args.dtype,
        "head_dim": args.head_dim,
        "mask": args.mask,
        "window_size": window_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "groups": args.groups,
        "interleave": args.interleave,
        "cases": [],
    }

    for case_index, (batch_size, heads, average_kv) in enumerate(cases):
        tensors = _make_inputs(
            batch_size,
            heads,
            args.head_dim,
            average_kv,
            dtype,
            device,
            seed=1000 + case_index,
            random_values=False,
        )
        k_lengths = tensors["k_lengths"]
        assert isinstance(k_lengths, list)
        case_result: dict[str, object] = {
            "batch_size": batch_size,
            "heads": heads,
            "average_kv_target": average_kv,
            "average_kv": sum(k_lengths) / batch_size,
            "min_kv": min(k_lengths),
            "max_kv": max(k_lengths),
            "total_kv": sum(k_lengths),
            "base_ctas": batch_size * heads,
            "results": {},
        }

        # Rotate the order so the same implementation is not always measured
        # first while the device is changing clocks or temperature.
        rotate = case_index % len(args.implementations)
        implementations = args.implementations[rotate:] + args.implementations[:rotate]
        compiled: dict[str, tuple[object, object, float]] = {}
        for implementation in implementations:
            try:
                if args.direction == "forward":
                    outputs, run, compile_seconds = _compile_forward(
                        tensors,
                        max(k_lengths),
                        window_size,
                        alpha,
                        scaling_seqlen,
                        implementation,
                    )
                else:
                    outputs, run, compile_seconds = _compile_backward(
                        tensors,
                        max(k_lengths),
                        window_size,
                        alpha,
                        scaling_seqlen,
                        implementation,
                    )
                if args.interleave:
                    compiled[implementation] = (outputs, run, compile_seconds)
                else:
                    measurement = {
                        "compile_seconds": compile_seconds,
                        **_measure_ms(run, args.warmup, args.iterations, args.groups),
                    }
                    case_result["results"][implementation] = measurement
                    del outputs, run
            except Exception as exc:  # Keep a long sweep useful after one failed variant.
                case_result["results"][implementation] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(
                    "ERROR "
                    + json.dumps(
                        {
                            "batch_size": batch_size,
                            "heads": heads,
                            "average_kv_target": average_kv,
                            "implementation": implementation,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                torch.cuda.empty_cache()

        if args.interleave and compiled:
            for _ in range(args.warmup):
                for implementation in implementations:
                    if implementation in compiled:
                        compiled[implementation][1]()
            torch.cuda.synchronize()
            samples: dict[str, list[float]] = {implementation: [] for implementation in compiled}
            active = [implementation for implementation in implementations if implementation in compiled]
            for group_idx in range(args.groups):
                group_rotate = group_idx % len(active)
                group_order = active[group_rotate:] + active[:group_rotate]
                for implementation in group_order:
                    run = compiled[implementation][1]
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        run()
                    end.record()
                    end.synchronize()
                    samples[implementation].append(start.elapsed_time(end) / args.iterations)
            for implementation, implementation_samples in samples.items():
                case_result["results"][implementation] = {
                    "compile_seconds": compiled[implementation][2],
                    "median_ms": statistics.median(implementation_samples),
                    "min_ms": min(implementation_samples),
                    "max_ms": max(implementation_samples),
                    "samples_ms": implementation_samples,
                }
            compiled.clear()

        print("CASE " + json.dumps(case_result, sort_keys=True), flush=True)
        payload["cases"].append(case_result)
        del tensors
        gc.collect()
        torch.cuda.empty_cache()

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="", flush=True)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
