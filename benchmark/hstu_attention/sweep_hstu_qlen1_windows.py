# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep several qlen=1 attention windows while reusing inputs and kernels."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from pathlib import Path

import cudnn
import torch
from benchmark_hstu_qlen1 import _compile_backward, _compile_forward, _make_inputs
from sweep_hstu_qlen1 import _default_cases, _parse_case


def _parse_window(value: str) -> tuple[str, tuple[int, int]]:
    if value == "causal":
        return value, (-1, 0)
    try:
        left, right = (int(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("windows must be 'causal' or LEFT:RIGHT") from exc
    if left < 0 or right < 0:
        raise argparse.ArgumentTypeError("local window sizes must be nonnegative")
    return value, (left, right)


def _time_interleaved(compiled, implementations, warmup, iterations, groups):
    if not compiled:
        return {}
    for _ in range(warmup):
        for implementation in implementations:
            if implementation in compiled:
                compiled[implementation][1]()
    torch.cuda.synchronize()
    samples = {implementation: [] for implementation in compiled}
    active = [implementation for implementation in implementations if implementation in compiled]
    for group_idx in range(groups):
        order = active[group_idx % len(active) :] + active[: group_idx % len(active)]
        for implementation in order:
            run = compiled[implementation][1]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                run()
            end.record()
            end.synchronize()
            samples[implementation].append(start.elapsed_time(end) / iterations)
    return {
        implementation: {
            "compile_seconds": compiled[implementation][2],
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "samples_ms": values,
        }
        for implementation, values in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("forward", "backward"), required=True)
    parser.add_argument("--implementations", nargs="+", required=True)
    parser.add_argument("--windows", type=_parse_window, nargs="+", required=True)
    parser.add_argument("--cases", type=_parse_case, nargs="*")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("sweep_hstu_qlen1_windows.py requires a CUDA GPU")

    cases = list(args.cases) if args.cases else _default_cases()
    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    alpha = 0.7
    scaling_seqlen = 2048.0
    payload = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cudnn_module": cudnn.__file__,
        "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
        "direction": args.direction,
        "implementations": args.implementations,
        "dtype": args.dtype,
        "head_dim": args.head_dim,
        "windows": [{"name": name, "size": size} for name, size in args.windows],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "groups": args.groups,
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
        case_result = {
            "batch_size": batch_size,
            "heads": heads,
            "average_kv_target": average_kv,
            "average_kv": sum(k_lengths) / batch_size,
            "min_kv": min(k_lengths),
            "max_kv": max(k_lengths),
            "total_kv": sum(k_lengths),
            "base_ctas": batch_size * heads,
            "windows": {},
        }
        for window_index, (window_name, window_size) in enumerate(args.windows):
            rotate = (case_index + window_index) % len(args.implementations)
            implementations = args.implementations[rotate:] + args.implementations[:rotate]
            compiled = {}
            errors = {}
            for implementation in implementations:
                try:
                    if args.direction == "forward":
                        outputs, run, compile_seconds = _compile_forward(tensors, max(k_lengths), window_size, alpha, scaling_seqlen, implementation)
                    else:
                        outputs, run, compile_seconds = _compile_backward(tensors, max(k_lengths), window_size, alpha, scaling_seqlen, implementation)
                    compiled[implementation] = (outputs, run, compile_seconds)
                # One rejected candidate must not discard a long multi-shape sweep.
                except Exception as exc:  # noqa: BLE001
                    errors[implementation] = f"{type(exc).__name__}: {exc}"
                    torch.cuda.empty_cache()
            results = _time_interleaved(
                compiled,
                implementations,
                args.warmup,
                args.iterations,
                args.groups,
            )
            results.update({implementation: {"error": error} for implementation, error in errors.items()})
            case_result["windows"][window_name] = {"window_size": window_size, "results": results}
            compiled.clear()
        print(
            "CASE "
            + json.dumps(
                {
                    "batch_size": batch_size,
                    "heads": heads,
                    "average_kv_target": average_kv,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        payload["cases"].append(case_result)
        del tensors
        gc.collect()
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
