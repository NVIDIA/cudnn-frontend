#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure cuDNN Frontend deterministic Top-K overhead.

The timed region is CUDA-graph replay only.  Compilation, allocation, capture,
validation, and a 256 MiB L2 flush are outside each measured interval.
"""

import argparse
import csv
import statistics
import sys

import torch
from cudnn.deepseek_sparse_attention.indexer_top_k import (
    indexer_top_k_decode_varlen as topk_kernel,
)


def _capture(call, warmup):
    """Warm one implementation, capture it, and return its graph and outputs."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            output = call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _interleaved_median_graph_ms(graphs, flush, samples):
    """Measure graph replay medians while rotating implementation order."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = {name: [] for name in graphs}
    names = list(graphs)
    for sample in range(samples):
        offset = sample % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            flush.zero_()
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end))
    return {name: statistics.median(values) for name, values in times.items()}


def _validate(scores, seq_len, k, outputs):
    """Require exact Top-K values, gather consistency, and canonical cuDNN ids."""
    valid_k = min(seq_len, k)
    expected = torch.topk(scores[:, :seq_len], valid_k, dim=-1, sorted=False).values.sort(dim=-1).values
    for name, (values, indices) in outputs.items():
        actual = values[:, :valid_k].sort(dim=-1).values
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        gathered = scores.gather(1, indices[:, :valid_k].to(torch.int64))
        torch.testing.assert_close(values[:, :valid_k], gathered, atol=0, rtol=0)
        assert torch.all(indices[:, valid_k:] == -1), name
        assert torch.all(torch.isneginf(values[:, valid_k:])), name
        if name.startswith("cudnn_det") and valid_k > 1:
            # The deterministic contract canonicalizes valid local ids.
            assert torch.all(indices[:, 1:valid_k] >= indices[:, : valid_k - 1]), name


def _make_call(scores, lengths, k, deterministic, ordering="auto"):
    """Compile one fixed strategy and return an allocation-free launch closure."""
    rows, cols = scores.shape
    compiled = topk_kernel.compile_topk_kernel(
        scores.dtype,
        rows,
        cols,
        k,
        1,
        return_val=True,
        deterministic=deterministic,
        deterministic_ordering=ordering,
    )
    buffer = torch.empty(
        rows,
        topk_kernel.buffer_numbers(scores.dtype),
        cols,
        dtype=torch.int32,
        device=scores.device,
    )
    indices = torch.empty(rows, k, dtype=torch.int32, device=scores.device)
    values = torch.empty(rows, k, dtype=scores.dtype, device=scores.device)

    def call():
        topk_kernel.launch_topk_kernel(compiled, scores, lengths, indices, values, buffer, 1)
        return {"indices": indices, "values": values}

    return call


def benchmark_case(rows, cols, k, dtype, args):
    """Validate and time every implementation for one Top-K shape."""
    torch.manual_seed(args.seed)
    scores = torch.randn(rows, cols, dtype=dtype, device="cuda")
    if args.ties:
        scores = (scores * 8).round_()
    seq_len = min(args.seq_len if args.seq_len is not None else cols, cols)
    lengths = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")

    calls = {
        "cudnn_fast": _make_call(scores, lengths, k, False),
        "cudnn_det_auto": _make_call(scores, lengths, k, True),
    }
    if args.compare_ordering:
        calls["cudnn_det_bitmap"] = _make_call(scores, lengths, k, True, "bitmap")
        calls["cudnn_det_bitonic"] = _make_call(scores, lengths, k, True, "bitonic")
    graphs = {}
    raw_outputs = {}
    try:
        for name, call in calls.items():
            graph, output = _capture(call, args.warmup)
            graphs[name] = graph
            raw_outputs[name] = (output["values"], output["indices"])
        _validate(scores, seq_len, k, raw_outputs)

        # Re-run deterministic graphs and require exact output identity.
        det_baselines = {name: (values.clone(), indices.clone()) for name, (values, indices) in raw_outputs.items() if name.startswith("cudnn_det")}
        for _ in range(5):
            for name in det_baselines:
                graphs[name].replay()
        torch.cuda.synchronize()
        for name, (values, indices) in det_baselines.items():
            current_values, current_indices = raw_outputs[name]
            assert torch.equal(current_values, values), name
            assert torch.equal(current_indices, indices), name

        timings = _interleaved_median_graph_ms(graphs, flush, args.samples)
        return {
            "rows": rows,
            "cols": cols,
            "seq_len": seq_len,
            "top_k": k,
            "dtype": str(dtype).removeprefix("torch."),
            "ties": args.ties,
            **{f"{name}_ms": value for name, value in timings.items()},
            **{f"{name}_over_fast": value / timings["cudnn_fast"] for name, value in timings.items() if name != "cudnn_fast"},
        }
    finally:
        for graph in graphs.values():
            graph.reset()


def main():
    """Run the requested benchmark matrix and optionally write its CSV rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--cols", type=int, nargs="+", default=[8192, 32768, 131072])
    parser.add_argument(
        "--seq-len",
        type=int,
        help="Effective length for every row; defaults to cols and is clamped to it.",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--ties", action="store_true")
    parser.add_argument(
        "--compare-ordering",
        action="store_true",
        help="Time forced bitmap and bitonic deterministic ordering paths.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-det-over-fast",
        type=float,
        help="Fail if deterministic / non-deterministic cuDNN exceeds this ratio.",
    )
    parser.add_argument("--csv")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    writer = None
    handle = open(args.csv, "x", newline="") if args.csv else None
    try:
        for rows in args.rows:
            for cols in args.cols:
                for k in args.top_k:
                    if k > cols:
                        continue
                    result = benchmark_case(rows, cols, k, dtype, args)
                    if writer is None:
                        streams = [sys.stdout] + ([handle] if handle else [])
                        writer = [csv.DictWriter(stream, fieldnames=result.keys()) for stream in streams]
                        for item in writer:
                            item.writeheader()
                    for item in writer:
                        item.writerow(result)
                    sys.stdout.flush()
                    if handle:
                        handle.flush()
                    if args.max_det_over_fast is not None and result["cudnn_det_auto_over_fast"] > args.max_det_over_fast:
                        raise RuntimeError(
                            "cuDNN deterministic Top-K missed the performance gate: "
                            f"{result['cudnn_det_auto_over_fast']:.3f} > "
                            f"{args.max_det_over_fast:.3f} for "
                            f"rows={rows}, cols={cols}, k={k}, dtype={args.dtype}, "
                            f"ties={args.ties}"
                        )
                    torch.cuda.empty_cache()
    finally:
        if handle:
            handle.close()


if __name__ == "__main__":
    main()
