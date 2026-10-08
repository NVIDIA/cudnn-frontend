#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare standalone DSA top-k policies 0/2 for one zigzag CP query shard.

All local query rows are measured, including internal scratch chunking.
Like benchmark_single_dsa.time_fn and CSA _median_graph_ms, report median
CUDA-event time per call, with a 256 MiB L2 flush before each timed replay.
Score generation, compilation, capture and sampled CPU validation are untimed.
This does not measure CP communication, indexer scoring or sparse attention.
"""

import argparse
import csv
import json
import statistics
import sys

import torch

from cudnn import DSA
from cudnn.deepseek_sparse_attention.indexer_top_k import api as topk_api


def fill_scores(scores, lengths, k, distribution):
    if distribution == "random":
        scores.uniform_(0, 1)
    elif distribution == "duplicates":
        scores.random_(0, 8)
    elif distribution == "equal":
        scores.fill_(1)
    else:
        scores.fill_(-1)
        scores[:, : k - 1].fill_(2)
        if distribution == "sparse-left":
            scores[:, k - 1 : k + 1].fill_(1)
        else:
            rows = torch.arange(scores.shape[0], device=scores.device)
            for offset in (0, 1):
                scores[rows, (lengths.to(torch.int64) - 1 - offset).clamp(min=0)] = 1


def check_samples(scores, indices, lengths_cpu, k, policy):
    q = scores.shape[0]
    rows = {0, q // 2, q - 1}
    rows.update(torch.where((lengths_cpu == k) | (lengths_cpu == k + 1))[0].tolist())
    for row in sorted(rows):
        length = int(lengths_cpu[row])
        values = scores[row, :length].cpu()
        count = min(k, length)
        actual = indices[row, :count].to(device="cpu", dtype=torch.int64)
        assert torch.all((actual >= 0) & (actual < length))
        assert actual.unique().numel() == count
        # Inputs contain no NaNs or negative zeros. Stable reverse sorting is an
        # independent exact source-index oracle; output slot order is irrelevant.
        expected = length - 1 - torch.argsort(values.flip(0), descending=True, stable=True)[:count]
        if policy == 2:
            torch.testing.assert_close(actual.sort().values, expected.sort().values, atol=0, rtol=0)
        else:
            torch.testing.assert_close(values[actual].sort().values, values[expected].sort().values, atol=0, rtol=0)


def benchmark(n, k, args):
    q, rank = args.num_queries, args.cp_rank
    half = q // 2
    lengths_cpu = torch.cat((torch.arange(rank * half + 1, (rank + 1) * half + 1), torch.arange(n - (rank + 1) * half + 1, n - rank * half + 1)))
    # Scores occupy 16 GiB at Q=8K, KV=512K. Include two graph
    # workspaces/outputs and headroom; allocator/driver overhead can still OOM.
    # Reassigning chunk_extra can briefly retain the previous scratch chunk.
    scratch = q * n * 8 if q * n * 8 <= 8 << 30 else 16 << 30
    estimate = q * n * 4 + 2 * (scratch + q * k * 4) + (2 << 30)
    free, _ = torch.cuda.mem_get_info()
    if free < estimate:
        raise RuntimeError(f"N={n}, K={k}: estimated {estimate / 2**30:.1f} GiB required, {free / 2**30:.1f} GiB free")
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    scores = torch.empty((q, n), dtype=torch.float32, device="cuda")
    lengths = lengths_cpu.to(device="cuda", dtype=torch.int32)
    fill_scores(scores, lengths, k, args.distribution)
    l2_flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    graphs, outputs = {}, {}
    times = {0: [], 2: []}
    try:
        for policy in (0, 2):

            def call(policy=policy):
                return DSA.indexer_top_k_wrapper(scores, lengths, k, next_n=1, return_val=False, tie_break=policy)

            # Follow the existing CSA graph-capture warmup on a side stream.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(args.warmup):
                    output = call()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                output = call()
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            torch.cuda.synchronize()
            del output
            graph = torch.cuda.CUDAGraph()
            graphs[policy] = graph  # Include partially captured graphs in cleanup.
            with torch.cuda.graph(graph):
                output = call()
            outputs[policy] = output
            graph.replay()
            torch.cuda.synchronize()
            assert output["values"] is None
            check_samples(scores, output["indices"], lengths_cpu, k, policy)
        # CPU references can leave the GPU idle; warm both policies together.
        for _ in range(args.warmup):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        end.synchronize()  # Prime lazy event creation outside measured intervals.
        for sample in range(args.samples):
            for policy in ((0, 2) if sample % 2 == 0 else (2, 0)):
                l2_flush.zero_()  # Match DSA time_fn: flush cost stays outside the events.
                start.record()
                graphs[policy].replay()
                end.record()
                end.synchronize()
                times[policy].append(start.elapsed_time(end))
        medians = {policy: statistics.median(values) for policy, values in times.items()}
        return {
            "num_queries": q,
            "num_kv": n,
            "cp_size": n // q,
            "cp_rank": rank,
            "top_k": k,
            "distribution": args.distribution,
            "mode0_ms": medians[0],
            "mode2_ms": medians[2],
            "speedup_0_over_2": medians[0] / medians[2],
            "mode0_samples_ms": json.dumps([round(t, 6) for t in times[0]]),
            "mode2_samples_ms": json.dumps([round(t, 6) for t in times[2]]),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            for graph in graphs.values():
                graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768, 131072, 524288], help="Global KV lengths")
    parser.add_argument("--num-queries", type=int, default=8192, help="Local queries per CP rank; must be even and divide each KV length")
    parser.add_argument("--cp-rank", type=int, default=0)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--distribution", choices=["random", "duplicates", "equal", "sparse-left", "sparse-right"], default="random")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", help="Write the same complete result rows to an exclusive CSV file")
    args = parser.parse_args()
    if min(args.tokens) < 1 or min(args.top_k) < 1 or max(args.top_k) > 2048 or min(args.tokens) < max(args.top_k):
        parser.error("Require tokens >= top_k and 0 < top_k <= 2048")
    if args.num_queries < 2 or args.num_queries % 2 or any(n % args.num_queries for n in args.tokens):
        parser.error("num-queries must be positive/even and divide every KV length")
    if args.cp_rank < 0 or any(args.cp_rank >= n // args.num_queries for n in args.tokens):
        parser.error("cp-rank must be within [0, KV/num-queries) for every case")
    if args.samples < 1 or args.warmup < 1:
        parser.error("samples and warmup must be positive")
    metadata = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda, **vars(args)}
    metadata.update(api_source=topk_api.__file__, kernel_source=topk_api._get_cute_dsl_topk_wrapper().__code__.co_filename)
    print("# " + json.dumps(metadata), flush=True)
    print("# Zigzag CP-local FP32 scores; cold-L2 graph GPU milliseconds per local selector call; speedup<1 means tie_break=2 is slower.", flush=True)
    handle = open(args.csv, "x", newline="") if args.csv else None
    try:
        writers = []
        for n in args.tokens:
            for k in args.top_k:
                result = benchmark(n, k, args)
                if not writers:
                    writers = [csv.DictWriter(stream, fieldnames=result) for stream in ([sys.stdout, handle] if handle else [sys.stdout])]
                    for writer in writers:
                        writer.writeheader()
                for writer in writers:
                    writer.writerow(result)
                sys.stdout.flush()
                if handle:
                    handle.flush()
                torch.cuda.empty_cache()  # Between shapes, outside all measurements.
    finally:
        if handle:
            handle.close()


if __name__ == "__main__":
    main()
