# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare prepared pooled DSA components with the matching vLLM stages.

This does not measure forward_native or score/top-k kernels. Inputs and
compilation are outside timing. Repeated calls reuse warm caches and fixed
metadata. FP8 cache writes and index expansion have optional vLLM baselines;
BF16 and MXFP4 cache writers have no upstream native baseline.
"""

import argparse
import json
import socket
import statistics
import subprocess
import time
from pathlib import Path


def measure(torch, fn, warmup, repeats, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall, span = [], []
    for _ in range(repeats):
        start.record()
        before = time.perf_counter()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - before) * 1e6 / iterations)
        span.append(start.elapsed_time(end) * 1000 / iterations)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    replay = []
    for _ in range(repeats):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        replay.append(start.elapsed_time(end) * 1000 / iterations)
    return {
        "wall_us": statistics.median(wall),
        "cuda_span_us": statistics.median(span),
        "graph_us": statistics.median(replay),
        "wall_samples_us": wall,
        "cuda_span_samples_us": span,
        "graph_samples_us": replay,
    }


def expansion_case(torch, pooled, rows, pool_size, phase, baseline):
    ids = torch.arange(512, device="cuda", dtype=torch.int32).repeat(rows, 1)
    positions = torch.full((rows,), 32768 + phase - 1, device="cuda", dtype=torch.int64)
    output = torch.empty(rows, 512 * pool_size + pool_size - 1, device="cuda", dtype=torch.int32)
    plan = pooled.ExpandIndices(ids, positions, output, pool_size)
    plan.compile()
    fn = lambda: plan.execute(ids, positions, output)
    fn()
    expected = torch.full_like(output, -1)
    expected[:, : 512 * pool_size] = torch.arange(512 * pool_size, device="cuda")
    tail = phase % pool_size
    if tail:
        expected[:, 512 * pool_size : 512 * pool_size + tail] = torch.arange(32768, 32768 + tail, device="cuda")
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    methods = {"cudnn_expand": fn}
    if baseline:
        from vllm.models.glm5next.nvidia.ops import kpool_compress as ops

        reference_out = torch.empty_like(output)

        def upstream():
            reference_out.fill_(-1)
            expanded = ops.expand_pools_and_append_tail(ids.to(torch.int64), positions.to(torch.int32) + 1, pool_size)
            reference_out.copy_(expanded)

        upstream()
        torch.testing.assert_close(reference_out, expected, rtol=0, atol=0)
        methods["vllm_expand_stage"] = upstream
    return methods


def cache_case(torch, pooled, rows, pool_size, phase, dtype, mode, baseline):
    page_size = 64
    requests = 1 if mode == "prefill" else rows
    width = page_size * {"bf16": 256, "fp8": 132, "fp4": 68}[dtype]
    pages_per_request = (rows // pool_size + page_size - 1) // page_size if mode == "prefill" else 1
    pages_per_request = max(pages_per_request, 1)
    keys = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    gates = torch.zeros_like(keys)
    ape = torch.zeros(pool_size, 128, device="cuda")
    cache = torch.zeros(requests * pages_per_request, width, device="cuda", dtype=torch.uint8)
    tail = torch.randn(requests, 2, pool_size, 128, device="cuda", dtype=torch.bfloat16)
    tail[:, 1].zero_()
    positions = (
        torch.arange(rows, device="cuda", dtype=torch.int64) if mode == "prefill" else torch.full((rows,), 32768 + phase - 1, device="cuda", dtype=torch.int64)
    )
    request_ids = torch.zeros(rows, device="cuda", dtype=torch.int64) if mode == "prefill" else torch.arange(rows, device="cuda")
    logical = positions // pool_size if mode == "prefill" else torch.zeros_like(positions)
    slots = torch.where((positions + 1) % pool_size == 0, request_ids * pages_per_request * page_size + logical, -1)
    tail_slots = request_ids * pool_size + positions % pool_size
    cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32) if mode == "prefill" else torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    args = (keys, gates, ape, cache, tail, slots, tail_slots, positions if mode == "decode" else None, cu if mode == "decode" else None)
    plan = pooled.PoolCacheUpdate(*args, pool_size=pool_size, page_size=page_size, cache_dtype=dtype, mode=mode)
    plan.compile()
    methods = {"cudnn_cache": lambda: plan.execute(*args)}
    initial_cache, initial_tail = cache.clone(), tail.clone()
    methods["cudnn_cache"]()
    if baseline and dtype == "fp8":
        from vllm.models.glm5next.nvidia import sparse_indexer
        from vllm.models.glm5next.nvidia.ops import kpool_compress as ops

        reference_cache = initial_cache.clone().view(-1, page_size, 132)
        reference_tail = initial_tail.clone()

        def upstream():
            if mode == "prefill":
                sparse_indexer._kpool_compress_insert(keys, gates, ape, reference_cache, slots, pool_size, 128, True)
                ops.kpool_seed_tail_cache(reference_tail, keys, gates, tail_slots, pool_size, 128)
            else:
                ops.kpool_decode_update_and_maybe_write_cache_batched(
                    reference_cache,
                    reference_tail,
                    tail_slots.view(rows, 1),
                    keys.view(rows, 1, 128),
                    gates.view(rows, 1, 128),
                    ape,
                    slots.view(rows, 1),
                    positions.to(torch.int32).view(rows, 1),
                    pool_size,
                    128,
                    True,
                )

        upstream()
        torch.testing.assert_close(reference_cache.flatten(1), cache, rtol=0, atol=0)
        torch.testing.assert_close(reference_tail, tail, rtol=0, atol=0)
        methods["vllm_cache_stage"] = upstream
    # Check cache state with the full numerical pytest suite before timing.
    # This harness checks shared FP8 paths exactly; it does not certify new dtypes.
    cache.copy_(initial_cache)
    tail.copy_(initial_tail)
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 64, 1024])
    parser.add_argument("--dtypes", nargs="+", choices=["bf16", "fp8", "fp4"], default=["bf16", "fp8", "fp4"])
    parser.add_argument("--pool-size", type=int, choices=[4, 16], default=4)
    parser.add_argument("--phases", type=int, nargs="+", help="Causal length modulo pool size; pool size means a complete pool")
    parser.add_argument("--with-vllm", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Run comparisons without performance measurements")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("pooled-components.json"))
    args = parser.parse_args()
    phases = args.phases or list(range(1, args.pool_size + 1))
    if min(args.rows + [args.warmup, args.repeats, args.iterations]) < 1 or any(p < 1 or p > args.pool_size for p in phases):
        parser.error("rows/timing counts must be positive and phases must be in [1, pool_size]")
    hostname = socket.gethostname()
    import torch
    import cudnn
    from cudnn.deepseek_sparse_attention import pooled

    torch.manual_seed(53)
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("Blackwell SM100 or SM103 is required")
    data = {
        "host": hostname,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_source": cudnn.__file__,
        "gpu": torch.cuda.get_device_name(),
        "scope": "component stages only; no score/top-k or forward_native",
        "options": {**vars(args), "output": str(args.output)},
        "results": [],
    }
    if args.with_vllm:
        import vllm

        data["vllm_source"] = vllm.__file__
    data["gpu_state"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout

    def run(kind, rows, phase, dtype, mode, methods):
        for method, fn in methods.items():
            result = {"component": kind, "rows": rows, "phase": phase, "dtype": dtype, "mode": mode, "method": method}
            if not args.check_only:
                result.update(measure(torch, fn, args.warmup, args.repeats, args.iterations))
            data["results"].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(data, indent=2) + "\n")
            print(json.dumps(result), flush=True)

    for rows in args.rows:
        for phase in phases:
            run("expand", rows, phase, "int32", "both", expansion_case(torch, pooled, rows, args.pool_size, phase, args.with_vllm))
        for mode in ("prefill", "decode"):
            for phase in ([args.pool_size] if mode == "prefill" else phases):
                for dtype in args.dtypes:
                    run("cache", rows, phase, dtype, mode, cache_case(torch, pooled, rows, args.pool_size, phase, dtype, mode, args.with_vllm))


if __name__ == "__main__":
    main()
