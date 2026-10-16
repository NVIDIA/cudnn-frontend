# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native blk64 and explicitly grouped blk64 FWD/BWD on SM100/SM103.

Install this checkout, then run on an idle GPU:
    python benchmark/bsa/benchmark_sm100_blk64_grouped.py --lengths 32768 131072 --topks 64 256
    python benchmark/bsa/benchmark_sm100_blk64_grouped.py --lengths 16384 32768 65536 131072 --sparsities .95 .90 .50 0
    python benchmark/bsa/benchmark_sm100_blk64_grouped.py --lengths 512 --topks 2 --heads 2 --seeds 4593 --rounds 3 --repeats 1

Each direction compares the ordinary public API with/without ``blk64_plan`` on
identical inputs. CUDA Graph timing includes GPU permutations, native CSR
construction, resets and pre/postprocessing, and excludes CPU preparation,
allocation, compilation and host enqueue time. Plan preparation is timed
separately. Numbers are useful sparse TFLOPS, not dense-equivalent TFLOPS:
FWD FLOPs = 4 * B * H * N * topK * 64 * D; BWD uses factor 10.

Synthetic patterns force Q pairs, KV pairs, both, or neither. Pair discovery
reports actual eligibility: additional pairs can occur, especially at 0%
sparsity. Exact pairs need not be adjacent. Each row is sorted and unique.
Warmup and measurement use randomized variant order. Reported throughputs are
medians of seed medians; speedups are medians of within-seed latency ratios.
Optional JSONL output contains provenance, per-round samples and accuracy
checks. No results are written unless --output is supplied.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch
import cudnn
from cudnn.block_sparse_attention import api


def make_routes(heads, blocks, topk, pattern):
    qf = 2 if pattern in ("q", "both") else 1
    kf = 2 if pattern in ("kv", "both") else 1
    assert topk % kf == 0
    scores = torch.rand((1, heads, blocks // qf, blocks // kf), device="cuda")
    ids = scores.topk(topk // kf, dim=-1).indices
    if kf == 2:
        ids = torch.cat((ids, blocks - 1 - ids), dim=-1)
    if qf == 2:
        ids = torch.cat((ids, ids.flip(-2)), dim=-2)
    return ids.sort(-1).values.int()


def difference(actual, expected, *, lse=False, grad=False):
    total_error = total_ref = maximum = 0.0
    x, y = actual.reshape(-1), expected.reshape(-1)
    for start in range(0, x.numel(), 8 * 1024**2):
        a, b = x[start : start + 8 * 1024**2].float(), y[start : start + 8 * 1024**2].float()
        torch.testing.assert_close(a, b, atol=1e-4 if lse else 0.03, rtol=1e-4 if lse else 0.03)
        delta = a - b
        maximum = max(maximum, delta.abs().max().item())
        total_error += delta.square().sum().item()
        total_ref += b.square().sum().item()
    relative = (total_error / max(total_ref, 1e-30)) ** 0.5
    assert relative < (0.001 if lse else 0.012 if grad else 0.008), relative
    return dict(max_abs=maximum, relative_l2=relative)


def measure(callables, rounds, seed, repeats=3):
    """Randomized repeated-call graphs, warmed together before collecting samples."""
    graphs, captured_outputs, samples = {}, {}, {name: [] for name in callables}
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(capture_stream):
            capture_order = list(callables)
            random.Random(seed).shuffle(capture_order)
            for name in capture_order:
                fn = callables[name]
                fn()
                fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                captured_outputs[name] = []
                with torch.cuda.graph(graph):
                    for _ in range(repeats):
                        captured_outputs[name].append(fn())
                graphs[name] = graph
            # Sustained GPU work before measurement avoids a cold first variant.
            for _ in range(8):
                for graph in graphs.values():
                    graph.replay()
            torch.cuda.synchronize()
            rng = random.Random(seed)
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for _ in range(rounds):
                order = list(graphs)
                rng.shuffle(order)
                for name in order:
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) / repeats)
    finally:
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()
    return {name: dict(median_ms=statistics.median(values), samples_ms=values) for name, values in samples.items()}


def run_case(args, n, topk, pattern, seed):
    torch.manual_seed(seed)
    q, k, v, dout = [torch.randn((1, n, args.heads, 128), device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    routes = make_routes(args.heads, n // 64, topk, pattern)
    torch.cuda.synchronize()
    start = time.perf_counter()
    plan = api.block_sparse_attention_blk64_prepare(routes, num_kv_blocks=n // 64, bucket_size_blocks=args.bucket_size_blocks)
    torch.cuda.synchronize()
    plan_ms = (time.perf_counter() - start) * 1000
    kwargs = dict(layout="bshd", sparse_block_size=64)
    out, lse = api.block_sparse_attention_forward(q, k, v, routes, **kwargs)
    functions = {
        "fwd": {
            "native": lambda: api.block_sparse_attention_forward(q, k, v, routes, **kwargs),
            "grouped": lambda: api.block_sparse_attention_forward(q, k, v, routes, **kwargs, blk64_plan=plan),
        },
        "bwd": {
            "native": lambda: api.block_sparse_attention_backward(dout, q, k, v, out, lse, routes, **kwargs, bucket_size_blocks=args.bucket_size_blocks),
            "grouped": lambda: api.block_sparse_attention_backward(
                dout, q, k, v, out, lse, routes, **kwargs, bucket_size_blocks=args.bucket_size_blocks, blk64_plan=plan
            ),
        },
    }
    record = dict(
        n=n,
        topk=topk,
        pattern=pattern,
        seed=seed,
        sparsity=1 - topk * 64 / n,
        plan_ms=plan_ms,
        eligible=dict(fwd=plan.forward_enabled, bwd=plan.backward_enabled),
        reason=dict(fwd=plan.forward_reason, bwd=plan.backward_reason),
    )
    directions = ("fwd", "bwd") if args.direction == "both" else (args.direction,)
    for direction in directions:
        callables = functions[direction]
        reference = callables["native"]()

        def check():
            return [
                difference(a, b, lse=direction == "fwd" and i == 1, grad=direction == "bwd") for i, (a, b) in enumerate(zip(callables["grouped"](), reference))
            ]

        before = check()
        timings = measure(callables, args.rounds, seed + (direction == "bwd"), repeats=args.repeats)
        for timing in timings.values():
            timing["tflops"] = (4 if direction == "fwd" else 10) * args.heads * n * topk * 64 * 128 / timing["median_ms"] / 1e9
        speedup = timings["native"]["median_ms"] / timings["grouped"]["median_ms"]
        record[direction] = dict(timings=timings, speedup=speedup, accuracy_before=before, accuracy_after=check())
        print(
            f"N={n} topK={topk} pattern={pattern} seed={seed} {direction.upper()} "
            f"native={timings['native']['tflops']:.1f} grouped={timings['grouped']['tflops']:.1f} TFLOPS "
            f"speedup={speedup:.3f}x eligible={record['eligible'][direction]} plan={plan_ms:.1f} ms",
            flush=True,
        )
    return record


def summarize(records):
    print("\n| N | topK | sparsity | pattern | direction | native ms | grouped ms | native TFLOPS | grouped TFLOPS | speedup | plan ms |")
    print("| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    cases = dict.fromkeys((r["n"], r["topk"], r["pattern"]) for r in records)
    for n, topk, pattern in cases:
        rows = [r for r in records if (r["n"], r["topk"], r["pattern"]) == (n, topk, pattern)]
        for direction in ("fwd", "bwd"):
            if direction not in rows[0]:
                continue
            values = [
                statistics.median(r[direction]["timings"][variant][metric] for r in rows)
                for metric in ("median_ms", "tflops")
                for variant in ("native", "grouped")
            ]
            speedup = statistics.median(r[direction]["speedup"] for r in rows)
            plan_ms = statistics.median(r["plan_ms"] for r in rows)
            print(
                f"| {n} | {topk} | {rows[0]['sparsity']:.2%} | {pattern} | {direction.upper()} | "
                + " | ".join(f"{v:.3f}" for v in values)
                + f" | {speedup:.3f}x | {plan_ms:.1f} |"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", nargs="+", type=int, default=[32768, 131072])
    density = parser.add_mutually_exclusive_group()
    density.add_argument("--topks", nargs="+", type=int, default=None)
    density.add_argument("--sparsities", nargs="+", type=float, help="Fractions removed; round to the nearest positive even topK")
    parser.add_argument("--patterns", nargs="+", choices=["both", "q", "kv", "neither"], default=["both"])
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--seeds", nargs="+", type=int, default=[4593, 4594, 4595])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3, help="Calls per CUDA Graph; use 1 for long dense cases to limit storage/work")
    parser.add_argument("--bucket-size-blocks", type=int, default=1024, help="Backward bucket width in Q64 blocks, matching the native API")
    parser.add_argument("--direction", choices=["both", "fwd", "bwd"], default="both")
    parser.add_argument("--output", type=Path, help="Optional JSONL path outside the source tree")
    args = parser.parse_args()
    if any(n <= 0 or n % 128 for n in args.lengths) or min(args.heads, args.rounds, args.repeats, args.bucket_size_blocks) <= 0:
        parser.error("lengths must be positive multiples of 128; heads, rounds, repeats and bucket size must be positive")
    if args.sparsities is not None and any(not 0 <= s < 1 for s in args.sparsities):
        parser.error("sparsities must be in [0, 1)")
    if args.sparsities is None:
        args.topks = [64, 256] if args.topks is None else args.topks
        if any(k <= 0 or k % 2 or k > min(args.lengths) // 64 for k in args.topks):
            parser.error("topks must be positive even counts <= the shortest sequence's KV64 block count")
    cases = []
    for n in args.lengths:
        topks = args.topks if args.sparsities is None else [max(2, min(n // 64, 2 * round((1 - s) * (n // 64) / 2))) for s in args.sparsities]
        if len(set(topks)) != len(topks):
            parser.error(f"duplicate topK cases after rounding at N={n}")
        cases.extend((n, k, pattern) for k in topks for pattern in args.patterns)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("requires an available SM100 or SM103 GPU")
    root = Path(__file__).resolve().parents[2]
    expected_api = root / "python/cudnn/block_sparse_attention/api.py"
    if Path(api.__file__).resolve() != expected_api.resolve():
        parser.error(f"cudnn imports from {api.__file__}; install this checkout before benchmarking")
    sources = [Path(__file__), expected_api, *sorted((root / "python/cudnn/block_sparse_attention").glob("_blk64*.py"))]
    sources += [
        root / f"python/cudnn/block_sparse_attention/csrc/{direction}/sm100_blk{block}/bsa_{direction}_sm100.py"
        for direction in ("fwd", "bwd")
        for block in (64, 128)
    ]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    provenance = dict(
        gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=torch.__version__,
        cutedsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        cudnn_path=cudnn.__file__,
        commit=commit,
        sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    print(json.dumps(dict(provenance=provenance)), flush=True)
    output = args.output.open("x") if args.output else None
    records = []
    try:
        if output:
            output.write(json.dumps(dict(provenance=provenance)) + "\n")
            output.flush()
        for n, topk, pattern in cases:
            for seed in args.seeds:
                record = run_case(args, n, topk, pattern, seed)
                records.append(record)
                if output:
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        if output:
            output.close()
    summarize(records)


if __name__ == "__main__":
    main()
