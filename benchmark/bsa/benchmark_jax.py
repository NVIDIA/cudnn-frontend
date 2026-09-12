# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare warm BSA API throughput with CUDA-graph device execution time."""

import argparse
import cProfile
import gc
import io
import json
import os
from pathlib import Path
import pstats
import statistics
import subprocess
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from cuda.bindings import runtime as cuda

from cudnn import BSA

CASES = {
    "small": (1, 2, 256, 512, 64, 2, None),
    "medium": (2, 8, 2048, 4096, 128, 8, None),
    "multi_bucket": (1, 8, 16384, 16384, 128, 16, 64),
}


def synchronize():
    torch.cuda.synchronize()


def require_exclusive_gpu():
    pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "--id=" + os.environ.get("CUDA_VISIBLE_DEVICES", "0")], text=True
    )
    others = [int(pid) for pid in pids.split() if int(pid) != os.getpid()]
    if others:
        raise RuntimeError(f"Other GPU processes detected; discard overlapping samples and rerun: {others}")


def measure(fn, iterations, samples):
    require_exclusive_gpu()
    totals, dispatches = [], []
    for _ in range(samples):
        synchronize()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            output = fn()
        submitted = time.perf_counter_ns()
        synchronize()
        finished = time.perf_counter_ns()
        totals.append((finished - start) / (1000 * iterations))
        dispatches.append((submitted - start) / (1000 * iterations))
    require_exclusive_gpu()
    return dict(median_us=statistics.median(totals), min_us=min(totals), max_us=max(totals), dispatch_us=statistics.median(dispatches), samples_us=totals)


def capture(fn, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            output = fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph.instantiate()
    return graph, output


def cuda_result(result):
    status, *values = result
    if status != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError(status)
    return values[0] if values else None


def device_time(single_graph, repeats, iterations, samples):
    graph = cuda_result(cuda.cudaGraphCreate(0))
    dependencies = []
    for _ in range(repeats):
        node = cuda_result(cuda.cudaGraphAddChildGraphNode(graph, dependencies, len(dependencies), single_graph.raw_cuda_graph()))
        dependencies = [node]
    executable = cuda_result(cuda.cudaGraphInstantiate(graph, 0))
    try:
        replay = lambda: cuda_result(cuda.cudaGraphLaunch(executable, torch.cuda.current_stream().cuda_stream))
        return measure_device(replay, repeats, iterations, samples)
    finally:
        cuda_result(cuda.cudaGraphExecDestroy(executable))
        cuda_result(cuda.cudaGraphDestroy(graph))


def measure_device(replay, repeats, iterations, samples):
    require_exclusive_gpu()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    warmup_until = time.perf_counter() + 0.25
    while time.perf_counter() < warmup_until:
        replay()
        synchronize()
    times = []
    for _ in range(samples):
        synchronize()
        start.record()
        for _ in range(iterations):
            replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (iterations * repeats))
    require_exclusive_gpu()
    return dict(median_us=statistics.median(times), min_us=min(times), max_us=max(times), samples_us=times)


def make_inputs(case, layout):
    b, h, sq, sk, d, count, bucket = CASES[case]
    rng = np.random.default_rng(19)
    shapes = ((b, h, sq, d), (b, h, sk, d), (b, h, sk, d), (b, h, sq, d))
    arrays = [rng.normal(size=shape).astype(np.float32) for shape in shapes]
    if layout == "bshd":
        arrays = [np.ascontiguousarray(x.transpose(0, 2, 1, 3)) for x in arrays]
    index = np.empty((b, h, sq // 128, count), np.int32)
    for row in index.reshape(-1, count):
        row[:] = rng.choice(sk // 128, count, replace=False)
    torch_inputs = tuple(torch.tensor(x, device="cuda", dtype=torch.bfloat16) for x in arrays) + (torch.tensor(index, device="cuda"),)
    jax_inputs = tuple(jnp.asarray(x, jnp.bfloat16) for x in arrays) + (jnp.asarray(index),)
    synchronize()
    return torch_inputs, jax_inputs, dict(block_sparse_num=count, layout=layout, sparse_block_size=128), bucket


def benchmark(case, layout, args):
    ti, ji, options, bucket = make_inputs(case, layout)
    tq, tk, tv, tdo, tind = ti
    jq, jk, jv, jdo, jind = ji
    tf = BSA.block_sparse_attention_forward(tq, tk, tv, tind, **options)
    jf = BSA.block_sparse_attention_forward_jax(jq, jk, jv, jind, **options)
    torch_fwd = lambda: BSA.block_sparse_attention_forward(tq, tk, tv, tind, **options)
    torch_bwd = lambda: BSA.block_sparse_attention_backward(tdo, tq, tk, tv, *tf, tind, bucket_size_blocks=bucket, **options)
    jax_fwd = lambda q, k, v, i: BSA.block_sparse_attention_forward_jax(q, k, v, i, **options)
    jax_bwd = lambda do, q, k, v, o, lse, i: BSA.block_sparse_attention_backward_jax(do, q, k, v, o, lse, i, bucket_size_blocks=bucket, **options)
    results = []
    for direction, torch_fn, jax_fn, inputs in (
        ("forward", torch_fwd, jax_fwd, (jq, jk, jv, jind)),
        ("backward", torch_bwd, jax_bwd, (jdo, jq, jk, jv, *jf, jind)),
    ):
        compiled = jax.jit(jax_fn)
        compiled.lower(*inputs).compile()
        command_buffer = jax.jit(jax_fn, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL", "xla_gpu_graph_min_graph_size": 1})
        command_buffer.lower(*inputs).compile()
        eager = lambda: jax_fn(*inputs)
        jitted = lambda: compiled(*inputs)
        jax_graph = lambda: command_buffer(*inputs)
        for fn in (torch_fn, eager, jitted, jax_graph):
            for _ in range(5):
                fn()
            synchronize()
        expected_outputs = torch_fn()
        for run in (jitted, jax_graph):
            for actual, expected in zip(run(), expected_outputs):
                np.testing.assert_allclose(np.asarray(actual.astype(jnp.float32)), expected.float().cpu().numpy(), atol=3e-2, rtol=3e-2)
        single_graph, single_outputs = capture(torch_fn, 1)
        record = dict(case=case, layout=layout, direction=direction, shape=CASES[case])
        with torch.cuda.nvtx.range(f"{case}/{layout}/{direction}/raw_sequence"):
            record["raw_sequence"] = device_time(single_graph, args.graph_repeats, args.iterations, args.samples)
        for actual, expected in zip(single_outputs, expected_outputs):
            torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        for name, fn in (
            ("jax_jit", jitted),
            ("jax_command_buffer", jax_graph),
            ("torch_eager", torch_fn),
            ("jax_eager", eager),
            ("torch_graph", single_graph.replay),
        ):
            with torch.cuda.nvtx.range(f"{case}/{layout}/{direction}/{name}"):
                record[name] = measure(fn, args.iterations, args.samples)
        for actual, expected in zip(single_outputs, expected_outputs):
            torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        if args.cpu_profile:
            for name, fn in (("torch_eager", torch_fn), ("jax_eager", eager), ("jax_jit", jitted)):
                profiler = cProfile.Profile()
                for _ in range(10):
                    profiler.runcall(fn)
                synchronize()
                output = io.StringIO()
                pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative").print_stats(20)
                record[name]["cpu_profile"] = output.getvalue()
        results.append(record)
        print(json.dumps(record), flush=True)
        del single_graph, single_outputs
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--layouts", nargs="+", choices=("bhsd", "bshd"), default=["bhsd", "bshd"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--graph-repeats", type=int, default=32)
    parser.add_argument("--cpu-profile", action="store_true", help="Record separate, untimed Python profiles")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (10, 0)
    assert len(jax.devices()) == 1
    gc.disable()
    report = dict(
        versions=dict(jax=jax.__version__, torch=torch.__version__),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        hostname=subprocess.check_output(["hostname"], text=True).strip(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        command_buffer_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL", "xla_gpu_graph_min_graph_size": 1},
        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem", "--format=csv,noheader"], text=True).strip(),
        settings=vars(args) | {"output": str(args.output)},
        results=[],
    )
    for case in args.cases:
        for layout in args.layouts:
            report["results"].extend(benchmark(case, layout, args))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    gc.enable()


if __name__ == "__main__":
    main()
