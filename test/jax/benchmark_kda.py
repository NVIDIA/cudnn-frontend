# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare warm JAX/torch calls with the full Frost GPU launch sequence.

Run with one visible GPU and XLA_PYTHON_CLIENT_PREALLOCATE=false. Both backends
use Frost and checkpoint recomputation in backward. JAX uses decay-warmup splits;
torch uses automatic scheduling, including piece chains. When schedules differ,
blocking-minus-raw includes that difference, not just framework overhead.
Raw GPU time uses CUDA events around batched, fixed-buffer CUDA-graph replay.
Blocking minus raw includes host dispatch, allocation, launch gaps and device
synchronization; host dispatch alone must not be subtracted from GPU time.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from cudnn.jax import (
    kimi_delta_attention as jax_kda,
    kimi_delta_attention_fwd as jax_fwd,
    kimi_delta_attention_bwd as jax_bwd,
)
from cudnn.linear_attention.ops.kda import (
    kimi_delta_attention as torch_kda,
    kda_bwd as torch_bwd,
)


def measure(fn, repetitions, warmup):
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    times, blocking = [], []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = fn()
        dispatched = time.perf_counter_ns()
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((dispatched - start) / 1000)
        blocking.append((end - start) / 1000)
    return dict(
        dispatch_median_us=float(np.median(times)),
        blocking_median_us=float(np.median(blocking)),
    )


def capture(fn, stream):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def measure_gpu(fn, stream, repetitions, warmup, batch_size):
    batch, outputs = capture(lambda: [fn() for _ in range(batch_size)], stream)
    for _ in range(warmup):
        batch.replay()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    elapsed = []
    for _ in range(repetitions):
        start.record(stream)
        batch.replay()
        end.record(stream)
        end.synchronize()
        elapsed.append(start.elapsed_time(end) * 1000 / batch_size)
    return float(np.median(elapsed)), outputs[-1]


def check_parity(actual, expected):
    errors = []
    for a, b in zip(actual, expected, strict=True):
        a = a.float().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, dtype=np.float32)
        b = b.float().cpu().numpy()
        error = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-10))
        assert error < 0.04, error
        errors.append(error)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--raw-batch-size", type=int, default=32)
    parser.add_argument("--command-buffer", action="store_true")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if min(options.tokens, options.heads, options.dim, options.repetitions, options.warmup, options.raw_batch_size) <= 0:
        parser.error("shape, repetition, warmup and batch sizes must be positive")
    rng = np.random.default_rng(1)
    shape = (options.tokens, options.heads, options.dim)
    q, k, v = (jnp.asarray(rng.normal(0, 0.08, shape), jnp.bfloat16) for _ in range(3))
    g = jnp.full(shape, -0.1, jnp.float32)
    beta = jnp.full(shape[:2], 0.5, jnp.float32)
    cu = jnp.asarray([0, options.tokens], jnp.int32)
    arrays = (q, k, v, g, beta, cu)
    torch_arrays = tuple(
        torch.tensor(
            np.asarray(x, dtype=np.int32 if x.dtype == jnp.int32 else np.float32),
            device="cuda",
            dtype=(torch.int32 if x.dtype == jnp.int32 else torch.bfloat16 if x.dtype == jnp.bfloat16 else torch.float32),
        )
        for x in arrays
    )
    compiler_options = {"xla_gpu_enable_command_buffer": "CUSTOM_CALL", "xla_gpu_graph_min_graph_size": 1} if options.command_buffer else {}
    compiled_fwd = jax.jit(jax_kda, compiler_options=compiler_options)
    output, _ = compiled_fwd(*arrays)
    _, _, residual = jax_fwd(*arrays)
    do = jnp.ones_like(output)
    compiled_bwd = jax.jit(jax_bwd, compiler_options=compiler_options)
    grads = compiled_bwd(residual, do)
    jax.block_until_ready((output, grads))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tout = torch_kda(*torch_arrays, plan_name="kda_frost")[0]
        tdo = torch.ones_like(tout)
        tgrads = torch_bwd(tdo, *torch_arrays, options.dim**-0.5, plan_name="kda_frost")
    torch.cuda.synchronize()
    parity = dict(jax_relative_l2=check_parity((output, *grads[:5]), (tout, *tgrads[:5])))
    jax_functions = dict(
        jax_fwd=lambda: compiled_fwd(*arrays),
        jax_bwd=lambda: compiled_bwd(residual, do),
    )
    torch_functions = dict(
        torch_fwd=lambda: torch_kda(*torch_arrays, plan_name="kda_frost"),
        torch_bwd=lambda: torch_bwd(tdo, *torch_arrays, options.dim**-0.5, plan_name="kda_frost"),
    )
    graphs = {}
    with torch.cuda.stream(stream):
        for name, fn in torch_functions.items():
            for _ in range(options.warmup):
                fn()
            torch.cuda.synchronize()
            graph, captured = capture(fn, stream)
            direction = name.removeprefix("torch_")
            parity[f"raw_{direction}_relative_l2"] = check_parity(
                captured[:1] if direction == "fwd" else captured[:5], (tout,) if direction == "fwd" else tgrads[:5]
            )
            graphs[direction] = (graph, captured)
    import cutlass
    import cudnn

    report = dict(
        shape=shape,
        dtype="bfloat16",
        checkpoint=0,
        batch_invariant=False,
        jax_schedule="decay-warmup split",
        torch_schedule="automatic (may use piece chains)",
        command_buffer=options.command_buffer,
        repetitions=options.repetitions,
        warmup=options.warmup,
        raw_batch_size=options.raw_batch_size,
        versions=dict(jax=jax.__version__, torch=torch.__version__, cutedsl=cutlass.__version__),
        cudnn_path=cudnn.__file__,
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        parity=parity,
    )
    from cuda.bindings import runtime as rt

    rt.cudaProfilerStart()
    for name, fn in jax_functions.items():
        with torch.cuda.nvtx.range(name):
            report[name] = measure(fn, options.repetitions, options.warmup)
    with torch.cuda.stream(stream):
        for name, fn in torch_functions.items():
            with torch.cuda.nvtx.range(name):
                report[name] = measure(fn, options.repetitions, options.warmup)
        for direction, (graph, captured) in graphs.items():
            with torch.cuda.nvtx.range(f"raw_{direction}"):
                gpu_us, batch_output = measure_gpu(torch_functions[f"torch_{direction}"], stream, options.repetitions, options.warmup, options.raw_batch_size)
                replay = measure(graph.replay, options.repetitions, options.warmup)
            parity[f"raw_{direction}_batch_relative_l2"] = check_parity(
                batch_output[:1] if direction == "fwd" else batch_output[:5], (tout,) if direction == "fwd" else tgrads[:5]
            )
            report[f"raw_{direction}"] = dict(gpu_median_us=gpu_us, **replay)
            for backend in ("jax", "torch"):
                timing = report[f"{backend}_{direction}"]
                timing["blocking_minus_raw_us"] = timing["blocking_median_us"] - gpu_us
    serialized = json.dumps(report, indent=2)
    if options.output:
        options.output.write_text(serialized + "\n")
    print(serialized, flush=True)
    rt.cudaProfilerStop()


if __name__ == "__main__":
    main()
