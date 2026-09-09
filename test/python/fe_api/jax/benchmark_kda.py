# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warm dispatch comparison. Use nsys CUDA/NVTX tracing for GPU kernel durations.

Run with one visible GPU and XLA_PYTHON_CLIENT_PREALLOCATE=false. Both backends
use Frost, the default split scheduler, and checkpoint recomputation in backward.
"""

import argparse
import json
import time

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


def measure(fn, synchronize, repetitions):
    times, blocking = [], []
    for _ in range(repetitions):
        synchronize()
        start = time.perf_counter_ns()
        result = fn()
        dispatched = time.perf_counter_ns()
        synchronize()
        end = time.perf_counter_ns()
        times.append((dispatched - start) / 1000)
        blocking.append((end - start) / 1000)
    return dict(
        dispatch_median_us=float(np.median(times)),
        blocking_median_us=float(np.median(blocking)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--command-buffer", action="store_true")
    options = parser.parse_args()
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
    for name, actual, expected in [("output", output, tout)] + [(f"d{i}", a, b) for i, (a, b) in enumerate(zip(grads[:5], tgrads[:5]))]:
        a, b = np.asarray(actual, dtype=np.float32), expected.float().cpu().numpy()
        error = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-10))
        assert error < 0.04, (name, error)
    jax_functions = dict(
        jax_fwd=lambda: compiled_fwd(*arrays),
        jax_bwd=lambda: compiled_bwd(residual, do),
    )
    torch_functions = dict(
        torch_fwd=lambda: torch_kda(*torch_arrays, plan_name="kda_frost"),
        torch_bwd=lambda: torch_bwd(tdo, *torch_arrays, options.dim**-0.5, plan_name="kda_frost"),
    )
    report = dict(shape=shape, checkpoint=0, batch_invariant=False, command_buffer=options.command_buffer)
    from cuda.bindings import runtime as rt

    rt.cudaProfilerStart()
    for name, fn in jax_functions.items():
        with torch.cuda.nvtx.range(name):
            report[name] = measure(fn, torch.cuda.synchronize, options.repetitions)
    with torch.cuda.stream(stream):
        for name, fn in torch_functions.items():
            with torch.cuda.nvtx.range(name):
                report[name] = measure(fn, torch.cuda.synchronize, options.repetitions)
    rt.cudaProfilerStop()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
