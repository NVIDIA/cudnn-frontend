# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure warm KDA autograd enqueue/E2E and captured backward separately.

Run against an explicitly selected source package. Compilation, input creation,
validation and profiling are excluded from final timings. CUDA graphs contain
four calls to reduce replay-submit overhead; inputs and autograd use one stream.
"""

import argparse
import cProfile
import hashlib
import json
import pstats
import statistics
import time
from pathlib import Path

import torch
import cudnn
from cudnn.linear_attention.ops.kda import bprop_cache, kda_bwd, kimi_delta_attention


def measure(call, repeats, rounds):
    """Return round samples including host enqueue and GPU event elapsed time."""
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        start = time.perf_counter_ns()
        for _ in range(repeats):
            result = call()
        host_us = (time.perf_counter_ns() - start) / repeats / 1000
        end.record()
        end.synchronize()
        samples.append({"host_us": host_us, "event_us": begin.elapsed_time(end) * 1000 / repeats})
    return {key: statistics.median(x[key] for x in samples) for key in samples[0]} | {"samples": samples}


def graph_measure(call, repeats, rounds):
    """Capture four complete calls, retaining results until graph reset."""
    graph = torch.cuda.CUDAGraph()
    results = []
    try:
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            for _ in range(4):
                results.append(call())
        measured = measure(graph.replay, repeats, rounds)
        for key in ("host_us", "event_us"):
            measured[key] /= 4
        for sample in measured["samples"]:
            for key in ("host_us", "event_us"):
                sample[key] /= 4
        graph.replay()
        torch.cuda.synchronize()
        fresh = call()
        for actual, expected in zip(results[-1], fresh, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return measured
    finally:
        torch.cuda.synchronize()
        graph.reset()


def run_case(args, batch, tokens, heads, checkpoint):
    """Construct one stable pretraining shape and measure both public paths."""
    generator = torch.Generator().manual_seed(1234)
    dtype = getattr(torch, args.dtype)
    shape = (batch * tokens, heads, args.dim)

    def rand(shape, dtype):
        """Draw reproducible CPU inputs, then upload outside the timed region."""
        return torch.randn(shape, dtype=dtype, generator=generator).cuda()

    inputs = [rand(shape, dtype) for _ in range(3)]
    inputs += [rand(shape, torch.float32) - 4, rand(shape[:2], torch.float32)]
    inputs = tuple(t.detach().requires_grad_(True) for t in inputs)
    q, k, v, g, beta = inputs
    cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * tokens
    do = rand(shape, dtype)
    opts = dict(use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, safe_gate=True, plan_name="kda_frost")

    def forward():
        """Run public KDA forward, retaining the requested state checkpoints."""
        return kimi_delta_attention(*inputs, cu, checkpoint_every_n_tokens=checkpoint, **opts)

    def e2e():
        """Return the forward output and fresh gradients for all five input leaves."""
        out = forward()[0]
        return (out,) + torch.autograd.grad(out, inputs, do)

    saved = forward()
    cp = saved[2].detach() if checkpoint else None
    detached = tuple(t.detach() for t in inputs)

    def backward():
        """Measure complete backward using detached inputs and optional forward checkpoints."""
        return kda_bwd(do, *detached, cu, args.dim**-0.5, state_checkpoints=cp, checkpoint_every_n_tokens=checkpoint, **opts)

    for _ in range(5):
        before = e2e()
        backward()
    torch.cuda.synchronize()
    assert all(torch.isfinite(t).all() for t in before)
    row = {"batch": batch, "tokens": tokens, "heads": heads, "dim": args.dim, "checkpoint": checkpoint}
    row["fingerprints"] = [hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for t in before]
    graph, _ = next(reversed(bprop_cache.values()))
    plan = graph._compiled_plans[graph._plan_index].compiled
    row["schedule"] = "chain" if plan.chain else "uncut"
    row["workspace_bytes"] = graph.get_workspace_size()
    row["eager_e2e"] = measure(e2e, args.repeats, args.rounds)
    row["graph_e2e"] = graph_measure(e2e, args.repeats, args.rounds)
    row["graph_bwd"] = graph_measure(backward, args.repeats, args.rounds)
    row["eager_bwd"] = measure(backward, args.repeats, args.rounds)
    after = e2e()
    for got, want in zip(after, before, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    if args.cpu_profile:
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(20):
            e2e()
        profiler.disable()
        torch.cuda.synchronize()
        pstats.Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(24)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            backward()
            torch.cuda.synchronize()
        row["bwd_kernels"] = [
            {"name": event.name, "us": event.device_time_total} for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA
        ]
    print(json.dumps(row), flush=True)
    return row


def main():
    """Measure the requested shape grid and save timings with import/source provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", nargs="+", type=int, default=[1])
    parser.add_argument("--tokens", nargs="+", type=int, default=[2048])
    parser.add_argument("--heads", nargs="+", type=int, default=[32])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--checkpoint", nargs="+", type=int, default=[16])
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cpu-profile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    print(f"source={cudnn.__file__}", flush=True)
    report = {
        "source": cudnn.__file__,
        "torch": torch.__version__,
        "backend": cudnn.backend_version(),
        "source_sha256": {
            name: hashlib.sha256((Path(cudnn.__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "linear_attention/frost/kernel/kda_bprop_config.py",
                "linear_attention/frost/kernel/kda_bprop_f16.py",
                "linear_attention/frost/kernel/kda_bprop_summary_f16.py",
                "linear_attention/frost/kernel/kda_uncut_backward_f16.py",
                "linear_attention/frost/kda_engine.py",
                "linear_attention/ops/common.py",
            )
        },
        "args": vars(args) | {"json": str(args.json)},
        "cases": [],
    }
    for batch in args.batch:
        for tokens in args.tokens:
            for heads in args.heads:
                for checkpoint in args.checkpoint:
                    report["cases"].append(run_case(args, batch, tokens, heads, checkpoint))
    args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
