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
import math
import pstats
import statistics
import time
from pathlib import Path

import torch
import cudnn
from cudnn.linear_attention.ops.kda import bprop_cache, fprop_cache, kda_bwd, kimi_delta_attention

RAGGED_SUITE = (
    ("balanced_8x4k", [4096] * 8),
    ("two_tier_32k", [8192, 8192, 4096, 4096, 2048, 2048, 2048, 2048]),
    ("skewed_4x_32k", [16384, 4096, 4096, 2048, 2048, 1024, 1024, 2048]),
    ("skewed_7x_32k", [28672, 2048, 1024, 512, 256, 128, 64, 64]),
    ("empty_padded_32k", [32768, 0, 0, 0, 0, 0, 0, 0]),
    ("skewed_7x_8k", [7168, 512, 256, 64, 32, 16, 16, 128]),
    ("skewed_7x_64k", [57344, 4096, 2048, 1024, 512, 256, 128, 128]),
    ("many_short_32k", [24576, *([1024] * 4), *([512] * 4), *([256] * 8)]),
)


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
    measured = {key: statistics.median(x[key] for x in samples) for key in samples[0]} | {"samples": samples}
    for key in ("host_us", "event_us"):
        if not math.isfinite(measured[key]) or measured[key] <= 0:
            raise ValueError(f"invalid {key} median: {measured[key]}")
    return measured


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


def run_case(args, case_name, seq_lengths, heads, checkpoint):
    """Construct one stable packed pretraining shape and measure both public paths."""
    # Different cu_seqlens with the same declared tensors share a production
    # graph-cache key.  Isolate benchmark cases so plan introspection below
    # cannot accidentally report the previously inserted shape-equivalent plan.
    fprop_cache.clear()
    bprop_cache.clear()
    generator = torch.Generator().manual_seed(1234)
    dtype = getattr(torch, args.dtype)
    batch = len(seq_lengths)
    total_tokens = sum(seq_lengths)
    shape = (total_tokens, heads, args.dim)

    def rand(shape, dtype):
        """Draw reproducible CPU inputs, then upload outside the timed region."""
        return torch.randn(shape, dtype=dtype, generator=generator).cuda()

    inputs = [rand(shape, dtype) for _ in range(3)]
    inputs += [rand(shape, torch.float32) - 4, rand(shape[:2], torch.float32)]
    inputs = tuple(t.detach().requires_grad_(True) for t in inputs)
    q, k, v, g, beta = inputs
    cu = torch.tensor([0, *torch.tensor(seq_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    do = rand(shape, dtype)
    opts = dict(
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        batch_invariant=args.batch_invariant,
        plan_name="kda_frost",
    )

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
    row = {
        "case": case_name,
        "batch": batch,
        "seq_lengths": seq_lengths,
        "total_tokens": total_tokens,
        "max_over_mean": max(seq_lengths, default=0) / max(total_tokens / batch, 1),
        "heads": heads,
        "dim": args.dim,
        "checkpoint": checkpoint,
    }
    row["fingerprints"] = [hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for t in before]
    fwd_graph, _ = next(reversed(fprop_cache.values()))
    bwd_graph, _ = next(reversed(bprop_cache.values()))
    fwd_plan = fwd_graph._compiled_plans[fwd_graph._plan_index].compiled
    bwd_plan = bwd_graph._compiled_plans[bwd_graph._plan_index].compiled
    row["forward_schedule"] = "chain" if fwd_plan.chain else "uncut"
    row["backward_schedule"] = "chain" if bwd_plan.chain else "uncut"
    row["forward_pieces"] = int(fwd_plan.pieces)
    row["backward_pieces"] = int(bwd_plan.pieces)
    row["forward_workspace_bytes"] = fwd_graph.get_workspace_size()
    row["backward_workspace_bytes"] = bwd_graph.get_workspace_size()
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
    print(json.dumps(row, allow_nan=False), flush=True)
    return row


def main():
    """Measure the requested shape grid and save timings with import/source provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", nargs="+", type=int, default=[1])
    parser.add_argument("--tokens", nargs="+", type=int, default=[2048])
    parser.add_argument(
        "--seq-lengths",
        action="append",
        default=[],
        metavar="L0,L1,...",
        help="Packed sequence lengths for one case. Repeat for a case matrix; zero-length sequences are allowed.",
    )
    parser.add_argument("--ragged-suite", action="store_true", help="Run the built-in balanced, skewed, empty-padded, and many-short controls.")
    parser.add_argument("--heads", nargs="+", type=int, default=[32])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--checkpoint", nargs="+", type=int, default=[16])
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cpu-profile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--batch-invariant", action="store_true", help="Use the fixed per-sequence length rule for piece scheduling.")
    parser.add_argument(
        "--force-pieces",
        type=int,
        choices=range(17),
        help="Benchmark-only scheduler override: 0 is uncut, 1..16 force that piece capacity, omitted keeps auto.",
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    seq_length_cases = list(RAGGED_SUITE) if args.ragged_suite else []
    for index, spec in enumerate(args.seq_lengths):
        lengths = [int(value) for value in spec.split(",")]
        if not lengths or any(length < 0 for length in lengths) or sum(lengths) == 0:
            raise ValueError("--seq-lengths needs non-negative lengths with at least one token")
        seq_length_cases.append((f"cli_{index}", lengths))
    if not seq_length_cases:
        seq_length_cases = [(f"uniform_b{batch}_t{tokens}", [tokens] * batch) for batch in args.batch for tokens in args.tokens]

    if args.force_pieces is not None:
        from cudnn.linear_attention.frost.common import piece_chain

        auto_choose_pieces = piece_chain.choose_pieces

        def choose_pieces_override(**kwargs):
            _, unit_chunks = auto_choose_pieces(**kwargs)
            return args.force_pieces, unit_chunks

        def choose_kda_pieces_override(**kwargs):
            kwargs.pop("enable_ragged_fallback")
            _, unit_chunks = auto_choose_pieces(**kwargs)
            return args.force_pieces, unit_chunks, bool(args.force_pieces and args.batch_invariant)

        piece_chain.choose_pieces = choose_pieces_override
        piece_chain.choose_kda_pieces = choose_kda_pieces_override
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
    for case_name, seq_lengths in seq_length_cases:
        for heads in args.heads:
            for checkpoint in args.checkpoint:
                report["cases"].append(run_case(args, case_name, seq_lengths, heads, checkpoint))
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
