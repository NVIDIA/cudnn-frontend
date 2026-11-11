#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure E2E adaptive Q-pair DSA backward from unmodified top-k rows."""

import argparse
import math
import statistics

import torch

from cudnn import DSA


def make_pair_indices(q_length: int, kv_length: int, topk: int, overlap: float) -> torch.Tensor:
    """Build unique adjacent rows with the requested exact intersection."""
    pairs = q_length // 2
    shared = round(topk * overlap)
    offsets = (torch.arange(pairs, device="cuda") * 7919)[:, None]
    columns = torch.arange(topk, device="cuda")[None, :]
    first = (offsets + columns) % kv_length
    second_unique = (offsets + topk + columns[:, : topk - shared]) % kv_length
    second = torch.cat((first[:, :shared], second_unique), dim=1)
    rows = torch.stack((first, second), dim=1).reshape(2 * pairs, topk)
    if q_length % 2:
        tail = torch.arange(topk, device="cuda")[None, :]
        rows = torch.cat((rows, tail), dim=0)
    return rows.to(torch.int32).contiguous()


def make_inputs(heads: int, q_length: int, kv_length: int, topk: int, overlap: float):
    """Create one common input set for both benchmark modes."""
    dim = 576
    q = torch.randn(q_length, heads, dim, device="cuda", dtype=torch.bfloat16) / 10
    kv = torch.randn(kv_length, dim, device="cuda", dtype=torch.bfloat16) / 10
    out = torch.randn(q_length, heads, 512, device="cuda", dtype=torch.bfloat16) / 10
    dout = torch.randn_like(out) / 10
    lse = torch.full((q_length, heads), math.log(topk), device="cuda", dtype=torch.float32)
    sink = torch.linspace(-2.0, 2.0, heads, device="cuda", dtype=torch.float32)
    indices = make_pair_indices(q_length, kv_length, topk, overlap)
    return q, kv, out, dout, lse, sink, indices


def make_run(inputs, mode: str):
    """Create one wrapper closure with reusable workspace and gradient buffers."""
    q, kv, out, dout, lse, sink, indices = inputs
    plan = DSA.SparseAttentionBackward(q, kv, out, dout, lse, sink, indices, q_cluster_mode=mode)
    if not plan.check_support():
        raise RuntimeError(f"unsupported benchmark case: H{q.shape[1]} K{indices.shape[1]} mode={mode}")
    workspace = torch.empty(plan.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    dq, dkv = torch.empty_like(q), torch.empty_like(kv)

    def run():
        return DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            sink,
            indices,
            dq=dq,
            dkv=dkv,
            workspace=workspace,
            q_cluster_mode=mode,
        )

    return run


def capture(run):
    """Warm and capture one complete backward invocation."""
    result = run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    return graph, result


def elapsed_ms(run, repeat: int) -> float:
    """Return average device elapsed time over one sample."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def benchmark_case(heads: int, q_length: int, kv_length: int, topk: int, overlap: float, repeat: int):
    """Compare graph-captured baseline and adaptive E2E latency."""
    inputs = make_inputs(heads, q_length, kv_length, topk, overlap)
    baseline_run = make_run(inputs, "off")
    adaptive_run = make_run(inputs, "adaptive_pair")
    baseline_graph, baseline = capture(baseline_run)
    adaptive_graph, adaptive = capture(adaptive_run)
    baseline_graph.replay()
    adaptive_graph.replay()
    torch.cuda.synchronize()
    for name in ("dq", "dkv", "d_sink"):
        torch.testing.assert_close(adaptive[name], baseline[name], atol=5e-2, rtol=5e-2, msg=name)

    samples = ([], [])
    runs = (baseline_graph.replay, adaptive_graph.replay)
    for mode in (0, 1, 1, 0, 1, 0, 0, 1):
        samples[mode].append(elapsed_ms(runs[mode], repeat))
    baseline_ms, adaptive_ms = (statistics.median(values) for values in samples)
    return baseline_ms, adaptive_ms


def main() -> None:
    """Run a compact overlap sweep for H16/H32 Dqk576."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", default="16,32")
    parser.add_argument("--topks", default="1024,2048")
    parser.add_argument("--overlaps", default="0,.2,.4,.6,.8,1")
    parser.add_argument("--q-length", type=int, default=2048)
    parser.add_argument("--kv-length", type=int, default=32768)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(1234)
    print("heads,topk,overlap,baseline_ms,adaptive_e2e_ms,speedup", flush=True)
    for heads in map(int, args.heads.split(",")):
        for topk in map(int, args.topks.split(",")):
            for overlap in map(float, args.overlaps.split(",")):
                baseline_ms, adaptive_ms = benchmark_case(heads, args.q_length, args.kv_length, topk, overlap, args.repeat)
                print(f"{heads},{topk},{overlap:.2f},{baseline_ms:.4f},{adaptive_ms:.4f},{baseline_ms / adaptive_ms:.3f}", flush=True)


if __name__ == "__main__":
    main()
