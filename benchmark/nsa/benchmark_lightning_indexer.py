# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact MiniMax Lightning Indexer decode against its Torch reference."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from cudnn import NSA

BLOCK_SIZE = 128
TOP_K = 16
HEADS = 4
HEAD_DIM = 128


def reference(q, k, position_ids):
    """Run the dense PyTorch score, block-max, Top-K formulation."""
    batch, _, _, _ = q.shape
    k_len = k.shape[1]
    num_blocks = math.ceil(k_len / BLOCK_SIZE)
    scores = torch.matmul(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float().transpose(-1, -2),
    )
    key_positions = torch.arange(k_len, device=q.device)
    scores.masked_fill_(
        key_positions[None, None, None, :] > position_ids[:, None, :, None],
        float("-inf"),
    )
    if padding := num_blocks * BLOCK_SIZE - k_len:
        scores = torch.nn.functional.pad(scores, (0, padding), value=float("-inf"))
    block_scores = scores.view(batch, HEADS, 1, num_blocks, BLOCK_SIZE).amax(-1)
    current = (position_ids // BLOCK_SIZE)[:, None, :, None].expand(-1, HEADS, -1, -1)
    block_scores.scatter_(-1, current, float("inf"))
    selected = min(TOP_K, num_blocks)
    values, indices = torch.topk(block_scores, selected, dim=-1, sorted=False)
    result = torch.full(
        (batch, HEADS, 1, TOP_K),
        -1,
        dtype=torch.int64,
        device=q.device,
    )
    result[..., :selected] = indices.masked_fill(values == float("-inf"), -1)
    counts = (result >= 0).sum(-1, dtype=torch.int32)
    return result, counts


def canonical_sets(indices):
    """Sort valid IDs while moving -1 padding to the end."""
    sentinel = torch.iinfo(torch.int64).max
    return torch.sort(indices.to(torch.int64).masked_fill(indices < 0, sentinel), dim=-1).values


def measure(call, warmup, iterations):
    """Measure one asynchronous callable with CUDA events."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def benchmark_shape(batch, k_capacity, position, warmup, iterations, rounds):
    """Validate and interleave reference/prepared timings for one shape."""
    generator = torch.Generator(device="cuda").manual_seed(0x4D330000 + batch * 17 + k_capacity + position)
    q = torch.randn(
        (batch, 1, HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        (batch, k_capacity, 1, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    positions = torch.tensor([[position]], dtype=torch.int64, device="cuda").expand(batch, -1)
    block_indices = torch.empty(
        (batch, 1, HEADS, TOP_K),
        dtype=torch.int32,
        device="cuda",
    ).transpose(1, 2)
    block_counts = torch.empty(
        (batch, 1, HEADS),
        dtype=torch.int32,
        device="cuda",
    ).transpose(1, 2)
    plan = NSA.LightningIndexer(q, k, positions, block_indices, block_counts)
    plan.check_support()
    plan.compile()
    workspace = plan.make_workspace()

    def run_prepared_api():
        plan.execute(
            q,
            k,
            positions,
            block_indices,
            block_counts,
            workspace,
        )

    expected_indices, expected_counts = reference(q, k, positions)
    run_prepared_api()
    torch.cuda.synchronize()
    torch.testing.assert_close(block_counts, expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(
        canonical_sets(block_indices),
        canonical_sets(expected_indices),
        rtol=0,
        atol=0,
    )

    reference_us = []
    prepared_api_us = []
    for round_index in range(rounds):
        arms = (
            (("reference", lambda: reference(q, k, positions)), ("prepared_api", run_prepared_api))
            if round_index % 2 == 0
            else (("prepared_api", run_prepared_api), ("reference", lambda: reference(q, k, positions)))
        )
        samples = {}
        for name, call in arms:
            samples[name] = measure(call, warmup, iterations)
        reference_us.append(samples["reference"])
        prepared_api_us.append(samples["prepared_api"])

    graph = torch.cuda.CUDAGraph()
    run_prepared_api()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        run_prepared_api()
    graph_replay_us = [measure(graph.replay, warmup, iterations) for _ in range(rounds)]

    reference_median = statistics.median(reference_us)
    prepared_api_median = statistics.median(prepared_api_us)
    graph_replay_median = statistics.median(graph_replay_us)
    return {
        "batch": batch,
        "k_capacity": k_capacity,
        "position": position,
        "workspace_bytes": plan.workspace_size,
        "reference_us": reference_us,
        "prepared_api_us": prepared_api_us,
        "graph_replay_us": graph_replay_us,
        "reference_median_us": reference_median,
        "prepared_api_median_us": prepared_api_median,
        "graph_replay_median_us": graph_replay_median,
        "speedup_vs_eager_dense_reference": reference_median / prepared_api_median,
        "correct": True,
    }


def source_metadata():
    """Record enough provenance to tie an artifact to an exact worktree."""
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "repository": str(root),
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "command": [sys.executable, *sys.argv],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--k-capacity",
        type=int,
        action="append",
        dest="capacities",
        help="repeatable K capacity; defaults to 2048/8192/16384/32768",
    )
    parser.add_argument(
        "--position",
        type=int,
        help="shared explicit position (default: suffix for each capacity)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    """Run requested shapes and emit machine-readable evidence."""
    args = parse_args()
    capacities = args.capacities or [2048, 8192, 16384, 32768]
    if args.batch < 1:
        raise ValueError("batch must be positive")
    for capacity in capacities:
        if not 1 <= capacity <= 32768:
            raise ValueError(f"k capacity must be in [1, 32768], got {capacity}")
        if args.position is not None and not 0 <= args.position < capacity:
            raise ValueError(f"position must be in [0, {capacity}), got {args.position}")
    properties = torch.cuda.get_device_properties(0)
    rows = [
        benchmark_shape(
            args.batch,
            capacity,
            capacity - 1 if args.position is None else args.position,
            args.warmup,
            args.iterations,
            args.rounds,
        )
        for capacity in capacities
    ]
    result = {
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": properties.name,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
            "sm_count": properties.multi_processor_count,
        },
        "method": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rounds": args.rounds,
            "prepared_api": (
                "warmed execute call with native destinations/workspace; includes "
                "Python validation and stream lifetime tracking; excludes JIT and allocation"
            ),
            "reference": ("eager dense PyTorch FP32 matmul + causal mask + block max + " "topk; includes intermediate allocation"),
            "graph_replay": "native prepared API captured and replayed alone",
        },
        "source": source_metadata(),
        "results": rows,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
