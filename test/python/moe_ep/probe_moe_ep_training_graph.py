#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stateless SM107 multi-rank CUDA Graph training probe."""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from cudnn import MoeEp
from moe_ep.moe_ep_test_support import (
    _allocate_stateless_training_outputs,
    _allocate_training_weight_staging,
    _fixed_training_weights,
    _grad_output,
    make_distributed_forward_inputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-replays", type=int, default=2)
    parser.add_argument("--burst-replays", type=int, default=100)
    parser.add_argument("--multistream-replays", type=int, default=10)
    parser.add_argument("--max-recv-size-per-rank", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--skip-multistream", action="store_true")
    parser.add_argument("--expect-overflow-assert", action="store_true")
    return parser.parse_args()


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _capture_training_graph(
    op: MoeEp,
    lane,
    args,
    grad_output,
    native_forward,
    native_backward,
    forward_out,
    backward_out,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.training_forward(
            lane,
            args[0],
            args[3],
            args[4],
            weights=native_forward,
            out=forward_out,
        )
        op.training_backward(
            lane,
            grad_output,
            args[3],
            args[4],
            weights=native_backward,
            fc1_preact=forward_out.fc1_preact,
            fc1_a=forward_out.fc1_a,
            fc1_sfa=forward_out.fc1_sfa,
            valid_route_counts=forward_out.valid_route_counts,
            expert_offsets=forward_out.expert_offsets,
            out=backward_out,
        )
    return graph


def _prepare_case(
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    lane_count: int,
    max_recv_size_per_rank: int,
    drop_on_overflow: bool,
):
    args = make_distributed_forward_inputs(rank, world_size, device)
    args = (*args[:4], args[4].float().contiguous())
    grad_output = _grad_output(device, args[0].shape[0], seed=7000 + rank)
    source_weights = _fixed_training_weights(args)
    op = MoeEp(
        num_experts=2 * world_size,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        ep_group=dist.group.WORLD,
        # This is a collective ABI capacity, not the rank-local token count.
        # make_distributed_forward_inputs intentionally varies local shapes.
        max_tokens_per_rank=8,
        max_recv_size_per_rank=max_recv_size_per_rank,
        drop_on_overflow=drop_on_overflow,
        combine_format="bf16",
        weight_interleave_size=32,
    )
    requirements = op.prepare_training(lane_count=lane_count, device=device)
    forward_staging, backward_staging = _allocate_training_weight_staging(source_weights)
    native_forward = op.pack_forward_weights(
        source_weights[0],
        out=forward_staging,
    )
    native_backward = op.pack_backward_weights(
        source_weights[1],
        out=backward_staging,
    )
    output_pairs = tuple(_allocate_stateless_training_outputs(requirements, device) for _ in range(lane_count))
    return (
        op,
        args,
        grad_output,
        native_forward,
        native_backward,
        output_pairs,
    )


def _run_cycle(args: argparse.Namespace, *, device: torch.device, rank: int, world_size: int) -> None:
    lane_count = 1 if args.skip_multistream else 2
    case = _prepare_case(
        device=device,
        rank=rank,
        world_size=world_size,
        lane_count=lane_count,
        max_recv_size_per_rank=args.max_recv_size_per_rank,
        drop_on_overflow=not args.expect_overflow_assert,
    )
    op, inputs, grad_output, native_forward, native_backward, output_pairs = case
    try:
        # Warm each lane and every kernel specialization before capture.
        for lane, (forward_out, backward_out) in zip(
            op.training_lanes,
            output_pairs,
        ):
            op.training_forward(
                lane,
                inputs[0],
                inputs[3],
                inputs[4],
                weights=native_forward,
                out=forward_out,
            )
            op.training_backward(
                lane,
                grad_output,
                inputs[3],
                inputs[4],
                weights=native_backward,
                fc1_preact=forward_out.fc1_preact,
                fc1_a=forward_out.fc1_a,
                fc1_sfa=forward_out.fc1_sfa,
                valid_route_counts=forward_out.valid_route_counts,
                expert_offsets=forward_out.expert_offsets,
                out=backward_out,
            )
        torch.cuda.synchronize(device)
        dist.barrier(group=dist.group.WORLD, device_ids=[device.index])

        graphs = tuple(
            _capture_training_graph(
                op,
                lane,
                inputs,
                grad_output,
                native_forward,
                native_backward,
                forward_out,
                backward_out,
            )
            for lane, (forward_out, backward_out) in zip(
                op.training_lanes,
                output_pairs,
            )
        )
        dist.barrier(group=dist.group.WORLD, device_ids=[device.index])

        replay_count = args.diagnostic_replays + args.burst_replays
        if lane_count > 1:
            replay_count += args.multistream_replays
        caught = None
        try:
            for replay in range(replay_count):
                graphs[replay % len(graphs)].replay()
            torch.cuda.synchronize(device)
        except Exception as error:
            caught = error

        if args.expect_overflow_assert:
            if caught is None:
                raise AssertionError("expected the captured overflow assertion")
        elif caught is not None:
            raise caught
        dist.barrier(group=dist.group.WORLD, device_ids=[device.index])
    finally:
        op.close()


def main() -> None:
    args = _parse_args()
    for name in (
        "diagnostic_replays",
        "burst_replays",
        "multistream_replays",
        "max_recv_size_per_rank",
        "cycles",
        "timeout_seconds",
    ):
        _positive(name, getattr(args, name))

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.timeout_seconds),
        device_id=device,
    )
    try:
        if torch.cuda.get_device_capability(device) != (10, 7):
            raise RuntimeError("stateless training graph probe requires SM107")
        for _ in range(args.cycles):
            _run_cycle(args, device=device, rank=rank, world_size=world_size)
        if rank == 0:
            print(
                "stateless MoeEP training graph probe passed: " f"world_size={world_size}, cycles={args.cycles}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
