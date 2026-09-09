#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stateless SM107 multi-rank CUDA Graph training probe."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist

from cudnn import BlockScaledTensor, MoeEp
from cudnn.moe_ep._megamoe_backend._runtime import _runtime_debug
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
    parser.add_argument("--max-recv-size-per-rank", type=int, default=128)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--skip-multistream", action="store_true")
    parser.add_argument("--expect-overflow-assert", action="store_true")
    return parser.parse_args()


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _token_prefix(value, token_count: int):
    if not isinstance(value, BlockScaledTensor):
        return value[:token_count]
    return BlockScaledTensor(
        data=value.data[:token_count],
        scale=value.scale[:token_count],
        format=value.format,
        logical_shape=(token_count, *value.logical_shape[1:]),
        axis=value.axis,
    )


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
    _runtime_debug("probe.capture.begin", lane=lane.index)
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
    _runtime_debug("probe.capture.end", lane=lane.index)
    return graph


def _capture_forward_graph(
    op: MoeEp,
    lane,
    args,
    native_forward,
    forward_out,
) -> torch.cuda.CUDAGraph:
    _runtime_debug("probe.error.capture.begin", lane=lane.index)
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
    _runtime_debug("probe.error.capture.end", lane=lane.index)
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
    _runtime_debug(
        "probe.prepare.inputs.begin",
        lane_count=lane_count,
        physical_capacity=max_recv_size_per_rank,
        drop_on_overflow=drop_on_overflow,
    )
    args = make_distributed_forward_inputs(rank, world_size, device)
    args = (*args[:4], args[4].float().contiguous())
    grad_output = _grad_output(device, args[0].shape[0], seed=7000 + rank)
    source_weights = _fixed_training_weights(args)
    _runtime_debug(
        "probe.prepare.inputs.end",
        token_count=args[0].shape[0],
        route_count=args[3].numel(),
    )
    _runtime_debug("probe.operator.construct.begin")
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
    _runtime_debug("probe.operator.construct.end")
    _runtime_debug("probe.prepare_training.begin")
    requirements = op.prepare_training(lane_count=lane_count, device=device)
    _runtime_debug("probe.prepare_training.end")
    _runtime_debug("probe.pack_weights.begin")
    forward_staging, backward_staging = _allocate_training_weight_staging(source_weights)
    native_forward = op.pack_forward_weights(
        source_weights[0],
        out=forward_staging,
    )
    native_backward = op.pack_backward_weights(
        source_weights[1],
        out=backward_staging,
    )
    _runtime_debug("probe.pack_weights.end")
    _runtime_debug("probe.allocate_outputs.begin")
    output_pairs = tuple(
        _allocate_stateless_training_outputs(
            requirements,
            device,
            op.training_symmetric_buffers(lane),
        )
        for lane in op.training_lanes
    )
    _runtime_debug("probe.allocate_outputs.end")
    return (
        op,
        args,
        grad_output,
        native_forward,
        native_backward,
        output_pairs,
    )


def _require_all_ranks_to_overflow(op: MoeEp, topk_idx: torch.Tensor, world_size: int) -> None:
    state = op._training_state
    if state is None:
        raise RuntimeError("training state is not prepared")
    config = state.forward_prepared.config
    experts_per_rank = int(config.num_experts)
    invalid = (topk_idx < 0) | (topk_idx >= experts_per_rank * world_size)
    if bool(invalid.any().item()):
        raise ValueError("error-mode probe requires valid dense routing IDs")
    destinations = torch.div(
        topk_idx.reshape(-1).to(torch.int64),
        experts_per_rank,
        rounding_mode="floor",
    )
    receive_counts = torch.bincount(destinations, minlength=world_size)
    dist.all_reduce(receive_counts, op=dist.ReduceOp.SUM)
    logical_limit = int(config.max_recv_size_per_rank)
    counts = tuple(int(count) for count in receive_counts.cpu().tolist())
    _runtime_debug(
        "probe.error.route-preflight",
        logical_limit=logical_limit,
        receive_counts=counts,
    )
    if any(count <= logical_limit for count in counts):
        raise ValueError(
            "error-mode routing must overflow on every rank to avoid a "
            "partial-rank device assertion: "
            f"receive_counts={counts}, logical_limit={logical_limit}"
        )


def _run_error_mode_assert_probe(
    args: argparse.Namespace,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    case = _prepare_case(
        device=device,
        rank=rank,
        world_size=world_size,
        lane_count=1,
        max_recv_size_per_rank=args.max_recv_size_per_rank,
        drop_on_overflow=False,
    )
    op, inputs, _, native_forward, _, output_pairs = case
    lane = op.training_lanes[0]
    forward_out = output_pairs[0][0]

    try:
        _require_all_ranks_to_overflow(op, inputs[3], world_size)
        _runtime_debug("probe.error.warm.forward.begin", lane=lane.index)
        op.training_forward(
            lane,
            _token_prefix(inputs[0], 0),
            inputs[3][:0],
            inputs[4][:0],
            weights=native_forward,
            out=forward_out,
        )
        _runtime_debug("probe.error.warm.forward.end", lane=lane.index)
        _runtime_debug("probe.error.warm.synchronize.begin")
        torch.cuda.synchronize(device)
        _runtime_debug("probe.error.warm.synchronize.end")
        dist.barrier()
    except BaseException:
        op.close()
        raise

    try:
        graph = _capture_forward_graph(
            op,
            lane,
            inputs,
            native_forward,
            forward_out,
        )
        dist.barrier()
    except BaseException as error:
        _runtime_debug(
            "probe.error.capture.error",
            error_type=type(error).__name__,
            error=repr(error),
        )
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(2)

    try:
        _runtime_debug("probe.error.replay.begin")
        graph.replay()
        torch.cuda.synchronize(device)
        _runtime_debug("probe.error.replay.end")
    except BaseException as error:
        error_text = str(error)
        if "device-side assert triggered" in error_text or "Rubin MegaMoE receive route-pool overflow" in error_text:
            print(
                f"MOE_EP_EP{world_size}_ERROR_MODE_OVERFLOW_PASS " f"rank={rank} error={type(error).__name__}",
                flush=True,
            )
            os._exit(0)
        print(
            f"MOE_EP_EP{world_size}_ERROR_MODE_UNEXPECTED_FAILURE " f"rank={rank} error={error!r}",
            file=sys.stderr,
            flush=True,
        )
        os._exit(2)

    print(
        f"MOE_EP_EP{world_size}_ERROR_MODE_OVERFLOW_MISSING rank={rank}",
        file=sys.stderr,
        flush=True,
    )
    os._exit(1)


def _run_cycle(args: argparse.Namespace, *, device: torch.device, rank: int, world_size: int, cycle: int) -> None:
    lane_count = 1 if args.skip_multistream else 2
    _runtime_debug("probe.cycle.begin", cycle=cycle, lane_count=lane_count)
    case = _prepare_case(
        device=device,
        rank=rank,
        world_size=world_size,
        lane_count=lane_count,
        max_recv_size_per_rank=args.max_recv_size_per_rank,
        drop_on_overflow=True,
    )
    op, inputs, grad_output, native_forward, native_backward, output_pairs = case
    try:
        # Warm each lane and every kernel specialization before capture.
        for lane, (forward_out, backward_out) in zip(
            op.training_lanes,
            output_pairs,
        ):
            _runtime_debug("probe.warm.forward.begin", cycle=cycle, lane=lane.index)
            op.training_forward(
                lane,
                inputs[0],
                inputs[3],
                inputs[4],
                weights=native_forward,
                out=forward_out,
            )
            _runtime_debug("probe.warm.forward.end", cycle=cycle, lane=lane.index)
            _runtime_debug("probe.warm.backward.begin", cycle=cycle, lane=lane.index)
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
            _runtime_debug("probe.warm.backward.end", cycle=cycle, lane=lane.index)
        _runtime_debug("probe.warm.synchronize.begin", cycle=cycle)
        torch.cuda.synchronize(device)
        _runtime_debug("probe.warm.synchronize.end", cycle=cycle)

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

        try:
            replay_count = args.diagnostic_replays + args.burst_replays
            _runtime_debug(
                "probe.replay.enqueue.begin",
                cycle=cycle,
                replay_count=replay_count,
            )
            for replay in range(replay_count):
                graph_index = replay % len(graphs)
                graphs[graph_index].replay()
            _runtime_debug("probe.replay.enqueue.end", cycle=cycle)
            if lane_count > 1:
                _runtime_debug(
                    "probe.multistream.begin",
                    cycle=cycle,
                    replay_count=args.multistream_replays,
                )
                current_stream = torch.cuda.current_stream(device)
                streams = tuple(torch.cuda.Stream(device=device) for _ in range(lane_count))
                for stream in streams:
                    stream.wait_stream(current_stream)
                for _ in range(args.multistream_replays):
                    for stream, graph in zip(streams, graphs):
                        with torch.cuda.stream(stream):
                            graph.replay()
                for stream in streams:
                    stream.synchronize()
                _runtime_debug("probe.multistream.end", cycle=cycle)
            _runtime_debug("probe.replay.synchronize.begin", cycle=cycle)
            torch.cuda.synchronize(device)
            _runtime_debug("probe.replay.synchronize.end", cycle=cycle)
        except Exception as error:
            _runtime_debug(
                "probe.replay.error",
                cycle=cycle,
                error_type=type(error).__name__,
                error=repr(error),
            )
            raise
    finally:
        _runtime_debug("probe.close.begin", cycle=cycle)
        op.close()
        _runtime_debug("probe.close.end", cycle=cycle)
    _runtime_debug("probe.cycle.end", cycle=cycle)


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
        capability = torch.cuda.get_device_capability(device)
        if capability != (10, 7):
            raise RuntimeError("stateless training graph probe requires SM107")
        _runtime_debug(
            "probe.main.ready",
            world_size=world_size,
            physical_capacity=args.max_recv_size_per_rank,
            cycles=args.cycles,
        )
        if args.expect_overflow_assert:
            _run_error_mode_assert_probe(
                args,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            raise AssertionError("fatal overflow probe returned")
        for cycle in range(args.cycles):
            _run_cycle(
                args,
                device=device,
                rank=rank,
                world_size=world_size,
                cycle=cycle,
            )
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
