#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless SM107 multi-rank CUDA Graph training probe."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist

from cudnn import (
    BlockScaledTensor,
    MoeEp,
    MoeEpFc1WeightLayout,
    MoeEpTuningConfig,
)
from cudnn.moe_ep._megamoe_backend._runtime import _runtime_debug
from moe_ep.moe_ep_test_support import (
    _allocate_stateless_training_outputs,
    _allocate_training_weight_staging,
    _grad_output,
    _moe_ep_config,
    _training_graph_pattern,
    make_training_graph_pattern_inputs,
    make_training_graph_pattern_weights,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        choices=("smoke", "ds3_ep4_pattern"),
        default="smoke",
    )
    parser.add_argument(
        "--dgrad-optimization",
        choices=("baseline", "rolling", "ds3_ep4_pattern"),
        default="baseline",
    )
    parser.add_argument("--diagnostic-replays", type=int, default=2)
    parser.add_argument("--burst-replays", type=int, default=100)
    parser.add_argument("--physical-recv-pool-rows", type=int)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--expect-overflow-assert", action="store_true")
    parser.add_argument("--asymmetric-overflow", action="store_true")
    return parser.parse_args()


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _resolve_pattern_and_capacity(
    args: argparse.Namespace,
    world_size: int,
):
    pattern = _training_graph_pattern(args.pattern, world_size)
    if args.dgrad_optimization == "ds3_ep4_pattern" and pattern.name != "ds3_ep4_pattern":
        raise ValueError("dgrad_optimization='ds3_ep4_pattern' requires " "--pattern ds3_ep4_pattern")
    if pattern.name == "ds3_ep4_pattern" and args.expect_overflow_assert:
        raise ValueError("ds3_ep4_pattern uses the exact full route capacity and does not " "support the overflow-assert probe")
    physical_capacity = pattern.physical_recv_pool_size if args.physical_recv_pool_rows is None else args.physical_recv_pool_rows
    _positive("physical_recv_pool_rows", physical_capacity)
    if pattern.name == "ds3_ep4_pattern" and physical_capacity < pattern.physical_recv_pool_size:
        raise ValueError("ds3_ep4_pattern requires physical_recv_pool_rows >= " f"{pattern.physical_recv_pool_size}, got {physical_capacity}")
    return pattern, physical_capacity


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


def _repeat_token_rows(value, token_count: int):
    source_tokens = int(value.logical_shape[0] if isinstance(value, BlockScaledTensor) else value.shape[0])
    repeats = (token_count + source_tokens - 1) // source_tokens
    if not isinstance(value, BlockScaledTensor):
        return value.repeat((repeats, *([1] * (value.ndim - 1))))[:token_count].contiguous()
    return BlockScaledTensor(
        data=value.data.repeat((repeats, *([1] * (value.data.ndim - 1))))[:token_count].contiguous(),
        scale=value.scale.repeat((repeats, *([1] * (value.scale.ndim - 1))))[:token_count].contiguous(),
        format=value.format,
        logical_shape=(token_count, *value.logical_shape[1:]),
        axis=value.axis,
    )


def _capture_training_graph(
    op: MoeEp,
    args,
    grad_output,
    native_forward,
    native_backward,
    forward_out,
    backward_out,
    stream: torch.cuda.Stream,
) -> torch.cuda.CUDAGraph:
    if torch.cuda.current_stream() != stream:
        raise RuntimeError("training graph capture must remain on the execution stream")
    _runtime_debug("probe.capture.begin")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.training_forward(
            args[0],
            args[3],
            args[4],
            weights=native_forward,
            out=forward_out,
        )
        op.training_backward(
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
    _runtime_debug("probe.capture.end")
    return graph


def _capture_forward_graph(
    op: MoeEp,
    args,
    native_forward,
    forward_out,
) -> torch.cuda.CUDAGraph:
    _runtime_debug("probe.error.capture.begin")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.training_forward(
            args[0],
            args[3],
            args[4],
            weights=native_forward,
            out=forward_out,
        )
    _runtime_debug("probe.error.capture.end")
    return graph


def _prepare_case(
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    pattern,
    dgrad_optimization: str,
    physical_recv_pool_rows: int,
    drop_on_overflow: bool,
    asymmetric_overflow: bool = False,
):
    _runtime_debug(
        "probe.prepare.inputs.begin",
        physical_capacity=physical_recv_pool_rows,
        drop_on_overflow=drop_on_overflow,
        pattern=pattern.name,
        dgrad_optimization=dgrad_optimization,
    )
    args = make_training_graph_pattern_inputs(
        pattern,
        rank,
        world_size,
        device,
    )
    if asymmetric_overflow:
        if pattern.name != "smoke" or world_size < 2:
            raise ValueError("asymmetric overflow requires the smoke pattern with EP2+")
        token_count = pattern.max_tokens_per_rank
        activation = _repeat_token_rows(args[0], token_count)
        topk_idx = torch.zeros(
            (token_count, pattern.top_k),
            dtype=args[3].dtype,
            device=device,
        )
        if rank == 0:
            topk_idx.reshape(-1)[0] = pattern.num_experts // world_size
        topk_weights = args[4][:1].expand(token_count, -1).contiguous()
        args = (
            activation,
            args[1],
            args[2],
            topk_idx,
            topk_weights,
        )
    grad_output = _grad_output(
        device,
        args[0].shape[0],
        seed=7000 + rank,
        hidden_size=pattern.hidden_size,
    )
    source_weights = make_training_graph_pattern_weights(pattern, args)
    _runtime_debug(
        "probe.prepare.inputs.end",
        token_count=args[0].shape[0],
        route_count=args[3].numel(),
    )
    _runtime_debug("probe.operator.construct.begin")
    op = MoeEp(
        _moe_ep_config(
            num_experts=pattern.num_experts,
            hidden_size=pattern.hidden_size,
            intermediate_size=pattern.intermediate_size,
            top_k=pattern.top_k,
            ep_group=dist.group.WORLD,
            # This is a collective ABI capacity, not the rank-local token count.
            # Pattern input factories may intentionally vary local shapes.
            max_tokens_per_rank=pattern.max_tokens_per_rank,
            physical_recv_pool_rows=physical_recv_pool_rows,
            drop_on_overflow=drop_on_overflow,
            combine_format=pattern.combine_format,
            fc1_weight_layout=(MoeEpFc1WeightLayout.GATE_UP_INTERLEAVED_32),
            training_backward_tuning=MoeEpTuningConfig(
                dgrad_optimization=dgrad_optimization,
            ),
        )
    )
    _runtime_debug("probe.operator.construct.end")
    _runtime_debug("probe.prepare_training.begin")
    requirements = op.prepare_training(device=device)
    state = op._training_state
    if state is None:
        raise RuntimeError("training graph probe did not prepare training state")
    backward = state.backward_prepared
    if backward.config.dgrad_optimization != dgrad_optimization:
        raise RuntimeError("training graph probe prepared the wrong dgrad profile: " f"{backward.config.dgrad_optimization!r} != " f"{dgrad_optimization!r}")
    if backward.pool_token_capacity != physical_recv_pool_rows:
        raise RuntimeError("training graph probe prepared the wrong physical receive pool: " f"{backward.pool_token_capacity} != {physical_recv_pool_rows}")
    upstream_profile = backward.kernel.resolved_dgrad_config["dgrad_optimization_profile"]
    expected_upstream_profile = {
        "baseline": "explicit",
        "rolling": "optimized",
        "ds3_ep4_pattern": "ds3_ep4_v1",
    }[dgrad_optimization]
    if upstream_profile != expected_upstream_profile:
        raise RuntimeError("training graph probe did not select the requested upstream " f"profile: {upstream_profile!r} != " f"{expected_upstream_profile!r}")
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
    output_pair = _allocate_stateless_training_outputs(
        requirements,
        device,
        op.training_symmetric_buffers(),
    )
    _runtime_debug("probe.allocate_outputs.end")
    return (
        op,
        args,
        grad_output,
        native_forward,
        native_backward,
        output_pair,
    )


def _require_group_overflow(op: MoeEp, topk_idx: torch.Tensor, world_size: int) -> None:
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
    if not any(count > logical_limit for count in counts):
        raise ValueError("error-mode routing must overflow at least one receiving rank: " f"receive_counts={counts}, logical_limit={logical_limit}")


def _run_error_mode_assert_probe(
    args: argparse.Namespace,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    pattern,
    physical_capacity: int,
) -> None:
    case = _prepare_case(
        device=device,
        rank=rank,
        world_size=world_size,
        pattern=pattern,
        dgrad_optimization=args.dgrad_optimization,
        physical_recv_pool_rows=physical_capacity,
        drop_on_overflow=False,
        asymmetric_overflow=args.asymmetric_overflow,
    )
    op, inputs, _, native_forward, _, output_pair = case
    forward_out = output_pair[0]

    try:
        _require_group_overflow(op, inputs[3], world_size)
        _runtime_debug("probe.error.warm.forward.begin")
        op.training_forward(
            _token_prefix(inputs[0], 0),
            inputs[3][:0],
            inputs[4][:0],
            weights=native_forward,
            out=forward_out,
        )
        _runtime_debug("probe.error.warm.forward.end")
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
        if any(
            expected in error_text
            for expected in (
                "device-side assert triggered",
                "Rubin MegaMoE receive route-pool overflow",
                "unspecified launch failure",
                "cudaErrorLaunchFailure",
                "CUDA_ERROR_LAUNCH_FAILED",
            )
        ):
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


def _run_cycle(
    args: argparse.Namespace,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    pattern,
    physical_capacity: int,
    cycle: int,
) -> None:
    _runtime_debug("probe.cycle.begin", cycle=cycle)
    case = _prepare_case(
        device=device,
        rank=rank,
        world_size=world_size,
        pattern=pattern,
        dgrad_optimization=args.dgrad_optimization,
        physical_recv_pool_rows=physical_capacity,
        drop_on_overflow=True,
        asymmetric_overflow=args.asymmetric_overflow,
    )
    op, inputs, grad_output, native_forward, native_backward, output_pair = case
    forward_out, backward_out = output_pair
    try:
        execution_stream = torch.cuda.current_stream(device)
        _runtime_debug("probe.warm.forward.begin", cycle=cycle)
        op.training_forward(
            inputs[0],
            inputs[3],
            inputs[4],
            weights=native_forward,
            out=forward_out,
        )
        _runtime_debug("probe.warm.forward.end", cycle=cycle)
        _runtime_debug("probe.warm.backward.begin", cycle=cycle)
        op.training_backward(
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
        _runtime_debug("probe.warm.backward.end", cycle=cycle)
        _runtime_debug("probe.warm.synchronize.begin", cycle=cycle)
        execution_stream.synchronize()
        _runtime_debug("probe.warm.synchronize.end", cycle=cycle)

        if args.asymmetric_overflow:
            state = op._training_state
            if state is None:
                raise RuntimeError("training state disappeared after warmup")
            scratch = state.views(token_count=int(inputs[0].shape[0])).scratch
            phase_flags = torch.cat(
                (
                    scratch.forward_overflow,
                    scratch.backward_overflow,
                )
            )
            gathered_flags = [torch.empty_like(phase_flags) for _ in range(world_size)]
            dist.all_gather(gathered_flags, phase_flags)
            if any(tuple(int(value) for value in flags.cpu().tolist()) != (1, 1) for flags in gathered_flags):
                raise AssertionError("asymmetric forward/backward overflow flags were not " f"global: {[flags.cpu().tolist() for flags in gathered_flags]}")

        # Capture two graph executables over the same fixed instance resources.
        # Both capture and replay stay on one stream and execute sequentially.
        graphs = tuple(
            _capture_training_graph(
                op,
                inputs,
                grad_output,
                native_forward,
                native_backward,
                forward_out,
                backward_out,
                execution_stream,
            )
            for _ in range(2)
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
                if torch.cuda.current_stream(device) != execution_stream:
                    raise RuntimeError("training graph replay must remain on its capture stream")
                graphs[graph_index].replay()
            _runtime_debug("probe.replay.enqueue.end", cycle=cycle)
            _runtime_debug("probe.replay.synchronize.begin", cycle=cycle)
            execution_stream.synchronize()
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
        "cycles",
        "timeout_seconds",
    ):
        _positive(name, getattr(args, name))

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    pattern, physical_capacity = _resolve_pattern_and_capacity(
        args,
        world_size,
    )
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
            pattern=pattern.name,
            dgrad_optimization=args.dgrad_optimization,
            physical_capacity=physical_capacity,
            cycles=args.cycles,
        )
        if args.expect_overflow_assert:
            _run_error_mode_assert_probe(
                args,
                device=device,
                rank=rank,
                world_size=world_size,
                pattern=pattern,
                physical_capacity=physical_capacity,
            )
            raise AssertionError("fatal overflow probe returned")
        for cycle in range(args.cycles):
            _run_cycle(
                args,
                device=device,
                rank=rank,
                world_size=world_size,
                pattern=pattern,
                physical_capacity=physical_capacity,
                cycle=cycle,
            )
        if rank == 0:
            print(
                "stateless MoeEP training graph probe passed: "
                f"world_size={world_size}, pattern={pattern.name}, "
                f"dgrad_optimization={args.dgrad_optimization}, "
                f"cycles={args.cycles}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
