# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Picklable multiprocessing workers for distributed MoE EP tests."""

from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist

from moe_ep.moe_ep_test_support import (
    _assert_backward_matches,
    _assert_matches_reference,
    _assert_wgrads_match_reference,
    _fixed_training_reference,
    _fixed_training_weights,
    _forward_config,
    _grad_output,
    _output_as_float,
    _reference_forward,
    make_distributed_forward_inputs,
    quantize_mxfp8,
)

__all__ = [
    "_distributed_backward_reference_worker",
    "_distributed_output_worker",
    "_distributed_subgroup_backward_reference_worker",
    "_distributed_subgroup_output_worker",
    "_run_backward_reference_case",
    "_run_forward_output_case",
]


def _run_forward_output_case(
    *,
    device: torch.device,
    ep_group,
    ep_rank: int,
    ep_size: int,
    combine_format: str = "bf16",
    expected_global_ranks: tuple[int, ...] | None = None,
) -> None:
    """Run inference-forward parity and dropped-route checks."""

    from cudnn import MoeEp

    args = make_distributed_forward_inputs(ep_rank, ep_size, device)
    config = _forward_config(
        num_experts=2 * ep_size,
        ep_group=ep_group,
        max_tokens_per_rank=8,
        combine_format=combine_format,
    )
    expected = _reference_forward(args, **config)
    op = MoeEp(**config)
    try:
        actual = op(*args)
        actual_snapshot = _output_as_float(actual).clone()
        torch.cuda.synchronize(device)

        args[3].fill_(-1)
        dropped = op(*args)
        dropped_snapshot = _output_as_float(dropped).clone()
        torch.cuda.synchronize(device)

        dist.barrier(group=ep_group)
        assertion_error = None
        try:
            assert op.ep_rank == ep_rank
            if expected_global_ranks is not None:
                assert op.ep_global_ranks == expected_global_ranks
            _assert_matches_reference(actual_snapshot, expected)
            assert dropped_snapshot.eq(0).all()
        except BaseException as error:
            assertion_error = error
        dist.barrier(group=ep_group)
        if assertion_error is not None:
            raise assertion_error

        op.close()
        op = None
        dist.barrier(group=ep_group)
    finally:
        if op is not None:
            op.close()


def _distributed_output_worker(
    rank: int,
    world_size: int,
    init_file: str,
    combine_format: str = "bf16",
) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(seconds=180),
    )
    try:
        _run_forward_output_case(
            device=device,
            ep_group=dist.group.WORLD,
            ep_rank=rank,
            ep_size=world_size,
            combine_format=combine_format,
            expected_global_ranks=tuple(range(world_size)),
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _distributed_subgroup_output_worker(
    global_rank: int,
    global_world_size: int,
    init_file: str,
) -> None:
    """Run one of two disjoint, non-contiguous EP2 groups inside WORLD4."""

    device = torch.device("cuda", global_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=global_rank,
        world_size=global_world_size,
        device_id=device,
        timeout=timedelta(seconds=180),
    )
    try:
        subgroup_memberships = ((0, 2), (1, 3))
        subgroups = [
            dist.new_group(list(members), backend="nccl")
            for members in subgroup_memberships
        ]
        subgroup_index = global_rank % 2
        ep_group = subgroups[subgroup_index]
        ep_rank = dist.get_rank(ep_group)
        ep_size = dist.get_world_size(ep_group)
        actual_global_ranks = tuple(
            dist.get_global_rank(ep_group, group_rank)
            for group_rank in range(ep_size)
        )

        _run_forward_output_case(
            device=device,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            expected_global_ranks=subgroup_memberships[subgroup_index],
        )
        dist.barrier()
        assert actual_global_ranks == subgroup_memberships[subgroup_index]
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _make_distributed_backward_inputs(
    ep_rank: int,
    ep_size: int,
    device: torch.device,
):
    """Build a minimal local/remote/drop case with one empty local expert."""

    generator = torch.Generator(device=device).manual_seed(20260828 + ep_rank)
    local_experts, token_count, hidden, intermediate = 2, 2, 128, 256
    activation = (
        torch.randn(
            token_count,
            hidden,
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    fc1_weight = quantize_mxfp8(
        torch.randn(
            local_experts,
            hidden,
            2 * intermediate,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    fc2_weight = quantize_mxfp8(
        torch.randn(
            local_experts,
            intermediate,
            hidden,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    local_expert = ep_rank * local_experts
    remote_expert = ((ep_rank + 1) % ep_size) * local_experts
    topk_idx = torch.tensor(
        [[local_expert, remote_expert], [-1, local_expert]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.tensor(
        [[0.625, 0.375], [0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    grad_output = _grad_output(
        device,
        token_count,
        seed=20260901 + ep_rank,
    )
    return (
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
    ), grad_output


def _run_backward_reference_case(
    *,
    device: torch.device,
    ep_group,
    ep_rank: int,
    ep_size: int,
    combine_format: str = "bf16",
    gate_up_clamp: float | None = None,
    expected_global_ranks: tuple[int, ...] | None = None,
) -> None:
    """Run fixed-resource training after the independent distributed oracle."""

    from cudnn import MoeEp

    args, grad_output = _make_distributed_backward_inputs(
        ep_rank,
        ep_size,
        device,
    )
    num_experts = 2 * ep_size
    max_recv_size_per_rank = 3

    # Finish all collective reference work, including dense local dW, before
    # constructing or launching the production operator.
    expected = _fixed_training_reference(
        args,
        grad_output,
        combine_format=combine_format,
        gate_up_clamp=gate_up_clamp,
        ep_group=ep_group,
        num_experts=num_experts,
        max_recv_size_per_rank=max_recv_size_per_rank,
        drop_on_overflow=True,
    )
    expected_y, expected_dx, expected_dprob, expected_wgrads = expected
    expected_dense_wgrads = expected_wgrads.dense_wgrads()
    weights = _fixed_training_weights(args)

    op = MoeEp(
        num_experts=num_experts,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        ep_group=ep_group,
        max_tokens_per_rank=args[0].shape[0],
        max_recv_size_per_rank=max_recv_size_per_rank,
        drop_on_overflow=True,
        combine_format=combine_format,
        gate_up_clamp=gate_up_clamp,
    )
    try:
        resources = op.prepare_training_resources(
            weights,
            slot_count=1,
            lane_count=1,
        )
        slot = resources.slots[0]
        lane = resources.lanes[0]
        resources.refresh_weights()
        actual_y = resources.forward(
            slot,
            lane,
            args[0],
            args[3],
            args[4],
        )
        actual_dx, actual_dprob, actual_wgrads = resources.backward(
            slot,
            lane,
            grad_output,
        )
        overflow = resources.finalize_overflow((slot,), lane)
        torch.cuda.synchronize(device)

        # No rank may enter a local assertion while a peer is still inside a
        # collective kernel. A second barrier keeps cleanup aligned on failure.
        dist.barrier(group=ep_group)
        assertion_error = None
        try:
            assert op.ep_rank == ep_rank
            assert op.ep_size == ep_size
            if expected_global_ranks is not None:
                assert op.ep_global_ranks == expected_global_ranks
            assert overflow.eq(0).all()
            assert args[3][0, 0] // 2 == ep_rank
            assert args[3][0, 1] // 2 == (ep_rank + 1) % ep_size
            assert args[3].eq(-1).any()
            assert expected_wgrads.valid_route_counts[1].eq(0)
            assert actual_wgrads.valid_route_counts[1].eq(0)
            _assert_matches_reference(actual_y, expected_y)
            _assert_backward_matches(
                (actual_dx, actual_dprob),
                (expected_dx, expected_dprob),
                args[3],
            )
            _assert_wgrads_match_reference(
                actual_wgrads,
                expected_wgrads,
                expected_dense=expected_dense_wgrads,
            )
        except BaseException as error:
            assertion_error = error
        dist.barrier(group=ep_group)
        if assertion_error is not None:
            raise assertion_error
    finally:
        op.close()


def _distributed_subgroup_backward_reference_worker(
    global_rank: int,
    global_world_size: int,
    init_file: str,
) -> None:
    """Run fixed-resource backward in two non-contiguous EP2 groups."""

    device = torch.device("cuda", global_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=global_rank,
        world_size=global_world_size,
        device_id=device,
        timeout=timedelta(minutes=10),
    )
    ep_group = None
    try:
        subgroup_memberships = ((0, 2), (1, 3))
        # Every WORLD rank must create every subgroup in the same order.
        subgroups = [
            dist.new_group(
                list(members),
                backend="nccl",
                timeout=timedelta(minutes=10),
            )
            for members in subgroup_memberships
        ]
        subgroup_index = global_rank % 2
        expected_global_ranks = subgroup_memberships[subgroup_index]
        ep_group = subgroups[subgroup_index]
        ep_rank = dist.get_rank(ep_group)
        ep_size = dist.get_world_size(ep_group)
        actual_global_ranks = tuple(
            dist.get_global_rank(ep_group, group_rank)
            for group_rank in range(ep_size)
        )
        assert ep_size == len(expected_global_ranks)
        assert ep_rank == expected_global_ranks.index(global_rank)
        assert actual_global_ranks == expected_global_ranks

        _run_backward_reference_case(
            device=device,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            combine_format="bf16",
            expected_global_ranks=expected_global_ranks,
        )
    finally:
        if dist.is_initialized():
            try:
                # Keep both independent groups alive until all work is done,
                # then collectively finalize the process-local runtime.
                dist.barrier()
                from cudnn.moe_ep._megamoe_backend._runtime import (
                    get_runtime_manager,
                )

                get_runtime_manager().shutdown()
                dist.barrier()
            finally:
                if ep_group is not None:
                    dist.destroy_process_group(ep_group)
                dist.destroy_process_group()


def _distributed_backward_reference_worker(
    rank: int,
    world_size: int,
    init_file: str,
    combine_format: str,
    gate_up_clamp: float | None = None,
) -> None:
    """Initialize one local rank and run distributed training parity."""

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(minutes=10),
    )
    try:
        _run_backward_reference_case(
            device=device,
            ep_group=dist.group.WORLD,
            ep_rank=rank,
            ep_size=world_size,
            combine_format=combine_format,
            gate_up_clamp=gate_up_clamp,
        )
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
                from cudnn.moe_ep._megamoe_backend._runtime import (
                    get_runtime_manager,
                )

                get_runtime_manager().shutdown()
                dist.barrier()
            finally:
                dist.destroy_process_group()
