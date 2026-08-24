# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Picklable multiprocessing workers for distributed MoE EP tests."""

from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist

from moe_ep.moe_ep_backward_support import (
    _assert_backward_matches,
    _dense_wgrads_from_operands,
    _expected_backward,
    _grad_output,
    _reference_backward,
)
from moe_ep.moe_ep_forward_support import (
    _assert_matches_reference,
    _forward_config,
    _output_as_float,
    _reference_args,
    _reference_forward,
)
from moe_ep.moe_ep_test_data import make_distributed_forward_inputs

__all__ = [
    "_distributed_backward_worker",
    "_distributed_output_worker",
    "_distributed_subgroup_output_worker",
    "_distributed_wgrad_worker",
    "_run_forward_output_case",
    "_run_wgrad_operand_case",
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
    """Run forward parity and dropped-route checks on one initialized EP group."""

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
        assert op.ep_rank == ep_rank
        if expected_global_ranks is not None:
            assert op.ep_global_ranks == expected_global_ranks

        actual = op(*args)
        torch.cuda.synchronize(device)
        _assert_matches_reference(actual, expected)

        args[3].fill_(-1)
        dropped = op(*args)
        torch.cuda.synchronize(device)
        assert _output_as_float(dropped).eq(0).all()

        dist.barrier(group=ep_group)
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
        assert tuple(
            dist.get_global_rank(ep_group, group_rank)
            for group_rank in range(ep_size)
        ) == subgroup_memberships[subgroup_index]

        _run_forward_output_case(
            device=device,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            expected_global_ranks=subgroup_memberships[subgroup_index],
        )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _distributed_backward_worker(
    rank: int,
    world_size: int,
    init_file: str,
    combine_format: str,
    gate_up_clamp: float | None = None,
) -> None:
    """Run distributed forward stashing and backward reference parity."""

    from cudnn import MoeEp

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(seconds=300),
    )
    op = None
    try:
        args = make_distributed_forward_inputs(rank, world_size, device)
        config = _forward_config(
            num_experts=2 * world_size,
            ep_group=dist.group.WORLD,
            max_tokens_per_rank=8,
            generate_c=True,
            combine_format=combine_format,
            gate_up_clamp=gate_up_clamp,
        )
        reference = _reference_backward(config)
        grad_output = _grad_output(
            device,
            args[3].shape[0],
            seed=20260820 + rank,
        )
        op = MoeEp(**config)
        _, fc1_c, route_metadata = op(*args)
        stash = (fc1_c, route_metadata)
        expected = _expected_backward(reference, grad_output, args, stash)

        first = op.backward(grad_output, *args[1:], *stash)
        second = op.backward(grad_output, *args[1:], *stash)
        torch.cuda.synchronize(device)

        # Complete collective work before local assertions. A failure before
        # this barrier would leave peer ranks waiting for process-group timeout.
        dist.barrier()
        _assert_backward_matches(first, expected, args[3])
        _assert_backward_matches(second, expected, args[3])

        op.close()
        op = None
    finally:
        if op is not None:
            op.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _make_wgrad_inputs(
    rank: int,
    world_size: int,
    device: torch.device,
):
    """Build routes with negative weights, drops, and one empty local expert."""

    args = list(make_distributed_forward_inputs(rank, world_size, device))
    token_count = args[3].shape[0]
    local_expert = 2 * rank
    remote_expert = 2 * ((rank + 1) % world_size)
    topk_idx = torch.full(
        (token_count, 2),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.empty(
        (token_count, 2),
        dtype=torch.bfloat16,
        device=device,
    )
    weight_rows = (
        (0.5, -0.25),
        (7.0, -1.25),
        (1.5, -9.0),
    )
    for token in range(token_count):
        pattern = token % 3
        if pattern == 0:
            topk_idx[token] = torch.tensor(
                (local_expert, remote_expert),
                dtype=torch.int32,
                device=device,
            )
        elif pattern == 1:
            topk_idx[token, 1] = local_expert
        else:
            topk_idx[token, 0] = remote_expert
        topk_weights[token] = torch.tensor(
            weight_rows[pattern],
            dtype=torch.bfloat16,
            device=device,
        )
    args[3] = topk_idx
    args[4] = topk_weights
    return tuple(args)


def _source_route_expert(
    source_rank: int,
    token: int,
    slot: int,
    world_size: int,
) -> int:
    pattern = token % 3
    if pattern == 0:
        return (
            2 * source_rank
            if slot == 0
            else 2 * ((source_rank + 1) % world_size)
        )
    if pattern == 1:
        return 2 * source_rank if slot == 1 else -1
    return 2 * ((source_rank + 1) % world_size) if slot == 0 else -1


def _assert_local_operand_metadata(
    operands,
    reference_operands,
    *,
    rank: int,
    world_size: int,
) -> None:
    local_experts = 2
    assert operands.expert_offsets.shape == (local_experts,)
    assert operands.valid_route_counts.shape == (local_experts,)
    assert torch.equal(
        operands.route_metadata,
        reference_operands.route_metadata,
    )
    assert torch.equal(
        operands.valid_route_counts,
        reference_operands.valid_route_counts,
    )
    assert torch.equal(
        operands.expert_offsets,
        reference_operands.expert_offsets,
    )

    metadata = operands.route_metadata
    if metadata.numel():
        assert metadata[:, 0].ge(0).all()
        assert metadata[:, 0].lt(local_experts).all()
    counts = torch.bincount(
        metadata[:, 0].to(torch.int64),
        minlength=local_experts,
    ).to(torch.int32)
    assert torch.equal(operands.valid_route_counts, counts)
    assert counts[0] > 0
    assert counts[1] == 0

    expected_offsets = []
    padded_end = 0
    for count in counts.tolist():
        padded_end += ((count + 255) // 256) * 256
        expected_offsets.append(padded_end)
    assert operands.expert_offsets.tolist() == expected_offsets

    for local_expert, source_rank, token, slot in metadata.tolist():
        global_expert = _source_route_expert(
            source_rank,
            token,
            slot,
            world_size,
        )
        assert global_expert != -1
        assert global_expert // local_experts == rank
        assert global_expert % local_experts == local_expert


def _run_grouped_wgrad(
    operands,
    prefix: str,
    *,
    wgrad_tensor=None,
    accumulate_on_output: bool = False,
):
    import cudnn

    return cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=getattr(operands, f"{prefix}_a"),
        b_tensor=getattr(operands, f"{prefix}_b"),
        sfa_tensor=getattr(operands, f"{prefix}_sfa"),
        sfb_tensor=getattr(operands, f"{prefix}_sfb"),
        offsets_tensor=operands.expert_offsets,
        output_mode="dense",
        wgrad_tensor=wgrad_tensor,
        wgrad_dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sf_vec_size=32,
        accumulate_on_output=accumulate_on_output,
    )["wgrad_tensor"]


def _run_wgrad_operand_case(
    *,
    device: torch.device,
    ep_group,
    rank: int,
    world_size: int,
) -> None:
    """Exercise production FC1/FC2 operands through grouped wgrad."""

    from cudnn import MoeEp, MoeEpWgradOperands

    args = _make_wgrad_inputs(rank, world_size, device)
    config = _forward_config(
        num_experts=2 * world_size,
        ep_group=ep_group,
        max_tokens_per_rank=8,
        generate_c=True,
        backward_wgrad_mode="operands",
        token_padding_size=256,
        sf_padding_size=128,
    )
    reference = _reference_backward(config)
    grad_output = _grad_output(
        device,
        args[3].shape[0],
        seed=20260821 + rank,
    )

    with MoeEp(**config) as op:
        output, fc1_c, route_metadata, forward_stash = op(*args)
        (
            reference_output,
            reference_fc1_c,
            reference_metadata,
            reference_stash,
        ) = reference(*_reference_args(args))
        _assert_matches_reference(output, reference_output)
        assert torch.equal(route_metadata, reference_metadata)
        assert forward_stash.route_metadata is route_metadata

        backward = op.backward(
            grad_output,
            *args[1:],
            fc1_c,
            route_metadata,
            wgrad_forward_stash=forward_stash,
        )
        reference_backward = reference.backward(
            grad_output,
            *_reference_args(args)[1:],
            reference_fc1_c,
            reference_metadata,
            wgrad_forward_stash=reference_stash,
        )

    _assert_backward_matches(backward[:2], reference_backward[:2], args[3])
    operands = backward[2]
    reference_operands = reference_backward[2]
    assert isinstance(operands, MoeEpWgradOperands)
    assert operands.route_metadata is forward_stash.route_metadata
    assert operands.expert_offsets is forward_stash.expert_offsets
    assert operands.valid_route_counts is forward_stash.valid_route_counts
    _assert_local_operand_metadata(
        operands,
        reference_operands,
        rank=rank,
        world_size=world_size,
    )

    expected_wgrads = reference_operands.dense_wgrads()
    decoded_wgrads = _dense_wgrads_from_operands(operands)
    for actual, expected in zip(decoded_wgrads, expected_wgrads):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.15,
            atol=0.125,
        )
        assert actual[1].eq(0).all()

    # The in-tree grouped-wgrad implementation is an SM100 kernel. Rubin
    # validates the producer numerics above; the direct SM100 ABI test covers
    # execution and accumulate_on_output with the same public bundle layout.
    if torch.cuda.get_device_capability(device) != (10, 0):
        return

    for prefix, expected in zip(("fc1", "fc2"), expected_wgrads):
        actual = _run_grouped_wgrad(operands, prefix)
        initial = torch.full_like(actual, 0.25)
        accumulated = _run_grouped_wgrad(
            operands,
            prefix,
            wgrad_tensor=initial,
            accumulate_on_output=True,
        )
        torch.cuda.synchronize(device)
        assert accumulated is initial
        torch.testing.assert_close(
            actual.float(),
            expected,
            rtol=0.15,
            atol=0.125,
        )
        torch.testing.assert_close(
            accumulated.float(),
            expected + 0.25,
            rtol=0.15,
            atol=0.125,
        )
        assert actual[1].eq(0).all()
        assert accumulated[1].eq(0.25).all()


def _distributed_wgrad_worker(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(seconds=600),
    )
    try:
        _run_wgrad_operand_case(
            device=device,
            ep_group=dist.group.WORLD,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
