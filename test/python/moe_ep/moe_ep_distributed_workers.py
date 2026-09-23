# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Picklable multiprocessing workers for distributed MoE EP tests."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from moe_ep.moe_ep_test_support import (
    _allocate_stateless_training_outputs,
    _allocate_training_weight_staging,
    _assert_backward_matches,
    _assert_grouped_wgrads_match_reference,
    _assert_matches_reference,
    _assert_wgrads_match_reference,
    _dense_wgrads_from_grouped_kernel,
    _fixed_training_reference,
    _fixed_training_weights,
    _forward_config,
    _grad_output,
    _interleave_fc1_wgrad,
    _make_discrete_training_weights,
    _moe_ep_config,
    _output_as_float,
    _poison_training_outputs_for_test,
    _reference_forward,
    make_distributed_forward_inputs,
    quantize_mxfp8,
)

__all__ = [
    "_distributed_autotune_worker",
    "_distributed_backward_worker",
    "_distributed_inference_overflow_worker",
    "_distributed_inference_policy_mismatch_worker",
    "_distributed_output_worker",
    "_distributed_subgroup_output_worker",
    "_run_backward_reference_case",
    "_run_forward_output_case",
]


def _distributed_autotune_worker(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    """Tune before and after backend creation on rank-local CUDA devices."""

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
    op = None
    try:
        from cudnn import MoeEp

        args = make_distributed_forward_inputs(rank, world_size, device)
        config = _forward_config(
            num_experts=2 * world_size,
            ep_group=dist.group.WORLD,
            max_tokens_per_rank=8,
        )
        expected = _reference_forward(args, **config)
        records = []
        for warm_backend in (False, True):
            op = MoeEp(_moe_ep_config(**config))
            warm_snapshot = None
            backend_created_before_tuning = None
            if warm_backend:
                warm = op(*args)
                warm_snapshot = _output_as_float(warm).clone()
                torch.cuda.synchronize(device)
                backend_created_before_tuning = op._execution_state.backend is not None

            result = op.autotune_inference(
                *args,
                candidates=[op.inference_tuning],
                warmup_iters=0,
                timed_iters=1,
            )
            actual = op(*args)
            actual_snapshot = _output_as_float(actual).clone()
            torch.cuda.synchronize(device)
            winners = [None] * world_size
            dist.all_gather_object(winners, result.winner)
            records.append(
                (
                    warm_backend,
                    warm_snapshot,
                    backend_created_before_tuning,
                    actual_snapshot,
                    result,
                    winners,
                    op.inference_tuning,
                )
            )

            dist.barrier()
            op.close()
            op = None
            dist.barrier()

        assertion_error = None
        try:
            for (
                warm_backend,
                warm_snapshot,
                backend_created_before_tuning,
                actual_snapshot,
                result,
                winners,
                applied_tuning,
            ) in records:
                assert all(winner == result.winner for winner in winners)
                assert applied_tuning == result.winner
                if warm_backend:
                    assert backend_created_before_tuning
                    assert warm_snapshot is not None
                    _assert_matches_reference(warm_snapshot, expected)
                _assert_matches_reference(actual_snapshot, expected)
        except BaseException as error:
            assertion_error = error

        local_error = None if assertion_error is None else (type(assertion_error).__name__, str(assertion_error))
        rank_errors = [None] * world_size
        dist.all_gather_object(rank_errors, local_error)
        dist.barrier()
        if any(error is not None for error in rank_errors):
            raise AssertionError(f"distributed autotune assertions failed: {rank_errors}") from assertion_error
    finally:
        if op is not None:
            op.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_forward_output_case(
    *,
    device: torch.device,
    ep_group,
    ep_rank: int,
    ep_size: int,
    combine_format: str = "bf16",
    expected_global_ranks: tuple[int, ...] | None = None,
) -> None:
    """Run inference-forward parity across two dense routing patterns."""

    from cudnn import MoeEp

    args = make_distributed_forward_inputs(ep_rank, ep_size, device)
    config = _forward_config(
        num_experts=2 * ep_size,
        ep_group=ep_group,
        max_tokens_per_rank=8,
        combine_format=combine_format,
    )
    expected = _reference_forward(args, **config)
    alternate_topk_idx = args[3].flip(1).contiguous()
    alternate_args = (*args[:3], alternate_topk_idx, args[4])
    alternate_expected = _reference_forward(alternate_args, **config)
    op = MoeEp(_moe_ep_config(**config))
    try:
        actual = op(*args)
        actual_snapshot = _output_as_float(actual).clone()
        torch.cuda.synchronize(device)

        args[3].copy_(alternate_topk_idx)
        alternate = op(*args)
        alternate_snapshot = _output_as_float(alternate).clone()
        torch.cuda.synchronize(device)

        dist.barrier(group=ep_group)
        assertion_error = None
        try:
            assert op.ep_rank == ep_rank
            if expected_global_ranks is not None:
                assert op.ep_global_ranks == expected_global_ranks
            _assert_matches_reference(actual_snapshot, expected)
            _assert_matches_reference(alternate_snapshot, alternate_expected)
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


def _distributed_inference_overflow_worker(
    rank: int,
    world_size: int,
    init_file: str,
    drop_on_overflow: bool,
) -> None:
    """Exercise one-destination overflow with identical raw flags on all ranks."""

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
    op = None
    try:
        from cudnn import MoeEp

        non_overflow_tokens = 128 // world_size
        overflow_tokens = non_overflow_tokens + 1
        generator = torch.Generator(device=device).manual_seed(20260921 + rank)
        fc1_weight = quantize_mxfp8(
            torch.randn(
                1,
                128,
                512,
                generator=generator,
                device=device,
            )
            / 8,
            axis=1,
        )
        fc2_weight = quantize_mxfp8(
            torch.randn(
                1,
                256,
                128,
                generator=generator,
                device=device,
            )
            / 8,
            axis=1,
        )

        def make_args(tokens: int):
            activation = quantize_mxfp8(
                torch.randn(
                    tokens,
                    128,
                    generator=generator,
                    device=device,
                ),
                axis=1,
            )
            topk_idx = torch.zeros(
                (tokens, 1),
                dtype=torch.int32,
                device=device,
            )
            if rank == 0:
                # Keep one under-capacity route to rank 1 while rank 0
                # overflows, making local-only truncation observable.
                topk_idx[0, 0] = 1
            topk_weights = torch.ones(
                (tokens, 1),
                dtype=torch.bfloat16,
                device=device,
            )
            return (
                activation,
                fc1_weight,
                fc2_weight,
                topk_idx,
                topk_weights,
            )

        config = _forward_config(
            num_experts=world_size,
            hidden_size=128,
            intermediate_size=256,
            top_k=1,
            ep_group=dist.group.WORLD,
            max_tokens_per_rank=overflow_tokens,
            physical_recv_pool_rows=128,
            drop_on_overflow=drop_on_overflow,
        )
        warmup_args = make_args(non_overflow_tokens)
        overflow_args = make_args(overflow_tokens)
        reference_config = dict(config)
        reference_config.pop("physical_recv_pool_rows")
        reference_config.pop("drop_on_overflow")
        expected = _reference_forward(overflow_args, **reference_config) if drop_on_overflow else None
        op = MoeEp(_moe_ep_config(**config))
        op(*warmup_args)
        torch.cuda.synchronize(device)

        local_receive_counts = [overflow_tokens] + [0] * (world_size - 1)
        if rank == 0:
            local_receive_counts[0] -= 1
            local_receive_counts[1] += 1
        gathered_receive_counts = [None] * world_size
        dist.all_gather_object(
            gathered_receive_counts,
            local_receive_counts,
        )
        host_receive_counts = [sum(counts[destination] for counts in gathered_receive_counts) for destination in range(world_size)]
        expected_overflow = any(count > 128 for count in host_receive_counts)
        assert expected_overflow

        if not drop_on_overflow:
            try:
                op(*overflow_args)
                torch.cuda.synchronize(device)
            except BaseException as exc:
                message = str(exc).lower()
                expected_errors = (
                    "device-side assert",
                    "unspecified launch failure",
                    "cudaerrorlaunchfailure",
                    "cuda_error_launch_failed",
                    "receive route-pool overflow",
                )
                os._exit(0 if any(error in message for error in expected_errors) else 2)
            os._exit(1)

        actual = op(*overflow_args)
        torch.cuda.synchronize(device)
        assert actual.shape == (overflow_tokens, 128)
        if rank == 0:
            assert expected is not None
            _assert_matches_reference(actual[:1], expected[:1])

        backend = op._execution_state.backend
        assert backend is not None
        owner = backend._inference_resources
        assert owner is not None and owner._workspace is not None
        overflow_flag = owner._workspace.views(overflow_tokens).local["overflow_flag"][: torch.int32.itemsize].view(torch.int32)
        gathered_flags = [torch.empty_like(overflow_flag) for _ in range(world_size)]
        dist.all_gather(gathered_flags, overflow_flag)
        assert [int(flag.item()) for flag in gathered_flags] == [int(expected_overflow)] * world_size

        probe = torch.ones(1, device=device) + 1
        torch.cuda.synchronize(device)
        assert float(probe.item()) == 2.0
        dist.barrier()
        op.close()
        op = None
        dist.barrier()
    finally:
        if op is not None:
            op.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _distributed_inference_policy_mismatch_worker(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    """Reject a public overflow-policy mismatch before entering the kernel."""

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
        from cudnn import MoeEp

        args = make_distributed_forward_inputs(rank, world_size, device)
        op = MoeEp(
            _moe_ep_config(
                **_forward_config(
                    num_experts=2 * world_size,
                    ep_group=dist.group.WORLD,
                    max_tokens_per_rank=8,
                    drop_on_overflow=bool(rank % 2),
                )
            )
        )
        try:
            try:
                op(*args)
            except RuntimeError as exc:
                assert "must match on every expert-parallel rank" in str(exc)
            else:
                raise AssertionError("rank-mismatched drop_on_overflow reached the kernel")
        finally:
            op.close()
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
        subgroups = [dist.new_group(list(members), backend="nccl") for members in subgroup_memberships]
        subgroup_index = global_rank % 2
        ep_group = subgroups[subgroup_index]
        ep_rank = dist.get_rank(ep_group)
        ep_size = dist.get_world_size(ep_group)
        actual_global_ranks = tuple(dist.get_global_rank(ep_group, group_rank) for group_rank in range(ep_size))

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
        [[local_expert, remote_expert], [local_expert, local_expert]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.tensor(
        [[0.625, 0.375], [0.25, 0.75]],
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


def _make_distributed_uneven_backward_inputs(
    ep_rank: int,
    ep_size: int,
    device: torch.device,
):
    """Build training inputs with rank-dependent local token counts."""

    base_args = make_distributed_forward_inputs(ep_rank, ep_size, device)
    token_count = ep_rank + 1
    base_token_count = int(base_args[0].shape[0])
    repeats = (token_count + base_token_count - 1) // base_token_count
    activation = quantize_mxfp8(
        base_args[0].dequantize(torch.float32).repeat((repeats, 1))[:token_count].contiguous(),
        axis=1,
    )
    args = (
        activation,
        base_args[1],
        base_args[2],
        base_args[3].repeat((repeats, 1))[:token_count].contiguous(),
        base_args[4].float().repeat((repeats, 1))[:token_count].contiguous(),
    )
    grad_output = quantize_mxfp8(
        _grad_output(
            device,
            args[0].shape[0],
            seed=20261001 + ep_rank,
        ),
        axis=1,
    )
    return args, grad_output


def _run_backward_reference_case(
    *,
    device: torch.device,
    ep_group,
    ep_rank: int,
    ep_size: int,
    combine_format: str = "bf16",
    gate_up_clamp: float | None = None,
    expected_global_ranks: tuple[int, ...] | None = None,
    native_weight_storage_mode: str = "contiguous",
    uneven_token_input: bool = False,
) -> None:
    """Run stateless training after the independent distributed oracle."""

    from cudnn import (
        MoeEp,
        MoeEpFc1WeightLayout,
        MoeEpNativeWeightStorageMode,
    )

    if uneven_token_input:
        args, grad_output = _make_distributed_uneven_backward_inputs(
            ep_rank,
            ep_size,
            device,
        )
        max_tokens_per_rank = ep_size
        physical_recv_pool_rows = 256
        local_token_count = torch.tensor(
            [args[0].shape[0]],
            dtype=torch.int64,
            device=device,
        )
        gathered_token_counts = [torch.empty_like(local_token_count) for _ in range(ep_size)]
        dist.all_gather(
            gathered_token_counts,
            local_token_count,
            group=ep_group,
        )
        token_counts = tuple(int(count.item()) for count in gathered_token_counts)
    else:
        args, grad_output = _make_distributed_backward_inputs(
            ep_rank,
            ep_size,
            device,
        )
        max_tokens_per_rank = int(args[0].shape[0])
        physical_recv_pool_rows = 128
        token_counts = None
    num_experts = 2 * ep_size

    # Finish all collective reference work, including dense local dW, before
    # constructing or launching the production operator.
    reference_grad_output = grad_output if isinstance(grad_output, torch.Tensor) else grad_output.dequantize(torch.float32)
    expected = _fixed_training_reference(
        args,
        reference_grad_output,
        combine_format=combine_format,
        gate_up_clamp=gate_up_clamp,
        ep_group=ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        physical_recv_pool_rows=physical_recv_pool_rows,
        drop_on_overflow=True,
    )
    expected_y, expected_dx, expected_dprob, expected_wgrads = expected
    expected_fc1_wgrad, expected_fc2_wgrad = expected_wgrads.dense_wgrads()
    expected_dense_wgrads = (
        _interleave_fc1_wgrad(expected_fc1_wgrad),
        expected_fc2_wgrad,
    )
    weights = _fixed_training_weights(args)

    op = MoeEp(
        _moe_ep_config(
            num_experts=num_experts,
            hidden_size=128,
            intermediate_size=256,
            top_k=2,
            ep_group=ep_group,
            max_tokens_per_rank=max_tokens_per_rank,
            physical_recv_pool_rows=physical_recv_pool_rows,
            drop_on_overflow=True,
            combine_format=combine_format,
            gate_up_clamp=gate_up_clamp,
            fc1_weight_layout=(MoeEpFc1WeightLayout.GATE_UP_INTERLEAVED_32),
            training_weight_storage_mode=MoeEpNativeWeightStorageMode(native_weight_storage_mode),
        )
    )
    try:
        requirements = op.prepare_training(device=device)
        forward_staging, backward_staging = _allocate_training_weight_staging(weights)
        native_forward = op.pack_forward_weights(
            weights[0],
            out=forward_staging,
        )
        native_backward = op.pack_backward_weights(
            weights[1],
            out=backward_staging,
        )
        discrete_owners = None
        if native_weight_storage_mode == "discrete":
            (
                native_forward,
                native_backward,
                discrete_owners,
            ) = _make_discrete_training_weights(native_forward, native_backward)
            assert discrete_owners
        forward_out, backward_out = _allocate_stateless_training_outputs(
            requirements,
            device,
            op.training_symmetric_buffers(),
        )
        _poison_training_outputs_for_test(forward_out, backward_out)
        actual_y = op.training_forward(
            args[0],
            args[3],
            args[4],
            weights=native_forward,
            out=forward_out,
        )
        actual_dx, actual_dprob, actual_wgrads = op.training_backward(
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
        grouped_wgrads = _dense_wgrads_from_grouped_kernel(actual_wgrads)
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
            assert args[3][0, 0] // 2 == ep_rank
            assert args[3][0, 1] // 2 == (ep_rank + 1) % ep_size
            if uneven_token_input:
                assert token_counts == tuple(range(1, ep_size + 1))
                assert len(set(token_counts)) == ep_size
            else:
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
            expected_offsets = torch.cumsum(
                torch.div(
                    actual_wgrads.valid_route_counts + 127,
                    128,
                    rounding_mode="floor",
                )
                * 128,
                dim=0,
                dtype=actual_wgrads.expert_offsets.dtype,
            )
            torch.testing.assert_close(
                actual_wgrads.expert_offsets,
                expected_offsets,
                rtol=0,
                atol=0,
            )
            _assert_grouped_wgrads_match_reference(
                grouped_wgrads,
                expected_dense_wgrads,
                reference_name="the independent PyTorch MXFP8 reference",
            )
        except BaseException as error:
            assertion_error = error
        dist.barrier(group=ep_group)
        if assertion_error is not None:
            raise assertion_error
    finally:
        op.close()


def _distributed_backward_worker(
    rank: int,
    world_size: int,
    init_file: str,
    native_weight_storage_mode: str = "contiguous",
) -> None:
    """Run one WORLD-scoped distributed training parity case."""

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
        _run_backward_reference_case(
            device=device,
            ep_group=dist.group.WORLD,
            ep_rank=rank,
            ep_size=world_size,
            expected_global_ranks=tuple(range(world_size)),
            native_weight_storage_mode=native_weight_storage_mode,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
