# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Torchrun-native multi-node MoE EP forward acceptance tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from moe_ep.moe_ep_distributed_workers import (
    _run_forward_output_case,
)


pytestmark = [
    pytest.mark.L1,
    pytest.mark.gpu_exclusive,
    pytest.mark.moe_ep_multinode,
]

_TORCHRUN_ENV = ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "RANK", "WORLD_SIZE")
_PROCESS_GROUP_TIMEOUT = timedelta(minutes=10)


def _bind_torchrun_device_before_pytest_fixtures() -> None:
    """Bind before the root conftest creates its session CUDA handle."""

    value = os.environ.get("LOCAL_RANK")
    if value is None or not torch.cuda.is_available():
        return
    local_rank = int(value)
    if 0 <= local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)


_bind_torchrun_device_before_pytest_fixtures()


@dataclass(frozen=True)
class _TorchrunWorld:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    device: torch.device


def _require_torchrun_environment() -> tuple[int, int, int, int]:
    missing = [name for name in _TORCHRUN_ENV if name not in os.environ]
    if missing:
        pytest.skip(
            "multi-node MoE EP forward requires torchrun environment variables: "
            + ", ".join(missing)
        )
    return (
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["LOCAL_WORLD_SIZE"]),
    )


@pytest.fixture(scope="session")
def torchrun_world():
    if not dist.is_available() or not dist.is_nccl_available():
        pytest.skip("multi-node Rubin MXFP8 forward requires NCCL")

    rank, world_size, local_rank, local_world_size = (
        _require_torchrun_environment()
    )
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        pytest.skip(
            f"torchrun LOCAL_RANK={local_rank} is not backed by a visible GPU"
        )

    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (10, 7):
        pytest.skip(
            "multi-node Rubin MXFP8 forward requires exactly SM107 "
            "(compute capability 10.7) on every rank"
        )
    try:
        import nvshmem.core  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("multi-node Rubin MXFP8 forward requires NVSHMEM")

    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    torch.cuda.set_device(device)
    if dist.is_initialized():
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError(
                "existing process group does not match torchrun RANK/WORLD_SIZE"
            )
    else:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
            timeout=_PROCESS_GROUP_TIMEOUT,
        )

    context = _TorchrunWorld(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        device=device,
    )
    try:
        yield context
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


@pytest.mark.parametrize(
    (
        "ep_size",
        "required_world_size",
        "required_local_world_size",
        "ep_global_ranks",
    ),
    [
        pytest.param(
            7,
            14,
            2,
            tuple(range(0, 14, 2)),
            id="ep7-world14",
        ),
        pytest.param(
            12,
            12,
            4,
            tuple(range(12)),
            id="ep12-world12",
        ),
        pytest.param(
            15,
            20,
            4,
            tuple(rank for rank in range(20) if rank % 4 < 3),
            id="ep15-world20",
        ),
        pytest.param(
            16,
            16,
            4,
            tuple(range(16)),
            id="ep16-world16",
        ),
    ],
)
@pytest.mark.parametrize("combine_format", ["bf16", "mxfp8"])
def test_mxfp8_forward_multinode_matches_reference(
    torchrun_world,
    ep_size,
    required_world_size,
    required_local_world_size,
    ep_global_ranks,
    combine_format,
):
    world = torchrun_world
    if (
        world.world_size != required_world_size
        or world.local_world_size != required_local_world_size
    ):
        pytest.skip(
            f"EP{ep_size} requires torchrun WORLD_SIZE={required_world_size}, "
            f"LOCAL_WORLD_SIZE={required_local_world_size}; got "
            f"WORLD_SIZE={world.world_size}, "
            f"LOCAL_WORLD_SIZE={world.local_world_size}"
        )

    if ep_size == world.world_size:
        ep_group = dist.group.WORLD
    else:
        # All WORLD ranks must create subgroups in the same order, including
        # idle ranks that are not members of this balanced EP group.
        ep_group = dist.new_group(
            list(ep_global_ranks),
            backend="nccl",
            timeout=_PROCESS_GROUP_TIMEOUT,
        )

    is_ep_member = world.rank in ep_global_ranks
    try:
        if is_ep_member:
            ep_rank = dist.get_rank(ep_group)
            _run_forward_output_case(
                device=world.device,
                ep_group=ep_group,
                ep_rank=ep_rank,
                ep_size=ep_size,
                combine_format=combine_format,
                expected_global_ranks=ep_global_ranks,
            )
        dist.barrier()
    finally:
        if ep_group is not dist.group.WORLD and is_ep_member:
            dist.destroy_process_group(ep_group)
        dist.barrier()
