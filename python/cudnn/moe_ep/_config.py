# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public and resolved configuration contracts for :mod:`cudnn.moe_ep`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Literal

import torch.distributed as dist

from ._tuning import MoeEpTuningConfig
from ._types import MoeEpNativeWeightStorageMode, MoeFormat

_PHYSICAL_RECV_POOL_ALIGNMENT = 128
_MAX_KERNEL_ROUTE_INDEX = (1 << 31) - 1


class MoeEpFc1WeightLayout(str, Enum):
    """Logical gate/up ordering supplied by FC1 source weights."""

    GATE_THEN_UP = "gate_then_up"
    GATE_UP_INTERLEAVED_32 = "gate_up_interleaved_32"


def _require_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True, kw_only=True)
class MoeEpModelConfig:
    """Model geometry and routing fan-out."""

    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int

    def __post_init__(self) -> None:
        for name in (
            "num_experts",
            "hidden_size",
            "intermediate_size",
            "top_k",
        ):
            _require_positive_integer(name, getattr(self, name))
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot exceed " f"num_experts ({self.num_experts})")


@dataclass(frozen=True, kw_only=True)
class MoeEpParallelConfig:
    """Expert-parallel group, capacity, overflow, and padding policy."""

    ep_group: dist.ProcessGroup | None = None
    max_tokens_per_rank: int | None = None
    physical_recv_pool_rows: int | None = None
    drop_on_overflow: bool = False
    token_padding_size: int = 128
    sf_padding_size: int = 128

    def __post_init__(self) -> None:
        if self.ep_group is not None and not isinstance(self.ep_group, dist.ProcessGroup):
            raise ValueError("ep_group must be a torch.distributed.ProcessGroup or None, " f"got {type(self.ep_group).__name__}")
        if self.max_tokens_per_rank is not None and (
            isinstance(self.max_tokens_per_rank, bool) or not isinstance(self.max_tokens_per_rank, int) or self.max_tokens_per_rank < 0
        ):
            raise ValueError("max_tokens_per_rank must be a non-negative integer or None")
        if self.physical_recv_pool_rows is not None and (
            isinstance(self.physical_recv_pool_rows, bool) or not isinstance(self.physical_recv_pool_rows, int) or self.physical_recv_pool_rows <= 0
        ):
            raise ValueError("physical_recv_pool_rows must be a positive integer or None")
        if self.physical_recv_pool_rows is not None and self.physical_recv_pool_rows % _PHYSICAL_RECV_POOL_ALIGNMENT:
            raise ValueError("physical_recv_pool_rows must satisfy P % 128 == 0, " f"got P={self.physical_recv_pool_rows}")
        if self.physical_recv_pool_rows is not None and self.physical_recv_pool_rows > _MAX_KERNEL_ROUTE_INDEX:
            raise ValueError(
                "physical_recv_pool_rows exceeds the kernel Int32 route-index " f"limit ({_MAX_KERNEL_ROUTE_INDEX}), got " f"{self.physical_recv_pool_rows}"
            )
        if not isinstance(self.drop_on_overflow, bool):
            raise ValueError("drop_on_overflow must be a bool")
        for name in ("token_padding_size", "sf_padding_size"):
            _require_positive_integer(name, getattr(self, name))
        if self.sf_padding_size % 128:
            raise ValueError("sf_padding_size must be a positive multiple of 128, " f"got {self.sf_padding_size}")


@dataclass(frozen=True, kw_only=True)
class MoeEpDataPathConfig:
    """Formats, top-k placement, FC1 source layout, and clamp semantics."""

    output_format: MoeFormat = MoeFormat.BF16
    combine_format: MoeFormat = MoeFormat.BF16
    apply_topk_in_fc1: bool = True
    fc1_weight_layout: MoeEpFc1WeightLayout = MoeEpFc1WeightLayout.GATE_THEN_UP
    gate_up_clamp: float | None = None

    def __post_init__(self) -> None:
        for name in ("output_format", "combine_format"):
            value = getattr(self, name)
            if not isinstance(value, MoeFormat):
                raise TypeError(f"{name} must be a MoeFormat, got {type(value).__name__}")
        if not isinstance(self.apply_topk_in_fc1, bool):
            raise ValueError("apply_topk_in_fc1 must be a bool")
        if not isinstance(self.fc1_weight_layout, MoeEpFc1WeightLayout):
            raise TypeError("fc1_weight_layout must be a MoeEpFc1WeightLayout, " f"got {type(self.fc1_weight_layout).__name__}")
        value = self.gate_up_clamp
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("gate_up_clamp must be a finite non-negative real number or None")
        canonical = float(value)
        if not math.isfinite(canonical) or canonical < 0.0:
            raise ValueError("gate_up_clamp must be a finite non-negative real number or None")
        if canonical == 0.0:
            canonical = 0.0
        object.__setattr__(self, "gate_up_clamp", canonical)


@dataclass(frozen=True, kw_only=True)
class MoeEpConfig:
    """Complete public semantic configuration for one :class:`MoeEp`."""

    model: MoeEpModelConfig
    parallel: MoeEpParallelConfig = field(default_factory=MoeEpParallelConfig)
    data_path: MoeEpDataPathConfig = field(default_factory=MoeEpDataPathConfig)
    inference_tuning: MoeEpTuningConfig = field(default_factory=MoeEpTuningConfig)
    training_forward_tuning: MoeEpTuningConfig = field(default_factory=MoeEpTuningConfig)
    training_backward_tuning: MoeEpTuningConfig = field(default_factory=MoeEpTuningConfig)
    training_weight_storage_mode: MoeEpNativeWeightStorageMode = MoeEpNativeWeightStorageMode.CONTIGUOUS
    validation_mode: Literal["strict", "trusted"] = "strict"

    def __post_init__(self) -> None:
        for name, expected in (
            ("model", MoeEpModelConfig),
            ("parallel", MoeEpParallelConfig),
            ("data_path", MoeEpDataPathConfig),
            ("inference_tuning", MoeEpTuningConfig),
            ("training_forward_tuning", MoeEpTuningConfig),
            ("training_backward_tuning", MoeEpTuningConfig),
            (
                "training_weight_storage_mode",
                MoeEpNativeWeightStorageMode,
            ),
        ):
            value = getattr(self, name)
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be a {expected.__name__}, " f"got {type(value).__name__}")
        if self.validation_mode not in ("strict", "trusted"):
            raise ValueError("validation_mode must be 'strict' or 'trusted', " f"got {self.validation_mode!r}")
        model = self.model
        data_path = self.data_path
        for name, fmt in (
            ("output_format", data_path.output_format),
            ("combine_format", data_path.combine_format),
        ):
            required_multiple = 32 if fmt is MoeFormat.MXFP8 else 16 if fmt is MoeFormat.NVFP4 else 1
            if model.hidden_size % required_multiple:
                raise ValueError(f"hidden_size ({model.hidden_size}) must be divisible by " f"{required_multiple} for {name}={fmt.value}")
        for name, tuning in (
            ("inference_tuning", self.inference_tuning),
            ("training_forward_tuning", self.training_forward_tuning),
        ):
            if tuning.dgrad_optimization != "baseline":
                raise ValueError(
                    f"{name}.dgrad_optimization must be 'baseline'; " "dgrad optimization profiles are supported only by " "training_backward_tuning"
                )
            if tuning.reduce_topk_in_kernel and (
                data_path.combine_format is not MoeFormat.BF16 or data_path.output_format is not MoeFormat.BF16 or not data_path.apply_topk_in_fc1
            ):
                raise ValueError(f"{name}.reduce_topk_in_kernel requires BF16 " "combine/output and apply_topk_in_fc1=True")
        backward = self.training_backward_tuning
        if backward.token_back_mode != "epi_warps":
            raise ValueError("training_backward_tuning requires " "token_back_mode='epi_warps'")
        if backward.reduce_topk_in_kernel:
            raise ValueError("training_backward_tuning does not support " "reduce_topk_in_kernel=True")


@dataclass(frozen=True)
class _ResolvedMoeEpTopology:
    ep_size: int
    ep_rank: int
    ep_global_ranks: tuple[int, ...]
    experts_per_rank: int


def _required_padded_rows(
    logical_route_capacity: int,
    *,
    experts_per_rank: int,
    padding_block: int,
) -> int:
    """Mirror the upstream deterministic-router padding formula."""

    active_expert_count = min(experts_per_rank, logical_route_capacity)
    padded_block_count = active_expert_count + (logical_route_capacity - active_expert_count) // padding_block
    return padded_block_count * padding_block


def _max_safe_logical_route_capacity(
    physical_rows: int,
    *,
    raw_route_count: int,
    experts_per_rank: int,
    padding_block: int,
) -> int:
    """Return the largest topology-valid L whose padded rows fit in P."""

    lower = 0
    upper = raw_route_count
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        required_rows = _required_padded_rows(
            candidate,
            experts_per_rank=experts_per_rank,
            padding_block=padding_block,
        )
        if required_rows <= physical_rows:
            lower = candidate
        else:
            upper = candidate - 1
    if lower <= 0:
        raise ValueError("physical_recv_pool_rows cannot hold any positive logical route " f"capacity with padding_block={padding_block}: P={physical_rows}")
    return lower


@dataclass(frozen=True)
class _ResolvedMoeEpReceiveCapacity:
    """Static exact-P contract plus operation-scoped logical capacities."""

    raw_route_count: int
    physical_recv_pool_rows: int
    training_logical_route_capacity: int
    inference_logical_route_capacity: int
    training_required_padded_rows: int
    inference_required_padded_rows: int
    training_sf_pool_rows: int
    inference_sf_pool_rows: int


def _resolve_receive_capacity(
    config: MoeEpConfig,
    topology: _ResolvedMoeEpTopology,
) -> _ResolvedMoeEpReceiveCapacity:
    parallel = config.parallel
    max_tokens_per_rank = parallel.max_tokens_per_rank
    if max_tokens_per_rank is None or max_tokens_per_rank <= 0:
        raise ValueError("resolving receive capacity requires a positive " "max_tokens_per_rank")

    raw_route_count = topology.ep_size * max_tokens_per_rank * config.model.top_k
    if raw_route_count > _MAX_KERNEL_ROUTE_INDEX:
        raise ValueError("raw route count exceeds the kernel Int32 route-index limit: " f"R={raw_route_count}, limit={_MAX_KERNEL_ROUTE_INDEX}")

    training_padding = _PHYSICAL_RECV_POOL_ALIGNMENT
    inference_padding = parallel.token_padding_size
    if parallel.physical_recv_pool_rows is None:
        canonical_rows = max(
            _required_padded_rows(
                raw_route_count,
                experts_per_rank=topology.experts_per_rank,
                padding_block=training_padding,
            ),
            _required_padded_rows(
                raw_route_count,
                experts_per_rank=topology.experts_per_rank,
                padding_block=inference_padding,
            ),
        )
        physical_rows = (canonical_rows + _PHYSICAL_RECV_POOL_ALIGNMENT - 1) // _PHYSICAL_RECV_POOL_ALIGNMENT * _PHYSICAL_RECV_POOL_ALIGNMENT
    else:
        physical_rows = parallel.physical_recv_pool_rows
    if physical_rows > _MAX_KERNEL_ROUTE_INDEX:
        raise ValueError("resolved physical receive pool exceeds the kernel Int32 " f"route-index limit: P={physical_rows}")

    training_logical = _max_safe_logical_route_capacity(
        physical_rows,
        raw_route_count=raw_route_count,
        experts_per_rank=topology.experts_per_rank,
        padding_block=training_padding,
    )
    inference_logical = _max_safe_logical_route_capacity(
        physical_rows,
        raw_route_count=raw_route_count,
        experts_per_rank=topology.experts_per_rank,
        padding_block=inference_padding,
    )
    training_required = _required_padded_rows(
        training_logical,
        experts_per_rank=topology.experts_per_rank,
        padding_block=training_padding,
    )
    inference_required = _required_padded_rows(
        inference_logical,
        experts_per_rank=topology.experts_per_rank,
        padding_block=inference_padding,
    )

    if config.training_backward_tuning.dgrad_optimization == "ds3_ep4_pattern":
        required_for_all_routes = _required_padded_rows(
            raw_route_count,
            experts_per_rank=topology.experts_per_rank,
            padding_block=training_padding,
        )
        if physical_rows < required_for_all_routes:
            raise ValueError(
                "dgrad_optimization='ds3_ep4_pattern' requires "
                "physical_recv_pool_rows >= the padded full-topology "
                f"capacity ({required_for_all_routes}), got {physical_rows}"
            )
        training_logical = raw_route_count
        training_required = required_for_all_routes

    training_sf_required = _required_padded_rows(
        training_logical,
        experts_per_rank=topology.experts_per_rank,
        padding_block=_PHYSICAL_RECV_POOL_ALIGNMENT,
    )
    inference_sf_required = _required_padded_rows(
        inference_logical,
        experts_per_rank=topology.experts_per_rank,
        padding_block=parallel.sf_padding_size,
    )
    training_sf_rows = max(physical_rows, training_sf_required)
    inference_sf_rows = max(physical_rows, inference_sf_required)
    if max(training_sf_rows, inference_sf_rows) > _MAX_KERNEL_ROUTE_INDEX:
        raise ValueError(
            "resolved scale-factor row capacity exceeds the kernel Int32 " "route-index limit: " f"training={training_sf_rows}, inference={inference_sf_rows}"
        )
    return _ResolvedMoeEpReceiveCapacity(
        raw_route_count=raw_route_count,
        physical_recv_pool_rows=physical_rows,
        training_logical_route_capacity=training_logical,
        inference_logical_route_capacity=inference_logical,
        training_required_padded_rows=training_required,
        inference_required_padded_rows=inference_required,
        training_sf_pool_rows=training_sf_rows,
        inference_sf_pool_rows=inference_sf_rows,
    )


@dataclass(frozen=True)
class ResolvedMoeEpConfig:
    """Private operator contract after EP topology resolution."""

    public_config: MoeEpConfig
    topology: _ResolvedMoeEpTopology
    receive_capacity: _ResolvedMoeEpReceiveCapacity = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receive_capacity",
            _resolve_receive_capacity(self.public_config, self.topology),
        )


def resolve_moe_ep_config(config: MoeEpConfig) -> ResolvedMoeEpConfig:
    """Resolve process-group topology exactly once for a config generation."""

    if not isinstance(config, MoeEpConfig):
        raise TypeError(f"config must be a MoeEpConfig, got {type(config).__name__}")
    group = config.parallel.ep_group
    if group is None:
        ep_size = 1
        ep_rank = 0
        ep_global_ranks: tuple[int, ...] = ()
    else:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("ep_group requires an initialized torch.distributed process group")
        ep_size = dist.get_world_size(group)
        ep_rank = dist.get_rank(group)
        if ep_size <= 0 or ep_rank < 0 or ep_rank >= ep_size:
            raise ValueError("the current process must be a member of ep_group")
        ep_global_ranks = tuple(dist.get_global_rank(group, group_rank) for group_rank in range(ep_size))
        if len(set(ep_global_ranks)) != ep_size:
            raise RuntimeError("ep_group returned duplicate global ranks")
        if ep_global_ranks[ep_rank] != dist.get_rank():
            raise RuntimeError("ep_group rank mapping is inconsistent with the current global rank")

    num_experts = config.model.num_experts
    if num_experts % ep_size:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by EP size ({ep_size})")
    return ResolvedMoeEpConfig(
        public_config=config,
        topology=_ResolvedMoeEpTopology(
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_global_ranks=ep_global_ranks,
            experts_per_rank=num_experts // ep_size,
        ),
    )


__all__ = [
    "MoeEpConfig",
    "MoeEpDataPathConfig",
    "MoeEpFc1WeightLayout",
    "MoeEpModelConfig",
    "MoeEpParallelConfig",
]
