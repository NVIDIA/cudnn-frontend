# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Eager-only ownership and materialization for the MXFP8 FC1 stash."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..._contracts import ForwardConfig, ValidatedForwardRequest
from ..._types import MoeEpWgradForwardStash
from .._workspace import _align_up
from ._adapter import _GATE_UP_INTERLEAVE, Mxfp8LaunchInputs
from ._compile import PreparedMxfp8Kernel
from ._wgrad_layout import (
    assemble_discrete_col_requant_scales,
    cumulative_padded_offsets,
    pool_data_as_wgrad_a,
)

_TOKEN_SRC_METADATA_BYTES = 8


@dataclass(frozen=True)
class Mxfp8StashPlan:
    """One launch's logical counts over an instance-owned raw C buffer."""

    buffer: torch.Tensor
    expert_counts: tuple[int, ...]
    expert_offsets: tuple[int, ...]
    local_routes: int


class Mxfp8ForwardStash:
    """Own a padded high-watermark C buffer and compact public results."""

    def __init__(self, config: ForwardConfig, device: torch.device) -> None:
        if not config.generate_c:
            raise ValueError("Mxfp8ForwardStash requires generate_c=True")
        self.config = config
        self.device = torch.device(device)
        self._buffer: torch.Tensor | None = None
        self._allocation_count = 0
        self._closed = False

    @property
    def capacity(self) -> int:
        return 0 if self._buffer is None else int(self._buffer.shape[0])

    @property
    def allocation_count(self) -> int:
        return self._allocation_count

    def _local_expert_counts(
        self,
        request: ValidatedForwardRequest,
    ) -> tuple[int, ...]:
        flat_experts = request.topk_idx.reshape(-1).to(torch.int64)
        valid_experts = flat_experts[flat_experts >= 0]
        counts = torch.bincount(
            valid_experts,
            minlength=self.config.num_experts,
        ).to(torch.int64)
        if self.config.ep_size > 1:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "distributed generate_c route counting requires an "
                    "initialized torch.distributed process group"
                )
            dist.all_reduce(
                counts,
                op=dist.ReduceOp.SUM,
                group=self.config.ep_group,
            )
        begin = self.config.ep_rank * self.config.experts_per_rank
        end = begin + self.config.experts_per_rank
        return tuple(int(value) for value in counts[begin:end].cpu().tolist())

    def prepare(
        self,
        request: ValidatedForwardRequest,
        *,
        pool_token_capacity: int,
    ) -> Mxfp8StashPlan:
        if self._closed:
            raise RuntimeError("MXFP8 forward stash is closed")
        if request.device != self.device:
            raise ValueError(
                f"MXFP8 forward stash is bound to {self.device}, "
                f"got {request.device}"
            )
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "MoeEp generate_c=True is eager-only and does not support "
                "CUDA graph capture"
            )
        if pool_token_capacity <= 0:
            raise ValueError("pool_token_capacity must be positive")

        expert_counts = self._local_expert_counts(request)
        expert_offsets = []
        padded_routes = 0
        token_padding = (
            self.config.token_padding_size
            if self.config.backward_wgrad_mode == "operands"
            else 128
        )
        for count in expert_counts:
            expert_offsets.append(padded_routes)
            padded_routes += _align_up(
                count,
                token_padding,
            )

        if padded_routes > pool_token_capacity:
            raise RuntimeError(
                "forward stash route layout exceeds Rubin pool capacity: "
                f"{padded_routes} > {pool_token_capacity}"
            )
        # The upstream training kernel validates the receiver-domain C tensor
        # against its complete pool shape, even though only active expert rows
        # are materialized for the public stash.
        required_capacity = pool_token_capacity
        if self._buffer is None or self.capacity < required_capacity:
            self._buffer = torch.empty(
                required_capacity,
                2 * self.config.intermediate_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._allocation_count += 1

        return Mxfp8StashPlan(
            buffer=self._buffer,
            expert_counts=expert_counts,
            expert_offsets=tuple(expert_offsets),
            local_routes=sum(expert_counts),
        )

    def materialize(
        self,
        plan: Mxfp8StashPlan,
        inputs: Mxfp8LaunchInputs,
        prepared: PreparedMxfp8Kernel,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        MoeEpWgradForwardStash | None,
    ]:
        """Compact padded kernel rows into the documented public stash."""

        if self._closed:
            raise RuntimeError("MXFP8 forward stash is closed")
        if inputs.fc1_c is not plan.buffer:
            raise ValueError("launch inputs do not use this stash plan's buffer")
        if prepared.token_src_metadata_bytes != (
            prepared.pool_token_capacity * _TOKEN_SRC_METADATA_BYTES
        ):
            raise RuntimeError("unexpected token_src_metadata byte size")

        local_routes = plan.local_routes
        if local_routes == 0:
            fc1_c = torch.empty(
                (0, 2 * self.config.intermediate_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
            route_metadata = torch.empty(
                (0, 4),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            physical_rows = torch.cat(
                tuple(
                    torch.arange(
                        count,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    + offset
                    for count, offset in zip(
                        plan.expert_counts,
                        plan.expert_offsets,
                    )
                    if count
                )
            )
            local_experts = torch.repeat_interleave(
                torch.arange(
                    self.config.experts_per_rank,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor(
                    plan.expert_counts,
                    dtype=torch.int64,
                    device=self.device,
                ),
                output_size=local_routes,
            )

            metadata_region = inputs.shared_workspace.narrow(
                0,
                prepared.token_src_metadata_offset,
                prepared.token_src_metadata_bytes,
            )
            packed_metadata = metadata_region.view(torch.int64).index_select(
                0,
                physical_rows,
            )
            src_tokens = packed_metadata & 0xFFFFFFFF
            high = packed_metadata >> 32
            src_ranks = (high >> 16) & 0xFFFF
            src_slots = high & 0xFFFF

            order_key = (
                (
                    local_experts * self.config.ep_size
                    + src_ranks
                )
                * int(self.config.max_tokens_per_rank)
                + src_tokens
            ) * self.config.top_k + src_slots
            order = torch.argsort(order_key, stable=True)
            physical_rows = physical_rows.index_select(0, order)
            local_experts = local_experts.index_select(0, order)
            src_ranks = src_ranks.index_select(0, order)
            src_tokens = src_tokens.index_select(0, order)
            src_slots = src_slots.index_select(0, order)

            raw_fc1_c = plan.buffer.index_select(0, physical_rows)
            pairs = self.config.intermediate_size // _GATE_UP_INTERLEAVE
            gate_up_blocks = raw_fc1_c.reshape(
                local_routes,
                pairs,
                2,
                _GATE_UP_INTERLEAVE,
            )
            gate = gate_up_blocks[:, :, 0, :].reshape(
                local_routes,
                self.config.intermediate_size,
            )
            up = gate_up_blocks[:, :, 1, :].reshape(
                local_routes,
                self.config.intermediate_size,
            )
            fc1_c = torch.cat((gate, up), dim=1)
            route_metadata = torch.stack(
                (local_experts, src_ranks, src_tokens, src_slots),
                dim=1,
            ).to(torch.int32)

        wgrad_stash = None
        if self.config.backward_wgrad_mode == "operands":
            if inputs.col_quant_data is None or inputs.col_quant_sf is None:
                raise RuntimeError(
                    "wgrad operand mode requires forward column-requant outputs"
                )
            padded_ends, expert_offsets = cumulative_padded_offsets(
                plan.expert_counts,
                self.config.token_padding_size,
                self.device,
            )
            padded_routes = padded_ends[-1] if padded_ends else 0
            fc1_a = pool_data_as_wgrad_a(
                inputs.col_quant_data,
                padded_routes,
            )
            fc1_sfa = assemble_discrete_col_requant_scales(
                inputs.col_quant_sf,
                plan.expert_counts,
                padded_ends,
                self.config.hidden_size,
                self.config.sf_padding_size,
            )
            valid_route_counts = torch.tensor(
                plan.expert_counts,
                dtype=torch.int32,
                device=self.device,
            )
            wgrad_stash = MoeEpWgradForwardStash(
                fc1_a=fc1_a,
                fc1_sfa=fc1_sfa,
                expert_offsets=expert_offsets,
                valid_route_counts=valid_route_counts,
                route_metadata=route_metadata,
            )
        return fc1_c, route_metadata, wgrad_stash

    def close(self) -> None:
        self._buffer = None
        self._closed = True


__all__ = [
    "Mxfp8ForwardStash",
    "Mxfp8StashPlan",
]
