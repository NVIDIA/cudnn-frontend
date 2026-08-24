# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Explicit grad-output re-dispatch for semantic router gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist

from ..._contracts import ValidatedBackwardRequest


@dataclass(frozen=True)
class _DispatchPlan:
    send_token: torch.Tensor
    send_slot: torch.Tensor
    send_local_expert: torch.Tensor
    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]


@dataclass(frozen=True)
class RedispatchedGradOutput:
    """Route rows in the compact public ``route_metadata`` order."""

    grad_output: torch.Tensor


class Mxfp8BackwardRedispatch:
    """Recreate the identical-route grad-output exchange for dprob."""

    def __init__(self, request: ValidatedBackwardRequest) -> None:
        self.request = request
        self.config = request.config

    def _collective_device(self, device: torch.device) -> torch.device:
        if (
            device.type != "cpu"
            and self.config.ep_size > 1
            and dist.get_backend(self.config.ep_group) == "gloo"
        ):
            return torch.device("cpu")
        return device

    def _exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor:
        if self.config.ep_size == 1:
            return send_counts.clone()
        staged = send_counts.to(self._collective_device(send_counts.device))
        recv_counts = torch.empty_like(staged)
        dist.all_to_all_single(
            recv_counts,
            staged,
            group=self.config.ep_group,
        )
        return recv_counts.to(send_counts.device)

    def _all_to_all(
        self,
        send: torch.Tensor,
        send_counts: Sequence[int],
        recv_counts: Sequence[int],
    ) -> torch.Tensor:
        if self.config.ep_size == 1:
            return send.clone()
        comm_device = self._collective_device(send.device)
        staged = send.contiguous().to(comm_device)
        recv = torch.empty(
            (sum(recv_counts), *send.shape[1:]),
            dtype=send.dtype,
            device=comm_device,
        )
        dist.all_to_all_single(
            recv,
            staged,
            output_split_sizes=list(recv_counts),
            input_split_sizes=list(send_counts),
            group=self.config.ep_group,
        )
        return recv.to(send.device)

    def _plan(self) -> _DispatchPlan:
        config = self.config
        flat_expert = self.request.topk_idx.reshape(-1).to(torch.int64)
        valid = flat_expert != -1
        token = torch.arange(
            self.request.token_count,
            dtype=torch.int64,
            device=self.request.device,
        ).repeat_interleave(config.top_k)
        slot = torch.arange(
            config.top_k,
            dtype=torch.int64,
            device=self.request.device,
        ).repeat(self.request.token_count)
        expert = flat_expert[valid]
        destination = torch.div(
            expert,
            config.experts_per_rank,
            rounding_mode="floor",
        )
        order = torch.argsort(destination, stable=True)
        destination = destination.index_select(0, order)
        send_counts_tensor = torch.bincount(
            destination,
            minlength=config.ep_size,
        ).to(torch.int64)
        recv_counts_tensor = self._exchange_counts(send_counts_tensor)
        return _DispatchPlan(
            send_token=token[valid].index_select(0, order),
            send_slot=slot[valid].index_select(0, order),
            send_local_expert=expert.index_select(0, order).remainder(
                config.experts_per_rank
            ),
            send_counts=tuple(
                int(value) for value in send_counts_tensor.cpu().tolist()
            ),
            recv_counts=tuple(
                int(value) for value in recv_counts_tensor.cpu().tolist()
            ),
        )

    def run(self) -> RedispatchedGradOutput:
        config = self.config
        plan = self._plan()

        recv_token = self._all_to_all(
            plan.send_token,
            plan.send_counts,
            plan.recv_counts,
        )
        recv_slot = self._all_to_all(
            plan.send_slot,
            plan.send_counts,
            plan.recv_counts,
        )
        recv_expert = self._all_to_all(
            plan.send_local_expert,
            plan.send_counts,
            plan.recv_counts,
        )
        recv_grad_output = self._all_to_all(
            self.request.grad_output.index_select(0, plan.send_token).float(),
            plan.send_counts,
            plan.recv_counts,
        )
        recv_rank = torch.repeat_interleave(
            torch.arange(
                config.ep_size,
                dtype=torch.int64,
                device=self.request.device,
            ),
            torch.tensor(
                plan.recv_counts,
                dtype=torch.int64,
                device=self.request.device,
            ),
            output_size=self.request.local_routes,
        )
        if recv_grad_output.shape[0] != self.request.local_routes:
            raise ValueError(
                "route_metadata row count does not match the routes received "
                "from the re-supplied topk_idx"
            )

        key = (
            (
                (
                    recv_expert * config.ep_size
                    + recv_rank
                )
                * int(config.max_tokens_per_rank)
                + recv_token
            )
            * config.top_k
            + recv_slot
        )
        compact_order = torch.argsort(key, stable=True)
        actual_metadata = torch.stack(
            (
                recv_expert.index_select(0, compact_order),
                recv_rank.index_select(0, compact_order),
                recv_token.index_select(0, compact_order),
                recv_slot.index_select(0, compact_order),
            ),
            dim=1,
        ).to(torch.int32)
        if not torch.equal(actual_metadata, self.request.route_metadata):
            raise ValueError(
                "route_metadata does not match the re-supplied forward routes"
            )

        return RedispatchedGradOutput(
            grad_output=recv_grad_output.index_select(0, compact_order),
        )


__all__ = [
    "Mxfp8BackwardRedispatch",
    "RedispatchedGradOutput",
]
