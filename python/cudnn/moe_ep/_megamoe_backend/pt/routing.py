# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Routing metadata for dropless token-major dispatch.

Given ``topk_ids [T, K]``, each of the ``T*K`` routed copies is sorted by
destination global expert id (stable). Because experts are sharded
contiguously (expert ``e`` -> rank ``e // num_local_experts``), sorting by
expert id also groups copies by destination rank in ascending rank order —
exactly the layout ``all_to_all_single`` wants — AND keeps the sorted stream
per destination rank identical to what that rank's reverse (combine)
exchange returns, so one permutation and its inverse round-trip the data.

Everything here is pure index bookkeeping on non-differentiable integer
tensors; gradients never flow through routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .comm.base import TokenComm
from .config import EpConfig


@dataclass
class RoutingPlan:
    """Send-side and receive-side routing metadata for one forward.

    Saved by the layer for the whole forward/backward round trip (backward
    reuses the same permutations and split sizes in reverse).
    """

    # --- send side (this rank's T*K routed copies) ---
    sort_idx: torch.Tensor        # [T*K] int64; copy indices sorted by dest expert
    inv_sort_idx: torch.Tensor    # [T*K] int64; inverse permutation of sort_idx
    copy_token_idx: torch.Tensor  # [T*K] int64; source token for each sorted copy
    input_splits: List[int]       # rows sent to each rank (len ep_size)
    output_splits: List[int]      # rows received from each rank (len ep_size)

    # --- receive side (rows landed on this rank, arrival order) ---
    recv_sort_idx: torch.Tensor      # [R] int64; arrival rows sorted by local expert
    inv_recv_sort_idx: torch.Tensor  # [R] int64; inverse of recv_sort_idx
    tokens_per_expert: List[int]     # per-local-expert row counts (len num_local_experts)

    @property
    def num_recv_tokens(self) -> int:
        return int(self.recv_sort_idx.numel())


def build_routing_plan(
    topk_ids: torch.Tensor, cfg: EpConfig, comm: TokenComm
) -> RoutingPlan:
    """Compute the full routing plan; one host sync (split sizes)."""
    if topk_ids.dim() != 2 or topk_ids.shape[1] != cfg.top_k:
        raise ValueError(
            f"topk_ids must be [T, {cfg.top_k}], got {tuple(topk_ids.shape)}"
        )
    device = topk_ids.device
    num_local = cfg.num_local_experts
    top_k = cfg.top_k

    flat_expert = topk_ids.reshape(-1).long()  # [T*K]
    if flat_expert.numel() > 0 and (
        int(flat_expert.min()) < 0 or int(flat_expert.max()) >= cfg.num_experts
    ):
        raise ValueError("topk_ids out of range [0, num_experts)")

    # Stable sort by destination expert; contiguous sharding => also grouped
    # by destination rank in ascending order.
    sort_idx = torch.argsort(flat_expert, stable=True)
    inv_sort_idx = torch.argsort(sort_idx)
    sorted_expert = flat_expert[sort_idx]
    copy_token_idx = torch.div(sort_idx, top_k, rounding_mode="floor")

    dest_rank = torch.div(sorted_expert, num_local, rounding_mode="floor")
    send_counts = torch.bincount(dest_rank, minlength=cfg.ep_size)
    recv_counts = comm.exchange_counts(send_counts)

    # Host sync: all_to_all_single needs Python-int split sizes.
    input_splits = [int(c) for c in send_counts.tolist()]
    output_splits = [int(c) for c in recv_counts.tolist()]

    # Receiver side: exchange each copy's global expert id, then group the
    # arrivals by local expert for the grouped FFN.
    recv_expert = comm.all_to_all_no_grad(
        sorted_expert, output_splits, input_splits
    )
    local_expert = recv_expert - cfg.first_local_expert
    if local_expert.numel() > 0 and (
        int(local_expert.min()) < 0 or int(local_expert.max()) >= num_local
    ):
        raise RuntimeError("received a token routed to a non-local expert")

    recv_sort_idx = torch.argsort(local_expert, stable=True)
    inv_recv_sort_idx = torch.argsort(recv_sort_idx)
    tokens_per_expert = [
        int(c) for c in torch.bincount(local_expert, minlength=num_local).tolist()
    ]

    return RoutingPlan(
        sort_idx=sort_idx,
        inv_sort_idx=inv_sort_idx,
        copy_token_idx=copy_token_idx,
        input_splits=input_splits,
        output_splits=output_splits,
        recv_sort_idx=recv_sort_idx.to(device),
        inv_recv_sort_idx=inv_recv_sort_idx.to(device),
        tokens_per_expert=tokens_per_expert,
    )
