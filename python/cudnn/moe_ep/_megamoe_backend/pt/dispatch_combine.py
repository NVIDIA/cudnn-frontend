# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Differentiable dispatch / combine around the token all-to-all.

Forward data path (mirrors flashinfer moe_ep's split mode):

    dispatch:  x[T,H] --replicate by top-k & sort by dest expert-->
               a2a --> arrivals[R,H] --group by local expert--> x_grouped
    combine:   y_grouped --ungroup--> a2a (reversed splits) -->
               unsort --> [T,K,H] --fp32 top-k weighted sum--> out[T,H]

Backward comes out as the exact adjoint by construction:

- gather ``x[copy_token_idx]`` <-> scatter-add over the K copies of a token
- the all-to-all's backward is the swapped-splits all-to-all
  (see comm/torch_dist.py)
- the fp32 weighted sum routes grads to both ``y`` and ``topk_weights``

so dispatch.backward is combine-shaped and combine.backward is
dispatch-shaped — the property the future megakernel bprop must reproduce.
"""

from __future__ import annotations

import torch

from .comm.base import TokenComm
from .routing import RoutingPlan


def dispatch(
    x: torch.Tensor, plan: RoutingPlan, comm: TokenComm
) -> torch.Tensor:
    """Route ``x [T, hidden]`` to expert owners.

    Returns ``x_grouped [num_recv_tokens, hidden]`` on this rank, grouped by
    local expert (layout consumed by :func:`experts.grouped_expert_ffn`).
    """
    # Replicate each token for its top-k destinations, already in
    # dest-expert-sorted order. Differentiable: backward scatter-adds the K
    # copies' grads back into each source token row.
    x_sorted = x[plan.copy_token_idx]  # [T*K, hidden]
    x_recv = comm.all_to_all(x_sorted, plan.output_splits, plan.input_splits)
    # Arrival order -> grouped-by-local-expert order.
    return x_recv[plan.recv_sort_idx]


def combine(
    y_grouped: torch.Tensor,
    topk_weights: torch.Tensor,
    plan: RoutingPlan,
    out_dtype: torch.dtype,
    comm: TokenComm,
) -> torch.Tensor:
    """Return expert outputs to source ranks and reduce over top-k.

    ``y_grouped`` is ``[num_recv_tokens, hidden]`` in grouped-by-local-expert
    order; ``topk_weights`` is ``[T, K]`` (fp32 recommended; grads flow to it
    through the reweighting). Returns ``out [T, hidden]`` in ``out_dtype``.
    """
    # Grouped order -> arrival order, then the reverse exchange (splits
    # swapped relative to dispatch).
    y_recv = y_grouped[plan.inv_recv_sort_idx]
    y_sorted = comm.all_to_all(y_recv, plan.input_splits, plan.output_splits)
    # Sorted-copy order -> original (token-major) copy order.
    y_flat = y_sorted[plan.inv_sort_idx]  # [T*K, hidden]

    num_tokens, top_k = topk_weights.shape
    y_tk = y_flat.view(num_tokens, top_k, -1)
    out = (y_tk.float() * topk_weights.float().unsqueeze(-1)).sum(dim=1)
    return out.to(out_dtype)
