# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoEEpTrainingLayer — transparent EP MoE with autograd.

The training analogue of flashinfer's ``MoEEpSplitLayer``:
dispatch -> grouped expert FFN -> combine, all differentiable.

Each rank holds only its local experts' weights (``w13`` / ``w2`` in the
canonical flashinfer layout, sliced to ``num_local_experts``). Inputs are
per-rank token shards; ``forward`` returns the combined MoE output for this
rank's tokens and autograd produces:

- ``hidden_states.grad``   (dX, flows back through combine->FFN->dispatch)
- ``w13.grad`` / ``w2.grad`` (local expert wgrads, already summed over every
  rank's tokens that routed here — no extra reduction needed)
- ``topk_weights.grad``    (router gradient, via the combine reweighting)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .comm import TokenComm, create_comm
from .config import EpConfig
from .dispatch_combine import combine, dispatch
from .experts import grouped_expert_ffn
from .routing import RoutingPlan, build_routing_plan


class MoEEpTrainingLayer(nn.Module):
    def __init__(
        self,
        cfg: EpConfig,
        w13: torch.Tensor,
        w2: torch.Tensor,
        comm: str = "torch_dist",
    ):
        """``w13`` ``[num_local_experts, 2*intermediate, hidden]``,
        ``w2`` ``[num_local_experts, hidden, intermediate]`` — this rank's
        expert shard. Tensors are wrapped as parameters (not copied) unless
        they already are parameters.
        """
        super().__init__()
        expected_w13 = (cfg.num_local_experts, 2 * cfg.intermediate_size, cfg.hidden_size)
        expected_w2 = (cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size)
        if tuple(w13.shape) != expected_w13:
            raise ValueError(f"w13 shape {tuple(w13.shape)} != {expected_w13}")
        if tuple(w2.shape) != expected_w2:
            raise ValueError(f"w2 shape {tuple(w2.shape)} != {expected_w2}")
        self.cfg = cfg
        self.w13 = w13 if isinstance(w13, nn.Parameter) else nn.Parameter(w13)
        self.w2 = w2 if isinstance(w2, nn.Parameter) else nn.Parameter(w2)
        self.comm: TokenComm = create_comm(comm, group=cfg.process_group)
        self.last_plan: RoutingPlan | None = None  # exposed for tests/debug

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """``hidden_states [T, hidden]``, ``topk_ids [T, K]`` int,
        ``topk_weights [T, K]`` float (fp32 recommended). Returns
        ``[T, hidden]`` in ``hidden_states.dtype``.

        Collective: every EP rank must call forward (and backward) each
        iteration, even with T=0, since dispatch/combine are all-to-alls.
        """
        cfg = self.cfg
        if hidden_states.dim() != 2 or hidden_states.shape[1] != cfg.hidden_size:
            raise ValueError(
                f"hidden_states must be [T, {cfg.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("topk_ids / topk_weights must both be [T, top_k]")

        plan = build_routing_plan(topk_ids, cfg, self.comm)
        self.last_plan = plan

        x_grouped = dispatch(hidden_states, plan, self.comm)
        y_grouped = self._ffn(x_grouped, plan.tokens_per_expert)
        return combine(
            y_grouped, topk_weights, plan, hidden_states.dtype, self.comm
        )

    def _ffn(self, x_grouped: torch.Tensor, tokens_per_expert) -> torch.Tensor:
        """Expert-compute hook; quantized variants (layer_fp4) override this."""
        return grouped_expert_ffn(x_grouped, tokens_per_expert, self.w13, self.w2)
