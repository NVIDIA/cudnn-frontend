# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-process full-expert MoE reference (the numerics oracle).

Holds ALL experts' weights and processes the full global batch in one
process — no distribution, plain autograd. Op order deliberately mirrors the
EP layer (sort-by-expert -> grouped FFN -> unsort -> fp32 weighted top-k
sum) so forward parity vs the EP path is tight: per-token FFN rows are
independent, and the combine sums the same values in the same order.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .experts import grouped_expert_ffn


class ReferenceMoE(nn.Module):
    def __init__(self, w13: torch.Tensor, w2: torch.Tensor):
        """``w13`` ``[num_experts, 2*intermediate, hidden]``,
        ``w2`` ``[num_experts, hidden, intermediate]`` — the FULL expert set.
        """
        super().__init__()
        if w13.dim() != 3 or w2.dim() != 3 or w13.shape[0] != w2.shape[0]:
            raise ValueError("w13/w2 must be [num_experts, ...] with matching expert dim")
        if w13.shape[1] != 2 * w2.shape[2] or w13.shape[2] != w2.shape[1]:
            raise ValueError(
                f"inconsistent shapes: w13 {tuple(w13.shape)}, w2 {tuple(w2.shape)}"
            )
        self.w13 = w13 if isinstance(w13, nn.Parameter) else nn.Parameter(w13)
        self.w2 = w2 if isinstance(w2, nn.Parameter) else nn.Parameter(w2)

    @property
    def num_experts(self) -> int:
        return self.w13.shape[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, top_k = topk_ids.shape
        flat_expert = topk_ids.reshape(-1).long()

        sort_idx = torch.argsort(flat_expert, stable=True)
        inv_sort_idx = torch.argsort(sort_idx)
        copy_token_idx = torch.div(sort_idx, top_k, rounding_mode="floor")
        tokens_per_expert = [
            int(c)
            for c in torch.bincount(flat_expert, minlength=self.num_experts).tolist()
        ]

        x_grouped = hidden_states[copy_token_idx]  # [T*K, hidden]
        y_grouped = self._ffn(x_grouped, tokens_per_expert)
        y_flat = y_grouped[inv_sort_idx]

        y_tk = y_flat.view(num_tokens, top_k, -1)
        # Combine in at least fp32 (mirrors the EP layer) but never downcast:
        # fp64 stays fp64 so the oracle is exact under gradcheck/naive tests.
        acc = torch.promote_types(hidden_states.dtype, torch.float32)
        out = (y_tk.to(acc) * topk_weights.to(acc).unsqueeze(-1)).sum(dim=1)
        return out.to(hidden_states.dtype)

    def _ffn(self, x_grouped: torch.Tensor, tokens_per_expert) -> torch.Tensor:
        """Expert-compute hook; quantized variants (reference_fp4) override this."""
        return grouped_expert_ffn(x_grouped, tokens_per_expert, self.w13, self.w2)
