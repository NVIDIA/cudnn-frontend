# SPDX-License-Identifier: BSD-3-Clause
"""Eager custom autograd bindings for packed-mask attention."""

from __future__ import annotations

import torch

from cudnn.flex_attention.execution import _flex_attention_backward, _flex_attention_forward
from cudnn.flex_attention.plan.mask_plan import MaskPlan


class FlexAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_plan: MaskPlan,
        softmax_scale: float | None,
        deterministic: bool,
        return_lse: bool,
    ):
        result = _flex_attention_forward(
            q,
            k,
            v,
            mask_plan=mask_plan,
            softmax_scale=softmax_scale,
            return_lse=return_lse,
        )
        out, lse = result
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.mask_plan = mask_plan
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(out)
        result = _flex_attention_backward(
            q,
            k,
            v,
            out,
            dout,
            lse,
            mask_plan=ctx.mask_plan,
            softmax_scale=ctx.softmax_scale,
            deterministic=ctx.deterministic,
            dlse_tensor=dlse if ctx.return_lse else None,
        )
        dq, dk, dv = result
        return dq, dk, dv, None, None, None, None


__all__ = ["FlexAttnFunc"]
