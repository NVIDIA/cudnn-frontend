# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autograd boundary for the optional Cake causal-convolution backend."""

import torch

from .api import CausalConv1dPlan


class _CausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cu_seqlens):
        batch, tokens, channels = x.shape
        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, (batch + 1) * tokens, tokens, dtype=torch.int32, device=x.device)
        plan = CausalConv1dPlan(
            batch * tokens, channels, cu_seqlens.numel() - 1, device=x.device, weight_dtype=weight.dtype, bias_dtype=None if bias is None else bias.dtype
        )
        out = torch.empty_like(x)
        scratch = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=x.device)
        plan.forward(x.view(plan.shape), weight, cu_seqlens, out.view(plan.shape), scratch, bias=bias)
        ctx.plan, ctx.has_bias = plan, bias is not None
        ctx.save_for_backward(x, weight, cu_seqlens, *(() if bias is None else (bias,)))
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        x, weight, cu_seqlens, *saved_bias = ctx.saved_tensors
        if not grad_out.is_contiguous():
            raise ValueError("Cake causal_conv1d backward needs a native contiguous [B, T, D] gradient; no layout copy is performed")
        bias = saved_bias[0] if ctx.has_bias else None
        plan = ctx.plan
        grad_x, grad_weight = torch.empty_like(x), torch.empty_like(weight)
        grad_bias = None if bias is None else torch.empty_like(bias)
        scratch = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=x.device)
        plan.backward(
            grad_out.view(plan.shape), x.view(plan.shape), weight, cu_seqlens, grad_x.view(plan.shape), grad_weight, scratch, bias=bias, grad_bias=grad_bias
        )
        return grad_x, grad_weight, grad_bias, None


def causal_conv1d(x, weight, bias=None, cu_seqlens=None):
    """Run native BTD storage through the existing BDT semantic API."""
    if x.ndim != 3 or x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise ValueError("Cake causal_conv1d needs BF16 CUDA x[B, D, T]")
    x_btd = x.transpose(1, 2)
    if not x_btd.is_contiguous():
        raise ValueError("Cake causal_conv1d needs channel-last physical storage; no layout copy is performed")
    if x_btd.shape[0] <= 0 or x_btd.shape[1] <= 0:
        raise ValueError("Cake causal_conv1d requires positive batch and token counts")
    return _CausalConv1d.apply(x_btd, weight, bias, cu_seqlens).transpose(1, 2)
