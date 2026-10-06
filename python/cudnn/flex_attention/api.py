# SPDX-License-Identifier: BSD-3-Clause
"""Public Flex Attention API."""

from __future__ import annotations

import torch

from cudnn.flex_attention.autograd import FlexAttnFunc
from cudnn.flex_attention.execution import FlexAttentionBwd, FlexAttentionFwd
from cudnn.flex_attention.plan.builder import create_mask_plan
from cudnn.flex_attention.plan.mask_plan import MaskPlan
from cudnn.flex_attention.plan.validation import validate_call_options


def _validate_plan(mask_plan: MaskPlan) -> None:
    if not isinstance(mask_plan, MaskPlan):
        raise TypeError("mask_plan must be returned by create_mask_plan")


def flex_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask_plan: MaskPlan,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
    return_max_logit: bool = False,
):
    """Run fixed-length BSHD or variable-length THD interval-mask attention.

    Args:
        q: CUDA FP16/BF16 tensor shaped ``[B, Sq, Hq, Dqk]`` for fixed
            attention or ``[total_q, Hq, Dqk]`` for variable attention.
        k: CUDA tensor shaped ``[B, Sk, Hkv, Dqk]`` or
            ``[total_k, Hkv, Dqk]``.
        v: CUDA tensor shaped ``[B, Sk, Hkv, Dv]`` or
            ``[total_k, Hkv, Dv]``.
        mask_plan: A compatible plan from :func:`create_mask_plan`. Supplying
            cumulative sequence lengths when creating the plan selects the
            variable-length path automatically.
        softmax_scale: Score scale, or ``None`` for ``1 / sqrt(Dqk)``.
        deterministic: Select the deterministic backward path when supported.
        return_lse: Return FP32 log-sum-exp alongside the output.
        return_max_logit: Append the non-differentiable FP32 ``[Hq]`` maximum
            scaled logit over all batches and visible query/key positions.
            Requires non-negative softmax_scale; fully masked heads return -inf.

    Returns:
        ``out`` or ``(out, lse)`` when ``return_lse=True``. Fixed output/LSE
        shapes are ``[B, Sq, Hq, Dv]`` and ``[B, Hq, Sq]``; variable shapes
        are ``[total_q, Hq, Dv]`` and ``[Hq, total_q]``. With
        ``return_max_logit=True``, returns ``(out, max_logit)`` or
        ``(out, lse, max_logit)`` when both statistics are requested.
    """

    _validate_plan(mask_plan)
    validate_call_options(
        softmax_scale=softmax_scale,
        deterministic=deterministic,
        return_lse=return_lse,
        return_max_logit=return_max_logit,
    )
    mask_plan._validate_runtime(q, k, v)
    result = FlexAttnFunc.apply(
        q,
        k,
        v,
        mask_plan,
        softmax_scale,
        deterministic,
        return_lse,
        return_max_logit,
    )
    if return_lse and return_max_logit:
        return result
    if return_lse:
        return result[0], result[1]
    if return_max_logit:
        return result[0], result[2]
    return result[0]


__all__ = [
    "FlexAttentionBwd",
    "FlexAttentionFwd",
    "MaskPlan",
    "create_mask_plan",
    "flex_attn_func",
]
