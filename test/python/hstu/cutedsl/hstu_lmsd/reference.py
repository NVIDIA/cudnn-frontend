# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unfused PyTorch reference operations for HSTU LMSD tests."""

import torch
import torch.nn.functional as F


def layer_norm_stats(x, eps):
    x_float = x.float()
    mean = x_float.mean(dim=1)
    variance = (x_float.square().mean(dim=1) - mean.square()).clamp_min(0.0)
    return mean, torch.rsqrt(variance + eps)


def hstu_lmsd_forward_reference(
    x,
    u,
    weight,
    bias,
    mask,
    dropout_ratio,
    eps,
    *,
    apply_u_silu=True,
    concat_u=True,
    concat_x=True,
):
    x_float = x.float()
    u_float = u.float()
    mean, rstd = layer_norm_stats(x, eps)
    normalized = (x_float - mean[:, None]) * rstd[:, None]
    layer_norm = normalized * weight.float() + bias.float()
    activated_u = F.silu(u_float) if apply_u_silu else u_float

    outputs = []
    if concat_u:
        outputs.append((activated_u, 4))
    if concat_x:
        outputs.append((x_float, 2))
    outputs.append((layer_norm * activated_u, 1))

    if mask is not None:
        mask_int = mask.to(torch.int32)
        scale = 1.0 / (1.0 - dropout_ratio)
        zero = torch.zeros((), device=x.device)
        outputs = [(torch.where((mask_int & bit) != 0, value * scale, zero), bit) for value, bit in outputs]

    return torch.cat([value for value, _ in outputs], dim=1), mean, rstd


def hstu_lmsd_backward_reference(
    dy,
    x,
    u,
    weight,
    bias,
    mask,
    dropout_ratio,
    eps,
    *,
    apply_u_silu=True,
    concat_u=True,
    concat_x=True,
    compute_dweight=True,
):
    d = x.shape[1]
    x_float = x.float()
    u_float = u.float()
    weight_float = weight.float()
    bias_float = bias.float()
    dy_parts = [part.float() for part in dy.split(d, dim=1)]
    part = 0
    dy_u = dy_parts[part] if concat_u else torch.zeros_like(x_float)
    part += int(concat_u)
    dy_x = dy_parts[part] if concat_x else torch.zeros_like(x_float)
    part += int(concat_x)
    dy_lmsd = dy_parts[part]

    if mask is not None:
        mask_int = mask.to(torch.int32)
        scale = 1.0 / (1.0 - dropout_ratio)
        zero = torch.zeros((), device=x.device)
        direct_du = torch.where((mask_int & 4) != 0, dy_u * scale, zero)
        direct_dx = torch.where((mask_int & 2) != 0, dy_x * scale, zero)
        fused_dy = torch.where((mask_int & 1) != 0, dy_lmsd * scale, zero)
    else:
        direct_du = dy_u
        direct_dx = dy_x
        fused_dy = dy_lmsd

    mean, rstd = layer_norm_stats(x, eps)
    x_hat = (x_float - mean[:, None]) * rstd[:, None]
    layer_norm = x_hat * weight_float + bias_float
    if apply_u_silu:
        sigmoid = torch.sigmoid(u_float)
        activated_u = u_float * sigmoid
        activated_u_grad = sigmoid + activated_u * (1.0 - sigmoid)
    else:
        activated_u = u_float
        activated_u_grad = torch.ones_like(u_float)

    layer_norm_grad = fused_dy * activated_u
    dweight = torch.sum(layer_norm_grad * x_hat, dim=0) if compute_dweight else None
    dbias = torch.sum(layer_norm_grad, dim=0)
    weighted_grad = layer_norm_grad * weight_float
    mean_weighted_grad = torch.mean(weighted_grad, dim=1, keepdim=True)
    mean_x_hat_weighted_grad = torch.mean(x_hat * weighted_grad, dim=1, keepdim=True)
    dx = direct_dx + (weighted_grad - mean_weighted_grad - x_hat * mean_x_hat_weighted_grad) * rstd[:, None]
    du = (fused_dy * layer_norm + direct_du) * activated_u_grad
    return dx, du, dweight, dbias
