# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram math and native benchmark controls.

Gate semantics follow DeepSeek-V4.1-Flash inference/model.py, revision
dba1be0a40aa45a94ad051997016db3960a90277. These benchmark functions and
autograd reduction controls are authored here using PyTorch operations.
Opaque reductions preserve eager FP32 reduction ordering while permitting
Inductor to compile surrounding math. The FP64-dot control is separate.
"""

import torch
import torch.nn.functional as F

SOURCE_EPS = 1e-20


def gate_reference(x, key, value, weight, mask, dtype=torch.float32, *, eps):
    h, k, v, w = [t.to(dtype) for t in (x, key, value, weight)]
    scale = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
    dot = (h * w * k).sum(-1) * scale * x.shape[-1] ** (-0.5)
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-06).sqrt(), dot)).masked_fill(~mask[:, None], 0)
    return (h + gate[:, :, None] * v[:, None, :]).to(x.dtype)


@torch.library.custom_op("cudnn_bench_engram::sum_last", mutates_args=())
def source_sum_last(x: torch.Tensor) -> torch.Tensor:
    return x.sum(-1)


@source_sum_last.register_fake
def _sum_fake(x):
    return x.new_empty(x.shape[:-1])


@torch.library.custom_op("cudnn_bench_engram::mean_last", mutates_args=())
def source_mean_last(x: torch.Tensor) -> torch.Tensor:
    return x.mean(-1)


@source_mean_last.register_fake
def _mean_fake(x):
    return x.new_empty(x.shape[:-1])


def _setup(ctx, inputs, output):
    ctx.input_shape = inputs[0].shape


def _sum_backward(ctx, gradient):
    return gradient.unsqueeze(-1).expand(ctx.input_shape)


def _mean_backward(ctx, gradient):
    return gradient.unsqueeze(-1).expand(ctx.input_shape) / ctx.input_shape[-1]


source_sum_last.register_autograd(_sum_backward, setup_context=_setup)
source_mean_last.register_autograd(_mean_backward, setup_context=_setup)


def source_reductions(x, key, value, weight, mask, *, eps):
    h, k, v, w = [t.float() for t in (x, key, value, weight)]
    scale = torch.rsqrt(source_mean_last(h.square()) + eps) * torch.rsqrt(source_mean_last(k.square()) + eps)
    dot = source_sum_last(h * w * k) * scale * x.shape[-1] ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)).masked_fill(~mask[:, None], 0)
    return (h + gate[:, :, None] * v[:, None, :]).to(x.dtype)


def full64_dot(x, key, value, weight, mask, *, eps):
    h, k, v, w = [t.float() for t in (x, key, value, weight)]
    scale = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
    dot = (h.double() * w.double() * k.double()).sum(-1).float() * scale * x.shape[-1] ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)).masked_fill(~mask[:, None], 0)
    return (h + gate[:, :, None] * v[:, None, :]).to(x.dtype)


def source_dot_reduction(x, key, value, weight, mask, *, eps):
    h, k, v, w = [t.float() for t in (x, key, value, weight)]
    scale = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
    dot = source_sum_last(h * w * k) * scale * x.shape[-1] ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)).masked_fill(~mask[:, None], 0)
    return (h + gate[:, :, None] * v[:, None, :]).to(x.dtype)


def reference(x, embedding, weight, qw, kw, mask):
    n, hc, h = x.shape
    latent = F.linear(embedding, weight)
    return gate_reference(x, latent[:, : hc * h].view(n, hc, h), latent[:, hc * h :], qw * kw, mask, eps=SOURCE_EPS)


def source_reduction_consumer(x, embedding, weight, qw, kw, mask):
    n, hc, h = x.shape
    latent = F.linear(embedding, weight)
    return source_reductions(x, latent[:, : hc * h].view(n, hc, h), latent[:, hc * h :], qw * kw, mask, eps=SOURCE_EPS)


def full64_consumer(x, embedding, weight, qw, kw, mask):
    n, hc, h = x.shape
    latent = F.linear(embedding, weight)
    return full64_dot(x, latent[:, : hc * h].view(n, hc, h), latent[:, hc * h :], qw * kw, mask, eps=SOURCE_EPS)


def source_dot_consumer(x, embedding, weight, qw, kw, mask):
    n, hc, h = x.shape
    latent = F.linear(embedding, weight)
    return source_dot_reduction(x, latent[:, : hc * h].view(n, hc, h), latent[:, hc * h :], qw * kw, mask, eps=SOURCE_EPS)
