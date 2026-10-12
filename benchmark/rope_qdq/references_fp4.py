# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored PyTorch controls for the pinned DeepSeek indexer FP4 QDQ contract.

The mathematical reference is inference/{model,kernel}.py at DeepSeek revision
dba1be0a40aa45a94ad051997016db3960a90277. Native provider kernels are not copied.
These controls are not evidence of a training STE or packed-cache integration.
"""

import torch


def e2m1_roundtrip(x):
    magnitude = x.abs()
    q = torch.zeros_like(x)
    q = torch.where(magnitude > 0.25, 0.5, q)
    q = torch.where(magnitude >= 0.75, 1.0, q)
    q = torch.where(magnitude > 1.25, 1.5, q)
    q = torch.where(magnitude >= 1.75, 2.0, q)
    q = torch.where(magnitude > 2.5, 3.0, q)
    q = torch.where(magnitude >= 3.5, 4.0, q)
    q = torch.where(magnitude > 5.0, 6.0, q)
    return torch.copysign(q, x)


def qdq_values(x):
    groups = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = groups.abs().amax(dim=-1).clamp_min(6 * 2.0**-126)
    ratio = amax * (1.0 / 6.0)
    bits = ratio.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent << 23).view(torch.float32)
    normalized = (groups / scale[..., None]).clamp(-6, 6)
    return (e2m1_roundtrip(normalized) * scale[..., None]).to(torch.bfloat16).reshape(x.shape)


def quant_inplace(x):
    x.copy_(qdq_values(x))
    return x


def rotated_values(x, cache, positions, keep_bf16_boundary=True):
    cosine, sine = cache[positions.long()].split(32, dim=-1)
    cosine, sine = cosine[:, None, :], sine[:, None, :]
    a, b = x[..., -64:].float().unflatten(-1, (32, 2)).unbind(-1)
    tail = torch.stack((a * cosine - b * sine, b * cosine + a * sine), dim=-1).flatten(-2)
    if keep_bf16_boundary:
        tail = tail.to(torch.bfloat16).float()
    return torch.cat((x[..., :64].float(), tail), dim=-1)


def fused_inplace(x, cache, positions):
    x.copy_(qdq_values(rotated_values(x, cache, positions)))
    return x


def omitted_rounding_control(x, cache, positions):
    return qdq_values(rotated_fma_values(x, cache, positions, False))


def unpack_mxfp4_inplace(x, packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).to(torch.int32)
    magnitude = codes & 7
    # Exact IEEE FP32 bit patterns for {0,.5,1,1.5,2,3,4,6}.
    exponent = torch.where(magnitude < 2, 126, torch.where(magnitude < 4, 127, torch.where(magnitude < 6, 128, 129))).to(torch.int32)
    fraction = torch.where(magnitude < 2, 0, (magnitude & 1) << 22)
    magnitude_bits = torch.where(magnitude == 0, 0, (exponent << 23) | fraction)
    values = (magnitude_bits | ((codes & 8) << 28)).view(torch.float32)
    scale_codes = scales.view(torch.uint8).reshape(-1, 4).to(torch.int32)
    scale_bits = torch.where(scale_codes == 0, 0x00400000, torch.where(scale_codes == 255, 0x7FC00000, scale_codes << 23))
    decoded = values.reshape(-1, 4, 32) * scale_bits.view(torch.float32)[..., None]
    x.copy_(decoded.to(torch.bfloat16).reshape(x.shape))
    return x


def rotated_fma_values(x, cache, positions, keep_bf16_boundary=True):
    cosine, sine = cache[positions.long()].split(32, dim=-1)
    cosine, sine = cosine[:, None, :], sine[:, None, :]
    a, b = x[..., -64:].float().unflatten(-1, (32, 2)).unbind(-1)
    # Bit negation retains -0 and forces the separately rounded b*s operand.
    negative_bs = ((b * sine).view(torch.int32) ^ -2147483648).view(torch.float32)
    real = torch.addcmul(negative_bs, a, cosine)
    imag = torch.addcmul(a * sine, b, cosine)
    tail = torch.stack((real, imag), dim=-1).flatten(-2)
    if keep_bf16_boundary:
        tail = tail.to(torch.bfloat16).float()
    return torch.cat((x[..., :64].float(), tail), dim=-1)


def fused_fma_inplace(x, cache, positions):
    x.copy_(qdq_values(rotated_fma_values(x, cache, positions)))
    return x
