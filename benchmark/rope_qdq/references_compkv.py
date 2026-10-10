# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored Torch math and GPU dequantization glue for the pinned group16 E4M3-scale FP4 consumer."""

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


def qdq_values(x, snap=False):
    groups = x.float().reshape(*x.shape[:-1], 32, 16)
    ratio = groups.abs().amax(dim=-1).clamp_min(6.0 * 2**-9) / 6.0
    # CUDA source conversion is satfinite; plain Torch FP8 cast can produce NaN.
    scale = ratio.clamp_max(448.0).to(torch.float8_e4m3fn).float()
    normalized = (groups / scale[..., None]).clamp(-6.0, 6.0)
    if snap:
        normalized = normalized.to(torch.float16).float()
    return (e2m1_roundtrip(normalized) * scale[..., None]).to(torch.bfloat16).reshape(x.shape)


def quant_inplace(x):
    x.copy_(qdq_values(x))
    return x


def rotated_values(x, cache, positions, keep_bf16_boundary=True):
    # x is the native contiguous KV storage viewed as [B*S,1,512].
    cosine, sine = cache[positions.long()].split(32, dim=-1)
    cosine, sine = cosine[:, None, :], sine[:, None, :]
    a, b = x[..., -64:].float().unflatten(-1, (32, 2)).unbind(-1)
    tail = torch.stack((a * cosine - b * sine, b * cosine + a * sine), dim=-1).flatten(-2)
    if keep_bf16_boundary:
        tail = tail.to(torch.bfloat16).float()
    return torch.cat((x[..., :448].float(), tail), dim=-1)


def fused_inplace(x, cache, positions):
    x.copy_(qdq_values(rotated_values(x, cache, positions)))
    return x


def omitted_rounding_control(x, cache, positions):
    return qdq_values(rotated_fma_values(x, cache, positions, False))


def unpack_nvfp4_inplace(x, packed, scales):
    """GPU unpack for linear E4M3 scales and global scale exactly one."""
    packed = packed.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).to(torch.int32)
    magnitude = codes & 7
    exponent = torch.where(magnitude < 2, 126, torch.where(magnitude < 4, 127, torch.where(magnitude < 6, 128, 129))).to(torch.int32)
    fraction = torch.where(magnitude < 2, 0, (magnitude & 1) << 22)
    magnitude_bits = torch.where(magnitude == 0, 0, (exponent << 23) | fraction)
    values = (magnitude_bits | ((codes & 8) << 28)).view(torch.float32)
    decoded = values.reshape(-1, 32, 16) * scales.view(torch.float8_e4m3fn).float().reshape(-1, 32)[..., None]
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
    return torch.cat((x[..., :448].float(), tail), dim=-1)


def fused_fma_inplace(x, cache, positions):
    x.copy_(qdq_values(rotated_fma_values(x, cache, positions)))
    return x


def fused_snap_inplace(x, cache, positions):
    x.copy_(qdq_values(rotated_values(x, cache, positions), snap=True))
    return x


def fused_fma_snap_inplace(x, cache, positions):
    x.copy_(qdq_values(rotated_fma_values(x, cache, positions), snap=True))
    return x


def quant_snap_inplace(x):
    x.copy_(qdq_values(x, snap=True))
    return x
