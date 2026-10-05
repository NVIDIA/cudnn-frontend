# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored Torch math and GPU dequantization glue for the pinned FP8 consumer."""

import torch


def qdq_values(x):
    groups = x.float().reshape(*x.shape[:-1], 16, 32)
    ratio = groups.abs().amax(dim=-1).clamp_min(1e-4) * (1.0 / 448.0)
    bits = ratio.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent << 23).view(torch.float32)
    normalized = (groups / scale[..., None]).clamp(-448, 448)
    quantized = normalized.to(torch.float8_e4m3fn).float()
    return (quantized * scale[..., None]).to(torch.bfloat16).reshape(x.shape)


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


def unpack_mxfp8_inplace(x, packed, scales):
    """GPU-compilable linear-scale dequantization; all its cost belongs to baseline.

    Preserve UE8M0 code zero (2^-127) and code 255 (NaN). Native quantized
    values already have E4M3 dtype, so its exact FP32 cast keeps signed zero.
    """
    codes = scales.view(torch.uint8).reshape(-1, 16).to(torch.int32)
    bits = torch.where(codes == 0, 0x00400000, torch.where(codes == 255, 0x7FC00000, codes << 23)).to(torch.int32)
    decoded = packed.float().reshape(-1, 16, 32) * bits.view(torch.float32)[..., None]
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
