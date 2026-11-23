# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time weight packing for the SM100 Conv3D kernels."""

import torch
import torch.nn.functional as F


def _round_weight_tf32(weight: torch.Tensor) -> torch.Tensor:
    bits = weight.detach().contiguous().view(torch.int32)
    rounded = (bits + 4095 + ((bits >> 13) & 1)) & -8192
    return torch.where((bits & 0x7F800000) == 0x7F800000, bits, rounded).view(torch.float32)


def pack_conv3d_weight_sm100(weight: torch.Tensor, allow_tf32: bool = False) -> torch.Tensor:
    """Pack OITRS weights as OTRSC with K64 (BF16) or K32 (TF32) padding.

    This is a one-time model preparation step, not part of kernel execution.
    FP32 requires allow_tf32=True and rounds each significand to nearest/even TF32.
    Supported channel pairs match the valid-convolution APIs.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
    if weight.ndim != 5:
        raise ValueError(f"weight must have shape [K, C, T, R, S], got {tuple(weight.shape)}")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"weight must be BF16 or FP32, got {weight.dtype}")

    output_channels, input_channels, filter_t, filter_h, filter_w = weight.shape
    if (filter_t, filter_h, filter_w) != (3, 3, 3):
        raise ValueError(f"filter dimensions must be 3x3x3, got {(filter_t, filter_h, filter_w)}")
    from ._config import SUPPORTED_CHANNEL_PAIRS

    if (input_channels, output_channels) not in SUPPORTED_CHANNEL_PAIRS:
        raise ValueError(f"unsupported channel pair {(input_channels, output_channels)}")

    if weight.dtype == torch.float32:
        if not allow_tf32:
            raise ValueError("FP32 weight packing requires allow_tf32=True")
        weight = _round_weight_tf32(weight)
    packed = weight.detach().permute(0, 2, 3, 4, 1).contiguous()
    return F.pad(packed, (0, (-input_channels) % (32 if weight.dtype == torch.float32 else 64)))


def pack_causal_conv3d_weight_sm100(weight: torch.Tensor, allow_tf32: bool = False) -> torch.Tensor:
    """Pack weights for ``CausalConv3dWithCacheSm100``; BF16 or FP32 [160,12,3,3,3].

    BF16 returns [160,448], padding each filter position from C12 to C16
    and appending one zero filter position. FP32 requires allow_tf32=True
    and returns rounded [160,3,3,3,32]. Pack once and reuse during inference.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
    if weight.dtype not in (torch.bfloat16, torch.float32) or tuple(weight.shape) != (160, 12, 3, 3, 3):
        raise ValueError("weight must be BF16 or FP32 with shape [160,12,3,3,3]")
    if weight.dtype == torch.float32:
        if not allow_tf32:
            raise ValueError("FP32 weight packing requires allow_tf32=True")
        rounded = _round_weight_tf32(weight)
        return F.pad(rounded.permute(0, 2, 3, 4, 1).contiguous(), (0, 20))
    packed = torch.zeros((160, 28, 16), dtype=weight.dtype, device=weight.device)
    packed[:, :27, :12].copy_(weight.detach().permute(0, 2, 3, 4, 1).reshape(160, 27, 12))
    return packed.reshape(160, 448)


__all__ = ["pack_causal_conv3d_weight_sm100", "pack_conv3d_weight_sm100"]
