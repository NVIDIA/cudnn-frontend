# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time weight packing for the SM100 Conv3D kernels."""

import torch
import torch.nn.functional as F


def pack_conv3d_weight_sm100(weight: torch.Tensor) -> torch.Tensor:
    """Pack a contiguous OITRS Conv3D weight as KTRSC with K64 padding.

    This is a one-time model preparation step, not part of kernel execution.
    Supports 160->160, 160->320, 320->320, 320->640, and 640->640 channels.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
    if weight.ndim != 5:
        raise ValueError(f"weight must have shape [K, C, T, R, S], got {tuple(weight.shape)}")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"weight must be torch.bfloat16, got {weight.dtype}")

    output_channels, input_channels, filter_t, filter_h, filter_w = weight.shape
    if (filter_t, filter_h, filter_w) != (3, 3, 3):
        raise ValueError(f"filter dimensions must be 3x3x3, got {(filter_t, filter_h, filter_w)}")
    if (input_channels, output_channels) not in {
        (160, 160),
        (160, 320),
        (320, 320),
        (320, 640),
        (640, 640),
    }:
        raise ValueError("unsupported channel pair; expected one of " "(160,160), (160,320), (320,320), (320,640), or (640,640)")

    packed = weight.detach().permute(0, 2, 3, 4, 1).contiguous()
    return F.pad(packed, (0, (-input_channels) % 64))


def pack_causal_conv3d_weight_sm100(weight: torch.Tensor) -> torch.Tensor:
    """Pack weights for ``CausalConv3dWithCacheSm100``; currently only BF16 [160,12,3,3,3].

    Returns [160,448], with each filter position padded from C12 to C16 and
    a trailing zero filter position. Pack once and reuse during inference.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (160, 12, 3, 3, 3):
        raise ValueError("weight must be BF16 with shape [160,12,3,3,3]")
    packed = torch.zeros((160, 28, 16), dtype=weight.dtype, device=weight.device)
    packed[:, :27, :12].copy_(weight.detach().permute(0, 2, 3, 4, 1).reshape(160, 27, 12))
    return packed.reshape(160, 448)


__all__ = ["pack_causal_conv3d_weight_sm100", "pack_conv3d_weight_sm100"]
