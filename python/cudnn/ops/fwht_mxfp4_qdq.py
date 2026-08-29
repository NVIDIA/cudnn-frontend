# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-facing normalized H128 plus MXFP4 quantize/dequantize operation."""

import torch
from torch import Tensor

_ROW_WIDTH = 128
_REQUIRED_ALIGNMENT = 32


def _validate_input(input_tensor: Tensor) -> None:
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"input_tensor must be a torch.Tensor, got {type(input_tensor).__name__}")
    if not input_tensor.is_cuda:
        raise ValueError(f"input_tensor must be on CUDA, got device={input_tensor.device}")
    if input_tensor.dtype != torch.bfloat16:
        raise TypeError(f"input_tensor must have dtype torch.bfloat16, got {input_tensor.dtype}")
    if input_tensor.ndim < 1 or input_tensor.shape[-1] != _ROW_WIDTH:
        raise ValueError("input_tensor must have shape [..., 128], " f"got {tuple(input_tensor.shape)}")
    if not input_tensor.is_contiguous():
        raise ValueError("input_tensor must be contiguous; no implicit layout conversion is performed")
    if torch.is_grad_enabled() and input_tensor.requires_grad:
        raise NotImplementedError("fwht_mxfp4_qdq is inference-only and does not implement autograd")


@torch.library.custom_op(
    "cudnn::fwht_mxfp4_qdq_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _fwht_mxfp4_qdq_primitive(input_tensor: Tensor) -> Tensor:
    address_remainder = input_tensor.data_ptr() % _REQUIRED_ALIGNMENT
    if address_remainder != 0:
        raise ValueError(f"input_tensor must be {_REQUIRED_ALIGNMENT}-byte aligned, " f"got data_ptr mod {_REQUIRED_ALIGNMENT} = {address_remainder}")
    output_tensor = torch.empty_like(input_tensor)
    if input_tensor.numel() == 0:
        return output_tensor

    from ._fwht_mxfp4_qdq_cutedsl import run_fwht_mxfp4_qdq

    run_fwht_mxfp4_qdq(
        input_tensor.view(-1, _ROW_WIDTH),
        output_tensor.view(-1, _ROW_WIDTH),
    )
    return output_tensor


@torch.library.register_fake("cudnn::fwht_mxfp4_qdq_primitive")
def _fwht_mxfp4_qdq_fake(input_tensor: Tensor) -> Tensor:
    return torch.empty_like(input_tensor)


def fwht_mxfp4_qdq(input_tensor: Tensor) -> Tensor:
    r"""Apply normalized H128 followed by group-32 MXFP4 QDQ.

    The operation follows the DeepSeek-V4 indexer inference recipe:

    1. Compute the seven-stage Sylvester-order H128 transform in FP32.
    2. Scale by ``1 / sqrt(128)`` and round to BF16.
    3. Independently quantize each contiguous group of 32 values to finite
       E2M1 with a power-of-two UE8M0 scale, then dequantize to BF16.

    Args:
        input_tensor: Contiguous, 32-byte-aligned CUDA BF16 tensor with shape
            ``[..., 128]``. Leading dimensions are preserved.

    Returns:
        A new contiguous CUDA BF16 tensor with the same shape as
        ``input_tensor``.

    This experimental operation is inference-only. It never inserts a layout
    conversion or calls a framework fallback.
    """

    _validate_input(input_tensor)
    return torch.ops.cudnn.fwht_mxfp4_qdq_primitive(input_tensor)


__all__ = ["fwht_mxfp4_qdq"]
