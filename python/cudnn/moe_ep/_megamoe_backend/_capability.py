# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Capability policy for the private MegaMoE execution backend.

Inputs reaching this module already satisfy the public :mod:`cudnn.moe_ep`
contract.  These checks only describe the subset that the current MegaMoE
implementation can execute; they must run before runtime initialization,
allocation, compilation, or collectives.
"""

from __future__ import annotations

import torch

from .._contracts import (
    ForwardConfig,
    ValidatedBackwardRequest,
    ValidatedForwardRequest,
)
from .._types import BlockScaledTensor, MoeFormat


def _validate_operand(name, tensor) -> None:
    if isinstance(tensor, BlockScaledTensor):
        if tensor.format is MoeFormat.MXFP8:
            return
        raise NotImplementedError(
            "MoeEp training MegaMoE supports only MXFP8 BlockScaledTensor "
            f"inputs; {name} has format={tensor.format.value!r}"
        )
    if tensor.dtype not in {
        torch.bfloat16,
        torch.float16,
        torch.float32,
    }:
        raise NotImplementedError(
            f"MoeEp MegaMoE {name} staging supports BF16, FP16, "
            f"or FP32 plain tensors, got {tensor.dtype}"
        )


def _validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise NotImplementedError(
            f"MoeEp MegaMoE backend requires a CUDA device, got {device}"
        )

    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (10, 7):
        raise NotImplementedError(
            "MoeEp MegaMoE backend requires Rubin SM107 "
            "(compute capability 10.7); "
            f"found compute capability {major}.{minor}"
        )


def _is_cuda_stream_capturing(device: torch.device) -> bool:
    """Return capture state for the request device."""

    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _validate_wgrad_config(config: ForwardConfig) -> None:
    if config.backward_wgrad_mode not in ("none", "operands"):
        raise ValueError(
            "unsupported backward_wgrad_mode "
            f"{config.backward_wgrad_mode!r}"
        )
    if config.backward_wgrad_mode == "operands":
        if not config.generate_c:
            raise ValueError(
                "backward_wgrad_mode='operands' requires generate_c=True"
            )
        if config.token_padding_size != 256:
            raise ValueError(
                "backward_wgrad_mode='operands' requires "
                "token_padding_size=256"
            )
        if config.sf_padding_size != 128:
            raise ValueError(
                "backward_wgrad_mode='operands' requires "
                "sf_padding_size=128"
            )


def validate_config(config: ForwardConfig) -> None:
    """Reject static configurations outside the current MegaMoE milestone."""

    _validate_wgrad_config(config)
    if config.output_format != MoeFormat.BF16.value:
        raise NotImplementedError(
            "MoeEp training MegaMoE supports output_format='bf16' only"
        )
    supported_combine_formats = {
        MoeFormat.BF16.value,
        MoeFormat.MXFP8.value,
    }
    if config.combine_format not in supported_combine_formats:
        raise NotImplementedError(
            "MoeEp training MegaMoE supports combine_format='bf16' "
            "or 'mxfp8'"
        )
    if config.max_tokens_per_rank is None:
        raise NotImplementedError(
            "MoeEp MegaMoE backend requires an explicit max_tokens_per_rank"
        )
    if config.max_tokens_per_rank == 0:
        raise NotImplementedError(
            "MoeEp SM107 MXFP8 execution requires "
            "max_tokens_per_rank to be positive"
        )
    if config.hidden_size % 128:
        raise NotImplementedError(
            "MoeEp SM107 MXFP8 kernel currently requires hidden_size "
            f"to be divisible by 128, got {config.hidden_size}"
        )
    if config.intermediate_size % 256:
        raise NotImplementedError(
            "MoeEp SM107 MXFP8 kernel currently requires intermediate_size "
            f"to be divisible by 256, got {config.intermediate_size}"
        )
    if config.top_k > 32:
        raise NotImplementedError(
            "MoeEp SM107 MXFP8 dispatch currently requires top_k <= 32"
        )
    if not config.apply_topk_in_fc1:
        raise NotImplementedError(
            "MoeEp Rubin training MegaMoE requires apply_topk_in_fc1=True"
        )


def validate_request(request: ValidatedForwardRequest) -> None:
    """Reject valid requests outside the current MegaMoE input/device family."""

    for name, tensor in (
        ("activation", request.activation),
        ("fc1_weight", request.fc1_weight),
        ("fc2_weight", request.fc2_weight),
    ):
        _validate_operand(name, tensor)

    _validate_device(request.device)


def validate_backward_request(request: ValidatedBackwardRequest) -> None:
    """Reject backward requests outside the Rubin MXFP8 training path."""

    _validate_wgrad_config(request.config)
    for name, tensor in (
        ("fc1_weight", request.fc1_weight),
        ("fc2_weight", request.fc2_weight),
    ):
        _validate_operand(name, tensor)
    _validate_device(request.device)
    config = request.config
    if config.output_format != MoeFormat.BF16.value:
        raise NotImplementedError(
            "MoeEp MXFP8 backward currently requires output_format='bf16'"
        )
    if not config.apply_topk_in_fc1:
        raise NotImplementedError(
            "MoeEp MXFP8 backward currently requires apply_topk_in_fc1=True"
        )
    if _is_cuda_stream_capturing(request.device):
        raise NotImplementedError(
            "MoeEp MXFP8 backward does not support CUDA graph capture"
        )


__all__ = [
    "validate_backward_request",
    "validate_config",
    "validate_request",
]
