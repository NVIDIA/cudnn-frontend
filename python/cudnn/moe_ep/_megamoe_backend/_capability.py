# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability policy for the private MegaMoE execution backend.

Inputs reaching this module already satisfy the public :mod:`cudnn.moe_ep`
contract.  These checks only describe the subset that the current MegaMoE
implementation can execute; they must run before runtime initialization,
allocation, compilation, or collectives.
"""

from __future__ import annotations

import torch

from .._config import ResolvedMoeEpConfig
from .._contracts import _ForwardCall
from .._types import BlockScaledTensor, MoeFormat


def _validate_operand(name, tensor) -> None:
    if isinstance(tensor, BlockScaledTensor):
        if tensor.format is MoeFormat.MXFP8:
            return
        raise NotImplementedError("MoeEp training MegaMoE supports only MXFP8 BlockScaledTensor " f"inputs; {name} has format={tensor.format.value!r}")
    if tensor.dtype not in {
        torch.bfloat16,
        torch.float16,
        torch.float32,
    }:
        raise NotImplementedError(f"MoeEp MegaMoE {name} staging supports BF16, FP16, " f"or FP32 plain tensors, got {tensor.dtype}")


def _validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise NotImplementedError(f"MoeEp MegaMoE backend requires a CUDA device, got {device}")

    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (10, 7):
        raise NotImplementedError("MoeEp MegaMoE backend requires Rubin SM107 " "(compute capability 10.7); " f"found compute capability {major}.{minor}")


def _is_cuda_stream_capturing(device: torch.device) -> bool:
    """Return capture state for the request device."""

    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def validate_config(config: ResolvedMoeEpConfig) -> None:
    """Reject static configurations outside the current MegaMoE milestone."""

    model = config.public_config.model
    parallel = config.public_config.parallel
    data_path = config.public_config.data_path
    if data_path.output_format is not MoeFormat.BF16:
        raise NotImplementedError("MoeEp training MegaMoE supports output_format='bf16' only")
    supported_combine_formats = {
        MoeFormat.BF16,
        MoeFormat.MXFP8,
    }
    if data_path.combine_format not in supported_combine_formats:
        raise NotImplementedError("MoeEp training MegaMoE supports combine_format='bf16' " "or 'mxfp8'")
    if parallel.max_tokens_per_rank is None:
        raise NotImplementedError("MoeEp MegaMoE backend requires an explicit max_tokens_per_rank")
    if parallel.max_tokens_per_rank == 0:
        raise NotImplementedError("MoeEp SM107 MXFP8 execution requires " "max_tokens_per_rank to be positive")
    if model.hidden_size % 128:
        raise NotImplementedError("MoeEp SM107 MXFP8 kernel currently requires hidden_size " f"to be divisible by 128, got {model.hidden_size}")
    if model.intermediate_size % 256:
        raise NotImplementedError("MoeEp SM107 MXFP8 kernel currently requires intermediate_size " f"to be divisible by 256, got {model.intermediate_size}")
    if model.top_k > 32:
        raise NotImplementedError("MoeEp SM107 MXFP8 dispatch currently requires top_k <= 32")
    if not data_path.apply_topk_in_fc1:
        raise NotImplementedError("MoeEp Rubin training MegaMoE requires apply_topk_in_fc1=True")


def validate_request(
    config: ResolvedMoeEpConfig,
    request: _ForwardCall,
) -> None:
    """Reject valid requests outside the current MegaMoE input/device family."""

    del config
    for name, tensor in (
        ("activation", request.activation),
        ("fc1_weight", request.fc1_weight),
        ("fc2_weight", request.fc2_weight),
    ):
        _validate_operand(name, tensor)

    _validate_device(request.device)


__all__ = [
    "validate_config",
    "validate_request",
]
