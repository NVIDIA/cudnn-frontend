# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Current-stream launch of the Rubin MXFP8 dGLU backward product."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._plan import PreparedResources
from ._backward_compile import (
    CompiledMxfp8BackwardKernel,
    Mxfp8BackwardLaunchInputs,
    build_backward_runtime_kwargs,
)
from ._launch import _check_overflow


@dataclass(frozen=True)
class Mxfp8DgluResult:
    grad_activation: torch.Tensor
    grad_topk_weights: torch.Tensor
    fc1_recompute: torch.Tensor
    fc1_recompute_sf: torch.Tensor
    fc1_col_output: torch.Tensor
    fc1_col_output_sf: torch.Tensor
    grad_y2: torch.Tensor
    grad_y2_sf: torch.Tensor


def launch_backward_dglu(
    compiled: CompiledMxfp8BackwardKernel,
    inputs: Mxfp8BackwardLaunchInputs,
    resources: PreparedResources,
) -> Mxfp8DgluResult:
    runtime_kwargs = build_backward_runtime_kwargs(inputs, resources)
    compiled.callable(**runtime_kwargs)
    _check_overflow(inputs.overflow_flag)

    return Mxfp8DgluResult(
        grad_activation=inputs.output_activation[
            : inputs.token_count
        ].float(),
        # The dGLU epilogue has already returned source-order dprob through
        # the symmetric token-communication plane. Own the public result so a
        # later launch cannot overwrite it.
        grad_topk_weights=inputs.dprob[: inputs.token_count].clone(),
        fc1_recompute=inputs.fc1_recompute,
        fc1_recompute_sf=inputs.fc1_recompute_sf,
        fc1_col_output=inputs.fc1_col_output,
        fc1_col_output_sf=inputs.fc1_col_output_sf,
        grad_y2=inputs.grad_y2,
        grad_y2_sf=inputs.grad_y2_sf,
    )


__all__ = ["Mxfp8DgluResult", "launch_backward_dglu"]
