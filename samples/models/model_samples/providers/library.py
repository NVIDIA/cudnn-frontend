# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library kernels as providers of model blocks.

``cublas.bf16_dequantized`` is a fallback, and the table says so: the multiply of an 8-bit or 4-bit block runs as a bf16
``torch.matmul`` (cuBLAS) over a bf16 copy of the weight's code values made once at load, on the operand's code values
dequantised per call; the two scales are applied to the result.  The copies are exact (see ``weights.dequantize``), so the
result differs from a multiply on the stored formats by its accumulation order and by one extra bf16 rounding of the unscaled
sum.  It moves 2 bytes per weight element where the stored formats hold 1 or 0.56.

Adapters to kernels on the stored formats arrive with the stack that uses them.
"""

from __future__ import annotations

import torch

from ..weights import Dequantized, dequantize
from . import BlockBase
from .torch_ref import silu_mul_quantize


def matmul_bf16(qx, weight: Dequantized) -> torch.Tensor:
    """``[rows, in]`` quantised operand x ``[out, in]`` bf16 code copy -> ``[rows, out]`` bf16, scales applied to the result."""
    x = dequantize(qx)
    y = torch.matmul(x.data, weight.data.T).float()
    if isinstance(weight.scale, torch.Tensor):
        return (y * (weight.scale * float(x.scale))).to(torch.bfloat16)
    return (y * (float(weight.scale) * float(x.scale))).to(torch.bfloat16)


class CublasLinear(BlockBase):
    provider = "cuBLAS"
    kernels = ("gemm (bf16)",)

    def __call__(self, qx, weight):
        return matmul_bf16(qx, weight)


class CublasGateUp(BlockBase):
    """The merged gate / up multiply on cuBLAS, then ``SiLU(gate) * up`` and the 4-bit quantisation in PyTorch."""

    provider = "cuBLAS + PyTorch"
    kernels = ("gemm (bf16)",)

    def __call__(self, qx, weight, quant):
        self.product = matmul_bf16(qx, weight)
        return silu_mul_quantize(self.product, quant)

    def compared(self, out):
        return self.product
