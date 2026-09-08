# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rubin training MegaMoE (mxfp8 dGLU backward) kernel components.

The backward fused dfc2 + dswiglu + dfc1 MoE kernel used by the training path. Computes
grad_x (activation gradient -> output_activation) and dprob (routing-weight gradient,
pool region). Built on top of the forward GLU package (subclasses the shared
``Fc2OutputDest`` peer-store) and mirrors the forward mega structure.
"""

from .dglu_mxfp8_fc12_epilogue import DgluMxfp8Epilogue
from .dglu_mxfp8_fc12_extension import DgluMxFp8Fc12SchedExtension
from .dglu_mxfp8_fc12_kernel import Sm107Mxfp8DgluDfc21Kernel
from .dglu_mxfp8_mega_moe_kernel import Sm107MegaMoEMxfp8DgluKernel


__all__ = [
    "DgluMxfp8Epilogue",
    "DgluMxFp8Fc12SchedExtension",
    "Sm107Mxfp8DgluDfc21Kernel",
    "Sm107MegaMoEMxfp8DgluKernel",
]
