# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transparent pure-PyTorch MoE-EP with autograd (training bprop oracle).

See moe_ep_training/README.md for layout and conventions.
"""

from .config import EpConfig
from .layer import MoEEpTrainingLayer
from .layer_fp4 import MoEEpTrainingLayerFp4
from .quant import QuantConfig
from .reference import ReferenceMoE
from .reference_fp4 import ReferenceMoEFp4

__all__ = [
    "EpConfig",
    "MoEEpTrainingLayer",
    "MoEEpTrainingLayerFp4",
    "QuantConfig",
    "ReferenceMoE",
    "ReferenceMoEFp4",
]
