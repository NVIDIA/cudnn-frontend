# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral dense GEMM + squared-ReLU backward operation."""

from .. import data_type
from ..gemm.srelu import BlockScaledGemmSreluSm100OpBase


class GemmDsreluSm100Op(BlockScaledGemmSreluSm100OpBase):
    """Logical signature for block-scaled GEMM + squared-ReLU backward."""

    def check_support(self) -> bool:
        if self.dprob is None:
            raise ValueError("dprob is required by the dsReLU backward signature")
        if self.c.cudnn_dtype not in {data_type.HALF, data_type.BFLOAT16, data_type.FLOAT}:
            raise ValueError(f"dsReLU C must use float16, bfloat16, or float32, got {self.c.dtype}")
        return super().check_support()


__all__ = ["GemmDsreluSm100Op"]
