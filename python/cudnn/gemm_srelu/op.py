# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral dense GEMM + squared-ReLU operation."""

from ..gemm.srelu import BlockScaledGemmSreluSm100OpBase


class GemmSreluSm100Op(BlockScaledGemmSreluSm100OpBase):
    """Logical signature for block-scaled GEMM + squared ReLU."""

    def check_support(self) -> bool:
        if self.dprob is not None:
            raise ValueError("dprob is only part of the dsReLU backward signature")
        return super().check_support()


__all__ = ["GemmSreluSm100Op"]
