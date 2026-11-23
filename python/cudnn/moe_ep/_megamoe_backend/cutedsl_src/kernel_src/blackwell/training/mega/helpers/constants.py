# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Training-mega-specific numeric constants (MMA tiler extents)."""


# MMA tiler extents training FC12 kernels accept along M and N.
SupportedMmaTileM = (128, 256)
SupportedMmaTileN = (128, 256)


__all__ = [
    "SupportedMmaTileM",
    "SupportedMmaTileN",
]
