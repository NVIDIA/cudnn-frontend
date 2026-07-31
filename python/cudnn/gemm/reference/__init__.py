# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch MATMUL/POINTWISE correctness engine."""

from .reference_matmul_engine import ReferenceMatmulEngine

__all__ = ["ReferenceMatmulEngine"]
