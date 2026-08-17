# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent GEMM operation contracts (torch custom-op wrappers)."""

from .moe_grouped_matmul import moe_grouped_matmul
from .swiglu_mlp import swiglu_mlp

__all__ = ["moe_grouped_matmul", "swiglu_mlp"]
