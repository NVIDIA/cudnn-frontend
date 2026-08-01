# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys

from .sdpa import scaled_dot_product_attention

# moe_grouped_matmul now lives with the rest of the GEMM family in
# cudnn.gemm.ops. Alias the module into this package so that both
# ``from cudnn.experimental.ops import moe_grouped_matmul`` and
# ``import cudnn.experimental.ops.moe_grouped_matmul`` keep resolving.
_moe_mod = importlib.import_module("cudnn.gemm.ops.moe_grouped_matmul")
sys.modules[__name__ + ".moe_grouped_matmul"] = _moe_mod
moe_grouped_matmul = _moe_mod.moe_grouped_matmul

__all__ = [
    "scaled_dot_product_attention",
    "moe_grouped_matmul",
]
