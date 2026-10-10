# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared dtype tables for the sm_100 norm kernels (single source of truth).

Norm kernels currently support the three high/standard-precision floating types
only: bfloat16, float16, float32. Statistics (mean / inv_variance) and parameter
gradients (dscale / dbias) are always fp32, matching cuDNN's norm backend. Lower
precisions (fp8/fp4/mx) are intentionally deferred.
"""

from __future__ import annotations

from typing import Any

import cutlass

# internal dtype string -> cute-DSL / cutlass Numeric type.
DTYPE_TO_CUTLASS: dict[str, Any] = {
    "bf16": cutlass.BFloat16,
    "fp16": cutlass.Float16,
    "fp32": cutlass.Float32,
}

# internal dtype -> element size in bytes.
DTYPE_BYTES: dict[str, int] = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}

SUPPORTED_IO_DTYPES = ("bf16", "fp16", "fp32")

# Statistics and parameter gradients are always accumulated/stored in fp32.
STATS_DTYPE = "fp32"


def torch_dtype_to_str(dt: Any) -> str:
    """Map a ``torch.dtype`` to our internal dtype string."""
    import torch

    table = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }
    try:
        return table[dt]
    except KeyError:
        raise ValueError(f"norm sm_100 kernels support {SUPPORTED_IO_DTYPES} only; got torch dtype {dt}") from None


def str_to_torch_dtype(s: str) -> Any:
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]
