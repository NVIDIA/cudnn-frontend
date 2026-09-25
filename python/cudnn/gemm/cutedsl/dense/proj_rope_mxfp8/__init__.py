# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .api import (
    GemmProjRopeMxfp8Bf16InSm100,
    GemmProjRopeMxfp8Mxfp8InSm100,
    gemm_proj_rope_mxfp8_wrapper_sm100,
)
from .gemm_proj_rope_mxfp8_bf16in import gemm_proj_rope_mxfp8_reference

__all__ = [
    "GemmProjRopeMxfp8Bf16InSm100",
    "GemmProjRopeMxfp8Mxfp8InSm100",
    "gemm_proj_rope_mxfp8_wrapper_sm100",
    "gemm_proj_rope_mxfp8_reference",
    "gemm_proj_rope_mxfp8_jax_sm100",
]


def __getattr__(name):
    # Lazy: the jax entry point imports jax/cutlass.jax, which must not be pulled in
    # for torch-only users.
    if name == "gemm_proj_rope_mxfp8_jax_sm100":
        from .jax_api import gemm_proj_rope_mxfp8_jax_sm100

        return gemm_proj_rope_mxfp8_jax_sm100
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
