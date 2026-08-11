# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .api import (
    GemmDsreluSm100,
    gemm_dsrelu_wrapper_sm100,
)

__all__ = [
    "GemmDsreluSm100",
    "gemm_dsrelu_wrapper_sm100",
]


__all__.append("gemm_dsrelu_jax_sm100")


def __getattr__(name):
    # Lazy: the jax entry point imports jax/cutlass.jax, which must not be pulled in
    # for torch-only users.
    if name == "gemm_dsrelu_jax_sm100":
        from ..srelu.jax_api import gemm_dsrelu_jax_sm100

        return gemm_dsrelu_jax_sm100
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
