# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .api import (
    GemmSreluSm100,
    gemm_srelu_wrapper_sm100,
)

__all__ = [
    "GemmSreluSm100",
    "gemm_srelu_wrapper_sm100",
]


__all__.append("gemm_srelu_jax_sm100")


def __getattr__(name):
    # Lazy: the jax entry point imports jax/cutlass.jax, which must not be pulled in
    # for torch-only users.
    if name == "gemm_srelu_jax_sm100":
        from .jax_api import gemm_srelu_jax_sm100

        return gemm_srelu_jax_sm100
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
