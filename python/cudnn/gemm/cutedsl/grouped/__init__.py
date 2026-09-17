# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .swiglu.api import (
    GroupedGemmSwigluSm100,
    grouped_gemm_swiglu_wrapper_sm100,
)

from .dswiglu.api import (
    GroupedGemmDswigluSm100,
    grouped_gemm_dswiglu_wrapper_sm100,
)

from .quant.api import (
    GroupedGemmQuantSm100,
    grouped_gemm_quant_wrapper_sm100,
)

from .srelu.api import (
    GroupedGemmSreluSm100,
    grouped_gemm_srelu_wrapper_sm100,
)

from .dsrelu.api import (
    GroupedGemmDsreluSm100,
    grouped_gemm_dsrelu_wrapper_sm100,
)

from .glu.api import (
    GroupedGemmGluSm100,
    grouped_gemm_glu_wrapper_sm100,
)

from .glu_hadamard.api import (
    GroupedGemmGluHadamardSm100,
    grouped_gemm_glu_hadamard_wrapper_sm100,
)

from .glu_hadamard_quant.api import (
    GroupedGemmGluHadamardQuantSm100,
    grouped_gemm_glu_hadamard_quant_wrapper_sm100,
)

from .dglu.api import (
    GroupedGemmDgluSm100,
    grouped_gemm_dglu_wrapper_sm100,
)

from .wgrad.api import (
    GroupedGemmWgradSm100,
    grouped_gemm_wgrad_wrapper_sm100,
)

from .unfused.api import (
    GroupedGemmSm100,
    grouped_gemm_wrapper_sm100,
)

__all__ = [
    "GroupedGemmSwigluSm100",
    "grouped_gemm_swiglu_wrapper_sm100",
    "GroupedGemmDswigluSm100",
    "grouped_gemm_dswiglu_wrapper_sm100",
    "GroupedGemmQuantSm100",
    "grouped_gemm_quant_wrapper_sm100",
    "GroupedGemmSreluSm100",
    "grouped_gemm_srelu_wrapper_sm100",
    "GroupedGemmDsreluSm100",
    "grouped_gemm_dsrelu_wrapper_sm100",
    "GroupedGemmGluSm100",
    "grouped_gemm_glu_wrapper_sm100",
    "GroupedGemmGluHadamardSm100",
    "grouped_gemm_glu_hadamard_wrapper_sm100",
    "GroupedGemmGluHadamardQuantSm100",
    "grouped_gemm_glu_hadamard_quant_wrapper_sm100",
    "GroupedGemmDgluSm100",
    "grouped_gemm_dglu_wrapper_sm100",
    "GroupedGemmWgradSm100",
    "grouped_gemm_wgrad_wrapper_sm100",
    "GroupedGemmSm100",
    "grouped_gemm_wrapper_sm100",
    "grouped_gemm_jax_sm100",
    "grouped_gemm_glu_jax_sm100",
    "grouped_gemm_dglu_jax_sm100",
    "grouped_gemm_dsrelu_jax_sm100",
    "grouped_gemm_wgrad_jax_sm100",
]

# Lazy: the jax entry points import jax/cutlass.jax, which must not be pulled in
# for torch-only users.
_JAX_LAZY_EXPORTS = {
    "grouped_gemm_jax_sm100": ".unfused",
    "grouped_gemm_glu_jax_sm100": ".glu",
    "grouped_gemm_dglu_jax_sm100": ".dglu",
    "grouped_gemm_dsrelu_jax_sm100": ".dsrelu",
    "grouped_gemm_wgrad_jax_sm100": ".wgrad",
}


def __getattr__(name):
    module_name = _JAX_LAZY_EXPORTS.get(name)
    if module_name is not None:
        import importlib

        return getattr(importlib.import_module(module_name, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
