# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX integration for the cuDNN frontend CuTeDSL APIs.

Built on CuTeDSL's native JAX bridge (``cutlass.jax.cutlass_call``): kernels run on
XLA's compute stream as FFI custom calls, outputs are XLA-managed, and calls compose
with ``jax.jit`` (and CUDA graph capture). :func:`cudnn.jax.call` is a thin wrapper
adding the conveniences the cuDNN kernels need — pre-initialized accumulator outputs
and TensorSpec presets for the layouts the GEMM fusions use.

Requires jax >= 0.5 and the CuTeDSL JAX extensions (shipped with nvidia-cutlass-dsl).
"""

from .call import (
    call,
    gemm_operand_spec,
    row_major_desc,
    sf_atom_spec,
    zeros_init,
    neg_inf_init,
)
from cutlass.jax import TensorSpec

__all__ = [
    "call",
    "row_major_desc",
    "TensorSpec",
    "gemm_operand_spec",
    "sf_atom_spec",
    "zeros_init",
    "neg_inf_init",
    "grouped_gemm_swiglu",
    "grouped_gemm_dswiglu",
]


def __getattr__(name):
    if name in ("grouped_gemm_swiglu", "grouped_gemm_dswiglu"):
        from importlib import import_module
        from cudnn.frost.buffers import cutedsl_requirement_error

        requirement = cutedsl_requirement_error(name)
        if requirement:
            raise ImportError(requirement)
        operation = name.removeprefix("grouped_gemm_")
        module = import_module(f"cudnn.gemm.cutedsl.grouped.{operation}.jax_api")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
