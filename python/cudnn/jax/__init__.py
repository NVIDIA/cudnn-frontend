# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX integration for the cuDNN frontend CuTeDSL APIs.

Built on CuTeDSL's native JAX bridge (``cutlass.jax.cutlass_call``): kernels run on
XLA's compute stream as FFI custom calls, outputs are XLA-managed, and calls compose
with ``jax.jit`` (and CUDA graph capture). :func:`cudnn.jax.call` is a thin wrapper
adding the conveniences the cuDNN kernels need — pre-initialized accumulator outputs
and TensorSpec presets for the layouts the GEMM fusions use.

Requires jax >= 0.9.1 and nvidia-cutlass-dsl >= 4.7.1.
"""

from importlib import import_module

from cudnn import CUTEDSL_MIN_JAX_VERSION, CUTEDSL_MIN_JAX_VERSION_STR
from cudnn.frost.buffers import cutedsl_requirement_error as _cutedsl_requirement_error

_requirement_error = _cutedsl_requirement_error("cudnn.jax", floor=(4, 7, 1))
if _requirement_error:
    raise ImportError(_requirement_error)

import jax as _jax

if _jax.version.__version_info__ < CUTEDSL_MIN_JAX_VERSION:
    raise ImportError(f"cudnn.jax requires jax >= {CUTEDSL_MIN_JAX_VERSION_STR}; found {_jax.__version__}")

import cutlass.jax

if not cutlass.jax.is_available():  # pragma: no cover - guarded import surface
    raise ImportError(
        f"cudnn.jax requires jax >= {CUTEDSL_MIN_JAX_VERSION_STR} and the CuTeDSL JAX extensions (cutlass.jax); "
        "install/upgrade jax (`pip install --group jax` from a checkout)"
    )

from .call import (
    call,
    gemm_operand_spec,
    row_major_desc,
    sf_atom_spec,
    zeros_init,
    neg_inf_init,
)
from cutlass.jax import TensorSpec

_BSA_EXPORTS = ("block_sparse_attention",)


__all__ = [
    *_BSA_EXPORTS,
    "call",
    "row_major_desc",
    "TensorSpec",
    "gemm_operand_spec",
    "sf_atom_spec",
    "zeros_init",
    "neg_inf_init",
    "kimi_delta_attention",
]


def __getattr__(name):
    if name in _BSA_EXPORTS:
        value = getattr(import_module("cudnn.block_sparse_attention.jax_api"), name)
        globals()[name] = value
        return value
    if name == "kimi_delta_attention":
        from cudnn.linear_attention import jax_api as kda

        value = getattr(kda, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
