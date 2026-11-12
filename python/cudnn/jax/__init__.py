# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated namespace. JAX ops are the ``*_jax`` functions in :mod:`cudnn.ops`."""

import warnings
from importlib import import_module

_MOVED = {
    "call": "cudnn._jax",
    "row_major_desc": "cudnn._jax",
    "TensorSpec": "cudnn._jax",
    "gemm_operand_spec": "cudnn._jax",
    "sf_atom_spec": "cudnn._jax",
    "zeros_init": "cudnn._jax",
    "neg_inf_init": "cudnn._jax",
    "block_sparse_attention": "cudnn.block_sparse_attention.jax_api",
    "kimi_delta_attention": "cudnn.linear_attention.jax_api",
}

__all__ = list(_MOVED)


def __getattr__(name):
    if name not in _MOVED:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    ops = f"cudnn.ops.{name}_forward_jax and cudnn.ops.{name}_backward_jax" if name.endswith("attention") else "the `*_jax` functions in cudnn.ops"
    # FutureWarning, not DeprecationWarning: the default filters drop a DeprecationWarning raised from library code.
    warnings.warn(f"cudnn.jax.{name} is deprecated and will be removed in a future release; use {ops}", FutureWarning, stacklevel=2)
    return getattr(import_module(_MOVED[name]), name)
