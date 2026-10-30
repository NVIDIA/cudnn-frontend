# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public surface for the fused CSA/HCA Compressor kernels (re-exports from ``.api``)."""

from importlib import import_module

__all__ = [
    "CSACompressorBackward",
    "CSACompressorForward",
    "csa_compressor_backward_wrapper",
    "csa_compressor_forward_wrapper",
]


def __getattr__(name):
    # Lazy so the kernel modules (used by the JAX ops) import without torch.
    if name in __all__:
        value = getattr(import_module(".api", package=__name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
