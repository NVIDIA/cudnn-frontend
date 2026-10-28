# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental functions for HSTU LayerNorm-Multiply-SiLU-Dropout."""

from importlib import import_module

__all__ = [
    "hstu_lmsd_forward",
    "hstu_lmsd_backward",
]


# Resolved on first access so the JAX entry points (``jax_api``) import without torch.
def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".ops", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
