# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Blackwell HSTU attention FE-OSS APIs."""

from importlib import import_module

__all__ = [
    "HSTUFwdSm100",
    "HSTUBwdSm100",
    "hstu_attention_forward",
    "hstu_attention_backward",
]


# Resolved on first access so the JAX entry points (``jax_api``) import without torch.
def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".api", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
