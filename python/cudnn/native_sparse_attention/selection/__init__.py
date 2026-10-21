# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

__all__ = [
    "SelectionAttention",
    "selection_attention_wrapper",
]


# Resolved on first access so the kernel modules (used by ``jax_api``) import without torch.
def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".api", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
