# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compact causal GQA backward."""

__all__ = ["CompactGqaBackward", "compact_gqa_backward"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import api

    value = getattr(api, name)
    globals()[name] = value
    return value
