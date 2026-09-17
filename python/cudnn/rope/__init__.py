# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental rotary-position embedding APIs."""

from importlib import import_module

__all__ = ["VisionRoPEBackward", "vision_rope_backward_wrapper"]


def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".api", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
