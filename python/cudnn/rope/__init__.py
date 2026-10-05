# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental rotary-position embedding APIs."""

from importlib import import_module

_EXPORT_MODULES = {
    "TailRoPEForward": ".tail",
    "tail_rope": ".tail",
    "VisionRoPEBackward": ".api",
    "vision_rope_backward_wrapper": ".api",
    "RopeQDQInplace": ".qdq",
    "rope_qdq_inplace": ".qdq",
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    if name in _EXPORT_MODULES:
        value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
