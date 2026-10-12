# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental saved-state Engram gates; dependencies are loaded on access."""

from importlib import import_module

__all__ = ["EngramGateSavedForward", "EngramGateSavedBackward", "engram_gate_saved_forward", "engram_gate_saved_backward"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state, cutedsl_too_old

    if cutedsl_too_old(cutedsl_state()[1]):
        raise RuntimeError(cutedsl_requirement_error("Engram saved-state gate"))
    value = getattr(import_module(".api", __name__), name)
    globals()[name] = value
    return value
