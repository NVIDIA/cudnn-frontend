# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triangle attention (AlphaFold-class attention with a pair bias and a key mask) on the FROST SDPA engine.

The attention core is the shipped ``sdpa`` graph node served by the FROST ``sdpa_fwd_prefill_sm107`` engine row -- its additive
bias and per-key mask on the 32- / 64-wide head-dim tiles -- so nothing here is a kernel: this package is the API layer that
flattens cuEquivariance's 5-D ``(B, N, H, S, D)`` tensors onto the graph's 4-D vocabulary, hands the mask to the kernel in the
form the caller chose (right-padded lengths or a dense boolean mask), caches one compiled plan per declaration and launches
it on the caller's stream.

Public surface (``api.py``; cuEquivariance's ``triangle_attention`` signature, pinned to 0.12.0):

  ``triangle_attention(q, k, v, bias, mask=None, scale=None, return_aux=False, *, kv_lengths=None)``   the drop-in function
  ``mask_to_kv_lengths(mask)``                                                                            prefix-shaped mask -> lengths (or None)
  ``TriangleAttentionFwd``                                                                                 the prepared class API
  ``triangle_attention_reference``                                                                         the fp64 reference (``reference.py``)
  ``copy_counters`` / ``reset_copy_counters``                                                              the counted layout fallbacks of the function

Every name resolves lazily (``import cudnn.triangle_attention`` pulls no torch), and this package module is itself CALLABLE:
``cudnn.triangle_attention(q, ...)`` runs the function whether the attribute resolved to the function (the lazy export) or to this
package (Python binds an imported submodule on its parent, bypassing the lazy table -- a same-named function and package).
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

_LAZY_EXPORTS = {
    "TriangleAttentionFwd": (".api", "TriangleAttentionFwd"),
    "triangle_attention": (".api", "triangle_attention"),
    "mask_to_kv_lengths": (".api", "mask_to_kv_lengths"),
    "copy_counters": (".api", "copy_counters"),
    "reset_copy_counters": (".api", "reset_copy_counters"),
    "CUEQ_SIGNATURE_VERSION": (".api", "CUEQ_SIGNATURE_VERSION"),
    "ENGINE_ROW": (".api", "ENGINE_ROW"),
    "triangle_attention_reference": (".reference", "triangle_attention_reference"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    try:
        value = getattr(importlib.import_module(module_name, __name__), attr_name)
    except ImportError as error:
        from cudnn import _optional_dependency_message

        raise ImportError(_optional_dependency_message(name, error)) from error
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


class _CallablePackage(ModuleType):
    """``cudnn.triangle_attention(...)`` calls the function even when the attribute is this package (see the docstring)."""

    def __call__(self, *args, **kwargs):
        return self.triangle_attention(*args, **kwargs)


sys.modules[__name__].__class__ = _CallablePackage
