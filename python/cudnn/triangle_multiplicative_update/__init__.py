# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triangle multiplicative update (AlphaFold-class "outgoing" / "incoming" triangle multiplication) on the FROST kernels.

Three launches, one engine (forward):

  K1  ``kernels.trimul_proj``   LayerNorm over the channel + the gated dual projection (value || gate) + sigmoid * mul + mask
                                -> the channel-major planes ``a``, ``b`` of shape ``[c, B * N, n_pad]``
  K2  ``kernels.contraction``   the per-channel contraction as ONE batched GEMM on the FROST GEMM engine (batch = c * B, K = N):
                                outgoing ``y[c] = a[c] @ b[c]^T``, incoming ``y[c] = a[c]^T @ b[c]`` -> channel-major ``y``
  K3  ``kernels.trimul_tail``   LayerNorm over the channel of ``y`` + the output gate (sigmoid of a projection of the LayerNormed input)
                                * the output projection -> the pair-major update

The channel-major planes are the contract between the stages: K2 needs no kernel of its own -- the dense FROST GEMM template serves
the batched matmul graph directly (``contraction.contraction_graph``), so the contraction inherits every tile config, cluster shape and
epilogue of that engine.  K1 (``kernels.trimul_proj``) and K3 (``kernels.trimul_tail``) are the two arms of the LayerNorm-prologue dual
GEMM substrate ``cudnn.gemm.frost.ln_prologue``.

Public surface (``api.py``; cuEquivariance's ``triangle_multiplicative_update`` signature, pinned to 0.12.0):

  ``triangle_multiplicative_update(x, direction, mask, norm_in_weight, ..., eps=1e-5, precision=None)``   the drop-in function
  ``TriangleMultiplicativeUpdateFwd``                                                                     the prepared class API
  ``prepare_weights`` / ``TriMulWeights``                                                                 the kernels' weight layouts, once
  ``TriMulKnobs`` / ``heuristics.recommend``                                                              the performance knobs and their measured defaults
  ``triangle_multiplicative_update_reference``                                                            the fp64 reference (``reference.py``)

Every name resolves lazily (``import cudnn.triangle_multiplicative_update`` pulls no torch), and this package module is itself CALLABLE:
``cudnn.triangle_multiplicative_update(x, ...)`` runs the function whether the attribute resolved to the function (the lazy export) or to
this package (Python binds an imported submodule on its parent, bypassing the lazy table -- a same-named function and package).
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

_LAZY_EXPORTS = {
    "TriangleMultiplicativeUpdateFwd": (".api", "TriangleMultiplicativeUpdateFwd"),
    "TriMulWeights": (".api", "TriMulWeights"),
    "TriMulKnobs": (".heuristics", "TriMulKnobs"),
    "prepare_weights": (".api", "prepare_weights"),
    "triangle_multiplicative_update": (".api", "triangle_multiplicative_update"),
    "CUEQ_SIGNATURE_VERSION": (".api", "CUEQ_SIGNATURE_VERSION"),
    "triangle_multiplicative_update_reference": (".reference", "triangle_multiplicative_update_reference"),
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
    """``cudnn.triangle_multiplicative_update(...)`` calls the function even when the attribute is this package (see the docstring)."""

    def __call__(self, *args, **kwargs):
        return self.triangle_multiplicative_update(*args, **kwargs)


sys.modules[__name__].__class__ = _CallablePackage
