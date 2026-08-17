# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN drop-in acceleration for flash-linear-attention (FLA).

``accelerate_fla()`` monkeypatches the FLA ops cuDNN can serve so an existing
``import fla`` training/inference script gets cuDNN's Blackwell kernels with no
code change, and transparently falls back to FLA where cuDNN has no kernel — so
results never change. Today: ``gated_delta_rule`` (Gated DeltaNet) and ``kda``
(Kimi Delta Attention).

    import cudnn.fla
    cudnn.fla.accelerate_fla()   # call before importing FLA layers/models

Correctness is gated by ``test/python/linear_attention/test_fla_compat.py``, which
requires cuDNN to match FLA within its own bf16 noise on the output and every
gradient; a config that does not match must fall back rather than run.
"""

from __future__ import annotations

import sys

from .gated_delta_rule import make_chunk_gated_delta_rule, last_path
from .kda import make_chunk_kda

__all__ = ["accelerate_fla", "is_accelerated", "last_path"]

# The FLA ops cuDNN accelerates: (import path, attribute, shim factory).
_ACCELERATED = [
    ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule", make_chunk_gated_delta_rule),
    ("fla.ops.kda", "chunk_kda", make_chunk_kda),
]

_ORIGINALS: dict = {}


def is_accelerated() -> bool:
    return bool(_ORIGINALS)


def _rebind_everywhere(fn_name: str, original, replacement) -> None:
    """Rebind ``fn_name`` from ``original`` to ``replacement`` in every module that
    captured it by reference (e.g. FLA layers that did ``from ... import fn``)."""
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, fn_name, None) is original:
                setattr(mod, fn_name, replacement)
        except Exception:
            # Some modules raise on getattr of arbitrary names; skip them.
            continue


def accelerate_fla(verbose: bool = True) -> None:
    """Patch the FLA ops cuDNN accelerates. Idempotent; call before FLA models load."""
    if is_accelerated():
        return
    import importlib

    patched_names = []
    for mod_path, attr, maker in _ACCELERATED:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue  # this op not present in the installed FLA
        original = getattr(mod, attr, None)
        if original is None:
            continue
        _ORIGINALS[attr] = (mod_path, original)
        _rebind_everywhere(attr, original, maker(original))
        patched_names.append(attr)

    if not patched_names:
        raise ImportError("accelerate_fla() requires flash-linear-attention installed")
    if verbose:
        print(f"[cudnn.fla] accelerated FLA {', '.join(patched_names)} with cuDNN (SM100); " "unsupported configs fall back to FLA.")


def restore_fla() -> None:
    """Undo :func:`accelerate_fla`."""
    import importlib

    for attr, (mod_path, original) in list(_ORIGINALS.items()):
        mod = importlib.import_module(mod_path)
        current = getattr(mod, attr, None)
        if current is not None and current is not original:
            _rebind_everywhere(attr, current, original)
        setattr(mod, attr, original)  # restore the owning module's attribute explicitly
    _ORIGINALS.clear()
