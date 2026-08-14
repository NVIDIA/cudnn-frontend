# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN drop-in acceleration for flash-linear-attention (FLA).

``accelerate_fla()`` monkeypatches the FLA ops cuDNN can serve so an existing
``import fla`` training/inference script gets cuDNN's Blackwell kernels with no
code change, and transparently falls back to FLA where cuDNN has no kernel — so
results never change. Today: ``gated_delta_rule`` (Gated DeltaNet).

    import cudnn.fla
    cudnn.fla.accelerate_fla()   # call before importing FLA layers/models

Correctness is gated by ``test/python/linear_attention/test_fla_compat.py``, which
requires cuDNN to match FLA within its own bf16 noise on the output and every
gradient; a config that does not match must fall back rather than run.
"""

from __future__ import annotations

import sys

from .gated_delta_rule import make_chunk_gated_delta_rule, last_path

__all__ = ["accelerate_fla", "is_accelerated", "last_path"]

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
    try:
        import fla.ops.gated_delta_rule as gdr_mod
    except ImportError as e:
        raise ImportError("accelerate_fla() requires flash-linear-attention installed") from e

    original = gdr_mod.chunk_gated_delta_rule
    patched = make_chunk_gated_delta_rule(original)
    _ORIGINALS["chunk_gated_delta_rule"] = original
    _rebind_everywhere("chunk_gated_delta_rule", original, patched)

    if verbose:
        print("[cudnn.fla] accelerated FLA gated_delta_rule with cuDNN (SM100); " "unsupported configs fall back to FLA.")


def restore_fla() -> None:
    """Undo :func:`accelerate_fla`."""
    for fn_name, original in list(_ORIGINALS.items()):
        import fla.ops.gated_delta_rule as gdr_mod

        current = getattr(gdr_mod, fn_name, None)
        _rebind_everywhere(fn_name, current, original)
    _ORIGINALS.clear()
