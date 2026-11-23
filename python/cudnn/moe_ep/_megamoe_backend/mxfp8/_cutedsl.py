# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUTLASS DSL compatibility gate for Rubin MegaMoE kernels."""

from __future__ import annotations

import importlib.metadata
import importlib.util

RUBIN_CUTEDSL_MIN_VERSION = (4, 8, 0)


def _public_cutedsl_version() -> str | None:
    """Return public-wheel metadata without importing CUTLASS DSL."""

    try:
        return importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_version(version: str) -> tuple[int, int, int] | None:
    """Parse the numeric release prefix and tolerate prerelease suffixes."""

    parts = version.split("+", 1)[0].split(".")
    parsed = []
    try:
        for part in parts[:3]:
            digits = ""
            for character in part:
                if not character.isdigit():
                    break
                digits += character
            if not digits:
                return None
            parsed.append(int(digits))
    except (TypeError, ValueError):
        return None
    return tuple(parsed) if len(parsed) == 3 else None


def require_rubin_cutedsl() -> None:
    """Reject public CUTLASS DSL wheels older than Rubin kernel support."""

    version = _public_cutedsl_version()
    parsed = None if version is None else _parse_version(version)
    if parsed is not None and parsed < RUBIN_CUTEDSL_MIN_VERSION:
        raise RuntimeError(
            "Rubin MegaMoE MXFP8 kernels require "
            "nvidia-cutlass-dsl>=4.8.0; found "
            f"{version}. Other cuDNN Frontend APIs remain available with "
            "the package minimum of 4.5.0"
        )


def _installed_cutedsl_distributions() -> str:
    """Name every installed CUTLASS DSL distribution without importing it."""

    found = []
    for distribution in ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-internal"):
        try:
            found.append(f"{distribution} {importlib.metadata.version(distribution)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return ", ".join(found) if found else "none"


def _cutlass_installed() -> bool:
    """Whether a CUTLASS DSL package is importable, without importing it."""

    return importlib.util.find_spec("cutlass") is not None


def rubin_wgrad_cutedsl_error() -> str | None:
    """Return why the installed DSL cannot build the Rubin WGrad kernel, or None.

    The WGrad kernel issues Rubin TMA descriptor overrides with a residue
    tensor. The vendored ``helpers/dsl_helpers.py`` builds them from the DSL's
    override copy traits and its ``atom_make_tma_residue_tensor`` MLIR op, which
    public 4.8.0 and later ship, so the probe checks those capabilities rather
    than a version number.
    """

    try:
        require_rubin_cutedsl()
    except RuntimeError as exc:
        return str(exc)
    requirement = (
        f"MoeEp training WGrad requires nvidia-cutlass-dsl>={'.'.join(map(str, RUBIN_CUTEDSL_MIN_VERSION))} "
        "or another CUTLASS DSL build with Rubin TMA descriptor overrides"
    )
    if not _cutlass_installed():
        return f"{requirement}; no CUTLASS DSL is installed"
    try:
        import cutlass._mlir.dialects.cute_nvgpu as cute_nvgpu_ir
        from cutlass.cute.nvgpu.cpasync.copy import (  # noqa: F401
            CopyBulkTensorTileG2SOverrideNonExecTrait,
            CopyBulkTensorTileS2GOverrideNonExecTrait,
        )
    except ImportError as exc:
        return f"{requirement}; the installed DSL lacks the override copy traits ({exc}); found {_installed_cutedsl_distributions()}"
    if not hasattr(cute_nvgpu_ir, "atom_make_tma_residue_tensor"):
        return f"{requirement}; the installed DSL lacks the atom_make_tma_residue_tensor op; found {_installed_cutedsl_distributions()}"
    return None


def require_rubin_wgrad_cutedsl() -> None:
    """Raise unless the installed DSL can build the Rubin WGrad kernel."""

    error = rubin_wgrad_cutedsl_error()
    if error is not None:
        raise RuntimeError(error)


__all__ = [
    "RUBIN_CUTEDSL_MIN_VERSION",
    "require_rubin_cutedsl",
    "require_rubin_wgrad_cutedsl",
    "rubin_wgrad_cutedsl_error",
]
