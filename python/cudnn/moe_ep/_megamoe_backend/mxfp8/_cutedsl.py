# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CUTLASS DSL compatibility gate for Rubin MegaMoE kernels."""

from __future__ import annotations

import importlib.metadata

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


__all__ = ["RUBIN_CUTEDSL_MIN_VERSION", "require_rubin_cutedsl"]
