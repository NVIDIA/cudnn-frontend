# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free CuTe compilation options for framework adapters."""

from __future__ import annotations

_ARCH_MAP = {
    90: "sm_90a",
    100: "sm_100a",
    103: "sm_103a",
    107: "sm_100f",
}


def gpu_arch_flag_for_compute_capability(compute_capability: int) -> str:
    """Return the explicit CuTe architecture flag for a compilation target."""

    if isinstance(compute_capability, bool) or not isinstance(compute_capability, int):
        raise TypeError(f"compute_capability must be an int, got {type(compute_capability).__name__}")
    try:
        return _ARCH_MAP[compute_capability]
    except KeyError as error:
        supported = ", ".join(f"SM{value}" for value in sorted(_ARCH_MAP))
        raise RuntimeError(f"Unsupported GPU compute capability SM{compute_capability} for CuTe kernels; supported targets are {supported}") from error


def compile_options_for_target(compute_capability: int, extra: str = "") -> str:
    """Build CuTe compile options for an explicit framework-supplied target."""

    parts = [
        "--enable-tvm-ffi",
        f"--gpu-arch {gpu_arch_flag_for_compute_capability(compute_capability)}",
    ]
    if extra:
        parts.append(extra)
    return " ".join(parts)


__all__ = ["compile_options_for_target", "gpu_arch_flag_for_compute_capability"]
