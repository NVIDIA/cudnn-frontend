# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CuTe architecture naming shared by framework adapters."""

from __future__ import annotations

_ARCH_MAP = {
    90: "sm_90a",
    100: "sm_100a",
    103: "sm_103a",
    107: "sm_100f",
    110: "sm_110a",
    120: "sm_120a",
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


__all__ = ["gpu_arch_flag_for_compute_capability"]
