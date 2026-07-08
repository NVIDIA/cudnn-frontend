# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CUTLASS JAX compilation options."""

from __future__ import annotations

from ..common.cute_arch import gpu_arch_flag_for_compute_capability


def compile_options_for_target(compute_capability: int, extra: str = "") -> str:
    """Build CUTLASS JAX compile options for an explicit target."""

    parts = [f"--gpu-arch {gpu_arch_flag_for_compute_capability(compute_capability)}"]
    if extra:
        parts.append(extra)
    return " ".join(parts)


__all__ = ["compile_options_for_target"]
