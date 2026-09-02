# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared Rubin environment setup for direction-specific MXFP8 compilers."""

from __future__ import annotations

import os
from typing import Any

import torch

from ._config import Mxfp8KernelConfig
from ._cutedsl import require_rubin_cutedsl


def _prepare_rubin_environment(
    device: torch.device,
    config: Mxfp8KernelConfig,
    *,
    context: str,
) -> tuple[tuple[int, int], int]:
    require_rubin_cutedsl()
    torch.cuda.set_device(device)
    architecture = torch.cuda.get_device_capability(device)
    if architecture != (10, 7):
        raise RuntimeError(
            f"Rubin MXFP8 {context} preparation requires compute capability "
            f"(10, 7), got {architecture}"
        )

    configured_architecture = os.environ.get("CUTE_DSL_ARCH")
    if configured_architecture is None:
        os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    elif configured_architecture not in ("sm_107", "sm_107a"):
        raise RuntimeError(
            "CUTE_DSL_ARCH must target SM107 for Rubin MXFP8 "
            f"{context}, got {configured_architecture!r}"
        )

    import cutlass.utils as utils

    launch_cluster_count = int(
        utils.HardwareInfo().get_max_active_clusters(config.cluster_size)
    )
    if launch_cluster_count <= 0:
        raise RuntimeError(
            "hardware occupancy query returned no launchable Rubin clusters"
        )
    return architecture, launch_cluster_count


def _compile_kernel(kernel: Any, compile_kwargs: dict[str, Any]) -> Any:
    """Import CuTeDSL only on a cache miss and compile one callable."""

    require_rubin_cutedsl()
    import cutlass.cute as cute

    return cute.compile(kernel, **compile_kwargs)


__all__ = ["_compile_kernel", "_prepare_rubin_environment"]
