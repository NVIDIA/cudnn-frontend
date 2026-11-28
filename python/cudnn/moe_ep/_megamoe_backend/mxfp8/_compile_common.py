# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared device environment setup for direction-specific MXFP8 compilers."""

from __future__ import annotations

import os
from typing import Any

import torch

from ._arch import architecture_for
from ._cutedsl import require_megamoe_cutedsl


def _prepare_environment(
    device: torch.device,
    *,
    cluster_size: int,
    context: str,
) -> tuple[tuple[int, int], int]:
    require_megamoe_cutedsl()
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    architecture = architecture_for(capability)

    configured_architecture = os.environ.get("CUTE_DSL_ARCH")
    if configured_architecture is None:
        os.environ["CUTE_DSL_ARCH"] = architecture.cute_dsl_arch
    elif configured_architecture not in architecture.cute_dsl_arch_spellings:
        raise RuntimeError(
            f"CUTE_DSL_ARCH must target {architecture.cute_dsl_arch} for {architecture.name} MXFP8 " f"{context}, got {configured_architecture!r}"
        )

    import cutlass.utils as utils

    launch_cluster_count = int(utils.HardwareInfo().get_max_active_clusters(cluster_size))
    if launch_cluster_count <= 0:
        raise RuntimeError(f"hardware occupancy query returned no launchable {architecture.name} clusters")
    return capability, launch_cluster_count


def _compile_kernel(kernel: Any, compile_kwargs: dict[str, Any]) -> Any:
    """Import CuTeDSL only on a cache miss and compile one callable."""

    require_megamoe_cutedsl()
    import cutlass.cute as cute

    return cute.compile(kernel, **compile_kwargs)


__all__ = ["_compile_kernel", "_prepare_environment"]
