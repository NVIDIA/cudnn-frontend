# SPDX-License-Identifier: BSD-3-Clause
"""CUDA architecture helpers used by host-side dispatch."""

from __future__ import annotations

from functools import lru_cache

import torch

SUPPORTED_ARCHES = (90, 100, 103)


@lru_cache(maxsize=None)
def get_device_arch(device_index: int | None = None) -> int:
    """Return the CUDA compute capability encoded as ``major * 10 + minor``."""

    if not torch.cuda.is_available():
        raise RuntimeError("cudnn.flex_attention requires a CUDA device")
    major, minor = torch.cuda.get_device_capability(device_index)
    arch = major * 10 + minor
    if arch not in SUPPORTED_ARCHES:
        raise NotImplementedError(f"cudnn.flex_attention supports SM90, SM100, and SM103; got SM{arch}")
    return arch


__all__ = ["SUPPORTED_ARCHES", "get_device_arch"]
