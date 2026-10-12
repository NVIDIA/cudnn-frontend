# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight private backend seam for :class:`cudnn.moe_ep.MoeEp`.

Capability policy and the executable factory are imported lazily through this
backend-neutral seam. Importing :mod:`cudnn` still does not load CuTeDSL or
initialize CUDA.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ._config import ResolvedMoeEpConfig
from ._contracts import _ForwardCall
from ._types import MoeTensor


class MoeEpBackend(Protocol):
    """Instance-local backend created lazily for one static ``MoeEp`` config."""

    @property
    def resolved_config(self) -> ResolvedMoeEpConfig:
        """Return the exact resolved config generation owned by this backend."""

    @property
    def device(self) -> torch.device:
        """Return the concrete CUDA device owned by this backend."""

    def forward(self, request: _ForwardCall) -> MoeTensor:
        """Execute one already-validated forward request."""

    def close(self) -> None:
        """Release backend-owned resources."""


class BackendUnavailableError(RuntimeError):
    """The requested supported path has no executable runtime backend yet."""


def validate_config(config: ResolvedMoeEpConfig) -> None:
    """Run the selected backend's static capability gate lazily."""

    from ._megamoe_backend._capability import validate_config as validate

    validate(config)


def validate_request(
    config: ResolvedMoeEpConfig,
    request: _ForwardCall,
) -> None:
    """Run the selected backend's per-request capability gate lazily."""

    from ._megamoe_backend._capability import validate_request as validate

    validate(config, request)


def create_backend(
    config: ResolvedMoeEpConfig,
    device: torch.device,
) -> MoeEpBackend:
    """Create the default backend without an allocation-only fallback."""

    from ._megamoe_backend.mxfp8._backend import Mxfp8Backend

    return Mxfp8Backend(config, device)


__all__ = [
    "BackendUnavailableError",
    "MoeEpBackend",
    "create_backend",
    "validate_config",
    "validate_request",
]
