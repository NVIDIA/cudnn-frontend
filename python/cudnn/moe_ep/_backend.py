# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lightweight private backend seam for :class:`cudnn.moe_ep.MoeEp`.

Capability policy and the executable factory are imported lazily through this
backend-neutral seam. Importing :mod:`cudnn` still does not load CuTeDSL or
initialize CUDA.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ._contracts import ForwardConfig, ValidatedForwardRequest
from ._types import MoeTensor


class MoeEpBackend(Protocol):
    """Instance-local backend created lazily for one static ``MoeEp`` config."""

    def forward(self, request: ValidatedForwardRequest) -> MoeTensor:
        """Execute one already-validated forward request."""

    def close(self) -> None:
        """Release backend-owned resources."""


class BackendUnavailableError(RuntimeError):
    """The requested supported path has no executable runtime backend yet."""


def validate_config(config: ForwardConfig) -> None:
    """Run the selected backend's static capability gate lazily."""

    from ._megamoe_backend._capability import validate_config as validate

    validate(config)


def validate_request(request: ValidatedForwardRequest) -> None:
    """Run the selected backend's per-request capability gate lazily."""

    from ._megamoe_backend._capability import validate_request as validate

    validate(request)


def create_backend(
    config: ForwardConfig,
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
