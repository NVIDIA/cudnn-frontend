# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Common sample-signature helpers for grouped GEMM JAX APIs."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .._jax.api_base import ApiBaseJax, JaxTensorDesc


def immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable copy suitable for persistent JAX API state."""

    return MappingProxyType(dict(values))


def check_call_signatures(
    api: ApiBaseJax,
    expected: Mapping[str, JaxTensorDesc | None],
    values: Mapping[str, Any],
) -> None:
    """Validate invocation metadata against construction-time descriptors."""

    for name, expected_desc in expected.items():
        api.check_optional_tensor_signature(values[name], expected_desc, name=name)


__all__ = ["check_call_signatures", "immutable_mapping"]
