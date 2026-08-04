# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.linear_attention: the linear-attention operation family (GDN, KDA, GDN-2)."""

from typing import Any

_LAZY_EXPORTS = {
    "gated_delta_net": ("cudnn.linear_attention.ops", "gated_delta_net"),
    "kimi_delta_attention": ("cudnn.linear_attention.ops", "kimi_delta_attention"),
    "gated_delta_net_v2": ("cudnn.linear_attention.ops", "gated_delta_net_v2"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
