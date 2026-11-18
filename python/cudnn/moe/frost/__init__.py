# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.moe.frost: the FROST (CuTe DSL) MoE kernels -- the shared-expert decode pair ``SharedExpertDecode`` (bf16 shared expert next to
a routed MoE at decode-time token counts, two launches, SM100-class parts)."""

from typing import Any

_LAZY_EXPORTS = {
    "SharedExpertDecode": ("cudnn.moe.frost.shared_expert_decode", "SharedExpertDecode"),
    "Geometry": ("cudnn.moe.frost.kernel.shared_expert_decode_config", "Geometry"),
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
