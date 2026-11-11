# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The attention output-gate operation family: the gate + quantize decode step (``cudnn.attention_gate.frost``).  Lazy exports: nothing
here imports torch or the CuTe DSL at ``import cudnn`` time."""

from typing import Any

_LAZY_EXPORTS = {
    "AttentionGateQuantDecode": ("cudnn.attention_gate.frost.gate_quant_decode", "AttentionGateQuantDecode"),
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
