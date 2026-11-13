# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.attention_gate.frost: the FROST (CuTe DSL) attention-gate kernels -- the gate + quantize decode step ``AttentionGateQuantDecode``
(the sigmoid output gate of an attention block fused with the static E4M3 quantization of the following projection's input, bf16 in,
decode-time row counts, one launch, SM100-class parts)."""

from typing import Any

_LAZY_EXPORTS = {
    "AttentionGateQuantDecode": ("cudnn.attention_gate.frost.gate_quant_decode", "AttentionGateQuantDecode"),
    "Cfg": ("cudnn.attention_gate.frost.kernel.gate_quant_decode_config", "Cfg"),
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
