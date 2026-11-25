# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.row_quantize.frost: the FROST (CuTe DSL) row quantization decode steps -- ``RowQuantizeFp4Decode`` (the block-scaled FP4
(NVFP4) quantization of bf16 rows under a global scale in one launch: E2M1 codes with one E4M3 block scale per sixteen elements in the
128 x 4 scale layout; decode-time row counts; SM100-class parts) and ``RowQuantizeFp8Decode`` (the dynamic FP8 quantization of bf16
rows in one launch, each row under its own scale: E4M3 codes and one fp32 scale per row, the row maximum over 448 with a floor;
decode-time row counts; SM100-class parts).  ``Cfg`` is the FP4 step's config; each step's config is in its ``kernel`` module.
Lazy exports with no DSL gate at attribute access: an entry is constructible, and declines by type, on a host without the DSL."""

from typing import Any

_LAZY_EXPORTS = {
    "RowQuantizeFp4Decode": ("cudnn.row_quantize.frost.row_quantize_fp4_decode", "RowQuantizeFp4Decode"),
    "Cfg": ("cudnn.row_quantize.frost.kernel.row_quantize_fp4_decode_config", "Cfg"),
    "RowQuantizeFp8Decode": ("cudnn.row_quantize.frost.row_quantize_fp8_decode", "RowQuantizeFp8Decode"),
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
