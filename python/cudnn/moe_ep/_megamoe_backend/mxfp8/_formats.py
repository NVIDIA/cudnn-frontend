# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Public MoE format names to Rubin combine-wire encodings."""

from __future__ import annotations

from ..._types import MoeFormat, parse_format

_COMBINE_WIRE_FORMATS = {
    MoeFormat.BF16: "bf16",
    MoeFormat.MXFP8: "32e4m3xe8m0",
}


def combine_wire_format(value: MoeFormat | str) -> str:
    """Return the kernel encoding for one public combine format."""

    return _COMBINE_WIRE_FORMATS[parse_format(value)]


__all__ = ["combine_wire_format"]
