# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed KV cache formats of the sparse attention forward (``kv_format``).

A packed cache keeps each token's K = V row in one run of bytes: the quantized
channels, their block scales and, at ``D_qk = 576``, the 64 channels only K
reads. It is passed as the ``torch.uint8`` tensor of those rows, shape
``(S_kv, record_bytes)``, together with the name of its format; a cache with
padded rows is passed as that view of its buffer. The row pitch,
``stride(0)``, is a multiple of 16 bytes and at least ``record_bytes``.
Without a format name the pool is a typed ``(S_kv, D_qk)`` tensor: Q's
dtype, or ``float8_e4m3fn`` with a per-tensor descale.

One name fixes one byte layout and one rule for recovering channel values; a
variant that changes either is a new name, and names are only ever appended.
Byte offsets are within one row; V reads channels ``[0, 512)``.

``mxfp8_528`` (``D_qk = 512``)
    ``[0, 512)``: E4M3 codes in channel order; ``[512, 528)``: one UE8M0
    scale per 32 channels. ``value[c] = e4m3(code[c]) * 2 ** (sf[c // 32] - 127)``.

``nvfp4_288`` (``D_qk = 512``)
    ``[0, 256)``: E2M1 codes, two per byte, channel ``2i`` in the low nibble
    of byte ``i``; ``[256, 288)``: one E4M3 scale per 16 channels.
    ``value[c] = e2m1(code[c]) * e4m3(sf[c // 16])``.

``fp8_bf16rope_656`` (``D_qk = 576``)
    ``[0, 512)``: E4M3 codes of channels ``[0, 512)``; ``[512, 528)``: one
    FP32 scale per 128 of those channels; ``[528, 656)``: channels
    ``[512, 576)`` as BF16. ``value[c] = e4m3(code[c]) * sf[c // 128]`` for
    ``c < 512``, else the BF16 value.

``nvfp4_fp8rope_352`` (``D_qk = 576``)
    ``[0, 256)``: E2M1 codes of channels ``[0, 512)``, as in ``nvfp4_288``;
    ``[256, 320)``: channels ``[512, 576)`` as unscaled E4M3; ``[320, 352)``:
    one E4M3 scale per 16 of channels ``[0, 512)``, the scale of block ``b``
    at byte ``320 + 8 * (b % 4) + b // 4``.
    ``value[c] = e2m1(code[c]) * e4m3(sf[c // 16])`` for ``c < 512``, else the
    E4M3 value.

``fp8_528`` (``D_qk = 512``)
    ``fp8_bf16rope_656`` without its last 64 channels: ``[0, 512)``: E4M3
    codes; ``[512, 528)``: one FP32 scale per 128 channels.
    ``value[c] = e4m3(code[c]) * sf[c // 128]``. It has the width of
    ``mxfp8_528`` but not its layout, so a format is never inferred from a
    width.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KvFormat:
    """One packed KV cache format; the module docstring gives each layout."""

    name: str
    head_dim_qk: int
    record_bytes: int


KV_FORMATS: dict[str, KvFormat] = {
    fmt.name: fmt
    for fmt in (
        KvFormat("mxfp8_528", head_dim_qk=512, record_bytes=528),
        KvFormat("nvfp4_288", head_dim_qk=512, record_bytes=288),
        KvFormat("fp8_bf16rope_656", head_dim_qk=576, record_bytes=656),
        KvFormat("nvfp4_fp8rope_352", head_dim_qk=576, record_bytes=352),
        KvFormat("fp8_528", head_dim_qk=512, record_bytes=528),
    )
}
