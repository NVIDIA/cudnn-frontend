# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact widening of block-scaled KV codes to BF16, one 32-bit code word at a time.

A packed KV cache (see ``_kv_formats``) is read by widening its block-scaled
codes to the BF16 values the BF16 kernel would have gathered, then running
the unchanged BF16 MMAs. Each helper reproduces the torch reference bit for
bit:

- MXFP8: ``bf16(fp32(e4m3(code)) * fp32(2 ** (sf - 127)))``. Every E4M3 code
  and every UE8M0 scale is exact in BF16, so one ``mul.rn.bf16x2`` of the two
  rounds the exact product once to BF16, as the reference does (infinite
  above the BF16 range, subnormal below its normal range).
- NVFP4: ``bf16(e2m1(code) * e4m3(sf))``. Every product of an E2M1 value and
  a non-NaN E4M3 scale has at most six significant bits and a magnitude in
  ``[2**-10, 2688]``, so the BF16 multiply is exact.

NaN codes or scales (E4M3 ``0x7F`` / ``0xFF``, UE8M0 ``0xFF``) widen to a NaN
of unspecified payload. The conversions are ``cvt.rn.bf16x2.e4m3x2``,
``cvt.rn.bf16x2.e2m1x2`` and ``cvt.rn.bf16x2.ue8m0x2`` (sm_100a and the
sm_100f family); each is exact, so the multiply is the only rounding.

Byte order: a code word holds consecutive channels with the lowest channel in
the lowest byte (for E2M1, channel ``2i`` in the low nibble of byte ``i``); a
BF16 output word holds two consecutive channels with the lower one in the low
half, as a ``bf16x2`` store of the widened row expects.

Scales are widened two at a time: ``mxfp8_scale_pair`` / ``nvfp4_scale_pair``
pick two scale bytes with one ``prmt`` (its selector's low two nibbles; byte
indices 0-3 address the first word, 4-7 the second) and widen both with one
``cvt`` into a ``bf16x2`` word, the first byte in the low half. A widening
helper multiplies its codes by half ``half`` of that word.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx


@cute.jit
def mxfp8_scale_pair(scale_word: cutlass.Int32, selector: cutlass.Int32) -> cutlass.Int32:
    """Two UE8M0 scale bytes of ``scale_word`` (``prmt`` byte indices 0-3 in the low two nibbles of ``selector``) as one ``bf16x2`` word.

    Each half holds ``2 ** (b - 127)``: ``0xFF`` is NaN; byte 0 is the BF16 subnormal ``2 ** -127``.
    """
    return inline_ptx(
        "{ .reg .b16 s; .reg .b32 w;\n" "prmt.b32 w, $1, 0, $2;\n" "cvt.u16.u32 s, w;\n" "cvt.rn.bf16x2.ue8m0x2 $0, s; }",
        write_only_types=[cutlass.Int32],
        read_only_args=[scale_word, selector],
    )


def _broadcast_half(half: int) -> str:
    """PTX that broadcasts half ``half`` of the scale word ``$s`` into ``sc``."""
    return "mov.b32 {lo, hi}, $s;\n" + ("mov.b32 sc, {lo, lo};\n" if half == 0 else "mov.b32 sc, {hi, hi};\n")


@cute.jit
def widen_mxfp8_word(codes: cutlass.Int32, scale: cutlass.Int32, half: cutlass.Constexpr[int]):
    """Four E4M3 codes and their scale -> two ``bf16x2`` words (channels 0-1, then 2-3).

    Half ``half`` of the ``bf16x2`` word ``scale`` (``mxfp8_scale_pair``) scales the codes.
    """
    return inline_ptx(
        (
            "{ .reg .b16 c01, c23, lo, hi; .reg .b32 p01, p23, sc;\n" + _broadcast_half(half).replace("$s", "$3") + "mov.b32 {c01, c23}, $2;\n"
            "cvt.rn.bf16x2.e4m3x2 p01, c01;\n"
            "cvt.rn.bf16x2.e4m3x2 p23, c23;\n"
            "mul.rn.bf16x2 $0, p01, sc;\n"
            "mul.rn.bf16x2 $1, p23, sc; }"
        ),
        write_only_types=[cutlass.Int32, cutlass.Int32],
        read_only_args=[codes, scale],
    )


@cute.jit
def nvfp4_scale_pair(word_a: cutlass.Int32, word_b: cutlass.Int32, selector: cutlass.Int32) -> cutlass.Int32:
    """Two E4M3 scale bytes of ``word_a`` / ``word_b`` (``prmt`` byte indices 0-3 / 4-7 in the low two nibbles of ``selector``) as one ``bf16x2`` word.

    ``0x7F`` / ``0xFF`` are NaN.
    """
    return inline_ptx(
        "{ .reg .b16 s; .reg .b32 w;\n" "prmt.b32 w, $1, $2, $3;\n" "cvt.u16.u32 s, w;\n" "cvt.rn.bf16x2.e4m3x2 $0, s; }",
        write_only_types=[cutlass.Int32],
        read_only_args=[word_a, word_b, selector],
    )


@cute.jit
def widen_nvfp4_word(codes: cutlass.Int32, scale: cutlass.Int32, half: cutlass.Constexpr[int]):
    """Eight E2M1 codes and their scale -> four ``bf16x2`` words (channels 0-1, 2-3, 4-5, 6-7).

    Half ``half`` of the ``bf16x2`` word ``scale`` (``nvfp4_scale_pair``) scales the codes.
    """
    return inline_ptx(
        (
            "{ .reg .b8 b0, b1, b2, b3; .reg .b16 lo, hi; .reg .b32 h, sc;\n" + _broadcast_half(half).replace("$s", "$5") + "mov.b32 {b0, b1, b2, b3}, $4;\n"
            "cvt.rn.bf16x2.e2m1x2 h, b0;\n"
            "mul.rn.bf16x2 $0, h, sc;\n"
            "cvt.rn.bf16x2.e2m1x2 h, b1;\n"
            "mul.rn.bf16x2 $1, h, sc;\n"
            "cvt.rn.bf16x2.e2m1x2 h, b2;\n"
            "mul.rn.bf16x2 $2, h, sc;\n"
            "cvt.rn.bf16x2.e2m1x2 h, b3;\n"
            "mul.rn.bf16x2 $3, h, sc; }"
        ),
        write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
        read_only_args=[codes, scale],
    )
