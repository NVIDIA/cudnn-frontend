# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""The routed MXFP8 SwiGLU backward of one 128 x 128 tile.

The tile's 256 threads take two per row (``thread % 128`` the row, ``thread / 128`` the half), each
two 32-element blocks; the blocks a thread owns and their column rotation are fixed by its row (see
``compute``), so each thread reads and overwrites only its own words.

``compute`` is the backward with a BF16 Gate / Up context.  Its shared-memory operands are unswizzled
row-major 128 x 128 BF16 tiles (256 B rows): ``d_hidden``, ``gate`` and ``up``.  It applies the element
backward (``glu_backward``), writes BF16 d_gate / d_up in place over gate / up (each thread over the
words it read; the transposed quantization reads them there), quantizes d_gate / d_up in registers
and stores the e4m3 rows straight to global (the row's 64 bytes, back in column order) and the scale
bytes to a staging pair of 512 B tiles.  ``compute_e4m3_context`` is the backward of a
``gate_up_context="mxfp8"`` forward: it dequantizes e4m3 Gate / Up tiles (128 B rows) with their
scale tiles and writes the e4m3 outputs back in place.  ``transpose`` quantizes the transposed
d_gate / d_up, one ``quantize_tile`` call per warpgroup.  The caller owns the block syncs, the scale
and transposed stores and the proxy fence before the readiness release.

Private device module of the MXFP8 backward.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint16, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ._mxfp8_bwd_math import glu_backward, router_partial
from ._mxfp8_quant import (
    BLOCK_ELEMENTS,
    PAIRS_PER_BLOCK,
    PAYLOAD_WORDS,
    QUANT_TILE,
    cvt_bf16x2,
    dequantize_block,
    quantize_pairs,
    quantize_tile,
)

FP8_TILE_BYTES = QUANT_TILE * QUANT_TILE
BF16_TILE_BYTES = 2 * FP8_TILE_BYTES
SF_TILE_BYTES = 512


@dsl_user_op
def _select_b32(take_first: Int32, first: Int32, second: Int32, *, loc=None, ip=None) -> Int32:
    """``selp.b32``: ``first`` when ``take_first`` is non-zero, else ``second``."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(take_first).ir_value(loc=loc, ip=ip), Int32(first).ir_value(loc=loc, ip=ip), Int32(second).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}",
            "=r,r,r,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_global_v4(address: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None) -> None:
    """One 16 B ``st.global.v4.b32`` (``address`` 16-byte aligned)."""
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)] + [Int32(word).ir_value(loc=loc, ip=ip) for word in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _store_block_in_order(address: Int64, payload: cute.Tensor, rotation: Int32) -> None:
    """A block's eight e4m3 words, ``payload[k]`` holding column chunk ``(rotation + k) % 8``, stored in column
    order at ``address`` (32 B): chunk ``n`` is ``payload[(n - rotation) % 8]``, three conditional rotations by
    1, 2 and 4 words (no register indexing by a runtime value)."""
    words = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
    rotated = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
    for n in cutlass.range_constexpr(PAYLOAD_WORDS):
        words[n] = payload[n]
    for step in cutlass.range_constexpr(3):
        amount = 1 << step
        take = rotation & Int32(amount)
        for n in cutlass.range_constexpr(PAYLOAD_WORDS):
            rotated[n] = _select_b32(take, words[(n - amount) % PAYLOAD_WORDS], words[n])
        for n in cutlass.range_constexpr(PAYLOAD_WORDS):
            words[n] = rotated[n]
    _store_global_v4(address, words[0], words[1], words[2], words[3])
    _store_global_v4(address + Int64(16), words[4], words[5], words[6], words[7])


@cute.jit
def compute(
    d_hidden: cute.Pointer,
    gate: cute.Pointer,
    up: cute.Pointer,
    d_gate_row: Int64,
    d_up_row: Int64,
    d_scales: cute.Pointer,
    thread: Int32,
    inverse_weight: Float32,
    limit: Float32,
    clamped: cutlass.Constexpr[bool],
    situ: cutlass.Constexpr[bool] = False,
    situ_beta1=None,
    situ_beta2=None,
) -> Float32:
    """One thread's 64 elements of the tile; returns its router-weight partial.
    ``situ_beta1`` / ``situ_beta2``: Float32 with ``situ``, None (unread) otherwise.

    Row ``r = thread % 128`` holds blocks ``2p`` and ``2p + 1`` for ``p = ((r >> 1) & 1) ^ half``,
    the ``j``-th of them block ``2p + ((r + j) & 1)``; word ``k`` of a block sits at column
    ``block * 32 + ((r / 4 + k) % 8) * 4``.
    ``d_gate_row`` / ``d_up_row``: global address of row ``r``'s first column of this tile in the
    e4m3 d_gate / d_up outputs; ``d_scales``: this tile's d_gate then d_up 512 B scale tiles.
    """
    row = thread & Int32(QUANT_TILE - 1)
    half = thread >> Int32(7)
    gate_words = cute.make_tensor(cute.recast_ptr(gate, dtype=Uint32), cute.make_layout((BF16_TILE_BYTES // 4,)))
    up_words = cute.make_tensor(cute.recast_ptr(up, dtype=Uint32), cute.make_layout((BF16_TILE_BYTES // 4,)))
    d_hidden_words = cute.make_tensor(cute.recast_ptr(d_hidden, dtype=Uint32), cute.make_layout((BF16_TILE_BYTES // 4,)))
    # The row's packed scale word: byte (row % 32) * 16 + (row / 32) * 4 of the 512 B tile.
    scale_word = (row & Int32(31)) * Int32(4) + (row >> Int32(5))
    block_pair = ((row >> Int32(1)) & Int32(1)) ^ half
    rotation = row >> Int32(2)

    gate_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    up_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    d_hidden_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    d_gate_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    d_up_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    d_gate_payload = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
    d_up_payload = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
    partial = Float32(0.0)
    d_gate_scale_pair = Int32(0)
    d_up_scale_pair = Int32(0)
    for j in cutlass.range(2, unroll=1):
        parity = (row + j) & Int32(1)
        block = block_pair * Int32(2) + parity
        for k in cutlass.range_constexpr(PAYLOAD_WORDS):
            word = (row << Int32(6)) + (((block << Int32(5)) + (((rotation + Int32(k)) & Int32(7)) << Int32(2))) >> Int32(1))
            gate_pairs[2 * k] = gate_words[word]
            gate_pairs[2 * k + 1] = gate_words[word + Int32(1)]
            up_pairs[2 * k] = up_words[word]
            up_pairs[2 * k + 1] = up_words[word + Int32(1)]
            d_hidden_pairs[2 * k] = d_hidden_words[word]
            d_hidden_pairs[2 * k + 1] = d_hidden_words[word + Int32(1)]
        for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
            bits = d_hidden_pairs[pair]
            d_hidden_low = (bits << Uint32(16)).bitcast(Float32)
            d_hidden_high = (bits & Uint32(0xFFFF0000)).bitcast(Float32)
            gate_low = (gate_pairs[pair] << Uint32(16)).bitcast(Float32)
            gate_high = (gate_pairs[pair] & Uint32(0xFFFF0000)).bitcast(Float32)
            up_low = (up_pairs[pair] << Uint32(16)).bitcast(Float32)
            up_high = (up_pairs[pair] & Uint32(0xFFFF0000)).bitcast(Float32)
            d_gate_low, d_up_low, hidden_low = glu_backward(gate_low, up_low, d_hidden_low, limit, clamped, situ, situ_beta1, situ_beta2)
            d_gate_high, d_up_high, hidden_high = glu_backward(gate_high, up_high, d_hidden_high, limit, clamped, situ, situ_beta1, situ_beta2)
            d_gate_pairs[pair] = cvt_bf16x2(d_gate_low, d_gate_high)
            d_up_pairs[pair] = cvt_bf16x2(d_up_low, d_up_high)
            partial = router_partial(partial, inverse_weight, d_hidden_low, d_hidden_high, hidden_low, hidden_high)
        # Block 2p + parity's scale byte goes to byte ``parity`` of the pair.
        d_gate_scale_pair = d_gate_scale_pair | (quantize_pairs(d_gate_pairs, d_gate_payload) << (parity << Int32(3)))
        d_up_scale_pair = d_up_scale_pair | (quantize_pairs(d_up_pairs, d_up_payload) << (parity << Int32(3)))
        # BF16 d_gate / d_up over the words this thread read (the transposed quantization's input).
        for k in cutlass.range_constexpr(PAYLOAD_WORDS):
            word = (row << Int32(6)) + (((block << Int32(5)) + (((rotation + Int32(k)) & Int32(7)) << Int32(2))) >> Int32(1))
            gate_words[word] = d_gate_pairs[2 * k]
            gate_words[word + Int32(1)] = d_gate_pairs[2 * k + 1]
            up_words[word] = d_up_pairs[2 * k]
            up_words[word + Int32(1)] = d_up_pairs[2 * k + 1]
        block_bytes = Int64(block) * Int64(BLOCK_ELEMENTS)
        _store_block_in_order(d_gate_row + block_bytes, d_gate_payload, rotation)
        _store_block_in_order(d_up_row + block_bytes, d_up_payload, rotation)
    # Two scale bytes per gradient: halfword p of the row's word in this tile's staging scale tiles.
    gate_scale_halves = cute.make_tensor(cute.recast_ptr(d_scales, dtype=Uint16), cute.make_layout((256,)))
    up_scale_halves = cute.make_tensor(cute.recast_ptr(d_scales + SF_TILE_BYTES, dtype=Uint16), cute.make_layout((256,)))
    gate_scale_halves[(scale_word << Int32(1)) + block_pair] = Uint16(d_gate_scale_pair)
    up_scale_halves[(scale_word << Int32(1)) + block_pair] = Uint16(d_up_scale_pair)
    return partial


@cute.jit
def compute_e4m3_context(
    d_hidden: cute.Pointer,
    gate: cute.Pointer,
    up: cute.Pointer,
    gate_scales: cute.Pointer,
    up_scales: cute.Pointer,
    d_up_staging: cute.Pointer,
    thread: Int32,
    inverse_weight: Float32,
    limit: Float32,
    clamped: cutlass.Constexpr[bool],
    situ: cutlass.Constexpr[bool] = False,
    situ_beta1=None,
    situ_beta2=None,
) -> Float32:
    """``gate_up_context="mxfp8"``: ``gate`` / ``up`` are e4m3 tiles (128 B rows) with their
    512 B scale tiles, dequantized here; the e4m3 d_gate / d_up go back over gate / up, the scale bytes over
    the input scale tiles, BF16 d_gate over d_hidden and BF16 d_up over ``d_up_staging`` (the transposed
    quantization's inputs).  One thread's 64 elements of the tile; returns its router-weight partial.
    ``situ_beta1`` / ``situ_beta2``: Float32 with ``situ``, None (unread) otherwise.

    Row ``r = thread % 128`` holds blocks ``2p`` and ``2p + 1`` for ``p = ((r >> 1) & 1) ^ half``,
    the ``j``-th of them block ``2p + ((r + j) & 1)``; word ``k`` of a block sits at column
    ``block * 32 + ((r / 4 + k) % 8) * 4``.
    """
    row = thread & Int32(QUANT_TILE - 1)
    half = thread >> Int32(7)
    gate_words = cute.make_tensor(cute.recast_ptr(gate, dtype=Int32), cute.make_layout((FP8_TILE_BYTES // 4,)))
    up_words = cute.make_tensor(cute.recast_ptr(up, dtype=Int32), cute.make_layout((FP8_TILE_BYTES // 4,)))
    d_hidden_words = cute.make_tensor(cute.recast_ptr(d_hidden, dtype=Uint32), cute.make_layout((BF16_TILE_BYTES // 4,)))
    d_up_words = cute.make_tensor(cute.recast_ptr(d_up_staging, dtype=Uint32), cute.make_layout((BF16_TILE_BYTES // 4,)))
    # The row's packed scale word: byte (row % 32) * 16 + (row / 32) * 4 of the 512 B tile.
    scale_word = (row & Int32(31)) * Int32(4) + (row >> Int32(5))
    gate_scale_words = cute.make_tensor(cute.recast_ptr(gate_scales, dtype=Int32), cute.make_layout((128,)))
    up_scale_words = cute.make_tensor(cute.recast_ptr(up_scales, dtype=Int32), cute.make_layout((128,)))
    gate_scale = gate_scale_words[scale_word]
    up_scale = up_scale_words[scale_word]
    block_pair = ((row >> Int32(1)) & Int32(1)) ^ half
    rotation = row >> Int32(2)

    gate_payload = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
    up_payload = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
    d_hidden_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    gate_values = cute.make_rmem_tensor((BLOCK_ELEMENTS,), Float32)
    up_values = cute.make_rmem_tensor((BLOCK_ELEMENTS,), Float32)
    d_gate_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    d_up_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    d_gate_payload = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
    d_up_payload = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
    partial = Float32(0.0)
    d_gate_scale_pair = Int32(0)
    d_up_scale_pair = Int32(0)
    for j in cutlass.range(2, unroll=1):
        parity = (row + j) & Int32(1)
        block = block_pair * Int32(2) + parity
        for k in cutlass.range_constexpr(PAYLOAD_WORDS):
            column = (block << Int32(5)) + (((rotation + Int32(k)) & Int32(7)) << Int32(2))
            gate_payload[k] = gate_words[(row << Int32(5)) + (column >> Int32(2))]
            up_payload[k] = up_words[(row << Int32(5)) + (column >> Int32(2))]
            d_hidden_pairs[2 * k] = d_hidden_words[(row << Int32(6)) + (column >> Int32(1))]
            d_hidden_pairs[2 * k + 1] = d_hidden_words[(row << Int32(6)) + (column >> Int32(1)) + Int32(1)]
        shift = block << Int32(3)
        dequantize_block(gate_payload, (gate_scale >> shift) & Int32(0xFF), gate_values)
        dequantize_block(up_payload, (up_scale >> shift) & Int32(0xFF), up_values)
        for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
            bits = d_hidden_pairs[pair]
            d_hidden_low = (bits << Uint32(16)).bitcast(Float32)
            d_hidden_high = (bits & Uint32(0xFFFF0000)).bitcast(Float32)
            d_gate_low, d_up_low, hidden_low = glu_backward(
                gate_values[2 * pair], up_values[2 * pair], d_hidden_low, limit, clamped, situ, situ_beta1, situ_beta2
            )
            d_gate_high, d_up_high, hidden_high = glu_backward(
                gate_values[2 * pair + 1], up_values[2 * pair + 1], d_hidden_high, limit, clamped, situ, situ_beta1, situ_beta2
            )
            d_gate_pairs[pair] = cvt_bf16x2(d_gate_low, d_gate_high)
            d_up_pairs[pair] = cvt_bf16x2(d_up_low, d_up_high)
            partial = router_partial(partial, inverse_weight, d_hidden_low, d_hidden_high, hidden_low, hidden_high)
        # Block 2p + parity's scale byte goes to byte ``parity`` of the pair.
        d_gate_scale_pair = d_gate_scale_pair | (quantize_pairs(d_gate_pairs, d_gate_payload) << (parity << Int32(3)))
        d_up_scale_pair = d_up_scale_pair | (quantize_pairs(d_up_pairs, d_up_payload) << (parity << Int32(3)))
        for k in cutlass.range_constexpr(PAYLOAD_WORDS):
            column = (block << Int32(5)) + (((rotation + Int32(k)) & Int32(7)) << Int32(2))
            gate_words[(row << Int32(5)) + (column >> Int32(2))] = d_gate_payload[k]
            up_words[(row << Int32(5)) + (column >> Int32(2))] = d_up_payload[k]
            d_hidden_words[(row << Int32(6)) + (column >> Int32(1))] = d_gate_pairs[2 * k]
            d_hidden_words[(row << Int32(6)) + (column >> Int32(1)) + Int32(1)] = d_gate_pairs[2 * k + 1]
            d_up_words[(row << Int32(6)) + (column >> Int32(1))] = d_up_pairs[2 * k]
            d_up_words[(row << Int32(6)) + (column >> Int32(1)) + Int32(1)] = d_up_pairs[2 * k + 1]
    # Two scale bytes per gradient over the input scale tiles: halfword p of the row's word.
    gate_scale_halves = cute.make_tensor(cute.recast_ptr(gate_scales, dtype=Uint16), cute.make_layout((256,)))
    up_scale_halves = cute.make_tensor(cute.recast_ptr(up_scales, dtype=Uint16), cute.make_layout((256,)))
    gate_scale_halves[(scale_word << Int32(1)) + block_pair] = Uint16(d_gate_scale_pair)
    up_scale_halves[(scale_word << Int32(1)) + block_pair] = Uint16(d_up_scale_pair)
    return partial


@cute.jit
def transpose(
    d_gate_bf16: cute.Pointer,
    d_up_bf16: cute.Pointer,
    d_gate_t: cute.Pointer,
    d_gate_sc_t: cute.Pointer,
    d_up_t: cute.Pointer,
    d_up_sc_t: cute.Pointer,
    thread: Int32,
) -> None:
    """The transposed quantization (``quantize_tile``) of the BF16 d_gate tile by threads 0-127
    and of the BF16 d_up tile by threads 128-255."""
    if thread < Int32(QUANT_TILE):
        quantize_tile(d_gate_bf16, d_gate_t, d_gate_sc_t, d_gate_t, d_gate_sc_t, thread, QUANT_TILE, False, True)
    else:
        quantize_tile(d_up_bf16, d_up_t, d_up_sc_t, d_up_t, d_up_sc_t, thread - Int32(QUANT_TILE), QUANT_TILE, False, True)


__all__ = ["compute", "compute_e4m3_context", "transpose"]
