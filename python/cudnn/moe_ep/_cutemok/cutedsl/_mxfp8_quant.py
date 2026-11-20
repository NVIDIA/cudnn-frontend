# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""MXFP8 block quantization: the device code shared by every quantization point of the package.

One e8m0 scale covers 32 contiguous elements along K; the payload is e4m3.  The bytes are defined by
the steps below; every quantization point keeps these instructions and their order, so all of them
produce identical bytes:

* the amax is taken over the BF16-rounded values (|v| and max are exact in both BF16 and FP32, so
  they may be held as FP32),
* the e8m0 exponent comes from ``cvt.rp.satfinite.ue8m0x2.f32`` -- round toward +infinity, saturate
  to finite (https://arxiv.org/pdf/2506.08027),
* the reciprocal is assembled from the exponent bits, never divided,
* the payload comes from ``cvt.rn.satfinite.e4m3x2.f32``.

``quantize_tile`` is the 128-thread routine over a shared-memory tile; its bank-spreading rotation
permutes which element a thread touches first and changes no byte of the result.  The communication
streams' per-warp passes (``_persistent_mxfp8_fwd_mega``) build on the bank-conflict-free
shared-memory accessors below (16 B loads, ``ldmatrix.trans`` for the transposed pass).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Uint16, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

BLOCK_ELEMENTS = 32  # elements per e8m0 scale
QUANT_TILE = 128  # rows and columns of a quantization tile
BLOCKS_PER_TILE_ROW = QUANT_TILE // BLOCK_ELEMENTS  # 4 scale bytes per tile row
PAYLOAD_WORDS = BLOCK_ELEMENTS // 4  # one block packs into 8 int32 words
PAIRS_PER_BLOCK = BLOCK_ELEMENTS // 2  # packed BF16 pairs per block
SCALE_TILE_ROWS = 32  # the 512 B scale tile: 128 rows x 4 bytes, interleaved
SCALE_TILE_WORDS = 4  # the same tile seen as int32: 32 x 4

# 1 / 448 (e4m3's maximum): the float32 0x3b124925.
E4M3_MAX_RECIPROCAL = 0.002232142857
SCALE_FLOOR = 0.000000000001  # lower bound of a block's scale
# (254 - scale_byte) << 23: the float32 bits of 2^-(scale_byte - 127).
RECIPROCAL_BIAS = 254


@dsl_user_op
def cvt_e8m0_scale_byte(value: Float32, *, loc=None, ip=None) -> Int32:
    """The e8m0 exponent byte of ``value`` (``cvt.rp.satfinite.ue8m0x2.f32``).

    Both halves of the x2 instruction take the same input, and the 16-bit result is
    zero-extended inside the asm so the caller only ever sees an i32.
    """
    scale = Float32(value).ir_value(loc=loc, ip=ip)
    packed = llvm.inline_asm(
        T.i32(),
        [scale, scale],
        "{\n.reg .b16 s;\ncvt.rp.satfinite.ue8m0x2.f32 s, $1, $2;\ncvt.u32.u16 $0, s;\n}",
        "=r,f,f",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(packed) & Int32(0xFF)


@dsl_user_op
def cvt_e4m3x2(low: Float32, high: Float32, *, loc=None, ip=None) -> Int32:
    """Two e4m3 bytes, ``low`` in the low byte (``cvt.rn.satfinite.e4m3x2.f32``).

    PTX converts its *second* source into the low byte, hence the ``$2, $1`` operand order.
    """
    packed = llvm.inline_asm(
        T.i32(),
        [
            Float32(low).ir_value(loc=loc, ip=ip),
            Float32(high).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .b16 p;\ncvt.rn.satfinite.e4m3x2.f32 p, $2, $1;\ncvt.u32.u16 $0, p;\n}",
        "=r,f,f",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(packed)


@dsl_user_op
def cvt_bf16x2(low: Float32, high: Float32, *, loc=None, ip=None) -> Uint32:
    """Two FP32 values rounded to nearest BF16 and packed, ``low`` in the low half.

    PTX ``cvt.rn.bf16x2.f32`` puts its *first* source in the upper half.
    """
    packed = llvm.inline_asm(
        T.i32(),
        [
            Float32(high).ir_value(loc=loc, ip=ip),
            Float32(low).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Uint32(packed)


@dsl_user_op
def abs_max_bf16x2(running: Uint32, pair: Uint32, *, loc=None, ip=None) -> Uint32:
    """``max(running, |pair|)`` per half on packed BF16 pairs."""
    result = llvm.inline_asm(
        T.i32(),
        [
            Uint32(running).ir_value(loc=loc, ip=ip),
            Uint32(pair).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .b32 a;\nabs.bf16x2 a, $2;\nmax.bf16x2 $0, $1, a;\n}",
        "=r,r,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def abs_bf16x2(pair: Uint32, *, loc=None, ip=None) -> Uint32:
    """``|pair|`` per half of a packed BF16 pair."""
    result = llvm.inline_asm(
        T.i32(),
        [Uint32(pair).ir_value(loc=loc, ip=ip)],
        "abs.bf16x2 $0, $1;",
        "=r,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def store_shared_v4(address: cute.Pointer, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None) -> None:
    """One 16-byte shared-memory store; ``address`` is 16-byte aligned."""
    llvm.inline_asm(
        None,
        [address.toint(loc=loc, ip=ip).ir_value()] + [Int32(word).ir_value(loc=loc, ip=ip) for word in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mul_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a * b`` with flush-to-zero (``mul.ftz.f32``: subnormals flush to zero)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma_ftz(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """``a * b + c`` contracted, with flush-to-zero (``fma.rn.ftz.f32``)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip), Float32(c).ir_value(loc=loc, ip=ip)],
            "fma.rn.ftz.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_e4m3x2_f16x2(pair: Uint32, *, loc=None, ip=None) -> Uint32:
    """Two e4m3 bytes (low byte first) widened to a packed f16 pair, low half first
    (``cvt.rn.f16x2.e4m3x2``)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(pair).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 p;\ncvt.u16.u32 p, $1;\ncvt.rn.f16x2.e4m3x2 $0, p;\n}",
            "=r,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_f16_f32(half: Uint32, *, loc=None, ip=None) -> Float32:
    """The f16 in the low 16 bits of ``half`` as FP32 (exact)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(half).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\ncvt.f32.f16 $0, h;\n}",
            "=f,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def dequantize_block(payload: cute.Tensor, scale_byte: Int32, values: cute.Tensor) -> None:
    """Dequantize one block: ``PAYLOAD_WORDS`` e4m3 words and their e8m0 byte to
    ``BLOCK_ELEMENTS`` FP32 values, element order.

    e4m3 -> f16 -> f32 is exact; the scale is ``2^(byte - 127)`` built from its bits and
    the product is ``mul.ftz`` (flush-to-zero).
    """
    scale = (Int32(scale_byte) << Int32(23)).bitcast(Float32)
    for word in cutlass.range_constexpr(PAYLOAD_WORDS):
        bits = Uint32(payload[word])
        for half in cutlass.range_constexpr(2):
            pair = cvt_e4m3x2_f16x2(bits >> Uint32(16 * half))
            values[4 * word + 2 * half] = mul_ftz(cvt_f16_f32(pair), scale)
            values[4 * word + 2 * half + 1] = mul_ftz(cvt_f16_f32(pair >> Uint32(16)), scale)


@cute.jit
def scale_bf16x2(pair: Uint32, low_scale: Float32, high_scale: Float32) -> Uint32:
    """A packed BF16 pair with its low half times ``low_scale`` and its high half times
    ``high_scale`` (``mul.ftz``), rounded to nearest BF16."""
    low = mul_ftz((pair << Uint32(16)).bitcast(Float32), low_scale)
    high = mul_ftz((pair & Uint32(0xFFFF0000)).bitcast(Float32), high_scale)
    return cvt_bf16x2(low, high)


@cute.jit
def encode_block(amax: Float32, values: cute.Tensor, payload: cute.Tensor) -> Int32:
    """The e8m0 byte for ``amax``; ``payload`` receives the e4m3 bytes of ``values``.

    The block quantization after its amax.  ``values`` holds the BF16-rounded
    inputs as FP32 and ``payload`` receives ``PAYLOAD_WORDS`` int32 words, four e4m3
    bytes each, in element order.
    """
    scale = cute.arch.fmax(amax * Float32(E4M3_MAX_RECIPROCAL), Float32(SCALE_FLOOR))
    scale_byte = cvt_e8m0_scale_byte(scale)
    reciprocal = ((Int32(RECIPROCAL_BIAS) - scale_byte) << Int32(23)).bitcast(Float32)

    for word in cutlass.range_constexpr(PAYLOAD_WORDS):
        base = word * 4
        low = cvt_e4m3x2(values[base] * reciprocal, values[base + 1] * reciprocal)
        high = cvt_e4m3x2(values[base + 2] * reciprocal, values[base + 3] * reciprocal)
        payload[word] = low | (high << Int32(16))
    return scale_byte


@cute.jit
def quantize_pairs(pairs: cute.Tensor, payload: cute.Tensor) -> Int32:
    """Quantize one ``BLOCK_ELEMENTS`` block given as ``PAIRS_PER_BLOCK`` packed BF16 pairs.

    A pair holds element ``2k`` in its low half and ``2k + 1`` in its high half, as an
    aligned 32-bit shared-memory load returns them; widening BF16 to FP32 is the shift.
    The amax runs on the packed pairs (``abs.bf16x2`` / ``max.bf16x2``, then the max of
    the two halves), half the instructions of the FP32 form.
    """
    amax_pair = abs_bf16x2(pairs[0])
    for pair in cutlass.range_constexpr(1, PAIRS_PER_BLOCK):
        amax_pair = abs_max_bf16x2(amax_pair, pairs[pair])
    amax = cute.arch.fmax((amax_pair << Uint32(16)).bitcast(Float32), (amax_pair & Uint32(0xFFFF0000)).bitcast(Float32))
    values = cute.make_rmem_tensor((BLOCK_ELEMENTS,), Float32)
    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
        bits = pairs[pair]
        values[2 * pair] = (bits << Uint32(16)).bitcast(Float32)
        values[2 * pair + 1] = (bits & Uint32(0xFFFF0000)).bitcast(Float32)
    return encode_block(amax, values, payload)


# ---- shared-memory accessors without bank conflicts (the communication streams' per-warp passes) ----
#
# Same per-block arithmetic as quantize_tile (scale, reciprocal and conversions), so the e4m3 payloads
# and scales are byte-identical; only the shared-memory access pattern differs:
# * normal pass: each lane loads one 16 B chunk (eight BF16 of one row) and four lanes form a 32-value block
#   (two butterflies for the amax), so a load phase reads 128 contiguous bytes of one row;
# * transposed pass: ldmatrix.x4.trans with the eight addresses of a matrix stepping right (two rows x four
#   16 B chunks), so each lane owns one source column; four instructions give it the column's 32 rows (one
#   scale block, amax in-thread).  The two rows of a matrix must sit on different 16 B bank groups.


@dsl_user_op
def _ldmatrix_x4_trans(address: Int32, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16``: lanes ``8m .. 8m + 7`` give matrix ``m``'s eight
    row addresses; lane ``i`` receives, in register ``m``, column ``i / 4`` of rows ``2 (i % 4)`` (low half)
    and ``+ 1`` (high half)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), result, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def _load_shared_v4(address: Int32, *, loc=None, ip=None):
    """``ld.shared.v4.b32`` (16 B, eight BF16); a side effect, so it stays after the caller's barrier."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), result, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def _store_shared_v4_words(address: Int32, words: list, *, loc=None, ip=None) -> None:
    """``st.shared.v4.b32`` (16 B) of four words at an integer shared address (``store_shared_v4`` takes a
    pointer; the streams' per-warp passes compute byte addresses)."""
    llvm.inline_asm(
        None,
        [Int32(address).ir_value(loc=loc, ip=ip)] + [Uint32(w).ir_value(loc=loc, ip=ip) for w in words],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _abs_max_xorsign_bf16x2(running: Uint32, pair: Uint32, *, loc=None, ip=None) -> Uint32:
    """``max.xorsign.abs.bf16x2``: max(|running|, |pair|) per half in one instruction; the sign bits are garbage
    until ``& 0x7FFF7FFF`` (the magnitudes equal ``abs_max_bf16x2``'s)."""
    result = llvm.inline_asm(
        T.i32(),
        [Uint32(running).ir_value(loc=loc, ip=ip), Uint32(pair).ir_value(loc=loc, ip=ip)],
        "max.xorsign.abs.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@cute.jit
def _shuffle_xor(value: Uint32, lane_mask: cutlass.Constexpr[int]) -> Uint32:
    """Butterfly shuffle (no inline-asm side effects, so independent blocks' shuffles can interleave)."""
    return Uint32(cute.arch.shuffle_sync_bfly(Uint32(value), offset=lane_mask))


@cute.jit
def _amax_float(amax_pair: Uint32) -> Float32:
    """The larger magnitude of a BF16 pair produced by ``_abs_max_xorsign_bf16x2``, as FP32."""
    amax_pair = amax_pair & Uint32(0x7FFF7FFF)
    return cute.arch.fmax((amax_pair << Uint32(16)).bitcast(Float32), (amax_pair & Uint32(0xFFFF0000)).bitcast(Float32))


@cute.jit
def quantize_tile(
    source: cute.Pointer,
    payload: cute.Pointer,
    scales: cute.Pointer,
    payload_t: cute.Pointer,
    scales_t: cute.Pointer,
    thread: Int32,
    source_row_stride: cutlass.Constexpr[int],
    return_normal: cutlass.Constexpr[bool],
    return_transposed: cutlass.Constexpr[bool],
    split_passes: cutlass.Constexpr[bool] = False,
) -> None:
    """Quantize one 128 x 128 BF16 shared-memory tile.

    ``source`` is BF16 with ``source_row_stride`` elements per row; ``payload`` /
    ``payload_t`` receive row-major 128 x 128 e4m3 tiles (128 B rows) and ``scales`` /
    ``scales_t`` the 512 B scale tiles.  Threads 0-127 work, one tile row each; the
    others return.  The transposed pass runs first, then the normal pass.  No output
    may overlap ``source``: a thread stores its rows while the others still load
    theirs, and nothing here orders the two (no barrier).
    ``split_passes``: threads 0-127 run the transposed pass and 128-255 the normal pass
    at the same time.  Each thread starts at block ``t / 8`` and column ``t * 4`` of a
    block, and the transposed pass maps threads to rows ``(t % 64) * 2 + t / 64``, which
    spreads the accesses over the shared-memory banks.
    """
    working = thread < Int32(QUANT_TILE)
    tid = thread
    run_transposed = Boolean(True)
    run_normal = Boolean(True)
    if cutlass.const_expr(split_passes):
        working = thread < Int32(2 * QUANT_TILE)
        tid = thread & Int32(QUANT_TILE - 1)
        run_transposed = thread < Int32(QUANT_TILE)
        run_normal = thread >= Int32(QUANT_TILE)
    if working:
        source_words = cute.make_tensor(
            cute.recast_ptr(source, dtype=Uint32),
            cute.make_layout((QUANT_TILE * source_row_stride // 2,)),
        )
        source_halves = cute.make_tensor(
            cute.recast_ptr(source, dtype=Uint16),
            cute.make_layout((QUANT_TILE * source_row_stride,)),
        )
        payload_word = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
        # Indices as shifts and masks: tid is non-negative, so these equal / and % without
        # signed fix-ups.
        # A pair / word of block b sits at column b * 32 + (tid * 4 + 2k / 4k) % 32.
        element_rotation = (tid << Int32(2)) & Int32(BLOCK_ELEMENTS - 1)
        word_rotation = tid & Int32(PAYLOAD_WORDS - 1)
        first_block = tid >> Int32(3)
        if cutlass.const_expr(return_transposed):
            if run_transposed:
                # Output row r is source column r; its blocks run down 32 source rows.
                row = ((tid & Int32(63)) << Int32(1)) | (tid >> Int32(6))
                destination = cute.make_tensor(
                    cute.recast_ptr(payload_t, dtype=Int32),
                    cute.make_layout((QUANT_TILE * QUANT_TILE // 4,)),
                )
                scale_word = Int32(0)
                pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
                for step in cutlass.range(QUANT_TILE // BLOCK_ELEMENTS, unroll=1):
                    block = (step + first_block) & Int32(QUANT_TILE // BLOCK_ELEMENTS - 1)
                    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
                        column = (block << Int32(5)) + ((element_rotation + Int32(pair * 2)) & Int32(BLOCK_ELEMENTS - 1))
                        offset = column * Int32(source_row_stride) + row
                        low = Uint32(source_halves[offset])
                        high = Uint32(source_halves[offset + Int32(source_row_stride)])
                        pairs[pair] = low | (high << Uint32(16))
                    scale_word = scale_word | (quantize_pairs(pairs, payload_word) << (block << Int32(3)))
                    for word in cutlass.range_constexpr(PAYLOAD_WORDS):
                        # word (row * QUANT_TILE + column) / 4
                        destination[row * Int32(QUANT_TILE // 4) + (block << Int32(3)) + ((word_rotation + Int32(word)) & Int32(PAYLOAD_WORDS - 1))] = (
                            payload_word[word]
                        )
                store_scale_word(scales_t, row, scale_word)
        if cutlass.const_expr(return_normal):
            if run_normal:
                # One thread per row, all BLOCKS_PER_TILE_ROW blocks of it.
                normal_blocks = BLOCKS_PER_TILE_ROW
                row = tid
                destination = cute.make_tensor(
                    cute.recast_ptr(payload, dtype=Int32),
                    cute.make_layout((QUANT_TILE * QUANT_TILE // 4,)),
                )
                tile_pairs = cute.make_rmem_tensor((normal_blocks, PAIRS_PER_BLOCK), Uint32)
                pair_rotation = element_rotation >> Int32(1)
                for step in cutlass.range_constexpr(normal_blocks):
                    block = (Int32(step) + first_block) & Int32(normal_blocks - 1)
                    first_word = row * Int32(source_row_stride // 2) + (block << Int32(4))
                    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
                        # word (row * stride + column) / 2
                        tile_pairs[step, pair] = source_words[first_word + ((pair_rotation + Int32(pair)) & Int32(PAIRS_PER_BLOCK - 1))]
                scale_word = Int32(0)
                for step in cutlass.range_constexpr(normal_blocks):
                    block = (Int32(step) + first_block) & Int32(normal_blocks - 1)
                    scale_word = scale_word | (quantize_pairs(tile_pairs[step, None], payload_word) << (block << Int32(3)))
                    for word in cutlass.range_constexpr(PAYLOAD_WORDS):
                        destination[(row << Int32(5)) + (block << Int32(3)) + ((word_rotation + Int32(word)) & Int32(PAYLOAD_WORDS - 1))] = payload_word[word]
                store_scale_word(scales, row, scale_word)


@cute.jit
def store_scale_word(scales: cute.Pointer, row: Int32, scale_word: Int32) -> None:
    """Store tile row ``row``'s packed scale word (its four scale bytes) at byte
    ``(row % 32) * 16 + (row / 32) * 4`` of the 512 B scale tile at ``scales``."""
    words = cute.make_tensor(
        cute.recast_ptr(scales, dtype=Int32),
        cute.make_layout((SCALE_TILE_ROWS, SCALE_TILE_WORDS), stride=(SCALE_TILE_WORDS, 1)),
    )
    scale_row, scale_slot = scale_word_index(row)
    words[scale_row, scale_slot] = scale_word


@cute.jit
def scale_word_index(row: Int32) -> tuple[Int32, Int32]:
    """``(row, word)`` of one tile row's four scale bytes in a 32 x 4 int32 tile.

    The packed word sits at byte ``(row % 32) * 16 + (row / 32) * 4`` of the
    512-byte scale tile, which is the tcgen05 scale-factor atom
    ``((32,4),(32,4)) : ((16,4),(0,1))``; seen as int32 that is element
    ``[row % 32, row / 32]``.
    """
    return row & Int32(SCALE_TILE_ROWS - 1), row >> Int32(5)


__all__ = [
    "BLOCKS_PER_TILE_ROW",
    "BLOCK_ELEMENTS",
    "PAIRS_PER_BLOCK",
    "PAYLOAD_WORDS",
    "QUANT_TILE",
    "SCALE_TILE_ROWS",
    "SCALE_TILE_WORDS",
    "abs_bf16x2",
    "abs_max_bf16x2",
    "cvt_bf16x2",
    "cvt_e4m3x2",
    "cvt_e4m3x2_f16x2",
    "cvt_e8m0_scale_byte",
    "cvt_f16_f32",
    "dequantize_block",
    "encode_block",
    "fma_ftz",
    "mul_ftz",
    "quantize_pairs",
    "quantize_tile",
    "scale_bf16x2",
    "scale_word_index",
    "store_scale_word",
    "store_shared_v4",
]
