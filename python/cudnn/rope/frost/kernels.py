# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100 tail RoPE and microscaled quantize/dequantize fusion.

The mathematical contract follows DeepSeek-V4.1-Flash inference/model.py and
kernel.py at dba1be0a40aa45a94ad051997016db3960a90277. These kernels are
authored here; CUDA's FP4/FP8 headers were consulted for PTX operand order.
"""

import triton
import triton.language as tl


@triton.jit
def _e2m1_roundtrip(a, b):
    return tl.inline_asm_elementwise(
        """{
        .reg .b8 q;
        .reg .b32 h2;
        .reg .b16 lo, hi;
        cvt.rn.satfinite.e2m1x2.f32 q, $3, $2;
        cvt.rn.f16x2.e2m1x2 h2, q;
        mov.b32 {lo, hi}, h2;
        cvt.f32.f16 $0, lo;
        cvt.f32.f16 $1, hi;
        }""",
        constraints="=f,=f,f,f",
        args=[a, b],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e4m3_roundtrip(a, b):
    return tl.inline_asm_elementwise(
        """{
        .reg .b16 q;
        .reg .b32 h2;
        .reg .b16 lo, hi;
        cvt.rn.satfinite.e4m3x2.f32 q, $3, $2;
        cvt.rn.f16x2.e4m3x2 h2, q;
        mov.b32 {lo, hi}, h2;
        cvt.f32.f16 $0, lo;
        cvt.f32.f16 $1, hi;
        }""",
        constraints="=f,=f,f,f",
        args=[a, b],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def cudnn_frost_rope_qdq_inplace(X, CACHE, POSITIONS, N_ROWS, HEADS: tl.constexpr, FP4: tl.constexpr, ROWS: tl.constexpr, PACKED_IO: tl.constexpr):
    DIM: tl.constexpr = 128 if FP4 else 512
    PAIRS: tl.constexpr = DIM // 2
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    pair = tl.arange(0, PAIRS)
    offset = row[:, None] * PAIRS + pair[None, :]
    valid = row[:, None] < N_ROWS
    if PACKED_IO:
        pointer = X.to(tl.pointer_type(tl.uint32))
        words = tl.load(pointer + offset, valid, other=0)
        a = (words << 16).to(tl.float32, bitcast=True)
        b = (words & 0xFFFF0000).to(tl.float32, bitcast=True)
    else:
        a = tl.load(X + offset * 2, valid, other=0).to(tl.float32)
        b = tl.load(X + offset * 2 + 1, valid, other=0).to(tl.float32)
    pos = tl.load(POSITIONS + row // HEADS, row < N_ROWS, other=0).to(tl.int64)
    cache_offset = pos[:, None] * 64 + pair[None, :] - (PAIRS - 32)
    tail = valid & (pair[None, :] >= PAIRS - 32)
    cosine = tl.load(CACHE + cache_offset, tail, other=1.0)
    sine = tl.load(CACHE + cache_offset + 32, tail, other=0.0)
    # Sign-bit negation retains -0. Only these explicit FMAs may contract.
    negative_bs = ((b * sine).to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
    real = tl.fma(a, cosine, negative_bs).to(tl.bfloat16).to(tl.float32)
    imag = tl.fma(b, cosine, a * sine).to(tl.bfloat16).to(tl.float32)
    # The source stores BF16 before computing amax; preserve that boundary.
    a = tl.where(pair[None, :] >= PAIRS - 32, real, a)
    b = tl.where(pair[None, :] >= PAIRS - 32, imag, b)
    ar, br = a.reshape((ROWS, DIM // 32, 16)), b.reshape((ROWS, DIM // 32, 16))
    LIMIT: tl.constexpr = 6.0 if FP4 else 448.0
    FLOOR: tl.constexpr = 6.0 * (2.0**-126) if FP4 else 1e-4
    amax = tl.maximum(tl.max(tl.maximum(tl.abs(ar), tl.abs(br)), 2), FLOOR)
    ratio = amax * (1.0 / LIMIT)
    bits = ratio.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    # Finite post-RoPE BF16 values imply normal scale and reciprocal.
    reciprocal = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    sa = tl.minimum(tl.maximum(ar * reciprocal[:, :, None], -LIMIT), LIMIT)
    sb = tl.minimum(tl.maximum(br * reciprocal[:, :, None], -LIMIT), LIMIT)
    if FP4:
        qa, qb = _e2m1_roundtrip(sa, sb)
    else:
        qa, qb = _e4m3_roundtrip(sa, sb)
    ya = (qa * scale[:, :, None]).to(tl.bfloat16).reshape((ROWS, PAIRS))
    yb = (qb * scale[:, :, None]).to(tl.bfloat16).reshape((ROWS, PAIRS))
    if PACKED_IO:
        lo, hi = ya.to(tl.uint16, bitcast=True).to(tl.uint32), yb.to(tl.uint16, bitcast=True).to(tl.uint32)
        tl.store(pointer + offset, lo | (hi << 16), valid)
    else:
        tl.store(X + offset * 2, ya, valid)
        tl.store(X + offset * 2 + 1, yb, valid)
