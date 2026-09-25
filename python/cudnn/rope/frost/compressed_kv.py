# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100 compressed-KV RoPE + group16 E4M3-scale FP4 QDQ.

The mathematical contract follows DeepSeek-V4.1-Flash inference/model.py and
kernel.py at dba1be0a40aa45a94ad051997016db3960a90277. These kernels are
authored here; CUDA's FP4/FP8 headers were consulted for PTX operand order.
"""

import triton
import triton.language as tl


from .kernels import _e2m1_roundtrip, _e4m3_roundtrip


@triton.jit
def _normalized_pair_fp16(a, b, reciprocal):
    na = tl.minimum(tl.maximum(a * reciprocal, -6.0), 6.0)
    nb = tl.minimum(tl.maximum(b * reciprocal, -6.0), 6.0)
    # BF16 values and E4M3 scales have enough separation from E2M1 ties
    # for an FP16 rounding step to recover RN-division tie behavior.
    na = na.to(tl.float16).to(tl.float32)
    nb = nb.to(tl.float16).to(tl.float32)
    return na, nb


@triton.jit(do_not_specialize=["N_ROWS"])
def cudnn_frost_compkv_rope_qdq(X, CACHE, POSITIONS, N_ROWS, ROWS: tl.constexpr, PACKED_IO: tl.constexpr):
    DIM: tl.constexpr = 512
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
    pos = tl.load(POSITIONS + row, row < N_ROWS, other=0).to(tl.int64)
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
    ar, br = a.reshape((ROWS, 32, 8)), b.reshape((ROWS, 32, 8))
    amax = tl.maximum(tl.max(tl.maximum(tl.abs(ar), tl.abs(br)), 2), 6.0 * (2.0**-9))
    ratio = tl.div_rn(amax, 6.0)
    scale, _ = _e4m3_roundtrip(ratio, ratio)
    # The BF16/E4M3 operand contract permits a reciprocal plus a normalization cast.
    # Exhaustive finite-domain validation is recorded separately before timing.
    reciprocal = tl.div_rn(1.0, scale)
    sa, sb = _normalized_pair_fp16(ar, br, reciprocal[:, :, None])
    qa, qb = _e2m1_roundtrip(sa, sb)
    ya = (qa * scale[:, :, None]).to(tl.bfloat16).reshape((ROWS, PAIRS))
    yb = (qb * scale[:, :, None]).to(tl.bfloat16).reshape((ROWS, PAIRS))
    if PACKED_IO:
        lo, hi = ya.to(tl.uint16, bitcast=True).to(tl.uint32), yb.to(tl.uint16, bitcast=True).to(tl.uint32)
        tl.store(pointer + offset, lo | (hi << 16), valid)
    else:
        tl.store(X + offset * 2, ya, valid)
        tl.store(X + offset * 2 + 1, yb, valid)
