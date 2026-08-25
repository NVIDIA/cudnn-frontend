# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SMEM FWHT device functions for the fused RHT output (orthonormal Sylvester H16).

Split into LOAD helpers (sD -> registers) and rmem COMPUTE functions so the kernel
can signal "sD consumed" (lockstep-barrier arrive) right after the loads — the
FWHT/quant compute and all stores then overlap the ACT warps' next subtile."""

import cutlass
import cutlass.cute as cute

from .moe_kernel_helpers import fmax, fmin
from .quant_utils import (
    _nvfp4_quant_row,
    get_dtype_rcp_limits,
)

HADAMARD_SIZE = 16


@cute.jit
def load_colwise_pairs_bf16(sD, d_buffer, tidx):
    """16 vectorized 4-byte pair loads of the thread's TWO adjacent feature columns
    of one 16-token block, interleaved token-major: rmem[2*i + c] =
    sD(token_base + i, feature + c)."""
    token_block = tidx // HADAMARD_SIZE
    feat_pair = tidx % HADAMARD_SIZE
    token_base = token_block * HADAMARD_SIZE

    rmem_bf16 = cute.make_rmem_tensor((2 * HADAMARD_SIZE,), cutlass.BFloat16)
    rmem_pairs = cute.zipped_divide(rmem_bf16, (2,))
    sD_pairs = cute.zipped_divide(cute.slice_(sD, (None, None, d_buffer)), (1, 2))
    for i in cutlass.range_constexpr(HADAMARD_SIZE):
        cute.autovec_copy(
            cute.slice_(sD_pairs, ((None, None), (token_base + i, feat_pair))),
            cute.slice_(rmem_pairs, ((None,), i)),
        )
    return rmem_bf16


@cute.jit
def _colwise_fwht_inplace(tCompute):
    """In-place natural-order 16-point FWHT over both interleaved token-major
    columns (tCompute[2*i + c] = column c, token i)."""
    for c in cutlass.range_constexpr(2):
        for base in cutlass.range_constexpr(0, HADAMARD_SIZE, 2):
            x = tCompute[2 * base + c]
            y = tCompute[2 * (base + 1) + c]
            tCompute[2 * base + c] = x + y
            tCompute[2 * (base + 1) + c] = x - y
        for base in cutlass.range_constexpr(0, HADAMARD_SIZE, 4):
            for i in cutlass.range_constexpr(2):
                x = tCompute[2 * (base + i) + c]
                y = tCompute[2 * (base + i + 2) + c]
                tCompute[2 * (base + i) + c] = x + y
                tCompute[2 * (base + i + 2) + c] = x - y
        for base in cutlass.range_constexpr(0, HADAMARD_SIZE, 8):
            for i in cutlass.range_constexpr(4):
                x = tCompute[2 * (base + i) + c]
                y = tCompute[2 * (base + i + 4) + c]
                tCompute[2 * (base + i) + c] = x + y
                tCompute[2 * (base + i + 4) + c] = x - y
        for i in cutlass.range_constexpr(8):
            x = tCompute[2 * i + c]
            y = tCompute[2 * (i + 8) + c]
            tCompute[2 * i + c] = x + y
            tCompute[2 * (i + 8) + c] = x - y


@cute.jit
def hadamard_rmem_colwise_fwht(rmem_bf16, d_buffer, tidx, sRht):
    """Per-feature bf16 FWHT over 16-token blocks from the load_colwise_pairs_bf16
    registers, stored to sRht at the SAME (token, feature) coords the input was
    read from as one 4-byte pair store per token row.
    (The colwise QUANT path lives in hadamard_rmem_colwise_fwht_quant.)

    In-place natural-order FWHT => output i is the transformed value for token
    token_base+i, matching torch_rht's block layout."""
    token_block = tidx // HADAMARD_SIZE
    feat_pair = tidx % HADAMARD_SIZE
    token_base = token_block * HADAMARD_SIZE

    n_vals = 2 * HADAMARD_SIZE
    tCompute = cute.make_rmem_tensor((n_vals,), cutlass.Float32)
    for i in cutlass.range_constexpr(n_vals):
        tCompute[i] = rmem_bf16[i].to(cutlass.Float32)

    _colwise_fwht_inplace(tCompute)

    # Orthonormal scale, rounded through bf16 (== the bf16 output values).
    for i in cutlass.range_constexpr(n_vals):
        tCompute[i] = (tCompute[i] * cutlass.Float32(0.25)).to(cutlass.BFloat16).to(cutlass.Float32)

    rmem_st = cute.make_rmem_tensor((2,), cutlass.BFloat16)
    sRht_pairs = cute.zipped_divide(cute.slice_(sRht, (None, None, d_buffer)), (1, 2))
    for i in cutlass.range_constexpr(HADAMARD_SIZE):
        for c in cutlass.range_constexpr(2):
            rmem_st[c] = tCompute[2 * i + c].to(cutlass.BFloat16)
        cute.autovec_copy(rmem_st, cute.slice_(sRht_pairs, ((None, None), (token_base + i, feat_pair))))


@cute.jit
def hadamard_rmem_colwise_fwht_quant(rmem_bf16, d_buffer, tidx, norm_const, sRht, sSf, sf_row_base, sf_dtype):
    """Colwise FWHT + NVFP4 quantization from the load_colwise_pairs_bf16 registers,
    stored at the SAME (token, feature) coords the input was read from (the staging
    is f-major like every other output; packed nibbles pair ADJACENT FEATURES of one
    token, so each token row is one 1-byte store). Quantization blocks follow the
    transform: one (16, 1) token-block scale per feature, staged in the sSf smem
    rows (sf_row_base + feature, token_block); the kernel stores each thread's whole
    contiguous scale row once per tile."""
    token_block = tidx // HADAMARD_SIZE
    feat_pair = tidx % HADAMARD_SIZE
    token_base = token_block * HADAMARD_SIZE
    feature = 2 * feat_pair

    n_vals = 2 * HADAMARD_SIZE
    tCompute = cute.make_rmem_tensor((n_vals,), cutlass.Float32)
    for i in cutlass.range_constexpr(n_vals):
        tCompute[i] = rmem_bf16[i].to(cutlass.Float32)

    _colwise_fwht_inplace(tCompute)

    # Orthonormal scale, rounded through bf16 (== the bf16 output values).
    for i in cutlass.range_constexpr(n_vals):
        tCompute[i] = (tCompute[i] * cutlass.Float32(0.25)).to(cutlass.BFloat16).to(cutlass.Float32)

    # group_rht_cast's exact (fast_math=0) op sequence — NOT the flashinfer one:
    # gem = ge * (1/6) is pre-folded, and the encode scale is computed with EXACT
    # f32 divisions (enc = 1/(dec * gd), gd = 1/ge). The flashinfer
    # rcp_approx(dec) * ge form agrees at ge = 1 but drifts 1 ulp at non-dyadic
    # ge, flipping e2m1 codes on rounding boundaries.
    gem = norm_const * cutlass.Float32(get_dtype_rcp_limits(cutlass.Float4E2M1FN))
    gd = cutlass.Float32(1.0) / norm_const
    pv = cute.make_rmem_tensor((2,), cutlass.Float32)
    for c in cutlass.range_constexpr(2):
        pv[c] = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(HADAMARD_SIZE):
            v = tCompute[2 * i + c]
            pv[c] = fmax(pv[c], fmax(v, -v))
    pv[0], pv[1] = cute.arch.mul_packed_f32x2((pv[0], pv[1]), (gem, gem))

    # f32 -> fp8 scale format as a padded 4-wide vector (and widen back the same way):
    # nvgpu.cvt_fptrunc requires a 32-bit-aligned 1-d vector (4 x f8), never a
    # scalar. Same pattern as grouped_gemm_swiglu_quant's pvscale round-trip.
    pv_f32x4 = cute.make_rmem_tensor((4,), cutlass.Float32)
    for c in cutlass.range_constexpr(4):
        pv_f32x4[c] = pv[min(c, 1)]
    tCrSFC_f8x4 = cute.make_rmem_tensor((4,), sf_dtype)
    tCrSFC_f8x4.store(pv_f32x4.load().to(sf_dtype))
    tCrSFC_f32x4 = cute.make_rmem_tensor((4,), cutlass.Float32)
    tCrSFC_f32x4.store(tCrSFC_f8x4.load().to(cutlass.Float32))
    for c in cutlass.range_constexpr(2):
        sSf[(sf_row_base + feature + c, token_block)] = tCrSFC_f8x4[c]

    fp32_max = cutlass.Float32(3.40282346638528859812e38)
    acc_scale_min0 = fmin(cutlass.Float32(1.0) / (tCrSFC_f32x4[0] * gd), fp32_max, nan=True)
    acc_scale_min1 = fmin(cutlass.Float32(1.0) / (tCrSFC_f32x4[1] * gd), fp32_max, nan=True)
    for i in cutlass.range_constexpr(HADAMARD_SIZE):
        tCompute[2 * i], tCompute[2 * i + 1] = cute.arch.mul_packed_f32x2(
            (tCompute[2 * i], tCompute[2 * i + 1]),
            (acc_scale_min0, acc_scale_min1),
        )

    tRS_rC = cute.make_rmem_tensor(tCompute.shape, cutlass.Float4E2M1FN)
    tRS_rC.store(tCompute.load().to(cutlass.Float4E2M1FN))
    src_pairs = cute.zipped_divide(tRS_rC, (2,))
    sRht_pairs = cute.zipped_divide(cute.slice_(sRht, (None, None, d_buffer)), (1, 2))
    for i in cutlass.range_constexpr(HADAMARD_SIZE):
        cute.autovec_copy(
            cute.slice_(src_pairs, ((None,), i)),
            cute.slice_(sRht_pairs, ((None, None), (token_base + i, feat_pair))),
        )


@cute.jit
def hadamard_rmem_rowwise_fwht(rmem_bf16, d_buffer, tidx, sRht, norm_const, sSf, sf_pair, sf_dtype):
    """Per-token FWHT over the thread's full 32-feature row (two independent 16-feature
    Hadamard blocks) from the load_row_bf16 registers, stored to sRht at the same
    (token, feature) coords: bf16, or NVFP4 (packed e2m1 into sRht + e4m3/ue5m3 block
    scales into sSf slots (tidx, 2*sf_pair..)) when sRht is fp4-typed."""
    token = tidx
    row_feats = 2 * HADAMARD_SIZE

    tCompute = cute.make_rmem_tensor(rmem_bf16.shape, cutlass.Float32)
    for i in cutlass.range_constexpr(row_feats):
        tCompute[i] = rmem_bf16[i].to(cutlass.Float32)

    for off in cutlass.range_constexpr(0, row_feats, HADAMARD_SIZE):
        for base in cutlass.range_constexpr(0, HADAMARD_SIZE, 2):
            x = tCompute[off + base]
            y = tCompute[off + base + 1]
            tCompute[off + base] = x + y
            tCompute[off + base + 1] = x - y
        for base in cutlass.range_constexpr(0, HADAMARD_SIZE, 4):
            for i in cutlass.range_constexpr(2):
                x = tCompute[off + base + i]
                y = tCompute[off + base + i + 2]
                tCompute[off + base + i] = x + y
                tCompute[off + base + i + 2] = x - y
        for base in cutlass.range_constexpr(0, HADAMARD_SIZE, 8):
            for i in cutlass.range_constexpr(4):
                x = tCompute[off + base + i]
                y = tCompute[off + base + i + 4]
                tCompute[off + base + i] = x + y
                tCompute[off + base + i + 4] = x - y
        for i in cutlass.range_constexpr(8):
            x = tCompute[off + i]
            y = tCompute[off + i + 8]
            tCompute[off + i] = x + y
            tCompute[off + i + 8] = x - y

    # Orthonormal scale, rounded through bf16 (== the bf16 output values) either way.
    for i in cutlass.range_constexpr(row_feats):
        tCompute[i] = (tCompute[i] * cutlass.Float32(0.25)).to(cutlass.BFloat16).to(cutlass.Float32)

    if cutlass.const_expr(sRht.element_type == cutlass.Float4E2M1FN):
        _nvfp4_quant_row(tCompute, d_buffer, token, sRht, norm_const, sSf, sf_pair, sf_dtype)
    else:
        rmem_st = cute.make_rmem_tensor(rmem_bf16.shape, cutlass.BFloat16)
        for i in cutlass.range_constexpr(row_feats):
            rmem_st[i] = tCompute[i].to(cutlass.BFloat16)
        sRht_tiles = cute.zipped_divide(cute.slice_(sRht, (None, None, d_buffer)), (1, row_feats))
        dst = cute.slice_(sRht_tiles, ((None, None), (token, 0)))
        cute.autovec_copy(rmem_st, dst)
