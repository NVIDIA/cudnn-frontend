# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""NVFP4 quantization device functions (flashinfer blockscaled epilogue replicas)."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math

from .moe_kernel_helpers import fmin

HADAMARD_SIZE = 16


# blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:3354-3370
def get_dtype_rcp_limits(dtype):
    if dtype == cutlass.Float4E2M1FN:
        return 1 / 6.0
    if dtype == cutlass.Float8E4M3FN:
        return 1 / 448.0
    if dtype == cutlass.FloatNV8E5M3FNU:
        return 1 / 114688.0
    raise ValueError(f"unsupported quantized dtype {dtype}")


@cute.jit
def load_row_bf16(sD, d_buffer, tidx):
    """One vectorized copy of the thread's full 32-feature bf16 sD row into rmem.
    Split out of the quant/FWHT functions so the kernel can signal "sD consumed"
    (barrier arrive) right after the loads, before any compute."""
    row_feats = 2 * HADAMARD_SIZE
    sD_tiles = cute.zipped_divide(cute.slice_(sD, (None, None, d_buffer)), (1, row_feats))
    src = cute.slice_(sD_tiles, ((None, None), (tidx, 0)))
    rmem_bf16 = cute.make_rmem_tensor(src.shape, cutlass.BFloat16)
    cute.autovec_copy(src, rmem_bf16)
    return rmem_bf16


@cute.jit
def nvfp4_quant_rmem_row(rmem_bf16, d_buffer, tidx, sOut, norm_const, sSf, sf_pair, sf_dtype):
    """NVFP4-quantize the thread's full 32-feature bf16 row (from load_row_bf16, no
    transform) into sOut (packed e2m1) + sSf (fp8 block scales) — the fused-D
    analog of group_rht_cast's rowwise cast path."""
    row_feats = 2 * HADAMARD_SIZE
    tCompute = cute.make_rmem_tensor(rmem_bf16.shape, cutlass.Float32)
    for i in cutlass.range_constexpr(row_feats):
        tCompute[i] = rmem_bf16[i].to(cutlass.Float32)

    _nvfp4_quant_row(tCompute, d_buffer, tidx, sOut, norm_const, sSf, sf_pair, sf_dtype)


@cute.jit
def _nvfp4_quant_row(tCompute, d_buffer, token, sOut, norm_const, sSf, sf_pair, sf_dtype):
    """Flashinfer blockscaled-epilogue NVFP4 quantization of one thread's 32-feature
    f32 row: packed e2m1 data into sOut, one fp8 block scale per (1,16) block into
    sSf slots (token, num_vecs*sf_pair + vi)."""
    row_feats = cute.size(tCompute.shape)
    tCompute_flat = cute.make_tensor(tCompute.iterator, cute.make_layout((row_feats,)))

    # blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:2837-2847
    num_vecs = row_feats // HADAMARD_SIZE
    assert num_vecs % 2 == 0, "num_vecs must be even (packed f32x2 pair loops)"
    tTR_rAcc_frg = cute.logical_divide(tCompute_flat, cute.make_layout(HADAMARD_SIZE))
    acc_frg = tTR_rAcc_frg.load()
    abs_acc_frg_ir = math.absf(acc_frg.ir_value())
    abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
    tCrSFC_pvscale = cute.make_rmem_tensor((num_vecs,), cutlass.Float32)
    # blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:2850-2856
    for vi in cutlass.range_constexpr(num_vecs):
        tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
            cute.ReductionOp.MAX,
            cutlass.Float32(0.0),
            0,  # Use 0.0 as init for abs values
        )
    # blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:2856-2873
    for vi in cutlass.range_constexpr(0, num_vecs, 2):
        tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = cute.arch.mul_packed_f32x2(
            (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
            (
                get_dtype_rcp_limits(cutlass.Float4E2M1FN),
                get_dtype_rcp_limits(cutlass.Float4E2M1FN),
            ),
        )
        tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = cute.arch.mul_packed_f32x2(
            (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
            (norm_const, norm_const),
        )

    # blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:2887
    # f32 -> fp8 scale format as a padded 4-wide vector: nvgpu.cvt_fptrunc requires a
    # 32-bit-aligned 1-d vector (4 x f8), never a scalar. Same pattern as
    # grouped_gemm_swiglu_quant's pvscale_f32x4 -> sfd_f8x4 round-trip.
    assert num_vecs <= 4, "scale convert padding assumes at most 4 blocks per row"
    pvscale_f32x4 = cute.make_rmem_tensor((4,), cutlass.Float32)
    for vi in cutlass.range_constexpr(4):
        pvscale_f32x4[vi] = tCrSFC_pvscale[min(vi, num_vecs - 1)]
    tCrSFC_f8x4 = cute.make_rmem_tensor((4,), sf_dtype)
    tCrSFC_f8x4.store(pvscale_f32x4.load().to(sf_dtype))
    tCrSFC = cute.make_rmem_tensor((num_vecs,), sf_dtype)
    for vi in cutlass.range_constexpr(num_vecs):
        tCrSFC[vi] = tCrSFC_f8x4[vi]
    # fp8 scale -> f32 widening round-trip, vectorized for the same reason.
    tCrSFC_f32x4 = cute.make_rmem_tensor((4,), cutlass.Float32)
    tCrSFC_f32x4.store(tCrSFC_f8x4.load().to(cutlass.Float32))

    # SFC -> smem
    for vi in cutlass.range_constexpr(num_vecs):
        sSf[(token, num_vecs * sf_pair + vi)] = tCrSFC[vi]

    # blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:2900-2921
    fp32_max = cutlass.Float32(3.40282346638528859812e38)
    for vi in cutlass.range_constexpr(0, num_vecs, 2):
        acc_scale = cute.arch.mul_packed_f32x2(
            (
                cute.arch.rcp_approx(tCrSFC_f32x4[vi]),
                cute.arch.rcp_approx(tCrSFC_f32x4[vi + 1]),
            ),
            (norm_const, norm_const),
        )
        acc_scale_min0 = fmin(acc_scale[0], fp32_max, nan=True)
        acc_scale_min1 = fmin(acc_scale[1], fp32_max, nan=True)

        vec0 = tTR_rAcc_frg[None, vi]
        vec1 = tTR_rAcc_frg[None, vi + 1]
        for ei in cutlass.range_constexpr(HADAMARD_SIZE):
            vec0[ei], vec1[ei] = cute.arch.mul_packed_f32x2(
                (vec0[ei], vec1[ei]),
                (acc_scale_min0, acc_scale_min1),
            )

    # blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:2936-2937
    tRS_rC = cute.make_rmem_tensor(tCompute.shape, cutlass.Float4E2M1FN)
    tRS_rC.store(tCompute.load().to(cutlass.Float4E2M1FN))
    sOut_tiles = cute.zipped_divide(cute.slice_(sOut, (None, None, d_buffer)), (1, row_feats))
    dst = cute.slice_(sOut_tiles, ((None, None), (token, 0)))
    cute.autovec_copy(tRS_rC, dst)
