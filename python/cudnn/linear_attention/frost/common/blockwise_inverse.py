# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blockwise inverse of the beta-folded chunk factor ``I + strict_lower(M)`` (b_t = 64, f16 SMEM tiles in the 128B
XOR-swizzled U GEMM layout): the diagonal 8x8 Gauss-Jordan step, then the off-diagonal corrections 8 -> 16 -> 32 -> 64
(``C <- -D^-1 C A^-1`` on mma.sync fragments).  Every step reads the raw block from ``raw_base`` / ``in_base`` and writes
the inverse on ``base`` / ``out_base``; a kernel that inverts in place passes the same pointer twice.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm

from cudnn.frost.tile_dsl.mma import mma_step, mma_step_k8
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b


@cute.jit
def invert_diagonal_NxN(cfg, in_base, out_base, d_idx, tidx, N: int = 8):
    """Gauss-Jordan inversion of one diagonal NxN block, ``in_base`` -> ``out_base`` (f16 SMEM)."""
    tidx_in_group = tidx % N
    BT = cfg.b_t

    row_coord = d_idx * N + tidx_in_group
    row_off = row_coord * BT + swizzle_xor_128b(row_coord, d_idx * N)
    row_ptr_in = in_base + row_off
    row_ptr = out_base + row_off

    row = [(row_ptr_in + j).load().to(cutlass.Float32) for j in range(N)]
    for i in cutlass.range_constexpr(N):
        row[i] = cutlass.Float32(1.0) if tidx_in_group == i else row[i]
    for src_row in cutlass.range_constexpr(N - 1):
        row_scale = -row[src_row]
        for i in cutlass.range_constexpr(src_row):
            shfl_val = nvvm.shfl_sync(0xFFFFFFFF, row[i], src_row, 0b1100000011111, kind=nvvm.Shfl.IDX)
            row[i] = row[i] + row_scale * shfl_val if tidx_in_group > src_row else row[i]
        row[src_row] = row_scale if tidx_in_group > src_row else row[src_row]

    for j in cutlass.range_constexpr(N):
        (row_ptr + j).store(row[j].to(cfg.io_dtype))


@cute.jit
def blockwise_diagonal_8x8_to_16x16(cfg, base, raw_base, d_idx, lane_idx):
    """Off-diagonal correction 8x8 -> 16x16 (C <- -D^{-1} C A^{-1}); raw C from ``raw_base``, writes on ``base``."""
    BT = cfg.b_t
    row_lo = d_idx + lane_idx % 8
    row_hi = row_lo + 8
    off_d_inv = row_hi * BT + swizzle_xor_128b(row_hi, d_idx + 8)
    off_c = row_hi * BT + swizzle_xor_128b(row_hi, d_idx)
    off_a_inv = row_lo * BT + swizzle_xor_128b(row_lo, d_idx)
    d_inv_frag = nvvm.ldmatrix(base + off_d_inv, 1, nvvm.MMALayout.ROW)
    c_frag = nvvm.ldmatrix(raw_base + off_c, 1, nvvm.MMALayout.COL)

    # ---- T = -(D^-1 @ C) -------------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(c_regs, [d_inv_frag, d_inv_frag], [c_frag], k_step=0, M=16, N=8, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(2)]

    # ---- C = T @ A^-1 ----------------------------------------------------------------
    a_inv_frag = nvvm.ldmatrix(base + off_a_inv, 1, nvvm.MMALayout.COL)
    o_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(o_regs, a_pack, [a_inv_frag], k_step=0, M=16, N=8, ab_dtype=cfg.io_dtype)
    o_pack = fp32_to_fp16(o_regs[0], o_regs[1], dtype=cfg.io_dtype)

    # ---- store corrected C -----------------------------------------------------------
    nvvm.stmatrix(base + off_c, o_pack, nvvm.MMALayout.ROW)


@cute.jit
def blockwise_diagonal_16x16_to_32x32(cfg, base, raw_base, d_idx, lane_idx):
    """Off-diagonal correction 16x16 -> 32x32 (raw C from ``raw_base``)."""
    BT = cfg.b_t
    lane_row = lane_idx % 16
    lane_col = (lane_idx // 16) * 8
    row_lo = d_idx + lane_row
    row_hi = row_lo + 16
    off_d_inv = row_hi * BT + swizzle_xor_128b(row_hi, d_idx + 16 + lane_col)
    off_c = row_hi * BT + swizzle_xor_128b(row_hi, d_idx + lane_col)
    off_a_inv = row_lo * BT + swizzle_xor_128b(row_lo, d_idx + lane_col)
    d_inv_frags = list(nvvm.ldmatrix(base + off_d_inv, 4, nvvm.MMALayout.ROW))
    c_frags = list(nvvm.ldmatrix(raw_base + off_c, 4, nvvm.MMALayout.COL))

    # ---- T = -(D^-1 @ C) -------------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 8, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step(c_regs, d_inv_frags, c_frags, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]

    # ---- C = T @ A^-1 ----------------------------------------------------------------
    a_inv_frags = list(nvvm.ldmatrix(base + off_a_inv, 4, nvvm.MMALayout.COL))
    o_regs = cutlass.Array(cutlass.Float32, 8, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(8):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step(o_regs, a_pack, a_inv_frags, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]

    # ---- store corrected C -----------------------------------------------------------
    nvvm.stmatrix(base + off_c, o_pack, nvvm.MMALayout.ROW)


@cute.jit
def blockwise_diagonal_32x32_to_64x64(cfg, base, raw_base, warp_id, lane_idx, barrier_id, barrier_threads):
    """Off-diagonal correction 32x32 -> 64x64 (2 warps, one 16-row M-band each; raw C from ``raw_base``);
    the two bands meet on named barrier ``barrier_id`` (``barrier_threads`` arrivals) before the store."""
    band = warp_id % 2
    BT = cfg.b_t
    lane_row = lane_idx % 16
    lane_col = (lane_idx // 16) * 8
    row_d_inv = 32 + band * 16 + lane_row
    d_inv_frags = []
    for vs in cutlass.range_constexpr(2):
        d_inv_frags += list(nvvm.ldmatrix(base + row_d_inv * BT + swizzle_xor_128b(row_d_inv, 32 + vs * 16 + lane_col), 4, nvvm.MMALayout.ROW))
    c_frags = []
    for vs in cutlass.range_constexpr(4):
        row_c = 32 + (vs // 2) * 16 + lane_row
        c_frags += list(nvvm.ldmatrix(raw_base + row_c * BT + swizzle_xor_128b(row_c, (vs % 2) * 16 + lane_col), 4, nvvm.MMALayout.COL))

    # ---- T = -(D^-1 @ C) -------------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(16):
        c_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(c_regs, d_inv_frags, c_frags[ks * 8 : ks * 8 + 8], k_step=ks, M=16, N=32, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(16):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(8)]

    # ---- C = T @ A^-1 ----------------------------------------------------------------
    a_inv_frags = []
    for vs in cutlass.range_constexpr(4):
        row_a_inv = (vs // 2) * 16 + lane_row
        a_inv_frags += list(nvvm.ldmatrix(base + row_a_inv * BT + swizzle_xor_128b(row_a_inv, (vs % 2) * 16 + lane_col), 4, nvvm.MMALayout.COL))
    o_regs = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(16):
        o_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(o_regs, a_pack, a_inv_frags[ks * 8 : ks * 8 + 8], k_step=ks, M=16, N=32, ab_dtype=cfg.io_dtype)
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(8)]

    # ---- store corrected C -----------------------------------------------------------
    nvvm.barrier_cta_sync_aligned(barrier_id, thread_count=barrier_threads)
    nvvm.stmatrix(base + row_d_inv * BT + swizzle_xor_128b(row_d_inv, lane_col), o_pack[0:4], nvvm.MMALayout.ROW)
    nvvm.stmatrix(base + row_d_inv * BT + swizzle_xor_128b(row_d_inv, 16 + lane_col), o_pack[4:8], nvvm.MMALayout.ROW)
