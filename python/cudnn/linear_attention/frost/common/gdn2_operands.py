# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical stages shared by the GDN-2 forward, recompute and summary kernels.

These functions operate on one channel or row. Stage selection, shared-memory
publication and reuse barriers remain in the calling kernel schedules.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm

from cudnn.frost.tile_dsl.pointwise import fadd2, ffma2, fmul2, fp32_to_fp16, opaque_f32_zero, sigmoid
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b

from .steep_decay import exchange_offset, has_exact_path, row_operand_scales

LOG2_E = 1.4426950408889634
L2_NORM_EPS = 1.0e-6


@cute.jit
def gate_scale(cfg, raw_gate: cutlass.Float32) -> cutlass.Float32:
    """Map raw gate to the log2-domain decay increment."""

    if cutlass.const_expr(cfg.safe_gate):
        return cfg.gate_scale_log2 * sigmoid(raw_gate)
    if cutlass.const_expr(cfg.log_gate):
        return raw_gate * cutlass.Float32(LOG2_E)
    return cute.math.log2(raw_gate + cutlass.Float32(1e-10), fastmath=True)


@cute.jit
def scan_gate_prefixes(cfg: cutlass.Constexpr, gate_raw, chunk_start, batch_seqlen, cg0_a_log_exp, cg0_dt_bias_value):
    """Condition a channel's gates and scan in the original pairwise order.

    Exact kernels retain token logs in gate_raw. The caller owns the exchange
    stores, scan ballots and publication barriers.
    """
    g_prefix_regs = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
    if cutlass.const_expr(cfg.safe_gate):
        for row in cutlass.range_constexpr(cfg.b_t):
            g_prefix_regs[row] = gate_scale(cfg, cg0_a_log_exp * (gate_raw[row] + cg0_dt_bias_value))
    else:
        for row in cutlass.range_constexpr(cfg.b_t):
            g_prefix_regs[row] = gate_scale(cfg, gate_raw[row])

    # ---- ragged tail chunk: padded rows carry no decay -----------------------
    if chunk_start + cutlass.Int32(cfg.b_t) > batch_seqlen:
        for row in cutlass.range_constexpr(cfg.b_t):
            g_prefix_regs[row] = cutlass.Float32(0.0) if chunk_start + cutlass.Int32(row) >= batch_seqlen else g_prefix_regs[row]

    if cutlass.const_expr(has_exact_path(cfg)):
        for row in cutlass.range_constexpr(cfg.b_t):
            gate_raw[row] = g_prefix_regs[row]

    prefix_acc = cutlass.Float32(0.0)
    for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
        row0 = row_pair * 2
        row1 = row0 + 1
        gate0 = g_prefix_regs[row0]
        gate1 = g_prefix_regs[row1]
        prefix0, row_pair_sum = fadd2(prefix_acc, gate0, gate0, gate1)
        prefix1 = prefix_acc + row_pair_sum
        g_prefix_regs[row0] = prefix0
        g_prefix_regs[row1] = prefix1
        prefix_acc = prefix1
    return prefix_acc, g_prefix_regs


@cute.jit
def load_key_beta_row(cfg: cutlass.Constexpr, sK_ptr, sBeta_ptr, decay_row, lane_in_row_group, opaque_one):
    """Load a row's key/beta registers and optional key norm without synchronizing."""
    dk_halves = cutlass.const_expr(cfg.d_k // 64)
    raw_k_regs = cutlass.Array(cutlass.Float32, dk_halves * 8, alignment=16)
    raw_beta_regs = cutlass.Array(cutlass.Float32, dk_halves * 8, alignment=16)

    # ---- optional K L2-norm + K inv stage ------------------------------------
    if cutlass.const_expr(cfg.l2norm):
        kk_lo = opaque_f32_zero()
        kk_hi = opaque_f32_zero()
    for dim_half in cutlass.range_constexpr(dk_halves):
        dim_base = dim_half * 64 + lane_in_row_group * 8
        reg_base = dim_half * 8
        f16_segment = dim_base // 64
        f16_segment_dim = dim_base - f16_segment * 64
        raw_f16_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
        raw_k_frag = (sK_ptr + raw_f16_idx).load(count=8, alignment=16)
        raw_beta_frag = (sBeta_ptr + raw_f16_idx).load(count=8, alignment=16)
        raw_k_frag_f32 = raw_k_frag.to(cutlass.Float32)
        raw_beta_frag_f32 = raw_beta_frag.to(cutlass.Float32)
        for dim_offset in cutlass.range_constexpr(8):
            k_val = raw_k_frag_f32[dim_offset]
            raw_k_regs[reg_base + dim_offset] = k_val
            beta_val = raw_beta_frag_f32[dim_offset]
            if cutlass.const_expr(cfg.beta_sigmoid):
                beta_val = (sigmoid(beta_val) * (2.0 if cfg.allow_neg_eigval else 1.0)).to(cfg.io_dtype).to(cutlass.Float32)
            raw_beta_regs[reg_base + dim_offset] = beta_val
        if cutlass.const_expr(cfg.l2norm):
            # even dims in the lo lane, odd in the hi lane: same terms, same order
            for dim_pair in cutlass.range_constexpr(4):
                k_even = raw_k_frag_f32[2 * dim_pair]
                k_odd = raw_k_frag_f32[2 * dim_pair + 1]
                kk_lo, kk_hi = ffma2(k_even, k_odd, k_even, k_odd, kk_lo, kk_hi)

    k_inv_norm = opaque_one
    if cutlass.const_expr(cfg.l2norm):
        k_sum_sq = kk_lo + kk_hi
        k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY))
        k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY))
        k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY))
        k_inv_norm = cute.math.rsqrt(k_sum_sq + cutlass.Float32(L2_NORM_EPS), fastmath=True)
    return raw_k_regs, raw_beta_regs, k_inv_norm


@cute.jit
def key_operand_half(
    cfg: cutlass.Constexpr,
    dim_half: cutlass.Constexpr,
    raw_k_regs,
    raw_beta_regs,
    k_inv_norm,
    decay_row,
    lane_in_row_group,
    steep_bits,
    sGate_exchange_ptr,
    steep_last_ptr,
):
    """Build one 64-channel half of the decay, inverse and restore operands.

    Keep the original multiply/conversion order. Callers wait for each operand
    slot and store the returned values under their own pipeline schedule.
    """
    exp_g_regs = cutlass.Array(cutlass.Float32, 8, alignment=16)
    inverse_g_regs = cutlass.Array(cutlass.Float32, 8, alignment=16)
    k_inv_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
    k_restore_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
    dim_base = dim_half * 64 + lane_in_row_group * 8
    for f32_group in cutlass.range_constexpr(2):
        channel = dim_base + f32_group * 4
        prefix = (sGate_exchange_ptr + exchange_offset(cfg, decay_row, channel)).load(count=4, alignment=16)
        for j in cutlass.range_constexpr(4):
            exp_g_regs[f32_group * 4 + j] = prefix[j]

    # ---- K decay + K inv + K restore operands: Beta * K * exp2(+g), K * exp2(-g), K * exp2(g last - g) ----
    exp_g_last_half = cutlass.Array(cutlass.Float32, 8, alignment=16)
    for f32_group in cutlass.range_constexpr(2):
        channel = dim_base + f32_group * 4
        exp_g_last_frag = (sGate_exchange_ptr + exchange_offset(cfg, cfg.b_t - 1, channel)).load(count=4, alignment=16)
        for j in cutlass.range_constexpr(4):
            exp_g_last_half[f32_group * 4 + j] = exp_g_last_frag[j]
    for elem in cutlass.range_constexpr(8):
        decay, inverse, restore = row_operand_scales(
            cfg,
            steep_bits,
            dim_half * 8 + elem,
            decay_row,
            dim_base + elem,
            exp_g_regs[elem],
            exp_g_last_half[elem],
            sGate_exchange_ptr,
            steep_last_ptr,
        )
        exp_g_regs[elem] = decay
        inverse_g_regs[elem] = inverse
        exp_g_last_half[elem] = restore
    k_decay_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
    for pair_idx in cutlass.range_constexpr(4):
        dim0 = pair_idx * 2
        dim1 = dim0 + 1
        raw_reg_idx0 = dim_half * 8 + dim0
        raw_reg_idx1 = dim_half * 8 + dim1
        k_value0, k_value1 = fmul2(raw_k_regs[raw_reg_idx0], raw_k_regs[raw_reg_idx1], k_inv_norm, k_inv_norm)
        k_beta0, k_beta1 = fmul2(k_value0, k_value1, raw_beta_regs[raw_reg_idx0], raw_beta_regs[raw_reg_idx1])
        k_decay0, k_decay1 = fmul2(k_beta0, k_beta1, exp_g_regs[dim0], exp_g_regs[dim1])
        k_decay_pack[pair_idx] = fp32_to_fp16(k_decay0, k_decay1, dtype=cfg.io_dtype)
        exp_neg_g0 = inverse_g_regs[dim0]
        exp_neg_g1 = inverse_g_regs[dim1]
        k_inv0, k_inv1 = fmul2(k_value0, k_value1, exp_neg_g0, exp_neg_g1)
        k_inv_pack[pair_idx] = fp32_to_fp16(k_inv0, k_inv1, dtype=cfg.io_dtype)
        k_restore0, k_restore1 = fmul2(k_inv0, k_inv1, exp_g_last_half[dim0], exp_g_last_half[dim1])
        k_restore_pack[pair_idx] = fp32_to_fp16(k_restore0, k_restore1, dtype=cfg.io_dtype)

    k_inv_vec = cutlass.Vector.from_elements(
        (
            k_inv_pack[0],
            k_inv_pack[1],
            k_inv_pack[2],
            k_inv_pack[3],
        ),
        cutlass.Int32,
    ).bitcast(cfg.io_dtype)
    k_decay_vec = cutlass.Vector.from_elements(
        (
            k_decay_pack[0],
            k_decay_pack[1],
            k_decay_pack[2],
            k_decay_pack[3],
        ),
        cutlass.Int32,
    ).bitcast(cfg.io_dtype)
    return k_inv_vec, k_decay_vec, k_restore_pack
