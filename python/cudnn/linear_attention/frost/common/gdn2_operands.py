# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical stages shared by all six GDN-2 chunk kernel schedules.

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
def row_inverse_norm(even, odd):
    """Reduce one row in the kernel's fixed even/odd, 4/2/1 order."""
    total = even + odd
    for distance in cutlass.range_constexpr(3):
        total = total + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, total, 4 >> distance, 31, kind=nvvm.Shfl.BFLY))
    return cute.math.rsqrt(total + cutlass.Float32(L2_NORM_EPS), fastmath=True)


@cute.jit
def load_qk_beta_row(cfg: cutlass.Constexpr, sQ_ptr, sK_ptr, sBeta_ptr, row, lane_in_row_group, opaque_one):
    """Load raw operands, condition beta and compute row norms; Q is absent in summaries.

    Each scalar sum keeps its original even/odd FFMA order. The caller publishes
    any norms needed by other warp groups and owns the raw-stage lifetime.
    """
    with_q = cutlass.const_expr(sQ_ptr is not None)
    width = cutlass.const_expr(cfg.d_k // 64 * 8)
    raw_q_regs = None
    if cutlass.const_expr(with_q):
        raw_q_regs = cutlass.Array(cutlass.Float32, width, alignment=16)
    raw_k_regs = cutlass.Array(cutlass.Float32, width, alignment=16)
    raw_beta_regs = cutlass.Array(cutlass.Float32, width, alignment=16)
    if cutlass.const_expr(cfg.l2norm):
        k_even, k_odd = opaque_f32_zero(), opaque_f32_zero()
        if cutlass.const_expr(with_q):
            q_even, q_odd = opaque_f32_zero(), opaque_f32_zero()
    for dim_half in cutlass.range_constexpr(cfg.d_k // 64):
        dim_base = dim_half * 64 + lane_in_row_group * 8
        reg_base = dim_half * 8
        offset = dim_half * (cfg.b_t * 64) + row * 64 + swizzle_xor_128b(row, dim_base % 64, elem_bytes=2)
        k = (sK_ptr + offset).load(count=8, alignment=16).to(cutlass.Float32)
        beta = (sBeta_ptr + offset).load(count=8, alignment=16).to(cutlass.Float32)
        if cutlass.const_expr(with_q):
            q = (sQ_ptr + offset).load(count=8, alignment=16).to(cutlass.Float32)
        for j in cutlass.range_constexpr(8):
            raw_k_regs[reg_base + j] = k[j]
            if cutlass.const_expr(with_q):
                raw_q_regs[reg_base + j] = q[j]
            value = beta[j]
            if cutlass.const_expr(cfg.beta_sigmoid):
                value = (sigmoid(value) * (2.0 if cfg.allow_neg_eigval else 1.0)).to(cfg.io_dtype).to(cutlass.Float32)
            raw_beta_regs[reg_base + j] = value
        if cutlass.const_expr(cfg.l2norm):
            for pair in cutlass.range_constexpr(4):
                k_even, k_odd = ffma2(k[2 * pair], k[2 * pair + 1], k[2 * pair], k[2 * pair + 1], k_even, k_odd)
                if cutlass.const_expr(with_q):
                    q_even, q_odd = ffma2(q[2 * pair], q[2 * pair + 1], q[2 * pair], q[2 * pair + 1], q_even, q_odd)
    q_norm, k_norm = opaque_one, opaque_one
    if cutlass.const_expr(cfg.l2norm):
        k_norm = row_inverse_norm(k_even, k_odd)
        if cutlass.const_expr(with_q):
            q_norm = row_inverse_norm(q_even, q_odd)
    return raw_q_regs, raw_k_regs, raw_beta_regs, q_norm, k_norm


@cute.jit
def load_operand_scales(cfg: cutlass.Constexpr, row, lane_in_row_group, bits, exchange_ptr, log_last_ptr, with_restore: cutlass.Constexpr):
    """Read all exchange data before the schedule can release its raw stage.

    Backward needs decay and inverse scales only. Forward also retains the last
    factor for K restore. Exact steep channels use bounded prefix/suffix factors.
    """
    width = cutlass.const_expr(cfg.d_k // 64 * 8)
    decay_regs = cutlass.Array(cutlass.Float32, width, alignment=16)
    inverse_regs = cutlass.Array(cutlass.Float32, width, alignment=16)
    restore_regs = None
    if cutlass.const_expr(with_restore):
        restore_regs = cutlass.Array(cutlass.Float32, width, alignment=16)
    for half in cutlass.range_constexpr(cfg.d_k // 64):
        for group in cutlass.range_constexpr(2):
            channel = half * 64 + lane_in_row_group * 8 + group * 4
            values = (exchange_ptr + exchange_offset(cfg, row, channel)).load(count=4, alignment=16)
            if cutlass.const_expr(with_restore):
                last = (exchange_ptr + exchange_offset(cfg, cfg.b_t - 1, channel)).load(count=4, alignment=16)
            for j in cutlass.range_constexpr(4):
                index = half * 8 + group * 4 + j
                last_value = cutlass.Float32(1.0)
                if cutlass.const_expr(with_restore):
                    last_value = last[j]
                decay, inverse, restore = row_operand_scales(cfg, bits, index, row, channel + j, values[j], last_value, exchange_ptr, log_last_ptr)
                decay_regs[index] = decay
                inverse_regs[index] = inverse
                if cutlass.const_expr(with_restore):
                    restore_regs[index] = restore
    return decay_regs, inverse_regs, restore_regs


@cute.jit
def key_operand_half(cfg: cutlass.Constexpr, half: cutlass.Constexpr, raw_k, beta, k_norm, decay, inverse, restore):
    """Pack the three K operands using one multiply and rounding contract.

    This stage only reads registers: callers can release shared exchange storage
    before packing, and publish each operand under its own barrier.
    """
    decay_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
    inverse_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
    restore_pack = None
    if cutlass.const_expr(restore is not None):
        restore_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
    for pair in cutlass.range_constexpr(4):
        i = half * 8 + pair * 2
        k0, k1 = fmul2(raw_k[i], raw_k[i + 1], k_norm, k_norm)
        beta0, beta1 = fmul2(k0, k1, beta[i], beta[i + 1])
        decay0, decay1 = fmul2(beta0, beta1, decay[i], decay[i + 1])
        decay_pack[pair] = fp32_to_fp16(decay0, decay1, dtype=cfg.io_dtype)
        inverse0, inverse1 = fmul2(k0, k1, inverse[i], inverse[i + 1])
        inverse_pack[pair] = fp32_to_fp16(inverse0, inverse1, dtype=cfg.io_dtype)
        if cutlass.const_expr(restore is not None):
            restore0, restore1 = fmul2(inverse0, inverse1, restore[i], restore[i + 1])
            restore_pack[pair] = fp32_to_fp16(restore0, restore1, dtype=cfg.io_dtype)
    inverse_vec = cutlass.Vector.from_elements((inverse_pack[0], inverse_pack[1], inverse_pack[2], inverse_pack[3]), cutlass.Int32).bitcast(cfg.io_dtype)
    decay_vec = cutlass.Vector.from_elements((decay_pack[0], decay_pack[1], decay_pack[2], decay_pack[3]), cutlass.Int32).bitcast(cfg.io_dtype)
    return inverse_vec, decay_vec, restore_pack


@cute.jit
def query_operand_half(cfg: cutlass.Constexpr, half: cutlass.Constexpr, raw_q, q_scale, decay):
    """Pack Q decay after normalization (and the backward attention scale)."""
    packed = cutlass.Array(cutlass.Int32, 4, alignment=16)
    for pair in cutlass.range_constexpr(4):
        i = half * 8 + pair * 2
        q0, q1 = fmul2(raw_q[i], raw_q[i + 1], q_scale, q_scale)
        q0, q1 = fmul2(q0, q1, decay[i], decay[i + 1])
        packed[pair] = fp32_to_fp16(q0, q1, dtype=cfg.io_dtype)
    return cutlass.Vector.from_elements((packed[0], packed[1], packed[2], packed[3]), cutlass.Int32).bitcast(cfg.io_dtype)
