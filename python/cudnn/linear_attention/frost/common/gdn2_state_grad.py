# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State-gradient operands for bounded GDN-2 decay.

Mild channels retain the decayed tensor-core operand. Steep channels keep the
original FP32 gradient in caller-owned scratch and contribute through bounded
FP32 projections. Packing the original gradient to FP16 can overflow even when
the decayed gradient and every requested output are finite.

Scratch holds one [V, K] state and a ring of [BT, K] bounded keys per persistent
CTA. The schedules own publication and reuse: dstate_input_ready publishes the
state, k_decay_inv_ready publishes keys, and full backward's CG2 release protects
the state and U through the restored-key projection.
"""

import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b
from .steep_decay import has_exact_path, steep_channel_bit


@cute.jit
def pack_dstate_operand(
    cfg: cutlass.Constexpr, original, decayed, first_channel, words_ptr, steep_any, scratch_row, row_valid, width: cutlass.Constexpr[int] = 32
):
    """Save original steep channels in FP32 and zero their tensor-core operand."""
    operand = cutlass.Array(cutlass.Float32, width, alignment=16)
    for channel in cutlass.range_constexpr(width):
        operand[channel] = decayed[channel]
    if cutlass.const_expr(has_exact_path(cfg)):
        if steep_any != 0:
            steep_rows = cutlass.const_expr(cfg.d_k // len(cfg.compute_group_0_warp_ids))
            for channel in cutlass.range_constexpr(width):
                if steep_channel_bit(words_ptr, first_channel + channel, steep_rows) != 0:
                    if row_valid:
                        (scratch_row + first_channel + channel).store(original[channel])
                    operand[channel] = cutlass.Float32(0.0)
    packed = cutlass.Array(cutlass.Int32, width // 2, alignment=16)
    for pc in cutlass.range_constexpr(width // 2):
        packed[pc] = fp32_to_fp16(operand[2 * pc], operand[2 * pc + 1], dtype=cfg.io_dtype)
    return packed


@cute.jit
def store_steep_keys(cfg: cutlass.Constexpr, raw_k, norm, decay, bits, row, lane, keys):
    """Preserve bounded key operands before narrowing can erase a finite dU term."""
    if cutlass.const_expr(has_exact_path(cfg)):
        for i in cutlass.range_constexpr(cfg.d_k // 64 * 8):
            if ((bits >> i) & 1) != 0:
                channel = (i // 8) * 64 + lane * 8 + i % 8
                (keys + row * cfg.d_k + channel).store((raw_k[i] * norm) * decay[i])


@cute.jit
def add_steep_du(cfg: cutlass.Constexpr, values, scratch, keys, words, value_base, lane):
    """Add dstate @ K_restore to a 16x256b TMEM load's eight FP32 values.

    The caller holds the K-operand stage until dU is published. Both operands
    remain FP32: rounding K_restore to FP16 can erase early-token contributions
    that become representable after multiplication by the state gradient.
    """
    result = cutlass.Array(cutlass.Float32, 8, alignment=16)
    for i in cutlass.range_constexpr(8):
        result[i] = values[i]
    if cutlass.const_expr(has_exact_path(cfg)):
        steep_rows = cutlass.const_expr(cfg.d_k // len(cfg.compute_group_0_warp_ids))
        # Visit only set ballot bits, in increasing channel order. A full K scan
        # spends most of its time testing mild channels in a sparse steep tile.
        for scan_warp in cutlass.range(len(cfg.compute_group_0_warp_ids), unroll=1):
            remaining = (words + scan_warp).load().to(cutlass.Uint32)
            while remaining != 0:
                bit = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(remaining)).to(cutlass.Int32)
                channel = scan_warp * steep_rows + bit
                for cs in cutlass.range_constexpr(2):
                    value = value_base + lane // 4 + cs * 8
                    state = (scratch + value * cfg.d_k + channel).load()
                    for rep in cutlass.range_constexpr(2):
                        for n in cutlass.range_constexpr(2):
                            token = lane % 4 * 2 + rep * 8 + n
                            key = (keys + token * cfg.d_k + channel).load()
                            i = rep * 4 + cs * 2 + n
                            result[i] = result[i] + state * key
                remaining = remaining & (remaining - cutlass.Uint32(1))
    return cutlass.Vector.from_elements(tuple(result[i] for i in range(8)), cutlass.Float32)


@cute.jit
def steep_dk_restore(cfg: cutlass.Constexpr, scratch, u_ptr, channel, token, decay):
    """Compute the steep restored-key derivative without a narrow dstate operand."""
    total = cutlass.Float32(0.0)
    for value in cutlass.range(cfg.d_v, unroll=8):
        state = (scratch + value * cfg.d_k + channel).load() * decay
        offset = (value // 64) * (cfg.b_t * 64) + token * 64 + swizzle_xor_128b(token, value % 64, elem_bytes=2)
        total = total + state * (u_ptr + offset).load().to(cutlass.Float32)
    return total
