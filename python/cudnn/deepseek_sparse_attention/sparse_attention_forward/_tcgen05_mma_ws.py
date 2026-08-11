# Copyright (c) 2025, Tri Dao.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Small CuTe DSL bridge for the SM100 CTA1 ``tcgen05.mma.ws`` path.

CUTLASS DSL can describe an FP16/BF16 TS MMA, but the public MMA atom does not
expose the ``.ws`` instruction variant needed by the implicit-dual QK GEMM.
Keep that one missing instruction here; descriptor construction and Q
SMEM-to-TMEM copies continue to use CuTe DSL's public layout/copy APIs.

The functions in this module only issue asynchronous tcgen05 operations.  The
caller owns TMEM allocation/lifetime and must provide the appropriate
``tcgen05.fence::{after,before}_thread_sync`` and thread/cluster
synchronization around producer/consumer hand-offs.
"""

from __future__ import annotations

import re
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Boolean, Float16, Float32, Int32, const_expr
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import llvm

from cudnn.deepseek_sparse_attention.utils.sm100 import mma_desc as sm100_desc


def _i64_to_i32x2(value: int) -> Tuple[int, int]:
    """Split a compile-time 64-bit descriptor into PTX low/high words."""

    return value & 0xFFFF_FFFF, (value >> 32) & 0xFFFF_FFFF


def _parse_pointer_swizzle(ptr: cute.Pointer) -> cute.Swizzle:
    """Recover the pointer swizzle used by the canonical SMEM B view."""

    match = re.search(r"S<(\d+),(\d+),(\d+)>", str(ptr.type.swizzle_type))
    if match is None:
        raise ValueError("s_b must use a swizzled canonical tcgen05 SMEM layout")
    return cute.make_swizzle(*(int(value) for value in match.groups()))


def _validate_ws_f16_cta1_op(op: cute.nvgpu.tcgen05.mma.MmaOp) -> None:
    """Reject operations that do not match this bridge's common PTX spelling."""

    if op.cta_group != tcgen05.CtaGroup.ONE:
        raise ValueError("tcgen05.mma.ws bridge only supports CTA group ONE")
    if op.a_dtype not in (Float16, BFloat16) or op.b_dtype is not op.a_dtype or op.acc_dtype is not Float32:
        raise TypeError("tcgen05.mma.ws bridge requires matching FP16/BF16 inputs with FP32 accumulation")


def _validate_ws_ts_f16_cta1_op(op: cute.nvgpu.tcgen05.mma.MmaOp) -> None:
    _validate_ws_f16_cta1_op(op)
    if op.a_src != tcgen05.OperandSource.TMEM:
        raise ValueError("tcgen05.mma.ws TS bridge requires a TMEM A operand")


def _validate_ws_ss_f16_cta1_op(op: cute.nvgpu.tcgen05.mma.MmaOp) -> None:
    _validate_ws_f16_cta1_op(op)
    if op.a_src != tcgen05.OperandSource.SMEM:
        raise ValueError("tcgen05.mma.ws SS bridge requires an SMEM A operand")


@cute.jit
def mma_ws_ts_f16_cta1_desc(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    tmem_c_addr: Int32,
    tmem_a_addr: Int32,
    smem_b_desc_lo: Int32,
    smem_b_desc_hi: cutlass.Constexpr[int],
    accumulate: bool | Boolean,
) -> None:
    """Issue one CTA1 FP16/BF16 TS ``tcgen05.mma.ws`` from a split B descriptor.

    ``tmem_a_addr`` and ``tmem_c_addr`` are raw 32-bit hardware TMEM
    addresses, not byte pointers.  ``smem_b_desc_lo`` must already include the
    shared-memory start address; ``smem_b_desc_hi`` is the compile-time high
    word produced by :func:`make_smem_b_desc`.

    All lanes of the issuing warp must call this function convergently.  The
    inline PTX elects the single lane that executes the asynchronous MMA.
    """

    _validate_ws_ts_f16_cta1_op(op)
    idesc = const_expr(sm100_desc.mma_op_to_idesc(op))
    desc_hi = const_expr(smem_b_desc_hi & 0xFFFF_FFFF)

    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(tmem_c_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(tmem_a_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_b_desc_lo)).ir_value(),
            Int32(desc_hi).ir_value(),
            Int32(idesc).ir_value(),
            Int32(accumulate).ir_value(),
        ],
        "{\n\t"
        ".reg .pred leader_thread;\n\t"
        ".reg .pred accumulate_pred;\n\t"
        ".reg .b64 smem_b_desc;\n\t"
        "elect.sync _|leader_thread, -1;\n\t"
        "mov.b64 smem_b_desc, {$2, $3};\n\t"
        "setp.ne.b32 accumulate_pred, $5, 0;\n\t"
        "@leader_thread tcgen05.mma.ws.cta_group::1.kind::f16 "
        "[$0], [$1], smem_b_desc, $4, accumulate_pred, 0;\n\t"
        "}",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def mma_ws_ss_f16_cta1_desc(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    tmem_c_addr: Int32,
    smem_a_desc_lo: Int32,
    smem_a_desc_hi: cutlass.Constexpr[int],
    smem_b_desc_lo: Int32,
    smem_b_desc_hi: cutlass.Constexpr[int],
    accumulate: bool | Boolean,
) -> None:
    """Issue one CTA1 FP16/BF16 SS ``tcgen05.mma.ws`` from split descriptors.

    Both low descriptor words must already include their shared-memory start
    addresses.  This is the M64N256-capable path used by H64 PV, but the
    instruction shape is intentionally taken from ``op`` rather than fixed by
    this helper.
    """

    _validate_ws_ss_f16_cta1_op(op)
    idesc = const_expr(sm100_desc.mma_op_to_idesc(op))
    desc_a_hi = const_expr(smem_a_desc_hi & 0xFFFF_FFFF)
    desc_b_hi = const_expr(smem_b_desc_hi & 0xFFFF_FFFF)

    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(tmem_c_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_a_desc_lo)).ir_value(),
            Int32(desc_a_hi).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_b_desc_lo)).ir_value(),
            Int32(desc_b_hi).ir_value(),
            Int32(idesc).ir_value(),
            Int32(accumulate).ir_value(),
        ],
        "{\n\t"
        ".reg .pred leader_thread;\n\t"
        ".reg .pred accumulate_pred;\n\t"
        ".reg .b64 smem_a_desc;\n\t"
        ".reg .b64 smem_b_desc;\n\t"
        "elect.sync _|leader_thread, -1;\n\t"
        "mov.b64 smem_a_desc, {$1, $2};\n\t"
        "mov.b64 smem_b_desc, {$3, $4};\n\t"
        "setp.ne.b32 accumulate_pred, $6, 0;\n\t"
        "@leader_thread tcgen05.mma.ws.cta_group::1.kind::f16 "
        "[$0], smem_a_desc, smem_b_desc, $5, accumulate_pred, 0;\n\t"
        "}",
        "r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _make_smem_desc(
    layout: cute.Layout,
    ptr: cute.Pointer,
    element_width: int,
    major: sm100_desc.Major,
) -> Tuple[Int32, int]:
    """Build one split tcgen05 SMEM descriptor from a canonical view."""

    swizzle = _parse_pointer_swizzle(ptr)
    descriptor_base = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, element_width, layout),
            swizzle,
            major,
        )
    )
    descriptor_base_lo, descriptor_hi = _i64_to_i32x2(descriptor_base)
    descriptor_lo = Int32(const_expr(descriptor_base_lo) | sm100_desc.make_smem_desc_start_addr(ptr))
    return descriptor_lo, const_expr(descriptor_hi)


def make_smem_a_desc(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    s_a: cute.Tensor,
) -> Tuple[Int32, int]:
    """Build the split A descriptor for one canonical SS MMA K tile."""

    _validate_ws_ss_f16_cta1_op(op)
    if cute.rank(s_a.shape) != 2:
        raise ValueError("s_a must be a rank-2 single-instruction MMA tile")
    if cute.size(s_a.shape[0]) != op.shape_mnk[0] or cute.size(s_a.shape[1]) != op.shape_mnk[2]:
        raise ValueError("s_a shape must match the MMA instruction's (M, K) shape")
    if s_a.element_type is not op.a_dtype:
        raise TypeError("s_a element type must match the MMA A operand type")

    major = sm100_desc.Major.K if op.a_major_mode == OperandMajorMode.K else sm100_desc.Major.MN
    return _make_smem_desc(s_a.layout, s_a.iterator, op.a_dtype.width, major)


def make_smem_b_desc(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    s_b: cute.Tensor,
) -> Tuple[Int32, int]:
    """Build the split tcgen05 B descriptor for one canonical MMA K tile.

    ``s_b`` must be a rank-2, swizzled SMEM tensor for exactly one instruction
    tile: ``(N, K) == (op.shape_mnk[1], op.shape_mnk[2])``.  Larger staged
    tensors should be sliced/local-tiled before calling this helper.
    """

    _validate_ws_f16_cta1_op(op)
    if cute.rank(s_b.shape) != 2:
        raise ValueError("s_b must be a rank-2 single-instruction MMA tile")
    if cute.size(s_b.shape[0]) != op.shape_mnk[1] or cute.size(s_b.shape[1]) != op.shape_mnk[2]:
        raise ValueError("s_b shape must match the MMA instruction's (N, K) shape")
    if s_b.element_type is not op.b_dtype:
        raise TypeError("s_b element type must match the MMA B operand type")

    major = sm100_desc.Major.K if op.b_major_mode == OperandMajorMode.K else sm100_desc.Major.MN
    return _make_smem_desc(s_b.layout, s_b.iterator, op.b_dtype.width, major)


@cute.jit
def mma_ws_ts_f16_cta1(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    tmem_c_addr: Int32,
    tmem_a_addr: Int32,
    s_b: cute.Tensor,
    accumulate: bool | Boolean,
) -> None:
    """Tensor convenience wrapper for :func:`mma_ws_ts_f16_cta1_desc`."""

    descriptor_lo, descriptor_hi = make_smem_b_desc(op, s_b)
    mma_ws_ts_f16_cta1_desc(op, tmem_c_addr, tmem_a_addr, descriptor_lo, descriptor_hi, accumulate)


@cute.jit
def mma_ws_ss_f16_cta1(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    tmem_c_addr: Int32,
    s_a: cute.Tensor,
    s_b: cute.Tensor,
    accumulate: bool | Boolean,
) -> None:
    """Tensor convenience wrapper for :func:`mma_ws_ss_f16_cta1_desc`."""

    descriptor_a_lo, descriptor_a_hi = make_smem_a_desc(op, s_a)
    descriptor_b_lo, descriptor_b_hi = make_smem_b_desc(op, s_b)
    mma_ws_ss_f16_cta1_desc(
        op,
        tmem_c_addr,
        descriptor_a_lo,
        descriptor_a_hi,
        descriptor_b_lo,
        descriptor_b_hi,
        accumulate,
    )


@cute.jit
def tmem_load_32dp32b32x(tmem_addr: Int32) -> Tuple[Float32, ...]:
    """Load one raw 32-column FP32 TMEM chunk into each warp's registers.

    This intentionally bypasses the public non-WS C-fragment verifier.  The
    H64 implicit-dual score layout is a ``tmem_frg_ws_1sm`` fragment, which is
    not represented by CUTLASS DSL's public MMA fragment type.
    """

    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 32),
        [Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value()],
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{"
        "$0, $1, $2, $3, $4, $5, $6, $7, "
        "$8, $9, $10, $11, $12, $13, $14, $15, "
        "$16, $17, $18, $19, $20, $21, $22, $23, "
        "$24, $25, $26, $27, $28, $29, $30, $31"
        "}, [$32];",
        ",".join(["=r"] * 32 + ["r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), result, [idx])) for idx in range(32))


@cute.jit
def tmem_store_32dp32b32x(tmem_addr: Int32, values: cute.Tensor) -> None:
    """Store one raw 32-column FP32 register chunk to each TMEM data path."""

    if const_expr(values.element_type is not Float32 or cute.size(values) != 32):
        raise ValueError("values must contain exactly 32 FP32 elements")
    register_args = [Float32(values[idx]).ir_value() for idx in range(32)]
    register_names = ", ".join(f"${idx + 1}" for idx in range(32))
    llvm.inline_asm(
        None,
        [Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value()] + register_args,
        f"tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {{{register_names}}};",
        ",".join(["r"] * 33),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def copy_q_128xk_f16_s2t_cta1(s_q: cute.Tensor, t_q: cute.Tensor) -> None:
    """Issue public S2T copies for a physical ``128 x K`` FP16/BF16 Q segment.

    Each local tile is 128 rows x 256 bits, matching one
    :class:`tcgen05.Cp128x256bOp`.  Constructing the public copy per local tile
    is intentional: a single copy built from the forged implicit-dual M64 A
    fragment is rejected by the current CuTe DSL layout verifier.

    ``s_q`` must retain the source segment's canonical SMEM pointer swizzle
    (SW128 for the 512-d no-PE segment or SW64 for the 64-d RoPE segment), and
    ``t_q`` must use physical 16-bit TMEM stride ``(131072, 1)``.  ``K`` must be
    a multiple of 16.  The public S2T copy atom performs lane election; do not
    wrap this function in ``cute.arch.elect_one()``.
    """

    if const_expr(s_q.element_type not in (Float16, BFloat16) or t_q.element_type is not s_q.element_type):
        raise TypeError("s_q and t_q must have matching FP16 or BF16 elements")
    if const_expr(cute.rank(s_q.shape) != 2 or cute.rank(t_q.shape) != 2):
        raise ValueError("s_q and t_q must be rank-2 physical Q tensors")
    q_cols = const_expr(cute.size(t_q.shape[1]))
    if const_expr(cute.size(s_q.shape[0]) != 128 or cute.size(t_q.shape[0]) != 128 or cute.size(s_q.shape[1]) != q_cols or q_cols % 16 != 0):
        raise ValueError("s_q and t_q must both have physical shape (128, K), with K divisible by 16")

    copy_atom = cute.make_copy_atom(tcgen05.Cp128x256bOp(tcgen05.CtaGroup.ONE), s_q.element_type)
    for tile_idx in cutlass.range_constexpr(q_cols // 16):
        s_tile = cute.local_tile(s_q, (128, 16), (0, tile_idx))
        t_tile = cute.local_tile(t_q, (128, 16), (0, tile_idx))
        compact_s_tile = cute.filter_zeros(s_tile)
        compact_t_tile = cute.filter_zeros(t_tile)
        s2t_copy = tcgen05.make_s2t_copy(copy_atom, compact_t_tile)
        s2t_slice = s2t_copy.get_slice(0)
        s_partition = s2t_slice.partition_S(compact_s_tile)
        d_partition = s2t_slice.partition_D(compact_t_tile)
        s_descriptor = tcgen05.get_s2t_smem_desc_tensor(s2t_copy, s_partition)
        cute.copy(s2t_copy, s_descriptor, d_partition)


__all__ = [
    "copy_q_128xk_f16_s2t_cta1",
    "make_smem_a_desc",
    "make_smem_b_desc",
    "mma_ws_ss_f16_cta1",
    "mma_ws_ss_f16_cta1_desc",
    "mma_ws_ts_f16_cta1",
    "mma_ws_ts_f16_cta1_desc",
    "tmem_load_32dp32b32x",
    "tmem_store_32dp32b32x",
]
