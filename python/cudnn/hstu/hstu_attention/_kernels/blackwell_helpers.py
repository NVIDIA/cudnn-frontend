# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Optional, Tuple
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass._mlir.dialects import llvm

from . import mma_sm100_desc as sm100_desc


@cute.jit
def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[cutlass.Int32] = None,
    B_idx: Optional[cutlass.Int32] = None,
    zero_init: bool | cutlass.Boolean = False,
    num_unroll_groups: int = 1,
) -> None:
    rA = tCrA if cutlass.const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if cutlass.const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    for k in cutlass.range(
        cute.size(tCrA.shape[2]),
        unroll=cute.size(tCrA.shape[2]) // num_unroll_groups,
    ):
        mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
        cute.gemm(
            mma_atom,
            acc,
            rA[None, None, k],
            rB[None, None, k],
            acc,
        )


def i64_to_i32x2(i: int) -> Tuple[int, int]:
    """Convert a 64-bit integer to a tuple of two 32-bit integers."""
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


def _tcgen05_mma_kind(op: cute.nvgpu.tcgen05.mma.MmaOp) -> str:
    if isinstance(op, tcgen05.mma.MmaF16BF16Op):
        return "f16"
    if isinstance(op, tcgen05.mma.MmaTF32Op):
        return "tf32"
    if isinstance(op, tcgen05.mma.MmaI8Op):
        return "i8"
    if isinstance(op, (tcgen05.mma.MmaFP8Op, tcgen05.mma.MmaF8F6F4Op)):
        return "f8f6f4"
    if isinstance(op, tcgen05.mma.MmaMXF8Op):
        return "mxf8f6f4"
    if isinstance(op, tcgen05.mma.MmaMXF4Op):
        return "mxf4"
    if isinstance(op, tcgen05.mma.MmaMXF4NVF4Op):
        return "mxf4nvf4"
    raise TypeError(f"Unsupported tcgen05 MMA op kind: {type(op).__name__}")


@cute.jit
def declare_ptx_smem_desc(
    smem_desc_start: cutlass.Int32,
    smem_desc_base: int,
    fragment_layout: cute.Layout,
    var_name_prefix: str,
) -> None:
    num_k_tiles = cute.size(fragment_layout.shape[2])
    smem_desc_base_lo, smem_desc_hi = i64_to_i32x2(smem_desc_base)
    offsets = [cute.crd2idx((0, 0, k), fragment_layout) for k in range(num_k_tiles)]
    smem_desc_start_lo = cutlass.Int32(smem_desc_base_lo | smem_desc_start)
    llvm.inline_asm(
        None,
        [cutlass.Int32(cute.arch.make_warp_uniform(smem_desc_start_lo)).ir_value()],
        f".reg .b32 {var_name_prefix}_lo;\n\t"
        f".reg .b64 {var_name_prefix}_<{num_k_tiles}>;\n\t"
        f"mov.b64 {var_name_prefix}_0, {{$0, {hex(smem_desc_hi)}}};\n\t"
        + "".join(
            (
                f"add.s32 {var_name_prefix}_lo, $0, "
                f"{hex(offsets[k])};\n\t"
                f"mov.b64 {var_name_prefix}_{k}, "
                f"{{{var_name_prefix}_lo, {hex(smem_desc_hi)}}};\n\t"
            )
            for k in range(1, num_k_tiles)
        ),
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def declare_ptx_idesc(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    var_name: str,
) -> None:
    idesc = cutlass.const_expr(sm100_desc.mma_op_to_idesc(op))
    llvm.inline_asm(
        None,
        [],
        f".reg .b32 {var_name};\n\t" f"mov.b32 {var_name}, {hex(idesc)};\n\t",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_precomputed_varname(
    acc_tmem_addr: cutlass.Int32,
    smem_desc_start_b: cutlass.Int32,
    smem_desc_base_b: int,
    tCrB_layout: cute.Layout,
    smem_var_name_prefix: str,
    idesc_var_name: str,
    kind: str,
    smem_offset: int,
    zero_init: bool | cutlass.Boolean = False,
    cta_group: int = 1,
) -> None:
    num_k_tiles = cute.size(tCrB_layout.shape[2])
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    offsets_b = [cute.crd2idx((0, 0, k), tCrB_layout) for k in range(num_k_tiles)]
    smem_desc_start_b_lo = cutlass.Int32(smem_desc_base_b_lo | smem_desc_start_b)
    pred_str = "p" if isinstance(zero_init, cutlass.Boolean) else "0" if zero_init else "1"
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            cutlass.Int32(not zero_init).ir_value(),
            cutlass.Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ],
        "{\n\t"
        ".reg .pred leader_thread;\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 tmem_acc;\n\t"
        ".reg .b32 smem_desc_b_lo_start;\n\t"
        ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
        ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
        f".reg .b64 smem_desc_b_<{num_k_tiles}>;\n\t"
        "elect.sync _|leader_thread, -1;\n\t"
        "mov.b32 tmem_acc, $2;\n\t"
        "mov.b32 smem_desc_b_lo_start, $0;\n\t"
        f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
        f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, "
        f"{smem_var_name_prefix}_0;\n\t"
        f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
        f"mov.b64 {smem_var_name_prefix}_0, "
        "{smem_desc_a_lo, smem_desc_a_hi};\n\t"
        "mov.b64 smem_desc_b_0, "
        "{smem_desc_b_lo_start, smem_desc_b_hi};\n\t"
        + "".join(
            (
                "mov.b64 {smem_desc_a_lo, smem_desc_a_hi}, "
                f"{smem_var_name_prefix}_{k};\n\t"
                f"add.s32 smem_desc_a_lo, smem_desc_a_lo, "
                f"{smem_offset};\n\t"
                f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, "
                f"{hex(offsets_b[k])};\n\t"
                f"mov.b64 {smem_var_name_prefix}_{k}, "
                "{smem_desc_a_lo, smem_desc_a_hi};\n\t"
                f"mov.b64 smem_desc_b_{k}, "
                "{smem_desc_b_lo, smem_desc_b_hi};\n\t"
            )
            for k in range(1, num_k_tiles)
        )
        + "setp.ne.b32 p, $1, 0;\n\t"
        f"@leader_thread tcgen05.mma.cta_group::{cta_group}."
        f"kind::{kind} [tmem_acc], {smem_var_name_prefix}_0, "
        f"smem_desc_b_0, {idesc_var_name}, {pred_str};\n\t"
        + "".join(
            (
                f"@leader_thread tcgen05.mma.cta_group::{cta_group}."
                f"kind::{kind} [tmem_acc], {smem_var_name_prefix}_{k}, "
                f"smem_desc_b_{k}, {idesc_var_name}, 1;\n\t"
            )
            for k in range(1, num_k_tiles)
        )
        + "}\n",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    A_idx: Optional[cutlass.Int32] = None,
    B_idx: Optional[cutlass.Int32] = None,
    zero_init: bool | cutlass.Boolean = False,
    tA_addr: Optional[cutlass.Int32] = None,
    cta_group: int = 1,
) -> None:
    """Issue one complete SS or TS tcgen05 GEMM without mutating tiled_mma."""
    rA = tCrA if cutlass.const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if cutlass.const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    sA_cur = None
    if cutlass.const_expr(sA is not None):
        sA_cur = sA if cutlass.const_expr(A_idx is None) else sA[None, None, None, A_idx]
    sB_cur = sB if cutlass.const_expr(B_idx is None) else sB[None, None, None, B_idx]
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    _gemm_ptx_w_idx_impl(
        mma_atom.op,
        acc.iterator.toint(),
        rA,
        rB,
        sA_cur,
        sB_cur,
        zero_init=zero_init,
        tA_addr=tA_addr,
        cta_group=cta_group,
    )


@cute.jit
def _gemm_ptx_w_idx_impl(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: cutlass.Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    zero_init: bool | cutlass.Boolean = False,
    tA_addr: Optional[cutlass.Int32] = None,
    cta_group: int = 1,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if cutlass.const_expr(not is_ts):
        assert sA is not None
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = cutlass.const_expr(sm100_desc.mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if cutlass.const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = cutlass.const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(
                    128,
                    op.a_dtype.width,
                    sA_layout[0],
                ),
                sA_swizzle,
                (sm100_desc.Major.K if cutlass.const_expr(op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K) else sm100_desc.Major.MN),
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = cutlass.const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = cutlass.const_expr(smem_desc_a_hi)
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = cutlass.const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            (sm100_desc.Major.K if cutlass.const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K) else sm100_desc.Major.MN),
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = cutlass.const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = cutlass.const_expr(smem_desc_b_hi)

    tCrA_layout = (
        tCrA.layout
        if cutlass.const_expr(not is_ts)
        else cute.recast_layout(
            32,
            tCrA.element_type.width,
            tCrA.layout,
        )
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))]
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))]
    if cutlass.const_expr(not is_ts):
        smem_desc_start_a_lo = cutlass.Int32(smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))
    smem_desc_start_b_lo = cutlass.Int32(smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))

    pred_str = "p" if isinstance(zero_init, cutlass.Boolean) else "0" if zero_init else "1"
    if cutlass.const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                cutlass.Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                cutlass.Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                cutlass.Int32(not zero_init).ir_value(),
                cutlass.Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            "mov.b64 smem_desc_a, "
            "{smem_desc_a_lo_start, smem_desc_a_hi};\n\t"
            "mov.b64 smem_desc_b, "
            "{smem_desc_b_lo_start, smem_desc_b_hi};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}."
            f"kind::{kind} [tmem_acc], smem_desc_a, "
            f"smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    "add.u32 smem_desc_a_lo, "
                    f"smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    "add.u32 smem_desc_b_lo, "
                    f"smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    "mov.b64 smem_desc_a, "
                    "{smem_desc_a_lo, smem_desc_a_hi};\n\t"
                    "mov.b64 smem_desc_b, "
                    "{smem_desc_b_lo, smem_desc_b_hi};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}."
                    f"kind::{kind} [tmem_acc], smem_desc_a, "
                    "smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        return

    # CuTe does not preserve the TMEM A offset for every TS MMA layout.
    # Pass the known base address explicitly.
    tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            cutlass.Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            cutlass.Int32(not zero_init).ir_value(),
            cutlass.Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ],
        "{\n\t"
        ".reg .pred leader_thread;\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 idesc;\n\t"
        ".reg .b32 tmem_acc;\n\t"
        ".reg .b32 tmem_a;\n\t"
        ".reg .b32 smem_desc_b_lo_start;\n\t"
        ".reg .b32 smem_desc_b_lo;\n\t"
        ".reg .b32 smem_desc_b_hi;\n\t"
        ".reg .b64 smem_desc_b;\n\t"
        "elect.sync _|leader_thread, -1;\n\t"
        f"mov.b32 idesc, {hex(idesc)};\n\t"
        "mov.b32 tmem_acc, $3;\n\t"
        "mov.b32 tmem_a, $0;\n\t"
        "mov.b32 smem_desc_b_lo_start, $1;\n\t"
        f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
        "mov.b64 smem_desc_b, "
        "{smem_desc_b_lo_start, smem_desc_b_hi};\n\t"
        "setp.ne.b32 p, $2, 0;\n\t"
        f"@leader_thread tcgen05.mma.cta_group::{cta_group}."
        f"kind::{kind} [tmem_acc], [tmem_a], smem_desc_b, "
        f"idesc, {pred_str};\n\t"
        + "".join(
            (
                "add.u32 smem_desc_b_lo, smem_desc_b_lo_start, "
                f"{hex(offset_b[k])};\n\t"
                "mov.b64 smem_desc_b, "
                "{smem_desc_b_lo, smem_desc_b_hi};\n\t"
                f"@leader_thread tcgen05.mma.cta_group::{cta_group}."
                f"kind::{kind} [tmem_acc], "
                f"[tmem_a + {hex(offset_a[k])}], smem_desc_b, "
                "idesc, 1;\n\t"
            )
            for k in range(1, cute.size(tCrA.shape[2]))
        )
        + "}\n",
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: cutlass.Constexpr[int],
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    sA_swizzle: Optional[cute.Swizzle],
    sB_swizzle: cute.Swizzle,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[cutlass.Int32] = None,
    split_arrive: Optional[int] = None,
    zero_init: bool | cutlass.Boolean = False,
    cta_group: int = 1,
    acc_tmem_addr_dynamic: Optional[cutlass.Int32] = None,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if cutlass.const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
        assert sA_swizzle is not None, "sA_swizzle must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = cutlass.const_expr(sm100_desc.mma_op_to_idesc(op))
    if cutlass.const_expr(not is_ts):
        smem_desc_base_a: int = cutlass.const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                (sm100_desc.Major.K if cutlass.const_expr(op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K) else sm100_desc.Major.MN),
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = cutlass.const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = cutlass.const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    smem_desc_base_b: int = cutlass.const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            (sm100_desc.Major.K if cutlass.const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K) else sm100_desc.Major.MN),
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = cutlass.const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = cutlass.const_expr(smem_desc_b_hi)

    tCrA_layout = tCrA.layout if cutlass.const_expr(not is_ts) else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))]
    offset_a_diff = [offset_a[k] - offset_a[k - 1] for k in range(1, cute.size(tCrA.shape[2]))]
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))]
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))]

    if cutlass.const_expr(not is_ts):
        smem_desc_start_a_lo = cutlass.Int32(smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = cutlass.Int32(smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    pred_str = "p" if isinstance(zero_init, cutlass.Boolean) else "0" if zero_init else "1"
    if cutlass.const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                cutlass.Int32(smem_desc_start_a_lo).ir_value(),
                cutlass.Int32(smem_desc_start_b_lo).ir_value(),
                cutlass.Int32(not zero_init).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            "mov.b32 smem_desc_a_lo, $0;\n\t"
            "mov.b32 smem_desc_b_lo, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        input_args = [
            cutlass.Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
            cutlass.Int32(smem_desc_start_b_lo).ir_value(),
            cutlass.Int32(not zero_init).ir_value(),
        ]
        if cutlass.const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, "mbar_phase must be provided when mbar_ptr is not None"
            assert split_arrive is not None, "split_arrive must be provided when mbar_ptr is not None"
            split_arrive_idx = split_arrive // op.shape_mnk[2]
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(cutlass.Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$3], $4, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            split_arrive_idx = None
            mbar_wait_str = ""
        if cutlass.const_expr(acc_tmem_addr_dynamic is not None):
            acc_tmem_addr_operand = len(input_args)
            input_args.append(cutlass.Int32(acc_tmem_addr_dynamic).ir_value())
            set_acc_tmem_addr = f"mov.b32 tmem_acc, ${acc_tmem_addr_operand};\n\t"
        else:
            set_acc_tmem_addr = f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
        llvm.inline_asm(
            None,
            # [
            #     # acc.iterator.toint().ir_value(),
            #     cutlass.Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
            #     cutlass.Int32(smem_desc_start_b_lo).ir_value(),
            #     cutlass.Int32(not zero_init).ir_value(),
            # ],
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t" + set_acc_tmem_addr + f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    (cute.size(tCrA.shape[2]) if cutlass.const_expr(mbar_ptr is None) else split_arrive_idx),
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(split_arrive_idx, cute.size(tCrA.shape[2]))
                )
                if cutlass.const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            # "r,r,r",
            ",".join("r" for _ in input_args),
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
