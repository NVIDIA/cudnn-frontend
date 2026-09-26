# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM90 WGMMA helpers for SW128 shared memory and FP32 accumulators.

The supported forms are SS ``m64n64k16``/``m64n256k16`` and RS
``m64n256k16`` with F16 or BF16 inputs. For warpgroup-local thread ``t``, C
slot ``i`` maps to row ``16*(t//32) + (t%32)//4 + 8*((i//2)%2)`` and column
``2*(t%4) + i%2 + 8*(i//4)``; RS A words use the same rows. Callers frame the
instructions with the WGMMA fence, commit, and wait primitives and keep every
operand live until the wait completes.

The SS path calls the raw NVVM dialect op with explicit descriptor IR values:
the ``nvidia-cutlass-dsl 4.8.0.dev0`` public wrapper re-wraps descriptor DSL
values before forwarding them to an operand that requires ``ir.Value``. The RS
path uses LLVM inline assembly because the wrappers do not expose register A
for this instruction form.
"""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm


@cute.jit
def wgmma_smem_desc(
    region,
    *,
    leading_byte_offset: cutlass.Constexpr[int],
    stride_byte_offset: cutlass.Constexpr[int],
) -> prims.WgmmaSmemDesc:
    """Encode ``region`` as an SW128 WGMMA descriptor.

    The region base must be 16-byte aligned and below 256 KiB. The byte offsets
    become 16-byte fields; callers may add 16-byte units to the returned start
    field for tiles inside the region.
    """
    leading, stride = _field("leading_byte_offset", leading_byte_offset), _field("stride_byte_offset", stride_byte_offset)
    start = cutlass.Int64(region.data_ptr().toint(cutlass.Uint32)) >> 4
    return prims.WgmmaSmemDesc(start | (int(prims.WgmmaSwizzle.SWIZZLE_128B) << 62 | stride << 32 | leading << 16))


def _field(name, byte_offset):
    if byte_offset % 16 != 0:
        raise ValueError(f"wgmma_smem_desc: {name} must be a multiple of 16, got {byte_offset}")
    units = byte_offset // 16
    if not 0 <= units < 1 << 14:
        raise ValueError(f"wgmma_smem_desc: {name} must be a multiple of 16 below 256 KiB, got {byte_offset}")
    return units


@dsl_user_op
def wgmma_m64nNk16_f32(c, a, b_desc, ab_dtype, *, transpose_b=False, accumulate=True, loc=None, ip=None) -> cutlass.Vector:
    """Dispatch one ``m64n{64|256}k16`` WGMMA and return its FP32 C vector.

    SS A is a SMEM descriptor/value and accepts 32 C slots (n64) or 128
    slots (n256). RS A is a four-word ``Vector`` and accepts only 128 C slots.
    B is always a SMEM descriptor/value; ``transpose_b`` selects its major
    layout. The caller owns the surrounding fence/commit/wait and operand
    lifetime.
    """
    if cutlass.const_expr(ab_dtype != cutlass.Float16 and ab_dtype != cutlass.BFloat16):
        raise TypeError(f"A/B dtype must be Float16 or BFloat16, got {ab_dtype}")
    register_a = isinstance(a, cutlass.Vector)
    if cutlass.const_expr(register_a):
        if cutlass.const_expr(c.shape[0] != 128):
            raise ValueError(f"wgmma_m64nNk16_f32: RS A requires 128 C slots (m64n256), got {c.shape[0]}")
        if cutlass.const_expr(a.shape != (4,)):
            raise ValueError(f"wgmma_m64nNk16_f32: RS A must be four packed words, got shape {a.shape}")
    elif cutlass.const_expr(c.shape[0] not in (32, 128)):
        raise ValueError(f"wgmma_m64nNk16_f32: SS C must hold 32 or 128 slots, got {c.shape[0]}")
    kind = nvvm.WGMMATypes.f16 if cutlass.const_expr(ab_dtype == cutlass.Float16) else nvvm.WGMMATypes.bf16
    return _tied(_wgmma_rs if cutlass.const_expr(register_a) else _wgmma_ss, c, a, b_desc, kind, transpose_b, accumulate, loc=loc, ip=ip)


def _tied(issue, c, *args, loc, ip):
    """Adapt the tied struct result of one WGMMA to a FP32 ``Vector``."""
    acc_type = llvm.StructType.get_literal([cutlass.Float32.mlir_type] * c.shape[0])
    result = issue(acc_type, [value.ir_value(loc=loc, ip=ip) for value in c.to_elements()], *args, loc=loc, ip=ip)
    values = tuple(cutlass.Float32(llvm.extractvalue(cutlass.Float32.mlir_type, result, [i], loc=loc, ip=ip)) for i in range(c.shape[0]))
    return cutlass.Vector.from_elements(values, cutlass.Float32, loc=loc, ip=ip)


def _wgmma_ss(acc_type, acc, a, b_desc, kind, transpose_b, accumulate, *, loc, ip):
    """Issue an SS WGMMA through the NVVM dialect op.

    The 4.8.0.dev0 public wrapper re-wraps descriptors as DSL ``Int64`` values;
    the raw op requires ``ir.Value``, so this boundary converts them explicitly.
    Re-check this escape hatch when the DSL version changes.
    """
    acc_value = llvm.mlir_undef(acc_type, loc=loc, ip=ip)
    for i, value in enumerate(acc):
        acc_value = llvm.insertvalue(acc_value, value, [i], loc=loc, ip=ip)
    return nvvm.wgmma_mma_async(
        acc_type,
        acc_value,
        cutlass.Int64(a).ir_value(loc=loc, ip=ip),
        cutlass.Int64(b_desc).ir_value(loc=loc, ip=ip),
        ir.Attribute.parse(f"#nvvm.shape<m = 64, n = {2 * len(acc)}, k = 16>"),
        type_a=kind,
        type_b=kind,
        type_d=nvvm.WGMMATypes.f32,
        scale_d=nvvm.WGMMAScaleOut.one if cutlass.const_expr(accumulate) else nvvm.WGMMAScaleOut.zero,
        scale_a=nvvm.WGMMAScaleIn.one,
        scale_b=nvvm.WGMMAScaleIn.one,
        layout_a=nvvm.MMALayout.row,
        layout_b=nvvm.MMALayout.row if cutlass.const_expr(transpose_b) else nvvm.MMALayout.col,
        loc=loc,
        ip=ip,
    )


def _wgmma_rs(acc_type, acc, a, b_desc, kind, transpose_b, accumulate, *, loc, ip):
    """Issue the RS ``m64n256k16`` form through LLVM inline assembly."""
    # C outputs $0..$127 match the accumulator inputs; A is $256..$259 and B is
    # $260. Side effects keep the instruction between the caller's fence/commit.
    registers = "{" + ",".join(f"${i}" for i in range(128)) + "}, {$256,$257,$258,$259}, $260"
    accumulate_bit = int(cutlass.const_expr(accumulate))
    transpose_bit = int(cutlass.const_expr(transpose_b))
    return llvm.inline_asm(
        acc_type,
        acc + [v.ir_value(loc=loc, ip=ip) for v in (*a.to_elements(), b_desc)],
        f"wgmma.mma_async.sync.aligned.m64n256k16.f32.{kind}.{kind} {registers}, {accumulate_bit}, 1, 1, {transpose_bit};",
        ",".join(["=f"] * 128 + [str(i) for i in range(128)] + ["r"] * 4 + ["l"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
