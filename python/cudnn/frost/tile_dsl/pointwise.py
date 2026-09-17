# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import inspect
from typing import Type

import cutlass
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm
import cutlass.cute as cute
from cutlass._mlir.dialects import arith, vector
from cutlass._mlir.dialects import nvvm as nvvm_ops
from cutlass._mlir.extras import types as T_
from cutlass._mlir import ir as _ir

from .regtile import RegTile, vec_concat

# nvidia-cutlass-dsl 4.7.0a0 nightlies from 2026-07-27 (6986e65) on regenerated
# the nvvm bindings to take an explicit result type -- *_packed_f32x2(res,
# src_a, src_b) -- but the cutlass.experimental.primitives wrappers still call
# the two-positional form, so going through them raises "missing 1 required
# positional argument: 'src_b'".  Call the dialect ops directly and adapt to
# whichever binding generation is installed; older ones infer the result type.
_PACKED_RES_FIRST = "res" in inspect.signature(nvvm_ops.mul_packed_f32x2).parameters


def _packed_f32x2(op, vec_a, vec_b):
    """``op(vec_a, vec_b)`` for nvvm.{mul,add}_packed_f32x2 on f32x2 IR values."""
    if _PACKED_RES_FIRST:
        return op(vec_a.type, vec_a, vec_b, rnd=nvvm_ops.FPRoundingMode.RN)
    return op(vec_a, vec_b, rnd=nvvm_ops.FPRoundingMode.RN)


def tmem_load_max_reduction(tmem_addr, num: cutlass.Constexpr = 64):
    """tcgen05.ld.red.sync.aligned.32x32b.x{num}.f32.max — fused TMEM
    load + HW row-max reduction (LDTM.STAT).  ``num`` is the elements-per-
    thread count (= TILE_N/2 for the current dual-MMA softmax path).
    """
    _data_ops = ", ".join("{$w%d}" % i for i in range(num))
    _ptx = ("tcgen05.ld.red.sync.aligned.32x32b.x%d.f32.max " "{" + _data_ops + "}, {$w%d}, [{$r0}];") % (num, num)
    outs = inline_ptx(
        _ptx,
        write_only_types=[cutlass.Int32] * (num + 1),
        read_only_args=[tmem_addr],
    )
    return cutlass.Vector.from_elements(tuple(outs), cutlass.Int32)


def row_max_reduction(vec):
    n = int(vec.shape[0])
    elems = [vec[i] for i in range(n)]
    while len(elems) > 1:
        nxt = []
        for i in range(0, len(elems), 3):
            grp = elems[i : i + 3]
            acc = grp[0]
            for g in grp[1:]:
                acc = cute.math.max(acc, g, ftz=True)
            nxt.append(acc)
        elems = nxt
    return elems[0]


def row_reduction_pair(vec):
    n = int(vec.shape[0])
    assert n % 2 == 0, f"row_reduction_pair: N={n} must be even"
    half = n // 2
    paired_ty = _ir.VectorType.get([half, 2], T_.f32())
    paired = vector.shape_cast(paired_ty, vec.ir_value())

    acc = vector.extract(paired, dynamic_position=[], static_position=[0])
    for i in range(1, half):
        pair = vector.extract(paired, dynamic_position=[], static_position=[i])
        acc = arith.addf(acc, pair)
    return cutlass.Vector(acc, dtype=cutlass.Float32)


def tmem_load_max_reduction_x64(tmem_addr):
    return tmem_load_max_reduction(tmem_addr, num=64)


def row_max_reduction_64(vec64):
    return row_max_reduction(vec64)


def row_reduction_pair_64(vec64):
    return row_reduction_pair(vec64)


def tmem_load_tile(tmem_addr, num_elems: int, ld_num: int = 64) -> RegTile:
    assert num_elems % ld_num == 0, f"tmem_load_tile: num_elems={num_elems} must be a multiple of " f"ld_num={ld_num}"
    chunks = [
        nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(tmem_addr + cutlass.Int32(i * ld_num), cutlass.Float32),
            num=ld_num,
        )
        for i in range(num_elems // ld_num)
    ]
    return RegTile(vec_concat(chunks))


def tmem_load_max_reduction_tile(tmem_addr, num_elems: int):
    assert num_elems % 64 == 0, f"tmem_load_max_reduction_tile: num_elems={num_elems} must be a multiple of 64"
    raw_results = [tmem_load_max_reduction(tmem_addr + cutlass.Int32(i * 64), num=64) for i in range(num_elems // 64)]
    data_chunks = [cutlass.Vector.from_elements(tuple(res[:64]), cutlass.Int32).bitcast(cutlass.Float32) for res in raw_results]
    max_scalars = [cutlass.Vector.from_elements((res[64],), cutlass.Int32).bitcast(cutlass.Float32)[0] for res in raw_results]
    final_max = max_scalars[0]
    for m in max_scalars[1:]:
        final_max = cute.math.max(final_max, m)
    return RegTile(vec_concat(data_chunks)), final_max


@cute.jit
def fp32_to_fp16(lo, hi, *, dtype=cutlass.Float16):
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"fp32_to_fp16: dtype must be Float16 or BFloat16, got {dtype}")
    tag = "f16" if cutlass.const_expr(dtype == cutlass.Float16) else "bf16"
    return inline_ptx(
        f"cvt.rn.{tag}x2.f32 $0, $2, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo, hi],
    )


def fp32_to_fp8_pack(values, *, dtype: Type[cutlass.Numeric]):
    assert len(values) == 16, f"fp32_to_fp8_pack: expected 16 input values, got {len(values)}"
    if dtype == cutlass.Float8E4M3FN:
        dtype_tag = "e4m3"
    elif dtype == cutlass.Float8E5M2:
        dtype_tag = "e5m2"
    else:
        raise TypeError(f"fp32_to_fp8_pack: dtype must be Float8E4M3FN or Float8E5M2, got {dtype}")

    u0, u1, u2, u3 = inline_ptx(
        "{ .reg .b16 lo, hi;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $5,  $4;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $7,  $6;\n"
        "mov.b32 $0, {lo, hi};\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $9,  $8;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $11, $10;\n"
        "mov.b32 $1, {lo, hi};\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $13, $12;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $15, $14;\n"
        "mov.b32 $2, {lo, hi};\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $17, $16;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $19, $18;\n"
        "mov.b32 $3, {lo, hi}; }",
        write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
        read_only_args=list(values),
    )
    return cutlass.Vector.from_elements((u0, u1, u2, u3), cutlass.Int32)


def fp32_to_e2m1_pack(values):
    """Pack 16 fp32 into 8 E2M1 bytes (two Int32 words), element i in nibble i.

    ``cvt.rn.satfinite.e2m1x2.f32 d, a, b`` puts ``a`` in the upper nibble, so
    the pair (values[2j+1], values[2j]) lands as byte j.
    """
    assert len(values) == 16, f"fp32_to_e2m1_pack: expected 16 input values, got {len(values)}"
    w0, w1 = inline_ptx(
        "{ .reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, $3,  $2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, $5,  $4;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, $7,  $6;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, $9,  $8;\n"
        "mov.b32 $0, {b0, b1, b2, b3};\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, $11, $10;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, $13, $12;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, $15, $14;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, $17, $16;\n"
        "mov.b32 $1, {b0, b1, b2, b3}; }",
        write_only_types=[cutlass.Int32, cutlass.Int32],
        read_only_args=list(values),
    )
    return cutlass.Vector.from_elements((w0, w1), cutlass.Int32)


@cute.jit
def fp32_to_e2m1x2(lo: cutlass.Float32, hi: cutlass.Float32) -> cutlass.Int32:
    """Two fp32 -> one E2M1 pair byte (lo in the low nibble), returned zero-extended in an Int32."""
    return inline_ptx(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $2, $1; cvt.u32.u8 $0, b; }",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo, hi],
    )


@cute.jit
def e4m3_scale_rcp(sf: cutlass.Float32):
    """Round a positive fp32 scale to E4M3 and return ``(byte, 1/decoded)``.

    The reciprocal is of the DECODED (rounded) scale so quantized data and the
    stored scale stay self-consistent; a zero scale (all-zero block) yields 0.
    """
    byte_i32, dec = inline_ptx(
        "{ .reg .b16 s16, h2lo; .reg .b32 h2;\n"
        "cvt.rn.satfinite.e4m3x2.f32 s16, $2, $2;\n"
        "cvt.u32.u16 $0, s16;\n"
        "and.b32 $0, $0, 0xFF;\n"
        "cvt.rn.f16x2.e4m3x2 h2, s16;\n"
        "mov.b32 {h2lo, s16}, h2;\n"
        "cvt.f32.f16 $1, h2lo; }",
        write_only_types=[cutlass.Int32, cutlass.Float32],
        read_only_args=[sf],
    )
    inv = cutlass.Float32(1.0) / dec
    if dec == cutlass.Float32(0.0):
        inv = cutlass.Float32(0.0)
    return byte_i32, inv


@cute.jit
def amax_to_ue8m0_rp(amax: cutlass.Float32):
    """Encode ``amax / 448`` as a UE8M0 exponent byte (round up) and return ``(byte, 2^(127-e))``.

    Same arithmetic as the MXFP8 bprop kernels' ``cvt_amax_to_e8m0_rp``: 448 is
    ``1.75 * 2**8``, so the rounded-up scale exponent is the amax exponent minus
    eight plus one when the significand exceeds 1.75. Byte 0 (amax == 0) maps to
    an inverse of 0 so an all-zero block quantizes to zeros.
    """
    amax_bits = amax.bitcast(cutlass.Uint32)
    exponent = (amax_bits >> 23) & cutlass.Uint32(0xFF)
    mantissa = amax_bits & cutlass.Uint32(0x7FFFFF)
    scale_exp = cutlass.Int32(exponent) - cutlass.Int32(8)
    if mantissa > cutlass.Uint32(0x600000):
        scale_exp = scale_exp + cutlass.Int32(1)
    if exponent == cutlass.Uint32(0xFF):
        scale_exp = cutlass.Int32(254)
    if scale_exp < cutlass.Int32(0):
        scale_exp = cutlass.Int32(0)
    if scale_exp > cutlass.Int32(254):
        scale_exp = cutlass.Int32(254)
    inv = ((cutlass.Uint32(254) - cutlass.Uint32(scale_exp)) << 23).bitcast(cutlass.Float32)
    if scale_exp == cutlass.Int32(0):
        inv = cutlass.Float32(0.0)
    if scale_exp == cutlass.Int32(254):
        inv = cutlass.Uint32(1 << 22).bitcast(cutlass.Float32)
    return scale_exp, inv


@cute.jit
def fp32_to_fp8x2(lo: cutlass.Float32, hi: cutlass.Float32, *, dtype: cutlass.Constexpr[Type[cutlass.Numeric]] = cutlass.Float8E4M3FN) -> cutlass.Uint16:
    """Pack two fp32 into fp8 bytes: low byte = fp8(lo), byte 1 = fp8(hi)."""
    if cutlass.const_expr(dtype != cutlass.Float8E4M3FN and dtype != cutlass.Float8E5M2):
        raise TypeError(f"Invalid FP8 dtype: {dtype}")
    cvt_tag = "e4m3x2" if cutlass.const_expr(dtype == cutlass.Float8E4M3FN) else "e5m2x2"
    return cute.arch.inline_ptx(
        "{ .reg .f32 fa, fb; mov.b32 fa, {$r0}; mov.b32 fb, {$r1}; " + f"cvt.rn.satfinite.{cvt_tag}.f32 " + "{$w0}, fa, fb; }",
        write_only_types=[cutlass.Uint16],
        read_only_args=[hi.bitcast(cutlass.Int32), lo.bitcast(cutlass.Int32)],
    )


@cute.jit
def pack_fp8x2_pairs(pair0: cutlass.Uint16, pair1: cutlass.Uint16) -> cutlass.Int32:
    """Two fp8x2 halves into one 32-bit MMA A/B operand (pair0 = low half)."""
    return cute.arch.inline_ptx(
        "mov.b32 $0, {$1, $2};",
        write_only_types=[cutlass.Int32],
        read_only_args=[pair0, pair1],
    )


@cute.jit
def pack_u16x2(lo: cutlass.Uint16, hi: cutlass.Uint16) -> cutlass.Int32:
    """Two 16-bit patterns into one 32-bit word (lo = low half)."""
    return cute.arch.inline_ptx(
        "mov.b32 $0, {$1, $2};",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo, hi],
    )


def vec_scale_pair(vec, scalar, N):
    assert N % 2 == 0, f"vec_scale_pair: N={N} must be even"
    pair_ty = _ir.VectorType.get([2], T_.f32())
    paired_ty = _ir.VectorType.get([N // 2, 2], T_.f32())
    flat_ty = _ir.VectorType.get([N], T_.f32())

    scalar_pair = vector.broadcast(pair_ty, scalar.ir_value())

    paired_in = vector.shape_cast(paired_ty, vec.ir_value())
    result = paired_in
    for i in range(N // 2):
        pair = vector.extract(paired_in, dynamic_position=[], static_position=[i])
        scaled = _packed_f32x2(nvvm_ops.mul_packed_f32x2, pair, scalar_pair)
        result = vector.insert(
            scaled,
            result,
            dynamic_position=[],
            static_position=[i],
        )
    return cutlass.Vector(vector.shape_cast(flat_ty, result), dtype=cutlass.Float32)


@cutlass.cute.jit
def f16x2_to_f32(word, *, dtype=cutlass.Float16):
    """Unpack one Int32 (= 2 packed halves) into ``(lo_f32, hi_f32)`` Float32.

    The inverse of :func:`fp32_to_fp16`.  bf16 IS the top 16 bits of an fp32,
    so f32 = bf16 << 16 (bit move, no PRMT storm); masks stay 32-bit (a Python
    ``0xFFFF0000`` promotes to i64 -> mov.b32 mismatch).  fp16 needs the real
    ``cvt.f32.f16`` converts.
    """
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"f16x2_to_f32: dtype must be Float16 or BFloat16, got {dtype}")
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        lo, hi = inline_ptx(
            "{ .reg .b16 l, h; mov.b32 {l, h}, $2; mov.b32 $0, {0, l}; mov.b32 $1, {0, h}; }",
            write_only_types=[cutlass.Float32, cutlass.Float32],
            read_only_args=[word],
        )
    else:
        lo, hi = inline_ptx(
            "{ .reg .b16 h0, h1; mov.b32 {h0, h1}, $2; " "cvt.f32.f16 $0, h0; cvt.f32.f16 $1, h1; }",
            write_only_types=[cutlass.Float32, cutlass.Float32],
            read_only_args=[word],
        )
    return lo, hi


@cutlass.cute.jit
def opaque_f32_zero():
    """A 0.0f the optimizer cannot prove constant.

    Use for values that feed packed-asm operands (:func:`fmul2` /
    :func:`ffma2`) and could otherwise fold to a literal: the
    ``nvvm.inline_ptx`` lowering gives constant float operands the ``n``
    immediate constraint, which ICEs libNVVM."""
    return inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Float32])


@cutlass.cute.jit
def opaque_i32_zero():
    """A packed-zero b32 word the optimizer cannot prove constant.

    Same libNVVM immediate-constraint hazard as :func:`opaque_f32_zero`, for
    the packed 16x2 operands (:func:`sub_f16x2` / :func:`mul_f16x2`)."""
    return inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Int32])


@cutlass.cute.jit
def opaque_i32(value: cutlass.Int32) -> cutlass.Int32:
    """Identity mov.b32 that pins a per-lane loop invariant in its register."""
    return inline_ptx("mov.b32 $0, $1;", write_only_types=[cutlass.Int32], read_only_args=[value])


@cutlass.cute.jit
def fmul2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 multiply (SM100 FMUL2): ``(a_lo * b_lo, a_hi * b_hi)``.

    ``mul.f32x2`` operates on register pairs; the ``mov.b64`` packs map to
    register-pair allocation and usually fold away in SASS."""
    return inline_ptx(
        "{ .reg .b64 pa, pb, pc; mov.b64 pa, {$2, $3}; mov.b64 pb, {$4, $5}; mul.f32x2 pc, pa, pb; mov.b64 {$0, $1}, pc; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a_lo, a_hi, b_lo, b_hi],
    )


@cutlass.cute.jit
def fadd2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 add (SM100 FADD2): ``(a_lo + b_lo, a_hi + b_hi)``.

    Native NVVM op, not inline asm: libNVVM rejects ``add.f32x2`` in asm
    blocks (mul/fma made it in, add did not)."""
    vec_a = cutlass.Vector.from_elements((a_lo, a_hi), cutlass.Float32)
    vec_b = cutlass.Vector.from_elements((b_lo, b_hi), cutlass.Float32)
    res = cutlass.Vector(_packed_f32x2(nvvm_ops.add_packed_f32x2, vec_a.ir_value(), vec_b.ir_value()), dtype=cutlass.Float32)
    return cutlass.Float32(res[0]), cutlass.Float32(res[1])


@cutlass.cute.jit
def ffma2(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi):
    """Packed fp32 fma (SM100 FFMA2): ``(a_lo*b_lo + c_lo, a_hi*b_hi + c_hi)``."""
    return inline_ptx(
        "{ .reg .b64 pa, pb, pc, pd; mov.b64 pa, {$2, $3}; mov.b64 pb, {$4, $5}; mov.b64 pc, {$6, $7}; " "fma.rn.f32x2 pd, pa, pb, pc; mov.b64 {$0, $1}, pd; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a_lo, a_hi, b_lo, b_hi, c_lo, c_hi],
    )


@cutlass.cute.jit
def movmatrix_16b(value: cutlass.Int32) -> cutlass.Int32:
    """Transpose one packed m8n8 b16 register fragment."""
    return inline_ptx(
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[value],
    )


@cutlass.cute.jit
def mul_fp16x2(value: cutlass.Int32, scale: cutlass.Int32) -> cutlass.Int32:
    """Multiply two packed FP16 pairs."""
    return inline_ptx(
        "mul.f16x2 $0, $1, $2;",
        write_only_types=[cutlass.Int32],
        read_only_args=[value, scale],
    )


@cutlass.cute.jit
def sub_f16x2(lhs: cutlass.Int32, rhs: cutlass.Int32, input_dtype: cutlass.Constexpr) -> cutlass.Int32:
    """Subtract two packed pairs using the compile-time input dtype."""
    if cutlass.const_expr(input_dtype is cutlass.BFloat16):
        return inline_ptx("sub.bf16x2 $0, $1, $2;", write_only_types=[cutlass.Int32], read_only_args=[lhs, rhs])
    return inline_ptx("sub.f16x2 $0, $1, $2;", write_only_types=[cutlass.Int32], read_only_args=[lhs, rhs])


@cutlass.cute.jit
def mul_f16x2(lhs: cutlass.Int32, rhs: cutlass.Int32, input_dtype: cutlass.Constexpr) -> cutlass.Int32:
    """Multiply two packed pairs using the compile-time input dtype."""
    if cutlass.const_expr(input_dtype is cutlass.BFloat16):
        return nvvm.mul_bf16x2(lhs, rhs)
    return mul_fp16x2(lhs, rhs)


@cute.jit
def sigmoid_f16x2(logit_pair: cutlass.Int32, input_dtype: cutlass.Constexpr):
    """Sigmoid of a packed 16-bit logit pair, returned as two fp32 values."""
    logit_vec_f32 = cutlass.Vector.from_elements((logit_pair,), cutlass.Int32).bitcast(input_dtype).to(cutlass.Float32)
    return sigmoid2(logit_vec_f32[0], logit_vec_f32[1])


L2_NORM_EPS = 1.0e-12


@cute.jit
def lane_group_sum(value: cutlass.Float32, lanes: cutlass.Constexpr[int]) -> cutlass.Float32:
    """Sum ``value`` across a power-of-two group of consecutive lanes via
    butterfly shuffles (every lane ends up holding the group total)."""
    offset = lanes // 2
    while offset >= 1:
        value = value + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, value, offset, 31, kind=nvvm.Shfl.BFLY))
        offset = offset // 2
    return value


@cute.jit
def l2norm_inv(sum_sq: cutlass.Float32) -> cutlass.Float32:
    """Inverse L2 norm with the shared epsilon floor: rows at or below the
    floor normalize by ``1 / L2_NORM_EPS`` instead of dividing by zero."""
    norm_floor_sq = cutlass.Float32(L2_NORM_EPS * L2_NORM_EPS)
    return cute.math.rsqrt(cute.math.max(sum_sq, norm_floor_sq), fastmath=True)


@cute.jit
def sigmoid(x: cutlass.Float32) -> cutlass.Float32:
    """sigmoid(x) via the tanh identity (single MUFU on Blackwell)."""
    half = cutlass.Float32(0.5)
    return cute.math.tanh(x * half, approx=True) * half + half


@cute.jit
def sigmoid2(x_lo, x_hi):
    """``(sigmoid(x_lo), sigmoid(x_hi))`` via the tanh identity, with the
    halving and the scale-bias folded into one FMUL2 and one FFMA2."""
    half = opaque_f32_zero() + cutlass.Float32(0.5)
    scaled_lo, scaled_hi = fmul2(x_lo, x_hi, half, half)
    tanh_lo = cute.math.tanh(scaled_lo, approx=True)
    tanh_hi = cute.math.tanh(scaled_hi, approx=True)
    return ffma2(tanh_lo, tanh_hi, half, half, half, half)


@cute.jit
def softplus(x: cutlass.Float32) -> cutlass.Float32:
    """log(1 + exp(x)) with the linear tail (x > 20 returns x: exp saturates
    fp32 there and log1p(exp(x)) == x to fp32 precision)."""
    result = x
    if x < cutlass.Float32(20.0):
        result = cute.math.log(cutlass.Float32(1.0) + cute.math.exp(x, fastmath=True), fastmath=True)
    return result


@cute.jit
def softplus2(x_lo, x_hi):
    """``(softplus(x_lo), softplus(x_hi))`` with the ``1 + exp`` step packed
    into one FADD2 and the linear tail applied as a select."""
    one = cutlass.Float32(1.0)
    tail = cutlass.Float32(20.0)
    exp_lo = cute.math.exp(x_lo, fastmath=True)
    exp_hi = cute.math.exp(x_hi, fastmath=True)
    sum_lo, sum_hi = fadd2(exp_lo, exp_hi, one, one)
    log_lo = cute.math.log(sum_lo, fastmath=True)
    log_hi = cute.math.log(sum_hi, fastmath=True)
    return (log_lo if x_lo < tail else x_lo), (log_hi if x_hi < tail else x_hi)


@cute.jit
def ex2_f16x2(pair: cutlass.Int32, *, dtype=cutlass.Float16) -> cutlass.Int32:
    """Packed-pair ``exp2`` in ONE MUFU op: ``ex2.approx.f16x2`` (fp16) or
    ``ex2.approx.ftz.bf16x2`` (bf16).

    PTX/target contract (ISA 9.4, 9.7.4.10):

    - ``.f16x2``: PTX ISA 7.0, sm_75+. Max relative error 2^-9.9; subnormal
      inputs are supported.
    - ``.ftz.bf16x2``: PTX ISA 7.8, sm_90+. Max relative error 2^-7; ``ftz``
      is mandatory for bf16 — subnormal inputs and results flush to
      sign-preserving zero (so +/-subnormal -> +1.0; -Inf -> +0.0;
      NaN -> NaN).
    """
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"ex2_f16x2: dtype must be Float16 or BFloat16, got {dtype}")
    op = "ex2.approx.f16x2" if cutlass.const_expr(dtype == cutlass.Float16) else "ex2.approx.ftz.bf16x2"
    return inline_ptx(
        f"{op} $0, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[pair],
    )


@cute.jit
def f16x2x2_to_fp8_word(lo_pair: cutlass.Int32, hi_pair: cutlass.Int32, dtype_tag: cutlass.Constexpr, *, dtype=cutlass.Float16) -> cutlass.Int32:
    """Two packed half-pair words (elems 0..3 in order) into one 4-byte FP8 word.

    ``cvt.rn.satfinite.{e4m3,e5m2}x2.{f16x2,bf16x2}`` converts a packed pair
    directly (no f32 round-trip): byte 0 = fp8(lo of lo_pair) ... byte 3 =
    fp8(hi of hi_pair), matching :func:`fp32_to_fp8_pack`'s element order.
    ``satfinite`` semantics: NaN converts to NaN in the destination format;
    |x| > MAX_NORM saturates to sign-preserved MAX_NORM (448 e4m3 / 57344
    e5m2) — never Inf.

    PTX/target contract (ISA 9.4, 9.7.10.24):

    - ``.f16x2`` source: PTX ISA 7.8, sm_90+ (sm_89 from ISA 8.1).
    - ``.bf16x2`` source: PTX ISA 9.1, family-specific targets only
      (sm_100f/sm_110f/sm_120f or higher in family) — a CUDA 13.1+
      toolchain floor.
    """
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"f16x2x2_to_fp8_word: dtype must be Float16 or BFloat16, got {dtype}")
    if cutlass.const_expr(dtype_tag != "e4m3" and dtype_tag != "e5m2"):
        raise ValueError(f"f16x2x2_to_fp8_word: dtype_tag must be 'e4m3' or 'e5m2', got {dtype_tag}")
    src = "f16x2" if cutlass.const_expr(dtype == cutlass.Float16) else "bf16x2"
    dst = "e4m3x2" if cutlass.const_expr(dtype_tag == "e4m3") else "e5m2x2"
    return inline_ptx(
        f"{{ .reg .b16 lo, hi; cvt.rn.satfinite.{dst}.{src} lo, $1; cvt.rn.satfinite.{dst}.{src} hi, $2; mov.b32 $0, {{lo, hi}}; }}",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo_pair, hi_pair],
    )


# ---------------------------------------------------------------------------
# MXFP8 E8M0 block scales (TransformerEngine semantics; the torch oracle is
# test/python/sdpa/mxfp8_quant.py::quantize_blocks).  Ported from the DSv3
# projection kernel's hand-rolled copy (gemm/cutedsl/dense/proj_rope_mxfp8/
# gemm_proj_rope_mxfp8.py:70-107) the moment a second kernel -- the gated
# attention block's MXFP8 quantizer -- needed the same four lines.
# ---------------------------------------------------------------------------

# The fp32 rounding of 1/448 (TE ``Quantized_Limits<E4M3>::max_norm_rcp``).  The
# scale is ``cvt.rp(amax * THIS)`` -- ONE fp32 multiply by exactly these bits, so
# the kernel and the oracle agree on every rounding corner (448 -> 0x7F, 449 -> 0x80).
E8M0_RCP_E4M3_MAX_BITS = 0x3B124925


@cute.jit
def opaque_e4m3_max_rcp() -> cutlass.Float32:
    """``fp32(1/448)`` as a REGISTER operand the optimizer cannot fold.

    It multiplies the block amax right before an ``inline_ptx`` ``cvt``; a folded
    float immediate reaching an asm operand gets the ``n`` constraint and ICEs
    libNVVM (:func:`opaque_f32_zero`).  ptxas re-folds the MOV into the FMUL."""
    return inline_ptx(f"mov.b32 {{$w0}}, 0x{E8M0_RCP_E4M3_MAX_BITS:08X};", write_only_types=[cutlass.Float32])


@cute.jit
def e8m0_rcp(byte: cutlass.Int32) -> cutlass.Float32:
    """Biased E8M0 byte -> the EXACT fp32 dequant reciprocal ``2^(127 - e)`` = ``bits((254 - e) << 23)``.

    Valid for ``e <= 253`` (TE ``exp2f_rcp`` special-cases ``e == 254`` to the fp32
    subnormal 2^-127; an inf/NaN amax is out of contract for the callers here)."""
    return ((cutlass.Int32(254) - byte) << 23).bitcast(cutlass.Float32)


@cute.jit
def e8m0_from_amax(amax: cutlass.Float32):
    """``(rcp, byte)`` for one 32-element block: ``byte = cvt.rp.satfinite.ue8m0x2.f32(amax * fp32(1/448))``,
    ``rcp = bits((254 - byte) << 23)`` -- the exact power-of-two the data is multiplied by before the e4m3 cast.

    Bit-exact with ``mxfp8_quant.quantize_blocks`` for FINITE inputs (max bf16 amax * 1/448 -> ``e <= 247``;
    ``amax == 0 -> 0x00``).  ``e in {254, 255}`` (inf / NaN amax) is OUT OF CONTRACT: the oracle's
    ``exp2_rcp`` special-cases ``e == 254`` and this does not.  ``byte`` is an ``Int32`` in ``[0, 255]``.
    """
    scaled = amax * opaque_e4m3_max_rcp()
    # ue8m0x2 packs TWO scales (operand a -> upper byte, b -> lower); feeding the
    # same value twice keeps the asm immediate-free, the mask keeps the low one.
    packed = inline_ptx("cvt.rp.satfinite.ue8m0x2.f32 {$w0}, {$r0}, {$r0};", write_only_types=[cutlass.Uint16], read_only_args=[scaled])
    byte = cutlass.Int32(packed) & cutlass.Int32(0xFF)
    return e8m0_rcp(byte), byte


@cute.jit
def e8m0_pair(amax0: cutlass.Float32, amax1: cutlass.Float32):
    """``(rcp0, rcp1, packed)`` for two blocks with ONE ``cvt``: ``byte0 = packed & 0xFF`` (amax0),
    ``byte1 = (packed >> 8) & 0xFF`` (amax1) -- little-endian, so a ``st.b16`` of ``packed`` lays byte0 first.
    Same contract as :func:`e8m0_from_amax`."""
    scaled0 = amax0 * opaque_e4m3_max_rcp()
    scaled1 = amax1 * opaque_e4m3_max_rcp()
    # operand a lands in the UPPER byte: pass amax1's scale first so amax0's byte is the low one.
    packed16 = inline_ptx("cvt.rp.satfinite.ue8m0x2.f32 {$w0}, {$r1}, {$r0};", write_only_types=[cutlass.Uint16], read_only_args=[scaled0, scaled1])
    packed = cutlass.Int32(packed16) & cutlass.Int32(0xFFFF)
    return e8m0_rcp(packed & cutlass.Int32(0xFF)), e8m0_rcp((packed >> 8) & cutlass.Int32(0xFF)), packed


# ---------------------------------------------------------------------------
# max.f32 that FUSES.  ``row_max_reduction`` above keeps ``cute.math.max`` (the
# production SDPA softmax; switching it is a measured A/B on those kernels, not a
# drive-by).  New reductions reach for these two.
# ---------------------------------------------------------------------------


@cutlass.cute.jit
def fmax_f32(a: cutlass.Float32, b: cutlass.Float32) -> cutlass.Float32:
    """PTX ``max.f32`` -- the max ptxas fuses into ``FMNMX`` / ``FMNMX3``.

    ``cute.math.max`` lowers to ``arith.maxnumf``, which the CuTe-DSL -> NVVM path
    emits as compare + select.  Measured on cutlass-dsl 4.8.0.dev0 / sm_107a in the
    gated attention block's MXFP8 quantizer (two ternary abs-max trees per lane):
    ``cute.math.max`` -> 256 FSETP + 224 FSEL and ZERO FMNMX in the whole kernel;
    this ``max.f32`` -> 56 (rowwise) / 60 (columnwise) FMNMX3, and the kernel shrank
    from 1162 -> 762 and 2778 -> 1163 SASS lines.  NaN semantics are the SAME as
    ``maxnumf`` (one NaN operand -> the other operand is returned; PTX ``max.f32``
    without ``.NaN``), so this is purely a lowering choice.  ``|x|`` folds into the
    operand modifier: ``fmax_f32(abs(a), abs(b))`` is one instruction."""
    return inline_ptx("max.f32 $0, $1, $2;", write_only_types=[cutlass.Float32], read_only_args=[a, b])


def abs_max_tree(vals):
    """``max |v|`` over a Python list of ``Float32`` as a TERNARY tree on :func:`fmax_f32`.

    The :func:`row_max_reduction` shape -- each node is ``max(max(a, b), c)``, the
    DEPENDENT pair FMNMX3 fuses, ~log3(N) deep -- with the abs folded into the
    operands.  Trace-time helper (plain Python over traced values), like
    :func:`row_max_reduction`.  Finite inputs; ``len(vals) >= 1``."""
    if len(vals) == 0:
        raise ValueError("abs_max_tree: need at least one value")
    elems = [cute.math.abs(v) for v in vals]
    while len(elems) > 1:
        nxt = []
        for i in range(0, len(elems), 3):
            grp = elems[i : i + 3]
            acc = grp[0]
            for g in grp[1:]:
                acc = fmax_f32(acc, g)
            nxt.append(acc)
        elems = nxt
    return elems[0]
