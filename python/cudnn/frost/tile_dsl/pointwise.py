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
