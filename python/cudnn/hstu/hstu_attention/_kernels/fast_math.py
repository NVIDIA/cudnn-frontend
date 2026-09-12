# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: MIT

from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32, Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from .utils import fma_packed_f32x2, mul_packed_f32x2, sub_packed_f32x2, tanhf


@cute.jit
def clz(x: Int32) -> Int32:
    # CuTe DSL does not support an early return from this loop.
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res


def find_log2(x: Int32) -> Int32:
    a: Int32 = Int32(31 - clz(x))
    return a + ((x & (x - 1)) != 0)  # Round up, add 1 if not a power of 2.


@dsl_user_op
def umulhi(a: Int32, b: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class FastDivmod:
    def __init__(
        self,
        divisor: Int32,
        multiplier: Uint32,
        shift_right: Uint32,
        *,
        loc=None,
        ip=None,
    ):
        self.divisor = divisor
        self.multiplier = multiplier
        self.shift_right = shift_right
        self._loc = loc

    # called by host
    @staticmethod
    def create(divisor: Int32, *, loc=None, ip=None) -> "FastDivmod":
        """Construct the FastDivmod object, in host code.
        This precomputes some values based on the divisor and is computationally expensive.
        """
        p = Uint32(31 + find_log2(divisor))
        divisor_u32 = Uint32(divisor)
        multiplier = Uint32(((cutlass.Uint64(1) << p) + divisor_u32 - 1) // divisor_u32)
        shift_right = Uint32(p - 32)
        return FastDivmod(divisor, multiplier, shift_right, loc=loc, ip=ip)

    @cute.jit
    def div(self, dividend: Int32) -> Int32:
        return Int32(umulhi(dividend, self.multiplier) >> self.shift_right) if self.divisor != 1 else dividend

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.divisor, self.multiplier, self.shift_right]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.divisor, self.multiplier, self.shift_right], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FastDivmod(*(tuple(obj_list)), loc=self._loc)


class FastSilU:
    def __init__(
        self,
        score_scale: Float32,
        score_scale_half: Optional[Float32] = None,
        loc=None,
        ip=None,
    ):
        self._loc = loc
        self.score_scale = score_scale
        self.score_scale_half = score_scale * 0.5 if score_scale_half is None else score_scale_half

    @cute.jit
    def silu_x2(
        self,
        acc_S: cute.Tensor,
        acc_S_converted: cute.Tensor,
        preds: Optional[cute.Tensor],
        r2p_masks: Optional[cute.Tensor] = None,
        mask_fn: Optional[Callable] = None,
        r2p_mask_fn: Optional[Callable] = None,
    ):
        for i in cutlass.range_constexpr(0, cute.size(acc_S), 2):
            v0, v1 = mul_packed_f32x2(
                (acc_S[i], acc_S[i + 1]),
                (self.score_scale_half, self.score_scale_half),
            )
            tanh_v0 = tanhf(v0)
            tanh_v1 = tanhf(v1)
            out_v0, out_v1 = fma_packed_f32x2(
                (v0, v1),
                (tanh_v0, tanh_v1),
                (v0, v1),
            )
            acc_S[i] = out_v0
            acc_S[i + 1] = out_v1
        if const_expr(mask_fn is not None):
            for i in cutlass.range_constexpr(cute.size(acc_S), unroll_full=True):
                acc_S[i] = acc_S[i] if preds[i] else acc_S.element_type(0)
        if const_expr(r2p_mask_fn is not None):
            for chunk in cutlass.range_constexpr(cute.size(r2p_masks), unroll_full=True):
                keep_mask = r2p_masks[chunk]
                for bit in cutlass.range_constexpr(32, unroll_full=True):
                    i = chunk * 32 + bit
                    in_bound = cutlass.Boolean(keep_mask & (Uint32(1) << bit))
                    acc_S[i] = acc_S[i] if in_bound else acc_S.element_type(0)
        acc_S_converted.store(acc_S.load().to(acc_S_converted.element_type))

    @cute.jit
    def dsilu_bwd_x2(
        self,
        acc_S: cute.Tensor,
        acc_P: cute.Tensor,
        preds: Optional[cute.Tensor],
        score_scale: Float32,
        mask_fn: Optional[Callable] = None,
    ):
        for i in cutlass.range_constexpr(0, cute.size(acc_S), 2):
            v0, v1 = mul_packed_f32x2((acc_S[i], acc_S[i + 1]), (score_scale, score_scale))
            tanh_in0, tanh_in1 = mul_packed_f32x2((v0, v1), (0.5, 0.5))
            tanh_v0 = tanhf(tanh_in0)
            tanh_v1 = tanhf(tanh_in1)
            sigmoid_v0, sigmoid_v1 = fma_packed_f32x2((0.5, 0.5), (tanh_v0, tanh_v1), (0.5, 0.5))
            if const_expr(mask_fn is not None):
                sigmoid_v0 = sigmoid_v0 if preds[i] else acc_S.element_type(0)
                sigmoid_v1 = sigmoid_v1 if preds[i + 1] else acc_S.element_type(0)
            out_v0, out_v1 = mul_packed_f32x2((v0, v1), (sigmoid_v0, sigmoid_v1))
            one_minus_sig0, one_minus_sig1 = sub_packed_f32x2((1.0, 1.0), (sigmoid_v0, sigmoid_v1))
            inner0, inner1 = fma_packed_f32x2((v0, v1), (one_minus_sig0, one_minus_sig1), (1.0, 1.0))
            dsilu0, dsilu1 = mul_packed_f32x2((sigmoid_v0, sigmoid_v1), (inner0, inner1))
            acc_S[i] = dsilu0
            acc_S[i + 1] = dsilu1
            acc_P[i] = out_v0
            acc_P[i + 1] = out_v1

    @cute.jit
    def dsilu_bwd_quantize_x2(
        self,
        acc_S: cute.Tensor,
        acc_P: cute.Tensor,
        preds: Optional[cute.Tensor],
        r2p_mask: Optional[Uint32] = None,
    ):
        """Compute FP32 SiLU derivative and quantized SiLU output."""
        if const_expr(r2p_mask is not None):
            for i in cutlass.range_constexpr(0, cute.size(acc_S), 2):
                acc_S[i], acc_S[i + 1] = mul_packed_f32x2(
                    (acc_S[i], acc_S[i + 1]),
                    (self.score_scale_half, self.score_scale_half),
                )
            for i in cutlass.range_constexpr(
                cute.size(acc_S),
                unroll_full=True,
            ):
                valid = cutlass.Boolean(r2p_mask & (Uint32(1) << i))
                # tanh.approx(-128) saturates to -1, so both outputs become zero.
                acc_S[i] = acc_S[i] if valid else acc_S.element_type(-128.0)
        for i in cutlass.range_constexpr(0, cute.size(acc_S), 2):
            if const_expr(r2p_mask is not None):
                half_x0, half_x1 = acc_S[i], acc_S[i + 1]
            else:
                half_x0, half_x1 = mul_packed_f32x2(
                    (acc_S[i], acc_S[i + 1]),
                    (self.score_scale_half, self.score_scale_half),
                )
            tanh_v0 = tanhf(half_x0)
            tanh_v1 = tanhf(half_x1)
            out_v0, out_v1 = fma_packed_f32x2(
                (half_x0, half_x1),
                (tanh_v0, tanh_v1),
                (half_x0, half_x1),
            )
            sigmoid_v0, sigmoid_v1 = fma_packed_f32x2(
                (0.5, 0.5),
                (tanh_v0, tanh_v1),
                (0.5, 0.5),
            )
            one_minus_sig0, one_minus_sig1 = sub_packed_f32x2(
                (1.0, 1.0),
                (sigmoid_v0, sigmoid_v1),
            )
            dsilu0, dsilu1 = fma_packed_f32x2(
                (out_v0, out_v1),
                (one_minus_sig0, one_minus_sig1),
                (sigmoid_v0, sigmoid_v1),
            )
            if const_expr(preds is not None):
                valid0 = preds[i]
                valid1 = preds[i + 1]
                dsilu0 = dsilu0 if valid0 else acc_S.element_type(0)
                dsilu1 = dsilu1 if valid1 else acc_S.element_type(0)
                out_v0 = out_v0 if valid0 else acc_S.element_type(0)
                out_v1 = out_v1 if valid1 else acc_S.element_type(0)
            acc_S[i] = dsilu0
            acc_S[i + 1] = dsilu1
            acc_P[i] = acc_P.element_type(out_v0)
            acc_P[i + 1] = acc_P.element_type(out_v1)
