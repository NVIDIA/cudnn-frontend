# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""FP32 math of the MXFP8 SwiGLU backward (the routed ``swiglu_bwd_pair``, the shared expert's
unclamped element backward, SiTU-GLU and the router-weight partial), one PTX instruction at a time.

The results have to be bit-identical to the reference, which is fast-math arithmetic: every FP32
add / multiply / compare / min / max flushes subnormals (``.ftz``), ``a * b + c`` contracts to one
``fma.rn.ftz``, ``exp(-x)`` is ``ex2.approx.ftz`` of ``x * -log2(e)`` and ``1.0f / x`` is
``rcp.approx.ftz``.  The one exception is the shared expert's unclamped backward, whose multiplies
and adds round to nearest without flush-to-zero (``swiglu_bwd_shared``).  cutlass-dsl 4.5.2 has no
global ftz switch, so the helpers below spell each instruction out as inline PTX; the instruction
choice and the operation order of every function here are part of that contract.

Private device module of the MXFP8 backward.
"""

from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ._compat.activation import swiglu

from ._mxfp8_quant import fma_ftz, mul_ftz

# exp(-x) = ex2.approx.ftz(x * -log2(e)); this is that float32 exactly (0xBFB8AA3B).
NEG_LOG2E = -1.44269502162933349609375
LOG2E = 1.44269502162933349609375  # 0x3FB8AA3B


def _f32_op(text: str, *operands, loc=None, ip=None) -> Float32:
    constraint = "=f," + ",".join("f" for _ in operands)
    return Float32(
        llvm.inline_asm(
            T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in operands], text, constraint, has_side_effects=False, asm_dialect=0, loc=loc, ip=ip
        )
    )


@dsl_user_op
def add_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_op("add.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def sub_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_op("sub.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def min_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_op("min.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def max_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_op("max.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def neg_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_op("neg.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``mul.rn.f32``, without flush-to-zero."""
    return _f32_op("mul.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``add.rn.f32``, without flush-to-zero."""
    return _f32_op("add.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def div_approx_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``div.approx.ftz.f32``: fast-math ``a / b``."""
    return _f32_op("div.approx.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def ex2_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_op("ex2.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def rcp_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_op("rcp.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def tanh_approx(a: Float32, *, loc=None, ip=None) -> Float32:
    """``tanh.approx.f32``: fast-math ``tanhf``."""
    return _f32_op("tanh.approx.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def select_ftz(compare: str, a: Float32, b: Float32, if_true: Float32, if_false: Float32, *, loc=None, ip=None) -> Float32:
    """``setp.<compare>.ftz.f32 p, a, b; selp.f32 d, if_true, if_false, p``."""
    return _f32_op("{\n.reg .pred p;\nsetp." + compare + ".ftz.f32 p, $1, $2;\nselp.f32 $0, $3, $4, p;\n}", a, b, if_true, if_false, loc=loc, ip=ip)


@dsl_user_op
def inverse_router_weight(weight: Float32, *, loc=None, ip=None) -> Float32:
    """``w > 0.0f ? 1.0f / w : 0.0f`` (``setp.gt.ftz`` + ``rcp.approx.ftz``)."""
    return select_ftz("gt", weight, Float32(0.0), rcp_ftz(weight, loc=loc, ip=ip), Float32(0.0), loc=loc, ip=ip)


def swiglu_bwd_pair(gate: Float32, up: Float32, d_hidden: Float32, limit: Float32, clamped: bool):
    """One element of the SwiGLU backward, clamped or not (``clamped`` is compile time):
    ``(d_gate, d_up, hidden)`` in FP32.

    ``d_gate = (dsilu * up) * d_hidden`` with ``dsilu = fma((1 - silu), sigmoid, silu)``;
    ``d_up = silu * d_hidden``; ``hidden = silu * up``.  Clamped: the gradient of a gate
    above ``limit`` (or NaN, ``setp.gtu``) and of an up outside ``[-limit, limit]`` is 0, and
    both inputs are clamped first (``min`` / ``max``, ``.ftz``).  The caller packs two
    elements with ``cvt.rn.bf16x2.f32``.
    """
    if clamped:
        negative_limit = neg_ftz(limit)
        gate_masked = select_ftz("gtu", gate, limit, Float32(1.0), Float32(0.0))
        up_in = select_ftz("ge", up, negative_limit, Float32(1.0), Float32(0.0))
        up_in = select_ftz("le", up, limit, up_in, Float32(0.0))
        gate = min_ftz(gate, limit)
        up = min_ftz(max_ftz(up, negative_limit), limit)
    exponent = ex2_ftz(mul_ftz(gate, Float32(NEG_LOG2E)))
    sigmoid = rcp_ftz(add_ftz(exponent, Float32(1.0)))
    silu = mul_ftz(gate, sigmoid)
    dsilu = fma_ftz(sigmoid, sub_ftz(Float32(1.0), silu), silu)
    d_gate = mul_ftz(d_hidden, mul_ftz(up, dsilu))
    d_up = mul_ftz(d_hidden, silu)
    if clamped:
        d_gate = select_ftz("gt", gate_masked, Float32(0.0), Float32(0.0), d_gate)
        d_up = select_ftz("gt", up_in, Float32(0.0), d_up, Float32(0.0))
    return d_gate, d_up, mul_ftz(up, silu)


def swiglu_bwd_shared(gate: Float32, up: Float32, d_hidden: Float32):
    """One element of the shared expert's unclamped SwiGLU backward.  Unlike ``swiglu_bwd_pair``
    its multiplies and adds do not flush to zero (``mul.rn`` / ``add.rn``), ``exp`` is ``mul.ftz``
    + ``ex2.approx.ftz`` and every ``/`` is ``div.approx.ftz``; the operation order below is part of
    the bit-exact contract.  Returns ``(d_gate, d_up)``.
    """
    exponent = ex2_ftz(mul_ftz(mul_rn(gate, Float32(-1.0)), Float32(LOG2E)))  # exp(-gate)
    denominator = add_rn(exponent, Float32(1.0))  # 1 / sigmoid(gate)
    silu = div_approx_ftz(gate, denominator)
    dsilu = add_rn(div_approx_ftz(add_rn(mul_rn(silu, Float32(-1.0)), Float32(1.0)), denominator), silu)
    d_up = mul_rn(silu, d_hidden)
    d_gate = mul_rn(mul_rn(d_hidden, dsilu), up)
    return d_gate, d_up


def situ_fwd(gate: Float32, up: Float32, beta1: Float32, beta2: Float32) -> Float32:
    """SiTU-GLU's hidden for one element:
    ``(beta1 * tanh(gate / beta1) * sigmoid(gate)) * (beta2 * tanh(up / beta2))``.

    Bit-exact instruction sequence: every ``/`` is ``div.approx.ftz``, ``tanh`` is
    ``tanh.approx``, ``exp(-g)`` is ``ex2.approx.ftz(g * -log2 e)``, products are ``mul.ftz``, and
    the sigmoid is a division, not a reciprocal -- ``beta1 * tanh(...) / (1 + exp(-g))``.
    """
    gate_term = mul_ftz(beta1, tanh_approx(div_approx_ftz(gate, beta1)))
    denominator = add_ftz(ex2_ftz(mul_ftz(gate, Float32(NEG_LOG2E))), Float32(1.0))
    gate_term = div_approx_ftz(gate_term, denominator)
    up_term = mul_ftz(beta2, tanh_approx(div_approx_ftz(up, beta2)))
    return mul_ftz(gate_term, up_term)


def situ_bwd(gate: Float32, up: Float32, d_hidden: Float32, beta1: Float32, beta2: Float32):
    """SiTU-GLU's backward for one element: ``(d_gate, d_up, hidden)`` in FP32.

    ``s = sigmoid(g)``, ``t = tanh(g / beta1)``, ``u = tanh(up / beta2)``, ``a = beta1 t s``,
    ``b = beta2 u``; ``d_gate = dh * b * ((1 - t^2) s + a (1 - s))``, ``d_up = dh * a * (1 - u^2)``,
    ``hidden = a * b``.  Bit-exact instruction sequence: here the sigmoid *is* ``rcp.approx.ftz``
    (unlike ``situ_fwd``), and the one contraction is ``fma.rn.ftz(s, 1 - t^2, a * (1 - s))``.
    """
    sigmoid = rcp_ftz(add_ftz(ex2_ftz(mul_ftz(gate, Float32(NEG_LOG2E))), Float32(1.0)))
    t = tanh_approx(div_approx_ftz(gate, beta1))
    u = tanh_approx(div_approx_ftz(up, beta2))
    a = mul_ftz(sigmoid, mul_ftz(beta1, t))
    b = mul_ftz(beta2, u)
    d_hidden_b = mul_ftz(d_hidden, b)
    one_minus_t2 = sub_ftz(Float32(1.0), mul_ftz(t, t))
    a_one_minus_s = mul_ftz(sub_ftz(Float32(1.0), sigmoid), a)
    d_gate = mul_ftz(d_hidden_b, fma_ftz(sigmoid, one_minus_t2, a_one_minus_s))
    d_up = mul_ftz(mul_ftz(d_hidden, a), sub_ftz(Float32(1.0), mul_ftz(u, u)))
    return d_gate, d_up, mul_ftz(a, b)


def glu_forward(gate: Float32, up: Float32, situ: bool, beta1=None, beta2=None) -> Float32:
    """The MLP activation's hidden for one element: SiTU-GLU when ``situ`` (a compile-time bool),
    else SwiGLU -- QuACK's ``swiglu``, which matches the reference bit for bit."""
    if situ:
        return situ_fwd(gate, up, beta1, beta2)
    return swiglu(gate, up)


def glu_backward(gate: Float32, up: Float32, d_hidden: Float32, limit: Float32, clamped: bool, situ: bool = False, beta1=None, beta2=None):
    """The MLP activation's backward for one element: ``(d_gate, d_up, hidden)``, SiTU-GLU's when
    ``situ`` (compile-time) and SwiGLU's (``swiglu_bwd_pair``) otherwise."""
    if situ:
        return situ_bwd(gate, up, d_hidden, beta1, beta2)
    return swiglu_bwd_pair(gate, up, d_hidden, limit, clamped)


def router_partial(
    partial: Float32, inverse_weight: Float32, d_hidden_low: Float32, d_hidden_high: Float32, hidden_low: Float32, hidden_high: Float32
) -> Float32:
    """``partial += dh.x * inv * h.x + dh.y * inv * h.y`` with the reference's contraction:
    ``add(partial, fma(inv * dh.x, h.x, (inv * dh.y) * h.y))``."""
    high = mul_ftz(mul_ftz(inverse_weight, d_hidden_high), hidden_high)
    return add_ftz(partial, fma_ftz(mul_ftz(inverse_weight, d_hidden_low), hidden_low, high))


__all__ = [
    "add_ftz",
    "add_rn",
    "div_approx_ftz",
    "ex2_ftz",
    "glu_backward",
    "glu_forward",
    "inverse_router_weight",
    "max_ftz",
    "min_ftz",
    "mul_rn",
    "neg_ftz",
    "rcp_ftz",
    "router_partial",
    "select_ftz",
    "situ_bwd",
    "situ_fwd",
    "sub_ftz",
    "swiglu_bwd_pair",
    "swiglu_bwd_shared",
    "tanh_approx",
]
