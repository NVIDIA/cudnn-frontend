# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Device helpers for the backward epilogue (SwiGLU-bwd elemwise).

Adjoint of ``common/moe_utils.py::swiglu_act`` (A = prob * silu(g) * u):

    dA_eff = prob * dacc            (prob = the row's topk weight; the GEMM1
                                     accumulator is pre-prob, BWD_DESIGN.md)
    du     = dA_eff * silu(g)
    dg     = dA_eff * u * s * (1 + g * (1 - s)),   s = sigmoid(g)

Same fastmath/exp2 sigmoid construction as swiglu_act so forward/backward
use identical nonlinearity approximations.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32

import megamoe.repo_path  # noqa: F401

from common.megamoe_constants import Log2E


@cute.jit
def swiglu_bwd(
    t_du: cute.Tensor,
    t_dg: cute.Tensor,
    t_dacc: cute.Tensor,
    t_up: cute.Tensor,
    t_gate: cute.Tensor,
    prob: Optional[Float32] = None,
) -> None:
    """du/dg from the GEMM1 accumulator block + stashed raw gate/up block."""
    for i in cutlass.range_constexpr(0, cute.size(t_du), 2):
        # s = sigmoid(gate) = 1 / (1 + exp2(-gate * log2(e)))
        neg = cute.arch.mul_packed_f32x2(
            (t_gate[i], t_gate[i + 1]), (-Log2E, -Log2E), rnd="rn", ftz=False,
        )
        s0, s1 = cute.arch.add_packed_f32x2(
            (
                cute.math.exp2(neg[0], fastmath=True),
                cute.math.exp2(neg[1], fastmath=True),
            ),
            (1.0, 1.0),
        )
        s0 = cute.arch.rcp_approx(s0)
        s1 = cute.arch.rcp_approx(s1)

        da0, da1 = t_dacc[i], t_dacc[i + 1]
        if cutlass.const_expr(prob is not None):
            da0, da1 = cute.arch.mul_packed_f32x2(
                (da0, da1), (prob, prob), rnd="rn", ftz=False,
            )

        # du = da * s * g   (silu(g) = s * g)
        silu0, silu1 = cute.arch.mul_packed_f32x2(
            (s0, s1), (t_gate[i], t_gate[i + 1]), rnd="rn", ftz=False,
        )
        t_du[i], t_du[i + 1] = cute.arch.mul_packed_f32x2(
            (da0, da1), (silu0, silu1), rnd="rn", ftz=False,
        )

        # dg = da * u * s * (1 + g - g*s)
        gs0, gs1 = cute.arch.mul_packed_f32x2(
            (t_gate[i], t_gate[i + 1]), (s0, s1), rnd="rn", ftz=False,
        )
        f0 = 1.0 + t_gate[i] - gs0
        f1 = 1.0 + t_gate[i + 1] - gs1
        dau0, dau1 = cute.arch.mul_packed_f32x2(
            (da0, da1), (t_up[i], t_up[i + 1]), rnd="rn", ftz=False,
        )
        sf0, sf1 = cute.arch.mul_packed_f32x2(
            (s0, s1), (f0, f1), rnd="rn", ftz=False,
        )
        t_dg[i], t_dg[i + 1] = cute.arch.mul_packed_f32x2(
            (dau0, dau1), (sf0, sf1), rnd="rn", ftz=False,
        )
