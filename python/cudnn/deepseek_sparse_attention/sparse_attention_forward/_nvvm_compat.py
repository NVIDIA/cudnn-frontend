# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feature-local NVVM compatibility helpers for CuTe DSL 4.5 and newer."""

from __future__ import annotations

import inspect

from cutlass import Float32, const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

# Some CuTe DSL distributions expose the generated NVVM op with an explicit
# result-type operand.  Keep the same compatibility pattern used by the
# existing block-sparse attention utilities in this repository.
_NVVM_FMAX_REQUIRES_RESULT_TYPE = (
    sum(1 for parameter in inspect.signature(nvvm.fmax).parameters.values() if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD))
    > 2
)


@dsl_user_op
def fmax_ftz_nonan(a: float | Float32, b: float | Float32, *, loc=None, ip=None) -> Float32:
    """Return ``max(a, b)`` with FTZ enabled and NaN propagation disabled."""

    operands = (
        Float32(a).ir_value(loc=loc, ip=ip),
        Float32(b).ir_value(loc=loc, ip=ip),
    )
    if const_expr(_NVVM_FMAX_REQUIRES_RESULT_TYPE):
        return Float32(
            nvvm.fmax(
                T.f32(),
                *operands,
                ftz=True,
                nan=False,
                abs=False,
                loc=loc,
                ip=ip,
            )
        )
    return Float32(
        nvvm.fmax(
            *operands,
            ftz=True,
            nan=False,
            abs=False,
            loc=loc,
            ip=ip,
        )
    )
