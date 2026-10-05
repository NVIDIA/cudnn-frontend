# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared elementwise epilogues for the Frost convolution templates."""

import cutlass
import cutlass.cute as cute
from cutlass import testing

_LOG2E = 1.4426950408889634
_TWO_LOG2E = 2.8853900817779268


def _full_like(x, value: float):
    filled = cutlass.full_like(x, cutlass.Float32(value))
    if isinstance(x, cute.TensorSSA):
        # cutlass.full_like returns a raw vector MLIR value for TensorSSA.
        # Reattach the fragment metadata so both TensorSSA arithmetic and the
        # cute.math helpers recognize it as a vector operand.
        return cute.TensorSSA(filled, x.shape, x.dtype)
    return filled


def _exp(x):
    return cute.math.exp2(x * _full_like(x, _LOG2E), fastmath=True)


def _sigmoid(x):
    return cute.math.rcp(_full_like(x, 1.0) + _exp(-x), approx=True, ftz=True)


def _tanh(x):
    # Native vector tanh currently fails during lowering. This equivalent
    # exp2/rcp form is also used by the Frost GEMM epilogue code generator.
    one = _full_like(x, 1.0)
    two = _full_like(x, 2.0)
    return one - two * cute.math.rcp(cute.math.exp2(x * _full_like(x, _TWO_LOG2E), fastmath=True) + one, approx=True, ftz=True)


def _elu(x):
    zero = _full_like(x, 0.0)
    return cute.math.max(x, zero) + _exp(cute.math.min(x, zero)) - _full_like(x, 1.0)


def _gelu(x):
    return _full_like(x, 0.5) * x * (_full_like(x, 1.0) + cute.math.erf(x * _full_like(x, 0.7071067811865475)))


def _gelu_approx_tanh(x):
    inner = _full_like(x, 0.7978845608028654) * (x + _full_like(x, 0.044715) * x * x * x)
    return _full_like(x, 0.5) * x * (_full_like(x, 1.0) + _tanh(inner))


def _logical_not(x):
    # Produce a numeric 0/1 vector without relying on vector comparison
    # lowering: step(d) is exactly 1 for d >= 0 and 0 otherwise.
    zero = _full_like(x, 0.0)
    one = _full_like(x, 1.0)
    pos = cute.math.min(cute.math.max(cute.math.floor(x) + one, zero), one)
    neg = cute.math.min(cute.math.max(cute.math.floor(-x) + one, zero), one)
    return pos * neg


def _softplus(x):
    zero = _full_like(x, 0.0)
    return cute.math.max(x, zero) + cute.math.log(_full_like(x, 1.0) + _exp(-cute.math.abs(x)))


def _swish(x):
    return x * _sigmoid(x)


def _tan(x):
    return cute.math.sin(x) * cute.math.rcp(cute.math.cos(x), approx=True, ftz=True)


# Stable module-level callables keep compile caches keyed by the semantic mode
# string instead of a fresh function object created by each graph analysis.
_EPILOGUE_OPS_WITHOUT_ATTRS = {
    "abs": cute.math.abs,
    "ceil": cute.math.ceil,
    "cos": cute.math.cos,
    "elu": _elu,
    "erf": cute.math.erf,
    "exp": _exp,
    "floor": cute.math.floor,
    "gelu": _gelu,
    "gelu_approx_tanh": _gelu_approx_tanh,
    "identity": lambda x: x,
    "log": cute.math.log,
    "logical_not": _logical_not,
    "neg": lambda x: -x,
    "reciprocal": lambda x: cute.math.rcp(x, approx=True, ftz=True),
    "rsqrt": cute.math.rsqrt,
    "sigmoid": _sigmoid,
    "sin": cute.math.sin,
    "softplus": _softplus,
    "sqrt": cute.math.sqrt,
    "swish": _swish,
    "tan": _tan,
    "tanh": _tanh,
}

_EPILOGUE_ATTRS = {
    "relu": frozenset(("negative_slope", "lower_clip", "upper_clip")),
    "leaky_relu": frozenset(("negative_slope",)),
    "swish": frozenset(("swish_beta",)),
}


def _make_relu_epilogue(**kwargs):
    negative_slope = kwargs.get("negative_slope")
    lower_clip = kwargs.get("lower_clip")
    upper_clip = kwargs.get("upper_clip")

    def op(x):
        zero = _full_like(x, 0.0)
        if negative_slope is None:
            out = cute.math.max(x, _full_like(x, 0.0 if lower_clip is None else lower_clip))
        else:
            out = cute.math.max(x, zero) + negative_slope * cute.math.min(x, zero)
            if lower_clip is not None:
                out = cute.math.max(out, _full_like(x, lower_clip))
        if upper_clip is not None:
            out = cute.math.min(out, _full_like(x, upper_clip))
        return out

    return op


def _make_leaky_relu_epilogue(**kwargs):
    negative_slope = kwargs["negative_slope"]

    def op(x):
        zero = _full_like(x, 0.0)
        return cute.math.max(x, zero) + negative_slope * cute.math.min(x, zero)

    return op


def _make_swish_epilogue(**kwargs):
    swish_beta = kwargs.get("swish_beta", 1.0)

    def op(x):
        return x * _sigmoid(x * _full_like(x, swish_beta))

    return op


_EPILOGUE_OP_WITH_ATTRS_GETTERS = {
    "relu": _make_relu_epilogue,
    "leaky_relu": _make_leaky_relu_epilogue,
    "swish": _make_swish_epilogue,
}

SUPPORTED_EPILOGUES = frozenset(_EPILOGUE_OPS_WITHOUT_ATTRS) | frozenset(_EPILOGUE_OP_WITH_ATTRS_GETTERS)


def get_epilogue_op(epilogue: str, epilogue_attrs: tuple[tuple[str, float], ...]):
    """Return the CuTe elementwise callable for a graph epilogue."""
    attrs = dict(epilogue_attrs)
    supported_attrs = _EPILOGUE_ATTRS.get(epilogue, frozenset())
    unsupported_attrs = set(attrs) - supported_attrs
    if unsupported_attrs:
        raise testing.CantImplementError(f"Unsupported attributes for convolution epilogue {epilogue!r}: {tuple(sorted(unsupported_attrs))}")
    if not supported_attrs:
        try:
            return _EPILOGUE_OPS_WITHOUT_ATTRS[epilogue]
        except KeyError as exc:
            raise testing.CantImplementError(f"Unsupported convolution epilogue: {epilogue!r}") from exc
    try:
        getter = _EPILOGUE_OP_WITH_ATTRS_GETTERS[epilogue]
    except KeyError as exc:
        raise testing.CantImplementError(f"Convolution epilogue {epilogue!r} does not accept attributes") from exc
    return getter(**attrs)


__all__ = ["SUPPORTED_EPILOGUES", "get_epilogue_op"]
