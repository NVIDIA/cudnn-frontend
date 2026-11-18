# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ready-made reducers for ``custom_reduction``.

Each function returns keyword arguments of ``custom_reduction``; pass them with ``**``.
The semantic keys are the keys of the C++ ``cudnn_frontend::graph::reduction_modifiers``.

- ``sum``, ``max`` and ``min`` (one stream) return ``reducer`` and ``reducer_semantic_key``.
  They set no identity and no compute type.
- ``argmax`` and ``argmin`` (a value stream and an index stream that holds the position of each
  element along the reduced dimensions) also return ``identity_values``. They set no compute type.
"""

import enum

from . import custom_reduction_identity, data_type

__all__ = ["sum", "max", "min", "ArgNanRule", "argmax", "argmin"]


class ArgNanRule(enum.Enum):
    """How ``argmax`` and ``argmin`` order NaN.

    Each rule gives the lowest index among equal values, and -0 equals +0.

    - ``TORCH``: NaN is above every number for argmax and below every number for argmin, so
      the first NaN wins (``torch.argmax``, ``torch.argmin``).
    - ``XLA``: the XLA comparator. A NaN wins, but it is not commutative for two NaN values, so
      when a row holds more than one NaN the result is one of the NaN positions, not specified which.
    - ``IGNORE_NAN``: a NaN loses to every number; a row of NaN only gives its first position.
    """

    TORCH = "torch"
    XLA = "xla"
    IGNORE_NAN = "ignore_nan"


def _single_stream(method, name, semantic_key):
    def reducer(graph, lhs, rhs):
        compute_type = lhs[0].get_data_type()
        result = getattr(graph, method)(lhs[0], rhs[0], compute_data_type=compute_type, name=name)
        result.set_data_type(compute_type)
        return [result]

    return {"reducer": reducer, "reducer_semantic_key": semantic_key}


def sum():
    """Reducer arguments for ``lhs + rhs``."""
    return _single_stream("add", "reduction_modifiers::sum", "cudnn_frontend.reduction_modifiers.sum.v1")


def max():
    """Reducer arguments for ``max(lhs, rhs)``."""
    return _single_stream("max", "reduction_modifiers::max", "cudnn_frontend.reduction_modifiers.max.v1")


def min():
    """Reducer arguments for ``min(lhs, rhs)``."""
    return _single_stream("min", "reduction_modifiers::min", "cudnn_frontend.reduction_modifiers.min.v1")


def _op(graph, method, name, compute_type, result_type, *operands):
    result = getattr(graph, method)(*operands, compute_data_type=compute_type, name=name)
    result.set_data_type(result_type)
    return result


# The operations of the C++ detail::arg_reducer, in the same order. A compare uses the type of its operands. The
# logical modes and the select of two predicates read one-bit operands, so they use the index type.
def _arg_reducer(is_max, rule):
    better = "cmp_gt" if is_max else "cmp_lt"
    better_or_equal = "cmp_ge" if is_max else "cmp_le"
    worse_or_equal = "cmp_le" if is_max else "cmp_ge"

    def reducer(graph, lhs, rhs):
        B = data_type.BOOLEAN
        T = lhs[0].get_data_type()
        I = lhs[1].get_data_type()
        vl, il = lhs[0], lhs[1]
        vr, ir = rhs[0], rhs[1]

        if rule is ArgNanRule.XLA:
            better_l = _op(graph, better, "better", T, B, vl, vr)
            nan_l = _op(graph, "cmp_neq", "nan_l", T, B, vl, vl)
            keep_vl = _op(graph, "logical_or", "keep_value_l", I, B, better_l, nan_l)
            value = _op(graph, "binary_select", "value", T, T, vl, vr, keep_vl)
            equal = _op(graph, "cmp_eq", "equal", T, B, vl, vr)
            lower = _op(graph, "cmp_lt", "lower", I, B, il, ir)
            tie = _op(graph, "logical_and", "tie_to_l", I, B, equal, lower)
            keep_il = _op(graph, "logical_or", "keep_index_l", I, B, keep_vl, tie)
            index = _op(graph, "binary_select", "index", I, I, il, ir, keep_il)
            return [value, index]

        # TORCH and IGNORE_NAN are total orders on (value, index), so the body is commutative and associative.
        # The lhs wins on win_if_lower when its index is the lower one, else on win_if_higher.
        lower = _op(graph, "cmp_lt", "lower", I, B, il, ir)
        if rule is ArgNanRule.TORCH:
            nan_l = _op(graph, "cmp_neq", "nan_l", T, B, vl, vl)
            at_l = _op(graph, better_or_equal, "better_or_equal", T, B, vl, vr)
            win_if_lower = _op(graph, "logical_or", "win_if_lower", I, B, nan_l, at_l)
            # No compare mode is unordered, so "better or unordered" is NOT(worse or equal).
            not_better = _op(graph, worse_or_equal, "worse_or_equal", T, B, vl, vr)
            better_u = _op(graph, "logical_not", "better_or_unordered", I, B, not_better)
            number_r = _op(graph, "cmp_eq", "number_r", T, B, vr, vr)
            win_if_higher = _op(graph, "logical_and", "win_if_higher", I, B, better_u, number_r)
        else:
            at_l = _op(graph, better_or_equal, "better_or_equal", T, B, vl, vr)
            nan_r = _op(graph, "cmp_neq", "nan_r", T, B, vr, vr)
            win_if_lower = _op(graph, "logical_or", "win_if_lower", I, B, at_l, nan_r)
            better_l = _op(graph, better, "better", T, B, vl, vr)
            number_l = _op(graph, "cmp_eq", "number_l", T, B, vl, vl)
            only_r_nan = _op(graph, "logical_and", "only_r_nan", I, B, nan_r, number_l)
            win_if_higher = _op(graph, "logical_or", "win_if_higher", I, B, better_l, only_r_nan)
        keep_l = _op(graph, "binary_select", "keep_l", I, B, win_if_lower, win_if_higher, lower)
        index = _op(graph, "binary_select", "index", I, I, il, ir, keep_l)
        value = _op(graph, "binary_select", "value", T, T, vl, vr, keep_l)
        return [value, index]

    return reducer


# Bits of (-inf, +inf, quiet NaN). Any other value type gets FLOAT bits, so the graph refuses a value stream that is
# not floating point.
_FLOAT_SPECIAL_BITS = {
    data_type.HALF: (0xFC00, 0x7C00, 0x7E00),
    data_type.BFLOAT16: (0xFF80, 0x7F80, 0x7FC0),
    data_type.FLOAT: (0xFF800000, 0x7F800000, 0x7FC00000),
    data_type.DOUBLE: (0xFFF0000000000000, 0x7FF0000000000000, 0x7FF8000000000000),
}
_NEGATIVE_INFINITY, _POSITIVE_INFINITY, _QUIET_NAN = range(3)


def _rule(rule):
    if isinstance(rule, ArgNanRule):
        return rule
    try:
        return ArgNanRule(rule)
    except ValueError:
        raise ValueError(f"rule must be an ArgNanRule or one of 'torch', 'xla', 'ignore_nan'; got {rule!r}") from None


def _arg_modifier(is_max, value_type, index_type, rule):
    rule = _rule(rule)
    kind = "argmax" if is_max else "argmin"
    # The largest index loses every tie, so the identity is two-sided for any index values. Under IGNORE_NAN a NaN
    # loses to every number, so the infinity identity would win against a NaN element; (NaN, largest index) does not.
    special = _QUIET_NAN if rule is ArgNanRule.IGNORE_NAN else (_NEGATIVE_INFINITY if is_max else _POSITIVE_INFINITY)
    value_dtype = value_type if value_type in _FLOAT_SPECIAL_BITS else data_type.FLOAT
    # Any index type other than INT64 gets an INT32 identity, so the graph refuses any other index stream type.
    index_dtype = data_type.INT64 if index_type == data_type.INT64 else data_type.INT32
    largest_index = (1 << 63) - 1 if index_dtype == data_type.INT64 else (1 << 31) - 1
    return {
        "reducer": _arg_reducer(is_max, rule),
        "reducer_semantic_key": f"cudnn_frontend.reduction_modifiers.{kind}.{rule.value}.v1",
        "identity_values": [
            custom_reduction_identity.from_bits(value_dtype, _FLOAT_SPECIAL_BITS[value_dtype][special]),
            custom_reduction_identity(index_dtype, largest_index),
        ],
    }


def argmax(value_type, index_type=data_type.INT32, rule=ArgNanRule.TORCH):
    """Reducer arguments for the argmax of a (value, index) pair of streams.

    ``value_type`` and ``index_type`` are the data types of the two streams. ``rule`` is an
    ``ArgNanRule`` or its value (``"torch"``, ``"xla"`` or ``"ignore_nan"``). The identities are
    (-inf, largest index), or (NaN, largest index) for ``IGNORE_NAN``.
    """
    return _arg_modifier(True, value_type, index_type, rule)


def argmin(value_type, index_type=data_type.INT32, rule=ArgNanRule.TORCH):
    """Reducer arguments for the argmin of a (value, index) pair of streams; see ``argmax``.

    The identities are (+inf, largest index), or (NaN, largest index) for ``IGNORE_NAN``.
    """
    return _arg_modifier(False, value_type, index_type, rule)
