# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Validate a kernel-supplied release-zero mask without requiring the experimental CuTeDSL primitives."""

import cutlass
from cutlass._mlir import ir as _ir

_MASK_OP = "arith.andi"


class ReleaseZero:
    """The kernel's release-zero argument, checked once by :func:`release_zero`.  A trace-time object: it emits no code."""

    __slots__ = ("_word",)

    def __init__(self, word):
        self._word = word

    @property
    def word(self):
        return self._word


def _is_ir_constant(value) -> bool:
    owner = value.owner
    return not isinstance(owner, _ir.Block) and getattr(owner, "name", "") == "arith.constant"


def release_zero(kernel_arg) -> ReleaseZero:
    """Wrap the kernel's release-zero ARGUMENT for the ``consume_then_*`` forms.

    ``kernel_arg`` must be the 32-bit integer parameter of the enclosing ``@cute.kernel`` itself -- the one its host passes as
    the literal 0 -- and this call must be its only use.  A Python number, a compile-time constant, a
    computed value or a value of another type raises ``TypeError``: a mask the compiler can see through removes the ordering
    without any other error.
    """
    if not isinstance(kernel_arg, (cutlass.Int32, cutlass.Uint32)):
        raise TypeError(
            f"release_zero: expected the kernel's Int32 release-zero argument, got {type(kernel_arg).__name__}; a Python number "
            "or another type cannot carry the ordering (the mask must be a kernel argument the host passes as 0)"
        )
    value = kernel_arg.value
    if not isinstance(value, _ir.Value):
        raise TypeError(
            f"release_zero: got the compile-time constant {value!r}; the release-zero must be a KERNEL ARGUMENT -- a zero the "
            "compiler can see is folded away and the loads are unread at the release again"
        )
    owner = value.owner
    if not isinstance(owner, _ir.Block):
        what = "a constant" if _is_ir_constant(value) else f"the result of {getattr(owner, 'name', type(owner).__name__)}"
        raise TypeError(f"release_zero: got {what}; pass the kernel's own release-zero parameter, unmodified")
    parent = owner.owner
    regions = getattr(parent, "regions", None)
    if regions is None or len(regions) == 0 or len(regions[0].blocks) == 0 or regions[0].blocks[0] != owner:
        raise TypeError("release_zero: the value is a region argument (a loop- or branch-carried value), not a parameter of the kernel")
    zero = ReleaseZero(kernel_arg)
    _check_zero(zero, "release_zero")
    return zero


def _check_zero(zero, who: str):
    if not isinstance(zero, ReleaseZero):
        raise TypeError(f"{who}: `zero` must be the ReleaseZero returned by release_zero(<the kernel's release-zero argument>), got {type(zero).__name__}")
    for use in zero.word.value.uses:
        name = use.owner.name
        if name != _MASK_OP:
            raise TypeError(
                f"{who}: the kernel's release-zero argument has a use that is not this module's mask ({name}).  Its only use may "
                "be release_zero(arg): a compare, branch, assert or any other expression on it lets the compiler replace the mask "
                "by a constant, and the loads are unread at the release again without any error"
            )
