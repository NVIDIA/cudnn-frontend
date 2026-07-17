# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Signature-compat wrappers for nvvm dialect builders that changed across
nvidia-cutlass-dsl releases."""

import inspect

from cutlass._mlir.dialects import nvvm

# nvidia-cutlass-dsl <= 4.5.x generates atomicrmw(res, op, ptr, a, ...) with an
# explicit result type; 4.6.0+ infers the result type and dropped the parameter.
_ATOMICRMW_TAKES_RES = "res" in inspect.signature(nvvm.atomicrmw).parameters


def atomicrmw(res, op, ptr, a, *, loc=None, ip=None):
    """nvvm.atomicrmw that works on both cutlass-dsl 4.5.x and 4.6.0+."""
    if _ATOMICRMW_TAKES_RES:
        return nvvm.atomicrmw(res=res, op=op, ptr=ptr, a=a, loc=loc, ip=ip)
    return nvvm.atomicrmw(op=op, ptr=ptr, a=a, loc=loc, ip=ip)
