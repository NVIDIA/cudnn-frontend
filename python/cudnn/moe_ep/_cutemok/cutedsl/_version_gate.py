# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CuTe DSL the kernels need, checked once, on first use, before any JIT (``python/cudnn/AGENTS.md`` Rule 7).

Every kernel module calls ``require_cutedsl()`` before it imports CUTLASS, and the hosts call it before they compile,
so an unsupported install is declined with a message that names the installed and the required versions instead of
failing inside a trace.  The version floor goes through cuDNN frontend's gate
(``cudnn.frost.buffers.cutedsl_requirement_error``, with this operator's floor), which reads package metadata only;
the compiler flavor check imports CUTLASS (it is about to be imported by the caller anyway).
"""

from __future__ import annotations

from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state

#: The oldest CuTe DSL the kernels are built and tested with; above ``cudnn.frost.buffers.CUTEDSL_MIN_VERSION``.
CUTEDSL_FLOOR = (4, 8, 0)
#: The CUDA major version of the CuTe DSL compiler flavor (the cu13 build).
CUDA_MAJOR = 13

_OPERATION = "CuTe MoK"


class CuteDslRequirementError(RuntimeError):
    """The installed CuTe DSL cannot build the CuTe MoK kernels."""


def flavor_error(cuda_major: int | None) -> str | None:
    """Why the loaded CuTe DSL compiler flavor is declined, or None when it is the cu13 build (or not known)."""
    if cuda_major is None or cuda_major == CUDA_MAJOR:
        return None
    floor = ".".join(map(str, CUTEDSL_FLOOR))
    return (
        f"{_OPERATION} supports only the cu{CUDA_MAJOR} build of the CuTe DSL: its kernels are built and tested with "
        f"the CUDA {CUDA_MAJOR} compiler, and the loaded build is cu{cuda_major}.  "
        f"Install nvidia-cutlass-dsl[cu{CUDA_MAJOR}]>={floor} with a driver for CUDA {CUDA_MAJOR}"
    )


_VERSION: str | None = None


def require_cutedsl() -> str:
    """Raise ``CuteDslRequirementError`` unless a CuTe DSL at or above ``CUTEDSL_FLOOR``, in its cu13 compiler flavor,
    is installed; return its version.  The check runs once per process."""
    global _VERSION
    if _VERSION is None:
        installed, version = cutedsl_state()
        if not installed:
            floor = ".".join(map(str, CUTEDSL_FLOOR))
            raise CuteDslRequirementError(f"{_OPERATION} requires nvidia-cutlass-dsl[cu{CUDA_MAJOR}]>={floor}; it is not installed")
        error = cutedsl_requirement_error(_OPERATION, floor=CUTEDSL_FLOOR)
        if error is None:
            # CuTe DSL >= 4.8 picks its compiler flavor (cu12 / cu13) when CUTLASS is imported.
            try:
                from cutlass.base_dsl.version_info import CUDA_VERSION
            except ImportError:
                CUDA_VERSION = None
            error = flavor_error(getattr(CUDA_VERSION, "major", None))
        if error is not None:
            raise CuteDslRequirementError(error)
        _VERSION = version[1] if version else "unknown"
    return _VERSION


__all__ = [
    "CUDA_MAJOR",
    "CUTEDSL_FLOOR",
    "CuteDslRequirementError",
    "flavor_error",
    "require_cutedsl",
]
