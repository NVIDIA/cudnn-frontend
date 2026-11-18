# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSL/runtime admission checks shared by the SM90 DSA forward adapters."""

from cudnn.frost.buffers import cutedsl_requirement_error


def sm90_dsa_cutedsl_requirement_error():
    """Return an actionable error before importing or compiling the kernel."""
    error = cutedsl_requirement_error("SM90 DSA forward")
    if error is not None:
        return error

    # The compiler flavor is a property of the loaded DSL wheel, not of
    # torch.version.cuda or CUDA_HOME. DSL 4.8.1's CUDA 12.9 compiler accepts
    # this kernel but emits an illegal instruction on SM90, so reject it before
    # JIT/launch instead of allowing a fatal CUDA-context error.
    from cutlass.base_dsl.version_info import CUDA_VERSION

    compiler_version = (int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
    if compiler_version < (13, 0):
        found = ".".join(map(str, compiler_version))
        return (
            "SM90 DSA forward requires the CUDA 13 CuTe DSL compiler; "
            f"found CUDA {found}. Install 'nvidia-cutlass-dsl[cu13]>=4.8.0' "
            "and use a CUDA 13-capable driver."
        )
    return None
