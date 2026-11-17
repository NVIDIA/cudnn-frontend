# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure operand admission rules shared by graph lowering and kernel selection.

These checks inspect declared dtype/layout metadata only; they do not select
an architecture, import a compiler, or prepare an execution plan.
"""


def per_k_scale_reject(*, op, operand_dtype, load_dtype, major, scale_stride, scale_dtype, compute_dtype, out_dtype) -> str | None:
    """Shared eligibility for the legacy unit-stride per-K loading ABI.

    Graph preparation uses this before choosing the optimization; the registry
    also checks it for callers supplying FusionChain directly.
    """
    checks = (
        (op != "mul", "only mul is implemented"),
        (operand_dtype not in ("bf16", "fp16"), "operand must be BF16 or FP16"),
        (load_dtype is not None, "operand load casts are not implemented"),
        (major != "k", "operand must be K-major"),
        (scale_stride != 1, "scale K axis must have unit stride"),
        (scale_dtype not in ("bf16", "fp16", "fp32"), "scale must be BF16, FP16 or FP32"),
        (compute_dtype != "fp32", "multiply must compute in FP32"),
        (out_dtype != operand_dtype, "multiply result must match the MMA operand dtype"),
    )
    return next((reason for rejected, reason in checks if rejected), None)


def linear_token_sf_reject(dim, stride, rows, scale_k) -> str | None:
    """GATHER SF layout support, shared by graph checks and runtime binding."""
    if dim is None or stride is None or tuple(dim) != (1, rows, scale_k):
        return f"MoE GATHER token SF must have shape [1, {rows}, {scale_k}]"
    if len(stride) != 3 or stride[2] != 1 or stride[1] < ((scale_k + 15) // 16) * 16 or stride[1] % 16:
        return "MoE GATHER token SF needs contiguous K scales and a row stride padded to a multiple of 16 bytes"
    return None
