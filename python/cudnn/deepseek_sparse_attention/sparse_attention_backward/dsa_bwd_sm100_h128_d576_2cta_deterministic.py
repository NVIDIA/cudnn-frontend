# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace capacity and ordered updates for deterministic D576 two-CTA DSA.

The main kernel owns accumulator views, query slots, and fixed-order folds.
Its callers order duplicate-index writers before the FP32 atomic update.
"""

from typing import Type
import cutlass
import cutlass.cute as cute


@cute.jit
def ordered_dkv_quad(target: cute.Tensor, values: cute.Tensor):
    """Add four FP32 values after the caller orders their writers.

    Warp and CTA barriers order different duplicate-index owners. Same-thread
    updates to each element retain program order under PTX SC-per-location.
    Cluster ranks must own disjoint columns or slots. The updates retain the
    FP32 global-memory atomic operation and its rounding behavior.
    """
    cute.arch.atomic_add(target.iterator.llvm_ptr, values.load())


class TwoCTADeterministicReduction:
    """Compact workspace capacity for the deterministic D576 two-CTA kernel."""

    deterministic = True

    @staticmethod
    def _get_workspace_size_dKV(k: int, d: int, b: int, acc_dtype: Type[cutlass.Numeric]):
        return (b, 96, (k + 7) // 8 * 8, (d + 7) // 8 * 8 * (acc_dtype.width // 8))
