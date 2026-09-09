# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal tcgen05 inter-thread synchronization intrinsics.

Keep the two PTX instructions local to sparse-attention forward and emit them
through the MLIR LLVM inline-assembly bridge shared by CUTLASS DSL 4.5 and 4.6.
"""

import cutlass.cute as cute
from cutlass._mlir.dialects import llvm


@cute.jit
def tcgen05_fence_before_thread_sync() -> None:
    """Order prior async tcgen05 work before a following thread sync."""

    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::before_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def tcgen05_fence_after_thread_sync() -> None:
    """Order following async tcgen05 work after a preceding thread sync."""

    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


__all__ = ["tcgen05_fence_after_thread_sync", "tcgen05_fence_before_thread_sync"]
