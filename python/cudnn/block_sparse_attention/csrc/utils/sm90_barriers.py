# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unaligned named barriers for warp-specialized Hopper kernels.

Legacy bar.sync/bar.arrive imply .aligned. Different warpgroups reaching
separate instructions with the same barrier ID need barrier.sync/arrive.
"""

import cutlass.cute as cute
from cutlass import Int32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def named_barrier_sync(*, barrier_id=None, number_of_threads=None, loc=None, ip=None):
    if barrier_id is None:
        cute.arch.barrier(loc=loc, ip=ip)
    else:
        llvm.inline_asm(
            None,
            [Int32(barrier_id).ir_value(loc=loc, ip=ip), Int32(number_of_threads).ir_value(loc=loc, ip=ip)],
            "barrier.sync $0, $1;",
            "r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def named_barrier_arrive(*, barrier_id, number_of_threads, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(barrier_id).ir_value(loc=loc, ip=ip), Int32(number_of_threads).ir_value(loc=loc, ip=ip)],
        "barrier.arrive $0, $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
