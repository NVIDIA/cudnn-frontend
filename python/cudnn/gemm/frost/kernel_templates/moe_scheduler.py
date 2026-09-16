# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-ring helpers shared by the SM100 and SM120 MoE templates."""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm


@cute.jit
def moe_load_sched_word(ptr):
    """Read a scheduler word before the consumer releases its ring slot.

    Call with a converged full warp. Load in lane zero, then broadcast the
    register value so no other lane can issue a shared load after release.
    Consumers must still converge before their release arrival.
    """
    value = cutlass.Int32(0)
    if cute.arch.lane_idx() == 0:
        value = ptr.load()
    return nvvm.shfl_sync(0xFFFFFFFF, value, 0, 0x1F, nvvm.Shfl.IDX)
