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


@cute.kernel
def reset_moe_sched_counter(workspace: cute.Tensor, counter_qword: cutlass.Int32):
    """One stream-ordered compute launch initializes the dynamic scheduler.

    The counter occupies the first four bytes of its reserved 128-byte slot.
    Keeping this launch in the compiled host function makes it part of the
    plan and its persistent artifact, with no execute-time compilation or
    framework dependency. A compute launch avoids the DMA-memset transition
    between the MoE compute stages.
    """
    counter = cute.make_tensor(
        cute.recast_ptr(workspace.iterator + counter_qword, dtype=cutlass.Int32),
        cute.make_layout(1),
    )
    counter[0] = cutlass.Int32(0)


reset_moe_sched_counter.set_name_prefix("cudnn", remove_cutlass_symbol=True)
