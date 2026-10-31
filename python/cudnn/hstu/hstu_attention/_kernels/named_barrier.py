# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named-barrier IDs used by the HSTU forward pipeline."""

from cutlass import Int32
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op

# ID 0 remains available for block synchronization. Each SiLU warp group owns
# the epilogue barrier at this base plus its group index, reserving IDs 1 and 2.
# A participant count does not identify the participants, so two groups that
# can be in flight together must never share an ID.
EPILOGUE_BARRIER_BASE = 1
TMEM_POINTER_BARRIER = 3
TMEM_RELEASE_BARRIER = 4


@dsl_user_op
def sync_unaligned(
    barrier_id: int | Int32,
    num_threads: int | Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Wait on a CTA named barrier from divergent warp-role control flow."""

    nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=Int32(num_threads).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )
