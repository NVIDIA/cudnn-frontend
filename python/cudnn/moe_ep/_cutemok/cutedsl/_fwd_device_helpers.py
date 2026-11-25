# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Device helpers shared by the forward megakernels: peer address lookup, readiness-counter arrive
and wait, a counting block barrier, the task swizzle and an overflow-safe ceiling division."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .persistent_bf16_fwd_contract import MLP_SUPERGROUP_SIZE


@cute.jit
def _peer_address(peer_table: cute.Tensor, peer_rank: Int32) -> Int64:
    """Peer base address by runtime rank, from the ep_size-long address table."""
    return Int64(peer_table[peer_rank])


@dsl_user_op
def _sync_threads_count(predicate: Int32, *, loc=None, ip=None) -> Int32:
    """``bar.red.popc.u32``: block-wide barrier that also returns the number of
    threads with a non-zero ``predicate`` (CUDA ``__syncthreads_count``)."""
    result = llvm.inline_asm(
        T.i32(),
        [Int32(predicate).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nbar.red.popc.u32 $0, 0, p;\n}",
        "=r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@cute.jit
def _counter_arrive_gpu(
    counter: cute.Tensor,
    index: Int32,
    increment: Int32 = Int32(1),
) -> None:
    """Add ``increment`` arrivals to the monotonic counter ``counter[index]`` (release, GPU scope)."""

    cute.arch.atomic_add(
        counter.iterator + index,
        increment,
        sem="release",
        scope="gpu",
    )


@cute.jit
def _swizzled_mlp_tile_coord_i32(
    row_blocks: Int32,
    col_blocks: Int32,
    task_idx: Int32,
) -> tuple[Int32, Int32]:
    """Task index -> (row block, column block) of a ``row_blocks`` x ``col_blocks`` tile grid: row-major
    inside supergroups of MLP_SUPERGROUP_SIZE columns, the row order reversed in every other supergroup
    (ThunderKittens' swizzle); columns past the last full supergroup form one row-major section."""

    supergroup_size = Int32(MLP_SUPERGROUP_SIZE)
    supergroup_numel = row_blocks * supergroup_size
    supergroup_idx = task_idx // supergroup_numel
    supersection_cols = (col_blocks // supergroup_size) * supergroup_size
    supersection_numel = row_blocks * supersection_cols
    finalsection_cols = col_blocks - supersection_cols
    # With a column count that is a multiple of the supergroup size the
    # remainder branch is unreachable for valid tasks, but CuTe still lowers
    # both dynamic regions; keep the dead-region divisor nonzero.
    safe_finalsection_cols = cutlass.max(Int32(1), finalsection_cols)
    row_idx = Int32(0)
    col_idx = Int32(0)
    if task_idx < supersection_numel:
        row_idx = (task_idx % supergroup_numel) // supergroup_size
        col_idx = supergroup_idx * supergroup_size + task_idx % supergroup_size
    else:
        remainder_task_id = task_idx - supersection_numel
        row_idx = remainder_task_id // safe_finalsection_cols
        col_idx = supersection_cols + remainder_task_id % safe_finalsection_cols
    if supergroup_idx % Int32(2) != Int32(0):
        row_idx = row_blocks - row_idx - Int32(1)
    return row_idx, col_idx


__all__ = [
    "_peer_address",
    "_sync_threads_count",
    "_counter_arrive_gpu",
    "_swizzled_mlp_tile_coord_i32",
]


@cute.jit
def _counter_wait_gpu(
    counter: cute.Tensor,
    index: Int32,
    required: Int32,
) -> None:
    """Spin until the monotonic counter ``counter[index]`` reaches ``required`` (acquire loads, GPU scope)."""

    pointer = counter.iterator + index
    observed = Int32(0)
    while observed < required:
        observed = cute.arch.load(
            pointer,
            Int32,
            sem="acquire",
            scope="gpu",
        )


@cute.jit
def _ceil_div_i32(value: Int32, divisor: Int32) -> Int32:
    # Nonnegative runtime counts: avoid overflow in value + divisor - 1.
    return value // divisor + Int32(value % divisor != Int32(0))
