# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Device helpers shared by the megakernels: mbarrier waits, the named-barrier sync of a warp-role subset,
peer-address lookup, raw global loads / stores, readiness-counter waits and arrivals, SwiGLU tile TMA copies and
the task swizzle.  Mostly used by the backward; the mbarrier waits and the named-barrier sync also serve the
forward kernels."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import cpasync

from .persistent_bf16_bwd_contract import MLP_SUPERGROUP_SIZE, SWIGLU_TILE_M, SWIGLU_TILE_N


@dsl_user_op
def _mbar_spin_wait(mbar_smem, phase: Int32, *, loc=None, ip=None) -> None:
    """Tight mbarrier.try_wait.parity spin without a suspend-time hint."""
    llvm.inline_asm(
        None,
        [mbar_smem.toint(loc=loc, ip=ip).ir_value(), phase.ir_value()],
        "{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;\n" "@P1 bra.uni DONE;\nbra.uni LAB_WAIT;\nDONE:\n}",
        "r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_backoff_wait(mbar_smem, phase: Int32, *, loc=None, ip=None) -> None:
    """mbarrier.try_wait.parity loop with a short nanosleep between failed polls.  Same completion as the
    tight spin (``_mbar_spin_wait``); only the polling rate of the many idle drain warps is lower."""
    llvm.inline_asm(
        None,
        [mbar_smem.toint(loc=loc, ip=ip).ir_value(), phase.ir_value()],
        "{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;\n"
        "@P1 bra.uni DONE;\nnanosleep.u32 32;\nbra.uni LAB_WAIT;\nDONE:\n}",
        "r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def named_barrier_sync_unaligned(barrier, *, loc=None, ip=None) -> None:
    """``barrier.cta.sync id, n`` without ``.aligned``: arrive at the named barrier ``barrier`` (a
    ``pipeline.NamedBarrier``) and wait until its ``num_threads`` participants have arrived.

    For a named barrier that only some warps of the CTA reach: the epilogue warps, the MMA and epilogue warps
    of the TMEM hand-over.  CuTe DSL 4.8 lowers ``NamedBarrier.arrive_and_wait`` (and ``wait_unaligned``, and
    ``TmemAllocator.wait_for_alloc``, which calls it) to ``bar.sync``, the aligned form: it promises that every
    thread of the CTA executes that same instruction, which a warp-role branch does not keep.  This is the same
    barrier with the same participant count, without that promise; CuTe DSL 4.8 has no public wrapper for it,
    so the NVVM operation is built here.  Its non-blocking counterpart is ``NamedBarrier.arrive_unaligned``.
    """
    nvvm.barrier_cta_sync(
        Int32(barrier.barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=Int32(barrier.num_threads).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _peer_address(peer_table: cute.Tensor, peer_rank: Int32) -> Int64:
    """Peer base address by runtime rank, read from the ep_size-long table."""
    return Int64(peer_table[peer_rank])


@dsl_user_op
def _store_global_zero16(address: Int64, *, loc=None, ip=None) -> None:
    """One 16 B zero store (``st.global.v4.b32``) to a runtime 64-bit address."""
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), Int32(0).ir_value(loc=loc, ip=ip)],
        "st.global.v4.b32 [$0], {$1, $1, $1, $1};",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_relaxed_sys_b32(address: Int64, *, loc=None, ip=None) -> Uint32:
    """``ld.relaxed.sys.global.b32`` from a runtime 64-bit address."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.sys.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_global_b32(address: Int64, bits: Uint32, *, loc=None, ip=None) -> None:
    """``st.global.b32`` to a runtime 64-bit address."""
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), Uint32(bits).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _load_peer_float(peer_table: cute.Tensor, peer_rank: Int32, route_index: Int32) -> Float32:
    """Load one flattened Float32 route score from peer ``peer_rank`` (table indexed by runtime rank)."""
    address = _peer_address(peer_table, peer_rank) + Int64(route_index) * Int64(4)
    return _load_relaxed_sys_b32(address).bitcast(Float32)


@cute.jit
def _store_peer_float(
    peer_table: cute.Tensor,
    peer_rank: Int32,
    route_index: Int32,
    value: Float32,
) -> None:
    ep_size = Int32(peer_table.shape[0])
    if peer_rank >= Int32(0) and peer_rank < ep_size:
        address = _peer_address(peer_table, peer_rank) + Int64(route_index) * Int64(4)
        _store_global_b32(address, value.bitcast(Uint32))


@dsl_user_op
def _fence_acquire_gpu(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="fence.acquire.gpu;",
        constraints="~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _nanosleep_compat(sleep_time: int, *, loc=None, ip=None) -> None:
    if cutlass.const_expr(hasattr(cute.arch, "nanosleep")):
        cute.arch.nanosleep(sleep_time=sleep_time, loc=loc, ip=ip)
    else:
        llvm.inline_asm(
            res=None,
            operands_=[Int32(sleep_time).ir_value(loc=loc, ip=ip)],
            asm_string="nanosleep.u32 $0;",
            constraints="r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )


@cute.jit
def _counter_wait_gpu(counter: cute.Tensor, index: Int32, required: Int32) -> None:
    # Relaxed poll with nanosleep(16), then fence.acquire.gpu.
    # A bare acquire spin from many waiting CTAs saturates L2 and slows every SM.
    observed = cute.arch.load(counter.iterator + index, Int32, sem="relaxed", scope="gpu")
    while observed < required:
        _nanosleep_compat(16)
        observed = cute.arch.load(counter.iterator + index, Int32, sem="relaxed", scope="gpu")
    _fence_acquire_gpu()


@cute.jit
def _counter_arrive_gpu(
    counter: cute.Tensor,
    index: Int32,
    increment: Int32 = Int32(1),
) -> None:
    cute.arch.atomic_add(
        counter.iterator + index,
        increment,
        sem="release",
        scope="gpu",
    )


@cute.jit
def _ceil_div_i32(value: Int32, divisor: Int32) -> Int32:
    return value // divisor + Int32(value % divisor != Int32(0))


@cute.jit
def _issue_swiglu_tma_load(
    atom: cute.CopyAtom,
    tensor: cute.Tensor,
    row_tile: Int32,
    column_tile: Int32,
    smem_tile: cute.Tensor,
    tma_barrier: cute.Pointer,
) -> None:
    """Load one 128x128 BF16 tile into one SwiGLU arena stage."""

    global_tile = cute.local_tile(
        tensor,
        (SWIGLU_TILE_M, SWIGLU_TILE_N),
        (row_tile, column_tile),
    )
    smem_flat = cute.group_modes(smem_tile, 0, cute.rank(smem_tile))
    global_flat = cute.group_modes(global_tile, 0, cute.rank(global_tile))
    smem_part, global_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        smem_flat,
        global_flat,
    )
    cute.copy(
        atom,
        global_part,
        smem_part,
        tma_bar_ptr=tma_barrier,
    )


@cute.jit
def _issue_swiglu_tma_store(
    atom: cute.CopyAtom,
    tensor: cute.Tensor,
    row_tile: Int32,
    column_tile: Int32,
    smem_tile: cute.Tensor,
) -> None:
    """Store one completed 128x128 BF16 tile from the reusable arena."""

    global_tile = cute.local_tile(
        tensor,
        (SWIGLU_TILE_M, SWIGLU_TILE_N),
        (row_tile, column_tile),
    )
    smem_flat = cute.group_modes(smem_tile, 0, cute.rank(smem_tile))
    global_flat = cute.group_modes(global_tile, 0, cute.rank(global_tile))
    smem_part, global_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        smem_flat,
        global_flat,
    )
    cute.copy(atom, smem_part, global_part)


@cute.jit
def _swizzled_tile_i32(row_blocks: Int32, column_blocks: Int32, task: Int32) -> tuple[Int32, Int32]:
    """Supergroup-8 serpentine task map."""

    supergroup = Int32(MLP_SUPERGROUP_SIZE)
    supergroup_numel = row_blocks * supergroup
    group = task // supergroup_numel
    full_columns = (column_blocks // supergroup) * supergroup
    full_numel = row_blocks * full_columns
    tail_columns = cutlass.max(Int32(1), column_blocks - full_columns)
    row = Int32(0)
    column = Int32(0)
    if task < full_numel:
        row = (task % supergroup_numel) // supergroup
        column = group * supergroup + task % supergroup
    else:
        tail = task - full_numel
        row = tail // tail_columns
        column = full_columns + tail % tail_columns
    if group % Int32(2) != Int32(0):
        row = row_blocks - row - Int32(1)
    return row, column


__all__ = [
    "_mbar_backoff_wait",
    "_mbar_spin_wait",
    "_peer_address",
    "_load_peer_float",
    "_store_peer_float",
    "_fence_acquire_gpu",
    "_nanosleep_compat",
    "_counter_wait_gpu",
    "_counter_arrive_gpu",
    "_ceil_div_i32",
    "_issue_swiglu_tma_load",
    "_issue_swiglu_tma_store",
    "_swizzled_tile_i32",
]
