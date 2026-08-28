# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from .constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL  # noqa: F401

from typing import NamedTuple

from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith

from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer

from .barrier import PipelineState, advance, wait, arrive_expect_tx


@cute.jit
def lpt_tile_coords(linear, q_h, batch, q_tiles):
    hb = q_h * batch
    row_rank = linear // hb
    within = linear % hb
    row = (q_tiles - cutlass.Int32(1)) - row_rank
    head = within % q_h
    batch_idx = within // q_h
    return row, head, batch_idx


@cute.jit
def lpt_l2_tile_coords(linear, q_h, batch, q_tiles, heads_per_kv, seqlen_kv, kv_bytes_per_row, l2_bytes):
    n_kh = q_h // heads_per_kv
    num_groups = n_kh * batch
    # Guard per_group == 0: a zero KV length would divide by zero below.
    per_group_raw = seqlen_kv * cutlass.Int32(kv_bytes_per_row)
    per_group = cutlass.Int32(
        arith.select(
            (per_group_raw < cutlass.Int32(1)).ir_value(),
            cutlass.Int32(1).ir_value(),
            per_group_raw.ir_value(),
        )
    )
    ag_raw = cutlass.Int32(l2_bytes) // per_group
    ag_min1 = cutlass.Int32(
        arith.select(
            (ag_raw < cutlass.Int32(1)).ir_value(),
            cutlass.Int32(1).ir_value(),
            ag_raw.ir_value(),
        )
    )
    active_groups = cutlass.Int32(
        arith.select(
            (ag_min1 > num_groups).ir_value(),
            num_groups.ir_value(),
            ag_min1.ir_value(),
        )
    )

    tiles_per_grp = q_tiles * heads_per_kv
    tiles_per_blk = active_groups * tiles_per_grp
    num_blocks = (num_groups + active_groups - cutlass.Int32(1)) // active_groups
    block_idx = linear // tiles_per_blk
    within_blk = linear % tiles_per_blk
    # The last block may hold fewer than active_groups groups — clamp so the
    # decoded group never lands outside [0, num_groups).
    is_last_block = (block_idx + cutlass.Int32(1)) == num_blocks
    agroup_eff = cutlass.Int32(
        arith.select(
            is_last_block.ir_value(),
            (num_groups - block_idx * active_groups).ir_value(),
            active_groups.ir_value(),
        )
    )

    row_rank = within_blk // (agroup_eff * heads_per_kv)
    in_rank = within_blk % (agroup_eff * heads_per_kv)
    # in_rank lays out as (sub_head, kv_group): every Q-head sharing a kv_head
    # lands in the same block, so they hit the same resident K/V.
    sub_head = in_rank // agroup_eff
    kv_group = (in_rank % agroup_eff) + block_idx * active_groups
    kv_head = kv_group % n_kh
    batch_idx = kv_group // n_kh
    head = kv_head * heads_per_kv + sub_head
    row = (q_tiles - cutlass.Int32(1)) - row_rank
    return row, head, batch_idx


@cute.jit
def read_tile_id_arrive(mb, cga_size: int):
    if cga_size == 1:
        if nvvm.elect_sync():
            nvvm.mbarrier_arrive(mb)
    else:
        lane_stride = 32 // cga_size
        lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
        for i in cutlass.range_constexpr(cga_size):
            target_lane = i * lane_stride
            if lane == cutlass.Int32(target_lane):
                peer_mb = nvvm.mapa(mb, cutlass.Int32(i))
                nvvm.mbarrier_arrive(peer_mb, scope=nvvm.MemScope.CLUSTER, relaxed=True)


class Sched(NamedTuple):
    mb_scheduler: object
    mb_read_tile_id: object
    tile_id_smem: object
    bidx_init: object
    bidy_init: object
    bidz_init: object


@cute.jit
def scheduler_warp_loop_persistent(
    sched,
    sched_stages: int,
    is_cga_first_cta,
    meta_t,
    ctr_off,
    live_off,
    cga_size: int,
    cga_m: int,
):
    """Persistent tile scheduler over a LIVE-ONLY unit range (THD).

    CLC sizes the grid to the work list, which for THD means the plan-time
    envelope and therefore dead clusters. Here the grid is occupancy-sized and
    the bound is a DEVICE value (``meta[live_off]``, written by the setup
    launch), so no unit past the live total is ever handed out.

    The cluster lead claims one unit with a global atomic and pushes the
    payload into every CTA's ``tile_id_smem`` over DSMEM, then arrives each
    peer's scheduler mbarrier -- the same shape as ``read_tile_id_arrive``.
    Multicast is not available here: it is a clusterlaunchcontrol facility, so
    dynamic claiming needs an explicit peer write.

    ``cga_size`` is the DSMEM broadcast fan-out (CTAs per cluster); ``cga_m`` is
    the stride the consumer's decode divides back out. They are equal while
    CGA_N == 1, which is every config today -- taking both keeps the handout
    correct if that ever stops holding.
    """
    meta = cutlass.make_array_view(meta_t)
    ctr_ptr = Pointer(meta_t.iterator.raw_ptr(), dtype=cutlass.Int32) + ctr_off
    state = PipelineState.start()
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)

        # Every CTA expects the 16-byte payload on its own mbarrier, exactly as
        # the CLC path did; the lead's remote store delivers it and completes
        # the barrier through the transaction count.
        if nvvm.elect_sync():
            arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)

        if nvvm.elect_sync() and is_cga_first_cta:
            uid = cutlass.Int32(nvvm.atomicrmw(nvvm.AtomicOp.ADD, ctr_ptr, cutlass.Int32(1)))
            live = cutlass.Int32(meta[live_off])
            valid = cutlass.Int32(arith.select((uid < live).ir_value(), cutlass.Int32(1).ir_value(), cutlass.Int32(0).ir_value()))
            # Stride by CGA_M, not the cluster size: the consumer decodes the
            # unit id back out as linear // CGA_M.
            linear = uid * cutlass.Int32(cga_m)
            # store_async_dsmem wants CuTe pointers; the smem arrays hand out
            # base-DSL ones, so convert through the raw addresses.
            _tile_ptr = cute.make_ptr(
                cutlass.Int32,
                sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)).data_ptr().toint(cutlass.Int32),
                cutlass.AddressSpace.smem,
                assumed_align=16,
            )
            _mbar_ptr = cute.make_ptr(
                cutlass.Int64,
                sched.mb_scheduler.subview(state.idx).data_ptr().toint(cutlass.Int32),
                cutlass.AddressSpace.smem,
                assumed_align=8,
            )
            # One word at a time rather than a v4 payload: store_async_dsmem
            # accepts a 2/4-tuple per its contract, but lowers whatever it was
            # handed through Int32(value), so a tuple raises at trace time. Four
            # scalar stores carry the same 16 bytes and so satisfy the same
            # transaction count the arrive above expects.
            _payload = (linear, cutlass.Int32(0), valid, cutlass.Int32(0))
            for i in cutlass.range_constexpr(cga_size):
                for w in cutlass.range_constexpr(4):
                    cute.arch.store_async_dsmem(_tile_ptr + w, _payload[w], _mbar_ptr, i)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        validity = (sched.tile_id_smem.subview(state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid = validity & cutlass.Int32(1)

        state = advance(state, sched_stages)


@cute.jit
def scheduler_warp_loop(sched, sched_stages: int, is_cga_first_cta):
    state = PipelineState.start()
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)

        if nvvm.elect_sync():
            arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)

        if nvvm.elect_sync() and is_cga_first_cta:
            nvvm.clusterlaunchcontrol_try_cancel(
                sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)),
                sched.mb_scheduler.subview(state.idx),
                multicast=1,
            )
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        validity = (sched.tile_id_smem.subview(state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid = validity & cutlass.Int32(1)

        state = advance(state, sched_stages)
