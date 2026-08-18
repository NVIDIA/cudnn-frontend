# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from .constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL  # noqa: F401

from typing import NamedTuple

from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith

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
