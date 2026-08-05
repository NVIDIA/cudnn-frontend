# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from .constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL  # noqa: F401

from typing import NamedTuple

from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute

from .barrier import PipelineState, advance, wait, arrive_expect_tx


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
