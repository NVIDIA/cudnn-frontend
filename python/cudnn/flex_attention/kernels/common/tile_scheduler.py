# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

try:
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override

import cutlass
from cutlass.pipeline import PipelineClcFetchAsync, PipelineState
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.utils import ClcDynamicPersistentTileScheduler, ClcDynamicPersistentTileSchedulerParams
from cutlass.cute.typing import Boolean
from cutlass.cutlass_dsl import (
    extract_mlir_values,
    new_from_mlir_values,
)

from cudnn.flex_attention._compat.cute_dsl_utils import ParamsBase

import cudnn.flex_attention.kernels.common.device_utils as utils
from cudnn.flex_attention.kernels.common.fast_math import clz
from cudnn.flex_attention.kernels.sm90.fwd.named_barrier import NamedBarrierFwd


@dataclass
class ClcState(ParamsBase):
    """Owns the runtime state shared by CLC-capable tile schedulers.

    `FlexAttentionForwardSm100` constructs this state because it owns the CLC
    response buffer, mbarrier storage, and launch geometry needed to initialize
    the hardware scheduler and async pipeline. Individual tile schedulers then
    consume this state and map the returned hardware work tiles into their own
    logical `WorkTileInfo` coordinates.

    To add CLC support to a scheduler:
    - implement `clc_problem_shape(params)` so the kernel can create the hardware scheduler
    - accept `clc: ClcState | None` in `create(...)` / `__init__`
    - map `clc.initial_work_tile_info()` and `clc.get_current_work()` into scheduler coordinates
    """

    _hw_scheduler: ClcDynamicPersistentTileScheduler
    _pipeline: PipelineClcFetchAsync
    _consumer_state: PipelineState
    _producer_state: PipelineState

    @staticmethod
    def create(
        *,
        hw_scheduler: ClcDynamicPersistentTileScheduler,
        pipeline: PipelineClcFetchAsync,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "ClcState":
        return ClcState(hw_scheduler, pipeline, consumer_state, producer_state)

    def initial_work_tile_info(self):
        return self._hw_scheduler.initial_work_tile_info()

    def get_current_work(self):
        return self._hw_scheduler.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)
        mbarrier_addr = self._pipeline.producer_get_barrier(self._producer_state, loc=loc, ip=ip)
        self._hw_scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)
        self._producer_state.advance(loc=loc, ip=ip)

    def consumer_wait(self, *, loc=None, ip=None):
        self._pipeline.consumer_wait(self._consumer_state, loc=loc, ip=ip)

    def consumer_release(self, *, loc=None, ip=None):
        self._pipeline.consumer_release(self._consumer_state, loc=loc, ip=ip)
        self._consumer_state.advance(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        self._pipeline.producer_tail(self._producer_state, loc=loc, ip=ip)


class WorkTileInfo(cutlass.utils.WorkTileInfo):
    """Altered WorkTileInfo which includes four axes: (block, head, batch, split)"""

    @override
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 5
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)


class PlanClcPersistentTileSchedulerSm100:
    """Map a one-dimensional hardware CLC queue to plan-owned work descriptors."""

    @dataclass
    class Params(ParamsBase):
        mWorkDesc: cute.Tensor
        num_tasks: Int32
        cta_group_size: cutlass.Constexpr[int]

    def __init__(
        self,
        params: Params,
        clc: ClcState,
        *,
        loc=None,
        ip=None,
    ) -> None:
        self.params = params
        self.clc = clc
        self._loc = loc
        self._ip = ip

    @staticmethod
    @cute.jit
    def to_underlying_arguments(
        mWorkDesc: cute.Tensor,
        cta_group_size: cutlass.Constexpr[int],
        *,
        loc=None,
        ip=None,
    ) -> "PlanClcPersistentTileSchedulerSm100.Params":
        return PlanClcPersistentTileSchedulerSm100.Params(
            mWorkDesc,
            Int32(mWorkDesc.shape[0]),
            cta_group_size,
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=(
                params.num_tasks * Int32(params.cta_group_size),
                Int32(1),
                Int32(1),
            ),
            cluster_shape_mnk=(params.cta_group_size, 1, 1),
        )

    @staticmethod
    @cute.jit
    def get_grid_shape(params: Params):
        return (
            params.num_tasks * Int32(params.cta_group_size),
            Int32(1),
            Int32(1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        clc: ClcState,
        *,
        loc=None,
        ip=None,
    ) -> "PlanClcPersistentTileSchedulerSm100":
        return PlanClcPersistentTileSchedulerSm100(
            params,
            clc,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def _work_to_desc(self, work) -> WorkTileInfo:
        task_idx = work.tile_idx[0] // Int32(self.params.cta_group_size)
        is_valid = work.is_valid_tile & (task_idx < self.params.num_tasks)
        safe_task_idx = cutlass.min(task_idx, self.params.num_tasks - Int32(1))
        m_block = Int32(0)
        head_idx = Int32(0)
        batch_idx = Int32(0)
        q_valid_rows = Int32(0)
        if is_valid:
            m_block = self.params.mWorkDesc[safe_task_idx, Int32(0)]
            head_idx = self.params.mWorkDesc[safe_task_idx, Int32(1)]
            batch_idx = self.params.mWorkDesc[safe_task_idx, Int32(2)]
            q_valid_rows = self.params.mWorkDesc[safe_task_idx, Int32(3)]
        return WorkTileInfo(
            (m_block, head_idx, batch_idx, q_valid_rows),
            is_valid,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self._work_to_desc(self.clc.get_current_work())

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self._work_to_desc(self.clc.initial_work_tile_info())

    def prefetch_next_work(self, *, loc=None, ip=None) -> None:
        self.clc.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        self.clc.consumer_wait(loc=loc, ip=ip)
        work = self.get_current_work(loc=loc, ip=ip)
        self.clc.consumer_release(loc=loc, ip=ip)
        return work

    def producer_tail(self, *, loc=None, ip=None) -> None:
        self.clc.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values.extend(cutlass.extract_mlir_values(self.clc))
        return values

    def __new_from_mlir_values__(self, values):
        params_values = cutlass.extract_mlir_values(self.params)
        params = cutlass.new_from_mlir_values(
            self.params,
            values[: len(params_values)],
        )
        clc = cutlass.new_from_mlir_values(self.clc, values[len(params_values) :])
        return PlanClcPersistentTileSchedulerSm100(
            params,
            clc,
            loc=self._loc,
            ip=self._ip,
        )


class PlanDynamicPersistentTileSchedulerSm90:
    """Consume plan-owned FWD descriptors through a software atomic queue."""

    @dataclass
    class Params(ParamsBase):
        mWorkDesc: cute.Tensor
        mTileCounter: cute.Tensor
        num_tasks: Int32
        num_sm: Int32
        num_sync_threads: cutlass.Constexpr[int]

    def __init__(
        self,
        params: Params,
        sWork: cute.Tensor,
        is_producer: cutlass.Constexpr[bool],
        next_tile_idx: Int32,
        block: Int32,
        head: Int32,
        batch: Int32,
        valid: Boolean,
        *,
        loc=None,
        ip=None,
    ) -> None:
        self.params = params
        self.sWork = sWork
        self.is_producer = is_producer
        self._next_tile_idx = next_tile_idx
        self._block = block
        self._head = head
        self._batch = batch
        self._valid = valid
        self._loc = loc
        self._ip = ip

    @staticmethod
    @cute.jit
    def to_underlying_arguments(
        mWorkDesc: cute.Tensor,
        mTileCounter: cute.Tensor,
        *,
        num_sm: Int32,
        num_mma_threads: int,
        loc=None,
        ip=None,
    ) -> "PlanDynamicPersistentTileSchedulerSm90.Params":
        return PlanDynamicPersistentTileSchedulerSm90.Params(
            mWorkDesc=mWorkDesc,
            mTileCounter=mTileCounter,
            num_tasks=Int32(mWorkDesc.shape[0]),
            num_sm=num_sm,
            num_sync_threads=num_mma_threads + cute.arch.WARP_SIZE,
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sWork: cute.Tensor,
        is_producer: cutlass.Constexpr[bool],
        *,
        loc=None,
        ip=None,
    ) -> "PlanDynamicPersistentTileSchedulerSm90":
        return PlanDynamicPersistentTileSchedulerSm90(
            params,
            sWork,
            is_producer,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Boolean(False),
            loc=loc,
            ip=ip,
        )

    @staticmethod
    @cute.jit
    def get_grid_shape(params: Params) -> Tuple[Int32, Int32, Int32]:
        return (cutlass.min(params.num_sm, params.num_tasks), Int32(1), Int32(1))

    @cute.jit
    def _fetch_next_tile(self) -> None:
        next_tile_idx = Int32(0)
        if cute.arch.lane_idx() == Int32(0):
            next_tile_idx = Int32(
                cute.arch.atomic_add(
                    ptr=self.params.mTileCounter.iterator,
                    val=Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
            )
        self._next_tile_idx = next_tile_idx

    @cute.jit
    def _map_tile(self, tile_idx: Int32) -> WorkTileInfo:
        tile_idx = cute.arch.shuffle_sync(tile_idx, Int32(0))
        valid = tile_idx < self.params.num_tasks
        block = Int32(0)
        head = Int32(0)
        batch = Int32(0)
        if cute.arch.lane_idx() == Int32(0) and valid:
            block = self.params.mWorkDesc[tile_idx, Int32(0)]
            head = self.params.mWorkDesc[tile_idx, Int32(1)]
            batch = self.params.mWorkDesc[tile_idx, Int32(2)]
        self._block = cute.arch.shuffle_sync(block, Int32(0))
        self._head = cute.arch.shuffle_sync(head, Int32(0))
        self._batch = cute.arch.shuffle_sync(batch, Int32(0))
        self._valid = valid
        return self.get_current_work()

    @cute.jit
    def _publish_current_work(self) -> None:
        if cute.arch.lane_idx() == Int32(0):
            self.sWork[0] = self._block
            self.sWork[1] = self._head
            self.sWork[2] = self._batch
            self.sWork[3] = Int32(1) if self._valid else Int32(0)
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.SchedulerFull),
            number_of_threads=self.params.num_sync_threads,
        )

    @cute.jit
    def _consume_current_work(self) -> WorkTileInfo:
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.SchedulerFull),
            number_of_threads=self.params.num_sync_threads,
        )
        self._block = self.sWork[0]
        self._head = self.sWork[1]
        self._batch = self.sWork[2]
        self._valid = self.sWork[3] != Int32(0)
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.SchedulerEmpty),
            number_of_threads=self.params.num_sync_threads,
        )
        return self.get_current_work()

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return WorkTileInfo((self._block, self._head, self._batch, Int32(0)), self._valid)

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.is_producer):
            self._fetch_next_tile()
            self._map_tile(self._next_tile_idx)
            self._publish_current_work()
            return self.get_current_work()
        return self._consume_current_work()

    @cute.jit
    def prefetch_next_work(self, *, loc=None, ip=None) -> None:
        if const_expr(self.is_producer):
            self._fetch_next_tile()

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.is_producer):
            self._map_tile(self._next_tile_idx)
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.SchedulerEmpty),
                number_of_threads=self.params.num_sync_threads,
            )
            self._publish_current_work()
            return self.get_current_work()
        return self._consume_current_work()

    def producer_tail(self, *, loc=None, ip=None) -> None:
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in (
            self.params,
            self.sWork,
            self._next_tile_idx,
            self._block,
            self._head,
            self._batch,
            self._valid,
        ):
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        objects = (
            self.params,
            self.sWork,
            self._next_tile_idx,
            self._block,
            self._head,
            self._batch,
            self._valid,
        )
        rebuilt = []
        for obj, n_items in zip(objects, self._values_pos):
            rebuilt.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return PlanDynamicPersistentTileSchedulerSm90(
            rebuilt[0],
            rebuilt[1],
            self.is_producer,
            *rebuilt[2:],
            loc=self._loc,
            ip=self._ip,
        )


@runtime_checkable
class TileSchedulerProtocol(Protocol):
    """Protocol defining the interface all tile schedulers must implement.

    Schedulers are responsible for:
    1. Coordinate mapping: linear tile index -> (m_block, head, batch, split)
    2. Work distribution: how to get the next tile (static grid-stride vs CLC dynamic)
    """

    def get_current_work(self) -> WorkTileInfo:
        """Get the current work tile coordinates."""
        ...

    def initial_work_tile_info(self) -> WorkTileInfo:
        """Get the initial work tile for this CTA."""
        ...

    def advance_to_next_work(self, *, loc=None, ip=None):
        """Consumer-side advance: move to next tile and return it.

        For static schedulers: grid-stride increment + get_current_work.
        For CLC schedulers: consumer wait + get_current_work + consumer release + state advance.
        """
        ...

    def prefetch_next_work(self, *, loc=None, ip=None) -> None:
        """Producer-side prefetch of next work tile (no-op for static schedulers).

        For CLC schedulers: producer acquire + issue CLC query + producer state advance.
        Only called by the scheduler warp.
        """
        ...

    def producer_tail(self, *, loc=None, ip=None) -> None:
        """Producer-side cleanup after the last tile.

        No-op for static schedulers. For CLC schedulers: pipeline producer_tail.
        """
        ...


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    mCuSeqlensQ: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    lpt: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)

        @staticmethod
        def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.cluster_shape_mn,
            )

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileScheduler":
        return SingleTileScheduler(params, cute.arch.block_idx(), loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # TODO: this hard-codes the fact that we only use cluster = (1, 1) or (2, 1)
        assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
        grid_x = cute.round_up(params.num_block, params.cluster_shape_mn[0])
        return (
            grid_x,
            params.num_head,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, Int32(0)),
            self._is_first_block,
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        scheduler = SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)
        scheduler._is_first_block = self._is_first_block
        return scheduler


class SingleTileMaxVarlenScheduler:
    """Launch a max-size 3D grid and reject per-sample tail tiles in O(1)."""

    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        tile_size: cutlass.Constexpr[int]
        mCuSeqlens: cute.Tensor

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        loc=None,
        ip=None,
    ) -> Params:
        assert args.mCuSeqlensQ is not None
        return SingleTileMaxVarlenScheduler.Params(
            num_block=args.num_block,
            num_head=args.num_head,
            num_batch=args.num_batch,
            tile_size=args.tile_shape_mn[0],
            mCuSeqlens=args.mCuSeqlensQ,
        )

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileMaxVarlenScheduler":
        return SingleTileMaxVarlenScheduler(params, cute.arch.block_idx(), loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(params: Params, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
        return (params.num_block, params.num_head, params.num_batch)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        seqlen = Int32(0)
        if batch_idx < self.params.num_batch:
            seqlen = self.params.mCuSeqlens[batch_idx + 1] - self.params.mCuSeqlens[batch_idx]
        valid = self._is_first_block and batch_idx < self.params.num_batch and block_idx * self.params.tile_size < seqlen
        return WorkTileInfo((Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        rebuilt = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            rebuilt.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        scheduler = self.__class__(*rebuilt, loc=self._loc, ip=self._ip)
        scheduler._is_first_block = self._is_first_block
        return scheduler


class SingleTileLPTBwdScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_block: Int32
        seqlen: Int32
        block_size: cutlass.Constexpr[int]
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisor
        l2_minor_divmod: FastDivmodDivisor
        l2_major_divmod: FastDivmodDivisor
        l2_minor_residual_divmod: FastDivmodDivisor
        num_hb_quotient: Int32
        mCuSeqlens: Optional[cute.Tensor] = None
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        spt: cutlass.Constexpr[bool] = True

        @staticmethod
        @cute.jit
        def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "SingleTileLPTBwdScheduler.Params":
            size_l2 = 40 * 1024 * 1024
            size_one_qdo_head = cutlass.Int64(args.seqlen_k) * (args.headdim + args.headdim_v) * args.element_size
            size_one_dqaccum_head = cutlass.Int64(args.seqlen_k) * args.headdim * 4
            size_one_head = size_one_qdo_head + size_one_dqaccum_head
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(Int32(size_l2 // size_one_head)))
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            num_block = cute.ceil_div(args.num_block, args.cluster_shape_mn[0])
            return SingleTileLPTBwdScheduler.Params(
                total_blocks=(num_block * args.cluster_shape_mn[0]) * args.num_head * args.num_batch,
                num_block=num_block,
                seqlen=args.num_block * args.tile_shape_mn[0],
                block_size=args.tile_shape_mn[0],
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(max(num_hb_remainder, 1)),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
                mCuSeqlens=args.mCuSeqlensQ,
                cluster_shape_mn=args.cluster_shape_mn,
                spt=args.lpt,
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileLPTBwdScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTBwdScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileLPTBwdScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (params.total_blocks, Int32(1), Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        cluster_idx = self._tile_idx // self.params.cluster_shape_mn[0]
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(cluster_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        seqlen = params.seqlen
        if const_expr(params.mCuSeqlens is not None):
            seqlen = params.mCuSeqlens[batch_idx + 1] - params.mCuSeqlens[batch_idx]
        num_blocks_actual = cute.ceil_div(
            cute.ceil_div(cutlass.max(seqlen, Int32(0)), params.block_size),
            params.cluster_shape_mn[0],
        )
        is_valid = self._tile_idx < params.total_blocks and block < num_blocks_actual
        if cutlass.const_expr(params.spt):
            block = num_blocks_actual - 1 - block
        if cutlass.const_expr(params.cluster_shape_mn[0] > 1):
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
            block = block * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


class SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        max_kvblock_in_l2: Int32
        tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
        mCuSeqlensQ: cute.Tensor
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
        lpt: cutlass.Constexpr[bool] = False
        head_swizzle: cutlass.Constexpr[bool] = False
        cluster_shape_m: cutlass.Constexpr[int] = 1

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            loc=None,
            ip=None,
        ) -> "SingleTileVarlenScheduler.Params":
            size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
            kv_block_size = (args.headdim + args.headdim_v) * args.element_size * args.tile_shape_mn[1]
            if args.head_swizzle:
                kv_block_size += args.headdim * 4 * args.tile_shape_mn[1]
            max_kvblock_in_l2 = size_l2 // kv_block_size
            assert args.mCuSeqlensQ is not None
            assert args.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
            return SingleTileVarlenScheduler.Params(
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                max_kvblock_in_l2=max_kvblock_in_l2,
                tile_shape_mn=args.tile_shape_mn,
                mCuSeqlensQ=args.mCuSeqlensQ,
                qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
                lpt=args.lpt,
                head_swizzle=args.head_swizzle,
                cluster_shape_m=args.cluster_shape_mn[0],
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._is_first_block = True
        self._loc = loc

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileVarlenScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileVarlenScheduler":
        return SingleTileVarlenScheduler(params, cute.arch.block_idx()[0], loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        total_blocks_max = (params.total_q + params.num_batch * (params.cluster_shape_m * params.tile_shape_mn[0] - 1)) // params.tile_shape_mn[0]
        # Round down to nearest multiple of cluster since odd excess is always padding.
        total_blocks_max = total_blocks_max // params.cluster_shape_m * params.cluster_shape_m
        total_blocks = total_blocks_max * params.num_head
        return (total_blocks, Int32(1), Int32(1))

    @cute.jit
    def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        cur_cu_seqlen = Int32(0)
        if batch_idx <= params.num_batch:
            cur_cu_seqlen = params.mCuSeqlensQ[batch_idx]
        next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
        seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(params.qhead_per_kvhead_packgqa > 1):
            seqlen *= params.qhead_per_kvhead_packgqa
        num_m_blocks = Int32(0)
        if batch_idx < params.num_batch:
            if lane < cute.arch.WARP_SIZE - 1:
                num_m_blocks = cute.ceil_div(
                    cute.ceil_div(seqlen, params.tile_shape_mn[0]),
                    params.cluster_shape_m,
                )
        return num_m_blocks

    @cute.jit
    def _varlen_coord_map(self) -> WorkTileInfo:
        """Map self._tile_idx to (block, head, batch) via warp-level prefix sums."""
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
        # Total number of blocks for the next 31 batches
        m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
        # Same for all lanes
        group_end_tile = m_blocks_in_group * params.num_head
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = self._tile_idx // params.cluster_shape_m
        while group_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= params.num_batch:
                batch_idx = Int32(params.num_batch)
                group_end_tile = next_tile_idx + 1
            else:
                num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=batch_idx)
                num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
                m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
                group_end_tile += m_blocks_in_group * params.num_head
        is_valid = False
        if batch_idx >= params.num_batch:
            block, head_idx, batch_idx = Int32(0), Int32(0), Int32(params.num_batch)
        else:
            group_start_tile = group_end_tile - m_blocks_in_group * params.num_head
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_group = cute.arch.popc(cute.arch.vote_ballot_sync(group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx))
            batch_idx += batch_idx_in_group
            num_m_blocks_prev_lane = 0 if batch_idx_in_group == 0 else cute.arch.shuffle_sync(num_m_blocks_cumulative, batch_idx_in_group - 1)
            num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
            mh_block = next_tile_idx - group_start_tile - num_m_blocks_prev_lane * params.num_head
            if cutlass.const_expr(params.lpt or params.head_swizzle):
                # This is a version of the SingleTileLPTScheduler, complicated by the fact that
                # the seqlen can vary per batch.
                # TODO: is there any case where num_m_blocks is 0?
                # TODO: by right we should read the seqlen_kv but we're assuming seqlen_q == seqlen_k here
                num_n_blocks = num_m_blocks * params.tile_shape_mn[0] * params.cluster_shape_m // params.qhead_per_kvhead_packgqa // params.tile_shape_mn[1]
                # Seems faster to have this be a power of 2
                nheads_in_l2 = (
                    16
                    if num_n_blocks * 16 <= params.max_kvblock_in_l2
                    else (
                        8
                        if num_n_blocks * 8 <= params.max_kvblock_in_l2
                        else (4 if num_n_blocks * 4 <= params.max_kvblock_in_l2 else (2 if num_n_blocks * 2 <= params.max_kvblock_in_l2 else 1))
                    )
                )
                nheads_in_l2 = min(nheads_in_l2, params.num_head)
                mh_in_l2 = nheads_in_l2 * num_m_blocks
                section_idx = mh_block // mh_in_l2
                l2_mod = mh_block - section_idx * mh_in_l2
                # Deal with tail section
                nheads_in_this_section = nheads_in_l2 if nheads_in_l2 * (section_idx + 1) <= params.num_head else params.num_head - section_idx * nheads_in_l2
                block = l2_mod // nheads_in_this_section
                head_idx_residual = l2_mod - block * nheads_in_this_section
                head_idx = section_idx * nheads_in_l2 + head_idx_residual
                if cutlass.const_expr(params.lpt):
                    block = num_m_blocks - 1 - block
            else:
                head_idx = mh_block // num_m_blocks
                block = mh_block - head_idx * num_m_blocks
            is_valid = self._is_first_block and batch_idx < params.num_batch
            if cutlass.const_expr(params.cluster_shape_m > 1):
                bidx_in_cluster = cute.arch.block_in_cluster_idx()
                block = block * params.cluster_shape_m + bidx_in_cluster[0]
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self._varlen_coord_map()

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self._varlen_coord_map()

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        scheduler = self.__class__(*obj_list, loc=self._loc)
        scheduler._is_first_block = self._is_first_block
        return scheduler


# -----------------------------------------------------------------------------
# SM100 FMHA-specific schedulers (kept separate from generic schedulers).
# -----------------------------------------------------------------------------


class Sm100FmhaStaticTileSchedulerParams:
    """Parameters for the SM100 FMHA static tile scheduler."""

    def __init__(
        self,
        problem_shape_mbh: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.problem_shape_mbh = problem_shape_mbh
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.problem_shape_mbh]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.problem_shape_mbh], self._values_pos):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return Sm100FmhaStaticTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)


class Sm100FmhaStaticTileScheduler:
    """A one-tile static scheduler for SM100 FMHA kernels."""

    def __init__(
        self,
        params: Sm100FmhaStaticTileSchedulerParams,
        blk_coord: cute.Coord,
        *,
        loc=None,
        ip=None,
    ):
        self._params = params
        self._blk_coord = blk_coord
        self._is_first_block = True

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Sm100FmhaStaticTileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        return params.problem_shape_mbh

    @staticmethod
    def check_valid_work_for_seqlen_q(
        q_tiler: int,
        current_idx: Int32,
        seqlen_q: Int32,
    ) -> Boolean:
        """
        Check if the current work index is valid for the given query sequence length.

        This method verifies that the current work tile index multiplied by the
        query tiler size is within the bounds of the query sequence length.

        :param q_tiler: Query tiler size.
        :type q_tiler: int
        :param current_idx: Current work index.
        :type current_idx: Int32
        :param seqlen_q: Query sequence length.
        :type seqlen_q: Int32

        :return: True if the work is valid, False otherwise.
        :rtype: Boolean
        """
        return current_idx * q_tiler < seqlen_q

    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        blk_coord = self._blk_coord
        # cur_tile_coord is (mid, 0, (bid, hid))
        cur_tile_coord = (
            blk_coord[0],
            0,
            (blk_coord[1], blk_coord[2]),
        )

        return cutlass.utils.WorkTileInfo(cur_tile_coord, self._is_first_block)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        """
        Get the initial work tile information.

        :return: Initial WorkTileInfo.
        :rtype: WorkTileInfo
        """
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        """No-op for static scheduler."""
        pass

    def producer_tail(self, *, loc=None, ip=None):
        """No-op for static scheduler."""
        pass

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self._params)
        values.extend(extract_mlir_values(self._blk_coord))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 6
        new_params = new_from_mlir_values(self._params, values[0:3])
        new_blk_coord = new_from_mlir_values(self._blk_coord, values[3:])
        scheduler = Sm100FmhaStaticTileScheduler(new_params, new_blk_coord)
        scheduler._is_first_block = self._is_first_block
        return scheduler


def compute_sm100_fmha_grid(
    o_shape: cute.Shape,
    cta_tiler: Tuple[int, int, int],
) -> Tuple[Sm100FmhaStaticTileSchedulerParams, Tuple[int, int, int]]:
    """Compute grid parameters for FMHA (static scheduler).

    The output tensor o has shape (s, d, ((h_r, h_k), b)).
    """
    tile_sched_params = Sm100FmhaStaticTileSchedulerParams(
        (
            cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]),
            cute.size(o_shape[2][0]),
            cute.size(o_shape[2][1]),
        ),
    )
    grid = Sm100FmhaStaticTileScheduler.get_grid_shape(tile_sched_params)
    return tile_sched_params, grid


##############################################################################
# Fused Mask
##############################################################################


def make_sm100_thread_cooperative_group(size: int):
    return cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, size)


SM100_TMEM_CAPACITY_COLUMNS = 512
