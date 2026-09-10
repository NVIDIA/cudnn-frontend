# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: MIT

from enum import IntEnum, auto
from typing import Optional, Tuple
from dataclasses import dataclass, fields

import cutlass
from cutlass.pipeline import PipelineAsync, PipelineClcFetchAsync, PipelineState
import cutlass.cute as cute
from cutlass import Boolean, Int32
from cutlass.utils import (
    ClcDynamicPersistentTileScheduler,
    ClcDynamicPersistentTileSchedulerParams,
)

from . import utils


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, cutlass.Constexpr)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, cutlass.Constexpr)}
        non_constexpr_fields = {n: f for n, f in all_fields.items() if not isinstance(f, cutlass.Constexpr)}
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


class SchedulingMode(IntEnum):
    NONE = auto()
    STATIC = auto()
    DYNAMIC = auto()
    CLC = auto()


@dataclass
class ClcState(ParamsBase):
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


@dataclass
class WorkDescriptor(ParamsBase):
    m_block: Int32
    head_idx: Int32
    batch_idx: Int32
    is_valid: Boolean
    offset_q: Int32
    offset_k: Int32
    seqlen_q: Int32
    seqlen_k: Int32

    @property
    def tile_idx(self):
        return self.m_block, self.head_idx, self.batch_idx

    @property
    def is_valid_tile(self):
        return self.is_valid


@dataclass
class ClcDescriptorState(ParamsBase):
    pipeline: PipelineAsync
    buffer_ptr: cute.Pointer
    consumer_state: PipelineState
    producer_state: PipelineState

    @staticmethod
    def create(
        *,
        pipeline: PipelineAsync,
        buffer_ptr: cute.Pointer,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "ClcDescriptorState":
        return ClcDescriptorState(
            pipeline,
            buffer_ptr,
            consumer_state,
            producer_state,
        )

    @cute.jit
    def publish(self, work: WorkDescriptor, *, loc=None, ip=None):
        self.pipeline.producer_acquire(self.producer_state, loc=loc, ip=ip)
        sWork = cute.make_tensor(
            self.buffer_ptr,
            cute.make_layout((8, 2), stride=(1, 8)),
        )
        rWork = cute.make_rmem_tensor((8,), Int32)
        rWork[0] = work.m_block
        rWork[1] = work.head_idx
        rWork[2] = work.batch_idx
        rWork[3] = Int32(work.is_valid)
        rWork[4] = work.offset_q
        rWork[5] = work.offset_k
        rWork[6] = work.seqlen_q
        rWork[7] = work.seqlen_k
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            Int32,
            num_bits_per_copy=128,
        )
        if cute.arch.lane_idx() == 0:
            cute.copy(
                copy_atom,
                rWork,
                sWork[None, self.producer_state.index],
            )
        cute.arch.sync_warp()
        self.pipeline.producer_commit(self.producer_state, loc=loc, ip=ip)
        self.producer_state.advance(loc=loc, ip=ip)

    @cute.jit
    def consume(self, *, loc=None, ip=None) -> WorkDescriptor:
        self.pipeline.consumer_wait(self.consumer_state, loc=loc, ip=ip)
        sWork = cute.make_tensor(
            self.buffer_ptr,
            cute.make_layout((8, 2), stride=(1, 8)),
        )
        rWork = cute.make_rmem_tensor((8,), Int32)
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            Int32,
            num_bits_per_copy=128,
        )
        lane_idx = cute.arch.lane_idx()
        if lane_idx == 0:
            cute.copy(
                copy_atom,
                sWork[None, self.consumer_state.index],
                rWork,
            )
        values = [Int32(0) for _ in range(cute.size(rWork))]
        for idx in cutlass.range_constexpr(cute.size(rWork)):
            value = Int32(0)
            if lane_idx == 0:
                value = rWork[idx]
            values[idx] = cute.arch.shuffle_sync(value, 0)
        return WorkDescriptor(
            values[0],
            values[1],
            values[2],
            values[3] != 0,
            values[4],
            values[5],
            values[6],
            values[7],
        )

    def release(self, *, loc=None, ip=None):
        self.pipeline.consumer_release(self.consumer_state, loc=loc, ip=ip)
        self.consumer_state.advance(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        self.pipeline.producer_tail(self.producer_state, loc=loc, ip=ip)


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
    cu_seqlens_q: Optional[cute.Tensor] = None
    cu_seqlens_k: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    use_clc_response_warp: cutlass.Constexpr[bool] = False


class SingleTileBwdScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)

        @staticmethod
        @cute.jit
        def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "SingleTileBwdScheduler.Params":
            return SingleTileBwdScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.cluster_shape_mn,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Tuple[Int32, Int32, Int32],
        is_first_block: Boolean = True,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._is_first_block = is_first_block
        self._loc = loc

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileBwdScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> "SingleTileBwdScheduler":
        block_idx = cute.arch.block_idx()
        tile_idx = (
            Int32(block_idx[0]),
            Int32(block_idx[1]),
            Int32(block_idx[2]),
        )
        return SingleTileBwdScheduler(
            params,
            tile_idx,
            True,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return cute.round_up(
            (params.num_block, params.num_head, params.num_batch),
            (*params.cluster_shape_mn, 1),
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None):
        return cutlass.utils.WorkTileInfo(
            self._tile_idx,
            self._is_first_block,
        )

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._is_first_block]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._is_first_block]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        params, tile_idx, is_first_block = obj_list
        return self.__class__(
            params,
            tile_idx,
            is_first_block=is_first_block,
            loc=self._loc,
        )


class SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        seqlen_k: Int32
        max_kvblock_in_l2: Int32
        tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
        cu_seqlens_q: Optional[cute.Tensor] = None
        cu_seqlens_k: Optional[cute.Tensor] = None
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
        lpt: cutlass.Constexpr[bool] = False
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC
        use_clc_descriptor: cutlass.Constexpr[bool] = False
        use_clc_response_warp: cutlass.Constexpr[bool] = False

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            use_clc_descriptor: bool = False,
            loc=None,
            ip=None,
        ) -> "SingleTileVarlenScheduler.Params":
            assert scheduling_mode in (
                SchedulingMode.STATIC,
                SchedulingMode.CLC,
            ), f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
            max_kvblock_in_l2 = size_l2 // ((args.headdim + args.headdim_v) * args.element_size * args.tile_shape_mn[1])
            assert args.cu_seqlens_q is not None, "At least one of cu_seqlens_q must be provided"
            return SingleTileVarlenScheduler.Params(
                num_block=args.num_block,
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                seqlen_k=args.seqlen_k,
                max_kvblock_in_l2=max_kvblock_in_l2,
                tile_shape_mn=args.tile_shape_mn,
                cu_seqlens_q=args.cu_seqlens_q,
                cu_seqlens_k=args.cu_seqlens_k,
                qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
                lpt=args.lpt,
                cluster_shape_mn=args.cluster_shape_mn,
                scheduling_mode=scheduling_mode,
                use_clc_descriptor=use_clc_descriptor,
                use_clc_response_warp=args.use_clc_response_warp,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        clc: Optional[ClcState] = None,
        descriptor: Optional[ClcDescriptorState] = None,
        descriptor_producer: bool = False,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._is_first_block = True
        self.clc = clc
        self.descriptor = descriptor
        self.descriptor_producer = descriptor_producer
        self._loc = loc

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        use_clc_descriptor: bool = False,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileVarlenScheduler.Params.create(
            args,
            scheduling_mode=scheduling_mode,
            use_clc_descriptor=use_clc_descriptor,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=(
                params.num_block * params.cluster_shape_mn[0],
                params.num_head,
                params.num_batch,
            ),
            cluster_shape_mnk=(1, 1, 1),
        )

    @staticmethod
    def create(
        params: Params,
        clc: Optional[ClcState] = None,
        descriptor: Optional[ClcDescriptorState] = None,
        descriptor_producer: bool = False,
        *,
        loc=None,
        ip=None,
    ) -> "SingleTileVarlenScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileVarlenScheduler(
            params,
            tile_idx,
            clc,
            descriptor,
            descriptor_producer,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if cutlass.const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return (
                params.num_block * params.cluster_shape_mn[0],
                params.num_head,
                params.num_batch,
            )
        total_blocks_max = (params.total_q + params.num_batch * (params.tile_shape_mn[0] - 1)) // params.tile_shape_mn[0]
        return (total_blocks_max * params.num_head, Int32(1), Int32(1))

    @cute.jit
    def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        assert params.cu_seqlens_q is not None
        cur_cu_seqlen = Int32(0)
        if batch_idx <= params.num_batch:
            cur_cu_seqlen = params.cu_seqlens_q[batch_idx]
        next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
        seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(params.qhead_per_kvhead_packgqa > 1):
            seqlen *= params.qhead_per_kvhead_packgqa
        return cute.ceil_div(seqlen, params.tile_shape_mn[0]) if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1 else Int32(0)

    @cute.jit
    def _clc_work_to_coords(self, clc_work):
        params = self.params
        block_idx = Int32(0)
        head_idx = Int32(0)
        batch_idx = Int32(0)
        if clc_work.is_valid_tile:
            q_block = Int32(clc_work.tile_idx[0]) // params.cluster_shape_mn[0]
            head_idx = Int32(clc_work.tile_idx[1])
            batch_idx = Int32(clc_work.tile_idx[2])
            seqlen_q = Int32(0)
            if cute.arch.lane_idx() == 0:
                assert params.cu_seqlens_q is not None
                seqlen_q = params.cu_seqlens_q[batch_idx + 1] - params.cu_seqlens_q[batch_idx]
            seqlen_q = cute.arch.shuffle_sync(seqlen_q, 0)
            if cutlass.const_expr(params.qhead_per_kvhead_packgqa > 1):
                seqlen_q *= params.qhead_per_kvhead_packgqa
            num_m_blocks = cute.ceil_div(seqlen_q, params.tile_shape_mn[0])
            if q_block < num_m_blocks:
                block_idx = num_m_blocks - 1 - q_block if cutlass.const_expr(params.lpt) else q_block
            else:
                block_idx = -cute.ceil_div(params.seqlen_k, params.tile_shape_mn[0]) - 1
        return cutlass.utils.WorkTileInfo(
            (block_idx, head_idx, batch_idx),
            clc_work.is_valid_tile,
        )

    @cute.jit
    def _clc_work_to_descriptor(self, clc_work) -> WorkDescriptor:
        params = self.params
        block_idx = Int32(0)
        head_idx = Int32(0)
        batch_idx = Int32(0)
        offset_q = Int32(0)
        offset_k = Int32(0)
        seqlen_q = Int32(0)
        seqlen_k = Int32(0)
        if clc_work.is_valid_tile:
            q_block = Int32(clc_work.tile_idx[0]) // params.cluster_shape_mn[0]
            head_idx = Int32(clc_work.tile_idx[1])
            batch_idx = Int32(clc_work.tile_idx[2])
            lane_idx = cute.arch.lane_idx()
            if lane_idx == 0:
                assert params.cu_seqlens_q is not None
                offset_q = params.cu_seqlens_q[batch_idx]
                seqlen_q = params.cu_seqlens_q[batch_idx + 1] - offset_q
            offset_q = cute.arch.shuffle_sync(offset_q, 0)
            seqlen_q = cute.arch.shuffle_sync(seqlen_q, 0)
            num_m_blocks = cute.ceil_div(
                seqlen_q * params.qhead_per_kvhead_packgqa,
                params.tile_shape_mn[0],
            )
            if q_block < num_m_blocks:
                if lane_idx == 0:
                    assert params.cu_seqlens_k is not None
                    offset_k = params.cu_seqlens_k[batch_idx]
                    seqlen_k = params.cu_seqlens_k[batch_idx + 1] - offset_k
                block_idx = num_m_blocks - 1 - q_block if cutlass.const_expr(params.lpt) else q_block
            else:
                block_idx = -cute.ceil_div(params.seqlen_k, params.tile_shape_mn[0]) - 1
            offset_k = cute.arch.shuffle_sync(offset_k, 0)
            seqlen_k = cute.arch.shuffle_sync(seqlen_k, 0)
        return WorkDescriptor(
            block_idx,
            head_idx,
            batch_idx,
            clc_work.is_valid_tile,
            offset_q,
            offset_k,
            seqlen_q,
            seqlen_k,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        if cutlass.const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            if cutlass.const_expr(self.params.use_clc_descriptor):
                assert self.descriptor_producer
                return self._clc_work_to_descriptor(self.clc.get_current_work())
            return self._clc_work_to_coords(self.clc.get_current_work())
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
        # Total number of blocks for the next 31 batches
        m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
        # Same for all lanes
        group_end_tile = m_blocks_in_group * params.num_head
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = self._tile_idx
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
            if cutlass.const_expr(params.lpt):
                # This is a version of the SingleTileLPTScheduler, complicated by the fact that
                # the seqlen can vary per batch.
                # The selected batch owns next_tile_idx, so num_m_blocks is nonzero.
                # L2 swizzling uses the Q length as its KV block-count estimate.
                num_n_blocks = num_m_blocks * params.tile_shape_mn[0] // params.qhead_per_kvhead_packgqa // params.tile_shape_mn[1]
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
                block = num_m_blocks - 1 - block
            else:
                head_idx = mh_block // num_m_blocks
                block = mh_block - head_idx * num_m_blocks
            is_valid = self._is_first_block and batch_idx < params.num_batch
        return cutlass.utils.WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx)), is_valid)

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if cutlass.const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            if cutlass.const_expr(self.params.use_clc_descriptor):
                if cutlass.const_expr(self.descriptor_producer):
                    work = self._clc_work_to_descriptor(self.clc.initial_work_tile_info())
                    self.descriptor.publish(work, loc=loc, ip=ip)
                    return work
                return self.descriptor.consume(loc=loc, ip=ip)
            return self._clc_work_to_coords(self.clc.initial_work_tile_info())
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        if cutlass.const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            if cutlass.const_expr(not self.params.use_clc_descriptor or self.descriptor_producer):
                self.clc.prefetch_next_work(loc=loc, ip=ip)

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None):
        if cutlass.const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            if cutlass.const_expr(self.params.use_clc_descriptor):
                if cutlass.const_expr(not self.descriptor_producer):
                    self.descriptor.release(loc=loc, ip=ip)
                    if cutlass.const_expr(not self.params.use_clc_response_warp):
                        self.clc.consumer_wait(loc=loc, ip=ip)
                        self._clc_work_to_coords(self.clc.get_current_work())
                        self.clc.consumer_release(loc=loc, ip=ip)
                    return self.descriptor.consume(loc=loc, ip=ip)
                self.clc.consumer_wait(loc=loc, ip=ip)
                work = self._clc_work_to_descriptor(self.clc.get_current_work())
                self.clc.consumer_release(loc=loc, ip=ip)
                self.descriptor.publish(work, loc=loc, ip=ip)
                return work
            self.clc.consumer_wait(loc=loc, ip=ip)
            work = self._clc_work_to_coords(self.clc.get_current_work())
            self.clc.consumer_release(loc=loc, ip=ip)
            return work
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._is_first_block = False
        return self.get_current_work(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        if cutlass.const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            if cutlass.const_expr(self.params.use_clc_descriptor and self.descriptor_producer):
                self.descriptor.producer_tail(loc=loc, ip=ip)
            self.clc.producer_tail(loc=loc, ip=ip)

    def consumer_tail(self, *, loc=None, ip=None):
        if cutlass.const_expr(self.params.scheduling_mode == SchedulingMode.CLC and self.params.use_clc_descriptor and not self.descriptor_producer):
            self.descriptor.release(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx]
        if cutlass.const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            if cutlass.const_expr(self.params.use_clc_descriptor):
                objs += [self.clc]
                objs += [self.descriptor]
            else:
                objs += [self.clc]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx]
        if cutlass.const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            if cutlass.const_expr(self.params.use_clc_descriptor):
                objs += [self.clc]
                objs += [self.descriptor]
            else:
                objs += [self.clc]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        params, tile_idx, *states = obj_list
        clc = None
        descriptor = None
        if cutlass.const_expr(params.scheduling_mode == SchedulingMode.CLC):
            if cutlass.const_expr(params.use_clc_descriptor):
                clc, descriptor = states
            else:
                clc = states[0]
        return self.__class__(
            params,
            tile_idx,
            clc=clc,
            descriptor=descriptor,
            descriptor_producer=self.descriptor_producer,
            loc=self._loc,
        )
