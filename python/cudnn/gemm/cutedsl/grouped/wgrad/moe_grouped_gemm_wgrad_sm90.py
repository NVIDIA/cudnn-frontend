# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""
MoE BF16 Grouped GEMM Kernel for Hopper (SM90): Weight Gradient (2Dx2D).

Computes:  A(hidden, tokens_sum) x B(tokens_sum, intermediate)
        -> C(experts, hidden, intermediate)

The SM90 counterpart of moe_grouped_gemm_wgrad.py, with the same ``__call__``
arguments. WGMMA accumulates in registers, so the MMA warp groups also run the
epilogue. K (tokens) varies per expert and M (hidden) and N (intermediate) are
fixed across experts.

Supports:
    - Dynamic persistent tile scheduling: a gmem atomic counter (no CLC on SM90)
    - Dense (contiguous 3-D C) output, Tensor2D input order
    - K-major or MN-major A and B
    - k_tile_cnt == 0 handling (zero output for empty experts)
    - Cooperative (both MMA warp groups on one tile) or pingpong (each MMA warp group owns alternate tiles)

Scheduler: moe_persistent_scheduler.py (scenario="2Dx2D", atomic_counter_claim=True)
Extension: moe_sched_extension.py (WgradGemmSchedExtension)
"""

from typing import Type, Tuple, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator, get_smem_capacity_in_bytes
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.typing import AddressSpace
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils

from ..moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from ..moe_utils import MoEWeightMode, TensormapWorkspace, WGradInputOrder
from ..moe_sched_extension import WgradGemmSchedExtension
from ..unfused.moe_grouped_gemm_sm90 import apply_schedule_override, pingpong_config_error
from cudnn.frost.tile_dsl.consume import consume_then_arrive, release_zero


def use_pingpong_wgrad(n: int, rows_per_expert: float, mma_tiler_mn: Optional[Tuple[int, int]], cluster_shape_mn: Optional[Tuple[int, int]]) -> bool:
    """The override, else the shape rule on the 128x192 / (2, 1) configuration only (from H100 sweeps over the output N,
    the output M and the average rows per expert): pingpong for output N >= 2048, and for 768 <= N < 2048 when N is a
    multiple of 192 (no partial 192-wide tile) or the average rows per expert is at most 1024 (a short mainloop, so the
    overlapped epilogue outweighs the partial tile); cooperative for N >= 4096 with 4096 or more rows per expert."""
    rule = n >= 2048 or (n >= 768 and (n % 192 == 0 or rows_per_expert <= 1024))
    rule = rule and not (n >= 4096 and rows_per_expert >= 4096)
    return apply_schedule_override(rule and pingpong_config_error(mma_tiler_mn, cluster_shape_mn) is None)


class MoEGroupedGemmWgradBF16KernelSm90:
    """BF16 grouped GEMM kernel for MoE weight gradient (2Dx2D) on SM90.

    Takes the same constructor and ``can_implement`` arguments as the SM100
    ``MoEGroupedGemmWgradBF16Kernel``. SM90 has no 2-CTA MMA, CLC or TMEM, so it
    accepts only ``use_2cta_instrs=False``, dense output, Tensor2D input order and
    ``accumulate_on_output=False``.

    :param acc_dtype: Accumulator data type (Float32).
    :param use_2cta_instrs: Must be False on SM90.
    :param mma_tiler_mn: CTA tile shape (M, N): M in {64, 128}, N in [32, 256] step 32.
    :param cluster_shape_mn: Cluster shape (M, N).
    :param accumulate_on_output: Must be False on SM90.
    :param expert_cnt: Number of experts.
    :param weight_mode: ``MoEWeightMode.DENSE``.
    :param input_order: ``WGradInputOrder.Tensor2D``.
    :param pingpong: Two MMA warp groups own alternate tiles, so one's epilogue overlaps the other's mainloop.
        The tile must be 128x192 and the cluster (2, 1).
    """

    FIX_PAD_SIZE = 256

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        group_k_list: list,
        expert_cnt: int,
        a_major: str,
        b_major: str,
        weight_mode: MoEWeightMode,
        input_order: WGradInputOrder,
        pingpong: bool = False,
    ) -> bool:
        """Check whether a BF16 wgrad testcase is supported on SM90."""
        if group_k_list is None or len(mma_tiler_mn) != 2 or len(cluster_shape_mn) != 2:
            return False
        tile_m, tile_n = mma_tiler_mn
        cluster_m, cluster_n = cluster_shape_mn

        def is_power_of_2(x: int) -> bool:
            return x > 0 and (x & (x - 1)) == 0

        def is_16b_aligned(dtype: Type[cutlass.Numeric], contiguous_elems: int) -> bool:
            return contiguous_elems % (16 * 8 // dtype.width) == 0

        tokens_sum = sum(group_k_list)
        return (
            ab_dtype == cutlass.BFloat16
            and out_dtype in (cutlass.BFloat16, cutlass.Float16, cutlass.Float32)
            and acc_dtype == cutlass.Float32
            and a_major in ("k", "m")
            and b_major in ("k", "n")
            and weight_mode == MoEWeightMode.DENSE
            and input_order == WGradInputOrder.Tensor2D
            and not use_2cta_instrs
            and m > 0
            and n > 0
            and expert_cnt > 0
            and len(group_k_list) == expert_cnt
            and all(k >= 0 and k % MoEGroupedGemmWgradBF16KernelSm90.FIX_PAD_SIZE == 0 for k in group_k_list)
            and tile_m in (64, 128)
            and tile_n in range(32, 257, 32)
            and (not pingpong or pingpong_config_error(mma_tiler_mn, cluster_shape_mn) is None)
            and is_power_of_2(cluster_m)
            and is_power_of_2(cluster_n)
            and cluster_m * cluster_n <= 8
            and is_16b_aligned(ab_dtype, tokens_sum if a_major == "k" else m)
            and is_16b_aligned(ab_dtype, tokens_sum if b_major == "k" else n)
            and is_16b_aligned(out_dtype, n)
        )

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 256),
        cluster_shape_mn: Tuple[int, int] = (2, 1),
        accumulate_on_output: bool = False,
        expert_cnt: int = 1,
        weight_mode: MoEWeightMode = MoEWeightMode.DENSE,
        input_order: WGradInputOrder = WGradInputOrder.Tensor2D,
        pingpong: bool = False,
    ):
        if use_2cta_instrs or accumulate_on_output:
            raise ValueError("SM90 wgrad has no 2-CTA MMA and no accumulate_on_output")
        if weight_mode != MoEWeightMode.DENSE or input_order != WGradInputOrder.Tensor2D:
            raise ValueError("SM90 wgrad supports dense output and Tensor2D input order only")
        self.expert_cnt = expert_cnt
        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.weight_mode = weight_mode
        self.input_order = input_order

        pingpong_error = pingpong_config_error(mma_tiler_mn, cluster_shape_mn) if pingpong else None
        if pingpong_error is not None:
            raise ValueError(f"SM90 wgrad: {pingpong_error}")
        self.pingpong = pingpong
        if pingpong:
            # Two MMA warp groups, each with the whole 128-row tile (two WGMMA M blocks) and its own epilogue.
            self.atom_layout_mnk = (1, 1, 1)
            self.num_mma_warp_groups = 2
        else:
            # WGMMA's M is 64: a CTA tile M of 128 with N > 128 uses two MMA warp groups (no acc spill).
            self.atom_layout_mnk = (2, 1, 1) if (mma_tiler_mn[0] > 64 and mma_tiler_mn[1] > 128) else (1, 1, 1)
            self.num_mma_warp_groups = self.atom_layout_mnk[0]
        # Warp groups that run an epilogue of their own: C smem stages, epilogue barrier, TMA-store warp.
        self.num_epi_groups = 2 if pingpong else 1
        self.num_threads_per_warp_group = 128
        self.num_mma_threads = self.num_mma_warp_groups * self.num_threads_per_warp_group
        # MMA/epilogue warp groups first, then one producer warp group: TMA warp, scheduler warp, 2 idle warps.
        self.epi_store_warp_id = 0
        self.tma_warp_id = self.num_mma_warp_groups * 4
        self.sched_warp_id = self.tma_warp_id + 1
        self.threads_per_cta = self.num_mma_threads + self.num_threads_per_warp_group
        # Every MMA and TMA-warp thread releases each tile_info entry it read (racecheck flags an elected-lane release).
        self.sched_consumer_threads = self.num_mma_threads + 32
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

        self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_mma_threads)
        self.sched_sync_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=32)
        # Pingpong: epilogue barrier of MMA warp group w is 2 + 2 * w; the math and epilogue order barriers of
        # group w are mma_order_barrier_id + w and epi_order_barrier_id + w.
        self.mma_order_barrier_id = 5
        self.epi_order_barrier_id = 7

        self.smem_capacity = get_smem_capacity_in_bytes("sm_90")
        self.num_tile_stage = 2

    # ------------------------------------------------------------------
    # Workspace: the scheduler's tile counter only
    # ------------------------------------------------------------------

    def get_workspace_bytes(self) -> int:
        return 4

    @cute.jit
    def _get_sched_counter_ptr(self, workspace_ptr):
        return cute.make_ptr(cutlass.Int32, workspace_ptr.toint(), AddressSpace.gmem, assumed_align=4)

    # ------------------------------------------------------------------
    # _setup_attributes
    # ------------------------------------------------------------------

    def _make_tiled_mma(self):
        return sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.atom_layout_mnk,
            (64, self.mma_tiler[1]),
            a_source=warpgroup.OperandSource.SMEM,
        )

    def _setup_attributes(self) -> None:
        tiled_mma = self._make_tiled_mma()
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * 4)
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (cute.make_layout(1),))
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.mma_tiler,
            self.c_dtype,
            is_cooperative=self.atom_layout_mnk == (2, 1, 1),
        )

        a_stage_one = sm90_utils.make_smem_layout_a(self.a_layout, self.mma_tiler, self.a_dtype, 1)
        b_stage_one = sm90_utils.make_smem_layout_b(self.b_layout, self.mma_tiler, self.b_dtype, 1)
        c_stage_one = sm90_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, 1)
        ab_bytes = cute.size_in_bytes(self.a_dtype, a_stage_one) + cute.size_in_bytes(self.b_dtype, b_stage_one)
        # One C stage for each epilogue group; num_c_stage counts the stages of one group.
        c_bytes = cute.size_in_bytes(self.c_dtype, c_stage_one) * self.num_epi_groups
        free_bytes = self.smem_capacity - 1024 - c_bytes
        self.num_ab_stage = free_bytes // ab_bytes
        # The bytes the A/B stages leave become C stages (up to one per subtile), so subtiles rarely wait on TMA stores.
        num_subtiles = -(-self.mma_tiler[0] // self.epi_tile[0]) * -(-self.mma_tiler[1] // self.epi_tile[1])
        self.num_c_stage = min(1 + free_bytes % ab_bytes // c_bytes, num_subtiles)
        self.num_tma_load_bytes = ab_bytes

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(self.a_layout, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(self.b_layout, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.c_smem_layout_staged = sm90_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage * self.num_epi_groups)

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mat_a: cute.Tensor,  # (hidden, tokens_sum)
        mat_b: cute.Tensor,  # (tokens_sum, intermediate)
        out: cute.Tensor,  # (experts, hidden, intermediate)
        offs: cute.Tensor,  # (experts,) cumsum end offsets, int32
        workspace: cute.Tensor,  # the scheduler's tile counter
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        out_single_expert: Optional[cute.Tensor] = None,
    ) -> None:
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)
        hidden, tokens_sum = mat_a.shape
        a_gemm = cute.make_tensor(mat_a.iterator, cute.make_layout((hidden, tokens_sum, c1), stride=(mat_a.stride[0], mat_a.stride[1], c0)))
        tokens_sum_b, intermediate = mat_b.shape
        b_gemm = cute.make_tensor(
            mat_b.iterator,
            cute.make_layout((intermediate, tokens_sum_b, c1), stride=(mat_b.stride[1], mat_b.stride[0], c0)),
        )
        experts, hidden_c, intermediate_c = out.shape
        c_gemm = cute.make_tensor(
            out.iterator,
            cute.make_layout((hidden_c, intermediate_c, experts), stride=(out.stride[1], out.stride[2], out.stride[0])),
        )

        self.a_dtype = a_gemm.element_type
        self.b_dtype = b_gemm.element_type
        self.c_dtype = c_gemm.element_type
        self.a_layout = LayoutEnum.from_tensor(a_gemm)
        self.b_layout = LayoutEnum.from_tensor(b_gemm)
        self.c_layout = LayoutEnum.from_tensor(c_gemm)
        self.a_major_mode = self.a_layout.sm90_mma_major_mode()
        self.b_major_mode = self.b_layout.sm90_mma_major_mode()

        self._setup_attributes()
        tiled_mma = self._make_tiled_mma()

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, cute.make_layout(1)),
            a_gemm,
            a_smem_layout,
            (self.mma_tiler[0], self.mma_tiler[2]),
            num_multicast=self.num_mcast_ctas_a,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, cute.make_layout(1)),
            b_gemm,
            b_smem_layout,
            (self.mma_tiler[1], self.mma_tiler[2]),
            num_multicast=self.num_mcast_ctas_b,
        )
        c_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c_gemm, c_smem_layout, self.epi_tile)

        sched_params = MoESchedulerParams(
            scenario="2Dx2D",
            expert_shape=(experts, intermediate, hidden),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            atomic_counter_claim=True,
        )
        grid = MoESchedulerParams.get_grid_shape(sched_params, max_active_clusters)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            scheduler: MoEPersistentTileScheduler.make_storage_struct(self.num_tile_stage, use_dynamic_sched=True)
            sC: cute.struct.Align[cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)], 1024]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)], 1024]

        self.shared_storage = SharedStorage

        self.helper_kernel(workspace.iterator).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream, min_blocks_per_mp=1)
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            offs,
            sched_params,
            workspace.iterator,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            cutlass.Int32(0),
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def helper_kernel(self, workspace_ptr):
        """One thread resets the scheduler's tile counter."""
        counter = cute.make_tensor(self._get_sched_counter_ptr(workspace_ptr), cute.make_layout(1))
        counter[0] = cutlass.Int32(0)

    helper_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    # ------------------------------------------------------------------
    # kernel (GPU device kernel)
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        offs: cute.Tensor,
        sched_params: MoESchedulerParams,
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        rz_arg: cutlass.Int32,
    ):
        """GPU device kernel for MoE BF16 wgrad on SM90."""
        # The release-zero argument of tile_dsl.consume: the host passes 0; its only use is this call.
        rz = release_zero(rz_arg)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sched_storage = storage.scheduler

        # Hopper consumers are whole warp groups: one arrive per MMA warp of the tile's tiled MMA per multicast partner.
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, mcast_size * self.atom_layout_mnk[0] * 4),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
        )
        sched_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.sched_consumer_threads),
        )

        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            offs,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=self._get_sched_counter_ptr(workspace_ptr),
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        if cutlass.const_expr(cluster_size > 1):
            cute.arch.cluster_arrive_relaxed()

        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sInfo = sched_storage.sInfo.get_tensor(cute.make_layout((4, self.num_tile_stage), stride=(1, 4)))

        a_mcast_mask = None
        b_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast):
            a_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        if cutlass.const_expr(self.pingpong):
            thr_mma = tiled_mma.get_slice(0)
        else:
            thr_mma = tiled_mma.get_slice(cute.make_layout(self.num_mma_warp_groups, stride=self.num_threads_per_warp_group)(warp_group_idx))
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
        gC_shape = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        accumulators = cute.make_rmem_tensor(thr_mma.partition_C(gC_shape).shape[:3], self.acc_dtype)

        if cutlass.const_expr(cluster_size > 1):
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        ext = WgradGemmSchedExtension(
            tensormap_ctor=TensormapWorkspace(workspace_ptr, []),
            weight_mode=self.weight_mode,
            input_order=self.input_order,
        )

        # Each setmaxnreg region must end at the next one: ptxas ignores a decrease that can reach the MMA code.
        if warp_idx >= self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # ==== Scheduler warp ====
        if warp_idx == self.sched_warp_id:
            sched_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tile_stage)
            work_tile_info = scheduler.initial_work_tile_info()
            while work_tile_info.is_valid_tile:
                sched_pipeline.producer_acquire(sched_producer_state)
                with cute.arch.elect_one():
                    sInfo[(0, sched_producer_state.index)] = work_tile_info.expert_idx
                    sInfo[(1, sched_producer_state.index)] = work_tile_info.tile_m_idx
                    sInfo[(2, sched_producer_state.index)] = work_tile_info.tile_n_idx
                    sInfo[(3, sched_producer_state.index)] = work_tile_info.k_tile_cnt
                cute.arch.fence_proxy("async.shared", space="cta")
                self.sched_sync_barrier.arrive_and_wait()
                sched_pipeline.producer_commit(sched_producer_state)
                sched_producer_state.advance()
                work_tile_info = scheduler.advance_to_next_work()
            # The invalid tile is the sentinel that ends every consumer's loop.
            sched_pipeline.producer_acquire(sched_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, sched_producer_state.index)] = work_tile_info.expert_idx
                sInfo[(1, sched_producer_state.index)] = work_tile_info.tile_m_idx
                sInfo[(2, sched_producer_state.index)] = work_tile_info.tile_n_idx
                sInfo[(3, sched_producer_state.index)] = work_tile_info.k_tile_cnt
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            sched_pipeline.producer_commit(sched_producer_state)
            sched_producer_state.advance()
            sched_pipeline.producer_tail(sched_producer_state)

        # ==== TMA load warp ====
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            sched_pipeline.consumer_wait(sched_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, sched_consumer_state.index)]
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(tile_info)
            cute.arch.fence_proxy("async.shared", space="cta")
            consume_then_arrive(sched_pipeline.consumer_get_barrier(sched_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz)
            sched_consumer_state.advance()
            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(offs, work_tile_info.expert_idx)
                real_a, _ = ext.get_gmem_tensor("a", mA_mkl, offs, work_tile_info)
                real_b, _ = ext.get_gmem_tensor("b", mB_nkl, offs, work_tile_info)
                gA_mkl = cute.local_tile(real_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB_nkl = cute.local_tile(real_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape),
                    cute.group_modes(sA, 0, 2),
                    cute.group_modes(gA_mkl, 0, 2),
                )
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape),
                    cute.group_modes(sB, 0, 2),
                    cute.group_modes(gB_nkl, 0, 2),
                )
                tAgA_slice = tAgA[(None, work_tile_info.tile_m_idx, None, 0)]
                tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]

                ab_producer_state.reset_count()
                peek_empty = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_empty = ab_pipeline.producer_try_acquire(ab_producer_state)
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    ab_pipeline.producer_acquire(ab_producer_state, peek_empty)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_mcast_mask,
                    )
                    ab_producer_state.advance()
                    peek_empty = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_empty = ab_pipeline.producer_try_acquire(ab_producer_state)

                sched_pipeline.consumer_wait(sched_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, sched_consumer_state.index)]
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(tile_info)
                cute.arch.fence_proxy("async.shared", space="cta")
                consume_then_arrive(sched_pipeline.consumer_get_barrier(sched_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz)
                sched_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        # ==== MMA + epilogue warp groups ====
        if warp_idx < self.tma_warp_id:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            num_k_blocks = cute.size(tCrA, mode=[2])
            copy_atom_c = cute.make_copy_atom(warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4), self.c_dtype)
            tiled_copy_r2s = cute.make_tiled_copy_S(
                sm90_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype),
                cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma),
            )
            epi_tidx = tidx
            if cutlass.const_expr(self.pingpong):
                epi_tidx = tidx % self.num_threads_per_warp_group
            thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            rC_shape = cute.shape(thr_copy_r2s.partition_S(sC))[:3]
            tRS_rC = cute.make_rmem_tensor(rC_shape, self.acc_dtype)
            tRS_rC_out = cute.make_rmem_tensor(rC_shape, self.c_dtype)
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads // self.num_epi_groups),
            )
            epi_args = (tma_atom_c, mC_mnl, offs, ext, epi_tile, sC, tRS_sC, tiled_copy_r2s, tRS_rAcc, tRS_rC, tRS_rC_out, c_pipeline)

            ab_read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            ab_release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            sched_pipeline.consumer_wait(sched_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, sched_consumer_state.index)]
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(tile_info)
            cute.arch.fence_proxy("async.shared", space="cta")
            consume_then_arrive(sched_pipeline.consumer_get_barrier(sched_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz)
            sched_consumer_state.advance()
            num_prev_subtiles = cutlass.Int32(0)
            if cutlass.const_expr(not self.pingpong):
                while work_tile_info.is_valid_tile:
                    k_tile_cnt = work_tile_info.k_tile_cnt

                    # ---- mainloop (as _mainloop; inline keeps the loop's carried values, and so the PTX, unchanged) ----
                    ab_read_state.reset_count()
                    ab_release_state.reset_count()
                    accumulators.fill(0.0)
                    tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.fence()
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        ab_pipeline.consumer_wait(ab_read_state)
                        for kblock_idx in cutlass.range_constexpr(num_k_blocks):
                            kblock_coord = (None, None, kblock_idx, ab_read_state.index)
                            cute.gemm(tiled_mma, accumulators, tCrA[kblock_coord], tCrB[kblock_coord], accumulators)
                        warpgroup.commit_group()
                        # Keep one WGMMA batch in flight; release the stage of the one before it.
                        if k_tile > 0:
                            warpgroup.wait_group(1)
                            ab_pipeline.consumer_release(ab_release_state)
                            ab_release_state.advance()
                        ab_read_state.advance()
                    warpgroup.wait_group(0)
                    if k_tile_cnt > 0:
                        ab_pipeline.consumer_release(ab_release_state)
                        ab_release_state.advance()

                    num_prev_subtiles = self._epilogue(
                        *epi_args, work_tile_info, self.epilog_sync_barrier, warp_idx, self.epi_store_warp_id, None, num_prev_subtiles
                    )

                    sched_pipeline.consumer_wait(sched_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, sched_consumer_state.index)]
                    work_tile_info = MoEWorkTileInfo.from_rmem_tensor(tile_info)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    consume_then_arrive(sched_pipeline.consumer_get_barrier(sched_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz)
                    sched_consumer_state.advance()
            else:
                # Pingpong: group w owns tiles w, w + 2, ... of the sequence; both read (and release) every entry.
                group = warp_group_idx
                mma_turn = pipeline.NamedBarrier(barrier_id=self.mma_order_barrier_id + group, num_threads=self.num_mma_threads)
                mma_next = pipeline.NamedBarrier(barrier_id=self.mma_order_barrier_id + 1 - group, num_threads=self.num_mma_threads)
                epi_turn = pipeline.NamedBarrier(barrier_id=self.epi_order_barrier_id + group, num_threads=self.num_mma_threads)
                epi_next = pipeline.NamedBarrier(barrier_id=self.epi_order_barrier_id + 1 - group, num_threads=self.num_mma_threads)
                epilog_barrier = pipeline.NamedBarrier(barrier_id=2 + 2 * group, num_threads=self.num_threads_per_warp_group)
                # Group 1 hands group 0 the first math and epilogue turns.
                if group == 1:
                    mma_next.arrive()
                    epi_next.arrive()
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                tile_parity = cutlass.Int32(0)
                while work_tile_info.is_valid_tile:
                    k_tile_cnt = work_tile_info.k_tile_cnt
                    if tile_parity == group:
                        mma_turn.arrive_and_wait()
                        ab_read_state, ab_release_state = self._mainloop(
                            tiled_mma, accumulators, tCrA, tCrB, ab_pipeline, ab_read_state, ab_release_state, k_tile_cnt, mma_next
                        )
                        epi_turn.arrive_and_wait()
                        num_prev_subtiles = self._epilogue(
                            *epi_args, work_tile_info, epilog_barrier, warp_idx, group * 4, group * self.num_c_stage, num_prev_subtiles
                        )
                        epi_next.arrive()
                    else:
                        # The peer's tile: step over its A/B stages.
                        ab_read_state.advance(k_tile_cnt)
                        ab_release_state.advance(k_tile_cnt)
                    tile_parity = 1 - tile_parity

                    sched_pipeline.consumer_wait(sched_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, sched_consumer_state.index)]
                    work_tile_info = MoEWorkTileInfo.from_rmem_tensor(tile_info)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    consume_then_arrive(sched_pipeline.consumer_get_barrier(sched_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz)
                    sched_consumer_state.advance()
                # The group the sentinel's turn falls to takes the peer's last arrive on both order barriers.
                if tile_parity == group:
                    mma_turn.arrive_and_wait()
                    epi_turn.arrive_and_wait()
            c_pipeline.producer_tail()

        # Every CTA stays until each peer's remote arrive on its dynamic-claim barriers has landed.
        if cutlass.const_expr(cluster_size > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def _mainloop(
        self,
        tiled_mma: cute.TiledMma,
        accumulators: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        ab_pipeline: pipeline.PipelineTmaAsync,
        ab_read_state: pipeline.PipelineState,
        ab_release_state: pipeline.PipelineState,
        k_tile_cnt: cutlass.Int32,
        mma_next: Optional[pipeline.NamedBarrier] = None,
    ):
        """One tile's WGMMA mainloop (ACCUMULATE set); an empty expert (k_tile_cnt == 0) keeps the zeroed accumulators."""
        num_k_blocks = cute.size(tCrA, mode=[2])
        ab_read_state.reset_count()
        ab_release_state.reset_count()
        accumulators.fill(0.0)
        warpgroup.fence()
        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state)
            for kblock_idx in cutlass.range_constexpr(num_k_blocks):
                kblock_coord = (None, None, kblock_idx, ab_read_state.index)
                cute.gemm(tiled_mma, accumulators, tCrA[kblock_coord], tCrB[kblock_coord], accumulators)
            warpgroup.commit_group()
            # Keep one WGMMA batch in flight; release the stage of the one before it.
            if k_tile > 0:
                warpgroup.wait_group(1)
                ab_pipeline.consumer_release(ab_release_state)
                ab_release_state.advance()
            ab_read_state.advance()
        if cutlass.const_expr(mma_next is not None):
            # Every WGMMA of the tile is issued: the peer group may start its mainloop.
            mma_next.arrive()
        warpgroup.wait_group(0)
        if k_tile_cnt > 0:
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        return ab_read_state, ab_release_state

    @cute.jit
    def _epilogue(
        self,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        offs: cute.Tensor,
        ext: WgradGemmSchedExtension,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
        tRS_sC: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        tRS_rAcc: cute.Tensor,
        tRS_rC: cute.Tensor,
        tRS_rC_out: cute.Tensor,
        c_pipeline: pipeline.PipelineTmaStore,
        work_tile_info: MoEWorkTileInfo,
        epilog_barrier: pipeline.NamedBarrier,
        warp_idx: cutlass.Int32,
        store_warp_idx,
        c_stage_base: Optional[cutlass.Int32],
        num_prev_subtiles: cutlass.Int32,
    ) -> cutlass.Int32:
        """Registers -> smem -> TMA store, one epi_tile subtile at a time, through C stages c_stage_base + i."""
        size_tRS_rC = cute.size(tRS_rC)
        ext.update_expert_info(offs, work_tile_info.expert_idx)
        real_c, _ = ext.get_gmem_tensor("c", mC_mnl, offs, work_tile_info)
        gC = cute.local_tile(real_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        gC_epi = cute.zipped_divide(gC[(None, None, work_tile_info.tile_m_idx, work_tile_info.tile_n_idx, 0)], epi_tile)
        bSG_sC, bSG_gC = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), gC_epi)
        subtile_shape = gC_epi.shape[1]
        subtile_layout = cute.make_layout(subtile_shape, stride=(subtile_shape[1], 1))
        for subtile_idx in cutlass.range_constexpr(cute.size(gC_epi, mode=[1])):
            for i in cutlass.range_constexpr(size_tRS_rC):
                tRS_rC[i] = tRS_rAcc[subtile_idx * size_tRS_rC + i]
            tRS_rC_out.store(tRS_rC.load().to(self.c_dtype))
            c_buffer = num_prev_subtiles % self.num_c_stage
            if cutlass.const_expr(c_stage_base is not None):
                c_buffer = c_stage_base + c_buffer
            num_prev_subtiles = num_prev_subtiles + 1
            cute.copy(tiled_copy_r2s, tRS_rC_out, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_barrier.arrive_and_wait()
            if warp_idx == store_warp_idx:
                cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_layout.get_hier_coord(subtile_idx))])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            epilog_barrier.arrive_and_wait()
        return num_prev_subtiles
