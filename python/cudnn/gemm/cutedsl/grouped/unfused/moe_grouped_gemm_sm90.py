# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""
MoE BF16 Grouped GEMM Kernel for Hopper (SM90).

The SM90 counterpart of moe_grouped_gemm.py for the 2Dx3D (forward and dgrad) scenario:
A (tokens_sum, K) x B (N, K, experts) -> C/D (tokens_sum, N).

Supports:
    - Dynamic persistent tile scheduling (MoEPersistentTileScheduler, gmem atomic counter)
    - Cooperative (both MMA warp groups on one tile) or pingpong (each warp group owns alternate tiles) schedules
    - Dense (contiguous 3-D B) weights, K-major (forward) or N-major (dgrad)
    - Routing-probability (prob) fusion and optional C output (generate_c)

WGMMA accumulates in registers, so the MMA warp groups also run the epilogue (no TMEM).
"""

from typing import Type, Tuple, Union, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator, get_smem_capacity_in_bytes
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warp, warpgroup
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.typing import Int32, AddressSpace
from ..canonical import kernel_facing_b, kernel_facing_mx, kernel_facing_prob
from ..moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from ..moe_utils import MoEWeightMode, rewrite_tensor_shape
from ..moe_sched_extension import ContiguousGroupedGemmSchedExtension
from ..moe_kernel_helpers import compute_grid, fmax, fmin, is_valid_tensor_alignment, silu_f32, silu_f32_geglu_scaled
from cudnn.frost.tile_dsl.consume import consume_then_arrive, release_zero


def _widen_for_cluster(tma_tensor, mode, tile, cluster_count):
    """A static extent of fewer tiles than cluster CTAs would alias later CTAs onto tile 0.

    Widen the TMA coordinate view to cluster_count tiles; the extra tiles are outside the descriptor (zero fill, clipped).
    """
    extent = tma_tensor.shape[mode]
    if not isinstance(extent, int) or -(-extent // tile) >= cluster_count:
        return tma_tensor
    shape = list(tma_tensor.shape)
    shape[mode] = tile * cluster_count
    return rewrite_tensor_shape(tma_tensor, tuple(shape))


KERNEL_SCHEDULES = ("cooperative", "pingpong")
# The one pingpong configuration (the only competitive one in an H100 tile sweep); every other tile or cluster is cooperative.
PINGPONG_TILE = (128, 192)
PINGPONG_CLUSTER = (2, 1)


def pingpong_config_error(mma_tiler_mn, cluster_shape_mn) -> Optional[str]:
    """Why a pingpong configuration is unsupported, or None (None tile or cluster: the pingpong default)."""
    tile = PINGPONG_TILE if mma_tiler_mn is None else tuple(mma_tiler_mn)
    cluster = PINGPONG_CLUSTER if cluster_shape_mn is None else tuple(cluster_shape_mn)
    if (tile, cluster) != (PINGPONG_TILE, PINGPONG_CLUSTER):
        return f"pingpong runs only mma_tiler_mn={PINGPONG_TILE} with cluster_shape_mn={PINGPONG_CLUSTER}, got {tile} and {cluster}"
    return None


def check_register_split(load_regs: int, mma_regs: int, num_mma_warp_groups: int) -> None:
    """setmaxnreg.inc waits for registers the decreases free: the split must fit the launch allocation or the MMA warps hang."""
    launch_regs = min(65536 // ((num_mma_warp_groups + 1) * 128) // 8 * 8, 256)
    if load_regs + num_mma_warp_groups * mma_regs > (num_mma_warp_groups + 1) * launch_regs:
        raise ValueError(f"setmaxnreg split {load_regs} + {num_mma_warp_groups} x {mma_regs} exceeds {num_mma_warp_groups + 1} x {launch_regs} registers")


def default_mma_tiler(pingpong: bool) -> Tuple[int, int]:
    """The SM90 CTA tile when the caller gives none: (128, 256) cooperative, (128, 192) pingpong."""
    return PINGPONG_TILE if pingpong else (128, 256)


# Tests and benchmarks only: "cooperative" or "pingpong" overrides every SM90 BF16 grouped GEMM's schedule.
_SCHEDULE_OVERRIDE: Optional[str] = None


def schedule_override() -> Optional[str]:
    """The schedule override (None: the shape rule's); wrapper memo and cache keys carry it."""
    return _SCHEDULE_OVERRIDE


def apply_schedule_override(rule_picks_pingpong: bool) -> bool:
    """The override's schedule if one is set, else the rule's."""
    if _SCHEDULE_OVERRIDE is None:
        return rule_picks_pingpong
    if _SCHEDULE_OVERRIDE not in KERNEL_SCHEDULES:
        raise ValueError(f"the SM90 schedule override must be 'cooperative' or 'pingpong', got {_SCHEDULE_OVERRIDE!r}")
    return _SCHEDULE_OVERRIDE == "pingpong"


def use_pingpong(n: int, k: int, mma_tiler_mn: Optional[Tuple[int, int]], cluster_shape_mn: Optional[Tuple[int, int]]) -> bool:
    """fwd / dgrad / GLU with or without C (n: the accumulator N, a GLU's 2I): the override, else pingpong for K <= 3072
    and N >= 2048 (from H100 sweeps over N and K) on the 128x192 / (2, 1) configuration only."""
    rule = k <= 3072 and n >= 2048 and pingpong_config_error(mma_tiler_mn, cluster_shape_mn) is None
    return apply_schedule_override(rule)


def check_pingpong_tile(pingpong: bool, mma_tiler_mn: Tuple[int, int], cluster_shape_mn: Tuple[int, int]) -> None:
    """A forced pingpong schedule with a configuration it cannot run raises; the heuristic never picks one."""
    error = pingpong_config_error(mma_tiler_mn, cluster_shape_mn) if pingpong else None
    if error is not None:
        raise ValueError(f"SM90: {error}")


class MoEGroupedGemmBf16KernelSm90:
    """Plain BF16 grouped GEMM kernel with MoE tile scheduling for Hopper.

    Dense weights, no bias, dynamic scheduling only. A/B use BF16 storage, WGMMA
    accumulates in FP32, and C/D are BF16 or FP16 (the epilogue stores through stmatrix).

    :param acc_dtype: Accumulator data type (Float32).
    :param use_2cta_instrs: Must be False (no 2-CTA MMA on Hopper).
    :param mma_tiler_mn: CTA tile shape (M, N); M is 64 or 128, N a multiple of 32 up to 256.
    :param cluster_shape_mn: Cluster shape (M, N).
    :param vectorized_f32: Accepted for API parity; Hopper has no packed FP32 math.
    :param generate_c: Generate C output tensor.
    :param enable_bias: Must be False.
    :param expert_cnt: Number of experts.
    :param weight_mode: Must be ``MoEWeightMode.DENSE``.
    :param use_dynamic_sched: Must be True.
    :param pingpong: Each MMA warp group owns whole tiles (alternate tiles of the scheduler's sequence) and
        the two overlap one's epilogue with the other's mainloop; a 128x192 tile and a (2, 1) cluster.
    """

    FIX_PAD_SIZE = 256

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        cd_major: str,
        m_aligned: int,
        pingpong: bool = False,
    ) -> bool:
        if pingpong and pingpong_config_error(mma_tiler_mn, cluster_shape_mn) is not None:
            return False
        fix_pad_size = MoEGroupedGemmBf16KernelSm90.FIX_PAD_SIZE
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        return (
            ab_dtype == cutlass.BFloat16
            and c_dtype in (cutlass.BFloat16, cutlass.Float16)
            and d_dtype in (cutlass.BFloat16, cutlass.Float16)
            and acc_dtype == cutlass.Float32
            and not use_2cta_instrs
            and mma_tiler_mn[0] in (64, 128)
            and mma_tiler_mn[1] in range(32, 257, 32)
            and is_power_of_2(cluster_shape_mn[0])
            and is_power_of_2(cluster_shape_mn[1])
            and cluster_shape_mn[0] * cluster_shape_mn[1] <= 8
            and m_aligned == fix_pad_size
            and fix_pad_size % (mma_tiler_mn[0] * cluster_shape_mn[0]) == 0
            and a_major == "k"
            and b_major in ("k", "n")
            and cd_major == "n"
            and is_valid_tensor_alignment(m, n, k, l, ab_dtype, d_dtype, a_major, b_major, cd_major)
            and n % 32 == 0
            and k > 0
            and m % 256 == 0
        )

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        generate_c: bool,
        enable_bias: bool,
        expert_cnt: int,
        weight_mode: MoEWeightMode = MoEWeightMode.DENSE,
        use_dynamic_sched: bool = True,
        act_func: Optional[str] = None,
        pingpong: bool = False,
    ):
        mma_tile_m = mma_tiler_mn[0]
        if act_func not in (None, "swiglu", "geglu"):
            raise ValueError(f"act_func must be None, 'swiglu' or 'geglu', got {act_func!r}")
        if act_func is not None and mma_tiler_mn[1] % 64 != 0:
            raise ValueError(f"A fused GLU pairs adjacent 32-column gate/up blocks: CTA tile N must be a multiple of 64, got {mma_tiler_mn[1]}.")
        if self.FIX_PAD_SIZE % (mma_tile_m * cluster_shape_mn[0]) != 0:
            # A cluster M tile past one padded expert segment would write the next expert's rows and read prob OOB.
            raise ValueError(
                f"FIX_PAD_SIZE ({self.FIX_PAD_SIZE}) must be divisible by mma_tiler_mn[0] * cluster_shape_mn[0] " f"({mma_tile_m} * {cluster_shape_mn[0]})."
            )
        if expert_cnt > 1024:
            raise ValueError("Expert count > 1024 is not supported.")
        if not isinstance(weight_mode, MoEWeightMode):
            raise TypeError(f"weight_mode must be a MoEWeightMode, got {type(weight_mode)}")
        if use_2cta_instrs:
            raise ValueError("SM90 has no 2-CTA MMA: use_2cta_instrs must be False.")
        if enable_bias or weight_mode != MoEWeightMode.DENSE or not use_dynamic_sched:
            raise ValueError("SM90 grouped GEMM: dense weights, no bias and the dynamic scheduler only.")
        if mma_tile_m not in (64, 128):
            raise ValueError(f"SM90 CTA tile M must be 64 or 128, got {mma_tile_m}.")
        if mma_tiler_mn[1] not in range(32, 257, 32):
            raise ValueError(f"SM90 CTA tile N must be a multiple of 32 in [32, 256], got {mma_tiler_mn[1]}.")
        pingpong_error = pingpong_config_error(mma_tiler_mn, cluster_shape_mn) if pingpong else None
        if pingpong_error is not None:
            raise ValueError(f"SM90: {pingpong_error}.")

        self.expert_cnt = expert_cnt
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = False
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.occupancy = 1

        # WGMMA's M is 64: a CTA tile M of 128 with N > 128 takes a second MMA warp group (no acc spill).
        # Pingpong: one warp group per tile, two warp groups on alternate tiles.
        self.pingpong = pingpong
        self.atom_layout_mnk = (2, 1, 1) if (not pingpong and mma_tiler_mn[0] > 64 and mma_tiler_mn[1] > 128) else (1, 1, 1)
        self.num_mma_warp_groups = 2 if pingpong else self.atom_layout_mnk[0] * self.atom_layout_mnk[1] * self.atom_layout_mnk[2]
        # Warp groups with their own tile, epilogue barrier, TMA-store warp and C/D SMEM buffers.
        self.num_epi_groups = 2 if pingpong else 1
        self.num_warps_per_warp_group = 4
        self.threads_per_warp = 32
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * self.threads_per_warp

        # MMA/epilogue warp groups first, then one producer warp group (TMA warp, scheduler warp, 2 idle warps).
        self.epilog_warp_id = tuple(range(self.num_mma_warp_groups * self.num_warps_per_warp_group))
        self.epi_store_warp_id = self.epilog_warp_id[0]
        self.tma_warp_id = self.num_mma_warp_groups * self.num_warps_per_warp_group
        self.sched_warp_id = self.tma_warp_id + 1
        self.num_mma_threads = self.num_mma_warp_groups * self.num_threads_per_warp_group
        self.num_epi_threads = self.num_mma_threads // self.num_epi_groups
        self.threads_per_cta = (self.num_mma_warp_groups + 1) * self.num_threads_per_warp_group
        self.threads_wo_sched = self.num_mma_threads + self.threads_per_warp
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        check_register_split(self.load_register_requirement, self.mma_register_requirement, self.num_mma_warp_groups)

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_threads,
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        # Pingpong: warp group w's epilogue barrier is 2 + w; its mainloop / epilogue turns are 5 + w / 7 + w.
        self.epilog_barrier_id_base = 2
        self.mma_order_barrier_id_base = 5
        self.epi_order_barrier_id_base = 7

        self.num_smem_capacity = get_smem_capacity_in_bytes("sm_90")
        self.vectorized_f32 = vectorized_f32
        self.generate_c = generate_c
        self.enable_bias = False
        self.weight_mode = weight_mode
        self.use_dynamic_sched = True
        self.num_epilog_warps = len(self.epilog_warp_id)
        # act_func fuses a GLU (interleaved 32-column gate/up blocks): D has half the accumulator's N columns.
        self.act_func = act_func
        self.fused_glu = act_func is not None

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

    def _setup_attributes(self):
        """Configure MMA / tile / stage / SMEM layouts from GEMM inputs."""
        tiled_mma = self._make_tiled_mma()
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        # No 2-CTA MMA on Hopper: the CTA tile is the MMA tile and the cluster's "V" mode is 1.
        self.cta_tile_shape_mnk = self.mma_tiler
        self.mma_tiler_d = (self.mma_tiler[0], self.mma_tiler[1] // 2, self.mma_tiler[2]) if self.fused_glu else self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (cute.make_layout(1),),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.mma_tiler,
            self.d_dtype,
            is_cooperative=self.atom_layout_mnk == (2, 1, 1),
        )
        if self.fused_glu and self.epi_tile[1] != 32:
            raise ValueError(f"A fused GLU needs a 32-column epilogue tile (the gate/up block), got {self.epi_tile[1]}.")

        (
            self.num_ab_stage,
            self.num_c_stage,
            self.num_d_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            self.mma_tiler,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.d_dtype,
            self.d_layout,
            self.num_smem_capacity,
            self.occupancy,
            self.generate_c,
            self.num_epi_groups,
            self.fused_glu,
        )

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.b_layout,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage * self.num_epi_groups,
        )
        self.d_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.d_dtype,
            self.d_layout,
            self.epi_tile,
            self.num_d_stage * self.num_epi_groups,
        )

    # ------------------------------------------------------------------
    # Stage computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stages(
        mma_tiler_mnk,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        epi_tile,
        c_dtype,
        c_layout,
        d_dtype,
        d_layout,
        num_smem_capacity,
        occupancy,
        generate_c,
        num_epi_groups=1,
        fused_glu=False,
    ):
        """A/B, C, D and tile-info stage counts; C/D stages are per epilogue group (each has its own buffers)."""
        num_tile_stage = 2

        a_smem_layout_stage_one = sm90_utils.make_smem_layout_a(a_layout, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_stage_one = sm90_utils.make_smem_layout_b(b_layout, mma_tiler_mnk, b_dtype, 1)
        c_smem_layout_stage_one = sm90_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
        d_smem_layout_stage_one = sm90_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)

        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(b_dtype, b_smem_layout_stage_one)
        mbar_helpers_bytes = 1024
        sinfo_bytes = 4 * 4 * num_tile_stage
        # Without C no sC is allocated, so its bytes go to A/B and D stages.
        c_bytes = cute.size_in_bytes(c_dtype, c_smem_layout_stage_one) * num_epi_groups if generate_c else 0
        d_bytes = cute.size_in_bytes(d_dtype, d_smem_layout_stage_one) * num_epi_groups

        free_bytes = num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes + d_bytes + sinfo_bytes)
        num_ab_stage = free_bytes // ab_bytes_per_stage
        # The bytes the A/B stages leave become D and C store stages in turn (up to one per subtile), so subtiles rarely wait on TMA stores.
        spare_bytes = free_bytes % ab_bytes_per_stage
        num_c_subtiles = -(-mma_tiler_mnk[0] // epi_tile[0]) * -(-mma_tiler_mnk[1] // epi_tile[1])
        num_d_subtiles = num_c_subtiles // 2 if fused_glu else num_c_subtiles
        num_c_stage, num_d_stage, grew = 1, 1, True
        while grew:
            grew = False
            if num_d_stage < num_d_subtiles and spare_bytes >= d_bytes:
                num_d_stage, spare_bytes, grew = num_d_stage + 1, spare_bytes - d_bytes, True
            if generate_c and num_c_stage < num_c_subtiles and spare_bytes >= c_bytes:
                num_c_stage, spare_bytes, grew = num_c_stage + 1, spare_bytes - c_bytes, True
        return num_ab_stage, num_c_stage, num_d_stage, num_tile_stage

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def get_desc_workspace_bytes(self) -> int:
        """Dense weights only: no per-expert TMA descriptors before the scheduler's counter."""
        return 0

    def get_workspace_bytes(self) -> int:
        return 4

    @cute.jit
    def _get_sched_counter_ptr(self, workspace_ptr):
        return cute.make_ptr(
            cutlass.Int32,
            workspace_ptr.toint() + self.get_desc_workspace_bytes(),
            AddressSpace.gmem,
            assumed_align=4,
        )

    # ------------------------------------------------------------------
    # helper_kernel: reset the dynamic scheduler's tile counter
    # ------------------------------------------------------------------

    @cute.kernel
    def helper_kernel(self, workspace_ptr):
        sched_counter = cute.make_tensor(self._get_sched_counter_ptr(workspace_ptr), cute.make_layout(1))
        sched_counter[0] = cutlass.Int32(0)

    helper_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        n: Int32,
        k: Int32,
        b_stride_size: cutlass.Int64,
        b_major_mode: cutlass.Constexpr,
        workspace_ptr,
        c: cute.Tensor,
        d: cute.Tensor,
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        bias: Optional[cute.Tensor],
        prob: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM.

        ``b`` is the dense (N, K, L) weight tensor; ``n``, ``k``, ``b_stride_size`` and ``bias`` are unused
        and kept so every grouped GEMM kernel takes the same arguments.
        """
        self._launch(a, b, b_major_mode, workspace_ptr, c, d, padded_offsets, alpha, prob, max_active_clusters, stream, None)

    def _launch(self, a, b, b_major_mode, workspace_ptr, c, d, padded_offsets, alpha, prob, max_active_clusters, stream, linear_offset):
        """Trace the launch for ``__call__``; ``linear_offset`` is the GeGLU up offset (None without a fused GLU)."""
        a = kernel_facing_mx(a)
        c = kernel_facing_mx(c)
        d = kernel_facing_mx(d)
        prob = kernel_facing_prob(prob)
        b = kernel_facing_b(b, b_major_mode == OperandMajorMode.MN)
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = a.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        self.a_layout = LayoutEnum.from_tensor(a)
        self.b_layout = LayoutEnum.from_tensor(b)
        self.c_layout = LayoutEnum.from_tensor(c)
        self.d_layout = LayoutEnum.from_tensor(d)
        self.a_major_mode = self.a_layout.sm90_mma_major_mode()
        self.b_major_mode = self.b_layout.sm90_mma_major_mode()

        if cutlass.const_expr(b.element_type != self.a_dtype):
            raise TypeError(f"A/B dtype must match: {self.a_dtype} != {b.element_type}")
        if cutlass.const_expr(self.d_dtype.width != 16 or self.c_dtype.width != 16):
            raise TypeError(f"SM90 epilogue stores C/D through stmatrix: they must be 16 bit, got {self.c_dtype}/{self.d_dtype}")

        self._setup_attributes()
        tiled_mma = self._make_tiled_mma()

        # SM90 A/B SMEM layouts are rank 3 (MN, K, STAGE), so the TMA atoms come from the tiler-driven builder.
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, cute.make_layout(1)),
            a,
            a_smem_layout,
            (self.mma_tiler[0], self.mma_tiler[2]),
            num_multicast=self.num_mcast_ctas_a,
        )

        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, cute.make_layout(1)),
            b,
            b_smem_layout,
            (self.mma_tiler[1], self.mma_tiler[2]),
            num_multicast=self.num_mcast_ctas_b,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = a_copy_size + b_copy_size

        c_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            c_smem_layout,
            self.epi_tile,
        )
        d_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            d,
            d_smem_layout,
            self.epi_tile,
        )
        cluster_n = self.cluster_shape_mn[1]
        tma_tensor_b = _widen_for_cluster(tma_tensor_b, 0, self.mma_tiler[1], cluster_n)
        tma_tensor_c = _widen_for_cluster(tma_tensor_c, 1, self.mma_tiler[1], cluster_n)
        tma_tensor_d = _widen_for_cluster(tma_tensor_d, 1, self.mma_tiler_d[1], cluster_n)

        # ---- Helper kernel: dynamic scheduler counter reset ----
        self.helper_kernel(workspace_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream, min_blocks_per_mp=1)

        # ---- Grid computation via MoE scheduler ----
        b_n, b_k, b_l = cute.shape(b)
        sched_params = MoESchedulerParams(
            scenario="2Dx3D",
            expert_shape=(self.expert_cnt, b_n, b_k),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            use_dynamic_sched=True,
        )
        self.sched_params, grid = compute_grid(sched_params, max_active_clusters, False)
        self.buffer_align_bytes = 1024

        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_tile_stage, True)

        # ---- Shared storage ----
        sC_cosize = cute.cosize(self.c_smem_layout_staged.outer)
        sC_field = cute.struct.Align[cute.struct.MemRange[self.c_dtype, sC_cosize], self.buffer_align_bytes]
        if cutlass.const_expr(not self.generate_c):
            sC_field = cute.struct.MemRange[self.c_dtype, 0]

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            scheduler: SchedulerStorage
            sC: sC_field
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, cute.cosize(self.d_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # ---- Launch ----
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_d,
            tma_tensor_d,
            padded_offsets,
            alpha,
            prob,
            linear_offset,
            workspace_ptr,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.d_smem_layout_staged,
            self.epi_tile,
            self.sched_params,
            cutlass.Int32(0),
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            max_number_threads=[self.threads_per_cta, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def epilog_r2s_copy(self, tiled_mma, layout, dtype):
        """Register-to-SMEM tiled copy for one epilogue tensor, tiled from the MMA's C layout (no TMEM on Hopper)."""
        copy_atom_c = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(layout.is_m_major_c(), 4),
            dtype,
        )
        tiled_copy_c_atom = cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma)
        copy_atom_r2s = sm90_utils.get_smem_store_op(layout, dtype, self.acc_dtype)
        return cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_c_atom)

    @cute.jit
    def store_c(
        self,
        tiled_copy_r2s_c,
        tma_atom_c,
        warp_idx,
        tRS_rC,
        tRS_rC_out,
        tRS_sC,
        bSG_gC,
        bSG_sC,
        c_pipeline,
        c_buffer,
        gmem_coord,
        epilog_barrier,
        epi_store_warp,
    ):
        tRS_rC_out.store(tRS_rC.load().to(self.c_dtype))
        cute.copy(tiled_copy_r2s_c, tRS_rC_out, tRS_sC[(None, None, None, c_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        epilog_barrier.arrive_and_wait()
        if warp_idx == epi_store_warp:
            cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, gmem_coord)])
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        epilog_barrier.arrive_and_wait()

    def _epi_buffer(self, subtile, num_stage, buffer_base):
        """SMEM buffer of an epilogue subtile; pingpong warp groups own disjoint buffer ranges."""
        if cutlass.const_expr(self.pingpong):
            return subtile % num_stage + buffer_base
        return subtile % num_stage

    # ------------------------------------------------------------------
    # GPU device kernel
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
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        prob: cute.Tensor,
        linear_offset: Optional[cutlass.Float32],
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cutlass.Constexpr,
        sched_params: MoESchedulerParams,
        rz_arg: cutlass.Int32,
    ):
        # The release-zero argument of tile_dsl.consume: the host passes 0; its only use is this call.
        rz = release_zero(rz_arg)
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_d)
            if cutlass.const_expr(self.generate_c):
                cpasync.prefetch_descriptor(tma_atom_c)

        total_token = padded_offsets[self.expert_cnt - 1]

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sched_storage = storage.scheduler

        # Hopper consumers are whole warp groups: one lane per warp per multicast partner arrives on the empty barrier.
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        # Pingpong: each A/B stage belongs to one warp group's tile, so one warp group releases it.
        ab_consumer_arrive_cnt = mcast_size * (self.num_mma_warp_groups // self.num_epi_groups) * self.num_warps_per_warp_group
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, ab_consumer_arrive_cnt)
        ab_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
        )

        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp)
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_wo_sched)
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # ---- Scheduler ----
        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            padded_offsets,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=self._get_sched_counter_ptr(workspace_ptr),
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner) if cutlass.const_expr(self.generate_c) else None
        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = sched_storage.sInfo.get_tensor(info_layout)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

        # SMEM fragments for WGMMA plus the register accumulator shared by MMA and epilogue.
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        if cutlass.const_expr(self.pingpong):
            # The tiled MMA spans one warp group: each pingpong warp group takes the whole tile.
            thr_mma = tiled_mma.get_slice(0)
        else:
            mma_warp_group_thread_layout = cute.make_layout(self.num_mma_warp_groups, stride=self.num_threads_per_warp_group)
            thr_mma = tiled_mma.get_slice(mma_warp_group_thread_layout(warp_group_idx))
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
        # Shape the accumulator from the CTA tile, not D: a fused GLU's D has half the N columns.
        acc_shape = thr_mma.partition_C(cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))).shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        if total_token <= 0:
            cute.arch.nvvm.exit()

        # The increase opens the MMA branch: ptxas ignores a decrease whose region reaches the MMA code (C7507).
        if warp_idx >= self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # ==== Scheduler warp (MoE Persistent Tile Scheduler) ====
        if warp_idx == self.sched_warp_id:
            work_tile_info = scheduler.initial_work_tile_info()
            tile_info_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tile_stage)
            while work_tile_info.is_valid_tile:
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = work_tile_info.expert_idx
                    sInfo[(1, tile_info_producer_state.index)] = work_tile_info.tile_m_idx
                    sInfo[(2, tile_info_producer_state.index)] = work_tile_info.tile_n_idx
                    sInfo[(3, tile_info_producer_state.index)] = work_tile_info.k_tile_cnt
                cute.arch.fence_proxy("async.shared", space="cta")
                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()
                work_tile_info = scheduler.advance_to_next_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = cutlass.Int32(-1)
                sInfo[(1, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(2, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # ==== DMA / TMA load warp ====
        if warp_idx == self.tma_warp_id:
            ext = ContiguousGroupedGemmSchedExtension()
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            consume_then_arrive(tile_info_pipeline.consumer_get_barrier(tile_info_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)

                real_a, _ = ext.get_gmem_tensor("a", mA_mkl, padded_offsets, work_tile_info)
                real_b, _ = ext.get_gmem_tensor("b", mB_nkl, padded_offsets, work_tile_info)
                gA_mkl = cute.local_tile(real_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB_nkl = cute.local_tile(real_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))

                # SM90 SMEM tensors are rank 3, so tma_partition groups 2 modes, not 3.
                a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 2),
                    cute.group_modes(gA_mkl, 0, 2),
                )
                b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 2),
                    cute.group_modes(gB_nkl, 0, 2),
                )

                tAgA_slice = tAgA[(None, work_tile_info.tile_m_idx, None, 0)]
                tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                for _ in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    cute.copy(tma_atom_a, tAgA_k, tAsA_pipe, tma_bar_ptr=tma_bar, mcast_mask=a_full_mcast_mask)
                    cute.copy(tma_atom_b, tBgB_k, tBsB_pipe, tma_bar_ptr=tma_bar, mcast_mask=b_full_mcast_mask)
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                consume_then_arrive(
                    tile_info_pipeline.consumer_get_barrier(tile_info_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz
                )
                tile_info_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        # ==== MMA + epilogue warp group(s) ====
        # WGMMA accumulates in these threads' registers, so the mainloop and the epilogue live in the same warps.
        if warp_idx < self.tma_warp_id and total_token > 0:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            epi_tidx = tidx
            epilog_barrier = self.epilog_sync_barrier
            epi_store_warp = self.epi_store_warp_id
            c_buffer_base = 0
            d_buffer_base = 0
            if cutlass.const_expr(self.pingpong):
                epi_tidx = tidx % self.num_threads_per_warp_group
                epilog_barrier = pipeline.NamedBarrier(
                    barrier_id=self.epilog_barrier_id_base + warp_group_idx,
                    num_threads=self.num_epi_threads,
                )
                epi_store_warp = warp_group_idx * self.num_warps_per_warp_group
                c_buffer_base = warp_group_idx * self.num_c_stage
                d_buffer_base = warp_group_idx * self.num_d_stage
                # A warp group waits on its own turn barriers and arrives on the other's.
                mma_turn = pipeline.NamedBarrier(barrier_id=self.mma_order_barrier_id_base + warp_group_idx, num_threads=self.num_mma_threads)
                mma_turn_next = pipeline.NamedBarrier(barrier_id=self.mma_order_barrier_id_base + 1 - warp_group_idx, num_threads=self.num_mma_threads)
                epi_turn = pipeline.NamedBarrier(barrier_id=self.epi_order_barrier_id_base + warp_group_idx, num_threads=self.num_mma_threads)
                epi_turn_next = pipeline.NamedBarrier(barrier_id=self.epi_order_barrier_id_base + 1 - warp_group_idx, num_threads=self.num_mma_threads)
            num_k_blocks = cute.size(tCrA, mode=[2])
            k_pipe_mmas = 1

            tiled_copy_r2s = self.epilog_r2s_copy(tiled_mma, self.d_layout, self.d_dtype)
            thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
            tRS_sD = thr_copy_r2s.partition_D(sD)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sD))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rC = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.d_dtype)
            size_tRS_rD = cute.size(tRS_rD)
            tRS_rProb = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.Float32)
            # Per-element (row, col) of the accumulator in the CTA tile: WGMMA's C layout scatters rows across a thread.
            cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
            # Sliced per thread: thr_mma is sliced per warp group, which would pin every thread to its first rows.
            tRS_cAcc = tiled_copy_r2s.retile(tiled_mma.get_slice(epi_tidx).partition_C(cAcc))

            if cutlass.const_expr(self.generate_c):
                tiled_copy_r2s_c = self.epilog_r2s_copy(tiled_mma, self.c_layout, self.c_dtype)
                thr_copy_r2s_c = tiled_copy_r2s_c.get_slice(epi_tidx)
                tRS_sC = thr_copy_r2s_c.partition_D(sC)
                tRS_rC_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
                if cutlass.const_expr(self.fused_glu):
                    tRS_rC_up = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

            epi_ext = ContiguousGroupedGemmSchedExtension()
            ab_consumer_read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            ab_consumer_release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_threads)
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)
            d_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_threads)
            d_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_d_stage, producer_group=d_producer_group)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            consume_then_arrive(tile_info_pipeline.consumer_get_barrier(tile_info_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz)
            tile_info_consumer_state.advance()

            if cutlass.const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # Warp group 0 takes the first turns.
                    mma_turn.arrive()
                    epi_turn.arrive()
                else:
                    # Warp group 1 starts at the second tile; skipping a tile skips its A/B stages.
                    if is_valid_tile:
                        ab_consumer_read_state.advance(cutlass.Int32(tile_info[3]))
                        ab_consumer_release_state.advance(cutlass.Int32(tile_info[3]))
                        tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                        for idx in cutlass.range(4, unroll_full=True):
                            tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                        is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        consume_then_arrive(
                            tile_info_pipeline.consumer_get_barrier(tile_info_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz
                        )
                        tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                epi_work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                k_tile_cnt = epi_work_tile_info.k_tile_cnt
                expert_idx = epi_work_tile_info.expert_idx
                epi_ext.update_expert_info(padded_offsets, expert_idx)
                alpha_val = cutlass.Float32(alpha[expert_idx])

                # ---- mainloop: one accumulator init and one wgmma fence per output tile ----
                ab_consumer_read_state.reset_count()
                ab_consumer_release_state.reset_count()
                # The first k-block zero-inits (K >= 1, so every tile has one): no register fill to sink past the WGMMA.
                tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
                if cutlass.const_expr(self.pingpong):
                    mma_turn.arrive_and_wait()
                warpgroup.fence()

                prologue_mma_cnt = cutlass.min(k_pipe_mmas, k_tile_cnt)
                for _ in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                    ab_pipeline.consumer_wait(ab_consumer_read_state)
                    for kblock_idx in cutlass.range_constexpr(num_k_blocks):
                        kblock_coord = (None, None, kblock_idx, ab_consumer_read_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[kblock_coord], tCrB[kblock_coord], accumulators)
                        tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.commit_group()
                    ab_consumer_read_state.advance()

                for _ in cutlass.range(prologue_mma_cnt, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.consumer_wait(ab_consumer_read_state)
                    for kblock_idx in cutlass.range_constexpr(num_k_blocks):
                        kblock_coord = (None, None, kblock_idx, ab_consumer_read_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[kblock_coord], tCrB[kblock_coord], accumulators)
                    warpgroup.commit_group()
                    warpgroup.wait_group(k_pipe_mmas)
                    ab_pipeline.consumer_release(ab_consumer_release_state)
                    ab_consumer_release_state.advance()
                    ab_consumer_read_state.advance()

                if cutlass.const_expr(self.pingpong):
                    # All of this tile's WGMMAs are issued: the other warp group's mainloop may start.
                    mma_turn_next.arrive()
                warpgroup.wait_group(0)
                for _ in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                    ab_pipeline.consumer_release(ab_consumer_release_state)
                    ab_consumer_release_state.advance()

                # ---- epilogue ----
                real_d, _ = epi_ext.get_gmem_tensor("d", mD_mnl, padded_offsets, epi_work_tile_info)
                real_c, _ = epi_ext.get_gmem_tensor("c", mC_mnl, padded_offsets, epi_work_tile_info)
                real_prob, _ = epi_ext.get_gmem_tensor("prob", prob, padded_offsets, epi_work_tile_info)
                m_base = epi_work_tile_info.tile_m_idx * self.cta_tile_shape_mnk[0]

                gD_mnl_loop = cute.local_tile(real_d, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                gD_tile = gD_mnl_loop[(None, None, epi_work_tile_info.tile_m_idx, epi_work_tile_info.tile_n_idx, 0)]
                gD_epi = cute.zipped_divide(gD_tile, epi_tile)
                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_d,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sD, 0, 2),
                    gD_epi,
                )
                if cutlass.const_expr(self.generate_c):
                    gC_mnl_loop = cute.local_tile(real_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                    gC_tile = gC_mnl_loop[(None, None, epi_work_tile_info.tile_m_idx, epi_work_tile_info.tile_n_idx, 0)]
                    gC_epi = cute.zipped_divide(gC_tile, epi_tile)
                    bSG_sC, bSG_gC = cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        gC_epi,
                    )
                    c_subtile_shape = gC_epi.shape[1]
                    c_subtile_layout = cute.make_layout(c_subtile_shape, stride=(c_subtile_shape[1], 1))

                subtile_cnt = cute.size(gD_epi, mode=[1])
                subtile_shape = gD_epi.shape[1]
                subtile_layout = cute.make_layout(subtile_shape, stride=(subtile_shape[1], 1))

                if cutlass.const_expr(self.pingpong):
                    epi_turn.arrive_and_wait()
                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                    gmem_coord = subtile_layout.get_hier_coord(subtile_idx)
                    if cutlass.const_expr(self.fused_glu):
                        # A load past the previous subtile's barriers keeps ptxas from hoisting this subtile's
                        # activation above the C stores (spills at SM90's flat 168 registers).
                        if cutlass.const_expr(subtile_idx > 0 and self.generate_c and not self.pingpong):
                            alpha_val = cutlass.Float32(alpha[expert_idx])
                        # D subtile s reads accumulator subtiles 2s (gate) and 2s+1 (up): same rows, 32 columns apart.
                        acc_gate = 2 * subtile_idx * size_tRS_rD
                        acc_up = acc_gate + size_tRS_rD
                        if cutlass.const_expr(self.generate_c and self.pingpong):
                            # Pingpong's 192-register accumulator: C (gate, then up) is stored first, straight from it, and the
                            # alpha reload past those stores' barriers keeps the activation below them (0 spills).
                            for half in cutlass.range_constexpr(2):
                                for i in cutlass.range_constexpr(size_tRS_rD):
                                    tRS_rC[i] = tRS_rAcc[acc_gate + half * size_tRS_rD + i] * alpha_val
                                self.store_c(
                                    tiled_copy_r2s_c,
                                    tma_atom_c,
                                    warp_idx,
                                    tRS_rC,
                                    tRS_rC_out,
                                    tRS_sC,
                                    bSG_gC,
                                    bSG_sC,
                                    c_pipeline,
                                    self._epi_buffer(2 * num_prev_subtiles + half, self.num_c_stage, c_buffer_base),
                                    c_subtile_layout.get_hier_coord(2 * subtile_idx + half),
                                    epilog_barrier,
                                    epi_store_warp,
                                )
                            alpha_val = cutlass.Float32(alpha[expert_idx])
                        for i in cutlass.range_constexpr(size_tRS_rD):
                            acc_coord = tRS_cAcc[acc_gate + i]
                            if cutlass.const_expr(subtile_idx % subtile_shape[1] == 0):
                                tRS_rProb[i] = real_prob[m_base + acc_coord[0], 0, 0]
                            gate_value = tRS_rAcc[acc_gate + i] * alpha_val
                            up_value = tRS_rAcc[acc_up + i] * alpha_val
                            if cutlass.const_expr(self.generate_c and not self.pingpong):
                                tRS_rC[i] = gate_value
                                tRS_rC_up[i] = up_value
                            if cutlass.const_expr(self.act_func == "geglu"):
                                gate_value = fmin(gate_value, cutlass.Float32(7.0))
                                up_value = fmax(fmin(up_value, cutlass.Float32(7.0)), cutlass.Float32(-7.0))
                                act_value = (up_value + linear_offset) * silu_f32_geglu_scaled(gate_value, fastmath=True)
                            else:
                                act_value = up_value * silu_f32(gate_value, fastmath=True)
                            tRS_rD[i] = act_value * tRS_rProb[i]
                    else:
                        acc_base = subtile_idx * size_tRS_rD
                        for i in cutlass.range_constexpr(size_tRS_rD):
                            acc_coord = tRS_cAcc[acc_base + i]
                            # The row scale alpha * prob, set at the first N subtile of each M subtile: D takes one multiply.
                            if cutlass.const_expr(subtile_idx % subtile_shape[1] == 0):
                                tRS_rProb[i] = real_prob[m_base + acc_coord[0], 0, 0] * alpha_val
                            if cutlass.const_expr(self.generate_c):
                                tRS_rC[i] = tRS_rAcc[acc_base + i] * alpha_val
                            tRS_rD[i] = tRS_rAcc[acc_base + i] * tRS_rProb[i]

                    if cutlass.const_expr(self.generate_c and self.fused_glu and not self.pingpong):
                        # C keeps the pre-activation gate and up blocks; two C stores per D subtile, own buffers.
                        self.store_c(
                            tiled_copy_r2s_c,
                            tma_atom_c,
                            warp_idx,
                            tRS_rC,
                            tRS_rC_out,
                            tRS_sC,
                            bSG_gC,
                            bSG_sC,
                            c_pipeline,
                            self._epi_buffer(2 * num_prev_subtiles, self.num_c_stage, c_buffer_base),
                            c_subtile_layout.get_hier_coord(2 * subtile_idx),
                            epilog_barrier,
                            epi_store_warp,
                        )
                        self.store_c(
                            tiled_copy_r2s_c,
                            tma_atom_c,
                            warp_idx,
                            tRS_rC_up,
                            tRS_rC_out,
                            tRS_sC,
                            bSG_gC,
                            bSG_sC,
                            c_pipeline,
                            self._epi_buffer(2 * num_prev_subtiles + 1, self.num_c_stage, c_buffer_base),
                            c_subtile_layout.get_hier_coord(2 * subtile_idx + 1),
                            epilog_barrier,
                            epi_store_warp,
                        )
                    elif cutlass.const_expr(self.generate_c and not self.fused_glu):
                        self.store_c(
                            tiled_copy_r2s_c,
                            tma_atom_c,
                            warp_idx,
                            tRS_rC,
                            tRS_rC_out,
                            tRS_sC,
                            bSG_gC,
                            bSG_sC,
                            c_pipeline,
                            self._epi_buffer(num_prev_subtiles, self.num_c_stage, c_buffer_base),
                            gmem_coord,
                            epilog_barrier,
                            epi_store_warp,
                        )

                    tRS_rD_out.store(tRS_rD.load().to(self.d_dtype))
                    d_buffer = self._epi_buffer(num_prev_subtiles, self.num_d_stage, d_buffer_base)
                    num_prev_subtiles = num_prev_subtiles + 1
                    cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, d_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    epilog_barrier.arrive_and_wait()
                    if warp_idx == epi_store_warp:
                        cute.copy(tma_atom_d, bSG_sD[(None, d_buffer)], bSG_gD[(None, gmem_coord)])
                        d_pipeline.producer_commit()
                        d_pipeline.producer_acquire()
                    epilog_barrier.arrive_and_wait()

                if cutlass.const_expr(self.pingpong):
                    # Own C/D buffers: no store drain before the other warp group's epilogue.
                    epi_turn_next.arrive()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                consume_then_arrive(
                    tile_info_pipeline.consumer_get_barrier(tile_info_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz
                )
                tile_info_consumer_state.advance()
                if cutlass.const_expr(self.pingpong):
                    # Skip the other warp group's next tile.
                    if is_valid_tile:
                        ab_consumer_read_state.advance(cutlass.Int32(tile_info[3]))
                        ab_consumer_release_state.advance(cutlass.Int32(tile_info[3]))
                        tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                        for idx in cutlass.range(4, unroll_full=True):
                            tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                        is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        consume_then_arrive(
                            tile_info_pipeline.consumer_get_barrier(tile_info_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz
                        )
                        tile_info_consumer_state.advance()

            if cutlass.const_expr(self.generate_c):
                c_pipeline.producer_tail()
            d_pipeline.producer_tail()

        # Cluster-exit barrier past every warp branch: a non-leader's remote arrive must not hit an exited leader.
        if cutlass.const_expr(self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
