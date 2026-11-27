# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""
MoE BF16 Grouped GEMM + dGLU Kernel for Hopper (SM90).

The SM90 counterpart of moe_grouped_gemm_dglu_dbias.py: A = dY (tokens, K) x B (N, K, experts), with
the dSwiGLU / dGeGLU epilogue fused. C (the forward's pre-activation) and D (dgate | dup) are
(tokens, 2 N) in interleaved 32-column gate/up blocks (``glu_layout="interleaved_32"``) or as gate
columns [0, N) then up columns [N, 2 N) (``glu_layout="gate_then_up"``, N a multiple of 128); dprob is an atomic per-row sum, or with
``deterministic`` one slot per (row, N tile) that the caller sums in a fixed order.

Supports:
    - Dynamic persistent tile scheduling only, dense K- or N-major weights
    - No dbias and no activation output
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
from ..glu_layout import validate_glu_layout
from ..moe_utils import MoEWeightMode, compute_expert_token_range, rewrite_tensor_shape
from ..moe_sched_extension import ContiguousGroupedGemmSchedExtension
from ..moe_kernel_helpers import atomic_add_float32, compute_grid, fmax, fmin, sigmoid_f32
from ..unfused.moe_grouped_gemm_sm90 import _widen_for_cluster, check_register_split
from cudnn.frost.tile_dsl.consume import consume_then_arrive, release_zero


class MoEGroupedGemmDgluBf16KernelSm90:
    """BF16 grouped GEMM with a fused dGLU for Hopper; takes the SM100 dGLU kernel's arguments.

    Reads C (gate | up) through a C-load warp and writes D (dgate | dup) and dprob. WGMMA accumulates
    in the MMA warp groups' registers, so they also run the epilogue (no TMEM).

    :param mma_tiler_mn: CTA tile shape; M is 128 (two MMA warp groups), N 128 or 256.
    :param act_func: "dswiglu" or "dgeglu".
    :param round_dgrad_to_input_dtype: Round the scaled dgrad to the A/B dtype before the activation.
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
        act_func: str = "dswiglu",
    ) -> bool:
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        fix_pad_size = MoEGroupedGemmDgluBf16KernelSm90.FIX_PAD_SIZE
        return (
            ab_dtype == cutlass.BFloat16
            and c_dtype in (cutlass.BFloat16, cutlass.Float16)
            and d_dtype in (cutlass.BFloat16, cutlass.Float16)
            and acc_dtype == cutlass.Float32
            and not use_2cta_instrs
            and mma_tiler_mn[0] == 128
            and mma_tiler_mn[1] in (128, 256)
            and is_power_of_2(cluster_shape_mn[0])
            and is_power_of_2(cluster_shape_mn[1])
            and cluster_shape_mn[0] * cluster_shape_mn[1] <= 8
            and m_aligned == fix_pad_size
            and fix_pad_size % (mma_tiler_mn[0] * cluster_shape_mn[0]) == 0
            and a_major == "k"
            and b_major in ("k", "n")
            and cd_major == "n"
            and act_func in ("dswiglu", "dgeglu")
            and n % 32 == 0
            and k % 8 == 0
            and m % 256 == 0
        )

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        expert_cnt: int,
        weight_mode: MoEWeightMode = MoEWeightMode.DENSE,
        use_dynamic_sched: bool = True,
        act_func: str = "dswiglu",
        geglu_alpha: float = 1.702,
        glu_clamp_max: float = 7.0,
        glu_clamp_min: float = -7.0,
        round_dgrad_to_input_dtype: bool = False,
        glu_layout: str = "interleaved_32",
        deterministic: bool = False,
    ):
        mma_tile_m = mma_tiler_mn[0]
        if self.FIX_PAD_SIZE % (mma_tile_m * cluster_shape_mn[0]) != 0:
            # A cluster M tile past one padded expert segment would write the next expert's rows.
            raise ValueError(
                f"FIX_PAD_SIZE ({self.FIX_PAD_SIZE}) must be divisible by mma_tiler_mn[0] * cluster_shape_mn[0] ({mma_tile_m} * {cluster_shape_mn[0]})."
            )
        if expert_cnt > 1024:
            raise ValueError("Expert count > 1024 is not supported.")
        if not isinstance(weight_mode, MoEWeightMode):
            raise TypeError(f"weight_mode must be a MoEWeightMode, got {type(weight_mode)}")
        if use_2cta_instrs:
            raise ValueError("SM90 has no 2-CTA MMA: use_2cta_instrs must be False.")
        if weight_mode != MoEWeightMode.DENSE or not use_dynamic_sched:
            raise ValueError("SM90 dGLU: dense weights and the dynamic scheduler only.")
        # Two MMA warp groups give each epilogue thread exactly two rows (the dprob row slots).
        if mma_tile_m != 128 or mma_tiler_mn[1] not in (128, 256):
            raise ValueError(f"SM90 dGLU CTA tile must be (128, 128) or (128, 256), got {tuple(mma_tiler_mn)}.")
        if act_func not in ("dswiglu", "dgeglu"):
            raise ValueError(f"act_func must be 'dswiglu' or 'dgeglu', got {act_func!r}")
        validate_glu_layout(glu_layout)
        self.act_func = act_func
        self.geglu_alpha = geglu_alpha
        self.glu_clamp_max = glu_clamp_max
        self.glu_clamp_min = glu_clamp_min
        self.round_dgrad_to_input_dtype = round_dgrad_to_input_dtype
        self.glu_layout = glu_layout
        # gate_then_up only moves C loads and D stores: accumulator column j pairs C/D columns j and N + j.
        self.gate_then_up = glu_layout == "gate_then_up"
        # dprob is (valid_m, n_slots, 1), one slot per N-tile, which the caller reduces in a fixed order.
        self.deterministic = deterministic
        self.weight_mode = weight_mode
        self.use_dynamic_sched = True

        self.expert_cnt = expert_cnt
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = False
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.occupancy = 1

        # WGMMA's M is 64: the CTA tile M of 128 comes from a second MMA warp group.
        self.atom_layout_mnk = (2, 1, 1)
        self.num_mma_warp_groups = self.atom_layout_mnk[0] * self.atom_layout_mnk[1] * self.atom_layout_mnk[2]
        self.num_warps_per_warp_group = 4
        self.threads_per_warp = 32
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * self.threads_per_warp

        # MMA/epilogue warp groups first, then one producer warp group (TMA, scheduler, C load, 1 idle warp).
        self.epilog_warp_id = tuple(range(self.num_mma_warp_groups * self.num_warps_per_warp_group))
        self.epi_store_warp_id = self.epilog_warp_id[0]
        self.tma_warp_id = self.num_mma_warp_groups * self.num_warps_per_warp_group
        self.sched_warp_id = self.tma_warp_id + 1
        self.c_load_warp_id = self.tma_warp_id + 2
        self.num_mma_threads = self.num_mma_warp_groups * self.num_threads_per_warp_group
        self.threads_per_cta = (self.num_mma_warp_groups + 1) * self.num_threads_per_warp_group
        self.threads_wo_sched = self.num_mma_threads + self.threads_per_warp * 2
        # The split must fit the 3 x 168 launch registers (48 + 2 x 232 hung); the producer warps spill at 40.
        self.load_register_requirement = 56
        self.mma_register_requirement = 224
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

        self.num_smem_capacity = get_smem_capacity_in_bytes("sm_90")
        self.vectorized_f32 = vectorized_f32
        self.num_epilog_warps = len(self.epilog_warp_id)

    # ------------------------------------------------------------------
    # _setup_attributes
    # ------------------------------------------------------------------

    def _setup_attributes(self):
        """Configure MMA / tile / stage / SMEM layouts from GEMM inputs."""
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.atom_layout_mnk,
            (64, self.mma_tiler[1]),
            a_source=warpgroup.OperandSource.SMEM,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        # No 2-CTA MMA on Hopper: the CTA tile is the MMA tile and the cluster's "V" mode is 1.
        self.cta_tile_shape_mnk = self.mma_tiler
        # C and D carry gate | up, twice the accumulator's N; A/B and the scheduler keep N.
        self.mma_tiler_d = (self.mma_tiler[0], self.mma_tiler[1] * 2, self.mma_tiler[2])
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
        # The gate/up interleave block is the epilogue N subtile, 32 as in the forward's C and the SM100 dGLU.
        if self.epi_tile != (128, 32):
            raise ValueError(f"SM90 dGLU needs a (128, 32) epilogue tile, got {self.epi_tile}.")

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
            self.num_c_stage,
        )
        self.d_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.d_dtype,
            self.d_layout,
            self.epi_tile,
            self.num_d_stage,
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
    ):
        # gate | up and D1 | D2 move in pairs, so stage counts stay even; at least 2 C pairs and 1 D pair.
        num_c_stage = 4
        num_d_stage = 2
        num_tile_stage = 2

        a_smem_layout_stage_one = sm90_utils.make_smem_layout_a(a_layout, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_stage_one = sm90_utils.make_smem_layout_b(b_layout, mma_tiler_mnk, b_dtype, 1)
        c_smem_layout_stage_one = sm90_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
        d_smem_layout_stage_one = sm90_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)

        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(b_dtype, b_smem_layout_stage_one)
        mbar_helpers_bytes = 1024
        sinfo_bytes = 4 * 4 * num_tile_stage
        c_stage_bytes = cute.size_in_bytes(c_dtype, c_smem_layout_stage_one)
        d_stage_bytes = cute.size_in_bytes(d_dtype, d_smem_layout_stage_one)

        epi_bytes = c_stage_bytes * num_c_stage + d_stage_bytes * num_d_stage
        ab_budget_bytes = num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)
        num_ab_stage = ab_budget_bytes // ab_bytes_per_stage
        spare_bytes = ab_budget_bytes % ab_bytes_per_stage
        # The bytes the A/B stages leave: D2 must not wait for D1's TMA store, then C loads further ahead.
        if spare_bytes >= 2 * d_stage_bytes:
            num_d_stage += 2
            spare_bytes -= 2 * d_stage_bytes
        num_c_stage += 2 * (spare_bytes // (2 * c_stage_bytes))
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
        counter_addr = workspace_ptr.toint() + self.get_desc_workspace_bytes()
        return cute.make_ptr(
            cutlass.Int32,
            counter_addr,
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
        beta: cute.Tensor,
        prob: cute.Tensor,
        dprob: cute.Tensor,
        linear_offset: cutlass.Float32,
        dbias_tensor: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        activation: Optional[cute.Tensor] = None,
    ):
        """Execute the GEMM.

        ``b`` is the dense (N, K, L) weight tensor; ``n``, ``k`` and ``b_stride_size`` are unused, so it takes the
        SM100 dGLU kernel's arguments. ``dbias_tensor`` and ``activation`` must be None on SM90.
        """
        if cutlass.const_expr(dbias_tensor is not None or activation is not None):
            raise NotImplementedError("SM90 dGLU writes neither dbias nor the activation output.")
        a = kernel_facing_mx(a)
        c = kernel_facing_mx(c)
        d = kernel_facing_mx(d)
        prob = kernel_facing_prob(prob)
        dprob = kernel_facing_prob(dprob)
        b = kernel_facing_b(b, b_major_mode == OperandMajorMode.MN)
        for row_tensor in (prob, dprob):  # _expert_row_view's, when the rows are static
            if cutlass.const_expr(isinstance(row_tensor.shape[0], int)):
                self.check_row_views_fit_32_bits(row_tensor.shape[0], cute.size(row_tensor.shape[1]))
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
            raise TypeError(f"SM90 dGLU moves C/D through ldmatrix/stmatrix: they must be 16 bit, got {self.c_dtype}/{self.d_dtype}")

        self._setup_attributes()
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.atom_layout_mnk,
            (64, self.mma_tiler[1]),
            a_source=warpgroup.OperandSource.SMEM,
        )

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
        self.num_tma_c_load_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
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
        if cutlass.const_expr(not self.gate_then_up):
            # gate_then_up addresses C and D by explicit in-mode coordinates (_cd_coord), which do not wrap.
            tma_tensor_c = _widen_for_cluster(tma_tensor_c, 1, self.mma_tiler_d[1], cluster_n)
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
        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            c_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage * 2]
            scheduler: SchedulerStorage
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
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
            beta,
            prob,
            dprob,
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
        """The register-to-SMEM tiled copy of one epilogue tensor: the MMA's C layout truncated to the store atom."""
        copy_atom_c = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(layout.is_m_major_c(), 4),
            dtype,
        )
        tiled_copy_c_atom = cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma)
        copy_atom_r2s = sm90_utils.get_smem_store_op(layout, dtype, self.acc_dtype)
        return cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_c_atom)

    def epilog_s2r_copy(self, tiled_mma, layout, dtype):
        """The inverse of epilog_r2s_copy: ldmatrix on the same C atom tiling, so C register i pairs accumulator
        register i. hopper_helpers has no load op; the forward's generate_c store wrote this SMEM layout."""
        copy_atom_c = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(layout.is_m_major_c(), 4),
            dtype,
        )
        tiled_copy_c_atom = cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma)
        copy_atom_s2r = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(layout.is_m_major_c(), 4), dtype)
        return cute.make_tiled_copy_S(copy_atom_s2r, tiled_copy_c_atom)

    def load_c(self, tiled_copy_s2r, c_pipeline, c_consumer_state, tSR_sC, tSR_rC, rz):
        """Wait for the next C subtile, move it to registers and give its SMEM stage back to the load warp.
        The caller advances c_consumer_state: only a call on the name in the kernel body makes it loop-carried.
        Every epilogue warp loads and releases: its signaling lane (lane 0) arrives, through tile_dsl.consume, on an
        address computed from every C register this lane loaded, so the stage is released only after those values are
        consumed. The fence stays: the refill is a TMA load."""
        c_pipeline.consumer_wait(c_consumer_state)
        cute.copy(tiled_copy_s2r, tSR_sC[(None, None, None, c_consumer_state.index)], tSR_rC)
        cute.arch.fence_proxy("async.shared", space="cta")
        consume_then_arrive(
            c_pipeline.consumer_get_barrier(c_consumer_state),
            cute.recast_tensor(tSR_rC, cutlass.Int32).load(),
            rz,
            pred=c_pipeline.is_signaling_thread,
        )

    @cute.jit
    def store_d(
        self,
        tiled_copy_r2s,
        tma_atom_d,
        warp_idx,
        tRS_rD,
        tRS_rD_out,
        tRS_sD,
        bSG_gD,
        bSG_sD,
        d_pipeline,
        d_buffer,
        gmem_coord,
        gate_then_up_coord=None,
    ):
        """``gate_then_up_coord``: _cd_coord's arguments, evaluated by the store warp only (no registers held across the
        activation math)."""
        tRS_rD_out.store(tRS_rD.load().to(self.d_dtype))
        cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, d_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        self.epilog_sync_barrier.arrive_and_wait()
        if warp_idx == self.epi_store_warp_id:
            if cutlass.const_expr(gate_then_up_coord is not None):
                cute.copy(tma_atom_d, bSG_sD[(None, d_buffer)], bSG_gD[(None, self._cd_coord(*gate_then_up_coord))])
            else:
                cute.copy(tma_atom_d, bSG_sD[(None, d_buffer)], bSG_gD[(None, gmem_coord)])
            d_pipeline.producer_commit()
            d_pipeline.producer_acquire()
        self.epilog_sync_barrier.arrive_and_wait()

    def _gate_then_up_epi(self, real, epi_tile, work_tile_info):
        """gate_then_up: C or D (gate | up, 2 N columns) from the tile's gate origin (row tile_m * 128, column
        tile_n * N_tile) in epilogue tiles, the branch's N in 32-column blocks, and the tile's gate blocks inside N."""
        branch_blocks = cute.size(real, mode=[1]) // (2 * epi_tile[1])
        first_block = work_tile_info.tile_n_idx * (self.mma_tiler[1] // epi_tile[1])
        origin = (work_tile_info.tile_m_idx * self.mma_tiler[0], first_block * epi_tile[1], 0)
        return cute.zipped_divide(cute.domain_offset(origin, real), epi_tile), (branch_blocks - first_block, branch_blocks)

    @staticmethod
    def _cd_coord(blocks, j, branch):
        """gate_then_up: the epilogue tile of accumulator subtile j's ``branch`` (0 gate, 1 up) from the tile's gate origin:
        block j, or N/32 + j for up. A gate block past N (a partial N tile) moves out of bounds: its TMA load fills
        zeros, as interleaved_32's does past 2 N, and its TMA store is dropped."""
        gate_blocks_in_n, branch_blocks = blocks
        if cutlass.const_expr(branch == 0):
            return (0, j + branch_blocks * cutlass.Int32(j >= gate_blocks_in_n), 0)
        return (0, j + branch_blocks, 0)

    @staticmethod
    def check_row_views_fit_32_bits(rows, *row_elements):
        """Host check for _expert_row_view: ``rows`` times each per-row tensor's elements per row stays below 2^31."""
        for elements in row_elements:
            if rows * elements >= 2**31:
                raise ValueError(f"{rows} rows x {elements} elements per row pass 2^31: _expert_row_view's 32-bit offset would wrap")

    @staticmethod
    def _expert_row_view(ext, tensor, padded_offsets, work_tile_info):
        """A per-row ``tensor`` (prob, dprob) at its expert's first row by a 32-bit offset, fewer registers than the
        64-bit one.  Needs rows x n < 2^31 (check_row_views_fit_32_bits)."""
        if cutlass.const_expr(hasattr(ext, "token_offset")):  # update_expert_info's, this tile's expert
            token_offset, tokens_i = ext.token_offset, ext.tokens_i
        else:
            token_offset, tokens_i = compute_expert_token_range(padded_offsets, work_tile_info.expert_idx)
        real = cute.domain_offset((token_offset, 0, 0), tensor)
        return rewrite_tensor_shape(real, (tokens_i, tensor.shape[1], cutlass.Int32(1)))

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
        beta: cute.Tensor,
        prob: cute.Tensor,
        dprob: cute.Tensor,
        linear_offset: cutlass.Float32,
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
            cpasync.prefetch_descriptor(tma_atom_c)

        total_token = padded_offsets[self.expert_cnt - 1]

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sched_storage = storage.scheduler

        # ---- Pipeline setup ----
        # Hopper consumers are whole warp groups: one lane per warp per multicast partner signals the empty barrier.
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_consumer_arrive_cnt = mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
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

        # C load: one lane of each epilogue warp signals a stage empty. tile_info_pipeline's create fences both inits.
        c_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epilog_warps)
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_mbar_ptr.data_ptr(),
            num_stages=self.num_c_stage,
            producer_group=c_pipeline_producer_group,
            consumer_group=c_pipeline_consumer_group,
            tx_count=self.num_tma_c_load_bytes,
            tidx=tidx,
            defer_sync=True,
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

        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = sched_storage.sInfo.get_tensor(info_layout)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        # Multicast masks must be created together when any mcast is active.
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

        # SMEM fragments for WGMMA plus the register accumulator shared by MMA and epilogue.
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups,
            stride=self.num_threads_per_warp_group,
        )
        thr_mma = tiled_mma.get_slice(mma_warp_group_thread_layout(warp_group_idx))
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
        # Shape the accumulator from the MMA tile, not from D: D has twice its N columns.
        acc_tile_shape = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        acc_shape = thr_mma.partition_C(acc_tile_shape).shape[:3]
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

        # ---- Scheduler warp (MoE Persistent Tile Scheduler) ----
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

        # ---- C load warp ----
        if warp_idx == self.c_load_warp_id:
            c_ext = self._make_extension(workspace_ptr)
            c_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_c_stage)
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
                c_ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)
                real_c, _ = c_ext.get_gmem_tensor("c", mC_mnl, padded_offsets, work_tile_info)
                if cutlass.const_expr(self.gate_then_up):
                    gC_epi, c_blocks = self._gate_then_up_epi(real_c, epi_tile, work_tile_info)
                else:
                    gC_mnl_loop = cute.local_tile(real_c, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                    gC_tile = gC_mnl_loop[(None, None, work_tile_info.tile_m_idx, work_tile_info.tile_n_idx, 0)]
                    gC_epi = cute.zipped_divide(gC_tile, epi_tile)
                bGS_sC, bGS_gC = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    gC_epi,
                )
                if cutlass.const_expr(self.gate_then_up):
                    c_subtile_cnt = self.mma_tiler_d[1] // epi_tile[1]
                else:
                    c_subtile_shape = gC_epi.shape[1]
                    c_subtile_layout = cute.make_layout(c_subtile_shape, stride=(c_subtile_shape[1], 1))
                    c_subtile_cnt = cute.size(gC_epi, mode=[1])

                for subtile_idx in cutlass.range_constexpr(c_subtile_cnt):
                    if cutlass.const_expr(self.gate_then_up):
                        c_coord = self._cd_coord(c_blocks, subtile_idx // 2, subtile_idx % 2)
                    else:
                        c_coord = c_subtile_layout.get_hier_coord(subtile_idx)
                    c_pipeline.producer_acquire(c_producer_state)
                    cute.copy(
                        tma_atom_c,
                        bGS_gC[(None, c_coord)],
                        bGS_sC[(None, c_producer_state.index)],
                        tma_bar_ptr=c_pipeline.producer_get_barrier(c_producer_state),
                    )
                    c_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                consume_then_arrive(
                    tile_info_pipeline.consumer_get_barrier(tile_info_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz
                )
                tile_info_consumer_state.advance()
            c_pipeline.producer_tail(c_producer_state)

        # ---- DMA / TMA load warp ----
        if warp_idx == self.tma_warp_id:
            ext = self._make_extension(workspace_ptr)
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

                mma_tile_coord_m = work_tile_info.tile_m_idx
                mma_tile_coord_n = work_tile_info.tile_n_idx
                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_n, None, 0)]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
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

        # ---- MMA + epilogue warp group(s) ----
        # WGMMA accumulates in these threads' registers: mainloop and epilogue share them.
        if warp_idx < self.tma_warp_id and total_token > 0:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            epi_tidx = tidx
            num_k_blocks = cute.size(tCrA, mode=[2])
            k_pipe_mmas = 1

            tiled_copy_r2s = self.epilog_r2s_copy(tiled_mma, self.d_layout, self.d_dtype)
            thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
            tRS_sD = thr_copy_r2s.partition_D(sD)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sD))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD1 = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD2 = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.d_dtype)
            size_tRS_rD = cute.size(tRS_rD1)
            # Per-element (row, col) of the accumulator in the CTA tile; slice PER THREAD (thr_mma is per warp group).
            cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
            tRS_cAcc = tiled_copy_r2s.retile(tiled_mma.get_slice(epi_tidx).partition_C(cAcc))
            # A thread owns 2 rows, picked by bit 1 of its element index: element 0 and element 2 name them.
            tRS_rProb = cute.make_rmem_tensor((2,), cutlass.Float32)
            tRS_rDprob = cute.make_rmem_tensor((2,), cutlass.Float32)

            tiled_copy_s2r = self.epilog_s2r_copy(tiled_mma, self.c_layout, self.c_dtype)
            tSR_sC = tiled_copy_s2r.get_slice(epi_tidx).partition_S(sC)
            tSR_rGate = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            tSR_rUp = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)

            epi_ext = self._make_extension(workspace_ptr)
            ab_consumer_read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            ab_consumer_release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            c_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_c_stage)
            d_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads)
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
                beta_val = cutlass.Float32(beta[expert_idx])
                square_alpha = alpha_val * alpha_val
                if cutlass.const_expr(self.act_func == "dgeglu"):
                    geglu_alpha = cutlass.Float32(self.geglu_alpha)
                    clamp_max = cutlass.Float32(self.glu_clamp_max)
                    clamp_min = cutlass.Float32(self.glu_clamp_min)

                # ---- mainloop: one accumulator init + one wgmma fence per output tile ----
                ab_consumer_read_state.reset_count()
                ab_consumer_release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.fence()

                prologue_mma_cnt = cutlass.min(k_pipe_mmas, k_tile_cnt)
                for k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                    ab_pipeline.consumer_wait(ab_consumer_read_state)
                    for kblock_idx in cutlass.range_constexpr(num_k_blocks):
                        kblock_coord = (None, None, kblock_idx, ab_consumer_read_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[kblock_coord], tCrB[kblock_coord], accumulators)
                    warpgroup.commit_group()
                    ab_consumer_read_state.advance()

                for k_tile in cutlass.range(prologue_mma_cnt, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.consumer_wait(ab_consumer_read_state)
                    for kblock_idx in cutlass.range_constexpr(num_k_blocks):
                        kblock_coord = (None, None, kblock_idx, ab_consumer_read_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[kblock_coord], tCrB[kblock_coord], accumulators)
                    warpgroup.commit_group()
                    warpgroup.wait_group(k_pipe_mmas)
                    ab_pipeline.consumer_release(ab_consumer_release_state)
                    ab_consumer_release_state.advance()
                    ab_consumer_read_state.advance()

                warpgroup.wait_group(0)
                for k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                    ab_pipeline.consumer_release(ab_consumer_release_state)
                    ab_consumer_release_state.advance()

                # ---- epilogue ----
                real_d, _ = epi_ext.get_gmem_tensor("d", mD_mnl, padded_offsets, epi_work_tile_info)
                real_prob = self._expert_row_view(epi_ext, prob, padded_offsets, epi_work_tile_info)
                real_dprob = self._expert_row_view(epi_ext, dprob, padded_offsets, epi_work_tile_info)
                m_base = epi_work_tile_info.tile_m_idx * self.cta_tile_shape_mnk[0]

                if cutlass.const_expr(self.gate_then_up):
                    gD_epi, d_blocks = self._gate_then_up_epi(real_d, epi_tile, epi_work_tile_info)
                else:
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

                if cutlass.const_expr(self.gate_then_up):
                    subtile_cnt = self.mma_tiler_d[1] // epi_tile[1]
                else:
                    subtile_cnt = cute.size(gD_epi, mode=[1])
                    subtile_shape = gD_epi.shape[1]
                    subtile_layout = cute.make_layout(subtile_shape, stride=(subtile_shape[1], 1))

                for row_slot in cutlass.range_constexpr(2):
                    tRS_rProb[row_slot] = real_prob[m_base + tRS_cAcc[2 * row_slot][0], 0, 0]
                    tRS_rDprob[row_slot] = cutlass.Float32(0.0)

                # Accumulator subtile s pairs C and D subtiles 2 s (gate) and 2 s + 1 (up): same rows, 32 columns apart.
                for pair_idx in cutlass.range_constexpr(subtile_cnt // 2):
                    self.load_c(tiled_copy_s2r, c_pipeline, c_consumer_state, tSR_sC, tSR_rGate, rz)
                    c_consumer_state.advance()
                    self.load_c(tiled_copy_s2r, c_pipeline, c_consumer_state, tSR_sC, tSR_rUp, rz)
                    c_consumer_state.advance()
                    acc_base = pair_idx * size_tRS_rD
                    # Hopper has no packed f32x2 math (SM100+), so vectorized_f32 is accepted for API parity only.
                    for i in cutlass.range_constexpr(size_tRS_rD):
                        row_slot = (i >> 1) & 1
                        dgrad_value = tRS_rAcc[acc_base + i] * square_alpha
                        if cutlass.const_expr(self.round_dgrad_to_input_dtype):
                            dgrad_value = dgrad_value.to(self.a_dtype).to(cutlass.Float32)
                        gate_value = tSR_rGate[i].to(cutlass.Float32) * beta_val
                        up_value = tSR_rUp[i].to(cutlass.Float32) * beta_val
                        if cutlass.const_expr(self.act_func == "dgeglu"):
                            # y1 = min(x1, max), y2 = clamp(x2); the clamps pass no gradient outside the bounds.
                            gate_clamped = fmin(gate_value, clamp_max)
                            gate_scaled = gate_clamped * geglu_alpha
                            sigmoid_value = sigmoid_f32(gate_scaled, fastmath=True)
                            gate_act = gate_clamped * sigmoid_value
                            up_shifted = fmax(fmin(up_value, clamp_max), clamp_min) + linear_offset
                            tRS_rDprob[row_slot] = tRS_rDprob[row_slot] + gate_act * up_shifted * dgrad_value
                            dgrad_value = dgrad_value * tRS_rProb[row_slot]
                            gate_filter = 1.0 if gate_value <= clamp_max else 0.0
                            up_filter = 1.0 if clamp_min <= up_value <= clamp_max else 0.0
                            tRS_rD1[i] = gate_filter * dgrad_value * sigmoid_value * (1.0 + gate_scaled * (1.0 - sigmoid_value)) * up_shifted
                            tRS_rD2[i] = up_filter * dgrad_value * gate_act
                        else:
                            sigmoid_value = sigmoid_f32(gate_value, fastmath=True)
                            swish_value = gate_value * sigmoid_value
                            tRS_rDprob[row_slot] = tRS_rDprob[row_slot] + dgrad_value * (up_value * swish_value)
                            dgrad_value = dgrad_value * tRS_rProb[row_slot]
                            tRS_rD1[i] = dgrad_value * up_value * sigmoid_value * (1.0 + gate_value * (1.0 - sigmoid_value))
                            tRS_rD2[i] = dgrad_value * swish_value

                    self.store_d(
                        tiled_copy_r2s,
                        tma_atom_d,
                        warp_idx,
                        tRS_rD1,
                        tRS_rD_out,
                        tRS_sD,
                        bSG_gD,
                        bSG_sD,
                        d_pipeline,
                        num_prev_subtiles % self.num_d_stage,
                        None if cutlass.const_expr(self.gate_then_up) else subtile_layout.get_hier_coord(2 * pair_idx),
                        (d_blocks, pair_idx, 0) if cutlass.const_expr(self.gate_then_up) else None,
                    )
                    num_prev_subtiles = num_prev_subtiles + 1
                    self.store_d(
                        tiled_copy_r2s,
                        tma_atom_d,
                        warp_idx,
                        tRS_rD2,
                        tRS_rD_out,
                        tRS_sD,
                        bSG_gD,
                        bSG_sD,
                        d_pipeline,
                        num_prev_subtiles % self.num_d_stage,
                        None if cutlass.const_expr(self.gate_then_up) else subtile_layout.get_hier_coord(2 * pair_idx + 1),
                        (d_blocks, pair_idx, 1) if cutlass.const_expr(self.gate_then_up) else None,
                    )
                    num_prev_subtiles = num_prev_subtiles + 1

                # A row spans several N tiles, maybe on other CTAs, so its dprob is an atomic sum.
                for row_slot in cutlass.range_constexpr(2):
                    if cutlass.const_expr(self.deterministic):
                        # One writer per (token, tile_n) slot; the atomic is uncontended, kept for one store path.
                        dprob_slot = real_dprob[(m_base + tRS_cAcc[2 * row_slot][0], epi_work_tile_info.tile_n_idx, None)]
                    else:
                        dprob_slot = real_dprob[(m_base + tRS_cAcc[2 * row_slot][0], None, None)]
                    atomic_add_float32(ptr=dprob_slot.iterator.llvm_ptr, value=tRS_rDprob[row_slot])

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                consume_then_arrive(
                    tile_info_pipeline.consumer_get_barrier(tile_info_consumer_state), [tile_info[0], tile_info[1], tile_info[2], tile_info[3]], rz
                )
                tile_info_consumer_state.advance()

            d_pipeline.producer_tail()

        # Cluster-exit barrier from ALL threads: a non-leader CTA's last claim arrives on the leader's mbarrier, which
        # could already be gone ("Cluster target block not present"). SM100 gets this from its TMEM dealloc handshake.
        if cutlass.const_expr(self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    # ------------------------------------------------------------------
    # Internal: the scheduler extension (dense weights)
    # ------------------------------------------------------------------

    @cute.jit
    def _make_extension(self, workspace_ptr):
        return ContiguousGroupedGemmSchedExtension()
