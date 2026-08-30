# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Fused fc1+fc2 GLU MXFP8 MegaMoE kernel for SM100.
"""

import dataclasses
from typing import Any, Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.nvgpu.tcgen05 import CollectorOp

from ..tmem_transpose import _TmemTranspose16x32Core
from .dglu_mxfp8_fc12_epilogue import DgluMxfp8Epilogue
from .....schedulers import BlockPhase
from .....schedulers.base import WorkIdAcquisitionMode
from .....schedulers.fc12_scheduler import BlackwellFusedFc12Scheduler
from .dglu_mxfp8_fc12_extension import DgluMxFp8Fc12SchedExtension
from ......api import ImplDesc, KernelClass, ProblemDesc, StaticOrRuntimeIntegerType
from ..helpers.constants import (
    SupportedMmaTileM,
    SupportedMmaTileN,
)
from ......helpers.iket_compat import iket
from ......helpers.device_workspace import DeviceWorkspace
from ......helpers.smem_workspace import SmemWorkspace
from ......helpers.dsl_helpers import spin_wait
from ......quant_def import CombineFormat, QuantKind
from ......communication.nvlink_domain.token_comm import TokenCommArgs


@dataclasses.dataclass(frozen=True)
class _EpilogueCommView:
    """Fields the dGLU epilogue reads for cross-rank dfc1/dprob routing."""
    token_src_metadata: Any
    combine_output: Any
    dprob_output: Any
    peer_rank_ptr_mapper: Any
    fc2_output_sf: Any = None
    fc2_done_counter: Any = None
    fc2_output_workspace: Any = None


# =============================================================================
# Sm107Mxfp8DgluDfc21Kernel
# =============================================================================

class Sm107Mxfp8DgluDfc21Kernel:

    # SMEM budget for buffers like mbarriers, sched, work-tile buffer, TMEM allocator state
    _SmemMiscBudget = 1024

    # Supported (ab_dtype, sf_vec_size) pairings.
    # MXFP8 → Float8E4M3FN / Float8E5M2 + sf_vec_size=32  (FP8-E8M0 scales, MmaMXF8Op)
    VALID_AB_DTYPE_SF_SIZE: dict = {
        32: (cutlass.Float8E4M3FN, cutlass.Float8E5M2,),
    }

    # Interleave granularity for gate and up in SwiGLU / GeGlu
    GateUpInterleave: int = 32

    def __init__(
        self,
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        load_balance_mode: Literal["static", "atomic_counter"] = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_vec_size: int = 32,
        ab_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        dfc2_recompute: bool = False,
        dfc2_col_output: bool = False,
        fc2_in_kernel_topk_reduce: bool = False,
        act_func: str = "swiglu",
        gate_up_clamp: Optional[float] = None,
    ) -> None:
        if not force_static_sched:
            raise NotImplementedError(
                "v1 only implements force_static_sched=True (lean 7-warp). "
                "Dynamic CLC (force_static_sched=False) is future work."
            )

        # Validate (ab_dtype, sf_vec_size) pairing.
        if sf_vec_size in self.VALID_AB_DTYPE_SF_SIZE:
            valid_ab = self.VALID_AB_DTYPE_SF_SIZE[sf_vec_size]
            if ab_dtype not in valid_ab:
                raise ValueError(
                    f"ab_dtype={ab_dtype.__name__} is not valid for "
                    f"sf_vec_size={sf_vec_size}. "
                    f"Expected one of: {[t.__name__ for t in valid_ab]}."
                )
        else:
            valid_sf_vec_sizes = tuple(self.VALID_AB_DTYPE_SF_SIZE)
            raise NotImplementedError(
                f"sf_vec_size must be one of {valid_sf_vec_sizes} (MXFP8); got {sf_vec_size}."
            )


        if load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"load_balance_mode must be 'static' or 'atomic_counter'; "
                f"got {load_balance_mode!r}."
            )
        if act_func not in ("swiglu", "geglu"):
            raise ValueError(
                f"act_func must be 'swiglu' or 'geglu'; got {act_func!r}."
            )
        if act_func != "swiglu":
            raise NotImplementedError(
                f"act_func={act_func!r} is not yet implemented; only "
                "'swiglu' is currently supported (geglu support is planned)."
            )

        # Store ab_dtype so workspace-size helpers can use it without tensors.
        self.ab_dtype = ab_dtype
        self.act_func = act_func

        self.acc_dtype = acc_dtype
        self.mma_tiler_mnk = mma_tiler_mnk
        self.cluster_shape_mn = (cluster_shape_mnk[0], cluster_shape_mnk[1])
        self.use_2cta_instrs = use_2cta_instrs
        self.force_static_sched = force_static_sched
        # static_expert_shape / clc_bundle_size / num_sched_stages
        self.static_expert_shape = static_expert_shape
        self.clc_bundle_size = clc_bundle_size
        self.num_sched_stages = num_sched_stages

        # Fused fc12 sched-side knobs
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.load_balance_mode = load_balance_mode

        self.sf_vec_size = sf_vec_size
        self.arch = "sm_107"
        self.epi_flag_batch = epi_flag_batch
        self.dfc2_recompute = dfc2_recompute
        self.dfc2_col_output = dfc2_col_output
        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.gate_up_clamp = abs(gate_up_clamp) if gate_up_clamp is not None else None

        self._validate_mma_tiler_and_cluster_shape()
        self.mma_tiler = mma_tiler_mnk

        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        # Warp specialization (9-warp / 288 thread: + dedicated preact-C load warp)
        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_a_warp_id = 5
        self.tma_b_warp_id = 6
        self.sched_warp_id = 7
        # Dedicated TMA-load warp for the forward pre-activation (dswiglu C),
        self.c_load_warp_id = 8
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_a_warp_id,
                self.tma_b_warp_id,
                self.sched_warp_id,
                self.c_load_warp_id,
                *self.epilogue_warp_id,
            )
        )

        # NamedBarriers.
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.epi_subtile_bar_ids = (4, 5, 6, 7)

        self.smem_capacity = utils.get_smem_capacity_in_bytes()
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(
            self.arch
        )

        # Warp-specialized register split.
        self.epi_reg_cnt = 256
        self.task_reg_cnt = 72

        # Token-comm (MegaMoE)
        self.enable_token_comm: bool = False
        self.dispatch_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_by_dispatch: bool = False
        self.token_back_standalone: bool = False
        self.token_back_warp_id: Optional[Tuple[int, int, int, int]] = None

    def _validate_mma_tiler_and_cluster_shape(self) -> None:
        """Validate user-provided geometry against v1 fused-fc12 constraints."""
        m, n, k = self.mma_tiler_mnk
        cm, cn = self.cluster_shape_mn

        if m not in SupportedMmaTileM:
            raise ValueError(
                f"mma_tiler M ({m}) must be one of {SupportedMmaTileM}"
            )

        per_cta_m = m // (2 if self.use_2cta_instrs else 1)
        if per_cta_m != 128:
            raise ValueError(
                f"per-CTA mma_tiler M must be 128, got {per_cta_m} "
                f"(mma_tiler_m={m}, use_2cta_instrs={self.use_2cta_instrs})"
            )

        for _name, _blk in (
            ("token_padding_block", self.token_padding_block),
            ("sf_padding_block", self.sf_padding_block),
        ):
            if _blk <= 0 or _blk % self.sf_vec_size != 0:
                raise ValueError(
                    f"{_name} ({_blk}) must be a positive multiple of "
                    f"sf_vec_size ({self.sf_vec_size}); the col-quant epilogue "
                    f"turns a per-expert row offset into a col-SF row-block "
                    f"index by an exact '// sf_vec_size' division."
                )

        if n not in SupportedMmaTileN:
            raise ValueError(
                f"mma_tiler N ({n}) must be one of {SupportedMmaTileN} in fused fc12 "
                f"(N=64 SFB hack is dropped; swap-AB sched handles short-N "
                f"via subtile early-exit)."
            )

        sf_k_granularity = self.sf_vec_size * 4
        if k % sf_k_granularity != 0:
            raise ValueError(
                f"mma_tiler K ({k}) must be a multiple of "
                f"sf_vec_size * 4 = {sf_k_granularity}"
            )

        if cm % (2 if self.use_2cta_instrs else 1) != 0:
            raise ValueError(
                f"cluster_shape M ({cm}) must be even when use_2cta_instrs=True"
            )

        is_pow2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if cm * cn > 16 or not is_pow2(cm) or not is_pow2(cn) or cm > 4 or cn > 4:
            raise ValueError(
                f"Invalid cluster_shape ({cm}, {cn}): each dim must be "
                f"a power of 2 and <= 4, product must be <= 16"
            )

        # v1 swap-AB requires cluster_n == 1.
        if cn != 1:
            raise NotImplementedError(
                f"v1 fused fc12 requires cluster_n == 1 (got {cn}).  "
                f"cluster_n > 1 needs sentinel-style acc/ab pipeline release."
            )

    def _create_tiled_mmas(self) -> Tuple[cute.TiledMma, cute.TiledMma]:
        """Return (tiled_mma, tiled_mma_sfb)."""
        common = (
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
        )
        # Rubin: the SM107 blockscaled FP8 MMA op hard-requires instruction
        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            *common, self.cta_group,
            (*self.mma_inst_shape_mn, 64),
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            *common, tcgen05.CtaGroup.ONE,
            (*self.mma_inst_shape_mn_sfb, 64),
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        return tiled_mma, tiled_mma_sfb

    def _build_scheduler(
        self, *, expert_cnt, intermediate_gateup, hidden_dim, launch_cluster_count
    ) -> None:
        """Construct FC12 scheduler and its SMEM/device workspaces."""
        work_id_mode = "grid_stride" if self.load_balance_mode == "static" else "atomic_counter"
        num_scheduler_consumer_threads = 32 * (len(self.epilogue_warp_id) + 4)
        if self.static_expert_shape is not None:
            expert_cnt, intermediate_gateup, hidden_dim = self.static_expert_shape
        problem_desc = ProblemDesc(
            {
                "expert_count": expert_cnt,
                "intermediate_gateup_size": intermediate_gateup,
                "hidden_size": hidden_dim,
            }
        )
        impl_desc = ImplDesc(
            {
                "num_scheduler_consumer_threads": num_scheduler_consumer_threads,
                "mma_tiler_mnk": self.mma_tiler,
                "cluster_shape_mn": self.cluster_shape_mn,
                "use_2cta_instrs": self.use_2cta_instrs,
                "hint": self.group_hint,
                "token_padding_block": self.token_padding_block,
                "sf_padding_block": self.sf_padding_block,
                "work_id_mode": work_id_mode,
                "is_swap_ab": False,
                "launch_cluster_count": launch_cluster_count,
            }
        )
        self.scheduler = BlackwellFusedFc12Scheduler(problem_desc, impl_desc)

        sched_smem_ws = SmemWorkspace()
        self.scheduler.register_smem_regions(sched_smem_ws)
        sched_smem_ws.finalize(max_bytes=self.smem_capacity)
        self.sched_smem_ws = sched_smem_ws

        sched_device_ws = DeviceWorkspace()
        self.scheduler.register_device_workspace(sched_device_ws)
        sched_device_ws.finalize()
        self.sched_device_ws = sched_device_ws

    def _setup_attributes(self) -> None:
        """Set up MMA / cluster / tile shapes, SMEM layouts, stage counts.

        The fc12 path shares ``mma_tiler_mnk`` and SMEM layouts across phases.
        """
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma, tiled_mma_sfb = self._create_tiled_mmas()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        assert self.mma_tiler[2] % mma_inst_shape_k == 0, (
            f"mma_tiler K ({self.mma_tiler[2]}) must be a multiple of "
            f"MMA instruction K ({mma_inst_shape_k})"
        )

        # SFB-specific tiler: rounded-up MN; same K as main tiler.
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Multicast CTA counts
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        _epi_common = dict(
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            use_2cta_instrs=self.use_2cta_instrs,
            sf_vec_size=self.sf_vec_size,
            fc1_output_dtype=self.fc1_output_dtype,
            fc1_output_layout=self.fc1_output_layout,
            acc_dtype=self.acc_dtype,
            epilog_sync_bar_id=self.epilog_sync_bar_id,
            epilogue_warp_ids=self.epilogue_warp_id,
            static_expert_shape=self.static_expert_shape,
            epi_flag_batch=self.epi_flag_batch,
            token_back_by_dispatch=self.token_back_by_dispatch,
            dfc2_recompute=self.dfc2_recompute,
            dfc2_col_output=self.dfc2_col_output,
            fc2_in_kernel_topk_reduce=self.fc2_in_kernel_topk_reduce,
            combine_format=getattr(self, "combine_format", None),
            combine_hidden=getattr(self, "hidden", None),
            act_func=self.act_func,
            gate_up_clamp=self.gate_up_clamp,
        )
        self.epilogue = DgluMxfp8Epilogue(**_epi_common)

        if self.num_sched_stages is None:
            self.num_sched_stages = 2

        # Reserve SMEM for the preact (dswiglu C) pipeline staging buffer
        self.num_c_stage = 2
        assert self.num_c_stage % 2 == 0, f"num_c_stage must be even, got {self.num_c_stage}"
        self.num_c_pipe_stage = self.num_c_stage // 2
        # One PipelineTmaStore stage contains every dFC2 data output tile.
        self.num_d_stage = self.epilogue.d_output_slots
        c_bytes_total = self.num_c_stage * self.epilogue.preact_bytes_per_stage
        d_bytes_total = self.num_d_stage * self.epilogue.d_bytes_per_stage
        self.c_bytes_total = c_bytes_total
        self.d_bytes_total = d_bytes_total

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_sched_stages,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            self.c_bytes_total,
            self.d_bytes_total,
            self.smem_capacity,
            self.occupancy,
            self.num_sched_stages,
            self._smem_misc_budget_bytes() - self._SmemMiscBudget,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        # Read epilogue's accumulator and scale-factor sizing decisions.
        self.num_acc_pipeline_stages = self.epilogue.num_acc_pipeline_stages
        self.num_acc_stage = self.epilogue.num_acc_stage
        self.num_sfa_tmem_cols = self.epilogue.num_sfa_tmem_cols
        self.num_sfb_tmem_cols = self.epilogue.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.epilogue.num_accumulator_tmem_cols

        # TMA load bytes per stage (A + B + SFA + SFB).
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        self.atom_thr_size = atom_thr_size  # store as Python int for use in @cute.kernel
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # SMEM usage report (all sizes are per-CTA)
        _ab_per_stage = a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        _misc_total = self._smem_misc_budget_bytes()
        _fixed = _misc_total + self.c_bytes_total + self.d_bytes_total
        _total_used = _fixed + self.num_ab_stage * _ab_per_stage
        _per_cta_budget = self.smem_capacity // self.occupancy
        _free = _per_cta_budget - _total_used
        _extra_misc = _misc_total - self._SmemMiscBudget
        print(
            f"[smem] capacity={self.smem_capacity}B ({self.smem_capacity//1024}KB)"
            f"  occupancy={self.occupancy}"
            f"  per-CTA budget={_per_cta_budget}B ({_per_cta_budget//1024}KB)\n"
            f"  AB stages: {self.num_ab_stage} × {_ab_per_stage}B ({_ab_per_stage/1024:.1f}KB)"
            f" = {self.num_ab_stage * _ab_per_stage}B"
            f"  [A={a_copy_size}B B={b_copy_size}B"
            f" SFA={sfa_copy_size}B SFB={sfb_copy_size}B]\n"
            f"  fixed: misc={_misc_total}B (base={self._SmemMiscBudget}B"
            f" + subclass_extra={_extra_misc}B)"
            f"  preact(C)={self.num_c_stage}×{self.epilogue.preact_bytes_per_stage}B"
            f"  sD(D)={self.num_d_stage}×{self.epilogue.d_bytes_per_stage}B"
            f"  used={_total_used}B ({_total_used/1024:.1f}KB)"
            f"  free={_free}B ({_free/1024:.1f}KB)\n"
        )

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_bytes_total: int,
        d_bytes_total: int,
        smem_capacity: int,
        occupancy: int,
        num_sched_stages: int,
        extra_misc_bytes: int = 0,
    ) -> Tuple[int, int, int]:
        """Compute stage counts for ACC, AB+SF, and scheduler.
        """
        num_acc_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1,
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1,
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )

        fixed_overhead = (
            Sm107Mxfp8DgluDfc21Kernel._SmemMiscBudget + extra_misc_bytes + c_bytes_total + d_bytes_total
        )

        num_ab_stage = (
            smem_capacity // occupancy - fixed_overhead
        ) // ab_bytes_per_stage
        return num_acc_stage, num_ab_stage, num_sched_stages

    def get_workspace_size_in_bytes(
        self,
        fc1_activation_tensor,
        fc1_weight_tensor,
    ) -> int:
        """Compute opaque workspace size for one fused dfc2+dfc1 launch."""
        sf_padding_block = self.sf_padding_block
        sf_vec_size = self.sf_vec_size

        mma_tiler_n = self.mma_tiler_mnk[1]

        data_total_rows, _hidden = fc1_activation_tensor.shape
        experts, _hidden_w, dfc2_weight_n = fc1_weight_tensor.shape
        # grad_y1 (doubled dswiglu output) width = intermediate = 2 * inter_half.
        intermediate_out = dfc2_weight_n * 2

        # Conservative upper bound for sf_total_rows.
        sf_total_rows_upper = data_total_rows + experts * sf_padding_block

        # grad_y1 byte size (MXFP8, 8-bit: 1 element per byte).
        fc1_output_bytes = (
            data_total_rows * intermediate_out * self.ab_dtype.width // 8
        )

        # grad_y1 SF sf_vec_size matches the kernel's sf_vec_size.
        fc1_out_sf_vec_size = self.sf_vec_size
        sf_block_cols = (
            (intermediate_out // fc1_out_sf_vec_size) + 3
        ) // 4 * 4
        fc1_output_sf_bytes = sf_total_rows_upper * sf_block_cols

        # fc1_recompute (forward-swiglu recompute): N = inter_half = intermediate_out // 2.
        fc1_recompute_bytes = (
            data_total_rows * dfc2_weight_n * self.ab_dtype.width // 8
        )
        fc1_recompute_row_blocks_upper = sf_total_rows_upper // fc1_out_sf_vec_size
        fc1_recompute_sf_bytes = fc1_recompute_row_blocks_upper * dfc2_weight_n

        # fc1_col_output (col-quant grad_y1): N = intermediate_out (same as
        # grad_y1's row-quant fc1_output).  Col-SF: row_blocks × intermediate.
        fc1_col_output_bytes = fc1_output_bytes
        fc1_col_output_sf_bytes = fc1_recompute_row_blocks_upper * intermediate_out

        # fc1_done_counter: one Int32 per CTA-level token block (each cluster block
        # has atom_thr_size CTAs, each with its own per-CTA counter slot).
        counter_slots_upper = (
            (data_total_rows + mma_tiler_n - 1) // mma_tiler_n
            + experts
        )
        fc1_done_counter_bytes = counter_slots_upper * 4

        # load_balance_counter: Int32 scalar.
        if self.load_balance_mode == "atomic_counter":
            load_balance_counter_bytes = 4
        else:
            load_balance_counter_bytes = 0

        total = (
            fc1_output_bytes
            + fc1_output_sf_bytes
            + fc1_recompute_bytes
            + fc1_recompute_sf_bytes
            + fc1_col_output_bytes
            + fc1_col_output_sf_bytes
            + fc1_done_counter_bytes
            + load_balance_counter_bytes
        )

        # 128B align (TMA tensor base address alignment requirement).
        alignment = 128
        total = ((total + alignment - 1) // alignment) * alignment
        return total

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """SMEM → TMEM tiled copy + partition for SFA / SFB."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # =========================================================================
    # Token-comm hook surface (MegaMoE-only; lean base = no-op stubs)
    #
    # Mirrors the hook interface in ``moe_mxfp8_glu.kernel_mxfp8_glu_fc12``
    # so that ``Sm107MegaMoEMxfp8DgluKernel`` can override exactly the same
    # methods.  The mega wrapper realigns dispatch onto warps 8-11 (128-aligned
    # for next token_comm) and relocates ``c_load_warp_id`` above the transfer
    # block (warp 12 or 16); the lean base keeps c_load at warp 8.
    # =========================================================================

    def _smem_misc_budget_bytes(self) -> int:
        """SMEM reserved for non-problem-tensor buffers (mbarriers, sched, TMEM state).

        MegaMoE subclass adds dispatch-warp SMEM on top via::

            return super()._smem_misc_budget_bytes() + self._dispatch_smem_bytes()
        """
        return self._SmemMiscBudget

    def token_comm_extra_smem_storage_class(self) -> type:
        """Return a ``@cute.struct`` for dispatch-warp SMEM, or None."""
        return None

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        """Return dispatch->fc1 release counter pointer, or None (lean: disabled)."""
        return None

    def sched_ext_fc1_peek_threshold(self) -> int:
        """Return the fc1 ready-counter peek threshold for DgluMxFp8Fc12SchedExtension."""
        return 0

    def sched_ext_fc1_counter_cumul_scale(self) -> int:
        """Return the scale factor for the fc1 ready-counter slot formula."""
        return 1

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        """Sched warp: wait for dispatch barrier before reading sizes.  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(self, token_comm_args, work_tile_info):
        """TMA-A warp: spin until dispatch-pulled tokens are resident.  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self, token_comm_args, token_comm_storage, *, warp_idx, lane_idx, tidx,
    ):
        """Body for dispatch warps 8-11 (MegaMoE-only).  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_token_back_warp_body(
        self, token_comm_args, token_comm_storage, *, warp_idx, lane_idx, tidx,
    ):
        """Body for standalone token-back warps 12-15 (MegaMoE-only).  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_kernel_tail(self, token_comm_args, *, warp_idx, lane_idx, tidx):
        """All-warp kernel tail (NVLink release, etc.).  No-op base."""
        pass

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,           # (token_sum_padded, hidden) = grad_out
        fc1_weight: cute.Tensor,           # (experts, hidden, inter_half)
        activation_sf: cute.Tensor,         # (token_sum_padded_sf, hidden / sf_vec_size)
        fc1_weight_sf: cute.Tensor,         # dfc2-weight SF
        fc1_output: cute.Tensor,         # (token_sum_padded, intermediate)
        fc1_output_sf: cute.Tensor,      # (token_sum_padded_sf, intermediate / sf_vec_size)
        fc1_recompute: Optional[cute.Tensor],      # (token_sum_padded, inter_half)
        fc1_recompute_sf: Optional[cute.Tensor],   # (token_sum_padded_sf, inter_half / sf_vec_size)
        fc1_col_output: Optional[cute.Tensor],     # (token_sum_padded, intermediate)
        fc1_col_output_sf: Optional[cute.Tensor],  # (sf_row_blocks, intermediate) col-SF
        fc2_weight: cute.Tensor,          # (experts, intermediate, hidden)
        fc2_weight_sf: cute.Tensor,        # dfc1-weight SF
        fc2_output: cute.Tensor,         # (token_sum_padded, hidden) BFloat16 = grad_x
        fc1_preact: cute.Tensor,         # (token_sum_padded, intermediate) BFloat16
        topk_scores: cute.Tensor,     # (token_sum_padded,) Float32
        beta: cute.Tensor,            # (experts,) Float32
        dprob: cute.Tensor,           # (token_sum_padded,) Float32
        fc1_done_counter: cute.Tensor,  # (max_token_block_per_rank,) Int32
        offs: Optional[cute.Tensor] = None,  # (experts,) Int32 cumulative end offsets
        max_active_clusters: cutlass.Constexpr = None,
        stream: cuda.CUstream = None,
        norm_const_tensor: Optional[cute.Tensor] = None,
        global_activation_sf: Optional[cute.Tensor] = None,
        global_fc1_weight_sf: Optional[cute.Tensor] = None,
        load_balance_counter: Optional[cute.Tensor] = None,
        expert_token_sizes: Optional[cute.Tensor] = None,
        token_comm_args=None,
        overflow_flag: cute.Tensor = None,
        mega_peer_rank_ptr_mapper=None,
        mega_local_rank: Optional[cutlass.Int32] = None,
        mega_local_workspace: Optional[cute.Pointer] = None,
        mega_shared_workspace: Optional[cute.Pointer] = None,
        mega_activation: Optional[cute.Tensor] = None,
        mega_activation_sf: Optional[cute.Tensor] = None,
        mega_pre_reduced_activation: Optional[cute.Tensor] = None,
        mega_pre_reduced_activation_sf: Optional[cute.Tensor] = None,
    ) -> None:
        """Launch the fused dfc2+dfc1 dGLU MXFP8 (backward) kernel."""
        if cutlass.const_expr(self.static_expert_shape is not None):
            (
                experts_static,
                intermediate_gateup_static,  # inter_half = dfc2 weight N
                hidden_static,
            ) = self.static_expert_shape
            intermediate_out_static = intermediate_gateup_static * 2  # grad_y1 / dfc1-K

            fc1_weight = cute.make_tensor(
                fc1_weight.iterator,
                cute.make_layout(
                    (experts_static, hidden_static, intermediate_gateup_static),
                    stride=fc1_weight.stride,
                ),
            )
            fc2_weight = cute.make_tensor(
                fc2_weight.iterator,
                cute.make_layout(
                    (experts_static, intermediate_out_static, hidden_static),
                    stride=fc2_weight.stride,
                ),
            )
            activation = cute.make_tensor(
                activation.iterator,
                cute.make_layout(
                    (activation.shape[0], hidden_static),
                    stride=activation.stride,
                ),
            )
            fc1_output = cute.make_tensor(
                fc1_output.iterator,
                cute.make_layout(
                    (fc1_output.shape[0], intermediate_out_static),
                    stride=fc1_output.stride,
                ),
            )
            fc1_recompute = cute.make_tensor(
                fc1_recompute.iterator,
                cute.make_layout(
                    (fc1_recompute.shape[0], intermediate_gateup_static),
                    stride=fc1_recompute.stride,
                ),
            )
            fc1_col_output = cute.make_tensor(
                fc1_col_output.iterator,
                cute.make_layout(
                    (fc1_col_output.shape[0], intermediate_out_static),
                    stride=fc1_col_output.stride,
                ),
            )
            fc1_preact = cute.make_tensor(
                fc1_preact.iterator,
                cute.make_layout(
                    (fc1_preact.shape[0], intermediate_out_static),
                    stride=fc1_preact.stride,
                ),
            )
            if cutlass.const_expr(len(fc2_output.shape) == 3):
                fc2_output = cute.make_tensor(
                    fc2_output.iterator,
                    cute.make_layout(
                        (fc2_output.shape[0], fc2_output.shape[1], hidden_static),
                        stride=fc2_output.stride,
                    ),
                )
            else:
                fc2_output = cute.make_tensor(
                    fc2_output.iterator,
                    cute.make_layout(
                        (fc2_output.shape[0], hidden_static),
                        stride=fc2_output.stride,
                    ),
                )

        # ── GEMM-domain transform for fc1 phase ──
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # A_gemm (fc1 activations): (tokens_sum, hidden) -> (M=tokens, K=hidden, L=1).
        tokens_sum, hidden = activation.shape
        activation_gemm = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (tokens_sum, hidden, 1),
                stride=(activation.stride[0], activation.stride[1], 0),
            ),
        )

        # B_gemm (W2T): reinterpret public C-contiguous (experts, hidden, inter_half)
        # as (N=inter_half, K=hidden, L=experts). The stride permutation makes this
        # an N-major GEMM operand without staging or moving data.
        experts, hidden_b, intermediate_gateup = fc1_weight.shape
        fc1_weight_gemm = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (intermediate_gateup, hidden_b, experts),
                stride=(fc1_weight.stride[2], fc1_weight.stride[1], fc1_weight.stride[0]),
            ),
        )

        intermediate_downproj = fc1_output.shape[1]
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (tokens_sum, intermediate_downproj, 1),
                stride=(fc1_output.stride[0], fc1_output.stride[1], 0),
            ),
        )

        # dFC2 auxiliary data planes use the public token-major ABI.
        fc1_recompute_gemm = cute.make_tensor(
            fc1_recompute.iterator,
            cute.make_layout(
                (fc1_recompute.shape[0], fc1_recompute.shape[1], 1),
                stride=(fc1_recompute.stride[0], fc1_recompute.stride[1], 0),
            ),
        )

        fc1_col_output_gemm = cute.make_tensor(
            fc1_col_output.iterator,
            cute.make_layout(
                (fc1_col_output.shape[0], fc1_col_output.shape[1], 1),
                stride=(fc1_col_output.stride[0], fc1_col_output.stride[1], 0),
            ),
        )

        # Forward pre-activation (gate||up)
        fc1_preact_gemm = cute.make_tensor(
            fc1_preact.iterator,
            cute.make_layout(
                (tokens_sum, intermediate_downproj, 1),
                stride=(fc1_preact.stride[0], fc1_preact.stride[1], 0),
            ),
        )

        # SFA / SFB scale tensors (atom-tiled)
        tokens_sum_padded = activation_sf.shape[0]
        hidden_padded = activation_sf.shape[1] * self.sf_vec_size
        activation_sf_gemm = cute.make_tensor(
            activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, hidden_padded, 1), self.sf_vec_size
            ),
        )
        intermediate_gateup_padded_mul_hidden_padded = fc1_weight_sf.shape[1]
        intermediate_gateup_padded = (
            intermediate_gateup_padded_mul_hidden_padded * self.sf_vec_size
        ) // hidden_padded
        fc1_weight_sf_gemm = cute.make_tensor(
            fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (intermediate_gateup_padded, hidden_padded, experts),
                self.sf_vec_size,
            ),
        )

        # GEMM-domain transform for fc2 phase. W1T is public C-contiguous
        # (experts, 2 * inter_half, hidden), with its reduction rows already in
        # 32-wide gate/up order. Preserve that K ordering and expose hidden as N.
        experts2, intermediate_downproj_b2, hidden_b2 = fc2_weight.shape
        fc2_weight_gemm = cute.make_tensor(
            fc2_weight.iterator,
            cute.make_layout(
                (hidden_b2, intermediate_downproj_b2, experts2),
                stride=(fc2_weight.stride[2], fc2_weight.stride[1], fc2_weight.stride[0]),
            ),
        )

        if cutlass.const_expr(len(fc2_output.shape) == 3):
            fc2_hidden_out = fc2_output.shape[2]
            fc2_output_gemm = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (fc2_output.shape[0], fc2_hidden_out, c1),
                    stride=(fc2_output.stride[0], fc2_output.stride[2], c0),
                ),
            )
        else:
            fc2_hidden_out = fc2_output.shape[1]
            fc2_output_gemm = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (tokens_sum, fc2_hidden_out, c1),
                    stride=(fc2_output.stride[0], fc2_output.stride[1], c0),
                ),
            )

        fc1_out_sf_vec_size = self.sf_vec_size
        tokens_sum_padded_sf = fc1_output_sf.shape[0]
        intermediate_downproj_padded = fc1_output_sf.shape[1] * fc1_out_sf_vec_size
        fc1_output_sf_gemm_for_fc2_load = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded_sf, intermediate_downproj_padded, 1),
                fc1_out_sf_vec_size,
            ),
        )

        hidden_padded_fc2_mul_intermediate_downproj_padded = fc2_weight_sf.shape[1]
        hidden_padded_fc2 = (
            hidden_padded_fc2_mul_intermediate_downproj_padded * self.sf_vec_size
        ) // intermediate_downproj_padded
        fc2_weight_sf_gemm = cute.make_tensor(
            fc2_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (hidden_padded_fc2, intermediate_downproj_padded, experts2),
                self.sf_vec_size,
            ),
        )

        expert_cnt = experts
        hidden_dim = hidden

        # Infer dtypes and major modes
        self.a_dtype: Type[cutlass.Numeric] = activation_gemm.element_type
        self.b_dtype: Type[cutlass.Numeric] = fc1_weight_gemm.element_type
        self.fc1_output_dtype: Type[cutlass.Numeric] = fc1_output_gemm.element_type
        self.sf_dtype: Type[cutlass.Numeric] = activation_sf_gemm.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(activation_gemm).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(fc1_weight_gemm).mma_major_mode()
        self.fc1_output_layout = utils.LayoutEnum.from_tensor(fc1_output_gemm)

        self._setup_attributes()
        tiled_mma, tiled_mma_sfb = self._create_tiled_mmas()

        # fc1 TMA atoms load A1
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_activation, tma_tensor_fc1_activation = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            activation_gemm,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load B1
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_weight, tma_tensor_fc1_weight = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            fc1_weight_gemm,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load SFA1
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_fc1_activation_sf, tma_tensor_fc1_activation_sf = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            activation_sf_gemm,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )

        # TMA load SFB1
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_fc1_weight_sf, tma_tensor_fc1_weight_sf = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            fc1_weight_sf_gemm,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )

        # Coalesced TMA G2S load of the forward preact (dswiglu C input).
        preact_tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_fc1_preact, tma_tensor_fc1_preact = cpasync.make_tiled_tma_atom(
            preact_tma_op,
            fc1_preact_gemm,
            self.epilogue.preact_smem_layout_one_stage,
            self.epilogue.preact_epi_tile,
        )

        # Coalesced TMA S2G store of grad_y1 (dfc2 fp8 output).
        grad_y1_tma_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_grad_y1, tma_tensor_grad_y1 = cpasync.make_tiled_tma_atom(
            grad_y1_tma_op,
            fc1_output_gemm,
            self.epilogue.d_smem_layout_one_stage,
            self.epilogue.d_epi_tile,
        )
        tma_atom_fc1_recompute, tma_tensor_fc1_recompute = cpasync.make_tiled_tma_atom(
            grad_y1_tma_op,
            fc1_recompute_gemm,
            self.epilogue.d_smem_layout_one_stage,
            self.epilogue.d_epi_tile,
        )
        tma_atom_fc1_col_output, tma_tensor_fc1_col_output = cpasync.make_tiled_tma_atom(
            grad_y1_tma_op,
            fc1_col_output_gemm,
            self.epilogue.d_smem_layout_one_stage,
            self.epilogue.d_epi_tile,
        )

        # fc1 SFC GMEM tensor (= fc1_output_sf user view).  No TMA atom; it is
        # per-thread STG.
        fc1_output_sf_gemm = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, intermediate_downproj, 1),
                self.sf_vec_size,
            ),
        )

        # Token-major blocked SF carriers.
        fc1_recompute_sf_gemm = cute.make_tensor(
            fc1_recompute_sf.iterator,
            cute.make_layout(
                (fc1_recompute_sf.shape[0], fc1_recompute_sf.shape[1], 1),
                stride=(fc1_recompute_sf.stride[0], fc1_recompute_sf.stride[1], 0),
            ),
        )

        fc1_col_output_sf_gemm = cute.make_tensor(
            fc1_col_output_sf.iterator,
            cute.make_layout(
                (fc1_col_output_sf.shape[0], fc1_col_output_sf.shape[1], 1),
                stride=(fc1_col_output_sf.stride[0], fc1_col_output_sf.stride[1], 0),
            ),
        )

        # ── fc2 TMA atoms: fc1_output → A-side (M=tokens), fc2_weight → B-side (N=hidden) ──
        tma_atom_fc2_activation, tma_tensor_fc2_activation = (
            cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                fc1_output_gemm,
                a_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_atom_fc2_weight, tma_tensor_fc2_weight = (
            cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                fc2_weight_gemm,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_atom_fc2_activation_sf, tma_tensor_fc2_activation_sf = (
            cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                fc1_output_sf_gemm_for_fc2_load,
                sfa_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
        )
        tma_atom_fc2_weight_sf, tma_tensor_fc2_weight_sf = (
            cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                fc2_weight_sf_gemm,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
        )

        # ── Scheduler params + grid + launch ──
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            if cutlass.const_expr(load_balance_counter is None):
                raise ValueError(
                    "load_balance_counter must be provided when "
                    "load_balance_mode == 'atomic_counter'"
                )
            load_balance_counter_ptr = load_balance_counter.iterator
        else:
            load_balance_counter_ptr = None

        # On the MegaMoE path the per-expert sizes come from the Router (device-side), so the
        # caller supplies neither offs nor expert_token_sizes.
        if cutlass.const_expr(not self.enable_token_comm):
            if cutlass.const_expr((offs is None) == (expert_token_sizes is None)):
                raise ValueError(
                    "Exactly one of `offs` / `expert_token_sizes` must be "
                    "non-None.  Got offs="
                    f"{'set' if offs is not None else 'None'}, "
                    f"expert_token_sizes="
                    f"{'set' if expert_token_sizes is not None else 'None'}."
                )

        self._build_scheduler(
            expert_cnt=expert_cnt,
            intermediate_gateup=intermediate_gateup,
            hidden_dim=hidden_dim,
            launch_cluster_count=max_active_clusters,
        )
        grid = self.scheduler.get_grid_shape(max_active_clusters=max_active_clusters)

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            # fc1 TMA atoms / tensors (A=activations, B=weights)
            tma_atom_fc1_activation,
            tma_tensor_fc1_activation,
            tma_atom_fc1_weight,
            tma_tensor_fc1_weight,
            tma_atom_fc1_activation_sf,
            tma_tensor_fc1_activation_sf,
            tma_atom_fc1_weight_sf,
            tma_tensor_fc1_weight_sf,
            # fc2 TMA atoms / tensors (fc1_output→A, fc2_weight→B)
            tma_atom_fc2_activation,
            tma_tensor_fc2_activation,
            tma_atom_fc2_weight,
            tma_tensor_fc2_weight,
            tma_atom_fc2_activation_sf,
            tma_tensor_fc2_activation_sf,
            tma_atom_fc2_weight_sf,
            tma_tensor_fc2_weight_sf,
            # GEMM-domain tensors (fc1)
            activation_gemm,
            fc1_weight_gemm,
            fc1_output_gemm,
            activation_sf_gemm,
            fc1_weight_sf_gemm,
            fc1_output_sf_gemm,
            # GEMM-domain tensors (fc2)
            fc2_weight_gemm,
            fc2_output_gemm,
            fc2_weight_sf_gemm,
            fc1_output_sf_gemm_for_fc2_load,
            # forward pre-activation (dswiglu input) — TMA G2S into SMEM
            tma_atom_fc1_preact,
            tma_tensor_fc1_preact,
            tma_atom_grad_y1,
            tma_tensor_grad_y1,
            # token-major auxiliary data — TMA S2G stores
            tma_atom_fc1_recompute,
            tma_tensor_fc1_recompute,
            fc1_recompute_sf_gemm,
            tma_atom_fc1_col_output,
            tma_tensor_fc1_col_output,
            fc1_col_output_sf_gemm,
            # topk / beta / dprob + cross-phase sync workspace
            topk_scores,
            beta,
            dprob,
            overflow_flag,
            fc1_done_counter,
            # Scheduling
            offs,
            expert_token_sizes,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            # SMEM layouts
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            token_comm_args,
            # MegaMoE push-model token-comm inputs (None on the lean path)
            mega_peer_rank_ptr_mapper,
            mega_local_rank,
            mega_local_workspace,
            mega_shared_workspace,
            mega_activation,
            mega_activation_sf,
            mega_pre_reduced_activation,
            mega_pre_reduced_activation_sf,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
        )


    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        # fc1 TMA atoms / tensors
        tma_atom_fc1_activation_1: cute.CopyAtom,
        tma_tensor_fc1_activation_1: cute.Tensor,
        tma_atom_weight: cute.CopyAtom,
        tma_tensor_weight: cute.Tensor,
        tma_atom_fc1_activation_1_sf: cute.CopyAtom,
        tma_tensor_fc1_activation_1_sf: cute.Tensor,
        tma_atom_fc1_weight_sf: cute.CopyAtom,
        tma_tensor_fc1_weight_sf: cute.Tensor,
        # fc2 TMA atoms / tensors (fc1_output→A, fc2_weight→B)
        tma_atom_fc2_activation: cute.CopyAtom,
        tma_tensor_fc2_activation: cute.Tensor,
        tma_atom_fc2_weight: cute.CopyAtom,
        tma_tensor_fc2_weight: cute.Tensor,
        tma_atom_fc2_activation_sf: cute.CopyAtom,
        tma_tensor_fc2_activation_sf: cute.Tensor,
        tma_atom_fc2_weight_sf: cute.CopyAtom,
        tma_tensor_fc2_weight_sf: cute.Tensor,
        # GEMM-domain tensors (fc1)
        activation_gemm: cute.Tensor,
        fc1_weight_gemm: cute.Tensor,
        fc1_output_gemm: cute.Tensor,
        activation_sf_gemm: cute.Tensor,
        fc1_weight_sf_gemm: cute.Tensor,
        fc1_output_sf_gemm: cute.Tensor,
        # GEMM-domain tensors (fc2)
        fc2_weight_gemm: cute.Tensor,
        fc2_output_gemm: cute.Tensor,
        fc2_weight_sf_gemm: cute.Tensor,
        fc1_output_sf_gemm_for_fc2_load: cute.Tensor,
        # forward pre-activation (dswiglu input) — TMA G2S into SMEM
        tma_atom_fc1_preact: cute.CopyAtom,
        tma_tensor_fc1_preact: cute.Tensor,
        # grad_y1 (dfc2 output) — TMA S2G store
        tma_atom_grad_y1: cute.CopyAtom,
        tma_tensor_grad_y1: cute.Tensor,
        # token-major auxiliary data — TMA S2G stores
        tma_atom_fc1_recompute: cute.CopyAtom,
        tma_tensor_fc1_recompute: cute.Tensor,
        fc1_recompute_sf_gemm: cute.Tensor,
        tma_atom_fc1_col_output: cute.CopyAtom,
        tma_tensor_fc1_col_output: cute.Tensor,
        fc1_col_output_sf_gemm: cute.Tensor,
        # topk / beta / dprob + cross-phase sync workspace
        topk_scores: cute.Tensor,
        beta: cute.Tensor,
        dprob: cute.Tensor,
        overflow_flag: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        # Scheduling
        offs: Optional[cute.Tensor],
        expert_token_sizes: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        # SMEM layouts
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        token_comm_args=None,
        # MegaMoE push-model token-comm inputs
        mega_peer_rank_ptr_mapper=None,
        mega_local_rank: Optional[cutlass.Int32] = None,
        mega_local_workspace: Optional[cute.Pointer] = None,
        mega_shared_workspace: Optional[cute.Pointer] = None,
        mega_activation: Optional[cute.Tensor] = None,
        mega_activation_sf: Optional[cute.Tensor] = None,
        mega_pre_reduced_activation: Optional[cute.Tensor] = None,
        mega_pre_reduced_activation_sf: Optional[cute.Tensor] = None,
    ):
        """Device kernel for fused fc1+fc2 swap-AB GLU MXFP8 grouped GEMM."""
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))

        # MegaMoE (push model): bind the device workspace here so the fc1_ready counter
        # pointer that the scheduler extension spins on (built just below) resolves.
        if cutlass.const_expr(self.enable_token_comm):
            self._mega_device_workspace.assign_device_members(
                mega_local_workspace, mega_shared_workspace
            )

        # fc2 waits for all fc1 intermediate N-tiles in the same token block.
        ext_fc2_spin_threshold = (
            fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[1] - 1
        ) // self.cta_tile_shape_mnk[1] * self.epilogue._atom_thr_size

        if cutlass.const_expr(self.enable_token_comm):
            _aux_expert_sizes = self.token_comm.local_expert_sizes(
                self._mega_device_workspace, mega_local_rank
            )
        else:
            _aux_expert_sizes = expert_token_sizes

        ext = DgluMxFp8Fc12SchedExtension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter.iterator,
            fc2_spin_threshold=ext_fc2_spin_threshold,
            # MegaMoE: peek the dispatch->fc1 ready counter (None on the lean
            # path).  Parity with the forward kernel's SchedExtension wiring.
            fc1_ready_counter_pointer=self.token_comm_hook_fc1_ready_counter_ptr(
                token_comm_args
            ),
            # Fold the 2 CTAs of a cluster onto one fc1_ready slot
            cluster_m=self.epilogue._atom_thr_size,
            expert_token_sizes=_aux_expert_sizes,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
        )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # MegaMoE (push model): bind token-comm device members (transfer-warp state + the
        # NVLink barrier's peer mapper) before any token_in / token_back / size-wait runs.
        if cutlass.const_expr(self.enable_token_comm):
            _mega_token_comm_args = TokenCommArgs(
                mega_activation,
                mega_activation_sf,
                mega_pre_reduced_activation,
                mega_pre_reduced_activation_sf,
                mega_peer_rank_ptr_mapper,
            )
            _mega_cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
            _, _, _mega_cluster_idx = cute.arch.block_idx()
            _mega_linear_cta_idx = cta_rank_in_cluster + _mega_cluster_idx * _mega_cluster_size
            self.token_comm.assign_device_members(
                device_workspace=self._mega_device_workspace,
                token_comm_args=_mega_token_comm_args,
                local_rank=mega_local_rank,
                linear_cta_idx=_mega_linear_cta_idx,
            )

        # preact (dswiglu C) pipeline
        num_c_stage = self.num_c_stage
        num_c_pipe_stage = self.num_c_pipe_stage
        num_d_stage = self.num_d_stage

        # SharedStorage (mainloop + epilogue SMEM). next's scheduler owns its own
        # SMEM workspace, allocated separately below.
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_pipeline_stages * 2
            ]
            c_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_c_pipe_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sPre: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(self.epilogue.preact_staged_smem_layout(num_c_stage).outer),
                ],
                1024,
            ]
            # Unified dFC2 data-output staging; slot count is compile-time gated.
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.fc1_output_dtype,
                    cute.cosize(self.epilogue.d_staged_smem_layout(num_d_stage).outer),
                ],
                1024,
            ]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # next scheduler SMEM: a self-contained workspace carved from the same
        # allocator; its transport regions resolve against ``sched_smem_base``.
        sched_storage = smem.allocate(self.sched_smem_ws.storage_class())
        sched_smem_base = sched_storage.buffer.data_ptr()

        # MegaMoE-only dispatch-warp SMEM (pull_buffer, mbarriers, etc.).
        # Kept out of ``SharedStorage`` so the lean path never allocates it.
        TokenCommStorageCls = self.token_comm_extra_smem_storage_class()
        if cutlass.const_expr(TokenCommStorageCls is not None):
            token_comm_storage = smem.allocate(TokenCommStorageCls)
        else:
            token_comm_storage = None

        # ── Pipelines: two TMA producer warps share the AB pipeline. ──

        ab_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 2
        )
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes // 2,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = (
            len(self.epilogue_warp_id) * 32 * (2 if use_2cta_instrs else 1)
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_pipeline_stages,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # preact (dswiglu C) pipeline
        c_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len(self.epilogue_warp_id)
        )
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_full_mbar_ptr.data_ptr(),
            num_stages=num_c_pipe_stage,
            producer_group=c_pipeline_producer_group,
            consumer_group=c_pipeline_consumer_group,
            tx_count=2 * self.epilogue.preact_bytes_per_stage,
            defer_sync=True,
        )
        # d pipeline
        d_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            32 * len(self.epilogue_warp_id),
        )
        d_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=num_d_stage // self.epilogue.d_output_slots,
            producer_group=d_producer_group,
        )


        # TMEM allocator
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        # Sched
        scheduler = self.scheduler
        if cutlass.const_expr(self.enable_token_comm):
            _sched_expert_sizes = self.token_comm.local_expert_sizes(
                self._mega_device_workspace, mega_local_rank
            )
            _sched_prefix_sum = None
            # Bind the scheduler's own device workspace
            self.sched_device_ws.assign_device_members(
                cute.make_ptr(
                    cutlass.Uint8,
                    mega_local_workspace.toint()
                    + self._mega_device_workspace.offset(self.sched_work_id_region),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                mega_shared_workspace,
            )
        else:
            _sched_expert_sizes = expert_token_sizes
            _sched_prefix_sum = offs
        scheduler.assign_device_members(
            expert_token_sizes=_sched_expert_sizes,
            expert_token_prefix_sum=_sched_prefix_sum,
            actual_expert_shape=None,
            block_idx=cute.arch.block_idx(),
            smem_workspace=self.sched_smem_ws,
            smem_base=sched_smem_base,
            device_workspace=self.sched_device_ws,
        )
        sched_consumer = scheduler.make_consumer()

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM tensors A / B / SFA / SFB (shared by fc1 / fc2)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        # preact (dswiglu C) staging tensor
        preact_smem_layout_staged = self.epilogue.preact_staged_smem_layout(
            num_c_stage
        )
        sPre = storage.sPre.get_tensor(
            preact_smem_layout_staged.outer,
            swizzle=preact_smem_layout_staged.inner,
        )

        # Unified dFC2 data-output store staging tensor.
        d_smem_layout_staged = self.epilogue.d_staged_smem_layout(num_d_stage)
        sD = storage.sD.get_tensor(
            d_smem_layout_staged.outer,
            swizzle=d_smem_layout_staged.inner,
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # acc_fake layout: (MMA, MMA_M, MMA_N, STAGE)
        acc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # Cluster wait before TMEM alloc.
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        mma_tiler_k = self.mma_tiler[2]
        k_tile_cnt_fc1 = (fc1_weight_gemm.shape[1] + mma_tiler_k - 1) // mma_tiler_k
        k_tile_cnt_fc2 = (fc2_weight_gemm.shape[1] + mma_tiler_k - 1) // mma_tiler_k
        # fc2 spin threshold: number of N-tiles per CTA (per-CTA counter now).
        fc2_spin_threshold = (
            (fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[1] - 1)
            // self.cta_tile_shape_mnk[1]
        ) * self.epilogue._atom_thr_size

        # ════════════════════════════════════════════════════════════════════
        # Scheduler warp (warp 7) — lean path
        # ════════════════════════════════════════════════════════════════════
        if warp_idx == self.sched_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)
            # MegaMoE: block until the Router has published this rank's per-expert sizes
            self.token_comm_hook_sched_warp_pre_init_wait(token_comm_args)
            work_tile = scheduler.gen_next_work()
            while work_tile.is_valid_tile:
                scheduler.publish_work(ext.prepare_work_tile(work_tile))
                work_tile = scheduler.gen_next_work()
            # Sentinel publish (the tile is already invalid here).
            scheduler.publish_work(work_tile)
            scheduler.produce_tail()

        # ════════════════════════════════════════════════════════════════════
        # TMA load warps (warps 5 / 6)
        # ════════════════════════════════════════════════════════════════════
        #
        # TMA-A loads weights/SFA; TMA-B loads activations/SFB and waits for
        # fc1 workspace readiness in fc2 phase.  Both feed the same AB pipeline.

        # ── TMA-A warp (warp 5) ─────────────────────────────────────────────
        if warp_idx == self.tma_a_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)
            a_full_mcast_mask = None
            sfa_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
                sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )

            b_full_mcast_mask = None
            if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
                )
            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )

            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            sfa_cta_layout = a_cta_layout

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )
                if is_phase_linear1:
                    iket.range_push("tma_weight_fc1")
                    # MegaMoE: spin until the dispatch (token_in) warps have pulled
                    ext.wait_for_input(work_tile_info)
                    self.token_comm_hook_fc1_tma_b_predispatch_spin(
                        token_comm_args, work_tile_info,
                    )

                    k_tile_cnt = k_tile_cnt_fc1
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "fc1_activation", tma_tensor_fc1_activation_1, work_tile_info,
                    )
                    real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                        "fc1_activation_sf", tma_tensor_fc1_activation_1_sf, work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    gSFA_mkl = cute.local_tile(
                        real_sfa,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc1_activation_1,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )
                    tAsSFA, tAgSFA = cpasync.tma_partition(
                        tma_atom_fc1_activation_1_sf,
                        block_in_cluster_coord_vmnk[2],
                        sfa_cta_layout,
                        cute.group_modes(sSFA, 0, 3),
                        cute.group_modes(tCgSFA, 0, 3),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)

                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                    tAgSFA_slice = tAgSFA[(None, mma_tile_m, None, 0)]

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status
                        )
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc1_activation_1,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc1_activation_1_sf,
                            tAgSFA_slice[(None, handle.count)],
                            tAsSFA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfa,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                    iket.range_pop()

                else:
                    # fc2 phase A-side: load fc1_output (M=tokens) + wait for fc1 done
                    iket.range_push("tma_token_fc2")
                    counter_slot = (
                        work_tile_info.cumulative_token_block_count
                        + work_tile_info.tile_m_idx // cutlass.Int32(self.epilogue._atom_thr_size)
                    )
                    counter_ptr = fc1_done_counter.iterator + counter_slot
                    iket.range_push("tma_token_fc2_a_wait")
                    spin_wait(
                        counter_ptr,
                        lambda v: v >= fc2_spin_threshold,
                        sleep_cycles=20,
                    )
                    iket.range_pop()
                    k_tile_cnt = k_tile_cnt_fc2
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "fc2_activation", tma_tensor_fc2_activation, work_tile_info,
                    )
                    real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                        "fc2_activation_sf", tma_tensor_fc2_activation_sf, work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    gSFA_mkl = cute.local_tile(
                        real_sfa,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc2_activation,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )
                    tAsSFA, tAgSFA = cpasync.tma_partition(
                        tma_atom_fc2_activation_sf,
                        block_in_cluster_coord_vmnk[2],
                        sfa_cta_layout,
                        cute.group_modes(sSFA, 0, 3),
                        cute.group_modes(tCgSFA, 0, 3),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)

                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                    tAgSFA_slice = tAgSFA[(None, mma_tile_m, None, 0)]

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status
                        )
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc2_activation,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc2_activation_sf,
                            tAgSFA_slice[(None, handle.count)],
                            tAsSFA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfa,
                            mcast_mask=sfa_full_mcast_mask,
                        )

                    iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            ab_producer.tail()

        # ── TMA-B warp (warp 6) ─────────────────────────────────────────────
        if warp_idx == self.tma_b_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)
            b_full_mcast_mask = None
            sfb_full_mcast_mask = None
            if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
                )
                sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_sfb_vmnk,
                    block_in_cluster_coord_sfb_vmnk,
                    mcast_mode=1,
                )

            # FC1: weight (B) is multicast (like original A)
            a_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )

            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            sfb_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
            )

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )

                if is_phase_linear1:
                    iket.range_push("tma_token_fc1")

                    k_tile_cnt = k_tile_cnt_fc1
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "fc1_weight", tma_tensor_weight, work_tile_info,
                    )
                    real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                        "fc1_weight_sf", tma_tensor_fc1_weight_sf, work_tile_info,
                    )

                    # N-K tiling for N-side weight (N=intermediate, K=hidden).
                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB_nkl)
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_weight,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        tma_atom_fc1_weight_sf,
                        block_in_cluster_coord_sfb_vmnk[1],
                        sfb_cta_layout,
                        cute.group_modes(sSFB, 0, 3),
                        cute.group_modes(tCgSFB, 0, 3),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)

                    tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]
                    tBgSFB_slice = tBgSFB[(None, work_tile_info.tile_n_idx, None, 0)]

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status
                        )
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
                        cute.copy(
                            tma_atom_weight,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_b,
                            mcast_mask=a_full_mcast_mask,  # same as A-loading for weights
                        )
                        cute.copy(
                            tma_atom_fc1_weight_sf,
                            tBgSFB_slice[(None, handle.count)],
                            tBsSFB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfb,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                    iket.range_pop()

                else:
                    # fc2 phase B-side: load fc2_weight (N=hidden), no counter wait
                    iket.range_push("tma_weight_fc2")
                    k_tile_cnt = k_tile_cnt_fc2
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "fc2_weight", tma_tensor_fc2_weight, work_tile_info,
                    )
                    real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                        "fc2_weight_sf", tma_tensor_fc2_weight_sf, work_tile_info,
                    )

                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB_nkl)
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_fc2_weight,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        tma_atom_fc2_weight_sf,
                        block_in_cluster_coord_sfb_vmnk[1],
                        sfb_cta_layout,
                        cute.group_modes(sSFB, 0, 3),
                        cute.group_modes(tCgSFB, 0, 3),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)

                    fc2_b_hidden_tile = work_tile_info.tile_n_idx
                    tBgB_slice = tBgB[(None, fc2_b_hidden_tile, None, 0)]
                    tBgSFB_slice = tBgSFB[(None, fc2_b_hidden_tile, None, 0)]

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status
                        )
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc2_weight,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_b,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc2_weight_sf,
                            tBgSFB_slice[(None, handle.count)],
                            tBsSFB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfb,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                    iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            ab_producer.tail()

        # ════════════════════════════════════════════════════════════════════
        # MMA warp (warp 4)
        # ════════════════════════════════════════════════════════════════════
        #
        # Both phases share tiled_mma and TMEM; only K-tile count differs.
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_base = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)

            # SFA TMEM tensor (placed after the acc cols).
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # SFB TMEM tensor (after acc + SFA cols).
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_pipeline_stages
            )

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )
                # Prebind k_tile_cnt due to DSL AST.
                k_tile_cnt = cutlass.Int32(0)
                if is_phase_linear1:
                    k_tile_cnt = k_tile_cnt_fc1
                    iket.range_push("mma_dfc2")
                else:
                    k_tile_cnt = k_tile_cnt_fc2
                    iket.range_push("mma_dfc1")

                acc_stage_index = acc_producer_state.index

                if is_leader_cta:
                    tCtAcc = acc_base[(None, None, None, acc_stage_index)]

                    ab_consumer.reset()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if k_tile_cnt > 0:
                        peek_ab_full_status = ab_consumer.try_wait()
                        acc_pipeline.producer_acquire(acc_producer_state)

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        iket.range_push("mma_ab_wait")
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()
                        iket.range_pop()

                        s2t_stage_coord = (None, None, None, None, handle.index)
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_stage_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[s2t_stage_coord],
                            tCtSFB_compact_s2t,
                        )

                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            [tCrA[tile_crd], tCtSFA],
                            [tCrB[tile_crd], tCtSFB],
                            tCtAcc,
                        )
                        handle.release()

                    if k_tile_cnt > 0:
                        acc_pipeline.producer_commit(acc_producer_state)
                if k_tile_cnt > 0:
                    acc_producer_state.advance()

                iket.range_pop()

                work_tile_info = sched_consumer.consume_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ════════════════════════════════════════════════════════════════════
        # Dedicated preact-C TMA-load warp (c_load_warp_id) — c_pipeline PRODUCER
        # ════════════════════════════════════════════════════════════════════
        #
        # Mirrors the reference's epilog_load_tma warp: consume the same work
        # tiles in lockstep, and for each Linear1 (dfc2) tile TMA-load gate
        # (epi-tile 2*s) then up (2*s+1) into successive c_pipeline stages.
        if warp_idx == self.c_load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

            c_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_c_pipe_stage
            )
            thr_mma_c = tiled_mma.get_slice(mma_tile_coord_v)
            preact_epi_tile = self.epilogue.preact_epi_tile
            c_subtile_cnt = self.cta_tile_shape_mnk[1] // 32  # 8

            work_tile_info = sched_consumer.consume_work()
            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )
                if is_phase_linear1:
                    real_preact, _ = ext.get_gmem_tensor(
                        "c", tma_tensor_fc1_preact, work_tile_info
                    )
                    gC_mnl = cute.local_tile(
                        real_preact, cute.slice_(self.mma_tiler, (None, None, 0)),
                        (None, None, None),
                    )
                    tCgC = thr_mma_c.partition_C(gC_mnl)
                    gC_epi = cute.flat_divide(
                        tCgC[((None, None), 0, 0, None, None, None)], preact_epi_tile
                    )
                    bGS_sPre, bGS_gC = cpasync.tma_partition(
                        tma_atom_fc1_preact, 0, cute.make_layout(1),
                        cute.group_modes(sPre, 0, 2),
                        cute.group_modes(gC_epi, 0, 2),
                    )
                    mma_m_coord = work_tile_info.tile_m_idx // cutlass.Int32(self.atom_thr_size)
                    mma_n_coord = work_tile_info.tile_n_idx * cutlass.Int32(2)
                    bGS_gC = bGS_gC[(None, None, None, mma_m_coord, mma_n_coord, 0)]
                    bGS_gC = cute.group_modes(bGS_gC, 1, cute.rank(bGS_gC))

                    for i in cutlass.range(0, c_subtile_cnt, 1, unroll=1):
                        subtile_idx = cutlass.Int32(i)
                        # gate (2*subtile) then up (2*subtile+1)
                        c_pipeline.producer_acquire(c_producer_state)
                        c_bar = c_pipeline.producer_get_barrier(c_producer_state)
                        c_slot = 2 * c_producer_state.index
                        cute.copy(
                            tma_atom_fc1_preact,
                            bGS_gC[(None, subtile_idx * cutlass.Int32(2) + cutlass.Int32(0))],
                            bGS_sPre[(None, c_slot)],
                            tma_bar_ptr=c_bar
                        )
                        cute.copy(
                            tma_atom_fc1_preact,
                            bGS_gC[(None, subtile_idx * cutlass.Int32(2) + cutlass.Int32(1))],
                            bGS_sPre[(None, c_slot + 1)],
                            tma_bar_ptr=c_bar,
                        )
                        c_producer_state.advance()

                work_tile_info = sched_consumer.consume_work()

            c_pipeline.producer_tail(c_producer_state)

        # ════════════════════════════════════════════════════════════════════
        # Epilogue warps (warps 0-3)
        # ════════════════════════════════════════════════════════════════════
        #
        # Fully delegated to ``self.epilogue.run(...)`` -- the epilogue owns
        # the entire 2-phase task-tile loop.
        if warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_alloc(self.epi_reg_cnt)
            epi_warp_idx = warp_idx

            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_tensor = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)

            # The epilogue is the preact c_pipeline CONSUMER (the dedicated
            # c_load warp is the producer); it reads gate/up from sPre stages.
            _run_kwargs = dict(
                tmem_acc_tensor=acc_tensor,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
                sched_ext=ext,
                gmem_fc1_output=tma_tensor_grad_y1,
                gmem_fc1_output_sf=fc1_output_sf_gemm,
                tma_atom_fc1_recompute=tma_atom_fc1_recompute,
                gmem_fc1_recompute=tma_tensor_fc1_recompute,
                gmem_fc1_recompute_sf=fc1_recompute_sf_gemm,
                tma_atom_fc1_col_output=tma_atom_fc1_col_output,
                gmem_fc1_col_output=tma_tensor_fc1_col_output,
                gmem_fc1_col_output_sf=fc1_col_output_sf_gemm,
                smem_preact_buffer=sPre,
                c_pipeline=c_pipeline,
                c_num_stage=num_c_pipe_stage,
                smem_d_buffer=sD,
                d_pipeline=d_pipeline,
                d_num_stage=num_d_stage,
                tma_atom_grad_y1=tma_atom_grad_y1,
                gmem_topk_scores=topk_scores,
                gmem_fc2_output=fc2_output_gemm,
                gmem_fc1_done_counter=fc1_done_counter,
                warp_idx=epi_warp_idx,
                tidx=tidx,
                alpha=cutlass.Float32(1.0),
                norm_const=cutlass.Float32(1.0),
                gmem_beta=beta,
                gmem_dprob=dprob,
            )

            # MegaMoE: pass token_comm_args only when it is a real bundle (not
            # None).  Passing Python None explicitly to @cute.jit methods
            # triggers a CuteDSL codegen issue; const_expr dispatch avoids any
            # None-as-JIT-argument path.
            if cutlass.const_expr(self.enable_token_comm):
                # MegaMoE (push model): bridge next's TokenComm accessors + peer mapper into
                # the dGLU epilogue's Fc2OutputDest peer-store expectations for grad_x combine.
                _epi_comm = _EpilogueCommView(
                    token_src_metadata=self.token_comm.token_src_metadata_tensor(
                        self._mega_device_workspace
                    ),
                    combine_output=mega_pre_reduced_activation,
                    dprob_output=dprob,
                    peer_rank_ptr_mapper=mega_peer_rank_ptr_mapper,
                    fc2_output_sf=self.token_comm.fc2_activation_sf_tensor(self._mega_device_workspace),
                    fc2_done_counter=self.token_comm.fc2_done_counter_tensor(self._mega_device_workspace),
                    fc2_output_workspace=self.token_comm.fc2_activation_tensor(self._mega_device_workspace),
                )
                self.epilogue.run(**_run_kwargs, token_comm_args=_epi_comm)
            elif cutlass.const_expr(token_comm_args is not None):
                self.epilogue.run(**_run_kwargs, token_comm_args=token_comm_args)
            else:
                self.epilogue.run(**_run_kwargs)

            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.fence_acq_rel_sys()

        # ════════════════════════════════════════════════════════════════════
        # Dispatch / token_back warps hook (warps 8-11 [+ 12-15]; MegaMoE-only)
        # ════════════════════════════════════════════════════════════════════
        #
        # ``enable_token_comm=False`` → these warps don't exist (lean base has 9
        # warps), so the guard is const_expr-eliminated in the lean path.
        # NOTE: c_load now lives ABOVE the transfer block (warp 12 or 16), so the
        # gate must be UPPER-bounded at the last transfer warp — otherwise the
        # c_load warp (already run above) would re-enter the dispatch body.
        if cutlass.const_expr(self.enable_token_comm):
            _last_transfer_warp = (
                self.token_back_warp_id[-1]
                if self.token_back_standalone
                else self.dispatch_warp_id[-1]
            )
            if (warp_idx >= self.dispatch_warp_id[0]) & (warp_idx <= _last_transfer_warp):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)
                lane_idx_for_dispatch = cute.arch.lane_idx()
                if cutlass.const_expr(self.token_back_standalone):
                    if warp_idx < self.token_back_warp_id[0]:
                        self.token_comm_hook_dispatch_warp_body(
                            token_comm_args,
                            token_comm_storage,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx_for_dispatch,
                            tidx=tidx,
                        )
                    else:
                        self.token_comm_hook_token_back_warp_body(
                            token_comm_args,
                            token_comm_storage,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx_for_dispatch,
                            tidx=tidx,
                        )
                else:
                    self.token_comm_hook_dispatch_warp_body(
                        token_comm_args,
                        token_comm_storage,
                        warp_idx=warp_idx,
                        lane_idx=lane_idx_for_dispatch,
                        tidx=tidx,
                    )

            # ════════════════════════════════════════════════════════════════════
            # Kernel tail hook (MegaMoE-only; lean base = no-op)
            # ════════════════════════════════════════════════════════════════════
            lane_idx = cute.arch.lane_idx()
            self.token_comm_hook_kernel_tail(
                token_comm_args,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )
