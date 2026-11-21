# Copyright (c) 2025-2026, QuACK team.
# Copyright (c) 2026, Tri Dao.
# Copyright (c) 2025, Wentao Guo, Tri Dao.
# Copyright (c) 2025-2026, Vijay Thakkar, Tri Dao.
# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py
"""Minimal SM100 GEMM collective base (GemmDefaultSm100, flattened) adapted from quack-kernels 0.5.0."""

import enum
import math
from enum import IntEnum
from typing import Callable, Literal, NamedTuple, Optional, Sequence, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Int32, Float32, Boolean, const_expr
from cutlass.cutlass_dsl import NumericMeta
from cutlass.utils import LayoutEnum

from . import copy_utils
from . import sm100_utils as quack_sm100_utils


class NamedBarrierGemm(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    # For mainloop load warps to signal that the epilogue load warp can start.
    # This is to avoid loading C too early, interfering with loading A and B.
    EpilogueLoad = enum.auto()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()
    TmemPtr = enum.auto()


class RoundingMode(IntEnum):
    """Rounding modes for epilogue dtype downconversion.

    RN — Round to nearest even (default hardware behavior)
    RS — Stochastic rounding (SM100+ only, BF16 output only)
    """

    RN = 0
    RS = 1


StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


def _namedtuple_new_from_mlir_values(self, values):
    """Generic __new_from_mlir_values__ for NamedTuples.

    Applied to NamedTuple classes via the ``@mlir_namedtuple`` decorator.

    Fields that are None or Constexpr (StaticTypes) are preserved from ``self`` (the compile-time
    template). Only non-static fields consume MLIR values. Multi-value fields (e.g. cute.Tensor)
    consume the correct number of values via ``cutlass.new_from_mlir_values``.

    Constexpr fields (annotated ``cutlass.Constexpr[T]``) are baked into the compiled kernel via
    a converter patch (see above). At call time, pass None for these fields.
    """
    from cutlass.base_dsl.typing import get_mlir_types

    values = list(values)
    new_fields = []
    for field_val in self:
        if field_val is None or isinstance(field_val, StaticTypes):
            new_fields.append(field_val)
        else:
            n_items = len(get_mlir_types(field_val))
            new_fields.append(cutlass.new_from_mlir_values(field_val, values[:n_items]))
            values = values[n_items:]
    return self.__class__(*new_fields)


def mlir_namedtuple(cls):
    """Decorator that adds MLIR value reconstruction to a NamedTuple class.

    Usage::

        @mlir_namedtuple
        class MyArgs(NamedTuple):
            tensor_arg: cute.Tensor
            const_arg: cutlass.Constexpr[int] = 0
    """
    cls.__new_from_mlir_values__ = _namedtuple_new_from_mlir_values
    return cls


@mlir_namedtuple
class VarlenArguments(NamedTuple):
    mCuSeqlensM: Optional[cute.Tensor] = None
    mCuSeqlensK: Optional[cute.Tensor] = None
    mAIdx: Optional[cute.Tensor] = None


class EpiOp:
    """Base class for composable epilogue operations."""

    def __init__(self, name):
        self.name = name

    def epi_m_major_score(self, arg_tensor, gemm):
        """Preference for epilogue subtile order. Positive prefers M-major, negative N-major."""
        return 0

    def is_tile_load(self):
        """Whether this op is a tile-sized epilogue input loaded through the C pipeline."""
        return False


class Scalar(EpiOp):
    """Loads a scalar value or device pointer once per tile. No smem."""

    def __init__(self, name, dtype=None):
        super().__init__(name)
        self.dtype = dtype


class VecLoad(EpiOp):
    """Base class for broadcast vector loads (row or col) via cp_async."""

    dim = None  # 0 for col (M), 1 for row (N)

    def epi_m_major_score(self, arg_tensor, gemm):
        # It costs more registers (say 4x) to keep rowvec in register vs keeping colvec in register
        return 4 if self.dim == 1 else -1


class RowVecLoad(VecLoad):
    """Loads a row vector (N,) via cp_async, broadcasts along M with stride (0,1)."""

    dim = 1


class ColVecLoad(VecLoad):
    """Loads a col vector (M,) via cp_async, broadcasts along N with stride (1,0)."""

    dim = 0


class GemmDefaultSm100:
    """QuACK's GemmDefaultSm100 (GemmDefaultEpiMixin -> ComposableEpiMixin -> GemmSm100 -> GemmTmaBase ->
    GemmBase) flattened to the members the CuteMoK collectives reach; each resolves as it did in that MRO."""

    arch = 100

    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecLoad("mColVecBroadcast"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],  # ignored for now
        mma_tiler_mnk: Union[Tuple[int, int], Tuple[int, int, int]],
        cluster_shape_mnk: Tuple[int, int, int],
        sf_vec_size: Optional[int] = None,
        gather_A: bool = False,
        use_tma_gather: bool = False,
        use_clc_persistence: bool = True,
        concat_layout: tuple | None = None,
        use_pdl: bool = True,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mnk: The (M, N) or (M, N, K) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mnk: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mnk: (M, N) or (M, N, K) shape of the MMA tile.
            If only (M, N) is given, K defaults to 4 * instruction K.
        :type mma_tiler_mnk: Union[Tuple[int, int], Tuple[int, int, int]]
        :param cluster_shape_mnk: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mnk: Tuple[int, int]
        """

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.sf_vec_size = sf_vec_size
        self.blockscaled = sf_vec_size is not None
        assert len(mma_tiler_mnk) in [2, 3], "MMA tiler must be (M, N) or (M, N, K)"
        valid_2cta_m = (128, 256) if not self.blockscaled else (256,)
        self.use_2cta_instrs = cluster_shape_mnk[0] % 2 == 0 and mma_tiler_mnk[0] in valid_2cta_m
        self.cluster_shape_mnk = cluster_shape_mnk
        assert cluster_shape_mnk[2] == 1, "Cluster shape K must be 1"
        # K dimension: if user provides 3 values, use their K; otherwise default in _setup_attributes
        if len(mma_tiler_mnk) == 3:
            self.mma_tiler = tuple(mma_tiler_mnk)
        else:
            self.mma_tiler = (*mma_tiler_mnk, 0)
        self.is_persistent = True
        self.use_clc_persistence = use_clc_persistence
        self.epi_m_major = True
        self.gather_A = gather_A
        self.concat_layout = concat_layout or ()
        self.use_tma_gather = use_tma_gather
        self.use_pdl = use_pdl
        if gather_A:
            assert cluster_shape_mnk[1] == 1, "Cluster shape N must be 1 for gather A "
        if use_tma_gather:
            assert gather_A, "TMA gather requires gather_A=True"

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.num_ab_load_warps = 1 if not self.gather_A else 4
        self.occupancy = 1
        # Set specialized warp ids
        self.epi_warps_per_accumulator = 4
        num_epi_warps = self.epi_warps_per_accumulator
        self.epilog_warp_id = tuple(range(num_epi_warps))
        self.mma_warp_id = len(self.epilog_warp_id)
        self.ab_load_warp_id = self.mma_warp_id + 1
        self.epi_load_warp_id = self.ab_load_warp_id + self.num_ab_load_warps
        self.scheduler_warp_id = self.epi_load_warp_id + 1
        # For gather_A: separate A-index prefetch warp (was the empty warp)
        self.a_prefetch_warp_id = self.scheduler_warp_id + 1 if self.gather_A else None
        self.num_epi_warps = len(self.epilog_warp_id)
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        # Register reallocation for gather_A (3 warp groups, 504 regs total, 168 per WG default).
        # Heavy epilogues (e.g. colvec_reduce in DGated) override these to avoid register spilling.
        # Without gather_A there are only 2 WGs (512 total, 256 per WG = max), no reallocation needed.
        self.num_regs_other = 120
        self.num_regs_epi = 256
        extra_warp_ids = (self.a_prefetch_warp_id,) if self.gather_A else ()
        self.threads_per_cta = cute.arch.WARP_SIZE * (
            self.num_ab_load_warps
            + len(
                (
                    self.mma_warp_id,
                    self.epi_load_warp_id,
                    self.scheduler_warp_id,
                    *self.epilog_warp_id,
                    *extra_warp_ids,
                )
            )
        )
        # Multiple of 4 warps to increase/decrease number of registers
        assert self.threads_per_cta % 128 == 0

    def epi_smem_warp_shape_mnk(self):
        # Mirrors cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape:
        # the epilogue tmem layout uses two M warps and two N warps when the
        # per-CTA M tile is 64 and the kernel uses 2-CTA instructions.
        warp_m, warp_n = (2, 2) if self.cta_tile_shape_mnk[0] == 64 and self.use_2cta_instrs else (4, 1)
        return (warp_m, warp_n, 1)

    def _setup_attributes(self, epilogue_args: EpilogueArguments, varlen_args: VarlenArguments):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        self.epi_m_major = self.resolve_epi_m_major(epilogue_args)

        # Compute mma instruction shapes
        mma_inst_bits_k = 256
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        mma_inst_shape_n = self.mma_tiler[1] if self.mma_tiler[1] <= 256 else self.mma_tiler[1] // 2
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            mma_inst_shape_n,
            mma_inst_bits_k // self.a_dtype.width,
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        # Configure tiled mma
        if const_expr(not self.blockscaled):
            self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_inst_shape_mnk[:2],
            )
            self.tiled_mma_sfb = None
        else:
            self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape_mnk[:2],
            )
            self.tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                tcgen05.CtaGroup.ONE,
                self.mma_inst_shape_mnk_sfb[:2],
            )

        # Compute mma/cluster/tile shapes
        if self.mma_tiler[2] > 0:
            assert self.mma_tiler[2] % self.mma_inst_shape_mnk[2] == 0, (
                f"MMA tiler K ({self.mma_tiler[2]}) must be divisible by " f"MMA instruction K ({self.mma_inst_shape_mnk[2]})"
            )
            mma_inst_tile_k = self.mma_tiler[2] // self.mma_inst_shape_mnk[2]
        else:
            mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        if const_expr(self.blockscaled):
            self.mma_tiler_sfb = (
                self.mma_inst_shape_mnk_sfb[0],
                self.mma_inst_shape_mnk_sfb[1],
                self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
            )
        else:
            self.mma_tiler_sfb = None
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(self.tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        if const_expr(self.blockscaled):
            self.cta_tile_shape_mnk_sfb = (
                self.mma_tiler_sfb[0] // cute.size(self.tiled_mma.thr_id.shape),
                self.mma_tiler_sfb[1],
                self.mma_tiler_sfb[2],
            )
        else:
            self.cta_tile_shape_mnk_sfb = None

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma.thr_id.shape,),
        )
        if const_expr(self.blockscaled):
            self.cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (self.tiled_mma_sfb.thr_id.shape,),
            )
        else:
            self.cluster_layout_sfb_vmnk = None

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        if self.gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        if const_expr(self.blockscaled):
            self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
            self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        tile_load_layout = None
        tile_load_dtype = None
        # If TileLoad exists without C, use the first non-None tile-load tensor as
        # the C-like input for SM100's epilogue tile shape. Multiple TileLoads
        # share the same epi_tile shape.
        for op in getattr(self, "_epi_ops", ()):
            if op.is_tile_load():
                tile_load_tensor = getattr(epilogue_args, op.name, None)
                if tile_load_tensor is not None:
                    tile_load_layout = LayoutEnum.from_tensor(tile_load_tensor)
                    tile_load_dtype = tile_load_tensor.element_type
                    break
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.d_layout if self.d_layout is not None else LayoutEnum.ROW_MAJOR,
            self.d_dtype if self.d_dtype is not None else cutlass.BFloat16,
            layout_c=self.c_layout if self.c_layout is not None else tile_load_layout,
            elem_ty_c=self.c_dtype if self.c_dtype is not None else tile_load_dtype,
        )
        # TMA store tile starts must stay aligned when advancing across CTA-N tiles.
        # There's a bug w compute_epilogue_tile_shape (as of cutlass-dsl 4.4.2) where if
        # tile_n = 224 and there's C, it will set epi_tile to (128, 64).
        if const_expr(self.cta_tile_shape_mnk[1] % cute.size(self.epi_tile[1]) != 0):
            warp_n = 2 if (self.cta_tile_shape_mnk[0] == 64 and self.use_2cta_instrs) else 1
            epi_tile_n = math.gcd(self.cta_tile_shape_mnk[1], cute.size(self.epi_tile[1]))
            epi_tile_n_layout = cute.make_layout((epi_tile_n // warp_n, warp_n), stride=(1, self.cta_tile_shape_mnk[1] // warp_n))
            self.epi_tile = (self.epi_tile[0], cute.coalesce(epi_tile_n_layout))

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        prefetch_A_idx = None if not self.gather_A else ("varlen_m" if varlen_args.mCuSeqlensM is not None else "varlen_k")
        (
            self.num_acc_stage,
            self.ab_stage,
            self.epi_stage,
            self.epi_c_stage,
        ) = self._compute_stages(
            self.tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            self.d_dtype,
            self.c_dtype,
            self.d_layout,
            self.c_layout,
            epilogue_args,
            prefetch_A_idx,
            cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}"),  # smem_capacity
            self.occupancy,
            self.epi_smem_warp_shape_mnk(),
        )
        self.sched_stage = 1
        self.a_prefetch_stage = 0 if not self.gather_A else (2 if varlen_args.mCuSeqlensM is not None else self.ab_stage)

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(self.tiled_mma, self.mma_tiler, self.a_dtype, self.ab_stage)
        self.a_smem_load_layout_staged = self.a_smem_layout_staged
        if const_expr(self.gather_A):
            if const_expr(self.use_tma_gather):
                self.a_smem_load_layout_staged = quack_sm100_utils.make_smem_layout_tma_gather_a(self.tiled_mma, self.mma_tiler, self.a_dtype, self.ab_stage)
            else:
                self.a_smem_load_layout_staged = quack_sm100_utils.make_smem_layout_cpasync_a(self.tiled_mma, self.mma_tiler, self.a_dtype, self.ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(self.tiled_mma, self.mma_tiler, self.b_dtype, self.ab_stage)
        self.epi_smem_layout_staged = None
        if const_expr(self.d_dtype is not None):
            self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.d_dtype, self.d_layout, self.epi_tile, self.epi_stage)
        self.epi_c_smem_layout_staged = None
        if const_expr(self.c_dtype is not None):
            self.epi_c_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.epi_c_stage)
        if const_expr(self.blockscaled):
            self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                self.ab_stage,
            )
            self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                self.ab_stage,
            )
        else:
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged = None, None

        # Compute the number of tensor memory allocation columns
        if const_expr(not self.blockscaled):
            self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(self.tiled_mma, self.mma_tiler, self.num_acc_stage)
        else:
            self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        # Overlapping accumulator and scaling factor in tmem, targetting the case tile_n == 256
        # For iter 0, 2, ..., accum is in col 0...255 and SF are in col 256...256+SF_size.
        # For iter 1, 3, ..., accum is in col 256...511 and SF are in col 0...0+SF_size.
        # During the epilogue, we release acc_pipeline after being done with @SF_size columns.
        # In the cute-dsl example,
        # https://github.com/NVIDIA/cutlass/blob/08185b9c3e90510ee2b656662ed0d53b06d28157/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py#L369
        # instead the 2 stages of accum are in col 0...255 and 256-SF_size...512-SF_size, and
        # the SF are in 512-SF_size...511. The 2 accum stages overlap, so in the epilogue,
        # they alternate the direction of epi tiles (from right to left, then from left to right)
        # to release acc_pipeline early.
        # The two approaches perform about the same.
        self.overlap_accum_sf = self.blockscaled and self.num_acc_stage == 1
        if const_expr(self.overlap_accum_sf):
            num_sf_tmem_cols = (
                (cute.ceil_div(self.cta_tile_shape_mnk[0], 128) + cute.ceil_div(self.cta_tile_shape_mnk[1], 128))
                * 4  # 4 cols per stage
                * (self.mma_inst_shape_mnk[2] // self.sf_vec_size)
            )
            self.iter_acc_early_release = num_sf_tmem_cols // cute.size(self.epi_tile[1])
        else:
            self.iter_acc_early_release = -1

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        acc_pipeline: cutlass.pipeline.PipelineAsync,
        ab_consumer_state: cutlass.pipeline.PipelineState,
        acc_producer_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        is_leader_cta: Boolean,
        cta_rank_in_cluster: Int32,
        tCtSFA: Optional[cute.Tensor] = None,
        tCtSFB: Optional[cute.Tensor] = None,
        copy_s2t_sfa: Optional[Callable] = None,
        copy_s2t_sfb: Optional[Callable] = None,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState, cute.TiledMma]:
        blockscaled = const_expr(copy_s2t_sfa is not None)
        if const_expr(blockscaled):
            assert all(x is not None for x in (tCtSFA, tCtSFB))
            assert copy_s2t_sfb is not None
        # If gather_A and use_2cta_instrs, the cp.async for the non-leader CTA will
        # arrive at an mbarrier on the non-leader CTA side, then the mma warp of the non-leader
        # CTA will wait for that then arrive at the mbarrier on the leader CTA.
        need_nonleader_cta = const_expr(self.gather_A and self.use_2cta_instrs and not self.use_tma_gather)
        # Peek (try_wait) AB buffer full for k_tile = 0
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt and (is_leader_cta or need_nonleader_cta):
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Wait for accumulator buffer empty
        if is_leader_cta:
            acc_pipeline.producer_acquire(acc_producer_state)
        # Reset the ACCUMULATE field for each tile
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        # Mma mainloop
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            if const_expr(need_nonleader_cta):
                if not is_leader_cta:
                    ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                    with cute.arch.elect_one():
                        # The odd CTA signals the even CTA
                        ab_pipeline.sync_object_full.arrive_mbarrier(ab_consumer_state.index, dst_rank=cta_rank_in_cluster & 0xFE)
            if is_leader_cta:
                # Conditionally wait for AB buffer full
                ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                #  Copy SFA/SFB from smem to tmem
                if const_expr(blockscaled):
                    copy_s2t_sfa(ab_consumer_state.index)
                    copy_s2t_sfb(ab_consumer_state.index)
                for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    k_blk_coord = (None, None, k_blk_idx, ab_consumer_state.index)
                    if const_expr(blockscaled):
                        # Set SFA/SFB tensor to tiled_mma
                        sf_kblock_coord = (None, None, k_blk_idx)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)
                    cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                # Async arrive AB buffer empty
                ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
            # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt and (is_leader_cta or need_nonleader_cta):
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Async arrive accumulator buffer full
        if is_leader_cta:
            acc_pipeline.producer_commit(acc_producer_state)
        acc_producer_state.advance()
        # If we don't return the tiled_mma, we get compiler error
        # "operand #0 does not dominate this use"
        return ab_consumer_state, acc_producer_state, tiled_mma

    @cute.jit
    def epi_load_acc_subtile(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tiled_copy_r2s: cute.TiledCopy,
        tTR_tAcc: cute.Tensor,
        tTR_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        epi_coord: [int, int],
        acc_pipeline: pipeline.PipelineAsync,
        acc_consumer_state: pipeline.PipelineState,
        acc_release_idx: int,
        clear_acc: Boolean = False,
    ):
        if not clear_acc:
            # Load accumulator from tensor memory buffer to register
            cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, epi_coord], tTR_rAcc)
            tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
            tRS_rD.store(tRS_rAcc.load())
        else:
            tRS_rD.fill(0.0)
        assert epi_coord[0] == 0  # For Sm100, we assume epi_M = 1
        if epi_coord[1] == acc_release_idx:
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)

    def epilog_tmem_copy_and_partition(
        self,
        tidx: Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout if self.d_layout is not None else LayoutEnum.ROW_MAJOR,
            self.d_dtype if self.d_dtype is not None else cutlass.BFloat16,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        cAcc_epi = cute.flat_divide(cAcc, epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(tTR_cAcc[None, None, None, 0, 0].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_store_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        d_layout: Optional[LayoutEnum],
        dtype: Optional[Type[cutlass.Numeric]],
        tTR_rD: cute.Tensor,
        sD: cute.Tensor,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rD: The partitioned accumulator tensor
        :type tTR_rD: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: Int32
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rD, tRS_sD) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rD: The partitioned tensor C (register source)
            - tRS_sD: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            d_layout if d_layout is not None else LayoutEnum.ROW_MAJOR,
            dtype if dtype is not None else cutlass.BFloat16,
            self.acc_dtype,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD) if sD is not None else None
        # (R2S, R2S_M, R2S_N)
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param num_acc_stage: The stage of the accumulator tensor.
        :type num_acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        return num_tmem_alloc_cols

    @cute.jit
    def load_tma(
        self,
        pipeline: cutlass.pipeline.PipelineAsync,
        producer_state: cutlass.pipeline.PipelineState,
        copy_fns: Sequence[Optional[Callable]],
        k_tile_cnt: Int32,
    ) -> cutlass.pipeline.PipelineState:
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt.
        peek_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_empty_status = pipeline.producer_try_acquire(producer_state)
        # TMA load
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Wait for A/B buffers to be empty before loading into them.
            # Also sets the transaction barrier for the A/B buffers.
            pipeline.producer_acquire(producer_state, peek_empty_status)
            tma_bar_ptr = pipeline.producer_get_barrier(producer_state)
            smem_idx = producer_state.index
            for copy_fn in copy_fns:
                if const_expr(copy_fn is not None):
                    copy_fn(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            # Mainloop pipeline's producer commit is a NOP for TMA pipelines.
            pipeline.producer_commit(producer_state)
            producer_state.advance()
            peek_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_empty_status = pipeline.producer_try_acquire(producer_state)
        return producer_state

    def epilog_gmem_copy_and_partition(
        self,
        atom: cute.CopyAtom | cute.TiledCopy,
        mD_mn: cute.Tensor,
        tile_shape_mn: cute.Tile,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        gD = cute.local_tile(mD_mn, tile_shape_mn, tile_coord_mnkl[:2])  # (bM, bN)
        tDgD_for_tma_partition = cute.zipped_divide(gD, epi_tile)
        is_s2g = isinstance(atom.op, (cpasync.CopyBulkTensorTileS2GOp, cpasync.CopyReduceBulkTensorTileS2GOp))
        src_tensor, dst_tensor = (sD, tDgD_for_tma_partition) if is_s2g else (tDgD_for_tma_partition, sD)
        return copy_utils.tma_get_copy_fn(
            atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=src_tensor,
            dst_tensor=dst_tensor,
        )

    def make_epi_store_pipeline(self):
        num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        epi_store_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_epi_threads)
        return pipeline.PipelineTmaStore.create(num_stages=self.epi_stage, producer_group=epi_store_producer_group)

    @staticmethod
    def _make_tma_epi_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
        op_type: Literal["store", "load", "add"],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for storing D or loading C."""
        assert op_type in ["load", "store", "add"]
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile)
        op = {
            "load": cpasync.CopyBulkTensorTileG2SOp(),
            "store": cpasync.CopyBulkTensorTileS2GOp(),
            "add": cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD),
        }[op_type]
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(op, tensor_d, epi_smem_layout, d_cta_v_layout)
        return tma_atom_d, tma_tensor_d

    def resolve_epi_m_major(self, args):
        # Runs inside _setup_attributes, before epi_to_underlying_arguments,
        # because epi_m_major drives epi_tile / smem layout choices that
        # epi_to_underlying_arguments later consumes. self._epi_ops is still
        # the class-level schema at this point, so we filter inline from args
        # to keep op.epi_m_major_score's non-None invariant.
        score = 0
        for op in type(self)._epi_ops:
            arg = getattr(args, op.name, None)
            if arg is not None:
                score += op.epi_m_major_score(arg, self)
        return score >= 0
