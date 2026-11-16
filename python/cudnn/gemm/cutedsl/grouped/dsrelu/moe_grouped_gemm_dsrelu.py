# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The kernel class in this file is derived from the BF16 grouped GEMM dGLU kernel of
# NVIDIA/cudnn-frontend@b03c3f3321f5a66b0582db44789f4b5be212aca5
# (python/cudnn/gemm/cutedsl/grouped/dglu/moe_grouped_gemm_dglu_dbias.py). Its epilogue follows the
# clamped dSReLU epilogue of the same commit's
# grouped/dsrelu/moe_blockscaled_grouped_gemm_dsrelu_quant.py. Synchronization follows
# python/cudnn/AGENTS.md instead of the dGLU kernel: the warp-role named barriers are unaligned, the
# tile-info and Z_save stages are released through tile_dsl.consume and, with discrete weights, the
# A/B TMA warp acquires each per-expert B descriptor before it loads through it (see the kernel class
# docstring).

"""BF16 MoE grouped GEMM ``dH = G @ B_e`` fused with the weighted squared-ReLU backward (SM100).

Per expert ``e`` and padded routed row ``m``::

    dH_acc[m, n]  = sum_k A[m, k] * B_e[n, k]                          BF16 tcgen05 MMA, FP32, never rounded
    t             = tanh(max(float(Z_save[m, n]), 0) * fp32(1/s))      FP32, the forward kernel's PTX
    c             = s * t,  A = c * c
    dZ[m, n]      = bf16_rn(((dH_acc * p[m]) * (c + c)) * (1 - t * t))
    dp_part[m, j] = sum over the columns n of N-tile j of A * dH_acc   (FP32, one writer)
    H_rc[m, n]    = bf16_rn(p[m] * A)                                  (only when requested)
    dp[m]         = sum_j dp_part[m, j] in increasing j                (FP32, _dprob_fold.py)
                  or, with dprob_atomic, FP32 atomic adds of dp_part[m, j] into dp[m] in the
                  epilogue, in whatever order the N tiles finish

``A`` is the gradient ``G`` of the FC2 output, ``B_e`` the FC2 weight viewed as ``(N, K)`` and
``Z_save`` (operand ``c``) the BF16 pre-activation the forward kernel wrote. ``t`` comes from the
forward kernel's device helper :func:`clamped_srelu_t_f32` (same inline PTX: NaN-propagating relu,
multiply by ``fp32(1/s)``, Cephes tanh or ``tanh.approx`` capped at 1) and ``c``, ``A`` and
``p * A`` are explicit ``mul.rn.f32``, the instruction sequence of the forward ``H``. So ``H_rc``
equals the forward ``H`` bit for bit when the forward ran with act_input == "saved", and the
derivative uses the same ``t`` as the forward.

Plain squared-ReLU (``clamp_scale=None``): ``r = max(float(Z_save), 0)`` from the forward kernel's
:func:`srelu_relu_f32`, ``A = r * r``, ``dZ = bf16_rn((dH_acc * p[m]) * (r + r))``; ``dp_part``
and ``H_rc`` as above with this ``A``.

Weights come either as a device array of E per-expert base addresses (DISCRETE mode; a helper
kernel builds the per-expert B TMA descriptors on the device) or as one ``(N, K, E)`` tensor
(DENSE mode), K-major or N-major.

Kernel structure (unchanged from the dGLU kernel): a persistent MoE tile scheduler warp, an A/B TMA
warp, a tcgen05 MMA warp with a double-buffered TMEM accumulator, a TMA warp that streams the saved
pre-activation C = Z_save into shared memory one 128x32 epilogue subtile at a time, and four
epilogue warps. Removed relative to dGLU: the gate/linear column pairing, alpha/beta scaling,
linear_offset and dbias. Added: the dSReLU math, an optional second TMA store output (H_rc) and the
deterministic dprob slots; the dGLU kernel's per-tile dprob atomic adds remain as dprob_atomic=True.

Determinism of dprob: each epilogue thread owns one row of a 128-row CTA tile; it sums its row of
every 32-column subtile in a fixed program order (two interleaved partial sums, even/odd column,
combined once per subtile) and the subtiles in increasing order. This kernel never reverses the
subtile loop (two accumulator stages, no overlapping accumulator), so it needs no "slot parking":
the block-scaled dSReLU kernel (``moe_blockscaled_grouped_gemm_dsrelu_quant.py``, as does the
block-scaled dGLU kernel) parks each subtile's partial (``dprob_slot_parking``) only because its
overlapping accumulator reverses the loop on alternate tiles. The tile's partial goes to slot
``(row, tile_n)`` of a zero-initialized
``[M, num_n_slots]`` FP32 buffer. Exactly one thread of one CTA writes each slot, so the FP32
atomic add used for the write is uncontended and stores the partial bit-for-bit. The slots are then
summed in increasing ``j`` by ``_dprob_fold.py``.

``dprob_atomic=True`` instead adds every tile's partial straight into a ``[M, 1]`` FP32 buffer with
the same atomic, as the dGLU kernel does: no ``[M, num_n_slots]`` buffer and no second kernel, but
the N tiles of a row are summed in the order their epilogues reach the atomic, so dprob may differ
run to run in the last FP32 bits. dZ and H_rc are unaffected.
"""

import math
import struct
from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols, get_smem_capacity_in_bytes
from cudnn.frost.tile_dsl.barrier import named_barrier_sync_unaligned
from cudnn.frost.tile_dsl.consume import consume_then_arrive, release_zero
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.typing import AddressSpace, Float32, Int32
from cutlass.cutlass_dsl import T

from ..moe_kernel_helpers import (
    atomic_add_float32,
    can_implement_bf16_grouped_gemm,
    compute_grid,
)
from ..moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from ..moe_sched_extension import (
    ContiguousGroupedGemmSchedExtension,
    DiscreteWeightGroupedGemmSchedExtension,
)
from ..moe_utils import MoEWeightMode, TensormapWorkspace, acquire_tma_desc, store_tma_desc

# The forward kernel's t (tanh-clamped) and r (plain), so that H_rc reproduces H bit for bit.
from ..srelu.moe_grouped_gemm_srelu import clamped_srelu_t_f32, srelu_relu_f32

__all__ = ["MoEGroupedGemmDsreluBf16Kernel"]

# Per-expert token counts must be multiples of this. It is also the M extent of one 2-CTA cluster
# tile, so a cluster tile never straddles two experts.
FIX_PAD_SIZE = 256


# ---------------------------------------------------------------------------------------------
# Small device helpers
# ---------------------------------------------------------------------------------------------


def mul_rn_f32(a, b, *, loc=None, ip=None) -> Float32:
    """IEEE round-to-nearest FP32 product (``mul.rn.f32``).

    Spelled out in PTX so that no fast-math reassociation or FMA contraction in the DSL / LLVM /
    ptxas pipeline can change ``c = s * t``, ``A = c * c`` or ``p * A``: these must be the same
    instructions as in the forward kernel (which uses ``mul.rn.f32`` for all three).
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _to_fp32(value: float) -> float:
    """Round a Python float to the nearest FP32 value."""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


# ---------------------------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------------------------


class MoEGroupedGemmDsreluBf16Kernel:
    """Persistent BF16 MoE grouped GEMM ``dH = G @ W2_e`` with the weighted clamped dsReLU epilogue.

    The scheduling, TMA loads (A, B and the saved pre-activation C), pipelines, tcgen05 MMA, TMEM
    handling and persistent tile scheduler are those of the BF16 dGLU kernel (synchronization aside:
    see the tables below). The epilogue computes dZ (output D), optionally H_rc (output H) and the
    per-(row, N-tile) dprob partials.

    :param acc_dtype: Accumulator type; only ``cutlass.Float32``.
    :param use_2cta_instrs: 2-CTA tcgen05 MMA (``mma_tiler_mn[0] == 256``).
    :param mma_tiler_mn: MMA tile (M, N); M in {128 (1-CTA), 256 (2-CTA)}, N in {32, 64, ..., 256}.
    :param cluster_shape_mn: Cluster shape; (2, 1) for 2-CTA, (1, 1) for 1-CTA.
    :param expert_cnt: Number of local experts E (<= 1024).
    :param clamp_scale: Soft-clamp scale ``s`` (trace-time constant). Pass the same Python float
        as to the forward kernel: the PTX immediates ``fp32(s)`` and ``fp32(1/s)`` are derived from
        it the same way.
        ``None`` selects the plain squared-ReLU (``tanh_mode`` is then unused).
    :param tanh_mode: ``"approx"`` (MUFU tanh.approx.f32 capped at 1) or ``"accurate"`` (the
        forward kernel's Cephes PTX tanh), as in the forward kernel. No default: the API passes its own.
    :param generate_h: Also write ``H_rc = bf16(p * A)``.
    :param weight_mode: ``MoEWeightMode.DISCRETE`` (per-expert pointer array) or ``DENSE``
        ([E, D, N] tensor).
    :param use_dynamic_sched: Atomic-counter dynamic tile scheduling (off by default).
    :param dprob_atomic: Add each tile's dprob partial atomically into slot 0 of a ``(M, 1, 1)``
        dprob (run-to-run order dependent) instead of into its own ``(row, tile_n)`` slot.

    Warps 0-3 epilogue, warp 4 MMA, warp 5 A/B TMA loads, warp 6 Z_save TMA loads, warp 7 scheduler.

    Named barriers. ``bar.sync`` (``barrier.sync.aligned``) promises that every thread of the CTA
    executes the barrier, so a barrier reached from a warp-role branch is spelled
    ``named_barrier_sync_unaligned`` (``barrier.sync id, n``; python/cudnn/AGENTS.md, CuTeDSL kernel
    bodies)::

        id  threads  participants                        form
        1   256      every thread, before the role dispatch  aligned (1-CTA clusters; larger ones use the cluster barrier)
        2   128      epilogue warps 0-3                  unaligned: around each subtile's dZ/H_rc store, before the TMEM free
        3   160      MMA warp 4 + epilogue warps 0-3     unaligned: the TMEM allocation gate (tmem_holding_buf published)
        4   32       scheduler warp 7                    unaligned: the sInfo stores in front of the tile-info commit

    Shared-memory buffers and why each release is safe (python/cudnn/AGENTS.md Rule 11)::

        sA / sB           TMA writes, only tcgen05.mma reads (no ordinary loads); the MMA warp releases a stage
                          with tcgen05.commit to ab_empty behind the stage's MMAs.
        sInfo             (tile-info ring, num_tile_stage stages of 4 words) the scheduler warp writes a stage with
                          ordinary stores; every thread of the TMA, MMA, epilogue and Z_save-load warps reads all
                          four words with ordinary loads and releases the stage itself through consume_then_arrive:
                          the address of its arrive is computed from the four loaded words through the release-zero
                          argument, so each load is read in front of the release in the generated code. One arrive
                          per thread (224 in all), exactly as PipelineAsync.consumer_release; the fence.proxy.async
                          in front of it is the template's and is not the ordering.
        sC (Z_save)       two 128 x 32 stages; warp 6 refills a stage with a TMA load (async proxy), every epilogue
                          thread reads its row's 32 values with ordinary loads. The stage is released through
                          consume_then_arrive with pred= lane 0 of each epilogue warp (the PipelineTmaAsync
                          signalling lane; c_empty init count 4): every lane of the warp computes the arrive
                          address from all 32 values it loaded, as the 16 packed 32-bit words its loads wrote.
                          fence.proxy.async.shared::cta between the last
                          load and the release (Rule 11's form for an async-proxy refill) is kept as well.
        sD / sH           two 128 x 32 stages each (sH only with generate_h); the epilogue warps write dZ (and H_rc)
                          with ordinary stores, warp 0 issues both TMA stores in one bulk group behind
                          fence.proxy.async and named barrier 2, then commits and waits cp.async.bulk.wait_group.read 1
                          (producer_acquire) in front of the next named barrier 2, so a stage is rewritten only after
                          the group two subtiles back finished reading it.
        tmem_holding_buf  written once by tcgen05.alloc (epilogue warp 0), read behind named barrier 3.
        TMEM accumulator  two stages read by tcgen05.ld; every loaded register feeds dZ, which is stored to sD in
                          front of the subtile's named barrier 2, and the elected acc_empty arrive follows the last
                          subtile's barrier (Rule 11's tensor-memory order).

    TMA descriptors and their acquire (python/cudnn/AGENTS.md, immutable tensor-map reuse)::

        A, C, D, H        kernel parameters (as is dense B), fixed for the launch; nothing to acquire.
        discrete B        one 128-byte descriptor per expert in the caller's workspace, rewritten in place by
                          helper_kernel before every launch. The TMA descriptor cache outlives launches, so with a
                          reused workspace address a B load could read the previous call's descriptor (the wrong
                          expert's weights). Warp 5 runs moe_utils.acquire_tma_desc on the tile's descriptor (every
                          lane, the extension's own pointer) before the tile's first B load whenever its expert
                          differs from last_acquired_expert, which starts at -1 in every launch; the descriptors do
                          not change during the launch, so later tiles of that expert reuse the acquire. Each CTA,
                          both of a 2-CTA pair included, acquires for its own loads.
    """

    FIX_PAD_SIZE = FIX_PAD_SIZE

    @staticmethod
    def can_implement(
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
    ) -> bool:
        """Host-side shape/tile check (A K-major, B N-major, C/D/H N-major, all BF16)."""
        return can_implement_bf16_grouped_gemm(
            cutlass.BFloat16,
            cutlass.BFloat16,
            cutlass.BFloat16,
            cutlass.Float32,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            l,
            "k",
            "n",
            "n",
            MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE,
            fix_pad_size=MoEGroupedGemmDsreluBf16Kernel.FIX_PAD_SIZE,
            n_align=32,
            tile_n_align=32,
        )

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        expert_cnt: int,
        clamp_scale: Optional[float],
        tanh_mode: str,
        generate_h: bool = False,
        weight_mode: MoEWeightMode = MoEWeightMode.DISCRETE,
        use_dynamic_sched: bool = False,
        dprob_atomic: bool = False,
    ):
        mma_tile_m = mma_tiler_mn[0]
        if self.FIX_PAD_SIZE % mma_tile_m != 0:
            raise ValueError(
                f"FIX_PAD_SIZE ({self.FIX_PAD_SIZE}) must be divisible by " f"mma_tiler_mn[0] ({mma_tile_m}); " "supported mma_tiler_mn[0] values: 128, 256."
            )
        if expert_cnt <= 0 or expert_cnt > 1024:
            raise ValueError(f"Expert count must be in [1, 1024], got {expert_cnt}.")
        if not isinstance(weight_mode, MoEWeightMode):
            raise TypeError(f"weight_mode must be a MoEWeightMode, got {type(weight_mode)}")
        if tanh_mode not in ("accurate", "approx"):
            raise ValueError(f"tanh_mode must be 'accurate' or 'approx', got {tanh_mode!r}")
        if acc_dtype is not cutlass.Float32:
            raise ValueError(f"acc_dtype must be cutlass.Float32, got {acc_dtype}")
        # s and 1/s become FP32 PTX immediates (the forward helper packs them with struct, which
        # raises on FP32 overflow), so both must be finite and nonzero in FP32. None: plain sReLU.
        if clamp_scale is not None:
            s_value = float(clamp_scale)
            try:
                s_ok = math.isfinite(s_value) and s_value > 0.0
                s_ok = s_ok and _to_fp32(s_value) > 0.0 and _to_fp32(1.0 / s_value) > 0.0
            except OverflowError:
                s_ok = False
            if not s_ok:
                raise ValueError(f"clamp_scale must be positive with s and 1/s finite and nonzero in FP32, " f"got {clamp_scale}")

        self.expert_cnt = expert_cnt
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        # Epilogue constants (trace-time). Kept as the Python float, exactly as the forward kernel
        # keeps it, so that
        # clamped_srelu_t_f32 bakes the same fp32(1/s) immediate and Float32(s) equals the forward's fp32(s).
        self.clamp_scale = None if clamp_scale is None else float(clamp_scale)
        self.tanh_mode = tanh_mode
        self.generate_h = generate_h
        self.dprob_atomic = dprob_atomic

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.epilog_load_tma_id = 6
        self.sched_warp_id = 7
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.epilog_load_tma_id,
                self.sched_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len((*self.epilog_warp_id, self.mma_warp_id, self.tma_warp_id, self.epilog_load_tma_id))

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * len(self.epilog_warp_id))
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)))
        self.sched_sync_barrier = pipeline.NamedBarrier(barrier_id=4, num_threads=self.threads_per_warp)
        # SM103 (GB300) has the same 228 KB shared memory as SM100.
        self.num_smem_capacity = get_smem_capacity_in_bytes("sm_100")

        self.use_dynamic_sched = use_dynamic_sched
        self.num_epilog_warps = len(self.epilog_warp_id)
        self.weight_mode = weight_mode

    def _setup_attributes(self):
        """Set up input-dependent BF16 GEMM attributes."""
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * mma_inst_tile_k)

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        # D (dZ), H (H_rc) and C (Z_save) all share the accumulator's tiling: one output column
        # per accumulator column (no gate/linear doubling as in dGLU).
        self.mma_tiler_d = self.mma_tiler

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,))

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Epilogue subtile: 128 rows (one per epilogue thread) x 32 columns
        self.epi_tile = (128, 32)

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_d_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.d_dtype,
            self.d_layout,
            self.num_smem_capacity,
            self.occupancy,
            self.generate_h,
        )

        # TMEM accumulator columns: derive from the actual accumulator fragment and round up to a
        # valid power-of-two allocation (as in the other BF16 grouped GEMM kernels).
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = get_num_tmem_alloc_cols(tCtAcc_fake)

        # Compute A/B/C/D shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage)
        # H_rc has D's dtype and layout and reuses this staged layout for its own buffer.
        self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.d_dtype, self.d_layout, self.epi_tile, self.num_d_stage)

    def get_desc_workspace_bytes(self) -> int:
        """Return descriptor workspace size in bytes."""
        if self.weight_mode == MoEWeightMode.DISCRETE:
            return TensormapWorkspace.size_bytes(1, self.expert_cnt)
        return 0

    def get_workspace_bytes(self) -> int:
        """Return descriptor workspace plus optional dynamic scheduler state."""
        desc_workspace_bytes = self.get_desc_workspace_bytes()
        dynamic_sched_bytes = 4 if self.use_dynamic_sched else 0
        return desc_workspace_bytes + dynamic_sched_bytes

    @cute.jit
    def _get_sched_counter_ptr(self, workspace_ptr):
        counter_addr = workspace_ptr.toint() + self.get_desc_workspace_bytes()
        return cute.make_ptr(cutlass.Int32, counter_addr, AddressSpace.gmem, assumed_align=4)

    @cute.kernel
    def helper_kernel(
        self,
        ptrs_b: cute.Pointer,
        n: Int32,
        k: Int32,
        b_stride_size: cutlass.Int64,
        b_major_mode: cutlass.Constexpr,
        workspace_ptr,
        tiled_mma_arg: cute.TiledMma,
        b_smem_layout_arg,
        cluster_layout_vmnk_shape_arg: cutlass.Constexpr,
    ):
        """Pre-main-kernel initialization (unchanged from the dGLU kernel).

        Launched with grid=(expert_cnt, 1, 1) for discrete mode, or grid=(1, 1, 1) for
        dense+dynamic mode. Discrete weight: each block builds the B TMA descriptor of one expert.
        Dynamic sched: block 0 resets the atomic tile counter to 0.
        """
        expert_idx = cute.arch.block_idx()[0]

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            b_tma_op_arg = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma_arg.thr_id)

            b_ptr_tensor = cute.make_tensor(
                cute.make_ptr(cutlass.Int64, ptrs_b.toint(), AddressSpace.gmem, assumed_align=8),
                cute.make_layout((self.expert_cnt,)),
            )

            c1 = cutlass.Int32(1)
            c0 = cutlass.Int64(0)
            c1_64 = 1
            if cutlass.const_expr(b_major_mode == OperandMajorMode.K):
                stride_n = b_stride_size
                stride_k = c1_64
            else:
                stride_n = c1_64
                stride_k = b_stride_size

            b_ptr_val = b_ptr_tensor[expert_idx]
            b_ptr = cute.make_ptr(self.b_dtype, b_ptr_val, AddressSpace.gmem)
            b_tensor_i = cute.make_tensor(b_ptr, cute.make_layout((n, k, c1), stride=(stride_n, stride_k, c0)))
            tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
                b_tma_op_arg,
                b_tensor_i,
                b_smem_layout_arg,
                self.mma_tiler,
                tiled_mma_arg,
                cluster_layout_vmnk_shape_arg,
            )

            workspace = TensormapWorkspace(workspace_ptr, ["b"])
            store_tma_desc(tma_atom_b, workspace.get_ptr("b", expert_idx))

        if cutlass.const_expr(self.use_dynamic_sched):
            if expert_idx == cutlass.Int32(0):
                sched_counter = cute.make_tensor(self._get_sched_counter_ptr(workspace_ptr), cute.make_layout(1))
                sched_counter[0] = cutlass.Int32(0)

    helper_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b,  # Dense: cute.Tensor (N,K,L) | Discrete: cute.Pointer to int64[]
        n: Int32,  # Ignored for dense mode
        k: Int32,  # Ignored for dense mode
        b_stride_size: cutlass.Int64,  # Ignored for dense mode
        b_major_mode: cutlass.Constexpr,  # Ignored for dense mode
        workspace_ptr,  # Descriptor workspace, plus dynamic scheduler counter when enabled
        c: cute.Tensor,  # INPUT: saved pre-activation Z_save (M, N, 1) bf16
        d: cute.Tensor,  # OUTPUT: dZ (M, N, 1) bf16
        h: Optional[cute.Tensor],  # OUTPUT: H_rc (M, N, 1) bf16, None unless generate_h
        padded_offsets: cute.Tensor,  # (E,) int32 cumulative padded token counts
        prob: cute.Tensor,  # INPUT: (M, 1, 1) fp32 routing probabilities
        dprob: cute.Tensor,  # OUTPUT, zeroed fp32: (M, num_n_slots, 1) per-(row, N-tile) partials,
        # or (M, 1, 1) dprob itself when dprob_atomic
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the fused GEMM.

        Dense mode: ``b`` is a 3-D cute.Tensor (N, K, L). Discrete mode: ``b`` is a cute.Pointer
        to a device int64[] array of per-expert base addresses; ``n``, ``k``, ``b_stride_size``
        and ``b_major_mode`` describe the uniform per-expert layout.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = a.element_type  # B must match A dtype
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        self.a_major_mode = LayoutEnum.from_tensor(a).mma_major_mode()

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            self.b_major_mode = LayoutEnum.from_tensor(b).mma_major_mode()
        else:
            self.b_major_mode = b_major_mode
        self.c_layout = LayoutEnum.from_tensor(c)
        self.d_layout = LayoutEnum.from_tensor(d)

        if cutlass.const_expr(self.generate_h != (h is not None)):
            raise ValueError("h must be passed exactly when the kernel was built with generate_h=True")
        if cutlass.const_expr(h is not None):
            if cutlass.const_expr(h.element_type != self.d_dtype):
                raise TypeError(f"H_rc dtype {h.element_type} must match dZ dtype {self.d_dtype}")

        # Setup attributes that depend on gemm inputs
        self._setup_attributes()

        # ---- B setup (mode-dependent) ----
        b_from_call_arg = b
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            c1 = cutlass.Int32(1)
            c0 = cutlass.Int64(0)
            c1_64 = 1
            if cutlass.const_expr(b_major_mode == OperandMajorMode.K):
                b_template_stride = (b_stride_size, c1_64, c0)
            else:
                b_template_stride = (c1_64, b_stride_size, c0)
            b_template_layout = cute.make_layout((n, k, c1), stride=b_template_stride)
            b_ptr_typed = cute.make_ptr(self.b_dtype, b.toint(), AddressSpace.gmem, assumed_align=16)
            b = cute.make_tensor(b_ptr_typed, b_template_layout)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA load for C (saved Z)
        c_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        self.tma_c_load_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), c, c_smem_layout, self.epi_tile)

        # Setup TMA store for D (dZ) and, optionally, H (H_rc)
        d_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), d, d_smem_layout, self.epi_tile)
        tma_atom_h = None
        tma_tensor_h = None
        if cutlass.const_expr(self.generate_h):
            tma_atom_h, tma_tensor_h = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), h, d_smem_layout, self.epi_tile)

        # Compute grid size using the MoE scheduler: per expert a (tokens_e x N x K) GEMM
        sched_params = MoESchedulerParams(
            scenario="2Dx3D",
            expert_shape=(
                self.expert_cnt,
                cute.size(d.shape, mode=[1]),
                cute.size(a.shape, mode=[1]),
            ),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            use_dynamic_sched=self.use_dynamic_sched,
        )
        self.sched_params, grid = compute_grid(sched_params, max_active_clusters, self.use_2cta_instrs)

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_tile_stage, self.use_dynamic_sched)
        sH_size = cute.cosize(self.d_smem_layout_staged.outer) if self.generate_h else 0

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            scheduler: SchedulerStorage
            # full + empty barriers of the C-load pipeline
            c_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, cute.cosize(self.d_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sH: cute.struct.Align[cute.struct.MemRange[self.d_dtype, sH_size], self.buffer_align_bytes]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Initialize per-expert B TMA descriptors in workspace
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        _need_helper = cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE or self.use_dynamic_sched)
        if cutlass.const_expr(_need_helper):
            _helper_grid_x = self.expert_cnt if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else 1
            _helper_args = (
                (b_from_call_arg if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cute.make_ptr(cutlass.Int64, 0, AddressSpace.gmem)),
                (n if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int32(0)),
                (k if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int32(0)),
                (b_stride_size if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int64(0)),
                (b_major_mode if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else self.b_major_mode),
                workspace_ptr,
                tiled_mma,
                b_smem_layout,
                self.cluster_layout_vmnk.shape,
            )
            self.helper_kernel(*_helper_args).launch(grid=(_helper_grid_x, 1, 1), block=(1, 1, 1), stream=stream, min_blocks_per_mp=1)

        # Launch the main kernel
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
            tma_atom_h,
            tma_tensor_h,
            padded_offsets,
            prob,
            dprob,
            workspace_ptr,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.d_smem_layout_staged,
            self.epi_tile,
            self.sched_params,
            cutlass.Int32(0),  # the release-zero (tile_dsl.consume): the literal 0, the only value ever passed
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
    # Internal: create extension based on weight_mode
    # ------------------------------------------------------------------

    @cute.jit
    def _make_extension(self, workspace_ptr):
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            desc_workspace = TensormapWorkspace(workspace_ptr, ["b"])
            return DiscreteWeightGroupedGemmSchedExtension(tensormap_ctor=desc_workspace)
        else:
            return ContiguousGroupedGemmSchedExtension()

    @cute.jit
    def dsrelu_subtile(
        self,
        acc_vec: cute.Tensor,
        z_vec: cute.Tensor,
        prob_val: Float32,
        dz_vec: cute.Tensor,
        h_vec: Optional[cute.Tensor] = None,
    ):
        """dsReLU of one thread's 32 values of an epilogue subtile; returns their dprob partial.

        ``acc_vec`` holds dH_acc (FP32, unrounded), ``z_vec`` float(Z_save) (exact BF16 -> FP32).
        Writes the FP32 dZ (rounded to BF16 by the caller) and, if ``h_vec`` is given, ``p * A``.
        ``t``, ``c``, ``A`` and ``p * A`` are the forward kernel's instruction sequence (its
        ``clamped_srelu_t_f32`` PTX, then ``mul.rn.f32`` for ``t * s``, ``c * c`` and ``A * p``),
        so ``p * A`` matches the forward ``H`` bit for bit at the same Z. The plain squared-ReLU
        (``clamp_scale`` None) takes ``r`` from the forward ``srelu_relu_f32`` PTX and
        ``A = r * r`` instead.
        """
        if cutlass.const_expr(self.clamp_scale is not None):
            s = Float32(self.clamp_scale)
        one = Float32(1.0)
        dprob_even = Float32(0.0)
        dprob_odd = Float32(0.0)
        for i in cutlass.range_constexpr(cute.size(acc_vec)):
            g = acc_vec[i]
            if cutlass.const_expr(self.clamp_scale is None):
                # r = max.NaN(z, 0) exactly as the forward computes it; z = float(Z_save) needs no
                # re-rounding (act_input="acc" here equals the forward's "saved" input). The derivative of
                # r^2 is 2r: (g * p) * (r + r) rounds like the original ((g * p) * 2) * r, since
                # both doublings are exact.
                r = srelu_relu_f32(z_vec[i], act_input="acc")
                a = mul_rn_f32(r, r)
                dz = g * prob_val
                dz = dz * (r + r)
            else:
                # t = tanh(max.NaN(z, 0) * fp32(1/s)) exactly as the forward computes it.
                # z = float(Z_save) is BF16-representable, so act_input="acc" (no re-rounding) equals
                # the forward's "saved" input. NaN in Z propagates; the approx mode is capped at 1,
                # so 1 - t^2 >= 0.
                t = clamped_srelu_t_f32(
                    z_vec[i],
                    clamp_scale=self.clamp_scale,
                    act_input="acc",
                    tanh_mode=self.tanh_mode,
                )
                c = mul_rn_f32(t, s)
                a = mul_rn_f32(c, c)
                dz = g * prob_val
                dz = dz * (c + c)
                dz = dz * (one - t * t)
            dz_vec[i] = dz
            if cutlass.const_expr(h_vec is not None):
                h_vec[i] = mul_rn_f32(a, prob_val)
            # Fixed-order partial sums (two chains for ILP), combined once at the end.
            if cutlass.const_expr(i % 2 == 0):
                dprob_even = dprob_even + a * g
            else:
                dprob_odd = dprob_odd + a * g
        return dprob_even + dprob_odd

    # GPU device kernel
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
        tma_atom_h: Optional[cute.CopyAtom],
        mH_mnl: Optional[cute.Tensor],
        padded_offsets: cute.Tensor,
        prob: cute.Tensor,
        dprob: cute.Tensor,
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        sched_params: MoESchedulerParams,
        release_zero_arg: cutlass.Int32,
    ):
        """GPU device kernel performing the persistent grouped GEMM + dsReLU epilogue.

        ``release_zero_arg`` is tile_dsl.consume's release-zero, the last argument: the host passes the
        literal 0 and its one use is ``release_zero`` below (python/cudnn/AGENTS.md Rule 11: a compare or
        a branch on it lets the compiler fold the mask, and the sInfo / Z_save loads are unordered again).
        """
        rz = release_zero(release_zero_arg)
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = tidx // 32
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        total_tokens = padded_offsets[self.expert_cnt - 1]

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
                cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
            cpasync.prefetch_descriptor(tma_atom_d)
            if cutlass.const_expr(self.generate_h):
                cpasync.prefetch_descriptor(tma_atom_h)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sched_storage = storage.scheduler

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Load C pipeline: the epilog-load warp produces, the epilogue warps consume
        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.epilog_warp_id))
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_mbar_ptr.data_ptr(),
            num_stages=self.num_c_stage,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            tx_count=self.tma_c_load_bytes,
            defer_sync=True,
        )

        # Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp * 1)
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_wo_sched)
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            padded_offsets,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=self._get_sched_counter_ptr(workspace_ptr),
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        # Tensor memory dealloc barrier init
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/C/D/H
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        sH = None
        if cutlass.const_expr(self.generate_h):
            sH = storage.sH.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = sched_storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/D
        # (SMEM/TMEM partitions stay global - they don't depend on per-expert tensors)
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        if total_tokens <= 0:
            cute.arch.nvvm.exit()
        k_tile_cnt = cute.ceil_div(cute.size(mB_nkl, mode=[1]), self.mma_tiler[2])

        #
        # Specialized Schedule warp (MoE Persistent Tile Scheduler)
        #
        if warp_idx == self.sched_warp_id:
            work_tile_info = scheduler.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tile_stage)

            while work_tile_info.is_valid_tile:
                # sInfo format: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = work_tile_info.expert_idx
                    sInfo[(1, tile_info_producer_state.index)] = work_tile_info.tile_m_idx
                    sInfo[(2, tile_info_producer_state.index)] = work_tile_info.tile_n_idx
                    sInfo[(3, tile_info_producer_state.index)] = work_tile_info.k_tile_cnt
                cute.arch.fence_proxy("async.shared", space="cta")

                named_barrier_sync_unaligned(self.sched_sync_barrier.barrier_id, self.sched_sync_barrier.num_threads)
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()

                work_tile_info = scheduler.advance_to_next_work()

            # Send invalid tile signal: expert_idx = -1
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = cutlass.Int32(-1)
                sInfo[(1, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(2, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            named_barrier_sync_unaligned(self.sched_sync_barrier.barrier_id, self.sched_sync_barrier.num_threads)
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            ext = self._make_extension(workspace_ptr)

            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            consume_then_arrive(tile_info_pipeline.sync_object_empty.get_barrier(tile_info_consumer_state.index).raw_ptr(), tile_info.load(), rz)
            tile_info_consumer_state.advance()

            # Expert whose workspace B descriptor this warp last acquired. Local to this warp and this
            # launch: -1 matches no expert, so every launch acquires afresh (see the class docstring).
            last_acquired_expert = cutlass.Int32(-1)
            while is_valid_tile:
                work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)

                # Get per-expert real tensors + TMA desc ptrs via extension
                real_a, _ = ext.get_gmem_tensor("a", mA_mkl, padded_offsets, work_tile_info)
                real_b, desc_ptr_b = ext.get_gmem_tensor("b", mB_nkl, padded_offsets, work_tile_info)
                if cutlass.const_expr(desc_ptr_b is not None):
                    # Discrete weights: helper_kernel rewrote this descriptor in place before the launch, and the
                    # TMA descriptor cache outlives launches. Acquire the address the B copies below load through
                    # (the extension's get_desc_ptr) before the first of them, whenever the expert changes.
                    if work_tile_info.expert_idx != last_acquired_expert:
                        acquire_tma_desc(ext.tensormap_ctor.get_desc_ptr("b", work_tile_info.expert_idx))
                        last_acquired_expert = work_tile_info.expert_idx

                # local_tile on per-expert tensors
                gA_mkl = cute.local_tile(real_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB_nkl = cute.local_tile(real_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))

                # MMA partition
                thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
                tCgA = thr_mma.partition_A(gA_mkl)
                tCgB = thr_mma.partition_B(gB_nkl)

                # TMA partition A
                a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tCgA, 0, 3),
                )
                # TMA partition B
                b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(tCgB, 0, 3),
                )
                # Slice to per mma tile index (L=0 since domain already offset'd)
                mma_tile_coord_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                mma_tile_coord_n = work_tile_info.tile_n_idx
                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_n, None, 0)]

                # Peek (try_wait) AB buffer empty
                peek_ab_empty_status = cutlass.Boolean(1)
                if k_tile_cnt > 0:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, k_tile)]
                    tBgB_k = tBgB_slice[(None, k_tile)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    ab_producer_state_next = ab_producer_state.clone()
                    ab_producer_state_next.advance()
                    if k_tile < k_tile_cnt - 1:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state_next)

                    # TMA load A (contiguous, global desc)
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    # TMA load B (discrete: per-expert desc from workspace; dense: global desc)
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                        tma_desc_ptr=desc_ptr_b,
                    )

                    ab_producer_state = ab_producer_state_next

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                consume_then_arrive(tile_info_pipeline.sync_object_empty.get_barrier(tile_info_consumer_state.index).raw_ptr(), tile_info.load(), rz)
                tile_info_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem (the subset barrier itself, not
            # TmemAllocator.wait_for_alloc(), which rebuilds it aligned)
            #
            named_barrier_sync_unaligned(self.tmem_alloc_barrier.barrier_id, self.tmem_alloc_barrier.num_threads)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Get the first tile info (sInfo format: expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            consume_then_arrive(tile_info_pipeline.sync_object_empty.get_barrier(tile_info_consumer_state.index).raw_ptr(), tile_info.load(), rz)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                peek_ab_full_status = cutlass.Boolean(1)
                if k_tile_cnt > 0 and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        ab_consumer_state_next = ab_consumer_state.clone()
                        ab_consumer_state_next.advance()
                        if k_tile < k_tile_cnt - 1:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state_next)

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)

                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state = ab_consumer_state_next
                    acc_pipeline.producer_commit(acc_producer_state)

                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                consume_then_arrive(tile_info_pipeline.sync_object_empty.get_barrier(tile_info_consumer_state.index).raw_ptr(), tile_info.load(), rz)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory (the subset barrier itself, not
            # TmemAllocator.wait_for_alloc(), which rebuilds it aligned)
            #
            named_barrier_sync_unaligned(self.tmem_alloc_barrier.barrier_id, self.tmem_alloc_barrier.num_threads)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue (SMEM/TMEM/register - invariant across experts)
            #
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, epi_tile, use_2cta_instrs)

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_s2r, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition_load(tiled_copy_t2r, tTR_rC, epi_tidx, sC)

            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_copy_and_partition_store(tiled_copy_t2r, tTR_rD, epi_tidx, sD)
            if cutlass.const_expr(self.generate_h):
                tTR_rH = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
                tiled_copy_r2s, tRS_rH, tRS_sH = self.epilog_smem_copy_and_partition_store(tiled_copy_t2r, tTR_rH, epi_tidx, sH)

            # Extension for per-expert domain conversion in epilogue
            epi_ext = self._make_extension(workspace_ptr)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            # Load C pipeline
            c_pipeline_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_c_stage)

            # Threads/warps participating in tma store pipeline. One commit group per subtile
            # (dZ and, optionally, H_rc), alternating over num_d_stage smem buffers.
            d_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilog_warp_id))
            d_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_d_stage, producer_group=d_producer_group)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Get the first tile info (sInfo format: expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            consume_then_arrive(tile_info_pipeline.sync_object_empty.get_barrier(tile_info_consumer_state.index).raw_ptr(), tile_info.load(), rz)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                # sInfo: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
                epi_work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                expert_idx = epi_work_tile_info.expert_idx
                mma_tile_coord_mnl = (
                    epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    epi_work_tile_info.tile_n_idx,
                    cutlass.Int32(0),
                )
                epi_ext.update_expert_info(padded_offsets, expert_idx)

                #
                # Per-expert gmem tensor setup via extension
                #
                thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)
                real_d, _ = epi_ext.get_gmem_tensor("d", mD_mnl, padded_offsets, epi_work_tile_info)
                gD_mnl_loop = cute.local_tile(real_d, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                tCgD_loop = thr_mma_epi.partition_C(gD_mnl_loop)
                bSG_sD, bSG_gD_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d, tCgD_loop, epi_tile, sD)
                bSG_gD = bSG_gD_partitioned[(None, None, None, mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], 0)]
                bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))
                if cutlass.const_expr(self.generate_h):
                    real_h, _ = epi_ext.get_gmem_tensor("h", mH_mnl, padded_offsets, epi_work_tile_info)
                    gH_mnl_loop = cute.local_tile(real_h, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                    tCgH_loop = thr_mma_epi.partition_C(gH_mnl_loop)
                    bSG_sH, bSG_gH_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_h, tCgH_loop, epi_tile, sH)
                    bSG_gH = bSG_gH_partitioned[(None, None, None, mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], 0)]
                    bSG_gH = cute.group_modes(bSG_gH, 1, cute.rank(bSG_gH))

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                #
                # Get PROB (per-expert local M position; one row per epilogue thread)
                #
                real_prob, _ = epi_ext.get_gmem_tensor("prob", prob, padded_offsets, epi_work_tile_info)
                mPosition = (
                    (epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)) * self.mma_tiler[0]
                    + mma_tile_coord_v * (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape))
                    + tidx
                )
                mProb = real_prob[mPosition, 0, 0]
                dProbVal = cutlass.Float32(0.0)

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                #
                # Epilogue over 128x32 subtiles in increasing order (the dprob summation order)
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                for subtile_idx in cutlass.range(0, subtile_cnt, 1, unroll=1):
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # Wait for this subtile of Z_save
                    c_pipeline.consumer_wait(c_pipeline_consumer_state)
                    cute.copy(
                        tiled_copy_s2r,
                        tRS_sC[(None, None, None, c_pipeline_consumer_state.index)],
                        tRS_rC,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    # Lane 0 of each epilogue warp releases the stage, as the pipeline's signalling lane did
                    # (init count 4); every lane computes the arrive address from all the Z_save it loaded,
                    # passed as the 16 packed 32-bit words the shared-memory loads wrote: tRS_rC is a retile of
                    # the compact tTR_rC, so its Int32 recast holds the same registers as the 32 BF16 values,
                    # without a zero-extension per value.
                    consume_then_arrive(
                        c_pipeline.sync_object_empty.get_barrier(c_pipeline_consumer_state.index).raw_ptr(),
                        cute.recast_tensor(tTR_rC, cutlass.Int32).load(),
                        rz,
                        pred=cute.arch.lane_idx() == cutlass.Int32(0),
                    )
                    c_pipeline_consumer_state.advance()

                    # Same logical element order in every view below (all are r2s retiles of
                    # fragments shaped like tTR_rAcc); copy into compact FP32 register tensors
                    # for the scalar math.
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc)
                    z_vec = tiled_copy_r2s.retile(tRS_rC)
                    acc_f32 = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
                    acc_f32.store(acc_vec.load())
                    z_f32 = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
                    z_f32.store(z_vec.load().to(cutlass.Float32))
                    dz_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
                    if cutlass.const_expr(self.generate_h):
                        h_vec = cute.make_rmem_tensor(acc_vec.shape, cutlass.Float32)
                    else:
                        h_vec = None

                    #
                    # dsReLU epilogue
                    #
                    dProbVal = dProbVal + self.dsrelu_subtile(acc_f32, z_f32, mProb, dz_vec, h_vec)

                    #
                    # Convert to D type (round to nearest even)
                    #
                    tRS_rD.store(dz_vec.load().to(self.d_dtype))
                    if cutlass.const_expr(self.generate_h):
                        tRS_rH.store(h_vec.load().to(self.d_dtype))

                    #
                    # Store D (and H) through shared memory with TMA
                    #
                    d_buffer = num_prev_subtiles % self.num_d_stage
                    num_prev_subtiles = num_prev_subtiles + 1
                    cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[(None, None, None, d_buffer)])
                    if cutlass.const_expr(self.generate_h):
                        cute.copy(tiled_copy_r2s, tRS_rH, tRS_sH[(None, None, None, d_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    named_barrier_sync_unaligned(self.epilog_sync_barrier.barrier_id, self.epilog_sync_barrier.num_threads)
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(tma_atom_d, bSG_sD[(None, d_buffer)], bSG_gD[(None, subtile_idx)])
                        if cutlass.const_expr(self.generate_h):
                            cute.copy(tma_atom_h, bSG_sH[(None, d_buffer)], bSG_gH[(None, subtile_idx)])
                        d_pipeline.producer_commit()
                        # Wait until the buffer written next is no longer read by an older store
                        d_pipeline.producer_acquire()
                    named_barrier_sync_unaligned(self.epilog_sync_barrier.barrier_id, self.epilog_sync_barrier.num_threads)

                #
                # Async arrive accumulator buffer empty
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                real_dprob, _ = epi_ext.get_gmem_tensor("dprob", dprob, padded_offsets, epi_work_tile_info)
                if cutlass.const_expr(self.dprob_atomic):
                    #
                    # Atomic dprob: every N tile of the row adds into its single slot, in the
                    # order the tiles get here (the dGLU kernel's dprob accumulation).
                    #
                    _ = atomic_add_float32(ptr=real_dprob[(mPosition, 0, None)].iterator.llvm_ptr, value=dProbVal)
                else:
                    #
                    # Deterministic dprob: this thread is the only writer of slot (row, tile_n)
                    # of the zero-initialized partial buffer, so the atomic add stores dProbVal
                    # exactly.
                    #
                    _ = atomic_add_float32(
                        ptr=real_dprob[(mPosition, epi_work_tile_info.tile_n_idx, None)].iterator.llvm_ptr,
                        value=dProbVal,
                    )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                consume_then_arrive(tile_info_pipeline.sync_object_empty.get_barrier(tile_info_consumer_state.index).raw_ptr(), tile_info.load(), rz)
                tile_info_consumer_state.advance()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            named_barrier_sync_unaligned(self.epilog_sync_barrier.barrier_id, self.epilog_sync_barrier.num_threads)
            tmem.free(tmem_ptr)
            #
            # Wait for D/H stores complete
            #
            d_pipeline.producer_tail()

        #
        # Specialized epilog load warp (loads C = Z_save from GMEM to SMEM via TMA)
        #
        if warp_idx == self.epilog_load_tma_id:
            c_load_ext = self._make_extension(workspace_ptr)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            consume_then_arrive(tile_info_pipeline.sync_object_empty.get_barrier(tile_info_consumer_state.index).raw_ptr(), tile_info.load(), rz)
            tile_info_consumer_state.advance()

            c_pipeline_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_c_stage)
            while is_valid_tile:
                c_work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                mma_tile_coord_mnl = (
                    c_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    c_work_tile_info.tile_n_idx,
                    cutlass.Int32(0),
                )
                c_load_ext.update_expert_info(padded_offsets, c_work_tile_info.expert_idx)

                # Per-expert C tensor via extension
                real_c, _ = c_load_ext.get_gmem_tensor("c", mC_mnl, padded_offsets, c_work_tile_info)
                gC_mnl_loop = cute.local_tile(real_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                thr_mma_c_load = tiled_mma.get_slice(mma_tile_coord_v)
                tCgC_loop = thr_mma_c_load.partition_C(gC_mnl_loop)

                bGS_sC, bGS_gC_partitioned = self.epilog_gmem_copy_and_partition(tidx, tma_atom_c, tCgC_loop, epi_tile, sC)
                bGS_gC = bGS_gC_partitioned[(None, None, None, mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], 0)]
                bGS_gC = cute.group_modes(bGS_gC, 1, cute.rank(bGS_gC))
                # One C subtile per epilogue subtile, in the epilogue's (increasing) order
                subtile_cnt = cute.size(bGS_gC.shape, mode=[1])
                for subtile_idx in cutlass.range(subtile_cnt, unroll=1):
                    c_pipeline.producer_acquire(c_pipeline_producer_state)
                    cute.copy(
                        tma_atom_c,
                        bGS_gC[(None, subtile_idx)],
                        bGS_sC[(None, c_pipeline_producer_state.index)],
                        tma_bar_ptr=c_pipeline.producer_get_barrier(c_pipeline_producer_state),
                    )
                    c_pipeline_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                consume_then_arrive(tile_info_pipeline.sync_object_empty.get_barrier(tile_info_consumer_state.index).raw_ptr(), tile_info.load(), rz)
                tile_info_consumer_state.advance()

            #
            # Wait C buffer tail complete
            #
            c_pipeline.producer_tail(c_pipeline_producer_state)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make the TMEM->register tiled copy and partition the accumulator (unchanged from dGLU).

        :return: (tiled_copy_t2r, tTR_tAcc, tTR_rAcc); the register shape is derived from the TMEM
            partition (no gmem dependency).
        """
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,
            self.d_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        tTR_rAcc = thr_copy_t2r.partition_D(tAcc_epi)

        # Derive register shape from TMEM partition (no gmem D needed)
        per_subtile_shape = cute.coalesce(tTR_rAcc[(None, None, None, 0, 0, 0)].layout, target_profile=((1, 1), 1, 1)).shape
        tTR_rAcc = cute.make_rmem_tensor(per_subtile_shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition_load(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make the smem->register copy of C matching the accumulator fragment.

        :return: (tiled_copy_s2r, tSR_rC, tSR_sC) - register destination and smem source partitions.
        """
        copy_atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        # (S2R, S2R_M, S2R_N, PIPE_C)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_D(sC)
        # (S2R, S2R_M, S2R_N)
        tSR_rC = tiled_copy_s2r.retile(tTR_rC)
        return tiled_copy_s2r, tSR_rC, tSR_sC

    def epilog_smem_copy_and_partition_store(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rD: cute.Tensor,
        tidx: cutlass.Int32,
        sD: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make the register->smem copy of an output (D or H) matching the accumulator fragment.

        :return: (tiled_copy_r2s, tRS_rD, tRS_sD) - register source and smem destination partitions.
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.d_layout, self.d_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        # (R2S, R2S_M, R2S_N)
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        """Partition shared memory and global memory for a TMA epilogue load/store (unchanged).

        :return: (bSG_sD, bSG_gD) with shapes ((ATOM_V, REST_V), EPI_M, EPI_N) and
            ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL).
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gD_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        sD_for_tma_partition = cute.group_modes(sD, 0, 2)
        gD_for_tma_partition = cute.group_modes(gD_epi, 0, 2)
        bSG_sD, bSG_gD = cpasync.tma_partition(atom, 0, cute.make_layout(1), sD_for_tma_partition, gD_for_tma_partition)
        return bSG_sD, bSG_gD

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: LayoutEnum,
        d_dtype: Type[cutlass.Numeric],
        d_layout: LayoutEnum,
        num_smem_capacity: int,
        occupancy: int,
        generate_h: bool,
    ) -> Tuple[int, int, int, int, int]:
        """Compute pipeline stages; A/B stages take the shared memory the epilogue leaves free."""
        num_acc_stage = 2
        num_c_stage = 2
        num_d_stage = 2
        num_tile_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)  # a tmp 1 stage is provided
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)  # a tmp 1 stage is provided
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
        d_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)

        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        sinfo_bytes = 4 * 4 * num_tile_stage
        c_bytes = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one) * num_c_stage
        # dZ buffers, plus the same again for H_rc
        d_bytes = cute.size_in_bytes(d_dtype, d_smem_layout_staged_one) * num_d_stage * (2 if generate_h else 1)
        epi_bytes = c_bytes + d_bytes

        num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)) // ab_bytes_per_stage

        return num_acc_stage, num_ab_stage, num_c_stage, num_d_stage, num_tile_stage
