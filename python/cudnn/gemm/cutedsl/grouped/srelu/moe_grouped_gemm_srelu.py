# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The kernel class in this file is derived from the BF16 GLU grouped GEMM kernel of
# NVIDIA/cudnn-frontend@b03c3f3321f5a66b0582db44789f4b5be212aca5
# (python/cudnn/gemm/cutedsl/grouped/glu/moe_grouped_gemm_glu_bias.py). The persistent MoE tile
# scheduler, TMA loads, A/B/accumulator pipelines, tcgen05 MMA and TMEM handling are kept as in that
# kernel; the GLU / bias / alpha epilogue is replaced by the weighted (optionally tanh-clamped)
# squared-ReLU epilogue below. Synchronization follows python/cudnn/AGENTS.md instead of that kernel:
# the warp-role named barriers are unaligned, the tile-info stages are released through
# tile_dsl.consume and, with discrete weights, the TMA warp acquires each per-expert B descriptor
# before it loads through it (see the kernel class docstring).

"""BF16 MoE grouped GEMM with a fused weighted squared-ReLU epilogue (SM100).

For expert ``e``, padded routed row ``m`` and output column ``n``:

    Z_acc  = sum_k A[m, k] * B_e[n, k]      BF16 operands, FP32 tcgen05 accumulation
    Z_save = bf16_rn(Z_acc)                 always written (output ``c``)
    z_in   = Z_acc                          act_input == "acc"   (default)
           = float(Z_save)                  act_input == "saved"
    u      = max(z_in, 0) * fp32(1 / s)     NaN propagates; exact division when s is a power of 2
    t      = tanh(u)                        tanh_mode "approx" (the API default) or "accurate"
    c      = s * t;  A = c * c              FP32, same operation order as an eager PyTorch reference
    H      = bf16_rn(A * p[m])              output ``d``; p is FP32, any sign

With ``clamp_scale=None`` (the plain squared-ReLU) there is no tanh:
``r = max(z_in, 0)`` (NaN propagates), ``A = r * r`` and ``H = bf16_rn(A * p[m])``.

``tanh_mode == "accurate"`` evaluates tanh with the Cephes ``tanhf`` scheme: an odd degree-11
minimax polynomial for u < 0.625 and ``1 - 2 / (exp(2u) + 1)`` above, where ``exp`` is
``ex2.approx.ftz.f32`` and the reciprocal is ``rcp.approx`` plus one Newton step. FP32 emulation
of this exact instruction sequence against ``math.tanh`` gives a maximum relative error of
7.4e-8 for u < 0.625, 1.6e-7 (2.7e-7 if ex2.approx is off by its documented 2 ulp) for
0.625 <= u < 1.2 and below 1e-7 above; tanh(u) == 1.0f for u >= 9.5, as in libm.
``tanh_mode == "approx"`` uses ``tanh.approx.f32`` (MUFU, ~2^-11 relative error) capped at 1.

Weights are either DISCRETE (a device array of E per-expert ``[N, K]`` base addresses; a helper
kernel builds one B TMA descriptor per expert in a small device workspace) or DENSE (one
``(N, K, E)`` tensor). Every expert segment of ``A`` is padded to a multiple of 256 rows; empty
experts are allowed.

The device helpers :func:`weighted_clamped_srelu_f32`, :func:`clamped_srelu_t_f32` and
:func:`srelu_relu_f32` are the single source of the epilogue arithmetic. The dSReLU backward
kernel (``dsrelu/moe_grouped_gemm_dsrelu.py``) recomputes ``t`` with :func:`clamped_srelu_t_f32`
(or ``r`` with :func:`srelu_relu_f32` for the plain squared-ReLU) and then ``c``, ``A`` and
``p * A`` with the same ``mul.rn.f32`` instructions as the ``"h"`` PTX here, so with
act_input == "saved" the forward ``H`` and the backward's recomputed ``H`` are equal bit for bit.
"""

import functools
import struct
from typing import Optional, Tuple, Union

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
    can_implement_bf16_grouped_gemm,
    compute_grid,
    epilog_gmem_copy_and_partition,
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
from ._bf16_options import ACT_INPUTS, TANH_MODES, _check_clamp_scale

__all__ = [
    "ACT_INPUTS",
    "TANH_MODES",
    "MoEGroupedGemmSreluBf16Kernel",
    "weighted_clamped_srelu_f32",
    "clamped_srelu_t_f32",
    "srelu_relu_f32",
    "clamped_srelu_ptx",
]


# =============================================================================
# Epilogue arithmetic (inline PTX, so the instruction sequence does not depend on the DSL's
# fast-math lowering of math.tanh / arith.divf)
#
# This is the single definition of the activation for the BF16 sReLU fusion. The dSReLU kernel
# (dsrelu/moe_grouped_gemm_dsrelu.py) computes t with clamped_srelu_t_f32 from this module and then
# c = mul.rn(t, fp32(s)), A = mul.rn(c, c), p * A = mul.rn(A, p): the instructions that
# clamped_srelu_ptx(..., output="h") emits after t. For the plain squared-ReLU (clamp_scale None)
# the dSReLU kernel takes r from srelu_relu_f32 and computes A = mul.rn(r, r), p * A = mul.rn(A, p),
# again the "h" instructions. That is what makes H (act_input == "saved") equal to the backward's
# recomputed H bit for bit. Keep the two in sync: any change to the "h" sequences below must be
# mirrored in the dSReLU kernel's dsrelu_subtile.
# =============================================================================

# Cephes tanhf polynomial: tanh(u) = u + u * u^2 * P(u^2) for |u| < 0.625.
_TANH_POLY = (
    -5.70498872745e-3,
    2.06390887954e-2,
    -5.37397155531e-2,
    1.33314422036e-1,
    -3.33332819422e-1,
)
_TANH_POLY_LIMIT = 0.625
# exp(2u) = 2^(u * 2 log2(e)); u is clamped to 10 first (tanh(10) == 1.0f), so exp stays finite.
_TANH_EXP_CLAMP = 10.0
_TWO_LOG2E = 2.0 * 1.4426950408889634


def _f32_imm(value: float) -> str:
    """PTX hexadecimal immediate (``0fXXXXXXXX``) of ``value`` rounded to FP32."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _tanh_ptx(dst: str, src: str, tanh_mode: str) -> Tuple[str, ...]:
    """PTX computing ``dst = tanh(src)`` for ``src >= 0`` or NaN (NaN propagates).

    Uses the scratch registers declared in :func:`_srelu_ptx_prologue`.
    """
    if tanh_mode == "approx":
        return (f"tanh.approx.f32 {dst}, {src};", f"min.NaN.f32 {dst}, {dst}, {_f32_imm(1.0)};")
    if tanh_mode != "accurate":
        raise ValueError(f"tanh_mode must be one of {TANH_MODES}, got {tanh_mode!r}")
    c0, c1, c2, c3, c4 = (_f32_imm(v) for v in _TANH_POLY)
    return (
        # Polynomial branch, valid for src < 0.625.
        f"mul.rn.f32 k1_q, {src}, {src};",
        f"fma.rn.f32 k1_p, k1_q, {c0}, {c1};",
        f"fma.rn.f32 k1_p, k1_p, k1_q, {c2};",
        f"fma.rn.f32 k1_p, k1_p, k1_q, {c3};",
        f"fma.rn.f32 k1_p, k1_p, k1_q, {c4};",
        "mul.rn.f32 k1_p, k1_p, k1_q;",
        f"fma.rn.f32 k1_p, k1_p, {src}, {src};",
        # Exponential branch 1 - 2 / (exp(2 src) + 1); min.NaN keeps NaN.
        f"min.NaN.f32 k1_e, {src}, {_f32_imm(_TANH_EXP_CLAMP)};",
        f"mul.rn.f32 k1_e, k1_e, {_f32_imm(_TWO_LOG2E)};",
        "ex2.approx.ftz.f32 k1_e, k1_e;",
        f"add.rn.f32 k1_e, k1_e, {_f32_imm(1.0)};",
        # r = 1 / (exp + 1): rcp.approx and one Newton step r += r * (1 - d * r).
        "rcp.approx.ftz.f32 k1_r, k1_e;",
        "neg.f32 k1_e, k1_e;",
        f"fma.rn.f32 k1_n, k1_e, k1_r, {_f32_imm(1.0)};",
        "fma.rn.f32 k1_r, k1_r, k1_n, k1_r;",
        f"fma.rn.f32 k1_e, k1_r, {_f32_imm(-2.0)}, {_f32_imm(1.0)};",
        # Select; NaN compares false and takes the (NaN) exponential branch.
        f"setp.lt.f32 k1_sm, {src}, {_f32_imm(_TANH_POLY_LIMIT)};",
        f"selp.f32 {dst}, k1_p, k1_e, k1_sm;",
    )


def _srelu_ptx_prologue(act_input: str, clamp_scale: Optional[float]) -> Tuple[str, ...]:
    """Register declarations plus ``k1_u = max(z_in, 0) / s`` from the FP32 accumulator ``$1``.

    Without a ``clamp_scale`` (plain squared-ReLU) ``k1_u = max(z_in, 0)``, unscaled.
    """
    if act_input not in ACT_INPUTS:
        raise ValueError(f"act_input must be one of {ACT_INPUTS}, got {act_input!r}")
    if clamp_scale is not None:
        _check_clamp_scale(clamp_scale)
    lines = [
        ".reg .pred k1_sm;",
        ".reg .b16 k1_zb, k1_zl;",
        ".reg .b32 k1_zw;",
        ".reg .f32 k1_z, k1_u, k1_t, k1_q, k1_p, k1_e, k1_r, k1_n, k1_c, k1_a;",
    ]
    if act_input == "saved":
        # z_in = float(bf16_rn(acc)): the same RN rounding that produces the stored Z_save.
        lines += [
            "cvt.rn.bf16.f32 k1_zb, $1;",
            "mov.b16 k1_zl, 0;",
            "mov.b32 k1_zw, {k1_zl, k1_zb};",
            "mov.b32 k1_z, k1_zw;",
        ]
    else:
        lines.append("mov.f32 k1_z, $1;")
    # relu that propagates NaN like torch.relu.
    lines.append(f"max.NaN.f32 k1_u, k1_z, {_f32_imm(0.0)};")
    if clamp_scale is not None:
        # relu(z) / s as relu(z) * fp32(1/s), 1/s taken in double: exact for power-of-two s. Eager
        # torch multiplies by 1.0f / float(s) instead; the two agree whenever s is FP32-representable
        # (e.g. 16) and can differ by one ulp otherwise.
        lines.append(f"mul.rn.f32 k1_u, k1_u, {_f32_imm(1.0 / clamp_scale)};")
    return tuple(lines)


@functools.lru_cache(maxsize=None)
def clamped_srelu_ptx(clamp_scale: Optional[float], act_input: str, tanh_mode: str, output: str) -> str:
    """Inline PTX of the forward epilogue for one element.

    Operands: ``$1`` = FP32 accumulator ``Z_acc``; for ``output == "h"`` also ``$2`` = FP32
    row prob ``p``. Result ``$0``, in FP32 before the final BF16 rounding: ``t`` (``output ==
    "t"``, tanh-clamped only), ``A * p`` (``output == "h"``) or ``r = max(z_in, 0)`` (``output ==
    "r"``, plain squared-ReLU only). ``clamp_scale=None`` selects the plain squared-ReLU, whose
    ``"h"`` is ``(r * r) * p`` and which ignores ``tanh_mode``.
    """
    if output not in ("t", "h", "r"):
        raise ValueError(f"output must be 't', 'h' or 'r', got {output!r}")
    if output == "t" and clamp_scale is None:
        raise ValueError("output 't' (the tanh) needs a clamp_scale")
    if output == "r" and clamp_scale is not None:
        raise ValueError("output 'r' (the unscaled relu) is for the plain squared-ReLU only")
    lines = ["{"]
    lines += _srelu_ptx_prologue(act_input, clamp_scale)
    if output == "r":
        lines.append("mov.f32 $0, k1_u;")
    elif clamp_scale is None:
        lines += ["mul.rn.f32 k1_a, k1_u, k1_u;", "mul.rn.f32 $0, k1_a, $2;"]  # A = r * r  # A * p
    elif output == "t":
        lines += _tanh_ptx("$0", "k1_u", tanh_mode)
    else:
        lines += _tanh_ptx("k1_t", "k1_u", tanh_mode)
        lines += [
            f"mul.rn.f32 k1_c, k1_t, {_f32_imm(clamp_scale)};",  # c = s * t
            "mul.rn.f32 k1_a, k1_c, k1_c;",  # A = c * c
            "mul.rn.f32 $0, k1_a, $2;",  # A * p
        ]
    lines.append("}")
    return "\n".join(lines)


def _scale_key(clamp_scale: Optional[float]) -> Optional[float]:
    """``clamp_scale`` as the Python float the PTX immediates are derived from (None: plain)."""
    return None if clamp_scale is None else float(clamp_scale)


def weighted_clamped_srelu_f32(
    z: Float32,
    prob: Float32,
    *,
    clamp_scale: Optional[float],
    act_input: str,
    tanh_mode: str,
    loc=None,
    ip=None,
) -> Float32:
    """Device helper: ``A(z_in) * p`` in FP32 (round to BF16 afterwards to get ``H``).

    ``z`` is the FP32 accumulator; with ``act_input == "saved"`` it is rounded to BF16 (RN)
    first. ``clamp_scale``, ``act_input`` and ``tanh_mode`` are trace-time constants.
    Equivalent, bit for bit, to :func:`clamped_srelu_t_f32` followed by ``mul.rn.f32`` of
    ``t * fp32(s)``, ``c * c`` and ``A * p`` (what the dSReLU kernel does for its recomputed
    ``H``); with
    ``clamp_scale=None``, to :func:`srelu_relu_f32` followed by ``mul.rn.f32`` of ``r * r`` and
    ``A * p``.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(z).ir_value(loc=loc, ip=ip), Float32(prob).ir_value(loc=loc, ip=ip)],
            clamped_srelu_ptx(_scale_key(clamp_scale), act_input, tanh_mode, "h"),
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def clamped_srelu_t_f32(z: Float32, *, clamp_scale: float, act_input: str, tanh_mode: str, loc=None, ip=None) -> Float32:
    """Device helper: ``t = tanh(max(z_in, 0) / s)`` exactly as the forward epilogue computes it.

    For the backward (t recomputed from ``float(Z_save)``) pass the FP32 value of Z_save with
    ``act_input="acc"`` (it is already BF16-representable, so both modes give the same t).
    Pass the same Python float ``clamp_scale`` as the forward got: ``fp32(1/s)`` is derived from it.
    """
    if clamp_scale is None:
        raise ValueError("clamped_srelu_t_f32 needs a clamp_scale; use srelu_relu_f32 without one")
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(z).ir_value(loc=loc, ip=ip)],
            clamped_srelu_ptx(float(clamp_scale), act_input, tanh_mode, "t"),
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def srelu_relu_f32(z: Float32, *, act_input: str, loc=None, ip=None) -> Float32:
    """Device helper: ``r = max(z_in, 0)`` (NaN propagates) as the plain-sReLU forward epilogue has it.

    The plain squared-ReLU counterpart of :func:`clamped_srelu_t_f32`: the dSReLU kernel squares it
    with the ``mul.rn.f32`` instructions of the plain ``"h"`` PTX, so its recomputed ``H`` reproduces
    ``H``.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(z).ir_value(loc=loc, ip=ip)],
            clamped_srelu_ptx(None, act_input, "accurate", "r"),
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Kernel
# =============================================================================


class MoEGroupedGemmSreluBf16Kernel:
    """BF16 grouped GEMM (FC1) with MoE scheduling and the weighted squared-ReLU epilogue
    (tanh-clamped with a ``clamp_scale``, plain without one).

    Organisation (unchanged from the BF16 GLU kernel): warps 0-3 epilogue, warp 4 MMA,
    warp 5 TMA loads, warp 6 persistent MoE tile scheduler. The epilogue walks the
    accumulator in independent 128 x 32 subtiles; per subtile it TMA-stores
    C = bf16(acc) (``z_save``) and then D = bf16(A * p) (``h``).

    :note: Constraints:
        - MMA tiler M 128 (1-CTA) or 256 (2-CTA), MMA tiler N a multiple of 32 up to 256
        - Cluster shape M a multiple of 2 for 2-CTA, power-of-two cluster dims, size <= 16
        - Every expert's padded token count (and so every padded_offsets[i]) is a multiple
          of FIX_PAD_SIZE (256)

    Named barriers. ``bar.sync`` (``barrier.sync.aligned``) promises that every thread of the CTA
    executes the barrier, so a barrier reached from a warp-role branch is spelled
    ``named_barrier_sync_unaligned`` (``barrier.sync id, n``; python/cudnn/AGENTS.md, CuTeDSL kernel
    bodies)::

        id  threads  participants                        form
        1   224      every thread, before the role dispatch  aligned (1-CTA clusters; larger ones use the cluster barrier)
        2   128      epilogue warps 0-3                  unaligned: around each subtile's C and D stores, before the TMEM free
        3   160      MMA warp 4 + epilogue warps 0-3     unaligned: the TMEM allocation gate (tmem_holding_buf published)
        4   32       scheduler warp 6                    unaligned: the sInfo stores in front of the tile-info commit

    Shared-memory buffers and why each release is safe (python/cudnn/AGENTS.md Rule 11)::

        sA / sB           TMA writes, only tcgen05.mma reads (no ordinary loads); the MMA warp releases a stage
                          with tcgen05.commit to ab_empty behind the stage's MMAs.
        sInfo             (tile-info ring, num_tile_stage stages of 4 words) the scheduler warp writes a stage with
                          ordinary stores; every thread of the TMA, MMA and epilogue warps reads all four words with
                          ordinary loads and releases the stage itself through consume_then_arrive: the address of
                          its arrive is computed from the four loaded words through the release-zero argument, so
                          each load is read in front of the release in the generated code. One arrive per thread
                          (192 in all), exactly as PipelineAsync.consumer_release; the fence.proxy.async in front
                          of it is the template's and is not the ordering.
        sC / sD           two 128 x 32 stages each; the epilogue warps write a stage with ordinary stores, warp 0
                          issues its TMA store behind fence.proxy.async and named barrier 2, then commits and waits
                          cp.async.bulk.wait_group.read 1 (producer_acquire) in front of the next named barrier 2,
                          so a stage is rewritten only after the store two subtiles back finished reading it.
        tmem_holding_buf  written once by tcgen05.alloc (epilogue warp 0), read behind named barrier 3.
        TMEM accumulator  two stages read by tcgen05.ld; every loaded register feeds the C/D staging stores in
                          front of the subtile's named barrier 2, and the elected acc_empty arrive follows the last
                          subtile's barrier (Rule 11's tensor-memory order).

    TMA descriptors and their acquire (python/cudnn/AGENTS.md, immutable tensor-map reuse)::

        A, C, D           kernel parameters (as is dense B), fixed for the launch; nothing to acquire.
        discrete B        one 128-byte descriptor per expert in the caller's workspace, rewritten in place by
                          helper_kernel before every launch. The TMA descriptor cache outlives launches, so with a
                          reused workspace address a B load could read the previous call's descriptor (the wrong
                          expert's weights). Warp 5 runs moe_utils.acquire_tma_desc on the tile's descriptor (every
                          lane, the extension's own pointer) before the tile's first B load whenever its expert
                          differs from last_acquired_expert, which starts at -1 in every launch; the descriptors do
                          not change during the launch, so later tiles of that expert reuse the acquire. Each CTA,
                          both of a 2-CTA pair included, acquires for its own loads.
    """

    # Fixed pad size for user-side padding (decoupled from kernel tile size)
    FIX_PAD_SIZE = 256

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
        """Shape/tile check of the BF16 grouped GEMM family (N and tile N multiples of 32)."""
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
            "k",
            "n",
            MoEGroupedGemmSreluBf16Kernel.FIX_PAD_SIZE,
            fix_pad_size=MoEGroupedGemmSreluBf16Kernel.FIX_PAD_SIZE,
            n_align=32,
            tile_n_align=32,
        )

    def __init__(
        self,
        acc_dtype,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        expert_cnt: int,
        clamp_scale: Optional[float],
        act_input: str,
        tanh_mode: str,
        weight_mode: MoEWeightMode = MoEWeightMode.DISCRETE,
        use_dynamic_sched: bool = False,
    ):
        mma_tile_m = mma_tiler_mn[0]
        if self.FIX_PAD_SIZE % mma_tile_m != 0:
            raise ValueError(f"FIX_PAD_SIZE ({self.FIX_PAD_SIZE}) must be divisible by " f"mma_tiler_mn[0] ({mma_tile_m}).")
        if expert_cnt > 1024:
            raise ValueError("Expert count > 1024 is not supported.")
        if not isinstance(weight_mode, MoEWeightMode):
            raise TypeError(f"weight_mode must be a MoEWeightMode, got {type(weight_mode)}")
        if act_input not in ACT_INPUTS:
            raise ValueError(f"act_input must be one of {ACT_INPUTS}, got {act_input!r}")
        if tanh_mode not in TANH_MODES:
            raise ValueError(f"tanh_mode must be one of {TANH_MODES}, got {tanh_mode!r}")
        if clamp_scale is not None:
            _check_clamp_scale(clamp_scale)

        self.expert_cnt = expert_cnt
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.threads_per_warp = 32

        all_warps = [*self.epilog_warp_id, self.mma_warp_id, self.tma_warp_id, self.sched_warp_id]
        warps_wo_sched = [*self.epilog_warp_id, self.mma_warp_id, self.tma_warp_id]
        self.threads_per_cta = self.threads_per_warp * len(all_warps)
        self.threads_wo_sched = self.threads_per_warp * len(warps_wo_sched)

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * len(self.epilog_warp_id))
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)))
        self.sched_sync_barrier = pipeline.NamedBarrier(barrier_id=4, num_threads=self.threads_per_warp)
        self.num_smem_capacity = get_smem_capacity_in_bytes("sm_100")

        self.weight_mode = weight_mode
        self.use_dynamic_sched = use_dynamic_sched

        # Epilogue constants (trace time)
        # Kept as the Python float: the dSReLU kernel passes its own float(s) to clamped_srelu_t_f32, and the
        # fp32(s) / fp32(1/s) immediates must be derived from the same value in both kernels.
        # None selects the plain squared-ReLU.
        self.clamp_scale = _scale_key(clamp_scale)
        self.act_input = act_input
        self.tanh_mode = tanh_mode

    def _setup_attributes(self):
        """Set up configurations that depend on the GEMM inputs.

        Tiled MMA, MMA/cluster/tile shapes, cluster layout, multicast CTA counts, epilogue
        subtile, stage counts and A/B/C/D shared memory layouts.
        """
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

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,))

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Epilogue subtile: C and D are both [M, N] and share one 128 x 32 subtile.
        self.epi_tile = (128, 32)

        # Setup A/B/C/D stage count in shared memory and ACC stage count in tensor memory
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
        )

        # TMEM accumulator columns: derive from the actual accumulator fragment and round up
        # to a valid power-of-two allocation.
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = get_num_tmem_alloc_cols(tCtAcc_fake)

        # Compute A/B/C/D shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage)
        self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.d_dtype, self.d_layout, self.epi_tile, self.num_d_stage)

    @staticmethod
    def _compute_stages(
        tiled_mma,
        mma_tiler_mnk,
        a_dtype,
        b_dtype,
        epi_tile,
        c_dtype,
        c_layout,
        d_dtype,
        d_layout,
        num_smem_capacity,
        occupancy,
    ):
        """Compute pipeline stages.

        Same budget rule as the BF16 GLU kernel. GLU stages one 128 x 64 C buffer and one
        128 x 32 D buffer (16 KB + 8 KB = 24 KB in BF16); this kernel stages two 128 x 32 C and
        two 128 x 32 D buffers (4 x 8 KB = 32 KB), because every subtile issues two TMA stores and
        with a single stage each store had to finish reading shared memory before the next
        subtile could proceed. A 256 x 256 2-CTA tile still keeps 6 A/B stages (32 KB each) in
        227 KB.
        """
        num_acc_stage = 2
        num_c_stage = 2
        num_d_stage = 2
        num_tile_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_stage_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        c_smem_layout_stage_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
        d_smem_layout_stage_one = sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)

        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(b_dtype, b_smem_layout_stage_one)
        mbar_helpers_bytes = 1024
        sinfo_bytes = 4 * 4 * num_tile_stage
        c_bytes = cute.size_in_bytes(c_dtype, c_smem_layout_stage_one) * num_c_stage
        d_bytes = cute.size_in_bytes(d_dtype, d_smem_layout_stage_one) * num_d_stage

        epi_bytes = c_bytes + d_bytes
        num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)) // ab_bytes_per_stage
        return num_acc_stage, num_ab_stage, num_c_stage, num_d_stage, num_tile_stage

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
        """Pre-main-kernel initialization.

        Launched with grid=(expert_cnt, 1, 1) for discrete mode, or grid=(1, 1, 1) for
        dense+dynamic mode.

        Discrete weight: each block builds a B TMA descriptor for one expert.
        Dynamic sched: block 0 resets the atomic tile counter to 0.
        """
        expert_idx = cute.arch.block_idx()[0]

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            # Discrete mode stores one B TMA descriptor per expert in the descriptor
            # workspace. The main kernel later reuses those descriptors through the
            # scheduler extension.
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
        b,  # Dense: cute.Tensor (N, K, L) | Discrete: cute.Pointer to int64[]
        n: Int32,  # Ignored for dense mode
        k: Int32,  # Ignored for dense mode
        b_stride_size: cutlass.Int64,  # Ignored for dense mode
        b_major_mode: cutlass.Constexpr,  # Ignored for dense mode
        workspace_ptr,
        c: cute.Tensor,
        d: cute.Tensor,
        padded_offsets: cute.Tensor,
        prob: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM.

        Dense mode: ``b`` is a 3-D cute.Tensor (N, K, L). Discrete mode: ``b`` is a
        cute.Pointer to a device int64[] array of per-expert base addresses; ``n``, ``k``,
        ``b_stride_size`` and ``b_major_mode`` describe the uniform per-expert layout.
        ``c`` receives bf16(Z_acc), ``d`` the weighted clamped sReLU of the accumulator.
        """
        self.a_dtype = a.element_type
        self.b_dtype = a.element_type
        self.c_dtype = c.element_type
        self.d_dtype = d.element_type
        self.a_major_mode = LayoutEnum.from_tensor(a).mma_major_mode()
        self.c_layout = LayoutEnum.from_tensor(c)
        self.d_layout = LayoutEnum.from_tensor(d)

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            self.b_major_mode = LayoutEnum.from_tensor(b).mma_major_mode()
        else:
            self.b_major_mode = b_major_mode

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        # C and D share the T2R/R2S copies, so they must have the same type and layout.
        if cutlass.const_expr(self.c_dtype != self.d_dtype):
            raise TypeError(f"C and D types must match: {self.c_dtype} != {self.d_dtype}")

        self._setup_attributes()

        # ---- B setup (mode-dependent) ----
        # Dense mode receives a normal (N,K,L) tensor. Discrete mode receives a device array
        # of per-expert B base addresses; we build a template tensor here only so the TMA
        # atom has the right dtype/layout at compile time.
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

        # Setup TMA store for C (z_save)
        c_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c, c_smem_layout, self.epi_tile)

        # Setup TMA store for D (h)
        d_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), d, d_smem_layout, self.epi_tile)

        # ---- Helper kernel: TMA desc init (discrete) + sched counter reset (dynamic) ----
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

        # ---- Grid computation via MoE scheduler ----
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            b_n, b_k, b_l = cute.shape(b)
            sched_expert_shape = (self.expert_cnt, b_n, b_k)
        else:
            sched_expert_shape = (self.expert_cnt, n, k)

        sched_params = MoESchedulerParams(
            scenario="2Dx3D",
            expert_shape=sched_expert_shape,
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            use_dynamic_sched=self.use_dynamic_sched,
        )
        self.sched_params, grid = compute_grid(sched_params, max_active_clusters, self.use_2cta_instrs)

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_tile_stage, self.use_dynamic_sched)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            scheduler: SchedulerStorage
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

        # Launch the kernel asynchronously on ``stream``
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
            prob,
            workspace_ptr,  # Contains per-expert B TMA descriptors
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
        padded_offsets: cute.Tensor,
        prob: cute.Tensor,
        workspace_ptr,  # Pointer to TMA descriptor workspace (from helper_kernel)
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        sched_params: MoESchedulerParams,
        release_zero_arg: cutlass.Int32,
    ):
        """GPU device kernel performing the persistent grouped GEMM + sReLU epilogue.

        ``release_zero_arg`` is tile_dsl.consume's release-zero, the last argument: the host passes the
        literal 0 and its one use is ``release_zero`` below (python/cudnn/AGENTS.md Rule 11: a compare or
        a branch on it lets the compiler fold the mask, and the sInfo loads are unordered again).
        """
        rz = release_zero(release_zero_arg)
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
                cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
            cpasync.prefetch_descriptor(tma_atom_d)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        total_token = padded_offsets[self.expert_cnt - 1]

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

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
        # Setup smem tensor A/B/C/D
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
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
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
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

        # All experts empty: nothing to compute or store.
        if total_token <= 0:
            cute.arch.nvvm.exit()

        #
        # Specialized Schedule warp (MoE Persistent Tile Scheduler)
        #
        if warp_idx == self.sched_warp_id:
            work_tile_info = scheduler.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tile_stage)

            while work_tile_info.is_valid_tile:
                # Write MoEWorkTileInfo directly to sInfo:
                # sInfo[0] = expert_idx (>= 0 means valid)
                # sInfo[1] = tile_m_idx (CTA-level M tile index)
                # sInfo[2] = tile_n_idx
                # sInfo[3] = k_tile_cnt
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
                # sInfo format: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
                work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                k_tile_cnt = work_tile_info.k_tile_cnt
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

                # Convert CTA tile index to MMA tile index
                mma_tile_coord_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                mma_tile_coord_n = work_tile_info.tile_n_idx
                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_n, None, 0)]

                # Peek (try_wait) AB buffer empty
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    ab_producer_state_next = ab_producer_state.clone()
                    ab_producer_state_next.advance()
                    if ab_producer_state_next.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state_next)
                    else:
                        peek_ab_empty_status = cutlass.Boolean(1)

                    # TMA load A (contiguous, global desc via domain_offset)
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    # TMA load B (discrete: per-expert desc from workspace)
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                        tma_desc_ptr=desc_ptr_b,
                    )
                    ab_producer_state.advance()

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
            acd_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Get the first tile info from pipeline
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            consume_then_arrive(tile_info_pipeline.sync_object_empty.get_barrier(tile_info_consumer_state.index).raw_ptr(), tile_info.load(), rz)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                k_tile_cnt = tile_info[3]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = 0
                acd_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acd_producer_state)

                # Get accumulator stage index
                tCtAcc = tCtAcc_base[(None, None, None, acd_producer_state.index)]
                # Wait for accumulator buffer empty
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acd_producer_state, peek_acc_empty_status)

                #
                # Mma mainloop; reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        ab_consumer_state_next = ab_consumer_state.clone()
                        ab_consumer_state_next.advance()
                        if ab_consumer_state_next.count < k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state_next)

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)

                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state = ab_consumer_state_next

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acd_producer_state)

                # Peek (try_wait) Acc buffer empty for the next tile
                acd_producer_state.advance()
                if acd_producer_state.count < k_tile_cnt:
                    if is_leader_cta:
                        peek_acc_empty_status = acc_pipeline.producer_try_acquire(acd_producer_state)

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
            acc_pipeline.producer_tail(acd_producer_state)

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
            # Partition for epilogue (shape-only: use global tensor for invariant setup).
            # C and D have the accumulator's N extent, so both use the MMA tiler.
            #
            epi_tidx = tidx
            thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)
            gD_mnl_shape = cute.local_tile(mD_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
            tCgD_shape = thr_mma_epi.partition_C(gD_mnl_shape)

            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, tCgD_shape, epi_tile, use_2cta_instrs)

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, epi_tidx, sC)
            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rD, epi_tidx, sD)

            epi_ext = self._make_extension(workspace_ptr)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            # Threads/warps participating in the TMA store pipelines
            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilog_warp_id))
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)
            d_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilog_warp_id))
            d_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_d_stage, producer_group=d_producer_group)

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

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                # sInfo format: (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)
                epi_work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                epi_ext.update_expert_info(padded_offsets, epi_work_tile_info.expert_idx)

                # Per-expert C/D tensors via extension (domain offset by the expert's first row)
                real_c, _ = epi_ext.get_gmem_tensor("c", mC_mnl, padded_offsets, epi_work_tile_info)
                real_d, _ = epi_ext.get_gmem_tensor("d", mD_mnl, padded_offsets, epi_work_tile_info)

                # local_tile + partition on per-expert tensors
                thr_mma_epi_loop = tiled_mma.get_slice(mma_tile_coord_v)
                gC_mnl_loop = cute.local_tile(real_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                tCgC_loop = thr_mma_epi_loop.partition_C(gC_mnl_loop)
                _, bSG_sC, bSG_gC_partitioned = epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC_loop, epi_tile, sC)
                gD_mnl_loop = cute.local_tile(real_d, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                tCgD_loop = thr_mma_epi_loop.partition_C(gD_mnl_loop)
                _, bSG_sD, bSG_gD_partitioned = epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d, tCgD_loop, epi_tile, sD)

                # Slice to per-expert tile coords (L=0, domain already offset)
                epi_mma_tile_coord = (
                    epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    epi_work_tile_info.tile_n_idx,
                    cutlass.Int32(0),
                )
                bSG_gC = bSG_gC_partitioned[(None, None, None, *epi_mma_tile_coord)]
                bSG_gD = bSG_gD_partitioned[(None, None, None, *epi_mma_tile_coord)]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))

                #
                # Row prob (per-expert via domain_offset). Each epilogue thread owns one
                # accumulator row (TMEM lane), i.e. T2R_M/EPI_M is 1.
                #
                mPosition = epi_work_tile_info.tile_m_idx * self.cta_tile_shape_mnk[0] + tidx
                real_prob, _ = epi_ext.get_gmem_tensor("prob", prob, padded_offsets, epi_work_tile_info)
                mProb = real_prob[mPosition, 0, 0]

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                #
                # Store the tile in 128 x 32 subtiles: C = bf16(acc), then D = bf16(A * p).
                # C and D each use two shared-memory stages indexed by num_prev_subtiles.
                # Both TMA stores are issued by the same thread, so the wait_group.read 1 after
                # each commit also retires the other pipeline's previous store: the stage about
                # to be overwritten (two stores back) is always free.
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                for subtile_idx in cutlass.range(0, subtile_cnt, 1, unroll=1):
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    epi_buffer = num_prev_subtiles % self.num_d_stage
                    num_prev_subtiles = num_prev_subtiles + 1

                    #
                    # C = Z_save = bf16_rn(Z_acc)
                    #
                    acc_vec_c = tiled_copy_r2s.retile(tTR_rAcc).load()
                    tRS_rC.store(acc_vec_c.to(self.c_dtype))
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, epi_buffer)])
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy("async.shared", space="cta")
                    named_barrier_sync_unaligned(self.epilog_sync_barrier.barrier_id, self.epilog_sync_barrier.num_threads)
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(tma_atom_c, bSG_sC[(None, epi_buffer)], bSG_gC[(None, subtile_idx)])
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    named_barrier_sync_unaligned(self.epilog_sync_barrier.barrier_id, self.epilog_sync_barrier.num_threads)

                    #
                    # D = H = bf16_rn(A(z_in) * p), elementwise in FP32
                    #
                    acc_vec = tTR_rAcc.load()
                    tCompute = cute.make_rmem_tensor(acc_vec.shape, self.acc_dtype)
                    for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                        tCompute[i] = weighted_clamped_srelu_f32(
                            acc_vec[i],
                            mProb,
                            clamp_scale=self.clamp_scale,
                            act_input=self.act_input,
                            tanh_mode=self.tanh_mode,
                        )

                    h_vec = tiled_copy_r2s.retile(tCompute).load()
                    tRS_rD.store(h_vec.to(self.d_dtype))
                    cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[(None, None, None, epi_buffer)])
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy("async.shared", space="cta")
                    named_barrier_sync_unaligned(self.epilog_sync_barrier.barrier_id, self.epilog_sync_barrier.num_threads)
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(tma_atom_d, bSG_sD[(None, epi_buffer)], bSG_gD[(None, subtile_idx)])
                        d_pipeline.producer_commit()
                        d_pipeline.producer_acquire()
                    named_barrier_sync_unaligned(self.epilog_sync_barrier.barrier_id, self.epilog_sync_barrier.num_threads)

                #
                # Async arrive accumulator buffer empty
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

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
            # Wait for C/D stores to complete
            #
            c_pipeline.producer_tail()
            d_pipeline.producer_tail()

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make the TMEM-load tiled copy and partition TMEM (source) and registers (dest).

        :return: (tiled_copy_t2r, tTR_tAcc, tTR_rAcc): the TMEM->register tiled copy, the
            partitioned accumulator (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE) and the FP32
            register fragment of one epilogue subtile.
        """
        # Make tiledCopy for tensor memory load
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

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gD_mnl_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gD = thr_copy_t2r.partition_D(gD_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gD[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rX: cute.Tensor,
        tidx: cutlass.Int32,
        sX: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make the register->SMEM tiled copy and partition registers (source) and SMEM (dest).

        :return: (tiled_copy_r2s, tRS_rX, tRS_sX) with tRS_sX shaped (R2S, R2S_M, R2S_N, STAGE).
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.d_layout, self.d_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sX = thr_copy_r2s.partition_D(sX)
        # (R2S, R2S_M, R2S_N)
        tRS_rX = tiled_copy_r2s.retile(tTR_rX)
        return tiled_copy_r2s, tRS_rX, tRS_sX
