# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage (1) of the gated attention block on the FP8 pipeline, with stages (2)+(3)
AND the Q/K/V quantization FUSED into its epilogue.

``PROJ = alpha * (h8 @ W8_qkvg^T)`` as the shipped FROST FP8 GEMM computes it
(e4m3 x e4m3, fp32 accumulate, the per-tensor descale ``alpha`` folded into the
epilogue), and then, per tile class:

  Q / K   per-head RMSNorm over D on the fp32 accumulator, partial RoPE on the
          leading ``ROPE_DIM`` columns, then ``e4m3_satfinite(x * scale_q|k)``
          into its OWN compact e4m3 tensor: ``q8 [T, h_q*d]`` / ``k8 [T, h_kv*d]``.
  V       ``e4m3_satfinite(alpha * acc * scale_v)`` into ``v8 [T, h_kv*d]``.
  GATE    the rendering's own epilogue -- ``bf16(alpha * acc)`` -- into
          ``gate16 [T, h_q*d]`` bf16.

So the block's whole stage-1 chain (GEMM + descale, norm + RoPE, three quantize
passes) is ONE launch writing FOUR compact per-tensor outputs -- each one ==
compact BSHD ``[B, S, H, d]``, exactly what the unfused path hands the FP8 SDPA
(``token_stride = 0``; the round-1 ``[T, N_QKV]`` slab + strided SDPA reads cost
+7 % at 32K once the gate stream evicted the 9216-B-strided K/V lines from L2 --
STATUS.md "SDPA decomposition").  Output layout (``proj_gemm.py`` / the FP8
fusion design contract, ROUND 2): tile class -> descriptor + column remap,
``q8 col = c``, ``k8 col = c - OFF_K``, ``v8 col = c - OFF_V``,
``gate16 col = c - OFF_GATE``; every descriptor's ``global_dims[0]`` is ITS OWN
width so the TMA OOB clip is right per tensor.

NUMERICS (one rounding per output; the oracle is the fp32 chain quantized once)
------------------------------------------------------------------------------
* ``rstd = rsqrt(sum(acc^2) * alpha^2 / D + eps)`` -- alpha MUST enter the eps
  term this way (``rstd(alpha*x) = rsqrt(alpha^2 * mean(x^2) + eps)``); norming
  the raw accumulator and descaling afterwards is a different function.  It is
  folded with the quant scale into ONE per-row multiplier
  ``srow = alpha * rstd * scale_q|k`` so pass B costs exactly what the bf16 fork's
  ``acc * rstd * w`` does.
* e4m3 conversion is ``cvt.rn.satfinite.e4m3x2.f32`` (``pointwise.fp32_to_fp8_pack``,
  bit-exact vs torch's ``.to(float8_e4m3fn)`` after a clamp to +-448).  The packed
  ``Int32`` words are BITCAST to a ``Float8E4M3FN`` vector before
  ``store_swizzled`` -- which otherwise value-casts to the pointer dtype.
* ``alpha, scale_q, scale_k, scale_v`` are a ``[4]`` fp32 DEVICE tensor
  (``qscal``), never module parameters: a re-calibration must not re-JIT, and a
  constant float reaching the pack asm becomes an immediate that ICEs libNVVM
  (``frost-tile-dsl.md`` S7).  It is read ONCE PER TILE in the pre-wait
  classification block (one ``ld.global.v4`` next to the ``ldg_early`` cos/sin
  loads, BEFORE the ``acc_full`` wait) and reduced there to the four per-tile
  scalars the arms consume: ``alpha`` and the tile's class scale
  (``qscal[1|2|3]``, selected on the ADDRESS so only two loaded registers exist)
  -- 2 registers live across the wait; ``alpha*scale_v`` and ``alpha^2/D`` are
  register arithmetic at the point of use.  The two placements that are NOT
  allowed, both measured: once before the persistent loop (spilled 20 cos/sin
  words), and per arm AFTER the wait (round 1: a dependent L2 round trip per
  tile -- the oversized-SMEM carveout leaves 8 KB of L1, which the per-tile
  cos/sin traffic evicts -- cost +21..26 % of the GEMM while the loads-deleted
  ``const`` arm sat at the unfused floor; STATUS.md "GEMM A/B").  No arm issues
  a global load that the epilogue's critical path depends on after the wait.
  The two registers are paid for by the pass-A load granule (``_PASS_A_SPLIT``);
  the epilogue's real register ceiling is 255, not ``epi_reg_count`` -- see the
  pass-A comment.

WHY THE FUSION IS TILE-LOCAL, AND THE EPILOGUE PER TILE
--------------------------------------------------------
Identical to the bf16 fork ``proj_gemm_norm_rope.py`` (read its docstring):
config ``CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma`` gives one CTA
output tile == one head (128 rows x 256 cols), so the RMSNorm reduction and the
rotate_half partner are THREAD-LOCAL.  The tile class (Q / K / V / GATE) is a
function of ``coord_n_c`` alone, decided BEFORE the accumulator wait, and is
warp-uniform, so the three arms are plain runtime ``if`` branches.  Every arm
runs the IDENTICAL per-subtile synchronization skeleton: ring-slot advance,
``fence_view_async_shared``, both ``EPI_SYNC_BAR_ID`` barriers, warp-0 TMA
store + commit + ``cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1)``, and the
``acc_empty`` arrive after the LAST ``tcgen05.ld`` of the tile.

The e4m3 subtile (128 rows x 32 B = 4 KiB) reuses the SAME SMEM-D ring slot the
bf16 subtile (8 KiB) uses, viewed as ``Float8E4M3FN``, staged with the shipped
compiler's own e4m3 TMA-store arm: ``store_swizzled(alignment=32,
Swizzle(1, 4, 3))`` against a descriptor ``dtype=Float8E4M3FN, box=[32, 128, 1],
swizzle=s32b`` whose ``global_dims[0] = N_QKV`` (not N) so the OOB clip is right.

``NORM_SOURCE``: ``ldg_early`` (DEFAULT, cos/sin issued before the accumulator
wait), ``ldg`` (issued in the arm; the A/B reference), ``const`` (the
loads-deleted diagnostic floor -- NOT correct), and the two SPLITS of ``const``
by load class -- diagnostic floors, NOT correct either: ``const_w`` (cos/sin
loaded exactly as ``ldg_early``, the per-subtile norm WEIGHT replaced by
``const``'s substitute) and ``const_cs`` (the weight loaded exactly as
``ldg_early``, cos/sin replaced by ``const``'s substitutes).  ``ldg_early -
const_w`` prices the weight loads, ``ldg_early - const_cs`` the cos/sin gather.
There is no ``off`` here: the
FP8 fork has no single bf16 slab to degenerate to.  The norm WEIGHT stays a
per-subtile L1 hit -- hoisting it spilled 44 STL / 88 LDL on the bf16 fork
(``frost-tile-dsl.md`` S11); not retried.  Tables (cos / sin / w) are bf16, so
their byte width is PINNED to 2 (``_ACT_BPE``), NOT derived from ``cd_dtype``.

BARRIER TABLE: UNCHANGED.  SMEM TABLE: UNCHANGED.  This fork adds no mbarrier,
no named barrier and no SMEM buffer.  It adds THREE TMA-store descriptors (the
rendering has one: q8 / k8 / v8 e4m3 + the bf16 gate) and per-lane global loads
(cos/sin/w, the ``[4]`` scales).

KEEP IN SYNC WITH ``gemm/frost/kernel_templates/sm100_matmul.py`` AND THE bf16 FORK
------------------------------------------------------------------------------------
This file is the rendered FP8+alpha expansion of that template for the config
above -- ``frost_dev/renderings/proj_gemm_fp8_alpha_rendered.py`` (e4m3 A/B,
``mma_kind=F8F6F4``, ``cta_tile_mnk=(128, 128, 128)``, ``mma_inst K=32``, the
``qkv_gate_proj_alpha`` aux) -- plus the fusion DELTA of the bf16 fork
(``diff frost_dev/renderings/proj_gemm_bf16_rendered.py
kernels/proj_gemm_norm_rope.py``) re-applied between the ``FP8 NORM+ROPE+QUANT
FUSION`` markers, with the alpha aux replaced by the ``qscal`` read.  It was NOT
produced by patching the bf16 fork's constants header (``frost-tile-dsl.md``
S5: diff the RENDERINGS, not the edit).  Mainloop / scheduler / TMEM pipeline /
register split are the rendering's, verbatim; any fix there applies to ALL
THREE files.

Parameters come from the FROST template loader as ``FROST_TEMPLATE_PARAMS``
(a :class:`~cudnn.gated_attention_block.kernels.proj_gemm.NormRopeFusionParams`
with ``quant_fp8=True``); a plain import gets the 397B geometry so the file runs
standalone.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp8_pack
from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v4
from cudnn.gated_attention_block.kernels.proj_gemm import NormRopeFusionParams, validate_norm_rope_params

import cutlass.experimental.primitives as nvvm
from cudnn.gemm.frost.sm100.kernel_templates._tile_helpers import (
    epi_subtile_spans as _epi_subtile_spans,
    l2_swizzle_tile as _l2_swizzle_tile,
    tcgen05_alloc as _tcgen05_alloc,
    tcgen05_dealloc as _tcgen05_dealloc,
    tcgen05_mma as _tcgen05_mma,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass._mlir_helpers.vector as _cvec
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda
from cutlass.cute.arch import clc as cute_clc

# Tile config: CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma
mma_inst_shape_mnk = (256, 256, 32)
cta_group = 2
cgrp_tile_mnk = (256, 256, 128)
cta_tile_mnk = (128, 128, 128)
epi_tile_mn = (128, 32)
threads_per_cta = 256
cluster_shape_mnk = (2, 1, 1)
matmul_a_batch = 1
matmul_b_batch = 1
a_is_m_major = False
b_is_n_major = False
mma_a_major = 0
mma_b_major = 0
ab_stages = 10
b_collector_ok = True
multicast_a = False
multicast_b = False
a_mcast_slices = 1
b_mcast_slices = 1
ab_empty_full_mask = False
ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B
a_smem_desc_leading_byte_offset = 16
a_smem_desc_stride_byte_offset = 1024
a_smem_k_step_bytes = 32
a_smem_m_step_bytes = 16384
a_tma_group_elems = 1
b_smem_desc_leading_byte_offset = 16
b_smem_desc_stride_byte_offset = 1024
b_smem_k_step_bytes = 32
b_tma_group_elems = 1
mma_size_m = 1
mma_size_n = 1
mma_size_k = 4
ab_tma_swizzle = _tma.TensorMapSwizzle.s128b

# Dtype family: A=fp8_e4m3->MMAfp8_e4m3, B=fp8_e4m3->MMAfp8_e4m3, out=bf16 (K_BYTES=128)
ab_dtype = cutlass.Float8E4M3FN
cd_dtype = cutlass.BFloat16
epi_store_dtype = cutlass.BFloat16
mma_a_dtype = cutlass.Float8E4M3FN
mma_b_dtype = cutlass.Float8E4M3FN
mma_c_dtype = cutlass.Float32
acc_widen_to_fp32 = False
ab_tma_dtype = cutlass.Float8E4M3FN
mma_kind = nvvm.Tcgen05MMAKind.F8F6F4
epi_n = 32
epi_row_elems = 32
tile_swizzle_n = 0
swizzle_l2_budget_bytes = 41943040
num_gemms = 1
num_a_operands = 1
num_b_operands = 1
gemm_a_idx = (0,)
gemm_b_idx = (0,)
num_tmem_alloc_cols = 576
tmem_alloc_exclusive = True
acc_stages = 2  # 256 acc cols/stage
vec_bytes_epi = 32
split_k_slices = 1
frost_compile_options = "--enable-tvm-ffi --gpu-arch sm_107a"
n_tma_outputs = 4  # FP8 NORM+ROPE+QUANT FUSION: compact e4m3 q8 / k8 / v8 + bf16 GATE (rendering: 1); the prefetch loop covers all four
moe_aligned_offsets = False
epi_slot_widen = 1
epi_packed_lanes = False
epi_dp22 = False
epi_stage_rows = 128
epi_chunk_elems = 32
ab_stages = 9  # SMEM-D 16400B fixed + cast LOAD 0B/stage + multi-GEMM 0B/stage
fallback_cluster_shape_mnk = None
mixed_a_pattern_pref = 1
mixed_b_pattern_pref = 1
mixed_a_pattern_fb = 1
mixed_b_pattern_fb = 1

# ---------------------------------------------------------------------------
# FP8 NORM+ROPE+QUANT FUSION: geometry + knobs (compile-time; every use is const_expr)
# ---------------------------------------------------------------------------
PARAMS: NormRopeFusionParams = globals().get("FROST_TEMPLATE_PARAMS", NormRopeFusionParams(quant_fp8=True))
if not PARAMS.quant_fp8:
    raise ValueError(f"{__name__}: this is the FP8 fork; it needs NormRopeFusionParams(quant_fp8=True) (the bf16 fork is proj_gemm_norm_rope.py)")
validate_norm_rope_params(PARAMS)
_D: int = PARAMS.d_head
_ROPE_DIM: int = PARAMS.rope_dim
_H_Q: int = PARAMS.h_q
_H_KV: int = PARAMS.h_kv
_OFF_Q, _OFF_GATE, _OFF_K, _OFF_V = PARAMS.offsets
_N_QKVG: int = PARAMS.n_qkvg
_N_GATE: int = PARAMS.n_gate  # bf16 gate16 width (= h_q * d)
_N_Q: int = _H_Q * _D  # e4m3 q8 width (== _N_GATE; kept apart so each descriptor names its own extent)
_N_KV: int = _H_KV * _D  # e4m3 k8 / v8 width
_EPS: float = PARAMS.eps
NORM_SOURCE: str = PARAMS.norm_source  # "ldg" | "ldg_early" | "const" | "const_w" | "const_cs"  (no "off" on the FP8 fork)
# False: the RoPE-ONLY Q/K arm -- no pass A (sum of squares), no rsqrt, no norm-weight loads;
# the per-row multiplier collapses to alpha * scale_q|k (== the V arm's `_s_v` form) and the
# Q/K tiles are descaled, rotated on the fp32 accumulator and quantized.  Every norm-only
# block below sits under const_expr(_QK_NORM); the weights are None at the ABI.
_QK_NORM: bool = PARAMS.qk_norm
# The two epilogue load classes are gated SEPARATELY so a diagnostic arm can delete one at a time:
#   _LDG_W    real per-subtile norm-weight loads  (w[col]: 64 B, same address for all lanes, after the wait)
#   _LDG_CS   real per-row cos/sin gather          (ld_global_v4 of the bf16 tables, 32 KiB per tile)
#   _CS_EARLY the cos/sin gather is issued BEFORE the accumulator wait (ldg_early); "ldg" issues it in the arm
# ldg_early / ldg / const trace EXACTLY as before (both classes real / real / deleted).  const_w = cos/sin as
# ldg_early with the weight replaced by const's substitute; const_cs = the weight as ldg_early with cos/sin
# replaced by const's substitutes.  Like const, const_w / const_cs are diagnostic floors and NOT correct.
_LDG_W: bool = NORM_SOURCE in ("ldg", "ldg_early", "const_cs")
_LDG_CS: bool = NORM_SOURCE in ("ldg", "ldg_early", "const_w")
_CS_EARLY: bool = NORM_SOURCE in ("ldg_early", "const_w")
_ROPE_SUBTILES: int = _ROPE_DIM // epi_n
_ROPE_HALF_SUBTILES: int = _ROPE_SUBTILES // 2
_ROPE_HALF: int = _ROPE_DIM // 2
# cos / sin / norm-weight tables are bf16 REGARDLESS of the GEMM dtypes: pin their
# byte width (deriving it from cd_dtype / ab_dtype halves the ld.global strides).
_ACT_BPE: int = 2
_E4M3_BPE: int = 1
# Pass A (sum of squares) reads each 32-column subtile as _PASS_A_SPLIT tcgen05.ld's of
# 32 / _PASS_A_SPLIT columns.  ptxas batches a fixed NUMBER of LDTMs ahead of the math,
# so the granule sets how many accumulator registers are in flight in pass A: 2 (16-wide,
# round 1) left 128 live and spilled once the round-2 pre-wait scale registers were added;
# 4 (8-wide) halves that.  Same values, same reduction (see the pass-A comment).
_PASS_A_SPLIT: int = 4
# The facts the fusion stands on, checked against THIS rendering's constants.
if epi_dp22 or (cgrp_tile_mnk[1] // cluster_shape_mnk[1]) != _D:
    raise ValueError(
        f"{__name__}: the fused epilogue needs one CTA output tile == one head: per-CTA N tile "
        f"{cgrp_tile_mnk[1] // cluster_shape_mnk[1]} (epi_dp22={epi_dp22}) vs d_head={_D}. Re-render for another config."
    )
if _ROPE_DIM % (2 * epi_n) != 0 or _ROPE_DIM >= _D:
    raise ValueError(f"{__name__}: rope_dim={_ROPE_DIM} must be a multiple of {2 * epi_n} (two whole subtiles per rotate_half pair) and < d_head={_D}")
if ab_dtype is not cutlass.Float8E4M3FN or mma_c_dtype is not cutlass.Float32 or cd_dtype is not cutlass.BFloat16 or acc_widen_to_fp32:
    raise ValueError(
        f"{__name__}: this fork was rendered for an e4m3 x e4m3 -> fp32 GEMM with a bf16 C (the gate slab); the rendering's constants say otherwise"
    )
if epi_packed_lanes or mma_size_m != 1 or num_gemms != 1 or epi_n != 32 or epi_tile_mn[0] != 128:
    raise ValueError(f"{__name__}: this fork needs the plain thread-per-row single-GEMM epilogue with 32-column subtiles (two e4m3 packs per subtile)")
if _N_Q % 32 or _N_KV % 32 or _N_GATE % 16:
    raise ValueError(f"{__name__}: output widths must be whole TMA rows: q8={_N_Q} / k8=v8={_N_KV} (32 e4m3), gate16={_N_GATE} (16 bf16)")


def _e4m3x32(elems):
    """32 fp32 -> one ``Vector[Float8E4M3FN, 32]`` via two ``cvt.rn.satfinite.e4m3x2.f32`` packs.

    ``fp32_to_fp8_pack`` returns 4 packed ``Int32`` words per 16 values in torch
    byte order; the BITCAST (not ``.to``) keeps the bit patterns -- a numeric cast
    of the words, which ``store_swizzled`` would do on an Int32 vector against a
    Float8E4M3FN pointer, is garbage.
    """
    _p0 = fp32_to_fp8_pack(elems[0:16], dtype=cutlass.Float8E4M3FN)
    _p1 = fp32_to_fp8_pack(elems[16:32], dtype=cutlass.Float8E4M3FN)
    return cutlass.Vector.from_elements((_p0[0], _p0[1], _p0[2], _p0[3], _p1[0], _p1[1], _p1[2], _p1[3]), cutlass.Int32).bitcast(cutlass.Float8E4M3FN)


# Rank decomposition below uses shifts and masks instead of runtime integer
# division.  The catalog satisfies this; keep synthesized configs from silently
# taking the fast path with a non-power-of-two cluster dimension.
if any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in cluster_shape_mnk[:2]):
    raise NotImplementedError(f"{__name__}: cluster M/N dimensions must be powers of two")
if fallback_cluster_shape_mnk is not None and any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in fallback_cluster_shape_mnk[:2]):
    raise NotImplementedError(f"{__name__}: fallback cluster M/N dimensions must be powers of two")

# Keep the two launch alternatives as host constants and spell the preferred /
# fallback operations at each use site. This exposes constant masks and shift
# alternatives before backend canonicalization.
_preferred_cluster_m_shift = cluster_shape_mnk[0].bit_length() - 1
_preferred_cluster_n_shift = cluster_shape_mnk[1].bit_length() - 1
_fallback_cluster_m_shift = _preferred_cluster_m_shift if fallback_cluster_shape_mnk is None else fallback_cluster_shape_mnk[0].bit_length() - 1
_fallback_cluster_n_shift = _preferred_cluster_n_shift if fallback_cluster_shape_mnk is None else fallback_cluster_shape_mnk[1].bit_length() - 1
_CTA_GROUP = nvvm.CTAGroup.CTA_2 if cta_group == 2 else nvvm.CTAGroup.CTA_1
_cta_group_shift = cta_group.bit_length() - 1


# Scheduler ring depth.
CLC_SCHED_STAGES = 1

# Programmatic Dependent Launch (PDL, sm_90+).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path.
EPI_SMEM_STAGES = 2

# Named barrier id for the 4-warp epilogue handoff around the TMA store.
EPI_SYNC_BAR_ID = 1

# Named barrier id for the TMEM-alloc handoff.
TMEM_ALLOC_BARRIER_ID = 2


@cute.jit
def _auto_swizzle_w(m, n, k, nt_n):
    """N-super-block width for the tile rasterization, resolved per launch.

    ``tile_swizzle_n > 0`` pins it. Otherwise: the walk keeps one operand slice
    resident and re-reads the other every super-block, so block along the SHORTER
    problem side. Once that side outgrows what L2 can hold onto while C streams
    through it, keeping it is no longer free -- fall back to the widest N block the
    budget does cover.
    """
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_dtype.width) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    if cutlass.min(m, n) * row_bytes <= budget and m <= n:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


def _a_collector_op(g):
    if cutlass.const_expr(num_gemms == 1 or num_a_operands != 1 or mma_size_m != 1):
        return None
    if cutlass.const_expr(g == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(g == num_gemms - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


def _b_collector_op(mi):
    if cutlass.const_expr(not b_collector_ok or mma_size_m == 1):
        return None
    if cutlass.const_expr(mi == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(mi == mma_size_m - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


@cute.kernel
def frost_sm100_matmul_128x256x128_128x256x32_cluster2x1_2ctamma(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_l_0: cutlass.Int64,
    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],  # e4m3 q8 [N_Q, M]
    # ---- FP8 NORM+ROPE+QUANT FUSION: appended, so the rendered prefix stays diffable ----
    tma_c_desc_1: cutlass.GridConstant[_tma.TensorMap],  # e4m3 k8 [N_KV, M]
    tma_c_desc_2: cutlass.GridConstant[_tma.TensorMap],  # e4m3 v8 [N_KV, M]
    tma_c_desc_3: cutlass.GridConstant[_tma.TensorMap],  # bf16 gate16 [N_GATE, M] (the rendering's own C descriptor form)
    w_q_norm: Optional[cute.Tensor],  # [D]      bf16, or None (qk_norm=False: folds out)
    w_k_norm: Optional[cute.Tensor],  # [D]      bf16, or None
    cos_tab: cute.Tensor,  # [T, ROPE_DIM] bf16, per-token, halves duplicated
    sin_tab: cute.Tensor,  # [T, ROPE_DIM]
    qscal: cute.Tensor,  # [4] fp32: alpha (= descale_h * descale_w), scale_q, scale_k, scale_v
) -> None:
    tma_a_descs = [tma_a_desc_0]
    tma_b_descs = [tma_b_desc_0]
    tma_c_descs = [tma_c_desc_0, tma_c_desc_1, tma_c_desc_2, tma_c_desc_3]

    mma_warp_id = 4
    tma_warp_id = 5
    scheduler_warp_id = 6
    unused_warp_id = 7
    num_epilogue_warps = 4
    epi_reg_count = 232
    prod_reg_count = 24

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]
    gridx = cute.arch.grid_dim()[0]
    gridy = cute.arch.grid_dim()[1]

    # Mixed CGA: the launch carries a preferred (wide) cluster plus a smaller
    # fallback one, and the device picks per cluster — a CTA can only tell which
    # by reading the hardware cluster dims. Everything cluster-shaped below then
    # follows from those, so the two kinds share one body; only the multicast bit
    # patterns are loop-built and come in precomputed per shape.
    a_mcast_pattern = mixed_a_pattern_pref
    if cutlass.const_expr(cta_group == 2):
        b_mcast_pattern = mixed_b_pattern_pref
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        cluster_m = cluster_shape_mnk[0]
        cluster_n = cluster_shape_mnk[1]
    else:
        cdim_x, cdim_y, _cdim_z = cute.arch.block_in_cluster_dim()
        cluster_m = cdim_x
        cluster_n = cdim_y
        a_mcast_pattern = cutlass.Int32(mixed_a_pattern_pref)
        if cutlass.const_expr(cta_group == 2):
            b_mcast_pattern = cutlass.Int32(mixed_b_pattern_pref)
        # Bitwise, not `or`: both operands are runtime Booleans (this is the form
        # cutlass.cute.experimental.is_preferred_cluster uses).
        if (cdim_x != cluster_shape_mnk[0]) | (cdim_y != cluster_shape_mnk[1]):
            a_mcast_pattern = cutlass.Int32(mixed_a_pattern_fb)
            if cutlass.const_expr(cta_group == 2):
                b_mcast_pattern = cutlass.Int32(mixed_b_pattern_fb)
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    # Every catalog cluster dimension is a power of two.  Mixed-CGA makes the
    # divisor runtime-visible, so spelling rank decomposition as div/mod would
    # otherwise lower to reciprocal-based integer division in every warp.
    m_rank = cta_rank_in_cluster & (cluster_shape_mnk[0] - 1)
    n_rank = cta_rank_in_cluster >> _preferred_cluster_m_shift
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
            m_rank = cta_rank_in_cluster & (fallback_cluster_shape_mnk[0] - 1)
            n_rank = cta_rank_in_cluster >> _fallback_cluster_m_shift

    if cutlass.const_expr(cta_group == 2):
        pair_member = m_rank % cta_group
        pair_m_idx = m_rank // cta_group
        is_pair_leader = pair_member == 0
        pair_leader_rank = pair_m_idx * cta_group + n_rank * cluster_m
    else:
        pair_member = 0
        pair_m_idx = m_rank
        is_pair_leader = True
        pair_leader_rank = cta_rank_in_cluster

    is_cluster_leader_cta = cta_rank_in_cluster == 0

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

        for _ci in cutlass.range_constexpr(n_tma_outputs):
            nvvm.prefetch_tensormap(tma_c_descs[_ci].get_ptr())

    init_raw_m = bidx >> _preferred_cluster_m_shift
    init_raw_n = bidy >> _preferred_cluster_n_shift
    init_nt_m = gridx >> _preferred_cluster_m_shift
    init_nt_n = gridy >> _preferred_cluster_n_shift
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
            init_raw_m = bidx >> _fallback_cluster_m_shift
            init_raw_n = bidy >> _fallback_cluster_n_shift
            init_nt_m = gridx >> _fallback_cluster_m_shift
            init_nt_n = gridy >> _fallback_cluster_n_shift
    swizzle_w = _auto_swizzle_w(m, n, k, init_nt_n)
    init_tile_m, init_tile_n = _l2_swizzle_tile(
        init_raw_m,
        init_raw_n,
        init_nt_m,
        init_nt_n,
        swizzle_w,
        identity=tile_swizzle_n == 1,
    )
    init_tile_l = bidz

    if cutlass.const_expr(cta_group == 1):
        a_pattern = a_mcast_pattern
        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
            b_pattern = (1 << cluster_m) - 1
        else:
            b_pattern = (cutlass.Int32(1) << cluster_m) - 1

        if cutlass.const_expr(multicast_a):
            tma_mcast_mask_a = cutlass.Int16(a_pattern) << m_rank
        else:
            tma_mcast_mask_a = cutlass.Int16(1) << cta_rank_in_cluster
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
        else:
            tma_mcast_mask_b = cutlass.Int16(1) << cta_rank_in_cluster
    else:
        if cutlass.const_expr(multicast_a):
            tma_mcast_mask_a = cutlass.Int16(a_mcast_pattern << m_rank)
        else:
            tma_mcast_mask_a = cutlass.Int16(1 << cta_rank_in_cluster)
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16((b_mcast_pattern << pair_member) << (n_rank * cluster_m))
        else:
            tma_mcast_mask_b = cutlass.Int16(1 << cta_rank_in_cluster)

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(cta_group == 2):
        tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    # CLC scheduler SMEM — 2-stage ring.
    _clc_response_raw = cutlass.Array(cutlass.Int128, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=16)
    clc_response_ptr_base = cute.make_ptr(
        cutlass.Int128,
        _clc_response_raw.data_ptr(),
        mem_space=cute.AddressSpace.smem,
    )
    clc_full_mbar_ptr = cutlass.Array(cutlass.Int64, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    clc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    clc_full_mbar_cute_base = cute.make_ptr(
        cutlass.Int64,
        clc_full_mbar_ptr.data_ptr(),
        mem_space=cute.AddressSpace.smem,
    )

    sA_elems = cta_tile_mnk[0] * cta_tile_mnk[2]
    sB_elems = cta_tile_mnk[1] * cta_tile_mnk[2]
    smem_a_list = [
        cutlass.Array(
            ab_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            ab_dtype,
            sB_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    # One epilogue subtile = one MMA-M block x 32 cols; the M blocks reuse it.
    # The ring slot is indexed by `tidx`, so its row count is the EPILOGUE THREAD
    # count -- which is epi_tile_mn[0] only when the MMA M block is 128.
    epi_subtile_elems = epi_stage_rows * epi_row_elems * epi_slot_widen
    smem_d_ptr = cutlass.Array(
        epi_store_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    acc_empty_count = num_epilogue_warps * cta_group
    if cutlass.const_expr(ab_empty_full_mask):
        if cutlass.const_expr(cta_group == 1):
            ab_empty_count = cluster_size
        else:
            ab_empty_count = cluster_size // cta_group
    else:
        if cutlass.const_expr(cta_group == 1):
            ab_empty_count = cluster_m + cluster_n - 1
        else:
            ab_empty_count = (cluster_m // cta_group) + cluster_n - 1
    num_consumer_warps_per_cta = 7
    clc_empty_count = num_consumer_warps_per_cta * cluster_size
    if warp_idx == 0:
        if cutlass.const_expr(cta_group == 2):
            if elect_one:
                nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
        for i in range(acc_stages):
            if elect_one:
                nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), acc_empty_count)
        for i in range(CLC_SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(clc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(clc_empty_mbar_ptr.subview(i), clc_empty_count)
    nvvm.fence_mbarrier_init()
    if cutlass.const_expr(cta_group == 1):

        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            nvvm.barrier_cluster_arrive_relaxed()
            nvvm.barrier_cluster_wait()
        else:
            nvvm.barrier_cta_sync(0)
    else:
        nvvm.barrier_cluster_arrive_relaxed()

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    if cutlass.const_expr(cta_group == 1):
        num_tma_copy_bytes = num_a_operands * sA_bytes + num_b_operands * sB_bytes
    else:
        num_tma_copy_bytes = (num_a_operands * sA_bytes + num_b_operands * sB_bytes) * 2

    # One descriptor for every MMA instruction of the tile — the CTA tile spans
    # mma_size_m of them, all the same shape.
    idesc = cutlass.experimental.primitives.Tcgen05InstrDesc.build(
        a_dtype=mma_a_dtype,
        b_dtype=mma_b_dtype,
        c_dtype=mma_c_dtype,
        n_dim=mma_inst_shape_mnk[1],
        m_dim=mma_inst_shape_mnk[0],
        a_major=mma_a_major,
        b_major=mma_b_major,
    )

    # Per-CTA logical tile — the cluster cancels out, so these stay compile-time
    # constants even when the cluster shape is only known at runtime.
    logical_cta_tile_m = cgrp_tile_mnk[0] // cluster_shape_mnk[0]
    logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_shape_mnk[1]
    pair_n_size = logical_cta_tile_n
    # Per-CTA output rows one MMA-M block covers. A 2-CTA pair splits M, so this
    # is the per-CTA mma_inst_m — half the instruction's hardware M.
    epi_rows_per_mma_m = cta_tile_mnk[0] // mma_size_m
    # TMEM accumulator layout, per acc stage: gemm g, M block mi -> columns
    # [g*cols_per_acc_stage + mi*epi_cols_per_mma_m, +epi_cols_per_mma_m), all at
    # TMEM lane base 0. N is NOT split across instructions, so the epilogue drains
    # a whole M block as one contiguous span.
    if cutlass.const_expr(epi_dp22):
        # cluster-MMA m=128: the pair also splits N, so each CTA drains N/2.
        epi_cols_per_mma_m = pair_n_size // 2
    else:
        epi_cols_per_mma_m = pair_n_size
    cols_per_acc_stage = mma_size_m * epi_cols_per_mma_m
    acc_region_cols = num_gemms * cols_per_acc_stage
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32

    if cutlass.const_expr(cta_group == 2):
        nvvm.barrier_cluster_wait()
        nvvm.barrier_cta_sync(0)

    pass

    vsize = epi_chunk_elems

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    # The tile this cluster owns spans its OWN cluster shape; both shapes walk
    # the grid as the identity map (tile == blockIdx), so they tile the problem
    # identically and every output tile is still covered exactly once.
    cgrp_tile_m_cur = logical_cta_tile_m * cluster_m
    cgrp_tile_n_cur = logical_cta_tile_n * cluster_n
    num_k_blocks = cta_tile_mnk[2] // mma_inst_shape_mnk[2]

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        sched_iter = cutlass.Int32(0)
        clc_empty_phase = cutlass.Int32(1)
        clc_full_phase = cutlass.Int32(0)
        is_valid_sched = cutlass.Int32(1)
        while is_valid_sched != 0:
            stage = sched_iter % CLC_SCHED_STAGES
            if stage == 0 and sched_iter != 0:
                clc_empty_phase = clc_empty_phase ^ 1
                clc_full_phase = clc_full_phase ^ 1

            if is_cluster_leader_cta:
                while not nvvm.mbarrier_try_wait_parity(clc_empty_mbar_ptr.subview(stage), clc_empty_phase, time_limit=10_000_000):
                    pass

            if elect_one:
                nvvm.mbarrier_arrive_expect_tx(clc_full_mbar_ptr.subview(stage), 16)

            if is_cluster_leader_cta:
                if elect_one:
                    cute_clc.issue_clc_query(
                        clc_full_mbar_cute_base + stage,
                        clc_response_ptr_base + stage,
                        multicast=True,
                    )

            while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(stage), clc_full_phase, time_limit=10_000_000):
                pass

            _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid_sched = vld

            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)

            sched_iter += 1

        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            if is_cluster_leader_cta:
                for _ in range(CLC_SCHED_STAGES):
                    stage = sched_iter % CLC_SCHED_STAGES
                    if stage == 0 and sched_iter != 0:
                        clc_empty_phase = clc_empty_phase ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        clc_empty_mbar_ptr.subview(stage),
                        clc_empty_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    sched_iter += 1

    if warp_idx == tma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ab_empty_phase_bit = cutlass.Int32(1)
        ab_iter = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_tma = cutlass.Int32(0)
        while is_valid != 0:
            coord_m_per_cta = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            if cutlass.const_expr(cta_group == 1):
                coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * cta_tile_mnk[1]
            else:
                coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]
            # Split-K: grid z carries batch*S
            if cutlass.const_expr(split_k_slices > 1):
                batch_tile_l = tile_l // split_k_slices
                split_idx = cutlass.Int64(tile_l % split_k_slices)
                k_tiles_per_split = num_k_tiles // split_k_slices
                k_tiles_remainder = num_k_tiles % split_k_slices
                k_begin = split_idx * k_tiles_per_split + cutlass.min(split_idx, k_tiles_remainder)
                k_end = (split_idx + 1) * k_tiles_per_split + cutlass.min(split_idx + 1, k_tiles_remainder)
            else:
                batch_tile_l = tile_l
                k_begin = cutlass.Int64(0)
                k_end = num_k_tiles
            if cutlass.const_expr(matmul_a_batch == 1):
                tile_l_a = cutlass.Int32(0)
            else:
                tile_l_a = batch_tile_l
            if cutlass.const_expr(matmul_b_batch == 1):
                tile_l_b = cutlass.Int32(0)
            else:
                tile_l_b = batch_tile_l

            for k_tile_idx in range(k_begin, k_end):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                    pass

                coord_k = k_tile_idx * cta_tile_mnk[2]
                if is_pair_leader:
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                for _ai in cutlass.range_constexpr(num_a_operands):
                    sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                    tma_a_desc = tma_a_descs[_ai]
                    if cutlass.const_expr(a_mcast_slices > 1):
                        _a_rows = cta_tile_mnk[0] // a_mcast_slices
                        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_stage.subview(n_rank * _a_rows * cta_tile_mnk[2]),
                                    tma_a_desc.get_ptr(),
                                    (coord_k, coord_m_per_cta + n_rank * _a_rows, tile_l_a),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=_CTA_GROUP,
                                )
                        else:
                            _a_per_cta = a_mcast_slices >> _preferred_cluster_n_shift
                            if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                                _a_per_cta = a_mcast_slices >> _fallback_cluster_n_shift
                            for _asl in cutlass.range(_a_per_cta):
                                _a_idx = n_rank * _a_per_cta + _asl
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(_a_idx * _a_rows * cta_tile_mnk[2]),
                                        tma_a_desc.get_ptr(),
                                        (coord_k, coord_m_per_cta + _a_idx * _a_rows, tile_l_a),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=_CTA_GROUP,
                                    )
                    elif cutlass.const_expr(multicast_a):
                        if n_rank == 0:
                            if cutlass.const_expr(a_is_m_major):
                                for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(m_group * a_tma_group_elems * cta_tile_mnk[2]),
                                            tma_a_desc.get_ptr(),
                                            (
                                                coord_m_per_cta + m_group * a_tma_group_elems,
                                                coord_k,
                                                tile_l_a,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=_CTA_GROUP,
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage,
                                        tma_a_desc.get_ptr(),
                                        (coord_k, coord_m_per_cta, tile_l_a),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=_CTA_GROUP,
                                    )
                    else:
                        if cutlass.const_expr(a_is_m_major):
                            for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(m_group * a_tma_group_elems * cta_tile_mnk[2]),
                                        tma_a_desc.get_ptr(),
                                        (
                                            coord_m_per_cta + m_group * a_tma_group_elems,
                                            coord_k,
                                            tile_l_a,
                                        ),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=_CTA_GROUP,
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_stage,
                                    tma_a_desc.get_ptr(),
                                    (coord_k, coord_m_per_cta, tile_l_a),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=_CTA_GROUP,
                                )

                for _bj in cutlass.range_constexpr(num_b_operands):
                    sB_stage = smem_b_list[_bj].subview(sB_elems * stage)
                    tma_b_desc = tma_b_descs[_bj]
                    if cutlass.const_expr(b_mcast_slices > 1):
                        _b_rows = cta_tile_mnk[1] // b_mcast_slices
                        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_stage.subview(pair_m_idx * _b_rows * cta_tile_mnk[2]),
                                    tma_b_desc.get_ptr(),
                                    (coord_k, coord_n_per_cta + pair_m_idx * _b_rows, tile_l_b),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=_CTA_GROUP,
                                )
                        else:
                            _b_per_cta = b_mcast_slices >> (_preferred_cluster_m_shift - _cta_group_shift)
                            if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                                _b_per_cta = b_mcast_slices >> (_fallback_cluster_m_shift - _cta_group_shift)
                            for _bsl in cutlass.range(_b_per_cta):
                                _b_idx = pair_m_idx * _b_per_cta + _bsl
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(_b_idx * _b_rows * cta_tile_mnk[2]),
                                        tma_b_desc.get_ptr(),
                                        (coord_k, coord_n_per_cta + _b_idx * _b_rows, tile_l_b),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=_CTA_GROUP,
                                    )
                    elif cutlass.const_expr(multicast_b):
                        if pair_m_idx == 0:
                            if cutlass.const_expr(b_is_n_major):
                                for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage.subview(n_group * b_tma_group_elems * cta_tile_mnk[2]),
                                            tma_b_desc.get_ptr(),
                                            (
                                                coord_n_per_cta + n_group * b_tma_group_elems,
                                                coord_k,
                                                tile_l_b,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=_CTA_GROUP,
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage,
                                        tma_b_desc.get_ptr(),
                                        (coord_k, coord_n_per_cta, tile_l_b),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=_CTA_GROUP,
                                    )
                    else:
                        if cutlass.const_expr(b_is_n_major):
                            for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(n_group * b_tma_group_elems * cta_tile_mnk[2]),
                                        tma_b_desc.get_ptr(),
                                        (
                                            coord_n_per_cta + n_group * b_tma_group_elems,
                                            coord_k,
                                            tile_l_b,
                                        ),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=_CTA_GROUP,
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_stage,
                                    tma_b_desc.get_ptr(),
                                    (coord_k, coord_n_per_cta, tile_l_b),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=_CTA_GROUP,
                                )

                ab_iter += 1

            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_tma = clc_full_phase_tma ^ 1
            while not nvvm.mbarrier_try_wait_parity(
                clc_full_mbar_ptr.subview(consumer_stage),
                clc_full_phase_tma,
                time_limit=10_000_000,
            ):
                pass
            m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid = vld
            tma_raw_m = m_idx >> _preferred_cluster_m_shift
            tma_raw_n = n_idx >> _preferred_cluster_n_shift
            tma_nt_m = gridx >> _preferred_cluster_m_shift
            tma_nt_n = gridy >> _preferred_cluster_n_shift
            if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                    tma_raw_m = m_idx >> _fallback_cluster_m_shift
                    tma_raw_n = n_idx >> _fallback_cluster_n_shift
                    tma_nt_m = gridx >> _fallback_cluster_m_shift
                    tma_nt_n = gridy >> _fallback_cluster_n_shift
            tile_m, tile_n = _l2_swizzle_tile(
                tma_raw_m,
                tma_raw_n,
                tma_nt_m,
                tma_nt_n,
                swizzle_w,
                identity=tile_swizzle_n == 1,
            )
            tile_l = l_idx
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            tile_iter += 1

        tail_stage = ab_iter % ab_stages
        tail_phase = ab_empty_phase_bit
        if tail_stage == 0 and ab_iter != 0:
            tail_phase = tail_phase ^ 1
        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            for _ in range(ab_stages):
                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
                    pass
                tail_stage = tail_stage + 1
                if tail_stage == ab_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1

    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        b_arrive_pattern = (1 << cluster_m) - 1
    else:
        b_arrive_pattern = (cutlass.Int32(1) << cluster_m) - 1
    a_part = a_mcast_pattern << m_rank
    if cutlass.const_expr(cta_group == 2):
        a_part = a_part | (a_part << 1)
    b_part = b_arrive_pattern << (n_rank * cluster_m)
    if cutlass.const_expr(ab_empty_full_mask):
        ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
    else:
        ab_empty_arrive_mask = cutlass.Int16(a_part | b_part)
    if cutlass.const_expr(cta_group == 2):
        acc_full_mcast = cutlass.Int16(3) << pair_leader_rank
    else:
        acc_full_mcast = None
    if warp_idx == mma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        _tcgen05_alloc(
            tmem_ptr_i32,
            cutlass.Int32(num_tmem_alloc_cols),
            is_exclusive=tmem_alloc_exclusive,
            group=_CTA_GROUP,
        )
        nvvm.bar_warp_sync(0xFFFFFFFF)
        nvvm.barrier_cta_arrive(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        if cutlass.const_expr(cta_group == 2):
            peer_cta_rank = cta_rank_in_cluster ^ 1

        if is_pair_leader:
            ab_full_phase_bit = cutlass.Int32(0)
            ab_iter = cutlass.Int32(0)
            acc_empty_phase_bit = cutlass.Int32(1)
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            clc_full_phase_mma = cutlass.Int32(0)
            acc_stage = cutlass.Int32(0)
            if cutlass.const_expr(split_k_slices > 1):
                tile_l = bidz  # this tile's z coord
            # Descriptor metadata and the SMEM allocation base are invariant
            # across persistent tiles.  Only the encoded start address advances.
            desc_a_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_a_list[i],
                    leading_byte_offset=a_smem_desc_leading_byte_offset,
                    stride_byte_offset=a_smem_desc_stride_byte_offset,
                    layout=ab_smem_swizzle,
                )
                for i in range(num_a_operands)
            ]
            desc_b_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_b_list[j],
                    leading_byte_offset=b_smem_desc_leading_byte_offset,
                    stride_byte_offset=b_smem_desc_stride_byte_offset,
                    layout=ab_smem_swizzle,
                )
                for j in range(num_b_operands)
            ]
            while is_valid != 0:
                acc_stage = tile_iter % acc_stages
                if acc_stage == 0 and tile_iter != 0:
                    acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(acc_stage),
                    acc_empty_phase_bit,
                    time_limit=10_000_000,
                ):
                    pass

                acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                # One accumulator per (gemm, M block). Column arithmetic stays on
                # the encoded (row << 16) | col integer.
                tmem_addr_mmas = [
                    [
                        cutlass.inttoptr(
                            (base_row_id << 16) | (acc_base_col + g * cols_per_acc_stage + mi * epi_cols_per_mma_m),
                            6,
                            cutlass.Int32,
                        )
                        for mi in range(mma_size_m)
                    ]
                    for g in range(num_gemms)
                ]

                if cutlass.const_expr(split_k_slices > 1):
                    split_idx = cutlass.Int64(tile_l % split_k_slices)
                    k_tiles_per_split = num_k_tiles // split_k_slices
                    k_tiles_remainder = num_k_tiles % split_k_slices
                    k_begin = split_idx * k_tiles_per_split + cutlass.min(split_idx, k_tiles_remainder)
                    k_end = (split_idx + 1) * k_tiles_per_split + cutlass.min(split_idx + 1, k_tiles_remainder)
                else:
                    k_begin = cutlass.Int64(0)
                    k_end = num_k_tiles
                scale_d = cutlass.Boolean(False)
                for k_tile_idx in range(k_begin, k_end):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_full_phase_bit = ab_full_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        ab_full_mbar_ptr.subview(stage),
                        ab_full_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass

                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        for g in cutlass.range_constexpr(num_gemms):
                            desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * stage + a_smem_k_step_bytes * k_block_idx)
                            desc_b = desc_b_roots[gemm_b_idx[g]].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                            for mi in cutlass.range_constexpr(mma_size_m):
                                # The M sub-block offset is a whole SMEM swizzle atom
                                # (mma_inst_m x cta_tile_k_bytes), so the descriptor's
                                # swizzle phase is preserved. B is shared by every M block.
                                desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                                if elect_one:
                                    _tcgen05_mma(
                                        mma_kind,
                                        _CTA_GROUP,
                                        tmem_addr_mmas[g][mi],
                                        desc_a,
                                        desc_b,
                                        idesc,
                                        scale_d,
                                        collector_op=_a_collector_op(g),
                                        b_collector_op=_b_collector_op(mi),
                                    )
                        # Every accumulator sees scale_d=False on exactly the first
                        # k_block of the tile, so the flip stays outside mi.
                        scale_d = cutlass.Boolean(True)

                    if elect_one:
                        nvvm.tcgen05_commit(
                            ab_empty_mbar_ptr.subview(stage),
                            multicast_mask=ab_empty_arrive_mask,
                            group=_CTA_GROUP,
                        )
                    ab_iter += 1

                if elect_one:
                    nvvm.tcgen05_commit(
                        acc_full_mbar_ptr.subview(acc_stage),
                        multicast_mask=acc_full_mcast,
                        group=_CTA_GROUP,
                    )

                consumer_stage = tile_iter % CLC_SCHED_STAGES
                if consumer_stage == 0 and tile_iter != 0:
                    clc_full_phase_mma = clc_full_phase_mma ^ 1
                while not nvvm.mbarrier_try_wait_parity(
                    clc_full_mbar_ptr.subview(consumer_stage),
                    clc_full_phase_mma,
                    time_limit=10_000_000,
                ):
                    pass
                _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                cute.arch.fence_proxy("async.shared", space="cta")
                is_valid = vld
                if cutlass.const_expr(split_k_slices > 1):
                    tile_l = _l_idx
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                    nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                tile_iter += 1

            # Dense only: an early-launched reducer would pile its many small CTAs
            # onto the few idle SMs, so split-K leaves the trigger to grid exit.
            if cutlass.const_expr(USE_PDL and split_k_slices == 1):
                nvvm.griddepcontrol("launch_dependents")

            tail_stage = acc_stage
            tail_phase = acc_empty_phase_bit
            for _ in range(acc_stages):
                tail_stage = tail_stage + 1
                if tail_stage == acc_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1
                while not nvvm.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(tail_stage),
                    tail_phase,
                    time_limit=10_000_000,
                ):
                    pass
            nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
            if cutlass.const_expr(cta_group == 2):
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            _tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=_CTA_GROUP,
            )
        else:
            if cutlass.const_expr(cta_group == 2):
                tile_iter = cutlass.Int32(0)
                is_valid = cutlass.Int32(1)
                clc_full_phase_mma = cutlass.Int32(0)
                while is_valid != 0:
                    consumer_stage = tile_iter % CLC_SCHED_STAGES
                    if consumer_stage == 0 and tile_iter != 0:
                        clc_full_phase_mma = clc_full_phase_mma ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        clc_full_mbar_ptr.subview(consumer_stage),
                        clc_full_phase_mma,
                        time_limit=10_000_000,
                    ):
                        pass
                    _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    is_valid = vld
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                        nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                    tile_iter += 1
                # Dense only, same reason as above.
                if cutlass.const_expr(USE_PDL and split_k_slices == 1):
                    nvvm.griddepcontrol("launch_dependents")
                nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
                _tcgen05_dealloc(
                    alloc_ptr,
                    cutlass.Int32(num_tmem_alloc_cols),
                    is_exclusive=tmem_alloc_exclusive,
                    group=_CTA_GROUP,
                )

    if warp_idx < num_epilogue_warps:
        nvvm.setmaxregister(epi_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        nvvm.barrier_cta_sync(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16

        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")

        tile_iter = cutlass.Int32(0)
        acc_full_phase_bit = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        is_valid = cutlass.Int32(1)
        clc_full_phase_epi = cutlass.Int32(0)

        # @@EPILOGUE_SETUP:BEGIN@@
        if cutlass.const_expr(epi_packed_lanes):
            row_id_with_warp_offset = base_row_id
        else:
            row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        if cutlass.const_expr(epi_packed_lanes):
            shape = nvvm.Tcgen05LdStShape.SHAPE_16X32BX2
            ld_half_off = 0
        else:
            shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
            ld_half_off = None
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@

        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)

        # ==== FP8 NORM+ROPE+QUANT FUSION: the [4] scales are NOT read here.  Holding
        # them live across the persistent loop spilled 20 of the ldg_early cos/sin
        # words (10 STL.64 / 20 LDL per Q/K tile at epi_reg_count=232; the bf16 fork
        # is at 0/0).  They are read once per tile in the pre-wait block below (4
        # registers live across the acc_full wait only), after the PDL wait above.

        while is_valid != 0:
            coord_m_tile = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            # @@EPILOGUE_DRAIN:BEGIN@@
            coord_n_c = tile_n * cgrp_tile_n_cur + n_rank * pair_n_size
            if cutlass.const_expr(epi_dp22):
                coord_n_c = coord_n_c + (warp_idx // 2) * epi_cols_per_mma_m

            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            # ==== FP8 NORM+ROPE+QUANT FUSION: tile classification, the per-tile
            # scalars, and (ldg_early) the cos/sin loads issued BEFORE the wait on
            # the accumulator, so they land while the MMA drains this tile instead
            # of on the epilogue's critical path.  Everything here depends only on
            # the tile coordinates (mma_size_m == 1, checked at import: the row is
            # coord_m_tile + tidx) and is warp-uniform.  Non-norm tiles point every
            # lane at the tables' row 0 -- one L1-resident line -- rather than
            # skipping the loads, so the values exist unconditionally.
            _n0 = coord_n_c
            _is_q = _n0 < cutlass.Int32(_OFF_GATE)
            _is_k = (_n0 >= cutlass.Int32(_OFF_K)) & (_n0 < cutlass.Int32(_OFF_V))
            _is_v = _n0 >= cutlass.Int32(_OFF_V)
            _is_norm = _is_q | _is_k
            if cutlass.const_expr(_QK_NORM):
                _w_base = cutlass.Int64(arith.select(_is_q.ir_value(), w_q_norm.iterator.toint().ir_value(), w_k_norm.iterator.toint().ir_value()))
            # The [4] scales, read HERE -- before the wait, ahead of the cos/sin loads -- as
            # exactly TWO per-tile registers (a tile is one class), so nothing below the
            # wait waits on global memory:
            #   _qs_a       alpha = qscal[0]                   (GATE: bf16(alpha*acc); Q/K: srow; V: alpha*scale_v)
            #   _scale_cls  qscal[1 | 2 | 3] for a Q | K | V tile (GATE: reads scale_k, unused)
            # The class is selected on the ADDRESS, not on loaded values: two scalar
            # ld.global.f32 to the same 16-B line.  With one ld.global.v4 + register
            # selects, LLVM/ptxas sank the FSELs past the wait and kept all FOUR raw words
            # live across it -> 2 STL.64 / 4 LDL per Q/K tile (4 ldg_early cos words), for
            # every spelling of the selects (4, 3 or 2 derived values).  Everything else
            # the arms need (alpha*scale_v, alpha^2/D) is register-only arithmetic on
            # these two at the point of use.  The other two placements, both measured:
            # read AFTER the wait (round 1) = 0 spills but a dependent L2 round trip per
            # tile, +21..26 % of the GEMM under the 8 KB L1 of the oversized carveout
            # (STATUS.md "GEMM A/B"); held across the persistent loop = 20 cos/sin words
            # spilled.
            # Both are the same value on every lane but they DO cost two per-lane registers
            # across the wait: ptxas already treats them as uniform (an explicit
            # shfl.sync.idx broadcast and cute.arch.make_warp_uniform were both deleted by
            # ptxas with a byte-identical cubin) and still allocates them in R, not UR.  The
            # register they cost is paid for by the pass-A load granule (_PASS_A_SPLIT).
            _qs_base = qscal.iterator.toint()
            _qs_a = ld_global(_qs_base, cutlass.Float32)
            _cls_off = cutlass.Int64(
                arith.select(
                    _is_v.ir_value(),
                    cutlass.Int64(3 * 4).ir_value(),
                    arith.select(_is_q.ir_value(), cutlass.Int64(1 * 4).ir_value(), cutlass.Int64(2 * 4).ir_value()),
                )
            )
            _scale_cls = ld_global(_qs_base + _cls_off, cutlass.Float32)
            # Tail rows (row >= m) exist in TMEM and are clipped by the TMA
            # store; their cos/sin READS must still be in bounds -> clamp.
            _row64 = (coord_m_tile + tidx).to(cutlass.Int64)
            _row_c = cutlass.min(_row64, m - 1)
            _cs_words_early = []
            _sn_words_early = []
            if cutlass.const_expr(_CS_EARLY):
                _tab_row = cutlass.Int64(arith.select(_is_norm.ir_value(), _row_c.ir_value(), cutlass.Int64(0).ir_value()))
                _cs_addr_e = cos_tab.iterator.toint() + _tab_row * cutlass.Int64(_ROPE_DIM * _ACT_BPE)
                _sn_addr_e = sin_tab.iterator.toint() + _tab_row * cutlass.Int64(_ROPE_DIM * _ACT_BPE)
                for _h in cutlass.range_constexpr(_ROPE_DIM * _ACT_BPE // 16):
                    for _w in ld_global_v4(_cs_addr_e + cutlass.Int64(_h * 16), cutlass.Int32):
                        _cs_words_early.append(_w)
                for _h in cutlass.range_constexpr(_ROPE_DIM * _ACT_BPE // 16):
                    for _w in ld_global_v4(_sn_addr_e + cutlass.Int64(_h * 16), cutlass.Int32):
                        _sn_words_early.append(_w)

            while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(acc_stage), acc_full_phase_bit, time_limit=10_000_000):
                pass

            acc_base_col = base_col_id_root + acc_stage * acc_region_cols

            for mi in cutlass.range_constexpr(mma_size_m):
                coord_m = coord_m_tile + mi * epi_rows_per_mma_m
                mi_col_base = acc_base_col + mi * epi_cols_per_mma_m
                tmem_col_addr_gemms = [(row_id_with_warp_offset << 16) | (mi_col_base + g * cols_per_acc_stage) for g in range(num_gemms)]

                if cutlass.const_expr(epi_packed_lanes):
                    row = coord_m + warp_idx * 16 + lane
                    row_active = lane < 16
                elif cutlass.const_expr(epi_dp22):
                    row = coord_m + (warp_idx % 2) * 32 + lane
                    row_active = True
                else:
                    row = coord_m + tidx
                    row_active = True

                # ==== FP8 NORM+ROPE+QUANT FUSION: BEGIN ==== (classification, scales and the early loads are above the acc_full wait)
                if _is_norm:
                    # ---------------- Q / K: norm + RoPE + e4m3 into q8 (desc 0) / k8 (desc 1) ----------------
                    # -- cos/sin for this row, issued FIRST so they land under pass A --
                    _cs_words = _cs_words_early
                    _sn_words = _sn_words_early
                    if cutlass.const_expr(NORM_SOURCE == "ldg"):
                        _cs_words = []
                        _sn_words = []
                        _cs_addr = cos_tab.iterator.toint() + _row_c * cutlass.Int64(_ROPE_DIM * _ACT_BPE)
                        _sn_addr = sin_tab.iterator.toint() + _row_c * cutlass.Int64(_ROPE_DIM * _ACT_BPE)
                        for _h in cutlass.range_constexpr(_ROPE_DIM * _ACT_BPE // 16):
                            for _w in ld_global_v4(_cs_addr + cutlass.Int64(_h * 16), cutlass.Int32):
                                _cs_words.append(_w)
                        for _h in cutlass.range_constexpr(_ROPE_DIM * _ACT_BPE // 16):
                            for _w in ld_global_v4(_sn_addr + cutlass.Int64(_h * 16), cutlass.Int32):
                                _sn_words.append(_w)

                    # -- pass A: sum of squares of the RAW accumulator over the whole row (8 subtiles),
                    # two subtiles per wait as in the bf16 fork, but each subtile as _PASS_A_SPLIT
                    # (= 4) 8-column tcgen05.ld's.  ptxas batches a fixed NUMBER of LDTMs ahead of
                    # the math, so the granule decides how many accumulator registers are in flight:
                    # 32-wide -> a 4th LDTM hoisted, 128 live acc regs, 8 STL.64 / 16 LDL of the
                    # ldg_early cos/sin words; 16-wide (round 1) -> 8 LDTM.x16 batched = still 128
                    # live, 0 spills ONLY while nothing else was live across the wait -- the two
                    # pre-wait scale registers of round 2 tipped it to 1 STL / 2 LDL; 8-wide -> 0.
                    # NOTE the real ceiling is 255, not epi_reg_count: this rendering's PTX has no
                    # .maxnreg, so ptxas drops every setmaxnreg (no SETMAXREG in SASS, the epilogue
                    # uses R252).  Verified on the sm_107a cubin (nvdisasm), not assumed.  Same
                    # values, same reduction, different chunking.
                    # (qk_norm=False: no pass A, no rsqrt -- the arm is descale + RoPE + quantize)
                    if cutlass.const_expr(_QK_NORM):
                        _sq_parts = []
                        for _s0 in cutlass.range_constexpr(0, subtile_cnt, 2):
                            _vs = []
                            for _s in cutlass.range_constexpr(_s0, min(_s0 + 2, subtile_cnt)):
                                _soff, _sw = epi_spans[_s]
                                for _hh in cutlass.range_constexpr(_PASS_A_SPLIT):
                                    _tm = cutlass.inttoptr(tmem_col_addr_gemms[0] + _soff + _hh * (_sw // _PASS_A_SPLIT), 6, mma_c_dtype)
                                    _vs.append(nvvm.tcgen05_ld(shape, _tm, num=_sw // _PASS_A_SPLIT, offset=ld_half_off))
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                            for _v in _vs:
                                _acc = _v[0] * _v[0]
                                for _i in cutlass.range_constexpr(1, epi_n // _PASS_A_SPLIT):
                                    _acc = _acc + _v[_i] * _v[_i]
                                _sq_parts.append(_acc)
                        # No global load here: the scales were read above the wait.
                        _sq = _sq_parts[0]
                        for _p in cutlass.range_constexpr(1, len(_sq_parts)):
                            _sq = _sq + _sq_parts[_p]
                        # rstd of the DEQUANTIZED row: alpha enters through the eps term,
                        # rstd(alpha*x) = rsqrt(alpha^2 * mean(x^2) + eps).  alpha^2 / D is recomputed
                        # from the pre-wait _qs_a here (registers only) rather than held across the wait.
                        _rstd = cute.math.rsqrt(_sq * ((_qs_a * _qs_a) * cutlass.Float32(1.0 / _D)) + cutlass.Float32(_EPS), fastmath=True)
                    # ONE per-row multiplier: alpha (descale) * rstd * scale_q|k -- pass B costs what the bf16 fork's acc*rstd*w does.
                    # RoPE-only: alpha * scale_q|k, the V arm's `_s_v` form (two pre-wait registers, one FMUL).
                    _srow = (_qs_a * _rstd) * _scale_cls if cutlass.const_expr(_QK_NORM) else _qs_a * _scale_cls
                    # The `const*` diagnostics substitute a LIVE register for every deleted load: rstd under
                    # the norm, the per-row multiplier under RoPE-only.
                    _diag = _rstd if cutlass.const_expr(_QK_NORM) else _srow
                    # Column remap: q8 col = c for a Q tile, k8 col = c - OFF_K for a K tile.
                    _col_shift = cutlass.Int32(arith.select(_is_q.ir_value(), cutlass.Int32(0).ir_value(), cutlass.Int32(_OFF_K).ir_value()))

                    # -- pass B: scale, rotate, quantize and store subtile by subtile --
                    _rot_elems = []  # per rope subtile: list of epi_n fp32
                    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                        subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                        if cutlass.const_expr(subtile_idx == 0):
                            # Load the whole rope group at once: partner of subtile s is s + ROPE_HALF_SUBTILES.
                            _rv = []
                            for _s in cutlass.range_constexpr(_ROPE_SUBTILES):
                                _soff, _sw = epi_spans[_s]
                                _tm = cutlass.inttoptr(tmem_col_addr_gemms[0] + _soff, 6, mma_c_dtype)
                                _rv.append(nvvm.tcgen05_ld(shape, _tm, num=_sw, offset=ld_half_off))
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                            _y = []  # ROPE_DIM normed (or descaled, qk_norm=False) + scaled fp32, flat
                            for _s in cutlass.range_constexpr(_ROPE_SUBTILES):
                                if cutlass.const_expr(_QK_NORM):
                                    _we = []
                                    if cutlass.const_expr(_LDG_W):
                                        _wa = _w_base + cutlass.Int64(epi_spans[_s][0] * _ACT_BPE)
                                        for _h in cutlass.range_constexpr(epi_n * _ACT_BPE // 16):
                                            for _w in ld_global_v4(_wa + cutlass.Int64(_h * 16), cutlass.Int32):
                                                _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                                                _we.append(_lo)
                                                _we.append(_hi)
                                    else:
                                        for _i in cutlass.range_constexpr(epi_n):
                                            _we.append(_rstd)  # DIAGNOSTIC: same ALU, no load
                                    for _i in cutlass.range_constexpr(epi_n):
                                        _y.append((_rv[_s][_i] * _srow) * _we[_i])
                                else:
                                    for _i in cutlass.range_constexpr(epi_n):
                                        _y.append(_rv[_s][_i] * _srow)
                            _c = []
                            _sn = []
                            if cutlass.const_expr(_LDG_CS):
                                for _i in cutlass.range_constexpr(_ROPE_DIM // 2):
                                    _lo, _hi = f16x2_to_f32(_cs_words[_i], dtype=cutlass.BFloat16)
                                    _c.append(_lo)
                                    _c.append(_hi)
                                    _lo, _hi = f16x2_to_f32(_sn_words[_i], dtype=cutlass.BFloat16)
                                    _sn.append(_lo)
                                    _sn.append(_hi)
                            else:
                                for _i in cutlass.range_constexpr(_ROPE_DIM):
                                    _c.append(_diag + cutlass.Float32(1.0))  # DIAGNOSTIC: no load
                                    _sn.append(_diag)
                            _flat = []
                            for _i in cutlass.range_constexpr(_ROPE_HALF):
                                _flat.append(_y[_i] * _c[_i] - _y[_i + _ROPE_HALF] * _sn[_i])
                            for _i in cutlass.range_constexpr(_ROPE_HALF):
                                _flat.append(_y[_i + _ROPE_HALF] * _c[_i + _ROPE_HALF] + _y[_i] * _sn[_i + _ROPE_HALF])
                            for _s in cutlass.range_constexpr(_ROPE_SUBTILES):
                                _rot_elems.append(_flat[_s * epi_n : (_s + 1) * epi_n])
                        if cutlass.const_expr(subtile_idx < _ROPE_SUBTILES):
                            _elems = _rot_elems[subtile_idx]
                        else:
                            _tm = cutlass.inttoptr(tmem_col_addr_gemms[0] + subtile_col_offset, 6, mma_c_dtype)
                            _v = nvvm.tcgen05_ld(shape, _tm, num=subtile_w, offset=ld_half_off)
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                            _elems = []
                            if cutlass.const_expr(_QK_NORM):
                                _we = []
                                if cutlass.const_expr(_LDG_W):
                                    _wa = _w_base + cutlass.Int64(subtile_col_offset * _ACT_BPE)
                                    for _h in cutlass.range_constexpr(epi_n * _ACT_BPE // 16):
                                        for _w in ld_global_v4(_wa + cutlass.Int64(_h * 16), cutlass.Int32):
                                            _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                                            _we.append(_lo)
                                            _we.append(_hi)
                                else:
                                    for _i in cutlass.range_constexpr(epi_n):
                                        _we.append(_rstd)  # DIAGNOSTIC: same ALU, no load
                                for _i in cutlass.range_constexpr(epi_n):
                                    _elems.append((_v[_i] * _srow) * _we[_i])
                            else:
                                for _i in cutlass.range_constexpr(epi_n):
                                    _elems.append(_v[_i] * _srow)  # RoPE-only passthrough dims: alpha * scale * acc

                        if mi == mma_size_m - 1 and subtile_idx == subtile_cnt - 1:
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                            if elect_one:
                                if cutlass.const_expr(cta_group == 2):
                                    nvvm.mbarrier_arrive(
                                        nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                        scope=nvvm.MemScope.CLUSTER,
                                        relaxed=True,
                                    )
                                else:
                                    nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                        col = coord_n_c + subtile_col_offset
                        vec_out = _e4m3x32(_elems)
                        epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                        # The SAME ring slot as the bf16 subtile, viewed as e4m3: 128 rows x 32 B (the
                        # shipped compiler's e4m3 TMA-store arm: alignment 32, Swizzle(1,4,3), s32b box [32,128,1]).
                        _tsv_8 = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape=4096, dtype=cutlass.Float8E4M3FN)
                        _tsv_8.data_ptr(tidx * 32).store_swizzled(vec_out, alignment=32, swizzle=cutlass.Swizzle(1, 4, 3))
                        cute.arch.fence_view_async_shared()
                        nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
                        if warp_idx == 0:
                            if elect_one:
                                # Q -> desc 0 (q8), K -> desc 1 (k8): warp-uniform runtime class, so a
                                # plain branch on the descriptor (the tensor-map pointer is a GridConstant).
                                if _is_q:
                                    nvvm.cp_async_bulk_tensor_global_shared_cta(
                                        tma_c_descs[0].get_ptr(),
                                        _tsv_8.data_ptr(),
                                        (col - _col_shift, coord_m, tile_l),
                                    )
                                else:
                                    nvvm.cp_async_bulk_tensor_global_shared_cta(
                                        tma_c_descs[1].get_ptr(),
                                        _tsv_8.data_ptr(),
                                        (col - _col_shift, coord_m, tile_l),
                                    )
                            if elect_one:
                                nvvm.cp_async_bulk_commit_group()
                            nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)
                        nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
                elif _is_v:
                    # ---------------- V: alpha * scale_v, e4m3 into v8 (desc 2) ----------------
                    _s_v = _qs_a * _scale_cls  # both pre-wait registers; one FMUL, no memory
                    _col_shift = cutlass.Int32(_OFF_V)  # v8 col = c - OFF_V
                    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                        subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                        _tm = cutlass.inttoptr(tmem_col_addr_gemms[0] + subtile_col_offset, 6, mma_c_dtype)
                        _v = nvvm.tcgen05_ld(shape, _tm, num=subtile_w, offset=ld_half_off)
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        _elems = []
                        for _i in cutlass.range_constexpr(epi_n):
                            _elems.append(_v[_i] * _s_v)

                        if mi == mma_size_m - 1 and subtile_idx == subtile_cnt - 1:
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                            if elect_one:
                                if cutlass.const_expr(cta_group == 2):
                                    nvvm.mbarrier_arrive(
                                        nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                        scope=nvvm.MemScope.CLUSTER,
                                        relaxed=True,
                                    )
                                else:
                                    nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                        col = coord_n_c + subtile_col_offset
                        vec_out = _e4m3x32(_elems)
                        epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                        # The SAME ring slot as the bf16 subtile, viewed as e4m3: 128 rows x 32 B (the
                        # shipped compiler's e4m3 TMA-store arm: alignment 32, Swizzle(1,4,3), s32b box [32,128,1]).
                        _tsv_8 = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape=4096, dtype=cutlass.Float8E4M3FN)
                        _tsv_8.data_ptr(tidx * 32).store_swizzled(vec_out, alignment=32, swizzle=cutlass.Swizzle(1, 4, 3))
                        cute.arch.fence_view_async_shared()
                        nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
                        if warp_idx == 0:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_global_shared_cta(
                                    tma_c_descs[2].get_ptr(),
                                    _tsv_8.data_ptr(),
                                    (col - _col_shift, coord_m, tile_l),
                                )
                            if elect_one:
                                nvvm.cp_async_bulk_commit_group()
                            nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)
                        nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
                else:
                    # ---------------- GATE: the rendering's own alpha epilogue, bf16 into gate16 (desc 3) ----------------
                    # (verbatim from the rendering: the alpha-aux broadcast is the pre-wait
                    # `_qs_a`, the descriptor is gate16's, the column is shifted by OFF_GATE.)
                    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                        subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                        c_rmem_vecs = []
                        for g in cutlass.range_constexpr(num_gemms):
                            subtile_tmem_addr = tmem_col_addr_gemms[g] + subtile_col_offset
                            tmem = cutlass.inttoptr(subtile_tmem_addr, 6, mma_c_dtype)
                            _cv = nvvm.tcgen05_ld(shape, tmem, num=subtile_w, offset=ld_half_off)
                            # INT8 int32 accumulate → widen to fp32 (skipped for int32 output).
                            if cutlass.const_expr(acc_widen_to_fp32):
                                _accf = _cv.to(cutlass.Float32)
                                # `+ 0.0` forces a fresh fp32 register so int32->fp32 isn't folded into an invalid int32->fp8 cast.
                                _cv = _accf + cutlass.full_like(_accf, 0.0)
                            c_rmem_vecs.append(_cv)
                        c_rmem_vec = c_rmem_vecs[0]

                        if mi == mma_size_m - 1 and subtile_idx == subtile_cnt - 1:
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                            if elect_one:
                                if cutlass.const_expr(cta_group == 2):
                                    nvvm.mbarrier_arrive(
                                        nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                        scope=nvvm.MemScope.CLUSTER,
                                        relaxed=True,
                                    )
                                else:
                                    nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                        col = coord_n_c + subtile_col_offset

                        vec_f32 = c_rmem_vec
                        col_j = col
                        linear_idx = tile_l * out_stride_l_0 + row * out_stride_m_0 + col_j * out_stride_n_0

                        _c_0_a = (vec_f32).to(cutlass.Float32)
                        _op_0 = _c_0_a * (cutlass.full_like(_c_0_a, _qs_a))
                        _r_0 = (_op_0).to(cutlass.BFloat16)
                        vec_out = (_r_0).to(cutlass.BFloat16)

                        epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                        _tsv_0 = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape=4096, dtype=cutlass.BFloat16)
                        _tsv_0.data_ptr(tidx * 32).store_swizzled(vec_out, alignment=64, swizzle=cutlass.Swizzle(2, 4, 3))
                        cute.arch.fence_view_async_shared()
                        nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
                        if warp_idx == 0:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_global_shared_cta(
                                    tma_c_descs[3].get_ptr(),
                                    _tsv_0.data_ptr(),
                                    (col - cutlass.Int32(_OFF_GATE), coord_m, tile_l),
                                )
                            if elect_one:
                                nvvm.cp_async_bulk_commit_group()
                            nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)
                        nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)

                # ==== FP8 NORM+ROPE+QUANT FUSION: END ====
            # @@EPILOGUE_DRAIN:END@@
            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_epi = clc_full_phase_epi ^ 1
            while not nvvm.mbarrier_try_wait_parity(
                clc_full_mbar_ptr.subview(consumer_stage),
                clc_full_phase_epi,
                time_limit=10_000_000,
            ):
                pass
            m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid = vld
            epi_raw_m = m_idx >> _preferred_cluster_m_shift
            epi_raw_n = n_idx >> _preferred_cluster_n_shift
            epi_nt_m = gridx >> _preferred_cluster_m_shift
            epi_nt_n = gridy >> _preferred_cluster_n_shift
            if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                    epi_raw_m = m_idx >> _fallback_cluster_m_shift
                    epi_raw_n = n_idx >> _fallback_cluster_n_shift
                    epi_nt_m = gridx >> _fallback_cluster_m_shift
                    epi_nt_n = gridy >> _fallback_cluster_n_shift
            tile_m, tile_n = _l2_swizzle_tile(
                epi_raw_m,
                epi_raw_n,
                epi_nt_m,
                epi_nt_n,
                swizzle_w,
                identity=tile_swizzle_n == 1,
            )
            tile_l = l_idx
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)

            tile_iter += 1

        if warp_idx == 0:
            nvvm.cp_async_bulk_wait_group(0, read=True)

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)


frost_sm100_matmul_128x256x128_128x256x32_cluster2x1_2ctamma.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    a_0: cute.Tensor,
    b_0: cute.Tensor,
    c_q8: cute.Tensor,  # e4m3 [M, N_Q]  (permuted to (N, M, L) by the runner)
    # ---- FP8 NORM+ROPE+QUANT FUSION: appended ----
    c_k8: cute.Tensor,  # e4m3 [M, N_KV]
    c_v8: cute.Tensor,  # e4m3 [M, N_KV]
    c_gate: cute.Tensor,  # bf16 [M, N_GATE]
    w_q_norm: Optional[cute.Tensor],  # None iff PARAMS.qk_norm is False
    w_k_norm: Optional[cute.Tensor],
    cos_tab: cute.Tensor,
    sin_tab: cute.Tensor,
    qscal: cute.Tensor,
    stream: _cuda.CUstream,
) -> None:
    _a_operands = [a_0]
    _b_operands = [b_0]
    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    batch = problem_size[3]
    _stride_idx = 4
    _a_stride_sets = []
    for _ in cutlass.range_constexpr(num_a_operands):
        _a_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
            )
        )
        _stride_idx += 3
    _b_stride_sets = []
    for _ in cutlass.range_constexpr(num_b_operands):
        _b_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
            )
        )
        _stride_idx += 3
    out_stride_m_0 = problem_size[_stride_idx]
    out_stride_n_0 = problem_size[_stride_idx + 1]
    out_stride_l_0 = problem_size[_stride_idx + 2]
    _stride_idx += 3
    # ---- FP8 NORM+ROPE+QUANT FUSION: k8, v8 and gate16 stride triples follow q8's ----
    out_stride_m_1 = problem_size[_stride_idx]
    out_stride_n_1 = problem_size[_stride_idx + 1]
    out_stride_l_1 = problem_size[_stride_idx + 2]
    _stride_idx += 3
    out_stride_m_2 = problem_size[_stride_idx]
    out_stride_n_2 = problem_size[_stride_idx + 1]
    out_stride_l_2 = problem_size[_stride_idx + 2]
    _stride_idx += 3
    out_stride_m_3 = problem_size[_stride_idx]
    out_stride_n_3 = problem_size[_stride_idx + 1]
    out_stride_l_3 = problem_size[_stride_idx + 2]
    _stride_idx += 3

    if cutlass.const_expr(matmul_a_batch == 1):
        a_batch = 1
    else:
        a_batch = batch
    if cutlass.const_expr(matmul_b_batch == 1):
        b_batch = 1
    else:
        b_batch = batch

    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_l = _a_stride_sets[_a_idx]
        if cutlass.const_expr(a_is_m_major):
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[m, k_sym, a_batch],
                    global_strides=[
                        a_stride_k * ab_dtype.width // 128,
                        a_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[a_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_sym, m, a_batch],
                    global_strides=[
                        a_stride_m * ab_dtype.width // 128,
                        a_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[0] // a_mcast_slices, 1],
                    swizzle=ab_tma_swizzle,
                )
            )
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[n, k_sym, b_batch],
                    global_strides=[
                        b_stride_k * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_sym, n, b_batch],
                    global_strides=[
                        b_stride_n * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
                    swizzle=ab_tma_swizzle,
                )
            )

    # ---- FP8 NORM+ROPE+QUANT FUSION: FOUR TMA-store descriptors, one per compact output ----
    # descs 0..2: e4m3 q8 [N_Q, M] / k8 [N_KV, M] / v8 [N_KV, M].  Each global_dims[0] is
    # ITS OWN width -- not n, not a shared slab width -- so a box past that tensor's last
    # column is clipped rather than written across the row.  The shipped compiler's own
    # e4m3 C form (`_host_tma_c_descs`): dtype Float8E4M3FN (TMA: uint8), strides in 16-B
    # units at 1 B/elem, box [32, 128, 1] = 32 B rows under s32b.
    tma_c_desc_list = []
    for _c8, _w8, _sm8, _sl8 in (
        (c_q8, _N_Q, out_stride_m_0, out_stride_l_0),
        (c_k8, _N_KV, out_stride_m_1, out_stride_l_1),
        (c_v8, _N_KV, out_stride_m_2, out_stride_l_2),
    ):
        tma_c_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_c8.iterator.toint(),
                dtype=cutlass.Float8E4M3FN,
                global_dims=[_w8, m, batch],
                global_strides=[
                    _sm8 * 8 // 128,
                    _sl8 * 8 // 128,
                ],
                box_dims=[32, epi_tile_mn[0], 1],
                swizzle=_tma.TensorMapSwizzle.s32b,
            )
        )
    # desc 3: bf16 gate16 -- the rendering's own C descriptor form over [N_GATE, M].
    tma_c_desc_list.append(
        _tma.create_tensor_map_tiled(
            global_address=c_gate.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=[_N_GATE, m, batch],
            global_strides=[
                out_stride_m_3 * 16 // 128,
                out_stride_l_3 * 16 // 128,
            ],
            box_dims=[32, epi_tile_mn[0], 1],
            swizzle=_tma.TensorMapSwizzle.s64b,
        )
    )

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cgrp_tile_m = cgrp_tile_mnk[0]
    cgrp_tile_n = cgrp_tile_mnk[1]
    num_tile_m_host = (m + cgrp_tile_m - 1) // cgrp_tile_m
    num_tile_n_host = (n + cgrp_tile_n - 1) // cgrp_tile_n
    grid_x = num_tile_m_host * cluster_m
    grid_y = num_tile_n_host * cluster_n
    grid_shape = (grid_x, grid_y, batch * split_k_slices)
    launch = frost_sm100_matmul_128x256x128_128x256x32_cluster2x1_2ctamma(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        tma_a_desc_list[0],
        tma_b_desc_list[0],
        out_stride_m_0,
        out_stride_n_0,
        out_stride_l_0,
        tma_c_desc_list[0],
        tma_c_desc_list[1],
        tma_c_desc_list[2],
        tma_c_desc_list[3],
        w_q_norm,
        w_k_norm,
        cos_tab,
        sin_tab,
        qscal,
    )
    # Mixed CGA: `cluster` is the preferred (wide) shape and `fallback_cluster`
    # the regular one the device groups blocks into when a preferred cluster does
    # not fit. The grid is already a multiple of the preferred shape, which the
    # driver requires.
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        launch.launch(
            grid=grid_shape,
            block=(threads_per_cta, 1, 1),
            cluster=cluster_shape_mnk,
            use_pdl=USE_PDL,
            stream=stream,
        )
    else:
        launch.launch(
            grid=grid_shape,
            block=(threads_per_cta, 1, 1),
            cluster=cluster_shape_mnk,
            fallback_cluster=fallback_cluster_shape_mnk,
            use_pdl=USE_PDL,
            stream=stream,
        )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    ab_stride_elems = 16 // (ab_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    # K tails are supported: the K loop is ceil_div and the TMA descriptor's global K
    # extent makes a partial box HW zero-filled. The only real K rule is the 16-byte
    # TMA contiguous-extent one, already gated by _tma_alignment_reject.
    sym_k = cute.sym_int64()
    sym_l = cute.sym_int64()
    if matmul_a_batch == 1:
        sym_a_l = 1
    else:
        sym_a_l = sym_l
    if matmul_b_batch == 1:
        sym_b_l = 1
    else:
        sym_b_l = sym_l

    def _make_fake_a():
        return make_fake_compact_tensor(
            mma_a_dtype,
            (sym_m, sym_k, sym_a_l),
            stride_order=(0, 1, 2) if a_is_m_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            mma_b_dtype,
            (sym_n, sym_k, sym_b_l),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    # ---- FP8 NORM+ROPE+QUANT FUSION: four compact outputs (their widths are NOT the weight's
    # n), the bf16 tables, and the [4] fp32 scale vector.  Static [D] weights: the artifact
    # then refuses a wrong-length weight at the call boundary; cos/sin are [T, x] over the
    # symbolic M.
    sym_n_q8 = cute.sym_int64(divisibility=32)  # e4m3: whole 32-B TMA rows
    sym_n_k8 = cute.sym_int64(divisibility=32)
    sym_n_v8 = cute.sym_int64(divisibility=32)
    sym_n_gate = cute.sym_int64(divisibility=16)  # bf16: 16 elems = 32 B
    fake_c_q8 = make_fake_compact_tensor(cutlass.Float8E4M3FN, (sym_m, sym_n_q8, sym_l), stride_order=(1, 0, 2), assumed_align=16)
    fake_c_k8 = make_fake_compact_tensor(cutlass.Float8E4M3FN, (sym_m, sym_n_k8, sym_l), stride_order=(1, 0, 2), assumed_align=16)
    fake_c_v8 = make_fake_compact_tensor(cutlass.Float8E4M3FN, (sym_m, sym_n_v8, sym_l), stride_order=(1, 0, 2), assumed_align=16)
    fake_c_gate = make_fake_compact_tensor(cutlass.BFloat16, (sym_m, sym_n_gate, sym_l), stride_order=(1, 0, 2), assumed_align=16)
    # qk_norm=False: the weights are None at the ABI (and the artifact then refuses a tensor there).
    fake_w_q = make_fake_compact_tensor(cutlass.BFloat16, (_D,), stride_order=(0,), assumed_align=16) if _QK_NORM else None
    fake_w_k = make_fake_compact_tensor(cutlass.BFloat16, (_D,), stride_order=(0,), assumed_align=16) if _QK_NORM else None
    fake_cos = make_fake_compact_tensor(cutlass.BFloat16, (sym_m, _ROPE_DIM), stride_order=(1, 0), assumed_align=16)
    fake_sin = make_fake_compact_tensor(cutlass.BFloat16, (sym_m, _ROPE_DIM), stride_order=(1, 0), assumed_align=16)
    fake_qscal = make_fake_compact_tensor(cutlass.Float32, (4,), stride_order=(0,), assumed_align=16)

    def _sym_operand_strides(is_mn_major: bool) -> tuple:
        # Operand is permuted to (M|N, K, L): the unit stride is mode 0 when MN-major, mode 1 when K-major, and never reaches TMA.
        unit = 0 if is_mn_major else 1
        return tuple(cute.sym_int64() if i == unit else cute.sym_int64(divisibility=ab_stride_elems) for i in range(3))

    sym_a_strides = []
    for _ in range(num_a_operands):
        sym_a_strides.extend(_sym_operand_strides(a_is_m_major))
    sym_b_strides = []
    for _ in range(num_b_operands):
        sym_b_strides.extend(_sym_operand_strides(b_is_n_major))
    sym_out_stride_m_0 = cute.sym_int64()
    sym_out_stride_n_0 = cute.sym_int64()
    sym_out_stride_l_0 = cute.sym_int64()
    sym_out_stride_m_1 = cute.sym_int64()
    sym_out_stride_n_1 = cute.sym_int64()
    sym_out_stride_l_1 = cute.sym_int64()
    sym_out_stride_m_2 = cute.sym_int64()
    sym_out_stride_n_2 = cute.sym_int64()
    sym_out_stride_l_2 = cute.sym_int64()
    sym_out_stride_m_3 = cute.sym_int64()
    sym_out_stride_n_3 = cute.sym_int64()
    sym_out_stride_l_3 = cute.sym_int64()
    fake_a_0 = _make_fake_a()
    fake_b_0 = _make_fake_b()
    problem_size = (
        sym_m,
        sym_n,
        sym_k,
        sym_l,
        *sym_a_strides,
        *sym_b_strides,
        sym_out_stride_m_0,
        sym_out_stride_n_0,
        sym_out_stride_l_0,
        sym_out_stride_m_1,
        sym_out_stride_n_1,
        sym_out_stride_l_1,
        sym_out_stride_m_2,
        sym_out_stride_n_2,
        sym_out_stride_l_2,
        sym_out_stride_m_3,
        sym_out_stride_n_3,
        sym_out_stride_l_3,
    )
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return cute.compile(
        _host,
        problem_size,
        fake_a_0,
        fake_b_0,
        fake_c_q8,
        fake_c_k8,
        fake_c_v8,
        fake_c_gate,
        fake_w_q,
        fake_w_k,
        fake_cos,
        fake_sin,
        fake_qscal,
        stream=_fake_stream,
        options=frost_compile_options,
    )
