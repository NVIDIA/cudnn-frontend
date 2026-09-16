# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage (1) of the gated attention block on the MXFP8 pipeline, with stages (2)+(3)
AND the Q/K/V BLOCK quantization (e4m3 codes + F8_128x4 E8M0 scale factors) FUSED
into its epilogue.

``PROJ = h8 @ W8_qkvg^T`` as the shipped FROST BLOCK-SCALE GEMM computes it (e4m3 x
e4m3 with one E8M0 scale per 32-element K block on BOTH operands, dequantized IN the
MMA -- ``tcgen05.mma.block_scale`` -- fp32 accumulate; there is NO alpha: the
accumulator is already in activation units), and then, per tile class:

  Q / K   [per-head RMSNorm over D if ``qk_norm``] on the fp32 accumulator, partial RoPE
          on the leading ``ROPE_DIM`` columns, then per 32-column subtile (== one MXFP8
          block along D, the ROWWISE quantization the SDPA's BMM1 consumes):
          ``e = cvt.rp.satfinite.ue8m0x2(amax * fp32(1/448))``, ``code = e4m3_rn_satfinite(x * 2^(127-e))``
          into its OWN compact e4m3 tensor ``q8 [T, h_q*d]`` / ``k8 [T, h_kv*d]``, and the
          E8M0 byte into the SDPA's per-(b, h, s_tile) 1024-B F8_128x4 tile of ``sf_q`` / ``sf_k``.
  V       per 32-column subtile, 32 warp-level abs-max reductions over the warp's 32
          consecutive rows (== one 32-token block along S, the COLUMNWISE quantization the
          SDPA's BMM2 consumes); lane ``l`` owns column ``col + l``: its code goes into
          ``v8 [T, h_kv*d]``, its E8M0 byte into the D-PLANE-major ``sf_v``.
  GATE    the rendering's own epilogue -- ``bf16(acc)`` -- into ``gate16 [T, h_q*d]``.

So the block's whole MXFP8 stage-1 chain (block-scale GEMM, norm + RoPE, THREE block
quantize passes = 5 launches unfused) is ONE launch writing FOUR compact tensors plus
THREE scale-factor blobs, byte-for-byte what the unfused path
(``kernels/quantize_mxfp8.py``) hands ``sm107/prefill_d256_mxfp8.py``.

SCALE-FACTOR LAYOUT (PR-B plan section 2.3; ``mma-tma-matrix.md`` section 7)
------------------------------------------------------------------------------
Every SF byte below is read by the SDPA's TMA descriptors as WHOLE tiles, so a wrong
byte is a wrong number, never an error, and an UNWRITTEN byte is E8M0 NaN:

* Q / K (rowwise): tile ``(b, h, s_tile)`` = ``4*D`` (1024) contiguous bytes at
  ``((b*H + h)*n_tiles + s_tile) * 1024``; inside, block ``c = d//32`` of row ``s`` sits at
  ``(c//4)*512 + (s%32)*16 + ((s%128)//32)*4 + c%4``.  One CTA tile is one head x 128 rows
  == one SF tile, and thread ``tidx`` IS row ``s % 128`` of it, so a thread's 8 subtile
  bytes are two contiguous 4-byte words: ``atom*512 + (tidx%32)*16 + ((tidx//32)%4)*4``
  with ``atom = subtile // 4`` -- two ``st.global.b32`` per thread per tile.
* V (columnwise): byte ``(b, h, s_tile, d, s)`` =
  ``(d//128) * (B*KH*n_tiles*512) + ((b*KH + h)*n_tiles + s_tile)*512 + ((d%128)%32)*16 + ((d%128)//32)*4 + (s%128)//32``.
  Lane ``l`` of warp ``w`` (rows ``32w..32w+31`` == one 32-token block) owns column
  ``col + l`` of subtile ``j``: ``plane = j // 4``, byte ``plane*v_sf_groups*512 + l*16 + (j%4)*4 + w``
  -- one ``st.global.b8`` per lane per subtile.  ``v_sf_groups = B*KH*n_tiles`` is a runtime
  argument (the plane stride GROWS with ``B*KH*S``).
* Tail rows (``row >= M``) come in TWO classes, and the epilogue handles them differently:
  - a PARTIALLY valid tile (``coord_m_tile < M < coord_m_tile + 128``; only reachable at
    ``B == 1``): the data store is clipped by the descriptor, the Q/K SF word is SELECTED
    to ``0x00`` (the accumulator residue may be NaN -> ``0xFF``; ``sdpa-invariants.md``
    section 2: a select, never a multiply), and the V block amax sees those rows as ZERO
    through a per-element select -- exactly the oracle's zero padding
    (``quantize_to_mxfp8`` pads S to 128, pad SF = ``0x00``);
  - a FULLY tail tile (``coord_m_tile >= M``): the cluster tile is 256 rows and each CTA
    drains its own 128, so whenever ``M % 256`` is in ``(0, 128]`` (``B=1 S=128``,
    ``B=3 S=128``, ...) the second CTA of the last cluster tile has NO tile slot in the
    SF blobs -- its ``(b, s_tile)`` decode reads ``b == B``.  Its SF stores are gated OFF
    on the warp-uniform ``_tile_valid = coord_m_tile < M`` (the data stores are TMA-
    clipped as before).  Ungated, they wrote ``0x00`` past every blob (in the block: over
    the neighbouring ``sf_k`` slot) and V's plane 0 ON the valid tiles' plane 1 (review
    finding, 2026-09-15; pinned by ``test_fused_mxfp8_fully_tail_cta_stores_no_scale_factor_bytes``).
* **``S % 128 != 0 and B > 1`` is a typed decline** (``run_fused_proj_gemm_mxfp8``): a
  128-row GEMM tile would then straddle two sequences and the (b, s_tile) decode above --
  done ONCE per tile from ``coord_m_tile`` -- would be wrong for part of the tile.  The
  unfused block-scale path serves that shape.

NUMERICS (one rounding per output; the oracle is the fp32 chain quantized once)
------------------------------------------------------------------------------
* ``rstd = rsqrt(sum(acc^2) / D + eps)`` -- no alpha term: the block-scale accumulator is
  dequantized.  ``qk_norm=False``: no pass A, no rsqrt, no weight loads -- RoPE on the raw
  accumulator, then quantize.
* E8M0: ``tile_dsl.pointwise.e8m0_from_amax`` (Q/K, one cvt per block) / ``e8m0_pair`` (V,
  one cvt per two columns) -- ``cvt.rp.satfinite.ue8m0x2.f32`` of ``amax * fp32(1/448)``,
  the EXACT ``0x3B124925`` as a register operand; ``rcp = bits((254 - e) << 23)``.  Bit-exact
  with ``mxfp8_quant.quantize_blocks`` for finite inputs.  Block amax: ``abs_max_tree``
  (PTX ``max.f32`` -> FMNMX3; ``cute.math.max`` gives compare + select) for the row
  blocks, ``redux.sync.max.abs.f32`` (``cute.arch.warp_redux_sync(..., "fmax", abs=True)``,
  sm_100a+) for the column blocks.
* e4m3 conversion is ``cvt.rn.satfinite.e4m3x2.f32`` (``pointwise.fp32_to_fp8_pack``); the
  packed ``Int32`` words are BITCAST to a ``Float8E4M3FN`` vector before ``store_swizzled``.
* No per-tensor scale vector, no alpha aux: the ONLY runtime scalars are ``seq_len`` and
  ``v_sf_groups`` (kernel parameters, constant bank -- nothing extra live across the
  accumulator wait).

WHY THE FUSION IS TILE-LOCAL, AND THE EPILOGUE PER TILE
--------------------------------------------------------
Identical to the FP8 fork ``proj_gemm_norm_rope_fp8.py`` and the bf16 fork
``proj_gemm_norm_rope.py`` (read their docstrings): config
``CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma`` gives one CTA output tile ==
one head (128 rows x 256 cols), one thread per row, one 32-column subtile per
``tcgen05.ld`` -- which is also exactly one MXFP8 block along D, and a warp's 32 rows are
exactly one MXFP8 block along S.  The tile class (Q / K / V / GATE) is a function of
``coord_n_c`` alone, decided BEFORE the accumulator wait, warp-uniform.  Every arm runs
the IDENTICAL per-subtile synchronization skeleton of the rendering: ring-slot advance,
``fence_view_async_shared``, both ``EPI_SYNC_BAR_ID`` barriers, warp-0 TMA store + commit
+ ``cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1)``, and the ``acc_empty`` arrive after
the LAST ``tcgen05.ld`` of the tile.

``NORM_SOURCE``: ``ldg_early`` (DEFAULT), ``ldg``, and the diagnostic floors ``const`` /
``const_w`` / ``const_cs`` exactly as the FP8 fork defines them (NOT correct; they price
the load classes).  There is no ``off`` here: no single bf16 slab to degenerate to.

BARRIER TABLE: UNCHANGED.  SMEM TABLE: UNCHANGED.  This fork adds no mbarrier, no
named barrier and no SMEM buffer to the block-scale rendering (whose SF rings are
declared FIRST so every tcgen05 descriptor root stays below 256 KiB -- see the
rendering's own comment).  It adds THREE TMA-store descriptors (q8 / k8 / v8 e4m3 + the
bf16 gate), per-lane global loads (cos / sin / w) and per-lane global SF byte stores.

KEEP IN SYNC WITH ``gemm/frost/kernel_templates/sm100_block_scale_matmul.py`` AND THE TWO FORKS
------------------------------------------------------------------------------------------------
This file is the rendered MXFP8 block-scale expansion of that template for the config
above -- ``frost_dev/renderings/proj_gemm_mxfp8_rendered.py`` (``frost_dev/render_proj_gemm.py
--dtype mxfp8``: e4m3 A/B, E8M0 SF, ``MXF8F6F4``, ``cta_tile_mnk=(128, 128, 128)``,
``mma_inst K=32``, 576 exclusive TMEM columns, 9 stages) -- plus the fusion DELTA of the
FP8 fork (``diff frost_dev/renderings/proj_gemm_fp8_alpha_rendered.py
kernels/proj_gemm_norm_rope_fp8.py``) re-applied between the ``MXFP8 NORM+ROPE+QUANT
FUSION`` markers, with the ``alpha`` / ``qscal`` reads dropped and the per-tensor scale
replaced by the per-block E8M0.  It was NOT produced by patching the FP8 fork's
constants header (``frost-tile-dsl.md`` S5: diff the RENDERINGS, not the edit).
Mainloop / SF rings / UTCCP / scheduler / TMEM pipeline / register split are the
rendering's, verbatim; any fix there applies to ALL of them.

Parameters come from the FROST template loader as ``FROST_TEMPLATE_PARAMS``
(a :class:`~cudnn.gated_attention_block.kernels.proj_gemm.NormRopeFusionParams`
with ``quant_mxfp8=True``); a plain import gets the 397B geometry so the file runs
standalone.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.pointwise import abs_max_tree, e8m0_from_amax, e8m0_pair, f16x2_to_f32, fp32_to_fp8_pack, opaque_f32_zero
from cudnn.frost.tile_dsl.sass import keep_sass_options
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global
from cudnn.gated_attention_block.kernels.proj_gemm import NormRopeFusionParams, validate_norm_rope_params

import cutlass.experimental.primitives as nvvm
from cudnn.gemm.frost.sm100.kernel_templates._tile_helpers import (
    epi_subtile_spans as _epi_subtile_spans,
    l2_swizzle_tile as _l2_swizzle_tile,
    tcgen05_alloc as _tcgen05_alloc,
    tcgen05_dealloc as _tcgen05_dealloc,
    tcgen05_mma_block_scale as _tcgen05_mma_block_scale,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass._mlir_helpers.vector as _cvec
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda
from cutlass.cute.arch import clc as cute_clc

# Block-scale config: CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma data=fp8_e4m3xfp8_e4m3 sf=fp8_e8m0 block=32
cta_group = 2
cta_tile_mnk = (128, 128, 128)
mma_size_m = 1
mma_size_n = 1
mma_size_k = 4
cgrp_tile_mnk = (256, 256, 128)
cgrp_tile_m = 256
cgrp_tile_n = 256
epi_tile_mn = (128, 32)
threads_per_cta = 256
cluster_shape_mnk = (2, 1, 1)
matmul_a_batch = 1
matmul_b_batch = 1
ab_stages = 9
# tcgen05 SMEM descriptor roots (bytes, declaration order; Tcgen05SmemDesc.build() keeps 14 bits of addr>>4, so each must stay < 262144): SFA[0]=2048 SFB[0]=7168 A[0]=16384 B[0]=163840
acc_stages = 2
use_acc_overlap = False
acc_stage_stride = 256
acc_overlap_subtiles = 0
num_gemms = 1
num_a_operands = 1
num_b_operands = 1
num_sfa_operands = 1
num_sfb_operands = 1
fake_dequant_a = False
fake_dequant_b = False
gemm_a_idx = (0,)
gemm_b_idx = (0,)
acc_region_cols = 256
acc_gemm_stride = 512
sfa_col_bases = (512,)
sfb_col_bases = (516,)
sfa_tmem_cols = 4
sfb_tmem_cols = 8
tile_swizzle_n = 0
swizzle_l2_budget_bytes = 41943040
multicast_a = False
multicast_b = False
a_mcast_slices = 1
a_tma_box_m = 128
b_mcast_slices = 1
ab_empty_full_mask = False

# packed data SMEM
a_dtype = cutlass.Float8E4M3FN
b_dtype = cutlass.Float8E4M3FN
a_smem_dtype = cutlass.Float8E4M3FN
b_smem_dtype = cutlass.Float8E4M3FN
ab_max_data_bits = 8
a_fake_dtype = cutlass.Float8E4M3FN
b_fake_dtype = cutlass.Float8E4M3FN
a_packed_per_row = 128
b_packed_per_row = 128
sA_packed_elems = 16384
sB_packed_elems = 16384
sA_tma_bytes = 16384
sB_tma_bytes = 16384
a_tma_desc_dtype = cutlass.Float8E4M3FN
b_tma_desc_dtype = cutlass.Float8E4M3FN
a_tma_format = None
b_tma_format = None
a_tma_swizzle = _tma.TensorMapSwizzle.s128b
b_tma_swizzle = _tma.TensorMapSwizzle.s128b
a_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B
b_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B
a_smem_desc_leading_byte_offset = 16
a_smem_desc_stride_byte_offset = 1024
a_smem_k_step_bytes = 32
a_tma_group_elems = 1
b_smem_desc_leading_byte_offset = 16
b_smem_desc_stride_byte_offset = 1024
b_smem_k_step_bytes = 32
b_tma_group_elems = 1
a_is_m_major = False
b_is_n_major = False
mma_a_major = 0
mma_b_major = 0

# output
cd_dtype = cutlass.BFloat16
epi_store_dtype = cutlass.BFloat16
vec_bytes_epi = 32
split_k_slices = 1
frost_compile_options = "--enable-tvm-ffi --gpu-arch sm_107a"
n_tma_outputs = 4  # MXFP8 NORM+ROPE+QUANT FUSION: compact e4m3 q8 / k8 / v8 + bf16 GATE (rendering: 1); the prefetch loop covers all four
moe_aligned_offsets = False
epi_slot_widen = 1
epi_stage_rows = 128
epi_chunk_elems = 32
epi_n = 32
epi_row_elems = 32

# block-scale MMA
mma_block_scale_kind = nvvm.MMABlockScaleKind.MXF8F6F4
scale_vec_size = nvvm.Tcgen05MMABlockScale.BLOCK32
idesc_a_dtype = cutlass.Float8E4M3FN
idesc_b_dtype = cutlass.Float8E4M3FN
sf_scale_format = 1
sf_one_word = 2139062143
mma_m_dim = 256
mma_n_dim = 256

# scale factors
block_size = 32
sf_cutlass_dtype = cutlass.Float8E8M0FNU
sf_scales_per_inst = 1
sf_insts_per_atom = 4
num_sf_atoms = 1
word_atoms = 1
num_blocks_m = 1
num_blocks_n = 2
registers_per_block = 4
epi_cols_per_mma_m = 256
mma_c_dtype = cutlass.Float32
a_smem_m_step_bytes = 16384
registers_per_atom = 4
sf_atom_desc_stride = 32
sf_block_desc_stride = 32
num_tmem_alloc_cols = 576
tmem_alloc_exclusive = True
b_collector_ok = True
sfa_smem_bytes = 512
sfb_smem_bytes = 1024
sf_tma_box_k = 1
sfa_tma_box_mn = 1
sfb_tma_box_mn = 2

# block-scale MMA: 4 MMAs per K-tile at mma_inst_k_bytes=32
idesc_is_omma = False
mma_k_dim_mode = 0
fallback_cluster_shape_mnk = None
mixed_a_pattern_pref = 1
mixed_b_pattern_pref = 1
mixed_a_pattern_fb = 1
mixed_b_pattern_fb = 1

# ---------------------------------------------------------------------------
# MXFP8 NORM+ROPE+QUANT FUSION: geometry + knobs (compile-time; every use is const_expr)
# ---------------------------------------------------------------------------
PARAMS: NormRopeFusionParams = globals().get("FROST_TEMPLATE_PARAMS", NormRopeFusionParams(quant_mxfp8=True))
if not PARAMS.quant_mxfp8:
    raise ValueError(
        f"{__name__}: this is the MXFP8 fork; it needs NormRopeFusionParams(quant_mxfp8=True) "
        "(the FP8 fork is proj_gemm_norm_rope_fp8.py, the bf16 fork proj_gemm_norm_rope.py)"
    )
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
NORM_SOURCE: str = PARAMS.norm_source  # "ldg" | "ldg_early" | "const" | "const_w" | "const_cs"  (no "off" on the MXFP8 fork)
# False: the RoPE-ONLY Q/K arm -- no pass A (sum of squares), no rsqrt, no norm-weight loads;
# the Q/K tiles are rotated on the fp32 accumulator and block-quantized.  Every norm-only
# block below sits under const_expr(_QK_NORM); the weights are None at the ABI.
_QK_NORM: bool = PARAMS.qk_norm
# The two epilogue load classes are gated SEPARATELY so a diagnostic arm can delete one at a time
# (exactly the FP8 fork's vocabulary; see its docstring):
_LDG_W: bool = NORM_SOURCE in ("ldg", "ldg_early", "const_cs")
_LDG_CS: bool = NORM_SOURCE in ("ldg", "ldg_early", "const_w")
_CS_EARLY: bool = NORM_SOURCE in ("ldg_early", "const_w")
_ROPE_SUBTILES: int = _ROPE_DIM // epi_n
_ROPE_HALF_SUBTILES: int = _ROPE_SUBTILES // 2
_ROPE_HALF: int = _ROPE_DIM // 2
# cos / sin / norm-weight tables are bf16 REGARDLESS of the GEMM dtypes: pin their
# byte width (deriving it from cd_dtype / a_dtype halves the ld.global strides).
_ACT_BPE: int = 2
# Pass A (sum of squares) reads each 32-column subtile as _PASS_A_SPLIT tcgen05.ld's of
# 32 / _PASS_A_SPLIT columns -- the FP8 fork's register-pressure lever (see its pass-A comment).
_PASS_A_SPLIT: int = 4
# ---- MXFP8 scale-factor geometry (PR-B section 2.3 / kernels/quantize_mxfp8.py; the SDPA's consumer layout) ----
_SF_BLOCK: int = 32  # elements per E8M0 scale (== epi_n: one subtile is one row block)
_SF_TILE_ROWS: int = 128  # rows of an F8_128x4 atom == the SDPA's Q / KV tile height == this CTA tile's rows
_SF_ATOM_BYTES: int = 512  # 128 rows x 4 blocks
_SF_ATOM_BLOCKS: int = 4  # blocks (columns) per atom
_SF_TILE_BYTES: int = _SF_TILE_ROWS * _D // _SF_BLOCK  # bytes per (b, h, s_tile) unit: 1024 at D=256 (rowwise == columnwise)
_SF_SUBTILES_PER_ATOM: int = _SF_ATOM_BLOCKS  # rowwise: 4 subtiles (4 blocks) fill one 512-B atom
_SF_SUBTILES_PER_PLANE: int = _SF_TILE_ROWS // epi_n  # columnwise: 4 subtiles (128 d) fill one D-plane
_LOG2_D: int = _D.bit_length() - 1  # head index = (col - class offset) >> _LOG2_D
_LOG2_SF_TILE_ROWS: int = _SF_TILE_ROWS.bit_length() - 1
# The facts the fusion stands on, checked against THIS rendering's constants.
if use_acc_overlap or (cgrp_tile_mnk[1] // cluster_shape_mnk[1]) != _D or epi_cols_per_mma_m != _D:
    raise ValueError(
        f"{__name__}: the fused epilogue needs one CTA output tile == one head, drained subtile-by-subtile in order: per-CTA N tile "
        f"{cgrp_tile_mnk[1] // cluster_shape_mnk[1]} (epi_cols_per_mma_m={epi_cols_per_mma_m}, use_acc_overlap={use_acc_overlap}) vs d_head={_D}. Re-render for another config."
    )
if _ROPE_DIM % (2 * epi_n) != 0 or _ROPE_DIM >= _D:
    raise ValueError(f"{__name__}: rope_dim={_ROPE_DIM} must be a multiple of {2 * epi_n} (two whole subtiles per rotate_half pair) and < d_head={_D}")
if a_dtype is not cutlass.Float8E4M3FN or b_dtype is not cutlass.Float8E4M3FN or mma_c_dtype is not cutlass.Float32 or cd_dtype is not cutlass.BFloat16:
    raise ValueError(
        f"{__name__}: this fork was rendered for an e4m3 x e4m3 block-scale GEMM -> fp32 with a bf16 C (the gate slab); the rendering's constants say otherwise"
    )
if sf_cutlass_dtype is not cutlass.Float8E8M0FNU or block_size != _SF_BLOCK or fake_dequant_a or fake_dequant_b:
    raise ValueError(
        f"{__name__}: this fork needs E8M0 scale factors on BOTH operands at block 32 (sf dtype {sf_cutlass_dtype}, block {block_size}, fake {fake_dequant_a}/{fake_dequant_b})"
    )
if (
    mma_size_m != 1
    or num_gemms != 1
    or epi_n != _SF_BLOCK
    or epi_tile_mn[0] != _SF_TILE_ROWS
    or epi_stage_rows != _SF_TILE_ROWS
    or cta_tile_mnk[0] != _SF_TILE_ROWS
):
    raise ValueError(
        f"{__name__}: this fork needs the plain thread-per-row single-GEMM epilogue with 32-column subtiles over a 128-row tile "
        f"(one subtile == one MXFP8 row block, one 128-row tile == one F8_128x4 SF tile); got mma_size_m={mma_size_m} epi_n={epi_n} rows={epi_tile_mn[0]}"
    )
if _D % _SF_TILE_ROWS or (_D // _SF_BLOCK) % _SF_ATOM_BLOCKS:
    raise ValueError(f"{__name__}: d_head={_D} must be a multiple of 128 (whole F8_128x4 atoms rowwise, whole D-planes columnwise)")
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


def _st_global_b8(addr, value):
    """8-bit global store of the low byte of an ``Int32`` (PTX lets an 8-bit ``st`` take a 32-bit register)."""
    nvvm.inline_ptx("st.global.b8 [$0], $1;", read_only_args=[addr, value])


def _sel_i32(cond, a, b):
    """``cond ? a : b`` on ``Int32`` values (a runtime ``Boolean`` condition; no branch)."""
    return cutlass.Int32(arith.select(cond.ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()))


def _sel_i64(cond, a, b):
    return cutlass.Int64(arith.select(cond.ir_value(), cutlass.Int64(a).ir_value(), cutlass.Int64(b).ir_value()))


def _sel_f32(cond, a, b):
    return cutlass.Float32(arith.select(cond.ir_value(), cutlass.Float32(a).ir_value(), cutlass.Float32(b).ir_value()))


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

if use_acc_overlap and any(_w != epi_n for _, _w in _epi_subtile_spans(epi_cols_per_mma_m, epi_n)):
    raise NotImplementedError(f"{__name__}: acc overlap reverses subtiles by index, which needs a uniform drain width")


# Scheduler ring depth.
CLC_SCHED_STAGES = 1

# Programmatic Dependent Launch (PDL, sm_90+).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path
EPI_SMEM_STAGES = 2

# Named barrier id for the 4-warp epilogue handoff around the TMA store.
EPI_SYNC_BAR_ID = 1

# Named barrier id for the TMEM-alloc handoff
TMEM_ALLOC_BARRIER_ID = 2

# Five-warp handoff after epilogue warps seed all fake-SF TMEM partitions.
TMEM_SCALE_ONE_BARRIER_ID = 3


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
    row_bytes = (cutlass.Int64(ab_max_data_bits) * k) // 8
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


@cute.jit
def _fill_scale_one(tmem_base, num_cols):
    """Fill one logical SF region with the selected format's encoded 1.0.

    The F8_128x4 layout uses 32 scale rows and a 32-bit cell carries four
    identical scale bytes.  One warp-wide ``32x32b`` store fills those rows in
    only the issuing warp's TMEM partition.  Calling this from warp positions
    0..3 supplies the same four-partition coverage as an SF
    ``tcgen05.cp(..., WARPX4)``.
    """
    one = cutlass.Uint32(sf_one_word)
    one_vec = cutlass.Vector.from_elements((one,), cutlass.Uint32)
    for col in cutlass.range_constexpr(num_cols):
        nvvm.tcgen05_st(
            "32x32b",
            nvvm.make_tmem_ptr(tmem_base + col, cutlass.Uint32),
            one_vec,
        )
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)


@cute.kernel
def frost_sm100_block_scale_matmul_128x256x128_128x256x32_cluster2x1_2ctamma(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_sfa_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_sfb_desc_0: cutlass.GridConstant[_tma.TensorMap],
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_l_0: cutlass.Int64,
    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],  # e4m3 q8 [N_Q, M]
    # ---- MXFP8 NORM+ROPE+QUANT FUSION: appended, so the rendered prefix stays diffable ----
    tma_c_desc_1: cutlass.GridConstant[_tma.TensorMap],  # e4m3 k8 [N_KV, M]
    tma_c_desc_2: cutlass.GridConstant[_tma.TensorMap],  # e4m3 v8 [N_KV, M]
    tma_c_desc_3: cutlass.GridConstant[_tma.TensorMap],  # bf16 gate16 [N_GATE, M] (the rendering's own C descriptor form)
    w_q_norm: Optional[cute.Tensor],  # [D]      bf16, or None (qk_norm=False: folds out)
    w_k_norm: Optional[cute.Tensor],  # [D]      bf16, or None
    cos_tab: cute.Tensor,  # [T, ROPE_DIM] bf16, per-token, halves duplicated
    sin_tab: cute.Tensor,  # [T, ROPE_DIM]
    sf_q_out: cute.Tensor,  # [B*H_Q*n_tiles*1024]  uint8, F8_128x4 rowwise tiles
    sf_k_out: cute.Tensor,  # [B*H_KV*n_tiles*1024] uint8
    sf_v_out: cute.Tensor,  # [B*H_KV*n_tiles*1024] uint8, D-plane-major columnwise
    seq_len: cutlass.Int32,  # S: (b, s) of a flat row; n_tiles = ceil(S/128)
    v_sf_groups: cutlass.Int32,  # B*H_KV*n_tiles: the V SF D-plane stride in 512-B atoms
) -> None:
    tma_a_descs = [tma_a_desc_0]
    tma_b_descs = [tma_b_desc_0]
    tma_sfa_descs = [tma_sfa_desc_0]
    tma_sfb_descs = [tma_sfb_desc_0]
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
        for _i in cutlass.range_constexpr(num_sfa_operands):
            nvvm.prefetch_tensormap(tma_sfa_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())
        for _j in cutlass.range_constexpr(num_sfb_operands):
            nvvm.prefetch_tensormap(tma_sfb_descs[_j].get_ptr())

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

        a_part_arrive = cutlass.Int16(a_pattern) << m_rank
        b_part_arrive = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
        if cutlass.const_expr(ab_empty_full_mask):
            ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
        else:
            ab_empty_arrive_mask = a_part_arrive | b_part_arrive
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
    sf_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

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

    sA_elems = sA_packed_elems
    sB_elems = sB_packed_elems
    # Declaration order IS the SMEM layout, and here it is load-bearing.  Every
    # ring ROOT feeds `Tcgen05SmemDesc.build(start_address=...)`, whose lowering
    # (cutlass-dsl experimental/primitives/descriptors.py:513-522, the
    # non-versioned `_tcgen05_mma_smem_desc` intrinsic) keeps only 14 bits of
    # `addr >> 4`: a root at or above 262144 B wraps to the bottom of SMEM with
    # no error, and the SF UTCCP then copies A-operand bytes into the SF TMEM
    # columns -> NaN/inf on every output.  Reachable on sm107 only, whose 327 KiB
    # carveout lets the AB ring run past 256 KiB.  `advance_start_address`
    # (descriptors.py:413-425) is a plain encoded add and carries the per-stage
    # and per-k-step offsets past the line correctly (the d512 SDPA kernels
    # already rely on it), so the SMALL scale-factor rings are declared FIRST and
    # the big A/B rings -- whose roots then stay far below the line -- follow.
    # The compiler models these roots (`_block_scale_smem_desc_roots`), trims
    # `ab_stages` when a deeper ring would still put one past the line (the
    # MoE template at sm107 512x128), and refuses a layout a single stage cannot
    # fit; the CPU test test_block_scale_smem_layout_sm107.py pins this order.
    # Do not reorder.
    smem_sfa_list = [
        cutlass.Array(
            cutlass.Uint8,
            sfa_smem_bytes * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_sfa_operands)
    ]
    smem_sfb_list = [
        cutlass.Array(
            cutlass.Uint8,
            sfb_smem_bytes * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_sfb_operands)
    ]
    smem_a_list = [
        cutlass.Array(
            a_smem_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            b_smem_dtype,
            sB_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    # The ring slot is indexed by `tidx`, so its row count is the EPILOGUE THREAD
    # count -- which is epi_tile_mn[0] only when the MMA M block is 128.
    epi_subtile_elems = epi_stage_rows * epi_row_elems * epi_slot_widen
    smem_d_ptr = cutlass.Array(
        epi_store_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    if cutlass.const_expr(cta_group == 2):
        acc_empty_count = num_epilogue_warps * 2
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
            if cutlass.const_expr(use_acc_overlap):
                if elect_one:
                    nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, num_epilogue_warps)
            else:
                if elect_one:
                    nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        else:
            for i in range(ab_stages):
                if elect_one:
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(sf_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
            for i in range(acc_stages):
                if elect_one:
                    nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), num_epilogue_warps)
            if cutlass.const_expr(use_acc_overlap):
                if elect_one:
                    nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, num_epilogue_warps)
        if cutlass.const_expr(cta_group == 2):
            for i in range(ab_stages):
                if elect_one:
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(sf_full_mbar_ptr.subview(i), 1)
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

    sA_bytes = sA_elems * (a_smem_dtype.width // 8)
    sB_bytes = sB_elems * (b_smem_dtype.width // 8)
    if cutlass.const_expr(cta_group == 1):
        ab_only_copy_bytes = num_a_operands * sA_tma_bytes + num_b_operands * sB_tma_bytes
        sf_only_copy_bytes = num_sfa_operands * sfa_smem_bytes + num_sfb_operands * sfb_smem_bytes
    else:
        ab_only_copy_bytes = (num_a_operands * sA_tma_bytes + num_b_operands * sB_tma_bytes) * 2
        sf_only_copy_bytes = (num_sfa_operands * sfa_smem_bytes + num_sfb_operands * sfb_smem_bytes) * 2

    if cutlass.const_expr(cta_group == 2):
        # Per-CTA logical tile — the cluster cancels out, so these stay compile-time
        # constants even when the cluster shape is only known at runtime.
        logical_cta_tile_m = cgrp_tile_mnk[0] // cluster_shape_mnk[0]
        logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_shape_mnk[1]
        pair_n_size = logical_cta_tile_n
        # Per-CTA output rows one MMA-M block covers. The pair splits M, so this is
        # the per-CTA mma_inst_m — half the instruction's hardware M.
    epi_rows_per_mma_m = cta_tile_mnk[0] // mma_size_m
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
    if cutlass.const_expr(cta_group == 1):
        cgrp_tile_m_cur = cta_tile_mnk[0] * cluster_m
        cgrp_tile_n_cur = cta_tile_mnk[1] * cluster_n
    else:
        cgrp_tile_m_cur = logical_cta_tile_m * cluster_m
        cgrp_tile_n_cur = logical_cta_tile_n * cluster_n

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
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
                coord_sf_k = k_tile_idx * sf_tma_box_k
                if cutlass.const_expr(cta_group == 1):
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), ab_only_copy_bytes)
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(sf_full_mbar_ptr.subview(stage), sf_only_copy_bytes)
                else:
                    coord_n_pair = tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n
                    sfb_n_block = coord_n_pair // 128

                    if is_pair_leader:
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), ab_only_copy_bytes)
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(sf_full_mbar_ptr.subview(stage), sf_only_copy_bytes)

                for _ai in cutlass.range_constexpr(num_sfa_operands):
                    sSFA_stage = smem_sfa_list[_ai].subview(sfa_smem_bytes * stage)
                    tma_sfa_desc = tma_sfa_descs[_ai]
                    sfa_m_block = coord_m_per_cta // 128
                    if cutlass.const_expr(multicast_a):
                        if n_rank == 0:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sSFA_stage,
                                    tma_sfa_desc.get_ptr(),
                                    (0, coord_sf_k, sfa_m_block, tile_l_a),
                                    sf_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=_CTA_GROUP,
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sSFA_stage,
                                tma_sfa_desc.get_ptr(),
                                (0, coord_sf_k, sfa_m_block, tile_l_a),
                                sf_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_a,
                                group=_CTA_GROUP,
                            )

                for _bj in cutlass.range_constexpr(num_sfb_operands):
                    sSFB_stage = smem_sfb_list[_bj].subview(sfb_smem_bytes * stage)
                    tma_sfb_desc = tma_sfb_descs[_bj]
                    if cutlass.const_expr(cta_group == 1):
                        sfb_n_block = coord_n_per_cta // 128
                    if cutlass.const_expr(multicast_b):
                        if pair_m_idx == 0:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sSFB_stage,
                                    tma_sfb_desc.get_ptr(),
                                    (0, coord_sf_k, sfb_n_block, tile_l_b),
                                    sf_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=_CTA_GROUP,
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sSFB_stage,
                                tma_sfb_desc.get_ptr(),
                                (0, coord_sf_k, sfb_n_block, tile_l_b),
                                sf_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_b,
                                group=_CTA_GROUP,
                            )

                for _ai in cutlass.range_constexpr(num_a_operands):
                    sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                    tma_a_desc = tma_a_descs[_ai]
                    if cutlass.const_expr(a_mcast_slices > 1):
                        _a_rows = cta_tile_mnk[0] // a_mcast_slices
                        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                            for _am in cutlass.range_constexpr(cta_tile_mnk[0] // a_mcast_slices // a_tma_box_m):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(n_rank * _a_rows * a_packed_per_row + _am * a_tma_box_m * a_packed_per_row),
                                        tma_a_desc.get_ptr(),
                                        (coord_k, coord_m_per_cta + n_rank * _a_rows + _am * a_tma_box_m, tile_l_a),
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
                                for _am in cutlass.range_constexpr(cta_tile_mnk[0] // a_mcast_slices // a_tma_box_m):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(_a_idx * _a_rows * a_packed_per_row + _am * a_tma_box_m * a_packed_per_row),
                                            tma_a_desc.get_ptr(),
                                            (coord_k, coord_m_per_cta + _a_idx * _a_rows + _am * a_tma_box_m, tile_l_a),
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
                                for _am in cutlass.range_constexpr(cta_tile_mnk[0] // a_mcast_slices // a_tma_box_m):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(_am * a_tma_box_m * a_packed_per_row),
                                            tma_a_desc.get_ptr(),
                                            (coord_k, coord_m_per_cta + _am * a_tma_box_m, tile_l_a),
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
                            for _am in cutlass.range_constexpr(cta_tile_mnk[0] // a_mcast_slices // a_tma_box_m):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(_am * a_tma_box_m * a_packed_per_row),
                                        tma_a_desc.get_ptr(),
                                        (coord_k, coord_m_per_cta + _am * a_tma_box_m, tile_l_a),
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
                                    sB_stage.subview(pair_m_idx * _b_rows * b_packed_per_row),
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
                                        sB_stage.subview(_b_idx * _b_rows * b_packed_per_row),
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

    if cutlass.const_expr(cta_group == 2):
        pair_mask = cutlass.Int16(3) << pair_leader_rank
        a_arrive_pattern = a_mcast_pattern
        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
            b_arrive_pattern = (1 << cluster_m) - 1
        else:
            b_arrive_pattern = (cutlass.Int32(1) << cluster_m) - 1
        a_part = a_arrive_pattern << m_rank
        a_part = a_part | (a_part << 1)
        b_part = b_arrive_pattern << (n_rank * cluster_m)
        if cutlass.const_expr(ab_empty_full_mask):
            ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
        else:
            ab_empty_arrive_mask = cutlass.Int16(a_part | b_part)
    if cutlass.const_expr(cta_group == 2):
        acc_full_mcast = pair_mask
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
        if cutlass.const_expr(fake_dequant_a or fake_dequant_b):
            nvvm.barrier_cta_sync(
                barrier_id=TMEM_SCALE_ONE_BARRIER_ID,
                thread_count=tmem_alloc_bar_count,
            )
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
                tile_l = bidz

            # fp4 packs its K-mode into the OMMA descriptor's 2-bit split field;
            # fp8 keeps the MX descriptor's 1-bit one. Both are built once,
            # outside the loops — the fields depend only on j (the scale id
            # within a word).
            if cutlass.const_expr(idesc_is_omma):
                idesc_by_j = [
                    cutlass.experimental.primitives.Tcgen05MxOmmaInstrDesc.build(
                        a_dtype=idesc_a_dtype,
                        b_dtype=idesc_b_dtype,
                        scale_format=sf_scale_format,
                        n_dim=mma_n_dim,
                        m_dim=mma_m_dim,
                        a_major=mma_a_major,
                        b_major=mma_b_major,
                        a_sf_id=j * sf_scales_per_inst,
                        b_sf_id=j * sf_scales_per_inst,
                        k_dim=mma_k_dim_mode,
                    )
                    for j in range(sf_insts_per_atom)
                ]
            else:
                idesc_by_j = [
                    cutlass.experimental.primitives.Tcgen05MxInstrDesc.build(
                        a_dtype=idesc_a_dtype,
                        b_dtype=idesc_b_dtype,
                        scale_format=sf_scale_format,
                        n_dim=mma_n_dim,
                        m_dim=mma_m_dim,
                        a_major=mma_a_major,
                        b_major=mma_b_major,
                        a_sf_id=j * sf_scales_per_inst,
                        b_sf_id=j * sf_scales_per_inst,
                        k_dim=mma_k_dim_mode,
                    )
                    for j in range(sf_insts_per_atom)
                ]

            sfa_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfa_col_bases[i]) for i in range(num_a_operands)]
            sfb_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfb_col_bases[j]) for j in range(num_b_operands)]
            s2t_shape, s2t_multicast = nvvm.S2TCopyMode.S2T_32x128b_WARPX4
            sfb_scale_ptrs = [nvvm.make_tmem_ptr(b, cutlass.Float32) for b in sfb_tmem_bases]
            # utccp destination per (MN-block, atom within the scale word). SFB
            # is atom-MAJOR across the N-blocks because ONE instruction walks
            # all of them; SFA is block-major because one instruction covers
            # exactly one 128-row block, so that word has to be contiguous.
            # Both collapse to the same addresses at a single block, and to
            # sm100's layout at word_atoms == 1.
            sfa_dst_ptrs = [
                [
                    [nvvm.make_tmem_ptr(sfa_tmem_bases[i] + m * registers_per_block + a * registers_per_atom, cutlass.Float32) for a in range(word_atoms)]
                    for m in range(mma_size_m)
                ]
                for i in range(num_a_operands)
            ]
            sfb_dst_ptrs = [
                [
                    [nvvm.make_tmem_ptr(sfb_tmem_bases[j] + (a * num_blocks_n + m) * registers_per_atom, cutlass.Float32) for a in range(word_atoms)]
                    for m in range(num_blocks_n)
                ]
                for j in range(num_b_operands)
            ]
            # Descriptor metadata and the SMEM allocation base are invariant
            # across persistent tiles. Only the encoded start address advances.
            desc_a_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_a_list[i],
                    leading_byte_offset=a_smem_desc_leading_byte_offset,
                    stride_byte_offset=a_smem_desc_stride_byte_offset,
                    layout=a_smem_swizzle,
                )
                for i in range(num_a_operands)
            ]
            desc_b_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_b_list[j],
                    leading_byte_offset=b_smem_desc_leading_byte_offset,
                    stride_byte_offset=b_smem_desc_stride_byte_offset,
                    layout=b_smem_swizzle,
                )
                for j in range(num_b_operands)
            ]
            desc_sfa_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_sfa_list[i],
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
                )
                for i in range(num_sfa_operands)
            ]
            desc_sfb_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_sfb_list[j],
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
                )
                for j in range(num_sfb_operands)
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

                if cutlass.const_expr(use_acc_overlap):
                    acc_base_col = base_col_id_root + (tile_iter % 2) * acc_stage_stride
                else:
                    acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                # One accumulator per (gemm, M block); M block mi sits
                # epi_cols_per_mma_m columns further into its GEMM's region and
                # reads SF word block mi (SF words are one per 128 rows).
                acc_tmem_ptrs = [
                    [
                        nvvm.make_tmem_ptr(
                            (base_row_id << 16) | (acc_base_col + g * acc_gemm_stride + mi * epi_cols_per_mma_m),
                            cutlass.Float32,
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

                    desc_a_bases = [desc_a_roots[i].advance_start_address(sA_bytes * stage) for i in range(num_a_operands)]
                    desc_b_bases = [desc_b_roots[j].advance_start_address(sB_bytes * stage) for j in range(num_b_operands)]
                    desc_sfa_bases = [desc_sfa_roots[i].advance_start_address(sfa_smem_bytes * stage) for i in range(num_sfa_operands)]
                    desc_sfb_bases = [desc_sfb_roots[j].advance_start_address(sfb_smem_bytes * stage) for j in range(num_sfb_operands)]

                    # One SF word per group of MMAs, refreshed right before they
                    # read it. A word spans word_atoms consecutive K-atoms in SMEM.
                    while not nvvm.mbarrier_try_wait_parity(
                        sf_full_mbar_ptr.subview(stage),
                        ab_full_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass

                    for sf_word in cutlass.range_constexpr(num_sf_atoms):
                        for _bj in cutlass.range_constexpr(num_sfb_operands):
                            for block_n in cutlass.range_constexpr(num_blocks_n):
                                for _a in cutlass.range_constexpr(word_atoms):
                                    if elect_one:
                                        nvvm.tcgen05_cp(
                                            s2t_shape,
                                            sfb_dst_ptrs[_bj][block_n][_a],
                                            desc_sfb_bases[_bj] + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * block_n),
                                            group=_CTA_GROUP,
                                            multicast=s2t_multicast,
                                        )
                        if cutlass.const_expr(sf_word == 0):
                            while not nvvm.mbarrier_try_wait_parity(
                                ab_full_mbar_ptr.subview(stage),
                                ab_full_phase_bit,
                                time_limit=10_000_000,
                            ):
                                pass
                        for mma_k_in_word in cutlass.range_constexpr(sf_insts_per_atom):
                            mma_k = sf_word * sf_insts_per_atom + mma_k_in_word
                            idesc_k = idesc_by_j[mma_k_in_word]
                            for gemm_i in cutlass.range_constexpr(num_gemms):
                                _ai = gemm_a_idx[gemm_i]
                                _bj = gemm_b_idx[gemm_i]
                                desc_a_k = desc_a_bases[_ai].advance_start_address(a_smem_k_step_bytes * mma_k)
                                desc_b = desc_b_bases[_bj].advance_start_address(b_smem_k_step_bytes * mma_k)
                                for mma_m in cutlass.range_constexpr(mma_size_m):
                                    if cutlass.const_expr(not fake_dequant_a and mma_k_in_word == 0 and _ai not in gemm_a_idx[:gemm_i]):
                                        for _a in cutlass.range_constexpr(word_atoms):
                                            if elect_one:
                                                nvvm.tcgen05_cp(
                                                    s2t_shape,
                                                    sfa_dst_ptrs[_ai][mma_m][_a],
                                                    desc_sfa_bases[_ai] + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * mma_m),
                                                    group=_CTA_GROUP,
                                                    multicast=s2t_multicast,
                                                )
                                    # The M sub-block offset is a whole SMEM swizzle atom, so
                                    # the descriptor's swizzle phase is preserved. B and its SF
                                    # are shared; A's SF word block follows the M block.
                                    desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mma_m)
                                    if elect_one:
                                        _tcgen05_mma_block_scale(
                                            mma_block_scale_kind,
                                            _CTA_GROUP,
                                            acc_tmem_ptrs[gemm_i][mma_m],
                                            desc_a,
                                            desc_b,
                                            idesc_k,
                                            enable_input_d=scale_d,
                                            scale_a=sfa_dst_ptrs[_ai][mma_m][0],
                                            scale_b=sfb_scale_ptrs[_bj],
                                            scale_vec_size=scale_vec_size,
                                            collector_op=_a_collector_op(gemm_i),
                                            b_collector_op=_b_collector_op(mma_m),
                                        )
                            # Every accumulator sees scale_d=False on exactly the first
                            # k_block of the tile, so the flip stays outside mma_m.
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
                if cutlass.const_expr(not use_acc_overlap):
                    nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            else:
                if cutlass.const_expr(use_acc_overlap):
                    while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                        pass
                nvvm.bar_warp_sync(0xFFFFFFFF)
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
                if cutlass.const_expr(not use_acc_overlap):
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

        if cutlass.const_expr(fake_dequant_a):
            for i in cutlass.range_constexpr(num_a_operands):
                _fill_scale_one((base_row_id << 16) | (base_col_id_root + sfa_col_bases[i]), sfa_tmem_cols)
        if cutlass.const_expr(fake_dequant_b):
            for j in cutlass.range_constexpr(num_b_operands):
                _fill_scale_one((base_row_id << 16) | (base_col_id_root + sfb_col_bases[j]), sfb_tmem_cols)
        if cutlass.const_expr(fake_dequant_a or fake_dequant_b):
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            nvvm.barrier_cta_sync(
                barrier_id=TMEM_SCALE_ONE_BARRIER_ID,
                thread_count=tmem_alloc_bar_count,
            )

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
        row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@

        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)

        while is_valid != 0:
            coord_m_tile = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            # @@EPILOGUE_DRAIN:BEGIN@@
            coord_n_c = tile_n * cgrp_tile_n_cur + n_rank * (cta_tile_mnk[1] * cta_group)

            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            # ==== MXFP8 NORM+ROPE+QUANT FUSION: tile classification and (ldg_early) the
            # cos/sin loads issued BEFORE the wait on the accumulator, so they land while
            # the MMA drains this tile instead of on the epilogue's critical path.
            # Everything here depends only on the tile coordinates (mma_size_m == 1,
            # checked at import: the row is coord_m_tile + tidx) and is warp-uniform.
            # Non-norm tiles point every lane at the tables' row 0 -- one L1-resident
            # line -- rather than skipping the loads, so the values exist unconditionally.
            # Unlike the FP8 fork there is NO scale vector to read: the block scales are
            # computed from the data, and the only runtime scalars (seq_len, v_sf_groups)
            # are kernel parameters.  Nothing but the class predicates and the cos/sin
            # words is live across the wait.
            _n0 = coord_n_c
            _is_q = _n0 < cutlass.Int32(_OFF_GATE)
            _is_k = (_n0 >= cutlass.Int32(_OFF_K)) & (_n0 < cutlass.Int32(_OFF_V))
            _is_v = _n0 >= cutlass.Int32(_OFF_V)
            _is_norm = _is_q | _is_k
            if cutlass.const_expr(_QK_NORM):
                _w_base = cutlass.Int64(arith.select(_is_q.ir_value(), w_q_norm.iterator.toint().ir_value(), w_k_norm.iterator.toint().ir_value()))
            # Tail rows (row >= m) exist in TMEM and are clipped by the TMA store; their
            # cos/sin READS must still be in bounds -> clamp.  `_row_valid` is what selects
            # their SF contribution away (Q/K: the SF word; V: the block-amax operand).
            _row64 = (coord_m_tile + tidx).to(cutlass.Int64)
            _row_valid = _row64 < m
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

            # ---- SF tile addressing, AFTER the wait (warp-uniform integer math on the tile coords, nothing
            # live across it).  Every row of this 128-row tile lies in ONE (b, s_tile): coord_m_tile % 128 == 0,
            # and at B > 1 the runner requires S % 128 == 0 (else a tile straddles sequences -> typed decline).
            _b_idx = coord_m_tile // seq_len
            _s_tile = (coord_m_tile - _b_idx * seq_len) >> _LOG2_SF_TILE_ROWS
            _n_tiles = (seq_len + cutlass.Int32(_SF_TILE_ROWS - 1)) >> _LOG2_SF_TILE_ROWS
            _cls_off = _sel_i32(_is_q, cutlass.Int32(0), _sel_i32(_is_k, cutlass.Int32(_OFF_K), cutlass.Int32(_OFF_V)))
            _head = (_n0 - _cls_off) >> _LOG2_D
            _heads = _sel_i32(_is_q, cutlass.Int32(_H_Q), cutlass.Int32(_H_KV))
            _sf_tile = ((_b_idx * _heads + _head) * _n_tiles + _s_tile).to(cutlass.Int64)  # (b*H + h)*n_tiles + s_tile
            # A CTA whose WHOLE 128-row tile lies past M -- the m_rank=1 CTA of the last 256-row cluster tile
            # whenever M % 256 is in (0, 128], e.g. B=1 S=128/384 or the ADMITTED B=3 S=128 -- has no (b, s_tile):
            # `_b_idx` decodes to b == B and every SF address above points PAST the blob (rowwise: the next
            # workspace slot; columnwise: onto the valid tiles' D-plane 1, racing their bytes).  Its data stores
            # are TMA-clipped; its SF stores must not execute at all.  The per-row SELECT-to-0x00 below stays
            # for PARTIALLY valid tiles (the SDPA reads whole tiles).  Warp-uniform -> a uniform branch.
            _tile_valid = coord_m_tile.to(cutlass.Int64) < m

            if cutlass.const_expr(use_acc_overlap):
                acc_buf_parity = tile_iter % 2
                acc_base_col = base_col_id_root + acc_buf_parity * acc_stage_stride
            else:
                acc_buf_parity = cutlass.Int32(0)
                acc_base_col = base_col_id_root + acc_stage * acc_region_cols

            for mi in cutlass.range_constexpr(mma_size_m):
                if cutlass.const_expr(use_acc_overlap and mma_size_m > 1):
                    _mi = mi + (1 - acc_buf_parity) * (mma_size_m - 1 - 2 * mi)
                else:
                    _mi = mi
                coord_m = coord_m_tile + _mi * epi_rows_per_mma_m
                mi_col_base = acc_base_col + _mi * epi_cols_per_mma_m
                tmem_col_addr_gemms = [(row_id_with_warp_offset << 16) | (mi_col_base + g * acc_gemm_stride) for g in range(num_gemms)]

                row = coord_m + tidx
                row_active = True

                # ==== MXFP8 NORM+ROPE+QUANT FUSION: BEGIN ==== (classification, the early loads and the SF tile index are above)
                if _is_norm:
                    # ---------------- Q / K: [norm] + RoPE + ROWWISE block quantize into q8 (desc 0) / k8 (desc 1) + sf_q / sf_k ----------------
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

                    # -- pass A: sum of squares of the accumulator over the whole row (8 subtiles), two
                    # subtiles per wait, each subtile as _PASS_A_SPLIT (= 4) 8-column tcgen05.ld's -- the
                    # FP8 fork's granule (its pass-A comment explains why: ptxas batches a fixed NUMBER
                    # of LDTMs ahead of the math, so the granule sets the live accumulator registers).
                    # (qk_norm=False: no pass A, no rsqrt -- the arm is RoPE + block quantize)
                    if cutlass.const_expr(_QK_NORM):
                        _sq_parts = []
                        for _s0 in cutlass.range_constexpr(0, subtile_cnt, 2):
                            _vs = []
                            for _s in cutlass.range_constexpr(_s0, min(_s0 + 2, subtile_cnt)):
                                _soff, _sw = epi_spans[_s]
                                for _hh in cutlass.range_constexpr(_PASS_A_SPLIT):
                                    _tm = cutlass.inttoptr(tmem_col_addr_gemms[0] + _soff + _hh * (_sw // _PASS_A_SPLIT), 6, mma_c_dtype)
                                    _vs.append(nvvm.tcgen05_ld(shape, _tm, num=_sw // _PASS_A_SPLIT))
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                            for _v in _vs:
                                _acc = _v[0] * _v[0]
                                for _i in cutlass.range_constexpr(1, epi_n // _PASS_A_SPLIT):
                                    _acc = _acc + _v[_i] * _v[_i]
                                _sq_parts.append(_acc)
                        _sq = _sq_parts[0]
                        for _p in cutlass.range_constexpr(1, len(_sq_parts)):
                            _sq = _sq + _sq_parts[_p]
                        # rstd of the row: the block-scale accumulator is already dequantized, so no alpha term.
                        _rstd = cute.math.rsqrt(_sq * cutlass.Float32(1.0 / _D) + cutlass.Float32(_EPS), fastmath=True)
                    # Column remap: q8 col = c for a Q tile, k8 col = c - OFF_K for a K tile.
                    _col_shift = cutlass.Int32(arith.select(_is_q.ir_value(), cutlass.Int32(0).ir_value(), cutlass.Int32(_OFF_K).ir_value()))
                    # SF: this thread's row of the (b, h, s_tile) tile -- byte (tidx%32)*16 + ((tidx//32)%4)*4 of atom subtile//4.
                    _sf_row_addr = (
                        _sel_i64(_is_q, sf_q_out.iterator.toint(), sf_k_out.iterator.toint())
                        + _sf_tile * cutlass.Int64(_SF_TILE_BYTES)
                        + ((tidx & cutlass.Int32(31)) * cutlass.Int32(16) + ((tidx >> 5) & cutlass.Int32(3)) * cutlass.Int32(4)).to(cutlass.Int64)
                    )
                    _sf_word = cutlass.Int32(0)

                    # -- pass B: [norm], rotate, block-quantize and store subtile by subtile --
                    _rot_elems = []  # per rope subtile: list of epi_n fp32
                    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                        subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                        if cutlass.const_expr(subtile_idx == 0):
                            # Load the whole rope group at once: partner of subtile s is s + ROPE_HALF_SUBTILES.
                            _rv = []
                            for _s in cutlass.range_constexpr(_ROPE_SUBTILES):
                                _soff, _sw = epi_spans[_s]
                                _tm = cutlass.inttoptr(tmem_col_addr_gemms[0] + _soff, 6, mma_c_dtype)
                                _rv.append(nvvm.tcgen05_ld(shape, _tm, num=_sw))
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                            # The `const*` diagnostics substitute a LIVE register for every deleted load:
                            # rstd under the norm, an accumulator value under RoPE-only.
                            _diag = _rstd if cutlass.const_expr(_QK_NORM) else _rv[0][0]
                            _y = []  # ROPE_DIM normed (or raw, qk_norm=False) fp32, flat
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
                                        _y.append((_rv[_s][_i] * _rstd) * _we[_i])
                                else:
                                    for _i in cutlass.range_constexpr(epi_n):
                                        _y.append(_rv[_s][_i])
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
                            _v = nvvm.tcgen05_ld(shape, _tm, num=subtile_w)
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
                                    _elems.append((_v[_i] * _rstd) * _we[_i])
                            else:
                                for _i in cutlass.range_constexpr(epi_n):
                                    _elems.append(_v[_i])  # RoPE-only passthrough dims: the raw accumulator

                        # -- ROWWISE block quantize: this 32-column subtile of this row IS one MXFP8 block --
                        _amax = abs_max_tree(_elems)
                        _rcp, _sf_byte = e8m0_from_amax(_amax)
                        _scaled = []
                        for _i in cutlass.range_constexpr(epi_n):
                            _scaled.append(_elems[_i] * _rcp)
                        # 4 subtile bytes = the 4 blocks of one 512-B atom = one contiguous b32 at this thread's row slot.
                        if cutlass.const_expr(subtile_idx % _SF_SUBTILES_PER_ATOM == 0):
                            _sf_word = _sf_byte
                        else:
                            _sf_word = _sf_word | (_sf_byte << (8 * (subtile_idx % _SF_SUBTILES_PER_ATOM)))
                        if cutlass.const_expr(subtile_idx % _SF_SUBTILES_PER_ATOM == _SF_SUBTILES_PER_ATOM - 1):
                            # A tail row (row >= m, TMA-clipped data) of a PARTIALLY valid tile writes SF 0x00 --
                            # SELECTED, never predicated off: the SDPA reads the whole 1024-B tile, and the residue
                            # in TMEM may be NaN (-> 0xFF).  A FULLY tail tile (`_tile_valid` false) has no tile
                            # slot in the blob and stores nothing.
                            if _tile_valid:
                                st_global(
                                    _sf_row_addr + cutlass.Int64((subtile_idx // _SF_SUBTILES_PER_ATOM) * _SF_ATOM_BYTES),
                                    _sel_i32(_row_valid, _sf_word, cutlass.Int32(0)),
                                    cutlass.Int32,
                                )

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
                        vec_out = _e4m3x32(_scaled)
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
                    # ---------------- V: COLUMNWISE block quantize into v8 (desc 2) + sf_v ----------------
                    # A warp's 32 lanes hold 32 CONSECUTIVE rows (tidx = row in tile) == one 32-token block, so
                    # the block amax of column i is a warp reduction; lane l then owns column col+l's SF byte.
                    _col_shift = cutlass.Int32(_OFF_V)  # v8 col = c - OFF_V
                    _zero = opaque_f32_zero()
                    # SF: D-plane-major.  plane (subtile // 4) sits v_sf_groups atoms away; inside the (b, h, s_tile)
                    # atom: lane*16 + (subtile % 4)*4 + warp (the warp IS the 32-token block index within the tile).
                    _v_plane_stride = v_sf_groups.to(cutlass.Int64) * cutlass.Int64(_SF_ATOM_BYTES)
                    _sf_lane_addr = (
                        sf_v_out.iterator.toint() + _sf_tile * cutlass.Int64(_SF_ATOM_BYTES) + (lane * cutlass.Int32(16) + warp_idx).to(cutlass.Int64)
                    )
                    # lane l reads its SF byte out of the 8 packed words: word l >> 2, byte l & 3 (a 3-level select mux)
                    _lb0 = (lane & cutlass.Int32(4)) != cutlass.Int32(0)
                    _lb1 = (lane & cutlass.Int32(8)) != cutlass.Int32(0)
                    _lb2 = (lane & cutlass.Int32(16)) != cutlass.Int32(0)
                    _lshift = (lane & cutlass.Int32(3)) << 3
                    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                        subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                        _tm = cutlass.inttoptr(tmem_col_addr_gemms[0] + subtile_col_offset, 6, mma_c_dtype)
                        _v = nvvm.tcgen05_ld(shape, _tm, num=subtile_w)
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        # Tail rows contribute ZERO to the column block amax (the oracle's zero padding) -- a SELECT,
                        # never a multiply: the residue may be NaN.  Their data store is TMA-clipped anyway.
                        _vals = []
                        for _i in cutlass.range_constexpr(epi_n):
                            _vals.append(_sel_f32(_row_valid, _v[_i], _zero))
                        # 32 warp-wide abs-max reductions (redux.sync.max.abs.f32): column i's block amax, in every lane.
                        _amaxes = []
                        for _i in cutlass.range_constexpr(epi_n):
                            _amaxes.append(cute.arch.warp_redux_sync(_vals[_i], "fmax", abs=True))
                        # One cvt per two columns; the packed 16-bit pairs are the SF bytes (byte0 = column 2p).
                        _rcps = []
                        _pairs = []
                        for _p in cutlass.range_constexpr(epi_n // 2):
                            _r0, _r1, _pk = e8m0_pair(_amaxes[2 * _p], _amaxes[2 * _p + 1])
                            _rcps.append(_r0)
                            _rcps.append(_r1)
                            _pairs.append(_pk)
                        _scaled = []
                        for _i in cutlass.range_constexpr(epi_n):
                            _scaled.append(_vals[_i] * _rcps[_i])
                        _words = []
                        for _q in cutlass.range_constexpr(epi_n // 4):
                            _words.append(_pairs[2 * _q] | (_pairs[2 * _q + 1] << 16))
                        _m01 = _sel_i32(_lb0, _words[1], _words[0])
                        _m23 = _sel_i32(_lb0, _words[3], _words[2])
                        _m45 = _sel_i32(_lb0, _words[5], _words[4])
                        _m67 = _sel_i32(_lb0, _words[7], _words[6])
                        _m03 = _sel_i32(_lb1, _m23, _m01)
                        _m47 = _sel_i32(_lb1, _m67, _m45)
                        _mw = _sel_i32(_lb2, _m47, _m03)
                        _sf_byte_v = (_mw >> _lshift) & cutlass.Int32(0xFF)
                        # A FULLY tail tile stores nothing (see `_tile_valid`): its plane-0 address IS a valid tile's
                        # plane-1 slot.  Partially valid tiles store every byte (tail rows entered the amax as zero).
                        if _tile_valid:
                            _st_global_b8(
                                _sf_lane_addr
                                + cutlass.Int64(subtile_idx // _SF_SUBTILES_PER_PLANE) * _v_plane_stride
                                + cutlass.Int64((subtile_idx % _SF_SUBTILES_PER_PLANE) * _SF_ATOM_BLOCKS),
                                _sf_byte_v,
                            )

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
                        vec_out = _e4m3x32(_scaled)
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
                    # ---------------- GATE: the rendering's own epilogue, bf16(acc) into gate16 (desc 3) ----------------
                    # (verbatim from the rendering: the descriptor is gate16's, the column is shifted by OFF_GATE.)
                    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                        subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                        c_rmem_vecs = []
                        for g in cutlass.range_constexpr(num_gemms):
                            subtile_tmem_addr = tmem_col_addr_gemms[g] + subtile_col_offset
                            tmem = cutlass.inttoptr(subtile_tmem_addr, 6, mma_c_dtype)
                            _cv = nvvm.tcgen05_ld(shape, tmem, num=subtile_w)
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

                        _r_mm = (vec_f32).to(cutlass.BFloat16)
                        vec_out = (_r_mm).to(cutlass.BFloat16)

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

                # ==== MXFP8 NORM+ROPE+QUANT FUSION: END ====
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

        if cutlass.const_expr(use_acc_overlap):
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if elect_one:
                nvvm.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        if warp_idx == 0:
            nvvm.cp_async_bulk_wait_group(0, read=True)

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)


frost_sm100_block_scale_matmul_128x256x128_128x256x32_cluster2x1_2ctamma.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    a_0: cute.Tensor,
    b_0: cute.Tensor,
    sfa_0: cute.Tensor,
    sfb_0: cute.Tensor,
    c_q8: cute.Tensor,  # e4m3 [M, N_Q]  (permuted to (N, M, L) by the runner) -- the rendering's c_0 slot
    # ---- MXFP8 NORM+ROPE+QUANT FUSION: appended ----
    c_k8: cute.Tensor,  # e4m3 [M, N_KV]
    c_v8: cute.Tensor,  # e4m3 [M, N_KV]
    c_gate: cute.Tensor,  # bf16 [M, N_GATE]
    w_q_norm: Optional[cute.Tensor],  # None iff PARAMS.qk_norm is False
    w_k_norm: Optional[cute.Tensor],
    cos_tab: cute.Tensor,
    sin_tab: cute.Tensor,
    sf_q_out: cute.Tensor,  # uint8 flat, B*H_Q*ceil(S/128)*1024
    sf_k_out: cute.Tensor,  # uint8 flat, B*H_KV*ceil(S/128)*1024
    sf_v_out: cute.Tensor,  # uint8 flat, B*H_KV*ceil(S/128)*1024
    seq_len: cutlass.Int32,
    v_sf_groups: cutlass.Int32,
    stream: _cuda.CUstream,
) -> None:
    _a_operands = [a_0]
    _b_operands = [b_0]
    _sfa_operands = [sfa_0]
    _sfb_operands = [sfb_0]

    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    batch = problem_size[3]
    a_stride_m = problem_size[4]
    a_stride_k = problem_size[5]
    a_stride_l = problem_size[6]
    b_stride_n = problem_size[7]
    b_stride_k = problem_size[8]
    b_stride_l = problem_size[9]

    out_stride_m_0 = problem_size[10]
    out_stride_n_0 = problem_size[11]
    out_stride_l_0 = problem_size[12]
    # ---- MXFP8 NORM+ROPE+QUANT FUSION: k8, v8 and gate16 stride triples follow q8's ----
    out_stride_m_1 = problem_size[13]
    out_stride_n_1 = problem_size[14]
    out_stride_l_1 = problem_size[15]
    out_stride_m_2 = problem_size[16]
    out_stride_n_2 = problem_size[17]
    out_stride_l_2 = problem_size[18]
    out_stride_m_3 = problem_size[19]
    out_stride_n_3 = problem_size[20]
    out_stride_l_3 = problem_size[21]

    if cutlass.const_expr(matmul_a_batch == 1):
        a_batch = 1
    else:
        a_batch = batch
    if cutlass.const_expr(matmul_b_batch == 1):
        b_batch = 1
    else:
        b_batch = batch

    rest_k = ((k_sym // block_size) + 3) // 4
    rest_m = (m + 127) // 128
    rest_n = (n + 127) // 128
    tma_a_desc_list = []
    tma_sfa_desc_list = []
    for _a_op in _a_operands:
        if cutlass.const_expr(a_is_m_major):
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=a_tma_desc_dtype,
                    global_dims=[m, k_sym, a_batch],
                    global_strides=[
                        a_stride_k * a_dtype.width // 128,
                        a_stride_l * a_dtype.width // 128,
                    ],
                    box_dims=[a_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=a_tma_swizzle,
                    tma_format=a_tma_format,
                )
            )
        else:
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=a_tma_desc_dtype,
                    global_dims=[k_sym, m, a_batch],
                    global_strides=[
                        a_stride_m * a_dtype.width // 128,
                        a_stride_l * a_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], a_tma_box_m, 1],
                    swizzle=a_tma_swizzle,
                    tma_format=a_tma_format,
                )
            )
    for _sfa_op in _sfa_operands:
        sfa_fp16_tensor = cute.make_tensor(
            cute.recast_ptr(_sfa_op.iterator, dtype=cutlass.Float16),
            cute.make_layout(
                (256, rest_k, rest_m, batch),
                stride=(
                    1,
                    256,
                    cute.assume(256 * rest_k, 8),
                    cute.assume(256 * rest_k * rest_m, 8),
                ),
            ),
        )
        tma_sfa_desc_list.append(
            _tma.create_tensor_map_tiled_from_view(
                sfa_fp16_tensor,
                dtype=cutlass.Uint16,
                box_dims=(256, sf_tma_box_k, sfa_tma_box_mn, 1),
                stride_order=(0, 1, 2, 3),
                swizzle=_tma.TensorMapSwizzle.none,
            )
        )
    tma_b_desc_list = []
    tma_sfb_desc_list = []
    for _b_op in _b_operands:
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=b_tma_desc_dtype,
                    global_dims=[n, k_sym, b_batch],
                    global_strides=[
                        b_stride_k * b_dtype.width // 128,
                        b_stride_l * b_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=b_tma_swizzle,
                    tma_format=b_tma_format,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=b_tma_desc_dtype,
                    global_dims=[k_sym, n, b_batch],
                    global_strides=[
                        b_stride_n * b_dtype.width // 128,
                        b_stride_l * b_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
                    swizzle=b_tma_swizzle,
                    tma_format=b_tma_format,
                )
            )
    for _sfb_op in _sfb_operands:
        sfb_fp16_tensor = cute.make_tensor(
            cute.recast_ptr(_sfb_op.iterator, dtype=cutlass.Float16),
            cute.make_layout(
                (256, rest_k, rest_n, batch),
                stride=(
                    1,
                    256,
                    cute.assume(256 * rest_k, 8),
                    cute.assume(256 * rest_k * rest_n, 8),
                ),
            ),
        )
        tma_sfb_desc_list.append(
            _tma.create_tensor_map_tiled_from_view(
                sfb_fp16_tensor,
                dtype=cutlass.Uint16,
                box_dims=(256, sf_tma_box_k, sfb_tma_box_mn, 1),
                stride_order=(0, 1, 2, 3),
                swizzle=_tma.TensorMapSwizzle.none,
            )
        )

    # ---- MXFP8 NORM+ROPE+QUANT FUSION: FOUR TMA-store descriptors, one per compact output ----
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
    launch = frost_sm100_block_scale_matmul_128x256x128_128x256x32_cluster2x1_2ctamma(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        tma_a_desc_list[0],
        tma_b_desc_list[0],
        tma_sfa_desc_list[0],
        tma_sfb_desc_list[0],
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
        sf_q_out,
        sf_k_out,
        sf_v_out,
        seq_len,
        v_sf_groups,
    )
    # Mixed CGA: `cluster` is the preferred (wide) shape and
    # `fallback_cluster` the regular one the device groups blocks into when a
    # preferred cluster does not fit. The grid is already a multiple of the
    # preferred shape, which the driver requires.
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
    a_stride_elems = 128 // a_dtype.width
    b_stride_elems = 128 // b_dtype.width
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    # K tails are supported: the K loop is ceil_div and the TMA descriptor's global K
    # extent makes a partial box HW zero-filled. The only real K rule is the 16-byte
    # TMA contiguous-extent one, already gated by _tma_alignment_reject.
    sym_k = cute.sym_int64()
    # Packed K extent: same reasoning as sym_k -- no CTA-tile multiple is required.
    sym_akp = cute.sym_int64()
    sym_bkp = cute.sym_int64()
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
            a_fake_dtype,
            (sym_m, sym_akp, sym_a_l),
            stride_order=(0, 1, 2) if a_is_m_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            b_fake_dtype,
            (sym_n, sym_bkp, sym_b_l),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    # SF reaches the kernel as a base pointer only; the host rebuilds the
    # F8_128x4 view from problem_size, so no SF mode carries a layout contract.
    def _make_fake_sfa():
        return cute.runtime.make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    def _make_fake_sfb():
        return cute.runtime.make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    def _make_fake_c(_dt, _div, _mm):
        return make_fake_compact_tensor(
            _dt,
            (sym_m, sym_n // _div, sym_l),
            stride_order=(0, 1, 2) if _mm else (1, 0, 2),
            assumed_align=16,
        )

    # ---- MXFP8 NORM+ROPE+QUANT FUSION: four compact outputs (their widths are NOT the weight's
    # n), the bf16 tables, the three uint8 SF blobs and the two runtime Int32 scalars.  Static
    # [D] weights: the artifact then refuses a wrong-length weight at the call boundary; cos/sin
    # are [T, x] over the symbolic M.  (`_make_fake_c` is the rendering's; unused here.)
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

    def _make_fake_sf_out():
        # A flat uint8 blob (B*H*ceil(S/128)*1024 bytes, F8_128x4 order); the kernel indexes it from its base.
        return cute.runtime.make_fake_tensor(cutlass.Uint8, (cute.sym_int64(),), stride=(1,), assumed_align=16)

    fake_sf_q = _make_fake_sf_out()
    fake_sf_k = _make_fake_sf_out()
    fake_sf_v = _make_fake_sf_out()

    fake_a_0 = _make_fake_a()
    fake_b_0 = _make_fake_b()
    fake_sfa_0 = _make_fake_sfa()
    fake_sfb_0 = _make_fake_sfb()

    # The operand's unit stride (m/n when MN-major, k when K-major) never reaches TMA, so it carries no 16B contract.
    sym_a_stride_m = cute.sym_int64() if a_is_m_major else cute.sym_int64(divisibility=a_stride_elems)
    sym_a_stride_k = cute.sym_int64(divisibility=a_stride_elems) if a_is_m_major else cute.sym_int64()
    sym_a_stride_l = cute.sym_int64(divisibility=a_stride_elems)
    sym_b_stride_n = cute.sym_int64() if b_is_n_major else cute.sym_int64(divisibility=b_stride_elems)
    sym_b_stride_k = cute.sym_int64(divisibility=b_stride_elems) if b_is_n_major else cute.sym_int64()
    sym_b_stride_l = cute.sym_int64(divisibility=b_stride_elems)

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

    problem_size = (
        sym_m,
        sym_n,
        sym_k,
        sym_l,
        sym_a_stride_m,
        sym_a_stride_k,
        sym_a_stride_l,
        sym_b_stride_n,
        sym_b_stride_k,
        sym_b_stride_l,
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
        fake_sfa_0,
        fake_sfb_0,
        fake_c_q8,
        fake_c_k8,
        fake_c_v8,
        fake_c_gate,
        fake_w_q,
        fake_w_k,
        fake_cos,
        fake_sin,
        fake_sf_q,
        fake_sf_k,
        fake_sf_v,
        cutlass.Int32(0),  # seq_len     ) runtime; the zeros pin the TYPE only
        cutlass.Int32(0),  # v_sf_groups )
        stream=_fake_stream,
        options=keep_sass_options(frost_compile_options, PARAMS.keep_sass),
    )
