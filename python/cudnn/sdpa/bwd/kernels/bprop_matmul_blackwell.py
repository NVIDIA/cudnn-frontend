# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100 batched GEMM with a 2-D ``(batch, head)`` batch — SDPA-backward stage 3.

Computes dV = S^T.dO, dK = dS^T.Q and dQ = dS.K.  On the bf16 / fp16 rows all
three are plain batched GEMMs with no epilogue.  The sm107 d256 fp8 chain renders
the FP8 ARM (``MatmulTemplateParams.dtype_qkv = DTYPE_E4M3``): e4m3 A (the dS
workspace, ``dS_q = e4m3(dS * scale_dP)``) and B (the Q / K payload) through
Rubin's dense-FP8 K64 MMA (``256x256x64``, idesc ``k_dim=1``, ``F8F6F4``) into
an fp32 accumulator, with a DESCALE (``acc * descale_dP * descale_{q|k}`` -> bf16,
the GQA fold's per-Q-head partial) or QUANT (``+ amax fold, * scale_{dQ|dK} ->
the gradient dtype``) epilogue -- ``config_sm100.MatmulTemplateParams`` carries
the story.  Everything fp8 is a ``const_expr`` off ``PARAMS``; the bf16 / fp16
renderings are byte-identical to what they were before the arm existed.

WHY THIS EXISTS -- the one thing the generic GEMM cannot express
---------------------------------------------------------------
``gemm/frost/sm100/kernel_templates/sm100_matmul.py`` carries a single batch axis
``l`` with ONE uniform stride.  The SDPA operands are BSHD ``[B, S, H, D]``, so
the batch element is the PAIR ``(b, h)`` at offset ``b*(S*H*D) + h*D`` -- a
two-level stride that no single uniform stride can express.  Flattening it
host-side would need a copy of every operand, per chunk, against a workspace
measured in GiB.

So the operands stay exactly as the user laid them out and the TMA descriptors
become **4-D** ``[k, m, h, b]`` (``cuTensorMapEncodeTiled`` allows 5), with
``h`` and ``b`` as separate coordinates.  The CLC scheduler still rasterizes a
single flat ``l`` -- ``_decode_bh`` splits it only where a TMA coordinate is
formed, which is the whole change.

KEEP IN SYNC WITH ``gemm/frost/sm100/kernel_templates/sm100_matmul.py``
-----------------------------------------------------------------------
This file is a FORK of that template, taken from its rendered dense-bf16
expansion (config ``sm100_128x256x128_128x256x32_cluster2x1_2ctamma``, no
epilogue fusion, TMA-store epilogue).  The mainloop, the CLC scheduler, the
TMEM/accumulator pipeline and the epilogue are otherwise unchanged.

**Any correctness fix or performance improvement to either file should be
applied to BOTH.**  The generic template carries the same note.  The diff
against it is deliberately narrow, so a `diff` against a fresh rendering of
that config is the intended way to review a change:
  * ``_decode_bh`` and its four call sites;
  * ``h``/``b`` in place of ``l`` in every TMA coordinate tuple -- B's head through
    ``_b_head`` (``h // b_head_group``: the K head that ``b_head_group`` consecutive
    Q heads share under GQA; the identity at the default 1) and its descriptor's head
    extent ``n_head // b_head_group``;
  * 4-D descriptors and one extra stride per operand;
  * ``problem_size`` carrying ``(n_head, n_batch)`` and 4 strides per operand.

Everything else in this file is generated output; do not hand-tune it in place
without making the same change upstream.

THE BLOCK-SCALE ARM (``MatmulTemplateParams.block_scale``, the MXFP8 d256 backward's dK / dQ)
--------------------------------------------------------------------------------------------
A = the block-scaled e4m3 dS workspace, B = the columnwise-quantized e4m3 Q / K payload; every 32-element K block of
either carries an E8M0 scale byte, and the K64 ``tcgen05.mma.block_scale`` (kind MXF8F6F4, BLOCK32, idesc ``k_dim=1``)
dequantizes both IN the MMA, so the fp32 accumulator is the TRUE-unit gradient and the epilogue is EPI_NONE.  Workspace
contract (the (b, h) pair decoded exactly like the operands', ``_decode_bh``)::

    dK: out[kv, d] = sum_q  dS[kv, q] Q[q, d]    A = ds_dk  [B, H, S_kv, S_q] K-major (q contiguous)   SFA = sf_ds_dk [B, H, S_kv/128, S_q/128, 512]
    dQ: out[q, d]  = sum_kv dS[kv, q] K[kv, d]   A = ds_dq  [B, H, S_kv, S_q] read as [q, kv], M-major SFA = sf_ds_dq [B, H, S_q/128, S_kv/128, 512]
    B  = q_T / k_T [B, S_K, H, D] (D contiguous: N-major); SFB = its columnwise F8_128x4 SF, D-PLANE-major: [D/128 planes][B*H*S_K/128 tiles][512 B]

THD leg of the arm (``block_scale`` + ``thd_varlen``; the MXFP8 d256 backward over packed tokens): A and its atoms ride the
kv-BLOCKED workspace exactly like the plain THD arm's A, so both SFA atom indices take the sequence's row offset with the
operand coordinate (dK: M tile ``(coord_m + row_off[b]) / 128``; dQ: K tile ``(coord_k + row_off[b]) / 128``); B's columnwise
SF is PACKED per sequence in whole 128-token tiles -- sequence b's tiles start at ``cu_sf[b] = SUM_{i<b} ceil(s_i / 128)``,
NOT at ``cu[b] / 128`` -- and both D planes of a (head, tile) sit together (plane stride one atom, tile stride the
``planes x 512``-byte slab: ``config_sm100.stage3_thd_sfb_layout``, the view ``_sf_planes_view_thd`` builds; the dense
D-plane-major view would fetch plane 1 from the wrong place by an S-dependent offset, rules/mma-tma-matrix.md s7).  The TMA
warp reads the sequence's tile prefix once per tile from the appended ``sf_meta_t`` operand (``[cu_sf_q(B+1) | cu_sf_k(B+1)]``,
``config_sm100.STAGE3_THD_SF_*``: ``cu_sf_k[b]`` when the reduction runs over kv tokens, ``cu_sf_q[b]`` over q tokens) and the
SFB coordinate becomes ``(0, plane, cu_sf[b] + k_tile, h, 0)``.  Every line of it is ``const_expr``-folded on ``_THD_MM``
inside the arm's guards: the dense block-scale rendering is byte-identical.  COVERAGE: the SF-prefix contract and the record
validation host-side (``test_sdpa_bwd_stage3_block_scale_sm107.py``), the device numerics through the sm107 MXFP8 d256 backward
row, which renders the leg under its block-scaled dS policy over packed sequences (``test_sdpa_bwd_thd_mxfp8_sm107.py``: packed
cells under both dS policies against the bf16 chain, GQA, one-sided empty sequences, the capacity tails, rebinding and replay).

The fp8 arm's EPI_QUANT amax fold under THD is gated PER ROW (``row < _thd_c_len``, with the tile's band live and its
reduction non-empty): a (head, sequence) group walks every M tile of the ENVELOPE grid, so a shorter sequence's spare tiles
compute foreign products the C descriptor clips -- and the per-tile store predicates do not exclude them (the epilogue
comment at the fold spells the two cases out).  Dense renderings fold the gate out.

One F8_128x4 atom (128 rows x 4 K-block scales = 512 B, byte ``(r % 32) * 16 + (r // 32) * 4 + c``) is exactly one 128-B
K stage of one 128-row block, so per K stage the CTA needs its ONE SFA atom and BOTH SFB plane atoms (its accumulator
spans the pair's full N = 256).

SMEM buffer table (declaration order == ``_smem_layout_bytes``; the operand rings and the epilogue staging keep the
fork's rows and swizzles):

    buffer     dtype x elems            writer (how)                          reader (how)                                  per-lane stride  swizzle + WHY
    smem_sfa   uint8 x 6 x 512 @ 2 KiB  TMA, one atom per stage (box 512 B)   UTCCP 32x128b WARPX4 via a tcgen05 SMEM desc  -- (no lane)     NONE: the atom's byte order IS the copy's order (leading 16 B, stride 128 B)
    smem_sfb   uint8 x 6 x 1024 @ 5 KiB TMA, 2 plane atoms per stage (1 KiB)  2 UTCCPs per stage, one per atom (+512 B)     -- (no lane)     NONE, as above
    smem_a/b   e4m3 x 6 x 16 KiB        TMA (s128b)                           MMA descriptor (SWIZZLE_128B)                 --               128 B: the descriptor's (job 1), unchanged
    smem_d     bf16 x 2 x 128 x 64      epilogue lanes `store_swizzled`       TMA store                                     128 B            Swizzle(3, 4, 3) + s128b, unchanged

The SF rings are declared FIRST so every tcgen05 descriptor root (the UTCCP sources included) stays below the 256 KiB
version-0 line: roots at 2 / 5 / 11 / 107 KiB, total 235 KiB (the Rubin oversized carveout; ``_smem_layout_bytes`` models
it and the import-time guard raises past the line).

TMEM (one 576-column EXCLUSIVE allocation, alloc and dealloc both pass ``num_tmem_alloc_cols``): columns [0, 512) the two
256-column accumulator stages as before; [512, 516) SFA -- one 4-column word, ``scale_a``; [516, 524) SFB -- the two plane
atoms at +0 / +4, ONE ``scale_b`` span the 256-wide instruction reads whole; [524, 576) free.  Refreshed per K stage in the
tcgen05 pipeline right before the two MMAs that read them (``a_sf_id = b_sf_id = 0`` for k-block 0, ``2`` for k-block 1).

Barrier table delta (every other row of the fork is unchanged; lane arithmetic per row)::

    ab_full[s]   producer TMA_LOAD   init 1   ONE elected lane of the LEADER's TMA warp arms expect_tx per stage: bytes =
                 (A + B + SFA + SFB) x 2 -- both CTAs' operand AND scale loads land on the leader's mbarrier (cta_group::2
                 tensor TMA); each CTA's TMA warp issues its own 4 loads (A, B, SFA, SFB) elect-gated: SUM(issuing lanes) = 1
                 arrive + tx bytes == init 1 + expect_tx.  Consumer: the leader's MMA warp (unchanged wait).
    ab_empty[s]  producer MMA_COMMIT init ab_empty_count  ONE elected lane's tcgen05.commit after the stage's MMAs -- the
                 commit tracks the two UTCCPs' SMEM reads as well, so the SF stage is free when the operand stage is.
    acc_*, clc_*, tmem_dealloc                            unchanged.

Neither the THD leg nor the amax row gate adds a barrier or an SMEM buffer: the SF tile prefix is one GMEM word read by the
TMA warp per tile (a register), the gate a per-thread select on a register value; the tables above are the whole story.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import cutlass.experimental.primitives as nvvm
from cudnn.gemm.frost.tile_helpers import (
    epi_subtile_spans as _epi_subtile_spans,
    l2_swizzle_tile as _l2_swizzle_tile,
    tcgen05_alloc as _tcgen05_alloc,
    tcgen05_dealloc as _tcgen05_dealloc,
    tcgen05_mma as _tcgen05_mma,
    tcgen05_mma_block_scale as _tcgen05_mma_block_scale,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass._mlir_helpers.vector as _cvec
from cutlass._mlir.dialects import arith
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as _cuda
from cutlass.cute.arch import clc as cute_clc

from cutlass.base_dsl.typing import Pointer

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16, DTYPE_FP32
from cudnn.frost.tile_dsl.pointwise import abs_max_tree, fmax_f32, opaque_f32_zero
from cudnn.frost.tile_dsl.sf_layout import SF_ATOM_BYTES, SF_ATOM_COLS, SF_ATOM_ROWS
from cudnn.frost.tile_dsl.thd import THD_SETUP_THREADS, TENSOR_MAP_QWORDS, emit_clamped_desc, emit_seq_descs
from cudnn.sdpa.bwd.config_sm100 import (
    CAUSAL_K_HI,
    CAUSAL_K_LO,
    CAUSAL_K_NONE,
    EPI_DESCALE,
    EPI_NONE,
    EPI_QUANT,
    STAGE3_BLOCK_SCALE_TMEM_COLS,
    STAGE3_MX_BLOCK,
    STAGE3_THD_SF_CU_K_OFF,
    STAGE3_THD_SF_CU_Q_OFF,
    MatmulTemplateParams,
    matmul_out_dtype,
    stage3_thd_sfb_layout,
    validate_matmul_params,
)

PARAMS = globals().get("FROST_TEMPLATE_PARAMS", MatmulTemplateParams())
validate_matmul_params(PARAMS)

# A/B io dtype.  BF16 and FP16 are both 2 B/element and both take the
# Tcgen05MMAKind.F16 path, so between them this is a token swap: nothing
# byte-sized below changes.  E4M3 is the fp8 ARM: 1 B/element, the K64 dense-FP8
# MMA (`mma_inst_shape_mnk` / `mma_k_dim` / `mma_kind` below), and a descale or
# quantize epilogue (`epi_mode`).  It must agree with the stage-2 template's dS
# workspace dtype -- stage 3 reads the S/dS workspace stage 2 wrote.
_DSL_DTYPES = {DTYPE_BF16: cutlass.BFloat16, DTYPE_FP16: cutlass.Float16, DTYPE_E4M3: cutlass.Float8E4M3FN}
_IO_DTYPE = _DSL_DTYPES[int(PARAMS.dtype_qkv)]
_IS_FP8 = int(PARAMS.dtype_qkv) == DTYPE_E4M3
_AB_BPE = _IO_DTYPE.width // 8
# D dtype: the io dtype on the bf16 / fp16 rows; on the fp8 arm the gradient dtype (QUANT) or the fp32 per-Q-head true-unit
# partial (DESCALE on the per-tensor arm, EPI_NONE on the block-scale arm whose MMA already dequantized: the GQA fold sums it in
# fp32 and rounds ONCE -- `validate_matmul_params` admits FP32 for exactly those two).  FP32 is an OUTPUT-only code: it never
# reaches `_IO_DTYPE`.
_OUT_DTYPE = {**_DSL_DTYPES, DTYPE_FP32: cutlass.Float32}[matmul_out_dtype(PARAMS)]
_CD_BPE = _OUT_DTYPE.width // 8
epi_mode = int(getattr(PARAMS, "epi_mode", EPI_NONE))
# The BLOCK-SCALE (MXFP8) arm of the fp8 rendering: e4m3 A / B whose 32-element K blocks each carry an E8M0 scale byte, dequantized
# IN the MMA (`tcgen05.mma.block_scale`, kind MXF8F6F4, BLOCK32) so the accumulator is TRUE-unit and the epilogue is EPI_NONE.
# Everything block-scale is a `const_expr` off this flag; at False the module renders byte-for-byte what it did before the arm
# existed (the SF geometry below folds to zeros / unused constants, the SF rings, descriptors and copies are not traced).
_BLOCK_SCALE = bool(getattr(PARAMS, "block_scale", False))
# The K64 dense-FP8 MMA form (idesc k_dim=1, K=64 e4m3 per instruction) is RUBIN's: on Blackwell it is silently WRONG
# (uniform ~0.4 |O-ref|, no crash -- rules/mma-tma-matrix.md S1).  This module has no arch at load time, so the guard is
# the caller's: only the sm107 adapter renders the fp8 arm (`api_dsl_sm107._stage3_params`) and
# `kernels/sm107/prepared_host._check_target` (107 <= sm <= 119) is the runtime backstop; `validate_matmul_params` pins
# the arm to the (256, 256) row whose upstream Rubin rendering the constants below are lifted from.


class _TileRow(NamedTuple):
    """One row of the tile-constants table: exactly the constants that DIFFER between two upstream renderings of
    this file's config family (``sm100_matmul`` at ``CONFIG_sm100_{cta_m}x256x128_128x256x32_cluster{m}x{n}_2ctamma``,
    2-CTA MMA 256x256x16, bf16 / fp16, TMA-store epilogue).  Every value is lifted VERBATIM from the upstream
    renderer's output for the named config (``gemm/frost/sm100/compiler._render_tile_constants``) -- do not
    hand-derive; the fork pins ``epi_n = epi_row_elems`` -- ONE staging row of D per lane: 64 elements (128 B at a 2-byte D,
    64 B at e4m3 out) or 32 at the fp32 DESCALE partial (128 B), derived from ``_CD_BPE`` below rather than read from a
    table column --, 512 non-exclusive TMEM columns and ``fallback_cluster_shape_mnk = None`` on every row, so those stay
    module constants below.

    A ``NamedTuple``, not a ``@dataclass``: this module runs under ``frost.template_loader`` BEFORE it is registered in
    ``sys.modules``, which a module-scope dataclass decorator needs (see ``MatmulTemplateParams``).
    """

    config: str
    cgrp_tile_mn: tuple  # the CLUSTER's (M, N) output tile; `_host` covers the problem in these (K per stage: `_K_STAGE_ELEMS`)
    cta_tile_mn: tuple  # per-CTA (A rows, B cols) per stage
    cluster_shape_mnk: tuple
    ab_stages: int  # operand ring depth at the 227 KiB opt-in budget: stages x (A + B per CTA) + 32 KiB staging
    multicast_a: bool  # A is TMA-multicast along the cluster's N only when cluster_n > 1
    a_mcast_k_major: tuple  # (a_mcast_slices, ab_empty_full_mask) for a K-major A
    a_mcast_m_major: tuple  # ... for an M-major A (the two differ only under a multicast, i.e. at cluster_n > 1)
    mma_size_m: int  # MMA-M blocks (128 rows each per CTA) per CTA tile
    acc_stages: int  # TMEM accumulator stages of `mma_size_m x 256` columns each (2 x 256 fits the 512 columns)
    mixed_a_pattern_pref: int  # the A multicast bit pattern of the preferred cluster (1 = self only)


# Keyed by ``MatmulTemplateParams.cgrp_tile_mn`` (its docstring carries the story).  The N tile is the one that
# matters at the head dim: `_host` sizes the grid as ``ceil(n / N)`` cluster tiles, so an N tile wider than the
# head dim spends whole CTAs on columns that do not exist (TMA-OOB zero loads, clipped stores).
_TILE_ROWS = {
    # The SM100 d512 chain's rendering (swapped from cluster2x1 on request: measured faster BOTH ways at B=1 H=128
    # S=8192 d=512 bf16, +3.5 % no_mask / +7.8 % causal -- `_causal_k_range`).  N = 512 fills its 512-wide tile.
    (512, 512): _TileRow(
        config="CONFIG_sm100_256x256x128_128x256x32_cluster2x2_2ctamma",
        cgrp_tile_mn=(512, 512),
        cta_tile_mn=(256, 128),
        cluster_shape_mnk=(2, 2, 1),
        ab_stages=4,
        multicast_a=True,
        # Under a cluster with cga_n > 1 the A operand's MULTICAST also follows its major -- a K-major A is split
        # across the cluster's N columns (2 slices) while an M-major A is broadcast whole.  Invisible at cluster2x1,
        # where both are (1, False), and why this pair cannot be reused across cluster shapes unchanged.
        a_mcast_k_major=(2, False),
        a_mcast_m_major=(1, True),
        mma_size_m=2,
        acc_stages=1,
        mixed_a_pattern_pref=5,
    ),
    # The d = 256 rendering (the sm107 d256 chain): upstream's DEFAULT_CONFIG and this fork's original config.  One
    # 256-row x 256-col cluster tile per pair, so a head dim of 256 is covered with NO padding; the 256-column
    # accumulator is double-buffered in the same 512 TMEM columns (the epilogue of tile i overlaps the mainloop of
    # tile i+1); 6 x 32 KiB of operands in flight = the (512, 512) row's 4 x 48 KiB.  Its cluster M tile equals the
    # stage-2 kv write block, which is what makes the causal K-trim TIGHT (`_causal_k_range`).
    (256, 256): _TileRow(
        config="CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
        cgrp_tile_mn=(256, 256),
        cta_tile_mn=(128, 128),
        cluster_shape_mnk=(2, 1, 1),
        ab_stages=6,
        multicast_a=False,
        a_mcast_k_major=(1, False),
        a_mcast_m_major=(1, False),
        mma_size_m=1,
        acc_stages=2,
        mixed_a_pattern_pref=1,
    ),
    # A/B alternate (selected by no adapter): the (512, 512) row's per-CTA work at a 256-wide N tile -- no padding at
    # d = 256 either, half the cluster tiles of the (256, 256) row, one 512-column accumulator (no double buffering:
    # 2 x 512 exceeds the TMEM), and a 512-row cluster M tile that is NOT causal-tight.  Kept so the wave-quantization
    # question (twice the tiles vs 25 % less L2 -> SMEM traffic per FLOP) is a one-constant A/B, not a rewrite.
    (512, 256): _TileRow(
        config="CONFIG_sm100_256x256x128_128x256x32_cluster2x1_2ctamma",
        cgrp_tile_mn=(512, 256),
        cta_tile_mn=(256, 128),
        cluster_shape_mnk=(2, 1, 1),
        ab_stages=4,
        multicast_a=False,
        a_mcast_k_major=(1, False),
        a_mcast_m_major=(1, False),
        mma_size_m=2,
        acc_stages=1,
        mixed_a_pattern_pref=1,
    ),
}
_ROW = _TILE_ROWS[tuple(getattr(PARAMS, "cgrp_tile_mn", (512, 512)))]

# Tile config: _ROW.config -- every constant below is lifted verbatim from the upstream rendering of that config (the
# row-independent ones from any of the three; they agree) -- do not hand-derive.  The fp8 arm's values are the upstream
# arch-107 rendering of CONFIG_sm100_128x256x128_128x256x64_cluster2x1_2ctamma (the same (256, 256) row at 1-byte operands).
#
# K per operand stage is ONE 128-byte swizzle row per operand row (the config family's `x128` K_BYTES): 64 bf16 / fp16
# elements, 128 e4m3.  The MMA instruction reads K = 16 (bf16 / fp16, F16) or K = 64 (e4m3, Rubin's dense-FP8 K64 form,
# idesc k_dim=1, F8F6F4) of it per step, so a stage is 4 (bf16) / 2 (e4m3) MMA k-blocks.
_K_STAGE_BYTES = 128
_K_STAGE_ELEMS = _K_STAGE_BYTES // _AB_BPE
mma_inst_shape_mnk = (256, 256, 64) if _IS_FP8 else (256, 256, 16)
mma_k_dim = 1 if _IS_FP8 else 0
cta_group = 2
cgrp_tile_mnk = (_ROW.cgrp_tile_mn[0], _ROW.cgrp_tile_mn[1], _K_STAGE_ELEMS)
cta_tile_mnk = (_ROW.cta_tile_mn[0], _ROW.cta_tile_mn[1], _K_STAGE_ELEMS)
epi_tile_mn = (128, 64)
threads_per_cta = 256
cluster_shape_mnk = _ROW.cluster_shape_mnk
# Upstream carries the rendering's batch size here purely to detect a BROADCAST
# operand (== 1 means "one batch element, reuse it for all"). No stage-3 GEMM
# broadcasts, so both are simply "batched"; the real extents are runtime.
matmul_a_batch = 0
matmul_b_batch = 0
# --- operand major, per FROST_TEMPLATE_PARAMS -------------------------------
# The three stage-3 GEMMs do NOT share one operand-major combination:
#   dV = S^T.dO  and  dK = dS^T.Q :  A m-major (S/dS's kv is contiguous and is M)
#   dQ = dS.K                     :  A k-major (dS's kv is contiguous and is K)
# B is n-major in all three (D is contiguous in BSHD, and D is N).
#
# Rendering the upstream template for each combination produces bodies that are
# textually IDENTICAL -- the whole difference is the ten constants below.  So one
# fork serves all three; the loader instantiates this module once per params set
# and every use is `cutlass.const_expr`, so each instance traces specialized code.
a_is_m_major = bool(PARAMS.a_is_m_major)
# THD / varlen.  Everything below folds out of the dense rendering, which is the
# one that has to keep diffing clean against the upstream template.
#
# Which axis is ragged follows from the operand major, so it needs no parameter
# of its own:
#     dV = S^T.dO, dK = dS^T.Q  (A m-major) -- M is kv, K is q tokens
#     dQ = dS.K                 (A k-major) -- M is q tokens, K is kv
# so A's blocked-workspace row offset lands on K in the first case and on M in
# the second, and B's token offset is cu_q in the first and cu_k in the second.
_THD_MM = bool(getattr(PARAMS, "thd_varlen", False))
# ... with ONE exception the operand major cannot express: which TOKEN axis the blocked
# workspace's ROWS are (`MatmulTemplateParams.thd_rows_kv`, appended 2026-10-01; read through
# getattr like `epi_mode`, so a record built before the field existed renders the q-major
# workspace it always did).  The row-offset placement above holds in both layouts (the blocked
# row axis is K for an m-major A and M for a k-major A); the TOKEN side does not: on a kv-major
# workspace (rows = packed kv tokens, the sm107 d256 chain) the k-major dK GEMM reduces over q
# tokens and the m-major dQ GEMM over kv tokens -- the opposite pairing.  `_THD_K_IS_KV` =
# "is this GEMM's K axis the kv token axis": every token-side selection below keys on it.  At
# the default (q-major rows) it equals `not a_is_m_major`, the pre-field spelling, so the SM100
# THD renderings are unchanged.
_THD_ROWS_KV = bool(getattr(PARAMS, "thd_rows_kv", False))
_THD_K_IS_KV = a_is_m_major == _THD_ROWS_KV
# This GEMM's OWN descriptor scratch: one clipped output descriptor per
# sequence, then the packed-total-clamped B operand.  Built in `_host`, which is
# the only place that knows these descriptors' box, swizzle and dim order -- for
# (n, m, h, b) operands the sequence axis is ord=1, which is NOT stage 2's.
_THD_MM_SEQ_ORD = 1
THD_MM_DESC_SLOTS = lambda b: b + 1  # noqa: E731
B_CLAMP_SLOT = lambda b: b  # noqa: E731


@cute.jit
def _thd_desc_ptr(desc_words, slot):
    """Generic-space pointer to one 128-B tensor map in the patched array."""
    return (desc_words.iterator.raw_ptr() + slot * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)


@cute.jit
def _thd_acquire_descs(desc_words, n_batch):
    """Acquire EVERY patched descriptor into the TMA proxy, once per warp.

    ``fence.proxy.tensormap::generic.acquire`` takes a size operand whose ONLY
    legal value is 128 -- one descriptor.  Acquiring the array's base therefore
    orders the FIRST slot and nothing else, so a warp that later selects slot
    ``tile_b`` (the per-sequence C descriptors) or slot ``n_batch`` (the clamped
    B descriptor) could read metadata the TMA proxy still has stale.  Loop the
    whole array instead: ``n_batch + 1`` fences, hoisted out of the persistent
    loop, because the patch launch writes these once before this kernel starts
    and never rewrites them.
    """
    for _slot in cutlass.range(n_batch + cutlass.Int32(1)):
        nvvm.fence_proxy_acquire(
            nvvm.MemScope.GPU,
            _thd_desc_ptr(desc_words, _slot),
            128,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )


b_is_n_major = bool(PARAMS.b_is_n_major)
causal_mode = int(PARAMS.causal_mode)
causal_gran = int(PARAMS.causal_gran)
causal_shift = int(PARAMS.causal_shift)
# The band's second edge (`MatmulTemplateParams.causal_window` / `.causal_diag`, appended 2026-09-29; read through getattr like
# `epi_mode` so a record built before the fields existed renders the one-sided band it always did).
causal_window = int(getattr(PARAMS, "causal_window", 0))
causal_diag = bool(getattr(PARAMS, "causal_diag", True))
# B's head group (`MatmulTemplateParams.b_head_group`, appended 2026-09-30; read through getattr like `epi_mode`, so a record
# built before the field existed renders B batched per A/C head as it always did).  The (b, h) batch hands A and C the decoded
# head `h`; B takes `h // b_head_group` (`_b_head`) -- under GQA the K head that `b_head_group` consecutive Q heads share -- and
# its descriptor's head extent is `n_head // b_head_group` (`_host`).  Every use is `const_expr`-folded at 1.
b_head_group = int(getattr(PARAMS, "b_head_group", 1))
# THD + a trimmed causal_mode = the per-sequence K-trim (`MatmulTemplateParams.thd_causal_bottom_right`, appended 2026-10-02;
# read through getattr like the fields above).  `_THD_TRIM` gates every line of it: a THD rendering at CAUSAL_K_NONE (the
# SM100 d512 chain's, the sm107 d256 chain's dense one) and every dense rendering trace exactly what they did.
thd_causal_bottom_right = bool(getattr(PARAMS, "thd_causal_bottom_right", False))
_THD_TRIM = _THD_MM and causal_mode != CAUSAL_K_NONE
mma_a_major = 1 if a_is_m_major else 0
mma_b_major = 1 if b_is_n_major else 0
ab_stages = _ROW.ab_stages
b_collector_ok = False
multicast_a = _ROW.multicast_a
multicast_b = False
# (a_mcast_slices, ab_empty_full_mask) follow the A major only under a multicast (`_TileRow.a_mcast_*`).
a_mcast_slices, ab_empty_full_mask = _ROW.a_mcast_m_major if a_is_m_major else _ROW.a_mcast_k_major
b_mcast_slices = 1
ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B
# MN-major packs one 128-byte swizzle row of elements per TMA group (64 bf16 / fp16, 128 e4m3) and walks K in
# whole-MMA-K-row steps (16 x 128 B = 2048, 64 x 128 B = 8192); K-major loads one group and steps one MMA K in bytes
# (32 / 64).  Values lifted verbatim from the upstream renderings of the same tile config at each operand width -- do
# not hand-derive them.
_MAJOR_CONSTS = {
    # bytes per element -> is_mn_major -> (desc_leading_byte_offset, k_step_bytes, tma_group_elems)
    2: {False: (16, 32, 1), True: (8192, 2048, 64)},
    1: {False: (16, 64, 1), True: (16384, 8192, 128)},
}[_AB_BPE]
a_smem_desc_leading_byte_offset, a_smem_k_step_bytes, a_tma_group_elems = _MAJOR_CONSTS[a_is_m_major]
b_smem_desc_leading_byte_offset, b_smem_k_step_bytes, b_tma_group_elems = _MAJOR_CONSTS[b_is_n_major]
a_smem_desc_stride_byte_offset = 1024
b_smem_desc_stride_byte_offset = 1024
a_smem_m_step_bytes = 16384
mma_size_m = _ROW.mma_size_m
mma_size_n = 1
mma_size_k = _K_STAGE_BYTES // (mma_inst_shape_mnk[2] * _AB_BPE)  # MMA k-blocks per stage: 4 (bf16 / fp16), 2 (e4m3)
ab_tma_swizzle = _tma.TensorMapSwizzle.s128b

# Dtype family: A=f16->MMAf16, B=f16->MMAf16, out=f16 (K_BYTES=128) on the bf16 / fp16 rows (`_IO_DTYPE` is BF16 or FP16
# per PARAMS.dtype_qkv; the MMA kind is the same); A=e4m3->MMAe4m3, B=e4m3->MMAe4m3, out=fp32 (DESCALE partial) | the gradient dtype (QUANT)
# (K_BYTES=128) on the fp8 arm.
ab_dtype = _IO_DTYPE
cd_dtype = _OUT_DTYPE
mma_a_dtype = _IO_DTYPE
mma_b_dtype = _IO_DTYPE
mma_c_dtype = cutlass.Float32
acc_widen_to_fp32 = False
ab_tma_dtype = _IO_DTYPE
mma_kind = nvvm.Tcgen05MMAKind.F8F6F4 if _IS_FP8 else nvvm.Tcgen05MMAKind.F16
# The epilogue drains a tile in `epi_n`-column subtiles, one staging ROW of `epi_row_elems` D elements per lane: 64 elements
# = 128 B (bf16 / fp16) or 64 B (e4m3 out), 32 elements = 128 B at the fp32 partial (the DESCALE partial or the block-scale arm's
# EPI_NONE partial; a 64-element fp32 row would be 256 B -- past the 128-B swizzle atom and a 2-way bank conflict per lane).  Same
# SMEM bytes per stage either way; the fp32
# arm drains twice the subtiles (8 x 32 at d = 256) and stores twice the bytes, which IS the fp32 partial's cost.
epi_row_elems = 32 if _CD_BPE == 4 else 64
epi_n = epi_row_elems
# The epilogue staging row is `epi_row_elems` x the D element: 128 B (bf16 / fp16 / fp32) or 64 B (e4m3 out).  The per-thread
# store's swizzle and the TMA-store descriptor's are ONE unit (rules/frost-tile-dsl.md S5): Swizzle(3, 4, 3) + s128b for the
# 128-B row, Swizzle(2, 4, 3) + s64b for the 64-B one -- and the per-thread stride IS the row, so the same XOR spreads the banks.
_EPI_ROW_BYTES = epi_row_elems * _CD_BPE
_EPI_SWIZZLE = {128: cutlass.Swizzle(3, 4, 3), 64: cutlass.Swizzle(2, 4, 3)}[_EPI_ROW_BYTES]
_EPI_TMA_SWIZZLE = {128: _tma.TensorMapSwizzle.s128b, 64: _tma.TensorMapSwizzle.s64b}[_EPI_ROW_BYTES]
tile_swizzle_n = 1
swizzle_l2_budget_bytes = 44214954
num_gemms = 1
num_a_operands = 1
num_b_operands = 1
gemm_a_idx = (0,)
gemm_b_idx = (0,)
num_tmem_alloc_cols = STAGE3_BLOCK_SCALE_TMEM_COLS if _BLOCK_SCALE else 512
tmem_alloc_exclusive = _BLOCK_SCALE  # the 576-column allocation is the Rubin line's exclusive mode (rules/frost-kernels.md s3)
acc_stages = _ROW.acc_stages  # mma_size_m x 256 acc cols/stage
# --- the block-scale arm: scale-factor geometry, DERIVED from the tile row (never a literal) ---------------------------------
# One E8M0 byte per STAGE3_MX_BLOCK (32) K elements of every A row / B column.  cuDNN's F8_128x4 atom (tile_dsl.sf_layout: 128
# rows x 4 K-block scales = 512 B, byte (r % 32) * 16 + (r // 32) * 4 + c) covers exactly one 128-B K stage of one 128-row (M)
# or 128-column (N) block, so per K stage:
#   SFA  1 atom  -- the CTA's 128 A rows                                             512 B of SMEM, 4 TMEM columns
#   SFB  num_blocks_n atoms -- the 256 B columns of the PAIR's instruction; each CTA
#        holds all of them (its own accumulator spans the full N)                  1024 B of SMEM, 8 TMEM columns, ONE scale_b span
# A K64 instruction consumes sf_scales_per_inst = 2 of the atom's 4 scales, selected by the idesc's a_sf_id / b_sf_id (0, 2), so
# one UTCCP of each atom serves the stage's num_k_blocks MMAs (sf_insts_per_atom == mma_size_k, asserted below).  The SF TMEM
# columns sit past the acc_stages accumulator stages: 512 + 4 + 8 = 524 of the 576 exclusive columns.  The UTCCP source
# descriptor is the F8_128x4 atom itself (leading 16 B, stride 128 B, no swizzle -- the atom order IS the 32x128b copy's order).
_SF_K_PER_STAGE = cta_tile_mnk[2] // STAGE3_MX_BLOCK  # 4 K-block scales per row per K stage
sfa_smem_bytes = cta_tile_mnk[0] * _SF_K_PER_STAGE if _BLOCK_SCALE else 0  # 512: (128 rows / 128) atoms x 512 B
num_blocks_n = mma_inst_shape_mnk[1] // SF_ATOM_ROWS  # 2: 128-column N blocks of the pair's instruction
sfb_smem_bytes = mma_inst_shape_mnk[1] * _SF_K_PER_STAGE if _BLOCK_SCALE else 0  # 1024: num_blocks_n atoms
sf_scales_per_inst = mma_inst_shape_mnk[2] // STAGE3_MX_BLOCK  # 2 at K64 (0 on the K16 f16 rows: no 32-block fits an instruction)
sf_insts_per_atom = SF_ATOM_COLS // sf_scales_per_inst if sf_scales_per_inst else 0  # 2 MMAs share one UTCCP'd atom
sfa_tmem_cols = mma_size_m * SF_ATOM_COLS  # 4
sfb_tmem_cols = num_blocks_n * SF_ATOM_COLS  # 8
_ACC_COLS_PER_STAGE = mma_size_m * (cgrp_tile_mnk[1] // cluster_shape_mnk[1])  # == the kernel's cols_per_acc_stage (256)
sfa_col_base = acc_stages * _ACC_COLS_PER_STAGE  # 512: first column past the accumulator stages
sfb_col_base = sfa_col_base + sfa_tmem_cols  # 516
if _BLOCK_SCALE:
    if not _IS_FP8 or mma_size_m != 1 or _SF_K_PER_STAGE != SF_ATOM_COLS or sf_insts_per_atom != mma_size_k:
        raise NotImplementedError(
            f"{__name__}: the block-scale arm is derived for the fp8 (256, 256) row -- one F8_128x4 atom per 128-B K stage per 128-row block "
            f"(mma_size_m 1, {SF_ATOM_COLS} scales per stage, {sf_insts_per_atom} K64 MMAs per atom == mma_size_k {mma_size_k}); got "
            f"fp8={_IS_FP8} mma_size_m={mma_size_m} scales/stage={_SF_K_PER_STAGE}"
        )
    if sfb_col_base + sfb_tmem_cols > STAGE3_BLOCK_SCALE_TMEM_COLS:
        raise NotImplementedError(
            f"{__name__}: {acc_stages} x {_ACC_COLS_PER_STAGE} accumulator + {sfa_tmem_cols} SFA + {sfb_tmem_cols} SFB TMEM columns exceed the "
            f"{STAGE3_BLOCK_SCALE_TMEM_COLS}-column exclusive allocation"
        )
mma_block_scale_kind = nvvm.MMABlockScaleKind.MXF8F6F4
scale_vec_size = nvvm.Tcgen05MMABlockScale.BLOCK32
sf_scale_format = 1  # UE8M0 -- `Tcgen05MxInstrDesc.scale_format`, the block-scale GEMM compiler's encoding for fp8_e8m0 scales
vec_bytes_epi = int(PARAMS.vec_bytes_epi)
n_tma_outputs = 1
moe_aligned_offsets = False
epi_slot_widen = 1
epi_packed_lanes = False
epi_dp22 = False
epi_stage_rows = 128
epi_chunk_elems = 64
ab_stages = _ROW.ab_stages  # SMEM-D 32784B fixed + cast LOAD 0B/stage + multi-GEMM 0B/stage
# Upstream renders (2, 1, 1) here -- a mixed-CGA fallback the driver may pick
# per cluster when the preferred shape does not fit. Pinned to None in this
# fork: `_host` always sizes the grid as a multiple of the preferred cluster, so
# the fallback is unreachable by construction, and its paths carry their own
# coordinate/multicast logic that the 2-D (b, h) batch rewrite has never
# exercised.
fallback_cluster_shape_mnk = None
mixed_a_pattern_pref = _ROW.mixed_a_pattern_pref
mixed_b_pattern_pref = 1
mixed_a_pattern_fb = 1
mixed_b_pattern_fb = 1

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


def _smem_layout_bytes() -> dict:
    """Byte offsets of the kernel's SMEM buffers, in DECLARATION order with each ``cutlass.Array``'s alignment
    (the order and alignments of the allocations in ``_bprop_matmul_bh_sm100_kernel``; keep the two in step).

    Why this exists: every tcgen05 SMEM descriptor here is built by ``Tcgen05SmemDesc.build()`` at its default
    (version 0), whose 14-bit start address covers the first 256 KiB of SMEM only; per-stage / per-k-step
    advances are bare adds on top of the ROOT (``smem_a_list[0]`` / ``smem_b_list[0]``), so a ring whose root sits
    at or past 256 KiB wraps silently -- an accumulator of exactly zero, operands provably right in SMEM
    (rules/mma-tma-matrix.md s6).  The import-time guard below turns that into a raise the moment a row (a deeper
    ring under Rubin's 327 KiB carveout) needs a version-1 descriptor; the host pins read the same numbers.
    """
    off = 0
    out = {}

    def place(name, nbytes, align):
        nonlocal off
        off = (off + align - 1) // align * align
        out[name] = off
        off += nbytes

    place("sys_reserved", 1024, 1)
    place("ab_full_mbar", 8 * ab_stages, 8)
    place("ab_empty_mbar", 8 * ab_stages, 8)
    place("acc_empty_mbar", 8 * acc_stages, 8)
    place("acc_full_mbar", 8 * acc_stages, 8)
    if cta_group == 2:
        place("tmem_dealloc_mbar", 8, 8)
    place("tmem_ptr", 4, 4)
    place("clc_response", 16 * CLC_SCHED_STAGES, 16)
    place("clc_full_mbar", 8 * CLC_SCHED_STAGES, 8)
    place("clc_empty_mbar", 8 * CLC_SCHED_STAGES, 8)
    ab_bpe = ab_dtype.width // 8
    if _BLOCK_SCALE:
        # The scale-factor rings go FIRST: their roots feed the UTCCP's version-0 tcgen05 SMEM descriptor too, and the small rings
        # ahead keep the big A / B roots far below the 256 KiB line (the block-scale GEMM template's declaration order).
        place("smem_sfa_0", sfa_smem_bytes * ab_stages, 1024)
        place("smem_sfb_0", sfb_smem_bytes * ab_stages, 1024)
    for i in range(num_a_operands):
        place(f"smem_a_{i}", cta_tile_mnk[0] * cta_tile_mnk[2] * ab_bpe * ab_stages, 1024)
    for j in range(num_b_operands):
        place(f"smem_b_{j}", cta_tile_mnk[1] * cta_tile_mnk[2] * ab_bpe * ab_stages, 1024)
    place("smem_d", epi_stage_rows * epi_row_elems * epi_slot_widen * (cd_dtype.width // 8) * EPI_SMEM_STAGES, 1024)
    out["total"] = off
    return out


_SMEM_DESC_V0_LIMIT = 1 << 18  # 256 KiB: the reach of a version-0 tcgen05 SMEM descriptor's start address
_smem_layout = _smem_layout_bytes()
if (
    max(_smem_layout[f"smem_a_{i}"] for i in range(num_a_operands)) >= _SMEM_DESC_V0_LIMIT
    or max(_smem_layout[f"smem_b_{j}"] for j in range(num_b_operands)) >= _SMEM_DESC_V0_LIMIT
    or (_BLOCK_SCALE and max(_smem_layout["smem_sfa_0"], _smem_layout["smem_sfb_0"]) >= _SMEM_DESC_V0_LIMIT)
):
    raise NotImplementedError(
        f"{__name__}: an MMA-operand ring root sits at or past 256 KiB ({_smem_layout}); the version-0 tcgen05 SMEM descriptor "
        f"`Tcgen05SmemDesc.build()` emits cannot address it (rules/mma-tma-matrix.md s6) -- this row needs a version-1 descriptor"
    )


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


@cute.jit
def _decode_bh(l, n_head):
    """Flat CLC batch index -> ``(head, batch)``.

    The CLC scheduler rasterizes ONE batch axis, so ``l`` stays flat there and
    the tile hand-out is unchanged.  The pair is only needed where a TMA
    coordinate is formed -- see the module docstring for why the tensors are
    not flattened instead.
    """
    return l % n_head, l // n_head


@cute.jit
def _b_head(tile_h):
    """B's head coordinate for the decoded A/C head ``tile_h``: ``tile_h // b_head_group`` -- under GQA the K head that
    ``b_head_group`` consecutive Q heads share (the dQ GEMM's B over a whole head chunk); the identity at 1, so every
    rendering that predates the field keeps its coordinate tuple.  Per tile, in the TMA warp: one division per tile."""
    return tile_h if cutlass.const_expr(b_head_group == 1) else tile_h // cutlass.Int32(b_head_group)


@cute.jit
def _thd_group(meta_t, tile_b, n_batch, num_k_tiles):
    """Per-sequence offsets and k-tile count for one (head, sequence) group.

    Returns ``(a_k_off, a_m_off, b_k_off, num_k_tiles)``.  Dense returns zeros
    and the kernel-wide tile count, so every use folds away.

    The k count comes from the sequence's REAL length, which is safe on both
    sides of the ragged axis because stage 2 zero-fills further than this reads:
    its blocked rows run to ``ceil(S_q/128)*128`` and its columns to
    ``ceil(S_kv/128)*128``, both at least the ``ceil(len/64)`` tiles counted
    here.  Reading further would reach the NEXT sequence's live rows -- nonzero
    data summed into the wrong gradient, with nothing to make it look wrong.
    """
    if cutlass.const_expr(not _THD_MM):
        return cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), num_k_tiles
    meta = cutlass.make_array_view(meta_t)
    cu_q0 = n_batch
    cu_k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
    row0 = cutlass.Int32(4) * n_batch + cutlass.Int32(4)
    q_tok = cutlass.Int32(meta[cu_q0 + tile_b])
    k_tok = cutlass.Int32(meta[cu_k0 + tile_b])
    s_q = cutlass.Int32(meta[cu_q0 + tile_b + cutlass.Int32(1)]) - q_tok
    s_kv = cutlass.Int32(meta[cu_k0 + tile_b + cutlass.Int32(1)]) - k_tok
    row_off = cutlass.Int32(meta[row0 + tile_b])
    a_k_off = row_off if cutlass.const_expr(a_is_m_major) else cutlass.Int32(0)
    a_m_off = cutlass.Int32(0) if cutlass.const_expr(a_is_m_major) else row_off
    # The token side of the reduction: kv tokens (B at cu_k, s_kv) or q tokens (B at cu_q, s_q) -- see `_THD_K_IS_KV`.
    b_k_off = k_tok if cutlass.const_expr(_THD_K_IS_KV) else q_tok
    k_len = s_kv if cutlass.const_expr(_THD_K_IS_KV) else s_q
    nkt = (k_len + cutlass.Int32(cta_tile_mnk[2] - 1)) // cutlass.Int32(cta_tile_mnk[2])
    return a_k_off, a_m_off, b_k_off, nkt


@cute.jit
def _thd_shift(meta_t, tile_b, n_batch):
    """THD: the causal diagonal's offset for sequence ``tile_b`` -- ``s_kv[b] - s_q[b]`` under ``thd_causal_bottom_right``
    (the kernels' per-sequence bottom-right anchor, ``compute_q_loop_bounds``), 0 for the top-left diagonal.  Reads the two
    ``(B+1,)`` prefixes of the metadata buffer; the grid is B sequences deep, so ``tile_b < n_batch`` always holds here."""
    if cutlass.const_expr(not thd_causal_bottom_right):
        return cutlass.Int32(0)
    meta = cutlass.make_array_view(meta_t)
    cu_q0 = n_batch
    cu_k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
    s_q = cutlass.Int32(meta[cu_q0 + tile_b + cutlass.Int32(1)]) - cutlass.Int32(meta[cu_q0 + tile_b])
    s_kv = cutlass.Int32(meta[cu_k0 + tile_b + cutlass.Int32(1)]) - cutlass.Int32(meta[cu_k0 + tile_b])
    return s_kv - s_q


@cute.jit
def _thd_sf_tile_base(sf_meta_t, tile_b, n_batch):
    """THD leg of the block-scale arm: the first PACKED scale-factor tile of sequence ``tile_b`` on the B (token) side.

    The packed MXFP8 SF convention pads every sequence to whole 128-token tiles, so the tile a sequence's k tile ``k_tile_idx``
    reads is ``cu_sf[b] + k_tile_idx`` with ``cu_sf[b] = SUM_{i<b} ceil(s_i / 128)`` -- never ``cu[b] // 128``, which is short by
    one tile for every ragged sequence before it (finite, plausible, wrong scales).  ``sf_meta_t`` is the int32
    ``[cu_sf_q(B+1) | cu_sf_k(B+1)]`` buffer (``config_sm100.STAGE3_THD_SF_*``); which prefix follows the token side of this
    GEMM's reduction exactly like ``_thd_group``'s ``b_k_off`` (``_THD_K_IS_KV``: kv tokens -> ``cu_sf_k``, q tokens -> ``cu_sf_q``).
    Called by the TMA warp once per tile, inside the arm's ``const_expr`` guards."""
    meta = cutlass.make_array_view(sf_meta_t)
    cu0 = cutlass.Int32(STAGE3_THD_SF_CU_K_OFF(n_batch)) if cutlass.const_expr(_THD_K_IS_KV) else cutlass.Int32(STAGE3_THD_SF_CU_Q_OFF)
    return cutlass.Int32(meta[cu0 + tile_b])


@cute.jit
def _sf_planes_view_thd(sf, planes: cutlass.Constexpr, tiles, heads: cutlass.Constexpr):
    """The block-scale arm's SFB view of a PACKED (THD) columnwise F8_128x4 scale tensor -- ``(512 B atom, D planes, packed
    tiles, H, 1)`` with the byte strides ``config_sm100.stage3_thd_sfb_layout`` spells (plane stride one atom, tile stride the
    ``planes x 512``-byte slab, the forward's per-sequence-tile-padded convention); ``tiles`` is the bound buffer's packed tile
    count (host int or traced ``Int32``).  The THD twin of the host's dense D-plane-major view: the kernel's THD SFB coordinate
    ``(0, plane, cu_sf[b] + k_tile, h, 0)`` is written against THIS layout."""
    shape, strides = stage3_thd_sfb_layout(planes, tiles, heads, SF_ATOM_BYTES)
    return cute.make_tensor(sf.iterator, cute.make_layout(shape, stride=strides))


@cute.jit
def _thd_causal_k_range(coord_m_cgrp, nkt, shift):
    """The THD arm of :func:`_causal_k_range`: the same two-sided band, in SEQUENCE-LOCAL rows.

    Under THD every coordinate the trim works in is the sequence's own: ``m0`` is the cluster M tile inside the sequence
    (the blocked workspace's ``row_off[b]`` and the packed ``cu_*[b]`` are added to the TMA coordinates AFTER the trim),
    ``nkt`` is ``ceil(len_b / tk)`` over the sequence's REAL reduction length (``_thd_group``) and ``shift`` is the
    sequence's diagonal offset (``_thd_shift``: ``s_kv[b] - s_q[b]`` bottom-right, 0 top-left).  The bounds are the dense
    arm's -- the causal edge rounded outward to ``causal_gran`` (the kernels' 256-row q pair, ``_q_loop_bounds``), the
    window edge to the k tile -- so every tile read was written by the sequence's own kv blocks (the host tile walk
    ``test_stage3_thd_band_arithmetic`` and the poisoned-workspace THD tests are the proof).

    Two things differ from the dense arm.  (1) A sequence's lengths are anything: ``shift`` may be NEGATIVE (bottom-right
    with ``s_q[b] > s_kv[b]``), ``nkt`` may be 0 (an empty reduction side) and, under a top-left window, a q pair may sit
    past ``s_kv[b] + W`` (no kv block writes it -- the one geometry the dense adapter still zero-fills for).  Every
    dividend is clamped at 0 before its ``//`` and every bound at ``nkt``.  (2) The range MAY BE EMPTY, and an empty range
    means "this tile's rows have no kept cell" -- a kv block no query attends, a q pair with no key in its band, an empty
    reduction -- whose gradient is exactly zero.  The mainloop then runs zero iterations and the epilogue stores zeros
    through a SELECT keyed on the same range (``_thd_store_live``), never the dense arm's never-empty clamp, which would
    read a tile the kernel did not write.
    """
    blk = cutlass.Int32(causal_gran)
    tk = cutlass.Int32(cta_tile_mnk[2])
    m0 = cutlass.Int32(coord_m_cgrp)
    zero = cutlass.Int32(0)
    if cutlass.const_expr(causal_mode == CAUSAL_K_LO):
        # dV / dK: M = kv (one 256-row block of the sequence), K = q tokens of the sequence.
        k_lo = zero
        if cutlass.const_expr(causal_diag):
            lo = cute.math.max(m0 - shift, zero)
            k_lo = cute.math.min(((lo // blk) * blk) // tk, nkt)
        k_hi = nkt
        if cutlass.const_expr(causal_window > 0):
            hi = cute.math.max(m0 + cutlass.Int32(cgrp_tile_mnk[0]) - shift + cutlass.Int32(causal_window), zero)
            k_hi = cute.math.min((hi + tk - cutlass.Int32(1)) // tk, nkt)
        k_hi = cute.math.max(k_hi, k_lo)
        return k_lo, k_hi
    # dQ: M = q (a 256-row pair of the sequence), K = kv tokens of the sequence.
    k_hi = nkt
    if cutlass.const_expr(causal_diag):
        hi_raw = m0 + cutlass.Int32(cgrp_tile_mnk[0] - 1) + shift
        hi = ((cute.math.max(hi_raw, zero) // blk) + cutlass.Int32(1)) * blk
        k_hi = cute.math.min((hi + tk - cutlass.Int32(1)) // tk, nkt)
        # The pair's LAST row still has no key (q + shift < 0: bottom-right with s_q > s_kv): no band at all.
        k_hi = cutlass.Int32(arith.select((hi_raw < zero).ir_value(), zero.ir_value(), k_hi.ir_value()))
    k_lo = zero
    if cutlass.const_expr(causal_window > 0):
        lo = cute.math.max(m0 + shift - cutlass.Int32(causal_window), zero)
        k_lo = cute.math.min(lo // tk, k_hi)
    return k_lo, k_hi


@cute.jit
def _causal_k_range(coord_m_cgrp, num_k_tiles, thd_shift=None):
    """``[k_begin, k_end)`` -- the K tiles this cluster M tile reads under the mask band stage 2 wrote.

    THE INVARIANT (both kernels, both directions): the GEMM reads exactly the tiles
    stage 2 writes for the band, plus at most one OUTWARD-rounded k tile per side,
    and only tiles the caller zeroed or the kernel wrote.  Stage 2 writes a kv
    block's q range rounded OUTWARD to ``causal_gran`` rows (the sm107 bodies'
    ``_q_loop_bounds``: two 128-row q tiles = the 256-row kv block = this GEMM's
    cluster M tile); every bound here rounds OUTWARD too (down for a low bound,
    up for a high one), first to ``causal_gran`` where the bound is the diagonal's,
    then to the k tile ``tk``.  So the range covers every structurally non-zero
    cell and stays inside the written region -- an unwritten tile is never read,
    and no zero-fill is needed (the poisoned-workspace tests
    ``test_masked_stage3_reads_only_what_stage2_wrote`` in the two sm107 suites
    are the runtime proof; ``test_stage3_two_sided_band_arithmetic`` the host one).

    The band, with ``shift`` = how far the diagonal sits past ``kv == q`` (the
    bottom-right ``S_kv - S_q``; 0 top-left) and ``W = causal_window``::

        causal edge (causal_diag)   kv <= q + shift          <=>  q >= kv - shift
        window edge (W > 0)         kv >= q + shift - W      <=>  q <= kv - shift + W

    and the two renderings, each keyed on the CLUSTER's M base ``m0`` (identical on
    both CTAs of a pair) over the M tile ``[m0, m0 + M)``:

        CAUSAL_K_LO  (dV / dK: M = kv, K = q)
            k_lo = floor((m0 - shift) / gran) * gran / tk         causal edge  (= stage 2's rounded first q tile)
            k_hi = ceil((m0 + M - shift + W) / tk)                window edge  (the tile's last kv row's last kept q)
        CAUSAL_K_HI  (dQ: M = q, K = kv)
            k_lo = floor((m0 + shift - W) / tk)                   window edge  (the tile's first q row's first kept kv)
            k_hi = ceil((floor((m0 + M - 1 + shift) / gran) + 1) * gran / tk)   causal edge (= stage 2's last kv block)

    An absent edge leaves its bound at 0 / ``nkt``.  Where a bound is the
    diagonal's it is rounded to ``causal_gran`` -- the granularity stage 2 writes
    at -- because the M tile of the OTHER GEMM spans that many rows and the two
    tiles of a pair must have been written together; where it is the window's it
    is rounded to ``tk`` only, which is finer than what stage 2 wrote (``tk`` divides
    the q tile), hence still inside it.  The two edges are independent, so a
    window without a diagonal (``causal_diag=False``, SWA-only) is the same code
    with one bound dropped.

    Tight-trim invariant: ``cgrp_tile_mnk[0] <= causal_gran`` -- one cluster M
    tile fits inside one stage-2 write block.  It HOLDS on the (256, 256) row
    (256 <= 256, the sm107 d256 chain) and does NOT hold on the (512, 512) and
    (512, 256) rows (512 > 256): a 512-row M tile straddles two 256-row stage-2
    blocks, and since the bounds are per tile no range can cover the tile's live
    rows without also covering its neighbour's skipped ones.  On those rows this
    is an OPTIMIZATION ONLY and the caller's zero-fill is what makes it correct --
    the SM100 adapter sets ``_zero_ws`` whenever a causal-family mask is active,
    which is exactly the condition under which any of this runs.  Do not weaken
    that zero-fill on the theory that the trim protects the aligned case; at 512
    rows it does not.  (The SM100 d512 chain keeps the (512, 512) row despite the
    looser trim because it measures faster BOTH ways at B=1 H=128 S=8192 d=512
    bf16: +3.5 % no_mask, +7.8 % causal.)

    NEVER return an empty range.  The mainloop would run zero iterations, but the
    EPILOGUE still stores the accumulator -- and ``scale_d`` starts False, so with
    no MMA the accumulator is uninitialised TMEM and the output row is garbage.
    The clamps below keep one k tile, chosen inside the written region: LO clamps
    ``k_lo`` to ``nkt - 1`` (the last q tile, which stage 2's forced fully-masked
    tile wrote with zeros for a kv block past the last query) and ``k_hi`` to
    ``k_lo + 1``; HI clamps ``k_hi`` to ``>= 1`` (kv block 0 wrote every q pair
    under a causal band) and ``k_lo`` to ``k_hi - 1``.  One structurally-masked
    tile contributes exactly 0, which is the answer those rows want.  The ONE
    geometry where a q pair is written by NO kv block -- a top-left window with
    ``S_q > roundup(S_kv + W, gran)`` -- is the one case the sm107 adapter still
    zero-fills for (``_stage3_needs_zero_fill``).

    Under THD (``_THD_TRIM``) the arm is :func:`_thd_causal_k_range`: the same
    band in SEQUENCE-LOCAL rows with the sequence's own ``nkt`` and diagonal
    offset ``thd_shift`` (``_thd_shift``), and an EMPTY range where the tile
    has no kept cell (the epilogue stores zeros for it).  The SM100 d512 chain
    renders the same arm for its packed causal graphs (``api_dsl.THD_STAGE3_TRIM``,
    diagonal edge only) and KEEPS its zero-fill: its 512-row M tile straddles
    two 256-row stage-2 blocks, so there the trim is the optimization and the
    fill the correctness (``SdpaBwdDslSm100.compile``).
    """
    # num_k_tiles is Int64 (it derives from the Int64 `k`); normalise so the
    # bounds and the min() / max() below share one numeric type.
    nkt = cutlass.Int32(num_k_tiles)
    if cutlass.const_expr(causal_mode == CAUSAL_K_NONE):
        return cutlass.Int32(0), nkt
    if cutlass.const_expr(_THD_TRIM):
        return _thd_causal_k_range(coord_m_cgrp, nkt, thd_shift)
    blk = cutlass.Int32(causal_gran)
    tk = cutlass.Int32(cta_tile_mnk[2])
    m0 = cutlass.Int32(coord_m_cgrp)
    shift = cutlass.Int32(causal_shift)
    # Every new operand is clamped at 0 BEFORE its `//`: the bounds are non-negative rows, and a negative dividend's
    # division direction is not something this arithmetic should depend on.
    if cutlass.const_expr(causal_mode == CAUSAL_K_LO):
        # dV / dK: output row is kv, so K (= q) starts at the stage-2 block holding kv -- pulled EARLIER by the shift,
        # because S[q, kv] is non-zero for q >= kv - shift.  Clamped at 0, then to the last q tile (never empty).
        k_lo = cutlass.Int32(0)
        if cutlass.const_expr(causal_diag):
            lo = m0 - shift
            lo = cute.math.max(lo, cutlass.Int32(0))
            k_lo = ((lo // blk) * blk) // tk
            k_lo = cute.math.min(k_lo, nkt - cutlass.Int32(1))
        k_hi = nkt
        if cutlass.const_expr(causal_window > 0):
            # ... and ENDS after the last q the tile's last kv row keeps: q <= kv - shift + W for kv < m0 + M, so
            # q < m0 + M - shift + W; rounded UP to the k tile, clamped to nkt and to at least one tile past k_lo.
            hi = m0 + cutlass.Int32(cgrp_tile_mnk[0]) - shift + cutlass.Int32(causal_window)
            hi = cute.math.max(hi, cutlass.Int32(0))
            k_hi = cute.math.min((hi + tk - cutlass.Int32(1)) // tk, nkt)
            k_hi = cute.math.max(k_hi, k_lo + cutlass.Int32(1))
        return k_lo, k_hi
    # dQ: output row is q, so K (= kv) ends after q's stage-2 block, pushed LATER by the shift (kv <= q + shift) ...
    k_hi = nkt
    if cutlass.const_expr(causal_diag):
        hi = ((m0 + cutlass.Int32(cgrp_tile_mnk[0] - 1) + shift) // blk + cutlass.Int32(1)) * blk
        hi = cute.math.max(hi, blk)
        k_hi = cute.math.min((hi + tk - cutlass.Int32(1)) // tk, nkt)
        k_hi = cute.math.max(k_hi, cutlass.Int32(1))
    k_lo = cutlass.Int32(0)
    if cutlass.const_expr(causal_window > 0):
        # ... and STARTS at the first kv the tile's first q row keeps: kv >= q + shift - W for q >= m0, so kv >= m0 + shift - W;
        # rounded DOWN to the k tile (clamped at 0) and to at least one tile before k_hi.
        lo = m0 + shift - cutlass.Int32(causal_window)
        lo = cute.math.max(lo, cutlass.Int32(0))
        k_lo = cute.math.min(lo // tk, k_hi - cutlass.Int32(1))
    return k_lo, k_hi


@cute.kernel
def _bprop_matmul_bh_sm100_kernel(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_h_0: cutlass.Int64,
    out_stride_b_0: cutlass.Int64,
    n_head: cutlass.Int32,
    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],
    # Dense: 1-element dummies.  THD: the metadata buffer the setup launch
    # published, and this GEMM's own descriptor scratch (per-sequence clipped C,
    # then the packed-total-clamped B).
    meta_t: cute.Tensor,
    desc_words: cute.Tensor,
    n_batch: cutlass.Int32,
    # The fp8 arm's epilogue operands (fp32 [1] device tensors; None-specialized away at EPI_NONE): the two descales whose
    # product undoes the operands' scaling (descale_dP of the dS workspace, descale_{q|k} of the payload), the output scale
    # and the amax target of EPI_QUANT (an amax the graph left virtual is None: its fold and atomic fold out).
    epi_descale_0: Optional[cute.Tensor],
    epi_descale_1: Optional[cute.Tensor],
    epi_scale_out: Optional[cute.Tensor],
    epi_amax: Optional[cute.Tensor],
    # The block-scale arm's THD leg: the int32 per-sequence SF TILE prefixes `[cu_sf_q(B+1) | cu_sf_k(B+1)]` the TMA warp's SFB
    # coordinate takes its tile base from (`_thd_sf_tile_base`); None -- specialized away -- on every dense rendering and on the
    # plain (non-block-scale) THD arm.  Sits AHEAD of the two tensor maps so those stay the kernel's last two parameters (the
    # kernel's own order is `_host`'s business; the public, append-only surface is `_host`'s signature).
    sf_meta_t: Optional[cute.Tensor],
    # The block-scale arm's scale-factor tensor maps (None-specialized away when PARAMS.block_scale is False): SFA over the A
    # operand's F8_128x4 atoms, SFB over the D-plane-major columnwise Q / K SF -- both 5-D, built by `_host`.
    tma_sfa_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_sfb_desc_0: cutlass.GridConstant[_tma.TensorMap],
) -> None:
    tma_a_descs = [tma_a_desc_0]
    tma_b_descs = [tma_b_desc_0]
    tma_c_descs = [tma_c_desc_0]

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
        # B and C's GridConstant descriptors are DEAD under THD: B loads take the
        # packed-total-clamped slot and C stores take the per-sequence one, both
        # from the patched array.  Prefetching them would warm metadata nobody
        # reads -- and the patched slots must NOT be prefetched here in their
        # place, because that would cache their contents ahead of the
        # `fence_proxy_acquire` that makes the patch visible to the TMA proxy.
        if cutlass.const_expr(not _THD_MM):
            for _j in cutlass.range_constexpr(num_b_operands):
                nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

            for _ci in cutlass.range_constexpr(n_tma_outputs):
                nvvm.prefetch_tensormap(tma_c_descs[_ci].get_ptr())
        if cutlass.const_expr(_BLOCK_SCALE):
            nvvm.prefetch_tensormap(tma_sfa_desc_0.get_ptr())
            nvvm.prefetch_tensormap(tma_sfb_desc_0.get_ptr())

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
    if cutlass.const_expr(_BLOCK_SCALE):
        # SMEM buffer table of the arm (declaration order == `_smem_layout_bytes`; the operand rings and the epilogue staging
        # keep the fork's rows):
        #   smem_sfa  uint8 x ab_stages x 512   TMA writes one F8_128x4 atom per stage; the UTCCP (32x128b, WARPX4) reads it through
        #             a tcgen05 SMEM descriptor (leading 16 B, stride 128 B, NO swizzle: the atom's byte order IS the copy's order)
        #   smem_sfb  uint8 x ab_stages x 1024  TMA writes num_blocks_n atoms per stage (both D planes of the columnwise SF);
        #             two UTCCPs per stage read them, one per atom
        # No lane addresses either buffer, so there is no bank job to do; the swizzle is the descriptor's (none).
        smem_sfa = cutlass.Array(cutlass.Uint8, sfa_smem_bytes * ab_stages, space=cutlass.AddressSpace.smem, alignment=1024)
        smem_sfb = cutlass.Array(cutlass.Uint8, sfb_smem_bytes * ab_stages, space=cutlass.AddressSpace.smem, alignment=1024)
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
        cd_dtype,
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
    if cutlass.const_expr(_BLOCK_SCALE):
        # Barrier table delta of the arm: the stage's SF atoms ride the SAME ab_full[stage] barrier as the operands -- a
        # cp.async.bulk.tensor at cta_group::2 lands EVERY byte of the pair on the leader's mbarrier, so the leader's one
        # expect_tx grows by both CTAs' SFA + SFB bytes (the forward's and the MXFP8 backward body's SF routing).  Arrive count
        # and phase are unchanged: 1 elected expect_tx per stage, consumed by the leader's MMA warp.
        num_tma_copy_bytes = num_tma_copy_bytes + (sfa_smem_bytes + sfb_smem_bytes) * cta_group

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
        k_dim=mma_k_dim,
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
        tile_h, tile_b = _decode_bh(tile_l, n_head)
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_tma = cutlass.Int32(0)
        # B's THD descriptor is CLAMPED to the packed total `cu_*[B]`, so the
        # last k tile of the last sequence reads the caller's unwritten capacity
        # tail as TMA zeros instead of live memory.  The GridConstant descriptor
        # is built at the buffer's CAPACITY (a declared `max_total_seq_len` is a
        # maximum, while the row that must read zero moves every step), so using
        # it here would multiply A's padding zeros by whatever the caller left
        # past `cu_*[B]` -- and `0 * NaN` is NaN.
        if cutlass.const_expr(_THD_MM):
            _thd_acquire_descs(desc_words, n_batch)
        while is_valid != 0:
            coord_m_per_cta = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            if cutlass.const_expr(cta_group == 1):
                coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * cta_tile_mnk[1]
            else:
                coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]
            # Broadcast operands sit at (h, b) = (0, 0); batched ones carry the
            # decoded pair.  Same const_expr shape as upstream, one coord wider.
            if cutlass.const_expr(matmul_a_batch == 1):
                tile_h_a = cutlass.Int32(0)
                tile_b_a = cutlass.Int32(0)
            else:
                tile_h_a = tile_h
                tile_b_a = tile_b
            if cutlass.const_expr(matmul_b_batch == 1):
                tile_h_b = cutlass.Int32(0)
                tile_b_b = cutlass.Int32(0)
            else:
                # B's head follows its head group (`_b_head`: `tile_h // b_head_group`, `tile_h` itself at the default 1).
                tile_h_b = _b_head(tile_h)
                tile_b_b = tile_b

            _a_k_off, _a_m_off, _b_k_off, _nkt = _thd_group(meta_t, tile_b, n_batch, num_k_tiles)
            if cutlass.const_expr(_THD_MM):
                # Packed operands have ONE batch element; the sequence is
                # reached by the coordinate offsets above, not by this axis.
                tile_b_a = cutlass.Int32(0)
                tile_b_b = cutlass.Int32(0)
            if cutlass.const_expr(_THD_TRIM):
                k_begin, k_end = _causal_k_range(tile_m * cgrp_tile_m_cur, _nkt, _thd_shift(meta_t, tile_b, n_batch))
            else:
                k_begin, k_end = _causal_k_range(tile_m * cgrp_tile_m_cur, _nkt)
            if cutlass.const_expr(_BLOCK_SCALE):
                # THD leg: B's scale factors are packed per SEQUENCE in whole 128-token tiles, so the sequence's first SF tile is
                # the prefix `cu_sf[b]` (not `cu[b] // 128`), read once per tile here and added to the k tile at the SFB load.
                if cutlass.const_expr(_THD_MM):
                    _sfb_tile_base = _thd_sf_tile_base(sf_meta_t, tile_b, n_batch)
            for k_tile_idx in range(k_begin, k_end):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                    pass

                coord_k = k_tile_idx * cta_tile_mnk[2]
                # A rides the blocked workspace, B the packed tokens, so the two
                # ragged bases differ and the k coordinate cannot be shared.
                coord_k_a = coord_k + _a_k_off
                coord_k_b = coord_k + _b_k_off
                coord_m_a = coord_m_per_cta + _a_m_off
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
                                    (coord_k_a, coord_m_a + n_rank * _a_rows, tile_h_a, tile_b_a),
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
                                        (coord_k_a, coord_m_a + _a_idx * _a_rows, tile_h_a, tile_b_a),
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
                                                coord_m_a + m_group * a_tma_group_elems,
                                                coord_k_a,
                                                tile_h_a,
                                                tile_b_a,
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
                                        (coord_k_a, coord_m_a, tile_h_a, tile_b_a),
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
                                            coord_m_a + m_group * a_tma_group_elems,
                                            coord_k_a,
                                            tile_h_a,
                                            tile_b_a,
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
                                    (coord_k_a, coord_m_a, tile_h_a, tile_b_a),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=_CTA_GROUP,
                                )

                for _bj in cutlass.range_constexpr(num_b_operands):
                    sB_stage = smem_b_list[_bj].subview(sB_elems * stage)
                    tma_b_desc = tma_b_descs[_bj]
                    # See the note above the persistent loop: THD substitutes
                    # the packed-total-clamped descriptor for the capacity-sized
                    # GridConstant one.  Dense folds back to `.get_ptr()`.
                    _b_desc_ptr = _thd_desc_ptr(desc_words, B_CLAMP_SLOT(n_batch)) if cutlass.const_expr(_THD_MM) else tma_b_desc.get_ptr()
                    if cutlass.const_expr(b_mcast_slices > 1):
                        _b_rows = cta_tile_mnk[1] // b_mcast_slices
                        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_stage.subview(pair_m_idx * _b_rows * cta_tile_mnk[2]),
                                    _b_desc_ptr,
                                    (coord_k_b, coord_n_per_cta + pair_m_idx * _b_rows, tile_h_b, tile_b_b),
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
                                        _b_desc_ptr,
                                        (coord_k_b, coord_n_per_cta + _b_idx * _b_rows, tile_h_b, tile_b_b),
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
                                            _b_desc_ptr,
                                            (
                                                coord_n_per_cta + n_group * b_tma_group_elems,
                                                coord_k_b,
                                                tile_h_b,
                                                tile_b_b,
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
                                        _b_desc_ptr,
                                        (coord_k_b, coord_n_per_cta, tile_h_b, tile_b_b),
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
                                        _b_desc_ptr,
                                        (
                                            coord_n_per_cta + n_group * b_tma_group_elems,
                                            coord_k_b,
                                            tile_h_b,
                                            tile_b_b,
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
                                    _b_desc_ptr,
                                    (coord_k_b, coord_n_per_cta, tile_h_b, tile_b_b),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=_CTA_GROUP,
                                )

                if cutlass.const_expr(_BLOCK_SCALE):
                    # This stage's scale-factor atoms, onto the same ab_full[stage] barrier as the operands (tx counted above).
                    # SFA: the atom of (this CTA's 128-row M block, this K stage) -- coordinates (byte-in-atom, K tile, M tile, h, b)
                    # over the 5-D atom tensor; the atom order along K and M is the F8_128x4 grid of the A operand's scale matrix.
                    # SFB: the num_blocks_n D-plane atoms of this K stage -- coordinates (byte-in-atom, first D plane, K tile, h, b),
                    # the box spanning the planes (the columnwise SF's plane stride grows with S; the descriptor carries it).  The
                    # first plane is the pair's N base in 128-column planes -- the same base the B operand load advances with
                    # (tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n, without the pair-member half), so a grid with more
                    # than one N tile dequantizes each tile with its own planes; the shipped records have one N tile (n == 256).
                    # THD: A's atoms ride the BLOCKED workspace, so both SFA atom indices take the sequence's row offset exactly as
                    # the operand coordinates do (dK: M tile `(coord_m + row_off[b]) / 128`; dQ: K tile `(coord_k + row_off[b]) / 128`
                    # -- `coord_k_a` / `coord_m_a` carry it), and the SFB tile is the sequence's PACKED tile prefix plus its own k
                    # tile (`_sfb_tile_base`, read per tile above).  Dense keeps the loop's own indices: the ternaries fold and the
                    # rendering is byte-identical.
                    _sfa_k_tile = coord_k_a // SF_ATOM_ROWS if cutlass.const_expr(_THD_MM) else k_tile_idx
                    _sfa_m_tile = coord_m_a // SF_ATOM_ROWS if cutlass.const_expr(_THD_MM) else coord_m_per_cta // SF_ATOM_ROWS
                    _sfb_k_tile = _sfb_tile_base + k_tile_idx if cutlass.const_expr(_THD_MM) else k_tile_idx
                    if elect_one:
                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                            smem_sfa.subview(sfa_smem_bytes * stage),
                            tma_sfa_desc_0.get_ptr(),
                            (cutlass.Int32(0), _sfa_k_tile, _sfa_m_tile, tile_h_a, tile_b_a),
                            ab_full_mbar_ptr.subview(stage),
                            [],
                            multicast_mask=tma_mcast_mask_a,
                            group=_CTA_GROUP,
                        )
                    if elect_one:
                        sfb_plane_base = (tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n) // SF_ATOM_ROWS
                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                            smem_sfb.subview(sfb_smem_bytes * stage),
                            tma_sfb_desc_0.get_ptr(),
                            (cutlass.Int32(0), sfb_plane_base, _sfb_k_tile, tile_h_b, tile_b_b),
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
            tile_h, tile_b = _decode_bh(tile_l, n_head)
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
            # The MMA warp tracks its own tile_m now: the causal K range depends on
            # it, and this arm must walk exactly the range the TMA warp walks.
            # THD's per-group k count needs the SEQUENCE for the same reason, so
            # the batch index is tracked here too.
            tile_m = init_tile_m
            _, tile_b_mma = _decode_bh(init_tile_l, n_head)
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
            if cutlass.const_expr(_BLOCK_SCALE):
                # UTCCP source descriptors over the SF rings (the F8_128x4 atom: leading 16 B, stride 128 B, no swizzle), one
                # Mx instruction descriptor per k-block (sf ids 0 / 2 pick the k-block's two scales out of the atom's four), and
                # the SF TMEM addresses: SFA one 4-column word, SFB one 8-column span that the 256-wide instruction reads whole
                # (atom bn of the stage lands at +bn * 4 columns).
                desc_sfa_root = cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_sfa,
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
                )
                desc_sfb_root = cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_sfb,
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
                )
                idesc_by_k = [
                    cutlass.experimental.primitives.Tcgen05MxInstrDesc.build(
                        a_dtype=mma_a_dtype,
                        b_dtype=mma_b_dtype,
                        scale_format=sf_scale_format,
                        n_dim=mma_inst_shape_mnk[1],
                        m_dim=mma_inst_shape_mnk[0],
                        a_major=mma_a_major,
                        b_major=mma_b_major,
                        a_sf_id=_kb * sf_scales_per_inst,
                        b_sf_id=_kb * sf_scales_per_inst,
                        k_dim=mma_k_dim,
                    )
                    for _kb in range(num_k_blocks)
                ]
                sfa_tmem_base = (base_row_id << 16) | (base_col_id_root + sfa_col_base)
                sfb_tmem_base = (base_row_id << 16) | (base_col_id_root + sfb_col_base)
                sfa_ptr = nvvm.make_tmem_ptr(sfa_tmem_base, cutlass.Float32)
                sfb_ptr = nvvm.make_tmem_ptr(sfb_tmem_base, cutlass.Float32)
                sfb_block_ptrs = [nvvm.make_tmem_ptr(sfb_tmem_base + _bn * SF_ATOM_COLS, cutlass.Float32) for _bn in range(num_blocks_n)]
                _s2t_shape, _s2t_multicast = nvvm.S2TCopyMode.S2T_32x128b_WARPX4
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

                # Same range as the TMA producer above, or the ab ring
                # desynchronises.  scale_d starts False here, so the accumulator
                # is overwritten on the first k-block wherever the range begins.
                # Under THD that means the same PER-GROUP count too: a consumer
                # still counting the kernel-wide tiles would wait for k-blocks
                # the producer never issues.
                _, _, _, _nkt_mma = _thd_group(meta_t, tile_b_mma, n_batch, num_k_tiles)
                if cutlass.const_expr(_THD_TRIM):
                    k_begin, k_end = _causal_k_range(tile_m * cgrp_tile_m_cur, _nkt_mma, _thd_shift(meta_t, tile_b_mma, n_batch))
                else:
                    k_begin, k_end = _causal_k_range(tile_m * cgrp_tile_m_cur, _nkt_mma)
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

                    if cutlass.const_expr(_BLOCK_SCALE):
                        # UTCCP this stage's SF atoms into their TMEM columns right before the MMAs that read them (same issuing
                        # thread, same tcgen05 pipeline: the copies are ordered ahead of the MMAs, and the commit below tracks the
                        # copies' SMEM reads along with the MMAs', so ab_empty frees the SF stage too); then one block-scale MMA per
                        # k-block, each reading its 2 of the atom's 4 scales through the idesc's sf ids.  Elect-gated like the dense
                        # MMA: a tcgen05.cp / tcgen05.mma issues once per executing lane.
                        desc_sfa_stage = desc_sfa_root.advance_start_address(sfa_smem_bytes * stage)
                        desc_sfb_stage = desc_sfb_root.advance_start_address(sfb_smem_bytes * stage)
                        for _bn in cutlass.range_constexpr(num_blocks_n):
                            if elect_one:
                                nvvm.tcgen05_cp(
                                    _s2t_shape,
                                    sfb_block_ptrs[_bn],
                                    desc_sfb_stage.advance_start_address(SF_ATOM_BYTES * _bn),
                                    group=_CTA_GROUP,
                                    multicast=_s2t_multicast,
                                )
                        if elect_one:
                            nvvm.tcgen05_cp(_s2t_shape, sfa_ptr, desc_sfa_stage, group=_CTA_GROUP, multicast=_s2t_multicast)
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            desc_a = desc_a_roots[0].advance_start_address(sA_bytes * stage + a_smem_k_step_bytes * k_block_idx)
                            desc_b = desc_b_roots[0].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                            if elect_one:
                                _tcgen05_mma_block_scale(
                                    mma_block_scale_kind,
                                    _CTA_GROUP,
                                    tmem_addr_mmas[0][0],
                                    desc_a,
                                    desc_b,
                                    idesc_by_k[k_block_idx],
                                    enable_input_d=scale_d,
                                    scale_a=sfa_ptr,
                                    scale_b=sfb_ptr,
                                    scale_vec_size=scale_vec_size,
                                    collector_op=_a_collector_op(0),
                                    b_collector_op=_b_collector_op(0),
                                )
                            scale_d = cutlass.Boolean(True)
                    else:
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
                mma_raw_m = _m_idx >> _preferred_cluster_m_shift
                mma_raw_n = _n_idx >> _preferred_cluster_n_shift
                mma_nt_m = gridx >> _preferred_cluster_m_shift
                mma_nt_n = gridy >> _preferred_cluster_n_shift
                if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                    if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                        mma_raw_m = _m_idx >> _fallback_cluster_m_shift
                        mma_raw_n = _n_idx >> _fallback_cluster_n_shift
                        mma_nt_m = gridx >> _fallback_cluster_m_shift
                        mma_nt_n = gridy >> _fallback_cluster_n_shift
                tile_m, _tile_n_mma = _l2_swizzle_tile(mma_raw_m, mma_raw_n, mma_nt_m, mma_nt_n, swizzle_w, identity=tile_swizzle_n == 1)
                _, tile_b_mma = _decode_bh(_l_idx, n_head)
                cute.arch.fence_proxy("async.shared", space="cta")
                is_valid = vld
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                    nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                tile_iter += 1

            if cutlass.const_expr(USE_PDL):
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
                if cutlass.const_expr(USE_PDL):
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

        # The fp8 arm's launch constants, read ONCE per epilogue warp after the PDL wait (the caller writes them on the
        # stream this launch may overlap): the folded descale of DESCALE / QUANT, the output scale of QUANT, and the
        # per-thread amax accumulator of QUANT (an opaque zero: a folded constant into `fmax_f32`'s inline PTX ICEs libNVVM).
        epi_d = (
            cutlass.Float32(cutlass.make_array_view(epi_descale_0)[0]) * cutlass.Float32(cutlass.make_array_view(epi_descale_1)[0])
            if cutlass.const_expr(epi_mode != EPI_NONE)
            else None
        )
        epi_s = cutlass.Float32(cutlass.make_array_view(epi_scale_out)[0]) if cutlass.const_expr(epi_mode == EPI_QUANT) else None
        _epi_fold_amax = epi_mode == EPI_QUANT and epi_amax is not None
        epi_amax_tile = opaque_f32_zero() if cutlass.const_expr(_epi_fold_amax) else None

        tile_iter = cutlass.Int32(0)
        acc_full_phase_bit = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        tile_h, tile_b = _decode_bh(tile_l, n_head)
        if cutlass.const_expr(_THD_MM):
            # Once per epilogue warp, not per store: the patch launch wrote the
            # array before this kernel started and never rewrites it.  But it is
            # one acquire PER SLOT -- the fence's size operand only accepts 128,
            # i.e. a single descriptor, so acquiring the base would order slot 0
            # alone while this warp goes on to select slot `tile_b`.
            _thd_acquire_descs(desc_words, n_batch)
        # The C-side cu_seqlens prefix inside the metadata buffer: dV/dK write
        # kv rows, dQ writes q rows -- the same choice `_thd_patch_descs_kernel`
        # makes when it bases each sequence's descriptor.
        _thd_meta = cutlass.make_array_view(meta_t) if cutlass.const_expr(_THD_MM) else None
        # C rows are the M-axis tokens = the side the reduction does NOT run over (`_THD_K_IS_KV`: K over kv tokens ->
        # C = q rows at cu_q; K over q tokens -> C = kv rows at cu_k).
        _thd_c_cu0 = (
            (n_batch if cutlass.const_expr(_THD_K_IS_KV) else (cutlass.Int32(2) * n_batch + cutlass.Int32(1)))
            if cutlass.const_expr(_THD_MM)
            else cutlass.Int32(0)
        )
        # The A-side (K) prefix is the OTHER one -- `_thd_group` reduces over the
        # token axis `_THD_K_IS_KV` names, exactly opposite to which axis each
        # GEMM writes.  Used only to detect a zero-length reduction; see
        # `_thd_k_len` in the loop.
        _thd_k_cu0 = (
            ((cutlass.Int32(2) * n_batch + cutlass.Int32(1)) if cutlass.const_expr(_THD_K_IS_KV) else n_batch)
            if cutlass.const_expr(_THD_MM)
            else cutlass.Int32(0)
        )
        is_valid = cutlass.Int32(1)
        clc_full_phase_epi = cutlass.Int32(0)

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

        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)

        while is_valid != 0:
            coord_m_tile = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            coord_n_c = tile_n * cgrp_tile_n_cur + n_rank * pair_n_size
            # A group whose OUTPUT sequence has zero rows. Its clipped C
            # descriptor is built at extent 1 rather than 0 (a tensor map with a
            # zero extent is INVALID and traps -- see tile_dsl.thd.emit_seq_descs),
            # so the hardware clip that drops every other overshooting tile
            # cannot drop this one: skip the store instead. Read once per tile,
            # not per subtile; `tile_b` is refreshed at the bottom of the loop.
            _thd_c_len = (
                cutlass.Int32(_thd_meta[_thd_c_cu0 + tile_b + cutlass.Int32(1)]) - cutlass.Int32(_thd_meta[_thd_c_cu0 + tile_b])
                if cutlass.const_expr(_THD_MM)
                else cutlass.Int32(1)
            )
            # A group whose REDUCTION axis is empty: S_q[b] == 0 for dV/dK,
            # S_kv[b] == 0 for dQ.  `_thd_group` then returns `nkt == 0`, the
            # mainloop runs zero iterations, and `scale_d` starts False -- so no
            # MMA ever wrote the accumulator and the TMEM read below returns
            # whatever the previous tile left there.
            #
            # Unlike `_thd_c_len` this canNOT be fixed by skipping the store:
            # the OUTPUT rows exist (`_thd_c_len > 0`) and the caller expects
            # them written.  Store ZEROS instead, which is also the right
            # answer -- a sequence with no queries contributes nothing to its
            # keys' and values' gradients, and one with no keys has no gradient
            # to receive.
            #
            # A select, not a multiply by zero: the uninitialised TMEM can hold
            # any bit pattern, and `0 * NaN` is NaN (issue #624's rule).
            _thd_k_len = (
                cutlass.Int32(_thd_meta[_thd_k_cu0 + tile_b + cutlass.Int32(1)]) - cutlass.Int32(_thd_meta[_thd_k_cu0 + tile_b])
                if cutlass.const_expr(_THD_MM)
                else cutlass.Int32(1)
            )
            # The trimmed THD arm generalises the test: a tile whose K RANGE is empty -- an empty reduction (nkt == 0), a kv
            # block no query attends, a q pair with no key in its band -- has an unwritten accumulator and a zero gradient.
            # The same `_causal_k_range` call the producer and the MMA warp made for this tile decides it (`_thd_causal_k_range`).
            _thd_store_live = cutlass.Boolean(True)
            if cutlass.const_expr(_THD_TRIM):
                _, _, _, _nkt_epi = _thd_group(meta_t, tile_b, n_batch, num_k_tiles)
                _kb_epi, _ke_epi = _causal_k_range(tile_m * cgrp_tile_m_cur, _nkt_epi, _thd_shift(meta_t, tile_b, n_batch))
                _thd_store_live = _ke_epi > _kb_epi
            # The EPI_QUANT amax fold's gate under THD: this tile's part (its band live / its reduction non-empty -- the store's own
            # predicate above), completed PER ROW at the fold by `row < _thd_c_len`.  Dense needs none: a tile's rows past M are
            # TMA-OOB on A (acc == 0) and fold a harmless 0.  THD differs in two ways, and the per-TILE predicates cover neither:
            #   * every (head, sequence) group walks ALL the M tiles of the ENVELOPE grid (`_host`: `m` is the longest sequence's
            #     extent), so a shorter sequence's spare tiles read A rows that are NOT its own -- the next sequence's blocked rows
            #     (dK: a foreign sequence's live dS), the workspace's slack / capacity tail, or the unwritten q columns past its
            #     pair-rounded tiles (dQ) -- against ITS OWN packed B rows: a finite, plausible, foreign product in rows the C
            #     descriptor clips (GLOBAL_DIM[seq] = the sequence's length), so the store never shows it;
            #   * `_thd_store_live` is the K BAND's emptiness, not the rows' validity: the dense THD arm (CAUSAL_K_NONE) keeps every
            #     tile live, and the trimmed dQ arm keeps a q pair PAST the sequence's rows live (its band ends at `nkt`, never
            #     empty for a late pair), so both would fold those foreign accumulators -- `amax_dQ` / `amax_dK` inflated (a NaN
            #     dropped by `max.f32`, a stale finite product kept) while every stored gradient is exact.
            # So the fold takes exactly the cells the TMA store writes.  A SELECT on the tree's scalar, never a multiply by zero
            # (sdpa-invariants s2: the residue may be NaN); folded out of every dense rendering.
            if cutlass.const_expr(_THD_MM):
                _thd_fold_live = _thd_store_live if cutlass.const_expr(_THD_TRIM) else (_thd_k_len > cutlass.Int32(0))
            if cutlass.const_expr(epi_dp22):
                coord_n_c = coord_n_c + (warp_idx // 2) * epi_cols_per_mma_m

            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1

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
                    linear_idx = tile_b * out_stride_b_0 + tile_h * out_stride_h_0 + row * out_stride_m_0 + col_j * out_stride_n_0

                    if cutlass.const_expr(epi_mode == EPI_NONE):
                        _r_mm = (vec_f32).to(cd_dtype)
                        vec_out = (_r_mm).to(cd_dtype)
                    elif cutlass.const_expr(epi_mode == EPI_DESCALE):
                        # The TRUE-unit gradient: acc * descale_dP * descale_{q|k}, rounded once into bf16.
                        vec_out = (vec_f32 * epi_d).to(cd_dtype)
                    else:
                        # EPI_QUANT: true value -> amax fold (ternary abs-max tree on max.f32, FMNMX3; never cute.math.max) ->
                        # * scale_{dQ|dK} -> the gradient dtype.  Dense: TMA-OOB rows past M carry acc == 0, so no row gate is
                        # needed.  THD: the fold is gated per ROW on the store's validity (`_thd_fold_live`, above) -- the spare
                        # envelope tiles' foreign, clipped rows must not reach amax.
                        _t = vec_f32 * epi_d
                        if cutlass.const_expr(_epi_fold_amax):
                            _t_amax = abs_max_tree([_t[_i] for _i in range(subtile_w)])
                            if cutlass.const_expr(_THD_MM):
                                _row_live = _thd_fold_live & (row < _thd_c_len)
                                _t_amax = cutlass.Float32(arith.select(_row_live.ir_value(), _t_amax.ir_value(), cutlass.Float32(0.0).ir_value()))
                            epi_amax_tile = fmax_f32(epi_amax_tile, _t_amax)
                        vec_out = (_t * epi_s).to(cd_dtype)

                    epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                    _tsv_0 = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape=epi_subtile_elems, dtype=cd_dtype)
                    # The branch is CTA-uniform (it reads only `tile_b`) and it
                    # wraps the store ALONE -- the fence and the named barrier
                    # below stay outside it, so no path through here can diverge
                    # on a sync.
                    if cutlass.const_expr(_THD_TRIM):
                        if _thd_store_live:
                            _tsv_0.data_ptr(tidx * epi_row_elems).store_swizzled(vec_out, alignment=_EPI_ROW_BYTES, swizzle=_EPI_SWIZZLE)
                        else:
                            _tsv_0.data_ptr(tidx * epi_row_elems).store_swizzled(
                                cutlass.full_like(vec_out, 0.0), alignment=_EPI_ROW_BYTES, swizzle=_EPI_SWIZZLE
                            )
                    elif cutlass.const_expr(_THD_MM):
                        if _thd_k_len > cutlass.Int32(0):
                            _tsv_0.data_ptr(tidx * epi_row_elems).store_swizzled(vec_out, alignment=_EPI_ROW_BYTES, swizzle=_EPI_SWIZZLE)
                        else:
                            _tsv_0.data_ptr(tidx * epi_row_elems).store_swizzled(
                                cutlass.full_like(vec_out, 0.0), alignment=_EPI_ROW_BYTES, swizzle=_EPI_SWIZZLE
                            )
                    else:
                        _tsv_0.data_ptr(tidx * epi_row_elems).store_swizzled(vec_out, alignment=_EPI_ROW_BYTES, swizzle=_EPI_SWIZZLE)
                    cute.arch.fence_view_async_shared()
                    nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
                    if warp_idx == 0:
                        if elect_one:
                            # THD stores through THIS SEQUENCE's descriptor: its
                            # GLOBAL_ADDRESS is the sequence's first output row
                            # and its GLOBAL_DIM[seq] is the sequence's length,
                            # so the last M tile -- which overshoots into the
                            # next sequence's rows with a live accumulator
                            # behind it -- is clipped by hardware.  The batch
                            # coordinate is then 0: the descriptor already
                            # carries the base, so the M coordinate stays
                            # sequence-relative.
                            if cutlass.const_expr(_THD_MM):
                                # Skipping only the STORE keeps the epilogue's
                                # pipeline intact: the commit below still runs,
                                # and an empty bulk group commits immediately.
                                if _thd_c_len > cutlass.Int32(0):
                                    nvvm.cp_async_bulk_tensor_global_shared_cta(
                                        (desc_words.iterator.raw_ptr() + tile_b * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic),
                                        _tsv_0.data_ptr(),
                                        (col, coord_m, tile_h, cutlass.Int32(0)),
                                    )
                            else:
                                nvvm.cp_async_bulk_tensor_global_shared_cta(
                                    tma_c_descs[0].get_ptr(),
                                    _tsv_0.data_ptr(),
                                    (col, coord_m, tile_h, tile_b),
                                )
                        if elect_one:
                            nvvm.cp_async_bulk_commit_group()
                        nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)
                    nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)

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
            tile_h, tile_b = _decode_bh(tile_l, n_head)
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)

            tile_iter += 1

        if cutlass.const_expr(_epi_fold_amax):
            # amax_{dQ|dK}: warp butterfly on max.f32, then ONE int32-bit-pattern atomicMax per epilogue warp (non-negative
            # fp32 orders like int32; the caller ZEROES the target on this stream before the launch, the fp8 body's idiom).
            for _sh in cutlass.range_constexpr(5):
                epi_amax_tile = fmax_f32(
                    epi_amax_tile,
                    nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=epi_amax_tile, offset=1 << (4 - _sh), mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY),
                )
            if lane == 0:
                nvvm.atomicrmw(nvvm.AtomicOp.MAX, Pointer(epi_amax.iterator.raw_ptr(), dtype=cutlass.Int32), epi_amax_tile.bitcast(cutlass.Int32))

        if warp_idx == 0:
            nvvm.cp_async_bulk_wait_group(0, read=True)

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)


_bprop_matmul_bh_sm100_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _thd_patch_descs_kernel(
    c_tensor: cute.Tensor,
    base_c_desc: cutlass.GridConstant[_tma.TensorMap],
    base_b_desc: cutlass.GridConstant[_tma.TensorMap],
    desc_words: cute.Tensor,
    meta_t: cute.Tensor,
    n_batch: cutlass.Int32,
    c_row_stride: cutlass.Int64,
) -> None:
    """This GEMM's own THD descriptors, patched from the published metadata.

    Built HERE and not in the shared setup launch because a descriptor's box,
    swizzle and dim ORDER are this kernel's: its operands are ``(n, m, h, b)``,
    so the sequence axis is ``ord=1`` -- stage 2's packed ``(d, head, seq,
    batch)`` operands put it at 2, and a shared builder told the wrong number
    clamps the head extent instead, silently.

    * one C descriptor per sequence, based at that sequence's first output row
      with ``GLOBAL_DIM[seq]`` set to its length, so the overshooting last M
      tile is clipped rather than writing into the next sequence;
    * one B descriptor clamped to the CURRENT packed total, so the last k tile
      reads the caller's unwritten capacity tail as exact zeros.  A declared
      ``max_total_seq_len`` cannot do this job: it is a maximum, while the row
      that must read zero is ``cu_*[B]``, which changes every step.

    Elected warp leaders build disjoint descriptors; each writer publishes to the TMA proxy and
    the kernel boundary orders them before the GEMM reads them.
    """
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    warp = cutlass.Int32(tidx) // cutlass.Int32(32)
    if nvvm.elect_sync():
        meta = cutlass.make_array_view(meta_t)
        cu_q0 = n_batch
        cu_k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
        # C rows are the side the reduction does not run over, B's tokens the side it does (`_THD_K_IS_KV`): on the q-major
        # workspace dV/dK write kv rows and read q tokens and dQ is the mirror; the kv-major workspace flips the pairing.
        c_cu0 = cu_q0 if cutlass.const_expr(_THD_K_IS_KV) else cu_k0
        b_cu0 = cu_k0 if cutlass.const_expr(_THD_K_IS_KV) else cu_q0
        emit_seq_descs(
            base_c_desc,
            desc_words,
            meta,
            c_cu0,
            c_tensor,
            n_batch,
            c_row_stride,
            seq_ord=_THD_MM_SEQ_ORD,
            first_batch=warp,
            batch_step=cutlass.Int32(nthreads) // 32,
        )
        b_total = cutlass.Int32(meta[b_cu0 + n_batch])
        if warp == cutlass.Int32(0):
            emit_clamped_desc(base_b_desc, desc_words, n_batch, b_total, seq_ord=_THD_MM_SEQ_ORD)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )


_thd_patch_descs_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _require_epi_operands(d0, d1, s_out, sfa=None, sfb=None, sf_meta=None) -> None:
    """Trace-time check that the epilogue the rendering carries has its operands (plain Python: the DSL rejects a raise under
    a staged `if`).  A missing scalar would otherwise surface as a None-specialized load deep in the kernel trace.  The
    block-scale arm's two SF tensors are checked the same way (both present iff the rendering block-scales), and its THD leg's
    `sf_meta_t` (present iff the rendering block-scales AND is THD: a dense block-scale or a plain THD rendering refuses it)."""
    if _BLOCK_SCALE and (sfa is None or sfb is None or (_THD_MM and sf_meta is None)):
        raise TypeError(
            f"{__name__}: block_scale needs sfa_0 (the A operand's F8_128x4 SF atoms) and sfb_0 (the columnwise B SF)"
            f"{' and, under THD, sf_meta_t (the int32 per-sequence SF tile prefixes [cu_sf_q(B+1) | cu_sf_k(B+1)])' if _THD_MM else ''}; "
            f"got sfa_0={'None' if sfa is None else 'set'}, sfb_0={'None' if sfb is None else 'set'}, sf_meta_t={'None' if sf_meta is None else 'set'}"
        )
    if not _BLOCK_SCALE and (sfa is not None or sfb is not None or sf_meta is not None):
        raise TypeError(f"{__name__}: this rendering does not block-scale (block_scale=False) but SF operands were passed")
    if sf_meta is not None and not _THD_MM:
        raise TypeError(f"{__name__}: sf_meta_t (the per-sequence SF tile prefixes) is the THD leg's operand; this rendering is dense (thd_varlen=False)")
    if epi_mode != EPI_NONE and (d0 is None or d1 is None):
        raise TypeError(f"{__name__}: epi_mode={epi_mode} needs epi_descale_0 and epi_descale_1 (fp32 [1] device tensors); got None")
    if epi_mode == EPI_QUANT and s_out is None:
        raise TypeError(f"{__name__}: EPI_QUANT needs epi_scale_out (the gradient's fp32 [1] scale); got None")
    if epi_mode == EPI_NONE and (d0 is not None or d1 is not None or s_out is not None):
        raise TypeError(f"{__name__}: this rendering has no epilogue (EPI_NONE) but epilogue operands were passed")


@cute.jit
def _host(
    problem_size: tuple,
    a_0: cute.Tensor,
    b_0: cute.Tensor,
    c_0: cute.Tensor,
    # Dense: 1-element dummies.  THD: the setup launch's metadata and this
    # GEMM's descriptor scratch.
    meta_t: cute.Tensor,
    desc_words: cute.Tensor,
    stream: _cuda.CUstream,
    # The fp8 arm's epilogue operands, TRAILING and defaulted so the SM100 chain's positional 7-argument call is unchanged:
    # fp32 [1] device tensors descale_dP (0) and descale_{q|k} (1) for EPI_DESCALE / EPI_QUANT, the output scale and the
    # amax target for EPI_QUANT (None = the graph left that amax virtual; its fold and atomic fold out).
    epi_descale_0: Optional[cute.Tensor] = None,
    epi_descale_1: Optional[cute.Tensor] = None,
    epi_scale_out: Optional[cute.Tensor] = None,
    epi_amax: Optional[cute.Tensor] = None,
    # The block-scale arm's scale factors (TRAILING, defaulted: every earlier call shape is unchanged), uint8 5-D views whose
    # strides are in BYTES:
    #   sfa_0  (512 B atom, K tiles, M tiles, H, B)   over the A operand's F8_128x4 atoms [B, H, M/128, K/128, 512]
    #   sfb_0  (512 B atom, D planes, K tiles, H, B)  over the columnwise B SF, D-plane-major (plane stride = B*H*tiles atoms)
    #          -- or, under THD, the PACKED per-sequence-tile view `_sf_planes_view_thd` (plane stride one atom, B = 1).
    #          Its H is B's OWN head extent, `n_head // b_head_group`: under a grouped dQ record the host hands the view windowed
    #          to the kv heads and the SFB coordinate takes `tile_h_b = _b_head(tile_h)` exactly like B's (dense and THD alike),
    #          so one launch covers a whole head chunk; the tensor map is sized from this view's shape.
    sfa_0: Optional[cute.Tensor] = None,
    sfb_0: Optional[cute.Tensor] = None,
    # The block-scale arm's THD leg (TRAILING, defaulted): the int32 `[cu_sf_q(B+1) | cu_sf_k(B+1)]` per-sequence SF TILE prefixes
    # (`config_sm100.STAGE3_THD_SF_META_WORDS(B)` words, `STAGE3_THD_SF_CU_*_OFF`) the TMA warp's SFB coordinate takes its tile
    # base from; required iff the rendering block-scales under `thd_varlen`, refused otherwise.
    sf_meta_t: Optional[cute.Tensor] = None,
) -> None:
    _require_epi_operands(epi_descale_0, epi_descale_1, epi_scale_out, sfa_0, sfb_0, sf_meta_t)
    _a_operands = [a_0]
    _b_operands = [b_0]
    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    # The 2-D batch: (n_head, n_batch) where upstream carries one flat `batch`,
    # and FOUR strides per operand (m/n, k, h, b) where upstream carries three.
    # `batch` stays as the product because the CLC scheduler and the grid still
    # rasterize one flat axis; only the TMA coordinates are 2-D.
    n_head = problem_size[3]
    n_batch = problem_size[4]
    batch = n_head * n_batch
    _stride_idx = 5
    _a_stride_sets = []
    for _ in cutlass.range_constexpr(num_a_operands):
        _a_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
                problem_size[_stride_idx + 3],
            )
        )
        _stride_idx += 4
    _b_stride_sets = []
    for _ in cutlass.range_constexpr(num_b_operands):
        _b_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
                problem_size[_stride_idx + 3],
            )
        )
        _stride_idx += 4
    out_stride_m_0 = problem_size[_stride_idx]
    out_stride_n_0 = problem_size[_stride_idx + 1]
    out_stride_h_0 = problem_size[_stride_idx + 2]
    out_stride_b_0 = problem_size[_stride_idx + 3]
    _stride_idx += 4
    # B's K extent, separate from A's.  They coincide on the dense path, but
    # under THD A's K axis is the BLOCKED workspace (rows padded per sequence)
    # while B's is the PACKED tokens -- different lengths for the same logical
    # reduction, so one shared symbol cannot describe both.
    k_b = problem_size[_stride_idx]
    # A's and C's M extents, likewise separate.  Dense passes all three equal;
    # THD does not: for dV/dK the A operand's M is the workspace's uniform kv
    # column count while C's is the PACKED output rows, and `m` itself is only
    # the grid's M -- the longest sequence, which every group's tiles cover and
    # a shorter one's spare tiles are clipped out of.
    m_a = problem_size[_stride_idx + 1]
    m_c = problem_size[_stride_idx + 2]
    _stride_idx += 3

    # A broadcast operand collapses BOTH batch extents, not just one.
    if cutlass.const_expr(matmul_a_batch == 1):
        a_h, a_b = 1, 1
    else:
        a_h, a_b = n_head, n_batch
    if cutlass.const_expr(matmul_b_batch == 1):
        b_h, b_b = 1, 1
    else:
        # B's head extent follows its head group: `n_head // b_head_group` B heads serve `n_head` A/C heads (every A/C head
        # `h` reads B head `h // b_head_group`, `_b_head`); `n_head` itself at the default 1.
        b_h = n_head if cutlass.const_expr(b_head_group == 1) else n_head // b_head_group
        b_b = n_batch
    # THD: `n_batch` is the SEQUENCE count -- it sizes the grid and indexes the
    # metadata -- but the packed operands hold ONE batch element, reached by the
    # coordinate offsets instead.  Describing them as n_batch-deep builds a
    # tensor map over memory that is not there, which fails inside
    # cuTensorMapEncodeTiled as an abort rather than an exception.
    if cutlass.const_expr(_THD_MM):
        a_b = 1
        b_b = 1
    c_batch = 1 if cutlass.const_expr(_THD_MM) else n_batch

    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_h, a_stride_b = _a_stride_sets[_a_idx]
        if cutlass.const_expr(a_is_m_major):
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[m_a, k_sym, a_h, a_b],
                    global_strides=[
                        a_stride_k * ab_dtype.width // 128,
                        a_stride_h * ab_dtype.width // 128,
                        a_stride_b * ab_dtype.width // 128,
                    ],
                    box_dims=[a_tma_group_elems, cta_tile_mnk[2], 1, 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_sym, m_a, a_h, a_b],
                    global_strides=[
                        a_stride_m * ab_dtype.width // 128,
                        a_stride_h * ab_dtype.width // 128,
                        a_stride_b * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[0] // a_mcast_slices, 1, 1],
                    swizzle=ab_tma_swizzle,
                )
            )
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_h, b_stride_b = _b_stride_sets[_b_idx]
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[n, k_b, b_h, b_b],
                    global_strides=[
                        b_stride_k * ab_dtype.width // 128,
                        b_stride_h * ab_dtype.width // 128,
                        b_stride_b * ab_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1, 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_b, n, b_h, b_b],
                    global_strides=[
                        b_stride_n * ab_dtype.width // 128,
                        b_stride_h * ab_dtype.width // 128,
                        b_stride_b * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1, 1],
                    swizzle=ab_tma_swizzle,
                )
            )

    _tma_c_outputs = [c_0]
    _c0 = _tma_c_outputs[0]
    tma_c_desc_0 = _tma.create_tensor_map_tiled(
        global_address=_c0.iterator.toint(),
        dtype=cd_dtype,
        global_dims=[n, m_c, n_head, c_batch],
        global_strides=[
            # `cd_dtype.width`, not a literal 16: the A/B descriptors above already
            # derive it, and this one is now dtype-parameterized too.
            out_stride_m_0 * cd_dtype.width // 128,
            out_stride_h_0 * cd_dtype.width // 128,
            out_stride_b_0 * cd_dtype.width // 128,
        ],
        box_dims=[epi_row_elems, epi_tile_mn[0], 1, 1],
        swizzle=_EPI_TMA_SWIZZLE,
    )
    tma_c_desc_list = [tma_c_desc_0]

    if cutlass.const_expr(_BLOCK_SCALE):
        # SF tensor maps -- the 512-B atom is the inner box (read as 256 x u16, the block-scale GEMM template's spelling: a
        # uint8 inner extent is capped at 256 elements), one atom per (K tile, M tile) for SFA, the num_blocks_n D-plane
        # atoms of one K tile for SFB.  Strides are the caller's BYTE strides in TMA's 16-byte units; the (h, b) pair is two
        # coordinates like the operand maps (a head window is a base offset plus the h stride, never a flat b*H + h).
        _sf_atom_u16 = SF_ATOM_BYTES // 2
        tma_sfa_desc_0 = _tma.create_tensor_map_tiled(
            global_address=sfa_0.iterator.toint(),
            dtype=cutlass.Uint16,
            global_dims=[_sf_atom_u16, sfa_0.shape[1], sfa_0.shape[2], sfa_0.shape[3], sfa_0.shape[4]],
            global_strides=[
                cutlass.Int64(sfa_0.stride[1]) // 16,
                cutlass.Int64(sfa_0.stride[2]) // 16,
                cutlass.Int64(sfa_0.stride[3]) // 16,
                cutlass.Int64(sfa_0.stride[4]) // 16,
            ],
            box_dims=[_sf_atom_u16, 1, 1, 1, 1],
            swizzle=_tma.TensorMapSwizzle.none,
        )
        tma_sfb_desc_0 = _tma.create_tensor_map_tiled(
            global_address=sfb_0.iterator.toint(),
            dtype=cutlass.Uint16,
            global_dims=[_sf_atom_u16, sfb_0.shape[1], sfb_0.shape[2], sfb_0.shape[3], sfb_0.shape[4]],
            global_strides=[
                cutlass.Int64(sfb_0.stride[1]) // 16,
                cutlass.Int64(sfb_0.stride[2]) // 16,
                cutlass.Int64(sfb_0.stride[3]) // 16,
                cutlass.Int64(sfb_0.stride[4]) // 16,
            ],
            box_dims=[_sf_atom_u16, num_blocks_n, 1, 1, 1],
            swizzle=_tma.TensorMapSwizzle.none,
        )
    else:
        tma_sfa_desc_0 = None
        tma_sfb_desc_0 = None

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cgrp_tile_m = cgrp_tile_mnk[0]
    cgrp_tile_n = cgrp_tile_mnk[1]
    num_tile_m_host = (m + cgrp_tile_m - 1) // cgrp_tile_m
    num_tile_n_host = (n + cgrp_tile_n - 1) // cgrp_tile_n
    grid_x = num_tile_m_host * cluster_m
    grid_y = num_tile_n_host * cluster_n
    grid_shape = (grid_x, grid_y, batch)
    if cutlass.const_expr(_THD_MM):
        # Ahead of the GEMM on the same stream; kernel-boundary ordering is what
        # makes the patched descriptors visible to it.
        _thd_patch_descs_kernel(
            _c0,
            tma_c_desc_0,
            tma_b_desc_list[0],
            desc_words,
            meta_t,
            n_batch,
            out_stride_m_0,
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)

    launch = _bprop_matmul_bh_sm100_kernel(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        tma_a_desc_list[0],
        tma_b_desc_list[0],
        out_stride_m_0,
        out_stride_n_0,
        out_stride_h_0,
        out_stride_b_0,
        n_head,
        tma_c_desc_list[0],
        meta_t,
        desc_words,
        n_batch,
        epi_descale_0,
        epi_descale_1,
        epi_scale_out,
        epi_amax,
        sf_meta_t,
        tma_sfa_desc_0,
        tma_sfb_desc_0,
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
