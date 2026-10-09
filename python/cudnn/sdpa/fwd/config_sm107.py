# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel configuration for the Frost SM107 (Rubin) DSL SDPA-forward flavors.

**Why this is a separate module from** :mod:`cudnn.sdpa.fwd.config_sm100`, and
not a widening of it: the Rubin kernels are a different kernel *lineage*, not
the Blackwell kernels recompiled. They implement different pipelines, so they
need different warp populations, stage depths and MMA k-steps. The two module's
flavors overlap in NAME (``d128`` / ``d256`` / ``d512``) and in almost nothing
else, and the overlap is what makes sharing dangerous rather than economical.

That is not a theoretical concern — it cost a debugging session. The SM107
``d256`` per-tensor FP8 kernel was routed through ``make_cfg_d256`` (Blackwell),
which derives its whole warp layout from a *split-P* switch that only the
Blackwell d256 kernel implements::

    SOFTMAX_WARPGROUPS   rubin-body=1     blackwell-cfg=2
    TOTAL_WARPS          rubin-body=12    blackwell-cfg=16
    THREADS_PER_CTA      rubin-body=384   blackwell-cfg=512
    MMA_WARP_ID          rubin-body=8     blackwell-cfg=12
    READ_TILE_ARRIVERS   rubin-body=11    blackwell-cfg=15

The last line is the fatal one: ``READ_TILE_ARRIVERS`` is the init count of the
scheduler's tile-handoff mbarrier, so a count of 15 against 11 actual arrivers
can never be reached and the *first* tile handoff never completes. The kernel
hung at every shape, down to ``B=1 H=1 S=128`` — 100 % GPU, no fault, no wrong
number, nothing for a sanitizer to find. Keeping Rubin's geometry in its own
module is what makes that class of mismatch impossible rather than merely
unlikely.

Provenance: every value here is the one the Rubin kernel BODIES were written and
validated against (the pre-upstream ``sdpa_config_{llama,dsv3,qwen,
dsv4}`` flavors, keyed here by op geometry per the engine contract, not by model
name). ``test_sdpa_fwd_config_sm107.py`` pins the ones a kernel body would
deadlock or corrupt on if they drifted.

Shared with SM100 by design: :class:`TemplateParams` (the engine-contract record
the loader injects) and the mask / dtype / scheduler vocabulary. Those are the
interface; everything below is the Rubin geometry behind it.

All configuration errors raise :class:`ValueError`. Anything a *user-built
graph* could trip must be rejected earlier, by ``graph_analyzer.probe`` /
``api_dsl.SdpaFwdDslSm100.check_support`` — a ``ValueError`` from here means
those support checks have a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclasses_replace
from typing import Optional, Tuple

from cudnn.frost.tile_dsl.constants import (
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)

# The engine-contract record is deliberately SHARED: one lowering builds it, and
# an arch-specific copy would let the two drift in ways the loader cannot see.
from cudnn.sdpa.fwd.config_sm100 import TemplateParams, bshd_compact, bshd_zero_copy_stride

# The d512 2x2-DATAPATH record is shared with the SM100 line too (CfgD512X2 + its arch-neutral geometry checks); this
# module adds only the Rubin arch facts to it (3-deep sub-chunk rings, DESC_VERSION derived from the layout, the 320 KiB
# usable budget) -- see make_cfg_d512_2x2 below.
from cudnn.sdpa.fwd.config_sm100 import (
    CfgD512X2,
    _pack_g,
    _validate_params as _validate_params_sm100,
    d512_2x2_geometry_checks,
    d512_2x2_p_ring_start_bytes,
    d512_2x2_smem_bytes,
)

__all__ = [
    "TemplateParams",
    "resolve_dtype_o",
    "TmaIters",
    "CfgD32",
    "CfgD64",
    "CfgD128",
    "CfgD192",
    "CfgD256",
    "CfgD512",
    "CfgD512X2",
    "make_cfg_d32",
    "make_cfg_d64",
    "make_cfg_d128",
    "make_cfg_d128_mxfp8",
    "make_cfg_d192",
    "make_cfg_d192_mxfp8",
    "make_cfg_d256",
    "make_cfg_d256_mxfp8",
    "make_cfg_d512",
    "make_cfg_d512_mxfp8",
    "make_cfg_d512_2x2",
    "SMEM_CAP_BYTES",
    "SMEM_USABLE_BYTES",
    "SM107_FP8_THD_SHAPES",
    "SM107_F16_THD_SHAPES",
    "SM107_EPILOGUE_GATE_SHAPES",
    "SM107_SCORE_STATS_SHAPES",
    "SM107_HEAD_DIM_TILE_WIDTHS",
    "SM107_F16_SMALL_D_SHAPES",
    "SM107_SMALL_D_CGAS",
    "SMALL_D_STAGES_KV",
    "SMALL_D_TILE_MS",
    "sm107_f16_tile_width",
    "sm107_small_tile_leg",
    "D128FamilyTmemLayout",
    "d128_family_tmem_layout",
    "epilogue_gate_layout_declarable",
    "TCGEN05_V0_ADDR_LIMIT",
    "d256_mxfp8_sf_smem_bytes",
    "d256_mxfp8_last_sf_tile_start",
]


# ---------------------------------------------------------------------------
# Rubin architecture constants
# ---------------------------------------------------------------------------

# Rubin SM10.7 oversized SMEM cap.  Reaching past the standard 227 KiB requires
# the launcher to set ALLOW_OVERSIZED_SHARED_MEMORY; the d128 FP8 9-stage ring
# and the d256/d512 flavors all depend on it.
SMEM_CAP_BYTES = 327 * 1024

# ...but the CAPACITY is not the budget. The shipped sm107/prefill_d128_fp8.py
# sizes its own guard against 320 KiB ("327 KiB capacity minus reserves"), and
# the kernels additionally spend ~2 KiB on barriers, the scheduler ring and the
# TMEM pointer. Validating Q/K/V/O against the raw 327 KiB waves through an
# allocation that silently overflows: d192 f16 at STAGES_KV=4 sums to exactly
# 320 KiB of Q/K/V/O, passed a 327 KiB check, and produced an output that was
# 50 % ZEROS with a numerically EXACT LSE (BMM1 and the softmax are fine; only
# the clobbered O half is wrong). Check against this, not the capacity.
SMEM_USABLE_BYTES = 320 * 1024
_SMEM_FIXED_OVERHEAD = 2 * 1024  # barriers + scheduler + tmem-ptr slack

# TMEM columns.  Rubin raises the Blackwell 512 to 576, which needs
# `tmem_alloc(..., is_exclusive=True)` (see tile_dsl/tmem.py).
TMEM_TOTAL_COLS = 576

_DTYPE_E4M3, _DTYPE_E5M2, _DTYPE_BF16, _DTYPE_FP16 = 0, 1, 2, 3
# Output-only block-scaled codes (per-tensor FP8 d128 epilogue; see config_sm100):
# 4 = E2M1 data + E4M3 scale per 16 d, 5 = E4M3 data + UE8M0 scale per 32 d.
_DTYPE_O_NVFP4, _DTYPE_O_MXFP8 = 4, 5
_O_BLOCK_SCALE_BY_DTYPE = {_DTYPE_O_NVFP4: 16, _DTYPE_O_MXFP8: 32}

# Head-dim shapes whose Rubin PER-TENSOR FP8 kernel carries the THD/varlen leg.
#
# ONE definition, consumed by BOTH the engine row (`engines._sm100_fp8_spec`'s
# ``thd_d_shapes`` on the Rubin arm) and the standalone adapter's THD gate
# (``api_dsl.SdpaFwdDslSm100.check_support``).  They are two enforcement points
# for one fact, and contract rule 8b' exists because keeping two copies in step
# by hand does not work: widen only the row and a graph enters the ranked list
# then dies with a bare NotImplementedError in check_support; widen only the
# wrapper and the row still declines.  Sharing the constant makes disagreement
# unrepresentable rather than merely tested.
#
# Membership rule: the shape's body must carry the FROST THD contract -- the
# 14-arg build_thd_meta_o_descs_kernel, 4B+4 metadata, (b+3) O-descriptor slots,
# the persistent claim-counter scheduler, the dead-unit O-store guard and the
# packed-total-clamped runtime K/V descriptors.
#
# All four qualify as of 2026-09-09.  d192xd128 came free -- it IS the shipped
# d128 body, only the config factory differs -- and d256 / d512 were moved onto
# the contract (frost_dev/port_thd_contract.py); they previously called the
# setup kernel with the pre-upstream 7-arg signature and allocated 3B+2 where
# the shared decode reads 4B+4.  Confirmed on w2u1g-lc-0030: 43 passed / 0
# failed across the per-tensor FP8 THD suite.
SM107_FP8_THD_SHAPES = frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})

# f16/bf16 flavor names whose kernel body HAS been ported to the FROST
# setup-kernel contract (the 14-arg build_thd_meta_o_descs_kernel + the 4B+4
# metadata the shared decode reads).  Keyed by the `flavor` string
# `_validate_params` already receives, so adding a ported flavor is one entry.
_F16_THD_FLAVORS = frozenset({"sm107 d128", "sm107 d192xd128", "sm107 d256", "sm107 d512"})

# Head-dim shapes whose Rubin f16/bf16 kernel carries the THD/varlen leg -- the
# same one-definition-two-consumers arrangement as SM107_FP8_THD_SHAPES above
# (engine row + standalone adapter gate; contract rule 8b').  Must stay in step
# with _F16_THD_FLAVORS, which is the same fact keyed by config-flavor name.
SM107_F16_THD_SHAPES = frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})

# Head-dim shapes whose Rubin kernels carry the FUSED EPILOGUE GATE
# (O := O * sigmoid(G), TemplateParams.epilogue_gate).  ONE named constant with
# three consumers -- the engine rows' ``epilogue_gate_d_shapes``, the standalone
# adapter's rule-8b twin in ``api_dsl.SdpaFwdDslSm100.check_support`` and the
# gated-attention block's geometry pin -- so the three cannot drift (rule 8b').
# EXACT dims: the envelope flavor that pads d=200 up to (256, 256) is NOT
# claimed (the gate tile is TMA'd at the kernel's TILE_O, and nothing has
# validated a padded G).  Membership rule: the flavor's f16/bf16, per-tensor
# FP8 AND block-scale MXFP8 bodies all carry the gate seams (sGate SmemTile,
# mb_gate_full/empty, the TMA-LDG issue after the KV loop, the packed tanh
# epilogue).  The MXFP8 kernel stages a bf16 G like the FP8 one and writes a
# gated e4m3 O UNSCALED (it has no per-tensor scale_o: block scales dequantize
# in-MMA), and its gate tile is declared AFTER the scale-factor slabs so the
# version-0 SF descriptors stay under 256 KiB (see d256_mxfp8_last_sf_tile_start).
SM107_EPILOGUE_GATE_SHAPES = frozenset({(256, 256)})

# Native half flavors whose Rubin PREFILL kernel stores the score side outputs (``score_max`` /
# ``score_sum_exp``, see config_sm100.SM100_SCORE_STATS_SHAPES for the contract): the d128 file
# (``sm107/prefill_d128_f16.py``) from its correction epilogue.  The engine row's
# ``score_stats_d_shapes`` and the adapter's check_support read THIS set (rule 8b').
SM107_SCORE_STATS_SHAPES = frozenset({(128, 128)})

# Head-dim TILE widths of the Rubin f16/bf16 d128-family prefill kernel (``sm107/prefill_d128_f16.py`` at
# ``TemplateParams.d_flavor``): ``TILE_K = TILE_O`` in {32, 64, 128} is a Cfg field, not a fork.  A graph's head dim rides
# the SMALLEST width covering max(d_qk, d_v): the TMA load box is the tile while the tensor extent is D, so the load
# zero-fills the missing columns (exact zero terms in every QK^T dot product and in P.V's padded V columns), BMM1 runs
# TILE_K / 16 k-steps instead of 8, BMM2's N is TILE_O, the O store clips at the true d_v, and the expect_tx byte count is
# the FULL box (an overshooting TMA box delivers every byte of the box, zero-filled).  D % 8 == 0 is the TMA 16-byte
# global-stride rule at 2 B/elem.  The two small shapes are the ROW's native flavors next to d128 (one constant for the
# engine row's d_shapes / cga domain, the adapter's kernel-file map and the tests -- rule 8b'); D > 128 belongs to the
# d192 / d256 / d512 rows.
#
# The small tiles carry the row's DENSE leg only (causal / bottom-right / right band / SWA / padded / sink / GQA / Stats, at
# the cga widths of SM107_SMALL_D_CGAS).  A graph that asks for a leg beyond it -- THD, paged KV, split-KV, PackGQA, the
# score side outputs (``sm107_small_tile_leg``) -- or for a cga outside the tile's domain is NOT declined: the pick walks
# PAST the small tiles to the next covering width, i.e. the d128 flavor the graph rode before the tiles existed (zero-filled
# to 128, where those legs are wired and validated), so nothing the row served before the tiles is withdrawn.  The engine
# row says it as ``Capabilities.dense_only_d_shapes``; the adapter's flavor pool and ``sm107_f16_tile_width(leg=, cga=)``
# are the twins (rule 8b').  Serving a leg natively on a small tile = validating it there and taking the tile out of that set.
SM107_HEAD_DIM_TILE_WIDTHS = (32, 64, 128)
SM107_F16_SMALL_D_SHAPES = frozenset({(32, 32), (64, 64)})
# cga widths the small tiles are VALIDATED at (the engine row's ``cgas_by_d_shape`` reads this; the factory refuses a width
# outside it).  cga1 is the default: the narrow slabs need no collective MMA to halve K/V, and a 512-row cga2 cluster wastes
# 25-50 % of its Q rows at the S = 256-384 crops these tiles exist for.  At TILE_O = 32 a cga2 V operand would be 16 columns
# = a 32-byte MN-major swizzle atom, which no shipped kernel runs -- declined until measured (a cga2 request at D <= 32 lands
# on the 64-wide tile, the smallest covering one validated at that width).
SM107_SMALL_D_CGAS = {(32, 32): frozenset({1}), (64, 64): frozenset({1, 2})}


def sm107_small_tile_leg(
    *, thd: bool = False, paged: bool = False, split_kv: Optional[int] = 1, pack_gqa: Optional[bool] = False, score_stats: bool = False
) -> Optional[str]:
    """The leg a graph asks for BEYOND the dense one the Rubin small head-dim tiles carry -- ``"THD"``, ``"paged KV"``,
    ``"split-KV"``, ``"PackGQA"`` or ``"score side outputs"`` -- or ``None`` for a graph on the dense leg.  A non-None leg
    routes the graph past the small tiles onto the d128 flavor (the pre-tile route); the engine row spells the same predicate
    on its graph facts and knobs (``engines._selected_d_shape`` over ``Capabilities.dense_only_d_shapes``), the adapter calls
    this one, and the tests assert the two agree on every (D, leg) cell (rule 8b')."""
    if thd:
        return "THD"
    if paged:
        return "paged KV"
    if (split_kv or 1) > 1:
        return "split-KV"
    if pack_gqa:
        return "PackGQA"
    if score_stats:
        return "score side outputs"
    return None


def sm107_f16_tile_width(d_qk: int, d_v: int, *, leg: Optional[str] = None, cga: Optional[int] = None) -> Optional[int]:
    """The head-dim tile width the Rubin f16/bf16 row serves ``(d_qk, d_v)`` on: the smallest of
    :data:`SM107_HEAD_DIM_TILE_WIDTHS` covering both -- 8 / 16 / 24 -> 32, 40 / 48 / 56 -> 64, 72..120 -> 128, and 32 / 64 /
    128 native -- or ``None`` when no width covers them (D > 128: the d192 / d256 / d512 rows) or a dim is not a multiple of 8
    (the TMA 16-byte global-stride rule at 2 B/elem: a typed decline, never a padded copy).

    ``leg`` (:func:`sm107_small_tile_leg`) and ``cga`` make the pick feature-aware: a small tile is taken only for a graph on
    the dense leg (``leg is None``) at a cga inside :data:`SM107_SMALL_D_CGAS` (or unset); otherwise the walk continues to the
    next covering width -- a split-KV / score-output graph at D <= 64 lands on 128 (its pre-tile route), a cga2 request at
    D <= 32 on 64.  The engine row (``d_shapes`` + ``dense_only_d_shapes``, the envelope walk of ``engines._selected_d_shape``)
    and the adapter (``api_dsl._pick_flavor`` over its Rubin pool) implement the SAME pick; the tests assert the three agree for
    every served D and every leg (rule 8b')."""
    d = max(int(d_qk), int(d_v))
    if min(int(d_qk), int(d_v)) <= 0 or int(d_qk) % 8 or int(d_v) % 8:
        return None
    for width in SM107_HEAD_DIM_TILE_WIDTHS:
        if d <= width:
            small = (width, width) in SM107_F16_SMALL_D_SHAPES
            if small and (leg is not None or (cga is not None and cga not in SM107_SMALL_D_CGAS[(width, width)])):
                continue
            return width
    return None


# ---------------------------------------------------------------------------
# Additive bias (resident row-block) + per-key boolean mask on the d128-file tiles
# ---------------------------------------------------------------------------
#
# ``TemplateParams.bias_mode`` / ``kv_mask_form`` (config_sm100): the sm107 f16/bf16 d128 FILE carries the graph's additive
# bias as a RESIDENT SMEM row-block -- the TMA-LDG warp fills ``BIAS_BLOCK_ROWS = TILES_Q * TILE_M`` bias rows x the work
# item's whole KV range once per item, the softmax warps read their row slice with 128-B-swizzled lane loads and add it
# pre-scale (``s + bias * inv_scale``, one FFMA2 per pair) before the band mask and the software row max -- and the per-key
# boolean ``kv_mask`` as one 32-bit keep word per 32 kv columns (a warp ballot over byte loads), folded into the bit-word mask.
# The block is sized from what the carveout leaves after Q / O / K / V at the flavor's ring depth, in whole TILE_N-column
# tiles; a graph whose KV range needs more tiles than fit takes the STREAMED form (``bias_mode = 2``): the TMA-LDG warp
# streams ONE ``BIAS_BLOCK_ROWS x TILE_N`` bias tile per kv tile through a ``STAGES_BIAS``-deep SMEM ring (2 / 3 stages, the
# deepest the carveout holds) that the softmax warps release per kv tile -- same swizzle, boxes and lane loads, any S_kv.  A
# width where neither form fits (d128 at either cga; the one-batch-entry item at the 64-wide tile's cga1 above 128) is a
# TYPED decline on the row AND in the adapter.  The kv_mask ALONE stages nothing in SMEM (byte loads + warp ballots per kv
# tile) -- no block, no S_kv budget: served at any S_kv.
BIAS_MODE_NONE, BIAS_MODE_RESIDENT, BIAS_MODE_STREAMED = 0, 1, 2
BIAS_MODES = frozenset({BIAS_MODE_NONE, BIAS_MODE_RESIDENT, BIAS_MODE_STREAMED})
# The streamed ring's depth domain: 2 or 3 stages (plan: "a 2-3-stage bias ring").  Deeper buys nothing the K/V ring's own
# lookahead does not already bound (the pair item streams 2 kv tiles ahead); shallower than 2 is no ring.
BIAS_RING_STAGES = (2, 3)
# ``kv_mask`` forms (TemplateParams.kv_mask_form): 1 = the BOOLEAN mask as a bit word (a masked key is -inf, a fully masked row
# reads O = 0 / Stats = -inf); 2 / 3 = the ADDITIVE per-key vector in the io dtype (bf16 / fp16 pairs) / in fp32, folded into the
# raw scores pre-scale like the bias (``s + v * inv_scale``, one packed FFMA per pair) from a warp-uniform 16-byte GMEM load --
# the models' finite -1e9 fold preserved exactly (a fully -1e9 row is the uniform average, NOT the empty-row contract).  Neither
# additive form stages anything in SMEM or adds a barrier row: the bool word's byte loads become 16-byte granule loads.
KV_MASK_NONE, KV_MASK_BOOL_WORD, KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32 = 0, 1, 2, 3
KV_MASK_FORMS = frozenset({KV_MASK_NONE, KV_MASK_BOOL_WORD, KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32})
# The additive vector's extent and strides are whole 16-byte granules (8 halves / 4 floats; the 16-bit form is the stricter one,
# applied to both so the rule reads the same for every dtype): the tail granule is CLAMPED to the last in-bounds one in-kernel.
KV_MASK_ADDITIVE_GRANULE_ELEMS = 8
# The KV tile width (TemplateParams.tile_n) each native flavor of the sm107 f16/bf16 row renders: the small tiles take 64 or 128
# (the one-chunk arm the softmax body carries, validated and trace-compiled at both), the d128 / d192 / d256 / d512 renderings keep
# 128; 256 is refused by _validate_tile_n at two sub-tiles (2 x 256 S columns + the O slots + STATS exceed the 576-column TMEM).
# The engine row's per-shape knob domain (Capabilities.tile_ns_by_d_shape) and the adapter's check_support read this ONE constant.
SM107_TILE_NS_BY_D_SHAPE = (((32, 32), frozenset({64, 128})), ((64, 64), frozenset({64, 128})))
# The KV tile width an fp32 ADDITIVE kv_mask vector TOGETHER WITH a bias is served at on the small tiles: 64, and only 64.  The
# 128-wide rendering of that pair spills on the softmax role (MEASURED, sm_107a, cutlass-dsl 4.8.0: STL / LDL 23 / 27 causal, 15 / 15
# padded, 16 / 16 at d64 cga2 -- the sixteen 16-byte granules of a 64-column fp32 chunk are 64 registers beside the second chunk's
# 128 live scores; BIAS_PRELOAD_CHUNKS = 0 leaves the counts unchanged, a streamed granule fold makes them worse), where the io-dtype
# vector packs two values per register and stays at the bit word's count and the 64-wide tile (64 live scores) fits again.  Read by
# the heuristics' pick, the engine row's gate (a requested 128 is a typed decline, never a degraded plan) and the adapter's default.
SM107_FP32_VECTOR_WITH_BIAS_TILE_N = 64
# The KV tile width the small tiles RUN a graph at when no knob asks for one: the MEASURED table of the knob sweep (bf16, D 32 and
# 64, S_q = S_kv = S in 128..2048 and the padding leg 200..600, B x S flattened rows at B 1 / 4, the bias batch-broadcast; a 212-SM
# cc 10.7 node at the 2376 MHz lock, 100 x 3 graph-replay rounds with a control twin, every ratio same-process, the kernel-only
# CUPTI time where the replay is launch-bound).  The half tile pays where the 128-wide body is over the instruction-cache cliff
# (a bias AND a kv_mask: +9..+55 % at every D / B / S measured through S 768 and d64 through S 2048; at d32 S 1024 / 2048 B = 1 the
# pair ring at the capped chunk is +2.4 / +1.4 % while the wave-quantised n32 / n64 rings tied or lost), at one or two kv tiles with a bias
# (nothing for the wider tile to amortise: +7..+37 %), and on the padded dense / bias body whose 128-wide rendering is
# pathologically slow (S_kv % 128 != 0: plain +16..+68 % at d32, +36 % at d64 S 200; bias +17..+38 %) -- a workaround for a
# code-generation effect of the padded dense arm, to retire with its fix.  Everywhere else the half tile doubles the per-kv-tile
# fixed work and LOSES: plain / kv_mask-only at S % 128 == 0 -3..-20 %, bias-only at S % 128 == 0 >= 384 -3..-13 %, the
# kv_mask-only padded body a wash or a loss (except one kv tile), the d64 padded plain body above 256 -4 %.
# Read by the heuristics' pick, the engine row's form evaluation and the adapter's default (one arithmetic, rule 8b'); a
# REQUESTED width is honoured or declined by the row / the adapter, never degraded.
SM107_SMALL_TILE_N_DEFAULT, SM107_SMALL_TILE_N_NARROW = 128, 64
# the d64 bias-only graph and the d64 padded body take the narrow tile up to this S_kv (one or two 128-wide kv tiles)
SM107_NARROW_TILE_S_KV_MAX = 256
# How many of a kv step's 64-column bias chunks the softmax lanes pre-load BEFORE the bmm1_done wait (the kernel's
# BIAS_PRELOAD_CHUNKS), per RENDERING, keyed on the KV tile width -- the sign of the pre-load is a property of the rendering, not of
# the kernel (MEASURED, bf16 D 32 / 64, a 212-SM cc 10.7 node at the 2376 MHz lock, 100 x 3 graph-replay rounds, a base / 0 / base
# cross-process A/B/A over 27 cells with the base drift <= 0.23 %): on the 128-wide tile loading NOTHING ahead of the wait is
# +1.4..+3.2 % on every bias form (+1.7..+1.9 % on the pair item, +3.1 % on the one-batch-entry item, +1.2 % with a kv_mask) -- the
# row read hides in place on the issue-bound body and the 32 packed words it kept live across the wait are freed; on the 64-wide
# tile the bias + kv_mask body LOSES 2.5..2.7 % without the pre-load (its one 64-column chunk has nothing to overlap the row read
# with), so that width keeps one chunk ahead of the wait (the bias-only 64-wide renderings are unmeasured either way and keep it
# too); two chunks ahead is -1.0 % (64 words live across the wait).  Read by the kernel module at render time (one arithmetic, rule
# 8b'); the domain {0, 1, 2} is the kernel's.
SM107_BIAS_PRELOAD_CHUNKS_BY_TILE_N = {SM107_SMALL_TILE_N_DEFAULT: 0, SM107_SMALL_TILE_N_NARROW: 1}


def sm107_bias_preload_chunks(tile_n: int) -> int:
    """The kernel's ``BIAS_PRELOAD_CHUNKS`` for a rendering at KV tile width ``tile_n`` (:data:`SM107_BIAS_PRELOAD_CHUNKS_BY_TILE_N`);
    a width outside the small tiles' domain (the d128 / d192 / d256 / d512 renderings, which carry no bias arm) keeps the one-chunk
    default so an unrelated rendering's instruction stream is unchanged."""
    return SM107_BIAS_PRELOAD_CHUNKS_BY_TILE_N.get(int(tile_n), 1)


# The padded TAIL form of the PLAIN padded body on the small tiles (TemplateParams.padded_tail_form; MASK_FLAGS == MASK_PADDED alone):
# 0 = the legacy form -- the tail kv tile runs the SOFTWARE-max masked body as a SECOND loop beside the HW-max body, so the 128-wide
# rendering carries two full softmax bodies per warp group (4139 vs 2776 SASS instructions on the d32 bf16 rendering, HOST
# trace-compile for sm_107a) and outgrows the per-SM instruction cache -- the padding leg's measured pathology (S 217 plain 25.9 us vs
# 15.4 for the DENSE S 256 cell and 19.6 for the same shape WITH a kv_mask, whose -inf body is one body); 1 = the PRE-MASKED tail --
# one loop, the HW-max body on every tile, and on the tail tile the softmax warp first rewrites its sub-tile's out-of-range S
# columns as -inf in TMEM (a plain load, the bit-word PADDED mask, a store, one tcgen05.wait::st) so the fused max load observes
# them masked.  MEASURED (212-SM cc 10.7 perf node, 2376 MHz lock, same-process interleaved arms, 100 launches x 3 rounds, control
# twins within 0.4 %): at TILE_N 128 form 1 wins EVERY padded plain cell -- d32 B 1 S 200 / 217 / 300 / 450 / 600 +39 / +39 / +43 /
# +29 / +17 %, d32 B 4 S 217 / 300 +81 / +56 %, d64 B 1 S 200 / 300 / 600 +37 / +37 / +13 % over the legacy 128-wide body -- so the
# 128-wide default is 1; at TILE_N 64 (two half bodies under the cliff already) it is within noise with a per-cell sign (d32
# -0.4..-2.2 %, d64 +1.2..+2.3 %), so the 64-wide default stays the legacy form.  A request or the sweep seam pins either form at
# either width.
SM107_PADDED_TAIL_FORM_DEFAULT = 1  # the 128-wide tile
SM107_PADDED_TAIL_FORM_DEFAULT_NARROW = 0  # the 64-wide tile
PADDED_TAIL_FORMS = frozenset({0, 1})
# PRIVATE, process-local sweep seam (the ``_N_CHUNK_OVERRIDE_FOR_SWEEPS`` rules: ``monkeypatch.setattr`` in a test or a sweep driver
# that resets it to None; never a plan path, an import or a config): pins the padded tail form a knob-less ELIGIBLE plan gets.
_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS: Optional[int] = None


def sm107_padded_tail_form_pick(
    d_flavor: int,
    *,
    padded: bool,
    has_bias: bool,
    has_kv_mask: bool,
    causal: bool = False,
    swa: bool = False,
    request: Optional[int] = None,
    tile_n: int = SM107_SMALL_TILE_N_DEFAULT,
) -> int:
    """The ``TemplateParams.padded_tail_form`` of a plan on the Rubin small tiles: 0 off the eligible set (a dense S_kv at a tile
    multiple, a bias or a kv_mask -- whose -inf body is ONE body already -- a causal / SWA band, the d128 flavor); on it ``request``
    when the caller made one (the adapter's ``padded_tail_form=``, validated there: honoured or declined, never degraded), else the
    sweep seam, else the measured default of the plan's KV tile width ``tile_n`` (appended, default 128):
    :data:`SM107_PADDED_TAIL_FORM_DEFAULT` on the 128-wide tile, :data:`SM107_PADDED_TAIL_FORM_DEFAULT_NARROW` on the 64-wide one."""
    if not padded or has_bias or has_kv_mask or causal or swa or int(d_flavor) not in (32, 64):
        return 0
    if request is not None:
        return int(request)
    if _PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS is not None:
        return int(_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS)
    return SM107_PADDED_TAIL_FORM_DEFAULT_NARROW if int(tile_n) == SM107_SMALL_TILE_N_NARROW else SM107_PADDED_TAIL_FORM_DEFAULT


# The DEAD-SUB-TILE QUIET body (TemplateParams.dead_subtile_quiet; the one-batch-entry item of the small tiles, no bias): a softmax
# sub-tile whose first row is at or past its batch entry's S_q has no live row -- sub-tile 1 of the last 256-row super-tile when
# S_q % 256 is in 1..128 (S 384: a quarter of the item's softmax work; S 300 / 600 on the padding leg), and under cga2 the non-leader
# CTA's sub-tiles past S_q.  0 = the legacy form: that warp group runs the full softmax body on rows nobody stores (exp2 / max / sum
# on values the correction trims or the TMA store clips); 1 = the QUIET body: the same barrier sequence as the live body (the
# bmm1_done wait, alpha = 1 and the stat_full arrive, the bmm2_ready arrives, the warp-group ping-pong, the stat_empty wait), no TMEM
# load and no math, so the live sibling warp group runs the ping-pong alone.  The barrier table is unchanged by construction (the same
# sites fire the same lanes); the MMA still runs the dead sub-tile's BMM1 / BMM2 (its P is TMEM residue, its O rows are the trimmed /
# clipped rows).  Default 0 until the A/B on an exclusive Rubin GPU; a flip is a commit with the numbers.
SM107_DEAD_SUBTILE_QUIET_DEFAULT = 0
DEAD_SUBTILE_QUIET_FORMS = frozenset({0, 1})
# PRIVATE, process-local sweep seam (the ``_N_CHUNK_OVERRIDE_FOR_SWEEPS`` rules): pins the quiet form a knob-less ELIGIBLE plan gets.
_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS: Optional[int] = None


def sm107_dead_subtile_quiet_pick(d_flavor: int, *, has_bias: bool, pair_item: bool = False, request: Optional[int] = None) -> int:
    """The ``TemplateParams.dead_subtile_quiet`` of a plan on the Rubin small tiles: 0 off the eligible set (a bias -- its warp groups
    also wait and release the bias barriers, which the quiet body does not replicate -- the pair item, whose dead sub-tile is the
    odd chunk tail recomputing row nb0, the d128 flavor); on it ``request`` when the caller made one (the adapter's
    ``dead_subtile_quiet=``, validated there: honoured or declined, never degraded), else the sweep seam, else
    :data:`SM107_DEAD_SUBTILE_QUIET_DEFAULT`."""
    if has_bias or pair_item or int(d_flavor) not in (32, 64):
        return 0
    if request is not None:
        return int(request)
    if _DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS is not None:
        return int(_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS)
    return SM107_DEAD_SUBTILE_QUIET_DEFAULT


def sm107_tile_n_pick(d_flavor: int, s_kv: int, *, has_bias: bool, has_kv_mask: bool, padded: bool = False, request: Optional[int] = None, domain=None) -> int:
    """The ``TemplateParams.tile_n`` a knob-less plan on the Rubin 32- / 64-wide tiles gets (the table above): ``request`` when the
    caller made one (the ``TILE_N`` knob / the adapter's ``tile_n=``: honoured or declined by the row / adapter, never degraded);
    else :data:`SM107_SMALL_TILE_N_NARROW` for (a) a bias TOGETHER WITH a mask -- a kv_mask of any form (the fp32 vector's only served
    width, :data:`SM107_FP32_VECTOR_WITH_BIAS_TILE_N`, agrees) or the kernel's padding-mask arm (per-batch KV lengths, or
    ``s_kv % 128 != 0``: with per-batch lengths of 0.75 S +11..+73 % over the pre-table pick at S 256 / 512 / 1024 / 2048, both D, B 1 / 4 --
    13 of 13 cells with the table's form measured, the d32 B 1 S 2048 cell through the one-batch-entry ring at 64 (+14 %); +17..+38 % on the
    ragged leg) -- the combined body is the one over the instruction-cache cliff; (b) a bias alone at one 128-wide kv tile (the pair n4 / n2: +26 / +24 % at d32 / d64 with 128 rows, +35 /
    +22 % with 512) and at two tiles on the 64-wide tile (the pair n2 / n8: +24 / +4 %) -- at two tiles on the 32-wide tile the
    one-batch-entry item's streamed ring at 128 is the measured best (+6 / +9 % with 256 / 1024 rows), so 128 stays there; (c) a kv_mask
    alone at one 128-wide kv tile (+7..+18 %; a wash or a loss above it); (d) a plain graph whose S_kv ends in the FIRST half of a 128-wide
    tile (``s_kv % 128`` in 1..64, both D): the half tile skips 64 dead columns the 128-wide tile would mask -- d32 S 300 B 1 / 4 +4.2 /
    +6.2 % over the 128-wide pre-masked tail, d64 S 300 a tie (0.2 %); where the tail holds MORE than 64 columns the 128-wide tile WITH
    the pre-masked tail (``padded_tail_form`` 1, the 128-wide default) wins: d32 S 200 / 217 / 450 / 600 +1.8 / +3.9 / +11.5 / +13.7 %,
    d32 B 4 S 217 +5.4 %, d64 S 200 / 600 +7.1 / +20 % over the half tile (212-SM cc 10.7 perf node, 2376 MHz lock, same process) --
    before the pre-masked tail the 128-wide padded body was the instruction-cache pathology and the half tile served every d32 S and
    d64 up to 256; not for per-batch lengths at a tile multiple (128 wins there: the narrow tile is -11..-18 % at S 512 / 1024 and -19 /
    -20 % at d32 / d64 S 2048 with 0.75 S keys).  The pick sees ``s_kv``, never the lengths: where the LENGTHS end mid-tile (0.75 S at
    S 256 = 192 keys) the half tile measured +58..+119 % and this rule cannot follow it; else :data:`SM107_SMALL_TILE_N_DEFAULT`.
    ``domain`` = the flavor's served widths (:data:`SM107_TILE_NS_BY_D_SHAPE`); a flavor without the narrow width keeps the default."""
    if request is not None:
        return int(request)
    dom = (
        frozenset(domain) if domain is not None else dict(SM107_TILE_NS_BY_D_SHAPE).get((int(d_flavor), int(d_flavor)), frozenset({SM107_SMALL_TILE_N_DEFAULT}))
    )
    if SM107_SMALL_TILE_N_NARROW not in dom:
        return SM107_SMALL_TILE_N_DEFAULT
    padded_body = bool(padded) or int(s_kv) % SM107_SMALL_TILE_N_DEFAULT != 0  # the kernel's padding-mask arm
    if has_bias and (has_kv_mask or padded_body):
        return SM107_SMALL_TILE_N_NARROW
    if has_bias and (int(s_kv) <= SM107_SMALL_TILE_N_DEFAULT or (int(d_flavor) == 64 and int(s_kv) <= SM107_NARROW_TILE_S_KV_MAX)):
        return SM107_SMALL_TILE_N_NARROW
    if has_kv_mask and int(s_kv) <= SM107_SMALL_TILE_N_DEFAULT:
        return SM107_SMALL_TILE_N_NARROW
    if padded_body and not has_kv_mask and 0 < int(s_kv) % SM107_SMALL_TILE_N_DEFAULT <= SM107_SMALL_TILE_N_NARROW:
        return SM107_SMALL_TILE_N_NARROW
    return SM107_SMALL_TILE_N_DEFAULT


# The bias block's TMA / lane swizzle: the 128-B atom.  BOTH swizzle jobs force it -- the TMA descriptor writes the
# block under it (job D) and the softmax lanes read one 128-B row slice each (job L: a per-lane stride that is a multiple
# of the 128-B bank cycle is a 32-way conflict unswizzled; the 16-B chunk index XOR (row % 8) spreads 8 rows over 8 bank
# groups -> 4 wavefronts per warp-wide 16-B load = ideal).  ``Swizzle(3, 4, 3)``: MBase + SShift = 7 = log2(128 B).
BIAS_SWZ_BYTES = 128


def sm107_bias_block_rows(tiles_q: int = 2, tile_m: int = 128, *, pair: bool = False) -> int:
    """Rows of the resident bias block = the Q rows one work item's softmax sub-tiles READ: ``tiles_q x tile_m`` on the
    one-batch-entry item (both sub-tiles are rows of one 256-row super-tile), ``tile_m`` on the (n, n+1) PAIR item
    (``TemplateParams.n_chunk >= 1``: the two sub-tiles are rows n / n + 1 of the SAME 128-row q tile and share the block)."""
    return tile_m if pair else tiles_q * tile_m


def sm107_bias_pair_form(bias_batch_extent: int, b_flat: int, cta_mma: int) -> bool:
    """Whether a BIAS plan runs the (n, n+1) PAIR work item (``TemplateParams.n_chunk >= 1``) rather than the
    one-batch-entry item (the 256-row super-tile form): the bias must be batch-BROADCAST (two rows of one chunk read ONE block; a per-batch bias
    ``[B, ., S_q, S_kv]`` keeps the 256-row item), the flattened batch must hold at least two rows (a single row would run a
    dead sub-tile on every item) and the plan runs cga1 (a collective MMA cannot span two rows).  ONE predicate for the
    engine row, the heuristics and the adapter (rule 8b')."""
    return int(bias_batch_extent) == 1 and int(b_flat) >= 2 and int(cta_mma) == 1


def sm107_bias_resident_tiles(
    tile_k: int, cta_mma: int, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, tiles_q: int = 2, tile_m: Optional[int] = None, pair: bool = False
) -> int:
    """How many ``tile_n``-column bias tiles (``BIAS_BLOCK_ROWS x tile_n`` elements each) the Rubin carveout holds beside
    the d128 family's Q / O slabs and K / V rings at ``tile_k = tile_o`` and the flavor's ring depth (``_stages_kv_d128``)
    -- the SAME budget ``_validate_smem`` enforces (320 KiB usable minus the fixed overhead).  0 = no resident block fits.
    Derived, never a literal: 3 at the 32-wide tile (cga1), 1 / 2 at the 64-wide one (cga1 / cga2), 0 at d128 on the
    256-row item; the PAIR item's 128-row block (``pair=True``) holds twice as many -- 6 at the 32-wide tile, 3 at the
    64-wide one (cga1)."""
    b = bpe(dtype_qkv)
    stages = _stages_kv_d128(dtype_qkv, cta_mma, mxfp8=False, tile_k=tile_k)
    # The Q tile height: the seam's (the TILE_M = 64 lever halves the Q / O slabs AND the block's rows) unless the caller pinned one.
    tile_m = _tile_m_d128(tile_k) if tile_m is None else int(tile_m)
    operands = sdpa_smem_bytes(tile_m, tile_n, tile_k, tile_k, tiles_q, stages, cta_mma, b, b)
    spare = SMEM_USABLE_BYTES - _SMEM_FIXED_OVERHEAD - operands
    tile_bytes = sm107_bias_block_rows(tiles_q, tile_m, pair=pair) * tile_n * b
    return max(0, spare // tile_bytes)


def sm107_bias_resident_s_kv_max(d_flavor: int, cta_mma: int, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, pair: bool = False) -> int:
    """The largest S_kv (in whole tiles) whose bias row-block the resident form serves at this width and cga: the work
    item's KV range is at most ceil(S_kv / tile_n) tiles.  0 = the width serves no bias.  ``pair``: the (n, n+1) item's
    128-row block (``sm107_bias_pair_form``).  Read by the engine row's mismatch, the adapter's check_support and the config
    factory (one arithmetic, rule 8b')."""
    return sm107_bias_resident_tiles(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair) * tile_n


def sm107_bias_ring_stages(
    tile_k: int, cta_mma: int, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, tiles_q: int = 2, tile_m: Optional[int] = None, pair: bool = False
) -> int:
    """The deepest STREAMED bias ring (``BIAS_MODE_STREAMED``) the Rubin carveout holds beside the operands at this width:
    ``min(max(BIAS_RING_STAGES), spare // tile_bytes)`` in whole ``BIAS_BLOCK_ROWS x tile_n`` tiles, **0 when fewer than
    ``min(BIAS_RING_STAGES)`` fit** (no ring).  The SAME spare the resident budget reads (one arithmetic, rule 8b').  Derived:
    3 at the 32-wide tile (both item forms, cga1), 3 at the 64-wide tile's pair item (32 KiB stages), 0 at its one-batch-entry
    item at cga1 (two 64 KiB stages exceed the 126 KiB spare) and 2 at cga2."""
    tiles = sm107_bias_resident_tiles(tile_k, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, tiles_q=tiles_q, tile_m=tile_m, pair=pair)
    stages = min(max(BIAS_RING_STAGES), tiles)
    return stages if stages >= min(BIAS_RING_STAGES) else 0


# PRIVATE, process-local TEST seams beside ``_N_CHUNK_OVERRIDE_FOR_SWEEPS`` (same rules: ``monkeypatch.setattr`` in a test or a
# sweep driver that resets them to None; never a plan path, an import or a config): ``_BIAS_MODE_OVERRIDE_FOR_SWEEPS`` pins the
# bias FORM a knob-less plan gets (1 = the resident block, declining above its budget -- the A/B's resident arm; 2 = the streamed
# ring wherever it fits, INCLUDING where the resident block would also fit -- the A/B's streamed arm); ``_BIAS_STAGES_OVERRIDE_FOR_SWEEPS``
# pins the ring depth (2 / 3).  A set value changes the plan the graph BUILDS and its ELIGIBILITY (the pick feeds the engine row's
# mismatch, the heuristics' width pick and the adapter), so neither is configuration nor part of the public surface.
_BIAS_MODE_OVERRIDE_FOR_SWEEPS: Optional[int] = None
_BIAS_STAGES_OVERRIDE_FOR_SWEEPS: Optional[int] = None
# ``_STAGES_KV_OVERRIDE_FOR_SWEEPS`` (same rules) pins the K / V ring depth of the 32- / 64-wide f16/bf16 tiles (``_stages_kv_d128``,
# the small-tile branch; 2 / 3 / 4 = :data:`SMALL_D_STAGES_KV`): the A/B arm of the ring-depth lever.  Read by the ONE helper every
# budget and the factory call, so a pinned depth moves the bias budgets with it (a shallower ring frees SMEM for the resident block /
# a deeper bias ring).  None = the derived depth (4).
_STAGES_KV_OVERRIDE_FOR_SWEEPS: Optional[int] = None
# The K / V ring depths the small-tile body is validated at (SMEM fit at every width in SM107_SMALL_D_CGAS, the ring-wrap axis
# n_kv = depth / depth + 1 / two wraps of the fresh-process census).  The validator pins cfg.STAGES_KV to this set on the small tiles.
SMALL_D_STAGES_KV = frozenset({2, 3, 4})
# ``_TILE_M_OVERRIDE_FOR_SWEEPS`` (same rules) pins the Q TILE HEIGHT of the 32- / 64-wide f16/bf16 tiles: 64 or 128 (:data:`SMALL_D_TILE_MS`);
# None = 128, the height every rendering has had.  Read by the ONE helper (``_tile_m_d128``) the config factory, the bias budgets
# (``sm107_bias_resident_tiles`` / ``sm107_bias_ring_stages``: the Q / O slabs and the bias block's rows both halve) and the kernel's
# TMEM / grid derivations share (rule 8b').  A 64-row tile doubles the work items of a given q extent (the parallelism lever at small S)
# and halves the MMA / softmax work per kv step; the kernel body renders it as a lane-mapping arm (``TILE_M64_LANE_FORM`` in
# ``sm107/prefill_d128_f16.py``).  An A/B arm, never a plan path: the engine row's ``tile_ms`` stays {128} until the arm has a measured win.
_TILE_M_OVERRIDE_FOR_SWEEPS: Optional[int] = None
# The Q tile heights the small-tile body is validated at.  The validator pins cfg.TILE_M to this set on the small tiles; every other
# flavor keeps 128 (the d128 / d192 renderings and their committed records, the quantized siblings' SF geometry).
SMALL_D_TILE_MS = frozenset({64, 128})


def sweep_seam_state() -> Optional[tuple]:
    """The private sweep seams' values as ONE hashable, ``None`` when every seam is unset (every plan path).  ``api_dsl`` folds it into
    the template-module cache key and the source digest (``template_loader.load_template(extra_key=)``): the module cache keys on the
    TemplateParams record, and a seam moves ``Cfg`` fields (the ring depth, the Q tile height) WITHOUT touching the record -- so two
    renderings that differ only by a seam would otherwise share one module in a process (the second request silently running the
    first rendering) and one compiled-plan artifact on disk."""
    state = (
        _N_CHUNK_OVERRIDE_FOR_SWEEPS,
        _BIAS_MODE_OVERRIDE_FOR_SWEEPS,
        _BIAS_STAGES_OVERRIDE_FOR_SWEEPS,
        _STAGES_KV_OVERRIDE_FOR_SWEEPS,
        _TILE_M_OVERRIDE_FOR_SWEEPS,
    )
    return None if all(v is None for v in state) else ("sm107_sweep_seams",) + tuple(None if v is None else int(v) for v in state)


def _tile_m_d128(tile_k: int) -> int:
    """The Q tile height of a d128-family rendering: 128 everywhere but the 32- / 64-wide f16/bf16 tiles, where the private seam
    ``_TILE_M_OVERRIDE_FOR_SWEEPS`` may pin 64 (the TILE_M = 64 lever).  Read by the factory AND the bias budgets so the Q / O slabs,
    the bias block rows and the resident / ring budgets move together (one arithmetic, rule 8b')."""
    if tile_k > 64:
        return 128
    if _TILE_M_OVERRIDE_FOR_SWEEPS is not None:
        want = int(_TILE_M_OVERRIDE_FOR_SWEEPS)
        if want not in SMALL_D_TILE_MS:
            raise ValueError(
                f"_TILE_M_OVERRIDE_FOR_SWEEPS={want}: the 32- / 64-wide tiles' Q tile height is validated at {sorted(SMALL_D_TILE_MS)} rows "
                f"(the warp-lane -> row mapping of the softmax / correction groups and the M = 64 tcgen05.mma datapath layout exist for these two only)"
            )
        return want
    return 128


# The ONE-BATCH-ENTRY item at the 32-wide tile's cga1 STREAMS its bias wherever its ring fits, resident budget or not: the 256-row
# item's block fill is exposed per item (nothing amortises it) and the ring overlaps the tile loads with the kv loop -- MEASURED
# (bf16, a 212-SM cc 10.7 node at the 2376 MHz lock, 100 x 3 graph-replay rounds, a control twin per process, the kernel-only CUPTI
# time at S 256) the 2-stage ring over the resident block: +6.4 % at S_q = S_kv = 256 with 256 flattened rows (22.11 -> 20.78 us),
# +10.0 % with 1024 rows (67.47 -> 61.34), +9.9 / +12.9 % at S 384 with 384 / 1536 rows (70.24 -> 63.90, 256.00 -> 226.74; the 3-stage
# ring +8.5 / +9.6 %).  The PAIR item keeps "resident where it fits" (its block IS amortised over the chunk: the 2-stage pair ring is
# -0.3 / -1.7 % at d32 S 384 / 768, a tie at d64), and so does the 64-wide tile (its one-batch-entry item has no cga1 ring; the cga2
# resident-vs-ring pick is unmeasured).  Measured at two and three kv tiles; at ONE tile the block stays (a ring has nothing to overlap).
SM107_ONE_BATCH_ENTRY_STREAMING_D_FLAVORS = frozenset({32})


def sm107_bias_mode_pick(d_flavor: int, cta_mma: int, s_kv: int, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, pair: bool = False, request=None) -> int:
    """The ``TemplateParams.bias_mode`` a BIAS plan gets at this width: ``BIAS_MODE_RESIDENT`` when the resident block holds
    ``s_kv`` (:func:`sm107_bias_resident_s_kv_max`) -- except the one-batch-entry item at a width in
    :data:`SM107_ONE_BATCH_ENTRY_STREAMING_D_FLAVORS` (cga1), which takes ``BIAS_MODE_STREAMED`` whenever its ring fits (the measured
    pick above) -- else ``BIAS_MODE_STREAMED`` when a ring of >= 2 stages fits (:func:`sm107_bias_ring_stages`), else **0 = no form at
    this width** (a typed decline).  ``request`` (the adapter's ``bias_mode=`` or the test seam) pins the form and returns 0 when THAT
    form does not fit here -- honoured or declined, never degraded.  ONE arithmetic for the engine row, the heuristics' width pick and
    the adapter (rule 8b')."""
    want = request if request is not None else _BIAS_MODE_OVERRIDE_FOR_SWEEPS
    resident_ok = s_kv <= sm107_bias_resident_s_kv_max(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair)
    ring_ok = sm107_bias_ring_stages(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair) > 0
    if want is not None:
        want = int(want)
        if want == BIAS_MODE_RESIDENT:
            return BIAS_MODE_RESIDENT if resident_ok else BIAS_MODE_NONE
        if want == BIAS_MODE_STREAMED:
            return BIAS_MODE_STREAMED if ring_ok else BIAS_MODE_NONE
        return BIAS_MODE_NONE
    if ring_ok and not pair and int(cta_mma) == 1 and int(d_flavor) in SM107_ONE_BATCH_ENTRY_STREAMING_D_FLAVORS and int(s_kv) > SM107_SMALL_TILE_N_DEFAULT:
        return BIAS_MODE_STREAMED  # measured at 2 and 3 kv tiles; one tile keeps the block (nothing to overlap)
    if resident_ok:
        return BIAS_MODE_RESIDENT
    return BIAS_MODE_STREAMED if ring_ok else BIAS_MODE_NONE


# The one-batch-entry item's streamed ring runs 2 stages, not the deepest that fits: its 64 KiB stage x 3 buys a lookahead the
# 2-sub-tile item never uses -- MEASURED +3.6 / +2.0 / +1.5 % at d32 S 512 / 768 / 1024 (B = 1), +3.2 / +2.1 / +1.6 % at B = 4,
# +2.0 / +1.2 % with a kv_mask, 6 of 6 processes (the sweep's node and protocol, see sm107_tile_n_pick).  The PAIR item's ring keeps
# the deepest: 2 stages are a tie there (d64 +0.3..+0.8 %) or a loss (d32 -0.3 / -1.7 %).
SM107_ONE_BATCH_ENTRY_RING_STAGES = 2


def sm107_bias_stages_pick(d_flavor: int, cta_mma: int, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, pair: bool = False, request: int = 0) -> int:
    """The ``STAGES_BIAS`` of a STREAMED plan: the request (``TemplateParams.bias_stages`` / the adapter's ``bias_stages=`` /
    the test seam) when it is in ``BIAS_RING_STAGES`` and fits, else the derived deepest ring for the PAIR item and
    ``min(deepest, SM107_ONE_BATCH_ENTRY_RING_STAGES)`` for the one-batch-entry item; 0 = no ring fits (the factory raises, the row /
    adapter decline).  A request that does not fit is 0 too (declined typed, never degraded)."""
    deepest = sm107_bias_ring_stages(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair)
    want = int(request) if request else (_BIAS_STAGES_OVERRIDE_FOR_SWEEPS or 0)
    if want:
        return want if (want in BIAS_RING_STAGES and want <= deepest) else 0
    return deepest if pair else min(deepest, SM107_ONE_BATCH_ENTRY_RING_STAGES)


def sm107_bias_auto_cga(d_flavor: int, s_kv: int, cgas, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, pair: bool = False) -> Optional[int]:
    """The NARROWEST cga in ``cgas`` whose resident block holds ``s_kv`` (``sm107_bias_resident_s_kv_max``), or ``None``
    when no width does.  The heuristics' auto pick for a BIAS graph (a kv_mask alone stages no block and keeps the flavor's
    own width rule), the width the standalone adapter lowers
    with when none is requested, and the engine row's knob-less eligibility all read it (rule 8b'): a d64 graph at S_kv
    256 lands on cga2 (two tiles) without a knob, S_kv <= 128 keeps the tuned cga1.  ``pair`` = the graph qualifies for the
    (n, n+1) item (a batch-broadcast bias on >= 2 rows): the 128-row block is evaluated at cga1 only (the pair runs cga1), the
    256-row one at every other width.  Second pass (the STREAMED fallback): when no width's resident block holds ``s_kv``, the
    narrowest width whose ring fits (:func:`sm107_bias_mode_pick` != 0) -- a per-batch bias at the 64-wide tile above S_kv 256
    lands on cga2 streamed, the pair item stays cga1.  ``None`` when neither form fits anywhere."""
    for cga in sorted(cgas):
        if sm107_bias_mode_pick(d_flavor, cga, s_kv, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair and cga == 1) == BIAS_MODE_RESIDENT:
            return cga
    for cga in sorted(cgas):
        if sm107_bias_mode_pick(d_flavor, cga, s_kv, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair and cga == 1) != BIAS_MODE_NONE:
            return cga
    return None


def sm107_n_chunk_rule(b_flat: int, h_q: int, s_q: int, sm_count: int, *, tile_m: int = 128, waves: int = 3) -> int:
    """The default chunk length of the (n, n+1) PAIR item: the largest power of two ``<= h_q * ceil(s_q / tile_m) * b_flat /
    (waves * sm_count)`` (the ">= 3 waves" reading of the plan's parallelism-vs-amortisation trade), clamped to
    ``[2, b_flat]`` and EVEN (an odd chunk runs a dead sub-tile on its last step).  ``sm_count`` 0 / None = 148.  Returns 0
    (the one-batch-entry item) when ``b_flat < 2``."""
    if b_flat < 2:
        return 0
    sms = int(sm_count) if sm_count else 148
    items = h_q * (-(-s_q // tile_m)) * b_flat
    n = 1
    while n * 2 <= items // (waves * sms):
        n *= 2
    n = max(2, min(n, b_flat))
    if n % 2:
        n -= 1
    return max(2, n)


# A PRIVATE, process-local TEST seam (None = the rule above; an int pins :func:`sm107_n_chunk_pick`'s result, 0 = the
# one-batch-entry item): the (n, n+1) item's chunk length is a performance axis of the frontend band (an autotune knob,
# append-only in knobs.h) once BOTH pybind extensions carry the enum member; until that rebuild the graph path cannot carry
# it, so a sweep driver or a test pins the pick here -- through ``monkeypatch.setattr`` (or a driver that resets it to None),
# never from a plan path or at import.  A set value changes the plan the graph BUILDS and its ELIGIBILITY (the pick feeds
# :func:`sm107_pair_item_active`, i.e. the engine row's mismatch, the heuristics and the adapter), so it is not configuration
# and is not part of the module's public surface.
_N_CHUNK_OVERRIDE_FOR_SWEEPS: Optional[int] = None


# The (n, n+1) pair item's chunk is capped at 8: the 3-wave rule's larger picks LOSE wherever they were measured against n8 --
# the RESIDENT block n16 -> n8 +3.1 / +3.7 % at 1536 rows (d32 / d64), n32 -> n8 +2.9 % at 2048 rows (d32, with a kv_mask), the
# earlier S = 768 B = 1 miss (16 vs 8, +3.9 %), n64 -> n16 +5.8 % at 3072 rows (n8 unmeasured there); the streamed RING (nothing to
# amortise over the chunk) n16 / n32 -> n4 / n8 at d64 S_kv 640 / 768 / 896 (+2.4 / +3.8 / +5.5 % for n4, n8 within 1 % of n4, two
# GPUs), n32 -> n16 +1.0 % at 1024.  n8 itself is the measured best at 512 rows and stays (see sm107_tile_n_pick for the sweep).
SM107_PAIR_N_CHUNK_MAX = 8
# At ONE 128-wide kv tile the per-item fixed cost (the block fill, the item barriers) dominates and a 4-row chunk amortises it -- as
# long as the chunking leaves the grid under-filled: n4 beats the clamp-2 pick by +8..+16 % at 128 flattened rows (every D, both
# widths) while at 512 rows (4.8 waves of 2-row items) n2 wins back by +8 % on the 64-wide tile.  The floor applies while the 2-row
# items number at most this many waves.
SM107_ONE_TILE_N_CHUNK_FLOOR, SM107_ONE_TILE_FLOOR_MAX_WAVES = 4, 2


def sm107_n_chunk_refine(n: int, b_flat: int, h_q: int, s_q: int, sm_count: int, *, s_kv: int, tile_m: int = 128) -> int:
    """The knob sweep's corrections to :func:`sm107_n_chunk_rule`'s 3-wave pick for the (n, n+1) pair item (the constants above
    carry the numbers): the pick is capped at :data:`SM107_PAIR_N_CHUNK_MAX`; at one 128-wide kv tile
    (``s_kv <= SM107_SMALL_TILE_N_DEFAULT``) it is floored at :data:`SM107_ONE_TILE_N_CHUNK_FLOOR` while the 2-row items fill at most
    :data:`SM107_ONE_TILE_FLOOR_MAX_WAVES` waves.  The result stays even and inside ``[2, b_flat]`` (:func:`sm107_n_chunk_rule`'s
    clamp)."""
    sms = int(sm_count) if sm_count else 148
    n = min(n, SM107_PAIR_N_CHUNK_MAX)
    items_at_2 = h_q * (-(-s_q // tile_m)) * (-(-b_flat // 2))
    if s_kv <= SM107_SMALL_TILE_N_DEFAULT and items_at_2 <= SM107_ONE_TILE_FLOOR_MAX_WAVES * sms:
        n = max(n, SM107_ONE_TILE_N_CHUNK_FLOOR)
    n = max(2, min(n, b_flat))
    if n % 2:
        n -= 1
    return max(2, n)


def sm107_n_chunk_pick(
    bias_batch_extent: int,
    b_flat: int,
    h_q: int,
    s_q: int,
    sm_count: int,
    cta_mma: int,
    *,
    s_kv: Optional[int] = None,
    d_flavor: Optional[int] = None,
    tile_n: int = 128,
) -> int:
    """The ``TemplateParams.n_chunk`` a knob-less BIAS plan gets: 0 unless :func:`sm107_bias_pair_form` holds, else the
    test seam ``_N_CHUNK_OVERRIDE_FOR_SWEEPS`` (0 = the one-batch-entry item, for its A/B arm) or :func:`sm107_n_chunk_rule` refined by
    :func:`sm107_n_chunk_refine` -- behind ONE measured guard:
    at ``s_q % 256 == 0`` with ``s_kv`` inside the one-batch-entry block's budget the 256-row super-tile wastes no Q rows and its
    two sub-tiles SHARE every K / V tile, where the pair streams them twice, so the one-batch-entry item is the faster form
    (S_q = S_kv = 256, d32: 25.44 vs 26.50 us at n_chunk 2, 26.66 at 4, 31.26 at 8 -- a 212-SM cc 10.7 node at the 2376 MHz lock,
    sampled 2364 MHz, one process, control pair 0.24 %); the pair wins where a super-tile would be partly empty (S_q 128: +30.6 %,
    384: +30.0 %, each against the one-batch-entry item in the same process) or where only its block holds S_kv (512+).
    ``s_kv`` / ``d_flavor`` None skips the guard (callers that pin the form through the override or a request never reach it).  The guard
    covers every ``s_kv`` the one-batch-entry item serves at the width (its block or its streamed ring): +8.6 % over the pair's resident block
    at S 768, +6..7 % over the pair's ring at 1024 / 2048 / 4096 (d32).  **The guard holds at ``tile_n`` 128 only** (the plan's KV tile
    width, :func:`sm107_tile_n_pick`): on the 64-wide tile the pair's resident block beats every one-batch-entry form -- n2 at S 256
    (+2.8 %), n8 at S 512 / 768 with a kv_mask (+12 / +13 % over the one-batch-entry ring at the same width), n4 / n8 at S 256 with
    1024 rows (+7 %); at S 1024 the pair ring and the one-batch-entry ring tie there."""
    if not sm107_bias_pair_form(bias_batch_extent, b_flat, cta_mma):
        return 0
    if _N_CHUNK_OVERRIDE_FOR_SWEEPS is not None:
        return max(0, int(_N_CHUNK_OVERRIDE_FOR_SWEEPS))
    if s_kv is not None and d_flavor is not None and s_q % 256 == 0 and int(tile_n) == SM107_SMALL_TILE_N_DEFAULT:
        # A full 256-row super-tile wastes no Q rows and SHARES every K / V tile across its two sub-tiles, where the pair streams them
        # twice; the pair's one advantage -- one resident block amortised over the chunk -- does not outweigh it: the one-batch-entry item
        # wins wherever a FORM of it fits at this width (its resident block -- the guard's first measured shape, S_q = S_kv = 256 above;
        # its streamed ring -- MEASURED on a 212-SM cc 10.7 node at the 2376 lock, one process per cell, control pairs <= 0.11 %:
        # S_q = S_kv = 768, d32, sampled 2364 MHz: the
        # one-batch-entry ring 303.86 us vs the pair's RESIDENT block 330.10 (+8.6 %) vs the pair's ring 332.86; 1024: 675.84 vs the pair's
        # ring 725.49 (+7.3 %); 2048: 5040.62 vs 5389.50 (+6.9 %); 4096 (both at a capped 2160 MHz): 44823.51 vs 47563.59 (+6.1 %)).  At the
        # 64-wide tile's cga1 the one-batch-entry item has no ring (sm107_bias_ring_stages = 0) above its 128-column block, so the pair
        # streams there.
        if sm107_bias_mode_pick(d_flavor, cta_mma, s_kv, pair=False) != BIAS_MODE_NONE:
            return 0
    n = sm107_n_chunk_rule(b_flat, h_q, s_q, sm_count)
    if s_kv is None:
        return n
    return sm107_n_chunk_refine(n, b_flat, h_q, s_q, sm_count, s_kv=s_kv)


def sm107_pair_item_active(
    bias_batch_extent: int,
    b_flat: int,
    h_q: int,
    s_q: int,
    sm_count: int,
    cta_mma: int,
    n_chunk_request=None,
    *,
    s_kv: Optional[int] = None,
    d_flavor: Optional[int] = None,
    tile_n: int = 128,
) -> bool:
    """Whether a BIAS plan RUNS the (n, n+1) pair item -- the form predicate AND a chunk length of at least 1 (the request when
    the caller made one, else :func:`sm107_n_chunk_pick` at the plan's KV tile width ``tile_n``; 0 from either = the one-batch-entry
    item).  THE predicate the engine row, the heuristics and the adapter evaluate the budget of (rule 8b'): a plan must never be
    admitted on the 128-row block's budget and built on the 256-row block."""
    if not sm107_bias_pair_form(bias_batch_extent, b_flat, cta_mma):
        return False
    n = (
        int(n_chunk_request)
        if n_chunk_request is not None
        else sm107_n_chunk_pick(bias_batch_extent, b_flat, h_q, s_q, sm_count, cta_mma, s_kv=s_kv, d_flavor=d_flavor, tile_n=tile_n)
    )
    return n >= 1


# The same fact keyed by the config-flavor string `_validate_params` receives
# (`make_cfg_d256` and `make_cfg_d256_mxfp8`).  Must stay in step with
# SM107_EPILOGUE_GATE_SHAPES: every dtype family of a listed shape carries the
# seams, and only those flavors may load with epilogue_gate=True.
_EPILOGUE_GATE_FLAVORS = frozenset({"sm107 d256", "sm107 d256 mxfp8"})


def epilogue_gate_layout_declarable(shape_bhsd: tuple, stride_bhsd: tuple, elem_bytes: int = 2) -> bool:
    """Whether a logical BHSD gate ``G`` binds ZERO-COPY on the Rubin d256 gated
    kernels: BSHD-compact, or a token-major declaration TMA can express
    (``config_sm100.bshd_zero_copy_stride``: D innermost-contiguous, then heads,
    then tokens; seq/head strides 16-byte multiples; covering).

    The gate has NO normalisation-copy fallback (a hidden per-execute repack is
    what the engine contract forbids), so this predicate is STRICTER than
    ``graph_analyzer.dense_layout_ok`` -- which admits any B/H/S stride order
    and any 16-byte-unaligned stride because the Q/K/V/O path copies those.  It
    is the ONE rule both sides of rule 8b consume: ``engines.mismatch`` applies
    it to G before admitting a gated graph, and
    ``api_dsl.SdpaFwdDslSm100.check_support`` raises on exactly the layouts it
    rejects, so a plan that entered the ranked list can never die on G's layout
    in the lowering.  ``elem_bytes`` defaults to the 2 B every Rubin gate kernel
    stores (``_CfgSm107.GATE_BPE``: Q-dtype on the half kernels, bf16 on the
    quantized one).
    """
    return bshd_compact(shape_bhsd, stride_bhsd) or bshd_zero_copy_stride(shape_bhsd, stride_bhsd, elem_bytes) is not None


# ---------------------------------------------------------------------------
# Derivation helpers (Rubin values — several differ from the SM100 twins)
# ---------------------------------------------------------------------------


def bpe(dtype: int) -> int:
    """Bytes per STORAGE element. FP8/MXFP8 = 1 (the block-scaled O codes are
    byte containers too), BF16/FP16 = 2."""
    return 1 if (dtype <= _DTYPE_E5M2 or dtype in (_DTYPE_O_NVFP4, _DTYPE_O_MXFP8)) else 2


def o_pack_div(dtype_o: int) -> int:
    """Logical O elements per storage byte: 2 for E2M1, else 1."""
    return 2 if dtype_o == _DTYPE_O_NVFP4 else 1


def resolve_dtype_o(params: TemplateParams) -> int:
    """``TemplateParams.dtype_o`` uses -1 as "inherit dtype_qkv"; every consumer
    must resolve it before use. Passing the raw -1 into ``bpe()`` would silently
    yield BPE_O=1 (an FP8-sized O buffer) for a half-precision output."""
    return params.dtype_qkv if params.dtype_o < 0 else params.dtype_o


def tile_k_hw(dtype_qkv: int) -> int:
    """HW k-step in elements.

    **This is the single most arch-sensitive value in the file.** Rubin runs the
    dense 2xFP8 MMA at K=64 per instruction and pairs it with idesc ``k_dim=1``;
    Blackwell has no K=64 FP8 path and uses K=32 with ``k_dim=0``. The pairing is
    arch-OPPOSITE, and a mismatch does not fail — it silently scrambles rows of
    the accumulator (see rules/mma-tma-matrix.md S1). The kernel bodies here
    hardcode ``k_dim=1``, so FP8/MXFP8 must be 64.

    BF16/FP16 use 16 on both arches: the 2-chunk (K=32) f16 path is silently
    wrong on SM10x.
    """
    return 64 if dtype_qkv <= _DTYPE_E5M2 else 16


def q_swz_bytes(tile_k: int, bpe_val: int) -> int:
    return 128 if (tile_k * bpe_val) % 128 == 0 else 64


def v_swz_bytes(tile_o: int, cta_mma: int, bpe_val: int) -> int:
    """V is split along d_v under cga2, so its swizzle follows the PER-CTA inner
    stride; Q/K/O use the full-tile inner extent."""
    inner = (tile_o // cta_mma) * bpe_val
    if inner % 128 == 0:
        return 128
    if inner % 64 == 0:
        return 64
    if inner % 32 == 0:
        return 32
    raise ValueError(f"V inner bytes {inner} is not a multiple of 32/64/128")


def o_swz_bytes(tile_o: int, bpe_o: int, pack_div: int = 1) -> int:
    return 128 if (tile_o * bpe_o // pack_div) % 128 == 0 else 64


def rescale_threshold(dtype_qkv: int) -> float:
    """4.0 for FP8/MXFP8 (low precision, rescale aggressively); 8.0 for
    BF16/FP16 (defer rescale to amortize the alpha == 1 fast path)."""
    return 4.0 if dtype_qkv <= _DTYPE_E5M2 else 8.0


def sdpa_smem_bytes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    tile_o: int,
    tiles_q: int,
    stages_kv: int,
    cta_mma: int,
    bpe_qkv: int,
    bpe_o: int,
    qo_alias: bool = False,
    gate_bpe: int = 0,
    sf_bytes: int = 0,
) -> int:
    """Q/K/V/O (+ GATE) (+ MXFP8 scale-factor slabs) SMEM footprint for the
    classic prefill pipeline.

    K (seq rows) and V (d_v cols) shrink by CTA_MMA; Q/O are per-CTA full.
    ``qo_alias`` = O reuses Q's slab in the epilogue (max instead of sum).
    ``gate_bpe`` = bytes per element of the fused epilogue gate's staging tile
    (0 = no gate).  The gate tile is one O-shaped slab per CTA -- full TILE_O
    width and NOT divided by CTA_MMA, because under cga2 each CTA of the pair
    owns a distinct 128-row Q block and gates its own full-width O rows.
    ``sf_bytes`` = the block-scale kernels' scale-factor slabs (Q, K ring, P,
    V ring; ``d256_mxfp8_sf_smem_bytes``), 0 on the f16 / per-tensor kernels.
    """
    s_q = tiles_q * (tile_m * tile_k) * bpe_qkv
    s_o = tiles_q * (tile_m * tile_o) * bpe_o
    s_k = stages_kv * (tile_n * tile_k // cta_mma) * bpe_qkv
    s_v = stages_kv * (tile_o * tile_n // cta_mma) * bpe_qkv
    s_gate = tiles_q * (tile_m * tile_o) * gate_bpe
    return (max(s_q, s_o) if qo_alias else s_q + s_o) + s_k + s_v + s_gate + sf_bytes


# 14-bit ``start_address`` of a version-0 tcgen05 SMEM descriptor: every
# descriptor-fed slab (MMA operand or UTCCP scale-factor source) of a module
# that pins DESC_VERSION=0 must START under this, or it wraps to offset 0 and
# the MMA/UTCCP reads the bottom of SMEM (rules/mma-tma-matrix.md S6).
TCGEN05_V0_ADDR_LIMIT = 256 * 1024

_SMEM_SLAB_ALIGN = 1024  # every kernel slab is a 1024-B-aligned cutlass.Array
_MX_BLOCK = 32  # MXFP8: 32 elements share one E8M0 scale byte


def _align_slab(nbytes: int) -> int:
    return -(-nbytes // _SMEM_SLAB_ALIGN) * _SMEM_SLAB_ALIGN


def _d256_mxfp8_sf_sizes(tile_m: int, tile_n: int, tile_k: int, tile_o: int) -> Tuple[int, int, int, int]:
    """(Q, K-per-stage, P, V-per-stage) scale-factor slab sizes in bytes, the
    kernel's own formulas (``sm107/prefill_d256_mxfp8.py`` SF_SMEM_SIZE_*): a
    128-row tile x ceil(D/128)*128 / 32 for the rowwise operands, TILE_N/32 x
    the 128-padded non-K extent for the BMM2 pair.  Full-size regardless of
    CTA_MMA (the cga2 peers multicast halves into the same slab)."""
    k128 = -(-tile_k // 128) * 128
    return (
        tile_m * k128 // _MX_BLOCK,
        tile_n * k128 // _MX_BLOCK,
        (-(-tile_m // 128) * 128) * tile_n // _MX_BLOCK,
        (-(-tile_o // 128) * 128) * tile_n // _MX_BLOCK,
    )


def d256_mxfp8_sf_smem_bytes(tile_m: int, tile_n: int, tile_k: int, tile_o: int, stages_kv: int) -> int:
    """SMEM the d256 MXFP8 kernel spends on its four scale-factor slabs, as the
    allocator lays them out (declaration order sQ_SF | sK_SF[stages] | sP_SF |
    sV_SF[stages], each slab 1024-B aligned -- the 512 B P slab is padded to the
    next KiB).  At d256, STAGES_KV=2: 1 + 2 + 1 + 2 = 6 KiB."""
    sf_q, sf_k, sf_p, sf_v = _d256_mxfp8_sf_sizes(tile_m, tile_n, tile_k, tile_o)
    return _align_slab(sf_q) + _align_slab(stages_kv * sf_k) + _align_slab(sf_p) + _align_slab(stages_kv * sf_v)


def d256_mxfp8_last_sf_tile_start(cfg) -> int:
    """Byte offset of the LAST scale-factor stage (sV_SF[STAGES_KV-1]) of the
    d256 MXFP8 kernel for ``cfg`` -- the highest address any of its version-0
    UTCCP descriptors names.  Config-side twin of the kernel's own
    ``_last_sf_tile_start`` (a test pins that the two agree), so a depth or an
    O dtype that pushes the scale factors past the 256 KiB window is a typed
    ``ValueError`` at config build rather than LSE = +inf / O = NaN at run time.

    Layout (declaration order in ``_kernel``, each slab 1024-B aligned):
    sQO alias (byte-max of Q@BPE and O@BPE_O -- a bf16 O doubles it) | sK ring |
    sV ring | sQ_SF | sK_SF ring | sP_SF | sV_SF ring.  The epilogue gate tile
    is declared AFTER sV_SF and is not in this sum; declared before it, a bf16 O
    at STAGES_KV=2 would put sQ_SF at exactly 262144.
    """
    sf_q, sf_k, sf_p, sf_v = _d256_mxfp8_sf_sizes(cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O)
    off = 0
    for nbytes in (
        max(cfg.TILE_M * cfg.TILE_K * cfg.BPE, cfg.TILE_M * cfg.TILE_O * cfg.BPE_O),  # sQO alias
        cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K // cfg.CTA_MMA) * cfg.BPE,  # sK ring
        cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N // cfg.CTA_MMA) * cfg.BPE,  # sV ring
        sf_q,  # sQ_SF
        cfg.STAGES_KV * sf_k,  # sK_SF ring
        sf_p,  # sP_SF
    ):
        off = _align_slab(off + nbytes)
    return off + (cfg.STAGES_KV - 1) * sf_v


@dataclass(frozen=True)
class TmaIters:
    """TMA inner-loop iteration counts and per-iteration element granularity."""

    QK_ITERS: int
    VO_ITERS: int
    QK_GRANU_ELEMS: int
    VO_GRANU_ELEMS: int


def _tma_iters_for(d_elems: int, bpe_val: int, swz_b: int) -> int:
    """TMA inner-loop iters when d*BPE exceeds the swizzle atom: each TMA-tile
    load expands into inner_bytes/swz_b sub-loads stepped by swz_b/BPE elems."""
    inner_bytes = d_elems * bpe_val
    if inner_bytes % swz_b != 0:
        raise ValueError(f"TMA inner bytes {inner_bytes} is not a multiple of swizzle {swz_b}")
    return inner_bytes // swz_b


def _tma_iters(cfg) -> TmaIters:
    qk = _tma_iters_for(cfg.TILE_K, cfg.BPE, cfg.Q_SWZ_BYTES)
    vo = _tma_iters_for(cfg.TILE_O, cfg.BPE, cfg.V_SWZ_BYTES)
    return TmaIters(QK_ITERS=qk, VO_ITERS=vo, QK_GRANU_ELEMS=cfg.TILE_K // qk, VO_GRANU_ELEMS=cfg.TILE_O // vo)


def _mask_flags_from(params: TemplateParams) -> int:
    """Kernel-facing MASK bits, derived from the band + padding. The kernels fold
    trace-time branches on these bits; the band VALUES ride WINDOW_LEFT /
    WINDOW_RIGHT (0 when the corresponding bit is unset)."""
    flags = MASK_NONE
    if params.window_right is not None:
        flags |= MASK_CAUSAL
    if params.window_left is not None:
        flags |= MASK_SWA
    if params.thd_varlen or params.seq_kv_lens_present:
        flags |= MASK_PADDED
    return flags


def _validate_params(flavor: str, k: TemplateParams, *, split_wired: bool = False, block_scaled_o_wired: bool = False) -> None:
    """Guard the TemplateParams a Rubin flavor can express. Every rejection here
    must also be a Capabilities decline — reaching this is an engine-row bug.

    ``split_wired`` identifies kernels with both SplitHelpers and FP32
    partial-output stores: half and per-tensor FP8 D128 and D192/V128. The engine
    row and standalone adapter expose the same flavor domain.

    ``block_scaled_o_wired`` says whether THIS flavor's kernel carries the
    block-scaled O epilogue (DTYPE_O 4 = NVFP4 / 5 = MXFP8 output). The two d128
    kernels (per-tensor FP8 and MXFP8) do; the d192xd128 siblings share the d128
    config family but accept DTYPE_O 0..3 only, so a flavor-name test
    ("d128" in flavor) would let them through to a specialization error."""
    # Fused epilogue gate FIRST, so an interaction decline names the feature the
    # caller asked for (`TemplateParams(epilogue_gate=True, split_kv=2)` reads
    # "epilogue_gate is dense, unsplit ...", not the generic Rubin split
    # message below).  Only the flavors whose BODIES carry the gate seams may
    # load with it -- an unlisted flavor would trace an ungated epilogue and
    # silently return O instead of O * sigmoid(G).  The feature interactions
    # are declined here as well as in the rows/adapter twin: THD has no gate
    # descriptor, a KV split would gate the partials the combine then
    # re-normalizes, paged KV and PackGQA are simply not wired through the gate
    # TMA coordinates.
    if k.epilogue_gate:
        if flavor not in _EPILOGUE_GATE_FLAVORS:
            raise ValueError(f"{flavor}: epilogue_gate is wired on {sorted(_EPILOGUE_GATE_FLAVORS)} only")
        if k.thd_varlen or (k.split_kv or 1) > 1 or k.pack_gqa or k.paged_kv:
            raise ValueError(f"{flavor}: epilogue_gate is dense, unsplit, unpaged, non-PackGQA only")
    if k.dtype_qkv not in (_DTYPE_E4M3, _DTYPE_E5M2, _DTYPE_BF16, _DTYPE_FP16):
        raise ValueError(f"{flavor}: dtype_qkv must be 0=E4M3/1=E5M2/2=BF16/3=FP16 (got {k.dtype_qkv}); Rubin has no TF32 prefill kernel")
    dtype_o = resolve_dtype_o(k)
    if dtype_o not in (_DTYPE_E4M3, _DTYPE_E5M2, _DTYPE_BF16, _DTYPE_FP16, _DTYPE_O_NVFP4, _DTYPE_O_MXFP8):
        raise ValueError(f"{flavor}: dtype_o must be 0..5 (got {k.dtype_o})")
    if dtype_o in (_DTYPE_O_NVFP4, _DTYPE_O_MXFP8):
        if k.dtype_qkv > _DTYPE_E5M2:
            raise ValueError(f"{flavor}: block-scaled O (dtype_o {dtype_o}) requires FP8 inputs")
        if not block_scaled_o_wired:
            raise ValueError(f"{flavor}: block-scaled O (dtype_o {dtype_o}) is wired in the d128 kernels (per-tensor FP8, MXFP8) only")
        if k.thd_varlen or k.seq_q_lens_present or (k.split_kv or 1) > 1 or k.pack_gqa:
            raise ValueError(f"{flavor}: block-scaled O (dtype_o {dtype_o}) serves dense, unsplit, unpacked graphs only")
    if k.dtype_qkv > _DTYPE_E5M2 and dtype_o != k.dtype_qkv:
        raise ValueError(f"{flavor}: half input (BF16/FP16) requires dtype_o == dtype_qkv; got dtype_o={dtype_o}")
    if k.sched_policy not in (None, SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"{flavor}: sched_policy must be NATURAL/LPT/LPT_L2 or None (got {k.sched_policy})")
    if k.qh_per_kh < 1:
        raise ValueError(f"{flavor}: qh_per_kh ({k.qh_per_kh}) must be >= 1")
    if k.split_kv and k.split_kv > 1 and not split_wired:
        raise ValueError(f"{flavor}: split_kv > 1 is not wired in this SM107 kernel (no SplitHelpers)")
    # THD/varlen is per-FLAVOR on the Rubin line, not per-dtype.  Every
    # QUANTIZED flavor carries it; on the f16/bf16 side only the flavors whose
    # BODY has been ported to the FROST setup-kernel contract do -- the rest
    # still call it with the pre-upstream 7-arg signature against a 14-arg
    # helper, and allocate the 3B+2 metadata buffer where the SHARED decode
    # (_common_blackwell._thd_decode) reads 4B+4 with a batch_remap.  That
    # mismatch is a HANG or a wrong batch, not an arity error, so it is declined
    # here rather than left to fail deep in a trace.
    if k.thd_varlen and k.dtype_qkv not in (_DTYPE_E4M3, _DTYPE_E5M2) and flavor not in _F16_THD_FLAVORS:
        raise ValueError(
            f"{flavor}: THD/varlen on the SM107 f16/bf16 line is served by {sorted(_F16_THD_FLAVORS)} only "
            f"(got dtype_qkv={k.dtype_qkv}); the other flavors' setup-kernel call sites are not ported"
        )
    if k.seq_q_lens_present:
        if k.thd_varlen:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT (padding mask)")


def _band_fields(params: TemplateParams) -> Tuple[int, int, int, int, int]:
    """(MASK_FLAGS, WINDOW_LEFT, WINDOW_RIGHT, BOTTOM_RIGHT, HAS_SINK).

    WINDOW_* are 0 when their mask bit is unset. Passing window_left without
    window_right is silently wrong rather than partial — the KV-tile trim already
    uses WINDOW_RIGHT, so the element-level mask must receive it too."""
    return (
        _mask_flags_from(params),
        params.window_left or 0,
        params.window_right or 0,
        int(params.bottom_right),
        int(params.has_sink),
    )


# ---------------------------------------------------------------------------
# Shared field surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CfgSm107:
    """Fields every Rubin prefill flavor carries.

    Per-flavor factories set every field explicitly; the defaults exist so the
    dataclass is constructible in a doc/test context, not as a fallback.
    """

    # --- tile geometry
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 128  # d_qk
    TILE_O: int = 128  # d_v

    # --- dtypes
    DTYPE_QKV: int = _DTYPE_FP16  # 0=E4M3 1=E5M2 2=BF16 3=FP16
    DTYPE_O: int = _DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2
    # Block-scaled O (per-tensor FP8 d128 only): scale block along d (0 = plain
    # O) and logical O elements per storage byte (2 for E2M1).
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1

    # --- cluster
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2
    SPLIT_PIPELINE: int = 0

    # --- swizzles
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    # --- MMA k-step (arch-sensitive; see tile_k_hw)
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    # --- pipeline depth
    TILES_Q: int = 2
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 4

    # --- warp roles
    SOFTMAX_WARPGROUPS: int = 2
    CORRECTION_WARPS: int = 4
    SOFTMAX_REGS: int = 192
    CORRECTION_REGS: int = 88
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    RESCALE_THRESHOLD: float = 8.0

    # --- graph-derived features
    MASK_FLAGS: int = MASK_NONE
    # The padded tail form (TemplateParams.padded_tail_form; the f16/bf16 small tiles' PLAIN padded body, MASK_FLAGS == MASK_PADDED
    # alone): 0 = the legacy second (software-max) body for the tail kv tile, 1 = the pre-masked tail (one HW-max body; the tail
    # tile's out-of-range S columns rewritten as -inf in TMEM before the fused max load).  Every kernel seam folds on
    # cutlass.const_expr(CFG.PADDED_TAIL_FORM == 1); the factory refuses 1 off the eligible set (sm107_padded_tail_form_pick).
    PADDED_TAIL_FORM: int = 0
    # The dead-sub-tile quiet body (TemplateParams.dead_subtile_quiet; the one-batch-entry item of the f16/bf16 small tiles, no
    # bias): 1 = a softmax sub-tile with no live row (its first row >= the batch entry's S_q) runs the kv range through the QUIET
    # body (the live body's barrier sequence, no math) and the live arms see an empty range; 0 = it runs the full body.  Every
    # kernel seam folds on cutlass.const_expr(CFG.DEAD_SUBTILE_QUIET == 1); the factory refuses 1 off the eligible set.
    DEAD_SUBTILE_QUIET: int = 0
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    BOTTOM_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    PACK_GQA: int = 0
    QH_PER_KH: int = 1
    SPLIT_KV: int = 1
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 64

    # --- warp population (READ_TILE_ARRIVERS must equal the number of warps that
    # actually call read_tile_id_arrive; a mismatch is an unreachable mbarrier
    # count, i.e. a hang at every shape)
    TOTAL_WARPS: int = 16
    THREADS_PER_CTA: int = 16 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    CORR_WARP_BASE: int = 8
    MMA_WARP_ID: int = 12
    TMALDG_WARP_ID: int = 13
    TMASTG_WARP_ID: int = 14
    SCHED_WARP_ID: int = 15

    # --- mbarrier lane constants
    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256
    READ_TILE_ARRIVERS: int = 15

    # --- fused epilogue gate (O := O * sigmoid(G)); every seam in the kernel
    # bodies folds on cutlass.const_expr(CFG.EPILOGUE_GATE), so 0 traces the
    # pre-gate instruction stream byte-for-byte.  Only the d256 factory ever
    # sets it (config_sm107._EPILOGUE_GATE_FLAVORS); d128/d192/d512 inherit 0.
    EPILOGUE_GATE: int = 0
    # Bytes per gate element: the gate is Q-dtype on the half kernels and bf16
    # on the quantized one -- both 2 B.  Every gate constant in a kernel (TMA
    # box, swizzle, subtile walk) derives from THIS, never from BPE_O: with an
    # e4m3 O the two geometries differ, and a gate walk borrowed from O's
    # passes every bf16-O test and fails only e4m3-O.
    GATE_BPE: int = 2

    # --- additive bias (resident row-block) + per-key boolean mask; d128-file f16/bf16 only.  Every seam in the body
    # folds on cutlass.const_expr(CFG.BIAS_MODE) / (CFG.KV_MASK_FORM), so 0 / 0 traces the pre-bias instruction stream
    # byte-for-byte.  BIAS_RESIDENT_TILES / BIAS_BLOCK_ROWS size the SMEM block (sm107_bias_resident_tiles -- what the
    # carveout leaves after the operands, in whole TILE_N-column tiles); BIAS_EMPTY_ARRIVERS is the mb_bias_empty init
    # count = the softmax warps (one elected arrive each per work item).
    BIAS_MODE: int = 0
    BIAS_RESIDENT_TILES: int = 0
    BIAS_BLOCK_ROWS: int = 0
    BIAS_SWZ_BYTES: int = 128
    BIAS_BOXES_PER_TILE: int = 0
    BIAS_EMPTY_ARRIVERS: int = 0
    KV_MASK_FORM: int = 0
    # The (n, n+1) PAIR work item (TemplateParams.n_chunk): 0 = the one-batch-entry item; n >= 1 = a work item is (a 128-row
    # q tile, a head, a chunk of up to n consecutive flattened-batch rows), the two sub-tiles rows (nb, nb+1) of that q tile
    # sharing the 128-row bias block (BIAS_BLOCK_ROWS = TILE_M).  Every seam folds on cutlass.const_expr(CFG.N_CHUNK > 0).
    N_CHUNK: int = 0
    # The STREAMED bias ring's depth (BIAS_MODE == 2; TemplateParams.bias_stages): 2 / 3 stages of BIAS_BLOCK_ROWS x TILE_N bias
    # tiles, one tile per kv tile, released per kv tile (mb_bias_ring_full / mb_bias_ring_empty, STAGES_BIAS stages each);
    # 0 on BIAS_MODE 0 / 1 (the resident block sizes itself by BIAS_RESIDENT_TILES).  A pure latency knob: no accumulator
    # parity is indexed from this ring.
    STAGES_BIAS: int = 0


@dataclass(frozen=True)
class CfgD128(_CfgSm107):
    """d_qk = d_v = 128 (the classic 2-warpgroup pipeline)."""

    QO_ALIAS: int = 0


@dataclass(frozen=True)
class CfgD64(CfgD128):
    """d_qk = d_v = 64 on the d128 body: the same 16-warp pipeline with ``TILE_K = TILE_O = 64`` (BMM1 in 4 k-steps, a
    64-wide O accumulator, half the Q/K/V/O SMEM).  Serves every f16/bf16 head dim in (32, 64] with D % 8 == 0 by TMA zero
    fill (the SM100 ``CfgD64`` precedent, which measured the zero-filled d128 tile taking the SAME time at d = 64)."""

    TILE_K: int = 64
    TILE_O: int = 64


@dataclass(frozen=True)
class CfgD32(CfgD128):
    """d_qk = d_v = 32 on the d128 body: ``TILE_K = TILE_O = 32`` (BMM1 in 2 k-steps, a 32-wide O accumulator, a quarter of
    the d128 SMEM; 64-byte Q/K/O swizzle atoms).  Serves every f16/bf16 head dim <= 32 with D % 8 == 0 by TMA zero fill.
    At this width the MMA is a quarter of d128's per score element while the softmax's exp / FMA work per element is unchanged,
    so the kernel's floor is the softmax issue rate, not the tensor core."""

    TILE_K: int = 32
    TILE_O: int = 32


@dataclass(frozen=True)
class CfgD192(CfgD128):
    """d_qk = 192, d_v = 128. Same pipeline as d128; Q/K swizzle drops to 64 B at
    FP8 because TILE_K * 1 = 192 is not a multiple of 128."""


@dataclass(frozen=True)
class CfgD256(_CfgSm107):
    """d_qk = d_v = 256. Q-union-O SMEM alias, single softmax warpgroup, a
    Q.K(i+1) -> S.V(i) lookahead MMA stream with two parity S_acc TMEM slots."""


@dataclass(frozen=True)
class CfgD512(_CfgSm107):
    """d_qk = d_v = 512, cga4x1 role-split: the cluster's 4 CTAs form two 2-CTA
    sub-groups (sg0 = BMM1 + softmax + ship, sg1 = recv + correction + BMM2)."""

    XFER_STAGES: int = 2
    BMM2_N_PER_CALL: int = 256
    BMM2_LOOP_N_BLOCKS: int = 2


# ---------------------------------------------------------------------------
# The d128-family TMEM column map -- ONE formula for the kernel body and the validator
# ---------------------------------------------------------------------------

# The top of the allocation holds the softmax stats: 8 columns per sub-tile (alpha; the final (max, sum); the score side
# outputs' true max in columns 2-3) for the two sub-tiles, plus 16 spare.  Keeping STATS at the TOP (TOTAL_COLS - 32 = 544)
# rather than right after the last O slot leaves the d128 map exactly what the kernel always had (S 0 / 128, O 256 / 384,
# STATS 544), so the d128 rendering is byte-identical to the committed records; at the small widths the gap between the
# last O slot and STATS simply grows.
_D128_FAMILY_STATS_REGION_COLS = 32


@dataclass(frozen=True)
class D128FamilyTmemLayout:
    """Column offsets (fp32 TMEM columns, 128 lanes) of the classic two-sub-tile pipeline, see :func:`d128_family_tmem_layout`."""

    TOTAL_COLS: int
    S_OFF: Tuple[int, ...]
    P_OFF: Tuple[int, ...]
    O_OFF: Tuple[int, ...]
    STATS_OFF: int
    STATS_COLS_PER_SUBTILE: int = 8


def d128_family_tmem_layout(cfg, *, stats_cols_per_subtile: int = 8) -> D128FamilyTmemLayout:
    """Derive the d128-family TMEM map from the Cfg: ``S[qs] = qs * TILE_N`` (the fp32 S accumulator, TILE_N columns per
    sub-tile); ``P[qs]`` in the TAIL of its own S slot -- the packed BMM2 A operand is ``TILE_N * BPE / 4`` columns wide
    (bf16 / f16 pack two probabilities per fp32 cell: 64 columns at TILE_N = 128, so P sits at +64; a 4-byte dtype would
    alias the full slot) and BMM1 has finished with S before P is written; ``O[qs] = TILES_Q * TILE_N + qs * TILE_O`` (the
    fp32 O accumulator, TILE_O columns per sub-tile: 128 / 64 / 32); STATS at the top of the allocation.  Every term is a
    Cfg field, never a literal copied from the d128 map (a literal O1 offset of 384 at TILE_O = 32 would leave 352 unused
    columns and no error; a literal P offset of 64 at a TILE_N of 64 would overlap the live S columns).

    ``stats_cols_per_subtile``: 8 on every rendering today (alpha; the final (max, sum); the score side outputs' true max in
    columns 2-3).  The 64-row tile's quad-fragment form (``TILE_M64_LANE_FORM = 1``) publishes each value 8 wide through the
    ``16x256b .x1`` store and takes 24; the region at the top grows by the extra columns per sub-tile, so the default keeps
    STATS_OFF at 544 and every committed record byte-identical."""
    if stats_cols_per_subtile < 8 or stats_cols_per_subtile % 8:
        raise ValueError(f"d128_family_tmem_layout: stats_cols_per_subtile must be a multiple of 8, at least 8 (got {stats_cols_per_subtile})")
    stats_region = _D128_FAMILY_STATS_REGION_COLS + cfg.TILES_Q * (stats_cols_per_subtile - 8)
    s_off = tuple(qs * cfg.TILE_N for qs in range(cfg.TILES_Q))
    p_width = (cfg.TILE_N * cfg.BPE) // 4
    p_off = tuple(s + cfg.TILE_N - p_width for s in s_off)
    o_base = cfg.TILES_Q * cfg.TILE_N
    o_off = tuple(o_base + qs * cfg.TILE_O for qs in range(cfg.TILES_Q))
    return D128FamilyTmemLayout(
        TOTAL_COLS=TMEM_TOTAL_COLS,
        S_OFF=s_off,
        P_OFF=p_off,
        O_OFF=o_off,
        STATS_OFF=TMEM_TOTAL_COLS - stats_region,
        STATS_COLS_PER_SUBTILE=stats_cols_per_subtile,
    )


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------


def _check(preds) -> None:
    for ok, msg in preds:
        if not ok:
            raise ValueError(msg)


def _validate_head_dim_tile(cfg, flavor: str) -> None:
    """The head-dim tile and everything coupled to it (frost-kernels.md s5 "bind the couplings" / s6).

    One Cfg field, ``TILE_K`` / ``TILE_O``, sets a TMA box, an MMA descriptor swizzle, the epilogue's SMEM store swizzle,
    the TMEM O slot width and the BMM1 k-step count at once; moving it without the rest raises HERE with the failure it
    would otherwise produce, instead of a numerically perfect slow kernel or a cos ~ 0.006 one."""
    lay = d128_family_tmem_layout(cfg)
    q_swz, v_swz, o_swz = q_swz_bytes(cfg.TILE_K, cfg.BPE), v_swz_bytes(cfg.TILE_O, cfg.CTA_MMA, cfg.BPE), o_swz_bytes(cfg.TILE_O, cfg.BPE_O, cfg.O_PACK_DIV)
    _check(
        [
            (
                cfg.TILE_K in (32, 64, 128, 192) and cfg.TILE_O in (32, 64, 128),
                f"{flavor}: TILE_K must be one of 32/64/128/192 and TILE_O one of 32/64/128 (got {cfg.TILE_K}/{cfg.TILE_O})",
            ),
            (
                cfg.TILE_K == cfg.TILE_O or (cfg.TILE_K, cfg.TILE_O) == (192, 128),
                f"{flavor}: the d128 family serves TILE_K == TILE_O (32/64/128) or the d192xd128 pair (got {cfg.TILE_K}/{cfg.TILE_O})",
            ),
            (
                cfg.TILE_M in SMALL_D_TILE_MS,
                f"{flavor}: TILE_M must be one of {sorted(SMALL_D_TILE_MS)} (got {cfg.TILE_M}): the softmax / correction lane mapping and the "
                f"tcgen05.mma datapath layout (M = 128: one lane per row; M = 64: 16 live rows per 32-lane TMEM sub-partition) exist for these two only",
            ),
            (
                cfg.TILE_M == 128 or (cfg.TILE_K <= 64 and cfg.DTYPE_QKV in (_DTYPE_BF16, _DTYPE_FP16)),
                f"{flavor}: the 64-row Q tile is wired on the f16/bf16 32- / 64-wide tiles only (TILE_K={cfg.TILE_K}, DTYPE_QKV={cfg.DTYPE_QKV}); the "
                f"d128 / d192 renderings keep their committed records and the quantized siblings' k_dim / SF geometry is tied to the 128-row tile",
            ),
            (
                cfg.TILE_M == 128 or cfg.CTA_MMA == 1,
                f"{flavor}: the 64-row Q tile runs cga1 only (CTA_MMA={cfg.CTA_MMA}): a cta_group::2 MMA with a 64-row per-CTA tile has a different "
                f"TMEM datapath layout than the one the warp-lane -> row mapping was derived for",
            ),
            (
                cfg.TILE_M == 128 or not (cfg.THD_VARLEN or cfg.SPLIT_KV > 1 or cfg.PACK_GQA or cfg.EPILOGUE_GATE),
                f"{flavor}: the 64-row Q tile serves dense, unsplit, non-PackGQA, ungated graphs only (the small-tile rows' own gates)",
            ),
            (
                cfg.TILE_K > 64 or cfg.STAGES_KV in SMALL_D_STAGES_KV,
                f"{flavor}: the 32- / 64-wide tiles' K / V ring is validated at STAGES_KV in {sorted(SMALL_D_STAGES_KV)} (got {cfg.STAGES_KV}); the body is "
                f"depth-generic but the SMEM fit and the ring-wrap census hold for these depths only -- a depth of 1 serialises every kv tile behind "
                f"its own BMM1 and a depth above 4 was never run",
            ),
            (
                cfg.TILE_K % cfg.TILE_K_HW_BMM1 == 0 and cfg.TILE_O % 16 == 0,
                f"{flavor}: TILE_K must be whole BMM1 k-steps of {cfg.TILE_K_HW_BMM1} and TILE_O a multiple of the correction warps' 16-column O chunk",
            ),
            (
                cfg.Q_SWZ_BYTES == q_swz and cfg.K_SWZ_BYTES == q_swz,
                f"{flavor}: Q/K swizzle {cfg.Q_SWZ_BYTES}/{cfg.K_SWZ_BYTES} B must follow the tile width ({q_swz} B rows at TILE_K={cfg.TILE_K}, BPE={cfg.BPE}): "
                f"the TMA descriptor and the MMA descriptor read the same atom, and a mismatch is cos ~ 0.006 on every output, not a slow kernel",
            ),
            (
                cfg.V_SWZ_BYTES == v_swz,
                f"{flavor}: V swizzle {cfg.V_SWZ_BYTES} B must follow the per-CTA d_v extent ({v_swz} B at TILE_O={cfg.TILE_O}, CTA_MMA={cfg.CTA_MMA})",
            ),
            (
                cfg.O_SWZ_BYTES == o_swz,
                f"{flavor}: O swizzle {cfg.O_SWZ_BYTES} B must follow the O row ({o_swz} B at TILE_O={cfg.TILE_O}): the epilogue's swizzled SMEM store and the O TMA descriptor are ONE unit",
            ),
            (
                (cfg.TILE_K * cfg.BPE) % cfg.Q_SWZ_BYTES == 0
                and (cfg.TILE_O * cfg.BPE) % cfg.V_SWZ_BYTES == 0
                and (cfg.TILE_O * cfg.BPE_O // cfg.O_PACK_DIV) % cfg.O_SWZ_BYTES == 0,
                f"{flavor}: every TMA inner extent must be whole swizzle atoms (TMA_QK/VO/O iteration counts are integers)",
            ),
            (
                cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N,
                f"{flavor}: the softmax chunking is TILE_N-derived (the KV axis), never TILE_O's: N_BMM2_CHUNKS x BMM2_CHUNK_SIZE must equal TILE_N "
                f"(got {cfg.N_BMM2_CHUNKS} x {cfg.BMM2_CHUNK_SIZE} vs {cfg.TILE_N}); a TILE_O-scaled chunk reads half of S and signals half the bmm2_ready barriers -- a silent NaN or a hang",
            ),
            (
                lay.O_OFF[-1] + cfg.TILE_O <= lay.STATS_OFF,
                f"{flavor}: TMEM map S {lay.S_OFF} | P {lay.P_OFF} | O {lay.O_OFF} (+{cfg.TILE_O}) must end at or before the STATS region (column {lay.STATS_OFF} of {lay.TOTAL_COLS}); "
                f"an overlap accumulates the next KV tile's O under the softmax stats",
            ),
            (
                all(o % 32 == 0 for o in lay.O_OFF)
                and all(sv % 32 == 0 for sv in lay.S_OFF)
                and all(pv % 8 == 0 for pv in lay.P_OFF)
                and lay.STATS_OFF % 8 == 0,
                f"{flavor}: accumulator slot bases must be 32-column aligned and P / STATS bases 8-column aligned (S {lay.S_OFF}, P {lay.P_OFF}, O {lay.O_OFF}, STATS {lay.STATS_OFF})",
            ),
        ]
    )


def _validate_regs(cfg, flavor: str) -> None:
    """The four hardware register constraints. Equality and alignment are HW
    requirements, not style: violating them gives wrong results or a crash, with
    no diagnostic."""
    _check(
        [
            (
                cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS,
                f"{flavor}: MMA/TMALDG/TMASTG/SCHEDULER regs must be equal",
            ),
            (
                cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512,
                f"{flavor}: register budget over 512",
            ),
            (
                all(r % 8 == 0 for r in (cfg.MMA_REGS, cfg.CORRECTION_REGS, cfg.SOFTMAX_REGS, cfg.OTHER_REGS)),
                f"{flavor}: every per-role register count must be a multiple of 8",
            ),
            (
                all(24 <= r <= 256 for r in (cfg.MMA_REGS, cfg.SOFTMAX_REGS) if r),
                f"{flavor}: per-warp register counts must be within 24..256",
            ),
        ]
    )


def _validate_warp_layout(cfg, flavor: str, *, expect_total: int, expect_arrivers: int) -> None:
    """Pin the warp population against the count the kernel BODY implements.

    This is the check whose absence hung the d256 FP8 port: the body's role
    dispatch and its ``read_tile_id_arrive`` call sites are fixed, so a config
    that disagrees produces an mbarrier init count nothing can reach."""
    _check(
        [
            (cfg.TOTAL_WARPS == expect_total, f"{flavor}: kernel body has {expect_total} warps, config says {cfg.TOTAL_WARPS}"),
            (cfg.THREADS_PER_CTA == cfg.TOTAL_WARPS * 32, f"{flavor}: THREADS_PER_CTA must be TOTAL_WARPS*32"),
            (
                cfg.READ_TILE_ARRIVERS == expect_arrivers,
                f"{flavor}: READ_TILE_ARRIVERS must be {expect_arrivers} for this body (got {cfg.READ_TILE_ARRIVERS}); "
                f"a wrong count is an unreachable scheduler mbarrier, i.e. a hang at EVERY shape",
            ),
        ]
    )


def _validate_dtype_k_step(cfg, flavor: str) -> None:
    """FP8/MXFP8 must ride Rubin's K=64 dense path (the bodies hardcode idesc
    k_dim=1); f16/bf16 must stay 1-chunk at 16."""
    want = tile_k_hw(cfg.DTYPE_QKV)
    _check(
        [
            (
                cfg.TILE_K_HW_BMM1 == want and cfg.TILE_K_HW_BMM2 == want,
                f"{flavor}: TILE_K_HW must be {want} at DTYPE_QKV={cfg.DTYPE_QKV} on Rubin "
                f"(got {cfg.TILE_K_HW_BMM1}/{cfg.TILE_K_HW_BMM2}); a mismatch against the body's "
                f"idesc k_dim silently scrambles accumulator rows",
            ),
        ]
    )


def _validate_smem(cfg, flavor: str, *, qo_alias: bool, sf_bytes: int = 0, bias_bytes: int = 0) -> None:
    """``sf_bytes``: the block-scale kernel's scale-factor slabs (0 on the f16 /
    per-tensor flavors); ``bias_bytes``: the resident bias row-block of the f16 d128-file
    tiles (0 without the bias); both named in the message so a reader sees every term."""
    gate_bpe = cfg.GATE_BPE if cfg.EPILOGUE_GATE else 0
    gate_kib = (cfg.TILES_Q * cfg.TILE_M * cfg.TILE_O * gate_bpe) // 1024
    used = (
        sdpa_smem_bytes(
            cfg.TILE_M,
            cfg.TILE_N,
            cfg.TILE_K,
            cfg.TILE_O,
            cfg.TILES_Q,
            cfg.STAGES_KV,
            cfg.CTA_MMA,
            cfg.BPE,
            cfg.BPE_O,
            qo_alias=qo_alias,
            gate_bpe=gate_bpe,
            sf_bytes=sf_bytes,
        )
        + bias_bytes
        + _SMEM_FIXED_OVERHEAD
    )
    if used > SMEM_USABLE_BYTES:
        gate_term = f" + {gate_kib} KiB GATE staging tile" if gate_bpe else ""
        sf_term = (
            f" + {sf_bytes // 1024} KiB MXFP8 scale-factor slabs (Q 1 | K {cfg.STAGES_KV}x1 | P 1 (512 B padded) | V {cfg.STAGES_KV}x1)" if sf_bytes else ""
        )
        if bias_bytes:
            _streamed = getattr(cfg, "BIAS_MODE", 0) == BIAS_MODE_STREAMED
            gate_term += (
                f" + {bias_bytes // 1024} KiB {'streamed bias ring' if _streamed else 'resident bias block'} "
                f"({getattr(cfg, 'STAGES_BIAS', 0) if _streamed else getattr(cfg, 'BIAS_RESIDENT_TILES', 0)} {'stages' if _streamed else 'tiles'} x "
                f"{getattr(cfg, 'BIAS_BLOCK_ROWS', 0)} rows x {cfg.TILE_N} cols x {cfg.BPE} B)"
            )
        raise ValueError(
            f"{flavor}: SMEM {used // 1024} KiB (Q/K/V/O + {_SMEM_FIXED_OVERHEAD // 1024} KiB fixed{gate_term}{sf_term}) exceeds the "
            f"{SMEM_USABLE_BYTES // 1024} KiB usable Rubin carveout at STAGES_KV={cfg.STAGES_KV}, CTA_MMA={cfg.CTA_MMA}"
            f"{', EPILOGUE_GATE=1' if gate_bpe else ''}. "
            f"Overflowing it does NOT fail the launch -- it clobbers the last buffer allocated (measured: O came back "
            f"50 % zeros with an exact LSE), so this must raise here"
        )


def _validate_swizzles(cfg, flavor: str) -> None:
    _check(
        [
            (cfg.Q_SWZ_BYTES in (64, 128), f"{flavor}: Q swizzle must be 64/128 B"),
            (cfg.K_SWZ_BYTES in (64, 128), f"{flavor}: K swizzle must be 64/128 B"),
            (cfg.V_SWZ_BYTES in (32, 64, 128), f"{flavor}: V swizzle must be 32/64/128 B"),
            (cfg.O_SWZ_BYTES in (64, 128), f"{flavor}: O swizzle must be 64/128 B"),
        ]
    )


def _validate_masks(cfg, flavor: str) -> None:
    _check(
        [
            (
                not (cfg.BOTTOM_RIGHT and not (cfg.MASK_FLAGS & MASK_CAUSAL)),
                f"{flavor}: bottom-right alignment requires a causal band",
            ),
            (
                not (cfg.THD_VARLEN and not (cfg.MASK_FLAGS & MASK_PADDED)),
                f"{flavor}: THD/varlen implies per-sequence padded masking",
            ),
            (
                not (cfg.THD_VARLEN and not cfg.SEQ_KV_LENS_PRESENT),
                f"{flavor}: THD/varlen must force SEQ_KV_LENS_PRESENT=1, or every sequence attends the whole pack",
            ),
            (
                not (
                    int(getattr(cfg, "DEAD_SUBTILE_QUIET", 0)) == 1
                    and (
                        int(getattr(cfg, "BIAS_MODE", 0))
                        or int(getattr(cfg, "N_CHUNK", 0))
                        or cfg.PACK_GQA
                        or cfg.THD_VARLEN
                        or int(getattr(cfg, "EPILOGUE_GATE", 0))
                    )
                ),
                f"{flavor}: DEAD_SUBTILE_QUIET=1 (the dead-sub-tile quiet body) needs the one-batch-entry item without a bias, PackGQA, THD or the "
                f"gated epilogue -- the quiet body replicates the plain row mapping's barrier sequence and none of the bias barriers (got "
                f"BIAS_MODE={getattr(cfg, 'BIAS_MODE', 0)}, N_CHUNK={getattr(cfg, 'N_CHUNK', 0)}, PACK_GQA={cfg.PACK_GQA}, THD_VARLEN={cfg.THD_VARLEN}, "
                f"EPILOGUE_GATE={getattr(cfg, 'EPILOGUE_GATE', 0)}); a bias arm kept quiet would skip the mb_bias_full wait its init count expects",
            ),
            (
                not (
                    int(getattr(cfg, "PADDED_TAIL_FORM", 0)) == 1
                    and (cfg.MASK_FLAGS != MASK_PADDED or int(getattr(cfg, "BIAS_MODE", 0)) or int(getattr(cfg, "KV_MASK_FORM", 0)))
                ),
                f"{flavor}: PADDED_TAIL_FORM=1 (the pre-masked tail) needs MASK_FLAGS == MASK_PADDED alone and no bias / kv_mask -- its single HW-max "
                f"loop has no masked segment for a band, and the -inf arms are one body already (got MASK_FLAGS={cfg.MASK_FLAGS}, "
                f"BIAS_MODE={getattr(cfg, 'BIAS_MODE', 0)}, KV_MASK_FORM={getattr(cfg, 'KV_MASK_FORM', 0)}); a band's masked tail silently kept "
                f"would read -inf on no column and the HW max would see the padding",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# d128 / d192 -- the classic 2-warpgroup pipeline (16 warps)
# ---------------------------------------------------------------------------
#
# Warp map: 8 softmax (2 warpgroups) + 4 correction + MMA + TMALDG + TMASTG +
# scheduler.  15 of the 16 warps arrive on the scheduler ring (all but the
# scheduler warp itself).


_D128_TOTAL_WARPS = 16
_D128_READ_TILE_ARRIVERS = 15


def _stages_kv_d128(dtype_qkv: int, cta_mma: int, *, mxfp8: bool, tile_k: int) -> int:
    """Rubin d128-family ring depth.

    Per-tensor FP8 at d128/cga2 runs a **9-stage** ring: a Rubin-specific tuning
    carried by the shipped ``sm107/prefill_d128_fp8.py`` (which previously spelled
    it as a post-hoc ``dataclasses.replace``), and the reason that kernel needs
    the oversized-SMEM launch mode.

    It is deliberately narrow. At cga1 the per-CTA K/V slabs double and 9 stages
    need 352 KiB, past the 327 KiB cap; at d192 (``tile_k=192``) the Q/K slabs
    grow and the tuning was never measured. Both fall back to 4, which is what
    the kernel bodies were validated with.
    """
    if dtype_qkv <= _DTYPE_E5M2:
        if not mxfp8 and tile_k == 128 and cta_mma == 2:
            return 9
        return 4
    # d192's wider Q/K slabs do not leave room for a 4-deep ring: Q/K/V/O alone
    # sum to exactly 320 KiB, which overflows the usable carveout once barriers
    # are counted. Measured on Rubin at n_kv=2: depth 4 -> 50 % of O zeroed,
    # cos = 0.0015; depth 2 -> cos = 1.000000. d128 at depth 4 sums to 256 KiB
    # and passes every n_kv, so this is a d192 constraint, not a family one.
    if tile_k == 192:
        return 2
    if tile_k <= 64:
        # The small head-dim tiles (TILE_K = TILE_O in {32, 64}): a quarter / half of d128's slabs, so a 4-deep ring is
        # 96 / 192 KiB at cga1 (64 / 128 KiB at cga2), inside the carveout at either width, and 4 is the depth the SM100 d64
        # flavor runs (its ncu read long_scoreboard as the dominant stall).  A perf lever, not a body constraint (the ring
        # is depth-generic; the body indexes no accumulator parity from it; ``PipelineState.advance`` runs the index and the
        # phase over any depth >= 1 on both the TMA-LDG and the MMA side, the pair item's two tiles per kv step included).
        # ``_STAGES_KV_OVERRIDE_FOR_SWEEPS`` (2 / 3 / 4, the depths the SMEM validator and the body were validated at) pins
        # the depth for the A/B of the small-tile knob sweep; it is read HERE so the bias budgets (``sm107_bias_resident_tiles``
        # / ``sm107_bias_ring_stages``, which call this helper) and the factory see ONE depth (rule 8b').  Measured on a
        # 212-SM cc 10.7 node at the 2376 MHz lock (bf16, graph replay, control pair; the knob sweep's A/B/A): see the
        # STAGES_KV row of the small-tile heuristics notes before moving the default.
        if _STAGES_KV_OVERRIDE_FOR_SWEEPS is not None:
            want = int(_STAGES_KV_OVERRIDE_FOR_SWEEPS)
            if want not in SMALL_D_STAGES_KV:
                raise ValueError(
                    f"_STAGES_KV_OVERRIDE_FOR_SWEEPS={want}: the 32- / 64-wide tiles' K / V ring is validated at {sorted(SMALL_D_STAGES_KV)} stages "
                    f"(SMEM fit + the ring-wrap axis of the fresh-process census); a depth outside it is unvalidated, not merely slow"
                )
            return want
        return 4
    return 2 * cta_mma


# The flavors that serve the bias / kv_mask at SOME validated cga -- DERIVED from the resident budget (the d128 tile has no
# spare for a block at either cga: its Q / O / K / V sum to 256 KiB).  The engine row's ``bias_d_shapes`` and the adapter
# read this set (rule 8b').  Defined here, after the ring-depth rule the budget reads.
SM107_BIAS_D_SHAPES = frozenset(
    shape
    for shape in SM107_F16_SMALL_D_SHAPES | {(128, 128)}
    if any(sm107_bias_resident_tiles(shape[0], cga) > 0 for cga in SM107_SMALL_D_CGAS.get(shape, {2}))
)


def operand_region_end_bytes(cfg) -> int:
    """Where the descriptor-fed region of a d128-family kernel ENDS: sQ, sO, sK, sV in the body's declaration order, each a
    1024-B-aligned slab -- the roots every tcgen05 SMEM descriptor is built from.  The lane-read bias block is declared
    AFTER them, so with this below ``TCGEN05_V0_ADDR_LIMIT`` every descriptor root stays inside the version-0 window
    whatever the block's size (the block itself may sit past 256 KiB: no descriptor touches it)."""
    s_q = _align_slab(cfg.TILES_Q * cfg.TILE_M * cfg.TILE_K * cfg.BPE)
    s_o = _align_slab(cfg.TILES_Q * cfg.TILE_M * cfg.TILE_O * cfg.BPE_O)
    s_k = _align_slab(cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K // cfg.CTA_MMA) * cfg.BPE)
    s_v = _align_slab(cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N // cfg.CTA_MMA) * cfg.BPE)
    return s_q + s_o + s_k + s_v


def bias_smem_tiles(cfg) -> int:
    """The bias block's SMEM extent in ``BIAS_BLOCK_ROWS x TILE_N`` tiles, by FORM: ``BIAS_RESIDENT_TILES`` on the resident block
    (BIAS_MODE 1), ``STAGES_BIAS`` on the streamed ring (BIAS_MODE 2), 0 without the bias.  The kernel hands this to
    ``tile_dsl.bias.bias_block_geometry(n_tiles=)`` so the TMA box count, the expect_tx and the lane offsets stay one arithmetic."""
    if cfg.BIAS_MODE == BIAS_MODE_RESIDENT:
        return cfg.BIAS_RESIDENT_TILES
    if cfg.BIAS_MODE == BIAS_MODE_STREAMED:
        return cfg.STAGES_BIAS
    return 0


def bias_resident_block_bytes(cfg) -> int:
    """SMEM bytes of the bias block in its FORM (0 without the bias): :func:`bias_smem_tiles` tiles of ``BIAS_BLOCK_ROWS x TILE_N``
    elements in the io dtype -- the resident block's tiles or the streamed ring's stages (the name predates the ring; the
    ``_validate_smem`` term it feeds names the form)."""
    return bias_smem_tiles(cfg) * cfg.BIAS_BLOCK_ROWS * cfg.TILE_N * cfg.BPE


def _validate_tile_n(cfg, flavor: str) -> None:
    """The KV tile width and what it couples to (M-2: TILE_N as a Cfg field).  The S accumulator takes TILE_N fp32 columns
    per sub-tile and P half of them in its tail; with two sub-tiles the TMEM map must still end under the STATS region, so
    256 is served only at ONE sub-tile -- a predicate, not a comment."""
    lay = d128_family_tmem_layout(cfg)
    _check(
        [
            (cfg.TILE_N in (64, 128, 256), f"{flavor}: TILE_N must be one of 64 / 128 / 256 (got {cfg.TILE_N})"),
            (
                cfg.TILE_N != 256 or cfg.TILES_Q == 1,
                f"{flavor}: TILE_N = 256 needs ONE softmax sub-tile: {cfg.TILES_Q} sub-tiles x 256 S columns + the O slots + STATS exceed the "
                f"{lay.TOTAL_COLS}-column TMEM allocation (the body indexes its S / P parity slots per sub-tile)",
            ),
            (
                cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N and cfg.BMM2_CHUNK_SIZE == 64,
                f"{flavor}: the softmax chunking is TILE_N / 64 chunks of 64 (got {cfg.N_BMM2_CHUNKS} x {cfg.BMM2_CHUNK_SIZE} vs TILE_N={cfg.TILE_N})",
            ),
            (
                lay.O_OFF[-1] + cfg.TILE_O <= lay.STATS_OFF,
                f"{flavor}: TILES_Q x TILE_N + TILES_Q x TILE_O = {lay.O_OFF[-1] + cfg.TILE_O} TMEM columns must end at or before STATS ({lay.STATS_OFF})",
            ),
        ]
    )


def _validate_bias(cfg, flavor: str) -> None:
    """The resident bias block and the kv_mask word: every coupling of the block to the body, as predicates whose
    messages name the failure (frost-kernels s5 "bind the couplings"; sdpa-invariants s6)."""
    if not cfg.BIAS_MODE and not cfg.KV_MASK_FORM:
        _check(
            [
                (cfg.BIAS_RESIDENT_TILES == 0 and cfg.BIAS_BLOCK_ROWS == 0, f"{flavor}: bias geometry fields must be 0 without BIAS_MODE"),
            ]
        )
        return
    preds = [
        (cfg.BIAS_MODE in BIAS_MODES, f"{flavor}: BIAS_MODE must be 0 (none), 1 (resident row-block) or 2 (streamed ring); got {cfg.BIAS_MODE}"),
        (
            cfg.KV_MASK_FORM in KV_MASK_FORMS,
            f"{flavor}: KV_MASK_FORM must be 0 (none), 1 (bit word), 2 (additive io-dtype vector) or 3 (additive fp32 vector); got {cfg.KV_MASK_FORM}",
        ),
        (
            cfg.BIAS_EMPTY_ARRIVERS == cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS,
            f"{flavor}: BIAS_EMPTY_ARRIVERS must equal the softmax warp count {cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS} (one elected arrive per "
            f"softmax warp per work item; got {cfg.BIAS_EMPTY_ARRIVERS}) -- an unreachable mbarrier count is a hang at EVERY shape",
        ),
        (
            cfg.DTYPE_QKV in (_DTYPE_BF16, _DTYPE_FP16),
            f"{flavor}: the bias / kv_mask arms are wired on the f16/bf16 kernel only (dtype_qkv={cfg.DTYPE_QKV})",
        ),
        (
            not (cfg.THD_VARLEN or cfg.SPLIT_KV > 1 or cfg.PACK_GQA or cfg.EPILOGUE_GATE),
            f"{flavor}: the bias / kv_mask arms serve dense, unsplit, unpacked, ungated graphs only",
        ),
    ]
    if cfg.BIAS_MODE:
        tile_bytes = cfg.BIAS_BLOCK_ROWS * cfg.TILE_N * cfg.BPE
        preds += [
            (
                cfg.BIAS_MODE != BIAS_MODE_RESIDENT or (cfg.BIAS_RESIDENT_TILES >= 1 and cfg.STAGES_BIAS == 0),
                f"{flavor}: a resident bias block needs at least one {cfg.TILE_N}-column tile and no ring (BIAS_RESIDENT_TILES={cfg.BIAS_RESIDENT_TILES}, "
                f"STAGES_BIAS={cfg.STAGES_BIAS})",
            ),
            (
                cfg.BIAS_MODE != BIAS_MODE_STREAMED or (cfg.STAGES_BIAS in BIAS_RING_STAGES and cfg.BIAS_RESIDENT_TILES == 0),
                f"{flavor}: the streamed bias ring is {sorted(BIAS_RING_STAGES)} stages of one {cfg.TILE_N}-column tile and holds no resident block "
                f"(STAGES_BIAS={cfg.STAGES_BIAS}, BIAS_RESIDENT_TILES={cfg.BIAS_RESIDENT_TILES}); the TMA-LDG side pre-arms STAGES_BIAS fills and the "
                f"softmax side advances the same count -- the two sides read ONE Cfg field",
            ),
            (
                cfg.BIAS_BLOCK_ROWS == (cfg.TILE_M if cfg.N_CHUNK > 0 else cfg.TILES_Q * cfg.TILE_M),
                f"{flavor}: BIAS_BLOCK_ROWS must be the Q rows the work item's sub-tiles READ: TILE_M = {cfg.TILE_M} on the (n, n+1) pair item "
                f"(N_CHUNK={cfg.N_CHUNK} > 0: both sub-tiles are rows of ONE q tile and share the block -- a 256-row block would feed row n's bias "
                f"into row n + 1's sub-tile, finite and wrong, no crash), TILES_Q x TILE_M = {cfg.TILES_Q * cfg.TILE_M} on the one-batch-entry item "
                f"(got {cfg.BIAS_BLOCK_ROWS})",
            ),
            (
                cfg.N_CHUNK >= 0,
                f"{flavor}: N_CHUNK must be >= 0 (0 = the one-batch-entry item; got {cfg.N_CHUNK})",
            ),
            (
                cfg.N_CHUNK == 0 or cfg.CTA_MMA == 1,
                f"{flavor}: the (n, n+1) pair item runs cga1 only (CTA_MMA={cfg.CTA_MMA}): a cta_group::2 collective MMA shares K across both CTAs' rows, "
                f"and rows n / n + 1 are DIFFERENT batch entries with different K / V",
            ),
            (
                cfg.N_CHUNK == 0 or cfg.TILES_Q == 2,
                f"{flavor}: the pair item is the two-sub-tile pipeline (TILES_Q={cfg.TILES_Q}): sub-tile qs is row n + qs of the chunk",
            ),
            (
                cfg.N_CHUNK == 0 or not cfg.BOTTOM_RIGHT,
                f"{flavor}: the pair item's per-step KV range is the UNION of two rows' top-left bands (monotone in the KV length); the bottom-right "
                f"diagonal differs per row -- not served (the graph contract refuses bottom-right causal with a bias anyway)",
            ),
            (
                cfg.N_CHUNK == 0 or not (cfg.THD_VARLEN or cfg.SPLIT_KV > 1 or cfg.PACK_GQA or cfg.EPILOGUE_GATE),
                f"{flavor}: the pair item is the dense, unsplit, non-PackGQA, ungated arm (the bias arm's own gates)",
            ),
            (
                cfg.BIAS_SWZ_BYTES == BIAS_SWZ_BYTES and cfg.BIAS_BOXES_PER_TILE == (cfg.TILE_N * cfg.BPE) // BIAS_SWZ_BYTES,
                f"{flavor}: the bias block is TMA-written AND lane-read under the 128-B swizzle (BIAS_SWZ_BYTES={cfg.BIAS_SWZ_BYTES}); a tile is "
                f"TILE_N x BPE / 128 = {(cfg.TILE_N * cfg.BPE) // BIAS_SWZ_BYTES} boxes of 64 columns (got {cfg.BIAS_BOXES_PER_TILE})",
            ),
            (
                cfg.BIAS_BLOCK_ROWS <= 256,
                f"{flavor}: a TMA box holds at most 256 rows; the block's {cfg.BIAS_BLOCK_ROWS} rows need a per-sub-tile box",
            ),
            (
                operand_region_end_bytes(cfg) <= TCGEN05_V0_ADDR_LIMIT,
                f"{flavor}: the descriptor-fed sQ / sO / sK / sV region ends at {operand_region_end_bytes(cfg)} B, past the 256 KiB version-0 tcgen05 "
                f"descriptor window: a root at or above 256 KiB wraps to offset 0 (the accumulator is EXACTLY zero); the bias block is placed after "
                f"them and may straddle the line, the operands may not",
            ),
            (
                bias_smem_tiles(cfg) * tile_bytes <= SMEM_USABLE_BYTES,
                f"{flavor}: {bias_smem_tiles(cfg)} bias tiles x {tile_bytes} B exceed the carveout on their own",
            ),
        ]
    else:
        preds += [(cfg.STAGES_BIAS == 0 and cfg.BIAS_RESIDENT_TILES == 0, f"{flavor}: bias geometry fields must be 0 without BIAS_MODE")]
    _check(preds)


def _make_cfg_d128_family(params: TemplateParams, *, flavor: str, tile_k: int, tile_o: int, mxfp8: bool):
    # Half and per-tensor FP8 D128/D192 use the shared split protocol.
    # MXFP8 has no split loop yet.
    split_wired = not mxfp8 and tile_k in (128, 192) and tile_o == 128
    # The block-scaled O epilogue lives in the two d128 kernels (per-tensor fp8 and
    # mxfp8); the d192xd128 siblings share this config family but not the epilogue.
    block_scaled_o_wired = tile_k == 128 and tile_o == 128
    _validate_params(flavor, params, split_wired=split_wired, block_scaled_o_wired=block_scaled_o_wired)
    # The f16x2-exponent arm and the pre-folded-scale arm live in the d128 MXFP8 kernel body (and the
    # per-tensor FP8 sibling carries the f16x2 exponent on its own); the d192xd128 kernels share this
    # config family but not those arms, so a flavor-name test would let them through to a silent no-op.
    if params.softmax_f16 and mxfp8 and tile_k != 128:
        raise ValueError(f"{flavor}: softmax_f16 on MXFP8 is wired on the d128 kernel only")
    if params.softmax_scale_prefolded and not (mxfp8 and tile_k == 128):
        raise ValueError(f"{flavor}: softmax_scale_prefolded is wired on the d128 MXFP8 kernel only")
    cta_mma = params.cta_mma
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    o_div = o_pack_div(dtype_o)
    # TILE_N is a Cfg field (TemplateParams.tile_n): the quantized siblings keep 128 (their SF geometry is tied to the 128-column
    # tile), the f16/bf16 small tiles take 64 / 128 (SM107_TILE_NS_BY_D_SHAPE; the 64-wide rendering is trace-compiled and served
    # through the TILE_N knob), the d128 rendering keeps 128; 256 is refused by _validate_tile_n.
    tile_n = int(getattr(params, "tile_n", 128))
    if tile_n != 128 and (params.dtype_qkv <= _DTYPE_E5M2 or tile_k > 64):
        raise ValueError(f"{flavor}: TILE_N = {tile_n} is wired on the f16/bf16 32- / 64-wide tiles only (the quantized and d128 renderings keep 128)")
    bias_mode = int(getattr(params, "bias_mode", 0))
    kv_mask_form = int(getattr(params, "kv_mask_form", 0))
    if bias_mode or kv_mask_form:
        if params.dtype_qkv <= _DTYPE_E5M2 or mxfp8:
            raise ValueError(f"{flavor}: the additive bias / kv_mask are wired on the f16/bf16 prefill kernel only (dtype_qkv={params.dtype_qkv})")
        if params.thd_varlen or (params.split_kv or 1) > 1 or params.pack_gqa or params.paged_kv or params.epilogue_gate:
            raise ValueError(f"{flavor}: the additive bias / kv_mask serve dense, unsplit, unpaged, non-PackGQA, ungated graphs only")
        if bias_mode not in BIAS_MODES or kv_mask_form not in KV_MASK_FORMS:
            raise ValueError(
                f"{flavor}: bias_mode must be 0 / 1 (resident row-block) / 2 (streamed ring) and kv_mask_form 0 / 1 (bit word) / 2 / 3 (additive "
                f"io-dtype / fp32 vector); got {bias_mode} / {kv_mask_form}"
            )
    stages_kv = _stages_kv_d128(params.dtype_qkv, cta_mma, mxfp8=mxfp8, tile_k=tile_k)
    # The (n, n+1) pair item (TemplateParams.n_chunk, appended; read through getattr so a record without the field renders
    # the one-batch-entry item): a 128-row block, cga1, the bias arm only -- _validate_bias pins the couplings.
    n_chunk = int(getattr(params, "n_chunk", 0))
    pair = n_chunk > 0
    if pair and bias_mode == BIAS_MODE_NONE:
        raise ValueError(f"{flavor}: n_chunk={n_chunk} is the (n, n+1) pair item of the BIAS arm; it needs bias_mode=1 / 2 (got {bias_mode})")
    bias_stages_req = int(getattr(params, "bias_stages", 0))
    if bias_stages_req and bias_mode != BIAS_MODE_STREAMED:
        raise ValueError(f"{flavor}: bias_stages={bias_stages_req} is the STREAMED ring's depth; it needs bias_mode=2 (got {bias_mode})")
    # The Q tile height (TILE_M): 128 on every flavor but the f16/bf16 small tiles, where the private seam may pin 64 (the TILE_M = 64
    # lever; ``_tile_m_d128`` is the ONE helper the budgets below read too, so the Q / O slabs, the block rows and the budgets agree).
    tile_m = _tile_m_d128(tile_k) if (tile_k <= 64 and params.dtype_qkv > _DTYPE_E5M2 and not mxfp8) else 128
    bias_tiles = 0
    stages_bias = 0
    if bias_mode == BIAS_MODE_RESIDENT:
        bias_tiles = sm107_bias_resident_tiles(tile_k, cta_mma, dtype_qkv=params.dtype_qkv, tile_n=tile_n, tile_m=tile_m, pair=pair)
        if bias_tiles == 0:
            raise ValueError(
                f"{flavor}: no resident bias block fits beside the Q / O slabs and the {stages_kv}-deep K / V ring at TILE_K={tile_k}, cta_mma={cta_mma} "
                f"({SMEM_USABLE_BYTES // 1024} KiB usable); the bias is served on the widths of SM107_BIAS_D_SHAPES={sorted(SM107_BIAS_D_SHAPES)}"
            )
    elif bias_mode == BIAS_MODE_STREAMED:
        stages_bias = sm107_bias_stages_pick(tile_k, cta_mma, dtype_qkv=params.dtype_qkv, tile_n=tile_n, pair=pair, request=bias_stages_req)
        if stages_bias == 0:
            raise ValueError(
                f"{flavor}: no {sorted(BIAS_RING_STAGES)}-stage streamed bias ring of {sm107_bias_block_rows(2, tile_m, pair=pair)} x {tile_n} tiles "
                f"{'(requested bias_stages=' + str(bias_stages_req) + ') ' if bias_stages_req else ''}fits beside the Q / O slabs and the {stages_kv}-deep "
                f"K / V ring at TILE_K={tile_k}, cta_mma={cta_mma} ({SMEM_USABLE_BYTES // 1024} KiB usable)"
            )
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    # The padded tail form (TemplateParams.padded_tail_form, appended; read through getattr so a record without the field renders the
    # legacy tail body): the pre-masked tail serves the f16/bf16 small tiles' PLAIN padded body only -- a band has masked segments on
    # both sides, the bias / kv_mask arms run the -inf body (one body already), the d128 flavor keeps its own rendering.
    padded_tail_form = int(getattr(params, "padded_tail_form", 0))
    if padded_tail_form not in PADDED_TAIL_FORMS:
        raise ValueError(f"{flavor}: padded_tail_form must be 0 (the legacy tail body) or 1 (the pre-masked tail); got {padded_tail_form}")
    if padded_tail_form == 1 and (mask_flags != MASK_PADDED or bias_mode or kv_mask_form or tile_k > 64 or mxfp8 or params.dtype_qkv <= _DTYPE_E5M2):
        raise ValueError(
            f"{flavor}: padded_tail_form=1 (the pre-masked tail) is wired on the f16/bf16 32- / 64-wide tiles' PLAIN padded body only "
            f"(MASK_FLAGS == MASK_PADDED alone, no bias / kv_mask); got MASK_FLAGS={mask_flags}, bias_mode={bias_mode}, kv_mask_form={kv_mask_form}, "
            f"TILE_K={tile_k}, dtype_qkv={params.dtype_qkv}"
        )
    # The dead-sub-tile quiet body (TemplateParams.dead_subtile_quiet, appended; read through getattr so a record without the field
    # renders the full body): the one-batch-entry item of the f16/bf16 small tiles without a bias -- the bias arms also wait and
    # release the bias barriers per item, the pair item's dead sub-tile is the odd chunk tail, the d128 flavor keeps its rendering.
    dead_subtile_quiet = int(getattr(params, "dead_subtile_quiet", 0))
    if dead_subtile_quiet not in DEAD_SUBTILE_QUIET_FORMS:
        raise ValueError(f"{flavor}: dead_subtile_quiet must be 0 (the full body on a dead sub-tile) or 1 (the quiet body); got {dead_subtile_quiet}")
    if dead_subtile_quiet == 1 and (
        bias_mode or pair or tile_k > 64 or mxfp8 or params.dtype_qkv <= _DTYPE_E5M2 or params.pack_gqa or params.thd_varlen or params.epilogue_gate
    ):
        raise ValueError(
            f"{flavor}: dead_subtile_quiet=1 (the dead-sub-tile quiet body) is wired on the f16/bf16 32- / 64-wide tiles' one-batch-entry item "
            f"without a bias, PackGQA, THD or the gated epilogue only; got bias_mode={bias_mode}, n_chunk={n_chunk}, TILE_K={tile_k}, "
            f"dtype_qkv={params.dtype_qkv}, pack_gqa={params.pack_gqa}, thd_varlen={params.thd_varlen}, epilogue_gate={params.epilogue_gate}"
        )
    cls = {192: CfgD192, 128: CfgD128, 64: CfgD64, 32: CfgD32}[tile_k]
    if tile_k <= 64:
        # The small tiles are the f16/bf16 kernel's: the quantized d128 files keep TILE_K = 128 (their k_dim / SF geometry
        # is tied to the 128-wide tile), and the body is validated at the widths SM107_SMALL_D_CGAS names.
        if params.dtype_qkv <= _DTYPE_E5M2:
            raise ValueError(f"{flavor}: the 32- and 64-wide head-dim tiles are wired on the f16/bf16 prefill kernel only (dtype_qkv={params.dtype_qkv})")
        if cta_mma not in SM107_SMALL_D_CGAS[(tile_k, tile_o)]:
            raise ValueError(
                f"{flavor}: cta_mma={cta_mma} is outside the validated widths {sorted(SM107_SMALL_D_CGAS[(tile_k, tile_o)])} for TILE_O={tile_o}; "
                f"at cga2 the per-CTA V operand is {tile_o // 2} columns = a {tile_o // 2 * b}-byte MN-major swizzle atom"
            )

    cfg = cls(
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        SPLIT_PIPELINE=0,
        QO_ALIAS=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o, o_div),
        O_BLOCK_SCALE=_O_BLOCK_SCALE_BY_DTYPE.get(dtype_o, 0),
        O_PACK_DIV=o_div,
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=2,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        SOFTMAX_WARPGROUPS=2,
        CORRECTION_WARPS=4,
        SOFTMAX_REGS=192,
        CORRECTION_REGS=88,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        PADDED_TAIL_FORM=padded_tail_form,
        DEAD_SUBTILE_QUIET=dead_subtile_quiet,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        N_BMM2_CHUNKS=tile_n // 64,
        BMM2_CHUNK_SIZE=64,
        TOTAL_WARPS=_D128_TOTAL_WARPS,
        THREADS_PER_CTA=_D128_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=4,
        CORR_WARP_BASE=8,
        MMA_WARP_ID=12,
        TMALDG_WARP_ID=13,
        TMASTG_WARP_ID=14,
        SCHED_WARP_ID=15,
        READ_TILE_ARRIVERS=_D128_READ_TILE_ARRIVERS,
        BIAS_MODE=bias_mode,
        BIAS_RESIDENT_TILES=bias_tiles,
        BIAS_BLOCK_ROWS=sm107_bias_block_rows(2, tile_m, pair=pair) if bias_mode else 0,
        BIAS_SWZ_BYTES=BIAS_SWZ_BYTES,
        BIAS_BOXES_PER_TILE=(tile_n * b) // BIAS_SWZ_BYTES if bias_mode else 0,
        BIAS_EMPTY_ARRIVERS=2 * 4 if (bias_mode or kv_mask_form) else 0,
        KV_MASK_FORM=kv_mask_form,
        N_CHUNK=n_chunk,
        STAGES_BIAS=stages_bias,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D128_TOTAL_WARPS, expect_arrivers=_D128_READ_TILE_ARRIVERS)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_head_dim_tile(cfg, flavor)
    _validate_tile_n(cfg, flavor)
    _validate_masks(cfg, flavor)
    _validate_bias(cfg, flavor)
    _validate_smem(cfg, flavor, qo_alias=bool(cfg.QO_ALIAS), bias_bytes=bias_resident_block_bytes(cfg))
    return cfg, _tma_iters(cfg)


def make_cfg_d32(params: TemplateParams) -> Tuple[CfgD32, TmaIters]:
    """The 32-wide head-dim tile of ``sm107/prefill_d128_f16.py`` (``TemplateParams.d_flavor=32``): every f16/bf16 head dim
    <= 32 with D % 8 == 0 (8 / 16 / 24 / 32) by TMA zero fill; cga1 (SM107_SMALL_D_CGAS); dense / causal / bottom-right /
    SWA / padded / sink / GQA like d128.  THD, split-KV, paged KV, PackGQA and the score side outputs are NOT wired on the
    tile: the row routes a graph asking for them onto the d128 flavor (``engines.Capabilities.dense_only_d_shapes``), and this
    factory refuses the THD record for the flavor."""
    return _make_cfg_d128_family(params, flavor="sm107 d32", tile_k=32, tile_o=32, mxfp8=False)


def make_cfg_d64(params: TemplateParams) -> Tuple[CfgD64, TmaIters]:
    """The 64-wide head-dim tile of ``sm107/prefill_d128_f16.py`` (``TemplateParams.d_flavor=64``): every f16/bf16 head dim in
    (32, 64] with D % 8 == 0 (40 / 48 / 56 / 64) by TMA zero fill; the same feature set and routing as ``make_cfg_d32`` (cga1
    and cga2)."""
    return _make_cfg_d128_family(params, flavor="sm107 d64", tile_k=64, tile_o=64, mxfp8=False)


def make_cfg_d128(params: TemplateParams) -> Tuple[CfgD128, TmaIters]:
    return _make_cfg_d128_family(params, flavor="sm107 d128", tile_k=128, tile_o=128, mxfp8=False)


def make_cfg_d128_mxfp8(params: TemplateParams) -> Tuple[CfgD128, TmaIters]:
    return _make_cfg_d128_family(params, flavor="sm107 d128 mxfp8", tile_k=128, tile_o=128, mxfp8=True)


def make_cfg_d192(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    """d_qk=192 / d_v=128, f16 and per-tensor FP8.

    Same family as ``make_cfg_d128`` with a wider K: the pre-upstream base kernel
    was flavor-generic (one body served d128 and d192xd128 by swapping the config),
    and ``CfgD192`` only widens ``TILE_K``.
    """
    return _make_cfg_d128_family(params, flavor="sm107 d192xd128", tile_k=192, tile_o=128, mxfp8=False)


def make_cfg_d192_mxfp8(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    """d_qk=192 / d_v=128, block-scale MXFP8.

    CGA2 ONLY, and that is a DESCRIPTOR constraint rather than a tuning choice.
    At ``cta_mma=1`` the K/V rings are not halved, so the four scale-factor tiles
    -- allocated last -- start at 256/258/276/278 KiB, i.e. past the **256 KiB
    version-0 tcgen05 descriptor window**.  A version-0 SF descriptor there wraps
    to offset 0 and the UTCCP copies Q DATA bytes into the SF TMEM columns:
    ``LSE = +inf`` and ``O = NaN`` on 100 % of cells, at every shape (the exact
    d512 MXFP8 failure in rules/mma-tma-matrix.md S6).  At ``cta_mma=2`` the
    highest slab sits at 200 KiB and version 0 is provably safe.

    So the Rubin MXFP8 engine row deliberately declares NO ``cgas_by_d_shape``
    entry for (192, 128), leaving it on the row default ``cgas={2}``.  Lifting
    that needs ``DESC_VERSION`` derived from the layout AND the version-1 SF path
    validated on Rubin -- which is NOT a free widening: setting
    ``desc_version=1`` on the d128/d256 MXFP8 tiles turned 21 green tests red
    (2026-09-08), so the bit is not a transparent superset.
    """
    return _make_cfg_d128_family(params, flavor="sm107 d192xd128 mxfp8", tile_k=192, tile_o=128, mxfp8=True)


# ---------------------------------------------------------------------------
# d256 -- Q-union-O alias, ONE softmax warpgroup (12 warps)
# ---------------------------------------------------------------------------
#
# Warp map: 4 softmax + 4 correction + MMA + TMALDG + TMASTG + scheduler.
# The arriver count is per-CTA-group: ((SOFTMAX_WG*4) + CORR + 2) * CGA_SIZE
# + (CGA_M / CTA_MMA).

_D256_TOTAL_WARPS = 12


def _d256_read_tile_arrivers(cta_mma: int) -> int:
    return ((1 * 4) + 4 + 2) * (cta_mma * 1) + (cta_mma // cta_mma)


def _make_cfg_d256_family(params: TemplateParams, *, flavor: str, mxfp8: bool):
    _validate_params(flavor, params)
    if params.softmax_scale_prefolded:
        raise ValueError(f"{flavor}: softmax_scale_prefolded is wired on the d128 MXFP8 kernel only")
    cta_mma = params.cta_mma
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    tile_k = tile_o = 256
    tile_n = 128
    # STAGES_KV is a TUNING KNOB again (2..4), and the note that said otherwise
    # was WRONG about why.
    #
    # It used to be pinned to 2, blaming a body that "conflates the KV ring
    # index with the 2-slot S_acc parity". That diagnosis does not survive
    # reading the body: every parity site derives parity from the ABSOLUTE
    # `kv_loop & 1`, and K/V ride separate PipelineStates advanced in lockstep
    # with the TMA warp, so the body is depth-agnostic.
    #
    # The real cause was the >256 KiB tcgen05 descriptor wrap. Buffers are laid
    # out sQO | sK[stages] | sV[stages]; at depth 4 the last two V stages start
    # at 256 KiB and 288 KiB, and a VERSION-0 descriptor truncates
    # `start_address` to 14 bits, so those stages alias the bottom of SMEM. The
    # answer goes wrong from the KV iteration that first touches a wrapped
    # stage -- which is why it looked like a ring bug and why it only shows up
    # at S_kv > 256.
    #
    # Measured on Rubin, d256 f16, cos vs an fp32 reference, n_kv = 2/3/4/8:
    #   depth 4, desc v0:  1.0000 / 0.6806 / 0.4980 / 0.4640   <- the old data
    #   depth 4, desc v1:  1.0000 / 1.0000 / 1.0000 / 1.0000   dense AND causal
    #   depth 3, desc v0:  1.0000 / 1.0000 / 1.0000 / 1.0000   (stays under)
    # `prefill_d256_f16.DESC_VERSION` is now DERIVED from the layout, so any
    # depth that fits SMEM is correct by construction. Do not re-literal it.
    # (The per-tensor FP8 sibling `prefill_d256_fp8.py` carried a literal 0
    # until the epilogue-gate PR ported the same `_needs_desc_v1` derivation;
    # at e4m3-O depth 4 its last V stage starts at exactly 262144, so the
    # literal was one depth away from the same wrap.)
    # `is not None`, NOT a truth test: a truthiness check maps an explicit
    # stages_kv=0 onto the default 2, which is a knob SUBSTITUTION -- the one
    # thing the engine contract forbids (honored or ineligible). Out-of-domain
    # values must reach the 2..4 check below and raise there.
    stages_kv = params.stages_kv if getattr(params, "stages_kv", None) is not None else 2
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    arrivers = _d256_read_tile_arrivers(cta_mma)
    # The four single-warp roles (MMA / TMA-LDG / TMA-STG / scheduler; the HW
    # equality constraint keeps them one number) run at 40 on every d256
    # module.  MEASURED DEAD END (2026-09-15, gated MXFP8): its sm_107a SASS
    # carries 25 STL/LDL against the ungated 6 (the SR_CTAID.{x,y,z} tile
    # coordinates, spilled at kernel entry and refilled in the TMA-LDG /
    # TMA-STG / scheduler regions), and raising these four roles to 48 left the
    # count at 25.  The refill sites carry MOV.SPILL UR -> R (5 vs 0 ungated):
    # the pressure starts in the UNIFORM register file, which setmaxnreg does
    # not govern, so this vector budget is not the lever.  The split itself IS
    # applied (USETMAXREG.TRY_ALLOC / DEALLOC present in the sm_107a SASS).  Do
    # not retry the bump; see the REGISTER NOTE in
    # sm107/prefill_d256_mxfp8.py's TMA-LDG warp for the levers not yet tried.
    other_regs = 40

    cfg = CfgD256(
        TILE_M=128,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        SPLIT_PIPELINE=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        # The body is a SINGLE-warpgroup pipeline. Two warpgroups is the
        # Blackwell d256 split-P design and does not exist here.
        SOFTMAX_WARPGROUPS=1,
        CORRECTION_WARPS=4,
        SOFTMAX_REGS=240,
        CORRECTION_REGS=96,
        MMA_REGS=other_regs,
        TMALDG_REGS=other_regs,
        TMASTG_REGS=other_regs,
        SCHEDULER_REGS=other_regs,
        OTHER_REGS=other_regs,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        N_BMM2_CHUNKS=tile_n // 64,
        BMM2_CHUNK_SIZE=64,
        TOTAL_WARPS=_D256_TOTAL_WARPS,
        THREADS_PER_CTA=_D256_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=0,  # unused at SOFTMAX_WARPGROUPS=1
        CORR_WARP_BASE=4,
        MMA_WARP_ID=8,
        TMALDG_WARP_ID=9,
        TMASTG_WARP_ID=10,
        SCHED_WARP_ID=11,
        READ_TILE_ARRIVERS=arrivers,
        # Fused epilogue gate (validated per flavor in _validate_params). The gate
        # is 2 B/elem on BOTH the half kernel (Q dtype) and the FP8 kernel (bf16).
        EPILOGUE_GATE=int(params.epilogue_gate),
        GATE_BPE=2,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D256_TOTAL_WARPS, expect_arrivers=arrivers)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_masks(cfg, flavor)
    # The STAGES_KV upper bound differs per dtype family, and the message says
    # which: the f16 and per-tensor FP8 bodies DERIVE their tcgen05 descriptor
    # version from the layout (`_needs_desc_v1`), so for them every depth that
    # fits SMEM is correct; the MXFP8 body PINS DESC_VERSION=0 (version 1
    # regressed 21 green d128/d256 MXFP8 tests, rules/mma-tma-matrix.md S6), so
    # for it the bound is the 256 KiB version-0 window on the LAST scale-factor
    # slab -- checked below with its own diagnostic.
    _depth_bound = (
        f"the {SMEM_USABLE_BYTES // 1024} KiB Rubin carveout, and the kernel derives its tcgen05 descriptor version from the resulting "
        f"layout so every depth in range is correct"
        if not mxfp8
        else f"the {SMEM_USABLE_BYTES // 1024} KiB Rubin carveout AND the {TCGEN05_V0_ADDR_LIMIT // 1024} KiB version-0 tcgen05 descriptor "
        f"window on the last scale-factor slab (this kernel pins DESC_VERSION=0; see d256_mxfp8_last_sf_tile_start)"
    )
    _check(
        [
            (cfg.TILES_Q == 1, f"{flavor}: the d256 pipeline mandates TILES_Q == 1"),
            (cfg.SOFTMAX_WARPGROUPS == 1, f"{flavor}: the d256 pipeline mandates SOFTMAX_WARPGROUPS == 1"),
            (
                2 <= cfg.STAGES_KV <= 4,
                f"{flavor}: STAGES_KV must be in 2..4 (got {cfg.STAGES_KV}); the upper bound is {_depth_bound}",
            ),
        ]
    )
    sf_bytes = 0
    if mxfp8:
        # MXFP8: the four scale-factor slabs are declared after sQO | sK | sV
        # and read through VERSION-0 UTCCP descriptors, so their START must sit
        # under the 14-bit window.  A bf16 O at STAGES_KV=3 (or an e4m3 O at 4)
        # puts sQ_SF at or past 262144: the descriptor wraps to offset 0 and the
        # UTCCP copies Q DATA bytes into the SF TMEM columns -- LSE = +inf and
        # O = NaN on 100 % of cells at every shape, the SF values provably
        # irrelevant (the d512 MXFP8 failure).  Latent until 2026-09-15 (depth 3
        # fit the SMEM check); raised HERE, before the kernel's own backstop.
        # The epilogue gate tile is declared AFTER the SF slabs (PR-B D7) and is
        # not in this sum.
        last_sf = d256_mxfp8_last_sf_tile_start(cfg)
        _check(
            [
                (
                    # The gated MXFP8 module was traced and launched at CTA_MMA=1 ONLY
                    # (the row pins cga1 for the quantized d256 flavors and so does the
                    # adapter, supported_cgas_for((256, 256), fp8=True) == (1,)).  A
                    # cga2 gated MXFP8 module (K/V rings halved, SF slabs full-size,
                    # gate per CTA) fits the tallies below but was never validated;
                    # a future widening of the row surfaces HERE, as a config raise,
                    # not as an inherited unvalidated arm.
                    not (cfg.EPILOGUE_GATE and cfg.CTA_MMA == 2),
                    f"{flavor}: the epilogue gate is validated at CTA_MMA=1 only (got CTA_MMA={cfg.CTA_MMA}); a cga2 gated "
                    f"MXFP8 module has never been traced or launched -- validate on Rubin before widening the row (engine-contract rule 9)",
                ),
                (
                    last_sf < TCGEN05_V0_ADDR_LIMIT,
                    f"{flavor}: the last scale-factor slab starts at {last_sf} B, at or past the {TCGEN05_V0_ADDR_LIMIT} B version-0 tcgen05 "
                    f"descriptor window (STAGES_KV={cfg.STAGES_KV}, BPE_O={cfg.BPE_O}, CTA_MMA={cfg.CTA_MMA}); a version-0 SF descriptor there "
                    f"wraps to offset 0 and the UTCCP copies Q data as scale factors (LSE = +inf, O = NaN on every cell).  Reduce STAGES_KV or "
                    f"write an e4m3 O -- the kernel pins DESC_VERSION=0 (version 1 is not a transparent widening, rules/mma-tma-matrix.md S6)",
                ),
            ]
        )
        sf_bytes = d256_mxfp8_sf_smem_bytes(cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O, cfg.STAGES_KV)
    # d256 always Q-union-O aliases.  With the epilogue gate on, the 64 KiB
    # staging tile is added here: depth 2 fits at both CTA_MMA widths (258 KiB
    # f16 cga2 / 226 KiB fp8 e4m3-O / 258 KiB fp8 bf16-O; MXFP8 adds its 6 KiB of
    # scale-factor slabs: 232 KiB e4m3-O / 264 KiB bf16-O), depth 3 does not
    # (322 KiB) -- the raise names every term so the reader sees why a depth
    # that fits ungated no longer does.
    _validate_smem(cfg, flavor, qo_alias=True, sf_bytes=sf_bytes)
    return cfg, _tma_iters(cfg)


def make_cfg_d256(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    return _make_cfg_d256_family(params, flavor="sm107 d256", mxfp8=False)


def make_cfg_d256_mxfp8(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    return _make_cfg_d256_family(params, flavor="sm107 d256 mxfp8", mxfp8=True)


# ---------------------------------------------------------------------------
# d512 -- cga4x1 role split (8 warps)
# ---------------------------------------------------------------------------
#
# Warp map: 4 compute (softmax on sg0 / correction on sg1) + MMA + TMALDG +
# TMASTG + scheduler.  READ_TILE_ARRIVERS counts across the whole cluster:
#   compute warps (4/CTA, both sgs)          = 4 * CGA_M * CGA_N          = 16
#   TMALDG        (1/CTA, both sgs)          = 1 * CGA_M * CGA_N          =  4
#   sg0 MMA leader (lead CTA only)           = (CGA_M*CGA_N)/(2*CTA_MMA)  =  1
#   sg1 MMA all   (P12 forwarder + leader)   = (CGA_M*CGA_N)/2            =  2
#   sg1 TMASTG    (sg1 CTAs only)            = CTA_MMA * CGA_N            =  2
#                                                                   total = 25

_D512_TOTAL_WARPS = 8
_D512_CGA_M, _D512_CGA_N, _D512_CTA_MMA = 4, 1, 2
_D512_READ_TILE_ARRIVERS = (
    ((1 * 4) + 1) * (_D512_CGA_M * _D512_CGA_N)
    + (_D512_CGA_M * _D512_CGA_N) // (2 * _D512_CTA_MMA)
    + (_D512_CGA_M * _D512_CGA_N) // 2
    + _D512_CTA_MMA * _D512_CGA_N
)


def _make_cfg_d512_family(params: TemplateParams, *, flavor: str, mxfp8: bool):
    _validate_params(flavor, params)
    if params.softmax_scale_prefolded:
        raise ValueError(f"{flavor}: softmax_scale_prefolded is wired on the d128 MXFP8 kernel only")
    if params.cta_mma != _D512_CTA_MMA:
        raise ValueError(f"{flavor}: the role-split pipeline is cga4x1 / CTA_MMA=2 only (got cta_mma={params.cta_mma})")
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    tile_k = tile_o = 512
    tile_n = 128
    is_fp8 = params.dtype_qkv <= _DTYPE_E5M2
    # FP8 (BPE=1) fits a deeper K/V ring and a 4-deep cross-sg P ring; f16/bf16
    # (BPE=2) are SMEM-cap driven down to 2/2.
    stages_kv = 3 if is_fp8 else 2
    xfer_stages = 4 if is_fp8 else 2
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)

    cfg = CfgD512(
        TILE_M=128,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=_D512_CGA_M,
        CGA_N=_D512_CGA_N,
        CTA_MMA=_D512_CTA_MMA,
        SPLIT_PIPELINE=1,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, _D512_CTA_MMA, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        XFER_STAGES=xfer_stages,
        # One 4-warp compute group; correction is fused into it, sub-group
        # conditional, so there are no separate correction warps or registers.
        SOFTMAX_WARPGROUPS=1,
        CORRECTION_WARPS=0,
        SOFTMAX_REGS=240,
        CORRECTION_REGS=0,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        BMM2_N_PER_CALL=256,
        BMM2_LOOP_N_BLOCKS=tile_o // 256,
        N_BMM2_CHUNKS=2,
        BMM2_CHUNK_SIZE=256,
        TOTAL_WARPS=_D512_TOTAL_WARPS,
        THREADS_PER_CTA=_D512_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=0,  # unused under the role split
        CORR_WARP_BASE=0,  # correction fused with the compute group
        MMA_WARP_ID=4,
        TMALDG_WARP_ID=5,
        TMASTG_WARP_ID=6,
        SCHED_WARP_ID=7,
        READ_TILE_ARRIVERS=_D512_READ_TILE_ARRIVERS,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D512_TOTAL_WARPS, expect_arrivers=_D512_READ_TILE_ARRIVERS)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_masks(cfg, flavor)
    _check([(cfg.SPLIT_PIPELINE == 1, f"{flavor}: the d512 flavor is role-split (SPLIT_PIPELINE=1)")])
    return cfg, _tma_iters(cfg)


def make_cfg_d512(params: TemplateParams):
    # The 2x2-datapath record (TemplateParams.mma_2x2, appended, default False) is read through getattr so a
    # record built before the field existed takes the role-split arm unchanged (the config_sm100 dispatch, mirrored).
    if getattr(params, "mma_2x2", False):
        return make_cfg_d512_2x2(params)
    return _make_cfg_d512_family(params, flavor="sm107 d512", mxfp8=False)


def make_cfg_d512_mxfp8(params: TemplateParams) -> Tuple[CfgD512, TmaIters]:
    return _make_cfg_d512_family(params, flavor="sm107 d512 mxfp8", mxfp8=True)


# ---------------------------------------------------------------------------
# d512 flavor on the 2x2 DATAPATH -- d_qk = d_v = 512, Rubin, sm107/prefill_d512_f16_2x2.py
#
# The SM100 CfgD512X2 record (config_sm100: one pipeline per CTA on the
# cta_group::2 M=128 atom, 64 Q rows per CTA, 12 warps, a (CGA_M, 1, 1)
# cluster of twin pairs sharing K/V by TMA multicast) with the Rubin deltas of
# the design (section 13):
#   * 3-deep K and V sub-chunk rings (STAGES_K_SUB = STAGES_V_SUB = 3: one
#     extra KV iteration of TMA allowance; the O staging keeps aliasing the V
#     ring).  K 4 / V 3 would put 320 KiB of data over the usable line.
#   * DESC_VERSION derived from the LAYOUT: sQ 64 KiB | sK 3 x 32 | sV 3 x 32 u
#     sO puts the P ring at exactly 262144 B = TCGEN05_V0_ADDR_LIMIT, so every
#     SmemTile of the kernel carries desc_version=1 (the 2026-09-04 silent-zero
#     class; test_sm107_descriptor_version_matches_the_smem_budget).
#   * The budget is checked against SMEM_USABLE_BYTES (320 KiB), never the
#     327 KiB capacity (the d192 f16 50 %-zeros lesson above).
#   * TMEM: 512 columns (388 used); the 576-col is_exclusive allocation is NOT
#     needed until a 3rd / 4th S parity is added.
#   * TILE_K_HW from tile_k_hw(dtype) = 16 for f16/bf16 (the K=64 2-chunk form
#     is the FP8 QMMA path).
# ---------------------------------------------------------------------------

_D512_2X2_STAGES_K_SUB = 3
_D512_2X2_STAGES_V_SUB = 3


def _d512_2x2_desc_version_for(cfg: CfgD512X2) -> int:
    """The tcgen05 SMEM-descriptor version the 2x2 layout NEEDS: 1 as soon as the last MMA-operand tile (the P ring,
    the highest slab) reaches the 256 KiB window of a version-0 descriptor, else 0."""
    p_end = d512_2x2_p_ring_start_bytes(cfg) + cfg.XFER_STAGES * cfg.TILE_M * cfg.TILE_N * cfg.BPE
    return 1 if p_end > TCGEN05_V0_ADDR_LIMIT else 0


def _validate_cfg_d512_2x2_sm107(cfg: CfgD512X2, flavor: str) -> None:
    """The arch-neutral 2x2 geometry checks plus the Rubin arch facts."""
    smem = d512_2x2_smem_bytes(cfg)
    want_desc = _d512_2x2_desc_version_for(cfg)
    _check(
        d512_2x2_geometry_checks(cfg)
        + (
            (
                cfg.STAGES_K_SUB == _D512_2X2_STAGES_K_SUB and cfg.STAGES_V_SUB == _D512_2X2_STAGES_V_SUB,
                f"{flavor}: the Rubin arm runs 3-deep K/V sub-chunk rings (got {cfg.STAGES_K_SUB}/{cfg.STAGES_V_SUB})",
            ),
            (
                cfg.DESC_VERSION == want_desc,
                f"{flavor}: DESC_VERSION={cfg.DESC_VERSION} but the layout puts the P ring at {d512_2x2_p_ring_start_bytes(cfg)} B "
                f"(version-0 window ends at {TCGEN05_V0_ADDR_LIMIT}); it needs version {want_desc}",
            ),
            (
                cfg.SMEM_CAP_BYTES == SMEM_USABLE_BYTES,
                f"{flavor}: the SMEM budget is checked against the {SMEM_USABLE_BYTES // 1024} KiB usable carveout, got {cfg.SMEM_CAP_BYTES}",
            ),
            (
                smem["total"] <= SMEM_USABLE_BYTES,
                f"{flavor}: SMEM {smem['total']} B (incl. {cfg.SMEM_ALIGN_PAD} B pad) exceeds the {SMEM_USABLE_BYTES // 1024} KiB usable Rubin "
                f"carveout: {smem}.  Overflowing it does NOT fail the launch -- it clobbers the last buffer allocated",
            ),
            (cfg.TMEM_COLS == 512, f"{flavor}: 512 TMEM columns (388 used); 576 + is_exclusive only with a 3rd/4th S parity"),
            (
                cfg.O_EMPTY_ARRIVERS == cfg.ONE_WARP * cfg.KV_SHARE,
                f"{flavor}: the O u V alias gate is PAIR-WIDE -- mb_o_empty init must be ONE_WARP x KV_SHARE = {cfg.ONE_WARP * cfg.KV_SHARE} "
                f"(got {cfg.O_EMPTY_ARRIVERS}); a twin's multicast V(t+1) lands in BOTH twins' sVO",
            ),
        )
    )
    _validate_dtype_k_step(cfg, flavor)


def make_cfg_d512_2x2(params: TemplateParams, *, cga_m: int = 4) -> Tuple[CfgD512X2, TmaIters]:
    """The Rubin 2x2-datapath d512 configuration (``TemplateParams.mma_2x2``).

    ``cga_m`` = 4 (two twin pairs sharing K/V by multicast, the shipped arm) or 2 (the bring-up / bitwise-twin arm:
    one pair, own-bit loads).  Not a knob: tests load the template with the arm they want through the module constant
    the kernel file reads (FROST_D512_2X2_CGA_M)."""
    flavor = "sm107 d512 2x2"
    if not getattr(params, "mma_2x2", False):
        raise ValueError(f"{flavor}: make_cfg_d512_2x2 needs a TemplateParams record with mma_2x2=True (make_cfg_d512 dispatches on it)")
    # The Rubin record guard keys its THD allowlist by FLAVOR NAME (_F16_THD_FLAVORS holds "sm107 d512"): the 2x2 sibling
    # carries the same ported setup-kernel call sites as the role-split d512 body, so it is validated under that name.
    _validate_params("sm107 d512", params)
    if params.softmax_scale_prefolded:
        raise ValueError("d512 2x2: softmax_scale_prefolded is wired on the d128 MXFP8 kernel only")
    # The 2x2 record's own domain (d512 only, half inputs, unpaged, whole-group PackGQA) is stated ONCE, in config_sm100.
    _validate_params_sm100("d512", params)
    if params.dtype_qkv not in (_DTYPE_BF16, _DTYPE_FP16):
        raise ValueError(f"{flavor}: BF16/FP16 inputs only")
    if params.cta_mma != 2:
        raise ValueError(f"{flavor}: cta_group::2 pairs only (got cta_mma={params.cta_mma})")
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    n_pairs = cga_m // CfgD512X2.CTA_MMA
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    base = CfgD512X2(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cga_m,
        KV_SHARE=n_pairs,
        Q_SUPERS_PER_CLUSTER=cga_m,
        ROWS_PER_CLUSTER=cga_m * CfgD512X2.TILE_M,
        READ_TILE_ARRIVERS=10 * cga_m + n_pairs,
        KV_EMPTY_ARRIVERS=n_pairs,
        # PAIR-WIDE O u V alias gate (fix-lane FATAL-1): every twin's TMA-STG warp arrives on both twins' mb_o_empty.
        O_EMPTY_ARRIVERS=CfgD512X2.ONE_WARP * n_pairs,
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        STAGES_K_SUB=_D512_2X2_STAGES_K_SUB,
        STAGES_V_SUB=_D512_2X2_STAGES_V_SUB,
        STAGES_KV=max(_D512_2X2_STAGES_K_SUB, _D512_2X2_STAGES_V_SUB),
        SMEM_CAP_BYTES=SMEM_USABLE_BYTES,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv or 1),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=_pack_g(params, CfgD512X2.TILE_M, partial=False),
    )
    # DESC_VERSION is DERIVED from the layout the fields above describe, never set by hand.
    cfg = dataclasses_replace(base, DESC_VERSION=_d512_2x2_desc_version_for(base))
    _validate_cfg_d512_2x2_sm107(cfg, flavor)
    return cfg, _tma_iters(cfg)
