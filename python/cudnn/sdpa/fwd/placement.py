# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the SDPA-forward family's proposals stand against the cuDNN backend, per measured shard.

:func:`place` answers one question for a graph the row admits: do FROST's proposals go ahead of
the backend's own block (``LEAD``) or behind it (``TRAIL``)? ``engines/heuristics._assemble`` then
splices the backend block where the family's :data:`~cudnn.engines.heuristics.BACKEND` marker sits.
Participation is not decided here -- an engine that admits the graph is always in the list, and
``build_plans`` walks past a declined entry -- only the default winner is.

Thresholds are tunable performance policy. The dense SM100 and SM90 shards below were re-fitted
2026-10-05 on a full B200 (148 SMs, 1000 W) and H100 SXM (132 SMs, 700 W): public cuDNN 9.27.0.42,
CuTeDSL 4.8.0, CUDA 13.4, BF16 BSHD, CUDA-graph replay of the backend's heuristic pick vs the
first FROST plan, warm and 512 MiB-flushed, ~1,100 cases plus a 140-case off-grid hold-out (MR
description has the tables). GPU time only, no host enqueue. Re-evaluate offline when kernels or
backend versions change; unit tests exercise placement-marker contracts with synthetic verdicts
rather than pinning workload winners.

SM100 f16/bf16 row (B200; SM103 runs the same thresholds, not re-measured there):

- decode-shaped, ``2 <= s_q <= 16``, dense or paged: 0.02-0.65 on every cell (llama d128, qwen35
  d256, gpt_oss d64, deepseek_v4 d512; q = 2, 3, 4, 8, 16; kv 2k-128k; b 1-128). The backend has
  no decode-class engine for ``s_q > 1`` and serves these rows with a prefill-class one, so
  FROST leads at every cache length. Short caches (B200, public cuDNN 9.26.0.51, this tree at
  a3a52aa88, bf16 paged KV, page 16, declared max KV 1024, CUDA-graph replay, kernel time,
  2026-10-06; 500 / 1000 live tokens): 64/4 d128 with two or four rows runs 17-25 us on the decode
  tile against 25-42 us (b = 1), 68-122 us (b = 8) and 223-412 us (b = 32) on the backend; 64/8
  d128 at b = 32 29-45 us against the same backend times; 32/2 d256 21-32 us against 43-78 us
  (b = 1), 105-195 us (b = 8) and 350-716 us (b = 32). The backend's time grows with b x KV
  like a prefill kernel; the tile's with the live tokens of one request.
- ``s_q == 1``, d512 (exact or the d320-d448 envelope): FROST wins once enough KV tokens are in
  flight, ``b * h_kv * s_kv >= 16k`` with >= 32 query heads (0.57-0.94 at the bound, 1.02-1.04 below)
  or ``>= 32k`` with fewer (0.56-0.92; 1.09-1.13 below); a single KV unit needs twice that. Both
  this and the d256 rule rely on split-KV: an ``s_kv`` off the KV tile cannot split, and unsplit
  small launches lose (b = 3, h_kv = 1, kv 12000: 1.98x d512; 12 units: 2.03x d256), so those need
  16 units (d512) or 32 (d256).
- ``s_q == 1``, d256: ``units = b * h_kv``; ``units >= 32`` wins 0.52-0.95, ``4 <= units < 32`` wins
  once ``units * s_kv >= 2**16`` (0.74-0.97), fewer units lose up to 128k (1.07-2.9).
- ``s_q == 1``, d64 / d128 / d192: the backend decode engine is ahead (1.04-2.4x) -> TRAIL.
- prefill, d512 and its d320-d448 envelope: 0.32-0.65 on every cell from a 2k cache; below that FROST
  still wins from ``b * h_q * s_q >= 4096`` query rows (0.39-0.88; 2048 rows lose 1.14-1.24).
- chunked prefill (``s_q < s_kv``), d64-d256, by launch size in 128-row Q tiles
  ``b * h_q * ceil(s_q / 128)``: ``<= 128`` tiles win from a 4k cache (0.30-0.89); 256+ tiles lose
  1.02-1.07 for d64/d128. d256 chunks win at any launch size (0.90-0.99, 2 of ~45 cells 1.00-1.04).
- dense squares d64-d256: parity to 1.07 (d64/d128 1.0-1.07, d192 0.98-1.08, d256 0.96-1.08), sliding
  window 2.8x -> TRAIL. Ragged prefill keeps the backend unless a split rule below applies.
- paged THD, exact d256 BF16, causal bottom-right without Stats/SWA/sinks: a separate public
  cuDNN 9.26.0.51 follow-up (2026-09-27, CuTeDSL 4.7, CUDA 13.0, CUPTI graph replay with L2
  flushed) measures 0.52-0.72 against the backend across per-rank heads 16/2, 8/1, 4/1, 2/1,
  HND/NHD pools and page sizes 16/128. The matrix includes full 2k/8k, mixed full and
  B4 Q2k/KV16k; a 32k endpoint at 16/2 uses HND/page128. Independent B2/B3 full/chunk and
  irregular-length controls confirm the bounded interpolation below. This is a cuDNN route improvement, not a uniform win over
  FA4/TRTLLM. Keep unmeasured graph features and larger declarations backend-first.
- paged THD, exact d128 BF16, B1..4 with 4..64 query heads and integral GQA1/2/4/8:
  prepared single-CTA split plans cover bounded short-query/long-cache work.
  The shared split rule below owns the measured shape/layout limits; no-Stats
  graphs lead the backend only when that rule actually selects splitting.
- nonpaged THD, exact d192/v128 BF16 with equal Q/KV head counts: the shared
  nonpaged split rule bounds the measured short-query/long-cache shard. Its
  single-CTA split leads the backend with or without packed Stats. Exact
  d128/v128 BF16 Blackwell prefixes without Stats reuse this first-wave rule
  for integral GQA1/2/4/8 and fixed graphs (B200, released cuDNN 9.27).
  The existing order remains when there is no first-wave split to use.

SM120 f16/bf16 row (RTX PRO 6000, 188 SMs): 0.16-0.69 on every model and phase, with two measured
exceptions: ``s_q == 1`` at b = 1 loses 1.13-1.85 on every head dim (fewer than 8 KV units), and the
d512 row with a 128-wide GQA group (DeepSeek-V4 "pro", 128 query heads over one KV head) loses
5-10x at every batch; sliding-window dense squares (gpt_oss 2k x 2k, 8k x 8k) are 1.28-1.48 -> TRAIL
on those three, LEAD everywhere else.
The small-batch d512 head-count shortcut was measured only at 128k KV on SM120 too;
restrict it to that domain as a conservative policy. No new SM120 timing is claimed.

SM90 f16/bf16 row (H100 SXM; d512 only, the row floors its envelope at 256): prefill 0.22-0.40 and
``2 <= s_q <= 16`` 0.02-0.63 on every cell. There is no split-KV on SM90, so ``s_q == 1`` wins only
with ``b * h_q >= 512`` query rows in flight (0.25-0.95); 256 rows are parity (0.96-1.11) and fewer
lose up to 33x. THD, sliding window, sinks and envelope widths below 512 were not measured -> TRAIL.

SM100 per-tensor FP8 row (B200, E4M3 Q/K/V/O, Amax_O; FROST does not produce Amax_S, so graphs
requesting it keep the backend): prefill d256 0.47-0.58 and d512 0.34-0.49 -> LEAD; d192 0.92-0.99 and
d128 0.98-1.03 squares -> TRAIL. ``s_q == 1``: d128 loses 1.3-2.7x; for d192-d512 the backend has no
engine. ``2 <= s_q <= 16``, THD, paged, window, sinks and block-scaled O were not measured -> TRAIL.
Short prefill (s_q 17-512, s_kv 64-8192, b 1/8; GPU replay and eager back-to-back submission): FROST
submits in ~16-18 us against the backend's ~13 us, so a single-wave launch with a cache below 1k is
host-bound and loses eager 1.03-1.37 while winning GPU time; d512 with >= 32 query heads wins both at
every cache (0.20-0.73 GPU), d512 with 8 heads loses GPU 1.04-1.11 at a 128 cache. Chunks: d192 at 128
Q tiles loses 1.19-1.72 (16-64 tiles win 0.34-0.90); d128 at 64-512 tiles wins 0.21-0.68 while
``s_q <= 128``, and <= 128 tiles win 0.19-0.93 at any ``s_q``. A mask-free graph whose S_kv is off the
KV tile runs the synthesized-padding path (no split-KV, ~68 us per eager submission): it leads only
from ``Q tiles * s_kv >= 2**21`` (0.41-0.89 above, up to 5.6x eager / 1.36 GPU below; bound fitted on
the 72-case random hold-out that found it).

SM107 half uses the shared paged/nonpaged native THD split selectors and the
measured packed-GQA paged prefill contract. Selected native splits retain
priority; other eligible graphs retain the backend first. Quantized Rubin rows
remain opt-in. Qualification and timing evidence are maintained internally.

Rows with no measurement (SM80, mxfp8) keep the historical order (LEAD); they are still
opt-in, so the order is only observable with ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``.
"""

from __future__ import annotations

import cudnn

from .engines import Capabilities, _selected_d_shape, _synth_kv_padding

LEAD = "lead"  # FROST's proposals ahead of the backend block
TRAIL = "trail"  # the backend block ahead of FROST's proposals

# SM100 f16/bf16 thresholds (provenance in the module docstring).
DECODE_SHAPED_MAX_S_Q = 16  # spec-decode verify depth; the backend has no decode-class engine above s_q == 1
SHORT_QUERY_MIN_KV_TOKENS = 2048  # lower measured KV bound for d512 prefill placement
SQ1_MIN_KV_UNITS = 32  # s_q == 1, d256 / d512: b * h_kv from which FROST wins at every kv (0.54-0.75)
SQ1_SMALL_BATCH_MIN_UNITS = 4  # s_q == 1, d256: below 4 units FROST loses up to 128k KV (1.07-2.9)
SQ1_SMALL_BATCH_MIN_KV_TOKENS = 2**16  # s_q == 1, d256, 4 <= units < 32: KV tokens in flight (units * s_kv) from which FROST wins (0.74-0.97)
SQ1_MQA_MIN_Q_HEADS = 32  # s_q == 1, d512 (one KV head): FROST packs the query group
SQ1_MQA_MIN_KV_TOKENS = 131072  # small-batch shortcut: shortest verified winning KV length; keep shorter caches on backend
# s_q == 1, d512 (SM100): KV tokens in flight (b * h_kv * s_kv) from which FROST wins, by query-head count; one KV unit needs twice that.
SQ1_D512_MIN_KV_TOKENS_WIDE = 2**14  # h_q >= 32: 0.57-0.94 at the bound, 1.02-1.04 below
SQ1_D512_MIN_KV_TOKENS_NARROW = 2**15  # h_q < 32: 0.56-0.92 at the bound, 1.09-1.13 below
SQ1_D512_WIDE_Q_HEADS = 32
SQ1_D512_UNSPLIT_MIN_UNITS = 16  # s_q == 1, d512, S_kv off the KV tile (no split): 24 units 0.10-0.66, 12 units 0.96, 2-3 units 1.01-2.17
D512_PREFILL_MIN_Q_ROWS = 4096  # d512 prefill below a 2k cache: b * h_q * s_q from which FROST wins (0.39-0.88); 2048 rows lose 1.14-1.24
# chunked prefill (a chunk attending to a longer cache), by launch size in 128-row Q tiles (b * h_q * ceil(s_q / 128)):
CHUNKED_MAX_Q_TILES = 128  # <= 128 tiles wins from a 4k cache (0.30-0.89, d64-d256); 256 tiles loses 1.02-1.07 for d64/d128
CHUNKED_MIN_KV_TOKENS = 4096
CHUNKED_SQUARE_MIN_KV_TOKENS = 32768  # s_q == s_kv at <= 128 tiles: kept from the 2026-09-18 bound, not re-measured

# B200 paged THD prefill shard; conservative bounds on graph declarations.
PAGED_D256_PREFILL_HEADS = frozenset({(16, 2), (8, 1), (4, 1), (2, 1)})
PAGED_D256_PREFILL_PAGE_SIZES = frozenset({16, 128})
PAGED_D256_PREFILL_MIN_Q = 2048
PAGED_D256_PREFILL_MAX_KV = 32768
PAGED_D256_PREFILL_MAX_BATCH = 4


# SM120 f16/bf16 thresholds.
SM120_SQ1_MIN_KV_UNITS = 8  # s_q == 1: b * h_kv below this (b = 1) loses 1.13-1.85 on every head dim
SM120_SQ1_MAX_GQA_GROUP = 64  # s_q == 1, d512: a 128-wide query group over one KV head loses 5-10x at every batch

# SM90 f16/bf16 thresholds.
SM90_SQ1_MIN_Q_ROWS = 512  # s_q == 1 (no split-KV on SM90): b * h_q >= 512 wins 0.25-0.95; 256 is parity, below loses up to 33x

# SM100 per-tensor FP8 thresholds (prefill, s_q > 16).
FP8_D512_WIDE_Q_HEADS = 32  # d512 with >= 32 query heads wins at every measured cache (0.20-0.73 GPU, 0.20-0.91 eager)
FP8_MIN_KV_TOKENS = 1024  # below, a single-wave launch is host-bound: FROST's ~4 us extra submit cost loses eager 1.03-1.37
FP8_D192_MAX_Q_TILES = 64  # d192 chunks: 16-64 tiles win 0.34-0.90; 128 tiles lose 1.19-1.72
FP8_D128_ANY_TILES_MAX_S_Q = 128  # d128 chunks with s_q <= 128 win at 64-512 tiles (0.21-0.68)
FP8_SYNTH_KV_MIN_TILE_KV = 2**21  # S_kv off the KV tile, mask-free: Q tiles * s_kv from which FROST wins (0.41-0.89)


def _q_tiles(facts) -> int:
    return facts.b * facts.h_q * -(-facts.s_q // 128)


def place(spec, facts) -> str:
    """``LEAD`` or ``TRAIL`` for the row ``spec`` serving ``facts`` (see the module docstring).

    Keyed by the row's name: the SM100, SM120 and SM90 f16/bf16 rows and the SM100 FP8 row each use
    their measured shard table. The SM107 half row leads for a selected native THD split
    or qualified packed paged prefill. Every unmeasured row (SM80,
    mxfp8) keeps the historical order -- those stay opt-in, so the order is only
    observable with the flag set, which ranks ours first anyway."""
    if spec.name == "sdpa_fwd_prefill_sm107":
        return _place_sm107_f16(spec.capabilities, facts)
    if spec.name == "sdpa_fwd_prefill_sm100":
        return _place_sm100_f16(spec.capabilities, facts)
    if spec.name == "sdpa_fwd_prefill_sm120":
        return _place_sm120_f16(spec.capabilities, facts)
    if spec.name == "sdpa_fwd_prefill_sm90":
        return _place_sm90_f16(spec.capabilities, facts)
    if spec.name == "sdpa_fwd_prefill_sm100_fp8":
        return _place_sm100_fp8(spec.capabilities, facts)
    return LEAD


def _place_sm107_f16(caps: Capabilities, facts) -> str:
    from .heuristics import _prefer_thd_pack_gqa, nonpaged_thd_split_choice, paged_thd_split_choice

    if facts.device_cc != (10, 7):
        return TRAIL
    # Reuse candidate generation's launch budget for the native packed split.
    if nonpaged_thd_split_choice(caps, facts) > 1 or paged_thd_split_choice(caps, facts)[0] > 1:
        return LEAD
    # The shared paged pipeline also benefits from GQA packing without a
    # split. Large-batch short queries recover unused Q rows without partials.
    # Smaller GPU-only gains do not reliably repay the host submission cost;
    # leave full/long-Q placement unchanged. Reuse the packing preference.
    if (
        facts.has_paged_kv
        and not facts.shape_overrides
        and facts.dtype == cudnn.data_type.BFLOAT16
        and 8 <= facts.b <= 64
        and 4 <= facts.h_q <= 64
        and facts.h_kv > 0
        and 64 <= facts.s_q <= 128
        and 2048 <= facts.s_kv <= 32768
        and facts.page_size == 16
        and facts.bottom_right
        and facts.window_left is None
        and not facts.has_sink
        and not facts.right_band_widening
        and facts.k_t is not None
        and facts.k_t.get_stride()[2] < facts.k_t.get_stride()[1]
        and _prefer_thd_pack_gqa(caps, facts)
    ):
        return LEAD
    return TRAIL


def _place_sm100_fp8(caps: Capabilities, facts) -> str:
    # Measured: dense per-tensor E4M3, O E4M3 + Amax_O, no sink / window / block-scaled O.
    if facts.thd or facts.has_paged_kv or facts.window_left is not None or facts.has_sink or facts.o_block_scale:
        return TRAIL
    flavor = _selected_d_shape(caps, facts)
    if facts.s_q == 1:
        return TRAIL if flavor in ((64, 64), (128, 128)) else LEAD  # d128 loses 1.3-2.7x; d192+ has no backend engine
    if facts.s_q <= DECODE_SHAPED_MAX_S_Q:
        return TRAIL  # not measured for FP8
    tiles = _q_tiles(facts)
    if _synth_kv_padding(caps, facts) and tiles * facts.s_kv < FP8_SYNTH_KV_MIN_TILE_KV:
        return TRAIL
    if flavor == (512, 512) and facts.h_q >= FP8_D512_WIDE_Q_HEADS:
        return LEAD
    if facts.s_kv < FP8_MIN_KV_TOKENS and tiles <= (facts.device_sm_count or 148):
        return TRAIL
    if flavor in ((256, 256), (512, 512)):
        return LEAD  # past the launch-bound region: d256 0.09-0.69, d512 0.20-0.76
    if facts.s_q >= facts.s_kv:
        return TRAIL  # d128 squares parity (0.85-1.05), d192 0.84-1.01
    if flavor == (192, 128):
        return LEAD if tiles <= FP8_D192_MAX_Q_TILES else TRAIL
    return LEAD if tiles <= CHUNKED_MAX_Q_TILES or facts.s_q <= FP8_D128_ANY_TILES_MAX_S_Q else TRAIL


def _place_sm90_f16(caps: Capabilities, facts) -> str:
    # Measured only dense, exact d512 without window/sink; everything else stays backend-first.
    if facts.thd or facts.window_left is not None or facts.has_sink or (facts.d_qk, facts.d_v) not in caps.d_shapes:
        return TRAIL
    if facts.s_q == 1 and facts.b * facts.h_q < SM90_SQ1_MIN_Q_ROWS:
        return TRAIL
    return LEAD


def _place_sm120_f16(caps: Capabilities, facts) -> str:
    if not facts.thd and facts.s_q == 1:
        units = facts.b * facts.h_kv
        if _selected_d_shape(caps, facts) == (512, 512):
            if facts.h_q // max(facts.h_kv, 1) > SM120_SQ1_MAX_GQA_GROUP:
                return TRAIL
            # one KV head: the b = 1, >=32-query-head win (0.44-0.85) was measured at 128k KV only.
            return LEAD if units >= SM120_SQ1_MIN_KV_UNITS or (facts.h_q >= SQ1_MQA_MIN_Q_HEADS and facts.s_kv >= SQ1_MQA_MIN_KV_TOKENS) else TRAIL
        return LEAD if units >= SM120_SQ1_MIN_KV_UNITS else TRAIL
    if facts.window_left is not None and facts.s_q == facts.s_kv:
        return TRAIL
    return LEAD


def _in_paged_d256_prefill_domain(facts) -> bool:
    """Bounded shard derived from measured workloads; inspect only graph declarations."""
    return (
        facts.device_cc == (10, 0)
        and facts.dtype == cudnn.data_type.BFLOAT16
        and (facts.d_qk, facts.d_v) == (256, 256)
        and facts.has_paged_kv
        and facts.thd
        and facts.causal
        and facts.bottom_right
        and facts.window_left is None
        and not (facts.has_sink or facts.wants_stats or facts.has_epilogue_gate)
        and (facts.h_q, facts.h_kv) in PAGED_D256_PREFILL_HEADS
        and facts.page_size in PAGED_D256_PREFILL_PAGE_SIZES
        and 1 <= facts.b <= PAGED_D256_PREFILL_MAX_BATCH
        and PAGED_D256_PREFILL_MIN_Q <= facts.s_q <= facts.s_kv <= PAGED_D256_PREFILL_MAX_KV
    )


def _place_sm100_f16(caps: Capabilities, facts) -> str:
    from .heuristics import nonpaged_thd_split_choice, paged_thd_split_choice

    # The prepared single-CTA split removes the underfilled paged D128
    # launch. Placement and the concrete split share one bounded rule.
    if not facts.wants_stats and paged_thd_split_choice(caps, facts)[0] > 1:
        return LEAD
    if nonpaged_thd_split_choice(caps, facts) > 1:
        return LEAD
    dense = not facts.thd
    if dense and 2 <= facts.s_q <= DECODE_SHAPED_MAX_S_Q:
        return LEAD  # the backend's multi-token path is prefill-class at every cache length
    flavor = _selected_d_shape(caps, facts)
    if dense and facts.s_q == 1:
        units = facts.b * facts.h_kv
        if flavor in ((256, 256), (512, 512)) and _synth_kv_padding(caps, facts):
            # S_kv off the KV tile cannot split; unsplit small launches lose (b3 h_kv=1: 1.98x d512, d256 12 units: 2.03x).
            return LEAD if units >= (SQ1_D512_UNSPLIT_MIN_UNITS if flavor == (512, 512) else SQ1_MIN_KV_UNITS) else TRAIL
        if flavor == (512, 512):
            need = SQ1_D512_MIN_KV_TOKENS_WIDE if facts.h_q >= SQ1_D512_WIDE_Q_HEADS else SQ1_D512_MIN_KV_TOKENS_NARROW
            return LEAD if units * facts.s_kv >= need * (2 if units == 1 else 1) else TRAIL
        if flavor == (256, 256):
            if units >= SQ1_MIN_KV_UNITS:
                return LEAD
            if units >= SQ1_SMALL_BATCH_MIN_UNITS and units * facts.s_kv >= SQ1_SMALL_BATCH_MIN_KV_TOKENS:
                return LEAD
            return TRAIL
        return TRAIL  # d64 / d128 / d192: the backend decode engine is ahead (1.04-2.4x)
    # prefill-shaped
    if _in_paged_d256_prefill_domain(facts):
        return LEAD
    if facts.thd or facts.has_paged_kv or facts.window_left is not None:
        return TRAIL
    if flavor == (512, 512):  # exact or envelope-served (d320-d448: 0.32-0.65)
        return LEAD if facts.s_kv >= SHORT_QUERY_MIN_KV_TOKENS or facts.b * facts.h_q * facts.s_q >= D512_PREFILL_MIN_Q_ROWS else TRAIL
    if (facts.d_qk, facts.d_v) not in caps.d_shapes:  # envelope widths below d512: unmeasured
        return TRAIL
    chunked = facts.s_q < facts.s_kv and facts.s_kv >= CHUNKED_MIN_KV_TOKENS
    if _q_tiles(facts) <= CHUNKED_MAX_Q_TILES and (chunked or facts.s_kv >= CHUNKED_SQUARE_MIN_KV_TOKENS):
        return LEAD
    if flavor == (256, 256) and facts.causal and chunked:
        return LEAD  # d256 chunked at any launch size: 0.90-0.99 (2 of ~45 cells 1.00-1.04)
    return TRAIL
