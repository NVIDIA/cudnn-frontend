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
- ``s_q == 1``, PAGED d512 (the d512 kernel's PAGED_KV specialization, #1093): no d512 decode
  tile exists, so the role-split prefill tile serves it and loses to the backend's paged decode
  engine on the FlashInfer shape (B200, public cuDNN 9.26.0.51, this tree, CUDA-graph replay,
  kernel time, 2026-10-07; b = 8, 64/1 and 64/8 heads, page 16, bf16, mixed KV <= 4096:
  77.8 vs 65.1 us and 136.9 vs 102.5 us) -> TRAIL; the
  dense d512 rule above was fitted on dense K/V and does not transfer. Multi-token paged d512
  keeps the decode-shaped LEAD (the backend's multi-token path is prefill-class there too).
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
- paged THD, exact d128 BF16, B1..4 with 4..64 query heads and integral GQA1/2/4/8/16:
  prepared single-CTA split plans cover Q64..1024 and KV2K..32K, with or without
  packed Stats. The shared split rule below owns the measured shape/layout
  limits; graphs lead the backend only when that rule selects splitting.
- nonpaged THD, exact d192/v128 BF16 with equal Q/KV head counts: the shared
  nonpaged split rule bounds the measured short-query/long-cache shard. Its
  single-CTA split leads the backend with or without packed Stats. Exact
  d128/v128 BF16 Blackwell prefixes without Stats reuse this first-wave rule
  for integral GQA1/2/4/8 and fixed graphs (B200, released cuDNN 9.27).
  The existing order remains when there is no first-wave split to use.
- nonpaged THD, d128 half, bottom-right causal GQA2/4/8/16 without window, sink or right band (the
  groups whose first plan is packed): unsplit
  FROST leads at KV > 512 with b * h_q * s_q >= 110 query rows per SM, or Q >= 256 at KV >= 1024 (B200,
  cuDNN 9.27, 2026-10-08: 90 qualifying cases, 0.44-1.00 warm, median 0.89; one cold 1.10 at warm 0.99;
  a 68-SM SM100: 19 of 31 small launches lead at 0.40-0.95, none slower; B300: 48 leads at 0.25-0.95).
  Smaller launches lost up to 1.54x; GQA1 and non-causal graphs (unpacked) keep the backend first.
  GQA2 (packed since 2026-10-09) also needs KV >= 4096: B300, 72 cases, packed 0.53-0.97 there; on a
  1-2k cache large batches lost up to 1.26x.

SM120 f16/bf16 row (RTX PRO 6000, 188 SMs): 0.16-0.69 on every model and phase, with two measured
exceptions: ``s_q == 1`` at b = 1 loses 1.13-1.85 on every head dim (fewer than 8 KV units), and the
d512 row with a 128-wide GQA group (DeepSeek-V4 "pro", 128 query heads over one KV head) loses
5-10x at every batch; sliding-window dense squares (gpt_oss 2k x 2k, 8k x 8k) are 1.28-1.48 -> TRAIL
on those three, LEAD everywhere else.
Re-measured 2026-10-08 against public cuDNN 9.27.0.42 (same part, CUDA-graph replay, 399 cases incl. a
random hold-out): prefill and ``2 <= s_q <= 16`` still lead (prefill median 0.42, decode-shaped
0.02-0.45 for d64/d128/d256), but ``s_q == 1`` changed: d64/d128 decode is backend-first at every
batch (FROST 0.96-3.5x), query groups FROST cannot pack (5, 6, 12) re-read K/V per head (1.5-5.6x),
and d256 needs 12 KV units -> TRAIL outside those (mean regret over the 399 cases 8.5% -> 2.2%). The
hold-out informed these bounds, so it is no longer out-of-sample. The d512 one-KV-head shortcut keeps its 128k bound: 32/1 heads won 0.29-0.75 from
6k KV here but ran 1.31x slower at 6k on an RTX 5090.

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
KV tile runs the KV-tail-mask path (no split-KV): it leads only from ``Q tiles * s_kv >= 2**21``
(0.41-0.89 above, up to 5.6x eager / 1.36 GPU below; bound fitted on the 72-case random hold-out that
found it, when the path also cost ~68 us per eager submission). #1425 removed that host cost (16-17 us,
as tile-aligned graphs); the bound stays until split-KV rides the path and the hold-out is re-fit.

SM107 half uses the shared paged/nonpaged native THD split selectors and the
measured packed-GQA paged prefill contract. Selected native splits retain
priority. Dense graphs (no THD, paging, window or sink), measured 2026-10-08 on a gr100 board
(216 SMs) against public cuDNN 9.27.0.42, 620 cases (decode, chunked prefill, squares, d512
envelope, random hold-out); develop led none of them (mean regret 116%):
- ``2 <= s_q <= 16``: d192/d256/d512 lead (0.04-0.75; the backend runs these 2-10x slower);
  d64/d128 lead at one KV unit or <= 8 units with >= 8k KV (0.07-0.63), 64+ units lose 1.1-1.6x.
- prefill: d512 and its envelope lead (0.08-0.67); chunks of <= 128 Q tiles from a 4k cache lead
  (d64-d256, 0.04-1.00). Squares and wider chunks are mixed (up to 1.98x) -> TRAIL.
- ``s_q == 1`` keeps the backend (d64/d128/d256/d512 1.6-25x; d192 mixed).
New mean regret 3.2%, no lead slower than 1.03x warm or cold. The board's timing is noisy for the
first arm measured (FROST), which biases these ratios against FROST. The per-tensor FP8
Rubin row remains opt-in.

- dense d128 half (the d64 envelope included) decode / verify WITH an attention sink, bottom-right
  causal ``1 <= s_q <= 16`` or mask-free ``s_q == 1``, GQA 4 / 8 / 16, ``b * h_kv >= 32`` units, caches
  1k-16k, no window / right band / pre-folded scale (issue #1472; 216-SM cc 10.7, cuDNN 9.26.0.51 and
  9.27.0.28, CUDA-graph replay, kernel time, 2026-10-08): the row's default -- the packed decode tile,
  or the packed cga2 prefill body once ``s_q * G > 128`` -- runs 0.06-0.42 of the backend's default plan
  on every multi-token cell of the 150-cell family (b 8 / 32 / 128 x 64/8, 64/4 x q 1 / 4 / 8 / 16 x
  KV 1k / 2k / 4k / 8k / 16k), 0.06-0.30 on the 54 multi-token cells of the 90-cell grid (b 32 / 64 /
  128 x 64/8, 64/4, 32/8 x q 1 / 4 / 8 / 16 x KV 2k / 8k; 32/8 at 0.23-0.30) and 0.06-0.21 on the 36
  multi-token cells of the 9.27 twin (b 32 / 128 x 64/8, 64/4 x q 1 / 4 / 8 / 16 x KV 2k / 8k / 16k;
  the issue's two cells 0.13 and 0.07 on both libraries); the review-fix pass over the band's corners (9.26.0.51, 2026-10-08, another lane's functional run sharing the
  GPU; ratios against the backend's default plan): the head-dim envelope WITHOUT a window -- d64 64/8 at b 8 / 32 /
  128 x q 1 / 4 / 8 x KV 2k / 4k 0.06-0.11 (an f16 twin 0.09; the mask-free q = 1 cell FROST-only), d96 (b8 64/8 q4
  KV 4k, b32 64/4 q16 KV 2k, b32 64/8 q8 KV 8k) 0.10-0.22 -- the 32-unit bound at G = 4 (32/8 b4, 16/4 b8, 4/1 b32,
  8/2 b16; q 4 / 8; KV 1k-8k) 0.26-0.62 and at G = 8 / 16 there (32/4 b8 q8 0.19; 128/8 b4 q = 1 FROST-only), and
  the band's f16 / Stats / padded twins at its corners (b8 64/4 q16 KV 1k 0.20 / 0.20 / 0.54; b128 64/8 q8 KV 16k
  0.13 / 0.13; b128 64/8 q8 KV 8k padded 0.13; b32 64/4 q16 KV 4k Stats + padded 0.27; b8 64/8 q4 KV 4k f16 + Stats
  0.10; b32 32/8 q8 KV 8k f16 0.24; the b32 q = 1 f16 + Stats decode cell FROST-only) -- the three families above are
  bf16, Stats-free and unpadded, so these cells are the band's only measurements of those attributes; the backend declines ``s_q == 1`` with a sink,
  so the decode arm names the row's own default (8-695 us across the band). Outside the band (sliding
  window: GPT-OSS d64 SWA 0.45-0.68 but only two cells; GQA 2 / 32, MHA, partial groups, caches past
  16k, fewer units) the backend keeps the lead until measured.

SM107 MXFP8 row (``sdpa_fwd_prefill_sm107_mxfp8``, offered by default since 2026-10): LEAD on exact
cc 10.7 for every graph the row admits -- dense BSHD, exact d128 / d192x128 / d256 / d512, E4M3 / E5M2
in, half / FP8 / block-scaled O, every mask, sink, Stats on or off, ``s_q >= 1`` on the prefill bodies.
A qualification verdict, not a per-shard timing one: the backend's cc 10.7 MXFP8 engines are not a
qualified alternative. (1) Their Amax_O is wrong on dense MXFP8 graphs (``BACKEND_AMAX_O_ISSUE`` in
test/python/sdpa/fp8.py, cuDNN 9.26.0.51). (2) Their planner crashes the process (SIGSEGV inside the
C++ plan creation -- the ``create_execution_plans`` heuristics query and the explicit
``create_execution_plan(engine_id, knobs)`` engine-config path alike -- after lowering, validate and
build_operation_graph completed) while planning any single-query MXFP8 graph without a sink token: dense
BSHD and BHSD and THD, Stats on or off, every O dtype, E4M3 and E5M2, causal or not, KV 128 / 2048 / 4096,
batch 1 / 2 / 4 -- every d128 contract of the detector's 21-contract matrix, measured on a 216-SM cc 10.7
board with cuDNN 9.26.0.51 and 9.27.0.28 (2026-10-08); a sink makes them plan, and d192x128 / d256 / d512
and paged pools decline cleanly there; on cuDNN 9.28.0 (the cc 10.7 CI lane) the heuristics plan every
contract but building the backend's BHSD single-query plan kills the process instead, so no build is clean yet;
``sdpa/fwd/backend_guard.py`` keeps the backend out of planning on that domain on every known build (the
detector re-measures the whole matrix through plan, check_support and build on every run of that lane), and
an explicit backend pin there is a typed decline. (3) Their d256 and d512
MXFP8 plans are offered but fail to build on both engines (NVRTC
``CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED``, same board, 9.26.0.51 and 9.27.0.28), so without
this row those two flavors have no provider on cc 10.7. Timing is evidence, not the criterion -- measured
2026-10-08 on that board (cuDNN 9.26.0.51; every plan of the flag-less [A, FALLBACK] list on a fresh graph,
CUDA-graph replay, 7 interleaved rounds of 50 replays after an L2 flush, SM clock 2364-2424 MHz unless
noted, median us; the harness numerics checks passed on every FROST plan and on every backend plan except
where Amax_O is named):
  B2 H8/2 S4096 d128 dense, bf16 O ........ FROST 49.6 vs eng16 49.0 (1.01x)
  B2 H8/2 S4096 d128 causal ............... FROST 37.6 (LPT_L2; LPT 35.6) vs eng16 30.0 (1.25x)
  B2 H8/2 S4096 d128 Stats, e4m3 O ........ FROST 52.2 vs eng16 50.5 (1.03x)
  B2 H8/2 S4096 d192x128 causal, sink ..... FROST 38.2 vs eng16 31.1 (1.23x)
  B1 H16/4 S8192 d256 causal .............. FROST 125.0; neither backend plan builds (defect 3)
  B2 H8/8 S4096 d512 dense ................ FROST 92.1; neither backend plan builds (defect 3)
  B4 H8/2 S_q 1 KV2048 d128 ............... FROST 16.1; the backend is not consulted (defect 2)
  B4 H8/2 S_q 1 KV2048 d128, sink ......... FROST 16.0 vs eng3 20.0 (0.80x; eng3 reports Amax_O = 0, defect 1), eng16 16.8
  B64 H8/2 S_q 1 KV4096 d128, sink ........ FROST 113.3 vs eng3 81.5 / eng16 70.1 (1.39x / 1.62x)
  B128 H32/8 S_q 1 KV2048 d128, sink ...... FROST 766.8 vs eng3 423.3 / eng16 375.8 (1.81x; board at 2256 MHz and falling)
  B64 H8/2 S_q 8 KV4096 d128 .............. FROST 114.2 vs eng3 81.6 / eng16 69.9 (1.40x / 1.63x)
  B64 H8/2 S_q 8 KV4096 d128 causal-BR, sink  FROST 129.3 (NATURAL 122.5) vs eng16 91.7 (1.41x)
Prefill shapes sit at parity to 1.25x of the backend's pick; decode-shaped graphs (S_q <= 8 at large batch)
run the row's 512-row 2-CTA prefill tile with 1..8 live rows and trail the backend's decode-shaped engines
by 1.4-1.8x where those plan at all (with a sink token; without one the backend's planner crashes). The
lead stays a qualification verdict (FROST-first; the backend's plans carry defects 1-3): the gap is a
kernel follow-up -- an MXFP8 decode tile for cc 10.7, recorded in SUPPORT_MATRIX_TRACKER.md's gaps table
-- not a backend-relative placement rule. Devices other than exact cc 10.7 (10.8-11.9 are in the row's
arch range) TRAIL until measured.

Rows with no measurement (SM80, SM100 mxfp8) keep the historical order (LEAD); they are still
opt-in, so the order is only observable with ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``.
"""

from __future__ import annotations

import cudnn

from .engines import Capabilities, _selected_d_shape, _synth_kv_padding, rubin_dense_d128_shared_leg

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
THD_PACKED_MIN_KV_TOKENS = 512  # exclusive: unsplit packed THD at KV 512 measured 1.0-1.54x the backend
THD_PACKED_GQA2_MIN_KV_TOKENS = 4096  # GQA2 packs two heads a tile: large batches on a 1-2k cache lost up to 1.26x
# Small unsplit packed launches lose to the backend (KV 576-1024 below ~110 query rows per SM: up to
# 1.43x on B200 and a 68-SM SM100); 8192 rows lost on B200 (148 SMs) and won 0.71-0.89 on 68 SMs.
THD_PACKED_MIN_Q_ROWS_PER_SM = 110  # h_q * (b * s_q, or a smaller declared token total) per SM
THD_PACKED_LONG_Q = 256  # ... except long sequences (by declared total / b): Q256 KV1024 at 4096-8192 rows ran 0.79-0.81
THD_PACKED_LONG_Q_MIN_KV = 1024
CHUNKED_SQUARE_MIN_KV_TOKENS = 32768  # s_q == s_kv at <= 128 tiles: kept from the 2026-09-18 bound, not re-measured

# B200 paged THD prefill shard; conservative bounds on graph declarations.
PAGED_D256_PREFILL_HEADS = frozenset({(16, 2), (8, 1), (4, 1), (2, 1)})
PAGED_D256_PREFILL_PAGE_SIZES = frozenset({16, 128})
PAGED_D256_PREFILL_MIN_Q = 2048
PAGED_D256_PREFILL_MAX_KV = 32768
PAGED_D256_PREFILL_MAX_BATCH = 4


# SM120 f16/bf16 thresholds.
SM120_SQ1_MIN_KV_UNITS = 8  # s_q == 1: b * h_kv below this (b = 1) loses 1.13-1.85 on every head dim
SM107_DECODE_SHAPED_MAX_UNITS = 8  # 2 <= s_q <= 16, d64/d128: b * h_kv above this lost to the backend
SM107_DECODE_SHAPED_MIN_KV_TOKENS = 8192  # ... and at 8 units a 2k cache is parity (0.91-1.06)
SM120_SQ1_D256_MIN_KV_UNITS = 12  # s_q == 1, d256: 8 units at 2k KV lost 1.55x; 12+ ran 0.74-1.04 (cuDNN 9.27)
SM120_SQ1_MAX_GQA_GROUP = 64  # s_q == 1, d512: a 128-wide query group over one KV head loses 5-10x at every batch

# SM90 f16/bf16 thresholds.
SM90_SQ1_MIN_Q_ROWS = 512  # s_q == 1 (no split-KV on SM90): b * h_q >= 512 wins 0.25-0.95; 256 is parity, below loses up to 33x

# SM107 half, dense d128 decode / verify WITH an attention sink on the shared SM100 bodies (the packed decode tile, or the
# packed cga2 prefill body once S_q x G > 128): issue #1472; the measured band is in the module docstring (216-SM cc 10.7,
# cuDNN 9.26.0.51 and 9.27.0.28, CUDA-graph replay, kernel time, round-robin arms, 2026-10-08).  Units are b * h_kv (one
# packed unit each) and KV tokens, never waves.
SM107_SINK_DECODE_MAX_S_Q = 16  # bottom-right causal verify rows measured (1 / 4 / 8 / 16)
SM107_SINK_DECODE_GROUPS = (4, 8, 16)  # the GQA groups measured (32/8, 64/8, 64/4); each divides the 128-row tile
SM107_SINK_DECODE_MIN_UNITS = 32  # b * h_kv from which the packed body beats the backend default at every measured KV
SM107_SINK_DECODE_KV_TOKENS = (1024, 16384)  # measured cache band (the backend declines s_q == 1 with a sink at any cache)

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
    their measured shard table. The SM107 half row leads for a selected native THD split,
    qualified packed paged prefill, the measured dense shards, or dense d128 decode / verify with an
    attention sink inside its measured band; the SM107 MXFP8 row leads on exact cc 10.7 for every graph it
    admits (a qualification verdict, module docstring). Every unmeasured row (SM80, SM100 mxfp8, the
    fp8 rows of SM107 / SM120) keeps the historical order -- those stay opt-in, so the order is only
    observable with the flag set, which ranks ours first anyway."""
    if spec.name == "sdpa_fwd_prefill_sm107":
        return _place_sm107_f16(spec.capabilities, facts)
    if spec.name == "sdpa_fwd_prefill_sm107_mxfp8":
        return _place_sm107_mxfp8(spec.capabilities, facts)
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
    from .heuristics import _prefer_paged_d256_lpt, _prefer_thd_pack_gqa, nonpaged_thd_split_choice, paged_d256_prefix_launch, paged_thd_split_choice

    if facts.device_cc != (10, 7):
        return TRAIL
    # Dense d128 half decode / verify with an attention sink (issue #1472): the legs that lower onto the shared SM100 bodies
    # (engines.rubin_dense_d128_shared_leg -- dense, the (128, 128) flavor incl. the d64 envelope, no pre-folded scale),
    # packed over a measured GQA group, so the default plan is the packed decode tile (S_q x G <= 128) or the packed cga2
    # prefill body above it.  LEAD only inside the measured band (module docstring); the backend declines s_q == 1 with a
    # sink, so the decode arm names the row's own default there.
    if (
        rubin_dense_d128_shared_leg(caps, facts)
        and not facts.shape_overrides
        and facts.has_sink
        and facts.dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16)
        and facts.window_left is None
        and not facts.right_band_widening
        and ((facts.causal and facts.bottom_right and 1 <= facts.s_q <= SM107_SINK_DECODE_MAX_S_Q) or (not facts.causal and facts.s_q == 1))
        and facts.h_kv > 0
        and facts.h_q % facts.h_kv == 0
        and facts.h_q // facts.h_kv in SM107_SINK_DECODE_GROUPS
        and facts.b * facts.h_kv >= SM107_SINK_DECODE_MIN_UNITS
        and SM107_SINK_DECODE_KV_TOKENS[0] <= facts.s_kv <= SM107_SINK_DECODE_KV_TOKENS[1]
    ):
        return LEAD
    if _prefer_paged_d256_lpt(facts):
        return LEAD
    # A full first wave can still favor FROST even when splitting adds cost.
    # Share the qualified prefix envelope with candidate generation.
    if paged_d256_prefix_launch(caps, facts) is not None or nonpaged_thd_split_choice(caps, facts)[0] > 1 or paged_thd_split_choice(caps, facts)[0] > 1:
        return LEAD
    if not (facts.thd or facts.has_paged_kv or facts.has_sink) and facts.window_left is None:
        # Dense, measured on a gr100 board (216 SMs) against cuDNN 9.27 (2026-10-08).
        flavor = _selected_d_shape(caps, facts)
        if 2 <= facts.s_q <= DECODE_SHAPED_MAX_S_Q:
            if flavor in ((192, 128), (256, 256), (512, 512)):
                return LEAD  # 0.04-0.75 of the backend
            units = facts.b * facts.h_kv
            if units == 1 or (units <= SM107_DECODE_SHAPED_MAX_UNITS and facts.s_kv >= SM107_DECODE_SHAPED_MIN_KV_TOKENS):
                return LEAD  # d64/d128: 0.07-0.63; 64+ units lost 1.1-1.6x
        elif facts.s_q > DECODE_SHAPED_MAX_S_Q:
            if flavor == (512, 512):
                return LEAD  # d512 and its d320-d448 envelope: 0.08-0.67
            if facts.s_q < facts.s_kv and facts.s_kv >= CHUNKED_MIN_KV_TOKENS and _q_tiles(facts) <= CHUNKED_MAX_Q_TILES:
                return LEAD  # chunked, d64-d256: 0.04-1.00; squares and wider chunks stay mixed (up to 1.98x)
    # The shared paged pipeline also benefits from GQA packing without a
    # split. Large-batch short queries recover unused Q rows without partials.
    # Smaller GPU-only gains do not reliably repay the host submission cost;
    # leave full/long-Q placement unchanged. Reuse the packing preference.
    if (
        facts.has_paged_kv
        and not facts.shape_overrides
        and facts.dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16)
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
        and facts.h_q // facts.h_kv in (4, 8)  # the measured groups; GQA16 packs by default now but was not timed at Q 64-128
        and _prefer_thd_pack_gqa(caps, facts)
    ):
        return LEAD
    return TRAIL


def _place_sm107_mxfp8(caps: Capabilities, facts) -> str:
    """LEAD on exact cc 10.7 for every graph the row admits; TRAIL on any other device.

    A qualification verdict, not a per-shard timing one (module docstring, "SM107 MXFP8 row"): the
    backend's cc 10.7 MXFP8 engines are not a qualified alternative.  Eligibility stays with
    ``engines.mismatch`` (dense BSHD, exact native head dims; THD / paged / split / PackGQA decline
    there) -- nothing is admitted here.  The row's arch range reaches cc 11.9 (``sm_hi``); any part other
    than the one it was qualified on trails, as the half row does."""
    return LEAD if facts.device_cc == (10, 7) else TRAIL


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
        group = facts.h_q // max(facts.h_kv, 1)
        if _selected_d_shape(caps, facts) in ((64, 64), (128, 128)) or group & (group - 1):
            return TRAIL  # d64/d128 and unpackable groups (5, 6, 12): 0.96-3.5x and 1.5-5.6x on cuDNN 9.27
        need = SM120_SQ1_D256_MIN_KV_UNITS if _selected_d_shape(caps, facts) == (256, 256) else SM120_SQ1_MIN_KV_UNITS
        return LEAD if units >= need else TRAIL
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
    from .heuristics import _prefer_thd_pack_gqa, nonpaged_thd_split_choice, paged_thd_split_choice

    # The prepared single-CTA split removes the underfilled paged D128
    # launch. Placement and the concrete split share one bounded rule.
    if paged_thd_split_choice(caps, facts)[0] > 1:
        return LEAD
    if nonpaged_thd_split_choice(caps, facts)[0] > 1:
        return LEAD
    dense = not facts.thd
    if dense and 2 <= facts.s_q <= DECODE_SHAPED_MAX_S_Q:
        return LEAD  # the backend's multi-token path is prefill-class at every cache length
    flavor = _selected_d_shape(caps, facts)
    if dense and facts.s_q == 1:
        units = facts.b * facts.h_kv
        if facts.has_paged_kv and flavor == (512, 512):
            # Paged d512 decode runs the role-split PREFILL tile (no d512 decode tile yet) and measures
            # behind the backend's paged decode engine (module docstring: 77.8 vs 65.1 us
            # at 64/1); the dense d512 rule below was fitted on dense K/V. TRAIL until the tile lands.
            return TRAIL
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
    if (
        facts.thd
        and not facts.has_paged_kv
        # Measured domain only: a 32-token window ran 2.7x slower packed.
        and facts.bottom_right
        and facts.window_left is None
        and not (facts.has_sink or facts.right_band_widening)
        and facts.s_kv > THD_PACKED_MIN_KV_TOKENS
        and (
            facts.h_q * min(facts.b * facts.s_q, facts.max_total_seq_len_q or facts.b * facts.s_q)
            >= THD_PACKED_MIN_Q_ROWS_PER_SM * (facts.device_sm_count or 148)
            or (
                min(facts.s_q, -(-(facts.max_total_seq_len_q or facts.b * facts.s_q) // facts.b)) >= THD_PACKED_LONG_Q
                and facts.s_kv >= THD_PACKED_LONG_Q_MIN_KV
            )
        )
        and (facts.h_q != 2 * facts.h_kv or facts.s_kv >= THD_PACKED_GQA2_MIN_KV_TOKENS)
        and _prefer_thd_pack_gqa(caps, facts)
    ):
        return LEAD  # unsplit packed causal GQA2/4/8/16
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
