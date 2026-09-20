# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the SDPA-forward family's proposals stand against the cuDNN backend, per measured shard.

:func:`place` answers one question for a graph the row admits: do FROST's proposals go ahead of
the backend's own block (``LEAD``) or behind it (``TRAIL``)? ``engines/heuristics._assemble`` then
splices the backend block where the family's :data:`~cudnn.engines.heuristics.BACKEND` marker sits.
Participation is not decided here -- an engine that admits the graph is always in the list, and
``build_plans`` walks past a declined entry -- only the default winner is.

Thresholds are fitted to the recorded measurements. Kernel time (torch profiler, L2 flushed,
median of 20) of the FROST plan divided by the backend plan on the same graph,
``benchmark/attention_inference`` (``cudnn_oss`` vs ``cudnn``), 2026-09-18, cuDNN 9.27, cutlass DSL
4.7.1, CSVs under ``benchmark/attention_inference/results/<config>/{b200,rtxpro6000}/``.
``test_sdpa_fwd_placement.py`` re-derives every decisive cell of those CSVs and asserts this table
agrees, so the numbers below cannot drift away from the data without a red test.
That check verifies the fit to its input data, not an independent holdout.

SM100 f16/bf16 row (B200, 148 SMs, 1965 MHz):

- decode-shaped, ``2 <= s_q <= 16``, dense or paged: 0.02-0.65 on every cell (llama d128, qwen35
  d256, gpt_oss d64, deepseek_v4 d512; q = 2, 3, 4, 8, 16; kv 2k-128k; b 1-128). The backend has
  no decode-class engine for ``s_q > 1``; whichever FROST tile serves the rows, it wins -> LEAD.
- ``s_q == 1``, d256: b = 1 loses everywhere (1.3-5.4x); with ``units = b * h_kv``, ``units >= 32``
  wins 0.54-0.75 at every kv, and ``8 <= units < 32`` wins once ``units * s_kv >= 2**17`` KV tokens
  are in flight (0.59-0.87) and loses below (1.27-1.57). d512 (one KV head): the backend does not
  pack the GQA group, so FROST also wins at b = 1 from 32 query heads at the measured 128k KV
  length. A 2026-09-19 public-cuDNN-9.26 BF16/FP16 follow-up found the same shortcut loses at
  2k-32k KV (1.7-2.2x), including shared K=V, but wins at 128k (0.80-0.88). Require 128k KV
  for this small-batch shortcut; it is a verified point, not an exact measured crossover.
  See ``results/d512_boundary/b200/README.md`` under the inference benchmark for provenance.
- ``s_q == 1``, d128 (and d64 through the d128 envelope): the backend's decode engine is ahead or
  at parity on every measured cell (1.04-1.06 at b = 128, 1.2-3.6x at b = 1, kv 128k) -> TRAIL.
- prefill, d512 (DeepSeek-V4 shared-KV MQA, 8-128 query heads): 0.33-0.75 on every cell, chunked
  and dense squares alike -> LEAD.
- prefill, d128 / d256: a chunk attending to a longer cache wins while the launch is small, in
  128-row Q tiles ``b * h_q * ceil(s_q / 128)`` (``prefill_sweep``: q 256-2048, kv 8k-128k, TP 1-8):
  ``<= 64`` tiles win from an 8k cache (0.52-0.77; 0.11-0.43 at >= 32k), ``<= 128`` tiles win from a
  32k cache (0.58-0.71; parity 0.96-1.03 at 8k), 256 tiles lose 1.09-1.16 at every cache length
  (d256 at 128 tiles is parity, 0.95-1.01). Dense squares 2k-16k are 1.0-1.44, sliding window 2.8x,
  d64 (through the d128 envelope) 1.3-1.5x -> TRAIL. THD and paged prefill are unmeasured -> TRAIL
  (the FlashInfer ragged prefill keeps the backend until it is timed).

SM120 f16/bf16 row (RTX PRO 6000, 188 SMs): 0.16-0.69 on every model and phase, with two measured
exceptions: ``s_q == 1`` at b = 1 loses 1.13-1.85 on every head dim (fewer than 8 KV units), and the
d512 row with a 128-wide GQA group (DeepSeek-V4 "pro", 128 query heads over one KV head) loses
5-10x at every batch; sliding-window dense squares (gpt_oss 2k x 2k, 8k x 8k) are 1.28-1.48 -> TRAIL
on those three, LEAD everywhere else.
The small-batch d512 head-count shortcut was measured only at 128k KV on SM120 too;
restrict it to that domain as a conservative policy. No new SM120 timing is claimed.

Rows with no measurement (SM107, SM80, fp8, mxfp8) keep the historical order (LEAD); they are still
opt-in, so the order is only observable with ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``.
"""

from __future__ import annotations

from .engines import Capabilities, _selected_d_shape

LEAD = "lead"  # FROST's proposals ahead of the backend block
TRAIL = "trail"  # the backend block ahead of FROST's proposals

# SM100 f16/bf16 thresholds (provenance in the module docstring).
DECODE_SHAPED_MAX_S_Q = 16  # spec-decode verify depth; the backend has no decode-class engine above s_q == 1
SQ1_MIN_KV_UNITS = 32  # s_q == 1, d256 / d512: b * h_kv from which FROST wins at every kv (0.54-0.75)
SQ1_SMALL_BATCH_MIN_UNITS = 8  # s_q == 1, d256: below 8 units (b = 1) FROST loses at every kv
SQ1_SMALL_BATCH_MIN_KV_TOKENS = 2**17  # s_q == 1, d256, 8 <= units < 32: KV tokens in flight (units * s_kv) from which FROST wins
SQ1_MQA_MIN_Q_HEADS = 32  # s_q == 1, d512 (one KV head): FROST packs the query group
SQ1_MQA_MIN_KV_TOKENS = 131072  # small-batch shortcut: shortest verified winning KV length; keep shorter caches on backend
# chunked prefill (a chunk attending to a longer cache), by launch size in 128-row Q tiles (b * h_q * ceil(s_q / 128)):
CHUNKED_SMALL_MAX_Q_TILES = 64  # <= 64 tiles wins from an 8k cache (0.52-0.77 at 8k, 0.11-0.43 at >= 32k)
CHUNKED_SMALL_MIN_KV_TOKENS = 8192
CHUNKED_MAX_Q_TILES = 128  # <= 128 tiles wins from a 32k cache (0.58-0.71; parity 0.96-1.03 at 8k); 256 tiles loses 1.09-1.16 at every cache length
CHUNKED_MIN_KV_TOKENS = 32768

# SM120 f16/bf16 thresholds.
SM120_SQ1_MIN_KV_UNITS = 8  # s_q == 1: b * h_kv below this (b = 1) loses 1.13-1.85 on every head dim
SM120_SQ1_MAX_GQA_GROUP = 64  # s_q == 1, d512: a 128-wide query group over one KV head loses 5-10x at every batch


def _q_tiles(facts) -> int:
    return facts.b * facts.h_q * -(-facts.s_q // 128)


def place(spec, facts) -> str:
    """``LEAD`` or ``TRAIL`` for the row ``spec`` serving ``facts`` (see the module docstring).

    Keyed by the row's name: the SM100 and SM120 f16/bf16 rows each use their
    measured shard table, and every unmeasured row (SM107, SM80, fp8,
    mxfp8) keeps the historical order -- those stay opt-in, so the order is only
    observable with the flag set, which ranks ours first anyway."""
    if spec.name == "sdpa_fwd_prefill_sm100":
        return _place_sm100_f16(spec.capabilities, facts)
    if spec.name == "sdpa_fwd_prefill_sm120":
        return _place_sm120_f16(spec.capabilities, facts)
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


def _place_sm100_f16(caps: Capabilities, facts) -> str:
    dense = not facts.thd
    if dense and 2 <= facts.s_q <= DECODE_SHAPED_MAX_S_Q:
        return LEAD
    flavor = _selected_d_shape(caps, facts)
    if dense and facts.s_q == 1:
        units = facts.b * facts.h_kv
        if flavor == (512, 512):
            return LEAD if units >= SQ1_MIN_KV_UNITS or (facts.h_q >= SQ1_MQA_MIN_Q_HEADS and facts.s_kv >= SQ1_MQA_MIN_KV_TOKENS) else TRAIL
        if flavor == (256, 256):
            if units >= SQ1_MIN_KV_UNITS:
                return LEAD
            if units >= SQ1_SMALL_BATCH_MIN_UNITS and units * facts.s_kv >= SQ1_SMALL_BATCH_MIN_KV_TOKENS:
                return LEAD
            return TRAIL
        return TRAIL  # d128 flavor (d64 rides its envelope), d192: backend decode engine ahead or at parity
    # prefill-shaped
    envelope_padded = (facts.d_qk, facts.d_v) not in caps.d_shapes
    if facts.thd or facts.has_paged_kv or facts.window_left is not None or envelope_padded:
        return TRAIL
    if flavor == (512, 512):
        return LEAD
    tiles = _q_tiles(facts)
    if tiles <= CHUNKED_SMALL_MAX_Q_TILES and facts.s_kv >= CHUNKED_SMALL_MIN_KV_TOKENS:
        return LEAD
    if tiles <= CHUNKED_MAX_Q_TILES and facts.s_kv >= CHUNKED_MIN_KV_TOKENS:
        return LEAD
    return TRAIL
