# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The heuristics TABLE of the Rubin (SM107) f16/bf16 small head-dim tiles for the pair-bias / per-key-mask graphs -- what
``recommend`` proposes first per (D, S, B, mask form) and the measured basis of every row.

The cells are the knob sweep's grid (bf16, D 32 / 64, S_q = S_kv = N = S in 128..2048 plus the ragged leg 200 / 217 / 300 / 450 /
600, H = 4, B x N flattened batch rows at B 1 / 4, the bias ``[1, H, S, S]`` batch-broadcast, the kv_mask ``[B_flat, 1, 1, S]`` bool,
a padding mask with per-batch KV lengths of 0.75 S), measured on a 212-SM cc 10.7 node at the 2376 MHz lock: CUDA-graph replay
100 x 3 with a control twin per process (replay AND the CUPTI kernel time), every ratio SAME-process, the kernel-only CUPTI time
where the replay is launch-bound (S <= 256).  Each row carries the public knob record the heuristics now propose (``TILE_N`` /
``TILE_CGA_M``), the private form axes behind it (``n_chunk`` / ``bias_mode`` / ``bias_stages``: the ``config_sm107`` picks until
the frontend-band knobs land) and the margin of the best measured arm over the pick the heuristics made BEFORE this table
(POSITIVE = faster, ``t_old / t_new - 1``); ``0`` = the old pick was the best measured arm.

Unit tier only (no GPU, no DSL): the table is a contract on the chooser, the engine row's eligibility and the config picks; the
kernels behind the picks are validated by the sm107 bias / kv_mask / streamed-ring modules.
"""

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import config_sm107 as c
from cudnn.sdpa.fwd import engines
from cudnn.sdpa.fwd import heuristics as heur
from cudnn.sdpa.fwd.engine import FrostSdpaFwdEngine
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from cudnn.sdpa.fwd.heuristics import recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = [pytest.mark.L0]

_ROW = "sdpa_fwd_prefill_sm107"
_ENGINE_ID = 20599
_OFFERED = {_ROW: _ENGINE_ID}
_SMS = 212  # the measured node class; the chunk rule reads the SM count
_H = 4


@pytest.fixture(autouse=True)
def _sm107_target_for_metadata(monkeypatch):
    """Model SM107 eligibility on other GPUs without requiring their DSL to target Rubin (the sm107 suite's fixture)."""
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


class _IrTensor:
    def __init__(self, dt):
        self._dt = dt

    def get_data_type(self):
        return self._dt


def _facts(d, s, b, form, *, padded=False):
    """The graph facts of one table cell: ``form`` in plain / mask / bias / both; ``padded`` = a padding mask with per-batch KV lengths."""
    b_flat = b * s
    bf16, boolean = cudnn.data_type.BFLOAT16, cudnn.data_type.BOOLEAN
    kw = dict(b=b_flat, h_q=_H, h_kv=_H, s_q=s, s_kv=s, d_qk=d, d_v=d, dtype=bf16, dtype_o=bf16, device_cc=(10, 7), device_sm_count=_SMS, padded=padded)
    if form in ("bias", "both"):
        kw.update(has_bias=True, bias_t=_IrTensor(bf16), bias_dim=(1, _H, s, s), bias_stride=(_H * s * s, s * s, s, 1), bias_dtype=bf16)
    if form in ("mask", "both"):
        kw.update(has_kv_mask=True, kv_mask_t=_IrTensor(boolean), kv_mask_dim=(b_flat, 1, 1, s), kv_mask_stride=(s, s, s, 1), kv_mask_dtype=boolean)
    return SdpaGraphFacts(**kw)


def _spec():
    return next(sp for sp in engines.ENGINE_SPECS if sp.name == _ROW)


# ---------------------------------------------------------------------------------------------------------------------------------
# THE TABLE.  Columns: (D, S, B, form, padded) -> (TILE_N, TILE_CGA_M, n_chunk, bias_mode, bias_stages) | best measured arm, margin %
# over the pre-table pick, the metric (CUPTI = kernel-only, launch-bound cells; replay otherwise).  n_chunk 0 = the one-batch-entry
# item; bias_mode 0 / 1 / 2 = none / resident block / streamed ring; bias_stages 0 off the ring.
# ---------------------------------------------------------------------------------------------------------------------------------
_TABLE = [
    # ---- D = 32, B = 1: the plan grid ----
    ((32, 128, 1, "plain", False), (128, 1, 0, 0, 0), "plain", 0.0, "CUPTI"),
    ((32, 128, 1, "mask", False), (64, 1, 0, 0, 0), "mask@TILE_N 64", 7.9, "CUPTI"),
    ((32, 128, 1, "bias", False), (64, 1, 4, 1, 0), "pair n4 / TILE_N 64", 26.3, "CUPTI"),
    ((32, 128, 1, "both", False), (64, 1, 4, 1, 0), "pair n4 / TILE_N 64", 30.6, "CUPTI"),
    ((32, 256, 1, "plain", False), (128, 1, 0, 0, 0), "plain", 0.0, "CUPTI"),
    ((32, 256, 1, "mask", False), (128, 1, 0, 0, 0), "mask", 0.0, "CUPTI"),
    (
        (32, 256, 1, "bias", False),
        (128, 1, 0, 2, 2),
        "one-batch-entry ring, 2 stages (+6.0 %; the pair n2 at 64 +6.0 %; the same ring on the 64-wide tile is a DIRECTION only: its +10.2 % is a CUPTI figure from a process whose CUPTI control twin spread 21.4 % under a degraded clock -- the replay there, at a 0.08 % twin, reads +3.5 % over the 128-wide ring)",
        6.0,
        "CUPTI",
    ),
    ((32, 256, 1, "both", False), (64, 1, 2, 1, 0), "pair n2 / TILE_N 64", 30.3, "CUPTI"),
    ((32, 384, 1, "plain", False), (128, 1, 0, 0, 0), "plain", 0.0, "replay"),
    ((32, 384, 1, "mask", False), (128, 1, 0, 0, 0), "mask", 0.0, "replay"),
    ((32, 384, 1, "bias", False), (128, 1, 4, 1, 0), "pair n4 (the pick; n8 a tie)", 0.0, "replay"),
    ((32, 384, 1, "both", False), (64, 1, 4, 1, 0), "pair n4 / TILE_N 64", 13.1, "replay"),
    ((32, 512, 1, "plain", False), (128, 1, 0, 0, 0), "plain", 0.0, "replay"),
    ((32, 512, 1, "mask", False), (128, 1, 0, 0, 0), "mask", 0.0, "replay"),
    ((32, 512, 1, "bias", False), (128, 1, 0, 2, 2), "one-batch-entry ring, 2 stages", 3.6, "replay"),
    ((32, 512, 1, "both", False), (64, 1, 8, 1, 0), "pair n8 resident / TILE_N 64", 17.6, "replay"),
    ((32, 768, 1, "plain", False), (128, 1, 0, 0, 0), "plain", 0.0, "replay"),
    ((32, 768, 1, "bias", False), (128, 1, 0, 2, 2), "one-batch-entry ring, 2 stages", 2.0, "replay"),
    ((32, 768, 1, "both", False), (64, 1, 8, 1, 0), "pair n8 resident / TILE_N 64", 13.4, "replay"),
    ((32, 1024, 1, "plain", False), (128, 1, 0, 0, 0), "plain", 0.0, "replay"),
    ((32, 1024, 1, "bias", False), (128, 1, 0, 2, 2), "one-batch-entry ring, 2 stages", 1.5, "replay"),
    (
        (32, 1024, 1, "both", False),
        (64, 1, 8, 2, 3),
        "pair n8 ring / TILE_N 64 (the wave-quantised n32 ring tied, the one-batch-entry 64-wide ring -1 %)",
        2.4,
        "replay",
    ),
    ((32, 2048, 1, "plain", False), (128, 1, 0, 0, 0), "plain", 0.0, "replay"),
    ((32, 2048, 1, "both", False), (64, 1, 8, 2, 3), "pair n8 ring / TILE_N 64 (n16 -1.5 %, the one-batch-entry 64-wide ring -4 %)", 1.4, "replay"),
    ((32, 2048, 1, "mask", False), (128, 1, 0, 0, 0), "mask", 0.0, "replay"),
    ((32, 2048, 1, "bias", False), (128, 1, 0, 2, 2), "one-batch-entry ring, 2 stages", 2.1, "replay"),
    ((32, 2048, 1, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths (TILE_N 64 -19 %)", 0.0, "replay"),
    # ---- D = 32, B = 4 ----
    (
        (32, 128, 4, "mask", False),
        (64, 1, 0, 0, 0),
        "mask@TILE_N 64 (+10.2 % CUPTI, whose twin spread was 18.6 % on this sub-20 us cell; the replay margin carries it)",
        8.5,
        "replay",
    ),
    ((32, 128, 4, "bias", False), (64, 1, 2, 1, 0), "pair n2 / TILE_N 64", 34.7, "CUPTI"),
    ((32, 128, 4, "both", False), (64, 1, 2, 1, 0), "pair n2 / TILE_N 64", 50.3, "CUPTI"),
    (
        (32, 256, 4, "bias", False),
        (128, 1, 0, 2, 2),
        "one-batch-entry ring, 2 stages (the best measured form; the pair n8 at 64 +2.3 %, its 64-wide ring +3.7 %)",
        9.1,
        "CUPTI",
    ),
    ((32, 256, 4, "both", False), (64, 1, 8, 1, 0), "pair n8 / TILE_N 64", 28.7, "CUPTI"),
    ((32, 384, 4, "bias", False), (128, 1, 8, 1, 0), "pair n8 (the rule's n16 halved)", 3.1, "replay"),
    ((32, 384, 4, "both", False), (64, 1, 8, 1, 0), "pair n8 / TILE_N 64", 20.4, "replay"),
    ((32, 512, 4, "bias", False), (128, 1, 0, 2, 2), "one-batch-entry ring, 2 stages", 3.2, "replay"),
    ((32, 512, 4, "both", False), (64, 1, 8, 1, 0), "pair n8 resident / TILE_N 64", 17.7, "replay"),
    ((32, 768, 4, "bias", False), (128, 1, 0, 2, 2), "one-batch-entry ring, 2 stages", 2.1, "replay"),
    ((32, 768, 4, "both", False), (64, 1, 8, 1, 0), "pair n8 resident / TILE_N 64 (n8 +3.7 % over n16, +3.2 % over n32 at 3072 rows)", 20.3, "replay"),
    ((32, 1024, 4, "bias", False), (128, 1, 0, 2, 2), "one-batch-entry ring, 2 stages", 1.6, "replay"),
    ((32, 1024, 4, "both", False), (64, 1, 8, 2, 3), "the 64-wide rings (one-batch-entry or pair)", 9.1, "replay"),
    ((32, 2048, 4, "bias", False), (128, 1, 0, 2, 2), "one-batch-entry ring, 2 stages", 2.0, "replay"),
    ((32, 2048, 4, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths (TILE_N 64 -19 %)", 0.0, "replay"),
    # ---- D = 32, the ragged leg (S not a tile multiple: the kernel's padding-mask arm with synthesized lengths) ----
    ((32, 200, 1, "plain", False), (64, 1, 0, 0, 0), "plain / TILE_N 64", 50.8, "CUPTI"),
    ((32, 217, 1, "plain", False), (64, 1, 0, 0, 0), "plain / TILE_N 64", 51.8, "CUPTI"),
    ((32, 300, 1, "plain", False), (64, 1, 0, 0, 0), "plain / TILE_N 64", 48.4, "replay"),
    ((32, 450, 1, "plain", False), (64, 1, 0, 0, 0), "plain / TILE_N 64", 16.4, "replay"),
    ((32, 600, 1, "plain", False), (64, 1, 0, 0, 0), "plain / TILE_N 64", 3.6, "replay"),
    ((32, 217, 1, "mask", False), (128, 1, 0, 0, 0), "mask (TILE_N 64 a wash / -10 % at 450)", 0.0, "CUPTI"),
    ((32, 450, 1, "mask", False), (128, 1, 0, 0, 0), "mask", 0.0, "replay"),
    ((32, 200, 1, "bias", False), (64, 1, 2, 1, 0), "pair / TILE_N 64 (n4 +2 % over n2, inside the floor)", 33.5, "CUPTI"),
    ((32, 600, 1, "bias", False), (64, 1, 8, 1, 0), "pair / TILE_N 64 (n4 measured +17.3 %, n16 +14.5 %)", 14.5, "replay"),
    ((32, 200, 1, "both", False), (64, 1, 2, 1, 0), "pair n2 / TILE_N 64", 34.7, "CUPTI"),
    ((32, 600, 1, "both", False), (64, 1, 8, 1, 0), "pair / TILE_N 64 (n16 measured +23.2 %)", 23.2, "replay"),
    # ---- D = 64, B = 1 ----
    ((64, 128, 1, "plain", False), (128, 1, 0, 0, 0), "plain, cga1 (cga2 -32 %)", 0.0, "CUPTI"),
    ((64, 128, 1, "mask", False), (64, 1, 0, 0, 0), "mask@TILE_N 64", 14.0, "CUPTI"),
    ((64, 128, 1, "bias", False), (64, 1, 4, 1, 0), "pair n4 / TILE_N 64", 24.1, "CUPTI"),
    ((64, 128, 1, "both", False), (64, 1, 4, 1, 0), "pair n4 / TILE_N 64", 38.0, "CUPTI"),
    ((64, 256, 1, "plain", False), (128, 1, 0, 0, 0), "plain, cga1 (cga2 -31 %)", 0.0, "CUPTI"),
    ((64, 256, 1, "mask", False), (128, 1, 0, 0, 0), "mask (TILE_N 64 -2.5 %)", 0.0, "CUPTI"),
    ((64, 256, 1, "bias", False), (64, 1, 2, 1, 0), "pair n2 / TILE_N 64", 23.6, "CUPTI"),
    ((64, 256, 1, "both", False), (64, 1, 2, 1, 0), "pair n2 / TILE_N 64", 44.0, "CUPTI"),
    ((64, 384, 1, "plain", False), (128, 1, 0, 0, 0), "plain, cga1 (cga2 a tie)", 0.0, "replay"),
    ((64, 384, 1, "bias", False), (128, 1, 4, 1, 0), "pair n4 resident (the pick; a 2-stage ring a tie)", 0.0, "replay"),
    ((64, 384, 1, "both", False), (64, 1, 4, 1, 0), "pair n4 / TILE_N 64", 22.0, "replay"),
    ((64, 512, 1, "plain", False), (128, 1, 0, 0, 0), "plain, cga1 (cga2 a tie)", 0.0, "replay"),
    ((64, 512, 1, "bias", False), (128, 1, 8, 2, 3), "pair n8 ring (the pick; 2 stages a tie, the cga2 one-batch-entry ring -1.3 %)", 0.0, "replay"),
    ((64, 512, 1, "both", False), (64, 1, 8, 2, 3), "pair n8 ring / TILE_N 64", 36.6, "replay"),
    ((64, 768, 1, "bias", False), (128, 1, 8, 2, 3), "pair ring n8 (n4 +3.8 %, n8 +3.0 % over the rule's n16)", 3.0, "replay"),
    ((64, 768, 1, "both", False), (64, 1, 8, 2, 3), "pair n8 ring / TILE_N 64", 37.8, "replay"),
    (
        (64, 1024, 1, "bias", False),
        (128, 1, 8, 2, 3),
        "pair ring (n16 +1.0 % over the rule's n32; the cga2 one-batch-entry ring +4.0 % is an isolated point, not a rule)",
        1.0,
        "replay",
    ),
    ((64, 1024, 1, "both", False), (64, 1, 8, 2, 3), "pair ring / TILE_N 64 (n32 measured +39 %)", 39.0, "replay"),
    ((64, 2048, 1, "plain", False), (128, 1, 0, 0, 0), "plain, cga1 (cga2 -2.7 % replay, -2.4 % CUPTI)", 0.0, "replay"),
    ((64, 2048, 1, "bias", False), (128, 1, 8, 2, 3), "pair ring n8 (+3.8 % over the rule's n128; n4 / n16 within 0.3 %)", 3.8, "replay"),
    ((64, 2048, 1, "both", False), (64, 1, 8, 2, 3), "pair n8 ring / TILE_N 64", 50.8, "replay"),
    ((64, 2048, 1, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths (TILE_N 64 -20 %)", 0.0, "replay"),
    # ---- D = 64, B = 4 ----
    ((64, 128, 4, "mask", False), (64, 1, 0, 0, 0), "mask@TILE_N 64 (the additive vector alone +31 % too)", 17.7, "CUPTI"),
    ((64, 128, 4, "bias", False), (64, 1, 2, 1, 0), "pair n2 / TILE_N 64", 22.2, "CUPTI"),
    ((64, 128, 4, "both", False), (64, 1, 2, 1, 0), "pair n2 / TILE_N 64", 32.3, "CUPTI"),
    ((64, 256, 4, "bias", False), (64, 1, 8, 1, 0), "pair n8 / TILE_N 64", 4.4, "CUPTI"),
    ((64, 256, 4, "both", False), (64, 1, 8, 1, 0), "pair n8 / TILE_N 64", 28.9, "CUPTI"),
    ((64, 384, 4, "bias", False), (128, 1, 8, 1, 0), "pair n8 (the rule's n16 halved)", 3.7, "replay"),
    ((64, 384, 4, "both", False), (64, 1, 8, 1, 0), "pair n8 / TILE_N 64", 26.0, "replay"),
    ((64, 512, 4, "bias", False), (128, 1, 8, 2, 3), "pair ring n8 (+2.1 % over the rule's n32; n16 +1.0 %)", 2.1, "replay"),
    ((64, 512, 4, "both", False), (64, 1, 8, 2, 3), "pair ring n8 / TILE_N 64 (+31.9 % over the 128-wide n32 ring; n8 +1.0 % over n32 at 64)", 31.9, "replay"),
    ((64, 1024, 4, "both", False), (64, 1, 8, 2, 3), "pair ring / TILE_N 64 (n128 measured +31.6 %)", 31.6, "replay"),
    ((64, 2048, 4, "bias", False), (128, 1, 8, 2, 3), "pair ring n8 (+4.3 % over the rule's n512, 8192 rows; lower-bound process)", 4.3, "replay"),
    ((64, 2048, 4, "both", False), (64, 1, 8, 2, 3), "pair ring / TILE_N 64 (n512 measured +39 %; lower-bound process)", 39.1, "replay"),
    # ---- the padding-mask leg: per-batch KV lengths of 0.75 S, both D, B 1 / 4, S 256 / 512 / 1024 (+ the d32 B 1 S 2048 bias cell), measured on
    # four CLEAN processes (2364 MHz) after the first cut of the table had generalised the bias-with-padding width rule from ONE cell ----
    (
        (32, 256, 1, "bias", True),
        (64, 1, 2, 1, 0),
        "bias + per-batch KV lengths 0.75 S: the table's pair n2 / resident / TILE_N 64 +52.4 %; the one-batch-entry / resident / TILE_N 64 +60.1 % -- the one-batch-entry item on the half tile runs 7.7 points ahead of the pair here (an open lever, not a guard change from one process)",
        60.1,
        "CUPTI",
    ),
    (
        (32, 256, 1, "plain", True),
        (128, 1, 0, 0, 0),
        "plain with per-batch KV lengths 0.75 S (TILE_N 64 +94.4 %: kv_len 192 ends mid-tile, the padded dense body's pathology -- the pick sees S_kv = 256, a tile multiple, and cannot follow the lengths; an open kernel item)",
        94.4,
        "CUPTI",
    ),
    (
        (32, 512, 1, "bias", True),
        (64, 1, 8, 1, 0),
        "bias + per-batch KV lengths 0.75 S: the table's pair n8 / resident / TILE_N 64 +23.5 %; the one-batch-entry / ring 3 st / TILE_N 64 +26.1 % -- the one-batch-entry item on the half tile runs 2.6 points ahead of the pair here (an open lever, not a guard change from one process)",
        26.1,
        "replay",
    ),
    ((32, 512, 1, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths 0.75 S (TILE_N 64 -11.5 %)", 0.0, "replay"),
    (
        (32, 1024, 1, "bias", True),
        (64, 1, 8, 2, 3),
        "bias + per-batch KV lengths 0.75 S: the table's pair n8 / ring 3 st / TILE_N 64 +11.3 %; the one-batch-entry / ring 3 st / TILE_N 64 +16.4 % -- the one-batch-entry item on the half tile runs 5.1 points ahead of the pair here (an open lever, not a guard change from one process)",
        16.4,
        "replay",
    ),
    ((32, 1024, 1, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths 0.75 S (TILE_N 64 -16.3 %)", 0.0, "replay"),
    (
        (32, 2048, 1, "bias", True),
        (64, 1, 8, 2, 3),
        "bias + per-batch KV lengths 0.75 S: one-batch-entry / ring 3 st / TILE_N 64 +14.3 % (the table's pair n8 ring at 64 unmeasured here)",
        14.3,
        "replay",
    ),
    ((32, 256, 4, "bias", True), (64, 1, 8, 1, 0), "bias + per-batch KV lengths 0.75 S: pair n8 / resident / TILE_N 64 +60.8 %", 60.8, "replay"),
    (
        (32, 256, 4, "plain", True),
        (128, 1, 0, 0, 0),
        "plain with per-batch KV lengths 0.75 S (TILE_N 64 +107.0 %: kv_len 192 ends mid-tile, the padded dense body's pathology -- the pick sees S_kv = 256, a tile multiple, and cannot follow the lengths; an open kernel item)",
        107.0,
        "replay",
    ),
    (
        (32, 512, 4, "bias", True),
        (64, 1, 8, 1, 0),
        "bias + per-batch KV lengths 0.75 S: the table's pair n8 / resident / TILE_N 64 +25.5 %; the one-batch-entry / ring 3 st / TILE_N 64 +29.2 % -- the one-batch-entry item on the half tile runs 3.8 points ahead of the pair here (an open lever, not a guard change from one process)",
        29.2,
        "replay",
    ),
    ((32, 512, 4, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths 0.75 S (TILE_N 64 -13.1 %)", 0.0, "replay"),
    (
        (32, 1024, 4, "bias", True),
        (64, 1, 8, 2, 3),
        "bias + per-batch KV lengths 0.75 S: the table's pair n8 / ring 3 st / TILE_N 64 +13.8 %; the one-batch-entry / ring 3 st / TILE_N 64 +16.8 % -- the one-batch-entry item on the half tile runs 3.0 points ahead of the pair here (an open lever, not a guard change from one process)",
        16.8,
        "replay",
    ),
    ((32, 1024, 4, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths 0.75 S (TILE_N 64 -16.6 %)", 0.0, "replay"),
    ((64, 256, 1, "bias", True), (64, 1, 2, 1, 0), "bias + per-batch KV lengths 0.75 S: pair n2 / resident / TILE_N 64 +50.9 %", 50.9, "replay"),
    (
        (64, 256, 1, "plain", True),
        (128, 1, 0, 0, 0),
        "plain with per-batch KV lengths 0.75 S (TILE_N 64 +66.6 %: kv_len 192 ends mid-tile, the padded dense body's pathology -- the pick sees S_kv = 256, a tile multiple, and cannot follow the lengths; an open kernel item)",
        66.6,
        "CUPTI",
    ),
    ((64, 512, 1, "bias", True), (64, 1, 8, 2, 3), "bias + per-batch KV lengths 0.75 S: pair n8 / ring 3 st / TILE_N 64 +39.3 %", 39.3, "replay"),
    ((64, 512, 1, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths 0.75 S (TILE_N 64 -12.8 %)", 0.0, "replay"),
    (
        (64, 1024, 1, "bias", True),
        (64, 1, 8, 2, 3),
        "bias + per-batch KV lengths 0.75 S: the table's pair n8 / ring 3 st / TILE_N 64 +29.8 %; the pair n2 / ring 3 st / TILE_N 64 +31.6 % -- a 2-row chunk 1.8 points ahead of the capped n8 here (one process; the cap at 8 stays)",
        31.6,
        "replay",
    ),
    ((64, 1024, 1, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths 0.75 S (TILE_N 64 -17.6 %)", 0.0, "replay"),
    ((64, 256, 4, "bias", True), (64, 1, 8, 1, 0), "bias + per-batch KV lengths 0.75 S: pair n8 / resident / TILE_N 64 +73.2 %", 73.2, "replay"),
    (
        (64, 256, 4, "plain", True),
        (128, 1, 0, 0, 0),
        "plain with per-batch KV lengths 0.75 S (TILE_N 64 +86.4 %: kv_len 192 ends mid-tile, the padded dense body's pathology -- the pick sees S_kv = 256, a tile multiple, and cannot follow the lengths; an open kernel item)",
        86.4,
        "replay",
    ),
    (
        (64, 512, 4, "bias", True),
        (64, 1, 8, 2, 3),
        "bias + per-batch KV lengths 0.75 S: pair n8 / ring 3 st / TILE_N 64 +47.9 % (the pair n2 / ring 3 st / TILE_N 64 +48.3 %, within a point)",
        47.9,
        "replay",
    ),
    ((64, 512, 4, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths 0.75 S (TILE_N 64 -14.9 %)", 0.0, "replay"),
    ((64, 1024, 4, "bias", True), (64, 1, 8, 2, 3), "bias + per-batch KV lengths 0.75 S: pair n8 / ring 3 st / TILE_N 64 +38.5 %", 38.5, "replay"),
    ((64, 1024, 4, "plain", True), (128, 1, 0, 0, 0), "plain with per-batch KV lengths 0.75 S (TILE_N 64 -17.7 %)", 0.0, "replay"),
]
_IDS = [f"d{d}_S{s}_B{b}_{form}{'_padded' if padded else ''}" for (d, s, b, form, padded), *_ in _TABLE]


def _primary(facts):
    plans = recommend("A", facts, _OFFERED)
    assert plans, "the row must be eligible for every table cell"
    return plans[0].knobs, plans


@pytest.mark.parametrize("cell, want, best, margin, metric", _TABLE, ids=_IDS)
def test_table_cell_is_the_heuristics_primary_plan(cell, want, best, margin, metric):
    """The first plan ``recommend`` proposes carries the table's public knobs (TILE_N / TILE_CGA_M, unsplit, unpacked), it is
    admissible on the row, and the private form axes behind it are the config picks at the plan's width and cga."""
    d, s, b, form, padded = cell
    tile_n, cga, n_chunk, bias_mode, bias_stages = want
    facts = _facts(d, s, b, form, padded=padded)
    knobs, plans = _primary(facts)
    assert (knobs.tile_n, knobs.cga) == (tile_n, cga), (cell, knobs, best, margin, metric)
    assert (knobs.split_kv or 1) == 1 and not knobs.pack_gqa and knobs.tile_m == 128, knobs
    assert engines.mismatch(_spec().capabilities, facts, knobs) is None, (cell, knobs)
    # every proposed set is admissible and names a different knob assignment
    assert len({(p.knobs.tile_n, p.knobs.cga, p.knobs.sched_policy, p.knobs.split_kv, p.knobs.pack_gqa) for p in plans}) == len(plans)
    # the private axes: the config picks the engine row, the adapter and the factory read (rule 8b')
    has_bias = form in ("bias", "both")
    got_n = c.sm107_n_chunk_pick(1, b * s, _H, s, _SMS, cga, s_kv=s, d_flavor=d, tile_n=tile_n) if has_bias else 0
    pair = has_bias and got_n > 0
    got_mode = c.sm107_bias_mode_pick(d, cga, s, pair=pair) if has_bias else 0
    got_stages = c.sm107_bias_stages_pick(d, cga, pair=pair) if got_mode == c.BIAS_MODE_STREAMED else 0
    assert (got_n, got_mode, got_stages) == (n_chunk, bias_mode, bias_stages), (cell, (got_n, got_mode, got_stages), best)


@pytest.mark.parametrize("cell, want, best, margin, metric", _TABLE, ids=_IDS)
def test_table_record_round_trips_through_the_public_knob_vocabulary(cell, want, best, margin, metric):
    """The proposed knobs convert to the ONE public vocabulary (``cudnn.knob_type`` -> int) and back, through the engine's
    ``knobs_to_public`` / ``knobs_from_public`` and the dataclass pair; a persisted record (integer keys) replays to the same knobs."""
    d, s, b, form, padded = cell
    knobs, _ = _primary(_facts(d, s, b, form, padded=padded))
    engine = FrostSdpaFwdEngine(_spec(), _ENGINE_ID)
    public = engine.knobs_to_public(knobs)
    assert public and all(isinstance(k, cudnn.knob_type) and isinstance(v, int) and not isinstance(v, bool) for k, v in public.items()), public
    assert public[cudnn.knob_type.TILE_N] == want[0] and public[cudnn.knob_type.TILE_CGA_M] == want[1], public
    assert engine.knobs_from_public(public) == knobs
    assert SdpaFwdKnobs.from_public(knobs.to_public()) == knobs
    assert engine.knobs_from_public({int(k): v for k, v in public.items()}) == knobs, "a persisted record carries the integers"


def test_width_pick_is_one_arithmetic_for_the_heuristics_and_the_row():
    """``heuristics._resident_arm_tile_n`` IS ``config_sm107.sm107_tile_n_pick`` on the small tiles (the engine row evaluates the
    pair / one-batch-entry form at the same width), and the pick keeps every measured dead end: the half tile for plain / kv_mask-only
    graphs at a tile multiple (-3..-20 %), for a bias alone at S % 128 == 0 >= 384 (-3..-13 %), for the kv_mask-only ragged body
    above one tile, for the d64 padded plain body above two tiles (-4 % at 600) and for per-batch lengths at a tile multiple."""
    spec = _spec()
    for d, s, b, form, padded in [cell for cell, *_ in _TABLE]:
        facts = _facts(d, s, b, form, padded=padded)
        assert heur._resident_arm_tile_n(spec, facts, 128) == c.sm107_tile_n_pick(
            d, s, has_bias=facts.has_bias, has_kv_mask=facts.has_kv_mask, padded=padded, domain=frozenset({64, 128})
        ), (d, s, b, form)
    T = c.sm107_tile_n_pick
    assert T(32, 512, has_bias=False, has_kv_mask=False) == 128 and T(64, 1024, has_bias=False, has_kv_mask=True) == 128
    assert T(32, 384, has_bias=True, has_kv_mask=False) == 128 and T(64, 1024, has_bias=True, has_kv_mask=False) == 128
    assert T(32, 450, has_bias=False, has_kv_mask=True) == 128 and T(64, 600, has_bias=False, has_kv_mask=False) == 128
    assert T(32, 2048, has_bias=False, has_kv_mask=False, padded=True) == 128
    # the narrow tile where it was measured faster
    assert T(32, 2048, has_bias=True, has_kv_mask=True) == 64 and T(64, 2048, has_bias=True, has_kv_mask=True) == 64
    assert T(32, 2048, has_bias=True, has_kv_mask=False, padded=True) == 64, "a bias with the padding mask: the combined body"
    assert (
        T(32, 128, has_bias=True, has_kv_mask=False) == 64
        and T(64, 256, has_bias=True, has_kv_mask=False) == 64
        and T(64, 128, has_bias=False, has_kv_mask=True) == 64
    )
    assert (
        T(32, 256, has_bias=True, has_kv_mask=False) == 128
    ), "two kv tiles on the 32-wide tile: the one-batch-entry streamed ring at 128 is the measured best"
    assert T(32, 600, has_bias=False, has_kv_mask=False) == 64 and T(64, 200, has_bias=False, has_kv_mask=False) == 64
    # a request is returned as made (honoured or declined downstream, never degraded); a flavor without the narrow width keeps 128
    assert T(32, 512, has_bias=True, has_kv_mask=True, request=128) == 128 and T(128, 512, has_bias=True, has_kv_mask=True) == 128
    assert T(32, 512, has_bias=False, has_kv_mask=False, domain=frozenset({128})) == 128


def test_fp32_vector_with_a_bias_keeps_its_only_served_width():
    """The fp32 ADDITIVE kv_mask vector WITH a bias ran TILE_N 64 before the table (the 128-wide rendering spills): the table agrees
    (a bias with any mask takes the narrow tile), so the row's typed decline of an explicit 128 and the pick stay consistent."""
    spec = _spec()
    f32 = cudnn.data_type.FLOAT
    facts = SdpaGraphFacts(
        b=384,
        h_q=_H,
        h_kv=_H,
        s_q=384,
        s_kv=384,
        d_qk=32,
        d_v=32,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(10, 7),
        device_sm_count=_SMS,
        has_bias=True,
        bias_t=_IrTensor(cudnn.data_type.BFLOAT16),
        bias_dim=(1, _H, 384, 384),
        bias_stride=(_H * 384 * 384, 384 * 384, 384, 1),
        bias_dtype=cudnn.data_type.BFLOAT16,
        has_kv_mask=True,
        kv_mask_t=_IrTensor(f32),
        kv_mask_dim=(384, 1, 1, 384),
        kv_mask_stride=(384, 384, 384, 1),
        kv_mask_dtype=f32,
    )
    knobs, _ = _primary(facts)
    assert knobs.tile_n == c.SM107_FP32_VECTOR_WITH_BIAS_TILE_N == 64
    assert engines.mismatch(spec.capabilities, facts, SdpaFwdKnobs(tile_n=128)) is not None, "an explicit 128 stays a typed decline"


def test_n_chunk_corrections_carry_the_measured_cells():
    """The pair chunk rule's measured corrections: the cap at 8 (n16 -> n8 +3.1 / +3.7 % at 1536 rows, n32 -> n8 +2.9 % at 2048 rows,
    the pair ring's n16 / n32 -> n4 / n8 +2.4..+5.5 % at d64 S_kv 640..896, n128 -> n8 +3.8 % at d64 S 2048); the floor of 4 at one
    128-wide kv tile while the 2-row items stay under 2 waves (+8..+16 % at 128 rows; n2 wins back by +8 % at 512 rows on the 64-wide
    tile); the one-batch-entry guard at TILE_N 128 only (the pair's resident block wins on the 64-wide tile: +2.8 % at S 256, +12 /
    +13 % at S 512 / 768 with a kv_mask)."""
    P = c.sm107_n_chunk_pick
    assert P(1, 128, _H, 128, _SMS, 1, s_kv=128, d_flavor=32, tile_n=64) == 4 and P(1, 128, _H, 128, _SMS, 1, s_kv=128, d_flavor=64, tile_n=64) == 4
    assert P(1, 512, _H, 128, _SMS, 1, s_kv=128, d_flavor=32, tile_n=64) == 2, "512 rows: 4.8 waves of 2-row items -- no floor"
    assert P(1, 256, _H, 256, _SMS, 1, s_kv=256, d_flavor=32) == 0, "the guard: a full super-tile inside the one-batch-entry budget at TILE_N 128"
    assert P(1, 256, _H, 256, _SMS, 1, s_kv=256, d_flavor=32, tile_n=64) == 2, "...but the pair on the 64-wide tile"
    assert P(1, 512, _H, 512, _SMS, 1, s_kv=512, d_flavor=32) == 0 and P(1, 512, _H, 512, _SMS, 1, s_kv=512, d_flavor=32, tile_n=64) == 8
    assert P(1, 1536, _H, 384, _SMS, 1, s_kv=384, d_flavor=32) == 8 and P(1, 1536, _H, 384, _SMS, 1, s_kv=384, d_flavor=64) == 8
    assert P(1, 2048, _H, 512, _SMS, 1, s_kv=512, d_flavor=32, tile_n=64) == 8
    for s in (640, 768, 896, 1024, 2048):
        assert P(1, s, _H, s, _SMS, 1, s_kv=s, d_flavor=64) == 8, s
    assert P(1, 384, _H, 384, _SMS, 1, s_kv=384, d_flavor=32) == 4 and P(1, 512, _H, 512, _SMS, 1, s_kv=512, d_flavor=64) == 8, "the rule's own picks stay"
    assert c.sm107_n_chunk_refine(64, 3072, _H, 768, _SMS, s_kv=768) == c.SM107_PAIR_N_CHUNK_MAX == 8
    assert c.sm107_n_chunk_refine(2, 2, _H, 128, _SMS, s_kv=128) == 2, "the clamp to b_flat holds through the floor"
    assert c.sm107_n_chunk_refine(2, 6, _H, 128, _SMS, s_kv=128) == 4 and c.sm107_n_chunk_refine(3, 400, _H, 400, _SMS, s_kv=400) == 2, "even, >= 2"


def test_one_batch_entry_item_streams_at_two_or_more_kv_tiles_on_the_32_wide_tile():
    """The one-batch-entry item's bias form at the 32-wide tile (cga1): the streamed ring wherever it fits and the range is two kv tiles or
    more (+6.4 / +10.0 % over its resident block at S 256 with 256 / 1024 rows, +9.9 / +12.9 % at S 384), the block at one tile; a RESIDENT
    request still gets the block; the pair item and the 64-wide tile keep "resident where it fits"."""
    assert c.SM107_ONE_BATCH_ENTRY_STREAMING_D_FLAVORS == frozenset({32})
    assert c.sm107_bias_mode_pick(32, 1, 256, pair=False) == c.BIAS_MODE_STREAMED and c.sm107_bias_mode_pick(32, 1, 384, pair=False) == c.BIAS_MODE_STREAMED
    assert c.sm107_bias_mode_pick(32, 1, 128, pair=False) == c.BIAS_MODE_RESIDENT, "one kv tile: nothing for a ring to overlap"
    assert c.sm107_bias_mode_pick(32, 1, 256, pair=False, request=1) == c.BIAS_MODE_RESIDENT, "a request is honoured where its form fits"
    assert c.sm107_bias_mode_pick(32, 1, 384, pair=True) == c.BIAS_MODE_RESIDENT and c.sm107_bias_mode_pick(64, 1, 128, pair=False) == c.BIAS_MODE_RESIDENT
    assert c.sm107_bias_mode_pick(64, 2, 256, pair=False) == c.BIAS_MODE_RESIDENT, "the 64-wide tile's cga2 pick is unmeasured and unchanged"
    assert c.sm107_bias_stages_pick(32, 1, pair=False) == 2, "...and the ring it takes runs 2 stages"


def test_one_batch_entry_ring_runs_two_stages_and_the_pair_ring_the_deepest():
    """The one-batch-entry item's streamed ring takes 2 stages (+1.5..+3.6 % over the derived 3 on 6 of 6 d32 processes, +2.1 % at
    S 2048), the pair ring the deepest that fits (2 stages a tie or a loss there); a request still wins where it fits."""
    assert c.SM107_ONE_BATCH_ENTRY_RING_STAGES == 2
    assert c.sm107_bias_stages_pick(32, 1, pair=False) == 2 and c.sm107_bias_stages_pick(32, 1, pair=True) == 3
    assert c.sm107_bias_stages_pick(64, 1, pair=True) == 3 and c.sm107_bias_stages_pick(64, 2, pair=False) == 2
    assert c.sm107_bias_stages_pick(32, 1, pair=False, request=3) == 3 and c.sm107_bias_stages_pick(32, 1, pair=False, request=4) == 0


def test_cluster_axis_keeps_cga1_on_the_small_tiles():
    """Cluster 1 vs 2 at small S (the wave-quantisation question): cga2 is -32 / -31 % at d64 S 128 / 256 (plain), a tie at 384 / 512
    and -2.7 % at 2048 (replay; -2.4 % CUPTI); a bias at cga2 is the one-batch-entry ring, -22 % at 384.  The d64 default stays cga1; a cga2 REQUEST at
    D <= 32 rides the 64-wide tile (the only validated cga2 width)."""
    for s in (128, 256, 384, 512, 2048):
        for form in ("plain", "mask", "bias", "both"):
            knobs, _ = _primary(_facts(64, s, 1, form))
            assert knobs.cga == 1, (s, form, knobs)
    assert c.sm107_f16_tile_width(32, 32, cga=2) == 64 and c.sm107_f16_tile_width(32, 32) == 32
